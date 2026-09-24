"""End-to-end FSM tests: real phases, layers, workbook, and sandbox; a scripted model.

See ``specs/phases/agent.md``. The workbook is ``samples/QA_sample.xlsx`` and the
question is the one the exploration run was made for.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from autotab.config import load_config
from autotab.qa import agent
from autotab.qa.schemas import Phase, RunStatus

WORKBOOK = Path("samples/QA_sample.xlsx")
QUERY = "Full name of the man who have the highest score"

PLAN = {
    "known_facts": [],
    "uncertainties": ["No gender column has been seen yet."],
    "execution_objective": "Read People!A1:G2 to learn the two-row header layout.",
    "rationale": "Names and scores sit under merged headers.",
}
READ_HEADERS = {
    "action_type": "tool",
    "rationale": "Read the header rows.",
    "tool_name": "inspect_range",
    "arguments": {"workbook_id": "workbook_1", "sheet_name": "People", "range_ref": "A1:G2"},
}
RANK = {
    "action_type": "code",
    "rationale": "Find the top score and its row.",
    "code": (
        "df = wb.sheet('People', header_row=None)\n"
        "data = df.iloc[2:]\n"
        "scores = [int(v) for v in data['G']]\n"
        "top = max(scores)\n"
        "row = scores.index(top) + 3\n"
        "cid = record_computation('max', ['People!G3:G22'], top)\n"
        "result = {'top': top, 'row': row, 'computation': cid}"
    ),
}


def _answer(
    text: str = "The highest score, 99, belongs to Nga Anh Ngo (row 13). The "
    "workbook records no gender, so 'the man' cannot be verified.",
) -> dict[str, Any]:
    return {
        "action_type": "answer",
        "rationale": "Every part is grounded.",
        "answer": {
            "answer_text": text,
            "claims": [
                {
                    "id": "claim_1",
                    "statement": "The highest score is 99.",
                    "value": 99,
                    "citations": ["People!G3:G22"],
                    "computation_id": "calc_1",
                },
                {
                    "id": "claim_2",
                    "statement": "Row 13 holds Nga Anh Ngo.",
                    "value": "Nga Anh Ngo",
                    "citations": ["People!B13:D13"],
                    "computation_id": None,
                },
            ],
        },
    }


def _judge(status: str = "PASS") -> dict[str, Any]:
    verdict: dict[str, Any] = {
        "status": status,
        "confidence_score": 0.8,
        "reason": "The top score and its row are read exactly.",
        "issues_found": [],
        "improvement_feedback": [],
        "final_assessment": "Grounded.",
    }
    if status == "FAIL":
        verdict.update(
            reason="The answer calls the person a man, which the workbook cannot show.",
            issues_found=["Gender is asserted without evidence."],
            improvement_feedback=[
                {
                    "code": "ANSWER_QUALITY.UNDISCLOSED_ASSUMPTION",
                    "source": "llm_judge",
                    "problem": "No column records gender.",
                    "evidence": ["People!A1:G2"],
                    "action": "State that gender cannot be verified from the workbook.",
                    "priority": "blocking",
                    "recheck": "LLM_JUDGE",
                }
            ],
        )
    return verdict


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    loaded = load_config("config.example.yaml")
    loaded["runtime"]["artifact_root"] = str(tmp_path)
    loaded["qa"]["exploration_enabled"] = False
    loaded["qa"]["max_turns"] = 6
    return loaded


def _phases(client: Any) -> list[str]:
    phases = []
    for prompt in client.prompts:
        for name in ("OBSERVE", "EXECUTE", "VERIFICATION"):
            if f"Phase: {name}" in prompt:
                phases.append(name)
    return phases


def test_the_happy_path_answers_with_one_model_call_per_turn(
    config: dict[str, Any], scripted: Any
) -> None:
    client = scripted(PLAN, READ_HEADERS, RANK, _answer(), _judge())

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.PASS, outcome.reason
    assert outcome.turns == 3
    # OBSERVE once on entry, then the data-gathering turns chain without re-planning.
    assert _phases(client) == ["OBSERVE", "EXECUTE", "EXECUTE", "EXECUTE", "VERIFICATION"]
    answer = (outcome.run_dir / "final" / "answer.md").read_text(encoding="utf-8")
    assert "Nga Anh Ngo" in answer


def test_the_code_turn_sees_what_the_tool_turn_read(config: dict[str, Any], scripted: Any) -> None:
    client = scripted(PLAN, READ_HEADERS, RANK, _answer(), _judge())

    agent.run(QUERY, [WORKBOOK], config, client=client)

    third_prompt = client.prompts[2]
    assert "People!A1:G2 (row | A | B | C | D | E | F | G)" in third_prompt
    assert "2 | No | First | Middle | Last | Birth | Admission | Score" in third_prompt
    assert "merged: A1:A2, B1:D1, E1:F1, G1:G2" in third_prompt


def test_a_failed_verdict_returns_to_observe_with_the_blocking_feedback(
    config: dict[str, Any], scripted: Any
) -> None:
    client = scripted(
        PLAN,
        READ_HEADERS,
        RANK,
        _answer("Nga Anh Ngo is the man with 99."),
        _judge("FAIL"),
        PLAN,
        _answer(),
        _judge("PASS"),
    )

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.PASS, outcome.reason
    assert _phases(client) == [
        "OBSERVE",
        "EXECUTE",
        "EXECUTE",
        "EXECUTE",
        "VERIFICATION",
        "OBSERVE",
        "EXECUTE",
        "VERIFICATION",
    ]
    replan = client.prompts[5]
    assert "FAILED verification" in replan
    assert "ANSWER_QUALITY.UNDISCLOSED_ASSUMPTION" in replan
    assert "State that gender cannot be verified from the workbook." in replan


def test_running_out_of_turns_fails_without_an_answer(
    config: dict[str, Any], scripted: Any
) -> None:
    config["qa"]["max_turns"] = 2
    read_scores = {
        **READ_HEADERS,
        "arguments": {**READ_HEADERS["arguments"], "range_ref": "G3:G22"},
    }
    client = scripted(PLAN, READ_HEADERS, read_scores)

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.FAIL
    assert outcome.turns == 2
    assert "turn" in outcome.reason.lower()
    assert not (outcome.run_dir / "final" / "answer.md").exists()


def test_an_answer_that_cites_a_missing_sheet_cannot_pass(
    config: dict[str, Any], scripted: Any
) -> None:
    config["qa"]["max_turns"] = 3
    bad = _answer()
    bad["answer"]["claims"][1]["citations"] = ["Peple!B13:D13"]
    # Reworded but still citing the missing sheet: an unchanged resubmission would be
    # refused as a repeat before it ever reached verification.
    reworded = _answer("Nga Anh Ngo scored 99; gender is not recorded in the workbook.")
    reworded["answer"]["claims"][1]["citations"] = ["Peple!B13:D13"]
    client = scripted(PLAN, RANK, bad, _judge("PASS"), PLAN, reworded, _judge("PASS"))

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.FAIL
    assert "ANSWER.CITATION_EXISTS" in outcome.reason


def test_the_run_leaves_a_redacted_manifest_and_an_ordered_history(
    config: dict[str, Any], scripted: Any
) -> None:
    config["models"]["llm"]["api_key"] = "sk-or-v1-never-in-artifacts"
    client = scripted(PLAN, READ_HEADERS, RANK, _answer(), _judge())

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    manifest_text = (outcome.run_dir / "manifest.json").read_text(encoding="utf-8")
    assert "sk-or-v1-never-in-artifacts" not in manifest_text
    manifest = json.loads(manifest_text)
    digest = hashlib.sha256(WORKBOOK.read_bytes()).hexdigest()
    assert manifest["workbooks"][0] == {
        "workbook_id": "workbook_1",
        "path": str(WORKBOOK),
        "sha256": digest,
    }
    phases = [
        json.loads(line)["phase"]
        for line in (outcome.run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert phases == [
        Phase.INITIALIZE.value,
        Phase.OBSERVE.value,
        Phase.EXECUTE.value,
        Phase.EXECUTE.value,
        Phase.EXECUTE.value,
        Phase.VERIFY.value,
        Phase.FINALIZE.value,
    ]
    assert (outcome.run_dir / "code" / "exec_2.py").read_text(encoding="utf-8") == RANK["code"]
    assert (outcome.run_dir / "computations" / "calc_1.json").exists()


def test_exploration_leads_reach_the_first_prompt(
    config: dict[str, Any], scripted: Any, tmp_path: Path
) -> None:
    exploration = tmp_path / "exploration"
    exploration.mkdir()
    (exploration / "evidence.filtered.json").write_text(
        json.dumps(
            [
                {
                    "keyword": "highest score",
                    "source_range": "People!G1",
                    "description": "Score is the column header in G1.",
                    "status": "keep",
                    "reason": "",
                }
            ]
        ),
        encoding="utf-8",
    )
    config["qa"]["exploration_enabled"] = True
    config["qa"]["exploration_path"] = str(exploration)
    client = scripted(PLAN, RANK, _answer(), _judge())

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.PASS, outcome.reason
    assert "Score is the column header in G1." in client.prompts[0]


def test_a_missing_exploration_artifact_fails_initialization(
    config: dict[str, Any], scripted: Any, tmp_path: Path
) -> None:
    config["qa"]["exploration_enabled"] = True
    config["qa"]["exploration_path"] = str(tmp_path / "nowhere")
    client = scripted()

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.FAIL
    assert "exploration" in outcome.reason.lower()
    assert client.prompts == []


def test_a_missing_workbook_fails_initialization(config: dict[str, Any], scripted: Any) -> None:
    outcome = agent.run(QUERY, [Path("samples/nope.xlsx")], config, client=scripted())

    assert outcome.status is RunStatus.FAIL
    assert "nope.xlsx" in outcome.reason


def test_the_recorded_maximum_holds_the_real_column(config: dict[str, Any], scripted: Any) -> None:
    client = scripted(PLAN, RANK, _answer(), _judge())

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    record = json.loads(
        (outcome.run_dir / "computations" / "calc_1.json").read_text(encoding="utf-8")
    )
    assert [item["reference"] for item in record["inputs"]] == [
        f"People!G{row}" for row in range(3, 23)
    ]
    assert max(item["value"] for item in record["inputs"]) == 99


def test_every_model_call_is_measured_and_the_run_is_totalled(
    config: dict[str, Any], scripted: Any
) -> None:
    client = scripted(PLAN, READ_HEADERS, RANK, _answer(), _judge())

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    events = [
        json.loads(line)
        for line in (outcome.run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    model_events = [e for e in events if e["phase"] in ("OBSERVE", "EXECUTE", "VERIFY")]
    assert len(model_events) == 5
    assert all(e["usage"]["calls"] == 1 and e["latency_ms"] == 5.0 for e in model_events)
    manifest = json.loads((outcome.run_dir / "manifest.json").read_text(encoding="utf-8"))
    totals = manifest["outcome"]["usage"]
    assert totals["calls"] == 5
    assert totals["prompt_tokens"] == sum(len(prompt) // 4 for prompt in client.prompts)
    assert totals["latency_ms"] == 25.0


def test_exploration_run_by_qa_shares_the_run_folder(
    config: dict[str, Any], scripted: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autotab.exploration.pipeline import ExplorationPipeline

    seen: dict[str, Any] = {}

    def fake_explore(self: Any, workbooks: list[str], query: str, run_id: str | None = None):
        seen["run_id"] = run_id
        folder = Path(self.config["runtime"]["artifact_root"]) / str(run_id) / "exploration"
        folder.mkdir(parents=True)
        (folder / "evidence.filtered.json").write_text(
            json.dumps(
                [
                    {
                        "keyword": "score",
                        "source_range": "People!G1",
                        "description": "Score header.",
                        "status": "keep",
                        "reason": "",
                    }
                ]
            ),
            encoding="utf-8",
        )
        return folder / "exploration.md"

    monkeypatch.setattr(ExplorationPipeline, "run", fake_explore)
    config["qa"]["exploration_enabled"] = True
    config["qa"]["exploration_path"] = None
    client = scripted(PLAN, RANK, _answer(), _judge())

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    run_folder = outcome.run_dir.parent
    assert outcome.run_dir.name == "qa"
    assert seen["run_id"] == run_folder.name
    assert (run_folder / "exploration" / "evidence.filtered.json").exists()
    assert "Score header." in client.prompts[0]


def test_qa_and_exploration_use_one_run_id_format(config: dict[str, Any], scripted: Any) -> None:
    import re

    outcome = agent.run(QUERY, [WORKBOOK], config, client=scripted(PLAN, RANK, _answer(), _judge()))

    assert re.fullmatch(r"\d{8}T\d{12}-\d+", outcome.run_dir.parent.name)


def test_resubmitting_a_failed_answer_unchanged_is_refused(
    config: dict[str, Any], scripted: Any
) -> None:
    client = scripted(
        PLAN,
        RANK,
        _answer("Nga Anh Ngo is the man with 99."),
        _judge("FAIL"),
        PLAN,
        _answer("Nga Anh Ngo is the man with 99."),
        _answer(),
        _judge("PASS"),
    )

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.PASS, outcome.reason
    # The unchanged resubmission was re-asked inside the phase, never re-judged.
    assert _phases(client).count("VERIFICATION") == 2
    assert any("repeats the previous action" in prompt for prompt in client.prompts)


def test_every_observe_and_execute_prompt_is_kept_verbatim(
    config: dict[str, Any], scripted: Any
) -> None:
    client = scripted(PLAN, READ_HEADERS, RANK, _answer(), _judge())

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    prompts = outcome.run_dir / "prompts"
    assert sorted(path.name for path in prompts.iterdir()) == [
        "0.observe.txt",
        "1.execute.txt",
        "2.execute.txt",
        "3.execute.txt",
    ]
    kept = [
        (prompts / name).read_text(encoding="utf-8")
        for name in ("0.observe.txt", "1.execute.txt", "2.execute.txt", "3.execute.txt")
    ]
    assert kept == client.prompts[:4]
    events = [
        json.loads(line)
        for line in (outcome.run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    execute = next(e for e in events if e["phase"] == "EXECUTE")
    assert "prompts/1.execute.txt" in execute["artifact_paths"]
    observe = next(e for e in events if e["phase"] == "OBSERVE")
    assert observe["artifact_paths"] == ["prompts/0.observe.txt"]
