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


def _plan(next_action: str, objective: str) -> dict[str, Any]:
    return {
        "known_facts": [],
        "uncertainties": ["No gender column has been seen yet."],
        "execution_objective": objective,
        "next_action": next_action,
        "rationale": "Names and scores sit under merged headers.",
    }


PLAN_READ = _plan("tool", "Read People!A1:G2 to learn the two-row header layout.")
PLAN_RANK = _plan("code", "Compute the top score over People!G3:G22 and find its row.")
PLAN_ANSWER = _plan("answer", "Answer with the top scorer's full name.")
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


STEP_OK = {
    "status": "PASS",
    "confidence_score": 0.9,
    "reason": "The step did what its objective asked.",
    "issues_found": [],
    "improvement_feedback": [],
    "final_assessment": "Usable.",
}
STEP_FAIL = {
    "status": "FAIL",
    "confidence_score": 0.7,
    "reason": "Only the header rows were read; the scores are still unseen.",
    "issues_found": ["No score was read."],
    "improvement_feedback": [
        {
            "code": "DATA_HANDLING.SCOPE_ERROR",
            "source": "llm_judge",
            "problem": "The step read headers only.",
            "evidence": ["People!A1:G2"],
            "action": "Load People!G3:G22 in code and compute the maximum.",
            "priority": "blocking",
            "recheck": "LLM_JUDGE",
        }
    ],
    "final_assessment": "Not enough yet.",
}


def _happy() -> tuple[Any, ...]:
    # A read is checked but not judged, so its turn makes no VERIFY call.
    return (
        PLAN_READ, READ_HEADERS,
        PLAN_RANK, RANK, STEP_OK,
        PLAN_ANSWER, _answer(), _judge(),
    )  # fmt: skip


def _rank_then_answer() -> tuple[Any, ...]:
    return (PLAN_RANK, RANK, STEP_OK, PLAN_ANSWER, _answer(), _judge())


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


def _answer_verifications(client: Any) -> int:
    return sum("Phase: VERIFICATION. " in prompt for prompt in client.prompts)


def _events(outcome: Any) -> list[dict[str, Any]]:
    history = (outcome.run_dir / "history.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in history.splitlines()]


def test_every_turn_observes_executes_and_verifies(config: dict[str, Any], scripted: Any) -> None:
    client = scripted(*_happy())

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.PASS, outcome.reason
    assert outcome.turns == 3
    assert _phases(client) == [
        "OBSERVE", "EXECUTE",
        "OBSERVE", "EXECUTE", "VERIFICATION",
        "OBSERVE", "EXECUTE", "VERIFICATION",
    ]  # fmt: skip
    answer = (outcome.run_dir / "final" / "answer.md").read_text(encoding="utf-8")
    assert "Nga Anh Ngo" in answer


def test_the_next_plan_sees_what_the_last_step_read(config: dict[str, Any], scripted: Any) -> None:
    client = scripted(*_happy())

    agent.run(QUERY, [WORKBOOK], config, client=client)

    for prompt in (client.prompts[2], client.prompts[3]):
        assert "People!A1:G2 (row | A | B | C | D | E | F | G)" in prompt
        assert "2 | No | First | Middle | Last | Birth | Admission | Score" in prompt
    assert "merged: A1:A2, B1:D1, E1:F1, G1:G2" in client.prompts[2]
    # Once a newer step exists, the read is a one-line digest.
    later_plan = client.prompts[5]
    digest = next(line for line in later_plan.splitlines() if line.startswith("Turn 1 (exec_1)"))
    assert "tool inspect_range" in digest and digest.endswith("…")
    assert "merged: A1:A2, B1:D1, E1:F1, G1:G2" not in later_plan


def test_a_failed_step_returns_to_observe_with_its_feedback(
    config: dict[str, Any], scripted: Any
) -> None:
    rank_again = {**RANK, "code": RANK["code"] + "\nprint(top)"}
    client = scripted(
        PLAN_RANK, RANK, STEP_FAIL,
        PLAN_RANK, rank_again, STEP_OK,
        PLAN_ANSWER, _answer(), _judge(),
    )  # fmt: skip

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.PASS, outcome.reason
    replan = client.prompts[3]
    assert "FAILED verification" in replan
    assert "DATA_HANDLING.SCOPE_ERROR" in replan
    assert "Load People!G3:G22 in code and compute the maximum." in replan
    # The next step passed, so its judgement replaces the failed one.
    assert "DATA_HANDLING.SCOPE_ERROR" not in client.prompts[6]


def test_a_failed_answer_keeps_its_feedback_until_an_answer_passes(
    config: dict[str, Any], scripted: Any
) -> None:
    client = scripted(
        PLAN_RANK, RANK, STEP_OK,
        PLAN_ANSWER, _answer("Nga Anh Ngo is the man with 99."), _judge("FAIL"),
        PLAN_RANK, RANK, STEP_OK,
        PLAN_ANSWER, _answer(), _judge("PASS"),
    )  # fmt: skip

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.PASS, outcome.reason
    after_fail = client.prompts[6]
    assert "FAILED verification" in after_fail
    assert "ANSWER_QUALITY.UNDISCLOSED_ASSUMPTION" in after_fail
    assert "State that gender cannot be verified from the workbook." in after_fail
    # A step that passed in between does not resolve what the answer judge found.
    assert "ANSWER_QUALITY.UNDISCLOSED_ASSUMPTION" in client.prompts[9]


def test_running_out_of_turns_fails_without_an_answer(
    config: dict[str, Any], scripted: Any
) -> None:
    config["qa"]["max_turns"] = 2
    read_scores = {
        **READ_HEADERS,
        "arguments": {**READ_HEADERS["arguments"], "range_ref": "G3:G22"},
    }
    client = scripted(PLAN_READ, READ_HEADERS, PLAN_READ, read_scores)

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.FAIL
    assert outcome.turns == 2
    assert "turn" in outcome.reason.lower()
    assert not (outcome.run_dir / "final" / "answer.md").exists()


def test_an_execute_without_a_valid_action_goes_back_to_the_plan(
    config: dict[str, Any], scripted: Any
) -> None:
    # Retrying the same plan let a run burn eleven turns while the executor kept
    # trying to answer a plan that asked for code; the planner has to hear about it.
    config["qa"]["max_turns"] = 2
    client = scripted(PLAN_READ, "junk", "more junk", PLAN_READ, READ_HEADERS)

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.FAIL
    assert outcome.turns == 2
    assert _phases(client) == ["OBSERVE", "EXECUTE", "EXECUTE", "OBSERVE", "EXECUTE"]
    replan = client.prompts[3]
    assert "The last EXECUTE produced no valid action" in replan
    assert "more junk" in replan
    assert (outcome.run_dir / "turns" / "02_t1_execute_invalid" / "reply.txt").exists()


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
    client = scripted(
        PLAN_RANK, RANK, STEP_OK,
        PLAN_ANSWER, bad, _judge("PASS"),
        PLAN_ANSWER, reworded, _judge("PASS"),
    )  # fmt: skip

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.FAIL
    assert "ANSWER.CITATION_EXISTS" in outcome.reason


def test_the_run_leaves_a_redacted_manifest_and_an_ordered_history(
    config: dict[str, Any], scripted: Any
) -> None:
    config["models"]["llm"]["api_key"] = "sk-or-v1-never-in-artifacts"
    client = scripted(*_happy())

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
    turn = [Phase.OBSERVE.value, Phase.EXECUTE.value, Phase.VERIFY.value]
    assert [e["phase"] for e in _events(outcome)] == [
        Phase.INITIALIZE.value,
        *turn * 3,
        Phase.FINALIZE.value,
    ]
    code = outcome.run_dir / "turns" / "05_t2_execute_code" / "code.py"
    assert code.read_text(encoding="utf-8") == RANK["code"]
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
    client = scripted(*_rank_then_answer())

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
    client = scripted(*_rank_then_answer())

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
    client = scripted(*_happy())

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    model_events = [e for e in _events(outcome) if e.get("latency_ms") is not None]
    assert len(model_events) == 8
    assert all(e["usage"]["calls"] == 1 and e["latency_ms"] == 5.0 for e in model_events)
    manifest = json.loads((outcome.run_dir / "manifest.json").read_text(encoding="utf-8"))
    totals = manifest["outcome"]["usage"]
    assert totals["calls"] == 8
    assert totals["prompt_tokens"] == sum(len(prompt) // 4 for prompt in client.prompts)
    assert totals["latency_ms"] == 40.0


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
    client = scripted(*_rank_then_answer())

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    run_folder = outcome.run_dir.parent
    assert outcome.run_dir.name == "qa"
    assert seen["run_id"] == run_folder.name
    assert (run_folder / "exploration" / "evidence.filtered.json").exists()
    assert "Score header." in client.prompts[0]


def test_qa_and_exploration_use_one_run_id_format(config: dict[str, Any], scripted: Any) -> None:
    import re

    outcome = agent.run(QUERY, [WORKBOOK], config, client=scripted(*_rank_then_answer()))

    assert re.fullmatch(r"\d{8}T\d{12}-\d+", outcome.run_dir.parent.name)


def test_resubmitting_a_failed_answer_unchanged_is_refused(
    config: dict[str, Any], scripted: Any
) -> None:
    client = scripted(
        PLAN_RANK, RANK, STEP_OK,
        PLAN_ANSWER, _answer("Nga Anh Ngo is the man with 99."), _judge("FAIL"),
        PLAN_ANSWER, _answer("Nga Anh Ngo is the man with 99."), _answer(), _judge("PASS"),
    )  # fmt: skip

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.PASS, outcome.reason
    # The unchanged resubmission was re-asked inside the phase, never re-judged.
    assert _answer_verifications(client) == 2
    assert any("repeats the previous action" in prompt for prompt in client.prompts)


TURN_FOLDERS = [
    "01_t0_observe",
    "02_t1_execute_tool",
    "03_t1_verify_tool",
    "04_t1_observe",
    "05_t2_execute_code",
    "06_t2_verify_code",
    "07_t2_observe",
    "08_t3_execute_answer",
    "09_t3_verify_answer",
]
_PLAN_FILES = ["plan.json", "prompt.txt", "reply.txt"]
_VERIFY_FILES = ["checks.json", "judge.json", "prompt.txt", "reply.txt", "verdict.json"]
_STEP_FILES = ["action.json", "observation.txt", "prompt.txt", "reply.txt", "result.json"]


def test_each_model_call_is_one_folder_in_call_order(config: dict[str, Any], scripted: Any) -> None:
    # Reading a run means following it in time: one folder per model call, holding
    # what the model saw, what it replied, and what came of it.
    client = scripted(*_happy())

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    turns = outcome.run_dir / "turns"
    assert sorted(path.name for path in turns.iterdir()) == TURN_FOLDERS
    saw = [
        (turns / name / "prompt.txt").read_text(encoding="utf-8")
        for name in TURN_FOLDERS
        if (turns / name / "prompt.txt").exists()
    ]
    assert saw == client.prompts
    files = {name: sorted(p.name for p in (turns / name).iterdir()) for name in TURN_FOLDERS}
    assert files == {
        "01_t0_observe": _PLAN_FILES,
        "02_t1_execute_tool": _STEP_FILES,
        "03_t1_verify_tool": ["checks.json", "verdict.json"],
        "04_t1_observe": _PLAN_FILES,
        "05_t2_execute_code": sorted([*_STEP_FILES, "code.py"]),
        "06_t2_verify_code": _VERIFY_FILES,
        "07_t2_observe": _PLAN_FILES,
        "08_t3_execute_answer": ["action.json", "prompt.txt", "reply.txt"],
        "09_t3_verify_answer": _VERIFY_FILES,
    }
    for gone in ("actions", "code", "prompts", "verification"):
        assert not (outcome.run_dir / gone).exists()


def test_a_turn_folder_says_what_the_model_replied_and_what_it_got(
    config: dict[str, Any], scripted: Any
) -> None:
    client = scripted(*_happy())

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    turns = outcome.run_dir / "turns"
    assert json.loads((turns / "01_t0_observe" / "reply.txt").read_text()) == PLAN_READ
    observation = (turns / "02_t1_execute_tool" / "observation.txt").read_text()
    assert "2 | No | First | Middle | Last | Birth | Admission | Score" in observation
    step_checks = json.loads((turns / "06_t2_verify_code" / "checks.json").read_text())
    assert step_checks["checks"][0]["id"] == "CALC.ARITHMETIC"
    assert json.loads((turns / "09_t3_verify_answer" / "verdict.json").read_text())["status"] == (
        "PASS"
    )
    code_event = next(
        e for e in _events(outcome) if (e.get("action") or {}).get("action_type") == "code"
    )
    assert "turns/05_t2_execute_code/code.py" in code_event["artifact_paths"]
    code_refs = [p["artifact_path"] for p in code_event["provenance"] if p.get("artifact_path")]
    assert code_refs == ["turns/05_t2_execute_code/code.py"]


def test_a_re_asked_reply_keeps_every_attempt_and_why_it_was_rejected(
    config: dict[str, Any], scripted: Any
) -> None:
    happy = _happy()
    client = scripted(happy[0], "I will read the headers first.", *happy[1:])

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    reply = (outcome.run_dir / "turns" / "02_t1_execute_tool" / "reply.txt").read_text()
    assert "=== attempt 1: rejected" in reply and "no JSON object" in reply
    assert "I will read the headers first." in reply
    assert "=== attempt 2: accepted ===" in reply


def test_the_first_plan_already_knows_what_is_where(config: dict[str, Any], scripted: Any) -> None:
    client = scripted(*_rank_then_answer())

    agent.run(QUERY, [WORKBOOK], config, client=client)

    first = client.prompts[0]
    assert "G: 21 filled (1 text, 20 number); numbers 58..99 in rows 3-22" in first
    assert 'text G1 "Score"' in first
    assert "1 | No | Name |  |  | Date of |  | Score" in first
    assert "3 | 1 | Ha | Minh | Dang | 2005-01-03 00:00:00" in first


def test_a_large_tool_read_is_refused_and_the_next_plan_is_told_to_use_code(
    config: dict[str, Any], scripted: Any
) -> None:
    config["qa"]["tools"]["max_read_cells"] = 20
    read_all = {**READ_HEADERS, "arguments": {**READ_HEADERS["arguments"], "range_ref": "A1:G22"}}
    client = scripted(PLAN_READ, read_all, *_rank_then_answer())

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.PASS, outcome.reason
    assert "at most 20 cells" in client.prompts[0]
    replan = client.prompts[2]
    assert "FAILED verification" in replan
    assert "154 cells" in replan and "code step" in replan


SELECT = {
    "action_type": "code",
    "rationale": "Find the maximum and exactly the rows that hold it.",
    "code": (
        "df = wb.sheet('People', header_row=None)\n"
        "scores = df['G'].loc[3:22].astype(int)\n"
        "top = record_computation('max', ['People!G3:G22'], scores.max())\n"
        "rows = [int(r) for r in scores.index[scores == scores.max()]]\n"
        "only = record_computation('rows_where', ['People!G3:G22'], rows, "
        "condition={'equals': top})\n"
        "result = {'rows': rows, 'names': [' '.join(df.loc[r, ['B', 'C', 'D']]) for r in rows]}"
    ),
}


def test_an_answer_that_names_the_only_matching_row_is_grounded(
    config: dict[str, Any], scripted: Any
) -> None:
    answer = _answer("Nga Anh Ngo (row 13) is the only person with the highest score, 99.")
    answer["answer"]["claims"].append(
        {
            "id": "claim_3",
            "statement": "Row 13 is the only row with the highest score.",
            "value": 13,
            "citations": ["People!G3:G22"],
            "computation_id": "calc_2",
        }
    )
    client = scripted(PLAN_RANK, SELECT, STEP_OK, PLAN_ANSWER, answer, _judge())

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.PASS, outcome.reason
    record = json.loads((outcome.run_dir / "computations" / "calc_2.json").read_text())
    assert record["output"] == [13]


def test_the_answer_judge_sees_the_values_of_the_cited_cells(
    config: dict[str, Any], scripted: Any
) -> None:
    client = scripted(*_rank_then_answer())

    agent.run(QUERY, [WORKBOOK], config, client=client)

    judge_prompt = client.prompts[-1]
    assert 'People!B13:D13: B13 = "Nga", C13 = "Anh", D13 = "Ngo"' in judge_prompt
    assert "People!G3:G22: 20 cells" in judge_prompt


def test_a_computation_recorded_before_a_later_error_still_grounds_an_answer(
    config: dict[str, Any], scripted: Any
) -> None:
    # A real run recorded the maximum, then failed on a later line of the same code; the
    # maximum was marked failed with it, and every answer citing it was refused.
    records_then_fails = {
        "action_type": "code",
        "rationale": "Record the maximum, then trip over something unrelated.",
        "code": "top = record_computation('max', ['People!G3:G22'], 99)\nbad = [1][5]",
    }
    client = scripted(
        PLAN_RANK, records_then_fails, STEP_FAIL,
        PLAN_ANSWER, _answer(), _judge(),
    )  # fmt: skip

    outcome = agent.run(QUERY, [WORKBOOK], config, client=client)

    assert outcome.status is RunStatus.PASS, outcome.reason
