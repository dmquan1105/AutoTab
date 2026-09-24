"""Append-only history and artifact layout; see ``specs/support/artifacts.md``."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from autotab.qa.schemas import HistoryEvent, Phase
from autotab.qa.trace import Trace, redact

SECRET = "sk-or-v1-should-never-be-written"


def _event(trace: Trace, phase: Phase = Phase.INITIALIZE) -> HistoryEvent:
    return HistoryEvent(
        event_id=trace.next_event_id(),
        run_id=trace.run_id,
        turn_id=0,
        phase=phase,
        timestamp=datetime.now(UTC),
    )


def test_run_root_is_namespaced_under_qa(tmp_path: Path) -> None:
    trace = Trace(tmp_path, "qa_run_1")

    assert trace.root == tmp_path.resolve() / "qa_run_1" / "qa"
    assert trace.root.is_dir()


def test_history_is_append_only_and_ordered(tmp_path: Path) -> None:
    trace = Trace(tmp_path, "qa_run_1")
    first, second = _event(trace), _event(trace, Phase.OBSERVE)

    trace.append(first)
    trace.append(second)

    assert [event.event_id for event in trace.history()] == [first.event_id, second.event_id]
    lines = (trace.root / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "normalized_observation" not in json.loads(lines[0])


def test_event_ids_are_stable_and_sequential(tmp_path: Path) -> None:
    trace = Trace(tmp_path, "qa_run_1")

    assert [trace.next_event_id() for _ in range(3)] == ["event_0001", "event_0002", "event_0003"]


def test_execution_artifacts_are_immutable(tmp_path: Path) -> None:
    trace = Trace(tmp_path, "qa_run_1")
    trace.write_json("actions/exec_1.json", {"action": "first"})

    with pytest.raises(FileExistsError):
        trace.write_json("actions/exec_1.json", {"action": "second"})


def test_json_artifacts_carry_schema_version_and_run_id(tmp_path: Path) -> None:
    trace = Trace(tmp_path, "qa_run_1")

    path = trace.write_json("computations/calc_1.json", {"id": "calc_1"})
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0"
    assert payload["run_id"] == "qa_run_1"
    assert payload["id"] == "calc_1"


@pytest.mark.parametrize("path", ["../escape.json", "/tmp/escape.json", "actions/../../x.json"])
def test_paths_outside_the_run_root_are_refused(tmp_path: Path, path: str) -> None:
    trace = Trace(tmp_path, "qa_run_1")

    with pytest.raises(ValueError):
        trace.write_json(path, {})


def test_manifest_and_final_answer_are_replaced_atomically(tmp_path: Path) -> None:
    trace = Trace(tmp_path, "qa_run_1")

    trace.write_manifest({"query": "first"})
    trace.write_manifest({"query": "second"})
    trace.write_final_answer("Nga Anh Ngo")

    manifest = json.loads((trace.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["query"] == "second"
    assert (trace.root / "final" / "answer.md").read_text(encoding="utf-8") == "Nga Anh Ngo"
    assert not list(trace.root.rglob("*.tmp"))


def test_redaction_hides_secrets_but_not_token_limits() -> None:
    # A redaction pattern that is too narrow leaked a key once in this project; one
    # that is too broad would hide max_tokens. Both directions are pinned here.
    config = {
        "models": {
            "llm": {"api_key": SECRET, "max_tokens": 4000, "model": "deepseek"},
            "vlm": {"apiKey": SECRET, "extra_body": {"reasoning": {"enabled": False}}},
            "embedding": {"access_token": SECRET, "password": SECRET, "secret": SECRET},
        },
        "qa": {"max_context_tokens": 32000},
    }

    redacted = redact(config)

    assert SECRET not in json.dumps(redacted)
    assert redacted["models"]["llm"]["max_tokens"] == 4000
    assert redacted["models"]["llm"]["model"] == "deepseek"
    assert redacted["qa"]["max_context_tokens"] == 32000
    assert config["models"]["llm"]["api_key"] == SECRET


def test_manifest_is_redacted_on_write(tmp_path: Path) -> None:
    trace = Trace(tmp_path, "qa_run_1")

    trace.write_manifest({"configuration": {"models": {"llm": {"api_key": SECRET}}}})

    assert SECRET not in (trace.root / "manifest.json").read_text(encoding="utf-8")


def test_a_second_run_cannot_write_into_an_existing_history(tmp_path: Path) -> None:
    # exploration/ and qa/ now share a run folder; two QA runs must never interleave
    # their events in one history.jsonl.
    first = Trace(tmp_path, "20260924T100000000000-1")
    first.append(_event(first))

    with pytest.raises(FileExistsError):
        Trace(tmp_path, "20260924T100000000000-1")
