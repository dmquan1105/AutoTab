"""CLI wiring for the qa command; the agent itself is replaced by a recorder."""

from pathlib import Path
from typing import Any

import pytest

from autotab import cli
from autotab.qa.agent import RunOutcome
from autotab.qa.schemas import RunStatus


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def fake_run(query: str, workbooks: list[str], config: dict[str, Any]) -> RunOutcome:
        calls.update(query=query, workbooks=workbooks, qa=dict(config["qa"]))
        return RunOutcome(
            status=calls.get("status", RunStatus.PASS),
            reason="checks and judge passed",
            run_dir=tmp_path,
            turns=3,
            answer_text="Nga Anh Ngo",
        )

    monkeypatch.setattr("autotab.qa.agent.run", fake_run)
    return calls


def test_qa_passes_query_workbooks_and_exploration_path(
    recorded: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "qa",
            "--config",
            "config.example.yaml",
            "--workbook",
            "samples/QA_sample.xlsx",
            "--query",
            "Who scored highest?",
            "--exploration",
            "outputs/some-run/exploration",
        ]
    )

    assert code == 0
    assert recorded["query"] == "Who scored highest?"
    assert recorded["workbooks"] == ["samples/QA_sample.xlsx"]
    assert recorded["qa"]["exploration_enabled"] is True
    assert recorded["qa"]["exploration_path"] == "outputs/some-run/exploration"
    out = capsys.readouterr().out
    assert "PASS" in out and "Nga Anh Ngo" in out


def test_no_exploration_disables_the_dependency(recorded: dict[str, Any]) -> None:
    cli.main(
        [
            "qa",
            "--config",
            "config.example.yaml",
            "--workbook",
            "samples/QA_sample.xlsx",
            "--query",
            "q",
            "--no-exploration",
        ]
    )

    assert recorded["qa"]["exploration_enabled"] is False


def test_a_failed_run_exits_non_zero(recorded: dict[str, Any]) -> None:
    recorded["status"] = RunStatus.FAIL

    code = cli.main(
        ["qa", "--config", "config.example.yaml", "--workbook", "w.xlsx", "--query", "q"]
    )

    assert code == 1
