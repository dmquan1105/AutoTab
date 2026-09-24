"""Execute-phase orchestration tests with tool and sandbox fakes; see execute_phase.md."""

from typing import Any

from pydantic import TypeAdapter

from autotab.qa.layers.context_management import ContextRequest
from autotab.qa.phases import execute_phase
from autotab.qa.schemas import (
    AgentAction,
    AnswerAction,
    CodeAction,
    ComputationRecord,
    ObservationSource,
    Phase,
    ResultState,
    RunStatus,
    SandboxResult,
    ToolAction,
    ToolResult,
)

TOOL_ACTION = {
    "action_type": "tool",
    "rationale": "Read the header rows.",
    "tool_name": "inspect_range",
    "arguments": {"workbook_id": "workbook_1", "range_ref": "A1:G2"},
}
CODE_ACTION = {
    "action_type": "code",
    "rationale": "Rank by score.",
    "code": "result = record_computation('max', [], 99)",
}
ANSWER_ACTION = {
    "action_type": "answer",
    "rationale": "Grounded.",
    "answer": {
        "answer_text": "Nga Anh Ngo, score 99.",
        "claims": [
            {
                "id": "claim_1",
                "statement": "Top score 99.",
                "value": 99,
                "citations": ["People!G13"],
                "computation_id": None,
            }
        ],
    },
}


class FakeRegistry:
    def __init__(self, result: ToolResult | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.result = result or ToolResult(
            state=ResultState.SUCCESS,
            tool_name="inspect_range",
            arguments=TOOL_ACTION["arguments"],
            result={
                "sheet": "People",
                "range": "G1",
                "cells": [
                    [{"coordinate": "G1", "value": "Score", "displayed": "Score", "formula": None}]
                ],
            },
        )

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append((tool_name, arguments))
        return self.result


class FakeSandbox:
    def __init__(self, output: Any = 99) -> None:
        self.calls: list[tuple[str, str]] = []
        self.computations: dict[str, ComputationRecord] = {}
        self.output = output

    def execute(self, code: str, execution_id: str) -> SandboxResult:
        self.calls.append((code, execution_id))
        self.computations["calc_1"] = ComputationRecord.model_validate_json(
            '{"id": "calc_1", "operation": "max", "inputs": ['
            '{"reference": "People!G3", "value": 92}, {"reference": "People!G13", "value": 99}'
            f'], "output": {self.output}}}'
        )
        return SandboxResult(
            state=ResultState.SUCCESS,
            execution_id=execution_id,
            result="calc_1",
            computation_ids=["calc_1"],
        )


def _request() -> ContextRequest:
    return ContextRequest(
        phase=Phase.EXECUTE,
        query="Full name of the man who have the highest score",
        workbooks=[{"workbook_id": "workbook_1", "file": "QA_sample.xlsx", "sheets": []}],
        budget_tokens=20_000,
        output_schema=TypeAdapter(AgentAction).json_schema(),
        tool_descriptions=[{"name": "inspect_range", "description": "Read.", "schema": {}}],
    )


def _run(client: Any, registry: Any = None, sandbox: Any = None) -> execute_phase.ExecuteOutcome:
    return execute_phase.run(
        _request(),
        client,
        registry=registry or FakeRegistry(),
        sandbox=sandbox or FakeSandbox(),
        execution_id="exec_1",
    )


def test_a_tool_action_dispatches_once_and_is_normalized(scripted: Any) -> None:
    registry, sandbox = FakeRegistry(), FakeSandbox()

    outcome = _run(scripted(TOOL_ACTION), registry, sandbox)

    assert isinstance(outcome.action, ToolAction)
    assert registry.calls == [("inspect_range", TOOL_ACTION["arguments"])]
    assert sandbox.calls == []
    assert outcome.observation is not None
    assert outcome.observation.source is ObservationSource.TOOL
    assert any("People!G1" in fact for fact in outcome.observation.facts)
    assert outcome.report is not None and outcome.report.status is RunStatus.PASS


def test_a_code_action_runs_once_and_checks_its_computations(scripted: Any) -> None:
    registry, sandbox = FakeRegistry(), FakeSandbox()

    outcome = _run(scripted(CODE_ACTION), registry, sandbox)

    assert isinstance(outcome.action, CodeAction)
    assert sandbox.calls == [(CODE_ACTION["code"], "exec_1")]
    assert registry.calls == []
    assert [c.id for c in outcome.computations] == ["calc_1"]
    assert outcome.report is not None and outcome.report.status is RunStatus.PASS


def test_a_wrong_computation_fails_action_scope_without_a_model_call(scripted: Any) -> None:
    client = scripted(CODE_ACTION)

    outcome = _run(client, sandbox=FakeSandbox(output=98))

    assert outcome.report is not None and outcome.report.status is RunStatus.FAIL
    assert len(client.prompts) == 1


def test_an_answer_action_runs_nothing(scripted: Any) -> None:
    registry, sandbox = FakeRegistry(), FakeSandbox()

    outcome = _run(scripted(ANSWER_ACTION), registry, sandbox)

    assert isinstance(outcome.action, AnswerAction)
    assert outcome.action.answer.answer_text == "Nga Anh Ngo, score 99."
    assert registry.calls == [] and sandbox.calls == []
    assert outcome.observation is None and outcome.report is None


def test_invalid_actions_dispatch_nothing(scripted: Any) -> None:
    registry, sandbox = FakeRegistry(), FakeSandbox()
    smuggled = {**TOOL_ACTION, "code": "print(1)"}

    outcome = _run(scripted("not json", smuggled), registry, sandbox)

    assert outcome.action is None
    assert outcome.reply.error
    assert registry.calls == [] and sandbox.calls == []


def test_a_rejected_tool_call_is_a_recoverable_observation(scripted: Any) -> None:
    rejected = ToolResult(
        state=ResultState.ERROR,
        tool_name="execute_excel",
        arguments={},
        error="unknown tool 'execute_excel'",
    )

    outcome = _run(scripted({**TOOL_ACTION, "tool_name": "execute_excel"}), FakeRegistry(rejected))

    assert outcome.observation is not None
    assert outcome.observation.status is ResultState.ERROR
    assert outcome.report is not None and outcome.report.status is RunStatus.FAIL


def test_an_unchanged_repeat_of_the_previous_action_is_re_asked(scripted: Any) -> None:
    # specs/phases/agent.md: never repeat an unchanged action; its result is known.
    previous = TypeAdapter(AgentAction).validate_python(TOOL_ACTION)
    reworded = {**TOOL_ACTION, "rationale": "Same read, different words."}
    registry, sandbox = FakeRegistry(), FakeSandbox()
    client = scripted(reworded, CODE_ACTION)

    outcome = execute_phase.run(
        _request(),
        client,
        registry=registry,
        sandbox=sandbox,
        execution_id="exec_2",
        previous=previous,
    )

    assert isinstance(outcome.action, CodeAction)
    assert registry.calls == []
    assert "repeats the previous action" in client.prompts[1]


def test_a_changed_action_is_not_a_repeat(scripted: Any) -> None:
    previous = TypeAdapter(AgentAction).validate_python(TOOL_ACTION)
    other_range = {**TOOL_ACTION, "arguments": {"workbook_id": "workbook_1", "range_ref": "A3:G22"}}
    registry = FakeRegistry()

    execute_phase.run(
        _request(),
        scripted(other_range),
        registry=registry,
        sandbox=FakeSandbox(),
        execution_id="exec_2",
        previous=previous,
    )

    assert len(registry.calls) == 1
