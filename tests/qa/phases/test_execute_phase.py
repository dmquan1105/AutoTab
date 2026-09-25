"""Execute-phase orchestration tests with tool and sandbox fakes; see execute_phase.md."""

from typing import Any

import pytest
from pydantic import TypeAdapter

from autotab.qa.layers.context_management import ContextRequest
from autotab.qa.phases import execute_phase
from autotab.qa.schemas import (
    ActionType,
    AgentAction,
    AnswerAction,
    CodeAction,
    ComputationRecord,
    ObservationSource,
    ObserveResult,
    Phase,
    ResultState,
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


def _request(plan: ObserveResult | None = None) -> ContextRequest:
    return ContextRequest(
        phase=Phase.EXECUTE,
        query="Full name of the man who have the highest score",
        workbooks=[{"workbook_id": "workbook_1", "file": "QA_sample.xlsx", "sheets": []}],
        budget_tokens=20_000,
        output_schema=TypeAdapter(AgentAction).json_schema(),
        tool_descriptions=[{"name": "inspect_range", "description": "Read.", "schema": {}}],
        plan=plan,
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


def test_a_code_action_runs_once_and_returns_its_computations(scripted: Any) -> None:
    registry, sandbox = FakeRegistry(), FakeSandbox()

    outcome = _run(scripted(CODE_ACTION), registry, sandbox)

    assert isinstance(outcome.action, CodeAction)
    assert sandbox.calls == [(CODE_ACTION["code"], "exec_1")]
    assert registry.calls == []
    assert [c.id for c in outcome.computations] == ["calc_1"]


def test_an_answer_action_runs_nothing(scripted: Any) -> None:
    registry, sandbox = FakeRegistry(), FakeSandbox()

    outcome = _run(scripted(ANSWER_ACTION), registry, sandbox)

    assert isinstance(outcome.action, AnswerAction)
    assert outcome.action.answer.answer_text == "Nga Anh Ngo, score 99."
    assert registry.calls == [] and sandbox.calls == []
    assert outcome.observation is None and outcome.raw_result is None


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


def _plan(next_action: str) -> ObserveResult:
    return ObserveResult(
        known_facts=[],
        uncertainties=[],
        execution_objective="Read People!A1:G2.",
        next_action=ActionType(next_action),
        rationale="Headers first.",
    )


def test_an_action_of_another_kind_than_the_plan_names_is_re_asked(scripted: Any) -> None:
    # EXECUTE carries out the plan; choosing the kind of step is OBSERVE's decision.
    registry, sandbox = FakeRegistry(), FakeSandbox()
    client = scripted(CODE_ACTION, TOOL_ACTION)

    outcome = execute_phase.run(
        _request(_plan("tool")), client, registry=registry, sandbox=sandbox, execution_id="e1"
    )

    assert isinstance(outcome.action, ToolAction)
    assert sandbox.calls == []
    assert 'the plan asks for a "tool" action' in client.prompts[1]


def test_an_answer_is_refused_when_the_plan_asks_for_more_data(scripted: Any) -> None:
    client = scripted(ANSWER_ACTION, ANSWER_ACTION)

    outcome = execute_phase.run(
        _request(_plan("code")),
        client,
        registry=FakeRegistry(),
        sandbox=FakeSandbox(),
        execution_id="e1",
    )

    assert outcome.action is None
    assert outcome.reply.error and '"code"' in outcome.reply.error


# --- the reply format follows the planned kind ------------------------------------------

CODE_REPLY = (
    "I will compute the maximum and the rows that hold it.\n\n"
    "```python\n"
    "top = record_computation('max', ['People!G3:G22'], 99)\n"
    "rows = record_computation('rows_where', ['People!G3:G22'], [13], "
    'condition={"equals": top})\n'
    "```\n"
)


def _planned(kind: str, client: Any, registry: Any = None, sandbox: Any = None) -> Any:
    return execute_phase.run(
        _request(_plan(kind)),
        client,
        registry=registry or FakeRegistry(),
        sandbox=sandbox or FakeSandbox(),
        execution_id="e1",
    )


def test_a_planned_code_step_is_read_from_a_python_block(scripted: Any) -> None:
    # Code inside a JSON string has to be escaped, and a real run lost every turn to
    # replies that put it in a fenced block instead, braces and all.
    sandbox = FakeSandbox()

    outcome = _planned("code", scripted(CODE_REPLY), sandbox=sandbox)

    assert isinstance(outcome.action, CodeAction)
    assert outcome.action.code.startswith("top = record_computation('max'")
    assert 'condition={"equals": top})' in outcome.action.code
    assert outcome.action.rationale == "I will compute the maximum and the rows that hold it."
    assert len(sandbox.calls) == 1


def test_a_planned_code_step_still_accepts_the_json_form(scripted: Any) -> None:
    outcome = _planned("code", scripted(CODE_ACTION))

    assert isinstance(outcome.action, CodeAction)
    assert outcome.action.code == CODE_ACTION["code"]


def test_a_code_block_without_rationale_takes_the_objective(scripted: Any) -> None:
    outcome = _planned("code", scripted("```python\nresult = 1\n```"))

    assert isinstance(outcome.action, CodeAction)
    assert outcome.action.rationale == "Read People!A1:G2."


def test_a_planned_tool_call_needs_only_its_name_and_arguments(scripted: Any) -> None:
    reply = {"tool_name": "inspect_range", "arguments": TOOL_ACTION["arguments"]}
    registry = FakeRegistry()

    outcome = _planned("tool", scripted(reply), registry=registry)

    assert isinstance(outcome.action, ToolAction)
    assert registry.calls == [("inspect_range", TOOL_ACTION["arguments"])]


def test_a_planned_answer_may_come_bare_or_wrapped(scripted: Any) -> None:
    bare = _planned("answer", scripted(ANSWER_ACTION["answer"]))
    wrapped = _planned("answer", scripted({"answer": ANSWER_ACTION["answer"]}))

    assert isinstance(bare.action, AnswerAction) and isinstance(wrapped.action, AnswerAction)
    assert bare.action.answer == wrapped.action.answer


@pytest.mark.parametrize(
    ("kind", "reply", "named"),
    [
        ("code", TOOL_ACTION, '"tool"'),
        ("tool", CODE_ACTION, '"code"'),
        ("tool", "```python\nx = 1\n```", '"code"'),
        ("answer", CODE_ACTION, '"code"'),
    ],
)
def test_a_reply_of_another_kind_is_named_in_the_re_ask(
    scripted: Any, kind: str, reply: Any, named: str
) -> None:
    client = scripted(reply, reply)

    outcome = _planned(kind, client)

    assert outcome.action is None
    assert f'the plan asks for a "{kind}" action, not {named}' in client.prompts[1]


def test_the_prompt_asks_for_the_planned_kinds_format_only(scripted: Any) -> None:
    code = _planned("code", scripted(CODE_REPLY)).prompt.text
    tool = _planned("tool", scripted({"tool_name": "inspect_range", "arguments": {}})).prompt.text

    assert "```python" in code and "JSON Schema" not in code
    assert "JSON Schema" in tool and '"tool_name"' in tool
    assert "record_computation(" not in tool


def test_code_nested_inside_its_own_envelope_is_unwrapped(scripted: Any) -> None:
    # A real reply wrapped a whole code action inside the "code" field.
    nested = {"rationale": "Rank.", "code": {"rationale": "Rank.", "code": "result = 1"}}

    outcome = _planned("code", scripted(nested))

    assert isinstance(outcome.action, CodeAction)
    assert outcome.action.code == "result = 1"


def test_a_fence_inside_the_code_does_not_end_the_block(scripted: Any) -> None:
    reply = '```python\nx = "```csv\\na,b\\n```"\nresult = x\n```\n'

    outcome = _planned("code", scripted(reply))

    assert isinstance(outcome.action, CodeAction)
    assert outcome.action.code == 'x = "```csv\\na,b\\n```"\nresult = x'


def test_a_tool_reply_may_carry_an_illustrative_snippet_before_its_json(scripted: Any) -> None:
    reply = (
        "I will look like this:\n```python\nwb.range('A1')\n```\n"
        '{"tool_name": "inspect_range", "arguments": {"workbook_id": "workbook_1", '
        '"range_ref": "A1:G2"}}'
    )

    outcome = _planned("tool", scripted(reply))

    assert isinstance(outcome.action, ToolAction)
