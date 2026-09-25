"""One EXECUTE interaction: call the model, validate one AgentAction, dispatch it.

Dispatches exactly the one tool call, sandbox run, or answer the plan names, then
normalizes the result. The plan names the kind, so the reply is asked for in that
kind's own format -- code in a fenced block, a tool call or an answer as a small JSON
object -- and the harness builds the action; a three-way JSON union with code escaped
inside a string was the format real replies kept breaking. Checking and judging the
result is VERIFY's job; this phase never transitions the FSM either; see
``specs/phases/execute_phase.md``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol

from pydantic import TypeAdapter

from ...models.client import ModelClient
from ..layers.context_management import AssembledPrompt, ContextRequest, assemble_context
from ..layers.observation import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_INLINE_CELLS,
    normalize_sandbox,
    normalize_tool,
)
from ..layers.verification import extract_json_object
from ..schemas import (
    ActionType,
    AgentAction,
    AnswerAction,
    AnswerReply,
    CodeAction,
    ComputationRecord,
    Observation,
    Phase,
    SandboxResult,
    ToolAction,
    ToolReply,
    ToolResult,
)
from ._ask import Reply, ask

_ACTION: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)
# The closing fence stands on a line of its own, so ``` inside the code cannot end it.
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)\n[ \t]*```[ \t]*(?:\n|$)", re.DOTALL)
# The JSON a reply must match, per planned kind; code comes in a fenced block instead.
REPLY_SCHEMAS: dict[ActionType, Mapping[str, Any] | None] = {
    ActionType.TOOL: ToolReply.model_json_schema(),
    ActionType.CODE: None,
    ActionType.ANSWER: AnswerReply.model_json_schema(),
}


class ToolDispatcher(Protocol):
    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult: ...


class CodeRunner(Protocol):
    @property
    def computations(self) -> Mapping[str, ComputationRecord]: ...

    def execute(self, code: str, execution_id: str) -> SandboxResult: ...


@dataclass(frozen=True)
class ExecuteOutcome:
    """The action and what running it produced.

    ``raw_result`` and ``observation`` are None for an answer action and when no
    valid action came back.
    """

    prompt: AssembledPrompt
    reply: Reply[AgentAction]
    raw_result: ToolResult | SandboxResult | None = None
    observation: Observation | None = None
    computations: tuple[ComputationRecord, ...] = ()

    @property
    def action(self) -> AgentAction | None:
        return self.reply.value


def _parse(raw: str) -> AgentAction:
    return _ACTION.validate_json(extract_json_object(raw))


def _wrong_kind(expected: ActionType, found: str) -> ValueError:
    return ValueError(
        f'the plan asks for a "{expected.value}" action, not "{found}"; take the action '
        "the plan names"
    )


def _payload(raw: str, expected: ActionType) -> dict[str, Any]:
    """The reply's JSON object, refusing one that declares or looks like another kind."""
    payload: dict[str, Any] = json.loads(extract_json_object(raw))
    declared = payload.pop("action_type", None)
    if declared is not None and declared != expected.value:
        raise _wrong_kind(expected, str(declared))
    if expected is not ActionType.CODE and "code" in payload:
        raise _wrong_kind(expected, "code")
    if expected is not ActionType.TOOL and "tool_name" in payload:
        raise _wrong_kind(expected, "tool")
    return payload


def _parse_planned(raw: str, expected: ActionType, objective: str) -> AgentAction:
    """Build the planned kind of action from a reply in that kind's format."""
    block = _CODE_BLOCK.search(raw)
    if expected is ActionType.CODE:
        if block is not None:
            rationale = (raw[: block.start()] + raw[block.end() :]).strip()
            code = block.group(1)
        else:
            payload = _payload(raw, expected)
            code, rationale = payload.get("code"), str(payload.get("rationale") or "")
            if isinstance(code, dict):  # a whole action nested inside its own envelope
                code = code.get("code")
            if not isinstance(code, str):
                raise ValueError("reply with the code in one ```python block")
        return CodeAction(
            action_type=ActionType.CODE, code=code.strip(), rationale=rationale or objective
        )
    try:
        payload = _payload(raw, expected)
    except ValueError:
        if block is not None:  # a fenced block and no JSON at all: a code reply
            raise _wrong_kind(expected, "code") from None
        raise
    if expected is ActionType.TOOL:
        tool = ToolReply.model_validate_json(json.dumps(payload))
        return ToolAction(
            action_type=ActionType.TOOL,
            tool_name=tool.tool_name,
            arguments=tool.arguments,
            rationale=tool.rationale or objective,
        )
    if "answer" not in payload and "answer_text" in payload:
        payload = {"answer": payload}  # the answer itself, without its envelope
    answer = AnswerReply.model_validate_json(json.dumps(payload))
    return AnswerAction(
        action_type=ActionType.ANSWER, answer=answer.answer, rationale=answer.rationale or objective
    )


def _effect(action: AgentAction) -> tuple[str, ...]:
    """What an action does, ignoring how its rationale is worded."""
    if isinstance(action, ToolAction):
        return ("tool", action.tool_name, json.dumps(action.arguments, sort_keys=True))
    if isinstance(action, CodeAction):
        return ("code", action.code.strip())
    return ("answer", action.answer.model_dump_json())


def _parser(
    previous: AgentAction | None, expected: ActionType | None, objective: str = ""
) -> Callable[[str], AgentAction]:
    """Parse a reply, refusing another kind than the plan names and unchanged repeats.

    specs/phases/agent.md: EXECUTE carries out the plan, and never repeats an unchanged
    action -- its result is already in the context. Raising here reuses the phase's
    re-ask, so neither refusal costs a turn.
    """

    def parse(raw: str) -> AgentAction:
        action = _parse(raw) if expected is None else _parse_planned(raw, expected, objective)
        if expected is not None and action.action_type is not expected:
            raise ValueError(
                f'the plan asks for a "{expected.value}" action, not '
                f'"{action.action_type.value}"; take the action the plan names'
            )
        if previous is not None and _effect(action) == _effect(previous):
            raise ValueError(
                "this repeats the previous action unchanged, and its result is already "
                "above; choose a different action that makes progress"
            )
        return action

    return parse


def run(
    request: ContextRequest,
    client: ModelClient,
    *,
    registry: ToolDispatcher,
    sandbox: CodeRunner,
    execution_id: str,
    code_path: str | None = None,
    retries: int = 1,
    max_observation_chars: int = DEFAULT_MAX_CHARS,
    max_inline_cells: int = DEFAULT_MAX_INLINE_CELLS,
    previous: AgentAction | None = None,
) -> ExecuteOutcome:
    """Choose and run exactly one action.

    Raises:
        ValueError: If the request is not for the EXECUTE phase.
        ContextBudgetError: If required content cannot fit the context budget.
    """
    if request.phase is not Phase.EXECUTE:
        raise ValueError(f"execute_phase cannot build a {request.phase.value} prompt")
    plan = request.plan
    if plan is not None:
        request = replace(request, output_schema=REPLY_SCHEMAS[plan.next_action])
    prompt = assemble_context(request)
    expected = plan.next_action if plan is not None else None
    objective = plan.execution_objective if plan is not None else ""
    reply = ask(client, prompt.text, _parser(previous, expected, objective), retries=retries)
    action = reply.value
    if isinstance(action, ToolAction):
        tool_result = registry.dispatch(action.tool_name, dict(action.arguments))
        return ExecuteOutcome(
            prompt,
            reply,
            raw_result=tool_result,
            observation=normalize_tool(
                tool_result, max_chars=max_observation_chars, max_inline_cells=max_inline_cells
            ),
        )
    if isinstance(action, CodeAction):
        sandbox_result = sandbox.execute(action.code, execution_id)
        computations = tuple(
            sandbox.computations[computation_id]
            for computation_id in sandbox_result.computation_ids
            if computation_id in sandbox.computations
        )
        return ExecuteOutcome(
            prompt,
            reply,
            raw_result=sandbox_result,
            observation=normalize_sandbox(
                sandbox_result, max_chars=max_observation_chars, code_path=code_path
            ),
            computations=computations,
        )
    # An answer runs nothing, and no valid action means nothing may run.
    return ExecuteOutcome(prompt, reply)
