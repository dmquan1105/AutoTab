"""Deterministic context assembly, compaction, and token budget.

Selection is application code, never a model call: identical inputs and budget give
an identical prompt. Raw history stays in the trace; this layer only decides what
the model sees. See ``specs/layers/context_management.md``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..prompts import (
    EXECUTE_INSTRUCTIONS,
    OBSERVE_AFTER_FAILURE,
    OBSERVE_INSTRUCTIONS,
    SANDBOX_CAPABILITIES,
    SYSTEM_INVARIANTS,
    VERIFICATION_INSTRUCTIONS,
    format_tool_descriptions,
    output_schema,
    rubric_text,
)
from ..schemas import (
    AgentAction,
    AnswerAction,
    CandidateAnswer,
    CodeAction,
    ComputationRecord,
    DeterministicCheckReport,
    FeedbackPriority,
    ImprovementFeedback,
    LLMJudgeResult,
    Observation,
    ObserveResult,
    Phase,
    RunStatus,
)
from .verification import feedback_for, is_not_applicable

CHARS_PER_TOKEN = 4
TOKEN_COUNTER = f"chars_per_token:{CHARS_PER_TOKEN}"
# Kept free while choosing optional content, so the omission note always fits.
_NOTE_RESERVE_TOKENS = 120


class ContextBudgetError(ValueError):
    """Raised when required content cannot fit; nothing blocking is ever dropped."""


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate; recorded in the manifest as TOKEN_COUNTER."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Step:
    """One executed turn: the action and its normalized observation."""

    turn: int
    action: AgentAction
    observation: Observation
    execution_id: str


@dataclass(frozen=True)
class ContextRequest:
    """Everything one phase prompt may draw on."""

    phase: Phase
    query: str
    workbooks: Sequence[Mapping[str, Any]]
    budget_tokens: int
    output_schema: Mapping[str, Any]
    exploration: Sequence[Observation] = ()
    steps: Sequence[Step] = ()
    plan: ObserveResult | None = None
    feedback: Sequence[ImprovementFeedback] = ()
    after_failure: bool = False
    tool_descriptions: Sequence[Mapping[str, Any]] = ()
    sandbox_names: Sequence[str] = ()
    candidate: CandidateAnswer | None = None
    report: DeterministicCheckReport | None = None
    computations: Mapping[str, ComputationRecord] = field(default_factory=dict)


@dataclass(frozen=True)
class AssembledPrompt:
    """A rendered prompt, its estimated size, and what was shortened to fit."""

    text: str
    tokens: int
    omitted: tuple[str, ...] = field(default_factory=tuple)


# --- rendering ---------------------------------------------------------------------


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _workbooks(workbooks: Sequence[Mapping[str, Any]]) -> str:
    lines = ["Workbooks:"]
    for book in workbooks:
        lines.append(f"- {book.get('workbook_id')} ({book.get('file')})")
        for sheet in book.get("sheets") or []:
            merged = sheet.get("merged") or []
            extra = f", merged ranges: {', '.join(merged)}" if merged else ""
            lines.append(
                f"  - sheet {sheet.get('name')!r}, used range {sheet.get('dimensions')}{extra}"
            )
    return "\n".join(lines)


def _where(observation: Observation) -> str:
    for reference in observation.provenance:
        if reference.sheet and reference.range_ref:
            return f"{reference.sheet}!{reference.range_ref}"
    return ""


def _lead(observation: Observation, compact: bool) -> str:
    head = f"- [{observation.summary}] {_where(observation)}"
    if compact:
        return f"{head} (lead; description omitted to fit the budget)"
    return f"{head} (lead, confirm with an exact read): {' '.join(observation.facts)}"


_LEADS_TITLE = (
    "Exploration leads (unverified). Each describes a small window around one cell, not "
    "the table: rows it mentions are examples, and the workbook manifest's used range "
    "shows the full extent."
)


def _action_line(action: AgentAction, compact: bool) -> str:
    if isinstance(action, CodeAction):
        if compact:
            return f"code ({len(action.code.splitlines())} lines)"
        return "code:\n" + "\n".join(f"    {line}" for line in action.code.splitlines())
    if isinstance(action, AnswerAction):
        return "answer" if compact else f"answer: {action.answer.answer_text}"
    return f"tool {action.tool_name} {_json(action.arguments)}"


def _step(step: Step, compact: bool) -> str:
    observation = step.observation
    head = f"Turn {step.turn} ({step.execution_id}): {_action_line(step.action, compact)}"
    if compact:
        return (
            f"{head} -> {observation.status.value}; details omitted to fit the budget "
            f"(actions/{step.execution_id}.json)"
        )
    lines = [head, f"Result: {observation.status.value} - {observation.summary}"]
    if observation.error:
        lines.append(f"  error: {observation.error}")
    lines.extend(f"  - {fact}" for fact in observation.facts)
    lines.extend(f"  uncertain: {note}" for note in observation.uncertainties)
    if observation.next_question:
        lines.append(f"  next question: {observation.next_question}")
    return "\n".join(lines)


def _plan(plan: ObserveResult) -> str:
    lines = ["Current plan:", f"Objective: {plan.execution_objective}"]
    if plan.known_facts:
        lines.append("Known facts:")
        lines.extend(f"- {fact}" for fact in plan.known_facts)
    if plan.uncertainties:
        lines.append("Uncertainties:")
        lines.extend(f"- {note}" for note in plan.uncertainties)
    lines.append(f"Rationale: {plan.rationale}")
    return "\n".join(lines)


def _feedback(items: Sequence[ImprovementFeedback], title: str) -> str:
    lines = [title]
    for item in items:
        lines.append(f"- [{item.code}] source={item.source.value}, recheck={item.recheck}")
        lines.append(f"  problem: {item.problem}")
        if item.evidence:
            lines.append(f"  evidence: {', '.join(item.evidence)}")
        lines.append(f"  recommended action: {item.action}")
    return "\n".join(lines)


def _report(report: DeterministicCheckReport) -> str:
    lines = [f"Deterministic checks: {report.status.value}"]
    lines.extend(f"- {c.id}: {c.status.value} - {c.reason}" for c in report.checks)
    return "\n".join(lines)


def _provenance(candidate: CandidateAnswer, computations: Mapping[str, Any]) -> str:
    """Where each computation the answer relies on came from.

    With large tables shown only as previews, this is how the judge checks coverage:
    a computation's sources and input count say exactly which cells it used.
    """
    lines = []
    cited = dict.fromkeys(str(c.computation_id) for c in candidate.claims if c.computation_id)
    for computation_id in cited:
        record = computations.get(computation_id)
        if record is None:
            continue
        inputs = record.inputs
        refs = [item.reference or "constant" for item in inputs]
        span = f"{refs[0]} … {refs[-1]}" if len(refs) > 1 else (refs[0] if refs else "none")
        sources = record.metadata.get("sources") or []
        origin = ", ".join(str(source) for source in sources) if sources else "declared inputs"
        skipped = record.metadata.get("skipped_blank") or []
        blanks = f"; {len(skipped)} blank skipped" if isinstance(skipped, list) and skipped else ""
        lines.append(
            f"- {record.id}: {record.operation} over {origin} -> {len(inputs)} inputs "
            f"({span}){blanks}; output {record.output!r}"
        )
    if not lines:
        return ""
    return "Recorded computations the answer relies on:\n" + "\n".join(lines)


def _candidate(candidate: CandidateAnswer) -> str:
    claims = [claim.model_dump(mode="json") for claim in candidate.claims]
    return (
        f"Candidate answer:\n{candidate.answer_text}\n"
        f"Claims:\n{json.dumps(claims, ensure_ascii=False, indent=1)}"
    )


def _capabilities(request: ContextRequest) -> str:
    if request.phase is Phase.EXECUTE:
        bound = (
            f"Currently bound in the session: {', '.join(request.sandbox_names)}"
            if request.sandbox_names
            else "Nothing is bound in the session yet."
        )
        return "\n\n".join(
            [format_tool_descriptions(request.tool_descriptions), SANDBOX_CAPABILITIES, bound]
        )
    if request.phase is Phase.OBSERVE:
        names = ", ".join(str(item["name"]) for item in request.tool_descriptions)
        return (
            "Capabilities of the next step (chosen there, not here): read-only workbook "
            f"tools ({names or 'none'}), a persistent Python session with pandas and a "
            "read-only workbook object, and answering with grounded claims."
        )
    return ""


def _instructions(request: ContextRequest) -> str:
    if request.phase is Phase.OBSERVE:
        return (
            f"{OBSERVE_INSTRUCTIONS}\n\n{OBSERVE_AFTER_FAILURE}"
            if request.after_failure
            else OBSERVE_INSTRUCTIONS
        )
    if request.phase is Phase.EXECUTE:
        return EXECUTE_INSTRUCTIONS
    return f"{VERIFICATION_INSTRUCTIONS}\n\n{rubric_text()}"


# --- selection ---------------------------------------------------------------------


class _Budget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def fits(self, text: str, reserve: int = 0) -> bool:
        return self.used + estimate_tokens(text) + reserve <= self.limit

    def take(self, text: str) -> str:
        self.used += estimate_tokens(text)
        return text


def assemble_context(request: ContextRequest) -> AssembledPrompt:
    """Build the bounded prompt for one phase.

    Raises:
        ContextBudgetError: If required sections or blocking feedback cannot fit.
    """
    if request.phase not in (Phase.OBSERVE, Phase.EXECUTE, Phase.VERIFY):
        raise ValueError(f"no prompt template for phase {request.phase.value}")
    budget = _Budget(request.budget_tokens)
    omitted: list[str] = []

    # 1. Reserved: invariants, verbatim query, workbooks, template, schema, and for
    #    verification the complete candidate, its checks, and the rubric.
    reserved = {
        "invariants": SYSTEM_INVARIANTS,
        "question": f"Question (verbatim):\n{request.query}",
        "workbooks": _workbooks(request.workbooks),
        "candidate": _candidate(request.candidate) if request.candidate else "",
        "provenance": (
            _provenance(request.candidate, request.computations)
            if request.candidate and request.phase is Phase.VERIFY
            else ""
        ),
        "report": _report(request.report) if request.report else "",
        "capabilities": _capabilities(request),
        "instructions": _instructions(request),
        "schema": output_schema(request.output_schema),
    }
    for text in reserved.values():
        budget.take(text)
    if budget.used > budget.limit:
        raise ContextBudgetError(
            f"required sections need {budget.used} tokens, above the budget of {budget.limit}"
        )

    # The judge sees the current checks, never earlier verdicts: an item resolved by the
    # candidate under review would otherwise still read as "must be resolved".
    feedback = () if request.phase is Phase.VERIFY else tuple(request.feedback)

    # 2. Every active blocking item, whole.
    blocking = [item for item in feedback if item.priority is FeedbackPriority.BLOCKING]
    blocking_text = ""
    if blocking:
        title = "Blocking feedback (must be resolved before the answer can pass):"
        blocking_text = _feedback(blocking, title)
        if not budget.fits(blocking_text):
            codes = ", ".join(item.code for item in blocking)
            raise ContextBudgetError(
                f"blocking feedback {codes} does not fit in {budget.limit} tokens; "
                "raise qa.max_context_tokens"
            )
        budget.take(blocking_text)

    # 3. The plan and the latest observation.
    plan_text = _plan(request.plan) if request.plan else ""
    if plan_text and budget.fits(plan_text, _NOTE_RESERVE_TOKENS):
        budget.take(plan_text)
    elif plan_text:
        omitted.append("the current plan")
        plan_text = ""

    def choose(full: str, compact: str, label: str) -> str:
        if budget.fits(full, _NOTE_RESERVE_TOKENS):
            return budget.take(full)
        omitted.append(f"{label} (shortened)")
        if budget.fits(compact, _NOTE_RESERVE_TOKENS):
            return budget.take(compact)
        omitted[-1] = f"{label} (omitted)"
        return ""

    steps = list(request.steps)
    latest = (
        choose(_step(steps[-1], False), _step(steps[-1], True), f"turn {steps[-1].turn}")
        if steps
        else ""
    )

    # 4. Earlier turns, newest first, then exploration leads.
    earlier: dict[int, str] = {}
    for step in reversed(steps[:-1]):
        earlier[step.turn] = choose(_step(step, False), _step(step, True), f"turn {step.turn}")
    leads = [
        choose(_lead(lead, False), _lead(lead, True), f"exploration lead {lead.summary!r}")
        for lead in request.exploration
    ]

    # 5. Non-blocking suggestions only from what is left.
    advisory = [item for item in feedback if item.priority is FeedbackPriority.NON_BLOCKING]
    advisory_text = ""
    if advisory:
        advisory_text = choose(
            _feedback(advisory, "Optional suggestions (non-blocking):"),
            "",
            "non-blocking suggestions",
        )

    note = ""
    if omitted:
        note = (
            "Note: to fit the context budget these were shortened or omitted: "
            + "; ".join(omitted)
            + ". The history above is incomplete; the full record is in the run artifacts."
        )
        budget.take(note)

    sections = [
        reserved["invariants"],
        reserved["question"],
        reserved["workbooks"],
        (_LEADS_TITLE + "\n" + "\n".join(t for t in leads if t) if any(leads) else ""),
        plan_text,
        (
            "Earlier turns:\n" + "\n".join(earlier[t] for t in sorted(earlier) if earlier[t])
            if any(earlier.values())
            else ""
        ),
        f"Latest turn:\n{latest}" if latest else "",
        blocking_text,
        advisory_text,
        reserved["report"],
        reserved["candidate"],
        reserved["provenance"],
        reserved["capabilities"],
        note,
        reserved["instructions"],
        reserved["schema"],
    ]
    text = "\n\n".join(section for section in sections if section)
    return AssembledPrompt(text=text, tokens=estimate_tokens(text), omitted=tuple(omitted))


def active_feedback(
    events: Sequence[DeterministicCheckReport | LLMJudgeResult],
) -> list[ImprovementFeedback]:
    """Derive the feedback still in force from ordered verification events.

    A deterministic item stays active until a newer, applicable PASS of the same
    check; a PASS that was only "not applicable" resolves nothing. The judge's set is
    exactly the latest judgement's feedback when that judgement failed.
    """
    checks: dict[str, ImprovementFeedback | None] = {}
    latest_judge: LLMJudgeResult | None = None
    for event in events:
        if isinstance(event, LLMJudgeResult):
            latest_judge = event
            continue
        failed = {item.recheck: item for item in feedback_for(event)}
        for check in event.checks:
            if check.status is RunStatus.FAIL:
                checks[check.id] = failed[check.id]
            elif not is_not_applicable(check):
                checks[check.id] = None
    items = [item for item in checks.values() if item is not None]
    if latest_judge is not None and latest_judge.status is RunStatus.FAIL:
        items.extend(latest_judge.improvement_feedback)
    return sorted(items, key=lambda item: item.priority is not FeedbackPriority.BLOCKING)
