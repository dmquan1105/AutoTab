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

from openpyxl.utils.cell import range_boundaries

from ..prompts import (
    EXECUTE_INSTRUCTIONS,
    OBSERVE_AFTER_FAILURE,
    OBSERVE_INSTRUCTIONS,
    PLANNED_EXECUTE_INSTRUCTIONS,
    SANDBOX_CAPABILITIES,
    STEP_VERIFICATION_INSTRUCTIONS,
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
    FeedbackSource,
    ImprovementFeedback,
    LLMJudgeResult,
    Observation,
    ObserveResult,
    Phase,
    RunStatus,
)
from .verification import feedback_for, is_not_applicable, parse_citation

CHARS_PER_TOKEN = 4
TOKEN_COUNTER = f"chars_per_token:{CHARS_PER_TOKEN}"
# Kept free while choosing optional content, so the omission note always fits.
_NOTE_RESERVE_TOKENS = 120
_NOTE_ITEMS = 4  # omitted items named in the note; the rest are counted


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
    # None when the reply is not JSON, as for a planned code step.
    output_schema: Mapping[str, Any] | None
    exploration: Sequence[Observation] = ()
    steps: Sequence[Step] = ()
    plan: ObserveResult | None = None
    feedback: Sequence[ImprovementFeedback] = ()
    after_failure: bool = False
    tool_descriptions: Sequence[Mapping[str, Any]] = ()
    sandbox_names: Sequence[str] = ()
    tool_read_cells: int | None = None
    # For an answer's judge: each citation's cells as the harness read them.
    cited_cells: Mapping[str, Sequence[tuple[str, Any]]] = field(default_factory=dict)
    # For OBSERVE: why the last EXECUTE produced no valid action, if it did not.
    rejected_execute: str | None = None
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


def _span(rows: Sequence[int]) -> str:
    return f"{rows[0]}-{rows[-1]}" if rows[0] != rows[-1] else str(rows[0])


def _column(profile: Mapping[str, Any]) -> str:
    counts = profile.get("counts") or {}
    kinds = ", ".join(f"{count} {kind}" for kind, count in counts.items())
    parts = [f"{profile.get('filled', 0)} filled" + (f" ({kinds})" if kinds else "")]
    for key in ("numbers", "dates"):
        spread = profile.get(key)
        if spread:
            parts.append(f"{key} {spread['min']}..{spread['max']} in rows {_span(spread['rows'])}")
    formulas = int(profile.get("formulas") or 0)
    if formulas:
        parts.append(f"{formulas} formula" + ("s" if formulas > 1 else ""))
    text = profile.get("text") or []
    if text:
        cells = ", ".join(
            f"{coordinate} {json.dumps(value, ensure_ascii=False)}" for coordinate, value in text
        )
        parts.append(f"text {cells}")
    return f"    - {profile.get('column')}: " + "; ".join(parts)


def _window(window: Mapping[str, Any]) -> list[str]:
    rows = window.get("rows") or []
    if not rows:
        return []
    lines = ["    top-left cells, verbatim:", "      row | " + " | ".join(window["columns"])]
    for row, values in rows:
        shown = ["" if value is None else str(value) for value in values]
        lines.append(f"      {row} | " + " | ".join(shown))
    return lines


def _structure(structure: Mapping[str, Any]) -> list[str]:
    lines = _window(structure.get("window") or {})
    lines.append("    cells by column:")
    lines.extend(_column(profile) for profile in structure.get("columns") or [])
    more = int(structure.get("more_columns") or 0)
    if more:
        lines.append(f"    - ... {more} more columns, not profiled")
    for key, label in (("total_rows", "rows with total/sum text"), ("blank_rows", "blank rows")):
        rows = structure.get(key) or []
        if rows:
            lines.append(f"    {label}: {', '.join(str(row) for row in rows)}")
    return lines


_STRUCTURE_NOTE = (
    "The top-left cells are shown exactly as they are, blank rows skipped and long text "
    "cut; the column profile is computed from every cell of each used range and is exact. "
    "Neither says what anything means: decide which cells are labels and where each table "
    "starts from the cells themselves, reading more when unsure."
)


def _workbooks(workbooks: Sequence[Mapping[str, Any]]) -> str:
    lines = ["Workbooks:"]
    profiled = False
    for book in workbooks:
        lines.append(f"- {book.get('workbook_id')} ({book.get('file')})")
        for sheet in book.get("sheets") or []:
            merged = sheet.get("merged") or []
            extra = f", merged ranges: {', '.join(merged)}" if merged else ""
            lines.append(
                f"  - sheet {sheet.get('name')!r}, used range {sheet.get('dimensions')}{extra}"
            )
            if sheet.get("structure"):
                profiled = True
                lines.extend(_structure(sheet["structure"]))
    if profiled:
        lines.append(_STRUCTURE_NOTE)
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


def render_observation(observation: Observation) -> str:
    """The observation exactly as a later prompt shows it."""
    lines = [f"Result: {observation.status.value} - {observation.summary}"]
    if observation.error:
        lines.append(f"  error: {observation.error}")
    lines.extend(f"  - {fact}" for fact in observation.facts)
    lines.extend(f"  uncertain: {note}" for note in observation.uncertainties)
    if observation.next_question:
        lines.append(f"  next question: {observation.next_question}")
    return "\n".join(lines)


_DIGEST_CHARS = 200


def _step(step: Step, compact: bool) -> str:
    """A turn whole, or as its one-line digest: action, status, and what it came to."""
    observation = step.observation
    head = f"Turn {step.turn} ({step.execution_id}): {_action_line(step.action, compact)}"
    if compact:
        outcome = observation.error or next(
            (fact for fact in observation.facts if fact.startswith("result")),
            None,
        )
        if outcome is None:
            # A read's digest keeps the start of what it returned, so it is not re-read.
            read = " / ".join(observation.facts)
            outcome = f"{observation.summary}: {read}" if read else observation.summary
        if len(outcome) > _DIGEST_CHARS:
            outcome = outcome[: _DIGEST_CHARS - 1] + "…"
        return f"{head} -> {observation.status.value}: {outcome}"
    return f"{head}\n{render_observation(observation)}"


def _supports(step: Step, candidate: CandidateAnswer) -> bool:
    """Whether a turn produced evidence the answer cites: a cited range or computation."""
    cited_ids = {claim.computation_id for claim in candidate.claims if claim.computation_id}
    cited = [
        parsed
        for claim in candidate.claims
        for citation in claim.citations
        if (parsed := parse_citation(citation)) is not None
    ]
    for reference in step.observation.provenance:
        if reference.computation_id in cited_ids:
            return True
        sheet, range_ref = reference.sheet, reference.range_ref
        if sheet and range_ref and any(s == sheet and _overlaps(r, range_ref) for s, r in cited):
            return True
    return False


def _box(range_ref: str) -> tuple[int, int, int, int] | None:
    """(min_col, min_row, max_col, max_row), or None for an open or invalid range."""
    try:
        min_col, min_row, max_col, max_row = range_boundaries(range_ref.upper())
    except (ValueError, TypeError):
        return None
    if min_col is None or min_row is None or max_col is None or max_row is None:
        return None
    return min_col, min_row, max_col, max_row


def _overlaps(first: str, second: str) -> bool:
    a, b = _box(first), _box(second)
    if a is None or b is None:  # a whole row or column: treat as touching everything
        return True
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _plan(plan: ObserveResult) -> str:
    lines = ["Current plan:", f"Objective: {plan.execution_objective}"]
    if plan.known_facts:
        lines.append("Known facts:")
        lines.extend(f"- {fact}" for fact in plan.known_facts)
    if plan.uncertainties:
        lines.append("Uncertainties:")
        lines.extend(f"- {note}" for note in plan.uncertainties)
    lines.append(f"Next action: {plan.next_action.value}")
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


def _blocking_feedback(items: Sequence[ImprovementFeedback]) -> str:
    """Blocking items, the exact checks first: where advice conflicts, a check wins."""
    checks = [item for item in items if item.source is not FeedbackSource.LLM_JUDGE]
    judged = [item for item in items if item.source is FeedbackSource.LLM_JUDGE]
    parts = ["Blocking feedback (must be resolved before the answer can pass):"]
    if checks:
        parts.append(
            _feedback(
                checks,
                "Failed checks (exact; their repairs are right, and where a judge's "
                "recommendation conflicts with one, follow the check):",
            )
        )
    if judged:
        parts.append(_feedback(judged, "Judge findings:"))
    return "\n".join(parts)


def _report(report: DeterministicCheckReport) -> str:
    lines = [f"Deterministic checks: {report.status.value}"]
    lines.extend(f"- {c.id}: {c.status.value} - {c.reason}" for c in report.checks)
    return "\n".join(lines)


def _provenance(ids: Sequence[str], computations: Mapping[str, Any], title: str) -> str:
    """Where each computation under review came from.

    With large tables shown only as previews, this is how a judge checks coverage: a
    computation's sources and input count say exactly which cells it used.
    """
    lines = []
    for computation_id in dict.fromkeys(ids):
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
        condition = record.metadata.get("condition")
        where = (
            f" where value {condition.get('op')} {condition.get('value')!r} "
            f"({condition.get('source')})"
            if isinstance(condition, dict)
            else ""
        )
        lines.append(
            f"- {record.id}: {record.operation} over {origin}{where} -> {len(inputs)} inputs "
            f"({span}){blanks}; output {record.output!r}"
        )
    if not lines:
        return ""
    return title + "\n" + "\n".join(lines)


def _review_provenance(request: ContextRequest) -> str:
    """The computations a verification judges: the answer's cited ones, or the step's."""
    if request.phase is not Phase.VERIFY:
        return ""
    if request.candidate is not None:
        cited = [str(c.computation_id) for c in request.candidate.claims if c.computation_id]
        title = "Recorded computations the answer relies on:"
        return _provenance(cited, request.computations, title)
    return _provenance(
        list(request.computations), request.computations, "Recorded computations of this step:"
    )


_CITED_INLINE = 12
_CITED_ENDS = 3
_MAX_CITATIONS = 20


def _cited(cited: Mapping[str, Sequence[tuple[str, Any]]]) -> str:
    """Each citation's cells as the harness read them, both ends of a long range."""
    if not cited:
        return ""

    def cell(coordinate: str, value: Any) -> str:
        return f"{coordinate} = {json.dumps(value, ensure_ascii=False, default=str)}"

    lines = ["Cited cells, read by the harness from the workbook:"]
    for citation in list(cited)[:_MAX_CITATIONS]:
        cells = list(cited[citation])
        if len(cells) <= _CITED_INLINE:
            shown = ", ".join(cell(c, v) for c, v in cells)
            lines.append(f"- {citation}: {shown}")
            continue
        filled = sum(1 for _, value in cells if value is not None)
        head = ", ".join(cell(c, v) for c, v in cells[:_CITED_ENDS])
        tail = ", ".join(cell(c, v) for c, v in cells[-_CITED_ENDS:])
        lines.append(f"- {citation}: {len(cells)} cells, {filled} filled; {head}, …, {tail}")
    return "\n".join(lines)


def _candidate(candidate: CandidateAnswer) -> str:
    claims = [claim.model_dump(mode="json") for claim in candidate.claims]
    return (
        f"Candidate answer:\n{candidate.answer_text}\n"
        f"Claims:\n{json.dumps(claims, ensure_ascii=False, indent=1)}"
    )


def _capabilities(request: ContextRequest) -> str:
    if request.phase is Phase.EXECUTE:
        kind = request.plan.next_action.value if request.plan is not None else None
        if kind == "answer":
            return ""
        if kind == "tool":
            return format_tool_descriptions(request.tool_descriptions)
        bound = (
            f"Currently bound in the session: {', '.join(request.sandbox_names)}"
            if request.sandbox_names
            else "Nothing is bound in the session yet."
        )
        if kind == "code":
            return f"{SANDBOX_CAPABILITIES}\n\n{bound}"
        return "\n\n".join(
            [format_tool_descriptions(request.tool_descriptions), SANDBOX_CAPABILITIES, bound]
        )
    if request.phase is Phase.OBSERVE:
        names = ", ".join(str(item["name"]) for item in request.tool_descriptions)
        return (
            "Capabilities of the next step (chosen there, not here): read-only workbook "
            f"tools ({names or 'none'}), a persistent Python session with pandas and a "
            "read-only workbook object, and answering with grounded claims."
            + (
                f" A tool read returns at most {request.tool_read_cells} cells; a larger "
                "range must be a code step."
                if request.tool_read_cells is not None
                else ""
            )
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
        if request.plan is None:
            return EXECUTE_INSTRUCTIONS
        kind = request.plan.next_action.value
        return (
            f"{PLANNED_EXECUTE_INSTRUCTIONS[kind]}\n\n"
            f'The plan asks for one "{kind}" action; take exactly that action toward the '
            "objective."
        )
    if request.candidate is None:
        return STEP_VERIFICATION_INSTRUCTIONS
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
        "provenance": _review_provenance(request),
        # A step verification judges the latest step, so it is shown whole.
        "under_review": (
            "Step under review:\n" + _step(request.steps[-1], False)
            if request.phase is Phase.VERIFY and request.candidate is None and request.steps
            else ""
        ),
        "report": _report(request.report) if request.report else "",
        "cited": (
            _cited(request.cited_cells)
            if request.phase is Phase.VERIFY and request.candidate
            else ""
        ),
        "rejected": (
            "The last EXECUTE produced no valid action, so nothing ran: "
            f"{request.rejected_execute}\nPlan in light of it -- for example, if it tried "
            "another kind of action than planned, consider whether that kind is the right "
            "next step."
            if request.phase is Phase.OBSERVE and request.rejected_execute
            else ""
        ),
        "capabilities": _capabilities(request),
        "instructions": _instructions(request),
        "schema": (
            output_schema(request.output_schema) if request.output_schema is not None else ""
        ),
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
        blocking_text = _blocking_feedback(blocking)
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
    if reserved["under_review"]:
        steps = steps[:-1]
    latest = (
        choose(_step(steps[-1], False), _step(steps[-1], True), f"turn {steps[-1].turn}")
        if steps
        else ""
    )

    # 4. Earlier turns, newest first, as one-line digests: the prompt follows the step
    #    at hand, and the plan's known facts carry what earlier turns established. An
    #    answer's judge sees the turns behind its evidence whole.
    earlier: dict[int, str] = {}
    candidate = request.candidate if request.phase is Phase.VERIFY else None
    for step in reversed(steps[:-1]):
        if candidate is not None and _supports(step, candidate):
            earlier[step.turn] = choose(_step(step, False), _step(step, True), f"turn {step.turn}")
        else:
            earlier[step.turn] = choose(_step(step, True), "", f"turn {step.turn}")
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
        # A bounded list, so the note always fits the reserve kept for it.
        listed = "; ".join(omitted[:_NOTE_ITEMS])
        more = len(omitted) - _NOTE_ITEMS
        note = (
            "Note: to fit the context budget these were shortened or omitted: "
            + listed
            + (f"; and {more} more" if more > 0 else "")
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
        reserved["rejected"],
        reserved["under_review"],
        blocking_text,
        advisory_text,
        reserved["report"],
        reserved["candidate"],
        reserved["provenance"],
        reserved["cited"],
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
