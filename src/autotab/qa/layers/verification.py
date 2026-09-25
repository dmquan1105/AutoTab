"""Deterministic checks, judge normalization, and status combination.

Pure: no model calls and no lifecycle state. Action scope checks one execution
result; answer scope checks a candidate against everything recorded in the run.
See ``specs/layers/verification.md``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from statistics import fmean
from typing import Any

from ..schemas import (
    LLM_JUDGE_RECHECK,
    CandidateAnswer,
    ComputationRecord,
    DeterministicCheckReport,
    DeterministicCheckResult,
    FeedbackPriority,
    FeedbackSource,
    ImprovementFeedback,
    LLMJudgeResult,
    ResultState,
    RunStatus,
    VerificationResult,
)

ACTION_CHECKS = ("CALC.ARITHMETIC", "CALC.INPUTS_DISTINCT", "RUNTIME.RESULT_COMPLETE")
ANSWER_CHECKS = (
    "ANSWER.REQUIRED_PARTS",
    "ANSWER.CITATION_FORMAT",
    "ANSWER.CITATION_EXISTS",
    "ANSWER.PROVENANCE",
    "CALC.ARITHMETIC",
    "CALC.INPUTS_DISTINCT",
    "RUNTIME.RESULT_COMPLETE",
    "RUNTIME.SOURCE_IMMUTABLE",
)

CitationResolver = Callable[[str, str], bool]

_CITATION = re.compile(
    r"^(?:'(?P<quoted>(?:[^']|'')+)'|(?P<plain>[^'!:]+))!"
    r"(?P<range>\$?[A-Za-z]{1,3}\$?\d+(?::\$?[A-Za-z]{1,3}\$?\d+)?)$"
)
_INCOMPLETE = frozenset({ResultState.ERROR, ResultState.TRUNCATED})
_RELATIVE_EPSILON = 1e-9

# How the next turn can repair each failed check. The verifier recommends; OBSERVE
# decides the objective and EXECUTE chooses the action.
_REPAIRS = {
    "ANSWER.REQUIRED_PARTS": "Answer again with at least one claim that carries a "
    "citation or a computation ID.",
    "ANSWER.CITATION_FORMAT": "Rewrite each citation as Sheet!A1 or Sheet!A1:B9, quoting "
    "sheet names that contain spaces.",
    "ANSWER.CITATION_EXISTS": "Cite only sheets and ranges that exist in the workbook; "
    "check the sheet name with an exact read.",
    # Never offer "cite cells instead": a model once took that to strip the computation
    # from a maximum, leaving a wrong aggregate that no check could replay.
    "ANSWER.PROVENANCE": "Compute the value in a code action, record it with "
    "record_computation, and reference the ID it returns; each claim's value must equal "
    "that computation's output. Do not drop a computation to get past this check.",
    "CALC.ARITHMETIC": "Recompute from the recorded inputs and record the corrected output.",
    "CALC.INPUTS_DISTINCT": "Remove repeated references so no cell is counted twice.",
    "RUNTIME.RESULT_COMPLETE": "Redo the failed or truncated read or computation before "
    "relying on it.",
    "RUNTIME.SOURCE_IMMUTABLE": "Stop: the source workbook changed during the run.",
}


def _check(
    check_id: str, passed: bool, observed: Any, expected: Any, reason: str
) -> DeterministicCheckResult:
    # observed/expected are typed Any here: DeterministicCheckResult validates them as
    # JSON values on construction, and list invariance makes JsonValue unassignable.
    return DeterministicCheckResult(
        id=check_id,
        status=RunStatus.PASS if passed else RunStatus.FAIL,
        observed=observed,
        expected=expected,
        reason=reason,
    )


NOT_APPLICABLE = "not applicable: "


def _not_applicable(check_id: str, why: str) -> DeterministicCheckResult:
    return _check(check_id, True, None, None, f"{NOT_APPLICABLE}{why}")


def is_not_applicable(check: DeterministicCheckResult) -> bool:
    """Whether a check passed only because it had nothing to inspect.

    Such a PASS says nothing about an earlier failure of the same check, so it must
    not resolve that failure's feedback.
    """
    return check.reason.startswith(NOT_APPLICABLE)


def _report(checks: list[DeterministicCheckResult]) -> DeterministicCheckReport:
    passed = all(check.status is RunStatus.PASS for check in checks)
    return DeterministicCheckReport(
        status=RunStatus.PASS if passed else RunStatus.FAIL, checks=checks
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _two(values: list[float], combine: Callable[[float, float], float]) -> float:
    if len(values) != 2:
        raise ValueError(f"needs exactly two inputs, got {len(values)}")
    return combine(values[0], values[1])


_NUMERIC_OPERATIONS: dict[str, Callable[[list[float]], float]] = {
    "sum": sum,
    "total": sum,
    "add": sum,
    "mean": fmean,
    "average": fmean,
    "avg": fmean,
    "min": min,
    "minimum": min,
    "max": max,
    "maximum": max,
    "product": math.prod,
    "multiply": math.prod,
    "difference": lambda v: _two(v, lambda a, b: a - b),
    "subtract": lambda v: _two(v, lambda a, b: a - b),
    "ratio": lambda v: _two(v, lambda a, b: a / b),
    "divide": lambda v: _two(v, lambda a, b: a / b),
}
# What record_computation accepts: exactly what CALC.ARITHMETIC can replay, so a
# recorded computation can never be one this check silently skips.
NUMERIC_OPERATIONS = frozenset(_NUMERIC_OPERATIONS)
# A row selection: which rows of one column meet a condition. Its output is the list of
# worksheet row numbers, and replay re-derives exactly that list from the inputs.
ROW_SELECTION = "rows_where"
CONDITIONS = ("equals", "not_equals", "gt", "ge", "lt", "le", "contains")
REPLAYABLE_OPERATIONS = NUMERIC_OPERATIONS | {"count", ROW_SELECTION}


_ORDER: dict[str, Callable[[Any, Any], bool]] = {
    "gt": lambda left, right: left > right,
    "ge": lambda left, right: left >= right,
    "lt": lambda left, right: left < right,
    "le": lambda left, right: left <= right,
}


def meets(value: Any, op: str, target: Any) -> bool:
    """Whether one cell value meets a row-selection condition.

    Numbers compare as numbers and text as text -- ISO dates, as wb.range returns dates,
    order correctly as text. ``contains`` is a case-insensitive substring test. A number
    and a text never meet an ordering, so a label in a numeric column is not selected.
    """
    if op == "contains":
        return isinstance(value, str) and str(target).casefold() in value.casefold()
    if _is_number(value) and _is_number(target):
        left, right = float(value), float(target)
        close = abs(left - right) <= _RELATIVE_EPSILON * max(1.0, abs(right))
        if op in ("equals", "not_equals"):
            return close if op == "equals" else not close
        return _ORDER[op](left, right)
    if op == "equals":
        return bool(value == target)
    if op == "not_equals":
        return bool(value != target)
    if isinstance(value, str) and isinstance(target, str):
        return _ORDER[op](value, target)
    return False


def _row_of(reference: str | None) -> int | None:
    parsed = parse_citation(reference) if reference else None
    if parsed is None:
        return None
    match = re.match(r"^[A-Z]+(\d+)$", parsed[1])
    return int(match.group(1)) if match else None


def _replay_selection(record: ComputationRecord) -> tuple[bool | None, str]:
    condition = record.metadata.get("condition")
    if not isinstance(condition, dict) or condition.get("op") not in CONDITIONS:
        return False, f"{record.id}: a row selection needs a recorded condition"
    op, target = str(condition["op"]), condition.get("value")
    rows = sorted(
        {
            row
            for item in record.inputs
            if meets(item.value, op, target) and (row := _row_of(item.reference)) is not None
        }
    )
    recorded = record.output if isinstance(record.output, list) else [record.output]
    numbers = [row for row in recorded if isinstance(row, int) and not isinstance(row, bool)]
    if len(numbers) != len(recorded):
        return False, f"{record.id}: a row selection's output must be worksheet row numbers"
    if rows == sorted(numbers):
        return True, f"{record.id}: rows where value {op} {target!r} replay to {rows}"
    return (
        False,
        f"{record.id}: rows where value {op} {target!r} replay to {rows}, recorded {recorded}",
    )


def _replay(record: ComputationRecord) -> tuple[bool | None, str]:
    """Return (passed, reason); passed is None when the record is not replayable."""
    operation = record.operation.strip().lower()
    if operation == ROW_SELECTION:
        return _replay_selection(record)
    if operation == "count":
        recomputed: float = len(record.inputs)
    elif operation in _NUMERIC_OPERATIONS:
        if not _is_number(record.output):
            return None, f"{record.id}: output {record.output!r} is not a number"
        values = []
        for item in record.inputs:
            if not _is_number(item.value):
                return False, f"{record.id}: input {item.reference} = {item.value!r} is not numeric"
            values.append(float(item.value))  # type: ignore[arg-type]
        if not values:
            return False, f"{record.id}: {operation} has no inputs"
        try:
            recomputed = _NUMERIC_OPERATIONS[operation](values)
        except (ValueError, ZeroDivisionError) as exc:
            return False, f"{record.id}: {operation} cannot be replayed ({exc})"
    else:
        return None, f"{record.id}: operation {record.operation!r} is not replayable"
    output = float(record.output)  # type: ignore[arg-type]
    allowed = record.tolerance + _RELATIVE_EPSILON * max(1.0, abs(output))
    if abs(recomputed - output) <= allowed:
        return True, f"{record.id}: {operation} replays to {recomputed:g}"
    return False, f"{record.id}: {operation} replays to {recomputed:g}, recorded {output:g}"


def _arithmetic(computations: Sequence[ComputationRecord]) -> DeterministicCheckResult:
    if not computations:
        return _not_applicable("CALC.ARITHMETIC", "no computation was recorded")
    outcomes = [_replay(record) for record in computations]
    failures = [reason for passed, reason in outcomes if passed is False]
    replayed = [reason for passed, reason in outcomes if passed is True]
    if failures:
        return _check("CALC.ARITHMETIC", False, failures, "recorded outputs recompute", failures[0])
    if not replayed:
        return _not_applicable("CALC.ARITHMETIC", "; ".join(r for _, r in outcomes))
    return _check(
        "CALC.ARITHMETIC", True, replayed, "recorded outputs recompute", "; ".join(replayed)
    )


def _distinct(computations: Sequence[ComputationRecord]) -> DeterministicCheckResult:
    if not computations:
        return _not_applicable("CALC.INPUTS_DISTINCT", "no computation was recorded")
    repeated = []
    for record in computations:
        seen: set[str] = set()
        for item in record.inputs:
            if item.reference is None:
                continue
            if item.reference in seen:
                repeated.append(f"{record.id}: {item.reference}")
            seen.add(item.reference)
    if repeated:
        return _check(
            "CALC.INPUTS_DISTINCT",
            False,
            repeated,
            "each reference counted once",
            f"references counted more than once: {', '.join(repeated)}",
        )
    return _check(
        "CALC.INPUTS_DISTINCT", True, None, "each reference counted once", "no reference repeats"
    )


def _complete(states: Mapping[str, ResultState]) -> DeterministicCheckResult:
    bad = sorted(label for label, state in states.items() if state in _INCOMPLETE)
    if bad:
        return _check(
            "RUNTIME.RESULT_COMPLETE",
            False,
            {label: states[label].value for label in bad},
            "success or empty",
            f"relied-upon executions did not complete: {', '.join(bad)}",
        )
    if not states:
        return _not_applicable("RUNTIME.RESULT_COMPLETE", "nothing relied on an execution")
    return _check("RUNTIME.RESULT_COMPLETE", True, None, "success or empty", "executions completed")


def run_action_checks(
    state: ResultState, computations: Sequence[ComputationRecord]
) -> DeterministicCheckReport:
    """Check one tool or code action right after it ran; no model call."""
    return _report(
        [
            _arithmetic(computations),
            _distinct(computations),
            _complete({"this action": state}),
        ]
    )


def parse_citation(citation: str) -> tuple[str, str] | None:
    """Return (sheet, A1 range) for ``Sheet!A1`` or ``'My Sheet'!A1:B9``, else None."""
    match = _CITATION.match(citation.strip())
    if match is None:
        return None
    quoted = match.group("quoted")
    sheet = quoted.replace("''", "'") if quoted is not None else match.group("plain").strip()
    return sheet, match.group("range").replace("$", "").upper()


def _values_match(claimed: Any, output: Any, tolerance: float) -> bool:
    # A lookup often returns the rows that match, so a one-item list means its item:
    # "Nga Anh Ngo" and ["Nga Anh Ngo"] state the same fact. Two items never do.
    if isinstance(output, list) and len(output) == 1:
        output = output[0]
    if isinstance(claimed, list) and len(claimed) == 1:
        claimed = claimed[0]
    if _is_number(claimed) and _is_number(output):
        allowed = tolerance + _RELATIVE_EPSILON * max(1.0, abs(float(output)))
        return abs(float(claimed) - float(output)) <= allowed
    return bool(claimed == output)


def run_answer_checks(
    candidate: CandidateAnswer,
    *,
    computations: Mapping[str, ComputationRecord],
    computation_states: Mapping[str, ResultState],
    resolve_citation: CitationResolver,
    source_hashes: Mapping[str, tuple[str, str]],
) -> DeterministicCheckReport:
    """Check a candidate answer against the evidence recorded in the run.

    Args:
        candidate: The answer under verification.
        computations: Every computation recorded in the run, by ID.
        computation_states: State of the execution that recorded each computation.
        resolve_citation: Whether (sheet, range) is readable in a workbook of the run.
        source_hashes: Workbook ID to (manifest hash, current hash).
    """
    claims = candidate.claims
    checks = [
        _check(
            "ANSWER.REQUIRED_PARTS",
            bool(claims),
            {"answer_text": bool(candidate.answer_text.strip()), "claims": len(claims)},
            "answer text and at least one grounded claim",
            (
                "answer text and grounded claims present"
                if claims
                else "the answer has no claims, so nothing in it is grounded"
            ),
        )
    ]

    citations = [citation for claim in claims for citation in claim.citations]
    parsed = {citation: parse_citation(citation) for citation in citations}
    malformed = [citation for citation, value in parsed.items() if value is None]
    checks.append(
        _not_applicable("ANSWER.CITATION_FORMAT", "the answer cites no range")
        if not citations
        else _check(
            "ANSWER.CITATION_FORMAT",
            not malformed,
            malformed or citations,
            "Sheet!A1 or Sheet!A1:B9",
            f"malformed citations: {', '.join(malformed)}" if malformed else "citations parse",
        )
    )

    well_formed = {c: v for c, v in parsed.items() if v is not None}
    missing = [c for c, (sheet, rng) in well_formed.items() if not resolve_citation(sheet, rng)]
    checks.append(
        _not_applicable("ANSWER.CITATION_EXISTS", "no well-formed citation to resolve")
        if not well_formed
        else _check(
            "ANSWER.CITATION_EXISTS",
            not missing,
            missing or list(well_formed),
            "every citation resolves to a readable range",
            (
                f"citations that do not resolve: {', '.join(missing)}"
                if missing
                else "citations resolve"
            ),
        )
    )

    # Two different faults with two different repairs, so they are kept apart.
    unknown_ids: list[str] = []
    mismatched: list[str] = []
    referenced = [claim for claim in claims if claim.computation_id is not None]
    for claim in referenced:
        record = computations.get(str(claim.computation_id))
        if record is None:
            unknown_ids.append(f"{claim.id}: computation {claim.computation_id} does not exist")
        elif claim.value is not None and not _values_match(
            claim.value, record.output, record.tolerance
        ):
            mismatched.append(
                f"{claim.id}: value {claim.value!r} differs from {record.id} output "
                f"{record.output!r}"
            )
    problems = unknown_ids + mismatched
    checks.append(
        _not_applicable("ANSWER.PROVENANCE", "no claim references a computation")
        if not referenced
        else _check(
            "ANSWER.PROVENANCE",
            not problems,
            (
                {"missing": unknown_ids, "mismatched": mismatched}
                if problems
                else [claim.id for claim in referenced]
            ),
            "computation IDs resolve and claim values equal their outputs",
            "; ".join(problems) if problems else "claim provenance is consistent",
        )
    )

    # Replay what the answer relies on. Every other record was already checked by
    # action scope when it was made; replaying all of them would let one early,
    # since-corrected slip fail every later answer.
    relied_ids = dict.fromkeys(
        str(claim.computation_id)
        for claim in referenced
        if str(claim.computation_id) in computations
    )
    relied_on = [computations[computation_id] for computation_id in relied_ids]
    checks.append(_arithmetic(relied_on))
    checks.append(_distinct(relied_on))
    relied = {
        str(claim.computation_id): computation_states[str(claim.computation_id)]
        for claim in referenced
        if str(claim.computation_id) in computation_states
    }
    checks.append(_complete(relied))

    changed = sorted(wid for wid, (before, after) in source_hashes.items() if before != after)
    checks.append(
        _check(
            "RUNTIME.SOURCE_IMMUTABLE",
            not changed,
            changed or sorted(source_hashes),
            "workbook hashes equal the manifest",
            (
                f"source workbooks changed: {', '.join(changed)}"
                if changed
                else "source workbooks unchanged"
            ),
        )
    )
    return _report(checks)


_MISSING_COMPUTATION = (
    "Reference only computation IDs that record_computation returned in this run; to use "
    "a value, record it first in a code action and cite the ID it returns. Do not drop a "
    "computation to get past this check."
)
_MISLINKED_CLAIM = (
    "Do not recompute: the computations exist, but a claim points at the wrong one. Give "
    "each computation_id only to the claim whose value is that computation's output. A "
    "looked-up label -- a name or category found in the rows a condition selects -- is a "
    "separate claim backed by citations of its exact cells, with computation_id null."
)


def _repair(check: DeterministicCheckResult) -> str:
    """How the next turn can fix this failure; provenance faults differ by cause."""
    if check.id == "ANSWER.PROVENANCE" and isinstance(check.observed, dict):
        parts = []
        if check.observed.get("missing"):
            parts.append(_MISSING_COMPUTATION)
        if check.observed.get("mismatched"):
            parts.append(_MISLINKED_CLAIM)
        if parts:
            return " ".join(parts)
    return _REPAIRS.get(check.id, "Correct the failed check and try again.")


def feedback_for(report: DeterministicCheckReport) -> list[ImprovementFeedback]:
    """Return one blocking item per failed check, rechecked by that same check."""
    return [
        ImprovementFeedback(
            code=check.id,
            source=FeedbackSource.DETERMINISTIC_CHECK,
            problem=check.reason,
            evidence=(
                [json.dumps(check.observed, ensure_ascii=False)]
                if check.observed is not None
                else []
            ),
            action=_repair(check),
            priority=FeedbackPriority.BLOCKING,
            recheck=check.id,
        )
        for check in report.checks
        if check.status is RunStatus.FAIL
    ]


def extract_json_object(raw: str) -> str:
    """Return the first complete JSON object in a model reply, tolerating fences and prose.

    Each ``{`` is tried in turn and the first that decodes as a whole object wins, so
    braces in surrounding prose or code no longer glue two fragments into one.

    Raises:
        ValueError: If the reply holds no JSON object.
    """
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            value, end = decoder.raw_decode(raw, match.start())
        except ValueError:
            continue
        if isinstance(value, dict):
            return raw[match.start() : end]
    raise ValueError("the reply contains no JSON object")


def normalize_llm_judge(raw: str) -> LLMJudgeResult:
    """Validate a raw judge reply as the compact holistic verdict.

    Raises:
        ValueError: If the reply holds no JSON object or it fails the contract;
            pydantic's ValidationError is a ValueError.
    """
    payload = json.loads(extract_json_object(raw))
    # A judge's feedback is sourced from the judge and rechecked by the judge, by
    # definition; those two fields are filled in, not left for the model to get wrong.
    items = payload.get("improvement_feedback") if isinstance(payload, dict) else None
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                item["source"] = FeedbackSource.LLM_JUDGE.value
                item["recheck"] = LLM_JUDGE_RECHECK
    return LLMJudgeResult.model_validate_json(json.dumps(payload))


def _verifier_failure(error: str) -> ImprovementFeedback:
    return ImprovementFeedback(
        code="VERIFIER.INVALID_RESPONSE",
        source=FeedbackSource.LLM_JUDGE,
        problem=f"The verifier did not return a valid judgement: {error}",
        action="Answer again with explicit claims so the verifier can judge them.",
        priority=FeedbackPriority.BLOCKING,
        recheck=LLM_JUDGE_RECHECK,
    )


def combine_checks(report: DeterministicCheckReport, *, artifact: str) -> VerificationResult:
    """The verdict of a step that is checked but not judged: its checks decide alone."""
    failed = [check for check in report.checks if check.status is RunStatus.FAIL]
    return VerificationResult(
        status=RunStatus.FAIL if failed else RunStatus.PASS,
        reason=(
            f"{failed[0].id}: {failed[0].reason}"
            if failed
            else "The read came back complete; a read is checked, not judged."
        ),
        artifact=artifact,
        improvement_feedback=feedback_for(report),
    )


def combine(
    report: DeterministicCheckReport,
    judge: LLMJudgeResult | None,
    *,
    artifact: str,
    verifier_error: str | None = None,
) -> VerificationResult:
    """Combine both sides: PASS only when every check and the judge pass."""
    deterministic = feedback_for(report)
    failed = [check for check in report.checks if check.status is RunStatus.FAIL]
    if judge is None:
        error = verifier_error or "no judgement was produced"
        return VerificationResult(
            status=RunStatus.FAIL,
            reason=f"The verifier returned no valid judgement: {error}",
            artifact=artifact,
            improvement_feedback=[*deterministic, _verifier_failure(error)],
        )
    passed = not failed and judge.status is RunStatus.PASS
    if failed:
        reason = f"{failed[0].id}: {failed[0].reason}"
    else:
        reason = judge.reason
    judge_feedback = [
        item
        for item in judge.improvement_feedback
        if not passed or item.priority is FeedbackPriority.NON_BLOCKING
    ]
    return VerificationResult(
        status=RunStatus.PASS if passed else RunStatus.FAIL,
        reason=reason,
        artifact=artifact,
        improvement_feedback=[*deterministic, *judge_feedback],
    )
