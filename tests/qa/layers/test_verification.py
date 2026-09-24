"""Deterministic checks, judge normalization, and combination; see verification.md."""

import json
from typing import Any

import pytest

from autotab.qa.layers.verification import (
    ACTION_CHECKS,
    ANSWER_CHECKS,
    combine,
    feedback_for,
    normalize_llm_judge,
    run_action_checks,
    run_answer_checks,
)
from autotab.qa.schemas import (
    CandidateAnswer,
    ComputationRecord,
    DeterministicCheckReport,
    FeedbackPriority,
    FeedbackSource,
    ResultState,
    RunStatus,
)


def _computation(
    operation: str = "max",
    values: list[Any] | None = None,
    output: Any = 99,
    computation_id: str = "calc_1",
    references: list[str | None] | None = None,
) -> ComputationRecord:
    values = [92, 60, 99] if values is None else values
    references = references or [f"People!G{i + 3}" for i in range(len(values))]
    return ComputationRecord.model_validate(
        {
            "id": computation_id,
            "operation": operation,
            "inputs": [{"reference": r, "value": v} for r, v in zip(references, values)],
            "output": output,
        }
    )


def _candidate(**claim: Any) -> CandidateAnswer:
    base = {
        "id": "claim_1",
        "statement": "The highest score is 99.",
        "value": 99,
        "citations": ["People!G3:G22"],
        "computation_id": "calc_1",
    }
    return CandidateAnswer.model_validate(
        {"answer_text": "The highest score is 99.", "claims": [{**base, **claim}]}
    )


def _status(report: DeterministicCheckReport, check_id: str) -> RunStatus:
    return next(check.status for check in report.checks if check.id == check_id)


def _answer_report(candidate: CandidateAnswer, **overrides: Any) -> DeterministicCheckReport:
    arguments: dict[str, Any] = {
        "computations": {"calc_1": _computation()},
        "computation_states": {"calc_1": ResultState.SUCCESS},
        "resolve_citation": lambda sheet, rng: sheet == "People",
        "source_hashes": {"workbook_1": ("abc", "abc")},
    }
    arguments.update(overrides)
    return run_answer_checks(candidate, **arguments)


# --- action scope -----------------------------------------------------------------


def test_action_scope_runs_exactly_its_three_checks() -> None:
    report = run_action_checks(ResultState.SUCCESS, [_computation()])

    assert [check.id for check in report.checks] == list(ACTION_CHECKS)
    assert report.status is RunStatus.PASS


@pytest.mark.parametrize(
    ("operation", "values", "output"),
    [
        ("sum", [50, 40], 90),
        ("mean", [1, 2, 3], 2),
        ("average", [2, 4], 3.0),
        ("min", [5, 3], 3),
        ("max", [92, 99], 99),
        ("count", ["a", "b", None], 3),
        ("product", [2, 3], 6),
        ("difference", [10, 4], 6),
        ("ratio", [1, 4], 0.25),
    ],
)
def test_arithmetic_replays_supported_operations(
    operation: str, values: list[Any], output: Any
) -> None:
    report = run_action_checks(ResultState.SUCCESS, [_computation(operation, values, output)])

    assert _status(report, "CALC.ARITHMETIC") is RunStatus.PASS


def test_arithmetic_catches_a_wrong_recorded_output() -> None:
    report = run_action_checks(ResultState.SUCCESS, [_computation("sum", [50, 40], 95)])

    assert _status(report, "CALC.ARITHMETIC") is RunStatus.FAIL
    assert report.status is RunStatus.FAIL


def test_arithmetic_is_not_applicable_without_computations_or_for_opaque_operations() -> None:
    empty = run_action_checks(ResultState.SUCCESS, [])
    opaque = run_action_checks(
        ResultState.SUCCESS, [_computation("groupby_sum", [1, 2], {"North": 3})]
    )

    for report in (empty, opaque):
        check = next(c for c in report.checks if c.id == "CALC.ARITHMETIC")
        assert check.status is RunStatus.PASS
        assert "not applicable" in check.reason


def test_non_numeric_input_to_a_numeric_operation_fails() -> None:
    report = run_action_checks(ResultState.SUCCESS, [_computation("sum", [50, "forty"], 90)])

    assert _status(report, "CALC.ARITHMETIC") is RunStatus.FAIL


def test_duplicate_input_references_fail_distinctness() -> None:
    duplicated = _computation("sum", [50, 50], 100, references=["People!G3", "People!G3"])

    report = run_action_checks(ResultState.SUCCESS, [duplicated])

    assert _status(report, "CALC.INPUTS_DISTINCT") is RunStatus.FAIL


@pytest.mark.parametrize("state", [ResultState.ERROR, ResultState.TRUNCATED])
def test_incomplete_execution_fails_result_completeness(state: ResultState) -> None:
    report = run_action_checks(state, [])

    assert _status(report, "RUNTIME.RESULT_COMPLETE") is RunStatus.FAIL


# --- answer scope -----------------------------------------------------------------


def test_answer_scope_runs_all_eight_checks_and_passes_a_grounded_answer() -> None:
    report = _answer_report(_candidate())

    assert [check.id for check in report.checks] == list(ANSWER_CHECKS)
    assert report.status is RunStatus.PASS


def test_an_answer_without_claims_is_ungrounded() -> None:
    candidate = CandidateAnswer.model_validate({"answer_text": "It is 99.", "claims": []})

    assert _status(_answer_report(candidate), "ANSWER.REQUIRED_PARTS") is RunStatus.FAIL


@pytest.mark.parametrize("citation", ["G3:G22", "People!", "People!G3:", "People G3"])
def test_malformed_citations_fail_format(citation: str) -> None:
    report = _answer_report(_candidate(citations=[citation]))

    assert _status(report, "ANSWER.CITATION_FORMAT") is RunStatus.FAIL


def test_quoted_sheet_names_are_valid_citations() -> None:
    report = _answer_report(
        _candidate(citations=["'Doanh thu quý 1'!A1"]),
        resolve_citation=lambda sheet, rng: sheet == "Doanh thu quý 1",
    )

    assert _status(report, "ANSWER.CITATION_FORMAT") is RunStatus.PASS
    assert _status(report, "ANSWER.CITATION_EXISTS") is RunStatus.PASS


def test_unresolvable_citations_fail_existence() -> None:
    report = _answer_report(_candidate(citations=["Peple!G3"]))

    assert _status(report, "ANSWER.CITATION_EXISTS") is RunStatus.FAIL


def test_provenance_fails_on_unknown_computation_or_value_mismatch() -> None:
    unknown = _answer_report(_candidate(computation_id="calc_9"))
    mismatch = _answer_report(_candidate(value=98))

    assert _status(unknown, "ANSWER.PROVENANCE") is RunStatus.FAIL
    assert _status(mismatch, "ANSWER.PROVENANCE") is RunStatus.FAIL


def test_claims_resting_on_a_failed_execution_fail_completeness() -> None:
    report = _answer_report(_candidate(), computation_states={"calc_1": ResultState.ERROR})

    assert _status(report, "RUNTIME.RESULT_COMPLETE") is RunStatus.FAIL


def test_a_changed_source_workbook_fails_immutability() -> None:
    report = _answer_report(_candidate(), source_hashes={"workbook_1": ("abc", "xyz")})

    assert _status(report, "RUNTIME.SOURCE_IMMUTABLE") is RunStatus.FAIL


# --- feedback, judge, combination --------------------------------------------------


def test_each_failed_check_yields_one_blocking_item_that_rechecks_itself() -> None:
    report = _answer_report(_candidate(value=98, citations=["Peple!G3"]))

    feedback = feedback_for(report)

    failed = [check.id for check in report.checks if check.status is RunStatus.FAIL]
    assert [item.recheck for item in feedback] == failed
    assert all(item.source is FeedbackSource.DETERMINISTIC_CHECK for item in feedback)
    assert all(item.priority is FeedbackPriority.BLOCKING for item in feedback)
    assert all(item.action for item in feedback)


def _judge(status: str = "PASS") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "rubric_version": "1.0",
        "status": status,
        "confidence_score": 0.8,
        "reason": "The answer names the top scorer from the Score column.",
        "issues_found": [],
        "improvement_feedback": [],
        "final_assessment": "Grounded.",
    }
    if status == "FAIL":
        payload["reason"] = "No gender column exists, so 'the man' was not verified."
        payload["issues_found"] = ["The 'man' constraint is asserted without evidence."]
        payload["improvement_feedback"] = [
            {
                "code": "DATA_HANDLING.UNSUPPORTED_CONSTRAINT",
                "source": "llm_judge",
                "problem": "Gender is not recorded in the workbook.",
                "evidence": ["People!A1:G2"],
                "action": "State that gender cannot be verified from the workbook.",
                "priority": "blocking",
                "recheck": "LLM_JUDGE",
            }
        ]
    return payload


@pytest.mark.parametrize(
    "wrap",
    [
        lambda s: s,
        lambda s: f"```json\n{s}\n```",
        lambda s: f"Here is my verdict:\n{s}\nThanks.",
    ],
)
def test_judge_output_is_parsed_through_common_wrappers(wrap: Any) -> None:
    result = normalize_llm_judge(wrap(json.dumps(_judge())))

    assert result.status is RunStatus.PASS


@pytest.mark.parametrize("raw", ["", "no json here", '{"status": "PASS"}'])
def test_invalid_judge_output_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_llm_judge(raw)


def test_combination_passes_only_when_both_sides_pass() -> None:
    passing = _answer_report(_candidate())
    failing = _answer_report(_candidate(value=98))
    judge_pass = normalize_llm_judge(json.dumps(_judge("PASS")))
    judge_fail = normalize_llm_judge(json.dumps(_judge("FAIL")))

    assert combine(passing, judge_pass, artifact="verification/1.json").status is RunStatus.PASS
    assert combine(failing, judge_pass, artifact="verification/1.json").status is RunStatus.FAIL
    assert combine(passing, judge_fail, artifact="verification/1.json").status is RunStatus.FAIL


def test_combined_failure_names_the_decisive_problem_and_carries_all_feedback() -> None:
    failing = _answer_report(_candidate(value=98))
    judge_fail = normalize_llm_judge(json.dumps(_judge("FAIL")))

    result = combine(failing, judge_fail, artifact="verification/2.json")

    assert "ANSWER.PROVENANCE" in result.reason
    sources = {item.source for item in result.improvement_feedback}
    assert sources == {FeedbackSource.DETERMINISTIC_CHECK, FeedbackSource.LLM_JUDGE}


def test_a_missing_judge_can_never_pass() -> None:
    result = combine(
        _answer_report(_candidate()),
        None,
        artifact="verification/3.json",
        verifier_error="judge output was not JSON",
    )

    assert result.status is RunStatus.FAIL
    assert "judge output was not JSON" in result.reason
    assert result.improvement_feedback


def test_an_unreferenced_wrong_computation_does_not_poison_every_later_answer() -> None:
    # A mistake recorded early in the run is caught by action scope when it happens.
    # If answer scope replayed every record, one early slip would fail every future
    # answer even after the agent recomputed correctly.
    stale = _computation("sum", [50, 40], 95, computation_id="calc_0")

    report = _answer_report(
        _candidate(),
        computations={"calc_0": stale, "calc_1": _computation()},
        computation_states={"calc_0": ResultState.SUCCESS, "calc_1": ResultState.SUCCESS},
    )

    assert _status(report, "CALC.ARITHMETIC") is RunStatus.PASS


def test_a_referenced_wrong_computation_still_fails_the_answer() -> None:
    wrong = _computation("max", [92, 60, 99], 98)

    report = _answer_report(_candidate(value=98), computations={"calc_1": wrong})

    assert _status(report, "CALC.ARITHMETIC") is RunStatus.FAIL


def test_the_provenance_repair_never_suggests_dropping_the_computation() -> None:
    # A first real run showed the model following "cite cells instead of a
    # computation" to strip the computation from a maximum, leaving an unchecked,
    # wrong aggregate. The repair must point at recording the calculation instead.
    report = _answer_report(_candidate(computation_id="calc_9"))

    action = next(i.action for i in feedback_for(report) if i.recheck == "ANSWER.PROVENANCE")

    assert "record_computation" in action
    assert "instead of a computation" not in action


def _provenance_repair(report: DeterministicCheckReport) -> str:
    return next(i.action for i in feedback_for(report) if i.recheck == "ANSWER.PROVENANCE")


def test_a_missing_computation_id_asks_for_an_id_that_was_returned() -> None:
    action = _provenance_repair(_answer_report(_candidate(computation_id="calc_9")))

    assert "returned" in action and "record_computation" in action


def test_a_mislinked_claim_is_told_to_fix_the_link_not_to_recompute() -> None:
    # A real run spent four turns recomputing because the repair always said "compute
    # it again", while the claim holding a name was simply linked to the max.
    action = _provenance_repair(_answer_report(_candidate(value="Nga Anh Ngo")))

    assert "not recompute" in action
    assert "citations" in action and "computation_id null" in action


def test_every_provenance_problem_is_reported_at_once() -> None:
    candidate = CandidateAnswer.model_validate(
        {
            "answer_text": "a",
            "claims": [
                {
                    "id": "c1",
                    "statement": "s",
                    "value": 1,
                    "citations": ["People!G3"],
                    "computation_id": "calc_9",
                },
                {
                    "id": "c2",
                    "statement": "s",
                    "value": "x",
                    "citations": ["People!G3"],
                    "computation_id": "calc_1",
                },
            ],
        }
    )

    check = next(c for c in _answer_report(candidate).checks if c.id == "ANSWER.PROVENANCE")

    assert "c1" in check.reason and "c2" in check.reason
    assert check.observed == {
        "missing": ["c1: computation calc_9 does not exist"],
        "mismatched": ["c2: value 'x' differs from calc_1 output 99"],
    }


@pytest.mark.parametrize(
    ("claimed", "output", "matches"),
    [
        ("Nga Anh Ngo", ["Nga Anh Ngo"], True),
        (["Nga Anh Ngo"], "Nga Anh Ngo", True),
        ("A", ["A", "B"], False),
        (99, [99.0], True),
    ],
)
def test_a_one_item_list_equals_its_item(claimed: Any, output: Any, matches: bool) -> None:
    record = _computation("max", [99], 99).model_copy(update={"output": output})

    report = _answer_report(_candidate(value=claimed), computations={"calc_1": record})

    assert (_status(report, "ANSWER.PROVENANCE") is RunStatus.PASS) is matches
