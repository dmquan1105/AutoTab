"""Strict contract tests for ``autotab.qa.schemas``.

The rubric is prompt-only: a judge response carries a holistic verdict, never
per-question or per-criterion results. See ``specs/schemas.md``.
"""

import json
from datetime import UTC, datetime
from math import inf, nan
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from autotab.qa.schemas import (
    ActionType,
    AgentAction,
    AnswerAction,
    CandidateAnswer,
    Claim,
    CodeAction,
    ComputationRecord,
    DeterministicCheckReport,
    DeterministicCheckResult,
    FeedbackPriority,
    FeedbackSource,
    HistoryEvent,
    ImprovementFeedback,
    LLMJudgeResult,
    Observation,
    ObservationSource,
    ObserveResult,
    Phase,
    ProvenanceReference,
    ResultState,
    RunStatus,
    SandboxResult,
    StrictModel,
    ToolAction,
    ToolResult,
    VerificationResult,
)


def _validate_json(model: Any, payload: dict[str, Any]) -> Any:
    return model.model_validate_json(json.dumps(payload))


def _workbook_provenance() -> dict[str, Any]:
    return {
        "kind": "workbook",
        "workbook_id": "workbook_1",
        "sheet": "Revenue",
        "range_ref": "B2:B4",
        "value": [50, 40, 60],
        "formula": None,
        "computation_id": None,
        "artifact_path": None,
    }


def _candidate_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "answer_text": "Total revenue is 150 million VND.",
        "claims": [
            {
                "id": "claim_1",
                "statement": "Total revenue is 150 million VND.",
                "value": 150,
                "citations": ["Revenue!B2:B4"],
                "computation_id": "calc_3",
            }
        ],
    }


def _feedback_payload(
    *,
    source: str = "llm_judge",
    priority: str = "blocking",
    recheck: str = "LLM_JUDGE",
) -> dict[str, Any]:
    return {
        "code": "DATA_HANDLING.SCOPE_ERROR",
        "source": source,
        "problem": "The selected range belongs to Net Sales.",
        "evidence": ["Summary!A3:B8"],
        "action": "Inspect and compare the Revenue and Net Sales header ranges.",
        "priority": priority,
        "recheck": recheck,
    }


def _judge_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "rubric_version": "1.0",
        "status": "PASS",
        "confidence_score": 0.9,
        "reason": "The cited range matches the requested metric.",
        "issues_found": [],
        "improvement_feedback": [],
        "final_assessment": "The answer is grounded in the cited range.",
    }


def _failing_judge_payload() -> dict[str, Any]:
    return {
        **_judge_payload(),
        "status": "FAIL",
        "reason": "The selected table does not represent the requested metric.",
        "issues_found": ["The selected range belongs to Net Sales rather than Revenue."],
        "improvement_feedback": [_feedback_payload()],
        "final_assessment": "The answer is not validated until the scope is corrected.",
    }


def _history_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "event_id": "event_1",
        "run_id": "run_1",
        "turn_id": 0,
        "phase": "INITIALIZE",
        "timestamp": "2026-09-16T08:30:00Z",
        "usage": {},
        "provenance": [],
        "artifact_paths": [],
    }


def test_exact_enum_vocabularies() -> None:
    assert [item.value for item in RunStatus] == ["PASS", "FAIL"]
    assert [item.value for item in Phase] == [
        "INITIALIZE",
        "OBSERVE",
        "EXECUTE",
        "VERIFY",
        "FINALIZE",
        "FAIL",
    ]
    assert [item.value for item in ActionType] == ["tool", "code", "answer"]
    assert [item.value for item in ResultState] == ["success", "empty", "error", "truncated"]
    assert [item.value for item in ObservationSource] == ["tool", "sandbox", "exploration"]
    assert [item.value for item in FeedbackPriority] == ["blocking", "non_blocking"]
    assert [item.value for item in FeedbackSource] == ["deterministic_check", "llm_judge"]


def test_strict_base_configuration() -> None:
    assert StrictModel.model_config["extra"] == "forbid"
    assert StrictModel.model_config["strict"] is True
    assert StrictModel.model_config["allow_inf_nan"] is False


def test_strict_python_validation_rejects_coercion_and_non_json_values() -> None:
    with pytest.raises(ValidationError):
        ObserveResult.model_validate(
            {
                "known_facts": ("fact",),
                "uncertainties": [],
                "execution_objective": "Inspect values.",
                "rationale": "Values are missing.",
            }
        )
    with pytest.raises(ValidationError):
        HistoryEvent.model_validate({**_history_payload(), "turn_id": "1"})
    with pytest.raises(ValidationError):
        CandidateAnswer.model_validate(
            {
                "schema_version": "1.0",
                "answer_text": "Invalid number.",
                "claims": [
                    {
                        "id": "claim_1",
                        "statement": "The value is invalid.",
                        "value": nan,
                        "citations": ["Revenue!B2"],
                        "computation_id": None,
                    }
                ],
            }
        )
    with pytest.raises(ValidationError):
        ComputationRecord.model_validate(
            {
                "schema_version": "1.0",
                "id": "calc_1",
                "operation": "sum",
                "inputs": [],
                "output": inf,
                "metadata": {},
                "tolerance": 0.0,
            }
        )


def test_history_event_accepts_enum_and_datetime_instances() -> None:
    event = HistoryEvent.model_validate(
        {
            **_history_payload(),
            "phase": Phase.OBSERVE,
            "timestamp": datetime.now(UTC),
        }
    )

    assert event.phase is Phase.OBSERVE


def test_float_only_history_latency_rejects_integer_json_values() -> None:
    with pytest.raises(ValidationError):
        _validate_json(HistoryEvent, {**_history_payload(), "latency_ms": 1})

    assert _validate_json(HistoryEvent, {**_history_payload(), "latency_ms": 1.0}).latency_ms == 1.0


def test_history_event_omits_unset_optional_fields_on_dump() -> None:
    event = _validate_json(HistoryEvent, _history_payload())

    dumped = event.model_dump(mode="json", exclude_none=True)

    assert dumped == _history_payload()
    assert "action" not in dumped
    assert "normalized_observation" not in dumped


def test_history_event_requires_an_action_for_execute_events() -> None:
    with pytest.raises(ValidationError):
        _validate_json(HistoryEvent, {**_history_payload(), "phase": "EXECUTE"})

    event = _validate_json(
        HistoryEvent,
        {
            **_history_payload(),
            "phase": "EXECUTE",
            "action": {
                "action_type": "tool",
                "rationale": "Inspect exact values.",
                "tool_name": "inspect_range",
                "arguments": {"workbook_id": "workbook_1", "range_ref": "B2:B4"},
            },
        },
    )

    assert isinstance(event.action, ToolAction)


def test_an_execute_turn_without_a_valid_action_is_recorded_with_its_error() -> None:
    # The model can fail to produce a valid action even after the re-ask; that turn
    # still happened and consumed budget, so it must be recordable.
    event = _validate_json(
        HistoryEvent,
        {**_history_payload(), "phase": "EXECUTE", "error": "the reply contains no JSON object"},
    )

    assert event.action is None


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../outside.json", "outputs/../../escape.json"],
)
def test_history_event_rejects_absolute_and_traversing_artifact_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        _validate_json(HistoryEvent, {**_history_payload(), "artifact_paths": [path]})


def test_observe_result_is_exact() -> None:
    payload = {
        "known_facts": ["Revenue is on the Revenue sheet."],
        "uncertainties": ["The exact range is unknown."],
        "execution_objective": "Confirm values under the selected header.",
        "rationale": "The range must be verified.",
    }

    assert _validate_json(ObserveResult, payload).model_dump(mode="json") == payload

    with pytest.raises(ValidationError):
        _validate_json(ObserveResult, {**payload, "tool_name": "inspect_range"})


def test_agent_action_discriminates_all_three_types() -> None:
    adapter: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)
    schema = adapter.json_schema()

    assert schema["discriminator"]["propertyName"] == "action_type"
    assert set(schema["discriminator"]["mapping"]) == {"tool", "code", "answer"}

    tool_payload: dict[str, Any] = {
        "action_type": "tool",
        "rationale": "Inspect exact values.",
        "tool_name": "future_registered_tool",
        "arguments": {"workbook_id": "workbook_1", "custom": [1, 2, 3]},
    }
    code_payload: dict[str, Any] = {
        "action_type": "code",
        "rationale": "Sum the retrieved values.",
        "code": "print(sum([50, 40, 60]))",
    }
    answer_payload: dict[str, Any] = {
        "action_type": "answer",
        "rationale": "Every requested figure is grounded.",
        "answer": _candidate_payload(),
    }

    tool = adapter.validate_json(json.dumps(tool_payload))
    code = adapter.validate_json(json.dumps(code_payload))
    answer = adapter.validate_json(json.dumps(answer_payload))

    assert isinstance(tool, ToolAction)
    assert isinstance(code, CodeAction)
    assert isinstance(answer, AnswerAction)
    assert isinstance(answer.answer, CandidateAnswer)
    for action, payload in ((tool, tool_payload), (code, code_payload), (answer, answer_payload)):
        assert adapter.dump_python(action, mode="json") == payload


@pytest.mark.parametrize(
    "payload",
    [
        {"action_type": "tool", "rationale": "Missing name.", "tool_name": "", "arguments": {}},
        {
            "action_type": "tool",
            "rationale": "Code is not a tool field.",
            "tool_name": "inspect_range",
            "arguments": {},
            "code": "print(1)",
        },
        {"action_type": "code", "rationale": "Empty code.", "code": ""},
        {
            "action_type": "code",
            "rationale": "Arguments are not a code field.",
            "code": "print(1)",
            "arguments": {"workbook_id": "workbook_1"},
        },
        {"action_type": "answer", "rationale": "Missing answer."},
        {
            "action_type": "answer",
            "rationale": "An answer runs nothing.",
            "answer": _candidate_payload(),
            "code": "print(1)",
        },
        {"action_type": "finalize", "rationale": "Unknown type.", "code": "print(1)"},
    ],
)
def test_agent_action_rejects_invalid_combinations(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(AgentAction).validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "payload",
    [
        _workbook_provenance(),
        {**_workbook_provenance(), "sheet": None, "range_ref": None},
        {**_workbook_provenance(), "formula": "=SUM(B2:B4)"},
        {
            "kind": "computation",
            "workbook_id": None,
            "sheet": None,
            "range_ref": None,
            "value": 150,
            "formula": None,
            "computation_id": "calc_3",
            "artifact_path": None,
        },
        {
            "kind": "artifact",
            "workbook_id": None,
            "sheet": None,
            "range_ref": None,
            "value": None,
            "formula": None,
            "computation_id": None,
            "artifact_path": "actions/exec_1.json",
        },
    ],
)
def test_provenance_accepts_each_valid_kind(payload: dict[str, Any]) -> None:
    assert _validate_json(ProvenanceReference, payload).model_dump(mode="json") == payload


@pytest.mark.parametrize(
    "changes",
    [
        {"workbook_id": None},
        {"sheet": None},
        {"range_ref": None},
        {"computation_id": "calc_3"},
        {"artifact_path": "actions/exec_1.json"},
        {"kind": "sheet"},
    ],
)
def test_workbook_provenance_rejects_invalid_combinations(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _validate_json(ProvenanceReference, {**_workbook_provenance(), **changes})


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "computation", "computation_id": None},
        {"kind": "computation", "computation_id": "calc_3", "sheet": "Revenue"},
        {"kind": "artifact", "artifact_path": None},
        {"kind": "artifact", "artifact_path": "/tmp/escape.json"},
        {"kind": "artifact", "artifact_path": "../escape.json"},
        {"kind": "computation", "computation_id": "calc_3", "formula": "=SUM(B2:B4)"},
    ],
)
def test_non_workbook_provenance_rejects_forbidden_fields(changes: dict[str, Any]) -> None:
    payload = {**_workbook_provenance(), "workbook_id": None, "sheet": None, "range_ref": None}
    with pytest.raises(ValidationError):
        _validate_json(ProvenanceReference, {**payload, **changes})


def test_tool_result_enforces_state_and_error_rules() -> None:
    payload: dict[str, Any] = {
        "state": "success",
        "tool_name": "inspect_range",
        "arguments": {"workbook_id": "workbook_1", "range_ref": "B2:B4"},
        "result": {"Revenue!B2": 50},
        "error": None,
        "provenance": [_workbook_provenance()],
    }

    assert _validate_json(ToolResult, payload).state is ResultState.SUCCESS

    with pytest.raises(ValidationError):
        _validate_json(ToolResult, {**payload, "state": "error", "error": None})
    with pytest.raises(ValidationError):
        _validate_json(ToolResult, {**payload, "state": "error", "error": ""})
    with pytest.raises(ValidationError):
        _validate_json(ToolResult, {**payload, "error": "unexpected"})


def test_sandbox_result_records_execution_and_computations() -> None:
    payload: dict[str, Any] = {
        "state": "success",
        "execution_id": "exec_2",
        "stdout": "150\n",
        "stderr": "",
        "result": 150,
        "error": None,
        "facade_calls": [{"method": "sheet", "sheet_name": "Revenue"}],
        "computation_ids": ["calc_3"],
    }

    result = _validate_json(SandboxResult, payload)

    assert result.computation_ids == ["calc_3"]
    assert result.model_dump(mode="json") == payload

    with pytest.raises(ValidationError):
        _validate_json(SandboxResult, {**payload, "state": "error", "error": None})
    with pytest.raises(ValidationError):
        _validate_json(SandboxResult, {**payload, "execution_id": ""})


def test_observation_has_exact_shape_and_error_rules() -> None:
    payload: dict[str, Any] = {
        "source": "exploration",
        "status": "success",
        "summary": "revenue",
        "facts": ["Revenue totals sit under the Revenue header."],
        "result": None,
        "error": None,
        "provenance": [_workbook_provenance()],
        "uncertainties": ["Exploration evidence is a lead and needs an exact read."],
        "next_question": None,
    }

    assert _validate_json(Observation, payload).model_dump(mode="json") == payload

    with pytest.raises(ValidationError):
        _validate_json(Observation, {**payload, "status": "error", "error": None})
    with pytest.raises(ValidationError):
        _validate_json(Observation, {**payload, "source": "verifier"})
    with pytest.raises(ValidationError):
        _validate_json(Observation, {**payload, "trace_id": "event_1"})


def test_candidate_answer_and_computation_are_exact() -> None:
    payload = _candidate_payload()

    candidate = _validate_json(CandidateAnswer, payload)

    assert candidate.model_dump(mode="json") == payload
    assert isinstance(candidate.claims[0], Claim)

    with pytest.raises(ValidationError):
        _validate_json(
            CandidateAnswer,
            {
                **payload,
                "claims": [{**payload["claims"][0], "unit": "million VND"}],
            },
        )
    with pytest.raises(ValidationError):
        _validate_json(CandidateAnswer, {**payload, "answer_text": ""})

    claim_without_support = {
        "id": "claim_2",
        "statement": "Revenue grew.",
        "value": None,
        "citations": [],
        "computation_id": None,
    }
    with pytest.raises(ValidationError):
        _validate_json(CandidateAnswer, {**payload, "claims": [claim_without_support]})

    computation = {
        "schema_version": "1.0",
        "id": "calc_3",
        "operation": "sum",
        "inputs": [
            {"reference": "Revenue!B2", "value": 50},
            {"reference": "Revenue!B3", "value": 40},
            {"reference": "Revenue!B4", "value": 60},
        ],
        "output": 150,
        "metadata": {"unit": "million VND"},
        "tolerance": 0.0,
    }
    assert _validate_json(ComputationRecord, computation).model_dump(mode="json") == computation


def test_deterministic_report_status_matches_check_statuses() -> None:
    passing = {
        "id": "ANSWER.CITATION_EXISTS",
        "status": "PASS",
        "observed": ["Revenue!B2:B4"],
        "expected": "All citations resolve",
        "reason": "The cited range exists and is readable.",
    }
    failing = {
        "id": "CALC.ARITHMETIC",
        "status": "FAIL",
        "observed": 149,
        "expected": 150,
        "reason": "The recorded sum does not recompute.",
    }

    assert _validate_json(DeterministicCheckResult, passing).status is RunStatus.PASS

    report = {"schema_version": "1.0", "status": "PASS", "checks": [passing]}
    assert _validate_json(DeterministicCheckReport, report).status is RunStatus.PASS

    with pytest.raises(ValidationError):
        _validate_json(
            DeterministicCheckReport,
            {"schema_version": "1.0", "status": "PASS", "checks": [passing, failing]},
        )
    with pytest.raises(ValidationError):
        _validate_json(
            DeterministicCheckReport,
            {"schema_version": "1.0", "status": "FAIL", "checks": [passing]},
        )


def test_judge_result_is_a_compact_holistic_verdict() -> None:
    payload = _judge_payload()

    result = _validate_json(LLMJudgeResult, payload)

    assert result.status is RunStatus.PASS
    assert result.model_dump(mode="json") == payload

    for obsolete in ("criteria", "questions", "blocking_issues"):
        with pytest.raises(ValidationError):
            _validate_json(LLMJudgeResult, {**payload, obsolete: []})


def test_failing_judge_requires_reason_issues_and_blocking_feedback() -> None:
    assert _validate_json(LLMJudgeResult, _failing_judge_payload()).status is RunStatus.FAIL

    for field in ("issues_found", "improvement_feedback"):
        with pytest.raises(ValidationError):
            _validate_json(LLMJudgeResult, {**_failing_judge_payload(), field: []})

    with pytest.raises(ValidationError):
        _validate_json(
            LLMJudgeResult,
            {
                **_failing_judge_payload(),
                "improvement_feedback": [_feedback_payload(priority="non_blocking")],
            },
        )
    with pytest.raises(ValidationError):
        _validate_json(LLMJudgeResult, {**_failing_judge_payload(), "reason": ""})


def test_passing_judge_allows_only_non_blocking_feedback() -> None:
    payload = {
        **_judge_payload(),
        "improvement_feedback": [_feedback_payload(priority="non_blocking")],
    }

    assert _validate_json(LLMJudgeResult, payload).status is RunStatus.PASS

    with pytest.raises(ValidationError):
        _validate_json(
            LLMJudgeResult,
            {**_judge_payload(), "improvement_feedback": [_feedback_payload()]},
        )
    with pytest.raises(ValidationError):
        _validate_json(LLMJudgeResult, {**_judge_payload(), "issues_found": ["late issue"]})


@pytest.mark.parametrize("confidence", [-0.1, 1.1, "0.5", nan, inf, True])
def test_confidence_is_finite_numeric_and_bounded(confidence: object) -> None:
    with pytest.raises(ValidationError):
        LLMJudgeResult.model_validate({**_judge_payload(), "confidence_score": confidence})


@pytest.mark.parametrize(
    "changes",
    [
        {"source": "model"},
        {"priority": "high"},
        {"code": ""},
        {"action": ""},
        {"recheck": ""},
        {"active": True},
        {"resolved": False},
        {"failed_check_or_question_id": "data.correct_scope"},
    ],
)
def test_feedback_vocabulary_and_fields_are_exact(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _validate_json(ImprovementFeedback, {**_feedback_payload(), **changes})


def test_deterministic_feedback_rechecks_its_own_check_id() -> None:
    payload = _feedback_payload(source="deterministic_check", recheck="CALC.ARITHMETIC")

    assert _validate_json(ImprovementFeedback, payload).recheck == "CALC.ARITHMETIC"

    with pytest.raises(ValidationError):
        _validate_json(
            ImprovementFeedback,
            _feedback_payload(source="deterministic_check", recheck="LLM_JUDGE"),
        )
    with pytest.raises(ValidationError):
        _validate_json(
            ImprovementFeedback,
            _feedback_payload(source="llm_judge", recheck="CALC.ARITHMETIC"),
        )


def test_verification_result_references_one_artifact_and_matches_status() -> None:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "FAIL",
        "reason": "The selected table does not represent the requested metric.",
        "artifact": "verification/3.json",
        "improvement_feedback": [_feedback_payload()],
    }

    result = _validate_json(VerificationResult, payload)

    assert result.status is RunStatus.FAIL
    assert result.model_dump(mode="json") == payload

    with pytest.raises(ValidationError):
        _validate_json(VerificationResult, {**payload, "artifact": "/tmp/verification.json"})
    with pytest.raises(ValidationError):
        _validate_json(VerificationResult, {**payload, "status": "PASS"})
    with pytest.raises(ValidationError):
        _validate_json(VerificationResult, {**payload, "reason": ""})
    with pytest.raises(ValidationError):
        _validate_json(
            VerificationResult,
            {
                **payload,
                "llm_judge_artifact": "verification/llm_judge.json",
                "deterministic_artifact": "verification/deterministic.json",
            },
        )

    passing = {
        **payload,
        "status": "PASS",
        "reason": "Every check and the judge passed.",
        "improvement_feedback": [],
    }
    assert _validate_json(VerificationResult, passing).status is RunStatus.PASS


def test_unknown_and_extra_fields_are_rejected() -> None:
    for model, payload in (
        (
            ObserveResult,
            {
                "known_facts": [],
                "uncertainties": [],
                "execution_objective": "Confirm the header.",
                "rationale": "The header is unconfirmed.",
            },
        ),
        (CandidateAnswer, _candidate_payload()),
        (LLMJudgeResult, _judge_payload()),
        (HistoryEvent, _history_payload()),
    ):
        with pytest.raises(ValidationError):
            _validate_json(model, {**payload, "unexpected": "value"})
