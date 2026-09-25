"""Verify-phase orchestration tests with a scripted judge; see verify_phase.md."""

from typing import Any

import pytest

from autotab.qa.layers.context_management import ContextRequest
from autotab.qa.phases import verify_phase
from autotab.qa.prompts import RUBRIC
from autotab.qa.schemas import (
    ActionType,
    CandidateAnswer,
    ComputationRecord,
    LLMJudgeResult,
    Phase,
    ResultState,
    RunStatus,
)

CANDIDATE = CandidateAnswer.model_validate_json(
    """{"answer_text": "The top scorer is Nga Anh Ngo with 99; gender is not recorded.",
        "claims": [
          {"id": "claim_1", "statement": "The highest score is 99.", "value": 99,
           "citations": ["People!G3:G22"], "computation_id": "calc_1"},
          {"id": "claim_2", "statement": "Row 13 is Nga Anh Ngo.", "value": "Nga Anh Ngo",
           "citations": ["People!B13:D13"], "computation_id": null}]}"""
)
COMPUTATION = ComputationRecord.model_validate_json(
    '{"id": "calc_1", "operation": "max", "inputs": ['
    '{"reference": "People!G3", "value": 92}, {"reference": "People!G13", "value": 99}],'
    ' "output": 99}'
)
JUDGE_PASS = {
    "status": "PASS",
    "confidence_score": 0.85,
    "reason": "The top score and its row are read exactly; the missing gender is disclosed.",
    "issues_found": [],
    "improvement_feedback": [],
    "final_assessment": "Grounded and honest about the unverifiable condition.",
}


def _evidence(**overrides: Any) -> verify_phase.AnswerEvidence:
    values: dict[str, Any] = {
        "computations": {"calc_1": COMPUTATION},
        "computation_states": {"calc_1": ResultState.SUCCESS},
        "resolve_citation": lambda sheet, rng: sheet == "People",
        "source_hashes": {"workbook_1": ("h", "h")},
    }
    values.update(overrides)
    return verify_phase.AnswerEvidence(**values)


def _request(candidate: CandidateAnswer | None = CANDIDATE) -> ContextRequest:
    return ContextRequest(
        phase=Phase.VERIFY,
        query="Full name of the man who have the highest score",
        workbooks=[{"workbook_id": "workbook_1", "file": "QA_sample.xlsx", "sheets": []}],
        budget_tokens=20_000,
        output_schema=LLMJudgeResult.model_json_schema(),
        candidate=candidate,
    )


def _run(client: Any, **evidence: Any) -> verify_phase.VerifyOutcome:
    return verify_phase.run(
        _request(), client, evidence=_evidence(**evidence), artifact="verification/1.json"
    )


def test_checks_run_first_and_the_judge_sees_them_with_the_rubric(scripted: Any) -> None:
    client = scripted(JUDGE_PASS)

    outcome = _run(client)

    assert outcome.result.status is RunStatus.PASS
    prompt = client.prompts[0]
    assert "Deterministic checks: PASS" in prompt
    assert "The top scorer is Nga Anh Ngo with 99" in prompt
    for _, questions in RUBRIC:
        assert all(question in prompt for question in questions)


def test_a_failed_check_fails_the_answer_even_when_the_judge_passes(scripted: Any) -> None:
    outcome = _run(scripted(JUDGE_PASS), source_hashes={"workbook_1": ("h", "changed")})

    assert outcome.result.status is RunStatus.FAIL
    assert "RUNTIME.SOURCE_IMMUTABLE" in outcome.result.reason


def test_an_unreadable_judge_is_re_asked_then_fails_the_answer(scripted: Any) -> None:
    client = scripted("looks good to me", "PASS")

    outcome = _run(client)

    assert len(client.prompts) == 2
    assert outcome.reply.value is None
    assert outcome.result.status is RunStatus.FAIL
    assert "no valid judgement" in outcome.result.reason


def test_a_judge_fail_carries_its_blocking_feedback(scripted: Any) -> None:
    judge_fail = {
        **JUDGE_PASS,
        "status": "FAIL",
        "reason": "The answer asserts the person is a man without evidence.",
        "issues_found": ["Gender is asserted, not read."],
        "improvement_feedback": [
            {
                "code": "ANSWER_QUALITY.UNDISCLOSED_ASSUMPTION",
                "source": "llm_judge",
                "problem": "Gender is not recorded.",
                "evidence": ["People!A1:G2"],
                "action": "Disclose that gender cannot be verified.",
                "priority": "blocking",
                "recheck": "LLM_JUDGE",
            }
        ],
    }

    outcome = _run(scripted(judge_fail))

    assert outcome.result.status is RunStatus.FAIL
    assert [i.code for i in outcome.result.improvement_feedback] == [
        "ANSWER_QUALITY.UNDISCLOSED_ASSUMPTION"
    ]


def test_verify_requires_a_candidate(scripted: Any) -> None:
    with pytest.raises(ValueError):
        verify_phase.run(
            _request(candidate=None),
            scripted(JUDGE_PASS),
            evidence=_evidence(),
            artifact="verification/1.json",
        )


# --- step verification: every tool or code action is checked and judged ---------------

STEP_JUDGE_FAIL = {
    "status": "FAIL",
    "confidence_score": 0.7,
    "reason": "The read stops at row 5 while the table runs to row 22.",
    "issues_found": ["Rows 6-22 were not read."],
    "improvement_feedback": [
        {
            "code": "DATA_HANDLING.SCOPE_ERROR",
            "source": "llm_judge",
            "problem": "Only rows 3-5 were read.",
            "evidence": ["People!A1:G5"],
            "action": "Load the whole used range People!A1:G22 in code.",
            "priority": "blocking",
            "recheck": "LLM_JUDGE",
        }
    ],
    "final_assessment": "Partial read.",
}


def _step_request() -> ContextRequest:
    from pydantic import TypeAdapter

    from autotab.qa.layers.context_management import Step
    from autotab.qa.schemas import AgentAction, Observation, ObservationSource, ObserveResult

    action = TypeAdapter(AgentAction).validate_python(
        {"action_type": "code", "rationale": "Rank.", "code": "top = 99"}
    )
    observation = Observation(
        source=ObservationSource.SANDBOX, status=ResultState.SUCCESS, summary="code run exec_2"
    )
    return ContextRequest(
        phase=Phase.VERIFY,
        query="Full name of the man who have the highest score",
        workbooks=[{"workbook_id": "workbook_1", "file": "QA_sample.xlsx", "sheets": []}],
        budget_tokens=20_000,
        output_schema=LLMJudgeResult.model_json_schema(),
        plan=ObserveResult(
            known_facts=[],
            uncertainties=[],
            execution_objective="Compute the top score.",
            next_action=ActionType.CODE,
            rationale="Needed.",
        ),
        steps=[Step(turn=2, action=action, observation=observation, execution_id="exec_2")],
    )


def _run_step(client: Any, output: int = 99, state: ResultState = ResultState.SUCCESS) -> Any:
    record = COMPUTATION.model_copy(update={"output": output})
    return verify_phase.run_step(
        _step_request(),
        client,
        state=state,
        computations=(record,),
        artifact="turns/03_t2_verify_code/verdict.json",
    )


def test_a_step_runs_its_action_checks_then_asks_the_judge(scripted: Any) -> None:
    client = scripted(JUDGE_PASS)

    outcome = _run_step(client)

    assert outcome.result.status is RunStatus.PASS
    assert [c.id for c in outcome.report.checks] == [
        "CALC.ARITHMETIC",
        "CALC.INPUTS_DISTINCT",
        "RUNTIME.RESULT_COMPLETE",
    ]
    assert "Phase: VERIFICATION of one step" in client.prompts[0]
    assert "CALC.ARITHMETIC: PASS" in client.prompts[0]


def test_a_wrong_computation_fails_the_step_even_when_the_judge_passes(scripted: Any) -> None:
    outcome = _run_step(scripted(JUDGE_PASS), output=98)

    assert outcome.result.status is RunStatus.FAIL
    assert any(item.recheck == "CALC.ARITHMETIC" for item in outcome.result.improvement_feedback)


def test_a_step_judge_fail_carries_its_blocking_feedback(scripted: Any) -> None:
    outcome = _run_step(scripted(STEP_JUDGE_FAIL))

    assert outcome.result.status is RunStatus.FAIL
    assert [i.code for i in outcome.result.improvement_feedback] == ["DATA_HANDLING.SCOPE_ERROR"]


def test_step_verification_needs_a_step_and_no_candidate(scripted: Any) -> None:
    from dataclasses import replace

    with pytest.raises(ValueError):
        verify_phase.run_step(
            replace(_step_request(), steps=()),
            scripted(),
            state=ResultState.SUCCESS,
            computations=(),
            artifact="a.json",
        )
    with pytest.raises(ValueError):
        verify_phase.run_step(
            replace(_step_request(), candidate=CANDIDATE),
            scripted(),
            state=ResultState.SUCCESS,
            computations=(),
            artifact="a.json",
        )


def _tool_step_request() -> ContextRequest:
    from dataclasses import replace

    from pydantic import TypeAdapter

    from autotab.qa.layers.context_management import Step
    from autotab.qa.schemas import AgentAction, Observation, ObservationSource

    action = TypeAdapter(AgentAction).validate_python(
        {
            "action_type": "tool",
            "rationale": "Headers.",
            "tool_name": "inspect_range",
            "arguments": {"workbook_id": "workbook_1", "range_ref": "A1:G2"},
        }
    )
    observation = Observation(
        source=ObservationSource.TOOL, status=ResultState.SUCCESS, summary="inspect_range"
    )
    step = Step(turn=1, action=action, observation=observation, execution_id="exec_1")
    return replace(_step_request(), steps=[step])


def test_a_read_is_checked_but_never_judged(scripted: Any) -> None:
    # A read has nothing to judge beyond whether it came back; the next plan sees it.
    client = scripted()

    outcome = verify_phase.run_step(
        _tool_step_request(),
        client,
        state=ResultState.SUCCESS,
        computations=(),
        artifact="turns/03_t1_verify_tool/verdict.json",
    )

    assert client.prompts == []
    assert outcome.prompt is None and outcome.reply.value is None
    assert outcome.result.status is RunStatus.PASS
    assert "not judged" in outcome.result.reason


def test_a_failed_read_fails_its_step_without_a_judge(scripted: Any) -> None:
    client = scripted()

    outcome = verify_phase.run_step(
        _tool_step_request(),
        client,
        state=ResultState.ERROR,
        computations=(),
        artifact="turns/03_t1_verify_tool/verdict.json",
    )

    assert client.prompts == []
    assert outcome.result.status is RunStatus.FAIL
    assert [i.recheck for i in outcome.result.improvement_feedback] == ["RUNTIME.RESULT_COMPLETE"]
