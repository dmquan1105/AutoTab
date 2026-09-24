"""Verify-phase orchestration tests with a scripted judge; see verify_phase.md."""

from typing import Any

import pytest

from autotab.qa.layers.context_management import ContextRequest
from autotab.qa.phases import verify_phase
from autotab.qa.prompts import RUBRIC
from autotab.qa.schemas import (
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
