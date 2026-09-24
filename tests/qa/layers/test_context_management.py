"""Deterministic context assembly; see ``specs/layers/context_management.md``."""

import json
from typing import Any

import pytest
from pydantic import TypeAdapter

from autotab.qa.layers.context_management import (
    ContextBudgetError,
    ContextRequest,
    Step,
    active_feedback,
    assemble_context,
)
from autotab.qa.layers.verification import run_action_checks
from autotab.qa.prompts import RUBRIC, SANDBOX_CAPABILITIES
from autotab.qa.schemas import (
    AgentAction,
    CandidateAnswer,
    ComputationRecord,
    FeedbackPriority,
    FeedbackSource,
    ImprovementFeedback,
    LLMJudgeResult,
    Observation,
    ObservationSource,
    ObserveResult,
    Phase,
    ResultState,
)

WORKBOOKS = [
    {
        "workbook_id": "workbook_1",
        "file": "QA_sample.xlsx",
        "sheets": [{"name": "People", "dimensions": "A1:G22", "merged": ["G1:G2"]}],
    }
]
TOOLS = [{"name": "inspect_range", "description": "Read cells.", "schema": {"type": "object"}}]


def _step(turn: int, facts: int = 3) -> Step:
    action = TypeAdapter(AgentAction).validate_python(
        {
            "action_type": "tool",
            "rationale": "Read scores.",
            "tool_name": "inspect_range",
            "arguments": {"workbook_id": "workbook_1", "range_ref": f"G{turn}"},
        }
    )
    observation = Observation(
        source=ObservationSource.TOOL,
        status=ResultState.SUCCESS,
        summary=f"inspect_range People!G{turn}",
        facts=[f"People!G{turn}: raw {90 + i} (workbook value) " + "x" * 40 for i in range(facts)],
    )
    return Step(turn=turn, action=action, observation=observation, execution_id=f"exec_{turn}")


def _blocking(code: str = "DATA_HANDLING.SCOPE_ERROR") -> ImprovementFeedback:
    return ImprovementFeedback(
        code=code,
        source=FeedbackSource.LLM_JUDGE,
        problem="The answer asserts 'man' but no column records gender.",
        evidence=["People!A1:G2"],
        action="Say that gender cannot be verified from the workbook.",
        priority=FeedbackPriority.BLOCKING,
        recheck="LLM_JUDGE",
    )


def _candidate() -> CandidateAnswer:
    return CandidateAnswer.model_validate(
        {
            "answer_text": "Nga Anh Ngo has the highest score, 99.",
            "claims": [
                {
                    "id": "claim_1",
                    "statement": "The highest score is 99.",
                    "value": 99,
                    "citations": ["People!G3:G22"],
                    "computation_id": "calc_1",
                },
                {
                    "id": "claim_2",
                    "statement": "Row 13 is Nga Anh Ngo.",
                    "value": "Nga Anh Ngo",
                    "citations": ["People!B13:D13"],
                    "computation_id": None,
                },
            ],
        }
    )


def _request(phase: Phase, **overrides: Any) -> ContextRequest:
    values: dict[str, Any] = {
        "phase": phase,
        "query": "Full name of the man who have the highest score",
        "workbooks": WORKBOOKS,
        "budget_tokens": 20_000,
        "output_schema": {"type": "object", "title": "Reply"},
        "tool_descriptions": TOOLS,
        "sandbox_names": ["df"],
    }
    values.update(overrides)
    return ContextRequest(**values)


def test_each_phase_uses_its_own_template() -> None:
    texts = {
        phase: assemble_context(_request(phase)).text
        for phase in Phase
        if phase in (Phase.OBSERVE, Phase.EXECUTE, Phase.VERIFY)
    }

    assert "Phase: OBSERVE" in texts[Phase.OBSERVE]
    assert "Phase: EXECUTE" in texts[Phase.EXECUTE]
    assert "Phase: VERIFICATION" in texts[Phase.VERIFY]
    assert len(set(texts.values())) == 3


def test_tools_and_sandbox_appear_only_in_execute() -> None:
    execute = assemble_context(_request(Phase.EXECUTE)).text
    observe = assemble_context(_request(Phase.OBSERVE)).text
    verify = assemble_context(_request(Phase.VERIFY, candidate=_candidate())).text

    assert "inspect_range: Read cells." in execute
    assert SANDBOX_CAPABILITIES in execute
    assert "Currently bound in the session: df" in execute
    for text in (observe, verify):
        assert "inspect_range: Read cells." not in text
        assert SANDBOX_CAPABILITIES not in text


def test_the_query_is_verbatim_and_the_workbook_manifest_is_present() -> None:
    text = assemble_context(_request(Phase.OBSERVE)).text

    assert "Full name of the man who have the highest score" in text
    assert "workbook_1" in text and "People" in text and "G1:G2" in text


def test_verification_prompt_has_every_rubric_question_and_the_full_answer() -> None:
    text = assemble_context(_request(Phase.VERIFY, candidate=_candidate())).text

    for section, questions in RUBRIC:
        assert section in text
        for question in questions:
            assert question in text
    assert "Nga Anh Ngo has the highest score, 99." in text
    assert "claim_1" in text and "claim_2" in text


def test_after_failure_observe_gets_every_blocking_item_intact() -> None:
    items = [_blocking(), _blocking("ANSWER_QUALITY.UNDISCLOSED_ASSUMPTION")]

    text = assemble_context(_request(Phase.OBSERVE, feedback=items, after_failure=True)).text

    assert "FAILED verification" in text
    for item in items:
        for part in (item.code, item.problem, item.action, item.recheck, *item.evidence):
            assert part in text


def test_execute_sees_the_plan_and_the_latest_observation() -> None:
    plan = ObserveResult(
        known_facts=["Scores are in column G."],
        uncertainties=["No gender column seen yet."],
        execution_objective="Rank all rows by score.",
        rationale="The top scorer is needed.",
    )

    text = assemble_context(_request(Phase.EXECUTE, plan=plan, steps=[_step(3)])).text

    assert "Rank all rows by score." in text
    assert "No gender column seen yet." in text
    assert "People!G3: raw 90" in text


def test_exploration_leads_are_marked_as_leads() -> None:
    lead = Observation(
        source=ObservationSource.EXPLORATION,
        status=ResultState.SUCCESS,
        summary="highest score",
        facts=["Score column header in G1."],
        uncertainties=["Exploration evidence is a lead and needs an exact read."],
    )

    text = assemble_context(_request(Phase.OBSERVE, exploration=[lead])).text

    assert "Score column header in G1." in text
    assert "lead" in text.lower()


def test_identical_inputs_give_identical_prompts() -> None:
    request = _request(Phase.EXECUTE, steps=[_step(3), _step(4)], feedback=[_blocking()])

    assert assemble_context(request).text == assemble_context(request).text


def test_older_steps_shrink_before_anything_blocking_is_dropped() -> None:
    steps = [_step(turn, facts=30) for turn in range(3, 13)]
    roomy = assemble_context(_request(Phase.EXECUTE, steps=steps, feedback=[_blocking()]))
    budget = roomy.tokens - 600

    tight = assemble_context(
        _request(Phase.EXECUTE, steps=steps, feedback=[_blocking()], budget_tokens=budget)
    )

    assert tight.tokens <= budget
    assert tight.omitted
    assert _blocking().problem in tight.text
    assert "People!G12: raw 90" in tight.text  # the latest observation stays whole
    assert "history above is incomplete" in tight.text


def test_a_budget_that_cannot_hold_blocking_feedback_fails_loudly() -> None:
    reserved = assemble_context(_request(Phase.OBSERVE)).tokens

    with pytest.raises(ContextBudgetError, match="DATA_HANDLING.SCOPE_ERROR"):
        assemble_context(
            _request(Phase.OBSERVE, feedback=[_blocking()], budget_tokens=reserved + 5)
        )


# --- active feedback ---------------------------------------------------------------


def _wrong_sum() -> ComputationRecord:
    return ComputationRecord.model_validate(
        {
            "id": "calc_1",
            "operation": "sum",
            "inputs": [{"reference": "A1", "value": 1}, {"reference": "A2", "value": 2}],
            "output": 4,
        }
    )


def _right_sum() -> ComputationRecord:
    return _wrong_sum().model_copy(update={"id": "calc_2", "output": 3})


def _judge(status: str) -> LLMJudgeResult:
    feedback = [_blocking().model_dump(mode="json")] if status == "FAIL" else []
    # Judge replies arrive as JSON, so validate them the way the verifier does.
    return LLMJudgeResult.model_validate_json(
        json.dumps(
            {
                "status": status,
                "confidence_score": 0.9,
                "reason": "r",
                "issues_found": ["i"] if status == "FAIL" else [],
                "improvement_feedback": feedback,
                "final_assessment": "a",
            }
        )
    )


def test_a_newer_real_pass_resolves_a_failed_check() -> None:
    failed = run_action_checks(ResultState.SUCCESS, [_wrong_sum()])
    fixed = run_action_checks(ResultState.SUCCESS, [_right_sum()])

    assert [item.recheck for item in active_feedback([failed])] == ["CALC.ARITHMETIC"]
    assert active_feedback([failed, fixed]) == []


def test_a_not_applicable_pass_does_not_resolve_a_failed_check() -> None:
    failed = run_action_checks(ResultState.SUCCESS, [_wrong_sum()])
    unrelated_read = run_action_checks(ResultState.SUCCESS, [])

    assert [item.recheck for item in active_feedback([failed, unrelated_read])] == [
        "CALC.ARITHMETIC"
    ]


def test_the_latest_judge_decides_the_judge_feedback_set() -> None:
    assert len(active_feedback([_judge("FAIL")])) == 1
    assert active_feedback([_judge("FAIL"), _judge("PASS")]) == []
    assert len(active_feedback([_judge("FAIL"), _judge("FAIL")])) == 1


def test_execute_and_verification_demand_full_coverage_for_aggregates() -> None:
    # A maximum concluded from the first rows of a longer table passed a real run.
    execute = assemble_context(_request(Phase.EXECUTE)).text
    verify = assemble_context(_request(Phase.VERIFY, candidate=_candidate())).text

    assert "every data row" in execute
    assert "used range" in verify and "never read" in verify


def test_exploration_leads_are_described_as_windows_not_extents() -> None:
    lead = Observation(
        source=ObservationSource.EXPLORATION,
        status=ResultState.SUCCESS,
        summary="highest score",
        facts=["Rows 3 to 5 hold 92, 60, 92."],
    )

    text = assemble_context(_request(Phase.OBSERVE, exploration=[lead])).text

    assert "window" in text and "used range" in text


def test_the_judge_never_sees_earlier_feedback() -> None:
    # A real run showed a resolved PROVENANCE failure still listed as "must be
    # resolved" in the judge prompt; the judge believed it over the current checks.
    text = assemble_context(
        _request(Phase.VERIFY, candidate=_candidate(), feedback=[_blocking()])
    ).text

    assert _blocking().problem not in text
    assert "Blocking feedback" not in text


def test_the_judge_is_told_to_account_for_every_condition_in_the_question() -> None:
    text = assemble_context(_request(Phase.VERIFY, candidate=_candidate())).text

    assert "every condition" in text
    assert "ordinary meaning" in text


def _max_over_scores() -> ComputationRecord:
    inputs = [{"reference": f"People!G{row}", "value": 60 + row} for row in range(3, 23)]
    return ComputationRecord.model_validate_json(
        json.dumps(
            {
                "id": "calc_1",
                "operation": "max",
                "inputs": inputs,
                "output": 82,
                "metadata": {"sources": ["People!G3:G22"], "skipped_blank": ["People!G9"]},
            }
        )
    )


def test_the_judge_sees_where_each_relied_on_computation_came_from() -> None:
    unused = _max_over_scores().model_copy(update={"id": "calc_7"})
    text = assemble_context(
        _request(
            Phase.VERIFY,
            candidate=_candidate(),
            computations={"calc_1": _max_over_scores(), "calc_7": unused},
        )
    ).text

    assert "calc_1: max over People!G3:G22 -> 20 inputs (People!G3 … People!G22)" in text
    assert "1 blank skipped" in text
    assert "calc_7" not in text


def test_computation_provenance_is_for_the_judge_only() -> None:
    text = assemble_context(
        _request(Phase.EXECUTE, computations={"calc_1": _max_over_scores()})
    ).text

    assert "Recorded computations" not in text


def test_every_phase_is_told_that_a_preview_is_partial() -> None:
    for phase in (Phase.OBSERVE, Phase.EXECUTE):
        assert "preview" in assemble_context(_request(phase)).text
    execute = assemble_context(_request(Phase.EXECUTE)).text
    assert "load the data in a code action" in execute
