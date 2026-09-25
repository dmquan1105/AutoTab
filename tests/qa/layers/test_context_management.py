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
    ActionType,
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
    ProvenanceKind,
    ProvenanceReference,
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
        next_action=ActionType.CODE,
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
    budget = roomy.tokens - 100

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
    assert "input count" in verify and "fewer rows than the question needs" in verify


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
    # Choosing between a look and a computation is the plan's decision now.
    observe = assemble_context(_request(Phase.OBSERVE)).text
    assert "load the data in a code action" in observe


def _plan(next_action: str = "code") -> ObserveResult:
    return ObserveResult(
        known_facts=["Scores are in column G."],
        uncertainties=[],
        execution_objective="Compute the maximum of People!G3:G22.",
        next_action=ActionType(next_action),
        rationale="The top score is needed.",
    )


def _code_step(turn: int = 2) -> Step:
    action = TypeAdapter(AgentAction).validate_python(
        {
            "action_type": "code",
            "rationale": "Rank the scores.",
            "code": "top = max(scores)\ncid = record_computation('max', ['People!G3:G22'], top)",
        }
    )
    observation = Observation(
        source=ObservationSource.SANDBOX,
        status=ResultState.SUCCESS,
        summary=f"code run exec_{turn}",
        facts=["result (agent-computed): 82", "recorded computations: calc_1"],
    )
    return Step(turn=turn, action=action, observation=observation, execution_id=f"exec_{turn}")


def test_observe_chooses_the_kind_of_the_next_action() -> None:
    text = assemble_context(_request(Phase.OBSERVE)).text

    assert "next_action" in text
    for kind in ('"tool"', '"code"', '"answer"'):
        assert kind in text


def test_execute_is_held_to_the_action_the_plan_names() -> None:
    text = assemble_context(_request(Phase.EXECUTE, plan=_plan("code"))).text

    assert 'The plan asks for one "code" action' in text


def test_a_step_verification_judges_the_latest_action_against_the_objective() -> None:
    report = run_action_checks(ResultState.SUCCESS, [_max_over_scores()])
    text = assemble_context(
        _request(
            Phase.VERIFY,
            plan=_plan(),
            steps=[_step(1), _code_step(2)],
            report=report,
            computations={"calc_1": _max_over_scores()},
        )
    ).text

    assert "Phase: VERIFICATION of one step" in text
    assert "Compute the maximum of People!G3:G22." in text
    # The step under review is shown whole: its code, its result, and its checks.
    assert "cid = record_computation('max', ['People!G3:G22'], top)" in text
    assert "result (agent-computed): 82" in text
    assert "CALC.ARITHMETIC" in text
    assert "calc_1: max over People!G3:G22 -> 20 inputs" in text
    # It is not the answer rubric, and it cannot act.
    assert "Does the answer address every material request part?" not in text
    assert "inspect_range: Read cells." not in text


def test_a_step_verification_never_sees_earlier_feedback() -> None:
    text = assemble_context(
        _request(Phase.VERIFY, plan=_plan(), steps=[_code_step()], feedback=[_blocking()])
    ).text

    assert _blocking().problem not in text


def test_the_sandbox_capabilities_list_what_is_allowed() -> None:
    assert "pd.to_numeric" in SANDBOX_CAPABILITIES
    assert "try/except" in SANDBOX_CAPABILITIES
    assert "rebind" in SANDBOX_CAPABILITIES


STRUCTURED = [
    {
        "workbook_id": "workbook_1",
        "file": "QA_sample.xlsx",
        "sheets": [
            {
                "name": "People",
                "dimensions": "A1:G23",
                "merged": ["G1:G2"],
                "structure": {
                    "columns": [
                        {"column": "B", "filled": 22, "counts": {"text": 22}, "formulas": 0,
                         "text": [["B1", "Name"], ["B2", "First"], ["B3", "Ha"]]},
                        {"column": "G", "filled": 22, "counts": {"text": 1, "number": 21},
                         "formulas": 1, "numbers": {"min": 58, "max": 1500, "rows": [3, 23]},
                         "text": [["G1", "Score"]]},
                    ],
                    "more_columns": 3,
                    "total_rows": [23],
                    "blank_rows": [],
                    "window": {
                        "columns": ["A", "B", "G"],
                        "rows": [[1, ["No", "Name", "Score"]], [3, [1, "Ha", None]]],
                    },
                },
            }
        ],
    }
]  # fmt: skip


def test_every_phase_sees_each_sheets_cell_profile() -> None:
    for phase in (Phase.OBSERVE, Phase.EXECUTE):
        text = assemble_context(_request(phase, workbooks=STRUCTURED)).text

        assert 'B: 22 filled (22 text); text B1 "Name", B2 "First", B3 "Ha"' in text
        assert (
            "G: 22 filled (1 text, 21 number); numbers 58..1500 in rows 3-23; 1 formula; "
            'text G1 "Score"'
        ) in text
        assert "3 more columns" in text
        assert "rows with total/sum text: 23" in text
        assert "top-left cells, verbatim:" in text
        assert "      row | A | B | G" in text
        assert "      1 | No | Name | Score" in text
        assert "      3 | 1 | Ha | " in text
        manifest = text.split("Workbooks:")[1].split("\n\n")[0]
        assert "header" not in manifest.lower()


def test_observe_knows_the_tool_read_limit() -> None:
    text = assemble_context(_request(Phase.OBSERVE, tool_read_cells=200)).text

    assert "at most 200 cells" in text


def test_a_step_judge_fails_only_for_a_defect_it_can_point_to() -> None:
    # A real run's correct first step was failed for checks it "could have made",
    # although its recorded computation already showed full coverage.
    text = assemble_context(_request(Phase.VERIFY, plan=_plan(), steps=[_code_step()])).text

    assert "concrete defect" in text
    assert "already settles" in text


def test_the_sandbox_capabilities_show_what_wb_range_returns() -> None:
    assert '[0][2]["value"]' in SANDBOX_CAPABILITIES
    assert "indexed by worksheet row" in SANDBOX_CAPABILITIES
    assert "print(" in SANDBOX_CAPABILITIES and "isinstance" in SANDBOX_CAPABILITIES


def _read(turn: int, range_ref: str, status: ResultState = ResultState.SUCCESS) -> Step:
    action = TypeAdapter(AgentAction).validate_python(
        {
            "action_type": "tool",
            "rationale": "Read.",
            "tool_name": "inspect_range",
            "arguments": {"workbook_id": "workbook_1", "range_ref": range_ref},
        }
    )
    observation = Observation(
        source=ObservationSource.TOOL,
        status=status,
        summary=f"inspect_range People!{range_ref}",
        facts=(
            [f"People!{range_ref}: detail row {i} " + "y" * 30 for i in range(5)]
            if status is ResultState.SUCCESS
            else []
        ),
        error="sheet 'Peple' does not exist" if status is ResultState.ERROR else None,
        provenance=[
            ProvenanceReference(
                kind=ProvenanceKind.WORKBOOK,
                workbook_id="workbook_1",
                sheet="People",
                range_ref=range_ref,
            )
        ],
    )
    return Step(turn=turn, action=action, observation=observation, execution_id=f"exec_{turn}")


def test_earlier_turns_are_one_line_digests_even_with_room_to_spare() -> None:
    # The prompt follows the step at hand: the latest turn whole, the rest as digests.
    # A real run carried eight earlier turns verbatim into every prompt.
    steps = [_read(1, "A1:G2"), _read(2, "G3:G22"), _read(3, "B13:D13")]

    text = assemble_context(_request(Phase.EXECUTE, steps=steps)).text

    assert "People!B13:D13: detail row 4" in text
    # Earlier reads keep only the start of what they returned, cut at 200 characters.
    assert "People!A1:G2: detail row 4" not in text
    assert "People!G3:G22: detail row 4" not in text
    assert "Turn 1 (exec_1): tool inspect_range" in text
    assert "-> success: inspect_range People!A1:G2" in text


def test_a_digest_keeps_why_an_earlier_turn_failed_and_what_code_returned() -> None:
    steps = [_read(1, "A1:G2", ResultState.ERROR), _code_step(2), _read(3, "B13:D13")]

    text = assemble_context(_request(Phase.OBSERVE, steps=steps)).text

    assert "Turn 1 (exec_1): tool inspect_range" in text
    assert "-> error: sheet 'Peple' does not exist" in text
    assert "Turn 2 (exec_2): code (2 lines) -> success: result (agent-computed): 82" in text


def test_the_answer_judge_sees_the_turns_behind_its_evidence_in_full() -> None:
    record = _max_over_scores()
    rank = _code_step(4).observation.model_copy(
        update={
            "provenance": [
                ProvenanceReference(kind=ProvenanceKind.COMPUTATION, computation_id="calc_1")
            ]
        }
    )
    steps = [
        _read(1, "G3:G22"),
        _read(2, "Z1:Z2"),
        _read(3, "B13:D13"),
        Step(turn=4, action=_code_step(4).action, observation=rank, execution_id="exec_4"),
        _read(5, "A1:A2"),
    ]

    text = assemble_context(
        _request(
            Phase.VERIFY,
            candidate=_candidate(),
            steps=steps,
            computations={"calc_1": record},
        )
    ).text

    assert "People!G3:G22: detail row 4" in text  # overlaps the cited People!G3:G22
    assert "People!B13:D13: detail row 4" in text  # the cited name cells
    assert "result (agent-computed): 82" in text  # recorded the cited calc_1
    assert "People!Z1:Z2: detail row 4" not in text  # a digest, not the whole read
    assert "Turn 2 (exec_2): tool inspect_range" in text


def test_no_judge_asks_for_a_label_to_be_recorded() -> None:
    # Two real runs looped after the answer judge demanded that a name be recorded
    # with record_computation, which only records numbers.
    answer = assemble_context(_request(Phase.VERIFY, candidate=_candidate())).text
    step = assemble_context(_request(Phase.VERIFY, plan=_plan(), steps=[_code_step()])).text

    for text in (answer, step):
        assert "never ask for a label to be recorded" in text


def test_execute_is_told_to_return_row_numbers_for_looked_up_labels() -> None:
    text = assemble_context(_request(Phase.EXECUTE)).text

    assert "worksheet row numbers" in text
    assert "exact cells" in text


def test_no_prompt_template_carries_benchmark_data() -> None:
    # Examples drawn from the benchmark sample leak its answer into every prompt: one
    # once showed the very range and maximum a sample question asks for.
    from autotab.qa import prompts
    from autotab.qa.layers import verification

    templates = [value for value in vars(prompts).values() if isinstance(value, str)]
    templates.append(prompts.rubric_text())
    # Check repairs reach the model as feedback, so they are prompt text too.
    templates.extend(verification._REPAIRS.values())
    templates.extend([verification._MISSING_COMPUTATION, verification._MISLINKED_CLAIM])
    for text in templates:
        for leaked in ("People", "G3:G22", "B13", "Nga", "Score", "99"):
            assert leaked not in text, (leaked, text[:80])


def test_a_judge_sees_the_condition_behind_a_row_selection() -> None:
    selection = _max_over_scores().model_copy(
        update={
            "id": "calc_2",
            "operation": "rows_where",
            "output": [13],
            "metadata": {
                "sources": ["People!G3:G22"],
                "condition": {"op": "equals", "value": 82, "source": "calc_1"},
            },
        }
    )
    claim = {
        "id": "claim_1",
        "statement": "Only row 13 holds the maximum.",
        "value": 13,
        "citations": ["People!G3:G22"],
        "computation_id": "calc_2",
    }
    candidate = CandidateAnswer.model_validate({"answer_text": "Row 13.", "claims": [claim]})

    text = assemble_context(
        _request(Phase.VERIFY, candidate=candidate, computations={"calc_2": selection})
    ).text

    assert "calc_2: rows_where over People!G3:G22" in text
    assert "where value equals 82 (calc_1)" in text
    assert "output [13]" in text


def test_execute_knows_how_to_record_a_row_selection() -> None:
    text = assemble_context(_request(Phase.EXECUTE)).text

    assert 'record_computation("rows_where"' in text
    assert "equals, not_equals, gt, ge, lt, le, contains" in text


def test_check_repairs_come_before_judge_advice_and_take_precedence() -> None:
    # A run followed a judge's advice to record a name, which a check had already said
    # was the wrong repair; the exact check has to win where the two disagree.
    check = ImprovementFeedback(
        code="ANSWER.PROVENANCE",
        source=FeedbackSource.DETERMINISTIC_CHECK,
        problem="claim_2: value 13 differs from calc_2 output 99",
        action="Do not recompute: give each computation_id only to its own claim.",
        priority=FeedbackPriority.BLOCKING,
        recheck="ANSWER.PROVENANCE",
    )

    text = assemble_context(
        _request(Phase.OBSERVE, feedback=[_blocking(), check], after_failure=True)
    ).text

    checks_at = text.index("Failed checks")
    judge_at = text.index("Judge findings")
    assert checks_at < text.index("ANSWER.PROVENANCE") < judge_at
    assert judge_at < text.index(_blocking().code)
    assert "follow the check" in text


def test_the_answer_judge_sees_the_cited_cells_as_the_harness_read_them() -> None:
    # A judge failed a correct answer because the named cells were "not in the
    # evidence"; the harness reads every citation itself, so the values are shown.
    scores = [(f"G{row}", 60 + row) for row in range(3, 23)]
    cited = {
        "People!B13:D13": [("B13", "Nga"), ("C13", "Anh"), ("D13", "Ngo")],
        "People!G3:G22": scores,
    }

    text = assemble_context(_request(Phase.VERIFY, candidate=_candidate(), cited_cells=cited)).text

    assert "Cited cells, read by the harness from the workbook:" in text
    assert 'People!B13:D13: B13 = "Nga", C13 = "Anh", D13 = "Ngo"' in text
    assert "People!G3:G22: 20 cells" in text
    assert "G3 = 63" in text and "G22 = 82" in text and "…" in text


def test_the_answer_judge_leaves_grounding_to_the_checks() -> None:
    text = assemble_context(_request(Phase.VERIFY, candidate=_candidate())).text

    assert "already established by code" in text
    assert "never fail an answer for evidence it does not repeat" in text
    assert "layout sample" in text


def test_observe_is_told_why_the_last_execute_produced_nothing() -> None:
    rejected = 'the plan asks for a "code" action, not "answer"; reply began: {"answer_text"'

    text = assemble_context(_request(Phase.OBSERVE, rejected_execute=rejected)).text

    assert "The last EXECUTE produced no valid action" in text
    assert rejected in text


def test_a_read_digest_keeps_what_the_read_returned() -> None:
    # A run read the same three name cells twice: the digest of the first read said
    # only which range it read, not what was there.
    steps = [_read(1, "B13:D13"), _read(2, "G3:G22")]

    text = assemble_context(_request(Phase.OBSERVE, steps=steps)).text

    digest = next(line for line in text.splitlines() if line.startswith("Turn 1 (exec_1)"))
    assert "People!B13:D13: detail row 0" in digest


def test_the_omission_note_never_breaks_the_budget() -> None:
    # Found in review: one label per omitted turn let the note outgrow its reserve.
    steps = [_step(turn, facts=12) for turn in range(1, 80)]

    prompt = assemble_context(_request(Phase.OBSERVE, steps=steps, budget_tokens=2000))

    assert prompt.tokens <= 2000
    assert "more" in prompt.text and "history above is incomplete" in prompt.text
