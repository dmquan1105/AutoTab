"""Pure normalization of raw results into observations; see observation.md."""

import json
from pathlib import Path

import pytest

from autotab.qa.layers.observation import (
    ExplorationError,
    load_exploration,
    normalize_sandbox,
    normalize_tool,
)
from autotab.qa.schemas import (
    ObservationSource,
    ProvenanceKind,
    ProvenanceReference,
    ResultState,
    SandboxResult,
    ToolResult,
)

WORKBOOK_ID = "workbook_1"


def _tool(state: ResultState = ResultState.SUCCESS, **overrides: object) -> ToolResult:
    payload: dict[str, object] = {
        "state": state,
        "tool_name": "inspect_range",
        "arguments": {"workbook_id": WORKBOOK_ID, "range_ref": "G3:G4"},
        "result": {
            "sheet": "People",
            "range": "G3:G4",
            "cells": [
                [{"coordinate": "G3", "value": 92, "displayed": 92, "formula": None}],
                [{"coordinate": "G4", "value": None, "displayed": None, "formula": "=MAX(G3)"}],
            ],
        },
        "provenance": [
            ProvenanceReference(
                kind=ProvenanceKind.WORKBOOK,
                workbook_id=WORKBOOK_ID,
                sheet="People",
                range_ref="G3:G4",
            )
        ],
    }
    payload.update(overrides)
    return ToolResult.model_validate(payload)


def _cells(first_row: int, values: list[list[object]], columns: str = "ABC") -> list:
    return [
        [
            {"coordinate": f"{col}{first_row + r}", "value": v, "displayed": v, "formula": None}
            for col, v in zip(columns, row)
        ]
        for r, row in enumerate(values)
    ]


def _read(cells: list, rng: str, **payload: object) -> ToolResult:
    return _tool(result={"sheet": "People", "range": rng, "cells": cells, **payload})


def test_a_small_read_is_a_grid_with_coordinates_recoverable() -> None:
    observation = normalize_tool(_tool())

    assert observation.source is ObservationSource.TOOL
    assert observation.status is ResultState.SUCCESS
    assert observation.facts[0] == "People!G3:G4 (row | G)"
    assert "3 | 92" in observation.facts
    # A formula without a cached value is reported as such, never as zero.
    assert any("People!G4" in fact and "no cached value" in fact for fact in observation.facts)
    assert observation.provenance[0].range_ref == "G3:G4"


def test_the_grid_keeps_blank_zero_and_not_available_apart() -> None:
    observation = normalize_tool(_read(_cells(7, [[None, 0, "N/A"]]), "A7:C7"))

    assert "7 |  | 0 | N/A" in observation.facts


def test_the_grid_escapes_separators_inside_values() -> None:
    observation = normalize_tool(_read(_cells(2, [["a|b", "line\nbreak", 1]]), "A2:C2"))

    assert "2 | a\\|b | line\\nbreak | 1" in observation.facts


def test_merged_ranges_are_listed_once() -> None:
    cells = _cells(1, [["Name", "Name", "Score"]])
    for cell in cells[0][:2]:
        cell["merged_with"] = "A1:B1"

    observation = normalize_tool(_read(cells, "A1:C1"))

    assert [fact for fact in observation.facts if fact.startswith("merged")] == ["merged: A1:B1"]


def _big_table() -> list:
    rows: list[list[object]] = [["Region", "Amount"], ["", "(VND)"]]
    rows += [[f"R{i}", i] for i in range(3, 60)]
    rows += [[None, None], ["Total", None]]
    cells = _cells(1, rows, columns="AB")
    total = cells[-1][1]
    total.update(formula="=SUM(B3:B59)", displayed=1767, value=None)
    return cells


def test_a_large_read_becomes_a_preview_that_keeps_both_ends() -> None:
    observation = normalize_tool(_read(_big_table(), "A1:B61"), max_inline_cells=40)
    text = "\n".join(observation.facts)

    assert observation.status is ResultState.TRUNCATED
    assert "61 rows × 2 columns" in text
    assert "1 | Region | Amount" in text  # header rows first
    assert "61 | Total | 1767" in text  # the total at the bottom survives
    assert "rows omitted" in text
    assert "blank rows: 60" in text
    assert "rows with formulas: 61" in text
    assert "B61: formula =SUM(B3:B59), cached 1767" in text
    assert any("code" in note and "wb." in note for note in observation.uncertainties)
    assert len(text) < 2_000


def test_a_preview_profiles_every_column_over_all_rows() -> None:
    observation = normalize_tool(_read(_big_table(), "A1:B61"), max_inline_cells=40)
    profiles = [fact for fact in observation.facts if fact.startswith("column ")]

    assert len(profiles) == 2
    amount = next(fact for fact in profiles if fact.startswith("column B"))
    assert "min 3" in amount and "max 1767" in amount


def test_a_large_dataframe_read_is_previewed_the_same_way() -> None:
    payload = {
        "sheet": "People",
        "header_row": 1,
        "range": "A2:B301",
        "columns": ["Region", "Amount"],
        "rows": [[f"R{i}", i] for i in range(2, 302)],
        "shape": [300, 2],
    }

    observation = normalize_tool(
        _tool(tool_name="get_sheet_as_dataframe", result=payload), max_inline_cells=40
    )
    text = "\n".join(observation.facts)

    assert "(row | Region | Amount)" in text
    assert "2 | R2 | 2" in text and "301 | R301 | 301" in text
    assert "rows omitted" in text


def test_many_search_matches_are_capped_with_a_count() -> None:
    matches = [{"reference": f"People!A{i}", "value": "x", "displayed": "x"} for i in range(1, 301)]

    observation = normalize_tool(
        _tool(tool_name="search", result={"matches": matches}), max_inline_cells=40
    )

    assert len(observation.facts) <= 41
    assert observation.facts[-1].startswith("… 260 more matches")


def test_tool_error_becomes_an_uncertainty_and_a_next_question() -> None:
    observation = normalize_tool(
        _tool(ResultState.ERROR, result=None, error="sheet 'Peple' does not exist", provenance=[])
    )

    assert observation.status is ResultState.ERROR
    assert observation.error == "sheet 'Peple' does not exist"
    assert observation.facts == []
    assert observation.next_question


def test_empty_tool_result_asks_for_a_different_read() -> None:
    observation = normalize_tool(
        _tool(ResultState.EMPTY, result={"sheet": "People", "range": "Z90", "cells": [[]]})
    )

    assert observation.status is ResultState.EMPTY
    assert observation.uncertainties
    assert observation.next_question


def test_the_character_budget_is_still_a_hard_safety_net() -> None:
    wide = _cells(1, [["x" * 300, "y" * 300, "z" * 300] for _ in range(10)])
    observation = normalize_tool(_read(wide, "A1:C10"), max_chars=500)

    assert observation.status is ResultState.TRUNCATED
    assert sum(len(fact) for fact in observation.facts) <= 500
    assert any("truncated" in note.lower() for note in observation.uncertainties)
    assert observation.provenance[0].range_ref == "G3:G4"


def test_sandbox_result_is_labeled_agent_computed() -> None:
    observation = normalize_sandbox(
        SandboxResult(
            state=ResultState.SUCCESS,
            execution_id="exec_3",
            stdout="99\n",
            result={"max_score": 99},
            computation_ids=["calc_1"],
            facade_calls=[{"method": "sheet", "workbook_id": WORKBOOK_ID, "sheet_name": "People"}],
        ),
        code_path="turns/03_t3_execute_code/code.py",
    )

    assert observation.source is ObservationSource.SANDBOX
    assert any("agent-computed" in fact for fact in observation.facts)
    assert any("calc_1" in fact for fact in observation.facts)
    kinds = {reference.kind for reference in observation.provenance}
    assert ProvenanceKind.COMPUTATION in kinds
    assert ProvenanceKind.ARTIFACT in kinds
    assert observation.provenance[0].artifact_path == "turns/03_t3_execute_code/code.py"


def test_sandbox_error_keeps_the_message_and_asks_to_fix_the_code() -> None:
    observation = normalize_sandbox(
        SandboxResult(
            state=ResultState.ERROR,
            execution_id="exec_4",
            error="KeyError: 'Scores'",
        )
    )

    assert observation.status is ResultState.ERROR
    assert observation.error == "KeyError: 'Scores'"
    assert observation.next_question


def _write_exploration(directory: Path, records: list[dict[str, str]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "evidence.filtered.json").write_text(json.dumps(records), encoding="utf-8")
    return directory


def test_exploration_bridge_keeps_only_kept_records_as_leads(tmp_path: Path) -> None:
    directory = _write_exploration(
        tmp_path / "exploration",
        [
            {
                "keyword": "score",
                "source_range": "People!G1",
                "description": "Score column.",
                "status": "keep",
                "reason": "",
            },
            {
                "keyword": "noise",
                "source_range": "People!A1",
                "description": "Irrelevant.",
                "status": "drop",
                "reason": "off topic",
            },
        ],
    )

    observations = load_exploration(directory, WORKBOOK_ID)

    assert len(observations) == 1
    lead = observations[0]
    assert lead.source is ObservationSource.EXPLORATION
    assert lead.summary == "score"
    assert lead.facts == ["Score column."]
    assert lead.provenance[0].sheet == "People"
    assert lead.provenance[0].range_ref == "G1"
    assert "exact read" in lead.uncertainties[0]


def test_exploration_bridge_accepts_quoted_sheet_names(tmp_path: Path) -> None:
    directory = _write_exploration(
        tmp_path / "exploration",
        [
            {
                "keyword": "vùng",
                "source_range": "'Doanh thu quý 1'!A1",
                "description": "d",
                "status": "keep",
                "reason": "",
            }
        ],
    )

    assert load_exploration(directory, WORKBOOK_ID)[0].provenance[0].sheet == "Doanh thu quý 1"


@pytest.mark.parametrize("content", [None, "not json", '{"not": "a list"}'])
def test_missing_or_malformed_exploration_fails_loudly(tmp_path: Path, content: str | None) -> None:
    directory = tmp_path / "exploration"
    directory.mkdir()
    if content is not None:
        (directory / "evidence.filtered.json").write_text(content, encoding="utf-8")

    with pytest.raises(ExplorationError):
        load_exploration(directory, WORKBOOK_ID)


def test_a_previewed_sandbox_result_is_marked_partial_but_the_run_complete() -> None:
    observation = normalize_sandbox(
        SandboxResult(
            state=ResultState.SUCCESS,
            execution_id="exec_5",
            result={
                "preview": True,
                "type": "DataFrame",
                "shape": [1000, 7],
                "head": [],
                "tail": [],
            },
        )
    )

    assert observation.status is ResultState.TRUNCATED
    assert any("bound in the session" in note for note in observation.uncertainties)


def test_ranges_code_read_through_wb_are_workbook_provenance() -> None:
    # The answer judge is shown the turns behind the cells an answer cites; a code step
    # that read those cells must say so.
    observation = normalize_sandbox(
        SandboxResult(
            state=ResultState.SUCCESS,
            execution_id="exec_2",
            result="Nga",
            facade_calls=[
                {
                    "method": "range",
                    "workbook_id": WORKBOOK_ID,
                    "range_ref": "B13:D13",
                    "sheet_name": "People",
                },
                {"method": "sheet", "workbook_id": WORKBOOK_ID, "sheet_name": "People"},
            ],
        )
    )

    workbook = [p for p in observation.provenance if p.kind is ProvenanceKind.WORKBOOK]
    assert [(p.sheet, p.range_ref) for p in workbook] == [("People", "B13:D13")]
