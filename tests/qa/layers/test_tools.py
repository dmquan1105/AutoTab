"""Read-only workbook registry behavior; see ``specs/layers/tools.md``."""

import hashlib
from pathlib import Path
from typing import Any

import pytest
from openpyxl import Workbook

from autotab.qa.layers.tools import (
    AttributeName,
    ToolRegistry,
    WorkbookReader,
)
from autotab.qa.schemas import ResultState, ToolResult

WORKBOOK_ID = "workbook_1"


@pytest.fixture
def workbook_path(tmp_path: Path) -> Path:
    book = Workbook()
    revenue = book.active
    revenue.title = "Revenue"
    revenue["A1"] = "Region"
    revenue["B1"] = "Amount"
    revenue["A2"] = "North"
    revenue["B2"] = 50
    revenue["A3"] = "South"
    revenue["B3"] = 40
    revenue["A4"] = "Total"
    revenue["B4"] = "=SUM(B2:B3)"
    revenue["A6"] = "Q1 Summary"
    revenue.merge_cells("A6:B6")
    revenue["B8"] = 0
    revenue["A9"] = "N/A"
    revenue["B2"].number_format = "#,##0"

    costs = book.create_sheet("Costs")
    costs["A1"] = "Region"
    costs["A2"] = "north"
    costs["A3"] = " North "

    notes = book.create_sheet("Notes")
    notes["A1"] = "internal"
    notes.sheet_state = "hidden"

    path = tmp_path / "book.xlsx"
    book.save(path)
    return path


@pytest.fixture
def registry(workbook_path: Path) -> ToolRegistry:
    reader = WorkbookReader(WORKBOOK_ID, workbook_path, max_range_cells=50)
    return ToolRegistry([reader])


def _dispatch(registry: ToolRegistry, tool: str, **arguments: Any) -> ToolResult:
    return registry.dispatch(tool, {"workbook_id": WORKBOOK_ID, **arguments})


def _cells(result: ToolResult) -> list[dict[str, Any]]:
    assert isinstance(result.result, dict)
    rows = result.result["cells"]
    assert isinstance(rows, list)
    return [cell for row in rows for cell in row]


def test_registry_exposes_exactly_the_v1_tools_with_generated_schemas(
    registry: ToolRegistry,
) -> None:
    descriptions = registry.descriptions()

    assert [item["name"] for item in descriptions] == [
        "inspect_range",
        "inspect_attributes",
        "search",
        "get_sheet_as_dataframe",
    ]
    for item in descriptions:
        assert item["description"].strip()
        assert item["schema"]["properties"]["workbook_id"]
        assert item["schema"]["additionalProperties"] is False


def test_unknown_tool_and_unknown_workbook_are_rejected_without_reading(
    registry: ToolRegistry,
) -> None:
    unknown_tool = registry.dispatch("execute_excel", {"workbook_id": WORKBOOK_ID})
    assert unknown_tool.state is ResultState.ERROR
    assert "execute_excel" in (unknown_tool.error or "")

    unknown_workbook = registry.dispatch(
        "inspect_range", {"workbook_id": "workbook_9", "range_ref": "A1"}
    )
    assert unknown_workbook.state is ResultState.ERROR
    assert "workbook_9" in (unknown_workbook.error or "")


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"range_ref": ""},
        {"range_ref": "A1", "sheet_name": "Revenue", "unexpected": 1},
        {"range_ref": "not-a-range"},
        {"range_ref": "A1", "sheet_name": "Missing"},
    ],
)
def test_invalid_arguments_fail_before_execution(
    registry: ToolRegistry, arguments: dict[str, Any]
) -> None:
    result = _dispatch(registry, "inspect_range", **arguments)

    assert result.state is ResultState.ERROR
    assert result.error


def test_inspect_range_reports_resolved_sheet_and_stable_shape(registry: ToolRegistry) -> None:
    single = _dispatch(registry, "inspect_range", range_ref="A2")
    matrix = _dispatch(registry, "inspect_range", range_ref="A1:B3", sheet_name="Revenue")

    assert single.state is ResultState.SUCCESS
    assert isinstance(single.result, dict)
    assert single.result["sheet"] == "Revenue"
    assert single.result["range"] == "A2"
    assert single.result["cells"] == [
        [
            {
                "coordinate": "A2",
                "value": "North",
                "displayed": "North",
                "formula": None,
                "data_type": "s",
                "number_format": "General",
            }
        ]
    ]

    assert isinstance(matrix.result, dict)
    assert len(matrix.result["cells"]) == 3
    assert [cell["coordinate"] for cell in matrix.result["cells"][0]] == ["A1", "B1"]
    assert matrix.provenance[0].sheet == "Revenue"
    assert matrix.provenance[0].range_ref == "A1:B3"


def test_formula_text_stays_separate_and_a_missing_cache_is_not_zero(
    registry: ToolRegistry,
) -> None:
    result = _dispatch(registry, "inspect_range", range_ref="B4")

    cell = _cells(result)[0]

    assert cell["formula"] == "=SUM(B2:B3)"
    assert cell["value"] is None
    assert cell["displayed"] is None


def test_blank_zero_and_not_available_stay_distinct(registry: ToolRegistry) -> None:
    # openpyxl round-trips a written "" to None, so a literal empty string cannot be
    # produced by this fixture; blank, zero, and "N/A" are the distinctions a
    # workbook can actually carry, and none of them may collapse into another.
    values = {
        cell["coordinate"]: cell["value"]
        for cell in _cells(_dispatch(registry, "inspect_range", range_ref="A7:B9"))
    }

    assert values["A7"] is None
    assert values["A8"] is None
    assert values["B8"] == 0
    assert values["A9"] == "N/A"


def test_covered_merged_cells_report_the_anchor_value(registry: ToolRegistry) -> None:
    cells = {
        cell["coordinate"]: cell
        for cell in _cells(_dispatch(registry, "inspect_range", range_ref="A6:B6"))
    }

    assert cells["A6"]["value"] == "Q1 Summary"
    assert cells["B6"]["value"] == "Q1 Summary"
    assert cells["B6"]["merged_with"] == "A6:B6"


def test_hidden_sheets_are_refused_unless_configured(workbook_path: Path) -> None:
    default = ToolRegistry([WorkbookReader(WORKBOOK_ID, workbook_path)])
    permissive = ToolRegistry([WorkbookReader(WORKBOOK_ID, workbook_path, include_hidden=True)])

    refused = _dispatch(default, "inspect_range", range_ref="A1", sheet_name="Notes")
    allowed = _dispatch(permissive, "inspect_range", range_ref="A1", sheet_name="Notes")

    assert refused.state is ResultState.ERROR
    assert "hidden" in (refused.error or "").lower()
    assert allowed.state is ResultState.SUCCESS
    assert "Notes" not in default.sheet_names(WORKBOOK_ID)


def test_oversized_range_is_refused_with_the_limit(registry: ToolRegistry) -> None:
    result = _dispatch(registry, "inspect_range", range_ref="A1:Z100")

    assert result.state is ResultState.ERROR
    assert "50" in (result.error or "")


def test_empty_range_reports_empty_without_inventing_values(registry: ToolRegistry) -> None:
    result = _dispatch(registry, "inspect_range", range_ref="D20:E21")

    assert result.state is ResultState.EMPTY
    assert all(cell["value"] is None for cell in _cells(result))


def test_inspect_attributes_accepts_only_registered_attributes(registry: ToolRegistry) -> None:
    result = _dispatch(
        registry,
        "inspect_attributes",
        range_ref="B2",
        attributes=[AttributeName.NUMBER_FORMAT.value, AttributeName.DATA_TYPE.value],
    )

    assert result.state is ResultState.SUCCESS
    assert _cells(result)[0] == {
        "coordinate": "B2",
        "number_format": "#,##0",
        "data_type": "n",
    }

    for attributes in ([], ["width"], ["formula", "formula"]):
        rejected = _dispatch(registry, "inspect_attributes", range_ref="B2", attributes=attributes)
        assert rejected.state is ResultState.ERROR


@pytest.mark.parametrize(
    ("search_type", "case_sensitive", "expected"),
    [
        ("partial", False, ["Costs!A2", "Costs!A3", "Revenue!A2"]),
        ("partial", True, ["Costs!A3", "Revenue!A2"]),
        ("whole", False, ["Costs!A2", "Revenue!A2"]),
        ("whole", True, ["Revenue!A2"]),
        ("strip", False, ["Costs!A2", "Costs!A3", "Revenue!A2"]),
        ("strip", True, ["Costs!A3", "Revenue!A2"]),
    ],
)
def test_search_matching_modes_are_deterministic(
    registry: ToolRegistry,
    search_type: str,
    case_sensitive: bool,
    expected: list[str],
) -> None:
    result = registry.dispatch(
        "search",
        {
            "workbook_id": WORKBOOK_ID,
            "value": "North",
            "search_type": search_type,
            "case_sensitive": case_sensitive,
        },
    )

    assert isinstance(result.result, dict)
    assert [match["reference"] for match in result.result["matches"]] == expected


def test_search_without_matches_is_empty(registry: ToolRegistry) -> None:
    result = registry.dispatch("search", {"workbook_id": WORKBOOK_ID, "value": "Neverland"})

    assert result.state is ResultState.EMPTY
    assert isinstance(result.result, dict)
    assert result.result["matches"] == []


def test_dataframe_records_selection_and_truncates_explicitly(registry: ToolRegistry) -> None:
    full = _dispatch(registry, "get_sheet_as_dataframe", sheet_name="Revenue")
    capped = _dispatch(registry, "get_sheet_as_dataframe", sheet_name="Revenue", max_rows=2)
    headerless = _dispatch(registry, "get_sheet_as_dataframe", sheet_name="Costs", header_row=None)

    assert isinstance(full.result, dict)
    assert full.result["columns"] == ["Region", "Amount"]
    assert full.result["header_row"] == 1
    assert full.result["rows"][0] == ["North", 50]
    assert full.result["shape"] == [len(full.result["rows"]), 2]

    assert capped.state is ResultState.TRUNCATED
    assert isinstance(capped.result, dict)
    assert len(capped.result["rows"]) == 2

    assert isinstance(headerless.result, dict)
    assert headerless.result["columns"] == ["A", "B"][: headerless.result["shape"][1]]
    assert headerless.result["rows"][0] == ["Region"]


def test_reads_never_modify_the_source_workbook(
    registry: ToolRegistry, workbook_path: Path
) -> None:
    before = hashlib.sha256(workbook_path.read_bytes()).hexdigest()

    _dispatch(registry, "inspect_range", range_ref="A1:B4")
    _dispatch(registry, "get_sheet_as_dataframe", sheet_name="Revenue")
    registry.dispatch("search", {"workbook_id": WORKBOOK_ID, "value": "North"})

    assert hashlib.sha256(workbook_path.read_bytes()).hexdigest() == before


def test_describe_lists_readable_sheets_with_extent_and_merges(workbook_path: Path) -> None:
    reader = WorkbookReader(WORKBOOK_ID, workbook_path)

    sheets = reader.describe()

    assert [sheet["name"] for sheet in sheets] == ["Revenue", "Costs"]
    revenue = sheets[0]
    assert revenue["dimensions"] == "A1:B9"
    assert revenue["merged"] == ["A6:B6"]
