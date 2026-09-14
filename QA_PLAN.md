# QA Implementation Plan

## Objective

Implement a small ReAct-style QA agent that answers questions about a workbook by alternating
between reasoning, sandboxed Python execution, observations, and a final answer.

The first implementation milestone exposes exactly two read-only workbook tools:

1. `load_dataframe` loads one worksheet from the run's workbook into a pandas `DataFrame`.
2. `inspect_range` reads cell values from a specified A1 range.

No other QA tools or write operations are part of this milestone.

## Constraints

- Follow `CODING_STANDARDS.md`: typed Python, Google-style docstrings, specific exceptions,
  Black and Ruff with a 100-character line length, isort, mypy, focused tests, and at least 80%
  coverage.
- Follow KISS and YAGNI. Add only the agent loop, sandbox boundary, response parsing, and the two
  tools needed for this milestone.
- Treat the workbook, user question, model response, and generated code as untrusted input.
- Keep the source workbook read-only. The QA agent must not alter or save it.
- The host application selects and validates the workbook path. Model-generated code never
  receives a general-purpose path or filesystem API.
- Reset sandbox state for every QA run and enforce bounded turns, execution time, memory, and
  captured output.

## Reference Pattern

Use `SheetBrain/modules/execution.py` as a behavioral reference, without copying its unrelated
Excel editing features.

Retain these ideas:

- A bounded multi-turn loop: model response, parsed action, sandbox execution, observation, then
  the next model turn.
- A strict response contract with either a code action or a final answer.
- A deny-by-default execution environment.
- An explicit allowlist built from known callable tools, rather than exposing all application
  globals.
- Per-run sandbox reset and a maximum turn count.
- Execution errors returned as safe observations so the model can recover without receiving
  internal paths, secrets, or stack traces.

Do not carry over the reference implementation's search, formatting, mutation, charting, save,
or formula helpers.

## Initial Flow

```text
question + workbook + optional exploration evidence
  -> build QA prompt
  -> model Thought + Python action
  -> validate and execute action in sandbox
  -> append Observation to conversation
  -> repeat until model returns Final Answer or the turn limit is reached
```

The prompt must describe only the two available tools and require exactly one response form per
turn:

```text
Thought: <reasoning>
Action: <one Python code block>
```

or:

```text
Thought: <reasoning>
Final Answer: <answer grounded in observations>
```

The parser rejects responses that mix an action and final answer, contain multiple code blocks,
or do not match either form. A format error becomes an observation/reminder and consumes a turn.

## Tool Contracts

### `load_dataframe`

```python
def load_dataframe(sheet_name: str | None = None) -> pandas.DataFrame:
    """Load the active or named worksheet as a DataFrame."""
```

- The workbook is bound to the tool when the QA run starts; there is no path argument.
- A workbook can contain multiple worksheets, while a `DataFrame` is rectangular. Therefore one
  worksheet is loaded per call: omit `sheet_name` for the active sheet or pass an exact sheet
  name.
- Use the worksheet's first row as column labels and subsequent rows as records.
- Preserve empty cells as missing values and do not infer domain-specific types or table
  boundaries in this milestone.
- Raise a specific validation error for an unknown sheet.
- Add pandas as a direct project dependency because `DataFrame` is part of the public tool
  contract.

### `inspect_range`

```python
def inspect_range(
    range_ref: str,
    sheet_name: str | None = None,
) -> list[list[object | None]]:
    """Read values from an A1 range on the active or named worksheet."""
```

- Return a two-dimensional row-major list, including a one-cell range as `[[value]]`.
- Omit `sheet_name` for the active sheet or pass an exact sheet name.
- Validate the sheet name, A1 syntax, worksheet bounds, and a fixed maximum cell count before
  reading.
- Return values only; do not expose cell objects, workbook objects, formulas APIs, formatting, or
  mutation methods.
- Raise specific validation errors whose messages are safe to return to the agent.

Both tools use the same read-only workbook session so their sheet selection and displayed values
remain consistent during a run.

## Sandbox Boundary

Execute generated Python in an isolated sandbox process with a deny-by-default policy. The
sandbox receives only:

- the callable names `load_dataframe` and `inspect_range`;
- a deliberately small set of safe Python built-ins required for basic analysis; and
- local variables created during the current action.

Reject imports, filesystem access, network access, subprocesses, environment access, dynamic
code execution, and access to non-allowlisted application objects. Do not treat name filtering
alone as a sandbox. Enforce the boundary outside the generated Python process and terminate work
that exceeds the configured time, memory, or output limits.

Tool registration must contain exactly:

```python
{"load_dataframe": load_dataframe, "inspect_range": inspect_range}
```

DataFrame operations are permitted only on the in-memory object returned by `load_dataframe`;
they must not provide a route back to arbitrary Python imports or I/O.

## Expected `config.yaml`

Keep the existing project, model, exploration, retrieval, runtime, and rendering sections. The QA
agent reuses `models.llm`; add only this QA-specific section:

```yaml
qa:
  max_turns: 10
  max_code_chars: 10000
  max_observation_chars: 20000
  sandbox:
    timeout_seconds: 10
    memory_limit_mb: 512
  tools:
    max_range_cells: 10000
```

- `max_turns` bounds the complete Thought/Action/Observation loop.
- `max_code_chars` rejects oversized model-generated actions before sandbox execution.
- `max_observation_chars` bounds captured output returned to the model.
- `sandbox.timeout_seconds` and `sandbox.memory_limit_mb` apply to each isolated execution.
- `tools.max_range_cells` limits `inspect_range`; it must be a positive integer.

Validate every QA limit as a positive integer and reject unknown keys under `qa` so
misconfiguration is visible. The tool allowlist is intentionally not configurable: the first
milestone always registers exactly `load_dataframe` and `inspect_range`.

## Planned Modules

```text
src/autotab/qa/
  agent.py       # bounded ReAct conversation loop
  parser.py      # strict action/final-answer response parsing
  prompts.py     # versioned QA system and turn prompts
  sandbox.py     # isolated execution boundary and limits
  tools.py       # read-only workbook session and the two tool implementations
  types.py       # small typed action, observation, and result records
```

Keep responsibilities narrow: the agent coordinates, the parser validates model output, the
sandbox executes approved actions, and the tool session owns workbook access. Inject the existing
`ModelClient` protocol so tests can use deterministic fakes.

## Implementation Order

1. Add the minimal QA configuration defaults and validation shown above.
2. Add pandas and implement the read-only workbook session plus the two tool contracts.
3. Add focused tests for worksheet selection, DataFrame headers and missing values, scalar and
   rectangular ranges, invalid sheet names, invalid ranges, and the range-size limit.
4. Implement the isolated sandbox with exactly the two registered tools and verify that imports,
   file reads, network/process access, and unregistered callables are rejected.
5. Implement the strict response parser and bounded ReAct loop.
6. Add an integration test with a fake model: inspect or load data, receive the observation, then
   return a grounded final answer.
7. Run the repository quality gates and keep coverage at or above 80%.

## Exit Criteria

This milestone is complete when:

- A QA run can answer a sample workbook question through at least one Thought/Action/Observation
  cycle and a final answer.
- Generated code runs only inside the sandbox and can call only `load_dataframe` and
  `inspect_range`.
- Both tools are read-only, workbook-scoped, typed, documented, and covered by focused tests.
- The expected QA configuration is loaded, validated, and covered by focused tests.
- Invalid model formats and tool inputs fail safely and deterministically.
- The source workbook remains byte-for-byte unchanged after successful and failed runs.
- These checks pass:

```bash
uv run pytest --cov=src/autotab --cov-fail-under=80
uv run ruff check .
uv run black --check .
uv run isort --check-only .
uv run mypy src
```

## Explicitly Deferred

- Any third tool, including search, formatting, workbook mutation, save, formula, chart, or shell
  helpers.
- Workbook-writing or repair workflows.
- Multi-agent orchestration, retries beyond simple format recovery, long-term memory, and
  persistent conversation state.
- Automatic table detection, semantic retrieval, plotting, and report generation.
- Optimizations or abstractions not required by the first two-tool QA path.
