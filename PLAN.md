# SheetBrain QA Integration Plan

## Goal

Create a `qa-sheetbrain` branch where AutoTab's current QA stage is removed and replaced by the QA phase from Microsoft SheetBrain.

SheetBrain's understanding output will be replaced directly with AutoTab's `exploration.md`. SheetBrain's execution, tools, code execution, validation, and retry loop will otherwise remain unchanged.

## Source

- Repository: `https://github.com/microsoft/SheetBrain`
- Pinned commit: `68fdda3c52603ce4f67b22f32c787c5cdcc1c59d`
- License: MIT

Use the pinned commit rather than copying from a moving branch.

## Target Flow

```text
workbook + query
        |
        v
AutoTab exploration
        |
        v
exploration.md
        |
        v
SheetBrain understanding_output variable
        |
        v
unchanged SheetBrain execution
        |
        v
unchanged SheetBrain validation
        |
        +--> passed: return answer
        |
        +--> failed: unchanged feedback and re-execution loop
```

No adapter class is needed. Read `exploration.md` and assign its contents directly to the variable that SheetBrain calls `understanding_output`.

## Implementation Steps

### 1. Create the branch safely

The shared worktree currently contains another agent's uncommitted edits. After those changes have been committed or otherwise preserved:

```text
git switch -c qa-sheetbrain
```

Do not stash, reset, discard, or overwrite the other agent's work.

### 2. Remove AutoTab's QA stage

Recheck the repository after the other agent finishes, then remove all AutoTab-authored QA implementation:

- QA modules, prompts, tools, execution code, validation code, and orchestration.
- QA imports and CLI wiring.
- QA-specific configuration and validation in `src/autotab/config.py`.
- The old `qa:` block in `config.example.yaml`.
- QA tests written for the removed implementation.
- `QA_PLAN.md` once it is obsolete.

Keep the exploration stage unchanged.

### 3. Copy SheetBrain QA directly

Copy the SheetBrain files required by its QA flow from the pinned commit. This includes its agent orchestration and the dependencies used by execution and validation:

- `core/agent.py`
- `modules/execution.py`
- `modules/validation.py`
- `modules/safe_execution.py`
- `modules/response_parser.py`
- `utils/excel_toolkit.py`
- `utils/excel_security.py`
- `utils/logger.py`
- `config/settings.py`
- Required `__init__.py` files
- Upstream `LICENSE`

Do not copy SheetBrain's tests.

Keep the copied files unchanged except for the explicit understanding replacement in `core/agent.py` and any minimal import-path changes required to place them inside AutoTab.

Do not rewrite SheetBrain modules into AutoTab-style equivalents. In particular, preserve:

- The complete execution system prompt and user prompt.
- Excel helper functions and their limits.
- The restricted code interpreter.
- Model response parsing.
- The validation prompt and response parsing.
- Validation feedback injection on later attempts.
- Stop conditions, confidence values, result fields, and iteration behavior.

### 4. Replace SheetBrain understanding directly

Do not run or copy SheetBrain's `UnderstandingModule`.

In SheetBrain's agent flow, replace this operation:

```python
understanding_output = self.understanding_module.analyze(user_question, table_image)
```

with the equivalent of:

```python
understanding_output = exploration_path.read_text(encoding="utf-8")
```

The integration must pass the exploration artifact path into the SheetBrain agent or its `run` method. The exact API shape can follow the surrounding AutoTab code, but the stored value must be the unmodified contents of `exploration.md`.

Requirements:

- Read the file only after exploration completes.
- Preserve its text exactly, including headings and trailing newline.
- Do not summarize, truncate, decorate, or add instructions.
- Do not make a SheetBrain understanding-model call.
- Fail clearly if the exploration file is missing or empty.
- Continue passing `understanding_output` into SheetBrain's existing execution and validation loop exactly where upstream does.

The execution prompt may keep SheetBrain's existing `**Understanding Context:**` heading. Only the value underneath it changes to AutoTab exploration output.

### 5. Wire the combined pipeline

Add a thin AutoTab pipeline that:

1. Runs `ExplorationPipeline` with the workbook and query.
2. Receives the generated `exploration.md` path.
3. Starts the copied SheetBrain agent with the same workbook.
4. Supplies the exploration path for the direct `understanding_output` assignment.
5. Runs SheetBrain execution and validation unchanged.
6. Saves the SheetBrain result as the QA output.

Expose the combined flow through an explicit command such as:

```text
autotab qa --config config.yaml --workbook path/to/file.xlsx --query "..."
```

Keep `autotab explore` available for exploration-only evaluation.

SheetBrain operates on one workbook, so the QA command should require exactly one workbook rather than changing SheetBrain's tool behavior.

### 6. Configuration and dependencies

Remove the old AutoTab QA configuration instead of translating it into SheetBrain behavior.

Use SheetBrain's defaults where possible:

- `max_turns`: `3`
- `total_token_budget`: `5000`
- validation enabled
- diagnostics disabled by default

Map AutoTab's configured OpenAI-compatible endpoint and model into SheetBrain's `api_key`, `base_url`, and `deployment` fields.

Add only dependencies required by the copied SheetBrain code that AutoTab does not already have, such as `openai`, `numpy`, `tiktoken`, and `matplotlib`.

## Verification

- Confirm the old AutoTab QA implementation and configuration are gone.
- Diff copied execution, validation, tools, parser, and safe-execution files against the pinned SheetBrain commit; they should have no behavioral changes.
- Confirm the only intentional SheetBrain agent-flow change is replacing its understanding call with reading `exploration.md`.
- Confirm `understanding_output` exactly equals the exploration file contents.
- Confirm no SheetBrain understanding-model request occurs.
- Confirm validation failures still feed SheetBrain's original feedback into the next execution attempt.
- Run the existing AutoTab test suite and add small AutoTab integration tests for the direct replacement and combined flow.
- Do not copy or maintain SheetBrain's upstream tests.

## Acceptance Criteria

- Implementation is on `qa-sheetbrain` after concurrent work is preserved.
- AutoTab's previous QA stage is completely removed.
- QA execution, tools, parsing, validation, and retries come directly from pinned SheetBrain.
- AutoTab's exact `exploration.md` contents are assigned directly to SheetBrain's `understanding_output`.
- SheetBrain's understanding module is not run.
- Exploration-only and exploration-plus-QA commands both work.
- No SheetBrain tests are copied.
