# Coding Standards

This document defines the working conventions for AutoTab, a Python research
project that explores spreadsheet workbooks with retrieval and language-model
assistance. The goal is code that is easy to inspect, reproduce, test, and
remove when an experiment ends.

## General Principles

### SOLID Principles

- Single Responsibility - Each class/function should have a single responsibility.
- Open/Closed - Open for extension, closed for modification.
- Liskov Substitution - Subclasses should be substitutable for their base classes.
- Interface Segregation - Prefer multiple small interfaces over one large interface.
- Dependency Inversion - Depend on abstractions, not concrete implementations.

### DRY (Don't Repeat Yourself)

- Avoid code duplication.
- Extract common logic into shared functions/classes.
- Use inheritance/composition appropriately.

### KISS (Keep It Simple, Stupid)

- Prioritize simplicity over complexity.
- Prefer readable code over "clever" code.
- Avoid premature optimization.

### YAGNI (You Aren't Gonna Need It)

- Implement only what is needed now.
- Don't build features "for the future."
- Refactor when necessary, rather than prematurely.

## Guiding principles

- Prefer a small, explicit design over clever abstractions.
- Keep domain logic independent from providers, file formats, and the CLI.
- Make research decisions visible: name assumptions, record configuration,
  preserve intermediate evidence, and avoid silent fallbacks.
- Treat workbook content, model output, configuration files, and external
  responses as untrusted input.
- Optimize only after measuring. Correctness and reproducibility come before
  throughput.
- Follow SOLID where it improves a boundary; do not introduce interfaces or
  inheritance without a concrete second implementation or testing benefit.

## Repository layout

Use the `src/` layout and keep production code under `src/autotab/`:

```text
src/autotab/
  cli.py                 # command-line wiring only
  config.py              # loading and validation
  exploration/           # ingestion, retrieval, rendering, evidence, output
  models/                # provider clients and model contracts
  artifacts/             # run manifests and artifact storage
  utils/                 # small, dependency-light shared helpers
  qa/                    # harness agent modules and QA logic
tests/                   # tests mirroring the source package
...
```

Modules should have one clear responsibility. A pipeline may coordinate
stages, but stage-specific behavior belongs in the stage module. Keep imports
acyclic and avoid importing the CLI from library code.

## Python version and tooling

- Target Python 3.11 or newer. Use `uv` to create environments, resolve
  dependencies, and run commands.
- Format with Black using a 100-character line length.
- Run Ruff for linting, isort-compatible import ordering, and simple fixes.
- Run mypy over `src` and keep public boundaries typed.
- Add runtime dependencies to `pyproject.toml`; do not rely on packages that
  happen to be installed globally. Keep development tools in the dev group.

The normal local check is:

```bash
uv run ruff check .
uv run black --check .
uv run isort --check-only .
uv run mypy src
uv run pytest --cov=autotab --cov-report=term-missing
```

## Style and naming

- Use four spaces and UTF-8 source files; prefer ASCII in identifiers,
  configuration keys, and generated artifact names.
- Use `snake_case` for functions, methods, variables, and modules; `PascalCase`
  for classes; `UPPER_SNAKE_CASE` for module constants.
- Name booleans with `is_`, `has_`, `can_`, or `should_` where that improves
  readability.
- Keep functions short enough to understand without scrolling through several
  unrelated concerns. Extract a helper when it has a name that clarifies the
  algorithm, not merely to reduce line count.
- Prefer guard clauses and straightforward control flow. Avoid hidden mutation,
  clever one-liners, and mutable default arguments.
- Use `pathlib.Path` for paths, f-strings for interpolation, and context
  managers for files, temporary directories, and resources.

## Types and data models

- Annotate every function argument and return value, including private helpers.
- Use built-in generics (`list[str]`, `dict[str, Any]`) and `X | None` syntax.
- Use dataclasses or typed models for records crossing stage boundaries. Avoid
  passing loosely shaped dictionaries through the pipeline.
- Use `Literal`, enums, or constrained types for finite states such as
  `keep`/`drop`/`uncertain`.
- Use `Any` only at unavoidable untyped boundaries (third-party libraries,
  decoded JSON, or provider responses), and validate immediately before the
  value enters typed code.
- Do not use truthiness to distinguish a missing value from a valid zero, empty
  string, or empty collection when that distinction matters.
- Keep serialization formats stable. Add explicit schema/version fields to
  manifests and snapshots that may be consumed by later runs or tools.

Public classes and functions must have Google-style docstrings when their
purpose, side effects, or failure modes are not obvious:

```python
def select_viewport(row: int, column: int, size: int) -> str:
    """Return the clipped odd-sized range centered on a worksheet cell.

    Args:
        row: One-based worksheet row.
        column: One-based worksheet column.
        size: Requested odd viewport size.

    Returns:
        An A1-style range string.

    Raises:
        ValueError: If `size` is not a positive odd number.
    """
```

## Research and experiment hygiene

- Make randomness explicit. Accept a seed in experiment configuration and set
  it for every library that uses randomness. If a provider is nondeterministic,
  record that fact.
- Record the run ID, input workbook hashes, configuration snapshot, dependency
  or application version, model names, prompt versions, and relevant thresholds
  in the run manifest.
- Keep prompts versioned in source files. Do not construct important prompts by
  scattering string fragments across the pipeline.
- Separate exploratory notebooks or scripts from reusable package code. A
  finding belongs in `src/` only after it has a clear contract and a test.
- Never overwrite source workbooks. Render or transform a temporary copy and
  write outputs below the configured artifact root.
- Preserve enough intermediate evidence to explain a result, including source
  ranges, component retrieval scores, model responses, and filtering decisions.
- Order outputs deterministically. Stable sorting must not depend on hash order,
  task completion order, or filesystem enumeration.

## Spreadsheet and untrusted data handling

- Treat cell values, formulas, sheet names, YAML, JSON, and model responses as
  data, not executable instructions.
- Validate configuration at the boundary: types, ranges, paths, timeouts,
  concurrency limits, and mutually dependent options. Raise an actionable
  error that identifies the field and expected value.
- Preserve workbook-native formatting and formulas where rendering requires it;
  do not replace display values with ad hoc string formatting.
- Handle merged cells, hidden sheets, blank cells, dates, percentages, and
  currencies deliberately. Covered merged cells are not independent findings.
- Sanitize user-controlled names before using them in paths, Markdown, logs, or
  subprocess arguments. Never build shell commands by interpolating raw input.
- Secrets come only from environment variables or a secret manager. Never put
  credentials in YAML, prompts, tests, fixtures, manifests, or committed logs.

## Boundaries, providers, and concurrency

- Define small protocols or adapter classes around LLM, VLM, embedding,
  rendering, and filesystem services. Core ranking and validation logic must be
  testable without a network call or LibreOffice installation.
- Keep provider-specific request and response translation inside the adapter.
  The pipeline should depend on a typed internal contract.
- Use async I/O for model and network work, with explicit timeouts, bounded
  retries, and a semaphore for concurrency. Never use unbounded `gather`.
- Do not mutate a shared `openpyxl` workbook concurrently. Give each render an
  isolated workbook copy, temporary profile, and output directory.
- Preserve deterministic result ordering even when work completes concurrently.
  Collect typed results, sort them, then write shared artifacts once.
- Record fallback behavior (for example lexical-only retrieval) in the
  manifest; never silently change ranking semantics.

## Errors, logging, and observability

- Raise domain-specific exceptions at library boundaries and add context with
  `raise ... from exc` when wrapping a lower-level error.
- Error messages should say what failed, which input or configuration field was
  involved, and how the user can correct it. Do not expose secrets or full
  workbook contents in errors.
- Use structured, concise logs. Include run ID, workbook/sheet/range, stage,
  and finding ID where relevant. Avoid logging every cell or model prompt by
  default.
- A failure in one independent finding should be represented as a typed warning
  or failed result when the run can continue. Fail fast for invalid config,
  missing required executables, or corrupted primary inputs.

## Testing standards

- Every bug fix adds a regression test. Prefer focused unit tests for pure logic
  and contract tests for adapters.
- Test normal, boundary, invalid, and failure cases. Important boundaries
  include merged ranges, clipped viewports, empty values, malformed model JSON,
  threshold filtering, retries, and concurrency limits.
- Keep tests deterministic and offline by default. Mock model clients and use
  small committed workbook fixtures; do not call paid services in the test
  suite.
- Use property-based tests when an invariant spans many ranges, scores, or
  normalization inputs and examples alone are insufficient.
- Test generated Markdown and manifests as exact, deterministic text where
  practical. Verify that dropped findings stay in traces but not user output.
- Aim for at least 80% coverage, while prioritizing pipeline decisions and
  validation paths over trivial getters.

## Documentation and review

- Update `README.md` when the CLI, configuration, artifacts, or setup changes.
- Document non-obvious algorithms, invariants, and research trade-offs close to
  the code. Comments should explain why, not restate what the code does.
- Keep examples runnable and free of secrets. Prefer `config.example.yaml` for
  configuration documentation.
- Before merging, review correctness, reproducibility, data-loss risks,
  security, performance limits, and missing tests. Include the verification
  commands run and any checks that could not be run.

## Git and change hygiene

- Make small, cohesive commits with imperative messages.
- Do not commit generated `outputs/`, temporary render files, local configs, or
  credentials.
- Avoid drive-by formatting or renaming in unrelated files.
- A pull request should explain the user-visible behavior, research assumptions,
  compatibility impact, and how to reproduce the result.
