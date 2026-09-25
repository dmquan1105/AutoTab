"""Strict Pydantic contracts shared across QA.

This module is the single validation authority for LLM payloads, persisted
artifacts, provenance, execution results, and verification reports. It performs no
model calls, workbook access, orchestration, or persistence; see
``specs/schemas.md``.
"""

from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated, Any, Final, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

SCHEMA_VERSION: Final = "1.0"
RUBRIC_VERSION: Final = "1.0"
LLM_JUDGE_RECHECK: Final = "LLM_JUDGE"

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class StrictModel(BaseModel):
    """Base for every QA contract: exact fields, no coercion, no inf/nan."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class RunStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class Phase(str, Enum):
    INITIALIZE = "INITIALIZE"
    OBSERVE = "OBSERVE"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    FINALIZE = "FINALIZE"
    FAIL = "FAIL"


class ActionType(str, Enum):
    TOOL = "tool"
    CODE = "code"
    ANSWER = "answer"


class ResultState(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    ERROR = "error"
    TRUNCATED = "truncated"


class ObservationSource(str, Enum):
    TOOL = "tool"
    SANDBOX = "sandbox"
    EXPLORATION = "exploration"


class FeedbackPriority(str, Enum):
    BLOCKING = "blocking"
    NON_BLOCKING = "non_blocking"


class FeedbackSource(str, Enum):
    DETERMINISTIC_CHECK = "deterministic_check"
    LLM_JUDGE = "llm_judge"


class ProvenanceKind(str, Enum):
    WORKBOOK = "workbook"
    COMPUTATION = "computation"
    ARTIFACT = "artifact"


def ensure_run_relative(path: str) -> str:
    """Return ``path`` when it stays inside the run root.

    Args:
        path: Candidate artifact path.

    Returns:
        The unchanged path.

    Raises:
        ValueError: If the path is empty, absolute, or escapes the run root.
    """
    if not path:
        raise ValueError("artifact paths must be non-empty")
    pure = PurePosixPath(path)
    if pure.is_absolute() or path.startswith("\\") or ":" in pure.parts[0]:
        raise ValueError(f"artifact paths must be run-relative, got {path!r}")
    if ".." in pure.parts:
        raise ValueError(f"artifact paths must not traverse upwards, got {path!r}")
    return path


def _reject_integer(value: Any) -> Any:
    """Reject JSON integers where the contract requires a float.

    Strict mode still widens ``1`` to ``1.0``; latency is declared as a float so a
    recorded duration cannot silently arrive with integer precision.
    """
    if isinstance(value, int):
        # Pydantic turns ValueError into ValidationError; a TypeError would escape.
        raise ValueError("expected a JSON float, got an integer")  # noqa: TRY004
    return value


RunRelativePath = Annotated[NonEmptyStr, AfterValidator(ensure_run_relative)]
StrictFloat = Annotated[float, BeforeValidator(_reject_integer)]


class ObserveResult(StrictModel):
    """The observe model's plan for the next execution turn.

    ``next_action`` is the kind of the one action EXECUTE must take for the objective;
    choosing it here keeps the decision of what to do next with the planner.
    """

    known_facts: list[str]
    uncertainties: list[str]
    execution_objective: NonEmptyStr
    next_action: ActionType
    rationale: NonEmptyStr


class Claim(StrictModel):
    """One checkable statement inside a candidate answer."""

    id: NonEmptyStr
    statement: NonEmptyStr
    value: JsonValue = None
    citations: list[NonEmptyStr]
    computation_id: NonEmptyStr | None

    @model_validator(mode="after")
    def _require_support(self) -> Claim:
        if not self.citations and self.computation_id is None:
            raise ValueError(f"claim {self.id!r} needs a citation, a computation ID, or both")
        return self


class CandidateAnswer(StrictModel):
    """The answer under verification, produced only by an answer action."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    answer_text: NonEmptyStr
    claims: list[Claim]


class ComputationInput(StrictModel):
    """One ordered input of a recorded computation."""

    reference: NonEmptyStr | None = None
    value: JsonValue = None


class ComputationRecord(StrictModel):
    """A replayable computation recorded from the sandbox."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    id: NonEmptyStr
    operation: NonEmptyStr
    inputs: list[ComputationInput]
    output: JsonValue = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    tolerance: float = 0.0


class _ActionBase(StrictModel):
    rationale: NonEmptyStr


class ToolReply(StrictModel):
    """EXECUTE's reply when the plan names a tool call; the harness builds the action."""

    tool_name: NonEmptyStr
    arguments: dict[str, JsonValue]
    rationale: str = ""


class AnswerReply(StrictModel):
    """EXECUTE's reply when the plan names an answer; the harness builds the action."""

    answer: CandidateAnswer
    rationale: str = ""


class ToolAction(_ActionBase):
    """One registered read-only workbook call."""

    action_type: Literal[ActionType.TOOL]
    tool_name: NonEmptyStr
    arguments: dict[str, JsonValue]


class CodeAction(_ActionBase):
    """One bounded run of generated Python in the run's sandbox session."""

    action_type: Literal[ActionType.CODE]
    code: NonEmptyStr


class AnswerAction(_ActionBase):
    """The only producer of a candidate answer."""

    action_type: Literal[ActionType.ANSWER]
    answer: CandidateAnswer


AgentAction = Annotated[ToolAction | CodeAction | AnswerAction, Field(discriminator="action_type")]


class ProvenanceReference(StrictModel):
    """Where a value came from: a workbook range, a computation, or an artifact."""

    kind: ProvenanceKind
    workbook_id: NonEmptyStr | None = None
    sheet: NonEmptyStr | None = None
    range_ref: NonEmptyStr | None = None
    value: JsonValue = None
    formula: NonEmptyStr | None = None
    computation_id: NonEmptyStr | None = None
    artifact_path: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _check_kind(self) -> ProvenanceReference:
        if self.kind is ProvenanceKind.WORKBOOK:
            if self.workbook_id is None:
                raise ValueError("workbook provenance requires workbook_id")
            if (self.sheet is None) != (self.range_ref is None):
                raise ValueError("workbook provenance needs both sheet and range_ref, or neither")
            self._forbid("workbook", computation_id=self.computation_id)
            self._forbid("workbook", artifact_path=self.artifact_path)
            return self

        self._forbid(self.kind.value, workbook_id=self.workbook_id)
        self._forbid(self.kind.value, sheet=self.sheet)
        self._forbid(self.kind.value, range_ref=self.range_ref)
        self._forbid(self.kind.value, formula=self.formula)
        if self.kind is ProvenanceKind.COMPUTATION:
            if self.computation_id is None:
                raise ValueError("computation provenance requires computation_id")
            self._forbid("computation", artifact_path=self.artifact_path)
            return self

        if self.artifact_path is None:
            raise ValueError("artifact provenance requires artifact_path")
        ensure_run_relative(self.artifact_path)
        self._forbid("artifact", computation_id=self.computation_id)
        return self

    @staticmethod
    def _forbid(kind: str, **fields: object) -> None:
        for name, value in fields.items():
            if value is not None:
                raise ValueError(f"{kind} provenance must not set {name}")


class _ResultBase(StrictModel):
    state: ResultState
    error: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _check_error(self) -> _ResultBase:
        if self.state is ResultState.ERROR and self.error is None:
            raise ValueError("an error result requires a non-empty error")
        if self.state is not ResultState.ERROR and self.error is not None:
            raise ValueError(f"a {self.state.value} result must not carry an error")
        return self


class ToolResult(_ResultBase):
    """The raw result of one registered tool call."""

    tool_name: NonEmptyStr
    arguments: dict[str, JsonValue]
    result: JsonValue = None
    provenance: list[ProvenanceReference] = Field(default_factory=list)


class SandboxResult(_ResultBase):
    """The raw result of one sandbox execution."""

    execution_id: NonEmptyStr
    stdout: str = ""
    stderr: str = ""
    result: JsonValue = None
    facade_calls: list[dict[str, JsonValue]] = Field(default_factory=list)
    computation_ids: list[NonEmptyStr] = Field(default_factory=list)


class Observation(StrictModel):
    """Normalized evidence handed to the next turn; never a correctness verdict."""

    source: ObservationSource
    status: ResultState
    summary: str
    facts: list[str] = Field(default_factory=list)
    result: JsonValue = None
    error: NonEmptyStr | None = None
    provenance: list[ProvenanceReference] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    next_question: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _check_error(self) -> Observation:
        if self.status is ResultState.ERROR and self.error is None:
            raise ValueError("an error observation requires a non-empty error")
        if self.status is not ResultState.ERROR and self.error is not None:
            raise ValueError(f"a {self.status.value} observation must not carry an error")
        return self


class ImprovementFeedback(StrictModel):
    """One material problem, its evidence, and how to recheck it."""

    code: NonEmptyStr
    source: FeedbackSource
    problem: NonEmptyStr
    evidence: list[NonEmptyStr] = Field(default_factory=list)
    action: NonEmptyStr
    priority: FeedbackPriority
    recheck: NonEmptyStr

    @model_validator(mode="after")
    def _check_recheck(self) -> ImprovementFeedback:
        is_judge = self.source is FeedbackSource.LLM_JUDGE
        if is_judge and self.recheck != LLM_JUDGE_RECHECK:
            raise ValueError(f"judge feedback must recheck {LLM_JUDGE_RECHECK}")
        if not is_judge and self.recheck == LLM_JUDGE_RECHECK:
            raise ValueError("deterministic feedback must recheck its own check ID")
        return self

    @property
    def is_blocking(self) -> bool:
        """Whether this item forces `FAIL` until it is rechecked."""
        return self.priority is FeedbackPriority.BLOCKING


def _blocking(feedback: list[ImprovementFeedback]) -> list[ImprovementFeedback]:
    return [item for item in feedback if item.is_blocking]


class DeterministicCheckResult(StrictModel):
    """One deterministic check over an execution result or the candidate answer."""

    id: NonEmptyStr
    status: RunStatus
    observed: JsonValue = None
    expected: JsonValue = None
    reason: NonEmptyStr


class DeterministicCheckReport(StrictModel):
    """Every deterministic check of one scope, plus their combined status."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    status: RunStatus
    checks: list[DeterministicCheckResult]

    @model_validator(mode="after")
    def _check_status(self) -> DeterministicCheckReport:
        expected = (
            RunStatus.PASS
            if all(check.status is RunStatus.PASS for check in self.checks)
            else RunStatus.FAIL
        )
        if self.status is not expected:
            raise ValueError(f"report status must be {expected.value} for these checks")
        return self


class LLMJudgeResult(StrictModel):
    """The judge's holistic verdict; the rubric itself stays in the prompt."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    rubric_version: Literal["1.0"] = RUBRIC_VERSION
    status: RunStatus
    confidence_score: float = Field(ge=0.0, le=1.0)
    reason: NonEmptyStr
    issues_found: list[NonEmptyStr] = Field(default_factory=list)
    improvement_feedback: list[ImprovementFeedback] = Field(default_factory=list)
    final_assessment: NonEmptyStr

    @model_validator(mode="after")
    def _check_verdict(self) -> LLMJudgeResult:
        blocking = _blocking(self.improvement_feedback)
        if self.status is RunStatus.FAIL:
            if not self.issues_found:
                raise ValueError("a failing judgement requires at least one issue")
            if not blocking:
                raise ValueError("a failing judgement requires at least one blocking item")
            return self
        if self.issues_found:
            raise ValueError("a passing judgement lists no issues")
        if blocking:
            raise ValueError("a passing judgement carries no blocking feedback")
        return self


class VerificationResult(StrictModel):
    """The combined verdict for one candidate answer."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    status: RunStatus
    reason: NonEmptyStr
    artifact: RunRelativePath
    improvement_feedback: list[ImprovementFeedback] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_verdict(self) -> VerificationResult:
        if self.status is RunStatus.PASS and _blocking(self.improvement_feedback):
            raise ValueError("a passing verification carries no blocking feedback")
        return self


class HistoryEvent(StrictModel):
    """One append-only record of what happened in a phase."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    event_id: NonEmptyStr
    run_id: NonEmptyStr
    turn_id: int = Field(ge=0)
    phase: Phase
    timestamp: AwareDatetime
    prompt_version: NonEmptyStr | None = None
    observable_model_response: JsonValue = None
    execution_objective: NonEmptyStr | None = None
    action: AgentAction | None = None
    raw_result: JsonValue = None
    normalized_observation: Observation | None = None
    verification_result: VerificationResult | None = None
    error: NonEmptyStr | None = None
    latency_ms: StrictFloat | None = Field(default=None, ge=0.0)
    usage: dict[str, JsonValue] = Field(default_factory=dict)
    provenance: list[ProvenanceReference] = Field(default_factory=list)
    artifact_paths: list[RunRelativePath] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_phase_fields(self) -> HistoryEvent:
        if self.phase is Phase.EXECUTE and self.action is None and self.error is None:
            raise ValueError("an EXECUTE event records the dispatched action or why none came")
        return self


__all__ = [
    "ActionType",
    "AgentAction",
    "AnswerAction",
    "AnswerReply",
    "CandidateAnswer",
    "Claim",
    "CodeAction",
    "ComputationInput",
    "ComputationRecord",
    "DeterministicCheckReport",
    "DeterministicCheckResult",
    "FeedbackPriority",
    "FeedbackSource",
    "HistoryEvent",
    "ImprovementFeedback",
    "LLMJudgeResult",
    "Observation",
    "ObservationSource",
    "ObserveResult",
    "Phase",
    "ProvenanceKind",
    "ProvenanceReference",
    "ResultState",
    "RunStatus",
    "SandboxResult",
    "StrictModel",
    "ToolAction",
    "ToolReply",
    "ToolResult",
    "VerificationResult",
]
