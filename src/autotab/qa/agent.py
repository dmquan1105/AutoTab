"""Top-level QA FSM.

Sequences INITIALIZE, OBSERVE, EXECUTE (repeating for data-gathering turns), VERIFY
on an answer action, then FINALIZE or FAIL. It is the only owner of transitions and
turn accounting; reusable behavior lives in layers and model calls in phases. See
``specs/phases/agent.md``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openpyxl.utils.cell import range_boundaries
from pydantic import TypeAdapter

from ..artifacts.store import new_run_id
from ..models.client import ModelClient, client_from_config
from .contracts import AgentState
from .layers.agent_state import advance_turn, can_transition, max_turns_reached
from .layers.context_management import (
    TOKEN_COUNTER,
    ContextBudgetError,
    ContextRequest,
    Step,
    active_feedback,
)
from .layers.observation import ExplorationError, load_exploration
from .layers.sandbox import Sandbox
from .layers.tools import ToolRegistry, WorkbookReader
from .phases import execute_phase, observe_phase, verify_phase
from .phases._ask import Reply
from .prompts import PROMPT_VERSIONS, RUBRIC_VERSION
from .schemas import (
    AgentAction,
    AnswerAction,
    CandidateAnswer,
    CodeAction,
    ComputationRecord,
    DeterministicCheckReport,
    FeedbackPriority,
    FeedbackSource,
    HistoryEvent,
    ImprovementFeedback,
    LLMJudgeResult,
    Observation,
    ObserveResult,
    Phase,
    ResultState,
    RunStatus,
    VerificationResult,
)
from .trace import Trace

_REPLY_SCHEMAS = {
    Phase.OBSERVE: ObserveResult.model_json_schema(),
    Phase.EXECUTE: TypeAdapter(AgentAction).json_schema(),
    Phase.VERIFY: LLMJudgeResult.model_json_schema(),
}


@dataclass(frozen=True)
class RunOutcome:
    """The terminal result of one QA run."""

    status: RunStatus
    reason: str
    run_dir: Path
    turns: int
    answer_text: str | None = None
    verification: VerificationResult | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class _Run:
    """The mutable pieces of one run; ``run()`` owns the order they change in."""

    query: str
    config: Mapping[str, Any]
    trace: Trace
    client: ModelClient | None
    state: AgentState
    registry: ToolRegistry | None = None
    sandbox: Sandbox | None = None
    workbooks: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)
    source_paths: dict[str, Path] = field(default_factory=dict)
    leads: list[Observation] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    plan: ObserveResult | None = None
    events: list[DeterministicCheckReport | LLMJudgeResult] = field(default_factory=list)
    computations: dict[str, ComputationRecord] = field(default_factory=dict)
    computation_states: dict[str, ResultState] = field(default_factory=dict)
    after_failure: bool = False
    usage_totals: dict[str, int | float] = field(default_factory=dict)
    last_action: AgentAction | None = None

    @property
    def qa(self) -> Mapping[str, Any]:
        return self.config["qa"]

    @property
    def retries(self) -> int:
        return int(self.qa["max_parse_retries"])

    # --- bookkeeping ---------------------------------------------------------------

    def _event(self, phase: Phase, **fields: Any) -> None:
        event = HistoryEvent(
            event_id=self.trace.next_event_id(),
            run_id=self.trace.run_id,
            turn_id=self.state.turn,
            phase=phase,
            timestamp=datetime.now(UTC),
            **fields,
        )
        self.trace.append(event)
        self.state.history_refs.append(event.event_id)

    def _model_event(self, phase: Phase, reply: Reply[Any], **fields: Any) -> None:
        """Record a phase that called the model, with what the calls cost."""
        for name, value in reply.usage.items():
            self.usage_totals[name] = self.usage_totals.get(name, 0) + value
        if reply.latency_ms is not None:
            self.usage_totals["latency_ms"] = (
                float(self.usage_totals.get("latency_ms", 0.0)) + reply.latency_ms
            )
        self._event(phase, usage=dict(reply.usage), latency_ms=reply.latency_ms, **fields)

    def _move(self, target: Phase, action_type: str | None = None) -> None:
        if not can_transition(self.state.phase, target.value, action_type=action_type):
            raise RuntimeError(f"illegal transition {self.state.phase} -> {target.value}")
        self.state.phase = target.value

    def _request(self, phase: Phase, **extra: Any) -> ContextRequest:
        registry = self.registry
        assert registry is not None
        return ContextRequest(
            phase=phase,
            query=self.query,
            workbooks=self.workbooks,
            budget_tokens=int(self.qa["max_context_tokens"]),
            output_schema=_REPLY_SCHEMAS[phase],
            exploration=self.leads,
            steps=self.steps,
            plan=self.plan,
            feedback=active_feedback(self.events),
            tool_descriptions=registry.descriptions(),
            **extra,
        )

    # --- INITIALIZE ------------------------------------------------------------------

    def start(self, workbooks: Sequence[str | Path]) -> str | None:
        """Prepare everything a turn needs; return why the run cannot start, if so."""
        self.manifest = {
            "query": self.query,
            "configuration": dict(self.config),
            "prompt_versions": PROMPT_VERSIONS,
            "rubric_version": RUBRIC_VERSION,
            "context_budget": {
                "max_tokens": int(self.qa["max_context_tokens"]),
                "counter": TOKEN_COUNTER,
            },
        }
        if self.client is None:
            self.client = client_from_config(
                self.config["models"].get("llm"),
                float(self.config["runtime"]["request_timeout_seconds"]),
            )
        if self.client is None:
            return "no model is configured for QA; set models.llm.base_url"
        self.manifest["models"] = {"llm": self.config["models"].get("llm", {}).get("model")}

        readers = []
        for index, raw_path in enumerate(workbooks, start=1):
            path = Path(raw_path)
            workbook_id = f"workbook_{index}"
            try:
                readers.append(
                    WorkbookReader(
                        workbook_id,
                        path,
                        include_hidden=bool(self.config["exploration"]["include_hidden_sheets"]),
                        max_range_cells=int(self.qa["tools"]["max_range_cells"]),
                    )
                )
            except (OSError, ValueError, KeyError) as exc:
                return f"workbook {raw_path} cannot be read: {exc}"
            self.source_paths[workbook_id] = path
            self.source_hashes[workbook_id] = _sha256(path)
            self.workbooks.append(
                {"workbook_id": workbook_id, "file": path.name, "sheets": readers[-1].describe()}
            )
        if not readers:
            return "no workbook was given"
        self.manifest["workbooks"] = [
            {"workbook_id": wid, "path": str(self.source_paths[wid]), "sha256": digest}
            for wid, digest in self.source_hashes.items()
        ]
        self.registry = ToolRegistry(readers)

        failure = self._load_exploration(workbooks)
        if failure is not None:
            return failure

        sandbox = self.qa["sandbox"]
        self.sandbox = Sandbox(
            self.registry,
            timeout_seconds=float(sandbox["timeout_seconds"]),
            memory_limit_mb=int(sandbox["memory_limit_mb"]),
            max_code_chars=int(self.qa["max_code_chars"]),
            max_output_chars=int(self.qa["max_observation_chars"]),
            max_inline_cells=int(self.qa["max_inline_cells"]),
        )
        self.manifest["sandbox"] = {"memory_limit_enforced": self.sandbox.memory_limit_enforced}
        self.trace.write_manifest(self.manifest)
        self._event(Phase.INITIALIZE, artifact_paths=["manifest.json"])
        return None

    def _load_exploration(self, workbooks: Sequence[str | Path]) -> str | None:
        enabled = bool(self.qa["exploration_enabled"])
        path = self.qa.get("exploration_path")
        if enabled and path is None:
            # Imported here: exploration pulls in rendering, which QA needs only now.
            from ..exploration.pipeline import ExplorationPipeline

            # Same run ID: this run's exploration/ and qa/ share one folder.
            report = ExplorationPipeline(dict(self.config)).run(
                [str(p) for p in workbooks], self.query, run_id=self.trace.run_id
            )
            path = str(report.parent)
        self.manifest["exploration"] = {"enabled": enabled, "path": path}
        if not enabled:
            return None
        try:
            # Exploration records carry no workbook identity; leads attach to the first.
            self.leads = load_exploration(str(path), "workbook_1")
        except ExplorationError as exc:
            return f"exploration is enabled but unusable: {exc}"
        self.manifest["exploration"]["leads"] = len(self.leads)
        return None

    # --- OBSERVE ---------------------------------------------------------------------

    def observe(self) -> None:
        assert self.client is not None
        request = self._request(Phase.OBSERVE, after_failure=self.after_failure)
        outcome = observe_phase.run(request, self.client, retries=self.retries)
        if outcome.plan is not None:
            self.plan = outcome.plan
            self.state.objective = outcome.plan.execution_objective
        # What the model saw, verbatim: the trajectory alone shows what it did, not why.
        prompt_path = f"prompts/{self.state.turn}.observe.txt"
        self.trace.write_text(prompt_path, outcome.prompt.text)
        self._model_event(
            Phase.OBSERVE,
            outcome.reply,
            prompt_version=PROMPT_VERSIONS["observe"],
            artifact_paths=[prompt_path],
            observable_model_response=(
                outcome.plan.model_dump(mode="json")
                if outcome.plan is not None
                else list(outcome.reply.raw)
            ),
            execution_objective=self.state.objective,
            error=outcome.reply.error if outcome.plan is None else None,
        )
        self.after_failure = False

    # --- EXECUTE ---------------------------------------------------------------------

    def execute(self) -> AgentAction | None:
        assert self.client is not None and self.registry is not None and self.sandbox is not None
        execution_id = f"exec_{self.state.turn + 1}"
        request = self._request(Phase.EXECUTE, sandbox_names=self.sandbox.bound_names())
        outcome = execute_phase.run(
            request,
            self.client,
            registry=self.registry,
            sandbox=self.sandbox,
            execution_id=execution_id,
            retries=self.retries,
            max_observation_chars=int(self.qa["max_observation_chars"]),
            max_inline_cells=int(self.qa["max_inline_cells"]),
            previous=self.last_action,
        )
        advance_turn(self.state)
        action = outcome.action
        if action is not None:
            self.last_action = action
        prompt_path = f"prompts/{self.state.turn}.execute.txt"
        self.trace.write_text(prompt_path, outcome.prompt.text)
        paths = [f"actions/{execution_id}.json", prompt_path]
        self.trace.write_json(
            paths[0],
            {
                "execution_id": execution_id,
                "turn": self.state.turn,
                "action": action.model_dump(mode="json") if action is not None else None,
                "model_replies": list(outcome.reply.raw),
                "raw_result": (
                    outcome.raw_result.model_dump(mode="json") if outcome.raw_result else None
                ),
                "action_checks": outcome.report.model_dump(mode="json") if outcome.report else None,
            },
        )
        if isinstance(action, CodeAction):
            code_path = f"code/{execution_id}.py"
            self.trace.write_text(code_path, action.code)
            paths.append(code_path)
        for record in outcome.computations:
            self.computations[record.id] = record
            state = outcome.raw_result.state if outcome.raw_result else ResultState.ERROR
            self.computation_states[record.id] = state
            paths.append(f"computations/{record.id}.json")
            self.trace.write_json(paths[-1], record.model_dump(mode="json"))
        if outcome.report is not None:
            self.events.append(outcome.report)
        if outcome.observation is not None and action is not None:
            self.steps.append(Step(self.state.turn, action, outcome.observation, execution_id))
            self.state.latest_observation = outcome.observation
        self._model_event(
            Phase.EXECUTE,
            outcome.reply,
            prompt_version=PROMPT_VERSIONS["execute"],
            execution_objective=self.state.objective,
            action=action,
            raw_result=outcome.raw_result.model_dump(mode="json") if outcome.raw_result else None,
            normalized_observation=outcome.observation,
            error=outcome.reply.error if action is None else None,
            provenance=list(outcome.observation.provenance) if outcome.observation else [],
            artifact_paths=paths,
        )
        if isinstance(action, AnswerAction):
            self.state.candidate = action.answer
        return action

    # --- VERIFY ----------------------------------------------------------------------

    def _resolves(self, sheet: str, range_ref: str) -> bool:
        try:
            range_boundaries(range_ref)
        except (ValueError, TypeError):
            return False
        assert self.registry is not None
        return any(sheet in reader.sheet_names() for reader in self.registry.readers())

    def verify(self, candidate: CandidateAnswer) -> VerificationResult:
        assert self.client is not None
        artifact = f"verification/{self.state.turn}.json"
        evidence = verify_phase.AnswerEvidence(
            computations=self.computations,
            computation_states=self.computation_states,
            resolve_citation=self._resolves,
            source_hashes={
                wid: (digest, _sha256(self.source_paths[wid]))
                for wid, digest in self.source_hashes.items()
            },
        )
        outcome = verify_phase.run(
            self._request(Phase.VERIFY, candidate=candidate, computations=self.computations),
            self.client,
            evidence=evidence,
            artifact=artifact,
            retries=self.retries,
        )
        self.trace.write_text(f"verification/{self.state.turn}.prompt.txt", outcome.prompt.text)
        self.trace.write_json(
            artifact,
            {
                "turn": self.state.turn,
                "deterministic": outcome.report.model_dump(mode="json"),
                "judge_raw": list(outcome.reply.raw),
                "judge": (
                    outcome.reply.value.model_dump(mode="json") if outcome.reply.value else None
                ),
                "result": outcome.result.model_dump(mode="json"),
            },
        )
        self.events.append(outcome.report)
        self.events.append(outcome.reply.value or _unreadable_judge(outcome.result))
        self.state.latest_verification = outcome.result
        self._model_event(
            Phase.VERIFY,
            outcome.reply,
            prompt_version=PROMPT_VERSIONS["verification"],
            observable_model_response=(
                outcome.reply.value.model_dump(mode="json")
                if outcome.reply.value
                else list(outcome.reply.raw)
            ),
            verification_result=outcome.result,
            artifact_paths=[artifact, f"verification/{self.state.turn}.prompt.txt"],
        )
        return outcome.result

    # --- terminal --------------------------------------------------------------------

    def finalize(self, candidate: CandidateAnswer, result: VerificationResult) -> RunOutcome:
        self._move(Phase.FINALIZE)
        evidence = [
            f"- {claim.statement} ({', '.join(claim.citations) or 'no citation'}"
            + (f"; computation {claim.computation_id}" if claim.computation_id else "")
            + ")"
            for claim in candidate.claims
        ]
        self.trace.write_final_answer(
            candidate.answer_text + "\n\n## Evidence\n" + "\n".join(evidence) + "\n"
        )
        self._event(Phase.FINALIZE, artifact_paths=["final/answer.md"])
        return self._outcome(RunStatus.PASS, result.reason, candidate.answer_text, result)

    def fail(self, reason: str) -> RunOutcome:
        self.state.phase = Phase.FAIL.value
        self._event(Phase.FAIL, error=reason)
        return self._outcome(RunStatus.FAIL, reason, None, self.state.latest_verification)

    def _outcome(
        self,
        status: RunStatus,
        reason: str,
        answer_text: str | None,
        verification: VerificationResult | None,
    ) -> RunOutcome:
        self.manifest["outcome"] = {
            "status": status.value,
            "turns": self.state.turn,
            "reason": reason,
            "usage": dict(self.usage_totals),
        }
        self.trace.write_manifest(self.manifest)
        return RunOutcome(
            status=status,
            reason=reason,
            run_dir=self.trace.root,
            turns=self.state.turn,
            answer_text=answer_text,
            verification=verification,
        )

    def close(self) -> None:
        if self.sandbox is not None:
            self.sandbox.close()


def _unreadable_judge(result: VerificationResult) -> LLMJudgeResult:
    """Stand in for a judge that never produced a valid verdict.

    Keeping one resolution mechanism means the verifier failure stays active feedback
    until a later judgement replaces it, exactly like a judge FAIL.
    """
    items = [i for i in result.improvement_feedback if i.source is FeedbackSource.LLM_JUDGE]
    blocking = items or [
        ImprovementFeedback(
            code="VERIFIER.INVALID_RESPONSE",
            source=FeedbackSource.LLM_JUDGE,
            problem=result.reason,
            action="Answer again with explicit claims so the verifier can judge them.",
            priority=FeedbackPriority.BLOCKING,
            recheck="LLM_JUDGE",
        )
    ]
    return LLMJudgeResult(
        status=RunStatus.FAIL,
        confidence_score=0.0,
        reason=result.reason,
        issues_found=[result.reason],
        improvement_feedback=blocking,
        final_assessment="No valid judgement was produced.",
    )


def run(
    query: str,
    workbooks: Sequence[str | Path],
    config: Mapping[str, Any],
    *,
    client: ModelClient | None = None,
) -> RunOutcome:
    """Answer ``query`` over ``workbooks`` and return a verified answer or a precise FAIL.

    Args:
        query: The question, verbatim.
        workbooks: Source workbooks; never modified.
        config: Loaded configuration (see ``autotab.config``).
        client: Model client; built from ``models.llm`` when omitted.
    """
    trace = Trace(config["runtime"]["artifact_root"], new_run_id())
    session = _Run(
        query=query,
        config=config,
        trace=trace,
        client=client,
        state=AgentState(query=query, max_turns=int(config["qa"]["max_turns"])),
    )
    try:
        failure = session.start(workbooks)
        if failure is not None:
            return session.fail(failure)
        session._move(Phase.OBSERVE)
        while True:
            if session.state.phase == Phase.OBSERVE.value:
                session.observe()
                session._move(Phase.EXECUTE)
                continue
            if session.state.phase == Phase.EXECUTE.value:
                if max_turns_reached(session.state):
                    return session.fail(
                        f"the budget of {session.state.max_turns} turns ran out before an "
                        "answer was verified"
                    )
                action = session.execute()
                action_type = action.action_type.value if action is not None else "tool"
                target = Phase.VERIFY if isinstance(action, AnswerAction) else Phase.EXECUTE
                session._move(target, action_type)
                continue
            candidate = session.state.candidate
            assert candidate is not None
            result = session.verify(candidate)
            if result.status is RunStatus.PASS:
                return session.finalize(candidate, result)
            if max_turns_reached(session.state):
                return session.fail(
                    f"the budget of {session.state.max_turns} turns ran out; the last "
                    f"verification failed: {result.reason}"
                )
            session.after_failure = True
            session._move(Phase.OBSERVE)
    except ContextBudgetError as exc:
        return session.fail(f"the prompt could not be built within budget: {exc}")
    finally:
        session.close()
