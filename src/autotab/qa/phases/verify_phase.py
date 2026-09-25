"""One VERIFY interaction: check and judge the action the turn just took.

A tool or code step gets its action-scope checks and a step judge that asks whether
it achieved the plan's objective; an answer gets the answer-scope checks and the full
rubric. Both combine checks and judge the same way. It never selects the next action
or transitions the FSM; see ``specs/phases/verify_phase.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from ...models.client import ModelClient
from ..layers.context_management import AssembledPrompt, ContextRequest, assemble_context
from ..layers.verification import (
    CitationResolver,
    combine,
    combine_checks,
    normalize_llm_judge,
    run_action_checks,
    run_answer_checks,
)
from ..schemas import (
    ComputationRecord,
    DeterministicCheckReport,
    LLMJudgeResult,
    Phase,
    ResultState,
    ToolAction,
    VerificationResult,
)
from ._ask import Reply, ask


@dataclass(frozen=True)
class AnswerEvidence:
    """What the answer-scope checks inspect, gathered by the agent during the run."""

    computations: Mapping[str, ComputationRecord]
    computation_states: Mapping[str, ResultState]
    resolve_citation: CitationResolver
    source_hashes: Mapping[str, tuple[str, str]]


@dataclass(frozen=True)
class VerifyOutcome:
    """The deterministic report, the judge prompt and reply, and the combined result.

    ``prompt`` is None, and ``reply`` empty, for a read, which is not judged.
    """

    report: DeterministicCheckReport
    prompt: AssembledPrompt | None
    reply: Reply[LLMJudgeResult]
    result: VerificationResult


def run(
    request: ContextRequest,
    client: ModelClient,
    *,
    evidence: AnswerEvidence,
    artifact: str,
    retries: int = 1,
) -> VerifyOutcome:
    """Verify the candidate carried by ``request``.

    Raises:
        ValueError: If the request is not for VERIFY or carries no candidate.
        ContextBudgetError: If required content cannot fit the context budget.
    """
    if request.phase is not Phase.VERIFY:
        raise ValueError(f"verify_phase cannot build a {request.phase.value} prompt")
    if request.candidate is None:
        raise ValueError("VERIFY is entered only with a candidate answer")
    report = run_answer_checks(
        request.candidate,
        computations=evidence.computations,
        computation_states=evidence.computation_states,
        resolve_citation=evidence.resolve_citation,
        source_hashes=evidence.source_hashes,
    )
    prompt = assemble_context(replace(request, report=report))
    reply = ask(client, prompt.text, normalize_llm_judge, retries=retries)
    result = combine(report, reply.value, artifact=artifact, verifier_error=reply.error)
    return VerifyOutcome(report, prompt, reply, result)


def run_step(
    request: ContextRequest,
    client: ModelClient,
    *,
    state: ResultState,
    computations: Sequence[ComputationRecord],
    artifact: str,
    retries: int = 1,
) -> VerifyOutcome:
    """Verify the latest step in ``request``: the tool or code action it took.

    Args:
        request: A VERIFY request whose last step is the one under review.
        client: Model client for the step judge.
        state: How the step's execution ended.
        computations: The computations the step recorded.
        artifact: Run-relative path of the verdict artifact.
        retries: Re-asks for a malformed judgement.

    Raises:
        ValueError: If the request is not for VERIFY, has no step, or carries a
            candidate (an answer is verified by ``run``).
        ContextBudgetError: If required content cannot fit the context budget.
    """
    if request.phase is not Phase.VERIFY:
        raise ValueError(f"verify_phase cannot build a {request.phase.value} prompt")
    if not request.steps or request.candidate is not None:
        raise ValueError("a step verification needs the step and no candidate answer")
    report = run_action_checks(state, list(computations))
    if isinstance(request.steps[-1].action, ToolAction):
        # A read has nothing to judge beyond whether it came back complete, which the
        # checks decide; what it means is the next plan's job. This saves a judge
        # call on every look at the workbook.
        return VerifyOutcome(
            report, None, Reply(None, (), None), combine_checks(report, artifact=artifact)
        )
    prompt = assemble_context(
        replace(request, report=report, computations={c.id: c for c in computations})
    )
    reply = ask(client, prompt.text, normalize_llm_judge, retries=retries)
    result = combine(report, reply.value, artifact=artifact, verifier_error=reply.error)
    return VerifyOutcome(report, prompt, reply, result)
