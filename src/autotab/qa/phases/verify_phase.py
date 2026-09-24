"""One VERIFY interaction for an existing candidate answer.

Runs the answer-scope deterministic checks, gives the judge the candidate with those
results and the rubric, and combines both sides. It never selects the next action or
transitions the FSM; see ``specs/phases/verify_phase.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from ...models.client import ModelClient
from ..layers.context_management import AssembledPrompt, ContextRequest, assemble_context
from ..layers.verification import (
    CitationResolver,
    combine,
    normalize_llm_judge,
    run_answer_checks,
)
from ..schemas import (
    ComputationRecord,
    DeterministicCheckReport,
    LLMJudgeResult,
    Phase,
    ResultState,
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
    """The deterministic report, the judge prompt and reply, and the combined result."""

    report: DeterministicCheckReport
    prompt: AssembledPrompt
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
