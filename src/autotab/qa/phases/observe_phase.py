"""One OBSERVE interaction: assemble context, call the model, parse ObserveResult.

Runs on entry to a run and after a verification FAIL only. It never selects a tool,
writes code, or transitions the FSM; see ``specs/phases/observe_phase.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...models.client import ModelClient
from ..layers.context_management import AssembledPrompt, ContextRequest, assemble_context
from ..layers.verification import extract_json_object
from ..schemas import ObserveResult, Phase
from ._ask import Reply, ask


@dataclass(frozen=True)
class ObserveOutcome:
    """The rendered prompt and the model's validated plan, if any."""

    prompt: AssembledPrompt
    reply: Reply[ObserveResult]

    @property
    def plan(self) -> ObserveResult | None:
        return self.reply.value


def _parse(raw: str) -> ObserveResult:
    return ObserveResult.model_validate_json(extract_json_object(raw))


def run(request: ContextRequest, client: ModelClient, *, retries: int = 1) -> ObserveOutcome:
    """Plan the next objective.

    Raises:
        ValueError: If the request is not for the OBSERVE phase.
        ContextBudgetError: If required content cannot fit the context budget.
    """
    if request.phase is not Phase.OBSERVE:
        raise ValueError(f"observe_phase cannot build a {request.phase.value} prompt")
    prompt = assemble_context(request)
    return ObserveOutcome(prompt, ask(client, prompt.text, _parse, retries=retries))
