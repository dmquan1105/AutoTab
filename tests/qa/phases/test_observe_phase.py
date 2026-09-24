"""Observe-phase orchestration tests with a scripted model; see observe_phase.md."""

import json
from typing import Any

import pytest

from autotab.models.client import ModelResponseError
from autotab.qa.layers.context_management import ContextRequest
from autotab.qa.phases import observe_phase
from autotab.qa.schemas import ObserveResult, Phase

PLAN = {
    "known_facts": ["People!G1 is the Score header."],
    "uncertainties": ["No gender column has been seen."],
    "execution_objective": "Read People!A1:G2 to learn the header layout.",
    "rationale": "The name and score headers are merged over two rows.",
}


def _request(**overrides: Any) -> ContextRequest:
    values: dict[str, Any] = {
        "phase": Phase.OBSERVE,
        "query": "Full name of the man who have the highest score",
        "workbooks": [{"workbook_id": "workbook_1", "file": "QA_sample.xlsx", "sheets": []}],
        "budget_tokens": 20_000,
        "output_schema": ObserveResult.model_json_schema(),
    }
    values.update(overrides)
    return ContextRequest(**values)


def test_context_is_assembled_before_the_single_model_call(scripted: Any) -> None:
    client = scripted(PLAN)

    outcome = observe_phase.run(_request(), client)

    assert outcome.plan == ObserveResult.model_validate_json(json.dumps(PLAN))
    assert len(client.prompts) == 1
    assert client.prompts[0] == outcome.prompt.text
    assert "Full name of the man who have the highest score" in client.prompts[0]
    assert "Phase: OBSERVE" in client.prompts[0]


def test_a_malformed_reply_is_re_asked_once_with_the_error(scripted: Any) -> None:
    client = scripted("I think we should read the headers.", PLAN)

    outcome = observe_phase.run(_request(), client, retries=1)

    assert outcome.plan is not None
    assert outcome.reply.attempts == 2
    assert "rejected" in client.prompts[1]
    assert client.prompts[1].startswith(client.prompts[0])


def test_the_re_ask_budget_is_honoured_and_failure_is_typed(scripted: Any) -> None:
    client = scripted("not json", "still not json")

    outcome = observe_phase.run(_request(), client, retries=1)

    assert outcome.plan is None
    assert outcome.reply.error
    assert len(client.prompts) == 2


def test_a_plan_that_smuggles_in_a_tool_call_is_rejected(scripted: Any) -> None:
    client = scripted({**PLAN, "tool_name": "inspect_range"}, {**PLAN, "code": "print(1)"})

    outcome = observe_phase.run(_request(), client, retries=1)

    assert outcome.plan is None


def test_a_failed_request_is_retried_and_reported(scripted: Any) -> None:
    client = scripted(ModelResponseError("HTTP 502: upstream error"), PLAN)

    outcome = observe_phase.run(_request(), client, retries=1)

    assert outcome.plan is not None


def test_only_the_observe_phase_is_accepted(scripted: Any) -> None:
    with pytest.raises(ValueError):
        observe_phase.run(_request(phase=Phase.EXECUTE), scripted(PLAN))


def test_usage_and_latency_add_up_across_a_re_ask(scripted: Any) -> None:
    client = scripted("not json", PLAN)

    outcome = observe_phase.run(_request(), client, retries=1)

    usage = outcome.reply.usage
    assert usage["calls"] == 2
    assert usage["prompt_tokens"] == sum(len(prompt) // 4 for prompt in client.prompts)
    assert outcome.reply.latency_ms == 10.0
