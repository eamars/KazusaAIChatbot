"""Direct-path size, replacement, exhaustion, and side-effect guard tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.character_carryover import (
    CharacterCarryoverServicesV1,
    run_character_carryover_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.prompt_budget import (
    PromptBudgetError,
    fit_evidence_texts_to_budget,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_character_production_state,
)
from kazusa_ai_chatbot.llm_interface.contracts import LLMCallConfig


NOW = "2026-08-02T00:00:00Z"


def _config() -> LLMCallConfig:
    """Build a bounded carry-over route configuration for direct tests."""

    return LLMCallConfig(
        stage_name="character_carryover_boundary_test",
        route_name="COGNITION_LLM_CHARACTER_CARRYOVER",
        base_url="http://test.invalid",
        api_key="test-key",
        model="test-model",
        temperature=0.0,
        top_p=None,
        top_k=None,
        max_completion_tokens=8192,
        presence_penalty=None,
    )


class _OversizedThenInvalidLLM:
    """Exercise replacement and typed exhaustion without a state write."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages
        del config
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(content="x" * 9001)
        return SimpleNamespace(content=json.dumps({"unknown": True}))


def test_prompt_budget_reduces_low_priority_text_and_preserves_handles() -> None:
    """Stable fitting keeps evidence handles and a non-empty text floor."""

    rows = [
        {"evidence_handle": "e1", "semantic_text": "A" * 500},
        {"evidence_handle": "e2", "semantic_text": "B" * 500},
    ]
    fitted = fit_evidence_texts_to_budget(
        rows,
        text_field="semantic_text",
        budget=180,
    )

    assert [row["evidence_handle"] for row in fitted] == ["e1", "e2"]
    assert all(row["semantic_text"] for row in fitted)


def test_prompt_budget_fails_closed_when_required_structure_cannot_fit() -> None:
    """An irreducible required structure returns a typed local error."""

    with pytest.raises(PromptBudgetError):
        fit_evidence_texts_to_budget(
            [{"required": "x" * 500}],
            text_field="required",
            budget=4,
        )


@pytest.mark.asyncio
async def test_oversized_output_replaces_then_degrades_without_side_effect() -> None:
    """Output caps never leak a raw limit exception or partial state update."""

    llm = _OversizedThenInvalidLLM()
    services = CharacterCarryoverServicesV1(llm=llm, config=_config())
    result = await run_character_carryover_cognition(
        source_episode_id="episode:limit",
        evidence=[
            {
                "source_kind": "episode",
                "source_id": "episode:limit",
                "occurred_at": NOW,
                "semantic_summary": "closed operational event",
                "evidence_handle": "e1",
            }
        ],
        base_state=build_character_production_state(updated_at=NOW),
        effective_at=NOW,
        services=services,
    )

    assert result.disposition == "degraded"
    assert result.error_code in {
        "output_limit",
        "contract_exhausted",
        "provider_exhausted",
    }
    assert result.state_update is None
    assert llm.calls <= 3
