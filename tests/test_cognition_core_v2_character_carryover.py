"""Focused native character carry-over decision and privacy tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.character_carryover import (
    CharacterCarryoverServicesV1,
    run_character_carryover_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_character_production_state,
)
from kazusa_ai_chatbot.llm_interface.contracts import LLMCallConfig


NOW = "2026-08-02T00:00:00Z"


def _evidence(source_id: str, *, text: str = "closed operational event") -> dict[str, str]:
    """Build one ref-complete current-episode evidence view."""

    return {
        "source_kind": "episode",
        "source_id": source_id,
        "occurred_at": NOW,
        "semantic_summary": text,
        "evidence_handle": f"evidence:{source_id}",
    }


class _NoCallLLM:
    """Fail if a source-free empty episode invokes the model."""

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
        raise AssertionError("empty carry-over input must not call the model")


class _UnsafeOutputLLM:
    """Return a model-authored emotion/cause class that must be rejected."""

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages
        del config
        return SimpleNamespace(
            content=json.dumps({
                "emotion_id": "anger",
                "cause_class": "relationship",
            })
        )


class _ValidNativeOutputLLM:
    """Return one source-free appraisal with no model-authored emotion."""

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages
        del config
        return SimpleNamespace(
            content=json.dumps({
                "question_id": "character_carryover",
                "propositions": [
                    {
                        "kind": "event",
                        "handle": "event:episode",
                        "deltas": {"outcome_impact": -60},
                        "evidence_handles": ["evidence:episode"],
                    }
                ],
            })
        )


def _services(llm: object) -> CharacterCarryoverServicesV1:
    """Build the carry-over service bundle with the required route config."""

    config = LLMCallConfig(
        stage_name="character_carryover_test",
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
    return CharacterCarryoverServicesV1(llm=llm, config=config)


@pytest.mark.asyncio
async def test_empty_or_incomplete_episode_returns_no_change_without_a_call() -> None:
    """No ref-complete evidence is a deterministic no-change terminal result."""

    llm = _NoCallLLM()
    result = await run_character_carryover_cognition(
        source_episode_id="episode-empty",
        evidence=[],
        base_state=build_character_production_state(updated_at=NOW),
        effective_at=NOW,
        services=_services(llm),
    )

    assert result.disposition == "no_change"
    assert result.decision.action == "no_change"
    assert result.attempts == 0
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_model_authored_emotion_or_cause_class_is_rejected() -> None:
    """The model proposes typed deltas; native code owns causes and emotions."""

    result = await run_character_carryover_cognition(
        source_episode_id="episode-unsafe",
        evidence=[_evidence("unsafe")],
        base_state=build_character_production_state(updated_at=NOW),
        effective_at=NOW,
        services=_services(_UnsafeOutputLLM()),
    )

    assert result.decision.privacy_disposition == "unsafe"
    assert result.disposition == "degraded"
    assert result.decision.semantic_appraisal is None
    assert result.state_update is None


@pytest.mark.asyncio
async def test_valid_carryover_can_apply_one_native_state_update() -> None:
    """One accepted appraisal yields at most one source-free replacement."""

    result = await run_character_carryover_cognition(
        source_episode_id="episode",
        evidence=[_evidence("episode")],
        base_state=build_character_production_state(updated_at=NOW),
        effective_at=NOW,
        services=_services(_ValidNativeOutputLLM()),
    )

    assert result.disposition in {"apply", "degraded"}
    if result.disposition == "apply":
        assert result.state_update is not None
        assert result.state_update["state_scope"] == "character"
        assert len(result.state_update["active_events"]) <= 32
        assert result.decision.semantic_appraisal is not None
