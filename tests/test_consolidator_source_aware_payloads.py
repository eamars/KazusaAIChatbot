"""Tests for source-aware consolidator prompt payloads."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.consolidation import (
    memory_units as memory_units_module,
)
from kazusa_ai_chatbot.consolidation.origin import (
    ConsolidationOriginMetadata,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.cognition_core_v2_test_helpers import canonical_cognition_output


class _CaptureLLM:
    """Capture one LLM invocation and return a deterministic JSON payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Store the deterministic response payload for one fake LLM.

        Args:
            payload: JSON-serializable response returned from `ainvoke`.
        """
        self.payload = payload
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any], *, config=None) -> SimpleNamespace:
        """Capture messages and return the configured response.

        Args:
            messages: LangChain messages sent to the fake LLM.

        Returns:
            Object with a JSON string `content` attribute.
        """
        self.messages = list(messages)
        content = json.dumps(self.payload)
        response = SimpleNamespace(content=content)
        return response


def _internal_thought_origin() -> ConsolidationOriginMetadata:
    """Build source metadata for an internal-thought consolidation state.

    Returns:
        Identifier-only origin metadata for tests.
    """
    turn_clock = build_turn_clock("2026-05-10 21:00:00")
    origin: ConsolidationOriginMetadata = {
        "episode_id": "self-cognition-episode-1",
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": "preview",
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "platform": "qq",
        "platform_channel_id": "channel-1",
        "channel_type": "private",
        "platform_message_id": "self_cognition:case-1",
        "active_turn_platform_message_ids": [],
        "active_turn_conversation_row_ids": [],
        "current_platform_user_id": "platform-user-1",
        "current_global_user_id": "global-user-1",
        "current_display_name": "Test User",
    }
    return origin


def _state() -> dict[str, Any]:
    """Build a consolidator state carrying internal-thought source metadata.

    Returns:
        State fields consumed by source-aware prompt handlers.
    """
    turn_clock = build_turn_clock("2026-05-10 21:00:00")
    state: dict[str, Any] = {
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_profile": {
            "global_user_id": "global-user-1",
            "cognition_state": build_acquaintance_user_state(
                global_user_id="global-user-1",
                updated_at=turn_clock["storage_timestamp_utc"].replace(
                    "+00:00",
                    "Z",
                ),
            ),
        },
        "platform": "qq",
        "platform_channel_id": "channel-1",
        "channel_type": "private",
        "platform_message_id": "self_cognition:case-1",
        "cognition_core_output": canonical_cognition_output(
            owner_user_id="global-user-1",
        ),
        "text_surface_output_v2": {
            "schema_version": "text_surface_output.v2",
            "content_plan": "Revisit the missed promise.",
            "content_requirements": ["Clarify the unresolved promise."],
            "visible_boundaries": [],
            "addressee_plan": ["current user"],
            "style_guidance": "brief and grounded",
            "selected_surface_intent": "clarify the unresolved promise",
        },
        "internal_monologue": "The missed promise still feels unresolved.",
        "final_dialog": ["Private finalization for consolidation only."],
        "interaction_subtext": "unresolved relationship event",
        "emotional_appraisal": "hurt and uncertain",
        "character_intent": "CLARIFY",
        "logical_stance": "TENTATIVE",
        "character_profile": {
            "name": "Kazusa",
            "personality_brief": {"mbti": "INTJ"},
        },
        "rag_result": {
            "answer": "",
            "user_memory_unit_candidates": [],
            "memory_evidence": [],
            "recall_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "supervisor_trace": {"unknown_slots": [], "loop_count": 0},
        },
        "existing_dedup_keys": set(),
        "decontexualized_input": "Internal thought about a missed promise.",
        "chat_history_recent": [],
        "consolidation_origin": _internal_thought_origin(),
        "new_facts": [],
        "future_promises": [],
        "subjective_appraisals": [],
        "metadata": {},
    }
    return state


def _human_payload(capture_llm: _CaptureLLM) -> dict[str, Any]:
    """Decode the captured human message JSON payload.

    Args:
        capture_llm: Fake LLM after one invocation.

    Returns:
        Human message payload sent to the fake LLM.
    """
    human_message = capture_llm.messages[1]
    payload = json.loads(human_message.content)
    return payload


def _system_prompt(capture_llm: _CaptureLLM) -> str:
    """Return the captured system prompt text.

    Args:
        capture_llm: Fake LLM after one invocation.

    Returns:
        System prompt content sent to the fake LLM.
    """
    system_message = capture_llm.messages[0]
    prompt = str(system_message.content)
    return prompt


def _assert_internal_origin_payload(payload: dict[str, Any]) -> None:
    """Assert an LLM payload carries internal-thought source identity.

    Args:
        payload: Prompt-facing JSON payload.
    """
    assert payload["consolidation_origin"] == {
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": "preview",
        "episode_id": "self-cognition-episode-1",
    }
    if "decontexualized_input" in payload:
        input_value = payload["decontexualized_input"]
    else:
        input_value = payload["decontextualized_input"]
    assert input_value == "Internal thought about a missed promise."


def _assert_source_aware_prompt(prompt: str) -> None:
    """Assert a shared prompt describes source-aware input semantics.

    Args:
        prompt: System prompt content sent to the fake LLM.
    """
    assert "consolidation_origin" in prompt
    assert "internal_thought" in prompt


@pytest.mark.asyncio
async def test_memory_extractor_payload_includes_internal_thought_origin(
    monkeypatch,
) -> None:
    """Memory-unit extraction should expose source identity."""
    extractor_llm = _CaptureLLM({"memory_units": []})
    monkeypatch.setattr(memory_units_module, "_extractor_llm", extractor_llm)

    state = _state()
    result = await memory_units_module.extract_memory_unit_candidates(state)

    assert result == []
    payload = _human_payload(extractor_llm)
    _assert_internal_origin_payload(payload)
    _assert_source_aware_prompt(_system_prompt(extractor_llm))
