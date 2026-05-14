"""Tests for source-aware consolidator prompt payloads."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_consolidator_facts as facts_module,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_consolidator_memory_units as memory_units_module,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_consolidator_reflection as reflection_module,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_origin import (
    ConsolidationOriginMetadata,
)
from kazusa_ai_chatbot.time_context import build_character_time_context


class _CaptureLLM:
    """Capture one LLM invocation and return a deterministic JSON payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Store the deterministic response payload for one fake LLM.

        Args:
            payload: JSON-serializable response returned from `ainvoke`.
        """
        self.payload = payload
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
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
    origin: ConsolidationOriginMetadata = {
        "episode_id": "self-cognition-episode-1",
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": "preview",
        "timestamp": "2026-05-10T21:00:00+12:00",
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
    timestamp = "2026-05-10T21:00:00+12:00"
    user_memory_context = {
        "stable_patterns": [],
        "recent_shifts": [],
        "objective_facts": [],
        "active_commitments": [],
        "milestones": [],
    }
    state: dict[str, Any] = {
        "timestamp": timestamp,
        "time_context": build_character_time_context(timestamp),
        "global_user_id": "global-user-1",
        "user_name": "Test User",
        "user_profile": {"affinity": 500},
        "platform": "qq",
        "platform_channel_id": "channel-1",
        "channel_type": "private",
        "platform_message_id": "self_cognition:case-1",
        "action_directives": {
            "linguistic_directives": {
                "content_anchors": ["[PROGRESSION] revisit missed promise"],
            },
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
            "user_image": {"user_memory_context": user_memory_context},
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
async def test_reflection_payloads_include_internal_thought_origin(
    monkeypatch,
) -> None:
    """Global and relationship reflection stages should expose source identity."""
    global_llm = _CaptureLLM(
        {
            "mood": "hurt",
            "global_vibe": "uneasy",
            "reflection_summary": "summary",
        }
    )
    relationship_llm = _CaptureLLM(
        {
            "skip": False,
            "subjective_appraisals": ["The silence felt disappointing."],
            "affinity_delta": -1,
            "last_relationship_insight": "unreliable",
        }
    )
    monkeypatch.setattr(reflection_module, "_global_state_updater_llm", global_llm)
    monkeypatch.setattr(
        reflection_module,
        "_relationship_recorder_llm",
        relationship_llm,
    )

    state = _state()
    await reflection_module.global_state_updater(state)
    await reflection_module.relationship_recorder(state)

    _assert_internal_origin_payload(_human_payload(global_llm))
    _assert_internal_origin_payload(_human_payload(relationship_llm))
    _assert_source_aware_prompt(_system_prompt(global_llm))
    _assert_source_aware_prompt(_system_prompt(relationship_llm))


@pytest.mark.asyncio
async def test_fact_payloads_include_internal_thought_origin(monkeypatch) -> None:
    """Fact harvester and evaluator should expose source identity."""
    harvester_llm = _CaptureLLM({"new_facts": [], "future_promises": []})
    evaluator_llm = _CaptureLLM(
        {
            "should_stop": True,
            "feedback": "pass",
            "contradiction_flags": [],
        }
    )
    monkeypatch.setattr(facts_module, "_facts_harvester_llm", harvester_llm)
    monkeypatch.setattr(
        facts_module,
        "_fact_harvester_evaluator_llm",
        evaluator_llm,
    )

    state = _state()
    state["fact_harvester_retry"] = 0
    state["fact_harvester_feedback_message"] = []
    await facts_module.facts_harvester(state)
    await facts_module.fact_harvester_evaluator(state)

    _assert_internal_origin_payload(_human_payload(harvester_llm))
    _assert_internal_origin_payload(_human_payload(evaluator_llm))
    _assert_source_aware_prompt(_system_prompt(harvester_llm))
    _assert_source_aware_prompt(_system_prompt(evaluator_llm))


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
