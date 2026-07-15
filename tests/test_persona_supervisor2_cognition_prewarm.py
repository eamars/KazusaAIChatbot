"""Patched cognition-subgraph tests for first-cycle memory prewarm."""

from __future__ import annotations

import pytest
pytest.skip("Stage 1 assertions replaced by the V2 contract suite", allow_module_level=True)

import asyncio
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.cognition_resolver.state import (
    build_empty_rag_result,
    new_resolver_state,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.time_boundary import build_turn_clock


def _empty_rag_result() -> dict[str, Any]:
    """Build the standard empty RAG result used by cognition state."""

    rag_result = build_empty_rag_result(
        current_user_id="user-1",
        character_user_id="character-1",
    )
    return rag_result


def _prewarm_rag_result() -> dict[str, Any]:
    """Build one projected shared-memory result for patched prewarm tests."""

    rag_result = _empty_rag_result()
    rag_result["memory_evidence"] = [
        {
            "summary": "shared memory summary",
            "content": "Shared memory evidence for L2a.",
        }
    ]
    return rag_result


def _resolver_state(cycle_index: int) -> dict[str, Any]:
    """Build a resolver cycle state at the requested cycle index."""

    state = new_resolver_state(
        decontexualized_input="Need a memory-backed stance.",
        max_cycles=3,
    )
    state["cycle_index"] = cycle_index
    return state


def _persona_state(*, cycle_index: int = 0) -> dict[str, Any]:
    """Build a persona state accepted by `call_cognition_subgraph`."""

    turn_clock = build_turn_clock("2026-06-08 09:00:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="cognition-prewarm-episode",
        percept_id="cognition-prewarm-percept",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="Need a memory-backed stance.",
        platform="debug",
        platform_channel_id="prewarm-channel",
        channel_type="private",
        platform_message_id="prewarm-message",
        platform_user_id="platform-user-1",
        global_user_id="user-1",
        user_name="Test User",
        target_addressed_user_ids=["character-1"],
        target_broadcast=False,
    )
    state = {
        "character_profile": {
            "name": "Kazusa",
            "global_user_id": "character-1",
        },
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "user_input": "Need a memory-backed stance.",
        "prompt_message_context": {
            "body_text": "Need a memory-backed stance.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "cognitive_episode": episode,
        "platform": "debug",
        "platform_channel_id": "prewarm-channel",
        "channel_type": "private",
        "platform_message_id": "prewarm-message",
        "platform_user_id": "platform-user-1",
        "global_user_id": "user-1",
        "user_name": "Test User",
        "user_profile": {"relationship_state": 500},
        "platform_bot_id": "platform-bot-1",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "prewarm test",
        "conversation_progress": None,
        "conversation_episode_state": None,
        "promoted_reflection_context": None,
        "internal_monologue_residue_context": "",
        "decontexualized_input": "Need a memory-backed stance.",
        "referents": [],
        "rag_result": _empty_rag_result(),
        "resolver_state": _resolver_state(cycle_index),
        "resolver_context": (
            f"resolver_state: status=running; cycle_index={cycle_index}; "
            "max_cycles=3"
        ),
    }
    return state


async def _l1_agent(_state: dict[str, Any]) -> dict[str, str]:
    """Return stable L1 outputs for patched graph tests."""

    return_value = {
        "emotional_appraisal": "calm",
        "interaction_subtext": "direct request",
    }
    return return_value


async def _l2a_agent(_state: dict[str, Any]) -> dict[str, str]:
    """Return stable L2a outputs for patched graph tests."""

    return_value = {
        "internal_monologue": "Memory evidence is available.",
        "logical_stance": "TENTATIVE",
        "character_intent": "RESPOND",
    }
    return return_value


async def _l2b_agent(_state: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Return stable L2b outputs for patched graph tests."""

    return_value = {
        "boundary_core_assessment": {
            "boundary_issue": "none",
            "acceptance": "allow",
            "stance_bias": "steady",
        },
    }
    return return_value


async def _l2c1_agent(_state: dict[str, Any]) -> dict[str, str]:
    """Return stable judgment outputs for patched graph tests."""

    return_value = {
        "logical_stance": "TENTATIVE",
        "character_intent": "RESPOND",
        "judgment_note": "Use evidence as context.",
    }
    return return_value


async def _l2c2_agent(_state: dict[str, Any]) -> dict[str, str]:
    """Return stable social-context outputs for patched graph tests."""

    return_value = {
        "social_distance": "friendly",
        "emotional_intensity": "low",
        "vibe_check": "calm",
        "relational_dynamic": "stable",
    }
    return return_value


async def _l2d_agent(_state: dict[str, Any]) -> dict[str, list]:
    """Return a terminal no-action L2d output for patched graph tests."""

    return_value: dict[str, list] = {
        "action_specs": [],
        "resolver_capability_requests": [],
    }
    return return_value


def _patch_cognition_nodes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    l2a_agent: Any = _l2a_agent,
    l2b_agent: Any = _l2b_agent,
) -> None:
    """Patch every LLM-backed cognition node with deterministic agents."""

    monkeypatch.setattr(l1_module, "call_cognition_subconscious", _l1_agent)
    monkeypatch.setattr(
        l2_module,
        "call_cognition_consciousness",
        l2a_agent,
    )
    monkeypatch.setattr(l2_module, "call_boundary_core_agent", l2b_agent)
    monkeypatch.setattr(l2_module, "call_judgment_core_agent", _l2c1_agent)
    monkeypatch.setattr(
        l2c2_module,
        "call_social_context_appraisal",
        _l2c2_agent,
    )
    monkeypatch.setattr(l2d_module, "select_semantic_actions", _l2d_agent)


@pytest.mark.asyncio
async def test_first_cycle_prewarm_evidence_reaches_l2a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L2a should consume merged shared-memory evidence on resolver cycle zero."""

    captured: dict[str, Any] = {}

    async def prewarm(_state: dict[str, Any]) -> dict[str, Any]:
        result = _prewarm_rag_result()
        return result

    async def l2a_agent(state: dict[str, Any]) -> dict[str, str]:
        captured["rag_result"] = state["rag_result"]
        result = await _l2a_agent(state)
        return result

    _patch_cognition_nodes(monkeypatch, l2a_agent=l2a_agent)
    monkeypatch.setattr(
        cognition_module,
        "run_first_cycle_shared_memory_prewarm",
        prewarm,
        raising=False,
    )

    result = await cognition_module.call_cognition_subgraph(_persona_state())

    l2a_rag_result = captured["rag_result"]
    assert l2a_rag_result["memory_evidence"] == (
        _prewarm_rag_result()["memory_evidence"]
    )
    for key in result["rag_result"]:
        assert result["rag_result"][key] == l2a_rag_result[key]


@pytest.mark.asyncio
async def test_call_cognition_subgraph_returns_merged_rag_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downstream resolver state should receive the same evidence L2a consumed."""

    async def prewarm(_state: dict[str, Any]) -> dict[str, Any]:
        result = _prewarm_rag_result()
        return result

    _patch_cognition_nodes(monkeypatch)
    monkeypatch.setattr(
        cognition_module,
        "run_first_cycle_shared_memory_prewarm",
        prewarm,
        raising=False,
    )

    result = await cognition_module.call_cognition_subgraph(_persona_state())

    assert result["rag_result"]["answer"] == ""
    assert result["rag_result"]["memory_evidence"] == (
        _prewarm_rag_result()["memory_evidence"]
    )


@pytest.mark.asyncio
async def test_prewarm_starts_only_on_resolver_cycle_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Later resolver cycles should not start unconditional prewarm work."""

    calls: list[str] = []
    captured: dict[str, Any] = {}

    async def prewarm(_state: dict[str, Any]) -> dict[str, Any]:
        calls.append("prewarm")
        result = _prewarm_rag_result()
        return result

    async def l2a_agent(state: dict[str, Any]) -> dict[str, str]:
        captured["rag_result"] = state["rag_result"]
        result = await _l2a_agent(state)
        return result

    _patch_cognition_nodes(monkeypatch, l2a_agent=l2a_agent)
    monkeypatch.setattr(
        cognition_module,
        "run_first_cycle_shared_memory_prewarm",
        prewarm,
        raising=False,
    )

    result = await cognition_module.call_cognition_subgraph(
        _persona_state(cycle_index=1),
    )

    assert calls == []
    expected = _empty_rag_result()
    for key in expected:
        assert captured["rag_result"][key] == expected[key]
    for key in result["rag_result"]:
        assert result["rag_result"][key] == captured["rag_result"][key]


@pytest.mark.asyncio
async def test_l2b_runs_independently_of_prewarm_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prewarm completes before chain starts, so L2b sees merged evidence."""

    events: list[str] = []

    async def prewarm(_state: dict[str, Any]) -> dict[str, Any]:
        events.append("prewarm_done")
        result = _prewarm_rag_result()
        return result

    async def l2b_agent(state: dict[str, Any]) -> dict[str, dict[str, str]]:
        events.append("l2b")
        result = await _l2b_agent(state)
        return result

    _patch_cognition_nodes(
        monkeypatch,
        l2b_agent=l2b_agent,
    )
    monkeypatch.setattr(
        cognition_module,
        "run_first_cycle_shared_memory_prewarm",
        prewarm,
        raising=False,
    )

    await asyncio.wait_for(
        cognition_module.call_cognition_subgraph(_persona_state()),
        timeout=5.0,
    )

    assert "prewarm_done" in events
    assert "l2b" in events
    assert events.index("prewarm_done") < events.index("l2b")


@pytest.mark.asyncio
async def test_l2a_uses_base_rag_result_when_prewarm_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty prewarm output should preserve the base RAG payload for L2a."""

    captured: dict[str, Any] = {}
    state = _persona_state()
    state["rag_result"]["answer"] = "base answer"

    async def prewarm(_state: dict[str, Any]) -> dict[str, Any]:
        result = _empty_rag_result()
        return result

    async def l2a_agent(l2a_state: dict[str, Any]) -> dict[str, str]:
        captured["rag_result"] = l2a_state["rag_result"]
        result = await _l2a_agent(l2a_state)
        return result

    _patch_cognition_nodes(monkeypatch, l2a_agent=l2a_agent)
    monkeypatch.setattr(
        cognition_module,
        "run_first_cycle_shared_memory_prewarm",
        prewarm,
        raising=False,
    )

    result = await cognition_module.call_cognition_subgraph(state)

    assert captured["rag_result"]["answer"] == "base answer"
    assert captured["rag_result"]["memory_evidence"] == []
    for key in result["rag_result"]:
        assert result["rag_result"][key] == captured["rag_result"][key]
