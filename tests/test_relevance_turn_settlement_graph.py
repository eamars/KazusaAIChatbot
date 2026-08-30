"""Graph handoff tests for settled relevance and atomic cognition claims."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.brain_service.graph import build_graph


def _state() -> dict:
    """Build the smallest state accepted by the test graph."""

    return {
        "debug_modes": {},
        "user_multimedia_input": [],
        "response_action": "proceed",
        "turn_id": "turn-1",
        "turn_version": 3,
        "should_respond": True,
    }


@pytest.mark.asyncio
async def test_proceed_requires_claim_before_downstream_cognition() -> None:
    """A settled proceed decision enters cognition only after a true claim."""

    claims: list[tuple[str, int]] = []

    async def _relevance(_state):
        return {
            "response_action": "proceed",
            "turn_id": "turn-1",
            "turn_version": 3,
        }

    async def _claim(state):
        claims.append((state["turn_id"], state["turn_version"]))
        return {"cognition_claimed": True, "should_respond": True}

    async def _load(_state):
        return {"conversation_episode_state": None}

    async def _persona(_state):
        return {
            "final_dialog": ["claimed"],
            "cognition_state_committed": True,
            "cognition_core_output": {"intention": {"route": "speech"}},
            "cognition_state_update": {},
        }

    async def _multimedia(_state):
        raise AssertionError("media node should not run for this text case")

    graph = build_graph(
        relevance_agent_node=_relevance,
        multimedia_descriptor_agent_node=_multimedia,
        load_conversation_episode_state_node=_load,
        persona_supervisor_node=_persona,
        claim_for_cognition_node=_claim,
    )

    result = await graph.ainvoke(_state())

    assert claims == [("turn-1", 3)]
    assert result["final_dialog"] == ["claimed"]
    assert result["should_respond"] is True


@pytest.mark.asyncio
async def test_ignore_ends_before_claim_and_cognition() -> None:
    """Settled ignore remains silent and cannot enter downstream nodes."""

    called = []

    async def _relevance(_state):
        return {"response_action": "ignore"}

    async def _claim(_state):
        called.append("claim")
        return {"cognition_claimed": True}

    async def _load(_state):
        called.append("load")
        return {}

    async def _persona(_state):
        called.append("persona")
        return {}

    async def _multimedia(_state):
        raise AssertionError("media node should not run for this text case")

    graph = build_graph(
        relevance_agent_node=_relevance,
        multimedia_descriptor_agent_node=_multimedia,
        load_conversation_episode_state_node=_load,
        persona_supervisor_node=_persona,
        claim_for_cognition_node=_claim,
    )

    result = await graph.ainvoke(_state())

    assert called == []
    assert result["response_action"] == "ignore"


@pytest.mark.asyncio
async def test_dsh_fields_cross_top_level_graph_boundary() -> None:
    """Top-level graph preserves pending DSH input and typed output."""

    pending = {
        "schema_version": "dsh_brain_interaction.v2",
        "interaction_id": "interaction-1",
        "kind": "question",
    }
    decision = {
        "schema_version": "dsh_brain_interaction.v2",
        "interaction_id": "interaction-1",
        "kind": "question",
        "decision": "answer",
        "answer": "The requested answer.",
        "reason": "The interaction can be answered.",
    }
    captured: dict[str, object] = {}

    async def _relevance(_state):
        return {"response_action": "proceed"}

    async def _claim(_state):
        return {"cognition_claimed": True}

    async def _load(_state):
        return {"conversation_episode_state": None}

    async def _persona(state):
        captured["state"] = state
        return {
            "final_dialog": [],
            "cognition_state_committed": True,
            "cognition_core_output": {},
            "cognition_state_update": {},
            "dsh_interaction_decision": decision,
        }

    async def _multimedia(_state):
        raise AssertionError("media node should not run for this text case")

    graph = build_graph(
        relevance_agent_node=_relevance,
        multimedia_descriptor_agent_node=_multimedia,
        load_conversation_episode_state_node=_load,
        persona_supervisor_node=_persona,
        claim_for_cognition_node=_claim,
    )
    state = _state()
    state.update({
        "pending_dsh_interaction": pending,
        "dsh_interaction_episode": True,
    })

    result = await graph.ainvoke(state)

    persona_state = captured["state"]
    assert isinstance(persona_state, dict)
    assert persona_state["pending_dsh_interaction"] == pending
    assert persona_state["dsh_interaction_episode"] is True
    assert "pending_dsh_reply" not in persona_state
    assert result["dsh_interaction_decision"] == decision
