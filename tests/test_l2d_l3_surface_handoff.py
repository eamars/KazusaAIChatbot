"""Deterministic tests for L2d-selected L3 text surface handoff."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from kazusa_ai_chatbot.action_spec.registry import SPEAK_CAPABILITY
from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.nodes import persona_supervisor2 as persona_module
from kazusa_ai_chatbot.nodes.persona_supervisor2 import persona_supervisor2
from kazusa_ai_chatbot.time_context import build_character_time_context

RETIRED_L3_FIELD = "expression" + "_willingness"


def _time_context() -> dict:
    timestamp = "2026-05-16T09:00:00+12:00"
    time_context = build_character_time_context(timestamp)
    return time_context


def _episode() -> dict:
    timestamp = "2026-05-16T09:00:00+12:00"
    episode = build_text_chat_cognitive_episode(
        episode_id="episode-123",
        percept_id="percept-123",
        timestamp=timestamp,
        time_context=_time_context(),
        user_input="Please answer.",
        platform="debug",
        platform_channel_id="channel-123",
        channel_type="private",
        platform_message_id="message-123",
        platform_user_id="platform-user-123",
        global_user_id="global-user-123",
        user_name="Test User",
        target_addressed_user_ids=["character-123"],
        target_broadcast=False,
    )
    return episode


def _character_profile() -> dict:
    return {
        "name": "Test Character",
        "global_user_id": "character-123",
        "mood": "neutral",
        "global_vibe": "calm",
        "reflection_summary": "",
        "personality_brief": {
            "logic": "direct",
            "tempo": "brief",
            "defense": "none",
            "quirks": "none",
            "taboos": "none",
            "mbti": "INTJ",
        },
        "linguistic_texture_profile": {
            "hesitation_density": 0.1,
            "fragmentation": 0.1,
            "emotional_leakage": 0.1,
            "rhythmic_bounce": 0.1,
            "direct_assertion": 0.8,
            "softener_density": 0.1,
            "counter_questioning": 0.1,
            "formalism_avoidance": 0.4,
            "abstraction_reframing": 0.2,
            "self_deprecation": 0.1,
        },
        "boundary_profile": {
            "self_integrity": 0.8,
            "control_sensitivity": 0.3,
            "control_intimacy_misread": 0.2,
            "relational_override": 0.4,
            "compliance_strategy": "resist",
            "boundary_recovery": "rebound",
        },
    }


def _persona_state() -> dict:
    timestamp = "2026-05-16T09:00:00+12:00"
    state = {
        "timestamp": timestamp,
        "time_context": _time_context(),
        "user_name": "Test User",
        "platform": "debug",
        "platform_message_id": "message-123",
        "platform_user_id": "platform-user-123",
        "global_user_id": "global-user-123",
        "user_input": "Please answer.",
        "message_envelope": {
            "body_text": "Please answer.",
            "raw_wire_text": "Please answer.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-123"],
            "broadcast": False,
        },
        "prompt_message_context": {
            "body_text": "Please answer.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-123"],
            "broadcast": False,
        },
        "user_multimedia_input": [],
        "user_profile": {
            "affinity": 500,
            "last_relationship_insight": "stable",
        },
        "platform_bot_id": "bot-123",
        "character_name": "Test Character",
        "character_profile": _character_profile(),
        "platform_channel_id": "channel-123",
        "channel_type": "private",
        "channel_name": "debug",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "should_respond": True,
        "reason_to_respond": "direct request",
        "use_reply_feature": False,
        "channel_topic": "debug",
        "indirect_speech_context": "",
        "debug_modes": {},
        "cognitive_episode": _episode(),
    }
    return state


def _cognition_state() -> dict:
    state = _persona_state()
    state.update({
        "decontexualized_input": "Please answer.",
        "referents": [],
        "rag_result": {"answer": "Known fact."},
        "emotional_appraisal": "calm",
        "interaction_subtext": "direct request",
        "internal_monologue": "I should answer directly.",
        "character_intent": "PROVIDE",
        "logical_stance": "CONFIRM",
        "judgment_note": "Answering is allowed.",
        "social_distance": "friendly",
        "emotional_intensity": "low",
        "vibe_check": "calm",
        "relational_dynamic": "direct request",
        "action_specs": [_speak_action_spec()],
    })
    return state


def _speak_action_spec(reason: str = "A text surface should exist.") -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": SPEAK_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "cognitive_episode",
                "ref_id": "episode-123",
                "owner": "cognition",
                "relationship": "basis",
                "evidence_refs": [],
            }
        ],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "current_channel",
            "target_id": None,
            "owner": "l3_text",
            "scope": {"surface": "text"},
        },
        "params": {
            "delivery_mode": "visible_reply",
            "execute_at": None,
            "surface_requirements": {"intent": "answer naturally"},
        },
        "urgency": "now",
        "visibility": "user_visible",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": reason,
    }


def _action_directives() -> dict:
    directives = {
        "contextual_directives": {
            "social_distance": "friendly",
            "emotional_intensity": "low",
            "vibe_check": "calm",
            "relational_dynamic": "direct request",
        },
        "linguistic_directives": {
            "rhetorical_strategy": "answer directly",
            "linguistic_style": "brief",
            "accepted_user_preferences": [],
            "content_anchors": [
                "[DECISION] Answer directly.",
                "[ANSWER] Known fact.",
            ],
            "forbidden_phrases": [],
        },
        "visual_directives": {
            "facial_expression": [],
            "body_language": [],
            "gaze_direction": [],
            "visual_vibe": [],
        },
    }
    return directives


@pytest.mark.asyncio
async def test_cognition_subgraph_runs_l2c2_before_l2d() -> None:
    """The cognition subgraph should feed social context into L2d."""

    call_order: list[str] = []

    async def l1_agent(state: dict) -> dict:
        call_order.append("l1")
        return {
            "emotional_appraisal": "calm",
            "interaction_subtext": "none",
        }

    async def l2a_agent(state: dict) -> dict:
        call_order.append("l2a")
        return {
            "internal_monologue": "I should decide the action.",
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
        }

    async def l2b_agent(state: dict) -> dict:
        call_order.append("l2b")
        return {
            "boundary_core_assessment": {
                "boundary_issue": "none",
                "acceptance": "allow",
                "stance_bias": "confirm",
            },
        }

    async def l2c1_agent(state: dict) -> dict:
        assert state["boundary_core_assessment"]["acceptance"] == "allow"
        call_order.append("l2c1")
        return {
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "judgment_note": "Answering is allowed.",
        }

    async def l2c2_agent(state: dict) -> dict:
        assert state["boundary_core_assessment"]["acceptance"] == "allow"
        call_order.append("l2c2")
        return {
            "social_distance": "friendly",
            "emotional_intensity": "low",
            "vibe_check": "calm",
            "relational_dynamic": "direct request",
        }

    async def l2d_agent(state: dict) -> dict:
        assert state["judgment_note"] == "Answering is allowed."
        assert state["social_distance"] == "friendly"
        assert state["emotional_intensity"] == "low"
        assert state["vibe_check"] == "calm"
        assert state["relational_dynamic"] == "direct request"
        call_order.append("l2d")
        return {"action_specs": []}

    with (
        patch.object(cognition_module, "call_cognition_subconscious", l1_agent),
        patch.object(cognition_module, "call_cognition_consciousness", l2a_agent),
        patch.object(cognition_module, "call_boundary_core_agent", l2b_agent),
        patch.object(cognition_module, "call_judgment_core_agent", l2c1_agent),
        patch.object(
            cognition_module,
            "call_social_context_appraisal",
            l2c2_agent,
            create=True,
        ),
        patch.object(cognition_module, "call_action_initializer", l2d_agent),
    ):
        result = await cognition_module.call_cognition_subgraph(_cognition_state())

    assert result["action_specs"] == []
    assert result["social_distance"] == "friendly"
    assert result["emotional_intensity"] == "low"
    assert result["vibe_check"] == "calm"
    assert result["relational_dynamic"] == "direct request"
    assert call_order.index("l2b") < call_order.index("l2c2")
    assert call_order.index("l2c1") < call_order.index("l2d")
    assert call_order.index("l2c2") < call_order.index("l2d")


@pytest.mark.asyncio
async def test_selected_speak_runs_l3_surface_and_dialog_once() -> None:
    """A selected text action should invoke selected L3 then dialog."""

    l3_handler = AsyncMock(return_value={"action_directives": _action_directives()})
    dialog = AsyncMock(return_value={
        "final_dialog": ["Known fact."],
        "target_addressed_user_ids": ["global-user-123"],
        "target_broadcast": False,
        "mention_target_user": True,
    })

    with (
        patch.object(
            persona_module,
            "call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={"decontexualized_input": "Please answer."},
        ),
        patch.object(
            persona_module,
            "stage_1_research",
            new_callable=AsyncMock,
            return_value={"rag_result": {"answer": "Known fact."}},
        ),
        patch.object(
            persona_module,
            "call_cognition_subgraph",
            new_callable=AsyncMock,
            return_value={
                "internal_monologue": "I should answer.",
                "interaction_subtext": "direct",
                "emotional_appraisal": "calm",
                "character_intent": "PROVIDE",
                "logical_stance": "CONFIRM",
                "action_specs": [_speak_action_spec()],
            },
        ),
        patch.object(
            persona_module,
            "call_l3_text_surface_handler",
            l3_handler,
            create=True,
        ),
        patch.object(persona_module, "dialog_agent", dialog),
    ):
        result = await persona_supervisor2(_persona_state())

    assert result["final_dialog"] == ["Known fact."]
    assert result["surface_outputs"][0]["surface_kind"] == "text"
    assert result["action_results"][0]["action_kind"] == SPEAK_CAPABILITY
    assert result["action_results"][0]["status"] == "executed"
    l3_handler.assert_awaited_once()
    dialog.assert_awaited_once()
    dialog_state = dialog.await_args.args[0]
    serialized_dialog_state = json.dumps(dialog_state, ensure_ascii=False)
    assert RETIRED_L3_FIELD not in serialized_dialog_state


@pytest.mark.asyncio
async def test_no_speak_skips_l3_surface_and_dialog_but_consolidates() -> None:
    """No selected text action should produce private consolidation evidence."""

    l3_handler = AsyncMock(return_value={"action_directives": _action_directives()})
    dialog = AsyncMock(return_value={
        "final_dialog": ["should not run"],
        "target_addressed_user_ids": ["global-user-123"],
        "target_broadcast": False,
        "mention_target_user": False,
    })

    with (
        patch.object(
            persona_module,
            "call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={"decontexualized_input": "Please answer."},
        ),
        patch.object(
            persona_module,
            "stage_1_research",
            new_callable=AsyncMock,
            return_value={"rag_result": {"answer": "Known fact."}},
        ),
        patch.object(
            persona_module,
            "call_cognition_subgraph",
            new_callable=AsyncMock,
            return_value={
                "internal_monologue": "I will not speak now.",
                "interaction_subtext": "private",
                "emotional_appraisal": "calm",
                "character_intent": "WAIT",
                "logical_stance": "CONFIRM",
                "action_specs": [],
            },
        ),
        patch.object(
            persona_module,
            "call_l3_text_surface_handler",
            l3_handler,
            create=True,
        ),
        patch.object(persona_module, "dialog_agent", dialog),
    ):
        result = await persona_supervisor2(_persona_state())

    assert result["final_dialog"] == []
    assert result["surface_outputs"][0]["surface_kind"] == "private"
    assert result["consolidation_state"]["surface_outputs"][0]["visibility"] == (
        "private"
    )
    l3_handler.assert_not_awaited()
    dialog.assert_not_awaited()


@pytest.mark.asyncio
async def test_l3_text_surface_contract_excludes_retired_response_field() -> None:
    """Selected L3 text output should contain presentation fields only."""

    from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface

    forbidden_social_agent = AsyncMock(
        side_effect=AssertionError("selected L3 must not run social appraisal"),
    )

    with (
        patch.object(
            l3_surface,
            "call_social_context_appraisal",
            forbidden_social_agent,
            create=True,
        ),
        patch.object(
            l3_surface,
            "call_social_context_appraisal",
            forbidden_social_agent,
            create=True,
        ),
        patch.object(
            l3_surface,
            "call_interaction_style_context_loader",
            new_callable=AsyncMock,
            return_value={"interaction_style_context": {}},
        ),
        patch.object(
            l3_surface,
            "call_style_agent",
            new_callable=AsyncMock,
            return_value={
                "rhetorical_strategy": "answer directly",
                "linguistic_style": "brief",
                "forbidden_phrases": [],
            },
        ),
        patch.object(
            l3_surface,
            "call_content_anchor_agent",
            new_callable=AsyncMock,
            return_value={
                "content_anchors": [
                    "[DECISION] Answer directly.",
                    "[ANSWER] Known fact.",
                ],
            },
        ),
        patch.object(
            l3_surface,
            "call_preference_adapter",
            new_callable=AsyncMock,
            return_value={"accepted_user_preferences": []},
        ),
        patch.object(
            l3_surface,
            "call_visual_agent",
            new_callable=AsyncMock,
            return_value={
                "facial_expression": [],
                "body_language": [],
                "gaze_direction": [],
                "visual_vibe": [],
            },
        ),
    ):
        result = await l3_surface.call_l3_text_surface_handler(_cognition_state())

    serialized = json.dumps(result, ensure_ascii=False)
    assert RETIRED_L3_FIELD not in serialized
    assert result["action_directives"] == _action_directives()
    forbidden_social_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_l3_content_anchor_receives_selected_speak_intent_only() -> None:
    """L3 should consume a prompt-safe text intent, not raw action specs."""

    from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface

    captured_content_state: dict = {}
    state = _cognition_state()
    speak_spec = _speak_action_spec(
        reason="The visible answer should use the retrieved fact.",
    )
    speak_spec["params"]["surface_requirements"] = {
        "decision": "visible_reply",
        "detail": "answer with the retrieved fact",
    }
    state["action_specs"] = [speak_spec]

    async def content_anchor(state: dict) -> dict:
        captured_content_state.update(state)
        return {
            "content_anchors": [
                "[DECISION] Answer directly.",
                "[ANSWER] Known fact.",
            ],
        }

    with (
        patch.object(
            l3_surface,
            "call_interaction_style_context_loader",
            new_callable=AsyncMock,
            return_value={"interaction_style_context": {}},
        ),
        patch.object(
            l3_surface,
            "call_style_agent",
            new_callable=AsyncMock,
            return_value={
                "rhetorical_strategy": "answer directly",
                "linguistic_style": "brief",
                "forbidden_phrases": [],
            },
        ),
        patch.object(
            l3_surface,
            "call_content_anchor_agent",
            content_anchor,
        ),
        patch.object(
            l3_surface,
            "call_preference_adapter",
            new_callable=AsyncMock,
            return_value={"accepted_user_preferences": []},
        ),
        patch.object(
            l3_surface,
            "call_visual_agent",
            new_callable=AsyncMock,
            return_value={
                "facial_expression": [],
                "body_language": [],
                "gaze_direction": [],
                "visual_vibe": [],
            },
        ),
    ):
        await l3_surface.call_l3_text_surface_handler(state)

    selected_intent = captured_content_state["selected_text_surface_intent"]
    assert "answer with the retrieved fact" in selected_intent
    assert "The visible answer should use the retrieved fact." in selected_intent
    assert "action_specs" not in captured_content_state
