"""Deterministic tests for L2d-selected L3 text surface handoff."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from kazusa_ai_chatbot.action_spec.registry import SPEAK_CAPABILITY
from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as l3_module
from kazusa_ai_chatbot.nodes import persona_supervisor2 as persona_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface_module
from kazusa_ai_chatbot.nodes.persona_supervisor2 import persona_supervisor2
from kazusa_ai_chatbot.time_boundary import build_turn_clock

RETIRED_L3_FIELD = "expression" + "_willingness"


def _time_context() -> dict:
    time_context = build_turn_clock("2026-05-16 09:00:00")[
        "local_time_context"
    ]
    return time_context


def _episode() -> dict:
    turn_clock = build_turn_clock("2026-05-16 09:00:00")
    episode = build_text_chat_cognitive_episode(
        episode_id="episode-123",
        percept_id="percept-123",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
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
    turn_clock = build_turn_clock("2026-05-16 09:00:00")
    state = {
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
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
            "content_plan": {
                "semantic_content": "Known fact.",
                "rendering": "One visible chat bubble; concise.",
            },
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


def test_background_artifact_acknowledgement_requires_pending_queue_result() -> None:
    """L3 should see promise permission only after durable enqueue succeeded."""

    state = _cognition_state()
    state["pre_surface_action_results"] = [
        {
            "action_attempt_id": "action_attempt:background-artifact-001",
            "action_kind": "background_artifact_request",
            "status": "pending",
            "queue_state": "queued",
            "work_kind": "coding_snippet",
            "objective_summary": "Generate a Fibonacci function snippet.",
            "operational_owner": "background_artifact_job",
            "job_ref": "background_artifact_job:job-001",
            "acknowledgement_constraint": "promise_allowed",
        }
    ]

    intent = l3_surface_module._selected_text_surface_intent(state)

    assert "background_artifact_request" in intent
    assert "coding_snippet" in intent
    assert "Generate a Fibonacci function snippet." in intent
    assert "promise_allowed" in intent
    assert "background_artifact_job:job-001" not in intent


def test_background_artifact_failed_enqueue_blocks_later_delivery_promise() -> None:
    """L3 should not promise later delivery when durable enqueue failed."""

    state = _cognition_state()
    state["pre_surface_action_results"] = [
        {
            "action_attempt_id": "action_attempt:background-artifact-001",
            "action_kind": "background_artifact_request",
            "status": "failed",
            "queue_state": "none",
            "work_kind": "coding_snippet",
            "objective_summary": "Generate a Fibonacci function snippet.",
            "operational_owner": "none",
            "job_ref": "",
            "acknowledgement_constraint": "promise_forbidden_explain_failure",
        }
    ]

    intent = l3_surface_module._selected_text_surface_intent(state)

    assert "background_artifact_request" in intent
    assert "promise_forbidden_explain_failure" in intent
    assert "promise_allowed" not in intent


def test_background_work_acknowledgement_requires_pending_queue_result() -> None:
    """L3 should see background-work promise permission only after enqueue."""

    state = _cognition_state()
    state["pre_surface_action_results"] = [
        {
            "action_attempt_id": "action_attempt:background-work-001",
            "action_kind": "background_work_request",
            "status": "pending",
            "queue_state": "queued",
            "task_summary": "Generate a Fibonacci function snippet.",
            "operational_owner": "background_work_job",
            "job_ref": "background_work_job:job-001",
            "acknowledgement_constraint": "promise_allowed",
        }
    ]

    intent = l3_surface_module._selected_text_surface_intent(state)

    assert "background_work_request" in intent
    assert "Generate a Fibonacci function snippet." in intent
    assert "promise_allowed" in intent
    assert "background_work_job:job-001" not in intent


def test_background_work_failed_enqueue_blocks_later_delivery_promise() -> None:
    """L3 should not promise later delivery when background-work enqueue failed."""

    state = _cognition_state()
    state["pre_surface_action_results"] = [
        {
            "action_attempt_id": "action_attempt:background-work-001",
            "action_kind": "background_work_request",
            "status": "failed",
            "queue_state": "none",
            "task_summary": "Generate a Fibonacci function snippet.",
            "operational_owner": "none",
            "job_ref": "",
            "acknowledgement_constraint": "promise_forbidden_explain_failure",
        }
    ]

    intent = l3_surface_module._selected_text_surface_intent(state)

    assert "background_work_request" in intent
    assert "promise_forbidden_explain_failure" in intent
    assert "promise_allowed" not in intent


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
async def test_cognition_subgraph_loads_group_engagement_before_l2d() -> None:
    """Targetless group self-cognition should load group engagement pre-L2d."""

    call_order: list[str] = []

    async def l1_agent(_state: dict) -> dict:
        call_order.append("l1")
        return {
            "emotional_appraisal": "curious",
            "interaction_subtext": "observing group scene",
        }

    async def l2a_agent(_state: dict) -> dict:
        call_order.append("l2a")
        return {
            "internal_monologue": "I am watching whether the group has an opening.",
            "logical_stance": "TENTATIVE",
            "character_intent": "EVADE",
        }

    async def l2b_agent(_state: dict) -> dict:
        call_order.append("l2b")
        return {
            "boundary_core_assessment": {
                "boundary_issue": "none",
                "acceptance": "allow",
                "stance_bias": "tentative",
            },
        }

    async def l2c1_agent(_state: dict) -> dict:
        call_order.append("l2c1")
        return {
            "logical_stance": "TENTATIVE",
            "character_intent": "EVADE",
            "judgment_note": "Keep the group context in view.",
        }

    async def l2c2_agent(_state: dict) -> dict:
        call_order.append("l2c2")
        return {
            "social_distance": "ambient group",
            "emotional_intensity": "low",
            "vibe_check": "busy group",
            "relational_dynamic": "no single target",
        }

    async def engagement_loader(state: dict) -> dict:
        assert state["channel_type"] == "group"
        assert state["global_user_id"] == ""
        call_order.append("engagement")
        return {
            "group_engagement_action_context": {
                "engagement_guidelines": [
                    "Stay with the current group topic.",
                ],
                "confidence": "medium",
            }
        }

    async def l2d_agent(state: dict) -> dict:
        assert state["group_engagement_action_context"] == {
            "engagement_guidelines": [
                "Stay with the current group topic.",
            ],
            "confidence": "medium",
        }
        call_order.append("l2d")
        return {"action_specs": []}

    state = _cognition_state()
    state.update({
        "channel_type": "group",
        "platform_channel_id": "group-123",
        "platform_user_id": "",
        "global_user_id": "",
        "user_name": "group audience",
    })
    episode = dict(state["cognitive_episode"])
    episode["trigger_source"] = "internal_thought"
    episode["input_sources"] = ["internal_monologue"]
    episode["output_mode"] = "preview"
    episode["target_scope"] = {
        "platform": "debug",
        "platform_channel_id": "group-123",
        "channel_type": "group",
        "current_platform_user_id": "",
        "current_global_user_id": "",
        "current_display_name": "group audience",
        "target_addressed_user_ids": [],
        "target_broadcast": True,
    }
    state["cognitive_episode"] = episode

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
        patch.object(
            cognition_module,
            "call_group_engagement_action_context_loader",
            engagement_loader,
            create=True,
        ),
        patch.object(cognition_module, "call_action_initializer", l2d_agent),
    ):
        result = await cognition_module.call_cognition_subgraph(state)

    assert result["action_specs"] == []
    assert call_order.index("l2c1") < call_order.index("engagement")
    assert call_order.index("l2c2") < call_order.index("engagement")
    assert call_order.index("engagement") < call_order.index("l2d")


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
            "call_cognition_resolver_loop",
            new_callable=AsyncMock,
            return_value=_cognition_state(),
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
async def test_parent_graph_preserves_social_context_for_selected_l3() -> None:
    """Parent persona graph should retain L2c2 fields for selected L3."""

    captured_content_state: dict = {}

    async def content_plan(state: dict) -> dict:
        captured_content_state.update(state)
        return {
            "content_plan": {
                "semantic_content": "Known fact.",
                "rendering": "One visible chat bubble; concise.",
            },
        }

    with (
        patch.object(
            persona_module,
            "call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={"decontexualized_input": "Please answer."},
        ),
        patch.object(
            persona_module,
            "call_cognition_resolver_loop",
            new_callable=AsyncMock,
            return_value={
                **_cognition_state(),
                "internal_monologue": "I should answer.",
                "interaction_subtext": "direct request",
                "emotional_appraisal": "calm",
                "character_intent": "PROVIDE",
                "logical_stance": "CONFIRM",
                "judgment_note": "Answering is allowed.",
                "social_distance": "friendly",
                "emotional_intensity": "low",
                "vibe_check": "calm",
                "relational_dynamic": "direct request",
                "action_specs": [_speak_action_spec()],
            },
        ),
        patch.object(
            l3_surface_module,
            "call_interaction_style_context_loader",
            new_callable=AsyncMock,
            return_value={"interaction_style_context": {}},
        ),
        patch.object(
            l3_surface_module,
            "call_style_agent",
            new_callable=AsyncMock,
            return_value={
                "rhetorical_strategy": "answer directly",
                "linguistic_style": "brief",
                "forbidden_phrases": [],
            },
        ),
        patch.object(l3_surface_module, "call_content_plan_agent", content_plan),
        patch.object(
            l3_surface_module,
            "call_preference_adapter",
            new_callable=AsyncMock,
            return_value={"accepted_user_preferences": []},
        ),
        patch.object(
            l3_surface_module,
            "call_visual_agent",
            new_callable=AsyncMock,
            return_value={
                "facial_expression": [],
                "body_language": [],
                "gaze_direction": [],
                "visual_vibe": [],
            },
        ),
        patch.object(
            persona_module,
            "dialog_agent",
            new_callable=AsyncMock,
            return_value={
                "final_dialog": ["Known fact."],
                "target_addressed_user_ids": ["global-user-123"],
                "target_broadcast": False,
                "mention_target_user": True,
            },
        ),
    ):
        result = await persona_supervisor2(_persona_state())

    assert result["final_dialog"] == ["Known fact."]
    assert captured_content_state["social_distance"] == "friendly"
    assert captured_content_state["emotional_intensity"] == "low"
    assert captured_content_state["vibe_check"] == "calm"
    assert captured_content_state["relational_dynamic"] == "direct request"


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
            "call_cognition_resolver_loop",
            new_callable=AsyncMock,
            return_value={
                **_cognition_state(),
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
            "call_content_plan_agent",
            new_callable=AsyncMock,
            return_value={
                "content_plan": {
                    "semantic_content": "Known fact.",
                    "rendering": "One visible chat bubble; concise.",
                },
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
async def test_l3_content_plan_receives_selected_speak_intent_only() -> None:
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

    async def content_plan(state: dict) -> dict:
        captured_content_state.update(state)
        return {
            "content_plan": {
                "semantic_content": "Known fact.",
                "rendering": "One visible chat bubble; concise.",
            },
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
            "call_content_plan_agent",
            content_plan,
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


def test_selected_text_surface_intent_keeps_resolved_hil_original_goal() -> None:
    """Resolved HIL continuations should keep the original user deliverable."""

    state = _cognition_state()
    state["resolver_state"] = {
        "pending_resume": {
            "schema_version": "resolver_pending_resume.v1",
            "resume_id": "resolver_pending:v1:goal",
            "capability_kind": "human_clarification",
            "status": "closed",
            "platform": "debug",
            "platform_channel_id": "channel-123",
            "global_user_id": "global-user-123",
            "source_message_id": "message-123",
            "prompt_safe_original_goal": (
                "安排两小时计划：吃饭，加一段可以走走看的路线。"
            ),
            "prompt_safe_question": "问位置。",
            "prompt_safe_approval_summary": "",
            "created_at_utc": "2026-05-30T00:00:00+00:00",
            "expires_at_utc": "2026-05-31T00:00:00+00:00",
        },
    }
    state["pending_resolver_resume"] = {
        "schema_version": "resolver_pending_resume.v1",
        "resume_id": "resolver_pending:v1:goal",
        "capability_kind": "human_clarification",
        "status": "closed",
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "global_user_id": "global-user-123",
        "source_message_id": "message-123",
        "prompt_safe_original_goal": (
            "安排两小时计划：吃饭，加一段可以走走看的路线。"
        ),
        "prompt_safe_question": "问位置。",
        "prompt_safe_approval_summary": "",
        "created_at_utc": "2026-05-30T00:00:00+00:00",
        "expires_at_utc": "2026-05-31T00:00:00+00:00",
    }
    state["resolver_pending_resolution"] = {
        "schema_version": "resolver_pending_resolution.v1",
        "resume_id": "resolver_pending:v1:goal",
        "decision": "answered",
        "reason": "用户补足了位置。",
    }

    selected_intent = l3_surface_module._selected_text_surface_intent(state)

    assert selected_intent.startswith(
        "原始目标：安排两小时计划：吃饭，加一段可以走走看的路线。"
    )
    assert "原始目标：安排两小时计划：吃饭，加一段可以走走看的路线。" in (
        selected_intent
    )

    state.pop("pending_resolver_resume")
    selected_intent = l3_surface_module._selected_text_surface_intent(state)

    assert "原始目标：安排两小时计划：吃饭，加一段可以走走看的路线。" in (
        selected_intent
    )


def test_selected_text_surface_intent_includes_goal_progress_checklist() -> None:
    """L3 text intent should receive resolver deliverables, not only evidence."""

    state = _cognition_state()
    state["resolver_state"] = {
        "goal_progress": {
            "schema_version": "resolver_goal_progress.v1",
            "original_goal": "安排两小时计划：吃饭，加一段路线。",
            "current_focus": "最终回答需要完整计划。",
            "deliverables": [
                {
                    "description": "晚餐候选和证据边界",
                    "status": "partial",
                    "note": "不能确认实时营业。",
                },
                {
                    "description": "两小时路线和时间切分",
                    "status": "pending",
                    "note": "最终回答必须覆盖。",
                },
            ],
            "missing_user_inputs": [],
            "evidence_dependencies": ["实时营业证据"],
            "attempted_paths": ["web_evidence: 奥克兰 CBD 晚餐"],
            "source_backed_facts": ["用户在奥克兰 CBD"],
            "assumptions_or_inferences": ["可以给出路线骨架"],
            "blockers": ["无法确认 19:30 营业状态"],
            "final_response_requirements": [
                "覆盖晚餐、散步、时间切分和最终核实清单",
            ],
        },
    }

    selected_intent = l3_surface_module._selected_text_surface_intent(state)

    assert "目标进度：" in selected_intent
    assert "晚餐候选和证据边界" in selected_intent
    assert "两小时路线和时间切分" in selected_intent
    assert "覆盖晚餐、散步、时间切分和最终核实清单" in selected_intent


def test_selected_text_surface_intent_includes_resolver_observation_summaries() -> None:
    """L3 text intent should receive prior prompt-safe evidence observations."""

    state = _cognition_state()
    state["resolver_state"] = {
        "observations": [
            {
                "schema_version": "resolver_observation.v1",
                "observation_id": "raw-observation-id",
                "capability_kind": "web_evidence",
                "request_objective": "Find CBD walking routes.",
                "request_reason": "The final answer needs route evidence.",
                "status": "succeeded",
                "prompt_safe_summary": "Found Wynyard Quarter and Britomart.",
                "rag_result": {
                    "answer": "Walking route answer.",
                    "external_evidence": [
                        {
                            "summary": (
                                "Wynyard Quarter and Britomart are CBD "
                                "evening walking options."
                            ),
                        },
                    ],
                },
                "evidence_refs": [],
                "created_at_utc": "2026-05-30T00:00:00+00:00",
            },
        ],
    }

    selected_intent = l3_surface_module._selected_text_surface_intent(state)

    assert "证据观察：" in selected_intent
    assert "Found Wynyard Quarter and Britomart." in selected_intent
    assert "Wynyard Quarter and Britomart are CBD evening walking options" in (
        selected_intent
    )
    assert "raw-observation-id" not in selected_intent


def test_l3_content_plan_input_includes_resolver_observation_summaries() -> None:
    """Content-plan input should keep prior prompt-safe evidence observations."""

    state = _cognition_state()
    state["resolver_state"] = {
        "observations": [
            {
                "schema_version": "resolver_observation.v1",
                "observation_id": "raw-observation-id",
                "capability_kind": "web_evidence",
                "request_objective": "Find CBD walking routes.",
                "request_reason": "The final answer needs route evidence.",
                "status": "succeeded",
                "prompt_safe_summary": "Found Wynyard Quarter and Britomart.",
                "rag_result": {
                    "answer": "Walking route answer.",
                    "external_evidence": [
                        {
                            "summary": (
                                "Wynyard Quarter and Britomart are CBD "
                                "evening walking options."
                            ),
                        },
                    ],
                },
                "evidence_refs": [],
                "created_at_utc": "2026-05-30T00:00:00+00:00",
            },
        ],
    }

    observation_context = l3_module._resolver_observations_for_content_plan(state)

    assert "resolver_obs_1" in observation_context
    assert "capability=web_evidence" in observation_context
    assert "status=succeeded" in observation_context
    assert "Wynyard Quarter and Britomart are CBD evening walking options" in (
        observation_context
    )
    assert "summary=Found Wynyard Quarter and Britomart." in observation_context
    assert "raw-observation-id" not in observation_context


@pytest.mark.asyncio
async def test_l3_content_plan_receives_goal_progress_before_generation() -> None:
    """Selected L3 should pass L2d goal progress into the content-plan agent."""

    from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface

    captured_content_state: dict = {}
    state = _cognition_state()
    state["resolver_goal_progress"] = {
        "schema_version": "resolver_goal_progress.v1",
        "original_goal": "安排两小时计划：吃饭，加一段路线。",
        "current_focus": "最终回答需要完整计划。",
        "deliverables": [
            {
                "description": "晚餐候选和证据边界",
                "status": "partial",
                "note": "不能确认实时营业。",
            },
            {
                "description": "两小时路线和时间切分",
                "status": "pending",
                "note": "最终回答必须覆盖。",
            },
        ],
        "missing_user_inputs": [],
        "evidence_dependencies": ["实时营业证据"],
        "attempted_paths": ["web_evidence: 奥克兰 CBD 晚餐"],
        "source_backed_facts": ["用户在奥克兰 CBD"],
        "assumptions_or_inferences": ["可以给出路线骨架"],
        "blockers": ["无法确认 19:30 营业状态"],
        "final_response_requirements": [
            "覆盖晚餐、散步、时间切分和最终核实清单",
        ],
    }

    async def content_plan(state: dict) -> dict:
        captured_content_state.update(state)
        return {
            "content_plan": {
                "semantic_content": (
                    "Give the compact final plan with dinner, walking route, "
                    "time split, and verification checklist."
                ),
                "rendering": "One visible chat bubble; cover all deliverables.",
            },
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
            "call_content_plan_agent",
            content_plan,
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
        result = await l3_surface.call_l3_text_surface_handler(state)

    selected_intent = captured_content_state["selected_text_surface_intent"]
    assert "目标进度：" in selected_intent
    assert "original_goal=安排两小时计划：吃饭，加一段路线。" in selected_intent
    assert "晚餐候选和证据边界" in selected_intent
    assert "两小时路线和时间切分" in selected_intent
    assert "source_backed_facts: 用户在奥克兰 CBD" in selected_intent
    assert "assumptions_or_inferences: 可以给出路线骨架" in selected_intent
    assert "blockers: 无法确认 19:30 营业状态" in selected_intent
    assert "覆盖晚餐、散步、时间切分和最终核实清单" in selected_intent
    assert result["action_directives"]["linguistic_directives"]["content_plan"] == {
        "semantic_content": (
            "Give the compact final plan with dinner, walking route, "
            "time split, and verification checklist."
        ),
        "rendering": "One visible chat bubble; cover all deliverables.",
    }


@pytest.mark.asyncio
async def test_l3_content_plan_logs_output(caplog) -> None:
    """Content-plan output should be visible in normal runtime logs."""

    llm = AsyncMock()
    llm.ainvoke.return_value = SimpleNamespace(
        content=json.dumps({
            "content_plan": {
                "semantic_content": "Answer directly.",
                "rendering": "One visible chat bubble; short.",
            },
        }),
    )

    with patch.object(l3_module, "_content_plan_agent_llm", llm):
        caplog.set_level(logging.INFO, logger=l3_module.__name__)
        result = await l3_module.call_content_plan_agent(_cognition_state())

    assert result["content_plan"] == {
        "semantic_content": "Answer directly.",
        "rendering": "One visible chat bubble; short.",
    }
    assert "Content plan output: entries=2" in caplog.text
    assert "Answer directly." in caplog.text


def test_legacy_background_artifact_no_handoff_produces_explicit_rejection() -> None:
    """Legacy background_artifact_request specs should get explicit failure results
    when no visible acknowledgement exists, not be silently dropped."""

    result = persona_module._background_no_handoff_result(
        {
            "kind": "background_artifact_request",
            "params": {"objective_summary": "Write a poem."},
        },
        _cognition_state(),
    )

    assert result["action_kind"] == "background_artifact_request"
    assert result["status"] == "failed"
    assert "visible acknowledgement missing" in result["result_summary"]
    assert result["acknowledgement_constraint"] == "promise_forbidden_explain_failure"
    assert result["handler_owner"] == "background_artifact"
    assert result["task_summary"] == "Write a poem."


def test_background_work_no_handoff_result_uses_task_brief() -> None:
    """background_work_request no-handoff should extract task_brief."""

    result = persona_module._background_no_handoff_result(
        {
            "kind": "background_work_request",
            "params": {"task_brief": "Generate a Fibonacci function."},
        },
        _cognition_state(),
    )

    assert result["action_kind"] == "background_work_request"
    assert result["status"] == "failed"
    assert result["handler_owner"] == "background_work"
    assert result["task_summary"] == "Generate a Fibonacci function."
