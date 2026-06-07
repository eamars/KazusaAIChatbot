"""Tests for persona_supervisor2.py — top-level orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.nodes.persona_supervisor2 import (
    _route_after_cognition,
    call_action_subgraph,
    persona_supervisor2,
    stage_3_no_response,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc


def _base_discord_state():
    """Minimal IMProcessState with all required keys."""
    storage_timestamp_utc = "2024-01-01T00:00:00+00:00"
    turn_clock = build_turn_clock_from_storage_utc(storage_timestamp_utc)
    local_time_context = turn_clock["local_time_context"]
    return {
        "storage_timestamp_utc": storage_timestamp_utc,
        "local_time_context": local_time_context,
        "user_name": "TestUser",
        "platform": "discord",
        "platform_message_id": "msg_123",
        "platform_user_id": "user_123",
        "global_user_id": "uuid-123",
        "user_input": "Hello",
        "message_envelope": {
            "body_text": "Hello",
            "raw_wire_text": "Hello",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": True,
        },
        "prompt_message_context": {
            "body_text": "Hello",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": True,
        },
        "user_multimedia_input": [],
        "user_profile": {"affinity": 500},
        "platform_bot_id": "bot_456",
        "character_name": "TestCharacter",
        "character_profile": {
            "name": "Character",
            "global_user_id": "character-uuid",
            "mood": "neutral",
            "global_vibe": "calm",
            "reflection_summary": "nothing notable",
        },
        "platform_channel_id": "chan_1",
        "channel_type": "group",
        "channel_name": "general",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "should_respond": True,
        "reason_to_respond": "user greeted",
        "use_reply_feature": False,
        "channel_topic": "greetings",
        "indirect_speech_context": "",
        "debug_modes": {},
        "cognitive_episode": build_text_chat_cognitive_episode(
            episode_id="episode-123",
            percept_id="percept-123",
            storage_timestamp_utc=storage_timestamp_utc,
            local_time_context=local_time_context,
            user_input="Hello",
            platform="discord",
            platform_channel_id="chan_1",
            channel_type="group",
            platform_message_id="msg_123",
            platform_user_id="user_123",
            global_user_id="uuid-123",
            user_name="TestUser",
            target_addressed_user_ids=["character-uuid"],
            target_broadcast=True,
        ),
    }


def _speak_action_spec() -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": "speak",
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
            "scope": {"delivery_mode": "visible_reply"},
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
        "reason": "A visible response is needed.",
    }


def _background_artifact_action_spec() -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": "background_artifact_request",
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
            "target_kind": "current_user",
            "target_id": None,
            "owner": "background_artifact",
            "scope": {"requester_display_name": "TestUser"},
        },
        "params": {
            "work_kind": "coding_snippet",
            "objective": "Generate a Fibonacci function snippet.",
            "input_summary": "The user asked for a simple Fibonacci generator.",
            "requested_delivery": "send_result_when_done",
            "max_output_chars": 3000,
        },
        "urgency": "background",
        "visibility": "private",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": "The character accepted bounded async snippet work.",
    }


def _background_artifact_pending_result() -> dict:
    return {
        "schema_version": "action_result.v1",
        "action_attempt_id": "action_attempt:background-artifact-001",
        "action_kind": "background_artifact_request",
        "handler_owner": "background_artifact",
        "status": "pending",
        "visibility": "private",
        "result_summary": "Background artifact job queued.",
        "result_refs": [
            {
                "schema_version": "evidence_ref.v1",
                "evidence_kind": "system_event",
                "evidence_id": "background_artifact_job:job-001",
                "owner": "background_artifact_job",
                "excerpt": "queued coding_snippet artifact request",
                "observed_at": "2024-01-01T00:00:00+00:00",
            }
        ],
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "completed_at": None,
    }


def _action_directives() -> dict:
    return {
        "contextual_directives": {
            "social_distance": "friendly",
            "emotional_intensity": "low",
            "vibe_check": "calm",
            "relational_dynamic": "direct reply",
        },
        "linguistic_directives": {
            "rhetorical_strategy": "answer directly",
            "linguistic_style": "brief",
            "accepted_user_preferences": [],
            "content_anchors": ["[DECISION] answer", "[SCOPE] short"],
            "forbidden_phrases": [],
        },
        "visual_directives": {
            "facial_expression": [],
            "body_language": [],
            "gaze_direction": [],
            "visual_vibe": [],
        },
    }


def _resolver_update(action_specs: list[dict]) -> dict:
    """Build a patched resolver result for persona graph plumbing tests."""

    return {
        "rag_result": {},
        "internal_monologue": "thinking...",
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "character_intent": "",
        "logical_stance": "",
        "judgment_note": "",
        "social_distance": "",
        "emotional_intensity": "",
        "vibe_check": "",
        "relational_dynamic": "",
        "action_specs": action_specs,
    }


@pytest.mark.asyncio
async def test_call_action_subgraph_returns_final_dialog():
    """call_action_subgraph wraps dialog_agent output correctly."""
    mock_dialog_result = {
        "final_dialog": ["Hello!", "How are you?"],
        "target_addressed_user_ids": ["uuid-123"],
        "target_broadcast": False,
        "mention_target_user": True,
    }

    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_l3_text_surface_handler",
            new_callable=AsyncMock,
            return_value={"action_directives": _action_directives()},
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
            new_callable=AsyncMock,
            return_value=mock_dialog_result,
        ),
    ):
        state = _base_discord_state()
        state["action_specs"] = [_speak_action_spec()]
        result = await call_action_subgraph(state)

    assert result["final_dialog"] == ["Hello!", "How are you?"]
    assert result["target_addressed_user_ids"] == ["uuid-123"]
    assert result["target_broadcast"] is False
    assert result["mention_target_user"] is True
    assert result["surface_outputs"][0]["surface_kind"] == "text"
    assert result["episode_trace"]["surface_outputs"][0]["fragments"] == [
        "Hello!",
        "How are you?",
    ]


@pytest.mark.asyncio
async def test_call_action_subgraph_empty_dialog():
    """call_action_subgraph handles empty dialog_agent output."""
    mock_dialog_result = {
        "final_dialog": [],
        "target_addressed_user_ids": [],
        "target_broadcast": False,
        "mention_target_user": False,
    }

    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_l3_text_surface_handler",
            new_callable=AsyncMock,
            return_value={"action_directives": _action_directives()},
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
            new_callable=AsyncMock,
            return_value=mock_dialog_result,
        ),
    ):
        state = _base_discord_state()
        state["action_specs"] = [_speak_action_spec()]
        result = await call_action_subgraph(state)

    assert result["final_dialog"] == []
    assert result["target_addressed_user_ids"] == []
    assert result["target_broadcast"] is False
    assert result["mention_target_user"] is False


def test_route_after_cognition_uses_l2d_speak_selection() -> None:
    """L2d action specs should own visible text routing when present."""

    state = {
        "action_specs": [_speak_action_spec()],
    }

    assert _route_after_cognition(state) == "respond"


def test_route_after_cognition_allows_no_visible_action() -> None:
    """A present but empty L2d action set means no text surface is required."""

    state = {
        "action_specs": [],
    }

    assert _route_after_cognition(state) == "silent"


@pytest.mark.asyncio
async def test_background_artifact_executes_before_l3_acknowledgement() -> None:
    """L3 acknowledgements should be based on pre-surface enqueue results."""

    state = _base_discord_state()
    action_specs = [_background_artifact_action_spec(), _speak_action_spec()]
    l3_states = []

    async def _execute_action_specs(
        selected_specs,
        *,
        storage_timestamp_utc,
        executed_action_attempt_ids=None,
        record_attempt_func=None,
        enqueue_background_artifact_func=None,
    ):
        del (
            storage_timestamp_utc,
            executed_action_attempt_ids,
            record_attempt_func,
            enqueue_background_artifact_func,
        )
        results = []
        for selected_spec in selected_specs:
            if selected_spec["kind"] == "background_artifact_request":
                results.append(_background_artifact_pending_result())
            elif selected_spec["kind"] == "speak":
                results.append({
                    "schema_version": "action_result.v1",
                    "action_attempt_id": "action_attempt:speak-001",
                    "action_kind": "speak",
                    "handler_owner": "l3_text",
                    "status": "executed",
                    "visibility": "user_visible",
                    "result_summary": "Text surface rendered.",
                    "result_refs": [],
                    "continuation": {
                        "schema_version": "action_continuation.v1",
                        "mode": "none",
                        "episode_type": None,
                        "max_depth": 0,
                        "include_result_as": None,
                    },
                    "completed_at": "2024-01-01T00:00:00+00:00",
                })
        return results

    async def _l3_text_surface_handler(l3_state):
        l3_states.append(dict(l3_state))
        return {"action_directives": _action_directives()}

    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={"decontexualized_input": "Hello"},
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_resolver_loop",
            new_callable=AsyncMock,
            return_value=_resolver_update(action_specs),
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2."
            "load_matching_pending_resume_into_state",
            new_callable=AsyncMock,
            side_effect=lambda persona_state: persona_state,
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2."
            "call_memory_lifecycle_update_handler",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2."
            "execute_action_specs_for_trace",
            _execute_action_specs,
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2."
            "call_l3_text_surface_handler",
            _l3_text_surface_handler,
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
            new_callable=AsyncMock,
            return_value={
                "final_dialog": ["I'll send it when it's ready."],
                "target_addressed_user_ids": ["uuid-123"],
                "target_broadcast": False,
                "mention_target_user": True,
            },
        ),
    ):
        result = await persona_supervisor2(state)

    assert l3_states[0]["pre_surface_action_results"] == [
        _background_artifact_pending_result()
    ]
    assert result["episode_trace"]["action_results"][0]["action_kind"] == (
        "background_artifact_request"
    )


@pytest.mark.asyncio
async def test_stage_3_no_response_records_private_trace_for_l2d() -> None:
    """No-speak L2d decisions should still leave consolidation evidence."""

    state = _base_discord_state()
    state["action_specs"] = []

    result = await stage_3_no_response(state)

    assert result["final_dialog"] == []
    assert result["surface_outputs"][0]["surface_kind"] == "private"
    assert result["episode_trace"]["trigger_source"] == "user_message"


@pytest.mark.asyncio
async def test_persona_supervisor2_returns_final_dialog_and_consolidation_state():
    """persona_supervisor2 should return dialog plus the consolidation snapshot."""
    state = _base_discord_state()

    # Mock graph nodes to avoid real LLM calls.
    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={"decontexualized_input": "Hello"},
        ) as m_decon,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_resolver_loop",
            new_callable=AsyncMock,
            return_value=_resolver_update([_speak_action_spec()]),
        ) as m_resolver,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_l3_text_surface_handler",
            new_callable=AsyncMock,
            return_value={"action_directives": _action_directives()},
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
            new_callable=AsyncMock,
            return_value={
                "final_dialog": ["Hi there!"],
                "target_addressed_user_ids": ["uuid-123"],
                "target_broadcast": False,
                "mention_target_user": True,
            },
        ) as m_dialog,
    ):
        result = await persona_supervisor2(state)

    assert "final_dialog" in result
    assert "future_promises" in result
    assert result["final_dialog"] == ["Hi there!"]
    assert result["target_addressed_user_ids"] == ["uuid-123"]
    assert result["target_broadcast"] is False
    assert result["mention_target_user"] is True
    assert result["future_promises"] == []
    assert result["consolidation_state"]["decontexualized_input"] == "Hello"
    assert result["consolidation_state"]["final_dialog"] == ["Hi there!"]
    assert result["consolidation_state"]["mention_target_user"] is True
    assert result["consolidation_state"]["reply_context"] == {}
    m_resolver.assert_awaited_once()


@pytest.mark.asyncio
async def test_persona_supervisor2_no_speak_action_skips_dialog():
    """A no-visible-action L2d decision should skip dialog."""

    state = _base_discord_state()

    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={"decontexualized_input": "Hello"},
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_resolver_loop",
            new_callable=AsyncMock,
            return_value={
                **_resolver_update([]),
                "internal_monologue": "choosing silence",
                "character_intent": "DISMISS",
                "logical_stance": "REFUSE",
            },
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
            new_callable=AsyncMock,
            return_value={
                "final_dialog": ["should not be used"],
                "target_addressed_user_ids": ["uuid-123"],
                "target_broadcast": False,
            },
        ) as m_dialog,
    ):
        result = await persona_supervisor2(state)

    assert result["should_respond"] is False
    assert result["final_dialog"] == []
    assert result["target_addressed_user_ids"] == []
    assert result["target_broadcast"] is False
    assert result["future_promises"] == []
    assert result["consolidation_state"]["should_respond"] is False
    assert result["consolidation_state"]["final_dialog"] == []
    m_dialog.assert_not_awaited()


@pytest.mark.asyncio
async def test_persona_supervisor2_scopes_group_history_before_persona_stages():
    """Only the decontextualizer should receive full recent channel history."""
    state = _base_discord_state()
    state["platform_user_id"] = "platform-user-a"
    state["global_user_id"] = "global-user-a"
    state["platform_bot_id"] = "platform-bot"
    state["chat_history_wide"] = [
        {
            "role": "user",
            "platform_user_id": "platform-user-a",
            "global_user_id": "global-user-a",
            "body_text": "current user secret",
            "addressed_to_global_user_ids": ["character-uuid"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": "2026-04-30T00:00:00+00:00",
        },
        {
            "role": "assistant",
            "platform_user_id": "platform-bot",
            "global_user_id": "character-uuid",
            "body_text": "current user reply",
            "addressed_to_global_user_ids": ["global-user-a"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": "2026-04-30T00:00:01+00:00",
        },
        {
            "role": "user",
            "platform_user_id": "platform-user-b",
            "global_user_id": "global-user-b",
            "body_text": "other user secret",
            "addressed_to_global_user_ids": ["character-uuid"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": "2026-04-30T00:00:02+00:00",
        },
        {
            "role": "assistant",
            "platform_user_id": "platform-bot",
            "global_user_id": "character-uuid",
            "body_text": "other user reply",
            "addressed_to_global_user_ids": ["global-user-b"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": "2026-04-30T00:00:03+00:00",
        },
    ]
    state["chat_history_recent"] = list(state["chat_history_wide"])

    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={"decontexualized_input": "Hello"},
        ) as m_decon,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_resolver_loop",
            new_callable=AsyncMock,
            return_value=_resolver_update([]),
        ) as m_resolver,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
            new_callable=AsyncMock,
            return_value={
                "final_dialog": ["Hi there!"],
                "target_addressed_user_ids": ["global-user-a"],
                "target_broadcast": False,
            },
        ),
    ):
        result = await persona_supervisor2(state)

    decon_state = m_decon.await_args.args[0]
    resolver_state = m_resolver.await_args.args[0]
    assert [
        row["body_text"] for row in decon_state["chat_history_recent"]
    ] == [
        "current user secret",
        "current user reply",
        "other user secret",
        "other user reply",
    ]
    assert [
        row["body_text"]
        for row in resolver_state["chat_history_recent"]
    ] == [
        "current user secret",
        "current user reply",
    ]
    assert [
        row["body_text"] for row in resolver_state["chat_history_wide"]
    ] == [
        "current user secret",
        "current user reply",
    ]
    assert result["consolidation_state"]["chat_history_recent"] == (
        resolver_state["chat_history_recent"]
    )


@pytest.mark.asyncio
async def test_persona_supervisor2_builds_scope_users_for_first_pass_only():
    """The decontextualizer should get a neutral roster without retry state."""

    state = _base_discord_state()
    state["platform"] = "qq"
    state["user_input"] = "还不报警抓他吗？"
    state["user_name"] = "Dangal"
    state["platform_user_id"] = "67889018"
    state["global_user_id"] = "745d7818-a9d3-4889-b7f3-8555078a2061"
    state["platform_bot_id"] = "3768713357"
    state["character_profile"] = {
        "name": "杏山千纱",
        "global_user_id": "00000000-0000-4000-8000-000000000001",
        "mood": "neutral",
        "global_vibe": "calm",
        "reflection_summary": "nothing notable",
    }
    state["message_envelope"]["body_text"] = state["user_input"]
    state["message_envelope"]["raw_wire_text"] = state["user_input"]
    state["message_envelope"]["mentions"] = [
        {
            "platform_user_id": "673225019",
            "global_user_id": "256e8a10-c406-47e9-ac8f-efd270d18160",
            "display_name": "蚝爹油",
            "entity_kind": "user",
        }
    ]
    state["message_envelope"]["addressed_to_global_user_ids"] = [
        "00000000-0000-4000-8000-000000000001",
        "256e8a10-c406-47e9-ac8f-efd270d18160",
    ]
    state["prompt_message_context"] = {
        "body_text": state["user_input"],
        "mentions": [
            {
                "platform_user_id": "673225019",
                "global_user_id": "256e8a10-c406-47e9-ac8f-efd270d18160",
                "display_name": "蚝爹油",
                "entity_kind": "user",
            }
        ],
        "attachments": [],
        "addressed_to_global_user_ids": state["message_envelope"][
            "addressed_to_global_user_ids"
        ],
        "broadcast": False,
    }
    state["reply_context"] = {
        "reply_to_platform_user_id": "13579",
        "reply_to_display_name": "回复对象",
        "reply_excerpt": "前文摘录",
    }
    state["chat_history_wide"] = [
        {
            "role": "user",
            "name": "Dangal-old",
            "display_name": "Dangal-old",
            "platform_user_id": "67889018",
            "global_user_id": "745d7818-a9d3-4889-b7f3-8555078a2061",
            "body_text": "反正现在有AI",
            "addressed_to_global_user_ids": [
                "00000000-0000-4000-8000-000000000001",
            ],
            "mentions": [],
            "broadcast": False,
            "reply_context": {},
            "timestamp": "2026-05-08T01:48:45+00:00",
        },
        {
            "role": "user",
            "name": "蚝爹油",
            "display_name": "蚝爹油",
            "platform_user_id": "673225019",
            "global_user_id": "256e8a10-c406-47e9-ac8f-efd270d18160",
            "body_text": "把对方解决掉也是解决问题的方式之一哦",
            "addressed_to_global_user_ids": [
                "00000000-0000-4000-8000-000000000001",
            ],
            "mentions": [],
            "broadcast": False,
            "reply_context": {},
            "timestamp": "2026-05-08T01:48:58+00:00",
        },
        {
            "role": "assistant",
            "name": "杏山千纱",
            "display_name": "杏山千纱",
            "platform_user_id": "3768713357",
            "global_user_id": "00000000-0000-4000-8000-000000000001",
            "body_text": "这个一点都不好笑。",
            "addressed_to_global_user_ids": [
                "256e8a10-c406-47e9-ac8f-efd270d18160",
            ],
            "mentions": [],
            "broadcast": False,
            "reply_context": {},
            "timestamp": "2026-05-08T01:49:02+00:00",
        },
    ]
    state["chat_history_recent"] = list(state["chat_history_wide"])

    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={
                "decontexualized_input": "@杏山千纱 还不报警抓他吗？",
                "referents": [
                    {
                        "phrase": "他",
                        "referent_role": "object",
                        "status": "unresolved",
                    },
                ],
            },
        ) as m_decon,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_resolver_loop",
            new_callable=AsyncMock,
            return_value={
                **_resolver_update([]),
                "internal_monologue": "choosing silence",
                "character_intent": "CLARIFY",
                "logical_stance": "UNKNOWN_REFERENT",
            },
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_memory_lifecycle_update_handler",
            new_callable=AsyncMock,
            return_value={},
        ),
    ):
        await persona_supervisor2(state)

    assert m_decon.await_count == 1
    decon_state = m_decon.await_args.args[0]
    scope_users = decon_state["scope_users"]
    by_global = {
        row["global_user_id"]: row
        for row in scope_users
        if row["global_user_id"]
    }
    by_platform = {
        row["platform_user_id"]: row
        for row in scope_users
        if row["platform_user_id"]
    }
    assert by_global["00000000-0000-4000-8000-000000000001"] == {
        "display_name": "杏山千纱",
        "platform_user_id": "3768713357",
        "global_user_id": "00000000-0000-4000-8000-000000000001",
        "aliases": [],
    }
    assert by_global["745d7818-a9d3-4889-b7f3-8555078a2061"][
        "display_name"
    ] == "Dangal"
    assert by_global["256e8a10-c406-47e9-ac8f-efd270d18160"][
        "display_name"
    ] == "蚝爹油"
    assert by_platform["13579"] == {
        "display_name": "回复对象",
        "platform_user_id": "13579",
        "global_user_id": "",
        "aliases": [],
    }
    for row in scope_users:
        assert set(row) == {
            "display_name",
            "platform_user_id",
            "global_user_id",
            "aliases",
        }
    assert all("retry" not in key for key in decon_state)


@pytest.mark.asyncio
async def test_persona_supervisor2_no_remember_skips_consolidation():
    """no_remember stays a service concern; supervisor still returns the consolidation snapshot."""
    state = _base_discord_state()
    state["debug_modes"] = {"no_remember": True}

    with (
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_msg_decontexualizer",
            new_callable=AsyncMock,
            return_value={"decontexualized_input": "Hello"},
        ) as m_decon,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_cognition_resolver_loop",
            new_callable=AsyncMock,
            return_value=_resolver_update([_speak_action_spec()]),
        ) as m_resolver,
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.call_l3_text_surface_handler",
            new_callable=AsyncMock,
            return_value={"action_directives": _action_directives()},
        ),
        patch(
            "kazusa_ai_chatbot.nodes.persona_supervisor2.dialog_agent",
            new_callable=AsyncMock,
            return_value={
                "final_dialog": ["Hi there!"],
                "target_addressed_user_ids": ["uuid-123"],
                "target_broadcast": False,
            },
        ) as m_dialog,
    ):
        result = await persona_supervisor2(state)

    assert result["final_dialog"] == ["Hi there!"]
    assert result["target_addressed_user_ids"] == ["uuid-123"]
    assert result["target_broadcast"] is False
    assert result["future_promises"] == []
    assert result["consolidation_state"]["debug_modes"] == {"no_remember": True}
