"""Tests for deterministic self-cognition delivery target binding."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from kazusa_ai_chatbot.self_cognition import models, projection, sources


def _source_kwargs(**overrides: Any) -> dict[str, Any]:
    """Build common resolver kwargs for target-binding tests."""

    kwargs: dict[str, Any] = {
        "platform": "qq",
        "source_platform_channel_id": "group-1",
        "source_channel_type": "group",
        "source_message_id": "msg-1",
        "source_ref": "case-1",
        "source_global_user_id": "global-target",
        "source_platform_bot_id": "bot-1",
        "source_character_name": "Character",
        "guild_id": "guild-1",
        "bot_permission_role": "user",
        "target_global_user_id": "global-target",
        "target_platform_user_id": "qq-target",
    }
    kwargs.update(overrides)
    return kwargs


def _private_channel_row() -> dict[str, str]:
    """Return a stored private channel row for the semantic target user."""

    row = {
        "platform": "qq",
        "platform_channel_id": "dm-1",
        "channel_type": "private",
        "platform_user_id": "qq-target",
    }
    return row


@pytest.mark.asyncio
async def test_resolver_prefers_known_private_channel() -> None:
    """Known private channel should win over the source channel."""

    calls: list[dict[str, Any]] = []

    async def latest_private_channel(**kwargs: Any) -> dict[str, str]:
        calls.append(dict(kwargs))
        return _private_channel_row()

    target = await sources.resolve_self_cognition_delivery_target(
        **_source_kwargs(get_latest_private_channel_func=latest_private_channel),
    )

    assert target["schema_version"] == "self_cognition_delivery_target.v1"
    assert target["source_kind"] == "target_private_channel"
    assert target["platform_channel_id"] == "dm-1"
    assert target["channel_type"] == "private"
    assert target["source_platform_channel_id"] == "group-1"
    assert target["source_channel_type"] == "group"
    assert target["target_global_user_id"] == "global-target"
    assert target["target_platform_user_id"] == "qq-target"
    assert target["fallback_reason"] == ""
    assert calls == [
        {
            "platform": "qq",
            "global_user_id": "global-target",
            "platform_user_id": "qq-target",
        }
    ]


@pytest.mark.asyncio
async def test_resolver_falls_back_to_source_channel_when_private_missing() -> None:
    """Missing private channel should bind the self-cognition source channel."""

    async def latest_private_channel(**kwargs: Any) -> None:
        del kwargs
        return None

    target = await sources.resolve_self_cognition_delivery_target(
        **_source_kwargs(get_latest_private_channel_func=latest_private_channel),
    )

    assert target["source_kind"] == "self_cognition_source_channel"
    assert target["platform_channel_id"] == "group-1"
    assert target["channel_type"] == "group"
    assert target["fallback_reason"] == "private_channel_unavailable"


@pytest.mark.asyncio
async def test_resolver_rejects_missing_private_and_source() -> None:
    """No known private channel and no source channel should fail binding."""

    async def latest_private_channel(**kwargs: Any) -> None:
        del kwargs
        return None

    failure = await sources.resolve_self_cognition_delivery_target(
        **_source_kwargs(
            source_platform_channel_id="",
            get_latest_private_channel_func=latest_private_channel,
        ),
    )

    assert failure["status"] == "target_binding_failed"
    assert failure["reason"] == "private_channel_unavailable_and_source_missing"


@pytest.mark.asyncio
async def test_resolver_does_not_infer_private_from_group_platform_user_id(
) -> None:
    """A group platform user id is not itself a known private channel."""

    async def latest_private_channel(**kwargs: Any) -> None:
        del kwargs
        return None

    target = await sources.resolve_self_cognition_delivery_target(
        **_source_kwargs(get_latest_private_channel_func=latest_private_channel),
    )

    assert target["source_kind"] == "self_cognition_source_channel"
    assert target["platform_channel_id"] == "group-1"
    assert target["platform_channel_id"] != "qq-target"


@pytest.mark.asyncio
async def test_resolver_rejects_invalid_source_channel_type() -> None:
    """Invalid source channel types cannot become fallback delivery targets."""

    async def latest_private_channel(**kwargs: Any) -> None:
        del kwargs
        return None

    failure = await sources.resolve_self_cognition_delivery_target(
        **_source_kwargs(
            source_channel_type="internal",
            get_latest_private_channel_func=latest_private_channel,
        ),
    )

    assert failure["status"] == "target_binding_failed"
    assert failure["reason"] == "private_channel_unavailable_and_source_invalid"


@pytest.mark.asyncio
async def test_resolver_normalizes_invalid_source_channel_when_private_exists(
) -> None:
    """Private target binding should provide valid source channel metadata."""

    async def latest_private_channel(**kwargs: Any) -> dict[str, str]:
        del kwargs
        return _private_channel_row()

    target = await sources.resolve_self_cognition_delivery_target(
        **_source_kwargs(
            source_platform_channel_id="",
            source_channel_type="internal",
            get_latest_private_channel_func=latest_private_channel,
        ),
    )

    assert target["source_kind"] == "target_private_channel"
    assert target["platform_channel_id"] == "dm-1"
    assert target["source_platform_channel_id"] == "dm-1"
    assert target["source_channel_type"] == "private"


@pytest.mark.asyncio
async def test_production_collectors_attach_delivery_target_before_cognition(
) -> None:
    """Active commitment cases should leave collection with a bound target."""

    now = datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc)

    async def list_active_commitments(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [
            {
                "unit_id": "promise-1",
                "due_at": "2026-05-17T05:56:00+00:00",
                "fact": "The target promised a dessert fare.",
                "global_user_id": "global-target",
            }
        ]

    async def get_conversation_history(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [
            {
                "platform": "qq",
                "platform_channel_id": "group-1",
                "channel_type": "group",
                "platform_message_id": "msg-1",
                "platform_user_id": "qq-target",
                "display_name": "Target User",
                "body_text": "I will pay the fare later.",
                "timestamp": "2026-05-17T05:50:00+00:00",
            }
        ]

    async def get_user_profile(global_user_id: str) -> dict[str, Any]:
        assert global_user_id == "global-target"
        return {"display_name": "Target User"}

    async def latest_private_channel(**kwargs: Any) -> dict[str, str]:
        del kwargs
        return _private_channel_row()

    cases = await sources.collect_active_commitment_cases(
        now=now,
        character_profile={"name": "Character"},
        max_cases=1,
        list_active_commitments_func=list_active_commitments,
        get_conversation_history_func=get_conversation_history,
        get_user_profile_func=get_user_profile,
        get_latest_private_channel_func=latest_private_channel,
    )

    assert cases[0]["target_binding_status"] == "bound"
    assert cases[0]["delivery_target"]["source_kind"] == (
        "target_private_channel"
    )
    assert cases[0]["delivery_target"]["platform_channel_id"] == "dm-1"


@pytest.mark.asyncio
async def test_collectors_return_failed_case_when_target_binding_fails() -> None:
    """Target binding failures should return auditable skipped cases."""

    now = datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc)
    event = {
        "event_id": "future-cognition-1",
        "tool": "trigger_future_cognition",
        "execute_at": "2026-05-17T05:57:00+00:00",
        "created_at": "2026-05-17T05:50:00+00:00",
        "status": "pending",
        "source_platform": "qq",
        "source_channel_id": "",
        "source_channel_type": "internal",
        "source_user_id": "global-target",
        "source_message_id": "msg-1",
        "source_platform_bot_id": "",
        "source_character_name": "Character",
        "guild_id": None,
        "bot_role": "user",
        "args": {
            "episode_type": "self_cognition",
            "continuation_objective": "Check whether the target is available.",
        },
    }

    async def list_due_events(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [event]

    async def latest_private_channel(**kwargs: Any) -> None:
        del kwargs
        return None

    cases = await sources.collect_scheduled_future_cognition_cases(
        now=now,
        character_profile={"name": "Character"},
        max_cases=1,
        list_due_events_func=list_due_events,
        get_latest_private_channel_func=latest_private_channel,
    )

    assert cases[0]["target_binding_status"] == "failed"
    assert cases[0]["target_binding_failure"]["status"] == (
        "target_binding_failed"
    )
    assert cases[0]["target_binding_failure"]["reason"] == (
        "private_channel_unavailable_and_source_missing"
    )


def test_delivery_target_never_enters_llm_facing_payloads() -> None:
    """Delivery-only routing metadata must stay out of LLM payloads."""

    case = {
        "case_name": models.CASE_TOPIC_RAG_FOLLOWUP,
        "case_id": "topic-followup-1",
        "idle_timestamp_utc": "2026-05-17T05:57:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-17T05:50:00+00:00",
        "trigger_kind": models.TRIGGER_BOUNDED_FOLLOWUP_TOPIC,
        "semantic_due_state": models.DUE_STATE_DUE_NOW,
        "actionability": "topic_followup_ready",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "group-1",
            "channel_type": "group",
            "user_id": "global-target",
            "platform_user_id": "qq-target",
            "display_name": "Target User",
        },
        "delivery_target": {
            "schema_version": "self_cognition_delivery_target.v1",
            "platform": "qq",
            "platform_channel_id": "dm-1",
            "channel_type": "private",
            "target_global_user_id": "global-target",
            "target_platform_user_id": "qq-target",
            "source_kind": "target_private_channel",
            "source_ref": "topic-followup-1",
            "source_platform_channel_id": "group-1",
            "source_channel_type": "group",
            "source_message_id": "msg-1",
            "source_global_user_id": "global-target",
            "source_platform_bot_id": "bot-1",
            "source_character_name": "Character",
            "guild_id": None,
            "bot_permission_role": "user",
            "fallback_reason": "",
        },
        "source_refs": [
            {
                "source_kind": "conversation_episode_state",
                "source_id": "episode-1",
                "summary": "A follow-up is open.",
            }
        ],
        "visible_context": [],
        "rag_query": "Find the open follow-up.",
    }

    source_packet = projection.build_source_packet(case)
    rag_request = projection.build_rag_request(case)
    serialized = json.dumps(
        {"source_packet": source_packet, "rag_request": rag_request},
        ensure_ascii=False,
    )

    assert "delivery_target" not in serialized
    assert "self_cognition_delivery_target.v1" not in serialized
    assert "dm-1" not in serialized
    assert "bot-1" not in serialized
