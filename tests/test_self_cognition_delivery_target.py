"""Tests for deterministic self-cognition delivery target binding."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from kazusa_ai_chatbot.calendar_scheduler import models as calendar_models
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


@pytest.mark.asyncio
async def test_group_resolver_binds_to_source_group_without_private_lookup(
) -> None:
    """Group-source self-cognition must target the same group channel."""

    async def latest_private_channel(**kwargs: Any) -> dict[str, str]:
        del kwargs
        raise AssertionError("group source must not use private-channel lookup")

    target = await sources.resolve_self_cognition_delivery_target(
        **_source_kwargs(get_latest_private_channel_func=latest_private_channel),
    )

    assert target["schema_version"] == "self_cognition_delivery_target.v1"
    assert target["source_kind"] == "self_cognition_source_channel"
    assert target["platform_channel_id"] == "group-1"
    assert target["channel_type"] == "group"
    assert target["source_platform_channel_id"] == "group-1"
    assert target["source_channel_type"] == "group"
    assert target["target_global_user_id"] == "global-target"
    assert target["target_platform_user_id"] == "qq-target"
    assert target["fallback_reason"] == ""


@pytest.mark.asyncio
async def test_private_resolver_binds_to_source_private_without_lookup() -> None:
    """Private-source self-cognition must target the same private channel."""

    async def latest_private_channel(**kwargs: Any) -> dict[str, str]:
        del kwargs
        raise AssertionError("private source must not use alternate lookup")

    target = await sources.resolve_self_cognition_delivery_target(
        **_source_kwargs(
            source_platform_channel_id="dm-source",
            source_channel_type="private",
            get_latest_private_channel_func=latest_private_channel,
        ),
    )

    assert target["source_kind"] == "self_cognition_source_channel"
    assert target["platform_channel_id"] == "dm-source"
    assert target["channel_type"] == "private"
    assert target["source_platform_channel_id"] == "dm-source"
    assert target["source_channel_type"] == "private"
    assert target["fallback_reason"] == ""


@pytest.mark.asyncio
async def test_resolver_rejects_missing_concrete_source_channel() -> None:
    """Cases without a concrete source channel should fail closed."""

    async def latest_private_channel(**kwargs: Any) -> dict[str, str]:
        del kwargs
        raise AssertionError("missing source must not use alternate lookup")

    failure = await sources.resolve_self_cognition_delivery_target(
        **_source_kwargs(
            source_platform_channel_id="",
            get_latest_private_channel_func=latest_private_channel,
        ),
    )

    assert failure["status"] == "target_binding_failed"
    assert failure["reason"] == "missing_delivery_target"


@pytest.mark.asyncio
async def test_resolver_does_not_infer_channel_from_group_platform_user_id(
) -> None:
    """A group platform user id is not itself a concrete channel target."""

    async def latest_private_channel(**kwargs: Any) -> dict[str, str]:
        del kwargs
        raise AssertionError("source-aligned binding must not use lookup")

    target = await sources.resolve_self_cognition_delivery_target(
        **_source_kwargs(get_latest_private_channel_func=latest_private_channel),
    )

    assert target["source_kind"] == "self_cognition_source_channel"
    assert target["platform_channel_id"] == "group-1"
    assert target["platform_channel_id"] != "qq-target"


@pytest.mark.asyncio
async def test_resolver_rejects_invalid_source_channel_type() -> None:
    """Invalid source channel types cannot become delivery targets."""

    async def latest_private_channel(**kwargs: Any) -> dict[str, str]:
        del kwargs
        raise AssertionError("invalid source must not use alternate lookup")

    failure = await sources.resolve_self_cognition_delivery_target(
        **_source_kwargs(
            source_channel_type="internal",
            get_latest_private_channel_func=latest_private_channel,
        ),
    )

    assert failure["status"] == "target_binding_failed"
    assert failure["reason"] == "missing_delivery_target"


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
        raise AssertionError("group source must not use private-channel lookup")

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
        "self_cognition_source_channel"
    )
    assert cases[0]["delivery_target"]["platform_channel_id"] == "group-1"
    assert cases[0]["delivery_target"]["channel_type"] == "group"
    assert cases[0]["delivery_target"]["fallback_reason"] == ""


@pytest.mark.asyncio
async def test_collectors_return_failed_case_when_target_binding_fails() -> None:
    """Target binding failures should return auditable skipped cases."""

    now = datetime(2026, 5, 17, 5, 57, tzinfo=timezone.utc)
    run = {
        "run_id": "calendar_run_future_1",
        "schedule_id": "calendar_schedule_future_1",
        "trigger_kind": calendar_models.TRIGGER_FUTURE_COGNITION,
        "due_at": "2026-05-17T05:57:00+00:00",
        "created_at": "2026-05-17T05:50:00+00:00",
        "status": calendar_models.RUN_STATUS_PENDING,
        "source_scope": {
            "source_platform": "qq",
            "source_channel_id": "",
            "source_channel_type": "internal",
            "source_user_id": "global-target",
            "source_message_id": "msg-1",
            "source_platform_bot_id": "",
            "source_character_name": "Character",
            "guild_id": None,
            "bot_role": "user",
        },
        "payload": {
            "episode_type": "self_cognition",
            "continuation_objective": "Check whether the target is available.",
        },
    }

    async def list_due_runs(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [run]

    async def latest_private_channel(**kwargs: Any) -> dict[str, str]:
        del kwargs
        raise AssertionError("invalid source must not use alternate lookup")

    async def user_profile(global_user_id: str) -> dict[str, Any]:
        return {"global_user_id": global_user_id, "relationship_state": 500}

    cases = await sources.collect_scheduled_future_cognition_cases(
        now=now,
        character_profile={"name": "Character"},
        max_cases=1,
        list_due_calendar_runs_func=list_due_runs,
        get_latest_private_channel_func=latest_private_channel,
        get_user_profile_func=user_profile,
    )

    assert cases[0]["target_binding_status"] == "failed"
    assert cases[0]["target_binding_failure"]["status"] == (
        "target_binding_failed"
    )
    assert cases[0]["target_binding_failure"]["reason"] == (
        "missing_delivery_target"
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
    }

    source_packet = projection.build_source_packet(case)
    serialized = json.dumps({"source_packet": source_packet}, ensure_ascii=False)

    assert "delivery_target" not in serialized
    assert "self_cognition_delivery_target.v1" not in serialized
    assert "dm-1" not in serialized
    assert "bot-1" not in serialized
