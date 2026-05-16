"""Tests for the self-cognition action-spec send-message bridge."""

from __future__ import annotations

from datetime import datetime, timezone

from kazusa_ai_chatbot.action_spec.evaluator import (
    build_raw_tool_call_from_action_spec,
)
from kazusa_ai_chatbot.action_spec.models import validate_action_spec
from kazusa_ai_chatbot.dispatcher import (
    AdapterRegistry,
    DispatchContext,
    SendResult,
    ToolCallEvaluator,
    ToolRegistry,
    build_send_message_tool,
)
from kazusa_ai_chatbot.dispatcher.task import RawToolCall
from kazusa_ai_chatbot.self_cognition.handoff import (
    build_send_message_action_spec,
)


class _NoopAdapter:
    """Minimal adapter registration for dispatcher evaluator compatibility."""

    platform = "qq"

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
    ) -> SendResult:
        del channel_id, text, channel_type, reply_to_msg_id
        return_value = SendResult(
            platform=self.platform,
            channel_id="673225019",
            message_id="msg-001",
            sent_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        )
        return return_value


def _action_spec() -> dict:
    return {
        "schema_version": "action_spec.v1",
        "kind": "send_message",
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "memory_unit",
                "ref_id": "promise-001",
                "owner": "user_memory_units",
                "relationship": "basis",
                "evidence_refs": [],
            }
        ],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "current_channel",
            "target_id": None,
            "owner": "dispatcher",
            "scope": {"channel_relation": "same"},
        },
        "params": {
            "target_channel": "same",
            "text": "Checking in now.",
            "execute_at": None,
            "delivery_mentions": [
                {
                    "entity_kind": "user",
                    "placement": "prefix",
                    "platform_user_id": "673225019",
                    "global_user_id": "user-001",
                    "display_name": "提拉米苏",
                    "requested_by": "self_cognition",
                }
            ],
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
        "reason": "The character selected an outward follow-up.",
    }


def _self_cognition_case() -> dict:
    return {
        "case_name": "active_commitment_past_due",
        "case_id": "active_commitment:promise-001:2026-05-07",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "673225019",
            "channel_type": "private",
            "user_id": "user-001",
            "platform_user_id": "673225019",
            "display_name": "提拉米苏",
        },
        "source_refs": [
            {
                "source_kind": "user_memory_unit",
                "source_id": "promise-001",
                "due_at": "2026-05-07T00:00:00+00:00",
                "summary": "揭晓香料谜底",
            }
        ],
        "character_profile": {"name": "杏山千纱"},
        "platform_bot_id": "bot-001",
    }


def _action_candidate() -> dict:
    return {
        "target_platform": "qq",
        "target_channel": "673225019",
        "target_channel_type": "private",
        "text": "Checking in now.",
        "execute_at": None,
        "dispatch_shape": "send_message",
        "production_handoff": False,
        "delivery_mentions": [
            {
                "entity_kind": "user",
                "placement": "prefix",
                "platform_user_id": "673225019",
                "global_user_id": "user-001",
                "display_name": "提拉米苏",
                "requested_by": "self_cognition",
            }
        ],
    }


def _dispatcher_evaluator() -> ToolCallEvaluator:
    tool_registry = ToolRegistry()
    tool_registry.register(build_send_message_tool())
    adapters = AdapterRegistry()
    adapters.register(_NoopAdapter())
    evaluator = ToolCallEvaluator(tool_registry, adapters)
    return evaluator


def _dispatch_context() -> DispatchContext:
    return DispatchContext(
        source_platform="qq",
        source_channel_id="673225019",
        source_user_id="user-001",
        source_message_id="self_cognition:promise-001",
        guild_id=None,
        bot_permission_role="user",
        now=datetime(2026, 5, 16, tzinfo=timezone.utc),
        source_channel_type="private",
    )


def test_send_message_action_spec_bridges_to_raw_tool_call() -> None:
    """Validated send actions should cross into dispatcher as RawToolCall only."""

    raw_call = build_raw_tool_call_from_action_spec(_action_spec())

    assert isinstance(raw_call, RawToolCall)
    assert raw_call.tool == "send_message"
    assert raw_call.args["target_channel"] == "same"
    assert raw_call.args["text"] == "Checking in now."
    assert raw_call.args["delivery_mentions"][0]["display_name"] == "提拉米苏"
    assert "handler_id" not in raw_call.args


def test_self_cognition_candidate_builds_valid_action_spec() -> None:
    """Legacy self-cognition delivery must cross the shared action contract."""

    action_spec = build_send_message_action_spec(
        _self_cognition_case(),
        _action_candidate(),
    )
    validated = validate_action_spec(action_spec)

    assert validated["kind"] == "send_message"
    assert validated["source_refs"][0]["ref_kind"] == "memory_unit"
    assert validated["params"]["target_channel"] == "673225019"
    assert validated["params"]["text"] == "Checking in now."
    assert validated["target"]["owner"] == "dispatcher"


def test_send_message_bridge_shape_is_accepted_by_dispatcher_evaluator() -> None:
    """The bridge must preserve existing dispatcher validation ownership."""

    raw_call = build_raw_tool_call_from_action_spec(_action_spec())
    result = _dispatcher_evaluator().evaluate(raw_call, _dispatch_context())

    assert result.ok is True
    assert result.task is not None
    assert result.task.tool == "send_message"
    assert result.task.args["target_channel"] == "673225019"
    assert result.task.args["target_channel_type"] == "private"
    assert result.task.args["delivery_mentions"][0]["global_user_id"] == "user-001"
