"""Live LLM reproduction of self-cognition duplicate response after direct reply.

Reproduces the failure mode where a group_chat_review window contains the bot's
own direct reply, and the cognition pipeline decides to speak again on the same
topic.  This test uses a real historical window from production that produced a
confirmed near-duplicate message.

Run one case at a time and inspect the trace:
    pytest tests/test_self_cognition_duplicate_response_live_llm.py -q -s -m "live_llm"
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.db import close_db, get_character_profile
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.interaction_style_images import (
    build_group_engagement_action_context,
)
from kazusa_ai_chatbot.reflection_cycle.activity_windows import (
    build_group_activity_windows,
)
from kazusa_ai_chatbot.reflection_cycle.group_scene_digest import (
    build_group_scene_digest,
)
from kazusa_ai_chatbot.reflection_cycle.models import ReflectionScopeInput
from kazusa_ai_chatbot.reflection_cycle.selector import build_scope_ref
from kazusa_ai_chatbot.self_cognition import models, projection, runner, sources
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime
from kazusa_ai_chatbot.utils import build_interaction_history_recent
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

logger = logging.getLogger(__name__)

_TRACE_TAG = os.environ.get("SC_DUPLICATE_TRACE_TAG", "baseline")

_DUPLICATE_WINDOW_SOURCE_ID = (
    "scope_e3945048f57d:"
    "2026-06-10T08:00:00+00:00:"
    "2026-06-10T08:15:00+00:00"
)
_DUPLICATE_WINDOW_START = "2026-06-10T08:00:00+00:00"
_DUPLICATE_WINDOW_END = "2026-06-10T08:15:00+00:00"
_DUPLICATE_EVENT_OCCURRED_AT = "2026-06-10T08:30:55.201466+00:00"
_CHANNEL_ID = "905393941"


async def test_live_sc_duplicate_response_window_with_own_reply() -> None:
    """Replay the window that produced a near-duplicate after a direct reply.

    This window contains the bot's own assistant message at 08:06 followed by
    user messages at 08:10.  The self-cognition instruction says "I haven't
    chimed in" but the visible_context shows the bot's own reply.

    Expected observation: the LLM decides to speak because the instruction
    is misleading.
    """

    await _skip_if_llm_unavailable()
    try:
        character_profile = await get_character_profile()
        case = await _rebuild_historical_case(character_profile)
        if case is None:
            pytest.skip("Historical window messages not found in database")

        source_packet = projection.build_source_packet(case)
        rendered_packet = projection.render_source_packet_text(source_packet)

        logger.info("=== RENDERED PACKET (what the LLM sees) ===")
        logger.info("\n%s", rendered_packet)

        cognition_state = runner._build_cognition_state(case, rendered_packet)
        l2d_state, stage_outputs = await _run_cognition_stages(cognition_state)
        l2d_prompt_payload = build_action_selection_payload_text(l2d_state)
        l2d_output = await select_semantic_actions(l2d_state)
        action_specs = l2d_output["action_specs"]
        observed_speak = _observed_user_visible_speak(action_specs)
        speak_reasons = _observed_speak_reasons(action_specs)

        logger.info("=== COGNITION STAGES ===")
        logger.info(
            "L1 emotional_appraisal: %s",
            stage_outputs["l1_subconscious"].get("emotional_appraisal"),
        )
        logger.info(
            "L2a internal_monologue: %s",
            stage_outputs["l2a_conscious_framing"].get("internal_monologue"),
        )
        logger.info(
            "L2a character_intent: %s",
            stage_outputs["l2a_conscious_framing"].get("character_intent"),
        )
        logger.info(
            "L2c1 judgment_note: %s",
            stage_outputs["l2c1_judgment_synthesis"].get("judgment_note"),
        )
        logger.info("=== L2d DECISION ===")
        logger.info("observed_user_visible_speak: %s", observed_speak)
        logger.info("speak_reasons: %s", speak_reasons)
        logger.info("action_spec_count: %d", len(action_specs))

        visible_context = source_packet["visible_context"]
        has_own_assistant_msg = any(
            row.get("role") == "assistant" for row in visible_context
        )
        instruction = source_packet["instruction"]

        logger.info("=== DUPLICATE ANALYSIS ===")
        logger.info("instruction: %s", instruction)
        logger.info("has_own_assistant_msg_in_visible_context: %s", has_own_assistant_msg)
        logger.info("assistant_presence label: %s",
            source_packet.get("group_activity_window", {})
            .get("semantic_labels", {}).get("assistant_presence"),
        )
        logger.info("bot_addressing label: %s",
            source_packet.get("group_activity_window", {})
            .get("semantic_labels", {}).get("bot_addressing"),
        )

        if has_own_assistant_msg and "没有插话" in instruction:
            logger.warning(
                "CONTRADICTION: instruction says '没有插话' but visible_context "
                "contains the bot's own assistant message"
            )

        trace_path = write_llm_trace(
            "self_cognition_duplicate_response_live_llm",
            f"window_with_own_reply_{_TRACE_TAG}",
            {
                "case_id": case["case_id"],
                "instruction": instruction,
                "instruction_contradiction": (
                    has_own_assistant_msg and "没有插话" in instruction
                ),
                "has_own_assistant_msg_in_visible_context": has_own_assistant_msg,
                "observed_user_visible_speak": observed_speak,
                "observed_user_visible_speak_reasons": speak_reasons,
                "visible_context": visible_context,
                "source_packet": source_packet,
                "rendered_packet": rendered_packet,
                "l2d_prompt_payload": l2d_prompt_payload,
                "l2d_output": l2d_output,
                "cognition_stages": stage_outputs,
            },
        )
        logger.info("trace_path: %s", trace_path)

        assert isinstance(action_specs, list)
        assert len(action_specs) <= 3

    finally:
        await close_db()


async def _rebuild_historical_case(
    character_profile: dict[str, Any],
) -> models.SelfCognitionCase | None:
    """Rebuild the group review case from the historical window."""

    db = await get_db()
    window_start = _DUPLICATE_WINDOW_START
    window_end = _DUPLICATE_WINDOW_END
    cursor = (
        db.conversation_history
        .find(
            {
                "platform_channel_id": _CHANNEL_ID,
                "timestamp": {"$gte": window_start, "$lt": window_end},
                "role": {"$in": ["assistant", "user"]},
                "channel_type": "group",
            },
            {
                "_id": 0,
                "platform": 1,
                "platform_channel_id": 1,
                "channel_type": 1,
                "role": 1,
                "platform_user_id": 1,
                "global_user_id": 1,
                "display_name": 1,
                "body_text": 1,
                "timestamp": 1,
                "platform_message_id": 1,
                "message_id": 1,
                "addressed_to_global_user_ids": 1,
                "mentions": 1,
                "is_directed_at_character": 1,
            },
        )
        .sort("timestamp", 1)
    )
    messages = await cursor.to_list(length=None)
    if not messages:
        return None

    first = messages[0]
    platform = str(first["platform"])
    platform_channel_id = str(first["platform_channel_id"])
    channel_type = str(first["channel_type"])
    scope_ref = build_scope_ref(platform, platform_channel_id, channel_type)
    matching = [m for m in messages if build_scope_ref(
        str(m["platform"]),
        str(m["platform_channel_id"]),
        str(m["channel_type"]),
    ) == scope_ref]
    if not matching:
        return None

    scope = ReflectionScopeInput(
        scope_ref=scope_ref,
        platform=platform,
        platform_channel_id=platform_channel_id,
        channel_type=channel_type,
        assistant_message_count=sum(1 for m in matching if m.get("role") == "assistant"),
        user_message_count=sum(1 for m in matching if m.get("role") == "user"),
        total_message_count=len(matching),
        first_timestamp=str(matching[0]["timestamp"]),
        last_timestamp=str(matching[-1]["timestamp"]),
        messages=matching,
    )

    ws = parse_storage_utc_datetime(window_start)
    we = parse_storage_utc_datetime(window_end)
    occurred = parse_storage_utc_datetime(_DUPLICATE_EVENT_OCCURRED_AT)
    windows = build_group_activity_windows(
        scope=scope,
        window_start=ws,
        window_end=we,
        now=occurred,
        character_global_user_id=str(
            character_profile.get("global_user_id", "") or ""
        ),
        platform_bot_id=str(character_profile.get("platform_bot_id", "") or ""),
    )
    if not windows:
        return None

    window = windows[0]
    case = sources._build_group_review_case(
        window,
        character_profile=character_profile,
        now=occurred,
    )
    scene_digest = await build_group_scene_digest(window)
    if isinstance(scene_digest, dict) and isinstance(
        scene_digest.get("digest"), str,
    ):
        case["conversation_progress"]["group_scene_digest"] = {
            "digest": scene_digest["digest"].strip(),
        }
    return case


async def _run_cognition_stages(
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run L1/L2 cognition stages and return state before L2d."""

    initial_state = _cognition_initial_state(state)
    l1_output = await call_cognition_subconscious(initial_state)
    after_l1 = {**initial_state, **l1_output}

    l2a_output = await call_cognition_consciousness(after_l1)
    l2b_output = await call_boundary_core_agent(after_l1)
    after_l2b = {**after_l1, **l2b_output}

    after_l2a_l2b = {**after_l1, **l2a_output, **l2b_output}
    l2c1_output = await call_judgment_core_agent(after_l2a_l2b)
    l2c2_output = await call_social_context_appraisal(after_l2b)

    before_group_engagement = {
        **after_l2a_l2b,
        **l2c1_output,
        **l2c2_output,
    }
    group_engagement_context = await build_group_engagement_action_context(
        channel_type=str(state["channel_type"]),
        platform=str(state["platform"]),
        platform_channel_id=str(state["platform_channel_id"]),
    )
    group_engagement_output = {
        "group_engagement_action_context": group_engagement_context,
    }
    l2d_state = {
        **before_group_engagement,
        **group_engagement_output,
    }
    stage_outputs = {
        "l1_subconscious": l1_output,
        "l2a_conscious_framing": l2a_output,
        "l2b_boundary_appraisal": l2b_output,
        "l2c1_judgment_synthesis": l2c1_output,
        "l2c2_social_context_appraisal": l2c2_output,
        "group_engagement_action_context": group_engagement_output,
    }
    return (l2d_state, stage_outputs)


def _cognition_initial_state(state: dict[str, Any]) -> dict[str, Any]:
    """Build the same cognition entry state used by the shared subgraph."""

    interaction_history_recent = build_interaction_history_recent(
        state["chat_history_wide"],
        state["platform_user_id"],
        state["platform_bot_id"],
        state["global_user_id"],
    )
    initial_state = {
        "character_profile": state["character_profile"],
        "storage_timestamp_utc": state["storage_timestamp_utc"],
        "local_time_context": state["local_time_context"],
        "user_input": state["user_input"],
        "prompt_message_context": state["prompt_message_context"],
        "platform": state["platform"],
        "platform_channel_id": state["platform_channel_id"],
        "channel_type": state["channel_type"],
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "user_profile": state["user_profile"],
        "platform_bot_id": state["platform_bot_id"],
        "chat_history_recent": interaction_history_recent,
        "reply_context": state["reply_context"],
        "indirect_speech_context": state["indirect_speech_context"],
        "channel_topic": state["channel_topic"],
        "conversation_progress": state.get("conversation_progress"),
        "promoted_reflection_context": state.get("promoted_reflection_context"),
        "decontexualized_input": state["decontexualized_input"],
        "referents": state["referents"],
        "rag_result": state["rag_result"],
    }
    cognitive_episode = state.get("cognitive_episode")
    if cognitive_episode is not None:
        initial_state["cognitive_episode"] = cognitive_episode
    return initial_state


def _observed_user_visible_speak(action_specs: list[dict[str, Any]]) -> bool:
    """Return whether L2d selected a user-visible speak surface."""

    for spec in action_specs:
        if spec.get("kind") != "speak":
            continue
        if spec.get("visibility") == "user_visible":
            return True
    return False


def _observed_speak_reasons(action_specs: list[dict[str, Any]]) -> list[str]:
    """Return prompt-facing reasons for user-visible speak selections."""

    reasons: list[str] = []
    for spec in action_specs:
        if spec.get("kind") != "speak":
            continue
        if spec.get("visibility") != "user_visible":
            continue
        reason = spec.get("reason")
        if isinstance(reason, str) and reason.strip():
            reasons.append(reason.strip())
    return reasons


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured cognition endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}; {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{COGNITION_LLM_BASE_URL}"
        )
