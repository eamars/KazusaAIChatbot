"""Persona-aware settled relevance for an assembled logical turn."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import json
import logging
import time
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.config import (
    CHARACTER_GLOBAL_USER_ID,
    RELEVANCE_AGENT_LLM_API_KEY,
    RELEVANCE_AGENT_LLM_BASE_URL,
    RELEVANCE_AGENT_LLM_MAX_COMPLETION_TOKENS,
    RELEVANCE_AGENT_LLM_MODEL,
    RELEVANCE_AGENT_LLM_THINKING_ENABLED,
    SETTLED_RELEVANCE_MAX_COMPLETION_TOKENS,
    SETTLED_RELEVANCE_MAX_INPUT_CHARS,
)
from kazusa_ai_chatbot.conversation_history_prompt_projection import (
    project_conversation_history_for_llm,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.state import IMProcessState
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime
from kazusa_ai_chatbot.utils import build_affinity_block, parse_llm_json_output


logger = logging.getLogger(__name__)

SETTLED_RELEVANCE_MAX_COMPLETION_TOKENS = (
    SETTLED_RELEVANCE_MAX_COMPLETION_TOKENS
)

_GROUP_ATTENTION_LOW = "low_noise"
_GROUP_ATTENTION_MEDIUM = "medium_noise"
_GROUP_ATTENTION_HIGH = "high_noise"
_GROUP_ATTENTION_CHAOTIC = "chaotic_noise"
_ACTIVE_WINDOW_SECONDS = 180
_ACTIVE_WINDOW_MAX_MESSAGES = 10
_FRAGMENT_TOTAL_CHARS = 6000
_HISTORY_TOTAL_CHARS = 4000
_CONTEXT_TOTAL_CHARS = 2000


class SettledRelevanceDecision(TypedDict):
    """Validated persona-aware response action."""

    response_action: Literal["ignore", "proceed", "wait"]
    reason_to_respond: str
    use_reply_feature: bool
    channel_topic: str
    indirect_speech_context: str


SettledRelevanceState = Mapping[str, Any]


_SETTLED_SYSTEM_PROMPT = '''You are the persona-aware settled relevance judge for an assembled user turn.
Decide whether the active character has a grounded reason to speak in the
current scene. Treat typed target and reply labels as authoritative evidence;
preserve the latest correction and distinguish the assembled turn from fresh
history. Ignore a turn when it is third-party traffic, redundant because a
fresh answer already resolved it, or lacks a reason for this character to
speak. Proceed when the turn is stable and relevant. Use wait only when more
observation is available and the turn still appears incomplete. At the hard
observation boundary choose only ignore or proceed.

Use only the supplied bounded projection. Raw platform IDs, turn IDs,
timestamps, deadlines, counters, queue state, futures, handles, base64 media,
and operational telemetry are absent by design.

Return exactly one JSON object:
{"response_action":"ignore|proceed|wait","reason_to_respond":"at most 180 characters","use_reply_feature":false,"channel_topic":"at most 60 characters","indirect_speech_context":"at most 100 characters"}'''

_llm_interface = LLInterface()
_relevance_agent_llm = LLInterface()
_relevance_agent_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="RELEVANCE_AGENT_LLM",
    base_url=RELEVANCE_AGENT_LLM_BASE_URL,
    api_key=RELEVANCE_AGENT_LLM_API_KEY,
    model=RELEVANCE_AGENT_LLM_MODEL,
    temperature=0,
    top_p=1.0,
    top_k=None,
    max_completion_tokens=min(
        RELEVANCE_AGENT_LLM_MAX_COMPLETION_TOKENS,
        SETTLED_RELEVANCE_MAX_COMPLETION_TOKENS,
    ),
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=RELEVANCE_AGENT_LLM_THINKING_ENABLED,
    ),
)


def _clip_text(value: object, limit: int) -> str:
    """Clip model-facing text while retaining its head and tail."""

    if not isinstance(value, str):
        return_value = ""
        return return_value

    clean_value = value.strip()
    if len(clean_value) <= limit:
        return_value = clean_value
        return return_value

    head_length = max(1, (limit - 3) // 2)
    tail_length = max(1, limit - 3 - head_length)
    return_value = (
        clean_value[:head_length]
        + "..."
        + clean_value[-tail_length:]
    )
    return return_value


def _parse_history_timestamp(value: object) -> datetime | None:
    """Parse one history timestamp for structural attention projection."""

    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed = parse_storage_utc_datetime(value)
    except ValueError as exc:
        logger.debug(f"Ignoring history row with invalid timestamp: {exc}")
        return None
    return_value = parsed
    return return_value


def _active_history_window(chat_history_wide: list[dict]) -> list[dict]:
    """Return the recent history suffix used for attention descriptors."""

    capped = list(chat_history_wide[-_ACTIVE_WINDOW_MAX_MESSAGES:])
    if not capped:
        return_value: list[dict] = []
        return return_value

    newest_timestamp = _parse_history_timestamp(capped[-1].get("timestamp"))
    if newest_timestamp is None:
        return capped

    active_reversed: list[dict] = []
    for row in reversed(capped):
        timestamp = _parse_history_timestamp(row.get("timestamp"))
        if timestamp is None:
            return capped
        delta_seconds = (newest_timestamp - timestamp).total_seconds()
        if delta_seconds <= _ACTIVE_WINDOW_SECONDS:
            active_reversed.append(row)
            continue
        break

    return_value = list(reversed(active_reversed))
    return return_value


def _is_addressed_to_character(row: Mapping[str, Any], character_id: str) -> bool:
    """Return whether typed addressee evidence names the character."""

    addressed_to = row.get("addressed_to_global_user_ids")
    if not isinstance(addressed_to, Sequence) or isinstance(
        addressed_to,
        (str, bytes),
    ):
        return_value = False
        return return_value
    return_value = character_id in addressed_to
    return return_value


def _is_non_bot_user_row(row: Mapping[str, Any], platform_bot_id: str) -> bool:
    """Return whether a row is a user message from another participant."""

    return_value = (
        row.get("role") == "user"
        and row.get("platform_user_id") != platform_bot_id
    )
    return return_value


def build_group_attention_context(
    *,
    chat_history_wide: list[dict],
    platform_bot_id: str,
    character_global_user_id: str = CHARACTER_GLOBAL_USER_ID,
) -> dict[str, str]:
    """Project recent group activity as descriptive scene evidence."""

    active_rows = _active_history_window(chat_history_wide)
    user_rows = [
        row
        for row in active_rows
        if _is_non_bot_user_row(row, platform_bot_id)
    ]
    speakers = {
        str(row.get("platform_user_id"))
        for row in user_rows
        if row.get("platform_user_id")
    }
    direct_count = sum(
        _is_addressed_to_character(row, character_global_user_id)
        for row in user_rows
    )
    if len(user_rows) >= 6 or len(speakers) >= 4:
        attention = _GROUP_ATTENTION_CHAOTIC
    elif len(user_rows) >= 4 or len(speakers) >= 3:
        attention = _GROUP_ATTENTION_HIGH
    elif len(user_rows) >= 2 or len(speakers) >= 2:
        attention = _GROUP_ATTENTION_MEDIUM
    else:
        attention = _GROUP_ATTENTION_LOW

    if direct_count > 0 and attention == _GROUP_ATTENTION_CHAOTIC:
        attention = _GROUP_ATTENTION_HIGH

    return_value = {"group_attention": attention}
    return return_value


def _has_latest_bot_turn_continuity(
    *,
    chat_history_wide: list[dict],
    platform_bot_id: str,
    current_global_user_id: str,
) -> bool:
    """Return whether the latest row is a bot turn aimed at this user."""

    if not chat_history_wide:
        return_value = False
        return return_value
    latest = chat_history_wide[-1]
    return_value = (
        latest.get("role") == "assistant"
        and latest.get("platform_user_id") == platform_bot_id
        and current_global_user_id in (
            latest.get("addressed_to_global_user_ids") or []
        )
    )
    return return_value


def _string_list(value: object, *, limit: int) -> list[str]:
    """Return a bounded list of text descriptors."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return_value: list[str] = []
        return return_value
    values = [
        _clip_text(item, 120)
        for item in value
        if isinstance(item, str) and item.strip()
    ]
    return_value = values[:limit]
    return return_value


def _project_fragment(item: Mapping[str, Any], *, body_limit: int) -> dict[str, Any]:
    """Project one assembled fragment without identifiers or timing."""

    body_text = item.get("body_text")
    if not isinstance(body_text, str):
        body_text = item.get("content", "")
    return_value = {
        "body_text": _clip_text(body_text, body_limit),
        "semantic_target_labels": _string_list(
            item.get("semantic_target_labels")
            or item.get("target_labels"),
            limit=4,
        ),
        "reply_target_label": _clip_text(
            item.get("reply_target_label"),
            80,
        ) or "none",
        "media_labels": _string_list(item.get("media_labels"), limit=4),
    }
    return return_value


def _project_fragments(state: SettledRelevanceState) -> list[dict[str, Any]]:
    """Project opening and newest assembled fragments under their cap."""

    fragments = state.get("assembled_fragments")
    if fragments is None:
        fragments = state.get("turn_fragments")
    if not isinstance(fragments, Sequence) or isinstance(fragments, (str, bytes)):
        user_input = state.get("user_input", "")
        fragments = [{"body_text": user_input}]

    selected = list(fragments[:2]) + list(fragments[2:])
    projected: list[dict[str, Any]] = []
    remaining = _FRAGMENT_TOTAL_CHARS
    for item in selected:
        if not isinstance(item, Mapping) or remaining <= 0:
            continue
        fragment = _project_fragment(
            item,
            body_limit=min(1200, max(200, remaining)),
        )
        rendered_length = len(json.dumps(fragment, ensure_ascii=False))
        if rendered_length > remaining:
            fragment = _project_fragment(item, body_limit=max(80, remaining // 2))
            rendered_length = len(json.dumps(fragment, ensure_ascii=False))
        projected.append(fragment)
        remaining -= rendered_length
    return_value = projected
    return return_value


def _project_media(state: SettledRelevanceState) -> dict[str, Any]:
    """Project at most four selected media descriptions and overflow."""

    media = state.get("media_descriptions")
    if media is None:
        media = state.get("user_multimedia_input")
    if not isinstance(media, Sequence) or isinstance(media, (str, bytes)):
        media = []

    descriptions: list[dict[str, str]] = []
    for item in list(media)[:4]:
        if not isinstance(item, Mapping):
            continue
        description = item.get("description", "")
        media_kind = item.get("media_kind") or item.get("content_type", "")
        if not isinstance(description, str) or not isinstance(media_kind, str):
            continue
        descriptions.append({
            "media_kind": _clip_text(media_kind, 40),
            "description": _clip_text(description, 600),
        })

    additional = bool(state.get("additional_media_present")) or len(media) > 4
    return_value = {
        "descriptions": descriptions,
        "additional_media_present": additional,
    }
    return return_value


def _project_history(state: SettledRelevanceState) -> list[dict[str, Any]]:
    """Project at most ten fresh history rows under their character cap."""

    history = state.get("fresh_history")
    if history is None:
        history = state.get("chat_history_recent")
    if history is None:
        history = state.get("chat_history_wide")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        return_value: list[dict[str, Any]] = []
        return return_value

    projected: list[dict[str, Any]] = []
    remaining = _HISTORY_TOTAL_CHARS
    for item in list(history)[-10:]:
        if not isinstance(item, Mapping) or remaining <= 0:
            continue
        body = item.get("body_text") or item.get("content", "")
        row = {
            "speaker": _clip_text(item.get("speaker") or item.get("role"), 40),
            "body_text": _clip_text(body, min(500, remaining)),
            "target_summary": _clip_text(item.get("target_summary"), 80),
            "reply_summary": _clip_text(item.get("reply_summary"), 80),
        }
        projected.append(row)
        remaining -= len(json.dumps(row, ensure_ascii=False))
    return_value = projected
    return return_value


def _project_context(state: SettledRelevanceState) -> dict[str, Any]:
    """Project bounded scene, relationship, and attention descriptors."""

    relationship_context = state.get("relationship_context", "")
    scene_context = state.get("scene_context", "")
    if not scene_context and state.get("channel_type") == "group":
        attention = build_group_attention_context(
            chat_history_wide=state.get("chat_history_wide") or [],
            platform_bot_id=state.get("platform_bot_id", ""),
            character_global_user_id=state.get(
                "character_global_user_id",
                CHARACTER_GLOBAL_USER_ID,
            ),
        )
        scene_context = attention.get("group_attention", "")

    user_profile = state.get("user_profile")
    if not isinstance(user_profile, Mapping):
        user_profile = {}
    affinity = user_profile.get("affinity")
    affinity_block = (
        build_affinity_block(affinity)
        if isinstance(affinity, int)
        else {"level": "", "instruction": ""}
    )
    return_value = {
        "scene_context": _clip_text(scene_context, 600),
        "relationship_context": _clip_text(relationship_context, 600),
        "mood": _clip_text(state.get("character_mood"), 200),
        "group_attention": _clip_text(state.get("group_attention"), 100),
        "bot_continuity": _clip_text(state.get("bot_continuity"), 200),
        "affinity_level": _clip_text(affinity_block.get("level"), 100),
        "engagement_guidelines": _string_list(
            state.get("engagement_guidelines"),
            limit=2,
        ),
    }
    context_json = json.dumps(return_value, ensure_ascii=False)
    if len(context_json) > _CONTEXT_TOTAL_CHARS:
        return_value["engagement_guidelines"] = []
        return_value["relationship_context"] = _clip_text(
            return_value["relationship_context"],
            300,
        )
    return_value = return_value
    return return_value


def _project_settled_state(
    state: SettledRelevanceState,
    observation_status: str,
) -> dict[str, Any]:
    """Build the complete bounded semantic settled payload."""

    return_value = {
        "observation_status": observation_status,
        "assembled_turn": {
            "fragments": _project_fragments(state),
            "media": _project_media(state),
        },
        "fresh_history": _project_history(state),
        "scene_and_relationship": _project_context(state),
    }
    return return_value


def build_settled_relevance_messages(
    state: SettledRelevanceState,
    observation_status: str = "observation_complete",
) -> tuple[SystemMessage, HumanMessage]:
    """Render bounded settled relevance messages."""

    if observation_status not in {"more_time_available", "observation_complete"}:
        raise ValueError("observation_status is invalid")
    payload = _project_settled_state(state, observation_status)
    human_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    available_human_chars = SETTLED_RELEVANCE_MAX_INPUT_CHARS - len(
        _SETTLED_SYSTEM_PROMPT
    )
    if len(human_content) > available_human_chars:
        payload["fresh_history"] = []
        payload["scene_and_relationship"]["engagement_guidelines"] = []
        human_content = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if len(human_content) > available_human_chars:
        fragments = payload["assembled_turn"]["fragments"]
        payload["assembled_turn"]["fragments"] = fragments[-2:]
        human_content = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if len(human_content) > available_human_chars:
        human_content = human_content[:available_human_chars]

    messages = (
        SystemMessage(content=_SETTLED_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    )
    return_value = messages
    return return_value


def validate_settled_relevance_decision(
    raw: Mapping[str, Any],
    *,
    observation_status: str,
) -> SettledRelevanceDecision:
    """Validate and bound settled relevance output."""

    if observation_status not in {"more_time_available", "observation_complete"}:
        raise ValueError("observation_status is invalid")
    if not isinstance(raw, Mapping):
        raise ValueError("settled relevance output must be an object")

    response_action = raw.get("response_action")
    if response_action not in {"ignore", "proceed", "wait"}:
        raise ValueError("settled response_action is invalid")
    if (
        observation_status == "observation_complete"
        and response_action == "wait"
    ):
        raise ValueError("settled complete phase cannot wait")

    reason = raw.get("reason_to_respond")
    use_reply_feature = raw.get("use_reply_feature")
    channel_topic = raw.get("channel_topic")
    indirect_context = raw.get("indirect_speech_context")
    if not isinstance(reason, str):
        raise ValueError("settled reason_to_respond must be a string")
    if not isinstance(use_reply_feature, bool):
        raise ValueError("settled use_reply_feature must be bool")
    if not isinstance(channel_topic, str):
        raise ValueError("settled channel_topic must be a string")
    if not isinstance(indirect_context, str):
        raise ValueError("settled indirect_speech_context must be a string")

    return_value: SettledRelevanceDecision = {
        "response_action": response_action,
        "reason_to_respond": _clip_text(reason, 180),
        "use_reply_feature": use_reply_feature,
        "channel_topic": _clip_text(channel_topic, 60),
        "indirect_speech_context": _clip_text(indirect_context, 100),
    }
    return return_value


def _ignore_decision(reason: str) -> SettledRelevanceDecision:
    """Return the structural fail-closed settled decision."""

    return_value: SettledRelevanceDecision = {
        "response_action": "ignore",
        "reason_to_respond": _clip_text(reason, 180),
        "use_reply_feature": False,
        "channel_topic": "",
        "indirect_speech_context": "",
    }
    return return_value


async def relevance_agent(state: IMProcessState) -> dict[str, Any]:
    """Run settled relevance and expose the downstream compatibility fields."""

    observation_status = state.get(
        "observation_status",
        "observation_complete",
    )
    messages = build_settled_relevance_messages(state, observation_status)
    started_at = time.perf_counter()
    response = await _relevance_agent_llm.ainvoke(
        list(messages),
        config=_relevance_agent_llm_config,
    )
    parsed_output = parse_llm_json_output(
        str(response.content),
        deterministic_only=True,
    )
    parse_status = "succeeded"
    try:
        decision = validate_settled_relevance_decision(
            parsed_output,
            observation_status=observation_status,
        )
    except ValueError as exc:
        logger.warning(f"Invalid settled relevance output: {exc}")
        decision = _ignore_decision("invalid settled relevance output")
        parse_status = "invalid"

    return_value: dict[str, Any] = {
        **decision,
        "should_respond": decision["response_action"] == "proceed",
        "user_input": state.get("user_input", ""),
    }
    if "turn_id" in state:
        return_value["turn_id"] = state["turn_id"]
    if "turn_version" in state:
        return_value["turn_version"] = state["turn_version"]

    await llm_tracing.record_llm_trace_step(
        trace_id=str(state.get("llm_trace_id", "")),
        stage_name="persona_relevance_agent",
        route_name="relevance",
        model_name=RELEVANCE_AGENT_LLM_MODEL,
        messages=list(messages),
        response_text=str(response.content),
        parsed_output=return_value,
        parse_status=parse_status,
        status="succeeded",
        duration_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
        output_state_fields=[
            "response_action",
            "should_respond",
            "reason_to_respond",
            "use_reply_feature",
            "channel_topic",
            "indirect_speech_context",
        ],
    )
    return return_value
