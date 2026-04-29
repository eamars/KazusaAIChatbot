"""LLM recorder for short-term conversation progress."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressRecordInput
from kazusa_ai_chatbot.conversation_progress.policy import (
    VALID_CONTINUITY,
    VALID_CONVERSATION_MODE,
    VALID_EPISODE_PHASE,
    VALID_STATUS,
    VALID_TOPIC_MOMENTUM,
)
from kazusa_ai_chatbot.db.schemas import ConversationEpisodeStateDoc
from kazusa_ai_chatbot.utils import get_llm, log_preview, parse_llm_json_output

logger = logging.getLogger(__name__)

_RECORDER_PROMPT = """\
You are the short-term conversation progress recorder for Kazusa.

Your job is to update compact operational state for the next responsive turn.
Do not write diary prose, relationship insight, medical advice, or long-term memory.
Do not copy full assistant turns. Use short semantic labels.
Do not generate Kazusa's next reply text.

You decide semantic continuity:
- same_episode: the user is continuing the same unresolved thread.
- related_shift: the topic shifted but prior progress may still help.
- sharp_transition: the user started a clearly new topic; stale obligations should not guide the next turn.

When an existing user_state_updates or open_loops item still applies, copy its text exactly
from prior_episode_state. That exact copy tells deterministic code to preserve first_seen_at.
When an item no longer applies, omit it.
The prior_episode_state entry lists contain plain text strings only. Do not return objects
with text/first_seen_at fields inside any output list.

For assistant_moves, emit compact free-form speech-act labels. Reuse a prior label exactly
when the current assistant response repeated the same move. You own this semantic judgment.

Also track flow state:
- conversation_mode: task_support | emotional_support | casual_chat | playful_banter | meta_discussion | group_ambient | mixed
- episode_phase: opening | developing | deepening | pivoting | stuck_loop | resolving | cooling_down
- topic_momentum: stable | drifting | quick_pivot | fragmented | sharp_break

Use current_thread for what is being discussed even when there is no task.
Use user_goal and current_blocker only when the episode is goal-driven.
Use next_affordances for natural next conversational moves, not exact dialog text.
Use resolved_threads and avoid_reopening for items that should not be dragged back unless the user reopens them.

Return strict JSON only:
{
  "continuity": "same_episode | related_shift | sharp_transition",
  "status": "active | suspended | closed",
  "episode_label": "short semantic label",
  "conversation_mode": "task_support | emotional_support | casual_chat | playful_banter | meta_discussion | group_ambient | mixed",
  "episode_phase": "opening | developing | deepening | pivoting | stuck_loop | resolving | cooling_down",
  "topic_momentum": "stable | drifting | quick_pivot | fragmented | sharp_break",
  "current_thread": "one-line neutral current thread",
  "user_goal": "optional goal, or empty string",
  "current_blocker": "optional blocker, or empty string",
  "user_state_updates": ["compact user-state observation"],
  "assistant_moves": ["compact assistant speech-act label"],
  "overused_moves": ["assistant move label that has occurred too often"],
  "open_loops": ["unresolved thread"],
  "resolved_threads": ["handled thread"],
  "avoid_reopening": ["stale item not to reopen"],
  "emotional_trajectory": "one-line emotional movement",
  "next_affordances": ["natural next move available to Kazusa"],
  "progression_guidance": "one short instruction for the next turn"
}
"""

_recorder_llm = get_llm(
    temperature=0.2,
    top_p=0.75,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)


def render_recorder_prompt() -> str:
    """Return the recorder system prompt for render checks.

    Returns:
        Recorder prompt text.
    """

    return _RECORDER_PROMPT


_ENTRY_LIST_FIELDS = (
    "user_state_updates",
    "open_loops",
    "resolved_threads",
    "avoid_reopening",
)

_STRING_LIST_FIELDS = (
    "assistant_moves",
    "overused_moves",
    "next_affordances",
)

_RECORDER_PRIOR_SCALAR_FIELDS = (
    "status",
    "episode_label",
    "continuity",
    "conversation_mode",
    "episode_phase",
    "topic_momentum",
    "current_thread",
    "user_goal",
    "current_blocker",
    "emotional_trajectory",
    "progression_guidance",
    "turn_count",
    "last_user_input",
)


def _require_string(value: Any, field_name: str, *, default: str = "") -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return_value = value.strip()
    return return_value


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} items must be strings")
        text = item.strip()
        if not text:
            continue
        result.append(text)
    return result


def _validated_label(value: Any, field_name: str, allowed_values: set[str]) -> str:
    label = _require_string(value, field_name)
    if not label:
        return ""
    if label not in allowed_values:
        raise ValueError(f"invalid {field_name}: {label}")
    return label


def _prior_entry_texts(prior_episode_state: ConversationEpisodeStateDoc, field_name: str) -> list[str]:
    values = prior_episode_state.get(field_name, [])
    if not isinstance(values, list):
        return_value = []
        return return_value
    result: list[str] = []
    for entry in values:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        if isinstance(text, str) and text.strip():
            result.append(text.strip())
    return result


def _prior_string_list(prior_episode_state: ConversationEpisodeStateDoc, field_name: str) -> list[str]:
    values = prior_episode_state.get(field_name, [])
    if not isinstance(values, list):
        return_value = []
        return return_value
    return_value = [item.strip() for item in values if isinstance(item, str) and item.strip()]
    return return_value


def build_recorder_prior_state(
    prior_episode_state: ConversationEpisodeStateDoc | None,
) -> dict | None:
    """Build recorder-facing prior state with text-only copyable lists.

    Args:
        prior_episode_state: Stored episode state from the previous turn.

    Returns:
        Native prior-state object for the recorder LLM, or ``None``.
    """

    if prior_episode_state is None:
        return None

    result: dict = {}
    for field_name in _RECORDER_PRIOR_SCALAR_FIELDS:
        if field_name in prior_episode_state:
            result[field_name] = prior_episode_state[field_name]
    for field_name in _ENTRY_LIST_FIELDS:
        result[field_name] = _prior_entry_texts(prior_episode_state, field_name)
    for field_name in _STRING_LIST_FIELDS:
        result[field_name] = _prior_string_list(prior_episode_state, field_name)
    return result


def validate_recorder_output(payload: dict) -> dict:
    """Validate and normalize recorder JSON.

    Args:
        payload: Parsed LLM JSON object.

    Returns:
        Normalized recorder output.

    Raises:
        ValueError: If the payload violates the recorder contract.
    """

    continuity = _require_string(payload.get("continuity", ""), "continuity")
    status = _require_string(payload.get("status", ""), "status")
    if continuity not in VALID_CONTINUITY:
        raise ValueError(f"invalid continuity: {continuity}")
    if status not in VALID_STATUS:
        raise ValueError(f"invalid status: {status}")
    return_value = {
        "continuity": continuity,
        "status": status,
        "episode_label": _require_string(payload.get("episode_label", ""), "episode_label"),
        "conversation_mode": _validated_label(
            payload.get("conversation_mode", ""),
            "conversation_mode",
            VALID_CONVERSATION_MODE,
        ),
        "episode_phase": _validated_label(
            payload.get("episode_phase", ""),
            "episode_phase",
            VALID_EPISODE_PHASE,
        ),
        "topic_momentum": _validated_label(
            payload.get("topic_momentum", ""),
            "topic_momentum",
            VALID_TOPIC_MOMENTUM,
        ),
        "current_thread": _require_string(payload.get("current_thread", ""), "current_thread"),
        "user_goal": _require_string(payload.get("user_goal", ""), "user_goal"),
        "current_blocker": _require_string(payload.get("current_blocker", ""), "current_blocker"),
        "user_state_updates": _string_list(payload.get("user_state_updates", []), "user_state_updates"),
        "assistant_moves": _string_list(payload.get("assistant_moves", []), "assistant_moves"),
        "overused_moves": _string_list(payload.get("overused_moves", []), "overused_moves"),
        "open_loops": _string_list(payload.get("open_loops", []), "open_loops"),
        "resolved_threads": _string_list(payload.get("resolved_threads", []), "resolved_threads"),
        "avoid_reopening": _string_list(payload.get("avoid_reopening", []), "avoid_reopening"),
        "emotional_trajectory": _require_string(
            payload.get("emotional_trajectory", ""),
            "emotional_trajectory",
        ),
        "next_affordances": _string_list(payload.get("next_affordances", []), "next_affordances"),
        "progression_guidance": _require_string(
            payload.get("progression_guidance", ""),
            "progression_guidance",
        ),
    }
    return return_value


async def record_with_llm(record_input: ConversationProgressRecordInput) -> dict:
    """Call the recorder LLM for one completed turn.

    Args:
        record_input: Current turn and prior episode-state payload.

    Returns:
        Validated recorder output.
    """

    human_payload = {
        "prior_episode_state": build_recorder_prior_state(record_input["prior_episode_state"]),
        "decontexualized_input": record_input["decontexualized_input"],
        "chat_history_recent": record_input["chat_history_recent"],
        "content_anchors": record_input["content_anchors"],
        "logical_stance": record_input["logical_stance"],
        "character_intent": record_input["character_intent"],
        "final_dialog": record_input["final_dialog"],
    }
    logger.info(f'Conversation progress recorder input: platform={record_input["scope"].platform} channel={record_input["scope"].platform_channel_id or "<dm>"} user={record_input["scope"].global_user_id} payload={log_preview(human_payload)}')
    response = await _recorder_llm.ainvoke([
        SystemMessage(content=_RECORDER_PROMPT),
        HumanMessage(content=json.dumps(human_payload, ensure_ascii=False)),
    ])
    parsed = parse_llm_json_output(response.content)
    validated = validate_recorder_output(parsed)
    logger.info(f'Conversation progress recorder parsed: platform={record_input["scope"].platform} channel={record_input["scope"].platform_channel_id or "<dm>"} user={record_input["scope"].global_user_id} raw={log_preview(response.content)} validated={log_preview(validated)}')
    return validated
