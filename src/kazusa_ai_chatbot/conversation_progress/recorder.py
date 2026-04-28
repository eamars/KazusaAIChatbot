"""LLM recorder for short-term conversation progress."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressRecordInput
from kazusa_ai_chatbot.conversation_progress.policy import VALID_CONTINUITY, VALID_STATUS
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

_RECORDER_PROMPT = """\
You are the short-term conversation progress recorder for Kazusa.

Your job is to update compact operational state for the next responsive turn.
Do not write diary prose, relationship insight, medical advice, or long-term memory.
Do not copy full assistant turns. Use short semantic labels.

You decide semantic continuity:
- same_episode: the user is continuing the same unresolved thread.
- related_shift: the topic shifted but prior progress may still help.
- sharp_transition: the user started a clearly new topic; stale obligations should not guide the next turn.

When an existing user_state_updates or open_loops item still applies, copy its text exactly
from prior_episode_state. That exact copy tells deterministic code to preserve first_seen_at.
When an item no longer applies, omit it.

For assistant_moves, emit compact free-form speech-act labels. Reuse a prior label exactly
when the current assistant response repeated the same move. You own this semantic judgment.

Return strict JSON only:
{
  "continuity": "same_episode | related_shift | sharp_transition",
  "status": "active | suspended | closed",
  "episode_label": "short semantic label",
  "user_state_updates": ["compact user-state observation"],
  "assistant_moves": ["compact assistant speech-act label"],
  "overused_moves": ["assistant move label that has occurred too often"],
  "open_loops": ["unresolved thread"],
  "progression_guidance": "one short instruction for the next turn"
}
"""

_recorder_llm = get_llm(temperature=0.2, top_p=0.75)


def render_recorder_prompt() -> str:
    """Return the recorder system prompt for render checks.

    Returns:
        Recorder prompt text.
    """

    return _RECORDER_PROMPT


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return [str(item).strip() for item in value if str(item).strip()]


def validate_recorder_output(payload: dict) -> dict:
    """Validate and normalize recorder JSON.

    Args:
        payload: Parsed LLM JSON object.

    Returns:
        Normalized recorder output.

    Raises:
        ValueError: If the payload violates the recorder contract.
    """

    continuity = str(payload.get("continuity", "")).strip()
    status = str(payload.get("status", "")).strip()
    if continuity not in VALID_CONTINUITY:
        raise ValueError(f"invalid continuity: {continuity}")
    if status not in VALID_STATUS:
        raise ValueError(f"invalid status: {status}")
    return {
        "continuity": continuity,
        "status": status,
        "episode_label": str(payload.get("episode_label", "")).strip(),
        "user_state_updates": _string_list(payload.get("user_state_updates", []), "user_state_updates"),
        "assistant_moves": _string_list(payload.get("assistant_moves", []), "assistant_moves"),
        "overused_moves": _string_list(payload.get("overused_moves", []), "overused_moves"),
        "open_loops": _string_list(payload.get("open_loops", []), "open_loops"),
        "progression_guidance": str(payload.get("progression_guidance", "")).strip(),
    }


async def record_with_llm(record_input: ConversationProgressRecordInput) -> dict:
    """Call the recorder LLM for one completed turn.

    Args:
        record_input: Current turn and prior episode-state payload.

    Returns:
        Validated recorder output.
    """

    human_payload = {
        "prior_episode_state": record_input["prior_episode_state"],
        "decontexualized_input": record_input["decontexualized_input"],
        "chat_history_recent": record_input["chat_history_recent"],
        "content_anchors": record_input["content_anchors"],
        "logical_stance": record_input["logical_stance"],
        "character_intent": record_input["character_intent"],
        "final_dialog": record_input["final_dialog"],
    }
    response = await _recorder_llm.ainvoke([
        SystemMessage(content=_RECORDER_PROMPT),
        HumanMessage(content=json.dumps(human_payload, ensure_ascii=False)),
    ])
    parsed = parse_llm_json_output(response.content)
    return validate_recorder_output(parsed)
