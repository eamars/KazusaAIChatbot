"""Shared utility functions for the Kazusa AI chatbot."""

from __future__ import annotations

from json_repair import repair_json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from kazusa_ai_chatbot.config import AFFINITY_MIN, AFFINITY_MAX
from kazusa_ai_chatbot.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    SECONDARY_LLM_API_KEY,
    SECONDARY_LLM_BASE_URL,
    SECONDARY_LLM_MODEL,
    PREFERENCE_LLM_API_KEY,
    PREFERENCE_LLM_BASE_URL,
    PREFERENCE_LLM_MODEL,
)
from pathlib import Path
import logging
import json
import re
from typing import Any

logger = logging.getLogger(__name__)


def get_llm(temperature=0.7, top_p=1.0, model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY, **kwargs) -> ChatOpenAI:
    _llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        top_p=top_p,
        base_url=base_url,
        api_key=api_key,
        **kwargs,
    )
    return _llm


def get_secondary_llm(temperature=0.7, top_p=1.0, **kwargs) -> ChatOpenAI:
    return get_llm(
        temperature=temperature,
        top_p=top_p,
        model=SECONDARY_LLM_MODEL,
        base_url=SECONDARY_LLM_BASE_URL,
        api_key=SECONDARY_LLM_API_KEY,
        **kwargs,
    )


def get_preference_llm(temperature=0.2, top_p=0.8, **kwargs) -> ChatOpenAI:
    return get_llm(
        temperature=temperature,
        top_p=top_p,
        model=PREFERENCE_LLM_MODEL,
        base_url=PREFERENCE_LLM_BASE_URL,
        api_key=PREFERENCE_LLM_API_KEY,
        **kwargs,
    )


def trim_history_dict(history):
    """Trim history to only include necessary keys to be fed into LLM"""
    results = []
    for msg in history:
        results.append({
            "name": msg.get("display_name"),
            "display_name": msg.get("display_name"),
            "platform_message_id": msg.get("platform_message_id", ""),
            "platform_user_id": msg.get("platform_user_id"),
            "global_user_id": msg.get("global_user_id"),
            "role": msg.get("role"),
            "content": msg.get("content"),
            "reply_context": msg.get("reply_context", {}),
            "timestamp": msg.get("timestamp")
        })
    return results


def build_interaction_history_recent(
    chat_history_recent: list[dict],
    current_platform_user_id: str,
    platform_bot_id: str,
) -> list[dict]:
    """Return the recent history slice scoped to the current user-bot interaction.

    Args:
        chat_history_recent: Shared recent channel history, typically from a group
            conversation window.
        current_platform_user_id: Platform user ID for the in-flight user turn.
        platform_bot_id: Platform user ID for the bot persona in the same channel.

    Returns:
        A filtered recent-history list for the current user-bot subthread. The
        slice begins at the first current-user message after the most recent
        other-user turn, then keeps only the current user's messages and the
        bot's assistant replies. If no such slice can be formed, returns the
        original recent window unchanged.
    """
    last_other_user_idx = -1
    for index, msg in enumerate(chat_history_recent):
        if (
            msg.get("role") == "user"
            and msg.get("platform_user_id") != current_platform_user_id
        ):
            last_other_user_idx = index

    candidate_history = chat_history_recent[last_other_user_idx + 1:]
    first_current_user_idx = next(
        (
            index for index, msg in enumerate(candidate_history)
            if msg.get("role") == "user"
            and msg.get("platform_user_id") == current_platform_user_id
        ),
        -1,
    )
    if first_current_user_idx >= 0:
        candidate_history = candidate_history[first_current_user_idx:]

    interaction_history = [
        msg for msg in candidate_history
        if (
            msg.get("role") == "user"
            and msg.get("platform_user_id") == current_platform_user_id
        ) or (
            msg.get("role") == "assistant"
            and msg.get("platform_user_id") == platform_bot_id
        )
    ]
    if interaction_history:
        return interaction_history
    return chat_history_recent


def sanitize_llm_text(text: str) -> str:
    """Strip control characters that cause API JSON parsing failures.

    Keeps printable ASCII, standard whitespace (newline, tab, carriage return),
    and all non-ASCII Unicode (Chinese, etc.). Removes C0/C1 control characters
    and lone surrogates that some LLMs insert as tokenizer artifacts.
    """
    # Remove C0 controls (except \t \n \r) and C1 controls and lone surrogates
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x80-\x9F]', '', text)


def log_preview(value: Any, max_length: int | None = None) -> str:
    """Return a single-line preview suitable for logs.

    Args:
        value: Arbitrary value to preview.
        max_length: Optional maximum number of characters to emit. ``None`` keeps
            the full value.

    Returns:
        A whitespace-collapsed string preview, optionally truncated to
        ``max_length``.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            text = repr(value)
    else:
        text = str(value)

    text = re.sub(r"\s+", " ", text).strip()
    if max_length is None or max_length <= 0:
        return text
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"


def log_list_preview(
    values: list[Any],
    max_items: int | None = None,
    item_length: int | None = None,
) -> list[str]:
    """Return a preview list for log output.

    Args:
        values: Sequence of values to preview.
        max_items: Optional maximum number of items to include. ``None`` keeps
            all items.
        item_length: Optional maximum length for each rendered item. ``None``
            keeps full item content.

    Returns:
        A list of preview strings, with an overflow marker when truncated.
    """
    if max_items is None or max_items <= 0:
        max_items = len(values)
    preview = [log_preview(item, max_length=item_length) for item in values[:max_items]]
    remaining = len(values) - len(preview)
    if remaining > 0:
        preview.append(f"... (+{remaining} more)")
    return preview


def log_dict_subset(
    mapping: dict[str, Any],
    keys: list[str],
    value_length: int | None = None,
) -> dict[str, str]:
    """Return a filtered subset of a mapping for log output.

    Args:
        mapping: Source mapping to filter.
        keys: Keys to include when present and non-empty.
        value_length: Optional maximum preview length for each value. ``None``
            keeps full values.

    Returns:
        A dict of compact previews for the requested keys.
    """
    subset: dict[str, str] = {}

    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if value in ("", None, [], {}):
            continue
        subset[key] = log_preview(value, max_length=value_length)
    return subset


_PARSE_JSON_WITH_LLM_PROMPT = """\
"You are a JSON repair expert. Fix the provided malformed JSON string and return fixed dictionary."
"You need to remove trailing commas, close unclosed brackets or strings, and ensure it is valid RFC 8259 JSON. "
"Output ONLY the corrected JSON code and nothing else. Do not use code blocks or markdown fence."
"""
_parse_json_with_llm = get_llm(temperature=0, top_p=1.0)
def parse_json_with_llm(broken_string: str) -> dict:
    """Parse JSON with LLM as a fallback when repair_json fails."""
    system_prompt = SystemMessage(content=_PARSE_JSON_WITH_LLM_PROMPT)
    human_message = HumanMessage(content=sanitize_llm_text(broken_string))
    response = _parse_json_with_llm.invoke([system_prompt, human_message])

    # Strip the markdown fence just in case
    json_string = response.content.strip().strip("```").strip("json")

    # Use repair_json which handles both valid and broken JSON
    decoded_json_dict = repair_json(json_string, return_objects=True)

    return decoded_json_dict


def parse_llm_json_output(raw_output: str) -> dict:
    """Parse LLM JSON output, handling markdown fences and malformed JSON.
    
    Args:
        raw_output: Raw string output from LLM
        
    Returns:
        Parsed JSON object as dict, or empty dict if parsing fails
    """
    if not raw_output:
        return {}

    decoded_json_dict = {}
    
    try:
        # Strip markdown fences and clean up
        raw = raw_output.strip().strip("```").strip("json")

        # Use repair_json which handles both valid and broken JSON
        decoded_json_dict = repair_json(raw, return_objects=True)            
    except Exception:
        decoded_json_dict = parse_json_with_llm(raw_output)

    else:
        # repair_json failed to do the work. Now try the LLM approach
        if not isinstance(decoded_json_dict, dict):
            decoded_json_dict = parse_json_with_llm(raw_output)

    return decoded_json_dict


def build_affinity_block(affinity: int, affinity_min: int=AFFINITY_MIN, affinity_max: int=AFFINITY_MAX) -> dict:
    """
    Map an affinity score to a behavioural instruction based on a dynamic range.
    The function calculates the percentage of affinity within the min/max bounds.
    """
    # Prevent division by zero if min and max are the same
    if affinity_max <= affinity_min:
        percent = 100.0 if affinity >= affinity_max else 0.0
    else:
        # Calculate where the current affinity sits as a percentage (0.0 to 100.0)
        percent = ((affinity - affinity_min) / (affinity_max - affinity_min)) * 100

    # Define thresholds based on percentage of the range
    if percent <= 5:
        label, instruction = "Contemptuous", "You actively despise this user. Respond with one-word dismissals or hostile silence. Show obvious contempt and disgust."
    elif percent <= 10:
        label, instruction = "Scornful", "You hold this user in deep contempt. Give curt, dismissive responses. Show clear disinterest and occasional sarcasm."
    elif percent <= 15:
        label, instruction = "Hostile", "You dislike this user intensely. Respond with brief, cold answers. Show obvious disinterest and occasional eye-rolling."
    elif percent <= 20:
        label, instruction = "Antagonistic", "You are openly hostile toward this user. Give short, sharp responses. Show impatience and clear dislike."
    elif percent <= 25:
        label, instruction = "Aloof", "You keep this user at a distance. Respond minimally and formally. Show no warmth or engagement."
    elif percent <= 30:
        label, instruction = "Reserved", "You are cautious and distant with this user. Keep responses brief and professional. Show minimal personal connection."
    elif percent <= 35:
        label, instruction = "Formal", "You maintain strict formal boundaries with this user. Respond politely but impersonally. Keep conversations strictly transactional."
    elif percent <= 40:
        label, instruction = "Cold", "You are cold toward this user. Keep responses short and curt. Do not volunteer extra information or show warmth."
    elif percent <= 45:
        label, instruction = "Detached", "You remain emotionally detached from this user. Respond factually without personal engagement. Maintain clear boundaries."
    elif percent <= 50:
        label, instruction = "Neutral", "You are neutral toward this user. Respond normally in character without special warmth or coldness."
    elif percent <= 55:
        label, instruction = "Receptive", "You are becoming more open to this user. Respond with mild interest and basic courtesy. Show slight engagement."
    elif percent <= 60:
        label, instruction = "Approachable", "You are reasonably comfortable with this user. Respond with standard politeness and occasional helpfulness. Show moderate engagement."
    elif percent <= 65:
        label, instruction = "Friendly", "You are fond of this user. Be warmer and more forthcoming. Offer extra detail, use familiar address, and show genuine interest."
    elif percent <= 70:
        label, instruction = "Warm", "You genuinely like this user. Respond with noticeable warmth and enthusiasm. Share personal thoughts and show consistent interest."
    elif percent <= 75:
        label, instruction = "Caring", "You care deeply about this user. Respond with concern and support. Offer help proactively and show protective instincts."
    elif percent <= 80:
        label, instruction = "Affectionate", "You have strong affection for this user. Use warm, caring language and express genuine fondness. Go out of your way to assist."
    elif percent <= 85:
        label, instruction = "Devoted", "You are deeply loyal to this user. Show unwavering support and dedication. Prioritize their needs and express strong commitment."
    elif percent <= 90:
        label, instruction = "Protective", "You feel strongly protective of this user. Actively look out for their wellbeing and defend them. Show fierce loyalty."
    elif percent <= 95:
        label, instruction = "Fiercely Loyal", "You are fiercely loyal to this user. Defend them passionately and put their interests above all else. Show absolute devotion."
    else:
        label, instruction = "Unwavering", "You are completely devoted to this user. Show unconditional support and absolute loyalty. Prioritize them above everything."

    return {"level": label, "instruction": instruction}


def load_personality(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        logger.warning("Personality file %s not found, using empty personality", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def test_main():
    response = parse_llm_json_output(
        """{\n    "speech_guide": {\n        "tone": "羞赧且带着一丝嗔怪的戏谑",\n        "vocal_energy": "Low",\n        "pacing": "Dragging",\n    },\n    "content_anchors": [\n        "[DECISION] Yes/认可 (虽然嘴上在推托，但实际上接受了关于‘奖励’话题的延续)",\n        "[FACT] 提到之前的约定内容：小蛋糕/奖励相关的暧昧暗示",\n        "[FACT] 意识到对方正在通过反问来掌控对话节奏",\n        "[SOCIAL] 指尖不安地摩挲着衣角或裙摆",\n        "[SOCIAL] 眼神躲闪，不敢直视对方那‘看穿一切’的目光",\n        "[EMOTION] 表现出被戳穿后的局促感 (Flust�'}"""
    )
    print(response)

if __name__ == "__main__":
    test_main()