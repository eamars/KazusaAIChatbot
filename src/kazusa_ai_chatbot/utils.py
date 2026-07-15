"""Shared utility functions for the Kazusa AI chatbot."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from json_repair import repair_json
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    AFFINITY_MAX,
    AFFINITY_MIN,
    JSON_REPAIR_LLM_API_KEY,
    JSON_REPAIR_LLM_BASE_URL,
    JSON_REPAIR_LLM_MODEL,
    JSON_REPAIR_LLM_MAX_COMPLETION_TOKENS,
    JSON_REPAIR_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.message_envelope import (
    MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS,
)

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
logger = logging.getLogger(__name__)

_IMAGE_DESCRIPTION_ELLIPSIS = "..."


def _is_image_attachment(attachment: dict) -> bool:
    """Return whether a stored attachment is an image summary source."""

    media_kind = attachment.get("media_kind")
    if media_kind == "image":
        return True

    media_type = attachment.get("media_type")
    if not isinstance(media_type, str):
        return False
    return_value = media_type.startswith("image/")
    return return_value


def _escape_image_description(description: str) -> str:
    """Escape literal image-boundary characters for prompt text."""

    escaped = description.replace("&", "&amp;")
    escaped = escaped.replace("<", "&lt;")
    escaped = escaped.replace(">", "&gt;")
    return_value = escaped
    return return_value


def _trim_image_description(description: str) -> str:
    """Bound a prompt-facing image description with the shared cap."""

    if len(description) <= MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS:
        return description

    body_limit = (
        MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS
        - len(_IMAGE_DESCRIPTION_ELLIPSIS)
    )
    trimmed = description[:body_limit].rstrip()
    return_value = f"{trimmed}{_IMAGE_DESCRIPTION_ELLIPSIS}"
    return return_value


def _render_image_block(description: str) -> str:
    """Render one escaped image description with exact image boundaries."""

    escaped_description = _escape_image_description(description.strip())
    trimmed_description = _trim_image_description(escaped_description)
    return_value = f"<image>{trimmed_description}</image>"
    return return_value


def _image_blocks_from_attachments(attachments: object) -> list[str]:
    """Extract prompt-facing image blocks from stored attachment summaries."""

    image_blocks: list[str] = []
    if not isinstance(attachments, list):
        return image_blocks

    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        if not _is_image_attachment(attachment):
            continue

        description = attachment.get("description")
        if not isinstance(description, str) or not description.strip():
            continue

        image_block = _render_image_block(description)
        image_blocks.append(image_block)
    return image_blocks


def _append_image_blocks(text: str, image_blocks: list[str]) -> str:
    """Append image blocks after any remaining human-authored text."""

    if not image_blocks:
        return text

    parts: list[str] = []
    if text:
        parts.append(text)
    parts.extend(image_blocks)
    projected_text = "\n".join(parts)
    return projected_text


def project_text_with_image_blocks(text: str, attachments: object) -> str:
    """Project typed image descriptions into prompt-facing text.

    Args:
        text: Already-sanitized authored text for a prompt row.
        attachments: Stored attachment summaries for the same row.

    Returns:
        Authored text with escaped image-description blocks appended, or only
        image-description blocks when no authored text is present.
    """

    image_blocks = _image_blocks_from_attachments(attachments)
    if not image_blocks:
        return text

    projected_text = _append_image_blocks(text, image_blocks)
    return projected_text


def trim_history_dict(history: list[dict]) -> list[dict]:
    """Project conversation-history rows to prompt-facing metadata.

    Args:
        history: Conversation-history documents in chronological order.

    Returns:
        A list of compact dictionaries that preserve clean `body_text` and
        typed-addressing metadata needed by downstream consumers.
    """
    results = []
    for msg in history:
        body_text = project_text_with_image_blocks(
            msg["body_text"],
            msg.get("attachments"),
        )
        raw_reply_context_value = msg.get("reply_context") or {}
        if isinstance(raw_reply_context_value, dict):
            raw_reply_context = raw_reply_context_value
        else:
            raw_reply_context = {}

        reply_context = {}
        for key in (
            "reply_to_message_id",
            "reply_to_platform_user_id",
            "reply_to_display_name",
        ):
            value = raw_reply_context.get(key)
            if value not in ("", None):
                reply_context[key] = value

        reply_excerpt = raw_reply_context.get("reply_excerpt")
        if not isinstance(reply_excerpt, str):
            reply_excerpt = ""
        reply_excerpt = project_text_with_image_blocks(
            reply_excerpt,
            raw_reply_context.get("reply_attachments"),
        )
        if reply_excerpt:
            reply_context["reply_excerpt"] = reply_excerpt

        trimmed_msg = {
            "name": msg.get("display_name"),
            "display_name": msg.get("display_name"),
            "platform_message_id": msg.get("platform_message_id", ""),
            "platform_user_id": msg.get("platform_user_id"),
            "global_user_id": msg.get("global_user_id"),
            "role": msg.get("role"),
            "body_text": body_text,
            "addressed_to_global_user_ids": msg["addressed_to_global_user_ids"],
            "mentions": msg["mentions"],
            "broadcast": bool(msg["broadcast"]),
            "reply_context": reply_context,
            "timestamp": msg.get("timestamp"),
        }
        results.append(trimmed_msg)
    return results


def _is_current_user_row(
    row: dict,
    current_platform_user_id: str,
    current_global_user_id: str,
) -> bool:
    """Return whether a typed history row was authored by the current user."""

    if row["role"] != "user":
        return False
    if current_global_user_id and row["global_user_id"] == current_global_user_id:
        return_value = True
        return return_value
    return_value = row["platform_user_id"] == current_platform_user_id
    return return_value


def _is_visible_assistant_row(
    row: dict,
    platform_bot_id: str,
    current_global_user_id: str,
) -> bool:
    """Return whether a typed assistant row is visible to the current user."""

    if row["role"] != "assistant":
        return False
    if row["platform_user_id"] != platform_bot_id:
        return False
    if row["broadcast"] is True:
        return True
    addressed = row["addressed_to_global_user_ids"]
    return_value = current_global_user_id in addressed
    return return_value


def build_interaction_history_recent(
    chat_history_recent: list[dict],
    current_platform_user_id: str,
    platform_bot_id: str,
    current_global_user_id: str = "",
) -> list[dict]:
    """Return the recent history slice scoped to the current user's interaction.

    Args:
        chat_history_recent: Shared recent channel history, typically from a group
            conversation window.
        current_platform_user_id: Platform user ID for the in-flight user turn.
        platform_bot_id: Platform user ID for the bot persona in the same channel.
        current_global_user_id: Internal UUID for the in-flight user.

    Returns:
        A filtered recent-history list for the current user's subthread. User
        rows are selected by author UUID; assistant rows are selected by
        addressee UUID or explicit assistant broadcast.
    """
    if not current_global_user_id:
        return_value: list[dict] = []
        return return_value

    interaction_history: list[dict] = []
    has_current_user_row = False
    for msg in chat_history_recent:
        if _is_current_user_row(msg, current_platform_user_id, current_global_user_id):
            interaction_history.append(msg)
            has_current_user_row = True
            continue
        if _is_visible_assistant_row(msg, platform_bot_id, current_global_user_id):
            interaction_history.append(msg)

    if not has_current_user_row:
        return_value: list[dict] = []
        return return_value

    return_value = interaction_history
    return return_value


def sanitize_llm_text(text: str) -> str:
    """Strip control characters that cause API JSON parsing failures.

    Keeps printable ASCII, standard whitespace (newline, tab, carriage return),
    and all non-ASCII Unicode (Chinese, etc.). Removes C0/C1 control characters
    and lone surrogates that some LLMs insert as tokenizer artifacts.
    """
    # Remove C0 controls (except \t \n \r) and C1 controls and lone surrogates
    return_value = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x80-\x9F]', '', text)
    return return_value


def text_or_empty(value: object) -> str:
    """Return stripped text only when the value is already a string.

    Args:
        value: Candidate payload from a parsed workflow boundary.

    Returns:
        The stripped string, or an empty string for any non-string value.
    """
    if not isinstance(value, str):
        return ""
    return_value = value.strip()
    return return_value


def log_preview(value: Any) -> str:
    """Return a complete JSON-safe rendering suitable for logs.

    Args:
        value: Arbitrary value to render.

    Returns:
        The full value rendered without truncation. Strings are JSON-encoded so
        newlines and control characters stay visible and machine-parseable.
    """
    if value is None:
        return "null"

    try:
        return_value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return return_value
    except TypeError as exc:
        logger.debug(f"Falling back to repr for unserializable log preview: {exc}")
        return_value = repr(value)
        return return_value


def log_list_preview(
    values: list[Any],
) -> str:
    """Return a complete list rendering for log output.

    Args:
        values: Sequence of values to render.

    Returns:
        The full list as JSON text without synthetic overflow markers.
    """
    return_value = log_preview(values)
    return return_value


def log_dict_subset(
    mapping: dict[str, Any],
    keys: list[str],
) -> str:
    """Return a requested-key mapping for structured log output.

    Args:
        mapping: Source mapping to filter.
        keys: Keys to include. Missing and empty values are recorded explicitly
            so the log does not hide their state.

    Returns:
        JSON text describing each requested key without truncating values.
    """
    subset: dict[str, Any] = {}

    for key in keys:
        if key not in mapping:
            subset[key] = {"present": False, "value": None}
            continue
        value = mapping[key]
        subset[key] = {
            "present": True,
            "empty": value in ("", None, [], {}),
            "value": value,
        }
    return_value = log_preview(subset)
    return return_value


_PARSE_JSON_WITH_LLM_PROMPT = """\
You are a JSON repair expert. Fix the provided malformed JSON string and return one valid JSON object.

# Generation Procedure
1. Read `broken_json` as malformed JSON-like text.
2. Assume deterministic repair already handled ordinary syntax fixes; this call is for hard residual cases.
3. Use the expected output format only as a shape contract, not as data.
4. If an expected output format shows one top-level array field and `broken_json` is a top-level array, wrap the raw array under that field.
5. For example, with `{"tool_calls": [...]}` as the target shape, `[]` repairs to `{"tool_calls": []}`.
6. Repair syntax, wrapper shape, or damaged object structure only when the raw values support that repair.
7. Preserve actual keys and values from the raw output as much as possible.
8. Never copy placeholder values from an expected output format as real data.
9. If the raw output cannot be reconciled with the target object shape, return `{}`.
10. Return only the corrected JSON object text.

# Input Format
{
    "broken_json": "malformed JSON text"
}

# Output Format
Return only valid RFC 8259 JSON object text. Do not use code blocks, markdown fences, explanations, or surrounding prose.
"""
_PARSE_JSON_WITH_LLM_EXPECTED_FORMAT_PROMPT = (
    _PARSE_JSON_WITH_LLM_PROMPT
    + "\nExpected output format from the original prompt:\n"
)
_llm_interface = LLInterface()
_parse_json_with_llm = LLInterface()
_parse_json_with_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="JSON_REPAIR_LLM",
    base_url=JSON_REPAIR_LLM_BASE_URL,
    api_key=JSON_REPAIR_LLM_API_KEY,
    model=JSON_REPAIR_LLM_MODEL,
    temperature=0,
    top_p=1.0,
    top_k=None,
    max_completion_tokens=JSON_REPAIR_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=JSON_REPAIR_LLM_THINKING_ENABLED,
    ),
)


def _validate_expected_output_format(expected_output_format: str | None) -> None:
    """Reject unsupported expected-format payloads at the parser boundary."""

    if expected_output_format is None:
        return
    if not isinstance(expected_output_format, str):
        raise TypeError("expected_output_format must be a string or None")


def _build_json_repair_prompt(expected_output_format: str | None) -> str:
    """Build the JSON-repair prompt variant for the current parse request."""

    if expected_output_format is None:
        return_value = _PARSE_JSON_WITH_LLM_PROMPT
        return return_value

    return_value = (
        _PARSE_JSON_WITH_LLM_EXPECTED_FORMAT_PROMPT
        + expected_output_format
        + "\n\nOnly the broken JSON text is sent as the repair call's input message.\n"
    )
    return return_value


def parse_json_with_llm(
    broken_string: str,
    *,
    expected_output_format: str | None = None,
) -> dict:
    """Repair malformed JSON text by asking the configured JSON-repair LLM.

    Args:
        broken_string: Raw malformed JSON-like text returned by an LLM.
        expected_output_format: Optional target output contract shown to the
            original LLM.

    Returns:
        Parsed JSON object from the repaired response, or ``{}`` if repair does
        not produce an object.
    """

    _validate_expected_output_format(expected_output_format)

    system_prompt = SystemMessage(
        content=_build_json_repair_prompt(expected_output_format)
    )
    human_message = HumanMessage(
        content=json.dumps(
            {
                "broken_json": sanitize_llm_text(broken_string),
            },
            ensure_ascii=False,
        )
    )
    response = _parse_json_with_llm.invoke([system_prompt, human_message], config=_parse_json_with_llm_config)

    # Strip the markdown fence just in case
    json_string = response.content.strip().strip("```").strip("json")

    # Use repair_json which handles both valid and broken JSON
    try:
        decoded_json_dict = repair_json(json_string, return_objects=True)
    except Exception as exc:
        logger.exception(f"LLM JSON repair response could not be parsed: {exc}")
        decoded_json_dict = {}

    if not isinstance(decoded_json_dict, dict):
        logger.error(
            f"LLM JSON repair returned non-object output: {decoded_json_dict}"
        )
        decoded_json_dict = {}

    return_value = decoded_json_dict
    return return_value


def parse_llm_json_output(
    raw_output: str,
    *,
    expected_output_format: str | None = None,
    deterministic_only: bool = False,
) -> dict:
    """Parse LLM JSON output, handling markdown fences and malformed JSON.
    
    Args:
        raw_output: Raw string output from LLM
        expected_output_format: Optional target output contract shown to the
            original LLM, used only by the LLM repair fallback.
        deterministic_only: Whether malformed output must fail closed without
            calling the JSON-repair LLM.
        
    Returns:
        Parsed JSON object as dict, or empty dict if parsing fails
    """
    _validate_expected_output_format(expected_output_format)

    if not raw_output:
        return_value = {}
        return return_value

    decoded_json_dict = {}

    try:
        # Strip markdown fences and clean up
        raw = raw_output.strip().strip("```").strip("json")

        # Use repair_json which handles both valid and broken JSON
        decoded_json_dict = repair_json(raw, return_objects=True)
    except Exception as exc:
        if deterministic_only:
            logger.warning(f"Deterministic JSON parsing failed: {exc}")
            return_value = {}
            return return_value

        logger.exception(
            f"repair_json failed; falling back to LLM JSON repair: {exc}"
        )
        try:
            decoded_json_dict = parse_json_with_llm(
                raw_output,
                expected_output_format=expected_output_format,
            )
        except Exception as repair_exc:
            logger.exception(f"LLM JSON repair failed: {repair_exc}")
            decoded_json_dict = {}

    else:
        # repair_json failed to do the work. Now try the LLM approach
        if not isinstance(decoded_json_dict, dict):
            if deterministic_only:
                return_value = {}
                return return_value

            try:
                decoded_json_dict = parse_json_with_llm(
                    raw_output,
                    expected_output_format=expected_output_format,
                )
            except Exception as exc:
                logger.exception(f"LLM JSON repair failed: {exc}")
                decoded_json_dict = {}

    if not isinstance(decoded_json_dict, dict):
        logger.error(
            f"Unable to parse LLM output {raw_output}. "
            f"Last attempt: {decoded_json_dict}"
        )
        decoded_json_dict = {}

    return_value = decoded_json_dict
    return return_value


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

    return_value = {"level": label, "instruction": instruction}
    return return_value


def load_personality(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        logger.warning(f'Personality file {path} not found, using empty personality')
        return_value = {}
        return return_value
    with open(path, "r", encoding="utf-8") as f:
        return_value = json.load(f)
        return return_value



def test_main():
    response = parse_llm_json_output(
        '''{
    "speech_guide": {
        "tone": "羞赧且带着一丝嗔怪的戏谑",
        "vocal_energy": "Low",
        "pacing": "Dragging"
    },
    "content_plan": {
        "visible_goal": "认可奖励话题的延续。",
        "semantic_content": "嘴上推托，但实际接受了小蛋糕/奖励相关的暧昧暗示；意识到对方正在通过反问掌控对话节奏。",
        "voice": "羞赧且带一点嗔怪的戏谑。",
        "rendering": "1 条普通文字消息；短句；不要写物理动作。"
    }
}'''
    )
    print(response)

if __name__ == "__main__":
    test_main()
