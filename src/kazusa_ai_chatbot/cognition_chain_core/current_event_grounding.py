"""Prompt-safe current text-chat event projection for cognition stages."""

import re
from collections.abc import Mapping
from typing import Any

MAX_CURRENT_MESSAGE_TEXT_CHARS = 1200
MAX_REPLY_EXCERPT_CHARS = 500
MAX_DISPLAY_NAME_CHARS = 80
MAX_NAME_LIST_ITEMS = 8
_RAW_WIRE_MARKER_PATTERNS = (
    re.compile(r"\[CQ:[^\]]*\]"),
    re.compile(r"<@!?\d+>"),
    re.compile(r"<@&\d+>"),
    re.compile(r"<#\d+>"),
)


def _strip_wire_markers(text: str) -> str:
    """Remove adapter wire markers from text before it reaches prompts."""

    cleaned_text = text
    for pattern in _RAW_WIRE_MARKER_PATTERNS:
        cleaned_text = pattern.sub(" ", cleaned_text)
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text)
    return_value = cleaned_text.strip()
    return return_value


def _trim_text(value: object, *, max_chars: int) -> str:
    """Return bounded prompt text with adapter wire markers removed."""

    if not isinstance(value, str):
        text = ""
    else:
        text = _strip_wire_markers(value)
    if len(text) > max_chars:
        text = f"{text[:max_chars - 3]}..."
    return text


def _clean_display_name(value: object) -> str:
    display_name = _trim_text(
        value,
        max_chars=MAX_DISPLAY_NAME_CHARS,
    )
    return display_name


def _dedupe_bounded_names(values: list[str]) -> list[str]:
    """Return display names without duplicates or unbounded list growth."""

    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = _clean_display_name(value)
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= MAX_NAME_LIST_ITEMS:
            break
    return names


def _string_list(value: object) -> list[str]:
    """Return only string entries from a list-shaped optional value."""

    if not isinstance(value, list):
        strings: list[str] = []
        return strings
    strings = [item for item in value if isinstance(item, str)]
    return strings


def _mention_display_names(value: object) -> list[str]:
    """Return prompt-safe display names from typed mention dictionaries."""

    if not isinstance(value, list):
        display_names: list[str] = []
        return display_names

    names: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        display_name = item.get("display_name")
        if isinstance(display_name, str):
            names.append(display_name)
    display_names = _dedupe_bounded_names(names)
    return display_names


def _addressed_display_names(
    *,
    prompt_message_context: Mapping[str, Any],
    mentions: list[str],
    active_character_display_name: str,
    active_character_global_user_id: str,
) -> list[str]:
    """Return readable names for explicit current-message addressees."""

    addressed_ids = _string_list(
        prompt_message_context.get("addressed_to_global_user_ids"),
    )
    active_character_is_addressed = (
        bool(active_character_global_user_id)
        and active_character_global_user_id in addressed_ids
    )
    names: list[str] = []
    if active_character_is_addressed:
        names.append(active_character_display_name)

    raw_mentions = prompt_message_context.get("mentions")
    if isinstance(raw_mentions, list):
        for item in raw_mentions:
            if not isinstance(item, Mapping):
                continue
            mention_id = item.get("global_user_id")
            if mention_id not in addressed_ids:
                continue
            display_name = item.get("display_name")
            if isinstance(display_name, str):
                names.append(display_name)

    if not names and active_character_is_addressed:
        names.extend(mentions)

    addressed_names = _dedupe_bounded_names(names)
    return addressed_names


def build_current_event_grounding_for_llm(
    *,
    user_input: object,
    prompt_message_context: Mapping[str, Any],
    reply_context: Mapping[str, Any],
    speaker_display_name: str,
    active_character_display_name: str,
    active_character_global_user_id: str,
) -> dict[str, Any]:
    """Return the value for human_payload["current_event_grounding"].

    Args:
        user_input: Current visible user input as carried in cognition state.
        prompt_message_context: Prompt-safe typed current-message projection.
        reply_context: Prompt-safe reply anchor for the current message.
        speaker_display_name: Display name of the current message speaker.
        active_character_display_name: Runtime display name of the character.
        active_character_global_user_id: Internal character id used only for
            deriving booleans and display-name inclusion.

    Returns:
        A bounded prompt payload containing visible current-message facts only.
    """

    body_text = prompt_message_context.get("body_text")
    current_message_text = _trim_text(
        body_text if isinstance(body_text, str) and body_text.strip()
        else user_input,
        max_chars=MAX_CURRENT_MESSAGE_TEXT_CHARS,
    )
    mentions = _mention_display_names(prompt_message_context.get("mentions"))
    addressed_ids = _string_list(
        prompt_message_context.get("addressed_to_global_user_ids"),
    )
    addresses_active_character = (
        bool(active_character_global_user_id)
        and active_character_global_user_id in addressed_ids
    )
    addressed_names = _addressed_display_names(
        prompt_message_context=prompt_message_context,
        mentions=mentions,
        active_character_display_name=active_character_display_name,
        active_character_global_user_id=active_character_global_user_id,
    )
    reply_to_display_name = _clean_display_name(
        reply_context.get("reply_to_display_name"),
    )
    reply_to_active_by_id = (
        bool(active_character_global_user_id)
        and reply_context.get("reply_to_global_user_id")
        == active_character_global_user_id
    )
    reply_to_active_by_display = (
        bool(active_character_display_name)
        and reply_to_display_name == active_character_display_name
    )
    grounding = {
        "speaker_display_name": _clean_display_name(speaker_display_name),
        "current_message_text": current_message_text,
        "mentions": mentions,
        "addressing": {
            "addresses_active_character": addresses_active_character,
            "addressed_display_names": addressed_names,
            "broadcast": prompt_message_context.get("broadcast") is True,
        },
        "reply": {
            "reply_to_display_name": reply_to_display_name,
            "reply_excerpt": _trim_text(
                reply_context.get("reply_excerpt"),
                max_chars=MAX_REPLY_EXCERPT_CHARS,
            ),
            "reply_to_active_character": (
                reply_to_active_by_id or reply_to_active_by_display
            ),
        },
    }
    return grounding
