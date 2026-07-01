"""Build platform-neutral outbound mention render candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_inline_delivery_mentions(
    *,
    text: str,
    users: Sequence[Mapping[str, Any]],
    character_global_user_id: str | None = None,
) -> list[dict[str, str]]:
    """Build minimal render candidates for authored inline user tags.

    Args:
        text: Outbound platform-neutral text authored by the brain.
        users: Bounded user rows already available to the response path.
        character_global_user_id: Internal id of the active character account,
            excluded so the bot does not mention itself.

    Returns:
        User mention candidates whose exact `@display_name` token appears
        outside fenced code blocks. Each candidate contains only the fields an
        adapter needs for native rendering.
    """

    display_name_counts = _display_name_counts(
        users=users,
        character_global_user_id=character_global_user_id,
    )
    sorted_users = sorted(
        users,
        key=_display_name_sort_key,
        reverse=True,
    )

    candidates: list[dict[str, str]] = []
    emitted_display_names: set[str] = set()
    for user in sorted_users:
        display_name = _display_name(user)
        if not display_name:
            continue
        if _global_user_id(user) == character_global_user_id:
            continue
        if display_name in emitted_display_names:
            continue
        if display_name_counts[display_name] > 1:
            continue
        platform_user_id = _platform_user_id(user)
        if not platform_user_id:
            continue
        if not _has_token_outside_fenced_blocks(text, display_name):
            continue

        candidates.append({
            "entity_kind": "user",
            "display_name": display_name,
            "platform_user_id": platform_user_id,
        })
        emitted_display_names.add(display_name)

    return candidates


def _display_name_counts(
    *,
    users: Sequence[Mapping[str, Any]],
    character_global_user_id: str | None,
) -> dict[str, int]:
    """Count visible names before renderability filtering for ambiguity."""

    counts: dict[str, int] = {}
    for user in users:
        if _global_user_id(user) == character_global_user_id:
            continue
        display_name = _display_name(user)
        if not display_name:
            continue
        counts[display_name] = counts.get(display_name, 0) + 1
    return counts


def _display_name_sort_key(user: Mapping[str, Any]) -> int:
    """Sort longer display names first for deterministic token scanning."""

    display_name = _display_name(user)
    return_value = len(display_name)
    return return_value


def _display_name(user: Mapping[str, Any]) -> str:
    """Normalize a user display label for exact outbound token matching."""

    value = user.get("display_name")
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _platform_user_id(user: Mapping[str, Any]) -> str:
    """Normalize adapter-renderable platform identity to a non-empty string."""

    value = user.get("platform_user_id")
    if value is None:
        return_value = ""
        return return_value
    return_value = str(value).strip()
    return return_value


def _global_user_id(user: Mapping[str, Any]) -> str:
    """Normalize internal identity for active-character exclusion."""

    value = user.get("global_user_id")
    if value is None:
        return_value = ""
        return return_value
    return_value = str(value).strip()
    return return_value


def _has_token_outside_fenced_blocks(text: str, display_name: str) -> bool:
    """Return whether text contains an exact valid tag outside fenced blocks."""

    token = f"@{display_name}"
    for start_index, end_index in _unfenced_spans(text):
        search_index = start_index
        while search_index < end_index:
            match_index = text.find(token, search_index, end_index)
            if match_index == -1:
                break
            after_index = match_index + len(token)
            if (
                _has_valid_token_prefix(text, match_index, start_index)
                and _has_valid_token_suffix(text, after_index, end_index)
            ):
                return_value = True
                return return_value
            search_index = match_index + 1
    return_value = False
    return return_value


def _has_valid_token_prefix(text: str, match_index: int, span_start: int) -> bool:
    """Check that an exact token is not embedded after a word character."""

    if match_index <= span_start:
        return_value = True
        return return_value
    previous_char = text[match_index - 1]
    return_value = not (previous_char.isalnum() or previous_char == "_")
    return return_value


def _has_valid_token_suffix(text: str, after_index: int, span_end: int) -> bool:
    """Check that an exact token is not a prefix of a longer identifier."""

    if after_index >= span_end:
        return_value = True
        return return_value
    next_char = text[after_index]
    return_value = not (next_char.isalnum() or next_char == "_")
    return return_value


def _unfenced_spans(text: str) -> list[tuple[int, int]]:
    """Return text spans outside triple-backtick fenced code blocks."""

    spans: list[tuple[int, int]] = []
    search_index = 0
    span_start = 0
    fence = "```"
    while search_index < len(text):
        fence_start = text.find(fence, search_index)
        if fence_start == -1:
            break
        spans.append((span_start, fence_start))
        fence_end = text.find(fence, fence_start + len(fence))
        if fence_end == -1:
            search_index = len(text)
            span_start = len(text)
            break
        search_index = fence_end + len(fence)
        span_start = search_index

    spans.append((span_start, len(text)))
    return spans
