"""Shared adapter helpers for platform-neutral inline mention tokens."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class InlineMention:
    """A platform-neutral user mention found in outbound text."""

    display_name: str
    platform_user_id: str

    @property
    def token(self) -> str:
        """Return the visible platform-neutral token."""

        return_value = f"@{self.display_name}"
        return return_value


InlineMentionPart = str | InlineMention


def inline_mention_parts(
    text: str,
    delivery_mentions: Sequence[dict] | None,
) -> list[InlineMentionPart]:
    """Split text into plain text and exact inline mention parts.

    Args:
        text: Outbound platform-neutral text from the brain service.
        delivery_mentions: Minimal user mention candidates supplied by the
            brain response or dispatcher callback.

    Returns:
        Ordered parts where renderable candidates replace exact visible tokens
        outside triple-backtick fenced code blocks. Stale or malformed
        candidates are ignored, leaving their visible tokens as text.
    """

    candidates = _mention_candidates(delivery_mentions)
    if not candidates:
        return_value: list[InlineMentionPart] = [text]
        return return_value

    parts: list[InlineMentionPart] = []
    for protected, start_index, end_index in _text_spans(text):
        span_text = text[start_index:end_index]
        if protected:
            if span_text:
                parts.append(span_text)
            continue
        _append_replaced_span(
            parts=parts,
            text=text,
            start_index=start_index,
            end_index=end_index,
            candidates=candidates,
        )

    if not parts:
        parts.append("")
    return parts


def _mention_candidates(
    delivery_mentions: Sequence[dict] | None,
) -> list[InlineMention]:
    """Normalize mention candidates for exact token scanning."""

    if not delivery_mentions:
        return_value: list[InlineMention] = []
        return return_value

    mentions: list[InlineMention] = []
    display_name_counts: dict[str, int] = {}
    for mention in delivery_mentions:
        if not isinstance(mention, dict):
            continue
        if mention.get("entity_kind") != "user":
            continue
        display_name = _display_name(mention)
        platform_user_id = _platform_user_id(mention)
        if not display_name or not platform_user_id:
            continue
        display_name_counts[display_name] = (
            display_name_counts.get(display_name, 0) + 1
        )
        mentions.append(InlineMention(
            display_name=display_name,
            platform_user_id=platform_user_id,
        ))

    candidates = [
        mention for mention in mentions
        if display_name_counts[mention.display_name] == 1
    ]
    return_value = sorted(
        candidates,
        key=lambda mention: len(mention.display_name),
        reverse=True,
    )
    return return_value


def _display_name(mention: dict) -> str:
    """Normalize a candidate display name."""

    value = mention.get("display_name")
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _platform_user_id(mention: dict) -> str:
    """Normalize a candidate platform id."""

    value = mention.get("platform_user_id")
    if value is None:
        return_value = ""
        return return_value
    return_value = str(value).strip()
    return return_value


def _append_replaced_span(
    *,
    parts: list[InlineMentionPart],
    text: str,
    start_index: int,
    end_index: int,
    candidates: Sequence[InlineMention],
) -> None:
    """Append a span with exact inline mention tokens split out."""

    index = start_index
    plain_start = start_index
    while index < end_index:
        match = _match_at_index(
            text=text,
            index=index,
            span_start=start_index,
            end_index=end_index,
            candidates=candidates,
        )
        if match is None:
            index += 1
            continue
        if plain_start < index:
            parts.append(text[plain_start:index])
        parts.append(match)
        index += len(match.token)
        plain_start = index

    if plain_start < end_index:
        parts.append(text[plain_start:end_index])


def _match_at_index(
    *,
    text: str,
    index: int,
    span_start: int,
    end_index: int,
    candidates: Sequence[InlineMention],
) -> InlineMention | None:
    """Return the first candidate whose exact token starts at index."""

    matched: InlineMention | None = None
    for candidate in candidates:
        token = candidate.token
        if not text.startswith(token, index, end_index):
            continue
        after_index = index + len(token)
        if not _has_valid_token_prefix(text, index, span_start):
            continue
        if not _has_valid_token_suffix(text, after_index, end_index):
            continue
        matched = candidate
        break
    return matched


def _has_valid_token_prefix(text: str, index: int, span_start: int) -> bool:
    """Check that a token is not embedded after a word character."""

    if index <= span_start:
        return_value = True
        return return_value
    previous_char = text[index - 1]
    return_value = not (previous_char.isalnum() or previous_char == "_")
    return return_value


def _has_valid_token_suffix(text: str, after_index: int, span_end: int) -> bool:
    """Check that a token is not a prefix of a longer identifier."""

    if after_index >= span_end:
        return_value = True
        return return_value
    next_char = text[after_index]
    return_value = not (next_char.isalnum() or next_char == "_")
    return return_value


def _text_spans(text: str) -> list[tuple[bool, int, int]]:
    """Return spans marked as protected when inside fenced code blocks."""

    spans: list[tuple[bool, int, int]] = []
    search_index = 0
    span_start = 0
    fence = "```"
    while search_index < len(text):
        fence_start = text.find(fence, search_index)
        if fence_start == -1:
            break
        if span_start < fence_start:
            spans.append((False, span_start, fence_start))
        fence_end = text.find(fence, fence_start + len(fence))
        if fence_end == -1:
            spans.append((True, fence_start, len(text)))
            span_start = len(text)
            search_index = len(text)
            break
        protected_end = fence_end + len(fence)
        spans.append((True, fence_start, protected_end))
        span_start = protected_end
        search_index = protected_end

    if span_start < len(text):
        spans.append((False, span_start, len(text)))
    if not spans:
        spans.append((False, 0, 0))
    return spans
