"""Runtime-owned constraints for RAG retrieval workers."""

from __future__ import annotations

import datetime
import re
from typing import Any

from kazusa_ai_chatbot.time_boundary import (
    local_date_bounds_to_storage_utc_iso,
    local_llm_datetime_to_storage_utc_iso,
    one_second_before_storage_utc_iso,
    parse_storage_utc_datetime,
)
from kazusa_ai_chatbot.utils import text_or_empty

_ASCII_DOUBLE_QUOTE_RE = re.compile(r'"([^"\n]{1,120})"')
_ASCII_SINGLE_QUOTE_RE = re.compile(r"'([^'\n]{1,120})'")
_CJK_QUOTE_PATTERNS = (
    re.compile(r'\u201c([^\u201d\n]{1,120})\u201d'),
    re.compile(r'\u2018([^\u2019\n]{1,120})\u2019'),
    re.compile(r'\u300c([^\u300d\n]{1,120})\u300d'),
    re.compile(r'\u300e([^\u300f\n]{1,120})\u300f'),
)
_URL_RE = re.compile(r'https?://[^\s)>\]}"\']+')
_EXACT_FIELD_RE = re.compile(
    r'\b(?:memory_name|dedup_key|tag|filename)\s+([^\s,;，；]+)',
    re.IGNORECASE,
)
_RELATIVE_TWO_DAY_MARKERS = (
    '这两天',
    '最近两天',
    '两天内',
    'last two days',
    'last 2 days',
    'past two days',
    'past 2 days',
    'these two days',
)
_RELATIVE_DAY_MARKERS = (
    ('前天', -2),
    ('day before yesterday', -2),
    ('昨天', -1),
    ('yesterday', -1),
    ('今天', 0),
    ('today', 0),
)
_TRUSTED_CONVERSATION_USER_SCOPES = {
    'current_user',
    'active_character',
    'person_resolved',
}


def literal_anchors_from_text(value: str, *, limit: int) -> list[str]:
    """Extract bounded literal anchors from task text.

    Args:
        value: Slot text or query text that may contain quoted literals,
            URLs, filenames, memory names, or other exact identifiers.
        limit: Maximum number of anchors to return.

    Returns:
        First-seen non-empty literal anchors.
    """

    anchors: list[str] = []
    text = text_or_empty(value)
    for pattern in (
        _ASCII_DOUBLE_QUOTE_RE,
        _ASCII_SINGLE_QUOTE_RE,
        *_CJK_QUOTE_PATTERNS,
        _URL_RE,
        _EXACT_FIELD_RE,
    ):
        for match in pattern.finditer(text):
            if match.lastindex:
                anchor = match.group(1)
            else:
                anchor = match.group(0)
            _append_anchor(anchors, anchor, limit=limit)
            if len(anchors) >= limit:
                return anchors

    return anchors


def apply_conversation_runtime_constraints(
    args: dict[str, Any],
    *,
    context: dict[str, Any],
    task: str,
    literal_anchor_limit: int,
) -> dict[str, Any]:
    """Reapply trusted conversation scope and relative-time constraints.

    Args:
        args: Arguments normalized from the LLM generator.
        context: Runtime context supplied by the caller.
        task: Slot text being served by the worker.
        literal_anchor_limit: Maximum literal anchors to retain.

    Returns:
        A copied argument dict whose platform/channel/user/time constraints are
        grounded in trusted runtime state rather than generated text alone.
    """

    constrained = dict(args)
    for key in ('platform', 'platform_channel_id'):
        value = text_or_empty(context.get(key))
        if value:
            constrained[key] = value

    user_filter = _conversation_user_filter(context)
    global_user_id = user_filter.get('global_user_id')
    if global_user_id:
        constrained['global_user_id'] = global_user_id
    else:
        constrained.pop('global_user_id', None)
    constrained.pop('display_name', None)

    context_bounds = _explicit_time_bounds_from_context(context)
    relative_bounds = _relative_time_bounds_from_context(task, context)
    if context_bounds:
        constrained.update(context_bounds)
    if relative_bounds is not None:
        from_timestamp, to_timestamp = relative_bounds
        if 'from_timestamp' not in context_bounds:
            constrained['from_timestamp'] = from_timestamp
        if 'to_timestamp' not in context_bounds:
            constrained['to_timestamp'] = to_timestamp

    generated_anchors = constrained.get('literal_anchors')
    anchors = generated_anchors if isinstance(generated_anchors, list) else []
    normalized_anchors: list[str] = []
    for anchor in anchors:
        _append_anchor(normalized_anchors, text_or_empty(anchor), limit=literal_anchor_limit)
    for anchor in literal_anchors_from_text(task, limit=literal_anchor_limit):
        _append_anchor(normalized_anchors, anchor, limit=literal_anchor_limit)
    if normalized_anchors:
        constrained['literal_anchors'] = normalized_anchors
    else:
        constrained.pop('literal_anchors', None)

    return constrained


def apply_conversation_filter_runtime_constraints(
    args: dict[str, Any],
    *,
    context: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    """Reapply trusted scope and relative-time bounds to filter retrieval."""

    constrained = dict(args)
    for key in ('platform', 'platform_channel_id'):
        value = text_or_empty(context.get(key))
        if value:
            constrained[key] = value

    user_filter = _conversation_user_filter(context)
    if user_filter:
        constrained.update(user_filter)
    else:
        constrained.pop('global_user_id', None)
        constrained.pop('display_name', None)

    context_bounds = _explicit_time_bounds_from_context(context)
    relative_bounds = _relative_time_bounds_from_context(task, context)
    if context_bounds:
        constrained.update(context_bounds)
    if relative_bounds is not None:
        from_timestamp, to_timestamp = relative_bounds
        if 'from_timestamp' not in context_bounds:
            constrained['from_timestamp'] = from_timestamp
        if 'to_timestamp' not in context_bounds:
            constrained['to_timestamp'] = to_timestamp

    return constrained


def apply_source_memory_runtime_constraints(
    args: dict[str, Any],
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Enforce trusted source-user filtering for shared memory search."""

    constrained = dict(args)
    source_global_user_id = text_or_empty(context.get('source_global_user_id'))
    if source_global_user_id:
        constrained['source_global_user_id'] = source_global_user_id
    else:
        constrained.pop('source_global_user_id', None)

    return constrained


def _append_anchor(anchors: list[str], raw_anchor: str, *, limit: int) -> None:
    """Append one normalized anchor when it is non-empty and new."""

    anchor = raw_anchor.strip()
    if not anchor or anchor in anchors or len(anchors) >= limit:
        return
    anchors.append(anchor)


def _conversation_user_filter(context: dict[str, Any]) -> dict[str, str]:
    """Return trusted author filters only for explicit author-scoped slots."""

    scope = text_or_empty(context.get('conversation_user_scope'))
    if scope not in _TRUSTED_CONVERSATION_USER_SCOPES:
        return_value: dict[str, str] = {}
        return return_value

    user_filter: dict[str, str] = {}
    global_user_id = text_or_empty(context.get('global_user_id'))
    if global_user_id:
        user_filter['global_user_id'] = global_user_id
    display_name = text_or_empty(context.get('display_name'))
    if display_name:
        user_filter['display_name'] = display_name
    return user_filter


def _explicit_time_bounds_from_context(context: dict[str, Any]) -> dict[str, str]:
    """Return caller-supplied timestamp bounds independently by field."""

    from_timestamp = _coerce_timestamp_bound(context.get('from_timestamp'))
    to_timestamp = _coerce_timestamp_bound(context.get('to_timestamp'))
    bounds: dict[str, str] = {}
    if from_timestamp:
        bounds['from_timestamp'] = from_timestamp
    if to_timestamp:
        bounds['to_timestamp'] = to_timestamp
    return bounds


def _coerce_timestamp_bound(value: object) -> str:
    """Normalize a trusted timestamp bound without accepting free text."""

    text = text_or_empty(value)
    if not text:
        return ''
    try:
        storage_datetime = parse_storage_utc_datetime(text)
    except ValueError:
        try:
            normalized = local_llm_datetime_to_storage_utc_iso(text)
        except ValueError:
            return_value = ''
            return return_value
        return normalized
    return_value = storage_datetime.isoformat()
    return return_value


def _relative_time_bounds_from_context(
    task: str,
    context: dict[str, Any],
) -> tuple[str, str] | None:
    """Infer local-date bounds for explicit relative-day retrieval tasks."""

    local_date = _current_local_date(context)
    if local_date is None:
        return_value = None
        return return_value

    haystack = _relative_time_haystack(task, context)
    if any(marker in haystack for marker in _RELATIVE_TWO_DAY_MARKERS):
        start_date = local_date - datetime.timedelta(days=1)
        bounds = _inclusive_local_date_range(start_date, local_date)
        return bounds

    for marker, offset_days in _RELATIVE_DAY_MARKERS:
        if marker not in haystack:
            continue
        target_date = local_date + datetime.timedelta(days=offset_days)
        bounds = _inclusive_local_date_range(target_date, target_date)
        return bounds

    return_value = None
    return return_value


def _relative_time_haystack(task: str, context: dict[str, Any]) -> str:
    """Build lowercase text used only for deterministic relative-day detection."""

    parts = [
        task,
        text_or_empty(context.get('original_query')),
        text_or_empty(context.get('current_slot')),
    ]
    haystack = '\n'.join(part for part in parts if part).lower()
    return haystack


def _current_local_date(context: dict[str, Any]) -> datetime.date | None:
    """Read the character-local current date from runtime time context."""

    local_time_context = context.get('local_time_context')
    if not isinstance(local_time_context, dict):
        return_value = None
        return return_value

    current_local_datetime = text_or_empty(
        local_time_context.get('current_local_datetime')
    )
    local_date_text = current_local_datetime.split(' ', 1)[0]
    try:
        local_date = datetime.date.fromisoformat(local_date_text)
    except ValueError:
        return_value = None
        return return_value

    return local_date


def _inclusive_local_date_range(
    start_date: datetime.date,
    end_date: datetime.date,
) -> tuple[str, str]:
    """Convert local start/end dates to inclusive UTC timestamp bounds."""

    start_timestamp, _ = local_date_bounds_to_storage_utc_iso(
        start_date.isoformat()
    )
    _, exclusive_end_timestamp = local_date_bounds_to_storage_utc_iso(
        end_date.isoformat()
    )
    to_timestamp = one_second_before_storage_utc_iso(exclusive_end_timestamp)
    return_value = (start_timestamp, to_timestamp)
    return return_value
