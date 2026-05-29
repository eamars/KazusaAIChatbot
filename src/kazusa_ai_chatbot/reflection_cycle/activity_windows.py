"""Deterministic activity windows derived from monitored group scopes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from kazusa_ai_chatbot.reflection_cycle.models import (
    READONLY_REFLECTION_MAX_MESSAGE_CHARS,
    ReflectionInputSet,
    ReflectionScopeInput,
)
from kazusa_ai_chatbot.time_boundary import (
    normalize_storage_utc_iso,
    parse_storage_utc_datetime,
)
from kazusa_ai_chatbot.utils import text_or_empty

GROUP_ACTIVITY_WINDOW_MINUTES = 15
GROUP_ACTIVITY_VISIBLE_CONTEXT_LIMIT = 6


@dataclass
class GroupActivityWindow:
    """Bounded group activity window used as self-cognition evidence."""

    scope_ref: str
    platform: str
    platform_channel_id: str
    channel_type: str
    window_index: int
    window_start: datetime
    window_end: datetime
    labels: dict[str, str]
    message_count: int
    visible_context: list[dict[str, str]]
    participant_rows: list[dict[str, Any]]
    source_refs: list[dict[str, Any]]

    @property
    def source_id(self) -> str:
        """Return the stable source id used for idempotency."""

        source_id = str(self.source_refs[0]["source_id"])
        return source_id

    @property
    def semantic_labels(self) -> dict[str, str]:
        """Return semantic labels under the source-collector name."""

        return_value = self.labels
        return return_value

    @property
    def last_evidence_timestamp_utc(self) -> str:
        """Return the latest visible context timestamp in storage UTC."""

        if self.visible_context:
            return_value = self.visible_context[-1]["timestamp"]
            return return_value
        return_value = normalize_storage_utc_iso(self.window_end.isoformat())
        return return_value

    @property
    def source_message_refs(self) -> list[dict[str, str]]:
        """Return bounded message refs derived from visible context rows."""

        refs = [
            {
                "message_id": row["platform_message_id"],
                "timestamp": row["timestamp"],
                "role": row["role"],
            }
            for row in self.visible_context
        ]
        return refs


def build_group_activity_windows(
    *,
    scope: ReflectionScopeInput,
    window_start: datetime,
    window_end: datetime,
    now: datetime,
    character_global_user_id: str = "",
    platform_bot_id: str = "",
) -> list[GroupActivityWindow]:
    """Project one monitored group into non-empty 15-minute windows.

    Args:
        scope: Monitor-eligible group scope and its bounded messages.
        window_start: Inclusive projection lower bound.
        window_end: Exclusive projection upper bound.
        now: Current tick time for recency labeling.
        character_global_user_id: Internal identity of the active character.
        platform_bot_id: Platform identity of the active bot account.

    Returns:
        Chronological windows. Empty windows and private scopes are omitted.
    """

    if scope.channel_type != "group":
        return_value: list[GroupActivityWindow] = []
        return return_value

    aligned_start = _aligned_window_start(window_start)
    aligned_end = parse_storage_utc_datetime(
        normalize_storage_utc_iso(window_end.isoformat()),
    )
    buckets: dict[datetime, list[dict[str, Any]]] = {}
    for message in scope.messages:
        timestamp = _message_timestamp(message)
        if timestamp is None:
            continue
        if timestamp < aligned_start or timestamp >= aligned_end:
            continue
        message_window_start = _aligned_window_start(timestamp)
        if message_window_start not in buckets:
            buckets[message_window_start] = []
        buckets[message_window_start].append(message)

    windows: list[GroupActivityWindow] = []
    for bucket_start in sorted(buckets):
        messages = sorted(
            buckets[bucket_start],
            key=lambda item: text_or_empty(item.get("timestamp")),
        )
        if not messages:
            continue
        bucket_end = bucket_start + timedelta(
            minutes=GROUP_ACTIVITY_WINDOW_MINUTES,
        )
        labels = _semantic_labels(
            messages=messages,
            window_end=bucket_end,
            now=now,
            character_global_user_id=character_global_user_id,
            platform_bot_id=platform_bot_id,
        )
        window = GroupActivityWindow(
            scope_ref=scope.scope_ref,
            platform=scope.platform,
            platform_channel_id=scope.platform_channel_id,
            channel_type=scope.channel_type,
            window_index=_window_index(
                base_window_start=aligned_start,
                current_window_start=bucket_start,
            ),
            window_start=bucket_start,
            window_end=bucket_end,
            labels=labels,
            message_count=len(messages),
            visible_context=_visible_context(messages),
            participant_rows=_participant_rows(messages),
            source_refs=_source_refs(
                scope_ref=scope.scope_ref,
                window_start=bucket_start,
                window_end=bucket_end,
                labels=labels,
            ),
        )
        windows.append(window)

    return_value = windows
    return return_value


def project_group_activity_windows(
    input_set: ReflectionInputSet,
) -> list[GroupActivityWindow]:
    """Project recent-first group windows from a reflection input set."""

    window_start = parse_storage_utc_datetime(input_set.effective_start)
    window_end = parse_storage_utc_datetime(input_set.effective_end)
    now = parse_storage_utc_datetime(input_set.effective_end)
    windows: list[GroupActivityWindow] = []
    for scope in input_set.selected_scopes:
        scope_windows = build_group_activity_windows(
            scope=scope,
            window_start=window_start,
            window_end=window_end,
            now=now,
        )
        windows.extend(scope_windows)
    windows.sort(key=lambda item: item.window_start, reverse=True)
    return windows


def build_hourly_aggregation_preview(
    windows: list[GroupActivityWindow],
) -> list[dict[str, Any]]:
    """Group four 15-minute labels into hourly reflection-compatible cards."""

    grouped: dict[datetime, list[GroupActivityWindow]] = {}
    for window in windows:
        hour_start = window.window_start.replace(
            minute=0,
            second=0,
            microsecond=0,
        )
        if hour_start not in grouped:
            grouped[hour_start] = []
        grouped[hour_start].append(window)

    preview: list[dict[str, Any]] = []
    for hour_start in sorted(grouped):
        hour_windows = sorted(
            grouped[hour_start],
            key=lambda item: item.window_start,
        )
        card = {
            "hour_start": normalize_storage_utc_iso(hour_start.isoformat()),
            "window_count": len(hour_windows),
            "message_count": sum(window.message_count for window in hour_windows),
            "summary_labels": [
                window.labels["window_summary"]
                for window in hour_windows
            ],
        }
        preview.append(card)
    return preview


def _semantic_labels(
    *,
    messages: list[dict[str, Any]],
    window_end: datetime,
    now: datetime,
    character_global_user_id: str,
    platform_bot_id: str,
) -> dict[str, str]:
    """Build descriptive labels from inspectable group-window metrics."""

    participant_count = len(_participant_keys(messages))
    message_count = len(messages)
    assistant_present = any(
        message.get("role") == "assistant"
        for message in messages
    )
    directly_addressed = _direct_addressed(
        messages,
        character_global_user_id=character_global_user_id,
        platform_bot_id=platform_bot_id,
    )
    labels = {
        "activity_level": _activity_level(message_count),
        "speaker_diversity": _speaker_diversity(participant_count),
        "assistant_presence": _assistant_presence(assistant_present),
        "bot_addressing": _bot_addressing(directly_addressed),
        "message_recency": _message_recency(window_end=window_end, now=now),
        "noise_level": _noise_level(
            message_count=message_count,
            participant_count=participant_count,
        ),
        "response_risk": _response_risk(
            message_count=message_count,
            participant_count=participant_count,
            assistant_present=assistant_present,
            directly_addressed=directly_addressed,
        ),
    }
    labels["window_summary"] = _window_summary(labels)
    return labels


def _activity_level(message_count: int) -> str:
    """Convert message volume into a stable activity label."""

    if message_count <= 3:
        return_value = "quiet"
    elif message_count <= 8:
        return_value = "active"
    elif message_count <= 45:
        return_value = "bursty"
    else:
        return_value = "very_busy"
    return return_value


def _speaker_diversity(participant_count: int) -> str:
    """Convert unique user count into a group participation label."""

    if participant_count <= 0:
        return_value = "no_speakers"
    elif participant_count == 1:
        return_value = "one_speaker"
    elif participant_count <= 4:
        return_value = "few_speakers"
    else:
        return_value = "many_speakers"
    return return_value


def _assistant_presence(assistant_present: bool) -> str:
    """Label whether the character spoke inside the activity window."""

    if assistant_present:
        return_value = "present"
    else:
        return_value = "not_in_window"
    return return_value


def _bot_addressing(directly_addressed: bool) -> str:
    """Return a semantic label for direct-address evidence."""

    if directly_addressed:
        return_value = "directly_addressed"
    else:
        return_value = "ambient_group_context"
    return return_value


def _message_recency(*, window_end: datetime, now: datetime) -> str:
    """Label whether the activity window is recent to the current tick."""

    if now - window_end <= timedelta(hours=1):
        return_value = "recent"
    else:
        return_value = "older"
    return return_value


def _noise_level(*, message_count: int, participant_count: int) -> str:
    """Convert activity and participation into a noise label."""

    if message_count >= 10 or participant_count >= 6:
        return_value = "high"
    elif message_count >= 4 or participant_count >= 3:
        return_value = "medium"
    else:
        return_value = "low"
    return return_value


def _response_risk(
    *,
    message_count: int,
    participant_count: int,
    assistant_present: bool,
    directly_addressed: bool,
) -> str:
    """Label how risky visible speech would be in the group window."""

    if directly_addressed or assistant_present:
        return_value = "low"
    elif message_count >= 10 or participant_count >= 5:
        return_value = "high"
    elif message_count >= 4 or participant_count >= 3:
        return_value = "medium"
    else:
        return_value = "unclear"
    return return_value


def _window_summary(labels: dict[str, str]) -> str:
    """Build the compact summary used by source refs and hourly previews."""

    summary = (
        f"{labels['activity_level']} group activity, "
        f"{labels['speaker_diversity']} speakers, "
        f"{labels['bot_addressing']}, "
        f"{labels['assistant_presence']}, "
        f"risk {labels['response_risk']}"
    )
    return summary


def _visible_context(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build bounded visible context rows for self-cognition."""

    selected_messages = messages[-GROUP_ACTIVITY_VISIBLE_CONTEXT_LIMIT:]
    visible_rows: list[dict[str, str]] = []
    for message in selected_messages:
        body_text = _trim_text(text_or_empty(message.get("body_text")))
        if not body_text:
            continue
        visible_row = {
            "timestamp": text_or_empty(message.get("timestamp")),
            "role": text_or_empty(message.get("role")),
            "display_name": text_or_empty(message.get("display_name")),
            "body_text": body_text,
            "platform_message_id": (
                text_or_empty(message.get("platform_message_id"))
                or text_or_empty(message.get("message_id"))
            ).replace("+", ""),
        }
        visible_rows.append(visible_row)
    return visible_rows


def _participant_rows(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build internal participant rows from the activity-window message set."""

    rows: list[dict[str, Any]] = []
    for message in messages:
        body_text = _trim_text(text_or_empty(message.get("body_text")))
        row = {
            "timestamp": text_or_empty(message.get("timestamp")),
            "role": text_or_empty(message.get("role")),
            "display_name": text_or_empty(message.get("display_name")),
            "body_text": body_text,
            "platform_message_id": (
                text_or_empty(message.get("platform_message_id"))
                or text_or_empty(message.get("message_id"))
            ).replace("+", ""),
            "global_user_id": text_or_empty(message.get("global_user_id")),
            "platform_user_id": text_or_empty(message.get("platform_user_id")),
            "addressed_to_global_user_ids": _string_list(
                message.get("addressed_to_global_user_ids"),
            ),
            "mentions": _mention_rows(message.get("mentions")),
            "is_directed_at_character": (
                message.get("is_directed_at_character") is True
            ),
            "reply_context": _reply_context(message.get("reply_context")),
        }
        rows.append(row)
    return rows


def _string_list(value: object) -> list[str]:
    """Return non-empty string values from an optional list-like field."""

    if not isinstance(value, list):
        return_value: list[str] = []
        return return_value

    items = [
        text_value
        for item in value
        if (text_value := text_or_empty(item))
    ]
    return items


def _mention_rows(value: object) -> list[dict[str, str]]:
    """Project mention metadata into string-only internal rows."""

    if not isinstance(value, list):
        return_value: list[dict[str, str]] = []
        return return_value

    rows: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        row = {
            key: text_value
            for key, raw_value in item.items()
            if isinstance(key, str)
            if (text_value := text_or_empty(raw_value))
        }
        if row:
            rows.append(row)
    return rows


def _reply_context(value: object) -> dict[str, Any]:
    """Return reply metadata only when the source row carries a mapping."""

    if not isinstance(value, dict):
        return_value: dict[str, Any] = {}
        return return_value
    return_value = dict(value)
    return return_value


def _source_refs(
    *,
    scope_ref: str,
    window_start: datetime,
    window_end: datetime,
    labels: dict[str, str],
) -> list[dict[str, Any]]:
    """Build the stable source ref for one activity window."""

    source_id = (
        f"{scope_ref}:"
        f"{normalize_storage_utc_iso(window_start.isoformat())}:"
        f"{normalize_storage_utc_iso(window_end.isoformat())}"
    )
    refs = [
        {
            "source_kind": "reflection_activity_window",
            "source_id": source_id,
            "due_at": None,
            "summary": labels["window_summary"],
        }
    ]
    return refs


def _direct_addressed(
    messages: list[dict[str, Any]],
    *,
    character_global_user_id: str,
    platform_bot_id: str,
) -> bool:
    """Return whether any row addresses the active character."""

    target_global_user_id = text_or_empty(character_global_user_id)
    target_platform_bot_id = text_or_empty(platform_bot_id)
    for message in messages:
        addressed = message.get("addressed_to_global_user_ids")
        if _addressed_list_targets_character(
            addressed,
            target_global_user_id=target_global_user_id,
        ):
            return_value = True
            return return_value
        mentions = message.get("mentions")
        if _mentions_target_character(
            mentions,
            target_global_user_id=target_global_user_id,
            target_platform_bot_id=target_platform_bot_id,
        ):
            return_value = True
            return return_value
        if message.get("is_directed_at_character") is True:
            return_value = True
            return return_value
    return_value = False
    return return_value


def _addressed_list_targets_character(
    addressed: object,
    *,
    target_global_user_id: str,
) -> bool:
    """Return whether addressed ids include the active character id."""

    if not target_global_user_id or not isinstance(addressed, list):
        return_value = False
        return return_value
    for addressed_id in addressed:
        if text_or_empty(addressed_id) == target_global_user_id:
            return_value = True
            return return_value
    return_value = False
    return return_value


def _mentions_target_character(
    mentions: object,
    *,
    target_global_user_id: str,
    target_platform_bot_id: str,
) -> bool:
    """Return whether mention metadata points to the active character."""

    if not isinstance(mentions, list):
        return_value = False
        return return_value
    for mention in mentions:
        if not isinstance(mention, dict):
            continue
        mention_global_user_id = text_or_empty(mention.get("global_user_id"))
        if (
            target_global_user_id
            and mention_global_user_id == target_global_user_id
        ):
            return_value = True
            return return_value
        mention_platform_user_id = text_or_empty(mention.get("platform_user_id"))
        if (
            target_platform_bot_id
            and mention_platform_user_id == target_platform_bot_id
        ):
            return_value = True
            return return_value
    return_value = False
    return return_value


def _participant_keys(messages: list[dict[str, Any]]) -> set[str]:
    """Return stable user participant keys for a group window."""

    keys: set[str] = set()
    for message in messages:
        if message.get("role") != "user":
            continue
        for field_name in ("global_user_id", "platform_user_id", "display_name"):
            value = text_or_empty(message.get(field_name))
            if value:
                keys.add(f"{field_name}:{value}")
                break
    return keys


def _message_timestamp(message: dict[str, Any]) -> datetime | None:
    """Parse one external message timestamp, ignoring malformed rows."""

    timestamp_text = text_or_empty(message.get("timestamp"))
    if not timestamp_text:
        return_value = None
        return return_value
    try:
        timestamp = parse_storage_utc_datetime(timestamp_text)
    except ValueError:
        return_value = None
        return return_value
    return timestamp


def _aligned_window_start(value: datetime) -> datetime:
    """Return the aligned 15-minute UTC window start."""

    normalized_utc = parse_storage_utc_datetime(
        normalize_storage_utc_iso(value.isoformat()),
    )
    minute = (
        normalized_utc.minute
        // GROUP_ACTIVITY_WINDOW_MINUTES
        * GROUP_ACTIVITY_WINDOW_MINUTES
    )
    window_start = normalized_utc.replace(
        minute=minute,
        second=0,
        microsecond=0,
    )
    return window_start


def _window_index(
    *,
    base_window_start: datetime,
    current_window_start: datetime,
) -> int:
    """Return the one-based 15-minute window index within a range."""

    elapsed = current_window_start - base_window_start
    index = int(elapsed.total_seconds() // 60 // GROUP_ACTIVITY_WINDOW_MINUTES)
    return_value = index + 1
    return return_value


def _trim_text(value: str) -> str:
    """Trim group-window text to the reflection prompt-safe row budget."""

    cleaned = " ".join(value.split())
    if len(cleaned) <= READONLY_REFLECTION_MAX_MESSAGE_CHARS:
        return_value = cleaned
        return return_value
    return_value = f"{cleaned[:READONLY_REFLECTION_MAX_MESSAGE_CHARS - 3]}..."
    return return_value
