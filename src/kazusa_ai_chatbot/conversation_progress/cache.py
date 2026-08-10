"""Process-local validated cache for active V2 progress packets."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass

from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationProgressScope,
    ConversationProgressStateV2,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    CACHE_TTL_SECONDS,
    is_unexpired_storage_timestamp,
)
from kazusa_ai_chatbot.conversation_progress.repository import (
    validate_active_packet,
)

_CacheKey = tuple[str, str, str]


@dataclass
class _CacheEntry:
    """One bounded cache entry."""

    packet: ConversationProgressStateV2
    monotonic_expires_at: float


_entries: dict[_CacheKey, _CacheEntry] = {}


def get_cached_packet(
    *,
    scope: ConversationProgressScope,
    current_timestamp_utc: str,
) -> ConversationProgressStateV2 | None:
    """Return one valid active unexpired packet, evicting stale entries."""

    key = _cache_key(scope)
    entry = _entries.get(key)
    if entry is None:
        return None
    if entry.monotonic_expires_at <= time.monotonic():
        _entries.pop(key, None)
        return None
    try:
        packet = validate_active_packet(entry.packet)
    except ValueError:
        _entries.pop(key, None)
        return None
    if (
        packet['status'] != 'active'
        or not is_unexpired_storage_timestamp(
            packet['expires_at'],
            current_timestamp_utc=current_timestamp_utc,
        )
    ):
        _entries.pop(key, None)
        return None
    return deepcopy(packet)


def put_cached_packet(
    *,
    scope: ConversationProgressScope,
    packet: ConversationProgressStateV2,
) -> bool:
    """Store only a newer valid active packet under its exact scope."""

    validated = validate_active_packet(packet)
    if validated['status'] != 'active':
        invalidate_cached_packet(scope=scope)
        return False
    if (
        validated['platform'] != scope.platform
        or validated['platform_channel_id'] != scope.platform_channel_id
        or validated['global_user_id'] != scope.global_user_id
    ):
        raise ValueError('cached packet scope does not match its key')
    key = _cache_key(scope)
    existing = _entries.get(key)
    if (
        existing is not None
        and existing.packet['turn_count'] >= validated['turn_count']
    ):
        return False
    _entries[key] = _CacheEntry(
        packet=deepcopy(validated),
        monotonic_expires_at=time.monotonic() + CACHE_TTL_SECONDS,
    )
    return True


def invalidate_cached_packet(*, scope: ConversationProgressScope) -> None:
    """Remove one scoped cache entry."""

    _entries.pop(_cache_key(scope), None)


def clear_cache() -> None:
    """Remove every progress cache entry for deterministic tests."""

    _entries.clear()


def _cache_key(scope: ConversationProgressScope) -> _CacheKey:
    """Build the exact stable scope key."""

    return (
        scope.platform,
        scope.platform_channel_id,
        scope.global_user_id,
    )
