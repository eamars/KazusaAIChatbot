"""Validated active-packet cache contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from kazusa_ai_chatbot.conversation_progress import cache
from tests.conversation_progress_v2_helpers import SCOPE, packet


def setup_function() -> None:
    """Clear process-local state before every cache contract."""

    cache.clear_cache()


def test_newer_active_packet_is_copied_into_and_out_of_cache():
    source = packet(turn_count=3)

    assert cache.put_cached_packet(scope=SCOPE, packet=source) is True
    source['turn_count'] = 99
    cached = cache.get_cached_packet(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    )

    assert cached is not None
    assert cached['turn_count'] == 3
    cached['turn_count'] = 88
    second = cache.get_cached_packet(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    )
    assert second is not None
    assert second['turn_count'] == 3


def test_equal_or_older_turn_count_cannot_replace_cached_packet():
    assert cache.put_cached_packet(
        scope=SCOPE,
        packet=packet(turn_count=3),
    )

    assert cache.put_cached_packet(
        scope=SCOPE,
        packet=packet(turn_count=3),
    ) is False
    assert cache.put_cached_packet(
        scope=SCOPE,
        packet=packet(turn_count=2),
    ) is False


def test_closed_packet_invalidates_scope():
    assert cache.put_cached_packet(
        scope=SCOPE,
        packet=packet(turn_count=3),
    )
    closed = packet(turn_count=3, status='closed')

    assert cache.put_cached_packet(scope=SCOPE, packet=closed) is False
    assert cache.get_cached_packet(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    ) is None


def test_expired_semantic_lifecycle_is_evicted():
    expired = packet()
    expired['expires_at'] = '2026-07-28T09:30:00+00:00'
    assert cache.put_cached_packet(scope=SCOPE, packet=expired)

    assert cache.get_cached_packet(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    ) is None


def test_monotonic_ttl_is_enforced(monkeypatch):
    monkeypatch.setattr(cache.time, 'monotonic', lambda: 100.0)
    assert cache.put_cached_packet(scope=SCOPE, packet=packet())
    monkeypatch.setattr(
        cache.time,
        'monotonic',
        lambda: 100.0 + cache.CACHE_TTL_SECONDS,
    )

    assert cache.get_cached_packet(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    ) is None


def test_scope_mismatch_is_rejected():
    wrong_scope_packet = deepcopy(packet())
    wrong_scope_packet['global_user_id'] = 'other-user'

    with pytest.raises(ValueError, match='scope'):
        cache.put_cached_packet(
            scope=SCOPE,
            packet=wrong_scope_packet,
        )
