"""Active V2 packet read and guarded replacement contracts."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo.errors import DuplicateKeyError

from kazusa_ai_chatbot.conversation_progress import repository
from kazusa_ai_chatbot.db import conversation_progress as progress_db
from tests.conversation_progress_v2_helpers import SCOPE, packet


@pytest.mark.asyncio
async def test_db_read_uses_exact_v2_active_scope(monkeypatch):
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=packet())
    monkeypatch.setattr(
        progress_db,
        'get_db',
        AsyncMock(return_value={
            'conversation_episode_state': collection,
        }),
    )

    result = await progress_db.load_active_episode_state(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    )

    collection.find_one.assert_awaited_once_with(
        {
            'platform': 'qq',
            'platform_channel_id': 'channel_test',
            'global_user_id': 'user_test',
            'schema_version': 'conversation_progress.v2',
            'status': 'active',
        },
        projection={'_id': 0},
    )
    assert result == packet()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'expires_at',
    [
        '2026-07-28T09:30:00+00:00',
        '2026-07-28T09:29:59+00:00',
        'invalid',
        None,
    ],
)
async def test_db_read_excludes_expired_or_invalid_lifecycle(
    monkeypatch,
    expires_at,
):
    document = packet()
    document['expires_at'] = expires_at
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=document)
    monkeypatch.setattr(
        progress_db,
        'get_db',
        AsyncMock(return_value={
            'conversation_episode_state': collection,
        }),
    )

    result = await progress_db.load_active_episode_state(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    )

    assert result is None


@pytest.mark.asyncio
async def test_repository_rejects_malformed_v2_after_db_read(monkeypatch):
    malformed = packet()
    malformed['events'] = [{'not': 'an event'}]
    monkeypatch.setattr(
        repository,
        'load_active_episode_state',
        AsyncMock(return_value=malformed),
    )

    result = await repository.load_active_packet(
        scope=SCOPE,
        current_timestamp_utc='2026-07-28T09:30:00+00:00',
    )

    assert result is None


@pytest.mark.asyncio
async def test_guarded_replacement_accepts_newer_or_invalid_lifecycle(
    monkeypatch,
):
    collection = MagicMock()
    collection.replace_one = AsyncMock(return_value=SimpleNamespace(
        upserted_id=None,
        modified_count=1,
    ))
    monkeypatch.setattr(
        progress_db,
        'get_db',
        AsyncMock(return_value={
            'conversation_episode_state': collection,
        }),
    )
    document = packet(turn_count=2)

    written = await progress_db.replace_episode_state_guarded(
        document=document,
    )

    assert written is True
    query, replacement = collection.replace_one.await_args.args
    assert replacement == document
    assert query['platform'] == 'qq'
    assert query['platform_channel_id'] == 'channel_test'
    assert query['global_user_id'] == 'user_test'
    assert {'turn_count': {'$lt': 2}} in query['$or']
    assert {
        'schema_version': {'$ne': 'conversation_progress.v2'}
    } in query['$or']
    assert {'status': {'$ne': 'active'}} in query['$or']
    assert {'expires_at': {'$exists': False}} in query['$or']
    assert collection.replace_one.await_args.kwargs == {'upsert': True}


@pytest.mark.asyncio
async def test_closed_tombstone_can_be_replaced_by_fresh_turn_one(
    monkeypatch,
):
    collection = MagicMock()
    collection.replace_one = AsyncMock(return_value=SimpleNamespace(
        upserted_id=None,
        modified_count=1,
    ))
    monkeypatch.setattr(
        progress_db,
        'get_db',
        AsyncMock(return_value={
            'conversation_episode_state': collection,
        }),
    )

    written = await progress_db.replace_episode_state_guarded(
        document=packet(turn_count=1),
    )

    query = collection.replace_one.await_args.args[0]
    assert written is True
    assert {'status': {'$ne': 'active'}} in query['$or']


@pytest.mark.asyncio
async def test_duplicate_scope_on_newer_active_packet_is_lost_write(
    monkeypatch,
):
    collection = MagicMock()
    collection.replace_one = AsyncMock(
        side_effect=DuplicateKeyError('newer active row won'),
    )
    monkeypatch.setattr(
        progress_db,
        'get_db',
        AsyncMock(return_value={
            'conversation_episode_state': collection,
        }),
    )

    written = await progress_db.replace_episode_state_guarded(
        document=deepcopy(packet(turn_count=2)),
    )

    assert written is False
