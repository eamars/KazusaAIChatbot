"""Scoped active-block retrieval, projection, and cache contracts."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.conversation_progress import (
    ConversationProgressScope,
)
from kazusa_ai_chatbot.db import conversation_progress_blocks as block_db
from kazusa_ai_chatbot.rag.cache2_policy import (
    build_conversation_search_cache_key,
)
from kazusa_ai_chatbot.rag.conversation_evidence import projection
from kazusa_ai_chatbot.rag.conversation_evidence.workers import search
from tests.conversation_progress_v2_helpers import event, packet


def _context(block_ids: list[str]) -> dict[str, object]:
    """Build matching packet/prompt routing metadata."""

    active_packet = packet(
        turn_count=50,
        compacted_block_refs=block_ids,
    )
    prompt = {
        'schema_version': 'conversation_progress_prompt.v2',
        'episode_state_id': active_packet['episode_state_id'],
        'compacted_block_refs': list(block_ids),
    }
    return {
        'platform': active_packet['platform'],
        'platform_channel_id': active_packet['platform_channel_id'],
        'global_user_id': active_packet['global_user_id'],
        'conversation_episode_state': active_packet,
        'conversation_progress': prompt,
    }


def _block_result(
    block_id: str,
    *,
    score: float = 0.8,
) -> dict[str, object]:
    """Build one prompt-safe block search result."""

    return {
        'source_kind': 'conversation_progress_block',
        'block_id': block_id,
        'narrative': 'Earlier completed interaction remains relevant.',
        'events': [event(
            event_id=f'event-{block_id}',
            state='completed',
            retention='background',
        )],
        'source_started_at': '2026-07-28T08:00:00+00:00',
        'source_ended_at': '2026-07-28T09:00:00+00:00',
        'covered_turn_refs': ['row:older-row'],
        'score': score,
    }


@pytest.mark.asyncio
async def test_no_active_block_refs_performs_no_block_search(monkeypatch):
    block_search = AsyncMock()
    monkeypatch.setattr(
        search,
        'search_conversation_progress_blocks',
        block_search,
    )

    rows = await search._active_progress_block_rows(
        args={'search_query': 'earlier completed interaction'},
        context=_context([]),
    )

    assert rows == []
    block_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_active_block_search_uses_exact_scope_refs_and_top_three(
    monkeypatch,
):
    block_search = AsyncMock(return_value=[
        _block_result('block-a'),
        _block_result('block-b', score=0.7),
    ])
    monkeypatch.setattr(
        search,
        'search_conversation_progress_blocks',
        block_search,
    )

    rows = await search._active_progress_block_rows(
        args={'search_query': 'earlier completed interaction'},
        context=_context(['block-a', 'block-b']),
    )

    block_search.assert_awaited_once_with(
        query='earlier completed interaction',
        scope=ConversationProgressScope(
            platform='qq',
            platform_channel_id='channel_test',
            global_user_id='user_test',
        ),
        episode_state_id='episode_progress_v2_test',
        active_block_ids=['block-a', 'block-b'],
        limit=3,
    )
    assert [row['block_id'] for row in rows] == ['block-a', 'block-b']
    assert rows[0]['methods'] == [
        'semantic:conversation_progress_block'
    ]


def test_score_merge_preserves_conversation_and_block_source_coverage():
    conversation_rows = [
        {
            'conversation_row_id': f'row-{index}',
            'score': 0.99 - index / 100,
        }
        for index in range(5)
    ]
    block_rows = [_block_result('block-low', score=0.01)]

    rows = search._merge_active_progress_block_rows(
        conversation_rows,
        block_rows,
        selected_limit=3,
    )

    assert len(rows) == 3
    assert any(
        row.get('source_kind') == 'conversation_progress_block'
        for row in rows
    )
    assert any('conversation_row_id' in row for row in rows)


def test_block_projection_exposes_semantics_and_protects_lineage():
    block = _block_result('block-a')

    projected = projection._message_projection([block])

    assert 'Earlier completed interaction remains relevant.' in (
        projected['summaries'][0]
    )
    assert 'state=completed' in projected['summaries'][0]
    assert 'source range:' in projected['summaries'][0]
    assert projected['rows'][0]['block_id'] == 'block-a'
    assert any(
        ref.get('ref_type') == 'conversation_progress_block'
        and ref.get('block_id') == 'block-a'
        for ref in projected['resolved_refs']
    )
    assert any(
        ref.get('ref_type') == 'conversation_progress_source'
        and ref.get('ref_id') == 'row_source_1'
        for ref in projected['resolved_refs']
    )


def test_active_block_signature_isolated_in_conversation_cache_key():
    without_blocks = build_conversation_search_cache_key(
        'find prior interaction',
        _context([]),
    )
    with_blocks = build_conversation_search_cache_key(
        'find prior interaction',
        _context(['block-a']),
    )
    reordered = build_conversation_search_cache_key(
        'find prior interaction',
        _context(['block-b', 'block-a']),
    )
    sorted_order = build_conversation_search_cache_key(
        'find prior interaction',
        _context(['block-a', 'block-b']),
    )

    assert without_blocks != with_blocks
    assert reordered == sorted_order


@pytest.mark.asyncio
async def test_active_block_worker_bypasses_read_and_write_cache(monkeypatch):
    agent = search.ConversationSearchAgent()
    read_cache = AsyncMock(return_value=[{'cached': True}])
    write_cache = AsyncMock()
    monkeypatch.setattr(agent, 'read_cache', read_cache)
    monkeypatch.setattr(agent, 'write_cache', write_cache)
    monkeypatch.setattr(
        search,
        '_generator',
        AsyncMock(return_value={'search_query': 'prior interaction'}),
    )
    monkeypatch.setattr(
        search,
        '_tool',
        AsyncMock(return_value=[_block_result('block-a')]),
    )
    monkeypatch.setattr(
        search,
        '_judge',
        AsyncMock(return_value=(True, '')),
    )

    result = await agent.run(
        'find prior interaction',
        _context(['block-a']),
        max_attempts=1,
    )

    read_cache.assert_not_awaited()
    write_cache.assert_not_awaited()
    assert result['cache']['reason'] == 'miss_active_progress_blocks'


class _AggregateCursor:
    """Capture one aggregate pipeline and return bounded rows."""

    def __init__(self, rows):
        self.rows = rows

    async def to_list(self, *, length):
        return self.rows[:length]


class _FindCursor:
    """Return exact graph rows selected by block ID."""

    def __init__(self, rows):
        self.rows = rows

    async def to_list(self, *, length):
        return self.rows[:length]


class _BlockCollection:
    """Capture graph reads, vector search, and expiry refreshes."""

    def __init__(self, rows, *, graph_rows=None):
        self.rows = rows
        self.graph_rows = (
            graph_rows
            if graph_rows is not None
            else [
                _graph_row(row['block_id'])
                for row in rows
            ]
        )
        self.pipeline = None
        self.find_queries = []
        self.update_query = None

    def aggregate(self, pipeline):
        self.pipeline = pipeline
        return _AggregateCursor(self.rows)

    def find(self, query, projection):
        del projection
        self.find_queries.append(query)
        selected_ids = set(query['block_id']['$in'])
        return _FindCursor([
            row for row in self.graph_rows
            if row['block_id'] in selected_ids
        ])

    async def update_many(self, query, update):
        self.update_query = (query, update)
        return type('_Result', (), {
            'modified_count': len(query['block_id']['$in']),
        })()


def _graph_row(
    block_id,
    *,
    source_block_ids=None,
    superseded_by_block_id='',
):
    """Build structural block-graph metadata for the DB boundary."""

    return {
        'block_id': block_id,
        'platform': 'qq',
        'platform_channel_id': 'channel_test',
        'global_user_id': 'user_test',
        'episode_state_id': 'episode_progress_v2_test',
        'source_block_ids': list(source_block_ids or []),
        'superseded_by_block_id': superseded_by_block_id,
    }


@pytest.mark.asyncio
async def test_db_vector_filter_is_exact_and_result_limit_is_three(
    monkeypatch,
):
    collection = _BlockCollection([
        _block_result(f'block-{index}', score=0.9 - index / 10)
        for index in range(5)
    ])
    monkeypatch.setattr(
        block_db,
        'get_db',
        AsyncMock(return_value={
            'conversation_episode_blocks': collection,
        }),
    )
    monkeypatch.setattr(
        block_db,
        'get_query_text_embedding',
        AsyncMock(return_value=[0.1, 0.2]),
    )

    rows = await block_db.search_conversation_progress_blocks(
        query='prior interaction',
        scope=ConversationProgressScope(
            'qq',
            'channel_test',
            'user_test',
        ),
        episode_state_id='episode_progress_v2_test',
        active_block_ids=['block-0', 'block-1', 'block-2'],
        limit=99,
    )

    vector_stage = collection.pipeline[0]['$vectorSearch']
    filters = vector_stage['filter']['$and']
    assert {'platform': {'$eq': 'qq'}} in filters
    assert {'platform_channel_id': {'$eq': 'channel_test'}} in filters
    assert {'global_user_id': {'$eq': 'user_test'}} in filters
    assert {
        'episode_state_id': {'$eq': 'episode_progress_v2_test'}
    } in filters
    assert {
        'block_id': {'$in': ['block-0', 'block-1', 'block-2']}
    } in filters
    assert vector_stage['limit'] == 3
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_db_search_includes_transitive_superseded_child_blocks(
    monkeypatch,
):
    """Search every exact child reachable from the active packet roots."""

    collection = _BlockCollection(
        [_block_result('child-block')],
        graph_rows=[
            _graph_row(
                'active-root',
                source_block_ids=['child-block'],
            ),
            _graph_row(
                'child-block',
                superseded_by_block_id='active-root',
            ),
        ],
    )
    monkeypatch.setattr(
        block_db,
        'get_db',
        AsyncMock(return_value={
            'conversation_episode_blocks': collection,
        }),
    )
    monkeypatch.setattr(
        block_db,
        'get_query_text_embedding',
        AsyncMock(return_value=[0.1, 0.2]),
    )

    rows = await block_db.search_conversation_progress_blocks(
        query='old child event',
        scope=ConversationProgressScope(
            'qq',
            'channel_test',
            'user_test',
        ),
        episode_state_id='episode_progress_v2_test',
        active_block_ids=['active-root'],
        limit=3,
    )

    vector_filters = collection.pipeline[0]['$vectorSearch']['filter']['$and']
    assert {
        'block_id': {'$in': ['active-root', 'child-block']}
    } in vector_filters
    assert rows[0]['block_id'] == 'child-block'
