"""Persistence and semantic search for immutable progress blocks."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, cast

from pymongo.errors import DuplicateKeyError

from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationEpisodeBlockV1,
    ConversationProgressBlockSearchResultV1,
    ConversationProgressScope,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    BLOCK_COLLECTION_NAME,
    BLOCK_VECTOR_INDEX_NAME,
    MAX_ACTIVE_BLOCK_REFS,
    MAX_BLOCK_GRAPH_DEPTH,
    MAX_BLOCK_SEARCH_RESULTS,
    MAX_BLOCK_SOURCE_BLOCKS,
    MAX_REACHABLE_BLOCK_REFS,
)
from kazusa_ai_chatbot.db._client import get_db, get_query_text_embedding
from kazusa_ai_chatbot.rag import cache2_runtime
from kazusa_ai_chatbot.rag.cache2_events import CacheInvalidationEvent

logger = logging.getLogger(__name__)


async def insert_conversation_progress_block(
    *,
    document: ConversationEpisodeBlockV1,
) -> bool:
    """Insert one immutable block idempotently and reject ID collisions."""

    db = await get_db()
    try:
        result = await db[BLOCK_COLLECTION_NAME].update_one(
            {'block_id': document['block_id']},
            {'$setOnInsert': dict(document)},
            upsert=True,
        )
    except DuplicateKeyError as exc:
        raise ValueError(
            'conversation progress block identity collision'
        ) from exc
    inserted = result.upserted_id is not None
    if not inserted:
        existing = await db[BLOCK_COLLECTION_NAME].find_one(
            {'block_id': document['block_id']},
            projection={'_id': 0},
        )
        if (
            existing is None
            or _immutable_block_projection(existing)
            != _immutable_block_projection(document)
        ):
            raise ValueError(
                'conversation progress block ID has different content'
            )
    if inserted:
        await _invalidate_conversation_search(document)
    return inserted


async def load_conversation_progress_block_graph(
    *,
    root_block_ids: Sequence[str],
    scope: ConversationProgressScope,
    episode_state_id: str,
) -> list[ConversationEpisodeBlockV1]:
    """Load the exact bounded block graph protected by active roots."""

    if not root_block_ids:
        return []
    db = await get_db()
    collection = db[BLOCK_COLLECTION_NAME]
    block_ids = await _reachable_block_ids(
        collection=collection,
        root_block_ids=root_block_ids,
        scope=scope,
        episode_state_id=episode_state_id,
    )
    cursor = collection.find(
        {
            'block_id': {'$in': block_ids},
            'platform': scope.platform,
            'platform_channel_id': scope.platform_channel_id,
            'global_user_id': scope.global_user_id,
            'episode_state_id': episode_state_id,
        },
        projection={'_id': 0},
    )
    documents = await cursor.to_list(length=len(block_ids))
    by_id = {document['block_id']: document for document in documents}
    if set(by_id) != set(block_ids):
        raise ValueError('conversation progress block graph is incomplete')
    return [
        cast(ConversationEpisodeBlockV1, by_id[block_id])
        for block_id in block_ids
    ]


async def touch_conversation_progress_blocks(
    *,
    block_ids: Sequence[str],
    expires_at: str,
    purge_after: object,
) -> int:
    """Refresh mutable expiry metadata on every protected reachable block."""

    if not block_ids:
        return 0
    db = await get_db()
    result = await db[BLOCK_COLLECTION_NAME].update_many(
        {'block_id': {'$in': list(block_ids)}},
        {'$set': {
            'expires_at': expires_at,
            'purge_after': purge_after,
        }},
    )
    return result.modified_count


async def supersede_conversation_progress_blocks(
    *,
    source_block_ids: Sequence[str],
    superseded_by_block_id: str,
) -> int:
    """Mark exact unsuperseded source blocks after packet replacement wins."""

    if not source_block_ids:
        return 0
    db = await get_db()
    result = await db[BLOCK_COLLECTION_NAME].update_many(
        {
            'block_id': {'$in': list(source_block_ids)},
            'superseded_by_block_id': '',
        },
        {'$set': {'superseded_by_block_id': superseded_by_block_id}},
    )
    return result.modified_count


async def search_conversation_progress_blocks(
    *,
    query: str,
    scope: ConversationProgressScope,
    episode_state_id: str,
    active_block_ids: Sequence[str],
    limit: int = MAX_BLOCK_SEARCH_RESULTS,
) -> list[ConversationProgressBlockSearchResultV1]:
    """Search the bounded graph protected by exact active packet roots."""

    if not query.strip() or not active_block_ids:
        return []
    effective_limit = min(max(limit, 1), MAX_BLOCK_SEARCH_RESULTS)
    db = await get_db()
    collection = db[BLOCK_COLLECTION_NAME]
    reachable_block_ids = await _reachable_block_ids(
        collection=collection,
        root_block_ids=active_block_ids,
        scope=scope,
        episode_state_id=episode_state_id,
    )
    query_embedding = await get_query_text_embedding(query)
    vector_filter = {
        '$and': [
            {'platform': {'$eq': scope.platform}},
            {
                'platform_channel_id': {
                    '$eq': scope.platform_channel_id,
                },
            },
            {'global_user_id': {'$eq': scope.global_user_id}},
            {'episode_state_id': {'$eq': episode_state_id}},
            {'block_id': {'$in': reachable_block_ids}},
        ],
    }
    pipeline: list[dict[str, Any]] = [
        {
            '$vectorSearch': {
                'index': BLOCK_VECTOR_INDEX_NAME,
                'path': 'embedding',
                'queryVector': query_embedding,
                'numCandidates': max(effective_limit * 8, 24),
                'limit': effective_limit,
                'filter': vector_filter,
            },
        },
        {'$addFields': {'score': {'$meta': 'vectorSearchScore'}}},
        {'$unset': ['_id', 'embedding']},
    ]
    rows = await collection.aggregate(pipeline).to_list(
        length=effective_limit
    )
    results: list[ConversationProgressBlockSearchResultV1] = []
    for row in rows:
        results.append({
            'source_kind': 'conversation_progress_block',
            'block_id': str(row['block_id']),
            'narrative': str(row['narrative']),
            'events': deepcopy(row['events']),
            'source_started_at': str(row['source_started_at']),
            'source_ended_at': str(row['source_ended_at']),
            'covered_turn_refs': list(row['covered_turn_refs']),
            'score': float(row.get('score', 0.0)),
        })
    return results


async def _reachable_block_ids(
    *,
    collection: Any,
    root_block_ids: Sequence[str],
    scope: ConversationProgressScope,
    episode_state_id: str,
) -> list[str]:
    """Resolve one bounded, exact, same-scope block graph breadth first."""

    roots = _validated_root_block_ids(root_block_ids)
    ordered_ids: list[str] = []
    pending_ids = roots
    parent_by_id: dict[str, str] = {
        block_id: '' for block_id in roots
    }
    depth = 0
    while pending_ids:
        if depth > MAX_BLOCK_GRAPH_DEPTH:
            raise ValueError(
                'conversation progress block graph exceeds its depth cap'
            )
        cursor = collection.find(
            {
                'block_id': {'$in': pending_ids},
                'platform': scope.platform,
                'platform_channel_id': scope.platform_channel_id,
                'global_user_id': scope.global_user_id,
                'episode_state_id': episode_state_id,
            },
            projection={
                '_id': 0,
                'block_id': 1,
                'source_block_ids': 1,
                'superseded_by_block_id': 1,
            },
        )
        documents = await cursor.to_list(length=len(pending_ids))
        by_id = {
            str(document.get('block_id', '')): document
            for document in documents
            if isinstance(document, Mapping)
        }
        if set(by_id) != set(pending_ids):
            raise ValueError(
                'conversation progress block graph is incomplete'
            )

        next_ids: list[str] = []
        for block_id in pending_ids:
            document = by_id[block_id]
            superseded_by = document.get('superseded_by_block_id')
            expected_parent = parent_by_id[block_id]
            if not isinstance(superseded_by, str):
                raise ValueError(
                    'conversation progress block supersession is invalid'
                )
            if expected_parent:
                if superseded_by not in {'', expected_parent}:
                    raise ValueError(
                        'conversation progress child block lineage is invalid'
                    )
            elif superseded_by:
                raise ValueError(
                    'active conversation progress block is superseded'
                )
            source_ids = _validated_source_block_ids(
                document.get('source_block_ids')
            )
            ordered_ids.append(block_id)
            for source_id in source_ids:
                prior_parent = parent_by_id.get(source_id)
                if prior_parent is not None:
                    raise ValueError(
                        'conversation progress block graph is cyclic '
                        'or has shared children'
                    )
                parent_by_id[source_id] = block_id
                next_ids.append(source_id)

        if len(ordered_ids) + len(next_ids) > MAX_REACHABLE_BLOCK_REFS:
            raise ValueError(
                'conversation progress block graph exceeds its node cap'
            )
        pending_ids = next_ids
        depth += 1
    return ordered_ids


def _validated_root_block_ids(value: Sequence[str]) -> list[str]:
    """Validate active root IDs before any graph query."""

    if (
        isinstance(value, (str, bytes))
        or len(value) > MAX_ACTIVE_BLOCK_REFS
    ):
        raise ValueError('active conversation progress roots are invalid')
    roots: list[str] = []
    for block_id in value:
        if not isinstance(block_id, str) or not block_id:
            raise ValueError(
                'active conversation progress root ID is invalid'
            )
        if block_id in roots:
            raise ValueError(
                'active conversation progress roots are duplicated'
            )
        roots.append(block_id)
    return roots


def _validated_source_block_ids(value: object) -> list[str]:
    """Validate one immutable block's direct child references."""

    if not isinstance(value, list) or len(value) > MAX_BLOCK_SOURCE_BLOCKS:
        raise ValueError(
            'conversation progress source block IDs are invalid'
        )
    source_ids: list[str] = []
    for block_id in value:
        if not isinstance(block_id, str) or not block_id:
            raise ValueError(
                'conversation progress source block ID is invalid'
            )
        if block_id in source_ids:
            raise ValueError(
                'conversation progress source block IDs are duplicated'
            )
        source_ids.append(block_id)
    return source_ids


def _immutable_block_projection(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Project every block field except the three mutable metadata fields."""

    return {
        field_name: deepcopy(field_value)
        for field_name, field_value in document.items()
        if field_name not in {
            'superseded_by_block_id',
            'expires_at',
            'purge_after',
        }
    }


async def _invalidate_conversation_search(
    document: ConversationEpisodeBlockV1,
) -> None:
    """Invalidate conversation evidence dependencies after a block write."""

    event = CacheInvalidationEvent(
        source='conversation_history',
        platform=document['platform'],
        platform_channel_id=document['platform_channel_id'],
        global_user_id=document['global_user_id'],
        storage_timestamp_utc=document['created_at'],
        reason='conversation_progress_block_insert',
    )
    try:
        await cache2_runtime.get_rag_cache2_runtime().invalidate(event)
    except Exception as exc:
        logger.warning(
            f'Cache2 invalidation after progress block insert failed: {exc}'
        )
