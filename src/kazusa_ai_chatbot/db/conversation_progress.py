"""Persistence boundary for active V2 conversation-progress packets."""

from __future__ import annotations

import logging
from typing import cast

from pymongo.errors import DuplicateKeyError

from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationProgressScope,
    ConversationProgressStateV2,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    COLLECTION_NAME,
    is_unexpired_storage_timestamp,
)
from kazusa_ai_chatbot.db._client import get_db

logger = logging.getLogger(__name__)

_STORAGE_UTC_PATTERN = (
    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
    r'(?:\.\d+)?\+00:00$'
)


async def load_active_episode_state(
    *,
    scope: ConversationProgressScope,
    current_timestamp_utc: str,
) -> ConversationProgressStateV2 | None:
    """Load only an active, unexpired V2 packet under the exact scope."""

    db = await get_db()
    document = await db[COLLECTION_NAME].find_one(
        {
            'platform': scope.platform,
            'platform_channel_id': scope.platform_channel_id,
            'global_user_id': scope.global_user_id,
            'schema_version': 'conversation_progress.v2',
            'status': 'active',
        },
        projection={'_id': 0},
    )
    if document is None:
        return None
    if not is_unexpired_storage_timestamp(
        document.get('expires_at'),
        current_timestamp_utc=current_timestamp_utc,
    ):
        return None
    return cast(ConversationProgressStateV2, document)


async def replace_episode_state_guarded(
    *,
    document: ConversationProgressStateV2,
) -> bool:
    """Replacement-write a strictly newer packet or replace invalid lifecycle."""

    db = await get_db()
    scope_filter = {
        'platform': document['platform'],
        'platform_channel_id': document['platform_channel_id'],
        'global_user_id': document['global_user_id'],
    }
    guarded_filter = {
        **scope_filter,
        '$or': [
            {'turn_count': {'$lt': document['turn_count']}},
            {'turn_count': {'$exists': False}},
            {'schema_version': {'$ne': 'conversation_progress.v2'}},
            {'status': {'$ne': 'active'}},
            {'expires_at': {'$exists': False}},
            {'expires_at': {'$lte': document['updated_at']}},
            {'expires_at': {'$not': {'$regex': _STORAGE_UTC_PATTERN}}},
        ],
    }
    try:
        result = await db[COLLECTION_NAME].replace_one(
            guarded_filter,
            dict(document),
            upsert=True,
        )
    except DuplicateKeyError as exc:
        logger.debug(
            f'Conversation progress guarded replacement lost race: {exc}'
        )
        return False
    return bool(result.upserted_id is not None or result.modified_count)
