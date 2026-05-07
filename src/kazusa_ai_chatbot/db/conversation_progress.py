"""Persistence helpers for conversation episode progress."""

from __future__ import annotations

import logging

from pymongo.errors import DuplicateKeyError

from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressScope
from kazusa_ai_chatbot.conversation_progress.policy import COLLECTION_NAME
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.schemas import ConversationEpisodeStateDoc

logger = logging.getLogger(__name__)


async def load_episode_state(
    *,
    scope: ConversationProgressScope,
) -> ConversationEpisodeStateDoc | None:
    """Load one episode-state document by scope without MongoDB ``_id``."""

    db = await get_db()
    doc = await db[COLLECTION_NAME].find_one(
        {
            "platform": scope.platform,
            "platform_channel_id": scope.platform_channel_id,
            "global_user_id": scope.global_user_id,
        },
        projection={"_id": 0},
    )
    return doc


async def upsert_episode_state_guarded(
    *,
    document: ConversationEpisodeStateDoc,
) -> bool:
    """Persist one episode document if its turn count is strictly newer."""

    db = await get_db()
    scope_filter = {
        "platform": document["platform"],
        "platform_channel_id": document["platform_channel_id"],
        "global_user_id": document["global_user_id"],
    }
    guarded_filter = {
        **scope_filter,
        "$or": [
            {"turn_count": {"$lt": int(document["turn_count"])}},
            {"turn_count": {"$exists": False}},
        ],
    }
    update = {"$set": dict(document)}
    try:
        result = await db[COLLECTION_NAME].update_one(
            guarded_filter,
            update,
            upsert=True,
        )
    except DuplicateKeyError as exc:
        logger.debug(f"Conversation progress guarded upsert lost race: {exc}")
        return_value = False
        return return_value

    return_value = bool(result.upserted_id is not None or result.modified_count)
    return return_value
