"""Database health helpers exposed through the public DB boundary."""

from __future__ import annotations

import logging

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.db._client import get_db

logger = logging.getLogger(__name__)


async def check_database_connection() -> bool:
    """Return whether the configured MongoDB database accepts a ping."""

    try:
        db = await get_db()
        await db.client.admin.command("ping")
    except PyMongoError as exc:
        logger.exception(f"Database health ping failed: {exc}")
        return_value = False
        return return_value

    return_value = True
    return return_value
