"""One-shot cleanup for legacy RAG1 MongoDB collections.

Run before deploying the RAG2-only build to ensure no stale collections persist.
Safe to run repeatedly.
"""

from __future__ import annotations

import asyncio
import logging

from kazusa_ai_chatbot.db._client import close_db, get_db

logger = logging.getLogger(__name__)


async def main() -> None:
    """Drop legacy RAG1 collections when they exist.

    Returns:
        None. Missing collections are logged and ignored.
    """
    db = await get_db()
    existing = set(await db.list_collection_names())
    for name in ("rag_cache_index", "rag_metadata_index"):
        if name in existing:
            await db.drop_collection(name)
            logger.info(f'Dropped collection \'{name}\'')
        else:
            logger.info(f'Collection \'{name}\' not present; skipping')
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
