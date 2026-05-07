"""One-shot cleanup for legacy RAG1 MongoDB collections.

Run before deploying the RAG2-only build to ensure no stale collections persist.
Safe to run repeatedly.
"""

from __future__ import annotations

import asyncio
import logging

from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db.script_operations import drop_legacy_rag_collections

logger = logging.getLogger(__name__)


async def main() -> None:
    """Drop legacy RAG1 collections when they exist.

    Returns:
        None. Missing collections are logged and ignored.
    """
    dropped = await drop_legacy_rag_collections(
        ("rag_cache_index", "rag_metadata_index"),
    )
    for name in ("rag_cache_index", "rag_metadata_index"):
        if name in dropped:
            logger.info(f'Dropped collection \'{name}\'')
        else:
            logger.info(f'Collection \'{name}\' not present; skipping')
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
