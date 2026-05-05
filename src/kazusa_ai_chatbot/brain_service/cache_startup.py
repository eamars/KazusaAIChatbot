"""Startup hydration helpers for persistent RAG initializer cache."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.utils import log_preview


LoadInitializerEntries = Callable[..., Awaitable[list[dict]]]
GetRuntime = Callable[[], object]


async def hydrate_rag_initializer_cache(
    *,
    load_initializer_entries_func: LoadInitializerEntries,
    get_rag_cache2_runtime_func: GetRuntime,
    cache_name: str,
    max_entries: int,
    logger: logging.Logger,
) -> int:
    """Hydrate persistent initializer cache rows into process-local Cache2.

    Args:
        load_initializer_entries_func: Persistent cache row loader.
        get_rag_cache2_runtime_func: Cache2 runtime accessor.
        cache_name: Cache namespace to hydrate.
        max_entries: Maximum number of persistent rows to load.
        logger: Logger used for compatibility with service logging.

    Returns:
        Number of valid rows loaded into the process-local Cache2 runtime.
    """

    try:
        rows = await load_initializer_entries_func(limit=max_entries)
    except PyMongoError as exc:
        logger.exception(f"Persistent initializer cache hydration failed: {exc}")
        return 0

    runtime = get_rag_cache2_runtime_func()
    loaded_count = 0
    for row in reversed(rows):
        cache_key = row.get("_id")
        result = row.get("result")
        if not isinstance(cache_key, str) or not isinstance(result, dict):
            logger.warning(
                f"Skipping malformed persistent initializer cache row: "
                f"{log_preview(row)}"
            )
            continue

        raw_metadata = row.get("metadata", {})
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        await runtime.store(
            cache_key=cache_key,
            cache_name=cache_name,
            result=result,
            dependencies=[],
            metadata=metadata,
        )
        loaded_count += 1

    logger.info(f"Hydrated {loaded_count} persistent RAG initializer cache entries")
    return loaded_count

