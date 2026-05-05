"""Health response helpers for the brain service."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from .contracts import (
    Cache2AgentStatsResponse,
    Cache2HealthResponse,
    HealthResponse,
)


GetDb = Callable[[], Awaitable[object]]
GetRuntime = Callable[[], object]


async def build_health_response(
    *,
    get_db_func: GetDb,
    get_rag_cache2_runtime_func: GetRuntime,
    logger: logging.Logger,
) -> HealthResponse:
    """Build the service health response from current runtime dependencies.

    Args:
        get_db_func: Database accessor used for the MongoDB ping.
        get_rag_cache2_runtime_func: Cache2 runtime accessor.
        logger: Logger used for compatibility with service logging.

    Returns:
        Health payload matching the public FastAPI response contract.
    """

    db_ok = False
    try:
        db = await get_db_func()
        await db.client.admin.command("ping")
        db_ok = True
    except Exception as exc:
        logger.exception(f"Health check database ping failed: {exc}")

    response = HealthResponse(
        status="ok" if db_ok else "degraded",
        db=db_ok,
        scheduler=True,
        cache2=Cache2HealthResponse(
            agents=[
                Cache2AgentStatsResponse(**row)
                for row in get_rag_cache2_runtime_func().get_agent_stats()
            ],
        ),
    )
    return response

