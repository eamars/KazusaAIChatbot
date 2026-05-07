"""Health response helpers for the brain service."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from .contracts import (
    Cache2AgentStatsResponse,
    Cache2HealthResponse,
    HealthResponse,
)


CheckDatabaseConnection = Callable[[], Awaitable[bool]]
GetRuntime = Callable[[], object]


async def build_health_response(
    *,
    check_database_connection_func: CheckDatabaseConnection,
    get_rag_cache2_runtime_func: GetRuntime,
    logger: logging.Logger,
) -> HealthResponse:
    """Build the service health response from current runtime dependencies.

    Args:
        check_database_connection_func: Database health check helper.
        get_rag_cache2_runtime_func: Cache2 runtime accessor.
        logger: Logger used for compatibility with service logging.

    Returns:
        Health payload matching the public FastAPI response contract.
    """

    del logger
    db_ok = await check_database_connection_func()

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
