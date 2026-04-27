"""Shared parent interface for RAG helper agents.

The parent class standardizes the agent surface without dictating the internal
execution algorithm. Concrete agents may use a generator-tool-judge loop,
LangGraph, direct DB reads, or any other implementation as long as they expose
the same ``run`` and cache-management interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from kazusa_ai_chatbot.rag.cache2_events import (
    CacheDependency,
    CacheInvalidationEvent,
)
from kazusa_ai_chatbot.rag.cache2_runtime import (
    RAGCache2Runtime,
    get_rag_cache2_runtime,
)


class BaseRAGHelperAgent(ABC):
    """Base interface for RAG helper agents.

    Args:
        name: Stable agent name used in logs, dispatch, and cache metadata.
        cache_name: Optional cache namespace. Leave empty for no-cache agents.
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(
        self,
        *,
        name: str,
        cache_name: str = "",
        cache_runtime: RAGCache2Runtime | None = None,
    ) -> None:
        self.name = name
        self.cache_name = cache_name
        self._cache_runtime = cache_runtime

    @abstractmethod
    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Run the helper agent.

        Args:
            task: Slot description or retrieval task from the outer supervisor.
            context: Runtime hints and known facts supplied by the caller.
            max_attempts: Maximum attempts for agents that support retry loops.

        Returns:
            Agent result dict. Existing supervisor2-compatible agents should
            include ``resolved``, ``result``, ``attempts``, and a standardized
            ``cache`` metadata block.
        """

    def cache_runtime(self) -> RAGCache2Runtime:
        """Return the Cache 2 runtime used by this agent.

        Returns:
            The injected runtime when present, otherwise the process-local
            Cache 2 singleton.
        """
        if self._cache_runtime is not None:
            return self._cache_runtime
        return get_rag_cache2_runtime()

    async def read_cache(self, cache_key: str) -> Any | None:
        """Read one cached payload by exact key.

        Args:
            cache_key: Stable cache key produced by the concrete agent policy.

        Returns:
            Cached payload, or ``None`` when the key is empty, this agent has no
            cache namespace, or the cache misses.
        """
        if not self.cache_name or not cache_key:
            return None
        return await self.cache_runtime().get(
            cache_key,
            cache_name=self.cache_name,
            agent_name=self.name,
        )

    def cache_status(
        self,
        *,
        hit: bool,
        reason: str,
        cache_key: str = "",
    ) -> dict[str, Any]:
        """Build the standardized cache metadata block for agent results.

        Args:
            hit: Whether this agent result was served from cache.
            reason: Short machine-readable cache outcome reason.
            cache_key: Optional exact cache key used for this lookup.

        Returns:
            Cache metadata dict suitable for the ``cache`` field in agent
            results.
        """
        status: dict[str, Any] = {
            "enabled": bool(self.cache_name),
            "hit": hit,
            "cache_name": self.cache_name,
            "reason": reason,
        }
        if cache_key:
            status["cache_key"] = cache_key
        return status

    def with_cache_status(
        self,
        result: dict[str, Any],
        *,
        hit: bool,
        reason: str,
        cache_key: str = "",
    ) -> dict[str, Any]:
        """Attach standardized cache metadata to an agent result.

        Args:
            result: Agent result payload.
            hit: Whether this result was served from cache.
            reason: Short machine-readable cache outcome reason.
            cache_key: Optional exact cache key used for this lookup.

        Returns:
            A copy of ``result`` with a standardized ``cache`` field.
        """
        with_status = dict(result)
        with_status["cache"] = self.cache_status(
            hit=hit,
            reason=reason,
            cache_key=cache_key,
        )
        return with_status

    async def write_cache(
        self,
        *,
        cache_key: str,
        result: Any,
        dependencies: list[CacheDependency],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write one cached payload for this agent.

        Args:
            cache_key: Stable cache key produced by the concrete agent policy.
            result: JSON-like payload to cache.
            dependencies: Data scopes that can invalidate this entry.
            metadata: Optional trace metadata.
        """
        if not self.cache_name or not cache_key:
            return
        await self.cache_runtime().store(
            cache_key=cache_key,
            cache_name=self.cache_name,
            result=result,
            dependencies=dependencies,
            metadata={
                "agent_name": self.name,
                **dict(metadata or {}),
            },
        )

    async def invalidate_cache(self, event: CacheInvalidationEvent) -> int:
        """Invalidate cache entries affected by an event.

        Args:
            event: Domain invalidation event emitted by a writer or agent-local
                invalidation helper.

        Returns:
            Number of Cache 2 entries invalidated.
        """
        return await self.cache_runtime().invalidate(event)
