"""Controller-facing V2 resolution-thread repository boundary."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from pymongo.errors import AutoReconnect

from agentic_resolver.contracts import ResolutionThreadRecordV2
from agentic_resolver.errors import ResolutionPersistenceError
from kazusa_ai_chatbot.db import resolution_threads
from kazusa_ai_chatbot.db.errors import DatabaseOperationError

_INDEX_RETRY_DELAYS_SECONDS = (0.05, 0.1)


class ResolutionThreadRepository(Protocol):
    """Durable async lifecycle operations required by the V2 controller."""

    async def get_thread(
        self, resolution_thread_id: str
    ) -> ResolutionThreadRecordV2 | None:
        """Return one validated V2 thread record."""

        ...

    async def get_operation(
        self, resolution_thread_id: str, operation_id: str
    ) -> dict[str, Any] | None:
        """Return one durable operation when present."""

        ...

    async def create_thread_v2(
        self, *args: Any, **kwargs: Any
    ) -> ResolutionThreadRecordV2:
        """Create one V2 thread and its first segment."""

        ...

    async def prepare_operation(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Create or return one operation under its identity fence."""

        ...

    async def update_operation(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Persist one operation outcome."""

        ...

    async def acquire_lease(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Acquire one fenced activation lease."""

        ...

    async def renew_lease(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Renew one current activation lease."""

        ...

    async def release_lease(self, *args: Any, **kwargs: Any) -> None:
        """Release one current activation lease."""

        ...

    async def rotate_segment(
        self, *args: Any, **kwargs: Any
    ) -> ResolutionThreadRecordV2:
        """Rotate the current thread segment under compatibility fencing."""

        ...

    async def update_segment(
        self, *args: Any, **kwargs: Any
    ) -> ResolutionThreadRecordV2:
        """Persist one segment lifecycle update."""

        ...

    async def validate_fence(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Validate the current activation and lease epoch."""

        ...


class MongoResolutionThreadRepository:
    """Async adapter delegating raw selectors to the V2 DB owner module."""

    def __init__(self) -> None:
        self._db = resolution_threads
        self._indexes_ready = False
        self._index_lock = asyncio.Lock()

    async def ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        async with self._index_lock:
            if self._indexes_ready:
                return
            for attempt in range(len(_INDEX_RETRY_DELAYS_SECONDS) + 1):
                try:
                    await self._db.ensure_indexes()
                except DatabaseOperationError as exc:
                    if (
                        not isinstance(exc.__cause__, AutoReconnect)
                        or attempt >= len(_INDEX_RETRY_DELAYS_SECONDS)
                    ):
                        raise ResolutionPersistenceError(
                            "failed to ensure resolution thread indexes",
                        ) from exc
                    await asyncio.sleep(_INDEX_RETRY_DELAYS_SECONDS[attempt])
                else:
                    self._indexes_ready = True
                    return

    async def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Translate database-owner failures into resolver taxonomy."""

        method = getattr(self._db, name)
        try:
            value = await method(*args, **kwargs)
        except DatabaseOperationError as exc:
            raise ResolutionPersistenceError(
                f"resolution repository operation failed: {name}",
            ) from exc
        return value

    async def get_thread(
        self, resolution_thread_id: str
    ) -> ResolutionThreadRecordV2 | None:
        value = await self._call("get_thread", resolution_thread_id)
        if value is None:
            return None
        record = ResolutionThreadRecordV2.from_mapping(value)
        return record

    async def get_operation(
        self, resolution_thread_id: str, operation_id: str
    ) -> dict[str, Any] | None:
        operation = await self._call(
            "get_operation",
            resolution_thread_id,
            operation_id,
        )
        return operation

    async def create_thread_v2(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ResolutionThreadRecordV2:
        await self.ensure_indexes()
        value = await self._call("create_thread_v2", *args, **kwargs)
        record = ResolutionThreadRecordV2.from_mapping(value)
        return record

    async def prepare_operation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        operation = await self._call("prepare_operation", *args, **kwargs)
        return operation

    async def update_operation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        operation = await self._call("update_operation", *args, **kwargs)
        return operation

    async def acquire_lease(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        lease = await self._call("acquire_lease", *args, **kwargs)
        return lease

    async def renew_lease(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        lease = await self._call("renew_lease", *args, **kwargs)
        return lease

    async def release_lease(self, *args: Any, **kwargs: Any) -> None:
        await self._call("release_lease", *args, **kwargs)

    async def rotate_segment(
        self, *args: Any, **kwargs: Any
    ) -> ResolutionThreadRecordV2:
        value = await self._call("rotate_segment", *args, **kwargs)
        record = ResolutionThreadRecordV2.from_mapping(value)
        return record

    async def update_segment(
        self, *args: Any, **kwargs: Any
    ) -> ResolutionThreadRecordV2:
        value = await self._call("update_segment", *args, **kwargs)
        record = ResolutionThreadRecordV2.from_mapping(value)
        return record

    async def validate_fence(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        fence = await self._call("validate_fence", *args, **kwargs)
        return fence
