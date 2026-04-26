"""Event and dependency types for the session-scoped RAG Cache 2 runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheDependency:
    """Describe the data scope a cache entry depends on.

    Args:
        source: Logical data source, for example ``"user_profile"`` or
            ``"conversation_history"``.
        platform: Optional platform scope.
        platform_channel_id: Optional channel scope.
        global_user_id: Optional internal user UUID scope.
        display_name: Optional display-name scope for lookup-style caches.
        from_timestamp: Optional lower time bound for historical ranges.
        to_timestamp: Optional upper time bound for historical ranges.
    """

    source: str
    platform: str = ""
    platform_channel_id: str = ""
    global_user_id: str = ""
    display_name: str = ""
    from_timestamp: str = ""
    to_timestamp: str = ""


@dataclass(frozen=True)
class CacheInvalidationEvent:
    """Describe a write event that may invalidate cache entries.

    Args:
        source: Logical data source that changed.
        platform: Optional platform scope.
        platform_channel_id: Optional channel scope.
        global_user_id: Optional internal user UUID scope.
        display_name: Optional display-name scope.
        timestamp: Optional write timestamp, used for range-overlap checks.
        reason: Human-readable invalidation reason for logs and metrics.
    """

    source: str
    platform: str = ""
    platform_channel_id: str = ""
    global_user_id: str = ""
    display_name: str = ""
    timestamp: str = ""
    reason: str = ""
