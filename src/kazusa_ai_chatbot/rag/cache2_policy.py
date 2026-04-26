"""Per-agent cache policies for RAG Cache 2."""

from __future__ import annotations

import re
from typing import Any

from kazusa_ai_chatbot.rag.cache2_events import CacheDependency
from kazusa_ai_chatbot.rag.cache2_runtime import stable_cache_key

USER_LOOKUP_CACHE_NAME = "rag2_user_lookup_agent"
USER_LOOKUP_POLICY_VERSION = "user_lookup:v1"

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_cache_text(value: object) -> str:
    """Normalize free-form text for stable cache keys.

    Args:
        value: Any value that should be represented as cache-key text.

    Returns:
        A stripped, whitespace-collapsed, case-folded string.
    """
    return _WHITESPACE_RE.sub(" ", str(value or "").strip()).casefold()


def _context_scope(context: dict[str, Any]) -> dict[str, str]:
    """Extract platform/channel scope from an agent context.

    Args:
        context: Runtime hints passed to a helper agent.

    Returns:
        Dict containing normalized platform and channel scope fields.
    """
    platform = context.get("platform") or context.get("target_platform") or ""
    channel = (
        context.get("platform_channel_id")
        or context.get("target_platform_channel_id")
        or ""
    )
    return {
        "platform": normalize_cache_text(platform),
        "platform_channel_id": normalize_cache_text(channel),
    }


def build_user_lookup_cache_key(display_name: str, context: dict[str, Any]) -> str:
    """Build the exact cache key for direct user-profile lookup results.

    Args:
        display_name: Display name extracted from the user-lookup task.
        context: Runtime hints passed to ``user_lookup_agent``.

    Returns:
        Stable exact-match cache key.
    """
    scope = _context_scope(context)
    return stable_cache_key(
        USER_LOOKUP_CACHE_NAME,
        {
            "policy_version": USER_LOOKUP_POLICY_VERSION,
            "lookup_mode": "profile",
            "display_name": normalize_cache_text(display_name),
            "platform": scope["platform"],
            "platform_channel_id": scope["platform_channel_id"],
        },
    )


def build_user_lookup_dependencies(
    display_name: str,
    context: dict[str, Any],
) -> list[CacheDependency]:
    """Build invalidation dependencies for a cached user lookup.

    Args:
        display_name: Display name extracted from the user-lookup task.
        context: Runtime hints passed to ``user_lookup_agent``.

    Returns:
        Dependencies affected by user-profile write/rename events.
    """
    scope = _context_scope(context)
    return [
        CacheDependency(
            source="user_profile",
            platform=scope["platform"],
            platform_channel_id=scope["platform_channel_id"],
            display_name=normalize_cache_text(display_name),
        )
    ]
