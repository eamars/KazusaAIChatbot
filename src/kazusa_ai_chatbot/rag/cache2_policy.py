"""Per-agent cache policies for RAG Cache 2."""

from __future__ import annotations

import re
from typing import Any

from kazusa_ai_chatbot.rag.cache2_events import CacheDependency
from kazusa_ai_chatbot.rag.cache2_runtime import stable_cache_key

# ---------------------------------------------------------------------------
# Cache name and version constants
# ---------------------------------------------------------------------------

INITIALIZER_CACHE_NAME = "rag2_initializer"
INITIALIZER_POLICY_VERSION = "initializer:v1"
INITIALIZER_PROMPT_VERSION = "initializer_prompt:v1"
INITIALIZER_AGENT_REGISTRY_VERSION = "rag_supervisor2_registry:v1"
INITIALIZER_STRATEGY_SCHEMA_VERSION = "initializer_strategy_schema:v1"

USER_LOOKUP_CACHE_NAME = "rag2_user_lookup_agent"
USER_LOOKUP_POLICY_VERSION = "user_lookup:v2"

USER_LIST_CACHE_NAME = "rag2_user_list_agent"
USER_LIST_POLICY_VERSION = "user_list:v1"

USER_PROFILE_CACHE_NAME = "rag2_user_profile_agent"
USER_PROFILE_POLICY_VERSION = "user_profile:v1"

CONVERSATION_FILTER_CACHE_NAME = "rag2_conversation_filter_agent"
CONVERSATION_FILTER_POLICY_VERSION = "conversation_filter:v1"

CONVERSATION_KEYWORD_CACHE_NAME = "rag2_conversation_keyword_agent"
CONVERSATION_KEYWORD_POLICY_VERSION = "conversation_keyword:v1"

CONVERSATION_SEARCH_CACHE_NAME = "rag2_conversation_search_agent"
CONVERSATION_SEARCH_POLICY_VERSION = "conversation_search:v1"

PERSISTENT_MEMORY_KEYWORD_CACHE_NAME = "rag2_persistent_memory_keyword_agent"
PERSISTENT_MEMORY_KEYWORD_POLICY_VERSION = "persistent_memory_keyword:v1"

PERSISTENT_MEMORY_SEARCH_CACHE_NAME = "rag2_persistent_memory_search_agent"
PERSISTENT_MEMORY_SEARCH_POLICY_VERSION = "persistent_memory_search:v1"

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

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


def _initializer_context_signature(context: dict[str, Any]) -> dict[str, str]:
    """Build the initializer's cache-relevant context signature.

    Args:
        context: Runtime context passed into the progressive RAG supervisor.

    Returns:
        Stable subset of context fields that can affect slot planning.
    """
    return {
        "platform": normalize_cache_text(context.get("platform", "")),
        "platform_channel_id": normalize_cache_text(
            context.get("platform_channel_id", "")
        ),
        "global_user_id": str(context.get("global_user_id", "")).strip(),
        "user_name": normalize_cache_text(context.get("user_name", "")),
    }


def build_initializer_cache_key(
    *,
    original_query: str,
    character_name: str,
    context: dict[str, Any],
) -> str:
    """Build the exact cache key for a RAG initializer strategy.

    Args:
        original_query: User's normalized/decontextualized question.
        character_name: Active character name used by the initializer prompt.
        context: Runtime context that can affect planning, such as current user
            identity for pronoun handling.

    Returns:
        Stable exact-match cache key for initializer strategy reuse.
    """
    return stable_cache_key(
        INITIALIZER_CACHE_NAME,
        {
            "policy_version": INITIALIZER_POLICY_VERSION,
            "initializer_prompt_version": INITIALIZER_PROMPT_VERSION,
            "agent_registry_version": INITIALIZER_AGENT_REGISTRY_VERSION,
            "strategy_schema_version": INITIALIZER_STRATEGY_SCHEMA_VERSION,
            "original_query": normalize_cache_text(original_query),
            "character_name": normalize_cache_text(character_name),
            "context_signature": _initializer_context_signature(context),
        },
    )


def is_closed_historical_range(args: dict[str, Any]) -> bool:
    """Return whether tool args describe a fully bounded historical time range.

    Args:
        args: Normalized tool arguments produced by a generator LLM.

    Returns:
        True when both ``from_timestamp`` and ``to_timestamp`` are present,
        meaning the query is anchored to a closed window safe for caching.
    """
    return bool(args.get("from_timestamp") and args.get("to_timestamp"))


# ---------------------------------------------------------------------------
# user_lookup_agent
# ---------------------------------------------------------------------------


def build_user_lookup_cache_key(display_name: str, context: dict[str, Any]) -> str:
    """Build the exact cache key for user-profile lookup results.

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
            "display_name": normalize_cache_text(display_name),
            "platform": scope["platform"],
            "platform_channel_id": scope["platform_channel_id"],
        },
    )


def build_user_lookup_dependencies(
    global_user_id: str,
    context: dict[str, Any],
) -> list[CacheDependency]:
    """Build invalidation dependencies for a cached user lookup.

    Args:
        global_user_id: Resolved user UUID from the lookup result.
        context: Runtime hints passed to ``user_lookup_agent``.

    Returns:
        Dependencies keyed by resolved UUID so the consolidator can invalidate
        correctly regardless of what display-name alias was queried.
    """
    scope = _context_scope(context)
    return [
        CacheDependency(
            source="user_profile",
            platform=scope["platform"],
            platform_channel_id=scope["platform_channel_id"],
            global_user_id=global_user_id.strip(),
        )
    ]


# ---------------------------------------------------------------------------
# user_list_agent
# ---------------------------------------------------------------------------


def build_user_list_cache_key(args: dict[str, Any], context: dict[str, Any]) -> str:
    """Build the exact cache key for a user-list enumeration.

    Args:
        args: Normalized extractor args (display_name_value, operator, source, limit).
        context: Runtime hints passed to ``user_list_agent``.

    Returns:
        Stable exact-match cache key.
    """
    scope = _context_scope(context)
    raw_limit = args.get("limit", 20)
    limit = int(raw_limit) if isinstance(raw_limit, int) and not isinstance(raw_limit, bool) else 20
    return stable_cache_key(
        USER_LIST_CACHE_NAME,
        {
            "policy_version": USER_LIST_POLICY_VERSION,
            "display_name_value": normalize_cache_text(args.get("display_name_value", "")),
            "operator": str(args.get("display_name_operator", "contains")),
            "source": str(args.get("source", "user_profiles")),
            "limit": limit,
            "platform": scope["platform"],
            "platform_channel_id": scope["platform_channel_id"],
        },
    )


def build_user_list_dependencies(
    args: dict[str, Any], context: dict[str, Any]
) -> list[CacheDependency]:
    """Build invalidation dependencies for a cached user-list result.

    Depends on ``"user_profile"`` events for profile-sourced results and on
    ``"conversation_history"`` events for participant-sourced results.

    Args:
        args: Normalized extractor args including the ``source`` field.
        context: Runtime hints passed to ``user_list_agent``.

    Returns:
        One or two dependencies matching the effective data source(s).
    """
    scope = _context_scope(context)
    source = str(args.get("source", "user_profiles"))
    deps: list[CacheDependency] = []
    if source in ("user_profiles", "both"):
        deps.append(
            CacheDependency(
                source="user_profile",
                platform=scope["platform"],
                platform_channel_id=scope["platform_channel_id"],
            )
        )
    if source in ("conversation_participants", "both"):
        deps.append(
            CacheDependency(
                source="conversation_history",
                platform=scope["platform"],
                platform_channel_id=scope["platform_channel_id"],
            )
        )
    if not deps:
        deps.append(
            CacheDependency(
                source="user_profile",
                platform=scope["platform"],
                platform_channel_id=scope["platform_channel_id"],
            )
        )
    return deps


# ---------------------------------------------------------------------------
# user_profile_agent
# ---------------------------------------------------------------------------


def build_user_profile_cache_key(global_user_id: str) -> str:
    """Build the exact cache key for a hydrated user-profile bundle.

    Args:
        global_user_id: Resolved user UUID used to fetch the profile.

    Returns:
        Stable exact-match cache key.
    """
    return stable_cache_key(
        USER_PROFILE_CACHE_NAME,
        {
            "policy_version": USER_PROFILE_POLICY_VERSION,
            "global_user_id": global_user_id.strip(),
        },
    )


def build_character_profile_cache_key(global_user_id: str) -> str:
    """Build the exact cache key for the character profile bundle.

    Args:
        global_user_id: Stable character UUID used by the user/profile lookup path.

    Returns:
        Stable exact-match cache key isolated from ordinary user-image bundles.
    """
    return stable_cache_key(
        USER_PROFILE_CACHE_NAME,
        {
            "policy_version": USER_PROFILE_POLICY_VERSION,
            "profile_source": "character_state",
            "global_user_id": global_user_id.strip(),
        },
    )


def build_user_profile_dependencies(global_user_id: str) -> list[CacheDependency]:
    """Build invalidation dependencies for a cached user-profile bundle.

    Args:
        global_user_id: Resolved user UUID whose profile data was fetched.

    Returns:
        Single dependency on user-profile writes for that UUID.
    """
    return [CacheDependency(source="user_profile", global_user_id=global_user_id.strip())]


# ---------------------------------------------------------------------------
# Conversation agents — shared helpers
# ---------------------------------------------------------------------------


def _build_conversation_cache_key(
    cache_name: str,
    policy_version: str,
    task: str,
    context: dict[str, Any],
) -> str:
    """Build a task-scoped cache key for any conversation retrieval agent.

    Args:
        cache_name: Agent-specific cache namespace constant.
        policy_version: Policy version string for this agent.
        task: Slot description produced by the outer-loop supervisor.
        context: Runtime hints containing platform/channel scope.

    Returns:
        Stable exact-match cache key.
    """
    scope = _context_scope(context)
    return stable_cache_key(
        cache_name,
        {
            "policy_version": policy_version,
            "task": normalize_cache_text(task),
            "platform": scope["platform"],
            "platform_channel_id": scope["platform_channel_id"],
        },
    )


def _build_conversation_history_dependencies(
    args: dict[str, Any],
    context: dict[str, Any],
) -> list[CacheDependency]:
    """Build conversation-history invalidation dependencies from winning tool args.

    Args:
        args: Winning generator args that produced the resolved result.
        context: Runtime hints supplying fallback scope fields.

    Returns:
        Single dependency with the effective scope and time range.
    """
    scope = _context_scope(context)
    return [
        CacheDependency(
            source="conversation_history",
            platform=str(args.get("platform", "")).strip() or scope["platform"],
            platform_channel_id=str(args.get("platform_channel_id", "")).strip()
            or scope["platform_channel_id"],
            global_user_id=str(args.get("global_user_id", "")).strip(),
            from_timestamp=str(args.get("from_timestamp", "")).strip(),
            to_timestamp=str(args.get("to_timestamp", "")).strip(),
        )
    ]


# ---------------------------------------------------------------------------
# conversation_filter_agent
# ---------------------------------------------------------------------------


def build_conversation_filter_cache_key(task: str, context: dict[str, Any]) -> str:
    """Build the cache key for a structured conversation filter query.

    Args:
        task: Slot description produced by the outer-loop supervisor.
        context: Runtime hints containing platform/channel scope.

    Returns:
        Stable exact-match cache key.
    """
    return _build_conversation_cache_key(
        CONVERSATION_FILTER_CACHE_NAME, CONVERSATION_FILTER_POLICY_VERSION, task, context
    )


def build_conversation_filter_dependencies(
    args: dict[str, Any], context: dict[str, Any]
) -> list[CacheDependency]:
    """Build invalidation dependencies for a cached conversation filter result.

    Args:
        args: Winning generator args that produced the resolved result.
        context: Runtime hints supplying fallback scope fields.

    Returns:
        Single conversation-history dependency with the effective scope and range.
    """
    return _build_conversation_history_dependencies(args, context)


# ---------------------------------------------------------------------------
# conversation_keyword_agent
# ---------------------------------------------------------------------------


def build_conversation_keyword_cache_key(task: str, context: dict[str, Any]) -> str:
    """Build the cache key for a keyword conversation search query.

    Args:
        task: Slot description produced by the outer-loop supervisor.
        context: Runtime hints containing platform/channel scope.

    Returns:
        Stable exact-match cache key.
    """
    return _build_conversation_cache_key(
        CONVERSATION_KEYWORD_CACHE_NAME, CONVERSATION_KEYWORD_POLICY_VERSION, task, context
    )


def build_conversation_keyword_dependencies(
    args: dict[str, Any], context: dict[str, Any]
) -> list[CacheDependency]:
    """Build invalidation dependencies for a cached keyword conversation result.

    Args:
        args: Winning generator args that produced the resolved result.
        context: Runtime hints supplying fallback scope fields.

    Returns:
        Single conversation-history dependency with the effective scope and range.
    """
    return _build_conversation_history_dependencies(args, context)


# ---------------------------------------------------------------------------
# conversation_search_agent
# ---------------------------------------------------------------------------


def build_conversation_search_cache_key(task: str, context: dict[str, Any]) -> str:
    """Build the cache key for a semantic conversation search query.

    Args:
        task: Slot description produced by the outer-loop supervisor.
        context: Runtime hints containing platform/channel scope.

    Returns:
        Stable exact-match cache key.
    """
    return _build_conversation_cache_key(
        CONVERSATION_SEARCH_CACHE_NAME, CONVERSATION_SEARCH_POLICY_VERSION, task, context
    )


def build_conversation_search_dependencies(
    args: dict[str, Any], context: dict[str, Any]
) -> list[CacheDependency]:
    """Build invalidation dependencies for a cached semantic conversation result.

    Args:
        args: Winning generator args that produced the resolved result.
        context: Runtime hints supplying fallback scope fields.

    Returns:
        Single conversation-history dependency with the effective scope and range.
    """
    return _build_conversation_history_dependencies(args, context)


# ---------------------------------------------------------------------------
# Persistent memory agents — shared helpers
# ---------------------------------------------------------------------------


def _build_persistent_memory_cache_key(
    cache_name: str,
    policy_version: str,
    task: str,
    context: dict[str, Any],
) -> str:
    """Build a task-scoped cache key for any persistent-memory retrieval agent.

    Args:
        cache_name: Agent-specific cache namespace constant.
        policy_version: Policy version string for this agent.
        task: Slot description produced by the outer-loop supervisor.
        context: Runtime hints containing platform/channel scope.

    Returns:
        Stable exact-match cache key.
    """
    scope = _context_scope(context)
    return stable_cache_key(
        cache_name,
        {
            "policy_version": policy_version,
            "task": normalize_cache_text(task),
            "platform": scope["platform"],
            "platform_channel_id": scope["platform_channel_id"],
        },
    )


def _build_persistent_memory_dependencies(args: dict[str, Any]) -> list[CacheDependency]:
    """Build user-profile invalidation dependencies from winning memory-search args.

    Uses ``source_global_user_id`` when present (narrows to one user's memories).
    An empty string acts as wildcard — any user-profile write invalidates the entry.

    Args:
        args: Winning generator args (after any subject-user adjustments).

    Returns:
        Single dependency on user-profile writes for the effective user scope.
    """
    global_user_id = str(args.get("source_global_user_id", "")).strip()
    return [CacheDependency(source="user_profile", global_user_id=global_user_id)]


# ---------------------------------------------------------------------------
# persistent_memory_keyword_agent
# ---------------------------------------------------------------------------


def build_persistent_memory_keyword_cache_key(task: str, context: dict[str, Any]) -> str:
    """Build the cache key for a persistent-memory keyword search.

    Args:
        task: Slot description produced by the outer-loop supervisor.
        context: Runtime hints containing platform/channel scope.

    Returns:
        Stable exact-match cache key.
    """
    return _build_persistent_memory_cache_key(
        PERSISTENT_MEMORY_KEYWORD_CACHE_NAME,
        PERSISTENT_MEMORY_KEYWORD_POLICY_VERSION,
        task,
        context,
    )


def build_persistent_memory_keyword_dependencies(args: dict[str, Any]) -> list[CacheDependency]:
    """Build invalidation dependencies for a cached persistent-memory keyword result.

    Args:
        args: Winning generator args from the resolved attempt.

    Returns:
        Single user-profile dependency scoped by the effective user filter.
    """
    return _build_persistent_memory_dependencies(args)


# ---------------------------------------------------------------------------
# persistent_memory_search_agent
# ---------------------------------------------------------------------------


def build_persistent_memory_search_cache_key(task: str, context: dict[str, Any]) -> str:
    """Build the cache key for a persistent-memory semantic search.

    Args:
        task: Slot description produced by the outer-loop supervisor.
        context: Runtime hints containing platform/channel scope.

    Returns:
        Stable exact-match cache key.
    """
    return _build_persistent_memory_cache_key(
        PERSISTENT_MEMORY_SEARCH_CACHE_NAME,
        PERSISTENT_MEMORY_SEARCH_POLICY_VERSION,
        task,
        context,
    )


def build_persistent_memory_search_dependencies(args: dict[str, Any]) -> list[CacheDependency]:
    """Build invalidation dependencies for a cached persistent-memory semantic result.

    Args:
        args: Winning generator args after any subject-user adjustments.

    Returns:
        Single user-profile dependency scoped by the effective user filter.
    """
    return _build_persistent_memory_dependencies(args)
