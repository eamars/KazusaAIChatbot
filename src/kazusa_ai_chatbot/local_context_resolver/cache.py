"""Cache2 policy helpers for the local-context resolver."""

from __future__ import annotations

from kazusa_ai_chatbot.rag.cache2_events import CacheDependency
from kazusa_ai_chatbot.rag.cache2_runtime import (
    normalize_cache_text,
    stable_cache_key,
)

RAG3_CACHE_POLICY_VERSION = "rag3_local_context_cache:v1"
RAG3_PLANNER_CACHE_NAME = "rag3_local_context_planner"
RAG3_ACTIVE_NODE_CACHE_NAME = "rag3_local_context_active_node"
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
LIVE_CONTEXT_NODE_TTL_SECONDS = SECONDS_PER_MINUTE
EXTERNAL_EVIDENCE_NODE_TTL_SECONDS = (
    HOURS_PER_DAY * MINUTES_PER_HOUR * SECONDS_PER_MINUTE
)


def build_planner_cache_key(
    *,
    request: dict[str, object],
    context: dict[str, object],
    options: dict[str, object],
    stage_identity: dict[str, object],
) -> str:
    """Build the aggressive planner key for semantic task decomposition."""

    payload = {
        "policy_version": RAG3_CACHE_POLICY_VERSION,
        "stage_identity": stage_identity,
        "objective": normalize_cache_text(request["objective"]),
        "source": str(request["source"]).strip(),
        "priority": str(request["priority"]).strip(),
        "scope": _scope_signature(context),
        "context_signature": _planner_context_signature(context),
        "limits": _limits_signature(options),
    }
    cache_key = stable_cache_key(RAG3_PLANNER_CACHE_NAME, payload)
    return cache_key


def build_active_node_cache_key(
    *,
    request: dict[str, object],
    context: dict[str, object],
    compact_context: dict[str, object],
    active_node: dict[str, object],
    dependency_context: list[dict[str, object]],
    options: dict[str, object],
    stage_identity: dict[str, object],
) -> str:
    """Build the exact active-node key for evidence-bearing node output."""

    payload = {
        "policy_version": RAG3_CACHE_POLICY_VERSION,
        "stage_identity": stage_identity,
        "request": {
            "objective": normalize_cache_text(request["objective"]),
            "source": str(request["source"]).strip(),
        },
        "scope": _scope_signature(context),
        "source_hydration_enabled": context.get("source_hydration_enabled") is True,
        "active_node": _node_signature(active_node),
        "compact_context_digest": _digest_payload(compact_context),
        "dependency_context_digest": _digest_payload(dependency_context),
        "limits": _limits_signature(options),
    }
    cache_key = stable_cache_key(RAG3_ACTIVE_NODE_CACHE_NAME, payload)
    return cache_key


def build_active_node_cache_dependencies(
    *,
    node_kind: str,
    context: dict[str, object],
) -> list[CacheDependency]:
    """Return Cache2 invalidation dependencies for one active node kind."""

    platform = str(context["platform"]).strip()
    platform_channel_id = str(context["platform_channel_id"]).strip()
    global_user_id = str(context["global_user_id"]).strip()

    dependencies: list[CacheDependency] = []
    if node_kind == "conversation_evidence":
        dependencies.append(CacheDependency(
            source="conversation_history",
            platform=platform,
            platform_channel_id=platform_channel_id,
        ))
    elif node_kind == "memory_evidence":
        dependencies.append(CacheDependency(source="memory"))
    elif node_kind == "scoped_memory":
        dependencies.append(CacheDependency(
            source="user_profile",
            global_user_id=global_user_id,
        ))
        dependencies.append(CacheDependency(
            source="memory",
            global_user_id=global_user_id,
        ))
    elif node_kind == "person_context":
        dependencies.append(CacheDependency(
            source="user_profile",
        ))
        dependencies.append(CacheDependency(source="character_state"))
    elif node_kind == "recall_evidence":
        dependencies.append(CacheDependency(
            source="conversation_history",
            platform=platform,
            platform_channel_id=platform_channel_id,
            global_user_id=global_user_id,
        ))
        dependencies.append(CacheDependency(
            source="user_profile",
            global_user_id=global_user_id,
        ))

    return dependencies


def active_node_cache_ttl_seconds(node_kind: str) -> int | None:
    """Return the time expiry policy for active-node cache entries."""

    if node_kind == "live_context":
        return LIVE_CONTEXT_NODE_TTL_SECONDS
    if node_kind == "external_evidence":
        return EXTERNAL_EVIDENCE_NODE_TTL_SECONDS
    return_value = None
    return return_value


def _scope_signature(context: dict[str, object]) -> dict[str, str]:
    """Return the scope fields that can affect planner or node behavior."""

    signature = {
        "platform": normalize_cache_text(context["platform"]),
        "platform_channel_id": str(context["platform_channel_id"]).strip(),
        "global_user_id": str(context["global_user_id"]).strip(),
        "user_name": normalize_cache_text(context["user_name"]),
    }
    return signature


def _planner_context_signature(context: dict[str, object]) -> dict[str, object]:
    """Return coarse prompt-safe context traits for planner reuse."""

    local_time_context = _dict_value(context, "local_time_context")
    prompt_message_context = _dict_value(context, "prompt_message_context")
    conversation_progress = _dict_value(context, "conversation_progress")
    chat_history_recent = _list_value(context, "chat_history_recent")
    chat_history_wide = _list_value(context, "chat_history_wide")
    signature = {
        "has_local_time_context": bool(local_time_context),
        "has_prompt_message_context": bool(prompt_message_context),
        "has_conversation_progress": bool(conversation_progress),
        "has_recent_history": bool(chat_history_recent),
        "has_wide_history": bool(chat_history_wide),
        "source_domains": _source_domains(
            chat_history_recent + chat_history_wide,
        ),
        "anchor_digest": _digest_payload(_semantic_anchor_payload(context)),
    }
    return signature


def _limits_signature(options: dict[str, object]) -> dict[str, int]:
    """Return resolver limits that affect stage output shape."""

    signature = {
        "max_iterations": int(options["max_iterations"]),
        "max_nodes": int(options["max_nodes"]),
        "max_depth": int(options["max_depth"]),
        "max_node_attempts": int(options["max_node_attempts"]),
        "max_subagent_attempts": int(options["max_subagent_attempts"]),
    }
    return signature


def _node_signature(active_node: dict[str, object]) -> dict[str, object]:
    """Return active-node fields that define one evidence task."""

    signature = {
        "node_kind": str(active_node["node_kind"]).strip(),
        "objective": normalize_cache_text(active_node["objective"]),
        "consumes": active_node["consumes"],
        "produces": active_node["produces"],
    }
    return signature


def _semantic_anchor_payload(context: dict[str, object]) -> dict[str, object]:
    """Return lightweight semantic anchors that influence decomposition."""

    prompt_message_context = _dict_value(context, "prompt_message_context")
    anchor_payload = {
        "prompt_message_context": prompt_message_context,
        "original_user_request": str(
            context.get("original_user_request", "")
        ).strip(),
    }
    return anchor_payload


def _source_domains(rows: list[object]) -> list[str]:
    """Return stable source-domain labels visible in supplied context rows."""

    domains: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for field_name in ("source", "source_system", "source_policy"):
            value = row.get(field_name)
            if not isinstance(value, str) or not value.strip():
                continue
            normalized_value = normalize_cache_text(value)
            if normalized_value not in domains:
                domains.append(normalized_value)
    domains.sort()
    return domains


def _dict_value(data: dict[str, object], field_name: str) -> dict[str, object]:
    """Return a dictionary field or an empty dictionary."""

    value = data.get(field_name)
    if isinstance(value, dict):
        return_value = value
        return return_value
    return_value = {}
    return return_value


def _list_value(data: dict[str, object], field_name: str) -> list[object]:
    """Return a list field or an empty list."""

    value = data.get(field_name)
    if isinstance(value, list):
        return_value = value
        return return_value
    return_value = []
    return return_value


def _digest_payload(value: object) -> str:
    """Return a stable digest for JSON-like cache-key material."""

    digest = stable_cache_key("rag3_local_context_digest", {"value": value})
    return digest
