"""Source-backed evidence hydration for local-context active nodes."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from kazusa_ai_chatbot.rag.conversation_evidence import ConversationEvidenceAgent
from kazusa_ai_chatbot.rag.memory_evidence import MemoryEvidenceAgent
from kazusa_ai_chatbot.rag.person_context import PersonContextAgent
from kazusa_ai_chatbot.rag.recall import RecallAgent
from kazusa_ai_chatbot.utils import text_or_empty

from .contracts import (
    LOCAL_CONTEXT_ARTIFACT_VERSION,
    LocalContextArtifactV1,
    LocalContextNodeV1,
    LocalContextResolverContextV1,
)

logger = logging.getLogger(__name__)

_SOURCE_AGENT_BY_NODE_KIND = {
    "conversation_evidence": "conversation_evidence_agent",
    "memory_evidence": "memory_evidence_agent",
    "person_context": "person_context_agent",
    "recall_evidence": "recall_agent",
    "scoped_memory": "memory_evidence_agent",
}

_ARTIFACT_TYPE_BY_AGENT = {
    "conversation_evidence_agent": "conversation_ref",
    "memory_evidence_agent": "memory_ref",
    "person_context_agent": "person_ref",
    "recall_agent": "recall_ref",
}

_TASK_PREFIX_BY_AGENT = {
    "conversation_evidence_agent": "Conversation-evidence",
    "memory_evidence_agent": "Memory-evidence",
    "person_context_agent": "Person-context",
    "recall_agent": "Recall",
}

_MEMORY_ROW_FIELDS = (
    "content",
    "summary",
    "fact",
    "memory_name",
    "name",
    "source_system",
    "source_kind",
    "memory_type",
    "status",
    "authority",
    "truth_status",
    "origin",
    "scope_type",
)

_CONVERSATION_ROW_FIELDS = (
    "_id",
    "conversation_row_id",
    "speaker",
    "role",
    "text",
    "body_text",
    "summary",
    "quote",
    "url",
    "relation",
    "context",
    "source",
)


async def hydrate_source_for_node(
    *,
    active_node: LocalContextNodeV1,
    context: LocalContextResolverContextV1,
    dependency_context: list[dict[str, object]],
    max_attempts: int,
) -> dict[str, object]:
    """Run the source-owned specialist for one supported active node.

    Args:
        active_node: The graph node currently being resolved.
        context: Trusted local-context resolver context from the caller.
        dependency_context: Resolved dependency summaries already in the graph.
        max_attempts: Bounded attempt count forwarded to the source agent.

    Returns:
        A small hydration envelope containing prompt-safe source records,
        deterministic artifacts, and a deterministic node update. Unsupported
        or disabled nodes return an envelope with ``called`` set to ``False``.
    """

    agent_name = _SOURCE_AGENT_BY_NODE_KIND.get(active_node["node_kind"])
    if context.get("source_hydration_enabled") is not True or agent_name is None:
        result = _empty_hydration(called=False, agent_name=agent_name or "")
        return result

    source_task = _source_task(active_node, agent_name)
    source_context = _source_context(
        active_node=active_node,
        context=context,
        dependency_context=dependency_context,
    )
    try:
        source_result = await _run_source_agent(
            agent_name=agent_name,
            task=source_task,
            context=source_context,
            max_attempts=max_attempts,
        )
    except Exception as exc:
        logger.exception(
            f"Local-context source hydration failed: "
            f"agent={agent_name} node_kind={active_node['node_kind']} "
            f"error={exc}"
        )
        result = {
            "called": True,
            "agent": agent_name,
            "resolved": False,
            "source_records": [{
                "agent": agent_name,
                "resolved": False,
                "missing_context": ["source_agent_unavailable"],
                "source_policy": f"source hydration unavailable: {exc}",
            }],
            "artifacts": [],
            "node_update": {},
        }
        return result

    source_payload = _source_payload(source_result)
    source_record = _source_record(
        agent_name=agent_name,
        source_result=source_result,
        source_payload=source_payload,
        active_node=active_node,
    )
    artifacts = _artifacts_from_source_result(
        agent_name=agent_name,
        active_node=active_node,
        source_result=source_result,
        source_payload=source_payload,
    )
    node_update = _node_update_from_source(
        agent_name=agent_name,
        active_node=active_node,
        source_payload=source_payload,
        artifacts=artifacts,
        resolved=bool(source_result.get("resolved")),
    )
    result = {
        "called": True,
        "agent": agent_name,
        "resolved": bool(source_result.get("resolved")),
        "source_records": [source_record],
        "artifacts": artifacts,
        "node_update": node_update,
    }
    return result


def _empty_hydration(
    *,
    called: bool,
    agent_name: str,
) -> dict[str, object]:
    """Return an empty source-hydration envelope."""

    result = {
        "called": called,
        "agent": agent_name,
        "resolved": False,
        "source_records": [],
        "artifacts": [],
        "node_update": {},
    }
    return result


def _source_task(active_node: LocalContextNodeV1, agent_name: str) -> str:
    """Build the specialist task text from a semantic active-node objective."""

    prefix = _TASK_PREFIX_BY_AGENT[agent_name]
    objective = active_node["objective"]
    if active_node["node_kind"] == "scoped_memory":
        objective = f"current user's private continuity evidence for: {objective}"
    source_task = f"{prefix}: {objective}"
    return source_task


def _source_context(
    *,
    active_node: LocalContextNodeV1,
    context: LocalContextResolverContextV1,
    dependency_context: list[dict[str, object]],
) -> dict[str, Any]:
    """Build trusted specialist-agent context without exposing it to prompts."""

    source_context: dict[str, Any] = {
        "platform": context["platform"],
        "platform_channel_id": context["platform_channel_id"],
        "global_user_id": context["global_user_id"],
        "user_name": context["user_name"],
        "display_name": context["user_name"],
        "current_timestamp_utc": text_or_empty(
            context.get("current_timestamp_utc")
        ),
        "current_platform_message_id": text_or_empty(
            context.get("current_platform_message_id")
        ),
        "active_turn_platform_message_ids": _string_list(
            context.get("active_turn_platform_message_ids")
        ),
        "active_turn_conversation_row_ids": _string_list(
            context.get("active_turn_conversation_row_ids")
        ),
        "local_time_context": dict(context["local_time_context"]),
        "conversation_progress": dict(context["conversation_progress"]),
        "chat_history_recent": list(context["chat_history_recent"]),
        "chat_history_wide": list(context["chat_history_wide"]),
        "original_query": text_or_empty(context.get("original_user_request")),
        "current_slot": active_node["objective"],
        "known_facts": _known_facts_from_dependencies(dependency_context),
        "character_profile": {"name": context["character_name"]},
    }
    return source_context


async def _run_source_agent(
    *,
    agent_name: str,
    task: str,
    context: dict[str, Any],
    max_attempts: int,
) -> dict[str, Any]:
    """Dispatch one task to the existing source-owned specialist agent."""

    if agent_name == "conversation_evidence_agent":
        result = await ConversationEvidenceAgent().run(
            task,
            context,
            max_attempts=max_attempts,
        )
        return result
    if agent_name == "memory_evidence_agent":
        result = await MemoryEvidenceAgent().run(
            task,
            context,
            max_attempts=max_attempts,
        )
        return result
    if agent_name == "person_context_agent":
        result = await PersonContextAgent().run(
            task,
            context,
            max_attempts=max_attempts,
        )
        return result
    if agent_name == "recall_agent":
        result = await RecallAgent().run(
            task,
            context,
            max_attempts=max_attempts,
        )
        return result
    raise ValueError(f"unsupported source hydration agent: {agent_name}")


def _source_payload(source_result: dict[str, Any]) -> dict[str, Any]:
    """Return the nested specialist result payload when present."""

    payload = source_result.get("result")
    if isinstance(payload, dict):
        return payload
    return_value: dict[str, Any] = {}
    return return_value


def _source_record(
    *,
    agent_name: str,
    source_result: dict[str, Any],
    source_payload: dict[str, Any],
    active_node: LocalContextNodeV1,
) -> dict[str, object]:
    """Build the compact model-facing source record for active-node review."""

    projection_payload = _projection_payload_for_agent(
        agent_name=agent_name,
        source_payload=source_payload,
    )
    source_record = {
        "agent": agent_name,
        "node_kind": active_node["node_kind"],
        "resolved": bool(source_result.get("resolved")),
        "selected_summary": text_or_empty(
            source_payload.get("selected_summary")
        ),
        "source_policy": _source_policy(agent_name, source_payload),
        "evidence": _string_list(source_payload.get("evidence")),
        "missing_context": _string_list(source_payload.get("missing_context")),
        "projection_payload": projection_payload,
    }
    return source_record


def _artifacts_from_source_result(
    *,
    agent_name: str,
    active_node: LocalContextNodeV1,
    source_result: dict[str, Any],
    source_payload: dict[str, Any],
) -> list[LocalContextArtifactV1]:
    """Project resolved specialist payloads into local-context artifacts."""

    if source_result.get("resolved") is not True:
        return []
    projection_payload = _projection_payload_for_agent(
        agent_name=agent_name,
        source_payload=source_payload,
    )
    if not _projection_has_evidence(projection_payload):
        return []

    artifact_id = f"{active_node['node_id']}_{agent_name}"
    summary = text_or_empty(source_payload.get("selected_summary"))
    if not summary:
        summary = _fallback_summary(projection_payload)
    artifact = {
        "schema_version": LOCAL_CONTEXT_ARTIFACT_VERSION,
        "artifact_id": artifact_id,
        "artifact_type": _ARTIFACT_TYPE_BY_AGENT[agent_name],
        "producer_node_id": active_node["node_id"],
        "summary": summary,
        "projection_payload": projection_payload,
        "source_policy": _source_policy(agent_name, source_payload),
        "prompt_visible": True,
    }
    artifacts = [artifact]
    return artifacts


def _projection_payload_for_agent(
    *,
    agent_name: str,
    source_payload: dict[str, Any],
) -> dict[str, object]:
    """Map one specialist payload into the retained RAG projection fields."""

    projection = source_payload.get("projection_payload")
    if not isinstance(projection, Mapping):
        projection = {}

    if agent_name == "memory_evidence_agent":
        memory_rows = _mapping_list(projection.get("memory_rows"))
        payload = {"memory_evidence": _project_memory_rows(memory_rows)}
        return payload

    if agent_name == "conversation_evidence_agent":
        packets = _mapping_list(projection.get("packets"))
        rows = _mapping_list(projection.get("rows"))
        summaries = _string_list(projection.get("summaries"))
        evidence_rows: list[object] = []
        if packets:
            evidence_rows = _project_conversation_rows(packets)
        elif rows:
            evidence_rows = _project_conversation_rows(rows)
        else:
            evidence_rows = summaries
        payload = {"conversation_evidence": evidence_rows}
        return payload

    if agent_name == "person_context_agent":
        payload = _person_projection_payload(projection)
        return payload

    if agent_name == "recall_agent":
        recall_row = _recall_projection_row(source_payload)
        payload = {"recall_evidence": [recall_row] if recall_row else []}
        return payload

    payload: dict[str, object] = {}
    return payload


def _person_projection_payload(projection: Mapping[str, object]) -> dict[str, object]:
    """Project person-context payload into profile-owned RAG fields."""

    profile_kind = text_or_empty(projection.get("profile_kind"))
    summary = text_or_empty(projection.get("summary"))
    profile = projection.get("profile")
    if isinstance(profile, Mapping):
        profile_payload: object = dict(profile)
    elif summary:
        profile_payload = {"summary": summary}
    else:
        profile_payload = {}

    if profile_kind == "current_user":
        payload = {"user_image": profile_payload}
        return payload
    if profile_kind == "active_character":
        payload = {"character_image": profile_payload}
        return payload
    if profile_payload:
        payload = {"third_party_profiles": [profile_payload]}
        return payload
    payload = {"third_party_profiles": []}
    return payload


def _project_memory_rows(rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Keep semantic memory fields and drop retrieval-ranking internals."""

    projected_rows = [
        _project_selected_fields(row, _MEMORY_ROW_FIELDS)
        for row in rows
    ]
    return projected_rows


def _project_conversation_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, object]]:
    """Keep semantic conversation fields and trace refs only."""

    projected_rows = [
        _project_selected_fields(row, _CONVERSATION_ROW_FIELDS)
        for row in rows
    ]
    return projected_rows


def _project_selected_fields(
    row: Mapping[str, object],
    field_names: tuple[str, ...],
) -> dict[str, object]:
    """Return selected non-empty fields from a source row."""

    projected: dict[str, object] = {}
    for field_name in field_names:
        value = row.get(field_name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, Mapping) and not value:
            continue
        projected[field_name] = value
    return projected


def _recall_projection_row(source_payload: dict[str, Any]) -> dict[str, object]:
    """Build a bounded recall-evidence row from Recall's result payload."""

    selected_summary = text_or_empty(source_payload.get("selected_summary"))
    if not selected_summary:
        return {}
    recall_row = {
        "summary": selected_summary,
        "recall_type": text_or_empty(source_payload.get("recall_type")),
        "primary_source": text_or_empty(source_payload.get("primary_source")),
        "supporting_sources": _string_list(
            source_payload.get("supporting_sources")
        ),
        "freshness_basis": text_or_empty(source_payload.get("freshness_basis")),
    }
    return recall_row


def _node_update_from_source(
    *,
    agent_name: str,
    active_node: LocalContextNodeV1,
    source_payload: dict[str, Any],
    artifacts: list[LocalContextArtifactV1],
    resolved: bool,
) -> dict[str, object]:
    """Build a deterministic node update from resolved source evidence."""

    if not resolved or not artifacts:
        return {}
    summary = text_or_empty(source_payload.get("selected_summary"))
    if not summary:
        summary = _fallback_summary(artifacts[0]["projection_payload"])
    source_policy = _source_policy(agent_name, source_payload)
    node_update = {
        "status": "resolved",
        "investigation_summary": [
            f"{agent_name} checked {active_node['objective']}.",
        ],
        "knowledge_we_know_so_far": [summary],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [source_policy],
        "produces": [artifact["artifact_id"] for artifact in artifacts],
    }
    return node_update


def _source_policy(agent_name: str, source_payload: dict[str, Any]) -> str:
    """Return a non-empty source policy label for one specialist result."""

    source_policy = text_or_empty(source_payload.get("source_policy"))
    if source_policy:
        return source_policy
    freshness_basis = text_or_empty(source_payload.get("freshness_basis"))
    if freshness_basis:
        return freshness_basis
    primary_worker = text_or_empty(source_payload.get("primary_worker"))
    if primary_worker:
        return primary_worker
    return_value = agent_name
    return return_value


def _known_facts_from_dependencies(
    dependency_context: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Convert resolved dependency summaries into specialist known facts."""

    known_facts: list[dict[str, object]] = []
    for dependency in dependency_context:
        if not isinstance(dependency, Mapping):
            continue
        fact = {
            "summary": list(_string_list(dependency.get("knowledge_we_know_so_far"))),
            "raw_result": {
                "resolved_refs": [],
            },
        }
        known_facts.append(fact)
    return known_facts


def _projection_has_evidence(projection_payload: dict[str, object]) -> bool:
    """Return whether a projected payload carries any visible evidence."""

    for value in projection_payload.values():
        if isinstance(value, list) and value:
            return True
        if isinstance(value, Mapping) and value:
            return True
    return False


def _fallback_summary(projection_payload: Mapping[str, object]) -> str:
    """Return a compact fallback summary from projected source evidence."""

    for value in projection_payload.values():
        if isinstance(value, list) and value:
            first_value = value[0]
            if isinstance(first_value, str):
                return first_value
            if isinstance(first_value, Mapping):
                for field_name in ("summary", "content", "text", "fact"):
                    summary = text_or_empty(first_value.get(field_name))
                    if summary:
                        return summary
        if isinstance(value, Mapping):
            for field_name in ("summary", "content", "text", "fact"):
                summary = text_or_empty(value.get(field_name))
                if summary:
                    return summary
    return_value = "Source evidence was retrieved."
    return return_value


def _mapping_list(value: object) -> list[dict[str, Any]]:
    """Return dictionary rows from a list-like source payload."""

    if not isinstance(value, list):
        rows: list[dict[str, Any]] = []
        return rows
    rows = [
        dict(item)
        for item in value
        if isinstance(item, Mapping)
    ]
    return rows


def _string_list(value: object) -> list[str]:
    """Return non-empty strings from a list-like value."""

    if not isinstance(value, list):
        rows: list[str] = []
        return rows
    rows = [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]
    return rows
