"""Adapter from RAG3 subagent IO to existing source-owned evidence agents."""

from __future__ import annotations

import logging
from typing import Mapping

from kazusa_ai_chatbot.rag.conversation_evidence import ConversationEvidenceAgent
from kazusa_ai_chatbot.rag.memory_evidence import MemoryEvidenceAgent
from kazusa_ai_chatbot.rag.person_context import PersonContextAgent
from kazusa_ai_chatbot.rag.recall import RecallAgent
from kazusa_ai_chatbot.utils import text_or_empty

from ..contracts import (
    LOCAL_CONTEXT_ARTIFACT_VERSION,
    LOCAL_CONTEXT_SUBAGENT_RESULT_VERSION,
    LocalContextSubagentRequestV1,
    LocalContextSubagentResultV1,
)

logger = logging.getLogger(__name__)


class SourceEvidenceSubagent:
    """Run one existing source owner behind the canonical RAG3 envelope."""

    def __init__(
        self,
        *,
        subagent: str,
        node_kinds: tuple[str, ...],
        agent_name: str | None = None,
    ) -> None:
        """Bind fixed source ownership metadata for one registered subagent."""

        self._subagent = subagent
        self._node_kinds = node_kinds
        self._agent_name = agent_name

    async def run(
        self,
        task: LocalContextSubagentRequestV1,
        context: dict[str, object],
        max_attempts: int = 1,
    ) -> LocalContextSubagentResultV1:
        """Return source evidence, artifacts, and a deterministic node update."""

        if task["subagent"] != self._subagent:
            raise ValueError("subagent: unexpected owner")
        if self._agent_name is None:
            return _unavailable(task, "no deterministic source helper applies")
        node_kind = _task_node_kind(task, self._node_kinds)
        source_context = _source_context(task, context)
        source_task = _source_task(task["objective"], node_kind, self._agent_name)
        try:
            source_result = await _run_source_agent(
                self._agent_name,
                source_task,
                source_context,
                max_attempts,
            )
        except Exception as exc:
            logger.exception("RAG3 source subagent failed: %s", self._subagent)
            return _failure(task, self._agent_name, str(exc))
        return _resolved_result(task, node_kind, self._agent_name, source_result)


def _task_node_kind(
    task: LocalContextSubagentRequestV1,
    node_kinds: tuple[str, ...],
) -> str:
    """Return the dispatcher-owned node kind after validating registration."""

    node_kind = task["payload"].get("node_kind")
    if not isinstance(node_kind, str) or node_kind not in node_kinds:
        raise ValueError("payload.node_kind: unexpected registered node kind")
    result = node_kind
    return result


def _source_context(
    task: LocalContextSubagentRequestV1,
    context: dict[str, object],
) -> dict[str, object]:
    """Build the existing helper context from trusted resolver data."""

    dependencies = task["payload"].get("dependency_context")
    known_facts: list[dict[str, object]] = []
    if isinstance(dependencies, list):
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                continue
            known_facts.append({
                "summary": dependency.get("knowledge_we_know_so_far", []),
                "raw_result": {"resolved_refs": []},
            })
    result = {
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
        "active_turn_platform_message_ids": list(
            context.get("active_turn_platform_message_ids", [])
        ),
        "active_turn_conversation_row_ids": list(
            context.get("active_turn_conversation_row_ids", [])
        ),
        "local_time_context": dict(context["local_time_context"]),
        "conversation_progress": dict(context["conversation_progress"]),
        "chat_history_recent": list(context["chat_history_recent"]),
        "chat_history_wide": list(context["chat_history_wide"]),
        "original_query": text_or_empty(context.get("original_user_request")),
        "current_slot": task["objective"],
        "known_facts": known_facts,
        "character_profile": {"name": context["character_name"]},
    }
    return result


def _source_task(objective: str, node_kind: str, agent_name: str) -> str:
    """Build the source-owned task string without exposing graph internals."""

    prefixes = {
        "conversation_evidence_agent": "Conversation-evidence",
        "memory_evidence_agent": "Memory-evidence",
        "person_context_agent": "Person-context",
        "recall_agent": "Recall",
    }
    if node_kind == "scoped_memory":
        objective = f"current user's private continuity evidence for: {objective}"
    result = f"{prefixes[agent_name]}: {objective}"
    return result


async def _run_source_agent(
    agent_name: str,
    task: str,
    context: dict[str, object],
    max_attempts: int,
) -> dict[str, object]:
    """Call the source-owned specialist using its existing public method."""

    agents = {
        "conversation_evidence_agent": ConversationEvidenceAgent,
        "memory_evidence_agent": MemoryEvidenceAgent,
        "person_context_agent": PersonContextAgent,
        "recall_agent": RecallAgent,
    }
    result = await agents[agent_name]().run(task, context, max_attempts=max_attempts)
    if not isinstance(result, dict):
        raise ValueError("source agent result: expected object")
    return result


def _resolved_result(
    task: LocalContextSubagentRequestV1,
    node_kind: str,
    agent_name: str,
    source_result: dict[str, object],
) -> LocalContextSubagentResultV1:
    """Project source helper output into retained RAG3 artifact shapes."""

    payload = source_result.get("result")
    if not isinstance(payload, dict):
        payload = {}
    projection_payload = _projection_payload(agent_name, payload)
    resolved = source_result.get("resolved") is True and bool(projection_payload)
    artifacts: list[dict[str, object]] = []
    if resolved:
        artifacts.append({
            "schema_version": LOCAL_CONTEXT_ARTIFACT_VERSION,
            "artifact_id": f"{task['node_id']}_{agent_name}",
            "artifact_type": _artifact_type(agent_name),
            "producer_node_id": task["node_id"],
            "summary": _summary(payload, projection_payload),
            "projection_payload": projection_payload,
            "source_policy": _source_policy(payload, agent_name),
            "prompt_visible": True,
        })
    source_record = {
        "agent": agent_name,
        "node_kind": node_kind,
        "resolved": resolved,
        "selected_summary": text_or_empty(payload.get("selected_summary")),
        "source_policy": _source_policy(payload, agent_name),
        "evidence": _string_list(payload.get("evidence")),
        "missing_context": _string_list(payload.get("missing_context")),
        "projection_payload": projection_payload,
    }
    node_update: dict[str, object] = {}
    if artifacts:
        node_update = {
            "status": "resolved",
            "investigation_summary": [f"{agent_name} checked {task['objective']}."],
            "knowledge_we_know_so_far": [_summary(payload, projection_payload)],
            "knowledge_still_lacking": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [_source_policy(payload, agent_name)],
            "produces": [artifact["artifact_id"] for artifact in artifacts],
        }
    result: LocalContextSubagentResultV1 = {
        "schema_version": LOCAL_CONTEXT_SUBAGENT_RESULT_VERSION,
        "resolved": bool(artifacts),
        "status": "resolved" if artifacts else "partial",
        "result": {
            "source_records": [source_record],
            "artifacts": artifacts,
            "node_update": node_update,
        },
        "attempts": int(source_result.get("attempts", 0)),
        "cache": _source_cache(source_result),
        "trace": {"source_agent": agent_name},
        "unresolved_items": _string_list(payload.get("missing_context")),
    }
    return result


def _projection_payload(
    agent_name: str,
    payload: dict[str, object],
) -> dict[str, object]:
    """Retain the prior source projection fields without source internals."""

    raw = payload.get("projection_payload")
    projection = dict(raw) if isinstance(raw, Mapping) else {}
    if agent_name == "memory_evidence_agent":
        rows = projection.get("memory_rows")
        return {"memory_evidence": _selected_rows(
            rows,
            (
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
            ),
        )}
    if agent_name == "conversation_evidence_agent":
        rows = (
            projection.get("packets")
            or projection.get("rows")
            or projection.get("summaries")
        )
        return {"conversation_evidence": _selected_rows(
            rows,
            (
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
            ),
        )}
    if agent_name == "person_context_agent":
        profile = projection.get("profile")
        if not isinstance(profile, dict):
            profile = {"summary": text_or_empty(projection.get("summary"))}
        profile_kind = text_or_empty(projection.get("profile_kind"))
        if profile_kind == "current_user":
            return {"user_image": profile}
        if profile_kind == "active_character":
            return {"character_image": profile}
        return {"third_party_profiles": [profile] if profile else []}
    summary = text_or_empty(payload.get("selected_summary"))
    return {"recall_evidence": [{"summary": summary}] if summary else []}


def _artifact_type(agent_name: str) -> str:
    """Map each established source owner to the retained artifact type."""

    result = {
        "conversation_evidence_agent": "conversation_ref",
        "memory_evidence_agent": "memory_ref",
        "person_context_agent": "person_ref",
        "recall_agent": "recall_ref",
    }[agent_name]
    return result


def _source_cache(source_result: dict[str, object]) -> dict[str, object]:
    """Return source cache metadata only when the helper returned a mapping."""

    cache = source_result.get("cache")
    if not isinstance(cache, dict):
        return {}
    result = cache
    return result


def _selected_rows(
    value: object,
    field_names: tuple[str, ...],
) -> list[dict[str, object]]:
    """Project source rows through the established prompt-safe field allowlist."""

    if not isinstance(value, list):
        return []
    rows: list[dict[str, object]] = []
    for raw_row in value:
        if not isinstance(raw_row, Mapping):
            continue
        row = {
            field_name: raw_row[field_name]
            for field_name in field_names
            if field_name in raw_row and raw_row[field_name] not in (None, "", [], {})
        }
        if row:
            rows.append(row)
    return rows


def _summary(payload: dict[str, object], projection_payload: dict[str, object]) -> str:
    """Return one compact source summary for graph and prompt evidence."""

    summary = text_or_empty(payload.get("selected_summary"))
    if summary:
        return summary
    for rows in projection_payload.values():
        if isinstance(rows, list) and rows:
            return str(rows[0])
        if isinstance(rows, dict) and rows:
            return str(rows)
    return "Source evidence was retrieved."


def _source_policy(payload: dict[str, object], agent_name: str) -> str:
    """Return the helper's declared evidence boundary or its stable owner name."""

    for field_name in ("source_policy", "freshness_basis", "primary_worker"):
        value = text_or_empty(payload.get(field_name))
        if value:
            return value
    return agent_name


def _string_list(value: object) -> list[str]:
    """Return non-empty strings from a list-like source result field."""

    if not isinstance(value, list):
        return []
    result = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return result


def _unavailable(
    task: LocalContextSubagentRequestV1,
    reason: str,
) -> LocalContextSubagentResultV1:
    """Return explicit no-source evidence for semantic-only node owners."""

    return _result(task, "unavailable", reason)


def _failure(
    task: LocalContextSubagentRequestV1,
    agent_name: str,
    reason: str,
) -> LocalContextSubagentResultV1:
    """Return a bounded source failure without backend details in the prompt."""

    del agent_name
    return _result(task, "failed", "source evidence was unavailable")


def _result(
    task: LocalContextSubagentRequestV1,
    status: str,
    reason: str,
) -> LocalContextSubagentResultV1:
    """Build one no-artifact subagent result envelope."""

    result: LocalContextSubagentResultV1 = {
        "schema_version": LOCAL_CONTEXT_SUBAGENT_RESULT_VERSION,
        "resolved": False,
        "status": status,
        "result": {
            "source_records": [],
            "artifacts": [],
            "node_update": {"evidence_boundary_notes": [reason]},
        },
        "attempts": 0,
        "cache": {"enabled": False},
        "trace": {"node_id": task["node_id"]},
        "unresolved_items": [reason],
    }
    return result
