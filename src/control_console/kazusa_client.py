"""HTTP client for existing Kazusa brain-service endpoints."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import time
from typing import Any, Literal
import uuid

import httpx
from pydantic import ValidationError

from control_console.contracts import (
    CognitionRunGraphEdge,
    CognitionRunGraphNode,
    CognitionRunGraphSnapshot,
    ConsoleDebugChatRequest,
)
from control_console.redaction import redact_mapping, redact_value
from kazusa_ai_chatbot.time_boundary import build_turn_clock

COGNITION_GRAPH_DETAIL_KEYS = frozenset(
    {
        "input",
        "reply_context",
        "decision",
        "reasoning",
        "internal_monologue",
        "logical_stance",
        "character_intent",
        "judgment_note",
        "retrieval_answer",
        "memory_evidence",
        "conversation_evidence",
        "external_evidence",
        "recall_evidence",
        "media_evidence",
        "user_continuity",
        "conversation_progress",
        "active_commitments",
        "selected_actions",
        "action_results",
        "action_continuation",
        "parallel_execution",
        "appraisal_results",
        "branch_results",
        "collapse",
        "selected_intention",
        "selected_bid_reason",
        "goal_resolution",
        "expression_policy",
        "affect_projection",
        "phase",
        "goal_kind",
        "selection",
        "intention",
        "desired_outcome",
        "concrete_detail",
        "reason",
        "private_monologue",
        "confidence",
        "expected_consequences",
        "facial_expression",
        "body_language",
        "gaze_direction",
        "visual_vibe",
        "messages",
        "empty_state",
        "failure",
        "failure_code",
    }
)
COGNITION_GRAPH_SCALAR_DETAIL_KEYS = frozenset(
    {
        "input",
        "decision",
        "reasoning",
        "internal_monologue",
        "logical_stance",
        "character_intent",
        "judgment_note",
        "retrieval_answer",
        "empty_state",
        "failure_code",
        "failure_stage",
        "safe_checkpoint",
        "attempt_count",
        "retryable",
        "selected_bid_reason",
        "goal_resolution",
        "phase",
        "goal_kind",
        "selection",
        "intention",
        "desired_outcome",
        "concrete_detail",
        "reason",
        "private_monologue",
        "confidence",
    }
)
COGNITION_GRAPH_TEXT_LIST_DETAIL_KEYS = frozenset(
    {
        "facial_expression",
        "body_language",
        "gaze_direction",
        "visual_vibe",
        "messages",
        "expected_consequences",
    }
)
COGNITION_GRAPH_MAPPING_DETAIL_KEYS = frozenset(
    {
        "reply_context",
        "user_continuity",
        "conversation_progress",
        "parallel_execution",
        "collapse",
        "selected_intention",
        "expression_policy",
        "failure",
    }
)
COGNITION_GRAPH_ROW_DETAIL_KEYS = frozenset(
    {
        "memory_evidence",
        "conversation_evidence",
        "external_evidence",
        "recall_evidence",
        "media_evidence",
        "active_commitments",
        "selected_actions",
        "action_results",
        "action_continuation",
        "appraisal_results",
        "branch_results",
        "affect_projection",
    }
)
COGNITION_GRAPH_NESTED_DETAIL_KEYS = frozenset(
    {
        "summary",
        "fact",
        "excerpt",
        "content",
        "title",
        "source",
        "source_kind",
        "relevance",
        "recency",
        "due_at",
        "due_state",
        "evidence_boundary_notes",
        "visual_observation",
        "description",
        "confidence",
        "role",
        "display_name",
        "media_kind",
        "summary_status",
        "reply_to_display_name",
        "reply_excerpt",
        "reply_attachments",
        "continuity",
        "current_thread",
        "current_blocker",
        "open_loops",
        "resolved_threads",
        "avoid_reopening",
        "overused_moves",
        "next_affordances",
        "progression_guidance",
        "current_goal",
        "progress_note",
        "goal",
        "next_step",
        "participant_context",
        "thread_reference_context",
        "group_scene_digest",
        "stable_patterns",
        "recent_shifts",
        "objective_facts",
        "milestones",
        "active_commitments",
        "kind",
        "cognition_mode",
        "urgency",
        "visibility",
        "deadline",
        "reason",
        "continuation",
        "action_kind",
        "status",
        "result_summary",
        "semantic_decision",
        "completed_at",
        "queue_state",
        "work_kind",
        "objective_summary",
        "accepted_task_state",
        "accepted_task_summary",
        "wait_guidance",
        "acknowledgement_constraint",
        "mode",
        "episode_type",
        "max_depth",
        "include_result_as",
        "next_topic",
        "condition",
        "text",
        "objective",
        "consolidation_called",
        "scheduled_event_count",
        "cache_evicted_count",
        "write_success",
        "question_kind",
        "semantic_question",
        "explanation",
        "propositions",
        "deltas",
        "proposition_kind",
        "semantic_value",
        "delta",
        "phase",
        "branch_index",
        "goal_kind",
        "selection",
        "intention",
        "desired_outcome",
        "concrete_detail",
        "private_monologue",
        "expected_consequences",
        "failure_code",
        "stage",
        "failure_stage",
        "safe_checkpoint",
        "retryable",
        "primary_branch_index",
        "supporting_branch_indices",
        "suppressed_branch_indices",
        "selection_reason",
        "selected_question_count",
        "dispatched_question_count",
        "selected_branch_count",
        "dispatched_branch_count",
        "completed_branch_count",
        "failed_branch_count",
        "maximum_concurrency",
        "overlap_ms",
        "dependency_wait_ms",
        "total_ms",
        "emotion",
        "intensity",
        "trend",
        "cause_summary",
        "route",
        "visibility",
        "emotional_tone",
        "directness",
    }
)
COGNITION_GRAPH_FORBIDDEN_DETAIL_KEYS = frozenset(
    {
        "id",
        "schema_version",
        "source_refs",
        "result_refs",
        "evidence_refs",
        "target",
        "params",
        "scope",
        "job_ref",
        "run_id",
    }
)
COGNITION_GRAPH_FORBIDDEN_DETAIL_PARTS = (
    "prompt",
    "raw",
    "embedding",
    "message_envelope",
    "handler",
    "attempt",
    "idempotency",
    "operational",
    "trace",
)
COGNITION_GRAPH_RAW_KEYS = (
    "cognition_graph",
    "cognition_snapshot",
    "self_cognition_graph",
)
CognitionGraphSource = Literal[
    "overview_latest",
    "debug_latest",
    "self_latest",
    "historical",
]


class KazusaClient:
    """Bounded HTTP client for the brain service."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create a client for one brain base URL."""

        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def get_health(self) -> dict[str, Any]:
        """Read the brain `/health` endpoint."""

        async with self._client() as client:
            response = await client.get("/health")
        response.raise_for_status()
        payload = response.json()
        return payload

    async def get_runtime_status(self) -> dict[str, Any]:
        """Read the brain runtime status endpoint."""

        async with self._client() as client:
            response = await client.get("/ops/runtime-status")
        response.raise_for_status()
        payload = response.json()
        return payload

    async def get_latest_cognition_graph(self) -> CognitionRunGraphSnapshot:
        """Read and project the brain latest cognition graph endpoint."""

        async with self._client() as client:
            response = await client.get("/ops/latest-cognition-graph")
        response.raise_for_status()
        payload = response.json()
        graph = project_cognition_graph_snapshot(
            source="overview_latest",
            payload=payload if isinstance(payload, dict) else {},
        )
        return graph

    async def get_latest_self_cognition_graph(self) -> CognitionRunGraphSnapshot:
        """Read and project the brain latest self-cognition graph endpoint."""

        async with self._client() as client:
            response = await client.get("/ops/latest-cognition-graph")
        response.raise_for_status()
        payload = response.json()
        graph = project_cognition_graph_snapshot(
            source="self_latest",
            payload=payload if isinstance(payload, dict) else {},
        )
        return graph

    async def send_debug_chat(
        self,
        request: ConsoleDebugChatRequest,
    ) -> dict[str, Any]:
        """Send a debug chat request through the existing `/chat` contract."""

        started_at = time.perf_counter()
        payload = _debug_chat_payload(request)
        async with self._client() as client:
            response = await client.post("/chat", json=payload)
        response.raise_for_status()
        response_payload = response.json()
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        result = {
            "request_id": f"cc-req-{uuid.uuid4().hex[:12]}",
            "brain_available": True,
            "request": redact_mapping(payload),
            "response": _project_debug_chat_response(response_payload),
            "tracking_id": response_payload.get("delivery_tracking_id"),
            "latency_ms": elapsed_ms,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
            "cognition_graph": project_cognition_graph_snapshot(
                source="debug_latest",
                payload=response_payload,
                run_id=_safe_optional_text(
                    response_payload.get("delivery_tracking_id"),
                ),
            ).model_dump(mode="json"),
        }
        return result

    def _client(self) -> httpx.AsyncClient:
        """Create one `httpx.AsyncClient` instance."""

        client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        )
        return client


def _debug_chat_payload(request: ConsoleDebugChatRequest) -> dict[str, Any]:
    """Build a brain `ChatRequest` payload for debug-console input."""

    debug_modes = {
        "listen_only": "listen_only" in request.debug_modes,
        "think_only": "think_only" in request.debug_modes,
        "no_remember": "no_remember" in request.debug_modes,
    }
    envelope = {
        "body_text": request.message_text,
        "raw_wire_text": request.message_text,
        "mentions": [],
        "reply": None,
        "attachments": [],
        "addressed_to_global_user_ids": [],
        "broadcast": False,
    }
    envelope.update(request.envelope_overrides)
    turn_clock = build_turn_clock()
    payload = {
        "platform": "debug",
        "platform_channel_id": request.channel_id,
        "channel_type": "private",
        "platform_message_id": f"debug-{uuid.uuid4().hex}",
        "platform_user_id": request.user_id,
        "platform_bot_id": "",
        "display_name": request.user_display_name,
        "channel_name": request.channel_id,
        "content_type": "text",
        "message_envelope": envelope,
        "local_timestamp": turn_clock["local_timestamp"],
        "debug_modes": debug_modes,
    }
    return payload


def _project_debug_chat_response(response_payload: dict[str, Any]) -> dict[str, Any]:
    """Project a brain chat response into a safe operator-debug summary."""

    raw_messages = response_payload.get("messages", [])
    if not isinstance(raw_messages, list):
        raw_messages = []
    safe_messages = redact_value(raw_messages)
    if not isinstance(safe_messages, list):
        safe_messages = []

    raw_attachments = response_payload.get("attachments", [])
    attachment_count = len(raw_attachments) if isinstance(raw_attachments, list) else 0
    raw_mentions = response_payload.get("delivery_mentions", [])
    mention_count = len(raw_mentions) if isinstance(raw_mentions, list) else 0

    projected_response: dict[str, Any] = {
        "messages": safe_messages,
        "attachment_count": attachment_count,
        "delivery_mention_count": mention_count,
    }
    for key in (
        "content_type",
        "use_reply_feature",
        "scheduled_followups",
        "delivery_tracking_id",
    ):
        if key in response_payload:
            projected_response[key] = redact_value(response_payload[key])
    return projected_response


def not_reported_cognition_graph(
    *,
    source: CognitionGraphSource,
    run_id: str | None = None,
    reason: str = "brain response did not report cognition graph telemetry",
) -> CognitionRunGraphSnapshot:
    """Return an explicit empty graph snapshot without dummy nodes."""

    snapshot = CognitionRunGraphSnapshot(
        source=source,
        status="not_reported",
        run_id=run_id,
        generated_at=datetime.now(timezone.utc),
        nodes=[],
        edges=[],
        redaction={
            "reason": reason,
            "excluded": [
                "prompts",
                "embeddings",
                "raw messages",
                "message envelopes",
            ],
        },
    )
    return snapshot


def project_cognition_graph_snapshot(
    *,
    source: CognitionGraphSource,
    payload: dict[str, Any],
    run_id: str | None = None,
) -> CognitionRunGraphSnapshot:
    """Project safe cognition graph telemetry from a brain/debug payload."""

    raw_graph = _first_graph_payload(payload, source=source)
    inferred_run_id = run_id or _safe_optional_text(payload.get("delivery_tracking_id"))
    if raw_graph is None:
        inferred_graph = _project_known_cognition_fields(
            payload=payload,
            source=source,
            run_id=inferred_run_id,
        )
        return inferred_graph

    normalized = {
        "source": source,
        "status": raw_graph.get("status", "partial"),
        "run_id": _safe_optional_text(raw_graph.get("run_id")) or inferred_run_id,
        "generated_at": datetime.now(timezone.utc),
        "nodes": _project_graph_nodes(raw_graph.get("nodes")),
        "edges": [],
        "redaction": {
            "detail": (
                "approved cognition semantic fields preserve full text; "
                "other fields remain excluded"
            ),
            "excluded": [
                "prompts",
                "embeddings",
                "raw messages",
                "message envelopes",
            ],
        },
    }
    normalized["edges"] = _project_graph_edges(
        raw_graph.get("edges"),
        node_ids={node["id"] for node in normalized["nodes"]},
    )
    try:
        snapshot = CognitionRunGraphSnapshot.model_validate(normalized)
    except ValidationError:
        snapshot = not_reported_cognition_graph(
            source=source,
            run_id=normalized["run_id"] or inferred_run_id,
            reason="brain cognition graph telemetry failed console validation",
        )
    return snapshot


def _first_graph_payload(
    payload: dict[str, Any],
    *,
    source: CognitionGraphSource,
) -> dict[str, Any] | None:
    """Return the first raw graph-like payload if one is present."""

    keys = (
        ("self_cognition_graph",)
        if source == "self_latest"
        else COGNITION_GRAPH_RAW_KEYS[:2]
    )
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _project_graph_nodes(raw_nodes: Any) -> list[dict[str, Any]]:
    """Project bounded graph nodes from external telemetry."""

    if not isinstance(raw_nodes, list):
        return []

    nodes: list[dict[str, Any]] = []
    for raw_node in raw_nodes[:64]:
        if not isinstance(raw_node, dict):
            continue
        projected_node = {
            "id": str(raw_node.get("id", "")),
            "label": str(raw_node.get("label", "")),
            "stage": str(raw_node.get("stage", "")),
            "lane": str(raw_node.get("lane", "")),
            "column": raw_node.get("column", 1),
            "branch": str(raw_node.get("branch", "")),
            "status": str(raw_node.get("status", "not_reported")),
            "detail": _project_node_detail(raw_node.get("detail", {})),
        }
        try:
            node = CognitionRunGraphNode.model_validate(projected_node)
        except ValidationError:
            continue
        nodes.append(node.model_dump(mode="json"))
    return nodes


def _project_graph_edges(
    raw_edges: Any,
    *,
    node_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Project bounded graph edges from external telemetry."""

    if not isinstance(raw_edges, list):
        return []

    edges: list[dict[str, Any]] = []
    for raw_edge in raw_edges[:96]:
        if not isinstance(raw_edge, dict):
            continue
        projected_edge = {
            "source": str(raw_edge.get("source", "")),
            "target": str(raw_edge.get("target", "")),
            "kind": str(raw_edge.get("kind", "sequence")),
            "label": str(raw_edge.get("label", "")),
        }
        if node_ids is not None and (
            projected_edge["source"] not in node_ids
            or projected_edge["target"] not in node_ids
        ):
            continue
        try:
            edge = CognitionRunGraphEdge.model_validate(projected_edge)
        except ValidationError:
            continue
        edges.append(edge.model_dump(mode="json"))
    return edges


def _project_node_detail(raw_detail: Any) -> dict[str, Any]:
    """Project approved cognition semantics without generic truncation."""

    if not isinstance(raw_detail, Mapping):
        return {}

    projected: dict[str, Any] = {}
    for raw_key, raw_value in raw_detail.items():
        if not isinstance(raw_key, str):
            continue
        if raw_key not in COGNITION_GRAPH_DETAIL_KEYS:
            continue
        if _cognition_graph_key_is_forbidden(raw_key):
            continue
        projected_value = _project_cognition_graph_detail_value(
            raw_key,
            raw_value,
        )
        if projected_value in (None, "", [], {}):
            continue
        projected[raw_key] = projected_value
    return projected


def _cognition_graph_key_is_forbidden(key: str) -> bool:
    """Return whether a nested graph key is sensitive or operational."""

    normalized = key.casefold().replace("-", "_")
    if normalized in COGNITION_GRAPH_FORBIDDEN_DETAIL_KEYS:
        return True
    if normalized.endswith("_id"):
        return True
    return any(
        part in normalized
        for part in COGNITION_GRAPH_FORBIDDEN_DETAIL_PARTS
    )


def _project_cognition_graph_detail_value(
    field_name: str,
    value: Any,
) -> Any:
    """Project one selected-detail field using its semantic shape."""

    if field_name in COGNITION_GRAPH_SCALAR_DETAIL_KEYS:
        return _project_cognition_graph_scalar(value)
    if field_name in COGNITION_GRAPH_TEXT_LIST_DETAIL_KEYS:
        return _project_cognition_graph_text_list(value)
    if field_name in COGNITION_GRAPH_MAPPING_DETAIL_KEYS:
        return _project_cognition_graph_mapping(value)
    if field_name in COGNITION_GRAPH_ROW_DETAIL_KEYS:
        return _project_cognition_graph_rows(value)
    return None


def _project_cognition_graph_scalar(value: Any) -> Any:
    """Preserve an approved scalar without character truncation."""

    if isinstance(value, str):
        return value if value.strip() else None
    return None


def _project_cognition_graph_text_list(value: Any) -> list[str]:
    """Preserve ordered complete text entries from an approved list."""

    if not isinstance(value, list):
        return []
    return [
        item
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _project_cognition_graph_mapping(value: Any) -> dict[str, Any]:
    """Project an approved nested mapping recursively."""

    if not isinstance(value, Mapping):
        return {}
    return _project_cognition_graph_nested(value)


def _project_cognition_graph_rows(value: Any) -> list[Any]:
    """Preserve ordered semantic rows without an item-count cap."""

    if not isinstance(value, list):
        return []
    projected_rows: list[Any] = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                projected_rows.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        projected_item = _project_cognition_graph_nested(item)
        if projected_item:
            projected_rows.append(projected_item)
    return projected_rows


def _project_cognition_graph_nested(value: Any) -> Any:
    """Project JSON-compatible semantic values with nested key filtering."""

    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                continue
            if raw_key not in COGNITION_GRAPH_NESTED_DETAIL_KEYS:
                continue
            if _cognition_graph_key_is_forbidden(raw_key):
                continue
            projected_value = _project_cognition_graph_nested(raw_value)
            if projected_value in (None, "", [], {}):
                continue
            projected[raw_key] = projected_value
        return projected
    if isinstance(value, list):
        projected_items: list[Any] = []
        for item in value:
            projected_item = _project_cognition_graph_nested(item)
            if projected_item in (None, "", [], {}):
                continue
            projected_items.append(projected_item)
        return projected_items
    return None


def _project_cognition_graph_message_fragments(
    payload: Mapping[str, Any],
) -> list[str]:
    """Project actual visible message fragments from a legacy payload."""

    for key in ("final_dialog", "visible_response", "messages"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return [value]
        if isinstance(value, list):
            fragments: list[str] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    fragments.append(item)
                    continue
                if isinstance(item, Mapping):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        fragments.append(text)
            if fragments:
                return fragments
    return []


def _project_known_cognition_fields(
    *,
    payload: dict[str, Any],
    source: CognitionGraphSource,
    run_id: str | None,
) -> CognitionRunGraphSnapshot:
    """Project explicit cognition fields if a debug payload exposes them."""

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    if _has_any(payload, ("internal_monologue", "logical_stance", "character_intent")):
        nodes.append({
            "id": "l2.reasoning",
            "label": "Reasoning",
            "stage": "L2",
            "lane": "cognition",
            "column": 1,
            "branch": "l2",
            "status": "completed",
            "detail": _project_node_detail(payload),
        })
    if _has_any(payload, ("action_specs", "decision", "scheduled_followups")):
        nodes.append({
            "id": "l2.decision",
            "label": "Decision",
            "stage": "L2",
            "lane": "decision",
            "column": 2,
            "branch": "action",
            "status": "completed",
            "detail": _project_node_detail(payload),
        })
    if _has_any(payload, ("messages", "final_dialog", "visible_response")):
        nodes.append({
            "id": "l3.surface",
            "label": "Visible response",
            "stage": "L3",
            "lane": "surface",
            "column": 3,
            "branch": "dialog",
            "status": "completed",
            "detail": {
                "messages": _project_cognition_graph_message_fragments(payload),
            },
        })
    for source_node, target_node in zip(nodes, nodes[1:]):
        edges.append({
            "source": source_node["id"],
            "target": target_node["id"],
            "kind": "sequence",
            "label": "",
        })

    if not nodes:
        snapshot = not_reported_cognition_graph(source=source, run_id=run_id)
        return snapshot

    snapshot = CognitionRunGraphSnapshot(
        source=source,
        status="partial",
        run_id=run_id,
        generated_at=datetime.now(timezone.utc),
        nodes=[
            CognitionRunGraphNode.model_validate(node)
            for node in nodes
        ],
        edges=[
            CognitionRunGraphEdge.model_validate(edge)
            for edge in edges
        ],
        redaction={
            "detail": "projected from explicit safe fields only",
            "excluded": [
                "prompts",
                "embeddings",
                "raw messages",
                "message envelopes",
            ],
        },
    )
    return snapshot


def _has_any(payload: dict[str, Any], keys: tuple[str, ...]) -> bool:
    """Return whether any named field is present and non-empty."""

    has_value = any(payload.get(key) not in (None, "", [], {}) for key in keys)
    return has_value


def _safe_optional_text(value: Any) -> str | None:
    """Return a bounded string value for ids or no value."""

    if value is None:
        return None
    text = str(value)
    if len(text) > 120:
        text = text[:120]
    return text
