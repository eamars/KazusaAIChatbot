"""HTTP client for existing Kazusa brain-service endpoints."""

from __future__ import annotations

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

COGNITION_GRAPH_DETAIL_KEYS = (
    "summary",
    "reasoning",
    "internal_monologue",
    "logical_stance",
    "character_intent",
    "judgment_note",
    "decision",
    "status",
)
COGNITION_GRAPH_RAW_KEYS = ("cognition_graph", "cognition_snapshot")
CognitionGraphSource = Literal["overview_latest", "debug_latest", "historical"]


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

    raw_graph = _first_graph_payload(payload)
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
        "edges": _project_graph_edges(raw_graph.get("edges")),
        "redaction": {
            "detail": "sensitive keys and unbounded text redacted",
            "excluded": [
                "prompts",
                "embeddings",
                "raw messages",
                "message envelopes",
            ],
        },
    }
    try:
        snapshot = CognitionRunGraphSnapshot.model_validate(normalized)
    except ValidationError:
        snapshot = not_reported_cognition_graph(
            source=source,
            run_id=inferred_run_id,
            reason="brain cognition graph telemetry failed console validation",
        )
    return snapshot


def _first_graph_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first raw graph-like payload if one is present."""

    for key in COGNITION_GRAPH_RAW_KEYS:
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


def _project_graph_edges(raw_edges: Any) -> list[dict[str, Any]]:
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
        try:
            edge = CognitionRunGraphEdge.model_validate(projected_edge)
        except ValidationError:
            continue
        edges.append(edge.model_dump(mode="json"))
    return edges


def _project_node_detail(raw_detail: Any) -> dict[str, Any]:
    """Keep only small, redacted reasoning details for hover disclosure."""

    if not isinstance(raw_detail, dict):
        return {}

    allowed_detail = {
        key: raw_detail[key]
        for key in COGNITION_GRAPH_DETAIL_KEYS
        if key in raw_detail
    }
    redacted_detail = redact_mapping(allowed_detail)
    return redacted_detail


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
            "detail": {"summary": "Visible response returned by brain."},
        })
    for source_node, target_node in zip(nodes, nodes[1:]):
        edges.append({
            "source": source_node["id"],
            "target": target_node["id"],
            "kind": "sequence",
            "label": "",
        })

    if not nodes:
        return not_reported_cognition_graph(source=source, run_id=run_id)

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
