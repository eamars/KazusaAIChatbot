"""HTTP client for existing Kazusa brain-service endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any
import uuid

import httpx

from control_console.contracts import ConsoleDebugChatRequest
from control_console.redaction import redact_mapping, redact_value
from kazusa_ai_chatbot.time_boundary import build_turn_clock


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
