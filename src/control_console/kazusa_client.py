"""Validation-only HTTP client for existing Kazusa brain endpoints."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from control_console.contracts import ConsoleDebugChatRequest
from control_console.redaction import redact_mapping, redact_value
from kazusa_ai_chatbot.brain_service.cognition_observation_contracts import (
    CognitionRunObservationV1,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock


class CognitionObservationProtocolError(ValueError):
    """Safe protocol error for an invalid Brain observation payload."""

    def __init__(self, _: str = "observation_contract_invalid") -> None:
        """Expose only the stable public protocol error code."""

        super().__init__("observation_contract_invalid")


@dataclass(frozen=True)
class KazusaDebugChatResult:
    """Safe debug response metadata and its direct Brain observation."""

    response_payload: dict[str, Any]
    cognition_observation: CognitionRunObservationV1 | None


class KazusaClient:
    """Bounded HTTP client for the Brain service."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        control_shared_secret: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create a client for one Brain base URL."""

        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._control_shared_secret = control_shared_secret.strip()
        self._transport = transport

    async def get_health(self) -> dict[str, Any]:
        """Read the Brain `/health` endpoint."""

        async with self._client() as client:
            response = await client.get("/health")
        response.raise_for_status()
        payload = response.json()
        return payload

    async def get_runtime_status(self) -> dict[str, Any]:
        """Read the Brain runtime status endpoint."""

        async with self._client() as client:
            response = await client.get("/ops/runtime-status")
        response.raise_for_status()
        payload = response.json()
        return payload

    async def get_latest_cognition_graph(
        self,
    ) -> CognitionRunObservationV1 | None:
        """Read and validate the latest live-turn observation."""

        async with self._client() as client:
            response = await client.get("/ops/latest-cognition-graph")
        response.raise_for_status()
        payload = response.json()
        raw_observation = _observation_field(
            payload,
            field_name="cognition_graph",
        )
        return _validate_observation(
            raw_observation,
            expected_run_kind="live_turn",
        )

    async def get_latest_self_cognition_graph(
        self,
    ) -> CognitionRunObservationV1 | None:
        """Read and validate the latest self-cognition observation."""

        async with self._client() as client:
            response = await client.get("/ops/latest-cognition-graph")
        response.raise_for_status()
        payload = response.json()
        raw_observation = _observation_field(
            payload,
            field_name="self_cognition_graph",
        )
        return _validate_observation(
            raw_observation,
            expected_run_kind="self_cognition",
        )

    async def send_debug_chat(
        self,
        request: ConsoleDebugChatRequest,
    ) -> KazusaDebugChatResult:
        """Send one debug chat and validate its direct observation only."""

        payload = _debug_chat_payload(request)
        headers = _debug_chat_headers(self._control_shared_secret)
        async with self._client() as client:
            response = await client.post("/chat", json=payload, headers=headers)
        response.raise_for_status()
        response_payload = response.json()
        if not isinstance(response_payload, dict):
            raise CognitionObservationProtocolError()
        raw_observation = _observation_field(
            response_payload,
            field_name="cognition_graph",
        )
        observation = _validate_observation(
            raw_observation,
            expected_run_kind="live_turn",
        )
        return KazusaDebugChatResult(
            response_payload=_project_debug_chat_response(response_payload),
            cognition_observation=observation,
        )

    def _client(self) -> httpx.AsyncClient:
        """Create one `httpx.AsyncClient` instance."""

        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        )


def _observation_field(
    payload: Any,
    *,
    field_name: str,
) -> Any:
    """Read one exact observation field from a Brain response envelope."""

    if not isinstance(payload, dict):
        raise CognitionObservationProtocolError()
    if field_name not in payload:
        raise CognitionObservationProtocolError()
    return payload[field_name]


def _validate_observation(
    raw_observation: Any,
    *,
    expected_run_kind: str,
) -> CognitionRunObservationV1 | None:
    """Validate one non-null Brain observation without reconstruction."""

    if raw_observation is None:
        return None
    if not isinstance(raw_observation, dict):
        raise CognitionObservationProtocolError()
    try:
        observation = CognitionRunObservationV1.model_validate(raw_observation)
    except ValidationError as exc:
        raise CognitionObservationProtocolError() from exc
    if observation.run_kind != expected_run_kind:
        raise CognitionObservationProtocolError()
    return observation


def _debug_chat_payload(request: ConsoleDebugChatRequest) -> dict[str, Any]:
    """Build a Brain `ChatRequest` payload for debug-console input."""

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
    return {
        "platform": "debug",
        "platform_channel_id": request.channel_id,
        "channel_type": "private",
        "platform_message_id": f"debug-{uuid.uuid4().hex}",
        "platform_user_id": request.user_id,
        "platform_bot_id": "debug-bot-001",
        "display_name": request.user_display_name,
        "channel_name": request.channel_id,
        "content_type": "text",
        "message_envelope": envelope,
        "local_timestamp": turn_clock["local_timestamp"],
        "debug_modes": debug_modes,
    }


def _debug_chat_headers(shared_secret: str) -> dict[str, str]:
    """Build the Brain authorization headers for Debug Chat only."""

    clean_secret = shared_secret.strip()
    if not clean_secret:
        return {}
    return {
        "X-Kazusa-Control-Console": "debug-v1",
        "X-Kazusa-Control-Console-Auth": clean_secret,
    }


def _project_debug_chat_response(
    response_payload: dict[str, Any],
) -> dict[str, Any]:
    """Project safe non-cognition debug response metadata."""

    raw_messages = response_payload.get("messages", [])
    if not isinstance(raw_messages, list):
        raw_messages = []
    safe_messages = redact_value(raw_messages)
    if not isinstance(safe_messages, list):
        safe_messages = []

    raw_attachments = response_payload.get("attachments", [])
    attachment_count = (
        len(raw_attachments) if isinstance(raw_attachments, list) else 0
    )
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
        "trace_id",
    ):
        if key in response_payload:
            projected_response[key] = redact_value(response_payload[key])
    return redact_mapping(projected_response)
