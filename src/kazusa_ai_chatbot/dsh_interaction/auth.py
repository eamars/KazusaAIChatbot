"""Authenticated loopback transport helpers for Brain interactions."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import replace
from datetime import UTC, datetime
from typing import Protocol

from kazusa_ai_chatbot.dsh_interaction.contracts import (
    ACTIVE_INTERACTION_SECONDS,
    INTERACTION_TIMESTAMP_SKEW_SECONDS,
    DshBrainInteractionRequestV2,
)
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import canonical_json


class InteractionNonceReplayOwner(Protocol):
    """Durable owner interface for interaction nonce replay exclusion."""

    def consume(self, issuer: str, nonce: str) -> None:
        """Consume an issuer/nonce pair or raise a replay fault."""




def sign_request(
    request: DshBrainInteractionRequestV2,
    *,
    secret: bytes,
) -> DshBrainInteractionRequestV2:
    """Attach a canonical HMAC-SHA256 MAC to one request."""

    if not isinstance(secret, bytes) or not secret:
        raise ValueError("Brain interaction secret is required")
    mac = _mac(request.unsigned_dict(), secret)
    return replace(request, mac=mac)


def verify_request(
    request: DshBrainInteractionRequestV2,
    *,
    secret: bytes,
    replay_owner: InteractionNonceReplayOwner,
    now: str | None = None,
) -> DshBrainInteractionRequestV2:
    """Verify MAC, timestamp window, and nonce replay before cognition."""

    validate_request(request, secret=secret, now=now)
    replay_owner.consume(request.issuer, request.nonce)
    return request


def validate_request(
    request: DshBrainInteractionRequestV2,
    *,
    secret: bytes,
    now: str | None = None,
) -> None:
    """Validate MAC and lifetime without claiming the durable nonce."""

    if not isinstance(secret, bytes) or not secret:
        raise ValueError("Brain interaction secret is required")
    if not request.mac:
        raise ValueError("interaction MAC is required")
    expected = _mac(request.unsigned_dict(), secret)
    if not hmac.compare_digest(request.mac, expected):
        raise ValueError("interaction MAC validation failed")
    current = _parse_time(now) if now is not None else datetime.now(UTC)
    issued = _parse_time(request.issued_at)
    expires = _parse_time(request.expires_at)
    lifetime = (expires - issued).total_seconds()
    if lifetime <= 0 or lifetime > ACTIVE_INTERACTION_SECONDS:
        raise ValueError("interaction lifetime is invalid")
    if (issued - current).total_seconds() > INTERACTION_TIMESTAMP_SKEW_SECONDS:
        raise ValueError("interaction timestamp is outside the allowed skew")
    if current > expires:
        raise ValueError("interaction request has expired")


def _mac(value: object, secret: bytes) -> str:
    """Compute the canonical interaction MAC."""

    payload = canonical_json(value)
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def _parse_time(value: str) -> datetime:
    """Parse a timezone-aware UTC timestamp."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("interaction timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("interaction timestamp must include a timezone")
    return parsed.astimezone(UTC)
