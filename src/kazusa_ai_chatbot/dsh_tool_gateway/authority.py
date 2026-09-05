"""HMAC activation authority for model-invisible semantic tool calls."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from kazusa_ai_chatbot.dsh_tool_gateway.catalog import SEMANTIC_TOOL_NAMES
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import canonical_json, content_digest

AUTHORITY_SCHEMA_VERSION = "kazusa_semantic_tool_authority.v1"
CALL_SCHEMA_VERSION = "kazusa_semantic_tool_call.v1"
ACTIVATION_TOKEN_PREFIX = "ksa1"
_ACTIVATION_MAC_DOMAIN = b"kazusa-semantic-activation-v1\x00"
_MAX_ARGUMENT_BYTES = 32 * 1024
_MUTATION_OPERATIONS = frozenset({
    "kazusa_remember_information",
    "kazusa_revise_memory",
    "kazusa_change_memory_lifecycle",
})


def activation_id_for(
    resolution_thread_id: str,
    segment_id: str,
    lease_epoch: int,
) -> str:
    """Derive the first host activation fence from stable thread identity."""

    _text(resolution_thread_id, "resolution_thread_id")
    _text(segment_id, "segment_id")
    if isinstance(lease_epoch, bool) or lease_epoch < 1:
        raise ValueError("lease_epoch must be positive")
    digest = hashlib.sha256(canonical_json({
        "resolution_thread_id": resolution_thread_id,
        "segment_id": segment_id,
        "lease_epoch": lease_epoch,
    })).hexdigest()[:32]
    return f"act_{digest}"


def _text(value: object, field: str) -> str:
    """Validate a required authority text field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _object(value: object, field: str) -> Mapping[str, Any]:
    """Require a JSON object with string keys."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} keys must be strings")
    return value


def _service_scope(value: object, field: str = "authority.service_scope") -> dict[str, str]:
    """Validate the exact platform/channel/user scope used by every call."""

    data = _object(value, field)
    expected = {"platform", "platform_channel_id", "global_user_id"}
    if set(data) != expected:
        raise ValueError(f"{field} has unknown or missing fields")
    return {
        key: _text(data[key], f"{field}.{key}")
        for key in ("platform", "platform_channel_id", "global_user_id")
    }


def _canonical_workspace(value: object, field: str) -> str:
    """Validate a canonical absolute Windows or POSIX workspace path."""

    path = _text(value, field)
    windows_absolute = len(path) >= 3 and path[1] == ":" and path[2] in "\\/"
    if not path.startswith("/") and not windows_absolute:
        raise ValueError(f"{field} must be an absolute path")
    if "\x00" in path:
        raise ValueError(f"{field} contains an invalid character")
    return path.replace("\\", "/") if windows_absolute else path


def canonical_workspace_root(value: object, field: str = "workspace_root") -> str:
    """Return the canonical absolute workspace spelling shared by both hosts."""

    return _canonical_workspace(value, field)


def _parse_time(value: str, field: str) -> datetime:
    """Parse a UTC ISO timestamp."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class SemanticActivationAuthorityV1:
    """Complete activation fence hidden from the model-facing contract."""

    activation_id: str
    lease_epoch: int
    resolution_thread_id: str
    segment_id: str
    brain_conversation_ref: str
    service_scope: Mapping[str, str]
    scope_fingerprint: str
    audience_fingerprint: str
    workspace_root: str
    route_digest: str
    catalog_digest: str
    profile_version: str
    model_route_digest: str
    workspace_fingerprint: str
    issued_reference_digest: str
    policy_epoch: str
    interaction_issuer: str
    issued_at: str
    expires_at: str
    token_id: str
    nonce: str

    def __post_init__(self) -> None:
        """Validate identity, fence, and bounded lifetime fields."""

        for name in (
            "activation_id",
            "resolution_thread_id",
            "segment_id",
            "brain_conversation_ref",
            "scope_fingerprint",
            "audience_fingerprint",
            "route_digest",
            "catalog_digest",
            "profile_version",
            "model_route_digest",
            "workspace_fingerprint",
            "issued_reference_digest",
            "policy_epoch",
            "interaction_issuer",
            "issued_at",
            "expires_at",
            "token_id",
            "nonce",
        ):
            _text(getattr(self, name), f"authority.{name}")
        _service_scope(self.service_scope)
        _canonical_workspace(self.workspace_root, "authority.workspace_root")
        if isinstance(self.lease_epoch, bool) or self.lease_epoch < 1:
            raise ValueError("authority.lease_epoch must be positive")
        issued = _parse_time(self.issued_at, "authority.issued_at")
        expires = _parse_time(self.expires_at, "authority.expires_at")
        if expires <= issued:
            raise ValueError("authority.expires_at must follow issued_at")
        if (expires - issued).total_seconds() > 300:
            raise ValueError("authority lifetime exceeds five minutes")
        scope = _service_scope(self.service_scope)
        if self.scope_fingerprint != content_digest(scope):
            raise ValueError("authority service scope fingerprint mismatch")
        workspace = _canonical_workspace(
            self.workspace_root, "authority.workspace_root"
        )
        if self.workspace_fingerprint != content_digest({
            "workspace_root": workspace,
        }):
            raise ValueError("authority workspace fingerprint mismatch")
        if self.route_digest != self.model_route_digest:
            raise ValueError("authority route digest mismatch")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical authority payload."""

        return {
            "schema_version": AUTHORITY_SCHEMA_VERSION,
            "activation_id": self.activation_id,
            "lease_epoch": self.lease_epoch,
            "resolution_thread_id": self.resolution_thread_id,
            "segment_id": self.segment_id,
            "brain_conversation_ref": self.brain_conversation_ref,
            "service_scope": _service_scope(self.service_scope),
            "scope_fingerprint": self.scope_fingerprint,
            "audience_fingerprint": self.audience_fingerprint,
            "workspace_root": self.workspace_root,
            "route_digest": self.route_digest,
            "catalog_digest": self.catalog_digest,
            "profile_version": self.profile_version,
            "model_route_digest": self.model_route_digest,
            "workspace_fingerprint": self.workspace_fingerprint,
            "issued_reference_digest": self.issued_reference_digest,
            "policy_epoch": self.policy_epoch,
            "interaction_issuer": self.interaction_issuer,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "token_id": self.token_id,
            "nonce": self.nonce,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "SemanticActivationAuthorityV1":
        """Parse one exact activation authority envelope."""

        data = _object(value, "authority")
        expected = {
            "schema_version",
            "activation_id",
            "lease_epoch",
            "resolution_thread_id",
            "segment_id",
            "brain_conversation_ref",
            "service_scope",
            "scope_fingerprint",
            "audience_fingerprint",
            "workspace_root",
            "route_digest",
            "catalog_digest",
            "profile_version",
            "model_route_digest",
            "workspace_fingerprint",
            "issued_reference_digest",
            "policy_epoch",
            "interaction_issuer",
            "issued_at",
            "expires_at",
            "token_id",
            "nonce",
        }
        if set(data) != expected:
            raise ValueError("authority has unknown or missing fields")
        if data["schema_version"] != AUTHORITY_SCHEMA_VERSION:
            raise ValueError("authority schema is unsupported")
        lease_epoch = data["lease_epoch"]
        if isinstance(lease_epoch, bool) or not isinstance(lease_epoch, int):
            raise ValueError("authority.lease_epoch must be an integer")
        return cls(
            activation_id=_text(data["activation_id"], "authority.activation_id"),
            lease_epoch=lease_epoch,
            resolution_thread_id=_text(
                data["resolution_thread_id"], "authority.resolution_thread_id"
            ),
            segment_id=_text(data["segment_id"], "authority.segment_id"),
            brain_conversation_ref=_text(
                data["brain_conversation_ref"],
                "authority.brain_conversation_ref",
            ),
            service_scope=_service_scope(data["service_scope"]),
            scope_fingerprint=_text(
                data["scope_fingerprint"], "authority.scope_fingerprint"
            ),
            audience_fingerprint=_text(
                data["audience_fingerprint"], "authority.audience_fingerprint"
            ),
            workspace_root=_canonical_workspace(
                data["workspace_root"], "authority.workspace_root"
            ),
            route_digest=_text(data["route_digest"], "authority.route_digest"),
            catalog_digest=_text(
                data["catalog_digest"], "authority.catalog_digest"
            ),
            profile_version=_text(
                data["profile_version"], "authority.profile_version"
            ),
            model_route_digest=_text(
                data["model_route_digest"], "authority.model_route_digest"
            ),
            workspace_fingerprint=_text(
                data["workspace_fingerprint"],
                "authority.workspace_fingerprint",
            ),
            issued_reference_digest=_text(
                data["issued_reference_digest"],
                "authority.issued_reference_digest",
            ),
            policy_epoch=_text(data["policy_epoch"], "authority.policy_epoch"),
            interaction_issuer=_text(
                data["interaction_issuer"], "authority.interaction_issuer"
            ),
            issued_at=_text(data["issued_at"], "authority.issued_at"),
            expires_at=_text(data["expires_at"], "authority.expires_at"),
            token_id=_text(data["token_id"], "authority.token_id"),
            nonce=_text(data["nonce"], "authority.nonce"),
        )


@dataclass(frozen=True, slots=True)
class SignedSemanticCallV1:
    """Signed semantic operation with complete reference lineage."""

    call_id: str
    operation: str
    arguments: dict[str, Any]
    authority: SemanticActivationAuthorityV1
    arguments_digest: str
    issued_reference_digest: str
    idempotency_key: str | None
    signature: str

    def payload(self) -> dict[str, object]:
        """Return the exact signed payload."""

        return {
            "schema_version": CALL_SCHEMA_VERSION,
            "call_id": self.call_id,
            "operation": self.operation,
            "arguments": self.arguments,
            "arguments_digest": self.arguments_digest,
            "issued_reference_digest": self.issued_reference_digest,
            "idempotency_key": self.idempotency_key,
            "authority": self.authority.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        """Return the JSON frame representation."""

        return {**self.payload(), "signature": self.signature}


class SemanticAuthorityReplayOwner(Protocol):
    """Durable owner interface for semantic-call replay exclusion."""

    def consume(
        self,
        authority: SemanticActivationAuthorityV1,
        call_id: str,
    ) -> None:
        """Consume one complete call identity or raise a replay fault."""




class DurableSemanticAuthorityReplayOwner:
    """Adapter for a persistent replay ledger owned outside this module."""

    def __init__(
        self,
        *,
        consume_callback: Callable[[str, str, str], None],
    ) -> None:
        self._consume_callback = consume_callback

    def consume(
        self,
        authority: SemanticActivationAuthorityV1,
        call_id: str,
    ) -> None:
        """Delegate the complete replay identity to the durable owner."""

        self._consume_callback(authority.token_id, authority.nonce, call_id)


def issue_semantic_call(
    authority: SemanticActivationAuthorityV1,
    *,
    operation: str,
    arguments: Mapping[str, Any],
    secret: bytes,
    call_id: str,
    now: str | None = None,
) -> SignedSemanticCallV1:
    """Sign one semantic call under an active complete authority claim."""

    if operation not in SEMANTIC_TOOL_NAMES:
        raise ValueError("semantic operation is unsupported")
    _text(call_id, "call.call_id")
    if not isinstance(secret, bytes) or not secret:
        raise ValueError("semantic authority secret is required")
    args = dict(arguments)
    encoded_args = canonical_json(args)
    if len(encoded_args) > _MAX_ARGUMENT_BYTES:
        raise ValueError("semantic arguments exceed the body limit")
    if now is not None:
        current = _parse_time(now, "call.now")
        issued = _parse_time(authority.issued_at, "authority.issued_at")
        expires = _parse_time(authority.expires_at, "authority.expires_at")
        if current < issued or current > expires:
            raise ValueError("semantic authority is outside its lifetime")
    digest = content_digest(args)
    idempotency_key = (
        _derive_idempotency_key(authority, operation, digest)
        if operation in _MUTATION_OPERATIONS
        else None
    )
    unsigned = SignedSemanticCallV1(
        call_id=call_id,
        operation=operation,
        arguments=args,
        authority=authority,
        arguments_digest=digest,
        issued_reference_digest=authority.issued_reference_digest,
        idempotency_key=idempotency_key,
        signature="",
    )
    signature = _sign(unsigned.payload(), secret)
    return SignedSemanticCallV1(
        call_id=call_id,
        operation=operation,
        arguments=args,
        authority=authority,
        arguments_digest=digest,
        issued_reference_digest=authority.issued_reference_digest,
        idempotency_key=idempotency_key,
        signature=signature,
    )


def authenticate_semantic_call(
    call: SignedSemanticCallV1,
    *,
    secret: bytes,
    now: str | None = None,
) -> SignedSemanticCallV1:
    """Authenticate a semantic call without consuming its replay identity."""

    if not isinstance(secret, bytes) or not secret:
        raise ValueError("semantic authority secret is required")
    if call.operation not in SEMANTIC_TOOL_NAMES:
        raise ValueError("semantic operation is unsupported")
    expected_digest = content_digest(call.arguments)
    if not hmac.compare_digest(call.arguments_digest, expected_digest):
        raise ValueError("semantic arguments digest mismatch")
    if call.issued_reference_digest != call.authority.issued_reference_digest:
        raise ValueError("semantic issued-reference lineage mismatch")
    expected_idempotency = (
        _derive_idempotency_key(
            call.authority,
            call.operation,
            call.arguments_digest,
        )
        if call.operation in _MUTATION_OPERATIONS
        else None
    )
    if call.idempotency_key != expected_idempotency:
        raise ValueError("semantic idempotency lineage mismatch")
    expected_signature = _sign(call.payload(), secret)
    if not hmac.compare_digest(call.signature, expected_signature):
        raise ValueError("semantic authority signature mismatch")
    current = _parse_time(now, "call.now") if now is not None else datetime.now(UTC)
    issued = _parse_time(call.authority.issued_at, "authority.issued_at")
    expires = _parse_time(call.authority.expires_at, "authority.expires_at")
    if current < issued or current > expires:
        raise ValueError("semantic authority has expired")
    return call


def verify_semantic_call(
    call: SignedSemanticCallV1,
    *,
    secret: bytes,
    replay_owner: SemanticAuthorityReplayOwner,
    now: str | None = None,
) -> SignedSemanticCallV1:
    """Authenticate and consume a semantic call before dispatch."""

    authenticated = authenticate_semantic_call(call, secret=secret, now=now)
    replay_owner.consume(authenticated.authority, authenticated.call_id)
    return authenticated


def _sign(payload: Mapping[str, Any], secret: bytes) -> str:
    """Return a canonical HMAC-SHA256 signature."""

    return hmac.new(secret, canonical_json(payload), hashlib.sha256).hexdigest()


def _derive_idempotency_key(
    authority: SemanticActivationAuthorityV1,
    operation: str,
    arguments_digest: str,
) -> str:
    """Derive mutation idempotency from stable semantic intent and lineage.

    The transport call id is deliberately excluded so a retry after worker
    loss resolves to the same mutation key.
    """

    value = {
        "activation_id": authority.activation_id,
        "lease_epoch": authority.lease_epoch,
        "resolution_thread_id": authority.resolution_thread_id,
        "segment_id": authority.segment_id,
        "operation": operation,
        "arguments_digest": arguments_digest,
        "issued_reference_digest": authority.issued_reference_digest,
        "service_scope": _service_scope(authority.service_scope),
        "scope_fingerprint": authority.scope_fingerprint,
        "audience_fingerprint": authority.audience_fingerprint,
    }
    return f"idem:sha256:{hashlib.sha256(canonical_json(value)).hexdigest()}"


def _token_b64encode(value: bytes) -> str:
    """Encode activation bytes without exposing transport padding."""

    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _token_b64decode(value: str) -> bytes:
    """Decode one activation payload from URL-safe base64."""

    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(
            f"{value}{padding}", altchars=b"-_", validate=True
        )
    except (ValueError, binascii.Error) as exc:
        raise ValueError("activation payload is not valid base64") from exc


def issue_activation_token(
    authority: SemanticActivationAuthorityV1,
    *,
    secret: bytes,
    now: str | None = None,
) -> str:
    """Issue the opaque model-hidden activation envelope for one authority."""

    if not isinstance(secret, bytes) or not secret:
        raise ValueError("semantic authority secret is required")
    if now is not None:
        current = _parse_time(now, "activation.now")
        issued = _parse_time(authority.issued_at, "authority.issued_at")
        expires = _parse_time(authority.expires_at, "authority.expires_at")
        if current < issued or current > expires:
            raise ValueError("activation authority is outside its lifetime")
    payload = canonical_json(authority.to_dict())
    mac = hmac.new(
        secret,
        _ACTIVATION_MAC_DOMAIN + payload,
        hashlib.sha256,
    ).hexdigest()
    return f"{ACTIVATION_TOKEN_PREFIX}.{_token_b64encode(payload)}.{mac}"


def verify_activation_token(
    token: str,
    *,
    secret: bytes,
    expected: Mapping[str, Any] | None = None,
    now: str | None = None,
) -> SemanticActivationAuthorityV1:
    """Verify and decode one activation envelope against an optional fence."""

    if not isinstance(secret, bytes) or not secret:
        raise ValueError("semantic authority secret is required")
    _text(token, "activation.token")
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != ACTIVATION_TOKEN_PREFIX:
        raise ValueError("activation token format is invalid")
    try:
        payload = _token_b64decode(parts[1])
    except ValueError as exc:
        raise ValueError("activation token payload is invalid") from exc
    if not payload or _token_b64encode(payload) != parts[1]:
        raise ValueError("activation token payload is not canonical")
    supplied_mac = parts[2]
    expected_mac = hmac.new(
        secret,
        _ACTIVATION_MAC_DOMAIN + payload,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(supplied_mac, expected_mac):
        raise ValueError("activation token authentication failed")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("activation token payload is invalid") from exc
    authority = SemanticActivationAuthorityV1.from_mapping(decoded)
    if canonical_json(authority.to_dict()) != payload:
        raise ValueError("activation token payload is not canonical")
    if expected is not None:
        expected_data = dict(_object(expected, "activation.expected"))
        actual = authority.to_dict()
        for key, value in expected_data.items():
            if key == "service_scope":
                value = _service_scope(value, "activation.expected.service_scope")
                candidate = actual.get(key)
            else:
                candidate = actual.get(key)
            if canonical_json(candidate) != canonical_json(value):
                raise ValueError(f"activation fence mismatch: {key}")
    current = _parse_time(now, "activation.now") if now is not None else datetime.now(UTC)
    issued = _parse_time(authority.issued_at, "authority.issued_at")
    expires = _parse_time(authority.expires_at, "authority.expires_at")
    if current < issued or current > expires:
        raise ValueError("activation authority has expired")
    return authority
