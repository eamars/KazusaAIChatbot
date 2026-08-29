"""Storage-independent contracts for the semantic capability gateway."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

SEMANTIC_RESULT_SCHEMA_VERSION = "kazusa_semantic_capability_result.v1"
EVIDENCE_RECEIPT_SCHEMA_VERSION = "evidence_receipt.v2"
MAX_SEMANTIC_ENTITIES = 64
MAX_SEMANTIC_EVIDENCE = 64
MAX_REFERENCE_LENGTH = 4096
_AUTHORITY_SCHEMA_VERSION = "kazusa_semantic_tool_authority.v1"

_STATUSES = frozenset({
    "ok",
    "empty",
    "denied",
    "invalid",
    "timeout",
    "unavailable",
})
_MUTATION_OUTCOMES = frozenset({
    "committed",
    "already_committed",
    "rejected",
    "uncertain",
})
def _object(value: object, field: str) -> Mapping[str, Any]:
    """Require a JSON object with string keys."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} keys must be strings")
    return value


def _strict(value: object, fields: set[str], field: str) -> Mapping[str, Any]:
    """Validate the exact field set for one contract object."""

    result = _object(value, field)
    unknown = set(result) - fields
    missing = fields - set(result)
    if unknown:
        raise ValueError(f"{field} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{field} is missing fields: {sorted(missing)}")
    return result


def _text(value: object, field: str, *, empty: bool = False) -> str:
    """Validate one bounded text value."""

    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{field} must be a non-empty string")
    if len(value) > MAX_REFERENCE_LENGTH:
        raise ValueError(f"{field} is too long")
    return value


def _optional_text(value: object, field: str) -> str | None:
    """Validate an optional text value."""

    if value is None:
        return None
    return _text(value, field)


def _bounded_list(value: object, field: str, maximum: int) -> list[object]:
    """Validate a bounded JSON list."""

    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{field} must be a bounded list")
    return value


def canonical_json(value: object) -> bytes:
    """Encode a JSON value deterministically for opaque references and MACs."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc
    return encoded.encode("utf-8")


def content_digest(value: object) -> str:
    """Return the stable digest used by semantic evidence receipts."""

    digest = hashlib.sha256(canonical_json(value)).hexdigest()
    return f"sha256:{digest}"


_REFERENCE_AUTHORITY_DIGEST_DOMAIN = b"kazusa-reference-authority-digest-v2\x00"


def _reference_authority_digest(authority: Mapping[str, Any]) -> str:
    """Digest one validated authority with a reference-specific domain."""

    digest = hashlib.sha256(
        _REFERENCE_AUTHORITY_DIGEST_DOMAIN + canonical_json(authority)
    ).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True, slots=True)
class OpaqueReferenceCodec:
    """Issue sealed, lineage-bound references without exposing storage ids."""

    secret: bytes
    issuer: str = "kazusa-semantic"
    authority: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Require an explicit secret and validate an optional authority bind."""

        if not isinstance(self.secret, bytes) or not self.secret:
            raise ValueError("reference codec secret is required")
        _text(self.issuer, "reference.issuer")
        if self.authority is not None:
            _reference_authority(self.authority)

    def with_authority(
        self,
        authority: Mapping[str, Any] | object,
    ) -> "OpaqueReferenceCodec":
        """Return a codec bound to one complete activation authority."""

        return OpaqueReferenceCodec(
            secret=self.secret,
            issuer=self.issuer,
            authority=_reference_authority(authority),
        )

    def issue(
        self,
        kind: str,
        value: Mapping[str, Any],
        *,
        authority: Mapping[str, Any] | object | None = None,
    ) -> str:
        """Create one deterministic sealed reference for a bound lineage."""

        _text(kind, "reference.kind")
        value_payload = dict(_object(value, "reference.value"))
        bound_authority = self._authority(authority)
        authority_digest = (
            _reference_authority_digest(bound_authority)
            if bound_authority is not None
            else None
        )
        body = {
            "issuer": self.issuer,
            "kind": kind,
            "authority_digest": authority_digest,
            "value": value_payload,
        }
        plaintext = canonical_json(body)
        nonce = hmac.new(
            self.secret,
            b"kazusa-reference-nonce-v2\x00" + plaintext,
            hashlib.sha256,
        ).digest()[:16]
        ciphertext = _seal(self.secret, nonce, plaintext)
        tag = hmac.new(
            self.secret,
            b"kazusa-reference-v2\x00" + nonce + ciphertext,
            hashlib.sha256,
        ).hexdigest()
        return f"kr2.{_b64encode(nonce)}.{_b64encode(ciphertext)}.{tag}"

    def resolve(
        self,
        reference: str,
        expected_kind: str,
        *,
        authority: Mapping[str, Any] | object | None = None,
    ) -> dict[str, Any]:
        """Validate and open one sealed reference for the expected lineage."""

        _text(reference, "reference")
        parts = reference.split(".")
        if len(parts) != 4 or parts[0] != "kr2":
            raise ValueError("reference format is invalid")
        nonce_text, ciphertext_text, supplied = parts[1], parts[2], parts[3]
        try:
            nonce = _b64decode_bytes(nonce_text)
            ciphertext = _b64decode_bytes(ciphertext_text)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("reference payload is invalid") from exc
        if len(nonce) != 16 or not ciphertext:
            raise ValueError("reference payload is invalid")
        expected = hmac.new(
            self.secret,
            b"kazusa-reference-v2\x00" + nonce + ciphertext,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(supplied, expected):
            raise ValueError("reference authentication failed")
        try:
            value = json.loads(_open(self.secret, nonce, ciphertext).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("reference payload is invalid") from exc
        data = _strict(
            value,
            {"issuer", "kind", "authority_digest", "value"},
            "reference",
        )
        issuer = _text(data["issuer"], "reference.issuer")
        kind = _text(data["kind"], "reference.kind")
        if issuer != self.issuer or kind != expected_kind:
            raise ValueError("reference lineage is invalid")
        token_authority_digest = data["authority_digest"]
        if token_authority_digest is not None:
            token_authority_digest = _text(
                token_authority_digest,
                "reference.authority_digest",
            )
        expected_authority = self._authority(authority)
        expected_authority_digest = (
            _reference_authority_digest(expected_authority)
            if expected_authority is not None
            else None
        )
        if token_authority_digest is None or expected_authority_digest is None:
            authority_matches = token_authority_digest == expected_authority_digest
        else:
            authority_matches = hmac.compare_digest(
                token_authority_digest,
                expected_authority_digest,
            )
        if not authority_matches:
            raise ValueError("reference authority lineage is invalid")
        resolved = _object(data["value"], "reference.value")
        return dict(resolved)

    def _authority(
        self,
        authority: Mapping[str, Any] | object | None,
    ) -> dict[str, Any] | None:
        """Resolve the call-specific authority or this codec's bound one."""

        if authority is None:
            authority = self.authority
        if authority is None:
            return None
        return _reference_authority(authority)


_REFERENCE_AUTHORITY_FIELDS = frozenset({
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
})


def _reference_authority(value: Mapping[str, Any] | object) -> dict[str, Any]:
    """Normalize the complete hidden authority used for reference binding."""

    if hasattr(value, "to_dict"):
        value = value.to_dict()  # type: ignore[union-attr]
    data = dict(_object(value, "reference.authority"))
    missing = _REFERENCE_AUTHORITY_FIELDS - set(data)
    if missing:
        raise ValueError(
            f"reference.authority is missing fields: {sorted(missing)}"
        )
    unknown = set(data) - _REFERENCE_AUTHORITY_FIELDS
    if unknown:
        raise ValueError(
            f"reference.authority has unknown fields: {sorted(unknown)}"
        )
    if data["schema_version"] != _AUTHORITY_SCHEMA_VERSION:
        raise ValueError("reference.authority schema is unsupported")
    for field in _REFERENCE_AUTHORITY_FIELDS - {
        "lease_epoch", "schema_version", "service_scope",
    }:
        _text(data[field], f"reference.authority.{field}")
    scope = data["service_scope"]
    if not isinstance(scope, Mapping) or set(scope) != {
        "platform", "platform_channel_id", "global_user_id",
    }:
        raise ValueError("reference.authority.service_scope is invalid")
    for field in ("platform", "platform_channel_id", "global_user_id"):
        _text(scope[field], f"reference.authority.service_scope.{field}")
    data["service_scope"] = {
        field: scope[field]
        for field in ("platform", "platform_channel_id", "global_user_id")
    }
    workspace = data["workspace_root"]
    if not isinstance(workspace, str) or not workspace:
        raise ValueError("reference.authority.workspace_root is invalid")
    workspace = workspace.replace("\\", "/")
    data["workspace_root"] = workspace
    if data["scope_fingerprint"] != content_digest(data["service_scope"]):
        raise ValueError("reference.authority.service scope fingerprint mismatch")
    if data["workspace_fingerprint"] != content_digest({
        "workspace_root": workspace,
    }):
        raise ValueError("reference.authority.workspace fingerprint mismatch")
    if data["route_digest"] != data["model_route_digest"]:
        raise ValueError("reference.authority.route digest mismatch")
    if not isinstance(data["lease_epoch"], int) or isinstance(
        data["lease_epoch"], bool
    ) or data["lease_epoch"] < 1:
        raise ValueError("reference.authority.lease_epoch must be positive")
    try:
        issued_at = datetime.fromisoformat(
            str(data["issued_at"]).replace("Z", "+00:00")
        )
        expires_at = datetime.fromisoformat(
            str(data["expires_at"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("reference.authority timestamps are invalid") from exc
    if issued_at.tzinfo is None or expires_at.tzinfo is None:
        raise ValueError("reference.authority timestamps require a timezone")
    issued_at = issued_at.astimezone(UTC)
    expires_at = expires_at.astimezone(UTC)
    if expires_at <= issued_at or (expires_at - issued_at).total_seconds() > 300:
        raise ValueError("reference.authority lifetime is invalid")
    current_time = datetime.now(UTC)
    if current_time < issued_at or current_time > expires_at:
        raise ValueError("reference.authority has expired")
    return data


def _b64encode(value: bytes) -> str:
    """Encode bytes in URL-safe unpadded base64."""

    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode_bytes(value: str) -> bytes:
    """Decode URL-safe unpadded base64 to bytes."""

    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(
            f"{value}{padding}",
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise ValueError("base64 payload is invalid") from exc


def _seal(secret: bytes, nonce: bytes, plaintext: bytes) -> bytes:
    """Encrypt JSON with a keyed stream whose ciphertext is not self-describing."""

    return _xor_stream(secret, nonce, plaintext)


def _open(secret: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    """Open one sealed payload using the same keyed stream."""

    return _xor_stream(secret, nonce, ciphertext)


def _xor_stream(secret: bytes, nonce: bytes, value: bytes) -> bytes:
    """Apply a HMAC-derived stream to one payload."""

    output = bytearray(len(value))
    for offset in range(0, len(value), hashlib.sha256().digest_size):
        block = hmac.new(
            secret,
            b"kazusa-reference-stream-v2\x00"
            + nonce
            + (offset // hashlib.sha256().digest_size).to_bytes(4, "big"),
            hashlib.sha256,
        ).digest()
        chunk = value[offset: offset + len(block)]
        output[offset: offset + len(chunk)] = bytes(
            left ^ right for left, right in zip(chunk, block)
        )
    return bytes(output)


@dataclass(frozen=True, slots=True)
class SemanticPageV1:
    """Opaque pagination state returned by semantic services."""

    has_more: bool
    next_page_ref: str | None

    @classmethod
    def from_mapping(cls, value: object) -> "SemanticPageV1":
        """Parse a page result with no storage cursor fields."""

        data = _strict(value, {"has_more", "next_page_ref"}, "page")
        has_more = data["has_more"]
        if not isinstance(has_more, bool):
            raise ValueError("page.has_more must be a boolean")
        next_page_ref = _optional_text(data["next_page_ref"], "page.next_page_ref")
        if has_more and next_page_ref is None:
            raise ValueError("page.next_page_ref is required when has_more")
        if not has_more and next_page_ref is not None:
            raise ValueError("page.next_page_ref requires has_more")
        return cls(has_more=has_more, next_page_ref=next_page_ref)

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""

        return {"has_more": self.has_more, "next_page_ref": self.next_page_ref}


@dataclass(frozen=True, slots=True)
class SemanticErrorV1:
    """Safe, typed semantic capability error."""

    code: str
    safe_message: str

    @classmethod
    def from_mapping(cls, value: object) -> "SemanticErrorV1":
        """Parse one semantic error."""

        data = _strict(value, {"code", "safe_message"}, "error")
        return cls(
            code=_text(data["code"], "error.code"),
            safe_message=_text(data["safe_message"], "error.safe_message"),
        )

    def to_dict(self) -> dict[str, str]:
        """Return the JSON representation."""

        return {"code": self.code, "safe_message": self.safe_message}


@dataclass(frozen=True, slots=True)
class SemanticMutationOutcomeV1:
    """Idempotent mutation outcome bound to an opaque semantic reference."""

    outcome: str
    semantic_ref: str
    idempotency_key: str

    @classmethod
    def from_mapping(cls, value: object) -> "SemanticMutationOutcomeV1":
        """Parse one mutation outcome."""

        data = _strict(
            value,
            {"outcome", "semantic_ref", "idempotency_key"},
            "mutation",
        )
        outcome = _text(data["outcome"], "mutation.outcome")
        if outcome not in _MUTATION_OUTCOMES:
            raise ValueError("mutation.outcome is unsupported")
        return cls(
            outcome=outcome,
            semantic_ref=_text(data["semantic_ref"], "mutation.semantic_ref"),
            idempotency_key=_text(
                data["idempotency_key"],
                "mutation.idempotency_key",
            ),
        )

    def to_dict(self) -> dict[str, str]:
        """Return the JSON representation."""

        return {
            "outcome": self.outcome,
            "semantic_ref": self.semantic_ref,
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True, slots=True)
class EvidenceReceiptV2:
    """Prompt-safe provenance receipt for a semantic result."""

    receipt_id: str
    source_kind: str
    semantic_ref: str
    content_digest: str
    occurred_at: str | None = None

    @classmethod
    def from_mapping(cls, value: object) -> "EvidenceReceiptV2":
        """Parse one evidence receipt."""

        data = _strict(
            value,
            {
                "receipt_id",
                "source_kind",
                "semantic_ref",
                "content_digest",
                "occurred_at",
            },
            "evidence",
        )
        return cls(
            receipt_id=_text(data["receipt_id"], "evidence.receipt_id"),
            source_kind=_text(data["source_kind"], "evidence.source_kind"),
            semantic_ref=_text(data["semantic_ref"], "evidence.semantic_ref"),
            content_digest=_text(
                data["content_digest"],
                "evidence.content_digest",
            ),
            occurred_at=_optional_text(data["occurred_at"], "evidence.occurred_at"),
        )

    def to_dict(self) -> dict[str, str | None]:
        """Return the JSON representation."""

        return {
            "receipt_id": self.receipt_id,
            "source_kind": self.source_kind,
            "semantic_ref": self.semantic_ref,
            "content_digest": self.content_digest,
            "occurred_at": self.occurred_at,
        }


@dataclass(frozen=True, slots=True)
class KazusaSemanticCapabilityResultV1:
    """Common result envelope shared by all thirteen semantic tools."""

    schema_version: str
    status: str
    entities: tuple[dict[str, Any], ...]
    page: SemanticPageV1
    evidence: tuple[EvidenceReceiptV2, ...]
    mutation: SemanticMutationOutcomeV1 | None
    error: SemanticErrorV1 | None

    @classmethod
    def from_mapping(cls, value: object) -> "KazusaSemanticCapabilityResultV1":
        """Parse and validate one semantic capability result."""

        fields = {
            "schema_version",
            "status",
            "entities",
            "page",
            "evidence",
            "mutation",
            "error",
        }
        data = _strict(value, fields, "semantic_result")
        schema_version = _text(data["schema_version"], "semantic_result.schema_version")
        if schema_version != SEMANTIC_RESULT_SCHEMA_VERSION:
            raise ValueError("semantic_result.schema_version is unsupported")
        status = _text(data["status"], "semantic_result.status")
        if status not in _STATUSES:
            raise ValueError("semantic_result.status is unsupported")
        entities_value = _bounded_list(
            data["entities"],
            "semantic_result.entities",
            MAX_SEMANTIC_ENTITIES,
        )
        entities: list[dict[str, Any]] = []
        for index, item in enumerate(entities_value):
            entity = dict(_object(item, f"semantic_result.entities[{index}]"))
            canonical_json(entity)
            entities.append(entity)
        evidence_value = _bounded_list(
            data["evidence"],
            "semantic_result.evidence",
            MAX_SEMANTIC_EVIDENCE,
        )
        evidence = tuple(EvidenceReceiptV2.from_mapping(item) for item in evidence_value)
        mutation_value = data["mutation"]
        mutation = (
            None
            if mutation_value is None
            else SemanticMutationOutcomeV1.from_mapping(mutation_value)
        )
        error_value = data["error"]
        error = None if error_value is None else SemanticErrorV1.from_mapping(error_value)
        if status in {"ok", "empty"} and error is not None:
            raise ValueError("semantic_result.error is status-incompatible")
        if status not in {"ok", "empty"} and error is None:
            raise ValueError("semantic_result.error is required for a fault")
        return cls(
            schema_version=schema_version,
            status=status,
            entities=tuple(entities),
            page=SemanticPageV1.from_mapping(data["page"]),
            evidence=evidence,
            mutation=mutation,
            error=error,
        )

    @classmethod
    def success(
        cls,
        *,
        entities: Sequence[Mapping[str, Any]] = (),
        evidence: Sequence[EvidenceReceiptV2] = (),
        page: SemanticPageV1 | None = None,
        mutation: SemanticMutationOutcomeV1 | None = None,
    ) -> "KazusaSemanticCapabilityResultV1":
        """Build a successful result while applying contract checks."""

        selected_page = page or SemanticPageV1(False, None)
        values = tuple(dict(entity) for entity in entities)
        for entity in values:
            canonical_json(entity)
        return cls(
            schema_version=SEMANTIC_RESULT_SCHEMA_VERSION,
            status="ok" if values or mutation is not None else "empty",
            entities=values,
            page=selected_page,
            evidence=tuple(evidence),
            mutation=mutation,
            error=None,
        )

    @classmethod
    def failure(
        cls,
        status: str,
        code: str,
        safe_message: str,
    ) -> "KazusaSemanticCapabilityResultV1":
        """Build a typed non-success result."""

        if status not in _STATUSES - {"ok", "empty"}:
            raise ValueError("failure status is unsupported")
        return cls(
            schema_version=SEMANTIC_RESULT_SCHEMA_VERSION,
            status=status,
            entities=(),
            page=SemanticPageV1(False, None),
            evidence=(),
            mutation=None,
            error=SemanticErrorV1(code, safe_message),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "entities": [dict(entity) for entity in self.entities],
            "page": self.page.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "mutation": None if self.mutation is None else self.mutation.to_dict(),
            "error": None if self.error is None else self.error.to_dict(),
        }


def new_evidence_receipt(
    *,
    receipt_id: str,
    source_kind: str,
    semantic_ref: str,
    value: object,
    occurred_at: str | None = None,
) -> EvidenceReceiptV2:
    """Create a content-bound evidence receipt for a semantic entity."""

    selected_time = occurred_at
    if selected_time is None:
        selected_time = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return EvidenceReceiptV2(
        receipt_id=_text(receipt_id, "evidence.receipt_id"),
        source_kind=_text(source_kind, "evidence.source_kind"),
        semantic_ref=_text(semantic_ref, "evidence.semantic_ref"),
        content_digest=content_digest(value),
        occurred_at=selected_time,
    )
