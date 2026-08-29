"""Versioned Brain interaction DTOs and their kind-specific rules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from kazusa_ai_chatbot.dsh_tool_gateway.contracts import canonical_json

INTERACTION_SCHEMA_VERSION = "dsh_brain_interaction.v1"
PENDING_SCHEMA_VERSION = "dsh_interaction_pending.v1"
MAX_INTERACTION_BODY_BYTES = 32 * 1024
MAX_TRANSIENT_DETAIL_CHARS = 8_000
MAX_BRAIN_ANSWER_CHARS = 2_000
INTERACTION_TIMESTAMP_SKEW_SECONDS = 60
ACTIVE_INTERACTION_SECONDS = 5 * 60
PENDING_INTERACTION_SECONDS = 24 * 60 * 60
GRANT_SECONDS = 10 * 60
INTERACTION_KINDS = frozenset({"approval", "question", "plan_review"})
DECISION_KINDS = frozenset({"answer", "allow_once", "reject", "relay_to_user"})
REPLY_DECISION_KINDS = frozenset({
    "answer",
    "allow_once",
    "reject",
    "continue_waiting",
})
RELAY_MODES = frozenset({"question", "approval", "plan_review"})


def _object(value: object, field: str) -> Mapping[str, Any]:
    """Require an object with string keys."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} keys must be strings")
    return value


def _strict(value: object, fields: set[str], field: str) -> Mapping[str, Any]:
    """Require an exact object shape."""

    result = _object(value, field)
    unknown = set(result) - fields
    missing = fields - set(result)
    if unknown:
        raise ValueError(f"{field} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{field} is missing fields: {sorted(missing)}")
    return result


def _text(value: object, field: str, *, empty: bool = False) -> str:
    """Validate bounded text."""

    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_text(value: object, field: str) -> str | None:
    """Validate optional text."""

    if value is None:
        return None
    return _text(value, field)


def _integer(value: object, field: str, minimum: int = 0) -> int:
    """Validate a bounded integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _json_value(value: object, field: str) -> None:
    """Validate only JSON shape for an internal persisted value."""

    try:
        canonical_json(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be canonical JSON") from exc


@dataclass(frozen=True, slots=True)
class DshBrainInteractionRequestV1:
    """Signed request from DSH to the Brain semantic boundary."""

    schema_version: str
    interaction_id: str
    kind: str
    resolution_thread_id: str
    segment_id: str
    activation_id: str
    lease_epoch: int
    dsh_call_id: str
    tool_name: str | None
    operation_id: str
    operation_payload_digest: str
    arguments_digest: str
    transient_detail: str
    brain_conversation_ref: str
    platform: str
    platform_channel_id: str
    global_user_id: str
    scope_fingerprint: str
    audience_fingerprint: str
    profile_version: str
    catalog_digest: str
    model_route_digest: str
    workspace_fingerprint: str
    policy_epoch: str
    issued_reference_digest: str
    nonce: str
    issued_at: str
    expires_at: str
    issuer: str
    mac: str

    @classmethod
    def from_mapping(cls, value: object) -> DshBrainInteractionRequestV1:
        """Parse the exact internal interaction request shape."""

        fields = {
            "schema_version", "interaction_id", "kind", "resolution_thread_id",
            "segment_id", "activation_id", "lease_epoch", "dsh_call_id",
            "tool_name", "operation_id", "operation_payload_digest",
            "arguments_digest", "transient_detail", "brain_conversation_ref",
            "platform", "platform_channel_id", "global_user_id",
            "scope_fingerprint", "audience_fingerprint", "profile_version",
            "catalog_digest", "model_route_digest", "workspace_fingerprint",
            "policy_epoch", "issued_reference_digest", "nonce", "issued_at",
            "expires_at", "issuer", "mac",
        }
        data = _strict(value, fields, "dsh_interaction_request")
        version = _text(data["schema_version"], "request.schema_version")
        if version != INTERACTION_SCHEMA_VERSION:
            raise ValueError("request.schema_version is unsupported")
        kind = _text(data["kind"], "request.kind")
        if kind not in INTERACTION_KINDS:
            raise ValueError("request.kind is unsupported")
        detail = _text(data["transient_detail"], "request.transient_detail")
        if len(detail) > MAX_TRANSIENT_DETAIL_CHARS:
            raise ValueError("request.transient_detail exceeds the bound")
        tool_name_value = data["tool_name"]
        tool_name = (
            None
            if tool_name_value is None
            else _text(tool_name_value, "request.tool_name")
        )
        mac = _text(data["mac"], "request.mac")
        return cls(
            schema_version=version,
            interaction_id=_text(data["interaction_id"], "request.interaction_id"),
            kind=kind,
            resolution_thread_id=_text(data["resolution_thread_id"], "request.resolution_thread_id"),
            segment_id=_text(data["segment_id"], "request.segment_id"),
            activation_id=_text(data["activation_id"], "request.activation_id"),
            lease_epoch=_integer(data["lease_epoch"], "request.lease_epoch", 1),
            dsh_call_id=_text(data["dsh_call_id"], "request.dsh_call_id"),
            tool_name=tool_name,
            operation_id=_text(data["operation_id"], "request.operation_id"),
            operation_payload_digest=_text(
                data["operation_payload_digest"],
                "request.operation_payload_digest",
            ),
            arguments_digest=_text(data["arguments_digest"], "request.arguments_digest"),
            transient_detail=detail,
            brain_conversation_ref=_text(data["brain_conversation_ref"], "request.brain_conversation_ref"),
            platform=_text(data["platform"], "request.platform"),
            platform_channel_id=_text(
                data["platform_channel_id"],
                "request.platform_channel_id",
            ),
            global_user_id=_text(data["global_user_id"], "request.global_user_id"),
            scope_fingerprint=_text(data["scope_fingerprint"], "request.scope_fingerprint"),
            audience_fingerprint=_text(data["audience_fingerprint"], "request.audience_fingerprint"),
            profile_version=_text(data["profile_version"], "request.profile_version"),
            catalog_digest=_text(data["catalog_digest"], "request.catalog_digest"),
            model_route_digest=_text(
                data["model_route_digest"],
                "request.model_route_digest",
            ),
            workspace_fingerprint=_text(
                data["workspace_fingerprint"],
                "request.workspace_fingerprint",
            ),
            policy_epoch=_text(data["policy_epoch"], "request.policy_epoch"),
            issued_reference_digest=_text(
                data["issued_reference_digest"],
                "request.issued_reference_digest",
            ),
            nonce=_text(data["nonce"], "request.nonce"),
            issued_at=_text(data["issued_at"], "request.issued_at"),
            expires_at=_text(data["expires_at"], "request.expires_at"),
            issuer=_text(data["issuer"], "request.issuer"),
            mac=mac,
        )

    @property
    def request_digest(self) -> str:
        """Return the stable digest over the unsigned request identity."""

        encoded = canonical_json(self.unsigned_dict())
        return f"sha256:{sha256(encoded).hexdigest()}"

    def unsigned_dict(self) -> dict[str, object]:
        """Return request fields excluding transport MAC."""

        return {
            "schema_version": self.schema_version,
            "interaction_id": self.interaction_id,
            "kind": self.kind,
            "resolution_thread_id": self.resolution_thread_id,
            "segment_id": self.segment_id,
            "activation_id": self.activation_id,
            "lease_epoch": self.lease_epoch,
            "dsh_call_id": self.dsh_call_id,
            "tool_name": self.tool_name,
            "operation_id": self.operation_id,
            "operation_payload_digest": self.operation_payload_digest,
            "arguments_digest": self.arguments_digest,
            "transient_detail": self.transient_detail,
            "brain_conversation_ref": self.brain_conversation_ref,
            "platform": self.platform,
            "platform_channel_id": self.platform_channel_id,
            "global_user_id": self.global_user_id,
            "scope_fingerprint": self.scope_fingerprint,
            "audience_fingerprint": self.audience_fingerprint,
            "profile_version": self.profile_version,
            "catalog_digest": self.catalog_digest,
            "model_route_digest": self.model_route_digest,
            "workspace_fingerprint": self.workspace_fingerprint,
            "policy_epoch": self.policy_epoch,
            "issued_reference_digest": self.issued_reference_digest,
            "nonce": self.nonce,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "issuer": self.issuer,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the transport representation."""

        return {**self.unsigned_dict(), "mac": self.mac}


@dataclass(frozen=True, slots=True)
class DshOneShotGrantV1:
    """Brain-authored grant for exactly one matching retry."""

    schema_version: str
    interaction_id: str
    resolution_thread_id: str
    segment_id: str
    activation_id: str
    lease_epoch: int
    tool_name: str
    arguments_digest: str
    workspace_fingerprint: str
    scope_fingerprint: str
    policy_epoch: str
    grant_status: str
    issued_at: str
    expires_at: str

    @classmethod
    def from_mapping(cls, value: object) -> DshOneShotGrantV1:
        """Parse one one-shot grant."""

        fields = {
            "schema_version", "interaction_id", "resolution_thread_id", "segment_id",
            "activation_id", "lease_epoch", "tool_name", "arguments_digest",
            "workspace_fingerprint", "scope_fingerprint", "policy_epoch",
            "grant_status", "issued_at", "expires_at",
        }
        data = _strict(value, fields, "dsh_grant")
        version = _text(data["schema_version"], "grant.schema_version")
        if version != INTERACTION_SCHEMA_VERSION:
            raise ValueError("grant.schema_version is unsupported")
        status = _text(data["grant_status"], "grant.grant_status")
        if status not in {"available", "consumed", "expired"}:
            raise ValueError("grant.grant_status is unsupported")
        return cls(
            schema_version=version,
            interaction_id=_text(data["interaction_id"], "grant.interaction_id"),
            resolution_thread_id=_text(data["resolution_thread_id"], "grant.resolution_thread_id"),
            segment_id=_text(data["segment_id"], "grant.segment_id"),
            activation_id=_text(data["activation_id"], "grant.activation_id"),
            lease_epoch=_integer(data["lease_epoch"], "grant.lease_epoch", 1),
            tool_name=_text(data["tool_name"], "grant.tool_name"),
            arguments_digest=_text(data["arguments_digest"], "grant.arguments_digest"),
            workspace_fingerprint=_text(data["workspace_fingerprint"], "grant.workspace_fingerprint"),
            scope_fingerprint=_text(data["scope_fingerprint"], "grant.scope_fingerprint"),
            policy_epoch=_text(data["policy_epoch"], "grant.policy_epoch"),
            grant_status=status,
            issued_at=_text(data["issued_at"], "grant.issued_at"),
            expires_at=_text(data["expires_at"], "grant.expires_at"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""

        return {
            "schema_version": self.schema_version,
            "interaction_id": self.interaction_id,
            "resolution_thread_id": self.resolution_thread_id,
            "segment_id": self.segment_id,
            "activation_id": self.activation_id,
            "lease_epoch": self.lease_epoch,
            "tool_name": self.tool_name,
            "arguments_digest": self.arguments_digest,
            "workspace_fingerprint": self.workspace_fingerprint,
            "scope_fingerprint": self.scope_fingerprint,
            "policy_epoch": self.policy_epoch,
            "grant_status": self.grant_status,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True, slots=True)
class DshBrainInteractionDecisionV1:
    """Brain semantic decision with kind-compatible payload fields."""

    schema_version: str
    interaction_id: str
    request_digest: str
    kind: str
    decision: str
    answer: str | None
    response_goal: str | None
    relay_mode: str | None
    reason: str

    @classmethod
    def from_mapping(cls, value: object) -> DshBrainInteractionDecisionV1:
        """Parse and enforce decision-kind compatibility."""

        fields = {
            "schema_version", "interaction_id", "request_digest", "kind",
            "decision", "answer", "response_goal", "relay_mode", "reason",
        }
        data = _strict(value, fields, "dsh_interaction_decision")
        version = _text(data["schema_version"], "decision.schema_version")
        if version != INTERACTION_SCHEMA_VERSION:
            raise ValueError("decision.schema_version is unsupported")
        kind = _text(data["kind"], "decision.kind")
        decision = _text(data["decision"], "decision.decision")
        if kind not in INTERACTION_KINDS or decision not in DECISION_KINDS:
            raise ValueError("decision kind or decision is unsupported")
        if decision == "answer" and kind not in {"question", "plan_review"}:
            raise ValueError("answer is incompatible with interaction kind")
        if decision == "allow_once" and kind not in {"approval", "plan_review"}:
            raise ValueError("allow_once is incompatible with interaction kind")
        answer = _optional_text(data["answer"], "decision.answer")
        if answer is not None and len(answer) > MAX_BRAIN_ANSWER_CHARS:
            raise ValueError("decision.answer exceeds the bound")
        response_goal = _optional_text(
            data["response_goal"],
            "decision.response_goal",
        )
        if response_goal is not None and len(response_goal) > MAX_BRAIN_ANSWER_CHARS:
            raise ValueError("decision.response_goal exceeds the bound")
        relay_mode = _optional_text(data["relay_mode"], "decision.relay_mode")
        if relay_mode is not None and relay_mode not in RELAY_MODES:
            raise ValueError("decision.relay_mode is unsupported")
        if decision == "answer" and answer is None:
            raise ValueError("answer is required for answer decision")
        if decision != "answer" and answer is not None:
            raise ValueError("answer is status-specific")
        if decision == "relay_to_user" and (
            response_goal is None or relay_mode is None
        ):
            raise ValueError("response_goal and relay_mode are required for relay decision")
        if decision != "relay_to_user" and (
            response_goal is not None or relay_mode is not None
        ):
            raise ValueError("relay fields are status-specific")
        return cls(
            schema_version=version,
            interaction_id=_text(data["interaction_id"], "decision.interaction_id"),
            request_digest=_text(data["request_digest"], "decision.request_digest"),
            kind=kind,
            decision=decision,
            answer=answer,
            response_goal=response_goal,
            relay_mode=relay_mode,
            reason=_text(data["reason"], "decision.reason"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""

        return {
            "schema_version": self.schema_version,
            "interaction_id": self.interaction_id,
            "request_digest": self.request_digest,
            "kind": self.kind,
            "decision": self.decision,
            "answer": self.answer,
            "response_goal": self.response_goal,
            "relay_mode": self.relay_mode,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class DshBrainReplyDecisionV1:
    """Cognition-owned semantic result for one matched user reply."""

    schema_version: str
    interaction_id: str
    request_digest: str
    decision: str
    answer: str | None
    reason: str

    @classmethod
    def from_mapping(cls, value: object) -> DshBrainReplyDecisionV1:
        """Parse the closed reply result without interpreting reply text."""

        data = _strict(
            value,
            {
                "schema_version", "interaction_id", "request_digest",
                "decision", "answer", "reason",
            },
            "dsh_brain_reply_decision",
        )
        version = _text(data["schema_version"], "reply.schema_version")
        if version != INTERACTION_SCHEMA_VERSION:
            raise ValueError("reply.schema_version is unsupported")
        decision = _text(data["decision"], "reply.decision")
        if decision not in REPLY_DECISION_KINDS:
            raise ValueError("reply.decision is unsupported")
        answer = _optional_text(data["answer"], "reply.answer")
        if answer is not None and len(answer) > MAX_BRAIN_ANSWER_CHARS:
            raise ValueError("reply.answer exceeds the bound")
        if decision == "answer" and answer is None:
            raise ValueError("reply.answer is required for answer decision")
        if decision != "answer" and answer is not None:
            raise ValueError("reply.answer is status-specific")
        return cls(
            schema_version=version,
            interaction_id=_text(data["interaction_id"], "reply.interaction_id"),
            request_digest=_text(data["request_digest"], "reply.request_digest"),
            decision=decision,
            answer=answer,
            reason=_text(data["reason"], "reply.reason"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""

        return {
            "schema_version": self.schema_version,
            "interaction_id": self.interaction_id,
            "request_digest": self.request_digest,
            "decision": self.decision,
            "answer": self.answer,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class DshInteractionPendingV1:
    """Durable relay state and exact reply lineage."""

    schema_version: str
    interaction_id: str
    request_digest: str
    resolution_thread_id: str
    segment_id: str
    brain_conversation_ref: str
    platform: str
    platform_channel_id: str
    global_user_id: str
    status: str
    response_goal: str
    relay_mode: str
    created_at: str
    expires_at: str
    delivered_platform_message_id: str | None
    delivery_receipt: dict[str, Any] | None
    replied_at: str | None
    reply_platform_message_id: str | None
    request_identity: dict[str, Any]
    decision: dict[str, Any] | None
    reply_result: dict[str, Any] | None
    grant: DshOneShotGrantV1 | None

    @classmethod
    def from_mapping(cls, value: object) -> DshInteractionPendingV1:
        """Parse a durable pending interaction row."""

        fields = {
            "schema_version", "interaction_id", "request_digest", "resolution_thread_id",
            "segment_id", "brain_conversation_ref", "platform", "platform_channel_id",
            "global_user_id", "status", "response_goal", "relay_mode", "created_at",
            "expires_at", "delivered_platform_message_id", "delivery_receipt", "replied_at",
            "reply_platform_message_id", "request_identity", "decision", "reply_result", "grant",
        }
        data = _strict(value, fields, "dsh_pending")
        version = _text(data["schema_version"], "pending.schema_version")
        if version != PENDING_SCHEMA_VERSION:
            raise ValueError("pending.schema_version is unsupported")
        status = _text(data["status"], "pending.status")
        if status not in {
            "pending",
            "delivered",
            "continuation_pending",
            "replied",
            "expired",
            "failed",
        }:
            raise ValueError("pending.status is unsupported")
        response_goal = _text(data["response_goal"], "pending.response_goal")
        if len(response_goal) > MAX_BRAIN_ANSWER_CHARS:
            raise ValueError("pending.response_goal exceeds the bound")
        relay_mode = _text(data["relay_mode"], "pending.relay_mode")
        if relay_mode not in RELAY_MODES:
            raise ValueError("pending.relay_mode is unsupported")
        delivery_receipt_value = data["delivery_receipt"]
        delivery_receipt = (
            None
            if delivery_receipt_value is None
            else dict(_object(delivery_receipt_value, "pending.delivery_receipt"))
        )
        request_identity = dict(
            _object(data["request_identity"], "pending.request_identity")
        )
        _json_value(request_identity, "pending.request_identity")
        decision_value = data["decision"]
        decision = (
            None
            if decision_value is None
            else dict(_object(decision_value, "pending.decision"))
        )
        if decision is not None:
            _json_value(decision, "pending.decision")
        reply_result_value = data["reply_result"]
        reply_result = (
            None
            if reply_result_value is None
            else dict(_object(reply_result_value, "pending.reply_result"))
        )
        if reply_result is not None:
            _json_value(reply_result, "pending.reply_result")
        grant_value = data["grant"]
        grant = None if grant_value is None else DshOneShotGrantV1.from_mapping(grant_value)
        return cls(
            schema_version=version,
            interaction_id=_text(data["interaction_id"], "pending.interaction_id"),
            request_digest=_text(data["request_digest"], "pending.request_digest"),
            resolution_thread_id=_text(data["resolution_thread_id"], "pending.resolution_thread_id"),
            segment_id=_text(data["segment_id"], "pending.segment_id"),
            brain_conversation_ref=_text(data["brain_conversation_ref"], "pending.brain_conversation_ref"),
            platform=_text(data["platform"], "pending.platform"),
            platform_channel_id=_text(data["platform_channel_id"], "pending.platform_channel_id"),
            global_user_id=_text(data["global_user_id"], "pending.global_user_id"),
            status=status,
            response_goal=response_goal,
            relay_mode=relay_mode,
            created_at=_text(data["created_at"], "pending.created_at"),
            expires_at=_text(data["expires_at"], "pending.expires_at"),
            delivered_platform_message_id=_optional_text(
                data["delivered_platform_message_id"],
                "pending.delivered_platform_message_id",
            ),
            delivery_receipt=delivery_receipt,
            replied_at=_optional_text(data["replied_at"], "pending.replied_at"),
            reply_platform_message_id=_optional_text(data["reply_platform_message_id"], "pending.reply_platform_message_id"),
            request_identity=request_identity,
            decision=decision,
            reply_result=reply_result,
            grant=grant,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""

        return {
            "schema_version": self.schema_version,
            "interaction_id": self.interaction_id,
            "request_digest": self.request_digest,
            "resolution_thread_id": self.resolution_thread_id,
            "segment_id": self.segment_id,
            "brain_conversation_ref": self.brain_conversation_ref,
            "platform": self.platform,
            "platform_channel_id": self.platform_channel_id,
            "global_user_id": self.global_user_id,
            "status": self.status,
            "response_goal": self.response_goal,
            "relay_mode": self.relay_mode,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "delivered_platform_message_id": self.delivered_platform_message_id,
            "delivery_receipt": self.delivery_receipt,
            "replied_at": self.replied_at,
            "reply_platform_message_id": self.reply_platform_message_id,
            "request_identity": self.request_identity,
            "decision": self.decision,
            "reply_result": self.reply_result,
            "grant": None if self.grant is None else self.grant.to_dict(),
        }
