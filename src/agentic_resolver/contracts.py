"""Strict versioned Kazusa DTOs for the standalone DSH sidecar."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from agentic_resolver.errors import ResolverContractError, RuntimeFaultCode

RPC_PROTOCOL_VERSION = "kazusa.dsh-resolution-rpc.v1"
INTAKE_SCHEMA_VERSION = "dsh_resolution_intake.v1"
THREAD_SCHEMA_VERSION = "resolution_thread_store.v1"
SEGMENT_SCHEMA_VERSION = "resolver_session_segment.v1"
PROFILE_VERSION = "kazusa-resolver-v1"
DSH_RELEASE = "0.1.1-rc.2"
SESSION_STORE_EPOCH = "dsh-sqlite-0.1.1-rc.2-v1"


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResolverContractError(f"{context} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ResolverContractError(f"{context} keys must be strings")
    return value


def _strict(value: object, keys: set[str], context: str) -> Mapping[str, Any]:
    result = _mapping(value, context)
    unknown = set(result) - keys
    missing = keys - set(result)
    if unknown:
        raise ResolverContractError(
            f"{context} has unknown fields: {sorted(unknown)}"
        )
    if missing:
        raise ResolverContractError(
            f"{context} is missing fields: {sorted(missing)}"
        )
    return result


def _text(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ResolverContractError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ResolverContractError(f"{field} must be an integer >= {minimum}")
    return value


def _texts(value: object, field: str, *, maximum: int = 64) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ResolverContractError(f"{field} must be a bounded list")
    result = tuple(_text(item, field) for item in value)
    if len(result) != len(set(result)):
        raise ResolverContractError(f"{field} must contain unique values")
    return result


def _optional_mapping(value: object, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return dict(_mapping(value, field))


@dataclass(frozen=True, slots=True)
class DSHResolutionRuntimeV1:
    """Deterministic operation authority excluded from model-visible input."""

    PRIORITIES: ClassVar[frozenset[str]] = frozenset({"now", "background"})
    request_id: str
    operation_id: str
    operation_payload_digest: str
    resolution_thread_id: str
    segment_id: str
    priority: str
    soft_deadline_at: str
    hard_deadline_at: str
    max_model_steps: int
    max_tool_calls: int
    max_tool_bytes: int
    capability_token: str
    scope_fingerprint: str
    audience_fingerprint: str
    resolver_profile_version: str
    dsh_release: str
    session_store_epoch: str
    model_route: str
    tool_catalog_digest: str
    policy_epoch: str

    @classmethod
    def from_mapping(cls, value: object) -> DSHResolutionRuntimeV1:
        keys = {
            "request_id", "operation_id", "operation_payload_digest",
            "resolution_thread_id", "segment_id", "priority",
            "soft_deadline_at", "hard_deadline_at", "max_model_steps",
            "max_tool_calls", "max_tool_bytes", "capability_token",
            "scope_fingerprint", "audience_fingerprint",
            "resolver_profile_version", "dsh_release", "session_store_epoch",
            "model_route", "tool_catalog_digest", "policy_epoch",
        }
        data = _strict(value, keys, "runtime")
        priority = _text(data["priority"], "runtime.priority")
        if priority not in cls.PRIORITIES:
            raise ResolverContractError("runtime.priority is unsupported")
        profile = _text(
            data["resolver_profile_version"], "runtime.resolver_profile_version"
        )
        if profile != PROFILE_VERSION:
            raise ResolverContractError("resolver_profile_version is unsupported")
        release = _text(data["dsh_release"], "runtime.dsh_release")
        if release != DSH_RELEASE:
            raise ResolverContractError("dsh_release is unsupported")
        store_epoch = _text(
            data["session_store_epoch"], "runtime.session_store_epoch"
        )
        if store_epoch != SESSION_STORE_EPOCH:
            raise ResolverContractError("session_store_epoch is unsupported")
        return cls(
            request_id=_text(data["request_id"], "runtime.request_id"),
            operation_id=_text(data["operation_id"], "runtime.operation_id"),
            operation_payload_digest=_text(
                data["operation_payload_digest"],
                "runtime.operation_payload_digest",
            ),
            resolution_thread_id=_text(
                data["resolution_thread_id"], "runtime.resolution_thread_id"
            ),
            segment_id=_text(data["segment_id"], "runtime.segment_id"),
            priority=priority,
            soft_deadline_at=_text(
                data["soft_deadline_at"], "runtime.soft_deadline_at"
            ),
            hard_deadline_at=_text(
                data["hard_deadline_at"], "runtime.hard_deadline_at"
            ),
            max_model_steps=_integer(
                data["max_model_steps"], "runtime.max_model_steps", 1
            ),
            max_tool_calls=_integer(
                data["max_tool_calls"], "runtime.max_tool_calls", 1
            ),
            max_tool_bytes=_integer(
                data["max_tool_bytes"], "runtime.max_tool_bytes", 1
            ),
            capability_token=_text(
                data["capability_token"], "runtime.capability_token"
            ),
            scope_fingerprint=_text(
                data["scope_fingerprint"], "runtime.scope_fingerprint"
            ),
            audience_fingerprint=_text(
                data["audience_fingerprint"], "runtime.audience_fingerprint"
            ),
            resolver_profile_version=profile,
            dsh_release=release,
            session_store_epoch=store_epoch,
            model_route=_text(data["model_route"], "runtime.model_route"),
            tool_catalog_digest=_text(
                data["tool_catalog_digest"], "runtime.tool_catalog_digest"
            ),
            policy_epoch=_text(data["policy_epoch"], "runtime.policy_epoch"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field != "PRIORITIES"
        }


@dataclass(frozen=True, slots=True)
class DSHResolutionModelInputV1:
    """Bounded semantic material rendered into the DSH waking message."""

    objective: str
    constraints: tuple[str, ...]
    success_criteria: tuple[str, ...]
    known_facts: tuple[str, ...]
    uncertainty: tuple[str, ...]
    literal_inputs: tuple[str, ...]
    continuation_delta: str | None
    prior_resolution_refs: tuple[str, ...]
    requested_evidence_quality: str
    notes: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> DSHResolutionModelInputV1:
        keys = {
            "objective", "constraints", "success_criteria", "known_facts",
            "uncertainty", "literal_inputs", "continuation_delta",
            "prior_resolution_refs", "requested_evidence_quality", "notes",
        }
        data = _strict(value, keys, "model_input")
        delta = data["continuation_delta"]
        if delta is not None:
            delta = _text(delta, "model_input.continuation_delta")
        return cls(
            objective=_text(data["objective"], "model_input.objective"),
            constraints=_texts(data["constraints"], "model_input.constraints"),
            success_criteria=_texts(
                data["success_criteria"], "model_input.success_criteria"
            ),
            known_facts=_texts(data["known_facts"], "model_input.known_facts"),
            uncertainty=_texts(data["uncertainty"], "model_input.uncertainty"),
            literal_inputs=_texts(
                data["literal_inputs"], "model_input.literal_inputs"
            ),
            continuation_delta=delta,
            prior_resolution_refs=_texts(
                data["prior_resolution_refs"],
                "model_input.prior_resolution_refs",
            ),
            requested_evidence_quality=_text(
                data["requested_evidence_quality"],
                "model_input.requested_evidence_quality",
            ),
            notes=_texts(data["notes"], "model_input.notes"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "constraints": list(self.constraints),
            "success_criteria": list(self.success_criteria),
            "known_facts": list(self.known_facts),
            "uncertainty": list(self.uncertainty),
            "literal_inputs": list(self.literal_inputs),
            "continuation_delta": self.continuation_delta,
            "prior_resolution_refs": list(self.prior_resolution_refs),
            "requested_evidence_quality": self.requested_evidence_quality,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class DSHResolutionIntakeV1:
    """Canonical standalone sidecar intake."""

    MODES: ClassVar[frozenset[str]] = frozenset({"start", "continue"})
    schema_version: str
    mode: str
    runtime: DSHResolutionRuntimeV1
    model_input: DSHResolutionModelInputV1

    @classmethod
    def from_mapping(cls, value: object) -> DSHResolutionIntakeV1:
        data = _strict(
            value,
            {"schema_version", "mode", "runtime", "model_input"},
            "intake",
        )
        version = _text(data["schema_version"], "intake.schema_version")
        if version != INTAKE_SCHEMA_VERSION:
            raise ResolverContractError("intake.schema_version is unsupported")
        mode = _text(data["mode"], "intake.mode")
        if mode not in cls.MODES:
            raise ResolverContractError("intake.mode is unsupported")
        return cls(
            schema_version=version,
            mode=mode,
            runtime=DSHResolutionRuntimeV1.from_mapping(data["runtime"]),
            model_input=DSHResolutionModelInputV1.from_mapping(
                data["model_input"]
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "runtime": self.runtime.to_dict(),
            "model_input": self.model_input.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SubmitResolutionV1:
    """Validated terminal semantic product returned by the terminal action."""

    STATUSES: ClassVar[frozenset[str]] = frozenset({
        "resolved", "partial", "needs_user_input", "approval_required",
        "unavailable", "failed",
    })
    status: str
    summary: str
    findings: tuple[dict[str, Any], ...]
    completed_subgoals: tuple[str, ...]
    remaining_needs: tuple[str, ...]
    clarification_request: dict[str, Any] | None
    approval_request: dict[str, Any] | None
    artifact_refs: tuple[str, ...]
    warnings: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> SubmitResolutionV1:
        keys = {
            "status", "summary", "findings", "completed_subgoals",
            "remaining_needs", "clarification_request", "approval_request",
            "artifact_refs", "warnings",
        }
        data = _strict(value, keys, "submit_resolution")
        status = _text(data["status"], "submit_resolution.status")
        if status not in cls.STATUSES:
            raise ResolverContractError("submit_resolution.status is unsupported")
        findings_value = data["findings"]
        if not isinstance(findings_value, list) or len(findings_value) > 64:
            raise ResolverContractError(
                "submit_resolution.findings must be a bounded list"
            )
        findings = tuple(
            dict(_mapping(item, "submit_resolution.findings item"))
            for item in findings_value
        )
        clarification = _optional_mapping(
            data["clarification_request"],
            "submit_resolution.clarification_request",
        )
        approval = _optional_mapping(
            data["approval_request"], "submit_resolution.approval_request"
        )
        if status == "needs_user_input" and clarification is None:
            raise ResolverContractError(
                "clarification_request is required for needs_user_input"
            )
        if status == "approval_required" and approval is None:
            raise ResolverContractError(
                "approval_request is required for approval_required"
            )
        if status != "needs_user_input" and clarification is not None:
            raise ResolverContractError(
                "clarification_request is status-specific"
            )
        if status != "approval_required" and approval is not None:
            raise ResolverContractError("approval_request is status-specific")
        return cls(
            status=status,
            summary=_text(data["summary"], "submit_resolution.summary"),
            findings=findings,
            completed_subgoals=_texts(
                data["completed_subgoals"],
                "submit_resolution.completed_subgoals",
            ),
            remaining_needs=_texts(
                data["remaining_needs"], "submit_resolution.remaining_needs"
            ),
            clarification_request=clarification,
            approval_request=approval,
            artifact_refs=_texts(
                data["artifact_refs"], "submit_resolution.artifact_refs"
            ),
            warnings=_texts(data["warnings"], "submit_resolution.warnings"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "summary": self.summary,
            "findings": [dict(item) for item in self.findings],
            "completed_subgoals": list(self.completed_subgoals),
            "remaining_needs": list(self.remaining_needs),
            "clarification_request": self.clarification_request,
            "approval_request": self.approval_request,
            "artifact_refs": list(self.artifact_refs),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class EvidenceReferenceV1:
    """Bounded public evidence identity projected from sidecar validation."""

    schema_version: str
    evidence_id: str
    resolution_thread_id: str
    segment_id: str
    scope_fingerprint: str
    audience_fingerprint: str
    policy_epoch: str
    tool_name: str
    source_kind: str
    source_id: str
    content_digest: str

    @classmethod
    def from_mapping(cls, value: object) -> EvidenceReferenceV1:
        keys = {
            "schema_version", "evidence_id", "resolution_thread_id",
            "segment_id", "scope_fingerprint", "audience_fingerprint",
            "policy_epoch", "tool_name", "source_kind", "source_id",
            "content_digest",
        }
        data = _strict(value, keys, "evidence_reference")
        version = _text(
            data["schema_version"], "evidence_reference.schema_version"
        )
        if version != "evidence_reference.v1":
            raise ResolverContractError(
                "evidence_reference.schema_version is unsupported"
            )
        return cls(**{
            key: _text(data[key], f"evidence_reference.{key}")
            for key in keys
        })

    def to_dict(self) -> dict[str, str]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class DSHResolutionExhaustV1:
    """Canonical terminal, checkpointed, or runtime-fault exhaust."""

    KINDS: ClassVar[set[str]] = {
        "terminal", "checkpointed", "runtime_fault"
    }
    kind: str
    terminal: SubmitResolutionV1 | None = None
    evidence: tuple[EvidenceReferenceV1, ...] = ()
    identity: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    last_committed_seq: int | None = None
    checkpoint: dict[str, Any] | None = None
    fault: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: object) -> DSHResolutionExhaustV1:
        data = _mapping(value, "exhaust")
        kind = _text(data.get("kind"), "exhaust.kind")
        if kind not in cls.KINDS:
            raise ResolverContractError("exhaust.kind is unsupported")
        allowed = {"kind"}
        if kind == "terminal":
            allowed |= {
                "terminal", "evidence", "identity", "usage",
                "last_committed_seq",
            }
        elif kind == "checkpointed":
            allowed |= {"checkpoint", "identity", "last_committed_seq"}
        else:
            allowed |= {"fault", "identity", "last_committed_seq"}
        unknown = set(data) - allowed
        if unknown:
            raise ResolverContractError(
                f"exhaust has unknown fields: {sorted(unknown)}"
            )
        terminal = None
        evidence: tuple[EvidenceReferenceV1, ...] = ()
        if kind == "terminal":
            terminal = SubmitResolutionV1.from_mapping(data.get("terminal"))
            evidence_value = data.get("evidence", [])
            if not isinstance(evidence_value, list) or len(evidence_value) > 64:
                raise ResolverContractError("exhaust.evidence must be bounded")
            evidence = tuple(
                EvidenceReferenceV1.from_mapping(item)
                for item in evidence_value
            )
        identity = _optional_mapping(data.get("identity"), "exhaust.identity")
        if identity is not None and evidence:
            cls.check_evidence_bindings(
                evidence,
                resolution_thread_id=_text(
                    identity.get("resolution_thread_id"),
                    "exhaust.identity.resolution_thread_id",
                ),
                segment_id=_text(
                    identity.get("segment_id"), "exhaust.identity.segment_id"
                ),
                scope_fingerprint=_text(
                    identity.get("scope_fingerprint"),
                    "exhaust.identity.scope_fingerprint",
                ),
                audience_fingerprint=_text(
                    identity.get("audience_fingerprint"),
                    "exhaust.identity.audience_fingerprint",
                ),
                policy_epoch=_text(
                    identity.get("policy_epoch"),
                    "exhaust.identity.policy_epoch",
                ),
            )
        sequence = data.get("last_committed_seq")
        if sequence is not None:
            sequence = _integer(sequence, "exhaust.last_committed_seq")
        return cls(
            kind=kind,
            terminal=terminal,
            evidence=evidence,
            identity=identity,
            usage=_optional_mapping(data.get("usage"), "exhaust.usage"),
            last_committed_seq=sequence,
            checkpoint=_optional_mapping(
                data.get("checkpoint"), "exhaust.checkpoint"
            ),
            fault=_optional_mapping(data.get("fault"), "exhaust.fault"),
        )

    @classmethod
    def from_terminal(
        cls,
        *,
        operation_id: str,
        operation_payload_digest: str,
        request_id: str,
        resolution_thread_id: str,
        segment_id: str,
        activation_id: str,
        lease_epoch: int,
        scope_fingerprint: str,
        audience_fingerprint: str,
        resolver_profile_version: str,
        dsh_release: str,
        session_store_epoch: str,
        model_route: str,
        tool_catalog_digest: str,
        policy_epoch: str,
        terminal: SubmitResolutionV1,
        evidence: Sequence[EvidenceReferenceV1],
        last_committed_seq: int,
        usage: Mapping[str, Any] | None = None,
    ) -> DSHResolutionExhaustV1:
        identity = {
            "operation_id": operation_id,
            "operation_payload_digest": operation_payload_digest,
            "request_id": request_id,
            "resolution_thread_id": resolution_thread_id,
            "segment_id": segment_id,
            "activation_id": activation_id,
            "lease_epoch": lease_epoch,
            "scope_fingerprint": scope_fingerprint,
            "audience_fingerprint": audience_fingerprint,
            "resolver_profile_version": resolver_profile_version,
            "dsh_release": dsh_release,
            "session_store_epoch": session_store_epoch,
            "model_route": model_route,
            "tool_catalog_digest": tool_catalog_digest,
            "policy_epoch": policy_epoch,
        }
        cls.check_evidence_bindings(
            evidence,
            resolution_thread_id=resolution_thread_id,
            segment_id=segment_id,
            scope_fingerprint=scope_fingerprint,
            audience_fingerprint=audience_fingerprint,
            policy_epoch=policy_epoch,
        )
        return cls(
            kind="terminal",
            terminal=terminal,
            evidence=tuple(evidence),
            identity=identity,
            usage=dict(usage or {}),
            last_committed_seq=last_committed_seq,
        )

    @staticmethod
    def check_evidence_bindings(
        evidence: Sequence[EvidenceReferenceV1],
        *,
        resolution_thread_id: str,
        segment_id: str,
        scope_fingerprint: str,
        audience_fingerprint: str,
        policy_epoch: str,
    ) -> None:
        expected = (
            resolution_thread_id, segment_id, scope_fingerprint,
            audience_fingerprint, policy_epoch,
        )
        for reference in evidence:
            actual = (
                reference.resolution_thread_id, reference.segment_id,
                reference.scope_fingerprint, reference.audience_fingerprint,
                reference.policy_epoch,
            )
            if actual != expected:
                raise ResolverContractError("evidence authority binding mismatch")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"kind": self.kind}
        if self.kind == "terminal":
            result.update({
                "terminal": self.terminal.to_dict() if self.terminal else None,
                "evidence": [item.to_dict() for item in self.evidence],
                "identity": self.identity or {},
                "usage": self.usage or {},
                "last_committed_seq": self.last_committed_seq,
            })
        elif self.kind == "checkpointed":
            result["checkpoint"] = self.checkpoint or {}
            if self.identity is not None:
                result["identity"] = self.identity
            if self.last_committed_seq is not None:
                result["last_committed_seq"] = self.last_committed_seq
        else:
            result["fault"] = self.fault or {
                "code": RuntimeFaultCode.RPC_CONTRACT_ERROR.value
            }
            if self.identity is not None:
                result["identity"] = self.identity
            if self.last_committed_seq is not None:
                result["last_committed_seq"] = self.last_committed_seq
        return result


@dataclass(frozen=True, slots=True)
class ResolutionThreadRecordV1:
    """Strict lifecycle metadata document stored by the Mongo owner."""

    schema_version: str
    resolution_thread_id: str
    brain_conversation_ref: str
    root_goal_ref: str
    current_segment_id: str
    state: str
    priority: str
    audience_fingerprint: str
    scope_fingerprint: str
    created_at: str
    updated_at: str
    last_terminal_status: str | None
    continuation_eligible_until: str
    document_revision: int
    lease_epoch: int
    current_lease: dict[str, Any] | None
    segments: tuple[dict[str, Any], ...]
    operations: tuple[dict[str, Any], ...]

    @classmethod
    def from_mapping(cls, value: object) -> ResolutionThreadRecordV1:
        keys = {
            "schema_version", "resolution_thread_id",
            "brain_conversation_ref", "root_goal_ref", "current_segment_id",
            "state", "priority", "audience_fingerprint",
            "scope_fingerprint", "created_at", "updated_at",
            "last_terminal_status", "continuation_eligible_until",
            "document_revision", "lease_epoch", "current_lease", "segments",
            "operations",
        }
        data = _strict(value, keys, "resolution_thread")
        if data["schema_version"] != THREAD_SCHEMA_VERSION:
            raise ResolverContractError(
                "resolution_thread.schema_version is unsupported"
            )
        segments_value = data["segments"]
        operations_value = data["operations"]
        if not isinstance(segments_value, list) or not segments_value:
            raise ResolverContractError(
                "resolution_thread.segments must be a non-empty list"
            )
        if not isinstance(operations_value, list):
            raise ResolverContractError(
                "resolution_thread.operations must be a list"
            )
        segments = tuple(
            cls._validate_segment(item) for item in segments_value
        )
        current_segment_id = _text(
            data["current_segment_id"],
            "resolution_thread.current_segment_id",
        )
        if current_segment_id not in {
            segment["segment_id"] for segment in segments
        }:
            raise ResolverContractError("current_segment_id is not present")
        lease_epoch = _integer(
            data["lease_epoch"], "resolution_thread.lease_epoch"
        )
        current_lease = _optional_mapping(
            data["current_lease"], "resolution_thread.current_lease"
        )
        if current_lease is not None:
            expected_lease_keys = {
                "activation_id", "lease_epoch", "owner_id", "expires_at"
            }
            current_lease = dict(_strict(
                current_lease,
                expected_lease_keys,
                "resolution_thread.current_lease",
            ))
            if _integer(
                current_lease["lease_epoch"],
                "resolution_thread.current_lease.lease_epoch",
                1,
            ) != lease_epoch:
                raise ResolverContractError("current lease epoch does not match")
        terminal_status = data["last_terminal_status"]
        if terminal_status is not None:
            terminal_status = _text(
                terminal_status, "resolution_thread.last_terminal_status"
            )
        return cls(
            schema_version=THREAD_SCHEMA_VERSION,
            resolution_thread_id=_text(
                data["resolution_thread_id"],
                "resolution_thread.resolution_thread_id",
            ),
            brain_conversation_ref=_text(
                data["brain_conversation_ref"],
                "resolution_thread.brain_conversation_ref",
            ),
            root_goal_ref=_text(
                data["root_goal_ref"], "resolution_thread.root_goal_ref"
            ),
            current_segment_id=current_segment_id,
            state=_text(data["state"], "resolution_thread.state"),
            priority=_text(data["priority"], "resolution_thread.priority"),
            audience_fingerprint=_text(
                data["audience_fingerprint"],
                "resolution_thread.audience_fingerprint",
            ),
            scope_fingerprint=_text(
                data["scope_fingerprint"],
                "resolution_thread.scope_fingerprint",
            ),
            created_at=_text(data["created_at"], "resolution_thread.created_at"),
            updated_at=_text(data["updated_at"], "resolution_thread.updated_at"),
            last_terminal_status=terminal_status,
            continuation_eligible_until=_text(
                data["continuation_eligible_until"],
                "resolution_thread.continuation_eligible_until",
            ),
            document_revision=_integer(
                data["document_revision"],
                "resolution_thread.document_revision",
            ),
            lease_epoch=lease_epoch,
            current_lease=current_lease,
            segments=segments,
            operations=tuple(
                dict(_mapping(item, "resolution_thread.operation"))
                for item in operations_value
            ),
        )

    @staticmethod
    def _validate_segment(value: object) -> dict[str, Any]:
        keys = {
            "schema_version", "segment_id", "resolution_thread_id",
            "dsh_session_id", "resolver_profile_version", "dsh_release",
            "session_store_epoch", "tool_catalog_digest", "policy_epoch",
            "scope_fingerprint", "audience_fingerprint", "model_route",
            "state", "last_committed_seq", "parent_segment_id",
            "rotation_reason", "created_at", "last_used_at",
        }
        data = dict(_strict(value, keys, "resolution_thread.segment"))
        if data["schema_version"] != SEGMENT_SCHEMA_VERSION:
            raise ResolverContractError("segment.schema_version is unsupported")
        _integer(data["last_committed_seq"], "segment.last_committed_seq")
        for key in keys - {
            "last_committed_seq", "parent_segment_id", "rotation_reason"
        }:
            _text(data[key], f"segment.{key}")
        for key in ("parent_segment_id", "rotation_reason"):
            if data[key] is not None:
                _text(data[key], f"segment.{key}")
        return data

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "resolution_thread_id": self.resolution_thread_id,
            "brain_conversation_ref": self.brain_conversation_ref,
            "root_goal_ref": self.root_goal_ref,
            "current_segment_id": self.current_segment_id,
            "state": self.state,
            "priority": self.priority,
            "audience_fingerprint": self.audience_fingerprint,
            "scope_fingerprint": self.scope_fingerprint,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_terminal_status": self.last_terminal_status,
            "continuation_eligible_until": self.continuation_eligible_until,
            "document_revision": self.document_revision,
            "lease_epoch": self.lease_epoch,
            "current_lease": self.current_lease,
            "segments": [dict(segment) for segment in self.segments],
            "operations": [dict(operation) for operation in self.operations],
        }
