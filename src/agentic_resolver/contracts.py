"""Strict DTOs for the standalone DSH V2 resolver boundary."""

from __future__ import annotations

import ntpath
import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from agentic_resolver.errors import ResolverContractError, RuntimeFaultCode

RPC_PROTOCOL_VERSION_V2 = "kazusa.dsh-resolution-rpc.v2"
RPC_PROTOCOL_VERSION = RPC_PROTOCOL_VERSION_V2
INTAKE_SCHEMA_VERSION = "dsh_resolution_intake.v2"
THREAD_SCHEMA_VERSION = "resolution_thread_store.v2"
SEGMENT_SCHEMA_VERSION = "resolver_session_segment.v2"
PROFILE_VERSION = "kazusa-resolver-standard-v2"
DSH_RELEASE = "0.1.1-rc.2"
SESSION_STORE_EPOCH = "dsh-sqlite-0.1.1-rc.2-standard-v2"
EVIDENCE_SCHEMA_VERSION = "evidence_receipt.v2"


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


def _canonical_workspace(value: object, field: str) -> str:
    path = _text(value, field)
    windows_absolute = ntpath.isabs(path)
    posix_absolute = posixpath.isabs(path)
    if not windows_absolute and not posix_absolute:
        raise ResolverContractError(f"{field} must be an absolute path")
    normalizer = ntpath.normpath if windows_absolute else posixpath.normpath
    normalized = normalizer(path)
    if normalized.replace("\\", "/") != path.replace("\\", "/"):
        raise ResolverContractError(f"{field} must be canonical")
    return path.replace("\\", "/")


class DSHResolutionModelInputV2(dict[str, object]):
    """The only model-visible input accepted by the V2 resolver."""

    @classmethod
    def from_mapping(cls, value: object) -> DSHResolutionModelInputV2:
        data = _strict(value, {"objective", "facts"}, "model_input")
        return cls(
            objective=_text(data["objective"], "model_input.objective"),
            facts=list(_texts(data["facts"], "model_input.facts")),
        )

    @property
    def objective(self) -> str:
        return str(self["objective"])

    @property
    def facts(self) -> tuple[str, ...]:
        value = self["facts"]
        if not isinstance(value, list):
            raise ResolverContractError("model_input.facts must be a list")
        return tuple(value)

    def to_dict(self) -> dict[str, object]:
        return {"objective": self.objective, "facts": list(self.facts)}


@dataclass(frozen=True, slots=True)
class DSHResolutionRuntimeV2:
    """Model-hidden V2 authority identity carried by an intake."""

    request_id: str
    operation_id: str
    operation_payload_digest: str
    resolution_thread_id: str
    segment_id: str
    brain_conversation_ref: str
    workspace_root: str
    route_digest: str
    semantic_tool_authority: dict[str, str]
    interaction_authority: dict[str, str]

    @classmethod
    def from_mapping(cls, value: object) -> DSHResolutionRuntimeV2:
        keys = {
            "request_id",
            "operation_id",
            "operation_payload_digest",
            "resolution_thread_id",
            "segment_id",
            "brain_conversation_ref",
            "workspace_root",
            "route_digest",
            "semantic_tool_authority",
            "interaction_authority",
        }
        data = _strict(value, keys, "runtime")
        semantic = _strict(
            data["semantic_tool_authority"],
            {"catalog_digest", "token"},
            "runtime.semantic_tool_authority",
        )
        interaction = _strict(
            data["interaction_authority"],
            {"issuer", "scope_fingerprint", "audience_fingerprint"},
            "runtime.interaction_authority",
        )
        return cls(
            request_id=_text(data["request_id"], "runtime.request_id"),
            operation_id=_text(data["operation_id"], "runtime.operation_id"),
            operation_payload_digest=_text(
                data["operation_payload_digest"],
                "runtime.operation_payload_digest",
            ),
            resolution_thread_id=_text(
                data["resolution_thread_id"],
                "runtime.resolution_thread_id",
            ),
            segment_id=_text(data["segment_id"], "runtime.segment_id"),
            brain_conversation_ref=_text(
                data["brain_conversation_ref"],
                "runtime.brain_conversation_ref",
            ),
            workspace_root=_canonical_workspace(
                data["workspace_root"], "runtime.workspace_root"
            ),
            route_digest=_text(data["route_digest"], "runtime.route_digest"),
            semantic_tool_authority={
                "catalog_digest": _text(
                    semantic["catalog_digest"],
                    "runtime.semantic_tool_authority.catalog_digest",
                ),
                "token": _text(
                    semantic["token"],
                    "runtime.semantic_tool_authority.token",
                ),
            },
            interaction_authority={
                "issuer": _text(
                    interaction["issuer"],
                    "runtime.interaction_authority.issuer",
                ),
                "scope_fingerprint": _text(
                    interaction["scope_fingerprint"],
                    "runtime.interaction_authority.scope_fingerprint",
                ),
                "audience_fingerprint": _text(
                    interaction["audience_fingerprint"],
                    "runtime.interaction_authority.audience_fingerprint",
                ),
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "operation_id": self.operation_id,
            "operation_payload_digest": self.operation_payload_digest,
            "resolution_thread_id": self.resolution_thread_id,
            "segment_id": self.segment_id,
            "brain_conversation_ref": self.brain_conversation_ref,
            "workspace_root": self.workspace_root,
            "route_digest": self.route_digest,
            "semantic_tool_authority": dict(self.semantic_tool_authority),
            "interaction_authority": dict(self.interaction_authority),
        }


@dataclass(frozen=True, slots=True)
class DSHResolutionIntakeV2:
    """Strict V2 sidecar intake separating host authority from model input."""

    MODES: ClassVar[frozenset[str]] = frozenset({"start", "continue"})
    schema_version: str
    mode: str
    request_id: str
    operation_id: str
    operation_payload_digest: str
    resolution_thread_id: str
    segment_id: str
    brain_conversation_ref: str
    workspace_root: str
    route_digest: str
    model_input: DSHResolutionModelInputV2
    semantic_tool_authority: dict[str, str]
    interaction_authority: dict[str, str]

    @classmethod
    def from_mapping(cls, value: object) -> DSHResolutionIntakeV2:
        keys = {
            "schema_version",
            "mode",
            "request_id",
            "operation_id",
            "operation_payload_digest",
            "resolution_thread_id",
            "segment_id",
            "brain_conversation_ref",
            "workspace_root",
            "route_digest",
            "model_input",
            "semantic_tool_authority",
            "interaction_authority",
        }
        data = _strict(value, keys, "intake")
        version = _text(data["schema_version"], "intake.schema_version")
        if version != INTAKE_SCHEMA_VERSION:
            raise ResolverContractError("intake.schema_version is unsupported")
        mode = _text(data["mode"], "intake.mode")
        if mode not in cls.MODES:
            raise ResolverContractError("intake.mode is unsupported")
        runtime = DSHResolutionRuntimeV2.from_mapping({
            key: data[key]
            for key in (
                "request_id",
                "operation_id",
                "operation_payload_digest",
                "resolution_thread_id",
                "segment_id",
                "brain_conversation_ref",
                "workspace_root",
                "route_digest",
                "semantic_tool_authority",
                "interaction_authority",
            )
        })
        return cls(
            schema_version=version,
            mode=mode,
            request_id=runtime.request_id,
            operation_id=runtime.operation_id,
            operation_payload_digest=runtime.operation_payload_digest,
            resolution_thread_id=runtime.resolution_thread_id,
            segment_id=runtime.segment_id,
            brain_conversation_ref=runtime.brain_conversation_ref,
            workspace_root=runtime.workspace_root,
            route_digest=runtime.route_digest,
            model_input=DSHResolutionModelInputV2.from_mapping(
                data["model_input"]
            ),
            semantic_tool_authority=dict(runtime.semantic_tool_authority),
            interaction_authority=dict(runtime.interaction_authority),
        )

    @property
    def model_visible_input(self) -> dict[str, object]:
        """Return the deliberately small model-visible projection."""

        return self.model_input.to_dict()

    def runtime_authority(self) -> DSHResolutionRuntimeV2:
        return DSHResolutionRuntimeV2(
            request_id=self.request_id,
            operation_id=self.operation_id,
            operation_payload_digest=self.operation_payload_digest,
            resolution_thread_id=self.resolution_thread_id,
            segment_id=self.segment_id,
            brain_conversation_ref=self.brain_conversation_ref,
            workspace_root=self.workspace_root,
            route_digest=self.route_digest,
            semantic_tool_authority=dict(self.semantic_tool_authority),
            interaction_authority=dict(self.interaction_authority),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            **self.runtime_authority().to_dict(),
            "model_input": self.model_input.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SubmitResolutionV2:
    """Validated terminal semantic product returned by the terminal action."""

    STATUSES: ClassVar[frozenset[str]] = frozenset({
        "resolved",
        "partial",
        "needs_user_input",
        "approval_required",
        "unavailable",
        "failed",
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
    def from_mapping(cls, value: object) -> SubmitResolutionV2:
        keys = {
            "status",
            "summary",
            "findings",
            "completed_subgoals",
            "remaining_needs",
            "clarification_request",
            "approval_request",
            "artifact_refs",
            "warnings",
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
            data["approval_request"],
            "submit_resolution.approval_request",
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
                data["remaining_needs"],
                "submit_resolution.remaining_needs",
            ),
            clarification_request=clarification,
            approval_request=approval,
            artifact_refs=_texts(
                data["artifact_refs"],
                "submit_resolution.artifact_refs",
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
class EvidenceReceiptV2:
    """Public evidence projection with no backend identifiers."""

    schema_version: str
    resolution_thread_id: str
    segment_id: str
    scope_fingerprint: str
    audience_fingerprint: str
    policy_epoch: str
    evidence_id: str
    source_kind: str
    semantic_ref: str
    content_digest: str
    provenance: dict[str, str]

    @classmethod
    def from_mapping(cls, value: object) -> EvidenceReceiptV2:
        keys = {
            "schema_version",
            "resolution_thread_id",
            "segment_id",
            "scope_fingerprint",
            "audience_fingerprint",
            "policy_epoch",
            "evidence_id",
            "source_kind",
            "semantic_ref",
            "content_digest",
            "provenance",
        }
        data = _strict(value, keys, "evidence_receipt")
        version = _text(
            data["schema_version"], "evidence_receipt.schema_version"
        )
        if version != EVIDENCE_SCHEMA_VERSION:
            raise ResolverContractError(
                "evidence_receipt.schema_version is unsupported"
            )
        provenance = _strict(
            data["provenance"], {"tool_name"}, "evidence_receipt.provenance"
        )
        return cls(
            schema_version=version,
            resolution_thread_id=_text(
                data["resolution_thread_id"],
                "evidence_receipt.resolution_thread_id",
            ),
            segment_id=_text(data["segment_id"], "evidence_receipt.segment_id"),
            scope_fingerprint=_text(
                data["scope_fingerprint"],
                "evidence_receipt.scope_fingerprint",
            ),
            audience_fingerprint=_text(
                data["audience_fingerprint"],
                "evidence_receipt.audience_fingerprint",
            ),
            policy_epoch=_text(
                data["policy_epoch"], "evidence_receipt.policy_epoch"
            ),
            evidence_id=_text(data["evidence_id"], "evidence_receipt.evidence_id"),
            source_kind=_text(
                data["source_kind"], "evidence_receipt.source_kind"
            ),
            semantic_ref=_text(
                data["semantic_ref"], "evidence_receipt.semantic_ref"
            ),
            content_digest=_text(
                data["content_digest"], "evidence_receipt.content_digest"
            ),
            provenance={
                "tool_name": _text(
                    provenance["tool_name"],
                    "evidence_receipt.provenance.tool_name",
                )
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "resolution_thread_id": self.resolution_thread_id,
            "segment_id": self.segment_id,
            "scope_fingerprint": self.scope_fingerprint,
            "audience_fingerprint": self.audience_fingerprint,
            "policy_epoch": self.policy_epoch,
            "evidence_id": self.evidence_id,
            "source_kind": self.source_kind,
            "semantic_ref": self.semantic_ref,
            "content_digest": self.content_digest,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class DSHResolutionExhaustV2:
    """Canonical terminal, checkpointed, canceled, or faulted exhaust."""

    KINDS: ClassVar[frozenset[str]] = frozenset({
        "terminal", "checkpointed", "runtime_fault", "canceled"
    })
    kind: str
    terminal: SubmitResolutionV2 | None = None
    evidence: tuple[EvidenceReceiptV2, ...] = ()
    identity: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    last_committed_seq: int | None = None
    checkpoint: dict[str, Any] | None = None
    fault: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: object) -> DSHResolutionExhaustV2:
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
        elif kind == "canceled":
            allowed |= {"identity", "last_committed_seq"}
        else:
            allowed |= {"fault", "identity", "last_committed_seq"}
        unknown = set(data) - allowed
        if unknown:
            raise ResolverContractError(
                f"exhaust has unknown fields: {sorted(unknown)}"
            )
        terminal = None
        evidence: tuple[EvidenceReceiptV2, ...] = ()
        if kind == "terminal":
            terminal = SubmitResolutionV2.from_mapping(data.get("terminal"))
            evidence_value = data.get("evidence")
            if not isinstance(evidence_value, list) or len(evidence_value) > 64:
                raise ResolverContractError("exhaust.evidence must be bounded")
            evidence = tuple(
                EvidenceReceiptV2.from_mapping(item) for item in evidence_value
            )
        identity = _optional_mapping(data.get("identity"), "exhaust.identity")
        if identity is not None and evidence:
            required = {
                key: _text(identity.get(key), f"exhaust.identity.{key}")
                for key in (
                    "resolution_thread_id",
                    "segment_id",
                    "scope_fingerprint",
                    "audience_fingerprint",
                    "policy_epoch",
                )
                if key in identity
            }
            if len(required) == 5:
                cls.check_evidence_bindings(evidence, **required)
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
        resolution_thread_id: str,
        segment_id: str,
        scope_fingerprint: str,
        audience_fingerprint: str,
        policy_epoch: str,
        terminal: SubmitResolutionV2,
        evidence: Sequence[EvidenceReceiptV2],
        last_committed_seq: int | None = None,
        **claims: object,
    ) -> DSHResolutionExhaustV2:
        thread = _text(resolution_thread_id, "resolution_thread_id")
        segment = _text(segment_id, "segment_id")
        scope = _text(scope_fingerprint, "scope_fingerprint")
        audience = _text(audience_fingerprint, "audience_fingerprint")
        policy = _text(policy_epoch, "policy_epoch")
        if not isinstance(terminal, SubmitResolutionV2):
            raise ResolverContractError("terminal must be SubmitResolutionV2")
        references = tuple(evidence)
        cls.check_evidence_bindings(
            references,
            resolution_thread_id=thread,
            segment_id=segment,
            scope_fingerprint=scope,
            audience_fingerprint=audience,
            policy_epoch=policy,
        )
        identity: dict[str, Any] = {
            "resolution_thread_id": thread,
            "segment_id": segment,
            "scope_fingerprint": scope,
            "audience_fingerprint": audience,
            "policy_epoch": policy,
        }
        for key, value in claims.items():
            if value is not None:
                identity[key] = value
        return cls(
            kind="terminal",
            terminal=terminal,
            evidence=references,
            identity=identity,
            usage={},
            last_committed_seq=(
                _integer(last_committed_seq, "last_committed_seq")
                if last_committed_seq is not None
                else None
            ),
        )

    @staticmethod
    def check_evidence_bindings(
        evidence: Sequence[EvidenceReceiptV2],
        *,
        resolution_thread_id: str,
        segment_id: str,
        scope_fingerprint: str,
        audience_fingerprint: str,
        policy_epoch: str,
    ) -> None:
        expected = (
            resolution_thread_id,
            segment_id,
            scope_fingerprint,
            audience_fingerprint,
            policy_epoch,
        )
        for reference in evidence:
            if not isinstance(reference, EvidenceReceiptV2):
                raise ResolverContractError("evidence reference is invalid")
            actual = (
                reference.resolution_thread_id,
                reference.segment_id,
                reference.scope_fingerprint,
                reference.audience_fingerprint,
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
        elif self.kind == "canceled":
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
class ResolutionThreadRecordV2:
    """Strict durable continuity metadata for the V2 Mongo owner."""

    PRIORITIES: ClassVar[frozenset[str]] = frozenset({"now", "background"})
    schema_version: str
    resolution_thread_id: str
    brain_conversation_ref: str
    root_goal_ref: str
    current_segment_id: str
    state: str
    priority: str
    workspace_root: str
    workspace_fingerprint: str
    route_digest: str
    profile_version: str
    dsh_release: str
    session_store_epoch: str
    standard_catalog_digest: str
    semantic_catalog_digest: str
    policy_epoch: str
    scope_fingerprint: str
    audience_fingerprint: str
    interaction_id: str
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
    def from_mapping(cls, value: object) -> ResolutionThreadRecordV2:
        keys = {
            "schema_version",
            "resolution_thread_id",
            "brain_conversation_ref",
            "root_goal_ref",
            "current_segment_id",
            "state",
            "priority",
            "workspace_root",
            "workspace_fingerprint",
            "route_digest",
            "profile_version",
            "dsh_release",
            "session_store_epoch",
            "standard_catalog_digest",
            "semantic_catalog_digest",
            "policy_epoch",
            "scope_fingerprint",
            "audience_fingerprint",
            "interaction_id",
            "created_at",
            "updated_at",
            "last_terminal_status",
            "continuation_eligible_until",
            "document_revision",
            "lease_epoch",
            "current_lease",
            "segments",
            "operations",
        }
        data = _strict(value, keys, "resolution_thread")
        if data["schema_version"] != THREAD_SCHEMA_VERSION:
            raise ResolverContractError(
                "resolution_thread.schema_version is unsupported"
            )
        priority = _text(data["priority"], "resolution_thread.priority")
        if priority not in cls.PRIORITIES:
            raise ResolverContractError("resolution_thread.priority is unsupported")
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
        segments = tuple(cls._validate_segment(item) for item in segments_value)
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
            current_lease = dict(_strict(
                current_lease,
                {"activation_id", "lease_epoch", "owner_id", "expires_at"},
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
        text_fields = (
            "resolution_thread.resolution_thread_id",
            "resolution_thread.brain_conversation_ref",
            "resolution_thread.root_goal_ref",
            "resolution_thread.workspace_fingerprint",
            "resolution_thread.route_digest",
            "resolution_thread.profile_version",
            "resolution_thread.dsh_release",
            "resolution_thread.session_store_epoch",
            "resolution_thread.standard_catalog_digest",
            "resolution_thread.semantic_catalog_digest",
            "resolution_thread.policy_epoch",
            "resolution_thread.scope_fingerprint",
            "resolution_thread.audience_fingerprint",
            "resolution_thread.interaction_id",
            "resolution_thread.created_at",
            "resolution_thread.updated_at",
            "resolution_thread.continuation_eligible_until",
        )
        values = (
            data["resolution_thread_id"],
            data["brain_conversation_ref"],
            data["root_goal_ref"],
            data["workspace_fingerprint"],
            data["route_digest"],
            data["profile_version"],
            data["dsh_release"],
            data["session_store_epoch"],
            data["standard_catalog_digest"],
            data["semantic_catalog_digest"],
            data["policy_epoch"],
            data["scope_fingerprint"],
            data["audience_fingerprint"],
            data["interaction_id"],
            data["created_at"],
            data["updated_at"],
            data["continuation_eligible_until"],
        )
        for value_item, field in zip(values, text_fields, strict=True):
            _text(value_item, field)
        return cls(
            schema_version=THREAD_SCHEMA_VERSION,
            resolution_thread_id=str(data["resolution_thread_id"]),
            brain_conversation_ref=str(data["brain_conversation_ref"]),
            root_goal_ref=str(data["root_goal_ref"]),
            current_segment_id=current_segment_id,
            state=_text(data["state"], "resolution_thread.state"),
            priority=priority,
            workspace_root=_canonical_workspace(
                data["workspace_root"], "resolution_thread.workspace_root"
            ),
            workspace_fingerprint=str(data["workspace_fingerprint"]),
            route_digest=str(data["route_digest"]),
            profile_version=str(data["profile_version"]),
            dsh_release=str(data["dsh_release"]),
            session_store_epoch=str(data["session_store_epoch"]),
            standard_catalog_digest=str(data["standard_catalog_digest"]),
            semantic_catalog_digest=str(data["semantic_catalog_digest"]),
            policy_epoch=str(data["policy_epoch"]),
            scope_fingerprint=str(data["scope_fingerprint"]),
            audience_fingerprint=str(data["audience_fingerprint"]),
            interaction_id=str(data["interaction_id"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            last_terminal_status=terminal_status,
            continuation_eligible_until=str(data["continuation_eligible_until"]),
            document_revision=_integer(
                data["document_revision"], "resolution_thread.document_revision"
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
            "schema_version",
            "segment_id",
            "resolution_thread_id",
            "dsh_session_id",
            "brain_conversation_ref",
            "workspace_root",
            "workspace_fingerprint",
            "route_digest",
            "resolver_profile_version",
            "dsh_release",
            "session_store_epoch",
            "standard_catalog_digest",
            "semantic_catalog_digest",
            "policy_epoch",
            "scope_fingerprint",
            "audience_fingerprint",
            "interaction_id",
            "state",
            "last_committed_seq",
            "parent_segment_id",
            "rotation_reason",
            "created_at",
            "last_used_at",
        }
        data = dict(_strict(value, keys, "resolution_thread.segment"))
        if data["schema_version"] != SEGMENT_SCHEMA_VERSION:
            raise ResolverContractError("segment.schema_version is unsupported")
        _integer(data["last_committed_seq"], "segment.last_committed_seq")
        _canonical_workspace(data["workspace_root"], "segment.workspace_root")
        nullable = {"parent_segment_id", "rotation_reason"}
        for key in keys - nullable - {"last_committed_seq", "workspace_root"}:
            _text(data[key], f"segment.{key}")
        for key in nullable:
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
            "workspace_root": self.workspace_root,
            "workspace_fingerprint": self.workspace_fingerprint,
            "route_digest": self.route_digest,
            "profile_version": self.profile_version,
            "dsh_release": self.dsh_release,
            "session_store_epoch": self.session_store_epoch,
            "standard_catalog_digest": self.standard_catalog_digest,
            "semantic_catalog_digest": self.semantic_catalog_digest,
            "policy_epoch": self.policy_epoch,
            "scope_fingerprint": self.scope_fingerprint,
            "audience_fingerprint": self.audience_fingerprint,
            "interaction_id": self.interaction_id,
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
