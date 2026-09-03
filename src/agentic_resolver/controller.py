"""Standalone resolution lifecycle controller."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from inspect import isawaitable
from typing import Any, Protocol
from uuid import uuid4

from agentic_resolver.contracts import (
    PROFILE_VERSION,
    SEGMENT_SCHEMA_VERSION,
    DSHResolutionIntakeV2,
)
from agentic_resolver.errors import (
    OperationIdReuseMismatchError,
    OperationOutcomeUncertainError,
    RpcContractError,
)
from agentic_resolver.fingerprints import (
    operation_payload_digest,
    workspace_fingerprint,
)
from agentic_resolver.persistence import ResolutionThreadRepository
from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
    SemanticActivationAuthorityV1,
    activation_id_for,
    issue_activation_token,
    verify_activation_token,
)
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest


class ResolutionRpc(Protocol):
    async def call(
        self,
        method: str,
        params: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call one sidecar lifecycle method."""


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _expires() -> str:
    return (datetime.now(UTC) + timedelta(seconds=30)).isoformat().replace(
        "+00:00", "Z"
    )


_ALLOW_ONCE_CONTINUATION_FACT = (
    "The earlier native approval cancellation was the checkpoint transport "
    "outcome and is superseded by Brain's one-shot approval. One short-lived "
    "grant is available only for a fresh call to the same native tool with "
    "semantically identical executable arguments. Retry that exact native "
    "call now with a fresh call id so the gateway can validate and atomically "
    "consume the grant."
)


def _continuation_model_facts(
    continuation_delta: Mapping[str, Any],
) -> list[str]:
    """Project a typed continuation decision into bounded model-visible facts."""

    if continuation_delta.get("decision") != "allow_once":
        return []
    return [_ALLOW_ONCE_CONTINUATION_FACT]


class ResolutionController:
    """Own thread compatibility, semantic operations, and lease fencing."""

    _COMPATIBILITY_FIELDS = (
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
    )
    _LEASE_RENEWAL_SECONDS = 10.0
    _COMMITTED_DISPOSITIONS = frozenset({
        "terminal", "checkpointed", "canceled", "faulted",
    })

    def __init__(
        self,
        repository: ResolutionThreadRepository,
        rpc: ResolutionRpc,
        *,
        owner_id: str,
        semantic_authority_secret: bytes,
    ) -> None:
        if not isinstance(semantic_authority_secret, bytes) or not semantic_authority_secret:
            raise ValueError("semantic_authority_secret is required")
        self._repository = repository
        self._rpc = rpc
        self._owner_id = owner_id
        self._semantic_authority_secret = semantic_authority_secret
        self._observed_states: dict[str, str] = {}
        self._disposal_lock = asyncio.Lock()

    @staticmethod
    async def recover_after_runtime_fault(
        fault: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return the durable identity needed to replay a DSH fault safely."""

        if not isinstance(fault, Mapping):
            raise TypeError("runtime fault must be an object")
        if fault.get("schema_version") != "dsh_runtime_fault.v1":
            raise ValueError("runtime fault schema is unsupported")
        required = (
            "resolution_thread_id",
            "segment_id",
            "dsh_session_id",
            "document_revision",
            "last_committed_seq",
            "fault_code",
        )
        if any(field not in fault for field in required):
            raise ValueError("runtime fault identity is incomplete")
        for field in required[:3] + ("fault_code",):
            value = fault[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"runtime fault {field} is invalid")
        for field in ("document_revision", "last_committed_seq"):
            value = fault[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"runtime fault {field} is invalid")
        return {
            "disposition": "recovery_required",
            "resolution_thread_id": fault["resolution_thread_id"],
            "segment_id": fault["segment_id"],
            "dsh_session_id": fault["dsh_session_id"],
            "document_revision": fault["document_revision"],
            "last_committed_seq": fault["last_committed_seq"],
            "fault_code": fault["fault_code"],
        }

    async def resolve(self, value: Mapping[str, Any]) -> dict[str, Any]:
        intake = DSHResolutionIntakeV2.from_mapping(value)
        if intake.mode == "start":
            return await self.open(intake.to_dict())
        return await self.continue_resolution(intake.to_dict())

    async def readiness(self) -> dict[str, str]:
        """Return the authenticated mounted-sidecar identity."""

        return await self._health_identity()

    async def open(self, value: Mapping[str, Any]) -> dict[str, Any]:
        intake = DSHResolutionIntakeV2.from_mapping(value)
        health = await self._health_identity()
        self._assert_intake_health(intake, health)
        self._verify_intake_authority(intake, health)
        record = await self._repository_call(
            "get_thread", intake.resolution_thread_id
        )
        if record is None:
            self._verify_intake_authority(
                intake,
                health,
                activation_id=activation_id_for(
                    intake.resolution_thread_id, intake.segment_id, 1
                ),
                lease_epoch=1,
            )
            segment = self._segment(
                intake,
                health,
                segment_id=intake.segment_id,
            )
            await self._repository_call(
                "create_thread_v2",
                resolution_thread_id=intake.resolution_thread_id,
                brain_conversation_ref=intake.brain_conversation_ref,
                root_goal_ref=intake.model_input.objective,
                priority="now",
                workspace_root=intake.workspace_root,
                workspace_fingerprint=workspace_fingerprint(
                    intake.workspace_root
                ),
                route_digest=intake.route_digest,
                profile_version=PROFILE_VERSION,
                standard_catalog_digest=health["native_catalog_digest"],
                semantic_catalog_digest=health["semantic_catalog_digest"],
                scope_fingerprint=(
                    intake.interaction_authority["scope_fingerprint"]
                ),
                audience_fingerprint=(
                    intake.interaction_authority["audience_fingerprint"]
                ),
                policy_epoch="dsh-standard-policy-v2",
                interaction_id=intake.request_id,
                segment=segment,
                now=str(segment["created_at"]),
            )
        return await self._activate("resolution.open", intake)

    async def continue_resolution(
        self, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        intake = DSHResolutionIntakeV2.from_mapping(value)
        health = await self._health_identity()
        self._assert_intake_health(intake, health)
        self._verify_intake_authority(intake, health)
        record = await self._repository_call(
            "get_thread", intake.resolution_thread_id
        )
        if record is None:
            raise RuntimeError("V2 resolution thread is not admitted")
        self._verify_intake_authority(
            intake,
            health,
            activation_id=activation_id_for(
                intake.resolution_thread_id,
                intake.segment_id,
                int(record.lease_epoch) + 1,
            ),
            lease_epoch=int(record.lease_epoch) + 1,
        )
        current = next(
            segment
            for segment in record.segments
            if segment["segment_id"] == record.current_segment_id
        )
        candidate = self._compatibility_values(intake, health)
        mismatch = next(
            (
                field
                for field in self._COMPATIBILITY_FIELDS
                if current[field] != candidate[field]
            ),
            None,
        )
        if mismatch is None and current["segment_id"] != intake.segment_id:
            raise RuntimeError("SEGMENT_ID_MISMATCH")
        if mismatch is not None:
            # A new segment is authenticated by the caller's activation token;
            # the controller preserves that exact hidden identity through the
            # durable rotation rather than inventing an unauthenticated id.
            rotated_segment_id = intake.segment_id
            await self._repository_call(
                "rotate_segment",
                intake.resolution_thread_id,
                self._segment(
                    intake,
                    health,
                    segment_id=rotated_segment_id,
                ),
                reason=f"{mismatch}_mismatch",
            )
            intake = replace(
                intake,
                segment_id=rotated_segment_id,
            )
        return await self._activate("resolution.continue", intake)

    async def amend(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
        continuation_delta: Mapping[str, Any],
    ) -> dict[str, Any]:
        record = await self._fence(
            resolution_thread_id, activation_id, lease_epoch
        )
        operation_id = f"op_{uuid4().hex}"
        digest = f"sha256:{operation_id}"
        await self._repository_call(
            "prepare_operation",
            resolution_thread_id,
            operation_id,
            digest,
            "resolution.amend",
            record.current_segment_id,
            activation_id,
            lease_epoch,
        )
        result = await self._rpc.call(
            "resolution.amend",
            {
                "operation_id": operation_id,
                "operation_payload_digest": digest,
                "resolution_thread_id": resolution_thread_id,
                "segment_id": record.current_segment_id,
                "activation_id": activation_id,
                "lease_epoch": lease_epoch,
                "amendment": dict(continuation_delta),
            },
            operation_id=operation_id,
            operation_payload_digest=digest,
        )
        await self._repository_call(
            "update_operation",
            resolution_thread_id,
            operation_id,
            disposition=str(result.get("disposition", "faulted")),
            fault_code=self._fault_code(result),
        )
        return result

    async def request_checkpoint(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> dict[str, Any]:
        return await self._control(
            "resolution.request_checkpoint",
            resolution_thread_id,
            activation_id,
            lease_epoch,
        )

    async def continue_after_terminal(
        self,
        *,
        resolution_thread_id: str,
        segment_id: str,
        activation_id: str,
        lease_epoch: int,
        continuation_delta: Mapping[str, Any],
        execution_context: Mapping[str, Any] | None = None,
        start_spec: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Open one fresh-fenced continuation on the existing segment."""

        if not isinstance(continuation_delta, Mapping):
            raise TypeError("continuation_delta must be an object")
        del activation_id, lease_epoch
        record = await self._repository_call(
            "get_thread", resolution_thread_id
        )
        if record is None:
            raise RuntimeError("V2 resolution thread is not admitted")
        if record.current_segment_id != segment_id:
            raise RuntimeError("SEGMENT_ID_MISMATCH")
        context = self._continuation_context(execution_context, start_spec)
        intake = await self._fresh_continuation_intake(
            record=record,
            resolution_thread_id=resolution_thread_id,
            segment_id=segment_id,
            continuation_delta=continuation_delta,
            execution_context=context,
        )
        result = await self.continue_resolution(intake)
        return {**result, "fresh_authority": True}

    async def continue_after_checkpoint(
        self,
        *,
        resolution_thread_id: str,
        segment_id: str,
        activation_id: str,
        lease_epoch: int,
        continuation_delta: Mapping[str, Any],
        execution_context: Mapping[str, Any] | None = None,
        start_spec: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Continue one checkpoint with a fresh lease on the same segment."""

        return await self.continue_after_terminal(
            resolution_thread_id=resolution_thread_id,
            segment_id=segment_id,
            activation_id=activation_id,
            lease_epoch=lease_epoch,
            continuation_delta=continuation_delta,
            execution_context=execution_context,
            start_spec=start_spec,
        )

    @staticmethod
    def _continuation_context(
        execution_context: Mapping[str, Any] | None,
        start_spec: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        """Select the trusted scope carrier for a fresh continuation token."""

        if execution_context is not None:
            if not isinstance(execution_context, Mapping):
                raise TypeError("execution_context must be an object")
            return execution_context
        if isinstance(start_spec, Mapping):
            candidate = start_spec.get("execution_context")
            if isinstance(candidate, Mapping):
                return candidate
        raise ValueError("continuation execution context is required")

    async def _fresh_continuation_intake(
        self,
        *,
        record: Any,
        resolution_thread_id: str,
        segment_id: str,
        continuation_delta: Mapping[str, Any],
        execution_context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build a same-thread V2 intake with a newly issued lease authority."""

        health = await self._health_identity()
        scope_values = {
            "platform": execution_context.get("platform"),
            "platform_channel_id": execution_context.get("channel_id"),
            "global_user_id": execution_context.get("requester_global_user_id"),
        }
        if any(
            not isinstance(value, str) or not value.strip()
            for value in scope_values.values()
        ):
            raise ValueError("continuation execution scope is incomplete")
        service_scope = {
            key: str(value).strip()
            for key, value in scope_values.items()
        }
        if content_digest(service_scope) != record.scope_fingerprint:
            raise RuntimeError("SCOPE_FINGERPRINT_MISMATCH")

        workspace_root = str(record.workspace_root)
        workspace_digest = workspace_fingerprint(workspace_root)
        if workspace_digest != record.workspace_fingerprint:
            raise RuntimeError("WORKSPACE_FINGERPRINT_MISMATCH")
        next_lease_epoch = int(record.lease_epoch) + 1
        next_activation_id = activation_id_for(
            resolution_thread_id,
            segment_id,
            next_lease_epoch,
        )
        operation_id = f"op_{uuid4().hex}"
        payload_digest = operation_payload_digest({
            "method": "resolution.continue",
            "params": {
                "resolution_thread_id": resolution_thread_id,
                "segment_id": segment_id,
                "continuation": dict(continuation_delta),
            },
        })
        objective = next(
            (
                value.strip()
                for field in (
                    "answer",
                    "response_goal",
                    "objective",
                    "instruction",
                )
                if isinstance(value := continuation_delta.get(field), str)
                and value.strip()
            ),
            str(record.root_goal_ref),
        )
        interaction_issuer = "brain.task_resolution"
        audience_fingerprint = str(record.audience_fingerprint)
        issued_reference_digest = content_digest({
            "resolution_thread_id": resolution_thread_id,
            "segment_id": segment_id,
            "brain_conversation_ref": record.brain_conversation_ref,
            "service_scope": service_scope,
            "audience_fingerprint": audience_fingerprint,
        })
        now = datetime.now(UTC)
        issued_at = now.isoformat().replace("+00:00", "Z")
        expires_at = (now + timedelta(minutes=5)).isoformat().replace(
            "+00:00", "Z"
        )
        authority = SemanticActivationAuthorityV1(
            activation_id=next_activation_id,
            lease_epoch=next_lease_epoch,
            resolution_thread_id=resolution_thread_id,
            segment_id=segment_id,
            brain_conversation_ref=str(record.brain_conversation_ref),
            service_scope=service_scope,
            scope_fingerprint=str(record.scope_fingerprint),
            audience_fingerprint=audience_fingerprint,
            workspace_root=workspace_root,
            route_digest=health["route_digest"],
            catalog_digest=health["semantic_catalog_digest"],
            profile_version=health["profile_version"],
            model_route_digest=health["route_digest"],
            workspace_fingerprint=workspace_digest,
            issued_reference_digest=issued_reference_digest,
            policy_epoch=health["policy_epoch"],
            interaction_issuer=interaction_issuer,
            issued_at=issued_at,
            expires_at=expires_at,
            token_id=f"tok_{uuid4().hex}",
            nonce=f"nonce_{uuid4().hex}",
        )
        token = issue_activation_token(
            authority,
            secret=self._semantic_authority_secret,
            now=issued_at,
        )
        return DSHResolutionIntakeV2.from_mapping({
            "schema_version": "dsh_resolution_intake.v2",
            "mode": "continue",
            "request_id": str(record.interaction_id),
            "operation_id": operation_id,
            "operation_payload_digest": payload_digest,
            "resolution_thread_id": resolution_thread_id,
            "segment_id": segment_id,
            "brain_conversation_ref": str(record.brain_conversation_ref),
            "workspace_root": workspace_root,
            "route_digest": health["route_digest"],
            "model_input": {
                "objective": objective,
                "facts": _continuation_model_facts(continuation_delta),
            },
            "semantic_tool_authority": {
                "catalog_digest": health["semantic_catalog_digest"],
                "token": token,
            },
            "interaction_authority": {
                "issuer": interaction_issuer,
                "scope_fingerprint": str(record.scope_fingerprint),
                "audience_fingerprint": audience_fingerprint,
            },
        }).to_dict()

    async def interaction_checkpoint(
        self,
        *,
        resolution_thread_id: str,
        segment_id: str,
        activation_id: str,
        lease_epoch: int,
        interaction_id: str,
    ) -> dict[str, Any]:
        """Checkpoint one active segment under a Brain interaction identity."""

        payload = {
            "resolution_thread_id": resolution_thread_id,
            "segment_id": segment_id,
            "activation_id": activation_id,
            "lease_epoch": lease_epoch,
        }
        digest = operation_payload_digest({
            "method": "resolution.request_checkpoint",
            "params": payload,
        })
        return await self._rpc.call(
            "resolution.request_checkpoint",
            {
                **payload,
                "operation_id": interaction_id,
                "operation_payload_digest": digest,
            },
            operation_id=interaction_id,
            operation_payload_digest=digest,
        )

    async def resume_after_interaction(
        self,
        *,
        resolution_thread_id: str,
        segment_id: str,
        activation_id: str,
        lease_epoch: int,
        interaction_id: str,
        continuation_delta: Mapping[str, Any],
        continuation_authority_token: str,
    ) -> dict[str, Any]:
        """Resume with a fresh Brain-issued activation authority."""

        if not isinstance(continuation_delta, Mapping):
            raise TypeError("continuation_delta must be an object")
        if "reply_text" in continuation_delta:
            raise ValueError(
                "continuation_delta must contain a cognition decision, not reply_text"
            )
        if not isinstance(continuation_authority_token, str) or not continuation_authority_token:
            raise ValueError("continuation_authority_token is required")
        payload = {
            "resolution_thread_id": resolution_thread_id,
            "segment_id": segment_id,
            "activation_id": activation_id,
            "lease_epoch": lease_epoch,
            "continuation": dict(continuation_delta),
        }
        digest = operation_payload_digest({
            "method": "resolution.continue",
            "params": {"interaction_id": interaction_id, **payload},
        })
        continuation_facts = _continuation_model_facts(continuation_delta)
        expected: dict[str, object] = {
            "activation_id": activation_id,
            "lease_epoch": lease_epoch,
            "resolution_thread_id": resolution_thread_id,
            "segment_id": segment_id,
        }
        if self._repository is None:
            try:
                authority = verify_activation_token(
                    continuation_authority_token,
                    secret=self._semantic_authority_secret,
                    expected=expected,
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError("SEMANTIC_AUTHORITY_INVALID") from exc
            objective = continuation_delta.get("answer")
            if not isinstance(objective, str) or not objective:
                objective = continuation_delta.get("response_goal")
            if not isinstance(objective, str) or not objective:
                raise ValueError("continuation_delta requires a typed answer")
            intake = {
                "schema_version": "dsh_resolution_intake.v2",
                "mode": "continue",
                "request_id": interaction_id,
                "operation_id": interaction_id,
                "operation_payload_digest": digest,
                "resolution_thread_id": resolution_thread_id,
                "segment_id": segment_id,
                "brain_conversation_ref": authority.brain_conversation_ref,
                "workspace_root": authority.workspace_root,
                "route_digest": authority.route_digest,
                "model_input": {
                    "objective": objective,
                    "facts": continuation_facts,
                },
                "semantic_tool_authority": {
                    "catalog_digest": authority.catalog_digest,
                    "token": continuation_authority_token,
                },
                "interaction_authority": {
                    "issuer": authority.interaction_issuer,
                    "scope_fingerprint": authority.scope_fingerprint,
                    "audience_fingerprint": authority.audience_fingerprint,
                },
            }
            return await self._rpc.call(
                "resolution.continue",
                {
                    **payload,
                    "operation_id": interaction_id,
                    "operation_payload_digest": digest,
                    "intake": DSHResolutionIntakeV2.from_mapping(intake).to_dict(),
                },
                operation_id=interaction_id,
                operation_payload_digest=digest,
            )
        record = await self._repository_call("get_thread", resolution_thread_id)
        if record is None:
            raise RuntimeError("V2 resolution thread is not admitted")
        if record.current_segment_id != segment_id:
            raise RuntimeError("SEGMENT_ID_MISMATCH")
        health = await self._health_identity()
        expected.update({
            "brain_conversation_ref": record.brain_conversation_ref,
            "scope_fingerprint": record.scope_fingerprint,
            "audience_fingerprint": record.audience_fingerprint,
            "workspace_root": record.workspace_root,
            "workspace_fingerprint": workspace_fingerprint(record.workspace_root),
            "route_digest": health["route_digest"],
            "model_route_digest": health["route_digest"],
            "catalog_digest": health["semantic_catalog_digest"],
            "profile_version": health["profile_version"],
            "policy_epoch": health["policy_epoch"],
        })
        try:
            authority = verify_activation_token(
                continuation_authority_token,
                secret=self._semantic_authority_secret,
                expected=expected,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("SEMANTIC_AUTHORITY_INVALID") from exc
        objective = continuation_delta.get("answer")
        if not isinstance(objective, str) or not objective:
            objective = continuation_delta.get("response_goal")
        if not isinstance(objective, str) or not objective:
            objective = record.root_goal_ref
        intake = {
            "schema_version": "dsh_resolution_intake.v2",
            "mode": "continue",
            "request_id": interaction_id,
            "operation_id": interaction_id,
            "operation_payload_digest": digest,
            "resolution_thread_id": resolution_thread_id,
            "segment_id": segment_id,
            "brain_conversation_ref": record.brain_conversation_ref,
            "workspace_root": record.workspace_root,
            "route_digest": record.route_digest,
            "model_input": {
                "objective": objective,
                "facts": continuation_facts,
            },
            "semantic_tool_authority": {
                "catalog_digest": authority.catalog_digest,
                "token": continuation_authority_token,
            },
            "interaction_authority": {
                "issuer": authority.interaction_issuer,
                "scope_fingerprint": authority.scope_fingerprint,
                "audience_fingerprint": authority.audience_fingerprint,
            },
        }
        parsed = DSHResolutionIntakeV2.from_mapping(intake)
        self._assert_intake_health(parsed, health)
        self._verify_intake_authority(
            parsed,
            health,
            activation_id=activation_id,
            lease_epoch=lease_epoch,
            segment_id=segment_id,
        )
        return await self._rpc.call(
            "resolution.continue",
            {
                "operation_id": interaction_id,
                "operation_payload_digest": digest,
                "activation_id": activation_id,
                "lease_epoch": lease_epoch,
                "intake": parsed.to_dict(),
            },
            operation_id=interaction_id,
            operation_payload_digest=digest,
        )

    async def cancel(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> dict[str, Any]:
        return await self._control(
            "resolution.cancel",
            resolution_thread_id,
            activation_id,
            lease_epoch,
        )

    async def inspect(self, resolution_thread_id: str) -> dict[str, Any]:
        record = await self._repository_call("get_thread", resolution_thread_id)
        if record is None:
            return {
                "resolution_thread_id": resolution_thread_id,
                "disposition": "not_admitted",
            }
        return {
            "resolution_thread_id": resolution_thread_id,
            "segment_id": record.current_segment_id,
            "state": self._observed_states.get(
                resolution_thread_id,
                record.state,
            ),
            "lease_epoch": record.lease_epoch,
            "document_revision": record.document_revision,
        }

    async def dispose_activation(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> dict[str, Any]:
        result = await self._control(
            "resolution.dispose_activation",
            resolution_thread_id,
            activation_id,
            lease_epoch,
        )
        await self._release_lease_if_current(
            resolution_thread_id,
            activation_id,
            lease_epoch,
        )
        return result

    async def renew_lease(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> dict[str, Any]:
        return await self._repository_call(
            "renew_lease",
            resolution_thread_id,
            activation_id,
            lease_epoch,
            _expires(),
        )

    async def takeover_expired(
        self, resolution_thread_id: str, *, now: str
    ) -> dict[str, Any]:
        return await self._repository_call(
            "acquire_lease",
            resolution_thread_id,
            f"act_{uuid4().hex}",
            self._owner_id,
            _expires(),
            now,
        )

    async def _activate(
        self, method: str, intake: DSHResolutionIntakeV2
    ) -> dict[str, Any]:
        runtime = intake
        health = await self._health_identity()
        self._assert_intake_health(runtime, health)
        self._verify_intake_authority(runtime, health)
        record = await self._repository_call(
            "get_thread", runtime.resolution_thread_id
        )
        if record is None:
            raise RuntimeError("created thread disappeared")
        segment_id = runtime.segment_id
        existing = await self._repository_call(
            "get_operation",
            runtime.resolution_thread_id,
            runtime.operation_id,
        )
        if existing is None and segment_id != record.current_segment_id:
            raise RuntimeError("SEGMENT_ID_MISMATCH")
        if existing is not None:
            if (
                existing["operation_payload_digest"]
                != runtime.operation_payload_digest
                or existing["method"] != method
            ):
                raise OperationIdReuseMismatchError(
                    "operation id was reused with a different method or digest"
                )
            inspected = await self._rpc.call(
                "resolution.inspect",
                {
                    "operation_id": runtime.operation_id,
                    "operation_payload_digest": (
                        runtime.operation_payload_digest
                    ),
                },
            )
            inspected_disposition = inspected.get("disposition")
            if inspected_disposition == "unknown":
                raise OperationOutcomeUncertainError(
                    "operation outcome remains unknown"
                )
            has_current_fence = self._operation_has_current_fence(
                record, existing
            )
            if inspected_disposition in {"not_admitted", "admitted_active"}:
                if has_current_fence:
                    result = await self._call_existing_activation(
                        method,
                        intake,
                        existing,
                        health,
                    )
                    return await self._complete_activation(
                        runtime.resolution_thread_id,
                        str(existing["segment_id"]),
                        runtime.operation_id,
                        str(existing["activation_id"]),
                        int(existing["lease_epoch"]),
                        result,
                    )
                if inspected_disposition == "admitted_active":
                    raise OperationOutcomeUncertainError(
                        "admitted operation has no matching controller fence"
                    )
            elif inspected_disposition in self._COMMITTED_DISPOSITIONS:
                activation_id = existing.get("activation_id")
                lease_epoch = existing.get("lease_epoch")
                return await self._complete_activation(
                    runtime.resolution_thread_id,
                    str(existing["segment_id"]),
                    runtime.operation_id,
                    activation_id if isinstance(activation_id, str) else None,
                    lease_epoch if isinstance(lease_epoch, int) else None,
                    inspected,
                )
            else:
                raise OperationOutcomeUncertainError(
                    "operation inspection returned an unsupported disposition"
                )

        activation_id = activation_id_for(
            runtime.resolution_thread_id,
            segment_id,
            int(record.lease_epoch) + 1,
        )
        lease = await self._repository_call(
            "acquire_lease",
            runtime.resolution_thread_id,
            activation_id,
            self._owner_id,
            _expires(),
            _now(),
        )
        prepared = await self._repository_call(
            "prepare_operation",
            runtime.resolution_thread_id,
            runtime.operation_id,
            runtime.operation_payload_digest,
            method,
            segment_id,
            activation_id,
            lease["lease_epoch"],
        )
        if (
            prepared["operation_payload_digest"]
            != runtime.operation_payload_digest
        ):
            raise OperationIdReuseMismatchError(
                "operation id was reused with a different digest"
            )
        self._verify_intake_authority(
            runtime,
            health,
            activation_id=activation_id,
            lease_epoch=int(lease["lease_epoch"]),
            segment_id=segment_id,
        )
        result = await self._call_with_lease_renewal(
            method,
            {
                "operation_id": runtime.operation_id,
                "operation_payload_digest": (
                    runtime.operation_payload_digest
                ),
                "activation_id": activation_id,
                "lease_epoch": lease["lease_epoch"],
                "intake": {
                    **intake.to_dict(),
                    "segment_id": segment_id,
                },
            },
            runtime.resolution_thread_id,
            activation_id,
            lease["lease_epoch"],
            runtime.operation_id,
            runtime.operation_payload_digest,
        )
        return await self._complete_activation(
            runtime.resolution_thread_id,
            segment_id,
            runtime.operation_id,
            activation_id,
            lease["lease_epoch"],
            result,
        )

    async def _call_existing_activation(
        self,
        method: str,
        intake: DSHResolutionIntakeV2,
        operation: Mapping[str, Any],
        health: Mapping[str, str],
    ) -> dict[str, Any]:
        activation_id = operation["activation_id"]
        lease_epoch = operation["lease_epoch"]
        segment_id = operation["segment_id"]
        if (
            not isinstance(activation_id, str)
            or not isinstance(lease_epoch, int)
            or not isinstance(segment_id, str)
        ):
            raise OperationOutcomeUncertainError(
                "prepared operation has no reusable activation fence"
            )
        self._verify_intake_authority(
            intake,
            health,
            activation_id=activation_id,
            lease_epoch=lease_epoch,
            segment_id=segment_id,
        )
        runtime = intake
        return await self._call_with_lease_renewal(
            method,
            {
                "operation_id": runtime.operation_id,
                "operation_payload_digest": (
                    runtime.operation_payload_digest
                ),
                "activation_id": activation_id,
                "lease_epoch": lease_epoch,
                "intake": {
                    **intake.to_dict(),
                    "segment_id": segment_id,
                },
            },
            runtime.resolution_thread_id,
            activation_id,
            lease_epoch,
            runtime.operation_id,
            runtime.operation_payload_digest,
        )

    async def _complete_activation(
        self,
        resolution_thread_id: str,
        segment_id: str,
        operation_id: str,
        activation_id: str | None,
        lease_epoch: int | None,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        disposition = str(result.get("disposition", "faulted"))
        last_committed_seq = result.get("last_committed_seq")
        if not isinstance(last_committed_seq, int):
            exhaust = result.get("exhaust")
            if isinstance(exhaust, Mapping):
                candidate = exhaust.get("last_committed_seq")
                if isinstance(candidate, int):
                    last_committed_seq = candidate
        message_source_id = result.get("dsh_message_source_id")
        if not isinstance(message_source_id, str):
            message_source_id = None
        await self._repository_call(
            "update_operation",
            resolution_thread_id,
            operation_id,
            disposition=disposition,
            dsh_message_source_id=message_source_id,
            last_committed_seq=last_committed_seq,
            fault_code=self._fault_code(result),
        )
        session_id = result.get("session_id")
        if not isinstance(session_id, str):
            session_id = self._dsh_session_id(
                resolution_thread_id, segment_id
            )
        segment_state = {
            "admitted_active": "live",
            "terminal": "terminal",
            "checkpointed": "checkpointed",
            "canceled": "canceled",
            "faulted": "faulted",
        }.get(disposition, "faulted")
        segment_changes: dict[str, Any] = {
            "dsh_session_id": session_id,
            "state": segment_state,
            "last_used_at": _now(),
        }
        if isinstance(last_committed_seq, int):
            segment_changes["last_committed_seq"] = last_committed_seq
        await self._repository_call(
            "update_segment",
            resolution_thread_id,
            segment_id,
            **segment_changes,
        )
        if (
            disposition in self._COMMITTED_DISPOSITIONS
            and activation_id is not None
            and lease_epoch is not None
        ):
            await self._dispose_and_release_if_current(
                resolution_thread_id,
                segment_id,
                activation_id,
                lease_epoch,
            )
        return self._activation_result(
            result,
            resolution_thread_id,
            segment_id,
            activation_id,
            lease_epoch,
        )

    async def _dispose_and_release_if_current(
        self,
        resolution_thread_id: str,
        segment_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> None:
        async with self._disposal_lock:
            record = await self._repository_call(
                "get_thread", resolution_thread_id
            )
            if record is None:
                raise RuntimeError(
                    "resolution thread disappeared during disposal"
                )
            if not self._lease_is_current(
                record,
                activation_id=activation_id,
                lease_epoch=lease_epoch,
            ):
                return
            await self._dispose_sidecar_activation(
                resolution_thread_id,
                segment_id,
                activation_id,
                lease_epoch,
            )
            await self._repository_call(
                "release_lease",
                resolution_thread_id,
                activation_id,
                lease_epoch,
            )

    async def _release_lease_if_current(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> None:
        """Serialize lease cleanup that already disposed its sidecar owner."""

        async with self._disposal_lock:
            record = await self._repository_call(
                "get_thread", resolution_thread_id
            )
            if record is None:
                raise RuntimeError(
                    "resolution thread disappeared during lease release"
                )
            if not self._lease_is_current(
                record,
                activation_id=activation_id,
                lease_epoch=lease_epoch,
            ):
                return
            await self._repository_call(
                "release_lease",
                resolution_thread_id,
                activation_id,
                lease_epoch,
            )

    @staticmethod
    def _lease_is_current(
        record: Any,
        *,
        activation_id: str,
        lease_epoch: int,
    ) -> bool:
        """Return whether one record still carries the exact cleanup fence."""

        current_lease = record.current_lease
        return (
            current_lease is not None
            and current_lease.get("activation_id") == activation_id
            and current_lease.get("lease_epoch") == lease_epoch
        )

    @staticmethod
    def _operation_has_current_fence(
        record: Any, operation: Mapping[str, Any]
    ) -> bool:
        current_lease = record.current_lease
        return (
            current_lease is not None
            and isinstance(operation.get("activation_id"), str)
            and isinstance(operation.get("lease_epoch"), int)
            and current_lease.get("activation_id")
            == operation.get("activation_id")
            and current_lease.get("lease_epoch")
            == operation.get("lease_epoch")
        )

    async def _call_with_lease_renewal(
        self,
        method: str,
        params: Mapping[str, Any],
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
        operation_id: str,
        payload_digest: str,
    ) -> dict[str, Any]:
        stopped = asyncio.Event()

        async def renew_until_stopped() -> None:
            while not stopped.is_set():
                try:
                    await asyncio.wait_for(
                        stopped.wait(),
                        timeout=self._LEASE_RENEWAL_SECONDS,
                    )
                except TimeoutError:
                    await self._repository_call(
                        "renew_lease",
                        resolution_thread_id,
                        activation_id,
                        lease_epoch,
                        _expires(),
                    )

        renewal = asyncio.create_task(renew_until_stopped())
        try:
            result = await self._rpc.call(
                method,
                params,
                operation_id=operation_id,
                operation_payload_digest=payload_digest,
            )
        finally:
            stopped.set()
            await renewal
        return result

    async def _dispose_sidecar_activation(
        self,
        resolution_thread_id: str,
        segment_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> None:
        operation_id = f"op_{uuid4().hex}"
        digest = f"sha256:{operation_id}"
        await self._rpc.call(
            "resolution.dispose_activation",
            {
                "operation_id": operation_id,
                "operation_payload_digest": digest,
                "resolution_thread_id": resolution_thread_id,
                "segment_id": segment_id,
                "activation_id": activation_id,
                "lease_epoch": lease_epoch,
            },
            operation_id=operation_id,
            operation_payload_digest=digest,
        )

    @staticmethod
    def _fault_code(result: Mapping[str, Any]) -> str | None:
        exhaust = result.get("exhaust")
        if not isinstance(exhaust, Mapping):
            return None
        fault = exhaust.get("fault")
        if not isinstance(fault, Mapping):
            return None
        code = fault.get("code")
        return code if isinstance(code, str) else None

    @staticmethod
    def _activation_result(
        result: Mapping[str, Any],
        resolution_thread_id: str,
        segment_id: str,
        activation_id: object,
        lease_epoch: object,
    ) -> dict[str, Any]:
        return {
            **result,
            "resolution_thread_id": resolution_thread_id,
            "segment_id": segment_id,
            "activation_id": activation_id,
            "lease_epoch": lease_epoch,
        }

    async def _control(
        self,
        method: str,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> dict[str, Any]:
        record = await self._fence(
            resolution_thread_id, activation_id, lease_epoch
        )
        operation_id = f"op_{uuid4().hex}"
        digest = f"sha256:{operation_id}"
        await self._repository_call(
            "prepare_operation",
            resolution_thread_id,
            operation_id,
            digest,
            method,
            record.current_segment_id,
            activation_id,
            lease_epoch,
        )
        result = await self._rpc.call(
            method,
            {
                "operation_id": operation_id,
                "operation_payload_digest": digest,
                "resolution_thread_id": resolution_thread_id,
                "segment_id": record.current_segment_id,
                "activation_id": activation_id,
                "lease_epoch": lease_epoch,
            },
            operation_id=operation_id,
            operation_payload_digest=digest,
        )
        disposition = str(result.get("disposition", "faulted"))
        last_committed_seq = result.get("last_committed_seq")
        if not isinstance(last_committed_seq, int):
            last_committed_seq = None
        message_source_id = result.get("dsh_message_source_id")
        if not isinstance(message_source_id, str):
            message_source_id = None
        await self._repository_call(
            "update_operation",
            resolution_thread_id,
            operation_id,
            disposition=disposition,
            dsh_message_source_id=message_source_id,
            last_committed_seq=last_committed_seq,
            fault_code=self._fault_code(result),
        )
        if method in {"resolution.request_checkpoint", "resolution.cancel"}:
            state = {
                "terminal": "terminal",
                "checkpointed": "checkpointed",
                "canceled": "canceled",
                "faulted": "faulted",
            }.get(disposition)
            if state is None:
                raise RpcContractError(
                    "DSH control returned a non-committed disposition"
                )
            changes: dict[str, Any] = {
                "state": state,
                "last_used_at": _now(),
            }
            if isinstance(last_committed_seq, int):
                changes["last_committed_seq"] = last_committed_seq
            await self._repository_call(
                "update_segment",
                resolution_thread_id,
                record.current_segment_id,
                **changes,
            )
            await self._dispose_and_release_if_current(
                resolution_thread_id,
                record.current_segment_id,
                activation_id,
                lease_epoch,
            )
            self._observed_states[resolution_thread_id] = state
        return result

    async def _fence(
        self,
        resolution_thread_id: str,
        activation_id: str,
        lease_epoch: int,
    ):
        await self._repository_call(
            "validate_fence",
            resolution_thread_id, activation_id, lease_epoch
        )
        record = await self._repository_call("get_thread", resolution_thread_id)
        if record is None:
            raise RuntimeError("validated thread disappeared")
        return record

    async def _repository_call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._repository, name)
        value = method(*args, **kwargs)
        return await value if isawaitable(value) else value

    async def _health_identity(self) -> dict[str, str]:
        """Read the actual mounted Standard identity before thread admission."""

        value = await self._rpc.call("system.health", {})
        if not isinstance(value, Mapping):
            raise RpcContractError("sidecar health identity is unavailable")
        if value.get("protocol_version") != "kazusa.dsh-resolution-rpc.v2":
            raise RpcContractError("sidecar health protocol is unsupported")
        if value.get("status") != "ready":
            raise RpcContractError("sidecar health is not ready")
        route = value.get("route")
        catalog = value.get("catalog")
        workspace = value.get("workspace")
        policy = value.get("policy")
        if not isinstance(route, Mapping):
            raise RpcContractError("sidecar health route identity is unavailable")
        if not isinstance(catalog, Mapping):
            raise RpcContractError("sidecar health catalog identity is unavailable")
        if not isinstance(workspace, Mapping):
            raise RpcContractError(
                "sidecar health workspace identity is unavailable"
            )
        if not isinstance(policy, Mapping):
            raise RpcContractError("sidecar health policy identity is unavailable")

        def required_text(value: object, field: str) -> str:
            if not isinstance(value, str) or not value:
                raise RpcContractError(
                    f"sidecar health {field} is unavailable"
                )
            return value

        root = required_text(workspace.get("root"), "workspace.root")
        return {
            "status": "ready",
            "route_digest": required_text(route.get("digest"), "route.digest"),
            "native_catalog_digest": required_text(
                catalog.get("native_catalog_digest"),
                "catalog.native_catalog_digest",
            ),
            "semantic_catalog_digest": required_text(
                catalog.get("semantic_catalog_digest"),
                "catalog.semantic_catalog_digest",
            ),
            "published_catalog_digest": required_text(
                catalog.get("published_catalog_digest"),
                "catalog.published_catalog_digest",
            ),
            "profile_version": required_text(
                value.get("profile_version"), "profile_version"
            ),
            "dsh_release": required_text(value.get("dsh_release"), "dsh_release"),
            "session_store_epoch": required_text(
                value.get("store_epoch"), "store_epoch"
            ),
            "policy_epoch": required_text(policy.get("epoch"), "policy.epoch"),
            "workspace_root": root.replace("\\", "/"),
        }

    @staticmethod
    def _assert_intake_health(
        intake: DSHResolutionIntakeV2,
        health: Mapping[str, str],
    ) -> None:
        """Require intake identity to name the actual sidecar health."""

        if intake.route_digest != health["route_digest"]:
            raise RuntimeError("ROUTE_DIGEST_MISMATCH")
        if intake.semantic_tool_authority["catalog_digest"] != health[
            "semantic_catalog_digest"
        ]:
            raise RuntimeError("SEMANTIC_CATALOG_DIGEST_MISMATCH")
        if intake.workspace_root.replace("\\", "/") != health["workspace_root"]:
            raise RuntimeError("WORKSPACE_ROOT_MISMATCH")

    def _verify_intake_authority(
        self,
        intake: DSHResolutionIntakeV2,
        health: Mapping[str, str],
        *,
        activation_id: str | None = None,
        lease_epoch: int | None = None,
        segment_id: str | None = None,
    ) -> None:
        """Verify the host-issued token against intake and the acquired fence."""

        expected: dict[str, object] = {
            "resolution_thread_id": intake.resolution_thread_id,
            "segment_id": segment_id or intake.segment_id,
            "brain_conversation_ref": intake.brain_conversation_ref,
            "scope_fingerprint": intake.interaction_authority[
                "scope_fingerprint"
            ],
            "audience_fingerprint": intake.interaction_authority[
                "audience_fingerprint"
            ],
            "workspace_root": intake.workspace_root,
            "workspace_fingerprint": workspace_fingerprint(
                intake.workspace_root
            ),
            "route_digest": health["route_digest"],
            "model_route_digest": health["route_digest"],
            "catalog_digest": health["semantic_catalog_digest"],
            "profile_version": health["profile_version"],
            "policy_epoch": health["policy_epoch"],
            "interaction_issuer": intake.interaction_authority["issuer"],
        }
        if activation_id is not None:
            expected["activation_id"] = activation_id
        if lease_epoch is not None:
            expected["lease_epoch"] = lease_epoch
        try:
            verify_activation_token(
                intake.semantic_tool_authority["token"],
                secret=self._semantic_authority_secret,
                expected=expected,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("SEMANTIC_AUTHORITY_INVALID") from exc

    @classmethod
    def _compatibility_values(
        cls,
        intake: DSHResolutionIntakeV2,
        health: Mapping[str, str],
    ) -> dict[str, str]:
        return {
            "brain_conversation_ref": intake.brain_conversation_ref,
            "workspace_root": intake.workspace_root,
            "workspace_fingerprint": workspace_fingerprint(
                intake.workspace_root
            ),
            "route_digest": intake.route_digest,
            "resolver_profile_version": health["profile_version"],
            "dsh_release": health["dsh_release"],
            "session_store_epoch": health["session_store_epoch"],
            "standard_catalog_digest": health["native_catalog_digest"],
            "semantic_catalog_digest": health["semantic_catalog_digest"],
            "policy_epoch": health["policy_epoch"],
            "scope_fingerprint": intake.interaction_authority[
                "scope_fingerprint"
            ],
            "audience_fingerprint": intake.interaction_authority[
                "audience_fingerprint"
            ],
            "interaction_id": intake.request_id,
        }

    @classmethod
    def _segment(
        cls,
        intake: DSHResolutionIntakeV2,
        health: Mapping[str, str],
        *,
        segment_id: str | None = None,
    ) -> dict[str, Any]:
        now = _now()
        values = cls._compatibility_values(intake, health)
        return {
            "schema_version": SEGMENT_SCHEMA_VERSION,
            "segment_id": segment_id or intake.segment_id,
            "resolution_thread_id": intake.resolution_thread_id,
            "dsh_session_id": ResolutionController._dsh_session_id(
                intake.resolution_thread_id, segment_id or intake.segment_id
            ),
            "brain_conversation_ref": values["brain_conversation_ref"],
            "workspace_root": values["workspace_root"],
            "workspace_fingerprint": values["workspace_fingerprint"],
            "route_digest": values["route_digest"],
            "resolver_profile_version": values["resolver_profile_version"],
            "dsh_release": values["dsh_release"],
            "session_store_epoch": values["session_store_epoch"],
            "standard_catalog_digest": values["standard_catalog_digest"],
            "semantic_catalog_digest": values["semantic_catalog_digest"],
            "policy_epoch": values["policy_epoch"],
            "scope_fingerprint": values["scope_fingerprint"],
            "audience_fingerprint": values["audience_fingerprint"],
            "interaction_id": values["interaction_id"],
            "state": "live",
            "last_committed_seq": 0,
            "parent_segment_id": None,
            "rotation_reason": None,
            "created_at": now,
            "last_used_at": now,
        }

    @staticmethod
    def _dsh_session_id(
        resolution_thread_id: str, segment_id: str
    ) -> str:
        identity = f"{resolution_thread_id}\0{segment_id}".encode()
        suffix = hashlib.sha256(identity).hexdigest()[:32]
        return f"kazusa-resolution-{suffix}"
