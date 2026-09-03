"""Public runtime entry point for the standalone DSH V2 sidecar."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from agentic_resolver.contracts import (
    DSH_RELEASE,
    PROFILE_VERSION,
    SESSION_STORE_EPOCH,
    DSHResolutionExhaustV2,
    DSHResolutionIntakeV2,
)
from agentic_resolver.controller import ResolutionController
from agentic_resolver.fingerprints import (
    audience_fingerprint,
    operation_payload_digest,
    scope_fingerprint,
)
from agentic_resolver.persistence import MongoResolutionThreadRepository
from agentic_resolver.rpc import DSHRpcClient
from kazusa_ai_chatbot.config import load_agentic_resolver_route_settings
from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
    SemanticActivationAuthorityV1,
    activation_id_for,
    issue_activation_token,
)
from kazusa_ai_chatbot.dsh_tool_gateway.catalog import semantic_catalog_digest
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest


class ResolverController(Protocol):
    async def resolve(self, intake: Mapping[str, Any]) -> Mapping[str, Any]:
        """Resolve one canonical intake through the independent sidecar."""

    async def readiness(self) -> dict[str, str]:
        """Return the authenticated mounted-sidecar identity."""


class AgenticResolverRuntime:
    """Small public runtime delegating lifecycle work to its controller."""

    def __init__(self, controller: ResolverController) -> None:
        self._controller = controller
        self._active_operations: dict[
            tuple[str, int], dict[str, object]
        ] = {}

    @classmethod
    def from_environment(
        cls, *, data_root: Path | None = None
    ) -> AgenticResolverRuntime:
        load_agentic_resolver_route_settings()
        configured_root = data_root
        if configured_root is None:
            raw_root = os.environ.get("KAZUSA_DSH_DATA_ROOT")
            if raw_root is None or not raw_root:
                raise ValueError("KAZUSA_DSH_DATA_ROOT is required")
            configured_root = Path(raw_root)
        if not configured_root.is_absolute():
            raise ValueError("data_root must be absolute")
        workspace = os.environ.get("AGENTIC_RESOLVER_WORKSPACE_ROOT")
        if workspace is None or not workspace:
            raise ValueError("AGENTIC_RESOLVER_WORKSPACE_ROOT is required")
        workspace_path = Path(workspace)
        if not workspace_path.is_absolute():
            raise ValueError("AGENTIC_RESOLVER_WORKSPACE_ROOT must be absolute")
        endpoint = os.environ.get("KAZUSA_DSH_SIDECAR_URL", "")
        token = os.environ.get("KAZUSA_DSH_RPC_TOKEN", "")
        semantic_secret = os.environ.get("KAZUSA_DSH_TOOL_GATEWAY_SECRET")
        if semantic_secret is None or not semantic_secret:
            raise ValueError("KAZUSA_DSH_TOOL_GATEWAY_SECRET is required")
        controller = ResolutionController(
            MongoResolutionThreadRepository(),
            DSHRpcClient(endpoint, token),
            owner_id=f"resolver_{uuid4().hex}",
            semantic_authority_secret=semantic_secret.encode("utf-8"),
        )
        return cls(controller)

    async def resolve(
        self, intake: Mapping[str, Any]
    ) -> DSHResolutionExhaustV2:
        value = await self._controller.resolve(intake)
        exhaust_value = value.get("exhaust", value)
        return DSHResolutionExhaustV2.from_mapping(exhaust_value)

    async def readiness(self) -> dict[str, str]:
        """Probe authenticated DSH sidecar readiness."""

        return await self._controller.readiness()

    async def open(
        self,
        *,
        task_session_id: str,
        operation_generation: int,
        request: Mapping[str, Any],
        execution_context: Mapping[str, Any],
        start_spec: Mapping[str, Any],
        before_resolve: Any | None = None,
    ) -> DSHResolutionExhaustV2:
        """Open one DSH session from the shared Brain task carrier."""

        if not isinstance(task_session_id, str) or not task_session_id.strip():
            raise ValueError("task_session_id is required")
        if operation_generation != 0:
            raise ValueError("initial DSH open must use generation zero")
        if not isinstance(request, Mapping):
            raise TypeError("request must be an object")
        if not isinstance(execution_context, Mapping):
            raise TypeError("execution_context must be an object")
        if not isinstance(start_spec, Mapping):
            raise TypeError("start_spec must be an object")
        objective = request.get("semantic_goal", request.get("objective"))
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("DSH task objective is required")
        facts = start_spec.get("model_facts")
        if not isinstance(facts, list) or len(facts) != 10:
            raise ValueError("DSH open requires exactly ten model facts")
        if any(not isinstance(fact, str) or not fact for fact in facts):
            raise ValueError("DSH model facts must be non-empty strings")
        model_facts_digest = start_spec.get("model_facts_digest")
        if model_facts_digest != content_digest(facts):
            raise ValueError("DSH model facts digest is invalid")
        conversation_ref = execution_context.get("brain_conversation_ref")
        platform = execution_context.get("platform")
        channel_id = execution_context.get("channel_id")
        user_id = execution_context.get("requester_global_user_id")
        goal_continuation_ref = execution_context.get("goal_continuation_ref")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (conversation_ref, platform, channel_id, user_id)
        ):
            raise ValueError("DSH execution context identity is incomplete")
        if not isinstance(goal_continuation_ref, Mapping):
            raise TypeError("DSH goal_continuation_ref is required")
        objective_ref = content_digest(goal_continuation_ref)
        if start_spec.get("objective_ref") != objective_ref:
            raise ValueError("DSH objective_ref is invalid")
        authority = self.new_runtime_authority(
            objective_ref=objective_ref,
            brain_conversation_ref=conversation_ref,
            service_scope={
                "platform": platform,
                "platform_channel_id": channel_id,
                "global_user_id": user_id,
            },
            audience={
                "kind": "kazusa_task_resolution",
                "goal_continuation_ref_digest": objective_ref,
                "requested_delivery": "send_result_when_done",
            },
            interaction_issuer="brain.task_resolution",
        )
        intake = self.build_intake(
            authority,
            objective=objective.strip(),
            facts=list(facts),
            mode="start",
        )
        operation_key = (task_session_id, operation_generation)
        self._active_operations[operation_key] = {
            "resolution_thread_id": authority["resolution_thread_id"],
            "segment_id": authority["segment_id"],
            "activation_id": authority["activation_id"],
            "lease_epoch": authority["lease_epoch"],
        }
        try:
            admitted_reference: Mapping[str, object] | None = None
            if before_resolve is not None:
                if not callable(before_resolve):
                    raise TypeError("before_resolve must be callable")
                reference = {
                    "schema_version": "dsh_resolution_ref.v1",
                    "resolution_thread_id": authority[
                        "resolution_thread_id"
                    ],
                    "segment_id": authority["segment_id"],
                    "dsh_session_id": task_session_id,
                    "activation_id": authority["activation_id"],
                    "lease_epoch": authority["lease_epoch"],
                    "document_revision": 0,
                    "last_committed_seq": 0,
                }
                admitted_reference = reference
                callback_result = before_resolve(reference)
                if hasattr(callback_result, "__await__"):
                    await callback_result
            exhaust = await self.resolve(intake.to_dict())
            serialized = exhaust.to_dict()
            identity = dict(serialized.get("identity") or {})
            identity.update({
                "resolution_thread_id": authority["resolution_thread_id"],
                "segment_id": authority["segment_id"],
                "dsh_session_id": task_session_id,
                "activation_id": authority["activation_id"],
                "lease_epoch": authority["lease_epoch"],
                "document_revision": int(
                    identity.get("document_revision", 0),
                ),
            })
            if (
                "last_committed_seq" not in identity
                and serialized.get("last_committed_seq") is None
                and admitted_reference is not None
            ):
                identity["last_committed_seq"] = admitted_reference[
                    "last_committed_seq"
                ]
            serialized["identity"] = identity
            return DSHResolutionExhaustV2.from_mapping(serialized)
        finally:
            self._active_operations.pop(operation_key, None)

    async def request_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
        """Request a cooperative checkpoint through the shared controller."""

        method = getattr(self._controller, "request_checkpoint", None)
        if not callable(method):
            raise TypeError("resolver controller cannot checkpoint")
        forwarded = dict(kwargs)
        forwarded.pop("segment_id", None)
        task_session_id = forwarded.pop("task_session_id", None)
        operation_generation = forwarded.pop("operation_generation", None)
        active: Mapping[str, object] | None = None
        if (
            isinstance(task_session_id, str)
            and isinstance(operation_generation, int)
        ):
            active = self._active_operations.get(
                (task_session_id, operation_generation),
            )
            if active is None:
                raise RuntimeError("DSH operation is not active")
            result = method(
                str(active["resolution_thread_id"]),
                str(active["activation_id"]),
                int(active["lease_epoch"]),
            )
        else:
            result = method(**forwarded)
        if hasattr(result, "__await__"):
            result = await result
        if not isinstance(result, Mapping):
            raise TypeError("resolver checkpoint returned a non-object")
        if (
            isinstance(task_session_id, str)
            and isinstance(operation_generation, int)
            and active is not None
        ):
            return {
                **dict(result),
                "resolution_thread_id": active["resolution_thread_id"],
                "segment_id": active["segment_id"],
                "activation_id": active["activation_id"],
                "lease_epoch": active["lease_epoch"],
                "dsh_session_id": task_session_id,
            }
        return dict(result)

    async def continue_after_terminal(self, **kwargs: Any) -> dict[str, Any]:
        """Continue a terminal DSH segment through the shared controller."""

        method = getattr(self._controller, "continue_after_terminal", None)
        if not callable(method):
            raise TypeError("resolver controller cannot continue a terminal session")
        result = method(**kwargs)
        if hasattr(result, "__await__"):
            result = await result
        if not isinstance(result, Mapping):
            raise TypeError("resolver continuation returned a non-object")
        return dict(result)

    async def amend(self, **kwargs: Any) -> dict[str, Any]:
        """Apply one fenced semantic amendment through the controller."""

        method = getattr(self._controller, "amend", None)
        if not callable(method):
            raise TypeError("resolver controller cannot amend a session")
        result = method(**kwargs)
        if hasattr(result, "__await__"):
            result = await result
        if not isinstance(result, Mapping):
            raise TypeError("resolver amendment returned a non-object")
        return dict(result)

    async def continue_after_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
        """Continue a checkpointed segment with the controller's fresh fence."""

        method = getattr(self._controller, "continue_after_checkpoint", None)
        if not callable(method):
            method = getattr(self._controller, "continue_after_terminal", None)
        if not callable(method):
            raise TypeError("resolver controller cannot continue a checkpoint")
        result = method(**kwargs)
        if hasattr(result, "__await__"):
            result = await result
        if not isinstance(result, Mapping):
            raise TypeError("resolver checkpoint continuation returned a non-object")
        return dict(result)

    async def cancel(self, **kwargs: Any) -> dict[str, Any]:
        """Cancel one fenced DSH session through the controller."""

        method = getattr(self._controller, "cancel", None)
        if not callable(method):
            raise TypeError("resolver controller cannot cancel a session")
        result = method(**kwargs)
        if hasattr(result, "__await__"):
            result = await result
        if not isinstance(result, Mapping):
            raise TypeError("resolver cancellation returned a non-object")
        return dict(result)

    async def inspect(self, **kwargs: Any) -> dict[str, Any]:
        """Inspect one durable DSH session without mutating it."""

        method = getattr(self._controller, "inspect", None)
        if not callable(method):
            raise TypeError("resolver controller cannot inspect a session")
        result = method(**kwargs)
        if hasattr(result, "__await__"):
            result = await result
        if not isinstance(result, Mapping):
            raise TypeError("resolver inspection returned a non-object")
        return dict(result)

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
        """Delegate one typed Brain continuation to the fenced controller."""

        controller = self._controller
        resume = getattr(controller, "resume_after_interaction", None)
        if not callable(resume):
            raise TypeError("resolver controller cannot continue an interaction")
        result = resume(
            resolution_thread_id=resolution_thread_id,
            segment_id=segment_id,
            activation_id=activation_id,
            lease_epoch=lease_epoch,
            interaction_id=interaction_id,
            continuation_delta=continuation_delta,
            continuation_authority_token=continuation_authority_token,
        )
        if hasattr(result, "__await__"):
            result = await result
        if not isinstance(result, Mapping):
            raise TypeError("resolver continuation returned a non-object")
        return dict(result)

    def new_runtime_authority(
        self,
        *,
        objective_ref: str,
        brain_conversation_ref: str,
        service_scope: Mapping[str, Any],
        audience: Mapping[str, Any],
        interaction_issuer: str,
    ) -> dict[str, object]:
        """Create model-hidden identity fields for one V2 intake.

        The caller owns the real Brain conversation and service scope.  This
        method only adds the host-issued activation envelope that the sidecar
        verifies before mounting an Agent.
        """

        route = load_agentic_resolver_route_settings()
        workspace = os.environ.get("AGENTIC_RESOLVER_WORKSPACE_ROOT")
        if workspace is None or not workspace:
            raise ValueError("AGENTIC_RESOLVER_WORKSPACE_ROOT is required")
        workspace_path = Path(workspace)
        if not workspace_path.is_absolute():
            raise ValueError("AGENTIC_RESOLVER_WORKSPACE_ROOT must be absolute")
        if not isinstance(objective_ref, str) or not objective_ref.strip():
            raise ValueError("objective_ref must be non-empty")
        if not isinstance(brain_conversation_ref, str) or not brain_conversation_ref.strip():
            raise ValueError("brain_conversation_ref must be non-empty")
        if not isinstance(interaction_issuer, str) or not interaction_issuer.strip():
            raise ValueError("interaction_issuer must be non-empty")
        expected_scope = {"platform", "platform_channel_id", "global_user_id"}
        if not isinstance(service_scope, Mapping) or set(service_scope) != expected_scope:
            raise ValueError("service_scope must contain platform, platform_channel_id, and global_user_id")
        if not isinstance(audience, Mapping) or not audience:
            raise ValueError("audience must be a non-empty object")
        for key in expected_scope:
            value = service_scope[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"service_scope.{key} must be non-empty")
        operation_id = f"op_{uuid4().hex}"
        resolution_thread_id = f"res_{uuid4().hex}"
        segment_id = f"seg_{uuid4().hex}"
        lease_epoch = 1
        payload_digest = operation_payload_digest({
            "method": "resolution.open",
            "params": {
                "objective_ref": objective_ref,
                "brain_conversation_ref": brain_conversation_ref,
                "service_scope": dict(service_scope),
                "audience": dict(audience),
            },
        })
        interaction_scope = scope_fingerprint(service_scope)
        audience_digest = audience_fingerprint(audience)
        canonical_workspace = str(workspace_path).replace("\\", "/")
        now = datetime.now(UTC)
        issued_at = now.isoformat().replace("+00:00", "Z")
        expires_at = (now + timedelta(minutes=5)).isoformat().replace(
            "+00:00", "Z"
        )
        semantic_digest = semantic_catalog_digest()
        issued_reference_digest = content_digest({
            "resolution_thread_id": resolution_thread_id,
            "segment_id": segment_id,
            "brain_conversation_ref": brain_conversation_ref,
            "service_scope": dict(service_scope),
            "audience_fingerprint": audience_digest,
        })
        activation_id = activation_id_for(
            resolution_thread_id, segment_id, lease_epoch
        )
        semantic_secret = os.environ.get("KAZUSA_DSH_TOOL_GATEWAY_SECRET")
        if semantic_secret is None or not semantic_secret:
            raise ValueError("KAZUSA_DSH_TOOL_GATEWAY_SECRET is required")
        activation = SemanticActivationAuthorityV1(
            activation_id=activation_id,
            lease_epoch=lease_epoch,
            resolution_thread_id=resolution_thread_id,
            segment_id=segment_id,
            brain_conversation_ref=brain_conversation_ref,
            service_scope=dict(service_scope),
            scope_fingerprint=interaction_scope,
            audience_fingerprint=audience_digest,
            workspace_root=canonical_workspace,
            route_digest=route.route_digest,
            catalog_digest=semantic_digest,
            profile_version=PROFILE_VERSION,
            model_route_digest=route.route_digest,
            workspace_fingerprint=content_digest({
                "workspace_root": canonical_workspace,
            }),
            issued_reference_digest=issued_reference_digest,
            policy_epoch="dsh-standard-policy-v2",
            interaction_issuer=interaction_issuer,
            issued_at=issued_at,
            expires_at=expires_at,
            token_id=f"tok_{uuid4().hex}",
            nonce=f"nonce_{uuid4().hex}",
        )
        token = issue_activation_token(
            activation,
            secret=semantic_secret.encode("utf-8"),
            now=issued_at,
        )
        return {
            "request_id": f"rrq_{uuid4().hex}",
            "operation_id": operation_id,
            "operation_payload_digest": payload_digest,
            "resolution_thread_id": resolution_thread_id,
            "segment_id": segment_id,
            "activation_id": activation_id,
            "lease_epoch": lease_epoch,
            "brain_conversation_ref": brain_conversation_ref,
            "workspace_root": canonical_workspace,
            "route_digest": route.route_digest,
            "semantic_tool_authority": {
                "catalog_digest": semantic_digest,
                "token": token,
            },
            "interaction_authority": {
                "issuer": interaction_issuer,
                "scope_fingerprint": interaction_scope,
                "audience_fingerprint": audience_digest,
            },
            "resolver_profile_version": PROFILE_VERSION,
            "dsh_release": DSH_RELEASE,
            "session_store_epoch": SESSION_STORE_EPOCH,
            "model_route": route.model,
            "model_route_digest": route.route_digest,
            "semantic_catalog_digest": semantic_digest,
            "audience_fingerprint": audience_digest,
        }

    def issue_continuation_authority(
        self,
        *,
        request: Mapping[str, Any] | Any,
        row: Mapping[str, Any],
        grant: Mapping[str, Any] | Any | None = None,
    ) -> str:
        """Issue a fresh canonical activation token for a Brain continuation."""

        del grant
        if not isinstance(row, Mapping):
            raise TypeError("continuation authority row is required")
        request_mapping = (
            request.to_dict()
            if hasattr(request, "to_dict")
            else dict(request)
            if isinstance(request, Mapping)
            else None
        )
        if not isinstance(request_mapping, Mapping):
            raise TypeError("continuation authority request is invalid")
        required_row_fields = (
            "resolution_thread_id", "segment_id", "activation_id",
            "lease_epoch", "brain_conversation_ref", "platform",
            "platform_channel_id", "global_user_id", "scope_fingerprint",
            "audience_fingerprint", "workspace_fingerprint",
            "model_route_digest", "catalog_digest", "profile_version",
            "policy_epoch", "issued_reference_digest",
        )
        if any(
            not isinstance(row.get(field), (str, int))
            or (isinstance(row.get(field), str) and not row[field].strip())
            for field in required_row_fields
        ):
            raise ValueError("continuation authority row identity is incomplete")
        if not isinstance(row["lease_epoch"], int) or isinstance(
            row["lease_epoch"], bool
        ) or row["lease_epoch"] < 1:
            raise ValueError("continuation authority lease_epoch is invalid")
        issuer = request_mapping.get("issuer")
        if not isinstance(issuer, str) or not issuer.strip():
            raise ValueError("continuation authority issuer is required")
        route = load_agentic_resolver_route_settings()
        workspace = os.environ.get("AGENTIC_RESOLVER_WORKSPACE_ROOT")
        if workspace is None or not workspace:
            raise ValueError("AGENTIC_RESOLVER_WORKSPACE_ROOT is required")
        workspace_path = Path(workspace)
        if not workspace_path.is_absolute():
            raise ValueError("AGENTIC_RESOLVER_WORKSPACE_ROOT must be absolute")
        canonical_workspace = str(workspace_path).replace("\\", "/")
        expected_workspace_fingerprint = content_digest({
            "workspace_root": canonical_workspace,
        })
        if row["workspace_fingerprint"] != expected_workspace_fingerprint:
            raise ValueError("continuation workspace fence does not match")
        if row["model_route_digest"] != route.route_digest:
            raise ValueError("continuation route fence does not match")
        if row["profile_version"] != PROFILE_VERSION:
            raise ValueError("continuation profile fence does not match")
        if row["policy_epoch"] != "dsh-standard-policy-v2":
            raise ValueError("continuation policy fence does not match")
        semantic_digest = semantic_catalog_digest()
        if row["catalog_digest"] != semantic_digest:
            raise ValueError("continuation catalog fence does not match")
        service_scope = {
            "platform": row["platform"],
            "platform_channel_id": row["platform_channel_id"],
            "global_user_id": row["global_user_id"],
        }
        if content_digest(service_scope) != row["scope_fingerprint"]:
            raise ValueError("continuation scope fence does not match")
        now = datetime.now(UTC)
        issued_at = now.isoformat().replace("+00:00", "Z")
        expires_at = (now + timedelta(minutes=5)).isoformat().replace(
            "+00:00", "Z"
        )
        activation = SemanticActivationAuthorityV1(
            activation_id=str(row["activation_id"]),
            lease_epoch=int(row["lease_epoch"]),
            resolution_thread_id=str(row["resolution_thread_id"]),
            segment_id=str(row["segment_id"]),
            brain_conversation_ref=str(row["brain_conversation_ref"]),
            service_scope=service_scope,
            scope_fingerprint=str(row["scope_fingerprint"]),
            audience_fingerprint=str(row["audience_fingerprint"]),
            workspace_root=canonical_workspace,
            route_digest=route.route_digest,
            catalog_digest=semantic_digest,
            profile_version=PROFILE_VERSION,
            model_route_digest=route.route_digest,
            workspace_fingerprint=expected_workspace_fingerprint,
            issued_reference_digest=str(row["issued_reference_digest"]),
            policy_epoch="dsh-standard-policy-v2",
            interaction_issuer=issuer,
            issued_at=issued_at,
            expires_at=expires_at,
            token_id=f"tok_{uuid4().hex}",
            nonce=f"nonce_{uuid4().hex}",
        )
        secret = os.environ.get("KAZUSA_DSH_TOOL_GATEWAY_SECRET")
        if secret is None or not secret:
            raise ValueError("KAZUSA_DSH_TOOL_GATEWAY_SECRET is required")
        return issue_activation_token(
            activation,
            secret=secret.encode("utf-8"),
            now=issued_at,
        )

    @staticmethod
    def build_intake(
        authority: Mapping[str, Any],
        *,
        objective: str,
        facts: list[str],
        mode: str = "start",
    ) -> DSHResolutionIntakeV2:
        """Bind model-visible objective/facts to a validated V2 authority."""

        value = {
            **{
                key: authority[key]
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
                if key in authority
            },
            "schema_version": "dsh_resolution_intake.v2",
            "mode": mode,
            "model_input": {"objective": objective, "facts": facts},
        }
        return DSHResolutionIntakeV2.from_mapping(value)
