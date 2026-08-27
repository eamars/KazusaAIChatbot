"""Preserved standalone runtime entry point for the DSH sidecar."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from agentic_resolver.contracts import (
    DSH_RELEASE,
    PROFILE_VERSION,
    SESSION_STORE_EPOCH,
    DSHResolutionExhaustV1,
)
from agentic_resolver.controller import ResolutionController
from agentic_resolver.fingerprints import (
    audience_fingerprint,
    operation_payload_digest,
    scope_fingerprint,
    tool_catalog_digest,
)
from agentic_resolver.persistence import MongoResolutionThreadRepository
from agentic_resolver.rpc import DSHRpcClient


class ResolverController(Protocol):
    async def resolve(self, intake: Mapping[str, Any]) -> Mapping[str, Any]:
        """Resolve one canonical intake through the independent sidecar."""


class AgenticResolverRuntime:
    """Small public runtime that delegates lifecycle work to its controller."""

    def __init__(self, controller: ResolverController) -> None:
        self._controller = controller

    @classmethod
    def from_environment(
        cls, *, data_root: Path | None = None
    ) -> AgenticResolverRuntime:
        if data_root is not None and not data_root.is_absolute():
            raise ValueError("data_root must be absolute")
        endpoint = os.environ.get("KAZUSA_DSH_SIDECAR_URL", "")
        token = os.environ.get("KAZUSA_DSH_RPC_TOKEN", "")
        controller = ResolutionController(
            MongoResolutionThreadRepository(),
            DSHRpcClient(endpoint, token),
            owner_id=f"resolver_{uuid4().hex}",
        )
        return cls(controller)

    async def resolve(
        self, intake: Mapping[str, Any]
    ) -> DSHResolutionExhaustV1:
        result = await self._controller.resolve(intake)
        value = result.get("exhaust", result)
        return DSHResolutionExhaustV1.from_mapping(value)

    def new_runtime_authority(
        self,
        *,
        objective_ref: str,
        scope: Mapping[str, Any],
        audience: Mapping[str, Any],
    ) -> dict[str, object]:
        operation_id = f"op_{uuid4().hex}"
        payload_digest = operation_payload_digest({
            "method": "resolution.open",
            "params": {
                "objective_ref": objective_ref,
                "scope": dict(scope),
                "audience": dict(audience),
            },
        })
        return {
            "request_id": f"rrq_{uuid4().hex}",
            "operation_id": operation_id,
            "operation_payload_digest": payload_digest,
            "resolution_thread_id": f"res_{uuid4().hex}",
            "segment_id": f"seg_{uuid4().hex}",
            "priority": "now",
            "soft_deadline_at": "2099-01-01T00:00:10Z",
            "hard_deadline_at": "2099-01-01T00:00:30Z",
            "max_model_steps": 4,
            "max_tool_calls": 4,
            "max_tool_bytes": 65536,
            "capability_token": f"cap_{uuid4().hex}",
            "scope_fingerprint": scope_fingerprint(scope),
            "audience_fingerprint": audience_fingerprint(audience),
            "resolver_profile_version": PROFILE_VERSION,
            "dsh_release": DSH_RELEASE,
            "session_store_epoch": SESSION_STORE_EPOCH,
            "model_route": os.environ.get("KAZUSA_DSH_MODEL", ""),
            "tool_catalog_digest": tool_catalog_digest([
                {"name": "submit_resolution", "version": "1"}
            ]),
            "policy_epoch": "kazusa-resolver-v1.1",
        }
