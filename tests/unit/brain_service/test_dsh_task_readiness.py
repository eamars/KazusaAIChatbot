"""Executable tests for the Brain-side DSH task readiness contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_resolver.errors import RpcTransportError


def test_task_capability_is_available_only_when_full_dsh_runtime_is_ready() -> None:
    """Readiness validates the sidecar and Brain bridge as one closed carrier."""

    from kazusa_ai_chatbot.brain_service.contracts import (
        DshInteractionHealthResponseV1,
    )

    ready_payload = {
        "schema_version": "dsh_brain_interaction_health.v1",
        "status": "ready",
        "configured": True,
        "durable_store": True,
        "cognition_judge": True,
        "task_resolution": {
            "status": "ready",
            "sidecar_identity": "sidecar-v2",
            "brain_bridge_identity": "brain-v2",
        },
    }
    health = DshInteractionHealthResponseV1.model_validate(ready_payload)
    assert health.task_resolution.status == "ready"
    assert health.task_resolution.sidecar_identity == "sidecar-v2"

    unavailable_payload = {
        **ready_payload,
        "task_resolution": {
            "status": "unavailable",
            "sidecar_identity": "sidecar-v2",
            "brain_bridge_identity": "brain-v2",
        },
    }
    unavailable = DshInteractionHealthResponseV1.model_validate(unavailable_payload)
    assert unavailable.task_resolution.status == "unavailable"

    with pytest.raises(ValidationError):
        DshInteractionHealthResponseV1.model_validate({
            **ready_payload,
            "task_resolution": {
                "status": "ready",
                "sidecar_identity": "",
                "brain_bridge_identity": "brain-v2",
            },
        })


class _InteractionService:
    """Provide the local Brain owners required by the health projection."""

    def __init__(self) -> None:
        self._interaction_store = object()
        self._judge = object()


class _ReadyRuntime:
    """Return one valid authenticated sidecar readiness identity."""

    async def readiness(self) -> dict[str, str]:
        return {
            "status": "ready",
            "route_digest": "route-digest",
            "semantic_catalog_digest": "catalog-digest",
        }


class _UnavailableRuntime:
    """Represent an unreachable configured DSH sidecar."""

    async def readiness(self) -> dict[str, str]:
        raise RpcTransportError("sidecar unavailable: connection refused")


@pytest.mark.asyncio
async def test_dsh_health_requires_live_sidecar_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local Brain construction is insufficient without sidecar readiness."""

    from kazusa_ai_chatbot import service

    monkeypatch.setattr(service, "MongoInteractionStore", object)
    monkeypatch.setattr(service, "_dsh_interaction_service", _InteractionService())
    monkeypatch.setattr(service, "_dsh_resolver_runtime", _ReadyRuntime())

    health = await service._dsh_interaction_health()

    assert health.status == "ready"
    assert health.task_resolution.status == "ready"


@pytest.mark.asyncio
async def test_dsh_health_is_unavailable_when_sidecar_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable sidecar should revoke Brain task readiness."""

    from kazusa_ai_chatbot import service

    monkeypatch.setattr(service, "MongoInteractionStore", object)
    monkeypatch.setattr(service, "_dsh_interaction_service", _InteractionService())
    monkeypatch.setattr(service, "_dsh_resolver_runtime", _UnavailableRuntime())

    health = await service._dsh_interaction_health()

    assert health.status == "unavailable"
    assert health.configured is True
    assert health.durable_store is True
    assert health.cognition_judge is True
    assert health.task_resolution.status == "unavailable"


def test_dsh_bridge_health_does_not_depend_on_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sidecar's Brain probe must terminate at local bridge owners."""

    from kazusa_ai_chatbot import service

    monkeypatch.setattr(service, "MongoInteractionStore", object)
    monkeypatch.setattr(service, "_dsh_interaction_service", _InteractionService())
    monkeypatch.setattr(service, "_dsh_resolver_runtime", _UnavailableRuntime())

    health = service._dsh_brain_bridge_health()

    assert health.schema_version == "dsh_brain_bridge_health.v1"
    assert health.status == "ready"
