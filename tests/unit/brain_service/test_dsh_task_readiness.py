"""Executable tests for the Brain-side DSH task readiness contract."""

from __future__ import annotations

import pytest

from agentic_resolver.errors import RpcTransportError




class _InteractionService:
    """Provide the local Brain owners required by the health projection."""

    def __init__(self) -> None:
        self._interaction_store = object()
        self._judge = object()




class _UnavailableRuntime:
    """Represent an unreachable configured DSH sidecar."""

    async def readiness(self) -> dict[str, str]:
        raise RpcTransportError("sidecar unavailable: connection refused")




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


