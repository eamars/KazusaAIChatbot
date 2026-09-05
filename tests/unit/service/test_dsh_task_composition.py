"""Executable tests for shared Brain/DSH task composition."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_lifespan_injects_one_shared_runtime_into_interaction_and_task_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composition root wires one resolver runtime into every owner."""

    from kazusa_ai_chatbot import service

    runtime = object()
    cognition_services = object()
    interaction_service = object()
    calls: dict[str, object] = {}

    monkeypatch.setattr(service, "_dsh_interaction_service", None)
    monkeypatch.setattr(service, "_dsh_resolver_runtime", None)
    monkeypatch.setattr(service, "_dsh_cognition_services", None)
    monkeypatch.setattr(service, "build_cognition_core_services", lambda: cognition_services)
    monkeypatch.setattr(
        service.AgenticResolverRuntime,
        "from_environment",
        classmethod(lambda _cls: runtime),
    )

    def build_interaction_service(**kwargs: object) -> object:
        calls.update(kwargs)
        return interaction_service

    monkeypatch.setattr(
        service.BrainInteractionService,
        "from_environment",
        classmethod(lambda _cls, **kwargs: build_interaction_service(**kwargs)),
    )
    monkeypatch.setenv("KAZUSA_DSH_BRAIN_SHARED_SECRET", "brain-secret")
    monkeypatch.setenv("KAZUSA_DSH_TOOL_GATEWAY_SECRET", "gateway-secret")

    await service._configure_dsh_interaction_service_from_lifespan()

    assert service._dsh_resolver_runtime is runtime
    assert service._dsh_cognition_services is cognition_services
    assert service._dsh_interaction_service is interaction_service
    assert set(calls) == {"judge", "context_provider"}
    assert isinstance(calls["judge"], service.BrainDecisionEngine)
    assert callable(calls["context_provider"])

    from kazusa_ai_chatbot.background_work.subagent import task_orchestrator

    assert task_orchestrator._TASK_RESOLUTION_RUNTIME is runtime


