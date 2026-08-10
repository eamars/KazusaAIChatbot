"""Tests for the brain service health payload."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot import service as service_module


class _FakeCacheRuntime:
    """Fake Cache2 runtime with deterministic agent stats."""

    def get_agent_stats(self) -> list[dict]:
        """Return one sanitized agent stats row.

        Returns:
            List containing one Cache2 agent stats payload.
        """
        return [
            {
                "agent_name": "user_profile_agent",
                "hit_count": 3,
                "miss_count": 1,
                "hit_rate": 0.75,
            }
        ]


class _ReadyAdapterRegistry:
    """Report the debug adapter as available for runtime projection tests."""

    def has(self, platform: str) -> bool:
        """Return whether the requested platform is the debug adapter."""

        return platform == "debug"


def test_action_availability_separates_queue_admission_from_worker_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disabled automatic loop leaves durable accepted-task intake usable."""

    monkeypatch.setattr(
        service_module,
        "BACKGROUND_WORK_WORKER_ENABLED",
        False,
    )
    monkeypatch.setattr(
        service_module,
        "_adapter_registry",
        _ReadyAdapterRegistry(),
    )

    runtime = service_module._action_availability_runtime_for_target(
        platform="debug",
        platform_channel_id="channel-1",
        channel_type="private",
    )

    assert runtime["worker_status"]["accepted_task"] == "healthy"
    assert runtime["worker_status"]["background_work"] == "degraded"
    assert runtime["adapter_target_status"]["debug"] == "healthy"


@pytest.mark.asyncio
async def test_health_includes_cache2_agent_stats(monkeypatch) -> None:
    """Health response should preserve core fields and include Cache2 stats."""

    async def _check_database_connection() -> bool:
        """Return a successful database health result."""
        return True

    monkeypatch.setattr(
        service_module,
        "check_database_connection",
        _check_database_connection,
    )
    monkeypatch.setattr(
        service_module,
        "get_rag_cache2_runtime",
        lambda: _FakeCacheRuntime(),
    )

    response = await service_module.health()

    assert response.status == "ok"
    assert response.db is True
    assert response.scheduler is True
    assert len(response.cache2.agents) == 1
    assert response.cache2.agents[0].agent_name == "user_profile_agent"
    assert response.cache2.agents[0].hit_count == 3
    assert response.cache2.agents[0].miss_count == 1
    assert response.cache2.agents[0].hit_rate == 0.75
