"""Tests for the brain service health payload."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot import service as service_module


class _FakeAdmin:
    """Fake Mongo admin object that accepts a ping command."""

    async def command(self, name: str) -> dict:
        """Return a successful response for one admin command.

        Args:
            name: Mongo admin command name.

        Returns:
            Minimal command response.
        """
        assert name == "ping"
        return {"ok": 1}


class _FakeClient:
    """Fake Mongo client wrapper used by the service health check."""

    def __init__(self) -> None:
        """Create a fake client with an admin command surface."""
        self.admin = _FakeAdmin()


class _FakeDb:
    """Fake Mongo database wrapper used by the service health check."""

    def __init__(self) -> None:
        """Create a fake database with a client attribute."""
        self.client = _FakeClient()


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


@pytest.mark.asyncio
async def test_health_includes_cache2_agent_stats(monkeypatch) -> None:
    """Health response should preserve core fields and include Cache2 stats."""

    async def _get_db() -> _FakeDb:
        """Return a fake database for the health check.

        Returns:
            Fake database object.
        """
        return _FakeDb()

    monkeypatch.setattr(service_module, "get_db", _get_db)
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
