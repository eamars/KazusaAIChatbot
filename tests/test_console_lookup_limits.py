"""Lookup route bound and redaction tests."""

from __future__ import annotations


def _login(client):
    """Authenticate a test client and return CSRF metadata."""

    login = client.post("/api/auth/login", json={"token": "secret"})
    payload = login.json()
    return payload["csrf_header_name"], payload["csrf_token"]


def test_lookup_routes_enforce_pagination_redaction_and_no_embeddings(
    monkeypatch,
    tmp_path,
) -> None:
    """Read-only lookup routes should reject unbounded limits."""

    from fastapi.testclient import TestClient

    from control_console import repository as repository_module
    from control_console.app import create_app
    from control_console.auth import hash_operator_token
    from control_console.settings import ControlConsoleSettings

    async def latest_character_status(self):
        _ = self
        return {"status": "empty", "items": []}

    async def global_growth_summary(self):
        _ = self
        return {"status": "empty", "items": []}

    async def character_entity(self, *, limit: int = 25):
        _ = self
        assert limit == 5
        return {
            "status": "empty",
            "owner": "character",
            "identity": {},
            "panels": {},
            "redaction": {"model_inputs": "excluded"},
        }

    async def lookup_user_entity(
        self,
        *,
        platform: str,
        platform_user_id: str,
        query: str,
        limit: int,
    ):
        _ = self
        assert platform == ""
        assert platform_user_id == ""
        assert query == ""
        assert limit == 5
        return {
            "status": "needs_input",
            "owner": "user",
            "identity": {},
            "panels": {"memory": {"items": []}},
            "redaction": {"embeddings": "excluded"},
        }

    async def lookup_group_entity(
        self,
        *,
        platform: str,
        group_id: str,
        limit: int,
    ):
        _ = self
        assert platform == ""
        assert group_id == ""
        assert limit == 5
        return {
            "status": "needs_input",
            "owner": "group",
            "identity": {},
            "panels": {"style": {"items": []}},
            "redaction": {"model_inputs": "excluded"},
        }

    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "latest_character_status",
        latest_character_status,
    )
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "global_growth_summary",
        global_growth_summary,
    )
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "character_entity",
        character_entity,
        raising=False,
    )
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "lookup_user_entity",
        lookup_user_entity,
        raising=False,
    )
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "lookup_group_entity",
        lookup_group_entity,
        raising=False,
    )

    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(create_app(settings=settings))
    _login(client)

    rejected = client.get("/api/lookups/memory?limit=101")
    assert rejected.status_code == 422

    rejected_memory_query = client.get(
        f"/api/lookups/memory?query={'x' * 241}&limit=5",
    )
    assert rejected_memory_query.status_code == 422

    accepted = client.get("/api/lookups/memory?limit=5")
    assert accepted.status_code == 200
    payload = accepted.json()
    assert payload["items"] == []
    assert payload["status"] == "needs_input"
    assert payload["redaction"]["embeddings"] == "excluded"
    assert "embedding" not in repr(payload["items"]).lower()

    style = client.get("/api/lookups/style?limit=5")
    assert style.status_code == 200
    style_payload = style.json()
    assert style_payload["items"] == []
    assert style_payload["status"] == "needs_input"
    assert style_payload["redaction"]["source_run_ids"] == "excluded"

    generic = client.get("/api/lookups/not-yet-wired?limit=5")
    assert generic.status_code == 200
    generic_payload = generic.json()
    assert generic_payload["items"] == []
    assert generic_payload["redaction"]["namespace"] == "not-yet-wired"

    character = client.get("/api/character/status")
    assert character.status_code == 200
    assert character.json()["status"] in {"unavailable", "empty", "available"}

    growth = client.get("/api/character/growth")
    assert growth.status_code == 200
    assert growth.json()["status"] in {"unavailable", "empty", "available"}

    rejected_user = client.get("/api/entities/user?limit=101")
    assert rejected_user.status_code == 422

    rejected_user_query = client.get(
        f"/api/entities/user?query={'x' * 241}&limit=5",
    )
    assert rejected_user_query.status_code == 422

    character_entity_page = client.get("/api/entities/character?limit=5")
    assert character_entity_page.status_code == 200
    assert character_entity_page.json()["owner"] == "character"

    user_entity_page = client.get("/api/entities/user?limit=5")
    assert user_entity_page.status_code == 200
    assert user_entity_page.json()["owner"] == "user"
    assert user_entity_page.json()["status"] == "needs_input"

    group_entity_page = client.get("/api/entities/group?limit=5")
    assert group_entity_page.status_code == 200
    assert group_entity_page.json()["owner"] == "group"
    assert group_entity_page.json()["status"] == "needs_input"
