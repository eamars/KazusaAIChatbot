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

    from control_console import app as app_module
    from control_console import repository as repository_module
    from control_console.app import create_app
    from control_console.auth import hash_operator_token
    from control_console.settings import ControlConsoleSettings
    from kazusa_ai_chatbot.brain_service.cognition_observation_contracts import (
        CognitionObservationSectionV1,
    )

    latest_section = CognitionObservationSectionV1(
        section_id="reasoning.context_consumption",
        label="Context consumption",
        category="context",
        presentation="records",
        status="not_reported",
        summary="",
        fields=[],
        records=[],
        reported_record_count=0,
        displayed_record_count=0,
        truncated=False,
    )

    async def latest_context_section(*, states, kazusa_client):
        _ = states, kazusa_client
        return latest_section

    monkeypatch.setattr(
        app_module,
        "_latest_context_section",
        latest_context_section,
    )

    async def character_entity(
        self,
        *,
        current_timestamp_utc: str | None = None,
        latest_context_section: CognitionObservationSectionV1 | None = None,
        include_operational_context: bool = False,
        limit: int = 25,
    ):
        _ = self
        assert current_timestamp_utc
        assert latest_context_section is not None
        assert latest_context_section.section_id == (
            "reasoning.context_consumption"
        )
        assert latest_context_section.status == "not_reported"
        assert include_operational_context is True
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
        platform_channel_id: str = "",
        channel_type: str = "",
        query: str,
        current_timestamp_utc: str | None = None,
        limit: int,
    ):
        _ = self
        assert platform == ""
        assert platform_user_id == ""
        assert platform_channel_id == ""
        assert channel_type == ""
        assert query == ""
        assert current_timestamp_utc
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
        participant_platform_user_id: str = "",
        current_timestamp_utc: str | None = None,
        limit: int,
    ):
        _ = self
        assert platform == ""
        assert group_id == ""
        assert participant_platform_user_id == ""
        assert current_timestamp_utc
        assert limit == 5
        return {
            "status": "needs_input",
            "owner": "group",
            "identity": {},
            "panels": {"style": {"items": []}},
            "redaction": {"model_inputs": "excluded"},
        }

    async def list_user_entities(self, *, limit: int):
        _ = self
        assert limit == 5
        return {
            "status": "empty",
            "items": [],
            "redaction": {"internal_global_ids": "excluded"},
        }

    async def list_group_entities(self, *, limit: int):
        _ = self
        assert limit == 5
        return {
            "status": "empty",
            "items": [],
            "redaction": {"raw_messages": "excluded"},
        }

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
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "list_user_entities",
        list_user_entities,
        raising=False,
    )
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "list_group_entities",
        list_group_entities,
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
    assert generic.status_code == 404

    rejected_user = client.get("/api/entities/users?limit=101")
    assert rejected_user.status_code == 422

    rejected_group = client.get("/api/entities/groups?limit=101")
    assert rejected_group.status_code == 422

    character_entity_page = client.get("/api/entities/character?limit=5")
    assert character_entity_page.status_code == 200
    assert character_entity_page.json()["owner"] == "character"

    user_entity_page = client.get("/api/entities/users?limit=5")
    assert user_entity_page.status_code == 200
    assert user_entity_page.json()["status"] == "empty"
    assert user_entity_page.json()["items"] == []

    group_entity_page = client.get("/api/entities/groups?limit=5")
    assert group_entity_page.status_code == 200
    assert group_entity_page.json()["status"] == "empty"
    assert group_entity_page.json()["items"] == []
