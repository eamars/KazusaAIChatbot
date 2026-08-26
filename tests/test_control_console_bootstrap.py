"""Bootstrap route contract tests."""

from __future__ import annotations


def test_bootstrap_wraps_canonical_observations_with_view_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    """Bootstrap should wrap direct Brain observations in view envelopes."""

    from fastapi.testclient import TestClient

    from control_console import app as app_module
    from control_console import repository as repository_module
    from control_console.auth import hash_operator_token
    from control_console.contracts import ServiceRuntimeState
    from control_console.settings import ControlConsoleSettings
    from kazusa_ai_chatbot.brain_service.cognition_observation_contracts import (
        CognitionRunObservationV1,
    )
    from tests.test_control_console_contracts import _canonical_live_observation

    async def application_identity(self):
        _ = self
        return {
            "status": "available",
            "character_name": "Test Character",
            "source": "character_state",
        }

    class RunningBrainSupervisor:
        def all_service_states(self):
            return [ServiceRuntimeState(
                id="brain",
                display_name="Brain service",
                kind="backend",
                actual_state="running",
            )]

    class FakeKazusaClient:
        def __init__(
            self,
            *,
            base_url: str,
            timeout_seconds: float,
            control_shared_secret: str = "",
        ) -> None:
            _ = base_url
            _ = timeout_seconds
            _ = control_shared_secret

        async def get_health(self) -> dict:
            return {"status": "ok", "db": True, "scheduler": True}

        async def get_runtime_status(self) -> dict:
            return {"workers": {}}

        async def get_latest_cognition_graph(self):
            return CognitionRunObservationV1.model_validate(
                _canonical_live_observation(run_id="bootstrap-live-run")
            )

        async def get_latest_self_cognition_graph(self):
            return None

    monkeypatch.setattr(app_module, "KazusaClient", FakeKazusaClient)
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "application_identity",
        application_identity,
    )
    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(app_module.create_app(
        settings=settings,
        supervisor=RunningBrainSupervisor(),
    ))

    assert client.post("/api/auth/login", json={"token": "secret"}).status_code == 200
    response = client.get("/api/bootstrap")

    assert response.status_code == 200
    payload = response.json()
    latest = payload["latest_cognition_observation"]
    assert latest["view_kind"] == "overview_latest"
    assert latest["availability"] == "available"
    assert latest["reason_code"] == ""
    assert latest["observation"]["schema_version"] == (
        "cognition_run_observation.v1"
    )
    assert latest["observation"]["correlation"]["run_id"] == (
        "bootstrap-live-run"
    )
    assert payload["latest_self_cognition_observation"] == {
        "view_kind": "self_latest",
        "availability": "not_reported",
        "reason_code": "brain_not_reported",
        "generated_at": payload["latest_self_cognition_observation"][
            "generated_at"
        ],
        "observation": None,
    }
    assert "latest_cognition_graph" not in payload
    assert "latest_self_cognition_graph" not in payload
    assert "latest_cognition_observation" in payload["overview"]
    assert "latest_self_cognition_observation" in payload["overview"]
    assert "cognition_observations" in payload["overview"]["panels"]
    assert "cognition_graphs" not in payload["overview"]["panels"]


def test_overview_api_returns_only_owner_aggregates(tmp_path) -> None:
    """Overview should expose exceptions and links without owner-page detail."""

    from fastapi.testclient import TestClient

    from control_console.app import create_app
    from control_console.auth import hash_operator_token
    from control_console.settings import ControlConsoleSettings

    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(create_app(settings=settings))
    login = client.post("/api/auth/login", json={"token": "secret"})
    assert login.status_code == 200

    response = client.get("/api/overview")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload["panels"]) == {
        "service_summary",
        "internal_readiness",
        "recent_failures",
        "recent_changes",
        "cognition_observations",
    }
    rendered = repr(payload)
    for duplicated_owner_detail in (
        "brain_health",
        "runtime_status",
        "cache2",
        "event_stream",
        "csrf_header_name",
        "model_routes",
    ):
        assert duplicated_owner_detail not in rendered


def test_bootstrap_returns_initial_state_session_csrf_services_and_stream_url(
    monkeypatch,
    tmp_path,
) -> None:
    """The UI should receive one coherent bootstrap snapshot after login."""

    from fastapi.testclient import TestClient

    from control_console import repository as repository_module
    from control_console.app import create_app
    from control_console.auth import hash_operator_token
    from control_console.settings import ControlConsoleSettings

    async def application_identity(self):
        _ = self
        return {
            "status": "available",
            "character_name": "杏山千纱 (Kyōyama Kazusa)",
            "source": "character_state",
        }

    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "application_identity",
        application_identity,
    )

    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    assert client.get("/api/bootstrap").status_code == 401

    login = client.post("/api/auth/login", json={"token": "secret"})
    assert login.status_code == 200
    login_payload = login.json()

    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 200
    payload = bootstrap.json()
    assert payload["operator"]["operator_id"] == "local_operator"
    assert payload["csrf_header_name"] == "x-kazusa-control-csrf"
    assert payload["csrf_token"] == login_payload["csrf_token"]
    assert payload["application_identity"]["character_name"] == (
        "杏山千纱 (Kyōyama Kazusa)"
    )
    assert {"brain", "adapter.discord", "adapter.debug"} <= {
        service["id"] for service in payload["services"]
    }
    assert payload["stream_url"] == "/api/stream"
    assert "event_stream" not in payload["ui_capabilities"]
    page_capabilities = payload["page_capabilities"]
    assert page_capabilities["overview"]["status"] == "ready"
    assert page_capabilities["events"]["status"] == "ready"
    assert "unsupported" not in page_capabilities["events"]
    for page_name in (
        "character",
        "users",
        "groups",
        "calendar",
        "background",
        "health",
        "audit",
    ):
        assert page_capabilities[page_name]["status"] == "ready"
    assert page_capabilities["character"]["label"] == "canonical cognition"
    assert page_capabilities["users"]["label"] == "directory + cognition"
    assert page_capabilities["groups"]["label"] == "activity + review"


def test_bootstrap_projects_live_health_without_overview_duplication(
    monkeypatch,
    tmp_path,
) -> None:
    """Health should own live readiness, worker, and cache information."""

    from fastapi.testclient import TestClient

    from control_console import app as app_module
    from control_console import repository as repository_module
    from control_console.auth import hash_operator_token
    from control_console.contracts import ServiceRuntimeState
    from control_console.settings import ControlConsoleSettings

    async def application_identity(self):
        _ = self
        return {
            "status": "available",
            "character_name": "Test Character",
            "source": "character_state",
        }

    class RunningBrainSupervisor:
        def all_service_states(self):
            state = ServiceRuntimeState(
                id="brain",
                display_name="Brain service",
                kind="backend",
                actual_state="running",
            )
            return [state]

    class FakeKazusaClient:
        def __init__(
            self,
            *,
            base_url: str,
            timeout_seconds: float,
            control_shared_secret: str = "",
        ) -> None:
            _ = base_url
            _ = timeout_seconds
            _ = control_shared_secret

        async def get_health(self) -> dict:
            return {
                "status": "ok",
                "db": True,
                "scheduler": True,
                "cache2": {
                    "agents": [
                        {
                            "agent_name": "memory_agent",
                            "hit_count": 4,
                            "miss_count": 1,
                            "hit_rate": 0.8,
                        },
                    ],
                },
            }

        async def get_runtime_status(self) -> dict:
            return {
                "semantic_descriptors": {
                    "worker_error_level": "ok",
                },
                "workers": {
                    "calendar": {
                        "enabled": True,
                        "task_alive": True,
                        "last_status": "succeeded",
                        "last_event_at": "2026-07-27T01:02:03+00:00",
                    },
                    "reflection": {
                        "enabled": False,
                        "task_alive": False,
                        "last_status": "disabled",
                        "last_event_at": "",
                    },
                },
            }

        async def get_latest_cognition_graph(self):
            from kazusa_ai_chatbot.brain_service.cognition_observation_contracts import (
                CognitionRunObservationV1,
            )
            from tests.test_control_console_contracts import _canonical_live_observation

            return CognitionRunObservationV1.model_validate(
                _canonical_live_observation(run_id="turn-123")
            )

    monkeypatch.setattr(app_module, "KazusaClient", FakeKazusaClient)
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "application_identity",
        application_identity,
    )
    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    app = app_module.create_app(
        settings=settings,
        supervisor=RunningBrainSupervisor(),
    )
    client = TestClient(app)

    login = client.post("/api/auth/login", json={"token": "secret"})
    assert login.status_code == 200

    bootstrap = client.get("/api/bootstrap")

    assert bootstrap.status_code == 200
    payload = bootstrap.json()
    health = payload["health"]
    assert set(health["panels"]) == {
        "readiness",
        "workers",
        "cache_agents",
    }
    readiness = health["panels"]["readiness"]["items"][0]
    assert readiness == {
        "status": "ok",
        "database": True,
        "scheduler": True,
        "worker_error_level": "ok",
    }
    workers = health["panels"]["workers"]["items"]
    assert workers == [
        {
            "worker_name": "calendar",
            "enabled": True,
            "task_alive": True,
            "last_status": "succeeded",
            "last_event_at": "2026-07-27T01:02:03+00:00",
        },
        {
            "worker_name": "reflection",
            "enabled": False,
            "task_alive": False,
            "last_status": "disabled",
            "last_event_at": "",
        },
    ]
    assert health["panels"]["cache_agents"]["items"] == [
        {
            "agent_name": "memory_agent",
            "hits": 4,
            "misses": 1,
            "total": 5,
            "hit_rate": 0.8,
        },
    ]
    overview = payload["overview"]
    assert set(overview["panels"]) == {
        "service_summary",
        "internal_readiness",
        "recent_failures",
        "recent_changes",
        "cognition_observations",
    }
    assert "workers" not in overview["panels"]
    assert "cache_agents" not in overview["panels"]
    observations = overview["panels"]["cognition_observations"]["items"]
    assert observations[0]["view"]["observation"]["correlation"]["run_id"] == (
        "turn-123"
    )
    assert payload["latest_cognition_observation"]["observation"][
        "correlation"
    ]["run_id"] == "turn-123"


def test_bootstrap_projects_live_health_when_brain_is_unmanaged(
    monkeypatch,
    tmp_path,
) -> None:
    """A live unmanaged brain endpoint should still feed health summaries."""

    from fastapi.testclient import TestClient

    from control_console import app as app_module
    from control_console import repository as repository_module
    from control_console.auth import hash_operator_token
    from control_console.contracts import ServiceRuntimeState
    from control_console.settings import ControlConsoleSettings

    async def application_identity(self):
        _ = self
        return {
            "status": "available",
            "character_name": "Test Character",
            "source": "character_state",
        }

    class UnmanagedBrainSupervisor:
        def all_service_states(self):
            state = ServiceRuntimeState(
                id="brain",
                display_name="Brain service",
                kind="backend",
                actual_state="conflict",
                last_error_preview=(
                    "configured endpoint is already in use by an unmanaged process"
                ),
            )
            return [state]

    class FakeKazusaClient:
        def __init__(
            self,
            *,
            base_url: str,
            timeout_seconds: float,
            control_shared_secret: str = "",
        ) -> None:
            _ = base_url
            _ = timeout_seconds
            _ = control_shared_secret

        async def get_health(self) -> dict:
            return {"status": "ok", "db": True}

        async def get_runtime_status(self) -> dict:
            return {
                "semantic_descriptors": {
                    "worker_error_level": "ok",
                },
            }

        async def get_latest_cognition_graph(self):
            return None

    monkeypatch.setattr(app_module, "KazusaClient", FakeKazusaClient)
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "application_identity",
        application_identity,
    )
    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    app = app_module.create_app(
        settings=settings,
        supervisor=UnmanagedBrainSupervisor(),
    )
    client = TestClient(app)

    login = client.post("/api/auth/login", json={"token": "secret"})
    assert login.status_code == 200

    bootstrap = client.get("/api/bootstrap")

    assert bootstrap.status_code == 200
    payload = bootstrap.json()
    readiness = payload["health"]["panels"]["readiness"]["items"][0]
    assert readiness["status"] == "ok"
    assert readiness["database"] is True
    assert readiness["worker_error_level"] == "ok"
    failures = payload["overview"]["panels"]["recent_failures"]["items"]
    assert failures[0]["outcome"] == "conflict"


def test_bootstrap_does_not_query_brain_for_stale_unowned_conflict(
    monkeypatch,
    tmp_path,
) -> None:
    """Only live endpoint conflicts should make brain HTTP calls available."""

    from fastapi.testclient import TestClient

    from control_console import app as app_module
    from control_console import repository as repository_module
    from control_console.auth import hash_operator_token
    from control_console.contracts import ServiceRuntimeState
    from control_console.settings import ControlConsoleSettings

    async def application_identity(self):
        _ = self
        return {
            "status": "available",
            "character_name": "Test Character",
            "source": "character_state",
        }

    class StaleConflictSupervisor:
        def all_service_states(self):
            state = ServiceRuntimeState(
                id="brain",
                display_name="Brain service",
                kind="backend",
                actual_state="conflict",
                last_error_preview="no console-owned process handle",
            )
            return [state]

    class FailingKazusaClient:
        def __init__(
            self,
            *,
            base_url: str,
            timeout_seconds: float,
            control_shared_secret: str = "",
        ) -> None:
            _ = base_url
            _ = timeout_seconds
            _ = control_shared_secret

        async def get_health(self) -> dict:
            raise AssertionError("stale conflicts must not query brain health")

        async def get_runtime_status(self) -> dict:
            raise AssertionError("stale conflicts must not query runtime status")

        async def get_latest_cognition_graph(self):
            raise AssertionError("stale conflicts must not query latest graph")

    monkeypatch.setattr(app_module, "KazusaClient", FailingKazusaClient)
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "application_identity",
        application_identity,
    )
    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    app = app_module.create_app(
        settings=settings,
        supervisor=StaleConflictSupervisor(),
    )
    client = TestClient(app)

    login = client.post("/api/auth/login", json={"token": "secret"})
    assert login.status_code == 200

    bootstrap = client.get("/api/bootstrap")

    assert bootstrap.status_code == 200
    payload = bootstrap.json()
    readiness = payload["health"]["panels"]["readiness"]
    assert readiness["status"] == "unavailable"
    assert readiness["reason"] == "brain service is conflict"
    overview_readiness = payload["overview"]["panels"]["internal_readiness"]
    assert overview_readiness["status"] == readiness["status"]
    assert overview_readiness["items"] == readiness["items"]
    assert overview_readiness["reason"] == readiness["reason"]
