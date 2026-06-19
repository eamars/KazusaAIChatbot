"""Bootstrap route contract tests."""

from __future__ import annotations


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
    assert payload["ui_capabilities"]["event_stream"] is True
    page_capabilities = payload["page_capabilities"]
    assert page_capabilities["overview"]["status"] == "ready"
    assert page_capabilities["events"]["status"] == "ready"
    assert "unsupported" not in page_capabilities["events"]
    assert page_capabilities["users"]["status"] == "partial"
    assert page_capabilities["groups"]["status"] == "partial"
    assert page_capabilities["calendar"]["status"] == "partial"
    assert page_capabilities["background"]["status"] == "partial"
    assert "remediated" not in page_capabilities["character"]["reason"]
    assert page_capabilities["character"]["reason"] == (
        "Character profile, state, growth, and safe learning panels are "
        "available; raw reflection output is excluded."
    )


def test_bootstrap_reports_live_brain_health_when_brain_is_running(
    monkeypatch,
    tmp_path,
) -> None:
    """Health/cache overview should use live brain data when brain is running."""

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
        def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
            _ = base_url
            _ = timeout_seconds

        async def get_health(self) -> dict:
            return {
                "status": "ok",
                "db": True,
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
                "worker_error_level": "ok",
                "workers": {"calendar": "running"},
            }

        async def get_latest_cognition_graph(self):
            from control_console.kazusa_client import project_cognition_graph_snapshot

            return project_cognition_graph_snapshot(
                source="overview_latest",
                payload={
                    "cognition_graph": {
                        "run_id": "turn-123",
                        "status": "completed",
                        "nodes": [
                            {
                                "id": "l2.reasoning",
                                "label": "Reasoning",
                                "stage": "L2",
                                "lane": "cognition",
                                "column": 3,
                                "branch": "reasoning",
                                "status": "completed",
                                "detail": {
                                    "internal_monologue": "bounded reason",
                                },
                            },
                        ],
                        "edges": [],
                    },
                },
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
    overview = bootstrap.json()["overview"]
    assert overview["brain_health"]["status"] == "ok"
    assert overview["brain_health"]["db"] is True
    assert overview["cache2"]["agents"][0]["agent_name"] == "memory_agent"
    assert overview["runtime_status"]["worker_error_level"] == "ok"
    assert overview["latest_cognition_graph"]["run_id"] == "turn-123"
    assert overview["latest_cognition_graph"]["nodes"][0]["id"] == "l2.reasoning"
    assert bootstrap.json()["latest_cognition_graph"]["run_id"] == "turn-123"


def test_bootstrap_reports_live_brain_health_when_brain_is_unmanaged(
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
        def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
            _ = base_url
            _ = timeout_seconds

        async def get_health(self) -> dict:
            return {"status": "ok", "db": True}

        async def get_runtime_status(self) -> dict:
            return {"worker_error_level": "ok"}

        async def get_latest_cognition_graph(self):
            from control_console.kazusa_client import not_reported_cognition_graph

            return not_reported_cognition_graph(source="overview_latest")

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
    overview = bootstrap.json()["overview"]
    assert overview["brain_health"]["status"] == "ok"
    assert overview["brain_health"]["db"] is True
    assert overview["runtime_status"]["worker_error_level"] == "ok"


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
        def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
            _ = base_url
            _ = timeout_seconds

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
    overview = bootstrap.json()["overview"]
    assert overview["brain_health"]["status"] == "unavailable"
    assert overview["brain_health"]["reason"] == "brain service is conflict"
