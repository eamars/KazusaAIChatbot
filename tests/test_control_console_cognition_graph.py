"""Cognition-chain graph contract tests for the control console."""

from __future__ import annotations


def test_cognition_graph_snapshot_projects_parallel_branches_and_redacts() -> None:
    """Graph projection should preserve safe branch structure only."""

    from control_console.kazusa_client import project_cognition_graph_snapshot

    payload = {
        "delivery_tracking_id": "run-123",
        "cognition_graph": {
            "run_id": "run-123",
            "status": "completed",
            "nodes": [
                {
                    "id": "intake",
                    "label": "Intake",
                    "stage": "L1",
                    "lane": "input",
                    "column": 1,
                    "status": "completed",
                    "detail": {
                        "summary": "Normalized debug input.",
                        "message_text": "must not leak",
                    },
                },
                {
                    "id": "l2.reasoning",
                    "label": "Reasoning",
                    "stage": "L2",
                    "lane": "cognition",
                    "column": 2,
                    "branch": "l2a",
                    "status": "completed",
                    "detail": {
                        "summary": "User wants bounded inspection.",
                        "internal_monologue": "Keep it short and grounded.",
                        "prompt": "must not leak",
                    },
                },
                {
                    "id": "l2.memory",
                    "label": "Memory check",
                    "stage": "L2",
                    "lane": "memory",
                    "column": 2,
                    "branch": "l2b",
                    "status": "completed",
                    "detail": {
                        "summary": "No memory write requested.",
                        "embedding": [0.1, 0.2],
                    },
                },
                {
                    "id": "surface",
                    "label": "Visible response",
                    "stage": "L3",
                    "lane": "surface",
                    "column": 3,
                    "status": "completed",
                    "detail": {"summary": "Returned visible answer."},
                },
            ],
            "edges": [
                {"source": "intake", "target": "l2.reasoning", "kind": "fork"},
                {"source": "intake", "target": "l2.memory", "kind": "fork"},
                {"source": "l2.reasoning", "target": "surface", "kind": "join"},
                {"source": "l2.memory", "target": "surface", "kind": "join"},
            ],
        },
    }

    graph = project_cognition_graph_snapshot(
        source="debug_latest",
        payload=payload,
    )

    assert graph.status == "completed"
    assert graph.run_id == "run-123"
    assert {node.branch for node in graph.nodes} >= {"l2a", "l2b"}
    assert {edge.kind for edge in graph.edges} == {"fork", "join"}
    graph_text = repr(graph.model_dump(mode="json"))
    detail_text = repr([node.detail for node in graph.nodes])
    assert "User wants bounded inspection." in graph_text
    assert "must not leak" not in detail_text
    assert "embedding" not in detail_text
    assert "prompt" not in detail_text


def test_cognition_graph_snapshot_reports_absent_telemetry_without_dummy_nodes() -> None:
    """No cognition telemetry should be shown as not reported, not faked."""

    from control_console.kazusa_client import project_cognition_graph_snapshot

    graph = project_cognition_graph_snapshot(
        source="overview_latest",
        payload={},
    )

    assert graph.status == "not_reported"
    assert graph.nodes == []
    assert graph.edges == []
    assert graph.redaction["reason"] == "brain response did not report cognition graph telemetry"


def test_bootstrap_and_debug_unavailable_return_graph_contract(tmp_path) -> None:
    """Overview and debug routes should expose the reusable graph contract."""

    from fastapi.testclient import TestClient

    from control_console.app import create_app
    from control_console.auth import hash_operator_token
    from control_console.contracts import ServiceRuntimeState
    from control_console.settings import ControlConsoleSettings

    class StoppedSupervisor:
        def all_service_states(self):
            return [
                ServiceRuntimeState(
                    id="brain",
                    display_name="Brain service",
                    kind="backend",
                    actual_state="stopped",
                ),
            ]

        def service_state(self, service_id: str):
            assert service_id == "brain"
            return self.all_service_states()[0]

    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(create_app(settings=settings, supervisor=StoppedSupervisor()))
    login = client.post("/api/auth/login", json={"token": "secret"}).json()
    headers = {login["csrf_header_name"]: login["csrf_token"]}

    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 200
    bootstrap_payload = bootstrap.json()
    assert bootstrap_payload["latest_cognition_graph"]["status"] == "not_reported"
    assert bootstrap_payload["overview"]["latest_cognition_graph"]["nodes"] == []

    debug = client.post(
        "/api/debug-chat",
        headers=headers,
        json={
            "channel_id": "debug",
            "user_id": "operator",
            "user_display_name": "Operator",
            "message_text": "hello",
        },
    )
    assert debug.status_code == 200
    debug_payload = debug.json()
    assert debug_payload["brain_available"] is False
    assert debug_payload["cognition_graph"]["status"] == "not_reported"
