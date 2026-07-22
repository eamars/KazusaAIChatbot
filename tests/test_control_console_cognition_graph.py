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
            "trigger_source": "internal_thought",
            "input_sources": ["internal_monologue"],
            "nodes": [
                {
                    "id": "intake",
                    "label": "Intake",
                    "stage": "L1",
                    "lane": "input",
                    "column": 1,
                    "status": "completed",
                    "detail": {
                        "input": "Normalized debug input.",
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
                        "retrieval_answer": "No memory write requested.",
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
                    "detail": {"messages": ["Returned visible answer."]},
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
    assert graph.trigger_source == "internal_thought"
    assert graph.input_sources == ["internal_monologue"]
    assert {node.branch for node in graph.nodes} >= {"l2a", "l2b"}
    assert {edge.kind for edge in graph.edges} == {"fork", "join"}
    graph_text = repr(graph.model_dump(mode="json"))
    detail_text = repr([node.detail for node in graph.nodes])
    assert "User wants bounded inspection." not in graph_text
    assert "Normalized debug input." in graph_text
    assert "Returned visible answer." in graph_text
    assert "must not leak" not in detail_text
    assert "embedding" not in detail_text
    assert "prompt" not in detail_text


def test_cognition_graph_semantic_projection_preserves_full_approved_values() -> None:
    """Approved semantic values keep order and full text past generic caps."""

    from control_console.kazusa_client import project_cognition_graph_snapshot

    long_text = "start\n" + ("evidence-" * 140) + "end <script>alert(1)</script>"
    evidence_rows = [
        {
            "fact": f"fact-{index}",
            "excerpt": long_text if index == 0 else "duplicate",
            "title": "source title",
            "prompt": "nested prompt must be excluded",
            "embedding": [index],
        }
        for index in range(55)
    ]
    payload = {
        "cognition_graph": {
            "run_id": "semantic-run",
            "status": "completed",
            "nodes": [
                {
                    "id": "intake",
                    "label": "Queued turn",
                    "stage": "L1",
                    "lane": "input",
                    "column": 1,
                    "status": "completed",
                    "detail": {
                        "input": long_text,
                        "reply_context": {
                            "reply_excerpt": "reply context",
                            "raw_message": "nested raw message must be excluded",
                        },
                    },
                },
                {
                    "id": "l2.memory",
                    "label": "Memory and evidence",
                    "stage": "L2",
                    "lane": "memory",
                    "column": 2,
                    "status": "completed",
                    "detail": {
                        "retrieval_answer": long_text,
                        "memory_evidence": evidence_rows,
                        "supervisor_trace": ["trace must be excluded"],
                    },
                },
                {
                    "id": "l3.surface",
                    "label": "Visible surface",
                    "stage": "L3",
                    "lane": "surface",
                    "column": 3,
                    "status": "completed",
                    "detail": {"messages": [long_text, "duplicate"]},
                },
            ],
            "edges": [],
        },
    }

    graph = project_cognition_graph_snapshot(
        source="debug_latest",
        payload=payload,
    )

    intake_detail = graph.nodes[0].detail
    memory_detail = graph.nodes[1].detail
    assert intake_detail["input"] == long_text
    assert intake_detail["reply_context"] == {"reply_excerpt": "reply context"}
    assert memory_detail["retrieval_answer"] == long_text
    assert len(memory_detail["memory_evidence"]) == 55
    assert memory_detail["memory_evidence"][0]["excerpt"] == long_text
    assert memory_detail["memory_evidence"][54]["fact"] == "fact-54"
    assert "prompt" not in repr(memory_detail)
    assert "embedding" not in repr(memory_detail)
    assert "supervisor_trace" not in memory_detail
    assert graph.nodes[2].detail["messages"] == [long_text, "duplicate"]


def test_cognition_graph_projection_handles_malformed_detail_without_throwing() -> None:
    """Malformed semantic detail should fail closed field by field."""

    from control_console.kazusa_client import project_cognition_graph_snapshot

    graph = project_cognition_graph_snapshot(
        source="debug_latest",
        payload={
            "cognition_graph": {
                "status": "completed",
                "nodes": [
                    {
                        "id": "l2.memory",
                        "label": "Memory",
                        "stage": "L2",
                        "lane": "memory",
                        "column": 1,
                        "status": "completed",
                        "detail": {
                            "retrieval_answer": 42,
                            "memory_evidence": "not a list",
                            "conversation_progress": {"goal": "preserve"},
                        },
                    },
                ],
                "edges": [],
            },
        },
    )

    assert graph.status == "partial"
    assert graph.trigger_source == "not_reported"
    assert graph.redaction["reason"] == "trigger_source_missing"
    detail = graph.nodes[0].detail
    assert "retrieval_answer" not in detail
    assert "memory_evidence" not in detail
    assert detail["conversation_progress"] == {"goal": "preserve"}


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


def test_cognition_graph_projection_handles_invalid_and_inferred_payloads() -> None:
    """Graph projection should fail closed and infer safe nodes when possible."""

    from control_console.kazusa_client import project_cognition_graph_snapshot

    invalid = project_cognition_graph_snapshot(
        source="overview_latest",
        payload={
            "delivery_tracking_id": "x" * 200,
            "cognition_graph": {
                "status": "invalid",
                "run_id": "bad-run",
                "nodes": [
                    "not a node",
                    {
                        "id": "INVALID ID",
                        "label": "",
                        "stage": "",
                        "lane": "",
                        "column": 0,
                        "detail": "bad detail",
                    },
                ],
                "edges": [
                    "not an edge",
                    {
                        "source": "INVALID",
                        "target": "",
                        "kind": "bad",
                    },
                ],
            },
        },
    )
    inferred = project_cognition_graph_snapshot(
        source="debug_latest",
        payload={
            "delivery_tracking_id": "debug-run",
            "internal_monologue": "bounded reasoning",
            "logical_stance": "grounded",
            "decision": "reply",
            "messages": [{"text": "visible"}],
        },
    )

    assert invalid.status == "partial"
    assert invalid.run_id == "bad-run"
    assert invalid.redaction["reason"] == "trigger_source_missing"
    assert inferred.status == "partial"
    assert [node.id for node in inferred.nodes] == [
        "l2.reasoning",
        "l2.decision",
        "l3.surface",
    ]
    assert [edge.kind for edge in inferred.edges] == ["sequence", "sequence"]


def test_cognition_graph_projection_fails_closed_for_unknown_source_metadata() -> None:
    """The console keeps malformed source metadata bounded and explicit."""

    from control_console.kazusa_client import project_cognition_graph_snapshot

    graph = project_cognition_graph_snapshot(
        source="overview_latest",
        payload={
            "cognition_graph": {
                "run_id": "source-check",
                "status": "completed",
                "trigger_source": "<script>unknown</script>" * 40,
                "input_sources": ["dialog_text", {"bad": "source"}],
                "nodes": [],
                "edges": [],
            },
        },
    )

    assert graph.trigger_source == "not_reported"
    assert graph.input_sources == ["dialog_text"]
    assert graph.status == "partial"
    assert "<script>" not in repr(graph.model_dump(mode="json"))


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
