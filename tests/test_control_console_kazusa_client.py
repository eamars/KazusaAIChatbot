"""HTTP client tests for brain-service calls from the control console."""

from __future__ import annotations

import json

import httpx
import pytest


@pytest.mark.asyncio
async def test_kazusa_client_reads_health_and_posts_debug_chat() -> None:
    """The console client should call bounded brain endpoints only."""

    from control_console.contracts import ConsoleDebugChatRequest
    from control_console.kazusa_client import KazusaClient
    from kazusa_ai_chatbot.time_boundary import parse_configured_local_datetime

    requests: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"})
        if request.url.path == "/ops/latest-cognition-graph":
            return httpx.Response(
                200,
                json={
                    "cognition_graph": {
                        "run_id": "turn-1",
                        "status": "completed",
                        "nodes": [
                            {
                                "id": "semantic.unexpected",
                                "label": "Semantic meaning",
                                "stage": "Future semantic",
                                "lane": "cognition",
                                "column": 2,
                                "category": "future_meaning",
                                "status": "completed",
                                "detail": {
                                    "schema_version": "private.schema",
                                    "evidence_refs": ["e1"],
                                    "root_refs": ["event-1"],
                                },
                            },
                        ],
                        "edges": [],
                    },
                },
            )
        if request.url.path == "/chat":
            assert request.headers["x-kazusa-control-console"] == "debug-v1"
            assert request.headers["x-kazusa-control-console-auth"] == (
                "shared-secret"
            )
            body = json.loads(request.read().decode("utf-8"))
            assert body["message_envelope"]["body_text"] == "hello"
            parse_configured_local_datetime(body["local_timestamp"])
            return httpx.Response(
                200,
                json={
                    "messages": ["hi"],
                    "content_type": "text",
                    "attachments": [{"url": "internal://asset"}],
                    "delivery_mentions": [
                        {
                            "global_user_id": "global-user-secret",
                            "platform_user_id": "platform-user-secret",
                            "display_name": "Operator",
                        },
                    ],
                    "delivery_tracking_id": "tracking-1",
                    "trace_id": "trace-debug-1",
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = KazusaClient(
        base_url="http://brain.local",
        timeout_seconds=1.0,
        control_shared_secret="shared-secret",
        transport=transport,
    )

    health = await client.get_health()
    latest_graph = await client.get_latest_cognition_graph()
    chat = await client.send_debug_chat(
        ConsoleDebugChatRequest.model_validate({
            "channel_id": "debug",
            "user_id": "operator",
            "user_display_name": "Operator",
            "message_text": "hello",
        })
    )

    assert health == {"status": "healthy"}
    assert latest_graph.run_id == "turn-1"
    assert latest_graph.source == "overview_latest"
    assert latest_graph.nodes[0].id == "semantic.unexpected"
    assert latest_graph.nodes[0].category == "future_meaning"
    assert latest_graph.nodes[0].stage == "Future semantic"
    assert latest_graph.nodes[0].detail == {}
    assert chat["response"]["messages"] == ["hi"]
    assert chat["response"]["content_type"] == "text"
    assert chat["response"]["attachment_count"] == 1
    assert chat["response"]["delivery_mention_count"] == 1
    assert "delivery_mentions" not in chat["response"]
    assert "global-user-secret" not in repr(chat)
    assert "platform-user-secret" not in repr(chat)
    assert chat["tracking_id"] == "tracking-1"
    assert chat["trace_id"] == "trace-debug-1"
    assert requests == [
        ("GET", "/health"),
        ("GET", "/ops/latest-cognition-graph"),
        ("POST", "/chat"),
        ("GET", "/ops/latest-cognition-graph"),
    ]


def test_graph_projection_preserves_semantic_cognition_rows() -> None:
    """Project goal, appraisal, axis, and cause semantics at the API boundary."""

    from control_console.kazusa_client import project_cognition_graph_snapshot

    snapshot = project_cognition_graph_snapshot(
        source="debug_latest",
        payload={
            "cognition_graph": {
                "run_id": "turn-semantic",
                "status": "completed",
                "nodes": [
                    {
                        "id": "future.meaning.v7",
                        "label": "Semantic meaning",
                        "stage": "Future inference",
                        "lane": "semantic-review",
                        "column": 7,
                        "category": "meaning_appraisal",
                        "status": "completed",
                        "detail": {
                            "appraisals": [{
                                "family": "event_agency",
                                "applicable": True,
                                "semantic_summary": "A new observation matters.",
                                "cause_summary": "The current message introduced it.",
                                "axis_changes": [{
                                    "axis": "novelty",
                                    "shift": "moderate_increase",
                                    "reason": "The observation is unfamiliar.",
                                }],
                            }],
                        },
                    },
                    {
                        "id": "future.goal.v7",
                        "label": "Active character goal",
                        "stage": "Goal selection",
                        "lane": "semantic-review",
                        "column": 12,
                        "category": "active_character_goal",
                        "status": "completed",
                        "detail": {
                            "goal": {
                                "goal_kind": "clarify",
                                "intent": "Clarify the observation.",
                                "reason": "The evidence is incomplete.",
                                "cause_summary": "The current message is ambiguous.",
                            },
                        },
                    },
                    {
                        "id": "future.plan.v7",
                        "label": "Response plan",
                        "stage": "Response planning",
                        "lane": "semantic-review",
                        "column": 18,
                        "category": "response_plan",
                        "status": "completed",
                        "detail": {
                            "response_goal": "Ask a focused question.",
                            "goal_resolution": "answerable_now",
                        },
                    },
                    {
                        "id": "future.affect.v7",
                        "label": "Affect and causes",
                        "stage": "Affect review",
                        "lane": "semantic-review",
                        "column": 21,
                        "category": "affect_causes",
                        "status": "completed",
                        "detail": {
                            "cause_provenance": [{
                                "family": "event_agency",
                                "cause_summary": "The observation remains concrete.",
                                "cause_status": "active",
                            }],
                        },
                    },
                ],
                "edges": [
                    {
                        "source": "future.meaning.v7",
                        "target": "future.goal.v7",
                        "kind": "sequence",
                        "label": "meaning informs goal",
                    },
                    {
                        "source": "future.goal.v7",
                        "target": "future.plan.v7",
                        "kind": "reference",
                        "label": "goal informs response",
                    },
                ],
            },
        },
    )

    node_by_category = {node.category: node for node in snapshot.nodes}
    assert node_by_category["meaning_appraisal"].id == "future.meaning.v7"
    assert node_by_category["meaning_appraisal"].stage == "Future inference"
    assert node_by_category["meaning_appraisal"].lane == "semantic-review"
    assert node_by_category["meaning_appraisal"].column == 7
    appraisal = node_by_category["meaning_appraisal"].detail["appraisals"][0]
    assert appraisal["family"] == "event_agency"
    assert appraisal["axis_changes"][0]["shift"] == "moderate_increase"
    assert node_by_category["active_character_goal"].detail["goal"]["intent"] == (
        "Clarify the observation."
    )
    assert node_by_category["affect_causes"].detail["cause_provenance"][0][
        "cause_status"
    ] == "active"
    assert {
        node.id for node in snapshot.nodes
    } == {
        "future.meaning.v7",
        "future.goal.v7",
        "future.plan.v7",
        "future.affect.v7",
    }
    assert [edge.label for edge in snapshot.edges] == [
        "meaning informs goal",
        "goal informs response",
    ]
    assert all(edge.kind in {"sequence", "reference"} for edge in snapshot.edges)
