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
                                "id": "l2.reasoning",
                                "label": "Reasoning",
                                "stage": "L2",
                                "lane": "cognition",
                                "column": 2,
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
    assert latest_graph.nodes[0].id == "l2.reasoning"
    assert chat["response"]["messages"] == ["hi"]
    assert chat["response"]["content_type"] == "text"
    assert chat["response"]["attachment_count"] == 1
    assert chat["response"]["delivery_mention_count"] == 1
    assert chat["cognition_chain_run"]["status"] == "not_reported"
    assert "delivery_mentions" not in chat["response"]
    assert "global-user-secret" not in repr(chat)
    assert "platform-user-secret" not in repr(chat)
    assert chat["tracking_id"] == "tracking-1"
    assert chat["trace_id"] == "trace-debug-1"
    assert requests == [
        ("GET", "/health"),
        ("GET", "/ops/latest-cognition-graph"),
        ("POST", "/chat"),
    ]


@pytest.mark.asyncio
async def test_kazusa_client_projects_correlated_live_and_self_chain_runs() -> None:
    """The client projects each graph's exact paired chain-run separately."""

    from control_console.kazusa_client import KazusaClient

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ops/latest-cognition-graph":
            return httpx.Response(
                200,
                json={
                    "cognition_graph": {
                        "run_id": "live-run",
                        "llm_trace_id": "live-trace",
                        "status": "completed",
                        "nodes": [],
                        "edges": [],
                    },
                    "self_cognition_graph": {
                        "run_id": "self-run",
                        "llm_trace_id": "self-trace",
                        "status": "completed",
                        "nodes": [],
                        "edges": [],
                    },
                    "cognition_chain_run": {
                        "chain_run_id": "cogchain-live",
                        "run_id": "live-run",
                        "llm_trace_id": "live-trace",
                        "terminal_disposition": "complete",
                    },
                    "self_cognition_chain_run": {
                        "chain_run_id": "cogchain-self",
                        "run_id": "self-run",
                        "llm_trace_id": "self-trace",
                        "terminal_disposition": "complete",
                    },
                },
            )
        return httpx.Response(404)

    client = KazusaClient(
        base_url="http://brain.local",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )
    live_graph, live_chain, self_graph, self_chain = (
        await client.get_latest_cognition_graph_with_chain_runs()
    )

    assert live_graph.run_id == "live-run"
    assert self_graph.run_id == "self-run"
    assert live_chain.run_id == "live-run"
    assert live_chain.chain_run_id == "cogchain-live"
    assert self_chain.run_id == "self-run"
    assert self_chain.chain_run_id == "cogchain-self"


@pytest.mark.asyncio
async def test_kazusa_client_rejects_cross_run_chain_correlation() -> None:
    """Mismatched chain rows remain explicitly absent instead of stale."""

    from control_console.kazusa_client import KazusaClient

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ops/latest-cognition-graph"
        return httpx.Response(
            200,
            json={
                "cognition_graph": {
                    "run_id": "live-run",
                    "llm_trace_id": "live-trace",
                    "status": "completed",
                    "nodes": [],
                    "edges": [],
                },
                "self_cognition_graph": {
                    "run_id": "self-run",
                    "llm_trace_id": "self-trace",
                    "status": "completed",
                    "nodes": [],
                    "edges": [],
                },
                "cognition_chain_run": {
                    "chain_run_id": "wrong-chain",
                    "run_id": "other-run",
                    "llm_trace_id": "other-trace",
                    "terminal_disposition": "complete",
                },
                "self_cognition_chain_run": {
                    "chain_run_id": "wrong-self-chain",
                    "run_id": "self-run",
                    "llm_trace_id": "other-trace",
                    "terminal_disposition": "complete",
                },
            },
        )

    client = KazusaClient(
        base_url="http://brain.local",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )
    _, live_chain, _, self_chain = (
        await client.get_latest_cognition_graph_with_chain_runs()
    )

    assert live_chain.status == "not_reported"
    assert self_chain.status == "not_reported"
