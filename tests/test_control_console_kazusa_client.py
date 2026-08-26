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
    from tests.test_control_console_contracts import _canonical_live_observation

    requests: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"})
        if request.url.path == "/ops/latest-cognition-graph":
            return httpx.Response(
                200,
                json={
                    "cognition_graph": _canonical_live_observation(
                        run_id="turn-1",
                        invocation_id="invocation-1",
                    ),
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
                    "cognition_graph": _canonical_live_observation(
                        run_id="debug-turn-1",
                        invocation_id="debug-invocation-1",
                    ),
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
    assert latest_graph.correlation.run_id == "turn-1"
    assert latest_graph.schema_version == "cognition_run_observation.v1"
    assert latest_graph.nodes[0].node_id == "input.turn"
    assert latest_graph.nodes[0].section_refs == ["input.turn"]
    assert chat.response_payload["messages"] == ["hi"]
    assert chat.response_payload["content_type"] == "text"
    assert chat.response_payload["attachment_count"] == 1
    assert chat.response_payload["delivery_mention_count"] == 1
    assert "delivery_mentions" not in chat.response_payload
    assert "global-user-secret" not in repr(chat)
    assert "platform-user-secret" not in repr(chat)
    assert chat.response_payload["delivery_tracking_id"] == "tracking-1"
    assert chat.response_payload["trace_id"] == "trace-debug-1"
    assert chat.cognition_observation is not None
    assert chat.cognition_observation.correlation.run_id == "debug-turn-1"
    assert requests == [
        ("GET", "/health"),
        ("GET", "/ops/latest-cognition-graph"),
        ("POST", "/chat"),
    ]


async def test_client_validates_canonical_cognition_observation_without_reprojection() -> None:
    """The client should validate Brain's observation without semantic projection."""

    from control_console.kazusa_client import KazusaClient
    from kazusa_ai_chatbot.brain_service.cognition_observation_contracts import (
        CognitionRunObservationV1,
    )
    from tests.test_control_console_contracts import _canonical_live_observation

    payload = _canonical_live_observation(run_id="direct-run")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ops/latest-cognition-graph"
        return httpx.Response(200, json={"cognition_graph": payload})

    client = KazusaClient(
        base_url="http://brain.local",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )

    observation = await client.get_latest_cognition_graph()

    assert isinstance(observation, CognitionRunObservationV1)
    assert observation.correlation.run_id == "direct-run"
    assert observation.generated_at.isoformat() == "2026-08-26T00:00:00+00:00"


async def test_client_raises_protocol_error_for_invalid_observation_version() -> None:
    """Invalid observation versions should expose only the safe protocol code."""

    from control_console.kazusa_client import (
        CognitionObservationProtocolError,
        KazusaClient,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ops/latest-cognition-graph"
        return httpx.Response(
            200,
            json={"cognition_graph": {"schema_version": "wrong.v9"}},
        )

    client = KazusaClient(
        base_url="http://brain.local",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(CognitionObservationProtocolError) as error:
        await client.get_latest_cognition_graph()

    assert str(error.value) == "observation_contract_invalid"
    assert "wrong.v9" not in repr(error.value)


async def test_client_rejects_invalid_latest_observation_without_reconstruction() -> None:
    """Missing and malformed Brain fields must fail closed without inference."""

    from control_console.kazusa_client import (
        CognitionObservationProtocolError,
        KazusaClient,
    )

    payloads = [
        {"cognition_graph": None},
        {},
        {
            "cognition_graph": {
                "messages": ["this must never become a cognition section"],
            },
        },
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ops/latest-cognition-graph"
        return httpx.Response(200, json=payloads.pop(0))

    client = KazusaClient(
        base_url="http://brain.local",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )

    assert await client.get_latest_cognition_graph() is None

    with pytest.raises(CognitionObservationProtocolError) as missing_error:
        await client.get_latest_cognition_graph()

    assert str(missing_error.value) == "observation_contract_invalid"

    with pytest.raises(CognitionObservationProtocolError) as error:
        await client.get_latest_cognition_graph()

    assert str(error.value) == "observation_contract_invalid"
    assert "this must never become" not in repr(error.value)


async def test_debug_client_returns_direct_response_observation_without_latest_fetch() -> None:
    """Debug chat should return its response observation without a second latest read."""

    from control_console.contracts import ConsoleDebugChatRequest
    from control_console.kazusa_client import KazusaClient, KazusaDebugChatResult
    from tests.test_control_console_contracts import _canonical_live_observation

    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path != "/chat":
            raise AssertionError(f"unexpected Brain request: {request.url.path}")
        return httpx.Response(
            200,
            json={
                "messages": [{"text": "reply"}],
                "delivery_tracking_id": "tracking-direct",
                "trace_id": "trace-direct",
                "cognition_graph": _canonical_live_observation(
                    run_id="debug-direct-run",
                    invocation_id="debug-direct-invocation",
                ),
            },
        )

    client = KazusaClient(
        base_url="http://brain.local",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )
    result = await client.send_debug_chat(ConsoleDebugChatRequest(
        channel_id="debug",
        user_id="operator",
        user_display_name="Operator",
        message_text="hello",
    ))

    assert isinstance(result, KazusaDebugChatResult)
    assert result.response_payload["messages"] == [{"text": "reply"}]
    assert result.cognition_observation is not None
    assert result.cognition_observation.correlation.run_id == "debug-direct-run"
    assert "cognition_graph" not in result.response_payload
    assert requests == ["/chat"]
