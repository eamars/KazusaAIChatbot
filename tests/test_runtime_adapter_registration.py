"""Unit tests for cross-process runtime adapter registration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.discord_adapter import DiscordAdapter
from adapters.napcat_qq_adapter import NapCatWSAdapter
from kazusa_ai_chatbot.dispatcher import AdapterRegistry, RemoteHttpAdapter
from kazusa_ai_chatbot import service as service_module


class _DummyResponse:
    """Tiny HTTP response double for adapter-registration tests."""

    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        """Pretend the HTTP request succeeded."""

    def json(self) -> dict:
        """Return the stored JSON payload."""

        return self._payload


class _FakeAsyncClient:
    """Capture one outbound HTTP request from ``RemoteHttpAdapter``."""

    last_call: dict | None = None

    def __init__(self, *, timeout: float):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb

    async def post(self, url: str, *, json: dict, headers: dict):
        type(self).last_call = {
            "url": url,
            "json": json,
            "headers": headers,
            "timeout": self.timeout,
        }
        return _DummyResponse({
            "platform": "qq",
            "channel_id": json["channel_id"],
            "message_id": "outbound-1",
            "sent_at": datetime(2026, 4, 25, 6, 0, tzinfo=timezone.utc).isoformat(),
        })


def test_register_remote_runtime_adapter_registers_proxy_in_service(monkeypatch):
    """The brain service should store a remote adapter proxy by platform."""

    registry = AdapterRegistry()
    monkeypatch.setattr(service_module, "_adapter_registry", registry)

    service_module.register_remote_runtime_adapter(
        platform="qq",
        callback_url="http://127.0.0.1:8011",
        shared_secret="secret-token",
        timeout_seconds=7.5,
    )

    assert registry.has("qq")
    adapter = registry.get("qq")
    assert isinstance(adapter, RemoteHttpAdapter)
    assert adapter.platform == "qq"


def test_register_runtime_adapter_payload_reuses_remote_registration(monkeypatch):
    """Heartbeat registration should use the same proxy-registration path as startup."""

    payload = service_module.RuntimeAdapterRegistrationRequest(
        platform="qq",
        callback_url="http://127.0.0.1:8011",
        shared_secret="secret-token",
        timeout_seconds=9.0,
    )
    register_remote = MagicMock()
    monkeypatch.setattr(service_module, "register_remote_runtime_adapter", register_remote)

    response = service_module._register_runtime_adapter_payload(payload, status="heartbeat_ok")

    register_remote.assert_called_once_with(
        platform="qq",
        callback_url="http://127.0.0.1:8011",
        shared_secret="secret-token",
        timeout_seconds=9.0,
    )
    assert response.status == "heartbeat_ok"
    assert response.platform == "qq"


@pytest.mark.asyncio
async def test_remote_http_adapter_posts_send_message_payload(monkeypatch):
    """The remote proxy should call the adapter callback endpoint with bearer auth."""

    monkeypatch.setattr("kazusa_ai_chatbot.dispatcher.remote_adapter.httpx.AsyncClient", _FakeAsyncClient)
    adapter = RemoteHttpAdapter(
        platform="qq",
        callback_url="http://127.0.0.1:8011",
        shared_secret="secret-token",
        timeout_seconds=7.5,
    )

    result = await adapter.send_message(
        channel_id="54369546",
        text="今天天气真好呀",
        reply_to_msg_id="1615877136",
    )

    assert _FakeAsyncClient.last_call == {
        "url": "http://127.0.0.1:8011/send_message",
        "json": {
            "channel_id": "54369546",
            "text": "今天天气真好呀",
            "reply_to_msg_id": "1615877136",
        },
        "headers": {"Authorization": "Bearer secret-token"},
        "timeout": 7.5,
    }
    assert result.platform == "qq"
    assert result.channel_id == "54369546"
    assert result.message_id == "outbound-1"


class _FakeNapCatWebSocket:
    """Small websocket double for NapCat API response tests."""

    def __init__(self, message_data: dict):
        self._message_data = message_data
        self.sent_payloads: list[dict] = []

    async def send(self, payload: str) -> None:
        """Capture the sent websocket frame."""

        self.sent_payloads.append(json.loads(payload))

    async def recv(self) -> str:
        """Return a get_msg response matching the most recent echo id."""

        echo_id = self.sent_payloads[-1]["echo"]
        return json.dumps({
            "echo": echo_id,
            "status": "ok",
            "data": self._message_data,
        })


@pytest.mark.asyncio
async def test_napcat_hydrates_reply_target_from_platform_get_msg():
    """QQ reply ids should resolve to hard reply-to-bot metadata in the adapter."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["54369546"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    ws = _FakeNapCatWebSocket({
        "message_id": 1733223276,
        "user_id": 3768713357,
        "sender": {"nickname": "杏山千纱"},
        "raw_message": "上一条千纱消息",
    })
    reply_context = {"reply_to_message_id": "1733223276"}

    await adapter._hydrate_reply_context_from_platform(reply_context, ws)
    adapter._finalize_reply_target(reply_context)

    assert reply_context == {
        "reply_to_message_id": "1733223276",
        "reply_to_platform_user_id": "3768713357",
        "reply_to_display_name": "杏山千纱",
        "reply_excerpt": "上一条千纱消息",
        "reply_to_current_bot": True,
    }
    assert ws.sent_payloads[0]["action"] == "get_msg"
    assert ws.sent_payloads[0]["params"] == {"message_id": 1733223276}
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_handle_event_forwards_reply_to_current_bot_metadata():
    """Inbound QQ replies to the bot should reach the brain with reply_to_current_bot=true."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["1082431481"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    adapter.bot_name = "杏山千纱"
    adapter.brain_client.post = AsyncMock(return_value=_DummyResponse({
        "messages": [],
        "should_reply": False,
    }))
    ws = _FakeNapCatWebSocket({
        "message_id": 1733223276,
        "user_id": 3768713357,
        "sender": {"nickname": "杏山千纱"},
        "raw_message": "上一条千纱消息",
    })

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 394466266,
            "group_id": 1082431481,
            "user_id": 3167827653,
            "sender": {"nickname": "赛博马里奥"},
            "message": [
                {"type": "reply", "data": {"id": "1733223276"}},
                {"type": "text", "data": {"text": "千纱さん，你来了呀"}},
            ],
        },
        ws,
    )

    payload = adapter.brain_client.post.await_args.kwargs["json"]
    assert payload["reply_context"]["reply_to_current_bot"] is True
    assert payload["reply_context"]["reply_to_platform_user_id"] == "3768713357"
    assert payload["reply_context"]["reply_to_message_id"] == "1733223276"
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_register_with_brain_posts_runtime_callback(monkeypatch):
    """NapCat startup should register its callback URL with the brain service."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        runtime_shared_secret="secret-token",
        channel_ids=["54369546"],
        debug_modes={},
    )
    post = AsyncMock(return_value=_DummyResponse({"status": "registered"}))
    adapter.brain_client.post = post

    await adapter._register_with_brain()

    post.assert_awaited_once_with(
        "http://127.0.0.1:8000/runtime/adapters/register",
        json={
            "platform": "qq",
            "callback_url": "http://127.0.0.1:8011",
            "shared_secret": "secret-token",
            "timeout_seconds": 10.0,
        },
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_heartbeat_posts_runtime_callback(monkeypatch):
    """NapCat heartbeat should keep refreshing runtime registration."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        runtime_shared_secret="secret-token",
        heartbeat_seconds=30.0,
        channel_ids=["54369546"],
        debug_modes={},
    )
    post = AsyncMock(return_value=_DummyResponse({"status": "heartbeat_ok"}))
    adapter.brain_client.post = post

    await adapter._send_heartbeat_once()

    post.assert_awaited_once_with(
        "http://127.0.0.1:8000/runtime/adapters/heartbeat",
        json={
            "platform": "qq",
            "callback_url": "http://127.0.0.1:8011",
            "shared_secret": "secret-token",
            "timeout_seconds": 10.0,
        },
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_discord_register_with_brain_posts_runtime_callback():
    """Discord startup should register its callback URL with the brain service."""

    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        channel_ids=["12345"],
        debug_modes={},
    )
    post = AsyncMock(return_value=_DummyResponse({"status": "registered"}))
    adapter._http_client.post = post

    await adapter._register_with_brain()

    post.assert_awaited_once_with(
        "http://127.0.0.1:8000/runtime/adapters/register",
        json={
            "platform": "discord",
            "callback_url": "http://127.0.0.1:8012",
            "shared_secret": "secret-token",
            "timeout_seconds": 10.0,
        },
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_discord_heartbeat_posts_runtime_callback():
    """Discord heartbeat should keep refreshing runtime registration."""

    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        heartbeat_seconds=30.0,
        channel_ids=["12345"],
        debug_modes={},
    )
    post = AsyncMock(return_value=_DummyResponse({"status": "heartbeat_ok"}))
    adapter._http_client.post = post

    await adapter._send_heartbeat_once()

    post.assert_awaited_once_with(
        "http://127.0.0.1:8000/runtime/adapters/heartbeat",
        json={
            "platform": "discord",
            "callback_url": "http://127.0.0.1:8012",
            "shared_secret": "secret-token",
            "timeout_seconds": 10.0,
        },
    )
    await adapter.close()
