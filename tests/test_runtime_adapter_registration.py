"""Unit tests for cross-process runtime adapter registration."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import adapters.delivery_receipts as delivery_receipts_module
import adapters.discord_adapter as discord_module
import adapters.napcat_qq_adapter as napcat_module
from adapters.discord_adapter import DiscordAdapter
from adapters.napcat_qq_adapter import NapCatWSAdapter
from kazusa_ai_chatbot.dispatcher import AdapterRegistry, RemoteHttpAdapter
from kazusa_ai_chatbot import service as service_module


class _DummyResponse:
    """Tiny HTTP response double for adapter-registration tests."""

    def __init__(self, payload: dict, *, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    @property
    def is_error(self) -> bool:
        """Return whether the fake status code represents an HTTP error."""

        return_value = self.status_code >= 400
        return return_value

    def raise_for_status(self) -> None:
        """Pretend the HTTP request succeeded."""

    def json(self) -> dict:
        """Return the stored JSON payload."""

        return self._payload


class _RaisingResponse:
    """HTTP response double that raises when status is checked."""

    def raise_for_status(self) -> None:
        """Raise an HTTP status error."""

        request = httpx.Request("POST", "http://127.0.0.1:8000/delivery_receipt")
        response = httpx.Response(500, request=request)
        raise httpx.HTTPStatusError(
            "server error",
            request=request,
            response=response,
        )


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
        if url.endswith("/send_message/capability"):
            return_value = _DummyResponse({"available": True})
            return return_value
        return _DummyResponse({
            "platform": "qq",
            "channel_id": json["channel_id"],
            "message_id": "outbound-1",
            "sent_at": datetime(2026, 4, 25, 6, 0, tzinfo=timezone.utc).isoformat(),
        })


class _FakeDiscordAuthor:
    """Minimal Discord author double for adapter event tests."""

    def __init__(self) -> None:
        self.id = 123456789
        self.display_name = "User A"


class _FakeDiscordSentMessage:
    """Minimal sent-message double carrying a platform message id."""

    def __init__(self, message_id: str) -> None:
        self.id = message_id


class _FakeDiscordTyping:
    """Async context manager double for Discord typing indicators."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb


class _FakeDiscordChannel:
    """Minimal Discord channel double for adapter event tests."""

    def __init__(self) -> None:
        self.id = 12345
        self.name = "general"
        self.sent_chunks: list[str] = []
        self.sent_kwargs: list[dict] = []

    def typing(self) -> _FakeDiscordTyping:
        """Return an async context manager for the typing indicator."""

        return_value = _FakeDiscordTyping()
        return return_value

    async def send(self, content: str, **kwargs) -> _FakeDiscordSentMessage:
        """Record a channel send and return a sent-message id."""

        self.sent_chunks.append(content)
        self.sent_kwargs.append(dict(kwargs))
        message_id = f"discord-send-{len(self.sent_chunks)}"
        return_value = _FakeDiscordSentMessage(message_id)
        return return_value


class _FakeDiscordMessage:
    """Minimal inbound Discord message double for adapter event tests."""

    def __init__(self) -> None:
        self.id = 999
        self.author = _FakeDiscordAuthor()
        self.channel = _FakeDiscordChannel()
        self.guild = object()
        self.reference = None
        self.attachments = []
        self.content = "hello"
        self.reply_chunks: list[str] = []

    async def reply(self, content: str) -> _FakeDiscordSentMessage:
        """Record a reply send and return a sent-message id."""

        self.reply_chunks.append(content)
        message_id = f"discord-reply-{len(self.reply_chunks)}"
        return_value = _FakeDiscordSentMessage(message_id)
        return return_value


def _target_user_mention(
    *,
    platform_user_id: str | None = "2787858400",
) -> dict:
    return {
        "entity_kind": "user",
        "placement": "prefix",
        "platform_user_id": platform_user_id,
        "global_user_id": "global-user-1",
        "display_name": "Target User",
        "requested_by": "dialog.mention_target_user",
    }


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


def test_self_cognition_uses_registered_runtime_adapter() -> None:
    """Self-cognition worker startup should receive the service registry."""

    source_text = Path(service_module.__file__).read_text(encoding="utf-8")
    call_start = source_text.index(
        "_self_cognition_worker_handle = start_self_cognition_worker("
    )
    call_block = source_text[call_start: call_start + 500]

    assert "adapter_registry_provider" in call_block
    assert "_adapter_registry" in call_block


@pytest.mark.asyncio
async def test_remote_http_adapter_posts_send_message_payload(monkeypatch):
    """The remote proxy should call the adapter callback endpoint with bearer auth."""

    mention = _target_user_mention()
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
        channel_type="private",
        delivery_mentions=[mention],
    )

    assert _FakeAsyncClient.last_call == {
        "url": "http://127.0.0.1:8011/send_message",
        "json": {
            "channel_id": "54369546",
            "channel_type": "private",
            "text": "今天天气真好呀",
            "reply_to_msg_id": "1615877136",
            "delivery_mentions": [mention],
        },
        "headers": {"Authorization": "Bearer secret-token"},
        "timeout": 7.5,
    }
    assert result.platform == "qq"
    assert result.channel_id == "54369546"
    assert result.message_id == "outbound-1"


@pytest.mark.asyncio
async def test_remote_http_adapter_posts_send_message_capability_payload(
    monkeypatch,
) -> None:
    """The remote proxy should query adapter channel capability."""

    _FakeAsyncClient.last_call = None
    monkeypatch.setattr(
        "kazusa_ai_chatbot.dispatcher.remote_adapter.httpx.AsyncClient",
        _FakeAsyncClient,
    )
    adapter = RemoteHttpAdapter(
        platform="qq",
        callback_url="http://127.0.0.1:8011",
        shared_secret="secret-token",
        timeout_seconds=7.5,
    )

    available = await adapter.can_send_message(
        channel_id="54369546",
        channel_type="group",
    )

    assert available is True
    assert _FakeAsyncClient.last_call == {
        "url": "http://127.0.0.1:8011/send_message/capability",
        "json": {
            "channel_id": "54369546",
            "channel_type": "group",
        },
        "headers": {"Authorization": "Bearer secret-token"},
        "timeout": 7.5,
    }


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


class _FakeNapCatActionWebSocket:
    """Websocket double that returns responses by OneBot action name."""

    def __init__(self, responses_by_action: dict[str, dict]):
        self._responses_by_action = responses_by_action
        self.sent_payloads: list[dict] = []

    async def send(self, payload: str) -> None:
        """Capture the sent websocket frame."""

        self.sent_payloads.append(json.loads(payload))

    async def recv(self) -> str:
        """Return a response matching the last requested OneBot action."""

        request_payload = self.sent_payloads[-1]
        action = request_payload["action"]
        response_payload = dict(self._responses_by_action[action])
        response_payload["echo"] = request_payload["echo"]
        return_value = json.dumps(response_payload)
        return return_value


class _FailingSendNapCatWebSocket:
    """Websocket double whose send_msg API call fails."""

    def __init__(self) -> None:
        """Create a failing send websocket fake."""

        self.sent_payloads: list[dict] = []

    async def send(self, payload: str) -> None:
        """Capture the sent websocket frame."""

        self.sent_payloads.append(json.loads(payload))

    async def recv(self) -> str:
        """Return a failed send_msg response."""

        echo_id = self.sent_payloads[-1]["echo"]
        return json.dumps({
            "echo": echo_id,
            "status": "failed",
            "retcode": 1200,
            "message": "send failed",
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

    assert reply_context == {
        "reply_to_message_id": "1733223276",
        "reply_to_platform_user_id": "3768713357",
        "reply_to_display_name": "杏山千纱",
        "reply_excerpt": "上一条千纱消息",
    }
    assert ws.sent_payloads[0]["action"] == "get_msg"
    assert ws.sent_payloads[0]["params"] == {"message_id": 1733223276}
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_handle_event_forwards_typed_bot_reply_metadata():
    """Inbound QQ replies to the bot should reach the brain as typed addressees."""

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
        "use_reply_feature": False,
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
    assert payload["message_envelope"]["body_text"] == "千纱さん，你来了呀"
    assert payload["message_envelope"]["raw_wire_text"].startswith(
        "[CQ:reply,id=1733223276]"
    )
    assert "content" not in payload
    assert "reply_context" not in payload
    assert "attachments" not in payload
    assert payload["message_envelope"]["reply"]["platform_user_id"] == "3768713357"
    assert payload["message_envelope"]["reply"]["platform_message_id"] == "1733223276"
    assert payload["message_envelope"]["reply"]["global_user_id"]
    assert payload["message_envelope"]["addressed_to_global_user_ids"]
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_handle_event_sends_readable_bot_mention_and_typed_envelope():
    """QQ adapter should replace bot CQ mentions before calling the brain."""

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
    adapter.bot_name = "Kazusa"
    adapter.brain_client.post = AsyncMock(return_value=_DummyResponse({
        "messages": [],
        "use_reply_feature": False,
    }))
    ws = _FakeNapCatWebSocket({
        "message_id": 1733223276,
        "user_id": 3768713357,
        "sender": {"nickname": "Kazusa"},
        "raw_message": "previous bot message",
    })

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 394466267,
            "group_id": 1082431481,
            "user_id": 3167827653,
            "sender": {"nickname": "User A"},
            "message": (
                "[CQ:reply,id=1733223276]"
                "[CQ:at,qq=3768713357] what does this mean?"
                "[CQ:face,id=1]"
            ),
        },
        ws,
    )

    payload = adapter.brain_client.post.await_args.kwargs["json"]
    assert "content" not in payload
    assert all(key != "mentioned" + "_bot" for key in payload)
    assert payload["message_envelope"]["body_text"] == "@Kazusa what does this mean?"
    assert payload["message_envelope"]["raw_wire_text"].startswith(
        "[CQ:reply,id=1733223276]"
    )
    assert payload["message_envelope"]["mentions"][0]["entity_kind"] == "bot"
    assert payload["message_envelope"]["mentions"][0]["display_name"] == "Kazusa"
    assert payload["message_envelope"]["addressed_to_global_user_ids"]
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_handle_event_uses_segment_nickname_without_lookup():
    """QQ segment mention labels should be used before platform lookup."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["905393941"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    adapter.bot_name = "Kazusa"
    adapter.brain_client.post = AsyncMock(return_value=_DummyResponse({
        "messages": [],
        "use_reply_feature": False,
    }))
    ws = _FakeNapCatActionWebSocket({})

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 394466271,
            "group_id": 905393941,
            "user_id": 3167827653,
            "sender": {"nickname": "User A"},
            "message": [
                {
                    "type": "at",
                    "data": {
                        "qq": "673225019",
                        "nickname": "Segment Nick",
                        "card": "Group Card",
                    },
                },
                {"type": "text", "data": {"text": " hello"}},
            ],
        },
        ws,
    )

    payload = adapter.brain_client.post.await_args.kwargs["json"]
    envelope = payload["message_envelope"]
    assert envelope["body_text"] == "@Segment Nick hello"
    assert envelope["mentions"][0]["display_name"] == "Segment Nick"
    assert ws.sent_payloads == []
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_handle_event_hydrates_human_mention_nickname_and_cache():
    """QQ human mention labels should use nickname before card and cache it."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["905393941"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    adapter.bot_name = "Kazusa"
    adapter.brain_client.post = AsyncMock(return_value=_DummyResponse({
        "messages": [],
        "use_reply_feature": False,
    }))
    ws = _FakeNapCatActionWebSocket({
        "get_group_member_info": {
            "status": "ok",
            "data": {
                "nickname": "Mention Nick",
                "card": "Group Card",
            },
        },
    })

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 394466268,
            "group_id": 905393941,
            "user_id": 3167827653,
            "sender": {"nickname": "User A"},
            "message": "[CQ:at,qq=673225019] 你怎么评价群友",
        },
        ws,
    )
    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 394466269,
            "group_id": 905393941,
            "user_id": 3167827653,
            "sender": {"nickname": "User A"},
            "message": "[CQ:at,qq=673225019] again",
        },
        ws,
    )

    payloads = [
        call.kwargs["json"]
        for call in adapter.brain_client.post.await_args_list
    ]
    first_envelope = payloads[0]["message_envelope"]
    second_envelope = payloads[1]["message_envelope"]
    assert first_envelope["body_text"] == "@Mention Nick 你怎么评价群友"
    assert first_envelope["mentions"][0]["display_name"] == "Mention Nick"
    assert second_envelope["body_text"] == "@Mention Nick again"
    lookup_payloads = [
        payload
        for payload in ws.sent_payloads
        if payload["action"] == "get_group_member_info"
    ]
    assert len(lookup_payloads) == 1
    assert lookup_payloads[0]["params"] == {
        "group_id": 905393941,
        "user_id": 673225019,
        "no_cache": False,
    }
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_mention_display_cache_evicts_oldest_label():
    """QQ mention display cache should stay bounded in long-running adapters."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["905393941"],
        debug_modes={},
    )

    for index in range(napcat_module._MENTION_DISPLAY_CACHE_LIMIT + 1):
        adapter._cache_qq_mention_display_name(
            ("905393941", str(index)),
            f"User {index}",
        )

    assert len(adapter._mention_display_cache) == (
        napcat_module._MENTION_DISPLAY_CACHE_LIMIT
    )
    assert ("905393941", "0") not in adapter._mention_display_cache
    assert ("905393941", str(napcat_module._MENTION_DISPLAY_CACHE_LIMIT)) in (
        adapter._mention_display_cache
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_handle_event_uses_occurrence_label_when_lookup_fails():
    """QQ mention lookup misses should not expose QQ ids in body text."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["905393941"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    adapter.bot_name = "Kazusa"
    adapter.brain_client.post = AsyncMock(return_value=_DummyResponse({
        "messages": [],
        "use_reply_feature": False,
    }))
    ws = _FakeNapCatActionWebSocket({
        "get_group_member_info": {
            "status": "failed",
            "message": "not found",
        },
    })

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 394466270,
            "group_id": 905393941,
            "user_id": 3167827653,
            "sender": {"nickname": "User A"},
            "message": "[CQ:at,qq=673225019] hello",
        },
        ws,
    )

    payload = adapter.brain_client.post.await_args.kwargs["json"]
    envelope = payload["message_envelope"]
    assert envelope["body_text"] == "@mentioned-user-1 hello"
    assert "673225019" not in envelope["body_text"]
    assert envelope["mentions"][0]["display_name"] == ""
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_handle_event_bounds_duplicate_timeout_lookups(
    monkeypatch,
):
    """QQ lookup timeouts should fallback once per mentioned id per message."""

    observed_timeouts: list[float] = []

    async def fake_wait_for(awaitable, timeout: float):
        """Simulate NapCat lookup timeout without waiting one real second."""

        observed_timeouts.append(timeout)
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise asyncio.TimeoutError("mention lookup timeout")

    monkeypatch.setattr(napcat_module.asyncio, "wait_for", fake_wait_for)
    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["905393941"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    adapter.bot_name = "Kazusa"
    adapter.brain_client.post = AsyncMock(return_value=_DummyResponse({
        "messages": [],
        "use_reply_feature": False,
    }))
    ws = _FakeNapCatActionWebSocket({})

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 394466272,
            "group_id": 905393941,
            "user_id": 3167827653,
            "sender": {"nickname": "User A"},
            "message": (
                "[CQ:at,qq=673225019] hello "
                "[CQ:at,qq=673225019] again"
            ),
        },
        ws,
    )

    payload = adapter.brain_client.post.await_args.kwargs["json"]
    envelope = payload["message_envelope"]
    assert observed_timeouts == [1.0]
    assert envelope["body_text"] == "@mentioned-user-1 hello @mentioned-user-2 again"
    assert "673225019" not in envelope["body_text"]
    assert [mention["display_name"] for mention in envelope["mentions"]] == ["", ""]
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_handle_event_sends_reply_as_message_segments():
    """Inbound reply intent should render as structured OneBot reply segments."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["905393941"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    adapter.bot_name = "Kazusa"
    adapter.brain_client.post = AsyncMock(return_value=_DummyResponse({
        "messages": ["hello there"],
        "use_reply_feature": True,
    }))
    ws = _FakeNapCatWebSocket({"message_id": "outbound-1"})

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 1602974844,
            "group_id": 905393941,
            "user_id": 2787858400,
            "sender": {"nickname": "User A"},
            "message": [
                {"type": "at", "data": {"qq": "3768713357"}},
                {"type": "text", "data": {"text": " hi"}},
            ],
        },
        ws,
    )

    send_payloads = [
        payload
        for payload in ws.sent_payloads
        if payload["action"] == "send_msg"
    ]
    assert len(send_payloads) == 1
    assert send_payloads[0]["params"]["message"] == [
        {"type": "reply", "data": {"id": "1602974844"}},
        {"type": "text", "data": {"text": "hello there"}},
    ]
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_handle_event_prefixes_delivery_mention_from_brain():
    """Normal QQ chat sends should render brain-provided mention metadata."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["905393941"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    adapter.bot_name = "Kazusa"
    adapter.brain_client.post = AsyncMock(return_value=_DummyResponse({
        "messages": ["hello there"],
        "use_reply_feature": False,
        "delivery_mentions": [
            _target_user_mention(platform_user_id="2787858400"),
        ],
    }))
    ws = _FakeNapCatWebSocket({"message_id": "outbound-mention"})

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 1602974845,
            "group_id": 905393941,
            "user_id": 2787858400,
            "sender": {"nickname": "User A"},
            "message": [
                {"type": "at", "data": {"qq": "3768713357"}},
                {"type": "text", "data": {"text": " hi"}},
            ],
        },
        ws,
    )

    send_payloads = [
        payload
        for payload in ws.sent_payloads
        if payload["action"] == "send_msg"
    ]
    assert len(send_payloads) == 1
    assert send_payloads[0]["params"]["message"] == [
        {"type": "at", "data": {"qq": "2787858400"}},
        {"type": "text", "data": {"text": " hello there"}},
    ]
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_handle_event_posts_delivery_receipt_after_send():
    """Successful normal chat sends should report platform message ids."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["905393941"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    adapter.bot_name = "Kazusa"
    adapter.brain_client.post = AsyncMock(side_effect=[
        _DummyResponse({
            "messages": ["hello there"],
            "use_reply_feature": False,
            "delivery_tracking_id": "delivery-1",
        }),
        _DummyResponse({"status": "updated", "updated": True}),
    ])
    ws = _FakeNapCatWebSocket({"message_id": "outbound-1"})

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 1602974844,
            "group_id": 905393941,
            "user_id": 2787858400,
            "sender": {"nickname": "User A"},
            "message": [{"type": "text", "data": {"text": " hi"}}],
        },
        ws,
    )

    receipt_call = adapter.brain_client.post.await_args_list[1]
    assert receipt_call.args == ("http://127.0.0.1:8000/delivery_receipt",)
    assert receipt_call.kwargs["json"] == {
        "platform": "qq",
        "platform_channel_id": "905393941",
        "delivery_tracking_id": "delivery-1",
        "platform_message_id": "outbound-1",
        "adapter": "napcat",
    }
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_delivery_receipt_retries_once_on_not_found(monkeypatch):
    """A not-found receipt should retry and stop after an updated response."""

    sleep = AsyncMock()
    monkeypatch.setattr(delivery_receipts_module.asyncio, "sleep", sleep)
    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["905393941"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    adapter.bot_name = "Kazusa"
    adapter.brain_client.post = AsyncMock(side_effect=[
        _DummyResponse({
            "messages": ["hello there"],
            "use_reply_feature": False,
            "delivery_tracking_id": "delivery-1",
        }),
        _DummyResponse({"status": "not_found", "updated": False}),
        _DummyResponse({"status": "updated", "updated": True}),
    ])
    ws = _FakeNapCatWebSocket({"message_id": "outbound-1"})

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 1602974844,
            "group_id": 905393941,
            "user_id": 2787858400,
            "sender": {"nickname": "User A"},
            "message": [{"type": "text", "data": {"text": " hi"}}],
        },
        ws,
    )

    receipt_calls = [
        call for call in adapter.brain_client.post.await_args_list
        if call.args[0].endswith("/delivery_receipt")
    ]
    assert len(receipt_calls) == 2
    sleep.assert_awaited_once_with(0.25)
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_delivery_receipt_stops_after_three_not_found(
    monkeypatch,
    caplog,
):
    """Repeated not-found receipts should stop after bounded retries."""

    sleep = AsyncMock()
    monkeypatch.setattr(delivery_receipts_module.asyncio, "sleep", sleep)
    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["905393941"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    adapter.bot_name = "Kazusa"
    adapter.brain_client.post = AsyncMock(side_effect=[
        _DummyResponse({
            "messages": ["hello there"],
            "use_reply_feature": False,
            "delivery_tracking_id": "delivery-1",
        }),
        _DummyResponse({"status": "not_found", "updated": False}),
        _DummyResponse({"status": "not_found", "updated": False}),
        _DummyResponse({"status": "not_found", "updated": False}),
    ])
    ws = _FakeNapCatWebSocket({"message_id": "outbound-1"})

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 1602974844,
            "group_id": 905393941,
            "user_id": 2787858400,
            "sender": {"nickname": "User A"},
            "message": [{"type": "text", "data": {"text": " hi"}}],
        },
        ws,
    )

    receipt_calls = [
        call for call in adapter.brain_client.post.await_args_list
        if call.args[0].endswith("/delivery_receipt")
    ]
    assert len(receipt_calls) == 3
    assert [call.args[0] for call in sleep.await_args_list] == [0.25, 0.75]
    assert "delivery-1" in caplog.text
    assert "outbound-1" in caplog.text
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_delivery_receipt_transport_failure_does_not_retry(
    caplog,
):
    """Transport failures should be warned and not retried."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["905393941"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    adapter.bot_name = "Kazusa"
    adapter.brain_client.post = AsyncMock(side_effect=[
        _DummyResponse({
            "messages": ["hello there"],
            "use_reply_feature": False,
            "delivery_tracking_id": "delivery-1",
        }),
        _RaisingResponse(),
    ])
    ws = _FakeNapCatWebSocket({"message_id": "outbound-1"})

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 1602974844,
            "group_id": 905393941,
            "user_id": 2787858400,
            "sender": {"nickname": "User A"},
            "message": [{"type": "text", "data": {"text": " hi"}}],
        },
        ws,
    )

    receipt_calls = [
        call for call in adapter.brain_client.post.await_args_list
        if call.args[0].endswith("/delivery_receipt")
    ]
    assert len(receipt_calls) == 1
    assert "delivery-1" in caplog.text
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_delivery_receipt_skips_empty_tracking_id():
    """Responses without tracking ids should not post delivery receipts."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["905393941"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    adapter.bot_name = "Kazusa"
    adapter.brain_client.post = AsyncMock(return_value=_DummyResponse({
        "messages": ["hello there"],
        "use_reply_feature": False,
        "delivery_tracking_id": "",
    }))
    ws = _FakeNapCatWebSocket({"message_id": "outbound-1"})

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 1602974844,
            "group_id": 905393941,
            "user_id": 2787858400,
            "sender": {"nickname": "User A"},
            "message": [{"type": "text", "data": {"text": " hi"}}],
        },
        ws,
    )

    assert adapter.brain_client.post.await_count == 1
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_delivery_receipt_skips_when_send_fails():
    """Failed sends should not post delivery receipts."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["905393941"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    adapter.bot_name = "Kazusa"
    adapter.brain_client.post = AsyncMock(return_value=_DummyResponse({
        "messages": ["hello there"],
        "use_reply_feature": False,
        "delivery_tracking_id": "delivery-1",
    }))
    ws = _FailingSendNapCatWebSocket()

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 1602974844,
            "group_id": 905393941,
            "user_id": 2787858400,
            "sender": {"nickname": "User A"},
            "message": [{"type": "text", "data": {"text": " hi"}}],
        },
        ws,
    )

    assert adapter.brain_client.post.await_count == 1
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_suppresses_normal_response_for_unlisted_group() -> None:
    """Listen-only QQ groups should not receive returned brain messages."""

    adapter = NapCatWSAdapter(
        ws_url="ws://napcat.local/ws",
        ws_token="token",
        brain_url="http://127.0.0.1:8000",
        brain_response_timeout=30,
        runtime_host="127.0.0.1",
        runtime_port=8011,
        runtime_public_url="http://127.0.0.1:8011",
        channel_ids=["905393941"],
        debug_modes={},
    )
    adapter.bot_id = "3768713357"
    adapter.bot_name = "Kazusa"
    adapter.brain_client.post = AsyncMock(return_value=_DummyResponse({
        "messages": ["this should stay local"],
        "use_reply_feature": False,
        "delivery_tracking_id": "delivery-suppressed",
    }))
    ws = _FakeNapCatWebSocket({"message_id": "should-not-send"})

    await adapter.handle_event(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 1602974844,
            "group_id": 99999999,
            "user_id": 2787858400,
            "sender": {"nickname": "User A"},
            "message": [{"type": "text", "data": {"text": " hi"}}],
        },
        ws,
    )

    chat_call = adapter.brain_client.post.await_args
    assert adapter.brain_client.post.await_count == 1
    assert chat_call.args == ("http://127.0.0.1:8000/chat",)
    assert chat_call.kwargs["json"]["debug_modes"]["listen_only"] is True
    assert ws.sent_payloads == []
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_runtime_capability_rejects_unlisted_group() -> None:
    """NapCat runtime capability should reject non-allowlisted groups."""

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
    adapter._ws = _FakeNapCatWebSocket({"message_id": "unused"})

    available = await adapter.can_send_message(
        channel_id="99999999",
        channel_type="group",
    )

    assert available is False
    assert adapter._ws.sent_payloads == []
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_runtime_send_rejects_unlisted_group() -> None:
    """NapCat runtime send should fail before send_msg for unlisted groups."""

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
    ws = _FakeNapCatWebSocket({"message_id": "unused"})
    adapter._ws = ws

    with pytest.raises(RuntimeError, match="NapCat target channel is not allowed"):
        await adapter.send_message(
            channel_id="99999999",
            text="scheduled hello",
            channel_type="group",
        )

    assert ws.sent_payloads == []
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_runtime_allows_unlisted_private_target() -> None:
    """NapCat private runtime sends should bypass public group allowlists."""

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
    ws = _FakeNapCatWebSocket({"message_id": "private-outbound"})
    adapter._ws = ws

    available = await adapter.can_send_message(
        channel_id="673225019",
        channel_type="private",
    )
    result = await adapter.send_message(
        channel_id="673225019",
        text="scheduled private hello",
        channel_type="private",
    )

    assert available is True
    assert ws.sent_payloads[0]["action"] == "send_msg"
    assert ws.sent_payloads[0]["params"] == {
        "message_type": "private",
        "user_id": 673225019,
        "message": "scheduled private hello",
    }
    assert result.message_id == "private-outbound"
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_runtime_send_message_uses_reply_segments():
    """Runtime reply sends should use structured OneBot reply segments."""

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
    ws = _FakeNapCatWebSocket({"message_id": "outbound-1"})
    adapter._ws = ws

    result = await adapter.send_message(
        channel_id="54369546",
        text="scheduled hello",
        channel_type="group",
        reply_to_msg_id="1615877136",
    )

    assert ws.sent_payloads[0]["action"] == "send_msg"
    assert ws.sent_payloads[0]["params"]["message"] == [
        {"type": "reply", "data": {"id": "1615877136"}},
        {"type": "text", "data": {"text": "scheduled hello"}},
    ]
    assert result.message_id == "outbound-1"
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_runtime_send_message_prefixes_delivery_mention():
    """QQ scheduled sends should render feasible prefix mention requests."""

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
    ws = _FakeNapCatWebSocket({"message_id": "outbound-mention"})
    adapter._ws = ws

    result = await adapter.send_message(
        channel_id="54369546",
        text="scheduled hello",
        channel_type="group",
        delivery_mentions=[_target_user_mention(platform_user_id="2787858400")],
    )

    assert ws.sent_payloads[0]["params"]["message"] == [
        {"type": "at", "data": {"qq": "2787858400"}},
        {"type": "text", "data": {"text": " scheduled hello"}},
    ]
    assert result.message_id == "outbound-mention"
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_runtime_send_message_noops_incomplete_delivery_mention():
    """QQ scheduled sends should ignore unrenderable mention requests."""

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
    ws = _FakeNapCatWebSocket({"message_id": "outbound-noop"})
    adapter._ws = ws

    result = await adapter.send_message(
        channel_id="54369546",
        text="scheduled hello",
        channel_type="group",
        delivery_mentions=[_target_user_mention(platform_user_id=None)],
    )

    assert ws.sent_payloads[0]["params"]["message"] == "scheduled hello"
    assert result.message_id == "outbound-noop"
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_runtime_send_message_noops_non_user_delivery_mention():
    """QQ scheduled sends should not render broad or malformed mention IDs."""

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
    ws = _FakeNapCatWebSocket({"message_id": "outbound-noop"})
    adapter._ws = ws

    result = await adapter.send_message(
        channel_id="54369546",
        text="scheduled hello",
        channel_type="group",
        delivery_mentions=[_target_user_mention(platform_user_id="all")],
    )

    assert ws.sent_payloads[0]["params"]["message"] == "scheduled hello"
    assert result.message_id == "outbound-noop"
    await adapter.close()


@pytest.mark.asyncio
async def test_napcat_runtime_endpoint_accepts_delivery_mentions():
    """The QQ callback endpoint should preserve mention metadata to send."""

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
    ws = _FakeNapCatWebSocket({"message_id": "outbound-endpoint"})
    adapter._ws = ws
    previous_runtime_adapter = napcat_module._runtime_adapter
    napcat_module._runtime_adapter = adapter

    try:
        transport = httpx.ASGITransport(app=napcat_module.runtime_app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://runtime.local",
        ) as client:
            response = await client.post(
                "/send_message",
                json={
                    "channel_id": "54369546",
                    "channel_type": "group",
                    "text": "scheduled hello",
                    "reply_to_msg_id": None,
                    "delivery_mentions": [
                        _target_user_mention(platform_user_id="2787858400")
                    ],
                },
            )

        assert response.status_code == 200
        assert ws.sent_payloads[0]["params"]["message"] == [
            {"type": "at", "data": {"qq": "2787858400"}},
            {"type": "text", "data": {"text": " scheduled hello"}},
        ]
    finally:
        napcat_module._runtime_adapter = previous_runtime_adapter
        await adapter.close()


@pytest.mark.asyncio
async def test_napcat_runtime_send_message_uses_private_message_type():
    """Runtime private sends should use OneBot private message parameters."""

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
    ws = _FakeNapCatWebSocket({"message_id": "outbound-private"})
    adapter._ws = ws
    previous_runtime_adapter = napcat_module._runtime_adapter
    napcat_module._runtime_adapter = adapter

    try:
        transport = httpx.ASGITransport(app=napcat_module.runtime_app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://runtime.local",
        ) as client:
            response = await client.post(
                "/send_message",
                json={
                    "channel_id": "673225019",
                    "channel_type": "private",
                    "text": "scheduled private hello",
                    "reply_to_msg_id": None,
                },
            )

        assert response.status_code == 200
        assert ws.sent_payloads[0]["action"] == "send_msg"
        assert ws.sent_payloads[0]["params"] == {
            "message_type": "private",
            "user_id": 673225019,
            "message": "scheduled private hello",
        }
    finally:
        napcat_module._runtime_adapter = previous_runtime_adapter
        await adapter.close()


@pytest.mark.asyncio
async def test_napcat_runtime_send_message_requires_channel_type():
    """NapCat runtime send requests must carry explicit channel-type metadata."""

    transport = httpx.ASGITransport(app=napcat_module.runtime_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://runtime.local",
    ) as client:
        response = await client.post(
            "/send_message",
            json={
                "channel_id": "673225019",
                "text": "scheduled private hello",
                "reply_to_msg_id": None,
            },
        )

    assert response.status_code == 422


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


@pytest.mark.asyncio
async def test_discord_handle_message_posts_first_delivery_receipt(
    monkeypatch,
):
    """Discord normal chat sends should report the first platform message id."""

    monkeypatch.setattr(
        discord_module,
        "_split_message",
        lambda text: ["first chunk", "second chunk"],
    )
    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        channel_ids=["12345"],
        debug_modes={},
    )
    adapter._http_client.post = AsyncMock(side_effect=[
        _DummyResponse({
            "messages": ["first chunk\nsecond chunk"],
            "use_reply_feature": True,
            "delivery_tracking_id": "delivery-1",
        }),
        _DummyResponse({"status": "updated", "updated": True}),
    ])
    message = _FakeDiscordMessage()

    await adapter.on_message(message)

    receipt_call = adapter._http_client.post.await_args_list[1]
    assert receipt_call.args == ("http://127.0.0.1:8000/delivery_receipt",)
    assert receipt_call.kwargs["json"] == {
        "platform": "discord",
        "platform_channel_id": "12345",
        "delivery_tracking_id": "delivery-1",
        "platform_message_id": "discord-reply-1",
        "adapter": "discord",
    }
    assert message.reply_chunks == ["first chunk"]
    assert message.channel.sent_chunks == ["second chunk"]
    await adapter.close()


@pytest.mark.asyncio
async def test_discord_on_message_prefixes_delivery_mention_from_brain():
    """Normal Discord chat sends should render brain-provided mentions."""

    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        channel_ids=["12345"],
        debug_modes={},
    )
    adapter._http_client.post = AsyncMock(return_value=_DummyResponse({
        "messages": ["hello there"],
        "use_reply_feature": False,
        "delivery_mentions": [
            _target_user_mention(platform_user_id="2787858400"),
        ],
        "delivery_tracking_id": "delivery-mention",
    }))
    message = _FakeDiscordMessage()

    await adapter.on_message(message)

    assert message.reply_chunks == []
    assert message.channel.sent_chunks == ["<@2787858400> hello there"]
    await adapter.close()


@pytest.mark.asyncio
async def test_discord_suppresses_normal_response_for_unlisted_group() -> None:
    """Listen-only Discord guild channels should not receive brain messages."""

    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        channel_ids=["12345"],
        debug_modes={},
    )
    adapter._http_client.post = AsyncMock(return_value=_DummyResponse({
        "messages": ["this should stay local"],
        "use_reply_feature": False,
        "delivery_tracking_id": "delivery-suppressed",
    }))
    message = _FakeDiscordMessage()
    message.channel.id = 99999

    await adapter.on_message(message)

    chat_call = adapter._http_client.post.await_args
    assert adapter._http_client.post.await_count == 1
    assert chat_call.args == ("http://127.0.0.1:8000/chat",)
    assert chat_call.kwargs["json"]["debug_modes"]["listen_only"] is True
    assert message.reply_chunks == []
    assert message.channel.sent_chunks == []
    await adapter.close()


@pytest.mark.asyncio
async def test_discord_runtime_capability_rejects_unlisted_group() -> None:
    """Discord runtime capability should reject non-allowlisted guild channels."""

    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        channel_ids=["12345"],
        debug_modes={},
    )
    adapter.get_channel = MagicMock(return_value=_FakeDiscordChannel())
    adapter.fetch_channel = AsyncMock(return_value=_FakeDiscordChannel())

    available = await adapter.can_send_message(
        channel_id="99999",
        channel_type="group",
    )

    assert available is False
    adapter.get_channel.assert_not_called()
    adapter.fetch_channel.assert_not_awaited()
    await adapter.close()


@pytest.mark.asyncio
async def test_discord_runtime_send_rejects_unlisted_group() -> None:
    """Discord runtime send should fail before native send for unlisted groups."""

    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        channel_ids=["12345"],
        debug_modes={},
    )
    channel = _FakeDiscordChannel()
    adapter.get_channel = MagicMock(return_value=channel)
    adapter.fetch_channel = AsyncMock(return_value=channel)

    with pytest.raises(RuntimeError, match="Discord target channel is not allowed"):
        await adapter.send_message(
            channel_id="99999",
            text="scheduled hello",
            channel_type="group",
        )

    assert channel.sent_chunks == []
    adapter.get_channel.assert_not_called()
    adapter.fetch_channel.assert_not_awaited()
    await adapter.close()


@pytest.mark.asyncio
async def test_discord_runtime_allows_unlisted_private_target() -> None:
    """Discord private runtime sends should bypass public channel allowlists."""

    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        channel_ids=["12345"],
        debug_modes={},
    )
    channel = _FakeDiscordChannel()
    adapter.get_channel = MagicMock(return_value=channel)

    available = await adapter.can_send_message(
        channel_id="99999",
        channel_type="private",
    )
    result = await adapter.send_message(
        channel_id="99999",
        text="scheduled private hello",
        channel_type="private",
    )

    assert available is True
    assert channel.sent_chunks == ["scheduled private hello"]
    assert result.message_id == "discord-send-1"
    await adapter.close()


@pytest.mark.asyncio
async def test_discord_runtime_send_message_accepts_channel_type():
    """Discord runtime sends should accept scheduler channel-type metadata."""

    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        channel_ids=["12345"],
        debug_modes={},
    )
    channel = _FakeDiscordChannel()
    adapter.get_channel = MagicMock(return_value=channel)

    result = await adapter.send_message(
        channel_id="12345",
        text="scheduled hello",
        channel_type="private",
    )

    assert channel.sent_chunks == ["scheduled hello"]
    assert result.message_id == "discord-send-1"
    await adapter.close()


@pytest.mark.asyncio
async def test_discord_runtime_send_message_prefixes_delivery_mention():
    """Discord scheduled sends should render feasible prefix mentions."""

    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        channel_ids=["12345"],
        debug_modes={},
    )
    channel = _FakeDiscordChannel()
    adapter.get_channel = MagicMock(return_value=channel)

    result = await adapter.send_message(
        channel_id="12345",
        text="scheduled hello",
        channel_type="group",
        delivery_mentions=[_target_user_mention(platform_user_id="2787858400")],
    )

    assert channel.sent_chunks == ["<@2787858400> scheduled hello"]
    assert result.message_id == "discord-send-1"
    await adapter.close()


@pytest.mark.asyncio
async def test_discord_runtime_send_message_noops_incomplete_delivery_mention():
    """Discord scheduled sends should ignore unrenderable mentions."""

    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        channel_ids=["12345"],
        debug_modes={},
    )
    channel = _FakeDiscordChannel()
    adapter.get_channel = MagicMock(return_value=channel)

    result = await adapter.send_message(
        channel_id="12345",
        text="scheduled hello",
        channel_type="group",
        delivery_mentions=[_target_user_mention(platform_user_id=None)],
    )

    assert channel.sent_chunks == ["scheduled hello"]
    assert result.message_id == "discord-send-1"
    await adapter.close()


@pytest.mark.asyncio
async def test_discord_runtime_send_message_noops_malformed_delivery_mention():
    """Discord scheduled sends should not interpolate malformed mention IDs."""

    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        channel_ids=["12345"],
        debug_modes={},
    )
    channel = _FakeDiscordChannel()
    adapter.get_channel = MagicMock(return_value=channel)

    result = await adapter.send_message(
        channel_id="12345",
        text="scheduled hello",
        channel_type="group",
        delivery_mentions=[
            _target_user_mention(platform_user_id="123> @everyone")
        ],
    )

    assert channel.sent_chunks == ["scheduled hello"]
    assert result.message_id == "discord-send-1"
    await adapter.close()


@pytest.mark.asyncio
async def test_discord_runtime_endpoint_accepts_delivery_mentions():
    """The Discord callback endpoint should preserve mention metadata to send."""

    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        channel_ids=["12345"],
        debug_modes={},
    )
    channel = _FakeDiscordChannel()
    adapter.get_channel = MagicMock(return_value=channel)
    previous_runtime_adapter = discord_module._runtime_adapter
    discord_module._runtime_adapter = adapter

    try:
        transport = httpx.ASGITransport(app=discord_module.runtime_app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://runtime.local",
        ) as client:
            response = await client.post(
                "/send_message",
                headers={"Authorization": "Bearer secret-token"},
                json={
                    "channel_id": "12345",
                    "channel_type": "group",
                    "text": "scheduled hello",
                    "reply_to_msg_id": None,
                    "delivery_mentions": [
                        _target_user_mention(platform_user_id="2787858400")
                    ],
                },
            )

        assert response.status_code == 200
        assert channel.sent_chunks == ["<@2787858400> scheduled hello"]
    finally:
        discord_module._runtime_adapter = previous_runtime_adapter
        await adapter.close()


@pytest.mark.asyncio
async def test_discord_runtime_send_message_requires_channel_type():
    """Discord runtime send requests must carry explicit channel-type metadata."""

    transport = httpx.ASGITransport(app=discord_module.runtime_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://runtime.local",
    ) as client:
        response = await client.post(
            "/send_message",
            json={
                "channel_id": "12345",
                "text": "scheduled hello",
                "reply_to_msg_id": None,
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_discord_delivery_receipt_transport_failure_does_not_retry(
    caplog,
):
    """Discord receipt transport failures should be warned and not retried."""

    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        channel_ids=["12345"],
        debug_modes={},
    )
    adapter._http_client.post = AsyncMock(side_effect=[
        _DummyResponse({
            "messages": ["hello there"],
            "use_reply_feature": False,
            "delivery_tracking_id": "delivery-1",
        }),
        _RaisingResponse(),
    ])
    message = _FakeDiscordMessage()

    await adapter.on_message(message)

    receipt_calls = [
        call for call in adapter._http_client.post.await_args_list
        if call.args[0].endswith("/delivery_receipt")
    ]
    assert len(receipt_calls) == 1
    assert message.channel.sent_chunks == ["hello there"]
    assert "delivery-1" in caplog.text
    await adapter.close()


@pytest.mark.asyncio
async def test_discord_delivery_receipt_skips_empty_tracking_id():
    """Discord responses without tracking ids should not post receipts."""

    adapter = DiscordAdapter(
        brain_url="http://127.0.0.1:8000",
        runtime_host="127.0.0.1",
        runtime_port=8012,
        runtime_public_url="http://127.0.0.1:8012",
        runtime_shared_secret="secret-token",
        channel_ids=["12345"],
        debug_modes={},
    )
    adapter._http_client.post = AsyncMock(return_value=_DummyResponse({
        "messages": ["hello there"],
        "use_reply_feature": False,
    }))
    message = _FakeDiscordMessage()

    await adapter.on_message(message)

    adapter._http_client.post.assert_awaited_once()
    chat_call = adapter._http_client.post.await_args
    assert chat_call.args == ("http://127.0.0.1:8000/chat",)
    assert chat_call.kwargs["json"]["platform"] == "discord"
    assert message.channel.sent_chunks == ["hello there"]
    await adapter.close()
