"""NapCat QQ adapter that normalizes platform events before brain intake."""

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import json
import logging
import os
import re
import sys
from types import SimpleNamespace
from typing import Optional

import httpx
import uvicorn
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from adapters.envelope_common import (
    addressed_to_from_envelope_parts,
    attachment_refs,
    normalize_body_spacing,
    resolve_mentions,
)
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.dispatcher import SendResult
from kazusa_ai_chatbot.logging_config import configure_adapter_logging
from kazusa_ai_chatbot.message_envelope import (
    AttachmentHandlerRegistryProtocol,
    MentionResolver,
    PassthroughMentionResolver,
    build_default_attachment_handler_registry,
)
from kazusa_ai_chatbot.message_envelope.types import (
    MessageEnvelope,
    RawMention,
    ReplyTarget,
)

configure_adapter_logging()

logger = logging.getLogger(__name__)

runtime_app = FastAPI(title="Kazusa NapCat Runtime Adapter")
_runtime_adapter: "NapCatWSAdapter | None" = None

_CQ_REPLY_PATTERN = re.compile(r"\[CQ:reply,id=([^\],]+)[^\]]*\]")
_CQ_AT_PATTERN = re.compile(r"\[CQ:at,qq=([^\],]+)[^\]]*\]")
_CQ_ANY_PATTERN = re.compile(r"\[CQ:[^\]]+\]")


class QQEnvelopeNormalizer:
    """Adapter-local QQ/NapCat normalizer for brain-safe envelopes."""

    def normalize(
        self,
        request: object,
        mention_resolver: MentionResolver,
        attachment_handlers: AttachmentHandlerRegistryProtocol,
    ) -> MessageEnvelope:
        """Normalize one QQ request.

        Args:
            request: Request-like object with QQ wire content and metadata.
            mention_resolver: Resolver used to project raw mentions.
            attachment_handlers: Registry used to normalize attachments.

        Returns:
            Message envelope with QQ CQ/reply/mention markers removed from
            `body_text`.
        """

        raw_wire_text = str(request.content or "")
        platform_bot_id = str(request.platform_bot_id)
        channel_type = str(request.channel_type)
        reply_context = dict(request.reply_context)

        raw_mentions = self._raw_mentions(raw_wire_text, platform_bot_id)
        reply = self._reply_target(raw_wire_text, reply_context, platform_bot_id)

        body_text = _CQ_REPLY_PATTERN.sub(" ", raw_wire_text)
        body_text = _CQ_AT_PATTERN.sub(" ", body_text)
        body_text = _CQ_ANY_PATTERN.sub(" ", body_text)
        body_text = normalize_body_spacing(body_text)

        mentions = resolve_mentions(raw_mentions, mention_resolver)
        addressed_to = addressed_to_from_envelope_parts(
            mentions=mentions,
            reply=reply,
            channel_type=channel_type,
        )
        envelope: MessageEnvelope = {
            "body_text": body_text,
            "raw_wire_text": raw_wire_text,
            "mentions": mentions,
            "attachments": attachment_refs(request.attachments, attachment_handlers),
            "addressed_to_global_user_ids": addressed_to,
            "broadcast": False,
        }
        if reply:
            envelope["reply"] = reply

        return envelope

    def _raw_mentions(
        self,
        raw_wire_text: str,
        platform_bot_id: str,
    ) -> list[RawMention]:
        """Extract QQ wire mention markers for bot/user addressing."""

        raw_mentions: list[RawMention] = []
        for match in _CQ_AT_PATTERN.finditer(raw_wire_text):
            platform_user_id = match.group(1)
            entity_kind = "bot" if platform_user_id == platform_bot_id else "user"
            if platform_user_id.lower() == "all":
                entity_kind = "everyone"
            raw_mentions.append({
                "platform": "qq",
                "platform_user_id": platform_user_id,
                "entity_kind": entity_kind,
                "raw_text": match.group(0),
            })

        return raw_mentions

    def _reply_target(
        self,
        raw_wire_text: str,
        reply_context: dict,
        platform_bot_id: str,
    ) -> ReplyTarget:
        """Extract the typed reply target from CQ text and adapter metadata."""

        reply: ReplyTarget = {}
        reply_match = _CQ_REPLY_PATTERN.search(raw_wire_text)
        if reply_match is not None:
            reply["platform_message_id"] = reply_match.group(1)
            reply["derivation"] = "platform_native"

        if reply_context.get("reply_to_message_id"):
            reply["platform_message_id"] = str(reply_context["reply_to_message_id"])
            reply["derivation"] = "platform_native"
        if reply_context.get("reply_to_platform_user_id"):
            platform_user_id = str(reply_context["reply_to_platform_user_id"])
            reply["platform_user_id"] = platform_user_id
            if platform_user_id == platform_bot_id:
                reply["global_user_id"] = CHARACTER_GLOBAL_USER_ID
        if reply_context.get("reply_to_display_name"):
            reply["display_name"] = str(reply_context["reply_to_display_name"])
        if reply_context.get("reply_excerpt"):
            reply["excerpt"] = str(reply_context["reply_excerpt"])
        return reply


class RuntimeSendMessageRequest(BaseModel):
    channel_id: str
    text: str
    reply_to_msg_id: str | None = None


class RuntimeSendMessageResponse(BaseModel):
    platform: str
    channel_id: str
    message_id: str
    sent_at: str


@runtime_app.post("/send_message", response_model=RuntimeSendMessageResponse)
async def send_message_endpoint(
    req: RuntimeSendMessageRequest,
    authorization: str = Header(default=""),
):
    """Deliver one scheduled outbound message through the live NapCat adapter."""

    if _runtime_adapter is None:
        raise HTTPException(status_code=503, detail="Runtime adapter is not ready")
    if _runtime_adapter.runtime_shared_secret:
        expected = f"Bearer {_runtime_adapter.runtime_shared_secret}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        result = await _runtime_adapter.send_message(
            channel_id=req.channel_id,
            text=req.text,
            reply_to_msg_id=req.reply_to_msg_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return_value = RuntimeSendMessageResponse(
        platform=result.platform,
        channel_id=result.channel_id,
        message_id=result.message_id,
        sent_at=result.sent_at.isoformat(),
    )
    return return_value


class NapCatWSAdapter:
    """Websocket adapter that forwards QQ/NapCat messages to the brain."""

    platform = "qq"

    def __init__(
        self,
        ws_url: str,
        ws_token: str,
        brain_url: str,
        brain_response_timeout: int,
        runtime_host: str,
        runtime_port: int,
        runtime_public_url: str,
        runtime_shared_secret: str = "",
        heartbeat_seconds: float = 30.0,
        channel_ids: list[str] | None = None,
        debug_modes: dict | None = None,
    ):
        # User arguments
        self.ws_url: str = ws_url
        self.ws_token = ws_token
        self.brain_url = brain_url.rstrip("/")
        self.channel_ids = set(channel_ids) if channel_ids is not None else None
        self.runtime_host = runtime_host
        self.runtime_port = runtime_port
        self.runtime_public_url = runtime_public_url.rstrip("/")
        self.runtime_shared_secret = runtime_shared_secret
        self.heartbeat_seconds = heartbeat_seconds

        # The following will be populated on connect
        self.bot_id: Optional[str] = None
        self.bot_name: Optional[str] = None
        self._ws = None
        self._api_response_futures: dict[str, asyncio.Future] = {}
        self._api_dispatch_enabled = False
        self._runtime_server: uvicorn.Server | None = None
        self._runtime_server_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None

        self.debug_modes = debug_modes or {}
        self._envelope_normalizer = QQEnvelopeNormalizer()
        self._attachment_handlers = build_default_attachment_handler_registry()
        self._mention_resolver = PassthroughMentionResolver()

        # Initialize connection
        self.brain_client = httpx.AsyncClient(base_url=self.brain_url, timeout=brain_response_timeout)

    async def connect(self):
        """Connect to NapCat and keep processing incoming websocket events."""

        headers = {"Authorization": f"Bearer {self.ws_token}"} if self.ws_token else {}

        while True:
            try:
                logger.info(f"Connecting to NapCat at {self.ws_url}...")
                async with websockets.connect(self.ws_url, additional_headers=headers) as ws:
                    self._ws = ws
                    # 1. Sync Bot Info immediately after connecting
                    await self._fetch_bot_info(ws)
                    await self._ensure_runtime_server_started()
                    await self._register_with_brain()
                    self._ensure_heartbeat_started()
                    logger.info(f"Logged in as {self.bot_name} (ID: {self.bot_id})")
                    if self.channel_ids is not None:
                        logger.info(f'Active in groups: {self.channel_ids}. Other groups are listen-only.')
                    else:
                        logger.info("No active groups configured — all groups are listen-only (Private chats are active).")

                    # 2. Main Event Loop
                    self._api_dispatch_enabled = True
                    while True:
                        message = await ws.recv()
                        data = json.loads(message)

                        if self._resolve_api_response(data):
                            continue

                        # Handle Events (messages, etc.)
                        asyncio.create_task(self.handle_event(data, ws))
            except Exception as exc:
                logger.debug(f"Handled exception in connect: {exc}")
                logger.exception(f"Connection lost. Retrying in 5s: {exc}")
                self._api_dispatch_enabled = False
                self._reject_pending_api_responses(exc)
                self._ws = None
                await asyncio.sleep(5)

    async def _ensure_runtime_server_started(self) -> None:
        """Start the adapter callback server exactly once for brain-side sends."""

        global _runtime_adapter

        _runtime_adapter = self
        if self._runtime_server_task is not None:
            return

        config = uvicorn.Config(
            runtime_app,
            host=self.runtime_host,
            port=self.runtime_port,
            log_level="warning",
        )
        self._runtime_server = uvicorn.Server(config)
        self._runtime_server_task = asyncio.create_task(self._runtime_server.serve())
        logger.info(f'NapCat runtime callback listening on {self.runtime_host}:{self.runtime_port}')

    async def _register_with_brain(self) -> None:
        """Register this adapter's outbound callback URL with the brain service."""

        payload = self._runtime_registration_payload()
        response = await self.brain_client.post(
            f"{self.brain_url}/runtime/adapters/register",
            json=payload,
        )
        response.raise_for_status()
        logger.info(f'Registered NapCat runtime adapter with brain: callback_url={self.runtime_public_url}')

    def _runtime_registration_payload(self) -> dict:
        """Return the shared registration payload for startup and heartbeat."""

        return_value = {
            "platform": self.platform,
            "callback_url": self.runtime_public_url,
            "shared_secret": self.runtime_shared_secret,
            "timeout_seconds": 10.0,
        }
        return return_value

    async def _send_heartbeat_once(self) -> None:
        """Refresh the adapter registration in the brain service."""

        response = await self.brain_client.post(
            f"{self.brain_url}/runtime/adapters/heartbeat",
            json=self._runtime_registration_payload(),
        )
        response.raise_for_status()

    def _ensure_heartbeat_started(self) -> None:
        """Start the re-registration heartbeat exactly once."""

        if self._heartbeat_task is not None:
            return
        self._heartbeat_task = asyncio.create_task(self._run_brain_heartbeat())

    async def _run_brain_heartbeat(self) -> None:
        """Keep refreshing adapter registration so brain restarts self-heal."""

        while True:
            await asyncio.sleep(self.heartbeat_seconds)
            try:
                await self._send_heartbeat_once()
            except httpx.HTTPError as exc:
                logger.warning(f'NapCat runtime heartbeat failed: {exc}')

    async def _fetch_bot_info(self, ws):
        """Retrieve the bot's QQ identity from NapCat.

        Args:
            ws: Active NapCat websocket connection.

        Returns:
            None.
        """

        response = await self._call_api(ws, "get_login_info")
        if response.get("status") == "ok":
            data = response.get("data", {})
            self.bot_id = str(data.get("user_id"))
            self.bot_name = data.get("nickname")
        else:
            logger.warning("Could not retrieve bot info, using defaults.")
            self.bot_id = "unknown"
            self.bot_name = "NapCat Bot"

    def _resolve_api_response(self, data: dict) -> bool:
        """Resolve a pending websocket API call from a received echo response.

        Args:
            data: Decoded NapCat websocket frame.

        Returns:
            True when the frame was consumed as an API response.
        """
        echo_id = data.get("echo")
        if not echo_id:
            return False

        future = self._api_response_futures.pop(echo_id, None)
        if future is None:
            return False

        if not future.done():
            future.set_result(data)
        return True

    def _reject_pending_api_responses(self, exc: BaseException) -> None:
        """Fail all pending websocket API calls after the connection drops.

        Args:
            exc: Exception that caused the websocket connection to stop.

        Returns:
            None.
        """
        pending = list(self._api_response_futures.values())
        self._api_response_futures.clear()
        for future in pending:
            if not future.done():
                future.set_exception(exc)

    async def _call_api(self, ws, action: str, params: dict | None = None):
        """Call a OneBot API action and wait for the matching echo response.

        Args:
            ws: Active NapCat websocket connection.
            action: OneBot action name.
            params: Optional action parameters.

        Returns:
            Decoded OneBot response payload for the requested echo id.
        """

        echo_id = f"sync_{action}_{asyncio.get_event_loop().time()}"
        payload = {
            "action": action,
            "params": params or {},
            "echo": echo_id,
        }

        if self._api_dispatch_enabled:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._api_response_futures[echo_id] = future
            await ws.send(json.dumps(payload))
            try:
                return_value = await asyncio.wait_for(future, timeout=10.0)
                return return_value
            finally:
                self._api_response_futures.pop(echo_id, None)

        await ws.send(json.dumps(payload))

        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=10.0)
            response = json.loads(message)
            if response.get("echo") == echo_id:
                return response

    def _apply_replied_message_metadata(self, reply_context: dict[str, str | bool], message_data: dict) -> None:
        """Populate reply target fields from a NapCat/OneBot message document.

        Args:
            reply_context: Mutable reply context that will be forwarded to the
                brain service.
            message_data: Message document returned by the platform ``get_msg``
                API for the replied-to message.

        Returns:
            None.
        """
        sender = message_data.get("sender")
        if not isinstance(sender, dict):
            sender = {}

        target_user_id = message_data.get("user_id") or sender.get("user_id")
        if target_user_id is not None:
            reply_context["reply_to_platform_user_id"] = str(target_user_id)

        target_name = sender.get("card") or sender.get("nickname")
        if target_name:
            reply_context["reply_to_display_name"] = str(target_name)

        reply_excerpt = message_data.get("raw_message") or message_data.get("message")
        if isinstance(reply_excerpt, str) and reply_excerpt:
            reply_context["reply_excerpt"] = reply_excerpt

    async def _hydrate_reply_context_from_platform(self, reply_context: dict[str, str | bool], ws) -> None:
        """Resolve reply target metadata from NapCat before calling the brain.

        Args:
            reply_context: Mutable reply context extracted from the incoming
                message event.
            ws: Active NapCat websocket used for OneBot API calls.

        Returns:
            None. The context is updated in place when platform metadata is
            available.
        """
        reply_to_message_id = reply_context.get("reply_to_message_id")
        if not reply_to_message_id or reply_context.get("reply_to_platform_user_id"):
            return

        params: dict[str, int | str] = {"message_id": str(reply_to_message_id)}
        if str(reply_to_message_id).isdigit():
            params["message_id"] = int(str(reply_to_message_id))

        try:
            response = await self._call_api(ws, "get_msg", params)
        except (asyncio.TimeoutError, websockets.exceptions.WebSocketException) as exc:
            logger.warning(f'Failed to resolve QQ reply target message_id={reply_to_message_id}: {exc}')
            return

        if response.get("status") != "ok":
            logger.warning(f'QQ reply target lookup returned status={response.get("status")} message_id={reply_to_message_id}')
            return

        message_data = response.get("data", {})
        if not isinstance(message_data, dict):
            logger.warning(f'QQ reply target lookup returned non-dict data for message_id={reply_to_message_id}')
            return

        self._apply_replied_message_metadata(reply_context, message_data)

    async def handle_event(self, data: dict, ws):
        """Normalize one NapCat event and forward message events to the brain.

        Args:
            data: Decoded NapCat websocket event.
            ws: Active NapCat websocket connection.

        Returns:
            None.
        """

        if data.get("post_type") != "message" or not self.bot_id:
            return

        user_id = str(data.get("user_id"))
        message_id = str(data.get("message_id", ""))
        group_id = data.get("group_id")

        message_data = data.get("message", [])
        reply_context: dict[str, str] = {}

        # Preprocess QQ message into a platform-faithful wire form. The typed
        # envelope normalizer below is the only layer that strips CQ markers.
        if isinstance(message_data, str):
            wire_content = message_data
            reply_match = re.search(r"\[CQ:reply,id=([^\],]+)[^\]]*\]", wire_content)
            if reply_match is not None:
                reply_context["reply_to_message_id"] = reply_match.group(1)
        else:
            # If it's a segment list, rebuild it to CQ-style wire text.
            wire_parts: list[str] = []
            for seg in message_data:
                if not isinstance(seg, dict):
                    continue
                seg_type = seg.get("type")
                seg_data = seg.get("data", {})
                if seg_type == "text":
                    wire_parts.append(seg_data.get("text", ""))
                elif seg_type == "at":
                    qq = seg_data.get("qq")
                    wire_parts.append(f"[CQ:at,qq={qq}]")
                elif seg_type == "reply":
                    reply_context["reply_to_message_id"] = str(seg_data.get("id", ""))
                    reply_sender_id = seg_data.get("user_id")
                    if reply_sender_id is not None:
                        reply_context["reply_to_platform_user_id"] = str(reply_sender_id)
                    reply_sender_name = seg_data.get("nickname")
                    if reply_sender_name:
                        reply_context["reply_to_display_name"] = str(reply_sender_name)
                    reply_text = seg_data.get("text")
                    if reply_text:
                        reply_context["reply_excerpt"] = str(reply_text)
                    wire_parts.append(f"[CQ:reply,id={seg_data.get('id', '')}]")
                elif seg_type == "face":
                    face_id = seg_data.get("id", "")
                    wire_parts.append(f"[CQ:face,id={face_id}]")
                elif seg_type == "image":
                    image_url = seg_data.get("url", "")
                    wire_parts.append(f"[CQ:image,url={image_url}]")
                # image/video etc. are handled by attachments array, so we omit them from raw text
            wire_content = "".join(wire_parts)

        wire_content = wire_content.strip()
        await self._hydrate_reply_context_from_platform(reply_context, ws)
        sender = data.get("sender", {})
        sender_name = sender.get("nickname", f"User {user_id}")

        is_group = data.get("message_type") == "group"
        channel_id = str(group_id) if is_group else user_id

        message_debug_modes = dict(self.debug_modes)
        is_active = (not is_group) or (self.channel_ids is not None and str(group_id) in self.channel_ids)

        if not is_active:
            message_debug_modes["listen_only"] = True

        mode_label = "LISTEN-ONLY" if message_debug_modes.get("listen_only") else "ACTIVE"

        logger.info(f'[{mode_label}] Incoming QQ message: channel_id={channel_id} is_group={is_group} sender={sender_name} raw_wire={wire_content}')

        # Attachments processing (same as before)
        attachments = []
        for seg in data.get("message", []):
            if isinstance(seg, dict) and seg.get("type") == "image":
                url = seg["data"].get("url")
                if url:
                    try:
                        resp = await self.brain_client.get(url, timeout=10.0)
                        resp.raise_for_status()
                        attachments.append({
                            "media_type": "image/jpeg",
                            "base64_data": base64.b64encode(resp.content).decode("utf-8"),
                            "description": "",
                        })
                    except httpx.HTTPError as exc:
                        logger.debug(f"Handled exception in handle_event: {exc}")
                        logger.exception(f"Image fetch error: {exc}")

        envelope_request = SimpleNamespace(
            platform="qq",
            channel_type="group" if is_group else "private",
            content=wire_content,
            platform_bot_id=self.bot_id,
            reply_context=reply_context,
            attachments=attachments,
        )
        envelope = self._envelope_normalizer.normalize(
            envelope_request,
            self._mention_resolver,
            self._attachment_handlers,
        )
        raw_content = envelope["body_text"]

        # Brain Forwarding
        payload = {
            "platform": "qq",
            "platform_channel_id": channel_id,
            "channel_type": "group" if is_group else "private",
            "platform_message_id": message_id,
            "platform_user_id": user_id,
            "platform_bot_id": self.bot_id,
            "display_name": sender_name,
            "channel_name": f"Group {group_id}" if is_group else "Private",
            "content_type": "text",
            "message_envelope": envelope,
            "debug_modes": message_debug_modes,
        }

        logger.info(f'Forwarding to brain: channel_id={channel_id} user_id={user_id} content={raw_content} attachments={len(attachments)}')

        try:
            resp = await self.brain_client.post(f"{self.brain_url}/chat", json=payload)
            resp.raise_for_status()
            brain_data = resp.json()
        except Exception as exc:
            logger.debug(f"Handled exception in handle_event: {exc}")
            logger.exception(f"Brain service failed: {exc}")
            return

        # Response handling
        replies = brain_data.get("messages", [])
        logger.info(f'Brain output: should_reply={brain_data.get("should_reply")} message_count={len(replies)} messages={replies}')
        if replies:
            combined = "\n".join(replies)
            msg_params = {
                "message_type": "group" if is_group else "private",
                "group_id" if is_group else "user_id": int(channel_id),
                "message": combined
            }
            if brain_data.get("should_reply"):
                msg_params["message"] = f"[CQ:reply,id={data['message_id']}]" + combined

            logger.info(f'Sending QQ message: channel_id={channel_id} message={msg_params["message"]}')

            # We don't need to wait for 'send_msg' response usually
            await ws.send(json.dumps({"action": "send_msg", "params": msg_params}))

    async def close(self):
        """Close adapter-owned network clients and callback server."""

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError as exc:
                logger.debug(f"Handled exception in close: {exc}")
                pass
        await self.brain_client.aclose()
        if self._runtime_server is not None:
            self._runtime_server.should_exit = True
        if self._runtime_server_task is not None:
            await self._runtime_server_task

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to_msg_id: str | None = None,
    ) -> SendResult:
        """Send one outbound message through NapCat's ``send_msg`` action.

        Args:
            channel_id: QQ group or private-chat identifier.
            text: Message body to send.
            reply_to_msg_id: Optional quoted message id.

        Returns:
            Structured send result for the dispatcher.
        """

        if self._ws is None:
            raise RuntimeError("NapCat websocket is not connected")

        message = text
        if reply_to_msg_id:
            message = f"[CQ:reply,id={reply_to_msg_id}]{message}"

        params = {
            "message_type": "group",
            "group_id": int(channel_id),
            "message": message,
        }
        await self._ws.send(json.dumps({"action": "send_msg", "params": params}))
        return_value = SendResult(
            platform=self.platform,
            channel_id=channel_id,
            message_id="",
            sent_at=datetime.now(timezone.utc),
        )
        return return_value


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="NapCat QQ adapter for Kazusa Brain Service")
    parser.add_argument("--channels", type=str, nargs="*", default=None, help="QQ Group IDs to actively participate in. Other groups will be listen-only.")
    parser.add_argument("--runtime-host", type=str, default=os.getenv("ADAPTER_RUNTIME_HOST", "127.0.0.1"))
    parser.add_argument("--runtime-port", type=int, default=int(os.getenv("NAPCAT_RUNTIME_PORT", "8011")))
    parser.add_argument("--runtime-public-url", type=str, default=os.getenv("ADAPTER_RUNTIME_PUBLIC_URL", ""))
    parser.add_argument("--heartbeat-seconds", type=float, default=float(os.getenv("ADAPTER_HEARTBEAT_SECONDS", "30")))

    args = parser.parse_args()

    ws_url = os.getenv("NAPCAT_WS_URL")
    if not ws_url:
        logger.error("NAPCAT_WS_URL environment variable is required")
        sys.exit(1)

    ws_token = os.getenv("NAPCAT_WS_TOKEN")
    if not ws_token:
        logger.error("NAPCAT_WS_TOKEN environment variable is required")
        sys.exit(1)

    brain_url = os.getenv("BRAIN_URL")
    if not brain_url:
        logger.error("BRAIN_URL environment variable is required")
        sys.exit(1)

    brain_response_timeout = os.getenv("BRAIN_RESPONSE_TIMEOUT", "120")
    brain_response_timeout = int(brain_response_timeout)
    runtime_public_url = args.runtime_public_url or f"http://127.0.0.1:{args.runtime_port}"
    runtime_shared_secret = os.getenv("ADAPTER_RUNTIME_SHARED_SECRET", "")

    adapter = NapCatWSAdapter(
        ws_url=ws_url,
        ws_token=ws_token,
        brain_url=brain_url,
        brain_response_timeout=brain_response_timeout,
        runtime_host=args.runtime_host,
        runtime_port=args.runtime_port,
        runtime_public_url=runtime_public_url,
        runtime_shared_secret=runtime_shared_secret,
        heartbeat_seconds=args.heartbeat_seconds,
        channel_ids=args.channels,
        debug_modes={},
    )

    try:
        asyncio.run(adapter.connect())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        asyncio.run(adapter.close())


if __name__ == "__main__":
    main()
