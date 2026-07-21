"""NapCat websocket orchestration, API dispatch, and brain forwarding."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import logging
from types import SimpleNamespace
from typing import Any, Optional

import httpx
import uvicorn
import websockets
from websockets.exceptions import WebSocketException

from adapters.delivery_receipts import post_delivery_receipt
from adapters.envelope_common import (
    normalize_mention_display_map,
    semantic_entity_fallback_label,
)
from adapters.outbound_sequence import followup_delay_seconds
from kazusa_ai_chatbot.dispatcher import SendResult
from kazusa_ai_chatbot.logging_config import configure_adapter_logging
from kazusa_ai_chatbot.message_envelope import (
    PassthroughMentionResolver,
    build_default_attachment_handler_registry,
)
from kazusa_ai_chatbot.utils import log_list_preview, log_preview

from .attachments import fetch_image_attachments
from .envelope_normalizer import QQEnvelopeNormalizer
from .inbound_segments import normalize_inbound_wire_message
from .mention_hydration import (
    cache_qq_mention_display_name,
    hydrate_mention_display_names,
    lookup_qq_mention_display_name,
    select_qq_display_name,
)
from .outbound import outbound_message_payload
from .reply_hydration import (
    apply_replied_message_metadata,
    hydrate_reply_context_from_platform,
)
from .runtime_api import bind_runtime_adapter, current_runtime_adapter, runtime_app


configure_adapter_logging()
logger = logging.getLogger(__name__)


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
        self.ws_url: str = ws_url
        self.ws_token = ws_token
        self.brain_url = brain_url.rstrip("/")
        self.channel_ids = set(channel_ids) if channel_ids is not None else None
        self.runtime_host = runtime_host
        self.runtime_port = runtime_port
        self.runtime_public_url = runtime_public_url.rstrip("/")
        self.runtime_shared_secret = runtime_shared_secret
        self.heartbeat_seconds = heartbeat_seconds

        self.bot_id: Optional[str] = None
        self.bot_name: Optional[str] = None
        self._ws = None
        self._api_response_futures: dict[str, asyncio.Future] = {}
        self._api_dispatch_enabled = False
        self._runtime_server: uvicorn.Server | None = None
        self._runtime_server_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._normal_chat_delivery_tasks: set[asyncio.Task] = set()
        self._mention_display_cache: OrderedDict[tuple[str, str], str] = (
            OrderedDict()
        )

        self.debug_modes = debug_modes or {}
        self._envelope_normalizer = QQEnvelopeNormalizer()
        self._attachment_handlers = build_default_attachment_handler_registry()
        self._mention_resolver = PassthroughMentionResolver()
        self.brain_client = httpx.AsyncClient(
            base_url=self.brain_url,
            timeout=brain_response_timeout,
        )

    def _outbound_channel_allowed(
        self,
        channel_id: str,
        *,
        channel_type: str,
    ) -> bool:
        """Return whether this adapter may visibly send to the target."""

        if channel_type == "private":
            return_value = True
            return return_value
        if channel_type != "group":
            return_value = False
            return return_value
        if self.channel_ids is None:
            return_value = False
            return return_value
        return_value = str(channel_id) in self.channel_ids
        return return_value

    async def connect(self):
        """Connect to NapCat and keep processing incoming websocket events."""

        headers = {"Authorization": f"Bearer {self.ws_token}"} if self.ws_token else {}

        while True:
            try:
                logger.info(f"Connecting to NapCat at {self.ws_url}...")
                async with websockets.connect(
                    self.ws_url,
                    additional_headers=headers,
                ) as ws:
                    self._ws = ws
                    await self._fetch_bot_info(ws)
                    await self._ensure_runtime_server_started()
                    await self._register_with_brain()
                    self._ensure_heartbeat_started()
                    logger.info(f"Logged in as {self.bot_name} (ID: {self.bot_id})")
                    if self.channel_ids is not None:
                        logger.info(
                            f"Active in groups: {self.channel_ids}. "
                            "Other groups are listen-only."
                        )
                    else:
                        logger.info(
                            "No active groups configured; all groups are "
                            "listen-only (Private chats are active)."
                        )

                    self._api_dispatch_enabled = True
                    while True:
                        message = await ws.recv()
                        data = json.loads(message)

                        if self._resolve_api_response(data):
                            continue

                        asyncio.create_task(self.handle_event(data, ws))
            except Exception as exc:
                logger.exception(f"Connection lost. Retrying in 5s: {exc}")
                self._api_dispatch_enabled = False
                self._reject_pending_api_responses(exc)
                self._ws = None
                await asyncio.sleep(5)

    async def _ensure_runtime_server_started(self) -> None:
        """Start the adapter callback server exactly once for brain-side sends."""

        bind_runtime_adapter(self)
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
        logger.info(
            f"NapCat runtime callback listening on "
            f"{self.runtime_host}:{self.runtime_port}"
        )

    async def _register_with_brain(self) -> None:
        """Register this adapter's outbound callback URL with the brain service."""

        payload = self._runtime_registration_payload()
        response = await self.brain_client.post(
            f"{self.brain_url}/runtime/adapters/register",
            json=payload,
        )
        response.raise_for_status()
        logger.info(
            f"Registered NapCat runtime adapter with brain: "
            f"callback_url={self.runtime_public_url}"
        )

    def _runtime_registration_payload(self) -> dict:
        """Return the shared registration payload for startup and heartbeat."""

        return_value = {
            "platform": self.platform,
            "callback_url": self.runtime_public_url,
            "shared_secret": self.runtime_shared_secret,
            "timeout_seconds": 10.0,
        }
        if self.bot_id:
            return_value["platform_bot_id"] = self.bot_id
        if self.bot_name:
            return_value["display_name"] = self.bot_name
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
                logger.warning(f"NapCat runtime heartbeat failed: {exc}")

    async def _fetch_bot_info(self, ws):
        """Retrieve the bot's QQ identity from NapCat."""

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
        """Resolve a pending websocket API call from an echo response."""

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
        """Fail all pending websocket API calls after the connection drops."""

        pending = list(self._api_response_futures.values())
        self._api_response_futures.clear()
        for future in pending:
            if not future.done():
                future.set_exception(exc)

    async def _call_api(self, ws, action: str, params: dict | None = None):
        """Call a OneBot API action and wait for the matching echo response."""

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

    async def _post_delivery_receipt(
        self,
        *,
        channel_id: str,
        delivery_tracking_id: str,
        logical_message_index: int,
        platform_message_id: str,
    ) -> None:
        """Best-effort report of a delivered normal chat response."""

        await post_delivery_receipt(
            http_client=self.brain_client,
            brain_url=self.brain_url,
            platform="qq",
            platform_channel_id=channel_id,
            delivery_tracking_id=delivery_tracking_id,
            logical_message_index=logical_message_index,
            platform_message_id=platform_message_id,
            adapter="napcat",
            logger=logger,
            log_label="QQ",
        )

    def _track_normal_chat_delivery_task(self, task: asyncio.Task) -> None:
        """Track adapter-owned follow-up sends for cancellation and tests."""

        self._normal_chat_delivery_tasks.add(task)
        task.add_done_callback(self._finalize_normal_chat_delivery_task)

    def _finalize_normal_chat_delivery_task(self, task: asyncio.Task) -> None:
        """Discard completed follow-up tasks and log unexpected failures."""

        self._normal_chat_delivery_tasks.discard(task)
        if task.cancelled():
            return
        task_exception = task.exception()
        if task_exception is None:
            return
        logger.warning(
            f"QQ follow-up delivery task failed unexpectedly: {task_exception}",
            exc_info=(
                type(task_exception),
                task_exception,
                task_exception.__traceback__,
            ),
        )

    def _normal_chat_send_msg_params(
        self,
        *,
        channel_id: str,
        channel_type: str,
        text: str,
        reply_to_msg_id: str | None,
        delivery_mentions: Sequence[dict] | None,
    ) -> dict[str, object]:
        """Build a NapCat ``send_msg`` payload for one normal chat message."""

        is_group = channel_type == "group"
        params: dict[str, object] = {
            "message_type": channel_type,
            "group_id" if is_group else "user_id": int(channel_id),
            "message": outbound_message_payload(
                text,
                reply_to_msg_id,
                delivery_mentions if is_group else None,
            ),
        }
        return params

    async def _send_normal_chat_followups(
        self,
        *,
        messages: Sequence[str],
        channel_id: str,
        channel_type: str,
        delivery_tracking_id: str,
        delivery_mentions: Sequence[dict] | None,
        ws: object,
    ) -> None:
        """Send delayed follow-up messages from one brain cognition."""

        mention_candidates = (
            delivery_mentions
            if channel_type == "group"
            else None
        )
        for logical_message_index, message_text in enumerate(messages, start=1):
            await asyncio.sleep(followup_delay_seconds(message_text))
            msg_params = self._normal_chat_send_msg_params(
                channel_id=channel_id,
                channel_type=channel_type,
                text=message_text,
                reply_to_msg_id=None,
                delivery_mentions=mention_candidates,
            )
            try:
                send_response = await self._call_api(ws, "send_msg", msg_params)
            except (
                asyncio.TimeoutError,
                WebSocketException,
            ) as exc:
                logger.warning(
                    f"QQ follow-up send_msg failed: "
                    f"channel_id={channel_id} error={exc}"
                )
                return

            send_status = send_response.get("status")
            if send_status != "ok":
                retcode = send_response.get("retcode")
                message = send_response.get("message")
                logger.warning(
                    f"QQ follow-up send_msg returned status={send_status} "
                    f"retcode={retcode} message={log_preview(message)}"
                )
                return

            response_data = send_response.get("data") or {}
            platform_message_id = ""
            if isinstance(response_data, dict):
                platform_message_id = str(response_data.get("message_id") or "")
            await self._post_delivery_receipt(
                channel_id=channel_id,
                delivery_tracking_id=delivery_tracking_id,
                logical_message_index=logical_message_index,
                platform_message_id=platform_message_id,
            )

    def _apply_replied_message_metadata(
        self,
        reply_context: dict[str, Any],
        message_data: dict,
    ) -> None:
        """Populate reply target fields from a NapCat message document."""

        apply_replied_message_metadata(
            reply_context,
            message_data,
            bot_id=str(self.bot_id or ""),
        )

    async def _hydrate_reply_context_from_platform(
        self,
        reply_context: dict[str, Any],
        ws,
    ) -> None:
        """Resolve reply target metadata from NapCat before brain forwarding."""

        await hydrate_reply_context_from_platform(
            reply_context,
            ws,
            call_api=self._call_api,
            bot_id=str(self.bot_id or ""),
            logger=logger,
        )

    def _select_qq_display_name(self, source: dict) -> str:
        """Choose the QQ label source that matches inbound sender naming."""

        display_name = select_qq_display_name(source)
        return display_name

    async def _lookup_qq_mention_display_name(
        self,
        *,
        platform_user_id: str,
        is_group: bool,
        group_id: object,
        ws: object,
    ) -> str:
        """Resolve one QQ mention label through a bounded NapCat lookup."""

        display_name = await lookup_qq_mention_display_name(
            platform_user_id=platform_user_id,
            is_group=is_group,
            group_id=group_id,
            ws=ws,
            call_api=self._call_api,
            logger=logger,
        )
        return display_name

    def _cache_qq_mention_display_name(
        self,
        cache_key: tuple[str, str],
        display_name: str,
    ) -> None:
        """Remember a positive QQ mention label with bounded LRU retention."""

        cache_qq_mention_display_name(
            self._mention_display_cache,
            cache_key,
            display_name,
        )

    async def _hydrate_mention_display_names(
        self,
        *,
        raw_wire_text: str,
        initial_display_names: dict[str, str],
        channel_id: str,
        is_group: bool,
        group_id: object,
        ws: object,
    ) -> dict[str, str]:
        """Hydrate QQ mention labels before the envelope reaches the brain."""

        display_names = await hydrate_mention_display_names(
            raw_wire_text=raw_wire_text,
            initial_display_names=initial_display_names,
            channel_id=channel_id,
            is_group=is_group,
            group_id=group_id,
            ws=ws,
            bot_id=self.bot_id,
            bot_name=self.bot_name,
            mention_display_cache=self._mention_display_cache,
            call_api=self._call_api,
            logger=logger,
        )
        return display_names

    async def _hydrate_reply_display_names(
        self,
        *,
        reply_context: dict[str, Any],
        mention_display_names: dict[str, str],
        channel_id: str,
        is_group: bool,
        group_id: object,
        ws: object,
    ) -> dict[str, str]:
        """Hydrate mention labels that appear only in replied-message text."""

        reply_excerpt = reply_context.get("reply_excerpt")
        target_user_id = str(
            reply_context.get("reply_to_platform_user_id") or "",
        ).strip()
        target_display_name = reply_context.get("reply_to_display_name")
        raw_wire_parts: list[str] = []
        if isinstance(reply_excerpt, str) and reply_excerpt.strip():
            raw_wire_parts.append(reply_excerpt)
        if target_user_id and not target_display_name:
            raw_wire_parts.append(f"[CQ:at,qq={target_user_id}]")

        raw_reply_display_names = reply_context.pop(
            "reply_mention_display_names",
            {},
        )
        reply_display_names = normalize_mention_display_map(
            raw_reply_display_names,
        )
        initial_display_names = dict(reply_display_names)
        initial_display_names.update(mention_display_names)
        if not raw_wire_parts:
            return_value = initial_display_names
            return return_value

        hydrated_display_names = await self._hydrate_mention_display_names(
            raw_wire_text=" ".join(raw_wire_parts),
            initial_display_names=initial_display_names,
            channel_id=channel_id,
            is_group=is_group,
            group_id=group_id,
            ws=ws,
        )
        hydrated_target_name = hydrated_display_names.get(target_user_id, "")
        if hydrated_target_name:
            reply_context["reply_to_display_name"] = hydrated_target_name

        return_value = hydrated_display_names
        return return_value

    async def handle_event(self, data: dict, ws):
        """Normalize one NapCat event and forward message events to the brain."""

        if data.get("post_type") != "message" or not self.bot_id:
            return

        user_id = str(data.get("user_id"))
        message_id = str(data.get("message_id", ""))
        group_id = data.get("group_id")
        is_group = data.get("message_type") == "group"
        channel_id = str(group_id) if is_group else user_id

        message_data = data.get("message", [])
        wire_content, reply_context, mention_display_names = (
            normalize_inbound_wire_message(message_data)
        )
        wire_content = wire_content.strip()
        await self._hydrate_reply_context_from_platform(reply_context, ws)
        mention_display_names = await self._hydrate_mention_display_names(
            raw_wire_text=wire_content,
            initial_display_names=mention_display_names,
            channel_id=channel_id,
            is_group=is_group,
            group_id=group_id,
            ws=ws,
        )
        mention_display_names = await self._hydrate_reply_display_names(
            reply_context=reply_context,
            mention_display_names=mention_display_names,
            channel_id=channel_id,
            is_group=is_group,
            group_id=group_id,
            ws=ws,
        )
        sender = data.get("sender", {})
        if not isinstance(sender, dict):
            sender = {}
        sender_name = self._select_qq_display_name(sender)
        if not sender_name:
            sender_name = semantic_entity_fallback_label(
                entity_kind="user",
                mention_context=False,
            )

        message_debug_modes = dict(self.debug_modes)
        is_active = (
            (not is_group)
            or (self.channel_ids is not None and str(group_id) in self.channel_ids)
        )

        if not is_active:
            message_debug_modes["listen_only"] = True

        mode_label = "LISTEN-ONLY" if message_debug_modes.get("listen_only") else "ACTIVE"

        logger.info(
            f"[{mode_label}] Incoming QQ message: channel_id={channel_id} "
            f"is_group={is_group} sender={sender_name}"
        )
        logger.debug(
            f"Incoming QQ message detail: channel_id={channel_id} "
            f"raw_wire={log_preview(wire_content)}"
        )

        attachments = await fetch_image_attachments(
            message_data=message_data,
            http_client=self.brain_client,
            logger=logger,
        )

        envelope_request = SimpleNamespace(
            platform="qq",
            channel_type="group" if is_group else "private",
            content=wire_content,
            platform_bot_id=self.bot_id,
            reply_context=reply_context,
            mention_display_names=mention_display_names,
            attachments=attachments,
        )
        envelope = self._envelope_normalizer.normalize(
            envelope_request,
            self._mention_resolver,
            self._attachment_handlers,
        )
        raw_content = envelope["body_text"]

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

        logger.debug(
            f"Forwarding to brain: channel_id={channel_id} user_id={user_id} "
            f"attachments={len(attachments)} content={log_preview(raw_content)}"
        )

        try:
            response = await self.brain_client.post(
                f"{self.brain_url}/chat",
                json=payload,
            )
            response.raise_for_status()
            brain_data = response.json()
        except Exception as exc:
            logger.exception(f"Brain service failed: {exc}")
            return

        replies = brain_data.get("messages", [])
        content_type = str(brain_data.get("content_type") or "text")
        operational_error = brain_data.get("operational_error")
        is_operational_response = content_type == "operational_error"
        if is_operational_response:
            if isinstance(operational_error, dict):
                logger.error(
                    "Brain returned a structured QQ operational response: "
                    f"error_code={operational_error.get('error_code')} "
                    f"status={operational_error.get('status')} "
                    f"attempt_count={operational_error.get('attempt_count')}"
                )
            else:
                logger.error(
                    "Brain returned an operational response without metadata"
                )
        logger.info(f"Brain output: messages={log_list_preview(replies)}")
        logger.debug(
            f'Brain output metadata: use_reply_feature={brain_data.get("use_reply_feature")} '
            f"message_count={len(replies)}"
        )
        if not replies:
            return

        channel_type = "group" if is_group else "private"
        if not self._outbound_channel_allowed(channel_id, channel_type=channel_type):
            logger.warning(
                "Suppressing QQ response for disallowed target: "
                f"channel_type={channel_type} channel_id={channel_id}"
            )
            return

        reply_to_msg_id = None
        if not is_operational_response and brain_data.get("use_reply_feature"):
            reply_to_msg_id = str(data["message_id"])
        raw_delivery_mentions = brain_data.get("delivery_mentions")
        delivery_mentions = (
            raw_delivery_mentions
            if isinstance(raw_delivery_mentions, list)
            else None
        )
        if is_operational_response:
            delivery_mentions = None
        msg_params = self._normal_chat_send_msg_params(
            channel_id=channel_id,
            channel_type=channel_type,
            text=replies[0],
            reply_to_msg_id=reply_to_msg_id,
            delivery_mentions=delivery_mentions if is_group else None,
        )

        logger.debug(
            f"Sending QQ message: channel_id={channel_id} "
            f'message={log_preview(msg_params["message"])}'
        )

        try:
            send_response = await self._call_api(ws, "send_msg", msg_params)
        except (
            asyncio.TimeoutError,
            WebSocketException,
        ) as exc:
            logger.warning(f"QQ send_msg failed: channel_id={channel_id} error={exc}")
            return

        send_status = send_response.get("status")
        if send_status != "ok":
            retcode = send_response.get("retcode")
            message = send_response.get("message")
            logger.warning(
                f"QQ send_msg returned status={send_status} "
                f"retcode={retcode} message={log_preview(message)}"
            )
            return

        response_data = send_response.get("data") or {}
        platform_message_id = ""
        if isinstance(response_data, dict):
            platform_message_id = str(response_data.get("message_id") or "")
        delivery_tracking_id = str(brain_data.get("delivery_tracking_id") or "")
        if not is_operational_response:
            await self._post_delivery_receipt(
                channel_id=channel_id,
                delivery_tracking_id=delivery_tracking_id,
                logical_message_index=0,
                platform_message_id=platform_message_id,
            )

        followup_messages = replies[1:]
        if followup_messages and not is_operational_response:
            followup_task = asyncio.create_task(
                self._send_normal_chat_followups(
                    messages=followup_messages,
                    channel_id=channel_id,
                    channel_type=channel_type,
                    delivery_tracking_id=delivery_tracking_id,
                    delivery_mentions=delivery_mentions,
                    ws=ws,
                )
            )
            self._track_normal_chat_delivery_task(followup_task)

    async def close(self):
        """Close adapter-owned network clients and callback server."""

        normal_chat_delivery_tasks = list(self._normal_chat_delivery_tasks)
        for task in normal_chat_delivery_tasks:
            task.cancel()
        if normal_chat_delivery_tasks:
            await asyncio.gather(
                *normal_chat_delivery_tasks,
                return_exceptions=True,
            )
            self._normal_chat_delivery_tasks.clear()

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError as exc:
                logger.debug(f"Heartbeat task cancelled during close: {exc}")
                pass
        await self.brain_client.aclose()
        if self._runtime_server is not None:
            self._runtime_server.should_exit = True
        if self._runtime_server_task is not None:
            await self._runtime_server_task
        if current_runtime_adapter() is self:
            bind_runtime_adapter(None)

    async def can_send_message(
        self,
        channel_id: str,
        *,
        channel_type: str,
    ) -> bool:
        """Return whether NapCat can accept one outbound target."""

        if self._ws is None:
            return_value = False
            return return_value
        if not self._outbound_channel_allowed(
            channel_id,
            channel_type=channel_type,
        ):
            return_value = False
            return return_value
        try:
            int(channel_id)
        except ValueError:
            return_value = False
            return return_value
        return_value = True
        return return_value

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
        delivery_mentions: Sequence[dict] | None = None,
    ) -> SendResult:
        """Send one outbound message through NapCat's ``send_msg`` action."""

        if channel_type not in {"private", "group"}:
            raise RuntimeError(f"Unsupported QQ channel_type: {channel_type}")
        if not self._outbound_channel_allowed(
            channel_id,
            channel_type=channel_type,
        ):
            raise RuntimeError(
                "NapCat target channel is not allowed: "
                f"channel_type={channel_type} channel_id={channel_id}"
            )
        if self._ws is None:
            raise RuntimeError("NapCat websocket is not connected")

        params: dict[str, object] = {
            "message": outbound_message_payload(
                text,
                reply_to_msg_id,
                delivery_mentions if channel_type == "group" else None,
            ),
        }
        if channel_type == "private":
            params["message_type"] = "private"
            params["user_id"] = int(channel_id)
        elif channel_type == "group":
            params["message_type"] = "group"
            params["group_id"] = int(channel_id)

        try:
            response = await self._call_api(self._ws, "send_msg", params)
        except (
            asyncio.TimeoutError,
            WebSocketException,
        ) as exc:
            raise RuntimeError(f"NapCat send_msg failed: {exc}") from exc

        status = response.get("status")
        if status != "ok":
            retcode = response.get("retcode")
            message = response.get("message")
            raise RuntimeError(
                f"NapCat send_msg failed: status={status} "
                f"retcode={retcode} message={message}"
            )

        response_data = response.get("data") or {}
        message_id = ""
        if isinstance(response_data, dict):
            message_id = str(response_data.get("message_id") or "")
        return_value = SendResult(
            platform=self.platform,
            channel_id=channel_id,
            message_id=message_id,
            sent_at=datetime.now(timezone.utc),
        )
        return return_value
