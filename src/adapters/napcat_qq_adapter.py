import os
import argparse
import asyncio
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
import json
import base64
import logging
import httpx
import uvicorn
import websockets
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import re

from kazusa_ai_chatbot.dispatcher import SendResult
from kazusa_ai_chatbot.logging_config import configure_adapter_logging

configure_adapter_logging()

logger = logging.getLogger(__name__)

runtime_app = FastAPI(title="Kazusa NapCat Runtime Adapter")
_runtime_adapter: "NapCatWSAdapter | None" = None


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

    return RuntimeSendMessageResponse(
        platform=result.platform,
        channel_id=result.channel_id,
        message_id=result.message_id,
        sent_at=result.sent_at.isoformat(),
    )


class NapCatWSAdapter:
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
        self._runtime_server: uvicorn.Server | None = None
        self._runtime_server_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None

        self.debug_modes = debug_modes or {}

        # Initialize connection
        self.brain_client = httpx.AsyncClient(base_url=self.brain_url, timeout=brain_response_timeout)

    async def connect(self):
        """Connects to NapCat and synchronizes bot identity."""
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
                        logger.info("Active in groups: %s. Other groups are listen-only.", self.channel_ids)
                    else:
                        logger.info("No active groups configured — all groups are listen-only (Private chats are active).")
                    
                    # 2. Main Event Loop
                    while True:
                        message = await ws.recv()
                        data = json.loads(message)

                        # Handle Events (messages, etc.)
                        asyncio.create_task(self.handle_event(data, ws))
            except Exception as e:
                logger.error(f"Connection lost: {e}. Retrying in 5s...")
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
        logger.info(
            "NapCat runtime callback listening on %s:%s",
            self.runtime_host,
            self.runtime_port,
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
            "Registered NapCat runtime adapter with brain: callback_url=%s",
            self.runtime_public_url,
        )

    def _runtime_registration_payload(self) -> dict:
        """Return the shared registration payload for startup and heartbeat."""

        return {
            "platform": self.platform,
            "callback_url": self.runtime_public_url,
            "shared_secret": self.runtime_shared_secret,
            "timeout_seconds": 10.0,
        }

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
                logger.warning("NapCat runtime heartbeat failed: %s", exc)

    async def _fetch_bot_info(self, ws):
        """Calls get_login_info to retrieve the bot's QQ ID and Nickname."""
        response = await self._call_api(ws, "get_login_info")
        if response.get("status") == "ok":
            data = response.get("data", {})
            self.bot_id = str(data.get("user_id"))
            self.bot_name = data.get("nickname")
        else:
            logger.warning("Could not retrieve bot info, using defaults.")
            self.bot_id = "unknown"
            self.bot_name = "NapCat Bot"

    async def _call_api(self, ws, action: str, params: dict = None):
        """Helper to call OneBot APIs and wait for the specific response."""
        echo_id = f"sync_{action}_{asyncio.get_event_loop().time()}"
        payload = {
            "action": action,
            "params": params or {},
            "echo": echo_id
        }

        await ws.send(json.dumps(payload))

        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=10.0)
            response = json.loads(message)
            if response.get("echo") == echo_id:
                return response

    def _is_bot_mentioned(self, mentioned_ids: list[str]) -> bool:
        """Return whether native message metadata mentioned this bot.

        Args:
            mentioned_ids: Platform user IDs extracted from native mention
                segments or CQ at codes by the adapter.

        Returns:
            True when this adapter's bot id is present.
        """
        return bool(self.bot_id and self.bot_id in mentioned_ids)

    async def handle_event(self, data: dict, ws):
        """Processes incoming messages (only if we are identified)."""
        if data.get("post_type") != "message" or not self.bot_id:
            return

        user_id = str(data.get("user_id"))
        message_id = str(data.get("message_id", ""))
        group_id = data.get("group_id")
        
        message_data = data.get("message", [])
        reply_context: dict[str, str | bool] = {}
        mentioned_ids: list[str] = []

        # Preprocess QQ message to a format that is recognized by the brain
        if isinstance(message_data, str):
            # If it's a raw string, normalize CQ codes
            reply_match = re.search(r'\[CQ:reply,id=([^\]]+)\]', message_data)
            if reply_match is not None:
                reply_context["reply_to_message_id"] = reply_match.group(1)
            mentioned_ids = re.findall(r'\[CQ:at,qq=([^\]]+)\]', message_data)
            raw_content = re.sub(r'\[CQ:at,qq=([^\]]+)\]', r'<@\1>', message_data)
            raw_content = re.sub(r'\[CQ:reply,id=([^\]]+)\]', r'[Reply to message] ', raw_content)
            raw_content = re.sub(r'\[CQ:[^\]]+\]', r'', raw_content) # Strip others
        else:
            # If it's a segment list, rebuild it to text
            raw_content = ""
            for seg in message_data:
                if not isinstance(seg, dict):
                    continue
                seg_type = seg.get("type")
                seg_data = seg.get("data", {})
                if seg_type == "text":
                    raw_content += seg_data.get("text", "")
                elif seg_type == "at":
                    qq = seg_data.get("qq")
                    if qq is not None:
                        mentioned_ids.append(str(qq))
                    raw_content += f"<@{qq}> "
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
                    raw_content += f"[Reply to message] "
                elif seg_type == "face":
                    raw_content += f"[Face] "
                # image/video etc. are handled by attachments array, so we omit them from raw text
        
        raw_content = raw_content.strip()
        mentioned_bot = self._is_bot_mentioned(mentioned_ids)
        sender_name = data.get("sender", {}).get("nickname", f"User {user_id}")
        
        is_group = data.get("message_type") == "group"
        channel_id = str(group_id) if is_group else user_id

        message_debug_modes = dict(self.debug_modes)
        is_active = (not is_group) or (self.channel_ids is not None and str(group_id) in self.channel_ids)
        
        if not is_active:
            message_debug_modes["listen_only"] = True
            
        mode_label = "LISTEN-ONLY" if message_debug_modes.get("listen_only") else "ACTIVE"

        logger.info(
            "[%s] Incoming QQ message: channel_id=%s is_group=%s sender=%s content=%s",
            mode_label,
            channel_id,
            is_group,
            sender_name,
            raw_content,
        )

        # Attachments processing (same as before)
        attachments = []
        for seg in data.get("message", []):
            if isinstance(seg, dict) and seg.get("type") == "image":
                url = seg["data"].get("url")
                if url:
                    try:
                        resp = await self.brain_client.get(url, timeout=10.0)
                        attachments.append({
                            "media_type": "image/jpeg",
                            "base64_data": base64.b64encode(resp.content).decode("utf-8"),
                            "description": ""
                        })
                    except Exception as e:
                        logger.error(f"Image fetch error: {e}")

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
            "content": raw_content,
            "content_type": "text",
            "mentioned_bot": mentioned_bot,
            "attachments": attachments,
            "reply_context": reply_context,
            "debug_modes": message_debug_modes,
        }

        logger.info(
            "Forwarding to brain: channel_id=%s user_id=%s content=%s attachments=%s",
            channel_id,
            user_id,
            raw_content,
            len(attachments),
        )

        try:
            resp = await self.brain_client.post(f"{self.brain_url}/chat", json=payload)
            resp.raise_for_status()
            brain_data = resp.json()
        except Exception:
            logger.exception("Brain service failed")
            return

        # Response handling
        replies = brain_data.get("messages", [])
        logger.info(
            "Brain output: should_reply=%s message_count=%s messages=%s",
            brain_data.get("should_reply"),
            len(replies),
            replies,
        )
        if replies:
            combined = "\n".join(replies)
            msg_params = {
                "message_type": "group" if is_group else "private",
                "group_id" if is_group else "user_id": int(channel_id),
                "message": combined
            }
            if brain_data.get("should_reply"):
                msg_params["message"] = f"[CQ:reply,id={data['message_id']}]" + combined

            logger.info(
                "Sending QQ message: channel_id=%s message=%s",
                channel_id,
                msg_params["message"],
            )
            
            # We don't need to wait for 'send_msg' response usually
            await ws.send(json.dumps({"action": "send_msg", "params": msg_params}))

    async def close(self):
        """Close adapter-owned network clients and callback server."""

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
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
        return SendResult(
            platform=self.platform,
            channel_id=channel_id,
            message_id="",
            sent_at=datetime.now(timezone.utc),
        )



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
