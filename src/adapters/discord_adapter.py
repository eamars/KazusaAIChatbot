"""Thin Discord adapter — translates Discord events into brain service HTTP calls.

Usage:
    python -m adapters.discord_adapter \
        --brain-url http://localhost:8000 \
        --channels 12345 67890 \
        --no-listen-all
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import logging
import sys
from pathlib import Path
import os
import discord
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from kazusa_ai_chatbot.dispatcher import SendResult
from kazusa_ai_chatbot.logging_config import configure_adapter_logging

configure_adapter_logging()

logger = logging.getLogger(__name__)

BRAIN_URL: str = "http://localhost:8000"
runtime_app = FastAPI(title="Kazusa Discord Runtime Adapter")
_runtime_adapter: "DiscordAdapter | None" = None


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
    """Deliver one scheduled outbound message through the live Discord adapter."""

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


class DiscordAdapter(discord.Client):
    """Discord client that forwards messages to the Kazusa brain service."""

    platform = "discord"

    def __init__(
        self,
        brain_url: str,
        runtime_host: str,
        runtime_port: int,
        runtime_public_url: str,
        runtime_shared_secret: str = "",
        heartbeat_seconds: float = 30.0,
        channel_ids: list[str] | None = None,
        debug_modes: dict | None = None,
        **kwargs,
    ):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)

        self.brain_url = brain_url.rstrip("/")
        self.runtime_host = runtime_host
        self.runtime_port = runtime_port
        self.runtime_public_url = runtime_public_url.rstrip("/")
        self.runtime_shared_secret = runtime_shared_secret
        self.heartbeat_seconds = heartbeat_seconds
        self.channel_ids = set(channel_ids) if channel_ids is not None else None
        self.debug_modes = debug_modes or {}
        self._http_client = httpx.AsyncClient(timeout=120.0)
        self._runtime_server: uvicorn.Server | None = None
        self._runtime_server_task: asyncio.Task | None = None
        self._brain_registration_done = False
        self._heartbeat_task: asyncio.Task | None = None

    async def setup_hook(self) -> None:
        """Start the runtime callback server before Discord events begin."""

        await self._ensure_runtime_server_started()

    async def on_ready(self):
        logger.info("Discord adapter logged in as %s (id=%s)", self.user, self.user.id)
        if not self._brain_registration_done:
            await self._register_with_brain()
            self._brain_registration_done = True
            self._ensure_heartbeat_started()
        if self.channel_ids is not None:
            logger.info("Active in channels: %s. Other channels are listen-only.", self.channel_ids)
        else:
            logger.info("No active channels configured — all channels are listen-only (DMs are active).")

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
            "Discord runtime callback listening on %s:%s",
            self.runtime_host,
            self.runtime_port,
        )

    async def _register_with_brain(self) -> None:
        """Register this adapter's outbound callback URL with the brain service."""

        payload = self._runtime_registration_payload()
        response = await self._http_client.post(
            f"{self.brain_url}/runtime/adapters/register",
            json=payload,
        )
        response.raise_for_status()
        logger.info(
            "Registered Discord runtime adapter with brain: callback_url=%s",
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

        response = await self._http_client.post(
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
                logger.warning("Discord runtime heartbeat failed: %s", exc)

    async def on_message(self, message: discord.Message):
        if message.author == self.user:
            return

        is_dm = message.guild is None
        channel_id_str = str(message.channel.id)
        reply_context: dict[str, str | bool] = {}

        if message.reference is not None and message.reference.message_id is not None:
            reply_context["reply_to_message_id"] = str(message.reference.message_id)

        referenced_message = message.reference.resolved if message.reference is not None else None
        if isinstance(referenced_message, discord.Message):
            reply_context["reply_to_platform_user_id"] = str(referenced_message.author.id)
            reply_context["reply_to_display_name"] = referenced_message.author.display_name
            reply_context["reply_to_current_bot"] = bool(self.user and referenced_message.author.id == self.user.id)
            reply_context["reply_excerpt"] = referenced_message.content

        message_debug_modes = dict(self.debug_modes)
        is_active = is_dm or (self.channel_ids is not None and channel_id_str in self.channel_ids)
        
        if not is_active:
            message_debug_modes["listen_only"] = True
            
        mode_label = "LISTEN-ONLY" if message_debug_modes.get("listen_only") else "ACTIVE"

        logger.info(
            "[%s] Incoming Discord message: channel_id=%s is_dm=%s author=%s content=%s",
            mode_label,
            channel_id_str,
            is_dm,
            message.author.display_name,
            message.content,
        )

        # Build attachments
        attachments = []
        for att in message.attachments:
            if att.size and att.size > (5 * 1024 * 1024):
                logger.warning("Attachment %s exceeds 5MB — skipping", att.filename)
                continue

            if att.content_type and att.content_type.startswith("image/"):
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(att.url)
                        resp.raise_for_status()
                        b64 = base64.b64encode(resp.content).decode("utf-8")
                        attachments.append({
                            "media_type": att.content_type,
                            "base64_data": b64,
                            "description": "",
                        })
                except httpx.HTTPError:
                    logger.exception("Failed to fetch attachment %s", att.url)

        # Determine channel name
        if message.guild is None:
            channel_name = f"Private chat with {message.author.display_name}"
        else:
            channel_name = str(message.channel.name)

        # Build the request payload
        payload = {
            "platform": "discord",
            "platform_channel_id": str(message.channel.id),
            "channel_type": "private" if is_dm else "group",
            "platform_message_id": str(message.id),
            "platform_user_id": str(message.author.id),
            "platform_bot_id": str(self.user.id) if self.user else "",
            "display_name": message.author.display_name,
            "channel_name": channel_name,
            "content": message.content,
            "content_type": "text",
            "attachments": attachments,
            "reply_context": reply_context,
            "debug_modes": message_debug_modes,
        }

        try:
            # Only show typing indicator if the bot is actually going to think and reply
            if not message_debug_modes.get("listen_only"):
                async with message.channel.typing():
                    resp = await self._http_client.post(
                        f"{self.brain_url}/chat",
                        json=payload,
                    )
            else:
                resp = await self._http_client.post(
                    f"{self.brain_url}/chat",
                    json=payload,
                )
            
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("Brain service request failed")
            if not message_debug_modes.get("listen_only"):
                await message.reply("I'm having trouble thinking right now, please try again later.")
            return

        # Send response messages
        messages = data.get("messages", [])
        if not messages:
            return

        use_reply = data.get("should_reply", False)
        combined = "\n".join(messages)

        for chunk in _split_message(combined):
            if use_reply:
                await message.reply(chunk)
                use_reply = False
            else:
                await message.channel.send(chunk)

        # Handle multimedia attachments from the response
        for att in data.get("attachments", []):
            # TODO: send images/stickers when the brain supports it
            pass

    async def close(self):
        """Close adapter-owned HTTP clients and callback server."""

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        await self._http_client.aclose()
        if self._runtime_server is not None:
            self._runtime_server.should_exit = True
        if self._runtime_server_task is not None:
            await self._runtime_server_task
        await super().close()

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to_msg_id: str | None = None,
    ) -> SendResult:
        """Send one outbound message through the Discord client.

        Args:
            channel_id: Discord channel identifier.
            text: Message body to send.
            reply_to_msg_id: Optional message id to reply to.

        Returns:
            Structured send result for the dispatcher.
        """

        channel = self.get_channel(int(channel_id))
        if channel is None:
            channel = await self.fetch_channel(int(channel_id))

        send_kwargs = {}
        if reply_to_msg_id:
            send_kwargs["reference"] = discord.PartialMessage(
                channel=channel,
                id=int(reply_to_msg_id),
            )

        message = await channel.send(text, **send_kwargs)
        return SendResult(
            platform=self.platform,
            channel_id=channel_id,
            message_id=str(message.id),
            sent_at=datetime.now(timezone.utc),
        )


def _split_message(text: str, limit: int = 2000) -> list[str]:
    """Split a message into chunks that fit within Discord's character limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = text.rfind(" ", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return chunks


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Discord adapter for Kazusa Brain Service")
    parser.add_argument("--channels", type=str, nargs="*", default=None, help="Discord channel IDs to actively participate in. Other channels will be listen-only.")
    parser.add_argument("--runtime-host", type=str, default=os.getenv("ADAPTER_RUNTIME_HOST", "127.0.0.1"))
    parser.add_argument("--runtime-port", type=int, default=int(os.getenv("DISCORD_RUNTIME_PORT", "8012")))
    parser.add_argument("--runtime-public-url", type=str, default=os.getenv("ADAPTER_RUNTIME_PUBLIC_URL", ""))
    parser.add_argument("--heartbeat-seconds", type=float, default=float(os.getenv("ADAPTER_HEARTBEAT_SECONDS", "30")))
    args = parser.parse_args()

    token = os.getenv("DISCORD_TOKEN", "")
    if not token:
        logger.error("DISCORD_TOKEN environment variable is required")
        sys.exit(1)

    brain_url = os.getenv("BRAIN_URL")
    if not brain_url:
        logger.error("BRAIN_URL environment variable is required")
        sys.exit(1)
    runtime_public_url = args.runtime_public_url or f"http://127.0.0.1:{args.runtime_port}"
    runtime_shared_secret = os.getenv("ADAPTER_RUNTIME_SHARED_SECRET", "")

    adapter = DiscordAdapter(
        brain_url=brain_url,
        runtime_host=args.runtime_host,
        runtime_port=args.runtime_port,
        runtime_public_url=runtime_public_url,
        runtime_shared_secret=runtime_shared_secret,
        heartbeat_seconds=args.heartbeat_seconds,
        channel_ids=args.channels,
        debug_modes={},
    )
    adapter.run(token, log_handler=None)


if __name__ == "__main__":
    main()
