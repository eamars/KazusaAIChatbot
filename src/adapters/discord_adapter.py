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
import logging
import sys
from pathlib import Path
import os
import discord
import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BRAIN_URL: str = "http://localhost:8000"


class DiscordAdapter(discord.Client):
    """Discord client that forwards messages to the Kazusa brain service."""

    def __init__(
        self,
        brain_url: str,
        channel_ids: list[str] | None = None,
        debug_modes: dict | None = None,
        **kwargs,
    ):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)

        self.brain_url = brain_url.rstrip("/")
        self.channel_ids = set(channel_ids) if channel_ids is not None else None
        self.debug_modes = debug_modes or {}
        self._http_client = httpx.AsyncClient(timeout=120.0)

    async def on_ready(self):
        logger.info("Discord adapter logged in as %s (id=%s)", self.user, self.user.id)
        if self.channel_ids is not None:
            logger.info("Active in channels: %s. Other channels are listen-only.", self.channel_ids)
        else:
            logger.info("No active channels configured — all channels are listen-only (DMs are active).")

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
            reply_context["reply_excerpt"] = referenced_message.content[:200]

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
        await self._http_client.aclose()
        await super().close()


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
    parser = argparse.ArgumentParser(description="Discord adapter for Kazusa Brain Service")
    parser.add_argument("--channels", type=str, nargs="*", default=None, help="Discord channel IDs to actively participate in. Other channels will be listen-only.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    load_dotenv()

    token = os.getenv("DISCORD_TOKEN", "")
    if not token:
        logger.error("DISCORD_TOKEN environment variable is required")
        sys.exit(1)

    brain_url = os.getenv("BRAIN_URL")
    if not brain_url:
        logger.error("BRAIN_URL environment variable is required")
        sys.exit(1)

    adapter = DiscordAdapter(
        brain_url=brain_url,
        channel_ids=args.channels,
        debug_modes={},
    )
    adapter.run(token, log_handler=None)


if __name__ == "__main__":
    main()
