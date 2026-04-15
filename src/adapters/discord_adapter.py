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
        channel_ids: list[int] | None = None,
        listen_all: bool = False,
        **kwargs,
    ):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)

        self.brain_url = brain_url.rstrip("/")
        self.channel_ids = set(channel_ids) if channel_ids else None
        self.listen_all = listen_all
        self._http_client = httpx.AsyncClient(timeout=120.0)

    async def on_ready(self):
        logger.info("Discord adapter logged in as %s (id=%s)", self.user, self.user.id)
        if self.listen_all:
            logger.info("Listening in ALL channels")
        elif self.channel_ids:
            logger.info("Listening in channels: %s", self.channel_ids)
        else:
            logger.info("No channels configured — responding to @mentions only")

    async def on_message(self, message: discord.Message):
        if message.author == self.user:
            return

        # Channel filtering
        if not self.listen_all:
            if self.channel_ids:
                if message.channel.id not in self.channel_ids:
                    return
            else:
                bot_mentioned = self.user in message.mentions if self.user else False
                if not bot_mentioned:
                    return

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
            "platform_user_id": str(message.author.id),
            "platform_bot_id": str(self.user.id) if self.user else "",
            "display_name": message.author.display_name,
            "channel_name": channel_name,
            "content": message.content,
            "content_type": "text",
            "attachments": attachments,
        }

        # Send to brain with typing indicator
        async with message.channel.typing():
            try:
                resp = await self._http_client.post(
                    f"{self.brain_url}/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                logger.exception("Brain service request failed")
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
    parser.add_argument("--brain-url", type=str, default="http://localhost:8000", help="Brain service URL")
    parser.add_argument("--channels", type=int, nargs="*", default=None, help="Discord channel IDs to listen in")
    parser.add_argument("--no-listen-all", action="store_true", default=False, help="Disable listening in all channels")
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

    listen_all = not args.no_listen_all and args.channels is None

    adapter = DiscordAdapter(
        brain_url=args.brain_url,
        channel_ids=args.channels,
        listen_all=listen_all,
    )
    adapter.run(token, log_handler=None)


if __name__ == "__main__":
    main()
