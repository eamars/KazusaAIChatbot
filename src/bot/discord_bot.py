"""Discord bot integration.

Connects the LangGraph pipeline to Discord using discord.py.
Listens for messages in configured channels and invokes the graph.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import discord

from bot.config import DISCORD_TOKEN
from bot.db import close_db, save_message
from bot.graph import build_graph
from bot.nodes.memory_writer import memory_writer
from bot.state import BotState

logger = logging.getLogger(__name__)


class RolePlayBot(discord.Client):
    """Discord client that routes messages through the LangGraph pipeline."""

    def __init__(
        self,
        personality_path: str | Path,
        channel_ids: list[int] | None = None,
        listen_all: bool = False,
        **kwargs,
    ):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)

        self.personality = self._load_personality(personality_path)
        self.channel_ids = set(channel_ids) if channel_ids else None
        self.listen_all = listen_all
        self.graph = build_graph()

    @staticmethod
    def _load_personality(path: str | Path) -> dict:
        path = Path(path)
        if not path.exists():
            logger.warning("Personality file %s not found, using empty personality", path)
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def on_ready(self):
        logger.info("Logged in as %s (id=%s)", self.user, self.user.id)
        if self.listen_all:
            logger.info("Listening in ALL channels")
        elif self.channel_ids:
            logger.info("Listening in channels: %s", self.channel_ids)
        else:
            logger.info("No channels configured — responding to @mentions only")

    async def on_message(self, message: discord.Message):
        # Ignore own messages
        if message.author == self.user:
            return

        # Ignore bot messages
        if message.author.bot:
            return

        # Determine whether to respond:
        # 1. listen_all=True → respond in every channel
        # 2. channel_ids set  → respond only in those channels
        # 3. neither          → respond only when @mentioned
        if not self.listen_all:
            if self.channel_ids:
                if message.channel.id not in self.channel_ids:
                    return
            else:
                bot_mentioned = self.user in message.mentions if self.user else False
                if not bot_mentioned:
                    return

        # Build initial state
        state: BotState = {
            "user_id": str(message.author.id),
            "user_name": message.author.display_name,
            "channel_id": str(message.channel.id),
            "guild_id": str(message.guild.id) if message.guild else "",
            "message_text": message.content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "should_respond": True,
            "personality": self.personality,
        }

        # Show typing indicator while processing
        async with message.channel.typing():
            try:
                result = await self.graph.ainvoke(state)
            except Exception:
                logger.exception("Graph invocation failed")
                await message.reply("*something went wrong*")
                return

        response = _strip_actions(result.get("response", ""))
        if not response:
            return

        # Use reply() in noisy channels so context is clear; send() otherwise
        use_reply = await self._should_reply(message)

        # Discord has a 2000 char limit; split if necessary
        for chunk in _split_message(response):
            if use_reply:
                await message.reply(chunk)
            else:
                await message.channel.send(chunk)

        # Fire-and-forget: save conversation history + extract user facts
        asyncio.create_task(
            _save_exchange(
                channel_id=str(message.channel.id),
                user_id=str(message.author.id),
                user_name=message.author.display_name,
                bot_id=str(self.user.id) if self.user else "bot",
                user_message=message.content,
                bot_response=response,
            )
        )
        asyncio.create_task(self._run_memory_writer(result))

    async def _should_reply(self, message: discord.Message) -> bool:
        """Decide whether to use reply() or send().

        Uses reply() when:
        - The bot was explicitly @mentioned (user expects a directed response)
        - The channel is noisy (3+ distinct authors in the last 10 messages)

        Otherwise uses a plain send() for a more natural conversation feel.
        """
        # Always reply when @mentioned
        if self.user and self.user in message.mentions:
            return True

        # Check recent channel activity for noise
        try:
            recent_authors: set[int] = set()
            async for msg in message.channel.history(limit=10, before=message):
                recent_authors.add(msg.author.id)
                if len(recent_authors) >= 3:
                    return True
        except Exception:
            pass

        return False

    async def _run_memory_writer(self, graph_result: dict) -> None:
        """Fire-and-forget: run memory writer after reply is sent."""
        try:
            await memory_writer(graph_result)
        except Exception:
            logger.exception("Deferred memory writer failed")

    async def close(self):
        await close_db()
        await super().close()


_ACTION_PATTERNS = [
    re.compile(r"<action>.*?</action>", re.DOTALL),   # <action>...</action>
    re.compile(r"\uff08[^\uff09]*\uff09"),              # （...）
    re.compile(r"\([^)]*\)"),                           # (...)
    re.compile(r"\*[^*]+\*"),                           # *...*
]


def _strip_actions(text: str) -> str:
    """Remove action/narration blocks from the bot response."""
    for pattern in _ACTION_PATTERNS:
        text = pattern.sub("", text)
    # Collapse leftover blank lines and whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_message(text: str, limit: int = 2000) -> list[str]:
    """Split a message into chunks that fit within Discord's character limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at a newline or space
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = text.rfind(" ", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return chunks


async def _save_exchange(
    channel_id: str,
    user_id: str,
    user_name: str,
    bot_id: str,
    user_message: str,
    bot_response: str,
) -> None:
    """Save both sides of the exchange to MongoDB conversation history."""
    ts = datetime.now(timezone.utc).isoformat()
    try:
        await save_message(channel_id, "user", user_id, user_name, user_message, ts)
        await save_message(channel_id, "assistant", bot_id, "bot", bot_response, ts)
    except Exception:
        logger.exception("Failed to save exchange to conversation history")


def run_bot(
    personality_path: str | Path,
    channel_ids: list[int] | None = None,
    listen_all: bool = False,
):
    """Blocking entry point to start the Discord bot."""
    bot = RolePlayBot(
        personality_path=personality_path,
        channel_ids=channel_ids,
        listen_all=listen_all,
    )
    bot.run(DISCORD_TOKEN, log_handler=None)
