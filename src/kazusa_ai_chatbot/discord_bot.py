"""Discord bot integration.

Connects the LangGraph pipeline to Discord using discord.py.
Listens for messages in configured channels and invokes the graph.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
import httpx
import base64

import discord

from kazusa_ai_chatbot.config import DISCORD_TOKEN, CONVERSATION_HISTORY_LIMIT
from kazusa_ai_chatbot.db import close_db, get_conversation_history, save_conversation, get_user_profile, get_character_state, resolve_global_user_id
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.state import DiscordProcessState, MultiMediaDoc
from kazusa_ai_chatbot.utils import load_personality, trim_history_dict

from langgraph.graph import END, START, StateGraph
from kazusa_ai_chatbot.nodes.relevance_agent import relevance_agent, multimedia_descriptor_agent
from kazusa_ai_chatbot.nodes.persona_supervisor2 import persona_supervisor2


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

        self.personality = load_personality(personality_path)
        self.channel_ids = set(channel_ids) if channel_ids else None
        self.listen_all = listen_all
        self.graph = self.build_graph()

    def build_graph(self):
        graph = StateGraph(DiscordProcessState)

        graph.add_node("relevance_agent", relevance_agent)
        graph.add_node("multimedia_descriptor_agent", multimedia_descriptor_agent)
        graph.add_node("persona_supervisor2", persona_supervisor2)

        # Build edges
        graph.add_conditional_edges(
            START,
            lambda state: "multimedia" if state["user_multimedia_input"] else "skip",
            {
                "multimedia": "multimedia_descriptor_agent",
                "skip": "relevance_agent",
            }
        )
        graph.add_edge("multimedia_descriptor_agent", "relevance_agent")

        # Stop early if relevance agent decide not to proceed
        graph.add_conditional_edges(
            "relevance_agent",
            lambda state: "continue" if state["should_respond"] else "end",
            {
                "continue": "persona_supervisor2",
                "end": END,
            }
        )
        
        graph.add_edge("persona_supervisor2", END)

        return graph.compile()

    async def on_ready(self):
        logger.info("Logged in as %s (id=%s)", self.user, self.user.id)
        if self.listen_all:
            logger.info("Listening in ALL channels")
        elif self.channel_ids:
            logger.info("Listening in channels: %s", self.channel_ids)
        else:
            logger.info("No channels configured — responding to @mentions only")

        # Connect to MCP tool servers
        try:
            await mcp_manager.start()
        except Exception:
            logger.exception("MCP manager failed to start — tools will be unavailable")


    async def on_message(self, message: discord.Message):
        # Ignore own messages
        if message.author == self.user:
            return

        # Ignore bot messages
        # if message.author.bot:
        #     return

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

        # Fetch attachment and encode into base64 format
        user_input = message.content
        multimedia_input = []

        for attachment in message.attachments:
            # Check attachment size
            if attachment.size and attachment.size > (5 * 1024 * 1024):
                logger.warning("Attachment size exceeds 5MB limit, skil")
                continue

            # Capture image
            if attachment.content_type and attachment.content_type.startswith("image/"):
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(attachment.url)
                        response.raise_for_status()

                        # Process content
                        image_data = response.content
                        image_base64 = base64.b64encode(image_data).decode("utf-8")
                        
                        # Append to the attachment list
                        media: MultiMediaDoc = {
                            "content_type": attachment.content_type,
                            "base64_data": image_base64
                        }
                        multimedia_input.append(media)
                except httpx.HTTPError as e:
                    logger.error(f"Failed to fetch attachment {attachment.url}: {e}")

        # Build initial state
        platform = "discord"
        timestamp = datetime.now(timezone.utc).isoformat()
        platform_user_id = str(message.author.id)
        user_name = message.author.display_name
        platform_bot_id = str(self.user.id) if self.user else ""
        bot_name = str(self.user.name) if self.user else ""  # This is the displayed name from discord, not the character's real name
        platform_channel_id = str(message.channel.id)
        character_state = await get_character_state()

        # Resolve global user ID (auto-creates profile if not found)
        global_user_id = await resolve_global_user_id(
            platform=platform,
            platform_user_id=platform_user_id,
            display_name=user_name,
        )
        user_profile = await get_user_profile(global_user_id)

        # Private message (DM)
        if (message.guild is None):
            channel_name = f"Private chat with {user_name}"
        else:
            channel_name = str(message.channel.name)

        history = await get_conversation_history(
            platform=platform,
            platform_channel_id=platform_channel_id,
            limit=CONVERSATION_HISTORY_LIMIT,
        )
        trimmed_history = trim_history_dict(history)

        initial_state: DiscordProcessState = {
            "timestamp": timestamp,

            "platform": platform,
            "platform_user_id": platform_user_id,
            "global_user_id": global_user_id,
            "user_name": user_name,
            "user_input": user_input,
            "user_multimedia_input": multimedia_input,
            "user_profile": user_profile,

            "platform_bot_id": platform_bot_id,
            "bot_name": bot_name,
            "character_profile": self.personality,
            "character_state": character_state,

            "platform_channel_id": platform_channel_id,
            "channel_name": channel_name,
            "chat_history": trimmed_history,
        }

        # Show typing indicator while processing
        async with message.channel.typing():
            try:
                result = await self.graph.ainvoke(initial_state)
            except Exception as e:
                logger.exception("Graph invocation failed: %s", e)
                await message.reply(f"{bot_name} is busy right now, please try again later.")
                return

        # Read output from persona supervisor
        final_dialog = result.get("final_dialog", [])
        if not final_dialog:
            return

        # Use reply feature decision from relevance agent
        use_reply = result.get("use_reply_feature", False)

        # Send messages
        msg_to_send = "\n".join(final_dialog)
        # Discord has a 2000 char limit; split if necessary
        for chunk in _split_message(msg_to_send):
            if use_reply:
                await message.reply(chunk)
                use_reply = False  # Only reply the first message
            else:
                await message.channel.send(chunk)

        # Fire-and-forget: save conversation history + extract user facts
        asyncio.create_task(self.conversation_saver(result))

    async def conversation_saver(self, graph_result: DiscordProcessState) -> None:
        platform = graph_result.get("platform", "discord")
        platform_channel_id = graph_result.get("platform_channel_id", "")
        platform_user_id = graph_result.get("platform_user_id", "")
        global_user_id = graph_result.get("global_user_id", "")
        user_name = graph_result.get("user_name", "")
        user_input = graph_result.get("user_input", "")
        platform_bot_id = graph_result.get("platform_bot_id", "")
        bot_name = graph_result.get("bot_name", "")
        bot_input = graph_result.get("final_dialog", [])

        # Save user message
        try:
            await save_conversation(
                {
                    "platform": platform,
                    "platform_channel_id": platform_channel_id,
                    "role": "user",
                    "platform_user_id": platform_user_id,
                    "global_user_id": global_user_id,
                    "display_name": user_name,
                    "content": user_input,
                    "timestamp": graph_result.get("timestamp", ""),
                }
            )
        except Exception as e:
            logger.exception("Failed to save conversation from user: %s", e)

        # save bot message
        current_timestamp = datetime.now(timezone.utc).isoformat()
        try:
            await save_conversation(
                {
                    "platform": platform,
                    "platform_channel_id": platform_channel_id,
                    "role": "assistant",
                    "platform_user_id": platform_bot_id,
                    "global_user_id": "",
                    "display_name": bot_name,
                    "content": "\n".join(bot_input),
                    "timestamp": current_timestamp,
                }
            )
        except Exception as e:
            logger.exception("Failed to save conversation from bot: %s", e)

    async def close(self):
        await mcp_manager.stop()
        await close_db()
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
        # Try to split at a newline or space
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = text.rfind(" ", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return chunks


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
