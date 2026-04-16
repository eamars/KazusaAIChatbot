"""FastAPI brain service — platform-agnostic entry point for the Kazusa AI chatbot.

Start with:
    uvicorn kazusa_ai_chatbot.service:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field

import os

# Configure root logger early so all application loggers respect the level.
# Set LOG_LEVEL=DEBUG in .env or environment to see full pipeline detail.
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Suppress noisy third-party debug logs
for _quiet in ("pymongo", "httpx", "httpcore", "hpack", "urllib3", "openai", "langsmith"):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

from kazusa_ai_chatbot.config import (
    CONVERSATION_HISTORY_LIMIT,
    PERSONALITY_PATH,
)
from kazusa_ai_chatbot.db import (
    close_db,
    db_bootstrap,
    get_character_state,
    get_conversation_history,
    get_user_profile,
    resolve_global_user_id,
    save_conversation,
)
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.state import DiscordProcessState, MultiMediaDoc
from kazusa_ai_chatbot.utils import load_personality, trim_history_dict
from kazusa_ai_chatbot import scheduler

from langgraph.graph import END, START, StateGraph
from kazusa_ai_chatbot.nodes.relevance_agent import relevance_agent, multimedia_descriptor_agent
from kazusa_ai_chatbot.nodes.persona_supervisor2 import persona_supervisor2

logger = logging.getLogger(__name__)


# ── Pydantic models for the API contract ────────────────────────────


class AttachmentIn(BaseModel):
    media_type: str = ""
    url: str = ""
    base64_data: str = ""
    description: str = ""


class ChatRequest(BaseModel):
    platform: str
    platform_channel_id: str = ""
    platform_user_id: str
    platform_bot_id: str = ""  # Bot's ID on this platform (e.g. Discord snowflake)
    display_name: str = ""
    channel_name: str = ""
    content: str = ""
    content_type: str = "text"
    attachments: list[AttachmentIn] = Field(default_factory=list)
    timestamp: str = ""
    reply_to_message_id: str | None = None


class AttachmentOut(BaseModel):
    media_type: str = ""
    url: str = ""
    base64_data: str = ""
    description: str = ""


class ChatResponse(BaseModel):
    messages: list[str] = Field(default_factory=list)
    content_type: str = "text"
    attachments: list[AttachmentOut] = Field(default_factory=list)
    should_reply: bool = False
    scheduled_followups: int = 0


class EventRequest(BaseModel):
    platform: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    db: bool
    scheduler: bool


# ── Graph builder ───────────────────────────────────────────────────

def _build_graph():
    """Build the LangGraph pipeline (identical to discord_bot.build_graph)."""
    graph = StateGraph(DiscordProcessState)

    graph.add_node("relevance_agent", relevance_agent)
    graph.add_node("multimedia_descriptor_agent", multimedia_descriptor_agent)
    graph.add_node("persona_supervisor2", persona_supervisor2)

    graph.add_conditional_edges(
        START,
        lambda state: "multimedia" if state.get("user_multimedia_input") else "skip",
        {"multimedia": "multimedia_descriptor_agent", "skip": "relevance_agent"},
    )
    graph.add_edge("multimedia_descriptor_agent", "relevance_agent")

    graph.add_conditional_edges(
        "relevance_agent",
        lambda state: "continue" if state.get("should_respond") else "end",
        {"continue": "persona_supervisor2", "end": END},
    )
    graph.add_edge("persona_supervisor2", END)

    return graph.compile()


# ── Conversation saver (background task) ────────────────────────────

async def _save_conversation(result: dict) -> None:
    """Persist user and bot messages to conversation history."""
    platform = result.get("platform", "")
    platform_channel_id = result.get("platform_channel_id", "")
    platform_user_id = result.get("platform_user_id", "")
    global_user_id = result.get("global_user_id", "")
    user_name = result.get("user_name", "")
    user_input = result.get("user_input", "")
    platform_bot_id = result.get("platform_bot_id", "")
    bot_name = result.get("bot_name", "")
    bot_output = result.get("final_dialog", [])

    # Save user message
    try:
        await save_conversation({
            "platform": platform,
            "platform_channel_id": platform_channel_id,
            "role": "user",
            "platform_user_id": platform_user_id,
            "global_user_id": global_user_id,
            "display_name": user_name,
            "content": user_input,
            "timestamp": result.get("timestamp", ""),
        })
    except Exception:
        logger.exception("Failed to save user message")

    # Save bot message
    if bot_output:
        try:
            await save_conversation({
                "platform": platform,
                "platform_channel_id": platform_channel_id,
                "role": "assistant",
                "platform_user_id": platform_bot_id,
                "global_user_id": "",
                "display_name": bot_name,
                "content": "\n".join(bot_output),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            logger.exception("Failed to save bot message")


# ── Lifespan ────────────────────────────────────────────────────────

_personality: dict = {}
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _personality, _graph

    # 1. Database bootstrap
    await db_bootstrap()

    # 2. Load personality
    _personality = load_personality(PERSONALITY_PATH)
    if not _personality:
        logger.warning("Personality file '%s' is empty or missing", PERSONALITY_PATH)

    # 3. Build the LangGraph pipeline
    _graph = _build_graph()

    # 4. Start MCP tool servers
    try:
        await mcp_manager.start()
    except Exception:
        logger.exception("MCP manager failed to start — tools will be unavailable")

    # 5. Load pending scheduled events
    await scheduler.load_pending_events()

    logger.info("Kazusa brain service is ready")

    yield

    # Shutdown
    await scheduler.shutdown()
    await mcp_manager.stop()
    await close_db()
    logger.info("Kazusa brain service shut down")


# ── App ─────────────────────────────────────────────────────────────

app = FastAPI(title="Kazusa Brain Service", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    db_ok = False
    try:
        from kazusa_ai_chatbot.db import get_db
        db = await get_db()
        await db.client.admin.command("ping")
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db=db_ok,
        scheduler=True,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    timestamp = req.timestamp or datetime.now(timezone.utc).isoformat()

    # Resolve global user ID
    global_user_id = await resolve_global_user_id(
        platform=req.platform,
        platform_user_id=req.platform_user_id,
        display_name=req.display_name,
    )
    user_profile = await get_user_profile(global_user_id)
    character_state = await get_character_state()

    # Convert attachments to MultiMediaDoc list
    multimedia_input: list[MultiMediaDoc] = []
    for att in req.attachments:
        if att.media_type.startswith("image/") and att.base64_data:
            multimedia_input.append({
                "content_type": att.media_type,
                "base64_data": att.base64_data,
                "description": att.description,
            })

    # Fetch conversation history
    history = await get_conversation_history(
        platform=req.platform,
        platform_channel_id=req.platform_channel_id,
        limit=CONVERSATION_HISTORY_LIMIT,
    )
    trimmed_history = trim_history_dict(history)

    # Build the character bot identity
    bot_name = _personality.get("name", "KazusaBot")

    initial_state: DiscordProcessState = {
        "timestamp": timestamp,
        "platform": req.platform,
        "platform_user_id": req.platform_user_id,
        "global_user_id": global_user_id,
        "user_name": req.display_name,
        "user_input": req.content,
        "user_multimedia_input": multimedia_input,
        "user_profile": user_profile,
        "platform_bot_id": req.platform_bot_id,
        "bot_name": bot_name,
        "character_profile": _personality,
        "character_state": character_state,
        "platform_channel_id": req.platform_channel_id,
        "channel_name": req.channel_name,
        "chat_history": trimmed_history,
    }

    # Invoke the graph
    try:
        result = await _graph.ainvoke(initial_state)
    except Exception:
        logger.exception("Graph invocation failed")
        return ChatResponse(messages=[f"{bot_name} is busy right now, please try again later."])

    final_dialog = result.get("final_dialog", [])
    should_reply = result.get("use_reply_feature", False)

    # Save conversation in background
    background_tasks.add_task(_save_conversation, result)

    # TODO: Extract scheduled_followups from result["future_promises"] and schedule them
    scheduled_followups = 0

    return ChatResponse(
        messages=final_dialog,
        content_type="text",
        attachments=[],
        should_reply=should_reply,
        scheduled_followups=scheduled_followups,
    )


@app.post("/event")
async def event(req: EventRequest):
    """Receive a platform event (user joined, topic change, etc.)."""
    logger.info("Received event: %s/%s", req.platform, req.event_type)
    # TODO: dispatch to event handlers
    return {"status": "accepted"}
