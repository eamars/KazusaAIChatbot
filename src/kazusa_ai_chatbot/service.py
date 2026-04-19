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
    BRAIN_EXECUTOR_COUNT,
    CONVERSATION_HISTORY_LIMIT,
    SCHEDULED_TASKS_ENABLED,
)
from kazusa_ai_chatbot.db import (
    close_db,
    db_bootstrap,
    get_character_profile,
    get_conversation_history,
    get_user_profile,
    resolve_global_user_id,
    save_conversation,
)
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.state import IMProcessState, MultiMediaDoc, DebugModes
from kazusa_ai_chatbot.utils import trim_history_dict
from kazusa_ai_chatbot import scheduler

from langgraph.graph import END, START, StateGraph
from kazusa_ai_chatbot.nodes.relevance_agent import relevance_agent, multimedia_descriptor_agent
from kazusa_ai_chatbot.nodes.persona_supervisor2 import persona_supervisor2
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import _get_rag_cache

logger = logging.getLogger(__name__)


# ── Pydantic models for the API contract ────────────────────────────


class AttachmentIn(BaseModel):
    media_type: str = ""
    url: str = ""
    base64_data: str = ""
    description: str = ""


class DebugModesIn(BaseModel):
    listen_only: bool = False
    think_only: bool = False
    no_remember: bool = False


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
    debug_modes: DebugModesIn = Field(default_factory=DebugModesIn)


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
    """Build the LangGraph pipeline for the brain service."""
    graph = StateGraph(IMProcessState)

    graph.add_node("relevance_agent", relevance_agent)
    graph.add_node("multimedia_descriptor_agent", multimedia_descriptor_agent)
    graph.add_node("persona_supervisor2", persona_supervisor2)

    def _start_router(state):
        debug = state.get("debug_modes") or {}
        if debug.get("listen_only"):
            return "end"
        if state.get("user_multimedia_input"):
            return "multimedia"
        return "skip"

    graph.add_conditional_edges(
        START,
        _start_router,
        {"multimedia": "multimedia_descriptor_agent", "skip": "relevance_agent", "end": END},
    )
    graph.add_edge("multimedia_descriptor_agent", "relevance_agent")

    def _route_after_relevance(state):
        if not state.get("should_respond"):
            return "end"
        return "continue"

    graph.add_conditional_edges(
        "relevance_agent",
        _route_after_relevance,
        {"continue": "persona_supervisor2", "end": END},
    )
    graph.add_edge("persona_supervisor2", END)

    return graph.compile()


# ── Bot message saver (background task) ──────────────────────────────

async def _save_bot_message(result: dict) -> None:
    """Persist the bot's response to conversation history (background task)."""
    platform = result.get("platform", "")
    platform_channel_id = result.get("platform_channel_id", "")
    platform_bot_id = result.get("platform_bot_id", "")
    bot_name = result.get("bot_name", "")
    bot_output = result.get("final_dialog", [])

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
_chat_executor_semaphore: asyncio.Semaphore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _personality, _graph, _chat_executor_semaphore

    executor_count = max(1, BRAIN_EXECUTOR_COUNT)
    _chat_executor_semaphore = asyncio.Semaphore(executor_count)
    logger.info("Chat executor limit set to %s", executor_count)

    # 1. Database bootstrap
    await db_bootstrap()

    # 2. Load character profile from database
    _personality = await get_character_profile()
    if not _personality.get("name"):
        raise RuntimeError(
            "No character profile found in the database. "
            "Please load one first with:  "
            "python -m scripts.load_character_profile personalities/kazusa.json"
        )

    # 3. Build the LangGraph pipeline
    _graph = _build_graph()

    # 4. Start MCP tool servers
    try:
        await mcp_manager.start()
    except Exception:
        logger.exception("MCP manager failed to start — tools will be unavailable")

    # 5. Warm-start the RAG cache from MongoDB (Stage 5b).
    rag_cache = await _get_rag_cache()
    logger.info("RAG cache warm-started: %s", rag_cache.get_stats())

    # 6. Load pending scheduled events
    if SCHEDULED_TASKS_ENABLED:
        await scheduler.load_pending_events()
    else:
        logger.info("Scheduler disabled via SCHEDULED_TASKS_ENABLED=false — skipping load_pending_events")

    logger.info("Kazusa brain service is ready")

    yield

    # Shutdown
    if SCHEDULED_TASKS_ENABLED:
        await scheduler.shutdown()
    try:
        cache = await _get_rag_cache()
        await cache.shutdown()
    except Exception:
        logger.exception("RAG cache shutdown failed")
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
    semaphore = _chat_executor_semaphore
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)

    async with semaphore:
        timestamp = req.timestamp or datetime.now(timezone.utc).isoformat()

        # Resolve global user ID
        global_user_id = await resolve_global_user_id(
            platform=req.platform,
            platform_user_id=req.platform_user_id,
            display_name=req.display_name,
        )
        user_profile = await get_user_profile(global_user_id)

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

        debug_modes: DebugModes = {
            "listen_only": req.debug_modes.listen_only,
            "think_only": req.debug_modes.think_only,
            "no_remember": req.debug_modes.no_remember,
        }
        active_flags = [k for k, v in debug_modes.items() if v]
        if active_flags:
            logger.info("Debug modes active: %s", active_flags)

        initial_state: IMProcessState = {
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
            "platform_channel_id": req.platform_channel_id,
            "channel_name": req.channel_name,
            "chat_history": trimmed_history,
            "debug_modes": debug_modes,
        }

        # Save user message immediately (before graph invocation)
        try:
            await save_conversation({
                "platform": req.platform,
                "platform_channel_id": req.platform_channel_id,
                "role": "user",
                "platform_user_id": req.platform_user_id,
                "global_user_id": global_user_id,
                "display_name": req.display_name,
                "content": req.content,
                "timestamp": timestamp,
            })
        except Exception:
            logger.exception("Failed to save user message")

        # Invoke the graph
        try:
            result = await _graph.ainvoke(initial_state)
        except Exception:
            logger.exception("Graph invocation failed")
            return ChatResponse(messages=[f"{bot_name} is busy right now, please try again later."])

        final_dialog = result.get("final_dialog", [])
        should_reply = result.get("use_reply_feature", False)

        # Save bot message in background only if the bot actually generated a response
        if final_dialog:
            background_tasks.add_task(_save_bot_message, result)

        # think_only: suppress dialog in response but still save internally
        if debug_modes.get("think_only"):
            logger.info("think_only active — suppressing %d dialog message(s) from user output", len(final_dialog))
            final_dialog = []

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
