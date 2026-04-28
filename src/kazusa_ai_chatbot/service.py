"""FastAPI brain service — platform-agnostic entry point for the Kazusa AI chatbot.

Start with:
    uvicorn kazusa_ai_chatbot.service:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field

from kazusa_ai_chatbot.config import (
    BRAIN_EXECUTOR_COUNT,
    CHARACTER_GLOBAL_USER_ID,
    CHAT_HISTORY_RECENT_LIMIT,
    CONVERSATION_HISTORY_LIMIT,
    SCHEDULED_TASKS_ENABLED,
)
from kazusa_ai_chatbot.conversation_progress import (
    ConversationProgressRecordInput,
    ConversationProgressScope,
    load_progress_context,
    record_turn_progress,
)
from kazusa_ai_chatbot.db import (
    backfill_character_conversation_identity,
    close_db,
    db_bootstrap,
    ensure_character_identity,
    get_character_profile,
    get_conversation_history,
    get_db,
    get_user_profile,
    resolve_global_user_id,
    save_conversation,
)
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.state import IMProcessState, MultiMediaDoc, DebugModes, ReplyContext
from kazusa_ai_chatbot.utils import log_list_preview, log_preview, trim_history_dict
from kazusa_ai_chatbot import scheduler
from kazusa_ai_chatbot.dispatcher import (
    AdapterRegistry,
    PendingTaskIndex,
    RemoteHttpAdapter,
    TaskDispatcher,
    ToolCallEvaluator,
    ToolRegistry,
    build_send_message_tool,
)

from langgraph.graph import END, START, StateGraph
from kazusa_ai_chatbot.nodes.relevance_agent import relevance_agent, multimedia_descriptor_agent
from kazusa_ai_chatbot.nodes.persona_supervisor2 import persona_supervisor2
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator import call_consolidation_subgraph
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_persistence import configure_task_dispatcher
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime

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


class ReplyContextIn(BaseModel):
    reply_to_message_id: str = ""
    reply_to_platform_user_id: str = ""
    reply_to_display_name: str = ""
    reply_to_current_bot: bool | None = None
    reply_excerpt: str = ""


class ChatRequest(BaseModel):
    platform: str
    platform_channel_id: str = ""
    channel_type: str = "group"
    platform_message_id: str = ""
    platform_user_id: str
    platform_bot_id: str = ""  # Bot's ID on this platform (e.g. Discord snowflake)
    display_name: str = ""
    channel_name: str = ""
    content: str = ""
    content_type: str = "text"
    mentioned_bot: bool = False
    attachments: list[AttachmentIn] = Field(default_factory=list)
    timestamp: str = ""
    reply_to_message_id: str | None = None
    reply_context: ReplyContextIn = Field(default_factory=ReplyContextIn)
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


class Cache2AgentStatsResponse(BaseModel):
    agent_name: str
    hit_count: int
    miss_count: int
    hit_rate: float


class Cache2HealthResponse(BaseModel):
    agents: list[Cache2AgentStatsResponse] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    db: bool
    scheduler: bool
    cache2: Cache2HealthResponse = Field(default_factory=Cache2HealthResponse)


class RuntimeAdapterRegistrationRequest(BaseModel):
    platform: str
    callback_url: str
    shared_secret: str = ""
    timeout_seconds: float = 10.0


class RuntimeAdapterRegistrationResponse(BaseModel):
    status: str
    platform: str
    callback_url: str


def _register_runtime_adapter_payload(
    req: RuntimeAdapterRegistrationRequest,
    *,
    status: str,
) -> RuntimeAdapterRegistrationResponse:
    """Register one remote adapter payload and return a normalized response.

    Args:
        req: Remote adapter registration or heartbeat payload.
        status: Response status string to return to the caller.

    Returns:
        Structured confirmation for the adapter process.
    """

    register_remote_runtime_adapter(
        platform=req.platform,
        callback_url=req.callback_url,
        shared_secret=req.shared_secret,
        timeout_seconds=req.timeout_seconds,
    )
    return RuntimeAdapterRegistrationResponse(
        status=status,
        platform=req.platform,
        callback_url=req.callback_url,
    )


# ── Graph builder ───────────────────────────────────────────────────

def _build_graph():
    """Build the LangGraph pipeline for the brain service."""
    graph = StateGraph(IMProcessState)

    graph.add_node("relevance_agent", relevance_agent)
    graph.add_node("multimedia_descriptor_agent", multimedia_descriptor_agent)
    graph.add_node("load_conversation_episode_state", load_conversation_episode_state)
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
        {"continue": "load_conversation_episode_state", "end": END},
    )
    graph.add_edge("load_conversation_episode_state", "persona_supervisor2")
    graph.add_edge("persona_supervisor2", END)

    return graph.compile()


async def load_conversation_episode_state(state: IMProcessState) -> dict:
    """Load prompt-facing conversation progress after relevance approves response.

    Args:
        state: Current service graph state after relevance.

    Returns:
        Partial state update containing stored episode state and compact progress.
    """

    scope = ConversationProgressScope(
        platform=state["platform"],
        platform_channel_id=state["platform_channel_id"],
        global_user_id=state["global_user_id"],
    )
    load_result = await load_progress_context(
        scope=scope,
        current_timestamp=state["timestamp"],
    )
    logger.debug(
        "Conversation progress loaded: platform=%s channel=%s user=%s source=%s turn_count=%d",
        scope.platform,
        scope.platform_channel_id or "<dm>",
        scope.global_user_id,
        load_result["source"],
        load_result["conversation_progress"]["turn_count"],
    )
    return {
        "conversation_episode_state": load_result["episode_state"],
        "conversation_progress": load_result["conversation_progress"],
    }


def _compact_reply_context(reply_context: ReplyContext) -> ReplyContext:
    compacted: ReplyContext = {}
    for key, value in reply_context.items():
        if value in ("", None):
            continue
        compacted[key] = value
    return compacted


async def _hydrate_reply_context(req: ChatRequest) -> ReplyContext:
    reply_context: ReplyContext = req.reply_context.model_dump(exclude_none=True)

    if req.reply_to_message_id and not reply_context.get("reply_to_message_id"):
        reply_context["reply_to_message_id"] = req.reply_to_message_id

    reply_to_message_id = reply_context.get("reply_to_message_id", "")
    if reply_to_message_id and not reply_context.get("reply_to_platform_user_id"):
        db = await get_db()
        reply_doc = await db.conversation_history.find_one(
            {
                "platform": req.platform,
                "platform_channel_id": req.platform_channel_id,
                "platform_message_id": reply_to_message_id,
            },
            projection={
                "platform_user_id": 1,
                "display_name": 1,
                "content": 1,
            },
        )
        if reply_doc is not None:
            reply_context["reply_to_platform_user_id"] = reply_doc.get("platform_user_id", "")
            reply_context["reply_to_display_name"] = reply_doc.get("display_name", "")
            reply_context["reply_excerpt"] = reply_doc.get("content", "")

    reply_to_platform_user_id = reply_context.get("reply_to_platform_user_id", "")
    if reply_to_platform_user_id:
        reply_context["reply_to_current_bot"] = reply_to_platform_user_id == req.platform_bot_id

    return _compact_reply_context(reply_context)


# ── Bot message saver (background task) ──────────────────────────────

async def _ensure_character_global_identity(
    *,
    platform: str,
    platform_bot_id: str,
    bot_name: str,
) -> str:
    """Ensure the character identity exists and old assistant rows are addressable.

    Args:
        platform: Runtime platform for the current request.
        platform_bot_id: Bot account ID on that platform.
        bot_name: Character display name used by the platform adapter.

    Returns:
        The configured stable character ``global_user_id``.
    """
    character_global_user_id = await ensure_character_identity(
        platform=platform,
        platform_user_id=platform_bot_id,
        display_name=bot_name,
        global_user_id=CHARACTER_GLOBAL_USER_ID,
    )

    backfill_key = (platform, platform_bot_id, character_global_user_id)
    if platform and platform_bot_id and backfill_key not in _character_identity_backfilled:
        updated_count = await backfill_character_conversation_identity(
            platform=platform,
            platform_user_id=platform_bot_id,
            global_user_id=character_global_user_id,
        )
        _character_identity_backfilled.add(backfill_key)
        if updated_count:
            logger.info(
                "Backfilled %d assistant conversation rows with character global_user_id=%s",
                updated_count,
                character_global_user_id,
            )

    return character_global_user_id


async def _save_bot_message(result: dict) -> None:
    """Persist the bot's response to conversation history (background task)."""
    platform = result.get("platform", "")
    platform_channel_id = result.get("platform_channel_id", "")
    platform_bot_id = result.get("platform_bot_id", "")
    bot_name = result.get("bot_name", "")
    bot_output = result.get("final_dialog", [])

    if bot_output:
        try:
            character_global_user_id = await _ensure_character_global_identity(
                platform=platform,
                platform_bot_id=platform_bot_id,
                bot_name=bot_name,
            )
            await save_conversation({
                "platform": platform,
                "platform_channel_id": platform_channel_id,
                "channel_type": result.get("channel_type", "group"),
                "role": "assistant",
                "platform_user_id": platform_bot_id,
                "global_user_id": character_global_user_id,
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
_task_dispatcher: TaskDispatcher | None = None
_adapter_registry: AdapterRegistry | None = None
_character_identity_backfilled: set[tuple[str, str, str]] = set()


async def _run_consolidation_background(state: dict) -> None:
    """Run stage-4 consolidation after the dialog has already been returned.

    Args:
        state: Stage-0..3 persona state snapshot needed by the consolidator.
    """

    try:
        await call_consolidation_subgraph(state)
    except Exception:
        logger.exception("Background consolidation failed")


async def _run_conversation_progress_record_background(state: dict) -> None:
    """Record conversation progress after dialog output has been returned.

    Args:
        state: Stage-0..3 persona state snapshot needed by the progress recorder.

    Returns:
        None.
    """

    try:
        linguistic_directives = state["action_directives"]["linguistic_directives"]
        scope = ConversationProgressScope(
            platform=state["platform"],
            platform_channel_id=state["platform_channel_id"],
            global_user_id=state["global_user_id"],
        )
        record_input: ConversationProgressRecordInput = {
            "scope": scope,
            "timestamp": state["timestamp"],
            "prior_episode_state": state.get("conversation_episode_state"),
            "decontexualized_input": state["decontexualized_input"],
            "chat_history_recent": state["chat_history_recent"],
            "content_anchors": linguistic_directives["content_anchors"],
            "logical_stance": state["logical_stance"],
            "character_intent": state["character_intent"],
            "final_dialog": state["final_dialog"],
        }
        result = await record_turn_progress(record_input=record_input)
        logger.debug(
            "Conversation progress recorded: platform=%s channel=%s user=%s written=%s turn_count=%d continuity=%s status=%s cache_updated=%s",
            scope.platform,
            scope.platform_channel_id or "<dm>",
            scope.global_user_id,
            result["written"],
            result["turn_count"],
            result["continuity"],
            result["status"],
            result["cache_updated"],
        )
    except Exception:
        logger.exception("Background conversation progress recording failed")


def register_runtime_adapter(adapter) -> None:
    """Register a live messaging adapter for scheduled tool delivery.

    Args:
        adapter: Adapter implementing the dispatcher messaging protocol.
    """

    if _adapter_registry is None:
        raise RuntimeError("Adapter registry is not initialized yet")
    _adapter_registry.register(adapter)


def register_remote_runtime_adapter(
    *,
    platform: str,
    callback_url: str,
    shared_secret: str = "",
    timeout_seconds: float = 10.0,
) -> None:
    """Register a cross-process adapter callback for scheduled delivery.

    Args:
        platform: Platform key such as ``qq`` or ``discord``.
        callback_url: Base callback URL exposed by the adapter process.
        shared_secret: Optional bearer token used when the brain calls back.
        timeout_seconds: Timeout for one outbound callback request.
    """

    register_runtime_adapter(
        RemoteHttpAdapter(
            platform=platform,
            callback_url=callback_url,
            shared_secret=shared_secret,
            timeout_seconds=timeout_seconds,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _personality, _graph, _chat_executor_semaphore, _task_dispatcher, _adapter_registry

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

    # 5. Build the task-dispatch runtime
    tool_registry = ToolRegistry()
    tool_registry.register(build_send_message_tool())
    adapter_registry = AdapterRegistry()
    _adapter_registry = adapter_registry
    pending_index = PendingTaskIndex()
    await pending_index.rebuild_from_db()
    evaluator = ToolCallEvaluator(tool_registry, adapter_registry)
    _task_dispatcher = TaskDispatcher(evaluator, pending_index)
    configure_task_dispatcher(_task_dispatcher, tool_registry)
    scheduler.configure_runtime(
        tool_registry=tool_registry,
        adapter_registry=adapter_registry,
        pending_index=pending_index,
    )

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
    await mcp_manager.stop()
    await close_db()
    logger.info("Kazusa brain service shut down")


# ── App ─────────────────────────────────────────────────────────────

app = FastAPI(title="Kazusa Brain Service", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    db_ok = False
    try:
        db = await get_db()
        await db.client.admin.command("ping")
        db_ok = True
    except Exception:
        logger.debug("Health check database ping failed", exc_info=True)

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db=db_ok,
        scheduler=True,
        cache2=Cache2HealthResponse(
            agents=[
                Cache2AgentStatsResponse(**row)
                for row in get_rag_cache2_runtime().get_agent_stats()
            ],
        ),
    )


@app.post("/runtime/adapters/register", response_model=RuntimeAdapterRegistrationResponse)
async def register_runtime_adapter_endpoint(req: RuntimeAdapterRegistrationRequest):
    """Register one cross-process adapter callback for scheduler delivery.

    Args:
        req: Remote adapter registration payload sent by an adapter process.

    Returns:
        Confirmation payload describing the registered callback.
    """

    return _register_runtime_adapter_payload(
        req,
        status="registered",
    )


@app.post("/runtime/adapters/heartbeat", response_model=RuntimeAdapterRegistrationResponse)
async def runtime_adapter_heartbeat_endpoint(req: RuntimeAdapterRegistrationRequest):
    """Refresh one adapter registration so the brain can recover after restarts.

    Args:
        req: Adapter heartbeat payload describing the callback endpoint.

    Returns:
        Confirmation payload describing the refreshed callback.
    """

    return _register_runtime_adapter_payload(
        req,
        status="heartbeat_ok",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    semaphore = _chat_executor_semaphore
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)

    async with semaphore:
        timestamp = req.timestamp or datetime.now(timezone.utc).isoformat()

        # Ensure the character is addressable by global_user_id before RAG.
        bot_name = _personality.get("name", "KazusaBot")
        await _ensure_character_global_identity(
            platform=req.platform,
            platform_bot_id=req.platform_bot_id,
            bot_name=bot_name,
        )

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

        # Fetch conversation history — wide slice for relevance, recent for downstream
        history = await get_conversation_history(
            platform=req.platform,
            platform_channel_id=req.platform_channel_id,
            limit=CONVERSATION_HISTORY_LIMIT,
        )
        chat_history_wide = trim_history_dict(history)
        chat_history_recent = chat_history_wide[-CHAT_HISTORY_RECENT_LIMIT:]
        reply_context = await _hydrate_reply_context(req)

        debug_modes: DebugModes = {
            "listen_only": req.debug_modes.listen_only,
            "think_only": req.debug_modes.think_only,
            "no_remember": req.debug_modes.no_remember,
        }
        active_flags = [k for k, v in debug_modes.items() if v]
        if active_flags:
            logger.info("Debug modes active: %s", active_flags)

        logger.debug(
            "Chat request: platform=%s channel=%s message=%s user=%s global_user=%s content_type=%s attachments=%d image_attachments=%d history_wide=%d history_recent=%d reply_context=%s debug_modes=%s content=%s",
            req.platform,
            req.platform_channel_id or "<dm>",
            req.platform_message_id or "<none>",
            req.platform_user_id,
            global_user_id,
            req.content_type,
            len(req.attachments),
            len(multimedia_input),
            len(chat_history_wide),
            len(chat_history_recent),
            log_preview(reply_context),
            active_flags,
            log_preview(req.content),
        )

        initial_state: IMProcessState = {
            "timestamp": timestamp,
            "platform": req.platform,
            "platform_message_id": req.platform_message_id,
            "platform_user_id": req.platform_user_id,
            "global_user_id": global_user_id,
            "user_name": req.display_name,
            "user_input": req.content,
            "user_multimedia_input": multimedia_input,
            "user_profile": user_profile,
            "platform_bot_id": req.platform_bot_id,
            "mentioned_bot": req.mentioned_bot,
            "bot_name": bot_name,
            "character_profile": _personality,
            "platform_channel_id": req.platform_channel_id,
            "channel_type": req.channel_type,
            "channel_name": req.channel_name,
            "chat_history_wide": chat_history_wide,
            "chat_history_recent": chat_history_recent,
            "reply_context": reply_context,
            "should_respond": False,
            "reason_to_respond": "",
            "use_reply_feature": False,
            "channel_topic": "",
            "indirect_speech_context": "",
            "debug_modes": debug_modes,
        }

        # Save user message immediately (before graph invocation)
        try:
            await save_conversation({
                "platform": req.platform,
                "platform_channel_id": req.platform_channel_id,
                "role": "user",
                "platform_message_id": req.platform_message_id,
                "platform_user_id": req.platform_user_id,
                "global_user_id": global_user_id,
                "display_name": req.display_name,
                "channel_type": req.channel_type,
                "content": req.content,
                "mentioned_bot": req.mentioned_bot,
                "reply_context": reply_context,
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
        consolidation_state = result.get("consolidation_state")

        logger.debug(
            "Chat result: platform=%s channel=%s message=%s user=%s should_respond=%s should_reply=%s final_dialog_count=%d future_promises=%d final_dialog=%s",
            req.platform,
            req.platform_channel_id or "<dm>",
            req.platform_message_id or "<none>",
            req.platform_user_id,
            result.get("should_respond"),
            should_reply,
            len(final_dialog),
            len(result.get("future_promises", [])),
            log_list_preview(final_dialog),
        )

        # Save bot message in background only if the bot actually generated a response
        if final_dialog:
            background_tasks.add_task(_save_bot_message, result)

        if final_dialog and isinstance(consolidation_state, Mapping):
            background_tasks.add_task(_run_conversation_progress_record_background, dict(consolidation_state))
            logger.debug(
                "Background conversation progress recorder queued: platform=%s channel=%s message=%s",
                req.platform,
                req.platform_channel_id or "<dm>",
                req.platform_message_id or "<none>",
            )

        if debug_modes.get("no_remember"):
            logger.debug("Background consolidation skipped: no_remember is active")
        elif isinstance(consolidation_state, Mapping):
            background_tasks.add_task(_run_consolidation_background, dict(consolidation_state))
            logger.debug(
                "Background consolidation queued: platform=%s channel=%s message=%s",
                req.platform,
                req.platform_channel_id or "<dm>",
                req.platform_message_id or "<none>",
            )
        else:
            logger.warning(
                "Background consolidation skipped: unexpected consolidation_state type=%s",
                type(consolidation_state).__name__,
            )

        # think_only: suppress dialog in response but still save internally
        if debug_modes.get("think_only"):
            logger.info("think_only active — suppressing %d dialog message(s) from user output", len(final_dialog))
            final_dialog = []

        # The dispatcher now runs in the background consolidator path, so the
        # synchronous /chat response no longer waits on scheduling.
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
