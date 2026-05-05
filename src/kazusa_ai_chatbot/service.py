"""FastAPI brain service — platform-agnostic entry point for the Kazusa AI chatbot.

Start with:
    uvicorn kazusa_ai_chatbot.service:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone

from fastapi import FastAPI, BackgroundTasks

from kazusa_ai_chatbot.config import (
    CHARACTER_GLOBAL_USER_ID,
    CHAT_HISTORY_RECENT_LIMIT,
    CONVERSATION_HISTORY_LIMIT,
    RAG_CACHE2_MAX_ENTRIES,
    REFLECTION_CYCLE_DISABLED,
    SCHEDULED_TASKS_ENABLED,
)
from kazusa_ai_chatbot.conversation_progress import (
    ConversationProgressScope,
    load_progress_context,
    record_turn_progress,
)
from kazusa_ai_chatbot.llm_route_report import render_llm_route_table
from kazusa_ai_chatbot.db import (
    backfill_character_conversation_identity,
    close_db,
    db_bootstrap,
    ensure_character_identity,
    get_character_profile,
    get_conversation_history,
    get_db,
    get_user_profile,
    load_initializer_entries,
    resolve_global_user_id,
    save_conversation,
)
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.state import IMProcessState, MultiMediaDoc, DebugModes, ReplyContext
from kazusa_ai_chatbot.time_context import build_character_time_context
from kazusa_ai_chatbot.chat_input_queue import ChatInputQueue, QueuedChatItem
from kazusa_ai_chatbot.message_envelope import (
    MessageEnvelope,
    project_prompt_message_context,
)
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

from kazusa_ai_chatbot.brain_service import (
    cache_startup as brain_cache_startup,
    graph as brain_graph,
    health as brain_health,
    intake as brain_intake,
    post_turn as brain_post_turn,
    runtime_adapters as brain_runtime_registry,
)
from kazusa_ai_chatbot.brain_service.contracts import (
    AttachmentIn,
    AttachmentOut,
    AttachmentRefIn,
    Cache2AgentStatsResponse,
    Cache2HealthResponse,
    ChatRequest,
    ChatResponse,
    DebugModesIn,
    EventRequest,
    HealthResponse,
    MentionIn,
    MessageEnvelopeIn,
    ReplyTargetIn,
    RuntimeAdapterRegistrationRequest,
    RuntimeAdapterRegistrationResponse,
)
from kazusa_ai_chatbot.nodes.relevance_agent import relevance_agent, multimedia_descriptor_agent
from kazusa_ai_chatbot.nodes.persona_supervisor2 import persona_supervisor2
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator import call_consolidation_subgraph
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_persistence import configure_task_dispatcher
from kazusa_ai_chatbot.rag.cache2_policy import INITIALIZER_CACHE_NAME
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from kazusa_ai_chatbot.reflection_cycle import (
    ReflectionWorkerHandle,
    build_promoted_reflection_context,
    start_reflection_cycle_worker,
    stop_reflection_cycle_worker,
)

logger = logging.getLogger(__name__)


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

    return_value = brain_runtime_registry.register_runtime_adapter_payload(
        req,
        status=status,
        register_remote_runtime_adapter_func=register_remote_runtime_adapter,
    )
    return return_value


# ── Graph builder ───────────────────────────────────────────────────

def _build_graph():
    """Build the LangGraph pipeline for the brain service."""
    return_value = brain_graph.build_graph(
        relevance_agent_node=relevance_agent,
        multimedia_descriptor_agent_node=multimedia_descriptor_agent,
        load_conversation_episode_state_node=load_conversation_episode_state,
        persona_supervisor_node=persona_supervisor2,
    )
    return return_value


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
    progress = load_result["conversation_progress"]
    logger.info(
        f"Conversation progress loaded: platform={scope.platform} "
        f'channel={scope.platform_channel_id or "<dm>"} '
        f'user={scope.global_user_id} source={load_result["source"]} '
        f'turn_count={progress["turn_count"]} '
        f'continuity={progress["continuity"]} status={progress["status"]}'
    )
    logger.debug(
        f"Conversation progress loaded detail: platform={scope.platform} "
        f'channel={scope.platform_channel_id or "<dm>"} '
        f"user={scope.global_user_id} progress={log_preview(progress)}"
    )
    return_value = {
        "conversation_episode_state": load_result["episode_state"],
        "conversation_progress": load_result["conversation_progress"],
    }
    return return_value


def _compact_reply_context(reply_context: ReplyContext) -> ReplyContext:
    compacted_context = brain_intake.compact_reply_context(reply_context)
    return compacted_context


async def _hydrate_reply_context(req: ChatRequest) -> ReplyContext:
    """Build service-facing reply context from the typed envelope only.

    Args:
        req: Incoming chat request from an adapter.

    Returns:
        Compact reply context projected from ``message_envelope.reply``.
    """

    return_value = await brain_intake.hydrate_reply_context(req)
    return return_value


async def _resolve_message_envelope_identities(req: ChatRequest) -> MessageEnvelope:
    """Resolve typed envelope mentions and reply targets to global user ids.

    Args:
        req: Incoming chat request carrying an adapter-normalized envelope.

    Returns:
        Message envelope with user/profile identities resolved and addressees
        recomputed from typed mentions, typed reply target, and DM defaults.
    """

    envelope = await brain_intake.resolve_message_envelope_identities(
        req,
        character_global_user_id=CHARACTER_GLOBAL_USER_ID,
        resolve_global_user_id_func=resolve_global_user_id,
    )
    return envelope


# ── Assistant message saver (background task) ────────────────────────

async def _ensure_character_global_identity(
    *,
    platform: str,
    platform_bot_id: str,
    character_name: str,
) -> str:
    """Ensure the character identity exists and old assistant rows are addressable.

    Args:
        platform: Runtime platform for the current request.
        platform_bot_id: Bot account ID on that platform.
        character_name: Active character display name.

    Returns:
        The configured stable character ``global_user_id``.
    """
    character_global_user_id = await ensure_character_identity(
        platform=platform,
        platform_user_id=platform_bot_id,
        display_name=character_name,
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
            logger.info(f'Backfilled {updated_count} assistant conversation rows with character global_user_id={character_global_user_id}')

    return character_global_user_id


async def _save_assistant_message(result: dict) -> None:
    """Persist the assistant-authored response to conversation history."""
    await brain_post_turn.save_assistant_message(
        result,
        ensure_character_global_identity_func=_ensure_character_global_identity,
        save_conversation_func=save_conversation,
        now_func=lambda: datetime.now(timezone.utc),
        logger=logger,
    )


# ── Lifespan ────────────────────────────────────────────────────────

_personality: dict = {}
_graph = None
_task_dispatcher: TaskDispatcher | None = None
_adapter_registry: AdapterRegistry | None = None
_character_identity_backfilled: set[tuple[str, str, str]] = set()
_chat_input_queue = ChatInputQueue()
_chat_queue_worker_task: asyncio.Task | None = None
_reflection_worker_handle: ReflectionWorkerHandle | None = None
_primary_interaction_active_count = 0


def _primary_interaction_busy() -> bool:
    """Return whether chat or post-dialog work currently has priority."""

    return_value = (
        _primary_interaction_active_count > 0
        or _chat_input_queue.pending_count() > 0
    )
    return return_value


def _active_turn_platform_message_ids(item: QueuedChatItem) -> list[str]:
    """Build the platform message ID list answered by one graph turn.

    Args:
        item: Surviving queued item that will run through the chat graph.

    Returns:
        Non-empty platform message IDs from the survivor and collapsed
        follow-ups, deduplicated in arrival order.
    """

    return_value = brain_intake.active_turn_platform_message_ids(item)
    return return_value


async def _save_user_message_from_item(
    item: QueuedChatItem,
    *,
    global_user_id: str,
    reply_context: ReplyContext,
    message_envelope: MessageEnvelope | None = None,
) -> None:
    """Persist one queued user message.

    Args:
        item: Queued chat item containing the request and timestamp.
        global_user_id: Resolved global user identifier.
        reply_context: Adapter-supplied reply metadata after compacting.
        message_envelope: Envelope after service-side identity resolution, when
            the caller already resolved it for graph input.

    Returns:
        None.
    """

    await brain_intake.save_user_message_from_item(
        item,
        global_user_id=global_user_id,
        reply_context=reply_context,
        save_conversation_func=save_conversation,
        resolve_message_envelope_identities_func=_resolve_message_envelope_identities,
        message_envelope=message_envelope,
        logger=logger,
    )


async def _resolve_queued_user(item: QueuedChatItem) -> tuple[str, dict]:
    """Resolve the user identity and profile for a queued item.

    Args:
        item: Queued chat item.

    Returns:
        Pair of global user ID and user profile.
    """

    return_value = await brain_intake.resolve_queued_user(
        item,
        resolve_global_user_id_func=resolve_global_user_id,
        get_user_profile_func=get_user_profile,
    )
    return return_value


async def _drop_queued_chat_item(item: QueuedChatItem) -> None:
    """Persist and complete one pruned queued item without running the graph.

    Args:
        item: Queued chat item selected for pruning.

    Returns:
        None.
    """

    try:
        global_user_id, _ = await _resolve_queued_user(item)
        reply_context = await _hydrate_reply_context(item.request)
        await _save_user_message_from_item(
            item,
            global_user_id=global_user_id,
            reply_context=reply_context,
        )
    except Exception as exc:
        logger.exception(f"Failed to persist dropped queued message: {exc}")

    _chat_input_queue.complete(item, ChatResponse())
    dropped_envelope: MessageEnvelope = item.request.message_envelope.model_dump(
        exclude_none=True,
        exclude_defaults=True,
    )
    logger.info(
        f"Queued chat item dropped: sequence={item.sequence} "
        f"platform={item.request.platform} "
        f'channel={item.request.platform_channel_id or "<dm>"} '
        f'message={item.request.platform_message_id or "<none>"} '
        f'user={item.request.platform_user_id or "<none>"} '
        f"tagged={_chat_input_queue.is_tagged(item)} "
        f"bot_reply={_chat_input_queue.is_bot_reply(item)}"
    )
    logger.debug(
        f"Queued chat item dropped detail: sequence={item.sequence} "
        f'display_name={item.request.display_name or "<none>"} '
        f'content={log_preview(dropped_envelope["body_text"])}'
    )


async def _persist_collapsed_queued_chat_item(
    item: QueuedChatItem,
    survivor: QueuedChatItem,
) -> None:
    """Persist and complete one queued item collapsed into a surviving turn.

    Args:
        item: Queued chat item collapsed into another item.
        survivor: Surviving queued item that will receive the character reply.

    Returns:
        None.
    """

    try:
        global_user_id, _ = await _resolve_queued_user(item)
        reply_context = await _hydrate_reply_context(item.request)
        await _save_user_message_from_item(
            item,
            global_user_id=global_user_id,
            reply_context=reply_context,
        )
    except Exception as exc:
        logger.exception(f"Failed to persist collapsed queued message: {exc}")

    _chat_input_queue.complete(item, ChatResponse())
    collapsed_envelope: MessageEnvelope = item.request.message_envelope.model_dump(
        exclude_none=True,
        exclude_defaults=True,
    )
    logger.info(
        f"Queued chat item collapsed: sequence={item.sequence} "
        f"survivor_sequence={survivor.sequence} "
        f"platform={item.request.platform} "
        f'channel={item.request.platform_channel_id or "<dm>"} '
        f'message={item.request.platform_message_id or "<none>"} '
        f'survivor_message={survivor.request.platform_message_id or "<none>"} '
        f'user={item.request.platform_user_id or "<none>"} '
        f"tagged={_chat_input_queue.is_tagged(item)} "
        f"bot_reply={_chat_input_queue.is_bot_reply(item)}"
    )
    logger.debug(
        f"Queued chat item collapsed detail: sequence={item.sequence} "
        f'display_name={item.request.display_name or "<none>"} '
        f'content={log_preview(collapsed_envelope["body_text"])}'
    )


async def _process_queued_chat_item(item: QueuedChatItem) -> None:
    """Run one queued item through the existing chat graph and post-writes.

    Args:
        item: Oldest surviving queued item selected by the worker.

    Returns:
        None.
    """

    req = item.request
    character_name = _personality.get("name", "Character")

    try:
        character_global_user_id = await _ensure_character_global_identity(
            platform=req.platform,
            platform_bot_id=req.platform_bot_id,
            character_name=character_name,
        )
        global_user_id, user_profile = await _resolve_queued_user(item)
        message_envelope = await _resolve_message_envelope_identities(req)

        multimedia_input: list[MultiMediaDoc] = []
        for queued_item in [item, *item.collapsed_items]:
            queued_envelope: MessageEnvelope = (
                queued_item.request.message_envelope.model_dump(
                    exclude_none=True,
                    exclude_defaults=True,
                )
            )
            for attachment in queued_envelope["attachments"]:
                media_type = attachment.get("media_type", "")
                if not media_type.startswith("image/"):
                    continue
                base64_data = attachment.get("base64_data", "")
                if not base64_data:
                    continue
                multimedia_input.append({
                    "content_type": media_type,
                    "base64_data": base64_data,
                    "description": attachment.get("description", ""),
                })

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
        active_flags = [key for key, value in debug_modes.items() if value]
        if active_flags:
            logger.info(f'Debug modes active: {active_flags}')

        user_input = item.combined_content or message_envelope["body_text"]
        prompt_message_context = project_prompt_message_context(
            message_envelope=message_envelope,
            multimedia_input=multimedia_input,
        )
        is_collapsed_turn = bool(item.collapsed_items)
        active_turn_platform_message_ids = _active_turn_platform_message_ids(item)
        character_profile = dict(_personality)
        character_profile["global_user_id"] = character_global_user_id

        logger.debug(f'Chat request: platform={req.platform} channel={req.platform_channel_id or "<dm>"} message={req.platform_message_id or "<none>"} user={req.platform_user_id} global_user={global_user_id} content_type={req.content_type} attachments={len(message_envelope["attachments"])} image_attachments={len(multimedia_input)} history_wide={len(chat_history_wide)} history_recent={len(chat_history_recent)} reply_context={log_preview(reply_context)} debug_modes={active_flags} collapsed={is_collapsed_turn} collapsed_count={len(item.collapsed_items)} content={log_preview(user_input)}')

        # Reply anchoring is a false-preserving graph latch. Normal turns start
        # enabled; collapsed turns start disabled so no later stage can re-enable
        # the platform reply feature for multi-message input.
        initial_use_reply_feature = not is_collapsed_turn

        time_context = build_character_time_context(item.timestamp)
        try:
            promoted_reflection_context = await build_promoted_reflection_context()
        except Exception as exc:
            logger.exception(f"Promoted reflection context load failed: {exc}")
            promoted_reflection_context = {}

        initial_state: IMProcessState = {
            "timestamp": item.timestamp,
            "time_context": time_context,
            "platform": req.platform,
            "platform_message_id": req.platform_message_id,
            "active_turn_platform_message_ids": active_turn_platform_message_ids,
            "platform_user_id": req.platform_user_id,
            "global_user_id": global_user_id,
            "user_name": req.display_name,
            "user_input": user_input,
            "message_envelope": message_envelope,
            "prompt_message_context": prompt_message_context,
            "user_multimedia_input": multimedia_input,
            "user_profile": user_profile,
            "platform_bot_id": req.platform_bot_id,
            "character_name": character_name,
            "character_profile": character_profile,
            "platform_channel_id": req.platform_channel_id,
            "channel_type": req.channel_type,
            "channel_name": req.channel_name,
            "chat_history_wide": chat_history_wide,
            "chat_history_recent": chat_history_recent,
            "reply_context": reply_context,
            "should_respond": False,
            "reason_to_respond": "",
            "use_reply_feature": initial_use_reply_feature,
            "channel_topic": "",
            "indirect_speech_context": "",
            "debug_modes": debug_modes,
            "final_dialog": [],
            "target_addressed_user_ids": [global_user_id],
            "target_broadcast": False,
            "future_promises": [],
            "consolidation_state": {},
            "promoted_reflection_context": promoted_reflection_context,
        }

        await _save_user_message_from_item(
            item,
            global_user_id=global_user_id,
            reply_context=reply_context,
            message_envelope=message_envelope,
        )

        try:
            result = await _graph.ainvoke(initial_state)
        except Exception as exc:
            logger.exception(f"Graph invocation failed: {exc}")
            response = ChatResponse(
                messages=[
                    f"{character_name} is busy right now, please try again later."
                ]
            )
            _chat_input_queue.complete(item, response)
            return

        final_dialog = result["final_dialog"]
        should_reply = bool(final_dialog) and bool(
            result["use_reply_feature"]
        )
        consolidation_state = result["consolidation_state"]

        logger.debug(f'Chat result: platform={req.platform} channel={req.platform_channel_id or "<dm>"} message={req.platform_message_id or "<none>"} user={req.platform_user_id} should_respond={result["should_respond"]} should_reply={should_reply} final_dialog_count={len(final_dialog)} future_promises={len(result["future_promises"])} final_dialog={log_list_preview(final_dialog)}')

        consolidation_state_dict: dict | None = None
        if isinstance(consolidation_state, Mapping):
            consolidation_state_dict = dict(consolidation_state)

        has_consolidation_state = bool(consolidation_state_dict)
        should_record_progress = bool(final_dialog) and has_consolidation_state
        if should_record_progress:
            logger.debug(f'Background conversation progress recorder queued: platform={req.platform} channel={req.platform_channel_id or "<dm>"} message={req.platform_message_id or "<none>"}')
        elif not final_dialog:
            logger.info(f'Background conversation progress recorder skipped: platform={req.platform} channel={req.platform_channel_id or "<dm>"} message={req.platform_message_id or "<none>"} should_respond={result["should_respond"]} final_dialog_count=0')
        else:
            logger.warning(f'Background conversation progress recorder skipped: unexpected consolidation_state type={type(consolidation_state).__name__}')

        should_consolidate = False
        if debug_modes.get("no_remember"):
            logger.debug("Background consolidation skipped: no_remember is active")
        elif final_dialog and has_consolidation_state:
            should_consolidate = True
            logger.debug(f'Background consolidation queued: platform={req.platform} channel={req.platform_channel_id or "<dm>"} message={req.platform_message_id or "<none>"}')
        elif not final_dialog:
            logger.info(f'Background consolidation skipped: platform={req.platform} channel={req.platform_channel_id or "<dm>"} message={req.platform_message_id or "<none>"} should_respond={result["should_respond"]} final_dialog_count=0')
        else:
            logger.warning(f'Background consolidation skipped: unexpected consolidation_state type={type(consolidation_state).__name__}')

        should_save_assistant_message = bool(final_dialog)
        response_dialog = final_dialog
        if debug_modes.get("think_only"):
            logger.info(f'think_only active — suppressing {len(final_dialog)} dialog message(s) from user output')
            response_dialog = []

        response = ChatResponse(
            messages=response_dialog,
            content_type="text",
            attachments=[],
            should_reply=should_reply,
            scheduled_followups=0,
        )
        _chat_input_queue.complete(item, response)

        if should_save_assistant_message:
            await _save_assistant_message(result)
        if should_record_progress and consolidation_state_dict is not None:
            await _run_conversation_progress_record_background(
                consolidation_state_dict,
            )
        if should_consolidate and consolidation_state_dict is not None:
            await _run_consolidation_background(consolidation_state_dict)
    except Exception as exc:
        logger.exception(f"Queued chat item failed: {exc}")
        response = ChatResponse(
            messages=[
                f"{character_name} is busy right now, please try again later."
            ]
        )
        _chat_input_queue.complete(item, response)


async def _chat_input_worker() -> None:
    """Consume queue handoffs and run service-owned message actions.

    Returns:
        None.
    """

    global _primary_interaction_active_count

    while True:
        dequeued_turn = await _chat_input_queue.wait_for_next()

        _primary_interaction_active_count += 1
        try:
            for dropped_item in dequeued_turn.dropped_items:
                await _drop_queued_chat_item(dropped_item)

            for collapsed_item, survivor in dequeued_turn.collapsed_items:
                await _persist_collapsed_queued_chat_item(collapsed_item, survivor)

            if dequeued_turn.next_item is not None:
                await _process_queued_chat_item(dequeued_turn.next_item)
        finally:
            _primary_interaction_active_count -= 1


def _ensure_chat_input_worker_started() -> None:
    """Ensure the process-local input worker exists for the current event loop.

    Returns:
        None.
    """

    global _chat_queue_worker_task
    if _chat_queue_worker_task is None or _chat_queue_worker_task.done():
        _chat_queue_worker_task = asyncio.create_task(_chat_input_worker())


async def _stop_chat_input_worker() -> None:
    """Stop the process-local input worker and resolve pending requests.

    Returns:
        None.
    """

    global _chat_input_queue, _chat_queue_worker_task
    task = _chat_queue_worker_task
    _chat_queue_worker_task = None
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    pending_items = await _chat_input_queue.drain()
    for item in pending_items:
        _chat_input_queue.complete(item, ChatResponse())
    _chat_input_queue = ChatInputQueue()


async def _enqueue_chat_request(req: ChatRequest) -> ChatResponse:
    """Enqueue one request and wait for the worker-produced response.

    Args:
        req: Incoming chat request.

    Returns:
        Chat response produced by the worker or drop/collapse policy.
    """

    _ensure_chat_input_worker_started()
    response = await _chat_input_queue.enqueue(req)
    return response


async def _run_consolidation_background(state: dict) -> None:
    """Run consolidation after the dialog has already been returned.

    Args:
        state: Persona graph state snapshot needed by the consolidator.
    """

    await brain_post_turn.run_consolidation_background(
        state,
        call_consolidation_subgraph_func=call_consolidation_subgraph,
        personality=_personality,
        logger=logger,
    )


async def _run_conversation_progress_record_background(state: dict) -> None:
    """Record conversation progress after dialog output has been returned.

    Args:
        state: Persona graph state snapshot needed by the progress recorder.

    Returns:
        None.
    """

    await brain_post_turn.run_conversation_progress_record_background(
        state,
        record_turn_progress_func=record_turn_progress,
        logger=logger,
    )


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


async def _hydrate_rag_initializer_cache() -> int:
    """Hydrate current-version persistent initializer cache rows into memory.

    Returns:
        Number of valid rows loaded into the process-local Cache2 runtime.
    """

    loaded_count = await brain_cache_startup.hydrate_rag_initializer_cache(
        load_initializer_entries_func=load_initializer_entries,
        get_rag_cache2_runtime_func=get_rag_cache2_runtime,
        cache_name=INITIALIZER_CACHE_NAME,
        max_entries=RAG_CACHE2_MAX_ENTRIES,
        logger=logger,
    )
    return loaded_count


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _personality, _graph, _task_dispatcher, _adapter_registry
    global _reflection_worker_handle

    # 1. Database bootstrap
    await db_bootstrap()

    # 2. Hydrate persistent RAG initializer cache into the process-local LRU
    await _hydrate_rag_initializer_cache()

    # 3. Load character profile from database
    _personality = await get_character_profile()
    if not _personality.get("name"):
        raise RuntimeError(
            "No character profile found in the database. "
            "Please load one first with:  "
            "python -m scripts.load_character_profile personalities/kazusa.json"
        )

    # 4. Build the LangGraph pipeline
    _graph = _build_graph()

    # 5. Start MCP tool servers
    try:
        await mcp_manager.start()
    except Exception as exc:
        logger.exception(
            f"MCP manager failed to start — tools will be unavailable: {exc}"
        )

    # 6. Build the task-dispatch runtime
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

    # 7. Load pending scheduled events
    if SCHEDULED_TASKS_ENABLED:
        await scheduler.load_pending_events()
    else:
        logger.info("Scheduler disabled via SCHEDULED_TASKS_ENABLED=false — skipping load_pending_events")

    logger.info(render_llm_route_table())
    _ensure_chat_input_worker_started()
    if REFLECTION_CYCLE_DISABLED:
        logger.info(
            "Reflection cycle worker disabled via REFLECTION_CYCLE_DISABLED=true"
        )
    else:
        _reflection_worker_handle = start_reflection_cycle_worker(
            is_primary_interaction_busy=_primary_interaction_busy,
        )
    logger.info("Kazusa brain service is ready")

    yield

    # Shutdown
    if _reflection_worker_handle is not None:
        await stop_reflection_cycle_worker(_reflection_worker_handle)
        _reflection_worker_handle = None
    await _stop_chat_input_worker()
    if SCHEDULED_TASKS_ENABLED:
        await scheduler.shutdown()
    await mcp_manager.stop()
    await close_db()
    logger.info("Kazusa brain service shut down")


# ── App ─────────────────────────────────────────────────────────────

app = FastAPI(title="Kazusa Brain Service", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return_value = await brain_health.build_health_response(
        get_db_func=get_db,
        get_rag_cache2_runtime_func=get_rag_cache2_runtime,
        logger=logger,
    )
    return return_value


@app.post("/runtime/adapters/register", response_model=RuntimeAdapterRegistrationResponse)
async def register_runtime_adapter_endpoint(req: RuntimeAdapterRegistrationRequest):
    """Register one cross-process adapter callback for scheduler delivery.

    Args:
        req: Remote adapter registration payload sent by an adapter process.

    Returns:
        Confirmation payload describing the registered callback.
    """

    return_value = _register_runtime_adapter_payload(
        req,
        status="registered",
    )
    return return_value


@app.post("/runtime/adapters/heartbeat", response_model=RuntimeAdapterRegistrationResponse)
async def runtime_adapter_heartbeat_endpoint(req: RuntimeAdapterRegistrationRequest):
    """Refresh one adapter registration so the brain can recover after restarts.

    Args:
        req: Adapter heartbeat payload describing the callback endpoint.

    Returns:
        Confirmation payload describing the refreshed callback.
    """

    return_value = _register_runtime_adapter_payload(
        req,
        status="heartbeat_ok",
    )
    return return_value


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    """Enqueue an inbound chat message and wait for queue processing.

    Args:
        req: Incoming chat request from an adapter.
        background_tasks: FastAPI background task container, unused because the
            queue worker owns post-response processing.

    Returns:
        Chat response produced by the input queue worker.
    """

    _ = background_tasks
    response = await _enqueue_chat_request(req)
    return response


@app.post("/event")
async def event(req: EventRequest):
    """Receive a platform event (user joined, topic change, etc.)."""
    logger.info(f'Received event: {req.platform}/{req.event_type}')
    # TODO: dispatch to event handlers
    return_value = {"status": "accepted"}
    return return_value
