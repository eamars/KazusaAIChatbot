"""FastAPI brain service — platform-agnostic entry point for the Kazusa AI chatbot.

Start with:
    uvicorn kazusa_ai_chatbot.service:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import logging
import os
import socket
import time
import traceback
from uuid import uuid4
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from typing import Any, Literal

from fastapi import FastAPI, BackgroundTasks

from kazusa_ai_chatbot.action_spec.execution import execute_action_specs_for_trace
from kazusa_ai_chatbot.action_spec.results import has_consolidatable_output
from kazusa_ai_chatbot.config import (
    CALENDAR_SCHEDULER_CLAIM_LIMIT,
    CALENDAR_SCHEDULER_ENABLED,
    CALENDAR_SCHEDULER_LEASE_SECONDS,
    CALENDAR_SCHEDULER_MAX_ATTEMPTS,
    CALENDAR_SCHEDULER_POLL_INTERVAL_SECONDS,
    BACKGROUND_WORK_INPUT_CHAR_LIMIT,
    BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    BACKGROUND_WORK_WORKER_CLAIM_LIMIT,
    BACKGROUND_WORK_WORKER_ENABLED,
    BACKGROUND_WORK_WORKER_INTERVAL_SECONDS,
    BACKGROUND_WORK_WORKER_LEASE_SECONDS,
    BACKGROUND_WORK_WORKER_MAX_ATTEMPTS,
    CHARACTER_GLOBAL_USER_ID,
    CHAT_HISTORY_RECENT_LIMIT,
    CONVERSATION_HISTORY_LIMIT,
    COGNITION_VISUAL_DIRECTIVES_ENABLED,
    MEDIA_DESCRIPTOR_CACHE_MAX_HYDRATION_ENTRIES,
    RAG_CACHE2_MAX_ENTRIES,
    REFLECTION_CYCLE_ENABLED,
    REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD,
    REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS,
    REFLECTION_WORKER_INTERVAL_SECONDS,
    SELF_COGNITION_ENABLED,
    SELF_COGNITION_MAX_CASES_PER_TICK,
    SELF_COGNITION_WORKER_INTERVAL_SECONDS,
)
from kazusa_ai_chatbot.calendar_scheduler import models as calendar_models
from kazusa_ai_chatbot.calendar_scheduler import repository as calendar_repository
from kazusa_ai_chatbot.calendar_scheduler.reflection_phase import (
    CalendarReflectionPhaseRunProvider,
    handle_reflection_phase_calendar_run,
)
from kazusa_ai_chatbot.calendar_scheduler.worker import (
    CALENDAR_SCHEDULER_LEASE_OWNER,
    CalendarRunHandlerRegistry,
    CalendarSchedulerWorkerHandle,
    start_calendar_scheduler_worker,
    stop_calendar_scheduler_worker,
)
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    build_reply_media_description_rows,
    build_text_chat_cognitive_episode,
    build_text_chat_media_description_rows,
)
from kazusa_ai_chatbot.conversation_progress import (
    ConversationProgressScope,
    load_progress_context,
    record_turn_progress,
)
from kazusa_ai_chatbot.internal_monologue_residue import (
    load_residue_context,
    record_completed_episode_residue,
)
from kazusa_ai_chatbot.past_dialog_cognition import (
    build_past_dialog_cognition_context,
    candidate_from_conversation_row,
)
from kazusa_ai_chatbot.llm_interface.route_report import render_llm_route_table
from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.reflection_cycle.phase_scheduler import (
    REFLECTION_PHASE_GROUPS_PER_SLOT,
)
from kazusa_ai_chatbot.runtime_coordination import (
    PipelineCoordinator,
    PipelineScope,
)
from kazusa_ai_chatbot.db import (
    DatabaseBackendError,
    backfill_character_conversation_identity,
    check_database_connection,
    close_db,
    compose_character_profile,
    db_bootstrap,
    ensure_character_identity,
    apply_assistant_delivery_receipt,
    get_character_profile,
    get_character_runtime_state,
    get_conversation_by_platform_message_id,
    get_conversation_history,
    get_user_profile,
    load_initializer_entries,
    load_media_descriptor_entries,
    query_active_commitment_memory_units_for_user,
    resolve_global_user_id,
    save_conversation,
    split_character_profile_runtime_state,
)
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.state import IMProcessState, MultiMediaDoc, DebugModes, ReplyContext
from kazusa_ai_chatbot.time_boundary import (
    parse_storage_utc_datetime,
    storage_utc_now,
    storage_utc_now_iso,
)
from kazusa_ai_chatbot.chat_input_queue import ChatInputQueue, QueuedChatItem
from kazusa_ai_chatbot.message_envelope import (
    MessageEnvelope,
    project_prompt_message_context,
    project_reply_attachment_summaries,
)
from kazusa_ai_chatbot.utils import log_list_preview, log_preview, trim_history_dict
from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.dispatcher import (
    AdapterRegistry,
    DispatchContext,
    RemoteHttpAdapter,
    handle_send_message,
)

from kazusa_ai_chatbot.brain_service import (
    cache_startup as brain_cache_startup,
    graph as brain_graph,
    health as brain_health,
    intake as brain_intake,
    post_turn as brain_post_turn,
    runtime_adapters as brain_runtime_registry,
)
from kazusa_ai_chatbot.brain_service.delivery_mentions import (
    build_inline_delivery_mentions,
)
from kazusa_ai_chatbot.brain_service.contracts import (
    AttachmentIn as AttachmentIn,
    AttachmentOut as AttachmentOut,
    AttachmentRefIn as AttachmentRefIn,
    Cache2AgentStatsResponse as Cache2AgentStatsResponse,
    Cache2HealthResponse as Cache2HealthResponse,
    ChatRequest,
    ChatResponse,
    DebugModesIn as DebugModesIn,
    DeliveryReceiptRequest,
    DeliveryReceiptResponse,
    EventRequest,
    HealthResponse,
    MentionIn as MentionIn,
    MessageEnvelopeIn as MessageEnvelopeIn,
    OpsLatestCognitionGraphResponse,
    OpsRuntimeStatusResponse,
    OpsSelfCognitionStatsResponse,
    OpsStatsResponse,
    ReplyTargetIn as ReplyTargetIn,
    RuntimeAdapterRegistrationRequest,
    RuntimeAdapterRegistrationResponse,
)
from kazusa_ai_chatbot.nodes.persona_relevance_agent import relevance_agent
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer import (
    multimedia_descriptor_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2 import persona_supervisor2
from kazusa_ai_chatbot.nodes.persona_supervisor2_memory_lifecycle import (
    call_post_surface_memory_lifecycle_review,
)
from kazusa_ai_chatbot.consolidation.core import call_consolidation_subgraph
from kazusa_ai_chatbot.rag.cache2_policy import (
    INITIALIZER_CACHE_NAME,
    MEDIA_DESCRIPTOR_CACHE_NAME,
)
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from kazusa_ai_chatbot.reflection_cycle import (
    ReflectionWorkerHandle,
    build_promoted_reflection_context,
    should_pause_self_cognition_for_affect_settling,
    start_reflection_cycle_worker,
    stop_reflection_cycle_worker,
)
from kazusa_ai_chatbot.self_cognition import (
    SelfCognitionWorkerHandle,
    start_self_cognition_worker,
    stop_self_cognition_worker,
)
from kazusa_ai_chatbot.self_cognition import models as self_cognition_models
from kazusa_ai_chatbot.background_work import (
    BackgroundWorkRuntimeHandle,
    start_background_work_runtime,
    stop_background_work_runtime,
)

logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000
SERVICE_COMPONENT = "brain_service"
CONVERSATION_HISTORY_COLLECTION = "conversation_history"


def _service_event_scope(req: ChatRequest) -> event_logging.EventScopeInput:
    """Project adapter request scope into the event-log public input shape.

    Args:
        req: Incoming chat request from an adapter.

    Returns:
        Scope fields allowed by the event-logging public interface.
    """

    scope = event_logging.EventScopeInput(
        platform=req.platform,
        channel_type=req.channel_type,
    )
    if req.platform_channel_id:
        scope["platform_channel_id"] = req.platform_channel_id
    return scope


def _chat_correlation_id(req: ChatRequest) -> str:
    """Build a stable non-content correlation id for one inbound chat request.

    Args:
        req: Incoming chat request from an adapter.

    Returns:
        Correlation id derived from platform metadata, not message body text or
        raw channel identifiers.
    """

    message_ref = req.platform_message_id or "no-message-id"
    channel_source = req.platform_channel_id or "direct"
    channel_bytes = f"{req.platform}:{channel_source}".encode(
        "utf-8",
        errors="replace",
    )
    channel_ref = f"ch_{hashlib.sha256(channel_bytes).hexdigest()[:16]}"
    correlation_id = f"chat:{req.platform}:{channel_ref}:{message_ref}"
    return correlation_id


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed monotonic milliseconds since a start marker."""

    elapsed = time.perf_counter() - started_at
    elapsed_ms = max(0, int(elapsed * MILLISECONDS_PER_SECOND))
    return elapsed_ms


def _queue_wait_ms(item: QueuedChatItem) -> int:
    """Return how long a queued item waited before service processing.

    Args:
        item: Queued chat item assigned storage UTC time at enqueue time.

    Returns:
        Non-negative queue wait duration in milliseconds. Invalid storage UTC
        values are treated as unknown and reported as zero.
    """

    try:
        queued_at = parse_storage_utc_datetime(item.storage_timestamp_utc)
    except ValueError:
        return_value = 0
        return return_value

    now = storage_utc_now()
    wait_seconds = max(0.0, (now - queued_at).total_seconds())
    wait_ms = int(wait_seconds * MILLISECONDS_PER_SECOND)
    return wait_ms


def _runtime_error_fields(exc: BaseException) -> tuple[str, str, str, str]:
    """Build sanitized runtime-error metadata from an exception.

    Args:
        exc: Exception caught at a service boundary.

    Returns:
        Tuple of error class, short preview, stack fingerprint, and top frame.
    """

    traceback_frames = traceback.extract_tb(exc.__traceback__)
    if traceback_frames:
        top_frame = traceback_frames[-1]
        top_frame_module = top_frame.filename
        fingerprint_source = (
            f"{exc.__class__.__module__}:{exc.__class__.__name__}:"
            f"{top_frame.filename}:{top_frame.name}:{top_frame.lineno}"
        )
    else:
        top_frame_module = ""
        fingerprint_source = f"{exc.__class__.__module__}:{exc.__class__.__name__}"

    fingerprint_bytes = fingerprint_source.encode("utf-8", errors="replace")
    stack_fingerprint = hashlib.sha256(fingerprint_bytes).hexdigest()[:16]
    error_class = exc.__class__.__name__
    error_preview = str(exc)
    return_value = (
        error_class,
        error_preview,
        stack_fingerprint,
        top_frame_module,
    )
    return return_value


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
    """Load prompt-facing pre-cognition context after relevance approves response.

    Args:
        state: Current service graph state after relevance.

    Returns:
        Partial state update containing compact progress and private residue.
    """

    scope = ConversationProgressScope(
        platform=state["platform"],
        platform_channel_id=state["platform_channel_id"],
        global_user_id=state["global_user_id"],
    )
    load_result = await load_progress_context(
        scope=scope,
        current_timestamp_utc=state["storage_timestamp_utc"],
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
    residue_scope = {
        "character_id": _character_id_from_profile(state["character_profile"]),
        "platform": state["platform"],
        "platform_channel_id": state["platform_channel_id"],
        "channel_type": state["channel_type"],
        "global_user_id": state["global_user_id"],
    }
    residue_result = await load_residue_context(
        trigger_scope=residue_scope,
        current_timestamp_utc=state["storage_timestamp_utc"],
    )
    logger.info(
        f"Internal monologue residue loaded: platform={scope.platform} "
        f'channel={scope.platform_channel_id or "<dm>"} '
        f'user={scope.global_user_id} status={residue_result["status"]} '
        f'selected={residue_result["selected_count"]}'
    )
    past_dialog_cognition_context = await _load_reply_past_dialog_context(
        state,
        character_global_user_id=residue_scope["character_id"],
    )
    return_value = {
        "conversation_episode_state": load_result["episode_state"],
        "conversation_progress": load_result["conversation_progress"],
        "internal_monologue_residue_context": (
            residue_result["internal_monologue_residue_context"]
        ),
        "past_dialog_cognition_context": past_dialog_cognition_context,
    }
    return return_value


def _character_id_from_profile(character_profile: Mapping[str, object]) -> str:
    """Return the configured character id from a composed profile."""

    value = character_profile.get("global_user_id")
    if isinstance(value, str) and value:
        character_id = value
    else:
        character_id = CHARACTER_GLOBAL_USER_ID
    return character_id


def _compact_reply_context(reply_context: ReplyContext) -> ReplyContext:
    compacted_context = brain_intake.compact_reply_context(reply_context)
    return compacted_context


async def _load_reply_past_dialog_context(
    state: Mapping[str, Any],
    *,
    character_global_user_id: str,
) -> str:
    """Return private residual for a directly replied assistant row.

    Args:
        state: Current service graph state after reply hydration.
        character_global_user_id: Internal id of the active character.

    Returns:
        Prompt-facing private context for L2a, or an empty string.
    """

    reply_context = state["reply_context"]
    reply_to_message_id = str(reply_context.get("reply_to_message_id") or "")
    if not reply_to_message_id:
        context = ""
        return context

    try:
        row = await get_conversation_by_platform_message_id(
            platform=state["platform"],
            platform_channel_id=state["platform_channel_id"],
            platform_message_id=reply_to_message_id,
        )
    except DatabaseBackendError as exc:
        logger.warning(
            f"Past-dialog cognition reply row lookup skipped: {exc}"
        )
        context = ""
        return context

    if row is None:
        context = ""
        return context

    candidate = candidate_from_conversation_row(row, source="reply_context")
    if candidate is None:
        context = ""
        return context

    lookup_result = await build_past_dialog_cognition_context(
        [candidate],
        character_global_user_id=character_global_user_id,
    )
    context = lookup_result["past_dialog_cognition_context"]
    return context


async def _hydrate_reply_context(req: ChatRequest) -> ReplyContext:
    """Build service-facing reply context from typed adapter metadata.

    Args:
        req: Incoming chat request from an adapter.

    Returns:
        Compact reply context projected from ``message_envelope.reply``, with
        missing reply target metadata filled from delivered conversation rows
        when an exact platform message ID match exists.
    """

    reply_context = await brain_intake.hydrate_reply_context(req)
    reply_to_message_id = str(reply_context.get("reply_to_message_id") or "")
    has_complete_metadata = all(
        reply_context.get(key)
        for key in (
            "reply_to_platform_user_id",
            "reply_to_display_name",
            "reply_excerpt",
        )
    )
    needs_reply_attachments = not reply_context.get("reply_attachments")
    if reply_to_message_id and (not has_complete_metadata or needs_reply_attachments):
        row = await get_conversation_by_platform_message_id(
            platform=req.platform,
            platform_channel_id=req.platform_channel_id,
            platform_message_id=reply_to_message_id,
        )
        if row is not None:
            attachment_summaries = project_reply_attachment_summaries(
                row.get("attachments", []),
            )
            if attachment_summaries:
                reply_context["reply_attachments"] = attachment_summaries
            if not reply_context.get("reply_to_platform_user_id"):
                platform_user_id = str(row.get("platform_user_id") or "")
                if platform_user_id:
                    reply_context["reply_to_platform_user_id"] = platform_user_id
            if not reply_context.get("reply_to_display_name"):
                display_name = str(row.get("display_name") or "")
                if display_name:
                    reply_context["reply_to_display_name"] = display_name
            if not reply_context.get("reply_excerpt"):
                body_text = str(row.get("body_text") or "")
                if body_text:
                    reply_context["reply_excerpt"] = body_text

    return_value = _compact_reply_context(reply_context)
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
        now_func=storage_utc_now,
        logger=logger,
    )


# ── Lifespan ────────────────────────────────────────────────────────

_static_character_profile: dict = {}
_runtime_character_state: dict = {}
_graph = None
_adapter_registry: AdapterRegistry | None = None
_character_identity_backfilled: set[tuple[str, str, str]] = set()
_chat_input_queue = ChatInputQueue()
_pipeline_coordinator = PipelineCoordinator()
_chat_queue_worker_task: asyncio.Task | None = None
_calendar_worker_handle: CalendarSchedulerWorkerHandle | None = None
_reflection_worker_handle: ReflectionWorkerHandle | None = None
_self_cognition_worker_handle: SelfCognitionWorkerHandle | None = None
_background_work_worker_handle: BackgroundWorkRuntimeHandle | None = None
_primary_interaction_active_count = 0
_latest_cognition_graph: dict[str, Any] | None = None


def _clear_latest_cognition_graph() -> None:
    """Clear the process-local latest cognition graph snapshot."""

    global _latest_cognition_graph

    _latest_cognition_graph = None


def _record_latest_cognition_graph(cognition_graph: dict[str, Any] | None) -> None:
    """Store a bounded copy of the latest cognition graph snapshot."""

    global _latest_cognition_graph

    if cognition_graph is None:
        return
    _latest_cognition_graph = deepcopy(cognition_graph)


def _latest_cognition_graph_response() -> OpsLatestCognitionGraphResponse:
    """Build the read-only latest cognition graph API response."""

    response = OpsLatestCognitionGraphResponse(
        cognition_graph=deepcopy(_latest_cognition_graph),
    )
    return response


def _worker_task_alive(handle: object | None) -> bool:
    """Return whether a process-local worker handle owns a running task."""

    if handle is None:
        alive = False
        return alive

    task = getattr(handle, "task", None)
    if task is None:
        alive = False
        return alive

    done = getattr(task, "done", None)
    if not callable(done):
        alive = False
        return alive

    alive = not bool(done())
    return alive


async def _handle_calendar_reflection_phase_run(
    run: dict[str, Any],
) -> dict[str, Any]:
    """Execute one calendar-owned reflection phase slot with service deps."""

    result = await handle_reflection_phase_calendar_run(
        run,
        now=storage_utc_now(),
        dry_run=False,
        is_primary_interaction_busy=_reflection_cycle_primary_interaction_busy,
        adapter_registry_provider=lambda: _adapter_registry,
        pipeline_coordinator=_pipeline_coordinator,
    )
    return result


def _ops_runtime_status_payload(
    base_status: Mapping[str, object],
) -> dict[str, object]:
    """Merge aggregate runtime status with service-owned config and liveness."""

    raw_workers = base_status.get("workers", {})
    workers = raw_workers if isinstance(raw_workers, Mapping) else {}
    raw_process = base_status.get("process", {})
    process = raw_process if isinstance(raw_process, Mapping) else {}
    calendar_scheduler_worker = dict(workers.get("calendar_scheduler", {}))
    reflection_worker = dict(workers.get("reflection_cycle", {}))
    self_cognition_worker = dict(workers.get("self_cognition", {}))
    background_work_worker = dict(workers.get("background_work", {}))
    calendar_scheduler_worker.update({
        "enabled": CALENDAR_SCHEDULER_ENABLED,
        "task_alive": _worker_task_alive(_calendar_worker_handle),
    })
    reflection_worker.update({
        "enabled": REFLECTION_CYCLE_ENABLED,
        "task_alive": _worker_task_alive(_reflection_worker_handle),
    })
    self_cognition_worker.update({
        "enabled": SELF_COGNITION_ENABLED,
        "task_alive": _worker_task_alive(_self_cognition_worker_handle),
    })
    background_work_worker.update({
        "enabled": BACKGROUND_WORK_WORKER_ENABLED,
        "task_alive": _worker_task_alive(_background_work_worker_handle),
    })
    payload = {
        "status": str(base_status.get("status", "ok")),
        "generated_at": str(base_status.get("generated_at", "")),
        "window_hours": int(base_status.get("window_hours", 24)),
        "config": {
            "calendar_scheduler_enabled": CALENDAR_SCHEDULER_ENABLED,
            "calendar_scheduler_poll_interval_seconds": (
                CALENDAR_SCHEDULER_POLL_INTERVAL_SECONDS
            ),
            "calendar_scheduler_claim_limit": CALENDAR_SCHEDULER_CLAIM_LIMIT,
            "calendar_scheduler_lease_seconds": CALENDAR_SCHEDULER_LEASE_SECONDS,
            "calendar_scheduler_max_attempts": CALENDAR_SCHEDULER_MAX_ATTEMPTS,
            "reflection_cycle_enabled": REFLECTION_CYCLE_ENABLED,
            "self_cognition_enabled": SELF_COGNITION_ENABLED,
            "background_work_worker_enabled": BACKGROUND_WORK_WORKER_ENABLED,
            "reflection_worker_interval_seconds": (
                REFLECTION_WORKER_INTERVAL_SECONDS
            ),
            "reflection_phase_min_slot_spacing_seconds": (
                REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS
            ),
            "reflection_phase_max_slots_per_period": (
                REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD
            ),
            "reflection_phase_groups_per_slot": REFLECTION_PHASE_GROUPS_PER_SLOT,
            "self_cognition_worker_interval_seconds": (
                SELF_COGNITION_WORKER_INTERVAL_SECONDS
            ),
            "self_cognition_max_cases_per_tick": (
                SELF_COGNITION_MAX_CASES_PER_TICK
            ),
            "background_work_worker_interval_seconds": (
                BACKGROUND_WORK_WORKER_INTERVAL_SECONDS
            ),
            "background_work_worker_claim_limit": BACKGROUND_WORK_WORKER_CLAIM_LIMIT,
            "background_work_worker_lease_seconds": (
                BACKGROUND_WORK_WORKER_LEASE_SECONDS
            ),
            "background_work_worker_max_attempts": (
                BACKGROUND_WORK_WORKER_MAX_ATTEMPTS
            ),
            "background_work_input_char_limit": BACKGROUND_WORK_INPUT_CHAR_LIMIT,
            "background_work_output_char_limit": BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
        },
        "process": dict(process),
        "workers": {
            "calendar_scheduler": calendar_scheduler_worker,
            "reflection_cycle": reflection_worker,
            "self_cognition": self_cognition_worker,
            "background_work": background_work_worker,
        },
        "semantic_descriptors": dict(
            base_status.get("semantic_descriptors", {}),
        ),
    }
    return payload


def _ops_self_cognition_stats_payload(
    stats: Mapping[str, object],
) -> dict[str, object]:
    """Merge self-cognition aggregates with service-owned worker state."""

    payload = dict(stats)
    payload["enabled"] = SELF_COGNITION_ENABLED
    payload["task_alive"] = _worker_task_alive(_self_cognition_worker_handle)
    return payload


async def _refresh_runtime_character_state() -> None:
    """Replace the process-local runtime state with the current DB projection."""
    global _runtime_character_state

    runtime_state = await get_character_runtime_state()
    _runtime_character_state = runtime_state


async def _update_runtime_character_state_from_consolidation(
    consolidation_result: dict,
) -> None:
    """Merge persisted runtime-state fields from a consolidation result."""
    global _runtime_character_state

    runtime_update = {}
    for field_name in ("mood", "global_vibe", "reflection_summary"):
        field_value = consolidation_result.get(field_name)
        if isinstance(field_value, str) and field_value:
            runtime_update[field_name] = field_value

    if runtime_update:
        _runtime_character_state = {
            **_runtime_character_state,
            **runtime_update,
        }


def _primary_interaction_busy() -> bool:
    """Return whether chat or post-dialog work currently has priority."""

    return_value = (
        _primary_interaction_active_count > 0
        or _chat_input_queue.pending_count() > 0
    )
    return return_value


def _reflection_cycle_primary_interaction_busy() -> bool:
    """Return whether the standalone reflection worker should pause for chat."""

    return_value = False
    return return_value


async def _release_queued_pipeline_handle(item: QueuedChatItem) -> None:
    """Release a queued item's foreground coordination handle if present."""

    handle = item.pipeline_run_handle
    if handle is None:
        return

    item.pipeline_run_handle = None
    await handle.__aexit__(None, None, None)


def _current_character_profile_snapshot() -> dict:
    """Return the current composed character profile for background workers."""

    character_profile = compose_character_profile(
        _static_character_profile,
        _runtime_character_state,
        CHARACTER_GLOBAL_USER_ID,
    )
    return character_profile


def _background_artifact_result_text(episode: CognitiveEpisode) -> str:
    """Build a compact current-source summary for result-ready cognition."""

    percepts = episode.get("percepts", [])
    result_percept = {}
    for percept in percepts:
        if percept.get("input_source") not in (
            "background_artifact_result",
            "background_work_result",
            "accepted_task_result",
        ):
            continue
        result_percept = percept
        break

    input_source = str(result_percept.get("input_source", "")).strip()
    metadata = result_percept.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    work_kind = str(metadata.get("work_kind", "")).strip()
    objective = str(metadata.get("objective_summary", "")).strip()
    accepted_task_summary = str(
        metadata.get("accepted_task_summary", "")
    ).strip()
    failure_summary = str(metadata.get("failure_summary", "")).strip()
    status = "failed" if failure_summary else "completed"
    result_label = "Background artifact result"
    if input_source == "background_work_result":
        result_label = "Background work result"
    elif input_source == "accepted_task_result":
        result_label = "Accepted task result"
    summary_parts = [
        f"{result_label} is {status}.",
    ]
    if work_kind:
        summary_parts.append(f"Work kind: {work_kind}.")
    if objective:
        summary_parts.append(f"Objective: {objective}.")
    if accepted_task_summary:
        summary_parts.append(f"Task: {accepted_task_summary}.")
    result_text = " ".join(summary_parts)
    return result_text


def _background_result_metadata(episode: CognitiveEpisode) -> Mapping[str, object]:
    """Return metadata from the background result percept."""

    for percept in episode["percepts"]:
        if percept.get("input_source") not in (
            "background_artifact_result",
            "background_work_result",
            "accepted_task_result",
        ):
            continue
        metadata = percept.get("metadata", {})
        if isinstance(metadata, Mapping):
            return metadata
        break

    return_value: Mapping[str, object] = {}
    return return_value


def _background_artifact_prompt_message_context(
    episode: CognitiveEpisode,
) -> dict[str, object]:
    """Build prompt-safe message context for a result-ready source packet."""

    target_scope = episode["target_scope"]
    raw_addressed_ids = target_scope.get("target_addressed_user_ids", [])
    addressed_ids = [
        value
        for value in raw_addressed_ids
        if isinstance(value, str) and value.strip()
    ]
    context = {
        "body_text": _background_artifact_result_text(episode),
        "addressed_to_global_user_ids": addressed_ids,
        "broadcast": bool(target_scope.get("target_broadcast", False)),
        "mentions": [],
        "attachments": [],
    }
    return context


def _chat_delivery_mention_users(
    *,
    req: ChatRequest,
    global_user_id: str,
    message_envelope: MessageEnvelope,
    result: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Collect bounded user rows usable for outbound inline mention rendering."""

    users: list[dict[str, object]] = [
        {
            "display_name": req.display_name,
            "platform_user_id": req.platform_user_id,
            "global_user_id": global_user_id,
        }
    ]

    mentions = message_envelope.get("mentions", [])
    if isinstance(mentions, list):
        for mention in mentions:
            if not isinstance(mention, Mapping):
                continue
            if mention.get("entity_kind") != "user":
                continue
            users.append({
                "display_name": mention.get("display_name", ""),
                "platform_user_id": mention.get("platform_user_id", ""),
                "global_user_id": mention.get("global_user_id", ""),
            })

    reply = message_envelope.get("reply")
    if isinstance(reply, Mapping):
        users.append({
            "display_name": reply.get("display_name", ""),
            "platform_user_id": reply.get("platform_user_id", ""),
            "global_user_id": reply.get("global_user_id", ""),
        })

    scope_users = result.get("scope_users", [])
    if isinstance(scope_users, list):
        for scope_user in scope_users:
            if not isinstance(scope_user, Mapping):
                continue
            users.append({
                "display_name": scope_user.get("display_name", ""),
                "platform_user_id": scope_user.get("platform_user_id", ""),
                "global_user_id": scope_user.get("global_user_id", ""),
            })

    return users


def _background_artifact_delivery_mentions(
    *,
    result: Mapping[str, object],
    episode: CognitiveEpisode,
) -> list[dict[str, str]]:
    """Build inline mention candidates for result-ready dispatcher delivery."""

    target_scope = episode["target_scope"]
    users = [
        {
            "global_user_id": target_scope["current_global_user_id"],
            "display_name": target_scope["current_display_name"],
            "platform_user_id": target_scope["current_platform_user_id"],
        }
    ]
    final_dialog = [
        fragment
        for fragment in result.get("final_dialog", [])
        if isinstance(fragment, str)
    ]
    delivery_mentions = build_inline_delivery_mentions(
        text="\n".join(final_dialog),
        users=users,
        character_global_user_id=CHARACTER_GLOBAL_USER_ID,
    )
    return delivery_mentions


async def _run_background_artifact_result_post_turn(
    consolidation_state: dict,
    *,
    visible_response_sent: bool,
) -> None:
    """Run non-blocking post-turn consumers for delivered artifact results."""

    try:
        has_consolidation_state = bool(consolidation_state)
        is_consolidatable = has_consolidatable_output(consolidation_state)
        if visible_response_sent and has_consolidation_state:
            consolidation_state = await _run_post_turn_memory_lifecycle_background(
                consolidation_state,
            )
            await _run_conversation_progress_record_background(
                consolidation_state,
            )
        if is_consolidatable:
            await _run_consolidation_background(consolidation_state)
            await _run_internal_monologue_residue_record_background(
                consolidation_state,
            )
    except Exception as exc:
        logger.exception(
            f"Background artifact post-turn handling failed after delivery: {exc}"
        )


async def _run_background_work_result_post_turn(
    consolidation_state: dict,
    *,
    visible_response_sent: bool,
) -> None:
    """Run non-blocking post-turn consumers for delivered background work."""

    await _run_background_artifact_result_post_turn(
        consolidation_state,
        visible_response_sent=visible_response_sent,
    )


async def _deliver_background_artifact_result_episode(
    episode: CognitiveEpisode,
) -> dict[str, Any]:
    """Run result-ready cognition and deliver selected text through dispatcher."""

    adapter_registry = _adapter_registry
    if adapter_registry is None:
        return {
            "status": "failed",
            "reason": "adapter registry is unavailable",
        }

    target_scope = episode["target_scope"]
    platform = target_scope["platform"]
    platform_channel_id = target_scope["platform_channel_id"]
    channel_type = target_scope["channel_type"]
    requester_global_user_id = target_scope["current_global_user_id"]
    requester_platform_user_id = target_scope["current_platform_user_id"]
    requester_display_name = target_scope["current_display_name"]
    if not requester_global_user_id:
        return {
            "status": "failed",
            "reason": "requester global user id is missing",
        }
    if not platform_channel_id:
        return {
            "status": "failed",
            "reason": "delivery channel id is missing",
        }

    try:
        await _refresh_runtime_character_state()
        user_profile = await get_user_profile(requester_global_user_id)
        character_name = _static_character_profile.get("name", "Character")
        result_metadata = _background_result_metadata(episode)
        source_name = str(
            result_metadata.get("source_character_name", "")
        ).strip()
        if source_name:
            character_name = source_name
        source_platform_bot_id = str(
            result_metadata.get("source_platform_bot_id", "")
        ).strip()
        if not source_platform_bot_id:
            raise ValueError("source_platform_bot_id is required")
        character_global_user_id = await _ensure_character_global_identity(
            platform=platform,
            platform_bot_id=source_platform_bot_id,
            character_name=character_name,
        )
        character_profile = compose_character_profile(
            _static_character_profile,
            _runtime_character_state,
            character_global_user_id,
        )
        history = await get_conversation_history(
            platform=platform,
            platform_channel_id=platform_channel_id,
            limit=CONVERSATION_HISTORY_LIMIT,
        )
        chat_history_wide = trim_history_dict(history)
        chat_history_recent = chat_history_wide[-CHAT_HISTORY_RECENT_LIMIT:]
        try:
            promoted_reflection_context = await build_promoted_reflection_context()
        except Exception as exc:
            logger.exception(
                f"Promoted reflection context load failed for background "
                f"artifact result: {exc}"
            )
            promoted_reflection_context = {}

        trigger_source = str(episode["trigger_source"])
        is_background_work_result = trigger_source == "background_work_result_ready"
        is_accepted_task_result = trigger_source == "accepted_task_result_ready"
        bot_permission_role = "background_artifact_result"
        if is_background_work_result:
            bot_permission_role = "background_work_result"
        elif is_accepted_task_result:
            bot_permission_role = "accepted_task_result"

        debug_modes: DebugModes = {}
        if not COGNITION_VISUAL_DIRECTIVES_ENABLED:
            debug_modes["no_visual_directives"] = True
        initial_state: IMProcessState = {
            "storage_timestamp_utc": episode["storage_timestamp_utc"],
            "local_time_context": episode["local_time_context"],
            "llm_trace_id": llm_tracing.build_trace_id(),
            "platform": platform,
            "platform_message_id": episode["origin_metadata"][
                "platform_message_id"
            ],
            "active_turn_platform_message_ids": list(
                episode["origin_metadata"]["active_turn_platform_message_ids"]
            ),
            "active_turn_conversation_row_ids": list(
                episode["origin_metadata"]["active_turn_conversation_row_ids"]
            ),
            "platform_user_id": requester_platform_user_id,
            "global_user_id": requester_global_user_id,
            "user_name": requester_display_name,
            "user_input": _background_artifact_result_text(episode),
            "prompt_message_context": (
                _background_artifact_prompt_message_context(episode)
            ),
            "cognitive_episode": episode,
            "user_multimedia_input": [],
            "user_profile": user_profile,
            "platform_bot_id": source_platform_bot_id,
            "character_name": character_name,
            "character_profile": character_profile,
            "platform_channel_id": platform_channel_id,
            "channel_type": channel_type,
            "channel_name": "",
            "chat_history_wide": chat_history_wide,
            "chat_history_recent": chat_history_recent,
            "reply_context": {},
            "should_respond": True,
            "reason_to_respond": trigger_source,
            "use_reply_feature": False,
            "channel_topic": "",
            "indirect_speech_context": "",
            "debug_modes": debug_modes,
            "final_dialog": [],
            "target_addressed_user_ids": [requester_global_user_id],
            "target_broadcast": False,
            "future_promises": [],
            "consolidation_state": {},
            "promoted_reflection_context": promoted_reflection_context,
        }
        progress_update = await load_conversation_episode_state(initial_state)
        initial_state.update(progress_update)
        result = await persona_supervisor2(initial_state)
        final_dialog = [
            fragment
            for fragment in result.get("final_dialog", [])
            if isinstance(fragment, str) and fragment.strip()
        ]
        if not final_dialog:
            return {
                "status": "failed",
                "reason": "result-ready cognition selected no visible text",
            }

        dispatch_result = await handle_send_message(
            {
                "target_platform": platform,
                "target_channel": platform_channel_id,
                "target_channel_type": channel_type,
                "text": "\n".join(final_dialog),
                "reply_to_msg_id": None,
                "delivery_mentions": _background_artifact_delivery_mentions(
                    result=result,
                    episode=episode,
                ),
            },
            DispatchContext(
                source_platform=platform,
                source_channel_id=platform_channel_id,
                source_user_id=requester_global_user_id,
                source_message_id=episode["origin_metadata"][
                    "platform_message_id"
                ],
                guild_id=None,
                bot_permission_role=bot_permission_role,
                now=storage_utc_now(),
                source_channel_type=channel_type,
                source_platform_bot_id=source_platform_bot_id,
                source_character_name=character_name,
            ),
            adapter_registry,
        )
    except Exception as exc:
        logger.exception(
            f"Background artifact result delivery failed: {exc}"
        )
        return {
            "status": "failed",
            "reason": str(exc),
        }

    consolidation_state = result.get("consolidation_state")
    if isinstance(consolidation_state, dict):
        if not is_background_work_result and not is_accepted_task_result:
            await _run_background_artifact_result_post_turn(
                consolidation_state,
                visible_response_sent=True,
            )
        else:
            await _run_background_work_result_post_turn(
                consolidation_state,
                visible_response_sent=True,
            )
    delivery_result = {
        "status": "delivered",
        "conversation_message_id": dispatch_result["conversation_message_id"],
        "delivery_tracking_id": dispatch_result["delivery_tracking_id"],
        "adapter_message_id": dispatch_result["adapter_message_id"],
    }
    return delivery_result


async def _deliver_background_work_result_episode(
    episode: CognitiveEpisode,
) -> dict[str, Any]:
    """Run background-work result cognition and dispatcher delivery."""

    delivery_result = await _deliver_background_artifact_result_episode(episode)
    return delivery_result


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


def _active_turn_conversation_row_ids(item: QueuedChatItem) -> list[str]:
    """Build the persisted row ID list answered by one graph turn.

    Args:
        item: Surviving queued item that will run through the chat graph.

    Returns:
        Non-empty conversation-history row IDs from the survivor and collapsed
        follow-ups, deduplicated in arrival order.
    """

    return_value = brain_intake.active_turn_conversation_row_ids(item)
    return return_value


def _has_prompt_usable_multimedia_input(
    multimedia_input: list[MultiMediaDoc],
) -> bool:
    """Return whether current-turn media can produce graph input."""

    for item in multimedia_input:
        content_type = item.get("content_type", "")
        base64_data = item.get("base64_data", "")
        description = item.get("description", "")
        if (
            content_type.startswith("image/")
            and bool(base64_data or description.strip())
        ):
            return_value = True
            return return_value
        if content_type.startswith("audio/") and bool(description.strip()):
            return_value = True
            return return_value

    return_value = False
    return return_value


def _is_no_content_turn(
    *,
    user_input: str,
    combined_content: str | None,
    multimedia_input: list[MultiMediaDoc],
    media_description_rows: list[Mapping[str, object]],
) -> bool:
    """Return whether a persisted turn has no prompt-usable content."""

    if user_input.strip():
        return_value = False
        return return_value
    if combined_content is not None and combined_content.strip():
        return_value = False
        return return_value
    if _has_prompt_usable_multimedia_input(multimedia_input):
        return_value = False
        return return_value
    if media_description_rows:
        return_value = False
        return return_value

    return_value = True
    return return_value


def _build_text_chat_episode_ids(
    *,
    platform: str,
    platform_channel_id: str,
    platform_message_id: str,
    conversation_row_id: str | None,
    queue_sequence: int,
) -> tuple[str, str]:
    """Build deterministic episode and percept IDs for a text chat item.

    Args:
        platform: Source adapter platform name.
        platform_channel_id: Source platform channel ID, when available.
        platform_message_id: Source platform message ID, when available.
        conversation_row_id: Persisted conversation row ID, when available.
        queue_sequence: Process-local queue sequence for the surviving item.

    Returns:
        Episode ID and its single dialog-text percept ID.
    """

    message_reference = (
        platform_message_id or conversation_row_id or f"queue-{queue_sequence}"
    )
    channel_reference = platform_channel_id or "direct"
    episode_id = f"user_message:{platform}:{channel_reference}:{message_reference}"
    percept_id = f"{episode_id}:dialog_text:0"
    return_value = (episode_id, percept_id)
    return return_value


async def _save_user_message_from_item(
    item: QueuedChatItem,
    *,
    global_user_id: str,
    reply_context: ReplyContext,
    message_envelope: MessageEnvelope | None = None,
) -> str | None:
    """Persist one queued user message.

    Args:
        item: Queued chat item containing the request and storage UTC time.
        global_user_id: Resolved global user identifier.
        reply_context: Adapter-supplied reply metadata after compacting.
        message_envelope: Envelope after service-side identity resolution, when
            the caller already resolved it for graph input.

    Returns:
        Inserted conversation row ID string when committed; otherwise None.
    """

    return_value = await brain_intake.save_user_message_from_item(
        item,
        global_user_id=global_user_id,
        reply_context=reply_context,
        save_conversation_func=save_conversation,
        resolve_message_envelope_identities_func=_resolve_message_envelope_identities,
        message_envelope=message_envelope,
        logger=logger,
    )
    return return_value


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


async def _drop_queued_chat_item(item: QueuedChatItem) -> bool:
    """Persist and complete one pruned queued item without running the graph.

    Args:
        item: Queued chat item selected for pruning.

    Returns:
        True when the incoming row was committed, otherwise false.
    """

    correlation_id = _chat_correlation_id(item.request)
    scope = _service_event_scope(item.request)
    save_started_at = 0.0
    persistence_error: BaseException | None = None
    try:
        global_user_id, _ = await _resolve_queued_user(item)
        reply_context = await _hydrate_reply_context(item.request)
        save_started_at = time.perf_counter()
        conversation_row_id = await _save_user_message_from_item(
            item,
            global_user_id=global_user_id,
            reply_context=reply_context,
        )
        if not conversation_row_id:
            await event_logging.record_database_operation_event(
                component=SERVICE_COMPONENT,
                collection=CONVERSATION_HISTORY_COLLECTION,
                operation_kind="insert_user_message",
                status="failed",
                idempotency_result="not_committed",
                latency_ms=_elapsed_ms(save_started_at),
                correlation_id=correlation_id,
                severity="warning",
            )
            persistence_error = RuntimeError(
                "dropped queued message was not committed to "
                "conversation_history"
            )
    except Exception as exc:
        logger.exception(f"Failed to persist dropped queued message: {exc}")
        persistence_error = exc
        if save_started_at:
            await event_logging.record_database_operation_event(
                component=SERVICE_COMPONENT,
                collection=CONVERSATION_HISTORY_COLLECTION,
                operation_kind="insert_user_message",
                status="failed",
                idempotency_result="exception",
                latency_ms=_elapsed_ms(save_started_at),
                correlation_id=correlation_id,
                severity="warning",
            )
        (
            error_class,
            error_preview,
            stack_fingerprint,
            top_frame_module,
        ) = _runtime_error_fields(exc)
        await event_logging.record_runtime_error_event(
            component=SERVICE_COMPONENT,
            error_class=error_class,
            error_preview=error_preview,
            stack_fingerprint=stack_fingerprint,
            top_frame_module=top_frame_module,
            recovered=True,
            status="recovered",
            correlation_id=correlation_id,
            severity="warning",
        )

    if persistence_error is not None:
        _chat_input_queue.fail(item, persistence_error)
        await event_logging.record_queue_intake_event(
            component=SERVICE_COMPONENT,
            correlation_id=correlation_id,
            status="failed",
            queue_depth=_chat_input_queue.pending_count(),
            coalesced_count=0,
            dropped_count=1,
            protected_by_reply=_chat_input_queue.is_bot_reply(item),
            listen_only=item.request.debug_modes.listen_only,
            scope=scope,
            severity="error",
        )
        await _release_queued_pipeline_handle(item)
        return_value = False
        return return_value

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
    await event_logging.record_queue_intake_event(
        component=SERVICE_COMPONENT,
        correlation_id=correlation_id,
        status="dropped",
        queue_depth=_chat_input_queue.pending_count(),
        coalesced_count=0,
        dropped_count=1,
        protected_by_reply=_chat_input_queue.is_bot_reply(item),
        listen_only=item.request.debug_modes.listen_only,
        scope=scope,
        severity="warning",
    )
    await _release_queued_pipeline_handle(item)
    return_value = True
    return return_value


async def _persist_collapsed_queued_chat_item(
    item: QueuedChatItem,
    survivor: QueuedChatItem,
) -> bool:
    """Persist and complete one queued item collapsed into a surviving turn.

    Args:
        item: Queued chat item collapsed into another item.
        survivor: Surviving queued item that will receive the character reply.

    Returns:
        True when the collapsed row was committed, otherwise false.
    """

    correlation_id = _chat_correlation_id(item.request)
    scope = _service_event_scope(item.request)
    save_started_at = 0.0
    persistence_error: BaseException | None = None
    try:
        global_user_id, _ = await _resolve_queued_user(item)
        reply_context = await _hydrate_reply_context(item.request)
        save_started_at = time.perf_counter()
        conversation_row_id = await _save_user_message_from_item(
            item,
            global_user_id=global_user_id,
            reply_context=reply_context,
        )
        if conversation_row_id:
            item.conversation_row_id = conversation_row_id
        else:
            logger.warning(
                "Collapsed queued message persisted without conversation row "
                f"id: sequence={item.sequence}"
            )
            persistence_error = RuntimeError(
                "collapsed queued message was not committed to "
                "conversation_history"
            )
            await event_logging.record_database_operation_event(
                component=SERVICE_COMPONENT,
                collection=CONVERSATION_HISTORY_COLLECTION,
                operation_kind="insert_user_message",
                status="failed",
                idempotency_result="not_committed",
                latency_ms=_elapsed_ms(save_started_at),
                correlation_id=correlation_id,
                severity="warning",
            )
    except Exception as exc:
        logger.exception(f"Failed to persist collapsed queued message: {exc}")
        persistence_error = exc
        if save_started_at:
            await event_logging.record_database_operation_event(
                component=SERVICE_COMPONENT,
                collection=CONVERSATION_HISTORY_COLLECTION,
                operation_kind="insert_user_message",
                status="failed",
                idempotency_result="exception",
                latency_ms=_elapsed_ms(save_started_at),
                correlation_id=correlation_id,
                severity="warning",
            )
        (
            error_class,
            error_preview,
            stack_fingerprint,
            top_frame_module,
        ) = _runtime_error_fields(exc)
        await event_logging.record_runtime_error_event(
            component=SERVICE_COMPONENT,
            error_class=error_class,
            error_preview=error_preview,
            stack_fingerprint=stack_fingerprint,
            top_frame_module=top_frame_module,
            recovered=True,
            status="recovered",
            correlation_id=correlation_id,
            severity="warning",
        )

    if persistence_error is not None:
        _chat_input_queue.fail(item, persistence_error)
        await event_logging.record_queue_intake_event(
            component=SERVICE_COMPONENT,
            correlation_id=correlation_id,
            status="failed",
            queue_depth=_chat_input_queue.pending_count(),
            coalesced_count=1,
            dropped_count=0,
            protected_by_reply=_chat_input_queue.is_bot_reply(item),
            listen_only=item.request.debug_modes.listen_only,
            scope=scope,
            severity="error",
        )
        await _release_queued_pipeline_handle(item)
        return_value = False
        return return_value

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
    await event_logging.record_queue_intake_event(
        component=SERVICE_COMPONENT,
        correlation_id=correlation_id,
        status="collapsed",
        queue_depth=_chat_input_queue.pending_count(),
        coalesced_count=1,
        dropped_count=0,
        protected_by_reply=_chat_input_queue.is_bot_reply(item),
        listen_only=item.request.debug_modes.listen_only,
        scope=scope,
    )
    await _release_queued_pipeline_handle(item)
    return_value = True
    return return_value


def _build_response_cognition_graph(
    *,
    graph_result: Mapping[str, Any],
    consolidation_state: Mapping[str, Any] | None,
    run_id: str,
) -> dict[str, Any]:
    """Build a bounded cognition graph snapshot for operator inspection."""

    state = consolidation_state or {}
    should_respond = graph_result.get("should_respond")
    final_dialog = graph_result.get("final_dialog")
    final_dialog_count = len(final_dialog) if isinstance(final_dialog, list) else 0
    future_promises = graph_result.get("future_promises")
    future_promise_count = (
        len(future_promises) if isinstance(future_promises, list) else 0
    )
    action_spec_count = _safe_sequence_count(state.get("action_specs"))
    action_result_count = _safe_sequence_count(state.get("action_results"))
    memory_status = "completed" if state.get("rag_result") else "skipped"
    action_status = (
        "completed"
        if action_spec_count or action_result_count or future_promise_count
        else "skipped"
    )
    reasoning_status = (
        "completed"
        if any(
            state.get(key)
            for key in (
                "internal_monologue",
                "logical_stance",
                "character_intent",
                "judgment_note",
            )
        )
        else "skipped"
    )

    nodes = [
        {
            "id": "intake",
            "label": "Queued turn",
            "stage": "L1",
            "lane": "input",
            "column": 1,
            "branch": "input",
            "status": "completed",
            "detail": {
                "summary": "Accepted by the brain input queue.",
                "status": _safe_graph_text(state.get("platform", "")),
            },
        },
        {
            "id": "l1.relevance",
            "label": "Response decision",
            "stage": "L1",
            "lane": "gate",
            "column": 2,
            "branch": "decision",
            "status": "completed",
            "detail": {
                "decision": _safe_graph_text(should_respond),
                "reasoning": _safe_graph_text(graph_result.get("reason_to_respond")),
            },
        },
        {
            "id": "l2.reasoning",
            "label": "Reasoning",
            "stage": "L2",
            "lane": "cognition",
            "column": 3,
            "branch": "reasoning",
            "status": reasoning_status,
            "detail": {
                "internal_monologue": _safe_graph_text(
                    state.get("internal_monologue"),
                ),
                "logical_stance": _safe_graph_text(state.get("logical_stance")),
                "character_intent": _safe_graph_text(state.get("character_intent")),
                "judgment_note": _safe_graph_text(state.get("judgment_note")),
            },
        },
        {
            "id": "l2.memory",
            "label": "Memory and evidence",
            "stage": "L2",
            "lane": "memory",
            "column": 3,
            "branch": "memory",
            "status": memory_status,
            "detail": {
                "summary": _memory_graph_summary(state.get("rag_result")),
                "status": memory_status,
            },
        },
        {
            "id": "l2.actions",
            "label": "Actions",
            "stage": "L2",
            "lane": "action",
            "column": 3,
            "branch": "action",
            "status": action_status,
            "detail": {
                "summary": (
                    f"{action_spec_count} action spec(s), "
                    f"{action_result_count} action result(s), "
                    f"{future_promise_count} follow-up(s)."
                ),
            },
        },
        {
            "id": "l3.surface",
            "label": "Visible surface",
            "stage": "L3",
            "lane": "surface",
            "column": 4,
            "branch": "dialog",
            "status": "completed" if final_dialog_count else "skipped",
            "detail": {
                "summary": f"{final_dialog_count} visible message(s) returned.",
            },
        },
    ]
    edges = [
        {"source": "intake", "target": "l1.relevance", "kind": "sequence"},
        {"source": "l1.relevance", "target": "l2.reasoning", "kind": "fork"},
        {"source": "l1.relevance", "target": "l2.memory", "kind": "fork"},
        {"source": "l1.relevance", "target": "l2.actions", "kind": "fork"},
        {"source": "l2.reasoning", "target": "l3.surface", "kind": "join"},
        {"source": "l2.memory", "target": "l3.surface", "kind": "join"},
        {"source": "l2.actions", "target": "l3.surface", "kind": "join"},
    ]
    snapshot = {
        "run_id": run_id,
        "status": "completed",
        "nodes": nodes,
        "edges": edges,
        "redaction": {
            "detail": "bounded graph-result and consolidation-state fields only",
            "excluded": [
                "prompts",
                "embeddings",
                "raw messages",
                "message envelopes",
                "raw user input",
            ],
        },
    }
    return snapshot


async def _publish_self_cognition_latest_graph(
    artifact_payloads: dict[str, Any],
) -> None:
    """Record a completed self-cognition run as the latest graph snapshot."""

    cognition_graph = _build_self_cognition_cognition_graph(artifact_payloads)
    if cognition_graph is None:
        return
    _record_latest_cognition_graph(cognition_graph)


def _build_self_cognition_cognition_graph(
    artifact_payloads: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build a bounded graph snapshot from self-cognition artifacts."""

    run_record = artifact_payloads.get(self_cognition_models.ARTIFACT_RUN_RECORD)
    if not isinstance(run_record, Mapping):
        return None
    cognition_output = artifact_payloads.get(
        self_cognition_models.ARTIFACT_COGNITION_OUTPUT,
    )
    if not isinstance(cognition_output, Mapping):
        cognition_output = {}
    route_effect = artifact_payloads.get(
        self_cognition_models.ARTIFACT_ROUTE_EFFECT,
    )
    if not isinstance(route_effect, Mapping):
        route_effect = {}
    action_attempt = artifact_payloads.get(
        self_cognition_models.ARTIFACT_ACTION_ATTEMPT,
    )
    if not isinstance(action_attempt, Mapping):
        action_attempt = {}
    consolidation_outcome = artifact_payloads.get(
        self_cognition_models.ARTIFACT_CONSOLIDATION_OUTCOME,
    )
    if not isinstance(consolidation_outcome, Mapping):
        consolidation_outcome = {}

    selected_route = _safe_graph_text(run_record.get("selected_route"))
    output_mode = _safe_graph_text(run_record.get("output_mode"))
    run_status = _graph_status(run_record.get("status"))
    has_dialog = bool(route_effect.get("visible_dialog"))
    has_action = bool(action_attempt)
    has_consolidation = bool(consolidation_outcome)
    nodes = [
        {
            "id": "self.source",
            "label": "Source case",
            "stage": "L1",
            "lane": "input",
            "column": 1,
            "branch": "source",
            "status": "completed",
            "detail": {
                "summary": "Accepted by the self-cognition worker.",
                "status": _safe_graph_text(run_record.get("trigger_kind")),
            },
        },
        {
            "id": "self.reasoning",
            "label": "Reasoning",
            "stage": "L2",
            "lane": "cognition",
            "column": 2,
            "branch": "reasoning",
            "status": run_status,
            "detail": {
                "internal_monologue": _safe_graph_text(
                    cognition_output.get("internal_monologue"),
                ),
                "logical_stance": _safe_graph_text(
                    cognition_output.get("logical_stance"),
                ),
                "character_intent": _safe_graph_text(
                    cognition_output.get("character_intent"),
                ),
            },
        },
        {
            "id": "self.route",
            "label": "Route decision",
            "stage": "L2",
            "lane": "decision",
            "column": 3,
            "branch": "route",
            "status": run_status,
            "detail": {
                "decision": selected_route,
                "status": output_mode,
            },
        },
        {
            "id": "self.action",
            "label": "Action output",
            "stage": "L3",
            "lane": "action",
            "column": 4,
            "branch": "action",
            "status": "completed" if has_action else "skipped",
            "detail": {
                "summary": "Action candidate recorded."
                if has_action
                else "No action candidate recorded.",
            },
        },
        {
            "id": "self.surface",
            "label": "Visible surface",
            "stage": "L3",
            "lane": "surface",
            "column": 4,
            "branch": "dialog",
            "status": "completed" if has_dialog else "skipped",
            "detail": {
                "summary": "Visible self-cognition output selected."
                if has_dialog
                else "No visible output selected.",
            },
        },
        {
            "id": "self.consolidation",
            "label": "Consolidation",
            "stage": "L4",
            "lane": "memory",
            "column": 5,
            "branch": "memory",
            "status": "completed" if has_consolidation else "skipped",
            "detail": {
                "summary": "Completed episode consolidation reported."
                if has_consolidation
                else "No consolidation outcome reported.",
            },
        },
    ]
    edges = [
        {"source": "self.source", "target": "self.reasoning", "kind": "sequence"},
        {"source": "self.reasoning", "target": "self.route", "kind": "sequence"},
        {"source": "self.route", "target": "self.action", "kind": "fork"},
        {"source": "self.route", "target": "self.surface", "kind": "fork"},
        {"source": "self.action", "target": "self.consolidation", "kind": "join"},
        {"source": "self.surface", "target": "self.consolidation", "kind": "join"},
    ]
    snapshot = {
        "run_id": _safe_graph_text(run_record.get("run_id"), max_chars=120),
        "status": run_status if run_status in {"completed", "failed"} else "partial",
        "nodes": nodes,
        "edges": edges,
        "redaction": {
            "detail": "bounded self-cognition artifact fields only",
            "excluded": [
                "prompts",
                "embeddings",
                "raw messages",
                "message envelopes",
                "raw source packet",
            ],
        },
    }
    return snapshot


def _graph_status(value: Any) -> str:
    """Return a graph-safe status value."""

    text = _safe_graph_text(value)
    if text in {"pending", "running", "completed", "skipped", "failed"}:
        return text
    return "completed" if text else "not_reported"


def _safe_sequence_count(value: Any) -> int:
    """Return a sequence count for graph metadata, or zero."""

    if isinstance(value, list):
        return len(value)
    return 0


def _safe_graph_text(value: Any, *, max_chars: int = 240) -> str:
    """Return a bounded scalar value for operator graph details."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if len(text) > max_chars:
        text = f"{text[:max_chars]}..."
    return text


def _memory_graph_summary(value: Any) -> str:
    """Summarize memory/RAG activity without exposing retrieved content."""

    if not isinstance(value, Mapping) or not value:
        return "No retrieved evidence reported."
    keys = sorted(str(key) for key in value.keys())[:6]
    return f"RAG data reported: {', '.join(keys)}."


async def _process_queued_chat_item(item: QueuedChatItem) -> None:
    """Run one queued item through the existing chat graph and post-writes.

    Args:
        item: Oldest surviving queued item selected by the worker.

    Returns:
        None.
    """

    req = item.request
    character_name = _static_character_profile.get("name", "Character")
    correlation_id = _chat_correlation_id(req)
    llm_trace_id = llm_tracing.build_trace_id()
    scope = _service_event_scope(req)
    turn_started_at = time.perf_counter()
    stages_reached: list[str] = []
    scheduled_followup_count = 0
    debug_mode_names: list[str] = []

    try:
        character_global_user_id = await _ensure_character_global_identity(
            platform=req.platform,
            platform_bot_id=req.platform_bot_id,
            character_name=character_name,
        )
        global_user_id, user_profile = await _resolve_queued_user(item)
        message_envelope = await _resolve_message_envelope_identities(req)
        stages_reached.append("identity")
        await llm_tracing.ensure_llm_trace_run(
            trace_id=llm_trace_id,
            platform=req.platform,
            platform_channel_id=req.platform_channel_id,
            channel_type=req.channel_type,
            platform_message_id=req.platform_message_id,
            global_user_id=global_user_id,
            started_at=item.storage_timestamp_utc,
        )

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
                base64_data = attachment.get("base64_data", "")
                description = attachment.get("description", "")
                if not isinstance(media_type, str):
                    continue
                if not isinstance(base64_data, str):
                    base64_data = ""
                if not isinstance(description, str):
                    description = ""
                has_image_payload = (
                    media_type.startswith("image/")
                    and bool(base64_data or description)
                )
                has_audio_description = (
                    media_type.startswith("audio/")
                    and bool(description)
                )
                if not has_image_payload and not has_audio_description:
                    continue
                multimedia_input.append({
                    "content_type": media_type,
                    "base64_data": base64_data,
                    "description": description,
                })

        history = await get_conversation_history(
            platform=req.platform,
            platform_channel_id=req.platform_channel_id,
            limit=CONVERSATION_HISTORY_LIMIT,
        )
        chat_history_wide = trim_history_dict(history)
        chat_history_recent = chat_history_wide[-CHAT_HISTORY_RECENT_LIMIT:]
        reply_context = await _hydrate_reply_context(req)
        stages_reached.append("history_loaded")
        user_save_started_at = time.perf_counter()
        try:
            conversation_row_id = await _save_user_message_from_item(
                item,
                global_user_id=global_user_id,
                reply_context=reply_context,
                message_envelope=message_envelope,
            )
        except Exception as exc:
            await event_logging.record_database_operation_event(
                component=SERVICE_COMPONENT,
                collection=CONVERSATION_HISTORY_COLLECTION,
                operation_kind="insert_user_message",
                status="failed",
                idempotency_result=f"exception:{exc.__class__.__name__}",
                latency_ms=_elapsed_ms(user_save_started_at),
                correlation_id=correlation_id,
                severity="warning",
            )
            raise

        if conversation_row_id:
            item.conversation_row_id = conversation_row_id
        else:
            await event_logging.record_database_operation_event(
                component=SERVICE_COMPONENT,
                collection=CONVERSATION_HISTORY_COLLECTION,
                operation_kind="insert_user_message",
                status="failed",
                idempotency_result="not_committed",
                latency_ms=_elapsed_ms(user_save_started_at),
                correlation_id=correlation_id,
                severity="warning",
            )
            logger.warning(
                "Surviving queued message did not commit conversation row; "
                "suppressing graph processing: "
                f"sequence={item.sequence}"
            )
            _chat_input_queue.fail(
                item,
                RuntimeError(
                    "surviving queued message was not committed to "
                    "conversation_history"
                ),
            )
            stages_reached.append("user_persist_failed")
            await event_logging.record_pipeline_turn_event(
                component=SERVICE_COMPONENT,
                correlation_id=correlation_id,
                status="failed",
                queue_wait_ms=_queue_wait_ms(item),
                stages_reached=stages_reached,
                final_outcome="user_persist_failed",
                scheduled_followups=0,
                debug_modes=debug_mode_names,
                scope=scope,
                duration_ms=_elapsed_ms(turn_started_at),
                severity="error",
            )
            return
        stages_reached.append("user_persisted")

        debug_modes: DebugModes = {
            "listen_only": req.debug_modes.listen_only,
            "think_only": req.debug_modes.think_only,
            "no_remember": req.debug_modes.no_remember,
        }
        if not COGNITION_VISUAL_DIRECTIVES_ENABLED:
            debug_modes["no_visual_directives"] = True

        initial_should_respond = not debug_modes["listen_only"]
        active_flags = [key for key, value in debug_modes.items() if value]
        debug_mode_names = active_flags
        if active_flags:
            logger.info(f'Debug modes active: {active_flags}')

        user_input = item.combined_content or message_envelope["body_text"]
        media_description_rows = [
            *build_text_chat_media_description_rows(multimedia_input),
            *build_reply_media_description_rows(reply_context),
        ]
        if _is_no_content_turn(
            user_input=user_input,
            combined_content=item.combined_content,
            multimedia_input=multimedia_input,
            media_description_rows=media_description_rows,
        ):
            logger.info(
                "No-content chat turn persisted without graph invocation: "
                f"platform={req.platform} "
                f'channel={req.platform_channel_id or "<dm>"} '
                f'message={req.platform_message_id or "<none>"}'
            )
            _chat_input_queue.complete(item, ChatResponse())
            stages_reached.append("no_content")
            stages_reached.append("response_completed")
            await event_logging.record_pipeline_turn_event(
                component=SERVICE_COMPONENT,
                correlation_id=correlation_id,
                status="completed",
                queue_wait_ms=_queue_wait_ms(item),
                stages_reached=stages_reached,
                final_outcome="no_content",
                scheduled_followups=0,
                debug_modes=debug_mode_names,
                scope=scope,
                duration_ms=_elapsed_ms(turn_started_at),
                severity="info",
            )
            return

        prompt_message_context = project_prompt_message_context(
            message_envelope=message_envelope,
            multimedia_input=multimedia_input,
            reply_context=reply_context,
        )
        is_collapsed_turn = bool(item.collapsed_items)
        active_turn_platform_message_ids = _active_turn_platform_message_ids(item)
        active_turn_conversation_row_ids = _active_turn_conversation_row_ids(item)
        await _refresh_runtime_character_state()
        character_profile = compose_character_profile(
            _static_character_profile,
            _runtime_character_state,
            character_global_user_id,
        )

        logger.debug(f'Chat request: platform={req.platform} channel={req.platform_channel_id or "<dm>"} message={req.platform_message_id or "<none>"} user={req.platform_user_id} global_user={global_user_id} content_type={req.content_type} attachments={len(message_envelope["attachments"])} media_attachments={len(multimedia_input)} history_wide={len(chat_history_wide)} history_recent={len(chat_history_recent)} reply_context={log_preview(reply_context)} debug_modes={active_flags} collapsed={is_collapsed_turn} collapsed_count={len(item.collapsed_items)} content={log_preview(user_input)}')

        local_time_context = item.local_time_context
        try:
            promoted_reflection_context = await build_promoted_reflection_context()
        except Exception as exc:
            logger.exception(f"Promoted reflection context load failed: {exc}")
            (
                error_class,
                error_preview,
                stack_fingerprint,
                top_frame_module,
            ) = _runtime_error_fields(exc)
            await event_logging.record_runtime_error_event(
                component=SERVICE_COMPONENT,
                error_class=error_class,
                error_preview=error_preview,
                stack_fingerprint=stack_fingerprint,
                top_frame_module=top_frame_module,
                recovered=True,
                status="recovered",
                correlation_id=correlation_id,
                severity="warning",
            )
            promoted_reflection_context = {}

        episode_id, percept_id = _build_text_chat_episode_ids(
            platform=req.platform,
            platform_channel_id=req.platform_channel_id,
            platform_message_id=req.platform_message_id,
            conversation_row_id=item.conversation_row_id or None,
            queue_sequence=item.sequence,
        )
        episode_output_mode: Literal["silent", "think_only", "visible_reply"]
        if debug_modes["listen_only"]:
            episode_output_mode = "silent"
        elif debug_modes["think_only"]:
            episode_output_mode = "think_only"
        else:
            episode_output_mode = "visible_reply"
        episode: CognitiveEpisode = build_text_chat_cognitive_episode(
            episode_id=episode_id,
            percept_id=percept_id,
            storage_timestamp_utc=item.storage_timestamp_utc,
            local_time_context=local_time_context,
            user_input=user_input,
            platform=req.platform,
            platform_channel_id=req.platform_channel_id,
            channel_type=req.channel_type,
            platform_message_id=req.platform_message_id,
            platform_user_id=req.platform_user_id,
            global_user_id=global_user_id,
            user_name=req.display_name,
            active_turn_platform_message_ids=active_turn_platform_message_ids,
            active_turn_conversation_row_ids=active_turn_conversation_row_ids,
            debug_modes=debug_modes,
            output_mode=episode_output_mode,
            target_addressed_user_ids=list(
                prompt_message_context["addressed_to_global_user_ids"]
            ),
            target_broadcast=bool(prompt_message_context["broadcast"]),
            media_description_rows=media_description_rows,
        )
        stages_reached.append("episode_built")

        initial_state: IMProcessState = {
            "storage_timestamp_utc": item.storage_timestamp_utc,
            "local_time_context": local_time_context,
            "llm_trace_id": llm_trace_id,
            "platform": req.platform,
            "platform_message_id": req.platform_message_id,
            "active_turn_platform_message_ids": active_turn_platform_message_ids,
            "active_turn_conversation_row_ids": active_turn_conversation_row_ids,
            "platform_user_id": req.platform_user_id,
            "global_user_id": global_user_id,
            "user_name": req.display_name,
            "user_input": user_input,
            "message_envelope": message_envelope,
            "prompt_message_context": prompt_message_context,
            "cognitive_episode": episode,
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
            "should_respond": initial_should_respond,
            "reason_to_respond": "",
            "use_reply_feature": False,
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

        try:
            result = await _graph.ainvoke(initial_state)
        except Exception as exc:
            logger.exception(f"Graph invocation failed: {exc}")
            fallback_text = (
                f"{character_name} is busy right now, please try again later."
            )
            delivery_tracking_id = uuid4().hex
            fallback_result = {
                "platform": req.platform,
                "platform_channel_id": req.platform_channel_id,
                "channel_type": req.channel_type,
                "platform_bot_id": req.platform_bot_id,
                "character_name": character_name,
                "global_user_id": global_user_id,
                "final_dialog": [fallback_text],
                "target_addressed_user_ids": [global_user_id],
                "target_broadcast": False,
                "delivery_tracking_id": delivery_tracking_id,
                "llm_trace_id": llm_trace_id,
            }
            try:
                await _save_assistant_message(fallback_result)
                response = ChatResponse(
                    messages=[fallback_text],
                    delivery_tracking_id=delivery_tracking_id,
                )
                stages_reached.append("assistant_persisted")
            except Exception as save_exc:
                logger.exception(
                    "Graph failure fallback suppressed because assistant "
                    f"history persistence failed: {save_exc}"
                )
                response = ChatResponse()
            _chat_input_queue.complete(item, response)
            (
                error_class,
                error_preview,
                stack_fingerprint,
                top_frame_module,
            ) = _runtime_error_fields(exc)
            await event_logging.record_runtime_error_event(
                component=SERVICE_COMPONENT,
                error_class=error_class,
                error_preview=error_preview,
                stack_fingerprint=stack_fingerprint,
                top_frame_module=top_frame_module,
                recovered=True,
                status="failed",
                correlation_id=correlation_id,
            )
            await event_logging.record_pipeline_turn_event(
                component=SERVICE_COMPONENT,
                correlation_id=correlation_id,
                status="failed",
                queue_wait_ms=_queue_wait_ms(item),
                stages_reached=stages_reached,
                final_outcome="graph_error",
                scheduled_followups=0,
                debug_modes=debug_mode_names,
                scope=scope,
                duration_ms=_elapsed_ms(turn_started_at),
                severity="error",
            )
            await llm_tracing.finalize_llm_trace_run(
                trace_id=llm_trace_id,
                status="failed",
                final_dialog_count=1,
                delivery_tracking_id=delivery_tracking_id,
            )
            return

        stages_reached.append("graph")
        final_dialog = result["final_dialog"]
        use_reply_feature = bool(final_dialog) and bool(
            result["use_reply_feature"]
        )
        consolidation_state = result["consolidation_state"]
        scheduled_followup_count = len(result["future_promises"])

        logger.debug(f'Chat result: platform={req.platform} channel={req.platform_channel_id or "<dm>"} message={req.platform_message_id or "<none>"} user={req.platform_user_id} should_respond={result["should_respond"]} use_reply_feature={use_reply_feature} final_dialog_count={len(final_dialog)} future_promises={len(result["future_promises"])} final_dialog={log_list_preview(final_dialog)}')

        consolidation_state_dict: dict | None = None
        if isinstance(consolidation_state, Mapping):
            consolidation_state_dict = dict(consolidation_state)

        has_consolidation_state = bool(consolidation_state_dict)
        is_consolidatable = (
            has_consolidatable_output(consolidation_state_dict)
            if consolidation_state_dict is not None
            else False
        )
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
        elif is_consolidatable and has_consolidation_state:
            should_consolidate = True
            logger.debug(f'Background consolidation queued: platform={req.platform} channel={req.platform_channel_id or "<dm>"} message={req.platform_message_id or "<none>"}')
        elif not is_consolidatable:
            logger.info(f'Background consolidation skipped: platform={req.platform} channel={req.platform_channel_id or "<dm>"} message={req.platform_message_id or "<none>"} should_respond={result["should_respond"]} consolidatable_output=false final_dialog_count={len(final_dialog)}')
        else:
            logger.warning(f'Background consolidation skipped: unexpected consolidation_state type={type(consolidation_state).__name__}')

        should_save_assistant_message = bool(final_dialog)
        response_dialog = final_dialog
        if debug_modes.get("think_only"):
            logger.info(f'think_only active — suppressing {len(final_dialog)} dialog message(s) from user output')
            response_dialog = []

        delivery_mentions: list[dict[str, str]] = []
        if response_dialog:
            delivery_mention_users = _chat_delivery_mention_users(
                req=req,
                global_user_id=global_user_id,
                message_envelope=message_envelope,
                result=result,
            )
            delivery_mentions = build_inline_delivery_mentions(
                text="\n".join(response_dialog),
                users=delivery_mention_users,
                character_global_user_id=CHARACTER_GLOBAL_USER_ID,
            )
        result["delivery_mentions"] = delivery_mentions

        delivery_tracking_id = ""
        if response_dialog and should_save_assistant_message:
            delivery_tracking_id = uuid4().hex
            result["delivery_tracking_id"] = delivery_tracking_id
        result["llm_trace_id"] = llm_trace_id

        cognition_graph = _build_response_cognition_graph(
            graph_result=result,
            consolidation_state=consolidation_state_dict,
            run_id=delivery_tracking_id or correlation_id,
        )
        _record_latest_cognition_graph(cognition_graph)
        response = ChatResponse(
            messages=response_dialog,
            content_type="text",
            attachments=[],
            use_reply_feature=use_reply_feature,
            delivery_mentions=delivery_mentions if response_dialog else [],
            scheduled_followups=0,
            delivery_tracking_id=delivery_tracking_id,
            cognition_graph=cognition_graph,
        )

        if should_save_assistant_message:
            assistant_save_started_at = time.perf_counter()
            try:
                await _save_assistant_message(result)
            except Exception as exc:
                await event_logging.record_database_operation_event(
                    component=SERVICE_COMPONENT,
                    collection=CONVERSATION_HISTORY_COLLECTION,
                    operation_kind="insert_assistant_message",
                    status="failed",
                    idempotency_result=f"exception:{exc.__class__.__name__}",
                    latency_ms=_elapsed_ms(assistant_save_started_at),
                    document_ref=delivery_tracking_id,
                    correlation_id=correlation_id,
                    severity="warning",
                )
                _chat_input_queue.complete(item, ChatResponse())
                stages_reached.append("assistant_persist_failed")
                await event_logging.record_pipeline_turn_event(
                    component=SERVICE_COMPONENT,
                    correlation_id=correlation_id,
                    status="failed",
                    queue_wait_ms=_queue_wait_ms(item),
                    stages_reached=stages_reached,
                    final_outcome="assistant_persist_failed",
                    scheduled_followups=scheduled_followup_count,
                    debug_modes=debug_mode_names,
                    scope=scope,
                    duration_ms=_elapsed_ms(turn_started_at),
                    severity="error",
                )
                await llm_tracing.finalize_llm_trace_run(
                    trace_id=llm_trace_id,
                    status="failed",
                    final_dialog_count=len(final_dialog),
                    delivery_tracking_id=delivery_tracking_id,
                )
                return

            stages_reached.append("assistant_persisted")
        _chat_input_queue.complete(item, response)
        stages_reached.append("response_completed")
        visible_response_sent = bool(response_dialog)
        think_only_suppressed = (
            bool(final_dialog)
            and debug_modes.get("think_only")
            and not visible_response_sent
        )
        if (
            not debug_modes.get("no_remember")
            and visible_response_sent
            and not think_only_suppressed
            and consolidation_state_dict is not None
        ):
            consolidation_state_dict = (
                await _run_post_turn_memory_lifecycle_background(
                    consolidation_state_dict,
                )
            )
        if should_record_progress and consolidation_state_dict is not None:
            await _run_conversation_progress_record_background(
                consolidation_state_dict,
            )
            stages_reached.append("progress_recorded")
        if should_consolidate and consolidation_state_dict is not None:
            await _run_consolidation_background(consolidation_state_dict)
            stages_reached.append("consolidation_recorded")
        if (
            not debug_modes.get("no_remember")
            and is_consolidatable
            and consolidation_state_dict is not None
        ):
            await _run_internal_monologue_residue_record_background(
                consolidation_state_dict,
            )
            stages_reached.append("residue_recorded")
        await llm_tracing.finalize_llm_trace_run(
            trace_id=llm_trace_id,
            status="succeeded",
            final_dialog_count=len(final_dialog),
            delivery_tracking_id=delivery_tracking_id,
        )
    except Exception as exc:
        logger.exception(f"Queued chat item failed: {exc}")
        _chat_input_queue.fail(item, exc)
        (
            error_class,
            error_preview,
            stack_fingerprint,
            top_frame_module,
        ) = _runtime_error_fields(exc)
        await event_logging.record_runtime_error_event(
            component=SERVICE_COMPONENT,
            error_class=error_class,
            error_preview=error_preview,
            stack_fingerprint=stack_fingerprint,
            top_frame_module=top_frame_module,
            recovered=True,
            status="failed",
            correlation_id=correlation_id,
        )
        await event_logging.record_pipeline_turn_event(
            component=SERVICE_COMPONENT,
            correlation_id=correlation_id,
            status="failed",
            queue_wait_ms=_queue_wait_ms(item),
            stages_reached=stages_reached,
            final_outcome="runtime_error",
            scheduled_followups=scheduled_followup_count,
            debug_modes=debug_mode_names,
            scope=scope,
            duration_ms=_elapsed_ms(turn_started_at),
            severity="error",
        )
        await llm_tracing.finalize_llm_trace_run(
            trace_id=llm_trace_id,
            status="failed",
            final_dialog_count=0,
            delivery_tracking_id="",
        )
    finally:
        await _release_queued_pipeline_handle(item)


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
            incoming_history_committed = True
            for dropped_item in dequeued_turn.dropped_items:
                dropped_committed = await _drop_queued_chat_item(dropped_item)
                incoming_history_committed = (
                    incoming_history_committed and dropped_committed
                )

            # Persist collapsed rows before the survivor builds active-turn
            # identity filters for RAG evidence.
            for collapsed_item, survivor in dequeued_turn.collapsed_items:
                collapsed_committed = await _persist_collapsed_queued_chat_item(
                    collapsed_item,
                    survivor,
                )
                incoming_history_committed = (
                    incoming_history_committed and collapsed_committed
                )

            if not incoming_history_committed:
                if dequeued_turn.next_item is not None:
                    _chat_input_queue.fail(
                        dequeued_turn.next_item,
                        RuntimeError(
                            "queued turn aborted because incoming history "
                            "persistence failed"
                        ),
                    )
                    await _release_queued_pipeline_handle(
                        dequeued_turn.next_item,
                    )
                continue

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
        await _release_queued_pipeline_handle(item)
    _chat_input_queue = ChatInputQueue()


async def _enqueue_chat_request(req: ChatRequest) -> ChatResponse:
    """Enqueue one request and wait for the worker-produced response.

    Args:
        req: Incoming chat request.

    Returns:
        Chat response produced by the worker or drop/collapse policy.
    """

    scope = PipelineScope(
        platform=req.platform,
        platform_channel_id=req.platform_channel_id,
        channel_type=req.channel_type,
    )
    _pipeline_coordinator.request_cancellation(
        scope=scope,
        requested_by="service.chat_queue",
        reason="same_scope_foreground_pending",
    )
    admission = await _pipeline_coordinator.start_run(
        scope=scope,
        owner="service.chat_queue",
        precedence="foreground",
        run_kind="chat",
    )
    handle = admission.handle
    _ensure_chat_input_worker_started()
    item_enqueued = False

    def _mark_item_enqueued() -> None:
        nonlocal item_enqueued
        item_enqueued = True

    try:
        response = await _chat_input_queue.enqueue(
            req,
            pipeline_run_handle=handle,
            on_enqueued=_mark_item_enqueued,
        )
    finally:
        if not item_enqueued and handle is not None:
            await handle.__aexit__(None, None, None)
    return response


async def _run_consolidation_background(state: dict) -> None:
    """Run consolidation after the dialog has already been returned.

    Args:
        state: Persona graph state snapshot needed by the consolidator.
    """

    await brain_post_turn.run_consolidation_background(
        state,
        call_consolidation_subgraph_func=call_consolidation_subgraph,
        update_character_runtime_state_func=(
            _update_runtime_character_state_from_consolidation
        ),
        logger=logger,
    )


async def _run_post_turn_memory_lifecycle_background(state: dict) -> dict:
    """Run post-surface active-commitment lifecycle review."""

    updated_state = await brain_post_turn.run_post_turn_memory_lifecycle_background(
        state,
        active_commitment_reader=query_active_commitment_memory_units_for_user,
        review_func=call_post_surface_memory_lifecycle_review,
        execute_action_specs_func=execute_action_specs_for_trace,
        logger=logger,
        no_remember=False,
        visible_response_sent=True,
        think_only_suppressed=False,
    )
    return updated_state


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


async def _run_internal_monologue_residue_record_background(
    state: dict,
) -> None:
    """Record private internal residue after response completion."""

    await brain_post_turn.run_internal_monologue_residue_record_background(
        state,
        record_completed_episode_residue_func=record_completed_episode_residue,
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
    platform_bot_id: str = "",
    display_name: str = "",
) -> None:
    """Register a cross-process adapter callback for scheduled delivery.

    Args:
        platform: Platform key such as ``qq`` or ``discord``.
        callback_url: Base callback URL exposed by the adapter process.
        shared_secret: Optional bearer token used when the brain calls back.
        timeout_seconds: Timeout for one outbound callback request.
        platform_bot_id: Platform account id for outbound history rows.
        display_name: Adapter-side display name fallback.
    """

    register_runtime_adapter(
        RemoteHttpAdapter(
            platform=platform,
            callback_url=callback_url,
            shared_secret=shared_secret,
            timeout_seconds=timeout_seconds,
            platform_bot_id=platform_bot_id,
            display_name=display_name,
        )
    )


async def _hydrate_rag_initializer_cache() -> int:
    """Hydrate current-version persistent initializer cache rows into memory.

    Returns:
        Number of valid rows loaded into the process-local Cache2 runtime.
    """

    loaded_count = await brain_cache_startup.hydrate_persistent_cache(
        load_entries_func=load_initializer_entries,
        get_rag_cache2_runtime_func=get_rag_cache2_runtime,
        cache_name=INITIALIZER_CACHE_NAME,
        label="RAG initializer",
        max_entries=RAG_CACHE2_MAX_ENTRIES,
        logger=logger,
    )
    return loaded_count


async def _hydrate_media_descriptor_cache() -> int:
    """Hydrate current-version persistent media descriptor cache rows into memory.

    Returns:
        Number of valid rows loaded into the process-local Cache2 runtime.
    """
    loaded_count = await brain_cache_startup.hydrate_persistent_cache(
        load_entries_func=load_media_descriptor_entries,
        get_rag_cache2_runtime_func=get_rag_cache2_runtime,
        cache_name=MEDIA_DESCRIPTOR_CACHE_NAME,
        label="media descriptor",
        max_entries=MEDIA_DESCRIPTOR_CACHE_MAX_HYDRATION_ENTRIES,
        logger=logger,
    )
    return loaded_count


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _static_character_profile, _runtime_character_state
    global _graph, _adapter_registry
    global _calendar_worker_handle
    global _reflection_worker_handle, _self_cognition_worker_handle
    global _background_work_worker_handle

    process_correlation_id = uuid4().hex
    host_label = socket.gethostname()
    try:
        # 1. Database bootstrap
        database_started_at = time.perf_counter()
        await db_bootstrap()
        await event_logging.record_resource_health_event(
            component=SERVICE_COMPONENT,
            resource_name="mongo",
            resource_kind="database",
            availability="available",
            latency_ms=_elapsed_ms(database_started_at),
            status="ok",
        )

        # 2. Hydrate persistent RAG initializer cache into the process-local LRU
        cache_started_at = time.perf_counter()
        await _hydrate_rag_initializer_cache()
        await event_logging.record_resource_health_event(
            component=SERVICE_COMPONENT,
            resource_name="rag_initializer_cache",
            resource_kind="cache",
            availability="available",
            latency_ms=_elapsed_ms(cache_started_at),
            status="ok",
        )

        # 2b. Hydrate persistent media descriptor cache
        media_cache_started_at = time.perf_counter()
        await _hydrate_media_descriptor_cache()
        await event_logging.record_resource_health_event(
            component=SERVICE_COMPONENT,
            resource_name="media_descriptor_cache",
            resource_kind="cache",
            availability="available",
            latency_ms=_elapsed_ms(media_cache_started_at),
            status="ok",
        )

        # 3. Load character profile from database
        character_profile = await get_character_profile()
        if not character_profile.get("name"):
            raise RuntimeError(
                "No character profile found in the database. "
                "Please load one first with:  "
                "python -m scripts.load_character_profile personalities/kazusa.json"
            )
        (
            _static_character_profile,
            _runtime_character_state,
        ) = split_character_profile_runtime_state(character_profile)
        await _refresh_runtime_character_state()

        # 4. Build the LangGraph pipeline
        _graph = _build_graph()

        # 5. Start MCP tool servers
        mcp_started_at = time.perf_counter()
        try:
            await mcp_manager.start()
        except Exception as exc:
            logger.exception(
                f"MCP manager failed to start — tools will be unavailable: {exc}"
            )
            await event_logging.record_resource_health_event(
                component=SERVICE_COMPONENT,
                resource_name="mcp_manager",
                resource_kind="tool_runtime",
                availability="degraded",
                latency_ms=_elapsed_ms(mcp_started_at),
                failure_class=exc.__class__.__name__,
                status="degraded",
                severity="warning",
            )
            (
                error_class,
                error_preview,
                stack_fingerprint,
                top_frame_module,
            ) = _runtime_error_fields(exc)
            await event_logging.record_runtime_error_event(
                component=SERVICE_COMPONENT,
                error_class=error_class,
                error_preview=error_preview,
                stack_fingerprint=stack_fingerprint,
                top_frame_module=top_frame_module,
                recovered=True,
                status="recovered",
                correlation_id=process_correlation_id,
                severity="warning",
            )
        else:
            await event_logging.record_resource_health_event(
                component=SERVICE_COMPONENT,
                resource_name="mcp_manager",
                resource_kind="tool_runtime",
                availability="available",
                latency_ms=_elapsed_ms(mcp_started_at),
                status="ok",
            )

        # 6. Build runtime adapter registry and background workers
        adapter_registry = AdapterRegistry()
        _adapter_registry = adapter_registry

        logger.info(render_llm_route_table())
        _ensure_chat_input_worker_started()
        if CALENDAR_SCHEDULER_ENABLED:
            calendar_handler_registry = CalendarRunHandlerRegistry()
            calendar_handler_registry.register(
                calendar_models.TRIGGER_REFLECTION_PHASE_SLOT,
                _handle_calendar_reflection_phase_run,
            )
            _calendar_worker_handle = start_calendar_scheduler_worker(
                repository=calendar_repository,
                handler_registry=calendar_handler_registry,
                poll_interval_seconds=(
                    CALENDAR_SCHEDULER_POLL_INTERVAL_SECONDS
                ),
                lease_owner=CALENDAR_SCHEDULER_LEASE_OWNER,
                lease_duration_seconds=CALENDAR_SCHEDULER_LEASE_SECONDS,
                claim_limit=CALENDAR_SCHEDULER_CLAIM_LIMIT,
                max_attempts=CALENDAR_SCHEDULER_MAX_ATTEMPTS,
            )
        else:
            logger.info(
                "Calendar scheduler worker disabled via "
                "CALENDAR_SCHEDULER_ENABLED=false"
            )
        if SELF_COGNITION_ENABLED:
            _self_cognition_worker_handle = start_self_cognition_worker(
                is_primary_interaction_busy=_primary_interaction_busy,
                character_profile_provider=_current_character_profile_snapshot,
                adapter_registry_provider=lambda: _adapter_registry,
                latest_cognition_graph_publisher=(
                    _publish_self_cognition_latest_graph
                ),
                should_pause_for_affect_settling=(
                    should_pause_self_cognition_for_affect_settling
                ),
                pipeline_coordinator=_pipeline_coordinator,
            )
        else:
            logger.info(
                "Self-cognition worker disabled via SELF_COGNITION_ENABLED=false"
            )
        if BACKGROUND_WORK_WORKER_ENABLED:
            _background_work_worker_handle = (
                start_background_work_runtime(
                    is_primary_interaction_busy=_primary_interaction_busy,
                    deliver_result_episode_func=(
                        _deliver_background_work_result_episode
                    ),
                )
            )
        else:
            logger.info(
                "Background work worker disabled via "
                "BACKGROUND_WORK_WORKER_ENABLED=false"
            )
        calendar_phase_provider = CalendarReflectionPhaseRunProvider()
        if REFLECTION_CYCLE_ENABLED:
            _reflection_worker_handle = start_reflection_cycle_worker(
                is_primary_interaction_busy=(
                    _reflection_cycle_primary_interaction_busy
                ),
                adapter_registry_provider=lambda: _adapter_registry,
                phase_run_provider=calendar_phase_provider,
                character_state_refresh_callback=(
                    _refresh_runtime_character_state
                ),
            )
        else:
            logger.info(
                "Reflection cycle worker disabled via REFLECTION_CYCLE_ENABLED=false"
            )
        await event_logging.record_process_event(
            event_type="startup",
            phase="lifespan",
            component=SERVICE_COMPONENT,
            status="ready",
            pid=os.getpid(),
            host_label=host_label,
            correlation_id=process_correlation_id,
        )
        logger.info("Kazusa brain service is ready")
    except Exception as exc:
        logger.exception(f"Kazusa brain service startup failed: {exc}")
        (
            error_class,
            error_preview,
            stack_fingerprint,
            top_frame_module,
        ) = _runtime_error_fields(exc)
        await event_logging.record_runtime_error_event(
            component=SERVICE_COMPONENT,
            error_class=error_class,
            error_preview=error_preview,
            stack_fingerprint=stack_fingerprint,
            top_frame_module=top_frame_module,
            recovered=False,
            status="failed",
            correlation_id=process_correlation_id,
            severity="critical",
        )
        await event_logging.record_process_event(
            event_type="lifespan_error",
            phase="startup",
            component=SERVICE_COMPONENT,
            status="failed",
            pid=os.getpid(),
            host_label=host_label,
            severity="critical",
            correlation_id=process_correlation_id,
        )
        raise

    try:
        yield
    finally:
        try:
            # Shutdown
            if _calendar_worker_handle is not None:
                await stop_calendar_scheduler_worker(_calendar_worker_handle)
                _calendar_worker_handle = None
            if _self_cognition_worker_handle is not None:
                await stop_self_cognition_worker(_self_cognition_worker_handle)
                _self_cognition_worker_handle = None
            if _background_work_worker_handle is not None:
                await stop_background_work_runtime(
                    _background_work_worker_handle,
                )
                _background_work_worker_handle = None
            if _reflection_worker_handle is not None:
                await stop_reflection_cycle_worker(_reflection_worker_handle)
                _reflection_worker_handle = None
            await _stop_chat_input_worker()
            await mcp_manager.stop()
            await event_logging.record_process_event(
                event_type="shutdown",
                phase="lifespan",
                component=SERVICE_COMPONENT,
                status="stopped",
                pid=os.getpid(),
                host_label=host_label,
                correlation_id=process_correlation_id,
            )
            await close_db()
            logger.info("Kazusa brain service shut down")
        except Exception as exc:
            logger.exception(f"Kazusa brain service shutdown failed: {exc}")
            (
                error_class,
                error_preview,
                stack_fingerprint,
                top_frame_module,
            ) = _runtime_error_fields(exc)
            await event_logging.record_runtime_error_event(
                component=SERVICE_COMPONENT,
                error_class=error_class,
                error_preview=error_preview,
                stack_fingerprint=stack_fingerprint,
                top_frame_module=top_frame_module,
                recovered=False,
                status="failed",
                correlation_id=process_correlation_id,
                severity="critical",
            )
            await event_logging.record_process_event(
                event_type="lifespan_error",
                phase="shutdown",
                component=SERVICE_COMPONENT,
                status="failed",
                pid=os.getpid(),
                host_label=host_label,
                severity="critical",
                correlation_id=process_correlation_id,
            )
            raise


# ── App ─────────────────────────────────────────────────────────────

app = FastAPI(title="Kazusa Brain Service", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return_value = await brain_health.build_health_response(
        check_database_connection_func=check_database_connection,
        get_rag_cache2_runtime_func=get_rag_cache2_runtime,
        logger=logger,
    )
    return return_value


@app.get("/ops/runtime-status", response_model=OpsRuntimeStatusResponse)
async def ops_runtime_status(
    window_hours: int = 24,
) -> OpsRuntimeStatusResponse:
    """Return bounded runtime status for trusted local operators."""

    base_status = await event_logging.build_runtime_status(
        window_hours=window_hours,
    )
    payload = _ops_runtime_status_payload(base_status)
    response = OpsRuntimeStatusResponse.model_validate(payload)
    return response


@app.get(
    "/ops/latest-cognition-graph",
    response_model=OpsLatestCognitionGraphResponse,
)
async def ops_latest_cognition_graph() -> OpsLatestCognitionGraphResponse:
    """Return the latest bounded cognition graph snapshot for operators."""

    response = _latest_cognition_graph_response()
    return response


@app.get("/ops/reflection/stats", response_model=OpsStatsResponse)
async def ops_reflection_stats(
    window_hours: int = 24,
) -> OpsStatsResponse:
    """Return aggregate reflection telemetry for trusted local operators."""

    stats = await event_logging.build_reflection_stats(
        window_hours=window_hours,
    )
    response = OpsStatsResponse.model_validate(stats)
    return response


@app.get(
    "/ops/self-cognition/stats",
    response_model=OpsSelfCognitionStatsResponse,
)
async def ops_self_cognition_stats(
    window_hours: int = 24,
) -> OpsSelfCognitionStatsResponse:
    """Return aggregate self-cognition telemetry for trusted local operators."""

    stats = await event_logging.build_self_cognition_stats(
        window_hours=window_hours,
    )
    payload = _ops_self_cognition_stats_payload(stats)
    response = OpsSelfCognitionStatsResponse.model_validate(payload)
    return response


@app.post("/runtime/adapters/register", response_model=RuntimeAdapterRegistrationResponse)
async def register_runtime_adapter_endpoint(req: RuntimeAdapterRegistrationRequest):
    """Register one cross-process adapter callback for trusted delivery.

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


@app.post("/delivery_receipt", response_model=DeliveryReceiptResponse)
async def delivery_receipt(
    req: DeliveryReceiptRequest,
) -> DeliveryReceiptResponse:
    """Record the platform message id for a delivered assistant response.

    Args:
        req: Adapter-provided delivery metadata for one chat response.

    Returns:
        Update status for the matching assistant conversation row.
    """

    delivered_at = req.delivered_at or storage_utc_now_iso()
    updated = await apply_assistant_delivery_receipt(
        platform=req.platform,
        platform_channel_id=req.platform_channel_id,
        delivery_tracking_id=req.delivery_tracking_id,
        logical_message_index=req.logical_message_index,
        platform_message_id=req.platform_message_id,
        delivered_at=delivered_at,
        adapter=req.adapter,
    )
    status = "updated" if updated else "not_found"
    response = DeliveryReceiptResponse(status=status, updated=updated)
    return response


@app.post("/event")
async def event(req: EventRequest):
    """Receive a platform event (user joined, topic change, etc.)."""
    logger.info(f'Received event: {req.platform}/{req.event_type}')
    # TODO: dispatch to event handlers
    return_value = {"status": "accepted"}
    return return_value
