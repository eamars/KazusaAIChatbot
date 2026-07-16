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
from kazusa_ai_chatbot.media_inspection.session_cache import (
    begin_session_turn,
    put_session_media,
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
from kazusa_ai_chatbot.brain_service.turn_settlement import (
    AssessmentLease,
    PersistedChatFragment,
    TurnSettlementCoordinator,
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
from kazusa_ai_chatbot.relevance import (
    build_group_attention_context,
    frontline_relevance_agent,
    relevance_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer import (
    multimedia_descriptor_agent,
    select_media_for_turn,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2 import persona_supervisor2
from kazusa_ai_chatbot.nodes.persona_supervisor2_memory_lifecycle import (
    call_post_surface_memory_lifecycle_review,
)
from kazusa_ai_chatbot.consolidation.core import call_consolidation_subgraph
from kazusa_ai_chatbot.rag.cache2_policy import (
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
        relevance_agent_node=_graph_relevance_node,
        multimedia_descriptor_agent_node=multimedia_descriptor_agent,
        load_conversation_episode_state_node=load_conversation_episode_state,
        persona_supervisor_node=persona_supervisor2,
        claim_for_cognition_node=_claim_for_cognition_node,
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


async def _evaluate_frontline_for_service(state: Mapping[str, Any]):
    """Run the service-owned frontline semantic stage."""

    return await frontline_relevance_agent(state)


async def _evaluate_settled_for_service(
    lease: AssessmentLease,
    state: Mapping[str, Any],
):
    """Run the service-owned settled relevance semantic stage."""

    del lease
    return await relevance_agent(state)




def _new_turn_settlement_coordinator() -> TurnSettlementCoordinator:
    """Create the coordinator bound to the service's active event loop."""

    return_value = TurnSettlementCoordinator(
        frontline_evaluator=_evaluate_frontline_for_service,
        settled_evaluator=_evaluate_settled_for_service,
    )
    return return_value


_turn_settlement_coordinator = _new_turn_settlement_coordinator()
_turn_settlement_worker_task: asyncio.Task[Any] | None = None


async def _claim_for_cognition_node(state: IMProcessState) -> dict[str, Any]:
    """Map the deterministic versioned claim into graph state."""

    if state.get("cognition_claimed") is True:
        return_value = {
            "cognition_claimed": True,
            "should_respond": True,
        }
        return return_value

    turn_id = state.get("turn_id", "")
    turn_version = state.get("turn_version")
    claimed = False
    if isinstance(turn_version, int) and turn_id:
        claimed = await _turn_settlement_coordinator.claim_for_cognition(
            turn_id,
            turn_version,
        )
    return_value = {
        "cognition_claimed": claimed,
        "should_respond": claimed,
    }
    return return_value


async def _graph_relevance_node(state: IMProcessState) -> dict[str, Any]:
    """Use the settled decision already made before graph entry."""

    if state.get("response_action") in {"ignore", "proceed", "wait"}:
        return_value = {
            "response_action": state["response_action"],
            "should_respond": state["response_action"] == "proceed",
            "reason_to_respond": state.get("reason_to_respond", ""),
            "use_reply_feature": state.get("use_reply_feature", False),
            "channel_topic": state.get("channel_topic", ""),
            "indirect_speech_context": state.get(
                "indirect_speech_context",
                "",
            ),
        }
        return return_value
    return_value = {
        "response_action": "ignore",
        "should_respond": False,
        "reason_to_respond": "settled relevance decision missing",
        "use_reply_feature": False,
        "channel_topic": "",
        "indirect_speech_context": "",
    }
    return return_value


def _frontline_target_labels(
    envelope: Mapping[str, Any],
    character_global_user_id: str,
) -> list[str]:
    """Project typed addressee evidence into semantic target labels."""

    addressed_to = envelope.get("addressed_to_global_user_ids") or []
    if not isinstance(addressed_to, list):
        addressed_to = list(addressed_to) if isinstance(addressed_to, tuple) else []
    has_character = character_global_user_id in addressed_to
    has_other = any(
        isinstance(value, str) and value != character_global_user_id
        for value in addressed_to
    )
    labels: list[str] = []
    if has_character:
        labels.append("character")
    if has_other:
        labels.append("other_participant")
    if not labels and envelope.get("broadcast"):
        labels.append("broadcast")
    if not labels:
        labels.append("none")
    return_value = labels
    return return_value


def _frontline_reply_label(
    envelope: Mapping[str, Any],
    character_global_user_id: str,
) -> str:
    """Project typed reply evidence without exposing reply identifiers."""

    reply = envelope.get("reply")
    if not isinstance(reply, Mapping):
        return_value = "none"
        return return_value
    if reply.get("global_user_id") == character_global_user_id:
        return_value = "character"
        return return_value
    if reply.get("global_user_id") or reply.get("platform_user_id"):
        return_value = "other_participant"
        return return_value
    return_value = "unknown_participant"
    return return_value


def _build_frontline_fragment(
    item: QueuedChatItem,
    *,
    message_envelope: Mapping[str, Any],
    global_user_id: str,
    character_global_user_id: str,
) -> PersistedChatFragment:
    """Build the persisted fragment and semantic labels for frontline work."""

    attachments = message_envelope.get("attachments") or []
    media_labels = tuple(
        str(attachment.get("media_type"))
        for attachment in attachments
        if isinstance(attachment, Mapping) and attachment.get("media_type")
    )
    semantic_targets = tuple(
        _frontline_target_labels(message_envelope, character_global_user_id)
    )
    reply_target_label = _frontline_reply_label(
        message_envelope,
        character_global_user_id,
    )
    media_descriptions = tuple(
        {
            "media_kind": str(attachment.get("media_type", "")),
            "description": str(attachment.get("description", "")),
        }
        for attachment in attachments
        if isinstance(attachment, Mapping)
        and attachment.get("description")
    )
    return_value = PersistedChatFragment(
        arrival_sequence=item.sequence,
        scope=(
            item.request.platform,
            item.request.platform_channel_id,
            item.request.channel_type,
        ),
        author_platform_user_id=item.request.platform_user_id,
        author_global_user_id=global_user_id,
        platform_message_id=item.request.platform_message_id,
        conversation_row_id=item.conversation_row_id,
        storage_timestamp_utc=item.storage_timestamp_utc,
        enqueue_monotonic=item.enqueue_monotonic,
        body_text=message_envelope.get("body_text", ""),
        semantic_target_labels=semantic_targets,
        reply_target_label=reply_target_label,
        media_labels=media_labels,
        media_descriptions=media_descriptions,
        attachments=tuple(
            dict(attachment)
            for attachment in attachments
            if isinstance(attachment, Mapping)
        ),
        request=item.request,
        future=item.future,
        pipeline_run_handle=item.pipeline_run_handle,
        queue_item=item,
    )
    return return_value


async def _build_frontline_state(
    fragment: PersistedChatFragment,
) -> dict[str, Any]:
    """Build the bounded frontline semantic projection from service state."""

    return_value = await _turn_settlement_coordinator.build_frontline_state(
        fragment,
    )
    return_value["active_character_name"] = _static_character_profile.get(
        "name",
        "Character",
    )
    queue_item = fragment.queue_item
    if isinstance(queue_item, QueuedChatItem):
        return_value["llm_trace_id"] = queue_item.llm_trace_id
    return return_value


def _collapsed_private_frontline_message(
    fragments: tuple[PersistedChatFragment, ...],
) -> dict[str, Any]:
    """Project one coalesced private logical input for frontline judgment."""

    body_text = "\n".join(
        fragment.body_text for fragment in fragments if fragment.body_text
    )
    semantic_target_labels = list(dict.fromkeys(
        label
        for fragment in fragments
        for label in fragment.semantic_target_labels
    ))
    media_labels = list(dict.fromkeys(
        label
        for fragment in fragments
        for label in fragment.media_labels
    ))
    reply_labels = {
        fragment.reply_target_label
        for fragment in fragments
        if fragment.reply_target_label != "none"
    }
    if "character" in reply_labels:
        reply_target_label = "character"
    elif "other_participant" in reply_labels:
        reply_target_label = "other_participant"
    else:
        reply_target_label = "none"
    return_value = {
        "body_text": body_text,
        "semantic_target_labels": semantic_target_labels,
        "reply_target_label": reply_target_label,
        "media_labels": media_labels,
    }
    return return_value


async def _prepare_frontline_fragment(
    item: QueuedChatItem,
) -> PersistedChatFragment:
    """Resolve identities and persist one message before frontline judgment."""

    req = item.request
    character_name = _static_character_profile.get("name", "Character")
    character_global_user_id = await _ensure_character_global_identity(
        platform=req.platform,
        platform_bot_id=req.platform_bot_id,
        character_name=character_name,
    )
    global_user_id, user_profile = await _resolve_queued_user(item)
    message_envelope = await _resolve_message_envelope_identities(req)
    item.global_user_id = global_user_id
    item.user_profile = dict(user_profile)
    item.resolved_message_envelope = message_envelope
    if not item.llm_trace_id:
        item.llm_trace_id = llm_tracing.build_trace_id()
    await llm_tracing.ensure_llm_trace_run(
        trace_id=item.llm_trace_id,
        platform=req.platform,
        platform_channel_id=req.platform_channel_id,
        channel_type=req.channel_type,
        platform_message_id=req.platform_message_id,
        global_user_id=global_user_id,
        started_at=item.storage_timestamp_utc,
    )
    if not item.conversation_row_id:
        conversation_row_id = await _save_user_message_from_item(
            item,
            global_user_id=global_user_id,
            reply_context=await _hydrate_reply_context(req),
            message_envelope=message_envelope,
        )
        if not conversation_row_id:
            raise RuntimeError(
                "frontline message persistence did not commit a row"
            )
        item.conversation_row_id = conversation_row_id
    fragment = _build_frontline_fragment(
        item,
        message_envelope=message_envelope,
        global_user_id=global_user_id,
        character_global_user_id=character_global_user_id,
    )
    return_value = fragment
    return return_value


async def _prepare_frontline_fragments(
    item: QueuedChatItem,
) -> tuple[PersistedChatFragment, ...]:
    """Prepare the leader and any private fragments as one intake unit."""

    leader = await _prepare_frontline_fragment(item)
    fragments = [leader]
    character_global_user_id = await _ensure_character_global_identity(
        platform=item.request.platform,
        platform_bot_id=item.request.platform_bot_id,
        character_name=_static_character_profile.get("name", "Character"),
    )
    for collapsed_item in sorted(
        item.collapsed_items,
        key=lambda queued_item: queued_item.sequence,
    ):
        message_envelope = collapsed_item.resolved_message_envelope
        if not isinstance(message_envelope, Mapping):
            message_envelope = await _resolve_message_envelope_identities(
                collapsed_item.request,
            )
            collapsed_item.resolved_message_envelope = message_envelope
        if not collapsed_item.global_user_id:
            global_user_id, user_profile = await _resolve_queued_user(
                collapsed_item,
            )
            collapsed_item.global_user_id = global_user_id
            collapsed_item.user_profile = dict(user_profile)
        if not collapsed_item.conversation_row_id:
            conversation_row_id = await _save_user_message_from_item(
                collapsed_item,
                global_user_id=collapsed_item.global_user_id,
                reply_context=await _hydrate_reply_context(
                    collapsed_item.request,
                ),
                message_envelope=message_envelope,
            )
            if not conversation_row_id:
                raise RuntimeError(
                    "private fragment persistence did not commit a row"
                )
            collapsed_item.conversation_row_id = conversation_row_id
        fragments.append(
            _build_frontline_fragment(
                collapsed_item,
                message_envelope=message_envelope,
                global_user_id=collapsed_item.global_user_id,
                character_global_user_id=character_global_user_id,
            )
        )
    return_value = tuple(fragments)
    return return_value


def _fragment_media_rows(
    fragment: PersistedChatFragment,
    *,
    fragment_index: int,
) -> list[dict[str, Any]]:
    """Project prompt-usable media rows with their owning fragment index."""

    rows: list[dict[str, Any]] = []
    for attachment in fragment.attachments:
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
        source_key = base64_data or description
        media_key = hashlib.sha256(source_key.encode("utf-8")).hexdigest()
        rows.append({
            "content_type": media_type,
            "base64_data": base64_data,
            "description": description,
            "_fragment_index": fragment_index,
            "_media_key": media_key,
        })
    return_value = rows
    return return_value


def _bounded_media_selection(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    preserve_opening: bool,
) -> list[dict[str, Any]]:
    """Select the opening and newest image rows within a remaining budget."""

    if limit <= 0 or not rows:
        return []
    selected, _overflow = select_media_for_turn(rows)
    if len(selected) <= limit:
        return selected
    if not preserve_opening:
        return selected[-limit:]
    if limit == 1:
        return [selected[0]]
    return_value = [selected[0], *selected[-(limit - 1):]]
    return return_value


def _cached_media_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Strip settlement bookkeeping from one reusable descriptor row."""

    return_value = {
        "content_type": str(row.get("content_type", "")),
        "base64_data": str(row.get("base64_data", "")),
        "description": str(row.get("description", "")),
    }
    if isinstance(row.get("image_observation"), Mapping):
        return_value["image_observation"] = dict(row["image_observation"])
    return return_value


async def _prepare_settled_media(
    lease: AssessmentLease,
) -> tuple[list[MultiMediaDoc], bool]:
    """Describe admitted media before settled relevance and bound image work."""

    all_rows = [
        row
        for index, fragment in enumerate(lease.fragments)
        for row in _fragment_media_rows(fragment, fragment_index=index)
    ]
    for row in all_rows:
        if not row.get("description"):
            continue
        fragment = lease.fragments[row["_fragment_index"]]
        media_key = row["_media_key"]
        fragment.media_description_cache.setdefault(
            media_key,
            _cached_media_row(row),
        )
        if str(row.get("content_type", "")).startswith("image/"):
            fragment.media_description_attempted_keys.add(media_key)

    attempted_image_keys = {
        media_key
        for fragment in lease.fragments
        for media_key in fragment.media_description_attempted_keys
    }
    image_rows = [
        row
        for row in all_rows
        if str(row.get("content_type", "")).startswith("image/")
    ]
    candidate_rows = [
        row
        for row in image_rows
        if row["_media_key"] not in attempted_image_keys
    ]
    remaining_budget = max(0, 4 - len(attempted_image_keys))
    selected_images = _bounded_media_selection(
        candidate_rows,
        limit=remaining_budget,
        preserve_opening=(
            bool(image_rows)
            and image_rows[0]["_media_key"] not in attempted_image_keys
        ),
    )
    selected_image_ids = {id(row) for row in selected_images}
    for row in selected_images:
        fragment = lease.fragments[row["_fragment_index"]]
        fragment.media_description_attempted_keys.add(row["_media_key"])

    all_image_keys = {row["_media_key"] for row in image_rows}
    attempted_image_keys.update(
        row["_media_key"] for row in selected_images
    )
    additional_media_present = bool(
        all_image_keys - attempted_image_keys
    )
    rows_by_fragment: dict[int, list[dict[str, Any]]] = {}
    for row in all_rows:
        is_image = str(row.get("content_type", "")).startswith("image/")
        if is_image and id(row) not in selected_image_ids:
            continue
        if not is_image:
            fragment = lease.fragments[row["_fragment_index"]]
            fragment.media_description_cache.setdefault(
                row["_media_key"],
                _cached_media_row(row),
            )
            continue
        fragment_index = row["_fragment_index"]
        rows_by_fragment.setdefault(fragment_index, []).append(row)

    for fragment_index, fragment in enumerate(lease.fragments):
        fragment_rows = rows_by_fragment.get(fragment_index, [])
        if not fragment_rows:
            continue
        queue_item = fragment.queue_item
        if not isinstance(queue_item, QueuedChatItem):
            continue
        message_envelope = queue_item.resolved_message_envelope
        if not isinstance(message_envelope, Mapping):
            message_envelope = await _resolve_message_envelope_identities(
                queue_item.request,
            )
        episode = build_text_chat_cognitive_episode(
            episode_id=(
                f"settlement_media:{fragment.platform_message_id}"
            ),
            percept_id=(
                f"settlement_media:{fragment.platform_message_id}:text"
            ),
            storage_timestamp_utc=fragment.storage_timestamp_utc,
            local_time_context=queue_item.local_time_context,
            user_input=fragment.body_text,
            platform=fragment.scope[0],
            platform_channel_id=fragment.scope[1],
            channel_type=fragment.scope[2],
            platform_message_id=fragment.platform_message_id,
            platform_user_id=fragment.author_platform_user_id,
            global_user_id=fragment.author_global_user_id,
            user_name=queue_item.request.display_name,
            target_addressed_user_ids=(
                [CHARACTER_GLOBAL_USER_ID]
                if "character" in fragment.semantic_target_labels
                else []
            ),
            target_broadcast="broadcast" in fragment.semantic_target_labels,
        )
        media_state = {
            "user_name": queue_item.request.display_name,
            "platform_user_id": fragment.author_platform_user_id,
            "platform": fragment.scope[0],
            "platform_channel_id": fragment.scope[1],
            "platform_message_id": fragment.platform_message_id,
            "user_multimedia_input": fragment_rows,
            "message_envelope": message_envelope,
            "reply_context": await _hydrate_reply_context(
                queue_item.request,
            ),
            "cognitive_episode": episode,
        }
        try:
            described_state = await multimedia_descriptor_agent(media_state)
        except Exception as exc:
            logger.warning(
                f"Settled media description failed; retaining supplied "
                f"descriptions: {exc}"
            )
            described_state = {
                "user_multimedia_input": fragment_rows,
                "additional_media_present": False,
            }
        described_rows = described_state.get("user_multimedia_input")
        if not isinstance(described_rows, list):
            described_rows = fragment_rows
        for index, source_row in enumerate(fragment_rows):
            described_row = (
                described_rows[index]
                if index < len(described_rows)
                and isinstance(described_rows[index], Mapping)
                else source_row
            )
            fragment.media_description_cache[source_row["_media_key"]] = (
                _cached_media_row(described_row)
            )

    prepared_rows: list[MultiMediaDoc] = []
    seen_cache_keys: set[str] = set()
    for fragment in lease.fragments:
        fragment.media_descriptions = tuple(
            {
                "media_kind": str(row.get("content_type", "")),
                "description": str(row.get("description", "")),
            }
            for row in fragment.media_description_cache.values()
            if row.get("description")
        )
        for media_key, cached_row in fragment.media_description_cache.items():
            if media_key in seen_cache_keys:
                continue
            seen_cache_keys.add(media_key)
            prepared_rows.append(_cached_media_row(cached_row))

    if lease.fragments and additional_media_present:
        lease.fragments[0].additional_media_present = True
    return_value = (prepared_rows, additional_media_present)
    return return_value


def _settled_state_from_lease(
    lease: AssessmentLease,
    *,
    history: list[dict],
) -> dict[str, Any]:
    """Project a leased turn into settled semantic input."""

    fragment_rows = [
        {
            "body_text": fragment.body_text,
            "semantic_target_labels": list(fragment.semantic_target_labels),
            "reply_target_label": fragment.reply_target_label,
            "media_labels": list(fragment.media_labels),
        }
        for fragment in lease.fragments
    ]
    media_descriptions = [
        dict(description)
        for fragment in lease.fragments
        for description in fragment.media_descriptions
    ]
    additional_media_present = any(
        fragment.additional_media_present for fragment in lease.fragments
    )
    active_platform_message_ids = {
        fragment.platform_message_id
        for fragment in lease.fragments
        if fragment.platform_message_id
    }
    active_conversation_row_ids = {
        fragment.conversation_row_id
        for fragment in lease.fragments
        if fragment.conversation_row_id
    }
    earliest_active_timestamp = (
        lease.fragments[0].storage_timestamp_utc
        if lease.fragments
        else ""
    )
    latest_active_timestamp = (
        lease.fragments[-1].storage_timestamp_utc
        if lease.fragments
        else ""
    )
    earliest_active_index = -1
    latest_active_index = -1
    external_history: list[tuple[int, dict]] = []
    for history_index, row in enumerate(history):
        is_active_row = (
            row.get("platform_message_id") in active_platform_message_ids
            or str(row.get("_id", "")) in active_conversation_row_ids
        )
        if is_active_row:
            if earliest_active_index < 0:
                earliest_active_index = history_index
            latest_active_index = history_index
            continue
        external_history.append((history_index, row))
    recent_external_history = external_history[-10:]
    fresh_history = trim_history_dict([
        row for _history_index, row in recent_external_history
    ])
    for projected_row, (history_index, _source_row) in zip(
        fresh_history,
        recent_external_history,
        strict=True,
    ):
        if latest_active_index < 0:
            temporal_relation = _history_relation_from_timestamp(
                row_timestamp=projected_row.get("timestamp"),
                earliest_active_timestamp=earliest_active_timestamp,
                latest_active_timestamp=latest_active_timestamp,
            )
        elif history_index > latest_active_index:
            temporal_relation = "after_active_turn"
        elif history_index > earliest_active_index:
            temporal_relation = "during_active_turn"
        else:
            temporal_relation = "before_active_turn"
        projected_row["turn_temporal_relation"] = temporal_relation
    leader = lease.fragments[0] if lease.fragments else None
    response_owner = next(
        (
            fragment
            for fragment in lease.fragments
            if fragment.arrival_sequence == lease.response_owner_sequence
        ),
        leader,
    )
    response_owner_item = (
        response_owner.queue_item
        if response_owner is not None
        and isinstance(response_owner.queue_item, QueuedChatItem)
        else None
    )
    request = (
        response_owner_item.request
        if response_owner_item is not None
        else None
    )
    user_profile = (
        response_owner_item.user_profile
        if response_owner_item is not None
        else {}
    )
    direct_participant = any(
        "character" in fragment.semantic_target_labels
        for fragment in lease.fragments
    )
    relationship_context = str(
        user_profile.get("last_relationship_insight", "")
    ).strip()
    if not relationship_context:
        relationship_context = (
            "direct participant" if direct_participant
            else "group participant"
        )
    group_attention = ""
    if request is not None and request.channel_type == "group":
        attention_context = build_group_attention_context(
            chat_history_wide=fresh_history,
            platform_bot_id=request.platform_bot_id,
        )
        group_attention = attention_context["group_attention"]
    return_value = {
        "conversation_scope": (
            response_owner.scope[2]
            if response_owner is not None
            else ""
        ),
        "active_character_name": _static_character_profile.get(
            "name",
            "Character",
        ),
        "assembled_fragments": fragment_rows,
        "media_descriptions": media_descriptions,
        "additional_media_present": additional_media_present,
        "fresh_history": fresh_history,
        "scene_context": (
            request.channel_name
            if request is not None and request.channel_name
            else ""
        ),
        "relationship_context": relationship_context,
        "character_mood": str(_runtime_character_state.get("mood", "")),
        "group_attention": group_attention,
        "bot_continuity": lease.latest_bot_continuity,
        "user_profile": user_profile,
        "current_author_global_user_id": (
            response_owner.author_global_user_id
            if response_owner is not None
            else ""
        ),
        "current_author_platform_user_id": (
            response_owner.author_platform_user_id
            if response_owner is not None
            else ""
        ),
        "character_global_user_id": CHARACTER_GLOBAL_USER_ID,
        "platform_bot_id": (
            request.platform_bot_id
            if request is not None
            else ""
        ),
        "llm_trace_id": (
            response_owner_item.llm_trace_id
            if response_owner_item is not None
            else ""
        ),
    }
    return return_value


def _history_relation_from_timestamp(
    *,
    row_timestamp: object,
    earliest_active_timestamp: str,
    latest_active_timestamp: str,
) -> str:
    """Relate a history row when the bounded window omits the active row."""

    if (
        not isinstance(row_timestamp, str)
        or not earliest_active_timestamp
        or not latest_active_timestamp
    ):
        return_value = "unknown"
        return return_value

    try:
        row_time = parse_storage_utc_datetime(row_timestamp)
        earliest_active_time = parse_storage_utc_datetime(
            earliest_active_timestamp,
        )
        active_time = parse_storage_utc_datetime(latest_active_timestamp)
    except ValueError:
        return_value = "unknown"
        return return_value

    if row_time > active_time:
        return_value = "after_active_turn"
        return return_value
    if row_time < earliest_active_time:
        return_value = "before_active_turn"
        return return_value
    if earliest_active_time < row_time < active_time:
        return_value = "during_active_turn"
        return return_value
    return_value = "unknown"
    return return_value


async def _complete_settled_fragments(
    lease: AssessmentLease,
    response: ChatResponse,
    *,
    release_response_owner: bool = True,
) -> None:
    """Deliver one response to its owner and silence every other fragment."""

    for fragment in lease.fragments:
        queue_item = fragment.queue_item
        if not isinstance(queue_item, QueuedChatItem):
            continue
        is_response_owner = (
            fragment.arrival_sequence == lease.response_owner_sequence
        )
        fragment_response = response if is_response_owner else ChatResponse()
        _chat_input_queue.complete(queue_item, fragment_response)
        if release_response_owner or not is_response_owner:
            await _release_queued_pipeline_handle(queue_item)


def _response_owner_item(lease: AssessmentLease) -> QueuedChatItem | None:
    """Return the queue item that owns the assembled turn response."""

    for fragment in lease.fragments:
        if fragment.arrival_sequence != lease.response_owner_sequence:
            continue
        if isinstance(fragment.queue_item, QueuedChatItem):
            return_value = fragment.queue_item
            return return_value
    return None


async def _frontline_intake_item(item: QueuedChatItem) -> None:
    """Run persistence and frontline admission for one queued item."""

    if item.request.debug_modes.listen_only:
        await _process_queued_chat_item(item)
        return

    try:
        fragments = await _prepare_frontline_fragments(item)
        leader_fragment = fragments[0]
        frontline_state = await _build_frontline_state(leader_fragment)
        if len(fragments) > 1:
            frontline_state["current_message"] = (
                _collapsed_private_frontline_message(fragments)
            )
        decision = await _turn_settlement_coordinator.evaluate_frontline(
            frontline_state,
        )
        outcome = await _turn_settlement_coordinator.apply_frontline_decision(
            leader_fragment,
            decision,
        )
        if outcome.action == "discard":
            discard_decision = {
                "intake_action": "discard",
                "append_target": "none",
                "prelude_targets": [],
                "reason": "collapsed private input follows discarded leader",
            }
            for fragment in fragments[1:]:
                await _turn_settlement_coordinator.apply_frontline_decision(
                    fragment,
                    discard_decision,
                )
            for fragment in fragments:
                queue_item = fragment.queue_item
                if isinstance(queue_item, QueuedChatItem):
                    _chat_input_queue.complete(queue_item, ChatResponse())
                    await _release_queued_pipeline_handle(queue_item)
                    if queue_item.llm_trace_id:
                        await llm_tracing.finalize_llm_trace_run(
                            trace_id=queue_item.llm_trace_id,
                            status="succeeded",
                            final_dialog_count=0,
                            delivery_tracking_id="",
                        )
            return

        if outcome.action == "append":
            _chat_input_queue.complete(item, ChatResponse())
            await _release_queued_pipeline_handle(item)
            await llm_tracing.finalize_llm_trace_run(
                trace_id=item.llm_trace_id,
                status="succeeded",
                final_dialog_count=0,
                delivery_tracking_id="",
            )
        for fragment in fragments[1:]:
            await _turn_settlement_coordinator.append_collapsed_private_fragment(
                fragment,
                turn_id=outcome.turn_id,
            )
            queue_item = fragment.queue_item
            if isinstance(queue_item, QueuedChatItem):
                _chat_input_queue.complete(queue_item, ChatResponse())
                await _release_queued_pipeline_handle(queue_item)
    except Exception as exc:
        logger.exception(f"Frontline intake failed: {exc}")
        for queued_item in [item, *item.collapsed_items]:
            await _turn_settlement_coordinator.complete_ingress(
                queued_item.sequence,
            )
            _chat_input_queue.fail(queued_item, exc)
            await _release_queued_pipeline_handle(queued_item)
            if queued_item.llm_trace_id:
                await llm_tracing.finalize_llm_trace_run(
                    trace_id=queued_item.llm_trace_id,
                    status="failed",
                    final_dialog_count=0,
                    delivery_tracking_id="",
                )


async def _process_settlement_lease(
    lease: AssessmentLease,
    response_owner: QueuedChatItem,
) -> None:
    """Assess and finish one ready turn while the worker owns its activity."""

    try:
        prepared_media, additional_media_present = await _prepare_settled_media(
            lease,
        )
        history = await get_conversation_history(
            platform=response_owner.request.platform,
            platform_channel_id=response_owner.request.platform_channel_id,
            limit=CONVERSATION_HISTORY_LIMIT,
        )
        settled_state = _settled_state_from_lease(
            lease,
            history=history,
        )
        decision = await _turn_settlement_coordinator.evaluate_settled(
            lease,
            settled_state,
        )
        outcome = await _turn_settlement_coordinator.apply_settled_decision(
            lease,
            decision,
        )
    except Exception as exc:
        logger.exception(f"Settled relevance failed: {exc}")
        failure_decision = {
            "response_action": "ignore",
            "reason_to_respond": "settled relevance failed closed",
            "use_reply_feature": False,
            "channel_topic": "",
            "indirect_speech_context": "",
        }
        failure_outcome = (
            await _turn_settlement_coordinator.apply_settled_decision(
                lease,
                failure_decision,
            )
        )
        if failure_outcome.stale:
            return
        await _complete_settled_fragments(lease, ChatResponse())
        if response_owner.llm_trace_id:
            await llm_tracing.finalize_llm_trace_run(
                trace_id=response_owner.llm_trace_id,
                status="failed",
                final_dialog_count=0,
                delivery_tracking_id="",
            )
        return

    if outcome.stale or outcome.response_action == "wait":
        return
    if outcome.response_action == "ignore":
        await _complete_settled_fragments(lease, ChatResponse())
        if response_owner.llm_trace_id:
            await llm_tracing.finalize_llm_trace_run(
                trace_id=response_owner.llm_trace_id,
                status="succeeded",
                final_dialog_count=0,
                delivery_tracking_id="",
            )
        return

    claimed = await _turn_settlement_coordinator.claim_for_cognition(
        lease.turn_id,
        lease.version,
    )
    if not claimed:
        return

    try:
        await _process_queued_chat_item(
            response_owner,
            settlement_fragments=list(lease.fragments),
            settled_decision=decision,
            skip_user_persist=True,
            settlement_turn_id=lease.turn_id,
            settlement_version=lease.version,
            settlement_claimed=True,
            prepared_media=prepared_media,
            media_prepared=True,
            additional_media_present=additional_media_present,
        )
        await _complete_settled_fragments(
            lease,
            ChatResponse(),
            release_response_owner=False,
        )
    finally:
        await _turn_settlement_coordinator.complete_cognition(
            lease.turn_id,
            lease.version,
        )


async def _turn_settlement_worker() -> None:
    """Assess ready turns globally and hand only claimed proceeds to cognition."""

    global _primary_interaction_active_count

    while True:
        lease = await _turn_settlement_coordinator.wait_for_assessment_ready()
        if not lease.fragments:
            continue
        response_owner = _response_owner_item(lease)
        if response_owner is None:
            continue
        _primary_interaction_active_count += 1
        try:
            await _process_settlement_lease(lease, response_owner)
        finally:
            _primary_interaction_active_count -= 1


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
    """Keep V2 cognition state as the sole runtime character authority."""
    global _runtime_character_state
    del consolidation_result


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


def _accepted_task_result_text(episode: CognitiveEpisode) -> str:
    """Build a compact source summary for accepted-task result cognition."""

    percepts = episode.get("percepts", [])
    result_percept = {}
    for percept in percepts:
        if percept.get("input_source") != "accepted_task_result":
            continue
        result_percept = percept
        break

    metadata = result_percept.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    accepted_task_summary = str(
        metadata.get("accepted_task_summary", "")
    ).strip()
    failure_summary = str(metadata.get("failure_summary", "")).strip()
    status = "failed" if failure_summary else "completed"
    summary_parts = [
        f"Accepted task result is {status}.",
    ]
    if accepted_task_summary:
        summary_parts.append(f"Task: {accepted_task_summary}.")
    result_text = " ".join(summary_parts)
    return result_text


def _accepted_task_result_metadata(
    episode: CognitiveEpisode,
) -> Mapping[str, object]:
    """Return metadata from the accepted-task result percept."""

    for percept in episode["percepts"]:
        if percept.get("input_source") != "accepted_task_result":
            continue
        metadata = percept.get("metadata", {})
        if isinstance(metadata, Mapping):
            return metadata
        break

    return_value: Mapping[str, object] = {}
    return return_value


def _accepted_task_prompt_message_context(
    episode: CognitiveEpisode,
) -> dict[str, object]:
    """Build prompt-safe message context for an accepted-task result."""

    target_scope = episode["target_scope"]
    raw_addressed_ids = target_scope.get("target_addressed_user_ids", [])
    addressed_ids = [
        value
        for value in raw_addressed_ids
        if isinstance(value, str) and value.strip()
    ]
    context = {
        "body_text": _accepted_task_result_text(episode),
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


def _accepted_task_delivery_mentions(
    *,
    result: Mapping[str, object],
    episode: CognitiveEpisode,
) -> list[dict[str, str]]:
    """Build inline mention candidates for accepted-task result delivery."""

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


async def _run_accepted_task_result_post_turn(
    consolidation_state: dict,
    *,
    visible_response_sent: bool,
) -> None:
    """Run non-consolidation post-turn consumers after accepted-task delivery."""

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
            await _run_internal_monologue_residue_record_background(
                consolidation_state,
            )
    except Exception as exc:
        logger.exception(
            f"Accepted task post-turn handling failed after delivery: {exc}"
        )


async def _deliver_accepted_task_result_episode(
    episode: CognitiveEpisode,
) -> dict[str, Any]:
    """Run accepted-task result cognition and deliver selected dispatcher text."""

    adapter_registry = _adapter_registry
    if adapter_registry is None:
        return {
            "status": "failed",
            "reason": "adapter registry is unavailable",
        }
    if episode["trigger_source"] != "accepted_task_result_ready":
        return {
            "status": "failed",
            "reason": "trigger_source must be accepted_task_result_ready",
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
        result_metadata = _accepted_task_result_metadata(episode)
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
                f"work result: {exc}"
            )
            promoted_reflection_context = {}

        trigger_source = str(episode["trigger_source"])

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
            "user_input": _accepted_task_result_text(episode),
            "prompt_message_context": (
                _accepted_task_prompt_message_context(episode)
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
                "delivery_mentions": _accepted_task_delivery_mentions(
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
                bot_permission_role="accepted_task_result",
                now=storage_utc_now(),
                source_channel_type=channel_type,
                source_platform_bot_id=source_platform_bot_id,
                source_character_name=character_name,
            ),
            adapter_registry,
        )
    except Exception as exc:
        logger.exception(
            f"Accepted task result delivery failed: {exc}"
        )
        return {
            "status": "failed",
            "reason": str(exc),
        }

    consolidation_state = result.get("consolidation_state")
    if isinstance(consolidation_state, dict):
        await _run_accepted_task_result_post_turn(
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
        global_user_id, user_profile = await _resolve_queued_user(item)
        message_envelope = await _resolve_message_envelope_identities(
            item.request,
        )
        item.global_user_id = global_user_id
        item.user_profile = dict(user_profile)
        item.resolved_message_envelope = message_envelope
        reply_context = await _hydrate_reply_context(item.request)
        save_started_at = time.perf_counter()
        conversation_row_id = await _save_user_message_from_item(
            item,
            global_user_id=global_user_id,
            reply_context=reply_context,
            message_envelope=message_envelope,
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
    """Persist one queued item retained in a surviving logical turn.

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
        global_user_id, user_profile = await _resolve_queued_user(item)
        message_envelope = await _resolve_message_envelope_identities(
            item.request,
        )
        item.global_user_id = global_user_id
        item.user_profile = dict(user_profile)
        item.resolved_message_envelope = message_envelope
        reply_context = await _hydrate_reply_context(item.request)
        save_started_at = time.perf_counter()
        conversation_row_id = await _save_user_message_from_item(
            item,
            global_user_id=global_user_id,
            reply_context=reply_context,
            message_envelope=message_envelope,
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


async def _process_queued_chat_item(
    item: QueuedChatItem,
    *,
    settlement_fragments: list[PersistedChatFragment] | None = None,
    settled_decision: Mapping[str, Any] | None = None,
    skip_user_persist: bool = False,
    settlement_turn_id: str = "",
    settlement_version: int = 0,
    settlement_claimed: bool = False,
    prepared_media: list[MultiMediaDoc] | None = None,
    media_prepared: bool = False,
    additional_media_present: bool = False,
) -> None:
    """Run one queued item through the existing chat graph and post-writes.

    Args:
        item: Oldest surviving queued item selected by the worker.

    Returns:
        None.
    """

    req = item.request
    character_name = _static_character_profile.get("name", "Character")
    correlation_id = _chat_correlation_id(req)
    llm_trace_id = item.llm_trace_id or llm_tracing.build_trace_id()
    item.llm_trace_id = llm_trace_id
    scope = _service_event_scope(req)
    turn_started_at = time.perf_counter()
    stages_reached: list[str] = []
    scheduled_followup_count = 0
    debug_mode_names: list[str] = []
    settlement_items = [item]
    if settlement_fragments:
        settlement_items = [
            fragment.queue_item
            for fragment in settlement_fragments
            if isinstance(fragment.queue_item, QueuedChatItem)
        ]
        if settlement_items:
            item.collapsed_items = [
                queued_item
                for queued_item in settlement_items
                if queued_item is not item
            ]

    try:
        character_global_user_id = await _ensure_character_global_identity(
            platform=req.platform,
            platform_bot_id=req.platform_bot_id,
            character_name=character_name,
        )
        if item.global_user_id:
            global_user_id = item.global_user_id
            user_profile = dict(item.user_profile)
        else:
            global_user_id, user_profile = await _resolve_queued_user(item)
        message_envelope = item.resolved_message_envelope
        if not isinstance(message_envelope, Mapping):
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
        media_scope = (
            req.platform,
            req.platform_channel_id,
            global_user_id,
        )
        begin_session_turn(media_scope)
        if prepared_media is not None:
            multimedia_input = [dict(row) for row in prepared_media]
        else:
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
        put_session_media(
            media_scope,
            [
                {
                    "media_kind": "image",
                    "content_type": item["content_type"],
                    "base64_data": item["base64_data"],
                    "source_summary": item["description"],
                }
                for item in multimedia_input
                if item["content_type"].startswith("image/")
                and bool(item["base64_data"])
            ],
        )

        history = await get_conversation_history(
            platform=req.platform,
            platform_channel_id=req.platform_channel_id,
            limit=CONVERSATION_HISTORY_LIMIT,
        )
        chat_history_wide = trim_history_dict(history)
        chat_history_recent = chat_history_wide[-CHAT_HISTORY_RECENT_LIMIT:]
        reply_context = await _hydrate_reply_context(req)
        stages_reached.append("history_loaded")
        user_save_started_at = 0.0
        if skip_user_persist and item.conversation_row_id:
            conversation_row_id = item.conversation_row_id
        else:
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

        if settlement_fragments:
            user_input = "\n".join(
                fragment.body_text
                for fragment in settlement_fragments
                if fragment.body_text.strip()
            )
        else:
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
            "additional_media_present": bool(
                additional_media_present
            ),
            "media_prepared": media_prepared,
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
            "response_action": (
                settled_decision.get("response_action")
                if isinstance(settled_decision, Mapping)
                else None
            ),
            "observation_status": "observation_complete",
            "turn_id": settlement_turn_id,
            "turn_version": settlement_version,
            "cognition_claimed": settlement_claimed,
            "should_respond": initial_should_respond,
            "reason_to_respond": (
                str(settled_decision.get("reason_to_respond", ""))
                if isinstance(settled_decision, Mapping)
                else ""
            ),
            "use_reply_feature": (
                bool(settled_decision.get("use_reply_feature", False))
                if isinstance(settled_decision, Mapping)
                else False
            ),
            "channel_topic": (
                str(settled_decision.get("channel_topic", ""))
                if isinstance(settled_decision, Mapping)
                else ""
            ),
            "indirect_speech_context": (
                str(settled_decision.get("indirect_speech_context", ""))
                if isinstance(settled_decision, Mapping)
                else ""
            ),
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
        reply_owner_is_effective_latest = (
            not settlement_fragments
            or settlement_fragments[-1].arrival_sequence == item.sequence
        )
        use_reply_feature = (
            bool(final_dialog)
            and bool(result["use_reply_feature"])
            and reply_owner_is_effective_latest
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
        if settlement_turn_id and response_dialog:
            await _turn_settlement_coordinator.record_bot_continuity(
                scope=(req.platform, req.platform_channel_id, req.channel_type),
                author_platform_user_id=req.platform_user_id,
                dialog_text="\n".join(response_dialog),
            )
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
                for queued_item in [
                    *dequeued_turn.dropped_items,
                    *(item for item, _survivor in dequeued_turn.collapsed_items),
                    *(
                        [dequeued_turn.next_item]
                        if dequeued_turn.next_item is not None
                        else []
                    ),
                ]:
                    await _turn_settlement_coordinator.complete_ingress(
                        queued_item.sequence,
                    )
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
                await _frontline_intake_item(dequeued_turn.next_item)
        finally:
            _primary_interaction_active_count -= 1


def _ensure_chat_input_worker_started() -> None:
    """Ensure the process-local input worker exists for the current event loop.

    Returns:
        None.
    """

    global _chat_queue_worker_task, _turn_settlement_worker_task
    if _chat_queue_worker_task is None or _chat_queue_worker_task.done():
        _chat_queue_worker_task = asyncio.create_task(_chat_input_worker())
    if (
        _turn_settlement_worker_task is None
        or _turn_settlement_worker_task.done()
    ):
        _turn_settlement_worker_task = asyncio.create_task(
            _turn_settlement_worker()
        )


async def _stop_chat_input_worker() -> None:
    """Stop the process-local input worker and resolve pending requests.

    Returns:
        None.
    """

    global _chat_input_queue, _chat_queue_worker_task
    global _turn_settlement_worker_task
    global _turn_settlement_coordinator

    current_loop = asyncio.get_running_loop()
    queue_task = _chat_queue_worker_task
    settlement_task = _turn_settlement_worker_task
    stale_loop = any(
        task is not None
        and not task.done()
        and task.get_loop() is not current_loop
        for task in (queue_task, settlement_task)
    )
    if stale_loop:
        _chat_queue_worker_task = None
        _turn_settlement_worker_task = None
        _chat_input_queue = ChatInputQueue()
        _turn_settlement_coordinator = _new_turn_settlement_coordinator()
        return

    task = _chat_queue_worker_task
    _chat_queue_worker_task = None
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    settlement_task = _turn_settlement_worker_task
    _turn_settlement_worker_task = None
    if settlement_task is not None:
        settlement_task.cancel()
        with suppress(asyncio.CancelledError):
            await settlement_task

    pending_items = await _chat_input_queue.drain()
    for item in pending_items:
        _chat_input_queue.complete(item, ChatResponse())
        await _release_queued_pipeline_handle(item)
    _chat_input_queue = ChatInputQueue()
    _turn_settlement_coordinator = _new_turn_settlement_coordinator()


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

    def _mark_item_enqueued(queued_item: QueuedChatItem) -> None:
        nonlocal item_enqueued
        item_enqueued = True
        if req.debug_modes.listen_only:
            return
        _turn_settlement_coordinator.register_ingress(
            sequence=queued_item.sequence,
            scope=(req.platform, req.platform_channel_id, req.channel_type),
            author_platform_user_id=req.platform_user_id,
            enqueue_monotonic=queued_item.enqueue_monotonic,
        )

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

        # 2. Hydrate persistent media descriptor cache
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
                        _deliver_accepted_task_result_episode
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
