"""Deterministic capability execution for cognition resolver requests."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from typing import Any, Literal
from uuid import uuid4

from openai import OpenAIError

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.cognition_episode import GoalContinuationRefV1
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    MAX_SHARED_MEMORY_PREWARM_LATENCY_MS,
    RESOLVER_EVIDENCE_STATE_VERSION,
    RESOLVER_OBSERVATION_VERSION,
    ResolverCapabilityRequestV1,
    ResolverObservationV1,
    ResolverValidationError,
    SharedMemoryPrewarmOutcomeV1,
    validate_resolver_capability_request,
    validate_resolver_observation,
    validate_shared_memory_prewarm_outcome,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
    CognitionContractError,
    CognitionEvidenceV2,
    DirectFactV2,
    ResolverCapabilityRequestV2,
    _validate_scene_context,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    CODING_AGENT_WORKSPACE_ROOT,
    TASK_RESOLUTION_INLINE_BUDGET_SECONDS,
)
from kazusa_ai_chatbot.db.errors import DatabaseBackendError
from kazusa_ai_chatbot.local_context_resolver import (
    DEFAULT_OPTION_LIMITS,
    LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
    LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
    LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
    project_local_context_packet,
    resolve_local_context,
)
from kazusa_ai_chatbot.local_context_resolver.contracts import (
    LocalContextResolverContextV1,
    LocalContextResolverOptionsV1,
    LocalContextResolverRequestV1,
)
from kazusa_ai_chatbot.media_inspection.session_cache import list_session_media_refs
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_projection import (
    project_known_facts,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.nodes.referent_resolution import (
    should_skip_rag_for_unresolved_referents,
    unresolved_referent_reason,
)
from kazusa_ai_chatbot.rag.cognitive_episode_adapter import (
    build_text_chat_rag_request,
)
from kazusa_ai_chatbot.rag.memory_evidence.workers.persistent_search import (
    PersistentMemorySearchAgent,
)
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)
from kazusa_ai_chatbot.task_resolution.contracts import (
    MAX_TASK_RESOLUTION_TEXT_CHARS,
    MAX_TASK_RESOLUTION_TEXT_ITEMS,
    TASK_RESOLUTION_EXECUTION_CONTEXT_VERSION,
    TaskResolutionContractError,
    TaskResolutionExecutionContextV1,
    TaskResolutionResultV1,
    validate_task_resolution_execution_context,
    validate_task_resolution_result,
)
from kazusa_ai_chatbot.task_resolution.service import (
    promote_deferred_task_resolution,
    resolve_task_inline,
    start_task_resolution_in_background,
)
from kazusa_ai_chatbot.utils import log_preview, text_or_empty

MILLISECONDS_PER_SECOND = 1000
PERSONA_RAG_COMPONENT = "nodes.persona_supervisor2"
SELF_GOAL_ALLOWED_TRIGGER_SOURCES = frozenset((
    "internal_thought",
    "self_cognition",
))
SHARED_MEMORY_SUMMARY_FIELDS = (
    "content",
    "description",
    "text",
    "summary",
    "fact",
)

logger = logging.getLogger(__name__)

RecordRagEventFunc = Callable[..., Awaitable[None]]


def project_resolver_observation_for_cognition(
    observation: Mapping[str, object],
    *,
    occurred_at: str,
) -> tuple[CognitionEvidenceV2, list[DirectFactV2]]:
    """Project one resolver result into typed evidence without state authority."""

    observation_id = text_or_empty(observation.get("observation_id")).strip()
    summary = text_or_empty(
        observation.get("semantic_summary")
        or observation.get("prompt_safe_summary")
    ).strip()
    capability = text_or_empty(
        observation.get("capability")
        or observation.get("capability_kind")
    ).strip()
    if not observation_id:
        raise ResolverValidationError("resolver observation id is required")
    if not summary:
        raise ResolverValidationError("resolver observation summary is required")
    semantic_segments = [
        f"{capability}: {summary}" if capability else summary,
    ]
    evidence_state = observation.get("task_resolution_evidence_state")
    factual_evidence_state = ""
    if isinstance(evidence_state, Mapping):
        state = text_or_empty(evidence_state.get("state")).strip()
        factual_evidence_state = state
        if state:
            semantic_segments.append(f"evidence_state={state}")
        remaining_needs = evidence_state.get("remaining_needs")
        if isinstance(remaining_needs, list):
            needs = [
                item.strip()
                for item in remaining_needs
                if isinstance(item, str) and item.strip()
            ]
            if needs:
                semantic_segments.append(
                    "remaining_needs=" + " | ".join(needs[:4])
                )
    raw_evidence_refs = observation.get("evidence_refs")
    if (
        factual_evidence_state in {"complete", "partial"}
        and isinstance(raw_evidence_refs, list)
    ):
        evidence_excerpts = [
            excerpt.strip()
            for evidence_ref in raw_evidence_refs
            if isinstance(evidence_ref, Mapping)
            for excerpt in [evidence_ref.get("excerpt")]
            if isinstance(excerpt, str) and excerpt.strip()
        ][:4]
        if evidence_excerpts:
            semantic_segments.append(
                "evidence_excerpts=" + " | ".join(evidence_excerpts)
            )
    semantic_text = "; ".join(semantic_segments)
    evidence = CognitionEvidenceV2(
        evidence_handle="e1",
        evidence_ref={
            "source_kind": "resolver_observation",
            "source_id": observation_id,
            "occurred_at": occurred_at,
            "semantic_summary": summary[:500],
        },
        semantic_text=semantic_text[:1000],
        visible_to=list(
            EVIDENCE_SOURCE_QUESTION_IDS["resolver_observation"]
        ),
        authority="contextual_fact_only",
    )
    direct_facts: list[DirectFactV2] = []
    return_value = (evidence, direct_facts)
    return return_value


async def run_rag_evidence_for_persona_state(
    state: GlobalPersonaState,
    *,
    agent_name: str,
    objective: str | None = None,
    reason: str | None = None,
    record_rag_stage_event_func: RecordRagEventFunc | None = None,
    component: str = PERSONA_RAG_COMPONENT,
) -> dict[str, Any]:
    """Run local-context resolver evidence for one persona objective."""

    started_at = time.perf_counter()
    correlation_id = _rag_correlation_id(state)
    if record_rag_stage_event_func is None:
        record_rag_stage_event_func = event_logging.record_rag_stage_event

    referents = state["referents"]
    if should_skip_rag_for_unresolved_referents(referents):
        referent_reason = unresolved_referent_reason(referents)
        rag_result = _empty_projected_rag_result(state)
        logger.info(
            f"Local context recall skipped: reason={log_preview(referent_reason)}"
        )
        logger.debug(
            f'Local context recall skipped metadata: platform={state["platform"]} '
            f'channel={state["platform_channel_id"] or "<dm>"} '
            f'user={state["global_user_id"]} '
            f'query={log_preview(state["decontextualized_input"])} '
            f"rag_result={log_preview(rag_result)}"
        )
        await _record_rag_event(
            record_rag_stage_event_func,
            component=component,
            correlation_id=correlation_id,
            agent_name=agent_name,
            status="skipped",
            slot_count=0,
            retrieval_count=0,
            latency_ms=_elapsed_ms(started_at),
        )
        return_value = rag_result
        return return_value

    fresh_query = _fresh_query_for_objective(
        objective,
        fallback_query=state["decontextualized_input"],
    )
    request_reason = text_or_empty(reason)
    if not request_reason:
        request_reason = "Cognition requested local context evidence."
    resolver_request: LocalContextResolverRequestV1 = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
        "objective": fresh_query,
        "source": "l2d",
        "reason": request_reason,
        "priority": "normal",
    }
    resolver_context = _local_context_resolver_context_from_state(state)
    resolver_options: LocalContextResolverOptionsV1 = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
        **DEFAULT_OPTION_LIMITS,
    }
    packet = await resolve_local_context(
        resolver_request,
        resolver_context,
        resolver_options,
    )
    rag_result = project_local_context_packet(packet)
    trace = rag_result["supervisor_trace"]
    logger.info(
        f'Local context projection output: answer={log_preview(rag_result["answer"])}'
    )
    logger.debug(
        f'Local context projection metadata: platform={state["platform"]} '
        f'channel={state["platform_channel_id"] or "<dm>"} '
        f'user={state["global_user_id"]} '
        f'query={log_preview(fresh_query)} '
        f'user_image={bool(rag_result["user_image"])} '
        f'character_image={bool(rag_result["character_image"])} '
        f'third_party_profiles={len(rag_result["third_party_profiles"])} '
        f'memory_evidence={len(rag_result["memory_evidence"])} '
        f'recall_evidence={len(rag_result["recall_evidence"])} '
        f'conversation_evidence={len(rag_result["conversation_evidence"])} '
        f'external_evidence={len(rag_result["external_evidence"])} '
        f'trace={log_preview(trace)} '
        f"rag_result={log_preview(rag_result)}"
    )
    retrieval_count = _retrieval_count(rag_result)
    safety_recovery_incidents = _safety_recovery_incidents(rag_result)
    safety_recovery_first = (
        safety_recovery_incidents[0]
        if safety_recovery_incidents
        else ""
    )
    execution_status = _local_context_execution_status(rag_result)
    await _record_rag_event(
        record_rag_stage_event_func,
        component=component,
        correlation_id=correlation_id,
        agent_name=agent_name,
        status=execution_status,
        slot_count=_local_context_evidence_node_count(packet),
        retrieval_count=retrieval_count,
        latency_ms=_elapsed_ms(started_at),
        cache_hit=_local_context_cache_hit(packet),
        safety_recovery_count=len(safety_recovery_incidents),
        safety_recovery_first=safety_recovery_first,
    )
    return_value = rag_result
    return return_value


def build_skipped_shared_memory_prewarm_outcome(
    reason_code: Literal["not_first_cycle", "unsupported_episode"],
) -> SharedMemoryPrewarmOutcomeV1:
    """Build the canonical outcome for a prewarm that was not eligible."""

    if reason_code not in {"not_first_cycle", "unsupported_episode"}:
        raise ResolverValidationError("prewarm skip reason is invalid")
    outcome = _build_shared_memory_prewarm_outcome(
        status="skipped",
        reason_code=reason_code,
        attempted=False,
        latency_ms=0,
        rag_result=_empty_projected_rag_result(),
    )
    return outcome


def _build_shared_memory_prewarm_outcome(
    *,
    status: str,
    reason_code: str,
    attempted: bool,
    latency_ms: int,
    rag_result: Mapping[str, object],
    retrieved_shared_count: int = 0,
    merged_shared_count: int = 0,
) -> SharedMemoryPrewarmOutcomeV1:
    """Validate one internal prewarm candidate before exposing it to state."""

    candidate = {
        "schema_version": "shared_memory_prewarm_outcome.v1",
        "status": status,
        "reason_code": reason_code,
        "attempted": attempted,
        "latency_ms": latency_ms,
        "retrieved_shared_count": retrieved_shared_count,
        "merged_shared_count": merged_shared_count,
        "rag_result": dict(rag_result),
    }
    validated = validate_shared_memory_prewarm_outcome(candidate)
    return_value = validated
    return return_value


def _prewarm_elapsed_ms(started_at: float) -> int:
    """Return monotonic prewarm latency clipped to its diagnostic bound."""

    elapsed = time.perf_counter() - started_at
    elapsed_ms = int(max(0.0, elapsed) * MILLISECONDS_PER_SECOND)
    bounded_elapsed_ms = min(elapsed_ms, MAX_SHARED_MEMORY_PREWARM_LATENCY_MS)
    return_value = bounded_elapsed_ms
    return return_value


async def run_first_cycle_shared_memory_prewarm(
    state: GlobalPersonaState,
) -> SharedMemoryPrewarmOutcomeV1:
    """Run one bounded shared-memory prewarm and retain its disposition.

    Args:
        state: Persona state after decontextualization and resolver input
            initialization.

    Returns:
        A validated outcome whose RAG payload contains only safe shared-memory
        evidence and whose disposition describes the worker and projection.
    """

    started_at = time.perf_counter()
    empty_rag_result = _empty_projected_rag_result(state)
    prewarm_task, prompt_message_context = _prewarm_request_projection(state)
    if not prewarm_task:
        return_value = _build_shared_memory_prewarm_outcome(
            status="skipped",
            reason_code="empty_query_after_character_mention",
            attempted=False,
            latency_ms=0,
            rag_result=empty_rag_result,
        )
        return return_value

    rag_request = build_text_chat_rag_request(
        episode=state["cognitive_episode"],
        decontextualized_input=prewarm_task,
        character_profile=state["character_profile"],
        user_profile=state["user_profile"],
        prompt_message_context=prompt_message_context,
        channel_topic=state["channel_topic"],
        chat_history_recent=_prewarm_history_projection(
            state["chat_history_recent"],
            state,
        ),
        chat_history_wide=_prewarm_history_projection(
            state["chat_history_wide"],
            state,
        ),
        reply_context=state["reply_context"],
        indirect_speech_context=state["indirect_speech_context"],
        conversation_progress=state.get("conversation_progress"),
        conversation_episode_state=state.get("conversation_episode_state"),
        promoted_reflection_context=state.get("promoted_reflection_context"),
        llm_trace_id=str(state.get("llm_trace_id", "")),
    )

    try:
        worker_result = await PersistentMemorySearchAgent().run(
            task=rag_request["original_query"],
            context=rag_request["context"],
            max_attempts=1,
        )
    except (OpenAIError, DatabaseBackendError, TimeoutError) as exc:
        logger.warning(f"Shared memory prewarm worker failed: {exc}")
        return_value = _build_shared_memory_prewarm_outcome(
            status="failed",
            reason_code="worker_error",
            attempted=True,
            latency_ms=_prewarm_elapsed_ms(started_at),
            rag_result=empty_rag_result,
        )
        return return_value

    if not isinstance(worker_result, Mapping):
        return_value = _build_shared_memory_prewarm_outcome(
            status="failed",
            reason_code="worker_contract_invalid",
            attempted=True,
            latency_ms=_prewarm_elapsed_ms(started_at),
            rag_result=empty_rag_result,
        )
        return return_value
    resolved = worker_result.get("resolved")
    if not isinstance(resolved, bool):
        return_value = _build_shared_memory_prewarm_outcome(
            status="failed",
            reason_code="worker_contract_invalid",
            attempted=True,
            latency_ms=_prewarm_elapsed_ms(started_at),
            rag_result=empty_rag_result,
        )
        return return_value
    if resolved is False:
        return_value = _build_shared_memory_prewarm_outcome(
            status="empty",
            reason_code="worker_unresolved",
            attempted=True,
            latency_ms=_prewarm_elapsed_ms(started_at),
            rag_result=empty_rag_result,
        )
        return return_value

    raw_rows = worker_result.get("result")
    if not isinstance(raw_rows, list):
        return_value = _build_shared_memory_prewarm_outcome(
            status="failed",
            reason_code="worker_contract_invalid",
            attempted=True,
            latency_ms=_prewarm_elapsed_ms(started_at),
            rag_result=empty_rag_result,
        )
        return return_value

    shared_rows = _shared_memory_prewarm_rows(raw_rows)
    if not shared_rows:
        return_value = _build_shared_memory_prewarm_outcome(
            status="empty",
            reason_code="no_shared_memory",
            attempted=True,
            latency_ms=_prewarm_elapsed_ms(started_at),
            rag_result=empty_rag_result,
        )
        return return_value

    summary = _shared_memory_prewarm_summary(shared_rows)
    known_facts = [
        {
            "slot": rag_request["original_query"],
            "agent": "persistent_memory_search_agent",
            "resolved": True,
            "summary": summary,
            "raw_result": shared_rows,
        }
    ]
    try:
        rag_result = project_known_facts(
            known_facts,
            current_user_id=rag_request["current_user_id"],
            character_user_id=rag_request["character_user_id"],
            answer="",
            unknown_slots=[],
            loop_count=0,
        )
        rag_result["user_memory_unit_candidates"] = []
        rag_result["recall_evidence"] = []
        rag_result["conversation_evidence"] = []
        rag_result["external_evidence"] = []
        retrieved_shared_count = len(rag_result["memory_evidence"])
        if retrieved_shared_count == 0:
            return_value = _build_shared_memory_prewarm_outcome(
                status="empty",
                reason_code="no_shared_memory",
                attempted=True,
                latency_ms=_prewarm_elapsed_ms(started_at),
                rag_result=empty_rag_result,
            )
        else:
            return_value = _build_shared_memory_prewarm_outcome(
                status="completed",
                reason_code="shared_memory_ready",
                attempted=True,
                latency_ms=_prewarm_elapsed_ms(started_at),
                retrieved_shared_count=retrieved_shared_count,
                rag_result=rag_result,
            )
    except (KeyError, TypeError, ValueError, ResolverValidationError) as exc:
        logger.warning(f"Shared memory prewarm projection failed: {exc}")
        return_value = _build_shared_memory_prewarm_outcome(
            status="failed",
            reason_code="projection_failed",
            attempted=True,
            latency_ms=_prewarm_elapsed_ms(started_at),
            rag_result=empty_rag_result,
        )
    return return_value


def _prewarm_request_projection(
    state: GlobalPersonaState,
) -> tuple[str, dict[str, Any]]:
    """Copy and structurally project the active-character request for prewarm."""

    prompt_message_context = deepcopy(state["prompt_message_context"])
    character_profile = state["character_profile"]
    character_global_user_id = character_profile.get("global_user_id")
    if not isinstance(character_global_user_id, str) or not character_global_user_id:
        return state["decontextualized_input"].strip(), prompt_message_context

    prewarm_task = state["decontextualized_input"]
    mentions = prompt_message_context.get("mentions", [])
    projected_mentions: list[object] = []
    if not isinstance(mentions, list):
        mentions = []
    for mention in mentions:
        if not isinstance(mention, Mapping):
            projected_mentions.append(mention)
            continue
        is_active_character = (
            mention.get("entity_kind") == "bot"
            and isinstance(mention.get("global_user_id"), str)
            and bool(mention.get("global_user_id"))
            and mention.get("global_user_id") == character_global_user_id
        )
        display_name = mention.get("display_name")
        if (
            not is_active_character
            or not isinstance(display_name, str)
            or not display_name.strip()
        ):
            projected_mentions.append(mention)
            continue
        mention_token = f"@{display_name.strip()}"
        mention_pattern = re.compile(
            rf"(?<!\S){re.escape(mention_token)}(?!\S)"
        )
        prewarm_task = mention_pattern.sub("", prewarm_task, count=1)
        body_text = prompt_message_context.get("body_text")
        if isinstance(body_text, str):
            prompt_message_context["body_text"] = mention_pattern.sub(
                "",
                body_text,
                count=1,
            ).strip()
    prompt_message_context["mentions"] = projected_mentions
    return prewarm_task.strip(), prompt_message_context


def _prewarm_history_projection(
    history: list[dict[str, Any]],
    state: GlobalPersonaState,
) -> list[dict[str, Any]]:
    """Copy history and omit only typed active-turn message or row IDs."""

    platform_message_ids = set(_prewarm_active_turn_ids(
        state,
        "active_turn_platform_message_ids",
    ))
    current_message_id = text_or_empty(state.get("platform_message_id"))
    if current_message_id:
        platform_message_ids.add(current_message_id)
    conversation_row_ids = set(_prewarm_active_turn_ids(
        state,
        "active_turn_conversation_row_ids",
    ))
    projected: list[dict[str, Any]] = []
    for row in deepcopy(history):
        platform_message_id = text_or_empty(row.get("platform_message_id"))
        conversation_row_id = text_or_empty(row.get("conversation_row_id"))
        if (
            platform_message_id in platform_message_ids
            or conversation_row_id in conversation_row_ids
        ):
            continue
        projected.append(row)
    return projected


def _prewarm_active_turn_ids(
    state: GlobalPersonaState,
    field_name: str,
) -> list[str]:
    """Collect typed active-turn IDs from state and episode origin metadata."""

    values = _string_list(state.get(field_name))
    episode = state.get("cognitive_episode")
    if isinstance(episode, Mapping):
        origin_metadata = episode.get("origin_metadata")
        if isinstance(origin_metadata, Mapping):
            for value in _string_list(origin_metadata.get(field_name)):
                if value not in values:
                    values.append(value)
    return values


def merge_shared_memory_prewarm_outcome(
    base_rag_result: dict[str, Any],
    outcome: SharedMemoryPrewarmOutcomeV1,
) -> tuple[dict[str, Any], SharedMemoryPrewarmOutcomeV1]:
    """Append one validated shared-memory outcome to the caller's RAG state.

    Args:
        base_rag_result: Existing cognition RAG payload.
        outcome: Ready prewarm outcome whose evidence is appended in source
            order and then finalized as merged.

    Returns:
        A deep-copied merged RAG payload and its finalized merged outcome.
    """

    validated_outcome = validate_shared_memory_prewarm_outcome(outcome)
    if (
        validated_outcome["status"] != "completed"
        or validated_outcome["reason_code"] != "shared_memory_ready"
    ):
        raise ResolverValidationError("prewarm_outcome_not_ready")
    if not isinstance(base_rag_result, Mapping):
        raise ResolverValidationError("base_rag_result_invalid")
    if (
        "memory_evidence" in base_rag_result
        and not isinstance(base_rag_result["memory_evidence"], list)
    ):
        raise ResolverValidationError("base_rag_result_invalid")
    prewarm_memory_evidence = validated_outcome["rag_result"][
        "memory_evidence"
    ]
    base_memory_evidence = base_rag_result.get("memory_evidence", [])
    if not isinstance(base_memory_evidence, list):
        raise ResolverValidationError("base_rag_result_invalid")
    merged_memory_evidence = deepcopy(base_memory_evidence)
    merged_memory_evidence.extend(deepcopy(prewarm_memory_evidence))

    merged = deepcopy(dict(base_rag_result))
    merged["memory_evidence"] = merged_memory_evidence
    finalized_candidate = deepcopy(dict(validated_outcome))
    finalized_candidate["reason_code"] = "shared_memory_merged"
    finalized_candidate["merged_shared_count"] = len(prewarm_memory_evidence)
    finalized_outcome = validate_shared_memory_prewarm_outcome(
        finalized_candidate
    )
    return_value = merged, finalized_outcome
    return return_value


async def execute_resolver_capability_request(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
) -> ResolverObservationV1:
    """Execute one deterministic resolver capability request."""

    validated_request = validate_resolver_capability_request(request)
    capability_kind = validated_request["capability_kind"]
    if capability_kind == "task_resolution_request":
        observation = await _execute_task_resolution_request(
            validated_request,
            state,
        )
        return observation
    if capability_kind == "human_clarification":
        observation = _blocked_observation(
            validated_request,
            state,
            summary_prefix="Human clarification required",
        )
        return observation
    if capability_kind == "approval_preparation":
        observation = _blocked_observation(
            validated_request,
            state,
            summary_prefix="Approval required before side effects",
        )
        return observation
    if capability_kind == "self_goal_resolution":
        observation = _self_goal_resolution_observation(validated_request, state)
        return observation

    raise ResolverValidationError(f"unsupported capability: {capability_kind}")


def validate_task_resolution_execution_readiness(
    state: GlobalPersonaState,
    *,
    cognition_scene_context: Mapping[str, object],
) -> None:
    """Validate every deterministic input required by task execution.

    Capability advertisement and execution use this same contract so a
    resolver request is advertised only when its executor can construct the
    task context without consulting the relevance-owned scene string.
    """

    _validate_cognition_scene_context(cognition_scene_context)
    character_profile = state.get("character_profile")
    if not isinstance(character_profile, Mapping):
        raise ResolverValidationError(
            "character_profile: expected trusted mapping"
        )
    character_name = text_or_empty(character_profile.get("name")).strip()
    if not character_name:
        raise ResolverValidationError(
            "character_profile.name: expected non-empty state text"
        )
    for field_name in (
        "platform",
        "platform_channel_id",
        "channel_type",
        "global_user_id",
        "platform_user_id",
        "platform_message_id",
        "storage_timestamp_utc",
        "platform_bot_id",
        "user_name",
    ):
        _required_state_text(state, field_name)
    for field_name in ("local_time_context", "prompt_message_context"):
        value = state.get(field_name)
        if not isinstance(value, Mapping):
            raise ResolverValidationError(f"{field_name}: expected mapping")
    for field_name in ("chat_history_recent", "chat_history_wide"):
        try:
            _history_rows(state.get(field_name))
        except ResolverValidationError as exc:
            raise ResolverValidationError(
                f"{field_name}: invalid history carrier: {exc}"
            ) from exc
    if (
        not isinstance(BACKGROUND_WORK_OUTPUT_CHAR_LIMIT, int)
        or isinstance(BACKGROUND_WORK_OUTPUT_CHAR_LIMIT, bool)
        or BACKGROUND_WORK_OUTPUT_CHAR_LIMIT <= 0
    ):
        raise ResolverValidationError(
            "background work output limit: expected positive integer"
        )
    _cognitive_episode_trigger_source(state)


async def _execute_task_resolution_request(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
) -> ResolverObservationV1:
    """Run the one L2d-visible task-resolution capability route.

    A background-priority request enters the direct durable handoff path with
    no inline specialist invocation.  A now-priority request runs inline first
    under the approximate foreground budget and defers only when it cannot
    finish.
    """

    continuation_ref = _task_continuation_ref(request)
    execution_context = _task_resolution_execution_context_from_state(
        state,
        goal_continuation_ref=continuation_ref,
    )
    task_request: ResolverCapabilityRequestV2 = {
        "capability": "task_resolution_request",
        "semantic_goal": request["objective"],
        "reason": request["reason"],
        "evidence_handles": [],
        "start_in_background": request["priority"] == "background",
        "goal_continuation_ref": continuation_ref,
    }
    if request["priority"] == "background":
        try:
            deferred_result = await start_task_resolution_in_background(
                task_request,
                execution_context,
                source_trigger_source=_cognitive_episode_trigger_source(state),
                source_platform_bot_id=_required_state_text(
                    state,
                    "platform_bot_id",
                ),
                requester_display_name=_required_state_text(state, "user_name"),
                source_llm_trace_id=text_or_empty(state.get("llm_trace_id")),
            )
        except TaskResolutionContractError as exc:
            return _task_resolution_failure_observation(request, state, exc)
        return _task_resolution_observation(
            request,
            state,
            deferred_result,
            durably_promoted=True,
        )
    try:
        result = await resolve_task_inline(
            task_request,
            execution_context,
            inline_budget_seconds=TASK_RESOLUTION_INLINE_BUDGET_SECONDS,
        )
    except TaskResolutionContractError as exc:
        return _task_resolution_failure_observation(request, state, exc)

    if result["status"] == "deferred":
        try:
            await promote_deferred_task_resolution(
                result,
                execution_context,
                source_trigger_source=_cognitive_episode_trigger_source(state),
                source_platform_bot_id=_required_state_text(
                    state,
                    "platform_bot_id",
                ),
                requester_display_name=_required_state_text(state, "user_name"),
                source_llm_trace_id=text_or_empty(state.get("llm_trace_id")),
            )
        except TaskResolutionContractError as exc:
            return _task_resolution_failure_observation(request, state, exc)
    return _task_resolution_observation(
        request,
        state,
        result,
        durably_promoted=result["status"] == "deferred",
    )


def _task_resolution_execution_context_from_state(
    state: GlobalPersonaState,
    *,
    goal_continuation_ref: GoalContinuationRefV1,
) -> TaskResolutionExecutionContextV1:
    """Project trusted persona state into the exact specialist context shape."""

    cognition_scene_context = _required_scene_context_from_state(state)
    validate_task_resolution_execution_readiness(
        state,
        cognition_scene_context=cognition_scene_context,
    )
    character_profile = state["character_profile"]
    character_name = text_or_empty(character_profile.get("name"))
    if not character_name:
        raise ResolverValidationError("character_profile.name: expected string")
    conversation_progress = state.get("conversation_progress")
    progress_context = (
        dict(conversation_progress)
        if isinstance(conversation_progress, Mapping)
        else {}
    )
    context: TaskResolutionExecutionContextV1 = {
        "schema_version": TASK_RESOLUTION_EXECUTION_CONTEXT_VERSION,
        "character_name": character_name,
        "platform": _required_state_text(state, "platform"),
        "channel_id": _required_state_text(state, "platform_channel_id"),
        "channel_type": _required_state_text(state, "channel_type"),
        "requester_global_user_id": _required_state_text(
            state,
            "global_user_id",
        ),
        "requester_platform_user_id": _required_state_text(
            state,
            "platform_user_id",
        ),
        "source_message_id": _required_state_text(
            state,
            "platform_message_id",
        ),
        "scene_context": cognition_scene_context,
        "goal_continuation_ref": goal_continuation_ref,
        "local_time_context": dict(state["local_time_context"]),
        "prompt_message_context": dict(state["prompt_message_context"]),
        "chat_history_recent": _history_rows(state["chat_history_recent"])[
            -MAX_TASK_RESOLUTION_TEXT_ITEMS:
        ],
        "chat_history_wide": _history_rows(state["chat_history_wide"])[
            -MAX_TASK_RESOLUTION_TEXT_ITEMS:
        ],
        "conversation_progress": progress_context,
        "persona_summary": _task_resolution_persona_summary(state),
        "conversation_summary": text_or_empty(
            state.get("decontextualized_input"),
        )[:MAX_TASK_RESOLUTION_TEXT_CHARS],
        "current_timestamp_utc": _required_state_text(
            state,
            "storage_timestamp_utc",
        ),
        "active_turn_platform_message_ids": (
            _active_turn_platform_message_ids(state)
        ),
        "active_turn_conversation_row_ids": _string_list(
            state.get("active_turn_conversation_row_ids"),
        ),
        "session_media_refs": list_session_media_refs((
            state["platform"],
            state["platform_channel_id"],
            state["global_user_id"],
        )),
        "coding_workspace_root": text_or_empty(CODING_AGENT_WORKSPACE_ROOT),
        "max_output_chars": BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    }
    validated_context = validate_task_resolution_execution_context(context)
    return validated_context


def _task_resolution_observation(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
    result: TaskResolutionResultV1,
    *,
    durably_promoted: bool,
) -> ResolverObservationV1:
    """Map one validated task result into the resolver recurrence contract."""

    validated_result = validate_task_resolution_result(result)
    _validate_task_result_request_binding(request, validated_result)
    status = validated_result["status"]
    if status == "deferred":
        if not durably_promoted:
            raise ResolverValidationError(
                "deferred task result requires durable promotion"
            )
        observation = _observation_base(
            request,
            state,
            status="succeeded",
            prompt_safe_summary=(
                "The bounded task was accepted for continued work; its later "
                "result will return through the normal conversation path."
            ),
        )
        observation["task_resolution_evidence_state"] = {
            "schema_version": RESOLVER_EVIDENCE_STATE_VERSION,
            "state": validated_result["evidence_state"],
            "remaining_needs": list(validated_result["remaining_needs"]),
        }
        validated_observation = validate_resolver_observation(observation)
        return validated_observation
    if status in {"resolved", "partial"}:
        observation = _observation_base(
            request,
            state,
            status="succeeded",
            prompt_safe_summary=validated_result["prompt_safe_summary"],
        )
        observation["task_resolution_evidence_state"] = {
            "schema_version": RESOLVER_EVIDENCE_STATE_VERSION,
            "state": validated_result["evidence_state"],
            "remaining_needs": list(validated_result["remaining_needs"]),
        }
        observation["evidence_refs"] = _task_resolution_evidence_refs(
            validated_result,
            observed_at=_created_at_utc(state),
        )
        observation["knowledge_projection"] = {
            "investigation_summary": validated_result["prompt_safe_summary"],
            "knowledge_we_know_so_far": list(
                validated_result["evidence_excerpts"]
            ),
            "knowledge_still_lacking": list(
                validated_result["remaining_needs"]
            ),
            "recommended_next_iteration": [],
            "evidence_boundary_notes": _task_resolution_limitations(
                validated_result,
            ),
        }
        validated_observation = validate_resolver_observation(observation)
        return validated_observation
    if status == "needs_user_input":
        observation = _observation_base(
            request,
            state,
            status="blocked",
            prompt_safe_summary=validated_result["prompt_safe_summary"],
        )
        observation["blocker_kind"] = "requires_user_input"
        observation["task_resolution_evidence_state"] = {
            "schema_version": RESOLVER_EVIDENCE_STATE_VERSION,
            "state": validated_result["evidence_state"],
            "remaining_needs": list(validated_result["remaining_needs"]),
        }
        validated_observation = validate_resolver_observation(observation)
        return validated_observation
    if status == "approval_required":
        observation = _observation_base(
            request,
            state,
            status="blocked",
            prompt_safe_summary=validated_result["prompt_safe_summary"],
        )
        observation["task_resolution_evidence_state"] = {
            "schema_version": RESOLVER_EVIDENCE_STATE_VERSION,
            "state": validated_result["evidence_state"],
            "remaining_needs": list(validated_result["remaining_needs"]),
        }
        validated_observation = validate_resolver_observation(observation)
        return validated_observation
    if status in {"unavailable", "failed"}:
        observation = _observation_base(
            request,
            state,
            status="failed",
            prompt_safe_summary=validated_result["prompt_safe_summary"],
        )
        observation["task_resolution_evidence_state"] = {
            "schema_version": RESOLVER_EVIDENCE_STATE_VERSION,
            "state": validated_result["evidence_state"],
            "remaining_needs": list(validated_result["remaining_needs"]),
        }
        validated_observation = validate_resolver_observation(observation)
        return validated_observation
    raise ResolverValidationError("task-resolution result status is unsupported")


def _task_resolution_failure_observation(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
    exc: TaskResolutionContractError,
) -> ResolverObservationV1:
    """Return a bounded failure observation without exposing internal errors."""

    logger.warning(f"Task-resolution capability failed validation: {exc}")
    observation = _observation_base(
        request,
        state,
        status="failed",
        prompt_safe_summary=(
            "The bounded task could not complete through its available "
            "resolution path."
        ),
    )
    observation["task_resolution_evidence_state"] = {
        "schema_version": RESOLVER_EVIDENCE_STATE_VERSION,
        "state": "blocked",
        "remaining_needs": [request["objective"]],
    }
    validated_observation = validate_resolver_observation(observation)
    return validated_observation


def _blocked_observation(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
    *,
    summary_prefix: str,
) -> ResolverObservationV1:
    """Build a blocked observation for user-owned input or approval."""

    summary = f"{summary_prefix}: {request['objective']}"
    if request["capability_kind"] == "approval_preparation":
        summary = (
            f"{summary} Capability boundary: approval preparation only; "
            "no reminder, scheduling, sending, file inspection, status check, "
            "checksum validation, download monitoring, or other side effect "
            "has executed. Do not claim unavailable inspection tools unless "
            "the user or runtime explicitly provided them."
        )
    observation = _observation_base(
        request,
        state,
        status="blocked",
        prompt_safe_summary=summary,
    )
    return_value = validate_resolver_observation(observation)
    return return_value


def _self_goal_resolution_observation(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
) -> ResolverObservationV1:
    """Block user-message self-resolution and allow only internal sources."""

    trigger_source = _cognitive_episode_trigger_source(state)
    if trigger_source not in SELF_GOAL_ALLOWED_TRIGGER_SOURCES:
        observation = _observation_base(
            request,
            state,
            status="blocked",
            prompt_safe_summary=(
                "Self goal resolution is private-only and blocked for "
                "user-message source."
            ),
        )
        return_value = validate_resolver_observation(observation)
        return return_value

    observation = _observation_base(
        request,
        state,
        status="succeeded",
        prompt_safe_summary=(
            "Self goal resolution accepted for internal cognition source."
        ),
    )
    return_value = validate_resolver_observation(observation)
    return return_value


def _history_rows(rows: object) -> list[dict[str, object]]:
    """Copy required prompt-safe conversation rows for specialist adapters."""

    if not isinstance(rows, list):
        raise ResolverValidationError("chat history: expected list")
    copied_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ResolverValidationError("chat history: expected mapping rows")
        copied_rows.append(dict(row))
    return copied_rows


def _task_resolution_persona_summary(state: GlobalPersonaState) -> str:
    """Build compact character context without raw adapter identifiers."""

    character_profile = state["character_profile"]
    character_name = text_or_empty(character_profile.get("name"))
    if not character_name:
        raise ResolverValidationError("character_profile.name: expected string")
    segments = [f"active_character={character_name}"]
    channel_type = text_or_empty(state.get("channel_type"))
    if channel_type:
        segments.append(f"channel_type={channel_type}")
    user_name = text_or_empty(state.get("user_name"))
    if user_name:
        segments.append(f"current_user_display_name={user_name}")
    relationship_context = text_or_empty(state.get("logical_stance"))
    if relationship_context:
        segments.append(f"current_stance={relationship_context[:400]}")
    summary = "; ".join(segments)
    return summary


def _task_resolution_evidence_refs(
    result: TaskResolutionResultV1,
    *,
    observed_at: str,
) -> list[dict[str, object]]:
    """Project result-owned factual evidence into resolver-safe references."""

    references: list[dict[str, object]] = []
    for evidence, evidence_handle, evidence_excerpt in zip(
        result["evidence"],
        result["evidence_handles"],
        result["evidence_excerpts"],
        strict=True,
    ):
        references.append({
            "schema_version": "evidence_ref.v1",
            "evidence_kind": "tool_result",
            "evidence_id": evidence_handle,
            "owner": evidence["specialist"],
            "excerpt": evidence_excerpt,
            "observed_at": observed_at,
        })
    return references


def _task_resolution_limitations(
    result: TaskResolutionResultV1,
) -> list[str]:
    """Deduplicate bounded evidence limitations for cognition context."""

    limitations: list[str] = []
    for evidence in result["evidence"]:
        for limitation in evidence["limitations"]:
            if limitation not in limitations:
                limitations.append(limitation)
    return limitations


def _task_continuation_ref(
    request: ResolverCapabilityRequestV1,
) -> GoalContinuationRefV1:
    """Require the exact validated continuation selected by cognition."""

    continuation_ref = request["goal_continuation_ref"]
    if continuation_ref is None:
        raise ResolverValidationError(
            "task-resolution request requires goal_continuation_ref"
        )
    return continuation_ref


def _validate_task_result_request_binding(
    request: ResolverCapabilityRequestV1,
    result: TaskResolutionResultV1,
) -> None:
    """Keep a task result bound to the request that authorized it."""

    if result["semantic_objective"] != request["objective"]:
        raise ResolverValidationError(
            "task-resolution result objective conflicts with resolver request"
        )
    if result["goal_continuation_ref"] != _task_continuation_ref(request):
        raise ResolverValidationError(
            "task-resolution result continuation reference conflicts with resolver request"
        )


def _required_scene_context_from_state(
    state: GlobalPersonaState,
) -> dict[str, object]:
    """Return a validated copy of the cognition-owned scene carrier."""

    scene_context = _validate_cognition_scene_context(
        state.get("cognition_scene_context")
    )
    return_value = scene_context
    return return_value


def _validate_cognition_scene_context(
    value: object,
) -> dict[str, object]:
    """Validate and copy the one scene carrier owned by cognition."""

    if not isinstance(value, Mapping):
        raise ResolverValidationError(
            "cognition_scene_context: expected canonical object"
        )
    try:
        _validate_scene_context(value)
    except CognitionContractError as exc:
        raise ResolverValidationError(
            "cognition_scene_context: invalid canonical object: "
            f"{exc}"
        ) from exc
    scene_context = deepcopy(dict(value))
    return scene_context


def _required_state_text(state: GlobalPersonaState, field_name: str) -> str:
    """Require one trusted non-empty persona-state text field."""

    value = state.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ResolverValidationError(
            f"{field_name}: expected non-empty state text"
        )
    text = value.strip()
    return text


def _cognitive_episode_trigger_source(state: GlobalPersonaState) -> str:
    """Read the trigger source that owns self-resolution eligibility."""

    episode = state["cognitive_episode"]
    if not isinstance(episode, dict):
        raise ResolverValidationError("cognitive_episode: expected mapping")
    trigger_source = episode["trigger_source"]
    if not isinstance(trigger_source, str) or not trigger_source.strip():
        raise ResolverValidationError(
            "cognitive_episode.trigger_source: expected string"
        )
    return_value = trigger_source.strip()
    return return_value


def _observation_base(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
    *,
    status: str,
    prompt_safe_summary: str,
) -> dict[str, Any]:
    """Build common resolver observation fields."""

    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": f"resolver_obs_{uuid4().hex}",
        "capability_kind": request["capability_kind"],
        "request_objective": request["objective"],
        "request_reason": request["reason"],
        "status": status,
        "prompt_safe_summary": prompt_safe_summary,
        "evidence_refs": [],
        "created_at_utc": _created_at_utc(state),
    }
    if request["capability_kind"] == "task_resolution_request":
        observation["goal_continuation_ref"] = _task_continuation_ref(request)
        observation["task_resolution_evidence_state"] = {
            "schema_version": RESOLVER_EVIDENCE_STATE_VERSION,
            "state": "missing" if status == "succeeded" else "blocked",
            "remaining_needs": [request["objective"]],
        }
    return observation


def _string_list(value: object) -> list[str]:
    """Return stripped strings from a list-like packet projection field."""

    if not isinstance(value, list):
        items: list[str] = []
        return items
    items = [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]
    return items


def _mapping_list_items(value: object) -> list[dict[str, object]]:
    """Return mapping items as plain dictionaries from an optional list."""

    if not isinstance(value, list):
        items: list[dict[str, object]] = []
        return items
    items = [
        dict(item)
        for item in value
        if isinstance(item, Mapping)
    ]
    return items


def _empty_projected_rag_result(
    _state: GlobalPersonaState | None = None,
) -> dict[str, Any]:
    """Build the normal projected empty RAG payload."""

    rag_result = {
        "answer": "",
        "user_image": {
            "user_memory_context": empty_user_memory_context(),
        },
        "user_memory_unit_candidates": [],
        "character_image": {},
        "third_party_profiles": [],
        "memory_evidence": [],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {
            "resolver": "local_context_resolver",
            "iterations": 0,
            "node_count": 0,
            "resolved_node_count": 0,
            "blocked_node_count": 0,
            "loop_count": 0,
            "unknown_slots": [],
            "dispatched": [],
        },
    }
    return rag_result


def _fresh_query_for_objective(
    objective: str | None,
    *,
    fallback_query: str,
) -> str:
    """Return the resolver objective or the original turn query."""

    if objective is None:
        return_value = fallback_query
        return return_value
    if not isinstance(objective, str) or not objective.strip():
        raise ResolverValidationError("objective: expected non-empty string")
    return_value = objective.strip()
    return return_value


def _rag_observation_summary(rag_result: dict[str, Any]) -> str:
    """Build a compact prompt-safe summary of one RAG capability result."""

    answer = str(rag_result.get("answer", "")).strip()
    retrieval_count = _retrieval_count(rag_result)
    execution_status = _local_context_execution_status(rag_result)
    if execution_status == "failed":
        summary = (
            "Local context evidence failed with no projected rows; continue "
            "without treating it as source-backed truth."
        )
        return summary
    no_confirmed_fact_markers = (
        "没有找到已确认事实",
        "没有找到相关证据",
        "没有返回已确认结果",
        "缺少 evidence",
        "缺少 live_evidence",
        "缺少 记忆证据",
    )
    has_no_confirmed_facts = any(
        marker in answer for marker in no_confirmed_fact_markers
    )
    if retrieval_count == 0 and has_no_confirmed_facts:
        summary = (
            "Local context evidence returned no projected rows and no "
            "confirmed facts; "
            f"treat as evidence_missing, not source-backed truth; "
            f"answer={log_preview(answer)}"
        )
        return summary
    if answer:
        summary = (
            "Local context evidence succeeded with "
            f"{retrieval_count} projected rows; "
            f"answer={log_preview(answer)}"
        )
        return summary
    summary = (
        f"Local context evidence succeeded with {retrieval_count} projected rows."
    )
    return summary


def _local_context_execution_status(
    rag_result: dict[str, Any],
) -> Literal["succeeded", "failed"]:
    """Derive capability truth from projected evidence and resolver status."""

    supervisor_trace = rag_result.get("supervisor_trace")
    blocked_node_count = 0
    if isinstance(supervisor_trace, Mapping):
        raw_blocked_count = supervisor_trace.get("blocked_node_count", 0)
        if isinstance(raw_blocked_count, int):
            blocked_node_count = raw_blocked_count
    if _retrieval_count(rag_result) == 0 and blocked_node_count > 0:
        return "failed"
    return "succeeded"


def _retrieval_count(rag_result: dict[str, Any]) -> int:
    """Count projected evidence rows in a RAG payload."""

    retrieval_count = (
        len(rag_result["memory_evidence"])
        + len(rag_result["recall_evidence"])
        + len(rag_result["conversation_evidence"])
        + len(rag_result["external_evidence"])
        + len(rag_result["third_party_profiles"])
        + len(rag_result["user_memory_unit_candidates"])
    )
    return retrieval_count


def _local_context_evidence_node_count(packet: object) -> int:
    """Count non-root graph nodes for RAG stage telemetry."""

    if not isinstance(packet, Mapping):
        count = 0
        return count
    graph = packet.get("graph")
    if not isinstance(graph, Mapping):
        count = 0
        return count
    nodes = graph.get("nodes")
    if not isinstance(nodes, Mapping):
        count = 0
        return count
    count = max(0, len(nodes) - 1)
    return count


def _local_context_resolver_context_from_state(
    state: GlobalPersonaState,
) -> LocalContextResolverContextV1:
    """Build the public local-context resolver context from persona state."""

    character_profile = state["character_profile"]
    character_name = text_or_empty(character_profile["name"])
    if not character_name:
        raise ResolverValidationError("character_profile.name: expected string")
    conversation_progress = state.get("conversation_progress")
    if isinstance(conversation_progress, Mapping):
        progress_context = dict(conversation_progress)
    else:
        progress_context = {}
    context: LocalContextResolverContextV1 = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
        "character_name": character_name,
        "platform": state["platform"],
        "platform_channel_id": state["platform_channel_id"],
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "scene_context": _required_scene_context_from_state(state),
        "local_time_context": _local_context_time_context_from_state(state),
        "prompt_message_context": dict(state["prompt_message_context"]),
        "chat_history_recent": list(state["chat_history_recent"]),
        "chat_history_wide": list(state["chat_history_wide"]),
        "conversation_progress": progress_context,
        "original_user_request": state["decontextualized_input"],
        "current_timestamp_utc": state["storage_timestamp_utc"],
        "current_platform_message_id": text_or_empty(
            state.get("platform_message_id")
        ),
        "active_turn_platform_message_ids": _active_turn_platform_message_ids(
            state,
        ),
        "active_turn_conversation_row_ids": _string_list(
            state.get("active_turn_conversation_row_ids"),
        ),
        "session_media_refs": list_session_media_refs((
            state["platform"],
            state["platform_channel_id"],
            state["global_user_id"],
        )),
    }
    return context


def _active_turn_platform_message_ids(
    state: GlobalPersonaState,
) -> list[str]:
    """Return active-turn message ids, including the current platform message."""

    message_ids = _string_list(state.get("active_turn_platform_message_ids"))
    current_message_id = text_or_empty(state.get("platform_message_id"))
    if current_message_id and current_message_id not in message_ids:
        message_ids.append(current_message_id)
    return message_ids


def _local_context_time_context_from_state(
    state: GlobalPersonaState,
) -> dict[str, object]:
    """Project persona local-time fields into the RAG3 context vocabulary."""

    time_context = dict(state["local_time_context"])
    current_local_datetime = text_or_empty(
        time_context.get("current_local_datetime")
    )
    if current_local_datetime:
        if "local_date" not in time_context:
            time_context["local_date"] = current_local_datetime[:10]
        if "local_time" not in time_context:
            if len(current_local_datetime) == 16:
                local_time = f"{current_local_datetime}:00"
            else:
                local_time = current_local_datetime
            time_context["local_time"] = local_time
    current_local_weekday = text_or_empty(
        time_context.get("current_local_weekday")
    )
    if current_local_weekday and "local_weekday" not in time_context:
        time_context["local_weekday"] = current_local_weekday
    return time_context


def _shared_memory_prewarm_rows(rows: list[Any]) -> list[dict[str, Any]]:
    """Return worker rows that belong to shared persistent memory."""

    shared_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if text_or_empty(row.get("source_system")) == "user_memory_units":
            continue
        shared_rows.append(dict(row))
    return_value = shared_rows
    return return_value


def _shared_memory_prewarm_summary(rows: list[dict[str, Any]]) -> str:
    """Return the first prompt-safe summary text from shared memory rows."""

    for row in rows:
        for field in SHARED_MEMORY_SUMMARY_FIELDS:
            summary = text_or_empty(row.get(field))
            if summary:
                return_value = summary
                return return_value
    return_value = ""
    return return_value


def _safety_recovery_incidents(rag_result: dict[str, Any]) -> list[str]:
    """Return compact RAG safety recovery labels from trace metadata."""

    supervisor_trace = rag_result.get("supervisor_trace")
    if not isinstance(supervisor_trace, dict):
        incidents: list[str] = []
        return incidents
    raw_incidents = supervisor_trace.get("safety_recovery")
    if not isinstance(raw_incidents, list):
        incidents = []
        return incidents
    incidents = [
        str(incident)
        for incident in raw_incidents
        if incident
    ]
    return incidents


def _rag_correlation_id(state: GlobalPersonaState) -> str:
    """Build a non-content correlation id for persona RAG work."""

    platform = str(state.get("platform", ""))
    message_ref = str(state.get("platform_message_id", "") or "no-message-id")
    correlation_id = f"rag:{platform}:{message_ref}"
    return correlation_id


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed monotonic milliseconds since a start marker."""

    elapsed = time.perf_counter() - started_at
    elapsed_ms = max(0, int(elapsed * MILLISECONDS_PER_SECOND))
    return elapsed_ms


async def _record_rag_event(
    record_func: RecordRagEventFunc,
    *,
    component: str,
    correlation_id: str,
    agent_name: str,
    status: str,
    slot_count: int,
    retrieval_count: int,
    latency_ms: int,
    cache_hit: bool = False,
    safety_recovery_count: int = 0,
    safety_recovery_first: str = "",
) -> None:
    """Record sanitized RAG stage telemetry."""

    await record_func(
        component=component,
        correlation_id=correlation_id,
        agent_name=agent_name,
        status=status,
        slot_count=slot_count,
        retrieval_count=retrieval_count,
        cache_hit=cache_hit,
        no_evidence=retrieval_count == 0,
        latency_ms=latency_ms,
        safety_recovery_count=safety_recovery_count,
        safety_recovery_first=safety_recovery_first,
    )


def _local_context_cache_hit(packet: dict[str, Any]) -> bool:
    """Return whether any RAG3 resolver stage was served from Cache2."""

    trace_summary = packet["trace_summary"]
    cache_hits = trace_summary.get("cache_hits")
    return_value = isinstance(cache_hits, int) and cache_hits > 0
    return return_value


def _created_at_utc(state: GlobalPersonaState) -> str:
    """Return the storage timestamp for deterministic observation time."""

    created_at = state.get("storage_timestamp_utc")
    if isinstance(created_at, str) and created_at.strip():
        return_value = created_at
        return return_value
    raise ResolverValidationError("storage_timestamp_utc: expected non-empty string")
