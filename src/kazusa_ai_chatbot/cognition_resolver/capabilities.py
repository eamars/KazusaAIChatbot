"""Deterministic capability execution for cognition resolver requests."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from uuid import uuid4

from openai import OpenAIError

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.complex_task_resolver import (
    COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
    COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
    COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
    ComplexTaskValidationError,
    project_complex_task_packet,
    resolve_complex_task,
    validate_complex_task_resolver_context,
    validate_complex_task_resolver_options,
    validate_complex_task_resolver_request,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_OBSERVATION_VERSION,
    ResolverCapabilityRequestV1,
    ResolverObservationV1,
    ResolverValidationError,
    validate_resolver_capability_request,
    validate_resolver_observation,
)
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
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.nodes.referent_resolution import (
    should_skip_rag_for_unresolved_referents,
    unresolved_referent_reason,
)
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)
from kazusa_ai_chatbot.utils import log_preview, text_or_empty

MILLISECONDS_PER_SECOND = 1000
PERSONA_RAG_COMPONENT = "nodes.persona_supervisor2"
SELF_GOAL_ALLOWED_TRIGGER_SOURCES = frozenset((
    "internal_thought",
    "self_cognition",
))

logger = logging.getLogger(__name__)

RecordRagEventFunc = Callable[..., Awaitable[None]]


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
            f'query={log_preview(state["decontexualized_input"])} '
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
        fallback_query=state["decontexualized_input"],
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
    await _record_rag_event(
        record_rag_stage_event_func,
        component=component,
        correlation_id=correlation_id,
        agent_name=agent_name,
        status="succeeded",
        slot_count=_local_context_evidence_node_count(packet),
        retrieval_count=retrieval_count,
        latency_ms=_elapsed_ms(started_at),
        safety_recovery_count=len(safety_recovery_incidents),
        safety_recovery_first=safety_recovery_first,
    )
    return_value = rag_result
    return return_value


async def run_first_cycle_shared_memory_prewarm(
    state: GlobalPersonaState,
) -> dict[str, Any]:
    """Return a projected RAG payload with shared persistent memory evidence.

    Args:
        state: Persona state after decontextualization and resolver input
            initialization.

    Returns:
        A normal ``rag_result`` shape containing only shared memory evidence, or
        the empty projected RAG payload when retrieval does not produce safe
        shared rows.
    """

    empty_rag_result = _empty_projected_rag_result(state)
    resolver_request: LocalContextResolverRequestV1 = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
        "objective": state["decontexualized_input"],
        "source": "prewarm",
        "reason": "First-cycle shared memory prewarm.",
        "priority": "normal",
    }
    resolver_context = _local_context_resolver_context_from_state(state)
    resolver_options: LocalContextResolverOptionsV1 = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
        **DEFAULT_OPTION_LIMITS,
    }

    try:
        packet = await resolve_local_context(
            resolver_request,
            resolver_context,
            resolver_options,
        )
    except (OpenAIError, TimeoutError) as exc:
        logger.warning(f"Shared memory prewarm resolver failed: {exc}")
        return_value = empty_rag_result
        return return_value

    projected_rag_result = project_local_context_packet(packet)
    shared_memory_evidence = _shared_memory_evidence_from_rag_result(
        projected_rag_result,
    )
    if not shared_memory_evidence:
        return_value = empty_rag_result
        return return_value

    prewarm_rag_result = dict(empty_rag_result)
    prewarm_rag_result["memory_evidence"] = shared_memory_evidence
    return_value = prewarm_rag_result
    return return_value


def merge_shared_memory_prewarm_result(
    base_rag_result: dict[str, Any],
    prewarm_rag_result: dict[str, Any],
) -> dict[str, Any]:
    """Append prompt-safe shared-memory evidence to an existing RAG payload.

    Args:
        base_rag_result: Existing cognition RAG payload.
        prewarm_rag_result: Projected prewarm payload returned by
            ``run_first_cycle_shared_memory_prewarm``.

    Returns:
        The base payload unchanged when no shared evidence is present, or a
        shallow copy with valid prewarm memory evidence appended.
    """

    prewarm_memory_evidence = _shared_memory_evidence_from_rag_result(
        prewarm_rag_result,
    )
    if not prewarm_memory_evidence:
        return_value = base_rag_result
        return return_value

    base_memory_evidence = base_rag_result.get("memory_evidence")
    if isinstance(base_memory_evidence, list):
        merged_memory_evidence = list(base_memory_evidence)
    else:
        merged_memory_evidence = []
    merged_memory_evidence.extend(prewarm_memory_evidence)

    merged = dict(base_rag_result)
    merged["memory_evidence"] = merged_memory_evidence
    return_value = merged
    return return_value


async def execute_resolver_capability_request(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
) -> ResolverObservationV1:
    """Execute one deterministic resolver capability request."""

    validated_request = validate_resolver_capability_request(request)
    capability_kind = validated_request["capability_kind"]
    if capability_kind == "local_context_recall":
        observation = await _execute_local_context_recall(
            validated_request,
            state,
        )
        return observation
    if capability_kind == "public_answer_research":
        observation = await _execute_public_answer_research(
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


async def _execute_local_context_recall(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
) -> ResolverObservationV1:
    """Execute local/private context recall through the local-context resolver."""

    rag_result = await run_rag_evidence_for_persona_state(
        state,
        agent_name=f"resolver_{request['capability_kind']}",
        objective=request["objective"],
        reason=request["reason"],
    )
    observation = _observation_base(
        request,
        state,
        status="succeeded",
        prompt_safe_summary=_rag_observation_summary(rag_result),
    )
    observation["rag_result"] = rag_result
    return_value = validate_resolver_observation(observation)
    return return_value


async def _execute_public_answer_research(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
) -> ResolverObservationV1:
    """Execute public answer research through the complex resolver IO."""

    resolver_request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": request["objective"],
        "reason": request["reason"],
        "source": "l2d",
        "priority": "normal",
    })
    resolver_context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": text_or_empty(state.get("decontexualized_input")),
        "persona_context_summary": _complex_task_persona_context_summary(state),
        "time_context": _complex_task_time_context(state),
        "available_evidence": [],
    })
    resolver_options = validate_complex_task_resolver_options({
        "schema_version": COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
        "limits": {},
    })

    try:
        packet = await resolve_complex_task(
            resolver_request,
            resolver_context,
            resolver_options,
        )
    except ComplexTaskValidationError as exc:
        observation = _complex_task_failure_observation(request, state, exc)
        return observation

    try:
        packet_projection = project_complex_task_packet(packet)
    except ComplexTaskValidationError as exc:
        observation = _complex_task_failure_observation(request, state, exc)
        return observation

    observation = _observation_base(
        request,
        state,
        status="succeeded",
        prompt_safe_summary=_complex_task_observation_summary(
            packet_projection,
        ),
    )
    observation["evidence_refs"] = _complex_task_evidence_refs(packet)
    observation["knowledge_projection"] = {
        "investigation_summary": text_or_empty(
            packet_projection["investigation_summary"],
        ),
        "knowledge_we_know_so_far": _string_list(
            packet_projection["knowledge_we_know_so_far"],
        ),
        "knowledge_still_lacking": _string_list(
            packet_projection["knowledge_still_lacking"],
        ),
        "recommended_next_iteration": _string_list(
            packet_projection["recommended_next_iteration"],
        ),
        "evidence_boundary_notes": _string_list(
            packet_projection["evidence_boundary_notes"],
        ),
    }
    return_value = validate_resolver_observation(observation)
    return return_value


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


def _complex_task_failure_observation(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
    exc: ComplexTaskValidationError,
) -> ResolverObservationV1:
    """Build a failed observation when public research packet IO is invalid."""

    observation = _observation_base(
        request,
        state,
        status="failed",
        prompt_safe_summary=f"public_answer_research failed: {exc}",
    )
    return_value = validate_resolver_observation(observation)
    return return_value


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
    return observation


def _complex_task_persona_context_summary(state: GlobalPersonaState) -> str:
    """Build a compact persona summary without raw platform identifiers."""

    segments: list[str] = []
    character_profile = state.get("character_profile")
    if isinstance(character_profile, Mapping):
        character_name = text_or_empty(character_profile.get("name"))
        if character_name:
            segments.append(f"active_character={character_name}")
    user_name = text_or_empty(state.get("user_name"))
    if user_name:
        segments.append(f"current_user_display_name={user_name}")
    channel_type = text_or_empty(state.get("channel_type"))
    if channel_type:
        segments.append(f"channel_type={channel_type}")
    summary = "; ".join(segments)
    return summary


def _complex_task_time_context(state: GlobalPersonaState) -> dict[str, object]:
    """Return prompt-safe time context for public answer research."""

    raw_time_context = state.get("local_time_context")
    if isinstance(raw_time_context, Mapping):
        time_context = dict(raw_time_context)
        return time_context
    storage_timestamp = text_or_empty(state.get("storage_timestamp_utc"))
    if storage_timestamp:
        time_context = {"storage_timestamp_utc": storage_timestamp}
        return time_context
    time_context: dict[str, object] = {}
    return time_context


def _complex_task_observation_summary(
    packet_projection: dict[str, object],
) -> str:
    """Build the prompt-safe semantic summary shown to cognition."""

    summary = text_or_empty(packet_projection["investigation_summary"])
    if summary:
        return_value = summary
        return return_value

    segments = ["public_answer_research returned semantic knowledge"]
    lacking_items = _string_list(packet_projection["knowledge_still_lacking"])
    if lacking_items:
        segments.append("lacking=" + " | ".join(lacking_items))
    recommendations = _string_list(
        packet_projection["recommended_next_iteration"]
    )
    if recommendations:
        segments.append("next=" + " | ".join(recommendations))
    trace_summary = packet_projection.get("trace_summary")
    if isinstance(trace_summary, Mapping):
        failure_reason = text_or_empty(trace_summary.get("failure_reason"))
        if failure_reason:
            segments.append(f"failure_reason={failure_reason}")
    summary = "; ".join(segments)
    return summary


def _complex_task_evidence_refs(packet: object) -> list[dict[str, object]]:
    """Collect typed evidence refs from a complex resolver packet."""

    if not isinstance(packet, Mapping):
        evidence_refs: list[dict[str, object]] = []
        return evidence_refs

    evidence_refs = _mapping_list_items(packet.get("evidence_refs"))
    graph = packet.get("graph")
    if not isinstance(graph, Mapping):
        return evidence_refs
    nodes = graph.get("nodes")
    if not isinstance(nodes, Mapping):
        return evidence_refs
    for node in nodes.values():
        if not isinstance(node, Mapping):
            continue
        node_refs = _mapping_list_items(node.get("evidence_refs"))
        evidence_refs.extend(node_refs)
    return evidence_refs


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


def _empty_projected_rag_result(_state: GlobalPersonaState) -> dict[str, Any]:
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
        "local_time_context": _local_context_time_context_from_state(state),
        "prompt_message_context": dict(state["prompt_message_context"]),
        "chat_history_recent": list(state["chat_history_recent"]),
        "chat_history_wide": list(state["chat_history_wide"]),
        "conversation_progress": progress_context,
        "original_user_request": state["decontexualized_input"],
    }
    return context


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


def _shared_memory_evidence_from_rag_result(
    rag_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return only shared memory evidence from a projected RAG payload."""

    raw_memory_evidence = rag_result.get("memory_evidence")
    if not isinstance(raw_memory_evidence, list):
        memory_evidence: list[dict[str, Any]] = []
        return memory_evidence
    memory_evidence = [
        dict(item)
        for item in raw_memory_evidence
        if (
            isinstance(item, dict)
            and text_or_empty(item.get("source_system")) != "user_memory_units"
        )
    ]
    return memory_evidence


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
        cache_hit=False,
        no_evidence=retrieval_count == 0,
        latency_ms=latency_ms,
        safety_recovery_count=safety_recovery_count,
        safety_recovery_first=safety_recovery_first,
    )


def _created_at_utc(state: GlobalPersonaState) -> str:
    """Return the storage timestamp for deterministic observation time."""

    created_at = state.get("storage_timestamp_utc")
    if isinstance(created_at, str) and created_at.strip():
        return_value = created_at
        return return_value
    raise ResolverValidationError("storage_timestamp_utc: expected non-empty string")
