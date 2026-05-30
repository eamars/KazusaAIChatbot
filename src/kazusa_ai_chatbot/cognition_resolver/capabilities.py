"""Deterministic capability execution for cognition resolver requests."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_OBSERVATION_VERSION,
    ResolverCapabilityRequestV1,
    ResolverObservationV1,
    ResolverValidationError,
    validate_resolver_capability_request,
    validate_resolver_observation,
)
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
from kazusa_ai_chatbot.rag.quote_aware_sequence import (
    call_quote_aware_rag_supervisor,
)
from kazusa_ai_chatbot.utils import log_preview

MILLISECONDS_PER_SECOND = 1000
PERSONA_RAG_COMPONENT = "nodes.persona_supervisor2"
SELF_GOAL_ALLOWED_TRIGGER_SOURCES = frozenset((
    "internal_thought",
    "self_cognition",
))

logger = logging.getLogger(__name__)

RagSupervisorFunc = Callable[..., Awaitable[dict[str, Any]]]
RecordRagEventFunc = Callable[..., Awaitable[None]]
BuildRagRequestFunc = Callable[..., dict[str, Any]]


async def run_rag_evidence_for_persona_state(
    state: GlobalPersonaState,
    *,
    agent_name: str,
    objective: str | None = None,
    call_rag_supervisor_func: RagSupervisorFunc | None = None,
    record_rag_stage_event_func: RecordRagEventFunc | None = None,
    build_rag_request_func: BuildRagRequestFunc | None = None,
    component: str = PERSONA_RAG_COMPONENT,
) -> dict[str, Any]:
    """Run the existing persona RAG path for one resolver objective."""

    started_at = time.perf_counter()
    correlation_id = _rag_correlation_id(state)
    if call_rag_supervisor_func is None:
        call_rag_supervisor_func = call_quote_aware_rag_supervisor
    if record_rag_stage_event_func is None:
        record_rag_stage_event_func = event_logging.record_rag_stage_event
    if build_rag_request_func is None:
        build_rag_request_func = build_text_chat_rag_request

    referents = state["referents"]
    if should_skip_rag_for_unresolved_referents(referents):
        referent_reason = unresolved_referent_reason(referents)
        rag_result = _empty_projected_rag_result(state)
        logger.info(
            f"RAG2 skipped output: reason={log_preview(referent_reason)}"
        )
        logger.debug(
            f'RAG2 skipped metadata: platform={state["platform"]} '
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

    rag_request = build_rag_request_func(
        episode=state["cognitive_episode"],
        decontexualized_input=state["decontexualized_input"],
        character_profile=state["character_profile"],
        user_profile=state["user_profile"],
        prompt_message_context=state["prompt_message_context"],
        channel_topic=state["channel_topic"],
        chat_history_recent=state["chat_history_recent"],
        chat_history_wide=state["chat_history_wide"],
        reply_context=state["reply_context"],
        indirect_speech_context=state["indirect_speech_context"],
        conversation_progress=state.get("conversation_progress"),
        conversation_episode_state=state.get("conversation_episode_state"),
        promoted_reflection_context=state.get("promoted_reflection_context"),
    )
    fresh_query = _fresh_query_for_objective(
        objective,
        fallback_query=rag_request["original_query"],
    )
    rag_context = dict(rag_request["context"])
    if objective is not None:
        rag_context["original_user_request"] = rag_request["original_query"]

    rag_supervisor_result = await call_rag_supervisor_func(
        fresh_query=fresh_query,
        reply_context=state["reply_context"],
        character_name=rag_request["character_name"],
        context=rag_context,
    )
    rag_result = project_known_facts(
        rag_supervisor_result["known_facts"],
        current_user_id=rag_request["current_user_id"],
        character_user_id=rag_request["character_user_id"],
        answer=str(rag_supervisor_result["answer"]),
        unknown_slots=rag_supervisor_result["unknown_slots"],
        loop_count=int(rag_supervisor_result["loop_count"] or 0),
    )
    trace = rag_result["supervisor_trace"]
    logger.info(
        f'RAG2 projection output: answer={log_preview(rag_result["answer"])}'
    )
    logger.debug(
        f'RAG2 projection metadata: platform={state["platform"]} '
        f'channel={state["platform_channel_id"] or "<dm>"} '
        f'user={state["global_user_id"]} '
        f'query={log_preview(fresh_query)} '
        f'dispatched={len(trace["dispatched"])} '
        f'user_image={bool(rag_result["user_image"])} '
        f'character_image={bool(rag_result["character_image"])} '
        f'third_party_profiles={len(rag_result["third_party_profiles"])} '
        f'memory_evidence={len(rag_result["memory_evidence"])} '
        f'recall_evidence={len(rag_result["recall_evidence"])} '
        f'conversation_evidence={len(rag_result["conversation_evidence"])} '
        f'external_evidence={len(rag_result["external_evidence"])} '
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
        slot_count=len(rag_supervisor_result["unknown_slots"]),
        retrieval_count=retrieval_count,
        latency_ms=_elapsed_ms(started_at),
        safety_recovery_count=len(safety_recovery_incidents),
        safety_recovery_first=safety_recovery_first,
    )
    return_value = rag_result
    return return_value


async def execute_resolver_capability_request(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
) -> ResolverObservationV1:
    """Execute one deterministic resolver capability request."""

    validated_request = validate_resolver_capability_request(request)
    capability_kind = validated_request["capability_kind"]
    if capability_kind in {"rag_evidence", "web_evidence"}:
        observation = await _execute_rag_like_capability(
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


async def _execute_rag_like_capability(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
) -> ResolverObservationV1:
    """Execute RAG or web evidence through the existing RAG supervisor path."""

    rag_result = await run_rag_evidence_for_persona_state(
        state,
        agent_name=f"resolver_{request['capability_kind']}",
        objective=request["objective"],
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


def _blocked_observation(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
    *,
    summary_prefix: str,
) -> ResolverObservationV1:
    """Build a blocked observation for user-owned input or approval."""

    summary = f"{summary_prefix}: {request['objective']}"
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


def _empty_projected_rag_result(state: GlobalPersonaState) -> dict[str, Any]:
    """Build the normal projected empty RAG payload."""

    rag_result = project_known_facts(
        [],
        current_user_id=state["global_user_id"],
        character_user_id=state["character_profile"]["global_user_id"],
        answer="",
        unknown_slots=[],
        loop_count=0,
    )
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
    if answer:
        summary = (
            f"RAG evidence succeeded with {retrieval_count} projected rows; "
            f"answer={log_preview(answer)}"
        )
        return summary
    summary = f"RAG evidence succeeded with {retrieval_count} projected rows."
    return summary


def _retrieval_count(rag_result: dict[str, Any]) -> int:
    """Count projected evidence rows in a RAG payload."""

    retrieval_count = (
        len(rag_result["memory_evidence"])
        + len(rag_result["recall_evidence"])
        + len(rag_result["conversation_evidence"])
        + len(rag_result["external_evidence"])
        + len(rag_result["third_party_profiles"])
    )
    return retrieval_count


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
