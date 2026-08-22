"""Canonical upstream connector for the V3 cognition chain."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, cast

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.action_spec.models import (
    ActionAvailabilityContextV1,
    RuntimeCapabilitySnapshotV1,
)
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    FUTURE_SPEAK_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
    build_episode_affordances,
    build_initial_action_capabilities,
    build_runtime_capability_snapshot,
    project_prompt_affordances,
)
from kazusa_ai_chatbot.action_spec.results import project_trace_action_result_v2
from kazusa_ai_chatbot.background_work.result_source import (
    ToolResultCognitionSourceV1,
    validate_tool_result_cognition_source,
)
from kazusa_ai_chatbot.character_identity_growth.models import (
    TOP_LEVEL_IDENTITY_KEYS,
)
from kazusa_ai_chatbot.character_identity_growth.projection import (
    project_identity_for_cognition,
)
from kazusa_ai_chatbot.character_identity_growth.runtime import (
    load_latest_identity_for_episode,
    snapshot_state_update,
)
from kazusa_ai_chatbot.cognition_core_v3 import run_cognition
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CognitionChainServicesV3,
)
from kazusa_ai_chatbot.cognition_core_v3.diagnostics import (
    bind_protected_chain_records,
    current_chain_scope,
    reset_protected_chain_records,
)
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeValidationError,
    GoalContinuationRefV1,
    build_goal_continuation_ref,
    project_dialog_response_operation,
    project_dialog_role_explicit_content,
    validate_cognitive_episode_v1,
)
from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    _task_resolution_execution_context_from_state,
    merge_shared_memory_prewarm_result,
    project_resolver_observation_for_cognition,
    run_first_cycle_shared_memory_prewarm,
    validate_task_resolution_execution_readiness,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ALLOWED_RESOLVER_CAPABILITIES,
    RESOLVER_CAPABILITY_REQUEST_VERSION,
    RESOLVER_CAPABILITY_SEMANTICS,
    ResolverValidationError,
    validate_resolver_capability_request,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
    PAST_DIALOG_COGNITION_CONTEXT_MAX_CHARS,
    SCHEDULED_AUTHORITY_PROPOSAL_SCHEMA_VERSION,
    ActionAffordanceV2,
    CognitionContractError,
    CognitionExecutionError,
    GroupEngagementActionContextV2,
    ResolverAffordanceV2,
    SceneContextV2,
    _validate_scene_context,
    validate_scheduled_authority_proposal,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
    resolve_state_scope,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_shared.state_projection import (
    project_character_operational_state,
    project_character_sleep_phase,
    project_duration,
    project_relationship_context,
    select_character_operational_context,
)
from kazusa_ai_chatbot.config import (
    AFFECT_SETTLING_WAKE_PREP_MINUTES,
    BACKGROUND_WORK_WORKER_ENABLED,
    CALENDAR_SCHEDULER_ENABLED,
    CHARACTER_SLEEP_LOCAL_PERIOD,
    CHARACTER_TIME_ZONE,
    COGNITION_STAGE_TIMEOUT_SECONDS,
    CognitionRouteSettingV1,
    get_cognition_v3_route_settings,
)
from kazusa_ai_chatbot.conversation_progress import (
    project_conversation_progress_evidence,
    project_conversation_progress_scene,
)
from kazusa_ai_chatbot.db import (
    compare_and_replace_character_cognition_state,
    compare_and_replace_user_cognition_state,
    get_character_cognition_state,
    get_user_cognition_state,
)
from kazusa_ai_chatbot.event_logging import (
    record_continuity_boundary_event,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_actions import (
    ACTION_SPEC_CAP,
    materialize_semantic_action_requests,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_memory_lifecycle import (
    has_trusted_active_commitments,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.time_boundary import (
    local_llm_datetime_to_storage_utc_iso,
    parse_storage_utc_datetime,
)

logger = logging.getLogger(__name__)
_llm_interface = LLInterface()
PERSONALITY_JUDGMENT_MAX_CHARS = 180

def _cognition_route_config(
    setting: CognitionRouteSettingV1,
    *,
    stage_name: str,
    route_name: str,
) -> LLMCallConfig:
    """Build one selected-engine cognition model binding."""

    config = LLMCallConfig(
        stage_name=stage_name,
        route_name=route_name,
        base_url=setting.base_url,
        api_key=setting.api_key,
        model=setting.model,
        temperature=0.1,
        top_p=0.7,
        top_k=None,
        max_completion_tokens=setting.max_completion_tokens,
        presence_penalty=None,
        timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
        thinking=LLMThinkingConfig(enabled=setting.thinking_enabled),
        context_window_tokens=setting.context_window_tokens,
    )
    return config


# L3 currently imports this shared route binding for its text-surface call.
_cognition_llm_config = _cognition_route_config(
    get_cognition_v3_route_settings().chain,
    stage_name="cognition_core_v3.l3_surface",
    route_name="COGNITION_V3_CHAIN_LLM",
)


def build_cognition_core_services(
) -> CognitionChainServicesV3:
    """Build the injected model bindings for the V3 cognition chain."""

    route_settings_v3 = get_cognition_v3_route_settings()
    chain_lane = _cognition_route_config(
        route_settings_v3.chain,
        stage_name="cognition_core_v3.chain",
        route_name="COGNITION_V3_CHAIN_LLM",
    )
    services = CognitionChainServicesV3(
        llm=_llm_interface,
        chain_lane=chain_lane,
        turn_deadline_seconds=route_settings_v3.turn_deadline_seconds,
    )
    return services


def build_scene_context_from_global_state(
    state: Mapping[str, Any],
) -> SceneContextV2:
    """Build and validate the one prompt-safe scene for a cognition episode.

    The returned scene is the canonical bounded representation shared by the
    canonical cognition input, resolver capabilities, and accepted coding context.
    It contains semantic roles, current scene text, temporal continuity,
    sleep phase, and the episode-local participant bindings without transport
    or persistent identifiers.
    """

    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        raise CognitionExecutionError(
            "canonical cognitive episode is required for scene context"
        )
    try:
        validate_cognitive_episode_v1(episode)
    except CognitiveEpisodeValidationError as exc:
        raise CognitionExecutionError(str(exc)) from exc

    character_profile = state.get("character_profile")
    if not isinstance(character_profile, Mapping):
        raise CognitionExecutionError(
            "character profile is required for scene context"
        )
    timestamp = _v2_timestamp(episode["created_at"])
    semantic_text = _semantic_episode_text(state)
    conversation_progress = state.get("conversation_progress")
    if conversation_progress is None:
        conversation_continuity = ""
    elif not isinstance(conversation_progress, Mapping):
        raise CognitionExecutionError(
            "conversation progress must be a canonical prompt mapping"
        )
    else:
        conversation_continuity = project_conversation_progress_scene(
            conversation_progress,
        )
    if conversation_continuity:
        conversation_continuity = (
            "Current participant continuity:\n"
            f"{conversation_continuity}"
        )[:2200].rstrip()

    character_name = _text(character_profile.get("name"))
    user_name = _text(state.get("user_name"))
    character_label = _named_role_label("当前角色", character_name)
    current_user_label = _named_role_label("当前用户", user_name)
    character_role = character_label
    current_user_role = current_user_label
    if (
        episode["trigger_source"] == "user_message"
        and _episode_has_source_kind(episode, "dialog")
    ):
        character_role = (
            f"{character_label}；dialog_text 的直接收件人，也是直接命令的隐含主语"
        )
        current_user_role = (
            f"{current_user_label}；dialog_text 的发言者，也是该证据中第一人称代词的所属者"
        )

    scene_context: dict[str, Any] = {
        "channel_scope": _scene_channel_scope(
            episode["target_scope"].get("channel_type"),
            episode.get("trigger_source"),
        ),
        "character_role": character_role,
        "current_user_role": current_user_role,
        "character_sleep_phase": project_character_sleep_phase(
            parse_storage_utc_datetime(timestamp),
            sleep_local_period=CHARACTER_SLEEP_LOCAL_PERIOD,
            character_time_zone=CHARACTER_TIME_ZONE,
            wake_prep_minutes=AFFECT_SETTLING_WAKE_PREP_MINUTES,
        ),
        "semantic_scene": semantic_text[:500],
        "public_group_scene": _text(state.get("public_group_scene"))[:1800],
        "conversation_continuity": conversation_continuity,
        "semantic_temporal_context": _semantic_temporal_context(
            state.get("conversation_episode_state"),
            current_timestamp=timestamp,
        ),
    }
    participant_bindings = state.get("scene_participant_bindings")
    if participant_bindings is not None:
        if not isinstance(participant_bindings, list):
            raise CognitionExecutionError(
                "scene participant bindings must be a list"
            )
        if any(
            not isinstance(binding, Mapping)
            for binding in participant_bindings
        ):
            raise CognitionExecutionError(
                "scene participant bindings must contain mappings"
            )
        scene_context["participant_bindings"] = [
            dict(binding)
            for binding in participant_bindings
        ]
    try:
        _validate_scene_context(scene_context)
    except CognitionContractError as exc:
        raise CognitionExecutionError(
            f"canonical scene context is invalid: {exc}"
        ) from exc
    validated_scene_context = cast(SceneContextV2, scene_context)
    return validated_scene_context


def build_cognition_input_from_global_state(
    state: GlobalPersonaState,
    *,
    mutable_state: Mapping[str, Any] | None = None,
    character_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Map adapter-neutral graph state into one canonical cognition scope."""

    episode = state.get("cognitive_episode")
    if not isinstance(episode, dict):
        raise CognitionExecutionError("canonical cognitive episode is required")
    try:
        validate_cognitive_episode_v1(episode)
    except CognitiveEpisodeValidationError as exc:
        raise CognitionExecutionError(str(exc)) from exc
    timestamp = _v2_timestamp(episode["created_at"])
    selected_character_state = character_state
    if selected_character_state is None:
        selected_character_state = state.get("character_cognition_state")
    if not isinstance(selected_character_state, Mapping):
        selected_character_state = build_character_production_state(
            updated_at=timestamp,
        )
    selected_mutable_state = mutable_state
    if selected_mutable_state is None:
        selected_mutable_state = state.get("cognition_state")
    if not isinstance(selected_mutable_state, Mapping):
        selected_mutable_state = build_acquaintance_user_state(
            global_user_id=state["global_user_id"],
            updated_at=timestamp,
        )
    selected_mutable_state = validate_cognition_state(selected_mutable_state)
    character_operational_view = project_character_operational_state(
        selected_character_state,
        effective_at=timestamp,
    )
    character_operational_context = select_character_operational_context(
        character_operational_view,
        consumer_role="appraisal branch",
    )
    relationship_operational_context: dict[str, Any] | None = None
    if selected_mutable_state["state_scope"] == "user":
        relationship_operational_context = project_relationship_context(
            selected_mutable_state,
            effective_at=timestamp,
        )
    character_profile = state.get("character_profile")
    if not isinstance(character_profile, Mapping):
        raise CognitionExecutionError(
            "character profile is required for cognition"
        )
    personality_brief = character_profile["personality_brief"]
    if not isinstance(personality_brief, Mapping):
        raise CognitionExecutionError(
            "character personality brief must be a mapping"
        )
    personality_judgment: dict[str, str] = {}
    for field_name in ("logic", "defense", "quirks", "taboos"):
        field_value = personality_brief[field_name]
        if (
            not isinstance(field_value, str)
            or not field_value.strip()
            or len(field_value) > PERSONALITY_JUDGMENT_MAX_CHARS
        ):
            raise CognitionExecutionError(
                f"character personality {field_name} must be non-empty text "
                f"within {PERSONALITY_JUDGMENT_MAX_CHARS} characters"
            )
        personality_judgment[field_name] = field_value
    constraints = {
        "drives": selected_character_state["drives"],
        "standards": selected_character_state["standards"],
        "meaning_state": selected_character_state["meaning_state"],
        "personality_judgment": personality_judgment,
    }
    raw_identity_context = state.get("character_identity_context")
    if isinstance(raw_identity_context, Mapping):
        character_identity_context = dict(raw_identity_context)
    else:
        effective_identity = {
            key: character_profile[key]
            for key in TOP_LEVEL_IDENTITY_KEYS
        }
        character_identity_context = project_identity_for_cognition(
            {"effective_identity": effective_identity},
            include_epistemic_core=(
                selected_mutable_state["state_scope"] == "character"
            ),
        )
    episode_id = episode["episode_id"]
    semantic_text = _semantic_episode_text(state)
    scene_context = build_scene_context_from_global_state(state)
    evidence = _episode_evidence(
        episode,
        episode_id=episode_id,
        occurred_at=timestamp,
        fallback_text=semantic_text,
    )
    evidence.extend(_media_evidence(
        state.get("user_multimedia_input"),
        episode_id,
        timestamp,
    ))
    conversation_progress = state.get("conversation_progress")
    if conversation_progress is not None:
        if not isinstance(conversation_progress, dict):
            raise CognitionExecutionError(
                "conversation progress must be a canonical prompt mapping"
            )
        evidence.extend(project_conversation_progress_evidence(
            conversation_progress,
            timestamp,
        ))
    evidence.extend(_rag_evidence(state.get("rag_result"), timestamp))
    evidence.extend(_promoted_reflection_evidence(
        state.get("promoted_reflection_context"),
        timestamp,
    ))
    resolver_observations = state.get("resolver_observations")
    resolver_state = state.get("resolver_state")
    if isinstance(resolver_state, Mapping):
        resolver_observations = resolver_state.get("observations")
    evidence.extend(_resolver_observation_evidence(
        resolver_observations,
        timestamp,
    ))
    evidence.extend(_action_result_evidence(
        state.get("action_results"),
        timestamp,
    ))
    scope = selected_mutable_state["state_scope"]
    payload: dict[str, Any] = {
        "schema_version": "cognition_input.v3",
        "episode": dict(episode),
        "state_scope": scope,
        "mutable_state": dict(selected_mutable_state),
        "character_constraints": constraints,
        "character_identity_context": character_identity_context,
        "character_operational_context": character_operational_context,
        "evidence": evidence[:32],
        "direct_facts": _typed_direct_facts(state.get("direct_facts")),
        "available_actions": _available_action_affordances(state),
        "available_resolver_capabilities": _available_resolver_affordances(
            state,
            cognition_scene_context=scene_context,
        ),
        "resolver_context": _text(state.get("resolver_context"))[:8000],
        "private_continuity_context": _text(
            state.get("internal_monologue_residue_context")
        )[:1000],
        "past_dialog_cognition_context": _text(
            state.get("past_dialog_cognition_context")
        )[:PAST_DIALOG_COGNITION_CONTEXT_MAX_CHARS],
        "group_engagement_action_context": dict(
            state.get(
                "group_engagement_action_context",
                {
                    "engagement_guidelines": [],
                    "confidence": "",
                },
            )
        ),
        "scene_context": scene_context,
    }
    if relationship_operational_context is not None:
        payload["relationship_context"] = relationship_operational_context
    runtime_limits = build_runtime_capability_limits(state)
    if runtime_limits:
        payload["runtime_capability_limits"] = runtime_limits
    pending_resume = state.get("pending_resolver_resume")
    if isinstance(pending_resume, Mapping):
        payload["pending_resolver_resume"] = dict(pending_resume)
    resolver_state = state.get("resolver_state")
    if isinstance(resolver_state, Mapping):
        cycle_index = resolver_state.get("cycle_index")
        if isinstance(cycle_index, int) and not isinstance(cycle_index, bool):
            payload["resolver_cycle_index"] = cycle_index
        goal_progress = resolver_state.get("goal_progress")
        if isinstance(goal_progress, Mapping):
            payload["resolver_goal_progress"] = dict(goal_progress)
        evidence_dependency = resolver_state.get(
            "required_resolver_evidence_dependency"
        )
        if isinstance(evidence_dependency, Mapping):
            payload["required_resolver_evidence_dependency"] = dict(
                evidence_dependency
            )
    return payload


async def call_cognition_subgraph(
    state: GlobalPersonaState,
    *,
    commit: bool = True,
) -> GlobalPersonaState:
    """Run one canonical cognition pass and expose its projections."""

    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        raise CognitionExecutionError(
            "cognitive episode is required for identity resolution"
        )
    try:
        validate_cognitive_episode_v1(episode)
    except CognitiveEpisodeValidationError as exc:
        raise CognitionExecutionError(str(exc)) from exc
    resolver_state = state.get("resolver_state")
    is_cycle_zero = (
        isinstance(resolver_state, Mapping)
        and resolver_state.get("cycle_index") == 0
    )
    prewarm_task: asyncio.Task[dict[str, Any]] | None = None
    group_self_cognition = _is_group_self_cognition_state(state)
    if (
        is_cycle_zero
        and _supports_first_cycle_shared_memory_prewarm(state)
    ):
        prewarm_task = asyncio.create_task(
            run_first_cycle_shared_memory_prewarm(state)
        )
    try:
        caller = _scope_caller(episode)
        target_user_id = state.get("global_user_id")
        origin_scope = (
            episode.get("origin_scope")
            if isinstance(episode, Mapping)
            else None
        )
        scope, owner = resolve_state_scope(
            caller,
            target_user_id,
            origin_scope=(
                tuple(origin_scope)
                if isinstance(origin_scope, list)
                else origin_scope
            ),
        )
        episode_id = _text(episode.get("episode_id"))
        if not episode_id:
            raise CognitionExecutionError(
                "cognitive episode_id is required for identity resolution"
            )
        include_epistemic_core = scope == "character"
        if not _state_has_episode_identity_snapshot(
            state,
            episode_id=episode_id,
            include_epistemic_core=include_epistemic_core,
        ):
            origin_metadata = episode.get("origin_metadata")
            correlation_id = ""
            if isinstance(origin_metadata, Mapping):
                correlation_id = _text(
                    origin_metadata.get("correlation_id")
                )
            if not correlation_id:
                correlation_id = (
                    _text(state.get("llm_trace_id")) or episode_id
                )
            identity_snapshot = await load_latest_identity_for_episode(
                episode_id=episode_id,
                correlation_id=correlation_id,
                include_epistemic_core=include_epistemic_core,
            )
            resolved_state = dict(state)
            resolved_state.update(snapshot_state_update(
                identity_snapshot,
                episode_id=episode_id,
                include_epistemic_core=include_epistemic_core,
            ))
            state = resolved_state  # type: ignore[assignment]
        prior_update = state.get("cognition_state_update")
        prior_replacement: Mapping[str, Any] | None = None
        character_base_updated_at: str | None = None
        if isinstance(prior_update, Mapping):
            replacement = prior_update.get("replacement_state")
            if (
                prior_update.get("state_scope") == scope
                and prior_update.get("owner_key") == owner
                and isinstance(replacement, Mapping)
            ):
                prior_replacement = replacement
        if scope == "character":
            if prior_replacement is None:
                mutable_state = await get_character_cognition_state()
                character_base_updated_at = _character_state_updated_at(
                    mutable_state,
                )
            else:
                mutable_state = prior_replacement
                character_base_updated_at = _character_state_base_updated_at(
                    state,
                )
            character_state = mutable_state
        else:
            if prior_replacement is None:
                mutable_state = await get_user_cognition_state(owner)
            else:
                mutable_state = prior_replacement
            character_state = await get_character_cognition_state()
    except (Exception, asyncio.CancelledError):
        await _cancel_cognition_preparation_tasks(
            prewarm_task,
        )
        raise
    try:
        resolved_state = dict(state)
        if prewarm_task is not None:
            prewarm_rag_result = await prewarm_task
            base_rag_result = state.get("rag_result")
            if isinstance(base_rag_result, Mapping):
                merge_base = dict(base_rag_result)
            else:
                merge_base = {}
            merged_rag_result = merge_shared_memory_prewarm_result(
                merge_base,
                prewarm_rag_result,
            )
            if (
                merged_rag_result is not merge_base
                and "answer" not in merged_rag_result
            ):
                merged_rag_result["answer"] = ""
            resolved_state["rag_result"] = merged_rag_result
        empty_group_context: GroupEngagementActionContextV2 = {
            "engagement_guidelines": [],
            "confidence": "",
        }
        group_engagement_context: Mapping[str, Any] = empty_group_context
        if group_self_cognition:
            style_snapshot = state.get("interaction_style_context")
            if not isinstance(style_snapshot, Mapping):
                raise CognitionExecutionError(
                    "interaction style turn snapshot is required"
                )
            existing_group_context = style_snapshot.get(
                "group_engagement_action_context"
            )
            if not isinstance(existing_group_context, Mapping):
                raise CognitionExecutionError(
                    "group engagement snapshot projection is required"
                )
            group_engagement_context = existing_group_context
        resolved_state["group_engagement_action_context"] = {
            "engagement_guidelines": list(
                group_engagement_context["engagement_guidelines"]
            ),
            "confidence": group_engagement_context["confidence"],
        }
    except (Exception, asyncio.CancelledError):
        await _cancel_cognition_preparation_tasks(
            prewarm_task,
        )
        raise
    state = resolved_state  # type: ignore[assignment]
    cognition_input = build_cognition_input_from_global_state(
        state,
        mutable_state=mutable_state,
        character_state=character_state,
    )
    reflection_count = sum(
        1
        for row in cognition_input["evidence"]
        if row["evidence_ref"]["source_kind"] == "promoted_reflection"
    )
    await record_continuity_boundary_event(
        component="persona_supervisor2_cognition",
        boundary="reflection_projection",
        status="succeeded" if reflection_count else "empty",
        scope_kind=_continuity_scope_kind(state),
        selected_count=reflection_count,
        source_age="unknown",
        trace_ref=_text(state.get("llm_trace_id")),
    )
    trace_token = llm_tracing.bind_trace_id(
        str(state.get("llm_trace_id") or ""),
    )
    diagnostics_token = None
    try:
        cognition_services = build_cognition_core_services()
        if current_chain_scope() is None:
            invocation_id = ""
            input_episode = cognition_input.get("episode")
            target_scope = (
                input_episode.get("target_scope")
                if isinstance(input_episode, Mapping)
                else None
            )
            platform = (
                str(target_scope.get("platform", "")).strip().lower()
                if isinstance(target_scope, Mapping)
                else ""
            )
            source_kind = "debug" if platform == "debug" else "live"
            run_id = invocation_id or episode_id
            trace_id = str(state.get("llm_trace_id") or run_id)
            diagnostics_token = bind_protected_chain_records(
                run_id=run_id,
                source_kind=source_kind,
                llm_trace_id=trace_id,
                cognition_invocation_id=(invocation_id or run_id),
            )
        # The canonical engine owns one direct A1/A2/G/P pass.
        output = await run_cognition(
            cognition_input,
            cognition_services,
        )
    finally:
        if diagnostics_token is not None:
            reset_protected_chain_records(diagnostics_token)
        llm_tracing.reset_trace_id(trace_token)
    if commit:
        await _commit_cognition_state(
            output,
            expected_character_updated_at=character_base_updated_at,
        )
    resolved_state["cognition_scene_context"] = deepcopy(
        cognition_input["scene_context"]
    )
    state = resolved_state  # type: ignore[assignment]
    update = _project_output_to_global_state(
        output,
        state,
        available_actions=cognition_input["available_actions"],
        available_resolver_capabilities=(
            cognition_input["available_resolver_capabilities"]
        ),
    )
    update["cognition_scene_context"] = deepcopy(
        cognition_input["scene_context"]
    )
    if character_base_updated_at is not None:
        update["character_cognition_base_updated_at"] = (
            character_base_updated_at
        )
    update["cognition_input"] = cognition_input
    update["cognition_core_output"] = output
    update["cognition_scope"] = output["state_projection"]["state_scope"]
    if "group_engagement_action_context" in cognition_input:
        update["group_engagement_action_context"] = dict(
            cognition_input["group_engagement_action_context"]
        )
    update.update(_episode_identity_state_update(state))
    return update  # type: ignore[return-value]


def _state_has_episode_identity_snapshot(
    state: Mapping[str, object],
    *,
    episode_id: str,
    include_epistemic_core: bool,
) -> bool:
    """Return whether one graph state already holds this episode's snapshot."""

    revision_number = state.get("character_identity_revision_number")
    return (
        state.get("character_identity_episode_id") == episode_id
        and state.get("character_identity_epistemic_core_included")
        is include_epistemic_core
        and isinstance(revision_number, int)
        and not isinstance(revision_number, bool)
        and revision_number >= 0
        and isinstance(state.get("character_profile"), Mapping)
        and isinstance(state.get("character_identity_context"), Mapping)
        and isinstance(
            state.get("character_identity_surface_context"),
            Mapping,
        )
        and isinstance(
            state.get("character_identity_projection_digest"),
            str,
        )
        and isinstance(
            state.get("character_identity_consumer_kinds"),
            list,
        )
    )


def _episode_identity_state_update(
    state: Mapping[str, object],
) -> dict[str, object]:
    """Carry one latest identity snapshot through resolver recurrence."""

    field_names = (
        "character_profile",
        "character_identity_revision_number",
        "character_identity_context",
        "character_identity_surface_context",
        "character_identity_projection_digest",
        "character_identity_consumer_kinds",
        "character_identity_episode_id",
        "character_identity_epistemic_core_included",
    )
    missing = [name for name in field_names if name not in state]
    if missing:
        raise CognitionExecutionError(
            f"identity snapshot is missing runtime fields: {missing}"
        )
    return {
        name: state[name]
        for name in field_names
    }


def _character_state_updated_at(state: Mapping[str, Any]) -> str:
    """Return the validated optimistic version for a character-state read."""

    validated_state = validate_cognition_state(state)
    if validated_state["state_scope"] != "character":
        raise CognitionExecutionError("character state version has wrong scope")
    return validated_state["updated_at"]


def _character_state_base_updated_at(state: Mapping[str, Any]) -> str:
    """Return the original persisted character version across resolver cycles."""

    value = state.get("character_cognition_base_updated_at")
    if not isinstance(value, str) or not value.strip():
        raise CognitionExecutionError(
            "character state recurrence is missing its base version"
        )
    return value.strip()


async def commit_cognition_output(
    output: Mapping[str, Any],
    *,
    expected_character_updated_at: str | None = None,
) -> None:
    """Commit one already-validated canonical result at the episode boundary."""

    await _commit_cognition_state(
        output,
        expected_character_updated_at=expected_character_updated_at,
    )


async def _commit_cognition_state(
    output: Mapping[str, Any],
    *,
    expected_character_updated_at: str | None = None,
) -> None:
    """Commit the validated replacement before any downstream surface/action work."""

    if output.get("schema_version") == "cognition_output.v3":
        projection = output.get("state_projection")
        if not isinstance(projection, Mapping):
            raise CognitionExecutionError(
                "canonical cognition state projection is missing"
            )
        replacement = projection.get("replacement_state")
        expected = projection.get("expected_previous_state")
        scope = projection.get("state_scope")
        owner_key = projection.get("owner_key")
        if not isinstance(replacement, Mapping) or not isinstance(expected, Mapping):
            raise CognitionExecutionError(
                "canonical cognition state projection is invalid"
            )
        if scope == "user":
            committed = await compare_and_replace_user_cognition_state(
                str(owner_key), dict(expected), dict(replacement)
            )
        elif scope == "character":
            if not expected_character_updated_at:
                raise CognitionExecutionError(
                    "canonical character commit requires its base version"
                )
            committed = await compare_and_replace_character_cognition_state(
                expected_updated_at=expected_character_updated_at,
                replacement=dict(replacement),
            )
        else:
            raise CognitionExecutionError("canonical state scope is invalid")
        if not committed:
            raise CognitionExecutionError(
                "canonical cognition state commit encountered a version conflict"
            )
        return

def _project_output_to_global_state(
    output: Mapping[str, Any],
    state: GlobalPersonaState,
    *,
    available_actions: object,
    available_resolver_capabilities: object,
) -> dict[str, Any]:
    """Expose semantic outputs while preserving deterministic action ownership."""

    if output.get("schema_version") != "cognition_output.v3":
        raise CognitionExecutionError("canonical cognition output is required")
    goal = output["active_character_goal"]
    plan = output["response_plan"]
    if not isinstance(goal, Mapping) or not isinstance(plan, Mapping):
        raise CognitionExecutionError(
            "canonical cognition goal or response plan is invalid"
        )
    replacement_state = output.get("state_projection", {}).get(
        "replacement_state",
    )
    if not isinstance(replacement_state, Mapping):
        raise CognitionExecutionError("canonical replacement state is required")
    action_specs = _materialize_canonical_action_requests(
        output,
        state,
        replacement_state,
        available_actions=available_actions,
    )
    resolver_requests = _materialize_canonical_resolver_requests(
        output,
        state,
        replacement_state,
        available_resolver_capabilities=available_resolver_capabilities,
    )
    update = {
        "cognition_state_update": output.get("state_projection"),
        "cognition_intention": {
            "route": "speech" if plan.get("response_goal") else "silence",
            "intention": plan.get("response_goal", ""),
            "reason": goal.get("reason", ""),
        },
        "semantic_affect_projection": output.get("affect_projection", []),
        "semantic_relationship_projection": output.get(
            "relationship_projection"
        ),
        "active_character_goal": dict(goal),
        "goal_resolution": plan.get("goal_resolution", "answerable_now"),
        "resolver_capability_requests": resolver_requests,
        "action_specs": action_specs,
        "internal_monologue": goal.get("reason", ""),
        "character_intent": goal.get("intent", ""),
        "logical_stance": output.get("relational_willingness", {}).get(
            "stance", ""
        ),
        "judgment_note": goal.get("cause_summary", ""),
        "emotional_appraisal": (
            output.get("affect_projection", [{}])[0].get("emotion", "平静")
            if output.get("affect_projection")
            else "平静"
        ),
        "should_respond": bool(plan.get("response_goal")),
        "cognition_cause_provenance": output.get("cause_provenance", []),
    }
    self_response = plan.get("self_cognition_response")
    if isinstance(self_response, Mapping):
        update["self_cognition_response"] = dict(self_response)
    return update


def _canonical_goal_continuation_ref(
    output: Mapping[str, Any],
    state: Mapping[str, Any],
    replacement_state: Mapping[str, Any],
) -> GoalContinuationRefV1:
    """Bind one continuation to the caller-owned ordinary-response goal."""

    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        raise CognitionExecutionError("cognitive episode is required for continuation")
    episode_id = str(episode.get("episode_id") or "").strip()
    if not episode_id:
        raise CognitionExecutionError("episode identity is required for continuation")
    origin_metadata = episode.get("origin_metadata")
    source_message_id = ""
    if isinstance(origin_metadata, Mapping):
        source_message_id = str(
            origin_metadata.get("platform_message_id") or ""
        ).strip()
    if not source_message_id:
        source_message_id = str(
            episode.get("source_message_id")
            or episode.get("message_id")
            or ""
        ).strip()
    goal_row = next(
        (
            row for row in replacement_state.get("goals", [])
            if isinstance(row, Mapping)
            and row.get("goal_kind") == "ordinary_response"
        ),
        None,
    )
    if not isinstance(goal_row, Mapping):
        raise CognitionExecutionError("ordinary-response goal is required for continuation")
    scope = str(replacement_state.get("state_scope") or "")
    if scope not in {"user", "character"}:
        raise CognitionExecutionError("continuation goal scope is invalid")
    return build_goal_continuation_ref(
        source_episode_id=episode_id,
        source_message_id=source_message_id,
        branch_id="ordinary_response",
        goal_ref={
            "scope": scope,
            "kind": "goal",
            "entity_id": str(goal_row["entity_id"]),
        },
    )


def _canonical_speak_surface_metadata(
    state: Mapping[str, Any],
) -> tuple[str, GoalContinuationRefV1 | None]:
    """Preserve typed tool-result lineage for the visible speak action."""

    episode = state.get("cognitive_episode")
    if (
        isinstance(episode, Mapping)
        and episode.get("trigger_source") == "tool_result"
    ):
        source = _tool_result_episode_cognition_source(episode)
        surface_role = (
            "task_result"
            if source["task_status"] in {"resolved", "partial"}
            else "task_status"
        )
        return surface_role, source["goal_continuation_ref"]
    return "ordinary", None


def _materialize_canonical_action_requests(
    output: Mapping[str, Any],
    state: Mapping[str, Any],
    replacement_state: Mapping[str, Any],
    *,
    available_actions: object,
) -> list[dict[str, Any]]:
    """Validate caller-bound action affordances and materialize action specs."""

    plan = output["response_plan"]
    if "self_cognition_response" in plan:
        return []
    rows = plan.get("action_requests", [])
    if not isinstance(rows, list):
        raise CognitionExecutionError("canonical action requests are invalid")
    if not isinstance(available_actions, list):
        raise CognitionExecutionError("caller action affordances are invalid")
    action_affordances = available_actions
    action_kinds = [
        row.get("action_kind")
        for row in action_affordances
        if isinstance(row, Mapping)
    ]
    if (
        len(action_kinds) != len(action_affordances)
        or any(not isinstance(kind, str) or not kind for kind in action_kinds)
        or len(set(action_kinds)) != len(action_kinds)
    ):
        raise CognitionExecutionError("caller action affordances are not unique")
    affordances = {
        str(row["action_kind"]): row
        for row in action_affordances
        if isinstance(row, Mapping)
    }
    requests: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise CognitionExecutionError("canonical action request is invalid")
        action_kind = str(row["action_kind"])
        affordance = affordances.get(action_kind)
        if affordance is None:
            raise CognitionExecutionError("canonical action capability is unavailable")
        request: dict[str, Any] = {
            "capability": action_kind,
            "decision": str(row["decision"]),
            "detail": str(row["detail"]),
            "reason": str(row["reason"]),
            "context_ref": str(affordance["context_ref"]),
            "target_roles": deepcopy(list(affordance["target_roles"])),
            "evidence_handles": [],
            "surface_role": "ordinary",
            "goal_continuation_ref": (
                _canonical_goal_continuation_ref(output, state, replacement_state)
                if action_kind == ACCEPTED_CODING_TASK_REQUEST_CAPABILITY
                else None
            ),
        }
        if action_kind == ACCEPTED_CODING_TASK_REQUEST_CAPABILITY:
            request["surface_role"] = "task_acknowledgement"
            request["task_execution_context"] = (
                _task_resolution_execution_context_from_state(
                    state,
                    goal_continuation_ref=request["goal_continuation_ref"],
                )
            )
        if action_kind == FUTURE_SPEAK_CAPABILITY:
            request["scheduled_authority_proposal"] = (
                _build_scheduled_authority_proposal(state, request)
            )
        requests.append(request)
    if str(plan.get("response_goal") or "").strip():
        surface_role, continuation_ref = _canonical_speak_surface_metadata(state)
        requests.append({
            "capability": SPEAK_CAPABILITY,
            "decision": "visible_reply",
            "detail": str(plan["response_goal"]),
            "reason": str(output["active_character_goal"]["reason"]),
            "target_roles": [_action_target_role(state)],
            "evidence_handles": [],
            "surface_role": surface_role,
            "goal_continuation_ref": continuation_ref,
        })
    if len(requests) > ACTION_SPEC_CAP:
        raise CognitionExecutionError("materialized action capacity is exceeded")
    materialized = materialize_semantic_action_requests(
        requests,
        state,
    )
    expected_count = len(requests)
    if len(materialized) != expected_count:
        raise CognitionExecutionError(
            "caller-owned action materialization dropped a requested capability"
        )
    return [dict(row) for row in materialized]


def _materialize_canonical_resolver_requests(
    output: Mapping[str, Any],
    state: Mapping[str, Any],
    replacement_state: Mapping[str, Any],
    *,
    available_resolver_capabilities: object,
) -> list[dict[str, Any]]:
    """Validate semantic resolver rows after caller-owned lineage binding."""

    plan = output["response_plan"]
    if "self_cognition_response" in plan:
        return []
    rows = plan.get("resolver_requests", [])
    if not isinstance(rows, list):
        raise CognitionExecutionError("canonical resolver requests are invalid")
    if not rows:
        return []
    if not isinstance(available_resolver_capabilities, list):
        raise CognitionExecutionError("caller resolver affordances are invalid")
    resolver_affordances = available_resolver_capabilities
    resolver_capabilities = [
        row.get("capability")
        for row in resolver_affordances
        if isinstance(row, Mapping)
    ]
    if (
        len(resolver_capabilities) != len(resolver_affordances)
        or any(
            not isinstance(capability, str) or not capability
            for capability in resolver_capabilities
        )
        or len(set(resolver_capabilities)) != len(resolver_capabilities)
    ):
        raise CognitionExecutionError(
            "caller resolver affordances are not unique"
        )
    allowed = {
        str(row["capability"])
        for row in resolver_affordances
        if isinstance(row, Mapping)
    }
    continuation = None
    if any(
        isinstance(row, Mapping)
        and row.get("capability") == "task_resolution_request"
        for row in rows
    ):
        continuation = _canonical_goal_continuation_ref(
            output,
            state,
            replacement_state,
        )
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise CognitionExecutionError("canonical resolver request is invalid")
        capability = str(row["capability"])
        if capability not in allowed:
            raise CognitionExecutionError("canonical resolver capability is unavailable")
        validated = validate_resolver_capability_request({
            "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
            "capability_kind": capability,
            "objective": str(row["goal"]),
            "reason": str(row["reason"]),
            "priority": "now",
            "goal_continuation_ref": continuation
            if capability == "task_resolution_request"
            else None,
        })
        result.append(dict(validated))
    return result


def _build_scheduled_authority_proposal(
    state: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, object]:
    """Bind one future-speak proposal to the trusted current episode."""

    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        raise CognitionExecutionError("future-speak episode is required")
    if episode.get("trigger_source") != "user_message":
        raise CognitionExecutionError(
            "future-speak requires a current user-message episode"
        )
    trigger_local = str(request["decision"])
    try:
        trigger_utc = local_llm_datetime_to_storage_utc_iso(trigger_local)
        accepted_at = parse_storage_utc_datetime(
            str(state.get("storage_timestamp_utc") or episode.get("created_at"))
        )
        if parse_storage_utc_datetime(trigger_utc) <= accepted_at:
            raise CognitionExecutionError(
                "future-speak trigger must be later than accepted time"
            )
    except ValueError as exc:
        raise CognitionExecutionError(
            f"future-speak trigger is invalid: {exc}"
        ) from exc
    semantic_summary = _trusted_user_message_summary(episode)
    if not semantic_summary:
        raise CognitionExecutionError(
            "future-speak requires trusted current user-message content"
        )
    proposal = {
        "schema_version": SCHEDULED_AUTHORITY_PROPOSAL_SCHEMA_VERSION,
        "temporal_alignment": "aligned",
        "authorized_content_summary": semantic_summary,
        "authorized_detail_refs": [{
            "evidence_handle": "current_episode",
            "semantic_summary": semantic_summary,
            "provenance_role": "current_event",
        }],
    }
    return dict(validate_scheduled_authority_proposal(proposal))


def _trusted_user_message_summary(episode: Mapping[str, Any]) -> str:
    """Read the current user-message percept as the authority summary."""

    if episode.get("trigger_source") != "user_message":
        return ""
    percepts = episode.get("percepts")
    if not isinstance(percepts, list):
        return ""
    for percept in percepts:
        if not isinstance(percept, Mapping):
            continue
        if percept.get("source_kind") not in {"dialog", "user_message"}:
            continue
        content = percept.get("content")
        if isinstance(content, Mapping):
            content = (
                content.get("semantic_summary")
                or content.get("semantic_text")
                or content.get("text")
            )
        summary = str(content or "").strip()
        if summary:
            return summary[:1000]
    return ""

def _available_action_affordances(
    state: GlobalPersonaState,
) -> list[ActionAffordanceV2]:
    """Project the deterministic capability registry into typed affordances."""

    current_user = _action_target_role(state)
    capabilities = build_initial_action_capabilities()
    availability_rows = {
        row["capability_kind"]: row
        for row in build_episode_affordances(
            capabilities,
            _action_availability_context(state),
            _build_action_availability_snapshot(state),
        )
    }
    prompt_affordances = {
        row["capability"]: row
        for row in project_prompt_affordances(capabilities)
    }
    available_contexts = _available_action_contexts(state)
    affordances: list[ActionAffordanceV2] = []
    for capability_kind in sorted(capabilities):
        if capability_kind in {
            SPEAK_CAPABILITY,
            APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        }:
            continue
        episode = state.get("cognitive_episode")
        if (
            capability_kind == TRIGGER_FUTURE_COGNITION_CAPABILITY
            and isinstance(episode, Mapping)
            and episode.get("trigger_source") == "scheduled_tick"
        ):
            continue
        if capability_kind not in availability_rows:
            continue
        prompt_affordance = prompt_affordances[capability_kind]
        availability_context = prompt_affordance["availability_context"]
        if not isinstance(availability_context, str):
            raise CognitionExecutionError(
                "action affordance availability context is invalid"
            )
        if (
            availability_context
            and availability_context not in available_contexts
        ):
            continue
        semantic_summary = prompt_affordance["semantic_input_summary"]
        if not isinstance(semantic_summary, list):
            raise CognitionExecutionError(
                "action affordance semantic summary is invalid"
            )
        affordance: ActionAffordanceV2 = {
            "action_kind": capability_kind,
            "capability": " ".join(str(row) for row in semantic_summary),
            "permission": "allowed",
            "decision_mode": prompt_affordance["decision_mode"],
            "allowed_decisions": list(
                prompt_affordance["allowed_decisions"]
            ),
            "default_decision": str(
                prompt_affordance["default_decision"]
            ),
            "decision_pattern": str(
                prompt_affordance["decision_pattern"]
            ),
            "context_ref": str(prompt_affordance["context_ref"]),
            "target_roles": [current_user],
        }
        if capability_kind == ACCEPTED_CODING_TASK_REQUEST_CAPABILITY:
            contextual_affordances = _coding_run_action_affordances(
                state,
                base_affordance=affordance,
            )
            affordances.extend(contextual_affordances)
            continue
        affordances.append(affordance)
    return affordances


def _action_target_role(state: Mapping[str, Any]) -> dict[str, str]:
    """Return a prompt-safe target role for user or targetless group scope."""

    global_user_id = str(state.get("global_user_id", "") or "").strip()
    if global_user_id:
        return {
            "role": "target",
            "entity_kind": "user",
            "entity_id": global_user_id,
        }
    episode = state.get("cognitive_episode")
    target_scope = episode.get("target_scope") if isinstance(
        episode,
        Mapping,
    ) else None
    if isinstance(target_scope, Mapping) and target_scope.get(
        "channel_type"
    ) == "group":
        return {
            "role": "target",
            "entity_kind": "group",
            "entity_id": "current group scene",
        }
    return {
        "role": "target",
        "entity_kind": "user",
        "entity_id": global_user_id,
    }


def build_action_availability_snapshot(
    state: Mapping[str, Any],
) -> RuntimeCapabilitySnapshotV1:
    """Build the current deterministic capability snapshot for one state."""

    return _build_action_availability_snapshot(state)


def build_runtime_capability_limits(
    state: Mapping[str, Any],
) -> list[str]:
    """Project trusted unavailable-owner facts into Chinese cognition context."""

    snapshot = build_action_availability_snapshot(state)
    unavailable = {"down", "unavailable", "disabled", "blocked"}
    limits: list[str] = []
    if snapshot["scheduler_status"] in unavailable:
        limits.append(
            "当前调度能力不可用，不能把未来提醒或主动联系说成已经安排、发送或完成。"
        )
        limits.append(
            "未来提醒和主动联系只属于 future_speak；该能力不可用时不能用其他能力代替。"
        )
    worker_status = snapshot["worker_status"]
    background_worker_unavailable = (
        worker_status.get("background_work") in unavailable
    )
    queue_only_coding = (
        background_worker_unavailable
        and snapshot["repository_access"].get(
            "background_work",
            "read_write",
        ) == "read_write"
        and snapshot["coding_workspace_status"] not in unavailable
    )
    if queue_only_coding:
        limits.append(
            "当前 coding worker 尚未运行；绑定既有 coding_run_ref 的生命周期动作"
            "可以记录并排队，结果保持待执行，不能表述为 worker 已执行或已完成。"
        )
    elif worker_status.get("background_work") == "degraded":
        limits.append(
            "当前通用任务解析只有 inline 能力；task_resolution_request 必须先在本轮"
            "预算内尝试，不能写成后台已经安排。"
        )
    elif any(
        worker_status.get(owner) in unavailable
        for owner in ("accepted_task", "background_work")
    ):
        limits.append(
            "当前后台任务能力不可用，不能把延迟任务说成已经创建、安排或完成。"
        )
    if any(
        worker_status.get(owner) in unavailable
        for owner in ("accepted_task", "background_work")
    ):
        limits.append(
            "当前仓库代码读取 owner 不可用；尚未完成的仓库分析或修改目标必须保持 "
            "goal_resolution=blocked、action_requests=[]、resolver_requests=[]；"
            "说明当前限制不等于 requires_user_input，也不能把请用户提供材料当作原目标已可继续。"
        )
    if any(
        status in unavailable
        for status in snapshot["adapter_target_status"].values()
    ):
        limits.append(
            "当前消息投递目标不可用，不能把消息说成已经发送。"
        )
    return limits[:8]


def _build_action_availability_snapshot(
    state: Mapping[str, Any],
) -> RuntimeCapabilitySnapshotV1:
    """Collect configured owner status without performing runtime effects."""

    runtime = state.get("action_availability_runtime")
    runtime_mapping = runtime if isinstance(runtime, Mapping) else {}
    worker_status = {
        "memory_lifecycle": "healthy",
        "memory_lifecycle_specialist": "healthy",
        "l3_text": "healthy",
        "accepted_task": "healthy",
        "background_work": (
            "healthy" if BACKGROUND_WORK_WORKER_ENABLED else "degraded"
        ),
        "orchestrator": (
            "healthy" if CALENDAR_SCHEDULER_ENABLED else "unavailable"
        ),
    }
    raw_worker_status = runtime_mapping.get("worker_status")
    if isinstance(raw_worker_status, Mapping):
        worker_status.update({
            str(key): str(value)
            for key, value in raw_worker_status.items()
        })
    scheduler_status = (
        "healthy" if CALENDAR_SCHEDULER_ENABLED else "unavailable"
    )
    raw_scheduler_status = runtime_mapping.get("scheduler_status")
    if isinstance(raw_scheduler_status, str):
        scheduler_status = raw_scheduler_status
    raw_target_status = runtime_mapping.get("adapter_target_status")
    adapter_target_status = (
        dict(raw_target_status)
        if isinstance(raw_target_status, Mapping)
        else {}
    )
    return build_runtime_capability_snapshot(
        route_health=_string_mapping(runtime_mapping.get("route_health")),
        repository_access=_string_mapping(
            runtime_mapping.get("repository_access"),
        ),
        worker_status=worker_status,
        scheduler_status=scheduler_status,
        adapter_target_status=adapter_target_status,
        coding_workspace_status=str(
            runtime_mapping.get("coding_workspace_status", "healthy")
        ),
        permissions=_bool_mapping(runtime_mapping.get("permissions")),
    )


def _action_availability_context(
    state: Mapping[str, Any],
) -> ActionAvailabilityContextV1:
    """Project trusted episode facts into registry availability probes."""

    context: ActionAvailabilityContextV1 = {}
    episode = state.get("cognitive_episode")
    if isinstance(episode, Mapping):
        source_kind = episode.get("trigger_source")
        if isinstance(source_kind, str):
            context["source_kind"] = source_kind
        target_scope = episode.get("target_scope")
        if isinstance(target_scope, Mapping):
            context["target_scope"] = dict(target_scope)
    return context


def _string_mapping(value: object) -> dict[str, str]:
    """Return bounded string mapping data from trusted runtime state."""

    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
    }


def _bool_mapping(value: object) -> dict[str, bool]:
    """Return boolean permission overrides from trusted runtime state."""

    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, bool)
    }


def _available_action_contexts(state: GlobalPersonaState) -> set[str]:
    """Return trusted runtime facts used by registry availability metadata."""

    contexts: set[str] = set()
    if has_trusted_active_commitments(state):
        contexts.add("active_commitment")
    episode = state.get("cognitive_episode")
    if isinstance(episode, Mapping) and episode.get("trigger_source") in {
        "internal_thought",
        "scheduled_tick",
    }:
        contexts.add("private_cognition_source")
    return contexts


def _coding_run_action_affordances(
    state: Mapping[str, Any],
    *,
    base_affordance: ActionAffordanceV2,
) -> list[ActionAffordanceV2]:
    """Project one generic action handle per trusted open coding run."""

    action_context = state.get("action_selection_context")
    if not isinstance(action_context, Mapping):
        return []
    raw_contexts = action_context.get("coding_runs")
    if not isinstance(raw_contexts, list):
        return []
    runtime_snapshot = _build_action_availability_snapshot(state)
    unavailable_statuses = {"down", "unavailable", "disabled", "blocked"}
    coding_worker_unavailable = (
        runtime_snapshot["worker_status"].get("background_work")
        in unavailable_statuses
    )
    registry_decisions = {
        "revise_proposal",
        "summarize",
        "status",
        "approve_and_verify",
        "respond_to_blocker",
        "cancel",
    }
    eligible_contexts: list[Mapping[str, Any]] = []
    for context in raw_contexts:
        if not isinstance(context, Mapping):
            continue
        context_ref = _text(context.get("coding_run_ref"))[:200]
        raw_decisions = context.get("allowed_next_actions")
        if not context_ref or not isinstance(raw_decisions, list):
            continue
        allowed_decisions = [
            decision
            for decision in raw_decisions
            if isinstance(decision, str) and decision in registry_decisions
            and not (
                decision == "status"
                and coding_worker_unavailable
            )
        ]
        allowed_decisions = list(dict.fromkeys(allowed_decisions))
        if not allowed_decisions:
            continue
        eligible_contexts.append(context)
    if len(eligible_contexts) != 1:
        return []
    affordances: list[ActionAffordanceV2] = []
    for context in eligible_contexts:
        context_ref = _text(context.get("coding_run_ref"))[:200]
        raw_decisions = context.get("allowed_next_actions")
        if not context_ref or not isinstance(raw_decisions, list):
            continue
        allowed_decisions = list(dict.fromkeys(
            decision
            for decision in raw_decisions
            if isinstance(decision, str) and decision in registry_decisions
            and not (
                decision == "status"
                and coding_worker_unavailable
            )
        ))
        if not allowed_decisions:
            continue
        status = _text(context.get("status"))[:80]
        objective = _text(context.get("objective_summary"))[:120]
        blocker_summary = _coding_run_blocker_summary(
            context.get("active_blocker")
        )
        available_decisions = "、".join(allowed_decisions)
        semantic_context = (
            " 这是绑定既有 coding run 的生命周期 affordance；当前作用域的既有 coding run "
            "只提供以下实际可用决定："
            f"{available_decisions}。"
            "每次只从这些决定中选择。"
            f" 当前状态：{status}。目标：{objective}。"
            f" 当前阻塞：{blocker_summary}。"
        )
        default_decision = (
            "status" if "status" in allowed_decisions else allowed_decisions[0]
        )
        affordances.append({
            "action_kind": base_affordance["action_kind"],
            "capability": (
                "代码工作由持久化 coding run 管理。" + semantic_context
            )[:500],
            "permission": base_affordance["permission"],
            "decision_mode": "closed",
            "allowed_decisions": allowed_decisions,
            "default_decision": default_decision,
            "decision_pattern": base_affordance["decision_pattern"],
            "context_ref": context_ref,
            "target_roles": list(base_affordance["target_roles"]),
        })
    return affordances


def _coding_run_blocker_summary(value: object) -> str:
    """Return bounded prompt-safe blocker details for one coding run."""

    if not isinstance(value, Mapping):
        return "none"
    blocker_kind = _text(value.get("blocker_kind"))[:80]
    question = _text(value.get("question"))[:60]
    raw_options = value.get("options")
    options = []
    if isinstance(raw_options, list):
        options = [
            option[:40]
            for option in raw_options
            if isinstance(option, str) and option.strip()
        ][:3]
    return f"kind={blocker_kind}; question={question}; options={options}"[:100]


def _available_resolver_affordances(
    state: Mapping[str, Any],
    *,
    cognition_scene_context: SceneContextV2,
) -> list[ResolverAffordanceV2]:
    """Project resolver capabilities as availability, not execution authority.

    Generic task resolution remains exposed during worker degradation because
    the action planner can select its inline mode. An unavailable worker owns
    neither inline nor background task resolution for the current turn, so
    the connector must remove that resolver instead of advertising a
    contradictory available handle beside the runtime limit.
    """

    snapshot = build_action_availability_snapshot(state)
    unavailable_worker_states = {"down", "unavailable", "disabled", "blocked"}
    available_capabilities = set(ALLOWED_RESOLVER_CAPABILITIES)
    task_resolution_owner_states = (
        snapshot["worker_status"].get("background_work"),
        snapshot["route_health"].get("background_work"),
        snapshot["repository_access"].get("background_work"),
    )
    if any(
        status in unavailable_worker_states
        for status in task_resolution_owner_states
    ):
        available_capabilities.discard("task_resolution_request")
    if "task_resolution_request" in available_capabilities:
        try:
            validate_task_resolution_execution_readiness(
                state,
                cognition_scene_context=cognition_scene_context,
            )
        except ResolverValidationError:
            available_capabilities.discard("task_resolution_request")

    affordances = [
        {
            "capability": capability,
            "semantic_capability": RESOLVER_CAPABILITY_SEMANTICS[capability],
            "availability": "available",
        }
        for capability in sorted(available_capabilities)
    ]
    return affordances


def _typed_direct_facts(value: object) -> list[dict[str, Any]]:
    """Accept only caller-supplied typed facts at the connector boundary."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _episode_evidence(
    episode: Mapping[str, Any],
    *,
    episode_id: str,
    occurred_at: str,
    fallback_text: str,
) -> list[dict[str, Any]]:
    """Project the current source through its canonical evidence kind."""

    trigger_source = str(episode.get("trigger_source", "user_message"))
    source_kind = {
        "tool_result": "tool_result",
        "scheduled_tick": "scheduler_event",
        "self_cognition": "episode",
        "internal_thought": "episode",
    }.get(trigger_source, "episode")
    source_id = f"episode:{episode_id}"
    semantic_text = fallback_text
    evidence_authority = "current_event"
    dialog_semantic_projection = None
    if source_kind == "episode":
        dialog_semantic_projection = _dialog_semantic_projection_text(
            episode,
        )
        if dialog_semantic_projection is not None:
            semantic_text = dialog_semantic_projection
    percepts = episode.get("percepts")
    if isinstance(percepts, list):
        for percept in percepts:
            if not isinstance(percept, Mapping):
                continue
            if percept.get("visibility") not in {None, "model_visible"}:
                continue
            raw_content = percept.get("content")
            content = _text(raw_content)
            if isinstance(raw_content, Mapping):
                content = _text(
                    raw_content.get("semantic_summary")
                    or raw_content.get("semantic_text")
                    or raw_content.get("text")
                    or raw_content.get("objective")
                    or raw_content.get("artifact_text")
                )
            metadata = percept.get("metadata")
            if not isinstance(metadata, Mapping) and isinstance(raw_content, Mapping):
                metadata = raw_content
            if source_kind == "tool_result":
                typed_source = _typed_tool_result_source(metadata)
                source_id = typed_source["source_id"]
                semantic_text = typed_source["semantic_summary"]
                evidence_authority = _tool_result_evidence_authority(
                    typed_source
                )
            else:
                cognition_source = (
                    metadata.get("cognition_source")
                    if isinstance(metadata, Mapping)
                    else None
                )
                if isinstance(cognition_source, Mapping):
                    typed_source_kind = _text(
                        cognition_source.get("source_kind")
                    )
                    typed_source_id = _text(cognition_source.get("source_id"))
                    typed_summary = _text(
                        cognition_source.get("semantic_summary")
                    )
                    if (
                        typed_source_kind == source_kind
                        and typed_source_id
                        and typed_summary
                    ):
                        source_id = typed_source_id
                        semantic_text = typed_summary
                elif content and not isinstance(cognition_source, Mapping):
                    semantic_text = dialog_semantic_projection or content
                    source_id = _text(percept.get("source_id")) or source_id
            break
    semantic_text = semantic_text[:1000]
    semantic_summary = semantic_text[:500]
    return [{
        "evidence_ref": {
            "source_kind": source_kind,
            "source_id": source_id,
            "occurred_at": occurred_at,
            "semantic_summary": semantic_summary,
        },
        "semantic_text": semantic_text,
        "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS[source_kind]),
        "authority": evidence_authority,
    }]


def _typed_tool_result_source(
    metadata: object,
) -> ToolResultCognitionSourceV1:
    """Require one validated typed tool-result source on a tool percept."""

    if not isinstance(metadata, Mapping):
        raise CognitionExecutionError(
            "tool-result percept requires a typed cognition_source"
        )
    raw_source = metadata.get("cognition_source")
    if not isinstance(raw_source, Mapping):
        raise CognitionExecutionError(
            "tool-result percept requires a typed cognition_source"
        )
    try:
        typed_source = validate_tool_result_cognition_source(raw_source)
    except ValueError as exc:
        raise CognitionExecutionError(
            f"tool-result cognition_source is invalid: {exc}"
        ) from exc
    return typed_source


def _tool_result_episode_cognition_source(
    episode: Mapping[str, Any],
) -> ToolResultCognitionSourceV1:
    """Return the validated typed source of one tool-result episode.

    The typed cognition source is projected from the stored task result and is
    the authoritative delayed-result contract for surface-role derivation.
    """

    percepts = episode.get("percepts")
    if not isinstance(percepts, list):
        raise CognitionExecutionError(
            "tool-result episode requires a typed cognition_source"
        )
    for percept in percepts:
        if not isinstance(percept, Mapping):
            continue
        if percept.get("source_kind") != "tool_result":
            continue
        metadata = percept.get("metadata")
        raw_content = percept.get("content")
        if not isinstance(metadata, Mapping) and isinstance(raw_content, Mapping):
            metadata = raw_content
        return _typed_tool_result_source(metadata)
    raise CognitionExecutionError(
        "tool-result episode requires a typed cognition_source"
    )


def _tool_result_evidence_authority(
    typed_source: ToolResultCognitionSourceV1,
) -> str:
    """Label tool-result evidence without upgrading status text to facts."""

    if typed_source["task_status"] in {"resolved", "partial"}:
        return "current_event"
    return "contextual_fact_only"


def _rag_evidence(value: object, occurred_at: str) -> list[dict[str, Any]]:
    """Convert public RAG evidence fields to typed cognition evidence."""

    if not isinstance(value, Mapping):
        return []
    evidence: list[dict[str, Any]] = []

    memory_rows = value.get("memory_evidence")
    for index, row in enumerate(
        memory_rows if isinstance(memory_rows, list) else [],
        start=1,
    ):
        if not isinstance(row, Mapping):
            continue
        text = _rag_text(row)
        if not text:
            continue

        memory_scope = (
            "current_user_continuity"
            if row.get("scope_type") == "user_continuity"
            else "shared_character_or_world"
        )
        evidence.append(_build_rag_evidence_row(
            text=text,
            source_kind="promoted_memory",
            source_id=str(row.get("id", f"memory:{index}")),
            occurred_at=occurred_at,
            memory_scope=memory_scope,
            authority=(
                "participant_continuity"
                if memory_scope == "current_user_continuity"
                else "character_world_context"
            ),
        ))

    conversation_items = value.get("conversation_evidence")
    for index, item in enumerate(
        conversation_items if isinstance(conversation_items, list) else [],
        start=1,
    ):
        text = (
            _rag_text(item)
            if isinstance(item, Mapping)
            else _text(item)
        )
        if not text:
            continue
        evidence.append(_build_rag_evidence_row(
            text=text,
            source_kind="conversation_evidence",
            source_id=f"rag-conversation:{index}",
            occurred_at=occurred_at,
            authority="participant_continuity",
        ))

    recall_items = value.get("recall_evidence")
    for index, item in enumerate(
        recall_items if isinstance(recall_items, list) else [],
        start=1,
    ):
        if not isinstance(item, Mapping):
            continue
        text = _rag_text(item)
        if not text:
            continue
        evidence.append(_build_rag_evidence_row(
            text=text,
            source_kind="recall_evidence",
            source_id=f"rag-recall:{index}",
            occurred_at=occurred_at,
            authority="contextual_fact_only",
        ))
    return evidence


def _promoted_reflection_evidence(
    value: object,
    occurred_at: str,
) -> list[dict[str, Any]]:
    """Convert bounded promoted reflection memory into typed cognition evidence."""

    if not isinstance(value, Mapping):
        return []
    evidence: list[dict[str, Any]] = []
    lane_fields = (
        ("promoted_lore", "lore"),
        ("promoted_self_guidance", "self_guidance"),
    )
    for field_name, lane_name in lane_fields:
        rows = value.get(field_name)
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows[:3], start=1):
            if not isinstance(row, Mapping):
                continue
            name = _text(row.get("memory_name"))
            content = _text(row.get("content"))
            semantic_text = " — ".join(
                part for part in (name, content) if part
            )
            if not semantic_text:
                continue
            source_timestamp = _reflection_source_timestamp(row)
            if source_timestamp is None:
                continue
            evidence.append(_build_rag_evidence_row(
                text=semantic_text,
                source_kind="promoted_reflection",
                source_id=(
                    f"promoted-reflection:{lane_name}:{index}"
                ),
                occurred_at=source_timestamp,
                authority=(
                    "conditional_character_guidance"
                    if lane_name == "self_guidance"
                    else "character_world_context"
                ),
            ))
    return evidence


def _rag_text(value: Mapping[str, Any]) -> str:
    """Select the canonical prompt-facing text from one RAG item."""

    for field in ("content", "summary", "claim", "text"):
        text = _text(value.get(field))
        if text:
            return text
    return ""


def _reflection_source_timestamp(row: Mapping[str, Any]) -> str | None:
    """Preserve a valid promoted-reflection source timestamp."""

    for field_name in ("updated_at", "effective_at", "created_at"):
        value = row.get(field_name)
        if isinstance(value, datetime):
            value = value.isoformat()
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            return _v2_timestamp(value)
        except ValueError:
            continue
    return None


def _build_rag_evidence_row(
    *,
    text: str,
    source_kind: str,
    source_id: str,
    occurred_at: str,
    authority: str,
    memory_scope: str | None = None,
) -> dict[str, Any]:
    """Build one bounded RAG evidence row with registered provenance."""

    semantic_text = text[:1000]
    row: dict[str, Any] = {
        "evidence_ref": {
            "source_kind": source_kind,
            "source_id": source_id,
            "occurred_at": occurred_at,
            "semantic_summary": semantic_text[:500],
        },
        "semantic_text": semantic_text,
        "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS[source_kind]),
        "authority": authority,
    }
    if memory_scope is not None:
        row["memory_scope"] = memory_scope
    return row


def _media_evidence(
    value: object,
    episode_id: str,
    occurred_at: str,
) -> list[dict[str, Any]]:
    """Project current semantic media observations with typed provenance."""

    if not isinstance(value, list):
        return []
    evidence: list[dict[str, Any]] = []
    for index, row in enumerate(value, start=1):
        if not isinstance(row, Mapping):
            continue
        description = _text(row.get("description"))
        if not description:
            continue
        evidence.append({
            "evidence_ref": {
                "source_kind": "media_observation",
                "source_id": f"episode:{episode_id}:media:{index}",
                "occurred_at": occurred_at,
                "semantic_summary": description,
            },
            "semantic_text": description,
            "visible_to": list(
                EVIDENCE_SOURCE_QUESTION_IDS["media_observation"]
            ),
            "authority": "current_event",
        })
    return evidence


def _resolver_observation_evidence(
    value: object,
    occurred_at: str,
) -> list[dict[str, Any]]:
    """Project resolver observations as typed evidence for the next cycle."""

    if not isinstance(value, list):
        return []
    evidence: list[dict[str, Any]] = []
    for index, row in enumerate(value, start=1):
        if not isinstance(row, Mapping):
            continue
        observation_time = _text(row.get("created_at_utc")) or occurred_at
        try:
            projected, _direct_facts = (
                project_resolver_observation_for_cognition(
                    row,
                    occurred_at=_v2_timestamp(observation_time),
                )
            )
        except (ValueError, TypeError):
            continue
        projected["authority"] = "contextual_fact_only"
        evidence.append(projected)
    return evidence


def _action_result_evidence(
    value: object,
    occurred_at: str,
) -> list[dict[str, Any]]:
    """Project prior action outcomes as typed evidence for later cognition."""

    if not isinstance(value, list):
        return []
    evidence: list[dict[str, Any]] = []
    for index, row in enumerate(value, start=1):
        if not isinstance(row, Mapping):
            continue
        projected = row.get("semantic_result_v2")
        if not isinstance(projected, Mapping):
            projected = project_trace_action_result_v2(row)
        action_kind = _text(projected.get("action_kind"))
        action_status = _text(projected.get("status"))
        summary = _text(projected.get("semantic_result"))
        action_attempt_id = _text(row.get("action_attempt_id"))
        if not action_kind or not action_status or not summary:
            continue
        evidence.append({
            "evidence_ref": {
                "source_kind": "action_result",
                "source_id": action_attempt_id or f"action-result:{index}",
                "occurred_at": _v2_timestamp(
                    _text(row.get("completed_at")) or occurred_at
                ),
                "semantic_summary": summary[:500],
            },
            "semantic_text": f"{action_kind} {action_status}: {summary}"[:1000],
            "visible_to": list(
                EVIDENCE_SOURCE_QUESTION_IDS["action_result"]
            ),
            "authority": "current_event",
        })
    return evidence


def _semantic_episode_text(state: Mapping[str, Any]) -> str:
    """Build one semantic episode description without platform wire syntax."""

    dialog_semantic_projection = None
    episode = state.get("cognitive_episode")
    if isinstance(episode, dict):
        dialog_semantic_projection = _dialog_semantic_projection_text(
            episode,
        )
    value = (
        dialog_semantic_projection
        or state.get("decontextualized_input")
        or state.get("user_input")
    )
    base_text = value.strip() if isinstance(value, str) else ""
    channel_name = _text(state.get("channel_name"))
    channel_topic = _text(state.get("channel_topic"))
    scene_text = base_text
    if channel_name and channel_topic:
        group_context = (
            f'“{channel_name}”群聊中正在讨论：{channel_topic}'
        )
        scene_text = (
            f"{group_context}。{base_text}"
            if base_text
            else group_context
        )
    if not scene_text:
        media = state.get("user_multimedia_input")
        if isinstance(media, list):
            descriptions = [
                _text(row.get("description"))
                for row in media
                if isinstance(row, Mapping) and _text(row.get("description"))
            ]
            if descriptions:
                scene_text = "; ".join(descriptions)
    coding_context = _active_coding_run_scene_text(state)
    if coding_context:
        scene_text = (
            f"{scene_text}。{coding_context}"
            if scene_text
            else coding_context
        )
    return scene_text[:1000] if scene_text else "没有有依据的语义事件"


def _active_coding_run_scene_text(state: Mapping[str, Any]) -> str:
    """Project bounded active coding state into the semantic scene."""

    action_context = state.get("action_selection_context")
    if not isinstance(action_context, Mapping):
        return ""
    raw_contexts = action_context.get("coding_runs")
    if not isinstance(raw_contexts, list):
        return ""
    summaries: list[str] = []
    for context in raw_contexts[:3]:
        if not isinstance(context, Mapping):
            continue
        status = _text(context.get("status"))[:80]
        objective = _text(context.get("objective_summary"))[:160]
        if not status or not objective:
            continue
        raw_actions = context.get("allowed_next_actions")
        actions = [
            action[:60]
            for action in raw_actions
            if isinstance(action, str) and action.strip()
        ] if isinstance(raw_actions, list) else []
        blocker = _coding_run_blocker_summary(
            context.get("active_blocker")
        )
        summaries.append(
            f"状态={status}；目标={objective}；后续动作={actions}；阻塞={blocker}"
        )
    if not summaries:
        return ""
    return "当前作用域已有持久化代码任务状态：" + "；".join(summaries)


def _dialog_semantic_projection_text(
    episode: Mapping[str, Any],
) -> str | None:
    """Render one model-owned current-dialog meaning for cognition."""

    role_explicit_content = project_dialog_role_explicit_content(episode)
    response_operation = project_dialog_response_operation(episode)
    if response_operation is None:
        return role_explicit_content
    projection: dict[str, Any] = {
        "response_operation": response_operation,
    }
    if role_explicit_content is not None:
        projection["role_explicit_content"] = role_explicit_content
    return json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
    )


def _v2_timestamp(value: str) -> str:
    """Project the adapter timestamp into the canonical UTC-Z contract."""

    parsed = parse_storage_utc_datetime(value).astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _continuity_scope_kind(state: Mapping[str, Any]) -> str:
    """Map the canonical episode channel to the event-log scope enum."""

    episode = state.get("cognitive_episode")
    if isinstance(episode, Mapping):
        target_scope = episode.get("target_scope")
        if isinstance(target_scope, Mapping):
            channel_type = target_scope.get("channel_type")
            if channel_type == "group":
                return "group_scene"
            if channel_type == "private":
                return "private"
    return "targetless"


def _semantic_temporal_context(
    conversation_progress: object,
    *,
    current_timestamp: str,
) -> str:
    """Derive scene temporal context from the newest surviving event age.

    When no progress event survives pruning, the derived value describes the
    current turn only.
    """

    if isinstance(conversation_progress, Mapping):
        progress_events = conversation_progress.get('events')
        if isinstance(progress_events, list):
            updated_at_values = [
                _v2_timestamp(event_row['updated_at'])
                for event_row in progress_events
                if (
                    isinstance(event_row, Mapping)
                    and isinstance(event_row.get('updated_at'), str)
                )
            ]
            if updated_at_values:
                newest_event_timestamp = max(updated_at_values)
                temporal_context = project_duration(
                    newest_event_timestamp,
                    current_timestamp,
                )
                return temporal_context
    temporal_context = project_duration(current_timestamp, current_timestamp)
    return temporal_context


def _text(value: object) -> str:
    """Return bounded connector text."""

    return value.strip() if isinstance(value, str) else ""


def _named_role_label(role_label: str, display_name: str) -> str:
    """Render a Chinese role label with its configured display name."""

    if display_name:
        return f"{role_label}（{display_name}）"
    return role_label


def _supports_first_cycle_shared_memory_prewarm(
    state: Mapping[str, Any],
) -> bool:
    """Return whether the episode can enter shared-memory prewarm.

    Args:
        state: Canonical graph state containing one cognitive episode.

    Returns:
        True for trigger sources admitted by the shared-memory helper.
    """

    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        return False
    return episode.get("trigger_source") in {
        "user_message",
        "internal_thought",
    }


def _is_group_self_cognition_state(state: Mapping[str, Any]) -> bool:
    """Return whether cognition observes a targetless group scene.

    Args:
        state: Canonical graph state with channel and episode scope.

    Returns:
        True only for targetless group self-cognition.
    """

    if state.get("channel_type") != "group":
        return False
    if _text(state.get("global_user_id")):
        return False
    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping) or not _is_self_cognition_episode(
        episode
    ):
        return False
    target_scope = episode.get("target_scope")
    if not isinstance(target_scope, Mapping):
        return False
    return (
        target_scope.get("channel_type") == "group"
        and not _text(target_scope.get("current_global_user_id"))
        and not _text(target_scope.get("current_platform_user_id"))
    )


async def _cancel_cognition_preparation_tasks(
    *tasks: asyncio.Task[dict[str, Any]] | None,
) -> None:
    """Cancel and join every started cognition-preparation task.

    Args:
        tasks: Optional preparation tasks owned by one cognition invocation.

    Returns:
        None.
    """

    started_tasks = [task for task in tasks if task is not None]
    for task in started_tasks:
        if not task.done():
            task.cancel()
    if started_tasks:
        await asyncio.gather(*started_tasks, return_exceptions=True)


def _scope_caller(episode: Mapping[str, Any] | object) -> str:
    """Map a typed trigger source to the frozen scope-matrix caller."""

    if isinstance(episode, Mapping) and _is_self_cognition_episode(episode):
        return "self_cognition"
    trigger = episode.get("trigger_source") if isinstance(episode, Mapping) else None
    return {
        "user_message": "persona_user_message",
        "tool_result": "tool_result",
        "self_cognition": "self_cognition",
        "scheduled_tick": "scheduled_tick",
        "internal_thought": "internal_thought",
    }.get(str(trigger), "persona_user_message")


def _is_self_cognition_episode(episode: Mapping[str, Any]) -> bool:
    """Identify the canonical self-cognition packet without parsing its prose."""

    if episode.get("trigger_source") in {
        "self_cognition",
        "scheduled_tick",
        "internal_thought",
    }:
        return True
    percepts = episode.get("percepts")
    if not isinstance(percepts, list):
        return False
    for percept in percepts:
        if not isinstance(percept, Mapping):
            continue
        metadata = percept.get("metadata") or percept.get("content")
        if (
            isinstance(metadata, Mapping)
            and metadata.get("source") == "self_cognition_source_packet"
        ):
            return True
    return False


def _episode_has_source_kind(
    episode: Mapping[str, Any],
    source_kind: str,
) -> bool:
    """Return whether a canonical episode carries one source-kind percept."""

    percepts = episode.get("percepts")
    if not isinstance(percepts, list):
        return False
    return any(
        isinstance(percept, Mapping)
        and percept.get("source_kind") == source_kind
        for percept in percepts
    )


def _scene_channel_scope(
    channel_type: object,
    trigger_source: object,
) -> str:
    """Select the semantic scene scope from the typed episode source."""

    if trigger_source in {
        "internal_thought",
        "self_cognition",
        "scheduled_tick",
    }:
        return "internal"
    if channel_type in {"dm", "private"}:
        return "private"
    return "group"
