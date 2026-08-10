"""Canonical upstream connector for the V2 cognition core."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from datetime import timezone
from typing import Any

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.action_spec.models import (
    ActionAvailabilityContextV1,
    RuntimeCapabilitySnapshotV1,
)
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    build_episode_affordances,
    build_initial_action_capabilities,
    build_runtime_capability_snapshot,
    project_prompt_affordances,
)
from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.results import project_trace_action_result_v2
from kazusa_ai_chatbot.config import (
    AFFECT_SETTLING_WAKE_PREP_MINUTES,
    BACKGROUND_WORK_WORKER_ENABLED,
    CALENDAR_SCHEDULER_ENABLED,
    CHARACTER_SLEEP_LOCAL_PERIOD,
    CHARACTER_TIME_ZONE,
    COGNITION_LLM_API_KEY,
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_MODEL,
    COGNITION_LLM_THINKING_ENABLED,
    COGNITION_STAGE_TIMEOUT_SECONDS,
    COGNITION_LLM_ACTION_AUTHORIZATION_API_KEY,
    COGNITION_LLM_ACTION_AUTHORIZATION_BASE_URL,
    COGNITION_LLM_ACTION_AUTHORIZATION_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_ACTION_AUTHORIZATION_MODEL,
    COGNITION_LLM_ACTION_AUTHORIZATION_THINKING_ENABLED,
    COGNITION_LLM_ACTION_PLANNING_API_KEY,
    COGNITION_LLM_ACTION_PLANNING_BASE_URL,
    COGNITION_LLM_ACTION_PLANNING_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_ACTION_PLANNING_MODEL,
    COGNITION_LLM_ACTION_PLANNING_THINKING_ENABLED,
    COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY_API_KEY,
    COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY_BASE_URL,
    COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY_MODEL,
    COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY_THINKING_ENABLED,
    COGNITION_LLM_APPRAISAL_EVENT_AGENCY_API_KEY,
    COGNITION_LLM_APPRAISAL_EVENT_AGENCY_BASE_URL,
    COGNITION_LLM_APPRAISAL_EVENT_AGENCY_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_APPRAISAL_EVENT_AGENCY_MODEL,
    COGNITION_LLM_APPRAISAL_EVENT_AGENCY_THINKING_ENABLED,
    COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE_API_KEY,
    COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE_BASE_URL,
    COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE_MODEL,
    COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE_THINKING_ENABLED,
    COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME_API_KEY,
    COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME_BASE_URL,
    COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME_MODEL,
    COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME_THINKING_ENABLED,
    COGNITION_LLM_APPRAISAL_MORAL_IDENTITY_API_KEY,
    COGNITION_LLM_APPRAISAL_MORAL_IDENTITY_BASE_URL,
    COGNITION_LLM_APPRAISAL_MORAL_IDENTITY_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_APPRAISAL_MORAL_IDENTITY_MODEL,
    COGNITION_LLM_APPRAISAL_MORAL_IDENTITY_THINKING_ENABLED,
    COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL_API_KEY,
    COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL_BASE_URL,
    COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL_MODEL,
    COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL_THINKING_ENABLED,
    COGNITION_LLM_GOAL_ACTIVE_BRANCH_API_KEY,
    COGNITION_LLM_GOAL_ACTIVE_BRANCH_BASE_URL,
    COGNITION_LLM_GOAL_ACTIVE_BRANCH_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_GOAL_ACTIVE_BRANCH_MODEL,
    COGNITION_LLM_GOAL_ACTIVE_BRANCH_THINKING_ENABLED,
    COGNITION_LLM_GOAL_ORDINARY_RESPONSE_API_KEY,
    COGNITION_LLM_GOAL_ORDINARY_RESPONSE_BASE_URL,
    COGNITION_LLM_GOAL_ORDINARY_RESPONSE_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_GOAL_ORDINARY_RESPONSE_MODEL,
    COGNITION_LLM_GOAL_ORDINARY_RESPONSE_THINKING_ENABLED,
    COGNITION_LLM_RESOLVER_AUTHORIZATION_API_KEY,
    COGNITION_LLM_RESOLVER_AUTHORIZATION_BASE_URL,
    COGNITION_LLM_RESOLVER_AUTHORIZATION_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_RESOLVER_AUTHORIZATION_MODEL,
    COGNITION_LLM_RESOLVER_AUTHORIZATION_THINKING_ENABLED,
    COGNITION_LLM_WORKSPACE_COLLAPSE_API_KEY,
    COGNITION_LLM_WORKSPACE_COLLAPSE_BASE_URL,
    COGNITION_LLM_WORKSPACE_COLLAPSE_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_WORKSPACE_COLLAPSE_MODEL,
    COGNITION_LLM_WORKSPACE_COLLAPSE_THINKING_ENABLED,
)
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeValidationError,
    project_dialog_response_operation,
    project_dialog_role_explicit_content,
    validate_cognitive_episode_v1,
)
from kazusa_ai_chatbot.cognition_core_v2 import run_cognition
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    ActionAffordanceV2,
    CognitionCoreInputV2,
    CognitionCoreOutputV2,
    CognitionCoreServicesV2,
    CognitionExecutionError,
    EVIDENCE_SOURCE_QUESTION_IDS,
    GroupEngagementActionContextV2,
    PAST_DIALOG_COGNITION_CONTEXT_MAX_CHARS,
    ResolverAffordanceV2,
    validate_cognition_core_input,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
    resolve_state_scope,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_character_sleep_phase,
    project_character_operational_state,
    project_duration,
    project_relationship_context,
    select_character_operational_context,
)
from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    merge_shared_memory_prewarm_result,
    project_resolver_observation_for_cognition,
    run_first_cycle_shared_memory_prewarm,
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
from kazusa_ai_chatbot.conversation_progress import (
    project_conversation_progress_evidence,
    project_conversation_progress_scene,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ALLOWED_RESOLVER_CAPABILITIES,
    CURRENT_TURN_RELATIONAL_WILLINGNESS_VERSION,
    RESOLVER_CAPABILITY_REQUEST_VERSION,
    RESOLVER_CAPABILITY_SEMANTICS,
    validate_current_turn_relational_willingness,
)
from kazusa_ai_chatbot.cognition_resolver.state import validate_resolver_state
from kazusa_ai_chatbot.db import (
    compare_and_replace_character_cognition_state,
    get_character_cognition_state,
    get_user_cognition_state,
    replace_user_cognition_state,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.event_logging import record_cognition_v2_event
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_actions import (
    materialize_semantic_action_requests,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_memory_lifecycle import (
    has_trusted_active_commitments,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime

logger = logging.getLogger(__name__)
_llm_interface = LLInterface()
PERSONALITY_JUDGMENT_MAX_CHARS = 180

_cognition_llm_config = LLMCallConfig(
    stage_name="persona_supervisor2_cognition",
    route_name="COGNITION_LLM",
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
    model=COGNITION_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=COGNITION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=COGNITION_LLM_THINKING_ENABLED),
)
_appraisal_event_agency_config = LLMCallConfig(
    stage_name="cognition_core_v2.appraisal_event_agency",
    route_name="COGNITION_LLM_APPRAISAL_EVENT_AGENCY",
    base_url=COGNITION_LLM_APPRAISAL_EVENT_AGENCY_BASE_URL,
    api_key=COGNITION_LLM_APPRAISAL_EVENT_AGENCY_API_KEY,
    model=COGNITION_LLM_APPRAISAL_EVENT_AGENCY_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=(
        COGNITION_LLM_APPRAISAL_EVENT_AGENCY_MAX_COMPLETION_TOKENS
    ),
    presence_penalty=None,
    timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=COGNITION_LLM_APPRAISAL_EVENT_AGENCY_THINKING_ENABLED
    ),
)
_appraisal_relationship_social_config = LLMCallConfig(
    stage_name="cognition_core_v2.appraisal_relationship_social",
    route_name="COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL",
    base_url=COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL_BASE_URL,
    api_key=COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL_API_KEY,
    model=COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=(
        COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL_MAX_COMPLETION_TOKENS
    ),
    presence_penalty=None,
    timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL_THINKING_ENABLED
    ),
)
_appraisal_moral_identity_config = LLMCallConfig(
    stage_name="cognition_core_v2.appraisal_moral_identity",
    route_name="COGNITION_LLM_APPRAISAL_MORAL_IDENTITY",
    base_url=COGNITION_LLM_APPRAISAL_MORAL_IDENTITY_BASE_URL,
    api_key=COGNITION_LLM_APPRAISAL_MORAL_IDENTITY_API_KEY,
    model=COGNITION_LLM_APPRAISAL_MORAL_IDENTITY_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=(
        COGNITION_LLM_APPRAISAL_MORAL_IDENTITY_MAX_COMPLETION_TOKENS
    ),
    presence_penalty=None,
    timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=COGNITION_LLM_APPRAISAL_MORAL_IDENTITY_THINKING_ENABLED
    ),
)
_appraisal_goal_threat_outcome_config = LLMCallConfig(
    stage_name="cognition_core_v2.appraisal_goal_threat_outcome",
    route_name="COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME",
    base_url=COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME_BASE_URL,
    api_key=COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME_API_KEY,
    model=COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=(
        COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME_MAX_COMPLETION_TOKENS
    ),
    presence_penalty=None,
    timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME_THINKING_ENABLED
    ),
)
_appraisal_epistemic_comparison_memory_config = LLMCallConfig(
    stage_name="cognition_core_v2.appraisal_epistemic_comparison_memory",
    route_name="COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY",
    base_url=COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY_BASE_URL,
    api_key=COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY_API_KEY,
    model=COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=(
        COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY_MAX_COMPLETION_TOKENS
    ),
    presence_penalty=None,
    timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=(
            COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY_THINKING_ENABLED
        )
    ),
)
_appraisal_existential_drive_config = LLMCallConfig(
    stage_name="cognition_core_v2.appraisal_existential_drive",
    route_name="COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE",
    base_url=COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE_BASE_URL,
    api_key=COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE_API_KEY,
    model=COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=(
        COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE_MAX_COMPLETION_TOKENS
    ),
    presence_penalty=None,
    timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE_THINKING_ENABLED
    ),
)
_goal_ordinary_response_config = LLMCallConfig(
    stage_name="cognition_core_v2.goal_ordinary_response",
    route_name="COGNITION_LLM_GOAL_ORDINARY_RESPONSE",
    base_url=COGNITION_LLM_GOAL_ORDINARY_RESPONSE_BASE_URL,
    api_key=COGNITION_LLM_GOAL_ORDINARY_RESPONSE_API_KEY,
    model=COGNITION_LLM_GOAL_ORDINARY_RESPONSE_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=(
        COGNITION_LLM_GOAL_ORDINARY_RESPONSE_MAX_COMPLETION_TOKENS
    ),
    presence_penalty=None,
    timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=COGNITION_LLM_GOAL_ORDINARY_RESPONSE_THINKING_ENABLED
    ),
)
_goal_active_branch_config = LLMCallConfig(
    stage_name="cognition_core_v2.goal_active_branch",
    route_name="COGNITION_LLM_GOAL_ACTIVE_BRANCH",
    base_url=COGNITION_LLM_GOAL_ACTIVE_BRANCH_BASE_URL,
    api_key=COGNITION_LLM_GOAL_ACTIVE_BRANCH_API_KEY,
    model=COGNITION_LLM_GOAL_ACTIVE_BRANCH_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=(
        COGNITION_LLM_GOAL_ACTIVE_BRANCH_MAX_COMPLETION_TOKENS
    ),
    presence_penalty=None,
    timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=COGNITION_LLM_GOAL_ACTIVE_BRANCH_THINKING_ENABLED
    ),
)
_workspace_collapse_config = LLMCallConfig(
    stage_name="cognition_core_v2.workspace_collapse",
    route_name="COGNITION_LLM_WORKSPACE_COLLAPSE",
    base_url=COGNITION_LLM_WORKSPACE_COLLAPSE_BASE_URL,
    api_key=COGNITION_LLM_WORKSPACE_COLLAPSE_API_KEY,
    model=COGNITION_LLM_WORKSPACE_COLLAPSE_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=(
        COGNITION_LLM_WORKSPACE_COLLAPSE_MAX_COMPLETION_TOKENS
    ),
    presence_penalty=None,
    timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=COGNITION_LLM_WORKSPACE_COLLAPSE_THINKING_ENABLED
    ),
)
_action_planning_config = LLMCallConfig(
    stage_name="cognition_core_v2.action_planning",
    route_name="COGNITION_LLM_ACTION_PLANNING",
    base_url=COGNITION_LLM_ACTION_PLANNING_BASE_URL,
    api_key=COGNITION_LLM_ACTION_PLANNING_API_KEY,
    model=COGNITION_LLM_ACTION_PLANNING_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=COGNITION_LLM_ACTION_PLANNING_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=COGNITION_LLM_ACTION_PLANNING_THINKING_ENABLED
    ),
)
_action_authorization_config = LLMCallConfig(
    stage_name="cognition_core_v2.action_authorization",
    route_name="COGNITION_LLM_ACTION_AUTHORIZATION",
    base_url=COGNITION_LLM_ACTION_AUTHORIZATION_BASE_URL,
    api_key=COGNITION_LLM_ACTION_AUTHORIZATION_API_KEY,
    model=COGNITION_LLM_ACTION_AUTHORIZATION_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=(
        COGNITION_LLM_ACTION_AUTHORIZATION_MAX_COMPLETION_TOKENS
    ),
    presence_penalty=None,
    timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=COGNITION_LLM_ACTION_AUTHORIZATION_THINKING_ENABLED
    ),
)
_resolver_authorization_config = LLMCallConfig(
    stage_name="cognition_core_v2.resolver_authorization",
    route_name="COGNITION_LLM_RESOLVER_AUTHORIZATION",
    base_url=COGNITION_LLM_RESOLVER_AUTHORIZATION_BASE_URL,
    api_key=COGNITION_LLM_RESOLVER_AUTHORIZATION_API_KEY,
    model=COGNITION_LLM_RESOLVER_AUTHORIZATION_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=(
        COGNITION_LLM_RESOLVER_AUTHORIZATION_MAX_COMPLETION_TOKENS
    ),
    presence_penalty=None,
    timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=COGNITION_LLM_RESOLVER_AUTHORIZATION_THINKING_ENABLED
    ),
)


def build_cognition_core_services() -> CognitionCoreServicesV2:
    """Build the injected V2 model bindings."""

    return CognitionCoreServicesV2(
        llm=_llm_interface,
        appraisal_event_agency_config=_appraisal_event_agency_config,
        appraisal_relationship_social_config=(
            _appraisal_relationship_social_config
        ),
        appraisal_moral_identity_config=_appraisal_moral_identity_config,
        appraisal_goal_threat_outcome_config=(
            _appraisal_goal_threat_outcome_config
        ),
        appraisal_epistemic_comparison_memory_config=(
            _appraisal_epistemic_comparison_memory_config
        ),
        appraisal_existential_drive_config=_appraisal_existential_drive_config,
        goal_ordinary_response_config=_goal_ordinary_response_config,
        goal_active_branch_config=_goal_active_branch_config,
        workspace_collapse_config=_workspace_collapse_config,
        action_planning_config=_action_planning_config,
        action_authorization_config=_action_authorization_config,
        resolver_authorization_config=_resolver_authorization_config,
    )


def build_cognition_input_from_global_state(
    state: GlobalPersonaState,
    *,
    mutable_state: Mapping[str, Any] | None = None,
    character_state: Mapping[str, Any] | None = None,
) -> CognitionCoreInputV2:
    """Map adapter-neutral graph state into one native V2 cognition scope."""

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
            "character profile is required for V2 cognition"
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
    conversation_episode_state = state.get("conversation_episode_state")
    if conversation_progress is None:
        conversation_continuity = ""
    elif not isinstance(conversation_progress, dict):
        raise CognitionExecutionError(
            "conversation progress must be a V2 prompt mapping"
        )
    else:
        conversation_continuity = project_conversation_progress_scene(
            conversation_progress,
        )
        evidence.extend(project_conversation_progress_evidence(
            conversation_progress,
            timestamp,
        ))
    if conversation_continuity:
        conversation_continuity = (
            'Current participant continuity:\n'
            f'{conversation_continuity}'
        )[:2200].rstrip()
    semantic_temporal_context = _semantic_temporal_context(
        conversation_episode_state,
        current_timestamp=timestamp,
    )
    public_group_scene = _text(state['public_group_scene'])[:1800]
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
    for index, row in enumerate(evidence, start=1):
        row["evidence_handle"] = f"e{index}"
    scope = selected_mutable_state["state_scope"]
    channel_scope = _scene_channel_scope(
        episode["target_scope"].get("channel_type"),
        episode.get("trigger_source"),
    )
    character_name = (
        _text(character_profile.get("name"))
        if isinstance(character_profile, Mapping)
        else ""
    )
    user_name = _text(state.get("user_name"))
    character_label = _named_role_label("当前角色", character_name)
    current_user_label = _named_role_label("当前用户", user_name)
    character_role = character_label
    current_user_role = current_user_label
    character_sleep_phase = project_character_sleep_phase(
        parse_storage_utc_datetime(timestamp),
        sleep_local_period=CHARACTER_SLEEP_LOCAL_PERIOD,
        character_time_zone=CHARACTER_TIME_ZONE,
        wake_prep_minutes=AFFECT_SETTLING_WAKE_PREP_MINUTES,
    )
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
    payload: CognitionCoreInputV2 = {
        "schema_version": "cognition_core_input.v2",
        "episode": dict(episode),
        "state_scope": scope,
        "mutable_state": dict(selected_mutable_state),
        "character_constraints": constraints,
        "character_identity_context": character_identity_context,
        "character_operational_context": character_operational_context,
        "evidence": evidence[:32],
        "direct_facts": _typed_direct_facts(state.get("direct_facts")),
        "available_actions": _available_action_affordances(state),
        "available_resolver_capabilities": _available_resolver_affordances(state),
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
        "scene_context": {
            "channel_scope": channel_scope,
            "character_role": character_role,
            "current_user_role": current_user_role,
            "character_sleep_phase": character_sleep_phase,
            "semantic_scene": semantic_text[:500],
            "public_group_scene": public_group_scene,
            "conversation_continuity": conversation_continuity,
            "semantic_temporal_context": semantic_temporal_context,
        },
    }
    participant_bindings = state.get("scene_participant_bindings")
    if isinstance(participant_bindings, list) and participant_bindings:
        payload["scene_context"]["participant_bindings"] = [
            dict(binding)
            for binding in participant_bindings
            if isinstance(binding, Mapping)
        ]
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
        relational_carrier = resolver_state.get(
            "current_turn_relational_willingness"
        )
        if isinstance(relational_carrier, Mapping):
            payload["current_turn_relational_willingness"] = dict(
                relational_carrier
            )
    return validate_cognition_core_input(payload)


async def call_cognition_subgraph(
    state: GlobalPersonaState,
    *,
    commit: bool = True,
) -> GlobalPersonaState:
    """Run V2 cognition, commit its one replacement state, then expose projections."""

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
    trace_token = llm_tracing.bind_trace_id(
        str(state.get("llm_trace_id") or ""),
    )
    try:
        output = await run_cognition(
            cognition_input,
            build_cognition_core_services(),
        )
    finally:
        llm_tracing.reset_trace_id(trace_token)
    if commit:
        await _commit_cognition_state(
            output,
            expected_character_updated_at=character_base_updated_at,
        )
    update = _project_output_to_global_state(output, state)
    if character_base_updated_at is not None:
        update["character_cognition_base_updated_at"] = (
            character_base_updated_at
        )
    update["cognition_input"] = cognition_input
    update["cognition_core_output"] = output
    update["cognition_scope"] = output["state_update"]["state_scope"]
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
    output: CognitionCoreOutputV2,
    *,
    expected_character_updated_at: str | None = None,
) -> None:
    """Commit one already-validated V2 result at the final episode boundary."""

    await _commit_cognition_state(
        output,
        expected_character_updated_at=expected_character_updated_at,
    )


async def _commit_cognition_state(
    output: CognitionCoreOutputV2,
    *,
    expected_character_updated_at: str | None = None,
) -> None:
    """Commit the validated replacement before any downstream surface/action work."""

    state_update = output["state_update"]
    replacement = state_update["replacement_state"]
    try:
        if state_update["state_scope"] == "user":
            await replace_user_cognition_state(
                state_update["owner_key"],
                replacement,
            )
        else:
            if not expected_character_updated_at:
                raise CognitionExecutionError(
                    "character state commit requires its base version"
                )
            committed = await compare_and_replace_character_cognition_state(
                expected_updated_at=expected_character_updated_at,
                replacement=replacement,
            )
            if not committed:
                raise CognitionExecutionError(
                    "character state commit encountered a version conflict"
                )
    except Exception:
        await _record_state_commit_event(output, succeeded=False)
        raise
    await _record_state_commit_event(output, succeeded=True)


async def _record_state_commit_event(
    output: CognitionCoreOutputV2,
    *,
    succeeded: bool,
) -> None:
    """Emit best-effort bounded telemetry for one terminal state commit."""

    intention = output["intention"]
    try:
        await record_cognition_v2_event(
            component="nodes.persona_supervisor2_cognition",
            cognition_component="state_commit",
            status="completed" if succeeded else "failed",
            stage_status="completed" if succeeded else "failed",
            selected_branch_id=intention.get("selected_branch_id", ""),
            state_scope=output["state_update"]["state_scope"],
            state_commit_status="committed" if succeeded else "failed",
            severity="info" if succeeded else "error",
        )
    except Exception as exc:
        logger.warning("V2 state-commit event write failed: %s", type(exc).__name__)


def _project_output_to_global_state(
    output: CognitionCoreOutputV2,
    state: GlobalPersonaState,
) -> dict[str, Any]:
    """Expose semantic outputs while preserving deterministic action ownership."""

    affect = output["affect_projection"]
    dominant = affect[0] if affect else None
    route = output["intention"]["route"]
    update: dict[str, Any] = {
        "cognition_state_update": output["state_update"],
        "cognition_intention": output["intention"],
        "semantic_affect_projection": affect,
        "semantic_relationship_projection": output.get("relationship_projection"),
        "goal_resolution": output["goal_resolution"],
        "resolver_capability_requests": [
            {
                "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
                "capability_kind": request["capability"],
                "objective": request["semantic_goal"],
                "reason": request["reason"],
                "priority": _resolver_request_priority(request),
            }
            for request in output["resolver_requests"]
        ],
        "resolver_pending_resolution": output["resolver_pending_resolution"],
        "resolver_goal_progress": output["resolver_goal_progress"],
        "cognition_resolver_progress": output["resolver_progress"],
        "action_specs": _materialize_v2_action_requests(output, state),
        "internal_monologue": output["private_monologue"],
        "interaction_subtext": output["selected_bid_reason"],
        "emotional_appraisal": dominant["emotion"] if dominant else "平静",
        "character_intent": output["intention"]["intention"],
        "logical_stance": output["intention"]["reason"],
        "judgment_note": output["intention"]["reason"],
        "social_distance": "受语义关系背景约束",
        "emotional_intensity": dominant["intensity"] if dominant else "无",
        "vibe_check": dominant["phase"] if dominant else "平静",
        "relational_dynamic": (
            output.get("relationship_projection", {}).get(
                "relationship_summary",
                "没有关系投影",
            )
            if isinstance(output.get("relationship_projection"), Mapping)
            else "没有关系投影"
        ),
        "should_respond": route != "silence",
        "rag_result": state.get("rag_result", {}),
    }
    if "self_cognition_response" in output:
        update["self_cognition_response"] = dict(
            output["self_cognition_response"]
        )
    if "self_cognition_response_contract_status" in output:
        update["self_cognition_response_contract_status"] = output[
            "self_cognition_response_contract_status"
        ]
    resolver_state = state.get("resolver_state")
    relational_decision = output.get("relational_willingness")
    if (
        isinstance(resolver_state, Mapping)
        and resolver_state.get("cycle_index") == 0
        and isinstance(relational_decision, Mapping)
    ):
        episode = state.get("cognitive_episode")
        episode_id = (
            episode.get("episode_id")
            if isinstance(episode, Mapping)
            else None
        )
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise CognitionExecutionError(
                "current-turn relational carrier requires episode identity"
            )
        carrier = {
            "schema_version": CURRENT_TURN_RELATIONAL_WILLINGNESS_VERSION,
            "episode_id": episode_id,
            "branch_id": "ordinary_response",
            "decision": dict(relational_decision),
        }
        validated_carrier = validate_current_turn_relational_willingness(
            carrier,
            episode_id=episode_id,
        )
        updated_resolver_state = dict(resolver_state)
        updated_resolver_state["current_turn_relational_willingness"] = (
            validated_carrier
        )
        update["resolver_state"] = validate_resolver_state(
            updated_resolver_state
        )
    return update


def _resolver_request_priority(request: Mapping[str, Any]) -> str:
    """Project the validated task-resolution boolean to the V1 priority."""

    if request["capability"] == "task_resolution_request":
        if request["start_in_background"] is True:
            return "background"
        return "now"
    return "now"


def _materialize_v2_action_requests(
    output: CognitionCoreOutputV2,
    state: GlobalPersonaState,
) -> list[dict[str, Any]]:
    """Materialize private requests and the selected V2 surface action."""

    requests = []
    evaluator = ActionSpecEvaluator()
    available_action_kinds = set(build_initial_action_capabilities())
    for request in output["action_requests"]:
        evaluation = evaluator.evaluate_v2_request(
            request,
            available_action_kinds=available_action_kinds,
        )
        if not evaluation["ok"]:
            raise CognitionExecutionError(
                "V2 action request failed deterministic validation"
            )
        requests.append({
            "capability": request["action_kind"],
            "decision": request["decision"],
            "context_ref": request["context_ref"],
            "detail": request["semantic_goal"],
            "reason": request["reason"],
            "target_roles": list(request["target_roles"]),
            "evidence_handles": list(request["evidence_handles"]),
        })
    materialization_state = dict(state)
    action_specs = materialize_semantic_action_requests(
        requests,
        materialization_state,
    )
    if output["intention"]["route"] != "speech":
        return action_specs

    admitted_bid = output.get("admitted_bid")
    evidence_handles = (
        list(admitted_bid["evidence_handles"])
        if isinstance(admitted_bid, Mapping)
        else []
    )
    speak_specs = materialize_semantic_action_requests(
        [{
            "capability": SPEAK_CAPABILITY,
            "decision": "visible_reply",
            "detail": output["intention"]["intention"],
            "reason": output["intention"]["reason"],
            "target_roles": list(output["intention"]["target_roles"]),
            "evidence_handles": evidence_handles,
        }],
        materialization_state,
    )
    if len(speak_specs) != 1:
        raise CognitionExecutionError(
            "V2 speech intention failed action-spec materialization"
        )
    return [*action_specs, speak_specs[0]]


def _available_action_affordances(
    state: GlobalPersonaState,
) -> list[ActionAffordanceV2]:
    """Project the deterministic capability registry into V2 affordances."""

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
    affordances: list[ActionAffordanceV2] = []
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

    return [
        {
            "capability": capability,
            "semantic_capability": RESOLVER_CAPABILITY_SEMANTICS[capability],
            "availability": "available",
        }
        for capability in sorted(available_capabilities)
    ]


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
    """Project the current source through its canonical V2 evidence kind."""

    trigger_source = str(episode.get("trigger_source", "user_message"))
    source_kind = {
        "tool_result": "tool_result",
        "scheduled_tick": "scheduler_event",
        "self_cognition": "episode",
        "internal_thought": "episode",
    }.get(trigger_source, "episode")
    source_id = f"episode:{episode_id}"
    semantic_text = fallback_text
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
            if source_kind == "tool_result" and isinstance(
                metadata,
                Mapping,
            ):
                    if not isinstance(cognition_source, Mapping):
                        source_id = (
                            _text(metadata.get("task_id"))
                            or _text(percept.get("source_id"))
                            or source_id
                        )
                    semantic_text = _tool_result_text(
                        metadata,
                        content,
                        fallback_text,
                    )
            elif content and not isinstance(cognition_source, Mapping):
                semantic_text = dialog_semantic_projection or content
                source_id = _text(percept.get("source_id")) or source_id
            break
    semantic_text = semantic_text[:1000]
    semantic_summary = semantic_text[:500]
    return [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": source_kind,
            "source_id": source_id,
            "occurred_at": occurred_at,
            "semantic_summary": semantic_summary,
        },
        "semantic_text": semantic_text,
        "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS[source_kind]),
    }]


def _tool_result_text(
    metadata: Mapping[str, Any],
    content: str,
    fallback_text: str,
) -> str:
    """Build bounded semantic evidence from a completed tool outcome."""

    parts = [
        _text(metadata.get("semantic_summary")),
        _text(metadata.get("result_summary")),
        _text(metadata.get("failure_summary")),
        content,
    ]
    semantic_text = "; ".join(part for part in parts if part)
    return semantic_text or fallback_text


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
        ))
    return evidence


def _promoted_reflection_evidence(
    value: object,
    occurred_at: str,
) -> list[dict[str, Any]]:
    """Convert bounded promoted reflection memory into typed V2 evidence."""

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
            evidence.append(_build_rag_evidence_row(
                text=semantic_text,
                source_kind="promoted_reflection",
                source_id=(
                    f"promoted-reflection:{lane_name}:{index}"
                ),
                occurred_at=occurred_at,
            ))
    return evidence


def _rag_text(value: Mapping[str, Any]) -> str:
    """Select the canonical prompt-facing text from one RAG item."""

    for field in ("content", "summary", "claim", "text"):
        text = _text(value.get(field))
        if text:
            return text
    return ""


def _build_rag_evidence_row(
    *,
    text: str,
    source_kind: str,
    source_id: str,
    occurred_at: str,
    memory_scope: str | None = None,
) -> dict[str, Any]:
    """Build one bounded RAG evidence row with registered provenance."""

    semantic_text = text[:1000]
    row: dict[str, Any] = {
        "evidence_handle": "ev0",
        "evidence_ref": {
            "source_kind": source_kind,
            "source_id": source_id,
            "occurred_at": occurred_at,
            "semantic_summary": semantic_text[:500],
        },
        "semantic_text": semantic_text,
        "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS[source_kind]),
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
            "evidence_handle": "e0",
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
        projected["evidence_handle"] = f"e{index}"
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
            "evidence_handle": f"e{index}",
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
    """Project the adapter timestamp into the native V2 UTC-Z contract."""

    parsed = parse_storage_utc_datetime(value).astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


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
