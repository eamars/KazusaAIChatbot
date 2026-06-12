"""Kazusa connector for the reusable cognition-chain core."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.config import (
    BOUNDARY_CORE_LLM_API_KEY,
    BOUNDARY_CORE_LLM_BASE_URL,
    BOUNDARY_CORE_LLM_MODEL,
    BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    COGNITION_LLM_API_KEY,
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MODEL,
)
from kazusa_ai_chatbot.action_spec.registry import (
    BACKGROUND_WORK_REQUEST_CAPABILITY,
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
    build_initial_action_capabilities,
    project_prompt_affordances,
)
from kazusa_ai_chatbot.cognition_chain_core.chain import run_cognition_chain
from kazusa_ai_chatbot.cognition_chain_core.contracts import (
    CognitionChainInputV1,
    CognitionChainOutputV1,
    CognitionChainServices,
    validate_cognition_chain_input,
    validate_cognition_chain_output,
)
from kazusa_ai_chatbot.cognition_chain_core.episode_projection import (
    public_output_mode,
)
from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    merge_shared_memory_prewarm_result,
    run_first_cycle_shared_memory_prewarm,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ResolverValidationError,
    validate_resolver_capability_request,
    validate_resolver_goal_progress,
    validate_resolver_pending_resolution,
)
from kazusa_ai_chatbot.db import build_group_engagement_action_context
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_actions import (
    materialize_semantic_action_requests,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
    GlobalPersonaState,
)
from kazusa_ai_chatbot.utils import (
    build_interaction_history_recent,
    get_llm,
    log_preview,
    parse_llm_json_output,
)

logger = logging.getLogger(__name__)

_CORE_FORBIDDEN_FIELD_NAMES = frozenset((
    "action_specs",
    "adapter_id",
    "collection_name",
    "credentials",
    "delivery_target",
    "handler_id",
    "job_id",
    "lease",
    "platform_bot_id",
    "platform_channel_id",
    "platform_message_id",
    "platform_user_id",
    "queue_future",
    "raw_adapter_payload",
    "scheduler",
    "target_id",
    "worker",
))

_cognition_llm = get_llm(
    temperature=0.1,
    top_p=0.7,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)
_boundary_core_llm = get_llm(
    temperature=0.1,
    top_p=0.7,
    model=BOUNDARY_CORE_LLM_MODEL,
    base_url=BOUNDARY_CORE_LLM_BASE_URL,
    api_key=BOUNDARY_CORE_LLM_API_KEY,
)


def build_cognition_chain_input_from_global_state(
    state: GlobalPersonaState,
) -> CognitionChainInputV1:
    """Project Kazusa graph state into the public cognition-chain input."""

    character_profile = state["character_profile"]
    user_profile = state["user_profile"]
    cognitive_episode = state.get("cognitive_episode", {})
    if not isinstance(cognitive_episode, Mapping):
        cognitive_episode = {}
    origin_metadata = cognitive_episode.get("origin_metadata")
    if not isinstance(origin_metadata, Mapping):
        origin_metadata = {
            "debug_modes": state.get("debug_modes", {}),
        }
    rag_result = state["rag_result"]
    if not isinstance(rag_result, Mapping):
        rag_result = {}
    interaction_history_recent = build_interaction_history_recent(
        state["chat_history_wide"],
        state["platform_user_id"],
        state["platform_bot_id"],
        state["global_user_id"],
    )
    memory_context = _memory_context_from_rag(rag_result)
    media_observations = _media_observations_from_state(state, cognitive_episode)
    model_visible_percepts = _model_visible_percepts_from_state(
        state,
        cognitive_episode,
        media_observations,
    )
    payload: CognitionChainInputV1 = {
        "schema_version": "cognition_chain_input.v1",
        "episode": {
            "episode_id": _text(cognitive_episode.get("episode_id")),
            "trigger_source": _text(cognitive_episode.get("trigger_source")),
            "input_sources": _string_list(cognitive_episode.get("input_sources")),
            "output_mode": _output_mode(cognitive_episode.get("output_mode")),
            "model_visible_percepts": model_visible_percepts,
            "target_scope_summary": state["channel_type"],
            "origin_summary": state["platform"],
            "origin_metadata": _prompt_safe_mapping(origin_metadata),
        },
        "character": {
            "character_global_id": _text(character_profile.get("global_user_id")),
            "name": _text(character_profile.get("name")),
            "description": _text(character_profile.get("description")),
            "gender": _text(character_profile.get("gender")),
            "age": _text(character_profile.get("age")),
            "birthday": _text(character_profile.get("birthday")),
            "backstory": _text(character_profile.get("background")),
            "personality_brief": _prompt_safe_mapping(
                character_profile.get("personality_brief")
            ),
            "boundary_profile": _prompt_safe_mapping(
                character_profile.get("boundary_profile")
            ),
            "linguistic_texture_profile": _prompt_safe_mapping(
                character_profile.get("linguistic_texture_profile")
            ),
            "mood": _text(character_profile.get("mood")),
            "global_vibe": _text(character_profile.get("global_vibe")),
        },
        "current_user": {
            "global_user_id": state["global_user_id"],
            "display_name": state["user_name"],
            "affinity": _affinity_value(user_profile.get("affinity")),
            "affinity_level": _text(user_profile.get("affinity_level")),
            "last_relationship_insight": _text(
                user_profile.get("last_relationship_insight")
            ),
            "memory_context": memory_context,
            "profile": _prompt_safe_mapping(user_profile),
        },
        "current_event": {
            "user_input": state["user_input"],
            "decontextualized_input": state["decontexualized_input"],
            "indirect_speech_context": state["indirect_speech_context"],
            "referents": _prompt_safe_mapping_list(state["referents"]),
            "media_observations": media_observations,
            "reply_context_summary": _text(state["reply_context"]),
            "prompt_message_context_summary": _text(
                state["prompt_message_context"]
            ),
            "reply_context": _prompt_safe_mapping(state["reply_context"]),
            "prompt_message_context": _prompt_safe_mapping(
                state["prompt_message_context"]
            ),
        },
        "scene": {
            "platform": state["platform"],
            "channel_type": state["channel_type"],
            "channel_topic": state["channel_topic"],
            "local_time_context": _prompt_safe_mapping(
                state["local_time_context"]
            ),
            "storage_timestamp_utc": state["storage_timestamp_utc"],
            "interaction_history_recent": _prompt_safe_mapping_list(
                interaction_history_recent
            ),
        },
        "conversation_context": {
            "conversation_progress": _prompt_safe_mapping(
                state.get("conversation_progress")
            ),
            "promoted_reflection_context": _prompt_safe_mapping(
                state.get("promoted_reflection_context")
            ),
            "internal_monologue_residue_context": state.get(
                "internal_monologue_residue_context",
                "",
            ),
            "previous_action_summary": "",
        },
        "evidence": {
            "rag_answer": _text(rag_result.get("answer")),
            "current_user_rag_bundle": _prompt_safe_value(
                rag_result.get("current_user_rag_bundle", {})
            ),
            "memory_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "recall_evidence": [],
            "supervisor_trace": [],
            "rag_result": _prompt_safe_mapping(rag_result),
        },
        "resolver": {
            "resolver_context": state.get("resolver_context", ""),
            "pending_resume": _prompt_safe_value(
                state.get("pending_resolver_resume", "")
            ),
            "goal_progress": _prompt_safe_value(
                state.get("resolver_goal_progress", "")
            ),
            "recent_observations": [],
            "max_projected_observations": 3,
            "resolver_state": _prompt_safe_mapping(state.get("resolver_state", {})),
        },
        "available_actions": _available_action_affordances(),
        "runtime_context": {
            "language_policy": "simplified_chinese_internal_text",
            "visual_directives_enabled": True,
            "max_action_requests": 3,
            "max_resolver_requests": 3,
            "background_work_output_char_limit": (
                BACKGROUND_WORK_OUTPUT_CHAR_LIMIT
            ),
        },
    }
    validated_payload = validate_cognition_chain_input(payload)
    return validated_payload


def apply_cognition_chain_output_to_global_state(
    output: CognitionChainOutputV1,
    state: GlobalPersonaState,
) -> dict[str, Any]:
    """Map core output back to Kazusa graph updates."""

    validated_output = validate_cognition_chain_output(output)
    residue = validated_output["cognition_residue"]
    semantic_requests = validated_output["semantic_action_requests"]
    action_specs = materialize_semantic_action_requests(
        semantic_requests,
        _initial_cognition_state_from_global_state(state),
    )
    resolver_capability_requests = _validated_resolver_capability_requests(
        validated_output["resolver_capability_requests"],
    )
    return_value: dict[str, Any] = {
        "internal_monologue": residue["internal_monologue"],
        "action_specs": action_specs,
        "resolver_capability_requests": resolver_capability_requests,
        "interaction_subtext": residue["interaction_subtext"],
        "emotional_appraisal": residue["emotional_appraisal"],
        "character_intent": residue["character_intent"],
        "logical_stance": residue["logical_stance"],
        "judgment_note": residue["judgment_note"],
        "social_distance": residue["social_distance"],
        "emotional_intensity": residue["emotional_intensity"],
        "vibe_check": residue["vibe_check"],
        "relational_dynamic": residue["relational_dynamic"],
        "rag_result": state["rag_result"],
    }
    resolver_pending_resolution = _validated_resolver_pending_resolution(
        validated_output.get("resolver_pending_resolution"),
    )
    if resolver_pending_resolution is not None:
        return_value["resolver_pending_resolution"] = resolver_pending_resolution
    resolver_goal_progress = _validated_resolver_goal_progress(
        validated_output.get("resolver_goal_progress"),
    )
    if resolver_goal_progress is not None:
        return_value["resolver_goal_progress"] = resolver_goal_progress
    return return_value


async def call_cognition_subgraph(state: GlobalPersonaState) -> GlobalPersonaState:
    """Run the cognition-chain core for one persona turn."""

    chain_input = build_cognition_chain_input_from_global_state(state)
    prewarm_task: asyncio.Task[dict[str, Any]] | None = None
    group_engagement_task: asyncio.Task[dict[str, Any]] | None = None
    resolver_state = state.get("resolver_state")
    if (
        isinstance(resolver_state, Mapping)
        and resolver_state.get("cycle_index") == 0
    ):
        prewarm_task = asyncio.create_task(
            run_first_cycle_shared_memory_prewarm(state),
        )
    if _is_group_self_cognition_state(state):
        group_engagement_task = asyncio.create_task(
            build_group_engagement_action_context(
                channel_type=state["channel_type"],
                platform=state["platform"],
                platform_channel_id=state["platform_channel_id"],
            ),
        )
    if prewarm_task is not None:
        prewarm_rag_result = await prewarm_task
        chain_input["evidence"]["rag_result"] = _prompt_safe_mapping(
            merge_shared_memory_prewarm_result(
                dict(chain_input["evidence"]["rag_result"]),
                prewarm_rag_result,
            ),
        )
    if group_engagement_task is not None:
        group_engagement_context = await group_engagement_task
        chain_input["action_selection_context"] = {
            "group_engagement_action_context": _prompt_safe_mapping(
                group_engagement_context
            ),
        }
    chain_input = validate_cognition_chain_input(chain_input)
    chain_output = await run_cognition_chain(
        chain_input,
        build_cognition_chain_services(),
    )
    update = apply_cognition_chain_output_to_global_state(chain_output, state)
    update["rag_result"] = chain_input["evidence"].get(
        "rag_result",
        state["rag_result"],
    )
    _log_cognition_output(chain_input, update)
    return update


def build_cognition_chain_services() -> CognitionChainServices:
    """Build injected services for the reusable core."""

    services = CognitionChainServices(
        cognition_llm=_cognition_llm,
        boundary_core_llm=_boundary_core_llm,
        action_selection_llm=_cognition_llm,
        style_llm=_cognition_llm,
        content_plan_llm=_cognition_llm,
        preference_llm=_cognition_llm,
        visual_llm=_cognition_llm,
        parse_json=parse_llm_json_output,
        logger=logger,
    )
    return services


def _initial_cognition_state_from_global_state(
    state: GlobalPersonaState,
) -> dict[str, Any]:
    """Project graph state into the moved internal stage-state shape."""

    interaction_history_recent = build_interaction_history_recent(
        state["chat_history_wide"],
        state["platform_user_id"],
        state["platform_bot_id"],
        state["global_user_id"],
    )
    initial_state: dict[str, Any] = {
        "character_profile": state["character_profile"],
        "storage_timestamp_utc": state["storage_timestamp_utc"],
        "local_time_context": state["local_time_context"],
        "user_input": state["user_input"],
        "prompt_message_context": state["prompt_message_context"],
        "platform": state["platform"],
        "platform_channel_id": state["platform_channel_id"],
        "channel_type": state["channel_type"],
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "user_profile": state["user_profile"],
        "platform_bot_id": state["platform_bot_id"],
        "chat_history_recent": interaction_history_recent,
        "reply_context": state["reply_context"],
        "indirect_speech_context": state["indirect_speech_context"],
        "channel_topic": state["channel_topic"],
        "conversation_progress": state.get("conversation_progress"),
        "promoted_reflection_context": state.get("promoted_reflection_context"),
        "internal_monologue_residue_context": state.get(
            "internal_monologue_residue_context",
            "",
        ),
        "decontexualized_input": state["decontexualized_input"],
        "referents": state["referents"],
        "rag_result": state["rag_result"],
        "resolver_context": state.get("resolver_context", ""),
    }
    for optional_key in (
        "resolver_state",
        "pending_resolver_resume",
        "resolver_goal_progress",
        "cognitive_episode",
    ):
        optional_value = state.get(optional_key)
        if optional_value is not None:
            initial_state[optional_key] = optional_value
    return initial_state


def _is_group_self_cognition_state(state: Mapping[str, Any]) -> bool:
    """Return whether the current cognition state is group self-cognition."""

    if state["channel_type"] != "group":
        return_value = False
        return return_value

    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        return_value = False
        return return_value

    trigger_source = episode.get("trigger_source")
    input_sources = episode.get("input_sources")
    is_internal_monologue = (
        isinstance(input_sources, list)
        and "internal_monologue" in input_sources
    )
    is_self_cognition = (
        trigger_source == "internal_thought"
        and is_internal_monologue
    )
    return is_self_cognition


def _chain_output_from_result(result: Mapping[str, Any]) -> CognitionChainOutputV1:
    """Project core graph state into the sealed core output contract."""

    output: CognitionChainOutputV1 = {
        "schema_version": "cognition_chain_output.v1",
        "cognition_residue": {
            "emotional_appraisal": _text(result.get("emotional_appraisal")),
            "interaction_subtext": _text(result.get("interaction_subtext")),
            "internal_monologue": _text(result.get("internal_monologue")),
            "logical_stance": _text(result.get("logical_stance")),
            "character_intent": _text(result.get("character_intent")),
            "judgment_note": _text(result.get("judgment_note")),
            "social_distance": _text(result.get("social_distance")),
            "emotional_intensity": _text(result.get("emotional_intensity")),
            "vibe_check": _text(result.get("vibe_check")),
            "relational_dynamic": _text(result.get("relational_dynamic")),
        },
        "semantic_action_requests": list(
            result.get("semantic_action_requests", [])
        ),
        "resolver_capability_requests": list(
            result.get("resolver_capability_requests", [])
        ),
        "chain_trace": {
            "stage_order": ["l1", "l2a", "l2b", "l2c1", "l2c2", "l2d"],
            "selected_actions_summary": "",
            "resolver_summary": "",
            "warnings": [],
        },
    }
    resolver_pending_resolution = result.get("resolver_pending_resolution")
    if isinstance(resolver_pending_resolution, dict):
        output["resolver_pending_resolution"] = resolver_pending_resolution
    resolver_goal_progress = result.get("resolver_goal_progress")
    if isinstance(resolver_goal_progress, dict):
        output["resolver_goal_progress"] = resolver_goal_progress
    validated_output = validate_cognition_chain_output(output)
    return validated_output


def _memory_context_from_rag(
    rag_result: Mapping[str, Any],
) -> dict[str, str]:
    """Project prompt-safe user memory context from RAG result."""

    user_image = rag_result.get("user_image")
    if not isinstance(user_image, Mapping):
        return_value = {
            "durable_profile_summary": "",
            "relationship_summary": "",
            "recent_commitments_summary": "",
            "known_preferences_summary": "",
        }
        return return_value
    memory_context = user_image.get("user_memory_context")
    if not isinstance(memory_context, Mapping):
        return_value = {
            "durable_profile_summary": "",
            "relationship_summary": "",
            "recent_commitments_summary": "",
            "known_preferences_summary": "",
        }
        return return_value
    return_value = {
        "durable_profile_summary": _text(memory_context.get("profile_summary")),
        "relationship_summary": _text(memory_context.get("relationship_summary")),
        "recent_commitments_summary": _text(memory_context.get("active_commitments")),
        "known_preferences_summary": _text(memory_context.get("preferences")),
    }
    return return_value


def _media_observations_from_state(
    state: Mapping[str, Any],
    cognitive_episode: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Project prompt-safe media observations into the public cognition ICD."""

    observations = _media_observations_from_multimedia(
        state.get("user_multimedia_input"),
    )
    if observations:
        return observations
    observations = _media_observations_from_episode(cognitive_episode)
    return observations


def _media_observations_from_multimedia(value: object) -> list[dict[str, str]]:
    """Build public media rows from graph multimedia input."""

    if not isinstance(value, list):
        return_value: list[dict[str, str]] = []
        return return_value
    observations: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        modality = _media_modality(item.get("content_type"))
        if modality == "unknown":
            continue
        observation = _media_observation_text(item)
        if not observation:
            continue
        observations.append({
            "modality": modality,
            "observation": observation,
            "source_summary": "current attachment",
        })
    return observations


def _media_observations_from_episode(
    cognitive_episode: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Build public media rows from already-projected episode percepts."""

    raw_percepts = cognitive_episode.get("percepts")
    if not isinstance(raw_percepts, list):
        return_value: list[dict[str, str]] = []
        return return_value
    observations: list[dict[str, str]] = []
    for percept in raw_percepts:
        if not isinstance(percept, Mapping):
            continue
        input_source = percept.get("input_source")
        if input_source == "image_observation":
            modality = "image"
        elif input_source == "audio_observation":
            modality = "audio"
        else:
            continue
        observation = _text(percept.get("content"))
        if not observation:
            continue
        observations.append({
            "modality": modality,
            "observation": observation,
            "source_summary": "episode percept",
        })
    return observations


def _model_visible_percepts_from_state(
    state: Mapping[str, Any],
    cognitive_episode: Mapping[str, Any],
    media_observations: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Build public model-visible percept rows from graph state."""

    percepts: list[dict[str, object]] = [{
        "percept_id": "current_input",
        "input_source": "chat_message",
        "content": state["decontexualized_input"],
        "metadata_summary": [],
    }]
    episode_media_percepts = _media_percepts_from_episode(
        cognitive_episode.get("percepts"),
    )
    if episode_media_percepts:
        percepts.extend(episode_media_percepts)
        return percepts
    for index, observation in enumerate(media_observations, start=1):
        input_source = _media_input_source(observation["modality"])
        if not input_source:
            continue
        percepts.append({
            "percept_id": f"current_media_{index}",
            "input_source": input_source,
            "content": observation["observation"],
            "metadata_summary": [observation["source_summary"]],
        })
    return percepts


def _media_percepts_from_episode(value: object) -> list[dict[str, object]]:
    """Project prompt-safe media percepts already present on the episode."""

    if not isinstance(value, list):
        return_value: list[dict[str, object]] = []
        return return_value
    percepts: list[dict[str, object]] = []
    for index, percept in enumerate(value, start=1):
        if not isinstance(percept, Mapping):
            continue
        input_source = percept.get("input_source")
        if input_source not in ("image_observation", "audio_observation"):
            continue
        content = _text(percept.get("content"))
        if not content:
            continue
        percepts.append({
            "percept_id": _text(percept.get("percept_id"))
            or f"episode_media_{index}",
            "input_source": input_source,
            "content": content,
            "metadata_summary": ["episode percept"],
        })
    return percepts


def _media_modality(content_type: object) -> str:
    """Return the public media modality for a content type."""

    if not isinstance(content_type, str):
        return_value = "unknown"
        return return_value
    if content_type.startswith("image/"):
        return_value = "image"
        return return_value
    if content_type.startswith("audio/"):
        return_value = "audio"
        return return_value
    return_value = "unknown"
    return return_value


def _media_observation_text(item: Mapping[str, Any]) -> str:
    """Return one prompt-safe media description."""

    image_observation = item.get("image_observation")
    if isinstance(image_observation, Mapping):
        summary = _text(image_observation.get("summary")).strip()
        if summary:
            return summary
    return_value = _text(item.get("description")).strip()
    return return_value


def _media_input_source(modality: str) -> str:
    """Map public media modality to prompt-selection input source."""

    if modality == "image":
        return_value = "image_observation"
        return return_value
    if modality == "audio":
        return_value = "audio_observation"
        return return_value
    return_value = ""
    return return_value


def _available_action_affordances() -> list[dict[str, object]]:
    """Project Kazusa action registry affordances into the core ICD."""

    capabilities = build_initial_action_capabilities()
    prompt_affordances = project_prompt_affordances(capabilities)
    allowed_capabilities = {
        SPEAK_CAPABILITY,
        MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        TRIGGER_FUTURE_COGNITION_CAPABILITY,
        BACKGROUND_WORK_REQUEST_CAPABILITY,
    }
    affordances: list[dict[str, object]] = []
    for prompt_affordance in prompt_affordances:
        capability = prompt_affordance.get("capability")
        if capability not in allowed_capabilities:
            continue
        visibility = prompt_affordance.get("visibility")
        if visibility == "user_visible":
            visibility = "public"
        if visibility not in ("public", "private", "internal"):
            visibility = "private"
        affordances.append({
            "capability": capability,
            "available": True,
            "visibility": visibility,
            "semantic_input_summary": prompt_affordance.get(
                "semantic_input_summary",
                "",
            ),
            "output_kind": "semantic_action_request",
        })
    return affordances


def _prompt_safe_mapping(value: object) -> dict[str, Any]:
    """Return a mapping scrubbed of raw adapter or execution fields."""

    safe_value = _prompt_safe_value(value)
    if isinstance(safe_value, dict):
        return safe_value
    return_value: dict[str, Any] = {}
    return return_value


def _prompt_safe_mapping_list(value: object) -> list[dict[str, Any]]:
    """Return prompt-safe mapping rows from a list-like value."""

    if not isinstance(value, list):
        return_value: list[dict[str, Any]] = []
        return return_value
    rows = [
        safe_item for safe_item in (
            _prompt_safe_value(item) for item in value
        )
        if isinstance(safe_item, dict)
    ]
    return rows


def _prompt_safe_value(value: object) -> object:
    """Recursively remove fields forbidden by the cognition core ICD."""

    if isinstance(value, Mapping):
        safe_mapping: dict[str, Any] = {}
        for key, nested_value in value.items():
            key_text = str(key)
            if key_text in _CORE_FORBIDDEN_FIELD_NAMES:
                continue
            safe_mapping[key_text] = _prompt_safe_value(nested_value)
        return safe_mapping
    if isinstance(value, list):
        return [_prompt_safe_value(item) for item in value]
    return value


def _affinity_value(value: object) -> int | str:
    """Return affinity without forcing numeric state into text."""

    if isinstance(value, int):
        return_value: int | str = value
        return return_value
    return_value = _text(value)
    return return_value


def _validated_resolver_capability_requests(value: object) -> list[dict]:
    """Validate L2d resolver requests before returning to persona graph."""

    if value is None:
        return_value: list[dict] = []
        return return_value
    if not isinstance(value, list):
        logger.warning("Cognition dropped non-list resolver capability requests")
        return_value = []
        return return_value

    validated_requests: list[dict] = []
    for raw_request in value:
        try:
            validated_request = validate_resolver_capability_request(raw_request)
        except ResolverValidationError as exc:
            logger.warning(f"Cognition dropped invalid resolver request: {exc}")
            continue
        validated_requests.append(validated_request)
    return_value = validated_requests
    return return_value


def _validated_resolver_pending_resolution(value: object) -> dict | None:
    """Validate an optional L2d pending-resolver decision."""

    if value is None:
        return_value = None
        return return_value
    try:
        validated_resolution = validate_resolver_pending_resolution(value)
    except ResolverValidationError as exc:
        logger.warning(f"Cognition dropped invalid pending resolver decision: {exc}")
        return_value = None
        return return_value
    return_value = validated_resolution
    return return_value


def _validated_resolver_goal_progress(value: object) -> dict | None:
    """Validate L2d's optional goal-progress checklist."""

    if value is None:
        return_value = None
        return return_value
    try:
        validated_progress = validate_resolver_goal_progress(value)
    except ResolverValidationError as exc:
        logger.warning(f"Cognition dropped invalid goal progress: {exc}")
        return_value = None
        return return_value
    return_value = validated_progress
    return return_value


def _log_cognition_output(
    chain_input: CognitionChainInputV1,
    update: Mapping[str, Any],
) -> None:
    """Log bounded cognition input and output summaries."""

    logger.info(
        f"Cognition output: stance={update['logical_stance']} "
        f"intent={update['character_intent']} "
        f"appraisal={log_preview(update['emotional_appraisal'])} "
        f"subtext={log_preview(update['interaction_subtext'])} "
        f"action_specs={log_preview(update.get('action_specs', []))} "
        f"resolver_capabilities="
        f"{log_preview(_resolver_request_log_rows(update.get('resolver_capability_requests', [])))} "
        f"resolver_pending_resolution="
        f"{log_preview(_pending_resolution_log_row(update.get('resolver_pending_resolution')))} "
        f"monologue={log_preview(update['internal_monologue'])}"
    )
    logger.debug(
        f"Cognition input: input="
        f"{log_preview(chain_input['current_event']['decontextualized_input'])}"
    )


def _resolver_request_log_rows(requests: object) -> list[dict[str, str]]:
    """Build bounded resolver request log rows without raw prompt text."""

    rows: list[dict[str, str]] = []
    if not isinstance(requests, list):
        return rows
    for request in requests:
        if not isinstance(request, Mapping):
            continue
        rows.append({
            "capability_kind": str(request.get("capability_kind", "")),
            "priority": str(request.get("priority", "")),
            "objective": str(request.get("objective", ""))[:120],
            "reason": str(request.get("reason", ""))[:120],
        })
    return rows


def _pending_resolution_log_row(resolution: object) -> dict[str, str]:
    """Build a small log row for pending resolver closure decisions."""

    if not isinstance(resolution, Mapping):
        return_value: dict[str, str] = {}
        return return_value
    return_value = {
        "decision": str(resolution.get("decision", "")),
        "reason": str(resolution.get("reason", ""))[:120],
    }
    return return_value


def _output_mode(value: object) -> str:
    """Return a supported core output mode."""

    return_value = public_output_mode(value)
    return return_value


def _string_list(value: object) -> list[str]:
    """Return a list of stringified values."""

    if not isinstance(value, list):
        return_value: list[str] = []
        return return_value
    return_value = [str(item) for item in value]
    return return_value


def _text(value: object) -> str:
    """Return a prompt-safe text representation for connector projection."""

    if value is None:
        return_value = ""
        return return_value
    if isinstance(value, str):
        return_value = value
        return return_value
    return_value = str(value)
    return return_value
