"""Canonical upstream connector for the V2 cognition core."""

from __future__ import annotations

import logging
from datetime import timezone
from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.action_spec.registry import (
    build_initial_action_capabilities,
)
from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.config import (
    BOUNDARY_CORE_LLM_API_KEY,
    BOUNDARY_CORE_LLM_BASE_URL,
    BOUNDARY_CORE_LLM_MAX_COMPLETION_TOKENS,
    BOUNDARY_CORE_LLM_MODEL,
    BOUNDARY_CORE_LLM_THINKING_ENABLED,
    COGNITION_LLM_API_KEY,
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_MODEL,
    COGNITION_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.cognition_core_v2 import run_cognition
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    ActionAffordanceV2,
    CognitionCoreInputV2,
    CognitionCoreOutputV2,
    CognitionCoreServicesV2,
    CognitionExecutionError,
    ResolverAffordanceV2,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
    resolve_state_scope,
    validate_cognition_state,
)
from kazusa_ai_chatbot.db import (
    get_character_cognition_state,
    get_user_cognition_state,
    replace_character_cognition_state,
    replace_user_cognition_state,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_actions import (
    materialize_semantic_action_requests,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime
from kazusa_ai_chatbot.utils import parse_llm_json_output


logger = logging.getLogger(__name__)
_llm_interface = LLInterface()

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
_boundary_core_llm_config = LLMCallConfig(
    stage_name="persona_supervisor2_boundary",
    route_name="BOUNDARY_CORE_LLM",
    base_url=BOUNDARY_CORE_LLM_BASE_URL,
    api_key=BOUNDARY_CORE_LLM_API_KEY,
    model=BOUNDARY_CORE_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=BOUNDARY_CORE_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=BOUNDARY_CORE_LLM_THINKING_ENABLED),
)


def build_cognition_core_services() -> CognitionCoreServicesV2:
    """Build the injected V2 model and parser bindings."""

    return CognitionCoreServicesV2(
        llm=_llm_interface,
        appraisal_config=_cognition_llm_config,
        goal_cognition_config=_cognition_llm_config,
        collapse_config=_cognition_llm_config,
        action_selection_config=_boundary_core_llm_config,
        parse_json=parse_llm_json_output,
        logger=logger,
    )


def build_cognition_input_from_global_state(
    state: GlobalPersonaState,
    *,
    mutable_state: Mapping[str, Any] | None = None,
    character_state: Mapping[str, Any] | None = None,
) -> CognitionCoreInputV2:
    """Map adapter-neutral graph state into one native V2 cognition scope."""

    timestamp = _v2_timestamp(state["storage_timestamp_utc"])
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
    constraints = {
        "drives": selected_character_state["drives"],
        "standards": selected_character_state["standards"],
        "meaning_state": selected_character_state["meaning_state"],
    }
    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        episode = {}
    episode_id = str(episode.get("episode_id", "episode"))
    semantic_text = _semantic_episode_text(state)
    evidence = [{
        "evidence_handle": "ev1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": f"episode:{episode_id}",
            "occurred_at": timestamp,
            "semantic_summary": semantic_text,
        },
        "semantic_text": semantic_text,
        "visible_to": ["cognition", "surface"],
    }]
    evidence.extend(_rag_evidence(state.get("rag_result"), timestamp))
    evidence.extend(_resolver_observation_evidence(
        state.get("resolver_observations"),
        timestamp,
    ))
    scope = selected_mutable_state["state_scope"]
    channel_scope = "private" if state["channel_type"] in {"dm", "private"} else "group"
    payload: CognitionCoreInputV2 = {
        "schema_version": "cognition_core_input.v2",
        "episode": {
            "episode_id": episode_id,
            "trigger_source": str(episode.get("trigger_source", "user_message")),
            "output_mode": _output_mode(episode.get("output_mode")),
            "semantic_scene": semantic_text,
            "semantic_temporal_context": "immediate",
            "interaction_style_context": _interaction_style_context(state),
        },
        "state_scope": scope,
        "mutable_state": dict(selected_mutable_state),
        "character_constraints": constraints,
        "evidence": evidence[:32],
        "direct_facts": _typed_direct_facts(state.get("direct_facts")),
        "available_actions": _available_action_affordances(state),
        "available_resolver_capabilities": _available_resolver_affordances(state),
        "scene_context": {
            "channel_scope": channel_scope,
            "character_role": "character",
            "current_user_role": "participant",
            "semantic_scene": semantic_text,
            "semantic_temporal_context": "immediate",
        },
    }
    if scope == "character" and isinstance(selected_mutable_state.get("relationship"), Mapping):
        payload["relationship_context"] = dict(selected_mutable_state["relationship"])
    return payload


async def call_cognition_subgraph(
    state: GlobalPersonaState,
    *,
    commit: bool = True,
) -> GlobalPersonaState:
    """Run V2 cognition, commit its one replacement state, then expose projections."""

    episode = state.get("cognitive_episode")
    caller = _scope_caller(episode)
    target_user_id = state.get("global_user_id")
    origin_scope = episode.get("origin_scope") if isinstance(episode, Mapping) else None
    scope, owner = resolve_state_scope(
        caller,
        target_user_id,
        origin_scope=tuple(origin_scope) if isinstance(origin_scope, list) else origin_scope,
    )
    if scope == "character":
        mutable_state = await get_character_cognition_state()
        character_state = mutable_state
    else:
        mutable_state = await get_user_cognition_state(owner)
        character_state = await get_character_cognition_state()
    cognition_input = build_cognition_input_from_global_state(
        state,
        mutable_state=mutable_state,
        character_state=character_state,
    )
    output = await run_cognition(cognition_input, build_cognition_core_services())
    if commit:
        await _commit_cognition_state(output)
    update = _project_output_to_global_state(output, state)
    update["cognition_input"] = cognition_input
    update["cognition_core_output"] = output
    update["cognition_scope"] = output["state_update"]["state_scope"]
    return update  # type: ignore[return-value]


async def commit_cognition_output(output: CognitionCoreOutputV2) -> None:
    """Commit one already-validated V2 result at the final episode boundary."""

    await _commit_cognition_state(output)


async def _commit_cognition_state(output: CognitionCoreOutputV2) -> None:
    """Commit the validated replacement before any downstream surface/action work."""

    state_update = output["state_update"]
    replacement = state_update["replacement_state"]
    if state_update["state_scope"] == "user":
        await replace_user_cognition_state(state_update["owner_key"], replacement)
    else:
        await replace_character_cognition_state(replacement)


def _project_output_to_global_state(
    output: CognitionCoreOutputV2,
    state: GlobalPersonaState,
) -> dict[str, Any]:
    """Expose semantic outputs while preserving deterministic action ownership."""

    affect = output["affect_projection"]
    dominant = affect[0] if affect else None
    route = output["intention"]["route"]
    return {
        "cognition_state_update": output["state_update"],
        "cognition_intention": output["intention"],
        "semantic_affect_projection": affect,
        "semantic_relationship_projection": output.get("relationship_projection"),
        "resolver_capability_requests": output["resolver_requests"],
        "cognition_resolver_progress": output["resolver_progress"],
        "action_specs": _materialize_v2_action_requests(output, state),
        "internal_monologue": output["residue"],
        "interaction_subtext": output["residue"],
        "emotional_appraisal": dominant["emotion"] if dominant else "composed",
        "character_intent": output["intention"]["intention"],
        "logical_stance": output["intention"]["reason"],
        "judgment_note": output["intention"]["reason"],
        "social_distance": "bounded by semantic relationship context",
        "emotional_intensity": dominant["intensity"] if dominant else "none",
        "vibe_check": dominant["phase"] if dominant else "composed",
        "relational_dynamic": (
            output.get("relationship_projection", {}).get(
                "relationship_summary",
                "no relationship projection",
            )
            if isinstance(output.get("relationship_projection"), Mapping)
            else "no relationship projection"
        ),
        "should_respond": route != "silence",
        "rag_result": state.get("rag_result", {}),
    }


def _materialize_v2_action_requests(
    output: CognitionCoreOutputV2,
    state: GlobalPersonaState,
) -> list[dict[str, Any]]:
    """Materialize only route-approved action requests through the existing owner."""

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
            "decision": output["intention"]["route"],
            "detail": request["semantic_goal"],
            "reason": output["intention"]["reason"],
            "target_roles": list(request["target_roles"]),
            "evidence_handles": list(request["evidence_handles"]),
        })
    if not requests:
        return []
    materialization_state = dict(state)
    return materialize_semantic_action_requests(requests, materialization_state)


def _available_action_affordances(
    state: Mapping[str, Any],
) -> list[ActionAffordanceV2]:
    """Project the deterministic capability registry into V2 affordances."""

    current_user = {
        "role": "target",
        "entity_kind": "user",
        "entity_id": str(state.get("global_user_id", "")),
    }
    return [
        {
            "action_kind": capability["capability_kind"],
            "capability": capability["capability_kind"],
            "permission": "allowed",
            "target_roles": [current_user],
        }
        for capability in build_initial_action_capabilities().values()
        if capability["capability_kind"] != "apply_memory_lifecycle_update"
    ]


def _available_resolver_affordances(
    state: Mapping[str, Any],
) -> list[ResolverAffordanceV2]:
    """Project resolver capabilities as availability, not execution authority."""

    return [{
        "capability": "local_context_recall",
        "semantic_capability": "retrieve bounded evidence",
        "availability": "available",
    }]


def _typed_direct_facts(value: object) -> list[dict[str, Any]]:
    """Accept only caller-supplied typed facts at the connector boundary."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _rag_evidence(value: object, occurred_at: str) -> list[dict[str, Any]]:
    """Convert RAG rows to evidence with complete source provenance."""

    if not isinstance(value, Mapping):
        return []
    rows = value.get("memory_evidence")
    if not isinstance(rows, list):
        return []
    evidence: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=2):
        if not isinstance(row, Mapping):
            continue
        text = _text(row.get("content") or row.get("summary"))
        if not text:
            continue
        evidence.append({
            "evidence_handle": f"ev{index}",
            "evidence_ref": {
                "source_kind": "promoted_memory",
                "source_id": str(row.get("id", f"memory:{index}")),
                "occurred_at": occurred_at,
                "semantic_summary": text,
            },
            "semantic_text": text,
            "visible_to": ["cognition"],
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
        observation_id = _text(row.get("observation_id"))
        summary = _text(
            row.get("semantic_summary") or row.get("prompt_safe_summary")
        )
        capability = _text(row.get("capability")) or _text(
            row.get("capability_kind")
        )
        if not observation_id or not summary:
            continue
        source_id = observation_id or f"resolver:{index}"
        evidence.append({
            "evidence_handle": f"resolver_ev{index}",
            "evidence_ref": {
                "source_kind": "resolver_observation",
                "source_id": source_id,
                "occurred_at": _text(row.get("created_at_utc")) or occurred_at,
                "semantic_summary": summary,
            },
            "semantic_text": (
                f"{capability}: {summary}" if capability else summary
            ),
            "visible_to": ["cognition"],
        })
    return evidence


def _semantic_episode_text(state: Mapping[str, Any]) -> str:
    """Build one semantic episode description without platform wire syntax."""

    value = state.get("decontexualized_input") or state.get("user_input")
    base_text = value.strip() if isinstance(value, str) else ""
    channel_name = _text(state.get("channel_name"))
    channel_topic = _text(state.get("channel_topic"))
    if channel_name and channel_topic:
        group_context = (
            f'“{channel_name}”群聊中正在讨论：{channel_topic}'
        )
        return f"{group_context}。{base_text}" if base_text else group_context
    if base_text:
        return base_text
    media = state.get("user_multimedia_input")
    if isinstance(media, list):
        descriptions = [
            _text(row.get("description"))
            for row in media
            if isinstance(row, Mapping) and _text(row.get("description"))
        ]
        if descriptions:
            return "; ".join(descriptions)
    return "no grounded semantic episode"


def _v2_timestamp(value: str) -> str:
    """Project the adapter timestamp into the native V2 UTC-Z contract."""

    parsed = parse_storage_utc_datetime(value).astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(value: object) -> str:
    """Return bounded connector text."""

    return value.strip() if isinstance(value, str) else ""


def _output_mode(value: object) -> str:
    """Normalize upstream expression policy to the V2 episode vocabulary."""

    if value in {"silence", "think_only", "preview", "visible_reply"}:
        return str(value)
    if value == "live_response":
        return "visible_reply"
    return "visible_reply"


def _scope_caller(episode: Mapping[str, Any] | object) -> str:
    """Map a typed trigger source to the frozen scope-matrix caller."""

    trigger = episode.get("trigger_source") if isinstance(episode, Mapping) else None
    return {
        "user_message": "persona_user_message",
        "accepted_task_result": "accepted_task_result_ready",
        "background_result": "background_result",
        "group_message": "group_sender",
        "reflection_signal": "reflection",
        "reflection_dry_run": "reflection_dry_run",
        "internal_thought": "internal_thought_cognition",
        "scheduler_event": "scheduler",
        "resolver_recurrence": "resolver_recurrence",
    }.get(str(trigger), "persona_user_message")


def _interaction_style_context(state: Mapping[str, Any]) -> str:
    """Project approved character style and boundary context semantically."""

    profile = state.get("character_profile")
    if not isinstance(profile, Mapping):
        return "use the character's established interaction style and boundaries"
    personality = profile.get("personality_brief")
    style = profile.get("linguistic_texture_profile")
    boundary = profile.get("boundary_profile")
    personality_terms = []
    if isinstance(personality, Mapping):
        for field_name in ("logic", "tempo", "defense", "quirks", "taboos"):
            value = _text(personality.get(field_name))
            if value:
                personality_terms.append(value)
    return "; ".join((
        "character style is semantically bounded",
        *personality_terms,
        _semantic_profile_terms(
            style,
            "voice texture",
            (
                "hesitation_density",
                "fragmentation",
                "emotional_leakage",
                "direct_assertion",
                "softener_density",
            ),
        ),
        _semantic_profile_terms(
            boundary,
            "boundary profile",
            (
                "self_integrity",
                "control_sensitivity",
                "relational_override",
                "authority_skepticism",
            ),
        ),
    ))


def _semantic_profile_terms(
    value: object,
    label: str,
    dimensions: tuple[str, ...],
) -> str:
    """Describe configured profile dimensions without exposing raw scalars."""

    if not isinstance(value, Mapping):
        return f"{label} remains established"
    present = [
        dimension.replace("_", " ")
        for dimension in dimensions
        if dimension in value
    ]
    if not present:
        return f"{label} remains established"
    return f"{label} covers {', '.join(present)}"
