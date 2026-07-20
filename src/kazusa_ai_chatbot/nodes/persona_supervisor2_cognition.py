"""Canonical upstream connector for the V2 cognition core."""

from __future__ import annotations

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
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    build_episode_affordances,
    build_initial_action_capabilities,
    build_runtime_capability_snapshot,
    project_prompt_affordances,
)
from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.results import project_trace_action_result_v2
from kazusa_ai_chatbot.config import (
    BOUNDARY_CORE_LLM_API_KEY,
    BOUNDARY_CORE_LLM_BASE_URL,
    BOUNDARY_CORE_LLM_MAX_COMPLETION_TOKENS,
    BOUNDARY_CORE_LLM_MODEL,
    BOUNDARY_CORE_LLM_THINKING_ENABLED,
    BACKGROUND_WORK_WORKER_ENABLED,
    CALENDAR_SCHEDULER_ENABLED,
    COGNITION_LLM_API_KEY,
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_MODEL,
    COGNITION_LLM_THINKING_ENABLED,
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
    ResolverAffordanceV2,
    validate_cognition_core_input,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
    resolve_state_scope,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    project_resolver_observation_for_cognition,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ALLOWED_RESOLVER_CAPABILITIES,
    RESOLVER_CAPABILITY_REQUEST_VERSION,
    RESOLVER_CAPABILITY_SEMANTICS,
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
    """Build the injected V2 model bindings."""

    return CognitionCoreServicesV2(
        llm=_llm_interface,
        appraisal_config=_cognition_llm_config,
        goal_cognition_config=_cognition_llm_config,
        collapse_config=_cognition_llm_config,
        action_selection_config=_boundary_core_llm_config,
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
    constraints = {
        "drives": selected_character_state["drives"],
        "standards": selected_character_state["standards"],
        "meaning_state": selected_character_state["meaning_state"],
    }
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
    evidence.extend(_rag_evidence(state.get("rag_result"), timestamp))
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
    character_role = "active character"
    current_user_role = "current conversation participant"
    if (
        episode["trigger_source"] == "user_message"
        and _episode_has_source_kind(episode, "dialog")
    ):
        character_role = (
            "the active character; direct addressee of dialog_text and "
            "implicit subject of a direct imperative"
        )
        current_user_role = (
            "the current message author; speaker of dialog_text and owner of "
            "first-person pronouns in that evidence"
        )
    payload: CognitionCoreInputV2 = {
        "schema_version": "cognition_core_input.v2",
        "episode": dict(episode),
        "state_scope": scope,
        "mutable_state": dict(selected_mutable_state),
        "character_constraints": constraints,
        "evidence": evidence[:32],
        "direct_facts": _typed_direct_facts(state.get("direct_facts")),
        "available_actions": _available_action_affordances(state),
        "available_resolver_capabilities": _available_resolver_affordances(state),
        "resolver_context": _text(state.get("resolver_context"))[:8000],
        "private_continuity_context": _text(
            state.get("internal_monologue_residue_context")
        )[:1000],
        "scene_context": {
            "channel_scope": channel_scope,
            "character_role": character_role,
            "current_user_role": current_user_role,
            "semantic_scene": semantic_text[:500],
            "conversation_continuity": _conversation_progress_text(
                state.get("conversation_progress")
            )[:1000],
            "semantic_temporal_context": "immediate",
        },
    }
    pending_resume = state.get("pending_resolver_resume")
    if isinstance(pending_resume, Mapping):
        payload["pending_resolver_resume"] = dict(pending_resume)
    resolver_state = state.get("resolver_state")
    if isinstance(resolver_state, Mapping):
        goal_progress = resolver_state.get("goal_progress")
        if isinstance(goal_progress, Mapping):
            payload["resolver_goal_progress"] = dict(goal_progress)
    return validate_cognition_core_input(payload)


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
    prior_update = state.get("cognition_state_update")
    prior_replacement: Mapping[str, Any] | None = None
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
        else:
            mutable_state = prior_replacement
        character_state = mutable_state
    else:
        if prior_replacement is None:
            mutable_state = await get_user_cognition_state(owner)
        else:
            mutable_state = prior_replacement
        character_state = await get_character_cognition_state()
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
    try:
        if state_update["state_scope"] == "user":
            await replace_user_cognition_state(
                state_update["owner_key"],
                replacement,
            )
        else:
            await replace_character_cognition_state(replacement)
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
    return {
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
                "priority": "now",
            }
            for request in output["resolver_requests"]
        ],
        "resolver_pending_resolution": output["resolver_pending_resolution"],
        "resolver_goal_progress": output["resolver_goal_progress"],
        "cognition_resolver_progress": output["resolver_progress"],
        "action_specs": _materialize_v2_action_requests(output, state),
        "internal_monologue": output["private_monologue"],
        "interaction_subtext": output["selected_bid_reason"],
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
            "decision": request["decision"],
            "context_ref": request["context_ref"],
            "detail": request["semantic_goal"],
            "reason": request["reason"],
            "target_roles": list(request["target_roles"]),
            "evidence_handles": list(request["evidence_handles"]),
        })
    if not requests:
        return []
    materialization_state = dict(state)
    return materialize_semantic_action_requests(requests, materialization_state)


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
        affordances.append({
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
        })
        if capability_kind == ACCEPTED_CODING_TASK_REQUEST_CAPABILITY:
            contextual_affordances = _coding_run_action_affordances(
                state,
                base_affordance=affordances[-1],
            )
            affordances[-1]["allowed_decisions"] = ["start"]
            affordances[-1]["default_decision"] = "start"
            affordances[-1]["capability"] += (
                " This base affordance starts a new run and has no active "
                "run context."
            )
            affordances.extend(contextual_affordances)
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
        "accepted_task": (
            "healthy" if BACKGROUND_WORK_WORKER_ENABLED else "unavailable"
        ),
        "background_work": (
            "healthy" if BACKGROUND_WORK_WORKER_ENABLED else "unavailable"
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
    registry_decisions = {
        "start",
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
        ]
        allowed_decisions = list(dict.fromkeys(allowed_decisions))
        if not allowed_decisions:
            continue
        status = _text(context.get("status"))[:80]
        objective = _text(context.get("objective_summary"))[:120]
        blocker_summary = _coding_run_blocker_summary(
            context.get("active_blocker")
        )
        semantic_context = (
            f" Active run status: {status}. Objective: {objective}. "
            f"Active blocker: {blocker_summary}."
        )
        default_decision = (
            "status" if "status" in allowed_decisions else allowed_decisions[0]
        )
        affordances.append({
            "action_kind": base_affordance["action_kind"],
            "capability": (
                base_affordance["capability"][:260] + semantic_context
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
    """Project resolver capabilities as availability, not execution authority."""

    return [
        {
            "capability": capability,
            "semantic_capability": RESOLVER_CAPABILITY_SEMANTICS[capability],
            "availability": "available",
        }
        for capability in sorted(ALLOWED_RESOLVER_CAPABILITIES)
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
            "visible_to": list(
                EVIDENCE_SOURCE_QUESTION_IDS["promoted_memory"]
            ),
        })
    return evidence


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
        or state.get("decontexualized_input")
        or state.get("user_input")
    )
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


def _conversation_progress_text(value: object) -> str:
    """Render only prompt-safe semantic fields from conversation progress."""

    if not isinstance(value, Mapping):
        return ""
    fragments: list[str] = []
    for field_name, label in (
        ("continuity", "continuity"),
        ("current_thread", "current thread"),
        ("user_goal", "user goal"),
        ("current_blocker", "current blocker"),
        ("emotional_trajectory", "emotional trajectory"),
        ("progression_guidance", "progression guidance"),
    ):
        text = _text(value.get(field_name))
        if text:
            fragments.append(f"{label}: {text}")
    for field_name, label in (
        ("overused_moves", "avoid repeating"),
        ("next_affordances", "next affordances"),
    ):
        rows = value.get(field_name)
        if isinstance(rows, list):
            texts = [_text(row) for row in rows if _text(row)]
            if texts:
                fragments.append(f"{label}: {', '.join(texts)}")
    for field_name, label in (
        ("open_loops", "open loops"),
        ("avoid_reopening", "avoid reopening"),
    ):
        rows = value.get(field_name)
        if not isinstance(rows, list):
            continue
        texts = [
            _text(row.get("text"))
            for row in rows
            if isinstance(row, Mapping) and _text(row.get("text"))
        ]
        if texts:
            fragments.append(f"{label}: {', '.join(texts)}")
    obligations = value.get("interaction_obligations")
    if isinstance(obligations, list):
        obligation_texts = []
        for obligation in obligations:
            if not isinstance(obligation, Mapping):
                continue
            actor = _text(obligation.get("actor"))
            action = _text(obligation.get("action"))
            if not actor or not action:
                continue
            details = [f"actor={actor}", f"action={action}"]
            for field_name in (
                "beneficiary",
                "precondition",
                "expected_outcome",
                "status",
                "source_kind",
                "age_hint",
            ):
                detail = _text(obligation.get(field_name))
                if detail:
                    details.append(f"{field_name}={detail}")
            obligation_texts.append(", ".join(details))
        if obligation_texts:
            fragments.append(
                f"interaction obligations: {' | '.join(obligation_texts)}"
            )
    return "; ".join(fragments)


def _v2_timestamp(value: str) -> str:
    """Project the adapter timestamp into the native V2 UTC-Z contract."""

    parsed = parse_storage_utc_datetime(value).astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(value: object) -> str:
    """Return bounded connector text."""

    return value.strip() if isinstance(value, str) else ""


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
