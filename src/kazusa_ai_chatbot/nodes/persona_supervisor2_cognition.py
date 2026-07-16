"""Canonical upstream connector for the V2 cognition core."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import timezone
from typing import Any

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
    build_initial_action_capabilities,
)
from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.results import project_trace_action_result_v2
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
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeValidationError,
    validate_cognitive_episode,
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
        validate_cognitive_episode(episode)
    except CognitiveEpisodeValidationError as exc:
        raise CognitionExecutionError(str(exc)) from exc
    timestamp = _v2_timestamp(episode["storage_timestamp_utc"])
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
    evidence.extend(_resolver_observation_evidence(
        state.get("resolver_observations"),
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
        episode["target_scope"]["channel_type"],
        episode.get("trigger_source"),
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
        "private_continuity_context": _text(
            state.get("internal_monologue_residue_context")
        )[:1000],
        "scene_context": {
            "channel_scope": channel_scope,
            "character_role": "character",
            "current_user_role": "participant",
            "semantic_scene": semantic_text[:500],
            "conversation_continuity": _conversation_progress_text(
                state.get("conversation_progress")
            )[:1000],
            "semantic_temporal_context": "immediate",
        },
    }
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
        "resolver_capability_requests": output["resolver_requests"],
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
        decision = output["intention"]["route"]
        if request["action_kind"] == ACCEPTED_CODING_TASK_REQUEST_CAPABILITY:
            decision = "start"
        requests.append({
            "capability": request["action_kind"],
            "decision": decision,
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
        "accepted_task_result_ready": "accepted_task_result",
        "scheduled_recall": "scheduler_event",
        "reflection_signal": "promoted_reflection",
    }.get(trigger_source, "episode")
    source_id = f"episode:{episode_id}"
    semantic_text = fallback_text
    percepts = episode.get("percepts")
    if isinstance(percepts, list):
        for percept in percepts:
            if not isinstance(percept, Mapping):
                continue
            if percept.get("visibility") != "model_visible":
                continue
            content = _text(percept.get("content"))
            metadata = percept.get("metadata")
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
            if source_kind == "accepted_task_result" and isinstance(
                metadata,
                Mapping,
            ):
                if not isinstance(cognition_source, Mapping):
                    source_id = (
                        _text(metadata.get("accepted_task_id")) or source_id
                    )
                    semantic_text = _accepted_task_result_text(
                        metadata,
                        content,
                        fallback_text,
                    )
            elif content and not isinstance(cognition_source, Mapping):
                semantic_text = content
                source_id = _text(percept.get("percept_id")) or source_id
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


def _accepted_task_result_text(
    metadata: Mapping[str, Any],
    content: str,
    fallback_text: str,
) -> str:
    """Build bounded semantic evidence from an accepted-task outcome."""

    parts = [
        _text(metadata.get("accepted_task_summary")),
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
        "accepted_task_result_ready": "accepted_task_result_ready",
        "background_result": "background_result",
        "group_message": "group_sender",
        "reflection_signal": "reflection",
        "reflection_dry_run": "reflection_dry_run",
        "internal_thought": "internal_thought_cognition",
        "scheduled_recall": "recall",
        "system_probe": "system_probe",
        "resolver_recurrence": "resolver_recurrence",
    }.get(str(trigger), "persona_user_message")


def _is_self_cognition_episode(episode: Mapping[str, Any]) -> bool:
    """Identify the canonical self-cognition packet without parsing its prose."""

    percepts = episode.get("percepts")
    if not isinstance(percepts, list):
        return False
    for percept in percepts:
        if not isinstance(percept, Mapping):
            continue
        metadata = percept.get("metadata")
        if (
            isinstance(metadata, Mapping)
            and metadata.get("source") == "self_cognition_source_packet"
        ):
            return True
    return False


def _scene_channel_scope(
    channel_type: object,
    trigger_source: object,
) -> str:
    """Select the semantic scene scope from the typed episode source."""

    if trigger_source in {
        "reflection_signal",
        "internal_thought",
        "system_probe",
    }:
        return "internal"
    if channel_type in {"dm", "private"}:
        return "private"
    return "group"
