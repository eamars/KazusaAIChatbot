"""Orchestrate one self-cognition dry-run case."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    validate_cognitive_episode,
)
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_agent
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator import (
    call_consolidation_subgraph,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    call_cognition_subgraph,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_supervisor2 import (
    call_rag_supervisor,
)
from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)
from kazusa_ai_chatbot.self_cognition import artifacts, models, projection, tracking
from kazusa_ai_chatbot.time_context import build_character_time_context


SelfCognitionClient = Callable[[dict[str, Any]], Any]
RagClient = Callable[..., Any]
ConsolidationBuildResult = tuple[dict[str, Any], dict[str, Any], bool]


def run_self_cognition_case(
    case: models.SelfCognitionCase,
    output_dir: str | Path,
    rag_client: RagClient | None = None,
    cognition_client: SelfCognitionClient | None = None,
    dialog_client: SelfCognitionClient | None = None,
    consolidation_client: SelfCognitionClient | None = None,
    *,
    apply_consolidation: bool = False,
    event_log_mirror: bool = False,
) -> dict[str, str]:
    """Run one dry-run case and write local tracking artifacts.

    Args:
        case: External self-cognition case file data.
        output_dir: Local output directory for artifacts.
        rag_client: Optional test seam for the RAG2 supervisor.
        cognition_client: Optional test seam for the shared cognition graph.
        dialog_client: Optional test seam for private finalization.
        consolidation_client: Optional test seam for the shared consolidator.
        apply_consolidation: When true, build private finalization and call the
            shared consolidation seam.
        event_log_mirror: When true, mirror sanitized artifact metadata through
            the public event-logging interface.

    Returns:
        Artifact names mapped to written paths.
    """

    written_paths = asyncio.run(
        run_self_cognition_case_async(
            case,
            output_dir,
            rag_client=rag_client,
            cognition_client=cognition_client,
            dialog_client=dialog_client,
            consolidation_client=consolidation_client,
            apply_consolidation=apply_consolidation,
            event_log_mirror=event_log_mirror,
        )
    )
    return written_paths


def build_self_cognition_case_artifacts(
    case: models.SelfCognitionCase,
    rag_client: RagClient | None = None,
    cognition_client: SelfCognitionClient | None = None,
    dialog_client: SelfCognitionClient | None = None,
    consolidation_client: SelfCognitionClient | None = None,
    *,
    apply_consolidation: bool = False,
) -> dict[str, Any]:
    """Build one self-cognition case's tracking records in memory.

    Args:
        case: External self-cognition case data.
        rag_client: Optional test seam for the RAG2 supervisor.
        cognition_client: Optional test seam for the shared cognition graph.
        dialog_client: Optional test seam for private finalization.
        consolidation_client: Optional test seam for the shared consolidator.
        apply_consolidation: When true, build private finalization and call the
            shared consolidation seam.

    Returns:
        Artifact names mapped to JSON-like payloads or Markdown text.
    """

    artifact_payloads = asyncio.run(
        build_self_cognition_case_artifacts_async(
            case,
            rag_client=rag_client,
            cognition_client=cognition_client,
            dialog_client=dialog_client,
            consolidation_client=consolidation_client,
            apply_consolidation=apply_consolidation,
        )
    )
    return artifact_payloads


async def run_self_cognition_case_async(
    case: models.SelfCognitionCase,
    output_dir: str | Path,
    rag_client: RagClient | None = None,
    cognition_client: SelfCognitionClient | None = None,
    dialog_client: SelfCognitionClient | None = None,
    consolidation_client: SelfCognitionClient | None = None,
    *,
    apply_consolidation: bool = False,
    event_log_mirror: bool = False,
) -> dict[str, str]:
    """Async implementation for one self-cognition dry-run case.

    Args:
        case: External self-cognition case file data.
        output_dir: Local output directory for artifacts.
        rag_client: Optional test seam for the RAG2 supervisor.
        cognition_client: Optional test seam for the shared cognition graph.
        dialog_client: Optional test seam for private finalization.
        consolidation_client: Optional test seam for the shared consolidator.
        apply_consolidation: When true, build private finalization and call the
            shared consolidation seam.
        event_log_mirror: When true, mirror sanitized artifact metadata through
            the public event-logging interface.

    Returns:
        Artifact names mapped to written paths.
    """

    artifact_payloads = await build_self_cognition_case_artifacts_async(
        case,
        rag_client=rag_client,
        cognition_client=cognition_client,
        dialog_client=dialog_client,
        consolidation_client=consolidation_client,
        apply_consolidation=apply_consolidation,
    )
    written_paths = artifacts.write_tracking_artifacts(
        output_dir,
        artifact_payloads,
    )
    if event_log_mirror:
        await _record_self_cognition_event_from_artifacts(
            case=case,
            artifact_payloads=artifact_payloads,
            dispatch_status="not_requested",
            component="self_cognition.runner",
        )
    return written_paths


async def build_self_cognition_case_artifacts_async(
    case: models.SelfCognitionCase,
    rag_client: RagClient | None = None,
    cognition_client: SelfCognitionClient | None = None,
    dialog_client: SelfCognitionClient | None = None,
    consolidation_client: SelfCognitionClient | None = None,
    *,
    apply_consolidation: bool = False,
) -> dict[str, Any]:
    """Async implementation for building self-cognition records in memory.

    Args:
        case: External self-cognition case data.
        rag_client: Optional test seam for the RAG2 supervisor.
        cognition_client: Optional test seam for the shared cognition graph.
        dialog_client: Optional test seam for private finalization.
        consolidation_client: Optional test seam for the shared consolidator.
        apply_consolidation: When true, build private finalization and call the
            shared consolidation seam.

    Returns:
        Artifact names mapped to JSON-like payloads or Markdown text.
    """

    case_name = projection.validate_case_name(case)
    trigger_record = tracking.build_trigger_record(case)
    artifact_payloads: dict[str, Any] = {
        models.ARTIFACT_TRIGGER_RECORD: trigger_record,
    }

    if case_name == models.CASE_GROUP_NOISE_REJECTED:
        selected_route = models.ROUTE_AUDIT_ONLY
        budget = _budget(rag_calls=0, cognition_calls=0, dialog_calls=0)
        run_record = tracking.build_run_record(
            case,
            trigger_record,
            selected_route,
            budget,
        )
        route_effect = _route_effect_for_route(run_record, selected_route)
        artifact_payloads[models.ARTIFACT_RUN_RECORD] = run_record
        artifact_payloads[models.ARTIFACT_ROUTE_EFFECT] = route_effect
        artifact_payloads[models.ARTIFACT_LOOP_TRACE] = _loop_trace(
            case,
            run_record,
            route_effect,
        )
        return artifact_payloads

    rag_output: dict[str, Any] | None = None
    rag_calls = 0
    if case_name == models.CASE_TOPIC_RAG_FOLLOWUP:
        rag_request = projection.build_rag_request(case)
        active_rag_client = rag_client or _default_rag_client
        rag_output = await _call_maybe_async(
            active_rag_client,
            rag_request["query"],
            character_name=_character_name(case),
            context=rag_request["context"],
        )
        rag_calls = models.RAG_SUPERVISOR_INVOCATION_LIMIT
        artifact_payloads[models.ARTIFACT_RAG_REQUEST] = rag_request
        artifact_payloads[models.ARTIFACT_RAG_OUTPUT] = rag_output

    source_packet = projection.build_source_packet(case, rag_output=rag_output)
    rendered_packet = projection.render_source_packet_text(source_packet)
    cognition_input = {
        "source_packet": source_packet,
        "rendered_text": rendered_packet,
    }
    artifact_payloads[models.ARTIFACT_COGNITION_INPUT] = cognition_input

    active_cognition_client = cognition_client or _default_cognition_client
    cognition_state = _build_cognition_state(case, rendered_packet)
    cognition_output = await _call_maybe_async(
        active_cognition_client,
        cognition_state,
    )
    artifact_payloads[models.ARTIFACT_COGNITION_OUTPUT] = cognition_output

    existing_attempts = _existing_attempts(case)
    selected_route = tracking.classify_route(case, cognition_output)
    action_attempt = None
    action_candidate = None
    dialog_output: dict[str, Any] | None = None
    dialog_calls = 0
    active_dialog_client = dialog_client or _default_dialog_client
    if selected_route == models.ROUTE_ACTION_CANDIDATE:
        action_attempt = tracking.build_action_attempt(
            case,
            trigger_record,
            existing_attempts,
        )
        selected_route = tracking.classify_route(
            case,
            cognition_output,
            action_attempt=action_attempt,
        )
        if action_attempt["status"] == models.ACTION_ATTEMPT_STATUS_CANDIDATE:
            action_text = tracking.extract_action_candidate_text(cognition_output)
            if not action_text:
                dialog_state = _build_dialog_state(
                    cognition_state,
                    cognition_output,
                )
                dialog_output = await _call_maybe_async(
                    active_dialog_client,
                    dialog_state,
                )
                dialog_calls = models.DIALOG_RENDER_CALL_LIMIT
                action_text = _dialog_text(dialog_output)
            action_candidate = tracking.build_action_candidate(
                case,
                action_attempt,
                action_text,
                mention_target_user=(
                    bool(dialog_output.get("mention_target_user", False))
                    if dialog_output is not None
                    else False
                ),
            )
        artifact_payloads[models.ARTIFACT_ACTION_ATTEMPT] = action_attempt
        if action_candidate is not None:
            artifact_payloads[models.ARTIFACT_ACTION_CANDIDATE] = (
                action_candidate
            )

    if apply_consolidation:
        consolidation_state, dialog_output, dialog_called = (
            await _build_consolidation_ready_state(
                cognition_state,
                cognition_output,
                rendered_packet,
                dialog_client=active_dialog_client,
                dialog_output=dialog_output,
            )
        )
        if dialog_called:
            dialog_calls += models.DIALOG_RENDER_CALL_LIMIT
        active_consolidation_client = (
            consolidation_client or _default_consolidation_client
        )
        consolidation_result = await _call_maybe_async(
            active_consolidation_client,
            consolidation_state,
        )
        artifact_payloads[models.ARTIFACT_CONSOLIDATION_OUTCOME] = (
            tracking.build_consolidation_outcome_record(
                consolidation_state,
                consolidation_result,
            )
        )

    budget = _budget(
        rag_calls=rag_calls,
        cognition_calls=1,
        dialog_calls=dialog_calls,
    )
    run_record = tracking.build_run_record(
        case,
        trigger_record,
        selected_route,
        budget,
    )
    route_effect = _route_effect_for_route(run_record, selected_route)
    artifact_payloads[models.ARTIFACT_RUN_RECORD] = run_record
    artifact_payloads[models.ARTIFACT_ROUTE_EFFECT] = route_effect
    artifact_payloads[models.ARTIFACT_LOOP_TRACE] = _loop_trace(
        case,
        run_record,
        route_effect,
        action_attempt=action_attempt,
        action_candidate=action_candidate,
    )
    return artifact_payloads


async def _record_self_cognition_event_from_artifacts(
    *,
    case: models.SelfCognitionCase,
    artifact_payloads: dict[str, Any],
    dispatch_status: str,
    component: str,
) -> None:
    """Mirror built tracking artifacts into the event log without raw text."""

    trigger_record = artifact_payloads.get(models.ARTIFACT_TRIGGER_RECORD)
    run_record = artifact_payloads.get(models.ARTIFACT_RUN_RECORD)
    action_attempt = artifact_payloads.get(models.ARTIFACT_ACTION_ATTEMPT)
    if not isinstance(trigger_record, dict) or not isinstance(run_record, dict):
        return
    if not isinstance(action_attempt, dict):
        action_attempt = {}
    consolidation_outcome = artifact_payloads.get(
        models.ARTIFACT_CONSOLIDATION_OUTCOME
    )
    if not isinstance(consolidation_outcome, dict):
        consolidation_outcome = None
    budget = run_record["budget"]
    await event_logging.record_self_cognition_event(
        component=component,
        case_id=_string_field(case, "case_id"),
        trigger_kind=str(trigger_record["trigger_kind"]),
        selected_route=str(run_record["selected_route"]),
        output_mode=str(run_record["output_mode"]),
        budget={
            "rag_calls": int(budget["rag_calls"]),
            "cognition_calls": int(budget["cognition_calls"]),
            "dialog_calls": int(budget["dialog_calls"]),
            "topic_limit": int(budget["topic_limit"]),
        },
        dispatch_status=dispatch_status,
        status=str(run_record["status"]),
        trigger_id=str(trigger_record["trigger_id"]),
        run_id=str(run_record["run_id"]),
        attempt_id=str(action_attempt.get("attempt_id") or ""),
        consolidation_outcome=consolidation_outcome,
    )


async def _default_rag_client(
    query: str,
    *,
    character_name: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Call the existing RAG2 supervisor for one bounded query.

    Args:
        query: Natural-language retrieval question.
        character_name: Runtime character name.
        context: Platform and channel context for RAG.

    Returns:
        RAG2 supervisor result.
    """

    rag_result = await call_rag_supervisor(
        query,
        character_name=character_name,
        context=context,
    )
    return rag_result


async def _default_cognition_client(state: dict[str, Any]) -> dict[str, Any]:
    """Call the existing L1/L2/L3 cognition graph.

    Args:
        state: Global persona state subset required by the cognition graph.

    Returns:
        Shared cognition output.
    """

    cognition_result = await call_cognition_subgraph(state)
    return cognition_result


async def _default_dialog_client(state: dict[str, Any]) -> dict[str, Any]:
    """Call the existing dialog generator/evaluator graph.

    Args:
        state: Global persona state merged with shared cognition output.

    Returns:
        Dialog graph result used only as a local dry-run candidate.
    """

    dialog_result = await dialog_agent(state)
    return dialog_result


async def _default_consolidation_client(state: dict[str, Any]) -> dict[str, Any]:
    """Call the existing post-dialog consolidator subgraph.

    Args:
        state: Self-cognition state after shared cognition and private
            finalization.

    Returns:
        Shared consolidator result with write metadata.
    """

    consolidation_result = await call_consolidation_subgraph(state)
    return consolidation_result


async def _call_maybe_async(
    callable_object: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call a sync or async test seam with a common awaitable contract."""

    result = callable_object(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


async def _build_consolidation_ready_state(
    cognition_state: dict[str, Any],
    cognition_output: dict[str, Any],
    rendered_packet: str,
    *,
    dialog_client: SelfCognitionClient,
    dialog_output: dict[str, Any] | None,
) -> ConsolidationBuildResult:
    """Build private-finalization state for same-path consolidation.

    Args:
        cognition_state: State originally sent into shared cognition.
        cognition_output: Shared cognition graph output.
        rendered_packet: Internal-thought evidence text used as the
            decontextualized consolidation input.
        dialog_client: Existing dialog/finalization seam.
        dialog_output: Previously rendered dialog output, if the action route
            already needed it.

    Returns:
        Consolidation-ready state, dialog output, and whether a new dialog call
        was needed.
    """

    dialog_called = False
    active_dialog_output = dialog_output
    if active_dialog_output is None:
        dialog_state = _build_dialog_state(cognition_state, cognition_output)
        active_dialog_output = await _call_maybe_async(
            dialog_client,
            dialog_state,
        )
        dialog_called = True

    consolidation_state = _build_consolidation_state(
        cognition_state,
        cognition_output,
        active_dialog_output,
        rendered_packet,
    )
    return_value = (consolidation_state, active_dialog_output, dialog_called)
    return return_value


def _build_consolidation_state(
    cognition_state: dict[str, Any],
    cognition_output: dict[str, Any],
    dialog_output: dict[str, Any],
    rendered_packet: str,
) -> dict[str, Any]:
    """Merge cognition and private finalization for the consolidator."""

    consolidation_state = dict(cognition_state)
    consolidation_state.update(cognition_output)
    consolidation_state.update(dialog_output)
    consolidation_state["decontexualized_input"] = rendered_packet
    consolidation_state["final_dialog"] = _dialog_fragments(dialog_output)
    return consolidation_state


def _build_cognition_state(
    case: models.SelfCognitionCase,
    rendered_packet: str,
) -> dict[str, Any]:
    """Build the shared cognition graph state for an idle dry run."""

    timestamp = _string_field(case, "idle_timestamp")
    time_context = build_character_time_context(timestamp)
    target_scope = _target_scope(case)
    chat_history = _chat_history(case, target_scope)
    episode = _build_cognitive_episode(
        case,
        rendered_packet,
        time_context=time_context,
    )
    user_id = target_scope["user_id"] or "self_cognition_target"
    state = {
        "character_profile": _character_profile(case),
        "timestamp": timestamp,
        "time_context": time_context,
        "user_input": models.SELF_COGNITION_INPUT_TEXT,
        "prompt_message_context": {
            "body_text": models.SELF_COGNITION_INPUT_TEXT,
            "addressed_to_global_user_ids": [],
            "broadcast": target_scope["channel_type"] == "group",
            "mentions": [],
            "attachments": [],
        },
        "cognitive_episode": episode,
        "user_multimedia_input": [],
        "platform": target_scope["platform"],
        "platform_channel_id": target_scope["platform_channel_id"],
        "channel_type": target_scope["channel_type"],
        "platform_message_id": f"self_cognition:{_string_field(case, 'case_id')}",
        "platform_user_id": user_id,
        "global_user_id": user_id,
        "user_name": _user_display_name(case, user_id),
        "user_profile": _user_profile(case),
        "platform_bot_id": _platform_bot_id(case),
        "chat_history_wide": chat_history,
        "chat_history_recent": chat_history,
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": _string_field(case, "channel_topic"),
        "conversation_progress": case.get("conversation_progress"),
        "promoted_reflection_context": case.get("promoted_reflection_context"),
        "debug_modes": {"no_visual_directives": True},
        "should_respond": False,
        "decontexualized_input": models.SELF_COGNITION_INPUT_TEXT,
        "referents": [],
        "rag_result": _rag_result(case),
        "internal_monologue": "",
        "action_directives": {},
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "character_intent": "",
        "logical_stance": "",
        "mood": "",
        "global_vibe": "",
        "reflection_summary": "",
        "subjective_appraisals": [],
        "affinity_delta": 0,
        "last_relationship_insight": "",
        "new_facts": [],
        "future_promises": [],
        "mention_target_user": False,
    }
    return state


def _build_dialog_state(
    cognition_state: dict[str, Any],
    cognition_output: dict[str, Any],
) -> dict[str, Any]:
    """Merge cognition output into the dialog graph's input state."""

    dialog_state = dict(cognition_state)
    dialog_state.update(cognition_output)
    dialog_state["final_dialog"] = []
    dialog_state["mention_target_user"] = False
    return dialog_state


def _dialog_text(dialog_output: dict[str, Any]) -> str:
    """Extract dry-run candidate text from dialog graph output."""

    fragments = _dialog_fragments(dialog_output)
    text = "\n".join(fragments)
    return text


def _dialog_fragments(dialog_output: dict[str, Any]) -> list[str]:
    """Extract normalized final-dialog fragments from dialog output."""

    value = dialog_output.get("final_dialog")
    if not isinstance(value, list):
        return_value: list[str] = []
        return return_value
    fragments = [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]
    return_value = fragments
    return return_value


def _build_cognitive_episode(
    case: models.SelfCognitionCase,
    rendered_packet: str,
    *,
    time_context: dict[str, str],
) -> CognitiveEpisode:
    """Represent the self-cognition source packet as an internal percept."""

    timestamp = _string_field(case, "idle_timestamp")
    target_scope = _target_scope(case)
    user_id = target_scope["user_id"] or "self_cognition_target"
    percept_content = json.dumps(
        {
            "residue": {
                "residue_id": f"self_cognition:{_string_field(case, 'case_id')}",
                "internal_monologue": rendered_packet,
            },
            "action_latch": {
                "status": "local_tracking",
                "outward_action": "allowed_to_be_considered",
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    episode: CognitiveEpisode = {
        "episode_id": f"self_cognition:dry_run:{_string_field(case, 'case_id')}",
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": "preview",
        "percepts": [
            {
                "percept_id": "self_cognition:source_packet",
                "input_source": "internal_monologue",
                "content": percept_content,
                "visibility": "model_visible",
                "metadata": {"source": "self_cognition_source_packet"},
            },
        ],
        "target_scope": {
            "platform": target_scope["platform"],
            "platform_channel_id": target_scope["platform_channel_id"],
            "channel_type": target_scope["channel_type"],
            "current_platform_user_id": user_id,
            "current_global_user_id": user_id,
            "current_display_name": _user_display_name(case, user_id),
            "target_addressed_user_ids": [user_id] if user_id else [],
            "target_broadcast": target_scope["channel_type"] == "group",
        },
        "origin_metadata": {
            "platform": target_scope["platform"],
            "platform_message_id": f"self_cognition:{_string_field(case, 'case_id')}",
            "active_turn_platform_message_ids": [],
            "active_turn_conversation_row_ids": [],
            "debug_modes": {"no_visual_directives": True},
        },
        "timestamp": timestamp,
        "time_context": time_context,
    }
    validate_cognitive_episode(episode)
    return episode


def _route_effect_for_route(
    run_record: dict[str, Any],
    route: str,
) -> dict[str, Any]:
    """Build the dry-run consumer effect for one selected route."""

    if route == models.ROUTE_ACTION_CANDIDATE:
        consumer = "local_action_candidate"
        effect_summary = (
            "Runner created or suppressed a send-message candidate; worker "
            "handoff is recorded separately when enabled."
        )
    elif route == models.ROUTE_PROGRESS_MAINTENANCE:
        consumer = "conversation_progress_candidate"
        effect_summary = (
            "Dry-run would keep conversation progress visible; no production "
            "write was performed."
        )
    else:
        consumer = "audit_log"
        effect_summary = (
            "Dry-run recorded the observation only; no production write was "
            "performed."
        )
    route_effect = tracking.build_route_effect(
        run_record,
        route,
        consumer,
        effect_summary,
        next_topic=models.EMPTY_ROUTE_EFFECT_NEXT_TOPIC,
    )
    return route_effect


def _loop_trace(
    case: models.SelfCognitionCase,
    run_record: dict[str, Any],
    route_effect: dict[str, Any],
    *,
    action_attempt: dict[str, Any] | None = None,
    action_candidate: dict[str, Any] | None = None,
) -> str:
    """Render a human-readable trace of the dry-run routing decision."""

    lines = [
        "# Self-Cognition Dry-Run Trace",
        "",
        f"- case_name: {_string_field(case, 'case_name')}",
        f"- trigger_id: {run_record['trigger_id']}",
        f"- run_id: {run_record['run_id']}",
        f"- selected_route: {run_record['selected_route']}",
        f"- consumer: {route_effect['consumer']}",
        f"- production_write: {route_effect['production_write']}",
    ]
    if action_attempt is not None:
        lines.append(f"- action_attempt_status: {action_attempt['status']}")
    if action_candidate is not None:
        lines.append("- action_candidate_written: true")
    else:
        lines.append("- action_candidate_written: false")
    trace = "\n".join(lines)
    return trace


def _budget(
    rag_calls: int,
    cognition_calls: int,
    dialog_calls: int,
) -> dict[str, int]:
    """Build local budget counters for the run record."""

    budget = {
        "rag_calls": rag_calls,
        "cognition_calls": cognition_calls,
        "dialog_calls": dialog_calls,
        "topic_limit": models.TOPIC_LIMIT,
    }
    return budget


def _rag_result(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Return supplied RAG context or the existing empty RAG projection."""

    rag_output = case.get("rag_output")
    if isinstance(rag_output, dict):
        return_value = rag_output
        return return_value
    return_value = {
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
            "loop_count": 0,
            "unknown_slots": [],
            "dispatched": [],
        },
    }
    return return_value


def _character_profile(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Return the supplied character profile or a dry-run default profile."""

    value = case.get("character_profile")
    if isinstance(value, dict) and value:
        return_value = value
        return return_value

    profile = {
        "name": "active character",
        "mood": _string_field(case, "current_mood") or "neutral",
        "global_vibe": _string_field(case, "global_vibe") or "neutral",
        "reflection_summary": "",
        "personality_brief": {
            "mbti": "INFP",
            "logic": "relationship-aware and careful",
            "tempo": "measured",
            "defense": "soft boundary preservation",
            "quirks": "uses concise emotional cues",
            "taboos": "does not invent facts",
        },
        "boundary_profile": {
            "self_integrity": 0.5,
            "control_sensitivity": 0.5,
            "relational_override": 0.5,
            "control_intimacy_misread": 0.5,
            "authority_skepticism": 0.5,
            "compliance_strategy": "evade",
            "boundary_recovery": "rebound",
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.5,
            "hesitation_density": 0.5,
            "counter_questioning": 0.5,
            "softener_density": 0.5,
            "formalism_avoidance": 0.5,
            "abstraction_reframing": 0.5,
            "direct_assertion": 0.5,
            "emotional_leakage": 0.5,
            "rhythmic_bounce": 0.5,
            "self_deprecation": 0.5,
        },
    }
    return profile


def _user_profile(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Return the supplied user profile or a dry-run default profile."""

    value = case.get("user_profile")
    if isinstance(value, dict) and value:
        return_value = value
        return return_value

    profile = {
        "affinity": models.DEFAULT_DRY_RUN_AFFINITY,
        "display_name": _target_scope(case)["user_id"] or "self cognition target",
        "last_relationship_insight": "",
    }
    return profile


def _target_scope(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Normalize the case target scope for graph state fields."""

    value = case.get("target_scope")
    if not isinstance(value, dict):
        value = {}
    platform = value.get("platform")
    platform_channel_id = value.get("platform_channel_id")
    channel_type = value.get("channel_type")
    user_id = value.get("user_id")
    scope = {
        "platform": platform if isinstance(platform, str) else "",
        "platform_channel_id": (
            platform_channel_id if isinstance(platform_channel_id, str) else ""
        ),
        "channel_type": channel_type if isinstance(channel_type, str) else "",
        "user_id": user_id if isinstance(user_id, str) else None,
    }
    return scope


def _existing_attempts(case: models.SelfCognitionCase) -> list[dict[str, Any]]:
    """Copy prior local action attempts supplied by the case file."""

    value = case.get("existing_attempts")
    if not isinstance(value, list):
        return_value: list[dict[str, Any]] = []
        return return_value
    attempts = [
        dict(item)
        for item in value
        if isinstance(item, dict)
    ]
    return attempts


def _chat_history(
    case: models.SelfCognitionCase,
    target_scope: dict[str, Any],
) -> list[dict[str, Any]]:
    """Project visible context rows into the shared chat-history shape."""

    value = case.get("visible_context")
    if not isinstance(value, list):
        return_value: list[dict[str, Any]] = []
        return return_value

    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = _string_field(item, "role")
        body_text = _string_field(item, "body_text")
        if not body_text:
            body_text = _string_field(item, "text")
        if not role or not body_text:
            continue
        global_user_id = target_scope["user_id"] or ""
        if role == "assistant":
            global_user_id = models.DRY_RUN_ASSISTANT_GLOBAL_USER_ID
        row = {
            "timestamp": _string_field(item, "timestamp"),
            "role": role,
            "platform_user_id": global_user_id,
            "global_user_id": global_user_id,
            "display_name": _string_field(item, "display_name"),
            "body_text": body_text,
            "addressed_to_global_user_ids": [target_scope["user_id"]]
            if role == "assistant" and target_scope["user_id"]
            else [],
            "broadcast": False,
        }
        rows.append(row)
    return rows


def _character_name(case: models.SelfCognitionCase) -> str:
    """Read the active character name from the dry-run profile."""

    profile = _character_profile(case)
    name = profile.get("name")
    if not isinstance(name, str):
        return_value = ""
        return return_value
    return_value = name
    return return_value


def _user_display_name(
    case: models.SelfCognitionCase,
    fallback_user_id: str,
) -> str:
    """Read the target display name with a stable fallback."""

    user_profile = _user_profile(case)
    display_name = user_profile.get("display_name")
    if isinstance(display_name, str) and display_name:
        return_value = display_name
    else:
        return_value = fallback_user_id
    return return_value


def _platform_bot_id(case: models.SelfCognitionCase) -> str:
    """Read the platform bot id used by the dialog graph."""

    value = case.get("platform_bot_id")
    if isinstance(value, str) and value:
        return_value = value
    else:
        return_value = "self_cognition_bot"
    return return_value


def _string_field(case: dict[str, Any], field_name: str) -> str:
    """Read an optional external string field safely."""

    value = case.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value
    return return_value
