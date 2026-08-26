"""Closed Brain-owned projection for live and self cognition observations."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from kazusa_ai_chatbot.action_spec.results import EpisodeTerminalStatusV1
from kazusa_ai_chatbot.brain_service.cognition_observation_contracts import (
    COGNITION_OBSERVATION_DISCLOSURE_POLICY,
    COGNITION_OBSERVATION_EXCLUSIONS,
    CognitionObservationCorrelationV1,
    CognitionObservationDisclosureV1,
    CognitionObservationEdgeV1,
    CognitionObservationFieldV1,
    CognitionObservationNodeV1,
    CognitionObservationRecordV1,
    CognitionObservationSectionV1,
    CognitionRunObservationV1,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    validate_shared_memory_prewarm_outcome,
)
from kazusa_ai_chatbot.config import COGNITION_VISUAL_DIRECTIVES_ENABLED
from kazusa_ai_chatbot.self_cognition import models as self_cognition_models

_MISSING = object()
_MAX_RECORDS = 24
_MAX_LIST_ITEMS = 24
_MAX_TEXT = 4000
_MAX_LIST_TEXT = 2000
_VALID_SECTION_STATUSES = {
    "completed",
    "empty",
    "skipped",
    "failed",
    "partial",
    "not_reported",
}
_STATUS_PRIORITY = (
    "failed",
    "partial",
    "completed",
    "empty",
    "skipped",
    "not_reported",
)
_VISUAL_KINDS = (
    "facial_expression",
    "body_language",
    "gaze_direction",
    "visual_vibe",
)
_EVIDENCE_FIELDS = (
    "summary",
    "fact",
    "excerpt",
    "content",
    "title",
    "relevance",
    "recency",
    "due_state",
    "evidence_boundary_notes",
)
_PROTECTED_FIELDS = {
    "prompt",
    "raw_model_output",
    "embedding",
    "raw_message",
    "message_envelope",
    "database_identifier",
    "adapter_identifier",
    "action_parameter",
    "handler_metadata",
    "worker_error_text",
}

_SECTION_META: dict[str, tuple[str, str, str]] = {
    "input.turn": ("Queued turn", "input", "fields"),
    "decision.response": ("Response decision", "decision", "fields"),
    "cognition.appraisals": (
        "Semantic appraisals",
        "appraisal",
        "records",
    ),
    "cognition.goal": ("Character goal", "goal", "fields"),
    "cognition.response_plan": ("Response plan", "response", "fields"),
    "cognition.affect": ("Affect projection", "affect", "records"),
    "reasoning.subjective": (
        "Subjective reasoning",
        "reasoning",
        "fields",
    ),
    "reasoning.context_consumption": (
        "Context consumption",
        "context",
        "records",
    ),
    "evidence.memory": ("Memory evidence", "memory", "records"),
    "evidence.shared_memory_prewarm": (
        "Shared-memory prewarm",
        "prewarm",
        "records",
    ),
    "context.conversation_progress": (
        "Conversation progress",
        "progress",
        "records",
    ),
    "context.public_group_scene": (
        "Public group scene",
        "group_scene",
        "records",
    ),
    "action.requests": ("Action requests", "action", "records"),
    "action.results": ("Action results", "action", "records"),
    "action.continuation": (
        "Action continuation",
        "continuation",
        "records",
    ),
    "surface.visual_directives": (
        "Visual directives",
        "visual",
        "records",
    ),
    "surface.visible_messages": (
        "Visible messages",
        "dialog",
        "records",
    ),
    "self.source": ("Self-cognition source", "source", "fields"),
    "self.route": ("Self-cognition route", "route", "fields"),
    "self.consolidation": (
        "Self-cognition consolidation",
        "continuity",
        "fields",
    ),
}

_LIVE_SECTION_IDS = (
    "input.turn",
    "decision.response",
    "cognition.appraisals",
    "cognition.goal",
    "cognition.response_plan",
    "cognition.affect",
    "reasoning.subjective",
    "reasoning.context_consumption",
    "evidence.memory",
    "evidence.shared_memory_prewarm",
    "context.conversation_progress",
    "context.public_group_scene",
    "action.requests",
    "action.results",
    "action.continuation",
    "surface.visual_directives",
    "surface.visible_messages",
)
_SELF_SECTION_IDS = (
    "cognition.appraisals",
    "cognition.goal",
    "cognition.response_plan",
    "cognition.affect",
    "reasoning.subjective",
    "reasoning.context_consumption",
    "evidence.memory",
    "evidence.shared_memory_prewarm",
    "context.conversation_progress",
    "context.public_group_scene",
    "action.requests",
    "action.results",
    "action.continuation",
    "surface.visual_directives",
    "surface.visible_messages",
    "self.source",
    "self.route",
    "self.consolidation",
)

_LIVE_NODE_DEFS = (
    (
        "input.turn",
        "Queued turn",
        "Input",
        "input",
        1,
        "input",
        ("input.turn",),
    ),
    (
        "decision.response",
        "Response decision",
        "Decision",
        "gate",
        2,
        "decision",
        ("decision.response",),
    ),
    (
        "cognition.meaning",
        "Meaning appraisal",
        "Cognition",
        "cognition",
        3,
        "appraisal",
        ("cognition.appraisals",),
    ),
    (
        "cognition.goal",
        "Character goal",
        "Cognition",
        "cognition",
        3,
        "goal",
        ("cognition.goal",),
    ),
    (
        "cognition.response",
        "Response plan",
        "Cognition",
        "cognition",
        3,
        "response",
        ("cognition.response_plan",),
    ),
    (
        "cognition.affect",
        "Affect projection",
        "Cognition",
        "cognition",
        3,
        "affect",
        ("cognition.affect",),
    ),
    (
        "reasoning.context",
        "Reasoning and context",
        "Reasoning",
        "cognition",
        3,
        "reasoning",
        ("reasoning.subjective", "reasoning.context_consumption"),
    ),
    (
        "evidence.memory",
        "Memory and context",
        "Evidence",
        "memory",
        3,
        "memory",
        (
            "evidence.shared_memory_prewarm",
            "evidence.memory",
            "context.conversation_progress",
            "context.public_group_scene",
        ),
    ),
    (
        "action.results",
        "Actions",
        "Actions",
        "action",
        3,
        "action",
        ("action.requests", "action.results", "action.continuation"),
    ),
    (
        "surface.visual",
        "Visual directive",
        "Surface",
        "surface",
        4,
        "visual",
        ("surface.visual_directives",),
    ),
    (
        "surface.visible",
        "Visible surface",
        "Surface",
        "surface",
        4,
        "dialog",
        ("surface.visible_messages",),
    ),
)
_SELF_NODE_DEFS = (
    (
        "self.source",
        "Source case",
        "Input",
        "input",
        1,
        "source",
        ("self.source",),
    ),
    (
        "cognition.meaning",
        "Meaning appraisal",
        "Cognition",
        "cognition",
        2,
        "appraisal",
        ("cognition.appraisals",),
    ),
    (
        "cognition.goal",
        "Character goal",
        "Cognition",
        "cognition",
        2,
        "goal",
        ("cognition.goal",),
    ),
    (
        "cognition.response",
        "Response plan",
        "Cognition",
        "cognition",
        2,
        "response",
        ("cognition.response_plan",),
    ),
    (
        "cognition.affect",
        "Affect projection",
        "Cognition",
        "cognition",
        2,
        "affect",
        ("cognition.affect",),
    ),
    (
        "reasoning.context",
        "Reasoning and context",
        "Reasoning",
        "cognition",
        2,
        "reasoning",
        ("reasoning.subjective", "reasoning.context_consumption"),
    ),
    (
        "evidence.memory",
        "Memory and context",
        "Evidence",
        "memory",
        2,
        "memory",
        (
            "evidence.shared_memory_prewarm",
            "evidence.memory",
            "context.conversation_progress",
            "context.public_group_scene",
        ),
    ),
    (
        "self.route",
        "Route decision",
        "Decision",
        "decision",
        3,
        "route",
        ("self.route",),
    ),
    (
        "action.results",
        "Actions",
        "Actions",
        "action",
        4,
        "action",
        ("action.requests", "action.results", "action.continuation"),
    ),
    (
        "surface.visual",
        "Visual directive",
        "Surface",
        "surface",
        4,
        "visual",
        ("surface.visual_directives",),
    ),
    (
        "surface.visible",
        "Visible surface",
        "Surface",
        "surface",
        4,
        "dialog",
        ("surface.visible_messages",),
    ),
    (
        "self.consolidation",
        "Consolidation",
        "Continuity",
        "memory",
        5,
        "continuity",
        ("self.consolidation",),
    ),
)


def build_live_cognition_observation(
    *,
    graph_result: Mapping[str, Any],
    persona_state: Mapping[str, Any],
    run_id: str,
    cognition_invocation_id: str,
    terminal_status: EpisodeTerminalStatusV1,
    visual_stage_failed: bool,
    visual_stage_reached: bool | None,
    failure_code: str,
    generated_at: datetime,
) -> CognitionRunObservationV1 | None:
    """Project one settled live graph result into the canonical observation."""

    if terminal_status == "cancelled":
        return None
    state = persona_state
    core, core_valid = _core_source(graph_result, state)
    sections = _live_sections(
        graph_result=graph_result,
        state=state,
        core=core,
        core_valid=core_valid,
        visual_stage_failed=visual_stage_failed,
        visual_stage_reached=visual_stage_reached,
    )
    observation_status = _terminal_observation_status(
        terminal_status,
        sections,
    )
    correlation = CognitionObservationCorrelationV1(
        run_id=_bounded_identifier(run_id),
        llm_trace_id=_bounded_identifier(graph_result.get("llm_trace_id")),
        cognition_invocation_id=_bounded_identifier(cognition_invocation_id),
    )
    observation = _build_observation(
        run_kind="live_turn",
        status=observation_status,
        generated_at=generated_at,
        correlation=correlation,
        sections=sections,
        node_defs=_LIVE_NODE_DEFS,
        edges=_live_edges(),
    )
    return observation


def build_self_cognition_observation(
    *,
    artifact_payloads: Mapping[str, Any],
    visual_stage_failed: bool,
    visual_stage_reached: bool | None,
    generated_at: datetime,
) -> CognitionRunObservationV1 | None:
    """Project a validated completed self-cognition artifact set."""

    run_record = artifact_payloads.get(self_cognition_models.ARTIFACT_RUN_RECORD)
    if not isinstance(run_record, Mapping):
        return None
    run_id = _strict_text(run_record.get("run_id"), maximum=120)
    run_status = _strict_text(run_record.get("status"), maximum=32)
    if not run_id or run_status not in {"completed", "failed", "cancelled"}:
        return None
    llm_trace_id = _optional_identifier(run_record.get("llm_trace_id"))
    if run_record.get("llm_trace_id") is not None and llm_trace_id is None:
        return None
    calendar_run_id = _optional_identifier(
        run_record.get("source_calendar_run_id")
    )
    if (
        run_record.get("source_calendar_run_id") is not None
        and calendar_run_id is None
    ):
        return None
    if run_status == "cancelled":
        return None

    cognition_input = _artifact_mapping(
        artifact_payloads,
        self_cognition_models.ARTIFACT_COGNITION_INPUT,
    )
    cognition_output_wrapper = _artifact_mapping(
        artifact_payloads,
        self_cognition_models.ARTIFACT_COGNITION_OUTPUT,
    )
    route_effect = _artifact_mapping(
        artifact_payloads,
        self_cognition_models.ARTIFACT_ROUTE_EFFECT,
    )
    action_attempt = _artifact_mapping(
        artifact_payloads,
        self_cognition_models.ARTIFACT_ACTION_ATTEMPT,
    )
    consolidation = _artifact_mapping(
        artifact_payloads,
        self_cognition_models.ARTIFACT_CONSOLIDATION_OUTCOME,
    )
    cognition_output, core_valid = _self_core_source(cognition_output_wrapper)
    sections = _self_sections(
        artifact_payloads=artifact_payloads,
        cognition_input=cognition_input,
        cognition_output_wrapper=cognition_output_wrapper,
        cognition_output=cognition_output,
        core_valid=core_valid,
        route_effect=route_effect,
        action_attempt=action_attempt,
        consolidation=consolidation,
        visual_stage_failed=visual_stage_failed,
        visual_stage_reached=visual_stage_reached,
    )
    observation_status = _terminal_observation_status(
        "failed" if run_status == "failed" else "completed_private",
        sections,
    )
    correlation = CognitionObservationCorrelationV1(
        run_id=run_id,
        llm_trace_id=llm_trace_id or run_id,
        cognition_invocation_id=run_id,
        source_calendar_run_id=calendar_run_id,
    )
    observation = _build_observation(
        run_kind="self_cognition",
        status=observation_status,
        generated_at=generated_at,
        correlation=correlation,
        sections=sections,
        node_defs=_SELF_NODE_DEFS,
        edges=_self_edges(),
    )
    return observation


def _build_observation(
    *,
    run_kind: str,
    status: str,
    generated_at: datetime,
    correlation: CognitionObservationCorrelationV1,
    sections: Sequence[CognitionObservationSectionV1],
    node_defs: Sequence[tuple[str, str, str, str, int, str, tuple[str, ...]]],
    edges: Sequence[tuple[str, str, str]],
) -> CognitionRunObservationV1:
    """Build the immutable DTO after all source projection is complete."""

    section_by_id = {section.section_id: section for section in sections}
    nodes: list[CognitionObservationNodeV1] = []
    for (
        node_id,
        label,
        stage,
        lane,
        column,
        category,
        section_refs,
    ) in node_defs:
        node_status = _aggregate_status(
            section_by_id[section_id].status for section_id in section_refs
        )
        summary = next(
            (
                section_by_id[section_id].summary
                for section_id in section_refs
                if section_by_id[section_id].summary
            ),
            node_status,
        )[:180]
        nodes.append(CognitionObservationNodeV1(
            node_id=node_id,
            label=label,
            stage=stage,
            lane=lane,
            column=column,
            category=category,
            status=node_status,
            summary=summary,
            section_refs=list(section_refs),
        ))
    observation_edges = [CognitionObservationEdgeV1(
        source=source,
        target=target,
        kind=kind,
        label="",
    ) for source, target, kind in edges]
    disclosure = CognitionObservationDisclosureV1(
        policy=COGNITION_OBSERVATION_DISCLOSURE_POLICY,
        excluded=list(COGNITION_OBSERVATION_EXCLUSIONS),
    )
    observation = CognitionRunObservationV1(
        schema_version="cognition_run_observation.v1",
        run_kind=run_kind,
        status=status,
        generated_at=generated_at,
        correlation=correlation,
        sections=list(sections),
        nodes=nodes,
        edges=observation_edges,
        disclosure=disclosure,
    )
    return observation


def _live_sections(
    *,
    graph_result: Mapping[str, Any],
    state: Mapping[str, Any],
    core: Mapping[str, Any] | None,
    core_valid: bool,
    visual_stage_failed: bool,
    visual_stage_reached: bool | None,
) -> list[CognitionObservationSectionV1]:
    """Project the fixed live section catalog in producer order."""

    sections = [
        _input_section(state),
        _decision_section(graph_result),
        _appraisals_section(core, core_valid),
        _goal_section(core, core_valid),
        _response_plan_section(core, core_valid),
        _affect_section(core, core_valid),
        _subjective_section(core, core_valid, state),
        _context_consumption_section(state, graph_result),
        _memory_section(state),
        _prewarm_section(state),
        _progress_section(state.get("conversation_progress", _MISSING)),
        _group_scene_section(state),
        _action_requests_section(core, core_valid),
        _action_results_section(state.get("action_results", _MISSING)),
        _continuation_section(state, graph_result),
        _visual_section(
            state,
            should_respond=graph_result.get("should_respond"),
            stage_failed=visual_stage_failed,
            stage_reached=visual_stage_reached,
        ),
        _messages_section(graph_result.get("final_dialog", _MISSING)),
    ]
    return sections


def _self_sections(
    *,
    artifact_payloads: Mapping[str, Any],
    cognition_input: Mapping[str, Any],
    cognition_output_wrapper: Mapping[str, Any],
    cognition_output: Mapping[str, Any] | None,
    core_valid: bool,
    route_effect: Mapping[str, Any],
    action_attempt: Mapping[str, Any],
    consolidation: Mapping[str, Any],
    visual_stage_failed: bool,
    visual_stage_reached: bool | None,
) -> list[CognitionObservationSectionV1]:
    """Project the fixed self-cognition section catalog in producer order."""

    merged_state = dict(cognition_input)
    merged_state.update(cognition_output_wrapper)
    sections = [
        _appraisals_section(cognition_output, core_valid),
        _goal_section(cognition_output, core_valid),
        _response_plan_section(cognition_output, core_valid),
        _affect_section(cognition_output, core_valid),
        _subjective_section(cognition_output, core_valid, merged_state),
        _context_consumption_section(merged_state, cognition_output or {}),
        _memory_section(merged_state),
        _prewarm_section(cognition_output_wrapper),
        _progress_section(_self_progress_source(cognition_input)),
        _not_reported_records_section("context.public_group_scene"),
        _action_requests_section(cognition_output, core_valid),
        _self_action_results_section(
            cognition_output,
            action_attempt,
        ),
        _self_continuation_section(
            cognition_output,
            route_effect,
        ),
        _visual_section(
            merged_state,
            should_respond=True,
            stage_failed=visual_stage_failed,
            stage_reached=visual_stage_reached,
        ),
        _self_messages_section(
            artifact_payloads,
            cognition_output,
            route_effect,
        ),
        _self_source_section(cognition_input),
        _self_route_section(
            artifact_payloads,
            route_effect,
        ),
        _consolidation_section(consolidation, artifact_payloads),
    ]
    return sections


def _input_section(state: Mapping[str, Any]) -> CognitionObservationSectionV1:
    """Project queued user input and reply scope fields."""

    if not isinstance(state, Mapping):
        return _fields_section("input.turn", "failed", [])
    pairs: list[tuple[str, str, object]] = []
    invalid = False
    if "user_input" in state:
        value = _strict_text(state["user_input"])
        if value is None and state["user_input"] is not None:
            invalid = True
        elif value is not None:
            pairs.append(("input", "Input", value))
    reply_context = state.get("reply_context", _MISSING)
    if isinstance(reply_context, Mapping):
        reply_values: list[str] = []
        for key in ("reply_to_display_name", "reply_excerpt"):
            if key in reply_context:
                value = _strict_text(reply_context[key])
                if value is None and reply_context[key] is not None:
                    invalid = True
                elif value:
                    reply_values.append(f"{key}={value}")
        attachments = reply_context.get("reply_attachments")
        if attachments is not None:
            if isinstance(attachments, list):
                for attachment in attachments:
                    if not isinstance(attachment, Mapping):
                        invalid = True
                        continue
                    values: list[str] = []
                    for key in ("media_kind", "description", "summary_status"):
                        value = _strict_text(attachment.get(key))
                        if value is None and attachment.get(key) is not None:
                            invalid = True
                        elif value:
                            values.append(f"{key}={value}")
                    if values:
                        reply_values.append("; ".join(values))
            else:
                invalid = True
        if reply_values:
            pairs.append(("reply_context", "Reply context", reply_values))
    elif reply_context is not _MISSING:
        invalid = True
    episode = state.get("cognitive_episode")
    if isinstance(episode, Mapping):
        target_scope = episode.get("target_scope")
        if isinstance(target_scope, Mapping):
            channel_type = _strict_text(target_scope.get("channel_type"))
            if channel_type is None and target_scope.get("channel_type") is not None:
                invalid = True
            elif channel_type:
                pairs.append((
                    "channel_scope",
                    "Channel scope",
                    channel_type,
                ))
        elif target_scope is not None:
            invalid = True
    elif episode is not None:
        invalid = True
    if invalid and pairs:
        status = "partial"
    elif invalid:
        status = "failed"
    else:
        status = "completed" if pairs else "empty"
    return _fields_section("input.turn", status, pairs)


def _decision_section(
    graph_result: Mapping[str, Any],
) -> CognitionObservationSectionV1:
    """Project the response gate without coercing its Boolean."""

    if not isinstance(graph_result, Mapping):
        return _fields_section("decision.response", "failed", [])
    pairs: list[tuple[str, str, object]] = []
    invalid = False
    if "should_respond" in graph_result:
        should_respond = graph_result["should_respond"]
        if isinstance(should_respond, bool):
            pairs.append(("should_respond", "Should respond", should_respond))
        else:
            invalid = True
    if "reason_to_respond" in graph_result:
        reason = _strict_text(graph_result["reason_to_respond"])
        if reason is None:
            invalid = graph_result["reason_to_respond"] is not None
        else:
            pairs.append(("reason", "Reason", reason))
    if invalid and pairs:
        status = "partial"
    elif invalid:
        status = "failed"
    else:
        status = "completed" if pairs else "not_reported"
    return _fields_section("decision.response", status, pairs)


def _appraisals_section(
    core: Mapping[str, Any] | None,
    core_valid: bool,
) -> CognitionObservationSectionV1:
    """Project semantic appraisal rows and bounded axis changes."""

    if core is None:
        return _records_section("cognition.appraisals", _MISSING, _appraisal_row)
    if not core_valid:
        return _records_section("cognition.appraisals", _MISSING, _appraisal_row, force_status="failed")
    raw = core.get("appraisals", _MISSING)
    if not isinstance(raw, list) and raw is not _MISSING:
        return _records_section("cognition.appraisals", raw, _appraisal_row)
    section = _records_section("cognition.appraisals", raw, _appraisal_row)
    applicable = sum(
        1
        for row in raw
        if isinstance(row, Mapping) and isinstance(row.get("applicable"), bool)
        and row["applicable"]
    ) if isinstance(raw, list) else 0
    fields = list(section.fields)
    fields.insert(0, _make_field("applicable_count", "Applicable count", applicable))
    fields.insert(1, _make_field("reported_count", "Reported count", section.reported_record_count))
    return section.model_copy(update={"fields": fields})


def _appraisal_row(
    row: object,
    index: int,
) -> tuple[CognitionObservationRecordV1 | None, bool]:
    """Project one appraisal mapping using only its declared semantic keys."""

    if not isinstance(row, Mapping):
        return None, False
    fields: list[tuple[str, str, object]] = []
    invalid = False
    for key, label in (
        ("family", "Family"),
        ("applicable", "Applicable"),
        ("semantic_summary", "Semantic summary"),
        ("cause_summary", "Cause summary"),
    ):
        value, value_invalid = _project_scalar_field(row, key, label)
        invalid = invalid or value_invalid
        if value is not None:
            fields.append(value)
    axis_changes = row.get("axis_changes", _MISSING)
    if axis_changes is not _MISSING:
        projected, list_invalid = _axis_changes(axis_changes)
        invalid = invalid or list_invalid
        if projected:
            fields.append(_make_field("axis_changes", "Axis changes", projected))
    return _make_record(index, "Appraisal", fields), not invalid


def _goal_section(
    core: Mapping[str, Any] | None,
    core_valid: bool,
) -> CognitionObservationSectionV1:
    """Project the active character goal fields."""

    if core is None:
        return _fields_section("cognition.goal", "not_reported", [])
    if not core_valid:
        return _fields_section("cognition.goal", "failed", [])
    goal = core.get("active_character_goal", _MISSING)
    if goal is _MISSING:
        return _fields_section("cognition.goal", "not_reported", [])
    if not isinstance(goal, Mapping):
        return _fields_section("cognition.goal", "failed", [])
    fields, invalid = _project_mapping_fields(
        goal,
        (
            ("goal_kind", "Goal kind"),
            ("intent", "Intent"),
            ("reason", "Reason"),
            ("cause_summary", "Cause summary"),
        ),
    )
    status = "partial" if invalid and fields else "failed" if invalid else (
        "completed" if fields else "empty"
    )
    return _fields_section("cognition.goal", status, fields)


def _response_plan_section(
    core: Mapping[str, Any] | None,
    core_valid: bool,
) -> CognitionObservationSectionV1:
    """Project response-plan prose and raw request counts."""

    if core is None:
        return _fields_section("cognition.response_plan", "not_reported", [])
    if not core_valid:
        return _fields_section("cognition.response_plan", "failed", [])
    plan = core.get("response_plan", _MISSING)
    if plan is _MISSING:
        return _fields_section("cognition.response_plan", "not_reported", [])
    if not isinstance(plan, Mapping):
        return _fields_section("cognition.response_plan", "failed", [])
    fields, invalid = _project_mapping_fields(
        plan,
        (
            ("goal_resolution", "Goal resolution"),
            ("response_goal", "Response goal"),
            ("epistemic_boundary", "Epistemic boundary"),
        ),
    )
    for key, label in (
        ("action_requests", "Action request count"),
        ("resolver_requests", "Resolver request count"),
    ):
        raw = plan.get(key, _MISSING)
        if raw is _MISSING:
            continue
        if isinstance(raw, list):
            fields.append(_make_field(
                key.removesuffix("s") + "_count",
                label,
                len(raw),
            ))
        else:
            invalid = True
    status = "partial" if invalid and fields else "failed" if invalid else (
        "completed" if fields else "empty"
    )
    return _fields_section("cognition.response_plan", status, fields)


def _affect_section(
    core: Mapping[str, Any] | None,
    core_valid: bool,
) -> CognitionObservationSectionV1:
    """Project bounded affect rows while preserving finite numeric intensity."""

    if core is None:
        return _records_section("cognition.affect", _MISSING, _affect_row)
    if not core_valid:
        return _records_section("cognition.affect", _MISSING, _affect_row, force_status="failed")
    raw = core.get("affect_projection", _MISSING)
    return _records_section("cognition.affect", raw, _affect_row)


def _affect_row(
    row: object,
    index: int,
) -> tuple[CognitionObservationRecordV1 | None, bool]:
    """Project one affect row with a numeric-or-text intensity field."""

    if not isinstance(row, Mapping):
        return None, False
    fields: list[CognitionObservationFieldV1] = []
    invalid = False
    for key, label in (
        ("emotion", "Emotion"),
        ("phase", "Phase"),
        ("trend", "Trend"),
        ("cause_summary", "Cause summary"),
    ):
        value, value_invalid = _project_scalar_field(row, key, label)
        invalid = invalid or value_invalid
        if value is not None:
            fields.append(value)
    if "intensity" in row:
        intensity = row["intensity"]
        if _finite_number(intensity) or isinstance(intensity, str):
            fields.append(_make_field("intensity", "Intensity", intensity))
        elif intensity is not None:
            invalid = True
    return _make_record(index, "Affect", fields), not invalid


def _subjective_section(
    core: Mapping[str, Any] | None,
    core_valid: bool,
    state: Mapping[str, Any],
) -> CognitionObservationSectionV1:
    """Project private monologue and direct persona-state stance fields."""

    if core is None and not state:
        return _fields_section("reasoning.subjective", "not_reported", [])
    if core is not None and not core_valid:
        return _fields_section("reasoning.subjective", "failed", [])
    fields: list[tuple[str, str, object]] = []
    invalid = False
    if core is not None:
        value = core.get("private_monologue", _MISSING)
        if value is not _MISSING:
            fields.append(("private_monologue", "Private monologue", value))
    for key, label in (
        ("logical_stance", "Logical stance"),
        ("character_intent", "Character intent"),
        ("judgment_note", "Judgment note"),
    ):
        if key in state:
            fields.append((key, label, state[key]))
    projected: list[tuple[str, str, object]] = []
    for key, label, value in fields:
        scalar = _strict_scalar(value)
        if scalar is None and value is not None:
            invalid = True
        elif scalar is not None:
            projected.append((key, label, scalar))
    status = "partial" if invalid and projected else "failed" if invalid else (
        "completed" if projected else "empty"
    )
    return _fields_section("reasoning.subjective", status, projected)


def _context_consumption_section(
    state: Mapping[str, Any],
    graph_result: Mapping[str, Any],
) -> CognitionObservationSectionV1:
    """Project the closed stage/source context-consumption record catalog."""

    records: list[CognitionObservationRecordV1] = []
    invalid_count = 0
    valid_count = 0
    source_count = 0
    settled = state.get("settled_relevance_context_consumption", _MISSING)
    cognition = state.get("cognition_input", _MISSING)
    text_surface = state.get("text_surface_input", _MISSING)
    style_context = state.get("interaction_style_context", _MISSING)
    episode_trace = state.get("episode_trace", graph_result.get("episode_trace", _MISSING))
    consolidation_metadata = state.get("consolidation_metadata", _MISSING)
    pairs: list[tuple[str, str, object, str]] = [
        ("settled_relevance", "character_operational_context", _nested_value(settled, "character_operational_context"), "character"),
        ("settled_relevance", "relationship_context", _nested_value(settled, "relationship_context"), "relationship"),
        ("settled_relevance", "style_relevance", _nested_value(settled, "style"), "style_relevance"),
        ("cognition", "character_operational_context", _nested_value(cognition, "character_operational_context"), "character"),
        ("cognition", "relationship_context", _nested_value(cognition, "relationship_context"), "relationship"),
        ("cognition", "group_engagement_action_context", _nested_value(cognition, "group_engagement_action_context"), "group_engagement"),
        (
            "surface",
            "style",
            (
                _nested_value(style_context, "surface")
                if isinstance(text_surface, Mapping)
                else _MISSING
            ),
            "style_surface",
        ),
        ("health", "predecessor", _nested_value(settled, "predecessor"), "predecessor"),
        ("health", "attempt_diagnostics", _nested_value(episode_trace, "attempt_diagnostics"), "attempts"),
        ("health", "operational_receipt", _nested_value(consolidation_metadata, "character_operational_receipt"), "receipt"),
    ]
    for stage, source_kind, raw, source_type in pairs:
        source_count += raw is not _MISSING
        record, disposition = _context_record(
            stage,
            source_kind,
            raw,
            source_type,
        )
        records.append(record)
        if disposition == "not_reported":
            continue
        if disposition == "failed":
            invalid_count += 1
        else:
            valid_count += 1
    if source_count == 0:
        status = "not_reported"
    elif invalid_count and valid_count:
        status = "partial"
    elif invalid_count:
        status = "failed"
    elif all(
        record.fields[2].value == "empty"
        for record in records
        if record.fields[2].value != "not_reported"
    ):
        status = "empty"
    else:
        status = "completed"
    fields = [
        ("overall_status", "Overall status", status),
        ("consumer_count", "Consumer count", len(records)),
    ]
    return _records_section(
        "reasoning.context_consumption",
        records,
        lambda row, index: (row, True),
        direct_records=True,
        status_override=status,
        header_fields=fields,
    )


def _context_record(
    stage: str,
    source_kind: str,
    raw: object,
    source_type: str,
) -> tuple[CognitionObservationRecordV1, str]:
    """Project one context source into the common detail record shape."""

    if raw is _MISSING:
        fields = [
            _make_field("stage", "Stage", stage),
            _make_field("source_kind", "Source kind", source_kind),
            _make_field("status", "Status", "not_reported"),
        ]
        return _make_record(0, "Context", fields, key=f"{stage}_{source_kind}"), "not_reported"
    if source_type == "attempts":
        details, invalid = _attempt_details(raw)
    elif source_type == "receipt":
        details, invalid = _receipt_details(raw)
    elif source_type.startswith("style"):
        details, invalid = _style_details(raw, source_type)
    elif source_type == "character":
        details, invalid = _character_details(raw)
    elif source_type == "relationship":
        details, invalid = _relationship_details(raw)
    elif source_type == "group_engagement":
        details, invalid = _group_engagement_details(raw)
    else:
        details, invalid = _predecessor_details(raw)
    if source_type == "attempts":
        raw_type_valid = isinstance(raw, list)
    else:
        raw_type_valid = isinstance(raw, Mapping) or raw is None
    if not raw_type_valid:
        invalid = True
    source_status = (
        _strict_text(raw.get("status"), maximum=40)
        if isinstance(raw, Mapping)
        else None
    )
    if source_status not in {
        "active",
        "empty",
        "missing",
        "failed",
        "healthy",
        "degraded",
    }:
        source_status = "completed" if details else "empty"
    if invalid and details:
        record_status = "partial"
    elif invalid:
        record_status = "failed"
    elif not details:
        record_status = "empty"
    else:
        record_status = "completed"
    fields = [
        _make_field("stage", "Stage", stage),
        _make_field("source_kind", "Source kind", source_kind),
        _make_field("status", "Status", source_status),
    ]
    if isinstance(raw, Mapping):
        summary = _first_text(raw, ("summary", "semantic_summary"))
        if summary:
            fields.append(_make_field("summary", "Summary", summary))
    if details:
        fields.append(_make_field("details", "Details", details))
    return _make_record(0, "Context", fields, key=f"{stage}_{source_kind}"), record_status


def _character_details(value: object) -> tuple[list[str], bool]:
    """Flatten the approved operational affect and pressure fields."""

    if not isinstance(value, Mapping):
        return [], value is not None
    details: list[str] = []
    invalid = False
    for collection_name, keys, limit in (
        (
            "affect",
            ("emotion_id", "intensity", "phase", "trend", "root_kind", "cause_class", "freshness"),
            3,
        ),
        (
            "pressures",
            ("kind", "salience", "lifecycle", "cause_class", "freshness"),
            4,
        ),
    ):
        rows = value.get(collection_name, _MISSING)
        if rows is _MISSING:
            continue
        if not isinstance(rows, list):
            invalid = True
            continue
        for row in rows[:limit]:
            if not isinstance(row, Mapping):
                invalid = True
                continue
            for key in keys:
                scalar = _strict_scalar(row.get(key))
                if scalar is not None:
                    details.append(f"{collection_name}.{key}={scalar}")
                elif row.get(key) is not None and key in row:
                    invalid = True
    return details, invalid


def _relationship_details(value: object) -> tuple[list[str], bool]:
    """Flatten the approved relationship axis and causal fields."""

    if not isinstance(value, Mapping):
        return [], value is not None
    details: list[str] = []
    invalid = False
    axes = value.get("axes", _MISSING)
    axis_keys = (
        "familiarity", "positive_regard", "trust", "attachment",
        "desired_closeness", "perceived_closeness", "care",
        "boundary_safety", "exclusivity", "unresolved_injury", "salience",
    )
    if axes is not _MISSING:
        if not isinstance(axes, Mapping):
            invalid = True
        else:
            for key in axis_keys:
                scalar = _strict_scalar(axes.get(key))
                if scalar is not None:
                    details.append(f"{key}={scalar}")
                elif axes.get(key) is not None and key in axes:
                    invalid = True
    for collection_name, keys, limit in (
        (
            "causal_context",
            ("entity_kind", "semantic_summary", "salience", "lifecycle", "freshness"),
            2,
        ),
        (
            "affect",
            ("emotion_id", "intensity", "phase", "trend", "freshness"),
            2,
        ),
    ):
        rows = value.get(collection_name, _MISSING)
        if rows is _MISSING:
            continue
        if not isinstance(rows, list):
            invalid = True
            continue
        for row in rows[:limit]:
            if not isinstance(row, Mapping):
                invalid = True
                continue
            for key in keys:
                scalar = _strict_scalar(row.get(key))
                if scalar is not None:
                    details.append(f"{collection_name}.{key}={scalar}")
                elif row.get(key) is not None and key in row:
                    invalid = True
    for key in ("relationship_freshness", "evidence_freshness"):
        if key not in value:
            continue
        scalar = _strict_scalar(value[key])
        if scalar is not None:
            details.append(f"{key}={scalar}")
        elif value[key] is not None:
            invalid = True
    return details, invalid


def _style_details(value: object, source_type: str) -> tuple[list[str], bool]:
    """Flatten relevance or surface style sources in role order."""

    if not isinstance(value, Mapping):
        return [], value is not None
    if source_type == "style_relevance":
        sources = value.get("relevance", _MISSING)
        keys = ("engagement_guidelines",)
        limit = 3
    else:
        sources = value
        keys = (
            "speech_guidelines",
            "social_guidelines",
            "pacing_guidelines",
            "engagement_guidelines",
        )
        limit = 8
    if not isinstance(sources, Mapping):
        return [], True
    details: list[str] = []
    invalid = False
    consumer_role = (
        "relevance"
        if source_type == "style_relevance"
        else "surface"
    )
    details.append(f"consumer_role={consumer_role}")
    for role in ("user", "group_channel"):
        source = sources.get(role, _MISSING)
        if source is _MISSING:
            continue
        if not isinstance(source, Mapping):
            invalid = True
            continue
        details.append(f"source={role}")
        source_details, source_invalid = _direct_details(
            source,
            ("status", "revision", "confidence"),
        )
        invalid = invalid or source_invalid
        details.extend(source_details)
        if "revision" in source and (
            isinstance(source["revision"], bool)
            or not isinstance(source["revision"], int)
            or source["revision"] < 0
        ):
            invalid = True
        if "overlay" in source:
            overlay = source["overlay"]
            if isinstance(overlay, Mapping):
                source = overlay
            elif source_type == "style_surface":
                invalid = True
                continue
        for key in keys:
            raw = source.get(key, _MISSING)
            if raw is _MISSING:
                continue
            values, list_invalid = _string_list(raw, limit=limit)
            invalid = invalid or list_invalid
            details.extend(f"{key}={item}" for item in values)
    return details, invalid


def _group_engagement_details(value: object) -> tuple[list[str], bool]:
    """Flatten the selected group-engagement guidance."""

    if not isinstance(value, Mapping):
        return [], value is not None
    details, invalid = _direct_details(value, ("confidence",))
    values, list_invalid = _string_list(
        value.get("engagement_guidelines", _MISSING),
        limit=3,
    )
    details.extend(f"engagement_guidelines={item}" for item in values)
    return details, invalid or list_invalid


def _predecessor_details(value: object) -> tuple[list[str], bool]:
    """Flatten predecessor health fields in their fixed order."""

    if not isinstance(value, Mapping):
        return [], value is not None
    return _direct_details(
        value,
        ("status", "watermark", "awaited_count", "timed_out_count", "wait_ms"),
    )


def _attempt_details(value: object) -> tuple[list[str], bool]:
    """Flatten bounded attempt diagnostics without checkpoint or identifiers."""

    if not isinstance(value, list):
        return [], value is not None
    details: list[str] = []
    invalid = False
    for row in value[:8]:
        if not isinstance(row, Mapping):
            invalid = True
            continue
        for key in ("stage", "error_code", "attempt_count", "final_status"):
            scalar = _strict_scalar(row.get(key))
            if scalar is not None:
                details.append(f"{key}={scalar}")
            elif key in row and row[key] is not None:
                invalid = True
    return details, invalid


def _receipt_details(value: object) -> tuple[list[str], bool]:
    """Flatten the safe operational receipt fields."""

    if not isinstance(value, Mapping):
        return [], value is not None
    return _direct_details(value, ("status", "error_code", "durable", "attempt_count"))


def _direct_details(
    value: Mapping[str, Any],
    keys: Sequence[str],
) -> tuple[list[str], bool]:
    """Read only explicit scalar keys into ``key=value`` strings."""

    details: list[str] = []
    invalid = False
    for key in keys:
        if key not in value:
            continue
        scalar = _strict_scalar(value[key])
        if scalar is None and value[key] is not None:
            invalid = True
        elif scalar is not None:
            details.append(f"{key}={scalar}")
    return details, invalid


def _memory_section(state: Mapping[str, Any]) -> CognitionObservationSectionV1:
    """Project the closed five-list retrieval evidence catalog."""

    rag = state.get("rag_result", _MISSING)
    if rag is _MISSING:
        return _records_section("evidence.memory", _MISSING, _evidence_row)
    if not isinstance(rag, Mapping):
        return _records_section("evidence.memory", rag, _evidence_row)
    rows: list[tuple[str, object]] = []
    invalid = False
    for source_kind in (
        "memory_evidence",
        "conversation_evidence",
        "external_evidence",
        "recall_evidence",
        "media_evidence",
    ):
        raw = rag.get(source_kind, _MISSING)
        if raw is _MISSING:
            continue
        if not isinstance(raw, list):
            invalid = True
            continue
        rows.extend((source_kind, row) for row in raw)
    section = _records_section(
        "evidence.memory",
        rows,
        lambda row, index: _evidence_row(row[1], index, source_kind=row[0]),
        pair_rows=True,
        force_invalid=invalid,
    )
    if "answer" in rag and rag["answer"] is not None and not isinstance(
        rag["answer"],
        str,
    ):
        invalid = True
    answer = _strict_text(rag.get("answer"))
    header_fields: list[tuple[str, str, object]] = []
    if answer:
        header_fields.append(("retrieval_answer", "Retrieval answer", answer))
    header_fields.append(("reported_count", "Reported count", section.reported_record_count))
    if invalid:
        invalid_status = "partial" if section.records else "failed"
        section = section.model_copy(update={
            "status": invalid_status,
            "summary": _section_summary(
                "evidence.memory",
                invalid_status,
            ),
        })
    return section.model_copy(update={"fields": _fields(header_fields)})


def _evidence_row(
    row: object,
    index: int,
    *,
    source_kind: str = "memory_evidence",
) -> tuple[CognitionObservationRecordV1 | None, bool]:
    """Project one evidence row through the fixed safe allowlist."""

    if isinstance(row, str):
        fields = [_make_field("source_kind", "Source kind", source_kind)]
        if row:
            fields.append(_make_field("content", "Content", row))
        return _make_record(index, "Evidence", fields), True
    if not isinstance(row, Mapping):
        return None, False
    fields: list[CognitionObservationFieldV1] = [
        _make_field("source_kind", "Source kind", source_kind),
    ]
    invalid = False
    for key, label in (
        ("summary", "Summary"),
        ("title", "Title"),
        ("relevance", "Relevance"),
        ("recency", "Recency"),
        ("due_state", "Due state"),
        ("evidence_boundary_notes", "Evidence boundary notes"),
    ):
        value, value_invalid = _project_scalar_field(row, key, label)
        invalid = invalid or value_invalid
        if value is not None:
            fields.append(value)
    content, content_invalid = _first_projected_scalar(
        row,
        ("fact", "excerpt", "content"),
        "Content",
    )
    invalid = invalid or content_invalid
    if content is not None:
        fields.insert(2, content)
    summary = _first_text(row, ("summary", "fact", "excerpt", "content", "title")) or ""
    return _make_record(index, "Evidence", fields, summary=summary), not invalid


def _prewarm_section(
    state: Mapping[str, Any],
) -> CognitionObservationSectionV1:
    """Project one validated shared-memory prewarm outcome."""

    raw = state.get("shared_memory_prewarm_outcome", _MISSING)
    if raw is _MISSING:
        return _records_section("evidence.shared_memory_prewarm", _MISSING, _evidence_row)
    try:
        outcome = validate_shared_memory_prewarm_outcome(raw)
    except (TypeError, ValueError):
        return _records_section(
            "evidence.shared_memory_prewarm",
            _MISSING,
            _evidence_row,
            force_status="failed",
        )
    rag_result = outcome["rag_result"]
    section = _records_section(
        "evidence.shared_memory_prewarm",
        rag_result["memory_evidence"],
        lambda row, index: _evidence_row(
            row,
            index,
            source_kind="shared_memory",
        ),
    )
    fields = [
        ("attempted", "Attempted", outcome["attempted"]),
        ("reason_code", "Reason code", outcome["reason_code"]),
        ("latency_ms", "Latency ms", outcome["latency_ms"]),
        ("retrieved_count", "Retrieved count", outcome["retrieved_shared_count"]),
        ("merged_count", "Merged count", outcome["merged_shared_count"]),
    ]
    status = outcome["status"]
    if section.status in {"partial", "failed"} and status == "completed":
        status = section.status
    return section.model_copy(update={
        "status": status,
        "fields": _fields(fields),
        "summary": _section_summary("evidence.shared_memory_prewarm", status),
    })


def _progress_section(raw: object) -> CognitionObservationSectionV1:
    """Project the exact conversation-progress v2 packet."""

    if raw is _MISSING:
        return _records_section("context.conversation_progress", _MISSING, _progress_row)
    if not isinstance(raw, Mapping):
        return _records_section("context.conversation_progress", raw, _progress_row)
    if raw.get("schema_version") != "conversation_progress_prompt.v2":
        return _records_section("context.conversation_progress", _MISSING, _progress_row, force_status="failed")
    events = raw.get("events", _MISSING)
    section = _records_section("context.conversation_progress", events, _progress_row)
    headers: list[tuple[str, str, object]] = []
    invalid_header = False
    for key, label in (
        ("status", "Status"),
        ("continuity", "Continuity"),
        ("turn_count", "Turn count"),
        ("current_thread", "Current thread"),
        ("character_stance", "Character stance"),
        ("user_goal", "User goal"),
        ("current_blocker", "Current blocker"),
        ("emotional_trajectory", "Emotional trajectory"),
        ("episode_narrative", "Episode narrative"),
        ("overused_moves", "Overused moves"),
    ):
        if key not in raw:
            continue
        value = _strict_scalar_list(raw[key]) if key == "overused_moves" else _strict_scalar(raw[key])
        if value is not None:
            headers.append((key, label, value))
        elif raw[key] is not None:
            invalid_header = True
    status = section.status
    if invalid_header:
        status = "partial" if section.records else "failed"
    return section.model_copy(update={
        "status": status,
        "summary": _section_summary("context.conversation_progress", status),
        "fields": _fields(headers),
    })


def _progress_row(
    row: object,
    index: int,
) -> tuple[CognitionObservationRecordV1 | None, bool]:
    """Project one conversation-progress event."""

    if not isinstance(row, Mapping):
        return None, False
    fields: list[CognitionObservationFieldV1] = []
    invalid = False
    for key, label in (
        ("semantic_summary", "Semantic summary"),
        ("state", "State"),
        ("actor", "Actor"),
        ("action", "Action"),
        ("object", "Object"),
        ("beneficiary", "Beneficiary"),
        ("precondition", "Precondition"),
    ):
        value, value_invalid = _project_scalar_field(row, key, label)
        invalid = invalid or value_invalid
        if value is not None:
            fields.append(value)
    summary = _first_text(row, ("semantic_summary", "action", "object")) or ""
    return _make_record(index, "Progress event", fields, summary=summary), not invalid


def _group_scene_section(state: Mapping[str, Any]) -> CognitionObservationSectionV1:
    """Project group-scene discriminator state and typed transient context."""

    projection_status = state.get("public_group_scene_projection_status", _MISSING)
    if projection_status is _MISSING:
        return _records_section("context.public_group_scene", _MISSING, _group_scene_row)
    if projection_status not in {"completed", "skipped", "failed"}:
        return _records_section("context.public_group_scene", _MISSING, _group_scene_row, force_status="failed")
    if projection_status == "skipped":
        return _records_section("context.public_group_scene", [], _group_scene_row, force_status="skipped")
    context = state.get("public_group_scene_context", _MISSING)
    if not isinstance(context, Mapping) or context.get("schema_version") != "group_scene_context.v1":
        return _records_section("context.public_group_scene", _MISSING, _group_scene_row, force_status="failed")
    turns = context.get("turns", _MISSING)
    section = _records_section("context.public_group_scene", turns, _group_scene_row)
    visible = context.get("visible_participants")
    omitted = context.get("omitted_turn_count")
    headers: list[tuple[str, str, object]] = []
    invalid_header = False
    headers.append(("status", "Status", projection_status))
    if isinstance(visible, list) and all(isinstance(item, str) for item in visible):
        headers.append(("visible_participants", "Visible participants", list(visible)))
        headers.append(("visible_participant_count", "Visible participant count", len(visible)))
    elif visible is not None:
        invalid_header = True
    if isinstance(omitted, int) and not isinstance(omitted, bool) and omitted >= 0:
        headers.append(("omitted_turn_count", "Omitted turn count", omitted))
    elif omitted is not None:
        invalid_header = True
    status = section.status
    if invalid_header:
        status = "partial" if section.records else "failed"
    return section.model_copy(update={
        "status": status,
        "summary": _section_summary("context.public_group_scene", status),
        "fields": _fields(headers),
    })


def _group_scene_row(
    row: object,
    index: int,
) -> tuple[CognitionObservationRecordV1 | None, bool]:
    """Project one public group-scene turn."""

    if not isinstance(row, Mapping):
        return None, False
    fields: list[CognitionObservationFieldV1] = []
    invalid = False
    for key, label in (
        ("role", "Role"),
        ("speaker_name", "Speaker name"),
        ("text", "Text"),
        ("addressed_names", "Addressed names"),
        ("reply_to_name", "Reply-to name"),
        ("scene_position", "Scene position"),
        ("anchor_kind", "Anchor kind"),
    ):
        value = row.get(key, _MISSING)
        if value is _MISSING:
            continue
        if key == "addressed_names":
            projected = _strict_scalar_list(value)
        else:
            projected = _strict_scalar(value)
        if projected is None and value is not None:
            invalid = True
        elif projected is not None:
            fields.append(_make_field(key, label, projected))
    summary = _strict_text(row.get("text")) or ""
    return _make_record(index, "Scene turn", fields, summary=summary), not invalid


def _action_requests_section(
    core: Mapping[str, Any] | None,
    core_valid: bool,
) -> CognitionObservationSectionV1:
    """Project cognition-owned action requests."""

    if core is None:
        return _records_section("action.requests", _MISSING, _action_request_row)
    if not core_valid:
        return _records_section("action.requests", _MISSING, _action_request_row, force_status="failed")
    plan = core.get("response_plan", _MISSING)
    if not isinstance(plan, Mapping):
        return _records_section("action.requests", _MISSING, _action_request_row)
    raw = plan.get("action_requests", _MISSING)
    section = _records_section("action.requests", raw, _action_request_row)
    return section.model_copy(update={
        "fields": _fields([("reported_count", "Reported count", section.reported_record_count)]),
    })


def _action_request_row(
    row: object,
    index: int,
) -> tuple[CognitionObservationRecordV1 | None, bool]:
    """Project one action request without parameters or handler metadata."""

    return _action_row(
        row,
        index,
        specs=(
            ("action_kind", "Action kind", ("action_kind",)),
            ("decision", "Decision", ("decision",)),
            ("detail", "Detail", ("detail",)),
            ("reason", "Reason", ("reason",)),
        ),
    )


def _action_results_section(raw: object) -> CognitionObservationSectionV1:
    """Project settled action results."""

    section = _records_section("action.results", raw, _action_result_row)
    return section.model_copy(update={
        "fields": _fields([("reported_count", "Reported count", section.reported_record_count)]),
    })


def _self_action_results_section(
    cognition_output: Mapping[str, Any] | None,
    action_attempt: Mapping[str, Any],
) -> CognitionObservationSectionV1:
    """Use the self action-attempt fallback only when results are absent."""

    raw = cognition_output.get("action_results", _MISSING) if cognition_output else _MISSING
    if isinstance(raw, list):
        if raw:
            return _action_results_section(raw)
        if action_attempt:
            return _action_results_section([action_attempt])
        return _action_results_section(raw)
    if raw is not _MISSING:
        return _action_results_section(raw)
    return _action_results_section(raw)


def _action_result_row(
    row: object,
    index: int,
) -> tuple[CognitionObservationRecordV1 | None, bool]:
    """Project one action result using explicit scalar fallbacks."""

    return _action_row(
        row,
        index,
        specs=(
            ("action_kind", "Action kind", ("action_kind", "kind")),
            ("status", "Status", ("status",)),
            ("visibility", "Visibility", ("visibility",)),
            ("outcome", "Outcome", ("result_summary", "outcome", "objective_summary")),
            ("reason", "Reason", ("reason",)),
            ("due_at", "Due at", ("due_at", "deadline")),
        ),
    )


def _action_row(
    row: object,
    index: int,
    *,
    specs: Sequence[tuple[str, str, tuple[str, ...]]],
) -> tuple[CognitionObservationRecordV1 | None, bool]:
    """Project a scalar-only action row from a closed key table."""

    if not isinstance(row, Mapping):
        return None, False
    fields: list[CognitionObservationFieldV1] = []
    invalid = False
    for output_key, label, input_keys in specs:
        found = False
        for input_key in input_keys:
            if input_key not in row:
                continue
            found = True
            value = row[input_key]
            scalar = _strict_scalar(value)
            if scalar is None and value is not None:
                invalid = True
            elif scalar is not None:
                fields.append(_make_field(output_key, label, scalar))
            break
        if found:
            continue
    summary = _first_text(row, tuple(key for _, _, keys in specs for key in keys)) or ""
    return _make_record(index, "Action", fields, summary=summary), not invalid


def _continuation_section(
    state: Mapping[str, Any],
    graph_result: Mapping[str, Any],
) -> CognitionObservationSectionV1:
    """Concatenate the explicit live continuation sources in order."""

    rows: list[object] = []
    invalid = False
    direct = state.get("action_continuation", _MISSING)
    if direct is not _MISSING:
        if isinstance(direct, Mapping):
            rows.append(direct)
        elif isinstance(direct, list):
            rows.extend(direct)
        else:
            invalid = True
    specs = state.get("action_specs", _MISSING)
    if specs is not _MISSING:
        if isinstance(specs, list):
            for spec in specs:
                if not isinstance(spec, Mapping):
                    invalid = True
                    continue
                continuation = spec.get("continuation", _MISSING)
                if continuation is _MISSING:
                    continue
                if isinstance(continuation, Mapping):
                    rows.append(continuation)
                elif isinstance(continuation, list):
                    rows.extend(continuation)
                else:
                    invalid = True
        else:
            invalid = True
    future = graph_result.get("future_promises", _MISSING)
    if future is not _MISSING:
        if isinstance(future, list):
            rows.extend(future)
        else:
            invalid = True
    section = _records_section("action.continuation", rows, _continuation_row, force_invalid=invalid)
    return section.model_copy(update={
        "fields": _fields([("reported_count", "Reported count", section.reported_record_count)]),
    })


def _self_continuation_section(
    cognition_output: Mapping[str, Any] | None,
    route_effect: Mapping[str, Any],
) -> CognitionObservationSectionV1:
    """Concatenate self continuation rows and route-effect next topic."""

    state = cognition_output or {}
    rows: list[object] = []
    invalid = False
    source_seen = False
    direct = state.get("action_continuation", _MISSING)
    if direct is not _MISSING:
        source_seen = True
        if isinstance(direct, Mapping):
            rows.append(direct)
        elif isinstance(direct, list):
            rows.extend(direct)
        else:
            invalid = True
    specs = state.get("action_specs", _MISSING)
    if specs is not _MISSING:
        source_seen = True
        if isinstance(specs, list):
            for spec in specs:
                if not isinstance(spec, Mapping):
                    invalid = True
                    continue
                continuation = spec.get("continuation", _MISSING)
                if continuation is _MISSING:
                    continue
                if isinstance(continuation, Mapping):
                    rows.append(continuation)
                elif isinstance(continuation, list):
                    rows.extend(continuation)
                else:
                    invalid = True
    next_topic = route_effect.get("next_topic", _MISSING)
    if next_topic is not _MISSING:
        source_seen = True
        if isinstance(next_topic, Mapping):
            raw_rows = [next_topic]
        elif isinstance(next_topic, list):
            raw_rows = next_topic
        else:
            raw_rows = []
            invalid = True
        rows.extend(raw_rows)
    if not source_seen:
        return _records_section(
            "action.continuation",
            _MISSING,
            _continuation_row,
        ).model_copy(update={
            "fields": _fields([(
                "reported_count",
                "Reported count",
                0,
            )]),
        })
    section = _records_section(
        "action.continuation",
        rows,
        _continuation_row,
        force_invalid=invalid,
    )
    return section.model_copy(update={
        "fields": _fields([
            ("reported_count", "Reported count", section.reported_record_count),
        ]),
    })


def _continuation_row(
    row: object,
    index: int,
) -> tuple[CognitionObservationRecordV1 | None, bool]:
    """Project one continuation mapping with scalar fallbacks."""

    return _action_row(
        row,
        index,
        specs=(
            ("mode", "Mode", ("mode", "episode_type")),
            ("objective", "Objective", ("objective", "objective_summary", "summary", "title", "text", "next_topic")),
            ("status", "Status", ("status", "due_state")),
            ("reason", "Reason", ("reason", "condition")),
            ("due_at", "Due at", ("due_at",)),
        ),
    )


def _visual_section(
    state: Mapping[str, Any],
    *,
    should_respond: object,
    stage_failed: bool,
    stage_reached: bool | None,
) -> CognitionObservationSectionV1:
    """Project visual directives with explicit eligibility dispositions."""

    if not _visual_enabled(state) or should_respond is False:
        return _records_section("surface.visual_directives", [], _visual_row, force_status="skipped")
    if stage_failed:
        return _records_section("surface.visual_directives", _MISSING, _visual_row, force_status="failed")
    if stage_reached is False:
        return _records_section("surface.visual_directives", _MISSING, _visual_row, force_status="not_reported")
    raw = _visual_raw(state)
    if raw is _MISSING:
        return _records_section("surface.visual_directives", _MISSING, _visual_row)
    if not isinstance(raw, Mapping):
        return _records_section("surface.visual_directives", _MISSING, _visual_row, force_status="failed")
    rows: list[object] = []
    invalid = False
    reported = 0
    for kind in _VISUAL_KINDS:
        values = raw.get(kind, [])
        reported += 1
        if not isinstance(values, list):
            invalid = True
            continue
        rows.append((kind, values))
    section = _records_section(
        "surface.visual_directives",
        rows,
        _visual_row,
        pair_rows=True,
        force_invalid=invalid,
    )
    status = section.status
    if not rows and invalid:
        status = "failed"
    elif not any(values for _, values in rows) and not invalid:
        status = "empty"
    return section.model_copy(update={
        "status": status,
        "reported_record_count": reported,
        "truncated": reported > section.displayed_record_count,
        "fields": _fields([("reported_kind_count", "Reported kind count", reported)]),
        "summary": _section_summary("surface.visual_directives", status),
    })


def _visual_row(
    row: object,
    index: int,
) -> tuple[CognitionObservationRecordV1 | None, bool]:
    """Project one visual directive kind and its strict string values."""

    if not isinstance(row, tuple) or len(row) != 2:
        return None, False
    kind, raw_values = row
    if not isinstance(kind, str) or not isinstance(raw_values, list):
        return None, False
    values, invalid = _string_list(raw_values, limit=_MAX_LIST_ITEMS)
    fields = [_make_field("directive_kind", "Directive kind", kind)]
    if values:
        fields.append(_make_field("values", "Values", values))
    return _make_record(index, "Directive", fields), not invalid


def _messages_section(raw: object) -> CognitionObservationSectionV1:
    """Project live visible messages as ordered strict strings."""

    section = _records_section("surface.visible_messages", raw, _message_row)
    reported = len(raw) if isinstance(raw, list) else section.reported_record_count
    return section.model_copy(update={
        "fields": _fields([
            ("message_count", "Message count", reported),
            ("reported_count", "Reported count", reported),
        ]),
    })


def _message_row(
    row: object,
    index: int,
) -> tuple[CognitionObservationRecordV1 | None, bool]:
    """Project one visible message fragment."""

    if not isinstance(row, str):
        return None, False
    field = _make_field("position", "Position", index)
    text_field = _make_field("text", "Text", row)
    return _make_record(index, "Message", [field, text_field], summary=row), True


def _self_messages_section(
    artifacts: Mapping[str, Any],
    cognition_output: Mapping[str, Any] | None,
    route_effect: Mapping[str, Any],
) -> CognitionObservationSectionV1:
    """Project self visible messages using the exact source precedence."""

    candidates = _artifact_mapping(artifacts, self_cognition_models.ARTIFACT_ACTION_CANDIDATE)
    messages: object = _MISSING
    if candidates:
        if "text" in candidates:
            text = _strict_text(candidates["text"])
            if text:
                messages = [text]
            elif candidates["text"] is not None and not isinstance(
                candidates["text"],
                str,
            ):
                messages = candidates["text"]
            elif "messages" in candidates:
                messages = candidates["messages"]
            else:
                messages = []
        elif "messages" in candidates:
            messages = candidates["messages"]
    if messages is _MISSING and cognition_output:
        messages = cognition_output.get("final_dialog", _MISSING)
    if messages is _MISSING:
        messages = route_effect.get("visible_dialog", _MISSING)
    return _messages_section(messages)


def _self_source_section(cognition_input: Mapping[str, Any]) -> CognitionObservationSectionV1:
    """Project the explicit source-packet fields."""

    packet = cognition_input.get("source_packet", _MISSING)
    if packet is _MISSING:
        return _fields_section("self.source", "not_reported", [])
    if not isinstance(packet, Mapping):
        return _fields_section("self.source", "failed", [])
    fields: list[CognitionObservationFieldV1] = []
    invalid = False
    for source_key, wire_key, label in (
        ("case_name", "source_kind", "Source kind"),
        ("instruction", "summary", "Summary"),
        ("actionability", "reason", "Reason"),
        ("semantic_due_state", "due_state", "Due state"),
    ):
        if source_key not in packet:
            continue
        value = packet[source_key]
        scalar = _strict_scalar(value)
        if scalar is None and value is not None:
            invalid = True
        elif scalar is not None:
            fields.append((wire_key, label, scalar))
    status = "partial" if invalid and fields else "failed" if invalid else (
        "completed" if fields else "empty"
    )
    return _fields_section("self.source", status, fields)


def _self_route_section(
    artifacts: Mapping[str, Any],
    route_effect: Mapping[str, Any],
) -> CognitionObservationSectionV1:
    """Project route decision, reason, and one scalar next topic."""

    run_record = _artifact_mapping(artifacts, self_cognition_models.ARTIFACT_RUN_RECORD)
    if not run_record:
        return _fields_section("self.route", "not_reported", [])
    pairs: list[tuple[str, str, object]] = []
    invalid = False
    for key, label in (("selected_route", "Decision"),):
        if key in run_record:
            pairs.append(("decision", label, run_record[key]))
    if "effect_summary" in route_effect:
        pairs.append(("reason", "Reason", route_effect["effect_summary"]))
    next_topic = route_effect.get("next_topic", _MISSING)
    if isinstance(next_topic, Mapping):
        for key in ("summary", "title", "text", "objective"):
            if key not in next_topic:
                continue
            pairs.append(("next_topic", "Next topic", next_topic[key]))
            break
    elif next_topic is not _MISSING:
        invalid = True
    projected: list[tuple[str, str, object]] = []
    for key, label, value in pairs:
        scalar = _strict_scalar(value)
        if scalar is None and value is not None:
            invalid = True
        elif scalar is not None:
            projected.append((key, label, scalar))
    status = "partial" if invalid and projected else "failed" if invalid else (
        "completed" if projected else "empty"
    )
    return _fields_section("self.route", status, projected)


def _consolidation_section(
    outcome: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> CognitionObservationSectionV1:
    """Project the bounded self-consolidation artifact and flat changes."""

    if not outcome:
        return _fields_section("self.consolidation", "skipped", [])
    fields: list[tuple[str, str, object]] = [("status", "Status", "completed")]
    summary = _strict_text(outcome.get("summary"))
    if summary:
        fields.append(("summary", "Summary", summary))
    changes: list[str] = []
    invalid = False
    for key, value in (
        ("consolidation_called", outcome.get("consolidation_called", _MISSING)),
        ("scheduled_event_count", outcome.get("scheduled_event_count", _MISSING)),
        ("cache_evicted_count", outcome.get("cache_evicted_count", _MISSING)),
    ):
        if value is _MISSING:
            continue
        if key == "consolidation_called" and isinstance(value, bool):
            changes.append(f"{key}={str(value).lower()}")
        elif key != "consolidation_called" and isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            changes.append(f"{key}={value}")
        else:
            invalid = True
    write_success = outcome.get("write_success", _MISSING)
    if write_success is not _MISSING:
        if not isinstance(write_success, Mapping):
            invalid = True
        else:
            for key, value in write_success.items():
                if (
                    isinstance(key, str)
                    and key.isidentifier()
                    and key == key.lower()
                    and isinstance(value, bool)
                    and key not in _PROTECTED_FIELDS
                ):
                    changes.append(f"{key}={str(value).lower()}")
                else:
                    invalid = True
    if changes:
        fields.append(("changes", "Changes", changes))
    status = "partial" if invalid and fields else "failed" if invalid else "completed"
    return _fields_section("self.consolidation", status, fields)


def _not_reported_records_section(section_id: str) -> CognitionObservationSectionV1:
    """Create an explicit unavailable records section."""

    return _records_section(section_id, _MISSING, _progress_row)


def _records_section(
    section_id: str,
    raw: object,
    row_builder,
    *,
    force_status: str | None = None,
    force_invalid: bool = False,
    direct_records: bool = False,
    pair_rows: bool = False,
    status_override: str | None = None,
    header_fields: Sequence[tuple[str, str, object]] = (),
) -> CognitionObservationSectionV1:
    """Build one records section with truthful source and display counts."""

    if raw is _MISSING:
        records: list[CognitionObservationRecordV1] = []
        reported = 0
        invalid = force_invalid
        base_status = "failed" if force_status == "failed" else "not_reported"
    elif direct_records:
        records = list(raw) if isinstance(raw, list) else []
        reported = len(records)
        invalid = force_invalid
        base_status = "completed" if records else "empty"
    elif not isinstance(raw, list):
        records = []
        reported = 0
        invalid = True
        base_status = "failed"
    else:
        records = []
        reported = len(raw)
        invalid = force_invalid
        for index, row in enumerate(raw, 1):
            if pair_rows:
                built, valid = row_builder(row, index)
            else:
                built, valid = row_builder(row, index)
            if built is not None:
                records.append(built)
            invalid = invalid or not valid
        if invalid and records:
            base_status = "partial"
        elif invalid:
            base_status = "failed"
        elif records:
            base_status = "completed"
        else:
            base_status = "empty"
    status = force_status or status_override or base_status
    if status not in _VALID_SECTION_STATUSES:
        status = "failed"
    displayed = records[:_MAX_RECORDS]
    normalized_records = [
        record.model_copy(update={"key": f"item_{index:02d}"})
        for index, record in enumerate(displayed, 1)
    ]
    label, category, _ = _SECTION_META[section_id]
    return CognitionObservationSectionV1(
        section_id=section_id,
        label=label,
        category=category,
        presentation="records",
        status=status,
        summary=_section_summary(section_id, status),
        fields=_fields(header_fields),
        records=normalized_records,
        reported_record_count=reported,
        displayed_record_count=len(normalized_records),
        truncated=reported > len(normalized_records),
    )


def _fields_section(
    section_id: str,
    status: str,
    pairs: Sequence[tuple[str, str, object]],
) -> CognitionObservationSectionV1:
    """Build one fields-only section."""

    label, category, _ = _SECTION_META[section_id]
    normalized_status = status if status in _VALID_SECTION_STATUSES else "failed"
    return CognitionObservationSectionV1(
        section_id=section_id,
        label=label,
        category=category,
        presentation="fields",
        status=normalized_status,
        summary=_section_summary(section_id, normalized_status),
        fields=_fields(pairs),
        records=[],
        reported_record_count=0,
        displayed_record_count=0,
        truncated=False,
    )


def _fields(
    pairs: Sequence[tuple[str, str, object]],
) -> list[CognitionObservationFieldV1]:
    """Build ordered fields from already selected safe values."""

    result: list[CognitionObservationFieldV1] = []
    for key, label, value in pairs:
        if value is _MISSING:
            continue
        safe = _strict_scalar_list(value) if isinstance(value, list) else _strict_scalar(value)
        if safe is None and value is not None:
            continue
        if safe is None:
            continue
        result.append(_make_field(key, label, safe))
    return result


def _make_field(
    key: str,
    label: str,
    value: object,
) -> CognitionObservationFieldV1:
    """Construct one field from a selected safe value."""

    return CognitionObservationFieldV1(key=key, label=label, value=value)


def _make_record(
    index: int,
    label: str,
    fields: Sequence[CognitionObservationFieldV1],
    *,
    summary: str = "",
    key: str | None = None,
) -> CognitionObservationRecordV1:
    """Construct one bounded record with unique source-order identity."""

    record = CognitionObservationRecordV1(
        key=key or f"item_{index:02d}",
        label=label,
        summary=summary[:600],
        fields=list(fields),
    )
    return record


def _project_mapping_fields(
    value: Mapping[str, Any],
    specs: Sequence[tuple[str, str]],
) -> tuple[list[tuple[str, str, object]], bool]:
    """Project explicitly named scalar fields from one mapping."""

    pairs: list[tuple[str, str, object]] = []
    invalid = False
    for key, label in specs:
        if key not in value:
            continue
        scalar = _strict_scalar(value[key])
        if scalar is None and value[key] is not None:
            invalid = True
        elif scalar is not None:
            pairs.append((key, label, scalar))
    return pairs, invalid


def _project_scalar_field(
    row: Mapping[str, Any],
    key: str,
    label: str,
) -> tuple[CognitionObservationFieldV1 | None, bool]:
    """Project one scalar field while distinguishing omitted from invalid."""

    if key not in row:
        return None, False
    value = row[key]
    scalar = _strict_scalar(value)
    if scalar is None and value is not None:
        return None, True
    if scalar is None:
        return None, False
    return _make_field(key, label, scalar), False


def _first_projected_scalar(
    row: Mapping[str, Any],
    keys: Sequence[str],
    label: str,
) -> tuple[CognitionObservationFieldV1 | None, bool]:
    """Project the first present scalar from one explicit fallback list."""

    for key in keys:
        if key not in row:
            continue
        value = row[key]
        scalar = _strict_scalar(value)
        if scalar is None and value is not None:
            return None, True
        if scalar is not None:
            return _make_field("content", label, scalar), False
        return None, False
    return None, False


def _axis_changes(value: object) -> tuple[list[str], bool]:
    """Flatten only axis, shift, and optional safe reason values."""

    if not isinstance(value, list):
        return [], value is not None
    result: list[str] = []
    invalid = False
    for row in value:
        if not isinstance(row, Mapping):
            invalid = True
            continue
        axis = _strict_text(row.get("axis"))
        shift = _strict_scalar(row.get("shift"))
        reason = _strict_text(row.get("reason"))
        if not axis or shift is None:
            invalid = True
            continue
        text = f"{axis}: {shift}"
        if reason:
            text += f" — {reason}"
        result.append(text[:_MAX_LIST_TEXT])
    return result, invalid


def _evidence_row_summary(row: Mapping[str, Any]) -> str:
    """Return the first explicit evidence summary."""

    return _first_text(row, _EVIDENCE_FIELDS) or ""


def _first_text(value: Mapping[str, Any], keys: Sequence[str]) -> str:
    """Return the first bounded strict text from explicit keys."""

    for key in keys:
        text = _strict_text(value.get(key))
        if text:
            return text
    return ""


def _strict_text(value: object, *, maximum: int = _MAX_TEXT) -> str | None:
    """Accept only strict strings and bound semantic text."""

    if not isinstance(value, str):
        return None
    return value.strip()[:maximum]


def _strict_scalar(value: object) -> str | int | float | bool | None:
    """Accept one finite strict JSON scalar."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _strict_text(value)
    return None


def _strict_scalar_list(value: object) -> list[str | int | float | bool | None] | None:
    """Accept only one-level lists of finite strict scalars."""

    if not isinstance(value, list) or len(value) > _MAX_LIST_ITEMS:
        return None
    result: list[str | int | float | bool | None] = []
    for item in value:
        scalar = _strict_scalar(item)
        if scalar is None and item is not None:
            return None
        if isinstance(scalar, str) and len(scalar) > _MAX_LIST_TEXT:
            return None
        result.append(scalar)
    return result


def _string_list(value: object, *, limit: int) -> tuple[list[str], bool]:
    """Project a bounded strict string list and report invalid items."""

    if value is _MISSING:
        return [], False
    if not isinstance(value, list):
        return [], True
    result: list[str] = []
    invalid = False
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip()[:_MAX_LIST_TEXT])
        else:
            invalid = True
    return result[:limit], invalid


def _finite_number(value: object) -> bool:
    """Return whether a value is a non-Boolean finite number."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    )


def _is_lower_snake_key(value: str) -> bool:
    """Return whether a source key uses the contract lower-snake grammar."""

    return re.fullmatch(r"[a-z][a-z0-9_]*", value) is not None


def _section_summary(section_id: str, status: str) -> str:
    """Produce a bounded producer-owned section summary."""

    if status in {"empty", "skipped", "not_reported"}:
        return ""
    return _SECTION_META[section_id][0]


def _aggregate_status(statuses: Sequence[str] | Any) -> str:
    """Aggregate dispositions using the fixed node priority."""

    values = set(statuses)
    for status in _STATUS_PRIORITY:
        if status in values:
            return status
    return "not_reported"


def _terminal_observation_status(
    terminal_status: str,
    sections: Sequence[CognitionObservationSectionV1],
) -> str:
    """Map an episode terminal status and component failures to v1 status."""

    if terminal_status == "failed":
        return "failed"
    component_partial = any(
        section.status in {"failed", "partial"}
        for section in sections
    )
    if component_partial:
        return "partial"
    return "completed"


def _core_source(
    graph_result: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, bool]:
    """Locate and validate the live cognition core output."""

    raw = state.get("cognition_core_output", graph_result.get("cognition_core_output", _MISSING))
    if raw is _MISSING:
        return None, True
    if not isinstance(raw, Mapping):
        return None, False
    return raw, raw.get("schema_version") == "cognition_output.v3"


def _self_core_source(
    wrapper: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, bool]:
    """Locate the canonical core object inside a self artifact wrapper."""

    if not wrapper:
        return None, True
    raw = wrapper.get("cognition_core_output", wrapper)
    if not isinstance(raw, Mapping):
        return None, False
    if not raw and "cognition_core_output" not in wrapper:
        return None, True
    return raw, raw.get("schema_version") == "cognition_output.v3"


def _self_progress_source(cognition_input: Mapping[str, Any]) -> object:
    """Read the self source-packet progress artifact explicitly."""

    source_packet = cognition_input.get("source_packet", _MISSING)
    if not isinstance(source_packet, Mapping):
        return _MISSING
    return source_packet.get("conversation_progress", _MISSING)


def _artifact_mapping(
    payloads: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Read one optional self artifact mapping."""

    value = payloads.get(key)
    return value if isinstance(value, Mapping) else {}


def _nested_mapping(value: object, key: str) -> Mapping[str, Any]:
    """Read one explicit nested mapping without shape inference."""

    if not isinstance(value, Mapping):
        return {}
    nested = value.get(key)
    return nested if isinstance(nested, Mapping) else {}


def _nested_value(value: object, key: str) -> object:
    """Read one explicit nested source value, preserving absence."""

    if not isinstance(value, Mapping):
        return _MISSING
    return value.get(key, _MISSING)


def _bounded_identifier(value: object) -> str | None:
    """Return a bounded strict correlation identifier or ``None``."""

    return _optional_identifier(value)


def _optional_identifier(value: object) -> str | None:
    """Validate an optional bounded strict identifier."""

    text = _strict_text(value, maximum=120)
    return text or None


def _visual_enabled(state: Mapping[str, Any]) -> bool:
    """Apply the global and per-run visual directive gate."""

    if not COGNITION_VISUAL_DIRECTIVES_ENABLED:
        return False
    for candidate in (
        state.get("debug_modes"),
        _nested_value(state.get("cognitive_episode"), "origin_metadata"),
    ):
        if not isinstance(candidate, Mapping):
            continue
        modes = candidate.get("debug_modes", candidate)
        if isinstance(modes, Mapping) and modes.get("no_visual_directives") is True:
            return False
    return True


def _visual_raw(state: Mapping[str, Any]) -> object:
    """Read visual directives from the two explicit state locations."""

    action_directives = state.get("action_directives")
    if isinstance(action_directives, Mapping) and "visual_directives" in action_directives:
        return action_directives["visual_directives"]
    if "visual_directives" in state:
        return state["visual_directives"]
    return _MISSING


def _live_edges() -> tuple[tuple[str, str, str], ...]:
    """Return the canonical live sequence and reference edges."""

    return (
        ("input.turn", "decision.response", "sequence"),
        ("decision.response", "cognition.meaning", "sequence"),
        ("cognition.meaning", "cognition.goal", "sequence"),
        ("cognition.goal", "cognition.response", "sequence"),
        ("cognition.response", "action.results", "sequence"),
        ("evidence.memory", "cognition.meaning", "reference"),
        ("cognition.response", "cognition.affect", "reference"),
        ("cognition.response", "reasoning.context", "reference"),
        ("cognition.response", "surface.visual", "reference"),
        ("reasoning.context", "surface.visual", "reference"),
        ("evidence.memory", "surface.visual", "reference"),
        ("action.results", "surface.visual", "reference"),
        ("cognition.response", "surface.visible", "reference"),
        ("reasoning.context", "surface.visible", "reference"),
        ("evidence.memory", "surface.visible", "reference"),
        ("action.results", "surface.visible", "reference"),
    )


def _self_edges() -> tuple[tuple[str, str, str], ...]:
    """Return the canonical self sequence and reference edges."""

    return (
        ("self.source", "cognition.meaning", "sequence"),
        ("cognition.meaning", "cognition.goal", "sequence"),
        ("cognition.goal", "cognition.response", "sequence"),
        ("cognition.response", "self.route", "sequence"),
        ("self.route", "action.results", "sequence"),
        ("action.results", "self.consolidation", "sequence"),
        ("evidence.memory", "cognition.meaning", "reference"),
        ("cognition.response", "cognition.affect", "reference"),
        ("cognition.response", "reasoning.context", "reference"),
        ("self.route", "surface.visual", "reference"),
        ("self.route", "surface.visible", "reference"),
        ("surface.visual", "self.consolidation", "reference"),
        ("surface.visible", "self.consolidation", "reference"),
    )
