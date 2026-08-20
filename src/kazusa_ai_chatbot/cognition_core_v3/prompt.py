"""Canonical dynamic question packets for the Cognition V3 chain."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    build_goal_output_contract,
)
from kazusa_ai_chatbot.cognition_core_v3.goal_cognition import (
    GOAL_BID_EVIDENCE_HANDLE_LIMIT,
    ORDINARY_GOAL_KIND,
)
from kazusa_ai_chatbot.cognition_core_v3.registry import APPRAISAL_GROUPING_MAPS
from kazusa_ai_chatbot.cognition_episode import project_model_visible_percepts

APPRAISAL_QUESTION_GUIDANCE = '''按 `semantic_appraisal_group.v1` 检查 payload 中依次列出的语义问题；逐项使用各自允许的证据、角色和状态路径，返回同键同序的微评估列表。'''

ORDINARY_GOAL_BID_QUESTION_GUIDANCE = '''按 `ordinary_goal_bid.v1` 使用已接受判断、确定性通知和本条 payload 的目标专属语境，返回角色此刻愿意推进的普通目标及关系立场。'''

ACTIVE_GOAL_BID_GROUP_QUESTION_GUIDANCE = '''按 `active_goal_bid_group.v1` 严格依照 payload 的分支名册顺序，为每个分支返回一个完整目标；不得预选胜者、改变名册或输出排名。'''

WORKSPACE_QUESTION_GUIDANCE = '''按 `workspace_partition.v1` 将已存在的完整目标句柄恰好分成主目标、支持目标和抑制目标；不得改写、合并或补造目标。'''

ACTION_PLAN_QUESTION_GUIDANCE = '''按 `action_plan.v1` 把已选目标转成当前可表达意图以及有依据的行动或信息请求；只用 payload 提供的能力与句柄，不声称未发生的执行。'''

CHAIN_QUESTION_POINTERS = MappingProxyType(
    {
        "semantic_appraisal_group.v1": APPRAISAL_QUESTION_GUIDANCE,
        "ordinary_goal_bid.v1": ORDINARY_GOAL_BID_QUESTION_GUIDANCE,
        "active_goal_bid_group.v1": (
            ACTIVE_GOAL_BID_GROUP_QUESTION_GUIDANCE
        ),
        "workspace_partition.v1": WORKSPACE_QUESTION_GUIDANCE,
        "action_plan.v1": ACTION_PLAN_QUESTION_GUIDANCE,
    }
)
CHAIN_CONTRACT_NAMES = tuple(CHAIN_QUESTION_POINTERS)
RUNTIME_PROMPT_TEXTS = tuple(CHAIN_QUESTION_POINTERS.values())

_EVALUATION_METADATA_FIELDS = frozenset(
    {
        "behavior_contract",
        "case_id",
        "expected_answer",
        "expected_decision",
        "fixture_hash",
        "fixture_id",
        "manifest_id",
        "pytest_node_id",
        "rubric",
        "rubric_dimension",
        "score",
        "trial_id",
    }
)
_GOAL_ONLY_FIELDS = frozenset(
    {
        "branch_intent_guidance",
        "character_sleep_phase",
        "group_engagement_action_context",
        "past_dialog_cognition_context",
        "private_continuity_context",
        "required_selection_operation_registry",
    }
)
_PRIVATE_RUNTIME_FIELDS = frozenset(
    {
        "active_turn_conversation_row_ids",
        "active_turn_platform_message_ids",
        "action_latch_ref",
        "api_key",
        "base_url",
        "calendar_event_id",
        "claim_id",
        "cognition_invocation_id",
        "context_window_tokens",
        "correlation_id",
        "current_global_user_id",
        "current_platform_user_id",
        "delivery_permission_ref",
        "entity_id",
        "episode_id",
        "evidence_id",
        "global_user_id",
        "invocation_id",
        "model",
        "origin_metadata",
        "owner_user_id",
        "permission_ref",
        "platform",
        "platform_channel_id",
        "platform_message_id",
        "platform_user_id",
        "prior_episode_id",
        "relationship_id",
        "result_ref",
        "route_name",
        "run_id",
        "source_case_ref",
        "source_id",
        "source_message_id",
        "stage_name",
        "target_addressed_user_ids",
        "target_scope",
        "target_scope_ref",
        "task_id",
        "trace_id",
        "user_id",
    }
)
_FIRST_PACKET_SECTION_FIELDS = MappingProxyType(
    {
        "constraints_and_operational_state": frozenset(
            {
                "character_constraints",
                "character_operational_context",
            }
        ),
        "relationship_and_mutable_state": frozenset(
            {
                "relationship",
                "mutable_state",
            }
        ),
        "episode_and_scene": frozenset(
            {
                "episode",
                "scene_context",
            }
        ),
        "evidence_and_affordances": frozenset(
            {
                "evidence",
                "direct_facts",
                "available_actions",
                "available_resolver_capabilities",
                "resolver_context",
            }
        ),
    }
)
_MUTABLE_STATE_FIELDS = frozenset(
    {
        "goals",
        "threats",
        "events",
        "knowledge_gaps",
        "affect",
        "causal_candidates",
    }
)
_EPISODE_FIELDS = frozenset(
    {
        "episode_ref",
        "trigger_source",
        "visible_percepts",
    }
)
_SCENE_CONTEXT_FIELDS = frozenset(
    {
        "channel_scope",
        "character_role",
        "current_user_role",
        "semantic_scene",
        "public_group_scene",
        "conversation_continuity",
        "semantic_temporal_context",
        "participant_bindings",
    }
)
_CURRENT_EPISODE_REF = "current_cognitive_episode"
_L1_AFFECT_BAND_CAP = 4
_L1_BOUNDARY_FIELD_CAP = 8


class PromptContractError(ValueError):
    """A dynamic packet violates chain scope or evaluation integrity."""


def _reject_fields(
    value: object,
    *,
    forbidden_fields: frozenset[str],
    error_label: str,
) -> None:
    """Reject closed structural field names recursively without reading prose."""

    if isinstance(value, Mapping):
        for field_name, nested_value in value.items():
            if not isinstance(field_name, str):
                raise PromptContractError(
                    "Prompt packet mapping keys must be strings"
                )
            if field_name in forbidden_fields:
                raise PromptContractError(
                    f"Prompt packet contains {error_label} field {field_name!r}"
                )
            _reject_fields(
                nested_value,
                forbidden_fields=forbidden_fields,
                error_label=error_label,
            )
        return
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for nested_value in value:
            _reject_fields(
                nested_value,
                forbidden_fields=forbidden_fields,
                error_label=error_label,
            )


def _sanitize_prompt_value(value: object) -> object:
    """Remove evaluation and private runtime metadata recursively."""

    forbidden = {
        "action_latch_ref", "active_turn_conversation_row_ids",
        "active_turn_platform_message_ids", "api_key", "base_url",
        "calendar_event_id", "claim_id", "cognition_invocation_id",
        "context_window_tokens", "correlation_id", "current_global_user_id",
        "current_platform_user_id", "delivery_permission_ref", "entity_id",
        "episode_id", "evidence_id", "global_user_id", "invocation_id",
        "model", "origin_metadata", "owner_user_id", "permission_ref",
        "platform", "platform_channel_id", "platform_message_id",
        "platform_user_id", "prior_episode_id", "relationship_id",
        "result_ref", "route_name", "run_id", "source_case_ref",
        "source_id", "source_message_id", "stage_name",
        "target_addressed_user_ids", "target_scope", "target_scope_ref",
        "task_id", "trace_id", "user_id",
    }
    if isinstance(value, Mapping):
        return {
            field_name: _sanitize_prompt_value(nested_value)
            for field_name, nested_value in value.items()
            if field_name not in forbidden
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_prompt_value(nested_value) for nested_value in value]
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    *,
    expected_fields: frozenset[str],
    error_label: str,
) -> None:
    """Require one prompt carrier to expose its complete fixed field set."""

    if set(value) != expected_fields:
        raise PromptContractError(
            f"{error_label} requires exact fields "
            f"{sorted(expected_fields)!r}"
        )


def _validate_first_packet_carriers(
    sections: Sequence[tuple[str, Mapping[str, object]]],
) -> None:
    """Validate exact prompt-safe carriers for the first chain request."""

    section_values = dict(sections)
    for section_name, section_value in sections:
        _require_exact_fields(
            section_value,
            expected_fields=_FIRST_PACKET_SECTION_FIELDS[section_name],
            error_label=f"First user packet section {section_name!r}",
        )

    constraints = section_values["constraints_and_operational_state"]
    if not isinstance(constraints["character_constraints"], Mapping):
        raise PromptContractError(
            "First user packet character_constraints must be a mapping"
        )
    if not isinstance(
        constraints["character_operational_context"],
        Mapping,
    ):
        raise PromptContractError(
            "First user packet character_operational_context must be a mapping"
        )

    relationship_state = section_values[
        "relationship_and_mutable_state"
    ]
    if not isinstance(relationship_state["relationship"], Mapping):
        raise PromptContractError(
            "First user packet relationship must be a mapping"
        )
    mutable_state = relationship_state["mutable_state"]
    if not isinstance(mutable_state, Mapping):
        raise PromptContractError(
            "First user packet mutable_state must be a mapping"
        )
    _require_exact_fields(
        mutable_state,
        expected_fields=_MUTABLE_STATE_FIELDS,
        error_label="First user packet mutable_state",
    )
    if any(not isinstance(mutable_state[field_name], list) for field_name in (
        "goals",
        "threats",
        "events",
        "knowledge_gaps",
        "affect",
        "causal_candidates",
    )):
        raise PromptContractError(
            "First user packet mutable_state fields must be lists"
        )

    episode_scene = section_values["episode_and_scene"]
    episode = episode_scene["episode"]
    if not isinstance(episode, Mapping):
        raise PromptContractError(
            "First user packet episode must be a mapping"
        )
    _require_exact_fields(
        episode,
        expected_fields=_EPISODE_FIELDS,
        error_label="First user packet episode",
    )
    if episode["episode_ref"] != _CURRENT_EPISODE_REF:
        raise PromptContractError(
            "First user packet episode_ref must use the stable current episode alias"
        )
    if (
        not isinstance(episode["trigger_source"], str)
        or not episode["trigger_source"].strip()
    ):
        raise PromptContractError(
            "First user packet trigger_source must be non-empty text"
        )
    if not isinstance(episode["visible_percepts"], list):
        raise PromptContractError(
            "First user packet visible_percepts must be a list"
        )

    scene_context = episode_scene["scene_context"]
    if not isinstance(scene_context, Mapping):
        raise PromptContractError(
            "First user packet scene_context must be a mapping"
        )
    _require_exact_fields(
        scene_context,
        expected_fields=_SCENE_CONTEXT_FIELDS,
        error_label="First user packet scene_context",
    )
    if any(not isinstance(scene_context[field_name], str) for field_name in (
        "channel_scope",
        "character_role",
        "current_user_role",
        "semantic_scene",
        "public_group_scene",
        "conversation_continuity",
        "semantic_temporal_context",
    )):
        raise PromptContractError(
            "First user packet scene_context text fields must be strings"
        )
    if not isinstance(scene_context["participant_bindings"], list):
        raise PromptContractError(
            "First user packet participant_bindings must be a list"
        )

    evidence_affordances = section_values["evidence_and_affordances"]
    for field_name in (
        "evidence",
        "direct_facts",
        "available_actions",
        "available_resolver_capabilities",
    ):
        if not isinstance(evidence_affordances[field_name], list):
            raise PromptContractError(
                f"First user packet {field_name} must be a list"
            )
    if not isinstance(evidence_affordances["resolver_context"], str):
        raise PromptContractError(
            "First user packet resolver_context must be a string"
        )


@dataclass(frozen=True)
class ChainQuestion:
    """One registered semantic contract pointer and its current-run payload."""

    contract_name: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        """Validate the closed contract identity and dynamic payload shape."""

        if (
            not isinstance(self.contract_name, str)
            or self.contract_name not in CHAIN_QUESTION_POINTERS
        ):
            raise PromptContractError(
                "Chain question contract_name must be registered"
            )
        if not isinstance(self.payload, Mapping):
            raise PromptContractError(
                "Chain question payload must be a mapping"
            )
        _reject_fields(
            self.payload,
            forbidden_fields=_EVALUATION_METADATA_FIELDS,
            error_label="evaluation metadata",
        )
        _reject_fields(
            self.payload,
            forbidden_fields=_PRIVATE_RUNTIME_FIELDS,
            error_label="private metadata",
        )


def build_first_user_message(
    *,
    constraints_and_operational_state: Mapping[str, object],
    relationship_and_mutable_state: Mapping[str, object],
    episode_and_scene: Mapping[str, object],
    evidence_and_affordances: Mapping[str, object],
    question: ChainQuestion,
) -> str:
    """Render the four volatility-ordered turn sections plus first question.

    Args:
        constraints_and_operational_state: Slow-changing constraints and
            prompt-safe operational character context.
        relationship_and_mutable_state: Current qualitative relationship and
            mutable cognition projection.
        episode_and_scene: Current episode and semantic scene projection.
        evidence_and_affordances: Ordered evidence, facts, and available
            action/resolver capability projections.
        question: First model-owned registered contract and dynamic payload.

    Returns:
        Compact canonical JSON preserving the required semantic section order.

    Raises:
        PromptContractError: Non-mapping sections, goal-only carriers, or
            evaluation-only structural metadata fail before model invocation.
    """

    named_sections = (
        ("constraints_and_operational_state", constraints_and_operational_state),
        ("relationship_and_mutable_state", relationship_and_mutable_state),
        ("episode_and_scene", episode_and_scene),
        ("evidence_and_affordances", evidence_and_affordances),
    )
    for section_name, section_value in named_sections:
        if not isinstance(section_value, Mapping):
            raise PromptContractError(
                f"First user packet section {section_name!r} must be a mapping"
            )
        _reject_fields(
            section_value,
            forbidden_fields=_EVALUATION_METADATA_FIELDS,
            error_label="evaluation metadata",
        )
        _reject_fields(
            section_value,
            forbidden_fields=_GOAL_ONLY_FIELDS,
            error_label="goal-only",
        )
        _reject_fields(
            section_value,
            forbidden_fields=_PRIVATE_RUNTIME_FIELDS,
            error_label="private metadata",
        )

    _validate_first_packet_carriers(named_sections)
    message_sections = [
        {section_name: dict(section_value)}
        for section_name, section_value in named_sections
    ]
    message_sections.append(
        {
            "question": {
                "contract_name": question.contract_name,
                "instruction": CHAIN_QUESTION_POINTERS[
                    question.contract_name
                ],
                "payload": dict(question.payload),
            }
        }
    )
    first_user_message = json.dumps(
        message_sections,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return first_user_message


def build_question_message(
    question: ChainQuestion,
    *,
    interludes: Sequence[Mapping[str, object]] = (),
) -> str:
    """Render later interlude notices and one question as a single user row."""

    interlude_rows: list[dict[str, object]] = []
    for interlude in interludes:
        if not isinstance(interlude, Mapping):
            raise PromptContractError(
                "Deterministic interlude notices must be mappings"
            )
        _reject_fields(
            interlude,
            forbidden_fields=_EVALUATION_METADATA_FIELDS,
            error_label="evaluation metadata",
        )
        _reject_fields(
            interlude,
            forbidden_fields=_PRIVATE_RUNTIME_FIELDS,
            error_label="private metadata",
        )
        interlude_rows.append(dict(interlude))

    message_sections: list[dict[str, object]] = []
    if interlude_rows:
        message_sections.append({"interludes": interlude_rows})
    message_sections.append(
        {
            "question": {
                "contract_name": question.contract_name,
                "instruction": CHAIN_QUESTION_POINTERS[
                    question.contract_name
                ],
                "payload": dict(question.payload),
            }
        }
    )
    question_message = json.dumps(
        message_sections,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return question_message


__all__ = [
    "ACTION_PLAN_QUESTION_GUIDANCE",
    "ACTIVE_GOAL_BID_GROUP_QUESTION_GUIDANCE",
    "APPRAISAL_QUESTION_GUIDANCE",
    "CHAIN_CONTRACT_NAMES",
    "CHAIN_QUESTION_POINTERS",
    "ORDINARY_GOAL_BID_QUESTION_GUIDANCE",
    "RUNTIME_PROMPT_TEXTS",
    "WORKSPACE_QUESTION_GUIDANCE",
    "ChainQuestion",
    "PromptContractError",
    "build_first_user_message",
    "build_question_message",
]

def build_first_packet_sections(
    *,
    projection_payload: Mapping[str, Any],
    scene_context: Mapping[str, Any],
    episode: Mapping[str, Any],
    direct_facts: Sequence[Mapping[str, object]],
    available_actions: Sequence[Mapping[str, object]],
    available_resolver_capabilities: Sequence[Mapping[str, object]],
    resolver_context: str,
) -> tuple[Mapping[str, object], ...]:
    if not isinstance(projection_payload, Mapping):
        raise PromptContractError("projection payload must be a mapping")
    if not isinstance(scene_context, Mapping):
        raise PromptContractError("scene context must be a mapping")
    if not isinstance(episode, Mapping):
        raise PromptContractError("episode must be a mapping")
    mutable_state = {
        field_name: list(projection_payload.get(field_name, []))
        for field_name in (
            "goals", "threats", "events", "knowledge_gaps", "affect", "causal_candidates",
        )
    }
    relationship = projection_payload.get("relationship")
    if not isinstance(relationship, Mapping):
        relationship = {
            "handle": "r1",
            "axes": {},
            "causal_context": [],
            "affect": [],
            "relationship_freshness": "",
            "evidence_freshness": "",
        }
    visible_percepts = project_model_visible_percepts(episode)
    scene_section = {
        field_name: scene_context.get(field_name, "")
        for field_name in (
            "channel_scope", "character_role", "current_user_role", "semantic_scene",
            "public_group_scene", "conversation_continuity", "semantic_temporal_context",
        )
    }
    scene_section["participant_bindings"] = []
    evidence_rows = list(projection_payload.get("evidence", []))
    normalized_evidence = [
        {
            "handle": row.get("handle", ""),
            "source_kind": row.get("source_kind", "unknown"),
            "semantic_summary": row.get("semantic_summary", ""),
        }
        for row in evidence_rows
        if isinstance(row, Mapping)
    ]
    sections = (
        {
            "character_constraints": dict(projection_payload.get("character_constraints", {})),
            "character_operational_context": dict(projection_payload.get("character_operational_context", {})),
        },
        {
            "relationship": dict(relationship),
            "mutable_state": mutable_state,
        },
        {
            "episode": {
                "episode_ref": "current_cognitive_episode",
                "trigger_source": str(episode.get("trigger_source", "user_message")),
                "visible_percepts": list(visible_percepts),
            },
            "scene_context": scene_section,
        },
        {
            "evidence": normalized_evidence,
            "direct_facts": list(direct_facts),
            "available_actions": list(available_actions),
            "available_resolver_capabilities": list(available_resolver_capabilities),
            "resolver_context": resolver_context,
        },
    )
    return tuple(_sanitize_prompt_value(section) for section in sections)


def build_l1_subconscious_packet(
    *,
    episode: Mapping[str, object],
    affect_bands: Sequence[Mapping[str, object]],
    boundary_summary: Mapping[str, object],
    supplied_evidence_handles: Sequence[str],
) -> str:
    """Render the bounded fresh packet consumed by the advisory L1 stage.

    The packet intentionally excludes chain history, semantic evidence text,
    goals, and relationship state. It preserves only the current model-visible
    percept wording, qualitative affect descriptors, compact boundary language,
    and the handles L1 may cite as advisory salience.

    Args:
        episode: Canonical episode projected without deterministic identifiers.
        affect_bands: Prompt-safe qualitative affect projections.
        boundary_summary: Prompt-safe boundary descriptors for the active
            cognition identity.
        supplied_evidence_handles: Exact evidence handles available this turn.

    Returns:
        One compact JSON object for the independent L1 sidecar boundary.
    """

    visible_percepts = project_model_visible_percepts(episode)
    current_percept_text = ""
    for percept in visible_percepts:
        content = percept.get("content")
        if not isinstance(content, Mapping):
            continue
        semantic_text = content.get("semantic_text")
        if not isinstance(semantic_text, str):
            semantic_text = content.get("text")
        if isinstance(semantic_text, str) and semantic_text:
            current_percept_text = semantic_text
            break

    qualitative_affect_bands = [
        {
            field_name: band[field_name]
            for field_name in ("emotion", "phase", "intensity", "trend")
            if field_name in band
        }
        for band in affect_bands[:_L1_AFFECT_BAND_CAP]
    ]
    bounded_boundary_summary = {
        field_name: boundary_summary[field_name]
        for field_name in sorted(boundary_summary)[:_L1_BOUNDARY_FIELD_CAP]
    }
    packet = {
        "current_percept_text": current_percept_text,
        "qualitative_affect_bands": qualitative_affect_bands,
        "boundary_summary": bounded_boundary_summary,
        "supplied_evidence_handles": list(supplied_evidence_handles),
    }
    rendered_packet = json.dumps(
        packet,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return rendered_packet


def build_appraisal_question_payload(*, questions, l1_residue=None):
    rows = []
    for question in questions:
        if not isinstance(question, Mapping):
            raise PromptContractError("appraisal questions must be mappings")
        rows.append({
            "family": question.get("question_kind", ""),
            "evidence_handles": list(question.get("evidence_handles", [])),
            "permitted_delta_paths": list(question.get("permitted_delta_paths", [])),
            "semantic_question": question.get("semantic_question", ""),
        })
    if not rows:
        raise PromptContractError("grouped appraisal requires at least one planned question")
    return {"questions": rows, "l1_residue": dict(l1_residue or {})}


def build_grouped_appraisal_questions(*, planned_questions, group_count, l1_residue=None):
    groups = APPRAISAL_GROUPING_MAPS.get(group_count)
    if groups is None:
        raise PromptContractError("appraisal group_count must be one of 1, 2, 3, or 6")
    by_family = {
        str(question.get("question_kind")): question
        for question in planned_questions
        if isinstance(question, Mapping) and isinstance(question.get("question_kind"), str)
    }
    result = []
    for step_id, family_names in groups:
        grouped = [by_family[family_name] for family_name in family_names if family_name in by_family]
        if not grouped:
            continue
        payload = build_appraisal_question_payload(questions=grouped, l1_residue=l1_residue)
        result.append(ChainQuestion(contract_name="semantic_appraisal_group.v1", payload=payload))
    return result


def build_goal_question_payload(*, goal_kind, goal_projection, evidence_handles, action_tendencies,
                                branch_intent_guidance, role_bindings, role_summaries, semantic_context,
                                appraisal_summaries, evidence_rows, selection_operations, progress_evidence,
                                carried_relational_willingness=None, l1_residue=None):
    selection_required = bool(selection_operations)
    required_evidence_handles = {
        operation["evidence_handle"]
        for operation in selection_operations
    }
    require_relational_willingness = (
        goal_kind == ORDINARY_GOAL_KIND
        and carried_relational_willingness is None
    )
    payload = {
        "goal_kind": goal_kind,
        "goal_projection": dict(goal_projection),
        "evidence_handles": list(evidence_handles),
        "action_tendencies": list(action_tendencies),
        "branch_intent_guidance": branch_intent_guidance,
        "role_bindings": dict(role_bindings),
        "role_summaries": dict(role_summaries),
        "semantic_context": dict(semantic_context),
        "appraisal_summaries": list(appraisal_summaries),
        "evidence_rows": [dict(row) for row in evidence_rows],
        "selection_operations": [dict(row) for row in selection_operations],
        "progress_evidence": [dict(row) for row in progress_evidence],
        "goal_output_contract": build_goal_output_contract(
            evidence_handles=set(evidence_handles),
            episode_evidence_handles=required_evidence_handles,
            required_evidence_handles=required_evidence_handles,
            role_bindings=role_bindings,
            selection_required=selection_required,
            require_relational_willingness=require_relational_willingness,
            maximum_evidence_handles=max(
                GOAL_BID_EVIDENCE_HANDLE_LIMIT,
                len(required_evidence_handles),
            ),
            authoritative_operation=(
                selection_operations[0]["response_operation"]
                if selection_required
                else None
            ),
            recurrence_relational_willingness=(
                carried_relational_willingness is not None
            ),
        ),
    }
    if carried_relational_willingness is not None:
        payload["carried_relational_willingness"] = dict(
            carried_relational_willingness
        )
    if l1_residue is not None:
        payload["l1_residue"] = dict(l1_residue)
    return payload


def build_active_goal_group_question_payload(*, roster, evidence_handles, semantic_context):
    return {
        "branch_roster": [dict(row) for row in roster],
        "evidence_handles": list(evidence_handles),
        "semantic_context": dict(semantic_context),
    }


def build_workspace_question_payload(*, bids, current_event, goal_contexts):
    return {
        "bids": [dict(row) for row in bids],
        "current_event": dict(current_event) if isinstance(current_event, Mapping) else {},
        "goal_contexts": dict(goal_contexts),
    }


def build_action_plan_question_payload(*, primary_bid, supporting_bids, available_actions,
                                       available_resolvers, resolver_context, runtime_capability_limits,
                                       current_goal_progress, required_resolver_evidence_dependency):
    payload = {
        "primary_bid": dict(primary_bid) if primary_bid is not None else None,
        "supporting_bids": [dict(row) for row in supporting_bids],
        "available_actions": [dict(row) for row in available_actions],
        "available_resolvers": [dict(row) for row in available_resolvers],
        "resolver_context": resolver_context,
        "runtime_capability_limits": list(runtime_capability_limits),
        "current_goal_progress": dict(current_goal_progress) if current_goal_progress is not None else None,
        "required_resolver_evidence_dependency": dict(required_resolver_evidence_dependency) if required_resolver_evidence_dependency is not None else None,
    }
    return _sanitize_prompt_value(payload)


def build_serial_question_sequence(*, planned_questions, group_count, ordinary_goal_payload,
                                   active_branch_roster, workspace_payload, action_plan_payload):
    sequence = []
    appraisal_questions = build_grouped_appraisal_questions(planned_questions=planned_questions, group_count=group_count)
    appraisal_step_ids = [step_id for step_id, _families in APPRAISAL_GROUPING_MAPS[group_count]]
    for step_id, appraisal_question in zip(appraisal_step_ids, appraisal_questions):
        sequence.append((step_id, appraisal_question))
    sequence.append(("G1a", ChainQuestion(contract_name="ordinary_goal_bid.v1", payload=dict(ordinary_goal_payload))))
    if active_branch_roster:
        sequence.append(("G1b", ChainQuestion(contract_name="active_goal_bid_group.v1", payload={"branch_roster": [dict(row) for row in active_branch_roster]})))
    if workspace_payload:
        sequence.append(("W1", ChainQuestion(contract_name="workspace_partition.v1", payload=dict(workspace_payload))))
    sequence.append(("P1", ChainQuestion(contract_name="action_plan.v1", payload=dict(action_plan_payload))))
    return sequence
