"""Canonical dynamic question packets for the Cognition V3 chain."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from kazusa_ai_chatbot.cognition_shared.contracts import (
    GOAL_RESOLUTION_VALUES,
    SCHEDULED_AUTHORITY_CURRENT_ROLE_VALUES,
    SCHEDULED_AUTHORITY_PROPOSAL_SCHEMA_VERSION,
    SELF_COGNITION_RESPONSE_DECISION_VALUES,
    SELF_COGNITION_RESPONSE_PARTICIPATION_VALUES,
    TEMPORAL_ALIGNMENT_VALUES,
    project_evidence_provenance_role,
)
from kazusa_ai_chatbot.cognition_core_v3.goal_cognition import (
    build_goal_output_contract,
)
from kazusa_ai_chatbot.cognition_core_v3.semantic_appraisal import (
    DELTA_LIMIT_NARROW,
    DELTA_LIMIT_WIDE,
)
from kazusa_ai_chatbot.cognition_core_v3.semantic_source_planner import (
    question_proposition_kind_semantics,
    question_proposition_kinds,
)
from kazusa_ai_chatbot.cognition_core_v3.goal_cognition import (
    GOAL_BID_EVIDENCE_HANDLE_LIMIT,
    ORDINARY_GOAL_KIND,
)
from kazusa_ai_chatbot.cognition_core_v3.registry import (
    APPRAISAL_STAGE_FAMILIES,
)
from kazusa_ai_chatbot.cognition_core_v3.workspace import prepare_partition
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    project_model_visible_percepts,
)

APPRAISAL_FAMILY_OBJECT_FIELDS = ("propositions", "deltas")
APPRAISAL_PROPOSITION_FIELDS = (
    "proposition_kind",
    "subject_handle",
    "evidence_handles",
    "role_assignments",
    "semantic_value",
)
APPRAISAL_DELTA_FIELDS = (
    "target_path",
    "delta",
    "evidence_handles",
    "reason",
)
APPRAISAL_ROLE_VALUES = (
    "actor",
    "experiencer",
    "target",
    "object",
    "affected_goal",
    "affected_relationship",
)
SELF_COGNITION_RESPONSE_REQUIRED_FIELDS = (
    "decision",
    "evidence_handles",
    "semantic_target_handle",
    "participation_basis",
    "response_goal",
    "reason",
)


def _appraisal_delta_bounds(paths: Sequence[str]) -> list[dict[str, object]]:
    """Describe canonical V2 numeric bounds beside each writable path."""

    bounds: list[dict[str, object]] = []
    for path in paths:
        state_field = path.split(".", maxsplit=1)[0]
        limit = (
            DELTA_LIMIT_NARROW
            if state_field in {"relationship", "meaning_state"}
            else DELTA_LIMIT_WIDE
        )
        bounds.append({
            "path": path,
            "minimum": -limit,
            "maximum": limit,
        })
    return bounds


def _appraisal_output_contract(family_names: Sequence[str]) -> dict[str, object]:
    """Build the exact model-facing family-object schema."""

    proposition_schema = {
        "type": "object",
        "required": list(APPRAISAL_PROPOSITION_FIELDS),
        "optional": ["object_handle"],
        "additionalProperties": False,
        "properties": {
            "proposition_kind": {"type": "string"},
            "subject_handle": {"type": "string"},
            "object_handle": {"type": ["string", "null"]},
            "evidence_handles": {
                "type": "array",
                "items": {"type": "string"},
            },
            "role_assignments": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "required": ["role", "entity_handle"],
                    "additionalProperties": False,
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": list(APPRAISAL_ROLE_VALUES),
                        },
                        "entity_handle": {"type": "string"},
                    },
                },
            },
            "semantic_value": {"type": "string"},
        },
    }
    delta_schema = {
        "type": "object",
        "required": list(APPRAISAL_DELTA_FIELDS),
        "optional": [],
        "additionalProperties": False,
        "properties": {
            "target_path": {"type": "string"},
            "delta": {"type": "integer"},
            "evidence_handles": {
                "type": "array",
                "items": {"type": "string"},
            },
            "reason": {"type": "string"},
        },
    }
    family_schema = {
        "type": "object",
        "required": list(APPRAISAL_FAMILY_OBJECT_FIELDS),
        "additionalProperties": False,
        "properties": {
            "propositions": {
                "type": "array",
                "maxItems": 8,
                "items": proposition_schema,
            },
            "deltas": {
                "type": "array",
                "maxItems": 8,
                "items": delta_schema,
            },
        },
    }
    json_schema = {
        "type": "object",
        "required": list(family_names),
        "additionalProperties": False,
        "family_value_schema": family_schema,
    }
    contract = {"json_schema": json_schema}
    return contract

GOAL_DIALOGUE_ROLE_BINDING_GUIDANCE = '''对话 payload 中的 dialogue_role_bindings 是权威的角色句柄绑定：第一人称代词使用 first_person_handle，第二人称代词使用 second_person_handle。它们约束语义命题中的 actor 与 target 方向；transport speaker 本身不改变被叙述事件的行动者或对象。'''

APPRAISAL_QUESTION_GUIDANCE = '''按 `semantic_appraisal_group.v1` 处理 payload 中按固定顺序列出的 appraisal family。每个 family 只返回一个对象，字段恰为 `propositions` 与 `deltas`；两个数组都只能保留有证据支持且通过该 family 域约束的项目，每个数组最多八项。''' + GOAL_DIALOGUE_ROLE_BINDING_GUIDANCE

ORDINARY_GOAL_BID_QUESTION_GUIDANCE = '''按 `ordinary_goal_bid.v1` 使用已接受判断、确定性通知和本条 payload 的目标专属语境，返回角色此刻愿意推进的普通目标及关系立场。''' + GOAL_DIALOGUE_ROLE_BINDING_GUIDANCE

ACTIVE_GOAL_BID_GROUP_QUESTION_GUIDANCE = '''按 `active_goal_bid_group.v1` 严格依照 payload 的分支名册顺序，为每个分支返回一个完整目标；不得预选胜者、改变名册或输出排名。''' + GOAL_DIALOGUE_ROLE_BINDING_GUIDANCE

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
_OBSERVATION_CONTEXT_FIELDS = frozenset(
    {
        "conversation_frame",
        "direct_facts",
        "entity_index",
        "evidence",
        "supplemental_context",
    }
)
_CONVERSATION_FRAME_FIELDS = frozenset(
    {
        "channel_scope",
        "character_role",
        "conversation_continuity",
        "current_user_role",
        "dialogue_role_bindings",
        "participant_bindings",
        "public_group_scene",
        "semantic_temporal_context",
    }
)
_SUPPLEMENTAL_CONTEXT_FIELDS = frozenset(
    {
        "dialogue_observation",
        "local_time_context",
        "non_dialog_percepts",
        "trigger_source",
    }
)
_EVIDENCE_REQUIRED_FIELDS = frozenset(
    {
        "authority",
        "handle",
        "provenance_role",
        "semantic_text",
        "source_kind",
    }
)
_EVIDENCE_OPTIONAL_FIELDS = frozenset(
    {
        "dialogue_role_binding",
        "memory_scope",
        "temporal_provenance",
    }
)
_DIALOGUE_EVIDENCE_SOURCE_KINDS = frozenset({"dialog", "episode"})
_DIALOGUE_ROLE_FIELDS = frozenset(
    {
        "addressee_handle",
        "first_person_handle",
        "implicit_imperative_subject_handle",
        "second_person_handle",
        "speaker_handle",
    }
)
_DIALOGUE_ROLE_TO_HANDLE = {
    CURRENT_CHARACTER_ROLE: "self",
    CURRENT_USER_ROLE: "current_user",
}
_RELATION_CONTEXT_REQUIRED_FIELDS = frozenset(
    {
        "character_constraints",
        "character_operational_context",
        "current_affect",
        "relationship_projection",
    }
)
_RELATION_CONTEXT_OPTIONAL_FIELDS = frozenset({"character_sleep_phase"})
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

    if isinstance(value, Mapping):
        return {
            field_name: _sanitize_prompt_value(nested_value)
            for field_name, nested_value in value.items()
            if field_name not in _PRIVATE_RUNTIME_FIELDS
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


def _normalize_dialogue_role_bindings(
    value: object,
    *,
    error_label: str,
) -> list[dict[str, str]]:
    """Validate and copy the closed dialogue role binding carrier."""

    if not isinstance(value, list):
        raise PromptContractError(f"{error_label} must be a list")
    normalized: list[dict[str, str]] = []
    for index, binding in enumerate(value):
        if (
            not isinstance(binding, Mapping)
            or set(binding) != _DIALOGUE_ROLE_FIELDS
            or any(not isinstance(binding[field], str) for field in binding)
            or any(
                binding[field_name] not in {"self", "current_user"}
                for field_name in binding
            )
        ):
            raise PromptContractError(
                f"{error_label}[{index}] fields are not exact"
            )
        if binding["second_person_handle"] != binding["addressee_handle"]:
            raise PromptContractError(
                f"{error_label}[{index}] second_person_handle must match "
                "addressee_handle"
            )
        normalized.append({
            field_name: binding[field_name]
            for field_name in sorted(_DIALOGUE_ROLE_FIELDS)
        })
    return normalized


def _canonical_dialogue_role_binding() -> dict[str, str]:
    """Build the typed binding carried by a canonical dialog percept."""

    role_binding = {
        "speaker_handle": _DIALOGUE_ROLE_TO_HANDLE[CURRENT_USER_ROLE],
        "addressee_handle": _DIALOGUE_ROLE_TO_HANDLE[CURRENT_CHARACTER_ROLE],
        "first_person_handle": _DIALOGUE_ROLE_TO_HANDLE[CURRENT_USER_ROLE],
        "implicit_imperative_subject_handle": _DIALOGUE_ROLE_TO_HANDLE[
            CURRENT_CHARACTER_ROLE
        ],
    }
    role_binding["second_person_handle"] = role_binding["addressee_handle"]
    return {
        field_name: role_binding[field_name]
        for field_name in sorted(_DIALOGUE_ROLE_FIELDS)
    }


def _episode_dialogue_bindings_by_source(
    episode: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    """Index canonical dialog role bindings without exposing source ids."""

    percepts = episode.get("percepts", [])
    if not isinstance(percepts, list):
        raise PromptContractError("Episode percepts must be a list")
    bindings_by_source: dict[str, dict[str, str]] = {}
    for percept in percepts:
        if not isinstance(percept, Mapping):
            raise PromptContractError("Episode percept must be a mapping")
        if (
            percept.get("source_kind") != "dialog"
            or percept.get("percept_kind") != "dialog"
        ):
            continue
        source_id = percept.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise PromptContractError(
                "Canonical dialog percept source linkage is invalid"
            )
        binding = _canonical_dialogue_role_binding()
        previous_binding = bindings_by_source.get(source_id)
        if previous_binding is not None and previous_binding != binding:
            raise PromptContractError(
                "Canonical dialog source has conflicting role bindings"
            )
        bindings_by_source[source_id] = binding
    return bindings_by_source


def _validate_observation_context(
    observation_context: Mapping[str, object],
) -> None:
    """Validate the one prompt-safe carrier consumed by the first stage."""

    _require_exact_fields(
        observation_context,
        expected_fields=_OBSERVATION_CONTEXT_FIELDS,
        error_label="Observation context",
    )
    conversation_frame = observation_context["conversation_frame"]
    if not isinstance(conversation_frame, Mapping):
        raise PromptContractError("Observation conversation_frame must be a mapping")
    _require_exact_fields(
        conversation_frame,
        expected_fields=_CONVERSATION_FRAME_FIELDS,
        error_label="Observation conversation_frame",
    )
    for field_name in (
        "channel_scope",
        "character_role",
        "conversation_continuity",
        "current_user_role",
        "public_group_scene",
        "semantic_temporal_context",
    ):
        if not isinstance(conversation_frame[field_name], str):
            raise PromptContractError(
                f"Observation conversation_frame.{field_name} must be text"
            )
    _normalize_dialogue_role_bindings(
        conversation_frame["dialogue_role_bindings"],
        error_label="Observation dialogue role binding",
    )
    if not isinstance(conversation_frame["participant_bindings"], list):
        raise PromptContractError(
            "Observation conversation_frame.participant_bindings must be a list"
        )
    for binding in conversation_frame["participant_bindings"]:
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"handle", "display_name", "entity_kind"}
            or any(not isinstance(binding[field], str) for field in binding)
            or binding["entity_kind"] != "third_party"
        ):
            raise PromptContractError(
                "Observation participant binding fields are not exact"
            )

    evidence = observation_context["evidence"]
    if not isinstance(evidence, list):
        raise PromptContractError("Observation evidence must be a list")
    for row in evidence:
        if not isinstance(row, Mapping):
            raise PromptContractError("Observation evidence rows must be mappings")
        fields = set(row)
        if fields - (_EVIDENCE_REQUIRED_FIELDS | _EVIDENCE_OPTIONAL_FIELDS):
            raise PromptContractError("Observation evidence fields are not exact")
        if not _EVIDENCE_REQUIRED_FIELDS <= fields:
            raise PromptContractError("Observation evidence fields are incomplete")
        for field_name in _EVIDENCE_REQUIRED_FIELDS:
            if not isinstance(row[field_name], str) or not row[field_name].strip():
                raise PromptContractError(
                    f"Observation evidence.{field_name} must be non-empty text"
                )
        for field_name in _EVIDENCE_OPTIONAL_FIELDS:
            if field_name in row and not isinstance(row[field_name], (str, Mapping)):
                raise PromptContractError(
                    f"Observation evidence.{field_name} has invalid type"
                )
        if "dialogue_role_binding" in row:
            _normalize_dialogue_role_bindings(
                [row["dialogue_role_binding"]],
                error_label="Observation evidence dialogue role binding",
            )

    direct_facts = observation_context["direct_facts"]
    entity_index = observation_context["entity_index"]
    if not isinstance(direct_facts, list) or not isinstance(entity_index, list):
        raise PromptContractError(
            "Observation direct_facts and entity_index must be lists"
        )
    for row in direct_facts:
        if not isinstance(row, Mapping):
            raise PromptContractError("Observation direct fact must be a mapping")
        required_fields = {
            "evidence_ref",
            "fact_kind",
            "producer",
            "target_handles",
            "target_roles",
        }
        if not required_fields <= set(row) or set(row) - (
            required_fields | {"evidence_handle", "observed_progress"}
        ):
            raise PromptContractError("Observation direct fact fields are not exact")
        if any(
            not isinstance(row[field_name], str) or not row[field_name]
            for field_name in ("fact_kind", "producer")
        ):
            raise PromptContractError("Observation direct fact text is invalid")
        if any(
            not isinstance(row[field_name], list)
            for field_name in ("target_handles", "target_roles")
        ):
            raise PromptContractError("Observation direct fact lists are invalid")
        if any(
            not isinstance(item, str) or not item.strip()
            for field_name in ("target_handles", "target_roles")
            for item in row[field_name]
        ):
            raise PromptContractError(
                "Observation direct fact target values are invalid"
            )
        if "observed_progress" in row and (
            isinstance(row["observed_progress"], bool)
            or not isinstance(row["observed_progress"], int)
            or not 0 <= row["observed_progress"] <= 100
        ):
            raise PromptContractError(
                "Observation direct fact progress is invalid"
            )
        evidence_ref = row["evidence_ref"]
        if (
            not isinstance(evidence_ref, Mapping)
            or set(evidence_ref) != {
                "occurred_at",
                "semantic_summary",
                "source_kind",
            }
            or any(
                not isinstance(evidence_ref[field_name], str)
                or not evidence_ref[field_name].strip()
                for field_name in evidence_ref
            )
        ):
            raise PromptContractError(
                "Observation direct fact evidence provenance is invalid"
            )
        if "evidence_handle" in row and (
            not isinstance(row["evidence_handle"], str)
            or not row["evidence_handle"].strip()
        ):
            raise PromptContractError(
                "Observation direct fact evidence handle is invalid"
            )
    for row in entity_index:
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "entity_kind",
                "evidence_handles",
                "handle",
                "semantic_state",
            }
            or not isinstance(row["handle"], str)
            or not isinstance(row["entity_kind"], str)
            or not isinstance(row["evidence_handles"], list)
            or any(
                not isinstance(handle, str)
                for handle in row["evidence_handles"]
            )
            or not isinstance(row["semantic_state"], Mapping)
        ):
            raise PromptContractError("Observation entity index row is invalid")

    supplemental = observation_context["supplemental_context"]
    if not isinstance(supplemental, Mapping):
        raise PromptContractError("Observation supplemental_context must be a mapping")
    _require_exact_fields(
        supplemental,
        expected_fields=_SUPPLEMENTAL_CONTEXT_FIELDS,
        error_label="Observation supplemental_context",
    )
    for field_name in (
        "dialogue_observation",
        "local_time_context",
        "non_dialog_percepts",
    ):
        if not isinstance(supplemental[field_name], list):
            raise PromptContractError(
                f"Observation supplemental_context.{field_name} must be a list"
            )
    if (
        not isinstance(supplemental["trigger_source"], str)
        or not supplemental["trigger_source"].strip()
    ):
        raise PromptContractError(
            "Observation supplemental_context.trigger_source must be text"
        )


def _validate_relation_context(value: Mapping[str, object]) -> None:
    """Validate the closed relationship-owned A2 carrier."""

    fields = set(value)
    if not _RELATION_CONTEXT_REQUIRED_FIELDS <= fields:
        raise PromptContractError("A2 relation_context fields are incomplete")
    if fields - (
        _RELATION_CONTEXT_REQUIRED_FIELDS | _RELATION_CONTEXT_OPTIONAL_FIELDS
    ):
        raise PromptContractError("A2 relation_context fields are not exact")
    for field_name in (
        "character_constraints",
        "character_operational_context",
        "relationship_projection",
    ):
        if not isinstance(value[field_name], Mapping):
            raise PromptContractError(
                f"A2 relation_context.{field_name} must be a mapping"
            )
    if not isinstance(value["current_affect"], list):
        raise PromptContractError("A2 relation_context.current_affect must be a list")
    if "character_sleep_phase" in value and not isinstance(
        value["character_sleep_phase"], str
    ):
        raise PromptContractError(
            "A2 relation_context.character_sleep_phase must be text"
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
    observation_context: Mapping[str, object],
    question: ChainQuestion,
) -> str:
    """Render one observation carrier and the first semantic question.

    Args:
        observation_context: Prompt-safe facts whose first semantic consumer is
            the first non-skipped appraisal or goal stage.
        question: First model-owned registered contract and dynamic payload.

    Returns:
        Compact canonical JSON containing the observation carrier and question.

    Raises:
        PromptContractError: A malformed carrier, goal-only carrier, or
            evaluation-only structural metadata fail before model invocation.
    """

    if not isinstance(observation_context, Mapping):
        raise PromptContractError("Observation context must be a mapping")
    _reject_fields(
        observation_context,
        forbidden_fields=_EVALUATION_METADATA_FIELDS,
        error_label="evaluation metadata",
    )
    _reject_fields(
        observation_context,
        forbidden_fields=_GOAL_ONLY_FIELDS,
        error_label="goal-only",
    )
    _reject_fields(
        observation_context,
        forbidden_fields=_PRIVATE_RUNTIME_FIELDS,
        error_label="private metadata",
    )
    _validate_observation_context(observation_context)
    message_sections = [{"observation_context": dict(observation_context)}]
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
    "build_observation_context",
    "build_question_message",
]

def _project_direct_facts(
    direct_facts: Sequence[Mapping[str, object]],
    evidence_rows: Sequence[Mapping[str, object]],
    handle_to_ref: Mapping[str, Mapping[str, str]] | None,
) -> list[dict[str, object]]:
    """Project typed direct facts without persistent references."""

    source_to_handle = {
        row["evidence_ref"]["source_id"]: row["evidence_handle"]
        for row in evidence_rows
        if isinstance(row.get("evidence_ref"), Mapping)
        and isinstance(row["evidence_ref"].get("source_id"), str)
    }
    projected: list[dict[str, object]] = []
    entity_to_handle = {
        ref.get("entity_id"): handle
        for handle, ref in (handle_to_ref or {}).items()
        if isinstance(ref, Mapping) and isinstance(ref.get("entity_id"), str)
    }
    for fact in direct_facts:
        if not isinstance(fact, Mapping):
            raise PromptContractError("Direct fact must be a mapping")
        evidence_ref = fact.get("evidence_ref")
        if not isinstance(evidence_ref, Mapping):
            raise PromptContractError("Direct fact evidence_ref must be a mapping")
        source_id = evidence_ref.get("source_id")
        evidence_handle = source_to_handle.get(source_id)
        provenance = {
            "source_kind": evidence_ref.get("source_kind"),
            "semantic_summary": evidence_ref.get("semantic_summary"),
            "occurred_at": evidence_ref.get("occurred_at"),
        }
        if any(
            not isinstance(value, str) or not value.strip()
            for value in provenance.values()
        ):
            raise PromptContractError(
                "Direct fact evidence provenance is invalid"
            )
        target_handles: list[str] = []
        target_roles: list[str] = []
        for target in fact.get("target_refs", []):
            if not isinstance(target, Mapping):
                raise PromptContractError("Direct fact target ref must be a mapping")
            if isinstance(target.get("role"), str):
                target_roles.append(target["role"])
            if isinstance(target.get("entity_id"), str):
                target_handle = entity_to_handle.get(target["entity_id"])
                if target_handle is not None:
                    target_handles.append(target_handle)
        row: dict[str, object] = {
            "producer": fact.get("producer", ""),
            "fact_kind": fact.get("fact_kind", ""),
            "evidence_ref": provenance,
            "target_handles": target_handles,
            "target_roles": target_roles,
        }
        if evidence_handle is not None:
            row["evidence_handle"] = evidence_handle
        if "observed_progress" in fact:
            row["observed_progress"] = fact["observed_progress"]
        projected.append(_sanitize_prompt_value(row))
    return projected


def _project_entity_index(
    projection_payload: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Project stable semantic entity handles for later goal consumers."""

    field_kinds = (
        ("goals", "goal"),
        ("threats", "threat"),
        ("events", "event"),
        ("knowledge_gaps", "knowledge_gap"),
        ("causal_candidates", None),
    )
    entity_index: list[dict[str, object]] = []
    for field_name, entity_kind in field_kinds:
        rows = projection_payload.get(field_name, [])
        if not isinstance(rows, list):
            raise PromptContractError(
                f"Projection {field_name} must be a list"
            )
        for row in rows:
            if not isinstance(row, Mapping):
                raise PromptContractError("Projected entity must be a mapping")
            handle = row.get("handle")
            if not isinstance(handle, str) or not handle.strip():
                raise PromptContractError("Projected entity handle is invalid")
            evidence_handles = row.get("evidence_handles", [])
            if not isinstance(evidence_handles, list):
                evidence_handles = []
            if isinstance(row.get("evidence_handle"), str):
                evidence_handles = [row["evidence_handle"]]
            if any(
                not isinstance(handle, str) or not handle.strip()
                for handle in evidence_handles
            ):
                raise PromptContractError(
                    "Projected entity evidence handle is invalid"
                )
            row_entity_kind = entity_kind
            if row_entity_kind is None:
                row_entity_kind = row.get("candidate_kind")
                if (
                    not isinstance(row_entity_kind, str)
                    or not row_entity_kind.strip()
                ):
                    raise PromptContractError(
                        "Projected causal candidate kind is invalid"
                    )
                if row_entity_kind not in {"event", "threat", "knowledge_gap"}:
                    raise PromptContractError(
                        "Projected causal candidate kind is unsupported"
                    )
            semantic_state = _sanitize_prompt_value({
                key: value
                for key, value in row.items()
                if key not in {"handle", "evidence_handle", "evidence_handles"}
            })
            if not isinstance(semantic_state, Mapping):
                raise PromptContractError("Projected entity semantic state is invalid")
            entity_index.append({
                "handle": handle,
                "entity_kind": row_entity_kind,
                "evidence_handles": list(evidence_handles),
                "semantic_state": dict(semantic_state),
            })
    return entity_index


def _project_evidence_rows(
    evidence_rows: Sequence[Mapping[str, object]],
    dialogue_bindings_by_source: Mapping[str, Mapping[str, str]] | None = None,
) -> list[dict[str, object]]:
    """Retain semantic evidence text and trusted provenance labels."""

    projected: list[dict[str, object]] = []
    for row in evidence_rows:
        if not isinstance(row, Mapping):
            raise PromptContractError("Evidence row must be a mapping")
        evidence_ref = row.get("evidence_ref")
        if not isinstance(evidence_ref, Mapping):
            raise PromptContractError("Evidence row reference must be a mapping")
        source_kind = evidence_ref.get("source_kind")
        handle = row.get("evidence_handle")
        semantic_text = row.get("semantic_text")
        authority = row.get("authority")
        if not all(isinstance(value, str) for value in (
            source_kind,
            handle,
            semantic_text,
            authority,
        )):
            raise PromptContractError("Evidence row fields are invalid")
        memory_scope = row.get("memory_scope")
        projected_row: dict[str, object] = {
            "handle": handle,
            "source_kind": source_kind,
            "semantic_text": semantic_text,
            "authority": authority,
            "provenance_role": project_evidence_provenance_role(
                source_kind,
                memory_scope,
            ),
        }
        if isinstance(memory_scope, str):
            projected_row["memory_scope"] = memory_scope
        temporal_provenance = row.get("temporal_provenance")
        if isinstance(temporal_provenance, Mapping):
            projected_row["temporal_provenance"] = dict(temporal_provenance)
        if (
            authority == "current_event"
            and evidence_ref.get("source_kind")
            in _DIALOGUE_EVIDENCE_SOURCE_KINDS
        ):
            source_id = evidence_ref.get("source_id")
            dialogue_binding = (
                dialogue_bindings_by_source or {}
            ).get(source_id)
            if dialogue_binding is not None:
                projected_row["dialogue_role_binding"] = dict(dialogue_binding)
        projected.append(projected_row)
    return projected


def build_observation_context(
    *,
    projection_payload: Mapping[str, Any],
    scene_context: Mapping[str, Any],
    episode: Mapping[str, Any],
    evidence: Sequence[Mapping[str, object]],
    direct_facts: Sequence[Mapping[str, object]],
    handle_to_ref: Mapping[str, Mapping[str, str]] | None = None,
) -> Mapping[str, object]:
    """Build the first-consumer observation carrier for the serial chain."""

    if not isinstance(projection_payload, Mapping):
        raise PromptContractError("projection payload must be a mapping")
    if not isinstance(scene_context, Mapping):
        raise PromptContractError("scene context must be a mapping")
    if not isinstance(episode, Mapping):
        raise PromptContractError("episode must be a mapping")
    visible_percepts = project_model_visible_percepts(episode)
    evidence_rows = list(evidence)
    dialogue_bindings_by_source = _episode_dialogue_bindings_by_source(episode)
    evidence = _project_evidence_rows(
        evidence_rows,
        dialogue_bindings_by_source,
    )
    evidence_texts = {
        row["semantic_text"]
        for row in evidence
        if isinstance(row.get("semantic_text"), str)
    }
    dialogue_observation: list[object] = []
    non_dialog_percepts: list[object] = []
    local_time_context: list[object] = []
    dialogue_role_bindings: list[dict[str, str]] = []
    dialogue_texts: set[str] = set()
    for percept in visible_percepts:
        input_source = percept.get("input_source")
        content = percept.get("content")
        if input_source == "dialog":
            if isinstance(content, Mapping):
                semantic_text = content.get("semantic_text", content.get("text"))
                if (
                    isinstance(semantic_text, str)
                    and semantic_text not in evidence_texts
                    and semantic_text not in dialogue_texts
                ):
                    dialogue_observation.append({"semantic_text": semantic_text})
                    dialogue_texts.add(semantic_text)
            role_fields = {
                "speaker_handle": "speaker_role",
                "addressee_handle": "addressee_role",
                "first_person_handle": "first_person_role",
                "implicit_imperative_subject_handle": (
                    "implicit_imperative_subject_role"
                ),
            }
            role_binding: dict[str, str] = {}
            for handle_field, role_field in role_fields.items():
                role_token = percept.get(role_field)
                if not isinstance(role_token, str):
                    raise PromptContractError(
                        f"Observation dialog role {role_field} is missing"
                    )
                try:
                    role_binding[handle_field] = _DIALOGUE_ROLE_TO_HANDLE[
                        role_token
                    ]
                except KeyError as exc:
                    raise PromptContractError(
                        f"Observation dialog role {role_field} is unsupported"
                    ) from exc
            role_binding["second_person_handle"] = role_binding[
                "addressee_handle"
            ]
            dialogue_role_bindings.append(role_binding)
        elif input_source == "local_time_context":
            local_time_context.append(_sanitize_prompt_value(content))
        else:
            non_dialog_percepts.append(_sanitize_prompt_value(percept))
    participant_bindings = scene_context.get("participant_bindings", [])
    if not isinstance(participant_bindings, list):
        raise PromptContractError(
            "scene context participant_bindings must be a list"
        )
    normalized_participant_bindings = []
    for index, binding in enumerate(participant_bindings):
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"handle", "display_name", "entity_kind"}
        ):
            raise PromptContractError(
                "scene context participant binding "
                f"{index} fields are not exact"
            )
        normalized_participant_bindings.append({
            "handle": binding["handle"],
            "display_name": binding["display_name"],
            "entity_kind": binding["entity_kind"],
        })
    conversation_frame = {
        "channel_scope": scene_context.get("channel_scope", ""),
        "character_role": scene_context.get("character_role", ""),
        "conversation_continuity": scene_context.get(
            "conversation_continuity", ""
        ),
        "current_user_role": scene_context.get("current_user_role", ""),
        "dialogue_role_bindings": dialogue_role_bindings,
        "participant_bindings": normalized_participant_bindings,
        "public_group_scene": scene_context.get("public_group_scene", ""),
        "semantic_temporal_context": scene_context.get(
            "semantic_temporal_context", ""
        ),
    }
    supplemental_context = {
        "dialogue_observation": dialogue_observation,
        "local_time_context": local_time_context,
        "non_dialog_percepts": non_dialog_percepts,
        "trigger_source": str(episode.get("trigger_source", "user_message")),
    }
    observation_context = {
        "conversation_frame": conversation_frame,
        "direct_facts": _project_direct_facts(
            direct_facts,
            evidence_rows,
            handle_to_ref,
        ),
        "entity_index": _project_entity_index(projection_payload),
        "evidence": evidence,
        "supplemental_context": supplemental_context,
    }
    _reject_fields(
        observation_context,
        forbidden_fields=_EVALUATION_METADATA_FIELDS,
        error_label="evaluation metadata",
    )
    _reject_fields(
        observation_context,
        forbidden_fields=_PRIVATE_RUNTIME_FIELDS,
        error_label="private metadata",
    )
    _validate_observation_context(observation_context)
    return observation_context


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


def build_appraisal_question_payload(
    *,
    questions: Sequence[Mapping[str, object]],
    stage_name: str,
    l1_residue: Mapping[str, object] | None = None,
    relation_context: Mapping[str, object] | None = None,
    role_assignment_handles_by_evidence: Mapping[
        str, Sequence[str]
    ] | None = None,
) -> dict[str, object]:
    """Build one fixed-stage appraisal request with adjacent family domains."""

    stage_families = dict(APPRAISAL_STAGE_FAMILIES).get(stage_name)
    if stage_families is None:
        raise PromptContractError("appraisal stage must be A1 or A2")
    by_family: dict[str, Mapping[str, object]] = {}
    for question in questions:
        if not isinstance(question, Mapping):
            raise PromptContractError("appraisal questions must be mappings")
        family = question.get("question_kind")
        if family not in stage_families:
            raise PromptContractError(
                f"appraisal family {family!r} is not owned by {stage_name}"
            )
        if family in by_family:
            raise PromptContractError(f"appraisal family {family!r} is duplicated")
        by_family[str(family)] = question
    family_names = tuple(
        family
        for family in stage_families
        if family in by_family
    )
    if not family_names:
        raise PromptContractError(
            f"appraisal stage {stage_name} requires one planned family"
        )

    rows: list[dict[str, object]] = []
    for family in family_names:
        question = by_family[family]
        evidence_handles = list(question["evidence_handles"])
        subject_handles = list(question["permitted_role_handles"])
        assignment_handles = list(
            question["permitted_role_assignment_handles"]
        )
        writable_paths = list(question["permitted_delta_paths"])
        row = {
            "family": family,
            "semantic_question": question["semantic_question"],
            "proposition_kinds": list(question_proposition_kinds(family)),
            "proposition_kind_semantics": (
                question_proposition_kind_semantics(family)
            ),
            "permitted_subject_handles": subject_handles,
            "permitted_object_handles": subject_handles,
            "permitted_role_assignment_handles": assignment_handles,
            "evidence_handles": evidence_handles,
            "writable_delta_paths": writable_paths,
            "delta_bounds": _appraisal_delta_bounds(writable_paths),
        }
        family_role_domains = role_assignment_handles_by_evidence
        if family_role_domains is None:
            candidate_role_domains = question.get(
                "permitted_role_assignment_handles_by_evidence"
            )
            if isinstance(candidate_role_domains, Mapping):
                family_role_domains = candidate_role_domains
        if family_role_domains is not None:
            for evidence_handle in evidence_handles:
                domain = family_role_domains.get(evidence_handle, ())
                if (
                    isinstance(domain, (str, bytes))
                    or not isinstance(domain, Sequence)
                    or any(not isinstance(handle, str) for handle in domain)
                ):
                    raise PromptContractError(
                        "appraisal evidence role authority is invalid"
                    )
            row["permitted_role_assignment_handles_by_evidence"] = {
                evidence_handle: list(
                    family_role_domains.get(
                        evidence_handle,
                        (),
                    )
                )
                for evidence_handle in evidence_handles
            }
        rows.append(row)
    payload = {
        "families": rows,
        "output_contract": _appraisal_output_contract(family_names),
    }
    if l1_residue:
        payload["l1_residue"] = dict(l1_residue)
    if stage_name == "A2":
        if not isinstance(relation_context, Mapping):
            raise PromptContractError(
                "A2 appraisal requires relation_context"
            )
        _validate_relation_context(relation_context)
        payload["relation_context"] = dict(relation_context)
    elif relation_context is not None:
        raise PromptContractError(
            "A1 appraisal cannot receive relation_context"
        )
    return payload


def build_appraisal_stage_question(
    *,
    planned_questions: Sequence[Mapping[str, object]],
    stage_name: str,
    l1_residue: Mapping[str, object] | None = None,
    relation_context: Mapping[str, object] | None = None,
    role_assignment_handles_by_evidence: Mapping[
        str, Sequence[str]
    ] | None = None,
) -> ChainQuestion:
    """Build the sole grouped question for a fixed A1 or A2 stage."""

    payload = build_appraisal_question_payload(
        questions=planned_questions,
        stage_name=stage_name,
        l1_residue=l1_residue,
        relation_context=relation_context,
        role_assignment_handles_by_evidence=(
            role_assignment_handles_by_evidence
        ),
    )
    question = ChainQuestion(
        contract_name="semantic_appraisal_group.v1",
        payload=payload,
    )
    return question


def build_goal_question_payload(
    *,
    goal_kind,
    goal_projection,
    evidence_handles,
    action_tendencies,
    branch_intent_guidance,
    role_bindings,
    selection_operations,
    progress_evidence,
    authoritative_state,
    continuity_context,
    current_episode_evidence_handles,
    dialogue_role_bindings,
    carried_relational_willingness=None,
    l1_residue=None,
):
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
        "action_tendencies": list(action_tendencies),
        "branch_intent_guidance": branch_intent_guidance,
        "selection_operations": [dict(row) for row in selection_operations],
        "progress_evidence": [dict(row) for row in progress_evidence],
        "authoritative_state": dict(authoritative_state),
        "continuity_context": dict(continuity_context),
        "dialogue_role_bindings": _normalize_dialogue_role_bindings(
            dialogue_role_bindings,
            error_label="Goal dialogue role binding",
        ),
        "goal_output_contract": build_goal_output_contract(
            evidence_handles=set(evidence_handles),
            episode_evidence_handles=set(current_episode_evidence_handles),
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
    return _sanitize_prompt_value(payload)


def build_active_goal_group_question_payload(
    *,
    roster,
    evidence_handles,
    role_handles,
    continuity_context,
    dialogue_role_bindings,
):
    """Build the compact frozen roster consumed by G1b."""

    payload = {
        "branch_roster": [dict(row) for row in roster],
        "allowed_evidence_handles": list(evidence_handles),
        "allowed_role_handles": sorted(role_handles),
        "continuity_context": dict(continuity_context),
        "dialogue_role_bindings": _normalize_dialogue_role_bindings(
            dialogue_role_bindings,
            error_label="Active goal dialogue role binding",
        ),
    }
    return _sanitize_prompt_value(payload)


def build_workspace_question_payload(*, bids, current_event, goal_contexts):
    """Build a prompt-safe workspace payload from complete branch bids.

    The workspace owner assigns stable ``bN`` handles and projects only the
    compact branch/goal index needed for partitioning.
    """

    if bids:
        partition_request = prepare_partition(
            bids,
            current_event,
            goal_contexts,
        )
        payload = partition_request.prompt_payload
    else:
        payload = {"bid_index": {}}
    _reject_fields(
        payload,
        forbidden_fields=_EVALUATION_METADATA_FIELDS,
        error_label="evaluation metadata",
    )
    _reject_fields(
        payload,
        forbidden_fields=_PRIVATE_RUNTIME_FIELDS,
        error_label="private metadata",
    )
    return {
        "bid_index": {
            str(handle): dict(row)
            for handle, row in payload["bid_index"].items()
        },
    }


def build_action_plan_question_payload(
    *,
    primary_bid_handle,
    supporting_bid_handles,
    bid_index,
    action_index,
    resolver_index,
    resolver_context,
    runtime_capability_limits,
    current_goal_progress,
    required_resolver_evidence_dependency,
    evidence=(),
    self_cognition_response_context=None,
):
    payload = {
        "primary_bid_handle": primary_bid_handle,
        "supporting_bid_handles": list(supporting_bid_handles),
        "bid_index": {
            str(handle): dict(row)
            for handle, row in bid_index.items()
        },
        "action_index": {
            str(handle): dict(row)
            for handle, row in action_index.items()
        },
        "resolver_index": {
            str(handle): dict(row)
            for handle, row in resolver_index.items()
        },
        "resolver_context": resolver_context,
        "runtime_capability_limits": list(runtime_capability_limits),
        "current_goal_progress": (
            dict(current_goal_progress)
            if current_goal_progress is not None
            else None
        ),
        "required_resolver_evidence_dependency": (
            dict(required_resolver_evidence_dependency)
            if required_resolver_evidence_dependency is not None
            else None
        ),
    }
    future_speak_handles = [
        str(handle)
        for handle, affordance in action_index.items()
        if isinstance(affordance, Mapping)
        and affordance.get("action_kind") == "future_speak"
    ]
    if future_speak_handles:
        if not isinstance(evidence, Sequence) or isinstance(
            evidence,
            (str, bytes),
        ):
            raise PromptContractError(
                "future-speak evidence must be a sequence"
            )
        allowed_detail_refs: list[dict[str, str]] = []
        for row in evidence:
            if not isinstance(row, Mapping):
                raise PromptContractError(
                    "future-speak evidence rows must be mappings"
                )
            evidence_handle = row.get("evidence_handle")
            authority = row.get("authority")
            if not isinstance(evidence_handle, str) or not evidence_handle.strip():
                raise PromptContractError(
                    "future-speak evidence handle is invalid"
                )
            if not isinstance(authority, str) or authority not in (
                SCHEDULED_AUTHORITY_CURRENT_ROLE_VALUES
            ):
                continue
            allowed_detail_refs.append({
                "evidence_handle": evidence_handle,
                "provenance_role": authority,
            })
        payload["scheduled_authority_contract"] = {
            "schema_version": SCHEDULED_AUTHORITY_PROPOSAL_SCHEMA_VERSION,
            "required_for_action_handles": future_speak_handles,
            "proposal_fields": [
                "schema_version",
                "temporal_alignment",
                "authorized_content_summary",
                "authorized_detail_refs",
            ],
            "detail_ref_fields": [
                "evidence_handle",
                "semantic_summary",
                "provenance_role",
            ],
            "temporal_alignment_values": sorted(
                TEMPORAL_ALIGNMENT_VALUES
            ),
            "required_temporal_alignment": "aligned",
            "temporal_alignment_rule": (
                "temporal_alignment must be aligned for acceptance"
            ),
            "allowed_detail_refs": allowed_detail_refs,
        }
    if self_cognition_response_context is not None:
        if not isinstance(self_cognition_response_context, Mapping):
            raise PromptContractError(
                "self_cognition_response_context must be a mapping"
            )
        expected_fields = {
            "required_fields",
            "allowed_decisions",
            "allowed_evidence_handles",
            "allowed_semantic_target_handles",
            "allowed_participation_basis_values",
            "response_goal_max_chars",
            "reason_max_chars",
        }
        if set(self_cognition_response_context) != expected_fields:
            raise PromptContractError(
                "self_cognition_response_context fields are not exact"
            )
        required_fields = self_cognition_response_context["required_fields"]
        expected_required_fields = list(
            SELF_COGNITION_RESPONSE_REQUIRED_FIELDS
        )
        if required_fields != expected_required_fields:
            raise PromptContractError(
                "self_cognition_response_context.required_fields are invalid"
            )
        for field_name in (
            "allowed_decisions",
            "allowed_evidence_handles",
            "allowed_semantic_target_handles",
            "allowed_participation_basis_values",
        ):
            values = self_cognition_response_context[field_name]
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip()
                for value in values
            ):
                raise PromptContractError(
                    "self_cognition_response_context list is invalid"
                )
            if len(values) != len(set(values)):
                raise PromptContractError(
                    "self_cognition_response_context list values are duplicated"
                )
        allowed_decisions = self_cognition_response_context[
            "allowed_decisions"
        ]
        if not allowed_decisions or not set(allowed_decisions).issubset(
            SELF_COGNITION_RESPONSE_DECISION_VALUES
        ):
            raise PromptContractError(
                "self_cognition_response_context decisions are invalid"
            )
        allowed_participation_values = self_cognition_response_context[
            "allowed_participation_basis_values"
        ]
        if not allowed_participation_values or not set(
            allowed_participation_values
        ).issubset(SELF_COGNITION_RESPONSE_PARTICIPATION_VALUES):
            raise PromptContractError(
                "self_cognition_response_context participation values are invalid"
            )
        if not self_cognition_response_context[
            "allowed_semantic_target_handles"
        ]:
            raise PromptContractError(
                "self_cognition_response_context target domain is empty"
            )
        for field_name in ("response_goal_max_chars", "reason_max_chars"):
            maximum = self_cognition_response_context[field_name]
            if (
                isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or maximum <= 0
            ):
                raise PromptContractError(
                    "self_cognition_response_context bound is invalid"
                )
        payload["self_cognition_response_context"] = dict(
            self_cognition_response_context
        )
    output_required_fields = [
        "action_requests",
        "resolver_requests",
        "goal_resolution",
        "resolver_pending_resolution",
        "resolver_goal_progress",
    ]
    output_properties: dict[str, object] = {
        "action_requests": {
            "type": "array",
            "items": {"type": "object"},
        },
        "resolver_requests": {
            "type": "array",
            "items": {"type": "object"},
        },
        "goal_resolution": {
            "type": "string",
            "enum": sorted(GOAL_RESOLUTION_VALUES),
        },
        "resolver_pending_resolution": {
            "type": ["object", "null"],
        },
        "resolver_goal_progress": {
            "type": ["object", "null"],
        },
    }
    if self_cognition_response_context is not None:
        output_required_fields.append("self_cognition_response")
        output_properties["self_cognition_response"] = {
            "type": "object",
            "required_fields": list(
                SELF_COGNITION_RESPONSE_REQUIRED_FIELDS
            ),
            "additionalProperties": False,
        }
    payload["action_plan_output_contract"] = {
        "required_fields": output_required_fields,
        "additionalProperties": False,
        "properties": output_properties,
    }
    return _sanitize_prompt_value(payload)
