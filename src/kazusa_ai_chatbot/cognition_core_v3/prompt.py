"""Canonical dynamic question packets for the Cognition V3 chain."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

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

    _reject_fields(
        question.payload,
        forbidden_fields=_GOAL_ONLY_FIELDS,
        error_label="goal-only",
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
