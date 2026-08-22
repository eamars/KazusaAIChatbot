"""Byte-stable system anchor for the serialized Cognition V3 chain."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

ENGINE_MANUAL = '''你是角色的私有认知链，只作当前合同要求的语义判断。

# 共享边界
- 当前问题的 payload 是唯一事实、句柄、名册、允许域和可写路径来源；先前接受的回答与确定性 interlude 属于同一过程，权威通知优先。
- 检索、历史、进度和内部观察只是材料，不自动成为事实、立场、权限或承诺。缺少证据时保留不确定性，不声称未发生的效果。
- `eN` 是证据，`evN`、`ceN`、`ctN`、`ckN` 是事件或候选，`gN` 是目标，`r1` 是关系，`self`、`current_user`、`pN` 是角色句柄，`bN` 是目标提案。只使用 payload 明确允许的句柄。
- 证据、角色、路径和合同 token 只出现在结构化字段；自然语言不复写 source id、运行元数据或私有标识。transport speaker 不等于被叙述事件的行动者。
- `dialogue_role_bindings` 是权威绑定：第一人称使用 `first_person_handle`，第二人称使用 `second_person_handle`；保持 actor、target、object 和受益者方向。

# 合同边界
当前注册合同是 `semantic_appraisal_group.v1`、`ordinary_goal_bid.v1`、`active_goal_bid_group.v1`、`workspace_partition.v1` 和 `action_plan.v1`。当前问题的局部 output contract 拥有精确字段、类型、数组上限、枚举、句柄域和可写路径；不要复制、删除或增加字段。

`semantic_appraisal_group.v1` 只判断证据支持的 proposition 与 delta，不选择目标、动作、情绪或生命周期。每个 family 返回自己的 `propositions` 与 `deltas` 数组，空数组表示没有受支持产品。

`ordinary_goal_bid.v1` 与 `active_goal_bid_group.v1` 只表达角色愿意推进的目标及其 evidence_handles、target_role_handles、expected_consequences、private_monologue、confidence 和 `relational_willingness`（普通目标）。保持当前事项一致；没有基础时不借用其他目标的动机。

`workspace_partition.v1` 只分区已有的 `bN`，使用 `primary_bid_handle`、`supporting_bid_handles` 和 `suppressed_bid_handles`，不改写或补造目标。

`action_plan.v1` 只把已选目标转成 action/resolver requests。保持 `action_requests`、`resolver_requests`、`goal_resolution`、`resolver_pending_resolution` 和 `resolver_goal_progress` 的精确形状；需要时才返回 `self_cognition_response`。关系 stance 是判断，不是执行许可；能力、授权和持久化由确定性代码处理。

# JSON 输出
字段集合、类型、顺序、空值和句柄严格服从当前问题。目标、分区和计划不得声称未授权或未执行的效果；`selected_response_operation`、`primary_bid_handle`、`action_requests`、`resolver_requests` 与 `goal_resolution` 只使用 payload 的域。返回前检查角色方向、证据来源和闭合字段。

`relational_willingness` 的 stance 可为 `reject`、`deflect`、`negotiate`、`conditional_accept`、`accept` 或 `not_applicable`；applicability 使用 `relationship_sensitive` 或 `not_relationship_sensitive`。`goal_resolution` 使用 `answerable_now`、`requires_required_evidence`、`requires_user_input` 或 `blocked`。

当 `self_cognition_response` 被要求时，decision 只能是 `stay_silent` 或 `propose_visible_reply`，并遵守 payload 的 target、evidence、participation basis 与 response goal 域。
'''

IDENTITY_PARTITION_ORDER = (
    "core",
    "personality",
    "boundaries",
    "self_image",
)

_DYNAMIC_SYSTEM_FIELDS = frozenset(
    {
        "attempt",
        "attempt_number",
        "available_actions",
        "available_resolver_capabilities",
        "case_id",
        "deadline",
        "deadline_monotonic",
        "direct_facts",
        "episode",
        "episode_and_scene",
        "episode_id",
        "evidence",
        "evidence_and_affordances",
        "expected_answer",
        "fixture_id",
        "invocation_id",
        "mutable_state",
        "pytest_node_id",
        "relationship_and_mutable_state",
        "relationship_context",
        "resolver_context",
        "resolver_cycle_index",
        "retry_count",
        "route_name",
        "rubric",
        "run_id",
        "scene",
        "scene_context",
        "trace_id",
    }
)


class AnchorContractError(ValueError):
    """The proposed static system head contains an invalid partition."""


def _validate_static_identity_value(value: object) -> None:
    """Reject run-specific or evaluation-only field names at any depth."""

    if isinstance(value, Mapping):
        for field_name, nested_value in value.items():
            if field_name in _DYNAMIC_SYSTEM_FIELDS:
                raise AnchorContractError(
                    f"Character identity contains dynamic field {field_name!r}"
                )
            _validate_static_identity_value(nested_value)
        return
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for nested_value in value:
            _validate_static_identity_value(nested_value)


def build_system_head(character_identity: Mapping[str, object]) -> str:
    """Render the exact manual-then-identity system content as canonical JSON.

    Args:
        character_identity: Prompt-safe identity union with exactly the core,
            personality, boundaries, and self-image partitions.

    Returns:
        Compact UTF-8-preserving JSON whose list order keeps the engine manual
        before the four ordered identity partitions.

    Raises:
        AnchorContractError: Missing, extra, non-mapping, or dynamic identity
            content fails before any model request is built.
    """

    if set(character_identity) != set(IDENTITY_PARTITION_ORDER):
        raise AnchorContractError(
            "Character identity requires the exact partitions "
            "core, personality, boundaries, and self_image"
        )

    identity_rows: list[dict[str, object]] = []
    for partition_name in IDENTITY_PARTITION_ORDER:
        partition = character_identity[partition_name]
        if not isinstance(partition, Mapping):
            raise AnchorContractError(
                f"Character identity partition {partition_name!r} must be a mapping"
            )
        _validate_static_identity_value(partition)
        identity_rows.append({partition_name: dict(partition)})

    system_sections = [
        {"engine_manual": ENGINE_MANUAL},
        {"character_identity": identity_rows},
    ]
    system_head = json.dumps(
        system_sections,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return system_head


__all__ = [
    "ENGINE_MANUAL",
    "IDENTITY_PARTITION_ORDER",
    "AnchorContractError",
    "build_system_head",
]
