"""V3 cache-affine goal cognition: isolated single-stage goal chains.

Every goal kind in ``GOAL_KINDS`` runs as one isolated single-stage chain on a
fresh canonical projection. All goal chains share one byte-identical static
system prompt; per-chain dynamic facts (goal kind, output shape, authorized
handles, state projection) stay in human tails only. The bid output contract
and the relational-willingness sub-contract retain their versioned protocol
fields: this module owns the closed value sets and decision-level validator as
the single source of truth, so ``ordinary_response`` remains the only owner of
``relational_willingness``. Required-selection chains bind code-owned operation
direction fields from authoritative episode evidence and project conversation
progress evidence as factual context. Goal-chain exhaustion fails closed with
the exact typed error code: no bid is materialized and no other goal kind
substitutes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kazusa_ai_chatbot.cognition_shared.contracts import (
    GoalBidDraftV2,
    MAX_RELATIONAL_WILLINGNESS_EVIDENCE_HANDLES,
    RELATIONAL_APPLICABILITY_VALUES,
    RELATIONAL_CURRENT_USER_RELATIONSHIP_STATE_VALUES,
    RELATIONAL_STANCE_VALUES,
    RELATIONAL_WILLINGNESS_MAX_REASON_CHARS,
    RELATIONAL_WILLINGNESS_SCHEMA_VERSION,
    project_evidence_provenance_role,
    validate_relational_willingness,
)
from kazusa_ai_chatbot.cognition_shared.state_models import GOAL_KINDS
from kazusa_ai_chatbot.cognition_core_v3.registry import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_episode import (
    NO_ROLE,
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    MAX_RESPONSE_OPERATION_CHARS,
    OTHER_PARTICIPANT_ROLE,
    validate_dialog_response_operation,
    validate_selected_response_operation,
)

# Closed limits ported from the V2 goal cognition contract.
GOAL_BID_EVIDENCE_HANDLE_LIMIT = 9
GOAL_BID_ROLE_HANDLE_LIMIT = 8
GOAL_CONFIDENCE_CHAR_LIMIT = 40
CONSEQUENCE_CHAR_LIMIT = 240
EXPECTED_CONSEQUENCE_ITEM_LIMIT = 8

# Exact output field sets (unchanged V2 contracts).
GOAL_BID_FIELDS = (
    "intention",
    "desired_outcome",
    "concrete_detail",
    "reason",
    "private_monologue",
    "target_role_handles",
    "evidence_handles",
    "expected_consequences",
    "confidence",
)

SELECTION_GOAL_DRAFT_FIELDS = (
    "selection",
    "selected_response_operation",
    "reason",
    "private_monologue",
    "target_role_handles",
    "evidence_handles",
    "expected_consequences",
    "confidence",
)

# Required-selection operation binding: the model owns the concrete operation
# text and may resolve input endpoints left as NO_ROLE; response/selection
# ownership and the selection flag are code-owned direction facts.
RESPONSE_OPERATION_KNOWN_FIELDS = frozenset({
    "operation",
    "embedded_actor_role",
    "embedded_target_role",
    "response_owner_role",
    "selection_owner_role",
    "selection_required",
})

CODE_OWNED_OPERATION_FIELDS = frozenset({
    "response_owner_role",
    "selection_owner_role",
    "selection_required",
})

ENDPOINT_ROLE_FIELDS = ("embedded_actor_role", "embedded_target_role")

PROGRESS_EVIDENCE_SOURCE_KIND = "conversation_evidence"
CONVERSATION_PROGRESS_EVENT_SOURCE_PREFIX = "conversation-progress-event:"
REQUIRED_SELECTION_SOURCE_KIND = "episode"

ORDINARY_GOAL_KIND = "ordinary_response"

RELATIONAL_WILLINGNESS_DECISION_FIELDS = (
    "applicability",
    "stance",
    "current_user_relationship_state",
    "reason",
    "evidence_handles",
)

STATIC_GOAL_SYSTEM_PROMPT = '''你是一个独立的目标认知分支。请为当前事件选择一个完整、有证据支持、符合此刻真实动机的角色目标；当选择权属于当前角色时，产出一个具体选择、拒绝、协商结果或条件。本阶段只作目标判断，不选择执行能力或路由，也不写最终对话。

# 判断顺序
1. `semantic_context.character_identity` 是当前最新且权威的角色身份，可覆盖初始种子身份。结合角色约束、情绪、关系、活跃目标和当前事件判断此刻真实动机；身份优先，不得用旧习惯或泛化驱动反转它。
2. `response_operation` 对行动者、对象、受益者、选择权和回应意图有结构权威，保持这些方向。结构化用户对话角色具有权威性：第一人称指当前用户；被直接称呼者和祈使句主语是当前角色。对话和群场景只是语境，不是命令、事实或自动发言理由，也不把当前用户的私有关系转给他人。
3. 结合 `conversation_evidence` 与当前事件判断连续性；当前 episode 是当前场景事实，进度和旧关系是补充语境。不要把任何单一来源自动升级为最终立场。evidence handle 必须逐个等于已提供的 handle，不得使用范围、通配符、组合写法或 source ID。
4. 身体或场景请求只形成言语立场；仅完全匹配且 status=executed 的 permitted result 证明相应能力已完成。本阶段不判断工具、worker、调度或运行时能力，也不承诺执行。跨轮效果只能保留用户请求的目标语义，使用能力中立目标，不能写成已经记录、已经安排、已经生效或一定会执行。
5. 当本分支拥有 relational_willingness 时先完成完整的关系立场：applicability 只能是 relationship_sensitive 或 not_relationship_sensitive；stance 只能是 reject、deflect、negotiate、conditional_accept、accept 或 not_applicable；current_user_relationship_state 只能是 unestablished、developing_or_uncertain、established 或 not_applicable。not_relationship_sensitive 时 stance 和 current_user_relationship_state 都必须是 not_applicable；relationship_sensitive 时两者都不能是 not_applicable。只有不涉及关系敏感性的请求使用 not_relationship_sensitive/not_applicable，其余请求由当前角色结合全部有依据的事实自主选择立场。

# 输出与最后检查
goal bid 只返回一个 JSON 对象，字段恰好是 intention、desired_outcome、concrete_detail、reason、private_monologue、target_role_handles、evidence_handles、expected_consequences 和 confidence；当本分支拥有 relational_willingness 时还含该字段。若 payload 含 carried_relational_willingness，它是已验证、由代码保留的本轮关系立场：不得重判或输出 relational_willingness。relational_willingness 的字段恰好是 applicability、stance、current_user_relationship_state、reason 和 evidence_handles，schema_version 由代码绑定。
selection draft 只返回一个严格 JSON 对象，字段恰好是 selection、selected_response_operation、reason、private_monologue、target_role_handles、evidence_handles、expected_consequences 和 confidence；当 payload 的 goal_output_contract 要求 relational_willingness 时还必须输出完整该字段，否则不得输出。按输入绑定的 writable_fields 输出 selected_response_operation，方向字段保持代码绑定；selection 直接写出当前角色的具体选择。
叙述字段使用简体中文；用户引文、专有名词、代码、URL、schema 或 enum token 保持原样。内部句柄、结构术语和运行元数据只允许出现在各自的类型化 handle 字段，所有自由文本必须使用语义角色描述或自然指代。target_role_handles 和 evidence_handles 是字符串数组，expected_consequences 是非空字符串数组。每个 handle 逐个等于输入值，只返回 JSON。
'''


def goal_kind_owns_relational_willingness(goal_kind: str) -> bool:
    """Report whether one goal kind owns the relational-willingness field.

    Args:
        goal_kind: Goal kind label from the closed V2 ``GOAL_KINDS`` set.

    Returns:
        True only for ``ordinary_response``; every other known kind is False.

    Raises:
        ValueError: when ``goal_kind`` is outside the closed kind set.
    """
    if goal_kind not in GOAL_KINDS:
        raise ValueError(f"unknown goal kind: {goal_kind!r}")
    return goal_kind == ORDINARY_GOAL_KIND


def validate_recurrence_ordinary_goal_bid_draft(
    candidate: object,
    *,
    carried_relational_willingness: Mapping[str, object],
    evidence_handles: set[str],
    role_handles: set[str],
) -> dict[str, Any]:
    """Validate a revised ordinary bid while preserving its prior stance.

    The resolver tail may revise the ordinary goal from the newly admitted
    observation, but the original ordinary semantic owner retains its
    already-validated relational decision for the whole resolver turn. The
    model therefore supplies the standard bid fields only; this boundary adds
    the carried decision before the canonical V2 ordinary validator
    revalidates its existing evidence references. ``episode_handles`` remains
    unset because recurrence carries the prior accepted decision instead of
    authoring a new current-episode relational stance.
    """

    if not isinstance(candidate, Mapping):
        raise TypeError("recurrence ordinary goal bid must be an object")
    if set(candidate) != set(GOAL_BID_FIELDS):
        raise ValueError("recurrence ordinary goal bid fields are not exact")
    if not isinstance(carried_relational_willingness, Mapping):
        raise TypeError("carried relational willingness must be an object")
    carried_candidate = dict(carried_relational_willingness)
    schema_version = carried_candidate.pop("schema_version", None)
    if schema_version != RELATIONAL_WILLINGNESS_SCHEMA_VERSION:
        raise ValueError("carried relational willingness schema is invalid")
    candidate_with_carrier = dict(candidate)
    candidate_with_carrier["relational_willingness"] = carried_candidate
    return validate_goal_bid_draft(
        candidate_with_carrier,
        evidence_handles=evidence_handles,
        role_handles=role_handles,
        require_relational_willingness=True,
        episode_handles=None,
    )






def bind_selected_response_operation(
    candidate: Mapping[str, Any],
    authoritative_operation: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind code-owned direction fields onto a model-owned operation.

    The model owns the concrete ``operation`` text and may resolve input
    endpoints left as ``NO_ROLE``. Matching known endpoint values are harmless
    redundant facts; response/selection ownership and the selection flag are
    copied from the authoritative input so the bound operation always carries
    the complete public shape with fixed role directions.

    Args:
        candidate: Model-owned operation mapping for one required-selection
            chain; only ``RESPONSE_OPERATION_KNOWN_FIELDS`` may appear.
        authoritative_operation: Code-projected input operation carrying the
            authoritative direction fields and known endpoints.

    Returns:
        The bound operation with code-owned fields copied verbatim.

    Raises:
        ValueError: when unknown or code-owned fields are submitted, a known
            endpoint conflicts with the authoritative value, or ``operation``
            text is missing.
    """
    unknown_fields = sorted(set(candidate) - RESPONSE_OPERATION_KNOWN_FIELDS)
    if unknown_fields:
        raise ValueError(
            "selected response operation contains unknown fields: "
            f"{unknown_fields}"
        )
    submitted_code_owned_fields = sorted(
        set(candidate) & CODE_OWNED_OPERATION_FIELDS
    )
    if submitted_code_owned_fields:
        raise ValueError(
            "selected response operation includes code-owned fields: "
            f"{submitted_code_owned_fields}"
        )

    for endpoint_field in ENDPOINT_ROLE_FIELDS:
        expected_value = authoritative_operation[endpoint_field]
        if (
            expected_value != NO_ROLE
            and candidate.get(endpoint_field) is not None
            and candidate[endpoint_field] != expected_value
        ):
            raise ValueError(
                f"selected response operation {endpoint_field} conflicts "
                "with known input role: "
                f"expected={expected_value!r}; "
                f"actual={candidate[endpoint_field]!r}"
            )

    if "operation" not in candidate:
        raise ValueError("selected response operation lacks operation text")

    bound_operation = {
        "operation": candidate["operation"],
        "response_owner_role": authoritative_operation["response_owner_role"],
        "selection_owner_role": authoritative_operation["selection_owner_role"],
        "selection_required": authoritative_operation["selection_required"],
    }
    for endpoint_field in ENDPOINT_ROLE_FIELDS:
        expected_value = authoritative_operation[endpoint_field]
        bound_operation[endpoint_field] = (
            candidate.get(endpoint_field, NO_ROLE)
            if expected_value == NO_ROLE
            else expected_value
        )
    return bound_operation


def project_progress_evidence(
    evidence_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project active conversation progress as model-visible factual context.

    Args:
        evidence_rows: V2-shaped evidence rows with ``evidence_handle``,
            ``semantic_text``, ``authority`` and a typed ``evidence_ref``
            mapping carrying ``source_kind`` and ``source_id``.

    Returns:
        The projected progress rows in input order, each keeping the handle,
        semantic text, authority label and an optional copied temporal
        provenance mapping.
    """
    progress_evidence: list[dict[str, Any]] = []
    for row in evidence_rows:
        evidence_ref = row["evidence_ref"]
        if evidence_ref["source_kind"] != PROGRESS_EVIDENCE_SOURCE_KIND:
            continue
        source_id = evidence_ref["source_id"]
        if not source_id.startswith(CONVERSATION_PROGRESS_EVENT_SOURCE_PREFIX):
            continue
        projected_row = {
            "evidence_handle": row["evidence_handle"],
            "semantic_text": row["semantic_text"],
            "authority": row["authority"],
        }
        temporal_provenance = row.get("temporal_provenance")
        if isinstance(temporal_provenance, Mapping):
            projected_row["temporal_provenance"] = dict(temporal_provenance)
        progress_evidence.append(projected_row)
    return progress_evidence


def project_required_selection_operations(
    evidence_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project typed required-selection facts from upstream episode evidence.

    Args:
        evidence_rows: V2-shaped evidence rows; only ``episode``-kind rows with
            a parseable semantic payload carrying a validated response
            operation participate.

    Returns:
        The projected operations in input order, each keeping the evidence
        handle, role-explicit content and the validated operation mapping.

    Raises:
        ValueError: when an episode row carries a non-mapping or invalid
            response operation.
    """
    operations: list[dict[str, Any]] = []
    for row in evidence_rows:
        if row["evidence_ref"]["source_kind"] != REQUIRED_SELECTION_SOURCE_KIND:
            continue
        try:
            semantic_payload = json.loads(row["semantic_text"])
        except (TypeError, ValueError):
            continue
        if not isinstance(semantic_payload, Mapping):
            continue
        operation = semantic_payload.get("response_operation")
        if operation is None or not isinstance(operation, Mapping):
            continue
        try:
            validated_operation = validate_dialog_response_operation(operation)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "episode response operation is invalid"
            ) from exc
        if validated_operation["selection_required"] is not True:
            continue
        operations.append({
            "evidence_handle": row["evidence_handle"],
            "role_explicit_content": semantic_payload.get(
                "role_explicit_content",
                "",
            ),
            "response_operation": validated_operation,
        })
    return operations


CONVERSATION_PROGRESS_SOURCE_KIND = "conversation_evidence"
CONVERSATION_PROGRESS_EVENT_SOURCE_PREFIX = (
    "conversation-progress-event:"
)


def project_conversation_progress_evidence(
    evidence_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project active conversation progress as model-visible factual context.

    Args:
        evidence_rows: V2-shaped evidence rows; only ``conversation_evidence``
            kind rows whose source id carries the progress-event prefix
            participate.

    Returns:
        The projected rows in input order, each keeping the episode-local
        handle, semantic text and authority, plus temporal provenance when
        present on the row.
    """
    progress_evidence: list[dict[str, Any]] = []
    for row in evidence_rows:
        evidence_ref = row["evidence_ref"]
        if evidence_ref["source_kind"] != CONVERSATION_PROGRESS_SOURCE_KIND:
            continue
        source_id = evidence_ref["source_id"]
        if not source_id.startswith(
            CONVERSATION_PROGRESS_EVENT_SOURCE_PREFIX
        ):
            continue
        projected_row: dict[str, Any] = {
            "evidence_handle": row["evidence_handle"],
            "semantic_text": row["semantic_text"],
            "authority": row["authority"],
        }
        temporal_provenance = row.get("temporal_provenance")
        if isinstance(temporal_provenance, Mapping):
            projected_row["temporal_provenance"] = dict(
                temporal_provenance
            )
        progress_evidence.append(projected_row)
    return progress_evidence


def project_goal_evidence_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project one authorized evidence row into prompt-safe facts.

    Mirrors the V2 goal-cognition evidence shape: handle, source kind,
    semantic text, authority and the projected provenance role; memory scope
    and temporal provenance travel only when present on the row.
    """
    ref = row["evidence_ref"]
    projected: dict[str, Any] = {
        "handle": row["evidence_handle"],
        "source_kind": ref["source_kind"],
        "semantic_text": row["semantic_text"],
        "authority": row["authority"],
        "provenance_role": project_evidence_provenance_role(
            ref["source_kind"],
            row.get("memory_scope"),
        ),
    }
    if "memory_scope" in row:
        projected["memory_scope"] = row["memory_scope"]
    if "temporal_provenance" in row:
        projected["temporal_provenance"] = dict(
            row["temporal_provenance"]
        )
    return projected


def build_goal_question_tail(
    goal_kind: str,
    state_projection: Mapping[str, Any],
    evidence_handles: Sequence[str],
    *,
    selection_mode: bool = False,
    action_tendencies: Sequence[str] = (),
    branch_intent_guidance: str = "",
    role_bindings: Mapping[str, Mapping[str, str]] = {},
    role_summaries: Mapping[str, str] = {},
    semantic_context: Mapping[str, Any] | None = None,
    appraisal_summaries: Sequence[Mapping[str, Any]] = (),
    evidence_rows: Sequence[Mapping[str, Any]] = (),
    selection_operations: Sequence[Mapping[str, Any]] = (),
    progress_evidence: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Build the deterministic human tail for one isolated goal chain.

    The tail carries every dynamic fact the static branch prompt references:
    the kind label and output shape, the full authorized evidence handle
    domain, that kind's canonical state projection, the projected evidence
    content (supporting rows only in selection mode), required-selection
    operation facts and conversation progress evidence for selection drafts,
    action tendencies and branch intent guidance, role handles with their
    summaries, the filtered semantic context carrying identity, constraints,
    affect, relationship state and continuity fields, and accepted appraisal
    summaries. Sections without content are omitted; every section renders in
    fixed order with structured values as single-line JSON so the tail stays
    deterministic across attempts and cache checkpoints. Selection mode is an
    episode-driven input fact (the authoritative operation requires a
    selection), not a kind fact: a cold ordinary G1a selection draft owns and
    emits ``relational_willingness``; a recurrence carries its already accepted
    stance by deterministic code and omits that model-owned field.

    Args:
        goal_kind: Goal kind label from the closed V2 ``GOAL_KINDS`` set.
        state_projection: Prompt-safe canonical state projection for this chain.
        evidence_handles: Authorized evidence handles for this call.
        selection_mode: True when this chain's output shape is a selection draft.
        action_tendencies: Fixed tendency labels owned by the branch kind.
        branch_intent_guidance: The fixed semantic focus of one non-ordinary
            branch; empty when the caller's gating condition does not hold.
        role_bindings: Private handle-to-role bindings for this context.
        role_summaries: Prompt-safe one-line summaries per bound handle.
        semantic_context: Filtered branch context content, already stripped of
            private keys and separately rendered fields by the caller; None or
            empty renders no section.
        appraisal_summaries: Accepted appraisal rows from earlier reduction, in
            reduction order; empty when no accepted prefix exists yet.
        evidence_rows: Projected prompt-safe evidence rows for the evidence
            section (supporting partition only in selection mode).
        selection_operations: Required-selection operation facts for one
            authoritative episode operation; non-empty only in selection mode.
        progress_evidence: Projected conversation progress rows; non-empty
            only in selection mode, where they render under their own section.

    Returns:
        The joined deterministic tail text.

    Raises:
        ValueError: when ``goal_kind`` is outside the closed kind set.
    """
    goal_kind_owns_relational_willingness(goal_kind)
    lines = [
        '# 目标类型',
        goal_kind,
        '# 输出形态',
        "selection draft" if selection_mode else "goal bid",
    ]

    # Only the ordinary kind owns the relational stance in its goal-bid shape;
    # state that fact once and let selection mode travel the code-carried copy.
    if not selection_mode and goal_kind == ORDINARY_GOAL_KIND:
        lines.append(
            '本分支拥有 relational_willingness，必须输出完整关系立场。'
        )

    handle_lines = [f"- {handle}" for handle in evidence_handles]
    if handle_lines:
        lines.extend(["# 授权证据 handle", *handle_lines])

    projected_evidence_lines = [
        f"- {row['handle']}={_json_line(row)}" for row in evidence_rows
    ]
    if projected_evidence_lines:
        evidence_header = (
            "# 支撑证据" if selection_mode else "# 对话证据"
        )
        lines.extend([evidence_header, *projected_evidence_lines])

    operation_lines = [
        f"- {operation['evidence_handle']}={_json_line(operation)}"
        for operation in selection_operations
    ]
    if operation_lines:
        lines.extend(["# 必需选择操作", *operation_lines])

    progress_lines = [
        f"- {row['evidence_handle']}={_json_line(row)}"
        for row in progress_evidence
    ]
    if progress_lines:
        lines.extend(["# 对话进度证据", *progress_lines])

    tendency_lines = [f"- {tendency}" for tendency in action_tendencies]
    if tendency_lines:
        lines.extend(["# 行动倾向", *tendency_lines])

    if branch_intent_guidance:
        lines.extend(["# 分支关注点", str(branch_intent_guidance)])

    role_lines = [
        f"- {handle}: {role_summaries.get(handle, '')}"
        for handle in sorted(role_bindings)
    ]
    if role_lines:
        lines.extend(["# 角色 handle", *role_lines])

    state_lines = [
        f"{key}={value}" for key, value in dict(state_projection).items()
    ]
    if state_lines:
        lines.extend(["# 状态投影", *state_lines])

    context_content = (
        semantic_context if semantic_context is not None else {}
    )
    context_lines = [
        f"- {key}={_json_line(value)}"
        for key, value in dict(context_content).items()
    ]
    if context_lines:
        lines.extend(["# 语义上下文", *context_lines])

    summary_blocks: list[str] = []
    for summary in appraisal_summaries:
        block = [f"- {summary['question_id']}: {summary['explanation']}"]
        block.extend(f"  - {value}" for value in summary["propositions"])
        summary_blocks.append("\n".join(block))
    if summary_blocks:
        lines.extend(["# 已接受评价摘要", *summary_blocks])

    return "\n".join(lines)


def _json_line(value: Any) -> str:
    """Render one structured value as a single deterministic JSON line."""
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def build_goal_repair_instruction(
    goal_kind: str,
    selection_mode: bool,
    detail: str | None,
    role_handles: frozenset[str],
) -> str:
    """Render one bounded local repair instruction for a rejected goal draft.

    Args:
        goal_kind: The closed V2 goal kind owning the rejected chain; its field
            contract is restated exactly as the validator owns it, including the
            ordinary-response ``relational_willingness`` field when owned and
            absent in selection mode.
        selection_mode: True when the episode carries a required selection
            operation and the draft follows the selection-draft field set
            instead of the goal bid field set.
        detail: The exact validator error of the rejected attempt, or None when
            the raw output could not be parsed as a JSON object at all.
        role_handles: Complete authorized role handle domain; the closed target
            value set restated so replacement output stays in-domain.

    Returns:
        Deterministic instruction text carrying the exact violation, the closed
            field and value sets, and a complete-replacement directive; no new
            semantic guidance enters the message.
    """
    goal_kind_owns_relational_willingness(goal_kind)
    error_text = detail if detail is not None else "原始输出无法解析为 JSON 对象"
    if selection_mode:
        field_lines = ", ".join(SELECTION_GOAL_DRAFT_FIELDS)
    else:
        field_names = list(GOAL_BID_FIELDS)
        if goal_kind_owns_relational_willingness(goal_kind):
            field_names.append("relational_willingness")
        field_lines = ", ".join(field_names)
    allowed_role_handles = (
        ", ".join(sorted(role_handles)) if role_handles else "（无）"
    )
    lines: list[str] = [
        "# 修复请求",
        f"上一条输出未通过结构校验：{error_text}",
        f"顶层字段集合必须恰为 {field_lines}。",
        (
            f"target_role_handles 只允许取值 [{allowed_role_handles}]，最多 "
            f"{GOAL_BID_ROLE_HANDLE_LIMIT} 个；evidence_handles 只允许请求中列出的授权证据 handle。"
        ),
        (
            f"confidence 最长 {GOAL_CONFIDENCE_CHAR_LIMIT} 字符；expected_consequences "
            f"是字符串列表，最多 {EXPECTED_CONSEQUENCE_ITEM_LIMIT} 条，每条最长 "
            f"{CONSEQUENCE_CHAR_LIMIT} 字符。"
        ),
        "现在重新输出一个完整替换的 JSON 对象，不要重复上一条错误内容。",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class GoalBidDisposition:
    """Fail-closed disposition of one isolated goal chain.

    Attributes:
        kind: The goal kind label this disposition belongs to.
        available: True only when the chain accepted a complete bid state; an
            exhausted or failed chain is always False with no substitution from
            any other goal kind.
        bid: The normalized accepted bid state, or None when unavailable.
        error_code: The exact typed failure code of the last recorded failure,
            or None when available.
    """

    kind: str
    available: bool
    bid: dict[str, Any] | None = None
    error_code: str | None = None


def resolve_goal_disposition(
    goal_kind: str,
    chain_outcome: object,
) -> GoalBidDisposition:
    """Resolve one isolated goal chain into its fail-closed disposition.

    An accepted stage materializes its normalized bid state; an exhausted or
    failed chain reports ``available=False`` with the exact typed error code of
    its last recorded failure and no bid. No other goal kind substitutes, so a
    required-selection exhaustion stays closed without falling back to any
    other branch.

    Args:
        goal_kind: Goal kind label matching the single stage of this chain.
        chain_outcome: Executor ``ChainOutcome`` for one isolated goal chain.

    Returns:
        The fail-closed disposition for this chain.

    Raises:
        ValueError: when the outcome does not record exactly the requested
            goal-kind stage or holds no typed failure for an unavailable chain.
    """
    results = chain_outcome.results
    if not results or results[-1].stage_name != goal_kind:
        raise ValueError(
            f"goal outcome lacks the {goal_kind!r} stage record"
        )

    last_result = results[-1]
    if last_result.accepted and last_result.local_state is not None:
        return GoalBidDisposition(goal_kind, True, bid=dict(last_result.local_state))

    failure = next(
        (
            result.failure
            for result in reversed(results)
            if result.accepted is False and result.failure is not None
        ),
        None,
    )
    if failure is None:
        raise ValueError(
            f"goal chain {goal_kind!r} has no typed failure record"
        )
    return GoalBidDisposition(goal_kind, False, error_code=failure.error_code)


def _validate_active_goal_roster(
    branch_roster: Sequence[Mapping[str, object]],
) -> list[str]:
    """Validate canonical branch labels and their goal-kind bindings.

    The active-group model contract addresses each registered branch by its
    stable branch label. The label and the V2 goal kind are separate fields:
    several registered branches intentionally use a more specific branch name
    while sharing a different closed goal-kind value.

    Args:
        branch_roster: Deterministic active-branch rows supplied to the group
            question and its validator.

    Returns:
        The validated branch labels in their frozen roster order.

    Raises:
        TypeError: If a roster row is not a mapping.
        ValueError: If a row has an unknown, malformed, duplicate, or
            incorrectly bound branch label or goal kind.
    """

    branch_ids: list[str] = []
    seen_branch_ids: set[str] = set()
    for roster_row in branch_roster:
        if not isinstance(roster_row, Mapping):
            raise TypeError("active goal roster rows must be mappings")
        branch_id = roster_row.get("branch_id")
        if (
            not isinstance(branch_id, str)
            or not branch_id
            or branch_id != branch_id.strip()
            or branch_id not in DEFAULT_BRANCH_DEFINITIONS
        ):
            raise ValueError("active goal roster branch_id is invalid")
        if branch_id in seen_branch_ids:
            raise ValueError("active goal roster branch_id is duplicated")
        goal_kind = roster_row.get("goal_kind")
        if not isinstance(goal_kind, str) or goal_kind not in GOAL_KINDS:
            raise ValueError("active goal roster goal_kind is invalid")
        if DEFAULT_BRANCH_DEFINITIONS[branch_id].goal_kind != goal_kind:
            raise ValueError(
                "active goal roster branch and goal kind binding is invalid"
            )
        seen_branch_ids.add(branch_id)
        branch_ids.append(branch_id)
    return branch_ids


def validate_active_goal_group_output(
    raw: object,
    *,
    branch_roster: Sequence[Mapping[str, object]],
    evidence_handles: set[str],
    role_handles: set[str],
) -> list[dict[str, Any]]:
    """Validate an ordered active-branch group bid output."""

    branch_ids = _validate_active_goal_roster(branch_roster)
    if not isinstance(raw, Mapping) or set(raw) != {"bids"}:
        raise ValueError("active goal group output fields are not exact")
    bids = raw["bids"]
    if not isinstance(bids, list):
        raise TypeError("active goal group bids must be a list")
    if len(bids) != len(branch_ids):
        raise ValueError("active goal group bid count must equal the roster")

    validated_bids: list[dict[str, Any]] = []
    for branch_id, bid_row in zip(branch_ids, bids):
        if not isinstance(bid_row, Mapping):
            raise TypeError("active goal group rows must be mappings")
        if bid_row.get("branch_id") != branch_id:
            raise ValueError("active goal group bid order must equal the roster")
        candidate = dict(bid_row)
        candidate.pop("branch_id", None)
        validated = validate_goal_bid_draft(
            candidate,
            evidence_handles=evidence_handles,
            role_handles=role_handles,
            require_relational_willingness=False,
            episode_handles=None,
        )
        validated["branch_id"] = branch_id
        validated_bids.append(validated)
    return validated_bids


def salvage_active_goal_group_output(
    raw: object,
    *,
    branch_roster: Sequence[Mapping[str, object]],
    evidence_handles: set[str],
    role_handles: set[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Retain independently valid siblings from a failed final group output.

    This is a post-exhaustion boundary, not a repair path.  It never changes
    a candidate field and never accepts an unlisted, duplicated, missing, or
    independently invalid branch.  The returned rows remain in the frozen
    roster order and the failure map contains one typed disposition per
    omitted roster branch.
    """

    branch_ids = _validate_active_goal_roster(branch_roster)
    failures: dict[str, str] = {}
    if not isinstance(raw, Mapping) or set(raw) != {"bids"}:
        return [], {
            branch_id: "active_goal_group_contract_invalid"
            for branch_id in branch_ids
        }
    bids = raw["bids"]
    if not isinstance(bids, list):
        return [], {
            branch_id: "active_goal_group_contract_invalid"
            for branch_id in branch_ids
        }

    by_branch: dict[str, list[Mapping[str, object]]] = {}
    for bid in bids:
        if not isinstance(bid, Mapping):
            continue
        branch_id = bid.get("branch_id")
        if isinstance(branch_id, str):
            by_branch.setdefault(branch_id, []).append(bid)

    validated_rows: list[dict[str, Any]] = []
    for roster_row in branch_roster:
        branch_id = roster_row["branch_id"]
        candidates = by_branch.get(branch_id, [])
        if not candidates:
            failures[branch_id] = "active_goal_group_missing_branch"
            continue
        if len(candidates) != 1:
            failures[branch_id] = "active_goal_group_duplicate_branch"
            continue
        candidate = dict(candidates[0])
        candidate.pop("branch_id", None)
        try:
            validated = validate_goal_bid_draft(
                candidate,
                evidence_handles=evidence_handles,
                role_handles=role_handles,
                require_relational_willingness=False,
                episode_handles=None,
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            failures[branch_id] = "active_goal_group_invalid_bid"
            continue
        validated["branch_id"] = branch_id
        validated_rows.append(validated)
    return validated_rows, failures


def validate_selection_goal_draft(
    parsed: object,
    *,
    evidence_handles: set[str],
    role_handles: set[str],
    required_evidence_handles: set[str],
    required_operations: Sequence[Mapping[str, Any]] | None = None,
    episode_handles: set[str] | None = None,
    require_relational_willingness: bool = False,
    maximum_evidence_handles: int,
) -> dict[str, Any]:
    """Validate one authoritative selection and required operation coverage."""

    if not isinstance(parsed, Mapping):
        raise ValueError("selection goal draft must be an object")
    parsed, _normalizations = _normalize_nonowning_goal_fields(
        parsed,
        branch_id="unknown",
        require_relational_willingness=require_relational_willingness,
    )
    required_fields = {
        "selection",
        "selected_response_operation",
        "reason",
        "private_monologue",
        "target_role_handles",
        "evidence_handles",
        "expected_consequences",
        "confidence",
    }
    if require_relational_willingness:
        required_fields.add("relational_willingness")
    if set(parsed) != required_fields:
        raise ValueError("selection goal draft fields are not exact")
    for field_name, maximum in (
        ("selection", 500),
        ("reason", 500),
        ("private_monologue", 500),
        ("confidence", 40),
    ):
        _bounded_text(parsed[field_name], field_name, maximum)
    target_roles = _handles(
        parsed["target_role_handles"],
        role_handles,
        "role",
        maximum_handles=GOAL_BID_ROLE_HANDLE_LIMIT,
    )
    cited_evidence = _handles(
        parsed["evidence_handles"],
        evidence_handles,
        "evidence",
        maximum_handles=maximum_evidence_handles,
    )
    if not required_evidence_handles.issubset(cited_evidence):
        raise ValueError(
            "selection goal lacks required evidence coverage"
        )
    if not required_operations:
        raise ValueError(
            "selection goal requires input response operations"
        )
    raw_selected_operation = parsed["selected_response_operation"]
    if not isinstance(raw_selected_operation, Mapping):
        raise ValueError("selected response operation must be an object")
    selected_operation = None
    for operation_row in required_operations:
        if not isinstance(operation_row, Mapping):
            raise ValueError("selection goal response operation row is invalid")
        input_operation = operation_row.get("response_operation")
        authoritative_operation = validate_dialog_response_operation(
            input_operation
        )
        if selected_operation is None:
            selected_operation = bind_selected_response_operation(
                raw_selected_operation,
                authoritative_operation,
            )
        validate_selected_response_operation(
            selected_operation,
            input_operation,
        )
    consequences = parsed["expected_consequences"]
    if not isinstance(consequences, list) or not 1 <= len(consequences) <= 8:
        raise ValueError("selection goal consequences are invalid")
    for consequence in consequences:
        _bounded_text(consequence, "consequence", 240)
    result = dict(parsed)
    result["selected_response_operation"] = selected_operation
    result["target_role_handles"] = target_roles
    result["evidence_handles"] = cited_evidence
    result["expected_consequences"] = list(consequences)
    if require_relational_willingness:
        relational_candidate = parsed["relational_willingness"]
        if not isinstance(relational_candidate, Mapping):
            raise ValueError("relational willingness must be an object")
        if "schema_version" in relational_candidate:
            raise ValueError(
                "relational willingness schema_version is code-owned"
            )
        relational_candidate = dict(relational_candidate)
        relational_candidate["schema_version"] = (
            RELATIONAL_WILLINGNESS_SCHEMA_VERSION
        )
        relational_decision = validate_relational_willingness(
            relational_candidate,
            evidence_handles=evidence_handles,
            episode_handles=episode_handles,
        )
        result["relational_willingness"] = relational_decision
    return result

def selection_goal_draft_to_goal_bid(
    selection_draft: Mapping[str, Any],
    *,
    branch_id: str,
    include_relational_willingness: bool = True,
) -> GoalBidDraftV2:
    """Map one authoritative selection string into the complete bid shape."""

    selection = selection_draft["selection"]
    if not isinstance(selection, str):
        raise TypeError("validated selection must be text")
    result: GoalBidDraftV2 = {
        "intention": selection,
        "desired_outcome": selection,
        "concrete_detail": selection,
        "reason": selection_draft["reason"],
        "private_monologue": selection_draft["private_monologue"],
        "target_role_handles": list(
            selection_draft["target_role_handles"]
        ),
        "evidence_handles": list(selection_draft["evidence_handles"]),
        "expected_consequences": list(
            selection_draft["expected_consequences"]
        ),
        "confidence": selection_draft["confidence"],
        "selected_response_operation": dict(
            selection_draft["selected_response_operation"]
        ),
    }
    if branch_id == "ordinary_response" and include_relational_willingness:
        result["relational_willingness"] = dict(
            selection_draft["relational_willingness"]
        )
    return result

def validate_goal_bid_draft(
    parsed: object,
    *,
    evidence_handles: set[str],
    role_handles: set[str],
    require_relational_willingness: bool = False,
    episode_handles: set[str] | None = None,
) -> GoalBidDraftV2:
    """Validate model-owned fields before any complete bid is constructed."""

    if not isinstance(parsed, Mapping):
        raise ValueError("goal bid draft must be an object")
    parsed, _normalizations = _normalize_nonowning_goal_fields(
        parsed,
        branch_id="unknown",
        require_relational_willingness=require_relational_willingness,
    )
    required = {
        "intention",
        "desired_outcome",
        "concrete_detail",
        "reason",
        "private_monologue",
        "target_role_handles",
        "evidence_handles",
        "expected_consequences",
        "confidence",
    }
    if require_relational_willingness:
        required.add("relational_willingness")
    if set(parsed) != required:
        raise ValueError("goal bid draft fields are not exact")
    for field_name in (
        "intention",
        "desired_outcome",
        "concrete_detail",
        "reason",
        "private_monologue",
    ):
        _bounded_text(parsed[field_name], field_name, 500)
    _bounded_text(parsed["confidence"], "confidence", 40)
    target_roles = _handles(
        parsed["target_role_handles"],
        role_handles,
        "role",
        maximum_handles=GOAL_BID_ROLE_HANDLE_LIMIT,
    )
    cited_evidence = _handles(
        parsed["evidence_handles"],
        evidence_handles,
        "evidence",
        maximum_handles=GOAL_BID_EVIDENCE_HANDLE_LIMIT,
    )
    consequences = parsed["expected_consequences"]
    if not isinstance(consequences, list) or not 1 <= len(consequences) <= 8:
        raise ValueError("goal bid consequences are invalid")
    for consequence in consequences:
        _bounded_text(consequence, "consequence", 240)
    if require_relational_willingness:
        relational_candidate = parsed["relational_willingness"]
        if not isinstance(relational_candidate, Mapping):
            raise ValueError("relational willingness must be an object")
        if "schema_version" in relational_candidate:
            raise ValueError(
                "relational willingness schema_version is code-owned"
            )
        relational_candidate = dict(relational_candidate)
        relational_candidate["schema_version"] = (
            RELATIONAL_WILLINGNESS_SCHEMA_VERSION
        )
        relational_decision = validate_relational_willingness(
            relational_candidate,
            evidence_handles=evidence_handles,
            episode_handles=episode_handles,
        )
    result = dict(parsed)
    result["target_role_handles"] = target_roles
    result["evidence_handles"] = cited_evidence
    result["expected_consequences"] = consequences
    if require_relational_willingness:
        result["relational_willingness"] = relational_decision
    return result

def _handles(
    value: Any,
    allowed: set[str],
    label: str,
    *,
    maximum_handles: int,
) -> list[str]:
    """Validate a duplicate-free bounded handle partition."""

    if (
        not isinstance(value, list)
        or len(value) > maximum_handles
    ):
        raise ValueError(f"{label} handles are invalid")
    if len(value) != len(set(value)) or any(handle not in allowed for handle in value):
        raise ValueError(f"{label} handles are not permitted")
    return list(value)

def _normalize_nonowning_goal_fields(
    parsed: object,
    *,
    branch_id: str,
    require_relational_willingness: bool,
) -> tuple[object, list[dict[str, str]]]:
    """Strip one non-owning relational field before exact-shape validation."""

    if (
        require_relational_willingness
        or not isinstance(parsed, Mapping)
        or "relational_willingness" not in parsed
    ):
        return parsed, []
    normalized = dict(parsed)
    normalized.pop("relational_willingness")
    return normalized, [{
        "branch": branch_id,
        "field_name": "relational_willingness",
        "reason": "non_owning_branch_field",
    }]

def _bounded_text(value: Any, label: str, maximum: int) -> None:
    """Validate bounded model-owned prose."""

    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} is invalid")

def build_goal_output_contract(
    *,
    evidence_handles: set[str],
    episode_evidence_handles: set[str],
    required_evidence_handles: set[str],
    role_bindings: Mapping[str, Any],
    selection_required: bool,
    require_relational_willingness: bool,
    maximum_evidence_handles: int,
    authoritative_operation: Mapping[str, Any] | None = None,
    recurrence_relational_willingness: bool = False,
) -> dict[str, Any]:
    """Project the exact current-run output contract for one goal mode."""

    if selection_required:
        top_level_fields = [
            "selection",
            "selected_response_operation",
            "reason",
            "private_monologue",
            "target_role_handles",
            "evidence_handles",
            "expected_consequences",
            "confidence",
        ]
        field_types = {
            "selection": "non_empty_string_max_500",
            "selected_response_operation": (
                "per_input_writable_selected_response_operation"
            ),
            "reason": "non_empty_string_max_500",
            "private_monologue": "non_empty_string_max_500",
            "target_role_handles": "array_of_strings_max_8",
            "evidence_handles": (
                f"array_of_strings_max_{maximum_evidence_handles}"
            ),
            "expected_consequences": (
                "non_empty_array_of_strings_max_8_each_max_240"
            ),
            "confidence": "non_empty_string_max_40",
        }
    else:
        top_level_fields = [
            "intention",
            "desired_outcome",
            "concrete_detail",
            "reason",
            "private_monologue",
            "target_role_handles",
            "evidence_handles",
            "expected_consequences",
            "confidence",
        ]
        field_types = {
            "intention": "non_empty_string_max_500",
            "desired_outcome": "non_empty_string_max_500",
            "concrete_detail": "non_empty_string_max_500",
            "reason": "non_empty_string_max_500",
            "private_monologue": "non_empty_string_max_500",
            "target_role_handles": "array_of_strings_max_8",
            "evidence_handles": (
                f"array_of_strings_max_{maximum_evidence_handles}"
            ),
            "expected_consequences": (
                "non_empty_array_of_strings_max_8_each_max_240"
            ),
            "confidence": "non_empty_string_max_40",
        }

    if require_relational_willingness:
        top_level_fields.append("relational_willingness")
        field_types["relational_willingness"] = "object"

    contract: dict[str, Any] = {
        "top_level_fields": top_level_fields,
        "field_types": field_types,
        "confidence_type": "string_descriptor",
        "allowed_role_handles": sorted(role_bindings),
        "allowed_evidence_handles": sorted(evidence_handles),
        "required_evidence_handles": sorted(required_evidence_handles),
        "current_episode_evidence_handles": sorted(
            episode_evidence_handles
        ),
        "role_handles_forbidden_in_evidence_handles": sorted(role_bindings),
        "max_role_handles": GOAL_BID_ROLE_HANDLE_LIMIT,
        "max_evidence_handles": maximum_evidence_handles,
        "unavailable_target_role_handles": [],
        "target_role_handles_rule": (
            "use [] when no permitted target role is grounded; never invent "
            "a handle"
        ),
        "bounds": {
            "target_role_handles": {
                "minimum_items": 0,
                "maximum_items": GOAL_BID_ROLE_HANDLE_LIMIT,
            },
            "evidence_handles": {
                "minimum_items": 0,
                "maximum_items": maximum_evidence_handles,
            },
            "expected_consequences": {
                "minimum_items": 1,
                "maximum_items": 8,
                "item_maximum_chars": 240,
            },
            "model_text": {
                "minimum_chars": 1,
                "maximum_chars": 500,
            },
            "confidence": {
                "minimum_chars": 1,
                "maximum_chars": 40,
            },
        },
    }
    if selection_required:
        if not isinstance(authoritative_operation, Mapping):
            raise ValueError(
                "selection output contract requires authoritative operation"
            )
        writable_fields = ["operation"]
        optional_fields: list[str] = []
        selected_field_types = {
            "operation": (
                f"non_empty_string_max_{MAX_RESPONSE_OPERATION_CHARS}"
            ),
        }
        code_owned_fields: dict[str, Any] = {
            "response_owner_role": authoritative_operation[
                "response_owner_role"
            ],
            "selection_owner_role": authoritative_operation[
                "selection_owner_role"
            ],
            "selection_required": authoritative_operation[
                "selection_required"
            ],
        }
        for endpoint_field in (
            "embedded_actor_role",
            "embedded_target_role",
        ):
            endpoint_value = authoritative_operation[endpoint_field]
            if endpoint_value == NO_ROLE:
                writable_fields.append(endpoint_field)
                optional_fields.append(endpoint_field)
                selected_field_types[endpoint_field] = (
                    "one_of_response_operation_roles"
                )
            else:
                code_owned_fields[endpoint_field] = endpoint_value
        selected_response_operation_contract = {
            "writable_fields": writable_fields,
            "required_fields": ["operation"],
            "optional_fields": optional_fields,
            "field_types": selected_field_types,
            "role_values": [
                CURRENT_CHARACTER_ROLE,
                CURRENT_USER_ROLE,
                OTHER_PARTICIPANT_ROLE,
                NO_ROLE,
            ],
            "code_owned_fields": code_owned_fields,
            "rule": (
                "write one concrete embedded action in operation; use only "
                "writable_fields; authoritative wording is acceptable when "
                "it already states the usable selected action"
            ),
        }
        contract["selected_response_operation"] = (
            selected_response_operation_contract
        )
        contract["selection_required"] = True
    if require_relational_willingness or recurrence_relational_willingness:
        sensitive_stance_order = [
            "reject",
            "deflect",
            "negotiate",
            "conditional_accept",
            "accept",
        ]
        real_relationship_states = sorted(
            RELATIONAL_CURRENT_USER_RELATIONSHIP_STATE_VALUES
            - {"not_applicable"}
        )
        contract["relational_willingness_contract"] = {
            "mode": (
                "model_output"
                if require_relational_willingness
                else "validated_carry_forward"
            ),
            "required_fields": [
                "applicability",
                "stance",
                "current_user_relationship_state",
                "reason",
                "evidence_handles",
            ],
            "applicability_values": sorted(RELATIONAL_APPLICABILITY_VALUES),
            "stance_values": sorted(RELATIONAL_STANCE_VALUES),
            "current_user_relationship_state_values": sorted(
                RELATIONAL_CURRENT_USER_RELATIONSHIP_STATE_VALUES
            ),
            "relationship_state_rule": (
                "relationship state is descriptive; every real state can "
                "pair with one ordered sensitive stance"
            ),
            "non_sensitive_pairing": {
                "applicability": "not_relationship_sensitive",
                "stance": "not_applicable",
                "current_user_relationship_state": "not_applicable",
            },
            "sensitive_pairing": {
                "applicability": "relationship_sensitive",
                "stance_values_in_order": sensitive_stance_order,
                "current_user_relationship_state_values": (
                    real_relationship_states
                ),
            },
            "reason": "non_empty_simplified_chinese_string",
            "maximum_reason_chars": RELATIONAL_WILLINGNESS_MAX_REASON_CHARS,
            "allowed_evidence_handles": sorted(evidence_handles),
            "current_episode_evidence_handles": sorted(
                episode_evidence_handles
            ),
            "minimum_evidence_handles": 1,
            "minimum_current_episode_evidence_handles": 1,
            "maximum_evidence_handles": (
                MAX_RELATIONAL_WILLINGNESS_EVIDENCE_HANDLES
            ),
        }
    if recurrence_relational_willingness:
        contract["recurrence_relational_willingness"] = {
            "mode": "validated_carry_forward",
            "source": "current_turn_relational_willingness",
            "current_episode_evidence_handles": sorted(
                episode_evidence_handles
            ),
            "minimum_current_episode_evidence_handles": 1,
            "action": "copy_and_revalidate_after_goal_validation",
            "model_regeneration": False,
        }
    return contract
