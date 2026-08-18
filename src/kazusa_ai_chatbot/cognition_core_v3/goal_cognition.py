"""V3 cache-affine goal cognition: isolated single-stage goal chains.

Every goal kind in ``GOAL_KINDS`` runs as one isolated single-stage chain on a
fresh canonical projection. All goal chains share one byte-identical static
system prompt; per-chain dynamic facts (goal kind, output shape, authorized
handles, state projection) stay in human tails only. The bid output contract
and the relational-willingness sub-contract are unchanged V2 contracts: this
module reuses the V2 closed value sets and the decision-level validator as the
single source of truth, so ``ordinary_response`` remains the only owner of
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

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    RELATIONAL_WILLINGNESS_SCHEMA_VERSION,
    validate_relational_willingness,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import GOAL_KINDS
from kazusa_ai_chatbot.cognition_episode import (
    NO_ROLE,
    validate_dialog_response_operation,
)

# Closed limits ported from the V2 goal cognition contract.
GOAL_BID_EVIDENCE_HANDLE_LIMIT = 9
GOAL_BID_ROLE_HANDLE_LIMIT = 8
GOAL_BID_TEXT_CHAR_LIMIT = 500
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
goal bid 只返回一个 JSON 对象，字段恰好是 intention、desired_outcome、concrete_detail、reason、private_monologue、target_role_handles、evidence_handles、expected_consequences 和 confidence；当本分支拥有 relational_willingness 时还含该字段。relational_willingness 的字段恰好是 applicability、stance、current_user_relationship_state、reason 和 evidence_handles，schema_version 由代码绑定。
selection draft 只返回一个严格 JSON 对象，字段恰好是 selection、selected_response_operation、reason、private_monologue、target_role_handles、evidence_handles、expected_consequences 和 confidence；按输入绑定的 writable_fields 输出 selected_response_operation，方向字段保持代码绑定；selection 直接写出当前角色的具体选择。
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


def _bounded_text(value: object, label: str, maximum: int) -> None:
    """Validate one bounded narrative field.

    Args:
        value: Candidate field value from a model-owned draft.
        label: Field label used in the error message.
        maximum: Inclusive character limit for the non-empty text.

    Raises:
        ValueError: when the value is not non-empty text within ``maximum``.
    """
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} is invalid")


def validate_handle_list(
    value: object,
    authorized_handles: set[str],
    label: str,
    limit: int,
) -> list[str]:
    """Validate a typed handle list against its authorized domain.

    Args:
        value: Candidate handle list from a model-owned draft.
        authorized_handles: Complete authorized handle set for this call.
        label: Field label used in the error message.
        limit: Inclusive maximum number of handles.

    Returns:
        The normalized handle list, preserving candidate order.

    Raises:
        ValueError: when any handle is missing from ``authorized_handles`` or
            the list exceeds ``limit``.
    """
    if not isinstance(value, list):
        raise ValueError(f"goal bid {label} handles are invalid")
    normalized = []
    for handle in value:
        if (
            not isinstance(handle, str)
            or not handle.strip()
            or handle not in authorized_handles
        ):
            raise ValueError(f"goal bid {label} handles are invalid")
        normalized.append(handle)
    if len(normalized) > limit:
        raise ValueError(f"goal bid {label} handles exceed the limit")
    return normalized


def _validate_expected_consequences(value: object) -> list[str]:
    """Validate the bounded non-empty consequence list.

    Args:
        value: Candidate ``expected_consequences`` field.

    Returns:
        The consequence list, preserving candidate order.

    Raises:
        ValueError: when the list is empty, over-sized, or holds an invalid
            narrative item.
    """
    if not isinstance(value, list) or len(value) > EXPECTED_CONSEQUENCE_ITEM_LIMIT:
        raise ValueError("goal bid consequences are invalid")
    for consequence in value:
        _bounded_text(consequence, "consequence", CONSEQUENCE_CHAR_LIMIT)
    return list(value)


def validate_relational_decision(
    candidate: object,
    *,
    evidence_handles: set[str],
) -> dict[str, Any]:
    """Validate one relational-willingness decision owned by ordinary_response.

    The sub-contract is the unchanged V2 contract: this function delegates to
    ``validate_relational_willingness`` with the code-stamped schema version so
    the closed value sets and cross-field rules stay a single source of truth.

    Args:
        candidate: Model-owned decision object without ``schema_version``.
        evidence_handles: Complete authorized handle set for this call; cited
            handles outside it are a structural contract error.

    Returns:
        The validated decision with the code-stamped schema version.

    Raises:
        ValueError: when ``candidate`` is not a mapping or carries a model-
            submitted ``schema_version`` field.
        CognitionContractError: when the V2 decision-level contract (closed
            value sets, cross-field rules, Simplified Chinese reason) is
            violated by the delegated validator.
    """
    if not isinstance(candidate, Mapping):
        raise ValueError("relational willingness must be an object")
    decision = dict(candidate)
    if "schema_version" in decision:
        raise ValueError(
            "relational willingness schema_version is code-owned"
        )
    decision["schema_version"] = RELATIONAL_WILLINGNESS_SCHEMA_VERSION
    return validate_relational_willingness(
        decision,
        evidence_handles=evidence_handles,
    )


def validate_goal_bid_draft(
    candidate: object,
    *,
    goal_kind: str,
    evidence_handles: set[str],
    role_handles: set[str],
) -> dict[str, Any]:
    """Validate one model-owned goal bid draft before materialization.

    The exact field set depends on the kind owner rule: only ``ordinary_response``
    carries ``relational_willingness``, and every other known kind rejects the
    field as a structural contract error.

    Args:
        candidate: Parsed model output for one goal chain.
        goal_kind: Goal kind label from the closed V2 ``GOAL_KINDS`` set.
        evidence_handles: Complete authorized evidence handle set.
        role_handles: Complete authorized role handle set.

    Returns:
        The normalized draft with validated handle lists and, when owned, the
        validated relational decision.

    Raises:
        ValueError: on any field-set, type, bound, or domain violation of the
            model-owned draft.
        CognitionContractError: when the owned relational decision violates the
            V2 decision-level contract during delegation.
    """
    if not isinstance(candidate, Mapping):
        raise ValueError("goal bid draft must be an object")
    owns_willingness = goal_kind_owns_relational_willingness(goal_kind)
    expected_fields = set(GOAL_BID_FIELDS)
    if owns_willingness:
        expected_fields.add("relational_willingness")
    if set(candidate) != expected_fields:
        raise ValueError("goal bid draft fields are not exact")

    for field_name in (
        "intention",
        "desired_outcome",
        "concrete_detail",
        "reason",
        "private_monologue",
    ):
        _bounded_text(candidate[field_name], field_name, GOAL_BID_TEXT_CHAR_LIMIT)
    _bounded_text(candidate["confidence"], "confidence", GOAL_CONFIDENCE_CHAR_LIMIT)

    normalized = dict(candidate)
    normalized["target_role_handles"] = validate_handle_list(
        candidate["target_role_handles"],
        role_handles,
        "role",
        GOAL_BID_ROLE_HANDLE_LIMIT,
    )
    normalized["evidence_handles"] = validate_handle_list(
        candidate["evidence_handles"],
        evidence_handles,
        "evidence",
        GOAL_BID_EVIDENCE_HANDLE_LIMIT,
    )
    normalized["expected_consequences"] = _validate_expected_consequences(
        candidate["expected_consequences"]
    )
    if owns_willingness:
        normalized["relational_willingness"] = validate_relational_decision(
            candidate["relational_willingness"],
            evidence_handles=evidence_handles,
        )
    return normalized


def validate_selection_goal_draft(
    candidate: object,
    *,
    evidence_handles: set[str],
    role_handles: set[str],
) -> dict[str, Any]:
    """Validate one model-owned required-selection draft.

    Args:
        candidate: Parsed model output for a selection-mode goal chain.
        evidence_handles: Complete authorized evidence handle set.
        role_handles: Complete authorized role handle set.

    Returns:
        The normalized draft with validated handle lists and an operation
        mapping ready for ``bind_selected_response_operation``.

    Raises:
        ValueError: on any field-set, type, bound, or domain violation.
    """
    if not isinstance(candidate, Mapping):
        raise ValueError("selection goal draft must be an object")
    if set(candidate) != set(SELECTION_GOAL_DRAFT_FIELDS):
        raise ValueError("selection goal draft fields are not exact")

    for field_name in ("selection", "reason", "private_monologue"):
        _bounded_text(
            candidate[field_name],
            f"goal bid {field_name}",
            GOAL_BID_TEXT_CHAR_LIMIT,
        )
    _bounded_text(candidate["confidence"], "confidence", GOAL_CONFIDENCE_CHAR_LIMIT)

    normalized = dict(candidate)
    if not isinstance(normalized["selected_response_operation"], Mapping):
        raise ValueError("selected response operation must be an object")
    normalized["target_role_handles"] = validate_handle_list(
        candidate["target_role_handles"],
        role_handles,
        "role",
        GOAL_BID_ROLE_HANDLE_LIMIT,
    )
    normalized["evidence_handles"] = validate_handle_list(
        candidate["evidence_handles"],
        evidence_handles,
        "evidence",
        GOAL_BID_EVIDENCE_HANDLE_LIMIT,
    )
    normalized["expected_consequences"] = _validate_expected_consequences(
        candidate["expected_consequences"]
    )
    return normalized


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


def build_goal_question_tail(
    goal_kind: str,
    state_projection: Mapping[str, Any],
    evidence_handles: Sequence[str],
    *,
    selection_mode: bool = False,
) -> str:
    """Build the deterministic human tail for one isolated goal chain.

    The tail carries every dynamic fact for this chain: its kind label, output
    shape and ownership note, authorized evidence handles and canonical state
    projection. No sibling-goal, appraisal or action content may enter; other
    chains run on their own fresh transcripts. Selection mode is an episode-
    driven input fact (the authoritative operation requires a selection), not a
    kind fact: in that shape the draft omits ``relational_willingness`` because
    the stance was already determined earlier and travels by deterministic code.

    Args:
        goal_kind: Goal kind label from the closed V2 ``GOAL_KINDS`` set.
        state_projection: Prompt-safe canonical state projection for this chain.
        evidence_handles: Authorized evidence handles for this call.
        selection_mode: True when this chain's output shape is a selection draft.

    Returns:
        The joined deterministic tail text.

    Raises:
        ValueError: when ``goal_kind`` is outside the closed kind set.
    """
    goal_kind_owns_relational_willingness(goal_kind)
    lines = [
        "# 目标类型",
        goal_kind,
        "# 输出形态",
        "selection draft" if selection_mode else "goal bid",
    ]

    # Only the ordinary kind owns the relational stance in its goal-bid shape;
    # state that fact once and let selection mode travel the code-carried copy.
    if not selection_mode and goal_kind == ORDINARY_GOAL_KIND:
        lines.append("本分支拥有 relational_willingness，必须输出完整关系立场。")

    handle_lines = [f"- {handle}" for handle in evidence_handles]
    if handle_lines:
        lines.extend(["# 授权证据 handle", *handle_lines])

    state_lines = [
        f"{key}={value}" for key, value in dict(state_projection).items()
    ]
    if state_lines:
        lines.extend(["# 状态投影", *state_lines])

    return "\n".join(lines)


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
