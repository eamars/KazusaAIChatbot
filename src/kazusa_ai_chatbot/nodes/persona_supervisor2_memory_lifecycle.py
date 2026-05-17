"""Memory lifecycle specialist for active-commitment route intents."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.action_spec.models import (
    ActionSourceRefV1,
    ActionSpecV1,
    LIFECYCLE_STATUS_BY_DECISION,
    validate_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
)
from kazusa_ai_chatbot.config import (
    COGNITION_LLM_API_KEY,
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MODEL,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.utils import get_llm, log_preview, parse_llm_json_output

logger = logging.getLogger(__name__)

ACTIVE_COMMITMENT_ALIAS_LIMIT = 12
LIFECYCLE_UPDATE_LIMIT = 3
_ALLOWED_LIFECYCLE_DECISIONS = frozenset(LIFECYCLE_STATUS_BY_DECISION)
_ALLOWED_CONTENT_ANCHOR_ROLES = frozenset(
    ("avoid_reopening", "acknowledge_fulfillment", "keep_waiting")
)


_MEMORY_LIFECYCLE_SPECIALIST_PROMPT = """\
你是活动承诺生命周期复核专员。
你只判断本轮输入是否已经改变某个活动承诺的生命周期状态。
你不会看到数据库 ID；只能通过 `target_alias` 选择输入里给出的 `commitment_1`、`commitment_2` 等别名。

# 生成步骤
1. 先读取 `current_input`、`formed_decision` 和活动承诺列表，确认当前回合是否直接兑现、取消、改写、推迟或使某个承诺过时。
2. 只在证据清楚时输出生命周期变化。含糊玩笑、普通寒暄、新计划、还没兑现、只是在继续等待，都不要关闭承诺。
3. 如果需要改变多个承诺，按证据从最清楚到最弱排序。
4. 只使用输入中存在的 `target_alias`。不要发明别名，不要输出数据库字段。
5. 为下游文字锚点提供简短角色语义锚点，帮助避免把已完成承诺重新说成未完成。

# 输入格式
用户消息是 JSON，包含：
- `current_input`: 当前输入语义摘要。
- `formed_decision`: 已形成的立场、意图、裁决和内心判断。
- `active_commitments`: 最多 12 个活动承诺，每项只有 `target_alias`、`fact`、`status`、`due_at`、`due_state`、`evidence_summary`。
- `memory_evidence`: 当前检索到的提示性记忆证据。
- `conversation_progress`: 当前短期对话进度。

# 输出格式
只返回合法 JSON 字符串：
{
  "decision": "lifecycle_change | no_lifecycle_change",
  "lifecycle_decisions": [
    {
      "target_alias": "commitment_1",
      "decision": "fulfilled | abandoned | obsolete | deferred",
      "role": "给下游看的简短语义角色",
      "evidence_anchor": "支持这个判断的短证据"
    }
  ],
  "content_anchor_roles": [
    {
      "role": "avoid_reopening | acknowledge_fulfillment | keep_waiting",
      "anchor": "给内容锚点生成器看的短语义锚点"
    }
  ]
}

如果没有生命周期变化，返回：
{"decision": "no_lifecycle_change", "lifecycle_decisions": [], "content_anchor_roles": []}
"""
_memory_lifecycle_specialist_llm = get_llm(
    temperature=0.1,
    top_p=0.7,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)


async def call_memory_lifecycle_update_handler(
    state: GlobalPersonaState,
) -> dict:
    """Run lifecycle specialist routing and materialize executable updates.

    Args:
        state: Persona graph state after L2d action selection.

    Returns:
        Partial state update containing prompt-safe lifecycle context and any
        executable lifecycle action specs derived from trusted aliases.
    """

    route_specs = _memory_lifecycle_route_specs(state)
    if not route_specs:
        return_value: dict[str, object] = {}
        return return_value

    prepared = prepare_memory_lifecycle_review(state)
    alias_bindings = prepared["alias_bindings"]
    if not alias_bindings:
        logger.warning("Memory lifecycle route skipped without active commitments")
        context = _memory_lifecycle_context(
            decision="skipped",
            visible_alias_count=0,
            omitted_alias_count=0,
            lifecycle_decisions=[],
            content_anchor_roles=[],
            warnings=["没有可复核的活动承诺。"],
        )
        return_value = {
            "action_specs": _selected_non_lifecycle_action_specs(state),
            "memory_lifecycle_context": context,
        }
        return return_value

    prompt_payload = prepared["prompt_payload"]
    system_prompt = SystemMessage(content=_MEMORY_LIFECYCLE_SPECIALIST_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(prompt_payload, ensure_ascii=False)
    )
    response = await _memory_lifecycle_specialist_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    parsed = parse_llm_json_output(response.content)
    normalized = normalize_memory_lifecycle_output(parsed, alias_bindings)
    materialized = materialize_memory_lifecycle_actions(
        normalized,
        alias_bindings,
        visible_alias_count=int(prepared["visible_alias_count"]),
        omitted_alias_count=int(prepared["omitted_alias_count"]),
    )
    executable_specs = materialized["action_specs"]
    combined_specs = (
        _selected_non_lifecycle_action_specs(state)
        + executable_specs
    )
    logger.debug(
        f"Memory lifecycle specialist: decision={normalized['decision']} "
        f"updates={len(executable_specs)} "
        f"warnings={log_preview(normalized['warnings'])}"
    )
    return_value = {
        "action_specs": combined_specs,
        "memory_lifecycle_context": materialized["memory_lifecycle_context"],
    }
    return return_value


def prepare_memory_lifecycle_review(
    state: GlobalPersonaState,
) -> dict[str, object]:
    """Prepare prompt-safe specialist input and trusted alias bindings.

    Args:
        state: Persona graph state after L2d action selection.

    Returns:
        A deterministic review packet with prompt payload, alias bindings, and
        alias-cap accounting. The payload never contains persistence IDs.
    """

    alias_bindings = _active_commitment_alias_rows(state)
    visible_alias_count = len(alias_bindings)
    prepared = {
        "prompt_payload": _specialist_prompt_payload(state, alias_bindings),
        "alias_bindings": alias_bindings,
        "visible_alias_count": visible_alias_count,
        "omitted_alias_count": _omitted_alias_count(state, visible_alias_count),
    }
    return prepared


def normalize_memory_lifecycle_output(
    parsed: object,
    alias_bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Normalize specialist JSON against trusted aliases."""

    normalized = _normalize_specialist_output(parsed, alias_bindings)
    normalized["errors"] = []
    return normalized


def materialize_memory_lifecycle_actions(
    normalized: Mapping[str, object],
    alias_bindings: list[dict[str, Any]],
    *,
    visible_alias_count: int,
    omitted_alias_count: int,
) -> dict[str, object]:
    """Resolve normalized alias decisions into executable lifecycle actions."""

    raw_lifecycle_decisions = normalized.get("lifecycle_decisions")
    lifecycle_decisions = (
        raw_lifecycle_decisions
        if isinstance(raw_lifecycle_decisions, list)
        else []
    )
    typed_decisions = [
        decision
        for decision in lifecycle_decisions
        if isinstance(decision, dict)
    ]
    action_specs = _materialize_lifecycle_updates(
        typed_decisions,
        alias_bindings,
    )
    raw_content_anchor_roles = normalized.get("content_anchor_roles")
    content_anchor_roles = (
        raw_content_anchor_roles
        if isinstance(raw_content_anchor_roles, list)
        else []
    )
    typed_roles = [
        role
        for role in content_anchor_roles
        if isinstance(role, dict)
    ]
    raw_warnings = normalized.get("warnings")
    warnings = raw_warnings if isinstance(raw_warnings, list) else []
    typed_warnings = [
        warning
        for warning in warnings
        if isinstance(warning, str)
    ]
    decision = normalized.get("decision")
    if not isinstance(decision, str):
        decision = "skipped"
    context = _memory_lifecycle_context(
        decision=decision,
        visible_alias_count=visible_alias_count,
        omitted_alias_count=omitted_alias_count,
        lifecycle_decisions=typed_decisions,
        content_anchor_roles=typed_roles,
        warnings=typed_warnings,
    )
    materialized = {
        "action_specs": action_specs,
        "memory_lifecycle_context": context,
        "errors": list(normalized.get("errors", []))
        if isinstance(normalized.get("errors"), list)
        else [],
    }
    return materialized


def _memory_lifecycle_route_specs(
    state: GlobalPersonaState,
) -> list[dict[str, Any]]:
    """Return selected route-intent specs for memory lifecycle review."""

    route_specs: list[dict[str, Any]] = []
    for action_spec in _selected_action_specs(state):
        if action_spec.get("kind") == MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
            route_specs.append(action_spec)
    return route_specs


def _selected_action_specs(state: GlobalPersonaState) -> list[dict[str, Any]]:
    """Return selected action specs from state as dictionaries."""

    raw_specs = state.get("action_specs")
    if not isinstance(raw_specs, list):
        return_value: list[dict[str, Any]] = []
        return return_value
    specs = [
        spec
        for spec in raw_specs
        if isinstance(spec, dict)
    ]
    return specs


def _selected_non_lifecycle_action_specs(
    state: GlobalPersonaState,
) -> list[dict[str, Any]]:
    """Return selected specs after consuming lifecycle route intents."""

    specs = [
        spec
        for spec in _selected_action_specs(state)
        if spec.get("kind") != MEMORY_LIFECYCLE_UPDATE_CAPABILITY
    ]
    return specs


def _active_commitment_alias_rows(
    state: GlobalPersonaState,
) -> list[dict[str, Any]]:
    """Build prompt-safe aliases backed by trusted active commitment IDs."""

    active_units = _trusted_active_commitment_units(state)
    sorted_units = sorted(
        active_units,
        key=lambda unit: _optional_text(unit, "due_at"),
    )
    visible_units = sorted_units[:ACTIVE_COMMITMENT_ALIAS_LIMIT]
    alias_rows: list[dict[str, Any]] = []
    for index, unit in enumerate(visible_units, start=1):
        alias = f"commitment_{index}"
        evidence_summary = _commitment_evidence_summary(unit)
        prompt_row = {
            "target_alias": alias,
            "fact": _optional_text(unit, "fact"),
            "status": _optional_text(unit, "status") or "active",
            "due_at": _nullable_text(unit, "due_at"),
            "due_state": _nullable_text(unit, "due_state"),
            "evidence_summary": evidence_summary,
        }
        alias_rows.append({
            "target_alias": alias,
            "unit_id": _optional_text(unit, "unit_id"),
            "due_at": _nullable_text(unit, "due_at"),
            "prompt_row": prompt_row,
        })
    return alias_rows


def _trusted_active_commitment_units(
    state: GlobalPersonaState,
) -> list[dict[str, Any]]:
    """Return active commitment rows that retain trusted unit IDs."""

    rag_result = state["rag_result"]
    if not isinstance(rag_result, dict):
        return_value: list[dict[str, Any]] = []
        return return_value

    raw_units: list[object] = []
    prompt_context_units: list[object] = []
    raw_candidates = rag_result.get("user_memory_unit_candidates")
    if isinstance(raw_candidates, list):
        raw_units.extend(raw_candidates)

    direct_user_units = rag_result.get("user_memory_units")
    if isinstance(direct_user_units, list):
        raw_units.extend(direct_user_units)

    user_image = rag_result.get("user_image")
    if isinstance(user_image, dict):
        raw_user_units = user_image.get("_user_memory_units")
        if isinstance(raw_user_units, list):
            raw_units.extend(raw_user_units)
        memory_context = user_image.get("user_memory_context")
        if isinstance(memory_context, dict):
            raw_active_commitments = memory_context.get(
                "active_commitments"
            )
            if isinstance(raw_active_commitments, list):
                prompt_context_units.extend(raw_active_commitments)

    active_units = _filter_active_commitment_units(raw_units)
    if active_units:
        return active_units
    return _filter_active_commitment_units(prompt_context_units)


def _filter_active_commitment_units(
    raw_units: list[object],
) -> list[dict[str, Any]]:
    """Filter active commitment rows with trusted unit ids."""

    active_units: list[dict[str, Any]] = []
    seen_unit_ids: set[str] = set()
    for raw_unit in raw_units:
        if not isinstance(raw_unit, dict):
            continue
        unit_id = _optional_text(raw_unit, "unit_id")
        if not unit_id:
            continue
        if unit_id in seen_unit_ids:
            continue
        unit_type = _optional_text(raw_unit, "unit_type") or "active_commitment"
        if unit_type != "active_commitment":
            continue
        status = _optional_text(raw_unit, "status") or "active"
        if status != "active":
            continue
        seen_unit_ids.add(unit_id)
        active_units.append(raw_unit)
    return active_units


def _commitment_evidence_summary(unit: Mapping[str, object]) -> str:
    """Build a prompt-safe semantic summary for one commitment."""

    summary_parts = []
    for field_name in ("subjective_appraisal", "relationship_signal", "updated_at"):
        field_value = _optional_text(unit, field_name)
        if field_value:
            summary_parts.append(field_value)
    evidence_summary = "；".join(summary_parts)
    return evidence_summary


def _specialist_prompt_payload(
    state: GlobalPersonaState,
    alias_rows: list[dict[str, Any]],
) -> dict[str, object]:
    """Build the prompt-safe JSON payload for the specialist."""

    formed_decision = {
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "judgment_note": state["judgment_note"],
        "internal_monologue": state["internal_monologue"],
    }
    active_commitments = [
        alias_row["prompt_row"]
        for alias_row in alias_rows
    ]
    prompt_payload = {
        "current_input": state["decontexualized_input"],
        "formed_decision": formed_decision,
        "active_commitments": active_commitments,
        "memory_evidence": _project_memory_evidence(state),
        "conversation_progress": state.get("conversation_progress"),
    }
    return prompt_payload


def _project_memory_evidence(state: GlobalPersonaState) -> list[dict[str, object]]:
    """Project memory evidence without storage identifiers or source names."""

    rag_result = state["rag_result"]
    if not isinstance(rag_result, dict):
        return_value: list[dict[str, object]] = []
        return return_value

    raw_evidence = rag_result.get("memory_evidence")
    if not isinstance(raw_evidence, list):
        return_value = []
        return return_value

    projected_evidence: list[dict[str, object]] = []
    for raw_entry in raw_evidence[:12]:
        if not isinstance(raw_entry, dict):
            continue
        projected_entry: dict[str, object] = {}
        for field_name in ("summary", "fact", "excerpt", "content", "due_at"):
            field_value = _optional_text(raw_entry, field_name)
            if field_value:
                projected_entry[field_name] = field_value
        due_state = _optional_text(raw_entry, "due_state")
        if due_state:
            projected_entry["due_state"] = due_state
        if projected_entry:
            projected_evidence.append(projected_entry)
    return projected_evidence


def _normalize_specialist_output(
    parsed: object,
    alias_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Normalize specialist JSON without changing semantic decisions."""

    warnings: list[str] = []
    if not isinstance(parsed, dict):
        normalized = {
            "decision": "skipped",
            "lifecycle_decisions": [],
            "content_anchor_roles": [],
            "warnings": ["生命周期专员输出不是对象。"],
        }
        return normalized

    decision = _optional_text(parsed, "decision")
    if decision == "no_lifecycle_change":
        lifecycle_decisions: list[dict[str, str]] = []
    elif decision == "lifecycle_change":
        raw_decisions = parsed.get("lifecycle_decisions")
        lifecycle_decisions = _valid_lifecycle_decisions(
            raw_decisions,
            alias_rows,
            warnings,
        )
    else:
        decision = "skipped"
        lifecycle_decisions = []
        warnings.append("生命周期专员输出了不支持的总决定。")

    content_anchor_roles = _valid_content_anchor_roles(
        parsed.get("content_anchor_roles"),
        warnings,
    )
    normalized = {
        "decision": decision,
        "lifecycle_decisions": lifecycle_decisions,
        "content_anchor_roles": content_anchor_roles,
        "warnings": warnings,
    }
    return normalized


def _valid_lifecycle_decisions(
    raw_decisions: object,
    alias_rows: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, str]]:
    """Keep valid alias-backed lifecycle decisions up to the update cap."""

    if not isinstance(raw_decisions, list):
        warnings.append("生命周期变化缺少有效列表。")
        return_value: list[dict[str, str]] = []
        return return_value

    known_aliases = {
        str(alias_row["target_alias"])
        for alias_row in alias_rows
    }
    valid_decisions: list[dict[str, str]] = []
    for raw_decision in raw_decisions:
        if not isinstance(raw_decision, dict):
            warnings.append("已忽略非对象生命周期决定。")
            continue
        target_alias = _optional_text(raw_decision, "target_alias")
        lifecycle_decision = _optional_text(raw_decision, "decision")
        role = _optional_text(raw_decision, "role")
        evidence_anchor = _optional_text(raw_decision, "evidence_anchor")
        if target_alias not in known_aliases:
            warnings.append(f"已忽略未知承诺别名：{target_alias or '空'}。")
            continue
        if lifecycle_decision not in _ALLOWED_LIFECYCLE_DECISIONS:
            warnings.append(f"已忽略不支持的生命周期决定：{target_alias}。")
            continue
        if not role or not evidence_anchor:
            warnings.append(f"已忽略缺少角色或证据锚点的决定：{target_alias}。")
            continue
        valid_decisions.append({
            "target_alias": target_alias,
            "decision": lifecycle_decision,
            "role": role,
            "evidence_anchor": evidence_anchor,
        })
    if len(valid_decisions) > LIFECYCLE_UPDATE_LIMIT:
        overflow_count = len(valid_decisions) - LIFECYCLE_UPDATE_LIMIT
        warnings.append(f"已保留前三个生命周期决定，另有 {overflow_count} 个留待后续。")
        logger.warning(
            f"Memory lifecycle update cap dropped {overflow_count} decisions"
        )
    capped_decisions = valid_decisions[:LIFECYCLE_UPDATE_LIMIT]
    return capped_decisions


def _valid_content_anchor_roles(
    raw_roles: object,
    warnings: list[str],
) -> list[dict[str, str]]:
    """Keep prompt-safe content-anchor roles from specialist output."""

    if raw_roles is None:
        return_value: list[dict[str, str]] = []
        return return_value
    if not isinstance(raw_roles, list):
        warnings.append("内容锚点角色不是有效列表。")
        return_value = []
        return return_value

    valid_roles: list[dict[str, str]] = []
    for raw_role in raw_roles:
        if not isinstance(raw_role, dict):
            warnings.append("已忽略非对象内容锚点角色。")
            continue
        role = _optional_text(raw_role, "role")
        anchor = _optional_text(raw_role, "anchor")
        if role not in _ALLOWED_CONTENT_ANCHOR_ROLES:
            warnings.append(f"已忽略不支持的内容锚点角色：{role or '空'}。")
            continue
        if not anchor:
            warnings.append(f"已忽略缺少锚点文本的内容角色：{role}。")
            continue
        valid_roles.append({
            "role": role,
            "anchor": anchor,
        })
    return valid_roles


def _materialize_lifecycle_updates(
    lifecycle_decisions: list[dict[str, str]],
    alias_rows: list[dict[str, Any]],
) -> list[ActionSpecV1]:
    """Resolve valid aliases to trusted unit IDs and action specs."""

    alias_by_name = {
        str(alias_row["target_alias"]): alias_row
        for alias_row in alias_rows
    }
    action_specs: list[ActionSpecV1] = []
    for lifecycle_decision in lifecycle_decisions:
        target_alias = lifecycle_decision["target_alias"]
        alias_row = alias_by_name[target_alias]
        unit_id = str(alias_row["unit_id"])
        action_spec = _build_apply_lifecycle_action_spec(
            unit_id=unit_id,
            lifecycle_decision=lifecycle_decision["decision"],
            due_at=alias_row["due_at"],
            reason=lifecycle_decision["evidence_anchor"],
        )
        validated_spec = validate_action_spec(action_spec)
        action_specs.append(validated_spec)
    return action_specs


def _build_apply_lifecycle_action_spec(
    *,
    unit_id: str,
    lifecycle_decision: str,
    due_at: str | None,
    reason: str,
) -> dict[str, object]:
    """Build the executable lifecycle update action spec."""

    action_spec = {
        "schema_version": "action_spec.v1",
        "kind": APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [
            _current_episode_source_ref(),
            _memory_unit_source_ref(unit_id),
        ],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "memory_unit",
            "target_id": unit_id,
            "owner": "user_memory_units",
            "scope": {"unit_type": "active_commitment"},
        },
        "params": {
            "memory_kind": "user_memory_unit",
            "unit_type": "active_commitment",
            "unit_id": unit_id,
            "lifecycle_decision": lifecycle_decision,
            "due_at": due_at,
        },
        "urgency": "background",
        "visibility": "private",
        "deadline": None,
        "continuation": _no_continuation(),
        "reason": reason,
    }
    return action_spec


def _current_episode_source_ref() -> ActionSourceRefV1:
    """Return a source reference for the current cognitive episode."""

    source_ref: ActionSourceRefV1 = {
        "schema_version": "action_source_ref.v1",
        "ref_kind": "cognitive_episode",
        "ref_id": "current_cognitive_episode",
        "owner": "cognition_episode",
        "relationship": "basis",
        "evidence_refs": [],
    }
    return source_ref


def _memory_unit_source_ref(unit_id: str) -> ActionSourceRefV1:
    """Return a trusted source reference for the resolved memory unit."""

    source_ref: ActionSourceRefV1 = {
        "schema_version": "action_source_ref.v1",
        "ref_kind": "memory_unit",
        "ref_id": unit_id,
        "owner": "user_memory_units",
        "relationship": "target",
        "evidence_refs": [],
    }
    return source_ref


def _no_continuation() -> dict[str, object]:
    """Return the default no-continuation execution contract."""

    continuation = {
        "schema_version": "action_continuation.v1",
        "mode": "none",
        "episode_type": None,
        "max_depth": 0,
        "include_result_as": None,
    }
    return continuation


def _memory_lifecycle_context(
    *,
    decision: str,
    visible_alias_count: int,
    omitted_alias_count: int,
    lifecycle_decisions: list[dict[str, str]],
    content_anchor_roles: list[dict[str, str]],
    warnings: list[str],
) -> dict[str, object]:
    """Build the prompt-safe lifecycle context for downstream L3 stages."""

    context = {
        "schema_version": "memory_lifecycle_context.v1",
        "source": "memory_lifecycle_specialist",
        "decision": decision,
        "visible_alias_count": visible_alias_count,
        "omitted_alias_count": omitted_alias_count,
        "lifecycle_decisions": lifecycle_decisions,
        "content_anchor_roles": content_anchor_roles,
        "warnings": warnings,
    }
    return context


def _omitted_alias_count(
    state: GlobalPersonaState,
    visible_alias_count: int,
) -> int:
    """Return how many trusted active commitments were omitted from the prompt."""

    active_count = len(_trusted_active_commitment_units(state))
    omitted_count = max(0, active_count - visible_alias_count)
    return omitted_count


def _optional_text(value: Mapping[str, object], field_name: str) -> str:
    """Return one stripped optional text field."""

    raw_value = value.get(field_name)
    if not isinstance(raw_value, str):
        return_value = ""
        return return_value
    return_value = raw_value.strip()
    return return_value


def _nullable_text(value: Mapping[str, object], field_name: str) -> str | None:
    """Return one stripped text field or None."""

    text = _optional_text(value, field_name)
    if not text:
        return_value = None
        return return_value
    return text
