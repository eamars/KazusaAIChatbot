"""Coarse consolidation lane routing and auditable lane pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.runner import (
    evaluate_episode_identity_growth,
)
from kazusa_ai_chatbot.config import (
    CHARACTER_IDENTITY_GROWTH_ENABLED,
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    CONSOLIDATION_LLM_MODEL,
    CONSOLIDATION_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.db.character_identity_growth import (
    get_current_identity,
)
from kazusa_ai_chatbot.consolidation.persistence import db_writer
from kazusa_ai_chatbot.consolidation.character_self_guidance import (
    character_self_guidance_specialist,
)
from kazusa_ai_chatbot.consolidation.metadata import (
    finalize_consolidation_metadata,
)
from kazusa_ai_chatbot.consolidation.source_policy import (
    ASSISTANT_ACCEPTANCE_SOURCE_KIND,
    build_consolidation_source_views,
    source_refs_from_views,
    validate_character_operational_sources,
    validate_lane_source_policy,
)
from kazusa_ai_chatbot.consolidation.target import (
    CHARACTER_TARGET_ALIAS,
    GROUP_CHANNEL_TARGET_ALIAS,
    INTERNAL_TARGET_ALIAS,
    USER_TARGET_ALIAS,
    ConsolidationTargetPlan,
    ConsolidationTargetValidationError,
    validate_write_intent,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.rag.prompt_projection import project_tool_result_for_llm
from kazusa_ai_chatbot.utils import parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

CONSOLIDATION_LANE_NAMES = (
    "user_memory_units",
    "active_commitment",
    "character_identity_growth",
    "character_self_guidance",
    "interaction_style_image",
    "shared_memory_promotion",
)

_ROUTER_TASK_KEYS = frozenset(("lane", "reason", "source_keys"))
_IDENTITY_ROUTER_TASK_KEYS = frozenset((
    "lane",
    "reason",
    "source_keys",
    "identity_evidence",
))
_IDENTITY_EVIDENCE_KEYS = frozenset((
    "decontextualized_event",
    "character_cognition_summary",
    "visible_self_expression_summary",
))
_FORBIDDEN_ROUTER_TASK_KEYS = frozenset(
    ("target_id", "write_lane", "payload", "fact")
)
_MAX_ROUTER_TASKS = 4
_CHARACTER_OPERATIONAL_TASK_KEY = "character_operational_state_task"
_CHARACTER_OPERATIONAL_TASK_KEYS = frozenset(("reason", "source_keys"))
_CHARACTER_OPERATIONAL_REASON_LIMIT = 160
_CHARACTER_OPERATIONAL_SOURCE_LIMIT = 4

_LANE_DESCRIPTIONS = {
    "user_memory_units": (
        "仅保存当前真实用户的持久事实、偏好、模式、变化或里程碑，且最终未形成角色未来行为。"
    ),
    "active_commitment": (
        "仅保存角色已接受且明确面向当前用户的个体未来行为。"
    ),
    "character_identity_growth": (
        "仅保存角色自身认同、自我概念或边界的持久变化。"
    ),
    "character_self_guidance": (
        "仅保存角色已接受、与具体对象和共享场景无关且普遍适用的未来行为。"
    ),
    "interaction_style_image": (
        "承接 source_role 已声明的 user_style_signal 或 group_channel_style_image，"
        "以及明确绑定具体群组、频道或公共场景的互动规范。"
    ),
    "shared_memory_promotion": "只把已经提升的反思证据接纳进共享记忆。",
}

_ROUTER_PROMPT = '''\
你负责把一个已经完成的片段路由到粗粒度的持久化整理通道任务项。

HumanMessage 中包含：
- target_plan：确定性代码给出的合格持久化目标；
- lane_roster：本片段唯一可选的通道名称；
- source_views：可安全用于提示词、并带有 source_key 的证据行。

从 lane_roster 中选择零到四项通道任务项。一项任务项表示本片段存在值得由对应专项处理器
检查的持久更新。另独立判断 character_operational_state_task：它不计入四项通道任务项，且仅表示
本片段是否需要一次无来源依赖的角色短期运行态评估。只返回通道名称、简短 reason 和
来自 source_views 的 source_key。持久化细节、记忆正文、目标标识、时间戳与缓存行为由后续确定性阶段负责。

# 运行态槽位
运行态槽位独立于持久化整理通道。它判断已完成的片段是否留下下一轮可能需要
使用的、无来源依赖的角色层面姿态。持久化通道和运行态任务项可以为同一片段
同时被选中。

当已接受的片段包含能够脱离当前场景延续的角色层面后果时，返回非空的运行态任务项，
例如：
- 故意伤害、羞辱、胁迫、拒绝或边界侵犯；
- 可能留下残余压力的威胁、损失、暴露或其他事件；
- 改变角色姿态的道歉、修复或已接受的变化；
- 结果可能影响角色下一轮回应的已接受任务结果。

对于这些情况，从给定的 source_views 中选择一到四个 source_keys。
优先同时选择 current_turn_user_message 和 assistant_final_dialog；只有在
internal_thought 或 episode_trace 存在且确实支持同一个无来源依赖后果时才加入。
运行态任务项与持久化通道任务项并存，永远不会替代或压制它们。

仅当片段明显属于普通、信息性或短暂内容，且没有留下超出当前场景的角色层面后果，
或不存在已接受的 assistant 对话时，才返回 null。仅仅选中了持久化通道，不能成为
对边界、修复或用户事实返回 null 的理由。

运行态对象必须且只能包含以下字段：
{"reason": "简短且有边界的理由", "source_keys": ["source_key"]}
顶层响应必须且只能包含 lane_tasks 和 character_operational_state_task。
始终保留该字段，并将其设为 null 或上述确切对象结构；不要把运行态字段放入持久化通道行。

# 输入与输出边界
target_plan 仅提供确定性资格和权限背景，不是路由词表；target_plan.write_lanes 绝不能复制到
lane_tasks[].lane。lane_tasks[].lane 只能逐字使用 lane_roster[].lane 中给出的值；不在该列表中的值一律不输出。

# 互斥归属层级
先判断已接受的未来角色行为，并把角色视为该行为的执行者；支持它的用户描述只作证据，不夺取归属。
命题提出者和所在频道不能代替命题明确写出的对象、受益者或适用范围。
1. 来源归属优先：若 source_views.source_role 已给出结构化角色，且 lane_roster 提供对应通道，保留该来源角色。
   user_style_signal 或 group_channel_style_image 应路由 interaction_style_image，不得改写为 user_memory_units；
   其他 source_role 仍按实际含义判断。
2. 已接受的角色未来行为按明确适用范围互斥路由：明确绑定当前用户的个体行为归 active_commitment；
   明确绑定某一具体群组、频道或公共场景的共享行为归 interaction_style_image；面向一般对象、
   没有具体共享场景约束且移除具体对象后仍适用的行为归 character_self_guidance。一般受众措辞本身
   不等于某一具体共享场景；提出者和频道本身不改变适用范围。
3. user_memory_units 只保存当前真实用户的持久事实、偏好、模式、变化或里程碑，且最终没有形成已接受的角色未来行为。
   角色仅确认、记住、尊重或配合用户描述时，该配合只能作为用户记忆的支持；若已形成角色未来行为，
   按行为的适用范围选择，不能改写为 user_memory_units。
4. character_identity_growth 只保存角色自身认同、自我概念或边界的持久变化；亲密关系经历也可能促成角色自己的持久变化，
   关系对象、关系事实与私密细节仍归原有作用域，只有角色自己的抽象变化进入此 lane。其 identity_evidence 仍必须
   只包含 decontextualized_event、character_cognition_summary 和 visible_self_expression_summary。
5. 若 lane_roster 中没有通道拥有该持久含义，返回空 lane_tasks。路由保持粗粒度，候选细节由所选专项处理器判断。

对已经接受的未来行为规则，如果请求来源和最终对话接受来源的 source_key 都可用，则同时引用。

# 跳过条件
一轮角色扮演或临时行为、聊天中的普通世界知识、与当前用户无关的第三方事实，以及最终对话尚未
接受或只在当前情境成立的未来行为，均返回空 lane_tasks。

# 输出格式
{
  "lane_tasks": [],
  "character_operational_state_task": {
  "reason": "故意的边界侵犯可能留下残余的角色姿态。",
    "source_keys": ["current_turn_user_message", "assistant_final_dialog"]
  }
}

character_identity_growth 通道行的 identity_evidence 必须是一个且只包含以下字段的对象：
decontextualized_event、character_cognition_summary 和 visible_self_expression_summary。
其他通道都省略 identity_evidence。
'''

_lane_router_llm = LLInterface()
_lane_router_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CONSOLIDATION_LLM",
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=0.0,
    top_p=1.0,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)


def build_lane_roster(
    target_plan: ConsolidationTargetPlan,
) -> list[dict[str, str]]:
    """Build router-visible lane roster from deterministic write lanes.

    Args:
        target_plan: Deterministic target plan attached before routing.

    Returns:
        Roster rows containing only currently possible lane names and
        descriptions.
    """

    write_lanes = set()
    target_kinds = set()
    for target in target_plan["targets"]:
        write_lanes.update(target["write_lanes"])
        target_kinds.add(target["target_kind"])

    reflection_origin = target_plan["origin_kind"].startswith("reflection")
    roster: list[dict[str, str]] = []
    if reflection_origin:
        if (
            "user_style_image" in write_lanes
            or "group_channel_style_image" in write_lanes
        ):
            roster.append(_roster_entry("interaction_style_image"))
        roster.append(_roster_entry("shared_memory_promotion"))
        return roster

    if "user_memory_units" in write_lanes:
        roster.append(_roster_entry("user_memory_units"))
        roster.append(_roster_entry("active_commitment"))
    if (
        CHARACTER_IDENTITY_GROWTH_ENABLED
        and "character_identity_growth" in write_lanes
    ):
        roster.append(_roster_entry("character_identity_growth"))
    if "character_self_guidance" in write_lanes:
        roster.append(_roster_entry("character_self_guidance"))
    if "group_channel_style_image" in write_lanes:
        roster.append(_roster_entry("interaction_style_image"))
    if "internal" in target_kinds:
        roster.append(_roster_entry("shared_memory_promotion"))

    return roster


def validate_lane_router_output(
    output: Mapping[str, Any],
    roster: list[dict[str, str]],
) -> dict[str, Any]:
    """Validate the four durable lanes and independent operational slot.

    Args:
        output: Parsed router JSON.
        roster: Lane roster built from the target plan.

    Returns:
        Exact route decision with validated durable and operational tasks.

    Raises:
        ValueError: If the output contains unknown lanes, non-roster lanes,
            persistence fields, memory payload fields, or malformed task rows.
    """

    if set(output) != {
        "lane_tasks",
        _CHARACTER_OPERATIONAL_TASK_KEY,
    }:
        raise ValueError("router output does not match the route decision contract")
    validated_tasks = _validate_durable_lane_tasks(
        output["lane_tasks"],
        roster,
    )
    operational_task = _validate_character_operational_task(
        output[_CHARACTER_OPERATIONAL_TASK_KEY],
    )
    return {
        "lane_tasks": validated_tasks,
        _CHARACTER_OPERATIONAL_TASK_KEY: operational_task,
    }


def _validate_route_decision_for_pipeline(
    output: object,
    roster: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    """Validate durable and operational routing without cross-slot loss.

    The top-level decision is all-or-nothing.  Once its exact shape is
    established, the operational slot has an independent bounded failure
    boundary: a malformed slot cannot suppress already valid durable work.
    """

    if not isinstance(output, Mapping):
        return [], None, "route_invalid"
    if set(output) != {"lane_tasks", _CHARACTER_OPERATIONAL_TASK_KEY}:
        return [], None, "route_invalid"
    try:
        lane_tasks = _validate_durable_lane_tasks(
            output["lane_tasks"],
            roster,
        )
    except (TypeError, ValueError):
        return [], None, "route_invalid"
    try:
        operational_task = _validate_character_operational_task(
            output[_CHARACTER_OPERATIONAL_TASK_KEY],
        )
    except (TypeError, ValueError):
        return lane_tasks, None, "route_invalid"
    return lane_tasks, operational_task, ""


def _validate_durable_lane_tasks(
    lane_tasks: object,
    roster: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Validate the existing bounded durable lane list independently."""

    if not isinstance(lane_tasks, list):
        raise ValueError("lane_tasks must be a list")
    if len(lane_tasks) > _MAX_ROUTER_TASKS:
        raise ValueError("lane_tasks exceeds the task limit")

    roster_lanes = {
        text_or_empty(entry.get("lane"))
        for entry in roster
        if isinstance(entry, Mapping)
    }
    validated_tasks: list[dict[str, Any]] = []
    seen_lanes: set[str] = set()
    for raw_task in lane_tasks:
        if not isinstance(raw_task, Mapping):
            raise ValueError("lane task must be an object")
        task_keys = set(raw_task)
        if task_keys & _FORBIDDEN_ROUTER_TASK_KEYS:
            raise ValueError("router task contains persistence or memory fields")

        lane = text_or_empty(raw_task.get("lane"))
        if lane not in CONSOLIDATION_LANE_NAMES:
            raise ValueError(f"unknown consolidation lane: {lane!r}")
        if lane not in roster_lanes:
            raise ValueError(f"lane is not in target roster: {lane!r}")
        expected_keys = (
            _IDENTITY_ROUTER_TASK_KEYS
            if lane == "character_identity_growth"
            else _ROUTER_TASK_KEYS
        )
        if task_keys != expected_keys:
            raise ValueError(
                "router task does not match its closed lane contract"
            )
        if lane in seen_lanes:
            raise ValueError(f"duplicate consolidation lane: {lane!r}")
        seen_lanes.add(lane)

        reason = text_or_empty(raw_task.get("reason"))
        raw_source_keys = raw_task.get("source_keys")
        if not isinstance(raw_source_keys, list):
            raise ValueError("source_keys must be a list")
        source_keys = [
            source_key.strip()
            for source_key in raw_source_keys
            if isinstance(source_key, str) and source_key.strip()
        ]
        validated_task = {
            "lane": lane,
            "reason": reason,
            "source_keys": source_keys,
        }
        if lane == "character_identity_growth":
            validated_task["identity_evidence"] = (
                _validate_router_identity_evidence(
                    raw_task.get("identity_evidence")
                )
            )
        validated_tasks.append(validated_task)

    return validated_tasks


def _validate_character_operational_task(
    value: object,
) -> dict[str, Any] | None:
    """Validate the independent optional operational router slot."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("character operational task must be an object or null")
    if set(value) != _CHARACTER_OPERATIONAL_TASK_KEYS:
        raise ValueError("character operational task keys are invalid")
    reason = text_or_empty(value.get("reason"))
    if not reason or len(reason) > _CHARACTER_OPERATIONAL_REASON_LIMIT:
        raise ValueError("character operational task reason is invalid")
    raw_source_keys = value.get("source_keys")
    if not isinstance(raw_source_keys, list):
        raise ValueError("character operational task source_keys must be a list")
    source_keys = [text_or_empty(source_key) for source_key in raw_source_keys]
    if (
        not 1 <= len(source_keys) <= _CHARACTER_OPERATIONAL_SOURCE_LIMIT
        or any(not source_key for source_key in source_keys)
        or len(source_keys) != len(set(source_keys))
    ):
        raise ValueError("character operational task source_keys are invalid")
    return {
        "reason": reason,
        "source_keys": source_keys,
    }


def _validate_router_identity_evidence(
    value: object,
) -> dict[str, str]:
    """Validate the router-owned semantic identity evidence card."""

    if not isinstance(value, Mapping):
        raise ValueError("identity_evidence must be an object")
    if set(value) != _IDENTITY_EVIDENCE_KEYS:
        raise ValueError("identity_evidence has an invalid key set")
    event = _bounded_router_text(
        value.get("decontextualized_event"),
        field_name="decontextualized_event",
        required=True,
    )
    cognition = _bounded_router_text(
        value.get("character_cognition_summary"),
        field_name="character_cognition_summary",
        required=False,
    )
    visible = _bounded_router_text(
        value.get("visible_self_expression_summary"),
        field_name="visible_self_expression_summary",
        required=False,
    )
    return {
        "decontextualized_event": event,
        "character_cognition_summary": cognition,
        "visible_self_expression_summary": visible,
    }


def _bounded_router_text(
    value: object,
    *,
    field_name: str,
    required: bool,
) -> str:
    """Require one bounded router semantic field."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    text = value.strip()
    if required and not text:
        raise ValueError(f"{field_name} must be nonempty")
    if len(text) > models.IDENTITY_EVIDENCE_CARD_TEXT_LIMIT:
        raise ValueError(f"{field_name} exceeds its text limit")
    return text


async def call_lane_router_llm(
    state: Mapping[str, Any],
    *,
    source_views: list[dict[str, Any]],
    roster: list[dict[str, str]],
) -> dict[str, Any]:
    """Call the background LLM that chooses coarse consolidation lanes.

    Args:
        state: Consolidator state carrying the target plan and turn metadata.
        source_views: Transient source-view rows built from the current state.
        roster: Deterministically pruned lane roster.

    Returns:
        Parsed JSON object returned by the router LLM.
    """

    target_plan = state["consolidation_target_plan"]
    payload = {
        "target_plan": project_tool_result_for_llm(target_plan),
        "lane_roster": roster,
        "source_views": _router_prompt_source_views(source_views),
    }
    system_prompt = SystemMessage(content=_ROUTER_PROMPT)
    human_message = HumanMessage(content=json.dumps(payload, ensure_ascii=False))
    response = await _lane_router_llm.ainvoke(
        [system_prompt, human_message],
        config=_lane_router_llm_config,
    )
    parsed_output = parse_llm_json_output(response.content)
    return parsed_output


def _router_prompt_source_views(
    source_views: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Remove repository identifiers from router-facing source views."""

    prompt_views: list[dict[str, str]] = []
    for source_view in source_views:
        prompt_view = {
            "source_key": text_or_empty(source_view.get("source_key")),
            "source_kind": text_or_empty(source_view.get("source_kind")),
            "summary": text_or_empty(source_view.get("summary")),
        }
        source_role = text_or_empty(source_view.get("source_role"))
        if source_role:
            prompt_view["source_role"] = source_role
        prompt_views.append(prompt_view)
    return prompt_views


async def run_consolidation_lane_pipeline(
    state: Mapping[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run source-view, lane-router, source-policy, and persistence handling.

    Args:
        state: Consolidator state after target planning.
        dry_run: When true, return write intents without persistence.

    Returns:
        Auditable packet containing mode, source views, router tasks,
        per-lane results, write intents, and the final working state.
    """

    source_views = build_consolidation_source_views(state)
    source_views_by_key = _source_views_by_key(source_views)
    target_plan = state["consolidation_target_plan"]
    roster = build_lane_roster(target_plan)
    try:
        router_output = await call_lane_router_llm(
            state,
            source_views=source_views,
            roster=roster,
        )
    except Exception:
        router_tasks = []
        character_operational_task = None
        router_validation_error = "route_invalid"
    else:
        (
            router_tasks,
            character_operational_task,
            router_validation_error,
        ) = _validate_route_decision_for_pipeline(router_output, roster)
    if router_validation_error:
        logger.warning("lane router output dropped or reduced")

    character_operational_error = router_validation_error
    character_operational_evidence: list[dict[str, Any]] = []
    if character_operational_task is not None:
        if not _operational_slot_is_available(state, source_views):
            character_operational_task = None
            character_operational_error = "route_invalid"
        else:
            try:
                character_operational_evidence = (
                    _validated_character_operational_evidence(
                        task=character_operational_task,
                        source_views_by_key=source_views_by_key,
                        state=state,
                    )
                )
            except ValueError:
                character_operational_task = None
                character_operational_error = "source_policy_rejected"

    lane_results: list[dict[str, Any]] = []
    write_intents: list[dict[str, Any]] = []
    accepted_lanes: list[str] = []
    accepted_user_memory_refs: list[dict[str, Any]] = []
    accepted_self_guidance_refs: list[dict[str, Any]] = []
    identity_routed = any(
        task["lane"] == "character_identity_growth"
        for task in router_tasks
    )
    identity_intent: dict[str, Any] | None = None
    identity_result: dict[str, Any] | None = None

    for task in router_tasks:
        selected_views = _selected_source_views(task, source_views_by_key)
        selected_views = _complete_required_source_views(
            task["lane"],
            selected_views,
            source_views_by_key,
        )
        selected_source_keys = [
            text_or_empty(source_view.get("source_key"))
            for source_view in selected_views
            if text_or_empty(source_view.get("source_key"))
        ]
        source_policy = validate_lane_source_policy(
            task["lane"],
            selected_views,
            privacy_review=_privacy_review_for_state_or_views(
                state,
                selected_views,
            ),
        )
        lane_result = {
            "lane": task["lane"],
            "reason": task["reason"],
            "source_policy": source_policy,
            "source_keys": selected_source_keys,
        }
        if not source_policy["accepted"]:
            lane_result["status"] = "rejected"
            lane_results.append(lane_result)
            continue

        source_refs = source_refs_from_views(selected_views)
        write_intent = _write_intent_for_lane(
            task["lane"],
            target_plan,
            source_refs,
            task=task,
            state=state,
        )
        if (
            task["lane"] == "character_identity_growth"
            and write_intent is None
        ):
            lane_result["status"] = "rejected"
            lane_result["source_policy"] = {
                "accepted": False,
                "reason": "source_refs_missing",
            }
            lane_results.append(lane_result)
            continue
        if write_intent is not None:
            write_intents.append(write_intent)
        if task["lane"] == "character_identity_growth":
            identity_intent = write_intent
        accepted_lanes.append(task["lane"])
        if task["lane"] in {"user_memory_units", "active_commitment"}:
            accepted_user_memory_refs.extend(source_refs)
        if task["lane"] == "character_self_guidance":
            accepted_self_guidance_refs.extend(source_refs)
        lane_result["status"] = "accepted"
        lane_result["write_intent"] = write_intent
        lane_results.append(lane_result)

    working_state = dict(state)
    working_state["enabled_consolidation_write_lanes"] = accepted_lanes
    working_state["user_memory_unit_source_refs"] = accepted_user_memory_refs
    working_state["character_self_guidance_source_refs"] = accepted_self_guidance_refs
    working_state["character_operational_work"] = {
        "status": _character_operational_route_status(
            task=character_operational_task,
            error_code=character_operational_error,
            available=_operational_slot_is_available(state, source_views),
        ),
        "error_code": character_operational_error or None,
        "task": character_operational_task,
        "evidence": character_operational_evidence,
    }
    _ensure_writer_defaults(working_state)

    if not dry_run and accepted_lanes:
        await _run_lane_specialists(working_state, accepted_lanes)
        writer_lanes = [
            lane
            for lane in accepted_lanes
            if lane != "character_identity_growth"
        ]
        working_state["enabled_consolidation_write_lanes"] = writer_lanes
        if writer_lanes:
            writer_result = await db_writer(working_state)
            working_state.update(writer_result)
        else:
            metadata = dict(working_state.get("metadata", {}) or {})
            metadata["write_success"] = {}
            working_state["metadata"] = metadata
        if identity_intent is not None:
            identity_result = await _run_identity_growth_intent(
                identity_intent,
                target_plan=target_plan,
            )
            for lane_result in lane_results:
                if lane_result["lane"] == "character_identity_growth":
                    lane_result["identity_growth_result"] = identity_result
                    break
        elif identity_routed:
            empty_source = _empty_identity_growth_source(state)
            if empty_source is not None:
                identity_result = await _run_identity_growth_source(
                    empty_source,
                    target_plan=target_plan,
                )
        working_state["enabled_consolidation_write_lanes"] = accepted_lanes
    else:
        metadata = dict(working_state.get("metadata", {}) or {})
        metadata["write_success"] = {}
        working_state["metadata"] = metadata

    metadata = dict(working_state.get("metadata", {}) or {})
    metadata["lane_pipeline"] = {
        "mode": "dry_run" if dry_run else "apply",
        "accepted_lanes": accepted_lanes,
        "write_intent_count": len(write_intents),
        "character_operational_route": {
            "status": working_state["character_operational_work"]["status"],
            "error_code": character_operational_error or None,
        },
    }
    if router_validation_error:
        metadata["lane_pipeline"]["router_validation_error"] = (
            router_validation_error
        )
    if not identity_routed:
        metadata["identity_growth_routing"] = {
            "status": "not_routed",
            "reason_code": "not_routed",
        }
    elif identity_intent is None:
        metadata["identity_growth_routing"] = {
            "status": "rejected",
            "reason_code": "no_eligible_evidence",
        }
    elif identity_result is None:
        metadata["identity_growth_routing"] = {
            "status": "accepted",
            "reason_code": "candidate_emerging",
        }
    else:
        metadata["identity_growth_routing"] = {
            "status": identity_result["status"],
            "reason_code": identity_result["policy_reason_code"],
        }
    working_state["metadata"] = finalize_consolidation_metadata(metadata)

    packet = {
        "mode": "dry_run" if dry_run else "apply",
        "accepted_lanes": accepted_lanes,
        "source_views": source_views,
        "router_tasks": router_tasks,
        "lane_results": lane_results,
        "write_intents": write_intents,
        "character_operational_work": working_state[
            "character_operational_work"
        ],
        "state": working_state,
    }
    return packet


def _roster_entry(lane: str) -> dict[str, str]:
    """Build one router-visible roster row."""

    roster_entry = {
        "lane": lane,
        "description": _LANE_DESCRIPTIONS[lane],
    }
    return roster_entry


def _source_views_by_key(
    source_views: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index source views by source key."""

    views_by_key: dict[str, dict[str, Any]] = {}
    for source_view in source_views:
        source_key = text_or_empty(source_view.get("source_key"))
        if source_key:
            views_by_key[source_key] = source_view
    return views_by_key


def _operational_slot_is_available(
    state: Mapping[str, Any],
    source_views: list[dict[str, Any]],
) -> bool:
    """Return whether this settled normal-chat episode may use the slot."""

    origin = state.get("consolidation_origin")
    if not isinstance(origin, Mapping):
        return False
    if text_or_empty(origin.get("trigger_source")) != "user_message":
        return False
    for source_view in source_views:
        if (
            text_or_empty(source_view.get("source_key"))
            == ASSISTANT_ACCEPTANCE_SOURCE_KIND
            and text_or_empty(source_view.get("summary"))
        ):
            return True
    return False


def _validated_character_operational_evidence(
    *,
    task: Mapping[str, Any],
    source_views_by_key: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind router source keys to current-episode operational evidence only."""

    origin = state.get("consolidation_origin")
    if not isinstance(origin, Mapping):
        raise ValueError("character operational origin is unavailable")
    source_episode_id = text_or_empty(origin.get("episode_id"))
    occurred_at = text_or_empty(origin.get("storage_timestamp_utc"))
    if not source_episode_id or not occurred_at:
        raise ValueError("character operational provenance is unavailable")

    selected_source_keys = task.get("source_keys")
    if not isinstance(selected_source_keys, list):
        raise ValueError("character operational source keys are unavailable")
    evidence: list[dict[str, Any]] = []
    for source_key in selected_source_keys:
        if not isinstance(source_key, str):
            raise ValueError("character operational source key is invalid")
        source_view = source_views_by_key.get(source_key)
        if source_view is None:
            raise ValueError("character operational source key is unknown")
        evidence.append({
            "source_key": source_key,
            "source_kind": text_or_empty(source_view.get("source_kind")),
            "source_id": source_episode_id,
            "occurred_at": occurred_at,
            "semantic_text": text_or_empty(source_view.get("summary")),
        })
    return validate_character_operational_sources(evidence)


def _character_operational_route_status(
    *,
    task: Mapping[str, Any] | None,
    error_code: str,
    available: bool,
) -> str:
    """Project the route boundary into one public bounded status."""

    if not available:
        return "not_eligible"
    if error_code:
        return "failed"
    if task is None:
        return "no_change"
    return "selected"


def _selected_source_views(
    task: Mapping[str, Any],
    source_views_by_key: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve router-selected source keys to source-view rows."""

    selected_views: list[dict[str, Any]] = []
    raw_source_keys = task.get("source_keys")
    if not isinstance(raw_source_keys, list):
        return selected_views
    for source_key in raw_source_keys:
        clean_source_key = text_or_empty(source_key)
        if not clean_source_key:
            continue
        source_view = source_views_by_key.get(clean_source_key)
        if source_view is not None:
            selected_views.append(source_view)
    return selected_views


def _complete_required_source_views(
    lane: str,
    selected_views: list[dict[str, Any]],
    source_views_by_key: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach structurally required provenance for accepted-rule lanes."""

    if lane not in {"active_commitment", "character_self_guidance"}:
        return selected_views

    completed_views = list(selected_views)
    selected_keys = {
        text_or_empty(source_view.get("source_key"))
        for source_view in selected_views
    }
    for required_key in (
        "current_turn_user_message",
        ASSISTANT_ACCEPTANCE_SOURCE_KIND,
    ):
        if required_key in selected_keys:
            continue
        source_view = source_views_by_key.get(required_key)
        if source_view is not None:
            completed_views.append(source_view)
            selected_keys.add(required_key)
    return completed_views


def _privacy_review_for_state_or_views(
    state: Mapping[str, Any],
    source_views: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return optional privacy-review payload from state or selected views."""

    privacy_review = state.get("privacy_review")
    if isinstance(privacy_review, dict):
        return privacy_review
    for source_view in source_views:
        privacy_review = source_view.get("privacy_review")
        if isinstance(privacy_review, dict):
            return privacy_review
    return_value = None
    return return_value


def _write_intent_for_lane(
    lane: str,
    target_plan: ConsolidationTargetPlan,
    source_refs: list[dict[str, Any]],
    *,
    task: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build and validate one lane-level write intent."""

    target_alias, write_lane = _target_alias_and_write_lane(lane, target_plan)
    if not target_alias or not write_lane:
        return_value = None
        return return_value

    if lane == "character_identity_growth":
        payload = _identity_growth_payload(task=task, state=state)
        if payload is None:
            return None
    else:
        payload = {"source_refs": source_refs}
    intent = {
        "target_alias": target_alias,
        "write_lane": write_lane,
        "payload": payload,
    }
    try:
        validated_intent = validate_write_intent(intent, target_plan)
    except ConsolidationTargetValidationError as exc:
        logger.debug(f"lane write intent denied: {lane}: {exc}")
        return_value = None
        return return_value
    return_value = validated_intent
    return return_value


def _target_alias_and_write_lane(
    lane: str,
    target_plan: ConsolidationTargetPlan,
) -> tuple[str, str]:
    """Map consolidation lane names to existing target-plan write lanes."""

    if lane in {"user_memory_units", "active_commitment"}:
        return_value = (USER_TARGET_ALIAS, "user_memory_units")
        return return_value
    if lane == "character_self_guidance":
        return_value = (CHARACTER_TARGET_ALIAS, "character_self_guidance")
        return return_value
    if lane == "character_identity_growth":
        return_value = (
            CHARACTER_TARGET_ALIAS,
            "character_identity_growth",
        )
        return return_value
    if lane == "interaction_style_image":
        for target in target_plan["targets"]:
            if (
                target["target_alias"] == GROUP_CHANNEL_TARGET_ALIAS
                and "group_channel_style_image" in target["write_lanes"]
            ):
                return_value = (
                    GROUP_CHANNEL_TARGET_ALIAS,
                    "group_channel_style_image",
                )
                return return_value
        return_value = (USER_TARGET_ALIAS, "user_style_image")
        return return_value
    if lane == "shared_memory_promotion":
        return_value = (INTERNAL_TARGET_ALIAS, "shared_memory_promotion")
        return return_value
    return_value = ("", "")
    return return_value


def _identity_growth_payload(
    *,
    task: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Join router semantics to trusted settled-episode provenance."""

    origin = state.get("consolidation_origin")
    if not isinstance(origin, Mapping):
        return None
    episode_id = text_or_empty(origin.get("episode_id"))
    correlation_id = (
        text_or_empty(origin.get("correlation_id"))
        or episode_id
    )
    captured_at = text_or_empty(origin.get("storage_timestamp_utc"))
    local_date = _character_local_date(state)
    scope_kind = _identity_scope_kind(origin)
    identity_evidence = task.get("identity_evidence")
    if (
        not episode_id
        or not correlation_id
        or not captured_at
        or not local_date
        or scope_kind is None
        or not isinstance(identity_evidence, Mapping)
    ):
        return None

    evidence_ref_id = _identity_evidence_ref_id(
        episode_id=episode_id,
        correlation_id=correlation_id,
    )
    evidence_ref = {
        "schema_version": models.IDENTITY_EVIDENCE_SCHEMA_VERSION,
        "evidence_ref_id": evidence_ref_id,
        "root_episode_id": episode_id,
        "correlation_id": correlation_id,
        "source_kind": "settled_episode",
        "derived_reflection_run_ids": [],
        "character_local_date": local_date,
        "scope_kind": scope_kind,
        "captured_at": captured_at,
    }
    evidence_card = {
        "schema_version": models.IDENTITY_EVIDENCE_CARD_SCHEMA_VERSION,
        "evidence_ref_id": evidence_ref_id,
        "source_kind": "settled_episode",
        "character_local_date": local_date,
        "scope_kind": scope_kind,
        "decontextualized_event": identity_evidence[
            "decontextualized_event"
        ],
        "character_cognition_summary": identity_evidence[
            "character_cognition_summary"
        ],
        "visible_self_expression_summary": identity_evidence[
            "visible_self_expression_summary"
        ],
    }
    return {
        "correlation_id": correlation_id,
        "llm_trace_id": text_or_empty(state.get("llm_trace_id")),
        "evidence_refs": [evidence_ref],
        "evidence_cards": [evidence_card],
    }


async def _run_identity_growth_intent(
    intent: Mapping[str, Any],
    *,
    target_plan: ConsolidationTargetPlan,
) -> dict[str, Any]:
    """Execute one accepted identity intent through the single owner."""

    payload = intent.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("identity growth intent requires a payload")
    return await _run_identity_growth_source(
        payload,
        target_plan=target_plan,
    )


async def _run_identity_growth_source(
    source: Mapping[str, Any],
    *,
    target_plan: ConsolidationTargetPlan,
) -> dict[str, Any]:
    """Evaluate one rooted or explicitly empty identity source."""

    character_id = _character_id_from_target_plan(target_plan)
    current_revision = await get_current_identity(
        character_id=character_id,
    )
    result = await evaluate_episode_identity_growth(
        settled_episode=source,
        current_revision=current_revision,
    )
    return dict(result)


def _empty_identity_growth_source(
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build an auditable routed source with no eligible evidence."""

    origin = state.get("consolidation_origin")
    if not isinstance(origin, Mapping):
        return None
    correlation_id = (
        text_or_empty(origin.get("correlation_id"))
        or text_or_empty(origin.get("episode_id"))
    )
    if not correlation_id:
        return None
    return {
        "correlation_id": correlation_id,
        "llm_trace_id": text_or_empty(state.get("llm_trace_id")),
        "evidence_refs": [],
        "evidence_cards": [],
    }


def _character_id_from_target_plan(
    target_plan: ConsolidationTargetPlan,
) -> str:
    """Return the deterministic global character target identifier."""

    for target in target_plan["targets"]:
        if target["target_alias"] != CHARACTER_TARGET_ALIAS:
            continue
        character_id = text_or_empty(
            target["target_id"].get("character_id")
        )
        if character_id:
            return character_id
    raise ValueError("identity growth requires a character target")


def _character_local_date(state: Mapping[str, Any]) -> str:
    """Return the trusted character-local date from runtime state."""

    local_time_context = state.get("local_time_context")
    if isinstance(local_time_context, Mapping):
        current_date = text_or_empty(
            local_time_context.get("current_date")
        )
        if current_date:
            return current_date
        current_datetime = text_or_empty(
            local_time_context.get("current_local_datetime")
        )
        if len(current_datetime) >= 10:
            return current_datetime[:10]
    return ""


def _identity_scope_kind(
    origin: Mapping[str, Any],
) -> str | None:
    """Map trusted episode scope onto the closed identity scope enum."""

    trigger_source = text_or_empty(origin.get("trigger_source"))
    if trigger_source in {
        "internal_thought",
        "self_cognition",
        "scheduled_tick",
    }:
        return "self_cognition"
    channel_type = text_or_empty(origin.get("channel_type"))
    if channel_type in {"private", "group"}:
        return channel_type
    return None


def _identity_evidence_ref_id(
    *,
    episode_id: str,
    correlation_id: str,
) -> str:
    """Derive one opaque evidence handle without participant identifiers."""

    encoded = json.dumps(
        {
            "root_episode_id": episode_id,
            "correlation_id": correlation_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"identity-evidence:{hashlib.sha256(encoded).hexdigest()}"


def _ensure_writer_defaults(working_state: dict[str, Any]) -> None:
    """Populate writer state defaults produced by omitted lane specialists."""

    working_state.setdefault("new_facts", [])
    working_state.setdefault("future_promises", [])
    working_state.setdefault("character_self_guidance", {})
    working_state.setdefault("group_channel_style_image", {})
    working_state.setdefault("metadata", {})


async def _run_lane_specialists(
    working_state: dict[str, Any],
    accepted_lanes: list[str],
) -> None:
    """Run existing lane-local specialists before persistence."""

    accepted_lane_set = set(accepted_lanes)
    if "character_self_guidance" in accepted_lane_set:
        self_guidance_patch = await character_self_guidance_specialist(
            working_state
        )
        working_state.update(self_guidance_patch)
        reviewed_guidance = working_state.get("character_self_guidance")
        if not isinstance(reviewed_guidance, dict) or not reviewed_guidance:
            _disable_accepted_lane(
                working_state,
                accepted_lanes,
                "character_self_guidance",
            )


def _disable_accepted_lane(
    working_state: dict[str, Any],
    accepted_lanes: list[str],
    lane: str,
) -> None:
    """Remove a reviewer-rejected lane from the persistence allow-list."""

    while lane in accepted_lanes:
        accepted_lanes.remove(lane)

    enabled_lanes = working_state.get("enabled_consolidation_write_lanes")
    if isinstance(enabled_lanes, list):
        working_state["enabled_consolidation_write_lanes"] = [
            enabled_lane
            for enabled_lane in enabled_lanes
            if enabled_lane != lane
        ]

    metadata = dict(working_state.get("metadata", {}) or {})
    rejected_lanes = list(metadata.get("review_rejected_lanes", []) or [])
    rejected_lanes.append(lane)
    metadata["review_rejected_lanes"] = rejected_lanes
    working_state["metadata"] = metadata
