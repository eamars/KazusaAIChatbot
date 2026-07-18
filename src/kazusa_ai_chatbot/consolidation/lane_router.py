"""Coarse consolidation lane routing and auditable lane pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    CONSOLIDATION_LLM_MODEL,
    CONSOLIDATION_LLM_THINKING_ENABLED,
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
    "character_self_guidance",
    "interaction_style_image",
    "shared_memory_promotion",
)

_ROUTER_TASK_KEYS = frozenset(("lane", "reason", "source_keys"))
_FORBIDDEN_ROUTER_TASK_KEYS = frozenset(
    ("target_id", "write_lane", "payload", "fact")
)
_MAX_ROUTER_TASKS = 4

_LANE_DESCRIPTIONS = {
    "user_memory_units": "保存关于当前真实用户的持久事实、模式、变化或里程碑。",
    "active_commitment": "保存当前角色已经接受、且专门面向当前用户的承诺或持续规则。",
    "character_self_guidance": "保存由当前角色承担的通用未来行为指导。",
    "interaction_style_image": "更新用户或群组的互动风格画像。",
    "shared_memory_promotion": "只把已经提升的反思证据接纳进共享记忆。",
}

_ROUTER_PROMPT = '''\
你负责把一个已经完成的 episode 路由到粗粒度的 consolidation lane task。

HumanMessage 中包含：
- target_plan：确定性代码给出的合格持久化目标；
- lane_roster：本 episode 唯一可选的 lane name；
- source_views：可安全用于 prompt、并带有 source_key 的证据行。

从 lane_roster 中选择零到四项 lane task。一项 task 表示本 episode 存在值得由对应 specialist
检查的持久更新。只返回 lane name、简短 reason 和来自 source_views 的 source_key。持久化细节、
记忆正文、target id、时间戳与缓存行为由后续确定性阶段负责。

# 判断步骤
1. 阅读 source_views，判断已完成回应之后是否形成持久记忆更新。
2. 识别更新的归属与范围：当前用户、角色指导、群组或频道风格，或已经批准提升的反思。
3. 从 lane_roster 选择匹配的 lane。若列表中没有 lane 拥有该持久更新，返回空 lane_tasks。
4. 对已经接受的未来行为规则，如果请求来源和最终对话接受来源的 source_key 都可用，则同时引用。
5. 当持久主题是用户事实或偏好，而当前角色只是确认、记住、尊重或配合它时，选择用户拥有的
   lane；角色的配合是该用户记忆的支持，不另建角色行为规则。
6. 路由保持粗粒度，实际记忆候选由所选 specialist 写入或拒绝。

# Lane 归属
- user_memory_units：关于当前真实用户的持久信息，例如个人事实、偏好、习惯、近期变化、里程碑，
  或对已回忆用户记忆的更新。
- active_commitment：当前角色已经接受、且仅面向当前用户的未来行为，例如承诺、提醒、称呼规则
  或持续互动规则。
- character_self_guidance：由当前角色承担、并普遍适用于未来社交场景的已接受行为指导。
- interaction_style_image：target plan 与来源角色允许时，记录用户风格或群组、频道互动规范。
- shared_memory_promotion：经过隐私检查并获准提升的反思或共享记忆证据。

# 跳过条件
一轮角色扮演或临时行为、聊天中的普通世界知识、与当前用户无关的第三方事实，以及最终对话尚未
接受或只在当前情境成立的未来行为，均返回空 lane_tasks。

# 输出格式
只返回有效 JSON：
{
  "lane_tasks": [
    {
      "lane": "lane_roster 中的一个 lane",
      "reason": "简短语义理由",
      "source_keys": ["source_key"]
    }
  ]
}
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
) -> dict[str, list[dict[str, Any]]]:
    """Validate that router output contains only coarse lane tasks.

    Args:
        output: Parsed router JSON.
        roster: Lane roster built from the target plan.

    Returns:
        Normalized router output with validated lane tasks.

    Raises:
        ValueError: If the output contains unknown lanes, non-roster lanes,
            persistence fields, memory payload fields, or malformed task rows.
    """

    lane_tasks = output.get("lane_tasks")
    if not isinstance(lane_tasks, list):
        raise ValueError("lane_tasks must be a list")

    roster_lanes = {
        text_or_empty(entry.get("lane"))
        for entry in roster
        if isinstance(entry, Mapping)
    }
    validated_tasks: list[dict[str, Any]] = []
    for raw_task in lane_tasks[:_MAX_ROUTER_TASKS]:
        if not isinstance(raw_task, Mapping):
            raise ValueError("lane task must be an object")
        task_keys = set(raw_task)
        if task_keys & _FORBIDDEN_ROUTER_TASK_KEYS:
            raise ValueError("router task contains persistence or memory fields")
        if task_keys != _ROUTER_TASK_KEYS:
            raise ValueError("router task must contain only lane, reason, source_keys")

        lane = text_or_empty(raw_task.get("lane"))
        if lane not in CONSOLIDATION_LANE_NAMES:
            raise ValueError(f"unknown consolidation lane: {lane!r}")
        if lane not in roster_lanes:
            raise ValueError(f"lane is not in target roster: {lane!r}")

        reason = text_or_empty(raw_task.get("reason"))
        raw_source_keys = raw_task.get("source_keys")
        if not isinstance(raw_source_keys, list):
            raise ValueError("source_keys must be a list")
        source_keys = [
            source_key.strip()
            for source_key in raw_source_keys
            if isinstance(source_key, str) and source_key.strip()
        ]
        validated_tasks.append(
            {
                "lane": lane,
                "reason": reason,
                "source_keys": source_keys,
            }
        )

    validated_output = {"lane_tasks": validated_tasks}
    return validated_output


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
        "source_views": source_views,
    }
    system_prompt = SystemMessage(content=_ROUTER_PROMPT)
    human_message = HumanMessage(content=json.dumps(payload, ensure_ascii=False))
    response = await _lane_router_llm.ainvoke(
        [system_prompt, human_message],
        config=_lane_router_llm_config,
    )
    parsed_output = parse_llm_json_output(response.content)
    return parsed_output


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
    router_output = await call_lane_router_llm(
        state,
        source_views=source_views,
        roster=roster,
    )
    router_validation_error = ""
    try:
        validated_router_output = validate_lane_router_output(
            router_output,
            roster,
        )
    except ValueError as exc:
        logger.warning(f"lane router output dropped: {exc}")
        router_tasks = []
        router_validation_error = str(exc)
    else:
        router_tasks = validated_router_output["lane_tasks"]

    lane_results: list[dict[str, Any]] = []
    write_intents: list[dict[str, Any]] = []
    accepted_lanes: list[str] = []
    accepted_user_memory_refs: list[dict[str, Any]] = []
    accepted_self_guidance_refs: list[dict[str, Any]] = []

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
        )
        if write_intent is not None:
            write_intents.append(write_intent)
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
    _ensure_writer_defaults(working_state)

    if not dry_run and accepted_lanes:
        await _run_lane_specialists(working_state, accepted_lanes)
        if accepted_lanes:
            writer_result = await db_writer(working_state)
            working_state.update(writer_result)
        else:
            metadata = dict(working_state.get("metadata", {}) or {})
            metadata["write_success"] = {}
            working_state["metadata"] = metadata
    else:
        metadata = dict(working_state.get("metadata", {}) or {})
        metadata["write_success"] = {}
        working_state["metadata"] = metadata

    metadata = dict(working_state.get("metadata", {}) or {})
    metadata["lane_pipeline"] = {
        "mode": "dry_run" if dry_run else "apply",
        "accepted_lanes": accepted_lanes,
        "write_intent_count": len(write_intents),
    }
    if router_validation_error:
        metadata["lane_pipeline"]["router_validation_error"] = (
            router_validation_error
        )
    working_state["metadata"] = finalize_consolidation_metadata(metadata)

    packet = {
        "mode": "dry_run" if dry_run else "apply",
        "accepted_lanes": accepted_lanes,
        "source_views": source_views,
        "router_tasks": router_tasks,
        "lane_results": lane_results,
        "write_intents": write_intents,
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
) -> dict[str, Any] | None:
    """Build and validate one lane-level write intent."""

    target_alias, write_lane = _target_alias_and_write_lane(lane, target_plan)
    if not target_alias or not write_lane:
        return_value = None
        return return_value

    intent = {
        "target_alias": target_alias,
        "write_lane": write_lane,
        "payload": {"source_refs": source_refs},
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
