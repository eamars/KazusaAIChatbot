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
    "user_memory_units": "保存关于当前真实用户的持久事实、模式、变化或里程碑。",
    "active_commitment": "保存当前角色已经接受、且专门面向当前用户的承诺或持续规则。",
    "character_identity_growth": (
        "评估当前角色自我认同中可能持久、由角色自身形成的变化。"
    ),
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
检查的持久更新。另独立判断 character_operational_state_task：它不计入四项 lane task，且仅表示
本 episode 是否需要一次 source-free 的角色短期运行状态评估。只返回 lane name、简短 reason 和
来自 source_views 的 source_key。持久化细节、记忆正文、target id、时间戳与缓存行为由后续确定性阶段负责。

# Operational slot
The operational slot is independent from durable consolidation lanes. It asks
whether the completed episode leaves a source-free character-level posture that
the next turn may need to consume. A durable lane and an operational task may
both be selected for the same episode.

Return a non-null operational task when the accepted episode contains a
character-facing consequence that can survive the current scene, including:
- deliberate harm, humiliation, coercion, rejection, or a boundary violation;
- a threat, loss, exposure, or other event that can leave residual pressure;
- an apology, repair, or accepted change that changes the character's posture;
- an accepted task result whose outcome can shape the next character turn.

For these cases, select one to four source_keys from the supplied source_views.
Prefer current_turn_user_message together with assistant_final_dialog; add
internal_thought or episode_trace only when they are present and materially
support the same source-free consequence. The operational task coexists with
durable lane tasks and never replaces or suppresses them.

Return null only when the episode is clearly ordinary, informational, or
transient and leaves no character-level consequence beyond the current scene,
or when no accepted assistant dialog exists. A durable lane selection alone
does not justify null for a boundary, repair, or user fact.

The operational object must contain exactly these keys and no others:
{"reason": "short bounded reason", "source_keys": ["source_key"]}
The top-level response must always contain exactly lane_tasks and
character_operational_state_task. Always include that key with either null or
the exact object shape above; keep operational fields out of durable lane rows.

# 判断步骤
1. 阅读 source_views，判断已完成回应之后是否形成持久记忆更新。
2. 识别更新的归属与范围：当前用户、角色认同、角色指导、群组或频道风格，或已经批准提升的反思。
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
- character_identity_growth：角色自己的认同、人格判断、边界或自我概念出现可能持久的变化。
  亲密关系经历也可能促成角色自己的持久变化；关系对象、关系事实与私密细节仍归原有作用域，
  只有角色自己的抽象变化进入此 lane。
  此 lane 的 task 额外返回 identity_evidence，其中 decontextualized_event 概括发生的事，
  character_cognition_summary 概括角色自身判断，visible_self_expression_summary 概括角色
  可见的自我表达。摘要保持抽象并适用于所有聊天范围。
- character_self_guidance：由当前角色承担、并普遍适用于未来社交场景的已接受行为指导。
- interaction_style_image：target plan 与来源角色允许时，记录用户风格或群组、频道互动规范。
- shared_memory_promotion：经过隐私检查并获准提升的反思或共享记忆证据。

# 跳过条件
一轮角色扮演或临时行为、聊天中的普通世界知识、与当前用户无关的第三方事实，以及最终对话尚未
接受或只在当前情境成立的未来行为，均返回空 lane_tasks。

# 输出格式
{
  "lane_tasks": [],
  "character_operational_state_task": {
    "reason": "A deliberate boundary violation may leave residual character posture.",
    "source_keys": ["current_turn_user_message", "assistant_final_dialog"]
  }
}

For a character_identity_growth lane row, identity_evidence is an object with
exactly these keys: decontextualized_event, character_cognition_summary, and
visible_self_expression_summary. Omit identity_evidence for every other lane.
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
        prompt_views.append({
            "source_key": text_or_empty(source_view.get("source_key")),
            "source_kind": text_or_empty(source_view.get("source_kind")),
            "summary": text_or_empty(source_view.get("summary")),
        })
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
