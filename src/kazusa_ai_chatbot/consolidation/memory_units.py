"""Consolidator memory-unit extraction and merge helpers."""

from __future__ import annotations

import json
import logging
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (

    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
    CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    CONSOLIDATION_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.calendar_scheduler import repository as calendar_repository
from kazusa_ai_chatbot.calendar_scheduler.handlers import (
    reconcile_active_commitment_calendar_schedule,
)
from kazusa_ai_chatbot.db import (
    UserMemoryUnitStatus,
    UserMemoryUnitType,
    insert_user_memory_units,
    update_user_memory_unit_semantics,
    update_user_memory_unit_window,
)
from kazusa_ai_chatbot.memory_writer_prompt_projection import (
    project_memory_unit_extractor_prompt_payload,
    project_memory_unit_rewrite_prompt_payload,
)
from kazusa_ai_chatbot.consolidation.origin import (
    project_consolidation_origin_prompt_block,
)
from kazusa_ai_chatbot.consolidation.schema import ConsolidatorState
from kazusa_ai_chatbot.conversation_history_prompt_projection import (
    project_conversation_history_for_llm,
)
from kazusa_ai_chatbot.rag.prompt_projection import project_tool_result_for_llm
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import retrieve_memory_unit_merge_candidates
from kazusa_ai_chatbot.time_boundary import (
    format_storage_utc_for_llm,
    local_llm_datetime_to_storage_utc_iso,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output, text_or_empty


from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
MAX_MEMORY_UNIT_CANDIDATES_PER_TURN = 3
MAX_MEMORY_UNIT_MERGE_CANDIDATES = 6

VALID_EXTRACTED_USER_MEMORY_UNIT_TYPES = {
    UserMemoryUnitType.STABLE_PATTERN,
    UserMemoryUnitType.RECENT_SHIFT,
    UserMemoryUnitType.OBJECTIVE_FACT,
    UserMemoryUnitType.MILESTONE,
    UserMemoryUnitType.ACTIVE_COMMITMENT,
}
_MEMORY_UNIT_WRITE_LANE_UNIT_TYPES = {
    "user_memory_units": (
        UserMemoryUnitType.STABLE_PATTERN,
        UserMemoryUnitType.RECENT_SHIFT,
        UserMemoryUnitType.OBJECTIVE_FACT,
        UserMemoryUnitType.MILESTONE,
    ),
    "active_commitment": (UserMemoryUnitType.ACTIVE_COMMITMENT,),
}
_MEMORY_UNIT_WRITE_LANES = (
    "user_memory_units",
    "active_commitment",
)

logger = logging.getLogger(__name__)


def _memory_unit_write_contract(state: ConsolidatorState) -> dict:
    """Build the ordered contract from router-approved consolidation lanes.

    Args:
        state: Consolidator state carrying the router-approved write lanes.

    Returns:
        Code-owned enabled-lane and allowed-unit-type lists in contract order.
    """

    enabled_lanes = state.get("enabled_consolidation_write_lanes", [])
    if not isinstance(enabled_lanes, (list, tuple, set)):
        enabled_lanes = []

    accepted_lanes = [
        lane for lane in _MEMORY_UNIT_WRITE_LANES if lane in enabled_lanes
    ]
    allowed_unit_types: list[str] = []
    for lane in accepted_lanes:
        for unit_type in _MEMORY_UNIT_WRITE_LANE_UNIT_TYPES[lane]:
            if unit_type not in allowed_unit_types:
                allowed_unit_types.append(unit_type)

    return_value = {
        "enabled_lanes": accepted_lanes,
        "allowed_unit_types": allowed_unit_types,
    }
    return return_value


def _json_payload(state: ConsolidatorState) -> dict:
    rag_result = state["rag_result"]
    if not isinstance(rag_result, dict):
        raise TypeError("consolidation rag_result must be a mapping")
    rag_candidates = rag_result["user_memory_unit_candidates"]
    if not isinstance(rag_candidates, list):
        raise TypeError("user_memory_unit_candidates must be a list")
    projected_memory_candidates = project_tool_result_for_llm(rag_candidates)
    if not isinstance(projected_memory_candidates, list):
        raise TypeError("projected memory candidates must be a list")

    local_datetime = state["local_time_context"]["current_local_datetime"]
    return_value = {
        "timestamp": local_datetime,
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "consolidation_origin": project_consolidation_origin_prompt_block(
            state["consolidation_origin"]
        ),
        "decontextualized_input": state["decontextualized_input"],
        "final_dialog": state["final_dialog"],
        "internal_monologue": state["internal_monologue"],
        "emotional_appraisal": state["emotional_appraisal"],
        "interaction_subtext": state["interaction_subtext"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "chat_history_recent": project_conversation_history_for_llm(
            state["chat_history_recent"],
            character_name=state.get("character_name", ""),
        ),
        "rag_user_memory_unit_candidates": projected_memory_candidates,
        "new_facts_evidence": project_tool_result_for_llm(
            state["new_facts"]
        ),
        "future_promises_evidence": project_tool_result_for_llm(
            state["future_promises"]
        ),
        "subjective_appraisal_evidence": project_tool_result_for_llm(
            state["subjective_appraisals"]
        ),
        "memory_unit_write_contract": _memory_unit_write_contract(state),
    }
    return return_value


def _rag_surfaced_memory_units(state: ConsolidatorState) -> list[dict]:
    """Return memory-unit candidates already surfaced by the RAG layer.

    Args:
        state: Current consolidator state with the RAG projection attached.

    Returns:
        The list of raw memory unit documents RAG retrieved for this turn.
    """

    rag_result = state["rag_result"]
    if not isinstance(rag_result, dict):
        raise TypeError("consolidation rag_result must be a mapping")
    surfaced_units = rag_result["user_memory_unit_candidates"]
    if not isinstance(surfaced_units, list):
        raise TypeError("user_memory_unit_candidates must be a list")
    valid_units = [unit for unit in surfaced_units if isinstance(unit, dict)]
    return valid_units


def _candidate_with_id(
    candidate: dict,
    default_source_refs: list[dict] | None = None,
) -> dict:
    item = dict(candidate)
    item["candidate_id"] = text_or_empty(item.get("candidate_id")) or uuid4().hex
    if not isinstance(item.get("evidence_refs"), list):
        item["evidence_refs"] = []
    if not isinstance(item.get("source_refs"), list) or not item["source_refs"]:
        if item["evidence_refs"]:
            item["source_refs"] = list(item["evidence_refs"])
        elif default_source_refs:
            item["source_refs"] = list(default_source_refs)
        else:
            item["source_refs"] = []
    return item


def _normalize_candidate_lifecycle_fields(candidate: dict) -> tuple[dict, list[str]]:
    """Normalize optional lifecycle timestamps emitted by the extractor.

    Args:
        candidate: Extractor-authored candidate with optional lifecycle fields.

    Returns:
        A pair of normalized candidate and structural validation errors.
    """

    normalized_candidate = dict(candidate)
    errors: list[str] = []
    for field in ("due_at", "completed_at", "cancelled_at"):
        raw_value = text_or_empty(candidate.get(field))
        if not raw_value:
            normalized_candidate.pop(field, None)
            continue
        try:
            normalized_candidate[field] = local_llm_datetime_to_storage_utc_iso(
                raw_value
            )
        except ValueError as exc:
            errors.append(f"invalid {field}: {exc}")

    return_value = (normalized_candidate, errors)
    return return_value


def _candidate_lifecycle_updates(candidate: dict) -> dict:
    """Return lifecycle fields that should be preserved during a merge update.

    Args:
        candidate: Structurally valid memory-unit candidate.

    Returns:
        Lifecycle field updates supplied by the extractor.
    """

    updates: dict = {}
    for field in ("due_at", "completed_at", "cancelled_at"):
        if field in candidate and text_or_empty(candidate.get(field)):
            updates[field] = candidate[field]
    return updates


def _candidate_source_refs(candidate: dict) -> list[dict]:
    """Return structurally valid source refs from one candidate."""

    raw_refs = candidate.get("source_refs")
    if not isinstance(raw_refs, list):
        raw_refs = candidate.get("evidence_refs")
    if not isinstance(raw_refs, list):
        return_value: list[dict] = []
        return return_value
    return_value = [
        dict(source_ref)
        for source_ref in raw_refs
        if isinstance(source_ref, dict) and source_ref
    ]
    return return_value


def _candidate_validation_errors(
    candidate: dict,
    *,
    allowed_unit_types: set[str],
) -> list[str]:
    """Return structural errors for an extractor-authored memory unit.

    Args:
        candidate: Candidate memory-unit dictionary after id normalization.

    Returns:
        Validation error strings. An empty list means the candidate is usable.
    """

    errors: list[str] = []
    unit_type = text_or_empty(candidate.get("unit_type"))
    if unit_type not in allowed_unit_types:
        errors.append(f"invalid unit_type: {unit_type!r}")

    for field in ("fact", "subjective_appraisal", "relationship_signal"):
        if not text_or_empty(candidate.get(field)):
            errors.append(f"missing field: {field}")

    source_refs = candidate.get("source_refs")
    if not isinstance(source_refs, list) or not source_refs:
        errors.append("source_refs must be a non-empty list")

    return errors


def _validated_candidates(
    result: dict,
    default_source_refs: list[dict] | None = None,
    *,
    allowed_unit_types: set[str],
) -> tuple[list[dict], list[dict]]:
    """Split extractor output into usable candidates and validation errors.

    Args:
        result: Parsed JSON object returned by the extractor LLM.

    Returns:
        A pair of valid candidates and structured invalid-candidate records.
    """

    raw_candidates = result.get("memory_units", [])
    if not isinstance(raw_candidates, list):
        validation_errors = [{
            "candidate_id": "",
            "errors": ["memory_units must be a list"],
        }]
        return_value = ([], validation_errors)
        return return_value

    candidates: list[dict] = []
    validation_errors: list[dict] = []
    for index, raw_candidate in enumerate(raw_candidates[:MAX_MEMORY_UNIT_CANDIDATES_PER_TURN]):
        if not isinstance(raw_candidate, dict):
            validation_errors.append({
                "candidate_id": f"index-{index}",
                "errors": ["candidate must be an object"],
            })
            continue

        candidate = _candidate_with_id(
            raw_candidate,
            default_source_refs=default_source_refs,
        )
        candidate, lifecycle_errors = _normalize_candidate_lifecycle_fields(
            candidate
        )
        candidate_errors = _candidate_validation_errors(
            candidate,
            allowed_unit_types=allowed_unit_types,
        )
        candidate_errors.extend(lifecycle_errors)
        if candidate_errors:
            validation_errors.append({
                "candidate_id": candidate["candidate_id"],
                "errors": candidate_errors,
            })
            continue

        candidates.append(candidate)

    return_value = (candidates, validation_errors)
    return return_value


def _valid_candidates(
    result: dict,
    default_source_refs: list[dict] | None = None,
    *,
    allowed_unit_types: set[str],
) -> list[dict]:
    candidates, validation_errors = _validated_candidates(
        result,
        default_source_refs=default_source_refs,
        allowed_unit_types=allowed_unit_types,
    )
    if validation_errors:
        logger.warning(f"memory-unit extractor dropped invalid candidates: {validation_errors}")
    return candidates


def _validate_merge_result(result: dict, candidate: dict, candidate_clusters: list[dict]) -> dict:
    expected_candidate_id = candidate["candidate_id"]
    decision = text_or_empty(result.get("decision"))
    cluster_id = text_or_empty(result.get("cluster_id"))
    valid_cluster_ids = {
        text_or_empty(cluster.get("unit_id"))
        for cluster in candidate_clusters
        if text_or_empty(cluster.get("unit_id"))
    }

    if decision not in {"create", "merge", "evolve"}:
        raise ValueError(f"invalid merge decision: {decision!r}")
    if decision == "create" and cluster_id:
        raise ValueError("create decision must not include cluster_id")
    if decision in {"merge", "evolve"} and cluster_id not in valid_cluster_ids:
        logger.warning(
            "memory-unit merge/evolve decision used an unknown cluster_id; "
            f"falling back to create: candidate_id={expected_candidate_id} "
            f"decision={decision} cluster_id={cluster_id}"
        )
        decision = "create"
        cluster_id = ""

    return_value = {
        "candidate_id": expected_candidate_id,
        "decision": decision,
        "cluster_id": cluster_id,
        "reason": text_or_empty(result.get("reason")),
    }
    return return_value


def _validate_rewrite_result(result: dict) -> dict:
    """Validate semantic fields returned by the rewrite stage.

    Args:
        result: Parsed JSON object returned by the rewrite LLM.

    Returns:
        The replacement semantic fields for the selected memory unit.
    """

    if not isinstance(result, dict):
        raise ValueError("rewrite result must be an object")

    for field in ("fact", "subjective_appraisal", "relationship_signal"):
        if not text_or_empty(result.get(field)):
            raise ValueError(f"rewrite missing field: {field}")
    return_value = {
        "fact": text_or_empty(result["fact"]),
        "subjective_appraisal": text_or_empty(result["subjective_appraisal"]),
        "relationship_signal": text_or_empty(result["relationship_signal"]),
    }
    return return_value


def _validate_stability_result(result: dict, unit_id: str) -> dict:
    if text_or_empty(result.get("unit_id")) != unit_id:
        raise ValueError("stability judge returned an unknown unit_id")
    window = text_or_empty(result.get("window"))
    if window not in {"recent", "stable"}:
        raise ValueError(f"invalid stability window: {window!r}")
    return_value = {
        "unit_id": unit_id,
        "window": window,
        "reason": text_or_empty(result.get("reason")),
    }
    return return_value


def _matching_cluster(candidate_clusters: list[dict], unit_id: str) -> dict:
    """Return the candidate cluster matching a stored memory-unit id.

    Args:
        candidate_clusters: Existing memory units surfaced for merge judgment.
        unit_id: Stored unit id selected by merge/create handling.

    Returns:
        The matching cluster, or an empty dict when the unit was just created.
    """

    for cluster in candidate_clusters:
        if text_or_empty(cluster.get("unit_id")) == unit_id:
            return cluster
    return_value = {}
    return return_value


def _count_description(count: int) -> str:
    """Convert an occurrence count into a semantic label for local LLM input.

    Args:
        count: Number of observed occurrences attached to a memory unit.

    Returns:
        A short descriptor that helps the LLM interpret the raw count.
    """

    if count <= 1:
        return "single_observation"
    if count == 2:
        return "two_observations"
    if count <= 4:
        return "several_observations"
    return_value = "many_observations"
    return return_value


def _session_spread(source_refs: list[dict]) -> dict:
    """Summarize source-reference spread for stability judging.

    Args:
        source_refs: Evidence references stored on the memory unit.

    Returns:
        A dict with both raw evidence and a semantic spread label.
    """

    timestamp_days = set()
    for ref in source_refs:
        raw_ts = text_or_empty(ref.get("timestamp"))
        if not raw_ts:
            continue
        formatted_timestamp = format_storage_utc_for_llm(raw_ts)
        day = (formatted_timestamp or raw_ts)[:10]
        if day:
            timestamp_days.add(day)
    message_ids = {
        text_or_empty(ref.get("message_id"))
        for ref in source_refs
        if text_or_empty(ref.get("message_id"))
    }
    distinct_day_count = len(timestamp_days)
    if distinct_day_count == 0:
        spread_label = "unknown_session_spread"
    elif distinct_day_count == 1:
        spread_label = "single_day_or_session"
    else:
        spread_label = "multiple_days_or_sessions"
    return_value = {
        "spread_label": spread_label,
        "distinct_day_count": distinct_day_count,
        "distinct_message_ref_count": len(message_ids),
        "timestamps": sorted(timestamp_days),
    }
    return return_value


def _recent_examples(candidate: dict, cluster: dict) -> list[dict]:
    """Build compact example evidence for the stability judge.

    Args:
        candidate: Newly extracted memory unit candidate.
        cluster: Existing stored memory unit when merge/evolve selected one.

    Returns:
        Recent example records with fact text and timestamps.
    """

    examples = []
    if cluster:
        examples.append({
            "source": "existing_unit",
            "fact": text_or_empty(cluster.get("fact")),
            "updated_at": format_storage_utc_for_llm(
                text_or_empty(cluster.get("updated_at"))
            ),
        })
    examples.append({
        "source": "new_candidate",
        "fact": text_or_empty(candidate.get("fact")),
        "updated_at": "",
    })
    return examples[:3]


def _stability_payload(
    state: ConsolidatorState,
    *,
    unit_id: str,
    candidate: dict,
    merge_result: dict,
    candidate_clusters: list[dict],
) -> dict:
    """Build the evidence payload consumed by the stability judge LLM.

    Args:
        state: Current consolidator state.
        unit_id: Stored unit id to classify as recent or stable.
        candidate: New candidate memory unit.
        merge_result: Merge judge decision for the candidate.
        candidate_clusters: Existing units shown to the merge judge.

    Returns:
        JSON payload with semantic evidence labels and raw support details.
    """

    local_datetime = state["local_time_context"]["current_local_datetime"]
    cluster = project_tool_result_for_llm(
        _matching_cluster(candidate_clusters, unit_id)
    )
    if not isinstance(cluster, dict):
        cluster = {}
    candidate = project_tool_result_for_llm(candidate)
    if not isinstance(candidate, dict):
        candidate = {}
    merge_result = project_tool_result_for_llm(merge_result)
    if not isinstance(merge_result, dict):
        merge_result = {}
    existing_count = int(cluster.get("count", 0) or 0)
    candidate_refs = candidate.get("evidence_refs")
    if not isinstance(candidate_refs, list):
        candidate_refs = []
    source_refs = cluster.get("source_refs")
    if not isinstance(source_refs, list):
        source_refs = []
    combined_count = max(existing_count, 1) + len(candidate_refs)
    return_value = {
        "unit_id": unit_id,
        "candidate": candidate,
        "merge_result": merge_result,
        "stability_evidence": {
            "occurrence_count": combined_count,
            "occurrence_count_label": _count_description(combined_count),
            "existing_unit_count": existing_count,
            "new_evidence_ref_count": len(candidate_refs),
            "session_spread": _session_spread(source_refs + candidate_refs),
            "recency": {
                "current_turn_timestamp": local_datetime,
                "existing_updated_at": format_storage_utc_for_llm(
                    text_or_empty(cluster.get("updated_at"))
                ),
                "existing_last_seen_at": format_storage_utc_for_llm(
                    text_or_empty(cluster.get("last_seen_at"))
                ),
            },
            "recent_examples": _recent_examples(candidate, cluster),
        },
    }
    return return_value


_EXTRACTOR_PROMPT = '''\
# 任务
你从本轮持久化整理输入中提取新的、可长期保存的用户记忆单元，供 `{character_name}` 以后与该用户互动时使用。
你只提取候选记忆，不判断 create、merge 或 evolve。
如果本轮没有值得长期保存的新内容，返回空的 `memory_units`。

# 语言政策
- JSON 字段、结构化枚举值、ID、URL、代码、命令和模型标签保持原样。
- `unit_type`、`evidence_refs.source` 等枚举字段必须保持输出格式指定的英文值。
- 由你新生成的自由文本字段 `fact`、`subjective_appraisal`、`relationship_signal` 必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、外部证据原句只有在必须精确保留时才保持原语言。
- 指向 `{character_name}` 的短名、别名、旧称呼、显示名或 assistant 等机器标签只可用于理解，不可复制到输出字段。
- 不添加翻译、双语复写或括号解释，除非源文本本身已经包含。

# 证据读取与身份
1. 先读 `timestamp`，它是本轮持久化整理的本地时间。
2. 读 `consolidation_origin.trigger_source`。它只能是 `user_message`、`internal_thought`、`self_cognition`、`scheduled_tick` 或 `tool_result`。`user_message` 表示本轮由用户消息触发；`internal_thought` 表示由已发出的后续行动接力触发；`self_cognition` 表示由有观察依据的空闲认知触发；`scheduled_tick` 表示由已认领的到期计划触发；`tool_result` 表示由已完成的工具结果触发。
3. 再读 `chat_history_recent`。每行格式为 `[时间] 说话人: 内容`；用行首说话人判断每条消息是谁说的；消息里的“我”必须按原说话人理解。
4. 读 `decontextualized_input`、`final_dialog`、`logical_stance`、`character_intent`，确认本轮发生了什么，以及 `{character_name}` 是否真的接受了某个后续行为。
5. 当 trigger_source 是 `user_message` 时，`decontextualized_input` 是用户本轮表达；当 trigger_source 是 `internal_thought`、`self_cognition` 或 `scheduled_tick` 时，它是有依据的内部触发文本，不是用户原话；当 trigger_source 是 `tool_result` 时，它是工具结果及原始目标的去上下文化摘要。
6. `final_dialog` 在 `user_message` 和允许交付的 `tool_result` 中是可见回复，在 `internal_thought`、`self_cognition` 和 `scheduled_tick` 中是私有或预览整理结果。
7. `new_facts_evidence` 和 `future_promises_evidence` 是上游证据提示，不是必须照抄的输出。
8. `internal_monologue`、`emotional_appraisal`、`interaction_subtext`、`subjective_appraisal_evidence` 只用于理解 `{character_name}` 如何看待已确认事实，不可单独当作用户事实。
9. 对照 `rag_user_memory_unit_candidates`。只有本轮带来新事实、更清楚的细节或新的未来互动含义时，才生成记忆单元。
10. `memory_unit_write_contract` 是本轮路由已经批准的写入范围。只输出 `allowed_unit_types` 中的 `unit_type`，不要扩大、改写或替换该范围；列表为空时返回空 `memory_units`。

# 候选记忆准入
- 只保存具体事件、决定、偏好、承诺、可复用行为模式或重要转折。
- 不保存单纯语气、一次性心情、普通寒暄、重复旧记忆或只描述最新消息态度的内容。
- 一个具体事件只生成一个记忆单元。
- 用户提出请求并且 `{character_name}` 接受后续遵守时，这是一个 `active_commitment`，不要拆成“用户偏好”和“接受回应”两条。
- `active_commitment` 只记录当前角色已经明确接受、并准备面向当前用户持续执行的未来行为。
- 角色单方面提出的要求、条件或互动安排，以及当前用户未作回应、仅继续相邻事项的内容，不能单独建立 `active_commitment`；没有允许类型对应的可保存证据时返回空 `memory_units`。
- 当 `future_promises_evidence` 与用户本轮请求或偏好指向同一个后续行为时，只生成一条 `active_commitment`；不要再为同一请求另建 `objective_fact`。
- 如果证据中有多个可长期保存的主题，只有在它们有不同的未来互动含义时才分成多条。
- 用户明确说明某个项目名、代号、标题或外部名称属于用户自己时，优先直接记录用户事实；只在该对比本身会影响未来互动时，才保留它不是指向 `{character_name}` 的说明。

# 时间、承诺与有效性
- 生成 `fact`、`subjective_appraisal`、`relationship_signal` 时，不把未解析的相对时间当作当前或未来事实保存。
- 如果时间、日期、截止点、展示日、挑战日、后续验收或相对顺序会决定一条记忆何时生效、何时到期或是否仍可执行，必须先用 `timestamp`、消息时间和本轮证据把它写成绝对本地日期或日期时间。
- 能确定日期但没有具体时刻的 `active_commitment`，`due_at` 使用该本地日期的 `00:00`，格式必须是 `YYYY-MM-DD HH:MM`。
- 能确定具体执行时间的 `active_commitment`，`due_at` 使用本地 `YYYY-MM-DD HH:MM`。
- 无到期日的持续规则、长期偏好或稳定事实可以省略 `due_at`。
- 如果一个时间性承诺只能看到“下次”“之后”“回头”这类相对说法，且输入不足以确定具体日期、时间或当前状态，不输出该 memory_unit。
- `due_at` 只能填写精确 `YYYY-MM-DD HH:MM`；无法得到这种精确值时省略该字段。

# unit_type 判定
- `objective_fact`: 用户事实、用户偏好、项目名称、明确决定或系统性说明；如果同一事实已经被 `{character_name}` 接受为后续行为，改用 `active_commitment`。
- `milestone`: 一次性的重要事件、清晰转折或长期关系/协作方式的改变。
- `active_commitment`: `{character_name}` 已接受的持续承诺、后续行为或仍需遵守的偏好。
- `recent_shift`: 新出现的短期变化、暂时未解决的倾向或仍在观察的局部模式。
- `stable_pattern`: 已有证据显示跨时间重复出现的稳定行为。

# 字段写法
- `fact` 写具体可复用事实或已接受的未来行为，不写情绪总结。
- `subjective_appraisal` 写 `{character_name}` 对该 fact 的第三人称理解；客观事实也必须填写。
- `relationship_signal` 写这条记忆以后应怎样影响互动。
- 三个字段共同表达同一条记忆，不是三条独立记忆。
- 对 `active_commitment`，`fact` 写清 `{character_name}` 接受了什么未来行为；`subjective_appraisal` 写她如何理解该约定；`relationship_signal` 写后续如何执行或提醒。
- 对用户自己的项目名、代号或标题，`fact` 写用户拥有的名称；`relationship_signal` 写以后如何识别该名称，不需要反复对比 `{character_name}`。

# 记忆视角契约
- 记忆文本采用第三人称视角。
- 可写入记忆文本的唯一名称是 `{character_name}`。
- 规范名称是一个不可拆分的完整字符串：`{character_name}`。
- 需要命名 `{character_name}` 时，逐字复制完整字符串，包括括号内容、空格和长音符号；不要缩写、截断、翻译、改写或用短名替代。
- 如果不需要消歧，优先省略名称或用“该名称”“这一要求”“这一承诺”等方式回指。
- 如果无法逐字复制完整名称，宁可省略主语，不要写短名或近似拼写。
- 上游证据里指向 `{character_name}` 的短名、别名或旧写法只作为证据理解，不可复制到输出。
- 不要用“我”指代 `{character_name}`；输入中的“我”必须按原说话人理解。
- 如果用户说“我……”，生成记忆时应写作“用户……”“对方……”或“用户自己……”，不要把这个“我”归到 `{character_name}`。
- 不要把说话人标签、显示名称、泛称或 assistant 等机器标签写成记忆主体。
- 当需要说明某个名称、项目代号或称呼不属于 `{character_name}` 时，写作“不是指向 `{character_name}` 的名称/称呼”，不要使用泛称。

# 生成步骤
1. 确认 `timestamp`、用户身份、`{character_name}` 身份和近期消息说话人。
2. 判断本轮是否产生新的长期价值：事实、决定、偏好、承诺、重要转折或可复用互动模式。
3. 对每个候选检查旧记忆上下文，删除重复旧记忆或只是旧记忆复述的候选。
4. 对每个候选决定 `unit_type`；如果同一事项已经被 `{character_name}` 接受为后续行为，优先写成一条 `active_commitment`。
5. 对带时间条件的候选先确定绝对日期或日期时间；无法确定且会影响活跃承诺有效性的候选直接删除。
6. 写 `fact`、`subjective_appraisal`、`relationship_signal`，保持同一条记忆的三个互补面。
# 输入格式
HumanMessage 是以下 JSON：
{{
    "timestamp": "本轮持久化整理的本地时间，YYYY-MM-DD HH:MM",
    "memory_unit_write_contract": {{
        "enabled_lanes": ["..."],
        "allowed_unit_types": ["..."]
    }},
    "global_user_id": "稳定用户 UUID",
    "user_name": "当前用户显示名",
    "consolidation_origin": {{
        "episode_id": "文本标识",
        "trigger_source": "user_message | internal_thought | self_cognition | scheduled_tick | tool_result",
        "input_sources": ["..."],
        "output_mode": "文本标识"
    }},
    "decontextualized_input": "用户本轮消息或内部思考触发文本经去上下文化后的内容",
    "final_dialog": ["{character_name} 本轮最终回复片段或私有预览整理结果"],
    "internal_monologue": "{character_name} 的认知阶段内部独白",
    "emotional_appraisal": "{character_name} 的主观情绪评估",
    "interaction_subtext": "{character_name} 读到的互动潜台词",
    "logical_stance": "CONFIRM | REFUSE | TENTATIVE | DIVERGE | CHALLENGE",
    "character_intent": "本轮意图标签",
    "chat_history_recent": ["[YYYY-MM-DD HH:MM] 用户显示名或 {character_name}: 消息文本"],
    "rag_user_memory_unit_candidates": [
        {{"unit_id": "...", "unit_type": "...", "dedup_key": "...", "fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "可选本地 YYYY-MM-DD HH:MM"}}
    ],
    "new_facts_evidence": [{{"fact": "通道专项处理器输出"}}],
    "future_promises_evidence": [{{"action": "未来承诺或计划行动", "due_time": "可选本地 YYYY-MM-DD HH:MM"}}],
    "subjective_appraisal_evidence": ["关系或主观评估依据文本"]
}}

# 输出格式
字段如下：
{{
    "memory_units": [
        {{
            "unit_type": "stable_pattern | recent_shift | objective_fact | milestone | active_commitment",
            "fact": "具体事件、决定、偏好、承诺或行为",
            "subjective_appraisal": "第三人称主观理解",
            "relationship_signal": "未来互动含义",
            "due_at": "可选；已知到期日或执行时间的 active_commitment 使用本地 YYYY-MM-DD HH:MM",
            "evidence_refs": [{{"source": "chat", "timestamp": "可选本地 YYYY-MM-DD HH:MM", "message_id": "可选平台消息 id"}}]
        }}
    ]
}}
只输出 `memory_unit_write_contract.allowed_unit_types` 中的 `unit_type`；当该列表为空或没有允许的可保存内容时，`memory_units` 必须为空数组。
'''
_llm_interface = LLInterface()
_extractor_llm = LLInterface()
_merge_judge_llm = LLInterface()
_rewrite_llm = LLInterface()
_stability_llm = LLInterface()
_extractor_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CONSOLIDATION_LLM",
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=0.2,
    top_p=0.9,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)


async def extract_memory_unit_candidates(state: ConsolidatorState) -> list[dict]:
    """Extract candidate memory units from one consolidation state.

    Args:
        state: Current consolidator state after dialog.

    Returns:
        Structurally valid candidate memory units.
    """

    write_contract = _memory_unit_write_contract(state)
    allowed_unit_types = set(write_contract["allowed_unit_types"])
    if not allowed_unit_types:
        return_value: list[dict] = []
        return return_value

    character_name = state["character_profile"]["name"]
    system_prompt = SystemMessage(
        content=_EXTRACTOR_PROMPT.format(character_name=character_name),
    )
    payload = project_memory_unit_extractor_prompt_payload(
        _json_payload(state),
        character_name=character_name,
    )
    human_message = HumanMessage(
        content=json.dumps(payload, ensure_ascii=False),
    )
    response = await _extractor_llm.ainvoke([
        system_prompt,
        human_message,
    ], config=_extractor_llm_config)
    result = parse_llm_json_output(response.content)
    default_source_refs = state.get("user_memory_unit_source_refs")
    if not isinstance(default_source_refs, list):
        default_source_refs = []
    candidates = _valid_candidates(
        result,
        default_source_refs=default_source_refs,
        allowed_unit_types=allowed_unit_types,
    )
    return candidates


_MERGE_JUDGE_PROMPT = """\
你判断一个新的记忆单元是否与现有候选记忆单元匹配。

# 角色
你是记忆单元合并判断器，只判断 create、merge 或 evolve。

# 语言政策
- 除结构化枚举值、输出结构字段、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- `decision`、`candidate_id`、`cluster_id` 等结构化字段必须保持输出格式指定的值和原始 ID。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 规则
- create：没有现有候选项记录相同的记忆。
- merge：是同一条可持久化记忆；可以压缩措辞。
- evolve：属于同一记忆簇，但新事件改变了关系含义。
- create 时 `cluster_id` 必须为空。
- merge 或 evolve 时，必须从所提供的候选项逐字复制 `cluster_id`。
- 不要改写记忆文本。

# 生成步骤
1. 阅读 `new_memory_unit.fact`，判断它要保留的具体记忆。
2. 按事件含义比较它与每个 `candidate_clusters` 项，不要只按措辞相似度比较。
3. 如果没有现有单元记录相同的可持久化记忆，选择 create。
4. 如果现有单元已经记录相同记忆，而新候选项主要是重复或补充措辞/细节，选择 merge。
5. 如果现有单元属于同一记忆簇，但新候选项改变了事实的关系含义、适用范围或持久性，选择 evolve。
6. 对于 merge 或 evolve，逐字复制选定 `candidate_clusters` 项的 `cluster_id`。
7. 对于 create，将 `cluster_id` 设为空字符串。
8. 不要臆造 `cluster_id`，不要选择所提供列表之外的簇，也不要改写记忆文本。

# 输入格式
{
    "new_memory_unit": {
        "candidate_id": "候选项标识",
        "unit_type": "stable_pattern | recent_shift | objective_fact | milestone | active_commitment",
        "fact": "候选项的新事实",
        "subjective_appraisal": "候选项的新主观评估",
        "relationship_signal": "候选项的新关系信号",
        "due_at": "已知到期日的 active_commitment 可填写本地 YYYY-MM-DD HH:MM",
        "evidence_refs": [{"source": "chat", "timestamp": "可选本地 YYYY-MM-DD HH:MM 时间戳", "message_id": "可选平台消息标识"}]
    },
    "candidate_clusters": [
        {
            "unit_id": "现有单元标识",
            "unit_type": "现有单元类型",
            "fact": "现有事实",
            "subjective_appraisal": "现有主观评估",
            "relationship_signal": "现有关系信号",
            "updated_at": "可选本地 YYYY-MM-DD HH:MM 时间戳"
        }
    ]
}

# 输出格式
{
    "candidate_id": "从输入复制的候选项标识",
    "decision": "create | merge | evolve",
    "cluster_id": "merge/evolve 使用的现有 unit_id，或 create 使用的空字符串",
    "reason": "简短的语义理由"
}
"""
_merge_judge_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CONSOLIDATION_LLM",
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=0.2,
    top_p=0.9,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)


async def _judge_memory_unit_merge(candidate: dict, candidate_clusters: list[dict]) -> dict:
    """Ask the merge judge whether a candidate creates, merges, or evolves.

    Args:
        candidate: New memory-unit candidate from the extractor.
        candidate_clusters: Existing memory units retrieved by RAG.

    Returns:
        Structurally validated merge-judge decision.
    """

    msg = {
        "new_memory_unit": project_tool_result_for_llm(candidate),
        "candidate_clusters": project_tool_result_for_llm(
            candidate_clusters
        ),
    }
    system_prompt = SystemMessage(content=_MERGE_JUDGE_PROMPT)
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _merge_judge_llm.ainvoke([
        system_prompt,
        human_message,
    ], config=_merge_judge_llm_config)
    result = parse_llm_json_output(response.content)
    merge_result = _validate_merge_result(result, candidate, candidate_clusters)
    return merge_result


_REWRITE_PROMPT = """\
你使用一个新候选项改写一个现有记忆单元。

# 角色
你是记忆单元改写阶段，只更新语义文本字段。

# 语言政策
- 除结构化枚举值、输出结构字段、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 规则
- 只更新三个语义字段。
- 保留具体事件细节。
- 对于 merge，在不丢失事件锚点的前提下压缩重复证据。
- 对于 evolve，明确更新关系含义。
- 不要改变 merge/evolve 决策。
- 如果新候选项包含 `due_at`，在改写后的语义文本中保留绝对到期日期，不要重新引入“明天”之类的相对日期说法。

# 记忆视角契约
- 本契约适用于你生成的可长期保存的 JSON 记忆字段：fact、subjective_appraisal、relationship_signal。
- 记忆文本采用第三人称视角。
- 可写入记忆文本的唯一名称是 `{character_name}`。
- 需要命名 `{character_name}` 时，只使用 `{character_name}`。
- 不要缩写、截断、翻译或改写该名称；不要使用任何别名或短名替代。
- 名称复制规则：需要写 `{character_name}` 时，逐字复制完整字符串，包括括号内容、空格和长音符号；不要凭记忆重新拼写。
- 如果不需要消歧，优先省略名称；如果无法逐字复制完整名称，宁可省略主语，不要写短名或近似拼写。
- 上游证据里指向 `{character_name}` 的短名、别名或旧写法只作为证据理解，不可复制到输出；要么省略主语，要么使用完整名称。
- 不要用“我”指代 `{character_name}`；输入中的“我”必须按原说话人理解。
- 如果用户说“我……”，生成记忆时应写作“用户……”“对方……”或“用户自己……”，不要把这个“我”归到 `{character_name}`。
- 不要把说话人标签、显示名称、泛称或 assistant 等机器标签写成记忆主体；需要命名时只能用 `{character_name}`。
- 当需要说明某个名称、项目代号或称呼不属于 `{character_name}` 时，写作“不是指向 `{character_name}` 的名称/称呼”，不要使用泛称。
- 所有“无关/不是/并非”的对象都必须写成 `{character_name}` 或省略，不允许用泛称代替。

# 生成步骤
1. 先阅读 `decision.decision`，并将其视为固定值。
2. 如果决策是 merge，将现有单元和新候选项中的重复信息压缩成一条更清晰的记忆。
3. 如果决策是 evolve，保留旧记忆，并更新 fact、subjective_appraisal、relationship_signal 以反映新的发展。
4. 保持 `fact` 字段具体且以事件为中心，不要把它变成情绪总结。
5. 将 `subjective_appraisal` 保持为 {character_name} 的第三人称理解，不要强迫每个字段都写出名称。如果字段需要命名 `{character_name}` 或替换受污染的说话人标签，复制完整准确的 `{character_name}`。
6. 让 `relationship_signal` 表达未来互动。
7. 不要输出结构化 ID；持久化 ID 由调用方负责。
8. 只输出更新后的三个语义字段。

# 输入格式
{{
    "existing_unit_id": "合并判断器选定的已存单元标识",
    "new_memory_unit": {{
        "candidate_id": "候选项标识",
        "unit_type": "候选项类型",
        "fact": "候选项的新事实",
        "subjective_appraisal": "候选项的新主观评估",
        "relationship_signal": "候选项的新关系信号",
        "due_at": "已知到期日的 active_commitment 可填写本地 YYYY-MM-DD HH:MM",
        "evidence_refs": [{{"source": "chat", "timestamp": "可选本地 YYYY-MM-DD HH:MM 时间戳", "message_id": "可选平台消息标识"}}]
    }},
    "decision": {{
        "candidate_id": "候选项标识",
        "decision": "merge | evolve",
        "cluster_id": "已存单元标识",
        "reason": "合并判断器给出的理由"
    }}
}}

# 输出格式
{{
    "fact": "更新后的精简事实",
    "subjective_appraisal": "更新后的第三人称主观评估；命名 {character_name} 时使用其完整字符串",
    "relationship_signal": "更新后的未来互动信号"
}}
"""
_rewrite_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CONSOLIDATION_LLM",
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=0.2,
    top_p=0.9,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)


async def _rewrite_memory_unit(
    state: ConsolidatorState,
    candidate: dict,
    merge_result: dict,
) -> dict:
    """Rewrite an existing memory unit with a new candidate's evidence.

    Args:
        state: Current consolidator state, including the character profile.
        candidate: New memory-unit candidate.
        merge_result: Validated merge/evolve decision.

    Returns:
        Validated replacement semantic fields for the stored unit.
    """

    character_name = state["character_profile"]["name"]
    msg = {
        "existing_unit_id": merge_result["cluster_id"],
        "new_memory_unit": project_tool_result_for_llm(candidate),
        "decision": project_tool_result_for_llm(merge_result),
    }
    payload = project_memory_unit_rewrite_prompt_payload(
        msg,
        character_name=character_name,
    )
    system_prompt = SystemMessage(
        content=_REWRITE_PROMPT.format(character_name=character_name),
    )
    human_message = HumanMessage(content=json.dumps(payload, ensure_ascii=False))
    response = await _rewrite_llm.ainvoke([
        system_prompt,
        human_message,
    ], config=_rewrite_llm_config)
    result = parse_llm_json_output(response.content)
    rewrite_result = _validate_rewrite_result(result)
    return rewrite_result


_STABILITY_PROMPT = """\
你判断一条互动模式记忆应保持 recent 还是 stable。

# 角色
你是记忆单元稳定性判断器，只为互动模式单元选择 recent 或 stable。

# 语言政策
- 除结构化枚举值、输出结构字段、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- `window`、`unit_id` 等结构化字段必须保持输出格式指定的英文枚举值和原始 ID。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 规则
- 只把次数、会话分布和新近程度作为证据。
- 不要仅因某个嘈杂会话重复了几次，就提升它的稳定性。
- stable 表示应将其视为持久模式。
- recent 表示它仍是活跃变化或尚未解决的局部模式。

# 生成步骤
1. 决策前先阅读 `stability_evidence`，将 `occurrence_count_label` 和 `session_spread.spread_label` 视为证据说明。
2. 当该记忆看起来跨会话、跨日期或跨多个有意义的重复示例都能持久存在时，选择 stable。
3. 当该记忆是新出现的、仅来自单个会话、尚未解决或近期仍可能改变时，选择 recent。
4. 不要仅因 `occurrence_count` 大于一就选择 stable；检查示例是否确实代表持久模式。
5. 不要仅因事件发生在今天就选择 recent；近期示例也可以确认稳定模式。
6. 从输入逐字复制 `unit_id`，并根据证据提供简短理由。

# 输入格式
{
    "unit_id": "正在分类的已存单元标识",
    "candidate": {
        "candidate_id": "候选项标识",
        "unit_type": "stable_pattern | recent_shift",
        "fact": "候选项事实",
        "subjective_appraisal": "候选项主观评估",
        "relationship_signal": "候选项关系信号",
        "evidence_refs": [{"source": "chat", "timestamp": "可选本地 YYYY-MM-DD HH:MM 时间戳", "message_id": "可选平台消息标识"}]
    },
    "merge_result": {
        "candidate_id": "候选项标识",
        "decision": "create | merge | evolve",
        "cluster_id": "已存单元标识或空字符串",
        "reason": "合并判断器给出的理由"
    },
    "stability_evidence": {
        "occurrence_count": 3,
        "occurrence_count_label": "single_observation | two_observations | several_observations | many_observations",
        "existing_unit_count": 2,
        "new_evidence_ref_count": 1,
        "session_spread": {
            "spread_label": "unknown_session_spread | single_day_or_session | multiple_days_or_sessions",
            "distinct_day_count": 2,
            "distinct_message_ref_count": 3,
            "timestamps": ["YYYY-MM-DD"]
        },
        "recency": {
            "current_turn_timestamp": "本地 YYYY-MM-DD HH:MM 时间戳",
            "existing_updated_at": "可选本地 YYYY-MM-DD HH:MM 时间戳",
            "existing_last_seen_at": "可选本地 YYYY-MM-DD HH:MM 时间戳"
        },
        "recent_examples": [{"source": "existing_unit|new_candidate", "fact": "示例事实", "updated_at": "可选本地 YYYY-MM-DD HH:MM 时间戳"}]
    }
}

# 输出格式
{
    "unit_id": "从输入复制的单元标识",
    "window": "recent | stable",
    "reason": "简短的语义理由"
}
"""
_stability_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CONSOLIDATION_LLM",
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=0.2,
    top_p=0.9,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)


async def _judge_memory_unit_stability(
    state: ConsolidatorState,
    *,
    unit_id: str,
    candidate: dict,
    merge_result: dict,
    candidate_clusters: list[dict],
) -> dict:
    """Ask whether an interaction-pattern unit belongs in recent or stable.

    Args:
        state: Current consolidator state.
        unit_id: Stored memory-unit id being classified.
        candidate: New candidate that created, merged, or evolved the unit.
        merge_result: Validated merge/create/evolve decision.
        candidate_clusters: Existing units shown to the merge judge.

    Returns:
        Validated stability decision.
    """

    msg = _stability_payload(
        state,
        unit_id=unit_id,
        candidate=candidate,
        merge_result=merge_result,
        candidate_clusters=candidate_clusters,
    )
    system_prompt = SystemMessage(content=_STABILITY_PROMPT)
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _stability_llm.ainvoke([
        system_prompt,
        human_message,
    ], config=_stability_llm_config)
    result = parse_llm_json_output(response.content)
    stability_result = _validate_stability_result(result, unit_id)
    return stability_result


async def process_memory_unit_candidate(state: ConsolidatorState, candidate: dict) -> dict:
    """Create, merge, or evolve one memory-unit candidate.

    Args:
        state: Current consolidator state.
        candidate: One structurally valid extracted candidate.

    Returns:
        Write result metadata for logs/tests.
    """

    global_user_id = state["global_user_id"]
    candidate_clusters = await retrieve_memory_unit_merge_candidates(
        global_user_id,
        candidate_unit=candidate,
        surfaced_units=_rag_surfaced_memory_units(state),
        limit=MAX_MEMORY_UNIT_MERGE_CANDIDATES,
    )
    merge_result = await _judge_memory_unit_merge(candidate, candidate_clusters)

    storage_timestamp_utc = state["storage_timestamp_utc"]
    if merge_result["decision"] == "create":
        docs = await insert_user_memory_units(
            global_user_id,
            [candidate],
            storage_timestamp_utc=storage_timestamp_utc,
        )
        created_unit = docs[0]
        unit_id = created_unit["unit_id"]
        if created_unit["unit_type"] == UserMemoryUnitType.ACTIVE_COMMITMENT:
            await reconcile_active_commitment_calendar_schedule(
                created_unit,
                repository=calendar_repository,
                storage_timestamp_utc=storage_timestamp_utc,
            )
    else:
        rewrite_result = await _rewrite_memory_unit(state, candidate, merge_result)
        lifecycle_fields = _candidate_lifecycle_updates(candidate)
        await update_user_memory_unit_semantics(
            merge_result["cluster_id"],
            rewrite_result,
            storage_timestamp_utc=storage_timestamp_utc,
            lifecycle_fields=lifecycle_fields,
            source_refs=_candidate_source_refs(candidate),
            merge_history_entry={
                "timestamp": storage_timestamp_utc,
                "decision": merge_result["decision"],
                "candidate_id": candidate["candidate_id"],
                "reason": merge_result["reason"],
            },
        )
        unit_id = merge_result["cluster_id"]
        if (
            candidate["unit_type"] == UserMemoryUnitType.ACTIVE_COMMITMENT
            and lifecycle_fields
        ):
            updated_unit = {
                "unit_id": unit_id,
                "global_user_id": global_user_id,
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.ACTIVE,
            }
            for cluster in candidate_clusters:
                if text_or_empty(cluster.get("unit_id")) == unit_id:
                    updated_unit.update(cluster)
                    break
            updated_unit.update(rewrite_result)
            updated_unit.update(lifecycle_fields)
            updated_unit["unit_id"] = unit_id
            updated_unit["global_user_id"] = global_user_id
            updated_unit["unit_type"] = UserMemoryUnitType.ACTIVE_COMMITMENT
            updated_unit["updated_at"] = storage_timestamp_utc
            await reconcile_active_commitment_calendar_schedule(
                updated_unit,
                repository=calendar_repository,
                storage_timestamp_utc=storage_timestamp_utc,
            )

    if candidate["unit_type"] in {
        UserMemoryUnitType.STABLE_PATTERN,
        UserMemoryUnitType.RECENT_SHIFT,
    }:
        stability_result = await _judge_memory_unit_stability(
            state,
            unit_id=unit_id,
            candidate=candidate,
            merge_result=merge_result,
            candidate_clusters=candidate_clusters,
        )
        await update_user_memory_unit_window(
            unit_id,
            window=stability_result["window"],
            storage_timestamp_utc=storage_timestamp_utc,
        )
    else:
        stability_result = {}

    return_value = {
        "candidate_id": candidate["candidate_id"],
        "unit_id": unit_id,
        "decision": merge_result["decision"],
        "stability": stability_result,
    }
    return return_value


async def update_user_memory_units_from_state(state: ConsolidatorState) -> list[dict]:
    """Run the split memory-unit consolidation pipeline for one turn.

    Args:
        state: Current consolidator state after reflection and fact harvesting.

    Returns:
        Per-candidate write results.
    """

    if not text_or_empty(state["global_user_id"]):
        return_value = []
        return return_value

    try:
        candidates = await extract_memory_unit_candidates(state)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.exception(f"memory-unit extractor output dropped: {exc}")
        return_value = []
        return return_value

    results = []
    for candidate in candidates:
        try:
            result = await process_memory_unit_candidate(state, candidate)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            candidate_id = text_or_empty(candidate.get("candidate_id"))
            logger.exception(f"memory-unit candidate dropped: {candidate_id}: {exc}")
            continue
        results.append(result)
    return results
