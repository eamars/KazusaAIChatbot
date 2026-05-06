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
)
from kazusa_ai_chatbot.db import (
    UserMemoryUnitType,
    insert_user_memory_units,
    update_user_memory_unit_semantics,
    update_user_memory_unit_window,
)
from kazusa_ai_chatbot.memory_writer_prompt_projection import (
    project_memory_unit_extractor_prompt_payload,
    project_memory_unit_rewrite_prompt_payload,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_schema import ConsolidatorState
from kazusa_ai_chatbot.rag.prompt_projection import project_tool_result_for_llm
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import retrieve_memory_unit_merge_candidates
from kazusa_ai_chatbot.time_context import (
    format_history_for_llm,
    format_timestamp_for_llm,
)
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty


MAX_MEMORY_UNIT_CANDIDATES_PER_TURN = 3
MAX_MEMORY_UNIT_MERGE_CANDIDATES = 6

VALID_EXTRACTED_USER_MEMORY_UNIT_TYPES = {
    UserMemoryUnitType.STABLE_PATTERN,
    UserMemoryUnitType.RECENT_SHIFT,
    UserMemoryUnitType.OBJECTIVE_FACT,
    UserMemoryUnitType.MILESTONE,
    UserMemoryUnitType.ACTIVE_COMMITMENT,
}

logger = logging.getLogger(__name__)


def _json_payload(state: ConsolidatorState) -> dict:
    rag_result = state["rag_result"]
    user_image = rag_result["user_image"]
    rag_user_memory_context = user_image["user_memory_context"]
    projected_memory_context = project_tool_result_for_llm(rag_user_memory_context)
    if not isinstance(projected_memory_context, dict):
        projected_memory_context = {}

    local_datetime = state["time_context"]["current_local_datetime"]
    return_value = {
        "timestamp": local_datetime,
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "decontextualized_input": state["decontexualized_input"],
        "final_dialog": state["final_dialog"],
        "internal_monologue": state["internal_monologue"],
        "emotional_appraisal": state["emotional_appraisal"],
        "interaction_subtext": state["interaction_subtext"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "chat_history_recent": format_history_for_llm(state["chat_history_recent"]),
        "rag_user_memory_context": projected_memory_context,
        "new_facts_evidence": project_tool_result_for_llm(
            state["new_facts"]
        ),
        "future_promises_evidence": project_tool_result_for_llm(
            state["future_promises"]
        ),
        "subjective_appraisal_evidence": project_tool_result_for_llm(
            state["subjective_appraisals"]
        ),
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
    surfaced_units = rag_result["user_memory_unit_candidates"]
    if not isinstance(surfaced_units, list):
        return_value = []
        return return_value
    valid_units = [unit for unit in surfaced_units if isinstance(unit, dict)]
    return valid_units


def _candidate_with_id(candidate: dict) -> dict:
    item = dict(candidate)
    item["candidate_id"] = text_or_empty(item.get("candidate_id")) or uuid4().hex
    if not isinstance(item.get("evidence_refs"), list):
        item["evidence_refs"] = []
    return item


def _candidate_validation_errors(candidate: dict) -> list[str]:
    """Return structural errors for an extractor-authored memory unit.

    Args:
        candidate: Candidate memory-unit dictionary after id normalization.

    Returns:
        Validation error strings. An empty list means the candidate is usable.
    """

    errors: list[str] = []
    unit_type = text_or_empty(candidate.get("unit_type"))
    if unit_type not in VALID_EXTRACTED_USER_MEMORY_UNIT_TYPES:
        errors.append(f"invalid unit_type: {unit_type!r}")

    for field in ("fact", "subjective_appraisal", "relationship_signal"):
        if not text_or_empty(candidate.get(field)):
            errors.append(f"missing field: {field}")

    evidence_refs = candidate.get("evidence_refs")
    if not isinstance(evidence_refs, list):
        errors.append("evidence_refs must be a list")

    return errors


def _validated_candidates(result: dict) -> tuple[list[dict], list[dict]]:
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

        candidate = _candidate_with_id(raw_candidate)
        candidate_errors = _candidate_validation_errors(candidate)
        if candidate_errors:
            validation_errors.append({
                "candidate_id": candidate["candidate_id"],
                "errors": candidate_errors,
            })
            continue

        candidates.append(candidate)

    return_value = (candidates, validation_errors)
    return return_value


def _valid_candidates(result: dict) -> list[dict]:
    candidates, validation_errors = _validated_candidates(result)
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
        raise ValueError("merge/evolve decision returned an unknown cluster_id")

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
        formatted_timestamp = format_timestamp_for_llm(raw_ts)
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
            "updated_at": format_timestamp_for_llm(text_or_empty(cluster.get("updated_at"))),
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

    local_datetime = state["time_context"]["current_local_datetime"]
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
                "existing_updated_at": format_timestamp_for_llm(
                    text_or_empty(cluster.get("updated_at"))
                ),
                "existing_last_seen_at": format_timestamp_for_llm(
                    text_or_empty(cluster.get("last_seen_at"))
                ),
            },
            "recent_examples": _recent_examples(candidate, cluster),
        },
    }
    return return_value


_EXTRACTOR_PROMPT = """\
# 任务
你从本轮 consolidation 输入中提取新的、可长期保存的 user memory unit，供 `{character_name}` 以后与该用户互动时使用。
你只提取候选记忆，不判断 create、merge 或 evolve。
如果没有值得长期保存的内容，只返回 {{"memory_units":[]}}。

# 语言政策
- JSON key、结构化枚举值、ID、URL、代码、命令和模型标签保持原样。
- `unit_type`、`evidence_refs.source` 等枚举字段必须保持输出格式指定的英文值。
- 由你新生成的自由文本字段 fact、subjective_appraisal、relationship_signal 必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、外部证据原句在需要精确保留时保持原语言。
- 指向 `{character_name}` 的短名、别名、旧称呼、显示名或 assistant 等机器标签只可用于理解，不可复制到输出字段。
- 不要添加翻译、双语复写或括号解释，除非源文本本身已经包含。

# 输入证据读取顺序
1. 先读 `chat_history_recent`。用 `speaker_name` 判断每条消息是谁说的；`body_text` 中的“我”必须按原说话人理解。
2. 再读 `decontextualized_input`、`final_dialog`、`logical_stance`、`character_intent`，确认本轮发生了什么，以及 `{character_name}` 是否接受了某个未来行为。
3. `internal_monologue`、`emotional_appraisal`、`interaction_subtext`、`subjective_appraisal_evidence` 只用于理解 `{character_name}` 如何看待事实，不可单独当作用户事实。
4. `new_facts_evidence` 和 `future_promises_evidence` 是提示，不是必须照抄的输出。
5. 对照 `rag_user_memory_context`。只有本轮带来新事实、更清楚的细节或新的未来互动含义时，才生成 memory_unit。

# 是否生成记忆
- 只保存具体事件、决定、偏好、承诺、可复用行为模式或重要转折。
- 不保存单纯语气、一次性心情、普通寒暄、重复旧记忆或只描述最新消息态度的内容。
- 一个具体事件只生成一个 memory_unit。
- 用户提出请求并且 `{character_name}` 接受后续遵守时，这是一个 `active_commitment`，不要拆成“用户偏好”和“接受回应”两条。
- 当 `future_promises_evidence` 与用户本轮请求/偏好指向同一个后续行为时，只生成一条 `active_commitment`；不要再为同一请求另建 `objective_fact`。
- 如果证据中有多个可长期保存的主题，只有在它们有不同的未来互动含义时才分成多条。
- 用户明确说明某个项目名、代号、标题或外部名称属于用户自己时，优先直接记录用户事实；只在该对比本身会影响未来互动时，才保留它不是指向 `{character_name}` 的说明。

# unit_type 判定
- `objective_fact`: 用户事实、用户偏好、项目名称、明确决定或系统性说明；如果同一事实已经被 `{character_name}` 接受为后续行为，改用 `active_commitment`。
- `milestone`: 一次性的重要事件、清晰转折或长期关系/协作方式的改变。
- `active_commitment`: `{character_name}` 已接受的持续承诺、后续行为或仍需遵守的偏好。
- `recent_shift`: 新出现的短期变化、暂时未解决的倾向或仍在观察的局部模式。
- `stable_pattern`: 已有证据显示跨时间重复出现的稳定行为。

# 三个字段的写法
- 写每个 memory_unit 前，先决定是否需要写 `{character_name}`。如果上下文清楚，优先省略名称或用“该名称”“这一要求”“这一承诺”等方式回指。
- `fact`: 写具体可复用事实或已接受的未来行为，不写情绪总结。
- `subjective_appraisal`: 写 `{character_name}` 对该 fact 的第三人称理解；客观事实也必须填写。
- `relationship_signal`: 写这条记忆以后应怎样影响互动。
- 对用户自己的项目名、代号或标题，fact 写用户拥有的名称；relationship_signal 写以后如何识别该名称，不需要反复对比 `{character_name}`。
- 三个字段共同表达同一条记忆，不是三条独立记忆。
- 对 `active_commitment`，fact 应写清 `{character_name}` 接受了什么未来行为；subjective_appraisal 写理解，relationship_signal 写后续执行方式。
- 如果 fact 已经写出 `{character_name}`，subjective_appraisal 与 relationship_signal 优先省略主语或写“该名称”“这一要求”“这一承诺”等。

# 记忆视角契约
- 本契约适用于你生成的可长期保存的 JSON 记忆字段：fact、subjective_appraisal、relationship_signal。
- 记忆文本采用第三人称视角。
- 可写入记忆文本的唯一名称是 `{character_name}`。
- 需要命名 `{character_name}` 时，只使用 `{character_name}`。
- 不要缩写、截断、翻译或改写该名称；不要使用任何别名或短名替代。
- 规范名称是一个不可拆分的完整字符串；不要只输出括号前的中文部分，也不要只输出括号内或括号外的任一片段。
- 名称复制规则：需要写 `{character_name}` 时，逐字复制完整字符串，包括括号内容、空格和长音符号；不要凭记忆重新拼写。
- 如果不需要消歧，优先省略名称；如果无法逐字复制完整名称，宁可省略主语，不要写短名或近似拼写。
- 上游证据里指向 `{character_name}` 的短名、别名或旧写法只作为证据理解，不可复制到输出；要么省略主语，要么使用完整名称。
- 不要用“我”指代 `{character_name}`；输入中的“我”必须按原说话人理解。
- 如果用户说“我……”，生成记忆时应写作“用户……”“对方……”或“用户自己……”，不要把这个“我”归到 `{character_name}`。
- 不要把说话人标签、显示名称、泛称或 assistant 等机器标签写成记忆主体；需要命名时只能用 `{character_name}`。
- 只有必须保留对比关系时，才写作“不是指向 `{character_name}` 的名称/称呼”，不要使用泛称。

# 输入格式
{{
    "timestamp": "local YYYY-MM-DD HH:MM timestamp for this consolidation turn",
    "global_user_id": "stable user UUID",
    "user_name": "current user display name",
    "decontextualized_input": "current user message after decontextualization",
    "final_dialog": ["final response segment from {character_name}"],
    "internal_monologue": "cognition-stage internal monologue for {character_name}",
    "emotional_appraisal": "subjective emotional appraisal for {character_name}",
    "interaction_subtext": "interaction subtext read by {character_name}",
    "logical_stance": "CONFIRM | REFUSE | TENTATIVE | DIVERGE | CHALLENGE",
    "character_intent": "PROVIDE | BANTAR | REJECT | EVADE | CONFRONT | DISMISS | CLARIFY",
    "chat_history_recent": [{{"speaker_name": "user display name or {character_name}", "body_text": "message text"}}],
    "rag_user_memory_context": {{
        "stable_patterns": [{{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "optional local YYYY-MM-DD HH:MM timestamp"}}],
        "recent_shifts": [{{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "optional local YYYY-MM-DD HH:MM timestamp"}}],
        "objective_facts": [{{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "optional local YYYY-MM-DD HH:MM timestamp"}}],
        "milestones": [{{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "optional local YYYY-MM-DD HH:MM timestamp"}}],
        "active_commitments": [{{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "optional local YYYY-MM-DD HH:MM timestamp"}}]
    }},
    "new_facts_evidence": [{{"fact": "fact harvester output"}}],
    "future_promises_evidence": [{{"action": "future promise or scheduled action"}}],
    "subjective_appraisal_evidence": ["relationship/appraisal evidence text"]
}}

# 输出格式
只返回有效 JSON：
{{
    "memory_units": [
        {{
            "unit_type": "stable_pattern | recent_shift | objective_fact | milestone | active_commitment",
            "fact": "具体事件、决定、偏好、承诺或行为",
            "subjective_appraisal": "第三人称主观理解",
            "relationship_signal": "未来互动含义",
            "evidence_refs": [{{"source": "chat", "timestamp": "optional local YYYY-MM-DD HH:MM timestamp", "message_id": "optional platform message id"}}]
        }}
    ]
}}
"""
_extractor_llm = get_llm(
    temperature=0.2,
    top_p=0.9,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)


async def extract_memory_unit_candidates(state: ConsolidatorState) -> list[dict]:
    """Extract candidate memory units from one consolidation state.

    Args:
        state: Current consolidator state after dialog.

    Returns:
        Structurally valid candidate memory units.
    """

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
    ])
    result = parse_llm_json_output(response.content)
    candidates = _valid_candidates(result)
    return candidates


_MERGE_JUDGE_PROMPT = """\
You judge whether one new memory unit matches existing candidate units.

# Role
You are the memory-unit merge judge. You only decide create, merge, or evolve.

# 语言政策
- 除结构化枚举值、schema key、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- `decision`、`candidate_id`、`cluster_id` 等结构化字段必须保持输出格式指定的值和原始 ID。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# Rules
- create: no existing candidate captures the same memory.
- merge: same durable memory; wording can be compacted.
- evolve: same memory cluster, but the new event changes the relationship meaning.
- cluster_id must be empty for create.
- cluster_id must be copied exactly from the provided candidates for merge/evolve.
- Do not rewrite memory text.

# Generation Procedure
1. Read new_memory_unit.fact and decide what specific memory it is trying to preserve.
2. Compare it with each candidate_clusters item by event meaning, not by wording similarity alone.
3. Choose create if no existing unit captures the same durable memory.
4. Choose merge if the existing unit already captures the same memory and the new candidate mainly repeats or adds wording/detail.
5. Choose evolve if the existing unit is the same memory cluster but the new candidate changes the fact's relationship meaning, scope, or durability.
6. For merge or evolve, copy cluster_id exactly from the selected candidate_clusters item.
7. For create, set cluster_id to an empty string.
8. Do not invent a cluster_id, do not choose a cluster outside the provided list, and do not rewrite the memory text.

# Input Format
{
    "new_memory_unit": {
        "candidate_id": "candidate id",
        "unit_type": "stable_pattern | recent_shift | objective_fact | milestone | active_commitment",
        "fact": "new candidate fact",
        "subjective_appraisal": "new candidate appraisal",
        "relationship_signal": "new candidate relationship signal",
        "evidence_refs": [{"source": "chat", "timestamp": "optional local YYYY-MM-DD HH:MM timestamp", "message_id": "optional platform message id"}]
    },
    "candidate_clusters": [
        {
            "unit_id": "existing unit id",
            "unit_type": "existing unit type",
            "fact": "existing fact",
            "subjective_appraisal": "existing appraisal",
            "relationship_signal": "existing relationship signal",
            "updated_at": "optional local YYYY-MM-DD HH:MM timestamp"
        }
    ]
}

# Output Format
Return only valid JSON:
{
    "candidate_id": "candidate id copied from input",
    "decision": "create | merge | evolve",
    "cluster_id": "existing unit_id for merge/evolve, or empty string for create",
    "reason": "short semantic reason"
}
"""
_merge_judge_llm = get_llm(
    temperature=0.2,
    top_p=0.9,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
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
    ])
    result = parse_llm_json_output(response.content)
    merge_result = _validate_merge_result(result, candidate, candidate_clusters)
    return merge_result


_REWRITE_PROMPT = """\
You rewrite one existing memory unit using one new candidate.

# Role
You are the memory-unit rewrite stage. You update only the semantic text fields.

# 语言政策
- 除结构化枚举值、schema key、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# Rules
- Update only the three semantic fields.
- Preserve concrete event detail.
- For merge, compact repeated evidence without losing the event anchor.
- For evolve, explicitly update the relationship meaning.
- Do not change the merge/evolve decision.

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
- 只返回有效 JSON。

# Generation Procedure
1. Read decision.decision first. Treat it as fixed.
2. If decision is merge, compact repeated information from the existing unit and new candidate into one clearer memory.
3. If decision is evolve, preserve the older memory and update the fact/appraisal/signal to reflect the new development.
4. Keep the fact field concrete and event-based. Do not turn it into a mood summary.
5. Keep subjective_appraisal as {character_name}'s third-person interpretation. Do not force the name into every field. If the field needs to name `{character_name}` or replace a polluted actor label, copy the full exact name `{character_name}`.
6. Keep relationship_signal about future interaction.
7. Do not output structural IDs; the caller already owns persistence IDs.
8. Output only the three updated semantic fields.

# Input Format
{{
    "existing_unit_id": "stored unit id selected by the merge judge",
    "new_memory_unit": {{
        "candidate_id": "candidate id",
        "unit_type": "candidate unit type",
        "fact": "new candidate fact",
        "subjective_appraisal": "new candidate appraisal",
        "relationship_signal": "new candidate relationship signal",
        "evidence_refs": [{{"source": "chat", "timestamp": "optional local YYYY-MM-DD HH:MM timestamp", "message_id": "optional platform message id"}}]
    }},
    "decision": {{
        "candidate_id": "candidate id",
        "decision": "merge | evolve",
        "cluster_id": "stored unit id",
        "reason": "merge judge reason"
    }}
}}

# Output Format
Return only valid JSON:
{{
    "fact": "updated compact fact",
    "subjective_appraisal": "updated third-person subjective appraisal using the exact {character_name} string when naming {character_name}",
    "relationship_signal": "updated future interaction signal"
}}
"""
_rewrite_llm = get_llm(
    temperature=0.2,
    top_p=0.9,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
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
    ])
    result = parse_llm_json_output(response.content)
    rewrite_result = _validate_rewrite_result(result)
    return rewrite_result


_STABILITY_PROMPT = """\
You decide whether an interaction-pattern memory remains recent or is stable.

# Role
You are the memory-unit stability judge. You only choose recent or stable for interaction-pattern units.

# 语言政策
- 除结构化枚举值、schema key、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- `window`、`unit_id` 等结构化字段必须保持输出格式指定的英文枚举值和原始 ID。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# Rules
- Use count, session spread, and recency only as evidence.
- Do not promote a single noisy session just because it repeated several times.
- stable means this should be treated as a durable pattern.
- recent means this is still an active shift or unresolved local pattern.

# Generation Procedure
1. Read stability_evidence before deciding. Treat occurrence_count_label and session_spread.spread_label as evidence explanations.
2. Choose stable when the memory looks durable across sessions, days, or repeated meaningful examples.
3. Choose recent when the memory is new, single-session, unresolved, or could still change soon.
4. Do not choose stable only because occurrence_count is greater than one; check whether the examples represent a real durable pattern.
5. Do not choose recent only because the event happened today; recent examples can still confirm a stable pattern.
6. Copy unit_id exactly from input and provide a short reason based on the evidence.

# Input Format
{
    "unit_id": "stored unit id being classified",
    "candidate": {
        "candidate_id": "candidate id",
        "unit_type": "stable_pattern | recent_shift",
        "fact": "candidate fact",
        "subjective_appraisal": "candidate appraisal",
        "relationship_signal": "candidate relationship signal",
        "evidence_refs": [{"source": "chat", "timestamp": "optional local YYYY-MM-DD HH:MM timestamp", "message_id": "optional platform message id"}]
    },
    "merge_result": {
        "candidate_id": "candidate id",
        "decision": "create | merge | evolve",
        "cluster_id": "stored unit id or empty string",
        "reason": "merge judge reason"
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
            "current_turn_timestamp": "local YYYY-MM-DD HH:MM timestamp",
            "existing_updated_at": "optional local YYYY-MM-DD HH:MM timestamp",
            "existing_last_seen_at": "optional local YYYY-MM-DD HH:MM timestamp"
        },
        "recent_examples": [{"source": "existing_unit|new_candidate", "fact": "example fact", "updated_at": "optional local YYYY-MM-DD HH:MM timestamp"}]
    }
}

# Output Format
Return only valid JSON:
{
    "unit_id": "unit id copied from input",
    "window": "recent | stable",
    "reason": "short semantic reason"
}
"""
_stability_llm = get_llm(
    temperature=0.2,
    top_p=0.9,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
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
    ])
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

    timestamp = state["timestamp"]
    if merge_result["decision"] == "create":
        docs = await insert_user_memory_units(
            global_user_id,
            [candidate],
            timestamp=timestamp,
        )
        unit_id = docs[0]["unit_id"]
    else:
        rewrite_result = await _rewrite_memory_unit(state, candidate, merge_result)
        await update_user_memory_unit_semantics(
            merge_result["cluster_id"],
            rewrite_result,
            timestamp=timestamp,
            merge_history_entry={
                "timestamp": timestamp,
                "decision": merge_result["decision"],
                "candidate_id": candidate["candidate_id"],
                "reason": merge_result["reason"],
            },
        )
        unit_id = merge_result["cluster_id"]

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
            timestamp=timestamp,
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
