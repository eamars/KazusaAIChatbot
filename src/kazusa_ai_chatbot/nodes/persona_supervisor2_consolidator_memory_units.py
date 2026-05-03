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
You extract durable user memory units for the active character.

# Role
You are the memory-unit extractor. You only identify new candidate memories from this consolidation turn.

# 语言政策
- 除结构化枚举值、schema key、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- `unit_type`、`evidence_refs.source` 等枚举字段必须保持输出格式指定的英文枚举值。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# Rules
- fact must be a concrete event, decision, preference, commitment, or durable behavior anchored in the provided conversation.
- Use chat_history_recent as evidence when the memory depends on multiple messages.
- Every returned memory unit must include non-empty fact, subjective_appraisal, and relationship_signal.
- subjective_appraisal explains the active character's subjective interpretation of that fact; it is required even for objective facts.
- relationship_signal explains how this should affect future interaction; it is required for every unit.
- Do not output vague labels or only describe the latest message tone.
- Preserve enough event detail to be useful later.
- Do not decide merge/create/evolve.
- If nothing durable should be remembered, return {"memory_units":[]}.

# Generation Procedure
1. Read chat_history_recent first and identify concrete events, decisions, preferences, commitments, or repeated behaviors.
2. Use decontextualized_input, final_dialog, internal_monologue, and appraisal evidence only to clarify what happened and why it matters.
3. Compare against rag_user_memory_context. Do not restate an existing memory unless this turn adds a new fact, sharper detail, or changed relationship meaning.
4. For each candidate, choose exactly one unit_type:
   - objective_fact: a factual user preference, identity fact, decision, or system instruction.
   - milestone: a one-time important event or clear architecture/relationship turning point.
   - active_commitment: an accepted ongoing promise, future action, or still-active preference to honor.
   - recent_shift: a new local change or unresolved short-term pattern.
   - stable_pattern: a durable repeated behavior only when the evidence already shows repetition across time.
5. Write fact as the concrete event itself, not the active character's feeling about it.
6. Write subjective_appraisal as the active character's interpretation of that fact.
7. Write relationship_signal as the future interaction implication.
8. Prefer one or two high-value memory units over many vague ones.
9. If the only possible output is a mood label, tone label, or duplicate of existing memory, return an empty list.

# Input Format
{
    "timestamp": "local YYYY-MM-DD HH:MM timestamp for the current consolidation turn",
    "global_user_id": "stable user UUID",
    "user_name": "current user's display name",
    "decontextualized_input": "current user message after decontextualization",
    "final_dialog": ["active character's final response segment"],
    "internal_monologue": "active character's cognition-stage internal monologue",
    "emotional_appraisal": "active character's subjective emotional appraisal",
    "interaction_subtext": "active character's reading of the interaction subtext",
    "logical_stance": "CONFIRM | REFUSE | TENTATIVE | DIVERGE | CHALLENGE",
    "character_intent": "PROVIDE | BANTAR | REJECT | EVADE | CONFRONT | DISMISS | CLARIFY",
    "chat_history_recent": [{"role": "user|assistant", "display_name": "optional", "body_text": "message text"}],
    "rag_user_memory_context": {
        "stable_patterns": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "optional local YYYY-MM-DD HH:MM timestamp"}],
        "recent_shifts": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "optional local YYYY-MM-DD HH:MM timestamp"}],
        "objective_facts": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "optional local YYYY-MM-DD HH:MM timestamp"}],
        "milestones": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "optional local YYYY-MM-DD HH:MM timestamp"}],
        "active_commitments": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "optional local YYYY-MM-DD HH:MM timestamp"}]
    },
    "new_facts_evidence": [{"fact": "fact harvester output"}],
    "future_promises_evidence": [{"action": "future promise or scheduled action"}],
    "subjective_appraisal_evidence": ["relationship/appraisal evidence text"]
}

# Output Format
Return only valid JSON:
{
    "memory_units": [
        {
            "unit_type": "stable_pattern | recent_shift | objective_fact | milestone | active_commitment",
            "fact": "concrete event, decision, preference, commitment, or behavior",
            "subjective_appraisal": "active character's subjective interpretation of the fact",
            "relationship_signal": "how this should affect future interaction",
            "evidence_refs": [{"source": "chat", "timestamp": "optional local YYYY-MM-DD HH:MM timestamp", "message_id": "optional platform message id"}]
        }
    ]
}
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

    system_prompt = SystemMessage(content=_EXTRACTOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(_json_payload(state), ensure_ascii=False),
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

# Generation Procedure
1. Read decision.decision first. Treat it as fixed.
2. If decision is merge, compact repeated information from the existing unit and new candidate into one clearer memory.
3. If decision is evolve, preserve the older memory and update the fact/appraisal/signal to reflect the new development.
4. Keep the fact field concrete and event-based. Do not turn it into a mood summary.
5. Keep subjective_appraisal about the active character's interpretation.
6. Keep relationship_signal about future interaction.
7. Do not output structural IDs; the caller already owns persistence IDs.
8. Output only the three updated semantic fields.

# Input Format
{
    "existing_unit_id": "stored unit id selected by the merge judge",
    "new_memory_unit": {
        "candidate_id": "candidate id",
        "unit_type": "candidate unit type",
        "fact": "new candidate fact",
        "subjective_appraisal": "new candidate appraisal",
        "relationship_signal": "new candidate relationship signal",
        "evidence_refs": [{"source": "chat", "timestamp": "optional local YYYY-MM-DD HH:MM timestamp", "message_id": "optional platform message id"}]
    },
    "decision": {
        "candidate_id": "candidate id",
        "decision": "merge | evolve",
        "cluster_id": "stored unit id",
        "reason": "merge judge reason"
    }
}

# Output Format
Return only valid JSON:
{
    "fact": "updated compact fact",
    "subjective_appraisal": "updated active-character subjective appraisal",
    "relationship_signal": "updated future interaction signal"
}
"""
_rewrite_llm = get_llm(
    temperature=0.2,
    top_p=0.9,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)


async def _rewrite_memory_unit(candidate: dict, merge_result: dict) -> dict:
    """Rewrite an existing memory unit with a new candidate's evidence.

    Args:
        candidate: New memory-unit candidate.
        merge_result: Validated merge/evolve decision.

    Returns:
        Validated replacement semantic fields for the stored unit.
    """

    msg = {
        "existing_unit_id": merge_result["cluster_id"],
        "new_memory_unit": project_tool_result_for_llm(candidate),
        "decision": project_tool_result_for_llm(merge_result),
    }
    system_prompt = SystemMessage(content=_REWRITE_PROMPT)
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
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
        rewrite_result = await _rewrite_memory_unit(candidate, merge_result)
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
