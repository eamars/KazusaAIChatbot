"""Consolidator memory-unit extraction and merge helpers."""

from __future__ import annotations

import json
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
    validate_user_memory_unit_semantics,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_schema import ConsolidatorState
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import retrieve_memory_unit_merge_candidates
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty


MAX_MEMORY_UNIT_CANDIDATES_PER_TURN = 3
MAX_MEMORY_UNIT_MERGE_CANDIDATES = 6

_memory_unit_llm = get_llm(
    temperature=0.2,
    top_p=0.9,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)

_EXTRACTOR_PROMPT = """You extract durable user memory units for Kazusa.

Return strict JSON with:
{"memory_units":[{"unit_type":"recent_shift|objective_fact|milestone|active_commitment","fact":"...","subjective_appraisal":"...","relationship_signal":"...","evidence_refs":[]}]}

Rules:
- fact must be a concrete event, decision, preference, commitment, or durable behavior anchored in the provided conversation.
- Use chat_history_recent as evidence when the memory depends on multiple messages.
- subjective_appraisal explains Kazusa's subjective interpretation of that fact.
- relationship_signal explains how this should affect future interaction.
- Do not output vague labels or only describe the latest message tone.
- Preserve enough event detail to be useful later.
- Do not decide merge/create/evolve.
- If nothing durable should be remembered, return {"memory_units":[]}.
"""

_MERGE_JUDGE_PROMPT = """You judge whether one new memory unit matches existing candidate units.

Return strict JSON with:
{"candidate_id":"...","decision":"create|merge|evolve","cluster_id":"existing unit_id or empty","reason":"short reason"}

Rules:
- create: no existing candidate captures the same memory.
- merge: same durable memory; wording can be compacted.
- evolve: same memory cluster, but the new event changes the relationship meaning.
- cluster_id must be empty for create.
- cluster_id must be copied exactly from the provided candidates for merge/evolve.
- Do not rewrite memory text.
"""

_REWRITE_PROMPT = """You rewrite one existing memory unit using one new candidate.

Return strict JSON with:
{"candidate_id":"...","cluster_id":"...","fact":"...","subjective_appraisal":"...","relationship_signal":"..."}

Rules:
- Update only the three semantic fields.
- Preserve concrete event detail.
- For merge, compact repeated evidence without losing the event anchor.
- For evolve, explicitly update the relationship meaning.
- Do not change the merge/evolve decision.
"""

_STABILITY_PROMPT = """You decide whether an interaction-pattern memory remains recent or is stable.

Return strict JSON with:
{"unit_id":"...","window":"recent|stable","reason":"short reason"}

Rules:
- Use count, session spread, and recency only as evidence.
- Do not promote a single noisy session just because it repeated several times.
- stable means this should be treated as a durable pattern.
- recent means this is still an active shift or unresolved local pattern.
"""


def _json_payload(state: ConsolidatorState) -> dict:
    rag_result = state["rag_result"]
    user_image = rag_result["user_image"]
    rag_user_memory_context = user_image["user_memory_context"]

    return_value = {
        "timestamp": state["timestamp"],
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "decontextualized_input": state["decontexualized_input"],
        "final_dialog": state["final_dialog"],
        "internal_monologue": state["internal_monologue"],
        "emotional_appraisal": state["emotional_appraisal"],
        "interaction_subtext": state["interaction_subtext"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "chat_history_recent": state["chat_history_recent"],
        "rag_user_memory_context": rag_user_memory_context,
        "new_facts_evidence": state["new_facts"],
        "future_promises_evidence": state["future_promises"],
        "subjective_appraisal_evidence": state["subjective_appraisals"],
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


async def _invoke_json(system_prompt: str, payload: dict) -> dict:
    response = await _memory_unit_llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ])
    parsed_response = parse_llm_json_output(response.content)
    return parsed_response


def _candidate_with_id(candidate: dict) -> dict:
    item = dict(candidate)
    item["candidate_id"] = text_or_empty(item.get("candidate_id")) or uuid4().hex
    if not isinstance(item.get("evidence_refs"), list):
        item["evidence_refs"] = []
    return item


def _valid_candidates(result: dict) -> list[dict]:
    raw_candidates = result.get("memory_units", [])
    if not isinstance(raw_candidates, list):
        return_value = []
        return return_value

    candidates: list[dict] = []
    for raw_candidate in raw_candidates[:MAX_MEMORY_UNIT_CANDIDATES_PER_TURN]:
        if not isinstance(raw_candidate, dict):
            continue
        candidate = _candidate_with_id(raw_candidate)
        validate_user_memory_unit_semantics(candidate)
        candidates.append(candidate)
    return candidates


def _validate_merge_result(result: dict, candidate: dict, candidate_clusters: list[dict]) -> dict:
    candidate_id = text_or_empty(result.get("candidate_id"))
    decision = text_or_empty(result.get("decision"))
    cluster_id = text_or_empty(result.get("cluster_id"))
    valid_cluster_ids = {
        text_or_empty(cluster.get("unit_id"))
        for cluster in candidate_clusters
        if text_or_empty(cluster.get("unit_id"))
    }

    if candidate_id != candidate["candidate_id"]:
        raise ValueError("merge judge returned an unknown candidate_id")
    if decision not in {"create", "merge", "evolve"}:
        raise ValueError(f"invalid merge decision: {decision!r}")
    if decision == "create" and cluster_id:
        raise ValueError("create decision must not include cluster_id")
    if decision in {"merge", "evolve"} and cluster_id not in valid_cluster_ids:
        raise ValueError("merge/evolve decision returned an unknown cluster_id")

    return_value = {
        "candidate_id": candidate_id,
        "decision": decision,
        "cluster_id": cluster_id,
        "reason": text_or_empty(result.get("reason")),
    }
    return return_value


def _validate_rewrite_result(result: dict, candidate: dict, cluster_id: str) -> dict:
    if text_or_empty(result.get("candidate_id")) != candidate["candidate_id"]:
        raise ValueError("rewrite returned an unknown candidate_id")
    if text_or_empty(result.get("cluster_id")) != cluster_id:
        raise ValueError("rewrite returned an unknown cluster_id")
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


async def extract_memory_unit_candidates(state: ConsolidatorState) -> list[dict]:
    """Extract candidate memory units from one consolidation state.

    Args:
        state: Current consolidator state after dialog.

    Returns:
        Structurally valid candidate memory units.
    """

    result = await _invoke_json(_EXTRACTOR_PROMPT, _json_payload(state))
    candidates = _valid_candidates(result)
    return candidates


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
    merge_result = _validate_merge_result(
        await _invoke_json(
            _MERGE_JUDGE_PROMPT,
            {"new_memory_unit": candidate, "candidate_clusters": candidate_clusters},
        ),
        candidate,
        candidate_clusters,
    )

    timestamp = state["timestamp"]
    if merge_result["decision"] == "create":
        docs = await insert_user_memory_units(
            global_user_id,
            [candidate],
            timestamp=timestamp,
        )
        unit_id = docs[0]["unit_id"]
    else:
        rewrite_result = _validate_rewrite_result(
            await _invoke_json(
                _REWRITE_PROMPT,
                {
                    "existing_unit_id": merge_result["cluster_id"],
                    "new_memory_unit": candidate,
                    "decision": merge_result,
                },
            ),
            candidate,
            merge_result["cluster_id"],
        )
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
        stability_result = _validate_stability_result(
            await _invoke_json(
                _STABILITY_PROMPT,
                {
                    "unit_id": unit_id,
                    "candidate": candidate,
                    "merge_result": merge_result,
                },
            ),
            unit_id,
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

    candidates = await extract_memory_unit_candidates(state)
    results = []
    for candidate in candidates:
        results.append(await process_memory_unit_candidate(state, candidate))
    return results
