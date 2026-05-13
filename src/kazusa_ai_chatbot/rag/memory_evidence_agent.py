"""Top-level RAG capability agent for durable memory evidence."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    RAG_MEMORY_EVIDENCE_TEXT_LIMIT,
    RAG_SEARCH_SELECTED_SUMMARY_LIMIT,
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.persistent_memory_search_agent import (
    PersistentMemorySearchAgent,
)
from kazusa_ai_chatbot.rag.prompt_projection import project_selector_input_for_llm
from kazusa_ai_chatbot.rag.user_memory_evidence_agent import UserMemoryEvidenceAgent
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_CAPABILITY_NAME = "memory_evidence"
_AGENT_NAME = "memory_evidence_agent"
_UNCACHED_REASON = "capability_orchestrator_uncached"
_KNOWN_WORKERS = {
    "persistent_memory_search_agent",
    "user_memory_evidence_agent",
    "incompatible",
}


def _clip_text(value: object, *, limit: int = 1000) -> str:
    """Return compact prompt-facing text."""
    text = text_or_empty(value)
    if len(text) <= limit:
        return text
    clipped = text[: limit - 1].rstrip()
    return_value = f"{clipped}…"
    return return_value


def _cache_status() -> dict[str, Any]:
    """Build no-cache metadata for top-level capability orchestration."""
    status = {
        "enabled": False,
        "hit": False,
        "cache_name": "",
        "reason": _UNCACHED_REASON,
    }
    return status


def _result_payload(
    *,
    selected_summary: object = "",
    primary_worker: str = "",
    source_policy: object = "",
    resolved_refs: list[dict[str, Any]] | None = None,
    projection_payload: dict[str, Any] | None = None,
    worker_payloads: dict[str, Any] | None = None,
    evidence: list[str] | None = None,
    missing_context: list[str] | None = None,
    conflicts: list[str] | None = None,
    observation_candidates: list[dict[str, Any]] | None = None,
    source_hints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the standard top-level memory capability payload."""
    payload = {
        "selected_summary": _clip_text(selected_summary),
        "capability": _CAPABILITY_NAME,
        "primary_worker": primary_worker,
        "supporting_workers": [],
        "source_policy": _clip_text(source_policy, limit=400),
        "resolved_refs": list(resolved_refs or []),
        "projection_payload": dict(projection_payload or {}),
        "worker_payloads": dict(worker_payloads or {}),
        "evidence": list(evidence or []),
        "missing_context": list(missing_context or []),
        "conflicts": list(conflicts or []),
        "observation_candidates": list(observation_candidates or []),
        "source_hints": list(source_hints or []),
    }
    return payload


def _agent_result(*, resolved: bool, payload: dict[str, Any]) -> dict[str, Any]:
    """Build the outer helper result for one capability-agent execution."""
    result = {
        "resolved": resolved,
        "result": payload,
        "attempts": 1,
        "cache": _cache_status(),
    }
    return result


def _strip_prefix(task: str) -> str:
    """Remove the semantic capability prefix when present."""
    if ":" not in task:
        return task.strip()
    _, _, remainder = task.partition(":")
    return_value = remainder.strip()
    return return_value


def _deterministic_plan(
    task: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Parse structured memory-evidence slots without selector LLM."""
    task_body = _strip_prefix(task)
    normalized = task_body.lower()

    if "active agreement" in normalized or "active promise" in normalized:
        plan = {
            "worker": "incompatible",
            "reason": "Recall",
        }
        return plan

    live_markers = (
        "current weather",
        "current temperature",
        "opening status",
        "current opening",
        "exchange rate",
        "current price",
    )
    if any(marker in normalized for marker in live_markers):
        plan = {
            "worker": "incompatible",
            "reason": "Live-context",
        }
        return plan

    person_markers = (
        "profile",
        "impression",
        "relationship",
        "compatibility",
    )
    if any(marker in normalized for marker in person_markers):
        plan = {
            "worker": "incompatible",
            "reason": "Person-context",
        }
        return plan

    slot_parts = [task_body]
    original_query_text = ""
    if isinstance(context, dict):
        current_slot = text_or_empty(context.get("current_slot"))
        if current_slot:
            slot_parts.append(current_slot)
        original_query_text = text_or_empty(context.get("original_query")).lower()
    slot_text = "\n".join(slot_parts).lower()

    scoped_user_scope_markers = (
        "current user's",
        "current user",
        "with the current user",
        "remember me",
        "recognize me",
        "当前用户",
        "私有",
        "记得我",
        "还记得我",
        "认识我",
    )
    context_scoped_user_scope_markers = (
        "private",
        "user-specific",
        "scoped",
        "with the current user",
        "私有",
    )
    scoped_user_topic_markers = (
        "continuity",
        "accepted preference",
        "shared experience",
        "past interaction",
        "past interactions",
        "prior interaction",
        "prior interactions",
        "interaction history",
        "prior shared interaction",
        "prior shared interactions",
        "shared history",
        "remember the current user",
        "recognize the current user",
        "remember me",
        "user memory evidence",
        "story lore",
        "story continuity",
        "private lore",
        "private continuity",
        "setting",
        "连续性",
        "设定",
        "过往互动",
        "历史互动",
        "之前的互动",
        "以前的互动",
        "共同经历",
        "记得我",
        "还记得我",
        "认识我",
        "记得当前用户",
        "认识当前用户",
    )
    # Query-level context can confirm private scope, but each slot must carry
    # its own scoped-user topic so mixed queries keep independent memory paths.
    has_slot_scoped_user_scope = any(
        marker in slot_text for marker in scoped_user_scope_markers
    )
    has_context_scoped_user_scope = any(
        marker in original_query_text
        for marker in context_scoped_user_scope_markers
    )
    has_scoped_user_scope = (
        has_slot_scoped_user_scope or has_context_scoped_user_scope
    )
    has_scoped_user_topic = any(
        marker in slot_text for marker in scoped_user_topic_markers
    )
    if has_scoped_user_scope and has_scoped_user_topic:
        plan = {
            "worker": "user_memory_evidence_agent",
            "reason": "scoped current-user continuity evidence",
        }
        return plan

    exact_markers = (
        "named fact",
        "proper noun",
        "memory_name",
        "dedup_key",
        "tag",
        "exact",
        '"',
    )
    if any(marker in normalized for marker in exact_markers):
        plan = {
            "worker": "persistent_memory_search_agent",
            "reason": "hybrid durable named fact or exact memory evidence",
        }
        return plan

    plan = {
        "worker": "persistent_memory_search_agent",
        "reason": "semantic durable memory evidence",
    }
    return plan


def _normalize_selector_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize an LLM selector payload to approved fields."""
    worker = text_or_empty(raw_plan.get("worker"))
    if worker == "persistent_memory_keyword_agent":
        worker = "persistent_memory_search_agent"
    if worker not in _KNOWN_WORKERS:
        worker = "persistent_memory_search_agent"
    reason = text_or_empty(raw_plan.get("reason"))
    plan = {
        "worker": worker,
        "reason": reason,
    }
    return plan


_SELECTOR_PROMPT = """\
You choose one bounded persistent-memory worker for a durable evidence slot.
Do not answer active agreements, person profiles, relationships, or live external facts.

# Generation Procedure
1. If the task asks for active agreements, promises, or current episode state,
   output worker="incompatible" and reason="Recall".
2. If the task asks for person profile, impression, compatibility, or
   relationship context, output worker="incompatible" and reason="Person-context".
3. If the task asks for current weather, temperature, opening status, prices,
   exchange rates, or any changing live value, output worker="incompatible"
   and reason="Live-context".
4. Use user_memory_evidence_agent for current-user durable memory, private
   continuity, accepted preference, user-specific lore, current-user
   recognition, prior interaction history, or prior shared experience with the
   current user.
5. Use persistent_memory_search_agent for natural-language durable facts,
   exact named facts, tags, memory_name/dedup_key lookups, proper nouns,
   quoted terms, home/address/location questions, fuzzy concepts, common
   sense, world knowledge, and character-world facts. The search worker
   performs hybrid semantic plus literal-anchor retrieval.

# Input Format
{
  "task": "Memory-evidence slot text",
  "original_query": "decontextualized user query when available",
  "current_slot": "slot label",
  "known_facts": "ordered facts from previous RAG2 slots"
}

# Output Format
Return valid JSON only:
{
  "worker": "user_memory_evidence_agent | persistent_memory_search_agent | incompatible",
  "reason": "short source selection explanation"
}
"""
_selector_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


async def _select_plan(task: str, context: dict[str, Any]) -> dict[str, Any]:
    """Select the bounded durable-memory worker path for one slot."""
    deterministic_plan = _deterministic_plan(task, context)
    if deterministic_plan is not None:
        return deterministic_plan

    llm_input = project_selector_input_for_llm(task, context)
    system_prompt = SystemMessage(content=_SELECTOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(llm_input, ensure_ascii=False, default=str)
    )
    response = await _selector_llm.ainvoke([system_prompt, human_message])
    raw_plan = parse_llm_json_output(response.content)
    if not isinstance(raw_plan, dict):
        raw_plan = {}
    plan = _normalize_selector_plan(raw_plan)
    return plan


def _rows_from_worker(worker_result: dict[str, Any]) -> list[object]:
    """Normalize a memory worker result into rows."""
    value = worker_result.get("result")
    if isinstance(value, dict):
        memory_rows = value.get("memory_rows")
        if isinstance(memory_rows, list):
            return memory_rows
    if isinstance(value, list):
        return value
    rows = [value]
    return rows


def _memory_rows(worker_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve raw memory rows while guaranteeing a content field."""
    rows: list[dict[str, Any]] = []
    for row in _rows_from_worker(worker_result):
        if isinstance(row, dict):
            rows.append(dict(row))
            continue
        text = text_or_empty(row)
        if text:
            rows.append({"content": text})
    return rows


def _content_from_row(row: dict[str, Any]) -> str:
    """Extract the durable memory content from one row."""
    for field in ("content", "description", "text", "summary", "fact"):
        text = text_or_empty(row.get(field))
        if text:
            return text
    return_value = ""
    return return_value


def _summaries_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Build prompt-facing memory evidence summaries."""
    summaries = [
        summary
        for row in rows
        if (
            summary := _clip_text(
                _content_from_row(row),
                limit=RAG_MEMORY_EVIDENCE_TEXT_LIMIT,
            )
        )
    ]
    return summaries


def _refs_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build structured memory references for downstream slots."""
    refs: list[dict[str, Any]] = []
    for row in rows:
        memory_name = text_or_empty(row.get("memory_name"))
        source_kind = text_or_empty(row.get("source_kind"))
        if memory_name or source_kind:
            refs.append(
                {
                    "ref_type": "memory",
                    "memory_name": memory_name,
                    "source_kind": source_kind,
                }
            )
    return refs


def _memory_observation_candidates(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build source-scoped observation rows for unresolved memory retrieval."""

    candidates: list[dict[str, Any]] = []
    for row in rows[:RAG_SEARCH_SELECTED_SUMMARY_LIMIT]:
        content = _clip_text(
            _content_from_row(row),
            limit=RAG_MEMORY_EVIDENCE_TEXT_LIMIT,
        )
        if not content:
            continue
        candidates.append(
            {
                "content": content,
                "source": _memory_row_source(row),
            }
        )
    return candidates


def _memory_row_source(row: dict[str, Any]) -> str:
    """Build a compact source label for one memory row."""

    memory_name = text_or_empty(row.get("memory_name"))
    if memory_name:
        source = f"memory:memory_name:{memory_name}"
        return source

    source_kind = text_or_empty(row.get("source_kind"))
    if source_kind:
        source = f"memory:source_kind:{source_kind}"
        return source

    mongo_id = text_or_empty(row.get("_id"))
    if mongo_id:
        source = f"memory:id:{mongo_id}"
        return source

    source = "memory:unknown"
    return source


class MemoryEvidenceAgent(BaseRAGHelperAgent):
    """Top-level RAG helper for durable memory and world evidence."""

    def __init__(self, *, cache_runtime=None) -> None:
        """Create the uncached memory-evidence capability agent."""
        super().__init__(
            name=_AGENT_NAME,
            cache_name="",
            cache_runtime=cache_runtime,
        )
        self.search_agent = PersistentMemorySearchAgent(
            cache_runtime=cache_runtime
        )
        self.user_memory_agent = UserMemoryEvidenceAgent(
            cache_runtime=cache_runtime
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """Resolve one durable memory evidence slot through one worker path."""

        del max_attempts

        plan = await _select_plan(task, context)
        primary_worker = plan["worker"]
        if primary_worker == "incompatible":
            reason = text_or_empty(plan["reason"]) or "unsupported"
            result = self._unresolved(
                source_policy=f"incompatible intent; use {reason}",
                missing_context=[f"incompatible_intent:{reason}"],
                primary_worker="",
                worker_payloads={},
            )
            return result

        worker = self._worker_for_name(primary_worker)
        worker_result = await worker.run(task, context, max_attempts=1)
        worker_payloads = {primary_worker: worker_result}
        memory_rows = _memory_rows(worker_result)
        summaries = _summaries_from_rows(memory_rows)
        selected_summary = "\n".join(summaries[:RAG_SEARCH_SELECTED_SUMMARY_LIMIT])
        resolved_refs = _refs_from_rows(memory_rows)
        resolved = bool(worker_result.get("resolved")) and bool(summaries)
        missing_context = [] if resolved else ["memory_evidence"]
        projection_rows = memory_rows
        evidence = summaries
        observation_candidates: list[dict[str, Any]] = []
        source_hints: list[dict[str, Any]] = []
        if not resolved:
            selected_summary = ""
            resolved_refs = []
            projection_rows = []
            evidence = []
            observation_candidates = _memory_observation_candidates(memory_rows)
            source_hints = [
                {
                    "kind": "memory",
                    "source": candidate["source"],
                }
                for candidate in observation_candidates
                if candidate.get("source")
            ]
        payload = _result_payload(
            selected_summary=selected_summary,
            primary_worker=primary_worker,
            source_policy=text_or_empty(plan["reason"]),
            resolved_refs=resolved_refs,
            projection_payload={"memory_rows": projection_rows},
            worker_payloads=worker_payloads,
            evidence=evidence,
            missing_context=missing_context,
            conflicts=[],
            observation_candidates=observation_candidates,
            source_hints=source_hints,
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved={resolved} "
            f"primary_worker={primary_worker} "
            f"missing_context={missing_context} "
            f"selected_summary={payload['selected_summary']} "
            f"cache_reason={_UNCACHED_REASON}"
        )
        logger.debug(
            f"{_AGENT_NAME} debug: resolved_refs={resolved_refs} "
            f"projection_payload={payload['projection_payload']} "
            f"worker_payloads={worker_payloads}"
        )
        result = _agent_result(resolved=resolved, payload=payload)
        return result

    def _worker_for_name(self, worker_name: str) -> BaseRAGHelperAgent:
        """Return the configured memory worker for an approved name."""
        if worker_name == "user_memory_evidence_agent":
            return self.user_memory_agent
        return_value = self.search_agent
        return return_value

    def _unresolved(
        self,
        *,
        source_policy: str,
        missing_context: list[str],
        primary_worker: str,
        worker_payloads: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an unresolved result without calling a memory source."""
        payload = _result_payload(
            selected_summary="",
            primary_worker=primary_worker,
            source_policy=source_policy,
            resolved_refs=[],
            projection_payload={"memory_rows": []},
            worker_payloads=worker_payloads,
            evidence=[],
            missing_context=missing_context,
            conflicts=[],
            observation_candidates=[],
            source_hints=[],
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved=False "
            f"primary_worker={primary_worker} "
            f"missing_context={missing_context} selected_summary='' "
            f"cache_reason={_UNCACHED_REASON}"
        )
        logger.debug(
            f"{_AGENT_NAME} debug: resolved_refs=[] "
            f"projection_payload={payload['projection_payload']} "
            f"worker_payloads={worker_payloads}"
        )
        result = _agent_result(resolved=False, payload=payload)
        return result
