"""Top-level RAG capability agent for durable memory evidence."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.persistent_memory_keyword_agent import (
    PersistentMemoryKeywordAgent,
)
from kazusa_ai_chatbot.rag.persistent_memory_search_agent import (
    PersistentMemorySearchAgent,
)
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_CAPABILITY_NAME = "memory_evidence"
_AGENT_NAME = "memory_evidence_agent"
_UNCACHED_REASON = "capability_orchestrator_uncached"
_KNOWN_WORKERS = {
    "persistent_memory_keyword_agent",
    "persistent_memory_search_agent",
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


def _deterministic_plan(task: str) -> dict[str, Any] | None:
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
            "worker": "persistent_memory_keyword_agent",
            "reason": "durable named fact or exact memory evidence",
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
4. Use persistent_memory_keyword_agent for exact named facts, tags,
   memory_name/dedup_key lookups, proper nouns, or quoted terms.
5. Use persistent_memory_search_agent for natural-language durable facts,
   home/address/location questions, fuzzy concepts, common sense, world
   knowledge, character-world facts, and durable user memory evidence.

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
  "worker": "persistent_memory_keyword_agent | persistent_memory_search_agent | incompatible",
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
    deterministic_plan = _deterministic_plan(task)
    if deterministic_plan is not None:
        return deterministic_plan

    user_input = {
        "task": task,
        "original_query": context.get("original_query"),
        "current_slot": context.get("current_slot"),
        "known_facts": context.get("known_facts"),
    }
    system_prompt = SystemMessage(content=_SELECTOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(user_input, ensure_ascii=False, default=str)
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
        if (summary := _clip_text(_content_from_row(row), limit=500))
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


class MemoryEvidenceAgent(BaseRAGHelperAgent):
    """Top-level RAG helper for durable memory and world evidence."""

    def __init__(self, *, cache_runtime=None) -> None:
        """Create the uncached memory-evidence capability agent."""
        super().__init__(
            name=_AGENT_NAME,
            cache_name="",
            cache_runtime=cache_runtime,
        )
        self.keyword_agent = PersistentMemoryKeywordAgent(
            cache_runtime=cache_runtime
        )
        self.search_agent = PersistentMemorySearchAgent(
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
        selected_summary = "\n".join(summaries[:5])
        resolved_refs = _refs_from_rows(memory_rows)
        resolved = bool(worker_result.get("resolved")) and bool(summaries)
        missing_context = [] if resolved else ["memory_evidence"]
        payload = _result_payload(
            selected_summary=selected_summary,
            primary_worker=primary_worker,
            source_policy=text_or_empty(plan["reason"]),
            resolved_refs=resolved_refs,
            projection_payload={"memory_rows": memory_rows},
            worker_payloads=worker_payloads,
            evidence=summaries,
            missing_context=missing_context,
            conflicts=[],
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
        if worker_name == "persistent_memory_keyword_agent":
            return self.keyword_agent
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
