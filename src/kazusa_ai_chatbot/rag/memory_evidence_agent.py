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
from kazusa_ai_chatbot.rag.evidence_coverage import (
    EvidenceCoverage,
    assess_evidence_coverage,
    coverage_allows_resolution,
    evidence_buckets_for_coverage,
    has_explicit_multi_target_request,
)
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
    coverage: EvidenceCoverage | None = None,
    confirmed_evidence: list[str] | None = None,
    partial_evidence: list[str] | None = None,
    nearby_evidence: list[str] | None = None,
) -> dict[str, Any]:
    """Build the standard top-level memory capability payload."""
    coverage_payload = coverage or assess_evidence_coverage(
        task="",
        evidence_items=[],
        worker_resolved=False,
    )
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
        "coverage": coverage_payload,
        "confirmed_evidence": list(confirmed_evidence or []),
        "partial_evidence": list(partial_evidence or []),
        "nearby_evidence": list(nearby_evidence or []),
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


def _coverage_fields(
    *,
    task: str,
    evidence_items: list[str],
    worker_resolved: bool,
) -> tuple[EvidenceCoverage, dict[str, list[str]]]:
    """Build coverage and quality-specific evidence buckets."""
    coverage = assess_evidence_coverage(
        task=task,
        evidence_items=evidence_items,
        worker_resolved=worker_resolved,
    )
    buckets = evidence_buckets_for_coverage(coverage, evidence_items)
    return_value = (coverage, buckets)
    return return_value


def _memory_coverage_task(task: str) -> str:
    """Return the task text only when memory evidence needs strict coverage."""

    if has_explicit_multi_target_request(task):
        return_value = task
        return return_value
    return_value = ""
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
    if isinstance(context, dict):
        current_slot = text_or_empty(context.get("current_slot"))
        if current_slot:
            slot_parts.append(current_slot)
    slot_text = "\n".join(slot_parts).lower()

    scoped_user_scope_markers = (
        "current user's",
        "current_user's",
        "current_user",
        "current user",
        "with the current user",
        "remember me",
        "recognize me",
    )
    ambiguous_user_scope_markers = (
        "user's",
        "users'",
        "user preferences",
        "user preference",
        "user decisions",
        "user decision",
    )
    scoped_user_topic_markers = (
        "continuity",
        "accepted preference",
        "accepted preferences",
        "preference",
        "preferences",
        "technical preference",
        "technical preferences",
        "shared experience",
        "past interaction",
        "past interactions",
        "prior interaction",
        "prior interactions",
        "interaction history",
        "prior shared interaction",
        "prior shared interactions",
        "shared history",
        "shared by current user",
        "shared by current_user",
        "current user's shared",
        "current-user's shared",
        "created by current user",
        "created by current_user",
        "current user's created",
        "current-user's created",
        "promise",
        "promises",
        "commitment",
        "commitments",
        "consideration",
        "considerations",
        "remember the current user",
        "recognize the current user",
        "remember me",
        "decision",
        "decisions",
        "choice",
        "choices",
        "care about",
        "cared about",
        "user memory evidence",
        "story lore",
        "story continuity",
        "private lore",
        "private continuity",
        "setting",
    )
    # Query-level context can confirm private scope, but each slot must carry
    # its own scoped-user topic so mixed queries keep independent memory paths.
    has_slot_scoped_user_scope = any(
        marker in slot_text for marker in scoped_user_scope_markers
    )
    has_scoped_user_topic = any(
        marker in slot_text for marker in scoped_user_topic_markers
    )
    if has_slot_scoped_user_scope and has_scoped_user_topic:
        plan = {
            "worker": "user_memory_evidence_agent",
            "reason": "scoped current-user continuity evidence",
        }
        return plan

    lifecycle_status_markers = (
        "completed",
        "outstanding",
        "fulfilled",
        "unfulfilled",
        "finished",
        "unfinished",
        "status",
    )
    has_runtime_user_scope = bool(
        isinstance(context, dict)
        and text_or_empty(context.get("global_user_id"))
    )
    has_lifecycle_status_topic = any(
        marker in slot_text
        for marker in lifecycle_status_markers
    )
    if has_runtime_user_scope and has_scoped_user_topic and has_lifecycle_status_topic:
        plan = {
            "worker": "user_memory_evidence_agent",
            "reason": "scoped current-user memory lifecycle evidence",
        }
        return plan

    has_ambiguous_user_scope = any(
        marker in slot_text for marker in ambiguous_user_scope_markers
    )
    if has_ambiguous_user_scope and has_scoped_user_topic:
        return None

    shared_memory_markers = (
        "official",
        "common sense",
        "world knowledge",
        "character-world",
        "character world",
        "character design",
        "home",
        "address",
        "location",
    )
    if any(marker in normalized for marker in shared_memory_markers):
        plan = {
            "worker": "persistent_memory_search_agent",
            "reason": "semantic durable memory evidence",
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

    if has_scoped_user_topic:
        return None

    if isinstance(context, dict) and text_or_empty(context.get("original_query")):
        return None

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


_SELECTOR_PROMPT = '''\
你要为一个 durable evidence 槽位选择一个有边界的 persistent-memory worker。
不要用本路径回答活跃约定、人物资料、关系判断或实时外部事实。

# 生成步骤
1. 如果任务询问实时活跃约定、活跃承诺或当前 episode 状态，输出
   worker="incompatible"，reason="Recall"。历史已完成/未完成状态证据不适用此条。
2. 如果任务询问人物资料、印象、相性或关系上下文，输出
   worker="incompatible"，reason="Person-context"。
3. 如果任务询问当前天气、温度、营业状态、价格、汇率或任何变化中的实时值，
   输出 worker="incompatible"，reason="Live-context"。
4. 当前用户 durable memory、私有连续性、已接受偏好、用户专属设定、
   当前用户识别、过往互动历史、与当前用户的共同经历，以及过往用户专属
   promise/commitment 的已完成或未完成生命周期状态，使用
   user_memory_evidence_agent。
5. 自然语言 durable facts、精确命名事实、tags、memory_name/dedup_key 查询、
   proper nouns、quoted terms、home/address/location 问题、模糊概念、common sense、
   world knowledge 和 character-world facts，使用 persistent_memory_search_agent。
   该 worker 会执行语义加字面锚点的混合检索。

# 输入格式
{
  "task": "Memory-evidence 槽位文本",
  "original_query": "可用时的去上下文化用户问题",
  "current_slot": "槽位标签",
  "known_facts": "之前 RAG2 槽位得到的有序事实"
}

# 输出格式
只返回有效 JSON：
{
  "worker": "user_memory_evidence_agent | persistent_memory_search_agent | incompatible",
  "reason": "简短来源选择说明"
}
'''
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


def _nearby_memory_rows(worker_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve nearby scoped-memory rows for unresolved observations."""
    value = worker_result.get("result")
    if not isinstance(value, dict):
        return_value: list[dict[str, Any]] = []
        return return_value

    raw_rows = value.get("nearby_memory_rows")
    if not isinstance(raw_rows, list):
        return_value = []
        return return_value

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if isinstance(row, dict):
            rows.append(dict(row))
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
                task=task,
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
        nearby_rows = _nearby_memory_rows(worker_result)
        summaries = _summaries_from_rows(memory_rows)
        worker_resolved = bool(worker_result.get("resolved"))
        coverage_task = _memory_coverage_task(task)
        coverage, evidence_buckets = _coverage_fields(
            task=coverage_task,
            evidence_items=summaries,
            worker_resolved=worker_resolved,
        )
        confirmed_evidence = evidence_buckets["confirmed_evidence"]
        selected_summary = "\n".join(
            confirmed_evidence[:RAG_SEARCH_SELECTED_SUMMARY_LIMIT],
        )
        resolved_refs = _refs_from_rows(memory_rows)
        resolved = (
            worker_resolved
            and bool(summaries)
            and coverage_allows_resolution(coverage)
        )
        missing_context = [] if resolved else ["memory_evidence"]
        projection_rows = memory_rows
        evidence = confirmed_evidence
        observation_candidates: list[dict[str, Any]] = []
        source_hints: list[dict[str, Any]] = []
        if not resolved:
            selected_summary = ""
            resolved_refs = []
            projection_rows = []
            evidence = []
            observation_rows = memory_rows or nearby_rows
            observation_candidates = _memory_observation_candidates(
                observation_rows,
            )
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
            coverage=coverage,
            confirmed_evidence=confirmed_evidence,
            partial_evidence=evidence_buckets["partial_evidence"],
            nearby_evidence=evidence_buckets["nearby_evidence"],
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
        task: str,
        source_policy: str,
        missing_context: list[str],
        primary_worker: str,
        worker_payloads: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an unresolved result without calling a memory source."""
        coverage_task = _memory_coverage_task(task)
        coverage, evidence_buckets = _coverage_fields(
            task=coverage_task,
            evidence_items=[],
            worker_resolved=False,
        )
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
            coverage=coverage,
            confirmed_evidence=evidence_buckets["confirmed_evidence"],
            partial_evidence=evidence_buckets["partial_evidence"],
            nearby_evidence=evidence_buckets["nearby_evidence"],
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
