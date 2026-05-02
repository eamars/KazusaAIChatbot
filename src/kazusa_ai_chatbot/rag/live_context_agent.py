"""Top-level RAG capability agent for live external context."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.conversation_search_agent import ConversationSearchAgent
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.persistent_memory_search_agent import PersistentMemorySearchAgent
from kazusa_ai_chatbot.rag.web_search_agent import WebSearchAgent
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_CAPABILITY_NAME = "live_context"
_AGENT_NAME = "live_context_agent"
_UNCACHED_REASON = "capability_orchestrator_uncached"
_URL_PATTERN = re.compile(r"https?://[^\s)>\]}\"']+")
_KNOWN_FACT_TYPES = {
    "weather",
    "temperature",
    "opening_status",
    "schedule",
    "price",
    "exchange_rate",
    "current_event_status",
    "other",
}
_KNOWN_TARGET_SOURCES = {
    "explicit",
    "active_character_default",
    "current_user_recent",
    "unknown",
}


def _clip_text(value: object, *, limit: int = 1000) -> str:
    """Return compact text for prompt-facing evidence fields.

    Args:
        value: Text-like evidence from a worker result.
        limit: Maximum output length.

    Returns:
        Stripped and clipped text.
    """

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
    supporting_workers: list[str] | None = None,
    source_policy: object = "",
    resolved_refs: list[dict[str, Any]] | None = None,
    projection_payload: dict[str, Any] | None = None,
    worker_payloads: dict[str, Any] | None = None,
    evidence: list[str] | None = None,
    missing_context: list[str] | None = None,
    conflicts: list[str] | None = None,
) -> dict[str, Any]:
    """Build the standard top-level capability payload.

    Args:
        selected_summary: Short factual summary selected for the evaluator.
        primary_worker: Worker/tool that produced the final live evidence.
        supporting_workers: Optional target/scope resolver workers.
        source_policy: Short authority and freshness explanation.
        resolved_refs: Structured refs available to later ordered slots.
        projection_payload: Public projection adapter payload.
        worker_payloads: Raw worker outputs for debug tracing.
        evidence: Prompt-facing evidence strings.
        missing_context: Missing scope fields for unresolved results.
        conflicts: Source conflicts discovered while resolving the target.

    Returns:
        Result payload accepted by the RAG2 evaluator and projection adapter.
    """

    payload = {
        "selected_summary": _clip_text(selected_summary),
        "capability": _CAPABILITY_NAME,
        "primary_worker": primary_worker,
        "supporting_workers": list(supporting_workers or []),
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


def _extract_fact_type(task: str) -> str:
    """Classify the live fact type from structured slot text.

    Args:
        task: Initializer slot or dispatcher task text.

    Returns:
        One approved live fact type label.
    """

    normalized = task.lower()
    if "temperature" in normalized:
        return "temperature"
    if "weather" in normalized:
        return "weather"
    if "opening status" in normalized or "open status" in normalized:
        return "opening_status"
    if "open " in normalized or "is open" in normalized:
        return "opening_status"
    if "schedule" in normalized or "timetable" in normalized:
        return "schedule"
    if "price" in normalized or "cost" in normalized:
        return "price"
    if "exchange rate" in normalized:
        return "exchange_rate"
    if "current event" in normalized or "event status" in normalized:
        return "current_event_status"
    return "other"


def _strip_live_prefix(task: str) -> str:
    """Remove the semantic capability prefix when present."""
    if ":" not in task:
        return task.strip()
    _, _, remainder = task.partition(":")
    return_value = remainder.strip()
    return return_value


def _clean_target(target: str) -> str:
    """Normalize a target phrase extracted from structured slot text."""
    cleaned = target.strip(" .,:;\"'")
    return cleaned


def _target_after_marker(task_body: str, marker: str) -> str:
    """Extract text following a structured target marker."""
    pattern = re.compile(rf"{re.escape(marker)}\s+(.+)$", re.IGNORECASE)
    match = pattern.search(task_body)
    if match is None:
        return_value = ""
        return return_value
    return_value = _clean_target(match.group(1))
    return return_value


def _deterministic_plan(task: str) -> dict[str, Any] | None:
    """Parse structured live-context slots without a selector LLM.

    Args:
        task: Slot text supplied by the RAG2 dispatcher.

    Returns:
        A selector plan when the slot contains approved structured markers,
        otherwise ``None`` so the selector LLM can classify the request.
    """

    task_body = _strip_live_prefix(task)
    normalized = task_body.lower()
    fact_type = _extract_fact_type(task_body)

    if "current user's location" in normalized:
        plan = {
            "fact_type": fact_type,
            "target_source": "current_user_recent",
            "target": "",
            "missing_context": [],
        }
        return plan

    if "active character's location" in normalized:
        plan = {
            "fact_type": fact_type,
            "target_source": "active_character_default",
            "target": "",
            "missing_context": [],
        }
        return plan

    target = _target_after_marker(task_body, "explicit location")
    if target:
        plan = {
            "fact_type": fact_type,
            "target_source": "explicit",
            "target": target,
            "missing_context": [],
        }
        return plan

    target = _target_after_marker(task_body, "explicit target")
    if target:
        plan = {
            "fact_type": fact_type,
            "target_source": "explicit",
            "target": target,
            "missing_context": [],
        }
        return plan

    if " for " in normalized:
        _, _, target = task_body.rpartition(" for ")
        target = _clean_target(target)
        if target and "location" not in target.lower():
            plan = {
                "fact_type": fact_type,
                "target_source": "explicit",
                "target": target,
                "missing_context": [],
            }
            return plan

    return None


def _normalize_selector_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize an LLM selector payload to the approved plan fields."""
    fact_type = text_or_empty(raw_plan.get("fact_type"))
    if fact_type not in _KNOWN_FACT_TYPES:
        fact_type = "other"

    target_source = text_or_empty(raw_plan.get("target_source"))
    if target_source not in _KNOWN_TARGET_SOURCES:
        target_source = "unknown"

    missing_context = raw_plan.get("missing_context")
    if not isinstance(missing_context, list):
        missing_context = []

    plan = {
        "fact_type": fact_type,
        "target_source": target_source,
        "target": text_or_empty(raw_plan.get("target")),
        "missing_context": [
            text
            for item in missing_context
            if (text := text_or_empty(item))
        ],
    }
    return plan


_SELECTOR_PROMPT = """\
You choose the bounded source path for one live external fact request.
The live value itself must come from public live/web evidence, never from memory.
Memory or recent conversation may only resolve the target/scope, such as a
stable character location or a recently stated user location.

# Generation Procedure
1. Identify the changing real-time fact type: weather, temperature,
   opening_status, schedule, price, exchange_rate, current_event_status, or other.
2. Identify target_source:
   - explicit: the task names the place, venue, market, product, event, or public target.
   - active_character_default: the task asks for the character's location/scope.
   - current_user_recent: the task asks for the current user's own location/scope.
   - unknown: no trusted target/scope is visible.
3. Fill target only when target_source is explicit.
4. If the target is unknown, include the missing context, usually "location" or "target".

# Input Format
{
  "task": "Live-context slot text",
  "original_query": "decontextualized user query when available",
  "current_slot": "slot label",
  "known_facts": "ordered facts from previous RAG2 slots"
}

# Output Format
Return valid JSON only:
{
  "fact_type": "weather | temperature | opening_status | schedule | price | exchange_rate | current_event_status | other",
  "target_source": "explicit | active_character_default | current_user_recent | unknown",
  "target": "explicit target text, otherwise empty",
  "missing_context": ["location or target when unresolved"]
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
    """Select a bounded live target path for the requested slot.

    Args:
        task: Slot text supplied by the dispatcher.
        context: RAG2 delegate context.

    Returns:
        Normalized source-selection plan.
    """

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


def _extract_text_from_rows(rows: object) -> str:
    """Extract a target/scope phrase from a worker result payload.

    Args:
        rows: Raw worker ``result`` value.

    Returns:
        First available human-readable target text.
    """

    if isinstance(rows, str):
        return rows.strip()

    if isinstance(rows, dict):
        for field in (
            "content",
            "body_text",
            "text",
            "description",
            "summary",
            "selected_summary",
            "fact",
        ):
            text = text_or_empty(rows.get(field))
            if text:
                return text
        return_value = ""
        return return_value

    if isinstance(rows, list):
        for item in rows:
            text = _extract_text_from_rows(item)
            if text:
                return text
        return_value = ""
        return return_value

    return_value = text_or_empty(rows)
    return return_value


def _extract_worker_text(worker_result: dict[str, Any]) -> str:
    """Extract target/scope text from a helper-agent result."""
    rows = worker_result.get("result")
    text = _extract_text_from_rows(rows)
    return text


def _url_from_text(text: str) -> str:
    """Return the first URL embedded in a web evidence string."""
    match = _URL_PATTERN.search(text)
    if match is None:
        return_value = ""
        return return_value
    return_value = match.group(0).rstrip(".,")
    return return_value


def _location_ref(*, role: str, text: str) -> dict[str, str]:
    """Build a structured location reference for downstream slots."""
    ref = {
        "ref_type": "location",
        "role": role,
        "text": text,
    }
    return ref


def _web_task(*, fact_type: str, target: str, original_task: str) -> str:
    """Build the delegated web worker task for a resolved live target."""
    task = (
        "Web-evidence: retrieve current live external evidence "
        f"for fact_type={fact_type}; target={target}; "
        f"original_task={original_task}"
    )
    return task


class LiveContextAgent(BaseRAGHelperAgent):
    """Top-level RAG helper for live facts with bounded target resolution."""

    def __init__(self, *, cache_runtime=None) -> None:
        """Create the uncached live-context capability agent.

        Args:
            cache_runtime: Optional cache runtime override for tests.
        """

        super().__init__(
            name=_AGENT_NAME,
            cache_name="",
            cache_runtime=cache_runtime,
        )
        self.web_agent = WebSearchAgent(cache_runtime=cache_runtime)
        self.memory_search_agent = PersistentMemorySearchAgent(
            cache_runtime=cache_runtime
        )
        self.conversation_search_agent = ConversationSearchAgent(
            cache_runtime=cache_runtime
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """Resolve one live external fact through target/scope then web.

        Args:
            task: ``Live-context:`` slot text from the RAG2 dispatcher.
            context: Existing RAG2 delegate context.
            max_attempts: Accepted for interface compatibility; this agent
                performs one bounded orchestration attempt.

        Returns:
            Standard RAG helper result using top-level capability payload keys.
        """

        del max_attempts

        plan = await _select_plan(task, context)
        fact_type = plan["fact_type"]
        target_source = plan["target_source"]
        target = text_or_empty(plan["target"])
        missing_context = list(plan["missing_context"])
        supporting_workers: list[str] = []
        worker_payloads: dict[str, Any] = {}
        resolved_refs: list[dict[str, Any]] = []
        source_policy = ""

        if target_source == "explicit" and target:
            resolved_refs.append(
                _location_ref(role="target_location", text=target)
            )
            source_policy = "explicit target supplied by Live-context slot"
        elif target_source == "active_character_default":
            target_result = await self._resolve_character_target(task, context)
            worker_payloads["persistent_memory_search_agent"] = target_result
            supporting_workers.append("persistent_memory_search_agent")
            if not bool(target_result.get("resolved")):
                result = self._missing_target_result(
                    source_policy=(
                        "target_scope_lookup failed for active character "
                        "default location"
                    ),
                    missing_context=["location"],
                    supporting_workers=supporting_workers,
                    worker_payloads=worker_payloads,
                )
                return result
            target = _extract_worker_text(target_result)
            if not target:
                result = self._missing_target_result(
                    source_policy=(
                        "target_scope_lookup returned no active character "
                        "location text"
                    ),
                    missing_context=["location"],
                    supporting_workers=supporting_workers,
                    worker_payloads=worker_payloads,
                )
                return result
            resolved_refs.append(_location_ref(role="character_default", text=target))
            source_policy = (
                "target_scope_lookup: persistent memory resolved stable "
                "active character target; web resolves the live value"
            )
        elif target_source == "current_user_recent":
            target_result = await self._resolve_user_recent_target(task, context)
            worker_payloads["conversation_search_agent"] = target_result
            supporting_workers.append("conversation_search_agent")
            if not bool(target_result.get("resolved")):
                result = self._missing_target_result(
                    source_policy=(
                        "target_scope_lookup failed for user_recent location; "
                        "no character fallback was attempted"
                    ),
                    missing_context=["location"],
                    supporting_workers=supporting_workers,
                    worker_payloads=worker_payloads,
                )
                return result
            target = _extract_worker_text(target_result)
            if not target:
                result = self._missing_target_result(
                    source_policy=(
                        "target_scope_lookup returned no user_recent "
                        "location text; no character fallback was attempted"
                    ),
                    missing_context=["location"],
                    supporting_workers=supporting_workers,
                    worker_payloads=worker_payloads,
                )
                return result
            resolved_refs.append(_location_ref(role="user_recent", text=target))
            source_policy = (
                "target_scope_lookup: recent same-user conversation resolved "
                "user_recent target; web resolves the live value"
            )
        else:
            if not missing_context:
                missing_context = ["location"]
            result = self._missing_target_result(
                source_policy="live target/scope unresolved",
                missing_context=missing_context,
                supporting_workers=supporting_workers,
                worker_payloads=worker_payloads,
            )
            return result

        web_task = _web_task(
            fact_type=fact_type,
            target=target,
            original_task=task,
        )
        web_result = await self.web_agent.run(
            web_task,
            context,
            max_attempts=1,
        )
        worker_payloads["web_search_agent2"] = web_result
        evidence_text = _clip_text(web_result.get("result"))
        resolved = bool(web_result.get("resolved")) and bool(evidence_text)
        projection_payload = {
            "external_text": evidence_text,
            "url": _url_from_text(evidence_text),
        }
        payload = _result_payload(
            selected_summary=evidence_text,
            primary_worker="web_search_agent2",
            supporting_workers=supporting_workers,
            source_policy=source_policy,
            resolved_refs=resolved_refs,
            projection_payload=projection_payload,
            worker_payloads=worker_payloads,
            evidence=[evidence_text] if evidence_text else [],
            missing_context=[] if resolved else ["live_evidence"],
            conflicts=[],
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved={resolved} "
            f"primary_worker=web_search_agent2 "
            f"missing_context={payload['missing_context']} "
            f"selected_summary={payload['selected_summary']} "
            f"cache_reason={_UNCACHED_REASON}"
        )
        logger.debug(
            f"{_AGENT_NAME} debug: resolved_refs={resolved_refs} "
            f"projection_payload={projection_payload} "
            f"worker_payloads={worker_payloads}"
        )
        result = _agent_result(resolved=resolved, payload=payload)
        return result

    async def _resolve_character_target(
        self,
        task: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve the active character's stable live target from memory."""
        lookup_task = (
            "target_scope_lookup: resolve the active character's stable "
            "location or venue for a live external fact; do not retrieve "
            f"live weather, temperature, schedule, price, or status. {task}"
        )
        result = await self.memory_search_agent.run(
            lookup_task,
            context,
            max_attempts=1,
        )
        return result

    async def _resolve_user_recent_target(
        self,
        task: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve the current user's recently stated live target."""
        lookup_task = (
            "target_scope_lookup: find the current user's recently stated "
            "location or venue for a live external fact; same-user scope only; "
            "do not retrieve live weather, temperature, schedule, price, "
            f"or status. {task}"
        )
        result = await self.conversation_search_agent.run(
            lookup_task,
            context,
            max_attempts=1,
        )
        return result

    def _missing_target_result(
        self,
        *,
        source_policy: str,
        missing_context: list[str],
        supporting_workers: list[str],
        worker_payloads: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an unresolved result for missing target/scope."""
        payload = _result_payload(
            selected_summary="",
            primary_worker="",
            supporting_workers=supporting_workers,
            source_policy=source_policy,
            resolved_refs=[],
            projection_payload={},
            worker_payloads=worker_payloads,
            evidence=[],
            missing_context=missing_context,
            conflicts=[],
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved=False primary_worker= "
            f"missing_context={missing_context} selected_summary='' "
            f"cache_reason={_UNCACHED_REASON}"
        )
        logger.debug(
            f"{_AGENT_NAME} debug: resolved_refs=[] projection_payload={{}} "
            f"worker_payloads={worker_payloads}"
        )
        result = _agent_result(resolved=False, payload=payload)
        return result
