"""Top-level RAG capability agent for conversation-history evidence."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.conversation_aggregate_agent import (
    ConversationAggregateAgent,
)
from kazusa_ai_chatbot.rag.conversation_filter_agent import ConversationFilterAgent
from kazusa_ai_chatbot.rag.conversation_keyword_agent import ConversationKeywordAgent
from kazusa_ai_chatbot.rag.conversation_search_agent import ConversationSearchAgent
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.prompt_projection import project_selector_input_for_llm
from kazusa_ai_chatbot.time_context import (
    format_timestamp_for_llm,
)
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_CAPABILITY_NAME = "conversation_evidence"
_AGENT_NAME = "conversation_evidence_agent"
_UNCACHED_REASON = "capability_orchestrator_uncached"
_URL_PATTERN = re.compile(r"https?://[^\s)>\]}\"']+")
_EXPLICIT_SPEAKER_SCOPE_PATTERN = re.compile(
    r"\bspeaker\s*=\s*(current_user|active_character|any_speaker)\b",
    re.IGNORECASE,
)
_PERSON_SPEAKER_SCOPE_PATTERN = re.compile(
    r"\bspeaker\s*=\s*person\s+resolved\s+in\s+slot\s+\d+\b",
    re.IGNORECASE,
)
_SCOPE_CURRENT_USER = "current_user"
_SCOPE_ACTIVE_CHARACTER = "active_character"
_SCOPE_ANY_SPEAKER = "any_speaker"
_SCOPE_PERSON_RESOLVED = "person_resolved"
_KNOWN_WORKERS = {
    "conversation_keyword_agent",
    "conversation_search_agent",
    "conversation_filter_agent",
    "conversation_aggregate_agent",
    "incompatible",
}
_EXACT_ANCHOR_CHARS = (
    '"',
    "'",
    "\u2018",
    "\u2019",
    "\u201c",
    "\u201d",
    "\u300c",
    "\u300d",
    "\u300e",
    "\u300f",
)


class _ConversationProjection(TypedDict):
    """Canonical evidence shape exposed by the conversation capability."""

    summaries: list[str]
    resolved_refs: list[dict[str, Any]]


def _clip_text(value: object, *, limit: int = 1000) -> str:
    """Return compact prompt-facing text.

    Args:
        value: Source value to convert to text.
        limit: Maximum number of characters.

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
    """Build the standard top-level conversation capability payload."""

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


def _strip_prefix(task: str) -> str:
    """Remove the semantic capability prefix when present."""
    if ":" not in task:
        return task.strip()
    _, _, remainder = task.partition(":")
    return_value = remainder.strip()
    return return_value


def _speaker_scope(task: str) -> str:
    """Extract the optional conversation author scope from a slot.

    Args:
        task: Conversation-evidence slot text.

    Returns:
        One approved scope constant, or an empty string for legacy unscoped
        slots.
    """

    task_body = _strip_prefix(task)
    if _PERSON_SPEAKER_SCOPE_PATTERN.search(task_body):
        return_value = _SCOPE_PERSON_RESOLVED
        return return_value

    explicit_match = _EXPLICIT_SPEAKER_SCOPE_PATTERN.search(task_body)
    if explicit_match is None:
        return_value = ""
        return return_value

    return_value = explicit_match.group(1).lower()
    return return_value


def _requires_person_ref(task: str) -> bool:
    """Return whether a slot needs a prior structured person reference.

    Args:
        task: Conversation-evidence slot text.

    Returns:
        True when the slot explicitly depends on a person from an earlier slot.
    """

    if _speaker_scope(task) == _SCOPE_PERSON_RESOLVED:
        return True

    normalized = _strip_prefix(task).lower()
    return_value = "resolved in slot" in normalized
    return return_value


def _deterministic_plan(task: str) -> dict[str, Any] | None:
    """Parse structured conversation-evidence slots without selector LLM."""

    task_body = _strip_prefix(task)
    normalized = task_body.lower()
    speaker_scope = _speaker_scope(task)
    requires_person_ref = _requires_person_ref(task)

    if "active agreement" in normalized or "current episode" in normalized:
        plan = {
            "worker": "incompatible",
            "reason": "Recall",
            "requires_person_ref": False,
        }
        return plan

    if "durable" in normalized or "official address" in normalized:
        plan = {
            "worker": "incompatible",
            "reason": "Memory-evidence",
            "requires_person_ref": False,
        }
        return plan

    if (
        "count " in normalized
        or "how many" in normalized
        or "ranking" in normalized
        or "grouped" in normalized
        or "aggregate" in normalized
    ):
        plan = {
            "worker": "conversation_aggregate_agent",
            "reason": "aggregate/count conversation evidence",
            "requires_person_ref": requires_person_ref,
        }
        return plan

    if (
        any(anchor in task_body for anchor in _EXACT_ANCHOR_CHARS)
        or "exact phrase" in normalized
        or "exact term" in normalized
        or "url" in normalized
        or "filename" in normalized
        or "literal" in normalized
    ):
        plan = {
            "worker": "conversation_keyword_agent",
            "reason": "literal phrase, URL, filename, or exact anchor",
            "requires_person_ref": requires_person_ref,
        }
        return plan

    if (
        "topic" in normalized
        or "about " in normalized
        or "semantic" in normalized
    ):
        plan = {
            "worker": "conversation_search_agent",
            "reason": "semantic or fuzzy topic conversation evidence",
            "requires_person_ref": requires_person_ref,
        }
        return plan

    if (
        "resolved in slot" in normalized
        or "recent messages from" in normalized
        or (
            speaker_scope
            and "recent messages" in normalized
        )
        or "date-window" in normalized
        or "from timestamp" in normalized
        or "to timestamp" in normalized
    ):
        plan = {
            "worker": "conversation_filter_agent",
            "reason": "structured conversation filter",
            "requires_person_ref": requires_person_ref,
        }
        return plan

    return None


def _normalize_selector_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize an LLM selector payload to approved fields."""
    worker = text_or_empty(raw_plan.get("worker"))
    if worker not in _KNOWN_WORKERS:
        worker = "conversation_search_agent"
    reason = text_or_empty(raw_plan.get("reason"))
    requires_person_ref = bool(raw_plan.get("requires_person_ref"))
    plan = {
        "worker": worker,
        "reason": reason,
        "requires_person_ref": requires_person_ref,
    }
    return plan


_SELECTOR_PROMPT = """\
You choose one bounded conversation-history worker path for a RAG evidence slot.
Do not answer from durable memory, active episode progress, user profiles, or web.

# Generation Procedure
1. If the task asks for an active/current agreement or episode state, output
   worker="incompatible" and reason="Recall".
2. If the task asks for a durable world fact, output worker="incompatible" and
   reason="Memory-evidence".
3. Use conversation_keyword_agent for exact phrases, URLs, filenames, literal
   terms, and provenance of a quoted message.
4. Use conversation_search_agent for fuzzy topics and semantic message evidence.
5. Use conversation_filter_agent for known user/time/date-window retrieval.
6. Use conversation_aggregate_agent for counts, rankings, or grouped stats.
7. Set requires_person_ref=true when the task uses
   speaker=person resolved in slot N or otherwise references a person from a
   previous slot.

# Input Format
{
  "task": "Conversation-evidence slot text",
  "original_query": "decontextualized user query when available",
  "current_slot": "slot label",
  "known_facts": "ordered facts from previous RAG2 slots"
}

# Output Format
Return valid JSON only:
{
  "worker": "conversation_keyword_agent | conversation_search_agent | conversation_filter_agent | conversation_aggregate_agent | incompatible",
  "reason": "short source selection explanation",
  "requires_person_ref": true
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
    """Select the bounded conversation worker path for one slot."""
    deterministic_plan = _deterministic_plan(task)
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


def _iter_known_refs(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect structured refs from previous RAG2 known facts."""
    refs: list[dict[str, Any]] = []
    known_facts = context.get("known_facts")
    if not isinstance(known_facts, list):
        return refs

    for fact in known_facts:
        if not isinstance(fact, dict):
            continue
        raw_result = fact.get("raw_result")
        if not isinstance(raw_result, dict):
            continue
        raw_refs = raw_result.get("resolved_refs")
        if not isinstance(raw_refs, list):
            continue
        for ref in raw_refs:
            if isinstance(ref, dict):
                refs.append(ref)
    return refs


def _first_person_ref(context: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first structured person reference from previous slots."""
    for ref in _iter_known_refs(context):
        if ref.get("ref_type") == "person":
            return ref
    return_value = None
    return return_value


def _worker_context(
    context: dict[str, Any],
    person_ref: dict[str, Any] | None,
    speaker_scope: str,
) -> dict[str, Any]:
    """Build worker context with approved structured handoff values.

    Args:
        context: Runtime context from the RAG supervisor.
        person_ref: Prior resolved person reference, when the slot depends on
            one.
        speaker_scope: Optional author scope declared by the conversation slot.

    Returns:
        Worker context with only the user filter implied by the author scope.
    """
    worker_context = dict(context)
    worker_context["exclude_current_question"] = True

    if speaker_scope == _SCOPE_ANY_SPEAKER:
        worker_context.pop("global_user_id", None)
        worker_context.pop("display_name", None)

    if speaker_scope == _SCOPE_ACTIVE_CHARACTER:
        worker_context.pop("global_user_id", None)
        worker_context.pop("display_name", None)
        character_profile = context.get("character_profile")
        if isinstance(character_profile, dict):
            global_user_id = text_or_empty(character_profile.get("global_user_id"))
            display_name = text_or_empty(character_profile.get("name"))
            if global_user_id:
                worker_context["global_user_id"] = global_user_id
            if display_name:
                worker_context["display_name"] = display_name

    if person_ref is not None:
        global_user_id = text_or_empty(person_ref.get("global_user_id"))
        display_name = text_or_empty(person_ref.get("display_name"))
        if global_user_id:
            worker_context["global_user_id"] = global_user_id
        if display_name:
            worker_context["display_name"] = display_name
    return worker_context


def _projection_from_worker(
    worker_name: str,
    worker_result: dict[str, Any],
) -> _ConversationProjection:
    """Project one worker's explicit raw contract into capability evidence."""
    raw_result = worker_result.get("result")

    if worker_name == "conversation_search_agent":
        rows = _semantic_message_rows(raw_result)
        projection = _message_projection(rows)
        return projection

    if worker_name in {
        "conversation_keyword_agent",
        "conversation_filter_agent",
    }:
        rows = _plain_message_rows(raw_result)
        projection = _message_projection(rows)
        return projection

    if worker_name == "conversation_aggregate_agent":
        projection = _aggregate_projection(raw_result)
        return projection

    projection = _empty_projection()
    return projection


def _empty_projection() -> _ConversationProjection:
    """Build an empty canonical conversation evidence projection."""
    projection: _ConversationProjection = {
        "summaries": [],
        "resolved_refs": [],
    }
    return projection


def _semantic_message_rows(value: object) -> list[dict[str, Any]]:
    """Extract message rows from semantic search ``(score, message)`` results."""
    rows: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return rows

    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue

        score, message = item
        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and isinstance(message, dict)
        ):
            rows.append(message)

    return rows


def _plain_message_rows(value: object) -> list[dict[str, Any]]:
    """Extract message rows from keyword and structured-filter results."""
    if not isinstance(value, list):
        return_value: list[dict[str, Any]] = []
        return return_value

    rows = [
        row
        for row in value
        if isinstance(row, dict)
    ]
    return rows


def _message_projection(rows: list[dict[str, Any]]) -> _ConversationProjection:
    """Project typed conversation message rows into summaries and refs."""
    summaries = [
        summary
        for row in rows
        if (summary := _clip_text(_message_row_text(row), limit=500))
    ]
    resolved_refs = _refs_from_message_rows(rows)
    projection: _ConversationProjection = {
        "summaries": summaries,
        "resolved_refs": resolved_refs,
    }
    return projection


def _message_row_text(row: dict[str, Any]) -> str:
    """Extract prompt-facing text from one canonical message row."""
    body = text_or_empty(row.get("body_text"))
    if not body:
        body = text_or_empty(row.get("content"))
    if not body:
        body = text_or_empty(row.get("summary"))
    if not body:
        body = text_or_empty(row.get("text"))

    display_name = text_or_empty(row.get("display_name"))
    timestamp = format_timestamp_for_llm(text_or_empty(row.get("timestamp")))
    if display_name and timestamp and body:
        text = f"{display_name} at {timestamp}: {body}"
        return text
    if display_name and body:
        text = f"{display_name}: {body}"
        return text
    return body


def _refs_from_message_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract structured speaker, message, and URL refs from message rows."""
    refs: list[dict[str, Any]] = []
    for row in rows:
        global_user_id = text_or_empty(row.get("global_user_id"))
        display_name = text_or_empty(row.get("display_name"))
        if global_user_id or display_name:
            refs.append(
                {
                    "ref_type": "person",
                    "role": "speaker",
                    "global_user_id": global_user_id,
                    "display_name": display_name,
                }
            )

        platform_message_id = text_or_empty(row.get("platform_message_id"))
        timestamp = text_or_empty(row.get("timestamp"))
        if platform_message_id or timestamp:
            refs.append(
                {
                    "ref_type": "message",
                    "platform_message_id": platform_message_id,
                    "timestamp": timestamp,
                    "global_user_id": global_user_id,
                    "display_name": display_name,
                }
            )

        text = _message_row_text(row)
        refs.extend(_url_refs_from_text(text))
    return refs


def _aggregate_projection(value: object) -> _ConversationProjection:
    """Project aggregate worker payloads into canonical conversation evidence."""
    if not isinstance(value, dict):
        projection = _empty_projection()
        return projection

    rows_value = value.get("rows")
    rows = rows_value if isinstance(rows_value, list) else []
    row_summaries = [
        row_summary
        for row in rows
        if isinstance(row, dict)
        if (row_summary := _aggregate_row_summary(row))
    ]
    total_count = value.get("total_count")
    total_text = _aggregate_total_text(total_count)
    aggregate = text_or_empty(value.get("aggregate")) or "conversation aggregate"
    time_window = text_or_empty(value.get("time_window"))

    summary_parts = [aggregate]
    if time_window:
        summary_parts.append(f"window={time_window}")
    if total_text:
        summary_parts.append(f"total={total_text}")
    if row_summaries:
        summary_parts.append("top rows: " + "; ".join(row_summaries[:5]))

    if len(summary_parts) == 1:
        summaries: list[str] = []
    else:
        summary = ", ".join(summary_parts)
        summaries = [summary]

    projection: _ConversationProjection = {
        "summaries": summaries,
        "resolved_refs": _refs_from_aggregate_rows(rows),
    }
    return projection


def _aggregate_total_text(value: object) -> str:
    """Render aggregate total count without assuming a raw payload type."""
    if isinstance(value, int) and not isinstance(value, bool):
        return_value = str(value)
        return return_value
    return_value = text_or_empty(value)
    return return_value


def _aggregate_row_summary(row: dict[str, Any]) -> str:
    """Render one aggregate row for the RAG finalizer."""
    display_name = _first_display_name(row.get("display_names"))
    if not display_name:
        display_name = text_or_empty(row.get("platform_user_id"))
    if not display_name:
        display_name = text_or_empty(row.get("global_user_id"))

    message_count = row.get("message_count")
    if isinstance(message_count, int) and not isinstance(message_count, bool):
        count_text = str(message_count)
    else:
        count_text = text_or_empty(message_count)

    last_timestamp = format_timestamp_for_llm(text_or_empty(row.get("last_timestamp")))
    parts = []
    if display_name:
        parts.append(display_name)
    if count_text:
        parts.append(f"{count_text} messages")
    if last_timestamp:
        parts.append(f"last={last_timestamp}")

    summary = ", ".join(parts)
    return summary


def _first_display_name(value: object) -> str:
    """Return the first non-empty display name from an aggregate row."""
    if not isinstance(value, list):
        return_value = text_or_empty(value)
        return return_value

    for item in value:
        display_name = text_or_empty(item)
        if display_name:
            return display_name

    return_value = ""
    return return_value


def _refs_from_aggregate_rows(rows: list[object]) -> list[dict[str, Any]]:
    """Extract person refs from aggregate result rows when available."""
    refs: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        global_user_id = text_or_empty(row.get("global_user_id"))
        display_name = _first_display_name(row.get("display_names"))
        if not global_user_id and not display_name:
            continue

        refs.append(
            {
                "ref_type": "person",
                "role": "aggregate_subject",
                "global_user_id": global_user_id,
                "display_name": display_name,
            }
        )

    return refs


def _url_refs_from_text(text: str) -> list[dict[str, str]]:
    """Extract URL refs from one text block."""
    refs = [
        {
            "ref_type": "url",
            "role": "posted_url",
            "url": match.group(0).rstrip(".,"),
        }
        for match in _URL_PATTERN.finditer(text)
    ]
    return refs


class ConversationEvidenceAgent(BaseRAGHelperAgent):
    """Top-level RAG helper for evidence from conversation history."""

    def __init__(self, *, cache_runtime=None) -> None:
        """Create the uncached conversation-evidence capability agent."""

        super().__init__(
            name=_AGENT_NAME,
            cache_name="",
            cache_runtime=cache_runtime,
        )
        self.keyword_agent = ConversationKeywordAgent(cache_runtime=cache_runtime)
        self.search_agent = ConversationSearchAgent(cache_runtime=cache_runtime)
        self.filter_agent = ConversationFilterAgent(cache_runtime=cache_runtime)
        self.aggregate_agent = ConversationAggregateAgent(
            cache_runtime=cache_runtime
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """Resolve one conversation evidence slot through one worker path."""

        del max_attempts

        plan = await _select_plan(task, context)
        primary_worker = plan["worker"]
        speaker_scope = _speaker_scope(task)
        if primary_worker == "incompatible":
            reason = text_or_empty(plan["reason"]) or "unsupported"
            result = self._unresolved(
                source_policy=f"incompatible intent; use {reason}",
                missing_context=[f"incompatible_intent:{reason}"],
                primary_worker="",
                worker_payloads={},
            )
            return result

        person_ref = _first_person_ref(context)
        if bool(plan["requires_person_ref"]) and person_ref is None:
            result = self._unresolved(
                source_policy="structured person ref required but missing",
                missing_context=["person_ref"],
                primary_worker=primary_worker,
                worker_payloads={},
            )
            return result

        effective_person_ref = person_ref if bool(plan["requires_person_ref"]) else None
        worker = self._worker_for_name(primary_worker)
        context_for_worker = _worker_context(
            context,
            effective_person_ref,
            speaker_scope,
        )
        worker_result = await worker.run(
            task,
            context_for_worker,
            max_attempts=1,
        )
        worker_payloads = {primary_worker: worker_result}
        projection = _projection_from_worker(primary_worker, worker_result)
        summaries = projection["summaries"]
        selected_summary = "\n".join(summaries[:5])
        resolved_refs = projection["resolved_refs"]
        resolved = bool(worker_result.get("resolved")) and bool(summaries)
        missing_context = [] if resolved else ["conversation_evidence"]
        payload = _result_payload(
            selected_summary=selected_summary,
            primary_worker=primary_worker,
            supporting_workers=[],
            source_policy=text_or_empty(plan["reason"]),
            resolved_refs=resolved_refs,
            projection_payload={"summaries": summaries},
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
        """Return the configured worker instance for an approved name."""
        if worker_name == "conversation_keyword_agent":
            return self.keyword_agent
        if worker_name == "conversation_filter_agent":
            return self.filter_agent
        if worker_name == "conversation_aggregate_agent":
            return self.aggregate_agent
        return self.search_agent

    def _unresolved(
        self,
        *,
        source_policy: str,
        missing_context: list[str],
        primary_worker: str,
        worker_payloads: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an unresolved result without calling another source."""
        payload = _result_payload(
            selected_summary="",
            primary_worker=primary_worker,
            supporting_workers=[],
            source_policy=source_policy,
            resolved_refs=[],
            projection_payload={"summaries": []},
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
