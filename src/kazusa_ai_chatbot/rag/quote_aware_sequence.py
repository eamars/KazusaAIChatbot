"""Quote-aware sequencing around the progressive RAG supervisor."""

from __future__ import annotations

import copy
import logging
from typing import Any

from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_supervisor2 import (
    call_rag_supervisor,
)
from kazusa_ai_chatbot.utils import log_preview

logger = logging.getLogger(__name__)

FACT_SUMMARY_CHAR_LIMIT = 1600


def _reply_excerpt_text(reply_context: dict[str, Any]) -> str:
    """Return normalized quote text when the reply context carries one."""
    if not isinstance(reply_context, dict):
        return_value = ""
        return return_value

    reply_excerpt = reply_context.get("reply_excerpt")
    if not isinstance(reply_excerpt, str):
        return_value = ""
        return return_value

    return_value = reply_excerpt.strip()
    return return_value


def _build_quote_grounding_query(reply_excerpt: str) -> str:
    """Build the first-pass query that grounds quoted factual content."""
    query = (
        "Research factual content contained in the quoted/replied text.\n"
        "Treat the quote as quoted material, not verified truth.\n"
        "Do not search conversation history or identify the speaker unless "
        "the quote itself asks for provenance.\n"
        "Preserve exact quoted names, model names, numbers, units, and claim "
        "values.\n"
        "Use quoted names and model names as primary search anchors.\n"
        "Treat quoted numbers and units as claim values to verify, not "
        "mandatory search keywords for every search.\n\n"
        f"Quoted/replied text:\n{reply_excerpt}"
    )
    return query


def _build_fresh_query_with_quote_facts(
    *,
    fresh_query: str,
    quote_result: dict[str, Any],
) -> str:
    """Build a fresh-message query with compact resolved quote evidence."""
    quote_facts = _compact_substantive_fact_summaries(quote_result)
    query = (
        "Known evidence from the quoted/replied message is already retrieved.\n"
        "Do not create retrieval slots for the same quoted-message facts again.\n"
        "Retrieve only additional missing facts needed by the current user "
        "message.\n"
        "Use quoted-message evidence only when relevant to the current user "
        "message.\n\n"
        f"Known quoted-message evidence:\n{quote_facts}\n\n"
        f"Current user message:\n{fresh_query}"
    )
    return query


def _build_combined_retry_query(
    *,
    fresh_query: str,
    reply_excerpt: str,
) -> str:
    """Build the single fallback query after both first passes miss."""
    query = (
        "Resolve the current user message using the quoted/replied text as "
        "context.\n"
        "Treat quoted text as quoted material, not verified truth.\n\n"
        f"Quoted/replied text:\n{reply_excerpt}\n\n"
        f"Current fresh message:\n{fresh_query}"
    )
    return query


def _context_for_pass(context: dict[str, Any], original_query: str) -> dict[str, Any]:
    """Clone RAG context and align prompt body text to one pass query."""
    pass_context = copy.deepcopy(context)
    prompt_message_context = pass_context["prompt_message_context"]
    prompt_message_context["body_text"] = original_query
    return pass_context


async def _run_quote_aware_pass(
    *,
    pass_name: str,
    original_query: str,
    character_name: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Run one quote-aware RAG pass with pass-local message context."""
    pass_context = _context_for_pass(context, original_query)
    result = await call_rag_supervisor(
        original_query=original_query,
        character_name=character_name,
        context=pass_context,
    )
    known_facts = _known_facts_from(result)
    unknown_slots = _unknown_slots_from(result)
    logger.info(
        f"Quote-aware RAG pass output: pass={pass_name} "
        f"known_facts={len(known_facts)} unknown_slots={len(unknown_slots)} "
        f"loop_count={_loop_count_from(result)} "
        f"answer={log_preview(_answer_from(result))} "
        f"query={log_preview(original_query)}"
    )
    return result


def _known_facts_from(result: dict[str, Any]) -> list[Any]:
    """Return known facts from a RAG result when the shape is valid."""
    known_facts = result.get("known_facts", [])
    if isinstance(known_facts, list):
        return_value = known_facts
    else:
        return_value = []
    return return_value


def _unknown_slots_from(result: dict[str, Any]) -> list[Any]:
    """Return unknown slots from a RAG result when the shape is valid."""
    unknown_slots = result.get("unknown_slots", [])
    if isinstance(unknown_slots, list):
        return_value = unknown_slots
    else:
        return_value = []
    return return_value


def _loop_count_from(result: dict[str, Any]) -> int:
    """Return the integer loop count from a RAG result."""
    raw_loop_count = result.get("loop_count", 0)
    loop_count = int(raw_loop_count or 0)
    return loop_count


def _answer_from(result: dict[str, Any]) -> str:
    """Return answer text from a RAG result."""
    raw_answer = result.get("answer", "")
    if raw_answer is None:
        answer = ""
    else:
        answer = str(raw_answer)
    return answer


def _is_pure_person_display_resolution(fact: dict[str, Any]) -> bool:
    """Return whether a fact only resolved a display name identity."""
    if fact.get("agent") != "person_context_agent":
        return_value = False
        return return_value

    raw_result = fact.get("raw_result")
    if not isinstance(raw_result, dict):
        return_value = False
        return return_value

    primary_worker = raw_result.get("primary_worker")
    supporting_workers = raw_result.get("supporting_workers")
    return_value = (
        primary_worker == "user_lookup_agent"
        and not supporting_workers
    )
    return return_value


def _is_substantive_fact(fact: dict[str, Any]) -> bool:
    """Return whether a known fact is content-bearing resolved evidence."""
    resolved = bool(fact.get("resolved", False))
    if not resolved:
        return_value = False
        return return_value

    return_value = not _is_pure_person_display_resolution(fact)
    return return_value


def _has_substantive_facts(result: dict[str, Any]) -> bool:
    """Return whether a RAG result contains usable resolved evidence."""
    for fact in _known_facts_from(result):
        if not isinstance(fact, dict):
            continue
        if _is_substantive_fact(fact):
            return_value = True
            return return_value
    return_value = False
    return return_value


def _clip_text(text: str, *, limit: int) -> str:
    """Clip prompt-facing evidence text to the configured character budget."""
    if len(text) <= limit:
        return_value = text
        return return_value
    clipped = text[: limit - 3].rstrip()
    return_value = f"{clipped}..."
    return return_value


def _compact_substantive_fact_summaries(result: dict[str, Any]) -> str:
    """Render resolved quote facts without copying raw helper payloads."""
    lines = []
    for fact in _known_facts_from(result):
        if not isinstance(fact, dict):
            continue
        if not _is_substantive_fact(fact):
            continue
        agent = str(fact.get("agent", "")).strip()
        slot = str(fact.get("slot", "")).strip()
        summary = str(fact.get("summary", "")).strip()
        line = f"- [{agent}] {slot}: {summary}".strip()
        lines.append(line)

    rendered = "\n".join(lines).strip()
    if not rendered:
        rendered = "- No resolved quoted-message facts."
    clipped = _clip_text(rendered, limit=FACT_SUMMARY_CHAR_LIMIT)
    return clipped


def _fact_key(fact: dict[str, Any]) -> tuple[str, str]:
    """Build the merge key for one known-fact row."""
    slot = str(fact.get("slot", ""))
    agent = str(fact.get("agent", ""))
    key = (slot, agent)
    return key


def _merge_known_facts(pass_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge known facts in pass order, preferring resolved duplicates."""
    merged_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, str]] = []
    for pass_result in pass_results:
        result = pass_result["result"]
        for fact in _known_facts_from(result):
            if not isinstance(fact, dict):
                continue
            key = _fact_key(fact)
            if key not in merged_by_key:
                merged_by_key[key] = fact
                ordered_keys.append(key)
                continue

            existing = merged_by_key[key]
            existing_resolved = bool(existing.get("resolved", False))
            incoming_resolved = bool(fact.get("resolved", False))
            if incoming_resolved and not existing_resolved:
                merged_by_key[key] = fact

    merged_facts = [merged_by_key[key] for key in ordered_keys]
    return merged_facts


def _merge_unknown_slots(pass_results: list[dict[str, Any]]) -> list[str]:
    """Merge string unknown slots in first-seen order."""
    seen_slots: set[str] = set()
    merged_slots: list[str] = []
    for pass_result in pass_results:
        result = pass_result["result"]
        for slot in _unknown_slots_from(result):
            if not isinstance(slot, str):
                continue
            if slot in seen_slots:
                continue
            seen_slots.add(slot)
            merged_slots.append(slot)
    return merged_slots


def _total_loop_count(pass_results: list[dict[str, Any]]) -> int:
    """Sum RAG supervisor loop counts across executed passes."""
    loop_count = 0
    for pass_result in pass_results:
        result = pass_result["result"]
        loop_count += _loop_count_from(result)
    return loop_count


def _choose_answer(pass_results: list[dict[str, Any]]) -> str:
    """Choose the latest non-empty answer from a substantive pass."""
    selected_answer = ""
    for pass_result in pass_results:
        result = pass_result["result"]
        answer = _answer_from(result).strip()
        if not answer:
            continue
        if _has_substantive_facts(result):
            selected_answer = answer

    if selected_answer:
        return_value = selected_answer
        return return_value

    for pass_result in pass_results:
        result = pass_result["result"]
        answer = _answer_from(result).strip()
        if answer:
            selected_answer = answer

    return selected_answer


def _merge_pass_results(pass_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge executed RAG passes into the public supervisor result shape."""
    merged_result = {
        "answer": _choose_answer(pass_results),
        "known_facts": _merge_known_facts(pass_results),
        "unknown_slots": _merge_unknown_slots(pass_results),
        "loop_count": _total_loop_count(pass_results),
    }
    logger.info(
        f"Quote-aware RAG merged output: passes="
        f"{[pass_result['pass_name'] for pass_result in pass_results]} "
        f"known_facts={len(merged_result['known_facts'])} "
        f"unknown_slots={len(merged_result['unknown_slots'])} "
        f"loop_count={merged_result['loop_count']} "
        f"answer={log_preview(merged_result['answer'])}"
    )
    return merged_result


async def call_quote_aware_rag_supervisor(
    *,
    fresh_query: str,
    reply_context: dict[str, Any],
    character_name: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Run quote-aware RAG sequencing around the existing RAG supervisor.

    Args:
        fresh_query: Current decontextualized user message.
        reply_context: Native reply metadata, including optional quote text.
        character_name: Active character name passed through to RAG.
        context: RAG runtime context built by the cognitive episode adapter.

    Returns:
        Public RAG supervisor result shape consumed by persona projection.
    """
    reply_excerpt = _reply_excerpt_text(reply_context)
    if not reply_excerpt:
        result = await call_rag_supervisor(
            original_query=fresh_query,
            character_name=character_name,
            context=context,
        )
        logger.info(
            f"Quote-aware RAG pass output: pass=fresh_only "
            f"known_facts={len(_known_facts_from(result))} "
            f"unknown_slots={len(_unknown_slots_from(result))} "
            f"loop_count={_loop_count_from(result)} "
            f"answer={log_preview(_answer_from(result))}"
        )
        return result

    pass_results: list[dict[str, Any]] = []
    quote_query = _build_quote_grounding_query(reply_excerpt)
    quote_result = await _run_quote_aware_pass(
        pass_name="quote_grounding",
        original_query=quote_query,
        character_name=character_name,
        context=context,
    )
    pass_results.append(
        {
            "pass_name": "quote_grounding",
            "result": quote_result,
        }
    )

    if _has_substantive_facts(quote_result):
        fresh_pass_query = _build_fresh_query_with_quote_facts(
            fresh_query=fresh_query,
            quote_result=quote_result,
        )
    else:
        fresh_pass_query = fresh_query

    fresh_result = await _run_quote_aware_pass(
        pass_name="fresh_after_quote",
        original_query=fresh_pass_query,
        character_name=character_name,
        context=context,
    )
    pass_results.append(
        {
            "pass_name": "fresh_after_quote",
            "result": fresh_result,
        }
    )

    if not any(
        _has_substantive_facts(pass_result["result"])
        for pass_result in pass_results
    ):
        retry_query = _build_combined_retry_query(
            fresh_query=fresh_query,
            reply_excerpt=reply_excerpt,
        )
        retry_result = await _run_quote_aware_pass(
            pass_name="combined_retry",
            original_query=retry_query,
            character_name=character_name,
            context=context,
        )
        pass_results.append(
            {
                "pass_name": "combined_retry",
                "result": retry_result,
            }
        )

    merged_result = _merge_pass_results(pass_results)
    return merged_result
