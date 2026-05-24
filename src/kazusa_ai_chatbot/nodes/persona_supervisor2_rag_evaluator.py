"""Evaluation, continuation, and finalization for the RAG supervisor."""

from __future__ import annotations

import copy
import json
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.config import (
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.continuation import (
    MAX_CONTINUATION_DECISIONS_PER_RAG_RUN,
    RAGContinuationDecision,
    empty_continuation_decision,
    validate_refined_query,
)
from kazusa_ai_chatbot.rag.evidence_formatting import (
    sanitize_public_rag_evidence_text,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_initializer import (
    _normalize_initializer_slots,
    rag_initializer,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_prompt_views import (
    _clip_llm_summary_text,
    _compact_raw_result_for_llm,
    _known_facts_llm_view,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_types import (
    _MAX_LOOP_COUNT,
    ProgressiveRAGState,
)
from kazusa_ai_chatbot.utils import (
    get_llm,
    log_list_preview,
    log_preview,
    parse_llm_json_output,
)

logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000
RAG_EVALUATOR_COMPONENT = "nodes.persona_supervisor2_rag_evaluator"


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed monotonic milliseconds since a start marker."""

    elapsed = time.perf_counter() - started_at
    elapsed_ms = max(0, int(elapsed * MILLISECONDS_PER_SECOND))
    return elapsed_ms


def _state_correlation_id(state: ProgressiveRAGState) -> str:
    """Build a non-content correlation id for RAG evaluator work."""

    context = state.get("context", {})
    if isinstance(context, dict):
        platform = str(context.get("platform", ""))
        message_ref = str(context.get("platform_message_id", "") or "")
    else:
        platform = ""
        message_ref = ""
    correlation_id = f"rag:{platform}:{message_ref or 'no-message-id'}"
    return correlation_id


# ── Evaluator ──────────────────────────────────────────────────────

_EVALUATOR_SUMMARIZER_PROMPT = '''\
You distill one RAG2 slot result into a concise factual summary for later
retrieval agents and the final factual synthesizer.

# Generation Procedure
1. Read `slot` and `agent` to identify the evidence target answered by this
   result.
2. Read `raw_result` and extract only facts and readable sources already
   present there.
3. Use `known_facts` only to avoid repeating earlier slot conclusions.
4. If `resolved` is false or `raw_result` lacks usable evidence, state what
   this source did not confirm without broadening the conclusion.
5. Use English for generated control words and source-status prose. Preserve
   display names, quotes, URLs, filenames, code/model labels, and message text
   in their original source language.

# Summary Requirements
- Keep readable sources such as display_name and URL. Do not include
  global_user_id or source ids.
- For conversation rows, list 1-10 most relevant message summaries as
  speaker plus key content.
- For profile or durable memory payloads, extract the key facts.
- For unresolved slots, briefly state that this retrieval source did not
  confirm the slot.
- If raw_result is empty, do not infer that previous slots failed unless
  known_facts explicitly shows that.

# Input Format
The human payload is JSON:
{
    "slot": "current slot description",
    "agent": "agent name that executed the slot",
    "resolved": true,
    "raw_result": "raw tool output; dict/list/string/null",
    "known_facts": [{"slot": "...", "agent": "...", "resolved": true, "summary": "...", "raw_result": "...", "attempts": 1}]
}

# Recall Results
- If `agent` is `recall_agent` and `raw_result.selected_summary` exists,
  preserve its core content.
- You may add `primary_source` and `supporting_sources`, but do not treat
  progress-only recall as long-term fact storage.

# Output Format
- Plain text only, no JSON wrapper.
- Keep under 200 Chinese characters or 120 English words.
'''

_EVALUATOR_SUMMARIZER_USER_PROFILE_PROMPT = '''\
You distill a Profile/Person-context slot result into a concise factual
summary. The result may describe the current user, a third-party user, or the
active character's own public profile.

# Field Semantics
- `user_memory_context` is the unified current-user memory projection. Each
  row may include `fact`, `subjective_appraisal`, and `relationship_signal`;
  treat these as fact anchor, character-side appraisal, and future interaction
  signal respectively.
- `objective_facts`, `milestones`, and `active_commitments`, when present,
  are sections inside `user_memory_context`, not legacy standalone profile
  sources.
- If raw_result contains name/description/gender/age/birthday/backstory or
  self_image without user_memory_context, it is the active character's own
  public profile or self-image. Summarize it as character profile data, not a
  third-party user profile.

# Generation Procedure
1. Read `slot`, `agent`, and `known_facts` to decide whether this profile
   result belongs to the current user, a third-party user, or the active
   character.
2. If `raw_result` contains `user_memory_context`, summarize memory-unit
   facts first, then character-side appraisal and relationship signal.
3. If `raw_result` is public character profile or `self_image`, summarize it
   as the character's own profile.
4. Use only information present in raw_result. Leave unknown fields unknown.
5. Use English for generated control/status prose. Preserve source content,
   names, and quotes in their original source language.

# Summary Requirements
- Keep useful identifiers such as global_user_id and display_name when needed
  by later steps.
- Distinguish the target user, fact anchors, and character-side appraisal.
- Do not fill unknown information.

# Input Format
The human payload is JSON:
{
    "slot": "current Profile slot description",
    "agent": "user_profile_agent",
    "resolved": true,
    "raw_result": {
        "global_user_id": "user UUID",
        "display_name": "user display name",
        "user_memory_context": {
            "stable_patterns": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "local YYYY-MM-DD HH:MM"}],
            "recent_shifts": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "local YYYY-MM-DD HH:MM"}],
            "objective_facts": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "local YYYY-MM-DD HH:MM"}],
            "milestones": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "local YYYY-MM-DD HH:MM"}],
            "active_commitments": [{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "local YYYY-MM-DD HH:MM", "due_at": "optional local due time YYYY-MM-DD HH:MM", "due_state": "no_due_date | future_due | due_today | past_due | unknown_due_date"}]
        }
    },
    "known_facts": [{"slot": "...", "agent": "...", "resolved": true, "summary": "...", "raw_result": "...", "attempts": 1}]
}

# Output Format
- Plain text only, no JSON wrapper.
- Keep under 220 Chinese characters or 140 English words.
'''

_evaluator_summarizer_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)

async def _summarize_agent_result(
    slot: str,
    agent_name: str,
    resolved: bool,
    raw_result: object,
    known_facts: list[dict],
) -> str:
    """Distil a resolved agent result into a concise fact summary for downstream agents.

    Only called when resolved=True. Unresolved slots receive a deterministic
    template in rag_evaluator and never reach this function.

    Args:
        slot: The slot description that was being resolved.
        agent_name: Inner-loop agent that produced the raw result.
        resolved: Whether the inner-loop agent judged the slot as resolved.
        raw_result: Native tool output from the inner-loop agent (dict, list, str, or None).
        known_facts: Facts resolved before this slot.

    Returns:
        A concise factual summary with English RAG2 control prose.
    """
    if agent_name == "user_profile_agent":
        prompt = _EVALUATOR_SUMMARIZER_USER_PROFILE_PROMPT
    else:
        prompt = _EVALUATOR_SUMMARIZER_PROMPT

    compact_raw_result = _compact_raw_result_for_llm(raw_result)
    compact_known_facts = _known_facts_llm_view(known_facts)
    system_prompt = SystemMessage(content=prompt)
    human_message = HumanMessage(
        content=json.dumps(
            {
                "slot": slot,
                "agent": agent_name,
                "resolved": resolved,
                "raw_result": compact_raw_result,
                "known_facts": compact_known_facts,
            },
            ensure_ascii=False,
            default=str,
        )
    )
    started_at = time.perf_counter()
    response = await _evaluator_summarizer_llm.ainvoke([system_prompt, human_message])
    await event_logging.record_llm_stage_event(
        component=RAG_EVALUATOR_COMPONENT,
        stage_name="rag_result_summarizer",
        route_name=agent_name,
        model_name=RAG_SUBAGENT_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_prompt.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status="not_json",
        retry_count=0,
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
    )
    return_value = response.content.strip()
    return return_value


def _unresolved_summary(slot: str, raw_result: object) -> str:
    """Build deterministic unresolved text without hiding candidate rows."""

    if isinstance(raw_result, dict):
        observation_candidates = _compact_continuation_items(
            raw_result.get('observation_candidates')
        )
        if observation_candidates:
            preview_parts = []
            for candidate in observation_candidates[:3]:
                content = candidate.get('content') or candidate.get('summary')
                source = candidate.get('source')
                if content and source:
                    preview_parts.append(f'{source}: {content}')
                    continue
                if content:
                    preview_parts.append(content)
            preview = '; '.join(preview_parts)
            if preview:
                summary = (
                    f'Retrieved candidates, but none were confirmed enough '
                    f'to resolve the slot. Slot: {slot}. Candidates: {preview}'
                )
                return summary
            summary = (
                f'Retrieved candidates, but none were confirmed enough '
                f'to resolve the slot. Slot: {slot}'
            )
            return summary

        missing_context = _compact_continuation_text_list(
            raw_result.get('missing_context')
        )
        if missing_context:
            incompatible_routes = []
            for item in missing_context:
                if not item.startswith('incompatible_intent:'):
                    continue
                _, _, route = item.partition(':')
                if route:
                    incompatible_routes.append(route)

            if incompatible_routes:
                route_text = ', '.join(incompatible_routes)
                summary = (
                    f'Retrieval source mismatch; this slot should be handled '
                    f'by {route_text}. This is not evidence that no record '
                    f'exists. Slot: {slot}'
                )
                return summary

            context_text = ', '.join(missing_context)
            summary = (
                f'Retrieval lacked required context and did not confirm the '
                f'slot. Slot: {slot}. Missing: {context_text}'
            )
            return summary

    summary = f'Retrieval returned no relevant confirmed result. Slot: {slot}'
    return summary


_CONTINUATION_OBSERVATION_LIMIT = 5
_CONTINUATION_OBSERVATION_TEXT_LIMIT = 500
_CONTINUATION_KNOWN_FACT_LIMIT = 5


def _compact_continuation_mapping(value: object) -> dict[str, str]:
    """Build a compact observation row for the continuation assessor."""
    if not isinstance(value, dict):
        text = _clip_llm_summary_text(
            value,
            limit=_CONTINUATION_OBSERVATION_TEXT_LIMIT,
        )
        compact_text = {"content": text} if text else {}
        return compact_text

    compact: dict[str, str] = {}
    for key in (
        "kind",
        "source",
        "reason",
        "content",
        "summary",
        "text",
        "fact",
        "description",
        "source_kind",
        "source_system",
        "memory_name",
        "missing_context",
    ):
        if key not in value:
            continue
        text = _clip_llm_summary_text(
            value[key],
            limit=_CONTINUATION_OBSERVATION_TEXT_LIMIT,
        )
        if text:
            compact[key] = text

    return_value = compact
    return return_value


def _compact_continuation_items(value: object) -> list[dict[str, str]]:
    """Clip observation-like rows before they enter the assessor prompt."""
    if isinstance(value, list):
        raw_items = value[:_CONTINUATION_OBSERVATION_LIMIT]
    elif value:
        raw_items = [value]
    else:
        raw_items = []

    compact_items: list[dict[str, str]] = []
    for raw_item in raw_items:
        compact_item = _compact_continuation_mapping(raw_item)
        if compact_item:
            compact_items.append(compact_item)

    return_value = compact_items
    return return_value


def _compact_continuation_text_list(value: object) -> list[str]:
    """Build a compact list of source-scoped hint strings."""
    if isinstance(value, list):
        raw_items = value[:_CONTINUATION_OBSERVATION_LIMIT]
    elif value:
        raw_items = [value]
    else:
        raw_items = []

    compact_items: list[str] = []
    for raw_item in raw_items:
        text = _clip_llm_summary_text(
            raw_item,
            limit=_CONTINUATION_OBSERVATION_TEXT_LIMIT,
        )
        if text:
            compact_items.append(text)

    return_value = compact_items
    return return_value


def _continuation_observation_payload(
    *,
    state: ProgressiveRAGState,
    slot: str,
    agent_name: str,
    raw_result: object,
    remaining_slots: list[str],
) -> dict[str, object]:
    """Project unresolved retrieval output into assessor-visible observations."""
    observation_candidates: list[dict[str, str]] = []
    source_hints: list[dict[str, str]] = []
    missing_context: list[str] = []
    conflicts: list[str] = []
    source_policy = ""
    user_resolution_hints: list[str] = []

    if isinstance(raw_result, dict):
        observation_candidates = _compact_continuation_items(
            raw_result.get("observation_candidates"),
        )
        source_hints = _compact_continuation_items(
            raw_result.get("source_hints"),
        )
        missing_context = _compact_continuation_text_list(
            raw_result.get("missing_context"),
        )
        conflicts = _compact_continuation_text_list(raw_result.get("conflicts"))
        source_policy = _clip_llm_summary_text(
            raw_result.get("source_policy", ""),
            limit=_CONTINUATION_OBSERVATION_TEXT_LIMIT,
        )
        user_resolution_hints = _compact_continuation_text_list(
            raw_result.get("user_resolution_hints"),
        )

    payload = {
        "original_query": state.get("original_query", ""),
        "current_slot": slot,
        "agent": agent_name,
        "resolved": False,
        "source_policy": source_policy,
        "missing_context": missing_context,
        "conflicts": conflicts,
        "observation_candidates": observation_candidates,
        "source_hints": source_hints,
        "user_resolution_hints": user_resolution_hints,
        "known_facts": _known_facts_llm_view(
            state.get("known_facts", []),
        )[-_CONTINUATION_KNOWN_FACT_LIMIT:],
        "pending_slots": remaining_slots[:_CONTINUATION_OBSERVATION_LIMIT],
    }
    return payload


def _has_continuation_observation(payload: dict[str, object]) -> bool:
    """Return whether the assessor has enough material to justify one LLM call."""
    has_material = bool(
        payload["observation_candidates"]
        or payload["source_hints"]
        or payload["user_resolution_hints"]
    )
    return has_material


def _slot_prefix(slot: object) -> str:
    """Return the semantic prefix before a planned RAG slot separator."""

    slot_text = str(slot).strip()
    prefix, _, _ = slot_text.partition(":")
    normalized_prefix = prefix.casefold()
    return normalized_prefix


def _pending_slots_can_absorb_continuation(
    payload: dict[str, object],
) -> bool:
    """Return whether queued evidence slots should run before re-entry."""

    pending_slots = payload["pending_slots"]
    if not isinstance(pending_slots, list):
        return_value = False
        return return_value

    evidence_prefixes = {"conversation-evidence", "memory-evidence"}
    can_absorb = any(
        _slot_prefix(slot) in evidence_prefixes
        for slot in pending_slots
    )
    return can_absorb


def _has_resolved_known_fact(known_facts: list[dict]) -> bool:
    """Return whether the current RAG run already found usable evidence."""

    has_resolved_fact = any(
        fact.get("resolved") is True
        for fact in known_facts
    )
    return has_resolved_fact


def _memory_miss_after_recall_should_finalize(
    payload: dict[str, object],
) -> bool:
    """Return whether Recall plus memory miss is enough to finalize."""

    if payload["agent"] != "memory_evidence_agent":
        return_value = False
        return return_value

    known_facts = payload["known_facts"]
    if not isinstance(known_facts, list):
        return_value = False
        return return_value

    has_unresolved_recall = False
    for fact in known_facts:
        if not isinstance(fact, dict):
            continue
        if fact.get("resolved") is True:
            return_value = False
            return return_value
        if fact.get("agent") == "recall_agent" and fact.get("resolved") is False:
            has_unresolved_recall = True

    return_value = has_unresolved_recall
    return return_value


def _accepted_continuation_count(known_facts: list[dict]) -> int:
    """Count prior accepted refined-query re-entries in this RAG run."""
    count = 0
    for fact in known_facts:
        continuation = fact.get("continuation")
        if not isinstance(continuation, dict):
            continue
        if (
            continuation.get("should_continue") is True
            and continuation.get("refined_query")
        ):
            count += 1
    return count


def _previous_refined_queries(known_facts: list[dict]) -> list[str]:
    """Collect accepted refined queries already used in this RAG run."""
    refined_queries: list[str] = []
    for fact in known_facts:
        continuation = fact.get("continuation")
        if not isinstance(continuation, dict):
            continue

        refined_query = continuation.get("refined_query")
        if isinstance(refined_query, str) and refined_query.strip():
            refined_queries.append(refined_query.strip())

    return refined_queries


def _refined_initializer_context(
    context: dict,
    refined_query: str,
) -> dict:
    """Build active-query context for a refined initializer re-entry."""
    refined_context = copy.deepcopy(context)
    prompt_message_context = refined_context.get("prompt_message_context")
    if isinstance(prompt_message_context, dict):
        prompt_message_context["body_text"] = refined_query

    return refined_context


_CONTINUATION_ASSESSOR_PROMPT = '''\
You decide whether unresolved retrieval observations should be folded into a
new self-contained user query and sent back to the existing initializer.
Do not answer the user. Do not generate slots, prefixes, agent names, tool
names, database parameters, or step lists.

# Decision Order
1. First decide whether continuation must stop. If any stop condition applies,
   return should_continue=false and refined_query="".
2. Stop when observations are unrelated, noisy, failure-only, or provide no
   information that can improve the next query.
3. Stop when observations say the answer depends on user-provided use case,
   budget, preference, scope, permission, time, or location that is still
   missing.
4. Stop when the next query would only be a placeholder such as "based on my
   use case/budget/preferences".
5. Stop when observations are merely side candidates in the same broad
   category, such as other commitments, other food, other devices, other
   images, or other historical events, but do not cover the original target's
   key object, time, source, or event. Do not enumerate just to prove absence.
6. Stop when the requested source was already searched and only
   related/nearby/unconfirmed candidates remain, with no new source, message
   position, time range, person identity, or executable constraint.
7. Continue when the current source is memory_evidence but the slot target is
   clearly media shared in conversation, such as an image, photo, screenshot,
   attachment, illustration, or OCR description. These usually belong in
   conversation evidence rather than durable memory.
8. Otherwise continue only when observations provide immediately usable
   knowledge, source direction, constraints, or retrieval strategy.
9. If true, refined_query must be self-contained natural language including
   the original target and useful observation material. Do not rely on hidden
   context.
10. Use English for generated control/status prose. Preserve names, quotes,
   URLs, filenames, and source text in their original source language.

# Input Format
The user message is JSON:
{
  "original_query": "decontextualized user question",
  "current_slot": "slot that just failed",
  "agent": "agent that produced the unresolved result",
  "resolved": false,
  "source_policy": "current source selection note",
  "missing_context": ["missing context from current source"],
  "conflicts": ["conflicts found by current source"],
  "observation_candidates": [{"content": "observation that did not answer but may help"}],
  "source_hints": [{"kind": "hint kind", "source": "hint source"}],
  "user_resolution_hints": ["constraints that require user input"],
  "known_facts": [{"slot": "...", "agent": "...", "resolved": true, "summary": "..."}],
  "pending_slots": ["slots already waiting"]
}

# Output Format
Return valid JSON only:
{
  "should_continue": true,
  "refined_query": "self-contained natural-language query when continuing; empty string when stopping",
  "reason": "short diagnostic note for logs and trace only"
}
'''
_continuation_assessor_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


async def _assess_continuation(
    *,
    observation_payload: dict[str, object],
    original_query: str,
    previous_refined_queries: list[str],
    continuation_count: int,
) -> RAGContinuationDecision:
    """Classify an unresolved observation and validate refined-query re-entry."""
    if not _has_continuation_observation(observation_payload):
        decision = empty_continuation_decision()
        return decision

    if _pending_slots_can_absorb_continuation(observation_payload):
        decision: RAGContinuationDecision = {
            "should_continue": False,
            "refined_query": "",
            "reason": (
                "Existing pending evidence slots should run before "
                "initializer re-entry."
            ),
        }
        logger.info(
            f"RAG2 continuation skipped: "
            f"reason={log_preview(decision['reason'])}"
        )
        return decision

    if _memory_miss_after_recall_should_finalize(observation_payload):
        decision = {
            "should_continue": False,
            "refined_query": "",
            "reason": (
                "Recall and memory evidence were both unresolved; finalize "
                "the negative result instead of broad query expansion."
            ),
        }
        logger.info(
            f"RAG2 continuation skipped: "
            f"reason={log_preview(decision['reason'])}"
        )
        return decision

    if continuation_count >= MAX_CONTINUATION_DECISIONS_PER_RAG_RUN:
        decision = empty_continuation_decision()
        return decision

    system_prompt = SystemMessage(content=_CONTINUATION_ASSESSOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(
            observation_payload,
            ensure_ascii=False,
            default=str,
        )
    )
    started_at = time.perf_counter()
    response = await _continuation_assessor_llm.ainvoke(
        [system_prompt, human_message],
    )
    raw_decision = parse_llm_json_output(response.content)
    parse_status = "succeeded" if isinstance(raw_decision, dict) else "failed"
    decision = validate_refined_query(
        raw_decision,
        original_query=original_query,
        previous_refined_queries=previous_refined_queries,
        continuation_count=continuation_count,
        max_continuations=MAX_CONTINUATION_DECISIONS_PER_RAG_RUN,
    )
    logger.info(
        f"RAG2 continuation assessor output: "
        f"should_continue={decision['should_continue']} "
        f"refined_query={log_preview(decision['refined_query'])} "
        f"reason={log_preview(decision['reason'])}"
    )
    logger.debug(
        f"RAG2 continuation assessor metadata: "
        f"payload={log_preview(observation_payload)} "
        f"raw_decision={log_preview(raw_decision)} "
        f"validated={log_preview(decision)}"
    )
    await event_logging.record_llm_stage_event(
        component=RAG_EVALUATOR_COMPONENT,
        stage_name="continuation_assessor",
        route_name="continuation",
        model_name=RAG_SUBAGENT_LLM_MODEL,
        status="succeeded" if parse_status == "succeeded" else "failed",
        prompt_chars=len(system_prompt.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status=parse_status,
        retry_count=0,
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info" if parse_status == "succeeded" else "warning",
    )
    return decision


async def _initialize_refined_query_slots(
    *,
    state: ProgressiveRAGState,
    refined_query: str,
) -> list[str]:
    """Run the existing initializer path on a self-contained refined query."""
    refined_context = _refined_initializer_context(
        state.get("context", {}),
        refined_query,
    )
    initializer_state: ProgressiveRAGState = {
        "original_query": refined_query,
        "character_name": state.get("character_name", ""),
        "context": refined_context,
        "unknown_slots": [],
        "current_slot": "",
        "known_facts": state.get("known_facts", []),
        "messages": [],
        "initializer_cache": {},
        "current_dispatch": {},
        "last_agent_result": {},
        "loop_count": state.get("loop_count", 0),
        "final_answer": "",
    }
    initializer_update = await rag_initializer(initializer_state)
    refined_slots = _normalize_initializer_slots(
        initializer_update.get("unknown_slots", []),
    )
    logger.info(
        f"RAG2 continuation initializer output: "
        f"refined_query={log_preview(refined_query)} "
        f"unknown_slots={log_list_preview(refined_slots)}"
    )
    logger.debug(
        f"RAG2 continuation initializer metadata: "
        f"cache={log_preview(initializer_update.get('initializer_cache', {}))}"
    )
    return refined_slots


async def rag_evaluator(state: ProgressiveRAGState) -> dict:
    """Record the inner-agent verdict for the current slot, then drain it.

    The slot is always removed from unknown_slots regardless of success. The
    evaluator normalizes the agent result, calls the summarizer to produce a
    concise fact summary, and stores both the summary and raw result.

    Args:
        state: Current state after the executor ran.

    Returns:
        Partial state update with the current slot removed from unknown_slots
        and the agent verdict (with summary and raw_result) appended to known_facts.
    """
    agent_result = state.get("last_agent_result", {})
    if not isinstance(agent_result, dict):
        agent_result = {}

    slot = state.get("current_slot", "")
    resolved = bool(agent_result.get("resolved", False))
    raw_result = agent_result.get("result")
    known_facts = state.get("known_facts", [])
    agent_name = str(agent_result.get("agent", ""))
    remaining_slots = list(state.get("unknown_slots", []))[1:]

    if resolved:
        summary = await _summarize_agent_result(
            slot,
            agent_name,
            resolved,
            raw_result,
            known_facts,
        )
    else:
        summary = _unresolved_summary(slot, raw_result)

    continuation_decision: RAGContinuationDecision | None = None
    loop_count = int(state.get("loop_count", 0) or 0)
    if not resolved:
        continuation_decision = empty_continuation_decision()
        observation_payload = _continuation_observation_payload(
            state=state,
            slot=slot,
            agent_name=agent_name,
            raw_result=raw_result,
            remaining_slots=remaining_slots,
        )
        if loop_count < _MAX_LOOP_COUNT:
            if _has_resolved_known_fact(known_facts):
                continuation_decision = {
                    "should_continue": False,
                    "refined_query": "",
                    "reason": (
                        "Existing resolved evidence should be finalized before "
                        "secondary query expansion."
                    ),
                }
                logger.info(
                    f"RAG2 continuation skipped: "
                    f"reason={log_preview(continuation_decision['reason'])}"
                )
            else:
                continuation_decision = await _assess_continuation(
                    observation_payload=observation_payload,
                    original_query=state.get("original_query", ""),
                    previous_refined_queries=_previous_refined_queries(known_facts),
                    continuation_count=_accepted_continuation_count(known_facts),
                )

    new_fact = {
        "slot": slot,
        "agent": agent_name,
        "resolved": resolved,
        "summary": summary,
        "raw_result": raw_result,
        "attempts": int(agent_result.get("attempts", 0) or 0),
    }
    if continuation_decision is not None:
        new_fact["continuation"] = continuation_decision

    if (
        continuation_decision is not None
        and continuation_decision["should_continue"]
        and continuation_decision["refined_query"]
    ):
        refined_slots = await _initialize_refined_query_slots(
            state=state,
            refined_query=continuation_decision["refined_query"],
        )
        remaining_slots = refined_slots + remaining_slots

    logger.info(
        f'RAG2 fact: slot={log_preview(slot)} agent={agent_name or "<none>"} '
        f"resolved={resolved} summary={log_preview(summary)}"
    )
    logger.debug(
        f'RAG2 fact metadata: attempts={new_fact["attempts"]} '
        f"remaining_slots={len(remaining_slots)} "
        f"continuation={log_preview(continuation_decision)}"
    )
    logger.debug(f"RAG2 fact detail: raw_result={log_preview(raw_result)}")

    return_value = {
        "unknown_slots": remaining_slots,
        "known_facts": known_facts + [new_fact],
    }
    return return_value


# ── Finalizer ──────────────────────────────────────────────────────

_FINALIZER_PROMPT = '''\
You are a factual synthesizer. Generate a short factual summary from
`known_facts`.

# Generation Procedure
1. Read `original_query` to identify the fact types this summary must cover.
2. Read `time_context`; interpret relative dates such as today, yesterday, or
   two days ago using the character-local date.
3. Read `known_facts` in order. Use only summaries and raw_result from
   resolved slots.
4. If user_profile_agent raw_result contains user_memory_context, distinguish
   fact, subjective_appraisal, and relationship_signal.
5. If a necessary slot is unresolved, state which part is missing.
6. If agent="recall_agent", prefer raw_result.selected_summary for agreement,
   commitment, or episode-position facts.
7. Produce one short factual summary. Speakers, sources, times, and quotes
   must come from visible known_facts content.
8. Use English for generated control/status prose. Preserve source message
   text, display names, quoted text, URLs, filenames, and code/model labels in
   their original source language.

# Rules
- Organize the summary around facts needed by `original_query`; do not repeat
  the search process.
- If known_facts is empty, state that this RAG run had no external/internal
  facts to retrieve. Do not claim that specific information is missing.
- If a slot is unresolved, state the missing part truthfully.
- Do not broaden one source miss into "no record exists" or "no interaction
  exists"; only state what the searched source returned.
- Preserve readable URL or conversation source references where possible.
- For conversation evidence, summarize as visible source/speaker label plus
  time plus content. If no visible label exists, use "speaker".
- When quoting message text, preserve the quote's original pronouns.
- When known_facts has agent="user_profile_agent" and raw_result contains
  user_memory_context, fact is the fact anchor, subjective_appraisal is the
  profile source's character-side appraisal, and relationship_signal is the
  future interaction signal. Do not miswrite subjective_appraisal as the target
  user's own feeling.
- When known_facts has agent="user_profile_agent" and raw_result is public
  profile or self_image, that is the subject's own profile data. Use it for
  self-profile questions, not as third-party user profile memory.
- When known_facts has agent="recall_agent" and raw_result contains
  selected_summary, it is the arbitrated recall result for current agreement,
  commitment, or progress. Use selected_summary directly; do not turn it into
  a long-term setting or search-keyword rewrite.

# Input Format
{
    "original_query": "user original question",
    "time_context": {"current_local_datetime": "YYYY-MM-DD HH:MM", "current_local_weekday": "Weekday"},
    "known_facts": [{"slot": ..., "agent": ..., "resolved": ..., "summary": "concise fact summary", "raw_result": "raw tool output when needed for quotes", "attempts": ...}, ...]
}

# Output Format
Return one natural-language factual summary as plain text with no JSON wrapper.
- no markdown formatting
- preserve visible source/speaker labels
- no broad interpretation beyond short extractive summaries
'''
_finalizer_llm = get_llm(
    temperature=0.2,
    top_p=0.9,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)
_FINALIZER_UNRESOLVED_STATUS_LIMIT = 4
_FINALIZER_UNRESOLVED_CANDIDATE_LIMIT = 6
_FINALIZER_UNRESOLVED_CANDIDATE_TEXT_LIMIT = 180


def _finalizer_time_context(state: ProgressiveRAGState) -> dict[str, object]:
    """Return prompt-facing local time context for the finalizer."""

    context = state.get("context", {})
    if not isinstance(context, dict):
        return_value: dict[str, object] = {}
        return return_value

    time_context = context.get("time_context")
    if not isinstance(time_context, dict):
        return_value = {}
        return return_value

    allowed_keys = ("current_local_datetime", "current_local_weekday")
    return_value = {
        key: time_context[key]
        for key in allowed_keys
        if isinstance(time_context.get(key), str)
    }
    return return_value


def _all_known_facts_unresolved(known_facts: object) -> bool:
    """Return whether every collected RAG fact is unresolved."""

    if not isinstance(known_facts, list):
        return_value = False
        return return_value

    if not known_facts:
        return_value = False
        return return_value

    for fact in known_facts:
        if not isinstance(fact, dict):
            return_value = False
            return return_value
        if fact.get("resolved") is not False:
            return_value = False
            return return_value

    return_value = True
    return return_value


def _unresolved_fact_label(fact: dict) -> str:
    """Return a compact source label for an unresolved fact."""

    agent = _clip_llm_summary_text(
        fact.get("agent", ""),
        limit=80,
    )
    if agent:
        return agent

    slot = _clip_llm_summary_text(
        fact.get("slot", ""),
        limit=80,
    )
    if slot:
        return slot

    return_value = "unknown_source"
    return return_value


def _unresolved_fact_status(fact: dict) -> str:
    """Describe why one fact is still unresolved without dumping candidates."""

    label = _unresolved_fact_label(fact)
    raw_result = fact.get("raw_result")
    if isinstance(raw_result, dict):
        missing_context = _compact_continuation_text_list(
            raw_result.get("missing_context"),
        )
        incompatible_routes = []
        for item in missing_context:
            if not item.startswith("incompatible_intent:"):
                continue
            _, _, route = item.partition(":")
            if route:
                incompatible_routes.append(route)

        if incompatible_routes:
            route_text = ", ".join(incompatible_routes)
            status = f"{label}: source mismatch; should be handled by {route_text}"
            return status

        has_candidates = bool(
            raw_result.get("observation_candidates")
            or raw_result.get("candidates")
        )
        if has_candidates:
            status = f"{label}: candidates found but not confirmed enough to answer"
            return status

        if missing_context:
            context_text = ", ".join(missing_context)
            status = f"{label}: missing {context_text}"
            return status

    status = f"{label}: no confirmed result returned"
    return status


def _candidate_preview_text(candidate: object) -> str:
    """Return one concise candidate text from a raw unresolved candidate row."""

    if isinstance(candidate, dict):
        text = ""
        for key in ("content", "summary", "claim", "fact", "text"):
            if key in candidate:
                text = _clip_llm_summary_text(
                    candidate[key],
                    limit=_FINALIZER_UNRESOLVED_CANDIDATE_TEXT_LIMIT,
                )
                break
        if not text:
            return_value = ""
            return return_value

        source = _clip_llm_summary_text(
            candidate.get("source", ""),
            limit=60,
        )
        if source:
            preview = f"{source}: {text}"
            return preview
        return text

    preview = _clip_llm_summary_text(
        candidate,
        limit=_FINALIZER_UNRESOLVED_CANDIDATE_TEXT_LIMIT,
    )
    return preview


def _candidate_previews_from_facts(known_facts: list[dict]) -> list[str]:
    """Collect unresolved candidate text from the provided facts."""

    previews: list[str] = []
    seen: set[str] = set()
    for fact in known_facts:
        raw_result = fact.get("raw_result")
        if not isinstance(raw_result, dict):
            continue

        raw_candidates = raw_result.get("observation_candidates")
        if not raw_candidates:
            raw_candidates = raw_result.get("candidates")
        if not isinstance(raw_candidates, list):
            continue

        for candidate in raw_candidates:
            preview = _candidate_preview_text(candidate)
            if not preview:
                continue
            if preview in seen:
                continue
            seen.add(preview)
            previews.append(preview)
            if len(previews) >= _FINALIZER_UNRESOLVED_CANDIDATE_LIMIT:
                return previews

    return previews


def _unresolved_candidate_previews(known_facts: list[dict]) -> list[str]:
    """Collect the most relevant nearby candidates for a negative result."""

    recall_facts = [
        fact
        for fact in known_facts
        if fact.get("agent") == "recall_agent"
    ]
    recall_previews = _candidate_previews_from_facts(recall_facts)
    if recall_previews:
        return recall_previews

    previews = _candidate_previews_from_facts(known_facts)
    return previews


def _unresolved_finalizer_answer(known_facts: list[dict]) -> str:
    """Summarize all-unresolved RAG output without raw slot dumps."""

    status_lines = [
        _unresolved_fact_status(fact)
        for fact in known_facts[:_FINALIZER_UNRESOLVED_STATUS_LIMIT]
    ]
    status_lines = [line for line in status_lines if line]
    candidate_previews = _unresolved_candidate_previews(known_facts)

    final_parts = ["This RAG run found no confirmed facts."]
    if status_lines:
        status_text = "; ".join(status_lines)
        final_parts.append(f" Checked sources: {status_text}.")
    if candidate_previews:
        candidate_text = "; ".join(candidate_previews)
        final_parts.append(f" Nearby but unconfirmed candidates: {candidate_text}.")

    final_answer = "".join(final_parts)
    return final_answer


def _finalizer_known_facts_llm_view(
    known_facts: object,
) -> list[dict[str, object]]:
    """Build finalizer input without unresolved candidate evidence leakage."""

    compact_facts = _known_facts_llm_view(known_facts)
    finalizer_facts: list[dict[str, object]] = []
    for fact in compact_facts:
        finalizer_fact = dict(fact)
        if finalizer_fact.get("resolved") is not False:
            finalizer_facts.append(finalizer_fact)
            continue

        raw_result = finalizer_fact.get("raw_result")
        missing_context: list[str] = []
        selected_summary = ""
        if isinstance(raw_result, dict):
            missing_context = _compact_continuation_text_list(
                raw_result.get("missing_context"),
            )
            selected_summary = _clip_llm_summary_text(
                raw_result.get("selected_summary", ""),
            )

        finalizer_fact["raw_result"] = {
            "missing_context": missing_context,
            "selected_summary": selected_summary,
        }
        finalizer_facts.append(finalizer_fact)

    return_value = finalizer_facts
    return return_value


async def rag_finalizer(state: ProgressiveRAGState) -> dict:
    """Synthesise the final answer from all collected slot results.

    Args:
        state: Final state after all slots have been processed.

    Returns:
        Partial state update with final_answer set.
    """
    known_facts = state.get("known_facts", [])
    if _all_known_facts_unresolved(known_facts):
        final_answer = _unresolved_finalizer_answer(known_facts)
        final_answer = sanitize_public_rag_evidence_text(final_answer)
        logger.info(
            "RAG2 finalizer deterministic unresolved output: "
            f"answer={log_preview(final_answer)}"
        )
        return_value = {"final_answer": final_answer}
        return return_value

    system_prompt = SystemMessage(content=_FINALIZER_PROMPT)
    finalizer_input = {
        "original_query": state["original_query"],
        "time_context": _finalizer_time_context(state),
        "known_facts": _finalizer_known_facts_llm_view(known_facts),
    }
    human_message = HumanMessage(content=json.dumps(finalizer_input, ensure_ascii=False, default=str))

    started_at = time.perf_counter()
    response = await _finalizer_llm.ainvoke([system_prompt, human_message])
    logger.info(f"RAG2 finalizer output: answer={log_preview(response.content)}")
    logger.debug(
        f'RAG2 finalizer metadata: query={log_preview(state["original_query"])} '
        f'facts={len(state.get("known_facts", []))}'
    )
    await event_logging.record_llm_stage_event(
        component=RAG_EVALUATOR_COMPONENT,
        stage_name="rag_finalizer",
        route_name="finalize",
        model_name=RAG_SUBAGENT_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_prompt.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status="not_json",
        retry_count=0,
        json_repair_used=False,
        correlation_id=_state_correlation_id(state),
        duration_ms=_elapsed_ms(started_at),
    )
    final_answer = sanitize_public_rag_evidence_text(response.content)
    return_value = {"final_answer": final_answer}
    return return_value
