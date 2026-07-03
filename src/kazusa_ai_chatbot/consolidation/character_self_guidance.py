"""Accepted conversation-derived character self-guidance lane."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    CONSOLIDATION_LLM_MODEL,
    CONSOLIDATION_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.memory_evolution.identity import (
    deterministic_memory_unit_id,
)
from kazusa_ai_chatbot.memory_evolution.models import (
    MemoryAuthority,
    MemorySourceKind,
    MemoryStatus,
)
from kazusa_ai_chatbot.memory_evolution.repository import insert_memory_unit
from kazusa_ai_chatbot.rag.prompt_projection import project_tool_result_for_llm
from kazusa_ai_chatbot.utils import parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

SELF_GUIDANCE_MEMORY_TYPE = "defense_rule"
_MAX_MEMORY_NAME_CHARS = 80
_MAX_CONTENT_CHARS = 500

_SPECIALIST_PROMPT = '''\
You are the character self-guidance specialist for consolidation.

Decide whether the routed sources contain durable future behavior guidance
owned by the active character. The guidance must be grounded in both a user
request or proposal and the character's acceptance in final dialog.

# Decision Procedure
1. Read the user request or proposal, final dialog, recent chat context, and
   source_refs.
2. Choose write when the accepted behavior belongs to the character generally
   rather than to one current-user obligation.
3. Choose no_action when acceptance is absent, the behavior is only temporary,
   or the durable owner is a different memory lane.
4. Write concise, general guidance. Preserve source-grounded meaning while
   omitting private user details that are not needed for future behavior.

# Output Format
Return only valid JSON:
{
  "action": "write | no_action",
  "memory_name": "short title for accepted character behavior guidance",
  "content": "general future behavior guidance owned by the character",
  "reason": "short reason"
}
'''

_REVIEW_PROMPT = '''\
You are the reviewer for one character self-guidance candidate.

Review whether the candidate is durable future behavior guidance owned by the
active character and grounded in the provided request plus final-dialog
acceptance.

# Review Procedure
1. Accept when the candidate is already grounded, general, and character-owned.
2. Revise only wording, title, or privacy-sensitive detail while preserving the
   accepted behavior.
3. Reject when the candidate lacks acceptance evidence, belongs to another
   memory lane, or adds behavior not supported by the supplied sources.

# Output Format
Return only valid JSON:
{
  "decision": "accept | revise | reject",
  "memory_name": "title to persist when accepted or revised",
  "content": "content to persist when accepted or revised",
  "reason": "short reason"
}
'''

_self_guidance_specialist_llm = LLInterface()
_self_guidance_reviewer_llm = LLInterface()
_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CONSOLIDATION_LLM",
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=0.1,
    top_p=0.9,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)


async def character_self_guidance_specialist(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract and review accepted character-owned self-guidance.

    Args:
        state: Consolidator state after lane source-policy acceptance.

    Returns:
        A state patch containing ``character_self_guidance`` when accepted.
    """

    source_refs = _source_refs_from_state(state)
    if not source_refs:
        return_value = {"character_self_guidance": {}}
        return return_value

    candidate = await _extract_self_guidance_candidate(state, source_refs)
    if not candidate:
        return_value = {"character_self_guidance": {}}
        return return_value

    reviewed_candidate = await _review_self_guidance_candidate(
        state,
        source_refs,
        candidate,
    )
    return_value = {"character_self_guidance": reviewed_candidate}
    return return_value


async def persist_character_self_guidance_from_state(
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Persist a reviewed character self-guidance candidate.

    Args:
        state: Consolidator state carrying a reviewed candidate and source refs.

    Returns:
        Stored memory mutation metadata, or ``None`` when no candidate exists.
    """

    candidate = state.get("character_self_guidance")
    if not isinstance(candidate, Mapping):
        return_value = None
        return return_value

    memory_name = text_or_empty(candidate.get("memory_name"))
    content = text_or_empty(candidate.get("content"))
    source_refs = _source_refs_from_state(state)
    if not memory_name or not content or not source_refs:
        return_value = None
        return return_value

    storage_timestamp_utc = text_or_empty(state.get("storage_timestamp_utc"))
    memory_doc = _memory_document(
        memory_name=memory_name,
        content=content,
        source_refs=source_refs,
        storage_timestamp_utc=storage_timestamp_utc,
    )
    stored = await insert_memory_unit(document=memory_doc)
    result = {
        "memory_unit_id": stored["memory_unit_id"],
        "lineage_id": stored["lineage_id"],
        "memory_type": stored["memory_type"],
        "memory_name": stored["memory_name"],
        "content": stored["content"],
    }
    return result


async def _extract_self_guidance_candidate(
    state: Mapping[str, Any],
    source_refs: list[dict[str, Any]],
) -> dict[str, str]:
    """Run the specialist LLM and validate its candidate shape."""

    payload = _prompt_payload(state, source_refs)
    response = await _self_guidance_specialist_llm.ainvoke(
        [
            SystemMessage(content=_SPECIALIST_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=_llm_config,
    )
    result = parse_llm_json_output(response.content)
    candidate = _normalize_specialist_result(result)
    return candidate


async def _review_self_guidance_candidate(
    state: Mapping[str, Any],
    source_refs: list[dict[str, Any]],
    candidate: Mapping[str, Any],
) -> dict[str, str]:
    """Run the reviewer LLM and validate the accepted candidate shape."""

    payload = _prompt_payload(state, source_refs)
    payload["candidate"] = project_tool_result_for_llm(candidate)
    response = await _self_guidance_reviewer_llm.ainvoke(
        [
            SystemMessage(content=_REVIEW_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=_llm_config,
    )
    result = parse_llm_json_output(response.content)
    reviewed_candidate = _normalize_reviewer_result(result, candidate)
    return reviewed_candidate


def _prompt_payload(
    state: Mapping[str, Any],
    source_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the prompt-safe payload for specialist and reviewer LLMs."""

    payload = {
        "timestamp": state.get("local_time_context", {}),
        "character_name": _character_name(state),
        "decontextualized_input": text_or_empty(
            state.get("decontexualized_input")
        ),
        "final_dialog": project_tool_result_for_llm(
            state.get("final_dialog", [])
        ),
        "chat_history_recent": project_tool_result_for_llm(
            state.get("chat_history_recent", [])
        ),
        "source_refs": project_tool_result_for_llm(source_refs),
    }
    return payload


def _normalize_specialist_result(result: Mapping[str, Any]) -> dict[str, str]:
    """Validate specialist output without semantic post-classification."""

    action = text_or_empty(result.get("action"))
    if action == "no_action":
        return_value: dict[str, str] = {}
        return return_value
    if action != "write":
        raise ValueError(f"invalid self-guidance action: {action!r}")

    candidate = _candidate_fields(result)
    return candidate


def _normalize_reviewer_result(
    result: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, str]:
    """Validate reviewer output without semantic post-classification."""

    decision = text_or_empty(result.get("decision"))
    if decision == "reject":
        return_value: dict[str, str] = {}
        return return_value
    if decision not in {"accept", "revise"}:
        raise ValueError(f"invalid self-guidance review decision: {decision!r}")

    if decision == "accept":
        memory_name = text_or_empty(result.get("memory_name")) or text_or_empty(
            candidate.get("memory_name")
        )
        content = text_or_empty(result.get("content")) or text_or_empty(
            candidate.get("content")
        )
        result = {
            **dict(result),
            "memory_name": memory_name,
            "content": content,
        }

    reviewed_candidate = _candidate_fields(result)
    reviewed_candidate["review_reason"] = text_or_empty(result.get("reason"))
    return reviewed_candidate


def _candidate_fields(result: Mapping[str, Any]) -> dict[str, str]:
    """Read and structurally validate candidate text fields."""

    memory_name = text_or_empty(result.get("memory_name"))
    content = text_or_empty(result.get("content"))
    if not memory_name:
        raise ValueError("self-guidance memory_name is required")
    if not content:
        raise ValueError("self-guidance content is required")

    candidate = {
        "memory_name": memory_name[:_MAX_MEMORY_NAME_CHARS].strip(),
        "content": content[:_MAX_CONTENT_CHARS].strip(),
        "memory_type": SELF_GUIDANCE_MEMORY_TYPE,
    }
    return candidate


def _source_refs_from_state(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return structurally valid self-guidance source refs from state."""

    raw_refs = state.get("character_self_guidance_source_refs")
    if not isinstance(raw_refs, list):
        return_value: list[dict[str, Any]] = []
        return return_value
    source_refs = [dict(ref) for ref in raw_refs if isinstance(ref, Mapping)]
    return source_refs


def _character_name(state: Mapping[str, Any]) -> str:
    """Return the active character display name for prompt context."""

    character_profile = state.get("character_profile")
    if isinstance(character_profile, Mapping):
        character_name = text_or_empty(character_profile.get("name"))
        if character_name:
            return character_name
    return_value = "the active character"
    return return_value


def _memory_document(
    *,
    memory_name: str,
    content: str,
    source_refs: list[dict[str, Any]],
    storage_timestamp_utc: str,
) -> dict[str, Any]:
    """Build one evolving memory document for accepted self-guidance."""

    source_key = json.dumps(
        source_refs,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    memory_unit_id = deterministic_memory_unit_id(
        "conversation",
        [SELF_GUIDANCE_MEMORY_TYPE, memory_name, content, source_key],
    )
    memory_doc = {
        "memory_unit_id": memory_unit_id,
        "lineage_id": memory_unit_id,
        "version": 1,
        "memory_name": memory_name,
        "content": content,
        "source_global_user_id": "",
        "memory_type": SELF_GUIDANCE_MEMORY_TYPE,
        "source_kind": MemorySourceKind.CONVERSATION_EXTRACTED,
        "authority": MemoryAuthority.CONVERSATION_ACCEPTED,
        "status": MemoryStatus.ACTIVE,
        "supersedes_memory_unit_ids": [],
        "merged_from_memory_unit_ids": [],
        "evidence_refs": source_refs,
        "privacy_review": {
            "private_detail_risk": "low",
            "user_details_removed": True,
            "boundary_assessment": (
                "Accepted conversation-derived character self-guidance."
            ),
            "reviewer": "automated_llm",
        },
        "confidence_note": (
            "Accepted by the character in final dialog and reviewed for "
            "character-owned self-guidance."
        ),
        "timestamp": storage_timestamp_utc,
    }
    return memory_doc
