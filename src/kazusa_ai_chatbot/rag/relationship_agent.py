"""RAG helper agent: read character relationship rankings."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import RAG_SUBAGENT_LLM_API_KEY, RAG_SUBAGENT_LLM_BASE_URL, RAG_SUBAGENT_LLM_MODEL
from kazusa_ai_chatbot.db import list_users_by_affinity
from kazusa_ai_chatbot.rag.cache2_policy import (
    RELATIONSHIP_CACHE_NAME,
    build_relationship_cache_key,
    build_relationship_dependencies,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.utils import build_affinity_block, get_llm, parse_llm_json_output, text_or_empty

_MAX_RELATIONSHIP_LIMIT = 5
_RELATIONSHIP_MODES = {"one", "n", "existence"}
_RELATIONSHIP_RANK_ORDERS = {"top", "bottom"}

_NEGATIVE_LABELS = {
    "Contemptuous",
    "Scornful",
    "Hostile",
    "Antagonistic",
    "Aloof",
    "Reserved",
    "Formal",
    "Cold",
    "Detached",
}
_POSITIVE_LABELS = {
    "Receptive",
    "Approachable",
    "Friendly",
    "Warm",
    "Caring",
    "Affectionate",
    "Devoted",
    "Protective",
    "Fiercely Loyal",
    "Unwavering",
}

_EXTRACTOR_PROMPT = """\
You are a parameter extractor for `relationship_agent`.

# Capability
This agent ranks profiled users by the character's internal relationship score.
It is for questions like:
- who the character likes most / favorite person / closest person
- whether there is a liked person
- who the character dislikes or hates most
- top-N relationship ranking

# Arguments
- mode:
  - "one": one strongest matching candidate
  - "n": ranked list with the requested count
  - "existence": whether any matching candidate exists
- rank_order:
  - "top": highest relationship score
  - "bottom": lowest relationship score
- limit:
  - Preserve explicit requested count.
  - Use 1 for mode="one".
  - Use 3 for mode="existence" unless the task explicitly requests another count.

# Generation Procedure
1. Read `task` and decide whether it asks for one user, a ranked list, or existence.
2. Choose `rank_order`: top for liked/closest/favorite, bottom for disliked/hated.
3. Preserve explicit count in `limit`; otherwise use the mode defaults.
4. Ignore tasks that require conversation evidence or persona interpretation beyond relationship ranking.

# Input Format
{
  "task": "slot description from the outer RAG supervisor",
  "context": "known facts and runtime hints"
}

# Output Format
Return valid JSON only:
{
  "mode": "one | n | existence",
  "rank_order": "top | bottom",
  "limit": 1
}
"""

_extractor_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


def _normalize_relationship_args(raw_args: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize extractor output into safe relationship-ranking arguments.

    Args:
        raw_args: Parsed JSON object from the relationship extractor LLM.

    Returns:
        Sanitized request dict, or ``None`` when required values are out of schema.
    """
    mode = text_or_empty(raw_args.get("mode"))
    rank_order = text_or_empty(raw_args.get("rank_order"))
    raw_limit = raw_args.get("limit")
    if mode not in _RELATIONSHIP_MODES or rank_order not in _RELATIONSHIP_RANK_ORDERS:
        return None
    if not isinstance(raw_limit, int) or isinstance(raw_limit, bool):
        return None

    limit = max(1, min(raw_limit, _MAX_RELATIONSHIP_LIMIT))
    if mode == "one":
        limit = 1

    return_value = {
        "mode": mode,
        "rank_order": rank_order,
        "limit": limit,
    }
    return return_value


async def _extract_relationship_args(
    task: str,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract relationship-ranking parameters from the agent task.

    Args:
        task: Relationship slot description produced by the outer supervisor.
        context: Runtime hints passed to the helper agent.

    Returns:
        Sanitized relationship args, or ``None`` when extraction failed.
    """
    system_prompt = SystemMessage(content=_EXTRACTOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps({"task": task, "context": context}, ensure_ascii=False, default=str)
    )
    response = await _extractor_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(response.content)
    if not isinstance(result, dict):
        return None
    return_value = _normalize_relationship_args(result)
    return return_value


def _relationship_band(label: str) -> str:
    """Classify a public affinity label into a coarse relationship band.

    Args:
        label: Label returned by ``build_affinity_block``.

    Returns:
        ``"positive"``, ``"negative"``, or ``"neutral"``.
    """
    if label in _POSITIVE_LABELS:
        return "positive"
    if label in _NEGATIVE_LABELS:
        return "negative"
    return "neutral"


def _public_candidate(doc: dict[str, Any], rank: int) -> dict[str, Any]:
    """Convert an affinity-ranked profile row into a prompt-safe candidate.

    Args:
        doc: Raw relationship row returned by the DB helper.
        rank: One-based rank after affinity sorting.

    Returns:
        Candidate payload with no raw affinity value.
    """
    affinity_block = build_affinity_block(doc["affinity"])
    label = affinity_block["level"]
    return_value = {
        "rank": rank,
        "global_user_id": doc["global_user_id"],
        "display_name": doc["display_name"],
        "platform": doc["platform"],
        "platform_user_id": doc["platform_user_id"],
        "relationship_label": label,
        "relationship_band": _relationship_band(label),
    }
    return return_value


class RelationshipAgent(BaseRAGHelperAgent):
    """RAG helper agent that ranks users by the character's relationship data.

    Args:
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(self, *, cache_runtime=None) -> None:
        super().__init__(
            name="relationship_agent",
            cache_name=RELATIONSHIP_CACHE_NAME,
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Return prompt-safe relationship candidates for the character.

        Args:
            task: Relationship slot description from the supervisor.
            context: Runtime hints supplying platform scope.
            max_attempts: Unused; kept for interface compatibility.

        Returns:
            Dict with resolved flag, sanitized relationship candidates, and
            cache metadata. Raw affinity values are never included.
        """
        del max_attempts

        args = await _extract_relationship_args(task, context)
        if args is None:
            return_value = self.with_cache_status(
                {
                    "resolved": False,
                    "result": "Could not extract relationship ranking parameters from task.",
                    "attempts": 1,
                },
                hit=False,
                reason="skipped_unextractable_relationship_args",
            )
            return return_value

        cache_key = build_relationship_cache_key(args, context)
        cached = await self.read_cache(cache_key)
        if cached is not None:
            return_value = self.with_cache_status(
                {"resolved": True, "result": cached, "attempts": 0},
                hit=True,
                reason="hit",
                cache_key=cache_key,
            )
            return return_value

        platform = str(context.get("platform") or "").strip() or None
        rows = await list_users_by_affinity(
            rank_order=args["rank_order"],
            platform=platform,
            limit=args["limit"],
        )
        candidates = [
            _public_candidate(row, rank)
            for rank, row in enumerate(rows, start=1)
        ]

        result = {
            "mode": args["mode"],
            "rank_order": args["rank_order"],
            "source": "user_profiles_relationship_rank",
            "candidates": candidates,
        }
        if args["mode"] == "existence":
            matching_band = "positive" if args["rank_order"] == "top" else "negative"
            result["has_matching_candidate"] = any(
                candidate["relationship_band"] == matching_band
                for candidate in candidates
            )

        if candidates:
            await self.write_cache(
                cache_key=cache_key,
                result=result,
                dependencies=build_relationship_dependencies(context),
                metadata={
                    "mode": args["mode"],
                    "rank_order": args["rank_order"],
                },
            )

        return_value = self.with_cache_status(
            {
                "resolved": bool(candidates),
                "result": result,
                "attempts": 1,
            },
            hit=False,
            reason="miss_stored" if candidates else "miss_unresolved",
            cache_key=cache_key,
        )
        return return_value
