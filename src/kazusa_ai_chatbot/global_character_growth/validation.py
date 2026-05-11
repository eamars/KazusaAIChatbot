"""Deterministic validation for global character growth candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from hashlib import sha256
import re
from typing import Any

from kazusa_ai_chatbot.global_character_growth.drift import (
    evidence_strength_for_candidate,
)
from kazusa_ai_chatbot.global_character_growth.models import (
    ALLOWED_GROWTH_AXES,
    DUPLICATE_OVERLAP_THRESHOLD,
    MAX_ACCEPTED_CANDIDATES,
    MAX_GUIDANCE_CHARS,
    MAX_TRAIT_NAME_CHARS,
    PROMPT_VERSION,
    AcceptedCandidate,
    MemoryCard,
    RejectedCandidate,
    ValidatedCandidateSet,
)

_SOURCE_TOKEN_PATTERNS = (
    re.compile(r"\bqq:\d+\b", re.IGNORECASE),
    re.compile(r"\bmemory-[A-Za-z0-9_.:-]+\b", re.IGNORECASE),
    re.compile(r"\brun-[A-Za-z0-9_.:-]+\b", re.IGNORECASE),
    re.compile(r"\breflection[-_ ]run[-_ ]?[A-Za-z0-9_.:-]*\b", re.IGNORECASE),
    re.compile(r"\bmessage[-_ ]?id\b", re.IGNORECASE),
    re.compile(r"\buser[-_ ]?id\b", re.IGNORECASE),
)
_DOMAIN_TOPIC_TOKENS = {
    "python",
    "async",
    "debug",
    "debugging",
    "technology",
    "product",
    "tea",
    "food",
    "cooking",
    "location",
    "hobby",
    "技术",
    "科技",
    "编程",
    "代码",
    "调试",
    "产品",
    "食物",
    "食材",
    "菜谱",
    "烹饪",
    "做饭",
    "茶",
    "地点",
    "城市",
    "爱好",
    "工具知识",
    "领域熟练度",
}


def validate_candidate_response(
    *,
    parsed_response: Mapping[str, Any],
    memory_cards: Sequence[MemoryCard],
    current_trait_rows: Sequence[Mapping[str, Any]],
) -> ValidatedCandidateSet:
    """Validate parsed LLM output and return accepted/rejected candidates."""

    accepted_candidates: list[AcceptedCandidate] = []
    rejected_candidates: list[RejectedCandidate] = []
    validation_warnings: list[str] = []
    raw_candidates = parsed_response.get("candidate_deltas", [])
    if not isinstance(raw_candidates, list):
        validation_warnings.append("candidate_deltas must be a list")
        raw_candidates = []

    cards_by_id = {
        card["source_card_id"]: card
        for card in memory_cards
    }
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, Mapping):
            rejected_candidates.append(_reject({}, "candidate must be an object"))
            continue
        candidate, reason = _validate_one(raw_candidate, cards_by_id)
        if reason:
            rejected_candidates.append(_reject(raw_candidate, reason))
            continue
        assert candidate is not None
        duplicate_reason = _duplicate_reason(
            candidate,
            current_trait_rows=current_trait_rows,
            accepted_candidates=accepted_candidates,
        )
        if duplicate_reason:
            rejected_candidates.append(_reject(raw_candidate, duplicate_reason))
            continue
        if len(accepted_candidates) >= MAX_ACCEPTED_CANDIDATES:
            rejected_candidates.append(_reject(raw_candidate, "accepted_candidate_cap"))
            continue
        accepted_candidates.append(candidate)

    result: ValidatedCandidateSet = {
        "accepted_candidates": accepted_candidates,
        "rejected_candidates": rejected_candidates,
        "validation_warnings": validation_warnings,
    }
    return result


def _validate_one(
    raw_candidate: Mapping[str, Any],
    cards_by_id: Mapping[str, MemoryCard],
) -> tuple[AcceptedCandidate | None, str]:
    action = str(raw_candidate.get("candidate_action", ""))
    if action == "no_action":
        reason = str(raw_candidate.get("rejection_reason", "")) or "no_action"
        return (None, reason)
    if action != "observe_trait":
        return (None, "candidate_action")

    growth_axis = str(raw_candidate.get("growth_axis", ""))
    if growth_axis not in ALLOWED_GROWTH_AXES:
        return (None, "growth_axis")
    if str(raw_candidate.get("scope_assessment", "")) != "global":
        return (None, "scope_assessment")
    if str(raw_candidate.get("private_detail_risk", "")) != "low":
        return (None, "private_detail_risk")

    trait_name = str(raw_candidate.get("trait_name", "")).strip()
    if not trait_name or len(trait_name) > MAX_TRAIT_NAME_CHARS:
        return (None, "trait_name")
    guidance = str(raw_candidate.get("guidance", "")).strip()
    if not guidance or len(guidance) > MAX_GUIDANCE_CHARS:
        return (None, "guidance")
    if not _contains_cjk(trait_name) or not _contains_cjk(guidance):
        return (None, "language")
    candidate_text = f"{trait_name} {guidance}"
    if _contains_source_detail(candidate_text):
        return (None, "source_detail")
    if _contains_domain_topic(candidate_text):
        return (None, "domain_topic")

    source_card_ids = _unique_strings(raw_candidate.get("source_card_ids", []))
    if len(source_card_ids) != len(raw_candidate.get("source_card_ids", [])):
        return (None, "source_card_ids")
    if len(source_card_ids) < 2:
        return (None, "source_card_ids")
    missing_cards = [
        card_id for card_id in source_card_ids
        if card_id not in cards_by_id
    ]
    if missing_cards:
        return (None, "source_card_ids")

    supporting_dates = _unique_strings(raw_candidate.get("supporting_dates", []))
    if len(supporting_dates) != len(raw_candidate.get("supporting_dates", [])):
        return (None, "supporting_dates")
    if len(supporting_dates) < 2:
        return (None, "supporting_dates")
    allowed_dates = {
        source_date
        for card_id in source_card_ids
        for source_date in cards_by_id[card_id]["character_local_dates"]
    }
    if not set(supporting_dates).issubset(allowed_dates):
        return (None, "supporting_dates")

    source_memory_ids = [
        cards_by_id[card_id]["memory_unit_id"]
        for card_id in source_card_ids
    ]
    source_run_ids = [
        run_id
        for card_id in source_card_ids
        for run_id in cards_by_id[card_id]["source_reflection_run_ids"]
    ]
    candidate: AcceptedCandidate = {
        "candidate_id": _candidate_id(
            growth_axis=growth_axis,
            guidance=guidance,
            source_memory_unit_ids=source_memory_ids,
        ),
        "growth_axis": growth_axis,
        "trait_name": trait_name,
        "guidance": guidance,
        "source_card_ids": source_card_ids,
        "supporting_dates": supporting_dates,
        "source_memory_unit_ids": _unique_sorted(source_memory_ids),
        "source_reflection_run_ids": _unique_sorted(source_run_ids),
        "support_level": str(raw_candidate.get("support_level", "")),
        "confidence": str(raw_candidate.get("confidence", "")),
        "evidence_strength": evidence_strength_for_candidate(raw_candidate),
        "novelty_reason": str(raw_candidate.get("novelty_reason", "")),
        "stability_reason": str(raw_candidate.get("stability_reason", "")),
    }
    return (candidate, "")


def _duplicate_reason(
    candidate: AcceptedCandidate,
    *,
    current_trait_rows: Sequence[Mapping[str, Any]],
    accepted_candidates: Sequence[AcceptedCandidate],
) -> str:
    for trait in current_trait_rows:
        if str(trait.get("status", "")) != "active":
            continue
        if str(trait.get("growth_axis", "")) != candidate["growth_axis"]:
            continue
        if _overlap(str(trait.get("guidance", "")), candidate["guidance"]):
            return_value = "duplicate_active_trait"
            return return_value
    for accepted in accepted_candidates:
        if accepted["growth_axis"] != candidate["growth_axis"]:
            continue
        if _overlap(accepted["guidance"], candidate["guidance"]):
            return_value = "duplicate_candidate"
            return return_value
    return_value = ""
    return return_value


def _overlap(left: str, right: str) -> bool:
    ratio = SequenceMatcher(None, _normalize(left), _normalize(right)).ratio()
    return_value = ratio >= DUPLICATE_OVERLAP_THRESHOLD
    return return_value


def _candidate_id(
    *,
    growth_axis: str,
    guidance: str,
    source_memory_unit_ids: Sequence[str],
) -> str:
    seed = "|".join([
        PROMPT_VERSION,
        growth_axis,
        _normalize(guidance),
        ",".join(_unique_sorted(source_memory_unit_ids)),
    ])
    digest = sha256(seed.encode("utf-8")).hexdigest()[:16]
    return_value = f"gcc_{digest}"
    return return_value


def _reject(raw_candidate: Mapping[str, Any], reason: str) -> RejectedCandidate:
    rejected: RejectedCandidate = {
        "growth_axis": str(raw_candidate.get("growth_axis", "")),
        "trait_name": str(raw_candidate.get("trait_name", "")),
        "guidance": str(raw_candidate.get("guidance", "")),
        "reason": reason,
        "source_card_ids": _unique_strings(raw_candidate.get("source_card_ids", [])),
    }
    return rejected


def _contains_source_detail(guidance: str) -> bool:
    for pattern in _SOURCE_TOKEN_PATTERNS:
        if pattern.search(guidance):
            return_value = True
            return return_value
    return_value = False
    return return_value


def _contains_domain_topic(guidance: str) -> bool:
    lowered = guidance.lower()
    return_value = any(token in lowered for token in _DOMAIN_TOPIC_TOKENS)
    return return_value


def _contains_cjk(value: str) -> bool:
    return_value = any("\u4e00" <= character <= "\u9fff" for character in value)
    return return_value


def _unique_strings(raw_values: object) -> list[str]:
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, str):
        return_value: list[str] = []
        return return_value
    values: list[str] = []
    for raw_value in raw_values:
        value = str(raw_value)
        if not value or value in values:
            continue
        values.append(value)
    return values


def _unique_sorted(values: Sequence[str]) -> list[str]:
    return_value = sorted({str(value) for value in values if str(value)})
    return return_value


def _normalize(value: str) -> str:
    return_value = " ".join(value.lower().split())
    return return_value
