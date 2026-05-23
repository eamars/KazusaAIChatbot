"""Deterministic coverage checks for RAG evidence payloads."""

from __future__ import annotations

import re
from typing import Literal, TypedDict

from kazusa_ai_chatbot.utils import text_or_empty

EvidenceQuality = Literal["confirmed", "partial", "nearby", "missing"]

_MAX_REQUESTED_ITEMS = 8
_MAX_ITEM_CHARS = 80
_QUOTE_PATTERNS = (
    re.compile(r'"([^"\n]{1,120})"'),
    re.compile(r"'([^'\n]{1,120})'"),
    re.compile(r"\u201c([^\u201d\n]{1,120})\u201d"),
    re.compile(r"\u2018([^\u2019\n]{1,120})\u2019"),
    re.compile(r"\u300c([^\u300d\n]{1,120})\u300d"),
    re.compile(r"\u300e([^\u300f\n]{1,120})\u300f"),
)
_URL_RE = re.compile(r"https?://[^\s)>\]}\"']+")
_SPEAKER_SCOPE_RE = re.compile(
    r"\bspeaker\s*=\s*(?:"
    r"current_user|active_character|any_speaker|"
    r"person\s+resolved\s+in\s+slot\s+\d+"
    r")\b",
    flags=re.IGNORECASE,
)
_CAPITALIZED_PHRASE_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9+._-]*"
    r"(?:\s+[A-Z][A-Za-z0-9+._-]*)*"
)
_NORMALIZE_RE = re.compile(r"[^0-9a-z\u3400-\u9fff]+", flags=re.IGNORECASE)
_ASCII_ALNUM_RE = re.compile(r"[a-z0-9]", flags=re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_CJK_ALNUM_CONNECTIVE_RE = re.compile(
    r"(?<=[\u3400-\u9fff])(?:\u7684|\u4e4b)(?=[a-z0-9])"
    r"|(?<=[a-z0-9])(?:\u7684|\u4e4b)(?=[\u3400-\u9fff])",
    flags=re.IGNORECASE,
)
_VALUE_EVIDENCE_RE = re.compile(
    r"(?:[$￥¥]\s*\d[\d,.]*"
    r"|\d[\d,.]*\s*(?:元|刀|usd|aud|nzd|rmb|块|k|w|万)"
    r"|\d{3,}(?:[,.]\d+)*)",
    flags=re.IGNORECASE,
)
_VALUE_TASK_RE = re.compile(
    r"\b(?:price|prices|pricing|cost|costs|quoted|quote|value)\b"
    r"|\u591a\u5c11\u94b1|\u4ef7\u683c|\u62a5\u4ef7|\u552e\u4ef7",
    flags=re.IGNORECASE,
)
_CAPABILITY_PREFIX_RE = re.compile(r"^[A-Za-z-]+:\s*")
_TARGET_REGION_RE = re.compile(
    r"\b(?:"
    r"about|around|for|regarding|relevant\s+to|"
    r"mentioning|mentions?|containing|contains?|"
    r"exact\s+term|who\s+said|said"
    r")\b\s*(?P<body>.+)",
    flags=re.IGNORECASE,
)
_MULTI_TARGET_CONNECTIVE_RE = re.compile(
    r"\b(?:and|or|versus|vs\.?|respectively)\b"
    r"|[,;]"
    r"|[\u3001\uff0c\uff1b]"
    r"|\u548c|\u4e0e|\u53ca|\u5206\u522b",
    flags=re.IGNORECASE,
)
_TARGET_STOPWORDS = {
    "a",
    "about",
    "active",
    "any",
    "character",
    "channel",
    "chat",
    "containing",
    "conversation",
    "current",
    "discord",
    "discussion",
    "dm",
    "durable",
    "evidence",
    "exact",
    "find",
    "for",
    "group",
    "groups",
    "live",
    "memory",
    "person",
    "platform",
    "price",
    "private",
    "qq",
    "recall",
    "retrieve",
    "said",
    "server",
    "speaker",
    "source",
    "telegram",
    "term",
    "the",
    "url",
    "user",
    "web",
    "wechat",
    "who",
}


class EvidenceCoverage(TypedDict):
    """Public deterministic coverage shape attached to RAG evidence payloads."""

    requested_items: list[str]
    covered_items: list[str]
    missing_items: list[str]
    evidence_quality: EvidenceQuality
    confidence: float
    reason: str


def assess_evidence_coverage(
    *,
    task: str,
    evidence_items: list[str],
    worker_resolved: bool,
    requires_value_evidence: bool | None = None,
) -> EvidenceCoverage:
    """Assess whether selected evidence covers the semantic targets in a task.

    Args:
        task: RAG slot text being answered.
        evidence_items: Prompt-facing evidence summaries selected by a worker.
        worker_resolved: Whether the worker judged its retrieval result as
            sufficient before deterministic coverage validation.
        requires_value_evidence: Optional caller-supplied value evidence
            requirement when context outside the task preserves this intent.

    Returns:
        A public coverage mapping with requested, covered, and missing items,
        an evidence-quality label, confidence, and a short reason.
    """

    clean_evidence = [
        evidence
        for item in evidence_items
        if (evidence := text_or_empty(item))
    ]
    requested_items = requested_coverage_items(task)
    if requires_value_evidence is None:
        value_evidence_required = task_requires_value_evidence(task)
    else:
        value_evidence_required = requires_value_evidence
    covered_items = [
        item
        for item in requested_items
        if _item_is_covered(
            item,
            clean_evidence,
            requires_value_evidence=value_evidence_required,
        )
    ]
    missing_items = [
        item
        for item in requested_items
        if item not in covered_items
    ]

    evidence_quality = _evidence_quality(
        requested_items=requested_items,
        covered_items=covered_items,
        clean_evidence=clean_evidence,
        worker_resolved=worker_resolved,
    )
    confidence = _coverage_confidence(
        evidence_quality=evidence_quality,
        requested_items=requested_items,
        covered_items=covered_items,
    )
    reason = _coverage_reason(
        evidence_quality=evidence_quality,
        requested_items=requested_items,
        covered_items=covered_items,
        missing_items=missing_items,
        worker_resolved=worker_resolved,
    )
    coverage: EvidenceCoverage = {
        "requested_items": requested_items,
        "covered_items": covered_items,
        "missing_items": missing_items,
        "evidence_quality": evidence_quality,
        "confidence": confidence,
        "reason": reason,
    }
    return coverage


def requested_coverage_items(task: str) -> list[str]:
    """Extract concrete semantic targets that should be covered together.

    Args:
        task: RAG slot text.

    Returns:
        Ordered target strings. The extractor is intentionally conservative:
        quoted literals, URLs, and capitalized proper-name phrases are retained,
        while generic routing words and scope annotations are discarded.
    """

    task_body = _task_body(task)
    candidates: list[str] = []
    for pattern in _QUOTE_PATTERNS:
        for match in pattern.finditer(task_body):
            _append_unique_item(candidates, match.group(1))

    for match in _URL_RE.finditer(task_body):
        _append_unique_item(candidates, match.group(0))

    target_text = _target_region_for_names(task_body)
    for match in _CAPITALIZED_PHRASE_RE.finditer(target_text):
        _append_unique_item(candidates, match.group(0))

    return candidates


def has_explicit_multi_target_request(task: str) -> bool:
    """Return whether a task names multiple targets that must all be present.

    Ordinary semantic memory slots can resolve through a worker's judgment even
    when the selected summary does not repeat the slot wording. Explicit
    multi-target slots need stricter structural coverage so one retrieved row is
    not treated as proof for every named target.
    """

    requested_items = requested_coverage_items(task)
    if len(requested_items) < 2:
        return_value = False
        return return_value

    task_body = _strip_scope_annotations(_task_body(task))
    target_text = _target_region_for_names(task_body)
    if _MULTI_TARGET_CONNECTIVE_RE.search(target_text):
        return_value = True
        return return_value

    explicit_items = _explicit_anchor_items(task_body)
    return_value = len(explicit_items) >= 2
    return return_value


def task_requires_value_evidence(task: str) -> bool:
    """Return whether a task asks for value-bearing evidence per target."""

    task_body = _task_body(task)
    requires_value = _VALUE_TASK_RE.search(task_body) is not None
    return_value = requires_value
    return return_value


def coverage_allows_resolution(coverage: EvidenceCoverage) -> bool:
    """Return whether coverage is strong enough for ``resolved=True``."""

    allows_resolution = coverage["evidence_quality"] == "confirmed"
    return allows_resolution


def evidence_buckets_for_coverage(
    coverage: EvidenceCoverage,
    evidence_items: list[str],
) -> dict[str, list[str]]:
    """Split evidence summaries into confirmed, partial, or nearby buckets.

    Args:
        coverage: Coverage assessment for the same evidence.
        evidence_items: Selected evidence summaries.

    Returns:
        Mapping with ``confirmed_evidence``, ``partial_evidence``, and
        ``nearby_evidence`` lists. The original evidence text is preserved.
    """

    clean_evidence = [
        evidence
        for item in evidence_items
        if (evidence := text_or_empty(item))
    ]
    buckets = {
        "confirmed_evidence": [],
        "partial_evidence": [],
        "nearby_evidence": [],
    }
    quality = coverage["evidence_quality"]
    if quality == "confirmed":
        buckets["confirmed_evidence"] = clean_evidence
    elif quality == "partial":
        buckets["partial_evidence"] = clean_evidence
    elif quality == "nearby":
        buckets["nearby_evidence"] = clean_evidence
    return buckets


def _task_body(task: str) -> str:
    """Return slot text without the top-level capability prefix."""

    body = _CAPABILITY_PREFIX_RE.sub("", text_or_empty(task)).strip()
    return body


def _strip_scope_annotations(task_body: str) -> str:
    """Remove routing annotations that are not semantic targets."""

    stripped = _SPEAKER_SCOPE_RE.sub("", task_body)
    return stripped


def _target_region_for_names(task_body: str) -> str:
    """Return the task region where unquoted proper-name anchors may appear."""

    stripped = _strip_scope_annotations(task_body)
    stripped = _remove_explicit_anchor_spans(stripped)
    match = _TARGET_REGION_RE.search(stripped)
    if match is not None:
        target_text = match.group("body")
    else:
        target_text = stripped
    return target_text


def _explicit_anchor_items(task_body: str) -> list[str]:
    """Extract quoted and URL anchors without using capitalized phrases."""

    candidates: list[str] = []
    for pattern in _QUOTE_PATTERNS:
        for match in pattern.finditer(task_body):
            _append_unique_item(candidates, match.group(1))

    for match in _URL_RE.finditer(task_body):
        _append_unique_item(candidates, match.group(0))

    return candidates


def _remove_explicit_anchor_spans(task_body: str) -> str:
    """Remove quoted and URL spans before proper-name extraction."""

    stripped = task_body
    for pattern in _QUOTE_PATTERNS:
        stripped = pattern.sub(" ", stripped)
    stripped = _URL_RE.sub(" ", stripped)
    return stripped


def _append_unique_item(items: list[str], raw_item: str) -> None:
    """Append a cleaned target item when it is useful and new."""

    item = _clean_item(raw_item)
    if not item or item in items:
        return
    if len(items) >= _MAX_REQUESTED_ITEMS:
        return
    items.append(item)


def _clean_item(raw_item: str) -> str:
    """Normalize one requested-item candidate."""

    item = text_or_empty(raw_item).strip(" \t\r\n,.;:，。；：()[]{}")
    if len(item) > _MAX_ITEM_CHARS:
        item = item[:_MAX_ITEM_CHARS].rstrip()
    if len(item) < 2:
        return ""
    normalized_words = [
        word.lower()
        for word in re.findall(r"[A-Za-z]+", item)
    ]
    if (
        normalized_words
        and all(word in _TARGET_STOPWORDS for word in normalized_words)
    ):
        return ""
    lowered = item.lower()
    if lowered.startswith(("speaker=", "slot ")):
        return ""
    return item


def _item_is_covered(
    item: str,
    evidence_items: list[str],
    *,
    requires_value_evidence: bool,
) -> bool:
    """Return whether one requested item is directly present in evidence."""

    normalized_item = _normalize_for_match(item)
    if not normalized_item:
        return_value = False
        return return_value

    cjk_alnum_item = ""
    if _is_cjk_alnum_target(item):
        cjk_alnum_item = _normalize_cjk_alnum_for_match(item)

    for evidence in evidence_items:
        normalized_evidence = _normalize_for_match(evidence)
        item_matches = normalized_item in normalized_evidence
        if item_matches and _value_requirement_is_met(
            evidence,
            requires_value_evidence=requires_value_evidence,
        ):
            return True
        if cjk_alnum_item:
            cjk_alnum_evidence = _normalize_cjk_alnum_for_match(evidence)
            cjk_item_matches = cjk_alnum_item in cjk_alnum_evidence
            if cjk_item_matches and _value_requirement_is_met(
                evidence,
                requires_value_evidence=requires_value_evidence,
            ):
                return True

    return_value = False
    return return_value


def _normalize_for_match(value: str) -> str:
    """Normalize text for conservative substring coverage matching."""

    normalized = _NORMALIZE_RE.sub(" ", text_or_empty(value)).lower()
    normalized = " ".join(normalized.split())
    return normalized


def _is_cjk_alnum_target(value: str) -> bool:
    """Return whether a target mixes CJK text with ASCII letters or numbers."""

    text = text_or_empty(value)
    has_cjk = _CJK_RE.search(text) is not None
    has_ascii_alnum = _ASCII_ALNUM_RE.search(text) is not None
    return_value = has_cjk and has_ascii_alnum
    return return_value


def _normalize_cjk_alnum_for_match(value: str) -> str:
    """Normalize CJK/model-code targets across spacing and simple particles."""

    normalized = _NORMALIZE_RE.sub(" ", text_or_empty(value)).lower()
    compact = "".join(normalized.split())
    compact = _CJK_ALNUM_CONNECTIVE_RE.sub("", compact)
    return compact


def _value_requirement_is_met(
    evidence: str,
    *,
    requires_value_evidence: bool,
) -> bool:
    """Return whether one evidence row satisfies an optional value condition."""

    if not requires_value_evidence:
        return_value = True
        return return_value
    return_value = _VALUE_EVIDENCE_RE.search(evidence) is not None
    return return_value


def _evidence_quality(
    *,
    requested_items: list[str],
    covered_items: list[str],
    clean_evidence: list[str],
    worker_resolved: bool,
) -> EvidenceQuality:
    """Classify evidence quality from worker verdict and item coverage."""

    if requested_items:
        if (
            worker_resolved
            and clean_evidence
            and len(covered_items) == len(requested_items)
        ):
            return_value: EvidenceQuality = "confirmed"
            return return_value
        if covered_items:
            return_value = "partial"
            return return_value
        if clean_evidence:
            return_value = "nearby"
            return return_value
        return_value = "missing"
        return return_value

    if worker_resolved and clean_evidence:
        return_value = "confirmed"
        return return_value
    if clean_evidence:
        return_value = "nearby"
        return return_value
    return_value = "missing"
    return return_value


def _coverage_confidence(
    *,
    evidence_quality: EvidenceQuality,
    requested_items: list[str],
    covered_items: list[str],
) -> float:
    """Return a stable confidence score for the quality label."""

    if evidence_quality == "confirmed":
        return 1.0
    if evidence_quality == "partial" and requested_items:
        ratio = len(covered_items) / len(requested_items)
        confidence = round(ratio, 2)
        return confidence
    if evidence_quality == "nearby":
        return 0.25
    return 0.0


def _coverage_reason(
    *,
    evidence_quality: EvidenceQuality,
    requested_items: list[str],
    covered_items: list[str],
    missing_items: list[str],
    worker_resolved: bool,
) -> str:
    """Build a compact human-readable coverage reason."""

    if evidence_quality == "confirmed":
        if requested_items:
            reason = "Evidence covers all requested items."
            return reason
        reason = "Worker returned selected evidence and judged it resolved."
        return reason

    if evidence_quality == "partial":
        covered_text = ", ".join(covered_items)
        missing_text = ", ".join(missing_items)
        reason = (
            f"Evidence covers {covered_text} but is missing {missing_text}."
        )
        return reason

    if evidence_quality == "nearby":
        if worker_resolved:
            reason = "Worker resolved the search, but requested items were not confirmed."
            return reason
        reason = "Evidence exists but was not confirmed as resolving the task."
        return reason

    if requested_items:
        missing_text = ", ".join(missing_items)
        reason = f"No selected evidence confirmed requested items: {missing_text}."
        return reason

    reason = "No selected evidence was available."
    return reason
