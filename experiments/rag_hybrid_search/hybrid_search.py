"""Reusable hybrid conversation-retrieval experiment code.

The experiment evaluates vector search, keyword search, and the same production
hybrid conversation-retrieval entrypoint against historical cases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from kazusa_ai_chatbot.config import (
    RAG_CONVERSATION_EVIDENCE_TEXT_LIMIT,
    RAG_HYBRID_NEIGHBOR_MESSAGE_LIMIT,
    RAG_HYBRID_NEIGHBOR_SEED_LIMIT,
    RAG_HYBRID_NEIGHBOR_WINDOW_MINUTES,
    RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR,
    RAG_SEARCH_SELECTED_LIMIT,
)
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db.conversation import search_conversation_history
from kazusa_ai_chatbot.rag import conversation_search_agent
from kazusa_ai_chatbot.rag.hybrid_retrieval import (
    HybridCandidate,
    candidate_prompt_text,
    hybrid_row_identity,
    merge_hybrid_candidates,
    select_neighbor_seed_candidates,
)

POSITIVE_CASE_KIND = "positive"
NEGATIVE_CASE_KIND = "negative"
DEFAULT_TEXT_LIMIT = 220
DEFAULT_COMPARE_TEXT_LIMIT = RAG_CONVERSATION_EVIDENCE_TEXT_LIMIT
DEFAULT_RESULT_LIMIT = RAG_SEARCH_SELECTED_LIMIT
DEFAULT_NEIGHBOR_SEED_LIMIT = RAG_HYBRID_NEIGHBOR_SEED_LIMIT
DEFAULT_NEIGHBOR_MESSAGE_LIMIT = RAG_HYBRID_NEIGHBOR_MESSAGE_LIMIT
DEFAULT_NEIGHBOR_WINDOW_MINUTES = RAG_HYBRID_NEIGHBOR_WINDOW_MINUTES
DEFAULT_SEMANTIC_ONLY_FLOOR = RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR


@dataclass(frozen=True)
class HybridSearchConfig:
    """Runtime controls for one experiment run.

    Args:
        semantic_top_k: Number of vector candidates to retrieve.
        keyword_top_k: Number of candidates per keyword to retrieve.
        selected_limit: Number of rows exposed to evaluation and artifacts.
        neighbor_seed_limit: Number of seed rows expanded into local context.
        neighbor_message_limit: Rows to keep on each side of a seed row.
        neighbor_window_minutes: Time window used to fetch seed neighbors.
        semantic_only_floor: Minimum vector score for semantic-only hybrid rows
            when no keyword anchor returns any result.
        text_limit: Maximum text per row in JSON/Markdown artifacts.
    """

    semantic_top_k: int = DEFAULT_RESULT_LIMIT
    keyword_top_k: int = DEFAULT_RESULT_LIMIT
    selected_limit: int = DEFAULT_RESULT_LIMIT
    neighbor_seed_limit: int = DEFAULT_NEIGHBOR_SEED_LIMIT
    neighbor_message_limit: int = DEFAULT_NEIGHBOR_MESSAGE_LIMIT
    neighbor_window_minutes: int = DEFAULT_NEIGHBOR_WINDOW_MINUTES
    semantic_only_floor: float = DEFAULT_SEMANTIC_ONLY_FLOOR
    text_limit: int = DEFAULT_TEXT_LIMIT


@dataclass(frozen=True)
class SearchCase:
    """One retrieval-quality case loaded from JSON."""

    case_id: str
    kind: str
    platform: str
    platform_channel_id: str
    query: str
    keywords: tuple[str, ...]
    expected_any: tuple[str, ...]
    expected_message_ids: tuple[str, ...]
    forbidden_any: tuple[str, ...]
    from_timestamp: str | None = None
    to_timestamp: str | None = None


def _text(value: object) -> str:
    """Return stripped text for external JSON and database fields."""

    if isinstance(value, str):
        return_value = value.strip()
        return return_value
    return_value = ""
    return return_value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    """Validate a JSON string-list field.

    Args:
        value: Raw JSON value.
        field_name: Field name for diagnostics.

    Returns:
        Non-empty string items as a tuple.

    Raises:
        ValueError: If the value is not a list.
    """

    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")

    items = tuple(
        item_text
        for item in value
        if (item_text := _text(item))
    )
    return items


def _normalize_case(raw_case: object, index: int) -> SearchCase:
    """Validate and normalize one experiment case."""

    if not isinstance(raw_case, dict):
        raise ValueError(f"case {index} must be an object")

    required_fields = (
        "case_id",
        "kind",
        "platform",
        "platform_channel_id",
        "query",
        "keywords",
        "expected_any",
        "expected_message_ids",
        "forbidden_any",
    )
    missing_fields = [
        field_name
        for field_name in required_fields
        if field_name not in raw_case
    ]
    if missing_fields:
        raise ValueError(f"case {index} missing fields: {', '.join(missing_fields)}")

    case_id = _text(raw_case["case_id"])
    kind = _text(raw_case["kind"])
    platform = _text(raw_case["platform"])
    channel_id = _text(raw_case["platform_channel_id"])
    query = _text(raw_case["query"])
    if kind not in {POSITIVE_CASE_KIND, NEGATIVE_CASE_KIND}:
        raise ValueError(f"case {index} kind must be positive or negative")
    for field_name, value in (
        ("case_id", case_id),
        ("platform", platform),
        ("platform_channel_id", channel_id),
        ("query", query),
    ):
        if not value:
            raise ValueError(f"case {index} {field_name} must not be empty")

    case = SearchCase(
        case_id=case_id,
        kind=kind,
        platform=platform,
        platform_channel_id=channel_id,
        query=query,
        keywords=_string_tuple(raw_case["keywords"], "keywords"),
        expected_any=_string_tuple(raw_case["expected_any"], "expected_any"),
        expected_message_ids=_string_tuple(
            raw_case["expected_message_ids"],
            "expected_message_ids",
        ),
        forbidden_any=_string_tuple(raw_case["forbidden_any"], "forbidden_any"),
        from_timestamp=_text(raw_case.get("from_timestamp")) or None,
        to_timestamp=_text(raw_case.get("to_timestamp")) or None,
    )
    if case.kind == POSITIVE_CASE_KIND and not case.expected_any:
        raise ValueError(f"case {index} positive case needs expected_any")
    if case.kind == NEGATIVE_CASE_KIND and not case.forbidden_any:
        raise ValueError(f"case {index} negative case needs forbidden_any")
    return case


def load_cases(cases_path: Path) -> list[SearchCase]:
    """Load and validate experiment cases from a JSON fixture."""

    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise ValueError("case fixture must contain a JSON list")

    cases = [
        _normalize_case(raw_case, index)
        for index, raw_case in enumerate(raw_cases)
    ]
    positive_count = sum(1 for case in cases if case.kind == POSITIVE_CASE_KIND)
    negative_count = sum(1 for case in cases if case.kind == NEGATIVE_CASE_KIND)
    if positive_count == 0:
        raise ValueError("case fixture needs at least one positive case")
    if negative_count == 0:
        raise ValueError("case fixture needs at least one negative case")
    return cases


def row_identity(row: Mapping[str, Any]) -> str:
    """Return the strongest stable identity for a conversation row."""

    return_value = hybrid_row_identity(row, source="conversation")
    return return_value


def row_text(row: Mapping[str, Any]) -> str:
    """Extract comparable text from one conversation row."""

    return_value = candidate_prompt_text(
        row,
        source="conversation",
        text_limit=DEFAULT_COMPARE_TEXT_LIMIT,
    )
    return return_value


def build_hybrid_seed_rows(
    semantic_rows: Sequence[Mapping[str, Any]],
    keyword_rows: Sequence[Mapping[str, Any]],
    *,
    semantic_only_floor: float,
) -> list[dict[str, Any]]:
    """Merge semantic and keyword candidates into neighbor-expansion seeds.

    Keyword anchors act as the precision guard. If no keyword rows are found,
    semantic-only rows must clear a score floor before they can seed hybrid
    evidence. This keeps weak semantic drift from turning absent-topic cases
    into false positives.
    """

    selected_limit = len(semantic_rows) + len(keyword_rows)
    candidates = merge_hybrid_candidates(
        list(semantic_rows),
        list(keyword_rows),
        semantic_only_floor=semantic_only_floor,
        selected_limit=max(1, selected_limit),
        source="conversation",
    )
    rows = _rows_from_candidates(candidates)
    return rows


def build_neighbor_seed_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    keyword_rows_present: bool,
    semantic_only_floor: float,
) -> list[dict[str, Any]]:
    """Select direct evidence rows that may expand into local context."""

    candidates = _candidates_from_rows(candidate_rows)
    seeds = select_neighbor_seed_candidates(
        candidates,
        keyword_rows_present=keyword_rows_present,
        semantic_only_floor=semantic_only_floor,
        seed_limit=len(candidates) or 1,
    )
    rows = _rows_from_candidates(seeds)
    return rows


def merge_candidates(
    existing_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge candidate rows by message identity while preserving provenance.

    Args:
        existing_rows: Rows already accumulated from previous methods.
        new_rows: Rows produced by one additional retrieval method.

    Returns:
        De-duplicated rows sorted by hybrid retrieval priority.
    """

    selected_limit = len(existing_rows) + len(new_rows)
    candidates = merge_hybrid_candidates(
        list(existing_rows),
        list(new_rows),
        semantic_only_floor=0.0,
        selected_limit=max(1, selected_limit),
        source="conversation",
    )
    rows = _rows_from_candidates(candidates)
    return rows


def _candidates_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[HybridCandidate]:
    """Convert projected experiment rows back into production candidates."""

    candidates: list[HybridCandidate] = []
    for rank, row in enumerate(rows, start=1):
        row_dict = dict(row)
        methods_value = row.get("methods")
        if isinstance(methods_value, list):
            methods = tuple(_text(method) for method in methods_value)
        else:
            methods = ()
        matched_anchors_value = row.get("matched_anchors")
        if isinstance(matched_anchors_value, list):
            matched_anchors = tuple(
                _text(anchor) for anchor in matched_anchors_value
            )
        else:
            matched_anchors = ()
        score = _numeric_row_value(row.get("score"))
        best_rank = _int_row_value(row.get("best_rank"), rank)
        candidate = HybridCandidate(
            row=row_dict,
            identity=row_identity(row_dict),
            source="conversation",
            methods=methods,
            matched_anchors=matched_anchors,
            score=score,
            best_rank=best_rank,
        )
        candidates.append(candidate)
    return candidates


def _rows_from_candidates(candidates: Sequence[HybridCandidate]) -> list[dict[str, Any]]:
    """Project production hybrid candidates into experiment row dictionaries."""

    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        row = dict(candidate.row)
        row["methods"] = list(candidate.methods)
        row["matched_anchors"] = list(candidate.matched_anchors)
        row["score"] = candidate.score
        row["best_rank"] = candidate.best_rank
        row["hybrid_rank"] = rank
        rows.append(row)
    return rows


def _numeric_row_value(value: object) -> float:
    """Return a numeric row value, or zero when the field is absent."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return_value = float(value)
        return return_value
    return_value = 0.0
    return return_value


def _int_row_value(value: object, default: int) -> int:
    """Return an integer row value, or a supplied default."""

    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


async def _semantic_rows(case: SearchCase, limit: int) -> list[dict[str, Any]]:
    """Retrieve vector candidates for one case."""

    results = await search_conversation_history(
        query=case.query,
        platform=case.platform,
        platform_channel_id=case.platform_channel_id,
        limit=limit,
        method="vector",
        from_timestamp=case.from_timestamp,
        to_timestamp=case.to_timestamp,
    )
    rows: list[dict[str, Any]] = []
    for rank, (score, row) in enumerate(results, start=1):
        row_copy = dict(row)
        row_copy["score"] = score
        row_copy["rank"] = rank
        row_copy["methods"] = ["semantic"]
        rows.append(row_copy)
    return rows


async def _keyword_rows(case: SearchCase, limit: int) -> list[dict[str, Any]]:
    """Retrieve keyword candidates for all literal anchors in one case."""

    rows: list[dict[str, Any]] = []
    for keyword in case.keywords:
        results = await search_conversation_history(
            query=keyword,
            platform=case.platform,
            platform_channel_id=case.platform_channel_id,
            limit=limit,
            method="keyword",
            from_timestamp=case.from_timestamp,
            to_timestamp=case.to_timestamp,
        )
        for rank, (_, row) in enumerate(results, start=1):
            row_copy = dict(row)
            row_copy["rank"] = rank
            row_copy["methods"] = [f"keyword:{keyword}"]
            rows.append(row_copy)
    return rows


async def search_case_methods(
    case: SearchCase,
    config: HybridSearchConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Run semantic, keyword, and hybrid retrieval for one case."""

    semantic_rows = await _semantic_rows(case, config.semantic_top_k)
    keyword_rows = await _keyword_rows(case, config.keyword_top_k)
    hybrid_rows = await _production_hybrid_rows(case, config)
    methods = {
        "semantic": semantic_rows[:config.selected_limit],
        "keyword": merge_candidates([], keyword_rows)[:config.selected_limit],
        "hybrid": hybrid_rows[:config.selected_limit],
    }
    return methods


async def _production_hybrid_rows(
    case: SearchCase,
    config: HybridSearchConfig,
) -> list[dict[str, Any]]:
    """Run the same hybrid retrieval entrypoint used by production RAG."""

    args: dict[str, Any] = {
        "search_query": case.query,
        "literal_anchors": list(case.keywords),
        "top_k": config.semantic_top_k,
        "platform": case.platform,
        "platform_channel_id": case.platform_channel_id,
    }
    if case.from_timestamp:
        args["from_timestamp"] = case.from_timestamp
    if case.to_timestamp:
        args["to_timestamp"] = case.to_timestamp

    rows = await conversation_search_agent.run_hybrid_conversation_search(
        args,
        semantic_only_floor=config.semantic_only_floor,
        selected_limit=config.selected_limit,
        neighbor_seed_limit=config.neighbor_seed_limit,
        neighbor_message_limit=config.neighbor_message_limit,
        neighbor_window_minutes=config.neighbor_window_minutes,
    )
    return rows


def _matching_terms(text_blocks: Sequence[str], terms: Sequence[str]) -> list[str]:
    """Return requested terms present in the result text."""

    folded_blocks = [text_block.casefold() for text_block in text_blocks]
    matched_terms: list[str] = []
    for term in terms:
        folded_term = term.casefold()
        if any(folded_term in text_block for text_block in folded_blocks):
            matched_terms.append(term)
    return matched_terms


def _first_expected_message_rank(
    rows: Sequence[Mapping[str, Any]],
    expected_message_ids: Sequence[str],
) -> int | None:
    """Return the first rank containing an expected platform message id."""

    if not expected_message_ids:
        return_value: int | None = None
        return return_value

    expected_ids = set(expected_message_ids)
    for rank, row in enumerate(rows, start=1):
        platform_message_id = _text(row.get("platform_message_id"))
        if platform_message_id in expected_ids:
            return rank
    return_value = None
    return return_value


def project_row(
    row: Mapping[str, Any],
    *,
    rank: int,
    text_limit: int,
) -> dict[str, Any]:
    """Project one row into a compact experiment artifact."""

    body_text = row_text(row)
    projected = {
        "rank": rank,
        "score": row.get("score", 0.0),
        "best_rank": row.get("best_rank", row.get("rank", rank)),
        "methods": row.get("methods", []),
        "timestamp": _text(row.get("timestamp")),
        "display_name": _text(row.get("display_name")),
        "platform_message_id": _text(row.get("platform_message_id")),
        "conversation_row_id": _text(row.get("conversation_row_id")),
        "body_text": body_text[:text_limit],
        "body_text_truncated": len(body_text) > text_limit,
    }
    return projected


def evaluate_case_method(
    case: SearchCase,
    method_name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    text_limit: int,
) -> dict[str, Any]:
    """Evaluate one method's rows against one case."""

    row_texts = [row_text(row) for row in rows]
    matched_expected_terms = _matching_terms(row_texts, case.expected_any)
    matched_forbidden_terms = _matching_terms(row_texts, case.forbidden_any)
    expected_message_rank = _first_expected_message_rank(
        rows,
        case.expected_message_ids,
    )
    is_positive = case.kind == POSITIVE_CASE_KIND
    false_negative = is_positive and (
        not matched_expected_terms
        or (
            bool(case.expected_message_ids)
            and expected_message_rank is None
        )
    )
    if is_positive:
        false_positive = bool(matched_forbidden_terms)
        resolved = not false_negative and not false_positive
    else:
        false_positive = bool(rows)
        resolved = not false_positive

    projected_rows = [
        project_row(row, rank=rank, text_limit=text_limit)
        for rank, row in enumerate(rows, start=1)
    ]
    evaluation = {
        "case_id": case.case_id,
        "kind": case.kind,
        "method": method_name,
        "resolved": resolved,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "matched_expected_terms": matched_expected_terms,
        "matched_forbidden_terms": matched_forbidden_terms,
        "expected_message_rank": expected_message_rank,
        "result_count": len(rows),
        "rows": projected_rows,
    }
    return evaluation


def summarize_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build method-level aggregate metrics for experiment results."""

    method_names = sorted({
        str(result["method"])
        for result in results
    })
    summary: dict[str, Any] = {}
    for method_name in method_names:
        method_results = [
            result
            for result in results
            if result["method"] == method_name
        ]
        positive_results = [
            result
            for result in method_results
            if result["kind"] == POSITIVE_CASE_KIND
        ]
        negative_results = [
            result
            for result in method_results
            if result["kind"] == NEGATIVE_CASE_KIND
        ]
        expected_ranks = [
            rank
            for result in positive_results
            if isinstance(rank := result.get("expected_message_rank"), int)
        ]
        resolved_count = sum(1 for result in method_results if result["resolved"])
        false_positive_count = sum(
            1
            for result in method_results
            if result["false_positive"]
        )
        false_negative_count = sum(
            1
            for result in method_results
            if result["false_negative"]
        )
        average_expected_rank = (
            sum(expected_ranks) / len(expected_ranks)
            if expected_ranks
            else None
        )
        summary[method_name] = {
            "case_count": len(method_results),
            "positive_case_count": len(positive_results),
            "negative_case_count": len(negative_results),
            "resolved_count": resolved_count,
            "false_positive_count": false_positive_count,
            "false_negative_count": false_negative_count,
            "expected_message_hit_count": len(expected_ranks),
            "average_expected_message_rank": average_expected_rank,
        }
    return summary


async def run_experiment(
    *,
    cases_path: Path,
    config: HybridSearchConfig,
) -> dict[str, Any]:
    """Run the hybrid retrieval experiment against real conversation data."""

    cases = load_cases(cases_path)
    results: list[dict[str, Any]] = []
    for case in cases:
        method_rows = await search_case_methods(case, config)
        for method_name, rows in method_rows.items():
            evaluation = evaluate_case_method(
                case,
                method_name,
                rows,
                text_limit=config.text_limit,
            )
            results.append(evaluation)

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": generated_at,
        "cases_path": str(cases_path),
        "config": {
            "semantic_top_k": config.semantic_top_k,
            "keyword_top_k": config.keyword_top_k,
            "selected_limit": config.selected_limit,
            "neighbor_seed_limit": config.neighbor_seed_limit,
            "neighbor_message_limit": config.neighbor_message_limit,
            "neighbor_window_minutes": config.neighbor_window_minutes,
            "semantic_only_floor": config.semantic_only_floor,
            "text_limit": config.text_limit,
        },
        "summary": summarize_results(results),
        "results": results,
    }
    return payload


def render_markdown_report(result: Mapping[str, Any]) -> str:
    """Render a compact Markdown report from an experiment payload."""

    lines = [
        "# RAG Hybrid Search Experiment",
        "",
        f"- Generated at: `{result['generated_at']}`",
        f"- Cases: `{result['cases_path']}`",
        "",
        "## Summary",
        "",
        "| Method | Resolved | False positives | False negatives | Expected hit count | Avg expected rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    summary = result["summary"]
    if not isinstance(summary, dict):
        raise ValueError("result summary must be a dict")

    for method_name in sorted(summary):
        method_summary = summary[method_name]
        average_rank = method_summary["average_expected_message_rank"]
        average_text = f"{average_rank:.2f}" if isinstance(average_rank, float) else ""
        line = (
            f"| {method_name} "
            f"| {method_summary['resolved_count']}/{method_summary['case_count']} "
            f"| {method_summary['false_positive_count']} "
            f"| {method_summary['false_negative_count']} "
            f"| {method_summary['expected_message_hit_count']} "
            f"| {average_text} |"
        )
        lines.append(line)

    lines.extend(["", "## Case Results", ""])
    results = result["results"]
    if not isinstance(results, list):
        raise ValueError("result results must be a list")
    for item in results:
        if not isinstance(item, dict):
            continue
        rank = item.get("expected_message_rank")
        rank_text = str(rank) if isinstance(rank, int) else "miss"
        line = (
            f"- `{item['case_id']}` / `{item['method']}`: "
            f"resolved={item['resolved']}, "
            f"expected_rank={rank_text}, "
            f"false_positive={item['false_positive']}, "
            f"false_negative={item['false_negative']}"
        )
        lines.append(line)

    report = "\n".join(lines) + "\n"
    return report


async def close_experiment_db() -> None:
    """Close database handles opened by experiment retrieval helpers."""

    await close_db()
