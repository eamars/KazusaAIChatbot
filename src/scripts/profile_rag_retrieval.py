"""Profile conversation RAG retrieval quality for tuning fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROFILE_ROW_TEXT_LIMIT = 180
POSITIVE_CASE_KIND = "positive"
NEGATIVE_CASE_KIND = "negative"
REQUIRED_CASE_FIELDS = (
    "case_id",
    "kind",
    "platform",
    "platform_channel_id",
    "query",
    "expected_any",
    "forbidden_any",
)


def _text(value: object) -> str:
    """Return a stripped string for diagnostics."""

    if isinstance(value, str):
        return_value = value.strip()
        return return_value
    return_value = ""
    return return_value


def _identity_text(value: object) -> str:
    """Return a stable text identity for profile artifacts."""

    if value is None:
        return_value = ""
        return return_value
    return_value = str(value)
    return return_value


def _utc_now_iso() -> str:
    """Return the current UTC timestamp for profile artifacts."""

    return_value = datetime.now(timezone.utc).isoformat()
    return return_value


def _string_list(value: object, field_name: str) -> list[str]:
    """Validate and normalize a list of strings."""

    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    result: list[str] = []
    for item in value:
        item_text = _text(item)
        if item_text:
            result.append(item_text)
    return result


def _normalize_case(raw_case: object, index: int) -> dict[str, Any]:
    """Validate and normalize one profile case."""

    if not isinstance(raw_case, dict):
        raise ValueError(f"profile case {index} must be an object")

    missing_fields = [
        field_name
        for field_name in REQUIRED_CASE_FIELDS
        if field_name not in raw_case
    ]
    if missing_fields:
        raise ValueError(
            f"profile case {index} missing fields: {', '.join(missing_fields)}"
        )

    case_kind = _text(raw_case["kind"])
    if case_kind not in {POSITIVE_CASE_KIND, NEGATIVE_CASE_KIND}:
        raise ValueError(
            f"profile case {index} kind must be positive or negative"
        )

    expected_any = _string_list(raw_case["expected_any"], "expected_any")
    forbidden_any = _string_list(raw_case["forbidden_any"], "forbidden_any")
    if case_kind == POSITIVE_CASE_KIND and not expected_any:
        raise ValueError(f"profile case {index} positive case needs expected_any")
    if case_kind == NEGATIVE_CASE_KIND and not forbidden_any:
        raise ValueError(f"profile case {index} negative case needs forbidden_any")

    normalized = {
        "case_id": _text(raw_case["case_id"]),
        "kind": case_kind,
        "platform": _text(raw_case["platform"]),
        "platform_channel_id": _text(raw_case["platform_channel_id"]),
        "query": _text(raw_case["query"]),
        "expected_any": expected_any,
        "forbidden_any": forbidden_any,
    }
    for field_name in ("case_id", "platform", "platform_channel_id", "query"):
        if not normalized[field_name]:
            raise ValueError(f"profile case {index} {field_name} must not be empty")
    return normalized


def load_profile_cases(cases_path: Path) -> list[dict[str, Any]]:
    """Load and validate a profile case fixture."""

    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise ValueError("profile case fixture must contain a JSON list")

    cases = [
        _normalize_case(raw_case, index)
        for index, raw_case in enumerate(raw_cases)
    ]
    positive_count = sum(
        1
        for case in cases
        if case["kind"] == POSITIVE_CASE_KIND
    )
    negative_count = sum(
        1
        for case in cases
        if case["kind"] == NEGATIVE_CASE_KIND
    )
    if positive_count == 0:
        raise ValueError("profile case fixture needs at least one positive case")
    if negative_count == 0:
        raise ValueError("profile case fixture needs at least one negative case")
    return cases


def _row_body_text(row: Mapping[str, Any]) -> str:
    """Extract comparable row text from supported conversation row shapes."""

    for field_name in ("body_text", "content", "summary", "text"):
        value = _text(row.get(field_name))
        if value:
            return value
    return_value = ""
    return return_value


def _matching_terms(text_blocks: Sequence[str], terms: Sequence[str]) -> list[str]:
    """Return terms found in any result row text."""

    folded_blocks = [text_block.casefold() for text_block in text_blocks]
    matched_terms: list[str] = []
    for term in terms:
        folded_term = term.casefold()
        if any(folded_term in text_block for text_block in folded_blocks):
            matched_terms.append(term)
    return matched_terms


def project_profile_row(
    *,
    rank: int,
    row: Mapping[str, Any],
    text_limit: int = PROFILE_ROW_TEXT_LIMIT,
) -> dict[str, Any]:
    """Project one retrieved row into a compact profile artifact."""

    body_text = _row_body_text(row)
    truncated_text = body_text[:text_limit]
    projected = {
        "rank": rank,
        "score": row.get("score", 0.0),
        "body_text": truncated_text,
        "body_text_truncated": len(body_text) > text_limit,
        "display_name": _text(row.get("display_name")),
        "timestamp": _text(row.get("timestamp")),
        "platform": _text(row.get("platform")),
        "platform_channel_id": _text(row.get("platform_channel_id")),
        "platform_message_id": _text(row.get("platform_message_id")),
        "conversation_row_id": _identity_text(row.get("_id")),
    }
    return projected


def evaluate_profile_case(
    case: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate retrieved rows against one false-positive/negative case."""

    row_texts = [_row_body_text(row) for row in rows]
    scores = [
        score
        for row in rows
        if isinstance(score := row.get("score"), int | float)
    ]
    max_score = max(scores) if scores else 0.0
    matched_expected_terms = _matching_terms(
        row_texts,
        list(case.get("expected_any", [])),
    )
    matched_forbidden_terms = _matching_terms(
        row_texts,
        list(case.get("forbidden_any", [])),
    )
    is_positive = case.get("kind") == POSITIVE_CASE_KIND
    false_negative = is_positive and not matched_expected_terms
    false_positive = bool(matched_forbidden_terms)
    resolved = is_positive and bool(matched_expected_terms) and not false_positive
    projected_rows = [
        project_profile_row(rank=index + 1, row=row)
        for index, row in enumerate(rows)
    ]
    result = {
        "case_id": case.get("case_id", ""),
        "kind": case.get("kind", ""),
        "resolved": resolved,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "matched_expected_terms": matched_expected_terms,
        "matched_forbidden_terms": matched_forbidden_terms,
        "max_score": max_score,
        "result_count": len(rows),
        "rows": projected_rows,
    }
    return result


async def _search_case(case: Mapping[str, Any], top_k: int) -> list[dict[str, Any]]:
    """Run one live conversation vector search for profiling."""

    from kazusa_ai_chatbot.db.conversation import (  # noqa: PLC0415
        search_conversation_history,
    )

    results = await search_conversation_history(
        query=str(case["query"]),
        platform=str(case["platform"]),
        platform_channel_id=str(case["platform_channel_id"]),
        limit=top_k,
        method="vector",
    )
    rows: list[dict[str, Any]] = []
    for rank, (score, row) in enumerate(results, start=1):
        row_copy = dict(row)
        row_copy["score"] = score
        row_copy["rank"] = rank
        rows.append(row_copy)
    return rows


async def run_profile(
    *,
    phase_label: str,
    cases_path: Path,
    top_ks: Sequence[int],
) -> dict[str, Any]:
    """Run live retrieval profile cases for each requested top-k."""

    cases = load_profile_cases(cases_path)
    profile_results: list[dict[str, Any]] = []
    for top_k in top_ks:
        for case in cases:
            rows = await _search_case(case, top_k)
            evaluation = evaluate_profile_case(case, rows)
            evaluation["top_k"] = top_k
            profile_results.append(evaluation)

    result = {
        "generated_at": _utc_now_iso(),
        "phase_label": phase_label,
        "cases_path": str(cases_path),
        "top_k_values": list(top_ks),
        "top_ks": list(top_ks),
        "results": profile_results,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Profile conversation RAG retrieval tuning cases.",
    )
    parser.add_argument("--phase-label", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _write_output(output_path: Path, result: dict[str, Any]) -> None:
    """Write the profile result artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    """Run profiling and close DB handles."""

    from kazusa_ai_chatbot.db import close_db  # noqa: PLC0415

    try:
        result = await run_profile(
            phase_label=str(args.phase_label),
            cases_path=args.cases,
            top_ks=list(args.top_k),
        )
    finally:
        await close_db()
    return result


def main() -> None:
    """CLI entry point."""

    from scripts._db_export import (  # noqa: PLC0415
        configure_logging,
        configure_stdout,
        load_project_env,
    )

    parser = _parser()
    args = parser.parse_args()
    configure_stdout()
    configure_logging(bool(args.verbose))
    load_project_env()

    result = asyncio.run(_main_async(args))
    _write_output(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
