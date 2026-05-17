"""Compare Nomic embedding prefix modes on exported conversation rows."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso

from scripts.profile_rag_retrieval import (
    load_profile_cases,
    project_profile_row,
)

PREFIX_MODE_NO_PREFIX = "no_prefix"
PREFIX_MODE_TRANSFORMERS = "transformers_manual_prefix"
PREFIX_MODE_SENTENCE_TRANSFORMERS = "sentence_transformers_prompt_equivalent"

_QUERY_PREFIX = "search_query: "
_DOCUMENT_PREFIX = "search_document: "
_EMBEDDING_BATCH_SIZE = 10


def sentence_transformers_prompt_config() -> dict[str, str]:
    """Return the checked prompt mapping for Nomic v2 MoE."""

    mapping = {
        "query": _QUERY_PREFIX,
        "passage": _DOCUMENT_PREFIX,
    }
    return mapping


def prefix_mode_definitions() -> list[dict[str, str]]:
    """Return the researched prefix comparison modes."""

    prompt_config = sentence_transformers_prompt_config()
    modes = [
        {
            "name": PREFIX_MODE_NO_PREFIX,
            "query_prefix": "",
            "document_prefix": "",
        },
        {
            "name": PREFIX_MODE_TRANSFORMERS,
            "query_prefix": _QUERY_PREFIX,
            "document_prefix": _DOCUMENT_PREFIX,
        },
        {
            "name": PREFIX_MODE_SENTENCE_TRANSFORMERS,
            "query_prefix": prompt_config["query"],
            "document_prefix": prompt_config["passage"],
        },
    ]
    return modes


def _prefix_for_mode(mode_name: str, role: str) -> str:
    """Return the effective prefix for one mode and semantic role."""

    field_name = f"{role}_prefix"
    for mode in prefix_mode_definitions():
        if mode["name"] == mode_name:
            prefix = mode[field_name]
            return prefix
    raise ValueError(f"unknown prefix mode: {mode_name}")


def _apply_prefix(prefix: str, text: str) -> str:
    """Apply a comparison prefix once for profile-only embedding input."""

    if not prefix:
        return text

    prefixed_text = f"{prefix}{text}"
    return prefixed_text


def effective_query_text(mode_name: str, text: str) -> str:
    """Return the query text sent to the embedding endpoint for one mode."""

    prefix = _prefix_for_mode(mode_name, "query")
    effective_text = _apply_prefix(prefix, text)
    return effective_text


def effective_document_text(mode_name: str, text: str) -> str:
    """Return the document text sent to the embedding endpoint for one mode."""

    prefix = _prefix_for_mode(mode_name, "document")
    effective_text = _apply_prefix(prefix, text)
    return effective_text


def _row_text(row: Mapping[str, Any]) -> str:
    """Build source text from an exported conversation row."""

    parts: list[str] = []
    body_text = str(row.get("body_text", "")).strip()
    if body_text:
        parts.append(body_text)
    attachments = row.get("attachments")
    if isinstance(attachments, list):
        for attachment in attachments:
            if not isinstance(attachment, Mapping):
                continue
            description = str(attachment.get("description", "")).strip()
            if description:
                parts.append(description)
    source_text = "\n".join(parts)
    return source_text


def _matching_rank(
    rows: Sequence[Mapping[str, Any]],
    terms: Sequence[str],
) -> int | None:
    """Return the first rank containing any requested term."""

    for index, row in enumerate(rows, start=1):
        text = _row_text(row).casefold()
        for term in terms:
            if term.casefold() in text:
                return_value = index
                return return_value
    return_value = None
    return return_value


def evaluate_prefix_mode_case(
    case: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    top_ks: Sequence[int],
) -> dict[str, Any]:
    """Evaluate ranked rows for hit and false-positive ranks."""

    expected_rank = _matching_rank(rows, list(case.get("expected_any", [])))
    forbidden_rank = _matching_rank(rows, list(case.get("forbidden_any", [])))
    hit_at = {
        str(top_k): expected_rank is not None and expected_rank <= top_k
        for top_k in top_ks
    }
    false_positive_at = {
        str(top_k): forbidden_rank is not None and forbidden_rank <= top_k
        for top_k in top_ks
    }
    result = {
        "case_id": case.get("case_id", ""),
        "kind": case.get("kind", ""),
        "first_expected_hit_rank": expected_rank,
        "first_forbidden_hit_rank": forbidden_rank,
        "hit_at": hit_at,
        "false_positive_at": false_positive_at,
        "rows": [
            project_profile_row(rank=index + 1, row=row)
            for index, row in enumerate(rows)
        ],
    }
    return result


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity for two embedding vectors."""

    vector_products = zip(left, right, strict=True)
    numerator = sum(
        left_value * right_value
        for left_value, right_value in vector_products
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return_value = 0.0
        return return_value
    return_value = numerator / (left_norm * right_norm)
    return return_value


async def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed exact effective texts without adapter-side prefixing."""

    from kazusa_ai_chatbot.config import (  # noqa: PLC0415
        EMBEDDING_API_KEY,
        EMBEDDING_BASE_URL,
        EMBEDDING_MODEL,
    )

    client = AsyncOpenAI(base_url=EMBEDDING_BASE_URL, api_key=EMBEDDING_API_KEY)
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        chunk = texts[start:start + _EMBEDDING_BATCH_SIZE]
        response = await client.embeddings.create(input=chunk, model=EMBEDDING_MODEL)
        sorted_data = sorted(response.data, key=lambda item: item.index)
        embeddings.extend(item.embedding for item in sorted_data)
    return embeddings


def _load_exported_messages(input_path: Path) -> list[dict[str, Any]]:
    """Load exported conversation rows from the supported artifact shapes."""

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        return_value = [
            dict(row) for row in payload["messages"] if isinstance(row, dict)
        ]
        return return_value
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return_value = [
            dict(row) for row in payload["records"] if isinstance(row, dict)
        ]
        return return_value
    if isinstance(payload, list):
        return_value = [
            dict(row) for row in payload if isinstance(row, dict)
        ]
        return return_value
    raise ValueError("input artifact must contain messages or records")


def _rank_rows(
    *,
    query_embedding: Sequence[float],
    document_embeddings: Sequence[Sequence[float]],
    rows: Sequence[Mapping[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Rank rows by cosine similarity."""

    scored_rows: list[dict[str, Any]] = []
    for row, document_embedding in zip(rows, document_embeddings, strict=True):
        scored_row = dict(row)
        scored_row["score"] = _cosine_similarity(query_embedding, document_embedding)
        scored_rows.append(scored_row)
    scored_rows.sort(key=lambda row: row["score"], reverse=True)
    ranked_rows = scored_rows[:top_k]
    return ranked_rows


async def run_prefix_mode_profile(
    *,
    phase_label: str,
    cases_path: Path,
    input_path: Path,
    top_ks: Sequence[int] = (5, 10, 20),
) -> dict[str, Any]:
    """Run the prefix-mode comparison against exported conversation rows."""

    cases = load_profile_cases(cases_path)
    rows = [
        row
        for row in _load_exported_messages(input_path)
        if _row_text(row)
    ]
    max_top_k = max(top_ks)
    mode_results: list[dict[str, Any]] = []
    for mode in prefix_mode_definitions():
        mode_name = mode["name"]
        document_texts = [
            effective_document_text(mode_name, _row_text(row))
            for row in rows
        ]
        document_embeddings = await _embed_texts(document_texts)
        case_results: list[dict[str, Any]] = []
        for case in cases:
            query_text = effective_query_text(mode_name, str(case["query"]))
            query_embeddings = await _embed_texts([query_text])
            query_embedding = query_embeddings[0]
            ranked_rows = _rank_rows(
                query_embedding=query_embedding,
                document_embeddings=document_embeddings,
                rows=rows,
                top_k=max_top_k,
            )
            metrics = evaluate_prefix_mode_case(
                case,
                ranked_rows,
                top_ks=top_ks,
            )
            case_results.append(metrics)
        mode_result = {
            **mode,
            "cases": case_results,
        }
        mode_results.append(mode_result)

    result = {
        "generated_at": storage_utc_now_iso(),
        "phase_label": phase_label,
        "cases_path": str(cases_path),
        "input_path": str(input_path),
        "top_ks": list(top_ks),
        "modes": mode_results,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Compare Nomic embedding prefix modes on exported rows.",
    )
    parser.add_argument("--phase-label", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write_output(output_path: Path, result: Mapping[str, Any]) -> None:
    """Write the profile artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    """Run the profile from the command line."""

    from scripts._db_export import configure_stdout, load_project_env  # noqa: PLC0415

    parser = _parser()
    args = parser.parse_args()
    configure_stdout()
    load_project_env()
    result = asyncio.run(
        run_prefix_mode_profile(
            phase_label=str(args.phase_label),
            cases_path=args.cases,
            input_path=args.input,
        )
    )
    _write_output(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
