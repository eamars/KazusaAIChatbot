"""Command-line runner for the hybrid RAG retrieval experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from experiments.rag_hybrid_search.hybrid_search import (
    HybridSearchConfig,
    close_experiment_db,
    render_markdown_report,
    run_experiment,
)
from kazusa_ai_chatbot.config import (
    RAG_CONVERSATION_EVIDENCE_TEXT_LIMIT,
    RAG_HYBRID_NEIGHBOR_MESSAGE_LIMIT,
    RAG_HYBRID_NEIGHBOR_SEED_LIMIT,
    RAG_HYBRID_NEIGHBOR_WINDOW_MINUTES,
    RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR,
    RAG_SEARCH_DEFAULT_TOP_K,
    RAG_SEARCH_SELECTED_LIMIT,
)


def _parser() -> argparse.ArgumentParser:
    """Build the experiment command-line parser."""

    parser = argparse.ArgumentParser(
        description="Evaluate semantic, keyword, and hybrid RAG retrieval.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("experiments/rag_hybrid_search/cases.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_artifacts/rag_hybrid_search_experiment.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("test_artifacts/rag_hybrid_search_experiment.md"),
    )
    parser.add_argument("--semantic-top-k", type=int, default=RAG_SEARCH_DEFAULT_TOP_K)
    parser.add_argument("--keyword-top-k", type=int, default=RAG_SEARCH_DEFAULT_TOP_K)
    parser.add_argument("--selected-limit", type=int, default=RAG_SEARCH_SELECTED_LIMIT)
    parser.add_argument(
        "--neighbor-seed-limit",
        type=int,
        default=RAG_HYBRID_NEIGHBOR_SEED_LIMIT,
    )
    parser.add_argument(
        "--neighbor-message-limit",
        type=int,
        default=RAG_HYBRID_NEIGHBOR_MESSAGE_LIMIT,
    )
    parser.add_argument(
        "--neighbor-window-minutes",
        type=int,
        default=RAG_HYBRID_NEIGHBOR_WINDOW_MINUTES,
    )
    parser.add_argument(
        "--semantic-only-floor",
        type=float,
        default=RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR,
    )
    parser.add_argument(
        "--text-limit",
        type=int,
        default=RAG_CONVERSATION_EVIDENCE_TEXT_LIMIT,
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> HybridSearchConfig:
    """Build a typed experiment config from parsed CLI arguments."""

    config = HybridSearchConfig(
        semantic_top_k=args.semantic_top_k,
        keyword_top_k=args.keyword_top_k,
        selected_limit=args.selected_limit,
        neighbor_seed_limit=args.neighbor_seed_limit,
        neighbor_message_limit=args.neighbor_message_limit,
        neighbor_window_minutes=args.neighbor_window_minutes,
        semantic_only_floor=args.semantic_only_floor,
        text_limit=args.text_limit,
    )
    return config


def _write_json(output_path: Path, payload: dict[str, object]) -> None:
    """Write the machine-readable experiment artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _write_report(report_path: Path, payload: dict[str, object]) -> None:
    """Write the human-readable experiment report."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = render_markdown_report(payload)
    report_path.write_text(report, encoding="utf-8")


async def _main_async(args: argparse.Namespace) -> dict[str, object]:
    """Run the experiment and close database resources."""

    config = _config_from_args(args)
    try:
        payload = await run_experiment(cases_path=args.cases, config=config)
    finally:
        await close_experiment_db()
    return payload


def main() -> None:
    """Run the experiment from the command line."""

    parser = _parser()
    args = parser.parse_args()
    payload = asyncio.run(_main_async(args))
    _write_json(args.output, payload)
    _write_report(args.report, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
