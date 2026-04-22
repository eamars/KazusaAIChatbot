"""Benchmark text embedding latency one query at a time.

This script measures embedding latency for individual queries first, then
optionally compares the same inputs as a batch request. It is intended for
isolating whether slowness comes from cold-start overhead, single-request
throughput, or batching behaviour.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import time

from kazusa_ai_chatbot.config import EMBEDDING_BASE_URL, EMBEDDING_MODEL
from kazusa_ai_chatbot.db import get_text_embedding, get_text_embeddings_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_QUERIES = [
    "hello",
    "DDR5内存现在多少钱",
    "你还记得我之前说过的话吗",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the embedding benchmark.

    Args:
        None.

    Returns:
        Parsed CLI arguments including queries, repeat count, and whether to
        compare a batch request after running single-query timings.
    """
    parser = argparse.ArgumentParser(description="Benchmark text embedding latency")
    parser.add_argument(
        "queries",
        nargs="*",
        help="Query strings to embed. Uses a built-in sample set when omitted.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to run each single-query request.",
    )
    parser.add_argument(
        "--compare-batch",
        action="store_true",
        help="After all single-query timings, run one batch timing over the same queries.",
    )
    return parser.parse_args()


async def benchmark_single_query(query: str, repeat: int) -> dict:
    """Measure embedding latency for a single query across repeated runs.

    Args:
        query: Text to embed.
        repeat: Number of sequential single-query requests to run.

    Returns:
        Summary dict containing the query text, embedding dimension, per-run
        timings, and aggregate latency statistics.
    """
    timings: list[float] = []
    embedding_dim = 0

    for attempt in range(1, repeat + 1):
        start = time.perf_counter()
        embedding = await get_text_embedding(query)
        elapsed = time.perf_counter() - start
        embedding_dim = len(embedding)
        timings.append(elapsed)
        logger.info(
            "single query attempt=%d query=%r elapsed_seconds=%.3f embedding_dim=%d",
            attempt,
            query,
            elapsed,
            embedding_dim,
        )

    return {
        "query": query,
        "embedding_dim": embedding_dim,
        "timings": timings,
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "avg_seconds": statistics.mean(timings),
    }


async def benchmark_batch(queries: list[str]) -> dict:
    """Measure a single batch embedding request for multiple queries.

    Args:
        queries: Query texts to embed together in one batch request.

    Returns:
        Summary dict containing batch latency, per-item latency estimate, and
        returned embedding dimensions.
    """
    start = time.perf_counter()
    embeddings = await get_text_embeddings_batch(queries)
    elapsed = time.perf_counter() - start
    embedding_dims = [len(embedding) for embedding in embeddings]
    logger.info(
        "batch query_count=%d elapsed_seconds=%.3f embedding_dims=%r",
        len(queries),
        elapsed,
        embedding_dims,
    )
    return {
        "query_count": len(queries),
        "elapsed_seconds": elapsed,
        "per_item_seconds": elapsed / max(1, len(queries)),
        "embedding_dims": embedding_dims,
    }


def print_single_summary(results: list[dict]) -> None:
    """Print a concise summary for the single-query benchmark results.

    Args:
        results: Result dicts returned by ``benchmark_single_query``.

    Returns:
        None.
    """
    print("\n=== Single-query summary ===")
    for result in results:
        print(
            f"query={result['query']!r} "
            f"dim={result['embedding_dim']} "
            f"min={result['min_seconds']:.3f}s "
            f"avg={result['avg_seconds']:.3f}s "
            f"max={result['max_seconds']:.3f}s"
        )


def print_batch_summary(result: dict) -> None:
    """Print a concise summary for the batch benchmark result.

    Args:
        result: Result dict returned by ``benchmark_batch``.

    Returns:
        None.
    """
    print("\n=== Batch summary ===")
    print(
        f"queries={result['query_count']} "
        f"elapsed={result['elapsed_seconds']:.3f}s "
        f"per_item_estimate={result['per_item_seconds']:.3f}s "
        f"dims={result['embedding_dims']}"
    )


async def main() -> None:
    """Run the text embedding benchmark from the command line.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()
    queries = args.queries or list(DEFAULT_QUERIES)

    logger.info("Embedding endpoint: %s", EMBEDDING_BASE_URL)
    logger.info("Embedding model: %s", EMBEDDING_MODEL)
    logger.info("Single-query benchmark count: %d", len(queries))
    logger.info("Repeat per query: %d", args.repeat)

    single_results: list[dict] = []
    for query in queries:
        single_results.append(await benchmark_single_query(query, args.repeat))
    print_single_summary(single_results)

    if args.compare_batch:
        batch_result = await benchmark_batch(queries)
        print_batch_summary(batch_result)


if __name__ == "__main__":
    asyncio.run(main())


def async_main() -> None:
    """Provide a synchronous wrapper used by some local script runners.

    Args:
        None.

    Returns:
        None.
    """
    asyncio.run(main())
