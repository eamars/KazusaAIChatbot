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
from kazusa_ai_chatbot.db import (
    get_document_text_embedding,
    get_document_text_embeddings_batch,
    get_query_text_embedding,
    get_query_text_embeddings_batch,
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
    parser.add_argument(
        "--role",
        choices=("query", "document"),
        default="query",
        help="Embedding role to benchmark.",
    )
    return_value = parser.parse_args()
    return return_value


async def benchmark_single_query(query: str, repeat: int, role: str) -> dict:
    """Measure embedding latency for a single query across repeated runs.

    Args:
        query: Text to embed.
        repeat: Number of sequential single-query requests to run.
        role: Embedding role to benchmark.

    Returns:
        Summary dict containing the query text, embedding dimension, per-run
        timings, and aggregate latency statistics.
    """
    timings: list[float] = []
    embedding_dim = 0

    for attempt in range(1, repeat + 1):
        start = time.perf_counter()
        if role == "document":
            embedding = await get_document_text_embedding(query)
        else:
            embedding = await get_query_text_embedding(query)
        elapsed = time.perf_counter() - start
        embedding_dim = len(embedding)
        timings.append(elapsed)
        logger.info(
            f"single query attempt={attempt} query={query!r} "
            f"elapsed_seconds={elapsed:.3f} embedding_dim={embedding_dim}"
        )

    return_value = {
        "query": query,
        "role": role,
        "embedding_dim": embedding_dim,
        "timings": timings,
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "avg_seconds": statistics.mean(timings),
    }
    return return_value


async def benchmark_batch(queries: list[str], role: str) -> dict:
    """Measure a single batch embedding request for multiple queries.

    Args:
        queries: Query texts to embed together in one batch request.
        role: Embedding role to benchmark.

    Returns:
        Summary dict containing batch latency, per-item latency estimate, and
        returned embedding dimensions.
    """
    start = time.perf_counter()
    if role == "document":
        embeddings = await get_document_text_embeddings_batch(queries)
    else:
        embeddings = await get_query_text_embeddings_batch(queries)
    elapsed = time.perf_counter() - start
    embedding_dims = [len(embedding) for embedding in embeddings]
    logger.info(
        f"batch query_count={len(queries)} elapsed_seconds={elapsed:.3f} "
        f"embedding_dims={embedding_dims!r}"
    )
    return_value = {
        "query_count": len(queries),
        "role": role,
        "elapsed_seconds": elapsed,
        "per_item_seconds": elapsed / max(1, len(queries)),
        "embedding_dims": embedding_dims,
    }
    return return_value


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
            f"role={result['role']} "
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
        f"role={result['role']} "
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

    logger.info(f'Embedding endpoint: {EMBEDDING_BASE_URL}')
    logger.info(f'Embedding model: {EMBEDDING_MODEL}')
    logger.info(f'Single-query benchmark count: {len(queries)}')
    logger.info(f'Repeat per query: {args.repeat}')
    logger.info(f'Embedding role: {args.role}')

    single_results: list[dict] = []
    for query in queries:
        single_results.append(
            await benchmark_single_query(query, args.repeat, args.role)
        )
    print_single_summary(single_results)

    if args.compare_batch:
        batch_result = await benchmark_batch(queries, args.role)
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
