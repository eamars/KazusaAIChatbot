"""Consolidation metadata contract helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def finalize_consolidation_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the public consolidation metadata envelope.

    Args:
        metadata: Branch-produced metadata from the consolidation pipeline.

    Returns:
        Metadata with required lifecycle telemetry present for every producer
        branch.

    Raises:
        ValueError: If a present metadata field has an invalid type.
    """

    if metadata is None:
        finalized: dict[str, Any] = {}
    elif isinstance(metadata, Mapping):
        finalized = dict(metadata)
    else:
        raise ValueError("consolidation metadata must be a mapping")

    raw_write_success = finalized.get("write_success")
    if raw_write_success is None:
        write_success: dict[str, Any] = {}
    elif isinstance(raw_write_success, Mapping):
        write_success = dict(raw_write_success)
    else:
        raise ValueError("consolidation metadata write_success must be a dict")
    finalized["write_success"] = write_success

    raw_cache_invalidated = finalized.get("cache_invalidated")
    if raw_cache_invalidated is None:
        cache_invalidated: list[str] = []
    elif isinstance(raw_cache_invalidated, list):
        cache_invalidated = []
        for cache_source in raw_cache_invalidated:
            if not isinstance(cache_source, str):
                raise ValueError(
                    "consolidation metadata cache_invalidated must be "
                    "a list of strings"
                )
            cache_invalidated.append(cache_source)
    else:
        raise ValueError(
            "consolidation metadata cache_invalidated must be a list"
        )
    finalized["cache_invalidated"] = cache_invalidated

    raw_cache_evicted_count = finalized.get("cache_evicted_count")
    if raw_cache_evicted_count is None:
        cache_evicted_count = 0
    elif (
        isinstance(raw_cache_evicted_count, int)
        and not isinstance(raw_cache_evicted_count, bool)
    ):
        cache_evicted_count = raw_cache_evicted_count
    else:
        raise ValueError(
            "consolidation metadata cache_evicted_count must be an int"
        )
    finalized["cache_evicted_count"] = cache_evicted_count

    return_value = finalized
    return return_value
