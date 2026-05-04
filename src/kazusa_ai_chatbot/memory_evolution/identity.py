"""Pure helpers for shared memory ids and semantic text."""

from __future__ import annotations

import hashlib
import json

from kazusa_ai_chatbot.memory_evolution.models import EvolvingMemoryDoc


def deterministic_memory_unit_id(prefix: str, parts: list[str]) -> str:
    """Build a stable memory-unit id from semantic key parts.

    Args:
        prefix: Human-readable id namespace.
        parts: Ordered stable key parts.

    Returns:
        A deterministic id safe to reuse across retries.
    """
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return_value = f"{prefix}_{digest[:32]}"
    return return_value


def seed_memory_unit_id(
    *,
    memory_name: str,
    source_global_user_id: str,
    source_kind: str,
) -> str:
    """Build the deterministic id for a seed-managed memory row."""
    return_value = deterministic_memory_unit_id(
        "seed",
        [memory_name, source_global_user_id, source_kind],
    )
    return return_value


def memory_embedding_source_text(doc: EvolvingMemoryDoc | dict) -> str:
    """Build the semantic text used for shared-memory embeddings.

    Args:
        doc: Memory document or candidate payload.

    Returns:
        Combined text used by the embedding service.
    """
    memory_name = str(doc.get("memory_name", ""))
    content = str(doc.get("content", ""))
    source_text = (
        f"type:{doc.get('memory_type', '')}\n"
        f"source:{doc.get('source_kind', '')}\n"
        f"title:{memory_name}\n"
        f"content:{content}"
    )
    return source_text
