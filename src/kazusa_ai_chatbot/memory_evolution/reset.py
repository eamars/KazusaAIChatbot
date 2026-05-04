"""Reset shared memory from repository-owned seed data."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.db import memory_evolution as memory_store
from kazusa_ai_chatbot.memory_evolution import repository
from kazusa_ai_chatbot.memory_evolution.models import (
    EvolvingMemoryDoc,
    MemoryAuthority,
    MemoryResetResult,
    MemorySourceKind,
    MemoryStatus,
    SEED_MANAGED_SOURCE_KINDS,
    VALID_MEMORY_STATUSES,
)

DEFAULT_MEMORY_SEED_PATH = Path("personalities/knowledge/memory_seed.jsonl")


@dataclass(frozen=True)
class SeedMemoryEntry:
    """One normalized seed row and its source line number."""

    data: dict[str, Any]
    line_number: int


def _empty_result(*, dry_run: bool, warnings: list[str]) -> MemoryResetResult:
    return_value: MemoryResetResult = {
        "dry_run": dry_run,
        "seed_rows_loaded": 0,
        "seed_rows_inserted": 0,
        "seed_rows_updated": 0,
        "seed_rows_unchanged": 0,
        "seed_rows_deleted": 0,
        "legacy_rows_deleted": 0,
        "runtime_rows_preserved": 0,
        "embeddings_computed": 0,
        "cache_invalidated": False,
        "warnings": list(warnings),
    }
    return return_value


def _normalize_seed_row(raw: dict[str, Any], line_number: int) -> SeedMemoryEntry:
    data = {
        "memory_name": str(raw.get("memory_name", "")).strip(),
        "content": str(raw.get("content", "")).strip(),
        "source_global_user_id": str(raw.get("source_global_user_id", "")).strip(),
        "memory_type": str(raw.get("memory_type", "fact")).strip(),
        "source_kind": str(raw.get("source_kind", MemorySourceKind.SEEDED_MANUAL)).strip(),
        "confidence_note": str(raw.get("confidence_note", "")).strip(),
        "status": str(raw.get("status", MemoryStatus.ACTIVE)).strip(),
        "expiry_timestamp": raw.get("expiry_timestamp"),
    }
    return_value = SeedMemoryEntry(data=data, line_number=line_number)
    return return_value


def load_seed_entries(path: Path) -> list[SeedMemoryEntry]:
    """Load seed rows from a JSONL file.

    Args:
        path: Source JSONL file path.

    Returns:
        Normalized seed entries in file order.
    """
    entries: list[SeedMemoryEntry] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Line {line_number}: seed row must be a JSON object")
        entries.append(_normalize_seed_row(parsed, line_number))
    return entries


def validate_seed_entries(entries: list[SeedMemoryEntry]) -> list[str]:
    """Validate seed rows before reset.

    Args:
        entries: Normalized seed entries.

    Returns:
        Human-readable validation errors.
    """
    errors: list[str] = []
    seen_keys: dict[tuple[str, str, str], int] = {}
    for entry in entries:
        data = entry.data
        line = entry.line_number
        if not data["memory_name"]:
            errors.append(f"Line {line}: memory_name is required.")
        if not data["content"]:
            errors.append(f"Line {line}: content is required.")
        if data["source_global_user_id"]:
            errors.append(
                f"Line {line}: source_global_user_id must be empty for seed memory."
            )
        if data["source_kind"] not in SEED_MANAGED_SOURCE_KINDS:
            errors.append(
                f"Line {line}: source_kind must be one of "
                f"{sorted(SEED_MANAGED_SOURCE_KINDS)}."
            )
        if data["status"] not in VALID_MEMORY_STATUSES:
            errors.append(
                f"Line {line}: status must be one of {sorted(VALID_MEMORY_STATUSES)}."
            )
        if data["expiry_timestamp"] is not None and not str(
            data["expiry_timestamp"]
        ).strip():
            errors.append(
                f"Line {line}: expiry_timestamp must be null or a non-empty ISO timestamp."
            )
        key = (
            data["memory_name"],
            data["source_global_user_id"],
            data["source_kind"],
        )
        if key in seen_keys:
            errors.append(
                f"Line {line}: duplicate seed key also appears on line {seen_keys[key]}."
            )
        else:
            seen_keys[key] = line
    return errors


def _seed_document(
    entry: dict[str, Any],
    *,
    timestamp: str,
    updated_at: str,
) -> EvolvingMemoryDoc:
    memory_unit_id = repository.seed_memory_unit_id(
        memory_name=entry["memory_name"],
        source_global_user_id=entry["source_global_user_id"],
        source_kind=entry["source_kind"],
    )
    document: EvolvingMemoryDoc = {
        "memory_unit_id": memory_unit_id,
        "lineage_id": memory_unit_id,
        "version": 1,
        "memory_name": entry["memory_name"],
        "content": entry["content"],
        "source_global_user_id": entry["source_global_user_id"],
        "memory_type": entry["memory_type"],
        "source_kind": entry["source_kind"],
        "authority": MemoryAuthority.SEED,
        "status": entry["status"],
        "supersedes_memory_unit_ids": [],
        "merged_from_memory_unit_ids": [],
        "evidence_refs": [
            {
                "captured_at": timestamp,
                "source": "seed_file",
            }
        ],
        "privacy_review": {
            "private_detail_risk": "low",
            "user_details_removed": True,
            "boundary_assessment": "seed-managed shared memory",
            "reviewer": "seed_tool",
        },
        "confidence_note": entry["confidence_note"],
        "timestamp": timestamp,
        "updated_at": updated_at,
        "expiry_timestamp": entry["expiry_timestamp"],
    }
    return document


async def _reset_counts(
    *,
    entries: list[SeedMemoryEntry],
    write_time: str,
    prune_unmanaged_global: bool,
) -> tuple[MemoryResetResult, list[EvolvingMemoryDoc], list[str]]:
    result = _empty_result(dry_run=False, warnings=[])
    result["seed_rows_loaded"] = len(entries)

    seed_ids: list[str] = []
    changed_docs: list[EvolvingMemoryDoc] = []
    for entry in entries:
        seed_id = repository.seed_memory_unit_id(
            memory_name=entry.data["memory_name"],
            source_global_user_id=entry.data["source_global_user_id"],
            source_kind=entry.data["source_kind"],
        )
        seed_ids.append(seed_id)
        existing = await memory_store.find_memory_unit_by_id(seed_id)
        timestamp = str(existing.get("timestamp", write_time)) if existing else write_time
        candidate = _seed_document(
            entry.data,
            timestamp=timestamp,
            updated_at=write_time,
        )
        normalized = repository.normalize_memory_document(
            candidate,
            updated_at=write_time,
        )
        if existing is None:
            result["seed_rows_inserted"] += 1
            changed_docs.append(normalized)
        elif repository.memory_documents_equivalent(existing, normalized):
            result["seed_rows_unchanged"] += 1
        else:
            result["seed_rows_updated"] += 1
            changed_docs.append(normalized)

    if prune_unmanaged_global:
        result["legacy_rows_deleted"] = (
            await memory_store.count_legacy_seed_managed_memory()
        )
        result["seed_rows_deleted"] = await memory_store.count_unmanaged_seed_memory(
            seed_ids
        )

    result["runtime_rows_preserved"] = (
        await memory_store.count_runtime_reflection_memory()
    )
    return result, changed_docs, seed_ids


async def reset_memory_from_entries(
    entries_data: list[dict[str, Any]],
    *,
    dry_run: bool,
    prune_unmanaged_global: bool,
) -> MemoryResetResult:
    """Reset seed-managed memory rows from already-loaded seed data.

    Args:
        entries_data: Seed rows from the repository JSONL source.
        dry_run: Whether to report changes without mutating MongoDB.
        prune_unmanaged_global: Whether absent seed-managed global rows should
            be deleted.

    Returns:
        Reset counters and warnings.
    """
    entries = [
        _normalize_seed_row(raw, line_number=index + 1)
        for index, raw in enumerate(entries_data)
    ]
    errors = validate_seed_entries(entries)
    if errors:
        raise ValueError("\n".join(errors))

    write_time = repository.now_iso()
    if dry_run:
        result, _, _ = await _reset_counts(
            entries=entries,
            write_time=write_time,
            prune_unmanaged_global=prune_unmanaged_global,
        )
        result["dry_run"] = True
        return result

    lock_acquired = await memory_store.acquire_memory_write_lock(
        "memory_seed_reset",
        write_time,
    )
    if not lock_acquired:
        raise RuntimeError("memory write or reset is already running")
    try:
        result, changed_docs, seed_ids = await _reset_counts(
            entries=entries,
            write_time=write_time,
            prune_unmanaged_global=prune_unmanaged_global,
        )
        result["dry_run"] = False
        if prune_unmanaged_global:
            await memory_store.delete_reset_seed_managed_memory(seed_ids)
        payloads = await asyncio.gather(
            *(
                repository.document_with_embedding(document)
                for document in changed_docs
            )
        )
        for payload in payloads:
            await memory_store.replace_memory_unit_document(payload)
        result["embeddings_computed"] = len(payloads)
        await repository.invalidate_memory_cache(
            document={
                "source_global_user_id": "",
                "updated_at": write_time,
            },
            reason="memory_seed_reset",
        )
        result["cache_invalidated"] = True
    finally:
        await memory_store.release_memory_write_lock()

    return result


async def reset_memory_from_seed_path(
    *,
    dry_run: bool,
    seed_path: Path,
    prune_unmanaged_global: bool = True,
) -> MemoryResetResult:
    """Reset memory using a specific seed file path.

    Args:
        dry_run: Whether to report changes without mutating MongoDB.
        seed_path: JSONL seed source path.
        prune_unmanaged_global: Whether absent seed-managed global rows should
            be deleted.

    Returns:
        Reset counters and warnings.
    """
    warnings: list[str] = []
    if not seed_path.exists():
        warnings.append(f"seed file missing: {seed_path}")
        entries_data: list[dict[str, Any]] = []
    else:
        entries = load_seed_entries(seed_path)
        errors = validate_seed_entries(entries)
        if errors:
            raise ValueError("\n".join(errors))
        entries_data = [entry.data for entry in entries]

    result = await reset_memory_from_entries(
        entries_data,
        dry_run=dry_run,
        prune_unmanaged_global=prune_unmanaged_global,
    )
    result["warnings"].extend(warnings)
    return result


async def reset_memory_from_seed(*, dry_run: bool) -> MemoryResetResult:
    """Reset memory from the repository's configured seed file.

    Args:
        dry_run: Whether to report changes without mutating MongoDB.

    Returns:
        Reset counters and warnings.
    """
    result = await reset_memory_from_seed_path(
        dry_run=dry_run,
        seed_path=DEFAULT_MEMORY_SEED_PATH,
        prune_unmanaged_global=True,
    )
    return result
