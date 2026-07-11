"""Retention and reclamation for immutable repository-index snapshots."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.repository_index.identity import (
    source_identity_hash,
)


def reclaim_unpinned_snapshots(
    *,
    workspace_root: Path,
    source_identity: Mapping[str, object],
    ledgers: Iterable[Mapping[str, object]],
    active_snapshot_ids: set[str],
) -> list[str]:
    """Remove only complete snapshots that no live ledger or cursor pins."""

    pinned_snapshot_ids = set(active_snapshot_ids)
    for ledger in ledgers:
        if ledger.get("status") == "archived":
            continue
        snapshot_id = ledger.get("index_snapshot_id")
        if isinstance(snapshot_id, str) and snapshot_id:
            pinned_snapshot_ids.add(snapshot_id)
    index_directory = _index_directory(workspace_root, source_identity)
    if not index_directory.is_dir():
        return []
    reclaimed_snapshot_ids: list[str] = []
    for database_path in sorted(index_directory.glob("*.sqlite")):
        if database_path.is_symlink():
            continue
        if database_path.name.endswith(".building.sqlite"):
            continue
        snapshot_id = database_path.stem
        if snapshot_id in pinned_snapshot_ids:
            continue
        if _snapshot_status(database_path) != "complete":
            continue
        database_path.unlink()
        reclaimed_snapshot_ids.append(snapshot_id)
    return reclaimed_snapshot_ids


def pin_snapshot(
    *,
    workspace_root: Path,
    source_identity: Mapping[str, object],
    snapshot_id: str,
    owner_id: str,
) -> None:
    """Persist one explicit owner pin for a complete immutable snapshot.

    Callers hold the Phase C source/run lock while updating this metadata.
    """

    if not snapshot_id or not owner_id:
        raise ValueError("snapshot pin identity is required")
    index_directory = _index_directory(workspace_root, source_identity)
    database_path = index_directory / f"{snapshot_id}.sqlite"
    if not database_path.is_file() or _snapshot_status(database_path) != "complete":
        raise ValueError("only complete snapshots may be pinned")
    pins = _load_pins(index_directory)
    owners = pins.setdefault(snapshot_id, [])
    if owner_id not in owners:
        owners.append(owner_id)
        owners.sort()
    _write_pins(index_directory, pins)


def release_snapshot(
    *,
    workspace_root: Path,
    source_identity: Mapping[str, object],
    snapshot_id: str,
    owner_id: str,
) -> None:
    """Release one explicit owner pin under the caller-held source/run lock."""

    index_directory = _index_directory(workspace_root, source_identity)
    pins = _load_pins(index_directory)
    owners = pins.get(snapshot_id, [])
    if owner_id in owners:
        owners.remove(owner_id)
    if owners:
        pins[snapshot_id] = owners
    else:
        pins.pop(snapshot_id, None)
    _write_pins(index_directory, pins)


def reclaim_released_snapshots(
    *,
    workspace_root: Path,
    source_identity: Mapping[str, object],
    active_snapshot_ids: set[str] | None = None,
) -> list[str]:
    """Reclaim only complete snapshots without explicit or active-cursor pins."""

    index_directory = _index_directory(workspace_root, source_identity)
    pins = _load_pins(index_directory)
    active_ids = active_snapshot_ids or set()
    reclaimed: list[str] = []
    for database_path in sorted(index_directory.glob("*.sqlite")):
        if database_path.is_symlink():
            continue
        if database_path.name.endswith(".building.sqlite"):
            continue
        snapshot_id = database_path.stem
        if snapshot_id in active_ids or pins.get(snapshot_id):
            continue
        if _snapshot_status(database_path) != "complete":
            continue
        database_path.unlink()
        reclaimed.append(snapshot_id)
    return reclaimed


def _index_directory(
    workspace_root: Path,
    source_identity: Mapping[str, object],
) -> Path:
    """Return the one deterministic source-identity index directory."""

    resolved_workspace = workspace_root.resolve(strict=False)
    index_root = resolved_workspace / "repository_indexes"
    index_directory = index_root / source_identity_hash(dict(source_identity))
    if index_root.is_symlink() or index_directory.is_symlink():
        raise ValueError("repository index directory is unsafe")
    if not index_directory.resolve(strict=False).is_relative_to(
        resolved_workspace,
    ):
        raise ValueError("repository index directory is unsafe")
    return index_directory


def _load_pins(index_directory: Path) -> dict[str, list[str]]:
    """Load bounded explicit pin ownership without inferring live state."""

    path = index_directory / "pins.json"
    if not path.is_file():
        return {}
    if path.is_symlink():
        raise ValueError("repository snapshot pins are invalid")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("repository snapshot pins are invalid")
    pins: dict[str, list[str]] = {}
    for snapshot_id, owners in loaded.items():
        if not isinstance(snapshot_id, str) or not isinstance(owners, list):
            raise ValueError("repository snapshot pins are invalid")
        if not all(isinstance(owner, str) and owner for owner in owners):
            raise ValueError("repository snapshot pins are invalid")
        pins[snapshot_id] = sorted(set(owners))
    return pins


def _write_pins(index_directory: Path, pins: Mapping[str, list[str]]) -> None:
    """Atomically replace explicit pin metadata under the caller-held lock."""

    index_directory.mkdir(parents=True, exist_ok=True)
    target_path = index_directory / "pins.json"
    temporary_path = target_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(dict(pins), sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(target_path)


def _snapshot_status(database_path: Path) -> str:
    """Return the persisted status for one candidate snapshot database."""

    connection = sqlite3.connect(database_path)
    try:
        row = connection.execute(
            "SELECT status FROM snapshot LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    if row is None or not isinstance(row[0], str):
        return "invalid"
    return row[0]
