"""Snapshot and restore the singleton character_state document.

Typical use:
    python -m scripts.character_state_snapshot snapshot
    python -m scripts.character_state_snapshot restore
    python -m scripts.character_state_snapshot snapshot --file test_artifacts/state.json
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bson import json_util

from scripts._db_export import configure_logging, configure_stdout, load_project_env
from kazusa_ai_chatbot.db import close_db, get_db

DEFAULT_SNAPSHOT_PATH = Path("test_artifacts") / "character_state_snapshot.json"
SNAPSHOT_TYPE = "character_state"
SNAPSHOT_VERSION = 1


def _utc_now_iso() -> str:
    """Return the current UTC time in a JSON-friendly string.

    Returns:
        ISO-8601 timestamp with a UTC offset.
    """
    return_value = datetime.now(timezone.utc).isoformat()
    return return_value


def build_snapshot_payload(document: dict[str, Any]) -> dict[str, Any]:
    """Build the on-disk snapshot payload for the character singleton.

    Args:
        document: Full MongoDB ``character_state`` document to persist locally.

    Returns:
        Snapshot envelope containing metadata and the captured state document.

    Raises:
        ValueError: If the document does not represent ``_id: "global"``.
    """
    document_id = document.get("_id")
    if document_id != "global":
        raise ValueError(
            f'character_state snapshot requires _id "global", got {document_id!r}'
        )

    payload = {
        "snapshot_type": SNAPSHOT_TYPE,
        "snapshot_version": SNAPSHOT_VERSION,
        "created_at": _utc_now_iso(),
        "query": {
            "collection": "character_state",
            "_id": "global",
        },
        "character_state": dict(document),
    }
    return payload


def extract_character_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and extract a restorable character_state document.

    Args:
        payload: Parsed snapshot or legacy export payload.

    Returns:
        Full replacement document for ``character_state/_id: global``.

    Raises:
        ValueError: If the payload cannot safely restore the singleton state.
    """
    state = payload.get("character_state")
    if not isinstance(state, dict) or not state:
        raise ValueError(
            "snapshot file must contain a non-empty character_state object"
        )

    restored_state = dict(state)
    restored_id = restored_state.setdefault("_id", "global")
    if restored_id != "global":
        raise ValueError(
            f'character_state restore requires _id "global", got {restored_id!r}'
        )

    return restored_state


def write_snapshot_file(file_path: Path, payload: dict[str, Any]) -> None:
    """Write a snapshot payload as relaxed Extended JSON.

    Args:
        file_path: Destination file path.
        payload: Snapshot envelope to serialize.

    Returns:
        None.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json_util.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_snapshot_file(file_path: Path) -> dict[str, Any]:
    """Read a snapshot payload from relaxed Extended JSON.

    Args:
        file_path: Source file path.

    Returns:
        Parsed snapshot envelope.

    Raises:
        ValueError: If the file does not contain a JSON object.
    """
    payload = json_util.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("snapshot file must decode to a JSON object")
    return payload


async def snapshot_character_state(file_path: Path) -> dict[str, Any]:
    """Capture the current character_state singleton into a local file.

    Args:
        file_path: Destination snapshot file.

    Returns:
        Snapshot payload written to disk.

    Raises:
        ValueError: If the singleton document does not exist.
    """
    db = await get_db()
    document = await db.character_state.find_one({"_id": "global"})
    if document is None:
        raise ValueError('character_state document "_id: global" does not exist')

    payload = build_snapshot_payload(dict(document))
    write_snapshot_file(file_path, payload)
    return payload


async def restore_character_state(file_path: Path) -> dict[str, Any]:
    """Restore the character_state singleton from a local snapshot file.

    Args:
        file_path: Source snapshot file.

    Returns:
        Replacement document written to MongoDB.
    """
    payload = read_snapshot_file(file_path)
    restored_state = extract_character_state(payload)

    db = await get_db()
    await db.character_state.replace_one(
        {"_id": "global"},
        restored_state,
        upsert=True,
    )
    return restored_state


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser for snapshot and restore commands.
    """
    parser = argparse.ArgumentParser(
        description="Snapshot or restore character_state/_id: global.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser(
        "snapshot",
        help="Write the current character_state document to a local file.",
    )
    snapshot_parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_SNAPSHOT_PATH,
        help=f"Snapshot path. Defaults to {DEFAULT_SNAPSHOT_PATH}.",
    )
    snapshot_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show project database logs.",
    )

    restore_parser = subparsers.add_parser(
        "restore",
        help="Replace character_state/_id: global from a local snapshot file.",
    )
    restore_parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_SNAPSHOT_PATH,
        help=f"Snapshot path. Defaults to {DEFAULT_SNAPSHOT_PATH}.",
    )
    restore_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show project database logs.",
    )

    return parser


async def main() -> None:
    """Run the character-state snapshot CLI.

    Returns:
        None.
    """
    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    try:
        if args.command == "snapshot":
            payload = await snapshot_character_state(args.file)
            created_at = payload["created_at"]
            print(f"wrote character state snapshot to {args.file} at {created_at}")
        else:
            restored_state = await restore_character_state(args.file)
            updated_at = restored_state.get("updated_at", "<missing>")
            print(f"restored character state from {args.file}; updated_at={updated_at}")
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper for the async CLI.

    Returns:
        None.
    """
    asyncio.run(main())


if __name__ == "__main__":
    async_main()
