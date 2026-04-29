"""Maintain curated world/common-sense entries for the shared memory collection.

The JSONL file is the source of truth. MongoDB is the indexed runtime copy used
by RAG2 persistent-memory retrieval.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts._db_export import configure_logging, configure_stdout, load_project_env
from kazusa_ai_chatbot.db import close_db, get_db
from kazusa_ai_chatbot.db._client import get_text_embedding

DEFAULT_KNOWLEDGE_PATH = Path("knowledge/memory_seed.jsonl")
VALID_MEMORY_TYPES = {"fact", "narrative", "impression", "defense_rule"}
VALID_SOURCE_KINDS = {"seeded_manual", "external_imported"}
VALID_STATUSES = {"active", "fulfilled", "expired", "superseded"}
MANAGED_FIELDS = (
    "memory_name",
    "content",
    "source_global_user_id",
    "memory_type",
    "source_kind",
    "confidence_note",
    "status",
    "expiry_timestamp",
)


@dataclass(frozen=True)
class KnowledgeEntry:
    """One curated shared-memory row loaded from the JSONL source.

    Args:
        data: Normalized row payload matching the shared ``memory`` schema.
        line_number: One-based line number in the JSONL file.
    """

    data: dict[str, Any]
    line_number: int

    @property
    def key(self) -> tuple[str, str]:
        """Return the stable de-duplication key for this entry.

        Returns:
            Tuple of ``(memory_name, source_global_user_id)``.
        """
        return_value = (
            str(self.data["memory_name"]).strip(),
            str(self.data["source_global_user_id"]).strip(),
        )
        return return_value


def _now_iso() -> str:
    """Return a timezone-aware UTC timestamp string.

    Returns:
        ISO-8601 UTC timestamp.
    """
    return_value = datetime.now(timezone.utc).isoformat()
    return return_value


def _combined_embedding_text(entry: dict[str, Any]) -> str:
    """Build the text payload embedded by ``db.memory.save_memory``.

    Args:
        entry: Normalized memory entry.

    Returns:
        Combined text used to produce the vector embedding.
    """
    return_value = (
        f"type:{entry.get('memory_type', '')}\n"
        f"source:{entry.get('source_kind', '')}\n"
        f"title:{entry['memory_name']}\n"
        f"content:{entry['content']}"
    )
    return return_value


def _managed_payload(entry: dict[str, Any]) -> dict[str, Any]:
    """Select the fields controlled by the JSONL source.

    Args:
        entry: Normalized memory entry.

    Returns:
        Dict containing only managed DB fields.
    """
    return_value = {field: entry[field] for field in MANAGED_FIELDS}
    return return_value


def _normalize_entry(raw: dict[str, Any], line_number: int) -> KnowledgeEntry:
    """Normalize one raw JSONL object into a knowledge entry.

    Args:
        raw: Parsed JSON object from one line.
        line_number: One-based source line number.

    Returns:
        Normalized knowledge entry.
    """
    data = {
        "memory_name": str(raw.get("memory_name", "")).strip(),
        "content": str(raw.get("content", "")).strip(),
        "source_global_user_id": str(raw.get("source_global_user_id", "")).strip(),
        "memory_type": str(raw.get("memory_type", "fact")).strip(),
        "source_kind": str(raw.get("source_kind", "seeded_manual")).strip(),
        "confidence_note": str(raw.get("confidence_note", "")).strip(),
        "status": str(raw.get("status", "active")).strip(),
        "expiry_timestamp": raw.get("expiry_timestamp"),
    }
    return_value = KnowledgeEntry(data=data, line_number=line_number)
    return return_value


def load_entries(path: Path) -> list[KnowledgeEntry]:
    """Load and normalize every entry from a JSONL source file.

    Args:
        path: JSONL file path.

    Returns:
        Ordered list of knowledge entries.
    """
    entries: list[KnowledgeEntry] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"Line {line_number} must be a JSON object.")
        entries.append(_normalize_entry(parsed, line_number))
    return entries


def write_entries(path: Path, entries: list[KnowledgeEntry]) -> None:
    """Write entries back to JSONL with stable field ordering.

    Args:
        path: Destination JSONL path.
        entries: Entries to write in order.

    Returns:
        None.
    """
    lines = [
        json.dumps(_managed_payload(entry.data), ensure_ascii=False, separators=(",", ":"))
        for entry in entries
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def validate_entries(entries: list[KnowledgeEntry]) -> list[str]:
    """Validate source rows and return human-readable problems.

    Args:
        entries: Loaded knowledge entries.

    Returns:
        List of validation error strings. Empty means the file is valid.
    """
    errors: list[str] = []
    seen_keys: dict[tuple[str, str], int] = {}
    for entry in entries:
        line = entry.line_number
        data = entry.data
        if not data["memory_name"]:
            errors.append(f"Line {line}: memory_name is required.")
        if not data["content"]:
            errors.append(f"Line {line}: content is required.")
        if data["source_global_user_id"]:
            errors.append(f"Line {line}: source_global_user_id must be empty for curated shared knowledge.")
        if data["memory_type"] not in VALID_MEMORY_TYPES:
            errors.append(f"Line {line}: memory_type must be one of {sorted(VALID_MEMORY_TYPES)}.")
        if data["source_kind"] not in VALID_SOURCE_KINDS:
            errors.append(f"Line {line}: source_kind must be one of {sorted(VALID_SOURCE_KINDS)}.")
        if data["status"] not in VALID_STATUSES:
            errors.append(f"Line {line}: status must be one of {sorted(VALID_STATUSES)}.")
        if data["expiry_timestamp"] is not None and not str(data["expiry_timestamp"]).strip():
            errors.append(f"Line {line}: expiry_timestamp must be null or a non-empty ISO timestamp.")

        key = entry.key
        if key in seen_keys:
            errors.append(f"Line {line}: duplicate key also appears on line {seen_keys[key]}.")
        else:
            seen_keys[key] = line
    return errors


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Maintain curated shared memory knowledge.")
    parser.add_argument("--file", type=Path, default=DEFAULT_KNOWLEDGE_PATH, help="Knowledge JSONL file.")
    parser.add_argument("--verbose", action="store_true", help="Show project database logs.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="Validate the JSONL source file.")
    subparsers.add_parser("list", help="List entries from the JSONL source file.")

    search_parser = subparsers.add_parser("search", help="Search local entries by substring.")
    search_parser.add_argument("query", help="Case-insensitive text to find in name or content.")

    add_parser = subparsers.add_parser("add", help="Add one local entry.")
    add_parser.add_argument("--name", required=True, help="Stable memory_name.")
    add_parser.add_argument("--content", required=True, help="Memory content.")
    add_parser.add_argument("--memory-type", default="fact", choices=sorted(VALID_MEMORY_TYPES))
    add_parser.add_argument("--source-kind", default="seeded_manual", choices=sorted(VALID_SOURCE_KINDS))
    add_parser.add_argument("--note", default="人工整理的共享知识。", help="Confidence/provenance note.")
    add_parser.add_argument("--replace", action="store_true", help="Replace an existing local entry with the same key.")

    remove_parser = subparsers.add_parser("remove", help="Remove one local entry by memory_name.")
    remove_parser.add_argument("--name", required=True, help="Entry name to remove.")

    sync_parser = subparsers.add_parser("sync", help="Synchronize JSONL entries into MongoDB.")
    sync_parser.add_argument("--apply", action="store_true", help="Apply changes. Omit for a dry run.")
    sync_parser.add_argument(
        "--prune-unmanaged-global",
        action="store_true",
        help="Delete global memory rows not present in the JSONL file.",
    )

    return parser


def _print_entries(entries: list[KnowledgeEntry]) -> None:
    """Print a compact local-entry listing.

    Args:
        entries: Entries to display.

    Returns:
        None.
    """
    for entry in entries:
        print(f"{entry.line_number}: {entry.data['memory_name']} [{entry.data['source_kind']}]")


def _validate_or_raise(entries: list[KnowledgeEntry]) -> None:
    """Raise when entries fail validation.

    Args:
        entries: Loaded entries.

    Returns:
        None.
    """
    errors = validate_entries(entries)
    if errors:
        raise ValueError("\n".join(errors))


def _entry_differs(db_doc: dict[str, Any], entry: dict[str, Any]) -> bool:
    """Return whether the DB row differs from the managed source fields.

    Args:
        db_doc: MongoDB memory document.
        entry: Normalized source entry.

    Returns:
        True when at least one managed field differs.
    """
    for field in MANAGED_FIELDS:
        if db_doc.get(field) != entry[field]:
            return True
    return False


async def _payload_with_embedding(entry: dict[str, Any], timestamp: str) -> dict[str, Any]:
    """Build a MongoDB payload with a fresh embedding.

    Args:
        entry: Normalized source entry.
        timestamp: Timestamp to store on the DB row.

    Returns:
        Full memory document payload.
    """
    embedding = await get_text_embedding(_combined_embedding_text(entry))
    return_value = {
        **_managed_payload(entry),
        "timestamp": timestamp,
        "embedding": embedding,
    }
    return return_value


async def sync_entries(
    entries: list[KnowledgeEntry],
    *,
    apply: bool,
    prune_unmanaged_global: bool,
) -> dict[str, int]:
    """Synchronize source entries into MongoDB and remove duplicate keys.

    Args:
        entries: Validated local source entries.
        apply: Whether to mutate the database.
        prune_unmanaged_global: Whether to delete global memory rows absent from
            the JSONL source.

    Returns:
        Counters describing planned or applied changes.
    """
    db = await get_db()
    counters = {
        "inserted": 0,
        "updated": 0,
        "unchanged": 0,
        "duplicates_deleted": 0,
        "pruned": 0,
    }
    source_names = {entry.data["memory_name"] for entry in entries}

    for entry in entries:
        key_filter = {
            "memory_name": entry.data["memory_name"],
            "source_global_user_id": entry.data["source_global_user_id"],
        }
        existing = await db.memory.find(key_filter).sort("timestamp", 1).to_list(length=None)
        timestamp = _now_iso()
        if not existing:
            counters["inserted"] += 1
            if apply:
                await db.memory.insert_one(await _payload_with_embedding(entry.data, timestamp))
            continue

        primary = existing[0]
        duplicate_ids = [doc["_id"] for doc in existing[1:]]
        if duplicate_ids:
            counters["duplicates_deleted"] += len(duplicate_ids)
            if apply:
                await db.memory.delete_many({"_id": {"$in": duplicate_ids}})

        if _entry_differs(primary, entry.data):
            counters["updated"] += 1
            if apply:
                payload = await _payload_with_embedding(entry.data, timestamp)
                await db.memory.update_one({"_id": primary["_id"]}, {"$set": payload})
        else:
            counters["unchanged"] += 1

    if prune_unmanaged_global:
        prune_filter = {
            "source_global_user_id": "",
            "memory_name": {"$nin": sorted(source_names)},
        }
        prune_count = await db.memory.count_documents(prune_filter)
        counters["pruned"] = prune_count
        if apply and prune_count:
            await db.memory.delete_many(prune_filter)

    return counters


def _add_entry(args: argparse.Namespace) -> None:
    """Add or replace one local JSONL entry.

    Args:
        args: Parsed CLI arguments.

    Returns:
        None.
    """
    entries = load_entries(args.file) if args.file.exists() else []
    new_entry = _normalize_entry(
        {
            "memory_name": args.name,
            "content": args.content,
            "source_global_user_id": "",
            "memory_type": args.memory_type,
            "source_kind": args.source_kind,
            "confidence_note": args.note,
            "status": "active",
            "expiry_timestamp": None,
        },
        line_number=len(entries) + 1,
    )
    existing_index = next(
        (index for index, entry in enumerate(entries) if entry.key == new_entry.key),
        None,
    )
    if existing_index is not None and not args.replace:
        raise ValueError(f"Entry already exists: {args.name}. Use --replace to overwrite it.")
    if existing_index is None:
        entries.append(new_entry)
    else:
        entries[existing_index] = KnowledgeEntry(data=new_entry.data, line_number=entries[existing_index].line_number)
    _validate_or_raise(entries)
    write_entries(args.file, entries)
    print(f"saved local entry: {args.name}")


def _remove_entry(args: argparse.Namespace) -> None:
    """Remove one local JSONL entry.

    Args:
        args: Parsed CLI arguments.

    Returns:
        None.
    """
    entries = load_entries(args.file)
    remaining = [entry for entry in entries if entry.data["memory_name"] != args.name]
    removed = len(entries) - len(remaining)
    if not removed:
        raise ValueError(f"No local entry named {args.name!r}.")
    write_entries(args.file, remaining)
    print(f"removed {removed} local entry row(s): {args.name}")


async def _run_async(args: argparse.Namespace) -> None:
    """Run commands that may need MongoDB.

    Args:
        args: Parsed CLI arguments.

    Returns:
        None.
    """
    entries = load_entries(args.file)
    _validate_or_raise(entries)
    try:
        counters = await sync_entries(
            entries,
            apply=args.apply,
            prune_unmanaged_global=args.prune_unmanaged_global,
        )
    finally:
        await close_db()
    mode = "applied" if args.apply else "dry-run"
    print(f"sync {mode}: {json.dumps(counters, ensure_ascii=False, sort_keys=True)}")


def main() -> None:
    """Run the knowledge-maintenance CLI.

    Returns:
        None.
    """
    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    if args.command == "add":
        _add_entry(args)
        return
    if args.command == "remove":
        _remove_entry(args)
        return

    entries = load_entries(args.file)
    if args.command == "validate":
        errors = validate_entries(entries)
        if errors:
            raise ValueError("\n".join(errors))
        print(f"valid: {len(entries)} entries")
        return
    if args.command == "list":
        _print_entries(entries)
        return
    if args.command == "search":
        query = args.query.casefold()
        matches = [
            entry
            for entry in entries
            if query in entry.data["memory_name"].casefold() or query in entry.data["content"].casefold()
        ]
        _print_entries(matches)
        print(f"matches: {len(matches)}")
        return
    if args.command == "sync":
        asyncio.run(_run_async(args))
        return


def async_main() -> None:
    """Console-script compatible wrapper.

    Returns:
        None.
    """
    main()


if __name__ == "__main__":
    main()
