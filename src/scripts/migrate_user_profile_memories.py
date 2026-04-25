"""Backfill ``user_profile_memories`` from legacy embedded profile arrays."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from kazusa_ai_chatbot.db import MemoryType, get_db, insert_profile_memories


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_memories_from_profile(profile: dict, timestamp: str) -> list[dict]:
    """Convert legacy embedded arrays into profile-memory docs.

    Args:
        profile: Legacy user profile document.
        timestamp: Fallback timestamp for rows without one.

    Returns:
        List of profile-memory docs ready for insertion.
    """
    memories: list[dict] = []
    for entry in profile.get("character_diary") or []:
        content = str(entry.get("entry", "")).strip()
        if content:
            memories.append({
                "memory_type": MemoryType.DIARY_ENTRY,
                "content": content,
                "created_at": entry.get("timestamp") or timestamp,
                "updated_at": timestamp,
                "confidence": entry.get("confidence", 0.8),
                "context": entry.get("context", ""),
            })

    for fact in profile.get("objective_facts") or []:
        content = str(fact.get("fact", "")).strip()
        if content:
            memories.append({
                "memory_type": MemoryType.OBJECTIVE_FACT,
                "content": content,
                "created_at": fact.get("timestamp") or timestamp,
                "updated_at": timestamp,
                "category": fact.get("category", "general"),
                "source": fact.get("source", "legacy_migration"),
                "confidence": fact.get("confidence", 0.8),
                "dedup_key": content.lower(),
            })

    for commitment in profile.get("active_commitments") or []:
        action = str(commitment.get("action", "")).strip()
        if action:
            memories.append({
                "memory_type": MemoryType.COMMITMENT,
                "content": action,
                "action": action,
                "commitment_id": commitment.get("commitment_id", ""),
                "target": commitment.get("target", ""),
                "commitment_type": commitment.get("commitment_type", ""),
                "status": commitment.get("status", "active"),
                "source": commitment.get("source", "legacy_migration"),
                "created_at": commitment.get("created_at") or timestamp,
                "updated_at": timestamp,
                "due_time": commitment.get("due_time"),
                "dedup_key": str(commitment.get("dedup_key") or action).strip().lower(),
            })
    return memories


async def async_main() -> None:
    """Run the legacy profile-memory backfill from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--user-ids", nargs="*", default=[])
    args = parser.parse_args()

    db = await get_db()
    query: dict = {}
    if args.user_ids:
        query["global_user_id"] = {"$in": args.user_ids}

    cursor = db.user_profiles.find(query).batch_size(args.batch_size)
    scanned = 0
    written = 0
    timestamp = _now_iso()
    async for profile in cursor:
        global_user_id = profile.get("global_user_id", "")
        if not global_user_id:
            continue
        memories = _build_memories_from_profile(profile, timestamp)
        scanned += 1
        if args.dry_run:
            written += len(memories)
            continue
        persisted = await insert_profile_memories(global_user_id, memories)
        written += len(persisted)

    print(f"scanned_profiles={scanned} memories={'would_write' if args.dry_run else 'written'}:{written}")


def main() -> None:
    """Synchronous console-script wrapper."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
