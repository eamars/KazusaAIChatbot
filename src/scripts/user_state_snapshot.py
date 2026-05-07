"""Snapshot and restore all authoritative state related to one user.

Typical use:
    python -m scripts.user_state_snapshot snapshot 3167827653 --platform qq
    python -m scripts.user_state_snapshot snapshot 263c883d-aeff-4e0b-a758-6f69186ae8ec
    python -m scripts.user_state_snapshot restore --file test_artifacts/user_state.json
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bson import json_util

from scripts._db_export import (
    configure_logging,
    configure_stdout,
    load_project_env,
    scrub_document,
)
from kazusa_ai_chatbot.db import close_db, get_user_profile
from kazusa_ai_chatbot.db.script_operations import (
    find_user_profile_for_export,
    load_user_state_alias_profile_refs,
    load_user_state_snapshot_documents,
    recalculate_user_state_embeddings,
    replace_user_state_snapshot_collection_rows,
    restore_user_state_alias_profile_refs,
)

SNAPSHOT_TYPE = "user_state"
SNAPSHOT_VERSION = 1
DERIVED_EMBEDDING_FIELD = "embedding"
USER_STATE_COLLECTIONS = (
    "user_profiles",
    "user_memory_units",
    "memory",
    "conversation_episode_state",
    "scheduled_events",
    "conversation_history",
)


def _utc_now_iso() -> str:
    """Return the current UTC time in a JSON-friendly string.

    Returns:
        ISO-8601 timestamp with a UTC offset.
    """
    return_value = datetime.now(timezone.utc).isoformat()
    return return_value


def _safe_identifier(value: str) -> str:
    """Convert a user identifier into a filesystem-safe filename component.

    Args:
        value: Raw global user id or platform user id.

    Returns:
        Sanitized identifier suitable for a local snapshot filename.
    """
    clean_value = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in value.strip()
    )
    return_value = clean_value or "unknown"
    return return_value


def default_snapshot_path(global_user_id: str) -> Path:
    """Build the default local snapshot path for one user.

    Args:
        global_user_id: Internal UUID for the user.

    Returns:
        Stable path under ``test_artifacts``.
    """
    safe_id = _safe_identifier(global_user_id)
    return_value = Path("test_artifacts") / f"user_state_snapshot_{safe_id}.json"
    return return_value


def platform_account_pairs(profile: dict[str, Any]) -> list[dict[str, str]]:
    """Extract platform account pairs from a user profile.

    Args:
        profile: ``user_profiles`` document or empty profile placeholder.

    Returns:
        Clean ``platform`` and ``platform_user_id`` pairs for legacy lookups.
    """
    accounts = profile.get("platform_accounts")
    if not isinstance(accounts, list):
        return_value: list[dict[str, str]] = []
        return return_value

    pairs: list[dict[str, str]] = []
    for account in accounts:
        if not isinstance(account, dict):
            continue
        platform = str(account.get("platform", "")).strip()
        platform_user_id = str(account.get("platform_user_id", "")).strip()
        if platform and platform_user_id:
            pairs.append({
                "platform": platform,
                "platform_user_id": platform_user_id,
            })
    return pairs


async def resolve_user_scope(identifier: str, platform: str | None) -> dict[str, Any]:
    """Resolve a global user id and profile from a global or platform id.

    Args:
        identifier: Global user id, or platform user id when ``platform`` is set.
        platform: Optional platform name for exact platform-account lookup.

    Returns:
        Identity scope containing ``global_user_id``, ``profile``, and linked
        platform-account pairs.

    Raises:
        ValueError: If a platform lookup cannot find a matching profile.
    """
    clean_identifier = identifier.strip()
    clean_platform = platform.strip() if platform else None

    if not clean_platform:
        profile = dict(await get_user_profile(clean_identifier))
        if profile:
            global_user_id = str(profile["global_user_id"])
            scope = {
                "global_user_id": global_user_id,
                "profile": profile,
                "platform_accounts": platform_account_pairs(profile),
            }
            return scope

    profile_doc = await find_user_profile_for_export(
        identifier=clean_identifier,
        platform=clean_platform,
        projection={},
    )
    if not profile_doc:
        if clean_platform:
            raise ValueError(f"No user profile found for {clean_platform}:{clean_identifier}")
        profile_doc = {"global_user_id": clean_identifier}

    profile = dict(profile_doc)
    global_user_id = str(profile["global_user_id"])
    scope = {
        "global_user_id": global_user_id,
        "profile": profile,
        "platform_accounts": platform_account_pairs(profile),
    }
    return scope


async def load_alias_profile_refs(
    global_user_id: str,
) -> list[dict[str, Any]]:
    """Load profile alias backlinks that point at the user.

    Args:
        global_user_id: Internal UUID for the user.

    Returns:
        Minimal profile rows needed to restore ``suspected_aliases`` backlinks.
    """
    alias_refs = await load_user_state_alias_profile_refs(global_user_id)
    return alias_refs


def build_snapshot_payload(
    *,
    identity: dict[str, Any],
    documents: dict[str, list[dict[str, Any]]],
    alias_profile_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the on-disk snapshot envelope.

    Args:
        identity: Resolved user identity and request metadata.
        documents: Captured documents grouped by collection.
        alias_profile_refs: Minimal alias backlink rows.

    Returns:
        Snapshot payload ready for Extended JSON serialization.
    """
    counts = {
        collection_name: len(rows)
        for collection_name, rows in documents.items()
    }
    scrubbed_documents = scrub_document(
        documents,
        {DERIVED_EMBEDDING_FIELD},
    )
    if not isinstance(scrubbed_documents, dict):
        raise TypeError("scrubbed snapshot documents must be a dictionary")

    payload = {
        "snapshot_type": SNAPSHOT_TYPE,
        "snapshot_version": SNAPSHOT_VERSION,
        "created_at": _utc_now_iso(),
        "identity": identity,
        "scope": {
            "collections": list(USER_STATE_COLLECTIONS),
            "alias_profile_refs": "user_profiles.suspected_aliases backlinks",
            "excluded": [
                "character_state",
                "rag_cache2_persistent_entries",
                "derived in-memory caches",
            ],
        },
        "counts": {
            **counts,
            "alias_profile_refs": len(alias_profile_refs),
        },
        "documents": scrubbed_documents,
        "alias_profile_refs": alias_profile_refs,
    }
    return payload


def write_snapshot_file(file_path: Path, payload: dict[str, Any]) -> None:
    """Write a user-state snapshot as relaxed Extended JSON.

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
    """Read a user-state snapshot from relaxed Extended JSON.

    Args:
        file_path: Source file path.

    Returns:
        Parsed snapshot envelope.

    Raises:
        ValueError: If the file does not contain a snapshot object.
    """
    payload = json_util.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("snapshot file must decode to a JSON object")
    return payload


def extract_snapshot_identity(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and return snapshot identity metadata.

    Args:
        payload: Parsed snapshot payload.

    Returns:
        Snapshot identity dict.

    Raises:
        ValueError: If the file is not a supported user-state snapshot.
    """
    snapshot_type = payload.get("snapshot_type")
    if snapshot_type != SNAPSHOT_TYPE:
        raise ValueError(f"snapshot_type must be {SNAPSHOT_TYPE!r}, got {snapshot_type!r}")

    version = payload.get("snapshot_version")
    if version != SNAPSHOT_VERSION:
        raise ValueError(f"snapshot_version must be {SNAPSHOT_VERSION}, got {version!r}")

    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("snapshot file must contain an identity object")

    global_user_id = str(identity.get("global_user_id", "")).strip()
    if not global_user_id:
        raise ValueError("snapshot identity must contain global_user_id")

    return identity


def extract_snapshot_documents(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Validate and return captured documents grouped by collection.

    Args:
        payload: Parsed snapshot payload.

    Returns:
        Mapping from collection name to document list.

    Raises:
        ValueError: If the document section is malformed.
    """
    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, dict):
        raise ValueError("snapshot file must contain a documents object")

    documents: dict[str, list[dict[str, Any]]] = {}
    for collection_name in USER_STATE_COLLECTIONS:
        rows = raw_documents.get(collection_name, [])
        if not isinstance(rows, list):
            raise ValueError(f"documents.{collection_name} must be a list")
        documents[collection_name] = [dict(row) for row in rows if isinstance(row, dict)]
    return documents


async def recalculate_embeddings(
    collection_name: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach fresh embeddings for restored rows whose vectors are derived.

    Args:
        collection_name: Collection being restored.
        rows: Snapshot rows with stored embeddings stripped.

    Returns:
        Rows ready for insertion, with regenerated embeddings when applicable.
    """
    restored_rows = await recalculate_user_state_embeddings(
        collection_name=collection_name,
        rows=rows,
    )
    return restored_rows


async def snapshot_user_state(
    *,
    identifier: str,
    platform: str | None,
    file_path: Path | None,
) -> dict[str, Any]:
    """Capture all authoritative state related to one user.

    Args:
        identifier: Global user id, or platform user id when ``platform`` is set.
        platform: Optional platform name for exact platform-account lookup.
        file_path: Optional destination file path.

    Returns:
        Snapshot payload written to disk.
    """
    scope = await resolve_user_scope(identifier, platform)
    global_user_id = str(scope["global_user_id"])
    platform_accounts = list(scope["platform_accounts"])
    documents = await load_user_state_snapshot_documents(
        global_user_id=global_user_id,
        platform_accounts=platform_accounts,
        collection_names=USER_STATE_COLLECTIONS,
    )

    alias_refs = await load_alias_profile_refs(global_user_id)
    identity = {
        "requested_identifier": identifier,
        "requested_platform": platform,
        "global_user_id": global_user_id,
        "platform_accounts": platform_accounts,
    }
    payload = build_snapshot_payload(
        identity=identity,
        documents=documents,
        alias_profile_refs=alias_refs,
    )
    output_path = file_path or default_snapshot_path(global_user_id)
    write_snapshot_file(output_path, payload)
    return payload


async def restore_alias_profile_refs(
    global_user_id: str,
    alias_refs: list[dict[str, Any]],
) -> None:
    """Restore only alias backlinks from other profile documents.

    Args:
        global_user_id: Internal UUID for the restored user.
        alias_refs: Minimal alias-ref rows from the snapshot.

    Returns:
        None.
    """
    await restore_user_state_alias_profile_refs(
        global_user_id=global_user_id,
        alias_refs=alias_refs,
    )


async def restore_user_state(
    *,
    file_path: Path,
    expected_global_user_id: str | None = None,
) -> dict[str, Any]:
    """Restore all authoritative state related to one user from a snapshot.

    Args:
        file_path: Source snapshot file.
        expected_global_user_id: Optional guard from a caller-supplied lookup.

    Returns:
        Summary containing restored global id and per-collection counts.

    Raises:
        ValueError: If the snapshot does not match the expected user.
    """
    payload = read_snapshot_file(file_path)
    identity = extract_snapshot_identity(payload)
    global_user_id = str(identity["global_user_id"])
    if expected_global_user_id and expected_global_user_id != global_user_id:
        raise ValueError(
            f"snapshot is for {global_user_id}, not {expected_global_user_id}"
        )

    platform_accounts = platform_account_pairs(identity)
    documents = extract_snapshot_documents(payload)
    raw_alias_refs = payload.get("alias_profile_refs", [])
    if not isinstance(raw_alias_refs, list):
        raise ValueError("alias_profile_refs must be a list")
    alias_refs = [dict(row) for row in raw_alias_refs if isinstance(row, dict)]

    restored_counts: dict[str, int] = {}
    for collection_name in USER_STATE_COLLECTIONS:
        rows = await recalculate_embeddings(
            collection_name,
            documents[collection_name],
        )
        row_count = await replace_user_state_snapshot_collection_rows(
            global_user_id=global_user_id,
            platform_accounts=platform_accounts,
            collection_name=collection_name,
            rows=rows,
        )
        restored_counts[collection_name] = row_count

    await restore_alias_profile_refs(global_user_id, alias_refs)
    summary = {
        "global_user_id": global_user_id,
        "counts": {
            **restored_counts,
            "alias_profile_refs": len(alias_refs),
        },
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser for user-state snapshot and restore.
    """
    parser = argparse.ArgumentParser(
        description="Snapshot or restore all authoritative state for one user.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser(
        "snapshot",
        help="Write all state related to a user to a local file.",
    )
    snapshot_parser.add_argument(
        "identifier",
        help="Global user id, or platform user id when --platform is set.",
    )
    snapshot_parser.add_argument(
        "--platform",
        help="Platform name for platform-account lookup, for example qq.",
    )
    snapshot_parser.add_argument("--file", type=Path, help="Destination snapshot path.")
    snapshot_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show project database logs.",
    )

    restore_parser = subparsers.add_parser(
        "restore",
        help="Replace current user-scoped rows from a local snapshot file.",
    )
    restore_parser.add_argument(
        "identifier",
        nargs="?",
        help="Optional global user id or platform user id used as a restore guard.",
    )
    restore_parser.add_argument(
        "--platform",
        help="Platform name for platform-account lookup when identifier is a platform id.",
    )
    restore_parser.add_argument("--file", type=Path, help="Source snapshot path.")
    restore_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show project database logs.",
    )

    return parser


async def main() -> None:
    """Run the user-state snapshot CLI.

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
            payload = await snapshot_user_state(
                identifier=args.identifier,
                platform=args.platform,
                file_path=args.file,
            )
            global_user_id = payload["identity"]["global_user_id"]
            output_path = args.file or default_snapshot_path(global_user_id)
            print(f"wrote user state snapshot for {global_user_id} to {output_path}")
            print(f"counts: {payload['counts']}")
            return

        expected_global_user_id = None
        source_path = args.file
        if args.identifier:
            scope = await resolve_user_scope(args.identifier, args.platform)
            expected_global_user_id = str(scope["global_user_id"])
            if source_path is None:
                source_path = default_snapshot_path(expected_global_user_id)
        if source_path is None:
            raise ValueError("restore requires --file or an identifier for the default path")

        summary = await restore_user_state(
            file_path=source_path,
            expected_global_user_id=expected_global_user_id,
        )
        print(f"restored user state for {summary['global_user_id']} from {source_path}")
        print(f"counts: {summary['counts']}")
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
