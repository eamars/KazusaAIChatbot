"""Ordered kernel locks for workspace-local coding-run mutations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import BinaryIO

if os.name == "nt":
    import msvcrt
else:
    import fcntl

LOCK_DIRECTORY_PARTS = (".locks", "coding_agent")
LOCK_TIMEOUT_SECONDS = 5.0
LOCK_RETRY_SECONDS = 0.05


def build_lock_keys(
    *,
    run_id: str,
    source_identity: Mapping[str, object] | None,
) -> list[str]:
    """Build the ordered identity set required for one coding mutation.

    Args:
        run_id: Durable coding-run id receiving the mutation.
        source_identity: Immutable normalized source identity when present.

    Returns:
        The sorted run and optional source lock keys.
    """

    keys = [f"run:{run_id}"]
    if source_identity is not None:
        source_payload = {
            key: value
            for key, value in source_identity.items()
            if key != "workspace_root"
        }
        serialized = json.dumps(source_payload, sort_keys=True, separators=(",", ":"))
        source_digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        keys.append(f"source:{source_digest}")
    ordered_keys = sorted(keys)
    return ordered_keys


@asynccontextmanager
async def acquire_workspace_locks(
    *,
    workspace_root: Path,
    keys: list[str],
    timeout_seconds: float = LOCK_TIMEOUT_SECONDS,
) -> AsyncIterator[bool]:
    """Acquire ordered nonblocking file locks within one workspace.

    Args:
        workspace_root: Coding workspace containing the lock-root directory.
        keys: Sorted stable lock identities required by the mutation.
        timeout_seconds: Maximum bounded wait for all requested locks.

    Yields:
        ``True`` while every lock is held; ``False`` when no lock was acquired.
    """

    lock_root = workspace_root.joinpath(*LOCK_DIRECTORY_PARTS)
    lock_root.mkdir(parents=True, exist_ok=True)
    handles: list[BinaryIO] = []
    deadline = time.monotonic() + timeout_seconds
    while True:
        handles = _open_and_try_lock(lock_root=lock_root, keys=keys)
        if len(handles) == len(keys):
            try:
                yield True
            finally:
                _release_handles(handles)
            return
        _release_handles(handles)
        if time.monotonic() >= deadline:
            yield False
            return
        await asyncio.sleep(LOCK_RETRY_SECONDS)


def _open_and_try_lock(*, lock_root: Path, keys: list[str]) -> list[BinaryIO]:
    """Open every ordered lock file until the first nonblocking contention."""

    handles: list[BinaryIO] = []
    for key in keys:
        lock_name = hashlib.sha256(key.encode("utf-8")).hexdigest()
        lock_path = lock_root / f"{lock_name}.lock"
        handle = lock_path.open("a+b")
        try:
            _lock_handle(handle)
        except OSError:
            handle.close()
            break
        handles.append(handle)
    return handles


def _lock_handle(handle: BinaryIO) -> None:
    """Acquire one nonblocking one-byte advisory lock for the current process."""

    handle.seek(0)
    handle.write(b"\0")
    handle.flush()
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_handles(handles: list[BinaryIO]) -> None:
    """Release acquired kernel locks in reverse acquisition order."""

    for handle in reversed(handles):
        _unlock_handle(handle)
        handle.close()


def _unlock_handle(handle: BinaryIO) -> None:
    """Release one kernel lock while retaining its lock file for reuse."""

    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
