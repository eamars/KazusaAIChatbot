"""Deterministic query helpers for immutable repository snapshots."""

from __future__ import annotations

import base64
import binascii
import fnmatch
import json
import multiprocessing
import queue
import re
import sqlite3
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.context_budget import (
    MAX_REGEX_QUERY_CHARS,
    MAX_SEARCH_EXCERPT_CHARS,
    MAX_SEARCH_RESULTS_PER_PAGE,
    REGEX_SEARCH_TIMEOUT_MS,
)
from kazusa_ai_chatbot.coding_agent.repository_index.identity import (
    source_identity_hash,
)
from kazusa_ai_chatbot.coding_agent.repository_index.regex_worker import (
    collect_regex_rows,
)
from kazusa_ai_chatbot.coding_agent.repository_index.overlay import CandidateOverlay


REGEX_SEARCH_TIMEOUT_SECONDS = REGEX_SEARCH_TIMEOUT_MS / 1000
SEARCH_PAGE_SIZE = MAX_SEARCH_RESULTS_PER_PAGE


def search_snapshot(
    *,
    workspace_root: Path,
    snapshot_id: str,
    mode: str,
    query: str,
    cursor: str | None = None,
    path_glob: str | None = None,
    overlay_revision: int = 0,
    source_identity: dict[str, object] | None = None,
    overlay_database_path: Path | None = None,
) -> dict[str, object]:
    """Search one complete snapshot and bind paging to its candidate view.

    Args:
        workspace_root: Coding workspace containing repository index files.
        snapshot_id: Immutable snapshot selected by a coding run.
        mode: Literal, regex, symbol, or path search mode.
        query: Search text normalized by this boundary.
        cursor: Optional opaque cursor from a prior matching search page.
        path_glob: Optional repo-relative glob limiting returned rows.
        overlay_revision: Candidate overlay revision visible to the controller.
        source_identity: Optional source identity used to locate the snapshot.

    Returns:
        Prompt-safe search rows and a cursor bound to the exact search view.
    """

    normalized_query = query.strip()
    normalized_glob = path_glob.strip() if path_glob else ""
    if mode not in {"literal", "regex", "symbol", "path"}:
        return {"status": "rejected", "rows": [], "cursor": None}
    if mode == "regex" and len(normalized_query) > MAX_REGEX_QUERY_CHARS:
        return {"status": "rejected", "rows": [], "cursor": None}
    if mode == "regex":
        try:
            re.compile(normalized_query)
        except re.error:
            return {"status": "rejected", "rows": [], "cursor": None}
    after_key = _cursor_sort_key(
        cursor,
        snapshot_id=snapshot_id,
        mode=mode,
        query=normalized_query,
        path_glob=normalized_glob,
        overlay_revision=overlay_revision,
    )
    if after_key is None:
        return {"status": "stale_cursor", "rows": [], "cursor": None}

    owned_paths: set[str] = set()
    overlay_rows: list[dict[str, object]] = []
    if overlay_database_path is not None:
        overlay = CandidateOverlay(overlay_database_path)
        try:
            owned_paths = overlay.owned_paths()
            try:
                overlay_rows = overlay.search(
                    normalized_query,
                    mode=mode,
                    path_glob=normalized_glob or None,
                )
            except TimeoutError:
                return {
                    "status": "regex_timeout",
                    "rows": [],
                    "cursor": None,
                }
        finally:
            overlay.close()

    database_path = _database_path(
        workspace_root=workspace_root,
        snapshot_id=snapshot_id,
        source_identity=source_identity,
    )
    connection = sqlite3.connect(database_path)
    try:
        if not _snapshot_is_complete(connection, snapshot_id):
            return {"status": "unavailable", "rows": [], "cursor": None}
        raw_rows = _search_rows(
            connection,
            database_path=database_path,
            snapshot_id=snapshot_id,
            mode=mode,
            query=normalized_query,
            after_key=after_key,
            path_glob=normalized_glob,
            excluded_paths=owned_paths,
            limit=SEARCH_PAGE_SIZE + 1,
        )
    finally:
        connection.close()

    if raw_rows is None:
        return {"status": "regex_timeout", "rows": [], "cursor": None}
    candidate_rows = [
        *raw_rows,
        *(
            row
            for row in overlay_rows
            if _row_sort_key(row) > after_key
        ),
    ]
    candidate_rows.sort(key=_row_sort_key)
    page_rows = candidate_rows[:SEARCH_PAGE_SIZE]
    search_rows = [
        _project_row(
            row,
            overlay_revision=overlay_revision,
            query=normalized_query,
        )
        for row in page_rows
    ]
    next_cursor = None
    if page_rows and (cursor is None or len(candidate_rows) > SEARCH_PAGE_SIZE):
        next_cursor = _encode_cursor({
            "snapshot_id": snapshot_id,
            "overlay_revision": overlay_revision,
            "mode": mode,
            "query": normalized_query,
            "path_glob": normalized_glob,
            "last_sort_tuple": list(_row_sort_key(page_rows[-1])),
        })
    result = {"status": "ok", "rows": search_rows, "cursor": next_cursor}
    return result


def read_snapshot(
    *,
    workspace_root: Path,
    snapshot_id: str,
    repo_path: str,
    start_line: int,
    end_line: int | None = None,
    source_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    """Read one bounded line span from a complete immutable snapshot.

    Args:
        workspace_root: Coding workspace containing repository index files.
        snapshot_id: Immutable snapshot selected by a coding run.
        repo_path: Repo-relative path selected through prior evidence.
        start_line: One-based first line of the bounded span.
        end_line: Optional one-based final line of the span.
        source_identity: Optional source identity used to locate the snapshot.

    Returns:
        Prompt-safe text evidence or a typed unavailable/stale observation.
    """

    if start_line < 1:
        return {"status": "rejected", "rows": []}
    requested_end_line = end_line if end_line is not None else start_line + 199
    if requested_end_line < start_line or requested_end_line - start_line >= 500:
        return {"status": "rejected", "rows": []}
    database_path = _database_path(
        workspace_root=workspace_root,
        snapshot_id=snapshot_id,
        source_identity=source_identity,
    )
    connection = sqlite3.connect(database_path)
    try:
        if not _snapshot_is_complete(connection, snapshot_id):
            return {"status": "unavailable", "rows": []}
        rows = connection.execute(
            "SELECT start_line, end_line, content FROM chunk "
            "WHERE snapshot_id = ? AND repo_path = ? AND end_line >= ? "
            "AND start_line <= ? ORDER BY start_line, chunk_id",
            (
                snapshot_id,
                repo_path,
                start_line,
                requested_end_line,
            ),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        return {"status": "stale", "rows": []}
    first_line = int(rows[0][0])
    content = "".join(str(row[2]) for row in rows)
    lines = content.splitlines(keepends=True)
    first_offset = start_line - first_line
    final_offset = requested_end_line - first_line + 1
    selected_lines = lines[first_offset:final_offset]
    if not selected_lines:
        return {"status": "stale", "rows": []}
    content = "".join(selected_lines)
    actual_end_line = start_line + len(selected_lines) - 1
    result = {
        "status": "ok",
        "rows": [{
            "repo_path": repo_path,
            "start_line": start_line,
            "end_line": actual_end_line,
            "content": content,
        }],
    }
    return result


def _cursor_sort_key(
    cursor: str | None,
    *,
    snapshot_id: str,
    mode: str,
    query: str,
    path_glob: str,
    overlay_revision: int,
) -> tuple[str, int, int, str] | None:
    """Return the last sort key only for an exact candidate-view binding."""

    if not cursor:
        return ("", 0, 0, "")
    try:
        cursor_data = _decode_cursor(cursor)
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    expected = {
        "snapshot_id": snapshot_id,
        "overlay_revision": overlay_revision,
        "mode": mode,
        "query": query,
        "path_glob": path_glob,
    }
    if any(cursor_data.get(key) != value for key, value in expected.items()):
        return None
    sort_value = cursor_data.get("last_sort_tuple")
    if (
        not isinstance(sort_value, list)
        or len(sort_value) != 4
        or not isinstance(sort_value[0], str)
        or not isinstance(sort_value[1], int)
        or isinstance(sort_value[1], bool)
        or not isinstance(sort_value[2], int)
        or isinstance(sort_value[2], bool)
        or not isinstance(sort_value[3], str)
    ):
        return None
    sort_key = (
        sort_value[0],
        sort_value[1],
        sort_value[2],
        sort_value[3],
    )
    return sort_key


def _snapshot_is_complete(connection: sqlite3.Connection, snapshot_id: str) -> bool:
    """Return whether a snapshot completed its atomic publication transaction."""

    status_row = connection.execute(
        "SELECT status FROM snapshot WHERE snapshot_id = ?",
        (snapshot_id,),
    ).fetchone()
    return status_row is not None and status_row[0] == "complete"


def _search_rows(
    connection: sqlite3.Connection,
    *,
    database_path: Path,
    snapshot_id: str,
    mode: str,
    query: str,
    after_key: tuple[str, int, int, str],
    path_glob: str,
    excluded_paths: set[str],
    limit: int,
) -> list[dict[str, object]] | None:
    """Read one deterministic page of canonical rows for the requested mode."""

    if mode in {"literal", "symbol", "path"}:
        rows = _paged_database_rows(
            connection=connection,
            snapshot_id=snapshot_id,
            mode=mode,
            query=query,
            after_key=after_key,
            path_glob=path_glob,
            excluded_paths=excluded_paths,
            limit=limit,
        )
        return rows
    if mode == "regex":
        regex_rows = _regex_rows(
            database_path=database_path,
            query=query,
            after_key=after_key,
            path_glob=path_glob,
            excluded_paths=excluded_paths,
            limit=limit,
        )
        if regex_rows is None:
            return None
        rows = _row_dicts(regex_rows, "regex")
        return rows
    return []


def _paged_database_rows(
    *,
    connection: sqlite3.Connection,
    snapshot_id: str,
    mode: str,
    query: str,
    after_key: tuple[str, int, int, str],
    path_glob: str,
    excluded_paths: set[str],
    limit: int,
) -> list[dict[str, object]]:
    """Fill one filtered keyset page without empty-page cursor stalls."""

    collected: list[dict[str, object]] = []
    scan_key = after_key
    batch_size = max(64, limit * 2)
    while len(collected) < limit:
        rows = _database_row_batch(
            connection=connection,
            snapshot_id=snapshot_id,
            mode=mode,
            query=query,
            after_key=scan_key,
            batch_size=batch_size,
        )
        if not rows:
            break
        projected_rows = _row_dicts(rows, mode)
        for row in projected_rows:
            scan_key = _row_sort_key(row)
            repo_path = str(row["repo_path"])
            if repo_path in excluded_paths:
                continue
            if path_glob and not fnmatch.fnmatchcase(repo_path, path_glob):
                continue
            collected.append(row)
            if len(collected) >= limit:
                break
        if len(rows) < batch_size:
            break
    return collected


def _database_row_batch(
    *,
    connection: sqlite3.Connection,
    snapshot_id: str,
    mode: str,
    query: str,
    after_key: tuple[str, int, int, str],
    batch_size: int,
) -> list[tuple[object, ...]]:
    """Read one canonical keyset batch for a non-regex search mode."""

    if mode == "literal":
        rows = connection.execute(
            "SELECT chunk.repo_path, chunk.start_line, chunk.end_line, "
            "chunk.content, file.content_sha256, '', chunk.chunk_id "
            "FROM chunk_fts "
            "JOIN chunk ON chunk.chunk_id = chunk_fts.chunk_id "
            "JOIN file ON file.snapshot_id = chunk.snapshot_id "
            "AND file.file_id = chunk.file_id "
            "WHERE chunk_fts MATCH ? AND chunk.snapshot_id = ? "
            "AND (chunk.repo_path, chunk.start_line, chunk.end_line, "
            "chunk.chunk_id) > (?, ?, ?, ?) "
            "ORDER BY chunk.repo_path, chunk.start_line, chunk.end_line, "
            "chunk.chunk_id LIMIT ?",
            (
                _literal_fts_query(query),
                snapshot_id,
                *after_key,
                batch_size,
            ),
        ).fetchall()
        return rows
    if mode == "symbol":
        rows = connection.execute(
            "SELECT file.repo_path, symbol.start_line, symbol.end_line, '', "
            "file.content_sha256, symbol.qualified_name, symbol.symbol_id "
            "FROM symbol JOIN file ON file.snapshot_id = symbol.snapshot_id "
            "AND file.file_id = symbol.file_id "
            "WHERE symbol.snapshot_id = ? AND symbol.qualified_name LIKE ? "
            "AND (file.repo_path, symbol.start_line, symbol.end_line, "
            "symbol.symbol_id) > (?, ?, ?, ?) "
            "ORDER BY file.repo_path, symbol.start_line, symbol.end_line, "
            "symbol.symbol_id LIMIT ?",
            (snapshot_id, f"%{query}%", *after_key, batch_size),
        ).fetchall()
        return rows
    rows = connection.execute(
        "SELECT repo_path, 1, line_count, '', content_sha256, '', file_id "
        "FROM file WHERE snapshot_id = ? AND repo_path LIKE ? "
        "AND (repo_path, 1, line_count, file_id) > (?, ?, ?, ?) "
        "ORDER BY repo_path, line_count, file_id LIMIT ?",
        (snapshot_id, f"%{query}%", *after_key, batch_size),
    ).fetchall()
    return rows


def _row_dicts(
    rows: list[tuple[object, ...]],
    match_kind: str,
) -> list[dict[str, object]]:
    """Convert ordered SQLite tuples into one internal search-row shape."""

    result_rows = [
        {
            "repo_path": str(repo_path),
            "start_line": int(start_line),
            "end_line": int(end_line),
            "content": str(content),
            "content_sha256": str(content_sha256),
            "symbol": str(symbol),
            "match_kind": match_kind,
            "_sort_id": str(sort_id),
        }
        for (
            repo_path,
            start_line,
            end_line,
            content,
            content_sha256,
            symbol,
            sort_id,
        ) in rows
    ]
    return result_rows


def _project_row(
    row: dict[str, object],
    *,
    overlay_revision: int,
    query: str,
) -> dict[str, object]:
    """Render one prompt-safe canonical result row for controller context."""

    content = str(row.get("content", row.get("excerpt", "")))
    if row["match_kind"] in {"literal", "regex"} and content:
        match = re.search(query, content) if row["match_kind"] == "regex" else None
        match_start = match.start() if match is not None else content.casefold().find(
            query.casefold()
        )
        if match_start >= 0:
            content = _match_excerpt(content, match_start)
    result = {
        "repo_path": row["repo_path"],
        "start_line": row["start_line"],
        "end_line": row["end_line"],
        "match_kind": row["match_kind"],
        "symbol": row["symbol"],
        "excerpt": content[:MAX_SEARCH_EXCERPT_CHARS],
        "content_sha256": row["content_sha256"],
        "candidate_revision": overlay_revision,
    }
    return result


def _row_sort_key(row: dict[str, object]) -> tuple[str, int, int, str]:
    """Return the opaque deterministic position for one candidate-view row."""

    return (
        str(row["repo_path"]),
        int(row.get("start_line", 1)),
        int(row.get("end_line", 1)),
        str(row.get("_sort_id", "")),
    )


def _match_excerpt(content: str, match_start: int) -> str:
    """Return a bounded excerpt centered on one base-snapshot match."""

    half_window = MAX_SEARCH_EXCERPT_CHARS // 2
    excerpt_start = max(0, match_start - half_window)
    excerpt_end = min(len(content), excerpt_start + MAX_SEARCH_EXCERPT_CHARS)
    return content[excerpt_start:excerpt_end]


def _regex_rows(
    *,
    database_path: Path,
    query: str,
    after_key: tuple[str, int, int, str],
    path_glob: str,
    excluded_paths: set[str],
    limit: int,
) -> list[tuple[object, ...]] | None:
    """Run one regex worker and reject all rows if it exceeds the limit."""

    result_queue: multiprocessing.Queue[object] = multiprocessing.Queue()
    worker = multiprocessing.Process(
        target=collect_regex_rows,
        args=(
            str(database_path),
            query,
            after_key,
            path_glob,
            excluded_paths,
            limit,
            result_queue,
        ),
    )
    worker.start()
    worker.join(REGEX_SEARCH_TIMEOUT_SECONDS)
    if worker.is_alive():
        worker.terminate()
        worker.join()
        return None
    if worker.exitcode != 0:
        return None
    try:
        result = result_queue.get(timeout=0.1)
    except queue.Empty:
        return None
    if not isinstance(result, list):
        return None
    rows = [row for row in result if isinstance(row, tuple) and len(row) == 7]
    return rows


def _database_path(
    *,
    workspace_root: Path,
    snapshot_id: str,
    source_identity: dict[str, object] | None,
) -> Path:
    """Locate the unique database that contains the pinned snapshot."""

    resolved_workspace = workspace_root.resolve(strict=False)
    index_root = resolved_workspace / "repository_indexes"
    if index_root.is_symlink():
        raise FileNotFoundError("repository index path is unsafe")
    if source_identity is not None:
        source_hash = source_identity_hash(source_identity)
        source_index_root = index_root / source_hash
        database_path = source_index_root / f"{snapshot_id}.sqlite"
        if source_index_root.is_symlink() or database_path.is_symlink():
            raise FileNotFoundError("repository snapshot path is unsafe")
        if not database_path.resolve(strict=False).is_relative_to(
            resolved_workspace,
        ):
            raise FileNotFoundError("repository snapshot path is unsafe")
        return database_path
    matches = [
        path
        for path in resolved_workspace.glob(
            f"repository_indexes/*/{snapshot_id}.sqlite",
        )
        if not path.is_symlink()
        and path.resolve(strict=False).is_relative_to(resolved_workspace)
    ]
    if len(matches) != 1:
        raise FileNotFoundError("repository snapshot was not uniquely found")
    return matches[0]


def _decode_cursor(cursor: str) -> dict[str, object]:
    """Decode one opaque deterministic search cursor."""

    padded_cursor = cursor + "=" * (-len(cursor) % 4)
    decoded = base64.urlsafe_b64decode(padded_cursor.encode("ascii"))
    decoded_cursor = json.loads(decoded.decode("utf-8"))
    if not isinstance(decoded_cursor, dict):
        raise ValueError("repository search cursor is invalid")
    return decoded_cursor


def _encode_cursor(cursor: dict[str, object]) -> str:
    """Encode one deterministic page binding without host-specific data."""

    encoded = json.dumps(cursor, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    cursor_text = base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")
    return cursor_text


def _literal_fts_query(query: str) -> str:
    """Return one FTS5 phrase query for literal user-provided text."""

    escaped_query = query.replace('"', '""')
    return f'"{escaped_query}"'
