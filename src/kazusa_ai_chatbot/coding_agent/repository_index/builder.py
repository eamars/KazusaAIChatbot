"""Build immutable SQLite snapshots for safe repository text."""

from __future__ import annotations

import ast
import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.context_budget import (
    MAX_INDEX_CHUNK_BYTES,
    MAX_INDEX_FILE_BYTES,
    MAX_INDEX_STORAGE_BYTES,
)
from kazusa_ai_chatbot.coding_agent.repository_index.identity import (
    EXCLUSION_POLICY_VERSION,
    excluded_path_reason,
    has_secret_marker,
    source_identity_hash,
)


INDEX_SCHEMA_VERSION = "repository_index.v1"


class _SourceChangedDuringIndex(ValueError):
    """Signal that source content changed after manifest construction."""


def build_repository_snapshot(
    *,
    workspace_root: Path,
    source_root: Path,
    source_identity: dict[str, object],
) -> dict[str, object]:
    """Build and atomically publish one immutable safe-text repository snapshot.

    Args:
        workspace_root: Coding workspace that owns persistent index artifacts.
        source_root: Resolved source directory to inspect without mutation.
        source_identity: Stable resolved-source fields used for snapshot identity.

    Returns:
        Complete snapshot metadata plus excluded repository-relative paths.
    """

    source_hash = source_identity_hash(source_identity)
    try:
        included_rows, excluded_paths, exhausted_resource = (
            _collect_safe_text_rows(source_root)
        )
    except OSError:
        return {
            "status": "blocked",
            "blocker_type": "environment",
            "code": "source_changed_during_index",
        }
    if exhausted_resource:
        return {
            "status": "blocked",
            "blocker_type": "index_resource_exhausted",
            "resource": exhausted_resource,
        }
    manifest_digest = _manifest_digest(included_rows)
    snapshot_id = _snapshot_id(source_hash, manifest_digest)
    index_root = workspace_root / "repository_indexes"
    index_directory = index_root / source_hash
    if index_root.is_symlink() or index_directory.is_symlink():
        return {
            "status": "blocked",
            "blocker_type": "environment",
            "code": "repository_index_path_invalid",
        }
    index_directory.mkdir(parents=True, exist_ok=True)
    database_path = index_directory / f"{snapshot_id}.sqlite"
    if database_path.is_symlink():
        return {
            "status": "blocked",
            "blocker_type": "environment",
            "code": "published_repository_snapshot_invalid",
        }
    if database_path.exists():
        if not _published_snapshot_matches(
            database_path=database_path,
            snapshot_id=snapshot_id,
            source_hash=source_hash,
            manifest_digest=manifest_digest,
        ):
            return {
                "status": "blocked",
                "blocker_type": "environment",
                "code": "published_repository_snapshot_invalid",
            }
        snapshot = {
            "snapshot_id": snapshot_id,
            "status": "complete",
            "manifest_digest": manifest_digest,
            "excluded_paths": sorted(excluded_paths),
        }
        return snapshot

    building_path = index_directory / f"{snapshot_id}.building.sqlite"
    if building_path.is_symlink():
        return {
            "status": "blocked",
            "blocker_type": "environment",
            "code": "building_repository_snapshot_invalid",
        }
    resumed_incomplete_build = building_path.exists()
    connection = sqlite3.connect(building_path)
    try:
        _create_schema(connection)
        connection.execute(
            "INSERT OR REPLACE INTO snapshot VALUES (?, ?, ?, ?, ?, ?)",
            (
                snapshot_id,
                source_hash,
                manifest_digest,
                INDEX_SCHEMA_VERSION,
                EXCLUSION_POLICY_VERSION,
                "building",
            ),
        )
        try:
            reused_content_hashes = _insert_rows(
                connection,
                snapshot_id,
                source_root,
                included_rows,
                reusable_database=_latest_complete_database(index_directory),
            )
        except (_SourceChangedDuringIndex, OSError):
            connection.rollback()
            return {
                "status": "blocked",
                "blocker_type": "environment",
                "code": "source_changed_during_index",
            }
        connection.execute(
            "UPDATE snapshot SET status = ? WHERE snapshot_id = ?",
            ("complete", snapshot_id),
        )
        connection.commit()
    finally:
        connection.close()
    building_path.replace(database_path)

    snapshot = {
        "snapshot_id": snapshot_id,
        "status": "complete",
        "manifest_digest": manifest_digest,
        "excluded_paths": sorted(excluded_paths),
        "resumed_incomplete_build": resumed_incomplete_build,
        "reused_content_hashes": sorted(reused_content_hashes),
    }
    return snapshot


def _published_snapshot_matches(
    *,
    database_path: Path,
    snapshot_id: str,
    source_hash: str,
    manifest_digest: str,
) -> bool:
    """Validate the complete immutable identity before reusing publication."""

    try:
        with sqlite3.connect(database_path) as connection:
            row = connection.execute(
                "SELECT snapshot_id, source_identity_hash, manifest_digest, "
                "schema_version, exclusion_policy_version, status "
                "FROM snapshot LIMIT 1",
            ).fetchone()
    except sqlite3.DatabaseError:
        return False
    expected = (
        snapshot_id,
        source_hash,
        manifest_digest,
        INDEX_SCHEMA_VERSION,
        EXCLUSION_POLICY_VERSION,
        "complete",
    )
    return row == expected


def _collect_safe_text_rows(
    source_root: Path,
) -> tuple[list[tuple[str, str, int, int]], list[str], str]:
    """Stream safe text identities without retaining repository content."""

    included_rows: list[tuple[str, str, int, int]] = []
    excluded_paths: list[str] = []
    total_storage_bytes = 0
    for source_path in sorted(source_root.rglob("*")):
        if not source_path.is_file() or source_path.is_symlink():
            continue
        relative_path = source_path.relative_to(source_root)
        exclusion_reason = excluded_path_reason(relative_path)
        if exclusion_reason:
            excluded_paths.append(relative_path.as_posix())
            continue
        if source_path.stat().st_size > MAX_INDEX_FILE_BYTES:
            return included_rows, excluded_paths, "MAX_INDEX_FILE_BYTES"
        scanned = _scan_safe_text_file(source_path)
        if scanned is None:
            excluded_paths.append(relative_path.as_posix())
            continue
        content_hash, normalized_byte_size, line_count, secret_found = scanned
        if secret_found:
            excluded_paths.append(relative_path.as_posix())
            continue
        total_storage_bytes += normalized_byte_size
        if total_storage_bytes > MAX_INDEX_STORAGE_BYTES:
            return included_rows, excluded_paths, "MAX_INDEX_STORAGE_BYTES"
        included_rows.append((
            relative_path.as_posix(),
            content_hash,
            normalized_byte_size,
            line_count,
        ))
    return included_rows, excluded_paths, ""


def _scan_safe_text_file(
    source_path: Path,
) -> tuple[str, int, int, bool] | None:
    """Hash and classify one complete text file through bounded reads."""

    content_hash = hashlib.sha256()
    normalized_byte_size = 0
    newline_count = 0
    has_content = False
    ends_with_newline = False
    secret_found = False
    marker_overlap = ""
    read_chars = max(1, MAX_INDEX_CHUNK_BYTES // 4)
    try:
        with source_path.open("r", encoding="utf-8", newline=None) as file_handle:
            while True:
                content = file_handle.read(read_chars)
                if not content:
                    break
                if "\x00" in content:
                    return None
                encoded = content.encode("utf-8")
                content_hash.update(encoded)
                normalized_byte_size += len(encoded)
                newline_count += content.count("\n")
                has_content = True
                ends_with_newline = content.endswith("\n")
                scan_window = marker_overlap + content
                if has_secret_marker(scan_window):
                    secret_found = True
                marker_overlap = scan_window[-256:]
    except UnicodeDecodeError:
        return None
    line_count = newline_count
    if has_content and not ends_with_newline:
        line_count += 1
    result = (
        content_hash.hexdigest(),
        normalized_byte_size,
        line_count,
        secret_found,
    )
    return result


def _manifest_digest(rows: list[tuple[str, str, int, int]]) -> str:
    """Return the immutable source-manifest digest for indexed text rows."""

    manifest = "\n".join(
        f"{path}:{content_hash}:{byte_size}:{line_count}"
        for path, content_hash, byte_size, line_count in rows
    )
    manifest_digest = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    return manifest_digest


def _latest_complete_database(index_directory: Path) -> Path | None:
    """Return the newest complete immutable snapshot available for row reuse."""

    candidates = sorted(
        (
            path
            for path in index_directory.glob("*.sqlite")
            if not path.name.endswith(".building.sqlite")
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for database_path in candidates:
        try:
            with sqlite3.connect(database_path) as connection:
                status = connection.execute(
                    "SELECT status, schema_version, exclusion_policy_version "
                    "FROM snapshot LIMIT 1"
                ).fetchone()
        except sqlite3.DatabaseError:
            continue
        if status == (
            "complete",
            INDEX_SCHEMA_VERSION,
            EXCLUSION_POLICY_VERSION,
        ):
            return database_path
    return None


def _snapshot_id(source_hash: str, manifest_digest: str) -> str:
    """Derive one immutable snapshot identity from source and policy versions."""

    snapshot_seed = ":".join((
        source_hash,
        manifest_digest,
        EXCLUSION_POLICY_VERSION,
        INDEX_SCHEMA_VERSION,
    ))
    snapshot_id = hashlib.sha256(snapshot_seed.encode("utf-8")).hexdigest()
    return snapshot_id


def _create_schema(connection: sqlite3.Connection) -> None:
    """Create the canonical v1 snapshot tables in a building database."""

    connection.execute(
        "CREATE TABLE IF NOT EXISTS snapshot (snapshot_id TEXT PRIMARY KEY, "
        "source_identity_hash TEXT NOT NULL, manifest_digest TEXT NOT NULL, "
        "schema_version TEXT NOT NULL, exclusion_policy_version TEXT NOT NULL, "
        "status TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS file (snapshot_id TEXT NOT NULL, "
        "file_id TEXT NOT NULL, "
        "repo_path TEXT NOT NULL, content_sha256 TEXT NOT NULL, byte_size INTEGER "
        "NOT NULL, language TEXT, line_count INTEGER NOT NULL, "
        "PRIMARY KEY (snapshot_id, file_id), UNIQUE (snapshot_id, repo_path))"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS chunk (snapshot_id TEXT NOT NULL, "
        "chunk_id TEXT NOT NULL, "
        "file_id TEXT NOT NULL, repo_path TEXT NOT NULL, start_line INTEGER NOT NULL, "
        "end_line INTEGER NOT NULL, content TEXT NOT NULL, "
        "PRIMARY KEY (snapshot_id, chunk_id))"
    )
    connection.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(chunk_id UNINDEXED, "
        "snapshot_id UNINDEXED, repo_path UNINDEXED, content)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS symbol (snapshot_id TEXT NOT NULL, "
        "symbol_id TEXT NOT NULL, "
        "file_id TEXT NOT NULL, qualified_name TEXT NOT NULL, symbol_kind TEXT NOT "
        "NULL, start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, signature TEXT, "
        "PRIMARY KEY (snapshot_id, symbol_id))"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS import_edge (snapshot_id TEXT NOT NULL, "
        "edge_id TEXT NOT NULL, "
        "file_id TEXT NOT NULL, imported_name TEXT NOT NULL, target_repo_path TEXT, "
        "PRIMARY KEY (snapshot_id, edge_id))"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS build_file (repo_path TEXT PRIMARY KEY, "
        "content_sha256 TEXT NOT NULL, status TEXT NOT NULL)"
    )


def _insert_rows(
    connection: sqlite3.Connection,
    snapshot_id: str,
    source_root: Path,
    rows: list[tuple[str, str, int, int]],
    *,
    reusable_database: Path | None,
) -> set[str]:
    """Insert files, chunks, symbols, and imports for one building snapshot."""

    reused_content_hashes: set[str] = set()
    for repo_path, content_hash, byte_size, line_count in rows:
        completed = connection.execute(
            "SELECT content_sha256, status FROM build_file WHERE repo_path = ?",
            (repo_path,),
        ).fetchone()
        if completed == (content_hash, "complete"):
            continue
        file_id = hashlib.sha256(
            f"{snapshot_id}:{repo_path}".encode("utf-8")
        ).hexdigest()
        language = "python" if repo_path.endswith(".py") else None
        with connection:
            _clear_file_rows(
                connection=connection,
                snapshot_id=snapshot_id,
                repo_path=repo_path,
                file_id=file_id,
            )
            connection.execute(
                "INSERT INTO file VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    file_id,
                    repo_path,
                    content_hash,
                    byte_size,
                    language,
                    line_count,
                ),
            )
            reused = _reuse_complete_file_rows(
                connection=connection,
                reusable_database=reusable_database,
                snapshot_id=snapshot_id,
                file_id=file_id,
                repo_path=repo_path,
                content_hash=content_hash,
            )
            if not reused:
                source_path = source_root / repo_path
                indexed_hash = hashlib.sha256()
                indexed_byte_size = 0
                python_content_parts: list[str] = []
                for chunk_number, start_line, end_line, content in (
                    _stream_text_chunks(source_path)
                ):
                    encoded_content = content.encode("utf-8")
                    indexed_hash.update(encoded_content)
                    indexed_byte_size += len(encoded_content)
                    if language == "python":
                        python_content_parts.append(content)
                    chunk_id = hashlib.sha256(
                        f"{file_id}:{chunk_number}".encode("utf-8")
                    ).hexdigest()
                    connection.execute(
                        "INSERT INTO chunk VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            snapshot_id,
                            chunk_id,
                            file_id,
                            repo_path,
                            start_line,
                            end_line,
                            content,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO chunk_fts VALUES (?, ?, ?, ?)",
                        (chunk_id, snapshot_id, repo_path, content),
                    )
                if (
                    indexed_hash.hexdigest() != content_hash
                    or indexed_byte_size != byte_size
                ):
                    raise _SourceChangedDuringIndex(
                        "source changed during repository indexing"
                    )
                if language == "python":
                    _insert_python_metadata(
                        connection,
                        snapshot_id,
                        file_id,
                        repo_path,
                        "".join(python_content_parts),
                    )
            else:
                reused_content_hashes.add(content_hash)
            connection.execute(
                "INSERT OR REPLACE INTO build_file VALUES (?, ?, ?)",
                (repo_path, content_hash, "complete"),
            )
    return reused_content_hashes


def _clear_file_rows(
    *,
    connection: sqlite3.Connection,
    snapshot_id: str,
    repo_path: str,
    file_id: str,
) -> None:
    """Remove one incomplete file transaction before deterministic replay."""

    chunk_ids = connection.execute(
        "SELECT chunk_id FROM chunk WHERE snapshot_id = ? AND file_id = ?",
        (snapshot_id, file_id),
    ).fetchall()
    connection.executemany(
        "DELETE FROM chunk_fts WHERE chunk_id = ?",
        chunk_ids,
    )
    connection.execute(
        "DELETE FROM chunk WHERE snapshot_id = ? AND file_id = ?",
        (snapshot_id, file_id),
    )
    connection.execute(
        "DELETE FROM symbol WHERE snapshot_id = ? AND file_id = ?",
        (snapshot_id, file_id),
    )
    connection.execute(
        "DELETE FROM import_edge WHERE snapshot_id = ? AND file_id = ?",
        (snapshot_id, file_id),
    )
    connection.execute(
        "DELETE FROM file WHERE snapshot_id = ? AND repo_path = ?",
        (snapshot_id, repo_path),
    )


def _reuse_complete_file_rows(
    *,
    connection: sqlite3.Connection,
    reusable_database: Path | None,
    snapshot_id: str,
    file_id: str,
    repo_path: str,
    content_hash: str,
) -> bool:
    """Copy unchanged indexed rows from one complete immutable snapshot."""

    if reusable_database is None:
        return False
    try:
        with sqlite3.connect(reusable_database) as reusable:
            prior_file = reusable.execute(
                "SELECT file_id FROM file WHERE content_sha256 = ? "
                "ORDER BY repo_path, file_id LIMIT 1",
                (content_hash,),
            ).fetchone()
            if prior_file is None:
                return False
            prior_file_id = prior_file[0]
            chunks = reusable.execute(
                "SELECT start_line, end_line, content FROM chunk "
                "WHERE file_id = ? ORDER BY start_line, chunk_id",
                (prior_file_id,),
            ).fetchall()
            symbols = reusable.execute(
                "SELECT qualified_name, symbol_kind, start_line, end_line, "
                "signature FROM symbol WHERE file_id = ? ORDER BY symbol_id",
                (prior_file_id,),
            ).fetchall()
            imports = reusable.execute(
                "SELECT imported_name, target_repo_path FROM import_edge "
                "WHERE file_id = ? ORDER BY edge_id",
                (prior_file_id,),
            ).fetchall()
    except sqlite3.DatabaseError:
        return False
    for chunk_number, row in enumerate(chunks, start=1):
        chunk_id = hashlib.sha256(
            f"{file_id}:{chunk_number}".encode("utf-8")
        ).hexdigest()
        start_line, end_line, content = row
        connection.execute(
            "INSERT INTO chunk VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot_id,
                chunk_id,
                file_id,
                repo_path,
                start_line,
                end_line,
                content,
            ),
        )
        connection.execute(
            "INSERT INTO chunk_fts VALUES (?, ?, ?, ?)",
            (chunk_id, snapshot_id, repo_path, content),
        )
    for symbol_number, row in enumerate(symbols, start=1):
        symbol_id = hashlib.sha256(
            f"{file_id}:symbol:{symbol_number}".encode("utf-8")
        ).hexdigest()
        connection.execute(
            "INSERT INTO symbol VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (snapshot_id, symbol_id, file_id, *row),
        )
    for import_number, row in enumerate(imports, start=1):
        edge_id = hashlib.sha256(
            f"{file_id}:import:{import_number}".encode("utf-8")
        ).hexdigest()
        connection.execute(
            "INSERT INTO import_edge VALUES (?, ?, ?, ?, ?)",
            (snapshot_id, edge_id, file_id, *row),
        )
    return True


def _stream_text_chunks(
    source_path: Path,
) -> Iterator[tuple[int, int, int, str]]:
    """Yield bounded line-aware chunks through bounded source reads."""

    buffer = ""
    chunk_number = 0
    start_line = 1
    read_chars = max(1, MAX_INDEX_CHUNK_BYTES // 4)
    with source_path.open("r", encoding="utf-8", newline=None) as file_handle:
        while True:
            content = file_handle.read(read_chars)
            if not content:
                break
            buffer += content
            while len(buffer.encode("utf-8")) > MAX_INDEX_CHUNK_BYTES:
                boundary = _bounded_chunk_boundary(buffer)
                chunk_content = buffer[:boundary]
                buffer = buffer[boundary:]
                chunk_number += 1
                end_line = start_line + chunk_content.count("\n")
                if chunk_content.endswith("\n"):
                    end_line -= 1
                end_line = max(start_line, end_line)
                yield (
                    chunk_number,
                    start_line,
                    end_line,
                    chunk_content,
                )
                start_line = (
                    end_line + 1
                    if chunk_content.endswith("\n")
                    else end_line
                )
    if buffer:
        chunk_number += 1
        end_line = start_line + buffer.count("\n")
        if buffer.endswith("\n"):
            end_line = max(start_line, end_line - 1)
        yield chunk_number, start_line, end_line, buffer


def _bounded_chunk_boundary(content: str) -> int:
    """Choose the largest byte-bounded prefix, preferring a full line."""

    low = 1
    high = len(content)
    while low < high:
        middle = (low + high + 1) // 2
        if len(content[:middle].encode("utf-8")) <= MAX_INDEX_CHUNK_BYTES:
            low = middle
        else:
            high = middle - 1
    newline_boundary = content.rfind("\n", 0, low)
    if newline_boundary >= 0:
        return newline_boundary + 1
    return low


def _insert_python_metadata(
    connection: sqlite3.Connection,
    snapshot_id: str,
    file_id: str,
    repo_path: str,
    content: str,
) -> None:
    """Extract stable Python symbols and imports without failing the snapshot."""

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbol_kind = "class" if isinstance(node, ast.ClassDef) else "function"
            symbol_id = hashlib.sha256(
                f"{file_id}:{node.name}:{node.lineno}".encode("utf-8")
            ).hexdigest()
            signature = (
                ast.unparse(node.args)
                if not isinstance(node, ast.ClassDef)
                else ""
            )
            connection.execute(
                "INSERT INTO symbol VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    symbol_id,
                    file_id,
                    node.name,
                    symbol_kind,
                    node.lineno,
                    getattr(node, "end_lineno", node.lineno),
                    signature,
                ),
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                _insert_import_edge(
                    connection,
                    snapshot_id,
                    file_id,
                    repo_path,
                    alias.name,
                )
        if isinstance(node, ast.ImportFrom) and node.module:
            _insert_import_edge(
                connection,
                snapshot_id,
                file_id,
                repo_path,
                node.module,
            )


def _insert_import_edge(
    connection: sqlite3.Connection,
    snapshot_id: str,
    file_id: str,
    repo_path: str,
    imported_name: str,
) -> None:
    """Persist one Python import reference with a deterministic identity."""

    edge_id = hashlib.sha256(
        f"{file_id}:{imported_name}".encode("utf-8")
    ).hexdigest()
    target_repo_path = f"{imported_name.replace('.', '/')}.py"
    connection.execute(
        "INSERT INTO import_edge VALUES (?, ?, ?, ?, ?)",
        (snapshot_id, edge_id, file_id, imported_name, target_repo_path),
    )
