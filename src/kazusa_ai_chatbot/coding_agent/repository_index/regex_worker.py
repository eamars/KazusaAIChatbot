"""Bounded standard-library worker for repository regex search."""

from __future__ import annotations

import fnmatch
import re
import sqlite3
from multiprocessing.queues import Queue
from pathlib import Path


def collect_regex_rows(
    database_path: str,
    query: str,
    after_key: tuple[str, int, int, str],
    path_glob: str,
    excluded_paths: set[str],
    limit: int,
    result_queue: Queue[object],
) -> None:
    """Search deterministic chunk order and return only complete result rows."""

    pattern = re.compile(query)
    connection = sqlite3.connect(Path(database_path))
    matching_rows: list[tuple[object, ...]] = []
    try:
        rows = connection.execute(
            "SELECT chunk.repo_path, chunk.start_line, chunk.end_line, "
            "chunk.content, file.content_sha256, '', chunk.chunk_id "
            "FROM chunk JOIN file ON file.snapshot_id = chunk.snapshot_id "
            "AND file.file_id = chunk.file_id "
            "ORDER BY chunk.repo_path, chunk.start_line, chunk.end_line, "
            "chunk.chunk_id",
        )
        for row in rows:
            repo_path, start_line, end_line, content, _, _, chunk_id = row
            sort_key = (repo_path, start_line, end_line, chunk_id)
            if sort_key <= after_key:
                continue
            if repo_path in excluded_paths:
                continue
            if path_glob and not fnmatch.fnmatchcase(repo_path, path_glob):
                continue
            if not pattern.search(content):
                continue
            matching_rows.append(row)
            if len(matching_rows) >= limit:
                break
    finally:
        connection.close()
    result_queue.put(matching_rows)
