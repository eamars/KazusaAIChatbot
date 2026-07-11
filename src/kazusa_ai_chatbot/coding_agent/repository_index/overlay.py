"""Run-local candidate overlay storage."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import multiprocessing
import queue
import re
import sqlite3
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.context_budget import (
    MAX_SEARCH_EXCERPT_CHARS,
    MAX_SEARCH_RESULTS_PER_PAGE,
    REGEX_SEARCH_TIMEOUT_MS,
)
from kazusa_ai_chatbot.coding_agent.safety import confined_managed_repo_path


REGEX_SEARCH_TIMEOUT_SECONDS = REGEX_SEARCH_TIMEOUT_MS / 1000


class CandidateOverlay:
    """Persist candidate-only paths and tombstones for one coding run."""

    def __init__(self, database_path: Path) -> None:
        """Open the overlay database and create its deterministic schema."""

        if database_path.is_symlink():
            raise ValueError("candidate overlay database path is unsafe")
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._database_path = database_path
        self._connection = sqlite3.connect(database_path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS overlay (repo_path TEXT PRIMARY KEY, "
            "content TEXT, tombstoned INTEGER NOT NULL, revision INTEGER NOT NULL)"
        )
        self._connection.commit()

    def rename(
        self,
        *,
        source_path: str,
        target_path: str,
        content: str,
        revision: int,
    ) -> None:
        """Tombstone a source path and expose its content at a new path."""

        self._connection.execute(
            "INSERT OR REPLACE INTO overlay VALUES (?, ?, ?, ?)",
            (source_path, None, 1, revision),
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO overlay VALUES (?, ?, ?, ?)",
            (target_path, content, 0, revision),
        )
        self._connection.commit()

    def upsert(
        self,
        *,
        repo_path: str,
        content: str,
        revision: int,
    ) -> None:
        """Expose created or modified candidate text at one revision."""

        self._connection.execute(
            "INSERT OR REPLACE INTO overlay VALUES (?, ?, ?, ?)",
            (repo_path, content, 0, revision),
        )
        self._connection.commit()

    def delete(self, *, repo_path: str, revision: int) -> None:
        """Hide one candidate path through a durable tombstone."""

        self._connection.execute(
            "INSERT OR REPLACE INTO overlay VALUES (?, ?, ?, ?)",
            (repo_path, None, 1, revision),
        )
        self._connection.commit()

    def close(self) -> None:
        """Close the run-local overlay database after one action."""

        self._connection.close()

    def describe_paths(self, repo_paths: list[str]) -> list[dict[str, object]]:
        """Describe current overlay state without returning stored content.

        Args:
            repo_paths: Normalized repository-relative paths to inspect in
                caller-provided order.

        Returns:
            One bounded identity record for every requested path.
        """

        records: list[dict[str, object]] = []
        for repo_path in repo_paths:
            row = self._connection.execute(
                "SELECT content, tombstoned, revision FROM overlay "
                "WHERE repo_path = ?",
                (repo_path,),
            ).fetchone()
            if row is None:
                records.append({
                    "repo_path": repo_path,
                    "state": "absent",
                    "content_sha256": None,
                    "revision": None,
                })
                continue
            content, tombstoned, revision = row
            if tombstoned:
                records.append({
                    "repo_path": repo_path,
                    "state": "tombstone",
                    "content_sha256": None,
                    "revision": revision,
                })
                continue
            if not isinstance(content, str):
                raise ValueError("overlay content row is invalid")
            content_sha256 = hashlib.sha256(
                content.encode("utf-8"),
            ).hexdigest()
            records.append({
                "repo_path": repo_path,
                "state": "content",
                "content_sha256": content_sha256,
                "revision": revision,
            })
        return records

    def restore_prior_state(
        self,
        *,
        records: list[dict[str, object]],
        candidate_source_root: Path,
    ) -> None:
        """Restore affected overlay rows in one transaction.

        Content rows are rebuilt from the already-restored candidate rather
        than retained as duplicate raw journal content.

        Args:
            records: Validated prior overlay identity records.
            candidate_source_root: Managed candidate source containing restored
                content for live rows.
        """

        restored_rows: list[tuple[str, str, str | None, int | None]] = []
        for record in records:
            repo_path = record["repo_path"]
            state = record["state"]
            revision = record["revision"]
            if not isinstance(repo_path, str) or state not in {
                "absent",
                "content",
                "tombstone",
            }:
                raise ValueError("prior overlay recovery record is invalid")
            if revision is not None and not isinstance(revision, int):
                raise ValueError("prior overlay revision is invalid")
            if state == "content":
                content_path = confined_managed_repo_path(
                    candidate_source_root,
                    repo_path,
                )
                try:
                    content = content_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as exc:
                    raise ValueError(
                        f"prior overlay content cannot be restored: {exc}",
                    ) from exc
                expected_sha256 = record["content_sha256"]
                actual_sha256 = hashlib.sha256(
                    content.encode("utf-8"),
                ).hexdigest()
                if actual_sha256 != expected_sha256:
                    raise ValueError("prior overlay content identity mismatch")
                restored_rows.append((repo_path, state, content, revision))
                continue
            if record["content_sha256"] is not None:
                raise ValueError("prior overlay non-content identity is invalid")
            restored_rows.append((repo_path, state, None, revision))

        self._connection.execute("BEGIN")
        try:
            for repo_path, state, content, revision in restored_rows:
                if state == "absent":
                    self._connection.execute(
                        "DELETE FROM overlay WHERE repo_path = ?",
                        (repo_path,),
                    )
                else:
                    self._connection.execute(
                        "INSERT OR REPLACE INTO overlay VALUES (?, ?, ?, ?)",
                        (repo_path, content, int(state == "tombstone"), revision),
                    )
        except sqlite3.Error:
            self._connection.rollback()
            raise
        self._connection.commit()

    def search(
        self,
        query: str,
        *,
        mode: str = "literal",
        path_glob: str | None = None,
    ) -> list[dict[str, object]]:
        """Return deterministic live rows for every supported search mode."""

        rows = self._connection.execute(
            "SELECT repo_path, content, revision FROM overlay "
            "WHERE tombstoned = 0 ORDER BY repo_path",
        ).fetchall()
        regex_matches: dict[tuple[str, int, str], int] = {}
        if mode == "regex":
            regex_matches = _bounded_overlay_regex_matches(
                database_path=self._database_path,
                query=query,
                path_glob=path_glob,
            )
        results: list[dict[str, object]] = []
        for repo_path, content, revision in rows:
            if not isinstance(repo_path, str) or not isinstance(content, str):
                raise ValueError("overlay content row is invalid")
            if path_glob and not fnmatch.fnmatchcase(repo_path, path_glob):
                continue
            regex_match_start = None
            if mode == "regex":
                content_sha256 = hashlib.sha256(
                    content.encode("utf-8"),
                ).hexdigest()
                regex_match_start = regex_matches.get(
                    (repo_path, revision, content_sha256),
                )
            results.extend(
                _matching_overlay_rows(
                    repo_path=repo_path,
                    content=content,
                    revision=revision,
                    mode=mode,
                    query=query,
                    regex_match_start=regex_match_start,
                ),
            )
        return results[:MAX_SEARCH_RESULTS_PER_PAGE + 1]

    def is_tombstoned(self, repo_path: str) -> bool:
        """Return whether an overlay row hides its base snapshot path."""

        row = self._connection.execute(
            "SELECT tombstoned FROM overlay WHERE repo_path = ?",
            (repo_path,),
        ).fetchone()
        tombstoned = bool(row and row[0])
        return tombstoned

    def owned_paths(self) -> set[str]:
        """Return paths whose current candidate state supersedes the base."""

        rows = self._connection.execute(
            "SELECT repo_path FROM overlay ORDER BY repo_path"
        ).fetchall()
        owned = {str(row[0]) for row in rows}
        return owned


def _matching_overlay_rows(
    *,
    repo_path: str,
    content: str,
    revision: int,
    mode: str,
    query: str,
    regex_match_start: int | None = None,
) -> list[dict[str, object]]:
    """Project one live overlay file through a closed search mode."""

    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if mode == "path":
        if query.casefold() not in repo_path.casefold():
            return []
        return [_overlay_result(
            repo_path=repo_path,
            content="",
            content_sha256=content_sha256,
            revision=revision,
            match_kind="path",
            start_line=1,
            end_line=_line_count(content),
        )]
    if mode == "symbol":
        results = _symbol_overlay_results(
            repo_path=repo_path,
            content=content,
            content_sha256=content_sha256,
            revision=revision,
            query=query,
        )
        return results
    if mode == "regex":
        match_start = regex_match_start
    else:
        match_index = content.casefold().find(query.casefold())
        match_start = match_index if match_index >= 0 else None
    if match_start is None:
        return []
    start_line = content.count("\n", 0, match_start) + 1
    excerpt, excerpt_start = _match_excerpt(content, match_start)
    excerpt_start_line = content.count("\n", 0, excerpt_start) + 1
    excerpt_end_line = excerpt_start_line + excerpt.count("\n")
    return [_overlay_result(
        repo_path=repo_path,
        content=excerpt,
        content_sha256=content_sha256,
        revision=revision,
        match_kind=mode,
        start_line=max(start_line, excerpt_start_line),
        end_line=max(start_line, excerpt_end_line),
    )]


def _bounded_overlay_regex_matches(
    *,
    database_path: Path,
    query: str,
    path_glob: str | None,
) -> dict[tuple[str, int, str], int]:
    """Search candidate overlay text outside the supervisor process."""

    result_queue: multiprocessing.Queue[object] = multiprocessing.Queue()
    worker = multiprocessing.Process(
        target=_collect_overlay_regex_matches,
        args=(str(database_path), query, path_glob or "", result_queue),
    )
    worker.start()
    worker.join(REGEX_SEARCH_TIMEOUT_SECONDS)
    if worker.is_alive():
        worker.terminate()
        worker.join()
        raise TimeoutError("candidate overlay regex search timed out")
    if worker.exitcode != 0:
        raise TimeoutError("candidate overlay regex worker failed")
    try:
        result = result_queue.get(timeout=0.1)
    except queue.Empty as exc:
        raise TimeoutError(
            "candidate overlay regex worker returned no result",
        ) from exc
    if result is None:
        return {}
    if not isinstance(result, list):
        raise TimeoutError("candidate overlay regex worker result is invalid")
    matches: dict[tuple[str, int, str], int] = {}
    for row in result:
        if (
            not isinstance(row, tuple)
            or len(row) != 4
            or not isinstance(row[0], str)
            or not isinstance(row[1], int)
            or not isinstance(row[2], str)
            or not isinstance(row[3], int)
        ):
            raise TimeoutError("candidate overlay regex row is invalid")
        matches[(row[0], row[1], row[2])] = row[3]
    return matches


def _collect_overlay_regex_matches(
    database_path: str,
    query: str,
    path_glob: str,
    result_queue: multiprocessing.Queue[object],
) -> None:
    """Collect exact candidate regex matches for the bounded parent call."""

    try:
        pattern = re.compile(query)
    except re.error:
        result_queue.put(None)
        return
    connection = sqlite3.connect(Path(database_path))
    matches: list[tuple[str, int, str, int]] = []
    try:
        rows = connection.execute(
            "SELECT repo_path, content, revision FROM overlay "
            "WHERE tombstoned = 0 ORDER BY repo_path",
        )
        for repo_path, content, revision in rows:
            if not isinstance(repo_path, str) or not isinstance(content, str):
                continue
            if path_glob and not fnmatch.fnmatchcase(repo_path, path_glob):
                continue
            match = pattern.search(content)
            if match is None:
                continue
            content_sha256 = hashlib.sha256(
                content.encode("utf-8"),
            ).hexdigest()
            matches.append(
                (repo_path, int(revision), content_sha256, match.start()),
            )
            if len(matches) > MAX_SEARCH_RESULTS_PER_PAGE:
                break
    finally:
        connection.close()
    result_queue.put(matches)


def _line_count(content: str) -> int:
    """Return the number of logical lines in one overlay file."""

    if not content:
        return 1
    count = content.count("\n")
    if not content.endswith("\n"):
        count += 1
    return max(1, count)


def _symbol_overlay_results(
    *,
    repo_path: str,
    content: str,
    content_sha256: str,
    revision: int,
    query: str,
) -> list[dict[str, object]]:
    """Extract matching Python symbols from one current candidate file."""

    if not repo_path.endswith(".py"):
        return []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    results: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        if query.casefold() not in node.name.casefold():
            continue
        result = _overlay_result(
            repo_path=repo_path,
            content="",
            content_sha256=content_sha256,
            revision=revision,
            match_kind="symbol",
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
        )
        result["symbol"] = node.name
        results.append(result)
    return results


def _overlay_result(
    *,
    repo_path: str,
    content: str,
    content_sha256: str,
    revision: int,
    match_kind: str,
    start_line: int,
    end_line: int,
) -> dict[str, object]:
    """Build one canonical prompt-safe overlay search row."""

    result = {
        "repo_path": repo_path,
        "start_line": start_line,
        "end_line": end_line,
        "match_kind": match_kind,
        "symbol": "",
        "excerpt": content[:MAX_SEARCH_EXCERPT_CHARS],
        "content_sha256": content_sha256,
        "candidate_revision": revision,
        "_sort_id": hashlib.sha256(
            (
                f"overlay\0{repo_path}\0{start_line}\0{end_line}\0"
                f"{match_kind}"
            ).encode("utf-8"),
        ).hexdigest(),
    }
    return result


def _match_excerpt(content: str, match_start: int) -> tuple[str, int]:
    """Return a bounded excerpt centered on the overlay match."""

    half_window = MAX_SEARCH_EXCERPT_CHARS // 2
    excerpt_start = max(0, match_start - half_window)
    excerpt_end = min(len(content), excerpt_start + MAX_SEARCH_EXCERPT_CHARS)
    return content[excerpt_start:excerpt_end], excerpt_start
