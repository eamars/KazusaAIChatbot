"""Bounded filesystem evidence collection for code reading."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kazusa_ai_chatbot.coding_agent.code_fetching.models import CodeSourceScope
from kazusa_ai_chatbot.coding_agent.code_reading.models import (
    CodeEvidenceRow,
    ProgrammerAssignment,
)
from kazusa_ai_chatbot.coding_agent.code_reading.planner import (
    is_binary_like_path,
    is_secret_like_path,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import (
    PathSafetyError,
    ensure_path_inside,
)

MAX_FILE_BYTES = 128_000
MAX_SEARCH_ROWS_PER_VALUE = 24


class EvidenceCollectionError(ValueError):
    """Raised when scoped evidence cannot be safely collected."""


@dataclass(frozen=True)
class EvidenceBundle:
    """Collected evidence plus file and limitation metadata."""

    rows: list[CodeEvidenceRow]
    files_read: list[str]
    limitations: list[str]
    trace_summary: list[str]


def collect_assignment_evidence(
    *,
    repo_root: Path,
    source_scope: CodeSourceScope,
    assignment: ProgrammerAssignment,
    max_files: int,
    max_excerpt_chars: int,
) -> EvidenceBundle:
    """Collect bounded source rows for one programmer assignment.

    Args:
        repo_root: Resolved checkout root from Phase 0.
        source_scope: Phase 0 source scope that bounds all reads.
        assignment: PM-selected file, directory, symbol, or search mission.
        max_files: Supervisor-owned maximum files one programmer may inspect.
        max_excerpt_chars: Supervisor-owned total excerpt character cap.

    Returns:
        Evidence rows, repo-relative files read, limitations, and safe trace
        notes for the programmer report.
    """

    root = repo_root.resolve(strict=True)
    scoped_files = _scoped_safe_files(root, source_scope)
    scope_kind = assignment["scope"]["kind"]
    values = _bounded_scope_values(assignment)
    trace_summary = [f"programmer_scope:{scope_kind}"]

    if scope_kind in ("file", "directory"):
        candidate_files = _files_for_path_values(
            root=root,
            scoped_files=scoped_files,
            values=values,
        )
        rows = _summary_rows(
            root=root,
            relative_paths=candidate_files[:max_files],
            topic=assignment["role"],
        )
    else:
        rows = _search_rows(
            root=root,
            scoped_files=scoped_files,
            values=values,
            symbol_mode=scope_kind == "symbol",
        )

    capped_rows = _cap_rows(rows, max_files, max_excerpt_chars)
    files_read = _files_from_rows(capped_rows, max_files)
    limitations = _limitations_for_rows(capped_rows, values)
    bundle = EvidenceBundle(
        rows=capped_rows,
        files_read=files_read,
        limitations=limitations,
        trace_summary=trace_summary,
    )
    return bundle


def find_definition_paths(
    *,
    repo_root: Path,
    source_scope: CodeSourceScope,
    symbol: str,
) -> list[str]:
    """Find repository-relative paths defining a class or function symbol."""

    root = repo_root.resolve(strict=True)
    scoped_files = _scoped_safe_files(root, source_scope)
    definition_terms = _definition_terms(symbol)
    paths: list[str] = []
    for relative_path in scoped_files:
        text = _read_text_file(root / relative_path)
        if text is None:
            continue
        for line in text.splitlines():
            if line.strip().startswith(definition_terms):
                paths.append(_to_posix(relative_path))
                break
    result = sorted(set(paths))
    return result


def list_scoped_safe_files(
    *,
    repo_root: Path,
    source_scope: CodeSourceScope,
) -> list[str]:
    """Return safe repo-relative files visible inside the source scope."""

    root = repo_root.resolve(strict=True)
    scoped_files = _scoped_safe_files(root, source_scope)
    safe_files = [_to_posix(path) for path in scoped_files]
    return safe_files


def _scoped_safe_files(root: Path, source_scope: CodeSourceScope) -> list[Path]:
    all_files = _rg_files(root)
    scoped_path = source_scope.get("repo_relative_path")
    if scoped_path is None:
        candidates = all_files
    else:
        scope_root = _safe_scope_root(root, scoped_path)
        if source_scope["kind"] == "file":
            relative_path = scope_root.relative_to(root)
            candidates = [relative_path]
        else:
            candidates = [
                item for item in all_files
                if ensure_path_inside(root / item, root).is_relative_to(scope_root)
            ]

    safe_files = [
        item for item in candidates
        if _is_safe_relative_file(item)
    ]
    return safe_files


def _safe_scope_root(root: Path, repo_relative_path: str) -> Path:
    scope_path = PurePosixPath(repo_relative_path.replace("\\", "/"))
    if scope_path.is_absolute() or ".." in scope_path.parts:
        raise EvidenceCollectionError("Source scope escapes the repository.")
    try:
        safe_root = ensure_path_inside(root / Path(*scope_path.parts), root)
    except PathSafetyError as exc:
        raise EvidenceCollectionError(
            f"Source scope escapes the repository: {exc}"
        ) from exc
    return safe_root


def _rg_files(root: Path) -> list[Path]:
    try:
        completed = subprocess.run(
            [
                "rg",
                "--files",
                "--hidden",
                "-g",
                "!.git/*",
                "-g",
                "!.tmp_pytest/**",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        files = _walk_files(root)
        return files

    if completed.returncode not in (0, 1):
        files = _walk_files(root)
        return files

    files: list[Path] = []
    for line in completed.stdout.splitlines():
        if line.strip():
            files.append(Path(line))
    return files


def _walk_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file():
            files.append(path.relative_to(root))
    return files


def _bounded_scope_values(assignment: ProgrammerAssignment) -> list[str]:
    values: list[str] = []
    for value in assignment["scope"]["values"]:
        clean_value = value.strip()
        if not clean_value:
            continue
        values.append(clean_value)
        if len(values) >= 12:
            break
    return values


def _files_for_path_values(
    *,
    root: Path,
    scoped_files: list[Path],
    values: list[str],
) -> list[Path]:
    scoped_set = {_to_posix(path) for path in scoped_files}
    files: list[Path] = []
    for value in values:
        path = PurePosixPath(value.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise EvidenceCollectionError("Assignment path escapes the repository.")
        candidate = Path(*path.parts)
        candidate_root = ensure_path_inside(root / candidate, root)
        if candidate_root.is_file() and _to_posix(candidate) in scoped_set:
            files.append(candidate)
            continue
        if candidate_root.is_dir():
            for scoped_file in scoped_files:
                if (root / scoped_file).is_relative_to(candidate_root):
                    files.append(scoped_file)
    deduped = _dedupe_paths(files)
    return deduped


def _summary_rows(
    *,
    root: Path,
    relative_paths: list[Path],
    topic: str,
) -> list[CodeEvidenceRow]:
    rows: list[CodeEvidenceRow] = []
    for relative_path in relative_paths:
        text = _read_text_file(root / relative_path)
        if text is None:
            continue
        excerpt = _summary_excerpt(text)
        if not excerpt:
            continue
        row: CodeEvidenceRow = {
            "path": _to_posix(relative_path),
            "line_start": 1,
            "line_end": _line_count(excerpt),
            "symbol_or_topic": topic,
            "excerpt": excerpt,
            "reason": "Selected bounded file-scope evidence.",
        }
        rows.append(row)
    return rows


def _search_rows(
    *,
    root: Path,
    scoped_files: list[Path],
    values: list[str],
    symbol_mode: bool,
) -> list[CodeEvidenceRow]:
    rows: list[CodeEvidenceRow] = []
    seen: set[tuple[str, int, str]] = set()
    for value in values:
        value_count = 0
        for relative_path in scoped_files:
            text = _read_text_file(root / relative_path)
            if text is None:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not _line_matches(line, value, symbol_mode=symbol_mode):
                    continue
                key = (_to_posix(relative_path), line_number, value.casefold())
                if key in seen:
                    continue
                seen.add(key)
                row = _row_around_line(
                    root=root,
                    relative_path=relative_path,
                    line_number=line_number,
                    topic=value,
                )
                if row is not None:
                    rows.append(row)
                    value_count += 1
                if value_count >= MAX_SEARCH_ROWS_PER_VALUE:
                    break
            if value_count >= MAX_SEARCH_ROWS_PER_VALUE:
                break
    return rows


def _line_matches(line: str, value: str, *, symbol_mode: bool) -> bool:
    if not symbol_mode:
        return value.casefold() in line.casefold()

    stripped = line.strip()
    prefixes = _definition_terms(value)
    if stripped.startswith(prefixes):
        return True
    if value in line:
        return True
    if "." not in value:
        return False
    parts = [part for part in value.split(".") if part]
    return any(part in line for part in parts)


def _definition_terms(symbol: str) -> tuple[str, ...]:
    symbols = [symbol]
    if "." in symbol:
        symbols.extend(part for part in symbol.split(".") if part)
    prefixes = tuple(
        prefix
        for item in symbols
        for prefix in (
            f"class {item}",
            f"def {item}",
            f"async def {item}",
        )
    )
    return prefixes


def _row_around_line(
    *,
    root: Path,
    relative_path: Path,
    line_number: int,
    topic: str,
) -> CodeEvidenceRow | None:
    text = _read_text_file(root / relative_path)
    if text is None:
        return None

    lines = text.splitlines()
    if line_number < 1 or line_number > len(lines):
        return None

    start = max(1, line_number - 3)
    end = min(len(lines), line_number + 3)
    excerpt = "\n".join(lines[start - 1:end])
    row: CodeEvidenceRow = {
        "path": _to_posix(relative_path),
        "line_start": start,
        "line_end": end,
        "symbol_or_topic": topic,
        "excerpt": excerpt,
        "reason": f"Matched assignment value: {topic}",
    }
    return row


def _summary_excerpt(text: str) -> str:
    selected_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        selected_lines.append(line)
        if len(selected_lines) >= 12:
            break

    excerpt = "\n".join(selected_lines).strip()
    return excerpt


def _cap_rows(
    rows: list[CodeEvidenceRow],
    max_files: int,
    max_excerpt_chars: int,
) -> list[CodeEvidenceRow]:
    capped_rows: list[CodeEvidenceRow] = []
    files_seen: list[str] = []
    used_chars = 0
    for row in rows:
        path = row["path"]
        if path not in files_seen:
            if len(files_seen) >= max_files:
                continue
            files_seen.append(path)
        remaining_chars = max_excerpt_chars - used_chars
        if remaining_chars <= 0:
            break
        excerpt = row["excerpt"][:remaining_chars].rstrip()
        if not excerpt:
            break
        capped_row: CodeEvidenceRow = {
            "path": row["path"],
            "line_start": row["line_start"],
            "line_end": row["line_end"],
            "symbol_or_topic": row["symbol_or_topic"],
            "excerpt": excerpt,
            "reason": row["reason"],
        }
        capped_rows.append(capped_row)
        used_chars += len(excerpt)
    return capped_rows


def _files_from_rows(rows: list[CodeEvidenceRow], max_files: int) -> list[str]:
    files: list[str] = []
    for row in rows:
        path = row["path"]
        if path in files:
            continue
        files.append(path)
        if len(files) >= max_files:
            break
    return files


def _limitations_for_rows(
    rows: list[CodeEvidenceRow],
    values: list[str],
) -> list[str]:
    limitations: list[str] = []
    if not rows:
        if values:
            limitations.append("No bounded source evidence matched assignment scope.")
        else:
            limitations.append("Programmer assignment had no usable scope values.")
    return limitations


def _is_safe_relative_file(relative_path: Path) -> bool:
    parts = relative_path.parts
    if not parts or relative_path.is_absolute() or ".." in parts:
        return False
    if ".git" in parts:
        return False

    path_text = _to_posix(relative_path)
    name = parts[-1].casefold()
    if name == ".env" or name.startswith(".env."):
        return False
    if is_secret_like_path(path_text):
        return False
    if is_binary_like_path(path_text):
        return False
    return True


def _read_text_file(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None

    if b"\x00" in data:
        return None

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = _to_posix(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _line_count(text: str) -> int:
    line_count = max(1, len(text.splitlines()))
    return line_count


def _to_posix(path: Path) -> str:
    return path.as_posix()
