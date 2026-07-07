"""Bounded filesystem evidence collection for code reading."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kazusa_ai_chatbot.coding_agent.code_fetching.models import CodeSourceScope
from kazusa_ai_chatbot.coding_agent.code_reading.models import (
    CodeEvidenceRow,
    ReadingProgrammerTask,
)
from kazusa_ai_chatbot.coding_agent.code_reading.planner import (
    is_binary_like_path,
    is_secret_like_path,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import (
    PathSafetyError,
    ensure_path_inside,
)

MAX_FILE_BYTES = 512_000
MAX_SEARCH_ROWS_PER_FILE = 5
MAX_SUMMARY_LINES = 8
MAX_FOCUSED_ROWS_PER_FILE = 6
MAX_FOCUSED_CANDIDATES_PER_FILE = 32
MAX_FOCUSED_CANDIDATES_PER_TERM = 12
MAX_COMPOUND_FOCUS_TERMS = 10
MAX_PATH_MATCH_CANDIDATES = 120
MAX_RG_SEARCH_TERMS = 16
MAX_RG_MATCHES_PER_TERM = 300
MAX_RG_MATCHES_PER_FILE = 12
RG_SEARCH_TIMEOUT_SECONDS = 10
DEFINITION_TOPIC_SCORE = 1
STATE_TRANSITION_BRANCH_SCORE = 2
MAX_WORD_SHAPE_VARIANTS = 6
ROW_CONTEXT_RADIUS = 20
DEFINITION_CONTEXT_BEFORE = 8
DEFINITION_CONTEXT_AFTER = 60
MAX_DISCOVERED_SYMBOLS = 12
MAX_IMPORTED_MODULES = 12
SOURCE_CLASS_ORDER = (
    "implementation",
    "docs",
    "tests",
    "scripts",
    "config",
    "plans",
    "generated",
    "other",
)
_LOW_SIGNAL_RUNTIME_PATH_PARTS = {
    "i18n",
    "l10n",
    "locale",
    "locales",
    "migrations",
    "translation",
    "translations",
}
_LOW_SIGNAL_RUNTIME_FILE_STEMS = {
    "announcement",
    "announcements",
    "changelog",
    "release_notes",
}
_SIMPLE_CODE_PATTERN_PREFIXES = (
    ".",
    "_",
    "is_",
    "has_",
    "should_",
    "can_",
)

_PYTHON_DEFINITION_RE = re.compile(
    r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
)
_PYTHON_FROM_IMPORT_RE = re.compile(
    r"^\s*from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import\s+",
)
_PYTHON_IMPORT_RE = re.compile(
    r"^\s*import\s+([A-Za-z_][A-Za-z0-9_.]*)",
)
_FOCUS_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DOTTED_FOCUS_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
)
_FOCUS_STOPWORDS = {
    "about",
    "after",
    "and",
    "are",
    "before",
    "bounded",
    "class",
    "code",
    "construct",
    "constructs",
    "did",
    "do",
    "does",
    "for",
    "e",
    "file",
    "from",
    "generic",
    "handle",
    "handles",
    "handling",
    "how",
    "in",
    "into",
    "involving",
    "is",
    "its",
    "local",
    "look",
    "logic",
    "or",
    "read",
    "reader",
    "represent",
    "represents",
    "requested",
    "resolving",
    "role",
    "source",
    "specific",
    "specifically",
    "stage",
    "structures",
    "target",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "worker",
}
_COMPOUND_FOCUS_LEFT_TERMS = (
    "entity",
    "event",
    "integration",
    "request",
    "response",
    "service",
    "state",
)
_COMPOUND_FOCUS_RIGHT_TERMS = (
    "broadcast",
    "call",
    "dispatcher",
    "handler",
    "registry",
    "service",
    "transition",
    "update",
)
_SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".ex",
    ".exs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".lua",
    ".m",
    ".mm",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".swift",
    ".ts",
    ".tsx",
}
_DOC_EXTENSIONS = {".adoc", ".md", ".rst"}
_SCRIPT_EXTENSIONS = {".bat", ".cmd", ".ps1", ".sh"}
_CONFIG_EXTENSIONS = {
    ".cfg",
    ".conf",
    ".ini",
    ".json",
    ".lock",
    ".toml",
    ".yaml",
    ".yml",
}


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
    assignment: ReadingProgrammerTask,
    max_files: int,
    max_excerpt_chars: int,
) -> EvidenceBundle:
    """Collect bounded source rows for one programmer assignment.

    Args:
        repo_root: Resolved checkout root from source fetching.
        source_scope: Source scope that bounds all reads.
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
        focus_terms = _assignment_focus_terms(
            assignment,
            include_scope_values=False,
        )
        candidate_files = _files_for_path_values(
            root=root,
            scoped_files=scoped_files,
            values=values,
        )
        rows = _summary_rows(
            root=root,
            relative_paths=candidate_files[:max_files],
            topic=assignment["role"],
            focus_terms=focus_terms,
        )
        ranking_terms = focus_terms
    else:
        focus_terms = _assignment_focus_terms(
            assignment,
            include_scope_values=True,
        )
        rows = _search_rows(
            root=root,
            source_scope=source_scope,
            scoped_files=scoped_files,
            values=values,
            focus_terms=focus_terms,
            symbol_mode=scope_kind == "symbol",
        )
        ranking_terms = [*values, *focus_terms]

    preferred_paths: set[str] = set()
    if scope_kind == "symbol":
        preferred_paths = _definition_paths_from_rows(rows)
    rows = _rank_rows(
        rows,
        ranking_terms=ranking_terms,
        preferred_paths=preferred_paths,
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


def source_class_for_path(path: str) -> str:
    """Classify a repo-relative path by generic source role."""

    posix_path = path.replace("\\", "/")
    parts = [part.casefold() for part in posix_path.split("/") if part]
    name = parts[-1] if parts else ""
    suffix = PurePosixPath(posix_path).suffix.casefold()

    if _is_generated_path(parts):
        return "generated"
    if _is_plan_path(parts):
        return "plans"
    if _is_test_path(parts, name):
        return "tests"
    if _is_doc_path(parts, name, suffix):
        return "docs"
    if _is_script_path(parts, suffix):
        return "scripts"
    if _is_config_path(name, suffix):
        return "config"
    if suffix in _SOURCE_EXTENSIONS:
        return "implementation"
    return "other"


def summarize_safe_source_file(
    repo_root: Path,
    relative_path: str,
) -> dict[str, object]:
    """Return bounded metadata for one safe repo-relative source file."""

    path = PurePosixPath(relative_path.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise EvidenceCollectionError("Source summary path escapes repository.")
    local_path = ensure_path_inside(repo_root / Path(*path.parts), repo_root)
    text = _read_text_file(local_path)
    if text is None:
        text = ""

    summary: dict[str, object] = {
        "path": _to_posix(Path(*path.parts)),
        "source_class": source_class_for_path(relative_path),
        "defined_symbols": _defined_symbols(text),
        "imported_modules": _imported_modules(text),
        "summary_excerpt": _source_summary_excerpt(text),
    }
    return summary


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
                "--no-ignore-parent",
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


def _bounded_scope_values(assignment: ReadingProgrammerTask) -> list[str]:
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
    focus_terms: list[str],
) -> list[CodeEvidenceRow]:
    rows: list[CodeEvidenceRow] = []
    for relative_path in relative_paths:
        text = _read_text_file(root / relative_path)
        if text is None:
            continue
        focused_rows = _focused_summary_rows(
            root=root,
            relative_path=relative_path,
            text=text,
            focus_terms=focus_terms,
        )
        if focused_rows:
            rows.extend(focused_rows)
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
    source_scope: CodeSourceScope,
    scoped_files: list[Path],
    values: list[str],
    focus_terms: list[str],
    symbol_mode: bool,
) -> list[CodeEvidenceRow]:
    rows: list[CodeEvidenceRow] = []
    search_terms: list[str] = []
    for term in [*values, *focus_terms]:
        _append_unique(search_terms, term)

    seen: set[tuple[str, int, str]] = set()
    ranked_files = _rank_files_for_search(scoped_files)
    candidate_map = _search_candidate_map(
        root=root,
        source_scope=source_scope,
        scoped_files=ranked_files,
        search_terms=search_terms,
        primary_term_count=len(values),
        symbol_mode=symbol_mode,
    )
    for relative_path in ranked_files:
        candidate_specs = candidate_map.get(_to_posix(relative_path))
        if not candidate_specs:
            continue
        text = _read_text_file(root / relative_path)
        if text is None:
            continue

        lines = text.splitlines()
        file_candidates: list[tuple[int, int, int, int, CodeEvidenceRow]] = []
        file_seen: set[tuple[int, str]] = set()
        for line_number, term_order, value, path_match in candidate_specs:
            if path_match:
                row = _path_match_row(
                    relative_path=relative_path,
                    text=text,
                    topic=value,
                )
                key = (row["line_start"], value.casefold())
                if key not in file_seen:
                    file_seen.add(key)
                    file_candidates.append((
                        -_row_relevance_score(row, search_terms),
                        0,
                        _match_rank(row),
                        term_order,
                        row,
                    ))
                continue

            if line_number < 1 or line_number > len(lines):
                continue
            line = lines[line_number - 1]
            if not _line_matches(line, value, symbol_mode=symbol_mode):
                continue
            key = (line_number, value.casefold())
            if key in file_seen:
                continue
            file_seen.add(key)
            row = _row_around_line_from_lines(
                relative_path=relative_path,
                lines=lines,
                line_number=line_number,
                topic=value,
            )
            if row is not None:
                file_candidates.append((
                    -_row_relevance_score(row, search_terms),
                    _focus_line_rank(line),
                    _match_rank(row),
                    term_order,
                    row,
                ))

        ranked_file_candidates = sorted(
            file_candidates,
            key=lambda item: item[:3],
        )
        selected_file_rows: list[CodeEvidenceRow] = []
        for *_rank, row in ranked_file_candidates:
            if _row_overlaps_any(row, selected_file_rows):
                continue
            selected_file_rows.append(row)
            if len(selected_file_rows) >= MAX_SEARCH_ROWS_PER_FILE:
                break

        for row in selected_file_rows:
            key = (_to_posix(relative_path), row["line_start"], row["symbol_or_topic"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def _search_candidate_map(
    *,
    root: Path,
    source_scope: CodeSourceScope,
    scoped_files: list[Path],
    search_terms: list[str],
    primary_term_count: int,
    symbol_mode: bool,
) -> dict[str, list[tuple[int, int, str, bool]]]:
    """Find bounded candidate source lines before reading matched files."""

    candidate_map: dict[str, list[tuple[int, int, str, bool]]] = {}
    seen: set[tuple[str, int, str, bool]] = set()
    _add_path_match_candidates(
        candidate_map=candidate_map,
        seen=seen,
        scoped_files=scoped_files,
        search_terms=search_terms,
        primary_term_count=primary_term_count,
    )
    rg_completed = _add_rg_match_candidates(
        candidate_map=candidate_map,
        seen=seen,
        root=root,
        source_scope=source_scope,
        scoped_files=scoped_files,
        search_terms=search_terms,
        primary_term_count=primary_term_count,
        symbol_mode=symbol_mode,
    )
    if not rg_completed:
        _add_scanned_match_candidates(
            candidate_map=candidate_map,
            seen=seen,
            root=root,
            scoped_files=scoped_files,
            search_terms=search_terms,
            primary_term_count=primary_term_count,
            symbol_mode=symbol_mode,
        )
    return candidate_map


def _add_path_match_candidates(
    *,
    candidate_map: dict[str, list[tuple[int, int, str, bool]]],
    seen: set[tuple[str, int, str, bool]],
    scoped_files: list[Path],
    search_terms: list[str],
    primary_term_count: int,
) -> None:
    """Add bounded file-path matches from PM-provided scope values only."""

    path_match_count = 0
    for relative_path in scoped_files:
        for term_order, value in enumerate(search_terms[:primary_term_count]):
            if not _value_matches_path(value, relative_path):
                continue
            _append_search_candidate(
                candidate_map=candidate_map,
                seen=seen,
                relative_path=relative_path,
                line_number=1,
                term_order=term_order,
                value=value,
                path_match=True,
            )
            path_match_count += 1
            if path_match_count >= MAX_PATH_MATCH_CANDIDATES:
                return


def _add_rg_match_candidates(
    *,
    candidate_map: dict[str, list[tuple[int, int, str, bool]]],
    seen: set[tuple[str, int, str, bool]],
    root: Path,
    source_scope: CodeSourceScope,
    scoped_files: list[Path],
    search_terms: list[str],
    primary_term_count: int,
    symbol_mode: bool,
) -> bool:
    """Add bounded ripgrep matches and report whether ripgrep completed."""

    scoped_set = {_to_posix(path) for path in scoped_files}
    rg_terms = _rg_search_terms(
        search_terms,
        primary_term_count=primary_term_count,
        symbol_mode=symbol_mode,
    )
    scope_paths = _rg_scope_paths(root, source_scope)
    for term_order, value in rg_terms:
        term_match_count = 0
        for pattern in _rg_patterns_for_value(value):
            if term_match_count >= MAX_RG_MATCHES_PER_TERM:
                break
            matches = _rg_matches_for_pattern(
                root=root,
                pattern=pattern,
                scope_paths=scope_paths,
                ignore_case=not symbol_mode,
            )
            if matches is None:
                return False
            for relative_path, line_number, line in matches:
                path_text = _to_posix(relative_path)
                if path_text not in scoped_set:
                    continue
                if not _line_matches(line, value, symbol_mode=symbol_mode):
                    continue
                _append_search_candidate(
                    candidate_map=candidate_map,
                    seen=seen,
                    relative_path=relative_path,
                    line_number=line_number,
                    term_order=term_order,
                    value=value,
                    path_match=False,
                )
                term_match_count += 1
                if term_match_count >= MAX_RG_MATCHES_PER_TERM:
                    break
    return True


def _add_scanned_match_candidates(
    *,
    candidate_map: dict[str, list[tuple[int, int, str, bool]]],
    seen: set[tuple[str, int, str, bool]],
    root: Path,
    scoped_files: list[Path],
    search_terms: list[str],
    primary_term_count: int,
    symbol_mode: bool,
) -> None:
    """Add bounded in-process line matches when ripgrep is unavailable."""

    rg_terms = _rg_search_terms(
        search_terms,
        primary_term_count=primary_term_count,
        symbol_mode=symbol_mode,
    )
    for term_order, value in rg_terms:
        term_match_count = 0
        for relative_path in scoped_files:
            if term_match_count >= MAX_RG_MATCHES_PER_TERM:
                break
            text = _read_text_file(root / relative_path)
            if text is None:
                continue
            file_match_count = 0
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not _line_matches(line, value, symbol_mode=symbol_mode):
                    continue
                _append_search_candidate(
                    candidate_map=candidate_map,
                    seen=seen,
                    relative_path=relative_path,
                    line_number=line_number,
                    term_order=term_order,
                    value=value,
                    path_match=False,
                )
                file_match_count += 1
                term_match_count += 1
                if file_match_count >= MAX_RG_MATCHES_PER_FILE:
                    break
                if term_match_count >= MAX_RG_MATCHES_PER_TERM:
                    break


def _append_search_candidate(
    *,
    candidate_map: dict[str, list[tuple[int, int, str, bool]]],
    seen: set[tuple[str, int, str, bool]],
    relative_path: Path,
    line_number: int,
    term_order: int,
    value: str,
    path_match: bool,
) -> None:
    """Record one deduplicated candidate line or path match."""

    path_text = _to_posix(relative_path)
    key = (path_text, line_number, value.casefold(), path_match)
    if key in seen:
        return
    seen.add(key)
    candidates = candidate_map.setdefault(path_text, [])
    candidates.append((line_number, term_order, value, path_match))


def _rg_search_terms(
    search_terms: list[str],
    *,
    primary_term_count: int,
    symbol_mode: bool,
) -> list[tuple[int, str]]:
    """Choose high-signal terms for external search candidate discovery."""

    terms: list[tuple[int, str]] = []
    candidate_terms = search_terms
    if symbol_mode:
        candidate_terms = search_terms[:primary_term_count]
    seen_term_keys: set[str] = set()
    for term_order, term in enumerate(candidate_terms):
        clean_term = term.strip()
        if not clean_term:
            continue
        normalized_term = _normalized_match_text(clean_term)
        if len(normalized_term) < 3:
            continue
        if normalized_term in seen_term_keys:
            continue
        seen_term_keys.add(normalized_term)
        primary_term = term_order < primary_term_count
        high_signal_term = (
            _focus_term_rank(clean_term) <= 1
            or len(normalized_term) >= 6
        )
        if not primary_term and not high_signal_term:
            continue
        terms.append((term_order, clean_term))
        if len(terms) >= MAX_RG_SEARCH_TERMS:
            break
    return terms


def _rg_patterns_for_value(value: str) -> list[str]:
    """Build fixed-string ripgrep patterns for one semantic search term."""

    patterns: list[str] = []
    clean_value = value.strip()
    if not clean_value:
        return patterns

    if "." in clean_value:
        _append_unique(patterns, clean_value)
        leaf_value = clean_value.rsplit(".", maxsplit=1)[-1]
        _append_unique(patterns, leaf_value)
        if "_" in leaf_value:
            _append_identifier_search_patterns(
                patterns,
                leaf_value,
                derived_first=False,
            )
        return patterns

    if " " in clean_value:
        base_parts = _base_focus_parts(clean_value)
        identifier_value = "_".join(base_parts)
        if identifier_value:
            _append_unique(patterns, identifier_value)
        for base_part in base_parts:
            for word_variant in _word_shape_variants(base_part):
                _append_simple_code_search_patterns(patterns, word_variant)
        _append_unique(patterns, clean_value)
        return patterns

    if "_" in clean_value:
        _append_identifier_search_patterns(
            patterns,
            clean_value,
            derived_first=True,
        )
    else:
        for word_variant in _word_shape_variants(clean_value):
            _append_simple_code_search_patterns(patterns, word_variant)
    return patterns


def _append_simple_code_search_patterns(patterns: list[str], value: str) -> None:
    """Append fixed-string patterns that prefer source-shaped occurrences."""

    if not value.isidentifier():
        _append_unique(patterns, value)
        return

    folded_value = value.casefold()
    if folded_value != value or len(folded_value) < 3:
        _append_unique(patterns, value)
        return

    for prefix in _SIMPLE_CODE_PATTERN_PREFIXES:
        _append_unique(patterns, f"{prefix}{folded_value}")
    _append_unique(patterns, f"{folded_value}_")
    _append_unique(patterns, folded_value)


def _append_identifier_search_patterns(
    patterns: list[str],
    value: str,
    *,
    derived_first: bool,
) -> None:
    """Append fixed-string patterns for identifier-like search values."""

    derived_patterns: list[str] = []
    if not value.startswith(("async_", "handle_")):
        derived_patterns = [
            f"handle_{value}",
            f"async_{value}",
        ]
    base_patterns = _identifier_base_search_values(value)

    if derived_first:
        for pattern in derived_patterns:
            _append_unique(patterns, pattern)
        _append_unique(patterns, value)
    else:
        _append_unique(patterns, value)
        for pattern in derived_patterns:
            _append_unique(patterns, pattern)
    for pattern in base_patterns:
        _append_unique(patterns, pattern)


def _identifier_base_search_values(value: str) -> list[str]:
    """Return base identifiers for generated variant-style function names."""

    if "_for_" not in value:
        return []

    base_value = value.split("_for_", maxsplit=1)[0]
    if base_value == value:
        return []
    if len(_normalized_match_text(base_value)) < 6:
        return []
    return [base_value]


def _rg_scope_paths(root: Path, source_scope: CodeSourceScope) -> list[str]:
    """Return repository-relative ripgrep path limits for a source scope."""

    scoped_path = source_scope.get("repo_relative_path")
    if scoped_path is None:
        return []

    scope_root = _safe_scope_root(root, scoped_path)
    relative_scope = scope_root.relative_to(root)
    scope_text = _to_posix(relative_scope)
    return [scope_text]


def _rg_matches_for_pattern(
    *,
    root: Path,
    pattern: str,
    scope_paths: list[str],
    ignore_case: bool,
) -> list[tuple[Path, int, str]] | None:
    """Run one bounded ripgrep query or return ``None`` when rg fails."""

    args = [
        "rg",
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color",
        "never",
        "--hidden",
        "--no-ignore-parent",
        "--fixed-strings",
        "--max-count",
        str(MAX_RG_MATCHES_PER_FILE),
        "--max-filesize",
        str(MAX_FILE_BYTES),
        "-g",
        "!.git/*",
        "-g",
        "!.tmp_pytest/**",
        "-g",
        "!**/.*/**",
        "-g",
        "!.env",
        "-g",
        "!.env.*",
        "-g",
        "!**/*secret*",
        "-g",
        "!**/*credential*",
        "-g",
        "!**/*private_key*",
        "-g",
        "!**/*id_rsa*",
        "-g",
        "!**/*token*",
        "-g",
        "!**/*.pem",
        "-g",
        "!**/*.key",
        "-g",
        "!**/*.p12",
        "-g",
        "!**/*.pfx",
        "-e",
        pattern,
    ]
    if ignore_case:
        args.append("--ignore-case")
    args.append("--")
    if scope_paths:
        args.extend(scope_paths)
    else:
        args.append(".")

    try:
        completed = subprocess.run(
            args,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=RG_SEARCH_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode not in (0, 1):
        return None

    matches: list[tuple[Path, int, str]] = []
    for line in completed.stdout.splitlines():
        parsed_match = _parse_rg_match_line(line)
        if parsed_match is None:
            continue
        matches.append(parsed_match)
        if len(matches) >= MAX_RG_MATCHES_PER_TERM:
            break
    return matches


def _parse_rg_match_line(line: str) -> tuple[Path, int, str] | None:
    """Parse one ripgrep ``path:line:text`` output row."""

    parts = line.split(":", maxsplit=2)
    if len(parts) != 3:
        return None
    path_text = parts[0].strip()
    line_number_text = parts[1].strip()
    if not path_text or not line_number_text.isdecimal():
        return None

    relative_path = Path(path_text)
    line_number = int(line_number_text)
    line_text = parts[2]
    parsed_match = (relative_path, line_number, line_text)
    return parsed_match


def _path_match_row(
    *,
    relative_path: Path,
    text: str,
    topic: str,
) -> CodeEvidenceRow:
    line_number = _first_definition_line(text)
    lines = text.splitlines()
    row = _row_around_line_from_lines(
        relative_path=relative_path,
        lines=lines,
        line_number=line_number,
        topic=topic,
    )
    if row is None:
        excerpt = _source_summary_excerpt(text)
        row = {
            "path": _to_posix(relative_path),
            "line_start": 1,
            "line_end": _line_count(excerpt),
            "symbol_or_topic": topic,
            "excerpt": excerpt,
            "reason": f"Matched assignment path value: {topic}",
        }
    return row


def _first_definition_line(text: str) -> int:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _PYTHON_DEFINITION_RE.match(line):
            return line_number
    return 1


def _focused_summary_rows(
    *,
    root: Path,
    relative_path: Path,
    text: str,
    focus_terms: list[str],
) -> list[CodeEvidenceRow]:
    if not focus_terms:
        return []

    candidates: list[tuple[int, int, int, int, CodeEvidenceRow]] = []
    seen_lines: set[int] = set()
    lines = text.splitlines()
    for term_order, term in enumerate(focus_terms):
        term_lower = term.casefold()
        term_candidates: list[tuple[int, int, int, CodeEvidenceRow]] = []
        for line_number, line in enumerate(lines, start=1):
            if line_number in seen_lines:
                continue
            if term_lower not in line.casefold():
                continue
            row = _row_around_line(
                root=root,
                relative_path=relative_path,
                line_number=line_number,
                topic=term,
            )
            if row is None:
                continue
            seen_lines.add(line_number)
            term_candidates.append((
                -_row_relevance_score(row, focus_terms),
                _focus_line_rank(line),
                line_number,
                row,
            ))

        ranked_term_candidates = sorted(
            term_candidates,
            key=lambda item: item[:3],
        )
        for row_score, line_rank, line_number, row in (
            ranked_term_candidates[:MAX_FOCUSED_CANDIDATES_PER_TERM]
        ):
            candidates.append((
                row_score,
                term_order,
                line_rank,
                line_number,
                row,
            ))
            if len(candidates) >= MAX_FOCUSED_CANDIDATES_PER_FILE:
                break
        if len(candidates) >= MAX_FOCUSED_CANDIDATES_PER_FILE:
            break
    ranked_candidates = sorted(candidates, key=lambda item: item[:3])
    rows: list[CodeEvidenceRow] = []
    for *_rank, row in ranked_candidates:
        if _row_overlaps_any(row, rows):
            continue
        rows.append(row)
        if len(rows) >= MAX_FOCUSED_ROWS_PER_FILE:
            break
    return rows


def _focus_line_rank(line: str) -> int:
    stripped = line.strip()
    if not stripped:
        return 6
    if stripped.startswith(("from ", "import ")):
        return 4
    if stripped.startswith(("#", '"""', "'''")):
        return 5
    if _line_contains_state_transition(stripped):
        return 0
    if stripped.startswith(("async def ", "def ", "class ")):
        return 0
    if "(" in stripped and ")" not in stripped:
        return 1
    if "(" in stripped or "." in stripped or "=" in stripped:
        return 2
    return 3


def _assignment_focus_terms(
    assignment: ReadingProgrammerTask,
    *,
    include_scope_values: bool,
) -> list[str]:
    """Return ranked source terms from assignment text and optional values."""

    scope = assignment["scope"]
    text_parts = [
        assignment["role"],
        *assignment["questions"],
        *assignment["required_slots"],
    ]
    if include_scope_values:
        text_parts.insert(1, " ".join(scope["values"]))
    terms: list[str] = []
    for text in text_parts:
        _append_dotted_focus_terms(terms, text)
        for match in _FOCUS_TOKEN_RE.finditer(text):
            token = match.group(0)
            if _is_specific_identifier(token):
                _append_unique(terms, token)
                for prefix in _identifier_prefixes(token):
                    _append_unique(terms, prefix)
            for term in _split_focus_token(token):
                if term.casefold() in _FOCUS_STOPWORDS:
                    continue
                _append_unique(terms, term)
    _append_compound_focus_terms(terms)
    ranked_terms = [
        term
        for _, term in sorted(
            enumerate(terms),
            key=lambda item: (
                _focus_term_rank(item[1]),
                item[0],
            ),
        )
    ]
    return ranked_terms[:24]


def _append_dotted_focus_terms(terms: list[str], text: str) -> None:
    """Preserve dotted code expressions from PM assignment text."""

    for match in _DOTTED_FOCUS_TOKEN_RE.finditer(text):
        expression = match.group(0).strip(".")
        normalized_expression = _normalized_match_text(expression)
        if len(normalized_expression) < 6:
            continue
        leaf = expression.rsplit(".", maxsplit=1)[-1]
        if len(leaf) < 3 and "_" not in leaf:
            continue
        _append_unique(terms, expression)


def _append_compound_focus_terms(terms: list[str]) -> None:
    """Add bounded snake_case terms that bridge prose nouns to code symbols."""

    simple_terms: set[str] = set()
    for term in terms:
        folded_term = term.casefold()
        if "_" in folded_term:
            continue
        if folded_term in _FOCUS_STOPWORDS:
            continue
        simple_terms.add(folded_term)

    added_count = 0
    for left_term in _COMPOUND_FOCUS_LEFT_TERMS:
        if left_term not in simple_terms:
            continue
        for right_term in _COMPOUND_FOCUS_RIGHT_TERMS:
            if right_term == left_term:
                continue
            if right_term not in simple_terms:
                continue
            compound_term = f"{left_term}_{right_term}"
            _append_unique(terms, compound_term)
            added_count += 1
            if added_count >= MAX_COMPOUND_FOCUS_TERMS:
                return


def _focus_term_rank(term: str) -> int:
    if "_" in term:
        return 0
    if _is_specific_identifier(term):
        return 1
    return 2


def _is_specific_identifier(token: str) -> bool:
    if token.isupper() and len(token) <= 3:
        return False
    return "_" in token or any(char.isupper() for char in token[1:])


def _identifier_prefixes(token: str) -> list[str]:
    if "_" not in token:
        return []
    parts = [part for part in token.split("_") if part]
    prefixes: list[str] = []
    for index in range(2, len(parts)):
        prefix = "_".join(parts[:index])
        if len(prefix) >= 3:
            prefixes.append(prefix)
    return prefixes


def _split_focus_token(token: str) -> list[str]:
    raw_parts = _base_focus_parts(token)
    parts: list[str] = []
    for part in raw_parts:
        for variant in _word_shape_variants(part):
            if variant.casefold() in _FOCUS_STOPWORDS:
                continue
            _append_unique(parts, variant)
    if not parts and len(token) >= 3:
        parts = [token]
    return parts


def _base_focus_parts(token: str) -> list[str]:
    """Split free-form assignment text into code-searchable base words."""

    parts = [
        part
        for part in re.split(r"[_\W]+", token)
        if len(part) >= 3 and part.casefold() not in _FOCUS_STOPWORDS
    ]
    return parts


def _word_shape_variants(token: str) -> list[str]:
    """Return bounded suffix variants for words used in source searches."""

    variants = [token]
    folded_token = token.casefold()
    if not folded_token.isidentifier():
        return variants

    if (
        folded_token.endswith("s")
        and not folded_token.endswith("ss")
        and not folded_token.endswith("us")
        and len(folded_token) > 3
    ):
        _append_unique(variants, token[:-1])
    if folded_token.endswith("ed") and len(folded_token) > 4:
        _append_unique(variants, token[:-2])

    bounded_variants = variants[:MAX_WORD_SHAPE_VARIANTS]
    return bounded_variants


def _line_matches(line: str, value: str, *, symbol_mode: bool) -> bool:
    candidate_values = [value, *_identifier_base_search_values(value)]
    if not symbol_mode:
        return any(
            candidate_value.casefold() in line.casefold()
            or _normalized_value_matches(candidate_value, line)
            for candidate_value in candidate_values
        )

    stripped = line.strip()
    for candidate_value in candidate_values:
        prefixes = _definition_terms(candidate_value)
        if stripped.startswith(prefixes):
            return True
        if candidate_value in line:
            return True
        if _normalized_value_matches(candidate_value, line):
            return True
        if "." not in candidate_value:
            continue
        leaf_value = candidate_value.rsplit(".", maxsplit=1)[-1]
        if leaf_value and leaf_value in line:
            return True
    return False


def _value_matches_path(value: str, relative_path: Path) -> bool:
    path_text = _to_posix(relative_path)
    if value.casefold() in path_text.casefold():
        return True
    if _is_specific_identifier(value):
        return False
    return _normalized_value_matches(value, path_text)


def _normalized_value_matches(value: str, text: str) -> bool:
    normalized_value = _normalized_match_text(value)
    if not normalized_value:
        return False
    normalized_text = _normalized_match_text(text)
    return normalized_value in normalized_text


def _normalized_match_text(value: str) -> str:
    return "".join(
        char.casefold()
        for char in value
        if char.isalnum()
    )


def _definition_terms(symbol: str) -> tuple[str, ...]:
    symbols = [symbol]
    if "." in symbol:
        symbols.append(symbol.rsplit(".", maxsplit=1)[-1])
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

    row = _row_around_line_from_lines(
        relative_path=relative_path,
        lines=lines,
        line_number=line_number,
        topic=topic,
    )
    return row


def _row_around_line_from_lines(
    *,
    relative_path: Path,
    lines: list[str],
    line_number: int,
    topic: str,
) -> CodeEvidenceRow | None:
    """Build a bounded excerpt row around an already-read line list."""

    if line_number < 1 or line_number > len(lines):
        return None

    stripped_line = lines[line_number - 1].strip()
    if stripped_line.startswith(("async def ", "def ", "class ")):
        start = max(1, line_number - DEFINITION_CONTEXT_BEFORE)
        end = min(len(lines), line_number + DEFINITION_CONTEXT_AFTER)
    else:
        start = max(1, line_number - ROW_CONTEXT_RADIUS)
        end = min(len(lines), line_number + ROW_CONTEXT_RADIUS)
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
    return _source_summary_excerpt(text)


def _source_summary_excerpt(text: str) -> str:
    selected_lines: list[str] = []
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_summary_worthy_line(stripped):
            _append_unique(selected_lines, line)
        if len(selected_lines) >= MAX_SUMMARY_LINES:
            break

    if len(selected_lines) < MAX_SUMMARY_LINES:
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            _append_unique(selected_lines, line)
            if len(selected_lines) >= MAX_SUMMARY_LINES:
                break

    excerpt = "\n".join(selected_lines).strip()
    return excerpt


def _rank_files_for_search(files: list[Path]) -> list[Path]:
    ranked_files = sorted(
        files,
        key=lambda path: (
            _source_class_rank(source_class_for_path(_to_posix(path))),
            len(path.parts),
            _to_posix(path).casefold(),
        ),
    )
    return ranked_files


def _rank_rows(
    rows: list[CodeEvidenceRow],
    *,
    ranking_terms: list[str] | None = None,
    preferred_paths: set[str] | None = None,
) -> list[CodeEvidenceRow]:
    terms = ranking_terms or []
    preferred = preferred_paths or set()
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            row["path"] not in preferred,
            _retrieval_source_tier(row["path"]),
            -_row_relevance_score(row, terms),
            _match_rank(row),
            _retrieval_path_penalty(row["path"]),
            _source_class_rank(source_class_for_path(row["path"])),
            len(PurePosixPath(row["path"]).parts),
            row["path"].casefold(),
            row["line_start"],
        ),
    )
    return ranked_rows


def _definition_paths_from_rows(rows: list[CodeEvidenceRow]) -> set[str]:
    """Return paths whose selected excerpts define the searched topic."""

    definition_paths: set[str] = set()
    for row in rows:
        if not _row_defines_topic(row):
            continue
        definition_paths.add(row["path"])
    return definition_paths


def _source_class_rank(source_class: str) -> int:
    try:
        return SOURCE_CLASS_ORDER.index(source_class)
    except ValueError:
        return len(SOURCE_CLASS_ORDER)


def _retrieval_source_tier(path: str) -> int:
    """Group source classes by usefulness for answering runtime flow questions."""

    source_class = source_class_for_path(path)
    if source_class in {"implementation", "docs"}:
        return 0
    if source_class in {"scripts", "config"}:
        return 1
    if source_class == "tests":
        return 2
    return 3


def _retrieval_path_penalty(path: str) -> int:
    """Deprioritize historical or localization files for runtime flow ties."""

    posix_path = path.replace("\\", "/")
    path_parts = [
        part.casefold()
        for part in posix_path.split("/")
        if part
    ]
    stem = PurePosixPath(posix_path).stem.casefold()
    if any(part in _LOW_SIGNAL_RUNTIME_PATH_PARTS for part in path_parts):
        return 1
    if stem in _LOW_SIGNAL_RUNTIME_FILE_STEMS:
        return 1
    return 0


def _match_rank(row: CodeEvidenceRow) -> int:
    topic_values = _topic_search_values(row["symbol_or_topic"])
    for line in row["excerpt"].splitlines():
        stripped = line.strip()
        for topic in topic_values:
            if stripped.startswith(_definition_terms(topic)):
                return 0
            if topic.casefold() in row["path"].casefold():
                return 1
            if topic.casefold() in PurePosixPath(row["path"]).stem.casefold():
                return 1
    return 2


def _row_relevance_score(
    row: CodeEvidenceRow,
    ranking_terms: list[str],
) -> int:
    """Score assignment-term coverage and generic decision evidence."""

    if not ranking_terms:
        return 0

    haystack = "\n".join([
        row["path"],
        row["symbol_or_topic"],
        row["excerpt"],
    ])
    folded_haystack = haystack.casefold()
    normalized_haystack = _normalized_match_text(haystack)
    score = 0
    seen_terms: set[str] = set()
    for term in ranking_terms:
        clean_term = term.strip()
        if not clean_term:
            continue
        term_key = clean_term.casefold()
        if term_key in seen_terms:
            continue
        seen_terms.add(term_key)
        normalized_term = _normalized_match_text(clean_term)
        if clean_term.casefold() in folded_haystack:
            score += 1
            continue
        if normalized_term and normalized_term in normalized_haystack:
            score += 1
    if _row_defines_topic(row):
        score += DEFINITION_TOPIC_SCORE
    if _row_contains_state_transition_branch(row["excerpt"]):
        score += STATE_TRANSITION_BRANCH_SCORE
    return score


def _row_defines_topic(row: CodeEvidenceRow) -> bool:
    """Return whether the excerpt contains a definition for its search topic."""

    topic_values = _topic_search_values(row["symbol_or_topic"])
    for line in row["excerpt"].splitlines():
        stripped = line.strip()
        for topic in topic_values:
            if stripped.startswith(_definition_terms(topic)):
                return True
    return False


def _topic_search_values(topic: str) -> list[str]:
    """Return topic identifiers that count as direct definition matches."""

    topic_values = [topic]
    for base_value in _identifier_base_search_values(topic):
        _append_unique(topic_values, base_value)
    return topic_values


def _row_contains_state_transition_branch(excerpt: str) -> bool:
    """Return whether an excerpt shows branch-controlled state transition."""

    has_branch = False
    has_transition = False
    for line in excerpt.splitlines():
        stripped = line.strip()
        if stripped.startswith(("if ", "elif ", "else:", "except ", "case ")):
            has_branch = True
        if _line_contains_state_transition(stripped):
            has_transition = True
        if has_branch and has_transition:
            return True
    return False


def _line_contains_state_transition(stripped_line: str) -> bool:
    """Return whether a line mutates a state-like field or state setter."""

    if "set_state(" in stripped_line:
        return True
    if ".state =" in stripped_line:
        return True
    if "state =" in stripped_line:
        return True
    if "state=" in stripped_line:
        return True
    if "flags |=" in stripped_line:
        return True
    return False


def _row_overlaps_any(
    row: CodeEvidenceRow,
    selected_rows: list[CodeEvidenceRow],
) -> bool:
    """Return whether a row materially overlaps a selected excerpt window."""

    for selected_row in selected_rows:
        if row["line_end"] < selected_row["line_start"]:
            continue
        if row["line_start"] > selected_row["line_end"]:
            continue
        overlap_start = max(row["line_start"], selected_row["line_start"])
        overlap_end = min(row["line_end"], selected_row["line_end"])
        overlap_lines = overlap_end - overlap_start + 1
        row_lines = row["line_end"] - row["line_start"] + 1
        selected_lines = (
            selected_row["line_end"] - selected_row["line_start"] + 1
        )
        shorter_window_lines = min(row_lines, selected_lines)
        if overlap_lines * 2 >= shorter_window_lines:
            return True
    return False


def _is_summary_worthy_line(stripped_line: str) -> bool:
    if _PYTHON_DEFINITION_RE.match(stripped_line):
        return True
    if _PYTHON_FROM_IMPORT_RE.match(stripped_line):
        return True
    if _PYTHON_IMPORT_RE.match(stripped_line):
        return True
    return False


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


def _defined_symbols(text: str) -> list[str]:
    symbols: list[str] = []
    for line in text.splitlines():
        match = _PYTHON_DEFINITION_RE.match(line)
        if match is None:
            continue
        _append_unique(symbols, match.group(1))
        if len(symbols) >= MAX_DISCOVERED_SYMBOLS:
            break
    return symbols


def _imported_modules(text: str) -> list[str]:
    modules: list[str] = []
    for line in text.splitlines():
        from_match = _PYTHON_FROM_IMPORT_RE.match(line)
        if from_match is not None:
            _append_unique(modules, from_match.group(1))
            if len(modules) >= MAX_IMPORTED_MODULES:
                break
            continue

        import_match = _PYTHON_IMPORT_RE.match(line)
        if import_match is None:
            continue
        for module_name in import_match.group(1).split(","):
            clean_module = module_name.strip().split(" as ", maxsplit=1)[0]
            _append_unique(modules, clean_module)
            if len(modules) >= MAX_IMPORTED_MODULES:
                break
        if len(modules) >= MAX_IMPORTED_MODULES:
            break
    return modules


def _is_generated_path(parts: list[str]) -> bool:
    generated_markers = {
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "generated",
        "node_modules",
        "vendor",
    }
    return any(part in generated_markers for part in parts)


def _is_plan_path(parts: list[str]) -> bool:
    plan_markers = {
        "development_plans",
        "plans",
        "roadmap",
    }
    return any(part in plan_markers for part in parts)


def _is_test_path(parts: list[str], name: str) -> bool:
    if any(part in {"spec", "specs", "test", "tests"} for part in parts):
        return True
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    if ".spec." in name or ".test." in name:
        return True
    return False


def _is_doc_path(parts: list[str], name: str, suffix: str) -> bool:
    if any(part in {"doc", "docs", "documentation"} for part in parts):
        return True
    if name in {"readme.md", "changelog.md", "contributing.md"}:
        return True
    return suffix in _DOC_EXTENSIONS


def _is_script_path(parts: list[str], suffix: str) -> bool:
    if parts and parts[0] in {"bin", "scripts", "tools"}:
        return True
    return suffix in _SCRIPT_EXTENSIONS


def _is_config_path(name: str, suffix: str) -> bool:
    config_names = {
        ".editorconfig",
        ".gitignore",
        "dockerfile",
        "makefile",
        "package-lock.json",
        "package.json",
        "poetry.lock",
        "pyproject.toml",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
        "tox.ini",
    }
    if name in config_names:
        return True
    if name.startswith("requirements") and suffix == ".txt":
        return True
    return suffix in _CONFIG_EXTENSIONS


def _is_safe_relative_file(relative_path: Path) -> bool:
    parts = relative_path.parts
    if not parts or relative_path.is_absolute() or ".." in parts:
        return False
    if ".git" in parts:
        return False
    if _is_hidden_support_path(parts):
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


def _is_hidden_support_path(parts: tuple[str, ...]) -> bool:
    """Exclude hidden support directories from broad source evidence scans."""

    for part in parts[:-1]:
        if part.startswith("."):
            return True
    return False


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


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)
