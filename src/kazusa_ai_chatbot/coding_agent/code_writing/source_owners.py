"""Structural source-owner candidates for bounded patch planning."""

from __future__ import annotations

import ast
import re
from pathlib import Path, PurePosixPath

from kazusa_ai_chatbot.coding_agent.code_reading.models import CodeEvidenceRow
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    SourceOwnerCandidate,
)
from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
    _safe_repo_relative_path,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import (
    PathSafetyError,
    ensure_path_inside,
)

MAX_OWNER_CANDIDATES = 12
MAX_OWNER_FILE_BYTES = 512000
MAX_OWNER_WINDOW_LINES = 160
MAX_OWNER_SYMBOLS = 12
PYTHON_SUFFIXES = {".py", ".pyi"}
DOC_SUFFIXES = {".md", ".markdown", ".rst", ".txt"}
CONFIG_NAMES = {
    "config.py",
    "settings.py",
    "configuration.py",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
}
EXCEPTION_TYPE_RE = re.compile(
    r"\b(?:raise|except)\s+([A-Z][A-Za-z0-9_]*)\b"
)
PYTEST_RAISES_RE = re.compile(
    r"\bpytest\.raises\(\s*([A-Z][A-Za-z0-9_]*)\b"
)


def collect_source_owner_candidates(
    *,
    repo_root: Path | None,
    reading_evidence: list[CodeEvidenceRow],
    max_candidates: int = MAX_OWNER_CANDIDATES,
) -> list[SourceOwnerCandidate]:
    """Build generic source-owner hints from evidence-backed repository files.

    Args:
        repo_root: Existing repository root used only for bounded structural
            inspection. `None` means no existing source owner candidates.
        reading_evidence: Phase 1 repo-relative evidence rows.
        max_candidates: Maximum candidates returned to model-facing stages.

    Returns:
        Ranked structural candidates with source roles, symbols, and generic
        feature markers. The function never interprets the user request.
    """

    if repo_root is None:
        return []

    evidence_by_path = _evidence_by_safe_path(reading_evidence)
    candidates: list[SourceOwnerCandidate] = []
    for safe_path, rows in evidence_by_path.items():
        text = _candidate_text(
            repo_root=repo_root,
            safe_path=safe_path,
            rows=rows,
        )
        role = _source_owner_role(safe_path)
        symbols = _defined_symbols(text) if _is_python_path(safe_path) else []
        exception_types = _exception_types(text)
        feature_markers = _feature_markers(
            safe_path=safe_path,
            role=role,
            text=text,
        )
        reasons = _candidate_reasons(
            role=role,
            feature_markers=feature_markers,
        )
        line_start = min(row["line_start"] for row in rows)
        line_end = max(row["line_end"] for row in rows)
        candidate: SourceOwnerCandidate = {
            "path": safe_path,
            "role": role,
            "line_start": line_start,
            "line_end": line_end,
            "symbols": symbols,
            "exception_types": exception_types,
            "feature_markers": feature_markers,
            "reasons": reasons,
            "evidence_refs": [_evidence_ref(row) for row in rows],
        }
        candidates.append(candidate)

    ranked_candidates = sorted(candidates, key=_candidate_rank)
    bounded_candidates = _bounded_candidates_with_role_coverage(
        ranked_candidates,
        max_candidates=max_candidates,
    )
    return bounded_candidates


def ordered_owner_paths(
    owner_candidates: list[SourceOwnerCandidate],
) -> list[str]:
    """Return ranked candidate paths without duplicate entries."""

    paths: list[str] = []
    for candidate in owner_candidates:
        path = candidate["path"]
        if path in paths:
            continue
        paths.append(path)
    return paths


def _bounded_candidates_with_role_coverage(
    ranked_candidates: list[SourceOwnerCandidate],
    *,
    max_candidates: int,
) -> list[SourceOwnerCandidate]:
    """Keep high-rank candidates while preserving non-runtime owner roles."""

    if max_candidates <= 0:
        return []

    selected = list(ranked_candidates[:max_candidates])
    selected_paths = {candidate["path"] for candidate in selected}
    for role in ("test", "docs", "config"):
        if any(candidate["role"] == role for candidate in selected):
            continue
        role_candidate = next(
            (
                candidate
                for candidate in ranked_candidates
                if candidate["role"] == role
                and candidate["path"] not in selected_paths
            ),
            None,
        )
        if role_candidate is None:
            continue
        selected.append(role_candidate)
        selected_paths.add(role_candidate["path"])

    selected = _trim_role_covered_candidates(
        selected,
        max_candidates=max_candidates,
    )
    return sorted(selected, key=_candidate_rank)


def _trim_role_covered_candidates(
    candidates: list[SourceOwnerCandidate],
    *,
    max_candidates: int,
) -> list[SourceOwnerCandidate]:
    selected = list(candidates)
    while len(selected) > max_candidates:
        role_counts = _role_counts(selected)
        remove_index = _removable_candidate_index(selected, role_counts)
        selected.pop(remove_index)
    return selected


def _role_counts(candidates: list[SourceOwnerCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        role = candidate["role"]
        counts[role] = counts.get(role, 0) + 1
    return counts


def _removable_candidate_index(
    candidates: list[SourceOwnerCandidate],
    role_counts: dict[str, int],
) -> int:
    for index in range(len(candidates) - 1, -1, -1):
        role = candidates[index]["role"]
        if role_counts.get(role, 0) > 1:
            return index
    return len(candidates) - 1


def _evidence_by_safe_path(
    reading_evidence: list[CodeEvidenceRow],
) -> dict[str, list[CodeEvidenceRow]]:
    evidence_by_path: dict[str, list[CodeEvidenceRow]] = {}
    for row in reading_evidence:
        safe_path = _safe_repo_relative_path(row["path"])
        if safe_path is None:
            continue
        evidence_by_path.setdefault(safe_path, []).append(row)
    return evidence_by_path


def _candidate_text(
    *,
    repo_root: Path,
    safe_path: str,
    rows: list[CodeEvidenceRow],
) -> str:
    file_text = _read_bounded_text(repo_root=repo_root, safe_path=safe_path)
    if file_text is not None:
        return _evidence_window_text(file_text=file_text, rows=rows)

    evidence_text = "\n".join(row["excerpt"] for row in rows)
    return evidence_text


def _evidence_window_text(*, file_text: str, rows: list[CodeEvidenceRow]) -> str:
    lines = file_text.splitlines()
    snippets: list[str] = []
    for row in sorted(rows, key=lambda item: item["line_start"]):
        start = max(row["line_start"], 1)
        end = max(row["line_end"], start)
        if end - start + 1 > MAX_OWNER_WINDOW_LINES:
            end = start + MAX_OWNER_WINDOW_LINES - 1
        bounded_start = min(start, len(lines))
        bounded_end = min(end, len(lines))
        if bounded_start > bounded_end:
            continue
        snippets.append("\n".join(lines[bounded_start - 1:bounded_end]))
    if snippets:
        return "\n".join(snippets)
    return file_text


def _read_bounded_text(*, repo_root: Path, safe_path: str) -> str | None:
    try:
        file_path = ensure_path_inside(repo_root / safe_path, repo_root)
    except PathSafetyError:
        return None

    try:
        if file_path.stat().st_size > MAX_OWNER_FILE_BYTES:
            return None
        raw_data = file_path.read_bytes()
    except OSError:
        return None

    if b"\x00" in raw_data:
        return None

    try:
        text = raw_data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text


def _source_owner_role(safe_path: str) -> str:
    path = PurePosixPath(safe_path)
    suffix = path.suffix.casefold()
    name = path.name.casefold()
    if _is_test_path(path):
        return "test"
    if name in CONFIG_NAMES or name.startswith("requirements"):
        return "config"
    if _is_documentation_path(path) or suffix in DOC_SUFFIXES:
        return "docs"
    if suffix in PYTHON_SUFFIXES:
        return "runtime"
    return "support"


def _is_documentation_path(path: PurePosixPath) -> bool:
    doc_parts = {"doc", "docs", "docs_src", "documentation", "example", "examples"}
    return any(part.casefold() in doc_parts for part in path.parts)


def _is_test_path(path: PurePosixPath) -> bool:
    name = path.name.casefold()
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    has_test_part = any(
        part.casefold() in ("test", "tests")
        for part in path.parts
    )
    return has_test_part


def _is_python_path(safe_path: str) -> bool:
    suffix = PurePosixPath(safe_path).suffix.casefold()
    return suffix in PYTHON_SUFFIXES


def _defined_symbols(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    symbols: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        if node.name in symbols:
            continue
        symbols.append(node.name)
        if len(symbols) >= MAX_OWNER_SYMBOLS:
            break
    return symbols


def _feature_markers(*, safe_path: str, role: str, text: str) -> list[str]:
    markers: list[str] = []
    if _is_python_path(safe_path):
        markers.append("python_source")
    if role == "test":
        markers.append("test_source")
    if role == "docs":
        markers.append("documentation")
    if role == "config":
        markers.append("configuration_surface")

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        _append_line_markers(markers, stripped)
    return markers


def _exception_types(text: str) -> list[str]:
    exception_types: list[str] = []
    for pattern in (EXCEPTION_TYPE_RE, PYTEST_RAISES_RE):
        for match in pattern.finditer(text):
            exception_type = match.group(1)
            if exception_type in ("Exception", "BaseException"):
                continue
            if exception_type in exception_types:
                continue
            exception_types.append(exception_type)
            if len(exception_types) >= MAX_OWNER_SYMBOLS:
                return exception_types
    return exception_types


def _append_line_markers(markers: list[str], stripped: str) -> None:
    marker_checks = (
        ("raises_errors", stripped.startswith("raise ")),
        ("handles_exceptions", stripped.startswith("except ")),
        ("asserts_invariants", stripped.startswith("assert ")),
        ("decision_branch", stripped.startswith(("if ", "elif ", "case "))),
        (
            "context_manager",
            stripped.startswith(("with ", "async with ")),
        ),
        ("pytest_raises", "pytest.raises" in stripped),
        ("test_assertion", stripped.startswith("assert ")),
    )
    for marker, should_add in marker_checks:
        if not should_add or marker in markers:
            continue
        markers.append(marker)


def _candidate_reasons(*, role: str, feature_markers: list[str]) -> list[str]:
    reasons: list[str] = []
    if role == "runtime":
        reasons.append("runtime source candidate")
    if role == "test":
        reasons.append("test source candidate")
    if role == "docs":
        reasons.append("documentation candidate")
    if role == "config":
        reasons.append("configuration candidate")
    if "raises_errors" in feature_markers:
        reasons.append("contains runtime error branch")
    if "handles_exceptions" in feature_markers:
        reasons.append("contains exception handler")
    if "pytest_raises" in feature_markers:
        reasons.append("contains exception test pattern")
    if not reasons:
        reasons.append("supporting source candidate")
    return reasons


def _candidate_rank(candidate: SourceOwnerCandidate) -> tuple[int, int, str, int]:
    role_order = {
        "runtime": 0,
        "config": 1,
        "test": 2,
        "docs": 3,
        "support": 4,
    }
    marker_score = 0
    feature_markers = candidate["feature_markers"]
    for marker in (
        "raises_errors",
        "handles_exceptions",
        "asserts_invariants",
        "pytest_raises",
        "decision_branch",
    ):
        if marker in feature_markers:
            marker_score -= 1
    rank = (
        role_order.get(candidate["role"], 9),
        marker_score,
        candidate["path"].casefold(),
        candidate["line_start"],
    )
    return rank


def _evidence_ref(row: CodeEvidenceRow) -> str:
    ref = f"{row['path']}:{row['line_start']}-{row['line_end']}"
    return ref
