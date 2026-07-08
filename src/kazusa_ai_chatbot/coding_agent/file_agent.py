"""Shared file planning for coding-agent workflows."""

from __future__ import annotations

import re
from pathlib import Path
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from kazusa_ai_chatbot.coding_agent.code_fetching.github import (
    is_safe_repo_relative_path,
)
from kazusa_ai_chatbot.coding_agent.code_reading.planner import (
    is_binary_like_path,
    is_secret_like_path,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import ensure_path_inside

if TYPE_CHECKING:
    from kazusa_ai_chatbot.coding_agent.code_writing.models import (
        ArtifactReservationResult,
        ReservedArtifactPath,
        WritingArtifactContract,
        WritingContentFormat,
        WritingFileKind,
    )

MAX_FILE_AGENT_ERRORS = 12
MAX_FILE_NAME_CHARS = 80
MAX_EXISTING_CONTEXT_FILES = 10
MAX_EXISTING_CONTEXT_FILE_CHARS = 16000
MAX_OWNER_CANDIDATES = 12
DEFAULT_SOURCE_DIR = "src"
DEFAULT_TEST_DIR = "tests"
DEFAULT_DOCS_DIR = "docs"
DEFAULT_DATA_DIR = "data"
FILE_NAME_CLEANUP_RE = re.compile(r"[^A-Za-z0-9_.-]+")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
COMMON_NAME_TOKENS = {
    "and",
    "artifact",
    "behavior",
    "code",
    "config",
    "data",
    "file",
    "for",
    "from",
    "input",
    "new",
    "output",
    "source",
    "test",
    "the",
    "this",
    "to",
}
SOURCE_FILE_SUFFIXES = (".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs")
DOC_FILE_SUFFIXES = (".md", ".rst", ".txt")


def plan_existing_source_files(
    *,
    question: str,
    repository: dict[str, object],
    source_scope: dict[str, object],
    reading_result: dict[str, object],
    max_context_files: int = MAX_EXISTING_CONTEXT_FILES,
    max_context_file_chars: int = MAX_EXISTING_CONTEXT_FILE_CHARS,
) -> dict[str, object]:
    """Plan bounded existing-source context from source-reading evidence."""

    local_root_text = repository.get("local_root")
    if not isinstance(local_root_text, str) or not local_root_text.strip():
        plan = _existing_source_failure(
            "Modification requires a resolved local source root."
        )
        return plan

    evidence_rows = reading_result.get("evidence")
    if not isinstance(evidence_rows, list) or not evidence_rows:
        plan = _existing_source_failure(
            "Modification requires at least one source evidence row."
        )
        return plan

    repo_root = Path(local_root_text).expanduser().resolve(strict=True)
    evidence = _existing_evidence_with_ids(evidence_rows)
    contexts, rejected_paths = _existing_file_contexts(
        repo_root=repo_root,
        evidence=evidence,
        max_context_files=max_context_files,
        max_context_file_chars=max_context_file_chars,
    )
    ranked_owners = _ranked_source_owner_candidates(
        question=question,
        contexts=contexts,
        evidence=evidence,
    )
    test_or_doc_paths = [
        context["path"]
        for context in contexts
        if context["role"] == "test_or_doc"
    ]
    caller_paths = [
        context["path"]
        for context in contexts
        if context["role"] == "caller"
    ]
    owner_paths = [
        candidate["path"]
        for candidate in ranked_owners[:MAX_OWNER_CANDIDATES]
    ]
    status = "accepted"
    missing_owner_signals: list[str] = []
    if not contexts:
        status = "rejected"
        missing_owner_signals.append("No safe text file contexts were available.")
    elif not owner_paths:
        status = "repair_required"
        missing_owner_signals.append(
            "Evidence contains only test, documentation, or caller context."
        )

    plan = {
        "status": status,
        "source_scope": source_scope,
        "evidence": evidence,
        "file_contexts": contexts,
        "ranked_source_owner_candidates": ranked_owners[:MAX_OWNER_CANDIDATES],
        "owned_path_candidates": owner_paths,
        "read_only_path_candidates": [],
        "caller_path_candidates": caller_paths,
        "test_or_doc_path_candidates": test_or_doc_paths,
        "missing_owner_signals": missing_owner_signals,
        "rejected_paths": rejected_paths,
        "limits": {
            "max_context_files": max_context_files,
            "max_context_file_chars": max_context_file_chars,
        },
    }
    return plan


def reserve_new_artifact_paths(
    artifact_contracts: list[WritingArtifactContract],
) -> ArtifactReservationResult:
    """Reserve safe repo-relative paths for PM-proposed new artifacts.

    Args:
        artifact_contracts: New-artifact contracts accepted for writing.

    Returns:
        Accepted reservations or compact repair feedback. This helper owns
        file mechanics only; artifact meaning stays with the writing PM.
    """

    errors: list[str] = []
    reservations: list[ReservedArtifactPath] = []
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()

    if not artifact_contracts:
        errors.append("PM decision did not provide artifact contracts.")

    for index, artifact in enumerate(artifact_contracts, start=1):
        reservation, artifact_errors = _reserve_one_artifact(
            artifact=artifact,
            index=index,
        )
        errors.extend(artifact_errors)
        if reservation is None:
            continue
        artifact_id = reservation["artifact_id"]
        path = reservation["path"]
        if artifact_id in seen_ids:
            errors.append(f"Artifact id {artifact_id!r} is duplicated.")
            continue
        if path in seen_paths:
            errors.append(f"Artifact path {path!r} is duplicated.")
            continue
        seen_ids.add(artifact_id)
        seen_paths.add(path)
        reservations.append(reservation)

    limited_errors = errors[:MAX_FILE_AGENT_ERRORS]
    if limited_errors:
        result: ArtifactReservationResult = {
            "status": "repair_required",
            "reserved_paths": [],
            "errors": limited_errors,
            "repair_feedback": _repair_feedback(limited_errors),
        }
        return result

    result = {
        "status": "accepted",
        "reserved_paths": reservations,
        "errors": [],
        "repair_feedback": [],
    }
    return result


def _existing_source_failure(limitation: str) -> dict[str, object]:
    plan = {
        "status": "rejected",
        "source_scope": {},
        "evidence": [],
        "file_contexts": [],
        "ranked_source_owner_candidates": [],
        "owned_path_candidates": [],
        "read_only_path_candidates": [],
        "caller_path_candidates": [],
        "test_or_doc_path_candidates": [],
        "missing_owner_signals": [limitation],
        "rejected_paths": [],
        "limits": {
            "max_context_files": MAX_EXISTING_CONTEXT_FILES,
            "max_context_file_chars": MAX_EXISTING_CONTEXT_FILE_CHARS,
        },
    }
    return plan


def _existing_evidence_with_ids(
    evidence_rows: list[object],
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for index, row in enumerate(evidence_rows, start=1):
        if not isinstance(row, dict):
            continue
        evidence_row = dict(row)
        evidence_id = evidence_row.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            evidence_row["evidence_id"] = f"evidence-{index}"
        evidence.append(evidence_row)
    return evidence


def _existing_file_contexts(
    *,
    repo_root: Path,
    evidence: list[dict[str, object]],
    max_context_files: int,
    max_context_file_chars: int,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    contexts: list[dict[str, object]] = []
    rejected_paths: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    evidence_ids_by_path = _evidence_ids_by_path(evidence)
    for row in evidence:
        path_value = row.get("path")
        if not isinstance(path_value, str):
            continue
        safe_path = _safe_path(path_value)
        if safe_path is None:
            rejected_paths.append({
                "path": path_value,
                "reason": "unsafe_or_filtered_path",
            })
            continue
        if safe_path in seen_paths:
            continue
        if len(contexts) >= max_context_files:
            rejected_paths.append({
                "path": safe_path,
                "reason": "context_file_limit_reached",
            })
            continue
        file_path = ensure_path_inside(repo_root / safe_path, repo_root)
        if not file_path.is_file():
            rejected_paths.append({
                "path": safe_path,
                "reason": "not_a_file",
            })
            continue
        content = file_path.read_text(encoding="utf-8", errors="replace")
        context = {
            "path": safe_path,
            "role": _existing_context_role(safe_path),
            "content": content[:max_context_file_chars],
            "truncated": len(content) > max_context_file_chars,
            "evidence_ids": evidence_ids_by_path.get(safe_path, []),
        }
        contexts.append(context)
        seen_paths.add(safe_path)
    return contexts, rejected_paths


def _evidence_ids_by_path(
    evidence: list[dict[str, object]],
) -> dict[str, list[str]]:
    ids_by_path: dict[str, list[str]] = {}
    for row in evidence:
        path_value = row.get("path")
        evidence_id = row.get("evidence_id")
        if not isinstance(path_value, str) or not isinstance(evidence_id, str):
            continue
        safe_path = _safe_path(path_value)
        if safe_path is None:
            continue
        ids_by_path.setdefault(safe_path, []).append(evidence_id)
    return ids_by_path


def _existing_context_role(path: str) -> str:
    lowered_path = path.casefold()
    if _is_test_or_doc_existing_path(lowered_path):
        return "test_or_doc"
    if _is_caller_like_path(lowered_path):
        return "caller"
    return "source_owner"


def _is_test_or_doc_existing_path(lowered_path: str) -> bool:
    if lowered_path.startswith("tests/") or "/tests/" in lowered_path:
        return True
    return lowered_path.endswith(DOC_FILE_SUFFIXES)


def _is_caller_like_path(lowered_path: str) -> bool:
    path = PurePosixPath(lowered_path)
    stem = path.stem
    return stem in {"cli", "main", "app", "api", "routes", "views"}


def _ranked_source_owner_candidates(
    *,
    question: str,
    contexts: list[dict[str, object]],
    evidence: list[dict[str, object]],
) -> list[dict[str, object]]:
    evidence_summary_by_path = _evidence_summary_by_path(evidence)
    companion_paths = [
        context["path"]
        for context in contexts
        if context.get("role") == "test_or_doc"
    ]
    question_tokens = set(_word_parts(question))
    candidates: list[dict[str, object]] = []
    for context in contexts:
        path_value = context.get("path")
        if not isinstance(path_value, str):
            continue
        if context.get("role") == "test_or_doc":
            continue
        owner_score = _owner_score(
            path=path_value,
            question_tokens=question_tokens,
            role=str(context.get("role", "")),
        )
        reason = _owner_reason(
            path=path_value,
            role=str(context.get("role", "")),
            evidence_summary=evidence_summary_by_path.get(path_value, ""),
        )
        evidence_ids = context.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        candidate = {
            "path": path_value,
            "owner_score": owner_score,
            "owner_reason": reason,
            "evidence_ids": evidence_ids,
            "companion_paths": companion_paths,
        }
        candidates.append(candidate)
    candidates.sort(key=_owner_candidate_sort_key)
    return candidates


def _evidence_summary_by_path(
    evidence: list[dict[str, object]],
) -> dict[str, str]:
    summary_by_path: dict[str, str] = {}
    for row in evidence:
        path_value = row.get("path")
        summary_value = row.get("summary")
        if not isinstance(path_value, str) or not isinstance(summary_value, str):
            continue
        safe_path = _safe_path(path_value)
        if safe_path is None:
            continue
        summary_by_path[safe_path] = summary_value
    return summary_by_path


def _owner_score(
    *,
    path: str,
    question_tokens: set[str],
    role: str,
) -> int:
    lowered_path = path.casefold()
    score = 20
    if role == "source_owner":
        score += 30
    if role == "caller":
        score += 15
    if lowered_path.endswith(SOURCE_FILE_SUFFIXES):
        score += 20
    path_tokens = set(_word_parts(path.replace("/", "_").replace(".", "_")))
    score += min(len(question_tokens.intersection(path_tokens)) * 8, 32)
    if any(token in path_tokens for token in ("model", "models", "store")):
        score += 8
    if any(token in path_tokens for token in ("converter", "scanner", "fetch")):
        score += 8
    return score


def _owner_reason(
    *,
    path: str,
    role: str,
    evidence_summary: str,
) -> str:
    if evidence_summary:
        reason = f"{path} is ranked as {role}: {evidence_summary}"
        return reason
    reason = f"{path} is ranked as {role} from safe source evidence."
    return reason


def _owner_candidate_sort_key(candidate: dict[str, object]) -> tuple[int, str]:
    score = candidate.get("owner_score")
    if not isinstance(score, int):
        score = 0
    path = candidate.get("path")
    if not isinstance(path, str):
        path = ""
    return -score, path


def _reserve_one_artifact(
    *,
    artifact: WritingArtifactContract,
    index: int,
) -> tuple[ReservedArtifactPath | None, list[str]]:
    artifact_id = _bounded_text(artifact.get("artifact_id")) or f"artifact-{index}"
    file_label = _bounded_text(artifact.get("file_label")) or artifact_id
    file_kind = _file_kind(artifact.get("file_kind"))
    content_format = _content_format(artifact.get("content_format"), file_kind)
    path = _safe_file_path(
        preferred_name=artifact.get("preferred_name"),
        file_label=file_label,
        artifact_id=artifact_id,
        file_kind=file_kind,
        content_format=content_format,
    )
    if path is None:
        return None, [f"Artifact {artifact_id!r} does not have a safe filename."]

    reservation: ReservedArtifactPath = {
        "artifact_id": artifact_id,
        "file_label": file_label,
        "path": path,
        "file_kind": file_kind,
        "content_format": content_format,
        "purpose": _bounded_text(artifact.get("purpose")),
    }
    return reservation, []


def _safe_file_path(
    *,
    preferred_name: object,
    file_label: str,
    artifact_id: str,
    file_kind: WritingFileKind,
    content_format: WritingContentFormat,
) -> str | None:
    safe_name = _safe_file_name(preferred_name)
    if safe_name is None:
        safe_name = _generated_file_name(
            file_label=file_label,
            artifact_id=artifact_id,
            file_kind=file_kind,
            content_format=content_format,
        )
    if safe_name is None:
        return None

    directory = _directory_for_file_kind(file_kind)
    path = PurePosixPath(directory) / safe_name
    safe_path = _safe_path(path.as_posix())
    return safe_path


def _directory_for_file_kind(file_kind: WritingFileKind) -> str:
    if file_kind == "test":
        return DEFAULT_TEST_DIR
    if file_kind == "docs":
        return DEFAULT_DOCS_DIR
    if file_kind == "data":
        return DEFAULT_DATA_DIR
    return DEFAULT_SOURCE_DIR


def _generated_file_name(
    *,
    file_label: str,
    artifact_id: str,
    file_kind: WritingFileKind,
    content_format: WritingContentFormat,
) -> str | None:
    stem_sources = [artifact_id, file_label]
    for source in stem_sources:
        words = _word_parts(source)
        if not words:
            continue
        stem = "_".join(words[:6])
        if file_kind == "test" and not stem.startswith("test_"):
            stem = f"test_{stem}"
        safe_name = _safe_file_name(f"{stem}{_suffix(content_format)}")
        if safe_name is not None:
            return safe_name
    return None


def _suffix(content_format: WritingContentFormat) -> str:
    suffixes = {
        "python": ".py",
        "markdown": ".md",
        "text": ".txt",
        "json": ".json",
        "csv": ".csv",
    }
    return suffixes[content_format]


def _file_kind(value: object) -> WritingFileKind:
    if value in {"source", "test", "docs", "config", "data"}:
        return value
    return "source"


def _content_format(
    value: object,
    file_kind: WritingFileKind,
) -> WritingContentFormat:
    if value in {"python", "markdown", "text", "json", "csv"}:
        return value
    if file_kind == "docs":
        return "markdown"
    if file_kind == "data":
        return "csv"
    if file_kind == "config":
        return "text"
    return "python"


def _safe_file_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.replace("\\", "/").strip().split("/")[-1]
    if not name or name in (".", ".."):
        return None
    safe_name = FILE_NAME_CLEANUP_RE.sub("_", name)[:MAX_FILE_NAME_CHARS]
    if "." not in safe_name:
        safe_name = f"{safe_name}.py"
    if _safe_path(safe_name) is None:
        return None
    return safe_name


def _safe_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        return None
    if not is_safe_repo_relative_path(normalized):
        return None
    path = PurePosixPath(normalized)
    safe_path = path.as_posix().rstrip("/")
    if not safe_path:
        return None
    if path.name.casefold().startswith(".env"):
        return None
    if is_secret_like_path(safe_path) or is_binary_like_path(safe_path):
        return None
    return safe_path


def _word_parts(text: str) -> list[str]:
    parts: list[str] = []
    for token in TOKEN_RE.findall(text.replace("-", "_")):
        for piece in CAMEL_BOUNDARY_RE.sub("_", token).split("_"):
            clean_piece = piece.strip().casefold()
            if not clean_piece or clean_piece in COMMON_NAME_TOKENS:
                continue
            if clean_piece in parts:
                continue
            parts.append(clean_piece)
    return parts


def _bounded_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.strip().split())
    return text


def _repair_feedback(errors: list[str]) -> list[str]:
    feedback = [
        "Return corrected new-artifact contracts for the same writing goal.",
        *errors,
    ]
    return feedback


__all__ = ["plan_existing_source_files", "reserve_new_artifact_paths"]
