"""Shared file planning for coding-agent workflows."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from kazusa_ai_chatbot.coding_agent.code_fetching.github import (
    is_safe_repo_relative_path,
)
from kazusa_ai_chatbot.coding_agent.code_reading.planner import (
    is_binary_like_path,
    is_secret_like_path,
)
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    ArtifactReservationResult,
    ReservedArtifactPath,
    WritingArtifactItem,
    WritingContentFormat,
    WritingFileKind,
)

MAX_FILE_AGENT_ERRORS = 12
MAX_FILE_NAME_CHARS = 80
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


def reserve_new_artifact_paths(
    artifact_items: list[WritingArtifactItem],
) -> ArtifactReservationResult:
    """Reserve safe repo-relative paths for PM-proposed new artifacts.

    Args:
        artifact_items: New-artifact contracts from the writing PM.

    Returns:
        Accepted reservations or compact repair feedback. This helper owns
        file mechanics only; artifact meaning stays with the writing PM.
    """

    errors: list[str] = []
    reservations: list[ReservedArtifactPath] = []
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()

    if not artifact_items:
        errors.append("PM decision did not provide artifact items.")

    for index, artifact in enumerate(artifact_items, start=1):
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


def _reserve_one_artifact(
    *,
    artifact: WritingArtifactItem,
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


__all__ = ["reserve_new_artifact_paths"]
