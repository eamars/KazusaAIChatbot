"""Strict image-only contracts for shared media inspection."""

from __future__ import annotations

from typing import Literal, TypedDict

MEDIA_INSPECTION_REQUEST_VERSION = "media_inspection_request.v1"
MEDIA_INSPECTION_RESULT_VERSION = "media_inspection_result.v1"
ALLOWED_MEDIA_INSPECTION_SOURCES = frozenset((
    "rag3_session_media",
    "complex_external_media",
    "test",
    "live_llm_review",
))
ALLOWED_MEDIA_INSPECTION_STATUSES = frozenset((
    "answered",
    "uncertain",
    "unsupported",
    "invalid_input",
    "failed",
))


class MediaInspectionValidationError(ValueError):
    """Raised when the shared media-inspection contract is invalid."""


class MediaInspectionRequestV1(TypedDict):
    """One trusted image payload and its bounded visual question."""

    schema_version: Literal["media_inspection_request.v1"]
    source: Literal[
        "rag3_session_media",
        "complex_external_media",
        "test",
        "live_llm_review",
    ]
    media_kind: Literal["image"]
    content_type: str
    base64_data: str
    question: str
    existing_descriptor: str


class MediaInspectionResultV1(TypedDict):
    """Prompt-safe visual answer with an explicit confidence boundary."""

    schema_version: Literal["media_inspection_result.v1"]
    status: Literal[
        "answered",
        "uncertain",
        "unsupported",
        "invalid_input",
        "failed",
    ]
    answer: str
    evidence_boundary_notes: list[str]


def validate_media_inspection_request(value: object) -> MediaInspectionRequestV1:
    """Validate the image-only inspector input envelope."""

    data = _mapping(value, "media_inspection_request")
    _version(data, MEDIA_INSPECTION_REQUEST_VERSION)
    _enum(data, "source", ALLOWED_MEDIA_INSPECTION_SOURCES)
    if data.get("media_kind") != "image":
        raise MediaInspectionValidationError("media_kind: expected image")
    content_type = _non_empty_string(data, "content_type")
    if not content_type.lower().startswith("image/"):
        raise MediaInspectionValidationError("content_type: expected image MIME type")
    _non_empty_string(data, "base64_data")
    _non_empty_string(data, "question")
    _string(data, "existing_descriptor")
    result = data
    return result


def validate_media_inspection_result(value: object) -> MediaInspectionResultV1:
    """Validate a bounded prompt-safe visual inspection result."""

    data = _mapping(value, "media_inspection_result")
    _version(data, MEDIA_INSPECTION_RESULT_VERSION)
    _enum(data, "status", ALLOWED_MEDIA_INSPECTION_STATUSES)
    _string(data, "answer")
    notes = data.get("evidence_boundary_notes")
    if not isinstance(notes, list) or any(
        not isinstance(item, str) or not item.strip() for item in notes
    ):
        raise MediaInspectionValidationError(
            "evidence_boundary_notes: expected non-empty string list"
        )
    result = data
    return result


def _mapping(value: object, label: str) -> dict:
    """Return a mapping payload or raise a contract error."""

    if not isinstance(value, dict):
        raise MediaInspectionValidationError(f"{label}: expected object")
    result = value
    return result


def _version(data: dict, expected: str) -> None:
    """Require one exact schema version."""

    if data.get("schema_version") != expected:
        raise MediaInspectionValidationError(
            f"schema_version: expected {expected}"
        )


def _enum(data: dict, field_name: str, allowed: frozenset[str]) -> str:
    """Require one closed-vocabulary string field."""

    value = data.get(field_name)
    if not isinstance(value, str) or value not in allowed:
        raise MediaInspectionValidationError(f"{field_name}: unsupported value")
    result = value
    return result


def _string(data: dict, field_name: str) -> str:
    """Require one string field that may be empty."""

    value = data.get(field_name)
    if not isinstance(value, str):
        raise MediaInspectionValidationError(f"{field_name}: expected string")
    result = value
    return result


def _non_empty_string(data: dict, field_name: str) -> str:
    """Require one non-empty string field."""

    value = _string(data, field_name)
    if not value.strip():
        raise MediaInspectionValidationError(f"{field_name}: expected non-empty string")
    result = value
    return result
