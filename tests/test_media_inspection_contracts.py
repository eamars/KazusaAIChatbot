"""Deterministic contracts for the shared image inspection boundary."""

import pytest

from kazusa_ai_chatbot.media_inspection.contracts import (
    MediaInspectionValidationError,
    validate_media_inspection_request,
    validate_media_inspection_result,
)


def test_image_request_and_answered_result_are_valid() -> None:
    """Accept the image-only shared inspector envelope."""

    request = validate_media_inspection_request({
        "schema_version": "media_inspection_request.v1",
        "source": "test",
        "media_kind": "image",
        "content_type": "image/png",
        "base64_data": "aGVsbG8=",
        "question": "What color is most visible?",
        "existing_descriptor": "",
    })
    result = validate_media_inspection_result({
        "schema_version": "media_inspection_result.v1",
        "status": "answered",
        "answer": "Blue is most visible.",
        "evidence_boundary_notes": [],
    })

    assert request["media_kind"] == "image"
    assert result["status"] == "answered"


def test_non_image_media_is_rejected() -> None:
    """Keep the first media-inspection contract image-only."""

    with pytest.raises(MediaInspectionValidationError):
        validate_media_inspection_request({
            "schema_version": "media_inspection_request.v1",
            "source": "test",
            "media_kind": "audio",
            "content_type": "audio/mpeg",
            "base64_data": "aGVsbG8=",
            "question": "What is being said?",
            "existing_descriptor": "",
        })
