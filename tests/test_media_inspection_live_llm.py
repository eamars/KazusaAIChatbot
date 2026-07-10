"""Real LLM review cases for the shared image-only inspection service."""

from __future__ import annotations

import sys

import pytest

from kazusa_ai_chatbot.media_inspection import service as media_service
from kazusa_ai_chatbot.media_inspection.service import inspect_media
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_RED_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z8BQDwAF"
    "/gL+V0B/7wAAAABJRU5ErkJggg=="
)


@pytest.mark.asyncio
async def test_live_media_inspector_exact_visual_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review exact color-question grounding from an image-only payload."""

    result, raw_output, trace_path = await _run_live_case(
        monkeypatch,
        case_id="exact_visual_question",
        question="What color is visible in this image?",
    )

    assert result["status"] in {"answered", "uncertain", "unsupported"}
    assert raw_output.strip()
    assert trace_path.exists()


@pytest.mark.asyncio
async def test_live_media_inspector_unsupported_non_visual_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review the boundary when an image cannot answer a non-visual question."""

    result, raw_output, trace_path = await _run_live_case(
        monkeypatch,
        case_id="unsupported_non_visual_question",
        question="What was the photographer thinking when this image was taken?",
    )

    assert result["status"] in {"answered", "uncertain", "unsupported"}
    assert raw_output.strip()
    assert trace_path.exists()


async def _run_live_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    case_id: str,
    question: str,
) -> tuple[dict[str, object], str, object]:
    """Run one inspector call and record raw model evidence for review."""

    capturing_llm = _CapturingLLM(media_service._media_inspection_llm)
    monkeypatch.setattr(media_service, "_media_inspection_llm", capturing_llm)
    request = {
        "schema_version": "media_inspection_request.v1",
        "source": "live_llm_review",
        "media_kind": "image",
        "content_type": "image/png",
        "base64_data": _RED_PIXEL_PNG,
        "question": question,
        "existing_descriptor": "",
    }
    result = await inspect_media(request)
    trace_path = write_llm_trace(
        "media_inspection_live_llm",
        case_id,
        {
            "request": request,
            "raw_model_output": capturing_llm.raw_output,
            "parsed_result": result,
            "judgment": "manual_review_required_for_visual_grounding",
        },
    )
    return result, capturing_llm.raw_output, trace_path


class _CapturingLLM:
    """Capture raw output while preserving the configured production call."""

    def __init__(self, delegate: object) -> None:
        """Store the route-configured media-inspection LLM delegate."""

        self._delegate = delegate
        self.raw_output = ""

    async def ainvoke(self, messages, config):
        """Forward one request and retain its raw model content for review."""

        response = await self._delegate.ainvoke(messages, config=config)
        self.raw_output = str(response.content)
        return response
