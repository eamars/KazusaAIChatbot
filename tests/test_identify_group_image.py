from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock

import pytest

from scripts import identify_group_image as script


def _group_style_doc() -> dict:
    """Build a representative group style image document."""

    return_value = {
        "style_image_id": "group_channel:qq:1082431481",
        "scope_type": "group_channel",
        "global_user_id": "",
        "platform": "qq",
        "platform_channel_id": "1082431481",
        "status": "active",
        "overlay": {
            "speech_guidelines": ["Keep replies compact."],
            "social_guidelines": ["Reward group participation."],
            "pacing_guidelines": [],
            "engagement_guidelines": [],
            "confidence": "high",
        },
        "source_reflection_run_ids": ["run-1"],
        "revision": 2,
        "created_at": "2026-05-06T00:00:00+00:00",
        "updated_at": "2026-05-07T00:00:00+00:00",
    }
    return return_value


@pytest.mark.asyncio
async def test_main_prints_compact_group_style_image(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default lookup prints the group style image in compact text form."""

    get_style_image = AsyncMock(return_value=_group_style_doc())
    close_db = AsyncMock()
    monkeypatch.setattr(script, "get_group_channel_style_image", get_style_image)
    monkeypatch.setattr(script, "close_db", close_db)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "identify_group_image",
            "1082431481",
            "--platform",
            "qq",
        ],
    )

    await script.main()

    output = capsys.readouterr().out
    assert "style_image_id: group_channel:qq:1082431481" in output
    assert "group_channel_style_image.speech_guidelines:" in output
    assert "1. speech: Keep replies compact." in output
    get_style_image.assert_awaited_once_with(
        platform="qq",
        platform_channel_id="1082431481",
    )
    close_db.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_prints_json_group_style_image(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """JSON lookup prints the stored group style image document."""

    get_style_image = AsyncMock(return_value=_group_style_doc())
    monkeypatch.setattr(script, "get_group_channel_style_image", get_style_image)
    monkeypatch.setattr(script, "close_db", AsyncMock())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "identify_group_image",
            "1082431481",
            "--platform",
            "qq",
            "--json",
        ],
    )

    await script.main()

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["style_image_id"] == "group_channel:qq:1082431481"
    assert payload["overlay"]["confidence"] == "high"


@pytest.mark.asyncio
async def test_main_reports_missing_group_style_image(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing group style image produces an explicit not-found message."""

    monkeypatch.setattr(
        script,
        "get_group_channel_style_image",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(script, "close_db", AsyncMock())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "identify_group_image",
            "missing-channel",
            "--platform",
            "qq",
        ],
    )

    await script.main()

    output = capsys.readouterr().out
    assert "No group channel style image found for qq:missing-channel." in output
