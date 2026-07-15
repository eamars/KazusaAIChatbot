"""Tests for consolidator character self-image rolling state."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.consolidation import (
    images as image_module,
)

STORAGE_TIMESTAMP_UTC = "2026-05-19T00:00:00+00:00"


def _state() -> dict[str, Any]:
    """Build the minimal state consumed by the character-image updater."""

    state: dict[str, Any] = {
        "mood": "steady",
        "vibe_check": "quiet",
        "character_reflection": "Kazusa noticed a stable self-image shift.",
        "character_profile": {"name": "Kazusa"},
    }
    return state


def _session_response(summary: str) -> SimpleNamespace:
    """Build a fake LangChain response with JSON content."""

    response = SimpleNamespace(
        content=f'{{"session_summary": "{summary}"}}',
    )
    return response


def _session_llm(summary: str) -> SimpleNamespace:
    """Build a fake image-session LLM object."""

    llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=_session_response(summary)),
    )
    return llm


@pytest.mark.asyncio
async def test_update_character_image_uses_explicit_existing_image(
    monkeypatch,
) -> None:
    """A projected character profile must not reset existing image state."""

    existing_image = {
        "milestones": [{"summary": "keeps long-term stance"}],
        "recent_window": [
            {
                "timestamp": "2026-05-18T00:00:00+00:00",
                "summary": "previous session",
            }
        ],
        "historical_summary": "older history",
        "meta": {"synthesis_count": 2},
    }
    monkeypatch.setattr(
        image_module,
        "_character_image_session_summary_llm",
        _session_llm("new session"),
    )

    result = await image_module._update_character_image(
        _state(),
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        existing_image=existing_image,
    )

    assert result == {
        "milestones": [{"summary": "keeps long-term stance"}],
        "recent_window": [
            {
                "timestamp": "2026-05-18T00:00:00+00:00",
                "summary": "previous session",
            },
            {
                "timestamp": STORAGE_TIMESTAMP_UTC,
                "summary": "new session",
            },
        ],
        "historical_summary": "older history",
        "meta": {
            "synthesis_count": 3,
            "last_updated": STORAGE_TIMESTAMP_UTC,
        },
    }


@pytest.mark.asyncio
async def test_update_character_image_rolls_oldest_recent_into_history(
    monkeypatch,
) -> None:
    """A seventh session moves the oldest recent item into history."""

    recent_window = [
        {
            "timestamp": f"2026-05-18T0{index}:00:00+00:00",
            "summary": f"summary-{index}",
        }
        for index in range(6)
    ]
    existing_image = {
        "milestones": [],
        "recent_window": recent_window,
        "historical_summary": "old history",
        "meta": {"synthesis_count": 6},
    }
    monkeypatch.setattr(
        image_module,
        "_character_image_session_summary_llm",
        _session_llm("new session"),
    )

    result = await image_module._update_character_image(
        _state(),
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        existing_image=existing_image,
    )

    assert result["historical_summary"] == "old history\nsummary-0"
    assert result["recent_window"] == [
        *recent_window[1:],
        {
            "timestamp": STORAGE_TIMESTAMP_UTC,
            "summary": "new session",
        },
    ]
    assert result["meta"] == {
        "synthesis_count": 7,
        "last_updated": STORAGE_TIMESTAMP_UTC,
    }
