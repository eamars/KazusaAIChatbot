"""Integration-boundary tests for internal monologue residue."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.brain_service import post_turn
from kazusa_ai_chatbot.internal_monologue_residue.recorder import (
    _build_residue_row,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
    CognitionState,
    GlobalPersonaState,
)
from kazusa_ai_chatbot.self_cognition import runner


def _completed_state() -> dict:
    """Build a completed persona-state fixture.

    Returns:
        Completed state containing the fields needed by residue recording.
    """

    return_value = {
        "storage_timestamp_utc": "2026-05-20T00:10:00+00:00",
        "character_profile": {"name": "Character"},
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "global_user_id": "user-1",
        "user_name": "Tobacco",
        "internal_monologue": '我还记得 Tobacco 用提拉米苏逗我。',
        "final_dialog": ["那你先把提拉米苏拿出来。"],
        "logical_stance": "TENTATIVE",
        "character_intent": "BANTAR",
        "action_specs": [],
        "action_results": [],
        "surface_outputs": [],
    }
    return return_value


def test_cognition_state_contract_allows_l2a_residue_context() -> None:
    """Persona graph state exposes residue context for L2a only."""

    assert "internal_monologue_residue_context" in GlobalPersonaState.__annotations__
    assert "internal_monologue_residue_context" in CognitionState.__annotations__


@pytest.mark.asyncio
async def test_post_turn_records_internal_monologue_residue_in_background() -> None:
    """Post-turn recorder runs after completed chat state exists."""

    recorder = AsyncMock(return_value={"status": "written", "written": True})
    logger = MagicMock()

    await post_turn.run_internal_monologue_residue_record_background(
        _completed_state(),
        record_completed_episode_residue_func=recorder,
        logger=logger,
    )

    recorder.assert_awaited_once()
    call_kwargs = recorder.await_args.kwargs
    assert call_kwargs["current_timestamp_utc"] == "2026-05-20T00:10:00+00:00"
    assert call_kwargs["completed_state"]["internal_monologue"] == (
        '我还记得 Tobacco 用提拉米苏逗我。'
    )


def test_residue_row_uses_configured_character_id_fallback() -> None:
    """Recorder rows use the same stable character identity as runtime loads."""

    row = _build_residue_row(
        completed_state=_completed_state(),
        residue_text='我还记得 Tobacco 用提拉米苏逗我。',
        current_timestamp_utc="2026-05-20T00:10:00+00:00",
        source_kind="chat",
    )

    assert row["character_id"] == CHARACTER_GLOBAL_USER_ID
    assert CHARACTER_GLOBAL_USER_ID in row["scope_key"]


@pytest.mark.asyncio
async def test_self_cognition_load_uses_configured_character_id_fallback(
    monkeypatch,
) -> None:
    """Self-cognition load and post-episode recording share fallback identity."""

    captured_scope = {}

    async def fake_load_residue_context(*, trigger_scope, current_timestamp_utc):
        captured_scope.update(trigger_scope)
        return {
            "internal_monologue_residue_context": '约1分钟前: 我还有点低落。',
            "selected_count": 1,
            "candidate_count": 1,
            "scope_order": ["character_global"],
            "status": "loaded",
        }

    monkeypatch.setattr(
        runner,
        "load_residue_context",
        fake_load_residue_context,
    )
    case = {
        "idle_timestamp_utc": "2026-05-20T00:10:00+00:00",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "group-1",
            "channel_type": "group",
            "user_id": "user-1",
        },
        "character_profile": {"name": "Character"},
    }

    context = await runner._load_residue_context_for_case(case)

    assert context == '约1分钟前: 我还有点低落。'
    assert captured_scope["character_id"] == CHARACTER_GLOBAL_USER_ID
