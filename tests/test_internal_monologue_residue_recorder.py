"""Recorder validation tests for internal monologue residue."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.internal_monologue_residue import recorder
from kazusa_ai_chatbot.internal_monologue_residue.recorder import (
    record_completed_episode_residue,
    validate_recorder_output,
)


def _validate(text: str, *, row_char_limit: int = 80) -> dict:
    """Validate one recorder output fixture.

    Args:
        text: Candidate residue text.
        row_char_limit: Configured row character limit.

    Returns:
        Validation result dictionary.
    """

    result = validate_recorder_output(
        {"residue_text": text},
        row_char_limit=row_char_limit,
    )
    return result


def test_validate_recorder_output_accepts_empty_no_write() -> None:
    """Empty output is a valid no-write result."""

    result = _validate("")

    assert result["accepted"] is True
    assert result["status"] == "empty"


def test_validate_recorder_output_allows_vague_relation_words() -> None:
    """Vague relation wording must pass through at this stage."""

    result = _validate('我还记得对方突然提到提拉米苏，让我有点期待。')

    assert result["accepted"] is True


def test_validate_recorder_output_rejects_third_person_self_reference() -> None:
    """Recorder residue must not refer to the character as a third person role."""

    result = _validate('角色还记得 Tobacco 的赌约，所以有些防备。')

    assert result["accepted"] is False
    assert result["failure_reason"] == "third_person_self_reference"


def test_validate_recorder_output_rejects_prompt_process_leakage() -> None:
    """Recorder residue must not persist prompt or implementation terms."""

    result = _validate('我会按照 system message 和语义表达层继续处理。')

    assert result["accepted"] is False
    assert result["failure_reason"] == "prompt_process_leakage"


def test_validate_recorder_output_rejects_over_limit_text() -> None:
    """Non-empty residue must stay within the configured row cap."""

    result = _validate('我' * 12, row_char_limit=10)

    assert result["accepted"] is False
    assert result["failure_reason"] == "row_char_limit"


@pytest.mark.asyncio
async def test_record_completed_episode_retries_wrong_schema_output(
    monkeypatch,
) -> None:
    """Wrong-schema recorder output must retry instead of becoming empty."""

    class FakeRecorderLlm:
        """Return predefined recorder responses."""

        def __init__(self) -> None:
            self.calls = 0
            self.outputs = [
                '{"wrong_field": "我其实还有点在意。"}',
                '{"residue_text": "我还记得 Tobacco 用提拉米苏逗我。"}',
            ]

        async def ainvoke(self, _messages):
            output = self.outputs[self.calls]
            self.calls += 1
            response = SimpleNamespace(content=output)
            return response

    fake_llm = FakeRecorderLlm()
    insert_row = AsyncMock()
    monkeypatch.setattr(recorder, "_recorder_llm", fake_llm)
    monkeypatch.setattr(
        recorder.db,
        "insert_internal_monologue_residue_row",
        insert_row,
    )
    monkeypatch.setattr(
        recorder.event_logging,
        "record_llm_stage_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        recorder.event_logging,
        "record_database_operation_event",
        AsyncMock(),
    )
    completed_state = {
        "character_profile": {"name": "Character"},
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "global_user_id": "user-1",
        "user_name": "Tobacco",
        "internal_monologue": '我还记得 Tobacco 用提拉米苏逗我。',
        "logical_stance": "TENTATIVE",
        "character_intent": "BANTAR",
        "emotional_appraisal": "",
        "interaction_subtext": "",
        "social_distance": "",
        "relational_dynamic": "",
        "final_dialog": ["那你先把提拉米苏拿出来。"],
        "cognitive_episode": {
            "episode_id": "episode-1",
            "trigger_source": "user_message",
            "origin_metadata": {},
        },
    }

    result = await record_completed_episode_residue(
        completed_state=completed_state,
        current_timestamp_utc="2026-05-20T00:10:00+00:00",
    )

    assert fake_llm.calls == 2
    assert result["status"] == "written"
    assert result["retry_count"] == 1
    insert_row.assert_awaited_once()
