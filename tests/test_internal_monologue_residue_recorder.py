"""Recorder validation tests for internal monologue residue."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.db import DatabaseOperationError
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
        {
            "disposition": "clear_scope" if not text else "append",
            "residue_text": text,
        },
        row_char_limit=row_char_limit,
    )
    return result


def _completed_state() -> dict[str, object]:
    """Build the smallest completed episode accepted by the recorder."""

    return {
        'character_profile': {
            'name': 'Character',
            'global_user_id': 'character-1',
        },
        'platform': 'qq',
        'platform_channel_id': 'group-1',
        'channel_type': 'group',
        'global_user_id': 'user-1',
        'user_name': 'User',
        'internal_monologue': 'I am still considering the current scene.',
        'logical_stance': 'TENTATIVE',
        'character_intent': 'CONSIDER',
        'emotional_appraisal': '',
        'interaction_subtext': '',
        'social_distance': '',
        'relational_dynamic': '',
        'final_dialog': [],
        'cognitive_episode': {
            'episode_id': 'episode-residue-test',
            'trigger_source': 'user_message',
            'origin_metadata': {},
        },
    }


def test_validate_recorder_output_accepts_empty_no_write() -> None:
    """Empty output is a valid no-write result."""

    result = _validate("")

    assert result["accepted"] is True
    assert result["status"] == "empty"
    assert result["disposition"] == "clear_scope"


def test_empty_residue_result_has_cleared_contract() -> None:
    """The empty recorder result is an explicit cleared disposition."""

    result = _validate("")

    assert result == {
        "accepted": True,
        "status": "empty",
        "failure_reason": "",
        "disposition": "clear_scope",
    }


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


def test_recorder_requires_typed_disposition() -> None:
    """Missing, unknown, or expanded disposition objects fail closed."""

    assert validate_recorder_output(
        {'residue_text': 'carry this'},
        row_char_limit=80,
    )['failure_reason'] == 'exact_output_fields_required'
    assert validate_recorder_output(
        {'disposition': 'unknown', 'residue_text': 'carry this'},
        row_char_limit=80,
    )['failure_reason'] == 'invalid_disposition'
    assert validate_recorder_output(
        {
            'disposition': 'append',
            'residue_text': 'carry this',
            'extra': 'not allowed',
        },
        row_char_limit=80,
    )['failure_reason'] == 'exact_output_fields_required'


def test_recorder_disposition_text_invariants_are_fail_closed() -> None:
    """Append/replace require text while clear requires an empty marker."""

    assert validate_recorder_output(
        {'disposition': 'append', 'residue_text': ''},
        row_char_limit=80,
    )['failure_reason'] == 'nonempty_text_required'
    assert validate_recorder_output(
        {'disposition': 'replace_scope', 'residue_text': ' '},
        row_char_limit=80,
    )['failure_reason'] == 'nonempty_text_required'
    assert validate_recorder_output(
        {'disposition': 'clear_scope', 'residue_text': 'carry this'},
        row_char_limit=80,
    )['failure_reason'] == 'clear_scope_requires_empty_text'


def test_build_recorder_input_includes_group_review_reliability_notes() -> None:
    """Ambiguous group-review subject warnings should reach recorder input."""

    completed_state = {
        "character_profile": {
            "name": "杏山千纱",
            "global_user_id": "character-global",
        },
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "global_user_id": "",
        "user_name": "group audience",
        "internal_monologue": (
            '我刚才看到灯说“你的头发软软的”，但这行属于雪凪和灯的侧线。'
        ),
        "logical_stance": "TENTATIVE",
        "character_intent": "DISMISS",
        "emotional_appraisal": "",
        "interaction_subtext": "",
        "social_distance": "",
        "relational_dynamic": "",
        "final_dialog": [],
        "conversation_progress": {
            "thread_reference_context": {
                "source": "group_review_thread_reference",
                "context_shape": (
                    "bounded_second_person_reference_warnings"
                ),
                "guidance": (
                    "二人称归属按同一行明确地址和可见线程读取；"
                    "缺少同一行当前角色指向时，保留为侧线/未定对象。"
                ),
                "ambiguous_second_person_rows": [
                    {
                        "speaker": "灯（23岁）",
                        "sample": (
                            "你的头发软软的，像rana家那只靠在暖气片旁边的猫。"
                        ),
                        "referent_status": "ambiguous_or_side_thread",
                        "basis": "same row has no direct active-character address",
                    },
                ],
            },
        },
        "cognitive_episode": {
            "episode_id": "episode-cat",
            "trigger_source": "internal_thought",
            "origin_metadata": {},
        },
    }

    recorder_input = recorder._build_recorder_input(completed_state)

    assert recorder_input is not None
    assert recorder_input["source_kind"] == "self_cognition"
    assert recorder_input["source_reliability_notes"] == [
        "group review contained ambiguous second-person side-thread rows",
    ]


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
                '{"disposition": "append", "residue_text": "我还记得 Tobacco 用提拉米苏逗我。"}',
            ]

        async def ainvoke(self, _messages, *, config=None):
            output = self.outputs[self.calls]
            self.calls += 1
            response = SimpleNamespace(content=output)
            return response

    fake_llm = FakeRecorderLlm()
    insert_row = AsyncMock(return_value={
        "status": "written",
        "residue_id": "residue-1",
    })
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
        "record_continuity_boundary_event",
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
    assert result["disposition"] == "append"
    insert_row.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_completed_episode_writes_scoped_clear_barrier_for_empty_output(
    monkeypatch,
) -> None:
    """A valid empty recorder result persists a typed clear marker."""

    monkeypatch.setattr(
        recorder,
        '_recorder_llm',
        SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
            content='{"disposition":"clear_scope","residue_text":""}',
        ))),
    )
    insert_row = AsyncMock(return_value={
        'status': 'written',
        'residue_id': 'clear-marker',
    })
    monkeypatch.setattr(
        recorder.db,
        'insert_internal_monologue_residue_row',
        insert_row,
    )
    monkeypatch.setattr(
        recorder.event_logging,
        'record_llm_stage_event',
        AsyncMock(),
    )
    monkeypatch.setattr(
        recorder.event_logging,
        'record_continuity_boundary_event',
        AsyncMock(),
    )

    result = await record_completed_episode_residue(
        completed_state=_completed_state(),
        current_timestamp_utc='2026-05-20T00:10:00+00:00',
    )

    assert result['status'] == 'written'
    assert result['written'] is True
    assert result['disposition'] == 'clear_scope'
    assert result['idempotency_result'] == 'written'
    assert insert_row.await_args.args[0]['disposition'] == 'clear_scope'
    assert insert_row.await_args.args[0]['residue_text'] == ''


@pytest.mark.asyncio
async def test_empty_residue_write_failure_is_not_reported_as_cleared(
    monkeypatch,
) -> None:
    """A failed clear-marker write remains a persistence failure."""

    monkeypatch.setattr(
        recorder,
        '_recorder_llm',
        SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
            content='{"disposition":"clear_scope","residue_text":""}',
        ))),
    )
    monkeypatch.setattr(
        recorder.db,
        'insert_internal_monologue_residue_row',
        AsyncMock(side_effect=DatabaseOperationError('write unavailable')),
    )
    monkeypatch.setattr(
        recorder.event_logging,
        'record_llm_stage_event',
        AsyncMock(),
    )
    monkeypatch.setattr(
        recorder.event_logging,
        'record_continuity_boundary_event',
        AsyncMock(),
    )

    result = await record_completed_episode_residue(
        completed_state=_completed_state(),
        current_timestamp_utc='2026-05-20T00:10:00+00:00',
    )

    assert result['status'] == 'write_failed'
    assert result['written'] is False
    assert result['disposition'] == 'clear_scope'
    assert result['idempotency_result'] == 'not_attempted'


def test_record_completed_episode_operation_id_is_stable() -> None:
    """The same episode and exact scope produce one transition identity."""

    state = _completed_state()
    first = recorder._build_residue_row(
        completed_state=state,
        residue_text='carry this branch',
        current_timestamp_utc='2026-05-20T00:10:00+00:00',
        source_kind='chat',
        disposition='append',
        episode_id='episode-residue-test',
    )
    second = recorder._build_residue_row(
        completed_state=state,
        residue_text='carry this branch',
        current_timestamp_utc='2026-05-20T00:10:00+00:00',
        source_kind='chat',
        disposition='append',
        episode_id='episode-residue-test',
    )

    assert first is not None
    assert second is not None
    assert first['operation_id'] == second['operation_id']
    assert first['residue_id'] != second['residue_id']
