"""Bounded logical-history and prompt projection policy tests."""

from __future__ import annotations

from kazusa_ai_chatbot.conversation_progress.history import (
    project_logical_turns_for_prompt,
    select_recent_logical_turns,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    AMBIENT_LOGICAL_TURN_LIMIT,
    AMBIENT_ROW_SCAN_LIMIT,
    INTERACTION_LOGICAL_TURN_LIMIT,
    INTERACTION_ROW_SCAN_LIMIT,
    MAX_AMBIENT_PROMPT_CHARS,
    MAX_INTERACTION_RECORDER_CHARS,
    MAX_LOGICAL_TURN_TEXT_CHARS,
)
from tests.conversation_progress_v2_helpers import logical_turn


def _turn(index: int, text: str = 'bounded text') -> dict:
    """Build one source-distinct logical turn."""

    turn = logical_turn(
        turn_id=f'row:row-{index}',
        row_id=f'row-{index}',
        trace_id=f'trace-{index}',
    )
    turn['occurred_at'] = f'2026-07-28T09:{index:02d}:00+00:00'
    turn['fragments'] = [text]
    return turn


def test_history_caps_match_approved_contract():
    assert AMBIENT_ROW_SCAN_LIMIT == 48
    assert INTERACTION_ROW_SCAN_LIMIT == 128
    assert AMBIENT_LOGICAL_TURN_LIMIT == 6
    assert INTERACTION_LOGICAL_TURN_LIMIT == 10
    assert MAX_LOGICAL_TURN_TEXT_CHARS == 600
    assert MAX_AMBIENT_PROMPT_CHARS == 1200
    assert MAX_INTERACTION_RECORDER_CHARS == 2000


def test_newest_ten_interaction_turns_preserve_chronology():
    turns = [_turn(index) for index in range(20)]

    selected = select_recent_logical_turns(
        turns,
        limit=INTERACTION_LOGICAL_TURN_LIMIT,
    )

    assert [turn['turn_id'] for turn in selected] == [
        f'row:row-{index}' for index in range(10, 20)
    ]


def test_prompt_projection_keeps_complete_newest_turns_within_budget():
    turns = [
        _turn(index, text=str(index) * 500)
        for index in range(10)
    ]

    lines = project_logical_turns_for_prompt(
        turns,
        maximum_chars=MAX_AMBIENT_PROMPT_CHARS,
    )

    assert len('\n'.join(lines)) <= MAX_AMBIENT_PROMPT_CHARS
    assert all(len(line) <= MAX_LOGICAL_TURN_TEXT_CHARS for line in lines)
    assert lines[-1].endswith('9' * (len(lines[-1].split(': ', 1)[1])))


def test_prompt_projection_never_exposes_protected_row_or_trace_ids():
    lines = project_logical_turns_for_prompt(
        [_turn(1, text='ordinary content')],
        maximum_chars=MAX_AMBIENT_PROMPT_CHARS,
    )

    rendered = '\n'.join(lines)
    assert 'row-1' not in rendered
    assert 'trace-1' not in rendered
    assert 'ordinary content' in rendered


def test_zero_turn_limit_returns_empty_without_partial_turns():
    assert select_recent_logical_turns(
        [_turn(1)],
        limit=0,
    ) == []
