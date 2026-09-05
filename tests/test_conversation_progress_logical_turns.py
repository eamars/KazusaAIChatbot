"""Logical-turn and participant-lane regression contracts."""

from __future__ import annotations

from copy import deepcopy

from kazusa_ai_chatbot.conversation_progress import (
    assemble_logical_turns,
    select_recent_logical_turns,
)
from kazusa_ai_chatbot.conversation_progress.history import (
    assemble_logical_turns_with_diagnostics,
    logical_turn_source_refs,
    project_logical_turns_for_prompt,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    AMBIENT_LOGICAL_TURN_LIMIT,
    INTERACTION_LOGICAL_TURN_LIMIT,
    MAX_AMBIENT_PROMPT_CHARS,
)
from tests.fixtures.conversation_progress_v2_asuna_houjing_regression import (
    EXPECTED_AMBIENT_LOGICAL_TURN_COUNT,
    EXPECTED_PARTICIPANT_LOGICAL_TURN_COUNT,
    TRACE_5,
    USER_A_GLOBAL_USER_ID,
    build_adjacent_history,
)




def _participant_rows() -> list[dict]:
    """Apply the exact DB query semantics to the source-faithful fixture."""

    return [
        row
        for row in build_adjacent_history()
        if (
            row['role'] == 'user'
            and row['global_user_id'] == USER_A_GLOBAL_USER_ID
        ) or (
            row['role'] == 'assistant'
            and USER_A_GLOBAL_USER_ID
            in row['addressed_to_global_user_ids']
        )
    ]


def test_segmented_assistant_rows_form_one_complete_logical_turn():
    turns = assemble_logical_turns(
        rows=build_adjacent_history(),
        excluded_row_ids=[],
    )
    selected = [turn for turn in turns if turn['llm_trace_id'] == TRACE_5]
    assert len(selected) == 1
    assert len(selected[0]['fragments']) == 7
    assert selected[0]['turn_id'] == f'trace:{TRACE_5}'


def test_grouped_assistant_lineage_uses_its_exact_trace_reference():
    """Avoid inventing per-row timestamps from one grouped turn timestamp."""

    turns = assemble_logical_turns(
        rows=build_adjacent_history(),
        excluded_row_ids=[],
    )
    grouped_turn = next(
        turn for turn in turns if turn['llm_trace_id'] == TRACE_5
    )

    assert logical_turn_source_refs([grouped_turn]) == [{
        'ref_kind': 'llm_trace',
        'ref_id': TRACE_5,
        'occurred_at': grouped_turn['occurred_at'],
    }]


def test_user_lineage_uses_its_exact_row_despite_incidental_trace_id():
    """Keep a user storage row authoritative over incidental trace metadata."""

    turns = assemble_logical_turns(
        rows=_participant_rows(),
        excluded_row_ids=[],
    )
    user_turn = next(turn for turn in turns if turn['role'] == 'user')
    row_id = user_turn['conversation_row_ids'][0]

    assert user_turn['llm_trace_id']
    assert logical_turn_source_refs([user_turn]) == [{
        'ref_kind': 'conversation_row',
        'ref_id': row_id,
        'occurred_at': user_turn['occurred_at'],
    }]


def test_ambient_and_participant_counts_are_independent():
    ambient = assemble_logical_turns(
        rows=build_adjacent_history(),
        excluded_row_ids=[],
    )
    participant = assemble_logical_turns(
        rows=_participant_rows(),
        excluded_row_ids=[],
    )
    assert len(ambient) == EXPECTED_AMBIENT_LOGICAL_TURN_COUNT
    assert len(participant) == EXPECTED_PARTICIPANT_LOGICAL_TURN_COUNT


def test_active_current_row_id_is_excluded_by_mongo_identity():
    rows = build_adjacent_history()
    excluded_id = str(rows[-1]['_id'])
    turns = assemble_logical_turns(
        rows=rows,
        excluded_row_ids=[excluded_id],
    )
    assert all(
        excluded_id not in turn['conversation_row_ids']
        for turn in turns
    )


def test_oldest_cut_off_assistant_suffix_is_dropped():
    rows = deepcopy(build_adjacent_history()[1:4])
    result = assemble_logical_turns_with_diagnostics(
        rows=rows,
        excluded_row_ids=[],
    )
    assert result.turns == []
    assert result.incomplete_or_malformed_turn_count == 1


def test_nonboundary_gapped_assistant_candidate_falls_back_per_row():
    rows = deepcopy(build_adjacent_history()[4:10])
    rows[4]['logical_message_index'] = 8
    result = assemble_logical_turns_with_diagnostics(
        rows=rows,
        excluded_row_ids=[],
    )
    assert len(result.turns) == 6
    assert result.incomplete_or_malformed_turn_count == 1
    assert all(
        len(turn['conversation_row_ids']) == 1
        for turn in result.turns[1:]
    )


def test_attachment_only_row_uses_stored_description():
    """Attachment descriptions make authored-empty rows prompt-usable."""

    row = deepcopy(build_adjacent_history()[4])
    row['body_text'] = ''
    row['attachments'] = [{
        'media_type': 'image/jpeg',
        'description': 'a desk with handwritten notes',
        'storage_shape': 'url_only',
    }]

    result = assemble_logical_turns_with_diagnostics(
        rows=[row],
        excluded_row_ids=[],
    )

    assert result.turns[0]['fragments'] == [
        '<image>a desk with handwritten notes</image>',
    ]
    assert result.incomplete_or_malformed_turn_count == 0


def test_prompt_empty_history_row_is_dropped_with_diagnostic():
    """A legacy row with no usable text must not poison later history."""

    history = build_adjacent_history()
    rows = deepcopy([history[4], history[14], history[15]])
    rows[1]['body_text'] = ''
    rows[1]['attachments'] = []

    result = assemble_logical_turns_with_diagnostics(
        rows=rows,
        excluded_row_ids=[],
    )

    assert len(result.turns) == 2
    assert all(
        rows[1]['_id'] not in turn['conversation_row_ids']
        for turn in result.turns
    )
    assert result.incomplete_or_malformed_turn_count == 1


def test_newest_complete_turn_caps_preserve_chronology():
    turns = assemble_logical_turns(
        rows=build_adjacent_history(),
        excluded_row_ids=[],
    )
    ambient = select_recent_logical_turns(
        turns,
        limit=AMBIENT_LOGICAL_TURN_LIMIT,
    )
    participant = select_recent_logical_turns(
        turns,
        limit=INTERACTION_LOGICAL_TURN_LIMIT,
    )
    assert len(ambient) == AMBIENT_LOGICAL_TURN_LIMIT
    assert len(participant) == INTERACTION_LOGICAL_TURN_LIMIT
    assert ambient == turns[-AMBIENT_LOGICAL_TURN_LIMIT:]


def test_ambient_prompt_projection_is_bounded_and_source_safe():
    turns = assemble_logical_turns(
        rows=build_adjacent_history(),
        excluded_row_ids=[],
    )
    lines = project_logical_turns_for_prompt(
        turns,
        maximum_chars=MAX_AMBIENT_PROMPT_CHARS,
    )
    assert len('\n'.join(lines)) <= MAX_AMBIENT_PROMPT_CHARS
    assert all('row_' not in line and 'trace_' not in line for line in lines)
