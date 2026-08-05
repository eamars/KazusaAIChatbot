"""Scene/evidence split, priority, handles, and continuation budgets."""

from __future__ import annotations

from copy import deepcopy

from kazusa_ai_chatbot.conversation_progress.policy import (
    MAX_CONTINUATION_CHARS,
    MAX_PROGRESS_EVIDENCE_CHARS,
    MAX_PROGRESS_EVIDENCE_ROWS,
    MAX_PROGRESS_SCENE_CHARS,
)
from kazusa_ai_chatbot.conversation_progress.projection import (
    build_progress_prompt,
    project_conversation_progress_evidence,
    project_conversation_progress_scene,
)
from tests.conversation_progress_v2_helpers import (
    NOW,
    event,
    logical_turn,
    packet,
)


def _progress():
    events = [
        event(
            event_id='completed_action',
            summary='user completed the previously selected action',
            state='completed',
            retention='decision_critical',
        ),
        event(
            event_id='open_next_step',
            summary='the next interaction choice remains open',
            state='open',
            retention='active_scene',
        ),
        event(
            event_id='background_note',
            summary='low-priority atmosphere note',
            state='completed',
            retention='background',
        ),
    ]
    active = packet(events=events)
    active['episode_narrative'] = 'n' * 900
    active['current_thread'] = 'advance after the completed action'
    turns = [
        {
            **logical_turn(
                turn_id=f'row:row_{index}',
                row_id=f'row_{index}',
            ),
            'fragments': ['t' * 600],
        }
        for index in range(10)
    ]
    return build_progress_prompt(
        active_packet=active,
        interaction_logical_turns=turns,
    )


def test_scene_uses_interaction_turns_and_respects_hard_cap():
    scene = project_conversation_progress_scene(_progress())
    assert 'advance after the completed action' in scene
    assert 'Recent interaction' in scene
    assert len(scene) <= MAX_PROGRESS_SCENE_CHARS


def test_decision_critical_completed_event_is_first_citeable_evidence():
    evidence = project_conversation_progress_evidence(_progress(), NOW)
    assert evidence[0]['evidence_ref']['source_id'].endswith(
        'completed_action'
    )
    assert 'state=completed' in evidence[0]['semantic_text']
    assert evidence[0]['evidence_ref']['source_kind'] == (
        'conversation_evidence'
    )


def test_evidence_occurred_at_uses_the_event_own_timestamp():
    evidence = project_conversation_progress_evidence(_progress(), NOW)

    assert all(
        row['evidence_ref']['occurred_at'] == '2026-07-28T09:30:00Z'
        for row in evidence
    )


def test_evidence_occurred_at_is_not_the_episode_timestamp():
    progress = _progress()
    progress['events'][0]['updated_at'] = '2026-07-27T08:15:30+00:00'

    evidence = project_conversation_progress_evidence(progress, NOW)

    assert evidence[0]['evidence_ref']['occurred_at'] == (
        '2026-07-27T08:15:30Z'
    )
    assert evidence[0]['evidence_ref']['occurred_at'] != (
        '2026-07-28T09:30:00Z'
    )


def test_event_evidence_respects_row_and_character_caps():
    progress = _progress()
    for index in range(20):
        row = deepcopy(progress['events'][1])
        row['event_id'] = f'event_{index:02d}'
        row['semantic_summary'] = 'x' * 220
        progress['events'].append(row)
    evidence = project_conversation_progress_evidence(progress, NOW)
    assert len(evidence) <= MAX_PROGRESS_EVIDENCE_ROWS
    assert sum(len(row['semantic_text']) for row in evidence) <= (
        MAX_PROGRESS_EVIDENCE_CHARS
    )


def test_evidence_pressure_preserves_every_selected_event_identity():
    """Keep all eight selected event identities visible under worst-case text."""

    active_events = []
    for index in range(MAX_PROGRESS_EVIDENCE_ROWS):
        row = event(
            event_id=f'critical_{index}',
            summary=f'event {index} ' + ('s' * 210),
            state='completed',
            retention='decision_critical',
        )
        row['actor'] = f'actor {index} ' + ('a' * 150)
        row['action'] = f'action {index} ' + ('b' * 149)
        row['object'] = f'object {index} ' + ('c' * 149)
        row['beneficiary'] = 'd' * 160
        row['precondition'] = 'e' * 160
        row['outcome'] = 'f' * 180
        active_events.append(row)
    progress = build_progress_prompt(
        active_packet=packet(events=active_events),
        interaction_logical_turns=[],
    )

    evidence = project_conversation_progress_evidence(progress, NOW)

    assert len(evidence) == MAX_PROGRESS_EVIDENCE_ROWS
    assert sum(len(row['semantic_text']) for row in evidence) <= (
        MAX_PROGRESS_EVIDENCE_CHARS
    )
    for index, evidence_row in enumerate(evidence):
        semantic_text = evidence_row['semantic_text']
        assert f'event {index}' in semantic_text
        assert 'state=completed' in semantic_text
        assert f'actor=actor {index}' in semantic_text
        assert f'action=action {index}' in semantic_text
        assert f'object=object {index}' in semantic_text


def test_continuation_projection_respects_combined_budget():
    progress = _progress()
    scene = project_conversation_progress_scene(progress)
    evidence = project_conversation_progress_evidence(progress, NOW)
    evidence_chars = sum(len(row['semantic_text']) for row in evidence)
    assert len(scene) + evidence_chars <= MAX_CONTINUATION_CHARS


def test_equal_priority_tie_uses_stable_event_id_order():
    progress = _progress()
    progress['events'] = [
        event(
            event_id='event_b',
            state='completed',
            retention='decision_critical',
        ),
        event(
            event_id='event_a',
            state='completed',
            retention='decision_critical',
        ),
    ]
    evidence = project_conversation_progress_evidence(progress, NOW)
    assert evidence[0]['evidence_ref']['source_id'].endswith('event_a')
