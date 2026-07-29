"""Deterministic compaction, lineage, survival, and write-order contracts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from kazusa_ai_chatbot.conversation_progress.compaction import (
    ConversationCompactionContractError,
    apply_compaction_to_packet,
    build_compaction_plan,
    create_block_from_plan,
    should_compact,
    validate_block,
    validate_compaction_plan,
)
from kazusa_ai_chatbot.conversation_progress.repository import (
    PreparedProgressWrite,
    persist_progress_write,
)
from tests.conversation_progress_v2_helpers import event, packet


def _plan_and_block():
    """Build one packet, deterministic plan, and immutable block."""

    archived = event(
        event_id='terminal_background',
        summary='A completed background event with exact stored semantics.',
        state='completed',
        retention='background',
    )
    active = packet(
        turn_count=20,
        events=[archived, event(
            event_id='critical_completed',
            state='completed',
            retention='decision_critical',
        )],
        recent_turn_refs=[f'row:row_{index}' for index in range(12)],
    )
    plan = build_compaction_plan(
        active_packet=active,
        active_blocks=[],
    )
    assert plan is not None
    block = create_block_from_plan(
        compaction_plan=plan,
        active_packet=active,
        active_blocks=[],
    )
    return active, plan, block


def test_structural_thresholds_trigger_without_text_matching() -> None:
    """Trigger compaction from numeric packet pressure only."""

    assert should_compact(
        active_event_count=18,
        recent_turn_ref_count=0,
        packet_chars=0,
    )
    assert should_compact(
        active_event_count=0,
        recent_turn_ref_count=12,
        packet_chars=0,
    )
    assert should_compact(
        active_event_count=0,
        recent_turn_ref_count=0,
        packet_chars=10000,
    )


def test_deterministic_compaction_plan_requires_no_model_output() -> None:
    """Select and render archival structure from validated stored labels."""

    active, plan, block = _plan_and_block()
    archived = active['events'][0]

    assert plan['archive_event_ids'] == ['terminal_background']
    assert plan['source_block_ids'] == []
    assert [row['event_id'] for row in block['events']] == [
        'terminal_background'
    ]
    assert archived['semantic_summary'] in block['narrative']


def test_compaction_archives_terminal_background_and_preserves_critical():
    """Remove only the deterministically eligible event from active state."""

    active, plan, block = _plan_and_block()
    validate_block(block)
    result = apply_compaction_to_packet(
        active_packet=active,
        compaction_plan=plan,
        block_id=block['block_id'],
    )

    assert [row['event_id'] for row in result['events']] == [
        'critical_completed'
    ]
    assert [row['event_id'] for row in block['events']] == [
        'terminal_background'
    ]
    assert result['compacted_block_refs'] == [block['block_id']]


def test_compaction_plan_rejects_protected_event_ids() -> None:
    """Keep decision-critical events outside deterministic archival."""

    active, _, _ = _plan_and_block()

    with pytest.raises(
        ConversationCompactionContractError,
        match='protected',
    ):
        validate_compaction_plan(
            {
                'archive_event_ids': ['critical_completed'],
                'covered_turn_refs': [],
                'source_block_ids': [],
            },
            active_packet=active,
            active_blocks=[],
        )


@pytest.mark.asyncio
async def test_block_insert_precedes_guarded_packet_write(monkeypatch):
    """Persist immutable evidence before publishing its active reference."""

    active, plan, block = _plan_and_block()
    packet_after = apply_compaction_to_packet(
        active_packet=active,
        compaction_plan=plan,
        block_id=block['block_id'],
    )
    calls: list[str] = []

    async def embed(_text):
        calls.append('embed')
        return [0.1]

    async def insert(*, document):
        calls.append('insert')
        return True

    async def replace(*, document):
        calls.append('replace')
        return True

    async def touch(**_kwargs):
        calls.append('touch')
        assert _kwargs['block_ids'] == [
            block['block_id'],
            'transitive-child',
        ]
        return 1

    monkeypatch.setattr(
        'kazusa_ai_chatbot.conversation_progress.repository.'
        'insert_conversation_progress_block',
        insert,
    )
    monkeypatch.setattr(
        'kazusa_ai_chatbot.conversation_progress.repository.'
        'replace_episode_state_guarded',
        replace,
    )
    monkeypatch.setattr(
        'kazusa_ai_chatbot.conversation_progress.repository.'
        'touch_conversation_progress_blocks',
        touch,
    )
    result = await persist_progress_write(
        PreparedProgressWrite(
            packet=packet_after,
            block=block,
            source_block_ids=[],
            protected_block_ids=[
                block['block_id'],
                'transitive-child',
            ],
        ),
        embed_block=embed,
    )

    assert result.written is True
    assert calls == ['embed', 'insert', 'replace', 'touch']


@pytest.mark.asyncio
async def test_lost_packet_write_leaves_source_blocks_unsuperseded(
    monkeypatch,
):
    """Leave prior lineage active when the guarded packet write loses."""

    active, _, block = _plan_and_block()
    calls: list[str] = []

    async def embed(_text):
        return [0.1]

    async def insert(**_kwargs):
        calls.append('insert')
        return True

    async def replace(**_kwargs):
        calls.append('replace')
        return False

    async def supersede(**_kwargs):
        calls.append('supersede')
        return 1

    monkeypatch.setattr(
        'kazusa_ai_chatbot.conversation_progress.repository.'
        'insert_conversation_progress_block',
        insert,
    )
    monkeypatch.setattr(
        'kazusa_ai_chatbot.conversation_progress.repository.'
        'replace_episode_state_guarded',
        replace,
    )
    monkeypatch.setattr(
        'kazusa_ai_chatbot.conversation_progress.repository.'
        'supersede_conversation_progress_blocks',
        supersede,
    )
    result = await persist_progress_write(
        PreparedProgressWrite(
            packet=deepcopy(active),
            block=block,
            source_block_ids=['source_block'],
            protected_block_ids=['source_block'],
        ),
        embed_block=embed,
    )

    assert result.disposition == 'lost_guarded_write'
    assert calls == ['insert', 'replace']
