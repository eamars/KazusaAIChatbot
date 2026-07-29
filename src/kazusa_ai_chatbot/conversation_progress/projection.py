"""Bounded scene and cognition-evidence projections."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
    CognitionEvidenceV2,
)
from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationLogicalTurnV1,
    ConversationProgressEventV2,
    ConversationProgressPromptV2,
    ConversationProgressStateV2,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    MAX_CONTINUATION_CHARS,
    MAX_PROGRESS_EVIDENCE_CHARS,
    MAX_PROGRESS_EVIDENCE_ROWS,
    MAX_PROGRESS_SCENE_CHARS,
    MAX_SCENE_LOGICAL_TURNS,
    MAX_SCENE_NARRATIVE_CHARS,
    MAX_SCENE_TURN_TEXT_CHARS,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime


def empty_progress_prompt(
    *,
    interaction_logical_turns: Sequence[ConversationLogicalTurnV1],
) -> ConversationProgressPromptV2:
    """Build the stable prompt shape when no active packet exists."""

    return {
        'schema_version': 'conversation_progress_prompt.v2',
        'episode_state_id': '',
        'status': 'empty',
        'continuity': 'sharp_transition',
        'turn_count': 0,
        'current_thread': '',
        'character_stance': '',
        'user_goal': '',
        'current_blocker': '',
        'emotional_trajectory': '',
        'episode_narrative': '',
        'events': [],
        'overused_moves': [],
        'interaction_logical_turns': [
            deepcopy(turn) for turn in interaction_logical_turns
        ],
        'compacted_block_refs': [],
    }


def build_progress_prompt(
    *,
    active_packet: ConversationProgressStateV2 | None,
    interaction_logical_turns: Sequence[ConversationLogicalTurnV1],
) -> ConversationProgressPromptV2:
    """Project one active packet into bounded prompt-facing continuation."""

    if active_packet is None:
        return empty_progress_prompt(
            interaction_logical_turns=interaction_logical_turns,
        )
    events = _ordered_events(active_packet['events'])[
        :MAX_PROGRESS_EVIDENCE_ROWS
    ]
    return {
        'schema_version': 'conversation_progress_prompt.v2',
        'episode_state_id': active_packet['episode_state_id'],
        'status': active_packet['status'],
        'continuity': active_packet['continuity'],
        'turn_count': active_packet['turn_count'],
        'current_thread': active_packet['current_thread'],
        'character_stance': active_packet['character_stance'],
        'user_goal': active_packet['user_goal'],
        'current_blocker': active_packet['current_blocker'],
        'emotional_trajectory': active_packet['emotional_trajectory'],
        'episode_narrative': active_packet['episode_narrative'],
        'events': [deepcopy(event) for event in events],
        'overused_moves': list(active_packet['overused_moves']),
        'interaction_logical_turns': [
            deepcopy(turn) for turn in interaction_logical_turns
        ],
        'compacted_block_refs': list(
            active_packet['compacted_block_refs']
        ),
    }


def project_conversation_progress_scene(
    progress: ConversationProgressPromptV2,
) -> str:
    """Render the required-first scene projection within 2,200 characters."""

    lines: list[str] = []
    _append_line(
        lines,
        'Current thread',
        progress['current_thread'],
        maximum_chars=MAX_PROGRESS_SCENE_CHARS,
        permit_truncation=True,
    )
    _append_line(
        lines,
        'Character stance',
        progress['character_stance'],
        maximum_chars=MAX_PROGRESS_SCENE_CHARS,
        permit_truncation=True,
    )
    _append_line(
        lines,
        'Episode narrative',
        progress['episode_narrative'][:MAX_SCENE_NARRATIVE_CHARS],
        maximum_chars=MAX_PROGRESS_SCENE_CHARS,
        permit_truncation=True,
    )

    recent_turns = progress['interaction_logical_turns'][
        -MAX_SCENE_LOGICAL_TURNS:
    ]
    for turn in recent_turns:
        speaker = turn['display_name'] or turn['role']
        turn_text = ' '.join(turn['fragments'])
        bounded_text = turn_text[:MAX_SCENE_TURN_TEXT_CHARS].rstrip()
        _append_line(
            lines,
            f'Recent interaction ({speaker})',
            bounded_text,
            maximum_chars=MAX_PROGRESS_SCENE_CHARS,
            permit_truncation=True,
        )

    _append_line(
        lines,
        'User goal',
        progress['user_goal'],
        maximum_chars=MAX_PROGRESS_SCENE_CHARS,
        permit_truncation=False,
    )
    _append_line(
        lines,
        'Current blocker',
        progress['current_blocker'],
        maximum_chars=MAX_PROGRESS_SCENE_CHARS,
        permit_truncation=False,
    )
    _append_line(
        lines,
        'Emotional trajectory',
        progress['emotional_trajectory'],
        maximum_chars=MAX_PROGRESS_SCENE_CHARS,
        permit_truncation=False,
    )
    _append_list_line(
        lines,
        'Overused moves',
        progress['overused_moves'],
        maximum_chars=MAX_PROGRESS_SCENE_CHARS,
    )
    scene = '\n'.join(lines)
    if len(scene) > MAX_PROGRESS_SCENE_CHARS:
        raise ValueError('conversation progress scene exceeds its hard cap')
    return scene


def project_conversation_progress_evidence(
    progress: ConversationProgressPromptV2,
    occurred_at: str,
) -> list[CognitionEvidenceV2]:
    """Project model-labelled events into ordered citeable cognition evidence."""

    if not isinstance(occurred_at, str) or not occurred_at:
        raise ValueError('conversation progress evidence occurred_at is required')
    selected_events = _ordered_events(progress['events'])[
        :MAX_PROGRESS_EVIDENCE_ROWS
    ]
    if not selected_events:
        return []
    base_budget, extra_chars = divmod(
        MAX_PROGRESS_EVIDENCE_CHARS,
        len(selected_events),
    )
    evidence: list[CognitionEvidenceV2] = []
    for index, event in enumerate(selected_events):
        row_budget = base_budget + (1 if index < extra_chars else 0)
        semantic_text = _bounded_event_semantic_text(
            event,
            maximum_chars=row_budget,
        )
        evidence.append({
            'evidence_handle': '',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': f'conversation-progress-event:{event["event_id"]}',
                'occurred_at': occurred_at,
                'semantic_summary': event['semantic_summary'],
            },
            'semantic_text': semantic_text,
            'visible_to': list(
                EVIDENCE_SOURCE_QUESTION_IDS['conversation_evidence']
            ),
        })
    used_chars = sum(len(row['semantic_text']) for row in evidence)
    if used_chars > MAX_PROGRESS_EVIDENCE_CHARS:
        raise ValueError('conversation progress evidence exceeds its hard cap')
    return evidence


def continuation_projection_chars(
    progress: ConversationProgressPromptV2,
    occurred_at: str,
) -> tuple[int, int]:
    """Return scene and evidence character counts after final projection."""

    scene = project_conversation_progress_scene(progress)
    evidence = project_conversation_progress_evidence(progress, occurred_at)
    evidence_chars = sum(len(row['semantic_text']) for row in evidence)
    if len(scene) + evidence_chars > MAX_CONTINUATION_CHARS:
        raise ValueError('combined continuation projection exceeds its hard cap')
    return len(scene), evidence_chars


def _ordered_events(
    events: Sequence[ConversationProgressEventV2],
) -> list[ConversationProgressEventV2]:
    """Apply the approved semantic-label priority with stable tie breaking."""

    return sorted(
        events,
        key=lambda event: (
            _event_tier(event),
            -parse_storage_utc_datetime(event['updated_at']).timestamp(),
            event['event_id'],
        ),
    )


def _event_tier(event: ConversationProgressEventV2) -> int:
    """Map exact model-owned state labels to the approved selection tier."""

    if event['retention'] == 'decision_critical':
        return 0
    if (
        event['retention'] == 'active_scene'
        and event['state'] in {'open', 'in_progress'}
    ):
        return 1
    if event['retention'] == 'active_scene':
        return 2
    return 3


def _event_semantic_text(event: ConversationProgressEventV2) -> str:
    """Render one event snapshot without performing semantic inference."""

    parts = [
        event['semantic_summary'],
        f'state={event["state"]}',
        f'retention={event["retention"]}',
    ]
    for label, field_name in (
        ('actor', 'actor'),
        ('action', 'action'),
        ('object', 'object'),
        ('beneficiary', 'beneficiary'),
        ('precondition', 'precondition'),
        ('outcome', 'outcome'),
    ):
        value = event[field_name]
        if value:
            parts.append(f'{label}={value}')
    return '; '.join(parts)


def _bounded_event_semantic_text(
    event: ConversationProgressEventV2,
    *,
    maximum_chars: int,
) -> str:
    """Fit one event fairly while preserving its decision identity."""

    full_text = _event_semantic_text(event)
    if len(full_text) <= maximum_chars:
        return full_text

    required_values = [
        event['semantic_summary'],
        event['actor'],
        event['action'],
        event['object'],
    ]
    value_limits = [
        min(48, len(required_values[0])),
        min(16, len(required_values[1])),
        min(16, len(required_values[2])),
        min(24, len(required_values[3])),
    ]

    def _render_required() -> str:
        return '; '.join([
            required_values[0][:value_limits[0]],
            f'state={event["state"]}',
            f'retention={event["retention"]}',
            f'actor={required_values[1][:value_limits[1]]}',
            f'action={required_values[2][:value_limits[2]]}',
            f'object={required_values[3][:value_limits[3]]}',
        ])

    semantic_text = _render_required()
    if len(semantic_text) > maximum_chars:
        raise ValueError(
            'conversation progress evidence row cannot preserve identity'
        )

    while len(semantic_text) < maximum_chars:
        expanded = False
        for index, value in enumerate(required_values):
            if value_limits[index] >= len(value):
                continue
            value_limits[index] += 1
            candidate = _render_required()
            if len(candidate) > maximum_chars:
                value_limits[index] -= 1
                continue
            semantic_text = candidate
            expanded = True
        if not expanded:
            break

    for label, field_name in (
        ('outcome', 'outcome'),
        ('beneficiary', 'beneficiary'),
        ('precondition', 'precondition'),
    ):
        value = event[field_name]
        if not value:
            continue
        prefix = f'; {label}='
        remaining_chars = maximum_chars - len(semantic_text)
        if remaining_chars <= len(prefix):
            continue
        bounded_value = value[:remaining_chars - len(prefix)].rstrip()
        if bounded_value:
            semantic_text += prefix + bounded_value
    return semantic_text


def _append_line(
    lines: list[str],
    label: str,
    value: str,
    *,
    maximum_chars: int,
    permit_truncation: bool,
) -> None:
    """Append one labelled line if it fits the aggregate scene budget."""

    if not value:
        return
    prefix = f'{label}: '
    used_chars = len('\n'.join(lines))
    separator_chars = 1 if lines else 0
    remaining = maximum_chars - used_chars - separator_chars
    line = prefix + value
    if len(line) <= remaining:
        lines.append(line)
        return
    if not permit_truncation or remaining <= len(prefix):
        return
    lines.append((prefix + value[:remaining - len(prefix)]).rstrip())


def _append_list_line(
    lines: list[str],
    label: str,
    values: Sequence[str],
    *,
    maximum_chars: int,
) -> None:
    """Append an optional list only when its complete line fits."""

    _append_line(
        lines,
        label,
        '; '.join(values),
        maximum_chars=maximum_chars,
        permit_truncation=False,
    )
