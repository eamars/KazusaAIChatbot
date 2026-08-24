"""Bounded scene and cognition-evidence projections."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import timedelta, timezone

from kazusa_ai_chatbot.cognition_shared.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
    CognitionEvidenceV2,
)
from kazusa_ai_chatbot.conversation_progress.models import (
    GroupSceneContextV1,
    GroupSceneProjectionFailure,
    GroupSceneTurnV1,
    ConversationLogicalTurnV1,
    ConversationProgressEventV2,
    ConversationProgressPromptV2,
    ConversationProgressStateV2,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    GROUP_SCENE_MAX_ADDRESSED_NAMES,
    GROUP_SCENE_MAX_NAME_CHARS,
    GROUP_SCENE_MAX_RENDERED_CHARS,
    GROUP_SCENE_MAX_TURN_AGE_MINUTES,
    GROUP_SCENE_MAX_TURN_TEXT_CHARS,
    GROUP_SCENE_MAX_TURNS,
    GROUP_SCENE_MAX_VISIBLE_PARTICIPANTS,
    MAX_CONTINUATION_CHARS,
    MAX_PROGRESS_EVIDENCE_CHARS,
    MAX_PROGRESS_EVIDENCE_ROWS,
    MAX_PROGRESS_SCENE_CHARS,
    MAX_OVERUSED_MOVE_PROJECTION_ROWS,
    MAX_SCENE_LOGICAL_TURNS,
    MAX_SCENE_NARRATIVE_CHARS,
    MAX_SCENE_TURN_TEXT_CHARS,
    cap_text,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime


logger = logging.getLogger(__name__)


class GroupSceneProjectionError(ValueError):
    """Raised when a protected public-scene minimum cannot be represented."""

    def __init__(self, code: str, *, protected_anchor_count: int = 0) -> None:
        super().__init__(code)
        self.code = code
        self.protected_anchor_count = protected_anchor_count

    def as_failure(self) -> GroupSceneProjectionFailure:
        """Return the typed degraded result without exposing scene content."""

        failure: GroupSceneProjectionFailure = {
            'code': self.code,
            'protected_anchor_count': self.protected_anchor_count,
        }
        return failure


def filter_group_scene_ambient_turns(
    *,
    ambient_logical_turns: Sequence[ConversationLogicalTurnV1],
    trigger_occurred_at: str,
) -> list[ConversationLogicalTurnV1]:
    """Keep only valid ambient turns within the group-scene age window.

    Args:
        ambient_logical_turns: Canonical logical turns available before the
            current group trigger.
        trigger_occurred_at: Current trigger timestamp used as the age
            reference.

    Returns:
        A copied sequence of valid ambient turns that remain within the
        configured group-scene retention window.
    """

    trigger_time = parse_storage_utc_datetime(trigger_occurred_at)
    maximum_turn_age = timedelta(minutes=GROUP_SCENE_MAX_TURN_AGE_MINUTES)
    required_fields = (
        'role',
        'display_name',
        'fragments',
        'addressed_to_global_user_ids',
        'reply_context',
    )
    filtered: list[ConversationLogicalTurnV1] = []
    for index, turn in enumerate(ambient_logical_turns):
        if not isinstance(turn, Mapping):
            logger.warning(
                f'Skipped malformed ambient group-scene row {index}: '
                'row is not a mapping'
            )
            continue
        if any(field not in turn for field in required_fields):
            logger.warning(
                f'Skipped malformed ambient group-scene row {index}: '
                'missing required projection field'
            )
            continue
        occurred_at = turn.get('occurred_at')
        if not isinstance(occurred_at, str):
            logger.warning(
                f'Skipped malformed ambient group-scene row {index}: '
                'missing occurred_at'
            )
            continue
        try:
            occurred_time = parse_storage_utc_datetime(occurred_at)
        except ValueError as exc:
            logger.warning(
                f'Skipped malformed ambient group-scene row {index}: '
                f'invalid occurred_at: {exc}'
            )
            continue
        if trigger_time - occurred_time > maximum_turn_age:
            continue
        filtered.append(deepcopy(dict(turn)))
    return filtered


def build_group_scene_context(
    *,
    ambient_logical_turns: Sequence[ConversationLogicalTurnV1],
    trigger_occurred_at: str,
    trigger_speaker_name: str,
    trigger_body_text: str,
    trigger_addressed_global_user_ids: Sequence[str],
    trigger_reply_to_display_name: str,
    scope_users: Sequence[Mapping[str, object]],
    current_global_user_id: str = '',
) -> GroupSceneContextV1:
    """Build one bounded chronological public scene without persistence."""

    trigger_time = parse_storage_utc_datetime(trigger_occurred_at)
    roster = _group_scene_roster(scope_users)
    ambient: list[tuple[object, int, GroupSceneTurnV1]] = []
    filtered_ambient_logical_turns = filter_group_scene_ambient_turns(
        ambient_logical_turns=ambient_logical_turns,
        trigger_occurred_at=trigger_occurred_at,
    )
    current_user_anchor_id = ''
    addressed_assistant_anchor_id = ''
    if current_global_user_id:
        for turn in reversed(filtered_ambient_logical_turns):
            if (
                turn['role'] == 'user'
                and turn['global_user_id'] == current_global_user_id
            ):
                current_user_anchor_id = turn['turn_id']
                break
        if current_user_anchor_id:
            current_user_anchor = next(
                turn for turn in filtered_ambient_logical_turns
                if turn['turn_id'] == current_user_anchor_id
            )
            current_user_time = parse_storage_utc_datetime(
                current_user_anchor['occurred_at']
            )
            for turn in reversed(filtered_ambient_logical_turns):
                if (
                    turn['role'] != 'assistant'
                    or turn['broadcast'] is True
                    or turn['addressed_to_global_user_ids'] != [
                        current_global_user_id
                    ]
                    or parse_storage_utc_datetime(turn['occurred_at'])
                    < current_user_time
                    or _reply_targets_other_user(
                        turn['reply_context'],
                        current_global_user_id,
                    )
                ):
                    continue
                addressed_assistant_anchor_id = turn['turn_id']
                break

    for index, turn in enumerate(filtered_ambient_logical_turns):
        occurred_time = parse_storage_utc_datetime(turn['occurred_at'])
        anchor_kind = 'none'
        if turn['turn_id'] == current_user_anchor_id:
            anchor_kind = 'current_user'
        elif turn['turn_id'] == addressed_assistant_anchor_id:
            anchor_kind = 'explicit_assistant'
        ambient.append((
            occurred_time,
            index,
            _normalize_group_scene_turn(
                role=turn['role'],
                speaker_name=turn['display_name'],
                fragments=turn['fragments'],
                addressed_global_user_ids=(
                    turn['addressed_to_global_user_ids']
                ),
                reply_to_display_name=_reply_display_name(
                    turn['reply_context']
                ),
                roster=roster,
                anchor_kind=anchor_kind,
            ),
        ))
    ambient.sort(key=lambda item: (item[0], item[1]))
    protected_anchor_count = sum(
        1
        for _, _, turn in ambient
        if turn.get('anchor_kind') != 'none'
    )
    if any(
        turn.get('anchor_kind') != 'none' and not turn['text']
        for _, _, turn in ambient
    ):
        raise GroupSceneProjectionError(
            'protected_minimum_unfit',
            protected_anchor_count=protected_anchor_count,
        )

    trigger = _normalize_group_scene_turn(
        role='user',
        speaker_name=trigger_speaker_name,
        fragments=[trigger_body_text],
        addressed_global_user_ids=trigger_addressed_global_user_ids,
        reply_to_display_name=trigger_reply_to_display_name,
        roster=roster,
        anchor_kind='none',
    )
    if not trigger['text']:
        raise GroupSceneProjectionError('trigger_empty')
    merged: list[tuple[object, int, GroupSceneTurnV1]] = [
        (occurred_time, 0, turn) for occurred_time, _, turn in ambient
    ]
    # A trigger follows all ambient turns at the same timestamp.
    merged.append((trigger_time, 1, trigger))
    merged.sort(key=lambda item: (item[0], item[1]))
    trigger_position = next(
        index for index, (_, kind, _) in enumerate(merged) if kind == 1
    )
    for index, (_, kind, turn) in enumerate(merged):
        if kind == 1:
            turn['scene_position'] = 'trigger'
        elif index < trigger_position:
            turn['scene_position'] = 'before_trigger'
        else:
            turn['scene_position'] = 'after_trigger'

    ambient_items = [item for item in merged if item[1] == 0]
    protected_items = [
        item for item in ambient_items
        if item[2].get('anchor_kind') in {
            'current_user',
            'explicit_assistant',
        }
    ]
    protected_ids = {id(item[2]) for item in protected_items}
    remaining_items = [
        item for item in ambient_items if id(item[2]) not in protected_ids
    ]
    fill_count = max(GROUP_SCENE_MAX_TURNS - 1 - len(protected_items), 0)
    ambient_fill = remaining_items[-fill_count:] if fill_count else []
    selected_ambient = [*protected_items, *ambient_fill]
    selected = [item[2] for item in selected_ambient]
    selected.append(trigger)
    selected.sort(key=lambda turn: _group_scene_turn_order(turn, merged))

    candidate_context = _fit_group_scene_turns(
        selected,
        total_ambient_count=len(ambient),
    )
    return candidate_context


def project_group_scene_prompt(context: GroupSceneContextV1) -> str:
    """Render one transient group scene within the hard render cap.

    Oversized visible fields are capped deterministically. If the capped
    scene still exceeds the budget, the oldest non-trigger turns are dropped
    and the trigger text is shortened before rendering.
    """

    turns = [
        _bounded_group_scene_turn(turn) for turn in context['turns']
    ]
    visible_participants: list[str] = []
    for name in context['visible_participants']:
        bounded_name = _bounded_name(name)
        if bounded_name and bounded_name not in visible_participants:
            visible_participants.append(bounded_name)
        if len(visible_participants) >= GROUP_SCENE_MAX_VISIBLE_PARTICIPANTS:
            break
    bounded_context: GroupSceneContextV1 = {
        'schema_version': 'group_scene_context.v1',
        'turns': turns,
        'visible_participants': visible_participants,
        'omitted_turn_count': max(0, int(context['omitted_turn_count'])),
    }
    if not any(turn['scene_position'] == 'trigger' for turn in turns):
        raise GroupSceneProjectionError('trigger_empty')
    if any(
        turn['scene_position'] == 'trigger' and not turn['text']
        for turn in turns
    ):
        raise GroupSceneProjectionError('trigger_empty')
    protected_anchor_count = _protected_anchor_count(bounded_context)
    if any(
        turn.get('anchor_kind', 'none') != 'none' and not turn['text']
        for turn in turns
    ):
        raise GroupSceneProjectionError(
            'protected_minimum_unfit',
            protected_anchor_count=protected_anchor_count,
        )
    ambient_turn_count = sum(
        1 for turn in turns if turn['scene_position'] != 'trigger'
    )
    fitted_context = _fit_group_scene_turns(
        turns,
        total_ambient_count=ambient_turn_count,
    )
    fitted_context['visible_participants'] = visible_participants
    rendered = _render_group_scene(fitted_context)
    if len(rendered) > GROUP_SCENE_MAX_RENDERED_CHARS:
        raise GroupSceneProjectionError(
            'protected_minimum_unfit',
            protected_anchor_count=_protected_anchor_count(fitted_context),
        )
    return rendered


def _group_scene_roster(
    scope_users: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    """Index the first visible name for each non-empty global user id."""

    roster: dict[str, str] = {}
    for index, row in enumerate(scope_users):
        if not isinstance(row, Mapping):
            logger.warning(
                f'Skipped malformed group-scene roster row {index}: '
                'row is not a mapping'
            )
            continue
        global_user_id = row.get('global_user_id')
        display_name = row.get('display_name')
        if not isinstance(global_user_id, str) or not global_user_id:
            continue
        if not isinstance(display_name, str):
            continue
        bounded_name = _bounded_name(display_name)
        if bounded_name and global_user_id not in roster:
            roster[global_user_id] = bounded_name
    return roster


def _normalize_group_scene_turn(
    *,
    role: str,
    speaker_name: object,
    fragments: object,
    addressed_global_user_ids: object,
    reply_to_display_name: object,
    roster: Mapping[str, str],
    anchor_kind: str = 'none',
) -> GroupSceneTurnV1:
    """Strip and cap one ambient or trigger turn into visible fields."""

    semantic_role = role if (
        isinstance(role, str) and role in {'user', 'assistant'}
    ) else 'user'
    if not isinstance(speaker_name, str):
        speaker_name = ''
    bounded_speaker = _bounded_name(speaker_name) or semantic_role
    if not isinstance(fragments, Sequence) or isinstance(fragments, str):
        fragments = []
    text = _bounded_text(' '.join(
        fragment for fragment in fragments if isinstance(fragment, str)
    ))
    addressed_names: list[str] = []
    if isinstance(addressed_global_user_ids, Sequence) and not isinstance(
        addressed_global_user_ids,
        str,
    ):
        for global_user_id in addressed_global_user_ids:
            if not isinstance(global_user_id, str) or not global_user_id:
                continue
            name = roster.get(global_user_id, '')
            if name and name not in addressed_names:
                addressed_names.append(name)
            if len(addressed_names) >= GROUP_SCENE_MAX_ADDRESSED_NAMES:
                break
    reply_name = (
        _bounded_name(reply_to_display_name)
        if isinstance(reply_to_display_name, str)
        else ''
    )
    normalized_anchor_kind = (
        anchor_kind
        if anchor_kind in {'none', 'current_user', 'explicit_assistant'}
        else 'none'
    )
    return {
        'role': semantic_role,  # type: ignore[typeddict-item]
        'speaker_name': bounded_speaker,
        'text': text,
        'addressed_names': addressed_names,
        'reply_to_name': reply_name,
        'scene_position': 'trigger',
        'anchor_kind': normalized_anchor_kind,
    }


def _reply_display_name(value: object) -> object:
    """Read only the visible reply display name from one turn context."""

    if not isinstance(value, Mapping):
        return ''
    reply_name = value.get('reply_to_display_name', '')
    return reply_name


def _reply_targets_other_user(
    value: object,
    current_global_user_id: str,
) -> bool:
    """Reject a reply context that explicitly names another participant."""

    if not isinstance(value, Mapping):
        return False
    reply_target = value.get('reply_to_global_user_id')
    return (
        isinstance(reply_target, str)
        and bool(reply_target)
        and reply_target != current_global_user_id
    )


def _bounded_name(value: str) -> str:
    """Apply the shared strip-and-cap policy to a visible name."""

    return cap_text(value, GROUP_SCENE_MAX_NAME_CHARS)


def _bounded_text(value: str) -> str:
    """Apply the public turn text cap."""

    return cap_text(value, GROUP_SCENE_MAX_TURN_TEXT_CHARS)


def _bounded_group_scene_turn(turn: GroupSceneTurnV1) -> GroupSceneTurnV1:
    """Cap one transient turn's visible fields before rendering."""

    addressed_names: list[str] = []
    for name in turn['addressed_names']:
        bounded_name = _bounded_name(name)
        if bounded_name and bounded_name not in addressed_names:
            addressed_names.append(bounded_name)
        if len(addressed_names) >= GROUP_SCENE_MAX_ADDRESSED_NAMES:
            break
    return {
        'role': turn['role'],
        'speaker_name': _bounded_name(turn['speaker_name']),
        'text': _bounded_text(turn['text']),
        'addressed_names': addressed_names,
        'reply_to_name': _bounded_name(turn['reply_to_name']),
        'scene_position': turn['scene_position'],
        'anchor_kind': turn.get('anchor_kind', 'none'),
    }


def _fit_group_scene_turns(
    turns: Sequence[GroupSceneTurnV1],
    *,
    total_ambient_count: int,
) -> GroupSceneContextV1:
    """Fit retained turns while preserving every protected semantic minimum."""

    selected: list[GroupSceneTurnV1] = []
    trigger_seen = False
    for turn in turns:
        if turn['scene_position'] == 'trigger':
            if trigger_seen:
                continue
            trigger_seen = True
        selected.append(turn)
    if not trigger_seen:
        empty_context = _group_scene_context(
            [],
            total_ambient_count=total_ambient_count,
        )
        return empty_context
    protected_turns = [
        turn for turn in selected
        if (
            turn['scene_position'] == 'trigger'
            or turn.get('anchor_kind', 'none') != 'none'
        )
    ]
    if any(not turn['text'] for turn in protected_turns):
        trigger_is_empty = any(
            turn['scene_position'] == 'trigger' and not turn['text']
            for turn in protected_turns
        )
        raise GroupSceneProjectionError(
            'trigger_empty' if trigger_is_empty else 'protected_minimum_unfit',
            protected_anchor_count=_protected_anchor_count(
                _group_scene_context(
                    selected,
                    total_ambient_count=total_ambient_count,
                )
            ),
        )
    maximum_ambient_turns = max(GROUP_SCENE_MAX_TURNS - 1, 0)
    ambient_turns = [
        turn for turn in selected if turn['scene_position'] != 'trigger'
    ]
    if len(ambient_turns) > maximum_ambient_turns:
        protected_ambient = [
            turn for turn in ambient_turns
            if turn.get('anchor_kind', 'none') != 'none'
        ]
        unprotected_ambient = [
            turn for turn in ambient_turns
            if turn.get('anchor_kind', 'none') == 'none'
        ]
        fill_count = max(
            maximum_ambient_turns - len(protected_ambient),
            0,
        )
        retained_ambient = [
            *protected_ambient,
            *unprotected_ambient[-fill_count:],
        ]
        retained_ids = {id(turn) for turn in retained_ambient}
        selected = [
            turn for turn in selected
            if (
                turn['scene_position'] == 'trigger'
                or id(turn) in retained_ids
            )
        ]
    while len(selected) > 1:
        candidate_context = _group_scene_context(
            selected,
            total_ambient_count=total_ambient_count,
        )
        if len(_render_group_scene(candidate_context)) <= (
            GROUP_SCENE_MAX_RENDERED_CHARS
        ):
            return candidate_context
        drop_index = next(
            (
                index for index, turn in enumerate(selected)
                if (
                    turn['scene_position'] != 'trigger'
                    and turn.get('anchor_kind', 'none') == 'none'
                )
            ),
            None,
        )
        if drop_index is None:
            break
        selected.pop(drop_index)
        if len(selected) == 1 and selected[0]['scene_position'] == 'trigger':
            break

    candidate_context = _group_scene_context(
        selected,
        total_ambient_count=total_ambient_count,
    )
    rendered = _render_group_scene(candidate_context)
    if len(rendered) > GROUP_SCENE_MAX_RENDERED_CHARS:
        protected_turns = [
            turn for turn in candidate_context['turns']
            if (
                turn['scene_position'] == 'trigger'
                or turn.get('anchor_kind', 'none') != 'none'
            )
        ]
        if not protected_turns:
            raise GroupSceneProjectionError('trigger_empty')
        for turn in sorted(
            protected_turns,
            key=lambda candidate: len(candidate['text']),
            reverse=True,
        ):
            while len(rendered) > GROUP_SCENE_MAX_RENDERED_CHARS:
                text_length = len(turn['text'])
                if text_length <= 1:
                    break
                turn['text'] = turn['text'][:text_length - 1].rstrip()
                rendered = _render_group_scene(candidate_context)
        if len(rendered) > GROUP_SCENE_MAX_RENDERED_CHARS:
            raise GroupSceneProjectionError(
                'protected_minimum_unfit',
                protected_anchor_count=_protected_anchor_count(
                    candidate_context
                ),
            )
    return candidate_context


def _group_scene_turn_order(
    turn: GroupSceneTurnV1,
    merged: Sequence[tuple[object, int, GroupSceneTurnV1]],
) -> tuple[int, int]:
    """Return the original merged position for a selected turn."""

    for index, (_, _, candidate) in enumerate(merged):
        if candidate is turn:
            return (index, 0)
    return (len(merged), 0)


def _group_scene_context(
    turns: Sequence[GroupSceneTurnV1],
    *,
    total_ambient_count: int,
) -> GroupSceneContextV1:
    """Build the public context shape and recompute visible participants."""

    visible_participants: list[str] = []
    for turn in turns:
        candidates = [
            turn['speaker_name'],
            *turn['addressed_names'],
            turn['reply_to_name'],
        ]
        for name in candidates:
            if name and name not in visible_participants:
                visible_participants.append(name)
            if len(visible_participants) >= (
                GROUP_SCENE_MAX_VISIBLE_PARTICIPANTS
            ):
                break
        if len(visible_participants) >= GROUP_SCENE_MAX_VISIBLE_PARTICIPANTS:
            break
    ambient_retained = sum(
        1 for turn in turns if turn['scene_position'] != 'trigger'
    )
    return {
        'schema_version': 'group_scene_context.v1',
        'turns': [dict(turn) for turn in turns],
        'visible_participants': visible_participants,
        'omitted_turn_count': max(total_ambient_count - ambient_retained, 0),
    }


def _render_group_scene(context: GroupSceneContextV1) -> str:
    """Render semantic public-scene text without transport metadata."""

    lines: list[str] = []
    if context['visible_participants']:
        lines.append(
            'Participants: ' + ', '.join(context['visible_participants'])
        )
    positions = (
        ('before_trigger', 'Before trigger:'),
        ('trigger', 'At trigger:'),
        ('after_trigger', 'After trigger:'),
    )
    for position, label in positions:
        rendered_turns: list[str] = []
        for turn in context['turns']:
            if turn['scene_position'] != position:
                continue
            suffixes: list[str] = []
            if turn['addressed_names']:
                suffixes.append(
                    'to ' + ', '.join(turn['addressed_names'])
                )
            if turn['reply_to_name']:
                suffixes.append('reply to ' + turn['reply_to_name'])
            suffix = f" ({'; '.join(suffixes)})" if suffixes else ''
            rendered_turns.append(
                f"{turn['speaker_name']}: {turn['text']}{suffix}"
            )
        lines.append(f"{label} " + ' | '.join(rendered_turns))
    rendered = '\n'.join(lines)
    return rendered


def _protected_anchor_count(context: GroupSceneContextV1) -> int:
    """Count protected anchor labels kept in a transient scene context."""

    return sum(
        turn.get('anchor_kind', 'none') != 'none'
        for turn in context['turns']
    )


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


def project_conversation_progress_overused_moves(
    progress: ConversationProgressPromptV2,
) -> list[str]:
    """Return the existing model-authored move rows within the prompt cap."""

    projected_moves = progress['overused_moves'][
        :MAX_OVERUSED_MOVE_PROJECTION_ROWS
    ]
    return list(projected_moves)


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
        event_occurred_at = (
            parse_storage_utc_datetime(event['updated_at'])
            .astimezone(timezone.utc)
            .strftime('%Y-%m-%dT%H:%M:%SZ')
        )
        evidence.append({
            'evidence_handle': '',
            'evidence_ref': {
                'source_kind': 'conversation_evidence',
                'source_id': f'conversation-progress-event:{event["event_id"]}',
                'occurred_at': event_occurred_at,
                'semantic_summary': event['semantic_summary'],
            },
            'semantic_text': semantic_text,
            'visible_to': list(
                EVIDENCE_SOURCE_QUESTION_IDS['conversation_evidence']
            ),
            'authority': 'participant_continuity',
            'temporal_provenance': {
                'occurred_at': event_occurred_at,
                'age_descriptor': _progress_age_descriptor(
                    event_occurred_at,
                    occurred_at,
                ),
            },
        })
    used_chars = sum(len(row['semantic_text']) for row in evidence)
    if used_chars > MAX_PROGRESS_EVIDENCE_CHARS:
        raise ValueError('conversation progress evidence exceeds its hard cap')
    return evidence


def _progress_age_descriptor(
    event_occurred_at: str,
    reference_occurred_at: str,
) -> str:
    """Return a bounded age label without changing source timestamps."""

    event_time = parse_storage_utc_datetime(event_occurred_at)
    reference_time = parse_storage_utc_datetime(reference_occurred_at)
    age_seconds = max((reference_time - event_time).total_seconds(), 0.0)
    if age_seconds <= 120:
        return 'fresh'
    if age_seconds <= 1800:
        return 'recent'
    return 'stale'


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
