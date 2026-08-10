"""Logical-turn assembly and bounded prompt projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationLogicalTurnV1,
    ConversationProgressSourceRefV2,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    MAX_LOGICAL_TURN_TEXT_CHARS,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime
from kazusa_ai_chatbot.utils import project_text_with_image_blocks


@dataclass(frozen=True)
class LogicalTurnAssembly:
    """Logical turns plus protected malformed-boundary diagnostics."""

    turns: list[ConversationLogicalTurnV1]
    incomplete_or_malformed_turn_count: int


def assemble_logical_turns(
    *,
    rows: Sequence[Mapping[str, object]],
    excluded_row_ids: Sequence[str],
) -> list[ConversationLogicalTurnV1]:
    """Assemble complete chronological turns from canonical storage rows."""

    return assemble_logical_turns_with_diagnostics(
        rows=rows,
        excluded_row_ids=excluded_row_ids,
    ).turns


def assemble_logical_turns_with_diagnostics(
    *,
    rows: Sequence[Mapping[str, object]],
    excluded_row_ids: Sequence[str],
) -> LogicalTurnAssembly:
    """Assemble rows while reporting unusable rows and malformed groups."""

    excluded = set(excluded_row_ids)
    eligible_rows: list[tuple[int, Mapping[str, object]]] = []
    malformed_count = 0
    for source_index, row in enumerate(rows):
        if _row_id(row) in excluded:
            continue
        if not _conversation_row_text(row):
            malformed_count += 1
            continue
        eligible_rows.append((source_index, row))

    ordered_rows = sorted(
        eligible_rows,
        key=lambda item: (
            _timestamp(item[1]),
            item[0],
        ),
    )
    turns: list[ConversationLogicalTurnV1] = []
    row_index = 0
    while row_index < len(ordered_rows):
        source_index, row = ordered_rows[row_index]
        role = _role(row)
        trace_id = _optional_text(row, 'llm_trace_id')
        if role != 'assistant' or not trace_id:
            turns.append(_single_row_turn(row))
            row_index += 1
            continue

        candidate: list[tuple[int, Mapping[str, object]]] = [
            (source_index, row)
        ]
        next_index = row_index + 1
        while next_index < len(ordered_rows):
            candidate_source_index, candidate_row = ordered_rows[next_index]
            if (
                _role(candidate_row) != 'assistant'
                or _optional_text(candidate_row, 'llm_trace_id') != trace_id
            ):
                break
            candidate.append((candidate_source_index, candidate_row))
            next_index += 1

        indexes = [_logical_message_index(item[1]) for item in candidate]
        if _complete_indexes(indexes):
            turns.append(_grouped_assistant_turn(candidate, trace_id))
        elif row_index == 0 and _starts_above_zero(indexes):
            malformed_count += 1
        else:
            malformed_count += 1
            turns.extend(_single_row_turn(item[1]) for item in candidate)
        row_index = next_index

    return LogicalTurnAssembly(
        turns=turns,
        incomplete_or_malformed_turn_count=malformed_count,
    )


def select_recent_logical_turns(
    turns: Sequence[ConversationLogicalTurnV1],
    *,
    limit: int,
) -> list[ConversationLogicalTurnV1]:
    """Keep the newest complete turns while preserving chronology."""

    if limit < 0:
        raise ValueError('logical turn limit must be non-negative')
    if limit == 0:
        return []
    return [dict(turn) for turn in turns[-limit:]]


def project_logical_turns_for_prompt(
    turns: Sequence[ConversationLogicalTurnV1],
    *,
    maximum_chars: int,
    maximum_turn_chars: int = MAX_LOGICAL_TURN_TEXT_CHARS,
) -> list[str]:
    """Render newest complete turns within one aggregate text budget."""

    if maximum_chars < 0 or maximum_turn_chars <= 0:
        raise ValueError('logical turn prompt budgets are invalid')
    selected_reversed: list[str] = []
    used_chars = 0
    for turn in reversed(turns):
        line = _logical_turn_line(turn, maximum_turn_chars)
        separator_chars = 1 if selected_reversed else 0
        if used_chars + separator_chars + len(line) > maximum_chars:
            continue
        selected_reversed.append(line)
        used_chars += separator_chars + len(line)
    selected_reversed.reverse()
    return selected_reversed


def logical_turn_source_refs(
    turns: Sequence[ConversationLogicalTurnV1],
) -> list[ConversationProgressSourceRefV2]:
    """Project one exact canonical source alias per protected logical turn."""

    refs: list[ConversationProgressSourceRefV2] = []
    seen: set[tuple[str, str]] = set()
    for turn in turns:
        occurred_at = turn['occurred_at']
        trace_id = turn['llm_trace_id']
        if trace_id and turn['turn_id'] == f'trace:{trace_id}':
            _append_source_ref(
                refs,
                seen,
                ref_kind='llm_trace',
                ref_id=trace_id,
                occurred_at=occurred_at,
            )
            continue

        row_ids = turn['conversation_row_ids']
        if (
            len(row_ids) != 1
            or turn['turn_id'] != f'row:{row_ids[0]}'
        ):
            raise ValueError(
                'logical turn has no exact canonical source identity'
            )
        _append_source_ref(
            refs,
            seen,
            ref_kind='conversation_row',
            ref_id=row_ids[0],
            occurred_at=occurred_at,
        )
    return refs


def logical_turns_as_history_rows(
    turns: Sequence[ConversationLogicalTurnV1],
) -> list[dict[str, object]]:
    """Project one row per logical turn for remaining history consumers."""

    rows: list[dict[str, object]] = []
    for turn in turns:
        rows.append({
            'role': turn['role'],
            'timestamp': turn['occurred_at'],
            'display_name': turn['display_name'],
            'body_text': '\n'.join(turn['fragments']),
            'platform_user_id': turn['platform_user_id'],
            'global_user_id': turn['global_user_id'],
            'addressed_to_global_user_ids': list(
                turn['addressed_to_global_user_ids']
            ),
            'broadcast': turn['broadcast'],
            'reply_context': dict(turn['reply_context']),
            'llm_trace_id': turn['llm_trace_id'],
        })
    return rows


def _grouped_assistant_turn(
    candidate: Sequence[tuple[int, Mapping[str, object]]],
    trace_id: str,
) -> ConversationLogicalTurnV1:
    """Build one complete assistant turn ordered by logical fragment index."""

    ordered = sorted(
        candidate,
        key=lambda item: (
            _logical_message_index(item[1]),
            _required_text(item[1], 'timestamp'),
            item[0],
        ),
    )
    rows = [item[1] for item in ordered]
    first = rows[0]
    addressed: list[str] = []
    for row in rows:
        for user_id in _string_list(row, 'addressed_to_global_user_ids'):
            if user_id not in addressed:
                addressed.append(user_id)
    reply_context: dict[str, object] = {}
    for row in rows:
        value = row.get('reply_context')
        if isinstance(value, Mapping) and value:
            reply_context = dict(value)
            break
    return {
        'turn_id': f'trace:{trace_id}',
        'role': 'assistant',
        'occurred_at': _required_text(first, 'timestamp'),
        'display_name': _optional_text(first, 'display_name'),
        'fragments': [_conversation_row_text(row) for row in rows],
        'conversation_row_ids': [_row_id(row) for row in rows],
        'llm_trace_id': trace_id,
        'platform_user_id': _optional_text(first, 'platform_user_id'),
        'global_user_id': _optional_text(first, 'global_user_id'),
        'addressed_to_global_user_ids': addressed,
        'broadcast': any(row.get('broadcast') is True for row in rows),
        'reply_context': reply_context,
    }


def _single_row_turn(
    row: Mapping[str, object],
) -> ConversationLogicalTurnV1:
    """Build one ungrouped turn from one canonical row."""

    row_id = _row_id(row)
    reply_context = row.get('reply_context')
    if not isinstance(reply_context, Mapping):
        reply_context = {}
    return {
        'turn_id': f'row:{row_id}',
        'role': _role(row),
        'occurred_at': _required_text(row, 'timestamp'),
        'display_name': _optional_text(row, 'display_name'),
        'fragments': [_conversation_row_text(row)],
        'conversation_row_ids': [row_id],
        'llm_trace_id': _optional_text(row, 'llm_trace_id'),
        'platform_user_id': _optional_text(row, 'platform_user_id'),
        'global_user_id': _optional_text(row, 'global_user_id'),
        'addressed_to_global_user_ids': _string_list(
            row,
            'addressed_to_global_user_ids',
        ),
        'broadcast': row.get('broadcast') is True,
        'reply_context': dict(reply_context),
    }


def _logical_turn_line(
    turn: ConversationLogicalTurnV1,
    maximum_chars: int,
) -> str:
    """Render one turn without exposing protected source identifiers."""

    speaker = turn['display_name'] or turn['role']
    reply_to = turn['reply_context'].get('reply_to_display_name')
    if isinstance(reply_to, str) and reply_to.strip():
        speaker = f'{speaker} reply_to {reply_to.strip()}'
    text = ' '.join(fragment.strip() for fragment in turn['fragments'])
    line = f'[{turn["occurred_at"]}] {speaker}: {text}'
    return line[:maximum_chars].rstrip()


def _row_id(row: Mapping[str, object]) -> str:
    """Return the canonical Mongo conversation-row identifier."""

    if '_id' not in row:
        raise ValueError('conversation history row _id is required')
    row_id = str(row['_id']).strip()
    if not row_id:
        raise ValueError('conversation history row _id is empty')
    return row_id


def _role(row: Mapping[str, object]) -> Literal['user', 'assistant']:
    """Validate one canonical conversation author role."""

    role = row.get('role')
    if role not in {'user', 'assistant'}:
        raise ValueError('conversation history row role is invalid')
    return role


def _required_text(row: Mapping[str, object], field_name: str) -> str:
    """Return one required non-empty row text field."""

    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'conversation history row {field_name} is required')
    return value.strip()


def _optional_text(row: Mapping[str, object], field_name: str) -> str:
    """Return one optional row text field after type validation."""

    value = row.get(field_name, '')
    if not isinstance(value, str):
        raise ValueError(f'conversation history row {field_name} must be text')
    return value.strip()


def _conversation_row_text(row: Mapping[str, object]) -> str:
    """Project authored text and stored image descriptions for one row."""

    body_text = row.get('body_text')
    if not isinstance(body_text, str):
        return_value = ''
        return return_value
    projected_text = project_text_with_image_blocks(
        body_text.strip(),
        row.get('attachments'),
    )
    return_value = projected_text.strip()
    return return_value


def _timestamp(row: Mapping[str, object]) -> str:
    """Return one canonical storage-UTC row timestamp."""

    timestamp = _required_text(row, 'timestamp')
    parse_storage_utc_datetime(timestamp)
    return timestamp


def _string_list(
    row: Mapping[str, object],
    field_name: str,
) -> list[str]:
    """Validate one optional unique string-list field."""

    value = row.get(field_name, [])
    if not isinstance(value, list):
        raise ValueError(f'conversation history row {field_name} must be a list')
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f'conversation history row {field_name} item is invalid'
            )
        item_text = item.strip()
        if item_text not in result:
            result.append(item_text)
    return result


def _logical_message_index(row: Mapping[str, object]) -> int | None:
    """Return a valid assistant fragment index or ``None``."""

    value = row.get('logical_message_index')
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _complete_indexes(indexes: Sequence[int | None]) -> bool:
    """Return whether indexes are unique and contiguous from zero."""

    if any(index is None for index in indexes):
        return False
    typed_indexes = [index for index in indexes if index is not None]
    return sorted(typed_indexes) == list(range(len(typed_indexes)))


def _starts_above_zero(indexes: Sequence[int | None]) -> bool:
    """Return whether an oldest fetched group is a cut-off suffix."""

    typed_indexes = [index for index in indexes if index is not None]
    return bool(typed_indexes) and min(typed_indexes) > 0


def _append_source_ref(
    refs: list[ConversationProgressSourceRefV2],
    seen: set[tuple[str, str]],
    *,
    ref_kind: Literal['conversation_row', 'llm_trace'],
    ref_id: str,
    occurred_at: str,
) -> None:
    """Append one unique source alias."""

    identity = (ref_kind, ref_id)
    if identity in seen:
        return
    seen.add(identity)
    refs.append({
        'ref_kind': ref_kind,
        'ref_id': ref_id,
        'occurred_at': occurred_at,
    })
