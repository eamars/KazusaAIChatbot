"""Mechanical limits and lifecycle helpers for conversation progress."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Mapping

from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationProgressPromptV2,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime

COLLECTION_NAME = 'conversation_episode_state'
BLOCK_COLLECTION_NAME = 'conversation_episode_blocks'
BLOCK_VECTOR_INDEX_NAME = 'conversation_episode_blocks_vector_index'

EPISODE_TTL = timedelta(hours=48)
CACHE_TTL_SECONDS = 60 * 60

AMBIENT_ROW_SCAN_LIMIT = 48
INTERACTION_ROW_SCAN_LIMIT = 128
AMBIENT_LOGICAL_TURN_LIMIT = 6
INTERACTION_LOGICAL_TURN_LIMIT = 10
MAX_LOGICAL_TURN_TEXT_CHARS = 600
MAX_AMBIENT_PROMPT_CHARS = 1200
MAX_INTERACTION_RECORDER_CHARS = 2000

MAX_EPISODE_NARRATIVE_CHARS = 900
MAX_THREAD_FIELD_CHARS = 240
MAX_MOVE_CHARS = 120
MAX_ACTIVE_EVENTS = 24
MAX_RECENT_TURN_REFS = 16
MAX_ACTIVE_BLOCK_REFS = 8
MAX_ACTIVE_PACKET_CHARS = 16000
COMPACTION_EVENT_SOFT_LIMIT = 18
COMPACTION_TURN_REF_SOFT_LIMIT = 12
COMPACTION_PACKET_SOFT_CHARS = 10000

MAX_EVENT_SUMMARY_CHARS = 220
MAX_EVENT_ROLE_CHARS = 160
MAX_EVENT_OUTCOME_CHARS = 180
MAX_EVENT_SOURCE_REFS = 4

MAX_BLOCK_EVENTS = 8
MAX_BLOCK_TURN_REFS = 24
MAX_BLOCK_SOURCE_BLOCKS = 4
MAX_BLOCK_CHARS = 12000
MAX_BLOCK_SEMANTIC_KEYS = 8
MAX_BLOCK_SEMANTIC_KEY_CHARS = 80
MAX_BLOCK_NARRATIVE_CHARS = 900
MAX_BLOCK_SEARCH_RESULTS = 3
MAX_BLOCK_GRAPH_DEPTH = 8
MAX_REACHABLE_BLOCK_REFS = 128

MAX_PROGRESS_SCENE_CHARS = 2200
MAX_SCENE_NARRATIVE_CHARS = 600
MAX_SCENE_LOGICAL_TURNS = 4
MAX_SCENE_TURN_TEXT_CHARS = 160
MAX_PROGRESS_EVIDENCE_CHARS = 1800
MAX_PROGRESS_EVIDENCE_ROWS = 8
MAX_CONTINUATION_CHARS = 4000
MAX_RECORDER_HUMAN_PAYLOAD_CHARS = 24000
MAX_SCENE_RECORDER_HUMAN_PAYLOAD_CHARS = 8000

VALID_CONTINUITY = frozenset({
    'same_episode',
    'related_shift',
    'sharp_transition',
})
VALID_STATUS = frozenset({'active', 'suspended', 'closed'})
VALID_EVENT_STATES = frozenset({
    'open',
    'in_progress',
    'completed',
    'rejected',
    'superseded',
})
TERMINAL_EVENT_STATES = frozenset({
    'completed',
    'rejected',
    'superseded',
})
VALID_EVENT_RETENTIONS = frozenset({
    'decision_critical',
    'active_scene',
    'background',
})
VALID_SOURCE_REF_KINDS = frozenset({'conversation_row', 'llm_trace'})


def cap_text(value: str, maximum_chars: int) -> str:
    """Strip and deterministically cap one Unicode text value."""

    if not isinstance(value, str):
        raise TypeError('cap_text value must be a string')
    if maximum_chars < 0:
        raise ValueError('maximum_chars must be non-negative')
    return value.strip()[:maximum_chars].rstrip()


def storage_expiry(
    storage_timestamp_utc: str,
) -> tuple[str, datetime]:
    """Return semantic and BSON physical expiry values for one write."""

    current = parse_storage_utc_datetime(storage_timestamp_utc)
    purge_after = current + EPISODE_TTL
    return purge_after.isoformat(), purge_after


def is_unexpired_storage_timestamp(
    expires_at: object,
    *,
    current_timestamp_utc: str,
) -> bool:
    """Return whether one valid storage timestamp is after the current time."""

    if not isinstance(expires_at, str) or not expires_at:
        return False
    try:
        expiry = parse_storage_utc_datetime(expires_at)
        current = parse_storage_utc_datetime(current_timestamp_utc)
    except ValueError:
        return False
    return expiry > current


def canonical_json_chars(value: Mapping[str, object]) -> int:
    """Measure compact sorted JSON with datetimes rendered as UTC text."""

    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            default=_json_default,
        )
    )


def prompt_payload_chars(payload: ConversationProgressPromptV2) -> int:
    """Measure one prompt projection using the canonical JSON rule."""

    return canonical_json_chars(payload)


def _json_default(value: object) -> object:
    """Render supported non-JSON lifecycle values deterministically."""

    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f'unsupported JSON value: {type(value).__name__}')
