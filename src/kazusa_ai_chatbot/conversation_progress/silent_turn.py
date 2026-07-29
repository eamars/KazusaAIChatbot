"""Explicit settled-fact gate for progress recording on silent turns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal


def select_recordable_turn_outcome(
    *,
    final_dialog: Sequence[str],
    episode_trace: Mapping[str, object],
    cognition_output: Mapping[str, object] | None,
    relevance_approved: bool,
    consolidatable: bool,
    listen_only: bool,
    pruned: bool,
) -> Literal['visible_response', 'cognition_silence'] | None:
    """Select a recordable outcome from typed upstream decisions only."""

    if (
        not relevance_approved
        or not consolidatable
        or listen_only
        or pruned
        or episode_trace.get('schema_version') != 'episode_trace.v2'
    ):
        return None
    if final_dialog:
        if episode_trace.get('terminal_status') != 'completed_visible':
            return None
        return 'visible_response'
    if (
        episode_trace.get('terminal_status') != 'completed_private'
        or cognition_output is None
    ):
        return None
    intention = cognition_output.get('intention')
    if not isinstance(intention, Mapping):
        return None
    if intention.get('route') != 'silence':
        return None
    return 'cognition_silence'
