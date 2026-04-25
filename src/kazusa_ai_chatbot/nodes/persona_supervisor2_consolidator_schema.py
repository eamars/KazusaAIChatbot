"""Stage 4 consolidator shared state schema."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts, with b's values overwriting a's."""
    result = dict(a)
    result.update(b)
    return result


def normalize_diary_entries(value: object) -> list[str]:
    """Normalize a diary payload into a clean ``list[str]``.

    Args:
        value: Raw diary payload from state or LLM output.

    Returns:
        A list of non-empty diary entry strings.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, list):
        items = value
    else:
        return []

    normalized: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized

class ConsolidatorState(TypedDict):
    # Inputs for db_writer
    timestamp: str
    global_user_id: str
    user_name: str
    user_profile: dict
    platform: str
    platform_channel_id: str
    channel_type: str
    platform_message_id: str

    # Character related
    action_directives: dict
    internal_monologue: str
    final_dialog: list
    interaction_subtext: str
    emotional_appraisal: str
    character_intent: str
    logical_stance: str
    character_profile: dict

    # Facts
    research_facts: dict

    # User related
    decontexualized_input: str

    # Stage-4a metadata bundle (seeded from RAG metadata, accumulated per node).
    metadata: Annotated[dict, _merge_dicts]

    # global state updater
    mood: str
    global_vibe: str
    reflection_summary: str

    # Relationship recorder
    diary_entry: list[str]
    affinity_delta: int
    last_relationship_insight: str

    # Facts harvester
    new_facts: list[dict]
    future_promises: list[dict]
    fact_harvester_retry: int
    fact_harvester_feedback_message: Annotated[list, add_messages]
    should_stop: bool
