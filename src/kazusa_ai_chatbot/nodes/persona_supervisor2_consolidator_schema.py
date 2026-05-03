"""Stage 4 consolidator shared state schema."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from kazusa_ai_chatbot.time_context import TimeContextDoc


def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts, with b's values overwriting a's."""
    result = dict(a)
    result.update(b)
    return result


def normalize_subjective_appraisals(value: object) -> list[str]:
    """Normalize a subjective-appraisal payload into a clean ``list[str]``.

    Args:
        value: Raw subjective-appraisal payload from state or LLM output.

    Returns:
        A list of non-empty diary entry strings.
    """
    if value is None:
        return_value = []
        return return_value
    if isinstance(value, str):
        text = value.strip()
        return_value = [text] if text else []
        return return_value
    if isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, list):
        items = value
    else:
        return_value = []
        return return_value

    normalized: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            normalized.append(text)
    return normalized


class ConsolidatorState(TypedDict):
    # Inputs for db_writer
    timestamp: str
    time_context: TimeContextDoc
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
    rag_result: dict
    existing_dedup_keys: set[str]

    # User related
    decontexualized_input: str
    chat_history_recent: list[dict]

    # Stage-4a metadata bundle (seeded from RAG metadata, accumulated per node).
    metadata: Annotated[dict, _merge_dicts]

    # global state updater
    mood: str
    global_vibe: str
    reflection_summary: str

    # Relationship recorder
    subjective_appraisals: list[str]
    affinity_delta: int
    last_relationship_insight: str

    # Facts harvester
    new_facts: list[dict]
    future_promises: list[dict]
    fact_harvester_retry: int
    fact_harvester_feedback_message: Annotated[list, add_messages]
    should_stop: bool
