"""Consolidator shared state schema."""

from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict


from kazusa_ai_chatbot.consolidation.origin import (
    ConsolidationOriginMetadata,
)
from kazusa_ai_chatbot.consolidation.target import ConsolidationTargetPlan
from kazusa_ai_chatbot.time_boundary import LocalTimeContextDoc


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


def content_plan_from_action_directives(
    action_directives: object,
) -> dict[str, str]:
    """Project a text content plan when a surface stage produced one."""

    if not isinstance(action_directives, dict):
        return_value: dict[str, str] = {}
        return return_value

    if action_directives.get("schema_version") == "text_surface_output.v2":
        content_plan = action_directives.get("content_plan")
        if isinstance(content_plan, str) and content_plan.strip():
            return {"semantic_content": content_plan.strip()}
        return {}

    linguistic_directives = action_directives.get("linguistic_directives")
    if not isinstance(linguistic_directives, dict):
        return_value = {}
        return return_value

    content_plan = linguistic_directives.get("content_plan")
    if not isinstance(content_plan, dict):
        return_value = {}
        return return_value

    normalized: dict[str, str] = {}
    for raw_key, raw_value in content_plan.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            continue
        key = raw_key.strip()
        value = raw_value.strip()
        if key and value:
            normalized[key] = value
    return_value = normalized
    return return_value


class ConsolidatorState(TypedDict):
    # Inputs for db_writer
    storage_timestamp_utc: str
    local_time_context: LocalTimeContextDoc
    global_user_id: str
    user_name: str
    user_profile: dict
    platform: str
    platform_channel_id: str
    channel_type: str
    platform_message_id: str

    # Character related
    cognition_core_output: NotRequired[dict]
    action_directives: NotRequired[dict]
    text_surface_output_v2: NotRequired[dict]
    internal_monologue: str
    final_dialog: list
    episode_trace_projection: dict
    interaction_subtext: str
    subjective_appraisals: list[str]
    emotional_appraisal: str
    character_intent: str
    logical_stance: str
    character_profile: dict
    group_channel_style_image: dict

    # Facts
    rag_result: dict
    existing_dedup_keys: set[str]

    # User related
    decontexualized_input: str
    chat_history_recent: list[dict]

    # Shared metadata bundle seeded from RAG metadata and accumulated per node.
    metadata: Annotated[dict, _merge_dicts]
    consolidation_origin: ConsolidationOriginMetadata
    consolidation_target_plan: ConsolidationTargetPlan

    # Consolidation memory rows
    new_facts: list[dict]
    future_promises: list[dict]
    should_stop: bool
