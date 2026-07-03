"""Public consolidation subgraph entrypoint."""

from __future__ import annotations

import logging

from kazusa_ai_chatbot.action_spec.results import (
    project_episode_trace_for_consolidation,
)
from kazusa_ai_chatbot.consolidation.lane_router import (
    run_consolidation_lane_pipeline,
)
from kazusa_ai_chatbot.consolidation.origin import (
    ConsolidationOriginError,
    build_reflection_consolidation_origin,
    build_self_cognition_consolidation_origin,
    build_user_message_consolidation_origin,
)
from kazusa_ai_chatbot.consolidation.schema import (
    ConsolidatorState,
    normalize_subjective_appraisals,
)
from kazusa_ai_chatbot.consolidation.target import (
    build_consolidation_target_plan,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
    GlobalPersonaState,
)
from kazusa_ai_chatbot.utils import (
    log_dict_subset,
    log_list_preview,
    log_preview,
)

logger = logging.getLogger(__name__)


def _build_consolidation_origin(
    global_state: GlobalPersonaState,
):
    """Build origin metadata for supported consolidation trigger sources.

    Args:
        global_state: Top-level persona-supervisor state.

    Returns:
        Identifier-only consolidation origin metadata.

    Raises:
        ConsolidationOriginError: If the cognitive episode trigger source is
            not supported by the consolidation graph.
    """

    episode = global_state["cognitive_episode"]
    trigger_source = episode["trigger_source"]
    if trigger_source == "user_message":
        origin = build_user_message_consolidation_origin(episode=episode)
    elif trigger_source == "internal_thought":
        origin = build_self_cognition_consolidation_origin(episode=episode)
    elif trigger_source == "reflection_signal":
        origin = build_reflection_consolidation_origin(episode=episode)
    else:
        raise ConsolidationOriginError(
            f"consolidation origin does not support trigger_source={trigger_source}"
        )
    return origin


def _record_existing_dedup_key(row: object, dedup_keys: set[str]) -> None:
    """Add a structured ``dedup_key`` from one profile row when present.

    Args:
        row: Candidate row from the hydrated user profile.
        dedup_keys: Mutable set receiving normalized structured keys.
    """

    if not isinstance(row, dict):
        return
    dedup_key = str(row.get("dedup_key") or "").strip().lower()
    if dedup_key:
        dedup_keys.add(dedup_key)


def _build_existing_dedup_keys(global_state: GlobalPersonaState) -> set[str]:
    """Build exclusion keys from the RAG-projected user memory context.

    Args:
        global_state: Top-level persona-supervisor state.

    Returns:
        Stable lower-cased dedup keys for known memory rows.
    """

    rag_result = global_state["rag_result"]
    user_image = rag_result["user_image"]
    user_memory_context = user_image["user_memory_context"]
    dedup_keys: set[str] = set()

    for entries in user_memory_context.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            _record_existing_dedup_key(entry, dedup_keys)

    return dedup_keys


async def call_consolidation_subgraph(global_state: GlobalPersonaState):
    """Run post-dialog consolidation through the lane-router pipeline.

    Args:
        global_state: Completed persona state after selected surfaces and
            action results are available.

    Returns:
        Consolidation output and sanitized persistence metadata.
    """

    consolidation_origin = _build_consolidation_origin(global_state)
    target_plan_state = {
        **global_state,
        "consolidation_origin": consolidation_origin,
    }
    consolidation_target_plan = build_consolidation_target_plan(
        target_plan_state,
    )
    sub_state = _build_consolidator_state(
        global_state,
        consolidation_origin=consolidation_origin,
        consolidation_target_plan=consolidation_target_plan,
    )

    packet = await run_consolidation_lane_pipeline(sub_state)
    result = packet["state"]

    mood = result.get("mood", "")
    global_vibe = result.get("global_vibe", "")
    reflection_summary = result.get("reflection_summary", "")
    subjective_appraisals = normalize_subjective_appraisals(
        result.get("subjective_appraisals")
    )
    affinity_delta = result.get("affinity_delta", 0)
    last_relationship_insight = result.get("last_relationship_insight", "")
    new_facts = result.get("new_facts", [])
    future_promises = result.get("future_promises", [])
    metadata = result.get("metadata", {}) or {}

    logger.info(
        f"Consolidation output: lanes={log_list_preview(packet['router_tasks'])} "
        f"memory_rows={log_list_preview(new_facts)} "
        f"commitments={log_list_preview(future_promises)} "
        f"mood={log_preview(mood)} vibe={log_preview(global_vibe)} "
        f"reflection={log_preview(reflection_summary)} "
        f"affinity_delta={affinity_delta}"
    )

    logger.debug(
        f"Consolidation metadata: "
        f"writes={log_dict_subset(metadata, ['write_success'])} "
        f"cache_invalidated={metadata.get('cache_invalidated', [])} "
        f"metadata={log_dict_subset(
            metadata,
            [
                'lane_pipeline',
                'affinity_before',
                'affinity_delta_processed',
            ],
        )}"
    )

    return_value = {
        "mood": mood,
        "global_vibe": global_vibe,
        "reflection_summary": reflection_summary,
        "subjective_appraisals": subjective_appraisals,
        "affinity_delta": affinity_delta,
        "last_relationship_insight": last_relationship_insight,
        "new_facts": new_facts,
        "future_promises": future_promises,
        "consolidation_metadata": metadata,
    }
    return return_value


def _build_consolidator_state(
    global_state: GlobalPersonaState,
    *,
    consolidation_origin: dict,
    consolidation_target_plan: dict,
) -> ConsolidatorState:
    """Build the internal state consumed by the lane pipeline."""

    chat_history_recent = global_state.get("chat_history_recent", [])
    sub_state: ConsolidatorState = {
        "storage_timestamp_utc": global_state["storage_timestamp_utc"],
        "local_time_context": global_state["local_time_context"],
        "global_user_id": global_state["global_user_id"],
        "user_name": global_state["user_name"],
        "user_profile": global_state["user_profile"],
        "platform": global_state["platform"],
        "platform_channel_id": global_state["platform_channel_id"],
        "channel_type": global_state["channel_type"],
        "platform_message_id": global_state["platform_message_id"],
        "action_directives": global_state["action_directives"],
        "internal_monologue": global_state["internal_monologue"],
        "final_dialog": global_state["final_dialog"],
        "episode_trace_projection": project_episode_trace_for_consolidation(
            global_state.get("episode_trace"),
        ),
        "interaction_subtext": global_state["interaction_subtext"],
        "emotional_appraisal": global_state["emotional_appraisal"],
        "character_intent": global_state["character_intent"],
        "logical_stance": global_state["logical_stance"],
        "character_profile": global_state["character_profile"],
        "group_channel_style_image": {},
        "rag_result": global_state["rag_result"],
        "existing_dedup_keys": _build_existing_dedup_keys(global_state),
        "decontexualized_input": global_state["decontexualized_input"],
        "chat_history_recent": chat_history_recent,
        "metadata": {},
        "consolidation_origin": consolidation_origin,
        "consolidation_target_plan": consolidation_target_plan,
        "mood": "",
        "global_vibe": "",
        "reflection_summary": "",
        "subjective_appraisals": [],
        "affinity_delta": 0,
        "last_relationship_insight": "",
        "new_facts": [],
        "future_promises": [],
    }  # pyright: ignore[reportAssignmentType]
    return sub_state
