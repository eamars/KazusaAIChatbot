"""Public consolidation subgraph entrypoint."""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from kazusa_ai_chatbot.action_spec.results import (
    project_episode_trace_for_consolidation,
)
from kazusa_ai_chatbot.consolidation.target import (
    build_consolidation_target_plan,
)
from kazusa_ai_chatbot.consolidation.facts import (
    fact_harvester_evaluator,
    facts_harvester,
)
from kazusa_ai_chatbot.consolidation.origin import (
    ConsolidationOriginError,
    build_self_cognition_consolidation_origin,
    build_user_message_consolidation_origin,
)
from kazusa_ai_chatbot.consolidation.persistence import (
    _normalize_future_promises,
    db_writer,
)
from kazusa_ai_chatbot.consolidation.reflection import (
    global_state_updater,
    relationship_recorder,
)
from kazusa_ai_chatbot.consolidation.schema import (
    ConsolidatorState,
    normalize_subjective_appraisals,
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


async def _consolidator_noop(_: ConsolidatorState) -> dict:
    """Return an empty state patch for graph synchronization barriers."""

    return_value: dict = {}
    return return_value


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
        Stable lower-cased dedup keys for known facts, milestones, and
        commitments.
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
    """Run post-dialog consolidation with deterministic target validation.

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

    sub_agent_builder = StateGraph(ConsolidatorState)
    reflection_barrier = "reflection_done"
    facts_barrier = "facts_done"

    sub_agent_builder.add_node("global_state_updater", global_state_updater)
    sub_agent_builder.add_node("relationship_recorder", relationship_recorder)
    sub_agent_builder.add_node("facts_harvester", facts_harvester)
    sub_agent_builder.add_node(
        "fact_harvester_evaluator",
        fact_harvester_evaluator,
    )
    sub_agent_builder.add_node(reflection_barrier, _consolidator_noop)
    sub_agent_builder.add_node(facts_barrier, _consolidator_noop)
    sub_agent_builder.add_node("db_writer", db_writer)

    sub_agent_builder.add_edge(START, "global_state_updater")
    sub_agent_builder.add_edge(START, "relationship_recorder")
    sub_agent_builder.add_edge(START, "facts_harvester")

    sub_agent_builder.add_edge(
        ["global_state_updater", "relationship_recorder"],
        reflection_barrier,
    )
    sub_agent_builder.add_conditional_edges(
        "facts_harvester",
        lambda state: (
            "skip_eval"
            if not state["new_facts"] and not state["future_promises"]
            else "evaluate"
        ),
        {
            "skip_eval": facts_barrier,
            "evaluate": "fact_harvester_evaluator",
        },
    )
    sub_agent_builder.add_conditional_edges(
        "fact_harvester_evaluator",
        lambda state: "loop" if not state["should_stop"] else "end",
        {
            "loop": "facts_harvester",
            "end": facts_barrier,
        },
    )
    sub_agent_builder.add_edge([reflection_barrier, facts_barrier], "db_writer")

    sub_agent_builder.add_edge("db_writer", END)

    sub_graph = sub_agent_builder.compile()

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
    }  # pyright: ignore[reportAssignmentType]

    result = await sub_graph.ainvoke(sub_state)

    mood = result.get("mood", "")
    global_vibe = result.get("global_vibe", "")
    reflection_summary = result.get("reflection_summary", "")
    subjective_appraisals = normalize_subjective_appraisals(
        result.get("subjective_appraisals")
    )
    affinity_delta = result.get("affinity_delta", 0)
    last_relationship_insight = result.get("last_relationship_insight", "")
    new_facts = result.get("new_facts", [])
    future_promises = _normalize_future_promises(
        result.get("future_promises", []),
        storage_timestamp_utc=result.get(
            "storage_timestamp_utc",
            global_state["storage_timestamp_utc"],
        ),
    )
    metadata = result.get("metadata", {}) or {}

    logger.info(
        f"Consolidation output: facts={log_list_preview(new_facts)} "
        f"promises={log_list_preview(future_promises)} "
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
                'contradiction_flags',
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
