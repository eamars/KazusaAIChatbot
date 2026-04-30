"""Post-dialog consolidator subgraph.

Wraps the post-dialog reflection pipeline. Runs three parallel reflection
nodes (``global_state_updater``, ``relationship_recorder``, ``facts_harvester``),
an evaluator loop over ``facts_harvester``, and a single ``db_writer`` that
commits everything to MongoDB and invalidates the RAG cache.

* A unified ``metadata`` bundle threaded through every node and accumulated
  at each step.
* The ``db_writer`` routes durable user memory through ``user_memory_units``,
  emits Cache2 invalidation events after successful writes, and hands accepted
  future obligations to the task dispatcher so tool calls can be scheduled
  through the shared scheduler.
"""

from __future__ import annotations

from datetime import datetime
import logging

from langgraph.graph import END, START, StateGraph

from kazusa_ai_chatbot.db import get_character_profile, get_conversation_history
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_facts import (
    fact_harvester_evaluator,
    facts_harvester,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_persistence import (
    _normalize_future_promises,
    db_writer,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_reflection import (
    global_state_updater,
    relationship_recorder,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_schema import (
    ConsolidatorState,
    normalize_subjective_appraisals,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.utils import (
    log_dict_subset,
    log_list_preview,
    log_preview,
    trim_history_dict,
)

logger = logging.getLogger(__name__)


async def _consolidator_noop(_: ConsolidatorState) -> dict:
    return_value = {}
    return return_value


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
        Stable lower-cased dedup keys for known facts, milestones, and commitments.
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
    sub_agent_builder = StateGraph(ConsolidatorState)
    reflection_barrier = "reflection_done"
    facts_barrier = "facts_done"

    sub_agent_builder.add_node("global_state_updater", global_state_updater)
    sub_agent_builder.add_node("relationship_recorder", relationship_recorder)
    sub_agent_builder.add_node("facts_harvester", facts_harvester)
    sub_agent_builder.add_node("fact_harvester_evaluator", fact_harvester_evaluator)
    sub_agent_builder.add_node(reflection_barrier, _consolidator_noop)
    sub_agent_builder.add_node(facts_barrier, _consolidator_noop)
    sub_agent_builder.add_node("db_writer", db_writer)

    sub_agent_builder.add_edge(START, "global_state_updater")
    sub_agent_builder.add_edge(START, "relationship_recorder")
    sub_agent_builder.add_edge(START, "facts_harvester")

    sub_agent_builder.add_edge(["global_state_updater", "relationship_recorder"], reflection_barrier)
    sub_agent_builder.add_conditional_edges(
        "facts_harvester",
        lambda state: "skip_eval" if not state["new_facts"] and not state["future_promises"] else "evaluate",
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
        "timestamp": global_state["timestamp"],
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
        "interaction_subtext": global_state["interaction_subtext"],
        "emotional_appraisal": global_state["emotional_appraisal"],
        "character_intent": global_state["character_intent"],
        "logical_stance": global_state["logical_stance"],
        "character_profile": global_state["character_profile"],
        "rag_result": global_state["rag_result"],
        "existing_dedup_keys": _build_existing_dedup_keys(global_state),
        "decontexualized_input": global_state["decontexualized_input"],
        "chat_history_recent": chat_history_recent,
        "metadata": {},
    } # pyright: ignore[reportAssignmentType]

    result = await sub_graph.ainvoke(sub_state)

    mood = result.get("mood", "")
    global_vibe = result.get("global_vibe", "")
    reflection_summary = result.get("reflection_summary", "")
    subjective_appraisals = normalize_subjective_appraisals(result.get("subjective_appraisals"))
    affinity_delta = result.get("affinity_delta", 0)
    last_relationship_insight = result.get("last_relationship_insight", "")
    new_facts = result.get("new_facts", [])
    future_promises = _normalize_future_promises(
        result.get("future_promises", []),
        timestamp=result.get("timestamp", global_state["timestamp"]),
    )
    metadata = result.get("metadata", {}) or {}

    logger.info(f'Consolidation summary: facts={len(new_facts)} promises={len(future_promises)} affinity_delta={affinity_delta} mood={log_preview(mood)} vibe={log_preview(global_vibe)} writes={log_dict_subset(metadata, ["write_success"])} cache_invalidated={metadata.get("cache_invalidated", [])}')

    logger.debug(f'Consolidation detail: facts={log_list_preview(new_facts)} promises={log_list_preview(future_promises)} metadata={log_dict_subset(
            metadata,
            [
                "scheduled_event_ids",
                "contradiction_flags",
                "affinity_before",
                "affinity_delta_processed",
            ],
        )}')

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


async def test_main():
    history = await get_conversation_history(
        platform="discord",
        platform_channel_id="1485606207069880361",
        limit=5,
    )
    trimmed_history = trim_history_dict(history)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    user_input = "既然作业已经写完了，active character 可以晚上可以好好奖励我么♥?"

    state: GlobalPersonaState = {
        "timestamp": current_time,
        "global_user_id": "320899931776745483",
        "user_name": "<current user>",
        "user_profile": {"affinity": 950},

        "internal_monologue": "心跳漏了一拍…这算哪门子'奖励'啊？带着期待的试探。",
        "action_directives": {
            "contextual_directives": {},
            "linguistic_directives": {
                "rhetorical_strategy": "",
                "linguistic_style": "",
                "content_anchors": [
                    "[DECISION] TENTATIVE: 拒绝正面回应关于‘奖励’的具体含义",
                    "[FACT] 现在的时间是深夜（22:24）",
                ],
                "forbidden_phrases": [],
            },
            "visual_directives": {},
        },
        "interaction_subtext": "带有暗示性的调情、索取关注",
        "emotional_appraisal": "心跳漏了一拍……这种轻浮的语气是怎么回事，好乱。",
        "character_intent": "BANTAR",
        "logical_stance": "CONFIRM",

        "final_dialog": ["唔……这种请求也算是一种奖励嘛……真是拿你没办法呢。"],
        "decontexualized_input": user_input,
        "rag_result": {
            "answer": "",
            "user_image": {},
            "character_image": {},
            "third_party_profiles": [],
            "memory_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "supervisor_trace": {
                "loop_count": 0,
                "unknown_slots": [],
                "dispatched": [],
            },
        },
        "chat_history_recent": trimmed_history[-5:],
        "character_profile": await get_character_profile(),
    }

    result = await call_consolidation_subgraph(state)
    print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
