"""Stage 4: consolidator subgraph.

Wraps the post-dialog reflection pipeline. Runs three parallel reflection
nodes (``global_state_updater``, ``relationship_recorder``, ``facts_harvester``),
an evaluator loop over ``facts_harvester``, and a single ``db_writer`` that
commits everything to MongoDB and invalidates the RAG cache.

Stage-4a additions:

* A unified ``metadata`` bundle threaded through every node, seeded from
  the RAG metadata produced in Stage 3 and accumulated at each step.
* The ``db_writer`` now routes diary entries / objective facts through the
  new structured helpers (``upsert_character_diary`` / ``upsert_objective_facts``),
  invalidates the matching RAG cache namespaces after a successful commit,
  bumps the per-user RAG version, and hands accepted future obligations to the
  task dispatcher so tool calls can be scheduled through the shared scheduler.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

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
    normalize_diary_entries,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.utils import log_dict_subset, log_list_preview, log_preview

logger = logging.getLogger(__name__)


async def _consolidator_noop(_: ConsolidatorState) -> dict:
    return {}


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
    sub_agent_builder.add_edge("facts_harvester", "fact_harvester_evaluator")
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

    raw_meta = global_state.get("research_metadata")
    seeded_metadata: dict = {}
    if isinstance(raw_meta, list):
        for m in raw_meta:
            if isinstance(m, dict):
                seeded_metadata.update(m)
    elif isinstance(raw_meta, dict):
        seeded_metadata = dict(raw_meta)

    sub_state: ConsolidatorState = {
        "timestamp": global_state["timestamp"],
        "global_user_id": global_state["global_user_id"],
        "user_name": global_state["user_name"],
        "user_profile": global_state["user_profile"],
        "platform": global_state["platform"],
        "platform_channel_id": global_state["platform_channel_id"],
        "channel_type": global_state.get("channel_type", "group"),
        "platform_message_id": global_state["platform_message_id"],
        "action_directives": global_state["action_directives"],
        "internal_monologue": global_state["internal_monologue"],
        "final_dialog": global_state["final_dialog"],
        "interaction_subtext": global_state["interaction_subtext"],
        "emotional_appraisal": global_state["emotional_appraisal"],
        "character_intent": global_state["character_intent"],
        "logical_stance": global_state["logical_stance"],
        "character_profile": global_state["character_profile"],
        "research_facts": global_state["research_facts"],
        "decontexualized_input": global_state["decontexualized_input"],
        "metadata": seeded_metadata,
    }

    result = await sub_graph.ainvoke(sub_state)

    mood = result.get("mood", "")
    global_vibe = result.get("global_vibe", "")
    reflection_summary = result.get("reflection_summary", "")
    diary_entry = normalize_diary_entries(result.get("diary_entry"))
    affinity_delta = result.get("affinity_delta", 0)
    last_relationship_insight = result.get("last_relationship_insight", "")
    new_facts = result.get("new_facts", [])
    future_promises = _normalize_future_promises(
        result.get("future_promises", []),
        timestamp=result.get("timestamp", global_state["timestamp"]),
    )
    metadata = result.get("metadata", {}) or {}

    logger.info(
        "Consolidation summary: facts=%d promises=%d affinity_delta=%s mood=%s vibe=%s writes=%s cache_invalidated=%s",
        len(new_facts),
        len(future_promises),
        affinity_delta,
        log_preview(mood, max_length=60),
        log_preview(global_vibe, max_length=60),
        log_dict_subset(metadata, ["write_success"], value_length=220),
        metadata.get("cache_invalidation_scope", []),
    )

    logger.debug(
        "Consolidation detail: facts=%s promises=%s metadata=%s",
        log_list_preview(new_facts),
        log_list_preview(future_promises),
        log_dict_subset(
            metadata,
            [
                "scheduled_event_ids",
                "contradiction_flags",
                "affinity_before",
                "affinity_delta_processed",
                "knowledge_base_entries_written",
            ],
        ),
    )

    return {
        "mood": mood,
        "global_vibe": global_vibe,
        "reflection_summary": reflection_summary,
        "diary_entry": diary_entry,
        "affinity_delta": affinity_delta,
        "last_relationship_insight": last_relationship_insight,
        "new_facts": new_facts,
        "future_promises": future_promises,
        "consolidation_metadata": metadata,
    }


async def test_main():
    import datetime

    from kazusa_ai_chatbot.db import get_character_profile, get_conversation_history
    from kazusa_ai_chatbot.utils import load_personality, trim_history_dict

    history = await get_conversation_history(platform="discord", platform_channel_id="1485606207069880361", limit=5)
    trimmed_history = trim_history_dict(history)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    user_input = "既然作业已经写完了，千纱可以晚上可以好好奖励我么♥?"

    state: GlobalPersonaState = {
        "timestamp": current_time,
        "global_user_id": "320899931776745483",
        "user_name": "EAMARS",
        "user_profile": {"affinity": 950},

        "internal_monologue": "心跳漏了一拍…这算哪门子'奖励'啊？带着期待的试探罢了。不过既然好感度这么高，这种程度的请求自然要全盘接受——毕竟我是他的千纱嘛。",
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
        "research_facts": f"现在的时间为{current_time}",
        "research_metadata": [{"cache_hit": False, "depth": "DEEP", "depth_confidence": 0.9}],
        "chat_history_recent": trimmed_history[-5:],
        "character_profile": await get_character_profile(),
    }

    result = await call_consolidation_subgraph(state)
    print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
