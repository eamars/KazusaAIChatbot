"""Progressive RAG supervisor graph entrypoint.

The domain implementation is split by responsibility:
initializer/cache, dispatch/execution, prompt-facing compact views, and
evaluation/finalization. This module keeps the public graph entrypoint and
compatibility shims used by focused tests.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_dispatch as _dispatch_domain
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_evaluator as _evaluator_domain
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_initializer as _initializer_domain
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_prompt_views as _prompt_view_domain
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_types import (
    _MAX_LOOP_COUNT,
    ProgressiveRAGState,
    RAGAgentCallable,
    RAGAgentRegistryEntry,
    RAGFactSource,
)
from kazusa_ai_chatbot.utils import log_list_preview, log_preview

logger = logging.getLogger(__name__)

_INITIALIZER_PROMPT = _initializer_domain._INITIALIZER_PROMPT
_DISPATCHER_PROMPT = _dispatch_domain._DISPATCHER_PROMPT
_EVALUATOR_SUMMARIZER_PROMPT = _evaluator_domain._EVALUATOR_SUMMARIZER_PROMPT
_EVALUATOR_SUMMARIZER_USER_PROFILE_PROMPT = (
    _evaluator_domain._EVALUATOR_SUMMARIZER_USER_PROFILE_PROMPT
)
_CONTINUATION_ASSESSOR_PROMPT = _evaluator_domain._CONTINUATION_ASSESSOR_PROMPT
_FINALIZER_PROMPT = _evaluator_domain._FINALIZER_PROMPT

_RAG_SUPERVISOR_AGENT_REGISTRY = _dispatch_domain._RAG_SUPERVISOR_AGENT_REGISTRY
_initializer_llm = _initializer_domain._initializer_llm
_dispatcher_llm = _dispatch_domain._dispatcher_llm
_evaluator_summarizer_llm = _evaluator_domain._evaluator_summarizer_llm
_continuation_assessor_llm = _evaluator_domain._continuation_assessor_llm
_finalizer_llm = _evaluator_domain._finalizer_llm

get_rag_cache2_runtime = _initializer_domain.get_rag_cache2_runtime
record_initializer_hit = _initializer_domain.record_initializer_hit
upsert_initializer_entry = _initializer_domain.upsert_initializer_entry
INITIALIZER_CACHE_NAME = _initializer_domain.INITIALIZER_CACHE_NAME
INITIALIZER_PROMPT_VERSION = _initializer_domain.INITIALIZER_PROMPT_VERSION
INITIALIZER_AGENT_REGISTRY_VERSION = (
    _initializer_domain.INITIALIZER_AGENT_REGISTRY_VERSION
)
INITIALIZER_STRATEGY_SCHEMA_VERSION = (
    _initializer_domain.INITIALIZER_STRATEGY_SCHEMA_VERSION
)

_normalize_initializer_slots = _initializer_domain._normalize_initializer_slots
_read_cached_initializer_slots = _initializer_domain._read_cached_initializer_slots
_initializer_cache_result = _initializer_domain._initializer_cache_result
_initializer_cache_metadata = _initializer_domain._initializer_cache_metadata
_initializer_cache_status = _initializer_domain._initializer_cache_status
_compact_memory_unit_rows = _prompt_view_domain._compact_memory_unit_rows
_compact_user_memory_context = _prompt_view_domain._compact_user_memory_context
_compact_profile_for_llm = _prompt_view_domain._compact_profile_for_llm
_compact_projection_payload_for_llm = (
    _prompt_view_domain._compact_projection_payload_for_llm
)
_compact_raw_result_for_llm = _prompt_view_domain._compact_raw_result_for_llm
_known_facts_llm_view = _prompt_view_domain._known_facts_llm_view
_clip_llm_summary_text = _prompt_view_domain._clip_llm_summary_text


def _sync_initializer_domain() -> None:
    """Apply monkeypatched initializer dependencies to the domain module."""
    _initializer_domain._initializer_llm = _initializer_llm
    _initializer_domain.get_rag_cache2_runtime = get_rag_cache2_runtime
    _initializer_domain.record_initializer_hit = record_initializer_hit
    _initializer_domain.upsert_initializer_entry = upsert_initializer_entry


def _sync_dispatch_domain() -> None:
    """Apply monkeypatched dispatcher dependencies to the domain module."""
    _dispatch_domain._dispatcher_llm = _dispatcher_llm
    _dispatch_domain._RAG_SUPERVISOR_AGENT_REGISTRY = (
        _RAG_SUPERVISOR_AGENT_REGISTRY
    )


def _sync_evaluator_domain() -> None:
    """Apply monkeypatched evaluator dependencies to the domain module."""
    _sync_initializer_domain()
    _evaluator_domain._evaluator_summarizer_llm = _evaluator_summarizer_llm
    _evaluator_domain._continuation_assessor_llm = _continuation_assessor_llm
    _evaluator_domain._finalizer_llm = _finalizer_llm


def build_rag_fact_source_map() -> dict[str, dict[str, Any]]:
    """Build the deterministic fact-source map for RAG known facts."""
    _sync_dispatch_domain()
    fact_source_map = _dispatch_domain.build_rag_fact_source_map()
    return fact_source_map


def _build_agent_name_union() -> str:
    """Render the dispatcher's allowed ``agent_name`` values."""
    _sync_dispatch_domain()
    agent_name_union = _dispatch_domain._build_agent_name_union()
    return agent_name_union


def _normalize_dispatch(raw_dispatch: dict, current_slot: str) -> dict:
    """Normalize dispatcher JSON into a safe executable dispatch payload."""
    _sync_dispatch_domain()
    dispatch = _dispatch_domain._normalize_dispatch(raw_dispatch, current_slot)
    return dispatch


def _build_delegate_context(state: ProgressiveRAGState, dispatch: dict) -> dict:
    """Build the helper-agent context for one dispatched slot."""
    _sync_dispatch_domain()
    delegate_context = _dispatch_domain._build_delegate_context(state, dispatch)
    return delegate_context


async def _summarize_agent_result(
    slot: str,
    agent_name: str,
    resolved: bool,
    raw_result: object,
    known_facts: list[dict],
) -> str:
    """Summarize one agent result through the evaluator domain."""
    _sync_evaluator_domain()
    summary = await _evaluator_domain._summarize_agent_result(
        slot,
        agent_name,
        resolved,
        raw_result,
        known_facts,
    )
    return summary


async def _assess_continuation(
    *,
    observation_payload: dict[str, object],
    original_query: str,
    previous_refined_queries: list[str],
    continuation_count: int,
) -> dict[str, object]:
    """Assess whether an unresolved observation should re-enter RAG."""
    _sync_evaluator_domain()
    decision = await _evaluator_domain._assess_continuation(
        observation_payload=observation_payload,
        original_query=original_query,
        previous_refined_queries=previous_refined_queries,
        continuation_count=continuation_count,
    )
    return decision


async def rag_initializer(state: ProgressiveRAGState) -> dict:
    """Decompose the input query into ordered unknown slots."""
    _sync_initializer_domain()
    update = await _initializer_domain.rag_initializer(state)
    return update


async def rag_dispatcher(state: ProgressiveRAGState) -> dict:
    """Dispatch the next unknown slot to one helper agent."""
    _sync_dispatch_domain()
    update = await _dispatch_domain.rag_dispatcher(state)
    dispatch = update.get("current_dispatch", {})
    if isinstance(dispatch, dict):
        logger.debug(
            f'RAG2 dispatch metadata: loop={update.get("loop_count", 0)} '
            f"slot={log_preview(update.get('current_slot', ''))} "
            f'max_attempts={dispatch.get("max_attempts", 0)} '
            f'route_source={dispatch.get("route_source", "")} '
            f"dispatch_context={log_preview(dispatch.get('context', {}))}"
        )
    return update


async def rag_executor(state: ProgressiveRAGState) -> dict:
    """Execute the helper-agent call selected by the dispatcher."""
    _sync_dispatch_domain()
    update = await _dispatch_domain.rag_executor(state)
    return update


async def rag_evaluator(state: ProgressiveRAGState) -> dict:
    """Evaluate one helper-agent result and update known facts."""
    _sync_evaluator_domain()
    update = await _evaluator_domain.rag_evaluator(state)
    return update


async def rag_finalizer(state: ProgressiveRAGState) -> dict:
    """Synthesize the final factual RAG answer."""
    _sync_evaluator_domain()
    update = await _evaluator_domain.rag_finalizer(state)
    return update


def _route_after_initializer(state: ProgressiveRAGState) -> str:
    """Skip to finalizer if the initializer produced no slots."""
    return_value = "dispatch" if state.get("unknown_slots") else "finalize"
    return return_value


def _route_after_dispatcher(state: ProgressiveRAGState) -> str:
    """Execute if the dispatcher produced a valid agent name."""
    dispatch = _normalize_dispatch(
        state.get("current_dispatch", {}),
        state.get("current_slot", ""),
    )
    if dispatch["agent_name"]:
        return "execute"
    return "finalize"


def _route_after_evaluator(state: ProgressiveRAGState) -> str:
    """Loop for remaining slots under the bounded supervisor cap."""
    known_facts = state.get("known_facts", [])
    if known_facts:
        last_fact = known_facts[-1]
    else:
        last_fact = {}

    if last_fact and not last_fact.get("resolved", True):
        continuation = last_fact.get("continuation")
        if (
            isinstance(continuation, dict)
            and continuation.get("should_continue") is True
            and continuation.get("refined_query")
            and state.get("unknown_slots")
            and state.get("loop_count", 0) < _MAX_LOOP_COUNT
        ):
            return "loop"
        return "finalize"
    if state.get("unknown_slots") and state.get("loop_count", 0) < _MAX_LOOP_COUNT:
        return "loop"
    return "finalize"


async def call_rag_supervisor(
    original_query: str,
    character_name: str = "",
    context: dict | None = None,
) -> dict:
    """Run the progressive RAG supervisor over a single query.

    Args:
        original_query: User's natural-language question.
        character_name: Display name of the active character.
        context: Optional platform, channel, user, and message-envelope fields.

    Returns:
        Dict with keys ``answer``, ``known_facts``, ``unknown_slots``, and
        ``loop_count``.
    """
    runtime_context = context or {}
    builder = StateGraph(ProgressiveRAGState)

    builder.add_node("rag_initializer", rag_initializer)
    builder.add_node("rag_dispatcher", rag_dispatcher)
    builder.add_node("rag_executor", rag_executor)
    builder.add_node("rag_evaluator", rag_evaluator)
    builder.add_node("rag_finalizer", rag_finalizer)

    builder.add_edge(START, "rag_initializer")
    builder.add_conditional_edges(
        "rag_initializer",
        _route_after_initializer,
        {"dispatch": "rag_dispatcher", "finalize": "rag_finalizer"},
    )
    builder.add_conditional_edges(
        "rag_dispatcher",
        _route_after_dispatcher,
        {"execute": "rag_executor", "finalize": "rag_finalizer"},
    )
    builder.add_edge("rag_executor", "rag_evaluator")
    builder.add_conditional_edges(
        "rag_evaluator",
        _route_after_evaluator,
        {"loop": "rag_dispatcher", "finalize": "rag_finalizer"},
    )
    builder.add_edge("rag_finalizer", END)

    graph = builder.compile()

    initial_state: ProgressiveRAGState = {
        "original_query": original_query,
        "character_name": character_name,
        "context": runtime_context,
        "unknown_slots": [],
        "current_slot": "",
        "known_facts": [],
        "messages": [],
        "initializer_cache": {},
        "current_dispatch": {},
        "last_agent_result": {},
        "loop_count": 0,
        "final_answer": "",
    }

    logger.debug(
        f'RAG2 request metadata: platform={runtime_context.get("platform", "")} '
        f'channel={runtime_context.get("platform_channel_id", "") or "<dm>"} '
        f'user={runtime_context.get("global_user_id", "")} '
        f"character={log_preview(character_name)} "
        f'history_recent={len(runtime_context.get("chat_history_recent", []))} '
        f'history_wide={len(runtime_context.get("chat_history_wide", []))} '
        f"query={log_preview(original_query)} "
        f"context={log_preview(runtime_context)}"
    )
    result = await graph.ainvoke(initial_state)
    known_facts = result.get("known_facts", [])
    unknown_slots = result.get("unknown_slots", [])
    loop_count = result.get("loop_count", 0)
    final_answer = result.get("final_answer", "")

    logger.debug(
        f'RAG2 summary metadata: platform={runtime_context.get("platform", "")} '
        f'channel={runtime_context.get("platform_channel_id", "") or "<dm>"} '
        f'user={runtime_context.get("global_user_id", "")} '
        f"loop_count={loop_count} known_facts={len(known_facts)} "
        f"unknown_slots={len(unknown_slots)} answer={log_preview(final_answer)} "
        f"facts={log_preview(known_facts)} "
        f"remaining_slots={log_list_preview(unknown_slots)}"
    )

    return_value = {
        "answer": final_answer,
        "known_facts": known_facts,
        "unknown_slots": unknown_slots,
        "loop_count": loop_count,
    }
    return return_value


async def test_main():
    """Simple debug entry-point."""
    try:
        await mcp_manager.start()
    except Exception as exc:
        logger.exception(f"MCP manager failed to start — web tools will be unavailable: {exc}")

    # Dummy GlobalPersonaState-equivalent fields
    character_profile = {
        "name": "<active character>",
        "description": "一个温柔的AI角色",
    }
    user_profile = {
        "affinity": 800,
        "display_name": "<current user>",
    }

    result = await call_rag_supervisor(
        original_query="<character mention><character mention>欢迎回来",
        character_name=character_profile["name"],
        context={
            "platform": "qq",
            "platform_channel_id": "902317662",
            "user_name": user_profile.get("display_name", ""),
            "current_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )

    print("=" * 80)
    print(f"Answer:\n{result['answer']}")
    print("-" * 80)
    print(f"Loop count:       {result['loop_count']}")
    print(f"Remaining slots:  {result['unknown_slots']}")
    print("-" * 80)
    print("Known facts:")
    print(json.dumps(result["known_facts"], ensure_ascii=False, indent=2, default=str))

    await mcp_manager.stop()


if __name__ == "__main__":
    asyncio.run(test_main())
