"""Shared types for the progressive RAG supervisor."""

from __future__ import annotations

from typing import Annotated, Awaitable, Callable, TypedDict

from langgraph.graph.message import add_messages


_MAX_LOOP_COUNT = 4



class ProgressiveRAGState(TypedDict):
    """Working state for the progressive RAG supervisor.

    Fields:
        original_query: The user's natural-language question.
        context: Optional auxiliary fields (platform, channel, ...).
        unknown_slots: Ordered slots still needing resolution; drains to empty.
        current_slot: The slot being targeted in the current iteration.
        known_facts: Slot results; each entry has slot, agent, resolved, summary, raw_result, attempts.
        messages: LangGraph message log — tool-call protocol only.
        initializer_cache: Cache metadata for the initializer strategy lookup.
        loop_count: Safety cap counter.
        final_answer: Synthesised answer from the finalizer.
        llm_trace_id: Turn-scoped trace id for protected LLM trace joins.
    """

    original_query: str
    character_name: str
    context: dict
    unknown_slots: list[str]
    current_slot: str
    known_facts: list[dict]
    messages: Annotated[list, add_messages]
    initializer_cache: dict
    current_dispatch: dict
    last_agent_result: dict
    loop_count: int
    final_answer: str
    llm_trace_id: str


RAGAgentCallable = Callable[..., Awaitable[dict]]


class RAGFactSource(TypedDict):
    """Deterministic source policy for facts returned by one RAG agent."""

    source_kind: str
    source_system: str
    consolidation_policy: str
    can_consolidate_as_new_knowledge: bool


class RAGAgentRegistryEntry(TypedDict):
    """Registry entry for a RAG agent and its provenance metadata."""

    agent: RAGAgentCallable
    fact_source: RAGFactSource
