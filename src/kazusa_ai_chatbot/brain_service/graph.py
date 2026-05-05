"""LangGraph construction helpers for the brain service."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from kazusa_ai_chatbot.state import IMProcessState


def build_graph(
    *,
    relevance_agent_node,
    multimedia_descriptor_agent_node,
    load_conversation_episode_state_node,
    persona_supervisor_node,
):
    """Build the chat-processing graph from service-supplied node callables.

    Args:
        relevance_agent_node: Node that decides whether the bot should answer.
        multimedia_descriptor_agent_node: Node that describes image inputs.
        load_conversation_episode_state_node: Node that loads progress state.
        persona_supervisor_node: Node that produces final dialog state.

    Returns:
        Compiled LangGraph runnable used by the brain service.
    """

    graph = StateGraph(IMProcessState)

    graph.add_node("relevance_agent", relevance_agent_node)
    graph.add_node("multimedia_descriptor_agent", multimedia_descriptor_agent_node)
    graph.add_node(
        "load_conversation_episode_state",
        load_conversation_episode_state_node,
    )
    graph.add_node("persona_supervisor2", persona_supervisor_node)

    def _start_router(state):
        debug = state.get("debug_modes") or {}
        if debug.get("listen_only"):
            return "end"
        if state.get("user_multimedia_input"):
            return "multimedia"
        return "skip"

    graph.add_conditional_edges(
        START,
        _start_router,
        {
            "multimedia": "multimedia_descriptor_agent",
            "skip": "relevance_agent",
            "end": END,
        },
    )
    graph.add_edge("multimedia_descriptor_agent", "relevance_agent")

    def _route_after_relevance(state):
        if not state.get("should_respond"):
            return "end"
        return "continue"

    graph.add_conditional_edges(
        "relevance_agent",
        _route_after_relevance,
        {"continue": "load_conversation_episode_state", "end": END},
    )
    graph.add_edge("load_conversation_episode_state", "persona_supervisor2")
    graph.add_edge("persona_supervisor2", END)

    compiled_graph = graph.compile()
    return compiled_graph
