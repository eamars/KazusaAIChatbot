"""LangGraph construction helpers for the brain service."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from kazusa_ai_chatbot.state import IMProcessState


def validate_v2_terminal_state(state: IMProcessState) -> dict:
    """Fail closed unless persona cognition committed before terminal handling."""

    if state.get("cognition_state_committed") is not True:
        raise ValueError("persona V2 state was not committed before terminal handling")
    if not isinstance(state.get("cognition_core_output"), dict):
        raise ValueError("persona V2 cognition output is missing at terminal handling")
    if not isinstance(state.get("cognition_state_update"), dict):
        raise ValueError("persona V2 state update is missing at terminal handling")
    return_value: dict = {}
    return return_value


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
    graph.add_node("validate_v2_terminal_state", validate_v2_terminal_state)

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
    graph.add_edge("persona_supervisor2", "validate_v2_terminal_state")
    graph.add_edge("validate_v2_terminal_state", END)

    compiled_graph = graph.compile()
    return compiled_graph
