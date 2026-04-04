"""LangGraph StateGraph wiring.

Connects pipeline stages into a compiled graph:

  intake → relevance_agent → persona_supervisor → speech_agent → END

memory_writer is NOT in this graph — it runs as a fire-and-forget
async task in the Discord bot layer after the reply has been sent.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from kazusa_ai_chatbot.agents.base import AGENT_REGISTRY, register_agent
from kazusa_ai_chatbot.agents.speech_agent import speech_agent
from kazusa_ai_chatbot.agents.web_search_agent import WebSearchAgent
from kazusa_ai_chatbot.nodes.intake import intake
from kazusa_ai_chatbot.nodes.persona_supervisor import persona_supervisor
from kazusa_ai_chatbot.nodes.relevance_agent import relevance_agent
from kazusa_ai_chatbot.state import BotState

logger = logging.getLogger(__name__)

# ── Register Agents ───────────────────────────────────────────────────
# We register agents here so the persona supervisor knows about them.

if "web_search_agent" not in AGENT_REGISTRY:
    register_agent(WebSearchAgent())


def _should_respond_after_intake(state: BotState) -> list[str]:
    """Conditional edge after intake — skip everything if nothing to respond to."""
    if state.get("should_respond"):
        return ["relevance_agent"]
    return [END]


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph pipeline."""
    graph = StateGraph(BotState)

    # ── Register nodes ──────────────────────────────────────────────
    graph.add_node("intake", intake)
    graph.add_node("relevance_agent", relevance_agent)
    graph.add_node("persona_supervisor", persona_supervisor)
    graph.add_node("speech_agent", speech_agent)

    # ── Edges ───────────────────────────────────────────────────────
    # Entry
    graph.add_edge(START, "intake")

    # After intake, check if we should respond
    graph.add_conditional_edges("intake", _should_respond_after_intake, ["relevance_agent", END])

    # Relevance Agent → Supervisor → Speech Agent → END
    graph.add_edge("relevance_agent", "persona_supervisor")
    graph.add_edge("persona_supervisor", "speech_agent")
    graph.add_edge("speech_agent", END)

    return graph.compile()
