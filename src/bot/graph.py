"""LangGraph StateGraph wiring.

Connects pipeline stages into a compiled graph:

  intake → router →┬→ rag_retriever    ─┬→ assembler → persona_agent → END
                   └→ memory_retriever ─┘

memory_writer is NOT in this graph — it runs as a fire-and-forget
async task in the Discord bot layer after the reply has been sent.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from bot.nodes.assembler import assembler
from bot.nodes.intake import intake
from bot.nodes.memory import memory_retriever
from bot.nodes.persona import persona_agent
from bot.nodes.rag import rag_retriever
from bot.nodes.router import router
from bot.state import BotState


def _should_respond(state: BotState) -> str:
    """Conditional edge after intake — skip everything if nothing to respond to."""
    if state.get("should_respond"):
        return "router"
    return END


def _retrieval_fan_out(state: BotState) -> list[str]:
    """Determine which retrieval nodes to run (parallel)."""
    targets = []
    if state.get("retrieve_rag"):
        targets.append("rag_retriever")
    if state.get("retrieve_memory"):
        targets.append("memory_retriever")
    # If neither is needed, go straight to assembler
    if not targets:
        targets.append("assembler")
    return targets


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph pipeline."""
    graph = StateGraph(BotState)

    # ── Register nodes ──────────────────────────────────────────────
    graph.add_node("intake", intake)
    graph.add_node("router", router)
    graph.add_node("rag_retriever", rag_retriever)
    graph.add_node("memory_retriever", memory_retriever)
    graph.add_node("assembler", assembler)
    graph.add_node("persona_agent", persona_agent)

    # ── Edges ───────────────────────────────────────────────────────
    # Entry
    graph.add_edge(START, "intake")

    # After intake, check if we should respond
    graph.add_conditional_edges("intake", _should_respond, ["router", END])

    # After router, fan out to retrieval nodes
    graph.add_conditional_edges(
        "router",
        _retrieval_fan_out,
        ["rag_retriever", "memory_retriever", "assembler"],
    )

    # Both retrievers converge on assembler
    graph.add_edge("rag_retriever", "assembler")
    graph.add_edge("memory_retriever", "assembler")

    # Assembler → Persona Agent → END
    graph.add_edge("assembler", "persona_agent")
    graph.add_edge("persona_agent", END)

    return graph.compile()
