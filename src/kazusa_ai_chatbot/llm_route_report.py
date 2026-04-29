"""Startup reporting for configured LLM routes."""

from __future__ import annotations

from kazusa_ai_chatbot.config import (
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MODEL,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
    DIALOG_EVALUATOR_LLM_BASE_URL,
    DIALOG_EVALUATOR_LLM_MODEL,
    DIALOG_GENERATOR_LLM_BASE_URL,
    DIALOG_GENERATOR_LLM_MODEL,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    JSON_REPAIR_LLM_BASE_URL,
    JSON_REPAIR_LLM_MODEL,
    MSG_DECONTEXTUALIZER_LLM_BASE_URL,
    MSG_DECONTEXTUALIZER_LLM_MODEL,
    RAG_PLANNER_LLM_BASE_URL,
    RAG_PLANNER_LLM_MODEL,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
    RELEVANCE_AGENT_LLM_BASE_URL,
    RELEVANCE_AGENT_LLM_MODEL,
    VISION_DESCRIPTOR_LLM_BASE_URL,
    VISION_DESCRIPTOR_LLM_MODEL,
    WEB_SEARCH_LLM_BASE_URL,
    WEB_SEARCH_LLM_MODEL,
)

LLM_ROUTE_CONFIGS: tuple[dict[str, str], ...] = (
    {
        "route": "RELEVANCE_AGENT_LLM",
        "model": RELEVANCE_AGENT_LLM_MODEL,
        "source_url": RELEVANCE_AGENT_LLM_BASE_URL,
    },
    {
        "route": "VISION_DESCRIPTOR_LLM",
        "model": VISION_DESCRIPTOR_LLM_MODEL,
        "source_url": VISION_DESCRIPTOR_LLM_BASE_URL,
    },
    {
        "route": "MSG_DECONTEXTUALIZER_LLM",
        "model": MSG_DECONTEXTUALIZER_LLM_MODEL,
        "source_url": MSG_DECONTEXTUALIZER_LLM_BASE_URL,
    },
    {
        "route": "RAG_PLANNER_LLM",
        "model": RAG_PLANNER_LLM_MODEL,
        "source_url": RAG_PLANNER_LLM_BASE_URL,
    },
    {
        "route": "RAG_SUBAGENT_LLM",
        "model": RAG_SUBAGENT_LLM_MODEL,
        "source_url": RAG_SUBAGENT_LLM_BASE_URL,
    },
    {
        "route": "WEB_SEARCH_LLM",
        "model": WEB_SEARCH_LLM_MODEL,
        "source_url": WEB_SEARCH_LLM_BASE_URL,
    },
    {
        "route": "COGNITION_LLM",
        "model": COGNITION_LLM_MODEL,
        "source_url": COGNITION_LLM_BASE_URL,
    },
    {
        "route": "DIALOG_GENERATOR_LLM",
        "model": DIALOG_GENERATOR_LLM_MODEL,
        "source_url": DIALOG_GENERATOR_LLM_BASE_URL,
    },
    {
        "route": "DIALOG_EVALUATOR_LLM",
        "model": DIALOG_EVALUATOR_LLM_MODEL,
        "source_url": DIALOG_EVALUATOR_LLM_BASE_URL,
    },
    {
        "route": "CONSOLIDATION_LLM",
        "model": CONSOLIDATION_LLM_MODEL,
        "source_url": CONSOLIDATION_LLM_BASE_URL,
    },
    {
        "route": "JSON_REPAIR_LLM",
        "model": JSON_REPAIR_LLM_MODEL,
        "source_url": JSON_REPAIR_LLM_BASE_URL,
    },
    {
        "route": "EMBEDDING",
        "model": EMBEDDING_MODEL,
        "source_url": EMBEDDING_BASE_URL,
    },
)


def render_llm_route_table() -> str:
    """Render configured LLM routes for startup logs.

    Returns:
        A fixed-width table containing route name, model, and source URL.
        API keys are intentionally excluded.
    """

    route_width = max(len("Route"), *(len(row["route"]) for row in LLM_ROUTE_CONFIGS))
    model_width = max(len("Model"), *(len(row["model"]) for row in LLM_ROUTE_CONFIGS))
    source_width = max(len("Source"), *(len(row["source_url"]) for row in LLM_ROUTE_CONFIGS))
    lines = [
        "Configured model routes:",
        f'{"Route":<{route_width}}  {"Model":<{model_width}}  {"Source":<{source_width}}',
    ]
    for row in LLM_ROUTE_CONFIGS:
        line = (
            f'{row["route"]:<{route_width}}  '
            f'{row["model"]:<{model_width}}  '
            f'{row["source_url"]:<{source_width}}'
        )
        lines.append(line)
    table = "\n".join(lines)
    return table
