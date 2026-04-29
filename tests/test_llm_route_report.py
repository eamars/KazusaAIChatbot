"""Tests for startup LLM route reporting."""

from __future__ import annotations

EXPECTED_ROUTE_TABLE_ROWS = (
    "RELEVANCE_AGENT_LLM",
    "VISION_DESCRIPTOR_LLM",
    "MSG_DECONTEXTUALIZER_LLM",
    "RAG_PLANNER_LLM",
    "RAG_SUBAGENT_LLM",
    "WEB_SEARCH_LLM",
    "COGNITION_LLM",
    "DIALOG_GENERATOR_LLM",
    "DIALOG_EVALUATOR_LLM",
    "CONSOLIDATION_LLM",
    "JSON_REPAIR_LLM",
    "EMBEDDING",
)


def test_llm_route_inventory_contains_all_routes_once() -> None:
    """Route inventory contains each startup table route exactly once."""

    from kazusa_ai_chatbot.llm_route_report import LLM_ROUTE_CONFIGS

    route_names = [row["route"] for row in LLM_ROUTE_CONFIGS]

    assert tuple(route_names) == EXPECTED_ROUTE_TABLE_ROWS
    assert len(route_names) == len(set(route_names))


def test_llm_route_inventory_uses_configured_models_and_sources() -> None:
    """Route inventory reads model and source values from config constants."""

    import kazusa_ai_chatbot.config as config
    from kazusa_ai_chatbot.llm_route_report import LLM_ROUTE_CONFIGS

    rows_by_route = {
        row["route"]: row
        for row in LLM_ROUTE_CONFIGS
    }

    for route_name in EXPECTED_ROUTE_TABLE_ROWS:
        if route_name == "EMBEDDING":
            assert rows_by_route[route_name]["model"] == config.EMBEDDING_MODEL
            assert rows_by_route[route_name]["source_url"] == config.EMBEDDING_BASE_URL
            continue

        assert rows_by_route[route_name]["model"] == getattr(config, f"{route_name}_MODEL")
        assert rows_by_route[route_name]["source_url"] == getattr(config, f"{route_name}_BASE_URL")


def test_llm_route_table_omits_api_keys() -> None:
    """Rendered startup table includes route values but excludes API keys."""

    import kazusa_ai_chatbot.config as config
    from kazusa_ai_chatbot.llm_route_report import render_llm_route_table

    table = render_llm_route_table()

    assert "Configured model routes:" in table
    assert "Route" in table
    assert "Model" in table
    assert "Source" in table
    for route_name in EXPECTED_ROUTE_TABLE_ROWS:
        assert route_name in table

    api_keys = (
        config.RELEVANCE_AGENT_LLM_API_KEY,
        config.VISION_DESCRIPTOR_LLM_API_KEY,
        config.MSG_DECONTEXTUALIZER_LLM_API_KEY,
        config.RAG_PLANNER_LLM_API_KEY,
        config.RAG_SUBAGENT_LLM_API_KEY,
        config.WEB_SEARCH_LLM_API_KEY,
        config.COGNITION_LLM_API_KEY,
        config.DIALOG_GENERATOR_LLM_API_KEY,
        config.DIALOG_EVALUATOR_LLM_API_KEY,
        config.CONSOLIDATION_LLM_API_KEY,
        config.JSON_REPAIR_LLM_API_KEY,
        config.EMBEDDING_API_KEY,
    )
    for api_key in api_keys:
        if api_key:
            assert api_key not in table
