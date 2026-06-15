"""Tests for startup LLM route reporting."""

from __future__ import annotations

import subprocess
import sys

EXPECTED_ROUTE_TABLE_ROWS = (
    "RELEVANCE_AGENT_LLM",
    "VISION_DESCRIPTOR_LLM",
    "MSG_DECONTEXTUALIZER_LLM",
    "RAG_PLANNER_LLM",
    "RAG_SUBAGENT_LLM",
    "WEB_SEARCH_LLM",
    "COGNITION_LLM",
    "BOUNDARY_CORE_LLM",
    "DIALOG_GENERATOR_LLM",
    "DIALOG_EVALUATOR_LLM",
    "CONSOLIDATION_LLM",
    "JSON_REPAIR_LLM",
    "BACKGROUND_ARTIFACT_LLM",
    "BACKGROUND_WORK_LLM",
    "EMBEDDING",
)


def test_llm_route_inventory_contains_all_routes_once() -> None:
    """Route inventory contains each startup table route exactly once."""

    from kazusa_ai_chatbot.llm_route_report import (
        _table_rows,
        configured_route_diagnostics,
    )

    route_names = [
        row["route_name"]
        for row in _table_rows(configured_route_diagnostics())
    ]

    assert tuple(route_names) == EXPECTED_ROUTE_TABLE_ROWS
    assert len(route_names) == len(set(route_names))


def test_llm_route_inventory_uses_configured_models_and_sources() -> None:
    """Route inventory reads model and source values from config constants."""

    import kazusa_ai_chatbot.config as config
    from kazusa_ai_chatbot.llm_route_report import (
        _table_rows,
        configured_route_diagnostics,
    )

    rows_by_route = {
        row["route_name"]: row
        for row in _table_rows(configured_route_diagnostics())
    }

    for route_name in EXPECTED_ROUTE_TABLE_ROWS:
        if route_name == "EMBEDDING":
            assert rows_by_route[route_name]["model"] == config.EMBEDDING_MODEL
            assert (
                rows_by_route[route_name]["normalized_base_url"]
                == config.EMBEDDING_BASE_URL
            )
            continue

        assert rows_by_route[route_name]["model"] == getattr(config, f"{route_name}_MODEL")
        assert (
            rows_by_route[route_name]["normalized_base_url"]
            == getattr(config, f"{route_name}_BASE_URL").rstrip("/")
        )


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
        config.BOUNDARY_CORE_LLM_API_KEY,
        config.DIALOG_GENERATOR_LLM_API_KEY,
        config.DIALOG_EVALUATOR_LLM_API_KEY,
        config.CONSOLIDATION_LLM_API_KEY,
        config.JSON_REPAIR_LLM_API_KEY,
        config.BACKGROUND_ARTIFACT_LLM_API_KEY,
        config.BACKGROUND_WORK_LLM_API_KEY,
        config.EMBEDDING_API_KEY,
    )
    for api_key in api_keys:
        if api_key:
            assert api_key not in table


def test_boundary_core_node_uses_boundary_route(tmp_path) -> None:
    """Boundary Core binds its LLM client to the dedicated route."""

    from tests.test_config import _configured_subprocess_env_without_dotenv

    env = _configured_subprocess_env_without_dotenv()
    env["COGNITION_LLM_BASE_URL"] = "http://cognition.example/v1"
    env["COGNITION_LLM_MODEL"] = "cognition-model"
    env["BOUNDARY_CORE_LLM_BASE_URL"] = "http://boundary.example/v1/"
    env["BOUNDARY_CORE_LLM_MODEL"] = "boundary-model"

    result = subprocess.run(
        [
            sys.executable,
                "-c",
                (
                    "from kazusa_ai_chatbot.nodes "
                    "import persona_supervisor2_cognition as c; "
                    "config = c._boundary_core_llm_config; "
                    "print('|'.join((config.base_url.rstrip('/'), config.model)))"
                ),
            ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "http://boundary.example/v1|boundary-model"
