"""Tests for startup LLM route reporting."""

from __future__ import annotations

import subprocess
import sys

from kazusa_ai_chatbot.llm_interface.diagnostics import RouteDiagnostic

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
    "CONSOLIDATION_LLM",
    "JSON_REPAIR_LLM",
    "BACKGROUND_WORK_LLM",
    "CODING_AGENT_PM_LLM",
    "CODING_AGENT_PROGRAMMER_LLM",
    "EMBEDDING",
)


def test_llm_route_inventory_contains_all_routes_once() -> None:
    """Route inventory contains each startup table route exactly once."""

    from kazusa_ai_chatbot.llm_interface.route_report import (
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
    from kazusa_ai_chatbot.llm_interface.route_report import (
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

        assert (
            rows_by_route[route_name]["model"]
            == getattr(config, f"{route_name}_MODEL")
        )
        assert (
            rows_by_route[route_name]["normalized_base_url"]
            == getattr(config, f"{route_name}_BASE_URL").rstrip("/")
        )


def test_llm_route_inventory_renders_optional_feature_tags() -> None:
    """Route inventory shows compact optional feature tags."""

    from kazusa_ai_chatbot.llm_interface.route_report import _table_rows

    diagnostics = (
        RouteDiagnostic(
            route_name="GEMMA_THINKING_LLM",
            backend="openai_compatible",
            model="gemma-4-27b-it",
            normalized_base_url="http://localhost:1234/v1",
            model_family="gemma4",
            thinking_strategy="gemma4_enabled",
            required=True,
            fallback_backed=False,
        ),
        RouteDiagnostic(
            route_name="QWEN_THINKING_LLM",
            backend="openai_compatible",
            model="qwen3.6-34b",
            normalized_base_url="http://localhost:1234/v1",
            model_family="qwen",
            thinking_strategy="qwen3_enabled",
            required=True,
            fallback_backed=False,
        ),
        RouteDiagnostic(
            route_name="UNSUPPORTED_THINKING_LLM",
            backend="openai_compatible",
            model="qwen2.5-32b",
            normalized_base_url="http://localhost:1234/v1",
            model_family="qwen",
            thinking_strategy="ignored_unsupported_model",
            required=True,
            fallback_backed=False,
        ),
        RouteDiagnostic(
            route_name="BACKGROUND_THINKING_LLM",
            backend="openai_compatible",
            model="gemma-4-27b-it",
            normalized_base_url="http://localhost:1234/v1",
            model_family="gemma4",
            thinking_strategy="gemma4_enabled",
            required=False,
            fallback_backed=True,
        ),
        RouteDiagnostic(
            route_name="PLAIN_LLM",
            backend="openai_compatible",
            model="plain-model",
            normalized_base_url="http://localhost:1234/v1",
            model_family="unknown",
            thinking_strategy="disabled",
            required=True,
            fallback_backed=False,
        ),
    )
    rows_by_route = {
        row["route_name"]: row
        for row in _table_rows(diagnostics)
    }

    assert (
        rows_by_route["GEMMA_THINKING_LLM"]["optional_feature"]
        == "thinking_on"
    )
    assert (
        rows_by_route["QWEN_THINKING_LLM"]["optional_feature"]
        == "thinking_on"
    )
    assert (
        rows_by_route["UNSUPPORTED_THINKING_LLM"]["optional_feature"]
        == "-"
    )
    assert (
        rows_by_route["BACKGROUND_THINKING_LLM"]["optional_feature"]
        == "thinking_on"
    )
    assert rows_by_route["PLAIN_LLM"]["optional_feature"] == "-"
    assert rows_by_route["EMBEDDING"]["optional_feature"] == "-"


def test_llm_route_table_omits_api_keys() -> None:
    """Rendered startup table includes route values but excludes API keys."""

    import kazusa_ai_chatbot.config as config
    from kazusa_ai_chatbot.llm_interface.route_report import (
        render_llm_route_table,
    )

    table = render_llm_route_table()

    assert "Configured model routes:" in table
    assert "Route" in table
    assert "Model" in table
    assert "Source" in table
    assert "Optional Feature" in table
    header = table.splitlines()[1]
    assert "Backend" not in header
    assert "Family" not in header
    assert "Thinking" not in header
    assert "Required" not in header
    assert "Fallback" not in header
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
        config.CONSOLIDATION_LLM_API_KEY,
        config.JSON_REPAIR_LLM_API_KEY,
        config.BACKGROUND_WORK_LLM_API_KEY,
        config.CODING_AGENT_PM_LLM_API_KEY,
        config.CODING_AGENT_PROGRAMMER_LLM_API_KEY,
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
