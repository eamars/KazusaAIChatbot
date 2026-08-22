"""Tests for startup LLM route reporting."""

from __future__ import annotations

import json
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
    "COGNITION_LLM_CHARACTER_CARRYOVER",
    "COGNITION_V3_CHAIN_LLM",
    "DIALOG_GENERATOR_LLM",
    "CONSOLIDATION_LLM",
    "JSON_REPAIR_LLM",
    "BACKGROUND_WORK_LLM",
    "CODING_AGENT_PM_LLM",
    "CODING_AGENT_PROGRAMMER_LLM",
    "CODING_AGENT_ACTION_LOOP_LLM",
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
    action_loop_diagnostic = next(
        diagnostic
        for diagnostic in configured_route_diagnostics()
        if diagnostic.route_name == "CODING_AGENT_ACTION_LOOP_LLM"
    )
    assert action_loop_diagnostic.required is False


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
    settings = config.get_cognition_v3_route_settings()

    for route_name in EXPECTED_ROUTE_TABLE_ROWS:
        if route_name == "EMBEDDING":
            assert rows_by_route[route_name]["model"] == config.EMBEDDING_MODEL
            assert (
                rows_by_route[route_name]["normalized_base_url"]
                == config.EMBEDDING_BASE_URL
            )
            continue

        if route_name == "COGNITION_V3_CHAIN_LLM":
            setting = settings.chain
        else:
            setting = None
        if setting is None:
            assert (
                rows_by_route[route_name]["model"]
                == getattr(config, f"{route_name}_MODEL")
            )
            assert (
                rows_by_route[route_name]["normalized_base_url"]
                == getattr(config, f"{route_name}_BASE_URL").rstrip("/")
            )
        else:
            assert rows_by_route[route_name]["model"] == setting.model
            assert (
                rows_by_route[route_name]["normalized_base_url"]
                == setting.base_url.rstrip("/")
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

    settings = config.get_cognition_v3_route_settings()
    api_keys = (
        *(
            settings.chain.api_key
            if route_name == "COGNITION_V3_CHAIN_LLM"
            else getattr(config, f"{route_name}_API_KEY")
            for route_name in EXPECTED_ROUTE_TABLE_ROWS
            if route_name != "EMBEDDING"
        ),
        config.EMBEDDING_API_KEY,
    )
    for api_key in api_keys:
        if api_key:
            assert api_key not in table


def test_route_report_includes_only_v3_cognition_routes_and_shared_generic_cognition_route(
    tmp_path,
) -> None:
    """Diagnostics include one selected core family plus shared cognition."""

    from tests.test_config import (
        _v3_configured_subprocess_env_without_dotenv,
    )

    script = (
        "import json\n"
        "from kazusa_ai_chatbot.llm_interface.route_report import "
        "_table_rows, configured_route_diagnostics\n"
        "print(json.dumps(_table_rows(configured_route_diagnostics())))\n"
    )
    v3_env = _v3_configured_subprocess_env_without_dotenv(
        include_sidecar=True,
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=v3_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    v3_rows = {
        row["route_name"]
        : row
        for row in json.loads(result.stdout)
    }

    assert v3_rows["COGNITION_V3_CHAIN_LLM"]["route_group"] == (
        "v3_cognition"
    )
    assert v3_rows["COGNITION_V3_SIDECAR_LLM"]["route_group"] == (
        "v3_cognition"
    )
    assert v3_rows["COGNITION_LLM"]["route_group"] == "shared_non_core"
    assert "COGNITION_LLM_CHARACTER_CARRYOVER" in v3_rows
