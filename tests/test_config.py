"""Tests for config.py — environment variable loading and constants."""

from __future__ import annotations

import os
import subprocess
import sys


REQUIRED_ROUTE_ENV_VARS = (
    "RELEVANCE_AGENT_LLM_BASE_URL",
    "RELEVANCE_AGENT_LLM_API_KEY",
    "RELEVANCE_AGENT_LLM_MODEL",
    "VISION_DESCRIPTOR_LLM_BASE_URL",
    "VISION_DESCRIPTOR_LLM_API_KEY",
    "VISION_DESCRIPTOR_LLM_MODEL",
    "MSG_DECONTEXTUALIZER_LLM_BASE_URL",
    "MSG_DECONTEXTUALIZER_LLM_API_KEY",
    "MSG_DECONTEXTUALIZER_LLM_MODEL",
    "RAG_PLANNER_LLM_BASE_URL",
    "RAG_PLANNER_LLM_API_KEY",
    "RAG_PLANNER_LLM_MODEL",
    "RAG_SUBAGENT_LLM_BASE_URL",
    "RAG_SUBAGENT_LLM_API_KEY",
    "RAG_SUBAGENT_LLM_MODEL",
    "WEB_SEARCH_LLM_BASE_URL",
    "WEB_SEARCH_LLM_API_KEY",
    "WEB_SEARCH_LLM_MODEL",
    "COGNITION_LLM_BASE_URL",
    "COGNITION_LLM_API_KEY",
    "COGNITION_LLM_MODEL",
    "DIALOG_GENERATOR_LLM_BASE_URL",
    "DIALOG_GENERATOR_LLM_API_KEY",
    "DIALOG_GENERATOR_LLM_MODEL",
    "DIALOG_EVALUATOR_LLM_BASE_URL",
    "DIALOG_EVALUATOR_LLM_API_KEY",
    "DIALOG_EVALUATOR_LLM_MODEL",
    "CONSOLIDATION_LLM_BASE_URL",
    "CONSOLIDATION_LLM_API_KEY",
    "CONSOLIDATION_LLM_MODEL",
    "JSON_REPAIR_LLM_BASE_URL",
    "JSON_REPAIR_LLM_API_KEY",
    "JSON_REPAIR_LLM_MODEL",
)


def _subprocess_env_without_dotenv() -> dict[str, str]:
    """Return an environment that imports package code without reading repo .env.

    Returns:
        Environment for a subprocess whose working directory is outside the
        repository root.
    """

    env = dict(os.environ)
    python_path = env.get("PYTHONPATH", "")
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = src_path if not python_path else f"{src_path}{os.pathsep}{python_path}"
    return env


def _configured_subprocess_env_without_dotenv() -> dict[str, str]:
    """Return a subprocess env with required route settings populated.

    Returns:
        Environment that imports config without relying on the repo `.env`.
    """

    env = _subprocess_env_without_dotenv()
    for name in REQUIRED_ROUTE_ENV_VARS:
        env[name] = "configured"
    env["EMBEDDING_BASE_URL"] = "configured"
    env["EMBEDDING_API_KEY"] = "configured"
    env["EMBEDDING_MODEL"] = "configured"
    return env


class TestAffinityConstants:
    def test_affinity_default_within_bounds(self):
        from kazusa_ai_chatbot.config import AFFINITY_DEFAULT, AFFINITY_MIN, AFFINITY_MAX
        assert AFFINITY_MIN <= AFFINITY_DEFAULT <= AFFINITY_MAX

    def test_affinity_min_less_than_max(self):
        from kazusa_ai_chatbot.config import AFFINITY_MIN, AFFINITY_MAX
        assert AFFINITY_MIN < AFFINITY_MAX

    def test_affinity_min_is_zero(self):
        from kazusa_ai_chatbot.config import AFFINITY_MIN
        assert AFFINITY_MIN == 0

    def test_affinity_max_is_1000(self):
        from kazusa_ai_chatbot.config import AFFINITY_MAX
        assert AFFINITY_MAX == 1000


class TestBreakpoints:
    def test_increment_breakpoints_sorted_by_threshold(self):
        from kazusa_ai_chatbot.config import AFFINITY_INCREMENT_BREAKPOINTS
        thresholds = [bp[0] for bp in AFFINITY_INCREMENT_BREAKPOINTS]
        assert thresholds == sorted(thresholds)

    def test_decrement_breakpoints_sorted_by_threshold(self):
        from kazusa_ai_chatbot.config import AFFINITY_DECREMENT_BREAKPOINTS
        thresholds = [bp[0] for bp in AFFINITY_DECREMENT_BREAKPOINTS]
        assert thresholds == sorted(thresholds)

    def test_increment_breakpoints_all_positive_scales(self):
        from kazusa_ai_chatbot.config import AFFINITY_INCREMENT_BREAKPOINTS
        for _, scale in AFFINITY_INCREMENT_BREAKPOINTS:
            assert scale > 0

    def test_decrement_breakpoints_all_positive_scales(self):
        from kazusa_ai_chatbot.config import AFFINITY_DECREMENT_BREAKPOINTS
        for _, scale in AFFINITY_DECREMENT_BREAKPOINTS:
            assert scale > 0


class TestRetryLimits:
    def test_retry_limits_are_positive(self):
        from kazusa_ai_chatbot.config import (
            MAX_MEMORY_RETRIEVER_AGENT_RETRY,
            MAX_WEB_SEARCH_AGENT_RETRY,
            MAX_DIALOG_AGENT_RETRY,
            MAX_FACT_HARVESTER_RETRY,
        )
        assert MAX_MEMORY_RETRIEVER_AGENT_RETRY > 0
        assert MAX_WEB_SEARCH_AGENT_RETRY > 0
        assert MAX_DIALOG_AGENT_RETRY > 0
        assert MAX_FACT_HARVESTER_RETRY > 0


class TestMcpServersDefault:
    def test_mcp_servers_is_dict(self):
        from kazusa_ai_chatbot.config import MCP_SERVERS
        assert isinstance(MCP_SERVERS, dict)


class TestCache2Config:
    def test_cache2_max_entries_is_positive(self):
        from kazusa_ai_chatbot.config import RAG_CACHE2_MAX_ENTRIES
        assert RAG_CACHE2_MAX_ENTRIES > 0


class TestConversationSearchConfig:
    def test_conversation_search_top_k_defaults_are_stable(self, tmp_path):
        env = _configured_subprocess_env_without_dotenv()
        env.pop("CONVERSATION_SEARCH_DEFAULT_TOP_K", None)
        env.pop("CONVERSATION_SEARCH_MAX_TOP_K", None)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import kazusa_ai_chatbot.config as config; "
                    "print(config.CONVERSATION_SEARCH_DEFAULT_TOP_K); "
                    "print(config.CONVERSATION_SEARCH_MAX_TOP_K)"
                ),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert result.stdout.splitlines() == ["20", "50"]

    def test_conversation_search_default_top_k_reads_environment(self, tmp_path):
        env = _configured_subprocess_env_without_dotenv()
        env["CONVERSATION_SEARCH_DEFAULT_TOP_K"] = "12"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import kazusa_ai_chatbot.config as config; "
                    "print(config.CONVERSATION_SEARCH_DEFAULT_TOP_K)"
                ),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert result.stdout.strip() == "12"

    def test_conversation_search_default_top_k_rejects_zero(self, tmp_path):
        env = _configured_subprocess_env_without_dotenv()
        env["CONVERSATION_SEARCH_DEFAULT_TOP_K"] = "0"

        result = subprocess.run(
            [sys.executable, "-c", "import kazusa_ai_chatbot.config"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert "CONVERSATION_SEARCH_DEFAULT_TOP_K must be >= 1" in result.stderr

    def test_conversation_search_max_top_k_must_cover_default(self, tmp_path):
        env = _configured_subprocess_env_without_dotenv()
        env["CONVERSATION_SEARCH_DEFAULT_TOP_K"] = "30"
        env["CONVERSATION_SEARCH_MAX_TOP_K"] = "20"

        result = subprocess.run(
            [sys.executable, "-c", "import kazusa_ai_chatbot.config"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        expected_error = (
            "CONVERSATION_SEARCH_MAX_TOP_K must be >= "
            "CONVERSATION_SEARCH_DEFAULT_TOP_K"
        )
        assert expected_error in result.stderr


class TestInteractionStyleConfig:
    def test_interaction_style_limits_default_to_positive_values(self):
        from kazusa_ai_chatbot.config import (
            INTERACTION_STYLE_STORAGE_GUIDELINES_PER_FIELD_LIMIT,
            L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT,
            RELEVANCE_USER_ENGAGEMENT_GUIDELINES_LIMIT,
        )

        assert INTERACTION_STYLE_STORAGE_GUIDELINES_PER_FIELD_LIMIT == 5
        assert L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT == 5
        assert RELEVANCE_USER_ENGAGEMENT_GUIDELINES_LIMIT == 3

    def test_interaction_style_storage_guideline_limit_rejects_zero(self, tmp_path):
        env = _configured_subprocess_env_without_dotenv()
        env["INTERACTION_STYLE_STORAGE_GUIDELINES_PER_FIELD_LIMIT"] = "0"

        result = subprocess.run(
            [sys.executable, "-c", "import kazusa_ai_chatbot.config"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        expected_error = (
            "INTERACTION_STYLE_STORAGE_GUIDELINES_PER_FIELD_LIMIT must be >= 1"
        )
        assert expected_error in result.stderr

    def test_l3_interaction_style_guideline_limit_rejects_zero(self, tmp_path):
        env = _configured_subprocess_env_without_dotenv()
        env["L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT"] = "0"

        result = subprocess.run(
            [sys.executable, "-c", "import kazusa_ai_chatbot.config"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        expected_error = (
            "L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT must be >= 1"
        )
        assert expected_error in result.stderr

    def test_relevance_user_engagement_limit_rejects_negative(self, tmp_path):
        env = _configured_subprocess_env_without_dotenv()
        env["RELEVANCE_USER_ENGAGEMENT_GUIDELINES_LIMIT"] = "-1"

        result = subprocess.run(
            [sys.executable, "-c", "import kazusa_ai_chatbot.config"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        expected_error = "RELEVANCE_USER_ENGAGEMENT_GUIDELINES_LIMIT must be >= 1"
        assert expected_error in result.stderr


class TestRouteLlmConfig:
    def test_generic_chat_llm_config_is_removed(self):
        import kazusa_ai_chatbot.config as config

        assert not hasattr(config, "LLM_BASE_URL")
        assert not hasattr(config, "LLM_API_KEY")
        assert not hasattr(config, "LLM_MODEL")

    def test_all_route_config_values_are_present(self):
        import kazusa_ai_chatbot.config as config

        for name in REQUIRED_ROUTE_ENV_VARS:
            assert getattr(config, name)

    def test_missing_route_config_crashes_import(self, tmp_path):
        env = _subprocess_env_without_dotenv()
        for name in REQUIRED_ROUTE_ENV_VARS:
            env[name] = "configured"
        env["EMBEDDING_BASE_URL"] = "configured"
        env["EMBEDDING_API_KEY"] = "configured"
        env["EMBEDDING_MODEL"] = "configured"
        del env["COGNITION_LLM_MODEL"]

        result = subprocess.run(
            [sys.executable, "-c", "import kazusa_ai_chatbot.config"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert "COGNITION_LLM_MODEL" in result.stderr


class TestCognitionVisualDirectivesConfig:
    def test_visual_directives_enabled_defaults_to_true(self, tmp_path):
        env = _configured_subprocess_env_without_dotenv()
        env.pop("COGNITION_VISUAL_DIRECTIVES_ENABLED", None)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import kazusa_ai_chatbot.config as config; "
                    "print(config.COGNITION_VISUAL_DIRECTIVES_ENABLED)"
                ),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert result.stdout.strip() == "True"

    def test_visual_directives_enabled_parses_false(self, tmp_path):
        env = _configured_subprocess_env_without_dotenv()
        env["COGNITION_VISUAL_DIRECTIVES_ENABLED"] = "false"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import kazusa_ai_chatbot.config as config; "
                    "print(config.COGNITION_VISUAL_DIRECTIVES_ENABLED)"
                ),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert result.stdout.strip() == "False"


class TestReflectionCycleConfig:
    def test_reflection_cycle_enabled_defaults_to_true(self, tmp_path):
        env = _configured_subprocess_env_without_dotenv()
        env.pop("REFLECTION_CYCLE_ENABLED", None)
        env.pop("REFLECTION_" + "CYCLE_DISABLED", None)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import kazusa_ai_chatbot.config as config; "
                    "print(config.REFLECTION_CYCLE_ENABLED)"
                ),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert result.stdout.strip() == "True"

    def test_reflection_cycle_enabled_parses_false(self, tmp_path):
        env = _configured_subprocess_env_without_dotenv()
        env["REFLECTION_CYCLE_ENABLED"] = "false"
        env.pop("REFLECTION_" + "CYCLE_DISABLED", None)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import kazusa_ai_chatbot.config as config; "
                    "print(config.REFLECTION_CYCLE_ENABLED)"
                ),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert result.stdout.strip() == "False"

    def test_removed_reflection_disable_env_is_ignored(self, tmp_path):
        env = _configured_subprocess_env_without_dotenv()
        env.pop("REFLECTION_CYCLE_ENABLED", None)
        env["REFLECTION_" + "CYCLE_DISABLED"] = "true"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import kazusa_ai_chatbot.config as config; "
                    "print(config.REFLECTION_CYCLE_ENABLED)"
                ),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert result.stdout.strip() == "True"

    def test_removed_reflection_flags_are_absent(self):
        import kazusa_ai_chatbot.config as config

        assert not hasattr(config, "REFLECTION_" + "CYCLE_DISABLED")
        assert not hasattr(config, "REFLECTION_" + "CONTEXT_ENABLED")
