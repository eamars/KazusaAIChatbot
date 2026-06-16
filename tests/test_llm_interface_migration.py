from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot import utils as utils_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "kazusa_ai_chatbot"


def _python_sources() -> list[Path]:
    """Return production Python sources for static migration checks."""

    sources = sorted(SRC_ROOT.rglob("*.py"))
    return sources


def test_production_sources_no_longer_call_get_llm() -> None:
    """The big-bang cutover removes get_llm from production code."""

    offenders: list[str] = []
    for source_path in _python_sources():
        text = source_path.read_text(encoding="utf-8")
        if "get_llm(" in text or "import get_llm" in text:
            offenders.append(str(source_path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def test_chatopenai_construction_stays_inside_provider_adapter() -> None:
    """Provider-specific imports must not leak into stage modules."""

    allowed_path = (
        SRC_ROOT
        / "llm_interface"
        / "providers"
        / "openai_compatible.py"
    )
    offenders: list[str] = []
    for source_path in _python_sources():
        text = source_path.read_text(encoding="utf-8")
        if "ChatOpenAI" not in text:
            continue
        if source_path != allowed_path:
            offenders.append(str(source_path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def test_legacy_reload_wrapper_names_are_removed_from_production_imports() -> None:
    """The old monitored_chat_model API is not a shipped compatibility layer."""

    offenders: list[str] = []
    forbidden_markers = (
        "MonitoredChatModel",
        "monitored_chat_model",
        "llm_reload_monitor",
    )
    for source_path in _python_sources():
        if "llm_reload_monitor.py" in str(source_path):
            continue
        text = source_path.read_text(encoding="utf-8")
        if any(marker in text for marker in forbidden_markers):
            offenders.append(str(source_path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def test_config_does_not_own_backend_kind_or_model_family() -> None:
    """Backend and model-family detection live in LLInterface, not config."""

    config_text = (SRC_ROOT / "config.py").read_text(encoding="utf-8")

    assert "_BACKEND_KIND" not in config_text
    assert "_MODEL_FAMILY" not in config_text


def test_temporary_message_equivalence_file_is_not_shipped() -> None:
    """Migration evidence tests must be deleted before final completion."""

    temporary_test = PROJECT_ROOT / "tests" / "test_llm_interface_message_equivalence.py"

    assert not temporary_test.exists()


def test_json_repair_call_preserves_messages_and_route_config(monkeypatch) -> None:
    """A representative migrated sync call keeps message content and config."""

    captured: dict[str, object] = {}

    class _RepairLLM:
        """Capture the sync interface call without invoking a real model."""

        def invoke(self, messages, *, config):
            captured["messages"] = list(messages)
            captured["config"] = config
            response = SimpleNamespace(content='{"tool_calls": []}')
            return response

    expected_format = """{
  "tool_calls": []
}"""
    monkeypatch.setattr(utils_module, "_parse_json_with_llm", _RepairLLM())

    result = utils_module.parse_json_with_llm(
        "[]",
        expected_output_format=expected_format,
    )

    messages = captured["messages"]
    config = captured["config"]
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[0].content == utils_module._build_json_repair_prompt(
        expected_format
    )
    assert json.loads(messages[1].content) == {"broken_json": "[]"}
    assert config is utils_module._parse_json_with_llm_config
    assert config.route_name == "JSON_REPAIR_LLM"
    assert config.max_completion_tokens == (
        utils_module.JSON_REPAIR_LLM_MAX_COMPLETION_TOKENS
    )
    assert result == {"tool_calls": []}
