"""Tests for background artifact public runtime entrypoints."""

from __future__ import annotations

import importlib
from pathlib import Path


def test_background_artifact_public_entrypoints_exist() -> None:
    """Service and action execution should import only the public entrypoint."""

    module = importlib.import_module("kazusa_ai_chatbot.background_artifact")

    for name in (
        "BackgroundArtifactQueueRequest",
        "BackgroundArtifactQueueResult",
        "BackgroundArtifactRuntimeHandle",
        "enqueue_background_artifact_request",
        "start_background_artifact_runtime",
        "stop_background_artifact_runtime",
    ):
        assert hasattr(module, name)


def test_action_handler_uses_public_background_artifact_entrypoint() -> None:
    """Action-spec integration should use the public artifact facade."""

    handler_path = Path(
        "src/kazusa_ai_chatbot/action_spec/handlers/background_artifact.py"
    )
    source = handler_path.read_text(encoding="utf-8")

    assert "kazusa_ai_chatbot.background_artifact.jobs" not in source
    assert "kazusa_ai_chatbot.background_artifact import" in source


def test_background_artifact_readme_has_icd_sections() -> None:
    """The new subsystem must document its interface boundary."""

    readme_path = Path("src/kazusa_ai_chatbot/background_artifact/README.md")

    assert readme_path.exists()
    readme_text = readme_path.read_text(encoding="utf-8")
    for heading in (
        "Document Control",
        "Purpose",
        "Scope",
        "Parties",
        "Boundary Summary",
        "Public Interface",
        "Job Lifecycle",
        "LLM Input Contract",
        "Forbidden Paths",
    ):
        assert heading in readme_text
