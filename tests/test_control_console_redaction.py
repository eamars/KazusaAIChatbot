"""Sensitive-data redaction contract tests."""

from __future__ import annotations


def test_responses_exclude_secrets_prompts_embeddings_env_values_and_raw_messages() -> None:
    """Redaction should remove known high-risk fields and long raw text."""

    from control_console.redaction import redact_mapping

    source = {
        "api_key": "secret-key",
        "Authorization": "Bearer secret-token",
        "prompt": "system prompt text",
        "embedding": [0.12, 0.34],
        "env": {"MODEL_API_KEY": "secret"},
        "raw_message": "hello from a private message",
        "safe_status": "running",
    }

    redacted = redact_mapping(source)
    rendered = repr(redacted)

    assert "secret-key" not in rendered
    assert "secret-token" not in rendered
    assert "system prompt text" not in rendered
    assert "0.12" not in rendered
    assert "hello from a private message" not in rendered
    assert "api_key" not in redacted
    assert "Authorization" not in redacted
    assert "prompt" not in redacted
    assert "embedding" not in redacted
    assert "raw_message" not in redacted
    assert redacted["safe_status"] == "running"


def test_canonical_observation_sections_bypass_legacy_semantic_reprojection() -> None:
    """Canonical Brain sections should pass through generic redaction unchanged."""

    from control_console import redaction
    from control_console.redaction import redact_mapping
    from kazusa_ai_chatbot.brain_service.cognition_observation_contracts import (
        CognitionRunObservationV1,
    )
    from tests.test_control_console_contracts import _canonical_live_observation

    observation = CognitionRunObservationV1.model_validate(
        _canonical_live_observation(run_id="redaction-run")
    )
    payload = {"observation": observation.model_dump(mode="json")}

    redacted = redact_mapping(payload)

    assert redacted["observation"] == payload["observation"]
    assert not hasattr(redaction, "redact_context_consumption")
    assert not hasattr(redaction, "redact_latest_context_consumption")
