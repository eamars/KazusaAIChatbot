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
