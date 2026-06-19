"""Settings loading tests for the control console."""

from __future__ import annotations


def test_settings_load_control_values_from_dotenv_with_environment_override(
    monkeypatch,
    tmp_path,
) -> None:
    """Control settings should read `.env` defaults before process overrides."""

    from control_console.settings import ControlConsoleSettings

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join([
            "KAZUSA_CONTROL_OPERATOR_TOKEN_HASH=hash-from-dotenv",
            "KAZUSA_CONTROL_PORT=8767",
            "KAZUSA_CONTROL_STATE_DIR=.console-from-dotenv",
        ]),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KAZUSA_CONTROL_OPERATOR_TOKEN_HASH", raising=False)
    monkeypatch.delenv("KAZUSA_CONTROL_PORT", raising=False)
    monkeypatch.delenv("KAZUSA_CONTROL_STATE_DIR", raising=False)

    dotenv_settings = ControlConsoleSettings.from_env()

    assert dotenv_settings.operator_token_hash == "hash-from-dotenv"
    assert dotenv_settings.port == 8767
    assert dotenv_settings.state_dir.name == ".console-from-dotenv"

    monkeypatch.setenv("KAZUSA_CONTROL_OPERATOR_TOKEN_HASH", "hash-from-env")
    monkeypatch.setenv("KAZUSA_CONTROL_PORT", "8768")
    monkeypatch.setenv("KAZUSA_CONTROL_STATE_DIR", ".console-from-env")

    injected_settings = ControlConsoleSettings.from_env()

    assert injected_settings.operator_token_hash == "hash-from-env"
    assert injected_settings.port == 8768
    assert injected_settings.state_dir.name == ".console-from-env"
