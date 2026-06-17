"""Service-registry safety tests for the control console."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


def _service(service_id: str, *, dependencies: list[str] | None = None) -> dict:
    """Build a minimal registry service document."""

    spec = {
        "id": service_id,
        "display_name": service_id,
        "kind": "backend",
        "command": ["python", "-m", "kazusa_ai_chatbot.main"],
        "cwd": ".",
        "env": {},
        "dependencies": dependencies or [],
        "health_url": "http://127.0.0.1:8000/health",
        "autostart": False,
    }
    return spec


def _write_registry(tmp_path, services: list[dict]):
    """Write a registry override and return its path."""

    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"services": services}), encoding="utf-8")
    return path


def test_registry_rejects_shell_strings_external_identifiers_duplicate_ids_and_dependency_cycles(
    tmp_path,
) -> None:
    """The registry should allow future services without arbitrary control."""

    from control_console.service_registry import (
        RegistryValidationError,
        default_service_registry,
        load_service_registry,
    )

    defaults = default_service_registry()
    assert {
        "brain",
        "adapter.discord",
        "adapter.napcat",
        "adapter.debug",
    } <= set(defaults)
    assert defaults["brain"].command[:3] == [
        sys.executable,
        "-m",
        "kazusa_ai_chatbot.main",
    ]
    assert defaults["adapter.napcat"].command == [
        sys.executable,
        "-m",
        "adapters.napcat_qq_adapter",
    ]
    assert defaults["adapter.napcat"].dependencies == ["brain"]
    assert defaults["adapter.debug"].health_url == "http://127.0.0.1:8080/api/health"

    duplicate_path = _write_registry(
        tmp_path,
        [_service("brain"), _service("brain")],
    )
    with pytest.raises(RegistryValidationError):
        load_service_registry(override_path=duplicate_path, repo_root=tmp_path)

    shell_path = _write_registry(
        tmp_path,
        [_service("brain") | {"command": ["python -m kazusa_ai_chatbot.main"]}],
    )
    with pytest.raises(RegistryValidationError):
        load_service_registry(override_path=shell_path, repo_root=tmp_path)

    shell_interpreter_path = _write_registry(
        tmp_path,
        [_service("brain") | {"command": ["powershell", "-Command", "Get-Process"]}],
    )
    with pytest.raises(RegistryValidationError):
        load_service_registry(
            override_path=shell_interpreter_path,
            repo_root=tmp_path,
        )

    inline_python_path = _write_registry(
        tmp_path,
        [_service("brain") | {"command": ["python", "-c", "print('unsafe')"]}],
    )
    with pytest.raises(RegistryValidationError):
        load_service_registry(override_path=inline_python_path, repo_root=tmp_path)

    external_path = _write_registry(
        tmp_path,
        [_service("brain") | {"process_id": 1234}],
    )
    with pytest.raises(RegistryValidationError):
        load_service_registry(override_path=external_path, repo_root=tmp_path)

    cycle_path = _write_registry(
        tmp_path,
        [
            _service("brain", dependencies=["adapter.debug"]),
            _service("adapter.debug", dependencies=["brain"]),
        ],
    )
    with pytest.raises(RegistryValidationError):
        load_service_registry(override_path=cycle_path, repo_root=tmp_path)


def test_brain_module_command_exposes_cli_help() -> None:
    """The default brain module command should invoke the uvicorn CLI wrapper."""

    result = subprocess.run(
        [sys.executable, "-m", "kazusa_ai_chatbot.main", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert "Run the Kazusa brain service" in result.stdout


def test_registry_override_rejects_invalid_json_shape_cwd_and_unknown_dependencies(
    tmp_path,
) -> None:
    """Registry overrides should fail closed before reaching the web UI."""

    from control_console.service_registry import (
        RegistryValidationError,
        load_service_registry,
    )

    missing_file = tmp_path / "missing.json"
    with pytest.raises(RegistryValidationError, match="cannot read"):
        load_service_registry(override_path=missing_file, repo_root=tmp_path)

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="not JSON"):
        load_service_registry(override_path=invalid_json, repo_root=tmp_path)

    missing_services = tmp_path / "missing-services.json"
    missing_services.write_text(json.dumps({"services": {}}), encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="services list"):
        load_service_registry(override_path=missing_services, repo_root=tmp_path)

    cwd_escape = _write_registry(
        tmp_path,
        [_service("brain") | {"cwd": ".."}],
    )
    with pytest.raises(RegistryValidationError, match="cwd escapes"):
        load_service_registry(override_path=cwd_escape, repo_root=tmp_path)

    unknown_dependency = _write_registry(
        tmp_path,
        [_service("adapter.debug", dependencies=["brain"])],
    )
    with pytest.raises(RegistryValidationError, match="unknown dependency"):
        load_service_registry(override_path=unknown_dependency, repo_root=tmp_path)
