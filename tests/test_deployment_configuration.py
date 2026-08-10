"""Deterministic deployment-artifact consistency checks."""

from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _required_config_environment_names() -> set[str]:
    """Return direct required ``os.environ`` keys from runtime config."""

    config_path = _ROOT / "src" / "kazusa_ai_chatbot" / "config.py"
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    required_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if (
            not isinstance(value, ast.Attribute)
            or value.attr != "environ"
            or not isinstance(value.value, ast.Name)
            or value.value.id != "os"
        ):
            continue
        if isinstance(node.slice, ast.Constant) and isinstance(
            node.slice.value,
            str,
        ):
            required_names.add(node.slice.value)
    return required_names


def test_compose_passes_every_required_runtime_environment_variable() -> None:
    """Container startup must receive every fail-fast runtime route."""

    compose_text = (_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    for variable_name in sorted(_required_config_environment_names()):
        expected_binding = (
            f"- {variable_name}=${{{variable_name}:?required}}"
        )
        assert expected_binding in compose_text
