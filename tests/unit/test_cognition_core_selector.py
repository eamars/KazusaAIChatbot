"""Deterministic tests for the closed process-level engine selector."""

from __future__ import annotations

import subprocess
import sys

import pytest

from kazusa_ai_chatbot.cognition_core_selector import (
    _ENGINE_MODULE_NAMES,
    resolve_engine_module,
)
from tests.test_config import _configured_subprocess_env_without_dotenv


def test_selector_resolves_exact_v2_and_v3_modules():
    assert _ENGINE_MODULE_NAMES == {
        "v2": "kazusa_ai_chatbot.cognition_core_v2",
        "v3": "kazusa_ai_chatbot.cognition_core_v3",
    }

    v2_module = resolve_engine_module("v2")
    assert v2_module is sys.modules["kazusa_ai_chatbot.cognition_core_v2"]

    v3_module = resolve_engine_module("v3")
    assert v3_module is sys.modules["kazusa_ai_chatbot.cognition_core_v3"]


def test_selector_rejects_unknown_engine(tmp_path):
    with pytest.raises(ValueError) as exc_info:
        resolve_engine_module("nope")

    message = str(exc_info.value)
    assert "v2" in message and "v3" in message

    env = _configured_subprocess_env_without_dotenv()
    env["COGNITION_CORE_ENGINE"] = "nope"

    result = subprocess.run(
        [sys.executable, "-c", "import kazusa_ai_chatbot.cognition_core_selector"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "must be one of" in result.stderr


def test_v3_failure_never_invokes_v2(tmp_path):
    # Pre-seed a failing v3 entrypoint and prove the selector neither falls
    # back to V2 nor masks the selected-engine failure.
    script_lines = [
        "import sys",
        "import types",
        'stub = types.ModuleType("kazusa_ai_chatbot.cognition_core_v3")',
        "def _fail():",
        '    raise RuntimeError("sentinel-v3-failure")',
        "stub.run_cognition = _fail",
        'sys.modules["kazusa_ai_chatbot.cognition_core_v3"] = stub',
        "import kazusa_ai_chatbot.cognition_core_selector as selector",
        "try:",
        "    selector.run_cognition()",
        "except RuntimeError as exc:",
        '    assert str(exc) == "sentinel-v3-failure"',
        'assert "kazusa_ai_chatbot.cognition_core_v2" not in sys.modules',
        'print("OK")',
    ]

    env = _configured_subprocess_env_without_dotenv()
    env["COGNITION_CORE_ENGINE"] = "v3"

    result = subprocess.run(
        [sys.executable, "-c", "\n".join(script_lines)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
