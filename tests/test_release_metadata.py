"""Deterministic checks for the v1.0 product release identity."""

from __future__ import annotations

from pathlib import Path

from control_console import __version__ as console_version
from kazusa_ai_chatbot.version import __version__ as package_version


_ROOT = Path(__file__).resolve().parents[1]


def test_product_release_identity_is_consistent() -> None:
    """The package, console, and build metadata must expose one version."""

    pyproject_text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert package_version == "1.0.0"
    assert console_version == package_version
    assert 'dynamic = ["version"]' in pyproject_text
    assert (
        'version = {attr = "kazusa_ai_chatbot.version.__version__"}'
        in pyproject_text
    )
