"""Deterministic verification for the native DSH coding fixture."""

from __future__ import annotations

import os

import pytest

from calculator import total_with_tax

pytestmark = pytest.mark.skipif(
    os.environ.get("KAZUSA_RUN_DSH_FIXTURE") != "1",
    reason="fixture verification is enabled by the native coding case",
)


def test_total_with_tax() -> None:
    """Tax is added to the original price."""

    assert total_with_tax(100.0, 0.15) == 115.0
