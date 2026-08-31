"""Live DSH sign-off coverage for durable internal-thought latches."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.dsh_trigger_source_e2e_support import run_trigger_source_case


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_internal_thought_file_check_reaches_dsh(
    tmp_path: Path,
) -> None:
    """A bound latch should claim, resolve its local fact, and be consumed."""

    await run_trigger_source_case("internal_thought_file_check", tmp_path)


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_internal_thought_comparison_reaches_dsh(
    tmp_path: Path,
) -> None:
    """A bound comparison latch should use one grounded DSH continuation."""

    await run_trigger_source_case("internal_thought_comparison", tmp_path)
