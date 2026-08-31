"""Live DSH recurrence-closure coverage for tool-result delivery."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.dsh_trigger_source_e2e_support import run_trigger_source_case


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_resolved_tool_result_delivers_without_recursive_dsh(
    tmp_path: Path,
) -> None:
    """A resolved typed result should deliver without opening another DSH task."""

    await run_trigger_source_case("tool_result_resolved", tmp_path)


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_failed_tool_result_settles_without_recursive_dsh(
    tmp_path: Path,
) -> None:
    """A failed typed result should remain failed and avoid recursive DSH."""

    await run_trigger_source_case("tool_result_failed", tmp_path)
