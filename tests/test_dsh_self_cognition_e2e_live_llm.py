"""Live DSH non-entry coverage for targetless self-cognition sources."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.dsh_trigger_source_e2e_support import run_trigger_source_case


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_targetless_group_review_omits_dsh_task_resolution(
    tmp_path: Path,
) -> None:
    """A targetless group review should settle without invented DSH authority."""

    await run_trigger_source_case("self_cognition_targetless_group", tmp_path)


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_promoted_group_review_omits_dsh_task_resolution(
    tmp_path: Path,
) -> None:
    """Promoted reflection context should not fabricate an executable user."""

    await run_trigger_source_case("self_cognition_promoted_group", tmp_path)
