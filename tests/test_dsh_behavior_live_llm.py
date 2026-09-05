"""Three user- and character-level real-model DSH behavior contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.dsh_behavior_e2e_support import run_live_behavior_case


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_foreground_task_resolution_is_grounded_and_character_owned(
    tmp_path: Path,
) -> None:
    """A natural foreground request resolves from evidence into character dialog."""

    await run_live_behavior_case("foreground", tmp_path)


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_deferred_task_result_recurs_and_delivers_once(
    tmp_path: Path,
) -> None:
    """A natural background request returns one grounded result delivery."""

    await run_live_behavior_case("deferred", tmp_path)


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_internal_dsh_judgment_is_character_owned(
    tmp_path: Path,
) -> None:
    """A signed DSH question receives one cognition-owned semantic decision."""

    await run_live_behavior_case("internal", tmp_path)
