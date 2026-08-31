"""Live DSH sign-off coverage for identity-bound scheduled sources."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.dsh_trigger_source_e2e_support import run_trigger_source_case


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_commitment_due_tick_reaches_dsh(tmp_path: Path) -> None:
    """A due active commitment should retain its user and enter DSH."""

    await run_trigger_source_case("scheduled_tick_commitment_due", tmp_path)


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_scheduled_future_tick_reaches_dsh(tmp_path: Path) -> None:
    """A due future-cognition run should enter DSH through the shared worker."""

    await run_trigger_source_case("scheduled_tick_future", tmp_path)
