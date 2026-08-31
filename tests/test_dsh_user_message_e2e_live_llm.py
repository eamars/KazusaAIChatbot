"""Live DSH sign-off coverage for the public user-message source."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.dsh_trigger_source_e2e_support import run_trigger_source_case


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_user_message_local_fact_reaches_dsh(
    tmp_path: Path,
) -> None:
    """A natural local fact request should enter and settle through DSH."""

    await run_trigger_source_case("user_message_local_fact", tmp_path)


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_user_message_background_summary_reaches_dsh(
    tmp_path: Path,
) -> None:
    """An explicitly delayed user request should settle and return once."""

    await run_trigger_source_case("user_message_background_summary", tmp_path)
