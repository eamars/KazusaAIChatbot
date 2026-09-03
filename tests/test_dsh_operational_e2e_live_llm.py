"""Live operational-failure sign-off for the DSH integration boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.dsh_trigger_source_e2e_support import (
    run_configured_weather_case,
    run_sidecar_loss_case,
)


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_sidecar_loss_fails_closed_without_graph_crash(
    tmp_path: Path,
) -> None:
    """An unavailable DSH sidecar must yield a coherent completed turn."""

    await run_sidecar_loss_case(tmp_path)


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_configured_services_resolve_christchurch_weather(
    tmp_path: Path,
) -> None:
    """Configured supervised services must resolve the production command."""

    await run_configured_weather_case(tmp_path)
