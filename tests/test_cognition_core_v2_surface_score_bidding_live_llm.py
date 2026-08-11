"""Reserved one-at-a-time live gates for accepted surface calibration."""

from __future__ import annotations

import pytest


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


async def test_live_surface_content_score_orders_candidates() -> None:
    """Calibration evidence must exist before this live gate can run."""

    pytest.skip(
        "surface_content_plan calibration is blocked: no trace-backed corpus"
    )


async def test_live_dialog_compliance_repair_score_orders_candidates() -> None:
    """Calibration evidence must exist before this live gate can run."""

    pytest.skip(
        "surface_dialog_compliance_repair calibration is blocked: "
        "no trace-backed corpus"
    )
