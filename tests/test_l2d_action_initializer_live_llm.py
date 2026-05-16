"""Real LLM routing checks for the L2d action initializer."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import httpx
import pytest

from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2d import (
    build_action_initializer_payload,
    call_action_initializer,
)
from tests.l2d_action_initializer_cases import (
    compare_action_specs_to_expectations,
    load_l2d_routing_case_set,
    select_l2d_routing_case,
)
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)

_CASE_FILE_ENV = "L2D_LIVE_CASE_FILE"
_CASE_ID_ENV = "L2D_LIVE_CASE_ID"
_FORBIDDEN_ACTION_SPEC_FRAGMENTS = (
    "handler_id",
    "credentials",
    "api_key",
    "mongodb",
    "mongo",
    "collection",
    "self_cognition_action_attempts",
)


async def test_l2d_live_case_against_frozen_upstream() -> None:
    """Run one frozen upstream case through L2d and compare route shape."""

    await _skip_if_llm_unavailable()
    case_file = _configured_case_file()
    case_id = _configured_case_id()
    case_set = load_l2d_routing_case_set(case_file)
    case = select_l2d_routing_case(case_set, case_id)
    frozen_state = case["frozen_l2d_state"]
    prompt_payload = build_action_initializer_payload(frozen_state)

    result = await call_action_initializer(frozen_state)
    action_specs = result["action_specs"]
    report = compare_action_specs_to_expectations(case, action_specs)
    leakage_errors = _action_spec_leakage_errors(action_specs)
    trace_path = write_llm_trace(
        "l2d_action_initializer_live_llm",
        case_id,
        {
            "case_id": case_id,
            "case_file": str(case_file),
            "source_kind": case["source_kind"],
            "historical_comparison": case["historical_comparison"],
            "prompt_payload": prompt_payload,
            "parsed_output": result,
            "comparison_report": report,
            "leakage_errors": leakage_errors,
            "judgment": "manual_review_required_for_l2d_route_quality",
        },
    )
    logger.info(
        f"L2D_ACTION_INITIALIZER_LIVE case={case_id} "
        f"trace={trace_path} report={json.dumps(report, ensure_ascii=True)}"
    )

    assert len(action_specs) <= 3
    assert leakage_errors == []
    assert report["ok"] is True, report["errors"]


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured cognition endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}; {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{COGNITION_LLM_BASE_URL}"
        )


def _configured_case_file() -> Path:
    """Read the selected frozen case file path from the environment."""

    raw_path = os.environ.get(_CASE_FILE_ENV)
    if raw_path is None or not raw_path.strip():
        pytest.skip(f"{_CASE_FILE_ENV} is required for one-case live L2d runs")
    case_file = Path(raw_path)
    if not case_file.exists():
        pytest.skip(f"{_CASE_FILE_ENV} does not exist: {case_file}")
    return case_file


def _configured_case_id() -> str:
    """Read the selected frozen case id from the environment."""

    case_id = os.environ.get(_CASE_ID_ENV)
    if case_id is None or not case_id.strip():
        pytest.skip(f"{_CASE_ID_ENV} is required for one-case live L2d runs")
    return case_id


def _action_spec_leakage_errors(action_specs: list[dict]) -> list[str]:
    """Return prompt-safety errors found in action-spec output."""

    serialized = json.dumps(action_specs, ensure_ascii=False).lower()
    errors = []
    for fragment in _FORBIDDEN_ACTION_SPEC_FRAGMENTS:
        if fragment in serialized:
            errors.append(f"forbidden runtime fragment leaked: {fragment}")
    return errors
