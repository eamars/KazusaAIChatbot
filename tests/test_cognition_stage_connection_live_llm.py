"""Real LLM full cognition-stage connection checks from captured QQ cases."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface_module
from tests.cognition_stage_connection_cases import (
    build_cognition_connection_comparison_report,
    load_cognition_stage_connection_case_set,
    select_cognition_stage_connection_case,
    speak_action_selected,
)
from tests.llm_trace import write_llm_trace

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)

_CASE_FILE_ENV = "COGNITION_CONNECTION_CASE_FILE"
_CASE_ID_ENV = "COGNITION_CONNECTION_CASE_ID"


async def test_live_cognition_stage_connection_case(monkeypatch) -> None:
    """Run one captured QQ case through cognition, selected L3, and dialog."""

    await _skip_if_llm_unavailable()
    _require_reconnected_production_symbols()
    case_file = _configured_case_file()
    case_id = _configured_case_id()
    case_set = load_cognition_stage_connection_case_set(case_file)
    case = select_cognition_stage_connection_case(case_set, case_id)
    stage_outputs: dict[str, Any] = {}
    _wrap_cognition_stages(monkeypatch, stage_outputs)
    _wrap_l3_stages(monkeypatch, stage_outputs)

    state = dict(case["seed_state"])
    cognition_update = await cognition_module.call_cognition_subgraph(state)
    state.update(cognition_update)
    action_specs = state.get("action_specs", [])
    if not isinstance(action_specs, list):
        action_specs = []

    selected_speak = speak_action_selected(action_specs)
    final_dialog: list[str] = []
    if selected_speak:
        l3_update = await l3_surface_module.call_l3_text_surface_handler(state)
        state.update(l3_update)
        dialog_update = await dialog_module.dialog_agent(state)
        state.update(dialog_update)
        final_dialog = list(dialog_update["final_dialog"])

    report = build_cognition_connection_comparison_report(
        case,
        action_specs=action_specs,
        final_dialog=final_dialog,
        l3_ran=_any_stage_with_prefix(stage_outputs, "l3_"),
        l4_ran="l4_surface_directive_collector" in stage_outputs,
    )
    trace_path = write_llm_trace(
        "cognition_stage_connection_live_llm",
        case_id,
        {
            "case_file": str(case_file),
            "case_id": case_id,
            "source_kind": case["source_kind"],
            "historical_user_message": case["historical_user_message"],
            "historical_assistant_reply": case["historical_assistant_reply"],
            "stage_outputs": stage_outputs,
            "action_specs": action_specs,
            "selected_speak": selected_speak,
            "final_dialog": final_dialog,
            "comparison_report": report,
            "judgment": "manual_review_required_for_full_cognition_connection",
        },
    )
    logger.info(
        f"COGNITION_STAGE_CONNECTION_LIVE case={case_id} "
        f"trace={trace_path} report={json.dumps(report, ensure_ascii=True)}"
    )

    assert report["ok"] is True, report["errors"]
    assert trace_path.exists()


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
    """Read the selected local case-set path from the environment."""

    raw_path = os.environ.get(_CASE_FILE_ENV)
    if raw_path is None or not raw_path.strip():
        pytest.skip(f"{_CASE_FILE_ENV} is required for one-case live runs")
    case_file = Path(raw_path)
    if not case_file.exists():
        pytest.skip(f"{_CASE_FILE_ENV} does not exist: {case_file}")
    return case_file


def _configured_case_id() -> str:
    """Read the selected case id from the environment."""

    case_id = os.environ.get(_CASE_ID_ENV)
    if case_id is None or not case_id.strip():
        pytest.skip(f"{_CASE_ID_ENV} is required for one-case live runs")
    return case_id


def _require_reconnected_production_symbols() -> None:
    """Fail clearly if the production graph is still on the legacy symbols."""

    required = (
        (cognition_module, "call_cognition_subgraph"),
        (l3_surface_module, "call_l3_text_surface_handler"),
        (l2c2_module, "call_social_context_appraisal"),
        (l3_module, "call_surface_directive_collector"),
    )
    missing = [
        name
        for module, name in required
        if not hasattr(module, name)
    ]
    if missing:
        raise AssertionError(f"missing reconnected production symbols: {missing}")


def _wrap_cognition_stages(monkeypatch, stage_outputs: dict[str, Any]) -> None:
    """Capture real cognition node outputs while the graph executes."""

    _wrap_async_stage(
        monkeypatch,
        l1_module,
        "call_cognition_subconscious",
        "l1_subconscious",
        stage_outputs,
    )
    _wrap_async_stage(
        monkeypatch,
        l2_module,
        "call_cognition_consciousness",
        "l2a_conscious_framing",
        stage_outputs,
    )
    _wrap_async_stage(
        monkeypatch,
        l2_module,
        "call_boundary_core_agent",
        "l2b_boundary_appraisal",
        stage_outputs,
    )
    _wrap_async_stage(
        monkeypatch,
        l2_module,
        "call_judgment_core_agent",
        "l2c1_judgment_synthesis",
        stage_outputs,
    )
    _wrap_async_stage(
        monkeypatch,
        l2c2_module,
        "call_social_context_appraisal",
        "l2c2_social_context_appraisal",
        stage_outputs,
    )
    _wrap_async_stage(
        monkeypatch,
        l2d_module,
        "select_semantic_actions",
        "l2d_action_selection",
        stage_outputs,
    )


def _wrap_l3_stages(monkeypatch, stage_outputs: dict[str, Any]) -> None:
    """Capture selected L3 and L4 outputs when speak is selected."""

    _wrap_async_stage(
        monkeypatch,
        l3_module,
        "call_interaction_style_context_loader",
        "l3_interaction_style_context_loader",
        stage_outputs,
    )
    _wrap_async_stage(
        monkeypatch,
        l3_module,
        "call_style_agent",
        "l3_style_agent",
        stage_outputs,
    )
    _wrap_async_stage(
        monkeypatch,
        l3_module,
        "call_content_plan_agent",
        "l3_content_plan_agent",
        stage_outputs,
    )
    _wrap_async_stage(
        monkeypatch,
        l3_module,
        "call_preference_adapter",
        "l3_preference_adapter",
        stage_outputs,
    )
    _wrap_async_stage(
        monkeypatch,
        l3_module,
        "call_visual_agent",
        "l3_visual_agent",
        stage_outputs,
    )
    _wrap_async_stage(
        monkeypatch,
        l3_module,
        "call_surface_directive_collector",
        "l4_surface_directive_collector",
        stage_outputs,
    )


def _wrap_async_stage(
    monkeypatch,
    module: object,
    function_name: str,
    stage_name: str,
    stage_outputs: dict[str, Any],
) -> None:
    """Patch one async stage so its real output is saved in the trace."""

    original = getattr(module, function_name)

    async def wrapped(state: dict[str, Any]) -> dict[str, Any]:
        if stage_name == "l2d_action_selection":
            stage_outputs["l2d_action_selection_prompt_payload"] = (
                l2d_module.build_action_selection_payload_text(state)
            )
        result = await original(state)
        stage_outputs[stage_name] = result
        return result

    monkeypatch.setattr(module, function_name, wrapped)


def _any_stage_with_prefix(stage_outputs: dict[str, Any], prefix: str) -> bool:
    """Return whether any captured stage key starts with a prefix."""

    for stage_name in stage_outputs:
        if stage_name.startswith(prefix):
            return True
    return False
