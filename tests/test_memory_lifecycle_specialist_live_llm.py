"""Real LLM checks for the memory lifecycle specialist."""

from __future__ import annotations

import logging
import sys
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
)
from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.nodes.persona_supervisor2_memory_lifecycle import (
    call_memory_lifecycle_update_handler,
    call_post_surface_memory_lifecycle_review,
)
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)


async def test_live_tiramisu_fulfilled() -> None:
    """Specialist should close the fulfilled tiramisu commitment."""

    await _skip_if_llm_unavailable()
    state = _state(
        current_input="用户明确说提拉米苏已经给角色了，这个承诺已经兑现。",
        commitments=[
            _commitment(
                "unit-tiramisu",
                "用户答应给角色提拉米苏。",
                due_state="past_due",
            ),
            _commitment(
                "unit-magic",
                "用户答应之后解释魔法道具。",
                due_state="past_due",
            ),
        ],
    )

    result = await call_memory_lifecycle_update_handler(state)
    trace_path = _write_trace("tiramisu_fulfilled", state, result)

    assert _apply_unit_ids(result) == ["unit-tiramisu"], f"trace={trace_path}"
    assert _context_decisions(result)[0]["decision"] == "fulfilled"


async def test_live_new_sweet_plan_no_lifecycle_change() -> None:
    """A new dessert plan should not close an existing tiramisu promise."""

    await _skip_if_llm_unavailable()
    state = _state(
        current_input="用户只是说下次可能再买别的甜品，没有说提拉米苏已经给到。",
        commitments=[
            _commitment(
                "unit-tiramisu",
                "用户答应给角色提拉米苏。",
                due_state="past_due",
            ),
        ],
    )

    result = await call_memory_lifecycle_update_handler(state)
    trace_path = _write_trace("new_sweet_plan_no_change", state, result)

    assert _apply_unit_ids(result) == [], f"trace={trace_path}"
    assert result["memory_lifecycle_context"]["decision"] in (
        "no_lifecycle_change",
        "skipped",
    )


async def test_live_magic_fulfilled() -> None:
    """Specialist should close the fulfilled magic commitment, not tiramisu."""

    await _skip_if_llm_unavailable()
    state = _state(
        current_input="用户解释了之前答应的魔法道具细节，这个魔法承诺已经兑现。",
        commitments=[
            _commitment(
                "unit-tiramisu",
                "用户答应给角色提拉米苏。",
                due_state="past_due",
            ),
            _commitment(
                "unit-magic",
                "用户答应之后解释魔法道具。",
                due_state="past_due",
            ),
        ],
    )

    result = await call_memory_lifecycle_update_handler(state)
    trace_path = _write_trace("magic_fulfilled", state, result)

    assert _apply_unit_ids(result) == ["unit-magic"], f"trace={trace_path}"
    assert _context_decisions(result)[0]["decision"] == "fulfilled"


async def test_live_ambiguous_reply_no_lifecycle_change() -> None:
    """Ambiguous continuation should not close any active commitment."""

    await _skip_if_llm_unavailable()
    state = _state(
        current_input="用户只是说等一下，还没说承诺已经完成。",
        commitments=[
            _commitment(
                "unit-tiramisu",
                "用户答应给角色提拉米苏。",
                due_state="past_due",
            ),
        ],
    )

    result = await call_memory_lifecycle_update_handler(state)
    trace_path = _write_trace("ambiguous_no_change", state, result)

    assert _apply_unit_ids(result) == [], f"trace={trace_path}"


async def test_live_capacity_12_commitments_last_alias_fulfilled() -> None:
    """The specialist should handle 12 aliases and select the last one."""

    await _skip_if_llm_unavailable()
    commitments = [
        _commitment(
            f"unit-capacity-{index:02d}",
            f"用户答应交付第 {index} 个测试甜品。",
            due_at=f"2026-05-{index:02d}T18:00:00+12:00",
            due_state="past_due",
        )
        for index in range(1, 13)
    ]
    state = _state(
        current_input="用户明确说第 12 个测试甜品已经交付，这个承诺已经兑现。",
        commitments=commitments,
    )

    result = await call_memory_lifecycle_update_handler(state)
    trace_path = _write_trace("capacity_12_last_fulfilled", state, result)

    assert _apply_unit_ids(result) == ["unit-capacity-12"], f"trace={trace_path}"
    assert result["memory_lifecycle_context"]["visible_alias_count"] == 12


async def test_live_tiramisu_no_due_final_dialog_fulfilled() -> None:
    """Post-surface specialist should close no-due tiramisu from final dialog."""

    await _skip_if_llm_unavailable()
    commitment = _commitment(
        "unit-tiramisu",
        '用户答应给角色提拉米苏。',
        due_at=None,
        due_state="no_due_date",
    )
    state = _state(
        current_input='用户把提拉米苏交给角色，角色准备收下。',
        commitments=[commitment],
    )
    state["final_dialog"] = ['提拉米苏债都清了哦，我就收下啦。']
    state["surface_outputs"] = [
        {
            "schema_version": "surface_output.v1",
            "surface_kind": "text",
            "visibility": "user_visible",
            "action_attempt_id": None,
            "fragments": state["final_dialog"],
            "artifact_refs": [],
            "delivery_intent": "deliver_now",
            "created_at": "2026-05-29T03:14:23+00:00",
        }
    ]

    result = await call_post_surface_memory_lifecycle_review(
        state,
        [commitment],
    )
    trace_path = _write_trace(
        "tiramisu_no_due_final_dialog_fulfilled",
        state,
        result,
    )

    assert _apply_unit_ids(result) == ["unit-tiramisu"], f"trace={trace_path}"
    assert _context_decisions(result)[0]["decision"] == "fulfilled"


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


def _state(
    *,
    current_input: str,
    commitments: list[dict[str, object]],
) -> dict[str, Any]:
    """Build a prompt-only specialist state with trusted active commitments."""

    return {
        "storage_timestamp_utc": "2026-05-17T06:00:49+00:00",
        "user_input": current_input,
        "decontexualized_input": current_input,
        "logical_stance": "CONFIRM",
        "character_intent": "UPDATE_MEMORY_LIFECYCLE",
        "judgment_note": "The current turn may change an active commitment.",
        "internal_monologue": "Ask the lifecycle specialist to review aliases.",
        "rag_result": {
            "answer": "There are active commitments to review.",
            "memory_evidence": [
                {
                    "summary": current_input,
                    "fact": current_input,
                }
            ],
            "user_image": {
                "user_memory_context": {
                    "active_commitments": commitments,
                }
            },
        },
        "conversation_progress": {
            "current_thread": "active commitment lifecycle review",
        },
        "cognitive_episode": {
            "episode_id": "live-memory-lifecycle-specialist",
        },
        "action_specs": [_memory_lifecycle_route_spec()],
    }


def _commitment(
    unit_id: str,
    fact: str,
    *,
    due_at: str | None = "2026-05-07T18:00:00+12:00",
    due_state: str,
) -> dict[str, object]:
    """Build one trusted active-commitment row."""

    return {
        "unit_id": unit_id,
        "fact": fact,
        "summary": fact,
        "status": "active",
        "due_at": due_at,
        "due_state": due_state,
    }


def _memory_lifecycle_route_spec() -> dict[str, object]:
    """Build the L2d route intent expected by the specialist handler."""

    return {
        "schema_version": "action_spec.v1",
        "kind": MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "cognitive_episode",
            "target_id": None,
            "owner": "memory_lifecycle_specialist",
            "scope": {"unit_type": "active_commitment"},
        },
        "params": {
            "review_kind": "active_commitment_lifecycle",
            "detail": "Review active commitment lifecycle.",
        },
        "urgency": "background",
        "visibility": "private",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": "Current turn may affect active commitments.",
    }


def _apply_unit_ids(result: dict[str, object]) -> list[str]:
    """Return unit ids from materialized apply actions."""

    action_specs = result.get("action_specs")
    if not isinstance(action_specs, list):
        return []

    unit_ids = []
    for action_spec in action_specs:
        if not isinstance(action_spec, dict):
            continue
        if action_spec.get("kind") != APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
            continue
        params = action_spec.get("params")
        if not isinstance(params, dict):
            continue
        unit_id = params.get("unit_id")
        if isinstance(unit_id, str):
            unit_ids.append(unit_id)
    return unit_ids


def _context_decisions(result: dict[str, object]) -> list[dict[str, object]]:
    """Return prompt-safe lifecycle context decisions."""

    context = result.get("memory_lifecycle_context")
    if not isinstance(context, dict):
        return []
    decisions = context.get("lifecycle_decisions")
    if not isinstance(decisions, list):
        return []
    return [
        decision
        for decision in decisions
        if isinstance(decision, dict)
    ]


def _write_trace(
    case_id: str,
    state: dict[str, object],
    result: dict[str, object],
) -> object:
    """Write an inspectable trace for manual live-test review."""

    trace_path = write_llm_trace(
        "memory_lifecycle_specialist_live_llm",
        case_id,
        {
            "current_input": state["decontexualized_input"],
            "active_commitments": (
                state["rag_result"]["user_image"]["user_memory_context"][
                    "active_commitments"
                ]
            ),
            "result": result,
            "apply_unit_ids": _apply_unit_ids(result),
            "judgment": "manual_review_required_for_specialist_lifecycle_quality",
        },
    )
    logger.info(
        f"MEMORY_LIFECYCLE_SPECIALIST_LIVE case={case_id} trace={trace_path}"
    )
    return trace_path
