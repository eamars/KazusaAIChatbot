"""One isolated live check for the complete canonical state transaction."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import pytest

from kazusa_ai_chatbot.cognition_core_v3.facade import (
    bind_protected_chain_records,
    reset_protected_chain_records,
    run_cognition,
    snapshot_protected_chain_records,
)
from kazusa_ai_chatbot.cognition_shared.state_models import validate_cognition_state
from kazusa_ai_chatbot.cognition_shared.state_reducers import materialize_causal_root
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from tests.unit.cognition_core_v3.test_handleless_contract import _input

pytestmark = pytest.mark.live_llm


def _production_trigger_input() -> dict[str, object]:
    """Build the bounded state shape observed at the production failure edge."""

    payload = deepcopy(_input())
    timestamp = str(payload["mutable_state"]["updated_at"])
    episode = payload["episode"]
    episode["percepts"][0]["content"]["semantic_text"] = "早上好~"
    payload["evidence"] = [{
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode:capacity-live-greeting",
            "occurred_at": timestamp,
            "semantic_summary": "早上好~",
        },
        "semantic_text": "早上好~",
        "authority": "current_event",
    }]
    state = payload["mutable_state"]
    for index in range(32):
        evidence = {
            "source_kind": "episode",
            "source_id": f"episode:capacity-live-{index}",
            "occurred_at": timestamp,
            "semantic_summary": f"retained production event {index}",
        }
        state, _root_id, _created = materialize_causal_root(
            state,
            kind="event",
            primary_evidence=evidence,
            description=f"retained production event {index}",
        )
        state["active_events"][-1]["salience"] = 25
    for row in state["active_events"][:26]:
        row["status"] = "resolved"
    validate_cognition_state(state)
    payload["mutable_state"] = state
    return payload


def _state_summary(state: object) -> dict[str, object]:
    if not isinstance(state, dict):
        return {}
    events = state.get("active_events", [])
    affect = state.get("affect_activations", [])
    return {
        "state_scope": state.get("state_scope"),
        "updated_at": state.get("updated_at"),
        "collection_counts": {
            name: len(state.get(name, []))
            for name in ("active_events", "threats", "knowledge_gaps", "goals")
        },
        "event_status_counts": {
            status: sum(
                1 for row in events
                if isinstance(row, dict) and row.get("status") == status
            )
            for status in ("resolved", "active")
        },
        "affect": [
            {
                "emotion_id": row.get("emotion_id"),
                "score": row.get("score"),
                "phase": row.get("phase"),
                "trend": row.get("trend"),
                "cause_status": row.get("cause_status"),
                "cause_summary": row.get("cause_summary", ""),
            }
            for row in affect
            if isinstance(row, dict)
        ],
    }


def _write_live_artifact(value: dict[str, object]) -> Path:
    directory = Path("test_artifacts/diagnostics/cognition_v3_capacity_live_llm")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"capacity_transaction_{time.time_ns()}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


async def test_live_captured_full_state_greeting_completes_first_pass() -> None:
    """Exercise the production-shaped bounded state through the live chain."""

    payload = _production_trigger_input()
    token = bind_protected_chain_records(
        run_id="gate7-capacity-live",
        source_kind="capacity_live_fixture",
    )
    started = time.monotonic()
    artifact: dict[str, object] = {
        "schema": "cognition_v3_capacity_live_evidence.v1",
        "input": {
            "observation": "早上好~",
            "active_events": 32,
            "resolved_events": 26,
            "active_event_rows": 6,
            "state_scope": payload["state_scope"],
        },
    }
    output = None
    try:
        output = await run_cognition(
            payload,
            build_cognition_core_services(),
        )
        artifact["output"] = {
            "schema_version": output.get("schema_version"),
            "goal": output.get("active_character_goal"),
            "response_plan": output.get("response_plan"),
            "affect_projection": output.get("affect_projection"),
            "cause_provenance": output.get("cause_provenance"),
            "state": _state_summary(
                output.get("state_projection", {}).get("replacement_state")
                if isinstance(output.get("state_projection"), dict)
                else None
            ),
            "binding_disposition_counts": {
                disposition: sum(
                    1
                    for row in output.get("state_projection", {}).get(
                        "binding_receipts", []
                    )
                    if isinstance(row, dict)
                    and row.get("disposition") == disposition
                )
                for disposition in (
                    "applied",
                    "clamped",
                    "no_numeric_change",
                    "scope_inapplicable",
                    "turn_local",
                    "capacity_deferred",
                )
            },
        }
    except Exception as exc:
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        artifact["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
        artifact["protected_records"] = list(snapshot_protected_chain_records())
        reset_protected_chain_records(token)
        path = _write_live_artifact(artifact)
        print(f"live cognition artifact: {path}")

    assert output is not None
    assert output["schema_version"] == "cognition_output.v3"
    assert output["active_character_goal"]["intent"]
    assert output["response_plan"]["goal_resolution"]
    replacement_state = output["state_projection"]["replacement_state"]
    assert len(replacement_state["active_events"]) <= 32
    assert len(output["state_projection"]["capacity_deferred"]) == 0
    assert len(output["state_projection"]["binding_receipts"]) >= 1
