"""DB-free real-LLM smokes for self-cognition trigger state contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_character_production_state,
)
from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as connector
from kazusa_ai_chatbot.self_cognition import models, projection, runner
from tests.cognition_test_helpers import (
    canonical_service_character_profile,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ARTIFACT_ROOT = Path("test_artifacts/diagnostics")
_GROUP_ARTIFACT = _ARTIFACT_ROOT / (
    "self_cognition_trigger_state_live_group.json"
)
_COMMITMENT_ARTIFACT = _ARTIFACT_ROOT / (
    "self_cognition_trigger_state_live_commitment.json"
)


async def test_live_group_review_state_contract_reaches_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one synthetic targetless group state through the real V2 model."""

    await _skip_if_llm_unavailable()
    case = _group_case()
    evidence = await _run_case(
        case,
        artifact_path=_GROUP_ARTIFACT,
        monkeypatch=monkeypatch,
    )

    assert evidence["validation"] == "cognition_core_input.v2 accepted"
    assert evidence["state_contract"]["conversation_progress"] is None
    assert evidence["state_contract"]["public_group_scene"]
    assert evidence["output"]["schema_version"] == "cognition_core_output.v2"


async def test_live_commitment_state_contract_reaches_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one synthetic due-commitment state through the real V2 model."""

    await _skip_if_llm_unavailable()
    case = _commitment_case()
    evidence = await _run_case(
        case,
        artifact_path=_COMMITMENT_ARTIFACT,
        monkeypatch=monkeypatch,
    )

    assert evidence["validation"] == "cognition_core_input.v2 accepted"
    assert evidence["state_contract"]["conversation_progress"] is None
    assert evidence["state_contract"]["public_group_scene"] == ""
    assert evidence["output"]["schema_version"] == "cognition_core_output.v2"


async def _run_case(
    case: dict[str, Any],
    *,
    artifact_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Build the repaired state, call V2, and save raw inspection evidence."""

    source_packet = projection.build_source_packet(case)
    rendered_packet = projection.render_source_packet_text(source_packet)
    group_target = case["target_scope"]["channel_type"] == "group"
    style_snapshot = _style_snapshot(group_target=group_target)
    public_group_scene = runner._build_public_group_scene(case)
    state = runner._build_cognition_state(
        case,
        rendered_packet,
        public_group_scene=public_group_scene,
        interaction_style_context=style_snapshot,
    )
    state["rag_result"] = {"answer": ""}
    now = str(case["idle_timestamp_utc"])

    async def load_character_state() -> dict[str, Any]:
        return build_character_production_state(updated_at=now)

    monkeypatch.setattr(
        connector,
        "get_character_cognition_state",
        load_character_state,
    )
    monkeypatch.setattr(
        connector,
        "_state_has_episode_identity_snapshot",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        connector,
        "_episode_identity_state_update",
        lambda current_state: {},
    )
    result = await connector.call_cognition_subgraph(state, commit=False)
    cognition_input = result["cognition_input"]
    output = result["cognition_core_output"]
    evidence = {
        "run_context": {
            "source": "synthetic_db_free_fixture",
            "model_path": (
                "persona_supervisor2_cognition.call_cognition_subgraph"
            ),
            "database_writes": False,
            "delivery_attempt": False,
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "state_contract": {
            "case_name": case["case_name"],
            "trigger_kind": case["trigger_kind"],
            "conversation_progress": state.get("conversation_progress"),
            "source_context": state.get("source_context"),
            "public_group_scene": state["public_group_scene"],
            "interaction_style_context": state[
                "interaction_style_context"
            ],
        },
        "source_packet": source_packet,
        "rendered_packet": rendered_packet,
        "cognition_input": cognition_input,
        "output": output,
        "validation": "cognition_core_input.v2 accepted",
    }
    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return evidence


def _group_case() -> dict[str, Any]:
    """Build a targetless group-review source case without persistence."""

    now = "2026-08-10T10:00:00Z"
    return {
        "case_name": models.CASE_GROUP_CHAT_REVIEW,
        "case_id": "live-smoke:group-review",
        "idle_timestamp_utc": now,
        "last_evidence_timestamp_utc": now,
        "trigger_kind": models.TRIGGER_GROUP_CHAT_REVIEW,
        "semantic_due_state": None,
        "actionability": "active_group_review_same_channel_no_fallback",
        "target_scope": {
            "platform": "debug",
            "platform_channel_id": "synthetic-group",
            "channel_type": "group",
            "user_id": None,
        },
        "source_refs": [{
            "source_kind": "reflection_activity_window",
            "source_id": "synthetic-window",
            "due_at": None,
            "summary": "A small group scene is available for review.",
        }],
        "visible_context": [
            {
                "role": "user",
                "display_name": "Speaker",
                "timestamp": now,
                "body_text": "The group is discussing a useful next step.",
            },
        ],
        "conversation_progress": None,
        "source_context": {
            "schema_version": "self_cognition_group_source_context.v1",
            "context_kind": "group_chat_review",
            "group_activity_window": {
                "source": "reflection_activity_window",
                "window_start": now,
                "window_end": now,
                "semantic_labels": {
                    "activity_level": "active",
                    "assistant_presence": "absent",
                    "bot_addressing": "ambient_group_context",
                },
            },
            "conversation_evidence": [],
        },
        "character_profile": canonical_service_character_profile(
            marker="self-cognition-live-group",
        ),
        "platform_bot_id": "synthetic-bot",
    }


def _commitment_case() -> dict[str, Any]:
    """Build a due commitment source case without persistence."""

    now = "2026-08-10T10:00:00Z"
    return {
        "case_name": models.CASE_COMMITMENT_PAST_DUE,
        "case_id": "live-smoke:commitment-due",
        "idle_timestamp_utc": now,
        "last_evidence_timestamp_utc": now,
        "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
        "semantic_due_state": models.DUE_STATE_PAST_DUE,
        "actionability": "past_due_commitment_contact_socially_available",
        "target_scope": {
            "platform": "debug",
            "platform_channel_id": "synthetic-group",
            "channel_type": "group",
            "user_id": None,
        },
        "source_refs": [{
            "source_kind": "user_memory_unit",
            "source_id": "synthetic-commitment",
            "due_at": "2026-08-10T09:00:00+00:00",
            "summary": "A scheduled follow-up check is due.",
        }],
        "visible_context": [],
        "conversation_progress": None,
        "source_context": None,
        "character_profile": canonical_service_character_profile(
            marker="self-cognition-live-commitment",
        ),
        "platform_bot_id": "synthetic-bot",
    }


def _style_snapshot(*, group_target: bool) -> dict[str, Any]:
    """Build one immutable prompt-safe snapshot for a synthetic target."""

    overlay = {
        "speech_guidelines": [],
        "social_guidelines": [],
        "pacing_guidelines": [],
        "engagement_guidelines": [],
        "confidence": "medium",
    }
    surface: dict[str, dict[str, dict[str, Any]]] = {
        "user": {"overlay": dict(overlay)},
    }
    application_order = ["user"]
    if group_target:
        surface["group_channel"] = {"overlay": dict(overlay)}
        application_order.append("group_channel")
    return {
        "schema_version": "interaction_style_turn_snapshot.v1",
        "sources": {},
        "relevance": {},
        "cognition": {},
        "surface": surface,
        "application_order": application_order,
        "user_style": dict(overlay),
        "group_engagement_action_context": {
            "engagement_guidelines": (
                ["Stay grounded in the observed scene."]
                if group_target
                else []
            ),
            "confidence": "medium" if group_target else "",
        },
        "snapshot_digest": (
            "live-group-style" if group_target else "live-private-style"
        ),
    }


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured real cognition endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f"LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}; {exc}"
        )
    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{COGNITION_LLM_BASE_URL}"
        )
