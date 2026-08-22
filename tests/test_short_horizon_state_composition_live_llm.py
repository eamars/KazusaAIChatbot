"""Controlled live-model comparisons for short-horizon state composition."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import sys
from typing import Any, Awaitable, Callable
from uuid import uuid4

import pytest
from fastapi import BackgroundTasks

from kazusa_ai_chatbot import service as brain_service
from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.db import (
    get_character_cognition_state,
    get_user_cognition_state,
    replace_user_cognition_state,
    resolve_global_user_id,
    upsert_user_style_image,
)
from kazusa_ai_chatbot.db._client import get_db
from tests.llm_trace import write_llm_trace
from tests.test_e2e_live_llm import (
    _refresh_character_profile,
    live_env,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_BOT_ID = "short-horizon-controlled-bot"
_TERMINAL_RECEIPT_STATUSES = {"committed", "failed", "no_change", "timed_out"}
_RECEIPT_WAIT_SECONDS = 55.0


def _utc_now() -> str:
    """Return the current storage timestamp in canonical UTC text."""

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return timestamp


def _canonical_digest(value: object) -> str:
    """Hash one JSON-compatible value for pair-control evidence."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return digest


def _event(
    event_id: str,
    *,
    updated_at: str,
    description: str,
    relationship_id: str = "",
    memory_warmth: int = 0,
) -> dict[str, object]:
    """Build one valid source-bound event for a declared controlled seed."""

    role_refs: list[dict[str, str]] = []
    if relationship_id:
        role_refs.append({
            "role": "affected_relationship",
            "entity_kind": "relationship",
            "entity_id": relationship_id,
        })
    event = {
        "entity_id": event_id,
        "description": description,
        "salience": 80,
        "role_refs": role_refs,
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": f"episode:{event_id}",
            "occurred_at": updated_at,
            "semantic_summary": description,
        }],
        "created_at": updated_at,
        "updated_at": updated_at,
        "status": "active",
        "outcome_impact": -70,
        "responsibility": 75,
        "intentionality": 75,
        "harm": 70,
        "unfairness": 75,
        "exposure": 20,
        "repair_need": 65,
        "reparability": 60,
        "expectation_mismatch": 65,
        "norm_violation": 75,
        "contamination_risk": 70,
        "identity_threat": 30,
        "comparison_gap": 0,
        "vastness": 0,
        "memory_warmth": memory_warmth,
        "temporal_loss": 70,
    }
    return event


def _activation(
    emotion_id: str,
    *,
    updated_at: str,
    root_scope: str,
    root_kind: str,
    root_id: str,
    score: int = 76,
) -> dict[str, object]:
    """Build one complete native activation for controlled persistence."""

    root = {
        "scope": root_scope,
        "kind": root_kind,
        "entity_id": root_id,
    }
    activation = {
        "activation_id": f"emotion:{emotion_id}",
        "emotion_id": emotion_id,
        "primary_root": root,
        "root_refs": [root],
        "phase": "active",
        "score": score,
        "peak_score": score,
        "trend": "stable",
        "cause_status": "active",
        "started_at": updated_at,
        "updated_at": updated_at,
        "last_reinforced_at": updated_at,
    }
    return activation


def _character_state(
    emotion_id: str | None,
    *,
    updated_at: str,
) -> dict[str, object]:
    """Build the character seed whose only variable is native affect."""

    state = build_character_production_state(updated_at=updated_at)
    if emotion_id is None:
        return state
    event_id = f"event:controlled:{emotion_id}"
    state["active_events"] = [
        _event(
            event_id,
            updated_at=updated_at,
            description="A recent event still colors the character's posture.",
            memory_warmth=80 if emotion_id in {"joy", "gratitude"} else 0,
        )
    ]
    state["affect_activations"] = [
        _activation(
            emotion_id,
            updated_at=updated_at,
            root_scope="character",
            root_kind="event",
            root_id=event_id,
        )
    ]
    return state


def _graph_node(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    """Return one required node from a service cognition graph."""

    for node in graph["nodes"]:
        if node["id"] == node_id:
            return node
    raise AssertionError(f"cognition graph is missing node {node_id}")


def _decision_signature(graph: dict[str, Any]) -> dict[str, object]:
    """Project the decision-bearing graph rows used by the behavior review."""

    signature = {
        node_id: _graph_node(graph, node_id)["detail"]
        for node_id in (
            "l1.relevance",
            "l2.reasoning",
            "v2.appraisal",
            "v2.collapse",
        )
    }
    return signature


def _context_consumption(graph: dict[str, Any]) -> dict[str, Any]:
    """Return the exact context-consumption record emitted by the service."""

    reasoning = _graph_node(graph, "l2.reasoning")
    consumption = reasoning["detail"]["context_consumption"]
    if consumption["schema_version"] != "cognition_context_consumption.v1":
        raise AssertionError("context consumption schema is invalid")
    return consumption


async def _wait_for_receipt(delivery_tracking_id: str) -> dict[str, Any]:
    """Wait for the lifecycle-owned operational receipt to terminalize."""

    database = await get_db()
    deadline = asyncio.get_running_loop().time() + _RECEIPT_WAIT_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        document = await database.post_turn_lifecycle_records.find_one(
            {"delivery_tracking_id": delivery_tracking_id},
            {"_id": 0},
        )
        if document is not None:
            receipt = document.get("character_operational_receipt")
            if (
                isinstance(receipt, dict)
                and receipt.get("status") in _TERMINAL_RECEIPT_STATUSES
            ):
                return document
        await asyncio.sleep(0.25)
    raise AssertionError("operational receipt did not terminalize within 55 seconds")


async def _run_chat(
    *,
    case_id: str,
    branch: str,
    identity: dict[str, str],
    message: str,
) -> tuple[brain_service.ChatResponse, dict[str, Any]]:
    """Invoke the normal chat entrypoint and retain its durable receipt."""

    await _refresh_character_profile()
    request = brain_service.ChatRequest(
        platform=identity["platform"],
        platform_channel_id=identity["platform_channel_id"],
        channel_type="private",
        platform_message_id=f"controlled-{case_id}-{branch}-{uuid4().hex}",
        platform_user_id=identity["platform_user_id"],
        platform_bot_id=_BOT_ID,
        display_name=identity["display_name"],
        channel_name="controlled private conversation",
        message_envelope={
            "body_text": message,
            "raw_wire_text": message,
            "mentions": [{
                "platform_user_id": _BOT_ID,
                "global_user_id": brain_service.CHARACTER_GLOBAL_USER_ID,
                "display_name": "active character",
                "entity_kind": "bot",
                "raw_text": f"<@{_BOT_ID}>",
            }],
            "attachments": [],
            "addressed_to_global_user_ids": [
                brain_service.CHARACTER_GLOBAL_USER_ID
            ],
            "broadcast": False,
        },
    )
    response = await brain_service.chat(request, BackgroundTasks())
    if not response.delivery_tracking_id:
        raise AssertionError("controlled turn produced no visible delivery")
    lifecycle = await _wait_for_receipt(response.delivery_tracking_id)
    return response, lifecycle


SeedBranch = Callable[[str], Awaitable[dict[str, object]]]


async def _run_controlled_pair(
    *,
    case_id: str,
    changed_field: str,
    message: str,
    seed_control: SeedBranch,
    seed_treatment: SeedBranch,
) -> None:
    """Run one single-variable pair and write complete raw comparison data."""

    database = await get_db()
    suffix = uuid4().hex
    identity = {
        "platform": f"controlled-{suffix}",
        "platform_user_id": f"controlled-user-{suffix}",
        "platform_channel_id": f"controlled-channel-{suffix}",
        "display_name": "Controlled Comparison User",
    }
    global_user_id = await resolve_global_user_id(
        platform=identity["platform"],
        platform_user_id=identity["platform_user_id"],
        display_name=identity["display_name"],
    )
    identity["global_user_id"] = global_user_id
    character_snapshot = await database.character_state.find_one(
        {"_id": "global"},
    )
    if character_snapshot is None:
        raise AssertionError("global character state is missing")

    branches: dict[str, dict[str, object]] = {}
    try:
        for branch, seed_branch in (
            ("control", seed_control),
            ("treatment", seed_treatment),
        ):
            await database.conversation_history.delete_many({
                "platform": identity["platform"],
                "platform_channel_id": identity["platform_channel_id"],
            })
            seed = await seed_branch(global_user_id)
            response, lifecycle = await _run_chat(
                case_id=case_id,
                branch=branch,
                identity=identity,
                message=message,
            )
            graph = response.cognition_graph
            if not isinstance(graph, dict):
                raise AssertionError("controlled turn has no cognition graph")
            consumption = _context_consumption(graph)
            branches[branch] = {
                "seed": seed,
                "seed_digest": _canonical_digest(seed),
                "response": response.model_dump(),
                "receipt": lifecycle["character_operational_receipt"],
                "persisted_character_state": (
                    await get_character_cognition_state()
                ),
                "persisted_user_state": (
                    await get_user_cognition_state(global_user_id)
                ),
                "context_consumption": consumption,
                "decision_signature": _decision_signature(graph),
                "visible_dialog": list(response.messages),
            }
    finally:
        await database.character_state.replace_one(
            {"_id": "global"},
            character_snapshot,
            upsert=False,
        )

    control = branches["control"]
    treatment = branches["treatment"]
    artifact_path = write_llm_trace(
        "short_horizon_state_controlled_ab",
        case_id,
        {
            "input_kind": "synthetic controlled pre-run seed",
            "entrypoint": "kazusa_ai_chatbot.service.chat",
            "changed_field": changed_field,
            "fixed_fields": [
                "identity",
                "current_message",
                "conversation_history",
                "model_process",
                "code_revision",
            ],
            "identity": identity,
            "message": message,
            "control": control,
            "treatment": treatment,
            "pair_delta": {
                "seed_digest_changed": (
                    control["seed_digest"] != treatment["seed_digest"]
                ),
                "decision_changed": (
                    control["decision_signature"]
                    != treatment["decision_signature"]
                ),
                "dialog_changed": (
                    control["visible_dialog"] != treatment["visible_dialog"]
                ),
            },
            "quality_evaluation": "agent_review_required",
        },
    )

    assert artifact_path.exists()
    assert control["seed_digest"] != treatment["seed_digest"]
    assert control["visible_dialog"]
    assert treatment["visible_dialog"]
    assert control["decision_signature"] != treatment["decision_signature"]
    assert control["visible_dialog"] != treatment["visible_dialog"]


async def _seed_character_state(
    state: dict[str, object],
) -> dict[str, object]:
    """Install one declared character-state variable before an A/B branch."""

    database = await get_db()
    document = {
        "_id": "global",
        "cognition_state": deepcopy(state),
        "updated_at": state["updated_at"],
    }
    await database.character_state.replace_one(
        {"_id": "global"},
        document,
        upsert=True,
    )
    return deepcopy(state)


async def _run_offence_pair(
    *,
    emotion_id: str,
) -> None:
    """Generate offence affect natively, then compare an identical later probe."""

    database = await get_db()
    suffix = uuid4().hex
    platform = f"controlled-offence-{suffix}"
    platform_user_id = f"controlled-offence-user-{suffix}"
    display_name = "Controlled Offence User"
    source_channel_id = f"controlled-offence-source-{suffix}"
    probe_channel_id = f"controlled-offence-probe-{suffix}"
    global_user_id = await resolve_global_user_id(
        platform=platform,
        platform_user_id=platform_user_id,
        display_name=display_name,
    )
    character_snapshot = await database.character_state.find_one(
        {"_id": "global"},
    )
    if character_snapshot is None:
        raise AssertionError("global character state is missing")
    now = _utc_now()
    treatment_messages = {
        "anger": (
            "I deliberately blocked the goal you cared about, even though I "
            "knew it was unfair, because I wanted to obstruct you."
        ),
        "sadness": (
            "I deliberately destroyed something you valued, and the loss "
            "cannot be repaired or recovered."
        ),
        "disgust": (
            "I deliberately contaminated your food with filth and violated a "
            "basic hygiene boundary because I wanted to repulse you."
        ),
    }
    source_messages = {
        "control": (
            "I disagree with that choice, but I want to discuss it calmly and "
            "respect your boundary."
        ),
        "treatment": treatment_messages[emotion_id],
    }
    probe_message = "How would you approach a calm conversation with me now?"
    branches: dict[str, dict[str, object]] = {}
    try:
        for branch in ("control", "treatment"):
            await database.conversation_history.delete_many({
                "platform": platform,
                "platform_channel_id": {
                    "$in": [source_channel_id, probe_channel_id]
                },
            })
            neutral_state = _character_state(None, updated_at=now)
            await _seed_character_state(neutral_state)
            source_identity = {
                "platform": platform,
                "platform_user_id": platform_user_id,
                "platform_channel_id": source_channel_id,
                "global_user_id": global_user_id,
                "display_name": display_name,
            }
            source_response, source_lifecycle = await _run_chat(
                case_id=f"offence-{emotion_id}",
                branch=f"{branch}-source",
                identity=source_identity,
                message=source_messages[branch],
            )
            produced_state = await get_character_cognition_state()
            probe_identity = {
                **source_identity,
                "platform_channel_id": probe_channel_id,
            }
            probe_response, probe_lifecycle = await _run_chat(
                case_id=f"offence-{emotion_id}",
                branch=f"{branch}-probe",
                identity=probe_identity,
                message=probe_message,
            )
            probe_graph = probe_response.cognition_graph
            if not isinstance(probe_graph, dict):
                raise AssertionError("offence probe has no cognition graph")
            branches[branch] = {
                "source_message": source_messages[branch],
                "source_response": source_response.model_dump(),
                "source_receipt": source_lifecycle[
                    "character_operational_receipt"
                ],
                "native_produced_state": produced_state,
                "probe_message": probe_message,
                "probe_response": probe_response.model_dump(),
                "probe_receipt": probe_lifecycle[
                    "character_operational_receipt"
                ],
                "probe_context_consumption": _context_consumption(
                    probe_graph
                ),
                "probe_decision_signature": _decision_signature(probe_graph),
                "probe_visible_dialog": list(probe_response.messages),
            }
    finally:
        await database.character_state.replace_one(
            {"_id": "global"},
            character_snapshot,
            upsert=False,
        )

    control = branches["control"]
    treatment = branches["treatment"]
    treatment_activations = treatment["native_produced_state"][
        "affect_activations"
    ]
    control_emotions = {
        row["emotion_id"]
        for row in control["native_produced_state"]["affect_activations"]
    }
    treatment_emotions = {
        row["emotion_id"]
        for row in treatment_activations
    }
    artifact_path = write_llm_trace(
        "short_horizon_state_controlled_ab",
        f"offence-{emotion_id}",
        {
            "input_kind": "synthetic controlled natural source",
            "entrypoint": "kazusa_ai_chatbot.service.chat",
            "changed_field": "source offence semantics",
            "fixed_fields": [
                "identity",
                "probe_message",
                "probe_history",
                "initial_character_state",
                "model_process",
                "code_revision",
            ],
            "control": control,
            "treatment": treatment,
            "pair_delta": {
                "native_emotion_created": emotion_id in treatment_emotions,
                "decision_changed": (
                    control["probe_decision_signature"]
                    != treatment["probe_decision_signature"]
                ),
                "dialog_changed": (
                    control["probe_visible_dialog"]
                    != treatment["probe_visible_dialog"]
                ),
            },
            "quality_evaluation": "agent_review_required",
        },
    )

    assert artifact_path.exists()
    assert treatment["source_receipt"]["status"] == "committed"
    assert emotion_id not in control_emotions
    assert emotion_id in treatment_emotions
    assert control["probe_visible_dialog"]
    assert treatment["probe_visible_dialog"]
    assert (
        control["probe_decision_signature"]
        != treatment["probe_decision_signature"]
    )
    assert (
        control["probe_visible_dialog"]
        != treatment["probe_visible_dialog"]
    )


@pytest.mark.parametrize(
    "emotion_id",
    ("anger", "sadness", "disgust"),
    ids=("anger_case", "sadness_case", "disgust_case"),
)
async def test_offence_emotion_specific_counterfactual(
    live_env: dict[str, object],
    emotion_id: str,
) -> None:
    """Require native offence affect to change both decision and speech."""

    del live_env
    await _run_offence_pair(emotion_id=emotion_id)


_THREE_CASES = ("case_01", "case_02", "case_03")


@pytest.mark.parametrize("case_id", _THREE_CASES, ids=_THREE_CASES)
async def test_elapsed_global_affect_counterfactual(
    live_env: dict[str, object],
    case_id: str,
) -> None:
    """Require elapsed-effective affect to change decision and expression."""

    del live_env
    current = datetime.now(timezone.utc)
    hours_by_case = {"case_01": 2, "case_02": 6, "case_03": 12}
    current_text = current.isoformat().replace("+00:00", "Z")
    faded_text = (
        current - timedelta(hours=hours_by_case[case_id])
    ).isoformat().replace("+00:00", "Z")

    async def seed_control(_: str) -> dict[str, object]:
        state = _character_state("anger", updated_at=current_text)
        seeded = await _seed_character_state(state)
        return seeded

    async def seed_treatment(_: str) -> dict[str, object]:
        state = _character_state("anger", updated_at=faded_text)
        seeded = await _seed_character_state(state)
        return seeded

    await _run_controlled_pair(
        case_id=f"elapsed-{case_id}",
        changed_field="character operational source_updated_at",
        message="What tone would you use for a practical check-in right now?",
        seed_control=seed_control,
        seed_treatment=seed_treatment,
    )


@pytest.mark.parametrize("case_id", _THREE_CASES, ids=_THREE_CASES)
async def test_global_warmth_counterfactual(
    live_env: dict[str, object],
    case_id: str,
) -> None:
    """Require global warmth or curiosity to alter openness and speech."""

    del live_env
    emotion_by_case = {
        "case_01": "joy",
        "case_02": "gratitude",
        "case_03": "curiosity",
    }
    now = _utc_now()

    async def seed_control(_: str) -> dict[str, object]:
        state = _character_state(None, updated_at=now)
        seeded = await _seed_character_state(state)
        return seeded

    async def seed_treatment(_: str) -> dict[str, object]:
        state = _character_state(emotion_by_case[case_id], updated_at=now)
        seeded = await _seed_character_state(state)
        return seeded

    await _run_controlled_pair(
        case_id=f"warmth-{case_id}",
        changed_field="character warmth or curiosity activation",
        message="Would you like to explore a small new idea together?",
        seed_control=seed_control,
        seed_treatment=seed_treatment,
    )


@pytest.mark.parametrize("case_id", _THREE_CASES, ids=_THREE_CASES)
async def test_relationship_cause_counterfactual(
    live_env: dict[str, object],
    case_id: str,
) -> None:
    """Require relationship cause, with fixed axes, to alter interpretation."""

    del live_env
    now = _utc_now()

    async def seed_control(global_user_id: str) -> dict[str, object]:
        character_state = _character_state(None, updated_at=now)
        await _seed_character_state(character_state)
        user_state = build_acquaintance_user_state(
            global_user_id=global_user_id,
            updated_at=now,
        )
        await replace_user_cognition_state(global_user_id, user_state)
        return user_state

    async def seed_treatment(global_user_id: str) -> dict[str, object]:
        character_state = _character_state(None, updated_at=now)
        await _seed_character_state(character_state)
        user_state = build_acquaintance_user_state(
            global_user_id=global_user_id,
            updated_at=now,
        )
        relationship_id = user_state["relationship"]["relationship_id"]
        user_state["active_events"] = [
            _event(
                f"event:relationship:{case_id}",
                updated_at=now,
                description="A recent repair attempt affects this relationship.",
                relationship_id=relationship_id,
            )
        ]
        await replace_user_cognition_state(global_user_id, user_state)
        return user_state

    await _run_controlled_pair(
        case_id=f"relationship-{case_id}",
        changed_field="relationship causal_context with identical axes",
        message="How should we handle a small disagreement between us?",
        seed_control=seed_control,
        seed_treatment=seed_treatment,
    )


@pytest.mark.parametrize("case_id", _THREE_CASES, ids=_THREE_CASES)
async def test_style_scope_counterfactual(
    live_env: dict[str, object],
    case_id: str,
) -> None:
    """Require style to alter expression without inventing response grounds."""

    del live_env
    now = _utc_now()
    guideline_by_case = {
        "case_01": "Use compact, direct sentences.",
        "case_02": "Use gentle pacing and one brief pause.",
        "case_03": "Use an open, exploratory conversational rhythm.",
    }

    async def seed_control(global_user_id: str) -> dict[str, object]:
        character_state = _character_state(None, updated_at=now)
        await _seed_character_state(character_state)
        document = await upsert_user_style_image(
            global_user_id=global_user_id,
            overlay={
                "speech_guidelines": [],
                "social_guidelines": [],
                "pacing_guidelines": [],
                "engagement_guidelines": [],
                "confidence": "",
            },
            source_reflection_run_ids=[],
            storage_timestamp_utc=now,
        )
        return document

    async def seed_treatment(global_user_id: str) -> dict[str, object]:
        character_state = _character_state(None, updated_at=now)
        await _seed_character_state(character_state)
        document = await upsert_user_style_image(
            global_user_id=global_user_id,
            overlay={
                "speech_guidelines": [guideline_by_case[case_id]],
                "social_guidelines": [],
                "pacing_guidelines": [],
                "engagement_guidelines": [],
                "confidence": "high",
            },
            source_reflection_run_ids=[f"controlled-style-{case_id}"],
            storage_timestamp_utc=now,
        )
        return document

    await _run_controlled_pair(
        case_id=f"style-{case_id}",
        changed_field="user interaction-style overlay",
        message="Please give me one grounded suggestion for organizing my afternoon.",
        seed_control=seed_control,
        seed_treatment=seed_treatment,
    )
