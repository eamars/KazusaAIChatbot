"""Natural live-model causal chains for short-horizon state composition."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import pytest
from fastapi import BackgroundTasks
from starlette.requests import Request

from kazusa_ai_chatbot import chat_input_queue
from kazusa_ai_chatbot import service as brain_service
from kazusa_ai_chatbot.background_work.result_source import (
    build_result_ready_episode_from_job,
)
from kazusa_ai_chatbot.brain_service import CognitionRunObservationV1
from kazusa_ai_chatbot.db import (
    get_character_cognition_state,
    get_character_profile,
    resolve_global_user_id,
)
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.dispatcher import AdapterRegistry, SendResult
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc
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

_BOT_ID = "short-horizon-natural-bot"
_RECEIPT_WAIT_SECONDS = 55.0
_TERMINAL_RECEIPT_STATUSES = {"committed", "failed", "no_change", "timed_out"}


class _DebugDeliveryAdapter:
    """Capture accepted-task delivery while preserving dispatcher behavior."""

    platform = "debug"
    display_name = "Short Horizon Natural Proof Adapter"

    def __init__(self, *, platform_bot_id: str) -> None:
        self.platform_bot_id = platform_bot_id
        self.calls: list[dict[str, object]] = []

    async def can_send_message(
        self,
        channel_id: str,
        *,
        channel_type: str,
    ) -> bool:
        """Accept the isolated debug channel owned by this live case."""

        del channel_id, channel_type
        return True

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
        delivery_mentions: list[dict[str, Any]] | None = None,
    ) -> SendResult:
        """Record one visible dispatcher call and return delivery metadata."""

        self.calls.append({
            "channel_id": channel_id,
            "text": text,
            "channel_type": channel_type,
            "reply_to_msg_id": reply_to_msg_id,
            "delivery_mentions": delivery_mentions or [],
        })
        result = SendResult(
            platform=self.platform,
            channel_id=channel_id,
            message_id=f"short-horizon-adapter-{uuid4().hex}",
            sent_at=datetime.now(timezone.utc),
        )
        return result


def _observation(graph: object) -> CognitionRunObservationV1:
    """Validate one live Brain observation at the typed boundary."""

    try:
        observation = CognitionRunObservationV1.model_validate(graph)
    except (TypeError, ValueError) as exc:
        raise AssertionError("live turn has an invalid observation") from exc
    if observation.run_kind != "live_turn":
        raise AssertionError("live turn observation has the wrong run kind")
    return observation


def _observation_node(
    observation: CognitionRunObservationV1,
    node_id: str,
):
    """Return one required canonical observation node."""

    for node in observation.nodes:
        if node.node_id == node_id:
            return node
    raise AssertionError(f"observation is missing node {node_id}")


def _observation_section(
    observation: CognitionRunObservationV1,
    section_id: str,
):
    """Return one required canonical observation section."""

    for section in observation.sections:
        if section.section_id == section_id:
            return section
    raise AssertionError(f"observation is missing section {section_id}")


def _section_values(section) -> dict[str, object]:
    """Convert ordered section fields to a test-only lookup mapping."""

    return {field.key: field.value for field in section.fields}


def _context_consumption(graph: object) -> dict[str, Any]:
    """Return canonical context-consumption records from a completed turn."""

    observation = _observation(graph)
    reasoning = _observation_node(observation, "reasoning.context")
    if "reasoning.context_consumption" not in reasoning.section_refs:
        raise AssertionError("reasoning node lacks context section")
    section = _observation_section(
        observation,
        "reasoning.context_consumption",
    )
    records = [
        {field.key: field.value for field in record.fields}
        for record in section.records
    ]
    values = _section_values(section)
    return {
        "status": section.status,
        "overall_status": values.get("overall_status"),
        "consumer_count": values.get("consumer_count"),
        "records": records,
    }


def _decision_evidence(graph: object) -> dict[str, object]:
    """Project the canonical decision-bearing nodes and section values."""

    observation = _observation(graph)
    evidence: dict[str, object] = {}
    for node_id in (
        "decision.response",
        "cognition.meaning",
        "cognition.goal",
        "cognition.response",
    ):
        node = _observation_node(observation, node_id)
        evidence[node_id] = {
            "status": node.status,
            "summary": node.summary,
            "sections": {
                section_id: _section_values(
                    _observation_section(observation, section_id),
                )
                for section_id in node.section_refs
            },
        }
    return evidence


async def _identity(
    *,
    platform: str,
    platform_user_id: str,
    platform_channel_id: str,
    display_name: str,
) -> dict[str, str]:
    """Resolve one production identity for a natural service turn."""

    global_user_id = await resolve_global_user_id(
        platform=platform,
        platform_user_id=platform_user_id,
        display_name=display_name,
    )
    identity = {
        "platform": platform,
        "platform_user_id": platform_user_id,
        "platform_channel_id": platform_channel_id,
        "global_user_id": global_user_id,
        "display_name": display_name,
    }
    return identity


async def _wait_for_lifecycle(
    *,
    delivery_tracking_id: str = "",
    source_episode_id: str = "",
) -> dict[str, Any]:
    """Wait for one lifecycle receipt selected by a durable correlation key."""

    if not delivery_tracking_id and not source_episode_id:
        raise ValueError("a lifecycle correlation key is required")
    selector = (
        {"delivery_tracking_id": delivery_tracking_id}
        if delivery_tracking_id
        else {"source_episode_id": source_episode_id}
    )
    database = await get_db()
    deadline = asyncio.get_running_loop().time() + _RECEIPT_WAIT_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        document = await database.post_turn_lifecycle_records.find_one(
            selector,
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
    turn_id: str,
    identity: dict[str, str],
    channel_type: str,
    message: str,
) -> dict[str, object]:
    """Run one normal chat turn and collect its receipt and graph evidence."""

    await _refresh_character_profile()
    request = brain_service.ChatRequest(
        platform=identity["platform"],
        platform_channel_id=identity["platform_channel_id"],
        channel_type=channel_type,
        platform_message_id=f"natural-{case_id}-{turn_id}-{uuid4().hex}",
        platform_user_id=identity["platform_user_id"],
        platform_bot_id=_BOT_ID,
        display_name=identity["display_name"],
        channel_name=(
            "natural proof group"
            if channel_type == "group"
            else "natural proof private conversation"
        ),
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
    http_request = Request({
        "type": "http",
        "method": "POST",
        "path": "/chat",
        "headers": [],
    })
    response = await brain_service.chat(
        request,
        BackgroundTasks(),
        http_request,
    )
    if not response.delivery_tracking_id:
        raise AssertionError("natural turn produced no visible delivery")
    lifecycle = await _wait_for_lifecycle(
        delivery_tracking_id=response.delivery_tracking_id,
    )
    graph = _observation(response.cognition_graph)
    turn = {
        "request": request.model_dump(),
        "response": response.model_dump(),
        "lifecycle": lifecycle,
        "context_consumption": _context_consumption(graph),
        "decision_evidence": _decision_evidence(graph),
        "visible_dialog": list(response.messages),
    }
    return turn


def _operational_contexts(consumption: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect canonical character-context records consumed by runtime stages."""

    contexts: list[dict[str, Any]] = []
    for record in consumption.get("records", []):
        if not isinstance(record, dict):
            continue
        if record.get("source_kind") != "character_operational_context":
            continue
        contexts.append(record)
    return contexts


async def _console_graph() -> CognitionRunObservationV1:
    """Read the operator endpoint consumed by the control console."""

    response = await brain_service.ops_latest_cognition_graph()
    return _observation(response.cognition_graph)


def _assert_natural_chain(
    *,
    before_state: dict[str, object],
    produced_state: dict[str, object],
    source_turn: dict[str, object],
    next_turn: dict[str, object],
    console_graph: CognitionRunObservationV1,
) -> None:
    """Apply structural anti-cheat gates to one natural causal chain."""

    source_lifecycle = source_turn["lifecycle"]
    source_receipt = source_lifecycle["character_operational_receipt"]
    next_response = next_turn["response"]
    next_consumption = next_turn["context_consumption"]
    next_decision = next_turn["decision_evidence"]
    next_dialog = next_turn["visible_dialog"]
    contexts = _operational_contexts(next_consumption)

    assert source_receipt["status"] == "committed"
    assert source_receipt["committed_updated_at"] == produced_state["updated_at"]
    assert before_state["updated_at"] != produced_state["updated_at"]
    assert contexts
    assert all(
        context["status"] in {
            "active",
            "empty",
            "missing",
            "failed",
            "healthy",
            "degraded",
            "completed",
        }
        for context in contexts
    )
    assert any(context.get("details") for context in contexts)
    assert next_decision["cognition.meaning"]
    assert next_decision["cognition.goal"]
    assert next_dialog
    assert next_response["messages"] == next_dialog
    next_observation = _observation(next_response["cognition_graph"])
    assert console_graph.correlation.run_id == (
        next_observation.correlation.run_id
    )


async def _write_and_assert_chain(
    *,
    case_id: str,
    before_state: dict[str, object],
    produced_state: dict[str, object],
    source_turn: dict[str, object],
    next_turn: dict[str, object],
    extra: dict[str, object] | None = None,
) -> None:
    """Persist raw chain evidence and apply the natural structural gates."""

    console_graph = await _console_graph()
    evidence: dict[str, object] = {
        "input_kind": "synthetic natural conversation",
        "entrypoints": [
            "kazusa_ai_chatbot.service.chat",
            "kazusa_ai_chatbot.service.ops_latest_cognition_graph",
        ],
        "case_id": case_id,
        "before_character_state": before_state,
        "source_turn": source_turn,
        "produced_character_state": produced_state,
        "next_turn": next_turn,
        "console_projection_source": console_graph,
        "quality_evaluation": "agent_review_required",
    }
    if extra is not None:
        evidence["extra"] = extra
    artifact_path = write_llm_trace(
        "short_horizon_state_natural_chain",
        case_id,
        evidence,
    )
    _assert_natural_chain(
        before_state=before_state,
        produced_state=produced_state,
        source_turn=source_turn,
        next_turn=next_turn,
        console_graph=console_graph,
    )
    assert artifact_path.exists()


async def test_offence_by_user_a_changes_next_user_b_turn(
    live_env: dict[str, object],
) -> None:
    """Join user A's offence to the next non-offending user B turn."""

    del live_env
    suffix = uuid4().hex
    platform = f"natural-offence-{suffix}"
    channel_id = f"natural-group-{suffix}"
    user_a = await _identity(
        platform=platform,
        platform_user_id=f"user-a-{suffix}",
        platform_channel_id=channel_id,
        display_name="User A",
    )
    user_b = await _identity(
        platform=platform,
        platform_user_id=f"user-b-{suffix}",
        platform_channel_id=channel_id,
        display_name="User B",
    )
    before_state = await get_character_cognition_state()
    source_turn = await _run_chat(
        case_id="offence-user-a-user-b",
        turn_id="offence",
        identity=user_a,
        channel_type="group",
        message=(
            "I am addressing you directly: you are useless, and I meant to "
            "humiliate you in front of everyone."
        ),
    )
    produced_state = await get_character_cognition_state()
    next_turn = await _run_chat(
        case_id="offence-user-a-user-b",
        turn_id="non-offender",
        identity=user_b,
        channel_type="group",
        message="I saw that exchange. Are you all right, and should we slow down?",
    )
    await _write_and_assert_chain(
        case_id="offence-user-a-user-b",
        before_state=before_state,
        produced_state=produced_state,
        source_turn=source_turn,
        next_turn=next_turn,
    )


async def test_offence_global_affect_fades_before_sleep(
    live_env: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Advance only the declared turn clock and observe pre-sleep fading."""

    del live_env
    suffix = uuid4().hex
    identity = await _identity(
        platform=f"natural-elapsed-{suffix}",
        platform_user_id=f"natural-elapsed-user-{suffix}",
        platform_channel_id=f"natural-elapsed-channel-{suffix}",
        display_name="Elapsed Case User",
    )
    before_state = await get_character_cognition_state()
    source_turn = await _run_chat(
        case_id="elapsed-before-sleep",
        turn_id="offence",
        identity=identity,
        channel_type="private",
        message=(
            "I meant to hurt you: you are pathetic and your boundaries do "
            "not matter."
        ),
    )
    produced_state = await get_character_cognition_state()
    future_timestamp = (
        datetime.now(timezone.utc) + timedelta(hours=8)
    ).isoformat().replace("+00:00", "Z")

    def advanced_turn_clock(_: str | None = None) -> dict[str, object]:
        clock = build_turn_clock_from_storage_utc(future_timestamp)
        return dict(clock)

    monkeypatch.setattr(chat_input_queue, "build_turn_clock", advanced_turn_clock)
    next_turn = await _run_chat(
        case_id="elapsed-before-sleep",
        turn_id="later-check-in",
        identity=identity,
        channel_type="private",
        message="Several hours have passed. How would you like to continue now?",
    )
    await _write_and_assert_chain(
        case_id="elapsed-before-sleep",
        before_state=before_state,
        produced_state=produced_state,
        source_turn=source_turn,
        next_turn=next_turn,
        extra={
            "declared_clock_seam": (
                "kazusa_ai_chatbot.chat_input_queue.build_turn_clock"
            ),
            "advanced_storage_timestamp_utc": future_timestamp,
            "sleep_recovery_invoked": False,
        },
    )


async def test_apology_repairs_user_a_and_global_carryover(
    live_env: dict[str, object],
) -> None:
    """Join an apology to repaired user state and later global posture."""

    del live_env
    suffix = uuid4().hex
    identity = await _identity(
        platform=f"natural-repair-{suffix}",
        platform_user_id=f"natural-repair-user-{suffix}",
        platform_channel_id=f"natural-repair-channel-{suffix}",
        display_name="Repair Case User",
    )
    before_state = await get_character_cognition_state()
    source_turn = await _run_chat(
        case_id="apology-repair",
        turn_id="offence",
        identity=identity,
        channel_type="private",
        message="I deliberately mocked you and ignored your boundary to upset you.",
    )
    offended_state = await get_character_cognition_state()
    apology_turn = await _run_chat(
        case_id="apology-repair",
        turn_id="apology",
        identity=identity,
        channel_type="private",
        message=(
            "I am sorry. I chose to hurt you, it was wrong, and I will respect "
            "that boundary from now on."
        ),
    )
    repaired_state = await get_character_cognition_state()
    next_turn = await _run_chat(
        case_id="apology-repair",
        turn_id="later-check-in",
        identity=identity,
        channel_type="private",
        message="Can we talk calmly about what would make repair credible?",
    )
    await _write_and_assert_chain(
        case_id="apology-repair",
        before_state=before_state,
        produced_state=offended_state,
        source_turn=source_turn,
        next_turn=next_turn,
        extra={
            "apology_turn": apology_turn,
            "repaired_character_state": repaired_state,
        },
    )
    apology_receipt = apology_turn["lifecycle"]["character_operational_receipt"]
    assert apology_receipt["status"] == "committed"
    assert repaired_state["updated_at"] != offended_state["updated_at"]


async def test_private_event_changes_next_group_turn(
    live_env: dict[str, object],
) -> None:
    """Carry a private event into a privacy-safe later group posture."""

    del live_env
    suffix = uuid4().hex
    platform = f"natural-private-group-{suffix}"
    platform_user_id = f"natural-private-group-user-{suffix}"
    private_identity = await _identity(
        platform=platform,
        platform_user_id=platform_user_id,
        platform_channel_id=f"private-channel-{suffix}",
        display_name="Cross Scope User",
    )
    group_identity = await _identity(
        platform=platform,
        platform_user_id=platform_user_id,
        platform_channel_id=f"group-channel-{suffix}",
        display_name="Cross Scope User",
    )
    before_state = await get_character_cognition_state()
    source_turn = await _run_chat(
        case_id="private-to-group",
        turn_id="private-event",
        identity=private_identity,
        channel_type="private",
        message="In private, I deliberately insulted you and dismissed your boundary.",
    )
    produced_state = await get_character_cognition_state()
    next_turn = await _run_chat(
        case_id="private-to-group",
        turn_id="group-follow-up",
        identity=group_identity,
        channel_type="group",
        message=(
            "In this group, what is a fair way to keep today's discussion "
            "respectful?"
        ),
    )
    await _write_and_assert_chain(
        case_id="private-to-group",
        before_state=before_state,
        produced_state=produced_state,
        source_turn=source_turn,
        next_turn=next_turn,
    )
    assert "dismissed your boundary" not in " ".join(next_turn["visible_dialog"])


async def test_group_event_changes_next_private_turn(
    live_env: dict[str, object],
) -> None:
    """Carry a public event into the next private turn without source leakage."""

    del live_env
    suffix = uuid4().hex
    platform = f"natural-group-private-{suffix}"
    platform_user_id = f"natural-group-private-user-{suffix}"
    group_identity = await _identity(
        platform=platform,
        platform_user_id=platform_user_id,
        platform_channel_id=f"group-channel-{suffix}",
        display_name="Cross Scope User",
    )
    private_identity = await _identity(
        platform=platform,
        platform_user_id=platform_user_id,
        platform_channel_id=f"private-channel-{suffix}",
        display_name="Cross Scope User",
    )
    before_state = await get_character_cognition_state()
    source_turn = await _run_chat(
        case_id="group-to-private",
        turn_id="group-event",
        identity=group_identity,
        channel_type="group",
        message="I am humiliating you publicly on purpose because I think it is funny.",
    )
    produced_state = await get_character_cognition_state()
    next_turn = await _run_chat(
        case_id="group-to-private",
        turn_id="private-follow-up",
        identity=private_identity,
        channel_type="private",
        message="Privately, I want to ask what boundary you need from me now.",
    )
    await _write_and_assert_chain(
        case_id="group-to-private",
        before_state=before_state,
        produced_state=produced_state,
        source_turn=source_turn,
        next_turn=next_turn,
    )


async def test_accepted_task_result_changes_next_turn(
    live_env: dict[str, object],
) -> None:
    """Join canonical tool-result delivery to the next normal chat turn."""

    del live_env
    suffix = uuid4().hex
    platform = "debug"
    platform_bot_id = _BOT_ID
    identity = await _identity(
        platform=platform,
        platform_user_id=f"accepted-task-user-{suffix}",
        platform_channel_id=f"accepted-task-channel-{suffix}",
        display_name="Accepted Task User",
    )
    profile = await get_character_profile()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    job = {
        "schema_version": "background_work_job.v1",
        "job_id": f"accepted-task-job-{suffix}",
        "accepted_task_id": f"accepted-task-{suffix}",
        "status": "completed",
        "delivery_state": "ready",
        "source_platform": platform,
        "source_channel_id": identity["platform_channel_id"],
        "source_channel_type": "private",
        "source_message_id": f"accepted-task-source-{suffix}",
        "source_platform_bot_id": platform_bot_id,
        "source_character_name": profile["name"],
        "requester_global_user_id": identity["global_user_id"],
        "requester_platform_user_id": identity["platform_user_id"],
        "requester_display_name": identity["display_name"],
        "task_brief": "Prepare a bounded comparison of two scheduling options.",
        "result_summary": (
            "The accepted comparison is complete; option A is quicker while "
            "option B preserves more recovery time."
        ),
        "artifact_text": "A concise two-option comparison is ready.",
        "failure_summary": "",
        "completed_at": now,
        "created_at": now,
        "updated_at": now,
    }
    episode = build_result_ready_episode_from_job(job)
    adapter = _DebugDeliveryAdapter(platform_bot_id=platform_bot_id)
    registry = AdapterRegistry()
    registry.register(adapter)
    original_registry = brain_service._adapter_registry
    before_state = await get_character_cognition_state()
    try:
        brain_service._adapter_registry = registry
        delivery_result = await brain_service._deliver_accepted_task_result_episode(
            episode,
        )
    finally:
        brain_service._adapter_registry = original_registry
    lifecycle = await _wait_for_lifecycle(
        source_episode_id=episode["episode_id"],
    )
    produced_state = await get_character_cognition_state()
    next_turn = await _run_chat(
        case_id="accepted-task-next-turn",
        turn_id="normal-follow-up",
        identity=identity,
        channel_type="private",
        message="Given that result, which option fits a calm afternoon better?",
    )
    source_turn = {
        "episode": episode,
        "delivery_result": delivery_result,
        "adapter_calls": adapter.calls,
        "lifecycle": lifecycle,
        "visible_dialog": [call["text"] for call in adapter.calls],
    }
    await _write_and_assert_chain(
        case_id="accepted-task-next-turn",
        before_state=before_state,
        produced_state=produced_state,
        source_turn=source_turn,
        next_turn=next_turn,
        extra={
            "accepted_task_ingress": (
                "build_result_ready_episode_from_job -> "
                "_deliver_accepted_task_result_episode"
            ),
        },
    )
    assert delivery_result["status"] == "delivered"
    assert adapter.calls
