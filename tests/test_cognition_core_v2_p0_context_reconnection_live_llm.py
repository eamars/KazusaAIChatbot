"""Live E2E evidence for the Cognition Core V2 P0 context reconnections."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from time import perf_counter
from uuid import uuid4

import pytest

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot import service as brain_service
from kazusa_ai_chatbot.cognition_core_v2 import facade as cognition_facade
from kazusa_ai_chatbot.db import build_memory_doc, save_memory
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.interaction_style_images import (
    upsert_group_channel_style_image,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as connector
from kazusa_ai_chatbot.self_cognition import models, projection, runner
from tests.llm_trace import write_llm_trace
from tests.test_e2e_live_llm import (
    _BOT_ID,
    _make_identity,
    _neutral_character_runtime_state,
    _refresh_character_profile,
    _run_chat,
    _seed_conversation,
    live_env,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

logger = logging.getLogger(__name__)

_TERMINAL_TRACE_STATUSES = {"completed", "failed", "succeeded"}


async def _trace_documents(
    trace_id: str,
    *,
    terminalize_guarded_run: bool = False,
) -> dict[str, object]:
    """Load one terminal protected trace run and its ordered steps.

    Args:
        trace_id: Protected trace correlation identifier.
        terminalize_guarded_run: Whether this test may close a run left open
            by an already completed guarded intake.

    Returns:
        Trace metadata, ordered steps, and test-terminalization provenance.
    """

    database = await get_db()
    trace_run = None
    for _ in range(200):
        trace_run = await database.llm_trace_runs.find_one(
            {"trace_id": trace_id},
            {"_id": 0},
        )
        if (
            isinstance(trace_run, dict)
            and trace_run.get("status") in _TERMINAL_TRACE_STATUSES
        ):
            break
        await asyncio.sleep(0.05)
    assert isinstance(trace_run, dict)
    trace_steps = None
    test_terminalized_trace = False
    if (
        trace_run.get("status") not in _TERMINAL_TRACE_STATUSES
        and terminalize_guarded_run
    ):
        trace_steps = await (
            database.llm_trace_steps.find(
                {"trace_id": trace_id},
                {"_id": 0},
            )
            .sort("sequence", 1)
            .to_list(length=None)
        )
        await llm_tracing.finalize_llm_trace_run(
            trace_id=trace_id,
            status="completed",
            final_dialog_count=1,
            delivery_tracking_id="",
        )
        trace_run = await database.llm_trace_runs.find_one(
            {"trace_id": trace_id},
            {"_id": 0},
        )
        test_terminalized_trace = True
    assert isinstance(trace_run, dict)
    assert trace_run.get("status") in _TERMINAL_TRACE_STATUSES
    if trace_steps is None:
        trace_steps = await (
            database.llm_trace_steps.find(
                {"trace_id": trace_id},
                {"_id": 0},
            )
            .sort("sequence", 1)
            .to_list(length=None)
        )
    return {
        "trace_run": trace_run,
        "trace_steps": trace_steps,
        "test_terminalized_trace": test_terminalized_trace,
    }


async def _run_terminal_self_cognition_trace(
    state: dict[str, object],
    *,
    trace_id: str,
) -> dict[str, object]:
    """Run self-cognition and terminalize its manually created trace.

    Args:
        state: Canonical self-cognition graph state.
        trace_id: Protected trace run owned by this invocation.

    Returns:
        The cognition connector update.
    """

    trace_token = llm_tracing.bind_trace_id(trace_id)
    try:
        output = await runner._default_cognition_client(state)
    except (Exception, asyncio.CancelledError):
        await llm_tracing.finalize_llm_trace_run(
            trace_id=trace_id,
            status="failed",
            final_dialog_count=0,
            delivery_tracking_id="",
        )
        raise
    else:
        await llm_tracing.finalize_llm_trace_run(
            trace_id=trace_id,
            status="completed",
            final_dialog_count=0,
            delivery_tracking_id="",
        )
        return output
    finally:
        llm_tracing.reset_trace_id(trace_token)


async def _latest_scope_trace_id(
    *,
    platform: str,
    platform_channel_id: str,
) -> str:
    """Return the newest protected trace id for one guarded test scope."""

    database = await get_db()
    trace_run = await database.llm_trace_runs.find_one(
        {
            "platform": platform,
            "platform_channel_id": platform_channel_id,
        },
        {"_id": 0, "trace_id": 1},
        sort=[("started_at", -1)],
    )
    assert isinstance(trace_run, dict)
    trace_id = trace_run.get("trace_id")
    assert isinstance(trace_id, str) and trace_id
    return trace_id


def _serialized(value: object) -> str:
    """Serialize structured evidence for exact marker assertions."""

    return json.dumps(value, ensure_ascii=False, default=str)


def _steps_for_prefix(
    trace_steps: list[dict[str, object]],
    prefix: str,
) -> list[dict[str, object]]:
    """Return trace steps whose stage name starts with one prefix."""

    return [
        step
        for step in trace_steps
        if str(step.get("stage_name", "")).startswith(prefix)
    ]


async def test_live_normal_chat_prewarm_recalls_guarded_shared_memory(
    live_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary chat carries real prewarm memory evidence into V2 cognition."""

    del live_env
    nonce = f"napcat-{uuid4().hex[:12]}"
    identity = await _make_identity("p0-prewarm", "P0PrewarmUser")
    memory_name = f"p0-prewarm-{nonce}"
    memory_content = (
        f"{nonce} distinctly prefers thunderstorms over clear weather."
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    memory_doc = build_memory_doc(
        memory_name=memory_name,
        content=memory_content,
        source_global_user_id=identity["global_user_id"],
        memory_type="fact",
        source_kind="conversation_extracted",
        confidence_note="p0_context_reconnection_live_seed",
    )
    await save_memory(memory_doc, timestamp)

    original_prewarm = connector.run_first_cycle_shared_memory_prewarm
    prewarm_results: list[dict[str, object]] = []
    original_run_cognition = connector.run_cognition
    cognition_inputs: list[dict[str, object]] = []
    cognition_outputs: list[dict[str, object]] = []

    async def capture_prewarm(state: dict[str, object]) -> dict[str, object]:
        result = await original_prewarm(state)  # type: ignore[arg-type]
        prewarm_results.append(result)
        return result

    async def capture_cognition(
        cognition_input: dict[str, object],
        services: object,
    ) -> dict[str, object]:
        cognition_inputs.append(dict(cognition_input))
        result = await original_run_cognition(
            cognition_input,  # type: ignore[arg-type]
            services,  # type: ignore[arg-type]
        )
        cognition_outputs.append(dict(result))
        return result  # type: ignore[return-value]

    monkeypatch.setattr(
        connector,
        "run_first_cycle_shared_memory_prewarm",
        capture_prewarm,
    )
    monkeypatch.setattr(
        connector,
        "run_cognition",
        capture_cognition,
    )
    platform_message_id = f"p0-prewarm-message-{uuid4().hex[:12]}"
    response = None
    trace_id = ""
    trace_documents: dict[str, object] = {}
    artifact_path = None
    elapsed_seconds = 0.0
    database = await get_db()
    try:
        started_at = perf_counter()
        async with _neutral_character_runtime_state():
            response, _ = await _run_chat(
                "p0-prewarm-reconnection",
                identity["display_name"],
                (
                    "What distinctive weather preference is associated with "
                    f"{nonce}?"
                ),
                platform=identity["platform"],
                platform_user_id=identity["platform_user_id"],
                platform_channel_id=identity["platform_channel_id"],
                platform_message_id=platform_message_id,
                direct_address=True,
            )
        elapsed_seconds = perf_counter() - started_at
        trace_id = await _latest_scope_trace_id(
            platform=identity["platform"],
            platform_channel_id=identity["platform_channel_id"],
        )
        trace_documents = await _trace_documents(trace_id)
        artifact_path = write_llm_trace(
            "cognition_core_v2_p0_context_reconnection",
            "shared_memory_prewarm",
            {
                "input": {
                    "platform": identity["platform"],
                    "platform_channel_id": identity["platform_channel_id"],
                    "platform_message_id": platform_message_id,
                    "text": (
                        "What distinctive weather preference is associated "
                        f"with {nonce}?"
                    ),
                    "guarded_memory_name": memory_name,
                    "guarded_memory_content": memory_content,
                },
                "response": response.model_dump(),
                "trace_id": trace_id,
                "elapsed_seconds": elapsed_seconds,
                "captured_prewarm_results": prewarm_results,
                "captured_cognition_inputs": cognition_inputs,
                "captured_cognition_outputs": cognition_outputs,
                **trace_documents,
            },
        )
    finally:
        await database.memory.delete_many({"memory_name": memory_name})

    assert response is not None
    assert len(prewarm_results) == 1
    assert prewarm_results[0].get("answer") == ""
    assert nonce in _serialized(prewarm_results[0].get("memory_evidence"))
    assert len(cognition_inputs) == 1
    assert nonce in _serialized(cognition_inputs[0]["evidence"])
    assert len(cognition_outputs) == 1
    trace_steps = trace_documents["trace_steps"]
    assert isinstance(trace_steps, list)
    goal_steps = _steps_for_prefix(trace_steps, "goal_cognition.")
    assert goal_steps
    assert "persistent_memory_search_agent" in _serialized(
        prewarm_results[0].get("supervisor_trace")
    )
    logger.info(
        f"P0_PREWARM_LIVE trace_id={trace_id} "
        f"artifact={artifact_path} elapsed_seconds={elapsed_seconds:.3f} "
        f"response={response.messages}"
    )


async def test_live_reply_residual_reaches_goal_only(
    live_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trace-backed replied dialog exposes residual only to V2 goals."""

    del live_env
    identity = await _make_identity("p0-past-dialog", "P0PastDialogUser")
    character_profile = await _refresh_character_profile()
    character_name = str(character_profile["name"])
    residual_marker = f"private-residual-{uuid4().hex[:12]}"
    old_trace_id = f"llmtrace_p0_old_{uuid4().hex}"
    old_step_id = f"{old_trace_id}_{uuid4().hex}"
    prior_message_id = f"p0-prior-dialog-{uuid4().hex[:12]}"
    database = await get_db()
    await database.llm_trace_steps.insert_one({
        "step_id": old_step_id,
        "trace_id": old_trace_id,
        "sequence": 1,
        "stage_name": "l2a_conscious_framing",
        "parsed_output": {
            "internal_monologue": (
                f"{residual_marker}: I withheld a premature conclusion."
            ),
            "logical_stance": "DIVERGE",
            "character_intent": "CLARIFY",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    await _seed_conversation(
        platform=identity["platform"],
        platform_channel_id=identity["platform_channel_id"],
        global_user_id=brain_service.CHARACTER_GLOBAL_USER_ID,
        display_name=character_name,
        content="I held back because the evidence was incomplete.",
        role="assistant",
        platform_user_id=_BOT_ID,
        platform_message_id=prior_message_id,
    )
    await database.conversation_history.update_one(
        {
            "platform": identity["platform"],
            "platform_channel_id": identity["platform_channel_id"],
            "platform_message_id": prior_message_id,
        },
        {"$set": {"llm_trace_id": old_trace_id}},
    )

    original_builder = connector.build_cognition_input_from_global_state
    mapped_inputs: list[dict[str, object]] = []

    def capture_input(
        state: dict[str, object],
        *,
        mutable_state=None,
        character_state=None,
    ):
        cognition_input = original_builder(
            state,  # type: ignore[arg-type]
            mutable_state=mutable_state,
            character_state=character_state,
        )
        mapped_inputs.append(dict(cognition_input))
        return cognition_input

    monkeypatch.setattr(
        connector,
        "build_cognition_input_from_global_state",
        capture_input,
    )
    original_goal_cognition = cognition_facade.run_goal_cognition
    goal_contexts: list[dict[str, object]] = []

    async def capture_goal_cognition(
        definition: object,
        goal_ref: object,
        semantic_context: dict[str, object],
        evidence: object,
        services: object,
    ) -> dict[str, object]:
        goal_contexts.append(dict(semantic_context))
        return await original_goal_cognition(
            definition,  # type: ignore[arg-type]
            goal_ref,  # type: ignore[arg-type]
            semantic_context,
            evidence,  # type: ignore[arg-type]
            services,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(
        cognition_facade,
        "run_goal_cognition",
        capture_goal_cognition,
    )
    response = None
    trace_id = ""
    trace_documents: dict[str, object] = {}
    artifact_path = None
    try:
        async with _neutral_character_runtime_state():
            response, _ = await _run_chat(
                "p0-past-dialog-reconnection",
                identity["display_name"],
                "What made you hesitate in that earlier answer?",
                channel_name="group",
                platform=identity["platform"],
                platform_user_id=identity["platform_user_id"],
                platform_channel_id=identity["platform_channel_id"],
                reply_context={
                    "reply_to_message_id": prior_message_id,
                    "reply_to_platform_user_id": _BOT_ID,
                    "reply_to_display_name": character_name,
                    "reply_excerpt": (
                        "I held back because the evidence was incomplete."
                    ),
                },
            )
        trace_id = await _latest_scope_trace_id(
            platform=identity["platform"],
            platform_channel_id=identity["platform_channel_id"],
        )
        trace_documents = await _trace_documents(
            trace_id,
            terminalize_guarded_run=True,
        )
        artifact_path = write_llm_trace(
            "cognition_core_v2_p0_context_reconnection",
            "past_dialog_goal_only",
            {
                "input": {
                    "reply_text": (
                        "What made you hesitate in that earlier answer?"
                    ),
                    "prior_visible_dialog": (
                        "I held back because the evidence was incomplete."
                    ),
                    "residual_marker": residual_marker,
                    "old_trace_id": old_trace_id,
                },
                "response": response.model_dump(),
                "trace_id": trace_id,
                "mapped_inputs": mapped_inputs,
                "captured_goal_contexts": goal_contexts,
                **trace_documents,
            },
        )
    finally:
        await database.conversation_history.delete_many({
            "platform": identity["platform"],
            "platform_channel_id": identity["platform_channel_id"],
        })
        await database.llm_trace_steps.delete_many({
            "trace_id": old_trace_id,
        })

    assert response is not None
    assert mapped_inputs
    assert residual_marker in str(
        mapped_inputs[0]["past_dialog_cognition_context"]
    )
    trace_steps = trace_documents["trace_steps"]
    assert isinstance(trace_steps, list)
    goal_steps = _steps_for_prefix(trace_steps, "goal_cognition.")
    assert goal_steps
    assert residual_marker in _serialized(goal_contexts)
    forbidden_steps = [
        step
        for step in trace_steps
        if not str(step.get("stage_name", "")).startswith("goal_cognition.")
    ]
    assert residual_marker not in _serialized(forbidden_steps)
    logger.info(
        f"P0_PAST_DIALOG_LIVE trace_id={trace_id} "
        f"artifact={artifact_path} response={response.messages}"
    )


async def test_live_group_self_cognition_uses_one_advisory_projection(
    live_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group self-cognition shares one style projection with goal and action."""

    del live_env
    suffix = uuid4().hex[:12]
    platform = f"pytest-live-p0-group-{suffix}"
    platform_channel_id = f"group-{suffix}"
    guideline = (
        "Join only when the observed topic has a clear opening and keep the "
        "contribution light."
    )
    character_profile = await _refresh_character_profile()
    now = datetime.now(timezone.utc).isoformat()
    case = {
        "case_name": models.CASE_GROUP_CHAT_REVIEW,
        "case_id": f"group_activity_window:p0:{suffix}",
        "idle_timestamp_utc": now,
        "last_evidence_timestamp_utc": now,
        "trigger_kind": models.TRIGGER_GROUP_CHAT_REVIEW,
        "semantic_due_state": None,
        "actionability": "active_group_review_same_channel_no_fallback",
        "target_scope": {
            "platform": platform,
            "platform_channel_id": platform_channel_id,
            "channel_type": "group",
            "user_id": None,
        },
        "source_refs": [{
            "source_kind": "reflection_activity_window",
            "source_id": f"scope-p0-{suffix}",
            "due_at": None,
            "summary": "Two people are discussing rainy-day photography.",
        }],
        "visible_context": [{
            "role": "user",
            "text": "Rain makes the street reflections easier to photograph.",
            "timestamp": now,
        }],
        "character_profile": character_profile,
        "platform_bot_id": _BOT_ID,
    }
    original_loader = connector.build_group_engagement_action_context
    loader_results: list[dict[str, object]] = []

    async def capture_loader(**kwargs) -> dict[str, object]:
        result = await original_loader(**kwargs)
        loader_results.append(result)
        return result

    monkeypatch.setattr(
        connector,
        "build_group_engagement_action_context",
        capture_loader,
    )
    original_group_goal = cognition_facade.run_goal_cognition
    captured_group_goal_contexts: list[dict[str, object]] = []

    async def capture_group_goal(
        definition: object,
        goal_ref: object,
        semantic_context: dict[str, object],
        evidence: object,
        services: object,
    ) -> dict[str, object]:
        captured_group_goal_contexts.append(dict(semantic_context))
        return await original_group_goal(
            definition,  # type: ignore[arg-type]
            goal_ref,  # type: ignore[arg-type]
            semantic_context,
            evidence,  # type: ignore[arg-type]
            services,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(
        cognition_facade,
        "run_goal_cognition",
        capture_group_goal,
    )
    original_action_planning = cognition_facade.plan_actions
    captured_action_inputs: list[dict[str, object]] = []

    async def capture_action_planning(
        **kwargs: object,
    ) -> dict[str, object]:
        captured_action_inputs.append({
            "episode": kwargs["episode"],
            "evidence": kwargs["evidence"],
            "group_engagement_action_context": kwargs[
                "group_engagement_action_context"
            ],
        })
        return await original_action_planning(  # type: ignore[call-overload]
            **kwargs,
        )

    monkeypatch.setattr(
        cognition_facade,
        "plan_actions",
        capture_action_planning,
    )

    source_packet = projection.build_source_packet(case)
    rendered_packet = projection.render_source_packet_text(source_packet)
    database = await get_db()
    try:
        await upsert_group_channel_style_image(
            platform=platform,
            platform_channel_id=platform_channel_id,
            overlay={
                "speech_guidelines": [],
                "social_guidelines": [],
                "pacing_guidelines": [],
                "engagement_guidelines": [guideline],
                "confidence": "high",
            },
            source_reflection_run_ids=[f"p0-live-{suffix}"],
            storage_timestamp_utc=now,
        )
        style_trace_id = llm_tracing.build_trace_id()
        await llm_tracing.ensure_llm_trace_run(
            trace_id=style_trace_id,
            platform=platform,
            platform_channel_id=platform_channel_id,
            channel_type="group",
            platform_message_id=f"self_cognition:{case['case_id']}",
            global_user_id="",
            started_at=now,
        )
        style_state = runner._build_cognition_state(case, rendered_packet)
        style_state["llm_trace_id"] = style_trace_id
        style_output = await _run_terminal_self_cognition_trace(
            style_state,
            trace_id=style_trace_id,
        )
        style_trace = await _trace_documents(style_trace_id)
        style_goal_context_count = len(captured_group_goal_contexts)
        style_action_input_count = len(captured_action_inputs)
    finally:
        await database.interaction_style_images.delete_many({
            "platform": platform,
            "platform_channel_id": platform_channel_id,
        })
    control_trace_id = llm_tracing.build_trace_id()
    await llm_tracing.ensure_llm_trace_run(
        trace_id=control_trace_id,
        platform=platform,
        platform_channel_id=platform_channel_id,
        channel_type="group",
        platform_message_id=f"self_cognition:control:{case['case_id']}",
        global_user_id="",
        started_at=now,
    )
    control_state = runner._build_cognition_state(case, rendered_packet)
    control_state["llm_trace_id"] = control_trace_id
    control_output = await _run_terminal_self_cognition_trace(
        control_state,
        trace_id=control_trace_id,
    )
    control_trace = await _trace_documents(control_trace_id)
    style_goal_contexts = captured_group_goal_contexts[
        :style_goal_context_count
    ]
    control_goal_contexts = captured_group_goal_contexts[
        style_goal_context_count:
    ]
    style_action_inputs = captured_action_inputs[
        :style_action_input_count
    ]
    control_action_inputs = captured_action_inputs[
        style_action_input_count:
    ]

    artifact_path = write_llm_trace(
        "cognition_core_v2_p0_context_reconnection",
        "group_engagement_style_and_control",
        {
            "input": {
                "case": case,
                "guideline": guideline,
            },
            "style_case": {
                "trace_id": style_trace_id,
                "loader_result": loader_results[0],
                "cognition_output": style_output,
                "captured_goal_contexts": style_goal_contexts,
                "captured_action_inputs": style_action_inputs,
                **style_trace,
            },
            "empty_control": {
                "trace_id": control_trace_id,
                "loader_result": loader_results[1],
                "cognition_output": control_output,
                "captured_goal_contexts": control_goal_contexts,
                "captured_action_inputs": control_action_inputs,
                **control_trace,
            },
        },
    )

    assert len(loader_results) == 2
    assert loader_results[0] == {
        "engagement_guidelines": [guideline],
        "confidence": "high",
    }
    assert loader_results[1] == {
        "engagement_guidelines": [],
        "confidence": "",
    }
    style_steps = style_trace["trace_steps"]
    control_steps = control_trace["trace_steps"]
    assert isinstance(style_steps, list)
    assert isinstance(control_steps, list)
    style_goal_steps = _steps_for_prefix(style_steps, "goal_cognition.")
    style_action_steps = _steps_for_prefix(style_steps, "action_planning")
    assert style_goal_steps
    assert style_action_steps
    assert guideline in _serialized(style_goal_contexts)
    assert guideline in _serialized(style_action_inputs)
    observed_scene = (
        "Rain makes the street reflections easier to photograph."
    )
    assert observed_scene in _serialized(style_goal_contexts)
    assert observed_scene in _serialized(style_action_inputs)
    assert observed_scene in _serialized(control_goal_contexts)
    assert observed_scene in _serialized(control_action_inputs)
    assert guideline not in _serialized(control_goal_contexts)
    assert guideline not in _serialized(control_action_inputs)
    logger.info(
        f"P0_GROUP_ENGAGEMENT_LIVE style_trace={style_trace_id} "
        f"control_trace={control_trace_id} artifact={artifact_path}"
    )
