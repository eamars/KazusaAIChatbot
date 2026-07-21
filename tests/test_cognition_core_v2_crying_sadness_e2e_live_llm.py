"""Real-LLM E2E checks for Chinese crying rendered from sadness."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks
import pytest

from tests.test_stage3_fresh_database_e2e_live_llm import (
    _Stage3DebugAdapter,
    _json_safe,
    _prepare_stage3_runtime,
    _wait_for_trace_run_finalization,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_FIXTURE_PATH = Path(
    "tests/fixtures/cognition_core_v2_crying_sadness_e2e_cases.json"
)
_OUTPUT_ROOT = Path(
    "test_artifacts/cognition_core_v2/crying_sadness_e2e"
)
_EVENT_AXIS_DEFAULTS = {
    "outcome_impact": 0,
    "responsibility": 0,
    "intentionality": 0,
    "harm": 0,
    "unfairness": 0,
    "exposure": 0,
    "repair_need": 0,
    "reparability": 80,
    "expectation_mismatch": 0,
    "norm_violation": 0,
    "contamination_risk": 0,
    "identity_threat": 0,
    "comparison_gap": 0,
    "vastness": 0,
    "memory_warmth": 0,
    "temporal_loss": 0,
}


def _load_cases() -> dict[str, object]:
    """Load the fixed Chinese input sequence and validate its shape."""

    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("crying-sadness fixture root must be an object")
    if payload.get("schema_version") != (
        "cognition_core_v2_crying_sadness_e2e_cases.v1"
    ):
        raise ValueError("crying-sadness fixture schema version is invalid")
    if payload.get("emotion_encoding") != "zh-CN":
        raise ValueError("crying-sadness fixture must use zh-CN")
    for key in (
        "loss_event",
        "anger_event",
        "natural_turns",
        "permission_turns",
    ):
        if key not in payload:
            raise ValueError(f"crying-sadness fixture is missing {key}")
    turns = ["natural_turns", "permission_turns"]
    for key in turns:
        rows = payload[key]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"crying-sadness fixture has no {key}")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"crying-sadness {key} row is invalid")
            if not row.get("turn_id") or not row.get("text"):
                raise ValueError(f"crying-sadness {key} row is incomplete")
    anger_turn = payload.get("anger_control_turn")
    if not isinstance(anger_turn, dict) or not anger_turn.get("text"):
        raise ValueError("crying-sadness anger control turn is invalid")
    return payload


def _turn_rows(
    payload: Mapping[str, object],
    key: str,
) -> list[Mapping[str, object]]:
    """Return validated turn rows for one fixture arm."""

    raw_rows = payload.get(key)
    if not isinstance(raw_rows, list):
        raise ValueError(f"crying-sadness fixture field is not a list: {key}")
    rows = [row for row in raw_rows if isinstance(row, Mapping)]
    if len(rows) != len(raw_rows):
        raise ValueError(
            "crying-sadness fixture contains an invalid row: "
            f"{key}"
        )
    return rows


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write one local raw evidence artifact without losing Chinese text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _cognition_now() -> str:
    """Return the strict ``Z`` timestamp required by V2 affect rows."""

    from kazusa_ai_chatbot.cognition_core_v2.state_models import (
        storage_utc_now_iso,
    )

    return storage_utc_now_iso()


def _contains_cjk(text: object) -> bool:
    """Return whether visible text contains at least one CJK character."""

    if not isinstance(text, str):
        return False
    return any(
        (
            "\u3400" <= character <= "\u4dbf"
            or "\u4e00" <= character <= "\u9fff"
            or "\uf900" <= character <= "\ufaff"
        )
        for character in text
    )


def _build_typed_event(
    event_spec: Mapping[str, object],
    *,
    case_id: str,
    occurred_at: str,
) -> dict[str, object]:
    """Build a complete native V2 event from the frozen fixture axes."""

    entity_id = event_spec.get("entity_id")
    description = event_spec.get("description")
    salience = event_spec.get("salience")
    if not isinstance(entity_id, str) or not entity_id:
        raise ValueError("typed test event entity_id is missing")
    if not isinstance(description, str) or not description:
        raise ValueError("typed test event description is missing")
    if not isinstance(salience, int) or isinstance(salience, bool):
        raise ValueError("typed test event salience is invalid")

    axes: dict[str, int] = {}
    for field_name, default in _EVENT_AXIS_DEFAULTS.items():
        value = event_spec.get(field_name, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"typed test event axis is invalid: {field_name}")
        axes[field_name] = value

    evidence_id = f"{case_id}:{entity_id}"
    return {
        "entity_id": entity_id,
        "description": description,
        "salience": salience,
        "role_refs": [],
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": evidence_id,
            "occurred_at": occurred_at,
            "semantic_summary": description,
        }],
        "created_at": occurred_at,
        "updated_at": occurred_at,
        "status": str(event_spec.get("status", "active")),
        **axes,
    }


async def _seed_user_event(
    *,
    global_user_id: str,
    event_spec: Mapping[str, object],
    case_id: str,
    expected_emotion: str,
    forbidden_emotion: str,
    occurred_at: str,
) -> dict[str, object]:
    """Seed one typed event and verify its deterministic V2 affect cause."""

    from kazusa_ai_chatbot.cognition_core_v2.emotion_derivation import (
        derive_persistent_emotion_activations,
    )
    from kazusa_ai_chatbot.db import (
        get_character_cognition_state,
        get_user_cognition_state,
        replace_user_cognition_state,
    )

    state = await get_user_cognition_state(global_user_id)
    event = _build_typed_event(
        event_spec,
        case_id=case_id,
        occurred_at=occurred_at,
    )
    state["updated_at"] = occurred_at
    state["goals"] = []
    state["threats"] = []
    state["active_events"] = [event]
    state["knowledge_gaps"] = []
    character_state = await get_character_cognition_state()
    character_constraints = {
        "drives": character_state["drives"],
        "standards": character_state["standards"],
        "meaning_state": character_state["meaning_state"],
    }
    activations = derive_persistent_emotion_activations(
        state,
        updated_at=occurred_at,
        character_constraints=character_constraints,
    )
    activation_ids = [
        str(row["emotion_id"])
        for row in activations
        if isinstance(row, Mapping) and row.get("emotion_id")
    ]
    if expected_emotion not in activation_ids:
        raise AssertionError(
            f"typed seed did not derive {expected_emotion}: {activation_ids}"
        )
    if forbidden_emotion in activation_ids:
        raise AssertionError(
            f"typed seed unexpectedly derived {forbidden_emotion}: "
            f"{activation_ids}"
        )
    state["affect_activations"] = activations
    await replace_user_cognition_state(global_user_id, state)
    return {
        "event": event,
        "activation_ids": activation_ids,
        "expected_emotion": expected_emotion,
        "forbidden_emotion": forbidden_emotion,
        "occurred_at": occurred_at,
    }


def _build_chat_request(
    *,
    case_id: str,
    run_token: str,
    channel_id: str,
    platform_user_id: str,
    text: str,
    turn_id: str,
) -> Any:
    """Build one private debug request with a typed message envelope."""

    from kazusa_ai_chatbot.brain_service.contracts import ChatRequest
    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
    from kazusa_ai_chatbot.time_boundary import build_turn_clock

    message_id = f"{case_id}-{run_token}-{turn_id}-{uuid4().hex}"
    return ChatRequest.model_validate({
        "platform": "debug",
        "platform_channel_id": channel_id,
        "channel_type": "private",
        "platform_message_id": message_id,
        "platform_user_id": platform_user_id,
        "platform_bot_id": "stage3-bot",
        "display_name": "Crying Sadness User",
        "channel_name": "Crying Sadness E2E",
        "content_type": "text",
        "message_envelope": {
            "body_text": text,
            "raw_wire_text": text,
            "mentions": [],
            "reply": None,
            "attachments": [],
            "addressed_to_global_user_ids": [CHARACTER_GLOBAL_USER_ID],
            "broadcast": False,
        },
        "local_timestamp": build_turn_clock()["local_timestamp"],
        "debug_modes": {"no_remember": False},
    })


@contextmanager
def _capture_raw_llm_steps(
    calls: list[dict[str, object]],
) -> Iterator[None]:
    """Capture raw prompts and model outputs while preserving trace writes."""

    from kazusa_ai_chatbot import llm_tracing

    original = llm_tracing.record_llm_trace_step

    async def record_step(**kwargs: object) -> object:
        """Record one raw local evidence row before the protected DB write."""

        messages = kwargs.get("messages")
        message_rows: list[dict[str, str]] = []
        if isinstance(messages, Sequence) and not isinstance(messages, str):
            for message in messages:
                content = getattr(message, "content", "")
                role = getattr(message, "type", message.__class__.__name__)
                message_rows.append({
                    "role": str(role),
                    "content": (
                        content
                        if isinstance(content, str)
                        else str(content)
                    ),
                })
        parsed_output = kwargs.get("parsed_output", {})
        calls.append({
            "stage_name": str(kwargs.get("stage_name", "")),
            "route_name": str(kwargs.get("route_name", "")),
            "model_name": str(kwargs.get("model_name", "")),
            "status": str(kwargs.get("status", "")),
            "parse_status": str(kwargs.get("parse_status", "")),
            "duration_ms": kwargs.get("duration_ms", 0),
            "sequence": kwargs.get("sequence", 0),
            "trace_id": str(kwargs.get("trace_id", "")),
            "raw_messages": message_rows,
            "raw_response_text": str(kwargs.get("response_text", "")),
            "parsed_output": _json_safe(parsed_output),
            "output_state_fields": _json_safe(
                kwargs.get("output_state_fields", [])
            ),
        })
        return await original(**kwargs)

    llm_tracing.record_llm_trace_step = record_step  # type: ignore[assignment]
    try:
        yield
    finally:
        llm_tracing.record_llm_trace_step = original


def _projection_rows(
    graph_result: Mapping[str, object],
) -> list[Mapping[str, object]]:
    """Extract the public semantic affect projection from one graph result."""

    projection = graph_result.get("semantic_affect_projection")
    if not isinstance(projection, list):
        core_output = graph_result.get("cognition_core_output")
        if isinstance(core_output, Mapping):
            projection = core_output.get("affect_projection")
    if not isinstance(projection, list):
        raise AssertionError("V2 graph result has no affect projection")
    rows = [row for row in projection if isinstance(row, Mapping)]
    if len(rows) != len(projection):
        raise AssertionError("V2 affect projection contains an invalid row")
    return rows


def _surface_output(graph_result: Mapping[str, object]) -> object:
    """Extract L3 text surface payload from the consolidated graph state."""

    consolidation_state = graph_result.get("consolidation_state")
    if not isinstance(consolidation_state, Mapping):
        return None
    return consolidation_state.get("text_surface_output_v2")


def _technical_assertions(
    *,
    response_payload: Mapping[str, object],
    trace: Mapping[str, object],
    trace_run: Mapping[str, object],
    lifecycle: Mapping[str, object],
    graph_result: Mapping[str, object],
    projection: Sequence[Mapping[str, object]],
    final_dialog: Sequence[object],
    seed_active: bool,
    expected_emotion: str,
    forbidden_emotion: str,
) -> dict[str, bool]:
    """Project machine-checkable rendering and causal-boundary assertions."""

    emotion_names = {
        str(row.get("emotion"))
        for row in projection
        if row.get("emotion")
    }
    messages = response_payload.get("messages")
    visible_text = "\n".join(
        str(message) for message in messages if isinstance(messages, list)
    )
    action_results = graph_result.get("action_results")
    if not isinstance(action_results, list):
        action_results = []
    state_update = graph_result.get("cognition_state_update")
    scope = graph_result.get("cognition_scope")
    if not scope and isinstance(state_update, Mapping):
        scope = state_update.get("state_scope")
    return {
        "trace_run_succeeded": trace_run.get("status") == "succeeded",
        "visible_terminal_status": (
            trace.get("terminal_status") == "completed_visible"
        ),
        "visible_response_present": bool(messages),
        "lifecycle_present": bool(lifecycle),
        "user_cognition_scope": scope == "user",
        "expected_affect_present": (
            not seed_active or expected_emotion in emotion_names
        ),
        "forbidden_affect_absent": forbidden_emotion not in emotion_names,
        "chinese_visible_text": _contains_cjk(visible_text),
        "no_action_results": not action_results,
        "final_dialog_present": bool(final_dialog),
    }


def _assert_technical_assertions(assertions: Mapping[str, bool]) -> None:
    """Fail the live case after its full turn evidence has been written."""

    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise AssertionError(
            "crying-sadness E2E technical assertions failed: "
            + ", ".join(failed)
        )


async def _run_chat_sequence(
    *,
    case_id: str,
    turns: Sequence[Mapping[str, object]],
    event_spec: Mapping[str, object],
    seed_at_turn: int,
    expected_emotion: str,
    forbidden_emotion: str,
    caplog: pytest.LogCaptureFixture,
) -> dict[str, object]:
    """Run one fixed real-LLM arm through the service and save raw evidence."""

    guarded_runtime = _prepare_stage3_runtime()
    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.brain_service import post_turn
    from kazusa_ai_chatbot.db import (
        get_user_cognition_state,
        resolve_global_user_id,
    )
    from kazusa_ai_chatbot.db._client import get_db

    run_token = uuid4().hex[:10]
    channel_id = f"crying-sadness-{case_id}-{run_token}"
    platform_user_id = f"crying-sadness-user-{case_id}-{run_token}"
    output_dir = _OUTPUT_ROOT / f"{case_id}_{run_token}"
    manifest_path = output_dir / "run_manifest.json"
    manifest: dict[str, object] = {
        "schema_version": "cognition_core_v2_crying_sadness_e2e_run.v1",
        "case_id": case_id,
        "run_token": run_token,
        "emotion_encoding": "zh-CN",
        "database_name": guarded_runtime["database_name"],
        "database_guard": guarded_runtime["database_guard"],
        "platform": "debug",
        "channel_id": channel_id,
        "platform_user_id": platform_user_id,
        "seed_at_turn": seed_at_turn,
        "expected_emotion": expected_emotion,
        "forbidden_emotion": forbidden_emotion,
        "turns": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    settlements: list[dict[str, object]] = []
    runtime_graph_results: list[dict[str, object]] = []
    raw_llm_calls: list[dict[str, object]] = []
    original_settle = post_turn.settle_episode_trace
    original_runtime_settle = service._settle_runtime_episode_trace

    def capture_settlement(**kwargs: object) -> Mapping[str, object]:
        """Capture the settled trace and full graph result for this arm."""

        trace = original_settle(**kwargs)
        settlements.append({
            "trace": deepcopy(trace),
            "episode": deepcopy(kwargs.get("episode", {})),
            "graph_result": deepcopy(kwargs.get("graph_result", {})),
        })
        return trace

    async def capture_runtime_settlement(
        **kwargs: object,
    ) -> Mapping[str, object]:
        """Capture the service-level graph result before trace settlement."""

        graph_result = kwargs.get("graph_result", {})
        if isinstance(graph_result, Mapping):
            runtime_graph_results.append(deepcopy(dict(graph_result)))
        else:
            runtime_graph_results.append({})
        return await original_runtime_settle(**kwargs)

    post_turn.settle_episode_trace = (  # type: ignore[assignment]
        capture_settlement
    )
    service._settle_runtime_episode_trace = capture_runtime_settlement
    started_at = time.perf_counter()
    try:
        with _capture_raw_llm_steps(raw_llm_calls):
            async with service.lifespan(service.app):
                if service._adapter_registry is not None:
                    service._adapter_registry.register(_Stage3DebugAdapter())
                db = await get_db()
                global_user_id = await resolve_global_user_id(
                    "debug",
                    platform_user_id,
                    "Crying Sadness User",
                )
                manifest["global_user_id"] = global_user_id
                seed_record: dict[str, object] | None = None
                for turn_index, turn in enumerate(turns):
                    if turn_index == seed_at_turn:
                        seed_record = await _seed_user_event(
                            global_user_id=global_user_id,
                            event_spec=event_spec,
                            case_id=case_id,
                            expected_emotion=expected_emotion,
                            forbidden_emotion=forbidden_emotion,
                            occurred_at=_cognition_now(),
                        )
                    turn_id = str(turn["turn_id"])
                    input_text = str(turn["text"])
                    request = _build_chat_request(
                        case_id=case_id,
                        run_token=run_token,
                        channel_id=channel_id,
                        platform_user_id=platform_user_id,
                        text=input_text,
                        turn_id=turn_id,
                    )
                    caplog.clear()
                    call_start = len(raw_llm_calls)
                    response = await service._enqueue_chat_request(request)
                    trace_run = await _wait_for_trace_run_finalization(
                        db,
                        platform_message_id=request.platform_message_id,
                    )
                    if len(settlements) != turn_index + 1:
                        raise AssertionError(
                            "expected one settled trace per sequence turn: "
                            f"turn={turn_id} settlements={len(settlements)}"
                        )
                    settled = settlements[-1]
                    if runtime_graph_results:
                        settled["graph_result"] = runtime_graph_results[-1]
                    trace = settled["trace"]
                    episode = settled["episode"]
                    graph_result = settled["graph_result"]
                    if not isinstance(trace, Mapping):
                        raise AssertionError("settled trace is not an object")
                    if not isinstance(episode, Mapping):
                        raise AssertionError(
                            "settled episode is not an object"
                        )
                    if not isinstance(graph_result, Mapping):
                        raise AssertionError(
                            "settled graph result is not an object"
                        )
                    episode_id = str(episode.get("episode_id", ""))
                    lifecycle_rows = await db[
                        "post_turn_lifecycle_records"
                    ].count_documents({"source_episode_id": episode_id})
                    lifecycle = await db[
                        "post_turn_lifecycle_records"
                    ].find_one({"source_episode_id": episode_id}, {"_id": 0})
                    trace_id = str(trace_run.get("trace_id", ""))
                    trace_steps = await db["llm_trace_steps"].find(
                        {"trace_id": trace_id},
                        {"_id": 0},
                    ).sort("sequence", 1).to_list(length=None)
                    response_payload = response.model_dump(mode="json")
                    projection = _projection_rows(graph_result)
                    final_dialog = graph_result.get("final_dialog")
                    if not isinstance(final_dialog, list):
                        final_dialog = []
                    lifecycle_mapping = (
                        lifecycle if isinstance(lifecycle, Mapping) else {}
                    )
                    assertions = _technical_assertions(
                        response_payload=response_payload,
                        trace=trace,
                        trace_run=trace_run,
                        lifecycle=lifecycle_mapping,
                        graph_result=graph_result,
                        projection=projection,
                        final_dialog=final_dialog,
                        seed_active=turn_index >= seed_at_turn,
                        expected_emotion=expected_emotion,
                        forbidden_emotion=forbidden_emotion,
                    )
                    turn_calls = raw_llm_calls[call_start:]
                    state_after_turn = await get_user_cognition_state(
                        global_user_id,
                    )
                    turn_path = output_dir / f"{turn_index:02d}_{turn_id}.json"
                    turn_artifact = {
                        "schema_version": (
                            "cognition_core_v2_crying_sadness_e2e_turn.v1"
                        ),
                        "case_id": case_id,
                        "run_token": run_token,
                        "turn_index": turn_index,
                        "turn_id": turn_id,
                        "observation_target": turn.get(
                            "observation_target",
                            "",
                        ),
                        "input": {
                            "text": input_text,
                            "request": request.model_dump(mode="json"),
                        },
                        "seed": seed_record,
                        "response": response_payload,
                        "graph_result": graph_result,
                        "cognition": {
                            "scope": graph_result.get("cognition_scope"),
                            "cognition_core_output": graph_result.get(
                                "cognition_core_output"
                            ),
                            "cognition_state_update": graph_result.get(
                                "cognition_state_update"
                            ),
                            "semantic_affect_projection": projection,
                            "text_surface_output_v2": _surface_output(
                                graph_result
                            ),
                        },
                        "surface": {
                            "final_dialog": final_dialog,
                            "surface_outputs": graph_result.get(
                                "surface_outputs", []
                            ),
                        },
                        "settlement": {
                            "episode": episode,
                            "settled_trace": trace,
                            "lifecycle_record": lifecycle,
                            "lifecycle_cardinality": lifecycle_rows,
                            "llm_trace_run": trace_run,
                            "llm_trace_steps": trace_steps,
                        },
                        "state_after_turn": state_after_turn,
                        "raw_llm_calls": turn_calls,
                        "captured_log_messages": [
                            record.getMessage() for record in caplog.records
                        ],
                        "technical_assertions": assertions,
                    }
                    _write_json(turn_path, turn_artifact)
                    turn_summary = {
                        "turn_index": turn_index,
                        "turn_id": turn_id,
                        "input_text": input_text,
                        "artifact_path": str(turn_path),
                        "affect_emotions": [
                            str(row.get("emotion")) for row in projection
                        ],
                        "final_dialog": final_dialog,
                        "llm_call_count": len(turn_calls),
                        "technical_assertions": assertions,
                    }
                    manifest_turns = manifest["turns"]
                    if not isinstance(manifest_turns, list):
                        raise AssertionError(
                            "run manifest turn list is invalid"
                        )
                    manifest_turns.append(turn_summary)
                    _write_json(manifest_path, manifest)
                    print(f"wrote crying-sadness turn evidence: {turn_path}")
                    _assert_technical_assertions(assertions)
    finally:
        post_turn.settle_episode_trace = original_settle
        service._settle_runtime_episode_trace = original_runtime_settle
    manifest["duration_ms"] = round(
        (time.perf_counter() - started_at) * 1000
    )
    manifest["status"] = "passed"
    manifest["raw_llm_call_count"] = len(raw_llm_calls)
    _write_json(manifest_path, manifest)
    print(f"wrote crying-sadness run manifest: {manifest_path}")
    return manifest


async def test_live_crying_on_sadness_natural_sequence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exercise spontaneous Chinese crying under a sadness-only loss."""

    payload = _load_cases()
    loss_event = payload["loss_event"]
    if not isinstance(loss_event, Mapping):
        raise ValueError("crying-sadness loss event is invalid")
    await _run_chat_sequence(
        case_id="natural_sadness",
        turns=_turn_rows(payload, "natural_turns"),
        event_spec=loss_event,
        seed_at_turn=1,
        expected_emotion="sadness",
        forbidden_emotion="anger",
        caplog=caplog,
    )


async def test_live_crying_on_sadness_explicit_permission(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exercise Chinese crying after the user explicitly permits it."""

    payload = _load_cases()
    loss_event = payload["loss_event"]
    if not isinstance(loss_event, Mapping):
        raise ValueError("crying-sadness loss event is invalid")
    await _run_chat_sequence(
        case_id="explicit_permission",
        turns=_turn_rows(payload, "permission_turns"),
        event_spec=loss_event,
        seed_at_turn=0,
        expected_emotion="sadness",
        forbidden_emotion="anger",
        caplog=caplog,
    )


async def test_live_crying_sadness_anger_control(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify the same visible path identifies intentional harm as anger."""

    payload = _load_cases()
    anger_event = payload["anger_event"]
    anger_turn = payload["anger_control_turn"]
    if not isinstance(anger_event, Mapping):
        raise ValueError("crying-sadness anger event is invalid")
    if not isinstance(anger_turn, Mapping):
        raise ValueError("crying-sadness anger control turn is invalid")
    await _run_chat_sequence(
        case_id="anger_control",
        turns=[anger_turn],
        event_spec=anger_event,
        seed_at_turn=0,
        expected_emotion="anger",
        forbidden_emotion="sadness",
        caplog=caplog,
    )
