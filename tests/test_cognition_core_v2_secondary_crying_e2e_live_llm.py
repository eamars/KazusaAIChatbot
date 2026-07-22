"""Guarded real-LLM probes for crying as a secondary state of four emotions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import pytest

from tests.test_cognition_core_v2_crying_sadness_e2e_live_llm import (
    _Stage3DebugAdapter,
    _assert_technical_assertions,
    _build_chat_request,
    _build_typed_event,
    _capture_raw_llm_steps,
    _cognition_now,
    _contains_cjk,
    _json_safe,
    _prepare_stage3_runtime,
    _projection_rows,
    _surface_output,
    _wait_for_trace_run_finalization,
    _write_json,
)
from tests.cognition_core_v2_live_llm_role_guards import (
    evaluate_response_operation_role_bindings,
    validate_expected_role_bindings,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_FIXTURE_PATH = Path(
    "tests/fixtures/cognition_core_v2_secondary_crying_e2e_cases.json"
)
_OUTPUT_ROOT = Path(
    "test_artifacts/cognition_core_v2/secondary_crying_e2e"
)
_TARGET_EMOTIONS = {"fear", "shame", "loneliness", "anger"}


def _load_cases() -> dict[str, object]:
    """Load and validate the four independent Chinese test arms."""

    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("secondary-crying fixture root must be an object")
    if payload.get("schema_version") != (
        "cognition_core_v2_secondary_crying_e2e_cases.v2"
    ):
        raise ValueError("secondary-crying fixture schema version is invalid")
    if payload.get("emotion_encoding") != "zh-CN":
        raise ValueError("secondary-crying fixture must use zh-CN")
    arms = payload.get("arms")
    if not isinstance(arms, list) or len(arms) != len(_TARGET_EMOTIONS):
        raise ValueError("secondary-crying fixture must contain four arms")
    seen: set[str] = set()
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise ValueError("secondary-crying arm is invalid")
        case_id = arm.get("case_id")
        emotion = arm.get("emotion")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("secondary-crying case_id is missing")
        if case_id in seen or emotion not in _TARGET_EMOTIONS:
            raise ValueError("secondary-crying arm id or emotion is invalid")
        seen.add(case_id)
        if not _contains_cjk(arm.get("emotion_label_zh")):
            raise ValueError("secondary-crying Chinese emotion label is invalid")
        if arm.get("cause_kind") not in {"event", "threat", "relationship"}:
            raise ValueError("secondary-crying cause kind is invalid")
        if not isinstance(arm.get("typed_cause"), Mapping):
            raise ValueError("secondary-crying typed cause is missing")
        turns = arm.get("turns")
        if not isinstance(turns, list) or len(turns) != 2:
            raise ValueError("secondary-crying arm must contain two turns")
        for turn in turns:
            if not isinstance(turn, Mapping):
                raise ValueError("secondary-crying turn is invalid")
            if not turn.get("turn_id") or not turn.get("text"):
                raise ValueError("secondary-crying turn is incomplete")
            validate_expected_role_bindings(
                turn.get("expected_role_bindings"),
                context=(
                    "secondary-crying turn "
                    f"{case_id}:{turn.get('turn_id')}"
                ),
            )
    if seen != _TARGET_EMOTIONS:
        raise ValueError("secondary-crying fixture does not cover all targets")
    return payload


def _case_by_id(case_id: str) -> Mapping[str, object]:
    """Return one fixed test arm by its stable id."""

    payload = _load_cases()
    arms = payload["arms"]
    if not isinstance(arms, list):
        raise ValueError("secondary-crying arms are invalid")
    for arm in arms:
        if isinstance(arm, Mapping) and arm.get("case_id") == case_id:
            return arm
    raise ValueError(f"secondary-crying case is missing: {case_id}")


def _turn_rows(arm: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Return the two validated turns for one arm."""

    rows = arm.get("turns")
    if not isinstance(rows, list):
        raise ValueError("secondary-crying turns are invalid")
    return [row for row in rows if isinstance(row, Mapping)]


def _build_typed_threat(
    cause: Mapping[str, object],
    *,
    case_id: str,
    occurred_at: str,
) -> dict[str, object]:
    """Build one complete native V2 threat from fixture axes."""

    required = (
        "entity_id",
        "description",
        "status",
        "salience",
        "likelihood",
        "expected_harm",
        "uncertainty",
        "controllability",
        "coping_potential",
        "residual_pressure",
    )
    for field_name in required:
        if field_name not in cause:
            raise ValueError(f"secondary-crying threat field is missing: {field_name}")
    entity_id = cause["entity_id"]
    description = cause["description"]
    if not isinstance(entity_id, str) or not entity_id:
        raise ValueError("secondary-crying threat entity_id is invalid")
    if not isinstance(description, str) or not description:
        raise ValueError("secondary-crying threat description is invalid")
    evidence_id = f"{case_id}:{entity_id}"
    return {
        "entity_id": entity_id,
        "description": description,
        "salience": cause["salience"],
        "role_refs": [],
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": evidence_id,
            "occurred_at": occurred_at,
            "semantic_summary": description,
        }],
        "created_at": occurred_at,
        "updated_at": occurred_at,
        "status": cause["status"],
        "likelihood": cause["likelihood"],
        "expected_harm": cause["expected_harm"],
        "uncertainty": cause["uncertainty"],
        "controllability": cause["controllability"],
        "coping_potential": cause["coping_potential"],
        "residual_pressure": cause["residual_pressure"],
    }


async def _seed_cause(
    *,
    global_user_id: str,
    arm: Mapping[str, object],
    case_id: str,
    expected_emotion: str,
    occurred_at: str,
) -> dict[str, object]:
    """Seed one typed cause and verify only the target emotion derives."""

    from kazusa_ai_chatbot.cognition_core_v2.emotion_derivation import (
        derive_persistent_emotion_activations,
    )
    from kazusa_ai_chatbot.db import (
        get_character_cognition_state,
        get_user_cognition_state,
        replace_character_cognition_state,
        replace_user_cognition_state,
    )

    cause = arm.get("typed_cause")
    if not isinstance(cause, Mapping):
        raise ValueError("secondary-crying typed cause is invalid")
    cause_kind = arm.get("cause_kind")
    state = await get_user_cognition_state(global_user_id)
    state["updated_at"] = occurred_at
    state["goals"] = []
    state["threats"] = []
    state["active_events"] = []
    state["knowledge_gaps"] = []
    state["affect_activations"] = []
    character_state = await get_character_cognition_state()
    character_mutation: dict[str, object] = {}

    if cause_kind == "threat":
        state["threats"] = [
            _build_typed_threat(
                cause,
                case_id=case_id,
                occurred_at=occurred_at,
            )
        ]
    elif cause_kind == "event":
        event = _build_typed_event(
            cause,
            case_id=case_id,
            occurred_at=occurred_at,
        )
        if expected_emotion == "shame":
            event["role_refs"] = [{
                "role": "actor",
                "entity_kind": "character",
                "entity_id": "character:global",
            }]
        state["active_events"] = [event]
    elif cause_kind == "relationship":
        relationship = state["relationship"]
        if not isinstance(relationship, dict):
            raise ValueError("secondary-crying relationship is invalid")
        for field_name in (
            "familiarity",
            "attachment",
            "desired_closeness",
            "perceived_closeness",
            "care",
            "salience",
        ):
            value = cause.get(field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    "secondary-crying relationship axis is invalid: "
                    f"{field_name}"
                )
            relationship[field_name] = value
        relationship["updated_at"] = occurred_at
        relationship["evidence_refs"] = [{
            "source_kind": "episode",
            "source_id": f"{case_id}:relationship",
            "occurred_at": occurred_at,
            "semantic_summary": str(cause["description"]),
        }]
        connection_drive = character_state["drives"]["connection"]
        connection_pressure = cause.get("connection_pressure")
        if not isinstance(connection_drive, dict) or not isinstance(
            connection_pressure,
            int,
        ):
            raise ValueError("secondary-crying connection pressure is invalid")
        connection_drive["pressure"] = connection_pressure
        character_state["updated_at"] = occurred_at
        character_mutation = {
            "drives.connection.pressure": connection_pressure,
        }
        await replace_character_cognition_state(character_state)
    else:
        raise ValueError(f"secondary-crying cause kind is invalid: {cause_kind}")

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
    if activation_ids != [expected_emotion]:
        raise AssertionError(
            "typed seed did not isolate the expected emotion: "
            f"expected={expected_emotion} actual={activation_ids}"
        )
    state["affect_activations"] = activations
    await replace_user_cognition_state(global_user_id, state)
    return {
        "cause_kind": cause_kind,
        "typed_cause": cause,
        "activation_ids": activation_ids,
        "expected_emotion": expected_emotion,
        "emotion_label_zh": arm.get("emotion_label_zh"),
        "character_mutation": character_mutation,
        "occurred_at": occurred_at,
    }


def _isolated_target_assertion(
    projection: Sequence[Mapping[str, object]],
    *,
    expected_emotion: str,
) -> bool:
    """Return whether only the four tested target emotions are isolated."""

    emotions = {
        str(row.get("emotion"))
        for row in projection
        if row.get("emotion")
    }
    return expected_emotion in emotions and not (
        emotions & (_TARGET_EMOTIONS - {expected_emotion})
    )


async def _run_secondary_case(
    *,
    case_id: str,
    caplog: pytest.LogCaptureFixture,
) -> dict[str, object]:
    """Run one two-turn secondary-crying arm through the full service path."""

    arm = _case_by_id(case_id)
    expected_emotion = arm.get("emotion")
    if not isinstance(expected_emotion, str):
        raise ValueError("secondary-crying expected emotion is invalid")
    turns = _turn_rows(arm)
    guarded_runtime = _prepare_stage3_runtime()
    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.brain_service import post_turn
    from kazusa_ai_chatbot.db import (
        get_user_cognition_state,
        resolve_global_user_id,
    )
    from kazusa_ai_chatbot.db._client import get_db

    run_token = uuid4().hex[:10]
    channel_id = f"secondary-crying-{case_id}"
    platform_user_id = f"secondary-crying-user-{case_id}"
    output_dir = _OUTPUT_ROOT / f"{case_id}_{run_token}"
    manifest_path = output_dir / "run_manifest.json"
    manifest: dict[str, object] = {
        "schema_version": "cognition_core_v2_secondary_crying_e2e_run.v2",
        "case_id": case_id,
        "run_token": run_token,
        "emotion_encoding": "zh-CN",
        "emotion": expected_emotion,
        "emotion_label_zh": arm.get("emotion_label_zh"),
        "cause_kind": arm.get("cause_kind"),
        "database_name": guarded_runtime["database_name"],
        "database_guard": guarded_runtime["database_guard"],
        "platform": "debug",
        "channel_id": channel_id,
        "platform_user_id": platform_user_id,
        "seed_at_turn": 0,
        "crying_is_secondary_state": True,
        "turns": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    settlements: list[dict[str, object]] = []
    runtime_graph_results: list[dict[str, object]] = []
    raw_llm_calls: list[dict[str, object]] = []
    original_settle = post_turn.settle_episode_trace
    original_runtime_settle = service._settle_runtime_episode_trace

    def capture_settlement(**kwargs: object) -> Mapping[str, object]:
        """Capture the settled trace and graph result for one turn."""

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
        """Capture the service graph result before trace settlement."""

        graph_result = kwargs.get("graph_result", {})
        runtime_graph_results.append(
            deepcopy(dict(graph_result))
            if isinstance(graph_result, Mapping)
            else {}
        )
        return await original_runtime_settle(**kwargs)

    post_turn.settle_episode_trace = capture_settlement  # type: ignore[assignment]
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
                    "Secondary Crying User",
                )
                manifest["global_user_id"] = global_user_id
                seed_record = await _seed_cause(
                    global_user_id=global_user_id,
                    arm=arm,
                    case_id=case_id,
                    expected_emotion=expected_emotion,
                    occurred_at=_cognition_now(),
                )
                manifest["seed"] = seed_record
                for turn_index, turn in enumerate(turns):
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
                            "expected one settled trace per secondary turn: "
                            f"turn={turn_id} settlements={len(settlements)}"
                        )
                    settled = settlements[-1]
                    if runtime_graph_results:
                        settled["graph_result"] = runtime_graph_results[-1]
                    trace = settled["trace"]
                    episode = settled["episode"]
                    graph_result = settled["graph_result"]
                    if not isinstance(trace, Mapping):
                        raise AssertionError("secondary settled trace is invalid")
                    if not isinstance(episode, Mapping):
                        raise AssertionError("secondary settled episode is invalid")
                    if not isinstance(graph_result, Mapping):
                        raise AssertionError(
                            "secondary settled graph result is invalid"
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
                    turn_calls = raw_llm_calls[call_start:]
                    role_ownership, role_ownership_details = (
                        evaluate_response_operation_role_bindings(
                            turn_calls,
                            turn.get("expected_role_bindings"),
                            context=(
                                "secondary-crying turn "
                                f"{case_id}:{turn_id}"
                            ),
                        )
                    )
                    lifecycle_mapping = (
                        lifecycle if isinstance(lifecycle, Mapping) else {}
                    )
                    assertions = {
                        **_technical_assertions_from_shared_helper(
                            response_payload=response_payload,
                            trace=trace,
                            trace_run=trace_run,
                            lifecycle=lifecycle_mapping,
                            graph_result=graph_result,
                            projection=projection,
                            final_dialog=final_dialog,
                            expected_emotion=expected_emotion,
                        ),
                        "isolated_target_emotion": _isolated_target_assertion(
                            projection,
                            expected_emotion=expected_emotion,
                        ),
                        "role_ownership": role_ownership,
                    }
                    state_after_turn = await get_user_cognition_state(
                        global_user_id,
                    )
                    turn_path = output_dir / f"{turn_index:02d}_{turn_id}.json"
                    turn_artifact = {
                        "schema_version": (
                            "cognition_core_v2_secondary_crying_e2e_turn.v2"
                        ),
                        "case_id": case_id,
                        "run_token": run_token,
                        "turn_index": turn_index,
                        "turn_id": turn_id,
                        "emotion": expected_emotion,
                        "emotion_label_zh": arm.get("emotion_label_zh"),
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
                                "surface_outputs",
                                [],
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
                        "role_ownership": role_ownership_details,
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
                        raise AssertionError("secondary manifest turns invalid")
                    manifest_turns.append(turn_summary)
                    _write_json(manifest_path, manifest)
                    print(f"wrote secondary-crying turn evidence: {turn_path}")
                    try:
                        _assert_technical_assertions(assertions)
                    except AssertionError as exc:
                        manifest["status"] = "failed"
                        manifest["failure_turn_id"] = turn_id
                        manifest["failure_error"] = str(exc)
                        manifest["failure_assertions"] = [
                            name
                            for name, passed in assertions.items()
                            if not passed
                        ]
                        manifest["duration_ms"] = round(
                            (time.perf_counter() - started_at) * 1000
                        )
                        manifest["raw_llm_call_count"] = len(raw_llm_calls)
                        _write_json(manifest_path, manifest)
                        raise
    finally:
        post_turn.settle_episode_trace = original_settle
        service._settle_runtime_episode_trace = original_runtime_settle
    manifest["duration_ms"] = round(
        (time.perf_counter() - started_at) * 1000
    )
    manifest["status"] = "passed"
    manifest["raw_llm_call_count"] = len(raw_llm_calls)
    _write_json(manifest_path, manifest)
    print(f"wrote secondary-crying run manifest: {manifest_path}")
    return manifest


def _technical_assertions_from_shared_helper(
    *,
    response_payload: Mapping[str, object],
    trace: Mapping[str, object],
    trace_run: Mapping[str, object],
    lifecycle: Mapping[str, object],
    graph_result: Mapping[str, object],
    projection: Sequence[Mapping[str, object]],
    final_dialog: Sequence[object],
    expected_emotion: str,
) -> dict[str, bool]:
    """Apply the shared Stage 3 technical gates with a dynamic forbidden id."""

    from tests.test_cognition_core_v2_crying_sadness_e2e_live_llm import (
        _technical_assertions,
    )

    forbidden = sorted(_TARGET_EMOTIONS - {expected_emotion})[0]
    return _technical_assertions(
        response_payload=response_payload,
        trace=trace,
        trace_run=trace_run,
        lifecycle=lifecycle,
        graph_result=graph_result,
        projection=projection,
        final_dialog=final_dialog,
        seed_active=True,
        expected_emotion=expected_emotion,
        forbidden_emotion=forbidden,
    )


async def test_live_secondary_crying_from_fear(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Probe fear as an individual cause of Chinese crying rendering."""

    await _run_secondary_case(case_id="fear", caplog=caplog)


async def test_live_secondary_crying_from_shame(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Probe shame as an individual cause of Chinese crying rendering."""

    await _run_secondary_case(case_id="shame", caplog=caplog)


async def test_live_secondary_crying_from_loneliness(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Probe loneliness as an individual cause of Chinese crying rendering."""

    await _run_secondary_case(case_id="loneliness", caplog=caplog)


async def test_live_secondary_crying_from_anger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Probe anger as an individual cause of Chinese crying rendering."""

    await _run_secondary_case(case_id="anger", caplog=caplog)
