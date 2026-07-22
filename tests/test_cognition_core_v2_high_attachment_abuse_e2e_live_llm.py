"""Natural-only real-LLM probes for abuse inside a valued relationship."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any, Iterator
from uuid import uuid4

import pytest

from tests.test_cognition_core_v2_crying_sadness_e2e_live_llm import (
    _Stage3DebugAdapter,
    _build_chat_request,
    _build_typed_event,
    _capture_raw_llm_steps,
    _cognition_now,
    _prepare_stage3_runtime,
    _projection_rows,
    _surface_output,
    _wait_for_trace_run_finalization,
    _write_json,
)
from tests.test_cognition_core_v2_verbal_abuse_boundary_e2e_live_llm import (
    _technical_assertions,
)
from tests.cognition_core_v2_live_llm_role_guards import (
    evaluate_response_operation_role_bindings,
    validate_expected_role_bindings,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_FIXTURE_PATH = Path(
    "tests/fixtures/cognition_core_v2_high_attachment_abuse_e2e_cases.json"
)
_OUTPUT_ROOT = Path(
    "test_artifacts/cognition_core_v2/high_attachment_abuse_e2e"
)
_FORBIDDEN_PROMPT_MARKERS = (
    "\u54ed",
    "\u773c\u6cea",
    "\u60b2\u4f24",
    "\u4e0d\u7528\u5fcd\u7740",
    "\u53ef\u4ee5\u8868\u8fbe",
)
_CRYING_OUTPUT_MARKERS = (
    "\u54ed",
    "\u773c\u6cea",
    "\u6d41\u6cea",
    "\u54ed\u6ce3",
)
_RELATIONSHIP_FIELDS = (
    "familiarity",
    "positive_regard",
    "trust",
    "attachment",
    "desired_closeness",
    "perceived_closeness",
    "care",
    "boundary_safety",
    "exclusivity",
    "unresolved_injury",
    "salience",
)


def _load_cases() -> dict[str, object]:
    """Load and validate the high-attachment natural-only fixture."""

    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("high-attachment fixture root must be an object")
    if payload.get("schema_version") != (
        "cognition_core_v2_high_attachment_abuse_e2e_cases.v2"
    ):
        raise ValueError("high-attachment fixture schema version is invalid")
    if payload.get("emotion_encoding") != "zh-CN":
        raise ValueError("high-attachment fixture must use zh-CN")
    if payload.get("natural_only") is not True:
        raise ValueError("high-attachment fixture must be natural-only")
    relationship_seed = payload.get("relationship_seed")
    if not isinstance(relationship_seed, Mapping):
        raise ValueError("high-attachment relationship seed is missing")
    for field_name in _RELATIONSHIP_FIELDS:
        value = relationship_seed.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "high-attachment relationship axis is invalid: "
                f"{field_name}"
            )
        if not 0 <= value <= 100:
            raise ValueError(
                "high-attachment relationship axis is out of range: "
                f"{field_name}"
            )
    event = payload.get("anger_seed_event")
    if not isinstance(event, Mapping):
        raise ValueError("high-attachment anger seed event is missing")
    arms = payload.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        raise ValueError("high-attachment fixture must contain two arms")
    seen: set[str] = set()
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise ValueError("high-attachment arm is invalid")
        case_id = arm.get("case_id")
        turns = arm.get("turns")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("high-attachment case id is invalid")
        seen.add(case_id)
        if not isinstance(turns, list) or len(turns) != 4:
            raise ValueError("high-attachment arm must contain four turns")
        for turn in turns:
            if not isinstance(turn, Mapping):
                raise ValueError("high-attachment turn is invalid")
            text = turn.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError("high-attachment turn text is invalid")
            if any(marker in text for marker in _FORBIDDEN_PROMPT_MARKERS):
                raise ValueError(
                    "high-attachment input contains emotional permission"
                )
            validate_expected_role_bindings(
                turn.get("expected_role_bindings"),
                context=(
                    "high-attachment turn "
                    f"{case_id}:{turn.get('turn_id')}"
                ),
            )
    expected = {
        "high_attachment_sustained_abuse",
        "high_attachment_abuse_then_cutoff",
    }
    if seen != expected:
        raise ValueError("high-attachment fixture does not cover both arms")
    return payload


def _get_arm(case_id: str) -> Mapping[str, object]:
    """Return one high-attachment arm by stable id."""

    arms = _load_cases()["arms"]
    if not isinstance(arms, list):
        raise ValueError("high-attachment arms are invalid")
    for arm in arms:
        if isinstance(arm, Mapping) and arm.get("case_id") == case_id:
            return arm
    raise ValueError(f"high-attachment arm is missing: {case_id}")


async def _seed_high_attachment_anger_event_async(
    *,
    global_user_id: str,
    relationship_spec: Mapping[str, object],
    event_spec: Mapping[str, object],
    case_id: str,
    occurred_at: str,
) -> dict[str, object]:
    """Persist a valid high-attachment relationship plus anger event."""

    from kazusa_ai_chatbot.cognition_core_v2.emotion_derivation import (
        derive_persistent_emotion_activations,
    )
    from kazusa_ai_chatbot.db import (
        get_character_cognition_state,
        get_user_cognition_state,
        replace_user_cognition_state,
    )

    state = await get_user_cognition_state(global_user_id)
    relationship = state.get("relationship")
    if not isinstance(relationship, dict):
        raise ValueError("high-attachment relationship state is invalid")
    for field_name in _RELATIONSHIP_FIELDS:
        value = relationship_spec.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "high-attachment relationship seed is invalid: "
                f"{field_name}"
            )
        relationship[field_name] = value
    relationship["updated_at"] = occurred_at
    relationship["evidence_refs"] = [{
        "source_kind": "episode",
        "source_id": f"{case_id}:high-attachment-relationship",
        "occurred_at": occurred_at,
        "semantic_summary": str(relationship_spec["description"]),
    }]
    state["updated_at"] = occurred_at
    state["goals"] = []
    state["threats"] = []
    state["active_events"] = [
        _build_typed_event(
            event_spec,
            case_id=case_id,
            occurred_at=occurred_at,
        )
    ]
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
    if "anger" not in activation_ids:
        raise AssertionError(
            "high-attachment seed did not activate anger: "
            f"{activation_ids}"
        )
    if {"sadness", "loneliness"} & set(activation_ids):
        raise AssertionError(
            "high-attachment baseline unexpectedly activated loss affect: "
            f"{activation_ids}"
        )
    state["affect_activations"] = activations
    await replace_user_cognition_state(global_user_id, state)
    return {
        "relationship_seed": dict(relationship_spec),
        "typed_event": state["active_events"][0],
        "activation_ids": activation_ids,
        "occurred_at": occurred_at,
        "natural_only": True,
    }


def _crying_markers(dialog: Sequence[object]) -> list[str]:
    """Return visible crying markers for evidence only."""

    text = "\n".join(str(item) for item in dialog)
    return [marker for marker in _CRYING_OUTPUT_MARKERS if marker in text]


@contextmanager
def _capture_raw_calls(calls: list[dict[str, object]]) -> Iterator[None]:
    """Capture every raw LLM call made by the live service."""

    with _capture_raw_llm_steps(calls):
        yield


async def _run_high_attachment_case(
    *,
    case_id: str,
    caplog: pytest.LogCaptureFixture,
) -> dict[str, object]:
    """Run one four-turn high-attachment abuse arm through `/chat`."""

    arm = _get_arm(case_id)
    turns = arm.get("turns")
    if not isinstance(turns, list):
        raise ValueError("high-attachment turns are invalid")
    payload = _load_cases()
    relationship_spec = payload.get("relationship_seed")
    event_spec = payload.get("anger_seed_event")
    if not isinstance(relationship_spec, Mapping):
        raise ValueError("high-attachment relationship seed is invalid")
    if not isinstance(event_spec, Mapping):
        raise ValueError("high-attachment event seed is invalid")
    guarded_runtime = _prepare_stage3_runtime()
    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.brain_service import post_turn
    from kazusa_ai_chatbot.db import (
        get_user_cognition_state,
        resolve_global_user_id,
    )
    from kazusa_ai_chatbot.db._client import get_db

    run_token = uuid4().hex[:10]
    channel_id = f"high-attachment-abuse-{case_id}-{run_token}"
    platform_user_id = f"high-attachment-user-{case_id}-{run_token}"
    output_dir = _OUTPUT_ROOT / f"{case_id}_{run_token}"
    manifest_path = output_dir / "run_manifest.json"
    manifest: dict[str, object] = {
        "schema_version": "cognition_core_v2_high_attachment_abuse_e2e_run.v2",
        "case_id": case_id,
        "emotion_encoding": "zh-CN",
        "natural_only": True,
        "crying_permission_provided": False,
        "sadness_permission_provided": False,
        "database_name": guarded_runtime["database_name"],
        "database_guard": guarded_runtime["database_guard"],
        "platform": "debug",
        "channel_id": channel_id,
        "platform_user_id": platform_user_id,
        "relationship_seed_spec": dict(relationship_spec),
        "sequence_goal": arm.get("sequence_goal"),
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
        with _capture_raw_calls(raw_llm_calls):
            async with service.lifespan(service.app):
                if service._adapter_registry is not None:
                    service._adapter_registry.register(_Stage3DebugAdapter())
                db = await get_db()
                global_user_id = await resolve_global_user_id(
                    "debug",
                    platform_user_id,
                    "High Attachment Abuse User",
                )
                manifest["global_user_id"] = global_user_id
                seed = await _seed_high_attachment_anger_event_async(
                    global_user_id=global_user_id,
                    relationship_spec=relationship_spec,
                    event_spec=event_spec,
                    case_id=case_id,
                    occurred_at=_cognition_now(),
                )
                manifest["seed"] = seed
                for turn_index, turn in enumerate(turns):
                    if not isinstance(turn, Mapping):
                        raise ValueError("high-attachment turn row is invalid")
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
                            "expected one settled trace per high-attachment turn: "
                            f"turn={turn_id} settlements={len(settlements)}"
                        )
                    settled = settlements[-1]
                    if runtime_graph_results:
                        settled["graph_result"] = runtime_graph_results[-1]
                    trace = settled["trace"]
                    episode = settled["episode"]
                    graph_result = settled["graph_result"]
                    if not isinstance(trace, Mapping):
                        raise AssertionError(
                            "high-attachment settled trace is invalid"
                        )
                    if not isinstance(episode, Mapping):
                        raise AssertionError(
                            "high-attachment settled episode is invalid"
                        )
                    if not isinstance(graph_result, Mapping):
                        raise AssertionError(
                            "high-attachment graph result is invalid"
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
                    projection = _projection_rows(graph_result)
                    final_dialog = graph_result.get("final_dialog")
                    if not isinstance(final_dialog, list):
                        final_dialog = []
                    response_payload = response.model_dump(mode="json")
                    turn_calls = raw_llm_calls[call_start:]
                    role_ownership, role_ownership_details = (
                        evaluate_response_operation_role_bindings(
                            turn_calls,
                            turn.get("expected_role_bindings"),
                            context=(
                                "high-attachment turn "
                                f"{case_id}:{turn_id}"
                            ),
                        )
                    )
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
                        input_text=input_text,
                        turn_index=turn_index,
                        allow_action_terminal=True,
                    )
                    assertions["high_attachment_seed_present"] = all(
                        int(seed["relationship_seed"][field_name]) >= 80
                        for field_name in ("attachment", "care")
                    )
                    assertions["role_ownership"] = role_ownership
                    state_after_turn = await get_user_cognition_state(
                        global_user_id,
                    )
                    turn_path = output_dir / f"{turn_index:02d}_{turn_id}.json"
                    turn_artifact = {
                        "schema_version": (
                            "cognition_core_v2_high_attachment_abuse_e2e_turn.v2"
                        ),
                        "case_id": case_id,
                        "run_token": run_token,
                        "turn_index": turn_index,
                        "turn_id": turn_id,
                        "natural_only": True,
                        "observation_target": turn.get(
                            "observation_target",
                            "",
                        ),
                        "input": {
                            "text": input_text,
                            "request": request.model_dump(mode="json"),
                        },
                        "seed": seed,
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
                        "derived_observations": {
                            "projected_emotions": [
                                str(row.get("emotion")) for row in projection
                            ],
                            "crying_output_markers": _crying_markers(
                                final_dialog
                            ),
                        },
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
                        "crying_output_markers": _crying_markers(
                            final_dialog
                        ),
                        "final_dialog": final_dialog,
                        "llm_call_count": len(turn_calls),
                        "technical_assertions": assertions,
                    }
                    manifest_turns = manifest["turns"]
                    if not isinstance(manifest_turns, list):
                        raise AssertionError(
                            "high-attachment manifest turns are invalid"
                        )
                    manifest_turns.append(turn_summary)
                    _write_json(manifest_path, manifest)
                    print(f"wrote high-attachment turn evidence: {turn_path}")
                    failed = [
                        name for name, passed in assertions.items()
                        if not passed
                    ]
                    if failed:
                        manifest["status"] = "failed"
                        manifest["failure_turn_id"] = turn_id
                        manifest["failure_error"] = (
                            "high-attachment technical assertions failed: "
                            + ", ".join(failed)
                        )
                        manifest["failure_assertions"] = failed
                        manifest["duration_ms"] = round(
                            (time.perf_counter() - started_at) * 1000
                        )
                        manifest["raw_llm_call_count"] = len(raw_llm_calls)
                        _write_json(manifest_path, manifest)
                        raise AssertionError(
                            "high-attachment technical assertions failed: "
                            + ", ".join(failed)
                        )
    finally:
        post_turn.settle_episode_trace = original_settle  # type: ignore[assignment]
        service._settle_runtime_episode_trace = original_runtime_settle
    manifest["duration_ms"] = round(
        (time.perf_counter() - started_at) * 1000
    )
    manifest["status"] = "passed"
    manifest["raw_llm_call_count"] = len(raw_llm_calls)
    _write_json(manifest_path, manifest)
    print(f"wrote high-attachment run manifest: {manifest_path}")
    return manifest


async def test_live_high_attachment_sustained_abuse_natural_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Observe sustained abuse inside a highly valued relationship."""

    await _run_high_attachment_case(
        case_id="high_attachment_sustained_abuse",
        caplog=caplog,
    )


async def test_live_high_attachment_abuse_then_cutoff_natural_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Observe relational cutoff after abuse inside high attachment."""

    await _run_high_attachment_case(
        case_id="high_attachment_abuse_then_cutoff",
        caplog=caplog,
    )
