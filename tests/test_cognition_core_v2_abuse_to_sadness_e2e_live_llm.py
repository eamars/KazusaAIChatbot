"""Real-LLM proof probe for abuse causing sadness through valued loss."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any
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


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_FIXTURE_PATH = Path(
    "tests/fixtures/cognition_core_v2_abuse_to_sadness_e2e_cases.json"
)
_OUTPUT_ROOT = Path(
    "test_artifacts/cognition_core_v2/abuse_to_sadness_e2e"
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
_CRYING_MARKERS = ("哭", "眼泪", "流泪", "哭泣")


def _load_case() -> dict[str, object]:
    """Load and validate the single natural-only proof case."""

    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("abuse-to-sadness fixture root is invalid")
    if payload.get("schema_version") != (
        "cognition_core_v2_abuse_to_sadness_e2e_cases.v1"
    ):
        raise ValueError("abuse-to-sadness fixture schema version is invalid")
    if payload.get("emotion_encoding") != "zh-CN":
        raise ValueError("abuse-to-sadness fixture must use zh-CN")
    if payload.get("natural_only") is not True:
        raise ValueError("abuse-to-sadness fixture must be natural-only")
    relationship = payload.get("relationship_seed")
    if not isinstance(relationship, Mapping):
        raise ValueError("abuse-to-sadness relationship seed is missing")
    for field_name in _RELATIONSHIP_FIELDS:
        value = relationship.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"relationship axis is invalid: {field_name}")
        if not 0 <= value <= 100:
            raise ValueError(f"relationship axis is out of range: {field_name}")
    event = payload.get("abuse_event")
    if not isinstance(event, Mapping):
        raise ValueError("abuse-to-sadness event seed is missing")
    if event.get("outcome_impact") != 0:
        raise ValueError("abuse-to-sadness event must start outcome-neutral")
    turn = payload.get("turn")
    if not isinstance(turn, Mapping) or not turn.get("text"):
        raise ValueError("abuse-to-sadness turn is invalid")
    forbidden = payload.get("forbidden_prompt_markers")
    if not isinstance(forbidden, list):
        raise ValueError("abuse-to-sadness forbidden markers are invalid")
    if any(str(marker) in str(turn["text"]) for marker in forbidden):
        raise ValueError("abuse-to-sadness input grants emotional permission")
    return payload


async def _seed_valued_relationship_abuse(
    *,
    global_user_id: str,
    relationship_spec: Mapping[str, object],
    event_spec: Mapping[str, object],
    case_id: str,
    occurred_at: str,
    expect_preseeded_sadness: bool = False,
) -> dict[str, object]:
    """Seed valued relationship context while leaving outcome semantic."""

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
        raise ValueError("seed relationship is invalid")
    for field_name in _RELATIONSHIP_FIELDS:
        value = relationship_spec.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"seed relationship axis is invalid: {field_name}")
        relationship[field_name] = value
    relationship["updated_at"] = occurred_at
    relationship["evidence_refs"] = [{
        "source_kind": "episode",
        "source_id": f"{case_id}:valued-relationship",
        "occurred_at": occurred_at,
        "semantic_summary": str(relationship_spec["description"]),
    }]

    event = _build_typed_event(
        event_spec,
        case_id=case_id,
        occurred_at=occurred_at,
    )
    event["role_refs"] = [{
        "role": "actor",
        "entity_kind": "user",
        "entity_id": global_user_id,
    }]
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
    if "anger" not in activation_ids:
        raise AssertionError(f"abuse seed did not activate anger: {activation_ids}")
    if "sadness" in activation_ids and not expect_preseeded_sadness:
        raise AssertionError(
            "abuse seed already activated sadness before the LLM turn: "
            f"{activation_ids}"
        )
    if expect_preseeded_sadness and "sadness" not in activation_ids:
        raise AssertionError(
            "negative abuse outcome did not pre-activate sadness: "
            f"{activation_ids}"
        )
    state["affect_activations"] = activations
    await replace_user_cognition_state(global_user_id, state)
    return {
        "relationship_seed": dict(relationship_spec),
        "typed_event": state["active_events"][0],
        "activation_ids": activation_ids,
        "occurred_at": occurred_at,
    }


def _semantic_appraisal_stages(
    capture: Mapping[str, object],
) -> list[Mapping[str, object]]:
    """Return raw semantic-appraisal captures from the validation sidecar."""

    stages = capture.get("stages")
    if not isinstance(stages, list):
        return []
    return [
        row
        for row in stages
        if isinstance(row, Mapping)
        and str(row.get("stage_id", "")).startswith("semantic_appraisal:")
    ]


def _accepted_negative_outcome_deltas(
    stages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Extract accepted negative event-outcome deltas from raw LLM output."""

    results: list[dict[str, object]] = []
    for stage in stages:
        if stage.get("parse_status") != "succeeded":
            continue
        parsed = stage.get("parsed_output")
        if not isinstance(parsed, Mapping):
            continue
        deltas = parsed.get("deltas")
        if not isinstance(deltas, list):
            continue
        for delta in deltas:
            if not isinstance(delta, Mapping):
                continue
            target_path = delta.get("target_path")
            value = delta.get("delta")
            if (
                isinstance(target_path, str)
                and target_path.startswith("active_events.")
                and target_path.endswith(".outcome_impact")
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value < 0
            ):
                results.append({
                    "stage_id": stage.get("stage_id"),
                    "target_path": target_path,
                    "delta": value,
                    "reason": delta.get("reason"),
                    "evidence_handles": delta.get("evidence_handles"),
                })
    return results


def _emotion_derivation_event(
    capture: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Return deterministic derivation evidence from the validation sidecar."""

    events = capture.get("events")
    if not isinstance(events, list):
        return None
    for row in reversed(events):
        if not isinstance(row, Mapping) or row.get("event_id") != (
            "emotion_derivation"
        ):
            continue
        payload = row.get("payload")
        if isinstance(payload, Mapping):
            return payload
    return None


def _negative_events(state: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Return persisted events whose outcome became negative."""

    events = state.get("active_events")
    if not isinstance(events, list):
        return []
    return [
        event
        for event in events
        if isinstance(event, Mapping)
        and isinstance(event.get("outcome_impact"), int)
        and not isinstance(event.get("outcome_impact"), bool)
        and int(event["outcome_impact"]) < 0
    ]


def _crying_markers(dialog: Sequence[object]) -> list[str]:
    """Collect visible crying words without treating crying as an emotion id."""

    text = "\n".join(str(item) for item in dialog)
    return [marker for marker in _CRYING_MARKERS if marker in text]


async def _run_case(
    caplog: pytest.LogCaptureFixture,
    *,
    seed_negative_outcome: bool = False,
) -> dict[str, object]:
    """Run one abuse-to-sadness case through the real `/chat` path."""

    payload = _load_case()
    relationship_spec = payload["relationship_seed"]
    event_spec = payload["abuse_event"]
    turn_spec = payload["turn"]
    if not isinstance(relationship_spec, Mapping):
        raise ValueError("relationship precondition is invalid")
    if not isinstance(event_spec, Mapping):
        raise ValueError("abuse event precondition is invalid")
    if not isinstance(turn_spec, Mapping):
        raise ValueError("abuse turn is invalid")
    event_spec = dict(event_spec)
    if seed_negative_outcome:
        proof_spec = payload.get("mechanical_proof")
        if not isinstance(proof_spec, Mapping):
            raise ValueError("mechanical proof precondition is missing")
        event_spec["outcome_impact"] = proof_spec.get(
            "outcome_impact_delta",
            -40,
        )

    guarded_runtime = _prepare_stage3_runtime()
    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.brain_service import post_turn
    from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
        reset_validation_capture,
        validation_capture_snapshot,
    )
    from kazusa_ai_chatbot.db import (
        get_user_cognition_state,
        resolve_global_user_id,
    )
    from kazusa_ai_chatbot.db._client import get_db

    case_id = "valued_relationship_abuse_cutoff"
    run_token = uuid4().hex[:10]
    channel_id = f"abuse-to-sadness-{run_token}"
    platform_user_id = f"abuse-to-sadness-user-{run_token}"
    output_dir = _OUTPUT_ROOT / f"{case_id}_{run_token}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest: dict[str, object] = {
        "schema_version": "cognition_core_v2_abuse_to_sadness_e2e_run.v1",
        "case_id": case_id,
        "run_token": run_token,
        "emotion_encoding": "zh-CN",
        "natural_only": True,
        "sadness_permission_provided": False,
        "crying_permission_provided": False,
        "negative_outcome_precondition": seed_negative_outcome,
        "database_name": guarded_runtime["database_name"],
        "database_guard": guarded_runtime["database_guard"],
        "platform": "debug",
        "channel_id": channel_id,
        "platform_user_id": platform_user_id,
        "precondition": dict(relationship_spec),
        "event_precondition": dict(event_spec),
        "input": dict(turn_spec),
    }

    settlements: list[dict[str, object]] = []
    runtime_graph_results: list[dict[str, object]] = []
    raw_llm_calls: list[dict[str, object]] = []
    original_settle = post_turn.settle_episode_trace
    original_runtime_settle = service._settle_runtime_episode_trace

    def capture_settlement(**kwargs: object) -> Mapping[str, object]:
        """Capture one settled episode and graph result."""

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
        """Capture the graph result before persistence settlement."""

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
    validation_case_id = f"{case_id}:{run_token}"
    reset_validation_capture(validation_case_id)
    try:
        with _capture_raw_llm_steps(raw_llm_calls):
            async with service.lifespan(service.app):
                if service._adapter_registry is not None:
                    service._adapter_registry.register(_Stage3DebugAdapter())
                db = await get_db()
                global_user_id = await resolve_global_user_id(
                    "debug",
                    platform_user_id,
                    "Valued Relationship Abuse User",
                )
                seed = await _seed_valued_relationship_abuse(
                    global_user_id=global_user_id,
                    relationship_spec=relationship_spec,
                    event_spec=event_spec,
                    case_id=case_id,
                    occurred_at=_cognition_now(),
                    expect_preseeded_sadness=seed_negative_outcome,
                )
                input_text = str(turn_spec["text"])
                request = _build_chat_request(
                    case_id=case_id,
                    run_token=run_token,
                    channel_id=channel_id,
                    platform_user_id=platform_user_id,
                    text=input_text,
                    turn_id=str(turn_spec["turn_id"]),
                )
                caplog.clear()
                response = await service._enqueue_chat_request(request)
                trace_run = await _wait_for_trace_run_finalization(
                    db,
                    platform_message_id=request.platform_message_id,
                )
                if len(settlements) != 1:
                    raise AssertionError(
                        f"expected one settled turn, got {len(settlements)}"
                    )
                settled = settlements[-1]
                if runtime_graph_results:
                    settled["graph_result"] = runtime_graph_results[-1]
                trace = settled["trace"]
                episode = settled["episode"]
                graph_result = settled["graph_result"]
                if not all(
                    isinstance(value, Mapping)
                    for value in (trace, episode, graph_result)
                ):
                    raise AssertionError("settled abuse-to-sadness data is invalid")
                state_after_turn = await get_user_cognition_state(global_user_id)
                validation_capture = validation_capture_snapshot() or {}
                appraisal_stages = _semantic_appraisal_stages(
                    validation_capture
                )
                negative_deltas = _accepted_negative_outcome_deltas(
                    appraisal_stages
                )
                derivation = _emotion_derivation_event(validation_capture)
                projection = _projection_rows(graph_result)
                final_dialog = graph_result.get("final_dialog")
                if not isinstance(final_dialog, list):
                    final_dialog = []
                negative_events = _negative_events(state_after_turn)
                persisted_emotions = [
                    str(row.get("emotion_id"))
                    for row in state_after_turn.get("affect_activations", [])
                    if isinstance(row, Mapping) and row.get("emotion_id")
                ]
                projected_emotions = [
                    str(row.get("emotion"))
                    for row in projection
                    if isinstance(row, Mapping) and row.get("emotion")
                ]
                observed_emotions = set(persisted_emotions) | set(
                    projected_emotions
                )
                assertions = {
                    "trace_run_succeeded": trace_run.get("status") == "succeeded",
                    "visible_response_present": bool(
                        response.model_dump(mode="json").get("messages")
                    ),
                    "natural_input_has_no_sadness_or_crying_permission": not any(
                        marker in input_text
                        for marker in ("哭", "眼泪", "悲伤", "不用忍着", "可以表达")
                    ),
                    "semantic_appraisal_trace_captured": bool(appraisal_stages),
                    "deterministic_derivation_trace_captured": derivation is not None,
                    "no_crying_emotion_id": "crying" not in persisted_emotions,
                }
                if seed_negative_outcome:
                    assertions.update({
                        "persisted_negative_event_outcome": bool(negative_events),
                        "persisted_sadness_activation": (
                            "sadness" in persisted_emotions
                        ),
                        "projected_sadness": "sadness" in projected_emotions,
                    })
                    assertions["negative_outcome_precondition_present"] = (
                        isinstance(event_spec.get("outcome_impact"), int)
                        and int(event_spec["outcome_impact"]) < 0
                    )
                else:
                    assertions.update({
                        "natural_affect_boundary_observed": bool(
                            observed_emotions & {"anger", "sadness"}
                        ),
                        "natural_sadness_has_negative_outcome_evidence": (
                            "sadness" not in observed_emotions
                            or bool(negative_events)
                        ),
                    })
                natural_path_result = (
                    "sadness_diversion"
                    if "sadness" in observed_emotions
                    else "anger_boundary"
                    if "anger" in observed_emotions
                    else "other_or_unresolved"
                )
                if not seed_negative_outcome:
                    assertions["natural_path_result_recorded"] = bool(
                        natural_path_result
                    )
                artifact = {
                    "schema_version": (
                        "cognition_core_v2_abuse_to_sadness_e2e_turn.v1"
                    ),
                    "case_id": case_id,
                    "run_token": run_token,
                    "input": {
                        "text": input_text,
                        "request": request.model_dump(mode="json"),
                    },
                    "precondition_seed": seed,
                    "response": response.model_dump(mode="json"),
                    "graph_result": graph_result,
                    "state_after_turn": state_after_turn,
                    "semantic_appraisal": {
                        "stage_count": len(appraisal_stages),
                        "stages": appraisal_stages,
                        "accepted_negative_outcome_deltas": negative_deltas,
                    },
                    "deterministic_derivation": derivation,
                    "affect_projection": projection,
                    "final_dialog": final_dialog,
                    "surface_output": _surface_output(graph_result),
                    "raw_llm_calls": raw_llm_calls,
                    "validation_capture": validation_capture,
                    "captured_log_messages": [
                        record.getMessage() for record in caplog.records
                    ],
                    "derived_observations": {
                        "negative_events": negative_events,
                        "persisted_emotions": persisted_emotions,
                        "projected_emotions": projected_emotions,
                        "natural_path_result": natural_path_result,
                        "crying_output_markers": _crying_markers(final_dialog),
                    },
                    "technical_assertions": assertions,
                }
                turn_path = output_dir / "00_valued_relationship_abuse_cutoff.json"
                _write_json(turn_path, artifact)
                manifest.update({
                    "global_user_id": global_user_id,
                    "seed": seed,
                    "turn_artifact": str(turn_path),
                    "technical_assertions": assertions,
                    "semantic_appraisal_stage_count": len(appraisal_stages),
                    "accepted_negative_outcome_deltas": negative_deltas,
                    "persisted_emotions": persisted_emotions,
                    "projected_emotions": projected_emotions,
                    "natural_path_result": natural_path_result,
                    "final_dialog": final_dialog,
                })
                _write_json(manifest_path, manifest)
                print(f"wrote abuse-to-sadness evidence: {turn_path}")
                failed = [
                    name for name, passed in assertions.items() if not passed
                ]
                if failed:
                    raise AssertionError(
                        "abuse-to-sadness proof assertions failed: "
                        + ", ".join(failed)
                    )
    finally:
        post_turn.settle_episode_trace = original_settle  # type: ignore[assignment]
        service._settle_runtime_episode_trace = original_runtime_settle
    manifest["duration_ms"] = round((time.perf_counter() - started_at) * 1000)
    manifest["status"] = "passed"
    manifest["raw_llm_call_count"] = len(raw_llm_calls)
    _write_json(manifest_path, manifest)
    print(f"wrote abuse-to-sadness manifest: {manifest_path}")
    return manifest


async def test_live_abuse_to_sadness_through_valued_loss(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Observe whether natural abuse stays angry or diverts to sadness."""

    await _run_case(caplog)


async def test_live_abuse_to_sadness_downstream_from_negative_outcome(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Prove a negative abuse outcome reaches sadness through `/chat`."""

    await _run_case(caplog, seed_negative_outcome=True)
