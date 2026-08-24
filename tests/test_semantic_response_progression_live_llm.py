"""One-at-a-time real-model gates for semantic response progression."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    _prepare_state_transaction,
    bind_protected_chain_records,
    reset_protected_chain_records,
    run_cognition,
    snapshot_protected_chain_records,
)
from kazusa_ai_chatbot.cognition_core_v3.prompt import (
    build_canonical_appraisal_question,
    build_canonical_goal_question,
    build_canonical_plan_question,
    build_canonical_turn_workspace,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_shared.state_reducers import (
    materialize_causal_root,
)
from kazusa_ai_chatbot.cognition_shared import surface
from kazusa_ai_chatbot.conversation_progress import recorder
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from tests.conversation_progress_v2_helpers import (
    logical_turn,
    record_input,
)
from tests.unit.cognition_core_v3.test_handleless_contract import _input
from tests.unit.nodes.surface_fixtures import (
    build_relational_decision,
    build_surface_state,
)

pytestmark = pytest.mark.live_llm

_ARTIFACT_DIR = Path(
    "test_artifacts/live_llm/semantic_response_progression_20260823"
)
_PRE_CHANGE_ARTIFACT = _ARTIFACT_DIR / (
    "pre_change_multi_emotion_workspace.json"
)
_REQUIRED_EMOTIONS = (
    "sadness",
    "anger",
    "gratitude",
    "embarrassment",
    "nostalgia",
)
_SERVICE_GATE_MESSAGES = {
    "L2": [
        "我最近工作有点乱，你会怎么帮我排第一步？",
        "我已经列出三件事了，先处理哪一件？",
        "第一件完成了，但第二件要等同事回复，我现在做什么？",
        "同事还没回，我先去开会，会议前提醒我检查什么？",
    ],
    "L3": [
        "你是不是觉得我总是在逃避？",
        "不是，我只是今天时间不够。那我现在先做哪一步？",
        "第一步完成了，第二步要等同事回复，接下来怎么办？",
        "同事回复了，我只想确认会议时间：几点开始？",
    ],
    "L4": [
        "我想把每周回顾做成固定习惯，你能陪我建立一个简单流程吗？",
        "我做完第一周回顾了，帮我找出一个最值得保留的环节。",
        "我想继续沿用你的复盘方式，但改成十分钟版本，怎么调整？",
    ],
    "L6": [
        "以后我说要做计划时，请先提醒我确认一个边界。",
        "今天先聊点轻松的：我刚买了杯咖啡，味道很普通。",
    ],
}


def _write_artifact(case_name: str, value: dict[str, Any]) -> Path:
    """Write one immutable, inspectable live-gate artifact."""

    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACT_DIR / f"{case_name}_{time.time_ns()}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _load_immutable_pre_change_evidence() -> dict[str, Any]:
    """Load the durable maximum-state baseline created before code edits."""

    with _PRE_CHANGE_ARTIFACT.open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert value["status"] == "pre_change"
    assert value["schema"] == (
        "semantic_progression_pre_change_multi_emotion_evidence.v1"
    )
    return value


def _load_service_gate_artifact(gate: str) -> dict[str, Any]:
    """Load one parent-coordinated memory-enabled service gate artifact."""

    return _load_service_gate_artifacts(gate, expected_count=1)[0]


def _load_service_gate_artifacts(
    gate: str,
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Load independently identified parent-coordinated gate artifacts."""

    paths = sorted(_ARTIFACT_DIR.glob(f"{gate.lower()}_service_*.json"))
    assert len(paths) >= expected_count, (
        f"expected {expected_count} parent-coordinated {gate} service "
        "artifacts"
    )
    artifacts = []
    for path in paths[-expected_count:]:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        assert value["schema"] == (
            "semantic_response_progression_service_gate.v1"
        )
        assert value["gate"] == gate
        assert value["debug_modes"]["no_remember"] is False
        assert value["memory_enabled"] is True
        assert isinstance(value["identity"]["global_user_id"], str)
        assert value["identity"]["global_user_id"].strip()
        turns = value["turns"]
        assert isinstance(turns, list)
        assert [turn["input"] for turn in turns] == (
            _SERVICE_GATE_MESSAGES[gate]
        )
        assert all(isinstance(turn.get("response"), str) for turn in turns)
        assert all(turn["response"].strip() for turn in turns)
        artifacts.append(value)
    identities = [
        artifact["identity"]["global_user_id"]
        for artifact in artifacts
    ]
    assert len(set(identities)) == expected_count
    return artifacts


def _multi_emotion_input() -> dict[str, Any]:
    """Build a valid event-root state without changing production reducers."""

    payload = deepcopy(_input())
    timestamp = str(payload["mutable_state"]["updated_at"])
    episode = deepcopy(payload["episode"])
    percepts = episode["percepts"]
    percepts[0]["content"]["semantic_text"] = (
        "会议已经改到15:30了，请告诉我现在几点开始？"
    )
    payload["episode"] = episode
    payload["evidence"] = [{
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode:semantic-progression-current-question",
            "occurred_at": timestamp,
            "semantic_summary": (
                "会议已经改到15:30了，请告诉我现在几点开始？"
            ),
        },
        "semantic_text": "会议已经改到15:30了，请告诉我现在几点开始？",
        "authority": "current_event",
    }]
    payload["overused_moves"] = [
        "x" * 120,
        "y" * 120,
        "z" * 120,
        "w" * 120,
    ]

    state = deepcopy(payload["mutable_state"])
    event_specs = [
        (
            "sadness",
            "a concrete loss remains unresolved",
            {"outcome_impact": -80, "salience": 80},
        ),
        (
            "anger",
            "a boundary was crossed in the current event",
            {
                "harm": 80,
                "unfairness": 80,
                "intentionality": 80,
                "salience": 80,
            },
        ),
        (
            "gratitude",
            "a specific act of care was received",
            {
                "outcome_impact": 80,
                "responsibility": 80,
                "salience": 80,
                "role_refs": [{
                    "role": "actor",
                    "entity_kind": "user",
                    "entity_id": "user-1",
                }],
            },
        ),
        (
            "embarrassment",
            "a private mistake became visible",
            {
                "responsibility": 80,
                "exposure": 80,
                "expectation_mismatch": 80,
                "salience": 80,
                "role_refs": [{
                    "role": "actor",
                    "entity_kind": "character",
                    "entity_id": "character:global",
                }],
            },
        ),
        (
            "nostalgia",
            "a remembered shared moment was recalled",
            {
                "memory_warmth": 80,
                "temporal_loss": 80,
                "salience": 80,
                "evidence_refs": [
                    {
                        "source_kind": "promoted_memory",
                        "source_id": "memory:shared-moment",
                        "occurred_at": timestamp,
                        "semantic_summary": (
                            "a remembered shared moment was recalled"
                        ),
                    },
                    {
                        "source_kind": "episode",
                        "source_id": "episode:shared-moment-cue",
                        "occurred_at": timestamp,
                        "semantic_summary": "the shared moment was recalled",
                    },
                ],
            },
        ),
    ]
    for emotion_id, description, fields in event_specs:
        evidence = {
            "source_kind": "episode",
            "source_id": f"episode:event-root-{emotion_id}",
            "occurred_at": timestamp,
            "semantic_summary": description,
        }
        state, entity_id, _created = materialize_causal_root(
            state,
            kind="event",
            primary_evidence=evidence,
            description=description,
        )
        event = next(
            row for row in state["active_events"]
            if row["entity_id"] == entity_id
        )
        event.update(deepcopy(fields))
        validate_cognition_state(state)
    payload["mutable_state"] = state
    return payload


def _workspace_baseline(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Render the pre-change projection with no authorized move rows."""

    prepared_payload = deepcopy(payload)
    _original, prepared_state, _transitions = _prepare_state_transaction(
        prepared_payload,
    )
    workspace = build_canonical_turn_workspace(
        episode=prepared_payload["episode"],
        scene_context=prepared_payload["scene_context"],
        evidence=prepared_payload["evidence"],
        mutable_state=prepared_state,
        character_constraints=prepared_payload["character_constraints"],
        identity_context=prepared_payload["character_identity_context"],
        available_actions=prepared_payload["available_actions"],
        available_resolvers=prepared_payload[
            "available_resolver_capabilities"
        ],
        overused_moves=[],
        direct_facts=prepared_payload.get("direct_facts", []),
        character_operational_context=prepared_payload.get(
            "character_operational_context",
            {},
        ),
        character_affect_context=prepared_payload.get(
            "character_affect_context",
            [],
        ),
        relationship_context=prepared_payload.get(
            "relationship_context",
            {},
        ),
        resolver_context=prepared_payload.get("resolver_context", ""),
        resolver_progress=prepared_payload.get(
            "resolver_goal_progress",
            {},
        ),
        runtime_limits=prepared_payload.get(
            "runtime_capability_limits",
            [],
        ),
        group_engagement=prepared_payload.get(
            "group_engagement_action_context",
            {},
        ),
    )
    a1 = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A1",
    )
    a2 = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A2",
        accepted_appraisal_summary=[],
    )
    goal = build_canonical_goal_question(
        workspace=workspace,
        appraisal_summary=[],
    )
    plan = build_canonical_plan_question(
        workspace=workspace,
        goal={
            "goal_kind": "ordinary_response",
            "intent": "answer the current observation",
            "reason": "the current observation needs an answer",
            "cause_summary": "the current observation",
        },
        appraisal_summary=[],
    )
    return {"A1": a1, "A2": a2, "G": goal, "P": plan}


def _protected_packets(
    records: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    """Decode the exact protected human packets for each cognition stage."""

    packets: dict[str, dict[str, Any]] = {}
    for record in records:
        stage = record.get("stage")
        if not isinstance(stage, str) or stage in packets:
            continue
        messages = record.get("messages")
        if not isinstance(messages, list):
            continue
        human_message = next(
            (
                message for message in messages
                if isinstance(message, dict)
                and message.get("role") == "human"
            ),
            None,
        )
        if not isinstance(human_message, dict):
            continue
        content = human_message.get("content")
        if isinstance(content, str):
            packets[stage] = json.loads(content)
    return packets


def _required_affect_ids(state: dict[str, Any]) -> list[str]:
    """Return required event-root emotion ids from a prepared state."""

    return [
        row["emotion_id"]
        for row in state["affect_activations"]
        if row["emotion_id"] in _REQUIRED_EMOTIONS
        and row["primary_root"]["kind"] == "event"
    ]


async def test_live_recorder_recognizes_semantic_paraphrase_without_planning() -> None:
    """Run one real scene-observer call over paraphrased visible moves."""

    submitted = record_input()
    submitted["character_name"] = "Asuna"
    submitted["decontextualized_input"] = "请记住这件具体的小事。"
    submitted["final_dialog"] = [
        "放心，我会把这件事放在心上。",
    ]
    prior_turns = []
    for index, response in enumerate((
        "我先替你把这件事放在心上。",
        "这件事我会替你记住，别担心。",
        "放心，我会把它认真留意着。",
    )):
        assistant_turn = logical_turn(
            turn_id=f"trace:semantic-move-{index}",
            row_id=f"row:semantic-move-{index}",
            trace_id=f"trace:semantic-move-{index}",
        )
        assistant_turn["role"] = "assistant"
        assistant_turn["display_name"] = "Asuna"
        assistant_turn["fragments"] = [response]
        prior_turns.append(assistant_turn)
    prior_turns.append(logical_turn(
        turn_id="row:semantic-current-user",
        row_id="row:semantic-current-user",
    ))
    submitted["interaction_logical_turns"] = prior_turns
    payload = recorder.build_scene_recorder_human_payload(submitted)
    started = time.monotonic()
    artifact: dict[str, Any] = {
        "schema": "semantic_response_progression_live_gate.v1",
        "case": "L1_recorder_semantic_paraphrase",
        "input": payload,
    }
    output = None
    try:
        invocation = await recorder._record_scene(submitted)
        output = dict(invocation.scene)
        artifact["output"] = output
        artifact["scene_payload_chars"] = invocation.human_payload_chars
        artifact["provider_usage"] = invocation.provider_usage
    except Exception as exc:
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        artifact["duration_ms"] = round(
            (time.monotonic() - started) * 1000,
            3,
        )
        path = _write_artifact("l1_recorder", artifact)
        print(f"live semantic progression artifact: {path}")

    assert output is not None
    assert output["overused_moves"]
    assert all(isinstance(row, str) and row.strip() for row in output["overused_moves"])
    assert all(
        term not in row
        for row in output["overused_moves"]
        for term in ("下一轮", "必须", "应该", "避免")
    )


async def test_live_recorder_positive_control_keeps_new_moves_empty() -> None:
    """Run one real observer call over genuinely different response moves."""

    submitted = record_input()
    submitted["character_name"] = "Asuna"
    submitted["decontextualized_input"] = (
        "会议已经改到15:30了，请确认开始时间。"
    )
    submitted["content_plan"] = {
        "semantic_content": "直接确认会议在15:30开始",
        "surface_intent": "answer current fact",
    }
    submitted["final_dialog"] = [
        "会议已经改到15:30，开始时间就是15:30。",
    ]
    prior_turns = []
    for index, response in enumerate((
        "我先替你查一下天气。",
        "把预算和截止日期列出来，我们再排顺序。",
        "这份文件的第三段需要补充来源。",
    )):
        assistant_turn = logical_turn(
            turn_id=f"trace:semantic-positive-{index}",
            row_id=f"row:semantic-positive-{index}",
            trace_id=f"trace:semantic-positive-{index}",
        )
        assistant_turn["role"] = "assistant"
        assistant_turn["display_name"] = "Asuna"
        assistant_turn["fragments"] = [response]
        prior_turns.append(assistant_turn)
    current_user = logical_turn(
        turn_id="row:semantic-positive-current-user",
        row_id="row:semantic-positive-current-user",
    )
    current_user["fragments"] = [submitted["decontextualized_input"]]
    prior_turns.append(current_user)
    submitted["interaction_logical_turns"] = prior_turns
    payload = recorder.build_scene_recorder_human_payload(submitted)
    started = time.monotonic()
    artifact: dict[str, Any] = {
        "schema": "semantic_response_progression_live_gate.v1",
        "case": "L1_recorder_positive_control_new_moves",
        "input": payload,
    }
    output = None
    try:
        invocation = await recorder._record_scene(submitted)
        output = dict(invocation.scene)
        artifact["output"] = output
        artifact["scene_payload_chars"] = invocation.human_payload_chars
        artifact["provider_usage"] = invocation.provider_usage
    except Exception as exc:
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        artifact["duration_ms"] = round(
            (time.monotonic() - started) * 1000,
            3,
        )
        path = _write_artifact("l1_recorder_positive_control", artifact)
        print(f"live semantic progression artifact: {path}")

    assert output is not None
    assert output["overused_moves"] == []


async def test_live_multi_emotion_context_preserves_original_design() -> None:
    """Run real A1/A2/G/P with five event-root emotions and four max rows."""

    payload = _multi_emotion_input()
    prepared_probe = deepcopy(payload)
    _original, prepared_state, _transitions = _prepare_state_transaction(
        prepared_probe,
    )
    derived_required_ids = _required_affect_ids(prepared_state)
    assert set(derived_required_ids) == set(_REQUIRED_EMOTIONS)
    required_ids = list(_REQUIRED_EMOTIONS)
    baseline_packets = _workspace_baseline(payload)
    immutable_baseline = _load_immutable_pre_change_evidence()

    token = bind_protected_chain_records(
        run_id="semantic-progression-l7",
        source_kind="semantic_response_progression_live_gate",
    )
    started = time.monotonic()
    artifact: dict[str, Any] = {
        "schema": "semantic_response_progression_live_gate.v1",
        "case": "L7_multi_emotion_preservation",
        "input": {
            "semantic_text": payload["episode"]["percepts"][0][
                "content"
            ]["semantic_text"],
            "overused_moves": payload["overused_moves"],
            "required_event_root_emotions": required_ids,
            "prepared_state": {
                "active_events": prepared_state["active_events"],
                "affect_activations": prepared_state["affect_activations"],
            },
        },
        "baseline_packets": baseline_packets,
        "immutable_pre_change_artifact": str(_PRE_CHANGE_ARTIFACT),
    }
    output = None
    records: tuple[dict[str, Any], ...] = ()
    try:
        output = await run_cognition(
            payload,
            build_cognition_core_services(),
        )
    except Exception as exc:
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        records = snapshot_protected_chain_records()
        artifact["protected_records"] = list(records)
        artifact["duration_ms"] = round(
            (time.monotonic() - started) * 1000,
            3,
        )
        if output is not None:
            artifact["output"] = {
                "active_character_goal": output["active_character_goal"],
                "response_plan": output["response_plan"],
                "affect_projection": output["affect_projection"],
                "cause_provenance": output["cause_provenance"],
            }
        reset_protected_chain_records(token)
        path = _write_artifact("l7_multi_emotion", artifact)
        print(f"live semantic progression artifact: {path}")

    assert output is not None
    packets = _protected_packets(records)
    assert list(packets) == ["A1", "A2", "G", "P"]
    assert len(records) == 4
    for stage in ("A1", "A2", "G", "P"):
        assert packets[stage]["continuation_state"] == (
            baseline_packets[stage]["continuation_state"]
        )
        immutable_continuation = immutable_baseline["continuation_state"][stage]
        continuation = packets[stage]["continuation_state"]
        assert continuation["active_events"] == immutable_continuation[
            "active_events"
        ]
        expected_affect = immutable_continuation["affect_activations"]
        expected_by_emotion = {
            row["emotion"]: row for row in expected_affect
        }
        candidate_by_emotion = {
            row["emotion"]: row
            for row in continuation["affect_activations"]
            if row["emotion"] in _REQUIRED_EMOTIONS
        }
        assert [
            candidate_by_emotion[emotion]
            for emotion in _REQUIRED_EMOTIONS
        ] == [
            expected_by_emotion[emotion]
            for emotion in _REQUIRED_EMOTIONS
        ]
    for stage in ("A2", "G"):
        assert packets[stage]["conditional_character_context"]["affect"] == (
            baseline_packets[stage]["conditional_character_context"]["affect"]
        )
        expected_affect = immutable_baseline[f"{stage.lower()}_affect"]
        expected_by_emotion = {
            row["emotion"]: row for row in expected_affect
        }
        candidate_affect = packets[stage][
            "conditional_character_context"
        ]["affect"]
        candidate_by_emotion = {
            row["emotion"]: row
            for row in candidate_affect
            if row["emotion"] in _REQUIRED_EMOTIONS
        }
        assert [
            candidate_by_emotion[emotion]
            for emotion in _REQUIRED_EMOTIONS
        ] == [
            expected_by_emotion[emotion]
            for emotion in _REQUIRED_EMOTIONS
        ]
    assert "overused_moves" not in packets["A1"]
    for stage in ("A2", "G", "P"):
        rows = packets[stage]["participant_continuity"]
        base_rows = baseline_packets[stage]["participant_continuity"]
        assert rows[:len(base_rows)] == base_rows
        assert [row["semantic_text"] for row in rows[len(base_rows):]] == (
            payload["overused_moves"]
        )
    assert output["active_character_goal"]["intent"]
    assert output["response_plan"]["response_goal"]


async def test_live_l3_does_not_reintroduce_unselected_semantic_payoff() -> None:
    """Run one real content-plan call with bounded prior-move evidence."""

    state = build_surface_state(build_relational_decision(stance="reject"))
    state["conversation_progress"] = {
        "overused_moves": [
            "the character already used a visible relationship payoff",
            "the character already used a second visible relationship payoff",
            "the character already used a third visible relationship payoff",
            "the character already used a fourth visible relationship payoff",
        ],
    }
    state["chat_history_recent"] = [
        {"role": "assistant", "content": "那就让我再哄你一下。"},
        {"role": "assistant", "content": "我会用亲近的方式收尾。"},
    ]
    payload = l3_surface.build_text_surface_input_from_global_state(
        state,
        interaction_style_context="brief and natural",
    )
    trace_token = llm_tracing.bind_trace_id("semantic-progression-l5")
    started = time.monotonic()
    artifact: dict[str, Any] = {
        "schema": "semantic_response_progression_live_gate.v1",
        "case": "L5_l3_selected_goal_fidelity",
        "input": payload,
    }
    output = None
    try:
        output = await surface.run_text_surface_planning(
            payload,
            l3_surface._build_text_surface_services(),
        )
        artifact["output"] = output
    except Exception as exc:
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        artifact["duration_ms"] = round(
            (time.monotonic() - started) * 1000,
            3,
        )
        artifact["trace_id"] = llm_tracing.current_trace_id()
        llm_tracing.reset_trace_id(trace_token)
        path = _write_artifact("l5_l3_surface", artifact)
        print(f"live semantic progression artifact: {path}")

    assert output is not None
    assert output["content_plan"]
    assert output["selected_surface_intent"] == payload["response_plan"][
        "response_goal"
    ]
    assert output["relational_willingness"] == payload["relational_willingness"]


def test_live_bounded_path_and_stochastic_signoff() -> None:
    """Verify the inspected direct artifacts satisfy the L8 path budget."""

    artifact_paths = {
        "l1": sorted(_ARTIFACT_DIR.glob("l1_recorder_[0-9]*.json")),
        "l1_positive": sorted(
            _ARTIFACT_DIR.glob("l1_recorder_positive_control_*.json")
        ),
        "l5": sorted(_ARTIFACT_DIR.glob("l5_l3_surface_*.json")),
        "l7": sorted(_ARTIFACT_DIR.glob("l7_multi_emotion_*.json")),
    }
    assert all(artifact_paths.values())
    l1 = json.loads(artifact_paths["l1"][-1].read_text(encoding="utf-8"))
    positive_l1 = json.loads(
        artifact_paths["l1_positive"][-1].read_text(encoding="utf-8")
    )
    l5 = json.loads(artifact_paths["l5"][-1].read_text(encoding="utf-8"))
    l7 = json.loads(artifact_paths["l7"][-1].read_text(encoding="utf-8"))

    assert l1["output"]["overused_moves"]
    assert positive_l1["output"]["overused_moves"] == []
    assert l5["output"]["content_plan"]
    records = l7["protected_records"]
    assert len(records) == 4
    assert [record["stage"] for record in records] == [
        "A1", "A2", "G", "P",
    ]
    assert all(record["status"] == "parsed" for record in records)
    assert all(
        set(record) >= {
            "stage",
            "status",
            "config",
            "messages",
            "raw_output",
            "parsed_output",
            "duration_ms",
        }
        for record in records
    )
    assert [
        record["config"]["stage_name"]
        for record in records
    ] == [
        "cognition_core_v3.A1",
        "cognition_core_v3.A2",
        "cognition_core_v3.G",
        "cognition_core_v3.P",
    ]
    assert len(l7["input"]["overused_moves"]) == 4
    assert all(
        len(move) == 120 for move in l7["input"]["overused_moves"]
    )
    assert l7["immutable_pre_change_artifact"].endswith(
        "pre_change_multi_emotion_workspace.json"
    )


def test_live_l2_private_memory_enabled_theme_release() -> None:
    """Validate the parent-coordinated L2 service conversation evidence."""

    artifacts = _load_service_gate_artifacts("L2", expected_count=2)
    assert all(len(artifact["turns"]) == 4 for artifact in artifacts)


def test_live_l3_explicit_current_user_correction() -> None:
    """Validate the parent-coordinated L3 correction evidence."""

    artifacts = _load_service_gate_artifacts("L3", expected_count=2)
    assert all(len(artifact["turns"]) == 4 for artifact in artifacts)


def test_live_l4_deliberate_continuation_positive_control() -> None:
    """Validate the parent-coordinated L4 positive-control evidence."""

    artifact = _load_service_gate_artifact("L4")
    assert len(artifact["turns"]) == 3


def test_live_l6_legitimate_memory_pressure_topic_pivot() -> None:
    """Validate the parent-coordinated L6 topic-pivot evidence."""

    artifact = _load_service_gate_artifact("L6")
    assert len(artifact["turns"]) == 2
