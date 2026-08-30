"""Focused deterministic proofs for typed stage context and evidence priority."""

from __future__ import annotations

import inspect
import json
import re

from kazusa_ai_chatbot.cognition_core_v3 import facade as facade_module
from kazusa_ai_chatbot.cognition_core_v3 import prompt as prompt_module
from kazusa_ai_chatbot.cognition_core_v3.prompt import (
    build_canonical_appraisal_question,
    build_canonical_goal_question,
    build_canonical_plan_question,
    build_canonical_turn_workspace,
    build_turn_workspace_stage_contracts,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    PENDING_TASK_CONTINUATION_VERSION,
    RESOLVER_PENDING_CONTINUATION_VERSION,
)
from kazusa_ai_chatbot.cognition_shared.state_models import validate_cognition_state
from kazusa_ai_chatbot.cognition_shared.state_reducers import materialize_causal_root
from tests.unit.cognition_core_v3.test_handleless_contract import _input

_CODE_SPAN_PATTERN = re.compile(r"`[^`]+`")
_HAN_PATTERN = re.compile(r"[\u3400-\u9fff]")
_MULTIWORD_ENGLISH_PATTERN = re.compile(
    r"\b[A-Za-z]+(?:[ \t]+[A-Za-z]+){2,}\b"
)


def _assert_native_chinese_instruction(value: str) -> None:
    assert _HAN_PATTERN.search(value)
    prose = _CODE_SPAN_PATTERN.sub("", value)
    assert _MULTIWORD_ENGLISH_PATTERN.search(prose) is None


def _workspace(
    payload: dict[str, object],
    evidence: list[dict[str, object]],
    *,
    character_affect_context: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return build_canonical_turn_workspace(
        episode=payload["episode"],
        scene_context=payload["scene_context"],
        evidence=evidence,
        mutable_state=payload["mutable_state"],
        character_constraints=payload["character_constraints"],
        identity_context=payload["character_identity_context"],
        available_actions=payload["available_actions"],
        available_resolvers=payload["available_resolver_capabilities"],
        character_operational_context=payload.get(
            "character_operational_context", {}
        ),
        relationship_context=payload["mutable_state"].get("relationship", {}),
        character_affect_context=character_affect_context,
        overused_moves=payload.get("overused_moves", []),
        response_plan_contract_variant=payload[
            "response_plan_contract_variant"
        ],
    )


def _effective_prompt(stage: str, packet: dict[str, object]) -> str:
    return facade_module._system_prompt_for_stage(stage=stage, packet=packet)


def test_cognition_chain_guidance_uses_native_chinese() -> None:
    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    packets = build_turn_workspace_stage_contracts(workspace=workspace)
    self_plan = build_canonical_plan_question(
        workspace=workspace,
        goal=packets["P"]["goal"],
        appraisal_summary=[],
        self_cognition=True,
    )
    for stage, packet in (*packets.items(), ("P", self_plan)):
        _assert_native_chinese_instruction(_effective_prompt(stage, packet))
    assert workspace["orientation"]["operation"] == "回应当前观察"
    assert "response_content_provider" not in workspace["orientation"]
    assert "selection_owner" not in workspace["orientation"]
    assert packets["P"]["goal"] == {
        "goal_kind": "open_goal",
        "intent": "理解当前请求",
        "reason": "当前观察需要一个有依据的回应",
        "cause_summary": "当前观察",
    }


def test_ordinary_plan_declares_response_goal_text_contract() -> None:
    """Keep the model-facing P schema aligned with deterministic validation."""

    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    packet = build_canonical_plan_question(
        workspace=workspace,
        goal={
            "goal_kind": "clarify",
            "intent": "answer the current observation",
            "reason": "the current observation needs an answer",
            "cause_summary": "the current observation",
        },
        appraisal_summary=[],
    )

    assert packet["output_contract"]["response_goal"] == {
        "type": "string",
        "minimum_characters": 1,
        "maximum_characters": 2000,
    }
    assert "可见措辞能够断言的内容" in _effective_prompt("P", packet)


def test_stage_context_preserves_identity_relationship_and_emotion_cause() -> None:
    payload = _input()
    payload["character_operational_context"] = {
        "schema_version": "operational.v1",
        "consumer_role": "cognition",
        "source_updated_at": payload["mutable_state"]["updated_at"],
        "effective_at": payload["mutable_state"]["updated_at"],
        "affect": [{"summary": "a concrete current feeling"}],
        "pressures": [{"summary": "a bounded current pressure"}],
    }
    affect_state = payload["mutable_state"]
    evidence = payload["evidence"][0]["evidence_ref"]
    affect_state, event_id, _created = materialize_causal_root(
        affect_state,
        kind="event",
        primary_evidence=evidence,
        description="the current observation recalls a concrete loss",
    )
    affect_state["active_events"][0]["responsibility"] = 40
    affect_state["active_events"][0]["salience"] = 40
    affect_state["affect_activations"] = [{
        "activation_id": "emotion:sadness",
        "emotion_id": "sadness",
        "primary_root": {
            "scope": "user", "kind": "event", "entity_id": event_id,
        },
        "root_refs": [{
            "scope": "user", "kind": "event", "entity_id": event_id,
        }],
        "phase": "active",
        "score": 65,
        "peak_score": 65,
        "trend": "rising",
        "cause_status": "active",
        "started_at": affect_state["updated_at"],
        "updated_at": affect_state["updated_at"],
        "last_reinforced_at": affect_state["updated_at"],
    }]
    validate_cognition_state(affect_state)
    payload["mutable_state"] = affect_state
    workspace = _workspace(payload, payload["evidence"])
    a2 = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A2",
        accepted_appraisal_summary=[{
            "family": "event_agency",
            "applicable": True,
            "semantic_summary": "the event has meaning",
            "cause_summary": "the observed action caused the change",
        }],
    )
    context = a2["conditional_character_context"]
    assert context["identity"]["name"] == "Test Character"
    assert context["constraints"]["standards"]
    assert context["relationship"]["axes"]["trust"] == "信任尚未建立"
    assert context["affect"][0]["emotion"] == "sadness"
    assert context["affect"][0]["cause_summary"] == (
        "the current observation recalls a concrete loss"
    )
    operational = context["operational"]
    assert set(operational) == {"affect", "pressures"}
    assert context["constraints"]["standards"][0][
        "importance"
    ] == "高"


def test_actual_identity_partitions_are_selected_by_stage() -> None:
    payload = _input()
    payload["character_identity_context"] = {
        family: {
            "core": {"name": "Test Character", "description": "grounded"},
            "personality": {"logic": "careful"},
            "boundaries": {"avoid_harm": "high"},
        }
        for family in (
            "event_agency", "goal_threat_outcome",
            "epistemic_comparison_memory", "relationship_social",
            "moral_identity", "existential_drive", "goal_cognition",
        )
    }
    workspace = _workspace(payload, payload["evidence"])
    a1 = build_canonical_appraisal_question(workspace=workspace, stage_name="A1")
    a2 = build_canonical_appraisal_question(workspace=workspace, stage_name="A2")
    goal = build_canonical_goal_question(
        workspace=workspace,
        appraisal_summary=[],
    )
    assert "conditional_character_context" not in a1
    assert set(a2["conditional_character_context"]["identity"]) == {
        "relationship_social", "moral_identity", "existential_drive",
    }
    assert set(goal["conditional_character_context"]["identity"]) == {
        "goal_cognition",
    }
    assert "constraints" in a2["conditional_character_context"]


def test_scheduler_visibility_reaches_a1_goal_threat_but_not_a2() -> None:
    payload = _input()
    scheduler = {
        "evidence_ref": {
            "source_kind": "scheduler_event",
            "source_id": "scheduler:due",
            "occurred_at": payload["mutable_state"]["updated_at"],
            "semantic_summary": "a scheduled task is due",
        },
        "semantic_text": "a scheduled task is due",
        "visible_to": ["q:goal_threat_outcome"],
        "authority": "current_event",
    }
    workspace = _workspace(payload, [*payload["evidence"], scheduler])
    a1 = build_canonical_appraisal_question(workspace=workspace, stage_name="A1")
    a2 = build_canonical_appraisal_question(workspace=workspace, stage_name="A2")
    assert any(
        row["semantic_text"] == "a scheduled task is due"
        for row in a1["current_observation"]["evidence"]
    )
    assert all(
        row["semantic_text"] != "a scheduled task is due"
        for row in a2["current_observation"]["evidence"]
    )


def test_current_resolver_and_action_evidence_precedes_supporting_rag() -> None:
    payload = _input()
    current_resolver = {
        "evidence_ref": {
            "source_kind": "resolver_observation",
            "source_id": "resolver:current",
            "occurred_at": payload["mutable_state"]["updated_at"],
            "semantic_summary": "current resolver result",
        },
        "semantic_text": "current resolver result",
        "authority": "supporting",
    }
    current_action = {
        "evidence_ref": {
            "source_kind": "action_result",
            "source_id": "action:current",
            "occurred_at": payload["mutable_state"]["updated_at"],
            "semantic_summary": "current action result",
        },
        "semantic_text": "current action result",
        "authority": "supporting",
    }
    rag_rows = [
        {
                "evidence_ref": {
                "source_kind": "promoted_memory",
                "source_id": f"rag:{index}",
                "occurred_at": payload["mutable_state"]["updated_at"],
                "semantic_summary": f"supporting memory {index}",
            },
            "semantic_text": f"supporting memory {index}",
            "authority": "supporting",
            "memory_scope": "current_user_continuity",
        }
        for index in range(64)
    ]
    workspace = _workspace(payload, [*rag_rows, current_resolver, current_action])
    a1 = build_canonical_appraisal_question(workspace=workspace, stage_name="A1")
    goal = build_canonical_goal_question(workspace=workspace, appraisal_summary=[])
    for packet in (a1, goal):
        evidence = packet["direct_facts"]["evidence"]
        assert len(evidence) == 32
        assert evidence[0]["semantic_text"] == "current resolver result"
        assert evidence[1]["semantic_text"] == "current action result"
        assert all(
            row["semantic_text"] != "supporting memory 0"
            for row in evidence[:2]
        )


def test_character_affect_projection_reaches_a2_and_goal_context() -> None:
    payload = _input()
    character_affect = [{
        "emotion": "sadness",
        "phase": "active",
        "intensity": "high",
        "trend": "rising",
        "cause_status": "active",
        "cause_summary": "the character remembers a concrete loss",
    }]
    workspace = _workspace(
        payload,
        payload["evidence"],
        character_affect_context=character_affect,
    )
    a2 = build_canonical_appraisal_question(workspace=workspace, stage_name="A2")
    goal = build_canonical_goal_question(workspace=workspace, appraisal_summary=[])
    for packet in (a2, goal):
        affect_context = packet["conditional_character_context"]["affect"]
        assert any(
            row.get("cause_summary") == "the character remembers a concrete loss"
            for row in affect_context
        )


def test_stage_authority_lanes_partition_fact_continuity_and_character_context() -> None:
    """Every semantic source reaches only its declared authority lane."""

    payload = _input()
    timestamp = payload["mutable_state"]["updated_at"]
    evidence = [
        *payload["evidence"],
        {
            "evidence_ref": {
                "source_kind": "resolver_observation",
                "source_id": "resolver:fact",
                "occurred_at": timestamp,
                "semantic_summary": "the resolver confirmed one fact",
            },
            "semantic_text": "the resolver confirmed one fact",
            "authority": "contextual_fact_only",
        },
        {
            "evidence_ref": {
                "source_kind": "conversation_evidence",
                "source_id": "conversation:prior",
                "occurred_at": timestamp,
                "semantic_summary": "the user discussed a prior action",
            },
            "semantic_text": "the user discussed a prior action",
            "authority": "participant_continuity",
        },
        {
            "evidence_ref": {
                "source_kind": "promoted_reflection",
                "source_id": "reflection:tendency",
                "occurred_at": timestamp,
                "semantic_summary": "the character tends to test reciprocity",
            },
            "semantic_text": "the character tends to test reciprocity",
            "authority": "conditional_character_guidance",
        },
    ]
    workspace = _workspace(payload, evidence)
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
            "goal_kind": "clarify",
            "intent": "clarify the observation",
            "reason": "the referent is unknown",
            "cause_summary": "the current observation",
        },
        appraisal_summary=[],
    )

    assert [
        row["semantic_text"]
        for row in a1["current_observation"]["evidence"]
    ] == ["the current observation"]
    assert [
        row["semantic_text"] for row in a1["direct_facts"]["evidence"]
    ] == ["the resolver confirmed one fact"]
    assert "participant_continuity" not in a1
    assert "conditional_character_context" not in a1
    assert [
        row["semantic_text"] for row in a2["participant_continuity"]
    ] == ["the user discussed a prior action"]
    assert [
        row["semantic_text"]
        for row in a2["conditional_character_context"]["evidence"]
    ] == ["the character tends to test reciprocity"]
    assert goal["participant_continuity"] == a2["participant_continuity"]
    assert plan["participant_continuity"] == a2["participant_continuity"]
    assert "conditional_character_context" not in plan
    prompt = _effective_prompt("P", plan)
    assert "缺少证据不等于否定事实" in prompt
    assert "不确定的解释" in prompt


def test_overused_moves_reach_participant_continuity_after_a1_only() -> None:
    """Expose observed response moves to A2/G/P while keeping A1 current-only."""

    payload = _input()
    payload["overused_moves"] = [
        "the character has already used one visible relationship maneuver",
        "the character has already used a second visible maneuver",
    ]
    workspace = _workspace(payload, payload["evidence"])

    a1 = build_canonical_appraisal_question(workspace=workspace, stage_name="A1")
    a2 = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A2",
        accepted_appraisal_summary=[],
    )
    goal = build_canonical_goal_question(workspace=workspace, appraisal_summary=[])
    plan = build_canonical_plan_question(
        workspace=workspace,
        goal={
            "goal_kind": "clarify",
            "intent": "answer the current observation",
            "reason": "the current observation needs an answer",
            "cause_summary": "the current observation",
        },
        appraisal_summary=[],
    )

    assert "participant_continuity" not in a1
    assert all(move not in str(a1) for move in payload["overused_moves"])
    for packet in (a2, goal, plan):
        assert [
            row["semantic_text"]
            for row in packet["participant_continuity"][-2:]
        ] == payload["overused_moves"]


def test_user_owned_semantic_correction_guides_a1_goal_and_plan_without_hidden_intent_assertion() -> None:
    """Keep explicit current-user meaning authoritative across stages."""

    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    a1 = build_canonical_appraisal_question(workspace=workspace, stage_name="A1")
    goal = build_canonical_goal_question(workspace=workspace, appraisal_summary=[])
    plan = build_canonical_plan_question(
        workspace=workspace,
        goal={
            "goal_kind": "clarify",
            "intent": "answer the current observation",
            "reason": "the current observation needs an answer",
            "cause_summary": "the current observation",
        },
        appraisal_summary=[],
    )

    for stage, packet in (("A1", a1), ("G", goal), ("P", plan)):
        prompt = _effective_prompt(stage, packet)
        assert "当前用户明确纠正自己的意思或感受时" in prompt
        assert "纠正本身不证明相反意思" in prompt


def test_goal_guidance_progresses_current_delta_and_preserves_deliberate_reopening() -> None:
    """Keep response goals current-delta grounded with an explicit reopen path."""

    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    for stage, packet in (
        ("G", build_canonical_goal_question(workspace=workspace, appraisal_summary=[])),
        ("P", build_canonical_plan_question(
            workspace=workspace,
            goal={
                "goal_kind": "clarify",
                "intent": "answer the current observation",
                "reason": "the current observation needs an answer",
                "cause_summary": "the current observation",
            },
            appraisal_summary=[],
        )),
    ):
        prompt = _effective_prompt(stage, packet)
        assert "当前观察新增加、改变、纠正、询问或仍未解决的内容" in prompt
        assert "此前回应模式" in prompt
        assert "当前明确重新打开相关事项" in prompt or "用户当前重新打开它们" in prompt


def test_recipient_scoped_permission_rule_reaches_a2_goal_and_plan() -> None:
    """A2, G, and P receive the same recipient-applicability boundary."""

    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    a1 = build_canonical_appraisal_question(workspace=workspace, stage_name="A1")
    a2 = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A2",
        accepted_appraisal_summary=[],
    )
    goal = build_canonical_goal_question(workspace=workspace, appraisal_summary=[])
    plan = build_canonical_plan_question(
        workspace=workspace,
        goal={
            "goal_kind": "clarify",
            "intent": "answer the current observation",
            "reason": "the current observation needs an answer",
            "cause_summary": "the current observation",
        },
        appraisal_summary=[],
    )

    recipient_rule = "公开可见不把一名参与者的同意、许可、承诺、关系或角色转移给另一名参与者"
    assert recipient_rule not in _effective_prompt("A1", a1)
    assert recipient_rule in _effective_prompt("A2", a2)
    assert recipient_rule in _effective_prompt("G", goal)
    assert recipient_rule in _effective_prompt("P", plan)


def test_semantic_progression_context_preserves_all_existing_multi_affect_rows_and_causes() -> None:
    """Adding bounded moves cannot alter any continuation or affect row."""

    payload = _input()
    state = payload["mutable_state"]
    timestamp = state["updated_at"]
    emotions = [
        ("sadness", "a concrete loss remains unresolved"),
        ("anger", "a boundary was crossed in the current event"),
        ("gratitude", "a specific act of care was received"),
        ("embarrassment", "a private mistake became visible"),
        ("nostalgia", "a remembered shared moment was recalled"),
    ]
    for index, (emotion, cause) in enumerate(emotions):
        state, root_id, _created = materialize_causal_root(
            state,
            kind="event",
            primary_evidence={
                "source_kind": "episode",
                "source_id": f"episode:multi-affect-{index}",
                "occurred_at": timestamp,
                "semantic_summary": cause,
            },
            description=cause,
        )
        state["active_events"][-1]["salience"] = 70 - index
        state["affect_activations"].append({
            "activation_id": f"emotion:{emotion}",
            "emotion_id": emotion,
            "primary_root": {
                "scope": "user",
                "kind": "event",
                "entity_id": root_id,
            },
            "root_refs": [{
                "scope": "user",
                "kind": "event",
                "entity_id": root_id,
            }],
            "phase": "active",
            "score": 70 - index,
            "peak_score": 70 - index,
            "trend": "rising",
            "cause_status": "active",
            "started_at": timestamp,
            "updated_at": timestamp,
            "last_reinforced_at": timestamp,
        })
    validate_cognition_state(state)
    payload["mutable_state"] = state
    baseline = _workspace(payload, payload["evidence"])
    payload["overused_moves"] = [
        "a" * 120,
        "b" * 120,
        "c" * 120,
        "d" * 120,
    ]
    candidate = _workspace(payload, payload["evidence"])
    baseline_packets = build_turn_workspace_stage_contracts(workspace=baseline)
    candidate_packets = build_turn_workspace_stage_contracts(workspace=candidate)

    for stage in ("A1", "A2", "G", "P"):
        assert candidate_packets[stage]["continuation_state"] == baseline_packets[stage]["continuation_state"]
        continuation = candidate_packets[stage]["continuation_state"]
        assert [
            (row["description"], row["status"])
            for row in continuation["active_events"]
        ] == [(cause, "active") for _emotion, cause in emotions]
        expected_emotions = {emotion for emotion, _cause in emotions}
        assert [
            (row["emotion"], row["cause_summary"])
            for row in continuation["affect_activations"]
            if row["emotion"] in expected_emotions
        ] == emotions
    for stage in ("A2", "G"):
        assert candidate_packets[stage]["conditional_character_context"]["affect"] == (
            baseline_packets[stage]["conditional_character_context"]["affect"]
        )
        assert [
            (row["emotion"], row["cause_summary"])
            for row in candidate_packets[stage][
                "conditional_character_context"
            ]["affect"]
        ] == emotions


def test_a1_excludes_conditional_character_context() -> None:
    """A1 remains a world-facing appraisal over facts and causal pressure."""

    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    packet = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A1",
    )

    rendered = str(packet)
    assert "conditional_character_context" not in packet
    assert "character_context" not in packet
    assert "personality" not in rendered


def test_a1_guidance_separates_current_observation_from_stable_direct_facts() -> None:
    """A1 keeps stable facts outside the current-observation authority lane."""

    payload = _input()
    timestamp = payload["mutable_state"]["updated_at"]
    stable_fact = {
        "evidence_ref": {
            "source_kind": "promoted_memory",
            "source_id": "promoted-memory:fact:stable-1",
            "occurred_at": timestamp,
            "semantic_summary": "stable background fact",
        },
        "semantic_text": "stable background fact",
        "visible_to": ["q:event_agency"],
        "authority": "character_world_context",
        "memory_scope": "shared_character_or_world",
    }
    workspace = _workspace(payload, [payload["evidence"][0], stable_fact])

    packet = build_canonical_appraisal_question(
        workspace=workspace,
        stage_name="A1",
    )

    current_text = [
        row["semantic_text"]
        for row in packet["current_observation"]["evidence"]
    ]
    direct_text = [
        row["semantic_text"]
        for row in packet["direct_facts"]["evidence"]
    ]
    assert "the current observation" in current_text
    assert "stable background fact" in direct_text
    assert "stable background fact" not in current_text


def test_goal_and_plan_packets_share_background_goal_authority_contract() -> None:
    """G and P packets consume one shared background-goal contract."""

    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    goal = build_canonical_goal_question(
        workspace=workspace,
        appraisal_summary=[],
    )
    plan = build_canonical_plan_question(
        workspace=workspace,
        goal={
            "goal_kind": "clarify",
            "intent": "answer the current observation",
            "reason": "the current observation needs an answer",
            "cause_summary": "the current observation",
        },
        appraisal_summary=[],
    )

    for stage, packet in (("G", goal), ("P", plan)):
        prompt = _effective_prompt(stage, packet)
        assert "只有当前观察把它们带入当前请求、决定或未解决事项时" in prompt


def test_request_agency_authority_contract_reaches_all_cognition_stage_packets() -> None:
    """Every cognition packet consumes one current-observation authority contract."""

    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    packets = build_turn_workspace_stage_contracts(workspace=workspace)
    for stage in ("A1", "A2", "G", "P"):
        prompt = _effective_prompt(stage, packets[stage])
        assert "current_observation` 是用户当下行动、意图、接受、许可和回应对象的唯一当前依据" in prompt
        assert "guidance" not in packets[stage]


def test_request_agency_contract_rejects_circular_restated_evidence() -> None:
    """The agency contract distinguishes independent facts from restatements."""

    required_invariants = (
        "同一请求的存在或清晰程度",
        "邀请参与",
        "上游改述",
        "独立证据",
        "另行表达的当前观察事实",
    )
    for prompt in (
        facade_module._A1_SYSTEM_PROMPT,
        facade_module._A2_SYSTEM_PROMPT,
        facade_module._G_SYSTEM_PROMPT,
        facade_module._P_ORDINARY_SYSTEM_PROMPT,
    ):
        assert all(invariant in prompt for invariant in required_invariants)


def test_response_content_provider_is_reply_content_fact_across_all_stages() -> None:
    """Procedural provider metadata stays outside model-facing cognition."""

    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    packets = build_turn_workspace_stage_contracts(workspace=workspace)
    hidden_fields = (
        "response_content_provider_role",
        "response_content_provider",
    )

    for stage in ("A1", "A2", "G", "P"):
        rendered_packet = str(packets[stage])
        assert all(field not in rendered_packet for field in hidden_fields)
        assert all(field not in _effective_prompt(stage, packets[stage]) for field in hidden_fields)


def test_a2_existential_drive_keeps_character_experience_distinct_from_user_state() -> None:
    """Existential-drive axes remain character-owned rather than user-owned."""

    required_invariants = ("existential_drive", "独立当前事实")
    prompt = facade_module._A2_SYSTEM_PROMPT
    assert all(invariant in prompt for invariant in required_invariants)


def test_g_relational_carriers_separate_character_motive_from_user_relationship_fact() -> None:
    """G carrier fields keep character motive distinct from user relationship facts."""

    required_invariants = (
        "relational_willingness.reason",
        "cause_summary",
        "private_monologue",
        "current_observation",
    )

    prompt = facade_module._G_SYSTEM_PROMPT
    assert all(invariant in prompt for invariant in required_invariants)


def test_request_agency_contract_separates_authorization_from_motivation() -> None:
    """Authorization scope does not establish motive or broader meaning."""

    required_invariants = (
        "授权只在当前观察写明的对象、行动、时间和条件内成立",
        "并不证明动机或更广含义",
    )
    for prompt in (
        facade_module._A1_SYSTEM_PROMPT,
        facade_module._A2_SYSTEM_PROMPT,
        facade_module._G_SYSTEM_PROMPT,
        facade_module._P_ORDINARY_SYSTEM_PROMPT,
    ):
        assert all(invariant in prompt for invariant in required_invariants)


def test_response_content_provider_is_reply_content_source_not_external_agency_transfer() -> None:
    """Explicit dialog meaning remains the model-facing authority source."""

    prompt = facade_module._G_SYSTEM_PROMPT
    assert "`current_observation`" in prompt
    assert "response_content_provider_role" not in prompt
    assert "response_content_provider" not in prompt


def test_a2_system_and_packet_share_relationship_state_evidence_contract() -> None:
    """A2 packet and system guidance share one relationship evidence owner."""

    required_invariants = (
        "relationship_social",
        "当前交互角色或范围许可",
        "不能改变任何关系轴",
        "另行表达的`current_observation`关系事实",
    )

    prompt = facade_module._A2_SYSTEM_PROMPT
    assert all(invariant in prompt for invariant in required_invariants)


def test_g_system_and_packet_share_relational_carrier_evidence_contract() -> None:
    """G packet and system guidance share one relational carrier owner."""

    required_invariants = (
        "relational_willingness",
        "private_monologue",
        "当前交互角色或范围许可",
        "不能为用户补写关系事实",
        "当前关系事实",
    )

    prompt = facade_module._G_SYSTEM_PROMPT
    assert all(invariant in prompt for invariant in required_invariants)
    assert "relational_willingness" not in facade_module._A1_SYSTEM_PROMPT


def test_g_relational_carrier_does_not_turn_unsupported_user_meaning_into_first_person_feeling() -> None:
    """Private monologue cannot launder unsupported user relationship meaning."""

    required_invariants = (
        "未经当前关系事实支持的用户关系含义",
        "第一人称感受、内心独白、被动经历",
        "relational_willingness",
        "private_monologue",
        "取得依据",
    )

    assert all(
        invariant in facade_module._G_SYSTEM_PROMPT
        for invariant in required_invariants
    )


def test_packets_project_only_state_and_typed_output_contracts() -> None:
    payload = _input()
    packets = build_turn_workspace_stage_contracts(
        workspace=_workspace(payload, payload["evidence"]),
    )

    for packet in packets.values():
        assert "guidance" not in packet
        assert "current_observation" in packet
        assert "output_contract" in packet
    rendered = json.dumps(packets, ensure_ascii=False)
    assert "repair_instruction" not in rendered
    assert "source_id" not in rendered


def test_all_complete_prompts_own_current_authority_and_repair_semantics() -> None:
    prompts = (
        facade_module._A1_SYSTEM_PROMPT,
        facade_module._A2_SYSTEM_PROMPT,
        facade_module._G_SYSTEM_PROMPT,
        facade_module._P_ORDINARY_SYSTEM_PROMPT,
        facade_module._P_PENDING_CLARIFICATION_SYSTEM_PROMPT,
        facade_module._P_DSH_INTERACTION_SYSTEM_PROMPT,
        facade_module._P_PENDING_AND_DSH_SYSTEM_PROMPT,
        facade_module._P_SELF_COGNITION_SYSTEM_PROMPT,
    )

    for prompt in prompts:
        assert "current_observation" in prompt
        assert "resolver_goal_progress" in prompt
        assert "contract_repair" in prompt
        assert "JSON" not in prompt
        assert "严格返回" not in prompt


def test_cognition_prompt_owners_do_not_compose_authored_prompt_fragments() -> None:
    """Prompt wording is one literal per exact model-call contract."""

    source = inspect.getsource(facade_module)
    assert ".format(" not in source
    assert "join((" not in source
    assert "repair_instruction" not in source
    assert "\"guidance\"" not in inspect.getsource(prompt_module)


def test_plan_variant_selection_matches_projected_pending_lanes() -> None:
    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    ordinary = build_canonical_plan_question(
        workspace=workspace,
        goal={"goal_kind": "open_goal"},
        appraisal_summary=[],
    )
    assert _effective_prompt("P", ordinary) == facade_module._P_ORDINARY_SYSTEM_PROMPT

    workspace["pending_resolver_continuation"] = {
        "schema_version": RESOLVER_PENDING_CONTINUATION_VERSION,
        "capability_kind": "human_clarification",
        "status": "waiting_for_user",
        "original_goal": "取得带背景要求的证据。",
        "question": "请补充地点。",
        "pending_task_continuation": {
            "schema_version": PENDING_TASK_CONTINUATION_VERSION,
            "on_answered_clarification": "background_task_admission",
        },
    }
    workspace["response_plan_contract_variant"] = "open_pending_resolution"
    pending = build_canonical_plan_question(
        workspace=workspace,
        goal={"goal_kind": "open_goal"},
        appraisal_summary=[],
    )
    assert "pending_resolver_continuation" in pending
    assert "response_plan_contract_variant" not in json.dumps(pending)
    assert _effective_prompt("P", pending) == (
        facade_module._P_PENDING_CLARIFICATION_SYSTEM_PROMPT
    )

    post_workspace = _workspace(payload, payload["evidence"])
    post_workspace["response_plan_contract_variant"] = "post_pending_resolution"
    post_workspace["capabilities"] = {
        **post_workspace["capabilities"],
        "resolvers": [
            row
            for row in post_workspace["capabilities"]["resolvers"]
            if row["capability"] != "human_clarification"
        ],
    }
    post_pending = build_canonical_plan_question(
        workspace=post_workspace,
        goal={"goal_kind": "open_goal"},
        appraisal_summary=[],
    )
    assert "response_plan_contract_variant" not in post_pending["output_contract"]
    assert "pending_resolver_continuation" not in post_pending
    assert "pending_task_continuation" not in post_pending["output_contract"]
    assert "pending_resolution_fields" not in post_pending["output_contract"]
    assert "post_pending_resolution" not in json.dumps(post_pending)
    assert all(
        row["capability"] != "human_clarification"
        for row in post_pending["capabilities"]["resolvers"]
    )
    assert _effective_prompt("P", post_pending) == (
        facade_module._P_ORDINARY_SYSTEM_PROMPT
    )


def test_resolver_goal_progress_survives_without_pending_clarification() -> None:
    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    workspace["resolver_goal_progress"] = {
        "schema_version": "resolver_goal_progress.v1",
        "original_goal": "完成带背景要求的证据任务。",
        "current_focus": "取得证据。",
        "deliverables": [],
        "missing_user_inputs": [],
        "evidence_dependencies": ["来源证据"],
        "attempted_paths": [],
        "source_backed_facts": [],
        "assumptions_or_inferences": [],
        "blockers": [],
        "final_response_requirements": ["精确标记：已完成"],
    }
    packets = build_turn_workspace_stage_contracts(workspace=workspace)

    for packet in packets.values():
        assert packet["resolver_goal_progress"]["original_goal"] == (
            "完成带背景要求的证据任务。"
        )
        assert "pending_resolver_continuation" not in packet
    assert "pending_resolution" not in packets["P"]["output_contract"][
        "required_fields"
    ]
