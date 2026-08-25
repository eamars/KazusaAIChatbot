"""Focused deterministic proofs for typed stage context and evidence priority."""

from __future__ import annotations

import re

from kazusa_ai_chatbot.cognition_core_v3 import facade as facade_module
from kazusa_ai_chatbot.cognition_core_v3 import prompt as prompt_module
from kazusa_ai_chatbot.cognition_core_v3.prompt import (
    A1_QUESTION_GUIDANCE,
    A2_QUESTION_GUIDANCE,
    APPRAISAL_QUESTION_GUIDANCE,
    BACKGROUND_CONTEXT_GOAL_AUTHORITY_GUIDANCE,
    CURRENT_OBSERVATION_AUTHORITY_GUIDANCE,
    GOAL_QUESTION_GUIDANCE,
    ORDINARY_PLAN_GUIDANCE,
    RECIPIENT_APPLICABILITY_GUIDANCE,
    SELF_PLAN_GUIDANCE,
    build_canonical_appraisal_question,
    build_canonical_goal_question,
    build_canonical_plan_question,
    build_canonical_turn_workspace,
    build_turn_workspace_stage_contracts,
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
    )


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
    guidance_values = (
        A1_QUESTION_GUIDANCE,
        A2_QUESTION_GUIDANCE,
        APPRAISAL_QUESTION_GUIDANCE,
        GOAL_QUESTION_GUIDANCE,
        ORDINARY_PLAN_GUIDANCE,
        SELF_PLAN_GUIDANCE,
        *(packet["guidance"] for packet in packets.values()),
        self_plan["guidance"],
    )
    for guidance in guidance_values:
        _assert_native_chinese_instruction(guidance)
    assert workspace["orientation"]["operation"] == "回应当前观察"
    assert "response_content_provider" not in workspace["orientation"]
    assert "selection_owner" not in workspace["orientation"]
    assert packets["P"]["goal"] == {
        "goal_kind": "open_goal",
        "intent": "理解当前请求",
        "reason": "当前观察需要一个有依据的回应",
        "cause_summary": "当前观察",
    }


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
    assert "缺少证据" in plan["guidance"]
    assert "明确表达不确定性" in plan["guidance"]
    assert "不得把输入中的权威通道名称复制到输出对象中" in plan["guidance"]


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

    for packet in (a1, goal, plan):
        assert "当前用户明确纠正自己的意思时" in packet["guidance"]
        assert "纠正本身不是相反意思的证据" in packet["guidance"]


def test_goal_guidance_progresses_current_delta_and_preserves_deliberate_reopening() -> None:
    """Keep response goals current-delta grounded with an explicit reopen path."""

    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    authority_distinctions = (
        "继续处理同一任务或话题，本身不会继续或重新打开角色此前使用或提出的回应方式、提议、要求、条件或关系性回报。",
        "角色尚未得到回应的提议只能作为参与者连续性，不能当作当前用户的意图、接受、承诺或必须追求的当前目标。",
        "只有当前用户回应、接受、拒绝、提及、询问、实质改变或明确重新打开该回应事项时，才可以再次选择它。",
        "角色倾向可以影响语气和立场，但不能取代当前语义增量成为主要目标。",
    )
    for packet in (
        build_canonical_goal_question(workspace=workspace, appraisal_summary=[]),
        build_canonical_plan_question(
            workspace=workspace,
            goal={
                "goal_kind": "clarify",
                "intent": "answer the current observation",
                "reason": "the current observation needs an answer",
                "cause_summary": "the current observation",
            },
            appraisal_summary=[],
        ),
    ):
        assert "当前观察新增加、改变、纠正、询问或仍未解决的内容" in packet["guidance"]
        assert "只有当前用户继续、深化、实质改变或重新打开同一事项时" in packet["guidance"]
        for distinction in authority_distinctions:
            assert distinction in packet["guidance"]


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

    normalized_rule = "".join(RECIPIENT_APPLICABILITY_GUIDANCE.split())
    assert normalized_rule not in "".join(a1["guidance"].split())
    assert normalized_rule in "".join(a2["guidance"].split())
    assert normalized_rule in "".join(goal["guidance"].split())
    assert normalized_rule in "".join(plan["guidance"].split())


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
    authority_contract = BACKGROUND_CONTEXT_GOAL_AUTHORITY_GUIDANCE.strip()
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

    assert goal["guidance"].count(authority_contract) == 1
    assert plan["guidance"].count(authority_contract) == 1


def test_request_agency_authority_contract_reaches_all_cognition_stage_packets() -> None:
    """Every cognition packet consumes one current-observation authority contract."""

    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    packets = build_turn_workspace_stage_contracts(workspace=workspace)
    authority_contract = CURRENT_OBSERVATION_AUTHORITY_GUIDANCE.strip()

    for stage in ("A1", "A2", "G", "P"):
        assert packets[stage]["guidance"].count(authority_contract) == 1


def test_request_agency_contract_rejects_circular_restated_evidence() -> None:
    """The agency contract distinguishes independent facts from restatements."""

    contract = CURRENT_OBSERVATION_AUTHORITY_GUIDANCE
    required_invariants = (
        "同一请求的存在或清晰程度",
        "上游改述",
        "独立证据",
        "另行表达的`current_observation`事实",
    )

    assert all(invariant in contract for invariant in required_invariants)


def test_response_content_provider_is_reply_content_fact_across_all_stages() -> None:
    """Procedural provider metadata stays outside model-facing cognition."""

    payload = _input()
    workspace = _workspace(payload, payload["evidence"])
    packets = build_turn_workspace_stage_contracts(workspace=workspace)
    hidden_fields = (
        "response_content_provider_role",
        "response_content_provider",
    )

    assert all(
        field not in CURRENT_OBSERVATION_AUTHORITY_GUIDANCE
        for field in hidden_fields
    )
    for stage in ("A1", "A2", "G", "P"):
        orientation = packets[stage]["orientation"]
        assert all(field not in orientation for field in hidden_fields)
        assert all(field not in packets[stage]["guidance"] for field in hidden_fields)
        assert all(
            field not in facade_module._STAGE_SYSTEM_PROMPTS[stage]
            for field in hidden_fields
        )


def test_a2_existential_drive_keeps_character_experience_distinct_from_user_state() -> None:
    """Existential-drive axes remain character-owned rather than user-owned."""

    contract = getattr(
        prompt_module,
        "A2_EXISTENTIAL_DRIVE_EVIDENCE_GUIDANCE",
        "",
    ).strip()

    required_invariants = ("existential_drive", "current_observation")
    assert contract
    assert all(invariant in contract for invariant in required_invariants)
    assert A2_QUESTION_GUIDANCE.count(contract) == 1
    assert facade_module._STAGE_SYSTEM_PROMPTS["A2"].count(contract) == 1


def test_g_relational_carriers_separate_character_motive_from_user_relationship_fact() -> None:
    """G carrier fields keep character motive distinct from user relationship facts."""

    contract = getattr(
        prompt_module,
        "G_RELATIONAL_CARRIER_EVIDENCE_GUIDANCE",
        "",
    ).strip()
    required_invariants = (
        "relational_willingness.reason",
        "cause_summary",
        "private_monologue",
        "current_observation",
    )

    assert contract
    assert all(invariant in contract for invariant in required_invariants)
    assert GOAL_QUESTION_GUIDANCE.count(contract) == 1
    assert facade_module._STAGE_SYSTEM_PROMPTS["G"].count(contract) == 1


def test_request_agency_contract_separates_authorization_from_motivation() -> None:
    """Authorization scope does not establish motive or broader meaning."""

    contract = CURRENT_OBSERVATION_AUTHORITY_GUIDANCE
    required_invariants = (
        "授权只证明",
        "当前观察所写对象、行动、时间和条件内的许可",
        "授权本身不证明动机",
        "上述更广含义",
    )

    assert all(invariant in contract for invariant in required_invariants)


def test_response_content_provider_is_reply_content_source_not_external_agency_transfer() -> None:
    """Explicit dialog meaning remains the model-facing authority source."""

    contract = CURRENT_OBSERVATION_AUTHORITY_GUIDANCE
    assert "`current_observation`" in contract
    assert "response_content_provider_role" not in contract
    assert "response_content_provider" not in contract


def test_a2_system_and_packet_share_relationship_state_evidence_contract() -> None:
    """A2 packet and system guidance share one relationship evidence owner."""

    contract = getattr(
        prompt_module,
        "A2_RELATIONSHIP_STATE_EVIDENCE_GUIDANCE",
        "",
    ).strip()
    required_invariants = (
        "relationship_social",
        "当前交互角色或范围许可",
        "不能改变任何关系轴",
        "另行表达的`current_observation`关系事实",
    )

    assert contract
    assert all(invariant in contract for invariant in required_invariants)
    assert A2_QUESTION_GUIDANCE.count(contract) == 1
    assert facade_module._STAGE_SYSTEM_PROMPTS["A2"].count(contract) == 1


def test_g_system_and_packet_share_relational_carrier_evidence_contract() -> None:
    """G packet and system guidance share one relational carrier owner."""

    contract = getattr(
        prompt_module,
        "G_RELATIONAL_CARRIER_EVIDENCE_GUIDANCE",
        "",
    ).strip()
    required_invariants = (
        "relational_willingness",
        "private_monologue",
        "当前交互角色或范围许可",
        "不能成为用户",
        "另行表达的`current_observation`关系事实",
    )

    assert contract
    assert "G_RELATIONAL_CARRIER_EVIDENCE_GUIDANCE" in prompt_module.__all__
    assert all(invariant in contract for invariant in required_invariants)
    assert GOAL_QUESTION_GUIDANCE.count(contract) == 1
    assert facade_module._STAGE_SYSTEM_PROMPTS["G"].count(contract) == 1
    for stage, guidance in (
        ("A1", A1_QUESTION_GUIDANCE),
        ("A2", A2_QUESTION_GUIDANCE),
        ("P", ORDINARY_PLAN_GUIDANCE),
    ):
        assert guidance.count(contract) == 0
        assert facade_module._STAGE_SYSTEM_PROMPTS[stage].count(contract) == 0


def test_g_relational_carrier_does_not_turn_unsupported_user_meaning_into_first_person_feeling() -> None:
    """Private monologue cannot launder unsupported user relationship meaning."""

    contract = getattr(
        prompt_module,
        "G_RELATIONAL_CARRIER_EVIDENCE_GUIDANCE",
        "",
    )
    required_invariants = (
        "未经当前关系事实支持的用户关系含义",
        "第一人称感受、内心判断或被动经历",
        "放入`private_monologue`",
        "获得依据",
    )

    assert contract
    assert all(invariant in contract for invariant in required_invariants)
