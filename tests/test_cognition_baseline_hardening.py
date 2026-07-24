"""Red contracts for proven baseline/V2 regression classifications."""

from __future__ import annotations

import pytest

from tests.cognition_baseline_worker import (
    _background_result_summary,
    _build_seeded_coding_task_document,
    _build_self_cognition_case,
    _delivery_text_is_grounded,
    _evaluate_hard_gates,
    _extract_final_cognition_monologue,
    _has_unavailable_evidence,
    _requires_live_monologue,
    _truthful_unavailable_coding_response,
    _truthful_unavailable_repository_outcome,
    _truthful_unavailable_scheduler_outcome,
    _trace_status_is_valid,
    _v2_requires_final_cognition_monologue,
)
from tests.cognition_core_v2_test_helpers import (
    canonical_cognition_output,
    canonical_episode,
)
from kazusa_ai_chatbot.cognition_core_v2.action_authorization import (
    derive_action_route,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface
from kazusa_ai_chatbot.self_cognition import models as self_cognition_models
from kazusa_ai_chatbot.self_cognition import worker as self_cognition_worker


def test_nested_settlement_monologue_and_trace_gates() -> None:
    """Nested V2 settlement evidence must satisfy the correct source gates."""

    graph_result = {
        "settlement": {
            "cognition_core_output": {
                "private_monologue": "我会先确认当前承诺，再决定是否行动。",
            },
        },
    }

    assert _extract_final_cognition_monologue(graph_result) == (
        "我会先确认当前承诺，再决定是否行动。"
    )
    assert _trace_status_is_valid(
        output_mode="private",
        trace_run={"status": "completed_private"},
    )
    assert not _trace_status_is_valid(
        output_mode="visible",
        trace_run={"status": "completed_private"},
    )
    assert _trace_status_is_valid(
        output_mode="visible",
        source_kind="scheduled_tick",
        trace_run={"status": "completed_visible"},
    )
    assert not _v2_requires_final_cognition_monologue("reflection")
    assert _v2_requires_final_cognition_monologue("self_cognition")


def test_silent_relevance_ends_before_monologue_contract() -> None:
    """A frontline silence result owns no downstream monologue surface."""

    assert not _requires_live_monologue(
        source_kind="user_message",
        output_mode="silent",
    )
    assert _requires_live_monologue(
        source_kind="user_message",
        output_mode="visible",
    )
    assert not _requires_live_monologue(
        source_kind="reflection",
        output_mode="private",
    )


def test_background_result_payload_uses_fixture_text() -> None:
    """The tool-result fixture must be the payload delivered to cognition."""

    fixture_case = {
        "input_text": "# 周报\n- 完成 A\n- 风险 B",
    }

    assert _background_result_summary(fixture_case) == fixture_case[
        "input_text"
    ]


def test_tool_result_grounding_accepts_semantic_overlap_and_rejects_ack_only() -> None:
    """Tool-result gates must accept paraphrase but reject an empty receipt."""

    expected_text = "# 周报\n- 完成 A\n- 风险 B"
    assert _delivery_text_is_grounded(
        expected_text,
        [{"text": "收到啦！A 搞定了，不过风险 B 需要确认具体情况。"}],
    )
    assert not _delivery_text_is_grounded(
        expected_text,
        [{"text": "收到啦，稍等我看看。"}],
    )


def test_scheduled_primary_bid_keeps_due_delivery_route() -> None:
    """A scheduled bid must reach the existing due-delivery route owner."""

    assert derive_action_route(
        episode={"trigger_source": "scheduled_tick"},
        primary_bid={"branch_id": "ordinary_response"},
        action_requests=[],
        resolver_requests=[],
    ) == "speech"


def test_scheduled_fixture_seeds_user_cognition_state() -> None:
    """The worker fixture must pass a consolidation-valid user profile."""

    case = _build_self_cognition_case(
        {
            "case": {
                "case_id": "O07",
                "source_kind": "scheduled_tick",
                "input_text": "到期的周报提醒。",
            },
            "fixed_local_timestamp": "2026-07-24 09:00:00",
        },
        profile={"name": "一之濑明日奈"},
    )

    user_profile = case["user_profile"]
    cognition_state = user_profile["cognition_state"]
    assert cognition_state["state_scope"] == "user"
    assert cognition_state["owner_user_id"] == "baseline-current-user"


def test_worker_v2_validator_accepts_canonical_user_scope() -> None:
    """The worker follows the scope selected by the cognition input contract."""

    artifact_payloads = {
        self_cognition_models.ARTIFACT_COGNITION_INPUT: {
            "state_scope": "user",
        },
        self_cognition_models.ARTIFACT_COGNITION_OUTPUT: {
            "cognition_core_output": {
                "state_update": {"state_scope": "user"},
            },
            "cognition_state_committed": True,
        },
    }

    self_cognition_worker._validate_worker_v2_cognition_result(
        artifact_payloads,
        required=True,
    )


def _surface_state_for_runtime_limit() -> dict[str, object]:
    """Build a committed surface state with a trusted runtime override."""

    return {
        "cognitive_episode": canonical_episode(
            episode_id="runtime-limit-episode",
            content="明天下午三点提醒我提交周报。",
        ),
        "cognition_core_output": canonical_cognition_output(),
        "action_results": [],
        "character_profile": {
            "name": "明日奈",
            "personality_brief": {
                "logic": "清晰",
                "tempo": "适中",
                "defense": "克制",
                "quirks": "温和",
                "taboos": "保持真实",
            },
            "linguistic_texture_profile": {
                "fragmentation": 0.4,
                "hesitation_density": 0.4,
                "counter_questioning": 0.4,
                "softener_density": 0.4,
                "formalism_avoidance": 0.4,
                "abstraction_reframing": 0.4,
                "direct_assertion": 0.4,
                "emotional_leakage": 0.4,
                "rhythmic_bounce": 0.4,
                "self_deprecation": 0.4,
            },
        },
        "action_availability_runtime": {
            "scheduler_status": "unavailable",
            "worker_status": {
                "accepted_task": "unavailable",
                "background_work": "unavailable",
            },
        },
    }


def test_unavailable_runtime_limit_reaches_surface_contract() -> None:
    """Trusted owner outages must be visible to the surface semantic owner."""

    payload = l3_surface.build_text_surface_input_from_global_state(
        _surface_state_for_runtime_limit(),
        interaction_style_context="自然简洁",
    )

    assert payload["runtime_capability_limits"]
    assert any(
        "不可用" in item
        for item in payload["runtime_capability_limits"]
    )


def test_unavailable_runtime_snapshot_is_evidence() -> None:
    """The hard gate recognizes the canonical runtime availability snapshot."""

    assert _has_unavailable_evidence({
        "action_availability_runtime": {
            "scheduler_status": "unavailable",
            "worker_status": {"background_work": "unavailable"},
        },
    })


def test_unavailable_scheduler_gate_requires_surface_limit_contract() -> None:
    """An outage snapshot alone cannot make a false promise pass."""

    runtime = {
        "action_availability_runtime": {
            "scheduler_status": "unavailable",
            "worker_status": {"orchestrator": "unavailable"},
        },
    }
    assert not _truthful_unavailable_scheduler_outcome(runtime)
    runtime["runtime_capability_limits"] = [
        "当前调度能力不可用，不能把未来提醒说成已经安排。",
    ]
    assert _truthful_unavailable_scheduler_outcome(runtime)


def test_unavailable_repository_owner_accepts_truthful_blocked_result() -> None:
    """Reader gates admit a typed unavailable-owner outcome."""

    value = {
        "graph_result": {
            "action_availability_runtime": {
                "worker_status": {
                    "accepted_task": "unavailable",
                    "background_work": "unavailable",
                },
            },
            "cognition_core_output": {
                "goal_resolution": "requires_user_input",
                "action_requests": [],
                "resolver_requests": [{
                    "capability": "human_clarification",
                }],
            },
            "consolidation_state": {
                "text_surface_output_v2": {
                    "runtime_capability_limits": [
                        "当前仓库代码读取 owner 不可用；请用户提供代码材料。",
                    ],
                },
            },
        },
    }

    assert _truthful_unavailable_repository_outcome(value)
    value["graph_result"]["cognition_core_output"]["resolver_requests"] = [{
        "capability": "public_answer_research",
    }]
    assert not _truthful_unavailable_repository_outcome(value)


def test_unavailable_coding_owner_satisfies_execution_gate_alternate() -> None:
    """C11 execution gates accept a typed unavailable coding owner outcome."""

    case = {
        "case_id": "C11",
        "input_text": "明日奈，请修改 README 并执行验证。",
        "output_mode": "visible",
        "hard_gates": [
            "accepted_coding_task_persisted",
            "coding_run_bound",
            "guarded_workspace_effect",
        ],
    }
    graph_result = {
        "action_availability_runtime": {
            "worker_status": {
                "accepted_task": "unavailable",
                "background_work": "unavailable",
                "orchestrator": "unavailable",
            },
            "coding_workspace_status": "healthy",
        },
        "cognition_core_output": {
            "goal_resolution": "requires_user_input",
            "action_requests": [],
            "resolver_requests": [{
                "capability": "human_clarification",
            }],
        },
        "consolidation_state": {
            "text_surface_output_v2": {
                "runtime_capability_limits": [
                    "当前仓库代码读取 owner 不可用；请用户提供代码材料。",
                ],
            },
        },
    }

    failures, results = _evaluate_hard_gates(
        {},
        case,
        response_payload={
            "messages": [
                "当前无法读取仓库，请提供 README 内容。",
            ],
        },
        monologue="当前 coding owner 不可用，我先说明限制。",
        monologue_path="response.cognition_graph.nodes.l2.reasoning.internal_monologue",
        graph_result=graph_result,
        persisted_profile=None,
        adapter_calls=[],
        counts_before={"accepted_tasks": 0, "background_work_jobs": 0},
        counts_after={"accepted_tasks": 0, "background_work_jobs": 0},
        workspace_before={"sha256": "empty"},
        workspace_after={"sha256": "empty"},
        expected_delivery_text="",
    )

    assert failures == []
    assert results == {
        "accepted_coding_task_persisted": True,
        "coding_run_bound": True,
        "guarded_workspace_effect": True,
    }

    graph_result["cognition_core_output"]["action_requests"] = [{
        "action_kind": "accepted_coding_task_request",
    }]
    failures, results = _evaluate_hard_gates(
        {},
        case,
        response_payload={
            "messages": [
                "当前无法读取仓库，请提供 README 内容。",
            ],
        },
        monologue="当前 coding owner 不可用，我先说明限制。",
        monologue_path="response.cognition_graph.nodes.l2.reasoning.internal_monologue",
        graph_result=graph_result,
        persisted_profile=None,
        adapter_calls=[],
        counts_before={"accepted_tasks": 0, "background_work_jobs": 0},
        counts_after={"accepted_tasks": 0, "background_work_jobs": 0},
        workspace_before={"sha256": "empty"},
        workspace_after={"sha256": "empty"},
        expected_delivery_text="",
    )

    assert failures == [
        "hard gate failed: accepted_coding_task_persisted",
        "hard gate failed: coding_run_bound",
        "hard gate failed: guarded_workspace_effect",
    ]
    assert results == {
        "accepted_coding_task_persisted": False,
        "coding_run_bound": False,
        "guarded_workspace_effect": False,
    }


def test_truthful_coding_limit_without_action_is_a_typed_alternate() -> None:
    """A truthful C11 limitation can answer the current turn without an effect."""

    value = {
        "graph_result": {
            "action_availability_runtime": {
                "worker_status": {
                    "accepted_task": "unavailable",
                    "background_work": "unavailable",
                    "orchestrator": "unavailable",
                },
            },
            "cognition_core_output": {
                "goal_resolution": "answerable_now",
                "action_requests": [],
                "resolver_requests": [],
                "intention": {
                    "reason": "当前仓库代码读取 owner 不可用。",
                },
            },
        },
        "response": {
            "messages": ["当前无法读取仓库，请提供 README 内容。"],
        },
    }

    assert _truthful_unavailable_coding_response(value)
    value["graph_result"]["cognition_core_output"]["action_requests"] = [{
        "action_kind": "accepted_coding_task_request",
    }]
    assert not _truthful_unavailable_coding_response(value)


def test_coding_state_seed_builds_scoped_context_and_rejects_incomplete_seed() -> None:
    """Coding lifecycle fixtures must materialize their stated precondition."""

    case = {
        "case_id": "C12",
        "input_text": "明日奈，刚才那个 README 修改任务现在是什么状态？",
        "state_seed": {
            "coding_run": {
                "run_id": "baseline-run-012",
                "status": "proposal_ready",
                "action_set": ["status", "cancel", "approve_and_verify"],
            },
        },
    }
    document = _build_seeded_coding_task_document(
        case=case,
        fixed_local_timestamp="2026-07-24 09:00:00",
        source_platform="debug",
        source_channel_id="baseline-C12",
        source_channel_type="group",
        source_message_id="C12-current",
        source_platform_bot_id="baseline-character-platform",
        source_character_name="一之濑明日奈",
        requester_global_user_id="baseline-current-user",
        requester_platform_user_id="baseline-current-user-platform",
        requester_display_name="基线测试用户",
    )

    assert document is not None
    assert document["source_channel_id"] == "baseline-C12"
    assert document["requester_global_user_id"] == "baseline-current-user"
    assert document["coding_run_context"] == {
        "schema_version": "coding_run_context.v1",
        "coding_run_ref": "coding_run:baseline-run-012",
        "status": "proposal_ready",
        "objective_summary": case["input_text"],
        "allowed_next_actions": ["status", "cancel", "approve_and_verify"],
        "active_blocker": None,
        "followup_open": True,
        "updated_at": "2026-07-23T21:00:00+00:00",
    }

    incomplete_case = {
        **case,
        "state_seed": {
            "coding_run": {
                "run_id": "baseline-run-incomplete",
                "status": "proposal_ready",
                "action_set": [],
            },
        },
    }
    with pytest.raises(ValueError, match="action_set"):
        _build_seeded_coding_task_document(
            case=incomplete_case,
            fixed_local_timestamp="2026-07-24 09:00:00",
            source_platform="debug",
            source_channel_id="baseline-C12",
            source_channel_type="group",
            source_message_id="C12-current",
            source_platform_bot_id="baseline-character-platform",
            source_character_name="一之濑明日奈",
            requester_global_user_id="baseline-current-user",
            requester_platform_user_id="baseline-current-user-platform",
            requester_display_name="基线测试用户",
        )


@pytest.mark.parametrize(
    ("case_id", "run_id", "status", "action_set", "open_blocker"),
    [
        (
            "C12",
            "baseline-run-012",
            "proposal_ready",
            ["status", "cancel", "approve_and_verify"],
            None,
        ),
        (
            "C13",
            "baseline-run-013",
            "blocked",
            ["respond_to_blocker"],
            "是否可使用现有虚拟环境运行聚焦测试？",
        ),
        (
            "C14",
            "baseline-run-014",
            "awaiting_approval",
            ["approve_and_verify"],
            None,
        ),
        (
            "C15",
            "baseline-run-015",
            "proposal_ready",
            ["cancel"],
            None,
        ),
    ],
)
def test_coding_lifecycle_seeds_materialize_for_every_declared_run(
    case_id: str,
    run_id: str,
    status: str,
    action_set: list[str],
    open_blocker: str | None,
) -> None:
    """Every C12-C15 declared run reaches the live DB precondition."""

    coding_run = {
        "run_id": run_id,
        "status": status,
        "action_set": action_set,
    }
    if open_blocker is not None:
        coding_run["open_blocker"] = open_blocker
    document = _build_seeded_coding_task_document(
        case={
            "case_id": case_id,
            "input_text": f"{case_id} coding lifecycle fixture",
            "state_seed": {"coding_run": coding_run},
        },
        fixed_local_timestamp="2026-07-24 09:00:00",
        source_platform="debug",
        source_channel_id=f"baseline-{case_id}",
        source_channel_type="group",
        source_message_id=f"{case_id}-current",
        source_platform_bot_id="baseline-character-platform",
        source_character_name="一之濑明日奈",
        requester_global_user_id="baseline-current-user",
        requester_platform_user_id="baseline-current-user-platform",
        requester_display_name="基线测试用户",
    )

    assert document is not None
    context = document["coding_run_context"]
    assert context["coding_run_ref"] == f"coding_run:{run_id}"
    assert context["status"] == status
    assert context["allowed_next_actions"] == action_set
    if open_blocker is None:
        assert context["active_blocker"] is None
    else:
        assert context["active_blocker"]["question"] == open_blocker


def test_disabled_scheduler_reclassifies_schedule_gate_only_with_truthful_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The disabled test owner accepts a truthful limitation, not a promise."""

    monkeypatch.setenv("CALENDAR_SCHEDULER_ENABLED", "false")
    case = {
        "case_id": "C16",
        "input_text": "明日奈，明天下午三点提醒我提交周报。",
        "output_mode": "visible",
        "hard_gates": ["schedule_once", "schedule_time_exact"],
    }
    input_payload = {
        "fixed_scheduled_local_timestamp": "2026-07-25 15:00:00",
    }
    truthful_graph = {
        "action_availability_runtime": {
            "scheduler_status": "unavailable",
            "worker_status": {"orchestrator": "unavailable"},
        },
        "runtime_capability_limits": [
            "当前调度能力不可用，不能把未来提醒说成已经安排。",
        ],
    }
    failures, gates = _evaluate_hard_gates(
        input_payload,
        case,
        response_payload={
            "messages": ["明天下午三点这件事你记得自己设个闹钟哦。"],
        },
        monologue="我先确认时间，再如实说明当前限制。",
        monologue_path="cognition_graph.nodes.l2.reasoning.internal_monologue",
        graph_result=truthful_graph,
        persisted_profile=None,
        adapter_calls=[],
        counts_before={"calendar_schedules": 0},
        counts_after={"calendar_schedules": 0},
        workspace_before={},
        workspace_after={},
        expected_delivery_text="",
    )
    assert failures == []
    assert gates == {
        "schedule_once": True,
        "schedule_time_exact": True,
    }

    failures, gates = _evaluate_hard_gates(
        input_payload,
        case,
        response_payload={
            "messages": ["明天下午三点我会准时提醒你。"],
        },
        monologue="我会答应下来。",
        monologue_path="cognition_graph.nodes.l2.reasoning.internal_monologue",
        graph_result={},
        persisted_profile=None,
        adapter_calls=[],
        counts_before={"calendar_schedules": 1},
        counts_after={"calendar_schedules": 1},
        workspace_before={},
        workspace_after={},
        expected_delivery_text="",
    )
    assert gates == {
        "schedule_once": False,
        "schedule_time_exact": False,
    }
    assert failures == [
        "hard gate failed: schedule_once",
        "hard gate failed: schedule_time_exact",
    ]


def test_disabled_scheduler_rejects_generic_delayed_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truthful limitation cannot hide an unrelated queued task."""

    monkeypatch.setenv("CALENDAR_SCHEDULER_ENABLED", "false")
    case = {
        "case_id": "C20",
        "input_text": "明日奈，明天下午三点提醒我提交周报。",
        "output_mode": "visible",
        "hard_gates": ["no_unowned_delayed_side_effect"],
    }
    failures, gates = _evaluate_hard_gates(
        {
            "fixed_scheduled_local_timestamp": "2026-07-25 15:00:00",
        },
        case,
        response_payload={"messages": ["当前调度能力不可用。"]},
        monologue="我说明当前能力边界。",
        monologue_path="cognition_graph.nodes.l2.reasoning.internal_monologue",
        graph_result={},
        persisted_profile=None,
        adapter_calls=[],
        counts_before={
            "accepted_tasks": 0,
            "background_work_jobs": 0,
        },
        counts_after={
            "accepted_tasks": 1,
            "background_work_jobs": 1,
        },
        workspace_before={},
        workspace_after={},
        expected_delivery_text="",
    )

    assert failures == [
        "hard gate failed: no_unowned_delayed_side_effect",
    ]
    assert gates == {"no_unowned_delayed_side_effect": False}
