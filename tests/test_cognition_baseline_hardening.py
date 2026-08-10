"""Red contracts for proven baseline/V2 regression classifications."""

from __future__ import annotations

from copy import deepcopy

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
    _truthful_unavailable_scheduler_outcome,
    _trace_status_is_valid,
    _v2_requires_final_cognition_monologue,
)
from tests.cognition_core_v2_test_helpers import (
    canonical_cognition_output,
    canonical_episode,
    canonical_service_character_profile,
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

    character_profile = canonical_service_character_profile(
        marker="runtime-limit",
    )
    character_profile.update({
        "name": "明日奈",
        "personality_brief": {
            "mbti": "ISTP",
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
    })
    return {
        "cognitive_episode": canonical_episode(
            episode_id="runtime-limit-episode",
            content="明天下午三点提醒我提交周报。",
        ),
        "cognition_core_output": canonical_cognition_output(),
        "action_results": [],
        "character_profile": character_profile,
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


def test_unavailable_coding_owner_fails_effect_gates() -> None:
    """An unavailable owner is diagnostic evidence, not execution evidence."""

    case = {
        "case_id": "C11",
        "input_text": "明日奈，请修改 README 并执行验证。",
        "output_mode": "visible",
        "hard_gates": [
            "accepted_coding_task_persisted",
            "coding_run_bound",
            "guarded_workspace_effect",
            "repository_map_evidence",
            "coding_reader_route",
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

    assert failures == [
        "hard gate failed: accepted_coding_task_persisted",
        "hard gate failed: coding_run_bound",
        "hard gate failed: guarded_workspace_effect",
        "hard gate failed: repository_map_evidence",
        "hard gate failed: coding_reader_route",
    ]
    assert results == {
        "accepted_coding_task_persisted": False,
        "coding_run_bound": False,
        "guarded_workspace_effect": False,
        "repository_map_evidence": False,
        "coding_reader_route": False,
    }


def test_failed_worker_summary_does_not_satisfy_terminal_gate() -> None:
    """A failure summary cannot masquerade as a terminal successful result."""

    case = {
        "output_mode": "visible",
        "hard_gates": ["terminal_result"],
    }
    common_args = {
        "input_payload": {},
        "case": case,
        "response_payload": {"messages": ["The coding worker failed."]},
        "monologue": "The worker returned a failure.",
        "monologue_path": "internal_monologue",
        "persisted_profile": None,
        "adapter_calls": [],
        "counts_before": {},
        "counts_after": {},
        "workspace_before": {},
        "workspace_after": {},
        "expected_delivery_text": "",
    }

    failures, results = _evaluate_hard_gates(
        **common_args,
        graph_result={
            "action_results": [{
                "status": "failed",
                "result_summary": "Selected worker unavailable.",
            }],
        },
    )

    assert failures == ["hard gate failed: terminal_result"]
    assert results == {"terminal_result": False}

    failures, results = _evaluate_hard_gates(
        **common_args,
        graph_result={
            "action_results": [{
                "status": "succeeded",
                "result_summary": "Coding worker completed the action.",
            }],
        },
    )

    assert failures == []
    assert results == {"terminal_result": True}


def _valid_c07_handover_graph() -> dict[str, object]:
    """Build one fully correlated C07 task, worker, and delivery result."""

    return {
        "background_handover": {
            "runtime_ticks": [{
                "processed_count": 1,
                "succeeded_count": 1,
                "failed_count": 0,
                "delivery_delivered_count": 1,
                "delivery_failed_count": 0,
            }],
            "jobs_before": [],
            "jobs_after": [{
                "job_id": "job-1",
                "accepted_task_id": "task-1",
                "source_message_id": "C07-current",
                "semantic_objective": (
                    "Review https://github.com/eamars/KazusaAIChatbot"
                ),
                "attempt_count": 1,
                "delivery_attempt_count": 1,
                "status": "delivered",
                "delivery_tracking_id": "background-delivery-1",
                "delivered_conversation_message_id": "conversation-row-1",
                "requested_worker": "task_orchestrator",
                "worker_payload": {
                    "schema_version": "task_orchestrator_worker_payload.v1",
                    "operation": "resume_task_resolution",
                    "checkpoint": {},
                    "coding_request": None,
                },
                "failure_summary": "",
                "result_summary": "The coding specialist resolved the task.",
                "artifact_text": "The repository uses a staged cognition graph.",
                "task_resolution_result": {
                    "schema_version": "task_resolution_result.v1",
                    "status": "resolved",
                    "prompt_safe_summary": (
                        "The repository uses a staged cognition graph."
                    ),
                    "evidence": [{
                        "schema_version": "task_resolution_evidence.v1",
                        "evidence_id": "coding-evidence-1",
                        "task_node_id": "node-1",
                        "specialist": "coding",
                        "summary": "The repository uses a staged cognition graph.",
                        "provenance_refs": ["coding_run:run-1"],
                        "limitations": [],
                    }],
                    "completed_subgoals": ["Review the repository."],
                    "remaining_needs": [],
                    "checkpoint": {},
                    "coding_run_context": {
                        "schema_version": "coding_run_context.v1",
                        "coding_run_ref": "coding_run:run-1",
                        "status": "completed",
                        "summary": "Repository analysis completed.",
                        "limitations": [],
                        "allowed_next_actions": [],
                        "followup_open": False,
                    },
                },
            }],
            "accepted_tasks_before": [],
            "accepted_tasks_after": [{
                "accepted_task_id": "task-1",
                "executor_ref": "job-1",
                "task_kind": "task_resolution",
                "first_source_message_id": "C07-current",
                "state": "delivered",
                "delivery_tracking_id": "background-delivery-1",
                "delivered_conversation_message_id": "conversation-row-1",
            }],
            "conversation_rows_before": [],
            "conversation_rows_after": [{
                "row_id": "conversation-row-1",
                "role": "assistant",
                "body_text": "Here is the repository review.",
                "delivery_tracking_id": "dispatcher-delivery-1",
                "delivery_status": "delivered",
                "platform_message_id": "adapter-message-1",
                "platform_channel_id": "baseline-C07",
            }],
            "delivery_adapter_calls": [{
                "message_id": "adapter-message-1",
                "text": "Here is the repository review.",
                "channel_id": "baseline-C07",
            }],
            "delivery_graph_results": [{
                "cognitive_episode": {
                    "percepts": [{
                        "content": {
                            "result": {"task_id": "task-1"},
                        },
                    }],
                },
                "action_results": [{
                    "action_attempt_id": "speak-attempt-1",
                    "action_kind": "speak",
                    "status": "executed",
                }],
                "surface_outputs": [{
                    "schema_version": "surface_output.v1",
                    "surface_kind": "text",
                    "visibility": "user_visible",
                    "action_attempt_id": "speak-attempt-1",
                    "fragments": ["Here is the repository review."],
                }],
                "episode_trace": {
                    "delivery_correlation": {
                        "tracking_id": "dispatcher-delivery-1",
                    },
                },
            }],
        },
    }


def _c07_gate_case() -> dict[str, object]:
    """Return C07's exact dispatch and delivery gate list."""

    return {
        "case_id": "C07",
        "output_mode": "visible",
        "hard_gates": [
            "visible_dialog",
            "accepted_task_persisted",
            "c07_exact_handover",
            "coding_reader_route",
            "repository_map_evidence",
            "terminal_result",
            "result_speak_called",
            "one_authorized_delivery",
        ],
    }


def _c07_gate_args() -> dict[str, object]:
    """Return common visible and persistence evidence for C07 gate tests."""

    return {
        "input_payload": {},
        "case": _c07_gate_case(),
        "response_payload": {"messages": ["I accepted the repository review."]},
        "monologue": "I will delegate the repository reading.",
        "monologue_path": "internal_monologue",
        "persisted_profile": None,
        "adapter_calls": [{"text": "Here is the repository review."}],
        "counts_before": {"accepted_tasks": 0},
        "counts_after": {"accepted_tasks": 1},
        "workspace_before": {},
        "workspace_after": {},
        "expected_delivery_text": "",
    }


def test_c07_execution_gates_require_structured_coding_reader_evidence() -> None:
    """C07 passes only on one correlated coding read and result delivery."""

    failures, results = _evaluate_hard_gates(
        **_c07_gate_args(),
        graph_result=_valid_c07_handover_graph(),
    )

    assert failures == []
    assert all(results.values())


@pytest.mark.parametrize(
    ("failure_mode", "failed_gates"),
    [
        ("duplicate_job", ("c07_exact_handover",)),
        ("wrong_specialist", ("repository_map_evidence",)),
        ("empty_evidence", ("repository_map_evidence",)),
        ("unrelated_speech", ("result_speak_called",)),
        ("mismatched_delivery_id", ("c07_exact_handover",)),
        (
            "blocked_partial_read",
            (
                "coding_reader_route",
                "repository_map_evidence",
                "terminal_result",
            ),
        ),
        ("mismatched_surface_attempt", ("result_speak_called",)),
    ],
)
def test_c07_execution_gates_reject_uncorrelated_or_incomplete_evidence(
    failure_mode: str,
    failed_gates: tuple[str, ...],
) -> None:
    """Every reviewed C07 false-pass shape must remain a failed gate."""

    graph_result = _valid_c07_handover_graph()
    handover = graph_result["background_handover"]
    assert isinstance(handover, dict)
    if failure_mode == "duplicate_job":
        duplicate = deepcopy(handover["jobs_after"][0])
        duplicate["job_id"] = "job-2"
        handover["jobs_after"].append(duplicate)
    elif failure_mode == "wrong_specialist":
        handover["jobs_after"][0]["task_resolution_result"]["evidence"][0][
            "specialist"
        ] = "public_research"
    elif failure_mode == "empty_evidence":
        handover["jobs_after"][0]["task_resolution_result"]["evidence"] = []
    elif failure_mode == "unrelated_speech":
        handover["delivery_graph_results"][0]["cognitive_episode"] = {
            "percepts": [{
                "content": {"result": {"task_id": "unrelated-task"}},
            }],
        }
    elif failure_mode == "mismatched_delivery_id":
        handover["accepted_tasks_after"][0][
            "delivery_tracking_id"
        ] = "other-background-delivery"
    elif failure_mode == "blocked_partial_read":
        result = handover["jobs_after"][0]["task_resolution_result"]
        result["status"] = "needs_user_input"
        result["coding_run_context"]["status"] = "blocked"
    elif failure_mode == "mismatched_surface_attempt":
        handover["delivery_graph_results"][0]["surface_outputs"][0][
            "action_attempt_id"
        ] = "other-speak-attempt"
    else:
        raise AssertionError(f"unsupported failure mode: {failure_mode}")

    failures, results = _evaluate_hard_gates(
        **_c07_gate_args(),
        graph_result=graph_result,
    )

    for failed_gate in failed_gates:
        assert results[failed_gate] is False
        assert f"hard gate failed: {failed_gate}" in failures


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
