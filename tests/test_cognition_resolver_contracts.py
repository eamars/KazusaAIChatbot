"""Tests for cognition resolver structural contracts."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    MAX_RESOLVER_SUMMARY_CHARS,
    MAX_RESOLVER_TRACE_CHARS,
    PENDING_TASK_CONTINUATION_VERSION,
    REQUIRED_RESOLVER_EVIDENCE_DEPENDENCY_VERSION,
    RESOLVER_CAPABILITY_REQUEST_VERSION,
    RESOLVER_CYCLE_STATE_VERSION,
    RESOLVER_GOAL_PROGRESS_VERSION,
    RESOLVER_PENDING_RESOLUTION_VERSION,
    RESOLVER_PENDING_RESUME_VERSION,
    ResolverValidationError,
    new_empty_goal_progress,
    project_goal_progress_for_cognition,
    project_observations_for_cognition,
    project_pending_resume_for_cognition,
    validate_pending_task_continuation,
    validate_required_resolver_evidence_dependency,
    validate_resolver_capability_request,
    validate_resolver_cycle_trace,
    validate_resolver_goal_progress,
    validate_resolver_observation,
    validate_resolver_pending_resolution,
    validate_resolver_pending_resume,
)
from kazusa_ai_chatbot.cognition_resolver.state import (
    MAX_PROJECTED_RESOLVER_OBSERVATIONS,
    append_cycle_trace,
    append_observation,
    build_empty_rag_result,
    ensure_initial_resolver_inputs,
    new_resolver_state,
    project_resolver_context,
    required_task_observation,
    validate_resolver_state,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.cognition_test_helpers import canonical_user_message_episode
from tests.task_resolution_test_helpers import resolver_task_observation


def _goal_continuation_ref() -> dict:
    return build_goal_continuation_ref(
        source_episode_id="resolver-test-episode",
        source_message_id="resolver-test-message",
        branch_id="task_resolution",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "resolver-test-goal",
        },
    )


def _capability_request() -> dict:
    return {
        "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
        "capability_kind": "task_resolution_request",
        "objective": "Retrieve relationship evidence for the current question.",
        "reason": "The current cognition cycle lacks enough evidence.",
        "priority": "now",
        "goal_continuation_ref": _goal_continuation_ref(),
    }




def _rag_observation() -> dict:
    observation = resolver_task_observation()
    observation["request_objective"] = "raw-user-id-should-stay-out"
    observation["rag_result"] = {
        "answer": "RAG prompt-safe answer with evidence.",
        "external_evidence": [
            {
                "summary": (
                    "CBD walking route evidence includes Wynyard Quarter "
                    "and Britomart."
                ),
                "raw_id": "raw-external-id-321",
            },
        ],
        "supervisor_trace": {
            "known_facts": [
                {"summary": "prompt-safe fact summary"},
            ],
            "raw_id": "raw-rag-id-789",
        },
    }
    return observation


def _cycle_trace() -> dict:
    return {
        "schema_version": "resolver_cycle_trace.v1",
        "cycle_index": 0,
        "status_before_cycle": "running",
        "l1_emotional_appraisal": "calm",
        "l1_interaction_subtext": "routine request",
        "l2_internal_monologue_summary": "Needs evidence before answering.",
        "l2_logical_stance": "TENTATIVE",
        "l2_character_intent": "CLARIFY",
        "l2_judgment_note": "Evidence is missing.",
        "l2d_resolver_capability_requests": [_capability_request()],
        "l2d_action_specs_summary": ["speak:" + ("x" * 700)],
        "selected_capability_kind": "task_resolution_request",
        "observation_ids": ["resolver_obs_1"],
        "final_surface_decision": "continue",
        "terminal_reason": "",
        "created_at_utc": "2026-05-30T00:00:00+00:00",
    }


def _pending_resume() -> dict:
    return {
        "schema_version": RESOLVER_PENDING_RESUME_VERSION,
        "resume_id": "resolver-pending-001",
        "capability_kind": "human_clarification",
        "status": "waiting_for_user",
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "global_user_id": "user-1",
        "source_message_id": "message-1",
        "prompt_safe_original_goal": "Plan a low-cost evening after location.",
        "prompt_safe_question": "Which city are you in?",
        "prompt_safe_approval_summary": "",
        "pending_task_continuation": {
            "schema_version": PENDING_TASK_CONTINUATION_VERSION,
            "on_answered_clarification": "background_task_admission",
        },
        "created_at_utc": "2026-05-30T00:00:00+00:00",
        "expires_at_utc": "2026-05-31T00:00:00+00:00",
    }


def _pending_resolution() -> dict:
    return {
        "schema_version": RESOLVER_PENDING_RESOLUTION_VERSION,
        "resume_id": "resolver-pending-001",
        "decision": "answered",
        "reason": "The user supplied the missing city.",
    }


def _goal_progress() -> dict:
    return {
        "schema_version": RESOLVER_GOAL_PROGRESS_VERSION,
        "original_goal": "帮我安排一个两小时的低预算晚间计划。",
        "current_focus": "用户已补充城市和预算，需要完成最终计划。",
        "deliverables": [
            {
                "description": "晚餐候选和证据边界",
                "status": "partial",
                "note": "已有候选类别，但营业状态未确认。",
            },
            {
                "description": "两小时步行路线和时间切分",
                "status": "pending",
                "note": "最终回答仍必须覆盖。",
            },
        ],
        "missing_user_inputs": [],
        "evidence_dependencies": ["当前营业状态和路线锚点"],
        "attempted_paths": ["task_resolution_request: CBD 平价晚餐"],
        "source_backed_facts": ["用户预算 20 NZD；地点奥克兰 CBD"],
        "assumptions_or_inferences": ["散步路线可以用公开海滨路线骨架给出"],
        "blockers": ["无法确认每家店 19:30 仍营业"],
        "final_response_requirements": [
            "区分已确认约束、未确认营业事实和最佳努力路线",
            "给出晚餐加散步的两小时安排",
        ],
    }


def _minimal_global_state() -> dict:
    return {
        "decontextualized_input": "User asks for evidence-backed judgment.",
        "global_user_id": "user-1",
        "character_profile": {"global_user_id": "character-1"},
        "cognitive_episode": {"episode_id": "resolver-test-episode"},
    }


def _minimal_global_state_with_media(
    media_description_rows: list[dict],
) -> dict:
    """Build an empty-text resolver state with current-turn media percepts."""

    turn_clock = build_turn_clock("2026-06-02 16:04:32")
    episode = canonical_user_message_episode(
        episode_id="resolver-image-only-episode",
        percept_id="resolver-image-only-percept",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="",
        platform="debug",
        platform_channel_id="channel-image-only",
        channel_type="private",
        platform_message_id="message-image-only",
        platform_user_id="platform-user-image-only",
        global_user_id="user-1",
        user_name="Image User",
        active_turn_platform_message_ids=["message-image-only"],
        active_turn_conversation_row_ids=["row-image-only"],
        debug_modes={},
        target_addressed_user_ids=["character-1"],
        target_broadcast=False,
        media_description_rows=media_description_rows,
    )
    state = _minimal_global_state()
    state["decontextualized_input"] = ""
    state["cognitive_episode"] = episode
    return state


def _first_image_observation_percept(state: dict) -> dict:
    episode = state["cognitive_episode"]
    for percept in episode["percepts"]:
        if percept["source_kind"] == "image_observation":
            return percept
    raise AssertionError("expected image observation percept")




def test_capability_request_validator_rejects_unknown_kind() -> None:
    """Capability kinds must stay inside the reviewed resolver roster."""

    request = _capability_request()
    request["capability_kind"] = "shell_command"

    with pytest.raises(ResolverValidationError, match="capability_kind"):
        validate_resolver_capability_request(request)


def test_capability_request_validator_rejects_empty_objective() -> None:
    """Capability requests need a semantic objective from cognition."""

    request = _capability_request()
    request["objective"] = " "

    with pytest.raises(ResolverValidationError, match="objective"):
        validate_resolver_capability_request(request)


def test_observation_validator_clips_prompt_safe_summary() -> None:
    """Long observations should be clipped before they enter cognition."""

    observation = resolver_task_observation()
    observation["prompt_safe_summary"] = "x" * (MAX_RESOLVER_SUMMARY_CHARS + 50)

    validated = validate_resolver_observation(observation)

    assert len(validated["prompt_safe_summary"]) == MAX_RESOLVER_SUMMARY_CHARS
    assert set(validated["prompt_safe_summary"]) == {"x"}




def test_observation_projection_hides_raw_ids() -> None:
    """Cognition projection should expose aliases and summaries, not raw ids."""

    projection = project_observations_for_cognition([_rag_observation()])

    assert "resolver_obs_1" in projection
    assert "Found two relevant relationship evidence rows." in projection
    assert "RAG prompt-safe answer with evidence." in projection
    assert "CBD walking route evidence includes Wynyard Quarter" in projection
    assert "raw-tool-run-123" not in projection
    assert "raw-evidence-row-456" not in projection
    assert "raw-user-id-should-stay-out" not in projection
    assert "raw-rag-id-789" not in projection
    assert "raw-external-id-321" not in projection




def test_validators_strip_unknown_fields() -> None:
    """Validation should not preserve raw handler metadata fields."""

    observation = resolver_task_observation()
    observation["raw_handler_payload"] = {"secret_id": "raw-secret"}
    pending = _pending_resume()
    pending["raw_scope"] = {"platform_user_id": "raw-user"}
    resolution = _pending_resolution()
    resolution["raw_model_payload"] = "raw-json"

    validated_observation = validate_resolver_observation(observation)
    validated_pending = validate_resolver_pending_resume(pending)
    validated_resolution = validate_resolver_pending_resolution(resolution)

    assert "raw_handler_payload" not in validated_observation
    assert "raw_scope" not in validated_pending
    assert "raw_model_payload" not in validated_resolution


def test_cycle_trace_clips_nested_requests_and_action_summaries() -> None:
    """Trace rows must stay bounded before telemetry or artifacts consume them."""

    trace = _cycle_trace()
    trace["l2d_resolver_capability_requests"][0]["objective"] = "y" * 700

    validated = validate_resolver_cycle_trace(trace)

    request = validated["l2d_resolver_capability_requests"][0]
    assert len(request["objective"]) == 400
    summary = validated["l2d_action_specs_summary"][0]
    assert len(summary) == MAX_RESOLVER_SUMMARY_CHARS


def test_pending_resume_validator_and_projection_are_prompt_safe() -> None:
    """Pending user-owned blockers should project scope-free prompt text."""

    raw_pending = _pending_resume()
    raw_pending["prompt_safe_goal_progress"] = _goal_progress()
    pending = validate_resolver_pending_resume(raw_pending)
    projection = project_pending_resume_for_cognition(pending)

    assert pending["status"] == "waiting_for_user"
    assert pending["prompt_safe_goal_progress"]["deliverables"][1][
        "description"
    ] == "两小时步行路线和时间切分"
    assert "Plan a low-cost evening after location." in projection
    assert "Which city are you in?" in projection
    assert "两小时步行路线和时间切分" not in projection
    assert "resume_id" not in projection
    assert "expires_at_utc" not in projection
    assert "channel-1" not in projection
    assert "user-1" not in projection


def test_pending_task_continuation_validator_requires_exact_v1_shape() -> None:
    """The answer-conditioned admission decision has no timing fallback shape."""

    valid = {
        "schema_version": PENDING_TASK_CONTINUATION_VERSION,
        "on_answered_clarification": "background_task_admission",
    }

    assert validate_pending_task_continuation(valid) == valid

    invalid = {
        "schema_version": PENDING_TASK_CONTINUATION_VERSION,
        "value": "background",
    }
    with pytest.raises(ResolverValidationError, match="fields are not exact"):
        validate_pending_task_continuation(invalid)

    invalid["on_answered_clarification"] = "background"
    invalid.pop("value")
    with pytest.raises(
        ResolverValidationError,
        match="on_answered_clarification: expected one of",
    ):
        validate_pending_task_continuation(invalid)


def _resolver_state_with_required_observation() -> dict:
    """Build one state whose V2 dependency references its sole observation."""

    observation = resolver_task_observation()
    state = new_resolver_state(
        decontextualized_input="Retrieve relationship evidence.",
        max_cycles=3,
        episode_id="resolver-test-episode",
    )
    state["observations"] = [observation]
    state["required_resolver_evidence_dependency"] = {
        "schema_version": REQUIRED_RESOLVER_EVIDENCE_DEPENDENCY_VERSION,
        "accepted_request_handle": "resolver_request_0_1",
        "observation_id": observation["observation_id"],
    }
    return state


def test_required_evidence_dependency_v2_accepts_reference_only() -> None:
    """The dependency contains identity only and owns no evidence semantics."""

    dependency = validate_required_resolver_evidence_dependency({
        "schema_version": REQUIRED_RESOLVER_EVIDENCE_DEPENDENCY_VERSION,
        "accepted_request_handle": "resolver_request_0_1",
        "observation_id": "resolver-observation-1",
    })

    assert dependency == {
        "schema_version": "required_resolver_evidence_dependency.v2",
        "accepted_request_handle": "resolver_request_0_1",
        "observation_id": "resolver-observation-1",
    }


def test_required_evidence_dependency_v2_rejects_legacy_copied_fields() -> None:
    """The big-bang V2 boundary rejects every legacy semantic snapshot."""

    dependency = {
        "schema_version": REQUIRED_RESOLVER_EVIDENCE_DEPENDENCY_VERSION,
        "accepted_request_handle": "resolver_request_0_1",
        "observation_id": "resolver-observation-1",
        "state": "complete",
    }

    with pytest.raises(ResolverValidationError, match="fields are not exact"):
        validate_required_resolver_evidence_dependency(dependency)


def test_required_evidence_dependency_resolves_one_task_observation() -> None:
    """State and prompt projection derive evidence from the referenced row."""

    state = validate_resolver_state(_resolver_state_with_required_observation())

    observation = required_task_observation(state)
    projection = project_resolver_context(state)

    assert observation == state["observations"][0]
    assert "observation_handle=resolver_observation_raw-tool-run-123" in projection
    assert "state=complete" in projection
    assert "evidence_handles=resolver_evidence_raw-tool-run-123_1" in projection


def test_required_evidence_dependency_rejects_missing_or_wrong_kind_observation(
) -> None:
    """A dependency cannot name an absent or non-task observation."""

    state = _resolver_state_with_required_observation()
    state["required_resolver_evidence_dependency"]["observation_id"] = "missing"

    with pytest.raises(ResolverValidationError, match="unavailable"):
        validate_resolver_state(state)

    wrong_kind_observation = resolver_task_observation()
    wrong_kind_observation["capability_kind"] = "human_clarification"
    wrong_kind_observation.pop("task_resolution_evidence_state")
    wrong_kind_observation.pop("goal_continuation_ref")
    state["observations"] = [wrong_kind_observation]
    state["required_resolver_evidence_dependency"]["observation_id"] = (
        wrong_kind_observation["observation_id"]
    )

    with pytest.raises(ResolverValidationError, match="wrong capability"):
        validate_resolver_state(state)


def test_pending_resume_v1_and_v2_fail_closed_without_continuation_fallback() -> None:
    """Earlier pending schemas cannot enter the V3 clarification lane."""

    legacy_pending = _pending_resume()
    legacy_pending["schema_version"] = "resolver_pending_resume.v1"

    with pytest.raises(ResolverValidationError, match="schema_version"):
        validate_resolver_pending_resume(legacy_pending)

    legacy_pending["schema_version"] = "resolver_pending_resume.v2"
    with pytest.raises(ResolverValidationError, match="schema_version"):
        validate_resolver_pending_resume(legacy_pending)


def test_pending_resolution_validator_accepts_cognition_decision() -> None:
    """Pending-row closure is driven by L2d's structural decision."""

    validated = validate_resolver_pending_resolution(_pending_resolution())

    assert validated["decision"] == "answered"
    assert validated["resume_id"] == "resolver-pending-001"


def test_goal_progress_validator_and_projection_preserve_deliverables() -> None:
    """Goal progress should carry the user-goal checklist into cognition."""

    validated = validate_resolver_goal_progress(_goal_progress())
    projection = project_goal_progress_for_cognition(validated)

    assert validated["schema_version"] == RESOLVER_GOAL_PROGRESS_VERSION
    assert validated["deliverables"][0]["status"] == "partial"
    assert "晚餐候选和证据边界" in projection
    assert "两小时步行路线和时间切分" in projection
    assert "final_response_requirements" in projection


def test_empty_goal_progress_shell_has_no_python_deliverable_guess() -> None:
    """Deterministic initialization may store the goal, not infer deliverables."""

    progress = new_empty_goal_progress(original_goal="帮我做一个复杂计划。")

    assert progress["original_goal"] == "帮我做一个复杂计划。"
    assert progress["deliverables"] == []
    assert progress["final_response_requirements"] == []


def test_new_resolver_state_initializes_cycle_zero() -> None:
    """A new resolver state should be empty and ready for cycle 0."""

    state = new_resolver_state(
        decontextualized_input="Need a deliberate answer.",
        max_cycles=3,
        episode_id="resolver-test-episode",
    )

    assert state["schema_version"] == RESOLVER_CYCLE_STATE_VERSION
    assert state["cycle_index"] == 0
    assert state["max_cycles"] == 3
    assert state["status"] == "running"
    assert state["observations"] == []
    assert state["cycle_traces"] == []
    assert state["held_action_specs"] == []
    assert state["goal_progress"]["original_goal"] == "Need a deliberate answer."
    assert state["goal_progress"]["deliverables"] == []
    assert "pending_resume" not in state


def test_append_observation_projects_alias_and_caps_context() -> None:
    """Observation projection should expose bounded aliases, not raw ids."""

    state = new_resolver_state(
        decontextualized_input="Need repeated evidence.",
        max_cycles=3,
        episode_id="resolver-test-episode",
    )
    for index in range(MAX_PROJECTED_RESOLVER_OBSERVATIONS + 2):
        observation = resolver_task_observation()
        observation["observation_id"] = f"raw-tool-run-{index}"
        observation["prompt_safe_summary"] = f"summary {index}"
        state = append_observation(state, observation)

    context = project_resolver_context(state)

    assert "resolver_obs_1" in context
    assert context.count("resolver_obs_") == MAX_PROJECTED_RESOLVER_OBSERVATIONS
    assert "summary 0" not in context
    assert f"summary {MAX_PROJECTED_RESOLVER_OBSERVATIONS + 1}" in context
    assert "raw-tool-run-" not in context


def test_append_cycle_trace_stores_bounded_trace_row() -> None:
    """Cycle traces should be normalized before they enter resolver state."""

    state = new_resolver_state(
        decontextualized_input="Need one resolver cycle.",
        max_cycles=3,
        episode_id="resolver-test-episode",
    )
    trace = _cycle_trace()
    trace["terminal_reason"] = "x" * (MAX_RESOLVER_TRACE_CHARS + 50)

    updated = append_cycle_trace(state, trace)

    assert updated["cycle_index"] == 1
    assert len(updated["cycle_traces"]) == 1
    stored_trace = updated["cycle_traces"][0]
    assert len(stored_trace["terminal_reason"]) == MAX_RESOLVER_TRACE_CHARS


def test_build_empty_rag_result_uses_existing_projection_shape() -> None:
    """The first resolver cycle needs a normal empty RAG payload."""

    rag_result = build_empty_rag_result(
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert rag_result["answer"] == ""
    assert rag_result["memory_evidence"] == []
    assert rag_result["recall_evidence"] == []
    assert rag_result["conversation_evidence"] == []
    assert rag_result["external_evidence"] == []
    assert rag_result["supervisor_trace"]["loop_count"] == 0
    assert "user_memory_context" in rag_result["user_image"]


def test_ensure_initial_resolver_inputs_adds_first_cycle_context() -> None:
    """Resolver entry should provide RAG, state, and context to cognition."""

    initialized = ensure_initial_resolver_inputs(
        _minimal_global_state(),
        max_cycles=3,
    )

    assert initialized["rag_result"]["answer"] == ""
    assert initialized["resolver_state"]["max_cycles"] == 3
    assert initialized["resolver_state"]["cycle_index"] == 0
    assert "resolver_state: status=running" in initialized["resolver_context"]
    assert "resolver_goal_progress:" in initialized["resolver_context"]
    assert "resolver_observations:" not in initialized["resolver_context"]


def test_targetless_group_self_cognition_bootstraps_without_user_owner() -> None:
    """Group review should keep its semantic targetless contract at bootstrap."""

    state = _minimal_global_state()
    state["global_user_id"] = ""
    state["cognitive_episode"] = {
        "episode_id": "targetless-self-cognition-episode",
        "trigger_source": "self_cognition",
        "target_scope": {
            "channel_type": "group",
            "current_global_user_id": "",
            "current_platform_user_id": "",
        },
    }

    initialized = ensure_initial_resolver_inputs(state, max_cycles=3)

    assert initialized["global_user_id"] == ""
    assert initialized["rag_result"]["answer"] == ""
    assert initialized["rag_result"]["memory_evidence"] == []


def test_initial_resolver_inputs_uses_image_observation_when_text_empty() -> None:
    """Image-only turns should bootstrap resolver goal from image observation."""

    image_summary = '一张浅绿色头发角色头像。'
    state = _minimal_global_state_with_media([
        {
            "content_type": "image/png",
            "description": image_summary,
        },
    ])

    initialized = ensure_initial_resolver_inputs(state, max_cycles=3)

    expected_goal = f'当前输入包含图片观察：{image_summary}'
    resolver_state = initialized["resolver_state"]
    assert initialized["decontextualized_input"] == ""
    assert resolver_state["original_decontextualized_input"] == expected_goal
    assert resolver_state["goal_progress"]["original_goal"] == expected_goal
    assert expected_goal in initialized["resolver_context"]
    assert _first_image_observation_percept(state)["content"]["description"] == (
        image_summary
    )


def test_initial_resolver_inputs_rejects_empty_text_without_image_goal() -> None:
    """Empty text without image evidence remains invalid for the resolver."""

    state = _minimal_global_state()
    state["decontextualized_input"] = ""

    with pytest.raises(ResolverValidationError, match="decontextualized_input"):
        ensure_initial_resolver_inputs(state, max_cycles=3)


def test_initial_resolver_inputs_rejects_audio_only_empty_text() -> None:
    """Audio observations must not become an image-only resolver fallback."""

    state = _minimal_global_state_with_media([
        {
            "content_type": "audio/ogg",
            "description": "user says the deadline is today",
        },
    ])

    with pytest.raises(ResolverValidationError, match="decontextualized_input"):
        ensure_initial_resolver_inputs(state, max_cycles=3)


def test_initial_resolver_inputs_rejects_audit_only_image_empty_text() -> None:
    """Non-model-visible image observations must not bootstrap resolver goals."""

    state = _minimal_global_state_with_media([
        {
            "content_type": "image/png",
            "description": "一张浅绿色头发角色头像。",
        },
    ])
    image_percept = _first_image_observation_percept(state)
    image_percept["content"]["visibility"] = "audit_only"

    with pytest.raises(ResolverValidationError, match="decontextualized_input"):
        ensure_initial_resolver_inputs(state, max_cycles=3)


def test_initial_resolver_inputs_rejects_empty_image_content_empty_text() -> None:
    """Image observations with empty visible content remain invalid."""

    state = _minimal_global_state_with_media([
        {
            "content_type": "image/png",
            "description": "一张浅绿色头发角色头像。",
        },
    ])
    image_percept = _first_image_observation_percept(state)
    image_percept["content"]["description"] = " "

    with pytest.raises(ResolverValidationError, match="decontextualized_input"):
        ensure_initial_resolver_inputs(state, max_cycles=3)
