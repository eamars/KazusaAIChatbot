import pytest
pytest.skip("Stage 1 assertions replaced by the V2 contract suite", allow_module_level=True)

"""Deterministic contracts for coding-run follow-up context."""

from kazusa_ai_chatbot.coding_agent.coding_run.ledger import (
    allowed_next_actions,
    project_coding_run_context,
)
from kazusa_ai_chatbot.background_work.result_source import (
    build_result_ready_episode_from_job,
)
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
    build_initial_action_capabilities,
    project_prompt_affordances,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    _coding_run_context_from_episode,
    _coding_run_followup,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_actions import (
    materialize_semantic_action_requests,
)


def test_resumable_blocker_projects_only_prompt_safe_context() -> None:
    """A resumable blocker exposes its user-facing question and action only."""

    ledger = {
        "run_id": "a" * 32,
        "status": "blocked",
        "goal": "Repair the fixture parser.",
        "updated_at": "2026-07-10T00:00:00Z",
        "blockers": [{
            "blocker_id": "private-blocker-id",
            "blocker_kind": "environment",
            "question": "Install the fixture dependency, then tell me to retry.",
            "options": ["I installed it", "Cancel this run"],
            "resume_target": "retry_verification",
            "details": {"private_path": "C:/private"},
            "status": "open",
        }],
    }

    context = project_coding_run_context(ledger)

    assert context == {
        "schema_version": "coding_run_context.v1",
        "coding_run_ref": f"coding_run:{'a' * 32}",
        "status": "blocked",
        "objective_summary": "Repair the fixture parser.",
        "allowed_next_actions": [
            "respond_to_blocker",
            "summarize",
            "status",
            "cancel",
        ],
        "active_blocker": {
            "blocker_kind": "environment",
            "question": "Install the fixture dependency, then tell me to retry.",
            "options": ["I installed it", "Cancel this run"],
        },
        "followup_open": True,
        "updated_at": "2026-07-10T00:00:00Z",
    }


def test_non_resumable_blocker_excludes_response_action() -> None:
    """A safety blocker remains visible without authorizing continuation."""

    ledger = {
        "status": "blocked",
        "blockers": [{
            "blocker_kind": "safety",
            "resume_target": "none",
            "status": "open",
        }],
    }

    actions = allowed_next_actions(ledger)

    assert actions == ["summarize", "status", "cancel"]


def test_l2d_context_excludes_private_blocker_and_worker_fields() -> None:
    """L2d receives current action affordances without ledger internals."""

    payload = build_action_selection_payload({
        "action_selection_context": {
            "coding_runs": [{
                "coding_run_ref": f"coding_run:{'b' * 32}",
                "status": "blocked",
                "objective_summary": "Repair the fixture parser.",
                "allowed_next_actions": ["respond_to_blocker", "status"],
                "active_blocker": {
                    "question": "Install the dependency, then retry.",
                    "options": ["Installed"],
                    "details": {"private_path": "C:/private"},
                },
                "approval_evidence": {"quote": "private"},
                "lock_key": "source:private",
            }],
        },
    })

    coding_run = payload["coding_runs"][0]

    assert coding_run == {
        "coding_run_ref": f"coding_run:{'b' * 32}",
        "status": "blocked",
        "objective_summary": "Repair the fixture parser.",
        "allowed_next_actions": ["respond_to_blocker", "status"],
        "active_blocker": {
            "question": "Install the dependency, then retry.",
            "options": ["Installed"],
        },
    }


def test_l2d_coding_action_semantics_name_every_continuation() -> None:
    """L2d receives the full closed action vocabulary and affordance rule."""

    affordances = project_prompt_affordances(build_initial_action_capabilities())
    coding_affordance = next(
        affordance
        for affordance in affordances
        if affordance["capability"] == ACCEPTED_CODING_TASK_REQUEST_CAPABILITY
    )
    summary = " ".join(coding_affordance["semantic_input_summary"])

    for action in (
        "start",
        "revise_proposal",
        "summarize",
        "status",
        "approve_and_verify",
        "respond_to_blocker",
        "cancel",
    ):
        assert action in summary
    assert "allowed_next_actions" in summary
    assert "multiple offered runs" in summary


def test_materialization_requires_an_offered_action_and_unambiguous_ref() -> None:
    """Continuation requests bind only to current legal coding-run contexts."""

    request = [{
        "capability": "accepted_coding_task_request",
        "decision": "approve_and_verify",
        "detail": "Run the approved verification.",
        "reason": "The user approved the proposal.",
        "coding_run_ref": "coding_run:run-001",
    }]
    state = _materialization_state([{
        "coding_run_ref": "coding_run:run-001",
        "allowed_next_actions": ["summarize"],
    }])

    specs = materialize_semantic_action_requests(request, state)

    assert specs == []


def test_materialization_does_not_guess_between_open_runs() -> None:
    """A ref-less semantic continuation is dropped when multiple runs allow it."""

    request = [{
        "capability": "accepted_coding_task_request",
        "decision": "respond_to_blocker",
        "detail": "The dependency is installed.",
        "reason": "The user answered the blocker.",
    }]
    state = _materialization_state([
        {
            "coding_run_ref": "coding_run:run-001",
            "allowed_next_actions": ["respond_to_blocker"],
        },
        {
            "coding_run_ref": "coding_run:run-002",
            "allowed_next_actions": ["respond_to_blocker"],
        },
    ])

    specs = materialize_semantic_action_requests(request, state)

    assert specs == []


def test_l3_followup_excludes_run_refs_and_operational_fields() -> None:
    """Visible wording receives only objective and blocker clarification facts."""

    followup = _coding_run_followup([{
        "coding_run_ref": "coding_run:private-run-id",
        "followup_open": True,
        "objective_summary": "Repair the fixture parser.",
        "active_blocker": {
            "question": "Install the missing dependency, then reply.",
            "options": ["Installed", "Cancel"],
            "details": {"private_path": "C:/private"},
        },
        "execution_specs": [{"tool": "pytest"}],
        "approval_evidence": {"quote": "private"},
    }])

    assert followup == {
        "mode": "single",
        "runs": [{
            "objective_summary": "Repair the fixture parser.",
            "active_blocker": {
                "question": "Install the missing dependency, then reply.",
                "options": ["Installed", "Cancel"],
            },
        }],
    }


def test_result_ready_episode_carries_only_the_sanitized_run_context() -> None:
    """Worker result delivery preserves the semantic context for L3 handoff."""

    episode = build_result_ready_episode_from_job({
        "accepted_task_id": "task-001",
        "created_at": "2026-07-10T00:00:00Z",
        "updated_at": "2026-07-10T00:00:00Z",
        "task_brief": "Repair the fixture parser.",
        "artifact_text": "Dependency is unavailable.",
        "failure_summary": "",
        "result_summary": "Blocked on a dependency.",
        "source_platform": "debug",
        "source_channel_id": "debug:user:test",
        "source_channel_type": "private",
        "source_message_id": "message-001",
        "requester_platform_user_id": "platform-user-001",
        "requester_global_user_id": "user-001",
        "requester_display_name": "Test User",
        "source_platform_bot_id": "bot-001",
        "source_character_name": "Test Character",
        "worker_metadata": {
            "coding_run_context": {
                "schema_version": "coding_run_context.v1",
                "coding_run_ref": "coding_run:run-001",
                "status": "blocked",
                "objective_summary": "Repair the fixture parser.",
                "allowed_next_actions": ["respond_to_blocker", "status"],
                "active_blocker": {
                    "blocker_kind": "environment",
                    "question": "Install the dependency and reply.",
                    "options": ["Installed"],
                },
                "followup_open": True,
                "updated_at": "2026-07-10T00:00:00Z",
                "execution_specs": [{"tool": "pytest"}],
                "approval_evidence": {"quote": "private"},
            },
        },
    })

    context = _coding_run_context_from_episode(episode)
    followup = _coding_run_followup([context])

    assert context is not None
    assert "execution_specs" not in context
    assert "approval_evidence" not in context
    assert followup == {
        "mode": "single",
        "runs": [{
            "objective_summary": "Repair the fixture parser.",
            "active_blocker": {
                "question": "Install the dependency and reply.",
                "options": ["Installed"],
            },
        }],
    }


def _materialization_state(
    contexts: list[dict[str, object]],
) -> dict[str, object]:
    """Build the trusted minimum state for deterministic action binding tests."""

    state = {
        "storage_timestamp_utc": "2026-07-10T00:00:00Z",
        "decontexualized_input": "Continue the coding work.",
        "platform": "debug",
        "platform_channel_id": "debug:user:test",
        "channel_type": "private",
        "platform_message_id": "message-001",
        "platform_bot_id": "bot-001",
        "global_user_id": "user-001",
        "platform_user_id": "platform-user-001",
        "user_name": "Test User",
        "character_profile": {
            "name": "Test Character",
            "global_user_id": "character-001",
        },
        "cognitive_episode": {
            "trigger_source": "user_message",
            "storage_timestamp_utc": "2026-07-10T00:00:00Z",
            "target_scope": {"current_global_user_id": "user-001"},
            "origin_metadata": {"platform_message_id": "message-001"},
            "percepts": [{
                "input_source": "dialog_text",
                "content": "Continue the coding work.",
            }],
        },
        "action_selection_context": {"coding_runs": contexts},
        "conversation_progress": {},
    }
    return state
