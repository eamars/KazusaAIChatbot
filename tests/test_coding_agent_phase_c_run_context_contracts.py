"""Deterministic contracts for coding-run follow-up context."""

from __future__ import annotations

from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
    build_initial_action_capabilities,
    project_prompt_affordances,
)
from kazusa_ai_chatbot.coding_agent.coding_run.ledger import (
    allowed_next_actions,
    project_coding_run_context,
)


def test_resumable_blocker_projects_only_prompt_safe_context() -> None:
    """Expose a user-facing blocker without private execution details."""

    ledger = {
        "run_id": "a" * 32,
        "status": "blocked",
        "goal": "Repair the fixture parser.",
        "updated_at": "2026-07-10T00:00:00Z",
        "blockers": [{
            "blocker_id": "private-blocker-id",
            "blocker_kind": "environment",
            "question": "Install the fixture dependency, then retry.",
            "options": ["Installed", "Cancel"],
            "resume_target": "retry_verification",
            "details": {"private_path": "C:/private"},
            "status": "open",
        }],
    }

    context = project_coding_run_context(ledger)

    assert context["coding_run_ref"] == f"coding_run:{'a' * 32}"
    assert context["allowed_next_actions"] == [
        "respond_to_blocker",
        "summarize",
        "status",
        "cancel",
    ]
    assert context["active_blocker"] == {
        "blocker_kind": "environment",
        "question": "Install the fixture dependency, then retry.",
        "options": ["Installed", "Cancel"],
    }
    assert "details" not in context["active_blocker"]


def test_non_resumable_blocker_excludes_response_action() -> None:
    """Keep a safety blocker visible without authorizing continuation."""

    ledger = {
        "status": "blocked",
        "blockers": [{
            "blocker_kind": "safety",
            "resume_target": "none",
            "status": "open",
        }],
    }

    assert allowed_next_actions(ledger) == ["summarize", "status", "cancel"]


def test_coding_affordance_names_closed_continuation_vocabulary() -> None:
    """Expose semantic actions while retaining deterministic legality checks."""

    affordances = project_prompt_affordances(build_initial_action_capabilities())
    coding_affordance = next(
        row
        for row in affordances
        if row["capability"] == ACCEPTED_CODING_TASK_REQUEST_CAPABILITY
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
