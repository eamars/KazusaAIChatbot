"""Deterministic checks for the self-cognition POC input framing."""

from experiments.self_cognition_loop_poc import models
from experiments.self_cognition_loop_poc.runner import _plain_residue_text


def test_self_cognition_framing_presents_agency_without_silence_bias() -> None:
    """The POC should frame idle cognition as agency, not passive waiting."""

    source_packet = {
        "idle_timestamp": "2026-05-10T00:30:00+00:00",
        "trigger_focus": {
            "kind": "active_commitment_due_check",
            "summary": "A due reminder should be evaluated.",
            "evidence_scope": "Selected visible conversation and memory.",
        },
        "source_window": {
            "platform": "qq",
            "platform_channel_id": "673225019",
            "channel_type": "private",
            "display_name": "user",
            "last_timestamp": "2026-05-10T00:00:00+00:00",
        },
        "recent_messages": [
            {
                "role": "user",
                "speaker": "user",
                "body_text": "Please remind me tomorrow.",
            },
        ],
        "conversation_progress": {
            "status": "active",
            "episode_phase": "open",
            "current_thread": "due reminder",
            "progression_guidance": "follow up when due",
            "open_loops": ["reminder due"],
            "avoid_reopening": [],
            "next_affordances": ["send reminder candidate"],
            "resolved_threads": [],
        },
        "current_inner_state": {
            "mood": "warm",
            "global_vibe": "attentive",
            "reflection_summary": "The active character feels caring.",
            "self_image_recent": [],
        },
        "memory_evidence": {
            "stable_patterns": [],
            "recent_shifts": [],
            "objective_facts": [],
            "milestones": [],
            "active_commitments": ["The user asked for a reminder."],
        },
        "dry_run_focus": [],
    }

    residue_text = _plain_residue_text(source_packet)

    assert models.SELF_COGNITION_INPUT_TEXT
    assert "Idle self-cognition trigger" in models.SELF_COGNITION_INPUT_TEXT
    assert "There is no new user message to answer" not in residue_text
    assert "Prefer silence" not in residue_text
    assert "Silence is allowed" in residue_text
    assert "proactive message candidate" in residue_text
    assert "[ACTION_CANDIDATE]" in residue_text
    assert "- idle_timestamp: 2026-05-10T00:30:00+00:00" in residue_text
    assert "- last_evidence_timestamp: 2026-05-10T00:00:00+00:00" in residue_text
