"""Deterministic tests for Phase 2 conversation-flow progress fields."""

from __future__ import annotations

from kazusa_ai_chatbot.conversation_progress import projection
from kazusa_ai_chatbot.conversation_progress import recorder
from kazusa_ai_chatbot.conversation_progress import repository
from kazusa_ai_chatbot.conversation_progress.models import ConversationProgressScope
from kazusa_ai_chatbot.conversation_progress.policy import (
    MAX_PROGRESS_PROMPT_CHARS,
    prompt_payload_chars,
)


def _entry(text: str, timestamp: str = "2026-04-28T01:00:00+00:00") -> dict:
    """Build a stored entry fixture.

    Args:
        text: Entry text.
        timestamp: First-seen timestamp.

    Returns:
        Stored conversation episode entry.
    """

    return {"text": text, "first_seen_at": timestamp}


def test_old_phase1_document_projects_with_empty_phase2_fields() -> None:
    """Old documents remain readable and project stable Phase 2 defaults."""

    prompt_doc = projection.project_prompt_doc(
        document={
            "status": "active",
            "episode_label": "essay_help",
            "continuity": "same_episode",
            "turn_count": 3,
            "user_state_updates": [_entry("user needs a third point")],
            "assistant_moves": ["reassurance"],
            "overused_moves": [],
            "open_loops": [_entry("third point remains unresolved")],
            "progression_guidance": "answer the missing point",
        },
        current_timestamp="2026-04-28T04:00:00+00:00",
    )

    assert prompt_doc["conversation_mode"] == ""
    assert prompt_doc["episode_phase"] == ""
    assert prompt_doc["topic_momentum"] == ""
    assert prompt_doc["current_thread"] == ""
    assert prompt_doc["resolved_threads"] == []
    assert prompt_doc["avoid_reopening"] == []
    assert prompt_doc["next_affordances"] == []
    assert prompt_doc["user_state_updates"][0]["age_hint"] == "~3h ago"


def test_sharp_transition_suppresses_stale_flow_fields() -> None:
    """Sharp transitions keep only transition metadata and drop old obligations."""

    prompt_doc = projection.project_prompt_doc(
        document={
            "status": "active",
            "episode_label": "old_task",
            "continuity": "sharp_transition",
            "turn_count": 9,
            "conversation_mode": "task_support",
            "episode_phase": "stuck_loop",
            "topic_momentum": "stable",
            "current_thread": "old unresolved work",
            "current_blocker": "old blocker",
            "resolved_threads": [_entry("old resolved item")],
            "avoid_reopening": [_entry("old stale item")],
            "next_affordances": ["continue old task"],
            "progression_guidance": "continue old task",
        },
        current_timestamp="2026-04-28T04:00:00+00:00",
    )

    assert prompt_doc["continuity"] == "sharp_transition"
    assert prompt_doc["conversation_mode"] == ""
    assert prompt_doc["episode_phase"] == ""
    assert prompt_doc["topic_momentum"] == "sharp_break"
    assert prompt_doc["current_thread"] == ""
    assert prompt_doc["current_blocker"] == ""
    assert prompt_doc["resolved_threads"] == []
    assert prompt_doc["avoid_reopening"] == []
    assert prompt_doc["next_affordances"] == []
    assert prompt_doc["progression_guidance"] == ""


def test_projected_phase2_payload_is_hard_capped() -> None:
    """Projection drops optional fields until the prompt payload is under cap."""

    long_text = "x" * 800
    prompt_doc = projection.project_prompt_doc(
        document={
            "status": "active",
            "episode_label": long_text,
            "continuity": "same_episode",
            "turn_count": 12,
            "conversation_mode": "task_support",
            "episode_phase": "stuck_loop",
            "topic_momentum": "stable",
            "current_thread": long_text,
            "user_goal": long_text,
            "current_blocker": long_text,
            "user_state_updates": [_entry(f"user state {index} {long_text}") for index in range(12)],
            "assistant_moves": [f"assistant move {index} {long_text}" for index in range(12)],
            "overused_moves": [f"overused move {index} {long_text}" for index in range(8)],
            "open_loops": [_entry(f"open loop {index} {long_text}") for index in range(12)],
            "resolved_threads": [_entry(f"resolved {index} {long_text}") for index in range(12)],
            "avoid_reopening": [_entry(f"avoid {index} {long_text}") for index in range(12)],
            "emotional_trajectory": long_text,
            "next_affordances": [f"affordance {index} {long_text}" for index in range(8)],
            "progression_guidance": long_text,
        },
        current_timestamp="2026-04-28T04:00:00+00:00",
    )

    assert prompt_payload_chars(prompt_doc) <= MAX_PROGRESS_PROMPT_CHARS
    assert len(prompt_doc["current_thread"]) <= 180
    assert len(prompt_doc["progression_guidance"]) <= 240


def test_build_episode_state_doc_caps_phase2_fields() -> None:
    """Repository documents cap Phase 2 stored fields before write."""

    scope = ConversationProgressScope("qq", "channel-1", "user-1")
    long_text = "x" * 500
    document = repository.build_episode_state_doc(
        scope=scope,
        timestamp="2026-04-28T04:00:00+00:00",
        prior_episode_state=None,
        recorder_output={
            "status": "active",
            "episode_label": long_text,
            "continuity": "same_episode",
            "conversation_mode": "task_support",
            "episode_phase": "developing",
            "topic_momentum": "stable",
            "current_thread": long_text,
            "user_goal": long_text,
            "current_blocker": long_text,
            "user_state_updates": [long_text for _ in range(12)],
            "assistant_moves": [long_text for _ in range(12)],
            "overused_moves": [long_text for _ in range(12)],
            "open_loops": [long_text for _ in range(12)],
            "resolved_threads": [long_text for _ in range(12)],
            "avoid_reopening": [long_text for _ in range(12)],
            "emotional_trajectory": long_text,
            "next_affordances": [long_text for _ in range(12)],
            "progression_guidance": long_text,
        },
        last_user_input="continue",
    )

    assert len(document["current_thread"]) == 180
    assert len(document["resolved_threads"]) == 5
    assert len(document["avoid_reopening"]) == 5
    assert len(document["next_affordances"]) == 4
    assert len(document["progression_guidance"]) == 240


def test_recorder_validator_accepts_phase2_fields() -> None:
    """Recorder validator accepts the user-approved Phase 2 labels."""

    payload = recorder.validate_recorder_output({
        "status": "active",
        "episode_label": "slides_help",
        "continuity": "same_episode",
        "conversation_mode": "task_support",
        "episode_phase": "developing",
        "topic_momentum": "stable",
        "current_thread": "thesis contribution page",
        "user_goal": "write a third contribution point",
        "current_blocker": "third point overlaps the second",
        "user_state_updates": ["user already has two contribution points"],
        "assistant_moves": ["specific_answer"],
        "overused_moves": ["reassurance"],
        "open_loops": ["third point still missing"],
        "resolved_threads": ["outline order is already settled"],
        "avoid_reopening": ["do not ask to redo the outline"],
        "emotional_trajectory": "frustrated but still engaged",
        "next_affordances": ["give a concrete third point"],
        "progression_guidance": "answer with a distinct third contribution angle",
    })

    assert payload["conversation_mode"] == "task_support"
    assert payload["episode_phase"] == "developing"
    assert payload["topic_momentum"] == "stable"
    assert payload["next_affordances"] == ["give a concrete third point"]


def test_recorder_validator_rejects_invalid_phase2_label() -> None:
    """Recorder validator rejects labels outside the agreed closed sets."""

    try:
        recorder.validate_recorder_output({
            "status": "active",
            "episode_label": "slides_help",
            "continuity": "same_episode",
            "conversation_mode": "unknown_mode",
            "episode_phase": "developing",
            "topic_momentum": "stable",
            "user_state_updates": [],
            "assistant_moves": [],
            "overused_moves": [],
            "open_loops": [],
            "progression_guidance": "",
        })
    except ValueError as exc:
        assert "conversation_mode" in str(exc)
    else:
        raise AssertionError("invalid conversation_mode was accepted")


def test_recorder_prompt_mentions_phase2_flow_contract() -> None:
    """Rendered recorder prompt contains the Phase 2 flow fields."""

    prompt = recorder.render_recorder_prompt()

    assert "conversation_mode" in prompt
    assert "episode_phase" in prompt
    assert "topic_momentum" in prompt
    assert "next_affordances" in prompt
    assert "Do not generate Kazusa's next reply text" in prompt
