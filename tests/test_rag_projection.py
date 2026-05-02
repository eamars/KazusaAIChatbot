"""Tests for projecting RAG2 known facts into persona context."""

from __future__ import annotations

from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2 import (
    _cognition_rag_result as _l2_cognition_rag_result,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3 import (
    _cognition_rag_result as _l3_cognition_rag_result,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_projection import project_known_facts


def test_project_known_facts_empty_payload() -> None:
    result = project_known_facts(
        [],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert result["answer"] == ""
    assert result["user_image"] == {
        "user_memory_context": {
            "stable_patterns": [],
            "recent_shifts": [],
            "objective_facts": [],
            "milestones": [],
            "active_commitments": [],
        }
    }
    assert result["character_image"] == {}
    assert result["recall_evidence"] == []
    assert result["supervisor_trace"]["dispatched"] == []


def test_project_known_facts_routes_current_and_character_profiles() -> None:
    result = project_known_facts(
        [
            {
                "slot": "current profile",
                "agent": "user_profile_agent",
                "resolved": True,
                "summary": "current user",
                "raw_result": {
                    "global_user_id": "user-1",
                    "user_memory_context": {
                        "objective_facts": [
                            {
                                "fact": "User likes tea",
                                "subjective_appraisal": "Kazusa sees this as a stable preference.",
                                "relationship_signal": "Offer tea-related continuity.",
                            }
                        ]
                    },
                    "_user_memory_units": [
                        {
                            "unit_id": "unit-1",
                            "unit_type": "objective_fact",
                            "fact": "User likes tea",
                            "subjective_appraisal": "Kazusa sees this as a stable preference.",
                            "relationship_signal": "Offer tea-related continuity.",
                        }
                    ],
                },
            },
            {
                "slot": "character profile",
                "agent": "user_profile_agent",
                "resolved": True,
                "summary": "character",
                "raw_result": {"global_user_id": "character-1", "self_image": {"historical_summary": "calm"}},
            },
        ],
        current_user_id="user-1",
        character_user_id="character-1",
        answer="done",
        unknown_slots=["missing"],
        loop_count=2,
    )

    assert result["answer"] == "done"
    assert result["user_image"]["user_memory_context"]["objective_facts"][0]["fact"] == "User likes tea"
    assert "_user_memory_units" not in result["user_image"]
    assert result["user_memory_unit_candidates"][0]["unit_id"] == "unit-1"
    assert result["character_image"]["self_image"]["historical_summary"] == "calm"
    assert result["supervisor_trace"]["loop_count"] == 2
    assert result["supervisor_trace"]["unknown_slots"] == ["missing"]


def test_project_known_facts_groups_summarized_evidence() -> None:
    result = project_known_facts(
        [
            {
                "slot": "lookup",
                "agent": "user_lookup_agent",
                "resolved": True,
                "summary": "小钳子 resolved to user-2",
                "raw_result": {"global_user_id": "user-2"},
            },
            {
                "slot": "memory",
                "agent": "persistent_memory_search_agent",
                "resolved": True,
                "summary": "memory summary",
                "raw_result": [{"content": "A" * 20}],
            },
            {
                "slot": "conversation",
                "agent": "conversation_search_agent",
                "resolved": True,
                "summary": "conversation summary",
                "raw_result": [{"content": "raw should not pass"}],
            },
            {
                "slot": "web",
                "agent": "web_search_agent2",
                "resolved": True,
                "summary": "web summary",
                "raw_result": "https://example.com " + ("B" * 20),
            },
        ],
        current_user_id="user-1",
        character_user_id="character-1",
        evidence_char_limit=8,
    )

    assert result["third_party_profiles"] == ["小钳子 resolved to user-2"]
    assert result["memory_evidence"] == [{"summary": "memory summary", "content": "AAAAAAA…"}]
    assert result["conversation_evidence"] == ["conversation summary"]
    assert result["external_evidence"][0]["summary"] == "web summary"
    assert result["external_evidence"][0]["content"] == "https:/…"
    assert result["external_evidence"][0]["url"] == ""


def test_project_known_facts_does_not_stringify_malformed_fact_values() -> None:
    """RAG projection must not expose repr text from malformed fact rows."""

    result = project_known_facts(
        [
            {
                "slot": {"bad": "slot"},
                "agent": "user_lookup_agent",
                "resolved": True,
                "summary": {"bad": "summary"},
                "raw_result": {"global_user_id": "user-2"},
            },
            {
                "slot": "memory",
                "agent": "persistent_memory_search_agent",
                "resolved": True,
                "summary": "memory summary",
                "raw_result": [{"content": {"bad": "content"}}],
            },
            {
                "slot": "web",
                "agent": "web_search_agent2",
                "resolved": True,
                "summary": "web summary",
                "raw_result": {"bad": "external content"},
            },
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    rendered = repr(result)

    assert "{'bad':" not in rendered
    assert result["supervisor_trace"]["dispatched"] == [
        {"slot": "", "agent": "user_lookup_agent", "resolved": True},
        {"slot": "memory", "agent": "persistent_memory_search_agent", "resolved": True},
        {"slot": "web", "agent": "web_search_agent2", "resolved": True},
    ]
    assert result["third_party_profiles"] == []
    assert result["memory_evidence"] == [{"summary": "memory summary", "content": ""}]
    assert result["external_evidence"] == [{"summary": "web summary", "content": "", "url": ""}]


def test_project_known_facts_projects_recall_agent_result() -> None:
    """Recall helper output should be exposed separately from conversation evidence."""

    result = project_known_facts(
        [
            {
                "slot": "Recall: retrieve active_episode_agreement relevant to today's appointment",
                "agent": "recall_agent",
                "resolved": True,
                "summary": "The active agreement is pickup at 9:30.",
                "raw_result": {
                    "selected_summary": "The active agreement is pickup at 9:30.",
                    "recall_type": "active_episode_agreement",
                    "primary_source": "conversation_progress",
                    "supporting_sources": ["user_memory_units"],
                    "freshness_basis": "Active progress is current.",
                    "conflicts": [],
                    "candidates": [
                        {
                            "source": "conversation_progress",
                            "claim": "Pickup at 9:30.",
                            "temporal_scope": "current_episode",
                            "lifecycle_status": "active",
                            "evidence_time": "2026-05-01T23:00:00+00:00",
                            "authority": "primary_for_current_episode",
                        }
                    ],
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert result["recall_evidence"] == [
        {
            "selected_summary": "The active agreement is pickup at 9:30.",
            "recall_type": "active_episode_agreement",
            "primary_source": "conversation_progress",
            "supporting_sources": ["user_memory_units"],
            "freshness_basis": "Active progress is current.",
            "conflicts": [],
            "candidates": [
                {
                    "source": "conversation_progress",
                    "claim": "Pickup at 9:30.",
                    "temporal_scope": "current_episode",
                    "lifecycle_status": "active",
                    "evidence_time": "2026-05-01T23:00:00+00:00",
                    "authority": "primary_for_current_episode",
                }
            ],
        }
    ]
    assert result["conversation_evidence"] == []


def test_project_known_facts_caps_recall_evidence_to_three_entries() -> None:
    """Projection should expose only the first three Recall results."""

    known_facts = [
        {
            "slot": f"Recall: retrieve active_episode_agreement relevant to plan {index}",
            "agent": "recall_agent",
            "resolved": True,
            "summary": f"Recall summary {index}",
            "raw_result": {
                "selected_summary": f"Recall summary {index}",
                "primary_source": "conversation_progress",
            },
        }
        for index in range(4)
    ]

    result = project_known_facts(
        known_facts,
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert [
        entry["selected_summary"]
        for entry in result["recall_evidence"]
    ] == [
        "Recall summary 0",
        "Recall summary 1",
        "Recall summary 2",
    ]
    assert len(result["supervisor_trace"]["dispatched"]) == 4
    assert result["supervisor_trace"]["dispatched"][3]["agent"] == "recall_agent"


def test_project_known_facts_maps_top_level_capability_payloads() -> None:
    """Projection should consume normalized top-level capability payloads."""
    current_profile = {
        "global_user_id": "user-1",
        "display_name": "Tester",
        "self_image": {"summary": "current user image"},
        "_user_memory_units": [{"unit_id": "unit-1", "fact": "likes tea"}],
    }
    character_profile = {
        "global_user_id": "character-1",
        "self_image": {"summary": "character image"},
    }

    result = project_known_facts(
        [
            {
                "slot": "live",
                "agent": "live_context_agent",
                "resolved": True,
                "summary": "live summary",
                "raw_result": {
                    "projection_payload": {
                        "external_text": "Auckland is 17 C.",
                        "url": "https://weather.example/auckland",
                    }
                },
            },
            {
                "slot": "conversation",
                "agent": "conversation_evidence_agent",
                "resolved": True,
                "summary": "conversation summary",
                "raw_result": {
                    "projection_payload": {
                        "summaries": ["speaker: phrase", "speaker: link"],
                    }
                },
            },
            {
                "slot": "memory",
                "agent": "memory_evidence_agent",
                "resolved": True,
                "summary": "memory summary",
                "raw_result": {
                    "projection_payload": {
                        "memory_rows": [
                            {"content": "official address is 123 Example Street"}
                        ],
                    }
                },
            },
            {
                "slot": "current user",
                "agent": "person_context_agent",
                "resolved": True,
                "summary": "current user",
                "raw_result": {
                    "projection_payload": {
                        "profile_kind": "current_user",
                        "owner_global_user_id": "user-1",
                        "profile": current_profile,
                        "summary": "Tester",
                    }
                },
            },
            {
                "slot": "character",
                "agent": "person_context_agent",
                "resolved": True,
                "summary": "character",
                "raw_result": {
                    "projection_payload": {
                        "profile_kind": "active_character",
                        "owner_global_user_id": "character-1",
                        "profile": character_profile,
                        "summary": "Character",
                    }
                },
            },
            {
                "slot": "third party",
                "agent": "person_context_agent",
                "resolved": True,
                "summary": "third party",
                "raw_result": {
                    "projection_payload": {
                        "profile_kind": "third_party",
                        "owner_global_user_id": "user-2",
                        "summary": "Third party summary",
                    }
                },
            },
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert result["external_evidence"] == [
        {
            "summary": "live summary",
            "content": "Auckland is 17 C.",
            "url": "https://weather.example/auckland",
        }
    ]
    assert result["conversation_evidence"] == ["speaker: phrase", "speaker: link"]
    assert result["memory_evidence"] == [
        {
            "summary": "memory summary",
            "content": "official address is 123 Example Street",
        }
    ]
    assert result["user_image"]["display_name"] == "Tester"
    assert "_user_memory_units" not in result["user_image"]
    assert result["user_memory_unit_candidates"] == [
        {"unit_id": "unit-1", "fact": "likes tea"}
    ]
    assert result["character_image"] == character_profile
    assert result["third_party_profiles"] == ["Third party summary"]


def test_project_known_facts_skips_unresolved_top_level_payload() -> None:
    """Unresolved capability results should remain only in supervisor trace."""
    result = project_known_facts(
        [
            {
                "slot": "weather",
                "agent": "live_context_agent",
                "resolved": False,
                "summary": "missing location",
                "raw_result": {
                    "missing_context": ["location"],
                    "projection_payload": {
                        "external_text": "should not project",
                    },
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert result["external_evidence"] == []
    assert result["supervisor_trace"]["dispatched"] == [
        {"slot": "weather", "agent": "live_context_agent", "resolved": False}
    ]


def test_cognition_rag_result_preserves_public_recall_payload() -> None:
    """Existing cognition payload shaping should not strip Recall evidence."""

    rag_result = {
        "answer": "The active agreement is pickup at 9:30.",
        "recall_evidence": [
            {
                "selected_summary": "The active agreement is pickup at 9:30.",
                "primary_source": "conversation_progress",
            }
        ],
        "user_memory_unit_candidates": [{"unit_id": "internal-1"}],
    }

    l2_payload = _l2_cognition_rag_result(rag_result)
    l3_payload = _l3_cognition_rag_result(rag_result)

    assert l2_payload["answer"] == "The active agreement is pickup at 9:30."
    assert l3_payload["answer"] == "The active agreement is pickup at 9:30."
    assert l2_payload["recall_evidence"][0]["primary_source"] == "conversation_progress"
    assert l3_payload["recall_evidence"][0]["primary_source"] == "conversation_progress"
    assert "user_memory_unit_candidates" not in l2_payload
    assert "user_memory_unit_candidates" not in l3_payload
