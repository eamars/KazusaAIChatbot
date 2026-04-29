"""Tests for projecting RAG2 known facts into persona context."""

from __future__ import annotations

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
