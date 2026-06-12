"""Tests for projecting RAG2 known facts into persona context."""

from __future__ import annotations

import json
import logging

from kazusa_ai_chatbot.cognition_chain_core.stages.l2 import (
    _cognition_rag_result as _l2_cognition_rag_result,
)
from kazusa_ai_chatbot.cognition_chain_core.stages.l3 import (
    _cognition_rag_result as _l3_cognition_rag_result,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_projection import project_known_facts


def _assert_ordered_evidence_block(text: str) -> None:
    assert text.startswith('结论：')
    conclusion_index = text.index('结论：')
    uncertainty_index = text.index('不确定性：')
    if '上下文：' in text:
        evidence_index = text.index('上下文：')
        assert conclusion_index < evidence_index < uncertainty_index
    else:
        assert conclusion_index < uncertainty_index


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


def test_project_known_facts_has_no_interaction_style_result_shape() -> None:
    """RAG projection shape does not expose interaction style fields."""

    result = project_known_facts(
        [],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    rendered = repr(result)

    assert "user_style_image" not in rendered
    assert "group_channel_style_image" not in rendered
    assert "interaction_style_context" not in rendered


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
                "agent": "web_agent3",
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
    assert result["memory_evidence"][0]["summary"].startswith('结论：memory summary')
    assert result["memory_evidence"][0]["content"].startswith('上下文：')
    assert "AAAAAAA…" in result["memory_evidence"][0]["content"]
    assert result["conversation_evidence"] == ['结论：conversation summary\n不确定性：无']
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
                "agent": "web_agent3",
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
        {"slot": "web", "agent": "web_agent3", "resolved": True},
    ]
    assert result["third_party_profiles"] == []
    assert result["memory_evidence"] == [
            {
                "summary": '结论：memory summary',
                "content": '不确定性：没有可用于提示的记忆证据。',
            }
        ]
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

    recall_entry = result["recall_evidence"][0]
    assert recall_entry["selected_summary"] == (
        '结论：The active agreement is pickup at 9:30.'
    )
    assert recall_entry["recall_type"] == "active_episode_agreement"
    assert recall_entry["primary_source"] == "conversation_progress"
    assert recall_entry["supporting_sources"] == ["user_memory_units"]
    assert recall_entry["freshness_basis"] == "Active progress is current."
    assert recall_entry["conflicts"] == []
    assert recall_entry["evidence_summary"].startswith('上下文：')
    assert "Pickup at 9:30." in recall_entry["evidence_summary"]
    assert "2026-05-02 11:00:00" in recall_entry["evidence_summary"]
    assert "2026-05-01T23:00:00+00:00" not in repr(recall_entry)
    assert "candidates" not in recall_entry
    assert result["conversation_evidence"] == []


def test_project_known_facts_omits_unsafe_recall_freshness_basis() -> None:
    """Trace-like Recall provenance should not crash public projection."""

    result = project_known_facts(
        [
            {
                "slot": "Recall: retrieve active_episode_agreement relevant to tiramisu",
                "agent": "recall_agent",
                "resolved": True,
                "summary": "The active agreement is unresolved tiramisu compensation.",
                "raw_result": {
                    "selected_summary": "The active agreement is unresolved tiramisu compensation.",
                    "recall_type": "active_episode_agreement",
                    "primary_source": "user_memory_units",
                    "supporting_sources": ["conversation_progress"],
                    "freshness_basis": (
                        "Selected user_memory_units for active_episode_agreement; "
                        "evidence_time=2026-05-24T09:18:11.123456+00:00."
                    ),
                    "conflicts": [],
                    "candidates": [
                        {
                            "source": "user_memory_units",
                            "claim": "Tiramisu compensation is still unresolved.",
                            "temporal_scope": "active_episode",
                            "lifecycle_status": "active",
                            "evidence_time": "2026-05-24T09:18:11.123456+00:00",
                            "authority": "primary_for_current_episode",
                        }
                    ],
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    recall_entry = result["recall_evidence"][0]
    rendered_entry = repr(recall_entry)

    assert "freshness_basis" not in recall_entry
    assert "Tiramisu compensation is still unresolved." in rendered_entry
    assert "2026-05-24 21:18:11" in rendered_entry
    assert "2026-05-24T09:18:11.123456+00:00" not in rendered_entry


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
        entry["selected_summary"].replace('结论：', '')
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
    assert result["conversation_evidence"] == [
        '结论：speaker: phrase\n不确定性：无',
        '结论：speaker: link\n不确定性：无',
    ]
    assert result["memory_evidence"][0]["summary"] == '结论：memory summary'
    assert result["memory_evidence"][0]["content"].startswith('上下文：')
    assert "official address is 123 Example Street" in result["memory_evidence"][0]["content"]
    assert result["user_image"]["display_name"] == "Tester"
    assert "_user_memory_units" not in result["user_image"]
    assert result["user_memory_unit_candidates"] == [
        {"unit_id": "unit-1", "fact": "likes tea"}
    ]
    assert result["character_image"] == character_profile
    assert result["third_party_profiles"] == ["Third party summary"]


def test_project_known_facts_preserves_external_url_query_marker() -> None:
    result = project_known_facts(
        [
            {
                "slot": "live",
                "agent": "live_context_agent",
                "resolved": True,
                "summary": "live summary",
                "raw_result": {
                    "projection_payload": {
                        "external_text": "External source was read.",
                        "url": (
                            "https://example.test/redirect?"
                            "url=https%3A%2F%2Ftarget.example%2Fpage"
                        ),
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert result["external_evidence"][0]["url"] == (
        "https://example.test/redirect?"
        "url=https%3A%2F%2Ftarget.example%2Fpage"
    )


def test_project_known_facts_preserves_external_url_uuid_path() -> None:
    result = project_known_facts(
        [
            {
                "slot": "live",
                "agent": "live_context_agent",
                "resolved": True,
                "summary": "live summary",
                "raw_result": {
                    "projection_payload": {
                        "external_text": "External source was read.",
                        "url": (
                            "https://example.test/resource/"
                            "123e4567-e89b-12d3-a456-426614174000"
                        ),
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert result["external_evidence"][0]["url"] == (
        "https://example.test/resource/"
        "123e4567-e89b-12d3-a456-426614174000"
    )


def test_project_known_facts_keeps_external_content_with_valid_url_query() -> None:
    result = project_known_facts(
        [
            {
                "slot": "web",
                "agent": "web_agent3",
                "resolved": True,
                "summary": "web summary",
                "raw_result": (
                    "Source says target value is 42 at "
                    "https://example.test/redirect?"
                    "url=https%3A%2F%2Ftarget.example%2Fpage"
                ),
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert "target value is 42" in result["external_evidence"][0]["content"]
    assert result["external_evidence"][0]["url"] == (
        "https://example.test/redirect?"
        "url=https%3A%2F%2Ftarget.example%2Fpage"
    )


def test_project_known_facts_blanks_malformed_external_url() -> None:
    result = project_known_facts(
        [
            {
                "slot": "live",
                "agent": "live_context_agent",
                "resolved": True,
                "summary": "live summary",
                "raw_result": {
                    "projection_payload": {
                        "external_text": "External source was read.",
                        "url": "http://[broken",
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert result["external_evidence"][0]["url"] == ""
    assert result["supervisor_trace"]["safety_recovery"]


def test_project_known_facts_drops_unsafe_cq_answer(caplog) -> None:
    caplog.set_level(
        logging.WARNING,
        logger="kazusa_ai_chatbot.nodes.persona_supervisor2_rag_projection",
    )

    result = project_known_facts(
        [],
        current_user_id="user-1",
        character_user_id="character-1",
        answer="[CQ:image,file=abc]",
    )

    assert result["answer"] == ""
    assert result["supervisor_trace"]["safety_recovery"]
    json.dumps(result, ensure_ascii=False)
    assert "[CQ:" not in caplog.text


def test_project_known_facts_records_recoverable_answer_sanitization() -> None:
    result = project_known_facts(
        [],
        current_user_id="user-1",
        character_user_id="character-1",
        answer=(
            "global_user_id: "
            "123e4567-e89b-12d3-a456-426614174000 says tea"
        ),
    )

    assert "says tea" in result["answer"]
    assert result["supervisor_trace"]["safety_recovery"]


def test_project_known_facts_drops_unsafe_memory_line() -> None:
    result = project_known_facts(
        [
            {
                "slot": "memory",
                "agent": "memory_evidence_agent",
                "resolved": True,
                "summary": "memory summary",
                "raw_result": {
                    "projection_payload": {
                        "memory_rows": [
                            {"content": "[CQ:image,file=abc]"},
                            {"content": "User prefers tea."},
                        ],
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    rendered = repr(result["memory_evidence"])

    assert "User prefers tea." in rendered
    assert "[CQ:" not in rendered
    assert result["supervisor_trace"]["safety_recovery"]


def test_project_known_facts_drops_unrecoverable_external_evidence() -> None:
    result = project_known_facts(
        [
            {
                "slot": "web",
                "agent": "web_agent3",
                "resolved": True,
                "summary": "[CQ:bad]",
                "raw_result": "[CQ:image,file=abc]",
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert result["external_evidence"] == []
    assert result["supervisor_trace"]["safety_recovery"]


def test_cognition_rag_result_omits_safety_recovery_trace() -> None:
    rag_result = {
        "answer": "safe answer",
        "memory_evidence": [],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {
            "loop_count": 1,
            "unknown_slots": [],
            "dispatched": [],
            "safety_recovery": ["dropped_text:rag_result.answer"],
        },
    }

    l2_payload = _l2_cognition_rag_result(rag_result)
    l3_payload = _l3_cognition_rag_result(rag_result)

    assert l2_payload["supervisor_trace"] == {
        "loop_count": 1,
        "unknown_slots": [],
        "dispatched": [],
    }
    assert l3_payload["supervisor_trace"] == {
        "loop_count": 1,
        "unknown_slots": [],
        "dispatched": [],
    }


def test_l2_cognition_rag_result_omits_raw_supervisor_source_refs() -> None:
    """L2 should receive evidence trace shape without raw durable ids."""

    rag_result = {
        "answer": "safe answer",
        "memory_evidence": [],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {
            "loop_count": 1,
            "dispatched": [
                {
                    "slot": "conversation evidence",
                    "agent": "conversation_search",
                    "resolved": True,
                    "source_refs": [
                        {
                            "conversation_row_id": "raw-row-id",
                            "platform_message_id": "raw-message-id",
                            "unit_id": "raw-unit-id",
                        },
                    ],
                    "nested": {
                        "raw_refs": [{"_id": "raw-object-id"}],
                        "summary": "kept",
                    },
                },
            ],
        },
    }

    l2_payload = _l2_cognition_rag_result(rag_result)
    rendered_payload = json.dumps(l2_payload, ensure_ascii=False)
    dispatched = l2_payload["supervisor_trace"]["dispatched"][0]

    assert "source_refs" not in dispatched
    assert "raw_refs" not in dispatched["nested"]
    assert dispatched["nested"]["summary"] == "kept"
    for raw_id in (
        "raw-row-id",
        "raw-message-id",
        "raw-unit-id",
        "raw-object-id",
    ):
        assert raw_id not in rendered_payload


def test_project_known_facts_sanitizes_third_party_profile_source_ids() -> None:
    """Third-party profile summaries should not expose source ids to cognition."""

    result = project_known_facts(
        [
            {
                "slot": "third party",
                "agent": "person_context_agent",
                "resolved": True,
                "summary": "fallback summary",
                "raw_result": {
                    "projection_payload": {
                        "profile_kind": "third_party",
                        "summary": (
                            "Night | "
                            "123e4567-e89b-12d3-a456-426614174000"
                        ),
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert result["third_party_profiles"] == ["Night"]


def test_project_known_facts_preserves_scoped_user_memory_metadata_and_candidates() -> None:
    """Scoped user-memory evidence should remain visible to cognition and consolidation."""

    result = project_known_facts(
        [
            {
                "slot": "memory",
                "agent": "memory_evidence_agent",
                "resolved": True,
                "summary": "scoped continuity summary",
                "raw_result": {
                    "projection_payload": {
                        "memory_rows": [
                            {
                                "unit_id": "unit-7",
                                "unit_type": "objective_fact",
                                "fact": "冰淇淋摊老板是千纱的初中学姐。",
                                "subjective_appraisal": "Kazusa treats this as shared private continuity.",
                                "relationship_signal": "Preserve the lore with this user.",
                                "content": "冰淇淋摊老板是千纱的初中学姐。",
                                "updated_at": "2026-05-03T00:00:00+00:00",
                                "source_system": "user_memory_units",
                                "scope_type": "user_continuity",
                                "scope_global_user_id": "user-1",
                                "authority": "scoped_continuity",
                                "truth_status": "character_lore_or_interaction_continuity",
                                "origin": "consolidated_interaction",
                            },
                            {
                                "memory_name": "active-character-official-address",
                                "content": "The active character's official address is 123 Example Street.",
                                "source_kind": "seeded_manual",
                            },
                        ],
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert len(result["memory_evidence"]) == 1
    entry = result["memory_evidence"][0]
    assert entry["summary"] == '结论：scoped continuity summary'
    assert entry["content"].startswith('上下文：')
    assert "冰淇淋摊老板是千纱的初中学姐。" in entry["content"]
    assert "The active character's official address is 123 Example Street." in entry["content"]
    assert entry["source_system"] == "user_memory_units"
    assert entry["scope_type"] == "user_continuity"
    assert entry["scope_global_user_id"] == "user-1"
    assert entry["authority"] == "scoped_continuity"
    assert entry["truth_status"] == "character_lore_or_interaction_continuity"
    assert entry["origin"] == "consolidated_interaction"
    assert result["user_memory_unit_candidates"] == [
        {
            "unit_id": "unit-7",
            "unit_type": "objective_fact",
            "fact": "冰淇淋摊老板是千纱的初中学姐。",
            "subjective_appraisal": "Kazusa treats this as shared private continuity.",
            "relationship_signal": "Preserve the lore with this user.",
            "content": "冰淇淋摊老板是千纱的初中学姐。",
            "updated_at": "2026-05-03T00:00:00+00:00",
            "source_system": "user_memory_units",
            "scope_type": "user_continuity",
            "scope_global_user_id": "user-1",
            "authority": "scoped_continuity",
            "truth_status": "character_lore_or_interaction_continuity",
            "origin": "consolidated_interaction",
        }
    ]


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


def test_project_known_facts_keeps_continuation_trace_out_of_public_evidence() -> None:
    """Continuation observations should stay in trace, not public evidence."""
    result = project_known_facts(
        [
            {
                "slot": "Memory-evidence: retrieve durable policy",
                "agent": "memory_evidence_agent",
                "resolved": False,
                "summary": "missing concrete memory evidence",
                "raw_result": {
                    "projection_payload": {
                        "memory_rows": [
                            {
                                "content": "candidate should not project",
                            }
                        ],
                    },
                    "observation_candidates": [
                        {
                            "content": "candidate should not project",
                        }
                    ],
                },
                "continuation": {
                    "should_continue": True,
                    "refined_query": (
                        "Need a current fact. Prior memory only provided a "
                        "source strategy, so retrieve fresh evidence."
                    ),
                    "reason": "fresh source direction",
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    public_payload = {
        key: value
        for key, value in result.items()
        if key != "supervisor_trace"
    }
    rendered_public = repr(public_payload)
    trace_entry = result["supervisor_trace"]["dispatched"][0]

    assert result["memory_evidence"] == []
    assert result["conversation_evidence"] == []
    assert result["external_evidence"] == []
    assert result["recall_evidence"] == []
    assert "candidate should not project" not in rendered_public
    assert "fresh_external_evidence" not in rendered_public
    assert trace_entry["continuation"]["should_continue"] is True
    assert "fresh evidence" in trace_entry["continuation"]["refined_query"]
    assert trace_entry["continuation"]["reason"] == "fresh source direction"


def test_project_known_facts_public_keys_unchanged() -> None:
    """The projected RAG result keeps the stable public key set."""
    result = project_known_facts(
        [
            {
                "slot": "Memory-evidence: retrieve durable policy",
                "agent": "memory_evidence_agent",
                "resolved": False,
                "summary": "missing concrete memory evidence",
                "raw_result": {},
                "continuation": {
                    "should_continue": False,
                    "refined_query": "",
                    "reason": "no source direction",
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert set(result.keys()) == {
        "answer",
        "user_image",
        "user_memory_unit_candidates",
        "character_image",
        "third_party_profiles",
        "memory_evidence",
        "recall_evidence",
        "conversation_evidence",
        "external_evidence",
        "supervisor_trace",
    }
    assert set(result["supervisor_trace"].keys()) >= {
        "loop_count",
        "unknown_slots",
        "dispatched",
    }


def test_project_known_facts_projects_formatted_memory_evidence() -> None:
    result = project_known_facts(
        [
            {
                "slot": "memory",
                "agent": "memory_evidence_agent",
                "resolved": True,
                "summary": "User prefers tea.",
                "raw_result": {
                    "projection_payload": {
                        "memory_rows": [
                            {
                                "content": "User prefers tea during late sessions.",
                                "updated_at": "2026-05-01T12:34:56.789000+00:00",
                                "source_system": "user_memory_units",
                            }
                        ],
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    entry = result["memory_evidence"][0]
    assert entry["summary"] == '结论：User prefers tea.'
    assert entry["content"].startswith('上下文：\n- ')
    assert "User prefers tea during late sessions." in entry["content"]
    assert "2026-05-02 00:34:56" in entry["content"]
    assert '不确定性：无' in entry["content"]
    assert "2026-05-01T12:34:56.789000+00:00" not in repr(entry)


def test_project_known_facts_projects_formatted_conversation_evidence() -> None:
    result = project_known_facts(
        [
            {
                "slot": "conversation",
                "agent": "conversation_evidence_agent",
                "resolved": True,
                "summary": "Tester promised to send the chart.",
                "raw_result": {
                    "projection_payload": {
                        "summaries": ["Tester: I will send the chart tonight."],
                        "rows": [
                            {
                                "summary": "Tester: I will send the chart tonight.",
                                "timestamp": "2026-05-01T12:34:56.789000+00:00",
                                "display_name": "Tester",
                                "conversation_row_id": "row-1",
                                "platform_message_id": "message-1",
                            }
                        ],
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    evidence = result["conversation_evidence"][0]
    _assert_ordered_evidence_block(evidence)
    assert "Tester promised to send the chart." in evidence
    assert "Tester（2026-05-02 00:34:56）" in evidence
    assert "Tester: I will send the chart tonight." in evidence
    assert "row-1" not in evidence
    assert "message-1" not in evidence
    assert "2026-05-01T12:34:56.789000+00:00" not in evidence


def test_project_known_facts_prefers_conversation_packets_over_flat_rows() -> None:
    result = project_known_facts(
        [
            {
                "slot": "conversation",
                "agent": "conversation_evidence_agent",
                "resolved": True,
                "summary": "Google Drive context was found.",
                "raw_result": {
                    "projection_payload": {
                        "summaries": [
                            "Nightfall: Google Drive 又不是第一次这样了。",
                            "Nightfall: <image>Google Drive 权限禁止的截图。</image>",
                        ],
                        "rows": [
                            {
                                "summary": "Nightfall: Google Drive 又不是第一次这样了。",
                                "timestamp": "2026-05-22T09:10:00+00:00",
                                "display_name": "Nightfall",
                                "platform_message_id": "seed",
                            },
                            {
                                "summary": "Nightfall: <image>Google Drive 权限禁止的截图。</image>",
                                "timestamp": "2026-05-22T09:09:50+00:00",
                                "display_name": "Nightfall",
                                "platform_message_id": "previous",
                            },
                        ],
                        "packets": [
                            {
                                "summary": (
                                    "命中消息：Nightfall: Google Drive 又不是第一次这样了。"
                                    "；上一条：Nightfall: "
                                    "<image>Google Drive 权限禁止的截图。</image>"
                                ),
                                "seed": {
                                    "platform_message_id": "seed",
                                },
                                "relations": [
                                    {
                                        "relation_type": "previous_message",
                                        "row": {
                                            "platform_message_id": "previous",
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    evidence = result["conversation_evidence"][0]
    assert "Google Drive context was found." in evidence
    assert "命中消息" in evidence
    assert "上一条" in evidence
    assert "Google Drive 权限禁止的截图" in evidence
    assert "seed" not in evidence
    assert "previous" not in evidence


def test_project_known_facts_redacts_source_ids_from_public_conversation_summary() -> None:
    result = project_known_facts(
        [
            {
                "slot": "conversation",
                "agent": "conversation_evidence_agent",
                "resolved": True,
                "summary": (
                    "Tester global_user_id: "
                    "123e4567-e89b-12d3-a456-426614174000 sent the chart."
                ),
                "raw_result": {
                    "projection_payload": {
                        "rows": [
                            {
                                "summary": "Tester: chart sent.",
                                "timestamp": "2026-05-01T12:34:56.789000+00:00",
                                "display_name": "Tester",
                            }
                        ],
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    evidence = result["conversation_evidence"][0]
    assert "123e4567-e89b-12d3-a456-426614174000" not in evidence
    assert "global_user_id" not in evidence
    assert "[来源标识已省略]" not in evidence
    assert "Tester sent the chart." in evidence


def test_project_known_facts_includes_later_relevant_conversation_rows() -> None:
    rows = [
        {
            "summary": f"Speaker {index}: filler message {index}.",
            "display_name": f"Speaker {index}",
        }
        for index in range(8)
    ]
    rows.append(
        {
            "summary": "Nightfall: <image>oxygen sensor product page</image>",
            "display_name": "Nightfall",
        }
    )

    result = project_known_facts(
        [
            {
                "slot": "conversation",
                "agent": "conversation_evidence_agent",
                "resolved": True,
                "summary": "Nightfall sent the oxygen sensor image.",
                "raw_result": {
                    "projection_payload": {
                        "rows": rows,
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    evidence = result["conversation_evidence"][0]
    assert "Nightfall sent the oxygen sensor image." in evidence
    assert "<image>oxygen sensor product page</image>" in evidence


def test_project_known_facts_projects_formatted_recall_evidence() -> None:
    result = project_known_facts(
        [
            {
                "slot": "recall",
                "agent": "recall_agent",
                "resolved": True,
                "summary": "The active agreement is pickup at 9:30.",
                "raw_result": {
                    "selected_summary": "The active agreement is pickup at 9:30.",
                    "recall_type": "active_episode_agreement",
                    "primary_source": "conversation_progress",
                    "candidates": [
                        {
                            "source": "conversation_progress",
                            "claim": "Pickup at 9:30.",
                            "evidence_time": "2026-05-01T23:00:00+00:00",
                        }
                    ],
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    entry = result["recall_evidence"][0]
    assert entry["selected_summary"] == (
        '结论：The active agreement is pickup at 9:30.'
    )
    assert entry["evidence_summary"].startswith('上下文：\n- ')
    assert "Pickup at 9:30." in entry["evidence_summary"]
    assert "2026-05-02 11:00:00" in entry["evidence_summary"]
    assert '不确定性：无' in entry["evidence_summary"]
    assert "candidates" not in entry
    assert "2026-05-01T23:00:00+00:00" not in repr(entry)


def test_project_known_facts_keeps_raw_refs_trace_only() -> None:
    result = project_known_facts(
        [
            {
                "slot": "conversation",
                "agent": "conversation_evidence_agent",
                "resolved": True,
                "summary": "Chart was sent.",
                "raw_result": {
                    "projection_payload": {
                        "summaries": ["Tester: here is the chart."],
                        "rows": [
                            {
                                "summary": "Tester: here is the chart.",
                                "conversation_row_id": "row-1",
                                "platform_message_id": "message-1",
                            }
                        ],
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    public_payload = {
        key: value
        for key, value in result.items()
        if key != "supervisor_trace"
    }

    assert "row-1" not in repr(public_payload)
    assert "message-1" not in repr(public_payload)
    assert "row-1" in repr(result["supervisor_trace"])
    assert "message-1" in repr(result["supervisor_trace"])


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


def test_project_known_facts_replaces_bare_id_in_conversation_evidence() -> None:
    """Conversation evidence with bare provenance ID must use display name."""

    result = project_known_facts(
        [
            {
                "slot": "conversation",
                "agent": "conversation_evidence_agent",
                "resolved": True,
                "summary": "ID: 1445207392 asked about the schedule.",
                "raw_result": {
                    "projection_payload": {
                        "summaries": [
                            "ID: 1445207392: What time is the meeting?"
                        ],
                        "rows": [
                            {
                                "summary": "ID: 1445207392: What time is the meeting?",
                                "display_name": "Sublime",
                                "timestamp": "2026-06-01T08:30:00+00:00",
                            }
                        ],
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    evidence = result["conversation_evidence"][0]

    assert "1445207392" not in evidence
    assert "ID:" not in evidence
    assert "Sublime" in evidence
    assert "asked about the schedule" in evidence


def test_project_known_facts_replaces_bare_fullwidth_id_in_conversation_evidence() -> None:
    """Fullwidth colon ID label must also be stripped from conversation evidence."""

    result = project_known_facts(
        [
            {
                "slot": "conversation",
                "agent": "conversation_evidence_agent",
                "resolved": True,
                "summary": 'ID\uff1a1445207392 mentioned the plan.',
                "raw_result": {
                    "projection_payload": {
                        "rows": [
                            {
                                "summary": 'ID\uff1a1445207392: mentioned the plan.',
                                "display_name": "Sublime",
                            }
                        ],
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    evidence = result["conversation_evidence"][0]

    assert "1445207392" not in evidence
    assert "Sublime" in evidence


def test_project_known_facts_does_not_present_scope_global_user_id_as_evidence_content() -> None:
    """Memory evidence content must not present scope_global_user_id as the target."""

    result = project_known_facts(
        [
            {
                "slot": "memory",
                "agent": "memory_evidence_agent",
                "resolved": True,
                "summary": (
                    'scope_global_user_id=user-1 prefers jasmine tea.'
                ),
                "raw_result": {
                    "projection_payload": {
                        "memory_rows": [
                            {
                                "content": (
                                    'scope_global_user_id=user-1: '
                                    'prefers jasmine tea in the evening.'
                                ),
                                "source_system": "user_memory_units",
                                "scope_type": "user_continuity",
                                "scope_global_user_id": "user-1",
                            },
                        ],
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    entry = result["memory_evidence"][0]
    summary_text = entry.get("summary", "")
    content_text = entry.get("content", "")

    assert "scope_global_user_id" not in summary_text
    assert "scope_global_user_id" not in content_text
    assert "=user-1" not in summary_text
    assert "=user-1" not in content_text
    assert "jasmine tea" in content_text
