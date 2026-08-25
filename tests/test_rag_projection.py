"""Tests for projecting RAG2 known facts into persona context."""

from __future__ import annotations

import json
import logging

from kazusa_ai_chatbot.cognition_shared.contracts import (
    _validate_evidence_rows,
)
from kazusa_ai_chatbot.consolidation.character_self_guidance import (
    _memory_document as _conversation_memory_document,
)
from kazusa_ai_chatbot.db.user_memory_units import (
    build_user_memory_unit_doc,
)
from kazusa_ai_chatbot.memory_evolution.reset import _seed_document
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import _rag_evidence
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_projection import (
    classify_typed_memory_row,
    project_known_facts,
)
from kazusa_ai_chatbot.rag.evidence_formatting import (
    recover_public_rag_evidence_prompt_safe,
)
from kazusa_ai_chatbot.rag.memory_evidence.workers.user_memory import (
    _project_row,
)
from kazusa_ai_chatbot.reflection_cycle.promotion import (
    _memory_document_for_decision,
)


def _typed_shared_memory_row(
    *,
    row_id: str,
    memory_unit_id: str,
    memory_type: str,
    source_kind: str,
    authority: str,
    content: str,
) -> dict[str, object]:
    """Build one complete learned-memory row in the writer's raw shape."""

    return {
        "_id": row_id,
        "memory_unit_id": memory_unit_id,
        "memory_name": memory_type,
        "content": content,
        "timestamp": "2026-06-01T00:00:00+00:00",
        "memory_type": memory_type,
        "source_kind": source_kind,
        "source_global_user_id": "",
        "authority": authority,
        "status": "active",
        "privacy_review": {
            "global_applicability": "global",
            "target_specific_meaning_removed": True,
            "affects_identity_or_boundaries": False,
            "private_detail_risk": "low",
            "user_details_removed": True,
            "boundary_assessment": "deidentified global meaning",
            "reviewer": "automated_llm",
        },
    }


def _user_memory_writer_row(
    *,
    user_id: str,
    unit_id: str,
    fact: str = "A current-user continuity fact.",
    storage_timestamp_utc: str = "2026-06-01T00:00:00+00:00",
) -> dict[str, object]:
    """Build a current-user row through the production user-memory writer."""

    writer_row = dict(build_user_memory_unit_doc(
        user_id,
        {
            "unit_id": unit_id,
            "unit_type": "objective_fact",
            "fact": fact,
            "subjective_appraisal": "A bounded continuity appraisal.",
            "relationship_signal": "A bounded continuity signal.",
        },
        storage_timestamp_utc=storage_timestamp_utc,
        unit_id=unit_id,
    ))
    projected_row = _project_row(writer_row, user_id)
    return projected_row


def _learned_privacy_review() -> dict[str, object]:
    """Build the complete certificate emitted by learned-memory writers."""

    return {
        "global_applicability": "global",
        "target_specific_meaning_removed": True,
        "affects_identity_or_boundaries": False,
        "private_detail_risk": "low",
        "user_details_removed": True,
        "boundary_assessment": "deidentified global meaning",
        "reviewer": "automated_llm",
    }


def _canonical_shared_writer_row(
    *,
    row_id: str,
    memory_unit_id: str,
    memory_type: str,
    source_kind: str,
    authority: str,
    content: str,
) -> dict[str, object]:
    """Build a shared row through its canonical production writer."""

    if authority == "conversation_accepted":
        row = _conversation_memory_document(
            memory_name=memory_type,
            content=content,
            source_refs=[{"conversation_history_id": row_id}],
            storage_timestamp_utc="2026-06-01T00:00:00+00:00",
            reviewer_certificate={
                **_learned_privacy_review(),
                "reason": "deidentified global meaning",
            },
        )
    elif authority == "reflection_promoted":
        lane = "lore" if memory_type == "fact" else "self_guidance"
        row = _memory_document_for_decision(
            decision={
                "lane": lane,
                "sanitized_memory_name": memory_type,
                "sanitized_content": content,
                "privacy_review": _learned_privacy_review(),
                "evidence_refs": [{"reflection_run_id": row_id}],
            },
            character_local_date="2026-06-01",
            global_run_id=row_id,
            source_unit_ids=[],
            source_lineage_ids=[],
            mutation_action="insert",
        )
    elif authority == "seed":
        row = _seed_document(
            {
                "memory_name": memory_type,
                "content": content,
                "source_global_user_id": "",
                "memory_type": memory_type,
                "source_kind": source_kind,
                "status": "active",
                "confidence_note": "test fixture",
                "expiry_timestamp": None,
            },
            storage_timestamp_utc="2026-06-01T00:00:00+00:00",
            updated_at="2026-06-01T00:00:00+00:00",
        )
    else:
        row = _typed_shared_memory_row(
            row_id=row_id,
            memory_unit_id=memory_unit_id,
            memory_type=memory_type,
            source_kind=source_kind,
            authority=authority,
            content=content,
        )
    row = dict(row)
    row["_id"] = row_id
    if row.get("memory_unit_id") != memory_unit_id:
        row["memory_unit_id"] = memory_unit_id
    return row


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
                "raw_result": [_typed_shared_memory_row(
                    row_id="grouped-memory-row",
                    memory_unit_id="grouped-memory-unit",
                    memory_type="fact",
                    source_kind="conversation_extracted",
                    authority="conversation_accepted",
                    content="A" * 20,
                )],
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
        {
            "slot": "memory",
            "agent": "persistent_memory_search_agent",
            "resolved": True,
            "projection_diagnostics": [
                {"reason": "learned memory metadata is incomplete"}
            ],
        },
        {"slot": "web", "agent": "web_agent3", "resolved": True},
    ]
    assert result["third_party_profiles"] == []
    assert result["memory_evidence"] == []
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
                                _typed_shared_memory_row(
                                    row_id="capability-memory-row",
                                    memory_unit_id="capability-memory-unit",
                                    memory_type="fact",
                                    source_kind="seeded_manual",
                                    authority="seed",
                                    content=(
                                        "official address is 123 Example Street"
                                    ),
                                )
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
                            _typed_shared_memory_row(
                                row_id="unsafe-memory-row",
                                memory_unit_id="unsafe-memory-unit",
                                memory_type="fact",
                                source_kind="seeded_manual",
                                authority="seed",
                                content="[CQ:image,file=abc]",
                            ),
                            _typed_shared_memory_row(
                                row_id="safe-memory-row",
                                memory_unit_id="safe-memory-unit",
                                memory_type="fact",
                                source_kind="seeded_manual",
                                authority="seed",
                                content="User prefers tea.",
                            ),
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
                            _user_memory_writer_row(
                                user_id="user-1",
                                unit_id="unit-7",
                                fact="冰淇淋摊老板是千纱的初中学姐。",
                                storage_timestamp_utc=(
                                    "2026-05-03T00:00:00+00:00"
                                ),
                            ),
                            _typed_shared_memory_row(
                                row_id="curated-address-row",
                                memory_unit_id="curated-address-unit",
                                memory_type="fact",
                                source_kind="seeded_manual",
                                authority="seed",
                                content=(
                                    "The active character's official address "
                                    "is 123 Example Street."
                                ),
                            ),
                        ],
                    }
                },
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    assert len(result["memory_evidence"]) == 2
    entry = result["memory_evidence"][0]
    assert entry["summary"] == '结论：scoped continuity summary'
    assert entry["content"].startswith('上下文：')
    assert "冰淇淋摊老板是千纱的初中学姐。" in entry["content"]
    assert entry["source_system"] == "user_memory_units"
    assert entry["scope_type"] == "user_continuity"
    assert entry["scope_global_user_id"] == "user-1"
    assert entry["authority"] == "scoped_continuity"
    assert entry["truth_status"] == "character_lore_or_interaction_continuity"
    assert entry["origin"] == "consolidated_interaction"
    curated_entry = result["memory_evidence"][1]
    assert "The active character's official address is 123 Example Street." in curated_entry["content"]
    assert curated_entry["source_kind"] == "seeded_manual"
    candidate = result["user_memory_unit_candidates"][0]
    assert candidate["unit_id"] == "unit-7"
    assert candidate["global_user_id"] == "user-1"
    assert candidate["unit_type"] == "objective_fact"
    assert candidate["fact"] == "冰淇淋摊老板是千纱的初中学姐。"
    assert candidate["status"] == "active"


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
                            _user_memory_writer_row(
                                user_id="user-1",
                                unit_id="formatted-unit",
                                fact="User prefers tea during late sessions.",
                                storage_timestamp_utc=(
                                    "2026-05-01T12:34:56.789000+00:00"
                                ),
                            )
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
                            _user_memory_writer_row(
                                user_id="user-1",
                                unit_id="scope-safe-unit",
                                fact=(
                                    'scope_global_user_id=user-1: '
                                    'prefers jasmine tea in the evening.'
                                ),
                            ),
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


def test_shared_memory_projection_partitions_fact_and_self_guidance_with_typed_metadata() -> None:
    """Writer-shaped rows retain typed authority through cognition safely."""

    fact_row = _canonical_shared_writer_row(
        row_id="fact-row-1",
        memory_unit_id="fact-unit-1",
        memory_type="fact",
        source_kind="reflection_inferred",
        authority="reflection_promoted",
        content="A validated world fact.",
    )
    self_guidance_row = _canonical_shared_writer_row(
        row_id="guidance-row-1",
        memory_unit_id="guidance-unit-1",
        memory_type="defense_rule",
        source_kind="conversation_extracted",
        authority="conversation_accepted",
        content="A certified character guidance rule.",
    )
    curated_fact = _canonical_shared_writer_row(
        row_id="curated-fact-row",
        memory_unit_id="curated-fact-unit",
        memory_type="fact",
        source_kind="seeded_manual",
        authority="seed",
        content="A curated world fact.",
    )
    curated_external = _canonical_shared_writer_row(
        row_id="curated-external-row",
        memory_unit_id="curated-external-unit",
        memory_type="fact",
        source_kind="external_imported",
        authority="seed",
        content="A curated imported fact.",
    )
    curated_defense = _typed_shared_memory_row(
        row_id="curated-defense-row",
        memory_unit_id="curated-defense-unit",
        memory_type="defense_rule",
        source_kind="external_imported",
        authority="manual",
        content="A curated defense rule.",
    )
    current_user_row = _user_memory_writer_row(
        user_id="user-1",
        unit_id="current-user-unit",
    )
    raw_user_row = dict(build_user_memory_unit_doc(
        "user-1",
        {
            "unit_id": "raw-user-unit",
            "unit_type": "objective_fact",
            "fact": "An unprojected current-user row.",
            "subjective_appraisal": "An unprojected appraisal.",
            "relationship_signal": "An unprojected signal.",
        },
        storage_timestamp_utc="2026-06-01T00:00:00+00:00",
        unit_id="raw-user-unit",
    ))
    other_user_row = _user_memory_writer_row(
        user_id="user-2",
        unit_id="other-user-unit",
    )
    incomplete_certificate = _typed_shared_memory_row(
        row_id="incomplete-row",
        memory_unit_id="incomplete-unit",
        memory_type="fact",
        source_kind="conversation_extracted",
        authority="conversation_accepted",
        content="An incomplete learned row.",
    )
    del incomplete_certificate["privacy_review"]["boundary_assessment"]
    untyped_row = {
        "_id": "untyped-row",
        "memory_type": "fact",
        "content": "An untyped row.",
    }
    rows = [
        fact_row,
        self_guidance_row,
        curated_fact,
        curated_external,
        curated_defense,
        current_user_row,
        raw_user_row,
        other_user_row,
        incomplete_certificate,
        untyped_row,
    ]

    assert classify_typed_memory_row(
        fact_row,
        current_user_id="user-1",
    ) == ("character_world_context", "validated learned global memory")
    assert classify_typed_memory_row(
        current_user_row,
        current_user_id="user-1",
    )[0] == "participant_continuity"
    assert classify_typed_memory_row(
        other_user_row,
        current_user_id="user-1",
    )[0] is None
    assert classify_typed_memory_row(
        raw_user_row,
        current_user_id="user-1",
    )[0] is None

    result = project_known_facts(
        [
            {
                "slot": "memory",
                "agent": "persistent_memory_search_agent",
                "resolved": True,
                "summary": "typed shared memory",
                "raw_result": rows,
            }
        ],
        current_user_id="user-1",
        character_user_id="character-1",
    )

    entries = result["memory_evidence"]
    assert [entry.get("memory_type", entry.get("unit_type")) for entry in entries] == [
        "fact",
        "defense_rule",
        "fact",
        "fact",
        "defense_rule",
        "objective_fact",
    ]
    assert [entry["authority"] for entry in entries] == [
        "reflection_promoted",
        "conversation_accepted",
        "seed",
        "seed",
        "manual",
        "scoped_continuity",
    ]
    assert entries[0]["scope_type"] == "global"
    assert entries[1]["privacy_review"] == self_guidance_row["privacy_review"]
    assert entries[5]["scope_global_user_id"] == "user-1"
    assert "source_global_user_id" not in repr(entries)
    assert "A validated world fact." in entries[0]["content"]
    assert "A certified character guidance rule." in entries[1]["content"]
    assert "A curated imported fact." in entries[3]["content"]

    safe_result, incidents = recover_public_rag_evidence_prompt_safe(result)
    assert incidents == []
    cognition_rows = _rag_evidence(
        safe_result,
        "2026-06-08T00:00:00Z",
        current_user_id="user-1",
    )
    for index, row in enumerate(cognition_rows, start=1):
        row["evidence_handle"] = f"e{index}"
    _validate_evidence_rows(cognition_rows)
    assert [row["authority"] for row in cognition_rows] == [
        "character_world_context",
        "conditional_character_guidance",
        "character_world_context",
        "character_world_context",
        "conditional_character_guidance",
        "participant_continuity",
    ]
    assert cognition_rows[1]["evidence_ref"]["source_id"] == (
        "promoted-memory:self_guidance:guidance-unit-1"
    )
    assert cognition_rows[5]["evidence_ref"]["source_id"] == (
        "promoted-memory:current_user_continuity:current-user-unit"
    )
    assert cognition_rows[5]["memory_metadata"]["unit_type"] == (
        "objective_fact"
    )
    assert cognition_rows[5]["memory_metadata"]["scope_global_user_id"] == (
        "user-1"
    )
    diagnostics = result["supervisor_trace"]["dispatched"][0][
        "projection_diagnostics"
    ]
    assert {item["source_id"] for item in diagnostics} == {
        "other-user-unit",
        "raw-user-unit",
        "incomplete-row",
        "untyped-row",
    }

    public_payload = {
        key: value
        for key, value in result.items()
        if key != "supervisor_trace"
    }
    assert "fact-row-1" not in repr(public_payload)
    assert "guidance-row-1" not in repr(public_payload)
    trace = repr(result["supervisor_trace"])
    assert "fact-row-1" in trace
    assert "guidance-row-1" in trace
    assert "source_global_user_id" not in trace
    assert "privacy_review" not in trace
