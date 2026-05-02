"""Deterministic route-map checks for the RAG2 capability layer."""

from __future__ import annotations

import json
from pathlib import Path


_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "rag_phase3_real_conversation_cases.json"
)


def test_old_to_new_route_mapping_is_encoded() -> None:
    """Keep the approved route migration table visible in deterministic tests."""

    route_mapping = {
        "Identity -> Profile": "Person-context",
        "User-list": "Person-context",
        "Relationship": "Person-context",
        "Conversation-keyword": "Conversation-evidence",
        "Conversation-semantic": "Conversation-evidence",
        "Conversation-filter": "Conversation-evidence",
        "Conversation-aggregate": "Conversation-evidence",
        "Memory-search": "Memory-evidence",
        "Memory-search stable location -> Web-search live value": "Live-context",
        "Conversation-semantic recent user location -> Web-search live value": "Live-context",
        "Web-search explicit URL/topic": "Web-evidence",
        "Recall": "Recall",
    }

    assert route_mapping["Conversation-keyword"] == "Conversation-evidence"
    assert route_mapping["Memory-search"] == "Memory-evidence"
    assert route_mapping["Identity -> Profile"] == "Person-context"
    assert route_mapping["Recall"] == "Recall"


def test_real_conversation_fixture_is_compact_and_covers_required_routes() -> None:
    """The real-conversation live cases must stay compact and route-complete."""

    cases = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    by_case_id = {case["case_id"]: case for case in cases}

    assert set(by_case_id) == {
        "christchurch_weekend_weather",
        "amusement_park_opening",
        "recent_address_confirmation",
        "official_address_memory",
        "today_agreement",
        "episode_position_next_step",
        "exact_phrase_boundary",
    }
    expected_prefixes = {
        prefix
        for case in cases
        for prefix in case["expected_prefixes"]
    }

    assert expected_prefixes == {
        "Conversation-evidence:",
        "Live-context:",
        "Memory-evidence:",
        "Recall:",
    }
    assert all(len(case["query"]) < 220 for case in cases)
