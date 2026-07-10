"""L2d reachability coverage for local conversation images."""

from kazusa_ai_chatbot.cognition_chain_core.action_selection import (
    build_action_selection_payload,
)


def test_local_context_recall_mentions_conversation_images() -> None:
    """Retain one capability while making local image evidence selectable."""

    payload = build_action_selection_payload({}, {})
    affordances = payload["capabilities"]["resolver_affordances"]
    local_context = next(
        row for row in affordances
        if row["capability_kind"] == "local_context_recall"
    )

    assert "conversation image" in local_context["semantic_input_summary"]


def test_action_selection_projects_visual_observations_without_raw_media() -> None:
    """Expose answered image evidence to cognition without trusted cache fields."""

    payload = build_action_selection_payload({
        "rag_result": {
            "media_evidence": [{
                "alias": "recent_media_1",
                "description": "The teal square is directly below the white circle.",
                "recency": "recent conversation",
                "evidence_boundary_notes": ["Visible image evidence only."],
                "cache_ref": "trusted-cache-ref",
                "base64_data": "not-prompt-safe",
            }],
        },
    }, {})

    media_evidence = payload["evidence"]["media_evidence"]
    assert media_evidence == [{
        "visual_observation": "The teal square is directly below the white circle.",
        "recency": "recent conversation",
        "evidence_boundary_notes": ["Visible image evidence only."],
    }]
    assert "trusted-cache-ref" not in str(payload)
    assert "not-prompt-safe" not in str(payload)
