"""V2 resolver-evidence re-entry tests for persona cognition."""

from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    project_resolver_observation_for_cognition,
)


def test_resolver_observation_reenters_as_typed_evidence_only() -> None:
    """Resolver output contributes evidence and has no cognition-state authority."""

    evidence, direct_facts = project_resolver_observation_for_cognition(
        {
            "observation_id": "resolver-observation-1",
            "capability": "local_context_recall",
            "semantic_summary": "A prior promise is relevant.",
            "replacement_state": {"forbidden": True},
        },
        occurred_at="2026-06-08T00:00:00Z",
    )

    assert evidence["evidence_ref"]["source_kind"] == "resolver_observation"
    assert evidence["evidence_ref"]["source_id"] == "resolver-observation-1"
    assert direct_facts == []
    assert "replacement_state" not in evidence
