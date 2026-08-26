"""Deterministic tests for the request-scoped prewarm checkpoint."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_resolver.contracts import (
    SharedMemoryPrewarmOutcomeV1,
)
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    bind_v2_attempt_ledger,
    clear_v2_shared_memory_prewarm_checkpoint,
    create_v2_attempt_ledger,
    record_v2_shared_memory_prewarm_checkpoint,
    reset_v2_attempt_ledger,
    snapshot_v2_shared_memory_prewarm_checkpoint,
)


def _outcome() -> SharedMemoryPrewarmOutcomeV1:
    """Build one valid completed prewarm outcome."""

    outcome: SharedMemoryPrewarmOutcomeV1 = {
        "schema_version": "shared_memory_prewarm_outcome.v1",
        "status": "completed",
        "reason_code": "shared_memory_ready",
        "attempted": True,
        "latency_ms": 4,
        "retrieved_shared_count": 1,
        "merged_shared_count": 0,
        "rag_result": {
            "answer": "",
            "user_image": {},
            "user_memory_unit_candidates": [],
            "character_image": {},
            "third_party_profiles": [],
            "memory_evidence": [{"summary": "shared"}],
            "recall_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "supervisor_trace": {},
        },
    }
    return outcome


def test_prewarm_checkpoint_is_deep_copied_scoped_to_graph_attempt_and_cleared() -> None:
    """Checkpoint state is isolated, attempt-scoped, and explicitly clearable."""

    ledger = create_v2_attempt_ledger("checkpoint-test")
    first_token = bind_v2_attempt_ledger(ledger, graph_attempt=1)
    try:
        source = _outcome()
        record_v2_shared_memory_prewarm_checkpoint(source)
        source["rag_result"]["memory_evidence"][0]["summary"] = "source mutation"  # type: ignore[index]

        snapshot = snapshot_v2_shared_memory_prewarm_checkpoint()
        assert snapshot is not None
        assert snapshot["rag_result"]["memory_evidence"][0]["summary"] == (
            "shared"
        )
        snapshot["rag_result"]["memory_evidence"][0]["summary"] = (  # type: ignore[index]
            "snapshot mutation"
        )
        fresh_snapshot = snapshot_v2_shared_memory_prewarm_checkpoint()
        assert fresh_snapshot is not None
        assert fresh_snapshot["rag_result"]["memory_evidence"][0][
            "summary"
        ] == "shared"

        second_token = bind_v2_attempt_ledger(ledger, graph_attempt=2)
        try:
            assert snapshot_v2_shared_memory_prewarm_checkpoint() is None
            record_v2_shared_memory_prewarm_checkpoint(_outcome())
            assert snapshot_v2_shared_memory_prewarm_checkpoint() is not None
            clear_v2_shared_memory_prewarm_checkpoint()
            assert snapshot_v2_shared_memory_prewarm_checkpoint() is None
        finally:
            reset_v2_attempt_ledger(second_token)
    finally:
        reset_v2_attempt_ledger(first_token)
