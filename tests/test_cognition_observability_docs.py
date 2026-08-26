"""Documentation tests for the cognition-observation ownership boundary."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    """Read one repository document as UTF-8 text."""

    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def test_icd_and_runtime_docs_name_one_brain_service_contract_owner() -> None:
    """The ICD should state the complete Brain-owned v1 boundary."""

    icd = _read("docs/architecture/cognition_observability_icd.md")
    contracts = _read("docs/architecture/cognition_contracts_design.md")
    brain_readme = _read("src/kazusa_ai_chatbot/brain_service/README.md")
    console_readme = _read("src/control_console/README.md")

    required_icd_clauses = (
        "The Brain service is the sole schema and projection owner",
        "producer",
        "publisher",
        "transport",
        "consumer",
        "CognitionRunObservationV1",
        "CognitionObservationCorrelationV1",
        "CognitionObservationDisclosureV1",
        "CognitionObservationSectionV1",
        "CognitionObservationFieldV1",
        "CognitionObservationRecordV1",
        "CognitionObservationNodeV1",
        "CognitionObservationEdgeV1",
        "run_kind",
        "generated_at",
        "completed",
        "empty",
        "skipped",
        "failed",
        "partial",
        "not_reported",
        "input.turn",
        "evidence.shared_memory_prewarm",
        "context.conversation_progress",
        "context.public_group_scene",
        "self.source",
        "self.route",
        "self.consolidation",
        "worker_unresolved",
        "worker_contract_invalid",
        "projection_failed",
        "no_shared_memory",
        "worker_error",
        "shared_memory_ready",
        "shared_memory_merged",
        "empty_query_after_character_mention",
        "not_first_cycle",
        "unsupported_episode",
        "approved_cognition_observation.v1",
        "prompt",
        "raw model output",
        "embeddings",
        "raw messages",
        "message envelopes",
        "database identifiers",
        "adapter identifiers",
        "action parameters",
        "handler metadata",
        "worker error text",
        "131072",
        "item_01",
        "item_24",
        "source-to-wire",
        "reported_record_count",
        "displayed_record_count",
        "truncated",
        "sequence",
        "reference",
        "ConsoleCognitionObservationView",
        "available",
        "unavailable",
        "invalid",
        "live_turn",
        "self_cognition",
        "breaking semantic reinterpretation",
        "new major schema",
        "Producer-approved additive sections",
        "Overview, Debug, and Self",
        "Cancellation publishes no",
    )
    for clause in required_icd_clauses:
        assert clause in icd, clause

    assert "cognition_run_observation.v1" in contracts
    assert "cognition_run_observation.v1" in brain_readme
    assert "Brain service" in icd
    assert "Brain service" in brain_readme
    assert "validation-only" in console_readme
    assert "projection owner" not in console_readme.casefold()
    assert "schema owner" not in console_readme.casefold()
    for stale_clause in (
        "The graph is a process-local semantic projection.",
        "Canonical cognition graph rows may include",
    ):
        assert stale_clause not in brain_readme


def test_process_local_observation_and_future_persisted_chain_run_are_distinct() -> None:
    """Architecture docs should keep semantic snapshots separate from persistence."""

    architecture = _read(
        "docs/architecture/cognition_v3_hybrid_agentic_loop_architecture.md"
    )

    for clause in (
        "process-local `cognition_run_observation.v1` snapshot",
        "historical `cognition_chain_run.v2` persistence record",
        "bounded operator observation",
        "observation is not historical persistence",
        "Brain service producer catalog",
        "typed prewarm",
        "public-group-scene sections",
        "validation-only mode",
        "generic section renderer",
        "additive producer sections",
        "Cancellation publishes no terminal observation",
    ):
        assert clause in architecture, clause


def test_howto_documents_canonical_observation_and_browser_checks() -> None:
    """Operator guidance should name the canonical and browser verification paths."""

    howto = _read("docs/HOWTO.md")

    for clause in (
        "cognition_run_observation.v1",
        "tests/control_console_e2e",
        "--collect-only",
        "in-app browser",
        "Playwright",
        "Overview",
        "Debug",
        "Self Latest",
        "HTML escaping",
        "zero page or console error logs",
        "live LLM cases",
    ):
        assert clause in howto, clause
    assert "latest_cognition_graph" not in howto


def test_runtime_readmes_document_prewarm_and_observation_carriers() -> None:
    """Runtime subsystem docs should describe the typed carrier boundaries."""

    resolver_readme = _read("src/kazusa_ai_chatbot/cognition_resolver/README.md")
    nodes_readme = _read("src/kazusa_ai_chatbot/nodes/README.md")

    for clause in (
        "SharedMemoryPrewarmOutcomeV1",
        "worker_unresolved",
        "worker_contract_invalid",
        "projection_failed",
        "no_shared_memory",
        "worker_error",
        "shared_memory_ready",
        "shared_memory_merged",
        "empty_query_after_character_mention",
        "not_first_cycle",
        "unsupported_episode",
        "latency_ms",
        "retrieved_count",
        "merged_count",
        "current graph attempt",
        "checkpoint",
        "cognition_run_observation.v1",
    ):
        assert clause in resolver_readme, clause
    for clause in (
        "public_group_scene_context",
        "public_group_scene_projection_status",
        "projection_unavailable",
        "cognition_run_observation.v1",
        "typed shared-memory prewarm outcome",
    ):
        assert clause in nodes_readme, clause
