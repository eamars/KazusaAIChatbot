"""Executable documentation contracts for the agentic resolver."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_README = (
    REPOSITORY_ROOT / "src" / "agentic_resolver" / "README.md"
)
LLM_INTERFACE_README = (
    REPOSITORY_ROOT
    / "src"
    / "kazusa_ai_chatbot"
    / "llm_interface"
    / "README.md"
)
ARCHITECTURE = (
    REPOSITORY_ROOT
    / "docs"
    / "architecture"
    / "agentic_resolver_architecture.md"
)


def _normalized(path: Path) -> str:
    """Return lower-cased single-space documentation text."""

    text = path.read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())
    return normalized


def test_module_readme_documents_public_runtime_and_forbidden_workflow_edges() -> None:
    """The module ICD fixes direct construction and absent inbound workflows."""

    text = _normalized(MODULE_README)

    assert "agenticresolverruntime.resolve" in text
    assert "standalone direct python construction" in text
    assert "no import, registration, selection, or call edge" in text
    for boundary in (
        "cognition",
        "brain service",
        "task resolution",
        "accepted tasks",
        "background work",
    ):
        assert boundary in text
    assert "later approved big-bang plan" in text


def test_module_readme_requires_thinking_stream_and_opaque_reasoning() -> None:
    """The module ICD makes streaming thinking mandatory and non-semantic."""

    text = _normalized(MODULE_README)

    assert "every root and child model step uses" in text
    assert "agenticmodelclient.astream" in text
    assert "thinking enabled" in text
    assert "reasoning is opaque assistant transport state" in text
    assert "kept separate from semantic json" in text
    assert "partial tool calls never reach a tool implementation" in text


def test_llm_interface_readme_documents_additive_native_tool_stream_contract() -> None:
    """The shared ICD records the additive stream and ordinary preservation."""

    text = _normalized(LLM_INTERFACE_README)

    assert "additive native-tool stream contract" in text
    assert "astream_tools" in text
    assert "does not replace or alter" in text
    assert "ordinary calls retain their existing" in text
    assert "canonical digest of the visible tool-schema roster" in text
    assert "only before the wrapper emits its first chunk" in text
    assert "once any chunk has been yielded" in text


def test_architecture_preserves_brain_call_and_execution_ownership() -> None:
    """The renewed resolver stays behind the existing brain-owned boundary."""

    text = _normalized(ARCHITECTURE)

    assert "the brain action selector remains the caller" in text
    assert "task_resolution_request remains the request surface" in text
    assert (
        "the resolver never decides whether a job belongs in the background"
        in text
    )
    assert "big-bang replacement inside the resolution layer" in text
    assert "the same durable session is checkpointed" in text


def test_architecture_exposes_complete_leaf_tool_catalog_without_hidden_dags() -> None:
    """Eligible base interfaces become leaf tools with explicit exclusions."""

    text = _normalized(ARCHITECTURE)

    assert "all eligible resolution-facing base-level interfaces" in text
    assert "a catalog completeness test fails" in text
    assert "conversation_search" in text
    assert "memory_search" in text
    assert "web_search" in text
    assert (
        "no live tool hides a task, rag, complex, or web orchestration dag"
        in text
    )


def test_architecture_adopts_recorded_harness_call_result_pairing() -> None:
    """Harness-derived sessions persist exact call/result correlations."""

    text = _normalized(ARCHITECTURE)

    assert "b150a551b8d465e31e418e1b2eaf5e79bbb7d28e" in text
    assert "bash-tool/session.jsonl" in text
    assert "skill-load/session.jsonl" in text
    assert "tool/call" in text
    assert "tool/result" in text
    assert "paired by the same call id" in text
    assert "a call is persisted before dispatch" in text
    assert "tool_outcome_unknown" in text


def test_architecture_governs_experience_derived_skill_promotion() -> None:
    """Past experience proposes skills but cannot activate them directly."""

    text = _normalized(ARCHITECTURE)

    assert "skillcandidate" in text
    assert "held-out replay" in text
    assert "independent or human approval" in text
    assert "the agent may propose a skillcandidate" in text
    assert "it cannot silently activate one" in text
    assert "skills teach procedure" in text


def test_architecture_includes_kazusa_memory_digging_examples() -> None:
    """The target describes iterative memory and active-recall use cases."""

    text = _normalized(ARCHITECTURE)

    assert "blue comet" in text
    assert "conversation_list around the strongest matches" in text
    assert "memory_read for the selected memory and provenance" in text
    assert "active_recall for current commitments" in text
    assert "cognition decides how kazusa emotionally interprets" in text
