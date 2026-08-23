"""Executable documentation contracts for the standalone resolver."""

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


def test_architecture_declares_standalone_first_pass_and_deferred_bigbang() -> None:
    """The governing decision keeps the first pass standalone."""

    text = _normalized(ARCHITECTURE)

    assert "first-pass boundary: additive standalone runtime" in text
    assert "later transition: a separate big-bang plan" in text
    assert "no inbound edge from cognition" in text
    assert "phase 2: later big-bang transition" in text


def test_architecture_requires_json_for_resolver_authored_semantic_envelopes() -> None:
    """The governing decision excludes free-form semantic control messages."""

    text = _normalized(ARCHITECTURE)

    assert "every non-empty resolver-authored semantic textual payload" in text
    assert "is a json object" in text
    assert "xml and pseudo-xml prompt frames are outside" in text
    assert "controller never interprets assistant prose as an action" in text


def test_architecture_declares_opaque_reasoning_replay_and_atomic_compaction() -> None:
    """Reasoning replay and history compaction preserve one atomic exchange."""

    text = _normalized(ARCHITECTURE)

    assert "opaque assistant reasoning channel" in text
    assert "reasoning_content" in text
    assert "empty field" in text
    assert "tool-call-free reasoning" in text
    assert "atomic" in text
    assert "reasoning" in text
    assert "tool result" in text


def test_architecture_declares_non_recursive_same_runtime_subagent() -> None:
    """Children use the same runtime and omit recursive delegation."""

    text = _normalized(ARCHITECTURE)

    assert "new instance of the same resolver runtime" in text
    assert "isolated session" in text
    assert "registry omits" in text
    assert "run_subagent" in text
    assert "fixing delegation depth at one" in text
    assert "parent receives the bounded child result" in text
