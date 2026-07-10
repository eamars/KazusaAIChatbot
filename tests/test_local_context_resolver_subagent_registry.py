"""Contract coverage for RAG3 resolver-local subagent registration."""

from kazusa_ai_chatbot.local_context_resolver.subagent import (
    get_subagent_registry,
)


def test_registry_owns_each_source_backed_node_kind_once() -> None:
    """Expose one canonical subagent owner for every planned source kind."""

    registry = get_subagent_registry()

    assert set(registry) == {
        "conversation",
        "external",
        "live_context",
        "media",
        "memory",
        "person",
        "recall",
    }
    owned_node_kinds = [
        node_kind
        for subagent in registry.values()
        for node_kind in subagent.OWNED_NODE_KINDS
    ]
    assert len(owned_node_kinds) == len(set(owned_node_kinds))
    assert set(owned_node_kinds) == {
        "conversation_evidence",
        "external_evidence",
        "live_context",
        "memory_evidence",
        "person_context",
        "recall_evidence",
        "scoped_memory",
        "current_turn_media",
        "recent_media",
    }
