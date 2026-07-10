"""Person-context RAG3 subagent registration."""

from .source import SourceEvidenceSubagent

SUBAGENT = "person"
DESCRIPTION = "local person and profile context evidence"
SUPPORTED_ACTIONS = ("collect_person",)
OWNED_NODE_KINDS = ("person_context",)
DEFAULT_ACTION = "collect_person"


def create() -> SourceEvidenceSubagent:
    """Create the person-context subagent."""

    return SourceEvidenceSubagent(
        subagent=SUBAGENT,
        node_kinds=OWNED_NODE_KINDS,
        agent_name="person_context_agent",
    )
