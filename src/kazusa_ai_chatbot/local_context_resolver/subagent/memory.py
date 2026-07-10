"""Memory and scoped-memory RAG3 subagent registration."""

from .source import SourceEvidenceSubagent

SUBAGENT = "memory"
DESCRIPTION = "local shared or user-scoped memory evidence"
SUPPORTED_ACTIONS = ("collect_memory",)
OWNED_NODE_KINDS = ("memory_evidence", "scoped_memory")
DEFAULT_ACTION = "collect_memory"


def create() -> SourceEvidenceSubagent:
    """Create the memory-evidence subagent."""

    return SourceEvidenceSubagent(
        subagent=SUBAGENT,
        node_kinds=OWNED_NODE_KINDS,
        agent_name="memory_evidence_agent",
    )
