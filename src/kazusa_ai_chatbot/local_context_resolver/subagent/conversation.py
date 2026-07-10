"""Conversation-evidence RAG3 subagent registration."""

from .source import SourceEvidenceSubagent

SUBAGENT = "conversation"
DESCRIPTION = "local conversation evidence for the active local-context objective"
SUPPORTED_ACTIONS = ("collect_conversation",)
OWNED_NODE_KINDS = ("conversation_evidence",)
DEFAULT_ACTION = "collect_conversation"


def create() -> SourceEvidenceSubagent:
    """Create the conversation-evidence subagent."""

    return SourceEvidenceSubagent(
        subagent=SUBAGENT,
        node_kinds=OWNED_NODE_KINDS,
        agent_name="conversation_evidence_agent",
    )
