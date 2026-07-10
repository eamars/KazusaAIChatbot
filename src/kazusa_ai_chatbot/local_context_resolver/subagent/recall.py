"""Recall-evidence RAG3 subagent registration."""

from .source import SourceEvidenceSubagent

SUBAGENT = "recall"
DESCRIPTION = "local recall evidence for a prior relationship or conversation fact"
SUPPORTED_ACTIONS = ("collect_recall",)
OWNED_NODE_KINDS = ("recall_evidence",)
DEFAULT_ACTION = "collect_recall"


def create() -> SourceEvidenceSubagent:
    """Create the recall-evidence subagent."""

    return SourceEvidenceSubagent(
        subagent=SUBAGENT,
        node_kinds=OWNED_NODE_KINDS,
        agent_name="recall_agent",
    )
