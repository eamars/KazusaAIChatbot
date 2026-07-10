"""Live-context RAG3 subagent registration."""

from .source import SourceEvidenceSubagent

SUBAGENT = "live_context"
DESCRIPTION = "current local time and supplied live conversation context"
SUPPORTED_ACTIONS = ("collect_live_context",)
OWNED_NODE_KINDS = ("live_context",)
DEFAULT_ACTION = "collect_live_context"


def create() -> SourceEvidenceSubagent:
    """Create the live-context subagent."""

    return SourceEvidenceSubagent(subagent=SUBAGENT, node_kinds=OWNED_NODE_KINDS)
