"""Supplied-external evidence RAG3 subagent registration."""

from .source import SourceEvidenceSubagent

SUBAGENT = "external"
DESCRIPTION = "caller-supplied external evidence already present in local context"
SUPPORTED_ACTIONS = ("collect_supplied_external",)
OWNED_NODE_KINDS = ("external_evidence",)
DEFAULT_ACTION = "collect_supplied_external"


def create() -> SourceEvidenceSubagent:
    """Create the supplied-external evidence subagent."""

    return SourceEvidenceSubagent(subagent=SUBAGENT, node_kinds=OWNED_NODE_KINDS)
