"""Resolver-local public evidence subagent registration."""

from __future__ import annotations

from kazusa_ai_chatbot.complex_task_resolver.contracts import (
    ComplexTaskSubagentV1,
)
from kazusa_ai_chatbot.complex_task_resolver.subagents import (
    ComplexTaskEvidenceSubagent,
)

SUBAGENT = "evidence"
DESCRIPTION = (
    "public source investigation through the existing evidence facility. Use "
    "it when the active node needs current, external, source-bound product, "
    "documentation, release, support, benchmark, or availability facts."
)
SUPPORTED_ACTIONS = ("collect_evidence",)
OWNED_NODE_KINDS = ("evidence_need",)
DEFAULT_ACTION = "collect_evidence"


def create() -> ComplexTaskSubagentV1:
    """Return the production evidence subagent."""

    return ComplexTaskEvidenceSubagent()
