"""Resolver-local deterministic arithmetic subagent registration."""

from __future__ import annotations

from kazusa_ai_chatbot.complex_task_resolver.algorithmic import (
    AlgorithmicSubagent,
)
from kazusa_ai_chatbot.complex_task_resolver.contracts import (
    ComplexTaskSubagentV1,
)

SUBAGENT = "algorithmic"
DESCRIPTION = (
    "deterministic numeric calculation only. Use it after the node resolver "
    "has interpreted units, selected the formula, and prepared a numeric "
    "expression with visible operand provenance."
)
SUPPORTED_ACTIONS = ("evaluate_expression", "missing_expression")
OWNED_NODE_KINDS = ("algorithmic_task",)
DEFAULT_ACTION = "missing_expression"


def create() -> ComplexTaskSubagentV1:
    """Return the production arithmetic subagent."""

    return AlgorithmicSubagent()
