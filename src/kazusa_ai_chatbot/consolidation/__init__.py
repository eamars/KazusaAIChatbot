"""Public consolidation subsystem interface."""

from __future__ import annotations

import importlib

from kazusa_ai_chatbot.consolidation.target import (
    ConsolidationTarget,
    ConsolidationTargetPlan,
    ConsolidationTargetValidationError,
    ConsolidationWriteIntent,
    build_consolidation_target_plan,
    validate_write_intent,
)


def __getattr__(name: str):
    """Load the graph entrypoint lazily to keep target imports lightweight."""

    if name == "call_consolidation_subgraph":
        core_module = importlib.import_module(
            "kazusa_ai_chatbot.consolidation.core"
        )
        call_consolidation_subgraph = core_module.call_consolidation_subgraph

        return_value = call_consolidation_subgraph
        return return_value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConsolidationTarget",
    "ConsolidationTargetPlan",
    "ConsolidationTargetValidationError",
    "ConsolidationWriteIntent",
    "build_consolidation_target_plan",
    "call_consolidation_subgraph",
    "validate_write_intent",
]
