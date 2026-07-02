"""Public entrypoint for scoped runtime pipeline coordination."""

from __future__ import annotations

from kazusa_ai_chatbot.runtime_coordination.coordinator import (
    PipelineCoordinator,
    PipelineRunAdmission,
    PipelineRunHandle,
)
from kazusa_ai_chatbot.runtime_coordination.models import (
    PipelineCancellation,
    PipelineCancelled,
    PipelinePrecedence,
    PipelineScope,
)


__all__ = [
    "PipelineCancellation",
    "PipelineCancelled",
    "PipelineCoordinator",
    "PipelinePrecedence",
    "PipelineRunAdmission",
    "PipelineRunHandle",
    "PipelineScope",
]
