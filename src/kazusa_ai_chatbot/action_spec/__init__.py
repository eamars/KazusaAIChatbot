"""Modality-neutral action-spec contracts and validation helpers."""

from kazusa_ai_chatbot.action_spec.evaluator import (
    ActionSpecEvaluator,
)
from kazusa_ai_chatbot.action_spec.execution import execute_action_specs_for_trace
from kazusa_ai_chatbot.action_spec.models import (
    ACTION_SPEC_VERSION,
    LIFECYCLE_STATUS_BY_DECISION,
    ActionValidationError,
    validate_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import (
    build_initial_action_capabilities,
    project_prompt_affordances,
)
from kazusa_ai_chatbot.action_spec.results import (
    build_action_result,
    build_episode_trace,
    build_private_surface_output,
    build_text_surface_output,
    has_consolidatable_output,
    project_episode_trace_for_consolidation,
)

__all__ = [
    "ACTION_SPEC_VERSION",
    "LIFECYCLE_STATUS_BY_DECISION",
    "ActionSpecEvaluator",
    "ActionValidationError",
    "build_action_result",
    "build_episode_trace",
    "build_initial_action_capabilities",
    "build_private_surface_output",
    "build_text_surface_output",
    "execute_action_specs_for_trace",
    "has_consolidatable_output",
    "project_prompt_affordances",
    "project_episode_trace_for_consolidation",
    "validate_action_spec",
]
