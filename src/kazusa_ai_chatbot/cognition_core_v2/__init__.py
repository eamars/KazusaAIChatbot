"""Public V2 cognition contracts and entrypoints."""

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContextLimitError,
    CognitionContractError,
    CognitionCoreInputV2,
    CognitionCoreOutputV2,
    CognitionCoreServicesV2,
    CognitionExecutionError,
    TextSurfaceInputV2,
    TextSurfaceOutputV2,
    TextSurfaceServicesV2,
)
from kazusa_ai_chatbot.cognition_core_v2.facade import run_cognition
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    CognitionStateError,
    build_acquaintance_user_state,
    build_character_production_state,
    prune_terminal_entities,
    resolve_state_scope,
    validate_cognition_state,
)

__all__ = [
    "CognitionCoreInputV2",
    "CognitionCoreOutputV2",
    "CognitionCoreServicesV2",
    "CognitionContextLimitError",
    "CognitionContractError",
    "CognitionExecutionError",
    "CognitionStateError",
    "TextSurfaceInputV2",
    "TextSurfaceOutputV2",
    "TextSurfaceServicesV2",
    "build_acquaintance_user_state",
    "build_character_production_state",
    "prune_terminal_entities",
    "resolve_state_scope",
    "run_cognition",
    "run_text_surface_planning",
    "validate_cognition_state",
]


def __getattr__(name: str) -> object:
    """Load the surface orchestrator only when the public API is requested."""

    if name == "run_text_surface_planning":
        from kazusa_ai_chatbot.cognition_core_v2.surface import (
            run_text_surface_planning,
        )

        return run_text_surface_planning
    raise AttributeError(f"module has no attribute {name!r}")
