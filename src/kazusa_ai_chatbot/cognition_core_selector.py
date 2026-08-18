"""Closed process-level selector for the cognition core engine.

The configured engine is resolved exactly once at import and bound to a single
``run_cognition`` entrypoint. The closed choice set contains only ``v2`` and
``v3``; an unknown value fails startup, and a selected-engine failure never
loads or invokes another engine.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from kazusa_ai_chatbot.config import COGNITION_CORE_ENGINE

_ENGINE_MODULE_NAMES: dict[str, str] = {
    "v2": "kazusa_ai_chatbot.cognition_core_v2",
    "v3": "kazusa_ai_chatbot.cognition_core_v3",
}


def resolve_engine_module(engine: str) -> ModuleType:
    """Resolve a configured engine name to its exact module.

    Args:
        engine: Engine value accepted by ``COGNITION_CORE_ENGINE``.

    Returns:
        The imported engine package that exposes the shared entrypoint.

    Raises:
        ValueError: Unknown engine values fail startup without loading another
            engine as a substitute.
    """
    if engine not in _ENGINE_MODULE_NAMES:
        allowed_text = ", ".join(sorted(_ENGINE_MODULE_NAMES))
        raise ValueError(f"Cognition core engine must be one of: {allowed_text}")

    return importlib.import_module(_ENGINE_MODULE_NAMES[engine])


_selected_engine_module = resolve_engine_module(COGNITION_CORE_ENGINE)

run_cognition = _selected_engine_module.run_cognition
