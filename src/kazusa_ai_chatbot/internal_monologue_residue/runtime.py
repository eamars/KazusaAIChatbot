"""Public facade for internal monologue residue runtime callers."""

from __future__ import annotations

from kazusa_ai_chatbot.internal_monologue_residue.loader import (
    load_residue_context,
)
from kazusa_ai_chatbot.internal_monologue_residue.models import (
    InternalMonologueResidueRow,
    ResidueLoadResult,
    ResidueRecordResult,
    ResidueTriggerScope,
)
from kazusa_ai_chatbot.internal_monologue_residue.projection import (
    project_residue_window,
)
from kazusa_ai_chatbot.internal_monologue_residue.recorder import (
    record_completed_episode_residue,
)

__all__ = [
    "InternalMonologueResidueRow",
    "ResidueLoadResult",
    "ResidueRecordResult",
    "ResidueTriggerScope",
    "load_residue_context",
    "project_residue_window",
    "record_completed_episode_residue",
]
