"""Placeholder for future multi-PM reading orchestration.

Code reading keeps one bounded reading PM. Larger questions return limitations
or ask for a narrower scope instead of adding another orchestration layer.
"""

from kazusa_ai_chatbot.coding_agent.code_reading.models import ReadingSupervisorState


def should_escalate_beyond_single_pm(state: ReadingSupervisorState) -> bool:
    """Return whether this run exceeds the standalone reading PM boundary."""

    _ = state
    return False
