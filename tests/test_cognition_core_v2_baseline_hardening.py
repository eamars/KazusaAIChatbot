"""Red contract for the native V2 diagnostic warning projection."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_core_v2.facade import (
    _deduplicate_diagnostics_warnings,
)


def test_diagnostics_warning_projection_is_unique() -> None:
    """Repeated stage warnings must not invalidate the final output shape."""

    assert _deduplicate_diagnostics_warnings([
        "resolver_failed:contract",
        "resolver_failed:contract",
        "branch_failed:goal",
        "resolver_failed:contract",
    ]) == [
        "resolver_failed:contract",
        "branch_failed:goal",
    ]
