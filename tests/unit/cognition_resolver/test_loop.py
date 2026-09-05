"""Behavior checks for resolver iteration and result handling."""

from __future__ import annotations

from importlib import import_module

import pytest

from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ResolverValidationError,
)

MODULE_PATH = "kazusa_ai_chatbot.cognition_resolver.loop"
EXPECTED_SYMBOLS = ["call_cognition_resolver_loop"]




def test_resolver_surface_provenance_targets_current_user() -> None:
    """Visible resolver fallbacks target the resolved current user exactly."""

    provenance = import_module(MODULE_PATH)._resolver_speak_cognition_provenance({
        "global_user_id": "global-user-123",
    })

    assert provenance == {
        "target_roles": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "global-user-123",
        }],
        "evidence_handles": [],
    }


def test_resolver_surface_provenance_requires_current_user() -> None:
    """Missing and blank current-user identity fails closed before L3."""

    for state in ({}, {"global_user_id": ""}, {"global_user_id": "   "}):
        with pytest.raises(ResolverValidationError, match="global_user_id"):
            import_module(MODULE_PATH)._resolver_speak_cognition_provenance(state)

