"""Closed engine-selection behavior for the cognition core.

These cases verify the process-level selector without model calls: the
connector and selector share one selected-engine binding, a reloaded selector
resolves exactly the configured engine entrypoint, and an unknown engine value
fails startup with an error that names the allowed set.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

import kazusa_ai_chatbot.cognition_core_selector as cognition_core_selector
import kazusa_ai_chatbot.nodes.persona_supervisor2_cognition as connector_module
from kazusa_ai_chatbot import config
from kazusa_ai_chatbot.cognition_core_v2.facade import (
    run_cognition as v2_run_cognition,
)
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    run_cognition as v3_run_cognition,
)


def test_connector_and_selector_share_one_selected_engine_binding() -> None:
    """Live and idle connector paths call the selector's one entrypoint."""

    assert connector_module.run_cognition is cognition_core_selector.run_cognition

    if config.COGNITION_CORE_ENGINE == "v2":
        assert cognition_core_selector.run_cognition is v2_run_cognition
    else:
        assert cognition_core_selector.run_cognition is v3_run_cognition


def test_reloaded_selector_resolves_the_configured_engine() -> None:
    """A reloaded selector binds exactly the configured engine entrypoint."""

    original_engine = config.COGNITION_CORE_ENGINE
    try:
        config.COGNITION_CORE_ENGINE = "v2"
        importlib.reload(cognition_core_selector)
        assert (
            cognition_core_selector.run_cognition is v2_run_cognition
        )

        config.COGNITION_CORE_ENGINE = "v3"
        importlib.reload(cognition_core_selector)
        assert cognition_core_selector.run_cognition is v3_run_cognition
        parameters = list(
            inspect.signature(v3_run_cognition).parameters
        )
        assert parameters == ["input_payload", "services"]
    finally:
        config.COGNITION_CORE_ENGINE = original_engine
        importlib.reload(cognition_core_selector)


def test_unknown_engine_value_fails_startup_naming_the_allowed_set() -> None:
    """An unknown engine value raises a ValueError naming v2 and v3."""

    with pytest.raises(ValueError) as excinfo:
        cognition_core_selector.resolve_engine_module("v9")

    message = str(excinfo.value)
    assert "Cognition core engine must be one of:" in message
    assert "v2" in message
    assert "v3" in message
