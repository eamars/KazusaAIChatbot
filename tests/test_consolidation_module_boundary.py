"""Tests for first-class consolidation module ownership."""

from __future__ import annotations

import importlib
import importlib.util
import inspect

from kazusa_ai_chatbot.consolidation import core

_LEGACY_CONSOLIDATOR_PREFIX = "persona_supervisor2_" + "consolidator"
_LEGACY_NODE_IMPORT_PREFIX = (
    "kazusa_ai_chatbot.nodes.persona_supervisor2_" + "consolidator"
)


def test_consolidation_helper_modules_live_in_consolidation_package() -> None:
    module_names = [
        "schema",
        "origin",
        "origin_policy",
        "facts",
        "reflection",
        "images",
        "memory_units",
        "persistence",
    ]

    for module_name in module_names:
        module = importlib.import_module(
            f"kazusa_ai_chatbot.consolidation.{module_name}"
        )
        assert module.__name__ == (
            f"kazusa_ai_chatbot.consolidation.{module_name}"
        )


def test_legacy_node_consolidator_modules_are_absent() -> None:
    legacy_names = [
        _LEGACY_CONSOLIDATOR_PREFIX,
        f"{_LEGACY_CONSOLIDATOR_PREFIX}_schema",
        f"{_LEGACY_CONSOLIDATOR_PREFIX}_origin",
        f"{_LEGACY_CONSOLIDATOR_PREFIX}_origin_policy",
        f"{_LEGACY_CONSOLIDATOR_PREFIX}_facts",
        f"{_LEGACY_CONSOLIDATOR_PREFIX}_reflection",
        f"{_LEGACY_CONSOLIDATOR_PREFIX}_images",
        f"{_LEGACY_CONSOLIDATOR_PREFIX}_memory_units",
        f"{_LEGACY_CONSOLIDATOR_PREFIX}_persistence",
    ]

    for legacy_name in legacy_names:
        spec = importlib.util.find_spec(
            f"kazusa_ai_chatbot.nodes.{legacy_name}"
        )
        assert spec is None


def test_consolidation_core_imports_no_legacy_node_helpers() -> None:
    source = inspect.getsource(core)

    assert _LEGACY_NODE_IMPORT_PREFIX not in source
