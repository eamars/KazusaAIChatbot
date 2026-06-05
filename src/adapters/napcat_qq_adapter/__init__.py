"""Public package surface for the NapCat QQ adapter."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "NapCatWSAdapter",
    "QQEnvelopeNormalizer",
    "main",
    "project_qq_semantic_text",
    "runtime_app",
]

_PUBLIC_SYMBOL_MODULES = {
    "NapCatWSAdapter": "adapters.napcat_qq_adapter.ws_adapter",
    "QQEnvelopeNormalizer": "adapters.napcat_qq_adapter.envelope_normalizer",
    "main": "adapters.napcat_qq_adapter.cli",
    "project_qq_semantic_text": "adapters.napcat_qq_adapter.cq_projection",
    "runtime_app": "adapters.napcat_qq_adapter.runtime_api",
}


def __getattr__(name: str) -> Any:
    """Load public adapter symbols without coupling runtime API to websocket code."""

    module_name = _PUBLIC_SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
