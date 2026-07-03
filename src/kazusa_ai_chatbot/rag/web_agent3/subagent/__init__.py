"""Auto-discovery for web_agent3 source subagents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import Any, cast

from kazusa_ai_chatbot.rag.web_agent3.contracts import _RouterDecision

_SubagentExecute = Callable[[_RouterDecision], Awaitable[Any]]


def _validate_subagent_module(
    module: ModuleType,
) -> tuple[str, str, tuple[str, ...], _SubagentExecute]:
    """Validate the module-level source subagent interface.

    Args:
        module: Imported candidate module from this package.

    Returns:
        Source name, prompt-facing description, and async execution callable.
    """
    source = getattr(module, "SOURCE", None)
    if not isinstance(source, str) or not source:
        raise RuntimeError(f"web_agent3 subagent {module.__name__} missing SOURCE")

    description = getattr(module, "DESCRIPTION", None)
    if not isinstance(description, str) or not description:
        raise RuntimeError(
            f"web_agent3 subagent {module.__name__} missing DESCRIPTION"
        )

    execute = getattr(module, "execute", None)
    if not callable(execute):
        raise RuntimeError(f"web_agent3 subagent {module.__name__} missing execute")

    supported_actions = getattr(module, "SUPPORTED_ACTIONS", None)
    if not isinstance(supported_actions, tuple) or not supported_actions:
        raise RuntimeError(
            f"web_agent3 subagent {module.__name__} missing SUPPORTED_ACTIONS"
        )
    for action in supported_actions:
        if not isinstance(action, str) or not action:
            raise RuntimeError(
                f"web_agent3 subagent {module.__name__} has invalid action"
            )

    execute_fn = cast(_SubagentExecute, execute)
    validated = (source, description, supported_actions, execute_fn)
    return validated


def _subagent_is_enabled(module: ModuleType) -> bool:
    """Return whether a discovered source module should be registered."""
    is_enabled = getattr(module, "is_enabled", None)
    if is_enabled is None:
        return_value = True
        return return_value
    if not callable(is_enabled):
        raise RuntimeError(
            f"web_agent3 subagent {module.__name__} has invalid is_enabled"
        )

    enabled = is_enabled()
    if not isinstance(enabled, bool):
        raise RuntimeError(
            f"web_agent3 subagent {module.__name__} is_enabled not bool"
        )

    return enabled


def _discover_subagents() -> dict[str, ModuleType]:
    """Discover source subagent modules in this package."""
    discovered: dict[str, ModuleType] = {}
    module_infos = sorted(iter_modules(__path__), key=lambda item: item.name)
    for module_info in module_infos:
        if module_info.ispkg or module_info.name.startswith("_"):
            continue

        module = import_module(f"{__name__}.{module_info.name}")
        source, _, _, _ = _validate_subagent_module(module)
        if not _subagent_is_enabled(module):
            continue
        if source in discovered:
            raise RuntimeError(f"duplicate web_agent3 subagent source: {source}")
        discovered[source] = module

    return_value = discovered
    return return_value


_SUBAGENTS = _discover_subagents()
_SUBAGENT_DESCRIPTIONS = {
    source: module.DESCRIPTION
    for source, module in _SUBAGENTS.items()
}
_SUBAGENT_SUPPORTED_ACTIONS = {
    source: module.SUPPORTED_ACTIONS
    for source, module in _SUBAGENTS.items()
}
_SUBAGENT_NAMES = tuple(_SUBAGENTS)

__all__ = [
    "_SUBAGENT_DESCRIPTIONS",
    "_SUBAGENT_NAMES",
    "_SUBAGENT_SUPPORTED_ACTIONS",
    "_SUBAGENTS",
]
