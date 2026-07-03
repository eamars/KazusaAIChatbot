"""Auto-discovery for resolver-local subagents."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import cast

from kazusa_ai_chatbot.complex_task_resolver.contracts import (
    ComplexTaskSubagentV1,
)

_SubagentFactory = Callable[[], ComplexTaskSubagentV1]


def _validate_subagent_module(
    module: ModuleType,
) -> tuple[
    str,
    str,
    tuple[str, ...],
    tuple[str, ...],
    str,
    _SubagentFactory,
]:
    """Validate the module-level resolver subagent interface."""

    subagent = getattr(module, "SUBAGENT", None)
    if not isinstance(subagent, str) or not subagent:
        raise RuntimeError(
            f"complex resolver subagent {module.__name__} missing SUBAGENT"
        )

    description = getattr(module, "DESCRIPTION", None)
    if not isinstance(description, str) or not description:
        raise RuntimeError(
            f"complex resolver subagent {module.__name__} missing DESCRIPTION"
        )

    supported_actions = getattr(module, "SUPPORTED_ACTIONS", None)
    if not isinstance(supported_actions, tuple) or not supported_actions:
        raise RuntimeError(
            f"complex resolver subagent {module.__name__} missing "
            "SUPPORTED_ACTIONS"
        )
    for action in supported_actions:
        if not isinstance(action, str) or not action:
            raise RuntimeError(
                f"complex resolver subagent {module.__name__} has invalid action"
            )

    owned_node_kinds = getattr(module, "OWNED_NODE_KINDS", None)
    if not isinstance(owned_node_kinds, tuple):
        raise RuntimeError(
            f"complex resolver subagent {module.__name__} missing "
            "OWNED_NODE_KINDS"
        )
    for node_kind in owned_node_kinds:
        if not isinstance(node_kind, str) or not node_kind:
            raise RuntimeError(
                f"complex resolver subagent {module.__name__} has invalid "
                "owned node kind"
            )

    default_action = getattr(module, "DEFAULT_ACTION", None)
    if (
        not isinstance(default_action, str)
        or default_action not in supported_actions
    ):
        raise RuntimeError(
            f"complex resolver subagent {module.__name__} has invalid "
            "DEFAULT_ACTION"
        )

    create = getattr(module, "create", None)
    if not callable(create):
        raise RuntimeError(
            f"complex resolver subagent {module.__name__} missing create"
        )

    create_fn = cast(_SubagentFactory, create)
    validated = (
        subagent,
        description,
        supported_actions,
        owned_node_kinds,
        default_action,
        create_fn,
    )
    return validated


def _subagent_is_enabled(module: ModuleType) -> bool:
    """Return whether a discovered resolver subagent should be registered."""

    is_enabled = getattr(module, "is_enabled", None)
    if is_enabled is None:
        return_value = True
        return return_value
    if not callable(is_enabled):
        raise RuntimeError(
            f"complex resolver subagent {module.__name__} has invalid is_enabled"
        )
    enabled = is_enabled()
    if not isinstance(enabled, bool):
        raise RuntimeError(
            f"complex resolver subagent {module.__name__} is_enabled not bool"
        )
    return enabled


def _discover_subagents() -> dict[str, ModuleType]:
    """Discover resolver-local subagent modules in this package."""

    discovered: dict[str, ModuleType] = {}
    module_infos = sorted(iter_modules(__path__), key=lambda item: item.name)
    for module_info in module_infos:
        if module_info.ispkg or module_info.name.startswith("_"):
            continue
        module = import_module(f"{__name__}.{module_info.name}")
        subagent, _, _, _, _, _ = _validate_subagent_module(module)
        if not _subagent_is_enabled(module):
            continue
        if subagent in discovered:
            raise RuntimeError(
                f"duplicate complex resolver subagent: {subagent}"
            )
        discovered[subagent] = module

    return_value = discovered
    return return_value


_SUBAGENTS = _discover_subagents()
_SUBAGENT_DESCRIPTIONS = {
    subagent: module.DESCRIPTION
    for subagent, module in _SUBAGENTS.items()
}
_SUBAGENT_SUPPORTED_ACTIONS = {
    subagent: module.SUPPORTED_ACTIONS
    for subagent, module in _SUBAGENTS.items()
}
_SUBAGENT_OWNED_NODE_KINDS = {
    subagent: module.OWNED_NODE_KINDS
    for subagent, module in _SUBAGENTS.items()
}
_SUBAGENT_DEFAULT_ACTIONS = {
    subagent: module.DEFAULT_ACTION
    for subagent, module in _SUBAGENTS.items()
}
_SUBAGENT_NAMES = tuple(_SUBAGENTS)
_SUBAGENT_TOOLS_TEXT = "\n".join(
    f"- {name}: {_SUBAGENT_DESCRIPTIONS[name]}"
    for name in _SUBAGENT_NAMES
)


def create_subagents() -> dict[str, ComplexTaskSubagentV1]:
    """Create fresh resolver-local subagent instances."""

    subagents = {
        subagent: cast(_SubagentFactory, module.create)()
        for subagent, module in _SUBAGENTS.items()
    }
    return subagents


def owned_subagent_for_node_kind(node_kind: str) -> str | None:
    """Return the resolver-local subagent that owns one node kind."""

    for subagent, node_kinds in _SUBAGENT_OWNED_NODE_KINDS.items():
        if node_kind in node_kinds:
            return subagent
    return None


__all__ = [
    "_SUBAGENT_DEFAULT_ACTIONS",
    "_SUBAGENT_DESCRIPTIONS",
    "_SUBAGENT_NAMES",
    "_SUBAGENT_OWNED_NODE_KINDS",
    "_SUBAGENT_SUPPORTED_ACTIONS",
    "_SUBAGENT_TOOLS_TEXT",
    "_SUBAGENTS",
    "create_subagents",
    "owned_subagent_for_node_kind",
]
