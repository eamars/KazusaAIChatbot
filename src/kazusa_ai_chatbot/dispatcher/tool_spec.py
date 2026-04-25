"""Registered tool definitions and registry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kazusa_ai_chatbot.dispatcher.adapter_iface import AdapterRegistry
    from kazusa_ai_chatbot.dispatcher.task import DispatchContext

TaskHandler = Callable[[dict, "DispatchContext", "AdapterRegistry"], Awaitable[None]]

_ROLE_ORDER = {"user": 0, "moderator": 1, "admin": 2}


def _has_permission(bot_role: str, required: Optional[str]) -> bool:
    """Return whether the current role satisfies a required permission.

    Args:
        bot_role: Current source-context bot role.
        required: Optional minimum required role.

    Returns:
        ``True`` when the role is sufficient.
    """

    if required is None:
        return True
    return _ROLE_ORDER.get(bot_role, 0) >= _ROLE_ORDER.get(required, 0)


@dataclass(frozen=True)
class ToolSpec:
    """Definition of one dispatchable tool.

    Args:
        name: Unique registry key.
        description: Prompt-facing tool description for the consolidator.
        args_schema: JSON-schema-like argument contract.
        handler: Async implementation called when the task fires.
        requires_permission: Optional minimum role needed to expose the tool.
        platforms: Optional allowlist of supported source platforms.
    """

    name: str
    description: str
    args_schema: dict
    handler: TaskHandler
    requires_permission: Optional[str] = None
    platforms: Optional[set[str]] = None


class ToolRegistry:
    """Registry of dispatchable tools with context-aware filtering."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register or replace one tool spec.

        Args:
            spec: Tool specification to register.
        """

        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        """Return a tool spec by name.

        Args:
            name: Tool registry key.

        Returns:
            Registered tool specification.
        """

        return self._tools[name]

    def filter(self, ctx: "DispatchContext") -> list[ToolSpec]:
        """Return the tools that are valid in one dispatch context.

        Args:
            ctx: Source-side dispatch context used for permission filtering.

        Returns:
            Tool specs visible to the consolidator in this context.
        """

        visible: list[ToolSpec] = []
        for spec in self._tools.values():
            if spec.platforms is not None and ctx.source_platform not in spec.platforms:
                continue
            if not _has_permission(ctx.bot_role, spec.requires_permission):
                continue
            visible.append(spec)
        return visible

    def visible_names(self, ctx: "DispatchContext") -> set[str]:
        """Return the set of tool names visible in one context."""

        return {spec.name for spec in self.filter(ctx)}
