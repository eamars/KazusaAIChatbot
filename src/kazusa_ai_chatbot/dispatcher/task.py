"""Dataclasses for dispatch context and delivery tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

BotPermissionRole = str


@dataclass(frozen=True)
class DispatchContext:
    """Source-side substitution data and permissions for tool-call validation.

    Args:
        source_platform: Platform of the user message that caused this dispatch.
        source_channel_id: Channel or DM identifier of the source message.
        source_user_id: Global user identifier of the source speaker.
        source_message_id: Platform message identifier of the source message.
        guild_id: Optional guild or server scope for permissions.
        bot_permission_role: Current permission level for the source context.
        now: Frozen dispatch time used for immediate tasks.
        source_channel_type: Source channel class such as ``group`` or
            ``private``.
        source_platform_bot_id: Platform account id for the active character.
        source_character_name: Display name for the active character.
    """

    source_platform: str
    source_channel_id: str
    source_user_id: str
    source_message_id: str
    guild_id: Optional[str]
    bot_permission_role: BotPermissionRole
    now: datetime
    source_channel_type: str
    source_platform_bot_id: str = ""
    source_character_name: str = ""


@dataclass(frozen=True)
class Task:
    """Validated scheduled tool invocation ready for persistence or execution.

    Args:
        tool: Registered tool name.
        args: Fully-populated tool arguments after evaluator validation.
        execute_at: Absolute UTC instant when the task should fire.
        source_event_id: Optional source event identifier for tracing.
        tags: Free-form tracing tags such as the originating instruction.
    """

    tool: str
    args: dict
    execute_at: datetime
    source_event_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
