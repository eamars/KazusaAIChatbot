"""Dataclasses for dispatch context and scheduled tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from kazusa_ai_chatbot.db import ScheduledEventDoc
from kazusa_ai_chatbot.time_boundary import (
    normalize_storage_utc_iso,
    parse_storage_utc_datetime,
)

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

    @classmethod
    def from_scheduler_doc(cls, doc: ScheduledEventDoc) -> "DispatchContext":
        """Rehydrate the original dispatch context from a scheduled event doc.

        Args:
            doc: Persisted scheduled event document.

        Returns:
            The dispatch context carried alongside the event.
        """

        return_value = cls(
            source_platform=doc["source_platform"],
            source_channel_id=doc["source_channel_id"],
            source_user_id=doc["source_user_id"],
            source_message_id=doc["source_message_id"],
            guild_id=doc.get("guild_id"),
            bot_permission_role=doc["bot_role"],
            now=parse_storage_utc_datetime(doc["execute_at"]),
            source_channel_type=str(doc["source_channel_type"]),
            source_platform_bot_id=str(doc.get("source_platform_bot_id") or ""),
            source_character_name=str(doc.get("source_character_name") or ""),
        )
        return return_value
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

    def to_scheduler_doc(self, ctx: DispatchContext) -> ScheduledEventDoc:
        """Serialize the task into the scheduler's MongoDB document shape.

        Args:
            ctx: Original source-side dispatch context.

        Returns:
            A ``ScheduledEventDoc`` ready for ``scheduler.schedule_event``.
        """

        execute_at_utc = normalize_storage_utc_iso(self.execute_at.isoformat())
        return_value = {
            "tool": self.tool,
            "args": self.args,
            "execute_at": execute_at_utc,
            "status": "pending",
            "source_platform": ctx.source_platform,
            "source_channel_id": ctx.source_channel_id,
            "source_channel_type": ctx.source_channel_type,
            "source_user_id": ctx.source_user_id,
            "source_message_id": ctx.source_message_id,
            "guild_id": ctx.guild_id,
            "bot_role": ctx.bot_permission_role,
        }
        if ctx.source_platform_bot_id:
            return_value["source_platform_bot_id"] = ctx.source_platform_bot_id
        if ctx.source_character_name:
            return_value["source_character_name"] = ctx.source_character_name
        return return_value

    @classmethod
    def from_scheduler_doc(cls, doc: ScheduledEventDoc) -> "Task":
        """Rehydrate a ``Task`` from a stored scheduled event document.

        Args:
            doc: Persisted scheduled event document.

        Returns:
            Reconstructed task instance.
        """

        return_value = cls(
            tool=doc["tool"],
            args=dict(doc["args"]),
            execute_at=parse_storage_utc_datetime(doc["execute_at"]),
        )
        return return_value
