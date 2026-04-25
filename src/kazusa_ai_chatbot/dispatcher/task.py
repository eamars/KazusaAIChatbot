"""Dataclasses for raw tool calls, dispatch context, and scheduled tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from kazusa_ai_chatbot.db import ScheduledEventDoc


def _to_utc_datetime(value: datetime) -> datetime:
    """Normalize a datetime into an aware UTC value.

    Args:
        value: Datetime to normalize.

    Returns:
        A timezone-aware UTC datetime.
    """

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_iso_datetime(value: str) -> datetime:
    """Parse an ISO-8601 string into a timezone-aware UTC datetime.

    Args:
        value: ISO-8601 timestamp string.

    Returns:
        Parsed UTC datetime.
    """

    normalized = value.replace("Z", "+00:00")
    return _to_utc_datetime(datetime.fromisoformat(normalized))


@dataclass(frozen=True)
class RawToolCall:
    """Unvalidated tool call emitted by the consolidator LLM.

    Args:
        tool: Tool name chosen by the LLM.
        args: Raw argument mapping emitted for that tool.
    """

    tool: str
    args: dict


@dataclass(frozen=True)
class DispatchContext:
    """Source-side defaults and permissions used while validating tool calls.

    Args:
        source_platform: Platform of the user message that caused this dispatch.
        source_channel_id: Channel or DM identifier of the source message.
        source_user_id: Global user identifier of the source speaker.
        source_message_id: Platform message identifier of the source message.
        guild_id: Optional guild or server scope for permissions.
        bot_role: Current bot permission level for the source context.
        now: Frozen dispatch time used for immediate tasks.
    """

    source_platform: str
    source_channel_id: str
    source_user_id: str
    source_message_id: str
    guild_id: Optional[str]
    bot_role: str
    now: datetime

    @classmethod
    def from_scheduler_doc(cls, doc: ScheduledEventDoc) -> "DispatchContext":
        """Rehydrate the original dispatch context from a scheduled event doc.

        Args:
            doc: Persisted scheduled event document.

        Returns:
            The dispatch context carried alongside the event.
        """

        return cls(
            source_platform=doc["source_platform"],
            source_channel_id=doc["source_channel_id"],
            source_user_id=doc["source_user_id"],
            source_message_id=doc["source_message_id"],
            guild_id=doc.get("guild_id"),
            bot_role=doc.get("bot_role", "user"),
            now=parse_iso_datetime(doc["execute_at"]),
        )


@dataclass(frozen=True)
class Task:
    """Validated scheduled tool invocation ready for persistence or execution.

    Args:
        tool: Registered tool name.
        args: Fully-populated tool arguments after evaluator defaulting.
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

        return {
            "tool": self.tool,
            "args": self.args,
            "execute_at": _to_utc_datetime(self.execute_at).isoformat(),
            "status": "pending",
            "source_platform": ctx.source_platform,
            "source_channel_id": ctx.source_channel_id,
            "source_user_id": ctx.source_user_id,
            "source_message_id": ctx.source_message_id,
            "guild_id": ctx.guild_id,
            "bot_role": ctx.bot_role,
        }

    @classmethod
    def from_scheduler_doc(cls, doc: ScheduledEventDoc) -> "Task":
        """Rehydrate a ``Task`` from a stored scheduled event document.

        Args:
            doc: Persisted scheduled event document.

        Returns:
            Reconstructed task instance.
        """

        return cls(
            tool=doc["tool"],
            args=dict(doc.get("args") or {}),
            execute_at=parse_iso_datetime(doc["execute_at"]),
        )


@dataclass(frozen=True)
class DispatchResult:
    """Dispatcher outcome containing scheduled and rejected tool calls.

    Args:
        scheduled: Validated tasks plus the persisted scheduler event id.
        rejected: Raw tool calls rejected before scheduling, with the reason.
    """

    scheduled: list[tuple[Task, str]]
    rejected: list[tuple[RawToolCall, str]]
