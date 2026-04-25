"""Adapter protocol and registry for tool delivery."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class SendResult:
    """Structured delivery result returned by messaging adapters.

    Args:
        platform: Platform identifier that accepted the outbound message.
        channel_id: Channel, group, or DM target that received the message.
        message_id: Platform-assigned outbound message identifier when available.
        sent_at: Timestamp when the adapter finished sending the message.
    """

    platform: str
    channel_id: str
    message_id: str
    sent_at: datetime


@runtime_checkable
class MessagingAdapter(Protocol):
    """Minimal outbound messaging interface implemented by platform adapters."""

    platform: str

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to_msg_id: Optional[str] = None,
    ) -> SendResult:
        """Send a message through the adapter."""


class UnknownPlatformError(KeyError):
    """Raised when no adapter is registered for the requested platform."""


class AdapterRegistry:
    """Lookup table for platform-specific messaging adapters."""

    def __init__(self) -> None:
        self._by_platform: dict[str, MessagingAdapter] = {}

    def register(self, adapter: MessagingAdapter) -> None:
        """Register or replace an adapter for its platform key.

        Args:
            adapter: Adapter instance implementing ``MessagingAdapter``.
        """

        self._by_platform[adapter.platform] = adapter

    def get(self, platform: str) -> MessagingAdapter:
        """Return the adapter for one platform or raise ``UnknownPlatformError``.

        Args:
            platform: Platform key such as ``discord`` or ``qq``.

        Returns:
            The registered adapter.
        """

        if platform not in self._by_platform:
            raise UnknownPlatformError(platform)
        return self._by_platform[platform]

    def has(self, platform: str) -> bool:
        """Return whether an adapter exists for one platform."""

        return platform in self._by_platform

    def platforms(self) -> set[str]:
        """Return the set of registered platform keys."""

        return set(self._by_platform)
