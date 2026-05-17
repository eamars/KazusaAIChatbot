"""Remote HTTP adapter bridge for cross-process scheduled delivery."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any

import httpx

from kazusa_ai_chatbot.dispatcher.adapter_iface import SendResult
from kazusa_ai_chatbot.time_boundary import (
    parse_storage_utc_datetime,
    storage_utc_now,
    storage_utc_now_iso,
)

logger = logging.getLogger(__name__)


class RemoteHttpAdapter:
    """Proxy outbound sends to an adapter-owned HTTP endpoint.

    Args:
        platform: Platform key such as ``qq`` or ``discord``.
        callback_url: Base URL exposed by the live adapter process.
        shared_secret: Optional bearer token used for adapter callback auth.
        timeout_seconds: HTTP timeout for one outbound send attempt.
    """

    def __init__(
        self,
        *,
        platform: str,
        callback_url: str,
        shared_secret: str = "",
        timeout_seconds: float = 10.0,
        platform_bot_id: str = "",
        display_name: str = "",
    ) -> None:
        self.platform = platform
        self.platform_bot_id = platform_bot_id
        self.display_name = display_name
        self._callback_url = callback_url.rstrip("/")
        self._shared_secret = shared_secret
        self._timeout_seconds = timeout_seconds

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
        delivery_mentions: Sequence[dict[str, Any]] | None = None,
    ) -> SendResult:
        """Send a message by calling the registered remote adapter.

        Args:
            channel_id: Target channel, group, or DM id.
            text: Message body to deliver.
            channel_type: Platform-neutral target scope such as ``group`` or
                ``private``.
            reply_to_msg_id: Optional message id to quote/reply to.
            delivery_mentions: Optional adapter-owned mention requests.

        Returns:
            Structured delivery result returned by the remote adapter.
        """

        headers: dict[str, str] = {}
        if self._shared_secret:
            headers["Authorization"] = f"Bearer {self._shared_secret}"

        payload = {
            "channel_id": channel_id,
            "channel_type": channel_type,
            "text": text,
            "reply_to_msg_id": reply_to_msg_id,
        }
        if delivery_mentions is not None:
            payload["delivery_mentions"] = list(delivery_mentions)

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                f"{self._callback_url}/send_message",
                json=payload,
                headers=headers,
            )
        response.raise_for_status()
        data = response.json()

        sent_at_raw = str(data.get("sent_at") or storage_utc_now_iso())
        try:
            sent_at = parse_storage_utc_datetime(sent_at_raw)
        except ValueError as exc:
            logger.debug(f"Using current time for invalid adapter sent_at: {exc}")
            sent_at = storage_utc_now()

        return_value = SendResult(
            platform=str(data.get("platform") or self.platform),
            channel_id=str(data.get("channel_id") or channel_id),
            message_id=str(data.get("message_id") or ""),
            sent_at=sent_at,
        )
        return return_value
