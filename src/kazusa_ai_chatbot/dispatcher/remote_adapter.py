"""Remote HTTP adapter bridge for cross-process scheduled delivery."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from kazusa_ai_chatbot.dispatcher.adapter_iface import SendResult
from kazusa_ai_chatbot.dispatcher.task import parse_iso_datetime

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
    ) -> None:
        self.platform = platform
        self._callback_url = callback_url.rstrip("/")
        self._shared_secret = shared_secret
        self._timeout_seconds = timeout_seconds

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        reply_to_msg_id: str | None = None,
    ) -> SendResult:
        """Send a message by calling the registered remote adapter.

        Args:
            channel_id: Target channel, group, or DM id.
            text: Message body to deliver.
            reply_to_msg_id: Optional message id to quote/reply to.

        Returns:
            Structured delivery result returned by the remote adapter.
        """

        headers: dict[str, str] = {}
        if self._shared_secret:
            headers["Authorization"] = f"Bearer {self._shared_secret}"

        payload = {
            "channel_id": channel_id,
            "text": text,
            "reply_to_msg_id": reply_to_msg_id,
        }
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                f"{self._callback_url}/send_message",
                json=payload,
                headers=headers,
            )
        response.raise_for_status()
        data = response.json()

        sent_at_raw = str(data.get("sent_at") or datetime.now(timezone.utc).isoformat())
        try:
            sent_at = parse_iso_datetime(sent_at_raw)
        except ValueError as exc:
            logger.debug(f"Handled exception in send_message: {exc}")
            sent_at = datetime.now(timezone.utc)

        return_value = SendResult(
            platform=str(data.get("platform") or self.platform),
            channel_id=str(data.get("channel_id") or channel_id),
            message_id=str(data.get("message_id") or ""),
            sent_at=sent_at,
        )
        return return_value
