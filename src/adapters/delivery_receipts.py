"""Shared adapter helpers for reporting delivered platform messages."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence

import httpx


_DEFAULT_RETRY_DELAYS = (0.25, 0.75)


async def post_delivery_receipt(
    *,
    http_client: httpx.AsyncClient,
    brain_url: str,
    platform: str,
    platform_channel_id: str,
    delivery_tracking_id: str,
    platform_message_id: str,
    adapter: str,
    logger: logging.Logger,
    log_label: str,
    retry_delays: Sequence[float] = _DEFAULT_RETRY_DELAYS,
    sleep_func: Callable[[float], Awaitable[None]] | None = None,
) -> None:
    """Best-effort report of one delivered normal chat response.

    Args:
        http_client: Adapter-owned HTTP client used to call the brain service.
        brain_url: Base URL of the brain service.
        platform: Platform identifier stored on the conversation row.
        platform_channel_id: Platform channel or DM identifier.
        delivery_tracking_id: Brain-generated row tracking identifier.
        platform_message_id: Platform-generated outbound message identifier.
        adapter: Adapter implementation name to store with the receipt.
        logger: Logger used for operational receipt warnings.
        log_label: Human-readable platform label for log messages.
        retry_delays: Delay schedule for ``not_found`` races after the first
            receipt attempt.
        sleep_func: Awaitable sleep function, injectable for deterministic
            retry tests.

    Returns:
        None.
    """

    if not delivery_tracking_id or not platform_message_id:
        return

    if sleep_func is None:
        sleep_func = asyncio.sleep

    payload = {
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "delivery_tracking_id": delivery_tracking_id,
        "platform_message_id": platform_message_id,
        "adapter": adapter,
    }
    delays = (0.0,) + tuple(retry_delays)
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            await sleep_func(delay)

        try:
            response = await http_client.post(
                f"{brain_url}/delivery_receipt",
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                f"{log_label} delivery receipt failed: "
                f"delivery_tracking_id={delivery_tracking_id} "
                f"platform_message_id={platform_message_id} error={exc}"
            )
            return

        receipt_data = response.json()
        status = str(receipt_data.get("status") or "")
        if status == "updated":
            return
        if status != "not_found":
            logger.warning(
                f"{log_label} delivery receipt returned unexpected status: "
                f"delivery_tracking_id={delivery_tracking_id} "
                f"platform_message_id={platform_message_id} status={status}"
            )
            return
        if attempt == len(delays):
            logger.warning(
                f"{log_label} delivery receipt row not found after retries: "
                f"delivery_tracking_id={delivery_tracking_id} "
                f"platform_message_id={platform_message_id}"
            )
