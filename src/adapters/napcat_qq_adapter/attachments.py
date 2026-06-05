"""Image attachment fetching and adapter-side normalization helpers."""

from __future__ import annotations

import base64
from typing import Any

import httpx


def image_segments(message_data: object) -> list[dict]:
    """Return image segments from a NapCat message list."""

    if not isinstance(message_data, list):
        return_value: list[dict] = []
        return return_value

    segments = [
        segment for segment in message_data
        if isinstance(segment, dict) and segment.get("type") == "image"
    ]
    return segments


async def fetch_image_attachments(
    *,
    message_data: object,
    http_client: httpx.AsyncClient,
    logger: Any,
) -> list[dict[str, str]]:
    """Fetch NapCat image segment URLs for current-turn attachment refs."""

    attachments: list[dict[str, str]] = []
    for segment in image_segments(message_data):
        segment_data = segment.get("data", {})
        if not isinstance(segment_data, dict):
            continue
        url = segment_data.get("url")
        if not url:
            continue
        try:
            response = await http_client.get(url, timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.exception(f"Image fetch error: {exc}")
            continue

        attachments.append({
            "media_type": "image/jpeg",
            "base64_data": base64.b64encode(response.content).decode("utf-8"),
            "description": "",
        })

    return attachments
