"""Group-channel consolidation persistence helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.db import upsert_group_channel_style_image


async def persist_group_channel_style_image(
    *,
    platform: str,
    platform_channel_id: str,
    overlay: Mapping[str, Any],
    source_reflection_run_ids: list[str],
    storage_timestamp_utc: str,
) -> dict[str, Any]:
    """Persist the validated style overlay for one group channel.

    Args:
        platform: Platform namespace for the group channel.
        platform_channel_id: Platform channel or group identifier.
        overlay: Candidate interaction-style overlay.
        source_reflection_run_ids: Reflection run ids that support the write.
        storage_timestamp_utc: Storage UTC timestamp for the write.

    Returns:
        The persisted group-channel interaction-style image document.
    """

    document = await upsert_group_channel_style_image(
        platform=platform,
        platform_channel_id=platform_channel_id,
        overlay=dict(overlay),
        source_reflection_run_ids=source_reflection_run_ids,
        storage_timestamp_utc=storage_timestamp_utc,
    )
    return_value = dict(document)
    return return_value
