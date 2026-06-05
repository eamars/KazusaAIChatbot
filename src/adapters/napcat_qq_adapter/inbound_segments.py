"""NapCat/OneBot segment-list conversion into canonical CQ wire text."""

from __future__ import annotations

import re

from .mention_hydration import select_qq_display_name


CQ_REPLY_ID_PATTERN = re.compile(r"\[CQ:reply,id=([^\],]+)[^\]]*\]")


def normalize_inbound_wire_message(
    message_data: object,
) -> tuple[str, dict[str, str], dict[str, str]]:
    """Convert inbound NapCat message content into canonical wire text.

    Args:
        message_data: NapCat ``message`` value, either a CQ string or a segment
            list with OneBot-like segment dictionaries.

    Returns:
        A tuple of canonical wire text, reply context, and mention display names
        discovered directly in the platform payload.
    """

    reply_context: dict[str, str] = {}
    mention_display_names: dict[str, str] = {}

    if isinstance(message_data, str):
        wire_content = message_data
        reply_match = CQ_REPLY_ID_PATTERN.search(wire_content)
        if reply_match is not None:
            reply_context["reply_to_message_id"] = reply_match.group(1)
        return_value = (wire_content, reply_context, mention_display_names)
        return return_value

    wire_parts: list[str] = []
    if not isinstance(message_data, list):
        return_value = ("", reply_context, mention_display_names)
        return return_value

    for segment in message_data:
        if not isinstance(segment, dict):
            continue
        segment_type = segment.get("type")
        segment_data = segment.get("data", {})
        if not isinstance(segment_data, dict):
            segment_data = {}
        if segment_type == "text":
            wire_parts.append(str(segment_data.get("text", "")))
        elif segment_type == "at":
            qq = segment_data.get("qq")
            platform_user_id = str(qq or "")
            label = select_qq_display_name(segment_data)
            if label:
                mention_display_names[platform_user_id] = label
            wire_parts.append(f"[CQ:at,qq={platform_user_id}]")
        elif segment_type == "reply":
            reply_context["reply_to_message_id"] = str(segment_data.get("id", ""))
            reply_sender_id = segment_data.get("user_id")
            if reply_sender_id is not None:
                reply_context["reply_to_platform_user_id"] = str(reply_sender_id)
            reply_sender_name = segment_data.get("nickname")
            if reply_sender_name:
                reply_context["reply_to_display_name"] = str(reply_sender_name)
            reply_text = segment_data.get("text")
            if reply_text:
                reply_context["reply_excerpt"] = str(reply_text)
            wire_parts.append(f"[CQ:reply,id={segment_data.get('id', '')}]")
        elif segment_type == "face":
            face_id = segment_data.get("id", "")
            wire_parts.append(f"[CQ:face,id={face_id}]")
        elif segment_type == "image":
            image_url = segment_data.get("url", "")
            wire_parts.append(f"[CQ:image,url={image_url}]")

    wire_content = "".join(wire_parts)
    return_value = (wire_content, reply_context, mention_display_names)
    return return_value
