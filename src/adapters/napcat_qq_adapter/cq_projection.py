"""CQ marker parsing and QQ semantic text projection."""

from __future__ import annotations

from collections.abc import Mapping
import re

from adapters.envelope_common import normalize_body_spacing, readable_mention_token

from .face_catalog import qq_face_image_description


CQ_REPLY_PATTERN = re.compile(r"\[CQ:reply,id=([^\],]+)[^\]]*\]")
CQ_AT_PATTERN = re.compile(r"\[CQ:at,qq=([^\],]+)[^\]]*\]")
CQ_FACE_PATTERN = re.compile(r"\[CQ:face(?P<params>(?:,[^\]]*)?)\]")
CQ_ANY_PATTERN = re.compile(r"\[CQ:[^\]]+\]")


def qq_mention_entity_kind(platform_user_id: str, platform_bot_id: str) -> str:
    """Classify a QQ mention target without interpreting message semantics."""

    if platform_user_id.lower() == "all":
        return_value = "everyone"
    elif platform_user_id == platform_bot_id:
        return_value = "bot"
    else:
        return_value = "user"
    return return_value


def cq_param_value(params: str, field_name: str) -> str:
    """Return one CQ segment parameter value from a comma-delimited tail."""

    for raw_param in params.split(","):
        if "=" not in raw_param:
            continue
        key, value = raw_param.split("=", 1)
        if key.strip() != field_name:
            continue
        return_value = value.strip()
        return return_value

    return_value = ""
    return return_value


def _escape_image_description(description: str) -> str:
    """Escape literal image-boundary characters in a QQ face description."""

    escaped = description.replace("&", "&amp;")
    escaped = escaped.replace("<", "&lt;")
    escaped = escaped.replace(">", "&gt;")
    return_value = escaped
    return return_value


def _qq_face_image_block(face_id: str) -> str | None:
    """Render one known QQ system face as prompt-facing visual text."""

    description = qq_face_image_description(face_id)
    if description is None:
        return_value = None
        return return_value

    escaped_description = _escape_image_description(description)
    return_value = f"<image>{escaped_description}</image>"
    return return_value


def project_qq_semantic_text(
    raw_wire_text: str,
    platform_bot_id: str,
    display_names: Mapping[str, str],
) -> str:
    """Project QQ wire text into adapter-owned semantic body text.

    Args:
        raw_wire_text: QQ wire text that may contain CQ transport markers.
        platform_bot_id: QQ id of the bot account receiving the event.
        display_names: Optional QQ id to display-name map for visible mentions.

    Returns:
        Clean semantic text suitable for `MessageEnvelope.body_text` or
        `MessageEnvelope.reply.excerpt`.
    """

    def replacement(match: re.Match[str]) -> str:
        platform_user_id = match.group(1)
        entity_kind = qq_mention_entity_kind(
            platform_user_id,
            platform_bot_id,
        )
        token = readable_mention_token(
            entity_kind=entity_kind,
            display_name=display_names.get(platform_user_id, ""),
            raw_label=platform_user_id if entity_kind == "everyone" else "",
        )
        return_value = f" {token} "
        return return_value

    def face_replacement(match: re.Match[str]) -> str:
        params = match.group("params")
        face_id = cq_param_value(params, "id")
        image_block = _qq_face_image_block(face_id)
        if image_block is None:
            return_value = ""
            return return_value
        return_value = f" {image_block} "
        return return_value

    body_text = CQ_REPLY_PATTERN.sub(" ", raw_wire_text)
    body_text = CQ_AT_PATTERN.sub(replacement, body_text)
    body_text = CQ_FACE_PATTERN.sub(face_replacement, body_text)
    body_text = CQ_ANY_PATTERN.sub(" ", body_text)
    projected_text = normalize_body_spacing(body_text)
    return projected_text
