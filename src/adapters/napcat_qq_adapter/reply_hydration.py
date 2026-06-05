"""Reply target hydration through NapCat platform metadata."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from websockets.exceptions import WebSocketException

from .cq_projection import project_qq_semantic_text


ApiCaller = Callable[[object, str, dict | None], Awaitable[dict]]


def apply_replied_message_metadata(
    reply_context: dict[str, str | bool],
    message_data: dict,
    *,
    bot_id: str,
) -> None:
    """Populate reply target fields from a NapCat/OneBot message document."""

    sender = message_data.get("sender")
    if not isinstance(sender, dict):
        sender = {}

    target_user_id = message_data.get("user_id") or sender.get("user_id")
    if target_user_id is not None:
        reply_context["reply_to_platform_user_id"] = str(target_user_id)

    target_name = sender.get("card") or sender.get("nickname")
    if target_name:
        reply_context["reply_to_display_name"] = str(target_name)

    reply_excerpt = message_data.get("raw_message") or message_data.get("message")
    if isinstance(reply_excerpt, str) and reply_excerpt:
        semantic_excerpt = project_qq_semantic_text(
            reply_excerpt,
            bot_id,
            {},
        )
        if semantic_excerpt:
            reply_context["reply_excerpt"] = semantic_excerpt


async def hydrate_reply_context_from_platform(
    reply_context: dict[str, str | bool],
    ws: object,
    *,
    call_api: ApiCaller,
    bot_id: str,
    logger: Any,
) -> None:
    """Resolve reply target metadata from NapCat before calling the brain."""

    reply_to_message_id = reply_context.get("reply_to_message_id")
    if not reply_to_message_id or reply_context.get("reply_to_platform_user_id"):
        return

    params: dict[str, int | str] = {"message_id": str(reply_to_message_id)}
    if str(reply_to_message_id).isdigit():
        params["message_id"] = int(str(reply_to_message_id))

    try:
        response = await call_api(ws, "get_msg", params)
    except (asyncio.TimeoutError, WebSocketException) as exc:
        logger.warning(
            f"Failed to resolve QQ reply target "
            f"message_id={reply_to_message_id}: {exc}"
        )
        return

    if response.get("status") != "ok":
        logger.warning(
            f'QQ reply target lookup returned status={response.get("status")} '
            f"message_id={reply_to_message_id}"
        )
        return

    message_data = response.get("data", {})
    if not isinstance(message_data, dict):
        logger.warning(
            f"QQ reply target lookup returned non-dict data "
            f"for message_id={reply_to_message_id}"
        )
        return

    apply_replied_message_metadata(reply_context, message_data, bot_id=bot_id)
