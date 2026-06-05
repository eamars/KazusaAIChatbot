"""QQ display-name cache and hydration helpers."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from websockets.exceptions import WebSocketException

from adapters.envelope_common import normalize_mention_display_label
from kazusa_ai_chatbot.utils import log_preview

from .cq_projection import CQ_AT_PATTERN


MENTION_DISPLAY_CACHE_LIMIT = 512
ApiCaller = Callable[[object, str, dict | None], Awaitable[dict]]


def select_qq_display_name(source: dict) -> str:
    """Choose the QQ label source that matches inbound sender naming."""

    nickname = normalize_mention_display_label(source.get("nickname"))
    if nickname:
        return nickname

    name = normalize_mention_display_label(source.get("name"))
    if name:
        return name

    card = normalize_mention_display_label(source.get("card"))
    return card


def cache_qq_mention_display_name(
    mention_display_cache: OrderedDict[tuple[str, str], str],
    cache_key: tuple[str, str],
    display_name: str,
) -> None:
    """Remember a positive QQ mention label with bounded LRU retention."""

    label = normalize_mention_display_label(display_name)
    if not label:
        return

    mention_display_cache[cache_key] = label
    mention_display_cache.move_to_end(cache_key)
    while len(mention_display_cache) > MENTION_DISPLAY_CACHE_LIMIT:
        mention_display_cache.popitem(last=False)


async def lookup_qq_mention_display_name(
    *,
    platform_user_id: str,
    is_group: bool,
    group_id: object,
    ws: object,
    call_api: ApiCaller,
    logger: Any,
) -> str:
    """Resolve one QQ mention label through a bounded NapCat lookup."""

    user_param: int | str = platform_user_id
    if platform_user_id.isdigit():
        user_param = int(platform_user_id)

    if is_group and group_id is not None:
        group_param: int | str = str(group_id)
        if group_param.isdigit():
            group_param = int(group_param)
        action = "get_group_member_info"
        params: dict[str, int | str | bool] = {
            "group_id": group_param,
            "user_id": user_param,
            "no_cache": False,
        }
    else:
        action = "get_stranger_info"
        params = {"user_id": user_param}

    try:
        response = await asyncio.wait_for(
            call_api(ws, action, params),
            timeout=1.0,
        )
    except (asyncio.TimeoutError, WebSocketException) as exc:
        logger.warning(
            f"QQ mention display lookup failed: "
            f"user_id={platform_user_id} action={action} error={exc}"
        )
        return_value = ""
        return return_value

    if not isinstance(response, dict):
        logger.warning(
            f"QQ mention display lookup returned non-dict response: "
            f"user_id={platform_user_id} action={action}"
        )
        return_value = ""
        return return_value

    if response.get("status") != "ok":
        logger.warning(
            f'QQ mention display lookup returned status={response.get("status")} '
            f'user_id={platform_user_id} action={action} '
            f'message={log_preview(response.get("message"))}'
        )
        return_value = ""
        return return_value

    response_data = response.get("data", {})
    if not isinstance(response_data, dict):
        logger.warning(
            f"QQ mention display lookup returned non-dict data: "
            f"user_id={platform_user_id} action={action}"
        )
        return_value = ""
        return return_value

    display_name = select_qq_display_name(response_data)
    return display_name


async def hydrate_mention_display_names(
    *,
    raw_wire_text: str,
    initial_display_names: dict[str, str],
    channel_id: str,
    is_group: bool,
    group_id: object,
    ws: object,
    bot_id: str | None,
    bot_name: str | None,
    mention_display_cache: OrderedDict[tuple[str, str], str],
    call_api: ApiCaller,
    logger: Any,
) -> dict[str, str]:
    """Hydrate QQ mention labels before the envelope reaches the brain."""

    display_names = dict(initial_display_names)
    attempted_lookup_keys: set[tuple[str, str]] = set()
    for match in CQ_AT_PATTERN.finditer(raw_wire_text):
        platform_user_id = match.group(1)
        if platform_user_id.lower() == "all":
            continue

        if bot_id is not None and platform_user_id == bot_id:
            bot_label = normalize_mention_display_label(bot_name)
            if bot_label:
                display_names[platform_user_id] = bot_label
            continue

        cache_key = (channel_id, platform_user_id)
        existing_label = display_names.get(platform_user_id, "")
        if existing_label:
            cache_qq_mention_display_name(
                mention_display_cache,
                cache_key,
                existing_label,
            )
            continue

        cached_label = mention_display_cache.get(cache_key, "")
        if cached_label:
            mention_display_cache.move_to_end(cache_key)
            display_names[platform_user_id] = cached_label
            continue

        if cache_key in attempted_lookup_keys:
            continue
        attempted_lookup_keys.add(cache_key)

        lookup_label = await lookup_qq_mention_display_name(
            platform_user_id=platform_user_id,
            is_group=is_group,
            group_id=group_id,
            ws=ws,
            call_api=call_api,
            logger=logger,
        )
        if lookup_label:
            display_names[platform_user_id] = lookup_label
            cache_qq_mention_display_name(
                mention_display_cache,
                cache_key,
                lookup_label,
            )

    return display_names
