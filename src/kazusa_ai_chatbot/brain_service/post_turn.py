"""Post-turn persistence helpers for the brain service."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime

from kazusa_ai_chatbot.conversation_progress import (
    ConversationProgressRecordInput,
    ConversationProgressScope,
)
from kazusa_ai_chatbot.utils import log_preview


EnsureCharacterIdentity = Callable[..., Awaitable[str]]
SaveConversation = Callable[[dict], Awaitable[None]]
CallConsolidation = Callable[[dict], Awaitable[dict]]
RecordTurnProgress = Callable[..., Awaitable[dict]]


async def save_assistant_message(
    result: dict,
    *,
    ensure_character_global_identity_func: EnsureCharacterIdentity,
    save_conversation_func: SaveConversation,
    now_func: Callable[[], datetime],
    logger: logging.Logger,
) -> None:
    """Persist the assistant-authored response to conversation history.

    Args:
        result: Final graph state containing assistant output and addressees.
        ensure_character_global_identity_func: Service identity backfill helper.
        save_conversation_func: Service-level persistence function.
        now_func: Clock function used for assistant-row timestamps.
        logger: Logger used for compatibility with service logging.

    Returns:
        None.
    """

    platform = result["platform"]
    platform_channel_id = result["platform_channel_id"]
    platform_bot_id = result["platform_bot_id"]
    character_name = result["character_name"]
    assistant_output = result["final_dialog"]

    if assistant_output:
        body_text = "\n".join(assistant_output)
        target_broadcast = bool(result["target_broadcast"])
        target_addressed_user_ids = result["target_addressed_user_ids"]
        if not target_addressed_user_ids and not target_broadcast:
            current_user_id = str(result["global_user_id"]).strip()
            target_addressed_user_ids = [current_user_id]
        try:
            character_global_user_id = await ensure_character_global_identity_func(
                platform=platform,
                platform_bot_id=platform_bot_id,
                character_name=character_name,
            )
            await save_conversation_func({
                "platform": platform,
                "platform_channel_id": platform_channel_id,
                "channel_type": result["channel_type"],
                "role": "assistant",
                "platform_user_id": platform_bot_id,
                "global_user_id": character_global_user_id,
                "display_name": character_name,
                "body_text": body_text,
                "raw_wire_text": body_text,
                "content_type": "text",
                "addressed_to_global_user_ids": target_addressed_user_ids,
                "mentions": [],
                "broadcast": target_broadcast,
                "attachments": [],
                "timestamp": now_func().isoformat(),
            })
        except Exception as exc:
            logger.exception(f"Failed to save assistant message: {exc}")


async def run_consolidation_background(
    state: dict,
    *,
    call_consolidation_subgraph_func: CallConsolidation,
    personality: dict,
    logger: logging.Logger,
) -> None:
    """Run consolidation after the dialog has already been returned.

    Args:
        state: Persona graph state snapshot needed by the consolidator.
        call_consolidation_subgraph_func: Service-level consolidator callable.
        personality: Mutable character profile cache owned by the service.
        logger: Logger used for compatibility with service logging.

    Returns:
        None.
    """

    try:
        result = await call_consolidation_subgraph_func(state)
    except Exception as exc:
        logger.exception(f"Background consolidation failed: {exc}")
        return

    metadata = result.get("consolidation_metadata") or {}
    write_success = metadata.get("write_success") or {}
    if not write_success.get("character_state"):
        return

    for field_name in ("mood", "global_vibe", "reflection_summary"):
        field_value = result.get(field_name)
        if isinstance(field_value, str) and field_value:
            personality[field_name] = field_value


async def run_conversation_progress_record_background(
    state: dict,
    *,
    record_turn_progress_func: RecordTurnProgress,
    logger: logging.Logger,
) -> None:
    """Record conversation progress after dialog output has been returned.

    Args:
        state: Persona graph state snapshot needed by the progress recorder.
        record_turn_progress_func: Service-level progress recorder callable.
        logger: Logger used for compatibility with service logging.

    Returns:
        None.
    """

    linguistic_directives = state["action_directives"]["linguistic_directives"]
    character_profile = state["character_profile"]
    boundary_profile = character_profile["boundary_profile"]
    scope = ConversationProgressScope(
        platform=state["platform"],
        platform_channel_id=state["platform_channel_id"],
        global_user_id=state["global_user_id"],
    )
    record_input: ConversationProgressRecordInput = {
        "scope": scope,
        "timestamp": state["timestamp"],
        "prior_episode_state": state.get("conversation_episode_state"),
        "decontexualized_input": state["decontexualized_input"],
        "chat_history_recent": state["chat_history_recent"],
        "content_anchors": linguistic_directives["content_anchors"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "final_dialog": state["final_dialog"],
        "boundary_profile": boundary_profile,
    }
    record_preview = {
        "timestamp": record_input["timestamp"],
        "prior_episode_state": record_input["prior_episode_state"],
        "decontexualized_input": record_input["decontexualized_input"],
        "chat_history_recent": record_input["chat_history_recent"],
        "content_anchors": record_input["content_anchors"],
        "logical_stance": record_input["logical_stance"],
        "character_intent": record_input["character_intent"],
        "final_dialog": record_input["final_dialog"],
        "boundary_profile_supplied": True,
    }
    logger.debug(
        f"Conversation progress record request detail: "
        f"platform={scope.platform} "
        f'channel={scope.platform_channel_id or "<dm>"} '
        f"user={scope.global_user_id} input={log_preview(record_preview)}"
    )

    try:
        result = await record_turn_progress_func(record_input=record_input)
    except Exception as exc:
        logger.exception(
            f"Background conversation progress recording failed: {exc}"
        )
        return

    logger.debug(
        f"Conversation progress recorded: platform={scope.platform} "
        f'channel={scope.platform_channel_id or "<dm>"} '
        f'user={scope.global_user_id} written={result["written"]} '
        f'turn_count={result["turn_count"]} '
        f'continuity={result["continuity"]} status={result["status"]} '
        f'cache_updated={result["cache_updated"]}'
    )

