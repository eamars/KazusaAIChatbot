"""Post-turn persistence helpers for the brain service."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, TypeAlias

from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
)
from kazusa_ai_chatbot.action_spec.results import build_episode_trace
from kazusa_ai_chatbot.conversation_progress import (
    ConversationProgressRecordInput,
    ConversationProgressScope,
)
from kazusa_ai_chatbot.brain_service.outbound import (
    record_assistant_outbound_message,
    utc_timestamp,
)
from kazusa_ai_chatbot.utils import log_preview


EnsureCharacterIdentity = Callable[..., Awaitable[str]]
SaveConversation = Callable[[dict], Awaitable[str | None]]
CallConsolidation = Callable[[dict], Awaitable[dict]]
UpdateCharacterRuntimeState = Callable[[dict], Awaitable[None]]
RecordTurnProgress = Callable[..., Awaitable[dict]]
RecordResidue = Callable[..., Awaitable[dict]]
ActiveCommitmentReader: TypeAlias = Callable[..., Awaitable[dict[str, object]]]
PostSurfaceMemoryLifecycleReview: TypeAlias = Callable[
    ...,
    Awaitable[dict[str, object]],
]
ExecuteActionSpecsForTrace: TypeAlias = Callable[
    ...,
    Awaitable[list[dict[str, Any]]],
]

POST_SURFACE_ACTIVE_COMMITMENT_REVIEW_LIMIT = 500
POST_SURFACE_LIFECYCLE_MAX_PASSES = 5


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
        logger: Retained for service call-site compatibility.

    Returns:
        None.
    """

    platform = result["platform"]
    platform_channel_id = result["platform_channel_id"]
    platform_bot_id = result["platform_bot_id"]
    character_name = result["character_name"]
    assistant_output: list[str] = []
    for fragment in result["final_dialog"]:
        clean_fragment = str(fragment).strip()
        if clean_fragment:
            assistant_output.append(clean_fragment)
    del logger

    if not assistant_output:
        return

    raw_delivery_mentions = result.get("delivery_mentions")
    if isinstance(raw_delivery_mentions, list):
        delivery_mentions = raw_delivery_mentions
    else:
        delivery_mentions = None

    for logical_message_index, body_text in enumerate(assistant_output):
        await record_assistant_outbound_message(
            platform=platform,
            platform_channel_id=platform_channel_id,
            channel_type=result["channel_type"],
            platform_bot_id=platform_bot_id,
            character_name=character_name,
            body_text=body_text,
            addressed_to_global_user_ids=result["target_addressed_user_ids"],
            broadcast=bool(result["target_broadcast"]),
            fallback_addressed_global_user_id=str(result["global_user_id"]),
            delivery_tracking_id=str(result.get("delivery_tracking_id") or ""),
            logical_message_index=logical_message_index,
            llm_trace_id=str(result.get("llm_trace_id") or ""),
            storage_timestamp_utc=utc_timestamp(now_func),
            ensure_character_global_identity_func=(
                ensure_character_global_identity_func
            ),
            save_conversation_func=save_conversation_func,
            mentions=delivery_mentions,
        )


async def run_post_turn_memory_lifecycle_background(
    state: dict,
    *,
    active_commitment_reader: ActiveCommitmentReader,
    review_func: PostSurfaceMemoryLifecycleReview,
    execute_action_specs_func: ExecuteActionSpecsForTrace,
    logger: logging.Logger,
    no_remember: bool,
    visible_response_sent: bool,
    think_only_suppressed: bool,
) -> dict:
    """Close fulfilled active commitments after visible response delivery.

    Args:
        state: Completed persona state passed to post-turn memory consumers.
        active_commitment_reader: Direct DB reader for active commitments.
        review_func: LLM-owned post-surface lifecycle review function.
        execute_action_specs_func: Existing action-spec executor.
        logger: Service logger for skip and saturation diagnostics.
        no_remember: Whether this turn forbids memory side effects.
        visible_response_sent: Whether the user received a visible response.
        think_only_suppressed: Whether visible text was hidden by think-only.

    Returns:
        The original state object when review is structurally skipped, or a
        shallow copied state with lifecycle specs/results and rebuilt trace.
    """

    if no_remember:
        return state
    if not visible_response_sent:
        return state
    if think_only_suppressed:
        return state

    global_user_id = _state_text(state, "global_user_id")
    if not global_user_id:
        return state
    if not _visible_final_dialog(state):
        return state
    if _has_executed_lifecycle_result(state):
        return state

    current_state = state
    for _ in range(POST_SURFACE_LIFECYCLE_MAX_PASSES):
        read_result = await active_commitment_reader(
            global_user_id=global_user_id,
            limit=POST_SURFACE_ACTIVE_COMMITMENT_REVIEW_LIMIT,
        )
        if read_result.get("limit_exceeded") is True:
            logger.warning(
                f"Post-turn lifecycle review skipped: active commitments "
                f"exceed limit={POST_SURFACE_ACTIVE_COMMITMENT_REVIEW_LIMIT} "
                f"user={global_user_id}"
            )
            return current_state

        documents = _document_rows(read_result.get("documents"))
        if not documents:
            return current_state

        review_result = await review_func(current_state, documents)
        lifecycle_specs = _apply_lifecycle_action_specs(review_result)
        if not lifecycle_specs:
            return current_state

        action_results = await execute_action_specs_func(
            lifecycle_specs,
            storage_timestamp_utc=current_state["storage_timestamp_utc"],
        )
        executed_results = _executed_lifecycle_results(action_results)
        if not executed_results:
            return current_state

        current_state = _append_lifecycle_trace(
            current_state,
            lifecycle_specs=lifecycle_specs,
            action_results=action_results,
        )

    return current_state


def _state_text(state: dict, field_name: str) -> str:
    """Read one optional string field from post-turn state."""

    value = state.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _visible_final_dialog(state: dict) -> list[str]:
    """Return non-empty visible final dialog fragments."""

    raw_final_dialog = state.get("final_dialog")
    if not isinstance(raw_final_dialog, list):
        return_value: list[str] = []
        return return_value
    final_dialog = [
        fragment
        for fragment in raw_final_dialog
        if isinstance(fragment, str) and fragment.strip()
    ]
    return final_dialog


def _has_executed_lifecycle_result(state: dict) -> bool:
    """Return whether this turn already executed lifecycle updates."""

    raw_results = state.get("action_results")
    if not isinstance(raw_results, list):
        return_value = False
        return return_value
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue
        if raw_result.get("action_kind") != APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
            continue
        if raw_result.get("status") == "executed":
            return_value = True
            return return_value
    return_value = False
    return return_value


def _document_rows(raw_documents: object) -> list[dict[str, object]]:
    """Return direct DB documents that have dictionary shape."""

    if not isinstance(raw_documents, list):
        return_value: list[dict[str, object]] = []
        return return_value
    documents = [
        document
        for document in raw_documents
        if isinstance(document, dict)
    ]
    return documents


def _apply_lifecycle_action_specs(
    review_result: dict[str, object],
) -> list[dict[str, Any]]:
    """Keep only executable memory-lifecycle action specs."""

    raw_specs = review_result.get("action_specs")
    if not isinstance(raw_specs, list):
        return_value: list[dict[str, Any]] = []
        return return_value
    lifecycle_specs = [
        action_spec
        for action_spec in raw_specs
        if (
            isinstance(action_spec, dict)
            and action_spec.get("kind") == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY
        )
    ]
    return lifecycle_specs


def _executed_lifecycle_results(
    action_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return executed memory-lifecycle action results."""

    executed_results = [
        action_result
        for action_result in action_results
        if (
            isinstance(action_result, dict)
            and action_result.get("action_kind")
            == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY
            and action_result.get("status") == "executed"
        )
    ]
    return executed_results


def _append_lifecycle_trace(
    state: dict,
    *,
    lifecycle_specs: list[dict[str, Any]],
    action_results: list[dict[str, Any]],
) -> dict:
    """Append lifecycle evidence and rebuild the episode trace."""

    updated_state = dict(state)
    prior_specs = _dict_list(updated_state.get("action_specs"))
    prior_results = _dict_list(updated_state.get("action_results"))
    surface_outputs = _dict_list(updated_state.get("surface_outputs"))
    updated_specs = prior_specs + lifecycle_specs
    updated_results = prior_results + action_results
    episode = updated_state["cognitive_episode"]
    existing_trace = updated_state.get("episode_trace")
    cognition_refs: list[dict[str, object]] | None = None
    if isinstance(existing_trace, dict):
        raw_cognition_refs = existing_trace.get("cognition_refs")
        if isinstance(raw_cognition_refs, list):
            cognition_refs = [
                ref for ref in raw_cognition_refs if isinstance(ref, dict)
            ]
    trace = build_episode_trace(
        episode_id=episode["episode_id"],
        trigger_source=episode["trigger_source"],
        created_at=updated_state["storage_timestamp_utc"],
        action_specs=updated_specs,
        action_results=updated_results,
        surface_outputs=surface_outputs,
        cognition_refs=cognition_refs,
    )
    updated_state["action_specs"] = updated_specs
    updated_state["action_results"] = updated_results
    updated_state["episode_trace"] = trace
    return updated_state


def _dict_list(raw_value: object) -> list[dict[str, Any]]:
    """Return dictionary rows from an optional list field."""

    if not isinstance(raw_value, list):
        return_value: list[dict[str, Any]] = []
        return return_value
    rows = [
        row
        for row in raw_value
        if isinstance(row, dict)
    ]
    return rows


async def run_consolidation_background(
    state: dict,
    *,
    call_consolidation_subgraph_func: CallConsolidation,
    update_character_runtime_state_func: UpdateCharacterRuntimeState,
    logger: logging.Logger,
) -> None:
    """Run consolidation after the dialog has already been returned.

    Args:
        state: Persona graph state snapshot needed by the consolidator.
        call_consolidation_subgraph_func: Service-level consolidator callable.
        update_character_runtime_state_func: Service callback that refreshes
            the process-local runtime character state after a successful write.
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

    await update_character_runtime_state_func(result)


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

    character_profile = state["character_profile"]
    boundary_profile = character_profile["boundary_profile"]
    surface_output = state.get("text_surface_output_v2")
    if isinstance(surface_output, dict):
        content_plan = {
            "semantic_content": surface_output["content_plan"],
            "surface_intent": surface_output["selected_surface_intent"],
            "style_guidance": surface_output["style_guidance"],
        }
    else:
        content_plan = {
            "semantic_content": state["character_intent"],
            "surface_intent": state["logical_stance"],
        }
    scope = ConversationProgressScope(
        platform=state["platform"],
        platform_channel_id=state["platform_channel_id"],
        global_user_id=state["global_user_id"],
    )
    record_input: ConversationProgressRecordInput = {
        "scope": scope,
        "storage_timestamp_utc": state["storage_timestamp_utc"],
        "character_name": character_profile["name"],
        "prior_episode_state": state.get("conversation_episode_state"),
        "decontexualized_input": state["decontexualized_input"],
        "chat_history_recent": state["chat_history_recent"],
        "content_plan": content_plan,
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "final_dialog": state["final_dialog"],
        "boundary_profile": boundary_profile,
    }
    record_preview = {
        "storage_timestamp_utc": record_input["storage_timestamp_utc"],
        "character_name": record_input["character_name"],
        "prior_episode_state": record_input["prior_episode_state"],
        "decontexualized_input": record_input["decontexualized_input"],
        "chat_history_recent": record_input["chat_history_recent"],
        "content_plan": record_input["content_plan"],
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


async def run_internal_monologue_residue_record_background(
    state: dict,
    *,
    record_completed_episode_residue_func: RecordResidue,
    logger: logging.Logger,
    current_timestamp_utc: str | None = None,
) -> None:
    """Record compact private residue after an episode has completed."""

    record_timestamp_utc = current_timestamp_utc
    if record_timestamp_utc is None:
        record_timestamp_utc = str(state["storage_timestamp_utc"])

    try:
        result = await record_completed_episode_residue_func(
            completed_state=state,
            current_timestamp_utc=record_timestamp_utc,
        )
    except Exception as exc:
        logger.exception(
            f"Background internal monologue residue recording failed: {exc}"
        )
        return

    result_status = result.get("status", "")
    result_written = result.get("written", False)
    result_retry_count = result.get("retry_count", 0)
    logger.debug(
        f"Internal monologue residue recorded: "
        f"platform={state['platform']} "
        f'channel={state["platform_channel_id"] or "<dm>"} '
        f'user={state["global_user_id"]} status={result_status} '
        f'written={result_written} retry_count={result_retry_count}'
    )
