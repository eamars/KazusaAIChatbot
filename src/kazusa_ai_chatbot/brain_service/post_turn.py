"""Post-turn persistence helpers for the brain service."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
)
from kazusa_ai_chatbot.action_spec.models import ActionSpecV1
from kazusa_ai_chatbot.action_spec.results import (
    ActionResultV1,
    DeliveryCorrelationV1,
    EpisodeAttemptDiagnosticV1,
    EpisodeTerminalStatusV1,
    EpisodeTraceV2,
    SurfaceOutputV1,
    build_text_surface_output,
    validate_episode_trace_v2,
)
from kazusa_ai_chatbot.conversation_progress import (
    ConversationProgressRecordInput,
    ConversationProgressScope,
)
from kazusa_ai_chatbot.brain_service.outbound import (
    record_assistant_outbound_message,
    utc_timestamp,
)
from kazusa_ai_chatbot.db.schemas import PostTurnLifecycleRecordV1
from kazusa_ai_chatbot.logging_retention import expiry_from_storage_iso
from kazusa_ai_chatbot.utils import log_preview

if TYPE_CHECKING:
    from kazusa_ai_chatbot.cognition_episode import CognitiveEpisodeV1


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
IssueInternalActionLatches: TypeAlias = Callable[..., Awaitable[None]]

POST_SURFACE_ACTIVE_COMMITMENT_REVIEW_LIMIT = 500
POST_SURFACE_LIFECYCLE_MAX_PASSES = 5


def settle_episode_trace(
    *,
    episode: CognitiveEpisodeV1,
    cognition_output: dict[str, object] | None,
    action_specs: Sequence[ActionSpecV1],
    action_results: Sequence[ActionResultV1],
    surface_outputs: Sequence[SurfaceOutputV1],
    terminal_status: EpisodeTerminalStatusV1,
    attempt_diagnostics: Sequence[EpisodeAttemptDiagnosticV1],
    delivery_correlation: DeliveryCorrelationV1,
    settled_at: str,
) -> EpisodeTraceV2:
    """Settle the sole immutable EpisodeTraceV2 for one episode."""

    episode_id = episode.get("episode_id")
    trigger_source = episode.get("trigger_source")
    created_at = episode.get("created_at")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode trace episode_id is required")
    if not isinstance(trigger_source, str) or not trigger_source:
        raise ValueError("episode trace trigger_source is required")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("episode trace created_at is required")
    if terminal_status not in {
        "completed_visible",
        "completed_private",
        "completed_action",
        "scheduled",
        "failed",
        "cancelled",
    }:
        raise ValueError("episode trace terminal status is invalid")
    receipt_status = delivery_correlation.get("receipt_status")
    if receipt_status not in {
        "not_applicable",
        "pending",
        "delivered",
        "failed",
        "unknown",
    }:
        raise ValueError("episode trace delivery receipt status is invalid")
    tracking_id = delivery_correlation.get("tracking_id", "")
    if receipt_status == "pending" and not tracking_id:
        raise ValueError("pending delivery requires a tracking id")

    normalized_cognition_refs: list[dict[str, object]] = []
    if cognition_output is not None:
        raw_refs = cognition_output.get("cognition_refs")
        if isinstance(raw_refs, list):
            normalized_cognition_refs = [
                dict(ref) for ref in raw_refs if isinstance(ref, dict)
            ]
        if not normalized_cognition_refs:
            diagnostics = cognition_output.get("diagnostics")
            run_id = (
                diagnostics.get("run_id")
                if isinstance(diagnostics, dict)
                else None
            )
            if isinstance(run_id, str) and run_id:
                normalized_cognition_refs = [{
                    "schema_version": "evidence_ref.v1",
                    "evidence_kind": "cognition_output",
                    "evidence_id": run_id,
                    "owner": "cognition_core_v2",
                    "observed_at": settled_at,
                }]
    attempt_ids: set[str] = set()
    normalized_results: list[ActionResultV1] = []
    for result in action_results:
        action_attempt_id = result.get("action_attempt_id")
        if not isinstance(action_attempt_id, str) or not action_attempt_id:
            raise ValueError("episode trace action attempt id is required")
        if action_attempt_id in attempt_ids:
            raise ValueError("episode trace contains duplicate action attempts")
        attempt_ids.add(action_attempt_id)
        normalized_results.append(dict(result))

    trace: EpisodeTraceV2 = {
        "schema_version": "episode_trace.v2",
        "episode_id": episode_id,
        "trigger_source": trigger_source,
        "terminal_status": terminal_status,
        "cognition_refs": normalized_cognition_refs,
        "action_specs": [dict(spec) for spec in action_specs],
        "action_results": normalized_results,
        "surface_outputs": [dict(output) for output in surface_outputs],
        "attempt_diagnostics": [
            dict(diagnostic) for diagnostic in attempt_diagnostics
        ],
        "delivery_correlation": dict(delivery_correlation),
        "created_at": created_at,
        "settled_at": settled_at,
    }
    return validate_episode_trace_v2(trace)


async def settle_runtime_episode_trace(
    *,
    episode: CognitiveEpisodeV1,
    graph_result: Mapping[str, object],
    response_dialog: Sequence[str],
    delivery_tracking_id: str,
    settled_at: str,
    issue_internal_action_latches_func: IssueInternalActionLatches | None = None,
) -> dict[str, object]:
    """Normalize one runtime outcome and settle its sole episode trace.

    The service and background workers provide runtime-owned outcome facts to
    this boundary.  This function owns the normalization, terminal-status
    selection, delivery correlation fallback, and the one call to the pure
    trace constructor.
    """

    action_specs = _dict_rows(graph_result.get("action_specs"))
    action_results = _dict_rows(graph_result.get("action_results"))
    surface_outputs = _dict_rows(graph_result.get("surface_outputs"))
    if not surface_outputs and response_dialog:
        surface_outputs = [build_text_surface_output(
            fragments=[str(fragment) for fragment in response_dialog],
            created_at=settled_at,
        )]

    raw_cognition_output = graph_result.get("cognition_core_output")
    if not isinstance(raw_cognition_output, Mapping):
        raw_cognition_output = graph_result.get("cognition_output")
    cognition_output = (
        dict(raw_cognition_output)
        if isinstance(raw_cognition_output, Mapping)
        else None
    )

    terminal_status = _runtime_terminal_status(
        graph_result=graph_result,
        response_dialog=response_dialog,
        action_results=action_results,
        cognition_output=cognition_output,
    )
    delivery_correlation = _runtime_delivery_correlation(
        graph_result=graph_result,
        response_dialog=response_dialog,
        delivery_tracking_id=delivery_tracking_id,
    )
    trace = settle_episode_trace(
        episode=episode,
        cognition_output=cognition_output,
        action_specs=action_specs,
        action_results=action_results,
        surface_outputs=surface_outputs,
        terminal_status=terminal_status,
        attempt_diagnostics=_dict_rows(
            graph_result.get("attempt_diagnostics"),
        ),
        delivery_correlation=delivery_correlation,
        settled_at=settled_at,
    )
    if issue_internal_action_latches_func is not None:
        await issue_internal_action_latches_func(
            episode=episode,
            trace=trace,
            now=settled_at,
        )
    return dict(trace)


def _dict_rows(value: object) -> list[dict[str, Any]]:
    """Copy dictionary rows from an optional runtime component list."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _runtime_terminal_status(
    *,
    graph_result: Mapping[str, object],
    response_dialog: Sequence[str],
    action_results: Sequence[Mapping[str, object]],
    cognition_output: Mapping[str, object] | None,
) -> EpisodeTerminalStatusV1:
    """Choose a terminal status from settled runtime facts."""

    requested_status = graph_result.get("terminal_status")
    if requested_status in {
        "completed_visible",
        "completed_private",
        "completed_action",
        "scheduled",
        "failed",
        "cancelled",
    }:
        return requested_status
    if response_dialog:
        return "completed_visible"
    statuses = {str(row.get("status") or "") for row in action_results}
    if statuses and statuses <= {"failed", "rejected", "cancelled"}:
        return "failed"
    if "scheduled" in statuses or "pending" in statuses:
        return "scheduled"
    route = ""
    if isinstance(cognition_output, Mapping):
        intention = cognition_output.get("intention")
        if isinstance(intention, Mapping):
            route = str(intention.get("route") or "")
    if route == "action" or action_results:
        return "completed_action"
    return "completed_private"


def _runtime_delivery_correlation(
    *,
    graph_result: Mapping[str, object],
    response_dialog: Sequence[str],
    delivery_tracking_id: str,
) -> DeliveryCorrelationV1:
    """Use a settled delivery projection or build the runtime fallback."""

    supplied = graph_result.get("delivery_correlation")
    if isinstance(supplied, Mapping):
        return dict(supplied)
    return {
        "schema_version": "delivery_correlation.v1",
        "delivery_intent": "deliver_now" if response_dialog else "do_not_deliver",
        "tracking_id": delivery_tracking_id,
        "receipt_status": "pending" if delivery_tracking_id else "not_applicable",
        "receipt_ref": "",
    }


def build_post_turn_lifecycle_record(
    *,
    source_episode_id: str,
    delivery_tracking_id: str,
    action_specs: Sequence[ActionSpecV1],
    action_results: Sequence[ActionResultV1],
    error_codes: Sequence[str],
    created_at: str,
) -> PostTurnLifecycleRecordV1:
    """Build the prompt-safe post-turn lifecycle audit projection."""

    projections: list[dict[str, object]] = []
    for result in action_results:
        action_kind = str(result.get("action_kind", ""))
        matching_spec = next(
            (
                spec
                for spec in action_specs
                if spec.get("kind") == action_kind
            ),
            None,
        )
        semantic_decision = ""
        if matching_spec is not None:
            semantic_decision = str(matching_spec.get("reason", ""))
        projections.append({
            "schema_version": "consolidation_action_projection.v1",
            "action_kind": action_kind,
            "status": str(result.get("status", "")),
            "visibility": str(result.get("visibility", "private")),
            "semantic_decision": semantic_decision,
            "result_summary": str(result.get("result_summary", "")),
            "evidence_refs": list(result.get("result_refs", [])),
        })

    if not action_results:
        status: Literal["skipped", "completed", "partial", "failed"] = "skipped"
    else:
        successful = sum(
            result.get("status") in {"executed", "scheduled", "pending"}
            for result in action_results
        )
        if successful == len(action_results):
            status = "completed"
        elif successful:
            status = "partial"
        else:
            status = "failed"

    sanitized_errors = [
        _sanitize_error_code(error_code)
        for error_code in error_codes
        if _sanitize_error_code(error_code)
    ]
    record: PostTurnLifecycleRecordV1 = {
        "schema_version": "post_turn_lifecycle_record.v1",
        "lifecycle_record_id": f"post-turn:{source_episode_id}",
        "source_episode_id": source_episode_id,
        "delivery_tracking_id": delivery_tracking_id,
        "action_projections": projections,
        "status": status,
        "error_codes": sanitized_errors,
        "created_at": created_at,
        "purge_after": _audit_expiry(created_at),
    }
    return record


def _sanitize_error_code(value: object) -> str:
    """Keep error codes typed and free of backend detail."""

    if not isinstance(value, str):
        return_value = ""
        return return_value
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", value.strip())
    return sanitized[:80]


def _audit_expiry(created_at: str) -> str:
    """Return the configured audit expiry for lifecycle records."""

    from kazusa_ai_chatbot.config import AUDIT_LOG_TTL_DAYS

    return expiry_from_storage_iso(
        created_at,
        ttl_days=AUDIT_LOG_TTL_DAYS,
    ).isoformat()


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
        shallow copied state with lifecycle specs/results for the independent
        post-turn lifecycle record.
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

        current_state = _append_lifecycle_components(
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


def _append_lifecycle_components(
    state: dict,
    *,
    lifecycle_specs: list[dict[str, Any]],
    action_results: list[dict[str, Any]],
) -> dict:
    """Return lifecycle components without mutating the settled trace."""

    updated_state = dict(state)
    prior_specs = _dict_list(updated_state.get("action_specs"))
    prior_results = _dict_list(updated_state.get("action_results"))
    updated_specs = prior_specs + lifecycle_specs
    updated_results = prior_results + action_results
    updated_state["action_specs"] = updated_specs
    updated_state["action_results"] = updated_results
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

    trace = _validated_episode_trace(state, logger=logger)
    if trace is None:
        return
    settled_state = dict(state)
    settled_state["episode_trace"] = trace
    settled_state["action_specs"] = trace["action_specs"]
    settled_state["action_results"] = trace["action_results"]
    settled_state["surface_outputs"] = trace["surface_outputs"]

    try:
        result = await call_consolidation_subgraph_func(settled_state)
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

    trace = _validated_episode_trace(state, logger=logger)
    if trace is None:
        return
    final_dialog = _visible_trace_dialog(trace)
    if not final_dialog:
        logger.debug(
            "Conversation progress skipped: settled trace has no visible text"
        )
        return

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
        "final_dialog": final_dialog,
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


def _validated_episode_trace(
    state: Mapping[str, object],
    *,
    logger: logging.Logger,
) -> EpisodeTraceV2 | None:
    """Return the immutable trace required by post-turn consumers."""

    try:
        return validate_episode_trace_v2(state.get("episode_trace"))
    except (TypeError, ValueError) as exc:
        logger.warning(
            f"Post-turn consumer skipped: settled episode trace is invalid "
            f"reason={exc.__class__.__name__}"
        )
        return None


def _visible_trace_dialog(trace: EpisodeTraceV2) -> list[str]:
    """Project delivered text fragments from the settled trace only."""

    fragments: list[str] = []
    for raw_output in trace["surface_outputs"]:
        if not isinstance(raw_output, Mapping):
            continue
        if raw_output.get("surface_kind") != "text":
            continue
        if raw_output.get("visibility") != "user_visible":
            continue
        if raw_output.get("delivery_intent") != "deliver_now":
            continue
        raw_fragments = raw_output.get("fragments")
        if isinstance(raw_fragments, list):
            fragments.extend(
                fragment for fragment in raw_fragments
                if isinstance(fragment, str) and fragment
            )
    return fragments


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
