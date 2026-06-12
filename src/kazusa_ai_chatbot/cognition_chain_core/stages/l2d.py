"""L2d route-only action selection stage."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import Any

from kazusa_ai_chatbot.cognition_chain_core.contracts import (
    AsyncChatModel,
    require_injected_llm,
)
from kazusa_ai_chatbot.cognition_chain_core.action_selection import (
    ACTION_REQUEST_CAP,
    OPEN_GOAL_DELIVERABLE_STATUSES,
    build_action_selection_payload_text,
    route_action_requests,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_GOAL_PROGRESS_VERSION,
    RESOLVER_PENDING_RESOLUTION_VERSION,
    ResolverCapabilityRequestV1,
    ResolverGoalProgressV1,
    ResolverPendingResolutionV1,
    ResolverValidationError,
    validate_resolver_capability_request,
    validate_resolver_goal_progress,
    validate_resolver_pending_resolution,
)
from kazusa_ai_chatbot.cognition_chain_core.utils import log_preview

logger = logging.getLogger(__name__)

SPEAK_CAPABILITY = "speak"

_action_selection_llm: AsyncChatModel | None = None
_action_selection_llm_context: ContextVar[AsyncChatModel | None] = ContextVar(
    "action_selection_llm",
    default=None,
)


def set_action_selection_llm(
    llm: AsyncChatModel | None,
) -> Token[AsyncChatModel | None]:
    """Bind the action-selection model for the current run context."""

    token = _action_selection_llm_context.set(llm)
    return token


def reset_action_selection_llm(token: Token[AsyncChatModel | None]) -> None:
    """Restore the previous action-selection model binding for this context."""

    _action_selection_llm_context.reset(token)


async def select_semantic_actions(state: dict[str, Any]) -> dict[str, Any]:
    """Run L2d and return semantic route requests without materialization."""

    llm = require_injected_llm(
        _action_selection_llm_context.get() or _action_selection_llm,
        "action_selection_llm",
    )
    parsed = await route_action_requests(llm, state)
    resolver_capability_requests = _normalize_resolver_capability_requests(
        parsed,
        state,
    )
    resolver_pending_resolution = _normalize_resolver_pending_resolution(
        parsed,
        state,
    )
    resolver_goal_progress = _normalize_resolver_goal_progress(parsed)
    has_raw_action_requests = (
        isinstance(parsed, dict)
        and isinstance(parsed.get("semantic_action_requests"), list)
        and bool(parsed["semantic_action_requests"])
    )
    repeated_pending_request = False
    if (
        resolver_capability_requests
        and _resolver_requests_repeat_pending_capability(
            resolver_capability_requests,
            state,
        )
    ):
        logger.warning(
            "L2d converted repeated pending resolver request to pending "
            "surface handling"
        )
        resolver_capability_requests = []
        repeated_pending_request = True
    if resolver_capability_requests:
        semantic_action_requests: list[dict[str, object]] = []
        if isinstance(parsed, dict) and parsed.get("semantic_action_requests"):
            logger.warning(
                "L2d dropped action requests while resolver requests are pending"
            )
    else:
        if repeated_pending_request and has_raw_action_requests:
            semantic_action_requests = _normalize_pending_resume_action_requests(
                parsed,
                state,
            )
        elif repeated_pending_request:
            semantic_action_requests = _pending_resume_speak_request(state)
        else:
            semantic_action_requests = _normalize_action_requests(parsed, state)
    resolver_goal_progress = _goal_progress_with_surface_requirements(
        resolver_goal_progress,
        semantic_action_requests,
    )
    logger.debug(
        f"L2d action selection: count={len(semantic_action_requests)} "
        f"kinds={log_preview([request['capability'] for request in semantic_action_requests])} "
        f"resolver_requests={len(resolver_capability_requests)} "
    )
    return_value: dict[str, Any] = {
        "semantic_action_requests": semantic_action_requests,
        "resolver_capability_requests": resolver_capability_requests,
    }
    if resolver_pending_resolution is not None:
        return_value["resolver_pending_resolution"] = resolver_pending_resolution
    if resolver_goal_progress is not None:
        return_value["resolver_goal_progress"] = resolver_goal_progress
    return return_value


def _normalize_action_requests(
    parsed: object,
    state: Mapping[str, object],
) -> list[dict[str, object]]:
    """Normalize core semantic requests into L2d's state update shape."""

    normalized_requests: list[dict[str, object]] = []
    if not isinstance(parsed, dict):
        return normalized_requests

    raw_requests = parsed.get("semantic_action_requests")
    if not isinstance(raw_requests, list):
        return normalized_requests

    for raw_request in raw_requests:
        if not isinstance(raw_request, Mapping):
            logger.warning("L2d dropped non-object action request")
            continue
        capability = _semantic_text(raw_request, "capability")
        reason = _semantic_text(raw_request, "reason")
        if not capability or not reason:
            logger.warning("L2d dropped action request without capability or reason")
            continue
        normalized_request: dict[str, object] = {
            "capability": capability,
            "reason": reason,
        }
        for optional_field in ("decision", "detail"):
            field_value = _semantic_text(raw_request, optional_field)
            if field_value:
                normalized_request[optional_field] = field_value
        normalized_requests.append(normalized_request)
        if len(normalized_requests) >= _request_cap(state, "max_action_requests"):
            break
    return normalized_requests


def _normalize_pending_resume_action_requests(
    parsed: object,
    state: Mapping[str, object],
) -> list[dict[str, object]]:
    """Recover pending HIL/approval answers emitted with resolver capability."""

    pending_resume = state.get("pending_resolver_resume")
    if not isinstance(pending_resume, dict):
        action_requests = _normalize_action_requests(parsed, state)
        return action_requests

    pending_capability = pending_resume.get("capability_kind")
    if pending_capability not in ("human_clarification", "approval_preparation"):
        action_requests = _normalize_action_requests(parsed, state)
        return action_requests

    if not isinstance(parsed, dict):
        return_value: list[dict[str, object]] = []
        return return_value

    raw_requests = parsed.get("semantic_action_requests")
    if not isinstance(raw_requests, list):
        return_value = []
        return return_value

    rewritten_requests: list[object] = []
    for raw_request in raw_requests:
        if not isinstance(raw_request, dict):
            rewritten_requests.append(raw_request)
            continue

        capability = _semantic_text(raw_request, "capability")
        if capability != pending_capability:
            rewritten_requests.append(raw_request)
            continue

        rewritten_request = dict(raw_request)
        rewritten_request["capability"] = SPEAK_CAPABILITY
        if not _semantic_text(rewritten_request, "decision"):
            rewritten_request["decision"] = "visible_reply"
        pending_detail = _pending_resume_surface_detail(
            pending_resume,
            pending_capability,
        )
        if pending_detail:
            rewritten_request["detail"] = pending_detail
        rewritten_requests.append(rewritten_request)

    rewritten_parsed = {"semantic_action_requests": rewritten_requests}
    action_requests = _normalize_action_requests(rewritten_parsed, state)
    return action_requests


def _pending_resume_speak_request(
    state: Mapping[str, object],
) -> list[dict[str, object]]:
    """Build a text action for an active pending HIL or approval row."""

    pending_resume = state.get("pending_resolver_resume")
    if not isinstance(pending_resume, dict):
        return_value: list[dict[str, object]] = []
        return return_value

    pending_capability = pending_resume.get("capability_kind")
    if pending_capability not in ("human_clarification", "approval_preparation"):
        return_value = []
        return return_value

    pending_detail = _pending_resume_surface_detail(
        pending_resume,
        pending_capability,
    )
    if not pending_detail:
        return_value = []
        return return_value

    action_request: dict[str, object] = {
        "capability": SPEAK_CAPABILITY,
        "decision": "visible_reply",
        "detail": pending_detail,
        "reason": "处理当前等待中的澄清或审批状态。",
    }
    action_requests = [action_request]
    return action_requests


def _pending_resume_surface_detail(
    pending_resume: Mapping[str, object],
    pending_capability: object,
) -> str:
    """Build visible-surface detail for an active pending row."""

    for field_name in (
        "prompt_safe_question",
        "prompt_safe_approval_summary",
        "prompt_safe_original_goal",
    ):
        value = pending_resume.get(field_name)
        if isinstance(value, str) and value.strip():
            return_value = value.strip()
            return return_value
    if pending_capability == "human_clarification":
        return_value = "回答当前等待中的澄清问题。"
        return return_value
    if pending_capability == "approval_preparation":
        return_value = "说明当前等待中的审批结果。"
        return return_value
    return_value = ""
    return return_value


def _normalize_resolver_capability_requests(
    parsed: object,
    state: Mapping[str, object],
) -> list[ResolverCapabilityRequestV1]:
    """Validate L2d resolver requests before returning to persona graph."""

    normalized_requests: list[ResolverCapabilityRequestV1] = []
    if not isinstance(parsed, dict):
        return normalized_requests
    raw_requests = parsed.get("resolver_capability_requests")
    if not isinstance(raw_requests, list):
        return normalized_requests
    for raw_request in raw_requests:
        try:
            normalized_request = validate_resolver_capability_request(raw_request)
        except ResolverValidationError as exc:
            logger.warning(f"L2d dropped invalid resolver request: {exc}")
            continue
        normalized_requests.append(normalized_request)
        if len(normalized_requests) >= _request_cap(state, "max_resolver_requests"):
            break
    return normalized_requests


def _normalize_resolver_pending_resolution(
    parsed: object,
    state: Mapping[str, object],
) -> ResolverPendingResolutionV1 | None:
    """Validate optional pending-resolver closure selected by L2d."""

    if not isinstance(parsed, dict):
        return_value = None
        return return_value
    raw_resolution = parsed.get("resolver_pending_resolution")
    if not isinstance(raw_resolution, dict):
        return_value = None
        return return_value
    if "schema_version" not in raw_resolution:
        raw_resolution = {
            **raw_resolution,
            "schema_version": RESOLVER_PENDING_RESOLUTION_VERSION,
        }
    pending_resume = state.get("pending_resolver_resume")
    if isinstance(pending_resume, dict):
        resume_id = pending_resume.get("resume_id")
        if isinstance(resume_id, str) and resume_id.strip():
            raw_resolution = {
                **raw_resolution,
                "resume_id": resume_id.strip(),
            }
    if "capability_kind" not in raw_resolution:
        if isinstance(pending_resume, dict):
            capability_kind = pending_resume.get("capability_kind")
            if isinstance(capability_kind, str):
                raw_resolution = {
                    **raw_resolution,
                    "capability_kind": capability_kind,
                }
    try:
        normalized_resolution = validate_resolver_pending_resolution(
            raw_resolution,
        )
    except ResolverValidationError as exc:
        logger.warning(f"L2d dropped invalid pending resolver decision: {exc}")
        return_value = None
        return return_value
    return_value = normalized_resolution
    return return_value


def _normalize_resolver_goal_progress(
    parsed: object,
) -> ResolverGoalProgressV1 | None:
    """Validate optional resolver goal progress selected by L2d."""

    if not isinstance(parsed, dict):
        return_value = None
        return return_value
    raw_progress = parsed.get("resolver_goal_progress")
    if not isinstance(raw_progress, dict):
        return_value = None
        return return_value
    if "schema_version" not in raw_progress:
        raw_progress = {
            **raw_progress,
            "schema_version": RESOLVER_GOAL_PROGRESS_VERSION,
        }
    try:
        normalized_progress = validate_resolver_goal_progress(raw_progress)
    except ResolverValidationError as exc:
        logger.warning(f"L2d dropped invalid goal progress: {exc}")
        return_value = None
        return return_value
    return_value = normalized_progress
    return return_value


def _goal_progress_with_surface_requirements(
    goal_progress: ResolverGoalProgressV1 | None,
    semantic_action_requests: list[dict[str, object]],
) -> ResolverGoalProgressV1 | None:
    """Mark open deliverables as final-surface requirements when speaking."""

    if goal_progress is None:
        return_value = None
        return return_value

    has_visible_surface = any(
        request["capability"] == SPEAK_CAPABILITY
        and request.get("decision") == "visible_reply"
        for request in semantic_action_requests
    )
    if not has_visible_surface:
        return_value = goal_progress
        return return_value

    updated_progress: dict[str, object] = dict(goal_progress)
    requirements = list(updated_progress.get("final_response_requirements", []))
    deliverables = updated_progress.get("deliverables", [])
    if isinstance(deliverables, list):
        for deliverable in deliverables:
            if not isinstance(deliverable, dict):
                continue
            status = deliverable.get("status")
            description = deliverable.get("description")
            note = deliverable.get("note")
            if (
                status in OPEN_GOAL_DELIVERABLE_STATUSES
                and isinstance(description, str)
                and description.strip()
            ):
                requirement_exists = any(
                    item == description.strip()
                    or item.startswith(f"{description.strip()}：")
                    for item in requirements
                    if isinstance(item, str)
                )
                if requirement_exists:
                    continue
                if isinstance(note, str) and note.strip():
                    requirement = f"{description.strip()}：{note.strip()}"
                else:
                    requirement = description.strip()
                requirements.append(requirement)
    updated_progress["final_response_requirements"] = requirements
    normalized_progress = validate_resolver_goal_progress(updated_progress)
    return_value = normalized_progress
    return return_value


def _resolver_requests_repeat_pending_capability(
    requests: list[ResolverCapabilityRequestV1],
    state: Mapping[str, object],
) -> bool:
    """Return whether every resolver request repeats the active pending kind."""

    if not requests:
        return_value = False
        return return_value

    pending_resume = state.get("pending_resolver_resume")
    if not isinstance(pending_resume, dict):
        return_value = False
        return return_value

    pending_capability = pending_resume.get("capability_kind")
    if pending_capability not in ("human_clarification", "approval_preparation"):
        return_value = False
        return return_value

    repeats_pending = all(
        request["capability_kind"] == pending_capability
        for request in requests
    )
    return_value = repeats_pending
    return return_value


def _semantic_text(value: Mapping[str, object], field_name: str) -> str:
    """Return one stripped semantic text field from LLM output."""

    raw_value = value.get(field_name)
    if not isinstance(raw_value, str):
        return_value = ""
        return return_value
    return_value = raw_value.strip()
    return return_value


def _request_cap(state: Mapping[str, object], field_name: str) -> int:
    """Return a positive per-input request cap."""

    raw_cap = state.get(field_name)
    if isinstance(raw_cap, int) and raw_cap > 0:
        return_value = raw_cap
        return return_value
    return_value = ACTION_REQUEST_CAP
    return return_value
