"""V3 action planning, goal resolution, and isolated authorization boundaries.

Action planning runs on a fresh boundary after the workspace collapse: the
planner payload carries only the projected scene context, evidence provenance,
and admitted bid handles, never transcript history or sibling output. The two
authorization boundaries (action and resolver) receive that same fresh minimal
context in isolation. Relational stance authority is deterministic: when the
selected ordinary owner's validated willingness declares a sensitive turn with
a non-accepting stance, every action and resolver effect is suppressed before
any authorization work runs. Future-speak rows must each carry a valid
scheduled-authority proposal; an invalid temporal or authority proposal on any
row rejects the whole planner candidate so the bounded replacement owns repair.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v2 import (
    action_authorization as v2_action_authorization,
)
from kazusa_ai_chatbot.cognition_core_v2 import (
    resolver_authorization as v2_resolver_authorization,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
    is_targetless_group_self_cognition_episode,
    project_evidence_provenance_role,
    validate_scheduled_authority_proposal,
)
from kazusa_ai_chatbot.cognition_episode import (
    build_goal_continuation_ref,
    validate_goal_continuation_ref,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    validate_resolver_goal_progress,
)
from kazusa_ai_chatbot.time_boundary import (
    local_llm_datetime_to_storage_utc_iso,
    parse_storage_utc_datetime,
)

# Exact scene-context fields projected into the fresh action-planning boundary.
SCENE_CONTEXT_REQUIRED_FIELDS = (
    "channel_scope",
    "character_role",
    "semantic_scene",
    "public_group_scene",
    "conversation_continuity",
    "semantic_temporal_context",
)

SCENE_CONTEXT_OPTIONAL_FIELDS = (
    "current_user_role",
    "character_sleep_phase",
)

RELATIONSHIP_SENSITIVE_APPLICABILITY = "relationship_sensitive"

ACCEPT_STANCE = "accept"

ANSWERABLE_NOW_GOAL_RESOLUTION = "answerable_now"

BLOCKED_GOAL_RESOLUTION = "blocked"

FUTURE_SPEAK_ACTION_KIND = "future_speak"

DECISION_TEXT_CHAR_LIMIT = 200

GOAL_PROGRESS_PROTOCOL_FIELDS = frozenset({"schema_version", "original_goal"})

GOAL_PROGRESS_CONTENT_FIELDS = (
    "current_focus",
    "deliverables",
    "missing_user_inputs",
    "evidence_dependencies",
    "attempted_paths",
    "source_backed_facts",
    "assumptions_or_inferences",
    "blockers",
    "final_response_requirements",
)

AuthorizationExecutor = Callable[..., Awaitable[dict[str, bool]]]


def project_scene_context_for_action_planning(
    scene_context: Mapping[str, Any] | None,
) -> dict[str, object]:
    """Project the validated bounded scene contract into a fresh boundary.

    The projection is the only scene carrier for both authorization
    boundaries: exactly the six required fields plus at most two optional
    fields and copied participant bindings. Keys outside this set never reach
    an authorizer.

    Args:
        scene_context: Validated bounded scene contract, or None when no
            scene context is available for this turn.

    Returns:
        The fresh minimal projection; an empty dict when the input is None.
    """
    if scene_context is None:
        return {}
    projected: dict[str, object] = {
        field_name: scene_context[field_name]
        for field_name in SCENE_CONTEXT_REQUIRED_FIELDS
    }
    for field_name in SCENE_CONTEXT_OPTIONAL_FIELDS:
        if field_name in scene_context:
            projected[field_name] = scene_context[field_name]
    if "participant_bindings" in scene_context:
        projected["participant_bindings"] = [
            dict(binding)
            for binding in scene_context["participant_bindings"]
        ]
    return projected


def project_authorizer_evidence_rows(
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    """Project admitted evidence rows into fresh authorization-boundary carriers.

    Each carrier carries only the handle, source kind, semantic text, and a
    provenance role derived exclusively from validated row metadata; unknown
    provenance fails closed in the contract layer so no free-text inference can
    assign authority to an authorizer input.

    Args:
        evidence: Admitted typed evidence rows for this turn, each carrying its
            handle, ``evidence_ref`` source kind, semantic text, and optional
            memory scope on promoted-memory rows.

    Returns:
        The projected carriers in input order.

    Raises:
        CognitionContractError: when a row's provenance cannot be mapped by the
            closed role contract.
    """
    return [
        {
            "handle": row["evidence_handle"],
            "source_kind": row["evidence_ref"]["source_kind"],
            "semantic_text": row["semantic_text"],
            "provenance_role": project_evidence_provenance_role(
                row["evidence_ref"]["source_kind"],
                row.get("memory_scope"),
            ),
        }
        for row in evidence
    ]


def apply_stance_suppression(
    decision: Mapping[str, Any],
    primary_bid: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Apply authoritative relational-stance effect suppression.

    When the selected ordinary owner's relational willingness declares a
    sensitive turn with a non-accepting stance, every action and resolver
    request is cleared before authorization work runs, goal resolution settles
    on answerable_now, pending resolution resets, and progress resets. A
    missing or non-sensitive decision passes through unchanged, so a
    non-relationship-sensitive turn never suppresses effects by itself.

    Args:
        decision: Planner-owned candidate decision carrying action requests,
            resolver requests, goal resolution, pending resolution, and goal
            progress slots.
        primary_bid: The selected complete bid whose ``relational_willingness``
            owns the stance authority for this turn.

    Returns:
        A fresh normalized decision copy plus whether suppression applied.
    """
    suppressed_decision = dict(decision)
    relational_decision = primary_bid.get("relational_willingness")
    if (
        isinstance(relational_decision, Mapping)
        and relational_decision["applicability"] == RELATIONSHIP_SENSITIVE_APPLICABILITY
        and relational_decision["stance"] != ACCEPT_STANCE
    ):
        suppressed_decision["action_requests"] = []
        suppressed_decision["resolver_requests"] = []
        suppressed_decision["goal_resolution"] = ANSWERABLE_NOW_GOAL_RESOLUTION
        suppressed_decision["resolver_pending_resolution"] = None
        suppressed_decision["resolver_goal_progress"] = None
        return suppressed_decision, True
    return suppressed_decision, False


def settle_resolver_outcome(
    decision: Mapping[str, Any],
    *,
    suppressed: bool,
    action_requests_materialized: int,
    resolver_requests_materialized: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], str]:
    """Materialize resolver effects and escalate owner-denied resolutions.

    Under suppression the resolver request list stays empty without any
    authorization work. A proposed action that has no materialized authorized
    row escalates to blocked even when the planner reported answerable_now;
    this keeps a denied effect from being reported as completed. Otherwise an
    already-settled answerable_now resolution keeps its empty resolver list.
    When the planner requested resolver work but no authorized row was
    materialized, goal resolution escalates to blocked when no resolver row
    survives; the escalated outcome resets progress so a blocked turn never
    carries stale progress forward.

    Args:
        decision: The stance-suppressed planner candidate decision.
        suppressed: Whether ``apply_stance_suppression`` cleared effects for
            this turn.
        action_requests_materialized: Count of authorized action rows that
            survived the fresh authorization boundary.
        resolver_requests_materialized: Authorized resolver rows that survived
            the fresh authorization boundary, in declared order.

    Returns:
        The materialized resolver request list plus the final goal resolution.
    """
    goal_resolution = str(decision["goal_resolution"])
    if suppressed:
        return [], goal_resolution
    action_owner_denied = (
        bool(decision["action_requests"]) and action_requests_materialized == 0
    )
    if action_owner_denied:
        if goal_resolution == ANSWERABLE_NOW_GOAL_RESOLUTION:
            return [], BLOCKED_GOAL_RESOLUTION
        return list(resolver_requests_materialized), BLOCKED_GOAL_RESOLUTION
    if goal_resolution == ANSWERABLE_NOW_GOAL_RESOLUTION:
        return [], goal_resolution
    resolver_owner_denied = (
        bool(decision["resolver_requests"])
        and len(resolver_requests_materialized) == 0
    )
    if resolver_owner_denied and not resolver_requests_materialized:
        return list(resolver_requests_materialized), BLOCKED_GOAL_RESOLUTION
    return list(resolver_requests_materialized), goal_resolution


def future_speak_proposal_contract(
    value: Mapping[str, object],
    *,
    action_kind: str,
    evidence: Sequence[Mapping[str, Any]],
    accepted_at_utc: str,
) -> dict[str, object]:
    """Validate one planner-owned future-speak authority proposal.

    The semantic temporal alignment is the action-planning owner's judgment;
    deterministic code additionally enforces that the normalized trigger is
    strictly later than the accepted time. A mismatch here raises for the whole
    planner candidate so every effect on it is denied instead of silently
    dropping one row.

    Args:
        value: One future-speak action request row carrying its authority
            proposal and bounded decision text.
        action_kind: The affordance kind the row's handle resolves to; only a
            future_speak kind may carry a scheduled-authority proposal.
        evidence: Admitted typed evidence rows the proposal references.
        accepted_at_utc: Canonical accepted storage UTC instant, empty when the
            episode carries no created time.

    Returns:
        The validated proposal copy with its provenance roles intact.

    Raises:
        ValueError: when a non-future-speak row carries a proposal, a
            future_speak row lacks one, the proposal is invalid or unaligned,
            or the trigger time does not sit strictly after the accepted time.
    """
    if action_kind != FUTURE_SPEAK_ACTION_KIND:
        if "scheduled_authority_proposal" in value:
            raise ValueError(
                "scheduled_authority_proposal is only valid for future_speak"
            )
        return {}
    if "scheduled_authority_proposal" not in value:
        raise ValueError(
            "future_speak action requires scheduled_authority_proposal"
        )
    validated_proposal = validate_scheduled_authority_proposal(
        value["scheduled_authority_proposal"],
        evidence=evidence or None,
    )
    if validated_proposal["temporal_alignment"] != "aligned":
        raise ValueError("future_speak temporal alignment is not aligned")
    decision_text: object = value["decision"]
    if (
        not isinstance(decision_text, str)
        or len(decision_text) > DECISION_TEXT_CHAR_LIMIT
    ):
        raise ValueError("action request decision is invalid")
    if accepted_at_utc:
        try:
            trigger_at_utc = local_llm_datetime_to_storage_utc_iso(
                decision_text
            )
            accepted_at_dt = parse_storage_utc_datetime(accepted_at_utc)
            trigger_at_dt = parse_storage_utc_datetime(trigger_at_utc)
        except ValueError as exc:
            raise ValueError("future_speak trigger time is invalid") from exc
        if trigger_at_dt <= accepted_at_dt:
            raise ValueError("future_speak trigger must be later than accepted time")
    return dict(validated_proposal)


def validate_future_speak_proposal_rows(
    values: Sequence[object],
    *,
    action_kind_by_handle: Mapping[str, str],
    evidence: Sequence[Mapping[str, Any]],
    accepted_at_utc: str,
) -> None:
    """Require every future-speak row to carry a valid authority proposal.

    A temporal or authority mismatch on any future-speak row invalidates the
    whole candidate so the bounded planner replacement owns the repair instead
    of silently dropping the row; that denial suppresses all effects on the
    candidate, not just the offending one.

    Args:
        values: Planner-owned action request rows for this candidate.
        action_kind_by_handle: Handle to affordance kind map for every handle
            a row may reference.
        evidence: Admitted typed evidence rows referenced by proposals.
        accepted_at_utc: Canonical accepted storage UTC instant, empty when the
            episode carries no created time.

    Raises:
        ValueError: on any proposal contract violation across the candidate.
    """
    for value in values:
        if not isinstance(value, Mapping):
            continue
        action_handle = value.get("action_handle")
        if (
            not isinstance(action_handle, str)
            or action_handle not in action_kind_by_handle
        ):
            continue
        future_speak_proposal_contract(
            value,
            action_kind=action_kind_by_handle[action_handle],
            evidence=evidence,
            accepted_at_utc=accepted_at_utc,
        )


def bind_goal_continuation_ref(
    episode: Mapping[str, Any],
    primary_bid: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the selected branch to deterministic source-goal lineage.

    Tool-result episodes carry their continuation reference in origin metadata
    and must present it validated; every other episode derives the reference
    from its episode id, platform message id when available, branch id, and
    goal ref.

    Args:
        episode: The selected turn's typed episode row.
        primary_bid: The complete bid whose branch and goal identity anchor the
            continuation lineage.

    Returns:
        The validated or built goal continuation reference.

    Raises:
        ValueError: when a tool-result origin metadata block is missing, not a
            mapping, or carries an invalid continuation reference.
    """
    origin_metadata = episode.get("origin_metadata")
    if episode.get("trigger_source") == "tool_result":
        if not isinstance(origin_metadata, Mapping):
            raise ValueError("tool-result origin metadata is invalid")
        return validate_goal_continuation_ref(
            origin_metadata["goal_continuation_ref"],
        )
    source_message_id = ""
    if isinstance(origin_metadata, Mapping):
        source_value = origin_metadata.get("platform_message_id")
        if isinstance(source_value, str):
            source_message_id = source_value
    return build_goal_continuation_ref(
        source_episode_id=episode["episode_id"],
        source_message_id=source_message_id,
        branch_id=primary_bid["branch_id"],
        goal_ref=primary_bid["goal_ref"],
    )


def build_selected_intention(
    primary_bid: Mapping[str, Any],
    route: str,
    goal_continuation_ref: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the selected branch into the exact V2 intention contract.

    The selected response operation travels as an independent copy when the
    bid carries one; a bid without it emits no key at all. The continuation
    reference is copied so later mutation cannot rewrite lineage.

    Args:
        primary_bid: The complete workspace-selected bid.
        route: The deterministic route derived for this turn.
        goal_continuation_ref: The bound continuation reference for the turn.

    Returns:
        The selected-intention mapping matching the V2 public surface exactly.
    """
    intention: dict[str, Any] = {
        "selected_branch_id": primary_bid["branch_id"],
        "route": route,
        "intention": primary_bid["intention"],
        "target_roles": list(primary_bid["target_roles"]),
        "reason": primary_bid["reason"],
        "goal_continuation_ref": dict(goal_continuation_ref),
    }
    if "selected_response_operation" in primary_bid:
        intention["selected_response_operation"] = dict(
            primary_bid["selected_response_operation"]
        )
    return intention


def _is_empty_goal_progress_shell(progress: Mapping[str, Any]) -> bool:
    """Return whether progress has no checklist content to update."""
    return not any(progress[field] for field in GOAL_PROGRESS_CONTENT_FIELDS)


def validate_goal_progress_choice(
    value: object,
    *,
    current_goal_progress: Mapping[str, Any] | None,
) -> dict | None:
    """Merge one semantic delta into protocol-owned resolver progress.

    A null choice clears no state and returns null. A non-null choice must be
    an object whose fields are a subset of the current progress content
    fields; protocol fields stay code-owned, an empty shell cannot accept an
    update, and the merged result revalidates against the full resolver
    progress contract before it is accepted.

    Args:
        value: Planner-owned goal progress delta candidate or null.
        current_goal_progress: The existing protocol-owned progress state for
            the active resolver goal, or None when no such goal exists.

    Returns:
        The validated merged progress copy, or None for a null choice.

    Raises:
        ValueError: when the choice type is wrong, required current state is
            missing, an update targets an empty shell, protocol fields are
            submitted, or any field name falls outside the content set.
    """
    if value is None:
        return_value = None
        return return_value
    if not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004
            "resolver goal progress must be an object or null"
        )
    if current_goal_progress is None:
        raise ValueError(
            "resolver goal progress requires existing current state"
        )

    current = dict(validate_resolver_goal_progress(current_goal_progress))
    if _is_empty_goal_progress_shell(current):
        raise ValueError("resolver goal progress cannot update an empty shell")
    if GOAL_PROGRESS_PROTOCOL_FIELDS.intersection(value):
        raise ValueError(
            "resolver goal progress protocol fields are code-owned"
        )
    allowed_fields = set(current) - GOAL_PROGRESS_PROTOCOL_FIELDS
    if not set(value).issubset(allowed_fields):
        raise ValueError("resolver goal progress update fields are invalid")
    raw_progress = dict(current)
    raw_progress.update(value)
    validated = validate_resolver_goal_progress(raw_progress)
    return_value = dict(validated)
    return return_value


# ---------------------------------------------------------------------------
# Fresh action-planning boundary (V3-owned planner with bounded replacement)
# and the two isolated semantic authorizer boundaries.
# ---------------------------------------------------------------------------

ACTION_AUTHORIZATION_OUTPUT_CAP = 3000
ACTION_AUTHORIZATION_TEXT_CAP = 400


def _self_cognition_target_handles(
    scene_context: Mapping[str, Any] | None,
) -> list[str]:
    """Build prompt-local target handles for group response proposals."""

    target_handles = ["self", "current_group_scene"]
    if not isinstance(scene_context, Mapping):
        return target_handles
    participant_bindings = scene_context.get("participant_bindings")
    if not isinstance(participant_bindings, list):
        return target_handles
    for binding in participant_bindings:
        if not isinstance(binding, Mapping):
            continue
        handle = binding.get("handle")
        if (
            isinstance(handle, str)
            and re.fullmatch(r"p[1-9][0-9]*", handle)
            and handle not in target_handles
        ):
            target_handles.append(handle)
    return target_handles

def _materialize_action_requests(
    requests: Sequence[Mapping[str, object]],
    bids: Mapping[str, Mapping[str, Any]],
    actions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    """Copy admitted provenance into planner-selected action requests."""

    result: list[dict[str, object]] = []
    for request in requests:
        bid = bids[request["bid_handle"]]
        affordance = actions[request["action_handle"]]
        row: dict[str, object] = {
            "action_kind": affordance["action_kind"],
            "decision": request["decision"],
            "context_ref": affordance["context_ref"],
            "semantic_goal": request["semantic_goal"],
            "reason": request["reason"],
            "target_roles": list(bid["target_roles"]),
            "evidence_handles": list(bid["evidence_handles"]),
        }
        if "scheduled_authority_proposal" in request:
            proposal = request["scheduled_authority_proposal"]
            row["scheduled_authority_proposal"] = {
                "schema_version": proposal["schema_version"],
                "temporal_alignment": proposal["temporal_alignment"],
                "authorized_content_summary": proposal[
                    "authorized_content_summary"
                ],
                "authorized_detail_refs": [
                    dict(ref) for ref in proposal["authorized_detail_refs"]
                ],
            }
        result.append(row)
    return result

def _materialize_resolver_requests(
    requests: Sequence[Mapping[str, Any]],
    bids: Mapping[str, Mapping[str, Any]],
    resolvers: Mapping[str, Mapping[str, Any]],
    goal_continuation_ref: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Copy admitted evidence provenance into resolver requests."""

    result: list[dict[str, object]] = []
    for request in requests:
        bid = bids[request["bid_handle"]]
        affordance = resolvers[request["resolver_handle"]]
        row: dict[str, object] = {
            "capability": affordance["capability"],
            "semantic_goal": request["semantic_goal"],
            "reason": request["reason"],
            "evidence_handles": list(bid["evidence_handles"]),
            "goal_continuation_ref": goal_continuation_ref,
        }
        if "start_in_background" in request:
            row["start_in_background"] = request["start_in_background"]
        result.append(row)
    return result

def derive_action_route(
    *,
    episode: Mapping[str, Any],
    primary_bid: Mapping[str, Any] | None,
    action_requests: Sequence[Mapping[str, Any]],
    resolver_requests: Sequence[Mapping[str, Any]],
    self_cognition_response: Mapping[str, Any] | None = None,
    goal_resolution: str | None = None,
    goal_continuation_ref: Mapping[str, Any] | None = None,
    required_resolver_evidence_dependency: Mapping[str, Any] | None = None,
) -> Literal["speech", "evidence", "action", "deferral", "silence"]:
    """Derive protocol route from output mode and validated request sets."""

    if action_requests and resolver_requests:
        raise CognitionExecutionError(
            "action and resolver requests are mutually exclusive"
        )
    _reject_same_goal_factual_pending_route(
        goal_resolution=goal_resolution,
        goal_continuation_ref=goal_continuation_ref,
        resolver_requests=resolver_requests,
        required_resolver_evidence_dependency=(
            required_resolver_evidence_dependency
        ),
    )
    output_mode = _episode_output_mode(episode)
    if output_mode == "silence":
        if action_requests or resolver_requests:
            raise CognitionExecutionError(
                "silence output mode cannot contain capability requests"
            )
        return "silence"
    if output_mode == "scheduled_action_request":
        if resolver_requests:
            raise CognitionExecutionError(
                "scheduled action output cannot request resolver work"
            )
        if action_requests:
            return "action"
        return "speech" if primary_bid is not None else "silence"
    if output_mode == "visible_reply":
        if resolver_requests:
            return "evidence"
        return "speech" if primary_bid is not None else "silence"
    if output_mode in {"think_only", "preview"}:
        if is_targetless_group_self_cognition_episode(episode):
            response_decision = (
                self_cognition_response.get("decision")
                if isinstance(self_cognition_response, Mapping)
                else None
            )
            if response_decision != "propose_visible_reply":
                return "silence"
            return "speech" if primary_bid is not None else "silence"
        if resolver_requests:
            return "evidence"
        if action_requests:
            return "action"
        return "silence"
    raise CognitionExecutionError("episode output mode is unsupported")

def _reject_same_goal_factual_pending_route(
    *,
    goal_resolution: str | None,
    goal_continuation_ref: Mapping[str, Any] | None,
    resolver_requests: Sequence[Mapping[str, Any]],
    required_resolver_evidence_dependency: Mapping[str, Any] | None,
) -> None:
    """Reject one continuation's factual route while its evidence is pending."""

    if (
        goal_resolution != "answerable_now"
        or not isinstance(goal_continuation_ref, Mapping)
    ):
        return
    continuation_id = goal_continuation_ref.get("continuation_id")
    if not isinstance(continuation_id, str) or not continuation_id:
        raise CognitionExecutionError(
            "factual continuation route requires a valid continuation id"
        )
    for request in resolver_requests:
        request_ref = request.get("goal_continuation_ref")
        if (
            isinstance(request_ref, Mapping)
            and request_ref.get("continuation_id") == continuation_id
        ):
            raise CognitionExecutionError(
                "factual continuation route conflicts with a pending resolver "
                "request for the same goal"
            )
    dependency = required_resolver_evidence_dependency
    if not isinstance(dependency, Mapping):
        return
    dependency_ref = dependency.get("goal_continuation_ref")
    if (
        isinstance(dependency_ref, Mapping)
        and dependency_ref.get("continuation_id") == continuation_id
        and dependency.get("state") != "complete"
    ):
        raise CognitionExecutionError(
            "factual continuation route conflicts with pending required "
            "resolver evidence for the same goal"
        )

def _episode_output_mode(episode: Mapping[str, Any]) -> str:
    """Resolve the protocol output mode from explicit or typed source data."""

    explicit_mode = episode.get("output_mode")
    if isinstance(explicit_mode, str) and explicit_mode:
        return explicit_mode
    source_modes = {
        "user_message": "visible_reply",
        "tool_result": "visible_reply",
        "scheduled_tick": "scheduled_action_request",
        "self_cognition": "think_only",
        "internal_thought": "think_only",
    }
    output_mode = source_modes.get(episode.get("trigger_source"))
    if output_mode is None:
        raise CognitionExecutionError("episode output mode is unsupported")
    return output_mode

async def authorize_action_requests(
    *,
    action_requests: Sequence[Mapping[str, str]],
    bid_handles: Mapping[str, Mapping[str, Any]],
    evidence: Sequence[Mapping[str, object]],
    action_handles: Mapping[str, Mapping[str, Any]],
    runtime_capability_limits: Sequence[str],
    authorization_executor: AuthorizationExecutor,
) -> list[dict[str, str]]:
    """Retain planner proposals whose real effects are semantically grounded."""

    if not action_requests:
        return []
    if authorization_executor is None:
        raise CognitionExecutionError(
            "action authorization requires an explicit executor"
        )

    evidence_by_handle = {
        row["evidence_handle"]: row
        for row in evidence
    }
    candidates: dict[str, dict[str, Any]] = {}
    candidate_requests: dict[str, dict[str, str]] = {}
    for index, request in enumerate(action_requests, start=1):
        bid_handle = request["bid_handle"]
        action_handle = request["action_handle"]
        if bid_handle not in bid_handles:
            raise CognitionExecutionError(
                "action authorization received an unknown bid handle"
            )
        if action_handle not in action_handles:
            raise CognitionExecutionError(
                "action authorization received an unknown action handle"
            )
        bid = bid_handles[bid_handle]
        affordance = action_handles[action_handle]
        cited_evidence = [
            evidence_by_handle[handle]["semantic_text"]
            for handle in bid["evidence_handles"]
            if handle in evidence_by_handle
        ]
        candidate_handle = f"c{index}"
        candidate_requests[candidate_handle] = dict(request)
        candidates[candidate_handle] = {
            "capability_kind": affordance["action_kind"],
            "semantic_capability": affordance["capability"],
            "proposed_decision": request["decision"],
            "proposed_semantic_goal": request["semantic_goal"],
            "proposed_reason": request["reason"],
            "admitted_bid": {
                "intention": bid["intention"],
                "desired_outcome": bid["desired_outcome"],
                "concrete_detail": bid["concrete_detail"],
                "reason": bid["reason"],
            },
            "current_evidence": cited_evidence,
        }
    prompt_payload = {
        "runtime_capability_limits": list(runtime_capability_limits),
        "candidates": candidates,
    }
    prompt_text = json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)
    if (
        len(v2_action_authorization.ACTION_AUTHORIZATION_PROMPT)
        + len(prompt_text)
        > v2_action_authorization.ACTION_AUTHORIZATION_PROMPT_CAP
    ):
        return []
    messages: list[BaseMessage] = [
        SystemMessage(
            content=v2_action_authorization.ACTION_AUTHORIZATION_PROMPT
        ),
        HumanMessage(content=prompt_text),
    ]
    decisions = await authorization_executor(
        messages=messages,
        candidate_handles=list(candidate_requests),
        stage_name="action_authorization",
        output_state_fields=["authorized_action_requests"],
        runtime_capability_limits=runtime_capability_limits,
        prompt_cap=v2_action_authorization.ACTION_AUTHORIZATION_PROMPT_CAP,
    )
    authorized_handles = {
        handle for handle, authorized in decisions.items() if authorized
    }
    return [
        candidate_requests[handle]
        for handle in candidate_requests
        if handle in authorized_handles
    ]


def _authorization_repair_message(
    *,
    response_text: str,
    contract_error: str,
    candidate_handles: list[str],
    runtime_capability_limits: Sequence[str] = (),
) -> HumanMessage:
    """Build one bounded exact-shape authorization replacement request."""

    bounded_response = response_text
    if len(bounded_response) > ACTION_AUTHORIZATION_OUTPUT_CAP:
        half_cap = ACTION_AUTHORIZATION_OUTPUT_CAP // 2
        bounded_response = (
            bounded_response[:half_cap]
            + "\n... 已截断的不合格输出 ...\n"
            + bounded_response[-half_cap:]
        )
    payload = {
        "repair_instruction": (
            "返回一个完整的替代核准对象。保留原有语义判断，为每个提供的 "
            "candidate handle 填写一个布尔值，并且只输出 JSON。"
        ),
        "candidate_handles_in_order": candidate_handles,
        "runtime_capability_limits": list(runtime_capability_limits),
        "contract_error": contract_error[:ACTION_AUTHORIZATION_TEXT_CAP],
        "invalid_response": bounded_response,
    }
    return HumanMessage(
        content=json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


async def authorize_resolver_requests(
    *,
    resolver_requests: Sequence[Mapping[str, Any]],
    bid_handles: Mapping[str, Mapping[str, Any]],
    evidence: Sequence[Mapping[str, object]],
    resolver_handles: Mapping[str, Mapping[str, Any]],
    resolver_context: str,
    authorization_executor: AuthorizationExecutor,
) -> list[dict[str, Any]]:
    """Retain proposed resolver work whose evidence need remains useful.

    Authorization judges only the unresolved evidence need and the capability
    match.  The validated routing boolean is preserved verbatim on each
    retained row; this stage never derives or reinterprets the route choice.
    """

    if not resolver_requests:
        return []
    if authorization_executor is None:
        raise CognitionExecutionError(
            "resolver authorization requires an explicit executor"
        )

    current_evidence = [
        {
            "handle": row["evidence_handle"],
            "source_kind": row["evidence_ref"]["source_kind"],
            "semantic_text": row["semantic_text"],
        }
        for row in evidence
    ]
    candidates: dict[str, dict[str, Any]] = {}
    candidate_requests: dict[str, dict[str, Any]] = {}
    for index, request in enumerate(resolver_requests, start=1):
        bid_handle = request["bid_handle"]
        resolver_handle = request["resolver_handle"]
        if bid_handle not in bid_handles:
            raise CognitionExecutionError(
                "resolver authorization received an unknown bid handle"
            )
        if resolver_handle not in resolver_handles:
            raise CognitionExecutionError(
                "resolver authorization received an unknown resolver handle"
            )
        bid = bid_handles[bid_handle]
        affordance = resolver_handles[resolver_handle]
        candidate_handle = f"c{index}"
        candidate_requests[candidate_handle] = dict(request)
        candidates[candidate_handle] = {
            "capability_kind": affordance["capability"],
            "semantic_capability": affordance["semantic_capability"],
            "proposed_semantic_goal": request["semantic_goal"],
            "proposed_reason": request["reason"],
            "owner_contract": {
                "task_resolution_request": (
                    "当前缺少的证据或有界工作由统一 task-resolution 会话处理"
                ),
                "specialist_boundary": (
                    "规划者不选择 specialist、时限、持久化或工具参数"
                ),
                "coding_lifecycle": (
                    "既有 coding run 的批准和生命周期保留给明确 action affordance"
                ),
                "approval_preparation": (
                    "对当前证据支持的持久或外部影响操作准备一个范围受限的明确批准问题；"
                    "不执行操作、不表示已经批准"
                ),
            },
            "admitted_bid": {
                "intention": bid["intention"],
                "desired_outcome": bid["desired_outcome"],
                "concrete_detail": bid["concrete_detail"],
                "reason": bid["reason"],
                "cited_evidence_handles": list(bid["evidence_handles"]),
            },
            "current_evidence": current_evidence,
            "resolver_context": resolver_context,
        }
    prompt_text = json.dumps(
        {"candidates": candidates},
        ensure_ascii=False,
        sort_keys=True,
    )
    if (
        len(v2_resolver_authorization.RESOLVER_AUTHORIZATION_PROMPT)
        + len(prompt_text)
        > v2_resolver_authorization.RESOLVER_AUTHORIZATION_PROMPT_CAP
    ):
        return []
    messages: list[BaseMessage] = [
        SystemMessage(
            content=v2_resolver_authorization.RESOLVER_AUTHORIZATION_PROMPT
        ),
        HumanMessage(content=prompt_text),
    ]
    decisions = await authorization_executor(
        messages=messages,
        candidate_handles=list(candidate_requests),
        stage_name="resolver_authorization",
        output_state_fields=["authorized_resolver_requests"],
        runtime_capability_limits=(),
        prompt_cap=v2_resolver_authorization.RESOLVER_AUTHORIZATION_PROMPT_CAP,
    )
    authorized_handles = {
        handle for handle, authorized in decisions.items() if authorized
    }
    return [
        candidate_requests[handle]
        for handle in candidate_requests
        if handle in authorized_handles
    ]
