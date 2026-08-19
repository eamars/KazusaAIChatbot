"""V3 complete-bid workspace collapse and authoritative relational disposition.

The workspace boundary runs after the complete-bid join, on a fresh canonical
projection: only admitted complete bids compete, the ordinary bid is the
current-turn baseline, persistent-goal matter provenance travels with each
persistent bid, and no confidence descriptor ever ranks, thresholds, or gates
a partition. The authoritative relational collapse is fully deterministic: when
the ordinary goal owner declared the turn relationship-sensitive with a
validated decision, that ordinary bid becomes primary without any semantic
reinterpretation of user text, relationship axes, memory, or bid prose. If a
model-authored partition never succeeds within its attempt budget, the
deterministic fallback keeps the registry-ordered first bid primary and
suppresses every other admitted bid.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import branch_order_key
from kazusa_ai_chatbot.cognition_core_v2.contracts import ROLE_ENTITY_KINDS, ROLE_VALUES
from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
    V2_MODEL_TOTAL_ATTEMPTS,
)
from kazusa_ai_chatbot.cognition_core_v2.prompt_budget import (
    PromptBudgetError,
    middle_truncate_text,
)
from kazusa_ai_chatbot.llm_tracing import failure_capsule
from kazusa_ai_chatbot.utils import parse_llm_json_output

# Complete-bid admission: the exact V2 ``ActionBidV2`` core field set plus its
# two optional fields. A candidate missing any required field, carrying an
# unknown field, or violating a field type is never admitted into competition.
COMPLETE_BID_REQUIRED_FIELDS = (
    "branch_id",
    "goal_ref",
    "intention",
    "desired_outcome",
    "concrete_detail",
    "reason",
    "private_monologue",
    "target_roles",
    "evidence_handles",
    "expected_consequences",
    "confidence",
)

COMPLETE_BID_OPTIONAL_FIELDS = frozenset({
    "selected_response_operation",
    "relational_willingness",
})

BID_NARRATIVE_FIELDS = (
    "intention",
    "desired_outcome",
    "concrete_detail",
    "reason",
    "private_monologue",
)

PARTITION_FIELDS = (
    "primary_bid_handle",
    "supporting_bid_handles",
    "suppressed_bid_handles",
)

# Byte-identical collapse instruction text owned by the V3 boundary.
COLLAPSE_PROMPT = '''把完整的目标候选划分为本轮主目标、支持目标和抑制目标。先依据 current_event
判断每个候选及其 persistent_goal 是否与本轮直接相关，再依据相关性分区。ordinary_response 是当前
回应的基线候选。非 ordinary_response 的持久目标只有在当前事件直接推进、阻碍、威胁或要求处理
同一具体事项时才可成为主目标或支持目标。仅有目标仍在进行中、同一用户、一般关系互动、关系
appraisal 存在、角色重视某项驱动，或分支 action tendency，不能证明当前相关。若
persistent_goal 与 current_event 是不同事项，必须抑制该候选，即使候选文本把当前请求改写成该
旧目标的边界问题。本阶段不判断工具、resolver、worker 或运行时能力，也不改写候选。

只返回包含 primary_bid_handle、supporting_bid_handles 和 suppressed_bid_handles 的 JSON。
保持候选内容原样，不复制内容，也不增添细节。每个提供的 bid handle 必须在三个分区中恰好出现
一次。
'''

WORKSPACE_COLLAPSE_ATTEMPT_LIMIT = V2_MODEL_TOTAL_ATTEMPTS
WORKSPACE_COLLAPSE_PROMPT_CAP = 24000
WORKSPACE_COLLAPSE_REPAIR_OUTPUT_CAP = 4000
WORKSPACE_CONTEXT_TEXT_FLOOR = 256

_PROVIDER_EXCEPTIONS = (
    OpenAIError,
    httpx.HTTPError,
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
)


def _bid_label(bid: Mapping[str, Any]) -> str:
    branch_id = bid.get("branch_id") if isinstance(bid, Mapping) else None
    return f"workspace bid {branch_id!r}"


def _validate_bid_target_roles(value: object, label: str) -> None:
    """Validate the structured role references carried by one complete bid.

    Each entry mirrors the V2 ``RoleRefV2`` shape produced by goal-bid
    materialization: an exact key set with closed enum values and a non-empty
    entity id, so admitted bids pass the V2 action-bid contract unchanged.
    """
    if not isinstance(value, list):
        raise ValueError(f"{label} has invalid target_roles")
    for index, role in enumerate(value):
        if (
            not isinstance(role, Mapping)
            or set(role) != {"role", "entity_kind", "entity_id"}
        ):
            raise ValueError(f"{label} has an invalid target_role entry at {index}")
        if role["role"] not in ROLE_VALUES:
            raise ValueError(f"{label} has an invalid target_role value at {index}")
        if role["entity_kind"] not in ROLE_ENTITY_KINDS:
            raise ValueError(
                f"{label} has an invalid target_role entity kind at {index}"
            )
        if (
            not isinstance(role["entity_id"], str)
            or not role["entity_id"].strip()
        ):
            raise ValueError(f"{label} has an empty target_role entity id at {index}")


def validate_complete_bids(
    bids: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Admit only complete current-matter bids into workspace competition.

    Admission is all-or-nothing: one incomplete candidate rejects the whole
    admission before any partition work. Ordinary bids must carry their
    relational-willingness decision; persistent-goal bids must carry a typed
    ``goal_ref`` whose matter context the caller provides at partition time.
    Target-role entries mirror the V2 ``RoleRefV2`` shape carried by goal-bid
    materialization, so admitted bids stay valid under the V2 action-bid
    contract without translation.

    Args:
        bids: Candidate branch-owned bid mappings for this turn.

    Returns:
        Normalized independent copies of all admitted bids, preserving input
        order.

    Raises:
        ValueError: when the sequence is empty or any candidate fails the
            complete-bid contract.
    """
    if not isinstance(bids, Sequence) or isinstance(bids, (str, bytes)) or not bids:
        raise ValueError("workspace collapse requires at least one bid")

    admitted = []
    for bid in bids:
        label = _bid_label(bid)
        if not isinstance(bid, Mapping):
            raise ValueError(f"{label} is not a mapping")
        allowed_fields = set(COMPLETE_BID_REQUIRED_FIELDS) | COMPLETE_BID_OPTIONAL_FIELDS
        missing_fields = [field for field in COMPLETE_BID_REQUIRED_FIELDS if field not in bid]
        unknown_fields = sorted(set(bid) - allowed_fields)
        if missing_fields or unknown_fields:
            raise ValueError(f"{label} is not complete: {missing_fields + unknown_fields}")

        if (
            not isinstance(bid["branch_id"], str)
            or not bid["branch_id"].strip()
        ):
            raise ValueError(f"{label} has an invalid branch id")

        goal_ref = bid["goal_ref"]
        if (
            not isinstance(goal_ref, Mapping)
            or not isinstance(goal_ref.get("entity_id"), str)
            or not goal_ref["entity_id"].strip()
        ):
            raise ValueError(f"{label} has an invalid goal ref")

        for field_name in BID_NARRATIVE_FIELDS:
            if not isinstance(bid[field_name], str) or not bid[field_name].strip():
                raise ValueError(f"{label} has an invalid {field_name}")

        string_handle_fields = ("evidence_handles", "expected_consequences")
        for field_name in string_handle_fields:
            values = bid[field_name]
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) or not value.strip() for value in values)
            ):
                raise ValueError(f"{label} has invalid {field_name}")

        _validate_bid_target_roles(bid["target_roles"], label)

        if not isinstance(bid["confidence"], str) or not bid["confidence"].strip():
            raise ValueError(f"{label} has an invalid confidence")

        if (
            bid["branch_id"] == "ordinary_response"
            and not isinstance(bid.get("relational_willingness"), Mapping)
        ):
            raise ValueError(f"{label} lacks the relational willingness decision")

        admitted.append(dict(bid))
    return admitted


def collapse_single_bid(bid: Mapping[str, Any]) -> dict[str, Any]:
    """Collapse one admitted bid into its primary-only envelope.

    Args:
        bid: One complete branch-owned candidate.

    Returns:
        The collapsed envelope with the sole bid primary and empty supporting,
        suppressed, and competing partitions.

    Raises:
        ValueError: when the single candidate fails the complete-bid contract.
    """
    normalized = validate_complete_bids([bid])
    only_bid = normalized[0]
    return {
        "primary_branch_id": only_bid["branch_id"],
        "supporting_branch_ids": [],
        "suppressed_branch_ids": [],
        "primary_bid": only_bid,
        "supporting_bids": [],
        "competing_bids": [],
    }


@dataclass(frozen=True)
class PartitionRequest:
    """Prompt-local partition inputs for the workspace collapse stage.

    Attributes:
        handles: Bid handle to bid mapping assigned in registry order, one
            ``bN`` handle per admitted bid starting at 1.
        prompt_payload: Deterministic payload carrying the current-event rows
            and each bid's branch id, persistent-goal matter provenance (None
            for the ordinary baseline), intention, desired outcome, and reason.
    """

    handles: Mapping[str, Mapping[str, Any]]
    prompt_payload: dict[str, Any]


def prepare_partition(
    bids: Sequence[Mapping[str, Any]],
    current_event: Sequence[Mapping[str, object]],
    goal_context_by_ref: Mapping[str, Mapping[str, object]],
) -> PartitionRequest:
    """Build the deterministic partition request for admitted complete bids.

    Bids are ordered by the frozen V2 branch registry position before handle
    assignment, so ``b1``..``bN`` is stable across runs and engines. The
    ordinary baseline carries no persistent goal; every other bid looks up its
    matter context through ``goal_context_by_ref`` with a fail-fast plain
    index into the caller-guaranteed entity id.

    Args:
        bids: Admitted complete branch-owned candidate mappings.
        current_event: Typed current-episode evidence rows projected for relevance.
        goal_context_by_ref: Bounded persistent-goal matter contexts keyed by
            the bid's ``goal_ref`` entity id.

    Returns:
        The frozen partition request with stable handle assignment and payload.

    Raises:
        ValueError: when any candidate fails the complete-bid contract.
        KeyError: when a non-ordinary bid's goal ref has no matter context row,
            which is an admission-order programming error.
    """
    ordered = sorted(bids, key=lambda bid: branch_order_key(bid["branch_id"]))
    handles = {f"b{index}": bid for index, bid in enumerate(ordered, start=1)}
    prompt_payload = {
        "current_event": [dict(row) for row in current_event],
        "bids": {
            handle: {
                "branch_id": bid["branch_id"],
                "persistent_goal": (
                    None
                    if bid["branch_id"] == "ordinary_response"
                    else dict(goal_context_by_ref[bid["goal_ref"]["entity_id"]])
                ),
                "intention": bid["intention"],
                "desired_outcome": bid["desired_outcome"],
                "reason": bid["reason"],
            }
            for handle, bid in handles.items()
        },
    }
    return PartitionRequest(handles=handles, prompt_payload=prompt_payload)


def validate_partition(
    parsed: object,
    handles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate exact handle partition output from workspace collapse.

    Args:
        parsed: Candidate partition mapping from the collapse stage.
        handles: The complete ``bN`` handle set assigned at request time; every
            handle must appear in exactly one of the three partitions.

    Returns:
        The validated partition mapping, unchanged.

    Raises:
        ValueError: when fields are not exact, a handle is unknown or
            duplicated, or the three partitions do not cover ``handles`` once.
    """
    handle_set = set(handles)
    if not isinstance(parsed, Mapping):
        raise ValueError("workspace partition must be an object")
    required = {
        "primary_bid_handle",
        "supporting_bid_handles",
        "suppressed_bid_handles",
    }
    if set(parsed) != required:
        raise ValueError("workspace partition fields are not exact")
    primary = parsed["primary_bid_handle"]
    if primary not in handle_set:
        raise ValueError("workspace primary handle is unavailable")
    partitions = []
    for field_name in ("supporting_bid_handles", "suppressed_bid_handles"):
        values = parsed[field_name]
        if not isinstance(values, list) or any(
            value not in handle_set for value in values
        ):
            raise ValueError("workspace partition handle is unavailable")
        if len(values) != len(set(values)):
            raise ValueError("workspace partition contains duplicate handles")
        partitions.extend(values)
    all_handles = [primary] + partitions
    if len(all_handles) != len(handle_set) or set(all_handles) != handle_set:
        raise ValueError("workspace partition is incomplete")
    if len(all_handles) != len(set(all_handles)):
        raise ValueError("workspace partition overlaps")
    return dict(parsed)


def materialize_partition(
    handles: Mapping[str, Mapping[str, Any]],
    partition: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize a validated handle partition into the collapsed envelope.

    Args:
        handles: The ``bN`` handle to bid mapping from the partition request.
        partition: A partition already validated by ``validate_partition``.

    Returns:
        The collapsed envelope with supporting and suppressed partitions in
        declared order and competing bids equal to the suppressed list,
        matching the V2 surface contract exactly.
    """
    primary = handles[partition["primary_bid_handle"]]
    declared_supporting = [
        handles[handle] for handle in partition["supporting_bid_handles"]
    ]
    suppressed = [
        handles[handle] for handle in partition["suppressed_bid_handles"]
    ]
    return {
        "primary_branch_id": primary["branch_id"],
        "supporting_branch_ids": [bid["branch_id"] for bid in declared_supporting],
        "suppressed_branch_ids": [bid["branch_id"] for bid in suppressed],
        "primary_bid": primary,
        "supporting_bids": declared_supporting,
        "competing_bids": suppressed,
    }


def fallback_partition_envelope(
    ordered_bids: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the fail-closed partition envelope when no partition succeeds.

    The registry-ordered first admitted bid stays primary and every other
    admitted bid is suppressed; supporting partitions stay empty and no
    confidence descriptor influences this fallback.

    Args:
        ordered_bids: Admitted complete bids in registry order (at least one).

    Returns:
        The fail-closed collapsed envelope.

    Raises:
        ValueError: when the sequence carries no admitted bid.
    """
    if not ordered_bids:
        raise ValueError("workspace collapse requires at least one bid")
    primary = ordered_bids[0]
    suppressed = list(ordered_bids[1:])
    return {
        "primary_branch_id": primary["branch_id"],
        "supporting_branch_ids": [],
        "suppressed_branch_ids": [bid["branch_id"] for bid in suppressed],
        "primary_bid": dict(primary),
        "supporting_bids": [],
        "competing_bids": [dict(bid) for bid in suppressed],
    }


def collapse_authoritative_relational_bid(
    bids: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve the ordinary relational owner without semantic reinterpretation.

    This deterministic collapse runs only when the ordinary goal owner declared
    the turn relationship-sensitive. It makes that ordinary bid primary,
    exposes no supporting bid, and places every other bid in ``competing_bids``.
    It never reads user text, relationship axes, memory, or bid prose, so no
    confidence descriptor can rank it out of position.

    Args:
        bids: Complete branch-owned candidates eligible for partition.
        decision: Validated ordinary character-owned relational stance whose
            applicability is ``relationship_sensitive``.

    Returns:
        The authoritative collapse envelope with the ordinary bid primary and
        every non-ordinary bid suppressed and competing in input order.

    Raises:
        ValueError: when no relationship-sensitive decision is supplied,
            exactly one ordinary bid carrying the equal decision is missing, or
            a competing ordinary bid is present.
    """
    if (
        not isinstance(decision, Mapping)
        or decision["applicability"] != "relationship_sensitive"
    ):
        raise ValueError(
            "authoritative relational collapse requires a sensitive decision"
        )
    ordinary_bids = [
        bid for bid in bids if bid["branch_id"] == "ordinary_response"
    ]
    if len(ordinary_bids) != 1:
        raise ValueError(
            "authoritative relational collapse requires exactly one ordinary bid"
        )
    ordinary_bid = ordinary_bids[0]
    if ordinary_bid.get("relational_willingness") != dict(decision):
        raise ValueError(
            "authoritative relational collapse requires the equal decision"
        )
    competing_bids = [
        bid for bid in bids if bid["branch_id"] != "ordinary_response"
    ]
    return {
        "primary_branch_id": ordinary_bid["branch_id"],
        "supporting_branch_ids": [],
        "suppressed_branch_ids": [bid["branch_id"] for bid in competing_bids],
        "primary_bid": dict(ordinary_bid),
        "supporting_bids": [],
        "competing_bids": [dict(bid) for bid in competing_bids],
    }


def _fit_workspace_prompt_payload(
    prompt_payload: dict[str, Any],
    *,
    maximum_chars: int,
) -> str:
    """Fit relevance text while retaining every bid and provenance handle."""

    if maximum_chars <= 0:
        raise PromptBudgetError(
            "workspace system prompt exhausts the aggregate character cap"
        )
    serialized = json.dumps(
        prompt_payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    if len(serialized) <= maximum_chars:
        return serialized

    text_slots: list[tuple[dict[str, Any], str, str]] = []
    for event in prompt_payload["current_event"]:
        semantic_text = event["semantic_text"]
        if not isinstance(semantic_text, str):
            raise TypeError("workspace current-event text must be a string")
        text_slots.append((event, "semantic_text", semantic_text))
    for bid_handle in sorted(prompt_payload["bids"]):
        persistent_goal = prompt_payload["bids"][bid_handle][
            "persistent_goal"
        ]
        if persistent_goal is None:
            continue
        description = persistent_goal["description"]
        if not isinstance(description, str):
            raise TypeError("workspace persistent-goal description must be a string")
        text_slots.append((persistent_goal, "description", description))

    for owner, field_name, original in text_slots:
        owner[field_name] = middle_truncate_text(
            original,
            WORKSPACE_CONTEXT_TEXT_FLOOR,
        )
    floor_serialized = json.dumps(
        prompt_payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    if len(floor_serialized) > maximum_chars:
        raise PromptBudgetError(
            "workspace required structure exceeds the aggregate character cap"
        )

    for owner, field_name, original in text_slots:
        if len(original) <= WORKSPACE_CONTEXT_TEXT_FLOOR:
            continue
        lower_bound = WORKSPACE_CONTEXT_TEXT_FLOOR
        upper_bound = len(original)
        retained_chars = WORKSPACE_CONTEXT_TEXT_FLOOR
        while lower_bound <= upper_bound:
            candidate_chars = (lower_bound + upper_bound) // 2
            owner[field_name] = middle_truncate_text(
                original,
                candidate_chars,
            )
            candidate = json.dumps(
                prompt_payload,
                ensure_ascii=False,
                sort_keys=True,
            )
            if len(candidate) <= maximum_chars:
                retained_chars = candidate_chars
                lower_bound = candidate_chars + 1
            else:
                upper_bound = candidate_chars - 1
        owner[field_name] = middle_truncate_text(original, retained_chars)

    return json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)


def _record_workspace_trace(
    *,
    services: Any,
    messages: Sequence[BaseMessage],
    response_text: str,
    parsed_output: object,
    parse_status: str,
    status: str,
    started_at: float,
    stage_name: str,
    attempt_index: int,
    validation_error: str,
) -> None:
    """Preserve one protected workspace-collapse model boundary."""

    config = services.workspace_collapse_config
    failure_capsule.append_model_attempt(
        stage_name=stage_name,
        messages=messages,
        response_text=response_text,
        parsed_output=parsed_output,
        parse_status=parse_status,
        status=status,
        config=config,
        attempt_index=attempt_index,
        validation_error=validation_error,
        started_at=started_at,
    )


async def collapse_bids_via_partition(
    bids: Sequence[Mapping[str, Any]],
    services: Any,
    *,
    current_event: Sequence[Mapping[str, object]],
    goal_context_by_ref: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Partition complete admitted bids on a fresh collapse boundary.

    One admitted bid collapses deterministically to its primary-only envelope.
    Multiple admitted bids run the byte-identical static collapse prompt over
    the partition request built by ``prepare_partition``; every attempt parses
    through the canonical entry point and validates against the exact handle
    set, with one bounded complete replacement per failed attempt under the
    owner's attempt cap. Provider failures retry the same request. When no
    model-authored partition succeeds within the budget, the deterministic
    fallback keeps the registry-ordered first bid primary and suppresses every
    other admitted bid.

    Args:
        bids: Admitted complete branch-owned candidate mappings.
        services: Injected LLM binding and workspace-collapse route config.
        current_event: Typed current episode evidence rows projected for
            relevance.
        goal_context_by_ref: Bounded persistent-goal matter contexts keyed by
            the bid's ``goal_ref`` entity id.

    Returns:
        The collapsed V2-shaped envelope with primary, supporting, and
        competing partitions in declared order.

    Raises:
        ValueError: when no admitted bid is supplied or a partition candidate
            fails the exact handle contract on every attempt without the
            deterministic fallback being reachable (no bids).
        KeyError: when a non-ordinary bid's goal ref has no matter context row,
            which is an admission-order programming error.
    """

    ordered = sorted(bids, key=lambda bid: branch_order_key(bid["branch_id"]))
    if not ordered:
        raise ValueError("workspace collapse requires at least one bid")
    if len(ordered) == 1:
        return collapse_single_bid(ordered[0])
    request = prepare_partition(
        ordered,
        current_event,
        goal_context_by_ref,
    )
    handles = request.handles
    payload_cap = WORKSPACE_COLLAPSE_PROMPT_CAP - len(COLLAPSE_PROMPT)
    available_attempts = WORKSPACE_COLLAPSE_ATTEMPT_LIMIT
    try:
        prompt_text = _fit_workspace_prompt_payload(
            dict(request.prompt_payload),
            maximum_chars=payload_cap,
        )
    except PromptBudgetError:
        prompt_text = ""
        available_attempts = 0
    request_messages: list[BaseMessage] = [
        SystemMessage(content=COLLAPSE_PROMPT),
        HumanMessage(content=prompt_text),
    ]
    partition: dict[str, Any] | None = None
    for attempt_index in range(available_attempts):
        started_at = perf_counter()
        stage_name = (
            "workspace_collapse"
            if attempt_index == 0
            else "workspace_collapse.repair"
        )
        try:
            response = await services.llm.ainvoke(
                request_messages,
                config=services.workspace_collapse_config,
            )
        except _PROVIDER_EXCEPTIONS as exc:
            _record_workspace_trace(
                services=services,
                messages=request_messages,
                response_text="",
                parsed_output={},
                parse_status="provider_error",
                status="failed",
                started_at=started_at,
                stage_name=stage_name,
                attempt_index=attempt_index + 1,
                validation_error=str(exc),
            )
            if attempt_index + 1 >= WORKSPACE_COLLAPSE_ATTEMPT_LIMIT:
                break
            request_messages = [
                SystemMessage(content=COLLAPSE_PROMPT),
                HumanMessage(content=prompt_text),
            ]
            continue
        response_text = str(getattr(response, "content", ""))
        parsed: object = {}
        try:
            parsed = parse_llm_json_output(
                response_text,
                repair_trace_hook=(
                    failure_capsule.append_json_repair_attempt
                ),
            )
            partition = validate_partition(parsed, handles)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            _record_workspace_trace(
                services=services,
                messages=request_messages,
                response_text=response_text,
                parsed_output=parsed,
                parse_status="contract_error",
                status="failed",
                started_at=started_at,
                stage_name=stage_name,
                attempt_index=attempt_index + 1,
                validation_error=str(exc),
            )
            if attempt_index + 1 >= WORKSPACE_COLLAPSE_ATTEMPT_LIMIT:
                break
            invalid_candidate = response_text
            if len(invalid_candidate) > WORKSPACE_COLLAPSE_REPAIR_OUTPUT_CAP:
                half_cap = WORKSPACE_COLLAPSE_REPAIR_OUTPUT_CAP // 2
                invalid_candidate = (
                    invalid_candidate[:half_cap]
                    + "\n... 已截断的不合格候选 ...\n"
                    + invalid_candidate[-half_cap:]
                )
            repair_payload = {
                **request.prompt_payload,
                "contract_repair": {
                    "reason": str(exc)[:500],
                    "invalid_candidate": invalid_candidate,
                },
            }
            try:
                repair_text = _fit_workspace_prompt_payload(
                    dict(repair_payload),
                    maximum_chars=payload_cap,
                )
            except PromptBudgetError:
                break
            request_messages = [
                SystemMessage(content=COLLAPSE_PROMPT),
                HumanMessage(content=repair_text),
            ]
            continue
        _record_workspace_trace(
            services=services,
            messages=request_messages,
            response_text=response_text,
            parsed_output=parsed,
            parse_status="succeeded",
            status="succeeded",
            started_at=started_at,
            stage_name=stage_name,
            attempt_index=attempt_index + 1,
            validation_error="",
        )
        break

    if partition is None:
        return fallback_partition_envelope(ordered)
    return materialize_partition(handles, partition)
