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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import branch_order_key

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


def _bid_label(bid: Mapping[str, Any]) -> str:
    branch_id = bid.get("branch_id") if isinstance(bid, Mapping) else None
    return f"workspace bid {branch_id!r}"


def validate_complete_bids(
    bids: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Admit only complete current-matter bids into workspace competition.

    Admission is all-or-nothing: one incomplete candidate rejects the whole
    admission before any partition work. Ordinary bids must carry their
    relational-willingness decision; persistent-goal bids must carry a typed
    ``goal_ref`` whose matter context the caller provides at partition time.

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

        handle_fields = ("target_roles", "evidence_handles", "expected_consequences")
        for field_name in handle_fields:
            values = bid[field_name]
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) or not value.strip() for value in values)
            ):
                raise ValueError(f"{label} has invalid {field_name}")

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
