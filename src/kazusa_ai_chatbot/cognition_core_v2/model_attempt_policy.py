"""Bounded retry and terminal-disposition metadata for V2 model owners."""

from __future__ import annotations

from typing import Literal, Mapping, TypedDict, cast


V2_MODEL_TOTAL_ATTEMPTS = 3
V2_VERIFIER_TOTAL_ATTEMPTS = 3

V2AttemptFailureKind = Literal[
    "provider",
    "parse",
    "structure",
    "semantic",
    "verifier_unavailable",
]
V2AttemptDisposition = Literal[
    "accepted",
    "recovered",
    "accepted_degraded",
    "retry_graph",
    "empty",
    "denied",
    "skipped",
    "unrecoverable",
]
V2SafeCheckpoint = Literal[
    "pre_state_commit",
    "post_cognition_commit",
]


class V2AttemptRecord(TypedDict):
    """Bounded diagnostic metadata for one V2 model owner."""

    stage: str
    failure_kind: V2AttemptFailureKind | None
    attempt_count: int
    total_attempt_limit: int
    selected_attempt: int | None
    disposition: V2AttemptDisposition
    safe_checkpoint: V2SafeCheckpoint


class V2ModelOwnerPolicy(TypedDict):
    """Attempt limit and exhausted disposition for one V2 model owner."""

    total_attempt_limit: int
    exhausted_disposition: V2AttemptDisposition


V2_MODEL_OWNER_POLICIES: dict[str, V2ModelOwnerPolicy] = {
    "image_descriptor": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "skipped",
    },
    "message_decontextualizer": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "accepted_degraded",
    },
    "semantic_appraisal": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "skipped",
    },
    "goal_bid_structure": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "retry_graph",
    },
    "workspace_collapse": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "accepted_degraded",
    },
    "action_planning": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "empty",
    },
    "action_authorization": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "denied",
    },
    "resolver_authorization": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "denied",
    },
    "surface_content_plan": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "accepted_degraded",
    },
    "surface_preference": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "accepted_degraded",
    },
    "surface_dialog_compliance_repair": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "accepted_degraded",
    },
    "surface_visual": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "skipped",
    },
    "dialog_generator": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "accepted_degraded",
    },
    "dialog_semantic_fidelity_verifier": {
        "total_attempt_limit": V2_VERIFIER_TOTAL_ATTEMPTS,
        "exhausted_disposition": "accepted_degraded",
    },
    "dialog_role_direction_verifier": {
        "total_attempt_limit": V2_VERIFIER_TOTAL_ATTEMPTS,
        "exhausted_disposition": "accepted_degraded",
    },
    "dialog_surface_integrity_verifier": {
        "total_attempt_limit": V2_VERIFIER_TOTAL_ATTEMPTS,
        "exhausted_disposition": "accepted_degraded",
    },
}

_ATTEMPT_RECORD_FIELDS = frozenset({
    "stage",
    "failure_kind",
    "attempt_count",
    "total_attempt_limit",
    "selected_attempt",
    "disposition",
    "safe_checkpoint",
})
_FAILURE_KINDS = frozenset({
    "provider",
    "parse",
    "structure",
    "semantic",
    "verifier_unavailable",
})
_DISPOSITIONS = frozenset({
    "accepted",
    "recovered",
    "accepted_degraded",
    "retry_graph",
    "empty",
    "denied",
    "skipped",
    "unrecoverable",
})
_SAFE_CHECKPOINTS = frozenset({
    "pre_state_commit",
    "post_cognition_commit",
})


def validate_v2_attempt_record(
    value: Mapping[str, object],
) -> V2AttemptRecord:
    """Validate bounded diagnostic metadata without accepting model content.

    Args:
        value: Candidate metadata produced by a V2 model-owning stage.

    Returns:
        A copied record containing only the approved diagnostic fields.
    """

    if set(value) != _ATTEMPT_RECORD_FIELDS:
        raise ValueError("V2 attempt record fields are not exact")

    stage = value["stage"]
    if (
        not isinstance(stage, str)
        or stage not in V2_MODEL_OWNER_POLICIES
    ):
        raise ValueError("V2 attempt record stage is invalid")

    total_attempt_limit = value["total_attempt_limit"]
    owner_limit = V2_MODEL_OWNER_POLICIES[stage]["total_attempt_limit"]
    if (
        isinstance(total_attempt_limit, bool)
        or not isinstance(total_attempt_limit, int)
        or total_attempt_limit != owner_limit
    ):
        raise ValueError("V2 attempt record total limit is invalid")

    attempt_count = value["attempt_count"]
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or not 1 <= attempt_count <= total_attempt_limit
    ):
        raise ValueError("V2 attempt record attempt count is invalid")

    selected_attempt = value["selected_attempt"]
    if selected_attempt is not None and (
        isinstance(selected_attempt, bool)
        or not isinstance(selected_attempt, int)
        or not 1 <= selected_attempt <= attempt_count
    ):
        raise ValueError("V2 attempt record selected attempt is invalid")

    failure_kind = value["failure_kind"]
    if failure_kind is not None and failure_kind not in _FAILURE_KINDS:
        raise ValueError("V2 attempt record failure kind is invalid")

    disposition = value["disposition"]
    if disposition not in _DISPOSITIONS:
        raise ValueError("V2 attempt record disposition is invalid")

    safe_checkpoint = value["safe_checkpoint"]
    if safe_checkpoint not in _SAFE_CHECKPOINTS:
        raise ValueError("V2 attempt record safe checkpoint is invalid")

    validated = cast(V2AttemptRecord, dict(value))
    return validated
