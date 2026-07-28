"""MongoDB owner for immutable character identity growth state."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import (
    DuplicateKeyError,
    OperationFailure,
    PyMongoError,
)

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.identity import (
    apply_identity_patches,
    candidate_transition_allowed,
    dedupe_evidence_refs,
    diff_effective_identities,
    derive_growth_health_state,
    evidence_counts,
)
from kazusa_ai_chatbot.character_identity_growth.validation import (
    validate_effective_identity,
    validate_identity_patch,
)
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.time_boundary import (
    local_date_bounds_to_storage_utc_iso,
    parse_storage_utc_datetime,
    storage_utc_now,
)


REVISIONS_COLLECTION = "character_identity_revisions"
CANDIDATES_COLLECTION = "character_identity_growth_candidates"
RUNS_COLLECTION = "character_identity_growth_runs"
GROWTH_COLLECTION_NAMES = (
    REVISIONS_COLLECTION,
    CANDIDATES_COLLECTION,
    RUNS_COLLECTION,
)

REVISION_SCHEMA_VERSION = "character_identity_revision.v1"
CANDIDATE_SCHEMA_VERSION = "character_identity_growth_candidate.v1"
RUN_SCHEMA_VERSION = "character_identity_growth_run.v1"

REVISION_ID_INDEX = "character_identity_revision_id_unique"
REVISION_NUMBER_INDEX = "character_identity_character_revision_unique"
REVISION_DESC_INDEX = "character_identity_character_revision_desc"
CANDIDATE_ID_INDEX = "character_identity_candidate_id_unique"
CANDIDATE_STATUS_INDEX = (
    "character_identity_candidate_character_status_updated"
)
CANDIDATE_BASE_INDEX = "character_identity_candidate_base_status"
CANDIDATE_ROOT_INDEX = "character_identity_candidate_character_root_unique"
RUN_ID_INDEX = "character_identity_run_id_unique"
RUN_COMPLETED_INDEX = "character_identity_run_kind_completed"
RUN_REVISION_INDEX = "character_identity_run_revision"
IDENTITY_INDEX_NAMES = frozenset({
    REVISION_ID_INDEX,
    REVISION_NUMBER_INDEX,
    REVISION_DESC_INDEX,
    CANDIDATE_ID_INDEX,
    CANDIDATE_STATUS_INDEX,
    CANDIDATE_BASE_INDEX,
    CANDIDATE_ROOT_INDEX,
    RUN_ID_INDEX,
    RUN_COMPLETED_INDEX,
    RUN_REVISION_INDEX,
})

_REVISION_KINDS = frozenset({
    "seed",
    "explicit_turning_point",
    "corroborated_growth",
    "operator_reset",
})
_REVISION_CONFIDENCE_VALUES = frozenset({"seed", "high", "operator"})
_REVISION_SCOPE_KINDS = frozenset({
    "private",
    "group",
    "reflection",
    "self_cognition",
    "operator",
})
_CANDIDATE_CHANGE_KINDS = frozenset({
    "explicit_self_redefinition",
    "inferred_growth",
})
_AUTHORSHIP_VALUES = frozenset({"self_declared", "inferred", "absent"})
_CONFIDENCE_VALUES = frozenset({"low", "medium", "high"})
_PRIVACY_REVIEW_VALUES = frozenset({"low", "high"})
_RUN_KINDS = frozenset({"episode", "daily_reflection", "operator_reset"})
_RUN_DISPOSITIONS = frozenset({
    "no_change",
    "candidate_updated",
    "revision_promoted",
    "rejected",
    "failed",
    "deferred",
})
_ATTEMPT_STAGE_KEYS = frozenset({"proposal", "review"})
_FIRST_CONSUMPTION_KEYS = frozenset({
    "episode_id",
    "correlation_id",
    "claimed_at",
    "loaded_revision_number",
    "consumer_kinds",
    "projection_digest",
    "status",
})

_REVISION_KEYS = frozenset({
    "schema_version",
    "revision_id",
    "character_id",
    "revision_number",
    "revision_kind",
    "base_revision_number",
    "effective_identity",
    "changed_paths",
    "change_diff",
    "evidence_summary",
    "source_scope_kinds",
    "evidence_refs",
    "promotion_run_id",
    "promotion_correlation_id",
    "proposal_confidence",
    "review_confidence",
    "created_at",
})
_CANDIDATE_KEYS = frozenset({
    "schema_version",
    "candidate_id",
    "character_id",
    "base_revision_number",
    "status",
    "change_kind",
    "proposed_changes",
    "semantic_summary",
    "evidence_refs",
    "distinct_episode_count",
    "distinct_local_dates",
    "source_scope_kinds",
    "claimed_root_episode_ids",
    "newest_root_captured_at",
    "reversal_of_paths",
    "fresh_post_revision_root_count",
    "character_authorship",
    "proposal_confidence",
    "review_confidence",
    "privacy_review",
    "promoted_revision_number",
    "rejection_reason",
    "created_at",
    "updated_at",
})
_RUN_KEYS = frozenset({
    "schema_version",
    "run_id",
    "run_kind",
    "character_id",
    "base_revision_number",
    "correlation_id",
    "root_episode_ids",
    "source_evidence_count",
    "attempt_count_by_stage",
    "lifecycle_state",
    "disposition",
    "proposal_reason_code",
    "review_reason_code",
    "policy_reason_code",
    "persistence_reason_code",
    "candidate_id",
    "promoted_revision_number",
    "validation_error_codes",
    "first_consumption",
    "post_commit_attempt_count",
    "started_at",
    "completed_at",
})


class CharacterIdentityPersistenceError(DatabaseOperationError):
    """Base failure for the identity persistence boundary."""


class IdentityLedgerNotFoundError(CharacterIdentityPersistenceError):
    """Raised when no revision exists for a character."""


class SeedIdentityConflictError(CharacterIdentityPersistenceError):
    """Raised when revision zero disagrees with the selected seed."""


class IdentityLedgerCorruptionError(CharacterIdentityPersistenceError):
    """Raised when persisted rows violate an immutable ledger invariant."""


class IdentityCandidateConflictError(CharacterIdentityPersistenceError):
    """Raised when a candidate identifier is reused with different content."""


class IdentityRunConflictError(CharacterIdentityPersistenceError):
    """Raised when a run identifier is reused with different content."""


class IdentityRootAlreadyClaimedError(CharacterIdentityPersistenceError):
    """Raised when one repository root belongs to another candidate."""


class ConcurrentIdentityPromotionError(CharacterIdentityPersistenceError):
    """Raised when another writer wins the current-base promotion."""


class IdentityTransactionUnavailableError(CharacterIdentityPersistenceError):
    """Raised when MongoDB cannot provide the required transaction boundary."""


class IdentityRevisionStaleError(CharacterIdentityPersistenceError):
    """Raised when an episode loaded a revision that is no longer latest."""

    def __init__(
        self,
        *,
        loaded_revision_number: int,
        latest_revision_number: int,
    ) -> None:
        """Retain only revision counters needed for one bounded retry."""

        self.loaded_revision_number = loaded_revision_number
        self.latest_revision_number = latest_revision_number
        super().__init__(
            "loaded identity revision is stale: "
            f"{loaded_revision_number} != {latest_revision_number}"
        )


class IdentityPostCommitPendingError(CharacterIdentityPersistenceError):
    """Raised when a revision has not completed post-commit reconciliation."""

    def __init__(self, *, run_id: str, revision_number: int) -> None:
        """Retain the pending run identifier for immediate reconciliation."""

        self.run_id = run_id
        self.revision_number = revision_number
        super().__init__(
            "identity revision post-commit work is pending: "
            f"revision {revision_number}"
        )


async def ensure_character_identity_growth_indexes() -> None:
    """Create the three identity collections and their declared indexes."""

    db = await get_db()
    revisions = db[REVISIONS_COLLECTION]
    candidates = db[CANDIDATES_COLLECTION]
    runs = db[RUNS_COLLECTION]

    await revisions.create_index(
        "revision_id",
        unique=True,
        name=REVISION_ID_INDEX,
    )
    await revisions.create_index(
        [("character_id", ASCENDING), ("revision_number", ASCENDING)],
        unique=True,
        name=REVISION_NUMBER_INDEX,
    )
    await revisions.create_index(
        [("character_id", ASCENDING), ("revision_number", DESCENDING)],
        name=REVISION_DESC_INDEX,
    )
    await candidates.create_index(
        "candidate_id",
        unique=True,
        name=CANDIDATE_ID_INDEX,
    )
    await candidates.create_index(
        [
            ("character_id", ASCENDING),
            ("status", ASCENDING),
            ("updated_at", DESCENDING),
        ],
        name=CANDIDATE_STATUS_INDEX,
    )
    await candidates.create_index(
        [
            ("character_id", ASCENDING),
            ("base_revision_number", ASCENDING),
            ("status", ASCENDING),
        ],
        name=CANDIDATE_BASE_INDEX,
    )
    await candidates.create_index(
        [
            ("character_id", ASCENDING),
            ("claimed_root_episode_ids", ASCENDING),
        ],
        unique=True,
        partialFilterExpression={
            "claimed_root_episode_ids.0": {"$exists": True},
        },
        name=CANDIDATE_ROOT_INDEX,
    )
    await runs.create_index(
        "run_id",
        unique=True,
        name=RUN_ID_INDEX,
    )
    await runs.create_index(
        [("run_kind", ASCENDING), ("completed_at", DESCENDING)],
        name=RUN_COMPLETED_INDEX,
    )
    await runs.create_index(
        [
            ("character_id", ASCENDING),
            ("promoted_revision_number", DESCENDING),
        ],
        name=RUN_REVISION_INDEX,
    )


async def ensure_seed_identity(
    *,
    character_id: str,
    seed: Mapping[str, object],
) -> dict[str, object]:
    """Insert revision zero or verify the existing immutable seed."""

    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    identity = validate_effective_identity(seed)
    db = await get_db()
    collection = db[REVISIONS_COLLECTION]
    existing = await collection.find_one({
        "character_id": normalized_character_id,
        "revision_number": 0,
    })
    if existing is not None:
        revision = _validate_revision_document(existing)
        _require_matching_seed(revision, identity)
        return revision

    later_revision = await collection.find_one({
        "character_id": normalized_character_id,
        "revision_number": {"$gt": 0},
    })
    if later_revision is not None:
        raise IdentityLedgerCorruptionError(
            "identity ledger has later revisions without revision zero"
        )

    created_at = _utc_now_iso()
    revision = _build_revision_document(
        character_id=normalized_character_id,
        revision_number=0,
        revision_kind="seed",
        base_revision_number=None,
        effective_identity=identity,
        changed_paths=[],
        change_diff=[],
        evidence_summary="seed",
        source_scope_kinds=[],
        evidence_refs=[],
        promotion_run_id=None,
        promotion_correlation_id=None,
        proposal_confidence="seed",
        review_confidence="seed",
        created_at=created_at,
    )
    try:
        await collection.insert_one(deepcopy(revision))
    except DuplicateKeyError as exc:
        existing = await collection.find_one({
            "character_id": normalized_character_id,
            "revision_number": 0,
        })
        if existing is None:
            raise IdentityLedgerCorruptionError(
                "revision-zero insert raced without a readable winner"
            ) from exc
        winner = _validate_revision_document(existing)
        _require_matching_seed(winner, identity)
        return winner
    except PyMongoError as exc:
        raise CharacterIdentityPersistenceError(
            f"failed to insert identity revision zero: {exc}"
        ) from exc
    return revision


async def get_current_identity(
    *,
    character_id: str,
) -> dict[str, object]:
    """Return the validated highest-numbered identity revision."""

    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    db = await get_db()
    document = await db[REVISIONS_COLLECTION].find_one(
        {"character_id": normalized_character_id},
        sort=[("revision_number", DESCENDING)],
    )
    if document is None:
        raise IdentityLedgerNotFoundError(
            f"no identity revision exists for character {normalized_character_id}"
        )
    return _validate_revision_document(document)


async def list_identity_revisions(
    *,
    character_id: str,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Return validated review history in descending revision order."""

    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    normalized_limit = _require_integer(
        limit,
        context="limit",
        minimum=1,
        maximum=500,
    )
    db = await get_db()
    cursor = (
        db[REVISIONS_COLLECTION]
        .find({"character_id": normalized_character_id})
        .sort("revision_number", DESCENDING)
        .limit(normalized_limit)
    )
    revisions = [
        _validate_revision_document(document)
        async for document in cursor
    ]
    return revisions


async def list_current_growth_candidates(
    *,
    character_id: str,
    base_revision_number: int,
    limit: int = models.IDENTITY_CANDIDATE_PROMPT_LIMIT,
) -> list[dict[str, object]]:
    """Return bounded active candidates for exactly one current base."""

    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    normalized_base = _require_integer(
        base_revision_number,
        context="base_revision_number",
        minimum=0,
    )
    normalized_limit = _require_integer(
        limit,
        context="limit",
        minimum=1,
        maximum=models.IDENTITY_CANDIDATE_PROMPT_LIMIT,
    )
    db = await get_db()
    cursor = (
        db[CANDIDATES_COLLECTION]
        .find({
            "character_id": normalized_character_id,
            "base_revision_number": normalized_base,
            "status": {"$in": ["emerging", "ready"]},
        })
        .sort("updated_at", DESCENDING)
        .limit(normalized_limit)
    )
    candidates = [
        _validate_candidate_document(document)
        async for document in cursor
    ]
    return candidates


async def list_identity_growth_candidates(
    *,
    character_id: str,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Return bounded candidate history for operator review."""

    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    normalized_limit = _require_integer(
        limit,
        context="limit",
        minimum=1,
        maximum=100,
    )
    db = await get_db()
    cursor = (
        db[CANDIDATES_COLLECTION]
        .find({"character_id": normalized_character_id})
        .sort("updated_at", DESCENDING)
        .limit(normalized_limit)
    )
    candidates = [
        _validate_candidate_document(document)
        async for document in cursor
    ]
    return candidates


async def list_recent_identity_growth_runs(
    *,
    character_id: str,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Return bounded sanitized run history for operator review."""

    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    normalized_limit = _require_integer(
        limit,
        context="limit",
        minimum=1,
        maximum=100,
    )
    db = await get_db()
    cursor = (
        db[RUNS_COLLECTION]
        .find({"character_id": normalized_character_id})
        .sort("started_at", DESCENDING)
        .limit(normalized_limit)
    )
    runs = [
        _validate_run_document(document)
        async for document in cursor
    ]
    return runs


async def build_identity_growth_health(
    *,
    character_id: str,
) -> models.CharacterIdentityGrowthHealthV1:
    """Derive the exact redacted health funnel from three identity ledgers."""

    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    current_revision = await get_current_identity(
        character_id=normalized_character_id,
    )
    latest_revision_number = int(current_revision["revision_number"])
    db = await get_db()
    candidates = db[CANDIDATES_COLLECTION]
    runs = db[RUNS_COLLECTION]
    revisions = db[REVISIONS_COLLECTION]

    latest_run_document = await runs.find_one(
        {"character_id": normalized_character_id},
        sort=[("started_at", DESCENDING)],
    )
    latest_run = (
        _validate_run_document(latest_run_document)
        if latest_run_document is not None
        else None
    )
    latest_candidate_document = await candidates.find_one(
        {
            "character_id": normalized_character_id,
            "base_revision_number": latest_revision_number,
        },
        sort=[("updated_at", DESCENDING)],
    )
    latest_candidate = (
        _validate_candidate_document(latest_candidate_document)
        if latest_candidate_document is not None
        else None
    )

    promotion_run_id = current_revision.get("promotion_run_id")
    receipt_run: dict[str, object] | None = None
    if isinstance(promotion_run_id, str) and promotion_run_id:
        receipt_run_document = await runs.find_one({
            "run_id": promotion_run_id,
            "character_id": normalized_character_id,
        })
        if receipt_run_document is None:
            raise IdentityLedgerCorruptionError(
                "latest identity revision has no matching growth run"
            )
        receipt_run = _validate_run_document(receipt_run_document)

    (
        routed_count,
        no_change_count,
        emerging_candidate_count,
        ready_candidate_count,
        rejected_count,
        failed_count,
        promoted_count,
        consumed_count,
    ) = await asyncio.gather(
        runs.count_documents({
            "character_id": normalized_character_id,
            "run_kind": {"$in": ["episode", "daily_reflection"]},
        }),
        runs.count_documents({
            "character_id": normalized_character_id,
            "disposition": "no_change",
        }),
        candidates.count_documents({
            "character_id": normalized_character_id,
            "status": "emerging",
        }),
        candidates.count_documents({
            "character_id": normalized_character_id,
            "status": "ready",
        }),
        runs.count_documents({
            "character_id": normalized_character_id,
            "disposition": "rejected",
        }),
        runs.count_documents({
            "character_id": normalized_character_id,
            "$or": [
                {"lifecycle_state": "failed"},
                {"disposition": "failed"},
            ],
        }),
        revisions.count_documents({
            "character_id": normalized_character_id,
            "revision_number": {"$gt": 0},
        }),
        runs.count_documents({
            "character_id": normalized_character_id,
            "first_consumption.status": "consumed",
        }),
    )

    latest_consumed_document = await runs.find_one(
        {
            "character_id": normalized_character_id,
            "first_consumption.status": "consumed",
        },
        sort=[("first_consumption.loaded_revision_number", DESCENDING)],
    )
    latest_consumed_revision_number: int | None = None
    if latest_consumed_document is not None:
        latest_consumed_run = _validate_run_document(
            latest_consumed_document
        )
        latest_receipt = latest_consumed_run["first_consumption"]
        if not isinstance(latest_receipt, Mapping):
            raise IdentityLedgerCorruptionError(
                "consumed identity run has no receipt"
            )
        latest_consumed_revision_number = int(
            latest_receipt["loaded_revision_number"]
        )

    receipt_status: object = None
    if receipt_run is not None:
        receipt = receipt_run["first_consumption"]
        if isinstance(receipt, Mapping):
            receipt_status = receipt["status"]

    latest_reason_code = _health_latest_reason_code(
        latest_revision_number=latest_revision_number,
        latest_run=latest_run,
        receipt_run=receipt_run,
        receipt_status=receipt_status,
    )
    latest_run_lifecycle_state = (
        str(latest_run["lifecycle_state"])
        if latest_run is not None
        else None
    )
    state = derive_growth_health_state(
        latest_revision_number=latest_revision_number,
        receipt_status=receipt_status,
        latest_run_lifecycle_state=latest_run_lifecycle_state,
        latest_reason_code=latest_reason_code,
        ready_candidate_count=ready_candidate_count,
        emerging_candidate_count=emerging_candidate_count,
    )

    root_count = 0
    local_date_count = 0
    if latest_candidate is not None:
        root_count = int(latest_candidate["distinct_episode_count"])
        local_dates = latest_candidate["distinct_local_dates"]
        if not isinstance(local_dates, list):
            raise IdentityLedgerCorruptionError(
                "identity candidate local dates are invalid"
            )
        local_date_count = len(local_dates)
    elif latest_run is not None:
        roots = latest_run["root_episode_ids"]
        if not isinstance(roots, list):
            raise IdentityLedgerCorruptionError(
                "identity growth run roots are invalid"
            )
        root_count = len(roots)

    return {
        "state": state,
        "routed_count": routed_count,
        "no_change_count": no_change_count,
        "emerging_candidate_count": emerging_candidate_count,
        "ready_candidate_count": ready_candidate_count,
        "rejected_count": rejected_count,
        "failed_count": failed_count,
        "promoted_count": promoted_count,
        "consumed_count": consumed_count,
        "latest_revision_number": latest_revision_number,
        "latest_consumed_revision_number": latest_consumed_revision_number,
        "latest_reason_code": latest_reason_code,
        "root_count": root_count,
        "local_date_count": local_date_count,
    }


async def get_growth_run(
    *,
    run_id: str,
) -> dict[str, object] | None:
    """Return one validated growth run for idempotent replay handling."""

    normalized_run_id = _require_identifier(run_id, context="run_id")
    db = await get_db()
    document = await db[RUNS_COLLECTION].find_one({
        "run_id": normalized_run_id,
    })
    if document is None:
        return None
    return _validate_run_document(document)


async def list_post_commit_pending_growth_runs(
    *,
    character_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Return bounded promoted runs awaiting idempotent side effects."""

    query: dict[str, object] = {
        "lifecycle_state": "post_commit_pending",
        "disposition": "revision_promoted",
    }
    if character_id is not None:
        query["character_id"] = _require_identifier(
            character_id,
            context="character_id",
        )
    normalized_limit = _require_integer(
        limit,
        context="limit",
        minimum=1,
        maximum=500,
    )
    db = await get_db()
    cursor = (
        db[RUNS_COLLECTION]
        .find(query)
        .sort([("completed_at", ASCENDING), ("run_id", ASCENDING)])
        .limit(normalized_limit)
    )
    return [
        _validate_run_document(document)
        async for document in cursor
    ]


async def complete_growth_run_post_commit(
    *,
    run_id: str,
    character_id: str,
    revision_number: int,
) -> dict[str, object]:
    """Mark one revision's invalidation and event side effects complete."""

    normalized_run_id = _require_identifier(run_id, context="run_id")
    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    normalized_revision = _require_integer(
        revision_number,
        context="revision_number",
        minimum=1,
    )
    db = await get_db()
    collection = db[RUNS_COLLECTION]
    try:
        result = await collection.update_one(
            {
                "run_id": normalized_run_id,
                "character_id": normalized_character_id,
                "promoted_revision_number": normalized_revision,
                "lifecycle_state": "post_commit_pending",
                "post_commit_attempt_count": {"$lt": 1000},
            },
            {
                "$set": {"lifecycle_state": "complete"},
                "$inc": {"post_commit_attempt_count": 1},
            },
        )
    except PyMongoError as exc:
        raise CharacterIdentityPersistenceError(
            f"failed to complete identity post-commit work: {exc}"
        ) from exc
    document = await collection.find_one({"run_id": normalized_run_id})
    if document is None:
        raise IdentityRunConflictError(
            "identity post-commit run does not exist"
        )
    run = _validate_run_document(document)
    if (
        run["character_id"] != normalized_character_id
        or run["promoted_revision_number"] != normalized_revision
        or run["disposition"] != "revision_promoted"
    ):
        raise IdentityRunConflictError(
            "identity post-commit run ownership does not match"
        )
    if result.modified_count == 0 and run["lifecycle_state"] != "complete":
        raise IdentityRunConflictError(
            "identity post-commit run could not transition to complete"
        )
    return run


async def record_growth_run_post_commit_failure(
    *,
    run_id: str,
    character_id: str,
    revision_number: int,
) -> dict[str, object]:
    """Increment retry evidence while retaining a pending promoted run."""

    normalized_run_id = _require_identifier(run_id, context="run_id")
    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    normalized_revision = _require_integer(
        revision_number,
        context="revision_number",
        minimum=1,
    )
    db = await get_db()
    collection = db[RUNS_COLLECTION]
    try:
        result = await collection.update_one(
            {
                "run_id": normalized_run_id,
                "character_id": normalized_character_id,
                "promoted_revision_number": normalized_revision,
                "lifecycle_state": "post_commit_pending",
                "post_commit_attempt_count": {"$lt": 1000},
            },
            {"$inc": {"post_commit_attempt_count": 1}},
        )
    except PyMongoError as exc:
        raise CharacterIdentityPersistenceError(
            f"failed to retain identity post-commit retry: {exc}"
        ) from exc
    if result.modified_count != 1:
        raise IdentityRunConflictError(
            "identity post-commit failure could not be recorded"
        )
    document = await collection.find_one({"run_id": normalized_run_id})
    if document is None:
        raise IdentityRunConflictError(
            "identity post-commit run disappeared after retry update"
        )
    return _validate_run_document(document)


async def claim_identity_revision_consumption(
    *,
    character_id: str,
    episode_id: str,
    correlation_id: str,
    loaded_revision_number: int,
    consumer_kinds: Sequence[str],
    projection_digest: str,
    now: datetime | None = None,
) -> models.IdentityRevisionConsumptionV1 | None:
    """Claim a promoted revision's first consumer after a latest check."""

    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    receipt = _validate_first_consumption({
        "episode_id": episode_id,
        "correlation_id": correlation_id,
        "claimed_at": _datetime_to_iso(now),
        "loaded_revision_number": loaded_revision_number,
        "consumer_kinds": list(consumer_kinds),
        "projection_digest": projection_digest,
        "status": "consumed",
    })
    db = await get_db()
    for attempt_number in range(1, 4):
        try:
            async with await db.client.start_session() as session:
                async with session.start_transaction():
                    claimed = await _claim_consumption_transaction(
                        db=db,
                        session=session,
                        character_id=normalized_character_id,
                        receipt=receipt,
                        require_latest_match=True,
                    )
            return claimed
        except PyMongoError as exc:
            committed = await _find_consumption_committed_for_revision(
                db=db,
                character_id=normalized_character_id,
                revision_number=receipt["loaded_revision_number"],
            )
            if committed is not None:
                return committed
            if _transaction_is_unavailable(exc):
                raise IdentityTransactionUnavailableError(
                    "identity consumption requires MongoDB transaction support"
                ) from exc
            if _transaction_is_retryable(exc) and attempt_number < 3:
                continue
            raise CharacterIdentityPersistenceError(
                "identity consumption transaction failed"
            ) from exc
    raise CharacterIdentityPersistenceError(
        "identity consumption exhausted its transaction attempts"
    )


async def record_identity_revision_consumption_mismatch(
    *,
    character_id: str,
    episode_id: str,
    correlation_id: str,
    loaded_revision_number: int,
    consumer_kinds: Sequence[str],
    projection_digest: str,
    now: datetime | None = None,
) -> models.IdentityRevisionConsumptionV1:
    """Persist a fail-closed receipt on the revision that stayed latest."""

    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    receipt = _validate_first_consumption({
        "episode_id": episode_id,
        "correlation_id": correlation_id,
        "claimed_at": _datetime_to_iso(now),
        "loaded_revision_number": loaded_revision_number,
        "consumer_kinds": list(consumer_kinds),
        "projection_digest": projection_digest,
        "status": "mismatch",
    })
    db = await get_db()
    for attempt_number in range(1, 4):
        try:
            async with await db.client.start_session() as session:
                async with session.start_transaction():
                    claimed = await _claim_consumption_transaction(
                        db=db,
                        session=session,
                        character_id=normalized_character_id,
                        receipt=receipt,
                        require_latest_match=False,
                    )
            if claimed is None:
                raise IdentityLedgerCorruptionError(
                    "seed revision cannot own a consumption mismatch"
                )
            return claimed
        except PyMongoError as exc:
            if _transaction_is_unavailable(exc):
                raise IdentityTransactionUnavailableError(
                    "identity mismatch receipt requires transaction support"
                ) from exc
            if _transaction_is_retryable(exc) and attempt_number < 3:
                continue
            raise CharacterIdentityPersistenceError(
                "identity mismatch receipt transaction failed"
            ) from exc
    raise CharacterIdentityPersistenceError(
        "identity mismatch receipt exhausted its transaction attempts"
    )


async def count_inferred_identity_promotions_on_local_date(
    *,
    character_id: str,
    character_local_date: str,
) -> int:
    """Count corroborated-growth revisions created on one local date."""

    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    start_utc, exclusive_end_utc = local_date_bounds_to_storage_utc_iso(
        character_local_date,
    )
    db = await get_db()
    count = await db[REVISIONS_COLLECTION].count_documents({
        "character_id": normalized_character_id,
        "revision_kind": "corroborated_growth",
        "created_at": {
            "$gte": start_utc,
            "$lt": exclusive_end_utc,
        },
    })
    return int(count)


async def insert_growth_candidate(
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Insert one validated candidate with exclusive root ownership."""

    validated = _validate_candidate_document(candidate)
    db = await get_db()
    collection = db[CANDIDATES_COLLECTION]
    try:
        await collection.insert_one(deepcopy(validated))
    except DuplicateKeyError as exc:
        existing = await collection.find_one({
            "candidate_id": validated["candidate_id"],
        })
        if existing is not None:
            current = _validate_candidate_document(existing)
            if current == validated:
                return current
            raise IdentityCandidateConflictError(
                f"candidate_id {validated['candidate_id']!r} "
                "already has different content"
            ) from exc
        claimed_root = await _find_claimed_root(
            collection=collection,
            character_id=validated["character_id"],
            roots=validated["claimed_root_episode_ids"],
        )
        if claimed_root is not None:
            raise IdentityRootAlreadyClaimedError(
                f"root episode {claimed_root!r} is already claimed"
            ) from exc
        raise IdentityCandidateConflictError(
            "candidate insert violated a unique identity constraint"
        ) from exc
    except PyMongoError as exc:
        raise CharacterIdentityPersistenceError(
            f"failed to insert identity candidate: {exc}"
        ) from exc
    return validated


async def update_growth_candidate(
    candidate: Mapping[str, object],
    *,
    expected_updated_at: str,
) -> dict[str, object]:
    """Replace one active candidate through a guarded lifecycle transition."""

    validated = _validate_candidate_document(candidate)
    normalized_expected_updated_at = _require_iso_datetime(
        expected_updated_at,
        context="expected_updated_at",
    )
    db = await get_db()
    collection = db[CANDIDATES_COLLECTION]
    existing_document = await collection.find_one({
        "candidate_id": validated["candidate_id"],
    })
    if existing_document is None:
        raise IdentityCandidateConflictError(
            "identity candidate does not exist for update"
        )
    existing = _validate_candidate_document(existing_document)
    if (
        existing["character_id"] != validated["character_id"]
        or existing["base_revision_number"]
        != validated["base_revision_number"]
        or existing["created_at"] != validated["created_at"]
    ):
        raise IdentityCandidateConflictError(
            "identity candidate immutable ownership changed"
        )
    if existing["updated_at"] != normalized_expected_updated_at:
        raise IdentityCandidateConflictError(
            "identity candidate changed before update"
        )
    if not candidate_transition_allowed(
        str(existing["status"]),
        str(validated["status"]),
    ):
        raise IdentityCandidateConflictError(
            "identity candidate lifecycle transition is not allowed"
        )

    try:
        result = await collection.replace_one(
            {
                "candidate_id": validated["candidate_id"],
                "status": existing["status"],
                "updated_at": normalized_expected_updated_at,
            },
            deepcopy(validated),
        )
    except DuplicateKeyError as exc:
        claimed_root = await _find_claimed_root(
            collection=collection,
            character_id=str(validated["character_id"]),
            roots=validated["claimed_root_episode_ids"],
            exclude_candidate_id=str(validated["candidate_id"]),
        )
        if claimed_root is not None:
            raise IdentityRootAlreadyClaimedError(
                f"root episode {claimed_root!r} is already claimed"
            ) from exc
        raise IdentityCandidateConflictError(
            "candidate update violated a unique identity constraint"
        ) from exc
    except PyMongoError as exc:
        raise CharacterIdentityPersistenceError(
            f"failed to update identity candidate: {exc}"
        ) from exc
    if result.modified_count != 1:
        raise IdentityCandidateConflictError(
            "identity candidate changed during update"
        )
    return validated


async def reject_growth_candidates(
    *,
    character_id: str,
    base_revision_number: int,
    candidate_ids: Sequence[str],
    reason_code: str,
    updated_at: str,
) -> int:
    """Reject reviewed incompatible active candidates by opaque identifiers."""

    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    normalized_base = _require_integer(
        base_revision_number,
        context="base_revision_number",
        minimum=0,
    )
    normalized_ids = _require_sorted_unique_strings(
        candidate_ids,
        context="candidate_ids",
        max_items=models.IDENTITY_CANDIDATE_PROMPT_LIMIT,
    )
    normalized_reason = _require_reason_code(
        reason_code,
        context="reason_code",
    )
    normalized_updated_at = _require_iso_datetime(
        updated_at,
        context="updated_at",
    )
    if not normalized_ids:
        return 0

    db = await get_db()
    try:
        result = await db[CANDIDATES_COLLECTION].update_many(
            {
                "character_id": normalized_character_id,
                "base_revision_number": normalized_base,
                "candidate_id": {"$in": normalized_ids},
                "status": {"$in": ["emerging", "ready"]},
            },
            {
                "$set": {
                    "status": "rejected",
                    "rejection_reason": normalized_reason,
                    "updated_at": normalized_updated_at,
                },
            },
        )
    except PyMongoError as exc:
        raise CharacterIdentityPersistenceError(
            f"failed to reject identity candidates: {exc}"
        ) from exc
    return int(result.modified_count)


async def insert_growth_run(
    run: Mapping[str, object],
) -> dict[str, object]:
    """Insert one sanitized identity growth run idempotently."""

    validated = _validate_run_document(run)
    db = await get_db()
    collection = db[RUNS_COLLECTION]
    try:
        await collection.insert_one(deepcopy(validated))
    except DuplicateKeyError as exc:
        existing = await collection.find_one({"run_id": validated["run_id"]})
        if existing is not None:
            current = _validate_run_document(existing)
            if current == validated:
                return current
        raise IdentityRunConflictError(
            f"run_id {validated['run_id']!r} already has different content"
        ) from exc
    except PyMongoError as exc:
        raise CharacterIdentityPersistenceError(
            f"failed to insert identity growth run: {exc}"
        ) from exc
    return validated


async def promote_ready_candidate(
    *,
    character_id: str,
    candidate_id: str,
    run_id: str,
    now: datetime | None = None,
) -> dict[str, object]:
    """Atomically promote one ready current-base candidate."""

    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    normalized_candidate_id = _require_identifier(
        candidate_id,
        context="candidate_id",
    )
    normalized_run_id = _require_identifier(run_id, context="run_id")
    created_at = _datetime_to_iso(now)
    db = await get_db()

    for attempt_number in range(1, 4):
        try:
            async with await db.client.start_session() as session:
                async with session.start_transaction():
                    revision = await _promote_ready_candidate_transaction(
                        db=db,
                        session=session,
                        character_id=normalized_character_id,
                        candidate_id=normalized_candidate_id,
                        run_id=normalized_run_id,
                        created_at=created_at,
                    )
            return revision
        except PyMongoError as exc:
            committed = await _find_revision_committed_by_run(
                db=db,
                character_id=normalized_character_id,
                run_id=normalized_run_id,
            )
            if committed is not None:
                return committed
            if _transaction_is_unavailable(exc):
                raise IdentityTransactionUnavailableError(
                    "identity promotion requires MongoDB transaction support"
                ) from exc
            if _transaction_is_retryable(exc) and attempt_number < 3:
                continue
            raise ConcurrentIdentityPromotionError(
                "identity promotion lost the current-base transaction"
            ) from exc
    raise ConcurrentIdentityPromotionError(
        "identity promotion exhausted its transaction attempts"
    )


async def create_operator_reset_revision(
    *,
    character_id: str,
    identity: Mapping[str, object],
    operator_action_id: str,
    correlation_id: str,
    now: datetime | None = None,
) -> dict[str, object]:
    """Create one idempotent full operator-reset revision without an LLM."""

    normalized_character_id = _require_identifier(
        character_id,
        context="character_id",
    )
    normalized_action_id = _require_identifier(
        operator_action_id,
        context="operator_action_id",
    )
    normalized_correlation_id = _require_identifier(
        correlation_id,
        context="correlation_id",
    )
    validated_identity = validate_effective_identity(identity)
    run_id = f"operator-reset:{normalized_action_id}"
    created_at = _datetime_to_iso(now)
    db = await get_db()

    for attempt_number in range(1, 4):
        try:
            async with await db.client.start_session() as session:
                async with session.start_transaction():
                    revision = await _operator_reset_transaction(
                        db=db,
                        session=session,
                        character_id=normalized_character_id,
                        identity=validated_identity,
                        run_id=run_id,
                        correlation_id=normalized_correlation_id,
                        created_at=created_at,
                    )
            return revision
        except PyMongoError as exc:
            committed = await _find_revision_committed_by_run(
                db=db,
                character_id=normalized_character_id,
                run_id=run_id,
            )
            if committed is not None:
                return committed
            if _transaction_is_unavailable(exc):
                raise IdentityTransactionUnavailableError(
                    "operator reset requires MongoDB transaction support"
                ) from exc
            if _transaction_is_retryable(exc) and attempt_number < 3:
                continue
            raise ConcurrentIdentityPromotionError(
                "operator reset lost the current-base transaction"
            ) from exc
    raise ConcurrentIdentityPromotionError(
        "operator reset exhausted its transaction attempts"
    )


async def _claim_consumption_transaction(
    *,
    db,
    session,
    character_id: str,
    receipt: models.IdentityRevisionConsumptionV1,
    require_latest_match: bool,
) -> models.IdentityRevisionConsumptionV1 | None:
    """Conditionally claim one latest revision inside a transaction."""

    revisions = db[REVISIONS_COLLECTION]
    runs = db[RUNS_COLLECTION]
    latest_document = await revisions.find_one(
        {"character_id": character_id},
        sort=[("revision_number", DESCENDING)],
        session=session,
    )
    if latest_document is None:
        raise IdentityLedgerNotFoundError(
            f"no identity revision exists for character {character_id}"
        )
    latest = _validate_revision_document(latest_document)
    latest_revision_number = int(latest["revision_number"])
    loaded_revision_number = receipt["loaded_revision_number"]
    if (
        require_latest_match
        and loaded_revision_number != latest_revision_number
    ):
        raise IdentityRevisionStaleError(
            loaded_revision_number=loaded_revision_number,
            latest_revision_number=latest_revision_number,
        )
    if latest_revision_number == 0:
        return None

    promotion_run_id = latest["promotion_run_id"]
    if not isinstance(promotion_run_id, str) or not promotion_run_id:
        raise IdentityLedgerCorruptionError(
            "promoted identity revision has no promotion run"
        )
    raw_run = await runs.find_one(
        {"run_id": promotion_run_id},
        session=session,
    )
    if raw_run is None:
        raise IdentityLedgerCorruptionError(
            "promoted identity revision has no growth run"
        )
    run = _validate_run_document(raw_run)
    if (
        run["character_id"] != character_id
        or run["promoted_revision_number"] != latest_revision_number
        or run["disposition"] != "revision_promoted"
    ):
        raise IdentityLedgerCorruptionError(
            "identity consumption run does not match latest revision"
        )
    if run["lifecycle_state"] == "post_commit_pending":
        raise IdentityPostCommitPendingError(
            run_id=promotion_run_id,
            revision_number=latest_revision_number,
        )
    if run["lifecycle_state"] != "complete":
        raise IdentityLedgerCorruptionError(
            "identity consumption requires a complete promotion run"
        )
    existing_receipt = run["first_consumption"]
    if existing_receipt is not None:
        return _validate_first_consumption(existing_receipt)

    result = await runs.update_one(
        {
            "run_id": promotion_run_id,
            "character_id": character_id,
            "promoted_revision_number": latest_revision_number,
            "lifecycle_state": "complete",
            "first_consumption": None,
        },
        {"$set": {"first_consumption": deepcopy(receipt)}},
        session=session,
    )
    if result.modified_count == 1:
        return deepcopy(receipt)
    winner_document = await runs.find_one(
        {"run_id": promotion_run_id},
        session=session,
    )
    if winner_document is None:
        raise IdentityLedgerCorruptionError(
            "identity consumption run disappeared during claim"
        )
    winner = _validate_run_document(winner_document)
    winner_receipt = winner["first_consumption"]
    if winner_receipt is None:
        raise IdentityRunConflictError(
            "identity consumption claim lost without a durable winner"
        )
    return _validate_first_consumption(winner_receipt)


async def _find_consumption_committed_for_revision(
    *,
    db,
    character_id: str,
    revision_number: int,
) -> models.IdentityRevisionConsumptionV1 | None:
    """Resolve an ambiguous transaction commit through immutable lineage."""

    if revision_number == 0:
        return None
    revision_document = await db[REVISIONS_COLLECTION].find_one({
        "character_id": character_id,
        "revision_number": revision_number,
    })
    if revision_document is None:
        return None
    revision = _validate_revision_document(revision_document)
    run_id = revision["promotion_run_id"]
    if not isinstance(run_id, str) or not run_id:
        return None
    run_document = await db[RUNS_COLLECTION].find_one({"run_id": run_id})
    if run_document is None:
        return None
    run = _validate_run_document(run_document)
    receipt = run["first_consumption"]
    if receipt is None:
        return None
    return _validate_first_consumption(receipt)


async def _promote_ready_candidate_transaction(
    *,
    db,
    session,
    character_id: str,
    candidate_id: str,
    run_id: str,
    created_at: str,
) -> dict[str, object]:
    """Execute the guarded candidate-promotion transaction."""

    runs = db[RUNS_COLLECTION]
    revisions = db[REVISIONS_COLLECTION]
    candidates = db[CANDIDATES_COLLECTION]
    raw_run = await runs.find_one({"run_id": run_id}, session=session)
    if raw_run is None:
        raise IdentityRunConflictError(
            f"identity promotion run does not exist: {run_id}"
        )
    run = _validate_run_document(raw_run)
    if run["character_id"] != character_id:
        raise IdentityRunConflictError(
            "identity promotion run belongs to another character"
        )
    if run["candidate_id"] != candidate_id:
        raise IdentityRunConflictError(
            "identity promotion run names another candidate"
        )
    if (
        run["disposition"] == "revision_promoted"
        and run["promoted_revision_number"] is not None
    ):
        existing = await revisions.find_one(
            {
                "character_id": character_id,
                "revision_number": run["promoted_revision_number"],
                "promotion_run_id": run_id,
            },
            session=session,
        )
        if existing is None:
            raise IdentityLedgerCorruptionError(
                "promoted run has no matching immutable revision"
            )
        return _validate_revision_document(existing)

    current_document = await revisions.find_one(
        {"character_id": character_id},
        sort=[("revision_number", DESCENDING)],
        session=session,
    )
    if current_document is None:
        raise IdentityLedgerNotFoundError(
            f"no seed revision exists for character {character_id}"
        )
    current = _validate_revision_document(current_document)
    raw_candidate = await candidates.find_one(
        {"candidate_id": candidate_id},
        session=session,
    )
    if raw_candidate is None:
        raise IdentityCandidateConflictError(
            f"identity candidate does not exist: {candidate_id}"
        )
    candidate = _validate_candidate_document(raw_candidate)
    if candidate["character_id"] != character_id:
        raise IdentityCandidateConflictError(
            "identity candidate belongs to another character"
        )
    if candidate["status"] != "ready":
        raise ConcurrentIdentityPromotionError(
            f"identity candidate is not ready: {candidate['status']}"
        )
    if candidate["base_revision_number"] != current["revision_number"]:
        raise ConcurrentIdentityPromotionError(
            "identity candidate base is no longer current"
        )
    if run["base_revision_number"] != current["revision_number"]:
        raise ConcurrentIdentityPromotionError(
            "identity promotion run base is no longer current"
        )

    effective_identity, change_diff = apply_identity_patches(
        current["effective_identity"],
        candidate["proposed_changes"],
    )
    next_revision_number = current["revision_number"] + 1
    promoted_result = await candidates.update_one(
        {
            "candidate_id": candidate_id,
            "character_id": character_id,
            "base_revision_number": current["revision_number"],
            "status": "ready",
        },
        {
            "$set": {
                "status": "promoted",
                "promoted_revision_number": next_revision_number,
                "updated_at": created_at,
            }
        },
        session=session,
    )
    if promoted_result.modified_count != 1:
        raise ConcurrentIdentityPromotionError(
            "identity candidate status changed during promotion"
        )

    changed_paths = [row["path"] for row in change_diff]
    await candidates.update_many(
        {
            "character_id": character_id,
            "candidate_id": {"$ne": candidate_id},
            "base_revision_number": current["revision_number"],
            "status": {"$in": ["emerging", "ready"]},
            "proposed_changes.path": {"$in": changed_paths},
        },
        {
            "$set": {
                "status": "superseded",
                "rejection_reason": "contradiction_blocked",
                "updated_at": created_at,
            }
        },
        session=session,
    )

    revision_kind = (
        "explicit_turning_point"
        if candidate["change_kind"] == "explicit_self_redefinition"
        else "corroborated_growth"
    )
    revision = _build_revision_document(
        character_id=character_id,
        revision_number=next_revision_number,
        revision_kind=revision_kind,
        base_revision_number=current["revision_number"],
        effective_identity=effective_identity,
        changed_paths=changed_paths,
        change_diff=change_diff,
        evidence_summary=candidate["semantic_summary"],
        source_scope_kinds=candidate["source_scope_kinds"],
        evidence_refs=candidate["evidence_refs"],
        promotion_run_id=run_id,
        promotion_correlation_id=run["correlation_id"],
        proposal_confidence="high",
        review_confidence="high",
        created_at=created_at,
    )
    await revisions.insert_one(deepcopy(revision), session=session)
    run_result = await runs.update_one(
        {
            "run_id": run_id,
            "lifecycle_state": "in_progress",
            "base_revision_number": current["revision_number"],
        },
        {
            "$set": {
                "lifecycle_state": "post_commit_pending",
                "disposition": "revision_promoted",
                "persistence_reason_code": "revision_promoted",
                "promoted_revision_number": next_revision_number,
                "completed_at": created_at,
            }
        },
        session=session,
    )
    if run_result.modified_count != 1:
        raise IdentityRunConflictError(
            "identity promotion run changed before finalization"
        )
    return revision


async def _operator_reset_transaction(
    *,
    db,
    session,
    character_id: str,
    identity: models.CharacterEffectiveIdentityV1,
    run_id: str,
    correlation_id: str,
    created_at: str,
) -> dict[str, object]:
    """Execute one full operator-reset transaction."""

    revisions = db[REVISIONS_COLLECTION]
    runs = db[RUNS_COLLECTION]
    existing_run = await runs.find_one({"run_id": run_id}, session=session)
    if existing_run is not None:
        run = _validate_run_document(existing_run)
        if (
            run["character_id"] != character_id
            or run["correlation_id"] != correlation_id
            or run["run_kind"] != "operator_reset"
        ):
            raise IdentityRunConflictError(
                "operator action ID is already linked to another reset"
            )
        if run["promoted_revision_number"] is None:
            raise IdentityLedgerCorruptionError(
                "operator reset run has no promoted revision"
            )
        revision = await revisions.find_one(
            {
                "character_id": character_id,
                "revision_number": run["promoted_revision_number"],
                "promotion_run_id": run_id,
            },
            session=session,
        )
        if revision is None:
            raise IdentityLedgerCorruptionError(
                "operator reset run has no matching revision"
            )
        validated_revision = _validate_revision_document(revision)
        if validated_revision["effective_identity"] != identity:
            raise IdentityRunConflictError(
                "operator action ID was retried with another identity"
            )
        return validated_revision

    current_document = await revisions.find_one(
        {"character_id": character_id},
        sort=[("revision_number", DESCENDING)],
        session=session,
    )
    if current_document is None:
        raise IdentityLedgerNotFoundError(
            "operator reset requires an existing seed revision"
        )
    current = _validate_revision_document(current_document)
    next_revision_number = current["revision_number"] + 1
    change_diff = diff_effective_identities(
        current["effective_identity"],
        identity,
    )
    changed_paths = [row["path"] for row in change_diff]
    run = _build_operator_run_document(
        run_id=run_id,
        character_id=character_id,
        base_revision_number=current["revision_number"],
        correlation_id=correlation_id,
        promoted_revision_number=next_revision_number,
        created_at=created_at,
    )
    revision = _build_revision_document(
        character_id=character_id,
        revision_number=next_revision_number,
        revision_kind="operator_reset",
        base_revision_number=current["revision_number"],
        effective_identity=identity,
        changed_paths=changed_paths,
        change_diff=change_diff,
        evidence_summary="operator reset",
        source_scope_kinds=["operator"],
        evidence_refs=[],
        promotion_run_id=run_id,
        promotion_correlation_id=correlation_id,
        proposal_confidence="operator",
        review_confidence="operator",
        created_at=created_at,
    )
    await runs.insert_one(deepcopy(run), session=session)
    await revisions.insert_one(deepcopy(revision), session=session)
    return revision


async def _find_revision_committed_by_run(
    *,
    db,
    character_id: str,
    run_id: str,
) -> dict[str, object] | None:
    """Resolve an idempotent committed result after a transaction error."""

    run_document = await db[RUNS_COLLECTION].find_one({"run_id": run_id})
    if run_document is None:
        return None
    run = _validate_run_document(run_document)
    if (
        run["character_id"] != character_id
        or run["promoted_revision_number"] is None
        or run["disposition"] != "revision_promoted"
    ):
        return None
    revision = await db[REVISIONS_COLLECTION].find_one({
        "character_id": character_id,
        "revision_number": run["promoted_revision_number"],
        "promotion_run_id": run_id,
    })
    if revision is None:
        raise IdentityLedgerCorruptionError(
            "committed growth run has no matching identity revision"
        )
    return _validate_revision_document(revision)


async def _find_claimed_root(
    *,
    collection,
    character_id: str,
    roots: Sequence[str],
    exclude_candidate_id: str | None = None,
) -> str | None:
    """Return the first input root already owned by another candidate."""

    if not roots:
        return None
    query: dict[str, object] = {
        "character_id": character_id,
        "claimed_root_episode_ids": {"$in": list(roots)},
    }
    if exclude_candidate_id is not None:
        query["candidate_id"] = {"$ne": exclude_candidate_id}
    existing = await collection.find_one(query)
    if existing is None:
        return None
    existing_roots = set(existing.get("claimed_root_episode_ids", []))
    for root in roots:
        if root in existing_roots:
            return root
    return None


def _build_revision_document(
    *,
    character_id: str,
    revision_number: int,
    revision_kind: str,
    base_revision_number: int | None,
    effective_identity: Mapping[str, object],
    changed_paths: Sequence[str],
    change_diff: Sequence[Mapping[str, object]],
    evidence_summary: str,
    source_scope_kinds: Sequence[str],
    evidence_refs: Sequence[Mapping[str, object]],
    promotion_run_id: str | None,
    promotion_correlation_id: str | None,
    proposal_confidence: str,
    review_confidence: str,
    created_at: str,
) -> dict[str, object]:
    """Build and validate one immutable revision document."""

    document: dict[str, object] = {
        "schema_version": REVISION_SCHEMA_VERSION,
        "revision_id": uuid4().hex,
        "character_id": character_id,
        "revision_number": revision_number,
        "revision_kind": revision_kind,
        "base_revision_number": base_revision_number,
        "effective_identity": deepcopy(dict(effective_identity)),
        "changed_paths": list(changed_paths),
        "change_diff": [deepcopy(dict(row)) for row in change_diff],
        "evidence_summary": evidence_summary,
        "source_scope_kinds": list(source_scope_kinds),
        "evidence_refs": [deepcopy(dict(row)) for row in evidence_refs],
        "promotion_run_id": promotion_run_id,
        "promotion_correlation_id": promotion_correlation_id,
        "proposal_confidence": proposal_confidence,
        "review_confidence": review_confidence,
        "created_at": created_at,
    }
    return _validate_revision_document(document)


def _build_operator_run_document(
    *,
    run_id: str,
    character_id: str,
    base_revision_number: int,
    correlation_id: str,
    promoted_revision_number: int,
    created_at: str,
) -> dict[str, object]:
    """Build the sanitized run linked to one operator reset."""

    document: dict[str, object] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "run_kind": "operator_reset",
        "character_id": character_id,
        "base_revision_number": base_revision_number,
        "correlation_id": correlation_id,
        "root_episode_ids": [],
        "source_evidence_count": 0,
        "attempt_count_by_stage": {"proposal": 0, "review": 0},
        "lifecycle_state": "post_commit_pending",
        "disposition": "revision_promoted",
        "proposal_reason_code": "revision_promoted",
        "review_reason_code": "revision_promoted",
        "policy_reason_code": "revision_promoted",
        "persistence_reason_code": "revision_promoted",
        "candidate_id": None,
        "promoted_revision_number": promoted_revision_number,
        "validation_error_codes": [],
        "first_consumption": None,
        "post_commit_attempt_count": 0,
        "started_at": created_at,
        "completed_at": created_at,
    }
    return _validate_run_document(document)


def _validate_revision_document(
    raw_document: Mapping[str, object],
) -> dict[str, object]:
    """Validate one complete persisted identity revision."""

    document = _without_mongo_id(raw_document)
    _require_exact_keys(
        document,
        expected=_REVISION_KEYS,
        context="identity revision",
    )
    if document["schema_version"] != REVISION_SCHEMA_VERSION:
        raise IdentityLedgerCorruptionError(
            "identity revision schema_version is unsupported"
        )
    revision_id = _require_identifier(
        document["revision_id"],
        context="revision_id",
    )
    character_id = _require_identifier(
        document["character_id"],
        context="character_id",
    )
    revision_number = _require_integer(
        document["revision_number"],
        context="revision_number",
        minimum=0,
    )
    revision_kind = _require_enum(
        document["revision_kind"],
        context="revision_kind",
        allowed=_REVISION_KINDS,
    )
    base_revision_number = _require_optional_integer(
        document["base_revision_number"],
        context="base_revision_number",
        minimum=0,
    )
    identity = _require_mapping(
        document["effective_identity"],
        context="effective_identity",
    )
    validated_identity = validate_effective_identity(identity)
    changed_paths = _require_sorted_unique_strings(
        document["changed_paths"],
        context="changed_paths",
        allowed=models.ALLOWED_IDENTITY_PATHS,
        max_items=len(models.ALLOWED_IDENTITY_PATHS),
    )
    change_diff = _validate_change_diff(
        document["change_diff"],
        changed_paths=changed_paths,
    )
    evidence_summary = _require_text(
        document["evidence_summary"],
        context="evidence_summary",
        max_chars=2400,
    )
    source_scope_kinds = _require_sorted_unique_strings(
        document["source_scope_kinds"],
        context="source_scope_kinds",
        allowed=_REVISION_SCOPE_KINDS,
        max_items=len(_REVISION_SCOPE_KINDS),
    )
    evidence_refs = dedupe_evidence_refs(
        _require_mapping_sequence(
            document["evidence_refs"],
            context="evidence_refs",
            max_items=64,
        )
    )
    promotion_run_id = _require_optional_identifier(
        document["promotion_run_id"],
        context="promotion_run_id",
    )
    promotion_correlation_id = _require_optional_identifier(
        document["promotion_correlation_id"],
        context="promotion_correlation_id",
    )
    proposal_confidence = _require_enum(
        document["proposal_confidence"],
        context="proposal_confidence",
        allowed=_REVISION_CONFIDENCE_VALUES,
    )
    review_confidence = _require_enum(
        document["review_confidence"],
        context="review_confidence",
        allowed=_REVISION_CONFIDENCE_VALUES,
    )
    created_at = _require_iso_datetime(
        document["created_at"],
        context="created_at",
    )

    if revision_number == 0:
        if revision_kind != "seed" or base_revision_number is not None:
            raise IdentityLedgerCorruptionError(
                "revision zero must be a seed without a base"
            )
        if changed_paths or change_diff or evidence_refs:
            raise IdentityLedgerCorruptionError(
                "revision zero cannot contain changes or evidence"
            )
        if source_scope_kinds:
            raise IdentityLedgerCorruptionError(
                "revision zero cannot contain source scopes"
            )
        if promotion_run_id is not None or promotion_correlation_id is not None:
            raise IdentityLedgerCorruptionError(
                "revision zero cannot name a promotion run"
            )
        if proposal_confidence != "seed" or review_confidence != "seed":
            raise IdentityLedgerCorruptionError(
                "revision zero requires seed confidence"
            )
    else:
        if base_revision_number != revision_number - 1:
            raise IdentityLedgerCorruptionError(
                "identity revision base must be the prior revision number"
            )
        if promotion_run_id is None or promotion_correlation_id is None:
            raise IdentityLedgerCorruptionError(
                "non-seed identity revision requires promotion lineage"
            )
        if revision_kind == "operator_reset":
            if evidence_refs:
                raise IdentityLedgerCorruptionError(
                    "operator reset cannot contain evidence refs"
                )
            if source_scope_kinds != ["operator"]:
                raise IdentityLedgerCorruptionError(
                    "operator reset requires only operator scope"
                )
            if (
                proposal_confidence != "operator"
                or review_confidence != "operator"
            ):
                raise IdentityLedgerCorruptionError(
                    "operator reset requires operator confidence"
                )
        elif (
            proposal_confidence != "high"
            or review_confidence != "high"
        ):
            raise IdentityLedgerCorruptionError(
                "growth revision requires high proposal and review confidence"
            )

    validated: dict[str, object] = {
        "schema_version": REVISION_SCHEMA_VERSION,
        "revision_id": revision_id,
        "character_id": character_id,
        "revision_number": revision_number,
        "revision_kind": revision_kind,
        "base_revision_number": base_revision_number,
        "effective_identity": validated_identity,
        "changed_paths": changed_paths,
        "change_diff": change_diff,
        "evidence_summary": evidence_summary,
        "source_scope_kinds": source_scope_kinds,
        "evidence_refs": evidence_refs,
        "promotion_run_id": promotion_run_id,
        "promotion_correlation_id": promotion_correlation_id,
        "proposal_confidence": proposal_confidence,
        "review_confidence": review_confidence,
        "created_at": created_at,
    }
    return validated


def _validate_candidate_document(
    raw_document: Mapping[str, object],
) -> dict[str, object]:
    """Validate one persisted growth candidate."""

    document = _without_mongo_id(raw_document)
    _require_exact_keys(
        document,
        expected=_CANDIDATE_KEYS,
        context="identity candidate",
    )
    if document["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("identity candidate schema_version is unsupported")
    candidate_id = _require_identifier(
        document["candidate_id"],
        context="candidate_id",
    )
    character_id = _require_identifier(
        document["character_id"],
        context="character_id",
    )
    base_revision_number = _require_integer(
        document["base_revision_number"],
        context="base_revision_number",
        minimum=0,
    )
    status = _require_enum(
        document["status"],
        context="status",
        allowed=frozenset(models.CANDIDATE_TRANSITIONS),
    )
    change_kind = _require_enum(
        document["change_kind"],
        context="change_kind",
        allowed=_CANDIDATE_CHANGE_KINDS,
    )
    proposed_change_mappings = _require_mapping_sequence(
        document["proposed_changes"],
        context="proposed_changes",
        min_items=1,
        max_items=5,
    )
    proposed_changes = [
        validate_identity_patch(change)
        for change in proposed_change_mappings
    ]
    proposed_paths = [change["path"] for change in proposed_changes]
    if len(proposed_paths) != len(set(proposed_paths)):
        raise ValueError("candidate proposed_changes contain duplicate paths")
    semantic_summary = _require_text(
        document["semantic_summary"],
        context="semantic_summary",
        max_chars=2400,
    )
    evidence_refs = dedupe_evidence_refs(
        _require_mapping_sequence(
            document["evidence_refs"],
            context="evidence_refs",
            min_items=1,
            max_items=64,
        )
    )
    counts = evidence_counts(evidence_refs)
    distinct_episode_count = _require_integer(
        document["distinct_episode_count"],
        context="distinct_episode_count",
        minimum=1,
        maximum=64,
    )
    if distinct_episode_count != counts["distinct_episode_count"]:
        raise ValueError(
            "candidate distinct_episode_count does not match evidence roots"
        )
    distinct_local_dates = _require_sorted_unique_strings(
        document["distinct_local_dates"],
        context="distinct_local_dates",
        max_items=64,
    )
    if distinct_local_dates != counts["distinct_local_dates"]:
        raise ValueError(
            "candidate distinct_local_dates do not match evidence roots"
        )
    source_scope_kinds = _require_sorted_unique_strings(
        document["source_scope_kinds"],
        context="source_scope_kinds",
        allowed=models.EVIDENCE_SCOPE_KINDS,
        max_items=len(models.EVIDENCE_SCOPE_KINDS),
    )
    expected_scopes = sorted({
        row["scope_kind"]
        for row in evidence_refs
    })
    if source_scope_kinds != expected_scopes:
        raise ValueError(
            "candidate source_scope_kinds do not match evidence refs"
        )
    claimed_roots = _require_sorted_unique_strings(
        document["claimed_root_episode_ids"],
        context="claimed_root_episode_ids",
        max_items=64,
    )
    expected_roots = sorted({
        row["root_episode_id"]
        for row in evidence_refs
    })
    if claimed_roots != expected_roots:
        raise ValueError(
            "candidate claimed roots do not match evidence refs"
        )
    newest_root_captured_at = _require_iso_datetime(
        document["newest_root_captured_at"],
        context="newest_root_captured_at",
    )
    if newest_root_captured_at != max(
        row["captured_at"]
        for row in evidence_refs
    ):
        raise ValueError(
            "candidate newest_root_captured_at does not match evidence"
        )
    reversal_of_paths = _require_sorted_unique_strings(
        document["reversal_of_paths"],
        context="reversal_of_paths",
        allowed=models.ALLOWED_IDENTITY_PATHS,
        max_items=5,
    )
    fresh_post_revision_root_count = _require_integer(
        document["fresh_post_revision_root_count"],
        context="fresh_post_revision_root_count",
        minimum=0,
        maximum=64,
    )
    character_authorship = _require_enum(
        document["character_authorship"],
        context="character_authorship",
        allowed=_AUTHORSHIP_VALUES,
    )
    proposal_confidence = _require_enum(
        document["proposal_confidence"],
        context="proposal_confidence",
        allowed=_CONFIDENCE_VALUES,
    )
    review_confidence = _require_enum(
        document["review_confidence"],
        context="review_confidence",
        allowed=_CONFIDENCE_VALUES,
    )
    privacy_review = _require_enum(
        document["privacy_review"],
        context="privacy_review",
        allowed=_PRIVACY_REVIEW_VALUES,
    )
    promoted_revision_number = _require_optional_integer(
        document["promoted_revision_number"],
        context="promoted_revision_number",
        minimum=1,
    )
    rejection_reason = _require_optional_reason_code(
        document["rejection_reason"],
        context="rejection_reason",
    )
    created_at = _require_iso_datetime(
        document["created_at"],
        context="created_at",
    )
    updated_at = _require_iso_datetime(
        document["updated_at"],
        context="updated_at",
    )
    if status == "promoted" and promoted_revision_number is None:
        raise ValueError("promoted candidate requires promoted revision")
    if status != "promoted" and promoted_revision_number is not None:
        raise ValueError(
            "unpromoted candidate cannot name a promoted revision"
        )

    validated: dict[str, object] = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "character_id": character_id,
        "base_revision_number": base_revision_number,
        "status": status,
        "change_kind": change_kind,
        "proposed_changes": proposed_changes,
        "semantic_summary": semantic_summary,
        "evidence_refs": evidence_refs,
        "distinct_episode_count": distinct_episode_count,
        "distinct_local_dates": distinct_local_dates,
        "source_scope_kinds": source_scope_kinds,
        "claimed_root_episode_ids": claimed_roots,
        "newest_root_captured_at": newest_root_captured_at,
        "reversal_of_paths": reversal_of_paths,
        "fresh_post_revision_root_count": fresh_post_revision_root_count,
        "character_authorship": character_authorship,
        "proposal_confidence": proposal_confidence,
        "review_confidence": review_confidence,
        "privacy_review": privacy_review,
        "promoted_revision_number": promoted_revision_number,
        "rejection_reason": rejection_reason,
        "created_at": created_at,
        "updated_at": updated_at,
    }
    return validated


def _validate_run_document(
    raw_document: Mapping[str, object],
) -> dict[str, object]:
    """Validate one sanitized growth-run row."""

    document = _without_mongo_id(raw_document)
    _require_exact_keys(
        document,
        expected=_RUN_KEYS,
        context="identity growth run",
    )
    if document["schema_version"] != RUN_SCHEMA_VERSION:
        raise ValueError("identity growth run schema_version is unsupported")
    run_id = _require_identifier(document["run_id"], context="run_id")
    run_kind = _require_enum(
        document["run_kind"],
        context="run_kind",
        allowed=_RUN_KINDS,
    )
    character_id = _require_identifier(
        document["character_id"],
        context="character_id",
    )
    base_revision_number = _require_integer(
        document["base_revision_number"],
        context="base_revision_number",
        minimum=0,
    )
    correlation_id = _require_identifier(
        document["correlation_id"],
        context="correlation_id",
    )
    root_episode_ids = _require_sorted_unique_strings(
        document["root_episode_ids"],
        context="root_episode_ids",
        max_items=64,
    )
    source_evidence_count = _require_integer(
        document["source_evidence_count"],
        context="source_evidence_count",
        minimum=0,
        maximum=64,
    )
    attempt_mapping = _require_mapping(
        document["attempt_count_by_stage"],
        context="attempt_count_by_stage",
    )
    _require_exact_keys(
        attempt_mapping,
        expected=_ATTEMPT_STAGE_KEYS,
        context="attempt_count_by_stage",
    )
    attempt_count_by_stage = {
        key: _require_integer(
            attempt_mapping[key],
            context=f"attempt_count_by_stage.{key}",
            minimum=0,
            maximum=3,
        )
        for key in sorted(_ATTEMPT_STAGE_KEYS)
    }
    lifecycle_state = _require_enum(
        document["lifecycle_state"],
        context="lifecycle_state",
        allowed=models.RUN_LIFECYCLE_STATES,
    )
    disposition = _require_enum(
        document["disposition"],
        context="disposition",
        allowed=_RUN_DISPOSITIONS,
    )
    proposal_reason_code = _require_reason_code(
        document["proposal_reason_code"],
        context="proposal_reason_code",
    )
    review_reason_code = _require_reason_code(
        document["review_reason_code"],
        context="review_reason_code",
    )
    policy_reason_code = _require_reason_code(
        document["policy_reason_code"],
        context="policy_reason_code",
    )
    persistence_reason_code = _require_reason_code(
        document["persistence_reason_code"],
        context="persistence_reason_code",
    )
    candidate_id = _require_optional_identifier(
        document["candidate_id"],
        context="candidate_id",
    )
    promoted_revision_number = _require_optional_integer(
        document["promoted_revision_number"],
        context="promoted_revision_number",
        minimum=1,
    )
    validation_error_codes = _require_sorted_unique_strings(
        document["validation_error_codes"],
        context="validation_error_codes",
        max_items=32,
    )
    first_consumption = document["first_consumption"]
    if first_consumption is not None:
        mapping = _require_mapping(
            first_consumption,
            context="first_consumption",
        )
        first_consumption = _validate_first_consumption(mapping)
    post_commit_attempt_count = _require_integer(
        document["post_commit_attempt_count"],
        context="post_commit_attempt_count",
        minimum=0,
        maximum=1000,
    )
    started_at = _require_iso_datetime(
        document["started_at"],
        context="started_at",
    )
    completed_at = _require_optional_iso_datetime(
        document["completed_at"],
        context="completed_at",
    )
    if run_kind == "operator_reset":
        if root_episode_ids or source_evidence_count != 0:
            raise ValueError(
                "operator reset run cannot contain evidence roots"
            )
        if candidate_id is not None:
            raise ValueError("operator reset run cannot name a candidate")
    if disposition == "revision_promoted":
        if promoted_revision_number is None:
            raise ValueError(
                "promoted identity run must name its revision number"
            )
        if lifecycle_state not in {"post_commit_pending", "complete"}:
            raise ValueError(
                "promoted identity run lifecycle state is invalid"
            )
    if lifecycle_state == "post_commit_pending" and first_consumption:
        raise ValueError(
            "post-commit-pending identity run cannot be consumed"
        )
    if first_consumption is not None:
        if lifecycle_state != "complete":
            raise ValueError(
                "identity consumption requires a complete growth run"
            )
        if (
            first_consumption["status"] == "consumed"
            and first_consumption["loaded_revision_number"]
            != promoted_revision_number
        ):
            raise ValueError(
                "consumed identity revision number must match its run"
            )

    validated: dict[str, object] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "run_kind": run_kind,
        "character_id": character_id,
        "base_revision_number": base_revision_number,
        "correlation_id": correlation_id,
        "root_episode_ids": root_episode_ids,
        "source_evidence_count": source_evidence_count,
        "attempt_count_by_stage": attempt_count_by_stage,
        "lifecycle_state": lifecycle_state,
        "disposition": disposition,
        "proposal_reason_code": proposal_reason_code,
        "review_reason_code": review_reason_code,
        "policy_reason_code": policy_reason_code,
        "persistence_reason_code": persistence_reason_code,
        "candidate_id": candidate_id,
        "promoted_revision_number": promoted_revision_number,
        "validation_error_codes": validation_error_codes,
        "first_consumption": first_consumption,
        "post_commit_attempt_count": post_commit_attempt_count,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    return validated


def _validate_first_consumption(
    raw_receipt: Mapping[str, object],
) -> models.IdentityRevisionConsumptionV1:
    """Validate the exact sanitized first-consumption receipt."""

    receipt = _require_mapping(
        raw_receipt,
        context="first_consumption",
    )
    _require_exact_keys(
        receipt,
        expected=_FIRST_CONSUMPTION_KEYS,
        context="first_consumption",
    )
    episode_id = _require_identifier(
        receipt["episode_id"],
        context="first_consumption.episode_id",
    )
    correlation_id = _require_identifier(
        receipt["correlation_id"],
        context="first_consumption.correlation_id",
    )
    claimed_at = _require_iso_datetime(
        receipt["claimed_at"],
        context="first_consumption.claimed_at",
    )
    loaded_revision_number = _require_integer(
        receipt["loaded_revision_number"],
        context="first_consumption.loaded_revision_number",
        minimum=0,
    )
    consumer_kinds = _require_sorted_unique_strings(
        receipt["consumer_kinds"],
        context="first_consumption.consumer_kinds",
        allowed=models.IDENTITY_CONSUMER_KINDS,
        max_items=len(models.IDENTITY_CONSUMER_KINDS),
    )
    if not consumer_kinds:
        raise ValueError(
            "first_consumption.consumer_kinds must be nonempty"
        )
    projection_digest = _require_text(
        receipt["projection_digest"],
        context="first_consumption.projection_digest",
        max_chars=64,
    )
    if (
        len(projection_digest) != 64
        or projection_digest != projection_digest.lower()
        or any(
            character not in "0123456789abcdef"
            for character in projection_digest
        )
    ):
        raise ValueError(
            "first_consumption.projection_digest must be lowercase SHA-256"
        )
    status = _require_enum(
        receipt["status"],
        context="first_consumption.status",
        allowed=frozenset({"consumed", "mismatch"}),
    )
    return {
        "episode_id": episode_id,
        "correlation_id": correlation_id,
        "claimed_at": claimed_at,
        "loaded_revision_number": loaded_revision_number,
        "consumer_kinds": consumer_kinds,
        "projection_digest": projection_digest,
        "status": status,
    }


def _validate_change_diff(
    value: object,
    *,
    changed_paths: Sequence[str],
) -> list[dict[str, object]]:
    """Validate exact diff rows and their path list."""

    rows = _require_mapping_sequence(
        value,
        context="change_diff",
        max_items=len(models.ALLOWED_IDENTITY_PATHS),
    )
    validated: list[dict[str, object]] = []
    for row in rows:
        _require_exact_keys(
            row,
            expected=frozenset({"path", "value_kind", "before", "after"}),
            context="identity change diff",
        )
        path = _require_enum(
            row["path"],
            context="change_diff.path",
            allowed=models.ALLOWED_IDENTITY_PATHS,
        )
        value_kind = _require_enum(
            row["value_kind"],
            context="change_diff.value_kind",
            allowed=frozenset({
                "text",
                "integer",
                "semantic_band",
                "closed_enum",
                "text_list",
            }),
        )
        if row["before"] == row["after"]:
            raise IdentityLedgerCorruptionError(
                f"identity diff is a no-op: {path}"
            )
        validated.append({
            "path": path,
            "value_kind": value_kind,
            "before": deepcopy(row["before"]),
            "after": deepcopy(row["after"]),
        })
    if [row["path"] for row in validated] != list(changed_paths):
        raise IdentityLedgerCorruptionError(
            "identity changed_paths do not match change_diff"
        )
    return validated


def _health_latest_reason_code(
    *,
    latest_revision_number: int,
    latest_run: Mapping[str, object] | None,
    receipt_run: Mapping[str, object] | None,
    receipt_status: object,
) -> str:
    """Resolve the latest public reason without exposing run identities."""

    if receipt_status == "mismatch":
        return "revision_consumption_mismatch"
    pipeline_reasons = {
        "proposal_contract_failed",
        "review_contract_failed",
        "promotion_write_failed",
    }
    if latest_run is not None:
        persistence_reason = str(latest_run["persistence_reason_code"])
        if (
            latest_run["lifecycle_state"] == "failed"
            or persistence_reason in pipeline_reasons
        ):
            return persistence_reason
    if latest_revision_number > 0 and receipt_status is None:
        return "awaiting_first_consumption"
    if latest_run is not None:
        if (
            receipt_status == "consumed"
            and receipt_run is not None
            and latest_run["run_id"] == receipt_run["run_id"]
        ):
            return "revision_consumed"
        return str(latest_run["persistence_reason_code"])
    if receipt_status == "consumed":
        return "revision_consumed"
    return "not_routed"


def _require_matching_seed(
    revision: Mapping[str, object],
    identity: Mapping[str, object],
) -> None:
    """Require persisted revision zero to match the selected profile."""

    if revision["effective_identity"] != identity:
        raise SeedIdentityConflictError(
            "selected profile conflicts with immutable revision zero"
        )


def _without_mongo_id(
    raw_document: Mapping[str, object],
) -> dict[str, object]:
    """Copy a MongoDB document without its storage-only object ID."""

    document = deepcopy(dict(raw_document))
    document.pop("_id", None)
    return document


def _require_exact_keys(
    payload: Mapping[str, object],
    *,
    expected: frozenset[str],
    context: str,
) -> None:
    """Require one mapping to contain exactly the declared keys."""

    actual = frozenset(payload)
    missing = sorted(expected.difference(actual))
    unknown = sorted(actual.difference(expected))
    if missing:
        raise ValueError(f"{context} missing required keys: {missing}")
    if unknown:
        raise ValueError(f"{context} contains unknown keys: {unknown}")


def _require_mapping(
    value: object,
    *,
    context: str,
) -> Mapping[str, object]:
    """Require one string-keyed mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} keys must be strings")
    return value


def _require_mapping_sequence(
    value: object,
    *,
    context: str,
    min_items: int = 0,
    max_items: int,
) -> list[Mapping[str, object]]:
    """Require a bounded list of mappings."""

    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    if not min_items <= len(value) <= max_items:
        raise ValueError(
            f"{context} must contain {min_items}..{max_items} items"
        )
    return [
        _require_mapping(item, context=f"{context}[{index}]")
        for index, item in enumerate(value)
    ]


def _require_identifier(value: object, *, context: str) -> str:
    """Require one bounded nonempty opaque identifier."""

    return _require_text(value, context=context, max_chars=500)


def _require_optional_identifier(
    value: object,
    *,
    context: str,
) -> str | None:
    """Require an optional opaque identifier."""

    if value is None:
        return None
    return _require_identifier(value, context=context)


def _require_text(
    value: object,
    *,
    context: str,
    max_chars: int,
) -> str:
    """Require bounded nonempty trimmed text."""

    if not isinstance(value, str):
        raise ValueError(f"{context} must be text")
    text = value.strip()
    if not text:
        raise ValueError(f"{context} must be nonempty")
    if len(text) > max_chars:
        raise ValueError(f"{context} exceeds maximum length {max_chars}")
    return text


def _require_enum(
    value: object,
    *,
    context: str,
    allowed: frozenset[str],
) -> str:
    """Require one closed string enum."""

    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{context} must be one of {sorted(allowed)}")
    return value


def _require_reason_code(value: object, *, context: str) -> str:
    """Require one closed identity reason code."""

    return _require_enum(
        value,
        context=context,
        allowed=models.IDENTITY_GROWTH_REASON_CODES,
    )


def _require_optional_reason_code(
    value: object,
    *,
    context: str,
) -> str | None:
    """Require an optional closed identity reason code."""

    if value is None:
        return None
    return _require_reason_code(value, context=context)


def _require_integer(
    value: object,
    *,
    context: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Require one bounded non-boolean integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    if value < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{context} must be at most {maximum}")
    return value


def _require_optional_integer(
    value: object,
    *,
    context: str,
    minimum: int,
) -> int | None:
    """Require an optional bounded integer."""

    if value is None:
        return None
    return _require_integer(value, context=context, minimum=minimum)


def _require_sorted_unique_strings(
    value: object,
    *,
    context: str,
    max_items: int,
    allowed: frozenset[str] | None = None,
) -> list[str]:
    """Require a bounded sorted list of unique nonempty strings."""

    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    if len(value) > max_items:
        raise ValueError(f"{context} exceeds maximum item count {max_items}")
    strings = [
        _require_text(
            item,
            context=f"{context}[{index}]",
            max_chars=500,
        )
        for index, item in enumerate(value)
    ]
    if strings != sorted(set(strings)):
        raise ValueError(f"{context} must be sorted and unique")
    if allowed is not None:
        unsupported = sorted(set(strings).difference(allowed))
        if unsupported:
            raise ValueError(
                f"{context} contains unsupported values: {unsupported}"
            )
    return strings


def _require_iso_datetime(value: object, *, context: str) -> str:
    """Require one timezone-aware ISO datetime string."""

    text = _require_text(value, context=context, max_chars=80)
    try:
        parse_storage_utc_datetime(text)
    except ValueError as exc:
        raise ValueError(
            f"{context} must be a storage UTC datetime"
        ) from exc
    return text


def _require_optional_iso_datetime(
    value: object,
    *,
    context: str,
) -> str | None:
    """Require an optional timezone-aware ISO datetime."""

    if value is None:
        return None
    return _require_iso_datetime(value, context=context)


def _utc_now_iso() -> str:
    """Return the current UTC storage timestamp."""

    return _datetime_to_iso(None)


def _datetime_to_iso(value: datetime | None) -> str:
    """Normalize an optional aware datetime to UTC storage text."""

    moment = value or storage_utc_now()
    if moment.tzinfo is None:
        raise ValueError("identity persistence time must include a timezone")
    utc_moment = moment.astimezone(timezone.utc)
    return utc_moment.isoformat().replace("+00:00", "Z")


def _transaction_is_unavailable(exc: PyMongoError) -> bool:
    """Return whether MongoDB rejected transaction capability."""

    if not isinstance(exc, OperationFailure):
        return False
    unavailable_codes = {20, 263, 303, 40573}
    if exc.code in unavailable_codes:
        return True
    message = str(exc).lower()
    unavailable_fragments = (
        "transaction numbers are only allowed",
        "transactions are not supported",
        "does not support retryable writes",
    )
    return any(fragment in message for fragment in unavailable_fragments)


def _transaction_is_retryable(exc: PyMongoError) -> bool:
    """Return whether MongoDB labels a transaction failure retryable."""

    return (
        exc.has_error_label("TransientTransactionError")
        or exc.has_error_label("UnknownTransactionCommitResult")
    )


__all__ = [
    "CANDIDATES_COLLECTION",
    "GROWTH_COLLECTION_NAMES",
    "IDENTITY_INDEX_NAMES",
    "REVISIONS_COLLECTION",
    "RUNS_COLLECTION",
    "CharacterIdentityPersistenceError",
    "ConcurrentIdentityPromotionError",
    "IdentityCandidateConflictError",
    "IdentityLedgerCorruptionError",
    "IdentityLedgerNotFoundError",
    "IdentityPostCommitPendingError",
    "IdentityRevisionStaleError",
    "IdentityRootAlreadyClaimedError",
    "IdentityRunConflictError",
    "IdentityTransactionUnavailableError",
    "SeedIdentityConflictError",
    "claim_identity_revision_consumption",
    "build_identity_growth_health",
    "complete_growth_run_post_commit",
    "create_operator_reset_revision",
    "ensure_character_identity_growth_indexes",
    "ensure_seed_identity",
    "count_inferred_identity_promotions_on_local_date",
    "get_current_identity",
    "get_growth_run",
    "insert_growth_candidate",
    "insert_growth_run",
    "list_current_growth_candidates",
    "list_identity_growth_candidates",
    "list_identity_revisions",
    "list_post_commit_pending_growth_runs",
    "list_recent_identity_growth_runs",
    "promote_ready_candidate",
    "record_growth_run_post_commit_failure",
    "record_identity_revision_consumption_mismatch",
    "reject_growth_candidates",
    "update_growth_candidate",
]
