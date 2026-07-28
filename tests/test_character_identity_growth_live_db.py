"""Guarded live-Mongo proof for the character identity ledger."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import os
from uuid import uuid4

import pytest

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.projection import (
    identity_projection_digest,
    project_identity_for_cognition,
    project_identity_for_surface,
    projected_identity_consumer_kinds,
)
from kazusa_ai_chatbot.db.character import (
    LegacyCharacterStateError,
    ensure_operational_character_state,
)
from kazusa_ai_chatbot.db.character_identity_growth import (
    CANDIDATES_COLLECTION,
    GROWTH_COLLECTION_NAMES,
    IDENTITY_INDEX_NAMES,
    REVISIONS_COLLECTION,
    RUNS_COLLECTION,
    ConcurrentIdentityPromotionError,
    IdentityRevisionStaleError,
    IdentityRootAlreadyClaimedError,
    build_identity_growth_health,
    claim_identity_revision_consumption,
    complete_growth_run_post_commit,
    create_operator_reset_revision,
    ensure_character_identity_growth_indexes,
    ensure_seed_identity,
    get_current_identity,
    insert_growth_candidate,
    insert_growth_run,
    list_identity_revisions,
    list_post_commit_pending_growth_runs,
    promote_ready_candidate,
    record_identity_revision_consumption_mismatch,
    update_growth_candidate,
)
from kazusa_ai_chatbot.db._client import (
    IDENTITY_GROWTH_DATABASE_GUARD_ENV,
    IDENTITY_GROWTH_TEST_DATABASE_ENV,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_db]

_CHARACTER_ID = "character-live-db"


def _identity(name: str = "Ledger Character") -> dict[str, object]:
    """Build one complete generic identity for persistence proof."""

    return {
        "name": name,
        "description": "A grounded character whose identity can change.",
        "gender": "unspecified",
        "age": 30,
        "birthday": "March 3",
        "backstory": "They learned to distinguish choice from compliance.",
        "personality_brief": {
            "mbti": "ISTP",
            "logic": "Evidence-led and practical.",
            "tempo": "Brief, measured, and responsive.",
            "defense": "Withdraws briefly before reassessing.",
            "quirks": "Checks assumptions aloud.",
            "taboos": "Rejects imposed self-definitions.",
        },
        "boundary_profile": {
            "self_integrity": 0.7,
            "control_sensitivity": 0.7,
            "compliance_strategy": "resist",
            "relational_override": 0.3,
            "control_intimacy_misread": 0.3,
            "boundary_recovery": "rebound",
            "authority_skepticism": 0.6,
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.2,
            "hesitation_density": 0.2,
            "counter_questioning": 0.4,
            "softener_density": 0.3,
            "formalism_avoidance": 0.7,
            "abstraction_reframing": 0.5,
            "direct_assertion": 0.7,
            "emotional_leakage": 0.3,
            "rhythmic_bounce": 0.4,
            "self_deprecation": 0.1,
        },
        "self_image": {
            "self_concept": (
                "I can revise my own judgment without surrendering agency."
            ),
            "current_growth_edges": [
                "Let sustained evidence soften automatic withdrawal.",
            ],
        },
        "visual_characterization": (
            "An alert adult with practical layers and an open stance."
        ),
    }


def _evidence_ref(root_number: int) -> dict[str, object]:
    """Build one repository-shaped root reference."""

    day = 1 if root_number < 3 else 2
    return {
        "schema_version": models.IDENTITY_EVIDENCE_SCHEMA_VERSION,
        "evidence_ref_id": f"evidence-{root_number}",
        "root_episode_id": f"episode-{root_number}",
        "correlation_id": f"correlation-{root_number}",
        "source_kind": "settled_episode",
        "derived_reflection_run_ids": [],
        "character_local_date": f"2026-07-{day:02d}",
        "scope_kind": "private",
        "captured_at": f"2026-07-{day:02d}T10:00:00+00:00",
    }


def _candidate(
    *,
    candidate_id: str,
    root_numbers: tuple[int, ...],
    replacement: str,
) -> dict[str, object]:
    """Build one reviewed ready candidate."""

    evidence_refs = [_evidence_ref(number) for number in root_numbers]
    roots = [row["root_episode_id"] for row in evidence_refs]
    local_dates = sorted({
        row["character_local_date"]
        for row in evidence_refs
    })
    return {
        "schema_version": "character_identity_growth_candidate.v1",
        "candidate_id": candidate_id,
        "character_id": _CHARACTER_ID,
        "base_revision_number": 0,
        "status": "ready",
        "change_kind": "inferred_growth",
        "proposed_changes": [{
            "path": "self_image.self_concept",
            "value_kind": "text",
            "replacement_text": replacement,
        }],
        "semantic_summary": "Sustained self-authored change.",
        "evidence_refs": evidence_refs,
        "distinct_episode_count": len(roots),
        "distinct_local_dates": local_dates,
        "source_scope_kinds": ["private"],
        "claimed_root_episode_ids": roots,
        "newest_root_captured_at": evidence_refs[-1]["captured_at"],
        "reversal_of_paths": [],
        "fresh_post_revision_root_count": 0,
        "character_authorship": "inferred",
        "proposal_confidence": "high",
        "review_confidence": "high",
        "privacy_review": "low",
        "promoted_revision_number": None,
        "rejection_reason": None,
        "created_at": "2026-07-02T10:00:00+00:00",
        "updated_at": "2026-07-02T10:00:00+00:00",
    }


def _run(run_id: str, candidate_id: str) -> dict[str, object]:
    """Build one sanitized ready-promotion run."""

    return {
        "schema_version": "character_identity_growth_run.v1",
        "run_id": run_id,
        "run_kind": "episode",
        "character_id": _CHARACTER_ID,
        "base_revision_number": 0,
        "correlation_id": f"correlation-{run_id}",
        "root_episode_ids": [],
        "source_evidence_count": 3,
        "attempt_count_by_stage": {"proposal": 1, "review": 1},
        "lifecycle_state": "in_progress",
        "disposition": "candidate_updated",
        "proposal_reason_code": "candidate_ready",
        "review_reason_code": "candidate_ready",
        "policy_reason_code": "candidate_ready",
        "persistence_reason_code": "candidate_ready",
        "candidate_id": candidate_id,
        "promoted_revision_number": None,
        "validation_error_codes": [],
        "first_consumption": None,
        "post_commit_attempt_count": 0,
        "started_at": "2026-07-02T10:00:00+00:00",
        "completed_at": None,
    }


@pytest.fixture
async def identity_database():
    """Yield an isolated guarded database with only test-owned cleanup."""

    if os.environ.get("MONGODB_URI") is None:
        pytest.skip("guarded live-DB proof requires process MONGODB_URI")
    if os.environ.get("KAZUSA_TEST_DB_GUARD") != "1":
        pytest.skip("identity live-DB proof requires KAZUSA_TEST_DB_GUARD=1")
    if os.environ.get(IDENTITY_GROWTH_DATABASE_GUARD_ENV) != "1":
        pytest.skip(
            "identity live-DB proof requires its dedicated database guard"
        )
    test_database_name = os.environ.get(
        IDENTITY_GROWTH_TEST_DATABASE_ENV,
        "",
    ).strip()
    if not test_database_name:
        pytest.skip("identity live-DB proof requires an explicit database")
    if os.environ.get("MONGODB_DB_NAME") != test_database_name:
        pytest.skip("identity live-DB proof requires the exact guarded database")
    if os.environ.get("PYTEST_XDIST_WORKER"):
        raise AssertionError("identity live-DB proof requires one process")

    from kazusa_ai_chatbot.db._client import close_db, get_db

    database = await get_db()
    owned_collections = [
        *GROWTH_COLLECTION_NAMES,
        "character_state",
        "global_character_growth_traits",
        "global_character_growth_runs",
    ]
    for collection_name in owned_collections:
        await database.drop_collection(collection_name)
    await ensure_character_identity_growth_indexes()

    try:
        yield database
    finally:
        cleanup_database = await get_db()
        for collection_name in owned_collections:
            await cleanup_database.drop_collection(collection_name)
        await close_db()


async def test_indexes_and_clean_seed_have_exact_shape(
    identity_database,
) -> None:
    """Clean persistence creates declared indexes and immutable revision zero."""

    await ensure_operational_character_state()
    revision = await ensure_seed_identity(
        character_id=_CHARACTER_ID,
        seed=_identity(),
    )
    verified = await ensure_seed_identity(
        character_id=_CHARACTER_ID,
        seed=_identity(),
    )

    assert revision == verified
    assert revision["revision_number"] == 0
    assert revision["revision_kind"] == "seed"
    assert revision["base_revision_number"] is None
    assert revision["effective_identity"] == _identity()
    assert revision["change_diff"] == []
    assert revision["evidence_refs"] == []
    state = await identity_database.character_state.find_one({"_id": "global"})
    assert frozenset(state) == {"_id", "cognition_state", "updated_at"}

    actual_index_names: set[str] = set()
    for collection_name in GROWTH_COLLECTION_NAMES:
        collection = identity_database[collection_name]
        async for index in collection.list_indexes():
            actual_index_names.add(index["name"])
    assert IDENTITY_INDEX_NAMES.issubset(actual_index_names)


async def test_legacy_semantic_state_fails_without_creating_seed(
    identity_database,
) -> None:
    """Legacy semantic state must fail before a revision is inserted."""

    await identity_database.character_state.insert_one({
        "_id": "global",
        "name": "Legacy",
        "cognition_state": {"state_scope": "character"},
        "updated_at": "2026-07-01T00:00:00Z",
    })

    with pytest.raises(LegacyCharacterStateError, match="clean target"):
        await ensure_operational_character_state()

    assert (
        await identity_database[REVISIONS_COLLECTION].count_documents({})
        == 0
    )


async def test_max_reader_history_and_operator_reset_survive_restart(
    identity_database,
) -> None:
    """Operator force creates a full new revision while preserving seed."""

    await ensure_operational_character_state()
    seed = await ensure_seed_identity(
        character_id=_CHARACTER_ID,
        seed=_identity(),
    )
    replacement = _identity(name="Operator Chosen Name")
    reset = await create_operator_reset_revision(
        character_id=_CHARACTER_ID,
        identity=replacement,
        operator_action_id="operator-action-1",
        correlation_id="operator-correlation-1",
        now=datetime(2026, 7, 3, tzinfo=timezone.utc),
    )
    retry = await create_operator_reset_revision(
        character_id=_CHARACTER_ID,
        identity=replacement,
        operator_action_id="operator-action-1",
        correlation_id="operator-correlation-1",
        now=datetime(2026, 7, 3, tzinfo=timezone.utc),
    )

    assert reset == retry
    assert reset["revision_number"] == 1
    assert reset["revision_kind"] == "operator_reset"
    assert reset["proposal_confidence"] == "operator"
    assert reset["review_confidence"] == "operator"
    assert reset["evidence_refs"] == []
    assert seed["effective_identity"]["name"] == "Ledger Character"
    history = await list_identity_revisions(character_id=_CHARACTER_ID)
    assert [row["revision_number"] for row in history] == [1, 0]
    assert history[1] == seed

    from kazusa_ai_chatbot.db._client import close_db

    await close_db()
    latest_after_restart = await get_current_identity(
        character_id=_CHARACTER_ID,
    )
    assert latest_after_restart == reset


async def test_one_root_can_be_claimed_by_only_one_candidate(
    identity_database,
) -> None:
    """The multikey unique index enforces repository-root ownership."""

    await ensure_seed_identity(
        character_id=_CHARACTER_ID,
        seed=_identity(),
    )
    first = _candidate(
        candidate_id="candidate-one",
        root_numbers=(1, 2, 3),
        replacement="I increasingly choose openness after evidence.",
    )
    second = _candidate(
        candidate_id="candidate-two",
        root_numbers=(3, 4),
        replacement="I increasingly choose patient distance.",
    )

    await insert_growth_candidate(first)
    with pytest.raises(IdentityRootAlreadyClaimedError, match="episode-3"):
        await insert_growth_candidate(second)


async def test_concurrent_promotions_create_one_next_revision(
    identity_database,
) -> None:
    """Two ready candidates on one base may produce only one revision."""

    await ensure_seed_identity(
        character_id=_CHARACTER_ID,
        seed=_identity(),
    )
    first_candidate = _candidate(
        candidate_id="candidate-race-one",
        root_numbers=(1, 2, 3),
        replacement="I now let earned trust temper defensive distance.",
    )
    second_candidate = _candidate(
        candidate_id="candidate-race-two",
        root_numbers=(4, 5, 6),
        replacement="I now preserve more distance while deciding.",
    )
    await insert_growth_candidate(first_candidate)
    await insert_growth_candidate(second_candidate)
    await insert_growth_run(_run("run-race-one", "candidate-race-one"))
    await insert_growth_run(_run("run-race-two", "candidate-race-two"))

    results = await asyncio.gather(
        promote_ready_candidate(
            character_id=_CHARACTER_ID,
            candidate_id="candidate-race-one",
            run_id="run-race-one",
        ),
        promote_ready_candidate(
            character_id=_CHARACTER_ID,
            candidate_id="candidate-race-two",
            run_id="run-race-two",
        ),
        return_exceptions=True,
    )

    revisions = await list_identity_revisions(character_id=_CHARACTER_ID)
    successes = [result for result in results if isinstance(result, dict)]
    failures = [
        result
        for result in results
        if isinstance(result, ConcurrentIdentityPromotionError)
    ]
    assert len(successes) == 1
    assert len(failures) == 1
    assert [row["revision_number"] for row in revisions] == [1, 0]
    latest = await get_current_identity(character_id=_CHARACTER_ID)
    assert latest == successes[0]


async def test_clean_target_contains_no_legacy_growth_collections(
    identity_database,
) -> None:
    """The identity owner must create only its three growth collections."""

    names = set(await identity_database.list_collection_names())

    assert set(GROWTH_COLLECTION_NAMES).issubset(names)
    assert "global_character_growth_traits" not in names
    assert "global_character_growth_runs" not in names
    assert {
        REVISIONS_COLLECTION,
        CANDIDATES_COLLECTION,
        RUNS_COLLECTION,
    } == set(GROWTH_COLLECTION_NAMES)


async def test_prior_revision_document_remains_unchanged_after_promotion(
    identity_database,
) -> None:
    """Promotion inserts a full snapshot and never mutates revision zero."""

    seed = await ensure_seed_identity(
        character_id=_CHARACTER_ID,
        seed=_identity(),
    )
    seed_before = deepcopy(seed)
    candidate = _candidate(
        candidate_id="candidate-immutable",
        root_numbers=(1, 2, 3),
        replacement="I can remain open after trust is repeatedly earned.",
    )
    await insert_growth_candidate(candidate)
    await insert_growth_run(_run("run-immutable", "candidate-immutable"))

    promoted = await promote_ready_candidate(
        character_id=_CHARACTER_ID,
        candidate_id="candidate-immutable",
        run_id="run-immutable",
    )
    history = await list_identity_revisions(character_id=_CHARACTER_ID)

    assert promoted["revision_number"] == 1
    assert history[1] == seed_before
    assert (
        promoted["effective_identity"]["self_image"]["self_concept"]
        == "I can remain open after trust is repeatedly earned."
    )


async def test_first_consumption_is_latest_checked_and_durable(
    identity_database,
) -> None:
    """A complete promoted run should retain exactly one consumer receipt."""

    await ensure_seed_identity(
        character_id=_CHARACTER_ID,
        seed=_identity(),
    )
    revision = await create_operator_reset_revision(
        character_id=_CHARACTER_ID,
        identity=_identity(name="Latest Name"),
        operator_action_id="operator-consumption-1",
        correlation_id="operator-consumption-correlation-1",
    )
    pending = await list_post_commit_pending_growth_runs(
        character_id=_CHARACTER_ID,
    )
    assert [run["run_id"] for run in pending] == [
        "operator-reset:operator-consumption-1"
    ]
    completed_run = await complete_growth_run_post_commit(
        run_id="operator-reset:operator-consumption-1",
        character_id=_CHARACTER_ID,
        revision_number=1,
    )
    assert completed_run["lifecycle_state"] == "complete"
    assert completed_run["post_commit_attempt_count"] == 1

    cognition = project_identity_for_cognition(revision)
    surface = project_identity_for_surface(revision)
    consumers = projected_identity_consumer_kinds(cognition)
    digest = identity_projection_digest(
        revision_number=1,
        cognition_context=cognition,
        surface_context=surface,
    )
    receipts = await asyncio.gather(
        claim_identity_revision_consumption(
            character_id=_CHARACTER_ID,
            episode_id="episode-consumer-a",
            correlation_id="correlation-consumer-a",
            loaded_revision_number=1,
            consumer_kinds=consumers,
            projection_digest=digest,
        ),
        claim_identity_revision_consumption(
            character_id=_CHARACTER_ID,
            episode_id="episode-consumer-b",
            correlation_id="correlation-consumer-b",
            loaded_revision_number=1,
            consumer_kinds=consumers,
            projection_digest=digest,
        ),
    )

    assert receipts[0] == receipts[1]
    receipt = receipts[0]
    assert receipt is not None
    assert receipt["status"] == "consumed"
    assert receipt["loaded_revision_number"] == 1
    assert receipt["projection_digest"] == digest
    persisted_run = await identity_database[RUNS_COLLECTION].find_one({
        "run_id": "operator-reset:operator-consumption-1",
    })
    assert persisted_run["first_consumption"] == receipt

    await create_operator_reset_revision(
        character_id=_CHARACTER_ID,
        identity=_identity(name="Newest Name"),
        operator_action_id="operator-consumption-2",
        correlation_id="operator-consumption-correlation-2",
    )
    await complete_growth_run_post_commit(
        run_id="operator-reset:operator-consumption-2",
        character_id=_CHARACTER_ID,
        revision_number=2,
    )
    with pytest.raises(IdentityRevisionStaleError) as exc_info:
        await claim_identity_revision_consumption(
            character_id=_CHARACTER_ID,
            episode_id="episode-stale",
            correlation_id="correlation-stale",
            loaded_revision_number=1,
            consumer_kinds=consumers,
            projection_digest=digest,
        )
    assert exc_info.value.loaded_revision_number == 1
    assert exc_info.value.latest_revision_number == 2


async def test_health_derivation_tracks_idle_waiting_and_ready(
    identity_database,
) -> None:
    """Candidate lifecycle should produce truthful evidence-wait health."""

    await ensure_seed_identity(
        character_id=_CHARACTER_ID,
        seed=_identity(),
    )
    idle = await build_identity_growth_health(character_id=_CHARACTER_ID)
    assert idle["state"] == "healthy_idle"
    assert idle["latest_reason_code"] == "not_routed"

    emerging = _candidate(
        candidate_id="candidate-health",
        root_numbers=(1, 2, 3),
        replacement="I can remain open after repeated earned trust.",
    )
    emerging["status"] = "emerging"
    await insert_growth_candidate(emerging)

    waiting = await build_identity_growth_health(character_id=_CHARACTER_ID)
    assert waiting["state"] == "waiting_for_evidence"
    assert waiting["emerging_candidate_count"] == 1
    assert waiting["root_count"] == 3
    assert waiting["local_date_count"] == 2

    ready = deepcopy(emerging)
    ready["status"] = "ready"
    ready["updated_at"] = "2026-07-02T11:00:00+00:00"
    await update_growth_candidate(
        ready,
        expected_updated_at="2026-07-02T10:00:00+00:00",
    )

    promotion_ready = await build_identity_growth_health(
        character_id=_CHARACTER_ID,
    )
    assert promotion_ready["state"] == "promotion_ready"
    assert promotion_ready["emerging_candidate_count"] == 0
    assert promotion_ready["ready_candidate_count"] == 1


async def test_health_derivation_distinguishes_rejection_and_pipeline_failure(
    identity_database,
) -> None:
    """Semantic rejection and process failure should remain separate states."""

    await ensure_seed_identity(
        character_id=_CHARACTER_ID,
        seed=_identity(),
    )
    rejected_run = _run("run-health-rejected", "candidate-health-rejected")
    rejected_run.update({
        "candidate_id": None,
        "lifecycle_state": "complete",
        "disposition": "rejected",
        "review_reason_code": "review_rejected",
        "policy_reason_code": "review_rejected",
        "persistence_reason_code": "review_rejected",
        "completed_at": "2026-07-02T10:05:00+00:00",
    })
    await insert_growth_run(rejected_run)

    rejected = await build_identity_growth_health(
        character_id=_CHARACTER_ID,
    )
    assert rejected["state"] == "semantic_rejection"
    assert rejected["latest_reason_code"] == "review_rejected"
    assert rejected["rejected_count"] == 1

    failed_run = _run("run-health-failed", "candidate-health-failed")
    failed_run.update({
        "candidate_id": None,
        "lifecycle_state": "failed",
        "disposition": "failed",
        "proposal_reason_code": "proposal_contract_failed",
        "review_reason_code": "proposal_contract_failed",
        "policy_reason_code": "proposal_contract_failed",
        "persistence_reason_code": "proposal_contract_failed",
        "started_at": "2026-07-02T11:00:00+00:00",
        "completed_at": "2026-07-02T11:05:00+00:00",
    })
    await insert_growth_run(failed_run)

    failed = await build_identity_growth_health(character_id=_CHARACTER_ID)
    assert failed["state"] == "pipeline_error"
    assert failed["latest_reason_code"] == "proposal_contract_failed"
    assert failed["failed_count"] == 1


async def test_health_derivation_tracks_awaiting_and_active_consumption(
    identity_database,
) -> None:
    """A promoted revision should stay awaiting until its durable receipt."""

    await ensure_seed_identity(
        character_id=_CHARACTER_ID,
        seed=_identity(),
    )
    revision = await create_operator_reset_revision(
        character_id=_CHARACTER_ID,
        identity=_identity(name="Health Active Name"),
        operator_action_id="operator-health-active",
        correlation_id="operator-health-active-correlation",
    )

    awaiting = await build_identity_growth_health(
        character_id=_CHARACTER_ID,
    )
    assert awaiting["state"] == "awaiting_consumption"
    assert awaiting["latest_reason_code"] == "awaiting_first_consumption"

    await complete_growth_run_post_commit(
        run_id="operator-reset:operator-health-active",
        character_id=_CHARACTER_ID,
        revision_number=1,
    )
    still_awaiting = await build_identity_growth_health(
        character_id=_CHARACTER_ID,
    )
    assert still_awaiting["state"] == "awaiting_consumption"

    cognition = project_identity_for_cognition(revision)
    surface = project_identity_for_surface(revision)
    consumers = projected_identity_consumer_kinds(cognition)
    digest = identity_projection_digest(
        revision_number=1,
        cognition_context=cognition,
        surface_context=surface,
    )
    await claim_identity_revision_consumption(
        character_id=_CHARACTER_ID,
        episode_id="episode-health-active",
        correlation_id="correlation-health-active",
        loaded_revision_number=1,
        consumer_kinds=consumers,
        projection_digest=digest,
    )

    active = await build_identity_growth_health(character_id=_CHARACTER_ID)
    assert active["state"] == "healthy_active"
    assert active["latest_reason_code"] == "revision_consumed"
    assert active["promoted_count"] == 1
    assert active["consumed_count"] == 1
    assert active["latest_consumed_revision_number"] == 1


async def test_health_derivation_reports_consumption_mismatch(
    identity_database,
) -> None:
    """A stale-load receipt should outrank all other public health states."""

    await ensure_seed_identity(
        character_id=_CHARACTER_ID,
        seed=_identity(),
    )
    revision = await create_operator_reset_revision(
        character_id=_CHARACTER_ID,
        identity=_identity(name="Health Mismatch Name"),
        operator_action_id="operator-health-mismatch",
        correlation_id="operator-health-mismatch-correlation",
    )
    await complete_growth_run_post_commit(
        run_id="operator-reset:operator-health-mismatch",
        character_id=_CHARACTER_ID,
        revision_number=1,
    )
    cognition = project_identity_for_cognition(revision)
    surface = project_identity_for_surface(revision)
    consumers = projected_identity_consumer_kinds(cognition)
    digest = identity_projection_digest(
        revision_number=1,
        cognition_context=cognition,
        surface_context=surface,
    )
    await record_identity_revision_consumption_mismatch(
        character_id=_CHARACTER_ID,
        episode_id="episode-health-mismatch",
        correlation_id="correlation-health-mismatch",
        loaded_revision_number=0,
        consumer_kinds=consumers,
        projection_digest=digest,
    )

    mismatch = await build_identity_growth_health(
        character_id=_CHARACTER_ID,
    )
    assert mismatch["state"] == "consumption_error"
    assert mismatch["latest_reason_code"] == (
        "revision_consumption_mismatch"
    )
