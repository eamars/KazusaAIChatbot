"""Latest-only identity resolution for one cognition episode."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.projection import (
    identity_projection_digest,
    project_identity_for_cognition,
    project_identity_for_surface,
    projected_identity_consumer_kinds,
)
from kazusa_ai_chatbot.character_identity_growth.runner import (
    reconcile_identity_growth_post_commit,
)
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.db.character import (
    compose_character_profile,
    get_character_runtime_state,
)
from kazusa_ai_chatbot.db.character_identity_growth import (
    IdentityPostCommitPendingError,
    IdentityRevisionStaleError,
    claim_identity_revision_consumption,
    get_current_identity,
    record_identity_revision_consumption_mismatch,
)
from kazusa_ai_chatbot.event_logging import (
    record_character_identity_growth_event,
)


class IdentityConsumptionMismatchError(RuntimeError):
    """Raised when a latest revision cannot be safely consumed."""


class IdentityConsumptionTelemetryError(RuntimeError):
    """Raised when a durable receipt cannot be mirrored to telemetry."""


async def load_latest_identity_for_episode(
    *,
    episode_id: str,
    correlation_id: str,
    include_epistemic_core: bool,
    character_id: str = CHARACTER_GLOBAL_USER_ID,
) -> models.IdentityEpisodeSnapshotV1:
    """Resolve, project, and transactionally claim the latest identity."""

    normalized_episode_id = _require_identifier(
        episode_id,
        context="episode_id",
    )
    normalized_correlation_id = _require_identifier(
        correlation_id,
        context="correlation_id",
    )
    pending_reconciled = False
    stale_attempt_count = 0
    while True:
        revision = await get_current_identity(character_id=character_id)
        snapshot = await _build_snapshot(
            revision,
            include_epistemic_core=include_epistemic_core,
            character_id=character_id,
        )
        revision_number = snapshot["revision_number"]
        if revision_number == 0:
            return snapshot
        try:
            receipt = await claim_identity_revision_consumption(
                character_id=character_id,
                episode_id=normalized_episode_id,
                correlation_id=normalized_correlation_id,
                loaded_revision_number=revision_number,
                consumer_kinds=snapshot["consumer_kinds"],
                projection_digest=snapshot["projection_digest"],
            )
        except IdentityPostCommitPendingError as exc:
            if pending_reconciled:
                raise IdentityConsumptionMismatchError(
                    "latest identity remained post-commit pending"
                ) from exc
            reconciliation = await reconcile_identity_growth_post_commit(
                run_id=exc.run_id,
            )
            if reconciliation["failed_count"]:
                raise IdentityConsumptionMismatchError(
                    "latest identity post-commit reconciliation failed"
                ) from exc
            pending_reconciled = True
            continue
        except IdentityRevisionStaleError as exc:
            stale_attempt_count += 1
            if stale_attempt_count < 2:
                continue
            mismatch = await record_identity_revision_consumption_mismatch(
                character_id=character_id,
                episode_id=normalized_episode_id,
                correlation_id=normalized_correlation_id,
                loaded_revision_number=revision_number,
                consumer_kinds=snapshot["consumer_kinds"],
                projection_digest=snapshot["projection_digest"],
            )
            await _record_consumption_event(
                receipt=mismatch,
                run_id="",
                revision_number=exc.latest_revision_number,
            )
            raise IdentityConsumptionMismatchError(
                "latest identity changed during both bounded load attempts"
            ) from exc

        if receipt is None:
            raise IdentityConsumptionMismatchError(
                "promoted identity has no first-consumption receipt"
            )
        if receipt["status"] != "consumed":
            raise IdentityConsumptionMismatchError(
                "latest identity has a prior mismatch receipt"
            )
        if (
            receipt["episode_id"] == normalized_episode_id
            and receipt["correlation_id"] == normalized_correlation_id
        ):
            await _record_consumption_event(
                receipt=receipt,
                run_id=str(revision["promotion_run_id"] or ""),
                revision_number=revision_number,
            )
        return snapshot


async def _build_snapshot(
    revision: Mapping[str, object],
    *,
    include_epistemic_core: bool,
    character_id: str,
) -> models.IdentityEpisodeSnapshotV1:
    """Compose exact runtime contexts from one validated revision."""

    raw_revision_number = revision.get("revision_number")
    if (
        not isinstance(raw_revision_number, int)
        or isinstance(raw_revision_number, bool)
        or raw_revision_number < 0
    ):
        raise ValueError("identity revision_number is invalid")
    identity = revision.get("effective_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("identity revision has no effective_identity")
    cognition_context = project_identity_for_cognition(
        revision,
        include_epistemic_core=include_epistemic_core,
    )
    surface_context = project_identity_for_surface(revision)
    consumer_kinds = projected_identity_consumer_kinds(
        cognition_context,
    )
    digest = identity_projection_digest(
        revision_number=raw_revision_number,
        cognition_context=cognition_context,
        surface_context=surface_context,
    )
    runtime_state = await get_character_runtime_state()
    character_profile = compose_character_profile(
        identity,
        runtime_state,
        character_id,
    )
    return {
        "revision_number": raw_revision_number,
        "character_profile": character_profile,
        "cognition_context": cognition_context,
        "surface_context": surface_context,
        "projection_digest": digest,
        "consumer_kinds": consumer_kinds,
    }


async def _record_consumption_event(
    *,
    receipt: models.IdentityRevisionConsumptionV1,
    run_id: str,
    revision_number: int,
) -> None:
    """Mirror one durable receipt without identity text or provenance."""

    event_result = await record_character_identity_growth_event(
        event_type="consumption",
        stage="latest_identity_reader",
        reason_code=(
            "revision_consumed"
            if receipt["status"] == "consumed"
            else "revision_consumption_mismatch"
        ),
        status=receipt["status"],
        correlation_id=receipt["correlation_id"],
        run_id=run_id,
        revision_number=revision_number,
        consumer_count=len(receipt["consumer_kinds"]),
        projection_digest=receipt["projection_digest"],
        severity=(
            "info"
            if receipt["status"] == "consumed"
            else "error"
        ),
    )
    if not event_result["accepted"]:
        raise IdentityConsumptionTelemetryError(
            "identity consumption event was not persisted"
        )


def _require_identifier(value: object, *, context: str) -> str:
    """Require one bounded nonempty runtime identifier."""

    if not isinstance(value, str):
        raise ValueError(f"{context} must be text")
    normalized = value.strip()
    if not normalized or len(normalized) > 500:
        raise ValueError(f"{context} must be 1..500 characters")
    return normalized


def snapshot_state_update(
    snapshot: Mapping[str, object],
    *,
    episode_id: str,
    include_epistemic_core: bool,
) -> dict[str, object]:
    """Project one detached identity snapshot onto graph-private fields."""

    return {
        "character_profile": deepcopy(snapshot["character_profile"]),
        "character_identity_revision_number": snapshot["revision_number"],
        "character_identity_context": deepcopy(
            snapshot["cognition_context"]
        ),
        "character_identity_surface_context": deepcopy(
            snapshot["surface_context"]
        ),
        "character_identity_projection_digest": snapshot[
            "projection_digest"
        ],
        "character_identity_consumer_kinds": list(
            snapshot["consumer_kinds"]
        ),
        "character_identity_episode_id": episode_id,
        "character_identity_epistemic_core_included": (
            include_epistemic_core
        ),
    }
