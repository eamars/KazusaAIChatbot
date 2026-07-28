"""Root provenance and transition tests for character identity growth."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.character_identity_growth.identity import (
    candidate_transition_allowed,
    dedupe_evidence_refs,
    evidence_counts,
)
from kazusa_ai_chatbot.character_identity_growth.validation import (
    validate_evidence_ref,
)


def test_same_episode_and_derived_reflection_count_once() -> None:
    """A derivative card must enrich one root without inflating cadence."""

    direct = _evidence(
        evidence_ref_id="evidence-direct",
        source_kind="settled_episode",
    )
    reflected = _evidence(
        evidence_ref_id="evidence-reflection",
        source_kind="daily_reflection",
        derived_reflection_run_ids=["reflection-daily-1"],
    )

    deduped = dedupe_evidence_refs([direct, reflected])

    assert len(deduped) == 1
    assert deduped[0]["root_episode_id"] == "episode-root-1"
    assert deduped[0]["source_kind"] == "settled_episode"
    assert deduped[0]["derived_reflection_run_ids"] == [
        "reflection-daily-1",
    ]
    assert evidence_counts(deduped) == {
        "distinct_episode_count": 1,
        "distinct_local_dates": ["2026-07-27"],
    }


def test_three_roots_across_two_dates_reach_the_default_threshold() -> None:
    """Repository roots and dates should produce the configured pace inputs."""

    refs = [
        _evidence(root_episode_id="episode-root-1"),
        _evidence(
            evidence_ref_id="evidence-2",
            root_episode_id="episode-root-2",
        ),
        _evidence(
            evidence_ref_id="evidence-3",
            root_episode_id="episode-root-3",
            character_local_date="2026-07-28",
        ),
    ]

    deduped = dedupe_evidence_refs(refs)

    assert evidence_counts(deduped) == {
        "distinct_episode_count": 3,
        "distinct_local_dates": ["2026-07-27", "2026-07-28"],
    }


def test_conflicting_repository_metadata_for_one_root_fails_closed() -> None:
    """One root cannot claim two dates or scope kinds."""

    direct = _evidence()
    conflicting = _evidence(
        evidence_ref_id="evidence-conflict",
        character_local_date="2026-07-28",
    )

    with pytest.raises(ValueError, match="episode-root-1"):
        dedupe_evidence_refs([direct, conflicting])


@pytest.mark.parametrize(
    ("current", "target", "expected"),
    [
        ("emerging", "emerging", True),
        ("emerging", "ready", True),
        ("emerging", "rejected", True),
        ("ready", "promoted", True),
        ("ready", "emerging", False),
        ("promoted", "ready", False),
        ("rejected", "emerging", False),
        ("superseded", "ready", False),
    ],
)
def test_candidate_transitions_are_enforced(
    current: str,
    target: str,
    expected: bool,
) -> None:
    """Only declared candidate lifecycle edges should be accepted."""

    assert candidate_transition_allowed(current, target) is expected


def test_evidence_ref_requires_repository_root_and_closed_shape() -> None:
    """Timestamps and derivative IDs cannot substitute for an episode root."""

    missing_root = _evidence()
    missing_root["root_episode_id"] = ""
    unknown_key = _evidence()
    unknown_key["platform_message_id"] = "private-message-id"

    with pytest.raises(ValueError, match="root_episode_id"):
        validate_evidence_ref(missing_root)
    with pytest.raises(ValueError, match="platform_message_id"):
        validate_evidence_ref(unknown_key)


def _evidence(
    *,
    evidence_ref_id: str = "evidence-1",
    root_episode_id: str = "episode-root-1",
    source_kind: str = "settled_episode",
    derived_reflection_run_ids: list[str] | None = None,
    character_local_date: str = "2026-07-27",
) -> dict[str, object]:
    """Build one character-generic repository evidence reference."""

    return {
        "schema_version": "character_identity_evidence_ref.v1",
        "evidence_ref_id": evidence_ref_id,
        "root_episode_id": root_episode_id,
        "correlation_id": "correlation-1",
        "source_kind": source_kind,
        "derived_reflection_run_ids": derived_reflection_run_ids or [],
        "character_local_date": character_local_date,
        "scope_kind": "private",
        "captured_at": "2026-07-27T12:00:00+00:00",
    }
