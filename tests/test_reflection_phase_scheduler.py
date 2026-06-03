"""Deterministic tests for reflection phase run-intent materialization."""

from __future__ import annotations

from datetime import datetime, timezone
import inspect

from kazusa_ai_chatbot.reflection_cycle import phase_scheduler
from kazusa_ai_chatbot.reflection_cycle.models import ReflectionScopeInput


PHASE_PERIOD_SECONDS = 900
MIN_SLOT_SPACING_SECONDS = 60
PROMPT_VERSION = "readonly_reflection_v1"


def test_phase_scheduler_spreads_fit_scopes_across_period_offsets() -> None:
    """Eligible scopes that fit should be assigned deterministic offsets."""

    period_start = datetime(1970, 1, 1, tzinfo=timezone.utc)
    scopes = [
        _scope("scope_a", channel_type="group"),
        _scope("scope_b", channel_type="private"),
        _scope("scope_c", channel_type="group"),
    ]

    intents = phase_scheduler.build_phase_run_intents(
        period_start_utc=period_start,
        eligible_scopes=scopes,
        phase_period_seconds=PHASE_PERIOD_SECONDS,
        max_slots_per_period=3,
        min_slot_spacing_seconds=MIN_SLOT_SPACING_SECONDS,
        prompt_version=PROMPT_VERSION,
    )

    assert [intent["source_scope"]["scope_ref"] for intent in intents] == [
        "scope_a",
        "scope_b",
        "scope_c",
    ]
    assert [intent["offset_seconds"] for intent in intents] == [0, 300, 600]
    assert [intent["due_at"] for intent in intents] == [
        "1970-01-01T00:00:00+00:00",
        "1970-01-01T00:05:00+00:00",
        "1970-01-01T00:10:00+00:00",
    ]


def test_phase_scheduler_rotates_overflow_without_group_packing() -> None:
    """Overflow scopes should rotate across periods without sharing slots."""

    first_period = datetime(1970, 1, 1, tzinfo=timezone.utc)
    second_period = datetime(1970, 1, 1, 0, 15, tzinfo=timezone.utc)
    scopes = [
        _scope("scope_a", channel_type="group"),
        _scope("scope_b", channel_type="group"),
        _scope("scope_c", channel_type="group"),
        _scope("scope_d", channel_type="group"),
        _scope("scope_e", channel_type="group"),
    ]

    first_intents = phase_scheduler.build_phase_run_intents(
        period_start_utc=first_period,
        eligible_scopes=scopes,
        phase_period_seconds=PHASE_PERIOD_SECONDS,
        max_slots_per_period=3,
        min_slot_spacing_seconds=MIN_SLOT_SPACING_SECONDS,
        prompt_version=PROMPT_VERSION,
    )
    second_intents = phase_scheduler.build_phase_run_intents(
        period_start_utc=second_period,
        eligible_scopes=scopes,
        phase_period_seconds=PHASE_PERIOD_SECONDS,
        max_slots_per_period=3,
        min_slot_spacing_seconds=MIN_SLOT_SPACING_SECONDS,
        prompt_version=PROMPT_VERSION,
    )

    assert [intent["source_scope"]["scope_ref"] for intent in first_intents] == [
        "scope_a",
        "scope_b",
        "scope_c",
    ]
    assert [intent["source_scope"]["scope_ref"] for intent in second_intents] == [
        "scope_d",
        "scope_e",
        "scope_a",
    ]
    assert len(first_intents) == 3
    assert len(second_intents) == 3
    assert all(
        intent["source_scope"]["channel_type"] == "group"
        for intent in first_intents + second_intents
    )


def test_phase_scheduler_is_deterministic_for_same_period_snapshot() -> None:
    """The same period snapshot should produce stable run intents."""

    period_start = datetime(1970, 1, 1, 1, 0, tzinfo=timezone.utc)
    scopes = [
        _scope("scope_b", platform_channel_id="b"),
        _scope("scope_a", platform_channel_id="a"),
    ]

    first_intents = phase_scheduler.build_phase_run_intents(
        period_start_utc=period_start,
        eligible_scopes=scopes,
        phase_period_seconds=PHASE_PERIOD_SECONDS,
        max_slots_per_period=3,
        min_slot_spacing_seconds=MIN_SLOT_SPACING_SECONDS,
        prompt_version=PROMPT_VERSION,
    )
    second_intents = phase_scheduler.build_phase_run_intents(
        period_start_utc=period_start,
        eligible_scopes=list(reversed(scopes)),
        phase_period_seconds=PHASE_PERIOD_SECONDS,
        max_slots_per_period=3,
        min_slot_spacing_seconds=MIN_SLOT_SPACING_SECONDS,
        prompt_version=PROMPT_VERSION,
    )

    assert first_intents == second_intents
    assert [intent["source_scope"]["scope_ref"] for intent in first_intents] == [
        "scope_a",
        "scope_b",
    ]


def test_phase_run_intent_shape_maps_to_calendar_run_fields() -> None:
    """Run intents should expose the fields future calendar runs need."""

    period_start = datetime(1970, 1, 1, tzinfo=timezone.utc)

    intents = phase_scheduler.build_phase_run_intents(
        period_start_utc=period_start,
        eligible_scopes=[_scope("scope_a")],
        phase_period_seconds=PHASE_PERIOD_SECONDS,
        max_slots_per_period=3,
        min_slot_spacing_seconds=MIN_SLOT_SPACING_SECONDS,
        prompt_version=PROMPT_VERSION,
    )

    intent = intents[0]
    assert set(intent) == {
        "run_id",
        "trigger_kind",
        "due_at",
        "period_start_utc",
        "slot_index",
        "offset_seconds",
        "source_scope",
        "payload",
        "idempotency_key",
    }
    assert intent["trigger_kind"] == "reflection_phase_slot"
    assert intent["period_start_utc"] == "1970-01-01T00:00:00+00:00"
    assert intent["payload"] == {
        "phase_period_seconds": PHASE_PERIOD_SECONDS,
        "max_slots_per_period": 3,
        "prompt_version": PROMPT_VERSION,
        "allowed_actions": [
            "reflection_hourly_slot",
            "group_self_cognition_review",
        ],
    }
    assert intent["run_id"] == intent["idempotency_key"]


def test_phase_scheduler_module_stays_pure() -> None:
    """The phase materializer should not own runtime side effects."""

    source = inspect.getsource(phase_scheduler)

    forbidden_terms = [
        "event_logging",
        "get_db",
        "asyncio",
        "sleep",
        "lease",
        "retry",
        "AdapterRegistry",
    ]
    for forbidden_term in forbidden_terms:
        assert forbidden_term not in source


def _scope(
    scope_ref: str,
    *,
    platform: str = "qq",
    platform_channel_id: str | None = None,
    channel_type: str = "group",
) -> ReflectionScopeInput:
    """Build a minimal reflection scope for phase scheduler tests."""

    if platform_channel_id is None:
        platform_channel_id = scope_ref
    scope = ReflectionScopeInput(
        scope_ref=scope_ref,
        platform=platform,
        platform_channel_id=platform_channel_id,
        channel_type=channel_type,
        assistant_message_count=1,
        user_message_count=1,
        total_message_count=2,
        first_timestamp="1970-01-01T00:00:00+00:00",
        last_timestamp="1970-01-01T00:01:00+00:00",
        messages=[],
    )
    return scope
