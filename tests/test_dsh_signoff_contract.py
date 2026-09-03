"""Deterministic guards for the DSH live sign-off oracle."""

from __future__ import annotations

from tests.dsh_trigger_source_e2e_support import (
    CASE_SPECS,
    SIDECAR_LOSS_CASE_SPEC,
    _configured_completion_failed,
    _configured_completion_settled,
    _current_weather_measurement_present,
    _evaluate_sidecar_loss_case,
    _expected_facts_present,
    _runtime_failure_events,
    _source_traces_succeeded,
)


def test_failed_source_trace_cannot_satisfy_signoff() -> None:
    """A finalized failed trace must remain a hard sign-off failure."""

    assert not _source_traces_succeeded([
        {"trace_id": "llmtrace-failed", "status": "failed"}
    ])
    assert _source_traces_succeeded([
        {"trace_id": "llmtrace-good", "status": "succeeded"}
    ])


def test_runtime_and_pipeline_failures_cannot_satisfy_signoff() -> None:
    """Any isolated code crash or failed turn blocks the case."""

    runtime_event = {
        "event_family": "runtime_error",
        "status": "failed",
        "recovered": False,
    }
    pipeline_event = {
        "event_family": "pipeline_turn",
        "status": "failed",
    }

    assert _runtime_failure_events({
        "event_log_events": [runtime_event, pipeline_event]
    }) == [runtime_event, pipeline_event]
    assert not _runtime_failure_events({
        "event_log_events": [{
            "event_family": "pipeline_turn",
            "status": "completed",
        }]
    })


def test_quality_oracle_accepts_paraphrase_but_requires_source_facts() -> None:
    """Quality gating follows required facts instead of exact sentences."""

    spec = CASE_SPECS["user_message_background_summary"]

    assert _expected_facts_present(
        spec,
        "Rowan owns the follow-up.",
    )
    assert not _expected_facts_present(
        spec,
        "The cache alert is stable and a follow-up is planned.",
    )

    result_spec = CASE_SPECS["tool_result_resolved"]
    assert _expected_facts_present(
        result_spec,
        "Morgan owns the handover; checksum review is complete.",
    )
    assert not _expected_facts_present(
        result_spec,
        "The requested handover fact is available.",
    )


def test_weather_quality_oracle_separates_semantics_from_provenance() -> None:
    """Wording may vary while measurement and native receipt both remain."""

    assert _current_weather_measurement_present({
        "prompt_safe_summary": (
            "Current Christchurch temperature is 14.2 degrees C."
        ),
        "evidence": [{
            "evidence_id": "native:weather-receipt",
            "summary": "dsh-native:web_search:call-weather",
            "provenance_refs": [
                "native:weather-receipt",
                "sha256:weather-content",
            ],
        }]
    })
    assert not _current_weather_measurement_present({
        "prompt_safe_summary": "Current Christchurch weather is clear.",
        "evidence": [{
            "evidence_id": "native:weather-receipt",
            "summary": "dsh-native:web_search:call-weather",
            "provenance_refs": ["sha256:weather-content"],
        }],
    })
    assert not _current_weather_measurement_present({
        "prompt_safe_summary": (
            "Current Christchurch temperature is 14.2 degrees C."
        ),
        "evidence": [],
    })


def test_configured_canary_requires_complete_deferred_delivery() -> None:
    """A late DSH result is green only after binding and delivery settlement."""

    binding = {
        "state": "terminal",
        "latest_task_resolution_result": {"status": "resolved"},
    }
    accepted_task = {"state": "delivered"}
    job = {"status": "delivered", "delivery_state": "delivered"}

    assert _configured_completion_settled(
        bindings=[binding],
        accepted_tasks=[accepted_task],
        jobs=[job],
        delivery_payloads=[{"messages": ["Current result"]}],
    )
    assert not _configured_completion_settled(
        bindings=[binding],
        accepted_tasks=[accepted_task],
        jobs=[{"status": "completed", "delivery_state": "pending"}],
        delivery_payloads=[],
    )


def test_configured_canary_accepts_consumed_inline_terminal_result() -> None:
    """A direct terminal result needs no synthetic background lineage."""

    assert _configured_completion_settled(
        bindings=[{
            "state": "consumed_inline",
            "latest_task_resolution_result": {"status": "resolved"},
        }],
        accepted_tasks=[],
        jobs=[],
        delivery_payloads=[],
    )


def test_configured_canary_stops_waiting_on_terminal_failure() -> None:
    """A faulted binding is immediately classified for RCA."""

    assert _configured_completion_failed(
        bindings=[{"state": "faulted"}],
        jobs=[],
    )
    assert _configured_completion_failed(
        bindings=[{"state": "terminal"}],
        jobs=[{"status": "delivery_failed"}],
    )
    assert not _configured_completion_failed(
        bindings=[{"state": "active"}],
        jobs=[{"status": "in_progress"}],
    )


def test_sidecar_loss_requires_terminally_faulted_admission_binding() -> None:
    """Graceful failure is green only after its audit binding is terminal."""

    evidence = {
        "mongo_state": {
            "llm_trace_runs": [{
                "trace_id": "trace-1",
                "status": "succeeded",
                "trigger_source": "user_message",
            }],
            "llm_trace_steps": [{
                "trace_id": "trace-1",
                "capability": "task_resolution_request",
                "status": "blocked",
                "response_plan": {
                    "terminal_work_disposition": "closed",
                    "surface_requirements": {
                        "decision": "explain terminal evidence blocker",
                    },
                },
            }],
            "event_log_events": [],
            "dsh_task_bindings": [{
                "state": "faulted",
                "latest_task_resolution_result": None,
            }],
        },
        "source_output": {
            "dsh_health_after_loss": "unavailable",
            "response": {
                "trace_id": "trace-1",
                "messages": ["I could not retrieve the current result."],
                "operational_error": None,
                "cognition_graph": {"status": "completed"},
            },
        },
        "dsh_sessions": [],
    }

    checks, _failures = _evaluate_sidecar_loss_case(
        SIDECAR_LOSS_CASE_SPEC,
        evidence,
    )

    assert checks["failed_admission_binding_terminally_faulted"]
    assert checks["terminal_surface_has_no_active_work"]
    evidence["mongo_state"]["dsh_task_bindings"][0]["state"] = "opening"
    stale_checks, _stale_failures = _evaluate_sidecar_loss_case(
        SIDECAR_LOSS_CASE_SPEC,
        evidence,
    )
    assert not stale_checks["failed_admission_binding_terminally_faulted"]

    evidence["mongo_state"]["llm_trace_steps"][0]["response_plan"].pop(
        "terminal_work_disposition"
    )
    false_pending_checks, _false_pending_failures = _evaluate_sidecar_loss_case(
        SIDECAR_LOSS_CASE_SPEC,
        evidence,
    )
    assert not false_pending_checks["terminal_surface_has_no_active_work"]

    evidence["mongo_state"]["llm_trace_steps"][0].update({
        "stage_name": "cognition_core_v3.P",
        "status": "succeeded",
        "parsed_output": {
            "goal_resolution": "answerable_now",
            "resolver_requests": [],
        },
    })
    terminal_plan_checks, _terminal_plan_failures = (
        _evaluate_sidecar_loss_case(SIDECAR_LOSS_CASE_SPEC, evidence)
    )
    assert terminal_plan_checks["terminal_surface_has_no_active_work"]


def test_trigger_matrix_has_exactly_two_cases_per_canonical_source() -> None:
    """Every canonical cognition source keeps two production scenarios."""

    counts: dict[str, int] = {}
    for spec in CASE_SPECS.values():
        counts[spec.trigger_source] = counts.get(spec.trigger_source, 0) + 1

    assert counts == {
        "user_message": 2,
        "internal_thought": 2,
        "self_cognition": 2,
        "scheduled_tick": 2,
        "tool_result": 2,
    }
