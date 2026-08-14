"""Deterministic contract tests for scheduled future-speech authority."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    build_scheduled_future_speech_authority,
    validate_scheduled_authority_carrier,
    validate_scheduled_authority_proposal,
    validate_scheduled_future_speech_authority,
    validate_scheduled_speech_semantic_verdict,
)
from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.models import (
    ActionValidationError,
    validate_semantic_action_request_v2,
)
from kazusa_ai_chatbot.action_spec.registry import FUTURE_SPEAK_CAPABILITY
from kazusa_ai_chatbot.calendar_scheduler import models as calendar_models
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_actions import (
    materialize_semantic_action_requests,
)
from kazusa_ai_chatbot.self_cognition import (
    models,
    projection,
    runner,
    sources,
    worker,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime


def _detail_ref(
    evidence_handle: str = "e1",
    semantic_summary: str = "当前对话明确约定在该时间开始补偿考核。",
    provenance_role: str = "current_event",
) -> dict[str, str]:
    """Build one bounded authorized detail reference."""

    return {
        "evidence_handle": evidence_handle,
        "semantic_summary": semantic_summary,
        "provenance_role": provenance_role,
    }


def _proposal(
    *,
    temporal_alignment: str = "aligned",
    detail_refs: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    """Build one closed scheduled authority proposal."""

    return {
        "schema_version": "scheduled_authority_proposal.v1",
        "temporal_alignment": temporal_alignment,
        "authorized_content_summary": "在约定时间开始补偿考核。",
        "authorized_detail_refs": detail_refs or [_detail_ref()],
    }


def _evidence(handle: str = "e1") -> dict[str, object]:
    """Build one current-episode evidence row matching the detail ref."""

    return {
        "evidence_handle": handle,
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": f"episode:{handle}",
            "occurred_at": "2026-05-09T00:00:00Z",
            "semantic_summary": "当前对话明确约定在该时间开始补偿考核。",
        },
        "semantic_text": "当前对话明确约定在该时间开始补偿考核。",
        "visible_to": ["q:event_agency"],
        "authority": "current_event",
    }


def _cognition_state() -> dict[str, object]:
    """Build the minimal trusted materializer state with source identity."""

    state = {
        "storage_timestamp_utc": "2026-05-09T21:00:00+00:00",
        "decontextualized_input": (
            "The user asks the character to remind them tonight at ten."
        ),
        "platform": "qq",
        "platform_channel_id": "480386272",
        "channel_type": "group",
        "platform_message_id": "227312230",
        "platform_bot_id": "qq-bot-001",
        "global_user_id": "global-user-001",
        "platform_user_id": "qq-user-001",
        "user_name": "Test User",
        "character_profile": {
            "name": "Test Character",
            "global_user_id": "character-global-001",
        },
        "cognitive_episode": {
            "episode_id": "episode-2026-05-09-001",
            "origin_metadata": {
                "platform_message_id": "227312230",
            },
        },
        "conversation_progress": {},
    }
    return state


def _authority(
    *,
    source_episode_id: str = "episode-2026-05-09-001",
    source_message_id: str = "227312230",
    source_action_attempt_id: str = "action_attempt:future-speak-001",
    accepted_at_utc: str = "2026-05-09T21:00:00+00:00",
    trigger_local: str = "2026-05-10 13:00",
    platform: str = "qq",
    channel_type: str = "group",
    audience_kind: str = "group",
    semantic_objective: str = "在约定时间开始补偿考核。",
    authorized_content_summary: str = "在约定时间开始补偿考核。",
) -> dict[str, object]:
    """Build one validated immutable scheduled future-speech authority."""

    authority = build_scheduled_future_speech_authority(
        source_episode_id=source_episode_id,
        source_message_id=source_message_id,
        source_action_attempt_id=source_action_attempt_id,
        source_llm_trace_id="llmtrace_source-1",
        accepted_at_utc=accepted_at_utc,
        timezone="Pacific/Auckland",
        trigger_local=trigger_local,
        platform=platform,
        channel_type=channel_type,
        audience_kind=audience_kind,
        semantic_objective=semantic_objective,
        authorized_content_summary=authorized_content_summary,
        authorized_detail_refs=[_detail_ref()],
        goal_continuation_ref=None,
    )
    return dict(authority)


def _authority_run() -> dict[str, object]:
    """Build one due scheduled run whose carrier matches its authority."""

    authority = _authority(
        accepted_at_utc="2026-05-09T21:00:00+00:00",
        trigger_local="2026-05-10 13:00",
    )
    run: dict[str, object] = {
        "run_id": "calendar_run_authority_001",
        "schedule_id": "calendar_schedule_authority_001",
        "trigger_kind": "future_cognition",
        "due_at": authority["trigger"]["utc"],
        "created_at": authority["accepted_at"]["utc"],
        "status": "pending",
        "payload": {
            "episode_type": "self_cognition",
            "trigger_at": authority["trigger"]["utc"],
            "continuation_objective": authority["semantic_objective"],
            "source_action_attempt_id": authority["source"][
                "source_action_attempt_id"
            ],
            "scheduled_future_speech_authority": dict(authority),
            "source_refs": [{
                "ref_id": calendar_models.FUTURE_SPEAK_SOURCE_REF_ID,
            }],
            "continuation": {},
        },
        "source_scope": {
            "source_platform": authority["target"]["platform"],
            "source_channel_id": "480386272",
            "source_channel_type": authority["target"]["channel_type"],
            "source_user_id": "self_cognition",
            "source_message_id": authority["source"]["source_message_id"],
        },
    }
    return run


def test_scheduled_authority_proposal_contract_is_closed() -> None:
    """Planner proposals reject open, missing, or mismatched fields."""

    validated = validate_scheduled_authority_proposal(
        _proposal(),
        evidence=[_evidence()],
    )
    assert validated["schema_version"] == "scheduled_authority_proposal.v1"
    assert validated["temporal_alignment"] == "aligned"
    assert validated["authorized_detail_refs"][0]["evidence_handle"] == "e1"

    with pytest.raises(CognitionContractError, match="fields are not exact"):
        validate_scheduled_authority_proposal(
            {
                **_proposal(),
                "decision": "accept",
            },
            evidence=[_evidence()],
        )

    with pytest.raises(CognitionContractError, match="temporal alignment"):
        validate_scheduled_authority_proposal(
            _proposal(temporal_alignment="maybe"),
            evidence=[_evidence()],
        )

    with pytest.raises(CognitionContractError, match="handle is unavailable"):
        validate_scheduled_authority_proposal(
            _proposal(detail_refs=[_detail_ref(evidence_handle="e9")]),
            evidence=[_evidence()],
        )

    historical_evidence = dict(_evidence())
    historical_evidence["evidence_ref"] = dict(
        historical_evidence["evidence_ref"]
    )
    historical_evidence["evidence_ref"]["source_kind"] = (
        "promoted_reflection"
    )
    historical_evidence["authority"] = "character_world_context"
    with pytest.raises(CognitionContractError, match="does not match"):
        validate_scheduled_authority_proposal(
            _proposal(detail_refs=[_detail_ref()]),
            evidence=[historical_evidence],
        )

    elevated_historical = dict(historical_evidence)
    with pytest.raises(CognitionContractError, match="current-episode"):
        validate_scheduled_authority_proposal(
            _proposal(
                detail_refs=[_detail_ref(provenance_role="character_world_context")]
            ),
            evidence=[elevated_historical],
        )


def test_persona_materializer_carries_validated_authority() -> None:
    """The materializer carries the proposal without carrier ids."""

    proposal = _proposal()
    requests = [
        {
            "capability": FUTURE_SPEAK_CAPABILITY,
            "decision": "2026-05-10 13:00",
            "detail": "在约定时间开始补偿考核。",
            "reason": "用户要求在未来时间开始补偿考核。",
            "surface_role": "ordinary",
            "goal_continuation_ref": None,
            "scheduled_authority_proposal": proposal,
        }
    ]

    action_specs = materialize_semantic_action_requests(
        requests,
        _cognition_state(),
    )

    assert len(action_specs) == 1
    params = action_specs[0]["params"]
    assert params["scheduled_authority_proposal"] == proposal
    assert params["source_episode_id"] == "episode-2026-05-09-001"
    assert params["source_message_id"] == "227312230"
    assert params["accepted_at_utc"] == "2026-05-09T21:00:00+00:00"
    serialized = json.dumps(params, ensure_ascii=False)
    assert "task-" not in serialized
    assert "job-" not in serialized
    assert "calendar_run" not in serialized


@pytest.mark.asyncio
async def test_action_execution_passes_authority_without_carrier_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution threads the proposal without later carrier ids."""

    from kazusa_ai_chatbot.action_spec import execution as execution_module

    captured: dict[str, object] = {}

    async def enqueue_future_speak_action(
        action_spec: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        captured["action_spec"] = action_spec
        captured["kwargs"] = kwargs
        return {
            "status": "pending",
            "job_id": "job-1",
            "job_ref": "background_work_job:job-1",
            "accepted_task_id": "task-1",
            "task_identity_key": "identity-1",
            "accepted_task_summary": "在约定时间开始补偿考核。",
            "accepted_task_state": "scheduled",
            "acknowledgement_constraint": "promise_allowed",
            "wait_guidance": "non_numeric_wait",
            "result_summary": "Accepted task scheduled.",
        }

    monkeypatch.setattr(
        execution_module,
        "enqueue_future_speak_action",
        enqueue_future_speak_action,
    )

    action_spec = materialize_semantic_action_requests(
        [
            {
                "capability": FUTURE_SPEAK_CAPABILITY,
                "decision": "2026-05-10 13:00",
                "detail": "在约定时间开始补偿考核。",
                "reason": "用户要求在未来时间开始补偿考核。",
                "surface_role": "ordinary",
                "goal_continuation_ref": None,
                "scheduled_authority_proposal": _proposal(),
            }
        ],
        _cognition_state(),
    )[0]

    await execution_module.execute_action_specs_for_trace(
        [action_spec],
        storage_timestamp_utc="2026-05-09T21:00:00+00:00",
    )

    proposal = captured["kwargs"]["scheduled_authority_proposal"]
    assert isinstance(proposal, dict)
    assert proposal["schema_version"] == "scheduled_authority_proposal.v1"
    serialized = json.dumps(proposal, ensure_ascii=False)
    assert "accepted_task" not in serialized
    assert "job-" not in serialized
    assert "calendar_schedule" not in serialized
    assert "delivery_tracking" not in serialized


def test_accepted_task_carrier_keeps_authority_immutable() -> None:
    """Accepted-task local ids cannot change the authority identity."""

    from kazusa_ai_chatbot.accepted_task.models import AcceptedTaskDoc

    authority = _authority()
    carrier: AcceptedTaskDoc = {
        "schema_version": "accepted_task.v2",
        "accepted_task_id": "task-001",
        "task_identity_key": "accepted_task:v2:abc",
        "scheduled_future_speech_authority": copy.deepcopy(authority),
    }

    assert carrier["scheduled_future_speech_authority"] == authority
    carrier["accepted_task_id"] = "task-other"
    validated = validate_scheduled_future_speech_authority(
        carrier["scheduled_future_speech_authority"]
    )
    assert validated["authority_id"] == authority["authority_id"]
    assert validate_scheduled_authority_carrier({
        "schema_version": "scheduled_authority_carrier.v1",
        "authority": authority,
        "accepted_task_id": "task-001",
    })


def test_calendar_run_carries_authority_identity() -> None:
    """Schedule and run payloads carry the exact authority."""

    from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
        build_future_cognition_calendar_documents,
    )
    from kazusa_ai_chatbot.calendar_scheduler.models import (
        SCHEDULED_AUTHORITY_PAYLOAD_KEY,
    )

    authority = _authority()
    action_spec = {
        "schema_version": "action_spec.v1",
        "kind": "trigger_future_cognition",
        "cognition_mode": "deliberative",
        "source_refs": [{
            "schema_version": "action_source_ref.v1",
            "ref_kind": "cognitive_episode",
            "ref_id": "episode-2026-05-09-001",
            "owner": "cognition",
            "relationship": "basis",
            "evidence_refs": [],
        }],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "cognitive_episode",
            "target_id": None,
            "owner": "orchestrator",
            "scope": {
                "episode_type": "self_cognition",
                "source_platform": "qq",
                "source_channel_id": "480386272",
                "source_channel_type": "group",
                "source_message_id": "227312230",
            },
        },
        "params": {
            "episode_type": "self_cognition",
            "trigger_at": "2026-05-10 13:00",
            "continuation_objective": "在约定时间开始补偿考核。",
            SCHEDULED_AUTHORITY_PAYLOAD_KEY: authority,
        },
        "urgency": "scheduled",
        "visibility": "private",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "scheduled_followup",
            "episode_type": "self_cognition",
            "max_depth": 1,
            "include_result_as": "scheduled_event",
        },
        "surface_role": "ordinary",
        "goal_continuation_ref": None,
        "reason": "The accepted future-speak task schedules later cognition.",
    }

    documents = build_future_cognition_calendar_documents(
        action_spec,
        storage_timestamp_utc="2026-05-09T21:00:00+00:00",
        action_attempt_id="action_attempt:future-speak-001",
    )
    schedule = documents["schedule"]
    run = documents["run"]

    assert schedule["payload"][SCHEDULED_AUTHORITY_PAYLOAD_KEY] == authority
    assert run["payload"][SCHEDULED_AUTHORITY_PAYLOAD_KEY] == authority
    assert parse_storage_utc_datetime(
        run["due_at"]
    ) == parse_storage_utc_datetime(authority["trigger"]["utc"])


def test_self_cognition_scheduled_models_reject_open_gate_fields() -> None:
    """Gate model output cannot author decision, attempt, or issue fields."""

    verdict = {
        "schema_version": "scheduled_speech_semantic_verdict.v1",
        "time_claim_alignment": "aligned",
        "objective_alignment": "aligned",
        "source_grounding": "current_authority",
        "audience_alignment": "aligned",
        "execution_claim": "aligned",
    }
    validated = validate_scheduled_speech_semantic_verdict(verdict)
    assert validated["schema_version"] == (
        "scheduled_speech_semantic_verdict.v1"
    )

    for forbidden_field in (
        "decision",
        "attempt",
        "attempt_count",
        "reason",
        "issues",
        "open_issues",
        "dispatch",
    ):
        with pytest.raises(CognitionContractError, match="fields are not exact"):
            validate_scheduled_speech_semantic_verdict({
                **verdict,
                forbidden_field: "x",
            })

    with pytest.raises(CognitionContractError, match="time claim"):
        validate_scheduled_speech_semantic_verdict({
            **verdict,
            "time_claim_alignment": "open",
        })

    with pytest.raises(CognitionContractError, match="objective alignment"):
        validate_scheduled_speech_semantic_verdict({
            **verdict,
            "objective_alignment": "unknown",
        })


def test_source_packet_projects_authority_without_delivery_ids() -> None:
    """Scheduled packets expose semantic authority but no delivery ids."""

    authority = _authority(
        trigger_local="2026-05-10 13:00",
    )
    case: models.SelfCognitionCase = {
        "case_name": models.CASE_SCHEDULED_FUTURE_COGNITION,
        "case_id": "scheduled_future_cognition_slot:test-001",
        "idle_timestamp_utc": "2026-05-10T01:00:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-10T01:00:00+00:00",
        "trigger_kind": models.TRIGGER_SCHEDULED_FUTURE_COGNITION,
        "semantic_due_state": models.DUE_STATE_DUE_NOW,
        "actionability": "scheduled_private_followup_ready_no_direct_contact",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "480386272",
            "channel_type": "group",
            "user_id": None,
        },
        "source_refs": [
            {
                "source_kind": "scheduled_future_cognition_slot",
                "source_id": "scheduled_future_cognition_slot:test-001",
                "due_at": authority["trigger"]["utc"],
                "summary": "在约定时间开始补偿考核。",
            }
        ],
        "visible_context": [],
        "conversation_progress": None,
        "source_context": {
            "schema_version": "self_cognition_scheduled_source_context.v1",
            "context_kind": "scheduled_future_cognition",
            "continuation_objective": "在约定时间开始补偿考核。",
            "continuation_mode": "observe_then_decide",
        },
        "scheduled_future_speech_authority": authority,
        "source_calendar_run_id": "calendar_run_001",
        "source_calendar_run_due_at": authority["trigger"]["utc"],
    }

    packet = projection.build_source_packet(case)
    source_context = packet["source_context"]
    assert isinstance(source_context, dict)
    projected_authority = source_context["scheduled_authority"]
    assert projected_authority["objective"] == "在约定时间开始补偿考核。"
    assert projected_authority["summary"] == "在约定时间开始补偿考核。"
    assert projected_authority["detail_refs"][0]["provenance_role"] == (
        "current_event"
    )
    assert projected_authority["audience_kind"] == "group"
    assert projected_authority["local_due_datetime"]
    serialized = json.dumps(packet, ensure_ascii=False).lower()
    rendered = projection.render_source_packet_text(packet).lower()
    combined = f"{serialized}\n{rendered}"
    for forbidden in (
        "480386272",
        "227312230",
        "calendar_run_001",
        "episode-2026-05-09-001",
        "action_attempt",
        "llmtrace",
        "authority_id",
        "delivery_tracking",
    ):
        assert forbidden not in combined


@pytest.mark.asyncio
async def test_source_collector_rejects_missing_authority() -> None:
    """Due runs without the new authority become typed skip work."""

    run = {
        "run_id": "calendar_run_legacy_001",
        "schedule_id": "calendar_schedule_legacy_001",
        "trigger_kind": "future_cognition",
        "due_at": "2026-05-10T01:00:00+00:00",
        "created_at": "2026-05-09T21:00:00+00:00",
        "status": "pending",
        "payload": {
            "episode_type": "self_cognition",
            "trigger_at": "2026-05-10T01:00:00+00:00",
            "continuation_objective": "在约定时间开始补偿考核。",
            "source_action_attempt_id": "action_attempt:legacy-001",
            "source_refs": [{
                "ref_id": calendar_models.FUTURE_SPEAK_SOURCE_REF_ID,
            }],
            "continuation": {},
        },
        "source_scope": {},
    }

    async def list_due_runs(**kwargs: object) -> list[dict[str, object]]:
        del kwargs
        return [run]

    cases = await sources.collect_scheduled_future_cognition_cases(
        now=datetime(2026, 5, 10, 1, 0, tzinfo=timezone.utc),
        character_profile={"name": "TestCharacter"},
        max_cases=3,
        list_due_calendar_runs_func=list_due_runs,
        get_latest_private_channel_func=lambda **kwargs: None,
    )

    assert len(cases) == 1
    assert cases[0]["source_calendar_skip_reason"] == (
        "scheduled_authority_missing"
    )
    assert cases[0]["source_calendar_run_id"] == "calendar_run_legacy_001"
    assert "scheduled_future_speech_authority" not in cases[0]


@pytest.mark.asyncio
async def test_source_collector_preserves_generic_future_cognition_without_authority() -> None:
    """Authority-free generic future cognition remains runnable source work."""

    run = {
        "run_id": "calendar_run_generic_001",
        "schedule_id": "calendar_schedule_generic_001",
        "trigger_kind": "future_cognition",
        "due_at": "2026-05-10T01:00:00+00:00",
        "created_at": "2026-05-09T21:00:00+00:00",
        "status": "pending",
        "payload": {
            "episode_type": "self_cognition",
            "trigger_at": "2026-05-10T01:00:00+00:00",
            "continuation_objective": "检查当前认知是否需要继续。",
            "source_action_attempt_id": "",
            "source_refs": [{
                "ref_id": "cognitive_episode",
            }],
            "continuation": {},
        },
        "source_scope": {},
    }

    async def list_due_runs(**kwargs: object) -> list[dict[str, object]]:
        del kwargs
        return [run]

    cases = await sources.collect_scheduled_future_cognition_cases(
        now=datetime(2026, 5, 10, 1, 0, tzinfo=timezone.utc),
        character_profile={"name": "TestCharacter"},
        max_cases=3,
        list_due_calendar_runs_func=list_due_runs,
        get_latest_private_channel_func=lambda **kwargs: None,
    )

    assert len(cases) == 1
    assert "source_calendar_skip_reason" not in cases[0]
    assert "scheduled_future_speech_authority" not in cases[0]
    source_packet = projection.build_source_packet(cases[0])
    assert "scheduled_authority" not in source_packet["source_context"]
    gate_result = await worker._apply_scheduled_content_gate(
        case=cases[0],
        artifact_payloads={
            models.ARTIFACT_ACTION_CANDIDATE: {
                "text": "继续检查当前认知。",
            }
        },
        now=datetime(2026, 5, 10, 1, 0, tzinfo=timezone.utc),
    )
    assert gate_result == {
        "accepted": True,
        "gate_codes": [],
        "attempt_count": 0,
    }


@pytest.mark.asyncio
async def test_source_collector_rejects_mismatched_authority_carrier_fields() -> None:
    """Scheduled runs contradicting their authority become typed skip work."""

    now = datetime(2026, 5, 10, 1, 30, tzinfo=timezone.utc)

    async def collect(run: dict[str, object]) -> list[models.SelfCognitionCase]:
        async def list_due_runs(**kwargs: object) -> list[dict[str, object]]:
            del kwargs
            return [run]

        cases = await sources.collect_scheduled_future_cognition_cases(
            now=now,
            character_profile={"name": "TestCharacter"},
            max_cases=3,
            list_due_calendar_runs_func=list_due_runs,
            get_latest_private_channel_func=lambda **kwargs: None,
        )
        return cases

    base_run = _authority_run()
    cases = await collect(base_run)
    assert len(cases) == 1
    assert "scheduled_future_speech_authority" in cases[0]

    mismatched_attempt = _authority_run()
    mismatched_attempt["payload"]["source_action_attempt_id"] = (  # type: ignore[index]
        "action_attempt:other"
    )
    cases = await collect(mismatched_attempt)
    assert cases[0]["source_calendar_skip_reason"] == (  # type: ignore[index]
        "scheduled_authority_invalid"
    )

    mismatched_objective = _authority_run()
    mismatched_objective["payload"]["continuation_objective"] = (  # type: ignore[index]
        "另一个目标。"
    )
    cases = await collect(mismatched_objective)
    assert cases[0]["source_calendar_skip_reason"] == (  # type: ignore[index]
        "scheduled_authority_invalid"
    )

    mismatched_trigger = _authority_run()
    mismatched_trigger["due_at"] = "2026-05-10T00:30:00+00:00"
    cases = await collect(mismatched_trigger)
    assert cases[0]["source_calendar_skip_reason"] == (  # type: ignore[index]
        "scheduled_authority_invalid"
    )

    mismatched_platform = _authority_run()
    mismatched_platform["source_scope"]["source_platform"] = "debug"  # type: ignore[index]
    cases = await collect(mismatched_platform)
    assert cases[0]["source_calendar_skip_reason"] == (  # type: ignore[index]
        "scheduled_authority_invalid"
    )

    mismatched_channel = _authority_run()
    mismatched_channel["source_scope"]["source_channel_type"] = "private"  # type: ignore[index]
    cases = await collect(mismatched_channel)
    assert cases[0]["source_calendar_skip_reason"] == (  # type: ignore[index]
        "scheduled_authority_invalid"
    )

    audience_mismatch_authority = _authority(
        accepted_at_utc="2026-05-09T21:00:00+00:00",
        trigger_local="2026-05-10 13:00",
        audience_kind="private",
    )
    mismatched_audience = _authority_run()
    mismatched_audience["payload"][  # type: ignore[index]
        "scheduled_future_speech_authority"
    ] = dict(audience_mismatch_authority)
    cases = await collect(mismatched_audience)
    assert cases[0]["source_calendar_skip_reason"] == (  # type: ignore[index]
        "scheduled_authority_invalid"
    )


def test_due_guard_rejects_early_run() -> None:
    """The deterministic due guard fails closed before the trigger."""

    authority = _authority(
        accepted_at_utc="2026-05-09T21:00:00+00:00",
        trigger_local="2026-05-10 13:00",
    )
    case: models.SelfCognitionCase = {
        "case_name": models.CASE_SCHEDULED_FUTURE_COGNITION,
        "case_id": "scheduled_future_cognition_slot:test-002",
        "idle_timestamp_utc": "2026-05-10T00:30:00+00:00",
        "trigger_kind": models.TRIGGER_SCHEDULED_FUTURE_COGNITION,
        "scheduled_future_speech_authority": authority,
        "source_calendar_run_due_at": authority["trigger"]["utc"],
    }

    passed, gate_codes = runner.enforce_scheduled_authority_due_guard(
        case,
        datetime(2026, 5, 10, 0, 30, tzinfo=timezone.utc),
    )
    assert passed is False
    assert gate_codes == ["scheduled_due_not_reached"]

    passed, gate_codes = runner.enforce_scheduled_authority_due_guard(
        case,
        datetime(2026, 5, 10, 1, 5, tzinfo=timezone.utc),
    )
    assert passed is True
    assert gate_codes == []

    mismatched = dict(case)
    mismatched["source_calendar_run_due_at"] = "2026-05-10T02:00:00+00:00"
    passed, gate_codes = runner.enforce_scheduled_authority_due_guard(
        mismatched,
        datetime(2026, 5, 10, 3, 0, tzinfo=timezone.utc),
    )
    assert passed is False
    assert gate_codes == ["scheduled_trigger_identity_mismatch"]

    missing = dict(case)
    missing.pop("scheduled_future_speech_authority")
    passed, gate_codes = runner.enforce_scheduled_authority_due_guard(
        missing,
        datetime(2026, 5, 10, 3, 0, tzinfo=timezone.utc),
    )
    assert passed is False
    assert gate_codes == ["scheduled_authority_missing"]


def test_scheduled_gate_truth_table_is_deterministic() -> None:
    """The worker truth table maps prerequisites and dimensions exactly."""

    from kazusa_ai_chatbot.self_cognition.worker import (
        evaluate_scheduled_content_gate,
    )

    verdict = {
        "schema_version": "scheduled_speech_semantic_verdict.v1",
        "time_claim_alignment": "aligned",
        "objective_alignment": "aligned",
        "source_grounding": "current_authority",
        "audience_alignment": "aligned",
        "execution_claim": "aligned",
    }
    accepted, gate_codes = evaluate_scheduled_content_gate(
        authority_missing=False,
        authority_invalid=False,
        trigger_identity_ok=True,
        due_reached=True,
        candidate_present=True,
        verdict=verdict,
    )
    assert accepted is True
    assert gate_codes == []

    truth_table = [
        (True, False, False, True, True, verdict, "scheduled_authority_missing"),
        (
            False,
            True,
            True,
            True,
            True,
            verdict,
            "scheduled_authority_invalid",
        ),
        (
            False,
            False,
            False,
            True,
            True,
            verdict,
            "scheduled_trigger_identity_mismatch",
        ),
        (
            False,
            False,
            True,
            False,
            True,
            verdict,
            "scheduled_due_not_reached",
        ),
        (
            False,
            False,
            True,
            True,
            False,
            verdict,
            "scheduled_candidate_empty",
        ),
        (
            False,
            False,
            True,
            True,
            True,
            None,
            "scheduled_evaluator_contract_error",
        ),
    ]
    for (
        authority_missing,
        authority_invalid,
        trigger_identity_ok,
        due_reached,
        candidate_present,
        candidate_verdict,
        expected_code,
    ) in truth_table:
        accepted, gate_codes = evaluate_scheduled_content_gate(
            authority_missing=authority_missing,
            authority_invalid=authority_invalid,
            trigger_identity_ok=trigger_identity_ok,
            due_reached=due_reached,
            candidate_present=candidate_present,
            verdict=candidate_verdict,
        )
        assert accepted is False
        assert gate_codes == [expected_code]

    dimension_cases = {
        "time_claim_alignment": "premature",
        "objective_alignment": "scope_expansion",
        "source_grounding": "historical_only",
        "audience_alignment": "mismatch",
        "execution_claim": "false",
    }
    expected_codes = {
        "time_claim_alignment": "scheduled_time_claim_mismatch",
        "objective_alignment": "scheduled_objective_mismatch",
        "source_grounding": "scheduled_source_not_current_authority",
        "audience_alignment": "scheduled_audience_mismatch",
        "execution_claim": "scheduled_execution_claim_mismatch",
    }
    for dimension, adverse_value in dimension_cases.items():
        adverse_verdict = dict(verdict)
        adverse_verdict[dimension] = adverse_value
        accepted, gate_codes = evaluate_scheduled_content_gate(
            authority_missing=False,
            authority_invalid=False,
            trigger_identity_ok=True,
            due_reached=True,
            candidate_present=True,
            verdict=adverse_verdict,
        )
        assert accepted is False
        assert gate_codes == [expected_codes[dimension]]

    unavailable_verdict = dict(verdict)
    unavailable_verdict["source_grounding"] = "unavailable"
    accepted, gate_codes = evaluate_scheduled_content_gate(
        authority_missing=False,
        authority_invalid=False,
        trigger_identity_ok=True,
        due_reached=True,
        candidate_present=True,
        verdict=unavailable_verdict,
    )
    assert accepted is False
    assert gate_codes == ["scheduled_evaluator_unavailable"]


def test_consolidation_admission_filters_rejected_candidate() -> None:
    """Suppressed admission metadata keeps candidate text out of memory input."""

    from kazusa_ai_chatbot.consolidation.schema import ConsolidatorState

    admission = {
        "schema_version": "scheduled_candidate_admission.v1",
        "disposition": "suppressed",
        "gate_codes": ["scheduled_objective_mismatch"],
        "authority_id": "sha256-abc",
        "dispatch_status": "scheduled_content_suppressed",
    }
    state: ConsolidatorState = {
        "storage_timestamp_utc": "2026-05-10T01:00:00+00:00",
        "local_time_context": {
            "current_local_datetime": "2026-05-10 13:00",
            "current_local_weekday": "Sunday",
        },
        "global_user_id": "global-user-001",
        "user_name": "Test User",
        "user_profile": {},
        "platform": "qq",
        "platform_channel_id": "480386272",
        "channel_type": "group",
        "platform_message_id": "msg-1",
        "internal_monologue": "",
        "final_dialog": [],
        "episode_trace_projection": {},
        "interaction_subtext": "",
        "subjective_appraisals": [],
        "emotional_appraisal": "",
        "character_intent": "",
        "logical_stance": "",
        "character_profile": {},
        "group_channel_style_image": {},
        "rag_result": {},
        "existing_dedup_keys": set(),
        "decontextualized_input": "来源：在约定时间开始补偿考核。",
        "chat_history_recent": [],
        "metadata": {},
        "consolidation_origin": {
            "origin_kind": "self_cognition",
            "origin_episode_id": "episode-1",
        },
        "consolidation_target_plan": {"targets": []},
        "new_facts": [],
        "future_promises": [],
        "should_stop": False,
        "scheduled_candidate_admission": admission,
    }

    assert state["final_dialog"] == []
    assert admission["disposition"] == "suppressed"
    assert "厕所隔间的检查" not in json.dumps(
        state["decontextualized_input"],
        ensure_ascii=False,
    )


def test_tracking_projection_is_deterministic() -> None:
    """Scheduled gate traces are deterministic and id-free."""

    from kazusa_ai_chatbot.self_cognition import tracking

    authority = _authority()
    case: models.SelfCognitionCase = {
        "case_name": models.CASE_SCHEDULED_FUTURE_COGNITION,
        "case_id": "scheduled_future_cognition_slot:test-003",
        "idle_timestamp_utc": "2026-05-10T01:00:00+00:00",
        "trigger_kind": models.TRIGGER_SCHEDULED_FUTURE_COGNITION,
        "scheduled_future_speech_authority": authority,
    }
    gate_result: models.SelfCognitionScheduledGateResult = {
        "schema_version": models.SCHEDULED_GATE_RESULT_SCHEMA_VERSION,
        "disposition": models.SCHEDULED_GATE_DISPOSITION_SUPPRESSED,
        "gate_codes": ["scheduled_objective_mismatch"],
        "evaluator_attempt_count": 1,
    }

    first_trace = tracking.build_scheduled_gate_trace(
        case,
        gate_result=gate_result,
        dispatch_status="scheduled_content_suppressed",
    )
    second_trace = tracking.build_scheduled_gate_trace(
        case,
        gate_result=gate_result,
        dispatch_status="scheduled_content_suppressed",
    )

    assert first_trace == second_trace
    assert first_trace["authority_id"] == authority["authority_id"]
    assert first_trace["gate_disposition"] == "suppressed"
    assert first_trace["gate_codes"] == ["scheduled_objective_mismatch"]
    assert first_trace["evaluator_attempt_count"] == 1
    assert first_trace["dispatch_status"] == "scheduled_content_suppressed"
    assert first_trace["source_episode_id"] == "episode-2026-05-09-001"
    assert first_trace["source_message_id"] == "227312230"
    serialized = json.dumps(first_trace, ensure_ascii=False)
    assert "task-" not in serialized
    assert "calendar_schedule" not in serialized
    assert "llmtrace" not in serialized


def test_scheduled_authority_builder_emits_native_utc_z() -> None:
    """Authority timestamps are canonicalized to native UTC Z text."""

    authority = _authority(
        accepted_at_utc="2026-05-09T21:00:00+00:00",
        trigger_local="2026-05-10 13:00",
    )

    assert authority["accepted_at"]["utc"].endswith("Z")
    assert authority["trigger"]["utc"].endswith("Z")
    assert "+00:00" not in authority["accepted_at"]["utc"]
    assert "+00:00" not in authority["trigger"]["utc"]
    validated = validate_scheduled_future_speech_authority(authority)
    assert validated["authority_id"] == authority["authority_id"]

    z_form = dict(_authority(accepted_at_utc="2026-05-09T21:00:00Z"))
    offset_form = dict(
        _authority(accepted_at_utc="2026-05-09T21:00:00+00:00")
    )
    assert z_form["accepted_at"]["utc"] == offset_form["accepted_at"]["utc"]
    assert z_form["authority_id"] == offset_form["authority_id"]


def test_authority_rejects_local_timezone_utc_inconsistency() -> None:
    """Authority timestamps fail closed when local or timezone contradicts UTC."""

    mutated_local = dict(_authority())
    mutated_local["trigger"]["local"] = "2026-05-10 14:00"
    with pytest.raises(CognitionContractError, match="local time"):
        validate_scheduled_future_speech_authority(mutated_local)

    mutated_timezone = dict(_authority())
    mutated_timezone["accepted_at"]["timezone"] = "Asia/Shanghai"
    with pytest.raises(CognitionContractError, match="timezone"):
        validate_scheduled_future_speech_authority(mutated_timezone)

    mutated_utc = dict(_authority())
    mutated_utc["accepted_at"]["utc"] = "2026-05-09T22:00:00Z"
    with pytest.raises(CognitionContractError, match="does not match utc"):
        validate_scheduled_future_speech_authority(mutated_utc)


def test_authority_id_binds_local_time_and_timezone() -> None:
    """Local and timezone mutation is rejected and changes the identity."""

    base = _authority()

    shifted_local = dict(base)
    shifted_local["trigger"]["local"] = "2026-05-10 14:00"
    with pytest.raises(CognitionContractError):
        validate_scheduled_future_speech_authority(shifted_local)

    shifted_timezone = dict(base)
    shifted_timezone["trigger"]["timezone"] = "Asia/Shanghai"
    with pytest.raises(CognitionContractError):
        validate_scheduled_future_speech_authority(shifted_timezone)

    with pytest.raises(CognitionContractError, match="timezone"):
        build_scheduled_future_speech_authority(
            source_episode_id="episode-2026-05-09-001",
            source_message_id="227312230",
            source_action_attempt_id="action_attempt:future-speak-001",
            source_llm_trace_id="llmtrace_source-1",
            accepted_at_utc="2026-05-09T21:00:00+00:00",
            timezone="Asia/Shanghai",
            trigger_local="2026-05-10 13:00",
            platform="qq",
            channel_type="group",
            audience_kind="group",
            semantic_objective="在约定时间开始补偿考核。",
            authorized_content_summary="在约定时间开始补偿考核。",
            authorized_detail_refs=[_detail_ref()],
            goal_continuation_ref=None,
        )

    rebuilt = build_scheduled_future_speech_authority(
        source_episode_id="episode-2026-05-09-001",
        source_message_id="227312230",
        source_action_attempt_id="action_attempt:future-speak-001",
        source_llm_trace_id="llmtrace_source-1",
        accepted_at_utc="2026-05-09T21:00:00+00:00",
        timezone="Pacific/Auckland",
        trigger_local="2026-05-10 14:00",
        platform="qq",
        channel_type="group",
        audience_kind="group",
        semantic_objective="在约定时间开始补偿考核。",
        authorized_content_summary="在约定时间开始补偿考核。",
        authorized_detail_refs=[_detail_ref()],
        goal_continuation_ref=None,
    )
    assert rebuilt["authority_id"] != base["authority_id"]
    assert rebuilt["trigger"]["local"] == "2026-05-10 14:00"


def test_future_speak_v2_request_requires_and_preserves_closed_authority_proposal() -> None:
    """The V2 request contract discriminates the future-speak proposal."""

    base_request = {
        "action_kind": FUTURE_SPEAK_CAPABILITY,
        "decision": "2026-05-10 13:00",
        "context_ref": "current episode",
        "semantic_goal": "在约定时间开始补偿考核。",
        "reason": "用户要求在未来时间开始补偿考核。",
        "target_roles": [],
        "evidence_handles": ["e1"],
    }

    validated = validate_semantic_action_request_v2(
        {
            **base_request,
            "scheduled_authority_proposal": _proposal(),
        },
        available_action_kinds={FUTURE_SPEAK_CAPABILITY},
    )
    assert validated["action_kind"] == FUTURE_SPEAK_CAPABILITY
    proposal = validated["scheduled_authority_proposal"]
    assert proposal["schema_version"] == "scheduled_authority_proposal.v1"
    assert proposal["temporal_alignment"] == "aligned"
    assert proposal["authorized_detail_refs"][0]["evidence_handle"] == "e1"

    with pytest.raises(ActionValidationError, match="fields are not exact"):
        validate_semantic_action_request_v2(
            dict(base_request),
            available_action_kinds={FUTURE_SPEAK_CAPABILITY},
        )

    with pytest.raises(ActionValidationError, match="fields are not exact"):
        validate_semantic_action_request_v2(
            {
                **base_request,
                "action_kind": "speak",
                "scheduled_authority_proposal": _proposal(),
            },
            available_action_kinds={"speak"},
        )

    with pytest.raises(
        ActionValidationError,
        match="scheduled_authority_proposal",
    ):
        validate_semantic_action_request_v2(
            {
                **base_request,
                "scheduled_authority_proposal": _proposal(
                    temporal_alignment="maybe"
                ),
            },
            available_action_kinds={FUTURE_SPEAK_CAPABILITY},
        )

    unrelated = {
        "action_kind": "speak",
        "decision": "visible_reply",
        "context_ref": "current episode",
        "semantic_goal": "在约定时间开始补偿考核。",
        "reason": "用户要求在未来时间开始补偿考核。",
        "target_roles": [],
        "evidence_handles": ["e1"],
    }
    validated_unrelated = validate_semantic_action_request_v2(
        unrelated,
        available_action_kinds={"speak"},
    )
    assert set(validated_unrelated) == set(unrelated)


def test_v2_evaluator_returns_future_speak_authority_proposal_unchanged() -> None:
    """The generic V2 evaluator propagates the validated proposal unchanged."""

    evaluator = ActionSpecEvaluator()

    evaluation = evaluator.evaluate_v2_request(
        {
            "action_kind": FUTURE_SPEAK_CAPABILITY,
            "decision": "2026-05-10 13:00",
            "context_ref": "current episode",
            "semantic_goal": "在约定时间开始补偿考核。",
            "reason": "用户要求在未来时间开始补偿考核。",
            "target_roles": [],
            "evidence_handles": ["e1"],
            "scheduled_authority_proposal": _proposal(),
        },
        available_action_kinds={FUTURE_SPEAK_CAPABILITY},
    )

    assert evaluation["ok"] is True
    request = evaluation["request"]
    proposal = request["scheduled_authority_proposal"]
    assert proposal["schema_version"] == "scheduled_authority_proposal.v1"
    assert proposal["temporal_alignment"] == "aligned"
    assert proposal["authorized_content_summary"] == "在约定时间开始补偿考核。"
    assert proposal["authorized_detail_refs"] == _proposal()[
        "authorized_detail_refs"
    ]


@pytest.mark.asyncio
async def test_future_speak_creation_persists_independent_authority_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The durable accepted-task document carries a deep authority copy."""

    from kazusa_ai_chatbot.accepted_task import lifecycle
    from kazusa_ai_chatbot.accepted_task.models import (
        AcceptedTaskCreateRequest,
    )

    authority = _authority()
    captured: dict[str, object] = {}

    async def insert_or_get_active_accepted_task(
        task: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        del kwargs
        captured["task"] = task
        return {"status": "created", "task": task}

    monkeypatch.setattr(
        "kazusa_ai_chatbot.db.accepted_tasks."
        "insert_or_get_active_accepted_task",
        insert_or_get_active_accepted_task,
    )
    request: AcceptedTaskCreateRequest = {
        "task_kind": "future_speak",
        "semantic_objective": "在约定时间开始补偿考核。",
        "accepted_task_summary": "在约定时间开始补偿考核。",
        "goal_continuation_ref": None,
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "source_trigger_source": "user_message",
        "source_platform": "qq",
        "source_channel_id": "480386272",
        "source_channel_type": "group",
        "source_message_id": "227312230",
        "source_platform_bot_id": "qq-bot-001",
        "source_character_name": "Test Character",
        "requester_global_user_id": "global-user-001",
        "requester_platform_user_id": "qq-user-001",
        "requester_display_name": "Test User",
        "storage_timestamp_utc": "2026-05-09T21:00:00+00:00",
        "scheduled_future_speech_authority": dict(authority),
    }

    result = await lifecycle.create_or_return_active_accepted_task(request)

    assert result["status"] == "created"
    persisted_authority = captured["task"][
        "scheduled_future_speech_authority"
    ]
    assert persisted_authority == authority
    request["scheduled_future_speech_authority"]["semantic_objective"] = (
        "mutated objective"
    )
    assert persisted_authority["semantic_objective"] == "在约定时间开始补偿考核。"
    validate_scheduled_future_speech_authority(persisted_authority)
