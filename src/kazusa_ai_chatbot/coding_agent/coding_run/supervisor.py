"""Deterministic supervisor for durable coding-agent runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from kazusa_ai_chatbot.coding_agent.code_verifying import (
    verify_and_repair_code_change,
)
from kazusa_ai_chatbot.coding_agent.code_verifying.execution_planning import (
    derive_base_execution_plan,
    extract_additive_execution_specs,
    patch_artifact_digest,
    validate_execution_plan_binding,
)
from kazusa_ai_chatbot.coding_agent.code_patching.apply import (
    materialize_managed_candidate,
)
from kazusa_ai_chatbot.coding_agent.code_executing import execute_code_check
from kazusa_ai_chatbot.config import CODING_AGENT_PREFLIGHT_EXECUTION
from kazusa_ai_chatbot.coding_agent.coding_run.ledger import (
    RUN_SCHEMA_VERSION,
    CodingRunPaths,
    allowed_next_actions,
    append_event,
    build_run_paths,
    empty_response,
    load_events,
    load_ledger,
    new_run_id,
    public_response,
    redaction_roots_from_ledger,
    utc_timestamp,
    write_ledger,
)
from kazusa_ai_chatbot.coding_agent.coding_run.locking import (
    LOCK_TIMEOUT_SECONDS,
    acquire_workspace_locks,
    build_lock_keys,
)
from kazusa_ai_chatbot.coding_agent.coding_run.models import (
    CodingRunContinueRequest,
    CodingRunGetRequest,
    CodingRunLedger,
    CodingRunResponse,
    CodingRunStartRequest,
)
from kazusa_ai_chatbot.coding_agent.supervisor import (
    answer_code_question,
    propose_code_change,
)

ALLOWED_OBJECTIVES = {"read_only", "propose_patch", "verify_repair"}
ALLOWED_ACTIONS = {
    "revise_proposal",
    "summarize",
    "status",
    "approve_and_verify",
    "respond_to_blocker",
    "cancel",
}
TERMINAL_STATUSES = {"completed", "rejected", "failed", "cancelled"}
SOURCE_FIELDS = (
    "source_url",
    "repo_url",
    "repo_hint",
    "local_root_hint",
    "local_path_hint",
    "requested_ref",
    "source_scope_hint",
    "workspace_root",
    "inline_sources",
)
COMMON_DIRECT_FIELDS = (
    *SOURCE_FIELDS,
    "question",
    "preferred_language",
    "max_answer_chars",
    "max_artifact_chars",
    "session_id",
)
ALLOWED_EXECUTION_TOOLS = {"python_compileall", "pytest"}
CODING_RUN_LOCK_TIMEOUT_SECONDS = LOCK_TIMEOUT_SECONDS
LOCK_SOURCE_FIELDS = (
    "source_url",
    "repo_url",
    "repo_hint",
    "local_root_hint",
    "local_path_hint",
    "requested_ref",
    "source_scope_hint",
    "inline_sources",
)


async def start_coding_run(
    request: CodingRunStartRequest,
) -> CodingRunResponse:
    """Start one durable run through a closed objective type."""

    validation_error = _start_validation_error(request)
    if validation_error:
        response = empty_response(
            status="rejected",
            objective_type=_request_text(request.get("objective_type")),
            goal=_request_text(request.get("question")),
            limitation=validation_error,
            trace_summary=["coding_run:rejected:start_request"],
        )
        return response

    objective_type = str(request["objective_type"])
    run_id = new_run_id()
    paths_result = build_run_paths(
        workspace_root_text=str(request["workspace_root"]),
        run_id=run_id,
        create=True,
        create_run_dir=False,
    )
    if isinstance(paths_result, str):
        response = empty_response(
            status="rejected",
            objective_type=objective_type,
            goal=_request_text(request.get("question")),
            limitation=paths_result,
            trace_summary=["coding_run:rejected:workspace"],
        )
        return response

    source_identity = _source_identity_from_request(request)
    lock_keys = build_lock_keys(
        run_id=run_id,
        source_identity=source_identity,
    )
    async with acquire_workspace_locks(
        workspace_root=paths_result.workspace_root,
        keys=lock_keys,
        timeout_seconds=CODING_RUN_LOCK_TIMEOUT_SECONDS,
    ) as acquired:
        if not acquired:
            response = _busy_response_without_ledger(
                run_id=run_id,
                objective_type=objective_type,
                goal=_request_text(request.get("question")),
            )
            return response
        locked_paths = build_run_paths(
            workspace_root_text=str(request["workspace_root"]),
            run_id=run_id,
            create=True,
        )
        if isinstance(locked_paths, str):
            response = empty_response(
                status="rejected",
                objective_type=objective_type,
                goal=_request_text(request.get("question")),
                limitation=locked_paths,
                trace_summary=["coding_run:rejected:workspace"],
            )
            return response
        ledger = _initial_ledger(
            run_id=run_id,
            request=request,
            objective_type=objective_type,
        )
        write_ledger(locked_paths, ledger)
        _record_event(
            paths=locked_paths,
            ledger=ledger,
            event_type="run_created",
            summary="Coding run was created.",
            public_payload={"objective_type": objective_type},
        )

        if objective_type == "read_only":
            response = await _start_read_only(paths=locked_paths, ledger=ledger)
            return response
        if objective_type == "propose_patch":
            response = await _start_propose_patch(paths=locked_paths, ledger=ledger)
            return response

        response = await _start_verify_repair(
            paths=locked_paths,
            ledger=ledger,
            request=request,
        )
        return response


async def continue_coding_run(
    request: CodingRunContinueRequest,
) -> CodingRunResponse:
    """Continue one existing durable run through a closed action."""

    paths_or_response = _paths_for_existing_run(request)
    if isinstance(paths_or_response, dict):
        return paths_or_response
    paths = paths_or_response
    ledger_or_error = load_ledger(paths)
    if isinstance(ledger_or_error, str):
        response = empty_response(
            status="rejected",
            run_id=_request_text(request.get("run_id")),
            limitation=ledger_or_error,
            trace_summary=["coding_run:rejected:missing_run"],
        )
        return response
    ledger = ledger_or_error

    action_error = _action_validation_error(request)
    action = _request_text(request.get("action"))
    if not action_error and action == "status":
        response = _status_run(paths=paths, ledger=ledger)
        return response
    source_identity = _source_identity_from_ledger(ledger)
    lock_keys = build_lock_keys(
        run_id=ledger["run_id"],
        source_identity=source_identity,
    )
    async with acquire_workspace_locks(
        workspace_root=paths.workspace_root,
        keys=lock_keys,
        timeout_seconds=CODING_RUN_LOCK_TIMEOUT_SECONDS,
    ) as acquired:
        if not acquired:
            response = _busy_response(paths=paths, ledger=ledger)
            return response
        locked_ledger_or_error = load_ledger(paths)
        if isinstance(locked_ledger_or_error, str):
            response = empty_response(
                status="rejected",
                run_id=_request_text(request.get("run_id")),
                limitation=locked_ledger_or_error,
                trace_summary=["coding_run:rejected:missing_run"],
            )
            return response
        locked_ledger = locked_ledger_or_error
        if action_error:
            response = _reject_continuation(
                paths=paths,
                ledger=locked_ledger,
                limitation=action_error,
            )
            return response
        if action not in allowed_next_actions(locked_ledger):
            response = _reject_continuation(
                paths=paths,
                ledger=locked_ledger,
                limitation="Coding run action is not currently allowed.",
            )
            return response
        if locked_ledger["status"] in TERMINAL_STATUSES and action != "summarize":
            response = _reject_continuation(
                paths=paths,
                ledger=locked_ledger,
                limitation="Coding run is already terminal.",
            )
            return response
        if action == "cancel":
            response = _cancel_run(
                paths=paths,
                ledger=locked_ledger,
                request=request,
            )
            return response
        if action == "revise_proposal":
            response = await _revise_proposal(
                paths=paths,
                ledger=locked_ledger,
                request=request,
            )
            return response
        if action == "summarize":
            response = _summarize_run(
                paths=paths,
                ledger=locked_ledger,
                request=request,
            )
            return response
        if action == "respond_to_blocker":
            response = await _respond_to_blocker(
                paths=paths,
                ledger=locked_ledger,
                request=request,
            )
            return response

        response = await _approve_and_verify(
            paths=paths,
            ledger=locked_ledger,
            request=request,
        )
        return response


async def get_coding_run(
    request: CodingRunGetRequest,
) -> CodingRunResponse:
    """Return the public projection for one durable coding-agent run."""

    paths_or_response = _paths_for_existing_run(request)
    if isinstance(paths_or_response, dict):
        return paths_or_response
    paths = paths_or_response
    ledger_or_error = load_ledger(paths)
    if isinstance(ledger_or_error, str):
        response = empty_response(
            status="rejected",
            run_id=_request_text(request.get("run_id")),
            limitation=ledger_or_error,
            trace_summary=["coding_run:rejected:missing_run"],
        )
        return response
    events = load_events(paths)
    response = public_response(ledger=ledger_or_error, events=events)
    return response


async def _start_read_only(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
) -> CodingRunResponse:
    read_request = _direct_request(ledger["source_request"])
    result = await answer_code_question(read_request)
    ledger["status"] = _terminal_status_from_direct(result["status"])
    ledger["answer_text"] = result["answer_text"]
    ledger["repository"] = result["repository"]
    ledger["source_scope"] = result["source_scope"]
    ledger["evidence"] = result["evidence"]
    ledger["limitations"] = result["limitations"]
    ledger["trace_summary"] = result["trace_summary"]
    if result["repository"] is not None:
        _record_event(
            paths=paths,
            ledger=ledger,
            event_type="source_resolved",
            summary="Source was resolved for the read-only run.",
            public_payload={"repository": result["repository"]},
        )
    if result["evidence"]:
        _record_event(
            paths=paths,
            ledger=ledger,
            event_type="evidence_collected",
            summary="Read-only evidence was collected.",
            public_payload={"evidence_count": len(result["evidence"])},
        )
    _record_terminal_event(paths=paths, ledger=ledger)
    write_ledger(paths, ledger)
    events = load_events(paths)
    response = public_response(ledger=ledger, events=events)
    return response


async def _start_propose_patch(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
) -> CodingRunResponse:
    write_request = _direct_request(ledger["source_request"])
    result = await propose_code_change(write_request)
    if result["status"] == "succeeded":
        ledger["status"] = "awaiting_approval"
    else:
        ledger["status"] = _terminal_status_from_direct(result["status"])
    ledger["answer_text"] = result["answer_text"]
    ledger["repository"] = result["repository"]
    ledger["source_scope"] = result["source_scope"]
    ledger["evidence"] = result["evidence"]
    ledger["patch_artifacts"] = result["patch_artifacts"]
    ledger["created_files"] = result["created_files"]
    ledger["changed_files"] = result["changed_files"]
    ledger["alignment"] = _mapping_or_none(result.get("alignment"))
    ledger["limitations"] = result["limitations"]
    ledger["trace_summary"] = result["trace_summary"]
    if result["status"] == "succeeded":
        _bind_proposal_and_preflight(paths=paths, ledger=ledger, result=result)
    if result["repository"] is not None:
        _record_event(
            paths=paths,
            ledger=ledger,
            event_type="source_resolved",
            summary="Source was resolved for the proposal run.",
            public_payload={"repository": result["repository"]},
        )
    if result["evidence"]:
        _record_event(
            paths=paths,
            ledger=ledger,
            event_type="evidence_collected",
            summary="Proposal evidence was collected.",
            public_payload={"evidence_count": len(result["evidence"])},
        )
    if ledger["status"] == "awaiting_approval":
        _record_event(
            paths=paths,
            ledger=ledger,
            event_type="proposal_ready",
            summary="Patch proposal is ready for approval.",
            public_payload={
                "created_files": result["created_files"],
                "changed_files": result["changed_files"],
                "alignment": result.get("alignment"),
            },
        )
        _record_event(
            paths=paths,
            ledger=ledger,
            event_type="awaiting_approval",
            summary="Coding run is waiting for structured approval.",
            public_payload={"patch_artifact_count": len(result["patch_artifacts"])},
        )
    else:
        _record_terminal_event(paths=paths, ledger=ledger)
    write_ledger(paths, ledger)
    events = load_events(paths)
    response = public_response(ledger=ledger, events=events)
    return response


async def _start_verify_repair(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
    request: CodingRunStartRequest,
) -> CodingRunResponse:
    validation_error = _verify_request_validation_error(request)
    if validation_error:
        ledger["status"] = "rejected"
        ledger["limitations"] = [validation_error]
        ledger["blockers"] = [_blocker(validation_error)]
        _record_event(
            paths=paths,
            ledger=ledger,
            event_type="rejected",
            summary=validation_error,
            public_payload={},
        )
        write_ledger(paths, ledger)
        events = load_events(paths)
        response = public_response(ledger=ledger, events=events)
        return response
    _record_approval(paths=paths, ledger=ledger, approval=request["approval"])
    verify_request = _verify_request_from_start(request)
    result = await verify_and_repair_code_change(verify_request)
    _record_verify_result(paths=paths, ledger=ledger, result=result)
    write_ledger(paths, ledger)
    events = load_events(paths)
    response = public_response(ledger=ledger, events=events)
    return response


def _cancel_run(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
    request: CodingRunContinueRequest,
) -> CodingRunResponse:
    ledger["status"] = "cancelled"
    reason = _request_text(request.get("reason"))
    _record_event(
        paths=paths,
        ledger=ledger,
        event_type="cancelled",
        summary="Coding run was cancelled.",
        public_payload={"reason": reason},
    )
    write_ledger(paths, ledger)
    events = load_events(paths)
    response = public_response(ledger=ledger, events=events)
    return response


async def _revise_proposal(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
    request: CodingRunContinueRequest,
) -> CodingRunResponse:
    if ledger["status"] != "awaiting_approval":
        response = _reject_continuation(
            paths=paths,
            ledger=ledger,
            limitation="Proposal revision is only valid for runs awaiting approval.",
        )
        return response
    revision_instruction = _request_text(request.get("revision_instruction"))
    if not revision_instruction:
        revision_instruction = _request_text(request.get("reason"))
    if not revision_instruction:
        response = _reject_continuation(
            paths=paths,
            ledger=ledger,
            limitation="Proposal revision requires an instruction.",
        )
        return response

    revise_request = _revision_proposal_request(
        paths=paths,
        ledger=ledger,
        revision_instruction=revision_instruction,
    )
    result = await propose_code_change(revise_request)
    if result["status"] == "succeeded":
        ledger["status"] = "awaiting_approval"
    else:
        ledger["status"] = _terminal_status_from_direct(result["status"])
    ledger["answer_text"] = result["answer_text"]
    ledger["repository"] = result["repository"]
    ledger["source_scope"] = result["source_scope"]
    ledger["evidence"] = result["evidence"]
    ledger["patch_artifacts"] = result["patch_artifacts"]
    ledger["created_files"] = result["created_files"]
    ledger["changed_files"] = result["changed_files"]
    ledger["alignment"] = _mapping_or_none(result.get("alignment"))
    ledger["limitations"] = result["limitations"]
    ledger["trace_summary"] = result["trace_summary"]
    if result["status"] == "succeeded":
        _bind_proposal_and_preflight(paths=paths, ledger=ledger, result=result)
    if ledger["status"] == "awaiting_approval":
        _record_event(
            paths=paths,
            ledger=ledger,
            event_type="proposal_revised",
            summary="Patch proposal was revised.",
            public_payload={
                "created_files": result["created_files"],
                "changed_files": result["changed_files"],
                "alignment": result.get("alignment"),
                "revision_instruction": revision_instruction,
            },
        )
    else:
        _record_terminal_event(paths=paths, ledger=ledger)
    write_ledger(paths, ledger)
    events = load_events(paths)
    response = public_response(ledger=ledger, events=events)
    return response


def _summarize_run(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
    request: CodingRunContinueRequest,
) -> CodingRunResponse:
    reason = _request_text(request.get("reason"))
    _record_event(
        paths=paths,
        ledger=ledger,
        event_type="summary_requested",
        summary="Coding run summary was requested.",
        public_payload={"reason": reason},
    )
    write_ledger(paths, ledger)
    events = load_events(paths)
    response = public_response(ledger=ledger, events=events)
    response["answer_text"] = _summary_answer_text(response)
    return response


def _status_run(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
) -> CodingRunResponse:
    events = load_events(paths)
    response = public_response(ledger=ledger, events=events)
    return response


async def _respond_to_blocker(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
    request: CodingRunContinueRequest,
) -> CodingRunResponse:
    """Resume the one active blocker through its stored deterministic target.

    Args:
        paths: Resolved workspace paths for the durable run.
        ledger: Current durable run state with exactly one active blocker.
        request: Validated user-follow-up continuation request.

    Returns:
        The refreshed public run projection after the authorized resume path.
    """

    blocker = _open_blocker(ledger)
    if blocker is None:
        response = _reject_continuation(
            paths=paths,
            ledger=ledger,
            limitation="Coding run has no active blocker to answer.",
        )
        return response
    answer = _request_text(request.get("revision_instruction"))
    if not answer:
        response = _reject_continuation(
            paths=paths,
            ledger=ledger,
            limitation="Coding run blocker response requires a user answer.",
        )
        return response
    resume_target = _request_text(blocker.get("resume_target"))
    if resume_target == "replan_proposal":
        blocker["status"] = "answered"
        blocker["answered_at"] = utc_timestamp()
        ledger["status"] = "awaiting_approval"
        response = await _revise_proposal(
            paths=paths,
            ledger=ledger,
            request={
                "workspace_root": request["workspace_root"],
                "run_id": request["run_id"],
                "action": "revise_proposal",
                "revision_instruction": answer,
            },
        )
        return response
    if resume_target == "retry_verification":
        response = await _retry_blocked_verification(
            paths=paths,
            ledger=ledger,
            blocker=blocker,
        )
        return response
    response = _reject_continuation(
        paths=paths,
        ledger=ledger,
        limitation="Coding run blocker cannot be resumed.",
    )
    return response


async def _retry_blocked_verification(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
    blocker: dict[str, object],
) -> CodingRunResponse:
    """Rerun the stored verification plan without creating repair work.

    Args:
        paths: Resolved workspace paths for the durable run.
        ledger: Blocked durable run holding the prior verified proposal.
        blocker: The active environment blocker being answered.

    Returns:
        The public response after one repair-free verification retry.
    """

    approval = _latest_approval(ledger)
    if approval is None:
        response = _reject_continuation(
            paths=paths,
            ledger=ledger,
            limitation="Blocked verification has no stored approval evidence.",
        )
        return response
    blocker["status"] = "answered"
    blocker["answered_at"] = utc_timestamp()
    ledger["status"] = "verifying"
    verify_request = _verify_request_from_continuation(
        ledger=ledger,
        request={
            "workspace_root": str(paths.workspace_root),
            "run_id": ledger["run_id"],
            "action": "approve_and_verify",
            "approval": approval,
            "repair_attempt_limit": 0,
        },
    )
    verify_request["repair_attempt_limit"] = 0
    result = await verify_and_repair_code_change(verify_request)
    _record_verify_result(paths=paths, ledger=ledger, result=result)
    write_ledger(paths, ledger)
    events = load_events(paths)
    response = public_response(ledger=ledger, events=events)
    return response


async def _approve_and_verify(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
    request: CodingRunContinueRequest,
) -> CodingRunResponse:
    if ledger["status"] != "awaiting_approval":
        response = _reject_continuation(
            paths=paths,
            ledger=ledger,
            limitation="Approval is only valid for runs awaiting approval.",
        )
        return response
    validation_error = _approval_continue_validation_error(request)
    if validation_error:
        response = _reject_continuation(
            paths=paths,
            ledger=ledger,
            limitation=validation_error,
        )
        return response
    plan_error = _stored_plan_binding_error(ledger)
    if plan_error:
        response = _reject_continuation(
            paths=paths,
            ledger=ledger,
            limitation=plan_error,
        )
        return response
    _record_approval(paths=paths, ledger=ledger, approval=request["approval"])
    ledger["status"] = "verifying"
    verify_request = _verify_request_from_continuation(
        ledger=ledger,
        request=request,
    )
    execution_request = _request_text(request.get("execution_request"))
    if execution_request:
        candidate_root = _planning_root(paths=paths, ledger=ledger)
        extra_specs = await extract_additive_execution_specs(
            user_request=execution_request,
            candidate_root=candidate_root,
        )
        if extra_specs:
            base_specs = verify_request["execution_specs"]
            verify_request["execution_specs"] = [*base_specs, *extra_specs]
    result = await verify_and_repair_code_change(verify_request)
    _record_verify_result(paths=paths, ledger=ledger, result=result)
    write_ledger(paths, ledger)
    events = load_events(paths)
    response = public_response(ledger=ledger, events=events)
    return response


def _reject_continuation(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
    limitation: str,
) -> CodingRunResponse:
    _record_event(
        paths=paths,
        ledger=ledger,
        event_type="rejected",
        summary=limitation,
        public_payload={},
    )
    write_ledger(paths, ledger)
    response_ledger = dict(ledger)
    response_ledger["status"] = "rejected"
    response_ledger["limitations"] = [limitation]
    response_ledger["blockers"] = [_blocker(limitation)]
    events = load_events(paths)
    response = public_response(ledger=response_ledger, events=events)
    return response


def _busy_response(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
) -> CodingRunResponse:
    """Project retryable lock contention without changing a stored run."""

    events = load_events(paths)
    response = public_response(ledger=ledger, events=events)
    response["operation_outcome"] = "busy"
    response["retry_guidance"] = "Retry the same coding action."
    return response


def _busy_response_without_ledger(
    *,
    run_id: str,
    objective_type: str,
    goal: str,
) -> CodingRunResponse:
    """Project start-lock contention before a durable ledger exists."""

    response = empty_response(
        status="busy",
        run_id=run_id,
        objective_type=objective_type,
        goal=goal,
        limitation="Coding run is busy.",
        trace_summary=["coding_run:busy:start_lock"],
    )
    response["operation_outcome"] = "busy"
    response["retry_guidance"] = "Retry the same coding action."
    return response


def _record_verify_result(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
    result: Mapping[str, object],
) -> None:
    status = _terminal_status_from_verify(_request_text(result.get("status")))
    ledger["status"] = status
    ledger["answer_text"] = _request_text(result.get("answer_text"))
    ledger["repository"] = _mapping_or_none(result.get("repository"))
    ledger["source_scope"] = _mapping_or_none(result.get("source_scope"))
    ledger["attempts"] = _list_of_dicts(result.get("attempts"))
    ledger["repair_attempts"] = ledger["attempts"][1:]
    ledger["patch_artifacts"] = _list_of_dicts(result.get("final_patch_artifacts"))
    final_created_files = _list_of_dicts(result.get("final_created_files"))
    if final_created_files:
        ledger["created_files"] = final_created_files
    ledger["changed_files"] = _list_of_dicts(result.get("final_changed_files"))
    final_alignment = _mapping_or_none(result.get("alignment"))
    if final_alignment is not None:
        ledger["alignment"] = final_alignment
    ledger["apply_attempts"] = _apply_attempts(result)
    ledger["execution_attempts"] = _execution_attempts(result)
    ledger["blockers"] = _durable_blockers(result.get("blockers"))
    ledger["limitations"] = _string_list(result.get("limitations"))
    ledger["trace_summary"] = _string_list(result.get("trace_summary"))
    for attempt in ledger["attempts"]:
        _record_event(
            paths=paths,
            ledger=ledger,
            event_type="apply_attempt_recorded",
            summary="Apply attempt was recorded.",
            public_payload={"attempt": attempt},
        )
        _record_event(
            paths=paths,
            ledger=ledger,
            event_type="execution_attempt_recorded",
            summary="Execution attempt was recorded.",
            public_payload={"attempt": attempt},
        )
    for attempt in ledger["repair_attempts"]:
        _record_event(
            paths=paths,
            ledger=ledger,
            event_type="repair_attempt_recorded",
            summary="Repair attempt was recorded.",
            public_payload={"attempt": attempt},
        )
    _record_terminal_event(paths=paths, ledger=ledger)


def _bind_proposal_and_preflight(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
    result: Mapping[str, object],
) -> None:
    """Bind the current proposal to one deterministic plan and candidate run."""

    patch_artifacts = ledger["patch_artifacts"]
    ledger["proposal_revision"] += 1
    artifact_digest = patch_artifact_digest(patch_artifacts)
    ledger["patch_artifact_digest"] = artifact_digest
    source_identity = _proposal_source_identity(ledger)
    candidate_baseline = _candidate_baseline(result)
    execution_plan = derive_base_execution_plan(
        candidate_root=_planning_root(paths=paths, ledger=ledger),
        patch_artifacts=patch_artifacts,
        run_id=ledger["run_id"],
        source_identity=source_identity,
        proposal_revision=ledger["proposal_revision"],
    )
    ledger["execution_plan"] = execution_plan
    candidate_response = _materialize_preflight_candidate(
        paths=paths,
        ledger=ledger,
        source_identity=source_identity,
        candidate_baseline=candidate_baseline,
    )
    if candidate_response is None:
        ledger["preflight"] = {
            "preflight_enabled": False,
            "execution_backend": "managed_copy_process",
            "status": "disabled",
        }
        return
    if candidate_response["status"] != "succeeded":
        ledger["status"] = "failed"
        ledger["preflight"] = {
            "preflight_enabled": True,
            "execution_backend": "managed_copy_process",
            "status": "failed",
            "limitations": candidate_response["limitations"],
        }
        ledger["limitations"].extend(candidate_response["limitations"])
        return
    execution_results = _run_preflight_specs(
        workspace_root=paths.workspace_root,
        candidate_response=candidate_response,
        execution_plan=execution_plan,
    )
    statuses = [result["status"] for result in execution_results]
    ledger["apply_attempts"] = [{
        "apply_package_id": candidate_response["apply_package_id"],
        "status": candidate_response["status"],
    }]
    ledger["execution_attempts"] = execution_results
    preflight_status = "passed" if all(status == "succeeded" for status in statuses) else "failed"
    ledger["preflight"] = {
        "preflight_enabled": True,
        "execution_backend": "managed_copy_process",
        "authorization_purpose": "preapproval_preflight",
        "status": preflight_status,
        "apply_package_id": candidate_response["apply_package_id"],
        "execution_plan_id": execution_plan["plan_id"],
    }
    if preflight_status == "failed":
        ledger["status"] = "failed"
        ledger["limitations"].append("Preapproval verification did not succeed.")


def _materialize_preflight_candidate(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
    source_identity: dict[str, object],
    candidate_baseline: str,
) -> dict[str, object] | None:
    """Materialize the configured managed candidate without human approval."""

    if not CODING_AGENT_PREFLIGHT_EXECUTION:
        return None
    request: dict[str, object] = {
        "workspace_root": str(paths.workspace_root),
        "source_identity": source_identity,
        "expected_source_identity": source_identity,
        "patch_artifacts": ledger["patch_artifacts"],
        "candidate_baseline": candidate_baseline,
        "authorization_purpose": "preapproval_preflight",
    }
    repository = ledger["repository"]
    if candidate_baseline == "resolved_source" and isinstance(repository, Mapping):
        source_root = _resolved_source_root_from_ledger(ledger)
        if source_root is not None:
            request["source_root"] = str(source_root)
    response = materialize_managed_candidate(request)
    return response


def _proposal_source_identity(ledger: CodingRunLedger) -> dict[str, object]:
    """Build the stable source identity stored with the current proposal."""

    repository = ledger["repository"]
    if isinstance(repository, Mapping):
        identity = _identity_from_repository(repository)
        return identity
    identity = {
        "provider": "generated",
        "owner": None,
        "repo": None,
        "current_commit": f"artifact-sha256:{ledger['patch_artifact_digest']}",
        "dirty_state": "clean",
    }
    return identity


def _candidate_baseline(result: Mapping[str, object]) -> str:
    """Choose the canonical candidate baseline from proposal ownership."""

    if result.get("mode") == "source_free":
        return "empty_source_free"
    return "resolved_source"


def _planning_root(*, paths: CodingRunPaths, ledger: CodingRunLedger) -> Path:
    """Select existing source evidence for planning before optional preflight."""

    repository = ledger["repository"]
    if isinstance(repository, Mapping):
        source_root = _resolved_source_root_from_ledger(ledger)
        if source_root is not None:
            return source_root
    return paths.workspace_root


def _resolved_source_root_from_ledger(ledger: CodingRunLedger) -> Path | None:
    """Recover the private local checkout root from the stored source request."""

    source_request = ledger["source_request"]
    root_hint = source_request.get("local_root_hint")
    path_hint = source_request.get("local_path_hint")
    hint = root_hint if isinstance(root_hint, str) and root_hint else path_hint
    if not isinstance(hint, str) or not hint:
        return None
    path = Path(hint).expanduser().resolve(strict=False)
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    if path.is_dir():
        return path
    return None


def _run_preflight_specs(
    *,
    workspace_root: Path,
    candidate_response: Mapping[str, object],
    execution_plan: Mapping[str, object],
) -> list[dict[str, object]]:
    """Run the bound deterministic plan through the existing executor allowlist."""

    specs_value = execution_plan.get("base_specs")
    if not isinstance(specs_value, list):
        return []
    execution_results: list[dict[str, object]] = []
    for spec in specs_value:
        if not isinstance(spec, Mapping):
            continue
        result = execute_code_check({
            "workspace_root": str(workspace_root),
            "apply_package_id": candidate_response["apply_package_id"],
            "apply_workspace_ref": candidate_response["apply_workspace_ref"],
            "execution": dict(spec),
        })
        execution_results.append(result)
    return execution_results


def _record_approval(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
    approval: Mapping[str, object],
) -> None:
    approval_record = {
        "approved": approval.get("approved"),
        "approved_by": approval.get("approved_by"),
        "approved_at": approval.get("approved_at"),
        "approval_reason": approval.get("approval_reason"),
        "approval_evidence": approval.get("approval_evidence"),
    }
    ledger["approvals"].append(approval_record)
    _record_event(
        paths=paths,
        ledger=ledger,
        event_type="approval_received",
        summary="Structured approval was received.",
        public_payload={"approval": approval_record},
    )


def _record_terminal_event(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
) -> None:
    event_type = ledger["status"]
    if event_type not in ("completed", "blocked", "rejected", "failed", "cancelled"):
        event_type = "failed"
    _record_event(
        paths=paths,
        ledger=ledger,
        event_type=event_type,
        summary=f"Coding run reached {event_type}.",
        public_payload={"status": ledger["status"]},
    )


def _record_event(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
    event_type: str,
    summary: str,
    public_payload: dict[str, object],
) -> None:
    append_event(
        paths=paths,
        run_id=ledger["run_id"],
        event_type=event_type,
        status=ledger["status"],
        summary=summary,
        public_payload=public_payload,
        redaction_roots=redaction_roots_from_ledger(ledger),
    )


def _initial_ledger(
    *,
    run_id: str,
    request: CodingRunStartRequest,
    objective_type: str,
) -> CodingRunLedger:
    now = utc_timestamp()
    source_request = _source_request_from_start(request)
    ledger: CodingRunLedger = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "created",
        "goal": _request_text(request.get("question")),
        "objective_type": objective_type,
        "created_at": now,
        "updated_at": now,
        "source_request": source_request,
        "repository": None,
        "source_scope": None,
        "answer_text": "",
        "evidence": [],
        "patch_artifacts": [],
        "created_files": [],
        "changed_files": [],
        "alignment": None,
        "approvals": [],
        "apply_attempts": [],
        "execution_attempts": [],
        "repair_attempts": [],
        "attempts": [],
        "blockers": [],
        "limitations": [],
        "trace_summary": [],
        "proposal_revision": 0,
        "patch_artifact_digest": "",
        "execution_plan": None,
        "preflight": {},
    }
    return ledger


def _source_request_from_start(
    request: CodingRunStartRequest,
) -> dict[str, object]:
    source_request: dict[str, object] = {}
    for field_name in COMMON_DIRECT_FIELDS:
        value = request.get(field_name)
        if value is not None:
            source_request[field_name] = value
    return source_request


def _source_identity_from_request(
    request: Mapping[str, object],
) -> dict[str, object] | None:
    """Build a stable explicit source identity for a start-time workspace lock."""

    identity: dict[str, object] = {}
    for field_name in LOCK_SOURCE_FIELDS:
        value = request.get(field_name)
        if isinstance(value, str):
            normalized_value = value.strip()
            if normalized_value:
                identity[field_name] = normalized_value
            continue
        if value is not None:
            identity[field_name] = value
    if not identity:
        return None
    return identity


def _source_identity_from_ledger(
    ledger: CodingRunLedger,
) -> dict[str, object] | None:
    """Return the persisted repository or source identity for a continuation."""

    repository = ledger["repository"]
    if isinstance(repository, Mapping):
        identity = _identity_from_repository(repository)
        if identity:
            return identity
    identity = _source_identity_from_request(ledger["source_request"])
    return identity


def _direct_request(source_request: Mapping[str, object]) -> dict[str, object]:
    request = {
        field_name: source_request[field_name]
        for field_name in COMMON_DIRECT_FIELDS
        if field_name in source_request
    }
    return request


def _verify_request_from_start(
    request: CodingRunStartRequest,
) -> dict[str, object]:
    verify_request = _source_request_from_start(request)
    for field_name in (
        "approval",
        "execution_specs",
        "repair_attempt_limit",
        "initial_patch_artifacts",
        "expected_source_identity",
    ):
        value = request.get(field_name)
        if value is not None:
            verify_request[field_name] = value
    return verify_request


def _verify_request_from_continuation(
    *,
    ledger: CodingRunLedger,
    request: CodingRunContinueRequest,
) -> dict[str, object]:
    verify_request = _direct_request(ledger["source_request"])
    verify_request["approval"] = request["approval"]
    execution_specs = request.get("execution_specs")
    if isinstance(execution_specs, list) and execution_specs:
        verify_request["execution_specs"] = execution_specs
    else:
        execution_plan = ledger["execution_plan"]
        if isinstance(execution_plan, Mapping):
            base_specs = execution_plan.get("base_specs")
            if isinstance(base_specs, list):
                verify_request["execution_specs"] = base_specs
    verify_request["initial_patch_artifacts"] = ledger["patch_artifacts"]
    repository = ledger.get("repository")
    if isinstance(repository, Mapping):
        verify_request["expected_source_identity"] = _identity_from_repository(
            repository,
        )
    else:
        verify_request["expected_source_identity"] = _proposal_source_identity(
            ledger,
        )
    repair_attempt_limit = request.get("repair_attempt_limit")
    if repair_attempt_limit is not None:
        verify_request["repair_attempt_limit"] = repair_attempt_limit
    return verify_request


def _revision_proposal_request(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
    revision_instruction: str,
) -> dict[str, object]:
    revise_request = _direct_request(ledger["source_request"])
    prior_projection = _prior_revision_projection(paths=paths, ledger=ledger)
    revise_request["question"] = (
        "Original goal:\n"
        f"{ledger['goal']}\n\n"
        "Current revision instruction:\n"
        f"{revision_instruction}\n\n"
        "Prior public run projection:\n"
        f"{json.dumps(prior_projection, ensure_ascii=False, sort_keys=True)}"
    )
    prior_artifacts = _prior_generated_artifacts_from_ledger(ledger)
    if prior_artifacts:
        revise_request["prior_generated_artifacts"] = prior_artifacts
    return revise_request


def _prior_generated_artifacts_from_ledger(
    ledger: CodingRunLedger,
) -> list[dict[str, str]]:
    created_roles = _created_file_roles(ledger["created_files"])
    if not created_roles:
        return []

    artifacts: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for patch_artifact in ledger["patch_artifacts"]:
        if not isinstance(patch_artifact, Mapping):
            continue
        diff_text = _request_text(patch_artifact.get("diff_text"))
        contents = _created_file_contents_from_diff(
            diff_text=diff_text,
            created_paths=set(created_roles),
        )
        for path, content in contents.items():
            if path in seen_paths:
                continue
            seen_paths.add(path)
            artifacts.append({
                "artifact_id": _prior_artifact_id(path),
                "file_label": PurePosixPath(path).name,
                "file_kind": _file_kind_for_prior_artifact(path),
                "content_format": _content_format_for_prior_artifact(path),
                "path": path,
                "content": content,
                "purpose": created_roles[path],
            })
    return artifacts


def _created_file_roles(
    created_files: list[dict[str, object]],
) -> dict[str, str]:
    roles: dict[str, str] = {}
    for row in created_files:
        if not isinstance(row, Mapping):
            continue
        path = _normalized_patch_path(_request_text(row.get("path")))
        role = _request_text(row.get("role"))
        if not path:
            continue
        if not role:
            role = "Prior generated artifact from stored source-free proposal."
        roles[path] = role
    return roles


def _created_file_contents_from_diff(
    *,
    diff_text: str,
    created_paths: set[str],
) -> dict[str, str]:
    contents: dict[str, str] = {}
    current_path = ""
    current_lines: list[str] = []
    in_hunk = False

    def flush_current_file() -> None:
        if current_path and current_path in created_paths:
            contents[current_path] = "\n".join(current_lines) + "\n"

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            flush_current_file()
            current_path = ""
            current_lines = []
            in_hunk = False
            continue
        if line.startswith("+++ "):
            flush_current_file()
            current_path = _diff_target_path(line[4:])
            current_lines = []
            in_hunk = False
            continue
        if not current_path:
            continue
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            current_lines.append(line[1:])
            continue
        if line.startswith(" "):
            current_lines.append(line[1:])

    flush_current_file()
    return contents


def _diff_target_path(path_text: str) -> str:
    path = path_text.strip()
    if path == "/dev/null":
        return ""
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return _normalized_patch_path(path)


def _normalized_patch_path(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def _prior_artifact_id(path: str) -> str:
    chars = [
        char.lower()
        if char.isalnum()
        else "_"
        for char in path
    ]
    artifact_id = "prior_" + "".join(chars).strip("_")
    return artifact_id


def _file_kind_for_prior_artifact(path: str) -> str:
    lower_path = path.casefold()
    basename = PurePosixPath(lower_path).name
    if lower_path.startswith("tests/") or basename.startswith("test_"):
        return "test"
    if lower_path.startswith("docs/") or lower_path.endswith(".md"):
        return "docs"
    if lower_path.endswith((".json", ".toml", ".yaml", ".yml", ".ini")):
        return "config"
    if lower_path.endswith((".csv", ".txt")):
        return "data"
    return "source"


def _content_format_for_prior_artifact(path: str) -> str:
    lower_path = path.casefold()
    if lower_path.endswith(".py"):
        return "python"
    if lower_path.endswith(".md"):
        return "markdown"
    if lower_path.endswith(".json"):
        return "json"
    if lower_path.endswith(".csv"):
        return "csv"
    return "text"


def _prior_revision_projection(
    *,
    paths: CodingRunPaths,
    ledger: CodingRunLedger,
) -> dict[str, object]:
    events = load_events(paths)
    projection = public_response(ledger=ledger, events=events)
    prior = {
        "status": projection["status"],
        "goal": projection["goal"],
        "objective_type": projection["objective_type"],
        "answer_text": projection["answer_text"],
        "repository": projection["repository"],
        "source_scope": projection["source_scope"],
        "created_files": projection["created_files"],
        "changed_files": projection["changed_files"],
        "alignment": projection["alignment"],
        "limitations": projection["limitations"],
        "blockers": projection["blockers"],
        "allowed_next_actions": projection["allowed_next_actions"],
    }
    return prior


def _summary_answer_text(response: CodingRunResponse) -> str:
    changed_paths: list[str] = []
    for row in response["changed_files"]:
        if not isinstance(row, Mapping):
            continue
        changed_path = _request_text(row.get("path"))
        if changed_path:
            changed_paths.append(changed_path)
    parts = [
        f"Status: {response['status']}",
    ]
    if changed_paths:
        parts.append(f"Changed files: {', '.join(changed_paths)}")
    if response["attempts"]:
        parts.append(f"Attempts: {len(response['attempts'])}")
    if response["limitations"]:
        parts.append(f"Limitations: {'; '.join(response['limitations'])}")
    if response["blockers"]:
        parts.append(f"Blockers: {len(response['blockers'])}")
    if response["allowed_next_actions"]:
        parts.append(
            "Allowed next actions: "
            f"{', '.join(response['allowed_next_actions'])}"
        )
    answer_text = "\n".join(parts)
    return answer_text


def _start_validation_error(request: Mapping[str, object]) -> str:
    workspace_root = request.get("workspace_root")
    if not isinstance(workspace_root, str) or not workspace_root.strip():
        return "Coding run requires a workspace root."
    objective_type = request.get("objective_type")
    if not isinstance(objective_type, str) or not objective_type.strip():
        return "Coding run requires an objective type."
    if objective_type not in ALLOWED_OBJECTIVES:
        return "Coding run objective type is unsupported."
    question = request.get("question")
    if not isinstance(question, str) or not question.strip():
        return "Coding run requires a question."
    return ""


def _action_validation_error(request: Mapping[str, object]) -> str:
    action = request.get("action")
    if not isinstance(action, str) or not action.strip():
        return "Coding run continuation requires an action."
    if action not in ALLOWED_ACTIONS:
        return "Coding run continuation action is unsupported."
    return ""


def _approval_continue_validation_error(request: Mapping[str, object]) -> str:
    approval_error = _approval_error(request.get("approval"))
    if approval_error:
        return approval_error
    execution_specs = request.get("execution_specs")
    if execution_specs is not None:
        execution_error = _execution_specs_error(execution_specs)
        if execution_error:
            return execution_error
    return ""


def _stored_plan_binding_error(ledger: CodingRunLedger) -> str:
    """Reject approval when the stored plan no longer matches the proposal."""

    execution_plan = ledger["execution_plan"]
    if not isinstance(execution_plan, Mapping):
        return "Coding run is missing a proposal-bound execution plan."
    error = validate_execution_plan_binding(
        plan=execution_plan,
        run_id=ledger["run_id"],
        source_identity=_proposal_source_identity(ledger),
        proposal_revision=ledger["proposal_revision"],
        patch_artifact_digest=ledger["patch_artifact_digest"],
    )
    return error


def _verify_request_validation_error(request: Mapping[str, object]) -> str:
    approval_error = _approval_error(request.get("approval"))
    if approval_error:
        return approval_error
    execution_error = _execution_specs_error(request.get("execution_specs"))
    if execution_error:
        return execution_error
    initial_artifacts = request.get("initial_patch_artifacts")
    if initial_artifacts is not None and not isinstance(initial_artifacts, list):
        return "Coding run initial patch artifacts must be a list."
    return ""


def _approval_error(approval_value: object) -> str:
    if not isinstance(approval_value, Mapping):
        return "Coding run requires structured approval."
    if approval_value.get("approved") is not True:
        return "Coding run approval requires approved=True."
    for key in ("approved_by", "approved_at", "approval_reason"):
        value = approval_value.get(key)
        if not isinstance(value, str) or not value.strip():
            return "Coding run approval is incomplete."
    return ""


def _execution_specs_error(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "Coding run requires at least one execution spec."
    for spec in value:
        if not isinstance(spec, Mapping):
            return "Coding run execution specs must be structured objects."
        tool = spec.get("tool")
        if tool not in ALLOWED_EXECUTION_TOOLS:
            return "Coding run execution tool is unsupported."
        if tool == "python_compileall":
            paths = spec.get("paths")
            if not isinstance(paths, list) or not paths:
                return "Coding run compile execution requires paths."
        if tool == "pytest":
            selectors = spec.get("pytest_selectors")
            if not isinstance(selectors, list) or not selectors:
                return "Coding run pytest execution requires selectors."
    return ""


def _paths_for_existing_run(
    request: Mapping[str, object],
) -> CodingRunPaths | CodingRunResponse:
    workspace_root = request.get("workspace_root")
    run_id = request.get("run_id")
    if not isinstance(workspace_root, str) or not workspace_root.strip():
        response = empty_response(
            status="rejected",
            run_id=_request_text(run_id),
            limitation="Coding run requires a workspace root.",
            trace_summary=["coding_run:rejected:workspace"],
        )
        return response
    if not isinstance(run_id, str) or not run_id.strip():
        response = empty_response(
            status="rejected",
            limitation="Coding run requires a run id.",
            trace_summary=["coding_run:rejected:run_id"],
        )
        return response
    paths_result = build_run_paths(
        workspace_root_text=workspace_root,
        run_id=run_id,
        create=False,
    )
    if isinstance(paths_result, str):
        response = empty_response(
            status="rejected",
            run_id=run_id,
            limitation=paths_result,
            trace_summary=["coding_run:rejected:path"],
        )
        return response
    return paths_result


def _terminal_status_from_direct(status: str) -> str:
    if status == "succeeded":
        return "completed"
    if status == "needs_user_input":
        return "blocked"
    if status == "rejected":
        return "rejected"
    return "failed"


def _terminal_status_from_verify(status: str) -> str:
    if status == "succeeded":
        return "completed"
    if status == "rejected":
        return "rejected"
    if status == "blocked":
        return "blocked"
    return "failed"


def _identity_from_repository(repository: Mapping[str, object]) -> dict[str, object]:
    identity_keys = ("provider", "owner", "repo", "current_commit", "dirty_state")
    identity = {
        key: repository[key]
        for key in identity_keys
        if key in repository
    }
    return identity


def _apply_attempts(result: Mapping[str, object]) -> list[dict[str, object]]:
    final_apply = result.get("final_apply")
    if not isinstance(final_apply, Mapping):
        return []
    attempt = {
        "status": final_apply.get("status"),
        "apply_package_id": final_apply.get("apply_package_id"),
        "applied_files": final_apply.get("applied_files"),
        "changed_files": final_apply.get("changed_files"),
        "validation": final_apply.get("validation"),
        "limitations": final_apply.get("limitations"),
        "trace_summary": final_apply.get("trace_summary"),
    }
    attempts = [attempt]
    return attempts


def _execution_attempts(result: Mapping[str, object]) -> list[dict[str, object]]:
    final_execution = result.get("final_execution")
    if not isinstance(final_execution, list):
        return []
    attempts: list[dict[str, object]] = []
    for item in final_execution:
        if not isinstance(item, Mapping):
            continue
        attempt = {
            "status": item.get("status"),
            "tool": item.get("tool"),
            "exit_code": item.get("exit_code"),
            "timed_out": item.get("timed_out"),
            "duration_ms": item.get("duration_ms"),
            "output_truncated": item.get("output_truncated"),
            "executed_paths": item.get("executed_paths"),
            "limitations": item.get("limitations"),
            "trace_summary": item.get("trace_summary"),
        }
        attempts.append(attempt)
    return attempts


def _mapping_or_none(value: object) -> dict[str, object] | None:
    if isinstance(value, Mapping):
        mapped = dict(value)
        return mapped
    return None


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    items = [
        dict(item)
        for item in value
        if isinstance(item, Mapping)
    ]
    return items


def _durable_blockers(value: object) -> list[dict[str, object]]:
    """Attach durable lifecycle fields to the verifier's one typed blocker."""

    blockers = _list_of_dicts(value)
    if len(blockers) > 1:
        raise ValueError("Coding run verification returned multiple blockers.")
    if not blockers:
        return []

    blocker = blockers[0]
    blocker["blocker_id"] = new_run_id()
    blocker["status"] = "open"
    blocker["created_at"] = utc_timestamp()
    blocker["answered_at"] = None
    durable_blockers = [blocker]
    return durable_blockers


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    strings = [
        item
        for item in value
        if isinstance(item, str)
    ]
    return strings


def _request_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return text


def _open_blocker(ledger: CodingRunLedger) -> dict[str, object] | None:
    """Return the one durable blocker whose answer may resume the run."""

    for blocker in ledger["blockers"]:
        if not isinstance(blocker, dict):
            continue
        if _request_text(blocker.get("status")) == "open":
            return blocker
    return None


def _latest_approval(ledger: CodingRunLedger) -> dict[str, object] | None:
    """Return the most recent stored approval needed for a verification retry."""

    if not ledger["approvals"]:
        return None
    approval = ledger["approvals"][-1]
    if not isinstance(approval, dict):
        return None
    latest_approval = dict(approval)
    return latest_approval


def _blocker(message: str) -> dict[str, object]:
    now = utc_timestamp()
    blocker = {
        "blocker_id": new_run_id(),
        "code": "request_rejected",
        "blocker_kind": "scope",
        "message": message,
        "question": message,
        "options": [],
        "resume_target": "none",
        "status": "open",
        "details": {},
        "created_at": now,
        "answered_at": None,
    }
    return blocker
