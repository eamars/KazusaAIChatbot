"""Deterministic supervisor for durable coding-agent runs."""

from __future__ import annotations

from collections.abc import Mapping

from kazusa_ai_chatbot.coding_agent.code_verifying import (
    verify_and_repair_code_change,
)
from kazusa_ai_chatbot.coding_agent.coding_run.ledger import (
    RUN_SCHEMA_VERSION,
    CodingRunPaths,
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
ALLOWED_ACTIONS = {"approve_and_verify", "cancel"}
TERMINAL_STATUSES = {"completed", "blocked", "rejected", "failed", "cancelled"}
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

    ledger = _initial_ledger(
        run_id=run_id,
        request=request,
        objective_type=objective_type,
    )
    write_ledger(paths_result, ledger)
    _record_event(
        paths=paths_result,
        ledger=ledger,
        event_type="run_created",
        summary="Coding run was created.",
        public_payload={"objective_type": objective_type},
    )

    if objective_type == "read_only":
        response = await _start_read_only(paths=paths_result, ledger=ledger)
        return response
    if objective_type == "propose_patch":
        response = await _start_propose_patch(paths=paths_result, ledger=ledger)
        return response

    response = await _start_verify_repair(
        paths=paths_result,
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
    if action_error:
        response = _reject_continuation(
            paths=paths,
            ledger=ledger,
            limitation=action_error,
        )
        return response

    if ledger["status"] in TERMINAL_STATUSES:
        response = _reject_continuation(
            paths=paths,
            ledger=ledger,
            limitation="Coding run is already terminal.",
        )
        return response

    action = str(request["action"])
    if action == "cancel":
        response = _cancel_run(paths=paths, ledger=ledger, request=request)
        return response

    response = await _approve_and_verify(paths=paths, ledger=ledger, request=request)
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
    ledger["changed_files"] = result["changed_files"]
    ledger["limitations"] = result["limitations"]
    ledger["trace_summary"] = result["trace_summary"]
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
    if result["status"] == "succeeded":
        _record_event(
            paths=paths,
            ledger=ledger,
            event_type="proposal_ready",
            summary="Patch proposal is ready for approval.",
            public_payload={"changed_files": result["changed_files"]},
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
    _record_approval(paths=paths, ledger=ledger, approval=request["approval"])
    ledger["status"] = "verifying"
    verify_request = _verify_request_from_continuation(
        ledger=ledger,
        request=request,
    )
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
    ledger["changed_files"] = _list_of_dicts(result.get("final_changed_files"))
    ledger["apply_attempts"] = _apply_attempts(result)
    ledger["execution_attempts"] = _execution_attempts(result)
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
        "changed_files": [],
        "approvals": [],
        "apply_attempts": [],
        "execution_attempts": [],
        "repair_attempts": [],
        "attempts": [],
        "blockers": [],
        "limitations": [],
        "trace_summary": [],
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
    verify_request["execution_specs"] = request["execution_specs"]
    verify_request["initial_patch_artifacts"] = ledger["patch_artifacts"]
    repository = ledger.get("repository")
    if isinstance(repository, Mapping):
        verify_request["expected_source_identity"] = _identity_from_repository(
            repository,
        )
    repair_attempt_limit = request.get("repair_attempt_limit")
    if repair_attempt_limit is not None:
        verify_request["repair_attempt_limit"] = repair_attempt_limit
    return verify_request


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
    execution_error = _execution_specs_error(request.get("execution_specs"))
    if execution_error:
        return execution_error
    return ""


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


def _blocker(message: str) -> dict[str, object]:
    blocker = {
        "code": "request_rejected",
        "message": message,
        "details": {},
    }
    return blocker
