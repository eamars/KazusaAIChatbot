"""Coding-agent background-work adapter."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.background_work.models import (
    BackgroundWorkResult,
    BackgroundWorkWorkerDecision,
)
from kazusa_ai_chatbot.coding_agent import (
    continue_coding_run,
    decide_background_coding_operation,
    get_coding_run,
    handle_background_coding_task,
    start_coding_run,
)
from kazusa_ai_chatbot.coding_agent.code_executing.models import (
    CodeExecutionSpec,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    CODING_AGENT_WORKSPACE_ROOT,
    CODING_AGENT_PM_LLM_API_KEY,
    CODING_AGENT_PM_LLM_BASE_URL,
    CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS,
    CODING_AGENT_PM_LLM_MODEL,
    CODING_AGENT_PM_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso
from kazusa_ai_chatbot.utils import parse_llm_json_output

WORKER = "coding_agent"
DESCRIPTION = (
    "Handles accepted coding tasks through the coding-agent supervisor. "
    "It may answer codebase questions or propose code artifacts, but does "
    "not apply patches, execute commands, install packages, or deliver "
    "adapter text directly."
)
CODING_AGENT_WORKER_PAYLOAD_VERSION = "coding_agent_worker_payload.v1"
CODING_RUN_REF_PREFIX = "coding_run:"
_LOCAL_PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:[\\/]|~[\\/]|/|\./|\../)[^\r\n\"'<>`]{1,500})"
)
_LOCAL_PATH_BOUNDARY_TOKENS = (
    " and ",
    " then ",
    " to ",
    " for ",
    ". ",
    ", ",
    "; ",
    "\n",
    "\r",
    ")",
)
EXECUTION_SPEC_PLANNER_PROMPT = """\
You are a coding-agent execution planner.

Convert an approval follow-up into bounded verification specs. Only these
tools are allowed:
- python_compileall with paths
- pytest with pytest_selectors

Return strict JSON:
{
  "execution_specs": [
    {
      "tool": "python_compileall | pytest",
      "paths": ["relative path"],
      "pytest_selectors": ["relative pytest selector"],
      "timeout_seconds": 20
    }
  ],
  "reason": "short reason"
}

Use python_compileall on "." when the user gave approval without a specific
test request. Do not invent shell commands, package installs, network access,
or broad command strings.
"""
_execution_spec_planner_llm = LLInterface()
_execution_spec_planner_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CODING_AGENT_PM_LLM",
    base_url=CODING_AGENT_PM_LLM_BASE_URL,
    api_key=CODING_AGENT_PM_LLM_API_KEY,
    model=CODING_AGENT_PM_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    timeout_seconds=120,
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_PM_LLM_THINKING_ENABLED,
    ),
)


async def execute(
    decision: BackgroundWorkWorkerDecision,
    *,
    max_output_chars: int = BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
) -> BackgroundWorkResult:
    """Run the public coding-agent reading interface for one background job."""

    worker_payload = _worker_payload(decision)
    if worker_payload.get("schema_version") == CODING_AGENT_WORKER_PAYLOAD_VERSION:
        result = await _execute_coding_run_payload(
            decision,
            worker_payload=worker_payload,
            max_output_chars=max_output_chars,
        )
        return result

    task_brief = _bounded_text(decision.get("task_brief"))
    if not task_brief:
        result = _failure_result(
            status="failed",
            summary="Coding agent did not receive a task brief.",
        )
        return result

    workspace_root = CODING_AGENT_WORKSPACE_ROOT.strip()
    if not workspace_root:
        result = _failure_result(
            status="failed",
            summary=(
                "Coding agent workspace is not configured for background work."
            ),
        )
        return result

    response = await handle_background_coding_task({
        "question": task_brief,
        "source_summary": _bounded_text(decision.get("source_summary")),
        "workspace_root": workspace_root,
        "max_answer_chars": max_output_chars,
        "max_artifact_chars": max_output_chars * 8,
        **_local_source_hints_from_task_brief(task_brief),
    })
    result = _map_coding_agent_response(
        response,
        max_output_chars=max_output_chars,
    )
    return result


async def _execute_coding_run_payload(
    decision: BackgroundWorkWorkerDecision,
    *,
    worker_payload: dict[str, object],
    max_output_chars: int,
) -> BackgroundWorkResult:
    """Execute one durable coding-run worker payload."""

    workspace_root = CODING_AGENT_WORKSPACE_ROOT.strip()
    if not workspace_root:
        result = _failure_result(
            status="failed",
            summary=(
                "Coding agent workspace is not configured for background work."
            ),
        )
        return result
    operation = _payload_text(worker_payload, "operation")
    if operation == "start":
        result = await _start_coding_run_from_payload(
            decision,
            worker_payload=worker_payload,
            workspace_root=workspace_root,
            max_output_chars=max_output_chars,
        )
        return result
    if operation == "status":
        result = await _get_coding_run_from_payload(
            worker_payload,
            workspace_root=workspace_root,
            max_output_chars=max_output_chars,
        )
        return result
    if operation in ("approve_and_verify", "cancel"):
        result = await _continue_coding_run_from_payload(
            decision,
            worker_payload=worker_payload,
            workspace_root=workspace_root,
            max_output_chars=max_output_chars,
        )
        return result
    result = _failure_result(
        status="rejected",
        summary="Coding run worker payload operation is unsupported.",
    )
    return result


async def _start_coding_run_from_payload(
    decision: BackgroundWorkWorkerDecision,
    *,
    worker_payload: dict[str, object],
    workspace_root: str,
    max_output_chars: int,
) -> BackgroundWorkResult:
    """Start one durable coding run from an accepted coding task."""

    task_brief = _payload_text(worker_payload, "task_brief")
    if not task_brief:
        task_brief = _bounded_text(decision.get("task_brief"))
    if not task_brief:
        result = _failure_result(
            status="failed",
            summary="Coding agent did not receive a task brief.",
        )
        return result
    route_request = {
        "question": task_brief,
        "source_summary": _bounded_text(decision.get("source_summary")),
        "workspace_root": workspace_root,
        "max_answer_chars": max_output_chars,
        "max_artifact_chars": max_output_chars * 8,
        **_local_source_hints_from_task_brief(task_brief),
    }
    operation, route_reason = await decide_background_coding_operation(
        route_request,
    )
    objective_type = _objective_type_for_operation(operation)
    if not objective_type:
        result = _failure_result(
            status="rejected",
            summary="Coding agent supervisor rejected the task as unsupported.",
        )
        result["worker_metadata"]["coding_operation"] = operation
        result["worker_metadata"]["route_reason"] = route_reason
        return result
    start_request = {
        "question": task_brief,
        "objective_type": objective_type,
        "workspace_root": workspace_root,
        "max_answer_chars": max_output_chars,
        "max_artifact_chars": max_output_chars * 8,
        **_local_source_hints_from_task_brief(task_brief),
    }
    response = await start_coding_run(start_request)
    result = _map_coding_run_response(
        response,
        coding_operation=operation,
        worker_operation="start",
        route_reason=route_reason,
        max_output_chars=max_output_chars,
    )
    return result


async def _get_coding_run_from_payload(
    worker_payload: dict[str, object],
    *,
    workspace_root: str,
    max_output_chars: int,
) -> BackgroundWorkResult:
    """Return one durable coding run's public status."""

    run_id = _run_id_from_ref(_payload_text(worker_payload, "coding_run_ref"))
    if not run_id:
        result = _failure_result(
            status="rejected",
            summary="Coding run status requires a prompt-safe coding_run_ref.",
        )
        return result
    response = await get_coding_run({
        "workspace_root": workspace_root,
        "run_id": run_id,
    })
    result = _map_coding_run_response(
        response,
        coding_operation="coding_run_status",
        worker_operation="status",
        route_reason="User asked for coding run status.",
        max_output_chars=max_output_chars,
    )
    return result


async def _continue_coding_run_from_payload(
    decision: BackgroundWorkWorkerDecision,
    *,
    worker_payload: dict[str, object],
    workspace_root: str,
    max_output_chars: int,
) -> BackgroundWorkResult:
    """Continue one durable coding run from an accepted follow-up."""

    operation = _payload_text(worker_payload, "operation")
    run_id = _run_id_from_ref(_payload_text(worker_payload, "coding_run_ref"))
    if not run_id:
        result = _failure_result(
            status="rejected",
            summary="Coding run continuation requires a prompt-safe coding_run_ref.",
        )
        return result
    request: dict[str, object] = {
        "workspace_root": workspace_root,
        "run_id": run_id,
        "action": operation,
        "reason": _bounded_text(decision.get("reason")),
    }
    if operation == "approve_and_verify":
        request["approval"] = _approval_from_payload(
            decision,
            worker_payload=worker_payload,
        )
        request["execution_specs"] = await _execution_specs_for_payload(
            worker_payload,
            max_output_chars=max_output_chars,
        )
        request["repair_attempt_limit"] = _repair_attempt_limit(worker_payload)
    response = await continue_coding_run(request)
    result = _map_coding_run_response(
        response,
        coding_operation=f"coding_run_{operation}",
        worker_operation=operation,
        route_reason=_bounded_text(decision.get("reason")),
        max_output_chars=max_output_chars,
    )
    return result


def _map_coding_run_response(
    response: Mapping[str, object],
    *,
    coding_operation: str,
    worker_operation: str,
    route_reason: str,
    max_output_chars: int,
) -> BackgroundWorkResult:
    """Map a durable coding-run response into background-work output."""

    run_status = _bounded_text(response.get("status"), limit=80)
    run_id = _bounded_text(response.get("run_id"), limit=120)
    coding_run_ref = _coding_run_ref(run_id)
    worker_status = _worker_status_for_run_status(run_status)
    answer_text = _bounded_text(
        response.get("answer_text"),
        limit=max_output_chars,
    )
    artifact_text = _coding_run_artifact_text(
        answer_text=answer_text,
        coding_run_ref=coding_run_ref,
        run_status=run_status,
        max_output_chars=max_output_chars,
    )
    failure_summary = ""
    if worker_status != "succeeded":
        failure_summary = _failure_summary(_text_list(response.get("limitations")))
        artifact_text = ""
    result: BackgroundWorkResult = {
        "status": worker_status,
        "worker": WORKER,
        "artifact_text": artifact_text,
        "failure_summary": failure_summary,
        "result_summary": _coding_run_result_summary(
            coding_run_ref=coding_run_ref,
            run_status=run_status,
            worker_operation=worker_operation,
        ),
        "worker_metadata": {
            "schema_version": "coding_agent_worker_metadata.v2",
            "coding_operation": coding_operation,
            "worker_operation": worker_operation,
            "coding_run_ref": coding_run_ref,
            "coding_run_status": run_status,
            "objective_type": _bounded_text(
                response.get("objective_type"),
                limit=80,
            ),
            "repository": _optional_mapping(response.get("repository")),
            "source_scope": _optional_mapping(response.get("source_scope")),
            "evidence_refs": _evidence_refs(response.get("evidence")),
            "patch_artifacts": _patch_artifact_summaries(
                response.get("patch_artifacts"),
            ),
            "changed_files": _file_summaries(response.get("changed_files")),
            "apply_attempts": _public_attempts(response.get("apply_attempts")),
            "execution_attempts": _public_attempts(
                response.get("execution_attempts"),
            ),
            "repair_attempts": _public_attempts(response.get("repair_attempts")),
            "blockers": _public_attempts(response.get("blockers")),
            "limitations": _text_list(response.get("limitations")),
            "trace_summary": [
                f"coding_run:{worker_operation}:{route_reason}",
                *_text_list(response.get("trace_summary")),
            ],
        },
    }
    return result


async def _execution_specs_for_payload(
    worker_payload: dict[str, object],
    *,
    max_output_chars: int,
) -> list[CodeExecutionSpec]:
    """Return trusted execution specs for an approved coding run."""

    raw_specs = worker_payload.get("execution_specs")
    specs = _valid_execution_specs(raw_specs)
    if specs:
        return specs
    execution_request = _payload_text(worker_payload, "execution_request")
    if not execution_request:
        return [_default_execution_spec()]
    planned_specs = await _plan_execution_specs(
        execution_request=execution_request,
        max_output_chars=max_output_chars,
    )
    if planned_specs:
        return planned_specs
    return [_default_execution_spec()]


async def _plan_execution_specs(
    *,
    execution_request: str,
    max_output_chars: int,
) -> list[CodeExecutionSpec]:
    """Ask the coding PM route for bounded verification specs."""

    payload = {
        "execution_request": execution_request,
        "allowed_tools": ["python_compileall", "pytest"],
        "max_output_chars": max_output_chars,
    }
    messages = [
        SystemMessage(content=EXECUTION_SPEC_PLANNER_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    response = await _execution_spec_planner_llm.ainvoke(
        messages,
        config=_execution_spec_planner_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, Mapping):
        return []
    specs = _valid_execution_specs(parsed.get("execution_specs"))
    return specs


def _valid_execution_specs(value: object) -> list[CodeExecutionSpec]:
    """Validate execution specs against the bounded tool contract."""

    if not isinstance(value, list):
        return []
    specs: list[CodeExecutionSpec] = []
    for row in value:
        if not isinstance(row, Mapping):
            continue
        tool = _bounded_text(row.get("tool"), limit=80)
        if tool == "python_compileall":
            paths = _safe_text_list(row.get("paths"), default=["."])
            spec: CodeExecutionSpec = {
                "tool": tool,
                "paths": paths,
            }
        elif tool == "pytest":
            selectors = _safe_text_list(row.get("pytest_selectors"))
            if not selectors:
                continue
            spec = {
                "tool": tool,
                "pytest_selectors": selectors,
            }
        else:
            continue
        timeout_seconds = row.get("timeout_seconds")
        if isinstance(timeout_seconds, int) and 1 <= timeout_seconds <= 600:
            spec["timeout_seconds"] = timeout_seconds
        specs.append(spec)
        if len(specs) >= 3:
            break
    return specs


def _default_execution_spec() -> CodeExecutionSpec:
    """Return the default bounded verification for approved runs."""

    return_value: CodeExecutionSpec = {
        "tool": "python_compileall",
        "paths": ["."],
        "timeout_seconds": 60,
    }
    return return_value


def _approval_from_payload(
    decision: BackgroundWorkWorkerDecision,
    *,
    worker_payload: dict[str, object],
) -> dict[str, object]:
    """Build structured approval from the accepted action payload."""

    source_scope = worker_payload.get("source_scope")
    approved_by = ""
    if isinstance(source_scope, Mapping):
        approved_by = _bounded_text(source_scope.get("source_user_id"), limit=200)
    if not approved_by:
        approved_by = "current_user"
    approved_at = _payload_text(worker_payload, "storage_timestamp_utc")
    if not approved_at:
        approved_at = storage_utc_now_iso()
    approval_reason = _bounded_text(decision.get("reason"), limit=500)
    if not approval_reason:
        approval_reason = "User approved the coding run for verification."
    approval = {
        "approved": True,
        "approved_by": approved_by,
        "approved_at": approved_at,
        "approval_reason": approval_reason,
    }
    return approval


def _repair_attempt_limit(worker_payload: Mapping[str, object]) -> int:
    """Return the bounded repair-attempt limit for approved verification."""

    value = worker_payload.get("repair_attempt_limit")
    if isinstance(value, int) and 0 <= value <= 3:
        return value
    return 1


def _objective_type_for_operation(operation: str) -> str:
    """Map coding supervisor operation to durable run objective."""

    if operation == "code_reading":
        return "read_only"
    if operation in ("code_writing", "code_modifying"):
        return "propose_patch"
    return ""


def _worker_status_for_run_status(run_status: str) -> str:
    """Map durable run status into background-worker status."""

    if run_status in ("rejected",):
        return "rejected"
    if run_status in ("failed",):
        return "failed"
    return "succeeded"


def _coding_run_artifact_text(
    *,
    answer_text: str,
    coding_run_ref: str,
    run_status: str,
    max_output_chars: int,
) -> str:
    """Build user-deliverable text for one coding run response."""

    parts: list[str] = []
    if coding_run_ref:
        parts.append(f"Coding run ref: {coding_run_ref}")
    if run_status:
        parts.append(f"Status: {run_status}")
    if answer_text:
        parts.append(answer_text)
    elif coding_run_ref:
        parts.append("Coding run state was updated.")
    return_value = "\n\n".join(parts)[:max_output_chars]
    return return_value


def _coding_run_result_summary(
    *,
    coding_run_ref: str,
    run_status: str,
    worker_operation: str,
) -> str:
    """Build compact accepted-task result summary for one coding run."""

    parts = ["coding_agent run"]
    if worker_operation:
        parts.append(worker_operation)
    if run_status:
        parts.append(run_status)
    if coding_run_ref:
        parts.append(coding_run_ref)
    return_value = "; ".join(parts)[:500]
    return return_value


def _coding_run_ref(run_id: str) -> str:
    """Return the prompt-safe coding run reference."""

    if not run_id:
        return ""
    if run_id.startswith(CODING_RUN_REF_PREFIX):
        return run_id
    return_value = f"{CODING_RUN_REF_PREFIX}{run_id}"
    return return_value


def _run_id_from_ref(coding_run_ref: str) -> str:
    """Return the durable run id from a prompt-safe coding run ref."""

    if not coding_run_ref.startswith(CODING_RUN_REF_PREFIX):
        return ""
    return_value = coding_run_ref.removeprefix(CODING_RUN_REF_PREFIX).strip()
    return return_value


def _worker_payload(
    decision: BackgroundWorkWorkerDecision,
) -> dict[str, object]:
    """Return the requested-worker payload if present."""

    worker_payload = decision.get("worker_payload")
    if not isinstance(worker_payload, Mapping):
        return_value: dict[str, object] = {}
        return return_value
    return_value = dict(worker_payload)
    return return_value


def _map_coding_agent_response(
    response: Mapping[str, object],
    *,
    max_output_chars: int,
) -> BackgroundWorkResult:
    """Map a public coding-agent response into a background-work result."""

    status = _bounded_text(response.get("status"))
    if status not in ("succeeded", "failed", "needs_user_input", "rejected"):
        status = "failed"
    answer_text = _bounded_text(
        response.get("answer_text"),
        limit=max_output_chars,
    )
    limitations = _text_list(response.get("limitations"))
    evidence_refs = _evidence_refs(response.get("evidence"))
    coding_operation = _bounded_text(
        response.get("operation"),
        limit=80,
    )
    if coding_operation not in (
        "code_reading",
        "code_writing",
        "code_modifying",
        "unsupported",
    ):
        coding_operation = "unsupported"
    failure_summary = ""
    if status != "succeeded":
        answer_text = ""
        failure_summary = _failure_summary(limitations)
    result: BackgroundWorkResult = {
        "status": status,
        "worker": WORKER,
        "artifact_text": answer_text,
        "failure_summary": failure_summary,
        "result_summary": _result_summary(
            response,
            status=status,
            evidence_count=len(evidence_refs),
        ),
        "worker_metadata": {
            "schema_version": "coding_agent_worker_metadata.v1",
            "coding_operation": coding_operation,
            "repository": _optional_mapping(response.get("repository")),
            "source_scope": _optional_mapping(response.get("source_scope")),
            "evidence_refs": evidence_refs,
            "patch_artifacts": _patch_artifact_summaries(
                response.get("patch_artifacts"),
            ),
            "created_files": _file_summaries(response.get("created_files")),
            "changed_files": _file_summaries(response.get("changed_files")),
            "validation": _optional_mapping(response.get("validation")),
            "limitations": limitations,
            "trace_summary": _text_list(response.get("trace_summary")),
        },
    }
    return result


def _failure_result(*, status: str, summary: str) -> BackgroundWorkResult:
    """Build one sanitized non-success result."""

    result: BackgroundWorkResult = {
        "status": status,
        "worker": WORKER,
        "artifact_text": "",
        "failure_summary": summary,
        "result_summary": summary,
        "worker_metadata": {
            "schema_version": "coding_agent_worker_metadata.v1",
            "coding_operation": "unsupported",
        },
    }
    return result


def _failure_summary(limitations: list[str]) -> str:
    """Choose a compact public failure summary."""

    for limitation in limitations:
        if limitation:
            return limitation
    return "Coding agent could not complete the request."


def _result_summary(
    response: Mapping[str, object],
    *,
    status: str,
    evidence_count: int,
) -> str:
    """Build a prompt-safe bounded result summary."""

    parts = [f"coding_agent {status}"]
    operation = _bounded_text(response.get("operation"), limit=80)
    if operation:
        parts.append(operation)
    repository = response.get("repository")
    if isinstance(repository, Mapping):
        owner = _bounded_text(repository.get("owner"), limit=80)
        repo = _bounded_text(repository.get("repo"), limit=80)
        if owner and repo:
            parts.append(f"{owner}/{repo}")
    parts.append(f"evidence={evidence_count}")
    created_files = _file_summaries(response.get("created_files"))
    changed_files = _file_summaries(response.get("changed_files"))
    file_count = len(created_files) + len(changed_files)
    if file_count:
        parts.append(f"files={file_count}")
    summary = "; ".join(parts)
    return summary[:500]


def _evidence_refs(value: object) -> list[dict[str, object]]:
    """Project evidence rows without raw source excerpts."""

    if not isinstance(value, list):
        return []
    refs: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, Mapping):
            continue
        path = _bounded_text(row.get("path"), limit=500)
        symbol_or_topic = _bounded_text(row.get("symbol_or_topic"), limit=200)
        reason = _bounded_text(row.get("reason"), limit=500)
        line_start = row.get("line_start")
        line_end = row.get("line_end")
        if not isinstance(line_start, int) or not isinstance(line_end, int):
            continue
        refs.append({
            "path": path,
            "line_start": line_start,
            "line_end": line_end,
            "symbol_or_topic": symbol_or_topic,
            "reason": reason,
        })
    return refs


def _optional_mapping(value: object) -> dict[str, object] | None:
    """Copy public mapping fields without private path-like keys."""

    if not isinstance(value, Mapping):
        return None
    forbidden = {
        "local_root",
        "workspace_root",
        "cache_key",
    }
    result = {
        str(key): row_value
        for key, row_value in value.items()
        if isinstance(key, str) and key not in forbidden
    }
    return result


def _patch_artifact_summaries(value: object) -> list[dict[str, object]]:
    """Project patch artifacts without raw diff text or file content."""

    if not isinstance(value, list):
        return []
    summaries: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, Mapping):
            continue
        summaries.append({
            "artifact_id": _bounded_text(row.get("artifact_id"), limit=200),
            "files": _text_list(row.get("files"), limit=12),
            "summary": _bounded_text(row.get("summary"), limit=500),
        })
        if len(summaries) >= 12:
            break
    return summaries


def _file_summaries(value: object) -> list[dict[str, object]]:
    """Project generated or changed file summaries without content."""

    if not isinstance(value, list):
        return []
    summaries: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, Mapping):
            continue
        summaries.append({
            "path": _bounded_text(row.get("path"), limit=500),
            "role": _bounded_text(row.get("role"), limit=120),
            "change_type": _bounded_text(row.get("change_type"), limit=120),
            "summary": _bounded_text(row.get("summary"), limit=500),
        })
        if len(summaries) >= 24:
            break
    return summaries


def _public_attempts(value: object) -> list[dict[str, object]]:
    """Project public run attempt rows without private local roots."""

    if not isinstance(value, list):
        return []
    attempts: list[dict[str, object]] = []
    forbidden = {
        "workspace_root",
        "local_root",
        "source_root",
        "cache_key",
        "stdout_excerpt",
        "stderr_excerpt",
    }
    for row in value:
        if not isinstance(row, Mapping):
            continue
        attempts.append({
            str(key): item
            for key, item in row.items()
            if isinstance(key, str) and key not in forbidden
        })
        if len(attempts) >= 12:
            break
    return attempts


def _text_list(value: object, *, limit: int = 12) -> list[str]:
    """Return a bounded list of text rows."""

    if not isinstance(value, list):
        return []
    rows = [
        _bounded_text(row)
        for row in value
        if isinstance(row, str) and row.strip()
    ]
    return rows[:limit]


def _safe_text_list(
    value: object,
    *,
    default: list[str] | None = None,
    limit: int = 12,
) -> list[str]:
    """Return a bounded non-empty text list or default."""

    rows = _text_list(value, limit=limit)
    if rows:
        return rows
    return_value = list(default or [])
    return return_value


def _local_source_hints_from_task_brief(task_brief: str) -> dict[str, str]:
    """Extract one visible existing local path hint from a task brief."""

    for match in _LOCAL_PATH_RE.finditer(task_brief):
        path_hint = _existing_path_prefix(match.group("path"))
        if path_hint:
            return {"local_path_hint": path_hint}
    return {}


def _existing_path_prefix(raw_text: str) -> str:
    """Return the longest existing filesystem path prefix from visible text."""

    text = raw_text.strip()
    if not text:
        return ""

    end_indexes = {len(text)}
    for token in _LOCAL_PATH_BOUNDARY_TOKENS:
        start_index = 0
        while True:
            index = text.find(token, start_index)
            if index < 0:
                break
            end_indexes.add(index)
            start_index = index + 1
    for index, character in enumerate(text):
        if character.isspace() or character in ",;.)]}":
            end_indexes.add(index)

    for end_index in sorted(end_indexes, reverse=True):
        candidate = _strip_path_suffix(text[:end_index])
        if not candidate:
            continue
        try:
            path = Path(candidate).expanduser()
        except (OSError, ValueError):
            continue
        try:
            if path.exists():
                return str(path)
        except OSError:
            continue
    return ""


def _strip_path_suffix(value: str) -> str:
    """Strip punctuation commonly attached to prose around a path."""

    return value.strip(" \t\r\n\"'`.,;:)]}")


def _payload_text(payload: Mapping[str, object], field_name: str) -> str:
    """Return one stripped worker-payload text field."""

    value = payload.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _bounded_text(value: object, *, limit: int = 4000) -> str:
    """Return stripped text capped to a local bound."""

    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()[:limit]
    return return_value
