"""Coding-agent background-work adapter."""

from __future__ import annotations

from collections.abc import Mapping

from kazusa_ai_chatbot.background_work.models import (
    BackgroundWorkResult,
    BackgroundWorkWorkerDecision,
)
from kazusa_ai_chatbot.coding_agent import handle_background_coding_task
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    CODING_AGENT_WORKSPACE_ROOT,
)

WORKER = "coding_agent"
DESCRIPTION = (
    "Handles accepted coding tasks through the coding-agent supervisor. "
    "It may answer codebase questions or propose code artifacts, but does "
    "not apply patches, execute commands, install packages, or deliver "
    "adapter text directly."
)


async def execute(
    decision: BackgroundWorkWorkerDecision,
    *,
    max_output_chars: int = BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
) -> BackgroundWorkResult:
    """Run the public coding-agent reading interface for one background job."""

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
    })
    result = _map_coding_agent_response(
        response,
        max_output_chars=max_output_chars,
    )
    return result


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
    if coding_operation not in ("code_reading", "code_writing", "unsupported"):
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


def _bounded_text(value: object, *, limit: int = 4000) -> str:
    """Return stripped text capped to a local bound."""

    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()[:limit]
    return return_value
