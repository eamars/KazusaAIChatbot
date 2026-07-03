"""Dedicated new-artifact patch-building boundary for code writing."""

from __future__ import annotations

import json
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.context_budget import (
    PATCHER_TARGET_INPUT_TOKEN_CAP,
    collect_selected_evidence_refs,
    prompt_budget_metadata,
)
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    ChangedFileSummary,
    CreatedFileSummary,
    GeneratedArtifact,
    PatchOperation,
    WritingPatcherInput,
    WritingPatcherReport,
)
from kazusa_ai_chatbot.coding_agent.code_writing.patch_operations import (
    compile_patch_operations,
)
from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
    _safe_repo_relative_path,
)
from kazusa_ai_chatbot.config import (
    CODING_AGENT_PROGRAMMER_LLM_MODEL,
    CODING_AGENT_PROGRAMMER_LLM_THINKING_ENABLED,
)

PATCHER_MATERIALIZATION_PROMPT = '''\
You are the patch builder for a code-writing agent.
You receive selected generated artifacts and reserved paths. Materialize those
artifacts into new-file patch output only. Do not reinterpret feature behavior,
add unassigned files, edit existing files, run commands, or apply patches.
'''


def materialize_patch_artifacts(
    *,
    repo_root: Path | None,
    patcher_input: WritingPatcherInput,
    max_files: int,
    max_diff_chars: int,
    trace: dict[str, object] | None = None,
) -> WritingPatcherReport:
    """Build selected generated artifacts into new-file patch output."""

    payload = _patcher_payload(patcher_input)
    payload_text = json.dumps(payload, ensure_ascii=False)
    context_budget = prompt_budget_metadata(
        system_prompt=PATCHER_MATERIALIZATION_PROMPT,
        payload_text=payload_text,
        target_input_tokens=PATCHER_TARGET_INPUT_TOKEN_CAP,
        selected_evidence_refs=collect_selected_evidence_refs(payload),
    )
    if context_budget["over_hard_cap"]:
        report = _report(
            status="failed",
            artifact_package={},
            diagnostics=["Patcher input exceeded the context budget."],
        )
        _fill_trace(
            trace,
            patcher_input=payload,
            report=report,
            context_budget=context_budget,
        )
        return report

    operations, diagnostics = _create_file_operations(patcher_input["artifacts"])
    if not operations:
        report = _report(
            status="failed",
            artifact_package={},
            diagnostics=diagnostics or ["No generated artifacts were provided."],
        )
        _fill_trace(
            trace,
            patcher_input=payload,
            report=report,
            context_budget=context_budget,
        )
        return report

    artifacts, created_files, changed_files, operation_errors = (
        compile_patch_operations(
            repo_root=repo_root,
            patch_operations=operations,
            max_files=max_files,
            max_diff_chars=max_diff_chars,
        )
    )
    diagnostics.extend(operation_errors)
    diagnostics.extend(_created_file_errors(created_files, changed_files))
    diagnostics.extend(_artifact_scope_errors(patcher_input["artifacts"]))
    status = "succeeded"
    if diagnostics or not artifacts:
        status = "failed"

    report = _report(
        status=status,
        artifact_package={
            "artifact_package_id": patcher_input["artifact_package_id"],
            "artifact_count": len(patcher_input["artifacts"]),
        },
        patch_artifacts=artifacts,
        created_files=_dedupe_created_files(created_files),
        changed_files=_dedupe_changed_files(changed_files),
        diagnostics=_dedupe_strings(diagnostics),
    )
    _fill_trace(
        trace,
        patcher_input=payload,
        report=report,
        context_budget=context_budget,
    )
    return report


def _patcher_payload(
    patcher_input: WritingPatcherInput,
) -> dict[str, object]:
    payload = {
        "artifact_package_id": patcher_input["artifact_package_id"],
        "reserved_paths": patcher_input["reserved_paths"],
        "artifacts": [
            {
                "artifact_id": artifact["artifact_id"],
                "file_label": artifact["file_label"],
                "file_kind": artifact["file_kind"],
                "content_format": artifact["content_format"],
                "path": artifact["path"],
                "content_chars": len(artifact["content"]),
                "purpose": artifact["purpose"],
            }
            for artifact in patcher_input["artifacts"]
        ],
        "max_artifact_chars": patcher_input["max_artifact_chars"],
    }
    return payload


def _create_file_operations(
    artifacts: list[GeneratedArtifact],
) -> tuple[list[PatchOperation], list[str]]:
    operations: list[PatchOperation] = []
    diagnostics: list[str] = []
    for artifact in artifacts:
        safe_path = _safe_repo_relative_path(artifact["path"])
        if safe_path is None:
            diagnostics.append(
                f"Generated artifact {artifact['artifact_id']!r} has unsafe path.",
            )
            continue
        content = artifact["content"]
        if not content:
            diagnostics.append(
                f"Generated artifact {artifact['artifact_id']!r} is empty.",
            )
            continue
        operation: PatchOperation = {
            "operation_id": f"{artifact['artifact_id']}-create",
            "kind": "create_file",
            "path": safe_path,
            "content": content,
            "summary": artifact["purpose"] or artifact["file_label"],
        }
        operations.append(operation)
    return operations, diagnostics


def _created_file_errors(
    created_files: list[CreatedFileSummary],
    changed_files: list[ChangedFileSummary],
) -> list[str]:
    errors: list[str] = []
    created_paths = {item["path"] for item in created_files}
    for changed_file in changed_files:
        if changed_file["path"] not in created_paths:
            errors.append(
                "Patcher produced a changed-file summary for a non-created file.",
            )
    return errors


def _artifact_scope_errors(artifacts: list[GeneratedArtifact]) -> list[str]:
    errors: list[str] = []
    seen_paths: set[str] = set()
    for artifact in artifacts:
        safe_path = _safe_repo_relative_path(artifact["path"])
        if safe_path is None:
            continue
        if safe_path in seen_paths:
            errors.append(f"Generated artifact path {safe_path!r} is duplicated.")
            continue
        seen_paths.add(safe_path)
    return errors


def _report(
    *,
    status: str,
    artifact_package: dict[str, object],
    diagnostics: list[str],
    patch_artifacts: list[dict[str, object]] | None = None,
    created_files: list[CreatedFileSummary] | None = None,
    changed_files: list[ChangedFileSummary] | None = None,
) -> WritingPatcherReport:
    report: WritingPatcherReport = {
        "status": status,
        "artifact_package": artifact_package,
        "patch_artifacts": patch_artifacts or [],
        "created_files": created_files or [],
        "changed_files": changed_files or [],
        "diagnostics": diagnostics,
    }
    return report


def _fill_trace(
    trace: dict[str, object] | None,
    *,
    patcher_input: dict[str, object],
    report: WritingPatcherReport,
    context_budget: dict[str, object],
) -> None:
    if trace is None:
        return

    trace.update({
        "route_name": "CODING_AGENT_PROGRAMMER_LLM",
        "model": CODING_AGENT_PROGRAMMER_LLM_MODEL,
        "thinking_enabled": CODING_AGENT_PROGRAMMER_LLM_THINKING_ENABLED,
        "materialization": "deterministic_new_file_patch_operations",
        "input": patcher_input,
        "context_budget": context_budget,
        "report": report,
    })


def _dedupe_created_files(
    created_files: list[CreatedFileSummary],
) -> list[CreatedFileSummary]:
    deduped: list[CreatedFileSummary] = []
    seen: set[str] = set()
    for created_file in created_files:
        path = created_file["path"]
        if path in seen:
            continue
        seen.add(path)
        deduped.append(created_file)
    return deduped


def _dedupe_changed_files(
    changed_files: list[ChangedFileSummary],
) -> list[ChangedFileSummary]:
    deduped: list[ChangedFileSummary] = []
    seen: set[str] = set()
    for changed_file in changed_files:
        path = changed_file["path"]
        if path in seen:
            continue
        seen.add(path)
        deduped.append(changed_file)
    return deduped


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value in deduped:
            continue
        deduped.append(value)
    return deduped
