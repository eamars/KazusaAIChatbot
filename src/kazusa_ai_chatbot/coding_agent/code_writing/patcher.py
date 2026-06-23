"""Dedicated patch-building boundary for code writing."""

from __future__ import annotations

import ast
import json
from pathlib import Path, PurePosixPath

from kazusa_ai_chatbot.coding_agent.context_budget import (
    PATCHER_TARGET_INPUT_TOKEN_CAP,
    collect_selected_evidence_refs,
    prompt_budget_metadata,
)
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    ChangedFileSummary,
    CreatedFileSummary,
    PatchArtifact,
    PatchOperation,
    WritingPatcherInput,
    WritingPatcherReport,
    WritingProgrammerReport,
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

PATCHER_MATERIALIZATION_PROMPT = """\
You are the patch builder for a code-writing agent.
You receive PM-selected programmer reports, an owned path map, base summaries,
and patch size limits. Convert selected implementation content into patches
only. Do not reinterpret the user request, invent feature behavior,
add unassigned files, run commands, apply patches, or turn unresolved
programmer gaps into successful patch content.
"""


def materialize_patch_artifacts(
    *,
    repo_root: Path | None,
    patcher_input: WritingPatcherInput,
    max_files: int,
    max_diff_chars: int,
    trace: dict[str, object] | None = None,
) -> WritingPatcherReport:
    """Build PM-selected programmer content into patch output.

    This role owns edit mechanics. It consumes programmer code artifacts and
    produces unified diffs through the existing deterministic compiler, while
    preserving the patcher trace boundary.
    """

    payload = _patcher_payload(patcher_input)
    payload_text = json.dumps(payload, ensure_ascii=False)
    context_budget = prompt_budget_metadata(
        system_prompt=PATCHER_MATERIALIZATION_PROMPT,
        payload_text=payload_text,
        target_input_tokens=PATCHER_TARGET_INPUT_TOKEN_CAP,
        selected_evidence_refs=collect_selected_evidence_refs(payload),
    )
    if context_budget["over_hard_cap"]:
        report = _blocked_report(
            diagnostics=["Patcher input exceeded the context budget."],
            unmaterialized_reports=_report_ids(
                patcher_input["selected_programmer_reports"],
            ),
        )
        _fill_trace(
            trace,
            patcher_input=payload,
            report=report,
            context_budget=context_budget,
        )
        return report

    owned_path_map = _safe_owned_path_map(patcher_input["owned_path_map"])
    operations, diagnostics, unmaterialized_reports = _owned_patch_operations(
        repo_root=repo_root,
        reports=patcher_input["selected_programmer_reports"],
        owned_path_map=owned_path_map,
    )
    if not operations:
        diagnostics.append(
            "Patcher received no PM-selected programmer code artifacts.",
        )
        report = _blocked_report(
            diagnostics=diagnostics,
            unmaterialized_reports=unmaterialized_reports
            or _report_ids(patcher_input["selected_programmer_reports"]),
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
    diagnostics.extend(_artifact_scope_errors(artifacts, owned_path_map))
    if not artifacts:
        diagnostics.append("Patcher could not build patch output.")

    status = "succeeded"
    if diagnostics or not artifacts:
        status = "blocked"
    report: WritingPatcherReport = {
        "status": status,
        "patch_artifacts": artifacts,
        "created_files": _dedupe_created_files(created_files),
        "changed_files": _dedupe_changed_files(changed_files),
        "edit_diagnostics": _dedupe_strings(diagnostics),
        "unmaterialized_reports": _dedupe_strings(unmaterialized_reports),
    }
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
    reports = [
        {
            "assignment_id": report["assignment_id"],
            "file_contract_id": report.get("file_contract_id", ""),
            "file_label": report["file_label"],
            "status": report["status"],
            "files_considered": report["files_considered"],
            "facts": report["facts"],
            "edit_mode": report["edit_mode"],
            "code_artifact_chars": len(report.get("code_artifact", "")),
            "created_files": report["created_files"],
            "changed_files": report["changed_files"],
            "open_questions": report["open_questions"],
        }
        for report in patcher_input["selected_programmer_reports"]
    ]
    payload = {
        "question": patcher_input["question"],
        "mode": patcher_input["mode"],
        "base_identity": patcher_input["base_identity"],
        "owned_path_map": patcher_input["owned_path_map"],
        "base_file_summaries": patcher_input["base_file_summaries"],
        "selected_programmer_reports": reports,
        "pm_integration_notes": patcher_input["pm_integration_notes"],
        "artifact_limits": patcher_input["artifact_limits"],
    }
    return payload


def _safe_owned_path_map(
    owned_path_map: dict[str, str],
) -> dict[str, str]:
    safe_map: dict[str, str] = {}
    for raw_path, owner in owned_path_map.items():
        safe_path = _safe_repo_relative_path(raw_path)
        if safe_path is None:
            continue
        safe_map[safe_path] = owner
    return safe_map


def _owned_patch_operations(
    *,
    repo_root: Path | None,
    reports: list[WritingProgrammerReport],
    owned_path_map: dict[str, str],
) -> tuple[list[PatchOperation], list[str], list[str]]:
    operations: list[PatchOperation] = []
    diagnostics: list[str] = []
    unmaterialized_reports: list[str] = []
    for report in reports:
        if report["status"] != "succeeded":
            continue

        code_artifact = report.get("code_artifact", "")
        if not code_artifact:
            unmaterialized_reports.append(report["assignment_id"])
            continue

        safe_path = _owned_path_for_report(
            report=report,
            owned_path_map=owned_path_map,
        )
        if safe_path is None:
            diagnostics.append(
                "Patcher could not resolve an owned path for programmer "
                f"report {report['assignment_id']}.",
            )
            unmaterialized_reports.append(report["assignment_id"])
            continue

        report_operations = _operations_from_code_artifact(
            repo_root=repo_root,
            safe_path=safe_path,
            report=report,
            code_artifact=code_artifact,
        )
        if not report_operations:
            diagnostics.append(
                f"Patcher could not anchor code artifact for {safe_path}.",
            )
            unmaterialized_reports.append(report["assignment_id"])
            continue
        operations.extend(report_operations)
    return operations, diagnostics, unmaterialized_reports


def _owned_path_for_report(
    *,
    report: WritingProgrammerReport,
    owned_path_map: dict[str, str],
) -> str | None:
    preferred_owner_ids = {
        report["assignment_id"],
        report.get("file_contract_id", ""),
        report.get("file_label", ""),
    }
    for safe_path, owner_id in owned_path_map.items():
        if owner_id in preferred_owner_ids:
            return safe_path

    considered_paths = [
        _safe_repo_relative_path(path)
        for path in report.get("files_considered", [])
    ]
    for safe_path in considered_paths:
        if safe_path is None:
            continue
        if _path_allowed(safe_path, owned_path_map):
            return safe_path
    return None


def _operations_from_code_artifact(
    *,
    repo_root: Path | None,
    safe_path: str,
    report: WritingProgrammerReport,
    code_artifact: str,
) -> list[PatchOperation]:
    current_text = _read_current_text(repo_root=repo_root, safe_path=safe_path)
    operation_id = f"{report['assignment_id']}-code"
    summary = f"Materialize programmer output for {report['file_label']}."
    if not current_text:
        operation: PatchOperation = {
            "operation_id": operation_id,
            "kind": "create_file",
            "path": safe_path,
            "content": code_artifact,
            "summary": summary,
        }
        return [operation]

    if report.get("edit_mode") == "complete_file":
        return [{
            "operation_id": operation_id,
            "kind": "replace",
            "path": safe_path,
            "anchor": current_text,
            "content": code_artifact,
            "summary": summary,
        }]

    if _is_python_path(safe_path):
        operations = _python_symbol_operations(
            current_text=current_text,
            code_artifact=code_artifact,
            safe_path=safe_path,
            operation_id=operation_id,
            summary=summary,
        )
        if operations:
            return operations

    anchor = _append_anchor(current_text)
    if not anchor:
        return []
    return [{
        "operation_id": operation_id,
        "kind": "insert_after",
        "path": safe_path,
        "anchor": anchor,
        "content": "\n" + code_artifact.strip() + "\n",
        "summary": summary,
    }]


def _python_symbol_operations(
    *,
    current_text: str,
    code_artifact: str,
    safe_path: str,
    operation_id: str,
    summary: str,
) -> list[PatchOperation]:
    try:
        current_tree = ast.parse(current_text)
        artifact_tree = ast.parse(code_artifact)
    except SyntaxError:
        return []

    current_lines = current_text.splitlines(keepends=True)
    artifact_lines = code_artifact.splitlines(keepends=True)
    operations: list[PatchOperation] = []
    unmatched_blocks: list[str] = []
    for node in artifact_tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            source = _node_source(artifact_lines, node)
            target = _matching_python_node(current_tree, node.name)
            if target is not None:
                anchor = _node_source(current_lines, target)
                content = _source_at_indent(source, target.col_offset)
                operations.append({
                    "operation_id": f"{operation_id}-{node.name}",
                    "kind": "replace",
                    "path": safe_path,
                    "anchor": anchor,
                    "content": content,
                    "summary": summary,
                })
                continue
            if _looks_like_method(node):
                method_operation = _insert_method_operation(
                    current_tree=current_tree,
                    current_lines=current_lines,
                    source=source,
                    safe_path=safe_path,
                    operation_id=f"{operation_id}-{node.name}",
                    summary=summary,
                )
                if method_operation is not None:
                    operations.insert(0, method_operation)
                    continue
        block = _node_source(artifact_lines, node).strip()
        if block and not block.startswith("self."):
            unmatched_blocks.append(block)

    if unmatched_blocks:
        anchor = _append_anchor(current_text)
        if anchor:
            operations.append({
                "operation_id": f"{operation_id}-append",
                "kind": "insert_after",
                "path": safe_path,
                "anchor": anchor,
                "content": "\n" + "\n\n".join(unmatched_blocks).strip() + "\n",
                "summary": summary,
            })
    return operations


def _is_python_path(safe_path: str) -> bool:
    suffix = PurePosixPath(safe_path).suffix.casefold()
    return suffix in {".py", ".pyi"}


def _node_source(lines: list[str], node: ast.AST) -> str:
    line_start = getattr(node, "lineno", 1)
    line_end = getattr(node, "end_lineno", line_start)
    return "".join(lines[line_start - 1:line_end])


def _matching_python_node(
    tree: ast.AST,
    name: str,
) -> ast.AST | None:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == name
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _source_at_indent(source: str, target_indent: int) -> str:
    source_lines = source.rstrip().splitlines()
    if not source_lines:
        return ""
    current_indent = len(source_lines[0]) - len(source_lines[0].lstrip(" "))
    indent_delta = target_indent - current_indent
    if indent_delta <= 0:
        return "\n".join(
            line[-indent_delta:] if line.startswith(" " * -indent_delta) else line
            for line in source_lines
        )
    prefix = " " * indent_delta
    return "\n".join(prefix + line if line.strip() else line for line in source_lines)


def _looks_like_method(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    if not node.args.args:
        return False
    return node.args.args[0].arg == "self"


def _insert_method_operation(
    *,
    current_tree: ast.AST,
    current_lines: list[str],
    source: str,
    safe_path: str,
    operation_id: str,
    summary: str,
) -> PatchOperation | None:
    classes = [node for node in ast.walk(current_tree) if isinstance(node, ast.ClassDef)]
    if len(classes) != 1:
        return None
    target_class = classes[0]
    anchor = _node_source(current_lines, target_class)
    class_source = anchor.rstrip()
    method_source = _source_at_indent(source, target_class.col_offset + 4)
    content = class_source + "\n\n" + method_source.rstrip() + "\n"
    return {
        "operation_id": operation_id,
        "kind": "replace",
        "path": safe_path,
        "anchor": anchor,
        "content": content,
        "summary": summary,
    }


def _read_current_text(*, repo_root: Path | None, safe_path: str) -> str:
    if repo_root is None:
        return ""
    file_path = repo_root / safe_path
    try:
        resolved_root = repo_root.expanduser().resolve(strict=True)
        resolved_file = file_path.expanduser().resolve(strict=True)
    except OSError:
        return ""
    if resolved_file == resolved_root or resolved_root not in resolved_file.parents:
        return ""
    if not resolved_file.is_file():
        return ""
    try:
        return resolved_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _append_anchor(text: str) -> str:
    lines = text.splitlines(keepends=True)
    for line in reversed(lines):
        if line.strip():
            return line
    return lines[-1] if lines else ""


def _artifact_scope_errors(
    artifacts: list[PatchArtifact],
    owned_path_map: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    for artifact in artifacts:
        for raw_path in artifact["files"]:
            safe_path = _safe_repo_relative_path(raw_path)
            if safe_path is None:
                errors.append("Patcher artifact contains an unsafe file path.")
                continue
            if not _path_allowed(safe_path, owned_path_map):
                errors.append(
                    "Patcher artifact contains a path outside the PM-owned "
                    f"path map: {safe_path}.",
                )
    return errors


def _path_allowed(
    safe_path: str,
    owned_path_map: dict[str, str],
) -> bool:
    if not owned_path_map:
        return False

    candidate = PurePosixPath(safe_path)
    for owned_path in owned_path_map:
        owner = PurePosixPath(owned_path)
        if candidate == owner or owner in candidate.parents:
            return True
    return False


def _blocked_report(
    *,
    diagnostics: list[str],
    unmaterialized_reports: list[str],
) -> WritingPatcherReport:
    report: WritingPatcherReport = {
        "status": "blocked",
        "patch_artifacts": [],
        "created_files": [],
        "changed_files": [],
        "edit_diagnostics": _dedupe_strings(diagnostics),
        "unmaterialized_reports": _dedupe_strings(unmaterialized_reports),
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
        "materialization": "deterministic_patch_operations",
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


def _report_ids(
    reports: list[WritingProgrammerReport],
) -> list[str]:
    return [report["assignment_id"] for report in reports]


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value in deduped:
            continue
        deduped.append(value)
    return deduped
