"""Deterministic executors for validated coding-loop actions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_action_loop.state import CandidateState
from kazusa_ai_chatbot.coding_agent.code_executing import execute_code_check
from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
    materialize_text_operation,
)
from kazusa_ai_chatbot.coding_agent.repository_index.overlay import CandidateOverlay
from kazusa_ai_chatbot.coding_agent.repository_index.search import (
    read_snapshot,
    search_snapshot,
)
from kazusa_ai_chatbot.coding_agent.safety import (
    normalize_safe_repo_relative_path,
)


def execute_action(
    *,
    action: dict[str, object],
    workspace_root: Path,
    snapshot_id: str,
    run_root: Path | None = None,
    objective_type: str = "read_only",
    run_context: dict[str, object] | None = None,
    operation_id: str | None = None,
) -> dict[str, object]:
    """Execute one validated read/search/note/finish/block request.

    Args:
        action: Validated closed action payload from the controller parser.
        workspace_root: Coding workspace containing the pinned repository index.
        snapshot_id: Immutable index snapshot selected for this run.

    Returns:
        Prompt-safe observation describing the deterministic action outcome.
    """

    action_name = action["action"]
    args = action["args"]
    if not isinstance(action_name, str) or not isinstance(args, dict):
        return {"outcome": "rejected", "kind": "invalid_action"}
    if action_name == "read":
        repo_path = args.get("repo_path")
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        symbol = args.get("symbol")
        if (
            isinstance(repo_path, str)
            and isinstance(symbol, str)
            and start_line is None
        ):
            symbol_span = _resolve_symbol_span(
                workspace_root=workspace_root,
                snapshot_id=snapshot_id,
                run_root=run_root,
                objective_type=objective_type,
                repo_path=repo_path,
                symbol=symbol,
            )
            if symbol_span is None:
                return {
                    "outcome": "stale",
                    "kind": "read_result",
                    "evidence": [],
                }
            start_line, end_line = symbol_span
        if (
            not isinstance(repo_path, str)
            or not isinstance(start_line, int)
            or (end_line is not None and not isinstance(end_line, int))
        ):
            return {"outcome": "rejected", "kind": "invalid_read"}
        if objective_type != "read_only" and run_root is not None:
            result = _read_candidate(
                run_root=run_root,
                repo_path=repo_path,
                start_line=start_line,
                end_line=end_line,
            )
        else:
            result = read_snapshot(
                workspace_root=workspace_root,
                snapshot_id=snapshot_id,
                repo_path=repo_path,
                start_line=start_line,
                end_line=end_line,
            )
        observation = {
            "outcome": result["status"],
            "kind": "read_result",
            "evidence": result["rows"],
        }
        return observation
    if action_name == "search":
        query = args.get("query")
        mode = args.get("mode")
        if not isinstance(query, str) or not isinstance(mode, str):
            return {"outcome": "rejected", "kind": "invalid_search"}
        overlay_revision = 0
        if objective_type != "read_only" and run_root is not None:
            candidate = CandidateState.load(run_root / "candidate")
            overlay_revision = candidate.revision
        result = search_snapshot(
            workspace_root=workspace_root,
            snapshot_id=snapshot_id,
            mode=mode,
            query=query,
            cursor=args.get("cursor") if isinstance(args.get("cursor"), str) else None,
            path_glob=(
                args.get("path_glob")
                if isinstance(args.get("path_glob"), str)
                else None
            ),
            overlay_revision=overlay_revision,
            overlay_database_path=(
                run_root / "candidate" / "overlay.sqlite"
                if objective_type != "read_only" and run_root is not None
                else None
            ),
        )
        observation = {
            "outcome": result["status"],
            "kind": "search_result",
            "evidence": result["rows"],
            "cursor": result["cursor"],
        }
        return observation
    if action_name == "edit" and objective_type != "read_only" and run_root is not None:
        edit_observation = _execute_edit(
            args=args,
            run_root=run_root,
            operation_id=operation_id,
        )
        return edit_observation
    if action_name == "run":
        run_observation = _execute_semantic_run(
            args=args,
            run_context=run_context,
        )
        return run_observation
    if action_name in {"note", "finish", "block"}:
        observation = {"outcome": "ok", "kind": f"{action_name}_result"}
        return observation
    observation = {"outcome": "unavailable", "kind": "action_unavailable"}
    return observation


def _resolve_symbol_span(
    *,
    workspace_root: Path,
    snapshot_id: str,
    run_root: Path | None,
    objective_type: str,
    repo_path: str,
    symbol: str,
) -> tuple[int, int] | None:
    """Resolve one exact symbol against the current candidate view."""

    if objective_type != "read_only" and run_root is not None:
        overlay = CandidateOverlay(run_root / "candidate" / "overlay.sqlite")
        try:
            overlay_state = overlay.describe_paths([repo_path])[0]["state"]
        finally:
            overlay.close()
        if overlay_state != "absent":
            result = search_snapshot(
                workspace_root=workspace_root,
                snapshot_id=snapshot_id,
                mode="symbol",
                query=symbol,
                path_glob=repo_path,
                overlay_revision=CandidateState.load(
                    run_root / "candidate"
                ).revision,
                overlay_database_path=run_root / "candidate" / "overlay.sqlite",
            )
            for row in result["rows"]:
                if row["repo_path"] == repo_path and row["symbol"] == symbol:
                    symbol_span = (
                        int(row["start_line"]),
                        int(row["end_line"]),
                    )
                    return symbol_span
            return None
    result = search_snapshot(
        workspace_root=workspace_root,
        snapshot_id=snapshot_id,
        mode="symbol",
        query=symbol,
        path_glob=repo_path,
    )
    for row in result["rows"]:
        if row["repo_path"] == repo_path and row["symbol"] == symbol:
            symbol_span = (int(row["start_line"]), int(row["end_line"]))
            return symbol_span
    return None


def _read_candidate(
    *,
    run_root: Path,
    repo_path: str,
    start_line: int,
    end_line: int | None,
) -> dict[str, object]:
    """Read one bounded span from the current managed candidate view."""

    safe_path = normalize_safe_repo_relative_path(repo_path)
    if safe_path is None or start_line < 1:
        return {"status": "rejected", "rows": []}
    requested_end = end_line if end_line is not None else start_line + 199
    if requested_end < start_line or requested_end - start_line >= 500:
        return {"status": "rejected", "rows": []}
    candidate = CandidateState.load(run_root / "candidate")
    try:
        content = candidate.read_safe_text(safe_path)
    except (OSError, ValueError):
        return {"status": "rejected", "rows": []}
    if content is None:
        return {"status": "stale", "rows": []}
    lines = content.splitlines(keepends=True)
    actual_end = min(requested_end, len(lines))
    if start_line > actual_end:
        return {"status": "stale", "rows": []}
    excerpt = "".join(lines[start_line - 1:actual_end])
    row = {
        "repo_path": safe_path,
        "start_line": start_line,
        "end_line": actual_end,
        "content": excerpt,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    return {"status": "ok", "rows": [row]}


def _execute_semantic_run(
    *,
    args: dict[str, object],
    run_context: dict[str, object] | None,
) -> dict[str, object]:
    """Execute one approved semantic verification request in its managed copy."""

    if run_context is None:
        return {"outcome": "unavailable", "kind": "action_unavailable"}
    profile = args.get("profile")
    targets = args.get("targets", [])
    if profile not in {"derived_base", "focused"}:
        return {"outcome": "rejected", "kind": "invalid_run", "evidence": []}
    if not isinstance(targets, list) or not all(
        isinstance(target, str) for target in targets
    ):
        return {"outcome": "rejected", "kind": "invalid_run", "evidence": []}
    execution_specs = run_context.get("execution_specs")
    if not isinstance(execution_specs, list):
        return {"outcome": "unavailable", "kind": "action_unavailable"}
    selected_specs = _selected_execution_specs(
        execution_specs=execution_specs,
        profile=profile,
        targets=targets,
    )
    if not selected_specs:
        return {"outcome": "unavailable", "kind": "action_unavailable"}
    workspace_root = run_context.get("workspace_root")
    candidate_execution_base = run_context.get("candidate_execution_base")
    if (
        not isinstance(workspace_root, str)
        or not isinstance(candidate_execution_base, dict)
    ):
        raise ValueError("trusted execution context is invalid")
    results: list[dict[str, object]] = []
    for execution_spec in selected_specs:
        execution_spec_digest = _canonical_json_digest(execution_spec)
        candidate_execution_identity = {
            **candidate_execution_base,
            "execution_spec_digest": execution_spec_digest,
        }
        execution_result = dict(execute_code_check({
            "workspace_root": workspace_root,
            "candidate_execution_identity": candidate_execution_identity,
            "execution": execution_spec,
        }))
        results.append(_project_execution_result(execution_result))
    outcome = (
        "ok"
        if all(row["status"] == "succeeded" for row in results)
        else "failed"
    )
    return {"outcome": outcome, "kind": "run_result", "evidence": results}


def _canonical_json_digest(value: object) -> str:
    """Return the stable identity of one structured execution specification."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest


def _selected_execution_specs(
    *,
    execution_specs: list[object],
    profile: object,
    targets: list[object],
) -> list[dict[str, object]]:
    """Select approved checks without synthesizing commands or selectors."""

    structured_specs = [
        dict(specification)
        for specification in execution_specs
        if isinstance(specification, dict)
    ]
    if profile == "derived_base":
        return structured_specs
    if not targets:
        return []
    target_set = set(targets)
    selected_specs: list[dict[str, object]] = []
    for specification in structured_specs:
        selectors = specification.get(
            "pytest_selectors",
            specification.get("paths", []),
        )
        if not isinstance(selectors, list):
            continue
        if target_set.issubset(set(selectors)):
            selected_specs.append(specification)
    return selected_specs


def _project_execution_result(
    execution_result: dict[str, object],
) -> dict[str, object]:
    """Keep semantic run evidence and exclude trusted operational identity."""

    projected: dict[str, object] = {}
    for field_name in (
        "status",
        "tool",
        "exit_code",
        "stdout_excerpt",
        "stderr_excerpt",
        "limitations",
        "trace_summary",
    ):
        value = execution_result.get(field_name)
        if isinstance(value, (str, int, list)) and not isinstance(value, bool):
            projected[field_name] = value
    return projected


def _execute_edit(
    *,
    args: dict[str, object],
    run_root: Path,
    operation_id: str | None = None,
) -> dict[str, object]:
    """Apply one revision-bound text mutation to the managed candidate."""

    operation_kind = args.get("operation")
    repo_path_value = args.get("repo_path")
    expected_revision = args.get("expected_candidate_revision")
    if (
        not isinstance(operation_kind, str)
        or not isinstance(repo_path_value, str)
        or not isinstance(expected_revision, int)
    ):
        return {"outcome": "rejected", "kind": "invalid_edit", "evidence": []}
    if not isinstance(operation_id, str) or not operation_id:
        return {
            "outcome": "rejected",
            "kind": "missing_operation_identity",
            "evidence": [],
        }
    repo_path = normalize_safe_repo_relative_path(repo_path_value)
    if repo_path is None:
        return {"outcome": "rejected", "kind": "unsafe_path", "evidence": []}
    candidate = CandidateState.load(run_root / "candidate")
    if expected_revision != candidate.revision:
        return {"outcome": "stale", "kind": "candidate_revision", "evidence": []}
    expected_hash = args.get("expected_sha256")
    try:
        current_content = candidate.read_safe_text(repo_path)
    except (OSError, ValueError):
        return {
            "outcome": "rejected",
            "kind": "unsafe_file",
            "evidence": [],
        }
    if operation_kind != "create_file":
        if current_content is None or not isinstance(expected_hash, str):
            return {"outcome": "stale", "kind": "content_precondition", "evidence": []}
        current_hash = hashlib.sha256(current_content.encode("utf-8")).hexdigest()
        if current_hash != expected_hash:
            return {"outcome": "stale", "kind": "content_precondition", "evidence": []}
    try:
        patch_operation = _apply_edit_operation(
            candidate=candidate,
            operation_kind=operation_kind,
            repo_path=repo_path,
            current_content=current_content,
            expected_revision=expected_revision,
            args=args,
            operation_id=operation_id,
        )
    except ValueError as exc:
        return {
            "outcome": "rejected",
            "kind": "edit_rejected",
            "summary": str(exc),
            "evidence": [],
        }
    journal_operation = patch_operation.pop("_journal_operation")
    if not isinstance(journal_operation, dict):
        raise ValueError("candidate mutation journal evidence is invalid")
    patch_operation["operation_id"] = journal_operation["operation_id"]
    patch_operation["expected_candidate_revision"] = expected_revision
    if isinstance(expected_hash, str):
        patch_operation["expected_source_sha256"] = expected_hash
    evidence = [{
        "operation_id": journal_operation["operation_id"],
        "patch_operation": patch_operation,
        "candidate_revision": candidate.revision,
    }]
    return {"outcome": "ok", "kind": "edit_result", "evidence": evidence}


def _apply_edit_operation(
    *,
    candidate: CandidateState,
    operation_kind: str,
    repo_path: str,
    current_content: str | None,
    expected_revision: int,
    args: dict[str, object],
    operation_id: str,
) -> dict[str, object]:
    """Materialize one closed edit operation and return its review record."""

    if operation_kind == "delete_file":
        operation = candidate.apply_journaled_mutation(
            operation_id=operation_id,
            kind="delete_file",
            repo_path=repo_path,
            replacement=None,
            expected_revision=expected_revision,
            expected_source_sha256=hashlib.sha256(
                (current_content or "").encode("utf-8"),
            ).hexdigest(),
        )
        return {
            "kind": "delete_file",
            "path": repo_path,
            "_journal_operation": operation,
        }
    if operation_kind == "rename_file":
        target_value = args.get("target_path")
        if not isinstance(target_value, str):
            raise ValueError("rename target path is required")
        target_path = normalize_safe_repo_relative_path(target_value)
        if target_path is None or target_path.casefold() == repo_path.casefold():
            raise ValueError("rename target path is invalid")
        operation = candidate.apply_journaled_mutation(
            operation_id=operation_id,
            kind="rename_file",
            repo_path=repo_path,
            replacement=None,
            expected_revision=expected_revision,
            expected_source_sha256=hashlib.sha256(
                (current_content or "").encode("utf-8"),
            ).hexdigest(),
            target_path=target_path,
        )
        return {
            "kind": "rename_file",
            "path": repo_path,
            "target_path": target_path,
            "_journal_operation": operation,
        }
    replacement = args.get("replacement")
    if not isinstance(replacement, str):
        raise ValueError("edit replacement is required")
    if operation_kind == "create_file":
        if current_content is not None:
            raise ValueError("create target already exists")
        new_content = materialize_text_operation(
            safe_path=repo_path,
            kind="replace_file_small",
            source_text="",
            anchor=None,
            content=replacement,
        )
        patch_operation = {
            "kind": "create_file",
            "path": repo_path,
            "content": replacement,
        }
    elif operation_kind == "replace_file_small":
        if current_content is None:
            raise ValueError("replace target is missing")
        new_content = materialize_text_operation(
            safe_path=repo_path,
            kind=operation_kind,
            source_text=current_content,
            anchor=None,
            content=replacement,
        )
        patch_operation = {
            "kind": "replace_file_small",
            "path": repo_path,
            "content": replacement,
        }
    elif operation_kind in {"replace_anchor", "insert_before", "insert_after"}:
        anchor = args.get("anchor")
        if not isinstance(anchor, str) or not anchor or current_content is None:
            raise ValueError("anchor edit requires current content and one anchor")
        if current_content.count(anchor) != 1:
            raise ValueError("anchor edit requires exactly one anchor match")
        patch_kind = (
            "replace" if operation_kind == "replace_anchor" else operation_kind
        )
        new_content = materialize_text_operation(
            safe_path=repo_path,
            kind=patch_kind,
            source_text=current_content,
            anchor=anchor,
            content=replacement,
        )
        patch_operation = {
            "kind": patch_kind,
            "path": repo_path,
            "anchor": anchor,
            "content": replacement,
        }
    else:
        raise ValueError("edit operation is unsupported")
    expected_source_sha256 = None
    if current_content is not None:
        expected_source_sha256 = hashlib.sha256(
            current_content.encode("utf-8"),
        ).hexdigest()
    operation = candidate.apply_journaled_mutation(
        operation_id=operation_id,
        kind=operation_kind,
        repo_path=repo_path,
        replacement=new_content,
        expected_revision=expected_revision,
        expected_source_sha256=expected_source_sha256,
    )
    patch_operation["_journal_operation"] = operation
    return patch_operation
