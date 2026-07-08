"""Supervisor for existing-source modification proposals."""

from __future__ import annotations

from kazusa_ai_chatbot.coding_agent.code_modifying.models import (
    CodeModificationRequest,
    CodeModificationResult,
    ModificationArtifact,
    ModifyingPMDecision,
    ModifyingProgrammerTask,
)
from kazusa_ai_chatbot.coding_agent.code_modifying.product_manager import (
    run_modifying_pm,
)
from kazusa_ai_chatbot.coding_agent.code_modifying.programmer import (
    run_modifying_programmer,
)
from kazusa_ai_chatbot.coding_agent.file_agent import plan_existing_source_files

MAX_PM_DECISIONS = 4
MAX_PROGRAMMER_TASKS = 3
MAX_PROGRAMMER_CONTRACT_REPAIRS = 2
MAX_TARGET_PATHS = 8


async def run(request: CodeModificationRequest) -> CodeModificationResult:
    """Produce structured modification artifacts from source evidence."""

    reading_result = request.get("reading_result")
    repository = request.get("repository")
    if not isinstance(reading_result, dict) or not isinstance(repository, dict):
        result = _failure_result("Modification requires source reading evidence.")
        return result

    evidence = reading_result.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        result = _failure_result("Modification requires at least one evidence row.")
        return result

    file_plan = plan_existing_source_files(
        question=request.get("question", ""),
        repository=repository,
        source_scope=request.get("source_scope", {}),
        reading_result=reading_result,
    )
    if file_plan["status"] != "accepted":
        result = _failure_result("No acceptable existing-source file plan was produced.")
        result["limitations"].extend(_string_list(file_plan.get("missing_owner_signals")))
        result["trace_summary"].append("modifying:file_plan_rejected")
        return result

    trace_summary = ["modifying:file_plan_ready"]
    pm_payload = _pm_payload(
        request=request,
        reading_result=reading_result,
        file_plan=file_plan,
        previous_programmer_reports=[],
    )
    programmer_results: list[dict[str, object]] = []
    collected_artifacts: list[ModificationArtifact] = []
    handoff_repair_sent = False
    programmer_task_count = 0
    limitations: list[str] = []

    for _ in range(MAX_PM_DECISIONS):
        decision = await run_modifying_pm(pm_payload)
        status = decision["status"]
        trace_summary.append(f"modifying_pm:decision={status}")

        if status == "create_programmer_task":
            task = decision.get("programmer_task")
            if task is None:
                result = _failure_result("Modifying PM did not provide a task.")
                result["trace_summary"].extend(trace_summary)
                return result
            handoff_errors = _handoff_validation_errors(
                task=task,
                decision=decision,
                file_plan=file_plan,
                programmer_task_count=programmer_task_count,
                repair_feedback=request.get("repair_feedback"),
            )
            if handoff_errors:
                if not handoff_repair_sent:
                    pm_payload["repair_feedback"] = _handoff_repair_feedback(
                        task=task,
                        handoff_errors=handoff_errors,
                        file_plan=file_plan,
                        repair_feedback=request.get("repair_feedback"),
                    )
                    trace_summary.append("modifying_pm:handoff_repair")
                    handoff_repair_sent = True
                    continue
                result = _failure_result("Modifying PM handoff was invalid.")
                result["limitations"].extend(handoff_errors)
                result["trace_summary"].extend(trace_summary)
                return result

            trace_summary.append(f"modifying_pm:programmer_task={task['task_id']}")
            programmer_payload = _programmer_payload(
                request=request,
                reading_result=reading_result,
                file_plan=file_plan,
                decision=decision,
                task=task,
            )
            programmer_result, repair_trace = await _run_programmer_with_contract_repair(
                programmer_payload,
                required_target_paths=set(task["target_paths"]),
            )
            trace_summary.extend(repair_trace)
            programmer_results.append(programmer_result)
            programmer_task_count += 1
            artifacts = _successful_artifacts(programmer_result.get("artifacts"))
            target_errors = _artifact_target_errors(
                artifacts=artifacts,
                required_target_paths=set(task["target_paths"]),
            )
            content_errors = _artifact_content_errors(
                artifacts=artifacts,
                file_contents=_file_content_by_path(
                    programmer_payload.get("file_contexts")
                ),
            )
            if target_errors or content_errors:
                result = _failure_result("Programmer artifacts missed target paths.")
                result["limitations"].extend(target_errors)
                result["limitations"].extend(content_errors)
                result["trace_summary"].extend(trace_summary)
                return result
            collected_artifacts.extend(artifacts)
            limitations.extend(_limitations_from_programmer_result(programmer_result))
            if artifacts:
                trace_summary.append("modifying_pm:sufficiency=programmer_artifacts_ready")
                result = _success_result(
                    artifacts=collected_artifacts,
                    answer_text=programmer_result.get("answer_text"),
                    limitations=limitations,
                    trace_summary=trace_summary,
                    file_plan=file_plan,
                    programmer_results=programmer_results,
                )
                return result
            pm_payload["previous_programmer_reports"] = _programmer_reports(
                programmer_results
            )
            trace_summary.append("modifying:programmer_no_artifacts")
            if programmer_task_count >= MAX_PROGRAMMER_TASKS:
                break
            continue

        if status == "complete":
            trace_summary.append("modifying_pm:sufficiency=complete")
            if collected_artifacts:
                result = _success_result(
                    artifacts=collected_artifacts,
                    answer_text="Prepared existing-source patch proposal artifacts.",
                    limitations=limitations,
                    trace_summary=trace_summary,
                    file_plan=file_plan,
                    programmer_results=programmer_results,
                )
                return result
            result = _failure_result("Modifying PM completed without artifacts.")
            result["trace_summary"].extend(trace_summary)
            return result

        if status == "request_information":
            result = _needs_user_input_result(
                "Modifying PM requested additional source information."
            )
            result["trace_summary"].extend(trace_summary)
            return result

        if status == "repair_child":
            repair = decision.get("repair_instruction")
            pm_payload["repair_feedback"] = repair or {}
            trace_summary.append("modifying_pm:repair_child")
            continue

        blocker = decision.get("blocker")
        result = _failure_result(_blocker_summary(blocker))
        result["trace_summary"].extend(trace_summary)
        return result

    result = _failure_result("Modification planning did not produce artifacts.")
    result["limitations"].extend(limitations)
    result["trace_summary"].extend(trace_summary)
    return result


def _pm_payload(
    *,
    request: CodeModificationRequest,
    reading_result: dict[str, object],
    file_plan: dict[str, object],
    previous_programmer_reports: list[dict[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "question": request.get("question", ""),
        "source_scope": request.get("source_scope", {}),
        "reading_answer": reading_result.get("answer_text", ""),
        "file_plan": _public_file_plan(file_plan),
        "previous_programmer_reports": previous_programmer_reports,
        "requirements": {
            "required_behavior": _string_list(request.get("required_behavior")),
            "forbidden_changes": _string_list(request.get("forbidden_changes")),
        },
        "limits": {
            "max_pm_decisions": MAX_PM_DECISIONS,
            "max_programmer_tasks": MAX_PROGRAMMER_TASKS,
            "max_target_paths": MAX_TARGET_PATHS,
        },
        "output_contract": _output_contract(),
    }
    repair_feedback = request.get("repair_feedback")
    if isinstance(repair_feedback, dict):
        payload["repair_feedback"] = repair_feedback
    return payload


def _public_file_plan(file_plan: dict[str, object]) -> dict[str, object]:
    return {
        "status": file_plan.get("status"),
        "source_scope": file_plan.get("source_scope", {}),
        "evidence": file_plan.get("evidence", []),
        "ranked_source_owner_candidates": file_plan.get(
            "ranked_source_owner_candidates",
            [],
        ),
        "owned_path_candidates": file_plan.get("owned_path_candidates", []),
        "read_only_path_candidates": file_plan.get("read_only_path_candidates", []),
        "caller_path_candidates": file_plan.get("caller_path_candidates", []),
        "test_or_doc_path_candidates": file_plan.get(
            "test_or_doc_path_candidates",
            [],
        ),
        "missing_owner_signals": file_plan.get("missing_owner_signals", []),
        "limits": file_plan.get("limits", {}),
    }


def _handoff_validation_errors(
    *,
    task: ModifyingProgrammerTask,
    decision: ModifyingPMDecision,
    file_plan: dict[str, object],
    programmer_task_count: int,
    repair_feedback: object,
) -> list[str]:
    errors: list[str] = []
    target_paths = task.get("target_paths", [])
    if not target_paths:
        errors.append("Programmer task must include target_paths.")
        return errors
    target_path_set = set(target_paths)
    if len(target_paths) > MAX_TARGET_PATHS:
        errors.append("Programmer task targets too many files.")
    file_context_paths = _path_set(file_plan.get("file_contexts"))
    owner_paths = set(_string_list(file_plan.get("owned_path_candidates")))
    caller_paths = set(_string_list(file_plan.get("caller_path_candidates")))
    companion_paths = set(_string_list(file_plan.get("test_or_doc_path_candidates")))
    read_only_paths = set(_string_list(decision.get("read_only_paths")))
    repair_constraints = _execution_repair_constraints(repair_feedback)
    required_owner_paths = set(repair_constraints["required_source_owner_paths"])
    protected_paths = set(repair_constraints["protected_verification_paths"])
    writable_companion_paths = companion_paths.difference(read_only_paths)
    allowed_target_paths = file_context_paths.intersection(
        owner_paths | writable_companion_paths
    )
    if required_owner_paths or protected_paths:
        allowed_target_paths = file_context_paths.intersection(
            owner_paths | caller_paths | required_owner_paths
        )
    for path in target_paths:
        if path not in file_context_paths:
            errors.append(f"Programmer target path {path!r} lacks file context.")
            continue
        if path in protected_paths:
            errors.append(
                f"Programmer target path {path!r} is protected verification "
                "evidence; include it in read_only_paths instead."
            )
            continue
        if path in read_only_paths:
            errors.append(
                f"Programmer target path {path!r} is read-only in the PM "
                "decision; remove it from target_paths."
            )
            continue
        if path not in allowed_target_paths:
            errors.append(f"Programmer target path {path!r} is not handoff-owned.")
    missing_required_paths = sorted(required_owner_paths.difference(target_path_set))
    for path in missing_required_paths:
        errors.append(f"Programmer task omitted required source-owner path {path!r}.")
    companion_target_paths = companion_paths.difference(
        protected_paths | read_only_paths
    )
    execution_repair_mode = bool(required_owner_paths or protected_paths)
    if programmer_task_count == 0 and owner_paths and not owner_paths.intersection(
        target_paths
    ):
        errors.append(
            "First programmer task must include at least one source-owner path."
        )
    if (
        programmer_task_count == 0
        and not execution_repair_mode
        and companion_target_paths
        and not companion_target_paths.intersection(target_paths)
    ):
        errors.append(
            "First programmer task must include at least one focused companion "
            "test or doc path."
        )
    return errors


def _handoff_repair_feedback(
    *,
    task: ModifyingProgrammerTask,
    handoff_errors: list[str],
    file_plan: dict[str, object],
    repair_feedback: object,
) -> dict[str, object]:
    repair_constraints = _execution_repair_constraints(repair_feedback)
    allowed_source_paths = _execution_allowed_source_target_paths(
        file_plan=file_plan,
        repair_constraints=repair_constraints,
    )
    feedback = {
        "child_id": task.get("task_id", ""),
        "feedback_source": "handoff_validation",
        "feedback": " ".join(handoff_errors),
        "expected_correction": (
            "Return a create_programmer_task decision whose target_paths include "
            "the relevant source-owner path and focused companion tests or docs."
        ),
    }
    if repair_constraints["required_source_owner_paths"]:
        feedback["required_source_owner_paths"] = repair_constraints[
            "required_source_owner_paths"
        ]
        feedback["protected_verification_paths"] = repair_constraints[
            "protected_verification_paths"
        ]
        feedback["allowed_source_target_paths"] = allowed_source_paths
        feedback["expected_correction"] = (
            "Return a create_programmer_task decision whose target_paths include "
            "required_source_owner_paths and are drawn only from "
            "allowed_source_target_paths. Include protected_verification_paths, "
            "failed_paths, tests, and docs only in read_only_paths."
        )
    return feedback


def _programmer_payload(
    *,
    request: CodeModificationRequest,
    reading_result: dict[str, object],
    file_plan: dict[str, object],
    decision: ModifyingPMDecision,
    task: ModifyingProgrammerTask,
) -> dict[str, object]:
    target_paths = set(task["target_paths"])
    read_only_paths = set(decision.get("read_only_paths", []))
    repair_constraints = _execution_repair_constraints(request.get("repair_feedback"))
    protected_paths = set(repair_constraints["protected_verification_paths"])
    selected_paths = target_paths | read_only_paths | protected_paths
    file_contexts = _selected_file_contexts(
        file_plan=file_plan,
        selected_paths=selected_paths,
    )
    payload = {
        "question": request.get("question", ""),
        "source_scope": request.get("source_scope", {}),
        "reading_answer": reading_result.get("answer_text", ""),
        "evidence": file_plan.get("evidence", []),
        "file_contexts": file_contexts,
        "file_plan": _public_file_plan(file_plan),
        "ownership_guidance": _ownership_guidance_from_file_plan(file_plan),
        "programmer_task": task,
        "required_behavior": task.get("required_behavior", []),
        "forbidden_changes": task.get("forbidden_changes", []),
        "output_contract": _output_contract(),
    }
    repair_feedback = request.get("repair_feedback")
    if isinstance(repair_feedback, dict):
        payload["repair_feedback"] = repair_feedback
    return payload


def _execution_repair_constraints(
    repair_feedback: object,
) -> dict[str, list[str]]:
    constraints: dict[str, list[str]] = {
        "required_source_owner_paths": [],
        "protected_verification_paths": [],
    }
    if not isinstance(repair_feedback, dict):
        return constraints
    if repair_feedback.get("feedback_source") != "execution_verification":
        return constraints

    constraints["required_source_owner_paths"] = _string_list(
        repair_feedback.get("required_source_owner_paths")
    )
    constraints["protected_verification_paths"] = _string_list(
        repair_feedback.get("protected_verification_paths")
    )
    return constraints


def _execution_allowed_source_target_paths(
    *,
    file_plan: dict[str, object],
    repair_constraints: dict[str, list[str]],
) -> list[str]:
    if not repair_constraints["required_source_owner_paths"]:
        return []

    file_context_paths = _path_set(file_plan.get("file_contexts"))
    owner_paths = set(_string_list(file_plan.get("owned_path_candidates")))
    caller_paths = set(_string_list(file_plan.get("caller_path_candidates")))
    required_paths = set(repair_constraints["required_source_owner_paths"])
    protected_paths = set(repair_constraints["protected_verification_paths"])
    allowed_paths = file_context_paths.intersection(
        owner_paths | caller_paths | required_paths
    )
    allowed_paths = allowed_paths.difference(protected_paths)
    sorted_paths = sorted(allowed_paths)
    return sorted_paths


def _selected_file_contexts(
    *,
    file_plan: dict[str, object],
    selected_paths: set[str],
) -> list[dict[str, object]]:
    contexts_value = file_plan.get("file_contexts")
    if not isinstance(contexts_value, list):
        return []
    selected_contexts: list[dict[str, object]] = []
    for context in contexts_value:
        if not isinstance(context, dict):
            continue
        path_value = context.get("path")
        if not isinstance(path_value, str):
            continue
        if path_value in selected_paths:
            selected_contexts.append(context)
    return selected_contexts


async def _run_programmer_with_contract_repair(
    payload: dict[str, object],
    required_target_paths: set[str],
) -> tuple[dict[str, object], list[str]]:
    trace: list[str] = []
    programmer_result = await run_modifying_programmer(payload)
    for _ in range(MAX_PROGRAMMER_CONTRACT_REPAIRS):
        contract_errors = _programmer_contract_errors(
            programmer_result=programmer_result,
            required_target_paths=required_target_paths,
            file_contents=_file_content_by_path(payload.get("file_contexts")),
        )
        if not contract_errors:
            return programmer_result, trace

        repair_payload = dict(payload)
        repair_payload["repair_feedback"] = {
            "feedback_source": "contract_validation",
            "validation": {
                "errors": contract_errors,
            },
            "required_target_paths": sorted(required_target_paths),
            "previous_modification_artifacts": programmer_result.get("artifacts", []),
            "instruction": (
                "Return a corrected complete artifact list that fixes the contract "
                "validation errors. Include one succeeded or blocked artifact for "
                "every required_target_paths entry. For missing test target paths, "
                "extend the existing test file using its local fake, fixture, or "
                "monkeypatch pattern instead of blocking because the exact new test "
                "case is absent. Do not repeat blocked artifact content."
            ),
        }
        programmer_result = await run_modifying_programmer(repair_payload)
        trace.append("modifying:programmer_contract_repair")
    return programmer_result, trace


def _programmer_contract_errors(
    *,
    programmer_result: dict[str, object],
    required_target_paths: set[str],
    file_contents: dict[str, str],
) -> list[str]:
    errors = _blocked_artifact_errors(programmer_result.get("artifacts"))
    artifacts = _successful_artifacts(programmer_result.get("artifacts"))
    errors.extend(_artifact_target_errors(
        artifacts=artifacts,
        required_target_paths=required_target_paths,
    ))
    errors.extend(_artifact_content_errors(
        artifacts=artifacts,
        file_contents=file_contents,
    ))
    return errors


def _artifact_target_errors(
    *,
    artifacts: list[ModificationArtifact],
    required_target_paths: set[str],
) -> list[str]:
    if not required_target_paths:
        return []
    artifact_paths = {
        artifact["target_path"]
        for artifact in artifacts
    }
    missing_paths = sorted(required_target_paths.difference(artifact_paths))
    errors = [
        f"Programmer task target {path!r} did not receive a succeeded artifact."
        for path in missing_paths
    ]
    return errors


def _artifact_content_errors(
    *,
    artifacts: list[ModificationArtifact],
    file_contents: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    for artifact in artifacts:
        target_path = artifact["target_path"]
        current_content = file_contents.get(target_path, "")
        replacement = artifact["replacement_or_insert_content"]
        operation_kind = artifact["operation_kind"]
        exact_anchor = artifact.get("exact_anchor", "")
        if operation_kind == "replace_file_small" and replacement == current_content:
            errors.append(f"{target_path}: replacement content is unchanged.")
            continue
        if operation_kind == "replace" and replacement == exact_anchor:
            errors.append(f"{target_path}: replacement content is unchanged.")
    return errors


def _file_content_by_path(value: object) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    contents: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        content = item.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            continue
        contents[path] = content
    return contents


def _blocked_artifact_errors(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    errors: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        blocker = item.get("blocker")
        target_path = item.get("target_path")
        if status != "blocked" or not isinstance(blocker, str):
            continue
        if isinstance(target_path, str) and target_path:
            errors.append(f"{target_path}: {blocker}")
            continue
        errors.append(blocker)
    return errors


def _ownership_guidance_from_file_plan(
    file_plan: dict[str, object],
) -> dict[str, object]:
    guidance = {
        "source_owner_paths": file_plan.get("owned_path_candidates", []),
        "test_or_doc_paths": file_plan.get("test_or_doc_path_candidates", []),
        "caller_paths": file_plan.get("caller_path_candidates", []),
        "rule": (
            "Modify the source owner path for runtime behavior. Modify caller "
            "paths when integration wiring fails. Update focused companion "
            "tests or docs only when requested or made stale."
        ),
    }
    return guidance


def _output_contract() -> dict[str, object]:
    return {
        "operation_kinds": [
            "replace",
            "insert_before",
            "insert_after",
            "replace_file_small",
        ],
        "raw_diffs_allowed": False,
        "command_execution_allowed": False,
    }


def _success_result(
    *,
    artifacts: list[ModificationArtifact],
    answer_text: object,
    limitations: list[str],
    trace_summary: list[str],
    file_plan: dict[str, object],
    programmer_results: list[dict[str, object]],
) -> CodeModificationResult:
    if not isinstance(answer_text, str) or not answer_text.strip():
        answer_text = "Prepared existing-source patch proposal artifacts."
    result: CodeModificationResult = {
        "status": "succeeded",
        "mode": _mode_from_artifacts(artifacts),
        "answer_text": answer_text,
        "modification_artifacts": artifacts,
        "created_files": [],
        "changed_files": _changed_files_from_artifacts(artifacts),
        "limitations": limitations,
        "trace_summary": [
            *trace_summary,
            f"modifying:artifacts={len(artifacts)}",
        ],
        "trace": {
            "file_context_count": len(_path_set(file_plan.get("file_contexts"))),
            "file_plan": _public_file_plan(file_plan),
            "programmer_reports": _programmer_reports(programmer_results),
        },
    }
    return result


def _programmer_reports(
    programmer_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for index, result in enumerate(programmer_results, start=1):
        artifacts = _successful_artifacts(result.get("artifacts"))
        reports.append({
            "programmer_turn": index,
            "artifact_count": len(artifacts),
            "target_paths": [
                artifact["target_path"]
                for artifact in artifacts
            ],
            "limitations": _limitations_from_programmer_result(result),
        })
    return reports


def _path_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    paths: set[str] = set()
    for item in value:
        if isinstance(item, str):
            paths.add(item)
            continue
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if isinstance(path, str):
            paths.add(path)
    return paths


def _blocker_summary(value: object) -> str:
    if isinstance(value, dict):
        summary = value.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary
    return "Modifying PM blocked the request."


def _needs_user_input_result(limitation: str) -> CodeModificationResult:
    result: CodeModificationResult = {
        "status": "needs_user_input",
        "mode": "edit_existing_repository",
        "answer_text": "",
        "modification_artifacts": [],
        "created_files": [],
        "changed_files": [],
        "limitations": [limitation],
        "trace_summary": ["modifying:needs_user_input"],
    }
    return result


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        strings.append(text)
    return strings


def _evidence_with_ids(evidence_rows: list[object]) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for index, row in enumerate(evidence_rows, start=1):
        if not isinstance(row, dict):
            continue
        evidence_row = dict(row)
        evidence_row["evidence_id"] = f"evidence-{index}"
        evidence.append(evidence_row)
    return evidence


def _file_contexts(
    *,
    repo_root: Path,
    evidence_rows: list[object],
) -> list[dict[str, object]]:
    contexts: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for row in evidence_rows:
        if not isinstance(row, dict):
            continue
        path_text = row.get("path")
        if not isinstance(path_text, str):
            continue
        safe_path = _safe_text_path(path_text)
        if safe_path is None or safe_path in seen_paths:
            continue
        file_path = ensure_path_inside(repo_root / safe_path, repo_root)
        if not file_path.is_file():
            continue
        content = file_path.read_text(encoding="utf-8", errors="replace")
        contexts.append({
            "path": safe_path,
            "content": content[:MAX_CONTEXT_FILE_CHARS],
            "truncated": len(content) > MAX_CONTEXT_FILE_CHARS,
        })
        seen_paths.add(safe_path)
        if len(contexts) >= MAX_CONTEXT_FILES:
            break
    return contexts


def _safe_text_path(path_text: str) -> str | None:
    normalized = path_text.replace("\\", "/").strip()
    if not normalized:
        return None
    if not is_safe_repo_relative_path(normalized):
        return None
    if is_binary_like_path(normalized) or is_secret_like_path(normalized):
        return None
    return normalized


def _ownership_guidance(
    file_contexts: list[dict[str, object]],
) -> dict[str, object]:
    source_owner_paths: list[str] = []
    test_or_doc_paths: list[str] = []
    for context in file_contexts:
        path_value = context.get("path")
        if not isinstance(path_value, str):
            continue
        lowered_path = path_value.casefold()
        if _is_test_or_doc_path(lowered_path):
            test_or_doc_paths.append(path_value)
            continue
        source_owner_paths.append(path_value)
    guidance = {
        "source_owner_paths": source_owner_paths,
        "test_or_doc_paths": test_or_doc_paths,
        "rule": (
            "When requested behavior maps to a helper/source owner path, "
            "modify that owner path and its focused tests instead of only "
            "changing a caller or wrapper."
        ),
    }
    return guidance


def _is_test_or_doc_path(lowered_path: str) -> bool:
    if lowered_path.startswith("tests/") or "/tests/" in lowered_path:
        return True
    if lowered_path.endswith(".md") or lowered_path.endswith(".rst"):
        return True
    return False


def _successful_artifacts(value: object) -> list[ModificationArtifact]:
    if not isinstance(value, list):
        return []
    artifacts: list[ModificationArtifact] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if status != "succeeded":
            continue
        artifacts.append(item)
    return artifacts


def _changed_files_from_artifacts(
    artifacts: list[ModificationArtifact],
) -> list[dict[str, str]]:
    changed_files: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for artifact in artifacts:
        path = artifact["target_path"]
        if path in seen_paths:
            continue
        summary = artifact["operation_summary"]
        changed_files.append({
            "path": path,
            "change_type": "modify",
            "summary": summary or "Existing-source modification.",
        })
        seen_paths.add(path)
    return changed_files


def _mode_from_artifacts(artifacts: list[ModificationArtifact]) -> str:
    if any(artifact["operation_kind"] == "replace_file_small" for artifact in artifacts):
        return "edit_existing_repository"
    return "edit_existing_repository"


def _limitations_from_programmer_result(
    programmer_result: dict[str, object],
) -> list[str]:
    limitations = programmer_result.get("limitations")
    if not isinstance(limitations, list):
        return []
    strings: list[str] = []
    for item in limitations:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        strings.append(text)
    return strings


def _failure_result(limitation: str) -> CodeModificationResult:
    result: CodeModificationResult = {
        "status": "failed",
        "mode": "edit_existing_repository",
        "answer_text": "",
        "modification_artifacts": [],
        "created_files": [],
        "changed_files": [],
        "limitations": [limitation],
        "trace_summary": ["modifying:failed"],
    }
    return result
