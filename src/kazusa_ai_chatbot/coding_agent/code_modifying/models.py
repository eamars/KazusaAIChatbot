"""Contracts and validators for existing-source modification proposals."""

import re
from typing import Literal, NotRequired, TypedDict

from kazusa_ai_chatbot.coding_agent.code_fetching.github import (
    is_safe_repo_relative_path,
)
from kazusa_ai_chatbot.coding_agent.code_patching.models import (
    PatchOperation,
)
from kazusa_ai_chatbot.coding_agent.code_reading.models import (
    CodeReadingResult,
)

ModificationStatus = Literal["succeeded", "failed", "needs_user_input", "rejected"]
ModificationArtifactStatus = Literal["succeeded", "blocked"]
ModificationOperationKind = Literal[
    "create_file",
    "replace",
    "insert_before",
    "insert_after",
    "replace_file_small",
]
ModifyingPMStatus = Literal[
    "request_information",
    "create_programmer_task",
    "repair_child",
    "complete",
    "blocked",
]

ALLOWED_OPERATION_KINDS = {
    "create_file",
    "replace",
    "insert_before",
    "insert_after",
    "replace_file_small",
}
ALLOWED_REPAIR_FEEDBACK_SOURCES = {
    "parser_validation",
    "handoff_validation",
    "patch_validation",
    "review_materialization",
    "contract_validation",
    "execution_verification",
}
RAW_DIFF_MARKERS = ("diff --git ", "--- a/", "+++ b/")
INDENTED_DEF_RE = re.compile(r"^\s+def\s+\w+\(([^)]*)\)\s*(?:->[^:]+)?:")


class CodeModificationRequest(TypedDict, total=False):
    """Existing-source modification request from the top-level supervisor."""

    question: str
    reading_result: CodeReadingResult
    repository: dict[str, object]
    source_scope: dict[str, object]
    workspace_root: str
    preferred_language: str
    max_answer_chars: int
    max_artifact_chars: int
    supervisor_facts: list[dict[str, object]]
    repair_feedback: dict[str, object]
    required_behavior: list[str]
    forbidden_changes: list[str]


class ModificationArtifact(TypedDict, total=False):
    """One structured existing-file modification artifact."""

    artifact_id: str
    status: ModificationArtifactStatus
    task_id: str
    target_path: str
    evidence_ids: list[str]
    operation_kind: ModificationOperationKind
    exact_anchor: str
    replacement_or_insert_content: str
    operation_summary: str
    risk_notes: list[str]
    tests_or_docs_to_update: list[str]
    blocker: str


class CodeModificationResult(TypedDict):
    """Modification result before deterministic patch assembly."""

    status: ModificationStatus
    mode: str
    answer_text: str
    modification_artifacts: list[ModificationArtifact]
    created_files: list[dict[str, str]]
    changed_files: list[dict[str, str]]
    limitations: list[str]
    trace_summary: list[str]
    trace: NotRequired[dict[str, object]]


class ModifyingProgrammerTask(TypedDict):
    """One bounded existing-source programmer task selected by a PM."""

    task_id: str
    target_paths: list[str]
    change_goal: str
    required_behavior: list[str]
    forbidden_changes: list[str]
    consumed_interfaces: list[str]
    expected_operations: list[str]
    acceptance_checks: list[str]
    local_risks: list[str]


class ModifyingRepairInstruction(TypedDict, total=False):
    """Structural correction request for one direct child."""

    child_id: str
    feedback_source: str
    feedback: str
    expected_correction: str


class ModifyingBlocker(TypedDict):
    """Terminal blocker for a PM decision."""

    summary: str
    missing_facts: list[str]
    why_information_request_is_not_enough: str


class ModifyingPMDecision(TypedDict, total=False):
    """Normalized PM lifecycle decision."""

    status: ModifyingPMStatus
    reason: str
    owned_paths: list[str]
    read_only_paths: list[str]
    required_evidence_ids: list[str]
    programmer_task: ModifyingProgrammerTask | None
    repair_instruction: ModifyingRepairInstruction | None
    blocker: ModifyingBlocker | None


def normalize_modification_artifact(
    raw_artifact: dict[str, object],
) -> ModificationArtifact:
    """Validate one modifying programmer artifact."""

    artifact_id = _bounded_text(raw_artifact.get("artifact_id")) or "artifact"
    task_id = _bounded_text(raw_artifact.get("task_id")) or artifact_id
    target_path = _bounded_text(raw_artifact.get("target_path"))
    operation_kind = _bounded_text(raw_artifact.get("operation_kind"))
    evidence_ids = _string_list(raw_artifact.get("evidence_ids"))
    anchor = _bounded_multiline_text(raw_artifact.get("exact_anchor"))
    content = _bounded_multiline_text(
        raw_artifact.get("replacement_or_insert_content")
    )
    summary = _bounded_text(raw_artifact.get("operation_summary"))
    risks = _string_list(raw_artifact.get("risk_notes"))
    related_updates = _string_list(raw_artifact.get("tests_or_docs_to_update"))

    blocker = _artifact_blocker(
        target_path=target_path,
        operation_kind=operation_kind,
        evidence_ids=evidence_ids,
        anchor=anchor,
        content=content,
        summary=summary,
    )
    status: ModificationArtifactStatus = "succeeded"
    if blocker:
        status = "blocked"

    artifact: ModificationArtifact = {
        "artifact_id": artifact_id,
        "status": status,
        "task_id": task_id,
        "target_path": target_path,
        "evidence_ids": evidence_ids,
        "operation_kind": operation_kind,
        "exact_anchor": anchor,
        "replacement_or_insert_content": content,
        "operation_summary": summary,
        "risk_notes": risks,
        "tests_or_docs_to_update": related_updates,
    }
    if blocker:
        artifact["blocker"] = blocker
    return artifact


def normalize_modifying_pm_decision(
    raw_decision: dict[str, object],
) -> ModifyingPMDecision:
    """Validate one modifying PM lifecycle decision."""

    status_text = _bounded_text(raw_decision.get("status"))
    reason = _bounded_text(raw_decision.get("reason"))
    owned_paths = _safe_path_list(raw_decision.get("owned_paths"))
    read_only_paths = _safe_path_list(raw_decision.get("read_only_paths"))
    evidence_ids = _string_list(raw_decision.get("required_evidence_ids"))

    if status_text not in _allowed_pm_statuses():
        decision = _blocked_pm_decision("unsupported status", reason=reason)
        return decision

    if status_text == "repair_child":
        repair = _repair_instruction(raw_decision.get("repair_instruction"))
        feedback_source = repair.get("feedback_source", "")
        if feedback_source not in ALLOWED_REPAIR_FEEDBACK_SOURCES:
            decision = _blocked_pm_decision(
                "repair_child accepts only structural contract feedback",
                reason=reason,
            )
            return decision
        decision = _base_pm_decision(
            status="repair_child",
            reason=reason,
            owned_paths=owned_paths,
            read_only_paths=read_only_paths,
            evidence_ids=evidence_ids,
        )
        decision["repair_instruction"] = repair
        return decision

    if status_text == "create_programmer_task":
        task = _programmer_task(raw_decision.get("programmer_task"))
        if task is None:
            decision = _blocked_pm_decision("programmer_task is required", reason=reason)
            return decision
        decision = _base_pm_decision(
            status="create_programmer_task",
            reason=reason,
            owned_paths=owned_paths,
            read_only_paths=read_only_paths,
            evidence_ids=evidence_ids,
        )
        decision["programmer_task"] = task
        return decision

    decision = _base_pm_decision(
        status=status_text,
        reason=reason,
        owned_paths=owned_paths,
        read_only_paths=read_only_paths,
        evidence_ids=evidence_ids,
    )
    return decision


def artifact_to_patch_operation(
    artifact: ModificationArtifact,
) -> PatchOperation | None:
    """Project one successful modification artifact into a patch operation."""

    if artifact["status"] != "succeeded":
        return None
    operation: PatchOperation = {
        "operation_id": artifact["artifact_id"],
        "kind": artifact["operation_kind"],
        "path": artifact["target_path"],
        "content": artifact["replacement_or_insert_content"],
        "summary": artifact["operation_summary"],
        "evidence_ids": artifact["evidence_ids"],
    }
    if artifact["operation_kind"] in ("replace", "insert_before", "insert_after"):
        operation["anchor"] = artifact["exact_anchor"]
    if artifact["operation_kind"] == "replace_file_small":
        operation["full_file_rationale"] = artifact["operation_summary"]
    return operation


def _artifact_blocker(
    *,
    target_path: str,
    operation_kind: str,
    evidence_ids: list[str],
    anchor: str,
    content: str,
    summary: str,
) -> str:
    if not target_path or not is_safe_repo_relative_path(target_path):
        return "target path is unsafe or missing"
    if operation_kind not in ALLOWED_OPERATION_KINDS:
        return "operation kind is unsupported"
    if not evidence_ids:
        return "evidence ids are required"
    if operation_kind not in ("create_file", "replace_file_small") and not anchor:
        return "exact anchor is required"
    if not content:
        return "replacement or insertion content is required"
    if operation_kind == "create_file" and not summary:
        return "operation summary is required"
    if _contains_raw_diff(content):
        return "raw diff content is not accepted"
    if target_path.endswith(".py") and _contains_indented_import(content):
        return "python imports must be top-level"
    if target_path.endswith(".py") and _contains_method_without_receiver(content):
        return "indented instance methods must keep self or cls"
    return ""


def _contains_raw_diff(text: str) -> bool:
    return any(marker in text for marker in RAW_DIFF_MARKERS)


def _contains_indented_import(text: str) -> bool:
    lines = text.splitlines()
    for line in lines:
        stripped = line.lstrip(" ")
        if line == stripped:
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            return True
    return False


def _contains_method_without_receiver(text: str) -> bool:
    lines = text.splitlines()
    for line in lines:
        match = INDENTED_DEF_RE.match(line)
        if match is None:
            continue
        arguments = match.group(1).strip()
        if not arguments:
            return True
        first_argument = arguments.split(",", 1)[0].strip()
        if first_argument in {"self", "cls"}:
            continue
        return True
    return False


def _allowed_pm_statuses() -> set[str]:
    statuses = {
        "request_information",
        "create_programmer_task",
        "repair_child",
        "complete",
        "blocked",
    }
    return statuses


def _base_pm_decision(
    *,
    status: str,
    reason: str,
    owned_paths: list[str],
    read_only_paths: list[str],
    evidence_ids: list[str],
) -> ModifyingPMDecision:
    decision: ModifyingPMDecision = {
        "status": status,
        "reason": reason,
        "owned_paths": owned_paths,
        "read_only_paths": read_only_paths,
        "required_evidence_ids": evidence_ids,
        "programmer_task": None,
        "repair_instruction": None,
        "blocker": None,
    }
    return decision


def _blocked_pm_decision(summary: str, *, reason: str) -> ModifyingPMDecision:
    decision = _base_pm_decision(
        status="blocked",
        reason=reason,
        owned_paths=[],
        read_only_paths=[],
        evidence_ids=[],
    )
    decision["blocker"] = {
        "summary": summary,
        "missing_facts": [],
        "why_information_request_is_not_enough": summary,
    }
    return decision


def _programmer_task(value: object) -> ModifyingProgrammerTask | None:
    if not isinstance(value, dict):
        return None
    task: ModifyingProgrammerTask = {
        "task_id": _bounded_text(value.get("task_id")),
        "target_paths": _safe_path_list(value.get("target_paths")),
        "change_goal": _bounded_text(value.get("change_goal")),
        "required_behavior": _string_list(value.get("required_behavior")),
        "forbidden_changes": _string_list(value.get("forbidden_changes")),
        "consumed_interfaces": _string_list(value.get("consumed_interfaces")),
        "expected_operations": _string_list(value.get("expected_operations")),
        "acceptance_checks": _string_list(value.get("acceptance_checks")),
        "local_risks": _string_list(value.get("local_risks")),
    }
    if not task["task_id"] or not task["target_paths"]:
        return None
    return task


def _repair_instruction(value: object) -> ModifyingRepairInstruction:
    if not isinstance(value, dict):
        return {}
    repair: ModifyingRepairInstruction = {
        "child_id": _bounded_text(value.get("child_id")),
        "feedback_source": _bounded_text(value.get("feedback_source")),
        "feedback": _bounded_text(value.get("feedback")),
        "expected_correction": _bounded_text(value.get("expected_correction")),
    }
    return repair


def _safe_path_list(value: object) -> list[str]:
    paths: list[str] = []
    for item in _object_list(value):
        path = _bounded_text(item)
        if not path or not is_safe_repo_relative_path(path):
            continue
        paths.append(path)
    return paths


def _string_list(value: object) -> list[str]:
    strings: list[str] = []
    for item in _object_list(value):
        text = _bounded_text(item)
        if not text:
            continue
        strings.append(text)
    return strings


def _object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        return []
    return value


def _bounded_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.strip().split())
    return text[:1000]


def _bounded_multiline_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip("\ufeff")
    return text[:24000]
