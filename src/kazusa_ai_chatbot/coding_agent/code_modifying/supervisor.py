"""Supervisor for existing-source modification proposals."""

from __future__ import annotations

from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_fetching.github import (
    is_safe_repo_relative_path,
)
from kazusa_ai_chatbot.coding_agent.code_modifying.models import (
    CodeModificationRequest,
    CodeModificationResult,
    ModificationArtifact,
)
from kazusa_ai_chatbot.coding_agent.code_modifying.programmer import (
    run_modifying_programmer,
)
from kazusa_ai_chatbot.coding_agent.code_reading.planner import (
    is_binary_like_path,
    is_secret_like_path,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import ensure_path_inside

MAX_CONTEXT_FILES = 10
MAX_CONTEXT_FILE_CHARS = 16000


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

    local_root_text = repository.get("local_root")
    if not isinstance(local_root_text, str) or not local_root_text:
        result = _failure_result("Modification requires a resolved local source root.")
        return result

    repo_root = Path(local_root_text).expanduser().resolve(strict=True)
    file_contexts = _file_contexts(repo_root=repo_root, evidence_rows=evidence)
    if not file_contexts:
        result = _failure_result("No safe text file context was available.")
        return result

    evidence_with_ids = _evidence_with_ids(evidence)
    payload = {
        "question": request.get("question", ""),
        "source_scope": request.get("source_scope", {}),
        "reading_answer": reading_result.get("answer_text", ""),
        "evidence": evidence_with_ids,
        "file_contexts": file_contexts,
        "ownership_guidance": _ownership_guidance(file_contexts),
        "output_contract": {
            "operation_kinds": [
                "replace",
                "insert_before",
                "insert_after",
                "replace_file_small",
            ],
            "raw_diffs_allowed": False,
            "command_execution_allowed": False,
        },
    }
    repair_feedback = request.get("repair_feedback")
    if isinstance(repair_feedback, dict):
        payload["repair_feedback"] = repair_feedback
    programmer_result = await run_modifying_programmer(payload)
    artifacts = _successful_artifacts(programmer_result.get("artifacts"))
    if not artifacts:
        limitations = _limitations_from_programmer_result(programmer_result)
        result = _failure_result("No valid modification artifacts were produced.")
        result["limitations"].extend(limitations)
        result["trace_summary"].append("modifying:programmer_no_artifacts")
        return result

    answer_text = programmer_result.get("answer_text")
    if not isinstance(answer_text, str) or not answer_text:
        answer_text = "Prepared existing-source patch proposal artifacts."

    result: CodeModificationResult = {
        "status": "succeeded",
        "mode": _mode_from_artifacts(artifacts),
        "answer_text": answer_text,
        "modification_artifacts": artifacts,
        "created_files": [],
        "changed_files": _changed_files_from_artifacts(artifacts),
        "limitations": _limitations_from_programmer_result(programmer_result),
        "trace_summary": [
            "modifying:source_context_ready",
            f"modifying:artifacts={len(artifacts)}",
        ],
        "trace": {
            "file_context_count": len(file_contexts),
            "programmer_raw_output": programmer_result.get("raw_output", ""),
        },
    }
    return result


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
