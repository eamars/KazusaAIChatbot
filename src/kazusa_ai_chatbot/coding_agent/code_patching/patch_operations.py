"""Compile LLM-selected edit operations into unified diff artifacts."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_patching.models import (
    CanonicalPatchOperationRecord,
    ChangedFileSummary,
    CreatedFileSummary,
    PatchArtifact,
    PatchOperation,
)
from kazusa_ai_chatbot.coding_agent.code_patching.patch_validation import (
    _safe_repo_relative_path,
)
from kazusa_ai_chatbot.coding_agent.safety import confined_managed_repo_path

REPLACE_FILE_SMALL_MAX_CHARS = 20000
TEXT_DOCUMENT_SUFFIXES = {".md", ".markdown", ".rst", ".txt"}
PYTHON_SUFFIXES = {".py", ".pyi"}
STRUCTURAL_TEXT_LINE_RE = re.compile(
    r"^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s|```|~~~|\|)"
)


def build_canonical_operation_records(
    *,
    repo_root: Path | None,
    patch_operations: list[PatchOperation],
    candidate_revision: int,
) -> list[CanonicalPatchOperationRecord]:
    """Build candidate-bound canonical records for the ordered edit sequence."""

    records: list[CanonicalPatchOperationRecord] = []
    candidate_content: dict[str, str | None] = {}
    for sequence, operation in enumerate(patch_operations, start=1):
        kind = operation.get("kind")
        source_path = _safe_repo_relative_path(operation.get("path", ""))
        if kind not in {
            "create_file",
            "insert_before",
            "insert_after",
            "replace",
            "replace_file_small",
            "delete_file",
            "rename_file",
        } or source_path is None:
            raise ValueError("canonical operation kind or source path is invalid")
        source_text, source_exists = _candidate_text(
            candidate_content=candidate_content,
            repo_root=repo_root,
            safe_path=source_path,
        )
        target_path: str | None = source_path
        expected_source_sha256: str | None = None
        result_sha256: str | None = None
        content_sha256: str | None = None
        if kind == "create_file":
            if source_exists:
                raise ValueError("canonical create target already exists")
            content = operation.get("content")
            if not isinstance(content, str) or not content:
                raise ValueError("canonical create content is invalid")
            result_text = _with_trailing_newline(content)
            candidate_content[source_path] = result_text
            source_record_path: str | None = None
            result_sha256 = hashlib.sha256(result_text.encode("utf-8")).hexdigest()
            content_sha256 = result_sha256
        elif not source_exists:
            raise ValueError("canonical operation source path is missing")
        else:
            source_record_path = source_path
            expected_source_sha256 = hashlib.sha256(
                source_text.encode("utf-8")
            ).hexdigest()
            supplied_hash = operation.get("expected_source_sha256")
            if supplied_hash is not None and supplied_hash != expected_source_sha256:
                raise ValueError("canonical operation source hash is stale")
        if kind == "rename_file":
            target_path = _safe_repo_relative_path(operation.get("target_path", ""))
            if target_path is None or target_path == source_path:
                raise ValueError("canonical rename target path is invalid")
            _, target_exists = _candidate_text(
                candidate_content=candidate_content,
                repo_root=repo_root,
                safe_path=target_path,
            )
            if target_exists:
                raise ValueError("canonical rename target path already exists")
            candidate_content[source_path] = None
            candidate_content[target_path] = source_text
            result_sha256 = expected_source_sha256
            content_sha256 = expected_source_sha256
        elif kind == "delete_file":
            target_path = None
            candidate_content[source_path] = None
            content_sha256 = expected_source_sha256
        elif kind != "create_file":
            result_text = _apply_record_content(
                kind=kind,
                safe_path=source_path,
                source_text=source_text,
                operation=operation,
            )
            candidate_content[source_path] = result_text
            result_sha256 = hashlib.sha256(result_text.encode("utf-8")).hexdigest()
            content_sha256 = result_sha256
        operation_id = operation.get("operation_id")
        if not isinstance(operation_id, str) or not operation_id:
            operation_id = f"operation-{sequence}"
        operation_revision = operation.get("expected_candidate_revision")
        if not isinstance(operation_revision, int):
            operation_revision = candidate_revision
        record: CanonicalPatchOperationRecord = {
            "operation_id": operation_id,
            "kind": kind,
            "source_path": source_record_path,
            "target_path": target_path,
            "expected_source_sha256": expected_source_sha256,
            "expected_candidate_revision": operation_revision,
            "result_sha256": result_sha256,
            "content_sha256": content_sha256,
        }
        records.append(record)
    return records


def _candidate_text(
    *,
    candidate_content: dict[str, str | None],
    repo_root: Path | None,
    safe_path: str,
) -> tuple[str, bool]:
    """Read the current ordered-operation view for canonical record building."""

    if safe_path in candidate_content:
        content = candidate_content[safe_path]
        return (content or "", content is not None)
    original = _read_original(repo_root=repo_root, safe_path=safe_path)
    return original


def _apply_record_content(
    *,
    kind: str,
    safe_path: str,
    source_text: str,
    operation: PatchOperation,
) -> str:
    """Apply one text operation to the canonical in-memory candidate view."""

    content = operation.get("content")
    if not isinstance(content, str):
        raise ValueError("canonical operation content is invalid")
    anchor = operation.get("anchor")
    materialized = materialize_text_operation(
        safe_path=safe_path,
        kind=kind,
        source_text=source_text,
        anchor=anchor if isinstance(anchor, str) else None,
        content=content,
    )
    return materialized


def materialize_text_operation(
    *,
    safe_path: str,
    kind: str,
    source_text: str,
    anchor: str | None,
    content: str,
) -> str:
    """Apply one canonical text operation with review-identical semantics."""

    if kind == "replace_file_small":
        normalized_content = _with_trailing_newline(content)
        return normalized_content
    if anchor is None or source_text.count(anchor) != 1:
        raise ValueError("canonical operation anchor is stale")
    normalized_anchor = _anchor_with_existing_line_ending(source_text, anchor)
    if kind == "insert_after":
        if _text_insert_uses_partial_line_anchor(safe_path, normalized_anchor):
            raise ValueError(
                "text insertion anchor must include a complete line ending"
            )
        if _python_insert_ends_inside_open_expression(
            safe_path,
            normalized_anchor,
        ):
            raise ValueError("Python insertion anchor ends inside an expression")
        normalized_anchor = _text_anchor_extended_to_block_end(
            safe_path=safe_path,
            current=source_text,
            anchor=normalized_anchor,
        )
        insertion = _insertion_text(
            content,
            after_anchor=normalized_anchor,
            before_tail=source_text,
        )
        materialized = source_text.replace(
            normalized_anchor,
            normalized_anchor + insertion,
            1,
        )
        return materialized
    if kind == "insert_before":
        insertion = _insertion_text(
            content,
            after_anchor="",
            before_tail=normalized_anchor,
        )
        materialized = source_text.replace(
            normalized_anchor,
            insertion + normalized_anchor,
            1,
        )
        return materialized
    if kind == "replace":
        replacement = _replacement_text(
            content,
            replaced_anchor=normalized_anchor,
        )
        materialized = source_text.replace(normalized_anchor, replacement, 1)
        return materialized
    raise ValueError("canonical text operation is unsupported")


def canonical_proposal_digest(
    records: list[CanonicalPatchOperationRecord],
) -> str:
    """Return the canonical ordered-operation digest for approval binding."""

    serialized = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest


def validate_canonical_operation_binding(
    *,
    records: list[CanonicalPatchOperationRecord],
    proposal_digest: str,
    candidate_revision: int,
) -> str:
    """Validate the reviewed operation sequence before candidate application."""

    if canonical_proposal_digest(records) != proposal_digest:
        return "Canonical operation proposal digest mismatch."
    if not records:
        return "Canonical operation records are empty."
    expected_revisions = [
        record["expected_candidate_revision"]
        for record in records
    ]
    first_revision = expected_revisions[0]
    required_revisions = list(range(first_revision, first_revision + len(records)))
    if expected_revisions != required_revisions:
        return "Canonical operation candidate revision sequence mismatch."
    if candidate_revision != first_revision + len(records):
        return "Canonical operation final candidate revision mismatch."
    return ""


def compile_patch_operations(
    *,
    repo_root: Path | None,
    patch_operations: list[PatchOperation],
    max_files: int,
    max_diff_chars: int,
) -> tuple[
    list[PatchArtifact],
    list[CreatedFileSummary],
    list[ChangedFileSummary],
    list[str],
]:
    """Compile structured edit operations into public patch artifacts."""

    if not patch_operations:
        return [], [], [], []

    originals: dict[str, str] = {}
    modified: dict[str, str] = {}
    existing_paths: set[str] = set()
    candidate_created_paths: set[str] = set()
    candidate_tombstones: set[str] = set()
    touched_paths: set[str] = set()
    created_files: list[CreatedFileSummary] = []
    changed_files: list[ChangedFileSummary] = []
    special_artifacts: list[PatchArtifact] = []
    errors: list[str] = []

    for operation in patch_operations:
        safe_path = _safe_repo_relative_path(operation.get("path", ""))
        if safe_path is None:
            errors.append("Patch operation includes an unsafe path.")
            continue
        operation_paths = {safe_path}
        if operation.get("kind") == "rename_file":
            target_path = _safe_repo_relative_path(
                operation.get("target_path", ""),
            )
            if target_path is not None:
                operation_paths.add(target_path)
        if len(touched_paths | operation_paths) > max_files:
            errors.append("Patch operations touch too many files.")
            continue
        touched_paths.update(operation_paths)

        kind = operation.get("kind")
        if kind == "delete_file":
            if safe_path in candidate_created_paths:
                if safe_path not in modified:
                    errors.append("Delete-file operation targets a missing file.")
                    continue
                modified.pop(safe_path)
                candidate_created_paths.remove(safe_path)
                candidate_tombstones.add(safe_path)
                created_files = [
                    row for row in created_files if row["path"] != safe_path
                ]
                changed_files = [
                    row for row in changed_files if row["path"] != safe_path
                ]
                continue
            if safe_path in candidate_tombstones:
                errors.append("Delete-file operation targets a missing file.")
                continue
            if safe_path in modified:
                original = originals[safe_path]
                exists = True
                modified.pop(safe_path)
            else:
                original, exists = _read_original(
                    repo_root=repo_root,
                    safe_path=safe_path,
                )
            if not exists:
                errors.append("Delete-file operation targets a missing file.")
                continue
            candidate_tombstones.add(safe_path)
            changed_files = [
                row for row in changed_files if row["path"] != safe_path
            ]
            special_artifacts.append(_delete_artifact(safe_path, original))
            changed_files.append(_changed_file(safe_path, "delete", operation))
            continue

        if kind == "rename_file":
            target_value = operation.get("target_path", "")
            target_path = _safe_repo_relative_path(target_value)
            if target_path is None or target_path == safe_path:
                errors.append("Rename-file operation has an invalid target path.")
                continue
            _, baseline_target_exists = _read_original(
                repo_root=repo_root,
                safe_path=target_path,
            )
            target_exists = (
                target_path in modified
                or (
                    target_path not in candidate_tombstones
                    and baseline_target_exists
                )
            )
            if safe_path in candidate_tombstones or target_exists:
                errors.append(
                    "Rename-file operation has an invalid source or target.",
                )
                continue
            source_was_created = safe_path in candidate_created_paths
            if safe_path in modified:
                content = modified.pop(safe_path)
                baseline_content = originals.get(safe_path, "")
                exists = True
            else:
                content, exists = _read_original(
                    repo_root=repo_root,
                    safe_path=safe_path,
                )
                baseline_content = content
            if not exists:
                errors.append(
                    "Rename-file operation has an invalid source or target.",
                )
                continue
            if source_was_created:
                if safe_path not in candidate_created_paths:
                    errors.append(
                        "Rename-file operation has an invalid source or target.",
                    )
                    continue
                candidate_created_paths.remove(safe_path)
                candidate_created_paths.add(target_path)
                candidate_tombstones.add(safe_path)
                candidate_tombstones.discard(target_path)
                originals.setdefault(target_path, "")
                modified[target_path] = content
                for row in created_files:
                    if row["path"] == safe_path:
                        row["path"] = target_path
                changed_files = [
                    row for row in changed_files if row["path"] != safe_path
                ]
                changed_files.append(
                    _changed_file(target_path, "add", operation),
                )
                continue
            special_artifacts.append(_rename_artifact(safe_path, target_path))
            originals.pop(safe_path, None)
            originals[target_path] = baseline_content
            modified[target_path] = content
            candidate_tombstones.add(safe_path)
            candidate_tombstones.discard(target_path)
            changed_files.append(_changed_file(safe_path, "rename_from", operation))
            changed_files.append({
                "path": target_path,
                "change_type": "rename_to",
                "summary": operation.get("summary", "") or "Renamed file.",
            })
            continue

        content = operation.get("content", "")
        if not isinstance(content, str) or not content:
            errors.append("Patch operation omitted replacement content.")
            continue

        current = modified.get(safe_path)
        if current is None and safe_path not in candidate_tombstones:
            original, exists = _read_original(repo_root=repo_root, safe_path=safe_path)
            originals[safe_path] = original
            if exists:
                existing_paths.add(safe_path)
            current = original

        if kind == "create_file":
            if safe_path in modified or safe_path in existing_paths:
                errors.append("Create-file operation targets an existing file.")
                continue
            candidate_tombstones.discard(safe_path)
            originals[safe_path] = ""
            modified[safe_path] = _with_trailing_newline(content)
            candidate_created_paths.add(safe_path)
            created_files.append({
                "path": safe_path,
                "role": operation.get("summary", "") or "created file",
            })
            changed_files.append(_changed_file(safe_path, "add", operation))
            continue

        if current is None:
            errors.append("Existing-file operation targets a missing file.")
            continue

        if repo_root is None and safe_path not in candidate_created_paths:
            errors.append("Existing-file operation requires repository context.")
            continue

        if kind == "replace_file_small":
            if len(current) > REPLACE_FILE_SMALL_MAX_CHARS:
                errors.append("full-file replacement target exceeds the cap.")
                continue
            if len(content) > REPLACE_FILE_SMALL_MAX_CHARS:
                errors.append("full-file replacement content exceeds the cap.")
                continue
            modified[safe_path] = materialize_text_operation(
                safe_path=safe_path,
                kind=kind,
                source_text=current,
                anchor=None,
                content=content,
            )
            changed_files.append(_changed_file(safe_path, "modify", operation))
            continue

        anchor_error = _exact_anchor_error(
            current=current,
            operation=operation,
            safe_path=safe_path,
        )
        if anchor_error:
            errors.append(anchor_error)
            continue
        anchor = operation["anchor"]
        try:
            modified[safe_path] = materialize_text_operation(
                safe_path=safe_path,
                kind=str(kind),
                source_text=current,
                anchor=anchor,
                content=content,
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if kind in {"insert_after", "insert_before", "replace"}:
            changed_files.append(_changed_file(safe_path, "modify", operation))
            continue

        errors.append("Patch operation kind is unsupported.")

    if errors:
        return [], [], [], _dedupe_strings(errors)

    artifacts: list[PatchArtifact] = list(special_artifacts)
    total_chars = sum(len(artifact["diff_text"]) for artifact in artifacts)
    for safe_path, new_text in modified.items():
        old_text = originals.get(safe_path, "")
        if old_text == new_text:
            continue
        if _markdown_fences_are_unbalanced(safe_path, new_text):
            errors.append(
                "Markdown code fences are unbalanced after patch operations."
            )
            continue
        diff_text = _diff_for_file(safe_path, old_text, new_text)
        total_chars += len(diff_text)
        if total_chars > max_diff_chars:
            errors.append("Compiled patch operations exceed the diff limit.")
            break
        artifacts.append({
            "artifact_id": f"compiled-{len(artifacts) + 1}",
            "base": "repository" if old_text else "new_file",
            "diff_text": diff_text,
            "files": [safe_path],
            "summary": "Compiled from structured patch operations.",
        })

    if errors:
        return [], [], [], _dedupe_strings(errors)

    return artifacts, created_files, _dedupe_changed_files(changed_files), []


def _read_original(*, repo_root: Path | None, safe_path: str) -> tuple[str, bool]:
    if repo_root is None:
        return "", False
    root = repo_root.expanduser().resolve(strict=True)
    file_path = confined_managed_repo_path(root, safe_path)
    if not file_path.exists():
        return "", False
    if not file_path.is_file():
        return "", False
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return text, True


def _exact_anchor_error(
    *,
    current: str,
    operation: PatchOperation,
    safe_path: str,
) -> str:
    anchor = operation.get("anchor", "")
    if not isinstance(anchor, str) or not anchor:
        return "Existing-file operation omitted an exact anchor."
    match_count = current.count(anchor)
    if match_count == 1:
        return ""
    if match_count == 0:
        return f"{_operation_label(operation, safe_path)} anchor was not found."
    return (
        f"{_operation_label(operation, safe_path)} anchor matched multiple "
        "locations."
    )


def _text_insert_splits_paragraph(
    *,
    safe_path: str,
    current: str,
    anchor: str,
) -> bool:
    """Detect document insertions that cut through a wrapped paragraph."""

    if not _is_text_document_path(safe_path):
        return False

    anchor_index = current.find(anchor)
    if anchor_index < 0:
        return False

    tail = current[anchor_index + len(anchor):]
    if not tail or tail.startswith("\n") or tail.startswith("\r\n"):
        return False

    anchor_line = anchor.rstrip("\r\n").splitlines()
    if not anchor_line:
        return False
    stripped_line = anchor_line[-1].strip()
    if not stripped_line:
        return False
    if STRUCTURAL_TEXT_LINE_RE.match(stripped_line) is not None:
        return False

    return True


def _text_anchor_extended_to_block_end(
    *,
    safe_path: str,
    current: str,
    anchor: str,
) -> str:
    if not _text_insert_splits_paragraph(
        safe_path=safe_path,
        current=current,
        anchor=anchor,
    ):
        return anchor

    anchor_index = current.find(anchor)
    if anchor_index < 0:
        return anchor

    tail_start = anchor_index + len(anchor)
    tail = current[tail_start:]
    extension_length = 0
    for line in tail.splitlines(keepends=True):
        if not line.strip():
            break
        extension_length += len(line)
    extended_anchor = current[anchor_index:tail_start + extension_length]
    return extended_anchor


def _anchor_with_existing_line_ending(current: str, anchor: str) -> str:
    if anchor.endswith(("\n", "\r\n")):
        return anchor

    anchor_index = current.find(anchor)
    if anchor_index < 0:
        return anchor

    tail = current[anchor_index + len(anchor):]
    if tail.startswith("\r\n"):
        return anchor + "\r\n"
    if tail.startswith("\n"):
        return anchor + "\n"
    return anchor


def _text_insert_uses_partial_line_anchor(safe_path: str, anchor: str) -> bool:
    if not _is_text_document_path(safe_path):
        return False
    return not anchor.endswith(("\n", "\r\n"))


def _python_insert_ends_inside_open_expression(
    safe_path: str,
    anchor: str,
) -> bool:
    if not _is_python_path(safe_path):
        return False
    stripped_anchor = anchor.rstrip()
    if not stripped_anchor:
        return False
    if stripped_anchor.endswith(("\\", ",")):
        return True

    balance = 0
    for char in stripped_anchor:
        if char in "([{":
            balance += 1
            continue
        if char in ")]}":
            balance -= 1
    return balance > 0


def _is_text_document_path(safe_path: str) -> bool:
    suffix = Path(safe_path).suffix.casefold()
    return suffix in TEXT_DOCUMENT_SUFFIXES


def _markdown_fences_are_unbalanced(safe_path: str, text: str) -> bool:
    suffix = Path(safe_path).suffix.casefold()
    if suffix not in (".md", ".markdown"):
        return False

    backtick_count = 0
    tilde_count = 0
    for line in text.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith("```"):
            backtick_count += 1
            continue
        if stripped_line.startswith("~~~"):
            tilde_count += 1
    return backtick_count % 2 != 0 or tilde_count % 2 != 0


def _is_python_path(safe_path: str) -> bool:
    suffix = Path(safe_path).suffix.casefold()
    return suffix in PYTHON_SUFFIXES


def _insertion_text(
    content: str,
    *,
    after_anchor: str,
    before_tail: str,
) -> str:
    insertion = content
    if (
        after_anchor
        and not after_anchor.endswith("\n")
        and not insertion.startswith("\n")
    ):
        insertion = "\n" + insertion
    if before_tail and not insertion.endswith("\n"):
        insertion = insertion + "\n"
    return insertion


def _replacement_text(content: str, *, replaced_anchor: str) -> str:
    replacement = content
    if replaced_anchor.endswith("\r\n") and not replacement.endswith(("\r\n", "\n")):
        return replacement + "\r\n"
    if replaced_anchor.endswith("\n") and not replacement.endswith(("\r\n", "\n")):
        return replacement + "\n"
    return replacement


def _with_trailing_newline(content: str) -> str:
    if content.endswith("\n"):
        return content
    return content + "\n"


def _diff_for_file(safe_path: str, old_text: str, new_text: str) -> str:
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    if old_text:
        from_file = f"a/{safe_path}"
        header = f"diff --git a/{safe_path} b/{safe_path}\n"
    else:
        from_file = "/dev/null"
        header = (
            f"diff --git a/{safe_path} b/{safe_path}\n"
            "new file mode 100644\n"
        )
    diff_lines = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=from_file,
        tofile=f"b/{safe_path}",
        lineterm="",
    )
    diff_body = "".join(_diff_line(line) for line in diff_lines)
    diff_text = header + diff_body
    return diff_text


def _delete_artifact(safe_path: str, old_text: str) -> PatchArtifact:
    """Render one deleted file as a standard unified-diff artifact."""

    diff_text = _diff_for_file(safe_path, old_text, "")
    artifact: PatchArtifact = {
        "artifact_id": f"delete-{safe_path}",
        "base": "repository",
        "diff_text": diff_text.replace(f"+++ b/{safe_path}", "+++ /dev/null"),
        "files": [safe_path],
        "summary": "Delete file.",
    }
    return artifact


def _rename_artifact(source_path: str, target_path: str) -> PatchArtifact:
    """Render content-preserving rename metadata for review and apply."""

    diff_text = (
        f"diff --git a/{source_path} b/{target_path}\n"
        "similarity index 100%\n"
        f"rename from {source_path}\n"
        f"rename to {target_path}\n"
    )
    artifact: PatchArtifact = {
        "artifact_id": f"rename-{source_path}-to-{target_path}",
        "base": "repository",
        "diff_text": diff_text,
        "files": [source_path, target_path],
        "summary": "Rename file.",
    }
    return artifact


def _diff_line(line: str) -> str:
    if line.endswith("\n"):
        return line
    return line + "\n"


def _operation_label(operation: PatchOperation, safe_path: str) -> str:
    operation_id = operation.get("operation_id", "")
    if isinstance(operation_id, str) and operation_id:
        return f"Patch operation {operation_id} for {safe_path}"
    return f"Patch operation for {safe_path}"


def _changed_file(
    safe_path: str,
    change_type: str,
    operation: PatchOperation,
) -> ChangedFileSummary:
    summary: ChangedFileSummary = {
        "path": safe_path,
        "change_type": change_type,
        "summary": operation.get("summary", "") or "Structured edit operation.",
    }
    return summary


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
