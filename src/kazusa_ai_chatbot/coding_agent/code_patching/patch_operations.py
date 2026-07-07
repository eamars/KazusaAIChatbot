"""Compile LLM-selected edit operations into unified diff artifacts."""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_patching.models import (
    ChangedFileSummary,
    CreatedFileSummary,
    PatchArtifact,
    PatchOperation,
)
from kazusa_ai_chatbot.coding_agent.code_patching.patch_validation import (
    _safe_repo_relative_path,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import ensure_path_inside

REPLACE_FILE_SMALL_MAX_CHARS = 20000
TEXT_DOCUMENT_SUFFIXES = {".md", ".markdown", ".rst", ".txt"}
PYTHON_SUFFIXES = {".py", ".pyi"}
STRUCTURAL_TEXT_LINE_RE = re.compile(
    r"^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s|```|~~~|\|)"
)


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
    created_files: list[CreatedFileSummary] = []
    changed_files: list[ChangedFileSummary] = []
    errors: list[str] = []

    for operation in patch_operations:
        safe_path = _safe_repo_relative_path(operation.get("path", ""))
        if safe_path is None:
            errors.append("Patch operation includes an unsafe path.")
            continue
        if len(modified) >= max_files and safe_path not in modified:
            errors.append("Patch operations touch too many files.")
            continue

        content = operation.get("content", "")
        if not isinstance(content, str) or not content:
            errors.append("Patch operation omitted replacement content.")
            continue

        kind = operation.get("kind")
        current = modified.get(safe_path)
        if current is None:
            original, exists = _read_original(repo_root=repo_root, safe_path=safe_path)
            originals[safe_path] = original
            if exists:
                existing_paths.add(safe_path)
            current = original

        if kind == "create_file":
            if safe_path in existing_paths:
                errors.append("Create-file operation targets an existing file.")
                continue
            modified[safe_path] = _with_trailing_newline(content)
            created_files.append({
                "path": safe_path,
                "role": operation.get("summary", "") or "created file",
            })
            changed_files.append(_changed_file(safe_path, "add", operation))
            continue

        if repo_root is None:
            errors.append("Existing-file operation requires repository context.")
            continue

        if kind == "replace_file_small":
            if repo_root is None:
                errors.append("full-file replacement requires repository context.")
                continue
            if len(current) > REPLACE_FILE_SMALL_MAX_CHARS:
                errors.append("full-file replacement target exceeds the cap.")
                continue
            if len(content) > REPLACE_FILE_SMALL_MAX_CHARS:
                errors.append("full-file replacement content exceeds the cap.")
                continue
            modified[safe_path] = _with_trailing_newline(content)
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
        anchor = _anchor_with_existing_line_ending(current, anchor)

        if kind == "insert_after":
            if _text_insert_uses_partial_line_anchor(safe_path, anchor):
                errors.append(
                    "Text insertion anchor must include a complete line ending."
                )
                continue
            if _python_insert_ends_inside_open_expression(safe_path, anchor):
                errors.append(
                    "Python insertion anchor ends inside an open expression."
                )
                continue
            anchor = _text_anchor_extended_to_block_end(
                safe_path=safe_path,
                current=current,
                anchor=anchor,
            )
            updated = current.replace(
                anchor,
                anchor + _insertion_text(content, after_anchor=anchor, before_tail=current),
                1,
            )
            modified[safe_path] = updated
            changed_files.append(_changed_file(safe_path, "modify", operation))
            continue

        if kind == "insert_before":
            updated = current.replace(
                anchor,
                _insertion_text(content, after_anchor="", before_tail=anchor) + anchor,
                1,
            )
            modified[safe_path] = updated
            changed_files.append(_changed_file(safe_path, "modify", operation))
            continue

        if kind == "replace":
            modified[safe_path] = current.replace(
                anchor,
                _replacement_text(content, replaced_anchor=anchor),
                1,
            )
            changed_files.append(_changed_file(safe_path, "modify", operation))
            continue

        errors.append("Patch operation kind is unsupported.")

    if errors:
        return [], [], [], _dedupe_strings(errors)

    artifacts: list[PatchArtifact] = []
    total_chars = 0
    for safe_path, new_text in modified.items():
        old_text = originals[safe_path]
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
    file_path = ensure_path_inside(root / safe_path, root)
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
    return f"{_operation_label(operation, safe_path)} anchor matched multiple locations."


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
    if after_anchor and not after_anchor.endswith("\n") and not insertion.startswith("\n"):
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
