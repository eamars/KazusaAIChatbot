"""Phase 5 public interface and background metadata contracts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def test_patch_apply_public_api_is_exported() -> None:
    """The top-level coding-agent package exposes the trusted apply boundary."""

    from kazusa_ai_chatbot.coding_agent import (
        CodingPatchApplyRequest,
        CodingPatchApplyResponse,
        apply_approved_patch,
    )

    assert isinstance(apply_approved_patch, Callable)
    assert "patch_artifacts" in CodingPatchApplyRequest.__annotations__
    assert "apply_workspace_ref" in CodingPatchApplyResponse.__annotations__


def test_background_metadata_preserves_code_modifying() -> None:
    """Background worker metadata must preserve the modifying operation."""

    from kazusa_ai_chatbot.background_work.subagent.coding_agent import (
        _map_coding_agent_response,
    )

    result = _map_coding_agent_response(
        {
            "status": "succeeded",
            "operation": "code_modifying",
            "answer_text": "Proposed a repository patch.",
            "evidence": [],
            "limitations": [],
            "repository": {
                "owner": "fixture",
                "repo": "demo",
            },
            "changed_files": [{
                "path": "app.py",
                "change_type": "modify",
                "summary": "Update app.",
            }],
            "trace_summary": ["modifying:succeeded"],
        },
        max_output_chars=1000,
    )

    assert result["worker_metadata"]["coding_operation"] == "code_modifying"
    assert "code_modifying" in result["result_summary"]
    assert result["worker_metadata"]["changed_files"] == [{
        "path": "app.py",
        "change_type": "modify",
        "summary": "Update app.",
    }]


def test_modifying_programmer_prompt_requires_requested_tests_docs() -> None:
    """Prompt contract keeps requested test/doc edits inside artifacts."""

    from kazusa_ai_chatbot.coding_agent.code_modifying.programmer import (
        MODIFYING_PROGRAMMER_PROMPT,
    )

    prompt = MODIFYING_PROGRAMMER_PROMPT.casefold()

    assert "return succeeded artifacts" in prompt
    assert "matching test or document" in prompt
    assert "repair_feedback" in prompt
    assert "validation error" in prompt
    assert "artifact for every target path" in prompt
    assert "source-only artifacts" in prompt
    assert "files are present in file_contexts" in prompt
    assert "instead of reporting it as a limitation" in prompt
    assert "existing test file shows a local mocking pattern" in prompt
    assert "do not block only because the exact new" in prompt
    assert "same as the current file text" in prompt
    assert "update a provided cli test file" in prompt
    assert "preserve existing ownership boundaries" in prompt
    assert "ownership_guidance.source_owner_paths" in prompt
    assert "modify that owner file" in prompt
    assert "durable entity state" in prompt
    assert "model owner file" in prompt
    assert "valid python" in prompt
    assert "prefer triple-quoted" in prompt
    assert "double escaped in the json text" in prompt
    assert "never use english pseudo-code" in prompt
    assert "introduces a new module reference" in prompt
    assert "required import" in prompt
    assert "local definition" in prompt
    assert "escaped \"\\n\" sequences" in prompt
    assert "do not duplicate an" in prompt
    assert "existing import block" in prompt
    assert "module top level" in prompt
    assert "never insert a python import after" in prompt
    assert "prefer replace_file_small" in prompt
    assert "never include an indented import line" in prompt
    assert "preserve method receivers" in prompt
    assert "self" in prompt
    assert "cls" in prompt


def test_modifying_payload_projects_owner_path_guidance() -> None:
    """Source owner and test/doc candidates are explicit in modifying payload."""

    from kazusa_ai_chatbot.coding_agent.code_modifying.supervisor import (
        _ownership_guidance,
    )

    guidance = _ownership_guidance([
        {"path": "mdlinkcheck/scanner.py"},
        {"path": "mdlinkcheck/anchors.py"},
        {"path": "tests/test_anchors.py"},
        {"path": "README.md"},
    ])

    assert guidance["source_owner_paths"] == [
        "mdlinkcheck/scanner.py",
        "mdlinkcheck/anchors.py",
    ]
    assert guidance["test_or_doc_paths"] == [
        "tests/test_anchors.py",
        "README.md",
    ]


def test_source_backed_write_uses_bounded_fallback_evidence(
    tmp_path: Path,
) -> None:
    """Concrete patch requests keep moving when reading PM returns no evidence."""

    from kazusa_ai_chatbot.coding_agent.supervisor import (
        _fallback_reading_result_for_write,
    )

    repo_root = tmp_path / "repo"
    package_root = repo_root / "contacts_jsonl_to_csv"
    tests_root = repo_root / "tests"
    package_root.mkdir(parents=True)
    tests_root.mkdir()
    (package_root / "converter.py").write_text("def convert():\n    pass\n")
    (package_root / "cli.py").write_text("def main():\n    pass\n")
    (tests_root / "test_converter.py").write_text("def test_convert():\n    pass\n")
    (tests_root / "test_cli.py").write_text("def test_main():\n    pass\n")
    (repo_root / "README.md").write_text("# Contacts\n")

    result = _fallback_reading_result_for_write(
        request={
            "question": "Update converter, CLI, tests, and README.",
            "workspace_root": str(tmp_path / "workspace"),
        },
        repository={
            "provider": "github",
            "owner": "fixture",
            "repo": "demo",
            "source_url": "local://fixture/demo",
            "requested_ref": None,
            "resolved_ref": "local",
            "current_commit": "abc123",
            "default_branch": "main",
            "local_root": str(repo_root),
            "storage_kind": "existing_local_checkout",
            "managed_checkout": False,
            "workspace_root": str(tmp_path / "workspace"),
            "cache_key": None,
            "dirty_state": "clean",
        },
        source_scope={
            "kind": "repository",
            "repo_relative_path": None,
            "source_url": "local://fixture/demo",
            "requested_ref": None,
            "interpretation": "Local fixture.",
        },
        prior_reading_result={
            "status": "needs_user_input",
            "answer_text": "",
            "evidence": [],
            "limitations": ["The PM could not identify enough bounded evidence slots."],
            "trace_summary": ["reading_pm:sufficiency=needs_user_input"],
        },
    )

    evidence_paths = [row["path"] for row in result["evidence"]]

    assert result["status"] == "succeeded"
    assert "contacts_jsonl_to_csv/converter.py" in evidence_paths
    assert "contacts_jsonl_to_csv/cli.py" in evidence_paths
    assert "tests/test_converter.py" in evidence_paths
    assert "tests/test_cli.py" in evidence_paths
    assert "README.md" in evidence_paths
    assert result["trace_summary"] == ["reading_fallback:evidence=5"]


def test_failed_patch_validation_triggers_modifying_repair() -> None:
    """Failed proposal validation is eligible for one modifying repair retry."""

    from kazusa_ai_chatbot.coding_agent.supervisor import (
        _write_response_needs_modifying_repair,
    )

    response = {
        "status": "failed",
        "validation": {
            "status": "failed",
            "errors": ["Patch uses Python module references without imports."],
            "warnings": [],
        },
    }

    assert _write_response_needs_modifying_repair(response) is True


def test_successful_patch_validation_skips_modifying_repair() -> None:
    """Successful proposal validation is not retried."""

    from kazusa_ai_chatbot.coding_agent.supervisor import (
        _write_response_needs_modifying_repair,
    )

    response = {
        "status": "succeeded",
        "validation": {
            "status": "succeeded",
            "errors": [],
            "warnings": [],
        },
    }

    assert _write_response_needs_modifying_repair(response) is False
