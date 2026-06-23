import asyncio
import subprocess
from pathlib import Path
from typing import Any


def _run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = result.stdout.strip()
    return output


def _make_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "module.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    _run_git(["init"], repo_root)
    _run_git(["config", "user.email", "test@example.com"], repo_root)
    _run_git(["config", "user.name", "Test User"], repo_root)
    _run_git(["add", "."], repo_root)
    _run_git(["commit", "-m", "initial"], repo_root)
    return repo_root


def _new_file_diff(relative_path: str) -> str:
    diff_text = "\n".join(
        [
            f"diff --git a/{relative_path} b/{relative_path}",
            "new file mode 100644",
            "index 0000000..b4d9cf5",
            "--- /dev/null",
            f"+++ b/{relative_path}",
            "@@ -0,0 +1,2 @@",
            "+def generated_value() -> int:",
            "+    return 42",
            "",
        ]
    )
    return diff_text


def _artifact(diff_text: str) -> dict[str, Any]:
    artifact = {
        "artifact_id": "main",
        "base": "repository",
        "diff_text": diff_text,
        "files": ["src/generated.py"],
        "summary": "Adds one generated helper.",
    }
    return artifact


def test_contracts_define_code_writing_shapes() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.models import (
        CodeWritingRequest,
        CodeWritingResult,
        PatchArtifact,
        PatchOperation,
        PatchValidationSummary,
        SourceOwnerCandidate,
        WritingFileDemand,
        WritingFileResolution,
        WritingPMContractEvaluation,
        WritingPMDecision,
        WritingPMInput,
        WritingPatcherInput,
        WritingPatcherReport,
        WritingProgrammerAssignment,
        WritingProgrammerReport,
    )
    from kazusa_ai_chatbot.coding_agent.models import (
        CodingAgentWriteRequest,
        CodingPatchProposalResponse,
    )

    assert {
        "question",
        "workspace_root",
        "session_id",
        "max_artifact_chars",
    } <= set(CodingAgentWriteRequest.__annotations__)
    assert {
        "status",
        "mode",
        "answer_text",
        "patch_artifacts",
        "validation",
        "trace_summary",
    } <= set(CodingPatchProposalResponse.__annotations__)
    assert {
        "question",
        "mode_hint",
        "repository",
        "source_scope",
        "reading_result",
        "supervisor_evidence_state",
        "workspace_root",
        "external_evidence",
    } <= set(CodeWritingRequest.__annotations__)
    assert {
        "status",
        "mode",
        "patch_artifacts",
        "validation",
        "reading_requests",
        "external_evidence_requests",
        "trace_summary",
    } <= set(CodeWritingResult.__annotations__)
    assert {
        "question",
        "mode",
        "repository_summary",
        "reading_reports",
        "supervisor_evidence_state",
        "owner_candidates",
        "previous_writing_reports",
        "file_resolution_feedback",
        "pm_contract_feedback",
    } <= set(WritingPMInput.__annotations__)
    assert {
        "status",
        "errors",
        "repair_feedback",
    } <= set(WritingPMContractEvaluation.__annotations__)
    assert {
        "path",
        "role",
        "line_start",
        "line_end",
        "symbols",
        "exception_types",
        "feature_markers",
        "reasons",
        "evidence_refs",
    } <= set(SourceOwnerCandidate.__annotations__)
    assert {
        "demand_id",
        "role",
        "purpose",
        "file_kind",
        "preferred_path",
        "preferred_name",
        "placement_hint",
        "related_paths",
        "read_only_paths",
        "interface_contract",
    } <= set(WritingFileDemand.__annotations__)
    assert {
        "status",
        "assignments",
        "errors",
        "repair_feedback",
    } <= set(WritingFileResolution.__annotations__)
    assert {
        "status",
        "mode",
        "intent",
        "file_demands",
        "assignments",
        "missing_slots",
        "reading_requests",
        "external_evidence_requests",
    } <= set(WritingPMDecision.__annotations__)
    assert {
        "assignment_id",
        "role",
        "scope",
        "questions",
        "required_slots",
    } <= set(WritingProgrammerAssignment.__annotations__)
    assert {
        "assignment_id",
        "status",
        "files_considered",
        "facts",
        "patch_operations",
        "open_questions",
    } <= set(WritingProgrammerReport.__annotations__)
    assert {
        "question",
        "mode",
        "base_identity",
        "owned_path_map",
        "selected_programmer_reports",
        "artifact_limits",
    } <= set(WritingPatcherInput.__annotations__)
    assert {
        "status",
        "patch_artifacts",
        "created_files",
        "changed_files",
        "edit_diagnostics",
        "unmaterialized_reports",
    } <= set(WritingPatcherReport.__annotations__)
    assert {
        "artifact_id",
        "base",
        "diff_text",
        "files",
        "summary",
    } <= set(PatchArtifact.__annotations__)
    assert {
        "kind",
        "path",
        "content",
    } <= set(PatchOperation.__annotations__)
    assert {
        "status",
        "parsed",
        "sandbox_applied",
        "errors",
    } <= set(PatchValidationSummary.__annotations__)


def test_programmer_report_preserves_patch_operation_indentation() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
        normalize_programmer_report,
    )

    assignment = {
        "assignment_id": "module_writer",
        "role": "Module writer",
        "scope": {"kind": "file", "values": ["module.py"]},
        "owned_paths": ["module.py"],
        "read_only_paths": [],
        "interface_contract": {
            "component": "module",
            "inputs": [],
            "outputs": [],
            "callers": [],
            "invariants": [],
        },
        "must_not_touch": [],
        "questions": ["Add a class method."],
        "required_slots": ["runtime_method"],
    }
    parsed = {
        "status": "succeeded",
        "patch_operations": [
            {
                "operation_id": "insert-method",
                "kind": "insert_after",
                "path": "module.py",
                "anchor": "        return 1\n",
                "content": "\n    def next_value(self) -> int:\n"
                "        return 2\n",
                "summary": "Adds a method.",
            }
        ],
    }

    report = normalize_programmer_report(
        parsed,
        assignment=assignment,
        reading_evidence=[],
    )

    assert report["patch_operations"][0]["anchor"].startswith("        ")
    assert report["patch_operations"][0]["content"].startswith("\n    def")
    assert "\n        return 2" in report["patch_operations"][0]["content"]


def test_patcher_rejects_out_of_scope_programmer_operation() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patcher import (
        materialize_patch_artifacts,
    )

    report = {
        "assignment_id": "runtime",
        "status": "succeeded",
        "files_considered": ["src/allowed.py"],
        "facts": [],
        "patch_operations": [{
            "operation_id": "create-outside",
            "kind": "create_file",
            "path": "src/outside.py",
            "content": "VALUE = 1\n",
            "summary": "Creates an unassigned file.",
        }],
        "open_questions": [],
        "created_files": [],
        "changed_files": [],
        "evidence": [],
    }
    trace: dict[str, object] = {}

    patcher_report = materialize_patch_artifacts(
        repo_root=None,
        patcher_input={
            "question": "Create one limited patch.",
            "mode": "create_new_project",
            "base_identity": "empty-workspace:test",
            "owned_path_map": {"src/allowed.py": "runtime"},
            "base_file_summaries": [],
            "selected_programmer_reports": [report],
            "pm_integration_notes": ["Use only assigned paths."],
            "artifact_limits": {
                "max_files": 4,
                "max_diff_chars": 4000,
            },
        },
        max_files=4,
        max_diff_chars=4000,
        trace=trace,
    )

    assert patcher_report["status"] == "blocked"
    assert patcher_report["patch_artifacts"] == []
    assert patcher_report["unmaterialized_reports"] == ["runtime"]
    assert any(
        "out-of-scope path src/outside.py" in diagnostic
        for diagnostic in patcher_report["edit_diagnostics"]
    )
    assert trace["route_name"] == "CODING_AGENT_PROGRAMMER_LLM"


def test_source_owner_candidates_prioritize_runtime_error_owner(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.source_owners import (
        collect_source_owner_candidates,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    (repo_root / "docs").mkdir()
    (repo_root / "src" / "runtime.py").write_text(
        "def handle(value: str) -> str:\n"
        "    if not value:\n"
        "        raise ValueError(\"value is required\")\n"
        "    return value\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_runtime.py").write_text(
        "import pytest\n"
        "\n"
        "from src.runtime import handle\n"
        "\n"
        "def test_handle_empty() -> None:\n"
        "    with pytest.raises(ValueError, match=\"required\"):\n"
        "        handle(\"\")\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "usage.md").write_text(
        "Runtime behavior notes.\n",
        encoding="utf-8",
    )
    reading_evidence = [
        {
            "path": "docs/usage.md",
            "line_start": 1,
            "line_end": 1,
            "symbol_or_topic": "usage",
            "excerpt": "Runtime behavior notes.",
            "reason": "Supporting documentation.",
        },
        {
            "path": "tests/test_runtime.py",
            "line_start": 1,
            "line_end": 7,
            "symbol_or_topic": "test behavior",
            "excerpt": "with pytest.raises(ValueError, match=\"required\"):",
            "reason": "Shows current test style.",
        },
        {
            "path": "src/runtime.py",
            "line_start": 1,
            "line_end": 4,
            "symbol_or_topic": "runtime behavior",
            "excerpt": "raise ValueError(\"value is required\")",
            "reason": "Shows current runtime branch.",
        },
    ]

    candidates = collect_source_owner_candidates(
        repo_root=repo_root,
        reading_evidence=reading_evidence,
        max_candidates=4,
    )

    assert candidates[0]["path"] == "src/runtime.py"
    assert candidates[0]["role"] == "runtime"
    assert "raises_errors" in candidates[0]["feature_markers"]
    assert "ValueError" in candidates[0]["exception_types"]
    assert "handle" in candidates[0]["symbols"]
    assert candidates[1]["path"] == "tests/test_runtime.py"
    assert candidates[1]["role"] == "test"
    assert "pytest_raises" in candidates[1]["feature_markers"]
    assert "ValueError" in candidates[1]["exception_types"]
    assert candidates[2]["path"] == "docs/usage.md"
    assert candidates[2]["role"] == "docs"


async def test_external_evidence_adapter_uses_web_helper(
    monkeypatch,
) -> None:
    from kazusa_ai_chatbot.coding_agent import external_evidence

    calls: list[dict[str, object]] = []

    class FakeWebAgent:
        async def run(self, *, task: str, context: dict[str, str]):
            calls.append({"task": task, "context": context})
            result = {
                "resolved": True,
                "result": "official reference summary",
            }
            return result

    monkeypatch.setattr(external_evidence, "WebAgent3", FakeWebAgent)
    trace_summary: list[str] = []

    evidence = await external_evidence.collect_external_evidence(
        [{
            "request_id": "docs",
            "task": "Find the public contract for a current API.",
            "reason": "The patch needs current public evidence.",
        }],
        trace_summary=trace_summary,
    )

    assert evidence == [{
        "request_id": "docs",
        "task": "Find the public contract for a current API.",
        "resolved": True,
        "result": "official reference summary",
    }]
    assert calls == [{
        "task": "Find the public contract for a current API.",
        "context": {
            "coding_agent_task": "patch_proposal_external_evidence",
        },
    }]
    assert trace_summary == ["external_evidence request=docs resolved=True"]


def test_source_owner_candidates_use_evidence_windows(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.source_owners import (
        collect_source_owner_candidates,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "docs_src").mkdir()
    (repo_root / "src" / "error_branch.py").write_text(
        "class DomainError(Exception):\n"
        "    pass\n"
        "\n"
        "def handle(value: str) -> str:\n"
        "    if not value:\n"
        "        raise DomainError(\"value is required\")\n"
        "    return value\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "helper.py").write_text(
        "def unrelated() -> None:\n"
        "    raise RuntimeError(\"unrelated\")\n"
        "\n"
        "def enter(value: str) -> str:\n"
        "    return value\n",
        encoding="utf-8",
    )
    (repo_root / "docs_src" / "example.py").write_text(
        "def example() -> None:\n"
        "    raise RuntimeError(\"example\")\n",
        encoding="utf-8",
    )
    reading_evidence = [
        {
            "path": "src/helper.py",
            "line_start": 4,
            "line_end": 5,
            "symbol_or_topic": "helper",
            "excerpt": "def enter(value: str) -> str:",
            "reason": "Shows a helper call site.",
        },
        {
            "path": "docs_src/example.py",
            "line_start": 1,
            "line_end": 2,
            "symbol_or_topic": "example",
            "excerpt": "raise RuntimeError(\"example\")",
            "reason": "Shows documentation example behavior.",
        },
        {
            "path": "src/error_branch.py",
            "line_start": 4,
            "line_end": 7,
            "symbol_or_topic": "runtime error branch",
            "excerpt": "raise DomainError(\"value is required\")",
            "reason": "Shows the runtime error branch.",
        },
    ]

    candidates = collect_source_owner_candidates(
        repo_root=repo_root,
        reading_evidence=reading_evidence,
        max_candidates=5,
    )
    by_path = {candidate["path"]: candidate for candidate in candidates}

    assert candidates[0]["path"] == "src/error_branch.py"
    assert "raises_errors" in candidates[0]["feature_markers"]
    assert by_path["src/helper.py"]["role"] == "runtime"
    assert "raises_errors" not in by_path["src/helper.py"]["feature_markers"]
    assert by_path["docs_src/example.py"]["role"] == "docs"


def test_pm_assignment_evaluator_rejects_self_overlapping_paths() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.pm_evaluator import (
        evaluate_writing_pm_contract,
    )

    decision = {
        "status": "need_programmers",
        "mode": "edit_existing_repository",
        "intent": "bounded module update",
        "assignments": [{
            "assignment_id": "state_and_tests",
            "role": "state and test programmer",
            "scope": {"kind": "directory", "values": ["src/app"]},
            "owned_paths": ["src/app/state.py", "tests/test_state.py"],
            "read_only_paths": [],
            "interface_contract": {
                "component": "state module",
                "inputs": ["queue item"],
                "outputs": ["state update"],
                "invariants": ["state update remains deterministic"],
            },
            "integration_contract": {
                "provides_to_pm": ["state implementation and coverage"],
                "consumes_from": [],
            },
            "change_goal": "Update state behavior and focused tests.",
            "questions": ["Implement the state behavior and coverage."],
            "required_slots": ["state behavior", "test coverage"],
            "validation_expectations": ["focused tests cover the state path"],
            "forbidden_paths": [],
        }],
        "missing_slots": [],
        "reading_requests": [],
        "external_evidence_requests": [],
    }

    evaluation = evaluate_writing_pm_contract(
        decision=decision,
        source_scope={
            "kind": "repository",
            "repo_relative_path": None,
            "source_url": "local://fixture",
            "requested_ref": None,
            "interpretation": "fixture repository",
        },
        mode="edit_existing_repository",
    )

    assert evaluation["status"] == "repair_required"
    assert any(
        "overlapping scope or owned paths" in error
        for error in evaluation["errors"]
    )


def test_file_agent_resolves_new_file_demand_to_assignment(tmp_path: Path) -> None:
    from kazusa_ai_chatbot.coding_agent.file_agent import (
        resolve_writing_file_demands,
    )

    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "runtime.py").write_text(
        "def run() -> None:\n    pass\n",
        encoding="utf-8",
    )
    repository = {
        "local_root": str(repo_root),
    }
    source_scope = {
        "kind": "repository",
        "repo_relative_path": None,
        "source_url": "local://fixture",
        "requested_ref": None,
        "interpretation": "fixture repository",
    }

    resolution = resolve_writing_file_demands(
        mode="edit_existing_repository",
        repository=repository,
        source_scope=source_scope,
        owner_candidates=[{
            "path": "src/runtime.py",
            "role": "runtime",
            "line_start": 1,
            "line_end": 2,
            "symbols": ["run"],
            "exception_types": [],
            "feature_markers": ["python_source"],
            "reasons": ["runtime source candidate"],
            "evidence_refs": ["src/runtime.py:1-2"],
        }],
        file_demands=[{
            "demand_id": "runtime-helper",
            "role": "runtime helper programmer",
            "purpose": "Add a helper next to the runtime owner.",
            "file_kind": "new",
            "preferred_name": "helper.py",
            "placement_hint": "src",
            "related_paths": ["src/runtime.py"],
            "read_only_paths": ["src/runtime.py"],
            "interface_contract": {
                "component": "runtime helper",
                "inputs": ["runtime call"],
                "outputs": ["helper result"],
                "invariants": ["module remains importable"],
            },
            "integration_contract": {
                "provides_to_pm": ["helper implementation"],
                "consumes_from": [],
            },
            "change_goal": "Create a helper module.",
            "questions": ["Create the helper module."],
            "required_slots": ["helper module"],
            "validation_expectations": ["helper imports"],
            "forbidden_paths": [],
        }],
    )

    assert resolution["status"] == "accepted"
    assignment = resolution["assignments"][0]
    assert assignment["scope"] == {
        "kind": "file",
        "values": ["src/helper.py"],
    }
    assert assignment["owned_paths"] == ["src/helper.py"]
    assert assignment["read_only_paths"] == ["src/runtime.py"]


def test_file_agent_rejects_new_file_collision(tmp_path: Path) -> None:
    from kazusa_ai_chatbot.coding_agent.file_agent import (
        resolve_writing_file_demands,
    )

    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "existing.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    resolution = resolve_writing_file_demands(
        mode="edit_existing_repository",
        repository={"local_root": str(repo_root)},
        source_scope={
            "kind": "repository",
            "repo_relative_path": None,
            "source_url": "local://fixture",
            "requested_ref": None,
            "interpretation": "fixture repository",
        },
        owner_candidates=[],
        file_demands=[{
            "demand_id": "new-module",
            "role": "module programmer",
            "purpose": "Create a module.",
            "file_kind": "new",
            "preferred_path": "src/existing.py",
            "questions": ["Create the module."],
            "required_slots": ["module"],
        }],
    )

    assert resolution["status"] == "repair_required"
    assert resolution["assignments"] == []
    assert any("already exists" in error for error in resolution["errors"])


def test_file_agent_rejects_source_scope_escape(tmp_path: Path) -> None:
    from kazusa_ai_chatbot.coding_agent.file_agent import (
        resolve_writing_file_demands,
    )

    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)

    resolution = resolve_writing_file_demands(
        mode="edit_existing_repository",
        repository={"local_root": str(repo_root)},
        source_scope={
            "kind": "directory",
            "repo_relative_path": "src/app",
            "source_url": "local://fixture/src/app",
            "requested_ref": None,
            "interpretation": "fixture directory",
        },
        owner_candidates=[],
        file_demands=[{
            "demand_id": "outside",
            "role": "outside programmer",
            "purpose": "Create a file outside the selected source scope.",
            "file_kind": "new",
            "preferred_path": "tests/test_app.py",
            "questions": ["Create the file."],
            "required_slots": ["file"],
        }],
    )

    assert resolution["status"] == "repair_required"
    assert resolution["assignments"] == []
    assert any("source scope" in error for error in resolution["errors"])


def test_writing_pm_payload_excludes_source_file_hints() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
        _pm_payload,
    )

    owner_candidates = [
        {
            "path": "src/runtime.py",
            "role": "runtime",
            "line_start": 1,
            "line_end": 4,
            "symbols": ["handle"],
            "exception_types": ["ValueError"],
            "feature_markers": ["raises_errors"],
            "reasons": ["contains runtime error branch"],
            "evidence_refs": ["src/runtime.py:1-4"],
        }
    ]

    payload = _pm_payload({
        "question": "Improve one behavior.",
        "mode": "edit_existing_repository",
        "repository_summary": {"repo": "fixture"},
        "reading_reports": [{
            "status": "succeeded",
            "answer_text": "src/runtime.py owns the behavior.",
            "evidence_refs": ["src/runtime.py:1-4"],
            "evidence": [{
                "path": "src/runtime.py",
                "line_start": 1,
                "line_end": 4,
                "excerpt": "def handle() -> None: pass",
            }],
        }],
        "owner_candidates": owner_candidates,
        "previous_writing_reports": [],
    })

    assert "source_file_hints" not in payload
    assert payload["reading_reports"] == [{
        "status": "succeeded",
        "scope_details": "reserved_for_file_agent_and_file_pm",
    }]


def test_writing_pm_payload_includes_source_reading_state() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
        _pm_payload,
    )

    evidence_state = {
        "remaining_reading_attempts": 0,
        "merged_reading_evidence_count": 4,
        "completed_reading_requests": [{
            "request_id": "read-owner",
            "task": "Inspect current source owners.",
        }],
    }

    payload = _pm_payload({
        "question": "Improve one behavior.",
        "mode": "edit_existing_repository",
        "repository_summary": {"repo": "fixture"},
        "reading_reports": [],
        "supervisor_evidence_state": evidence_state,
        "owner_candidates": [],
        "previous_writing_reports": [],
    })

    assert payload["source_reading_state"] == evidence_state


def test_writing_pm_reading_reports_include_compact_evidence() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        _reading_reports,
    )

    reports = _reading_reports({
        "status": "succeeded",
        "answer_text": "Runtime owner evidence was found.",
        "evidence": [{
            "path": "src/runtime.py",
            "line_start": 10,
            "line_end": 14,
            "symbol_or_topic": "runtime owner",
            "excerpt": "def current_runtime_owner() -> None:\n    pass\n",
            "reason": "Shows the existing owner.",
        }],
        "limitations": [],
        "trace_summary": [],
    })

    assert reports == [{
        "status": "succeeded",
        "answer_text": "Runtime owner evidence was found.",
        "evidence_refs": ["src/runtime.py:10-14"],
        "evidence": [{
            "path": "src/runtime.py",
            "line_start": 10,
            "line_end": 14,
            "symbol_or_topic": "runtime owner",
            "excerpt": "def current_runtime_owner() -> None:\n    pass\n",
            "reason": "Shows the existing owner.",
        }],
        "limitations": [],
    }]


def test_writing_pm_reading_reports_bound_repeated_evidence() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        MAX_READING_REPORT_EVIDENCE_ROWS_PER_PATH,
        MAX_READING_REPORT_EXCERPT_CHARS,
        MAX_READING_REPORT_LIMITATIONS,
        _reading_reports,
    )

    evidence = []
    for index in range(5):
        evidence.append({
            "path": "src/runtime.py",
            "line_start": index + 1,
            "line_end": index + 1,
            "symbol_or_topic": "runtime branch",
            "excerpt": "runtime detail " * 80,
            "reason": "Runtime owner evidence.",
        })
    for index in range(4):
        evidence.append({
            "path": "tests/test_runtime.py",
            "line_start": index + 1,
            "line_end": index + 1,
            "symbol_or_topic": "test branch",
            "excerpt": "test detail " * 80,
            "reason": "Test owner evidence.",
        })

    reports = _reading_reports({
        "status": "succeeded",
        "answer_text": "summary detail " * 400,
        "evidence": evidence,
        "limitations": ["limitation detail " * 80 for _ in range(8)],
        "trace_summary": [],
    })

    report = reports[0]
    evidence_paths = [row["path"] for row in report["evidence"]]

    assert len(report["limitations"]) == MAX_READING_REPORT_LIMITATIONS
    assert report["answer_text"].endswith("... [truncated]")
    assert evidence_paths.count("src/runtime.py") == (
        MAX_READING_REPORT_EVIDENCE_ROWS_PER_PATH
    )
    assert evidence_paths.count("tests/test_runtime.py") == (
        MAX_READING_REPORT_EVIDENCE_ROWS_PER_PATH
    )
    assert all(
        len(row["excerpt"]) <= MAX_READING_REPORT_EXCERPT_CHARS
        for row in report["evidence"]
    )


def test_programmer_evidence_is_selected_by_assignment_scope() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        _reading_evidence_for_assignment,
    )

    reading_evidence = [
        {
            "path": "src/runtime.py",
            "line_start": 1,
            "line_end": 4,
            "symbol_or_topic": "runtime",
            "excerpt": "runtime evidence",
            "reason": "Runtime owner.",
        },
        {
            "path": "src/service.py",
            "line_start": 10,
            "line_end": 14,
            "symbol_or_topic": "service",
            "excerpt": "service evidence",
            "reason": "Service owner.",
        },
        {
            "path": "docs/usage.md",
            "line_start": 20,
            "line_end": 22,
            "symbol_or_topic": "docs",
            "excerpt": "docs evidence",
            "reason": "Documentation owner.",
        },
    ]
    assignment = {
        "assignment_id": "service",
        "role": "service programmer",
        "scope": {
            "kind": "file",
            "values": ["src/service.py"],
        },
        "questions": ["Update service behavior."],
        "required_slots": ["service patch"],
    }

    selected = _reading_evidence_for_assignment(
        reading_evidence=reading_evidence,
        owner_candidates=[],
        assignment=assignment,
    )

    assert [row["path"] for row in selected] == ["src/service.py"]


def test_programmer_evidence_does_not_expand_missing_file_scope() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        _reading_evidence_for_assignment,
    )

    reading_evidence = [
        {
            "path": "src/runtime.py",
            "line_start": 1,
            "line_end": 4,
            "symbol_or_topic": "runtime",
            "excerpt": "runtime evidence",
            "reason": "Runtime owner.",
        },
        {
            "path": "tests/test_runtime.py",
            "line_start": 10,
            "line_end": 14,
            "symbol_or_topic": "test",
            "excerpt": "test evidence",
            "reason": "Test owner.",
        },
    ]
    assignment = {
        "assignment_id": "new-test",
        "role": "test programmer",
        "scope": {
            "kind": "file",
            "values": ["tests/test_new_behavior.py"],
        },
        "questions": ["Create a focused test file."],
        "required_slots": ["test coverage"],
    }

    selected = _reading_evidence_for_assignment(
        reading_evidence=reading_evidence,
        owner_candidates=[{
            "path": "src/runtime.py",
            "role": "runtime",
            "line_start": 1,
            "line_end": 4,
            "symbols": [],
            "exception_types": [],
            "feature_markers": [],
            "reasons": ["runtime owner"],
            "evidence_refs": ["src/runtime.py:1-4"],
        }],
        assignment=assignment,
    )

    assert selected == []


def test_many_file_assignment_uses_broad_evidence_budget() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        MAX_BROAD_SCOPE_READING_EVIDENCE_ROWS,
        _reading_evidence_for_assignment,
    )

    reading_evidence = [
        {
            "path": f"src/module_{index}.py",
            "line_start": 1,
            "line_end": 4,
            "symbol_or_topic": "module",
            "excerpt": "module evidence",
            "reason": "Generic owner evidence.",
        }
        for index in range(12)
    ]
    assignment = {
        "assignment_id": "many-files",
        "role": "many file programmer",
        "scope": {
            "kind": "file",
            "values": [f"src/module_{index}.py" for index in range(8)],
        },
        "questions": ["Update bounded files."],
        "required_slots": ["patch"],
    }

    selected = _reading_evidence_for_assignment(
        reading_evidence=reading_evidence,
        owner_candidates=[],
        assignment=assignment,
    )

    assert len(selected) == MAX_BROAD_SCOPE_READING_EVIDENCE_ROWS


def test_writing_pm_payload_strips_source_scope_from_reading_reports() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
        MAX_PM_READING_FACTS,
        _pm_payload,
    )

    payload = _pm_payload({
        "question": "Improve one behavior.",
        "mode": "edit_existing_repository",
        "repository_summary": {"repo": "fixture"},
        "reading_reports": [{
            "status": "succeeded",
            "facts": [f"semantic fact {index}" for index in range(20)],
            "evidence_refs": ["src/runtime.py:1-4"],
            "evidence": [{
                "path": "src/runtime.py",
                "line_start": 1,
                "line_end": 4,
                "excerpt": "def handle() -> None: pass",
            }],
            "limitations": ["limited source evidence"],
        }],
        "owner_candidates": [],
        "previous_writing_reports": [],
    })

    report = payload["reading_reports"][0]

    assert "source_file_hints" not in payload
    assert report["facts"] == [
        f"semantic fact {index}" for index in range(MAX_PM_READING_FACTS)
    ]
    assert report["limitations_present"] is True
    assert "evidence_refs" not in report
    assert "evidence" not in report


def test_writing_pm_accepts_new_project_file_demands() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
        normalize_writing_pm_decision,
    )

    decision = normalize_writing_pm_decision(
        {
            "status": "need_programmers",
            "mode": "create_new_project",
            "intent": "create limited files",
            "file_demands": [{
                "demand_id": "module",
                "role": "module programmer",
                "purpose": "Create a module and focused tests.",
                "file_kind": "new",
                "preferred_path": "package/module.py",
                "related_paths": ["tests/test_module.py"],
                "questions": ["Create the module and its tests."],
                "required_slots": ["Use one package layout."],
            }],
            "missing_slots": [],
            "external_evidence_requests": [],
        },
        mode="create_new_project",
    )

    assert decision["status"] == "need_programmers"
    assert decision["assignments"] == []
    assert decision["file_demands"][0]["preferred_path"] == "package/module.py"
    assert decision["file_demands"][0]["related_paths"] == ["tests/test_module.py"]


def test_writing_pm_normalizes_reading_evidence_request() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
        normalize_writing_pm_decision,
    )

    decision = normalize_writing_pm_decision(
        {
            "status": "need_reading",
            "mode": "edit_existing_repository",
            "intent": "bounded existing-source update",
            "reading_requests": [{
                "request_id": "owners",
                "task": "Inspect current runtime owners and validation paths.",
                "reason": "Source evidence is required before assignments.",
                "required_slots": [
                    "Current runtime owner.",
                    "Visible tests for the behavior.",
                ],
            }],
            "assignments": [{
                "assignment_id": "ignored",
                "role": "patch programmer",
                "scope": {
                    "kind": "file",
                    "values": ["src/runtime.py"],
                },
                "questions": ["Produce a patch."],
                "required_slots": ["runtime change"],
            }],
            "missing_slots": [],
            "external_evidence_requests": [],
        },
        mode="edit_existing_repository",
    )

    assert decision["status"] == "need_reading"
    assert decision["assignments"] == []
    assert decision["reading_requests"] == [{
        "request_id": "owners",
        "task": "Inspect current runtime owners and validation paths.",
        "reason": "Source evidence is required before assignments.",
        "required_slots": [
            "Current runtime owner.",
            "Visible tests for the behavior.",
        ],
    }]


def test_programmer_payload_includes_source_file_hints() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
        _programmer_payload,
    )

    owner_candidates = [
        {
            "path": "src/runtime.py",
            "role": "runtime",
            "line_start": 1,
            "line_end": 4,
            "symbols": ["handle"],
            "exception_types": ["ValueError"],
            "feature_markers": ["raises_errors"],
            "reasons": ["contains runtime error branch"],
            "evidence_refs": ["src/runtime.py:1-4"],
        }
    ]

    payload = _programmer_payload(
        question="Improve one behavior.",
        mode="edit_existing_repository",
        assignment={
            "assignment_id": "runtime",
            "role": "patch programmer",
            "scope": {
                "kind": "file",
                "values": ["src/runtime.py"],
            },
            "questions": ["Update the runtime owner."],
            "required_slots": ["runtime owner"],
        },
        repository_summary={"repo": "fixture"},
        reading_evidence=[],
        owner_candidates=owner_candidates,
        file_context=[],
        external_evidence=[],
        repair_context=None,
    )

    assert payload["source_file_hints"] == owner_candidates


def test_programmer_payload_filters_source_file_hints_by_assignment() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
        _programmer_payload,
    )

    owner_candidates = [
        {
            "path": "src/runtime.py",
            "role": "runtime",
            "line_start": 1,
            "line_end": 20,
            "symbols": ["run"],
            "exception_types": [],
            "feature_markers": ["python_source"],
            "reasons": ["runtime source candidate"],
            "evidence_refs": ["src/runtime.py:1-20"],
        },
        {
            "path": "src/unrelated.py",
            "role": "runtime",
            "line_start": 1,
            "line_end": 20,
            "symbols": ["unused"],
            "exception_types": [],
            "feature_markers": ["python_source"],
            "reasons": ["runtime source candidate"],
            "evidence_refs": ["src/unrelated.py:1-20"],
        },
    ]

    payload = _programmer_payload(
        question="Update the runtime behavior.",
        mode="edit_existing_repository",
        assignment={
            "assignment_id": "runtime",
            "role": "Runtime owner",
            "scope": {"kind": "file", "values": ["src/runtime.py"]},
            "questions": ["Update runtime.py."],
            "required_slots": ["runtime behavior"],
        },
        repository_summary=None,
        reading_evidence=[],
        owner_candidates=owner_candidates,
        file_context=[],
        external_evidence=[],
    )

    assert payload["source_file_hints"] == [owner_candidates[0]]


def test_programmer_payload_compacts_question_and_evidence() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
        MAX_PROGRAMMER_EVIDENCE_EXCERPT_CHARS,
        MAX_PROGRAMMER_EVIDENCE_REASON_CHARS,
        MAX_PROGRAMMER_QUESTION_CHARS,
        _programmer_payload,
    )

    payload = _programmer_payload(
        question="Q" * (MAX_PROGRAMMER_QUESTION_CHARS + 100),
        mode="edit_existing_repository",
        assignment={
            "assignment_id": "runtime",
            "role": "patch programmer",
            "scope": {
                "kind": "file",
                "values": ["src/runtime.py"],
            },
            "questions": ["Update the runtime owner."],
            "required_slots": ["runtime owner"],
        },
        repository_summary={"repo": "fixture"},
        reading_evidence=[{
            "path": "src/runtime.py",
            "line_start": 1,
            "line_end": 2,
            "symbol_or_topic": "runtime owner",
            "excerpt": "E" * (MAX_PROGRAMMER_EVIDENCE_EXCERPT_CHARS + 100),
            "reason": "R" * (MAX_PROGRAMMER_EVIDENCE_REASON_CHARS + 100),
        }],
        owner_candidates=[],
        file_context=[],
        external_evidence=[],
        repair_context=None,
    )

    evidence = payload["reading_evidence"][0]

    assert len(payload["question"]) == MAX_PROGRAMMER_QUESTION_CHARS
    assert len(evidence["excerpt"]) == MAX_PROGRAMMER_EVIDENCE_EXCERPT_CHARS
    assert len(evidence["reason"]) == MAX_PROGRAMMER_EVIDENCE_REASON_CHARS


def test_programmer_payload_includes_peer_assignments() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
        _programmer_payload,
    )

    peer_assignments = [
        {
            "assignment_id": "core",
            "role": "core programmer",
            "scope": {
                "kind": "file",
                "values": ["core.py"],
            },
            "questions": ["Implement the core module."],
            "required_slots": ["core.create_record returns one shared shape"],
        },
        {
            "assignment_id": "tests",
            "role": "test programmer",
            "scope": {
                "kind": "file",
                "values": ["tests/test_core.py"],
            },
            "questions": ["Test the core module."],
            "required_slots": ["tests import the core module consistently"],
        },
    ]

    payload = _programmer_payload(
        question="Create a small new project.",
        mode="create_new_project",
        assignment=peer_assignments[1],
        repository_summary=None,
        reading_evidence=[],
        owner_candidates=[],
        file_context=[],
        external_evidence=[],
        peer_assignments=peer_assignments,
        repair_context=None,
    )

    assert payload["peer_assignments"] == peer_assignments


def test_file_context_includes_import_window_for_large_python_file(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        _file_context,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    filler = "\n".join(
        f"VALUE_{index} = {index}"
        for index in range(900)
    )
    (repo_root / "src" / "large_runtime.py").write_text(
        "from src.errors import ExistingError\n"
        "\n"
        f"{filler}\n"
        "\n"
        "def handle(value: str) -> str:\n"
        "    if not value:\n"
        "        raise ExistingError(\"value is required\")\n"
        "    return value\n",
        encoding="utf-8",
    )
    reading_evidence = [{
        "path": "src/large_runtime.py",
        "line_start": 904,
        "line_end": 907,
        "symbol_or_topic": "runtime branch",
        "excerpt": "raise ExistingError(\"value is required\")",
        "reason": "Shows the target branch.",
    }]

    context_rows = _file_context(
        repository={"local_root": str(repo_root)},
        reading_evidence=reading_evidence,
        owner_candidates=[],
    )

    assert any(
        row["path"] == "src/large_runtime.py"
        and row["line_start"] == 1
        and "from src.errors import ExistingError" in row["text"]
        for row in context_rows
    )
    assert any(
        row["path"] == "src/large_runtime.py"
        and row["line_start"] > 1
        and "raise ExistingError" in row["text"]
        for row in context_rows
    )


def test_file_context_keeps_multiple_owner_paths_visible(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        _file_context,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "helper.py").write_text(
        "\n".join(f"VALUE_{index} = {index}" for index in range(1200)),
        encoding="utf-8",
    )
    (repo_root / "src" / "owner.py").write_text(
        "def handle(value: str) -> str:\n"
        "    if not value:\n"
        "        raise ValueError(\"value is required\")\n"
        "    return value\n",
        encoding="utf-8",
    )
    helper_rows = [
        {
            "path": "src/helper.py",
            "line_start": line,
            "line_end": line + 3,
            "symbol_or_topic": "helper",
            "excerpt": "helper function",
            "reason": "Shows helper context.",
        }
        for line in range(100, 1100, 80)
    ]
    reading_evidence = [
        *helper_rows,
        {
            "path": "src/owner.py",
            "line_start": 1,
            "line_end": 4,
            "symbol_or_topic": "runtime owner",
            "excerpt": "raise ValueError(\"value is required\")",
            "reason": "Shows runtime owner.",
        },
    ]
    owner_candidates = [
        {
            "path": "src/helper.py",
            "role": "runtime",
            "line_start": 1,
            "line_end": 900,
            "symbols": [],
            "exception_types": [],
            "feature_markers": ["python_source"],
            "reasons": ["runtime source candidate"],
            "evidence_refs": ["src/helper.py:1-40"],
        },
        {
            "path": "src/owner.py",
            "role": "runtime",
            "line_start": 1,
            "line_end": 4,
            "symbols": ["handle"],
            "exception_types": ["ValueError"],
            "feature_markers": ["python_source", "raises_errors"],
            "reasons": ["contains runtime error branch"],
            "evidence_refs": ["src/owner.py:1-4"],
        },
    ]

    context_rows = _file_context(
        repository={"local_root": str(repo_root)},
        reading_evidence=reading_evidence,
        owner_candidates=owner_candidates,
    )

    assert any(row["path"] == "src/helper.py" for row in context_rows)
    assert any(
        row["path"] == "src/owner.py" and "raise ValueError" in row["text"]
        for row in context_rows
    )


def test_file_context_includes_pm_assigned_writable_paths(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        _file_context,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "docs").mkdir()
    (repo_root / "src" / "owner.py").write_text(
        "def handle() -> str:\n"
        "    return \"ok\"\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "usage.md").write_text(
        "## Current Endpoint\n\nCurrent behavior details.\n",
        encoding="utf-8",
    )
    reading_evidence = [{
        "path": "src/owner.py",
        "line_start": 1,
        "line_end": 2,
        "symbol_or_topic": "runtime owner",
        "excerpt": "def handle() -> str:",
        "reason": "Shows runtime owner.",
    }]

    context_rows = _file_context(
        repository={"local_root": str(repo_root)},
        reading_evidence=reading_evidence,
        owner_candidates=[],
        assignment_paths=["docs/usage.md"],
    )

    assert any(
        row["path"] == "docs/usage.md"
        and "## Current Endpoint" in row["text"]
        for row in context_rows
    )


def test_broad_assignment_uses_tighter_context_budget(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        MAX_BROAD_SCOPE_FILE_CONTEXT_CHARS_PER_FILE,
        MAX_BROAD_SCOPE_FILE_CONTEXT_FILES,
        MAX_BROAD_SCOPE_READING_EVIDENCE_ROWS,
        _file_context_for_assignment,
        _reading_evidence_for_assignment,
    )

    repo_root = tmp_path / "write_repo"
    tests_dir = repo_root / "tests"
    tests_dir.mkdir(parents=True)
    reading_evidence = []
    for index in range(12):
        path = f"tests/test_case_{index}.py"
        (repo_root / path).write_text(
            "\n".join(f"VALUE_{line} = {line}" for line in range(1200)),
            encoding="utf-8",
        )
        reading_evidence.append({
            "path": path,
            "line_start": 100,
            "line_end": 140,
            "symbol_or_topic": "test context",
            "excerpt": "assert behavior",
            "reason": "Shows a test pattern.",
        })
    assignment = {
        "assignment_id": "tests",
        "role": "test owner",
        "scope": {
            "kind": "directory",
            "values": ["tests"],
        },
        "questions": ["Add focused tests."],
        "required_slots": ["test coverage"],
    }

    selected_evidence = _reading_evidence_for_assignment(
        reading_evidence=reading_evidence,
        owner_candidates=[],
        assignment=assignment,
    )
    context_rows = _file_context_for_assignment(
        repository={"local_root": str(repo_root)},
        reading_evidence=selected_evidence,
        owner_candidates=[],
        assignment=assignment,
    )

    assert len(selected_evidence) == MAX_BROAD_SCOPE_READING_EVIDENCE_ROWS
    assert len(context_rows) <= MAX_BROAD_SCOPE_FILE_CONTEXT_FILES
    assert all(
        len(str(row["text"])) <= MAX_BROAD_SCOPE_FILE_CONTEXT_CHARS_PER_FILE
        for row in context_rows
    )


def test_single_file_assignment_gets_larger_exact_context(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        MAX_FILE_CONTEXT_CHARS_PER_FILE,
        _file_context_for_assignment,
    )

    repo_root = tmp_path / "write_repo"
    source_dir = repo_root / "src"
    source_dir.mkdir(parents=True)
    filler = "\n".join(
        f"VALUE_{index} = {index}"
        for index in range(420)
    )
    (source_dir / "runtime.py").write_text(
        "class Runtime:\n"
        "    def __init__(self) -> None:\n"
        "        self.count = 0\n"
        f"{filler}\n"
        "    def current_behavior(self) -> int:\n"
        "        self.count += 1\n"
        "        return self.count\n",
        encoding="utf-8",
    )
    reading_evidence = [{
        "path": "src/runtime.py",
        "line_start": 420,
        "line_end": 426,
        "symbol_or_topic": "runtime owner",
        "excerpt": "def current_behavior",
        "reason": "Shows the target method.",
    }]
    assignment = {
        "assignment_id": "runtime",
        "role": "runtime owner",
        "scope": {
            "kind": "symbol",
            "values": ["Runtime.current_behavior"],
        },
        "owned_paths": ["src/runtime.py"],
        "questions": ["Update one runtime file."],
        "required_slots": ["runtime patch"],
    }

    context_rows = _file_context_for_assignment(
        repository={"local_root": str(repo_root)},
        reading_evidence=reading_evidence,
        owner_candidates=[],
        assignment=assignment,
    )

    runtime_rows = [
        row for row in context_rows
        if row["path"] == "src/runtime.py"
    ]
    assert runtime_rows
    assert len(str(runtime_rows[0]["text"])) > MAX_FILE_CONTEXT_CHARS_PER_FILE
    assert "def __init__" in str(runtime_rows[0]["text"])
    assert "def current_behavior" in str(runtime_rows[0]["text"])


def test_patch_validation_accepts_safe_diff_without_mutating_repo(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = _make_repo(tmp_path)
    workspace_root = tmp_path / "workspace"
    diff_text = _new_file_diff("src/generated.py")

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=workspace_root,
        patch_artifacts=[_artifact(diff_text)],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "succeeded"
    assert validation["parsed"] is True
    assert validation["sandbox_applied"] is True
    assert validation["errors"] == []
    assert not (repo_root / "src" / "generated.py").exists()
    assert "write_repo" not in repr(validation)
    assert "workspace" not in repr(validation)


def test_patch_validation_rejects_unsafe_paths(tmp_path: Path) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = _make_repo(tmp_path)
    diff_text = _new_file_diff("../outside.py")

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[_artifact(diff_text)],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "rejected"
    assert validation["parsed"] is False
    assert validation["sandbox_applied"] is False
    assert validation["errors"]
    assert not (tmp_path / "outside.py").exists()


def test_patch_validation_rejects_markdown_env_assignment_outside_fence(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    repo_root.mkdir()
    (repo_root / "docs.md").write_text(
        "```env\n"
        "EXISTING=value\n"
        "```\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/docs.md b/docs.md",
            "--- a/docs.md",
            "+++ b/docs.md",
            "@@ -1,3 +1,5 @@",
            " ```env",
            " EXISTING=value",
            " ```",
            "+# Runtime",
            "+NEW_SETTING=",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "docs",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["docs.md"],
            "summary": "Documents one setting.",
        }],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "failed"
    assert validation["parsed"] is True
    assert validation["sandbox_applied"] is False
    assert validation["errors"] == [
        "Markdown environment-style assignments must be inside "
        "fenced code blocks.",
    ]


def test_patch_validation_accepts_markdown_env_assignment_inside_fence(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    repo_root.mkdir()
    (repo_root / "docs.md").write_text(
        "```env\n"
        "EXISTING=value\n"
        "```\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/docs.md b/docs.md",
            "--- a/docs.md",
            "+++ b/docs.md",
            "@@ -1,3 +1,4 @@",
            " ```env",
            " EXISTING=value",
            "+NEW_SETTING=",
            " ```",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "docs",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["docs.md"],
            "summary": "Documents one setting.",
        }],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_validation_preserves_blank_context_between_artifacts(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    repo_root.mkdir()
    (repo_root / "first.py").write_text(
        "VALUE = \"old\"\n"
        "\n",
        encoding="utf-8",
    )
    (repo_root / "second.py").write_text(
        "OTHER = \"old\"\n",
        encoding="utf-8",
    )
    first_diff = "\n".join(
        [
            "diff --git a/first.py b/first.py",
            "--- a/first.py",
            "+++ b/first.py",
            "@@ -1,2 +1,2 @@",
            "-VALUE = \"old\"",
            "+VALUE = \"new\"",
            " ",
            "",
        ]
    )
    second_diff = "\n".join(
        [
            "diff --git a/second.py b/second.py",
            "--- a/second.py",
            "+++ b/second.py",
            "@@ -1 +1 @@",
            "-OTHER = \"old\"",
            "+OTHER = \"new\"",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[
            {
                "artifact_id": "first",
                "base": "repository",
                "diff_text": first_diff,
                "files": ["first.py"],
                "summary": "Updates first file.",
            },
            {
                "artifact_id": "second",
                "base": "repository",
                "diff_text": second_diff,
                "files": ["second.py"],
                "summary": "Updates second file.",
            },
        ],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_validation_rejects_git_apply_warnings(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    repo_root.mkdir()
    (repo_root / "sample.py").write_text(
        "def run() -> int:\n"
        "    return 1\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/sample.py b/sample.py",
            "--- a/sample.py",
            "+++ b/sample.py",
            "@@ -1,2 +1,3 @@",
            " def run() -> int:",
            "+    value = 2",
            "     return 1",
            "? unexpected line",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["sample.py"],
            "summary": "Adds one line with malformed diff context.",
        }],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "failed"
    assert validation["errors"] == [
        "Patch did not apply cleanly in isolated validation: "
        "warning: recount: unexpected line: ? unexpected line?",
    ]


def test_patch_validation_rejects_test_patch_without_runtime_behavior(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    (repo_root / "src" / "errors.py").write_text(
        "class ExistingError(Exception):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_errors.py").write_text(
        "def test_existing_error():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/errors.py b/src/errors.py",
            "--- a/src/errors.py",
            "+++ b/src/errors.py",
            "@@ -1,2 +1,6 @@",
            " class ExistingError(Exception):",
            "     pass",
            "+class NewError(Exception):",
            "+    \"\"\"Raised for the new failure case.\"\"\"",
            "+",
            "+",
            "diff --git a/tests/test_errors.py b/tests/test_errors.py",
            "--- a/tests/test_errors.py",
            "+++ b/tests/test_errors.py",
            "@@ -1,2 +1,5 @@",
            " def test_existing_error():",
            "     assert True",
            "+def test_new_error():",
            "+    assert True",
            "+",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/errors.py", "tests/test_errors.py"],
            "summary": "Adds declaration and test only.",
        }],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "failed"
    assert validation["parsed"] is True
    assert validation["sandbox_applied"] is False
    assert validation["errors"] == [
        "Patch updates tests but does not include executable runtime source "
        "changes.",
    ]


def test_patch_validation_accepts_test_patch_with_runtime_behavior(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    (repo_root / "src" / "runtime.py").write_text(
        "def handle(value: str) -> str:\n"
        "    return value\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_runtime.py").write_text(
        "def test_handle():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -1,2 +1,4 @@",
            " def handle(value: str) -> str:",
            "+    if not value:",
            "+        raise ValueError(\"value is required\")",
            "     return value",
            "diff --git a/tests/test_runtime.py b/tests/test_runtime.py",
            "--- a/tests/test_runtime.py",
            "+++ b/tests/test_runtime.py",
            "@@ -1,2 +1,5 @@",
            " def test_handle():",
            "     assert True",
            "+def test_handle_empty():",
            "+    assert True",
            "+",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/runtime.py", "tests/test_runtime.py"],
            "summary": "Adds runtime behavior and test.",
        }],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_validation_runs_tests_against_sandbox_src_layout(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src" / "kazusa_ai_chatbot").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    (repo_root / "src" / "kazusa_ai_chatbot" / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )
    (repo_root / "src" / "kazusa_ai_chatbot" / "service.py").write_text(
        "class ExistingModel:\n"
        "    pass\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/kazusa_ai_chatbot/service.py b/src/kazusa_ai_chatbot/service.py",
            "--- a/src/kazusa_ai_chatbot/service.py",
            "+++ b/src/kazusa_ai_chatbot/service.py",
            "@@ -1,2 +1,6 @@",
            " class ExistingModel:",
            "     pass",
            "+",
            "+",
            "+def generated_validation_value() -> int:",
            "+    return 7",
            "diff --git a/tests/test_service_import.py b/tests/test_service_import.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/tests/test_service_import.py",
            "@@ -0,0 +1,5 @@",
            "+from kazusa_ai_chatbot.service import generated_validation_value",
            "+",
            "+",
            "+def test_imports_patched_sandbox_module():",
            "+    assert generated_validation_value() == 7",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": [
                "src/kazusa_ai_chatbot/service.py",
                "tests/test_service_import.py",
            ],
            "summary": "Adds runtime model and import test.",
        }],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_validation_accepts_runtime_error_message_change(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    (repo_root / "src" / "runtime.py").write_text(
        "def handle() -> None:\n"
        "    raise RuntimeError(\n"
        "        \"old message\"\n"
        "    )\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_runtime.py").write_text(
        "import pytest\n"
        "\n"
        "from src.runtime import handle\n"
        "\n"
        "def test_handle() -> None:\n"
        "    with pytest.raises(RuntimeError, match=\"old message\"):\n"
        "        handle()\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -1,4 +1,4 @@",
            " def handle() -> None:",
            "     raise RuntimeError(",
            "-        \"old message\"",
            "+        \"new actionable message\"",
            "     )",
            "diff --git a/tests/test_runtime.py b/tests/test_runtime.py",
            "--- a/tests/test_runtime.py",
            "+++ b/tests/test_runtime.py",
            "@@ -3,5 +3,5 @@",
            " from src.runtime import handle",
            " ",
            " def test_handle() -> None:",
            "-    with pytest.raises(RuntimeError, match=\"old message\"):",
            "+    with pytest.raises(RuntimeError, match=\"new actionable message\"):",
            "         handle()",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "message",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/runtime.py", "tests/test_runtime.py"],
            "summary": "Improves error message.",
        }],
        max_files=10,
        max_diff_chars=5000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_validation_rejects_broad_runtime_exception_wrapper(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "runtime.py").write_text(
        "def handle() -> str:\n"
        "    return do_work()\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -1,2 +1,5 @@",
            " def handle() -> str:",
            "-    return do_work()",
            "+    try:",
            "+        return do_work()",
            "+    except Exception as exc:",
            "+        raise RuntimeError(\"work failed\") from exc",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/runtime.py"],
            "summary": "Adds broad wrapper.",
        }],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "failed"
    assert validation["errors"] == [
        "Patch adds broad runtime exception wrapping: src/runtime.py adds "
        "`except Exception as exc:`. Remove the new try/except and modify the "
        "smallest existing error branch, or use a specific observed exception "
        "type while preserving original exception propagation.",
    ]


def test_patch_validation_rejects_broad_test_exception_assertion(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    (repo_root / "src" / "runtime.py").write_text(
        "def handle(value: str) -> str:\n"
        "    if not value:\n"
        "        raise RuntimeError(\"old message\")\n"
        "    return value\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_runtime.py").write_text(
        "from src.runtime import handle\n"
        "\n"
        "def test_handle() -> None:\n"
        "    assert handle(\"ok\") == \"ok\"\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -1,4 +1,4 @@",
            " def handle(value: str) -> str:",
            "     if not value:",
            "-        raise RuntimeError(\"old message\")",
            "+        raise RuntimeError(\"new message\")",
            "     return value",
            "diff --git a/tests/test_runtime.py b/tests/test_runtime.py",
            "--- a/tests/test_runtime.py",
            "+++ b/tests/test_runtime.py",
            "@@ -2,3 +2,10 @@",
            " ",
            " def test_handle() -> None:",
            "     assert handle(\"ok\") == \"ok\"",
            "+",
            "+def test_error_message() -> None:",
            "+    try:",
            "+        handle(\"\")",
            "+    except Exception as exc:",
            "+        assert \"new message\" in str(exc)",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/runtime.py", "tests/test_runtime.py"],
            "summary": "Adds runtime behavior and a weak exception assertion.",
        }],
        max_files=10,
        max_diff_chars=5000,
    )

    assert validation["status"] == "failed"
    assert validation["errors"] == [
        "Patch adds broad exception handling in tests; use a specific "
        "exception type or a test-framework exception assertion so the test "
        "fails when the expected exception is not raised.",
    ]


def test_patch_validation_rejects_test_exception_rewrite_without_runtime_raise(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    (repo_root / "src" / "runtime.py").write_text(
        "class NewError(Exception):\n"
        "    pass\n"
        "\n"
        "def handle() -> None:\n"
        "    raise ValueError(\"old\")\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_runtime.py").write_text(
        "import pytest\n"
        "\n"
        "from src.runtime import handle, NewError\n"
        "\n"
        "def test_handle() -> None:\n"
        "    with pytest.raises(ValueError):\n"
        "        handle()\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -3,3 +3,3 @@",
            " ",
            " def handle() -> None:",
            "-    raise ValueError(\"old\")",
            "+    raise ValueError(\"new detail\")",
            "diff --git a/tests/test_runtime.py b/tests/test_runtime.py",
            "--- a/tests/test_runtime.py",
            "+++ b/tests/test_runtime.py",
            "@@ -4,5 +4,5 @@",
            " ",
            " def test_handle() -> None:",
            "-    with pytest.raises(ValueError):",
            "+    with pytest.raises(NewError):",
            "         handle()",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/runtime.py", "tests/test_runtime.py"],
            "summary": "Changes a test exception expectation.",
        }],
        max_files=10,
        max_diff_chars=5000,
    )

    assert validation["status"] == "failed"
    assert validation["errors"] == [
        "Patch changes test expected exception types without matching runtime "
        "raise changes."
    ]


def test_patch_validation_rejects_test_match_rewrite_without_runtime_exception(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    (repo_root / "src" / "runtime.py").write_text(
        "def handle() -> None:\n"
        "    raise RuntimeError(\"old runtime detail\")\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_runtime.py").write_text(
        "import pytest\n"
        "\n"
        "from src.runtime import handle\n"
        "\n"
        "def test_handle() -> None:\n"
        "    with pytest.raises(ValueError, match=\"old detail\"):\n"
        "        handle()\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -1,2 +1,2 @@",
            " def handle() -> None:",
            "-    raise RuntimeError(\"old runtime detail\")",
            "+    raise RuntimeError(\"new runtime detail\")",
            "diff --git a/tests/test_runtime.py b/tests/test_runtime.py",
            "--- a/tests/test_runtime.py",
            "+++ b/tests/test_runtime.py",
            "@@ -4,5 +4,8 @@",
            " ",
            " def test_handle() -> None:",
            "-    with pytest.raises(ValueError, match=\"old detail\"):",
            "+    with pytest.raises(",
            "+        ValueError,",
            "+        match=r\"old detail.*new runtime detail\",",
            "+    ):",
            "         handle()",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/runtime.py", "tests/test_runtime.py"],
            "summary": "Changes a test match for the wrong exception.",
        }],
        max_files=10,
        max_diff_chars=5000,
    )

    assert validation["status"] == "failed"
    assert validation["errors"] == [
        "Patch changes test expected exception messages without matching "
        "runtime raise or message changes for the same exception type."
    ]


def test_patch_validation_accepts_test_exception_rewrite_with_runtime_raise(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    (repo_root / "src" / "runtime.py").write_text(
        "class NewError(Exception):\n"
        "    pass\n"
        "\n"
        "def handle() -> None:\n"
        "    raise ValueError(\"old\")\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_runtime.py").write_text(
        "import pytest\n"
        "\n"
        "from src.runtime import handle, NewError\n"
        "\n"
        "def test_handle() -> None:\n"
        "    with pytest.raises(ValueError):\n"
        "        handle()\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -3,3 +3,3 @@",
            " ",
            " def handle() -> None:",
            "-    raise ValueError(\"old\")",
            "+    raise NewError(\"new detail\")",
            "diff --git a/tests/test_runtime.py b/tests/test_runtime.py",
            "--- a/tests/test_runtime.py",
            "+++ b/tests/test_runtime.py",
            "@@ -4,5 +4,5 @@",
            " ",
            " def test_handle() -> None:",
            "-    with pytest.raises(ValueError):",
            "+    with pytest.raises(NewError):",
            "         handle()",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/runtime.py", "tests/test_runtime.py"],
            "summary": "Changes runtime and test exception types together.",
        }],
        max_files=10,
        max_diff_chars=5000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_validation_allows_new_specific_exception_test(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    (repo_root / "src" / "runtime.py").write_text(
        "class ExistingError(Exception):\n"
        "    pass\n"
        "\n"
        "def handle() -> None:\n"
        "    raise ExistingError(\n"
        "        \"old detail\"\n"
        "    )\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_runtime.py").write_text(
        "import pytest\n"
        "\n"
        "from src.runtime import ExistingError, handle\n"
        "\n"
        "def test_existing() -> None:\n"
        "    handle()\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -4,5 +4,5 @@",
            " def handle() -> None:",
            "     raise ExistingError(",
            "-        \"old detail\"",
            "+        \"new detail\"",
            "     )",
            "diff --git a/tests/test_runtime.py b/tests/test_runtime.py",
            "--- a/tests/test_runtime.py",
            "+++ b/tests/test_runtime.py",
            "@@ -5,3 +5,7 @@",
            " def test_existing() -> None:",
            "-    handle()",
            "+    with pytest.raises(ExistingError, match=\"new detail\"):",
            "+        handle()",
            "+",
            "+def test_new_detail() -> None:",
            "+    with pytest.raises(ExistingError, match=\"new detail\"):",
            "+        handle()",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/runtime.py", "tests/test_runtime.py"],
            "summary": "Adds a focused exception test.",
        }],
        max_files=10,
        max_diff_chars=5000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_validation_rejects_test_docstring_only_change(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    (repo_root / "src" / "runtime.py").write_text(
        "def handle(value: str) -> str:\n"
        "    if not value:\n"
        "        raise RuntimeError(\"old message\")\n"
        "    return value\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_runtime.py").write_text(
        "import pytest\n"
        "\n"
        "from src.runtime import handle\n"
        "\n"
        "def test_error_message() -> None:\n"
        "    \"\"\"Current behavior.\"\"\"\n"
        "    with pytest.raises(RuntimeError, match=\"old message\"):\n"
        "        handle(\"\")\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -1,4 +1,4 @@",
            " def handle(value: str) -> str:",
            "     if not value:",
            "-        raise RuntimeError(\"old message\")",
            "+        raise RuntimeError(\"new message\")",
            "     return value",
            "diff --git a/tests/test_runtime.py b/tests/test_runtime.py",
            "--- a/tests/test_runtime.py",
            "+++ b/tests/test_runtime.py",
            "@@ -3,7 +3,7 @@",
            " from src.runtime import handle",
            " ",
            " def test_error_message() -> None:",
            "-    \"\"\"Current behavior.\"\"\"",
            "+    \"\"\"Updated behavior documentation.\"\"\"",
            "     with pytest.raises(RuntimeError, match=\"old message\"):",
            "         handle(\"\")",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/runtime.py", "tests/test_runtime.py"],
            "summary": "Changes runtime message but only edits test prose.",
        }],
        max_files=10,
        max_diff_chars=5000,
    )

    assert validation["status"] == "failed"
    assert validation["errors"] == [
        "Patch changes test files without adding or modifying executable "
        "test assertions.",
    ]


def test_patch_validation_rejects_invalid_patched_python(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "runtime.py").write_text(
        "def handle() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -1,2 +1,2 @@",
            " def handle() -> None:",
            "-    pass",
            "+    pass    assert True",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/runtime.py"],
            "summary": "Creates invalid Python syntax.",
        }],
        max_files=10,
        max_diff_chars=5000,
    )

    assert validation["status"] == "failed"
    assert validation["errors"] == [
        "Patched Python content is not syntactically valid: "
        "src/runtime.py line 2: invalid syntax."
    ]


def test_patch_validation_rejects_unsupported_response_text_assertion(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    (repo_root / "src" / "runtime.py").write_text(
        "def handle() -> None:\n"
        "    raise RuntimeError(\n"
        "        \"old detail\"\n"
        "    )\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_runtime.py").write_text(
        "def test_response(client) -> None:\n"
        "    response = client.get('/ok')\n"
        "    assert response.status_code == 200\n"
        "    assert response.json() == {'message': 'ok'}\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -1,4 +1,4 @@",
            " def handle() -> None:",
            "     raise RuntimeError(",
            "-        \"old detail\"",
            "+        \"new detail\"",
            "     )",
            "diff --git a/tests/test_runtime.py b/tests/test_runtime.py",
            "--- a/tests/test_runtime.py",
            "+++ b/tests/test_runtime.py",
            "@@ -1,4 +1,5 @@",
            " def test_response(client) -> None:",
            "     response = client.get('/ok')",
            "     assert response.status_code == 200",
            "     assert response.json() == {'message': 'ok'}",
            "+    assert \"new detail\" in response.text",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/runtime.py", "tests/test_runtime.py"],
            "summary": "Checks an error detail in an unchanged success body.",
        }],
        max_files=10,
        max_diff_chars=5000,
    )

    assert validation["status"] == "failed"
    assert validation["errors"] == [
        "Patch adds response text assertions that are not supported by the "
        "visible response body assertions."
    ]


def test_patch_validation_accepts_specific_runtime_exception_handler(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "runtime.py").write_text(
        "def handle() -> str:\n"
        "    return do_work()\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -1,2 +1,5 @@",
            " def handle() -> str:",
            "-    return do_work()",
            "+    try:",
            "+        return do_work()",
            "+    except ValueError as exc:",
            "+        raise RuntimeError(\"work failed\") from exc",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/runtime.py"],
            "summary": "Adds specific wrapper.",
        }],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_validation_rejects_unimported_python_symbol(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    (repo_root / "src" / "errors.py").write_text(
        "class ExistingError(Exception):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "runtime.py").write_text(
        "def handle() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_runtime.py").write_text(
        "import pytest\n"
        "\n"
        "def test_handle() -> None:\n"
        "    handle()\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/errors.py b/src/errors.py",
            "--- a/src/errors.py",
            "+++ b/src/errors.py",
            "@@ -1,2 +1,6 @@",
            " class ExistingError(Exception):",
            "     pass",
            "+class NewError(Exception):",
            "+    pass",
            "+",
            "+",
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -1,2 +1,3 @@",
            " def handle() -> None:",
            "+    raise NewError(\"failed\")",
            "     pass",
            "diff --git a/tests/test_runtime.py b/tests/test_runtime.py",
            "--- a/tests/test_runtime.py",
            "+++ b/tests/test_runtime.py",
            "@@ -2,4 +2,7 @@",
            " ",
            " def test_handle() -> None:",
            "     handle()",
            "+",
            "+def test_new_error() -> None:",
            "+    with pytest.raises(NewError):",
            "+        handle()",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": [
                "src/errors.py",
                "src/runtime.py",
                "tests/test_runtime.py",
            ],
            "summary": "Uses a new symbol without importing it.",
        }],
        max_files=10,
        max_diff_chars=6000,
    )

    assert validation["status"] == "failed"
    assert validation["errors"] == [
        "Patch uses Python symbols without imports or local definitions: "
        "src/runtime.py: NewError; tests/test_runtime.py: NewError.",
    ]


def test_patch_validation_ignores_capitalized_words_inside_strings(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "cli.py").write_text(
        "def build_parser(parser) -> None:\n"
        "    parser.add_argument('input')\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/cli.py b/src/cli.py",
            "--- a/src/cli.py",
            "+++ b/src/cli.py",
            "@@ -1,2 +1,5 @@",
            " def build_parser(parser) -> None:",
            "-    parser.add_argument('input')",
            "+    parser.add_argument(",
            "+        'input',",
            "+        help='Path containing URLs (txt or csv).',",
            "+    )",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "cli",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/cli.py"],
            "summary": "Adds help text.",
        }],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_validation_accepts_imported_python_symbol(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    (repo_root / "src" / "errors.py").write_text(
        "class ExistingError(Exception):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "runtime.py").write_text(
        "def handle() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_runtime.py").write_text(
        "import pytest\n"
        "\n"
        "from src.runtime import handle\n"
        "\n"
        "def test_handle() -> None:\n"
        "    handle()\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/errors.py b/src/errors.py",
            "--- a/src/errors.py",
            "+++ b/src/errors.py",
            "@@ -1,2 +1,6 @@",
            " class ExistingError(Exception):",
            "     pass",
            "+class NewError(Exception):",
            "+    pass",
            "+",
            "+",
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -1,2 +1,5 @@",
            "+from src.errors import NewError",
            "+",
            " def handle() -> None:",
            "+    raise NewError(\"failed\")",
            "     pass",
            "diff --git a/tests/test_runtime.py b/tests/test_runtime.py",
            "--- a/tests/test_runtime.py",
            "+++ b/tests/test_runtime.py",
            "@@ -1,6 +1,10 @@",
            " import pytest",
            " ",
            "+from src.errors import NewError",
            " from src.runtime import handle",
            " ",
            " def test_handle() -> None:",
            "-    handle()",
            "+    with pytest.raises(NewError):",
            "+        handle()",
            "+",
            "+def test_new_error() -> None:",
            "+    with pytest.raises(NewError):",
            "+        handle()",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": [
                "src/errors.py",
                "src/runtime.py",
                "tests/test_runtime.py",
            ],
            "summary": "Imports a new symbol before using it.",
        }],
        max_files=10,
        max_diff_chars=6000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_validation_rejects_inconsistent_local_imports(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    diff_text = "\n".join(
        [
            "diff --git a/cli.py b/cli.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/cli.py",
            "@@ -0,0 +1,6 @@",
            "+from fetcher import fetch_page",
            "+from parser_module import parse_html",
            "+",
            "+",
            "+def main() -> None:",
            "+    parse_html(fetch_page(\"https://example.invalid\"))",
            "diff --git a/fetcher.py b/fetcher.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/fetcher.py",
            "@@ -0,0 +1,3 @@",
            "+def fetch_html(url: str) -> str:",
            "+    return url",
            "diff --git a/parser.py b/parser.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/parser.py",
            "@@ -0,0 +1,3 @@",
            "+def parse_html(html: str) -> dict[str, str]:",
            "+    return {\"html\": html}",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=None,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "project",
            "base": "new_file",
            "diff_text": diff_text,
            "files": ["cli.py", "fetcher.py", "parser.py"],
            "summary": "Adds a small project with inconsistent imports.",
        }],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "failed"
    assert validation["errors"] == [
        "Patch has unresolved local Python imports: "
        "cli.py imports fetch_page from fetcher; "
        "cli.py imports parser_module. Available local Python modules: "
        "cli, fetcher, parser."
    ]


def test_patch_validation_accepts_consistent_new_project_imports(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    diff_text = "\n".join(
        [
            "diff --git a/cli.py b/cli.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/cli.py",
            "@@ -0,0 +1,7 @@",
            "+from fetcher import fetch_html",
            "+from parser import parse_html",
            "+",
            "+",
            "+def main() -> dict[str, str]:",
            "+    result = parse_html(fetch_html(\"https://example.invalid\"))",
            "+    return result",
            "diff --git a/fetcher.py b/fetcher.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/fetcher.py",
            "@@ -0,0 +1,3 @@",
            "+def fetch_html(url: str) -> str:",
            "+    return url",
            "diff --git a/parser.py b/parser.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/parser.py",
            "@@ -0,0 +1,3 @@",
            "+def parse_html(html: str) -> dict[str, str]:",
            "+    return {\"html\": html}",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=None,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "project",
            "base": "new_file",
            "diff_text": diff_text,
            "files": ["cli.py", "fetcher.py", "parser.py"],
            "summary": "Adds a small project with local imports.",
        }],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_validation_rejects_undefined_module_reference(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    diff_text = "\n".join(
        [
            "diff --git a/cli.py b/cli.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/cli.py",
            "@@ -0,0 +1,8 @@",
            "+def read_rows(path: str) -> list[list[str]]:",
            "+    with open(path, newline='', encoding='utf-8') as file_handle:",
            "+        reader = csv.reader(file_handle)",
            "+        rows = list(reader)",
            "+    return rows",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=None,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "project",
            "base": "new_file",
            "diff_text": diff_text,
            "files": ["cli.py"],
            "summary": "Adds code with a missing import.",
        }],
        max_files=10,
        max_diff_chars=2000,
    )

    assert validation["status"] == "failed"
    assert validation["errors"] == [
        "Patch uses Python module references without imports or local "
        "definitions: cli.py: csv.reader."
    ]


def test_patch_validation_accepts_later_defined_module_global_reference(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    repo_root.mkdir()
    (repo_root / "service.py").write_text(
        "def remember(value: str) -> None:\n"
        "    seen_values.add(value)\n"
        "\n"
        "seen_values: set[str] = set()\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/service.py b/service.py",
            "--- a/service.py",
            "+++ b/service.py",
            "@@ -1,4 +1,5 @@",
            "+from collections.abc import Iterable",
            " def remember(value: str) -> None:",
            "     seen_values.add(value)",
            " ",
            " seen_values: set[str] = set()",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "service",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["service.py"],
            "summary": "Adds an import without changing global lookup.",
        }],
        max_files=10,
        max_diff_chars=2000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_validation_accepts_comprehension_target_reference(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    diff_text = "\n".join(
        [
            "diff --git a/reader.py b/reader.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/reader.py",
            "@@ -0,0 +1,5 @@",
            "+def read_urls(lines: list[str]) -> list[str]:",
            "+    urls = [line.strip() for line in lines if line.strip()]",
            "+    return urls",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=None,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "reader",
            "base": "new_file",
            "diff_text": diff_text,
            "files": ["reader.py"],
            "summary": "Adds a reader with a comprehension.",
        }],
        max_files=10,
        max_diff_chars=2000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_validation_rejects_generated_test_failure(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    diff_text = "\n".join(
        [
            "diff --git a/parser.py b/parser.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/parser.py",
            "@@ -0,0 +1,6 @@",
            "+def extract_metadata(html: str) -> dict[str, str]:",
            "+    return {",
            "+        'title': 'Example',",
            "+        'description': 'Body',",
            "+    }",
            "diff --git a/tests/test_parser.py b/tests/test_parser.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/tests/test_parser.py",
            "@@ -0,0 +1,8 @@",
            "+from parser import extract_metadata",
            "+",
            "+",
            "+def test_extract_metadata_includes_url() -> None:",
            "+    result = extract_metadata('<html></html>', url='https://example.invalid')",
            "+",
            "+    assert result['url'] == 'https://example.invalid'",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=None,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "project",
            "base": "new_file",
            "diff_text": diff_text,
            "files": ["parser.py", "tests/test_parser.py"],
            "summary": "Adds inconsistent parser code and tests.",
        }],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "failed"
    assert validation["errors"]
    assert validation["errors"][0].startswith(
        "Patched Python tests fail in isolated validation:"
    )
    assert "unexpected keyword argument 'url'" in validation["errors"][0]


def test_patch_validation_ignores_attribute_calls_and_docstrings(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    repo_root.mkdir()
    diff_text = "\n".join(
        [
            "diff --git a/jsonl_to_csv.py b/jsonl_to_csv.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/jsonl_to_csv.py",
            "@@ -0,0 +1,16 @@",
            "+\"\"\"Convert JSONL (JSON Lines) files to CSV.\"\"\"",
            "+",
            "+import argparse",
            "+import csv",
            "+import sys",
            "+",
            "+",
            "+def main() -> None:",
            "+    parser = argparse.ArgumentParser(",
            "+        description=\"Convert JSONL to CSV.\"",
            "+    )",
            "+    writer = csv.DictWriter(sys.stdout, fieldnames=[\"name\"])",
            "+    writer.writeheader()",
            "+",
            "+",
            "+if __name__ == \"__main__\":",
            "+    main()",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=[{
            "artifact_id": "script",
            "base": "new_file",
            "diff_text": diff_text,
            "files": ["jsonl_to_csv.py"],
            "summary": "Adds a JSONL conversion script.",
        }],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_validation_applies_inside_unrelated_parent_git_repo(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    outer_root = tmp_path / "outer"
    repo_root = outer_root / "write_repo"
    workspace_root = outer_root / "workspace"
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    repo_root.mkdir(parents=True)
    (repo_root / "src").mkdir()
    (repo_root / "src" / "errors.py").write_text(
        "class ExistingError(Exception):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "runtime.py").write_text(
        "def handle() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )
    diff_text = "\n".join(
        [
            "diff --git a/src/runtime.py b/src/runtime.py",
            "--- a/src/runtime.py",
            "+++ b/src/runtime.py",
            "@@ -1,2 +1,5 @@",
            "+from src.errors import ExistingError",
            "+",
            " def handle() -> None:",
            "+    raise ExistingError(\"failed\")",
            "     pass",
            "",
        ]
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=workspace_root,
        patch_artifacts=[{
            "artifact_id": "runtime",
            "base": "repository",
            "diff_text": diff_text,
            "files": ["src/runtime.py"],
            "summary": "Imports a symbol before using it.",
        }],
        max_files=10,
        max_diff_chars=4000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []


def test_patch_operations_compile_to_valid_diff(tmp_path: Path) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_operations import (
        compile_patch_operations,
    )
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = _make_repo(tmp_path)
    workspace_root = tmp_path / "workspace"
    artifacts, created_files, changed_files, errors = compile_patch_operations(
        repo_root=repo_root,
        patch_operations=[
            {
                "operation_id": "insert-constant",
                "kind": "insert_after",
                "path": "src/module.py",
                "anchor": "VALUE = 1\n",
                "content": "EXTRA = 2\n",
                "summary": "Adds one constant.",
            }
        ],
        max_files=10,
        max_diff_chars=4000,
    )

    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=workspace_root,
        patch_artifacts=artifacts,
        max_files=10,
        max_diff_chars=4000,
    )

    assert errors == []
    assert created_files == []
    assert changed_files == [
        {
            "path": "src/module.py",
            "change_type": "modify",
            "summary": "Adds one constant.",
        }
    ]
    assert validation["status"] == "succeeded"
    assert not (repo_root / "src" / "generated.py").exists()


def test_patch_operations_extend_text_insert_to_paragraph_boundary(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_operations import (
        compile_patch_operations,
    )

    repo_root = tmp_path / "write_repo"
    repo_root.mkdir()
    (repo_root / "docs.md").write_text(
        "A wrapped paragraph continues on the next\n"
        "line without a blank separator.\n",
        encoding="utf-8",
    )

    artifacts, created_files, changed_files, errors = compile_patch_operations(
        repo_root=repo_root,
        patch_operations=[
            {
                "operation_id": "bad-doc-insert",
                "kind": "insert_after",
                "path": "docs.md",
                "anchor": "A wrapped paragraph continues on the next\n",
                "content": "\nNew paragraph.\n",
                "summary": "Adds a paragraph.",
            }
        ],
        max_files=10,
        max_diff_chars=4000,
    )

    assert errors == []
    assert created_files == []
    assert changed_files == [
        {
            "path": "docs.md",
            "change_type": "modify",
            "summary": "Adds a paragraph.",
        }
    ]
    assert len(artifacts) == 1
    assert (
        " line without a blank separator.\n"
        "+\n"
        "+New paragraph.\n"
    ) in artifacts[0]["diff_text"]


def test_patch_operations_reject_text_insert_with_partial_line_anchor(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_operations import (
        compile_patch_operations,
    )

    repo_root = tmp_path / "write_repo"
    repo_root.mkdir()
    (repo_root / "docs.md").write_text(
        "FIRST_SETTING=value\n"
        "SECOND_SETTING=value\n",
        encoding="utf-8",
    )

    artifacts, created_files, changed_files, errors = compile_patch_operations(
        repo_root=repo_root,
        patch_operations=[
            {
                "operation_id": "bad-partial-anchor",
                "kind": "insert_after",
                "path": "docs.md",
                "anchor": "FIRST_SETTING",
                "content": "\nNEW_SETTING=value\n",
                "summary": "Adds a setting.",
            }
        ],
        max_files=10,
        max_diff_chars=4000,
    )

    assert artifacts == []
    assert created_files == []
    assert changed_files == []
    assert errors == [
        "Text insertion anchor must include a complete line ending.",
    ]


def test_patch_operations_expands_full_line_anchor_without_newline(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_operations import (
        compile_patch_operations,
    )

    repo_root = tmp_path / "write_repo"
    repo_root.mkdir()
    (repo_root / "settings.py").write_text(
        "VALUE = 1\n"
        "NEXT_VALUE = 2\n",
        encoding="utf-8",
    )

    artifacts, created_files, changed_files, errors = compile_patch_operations(
        repo_root=repo_root,
        patch_operations=[
            {
                "operation_id": "line-anchor",
                "kind": "insert_after",
                "path": "settings.py",
                "anchor": "VALUE = 1",
                "content": "EXTRA_VALUE = 3\n",
                "summary": "Adds a setting.",
            }
        ],
        max_files=10,
        max_diff_chars=4000,
    )

    assert errors == []
    assert created_files == []
    assert changed_files == [
        {
            "path": "settings.py",
            "change_type": "modify",
            "summary": "Adds a setting.",
        }
    ]
    assert len(artifacts) == 1
    assert "+EXTRA_VALUE = 3\n NEXT_VALUE = 2" in artifacts[0]["diff_text"]


def test_patch_operations_reject_python_insert_inside_open_expression(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_operations import (
        compile_patch_operations,
    )

    repo_root = tmp_path / "write_repo"
    repo_root.mkdir()
    (repo_root / "settings.py").write_text(
        "VALUE = build_value(\n"
        "    \"current\",\n"
        ")\n",
        encoding="utf-8",
    )

    artifacts, created_files, changed_files, errors = compile_patch_operations(
        repo_root=repo_root,
        patch_operations=[
            {
                "operation_id": "bad-python-insert",
                "kind": "insert_after",
                "path": "settings.py",
                "anchor": "VALUE = build_value(\n",
                "content": "OTHER_VALUE = 2\n",
                "summary": "Adds a setting.",
            }
        ],
        max_files=10,
        max_diff_chars=4000,
    )

    assert artifacts == []
    assert created_files == []
    assert changed_files == []
    assert errors == [
        "Python insertion anchor ends inside an open expression.",
    ]


def test_patch_operations_anchor_error_names_operation_and_path(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_operations import (
        compile_patch_operations,
    )

    repo_root = tmp_path / "write_repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    artifacts, created_files, changed_files, errors = compile_patch_operations(
        repo_root=repo_root,
        patch_operations=[
            {
                "operation_id": "bad-anchor",
                "kind": "replace",
                "path": "module.py",
                "anchor": "MISSING = 1\n",
                "content": "VALUE = 2\n",
                "summary": "Updates a value.",
            }
        ],
        max_files=10,
        max_diff_chars=4000,
    )

    assert artifacts == []
    assert created_files == []
    assert changed_files == []
    assert errors == [
        "Patch operation bad-anchor for module.py anchor was not found.",
    ]


def test_patch_operations_replace_line_preserves_following_line_boundary(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_operations import (
        compile_patch_operations,
    )
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    repo_root = tmp_path / "write_repo"
    module_dir = repo_root / "src"
    module_dir.mkdir(parents=True)
    (module_dir / "module.py").write_text(
        "class Runtime:\n"
        "    async def get(self, value: str) -> str:\n"
        "        \"\"\"Return a value.\"\"\"\n"
        "        return value\n",
        encoding="utf-8",
    )

    artifacts, created_files, changed_files, errors = compile_patch_operations(
        repo_root=repo_root,
        patch_operations=[
            {
                "operation_id": "replace-signature",
                "kind": "replace",
                "path": "src/module.py",
                "anchor": "    async def get(self, value: str) -> str:",
                "content": (
                    "    async def get(\n"
                    "        self,\n"
                    "        value: str,\n"
                    "        *,\n"
                    "        label: str | None = None,\n"
                    "    ) -> str:"
                ),
                "summary": "Adds a keyword-only argument.",
            }
        ],
        max_files=10,
        max_diff_chars=4000,
    )
    validation = validate_patch_artifacts(
        repo_root=repo_root,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=artifacts,
        max_files=10,
        max_diff_chars=4000,
    )

    assert errors == []
    assert created_files == []
    assert changed_files == [{
        "path": "src/module.py",
        "change_type": "modify",
        "summary": "Adds a keyword-only argument.",
    }]
    assert validation["status"] == "succeeded"
    assert ') -> str:\n         """Return a value.' in artifacts[0]["diff_text"]


def test_programmer_repair_context_names_failed_operations() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        _programmer_repair_context,
    )

    validation = {
        "status": "failed",
        "parsed": True,
        "sandbox_applied": False,
        "errors": [
            "Patched Python tests fail in isolated validation: "
            "RuntimeWarning: coroutine 'call' was never awaited",
            "Patch operation bad-anchor for module.py anchor was not found.",
        ],
        "warnings": [],
        "files": ["module.py", "tests/test_module.py"],
    }
    report = {
        "assignment_id": "runtime",
        "status": "succeeded",
        "files_considered": ["module.py"],
        "facts": [],
        "patch_operations": [
            {
                "operation_id": "bad-anchor",
                "kind": "replace",
                "path": "module.py",
                "anchor": "MISSING = 1\n",
                "content": "VALUE = 2\n",
                "summary": "Updates a value.",
            }
        ],
        "patch_artifacts": [],
        "open_questions": [],
        "created_files": [],
        "changed_files": [],
        "evidence": [],
    }

    context = _programmer_repair_context(
        validation=validation,
        report=report,
    )

    assert context["repair_notes"] == [
        "Repair the shared runtime/test contract so touched tests pass "
        "in the isolated validation environment.",
        "A coroutine was created without being awaited. Make the test async "
        "and await coroutine calls directly.",
        "Rewrite named structured operations with exact anchors from "
        "file_context, or merge dependent edits into one operation anchored "
        "to original current text.",
    ]
    assert context["failed_operations"] == [
        {
            "failure": "anchor_not_found",
            "operation_id": "bad-anchor",
            "path": "module.py",
            "kind": "replace",
            "summary": "Updates a value.",
            "anchor_excerpt": "MISSING = 1\n",
            "content_excerpt": "VALUE = 2\n",
        }
    ]


def test_programmer_repair_context_names_broad_exception_operation() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        _programmer_repair_context,
    )

    validation = {
        "status": "failed",
        "parsed": True,
        "sandbox_applied": False,
        "errors": [
            "Patch adds broad runtime exception wrapping: src/service.py "
            "adds `except Exception:`. Remove the new try/except and modify "
            "the smallest existing error branch, or use a specific observed "
            "exception type while preserving original exception propagation.",
        ],
        "warnings": [],
        "files": ["src/service.py"],
    }
    report = {
        "assignment_id": "service",
        "status": "succeeded",
        "files_considered": ["src/service.py"],
        "facts": [],
        "patch_operations": [{
            "operation_id": "optional-status",
            "kind": "replace",
            "path": "src/service.py",
            "anchor": "def status() -> dict:\n    return {}\n",
            "content": (
                "def status() -> dict:\n"
                "    try:\n"
                "        value = collect_status()\n"
                "    except Exception:\n"
                "        value = {}\n"
                "    return value\n"
            ),
            "summary": "Adds optional status collection.",
        }],
        "patch_artifacts": [],
        "open_questions": [],
        "created_files": [],
        "changed_files": [],
        "evidence": [],
    }

    context = _programmer_repair_context(
        validation=validation,
        report=report,
    )

    assert context["repair_notes"] == [
        "Remove added broad runtime exception wrapping from the named "
        "operation. Prefer direct existing control flow or a specific "
        "observed exception type; do not add new lines beginning with `try:`, "
        "`except Exception`, or `except BaseException`.",
    ]
    assert context["failed_operations"][0]["failure"] == (
        "broad_runtime_exception"
    )
    assert context["failed_operations"][0]["operation_id"] == (
        "optional-status"
    )


def test_validation_replan_repair_context_filters_to_assignment_scope() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        _repair_context_for_assignment,
        _validation_replan_repair_context,
    )

    validation = {
        "status": "failed",
        "parsed": True,
        "sandbox_applied": False,
        "errors": [
            "Patched Python content is not syntactically valid: "
            "src/runtime.py line 12: invalid syntax.",
        ],
        "warnings": [],
        "files": ["src/runtime.py", "tests/test_runtime.py"],
    }
    runtime_report = {
        "assignment_id": "runtime",
        "status": "succeeded",
        "files_considered": ["src/runtime.py"],
        "facts": [],
        "patch_operations": [{
            "operation_id": "runtime-change",
            "kind": "replace",
            "path": "src/runtime.py",
            "anchor": "VALUE = 1\n",
            "content": "VALUE = 2\n",
            "summary": "Updates runtime.",
        }],
        "patch_artifacts": [],
        "open_questions": [],
        "created_files": [],
        "changed_files": [{
            "path": "src/runtime.py",
            "change_type": "modify",
            "summary": "Updates runtime.",
        }],
        "evidence": [],
    }
    test_report = {
        "assignment_id": "tests",
        "status": "succeeded",
        "files_considered": ["tests/test_runtime.py"],
        "facts": [],
        "patch_operations": [{
            "operation_id": "test-change",
            "kind": "replace",
            "path": "tests/test_runtime.py",
            "anchor": "assert VALUE == 1\n",
            "content": "assert VALUE == 2\n",
            "summary": "Updates test.",
        }],
        "patch_artifacts": [],
        "open_questions": [],
        "created_files": [],
        "changed_files": [{
            "path": "tests/test_runtime.py",
            "change_type": "modify",
            "summary": "Updates test.",
        }],
        "evidence": [],
    }
    repair_context = _validation_replan_repair_context(
        validation=validation,
        programmer_reports=[runtime_report, test_report],
    )
    assignment = {
        "assignment_id": "tests",
        "role": "test programmer",
        "scope": {
            "kind": "file",
            "values": ["tests/test_runtime.py"],
        },
        "owned_paths": ["tests/test_runtime.py"],
        "read_only_paths": ["src/runtime.py"],
        "questions": ["Update tests."],
        "required_slots": ["test coverage"],
    }

    scoped_context = _repair_context_for_assignment(
        repair_context,
        assignment=assignment,
    )

    assert scoped_context is not None
    assert scoped_context["previous_candidate_files"] == [
        "src/runtime.py",
        "tests/test_runtime.py",
    ]
    assert [
        report["assignment_id"]
        for report in scoped_context["previous_reports"]
    ] == ["tests"]
    assert scoped_context["omitted_previous_report_count"] == 1


def test_repair_report_payload_compacts_generated_operations() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        MAX_REPAIR_EXCERPT_CHARS,
        MAX_REPAIR_REPORT_FACTS,
        MAX_REPAIR_REPORT_ITEMS,
        MAX_REPAIR_REPORT_OPERATIONS,
        _repair_report_payload,
    )

    long_content = "VALUE = 1\n" * 500
    operations = [
        {
            "operation_id": f"create-large-test-{index}",
            "kind": "create_file",
            "path": f"tests/test_generated_{index}.py",
            "content": long_content,
            "summary": "Adds generated tests.",
        }
        for index in range(MAX_REPAIR_REPORT_OPERATIONS + 2)
    ]
    report = {
        "assignment_id": "tests",
        "status": "succeeded",
        "files_considered": [
            f"tests/test_generated_{index}.py"
            for index in range(MAX_REPAIR_REPORT_ITEMS + 2)
        ],
        "facts": [
            {
                "kind": "implementation_note",
                "summary": f"fact {index}",
                "evidence_refs": [],
            }
            for index in range(MAX_REPAIR_REPORT_FACTS + 2)
        ],
        "patch_operations": operations,
        "open_questions": [
            f"question {index}"
            for index in range(MAX_REPAIR_REPORT_ITEMS + 2)
        ],
        "created_files": [
            {
                "path": f"generated_{index}.py",
                "summary": "Created file.",
            }
            for index in range(MAX_REPAIR_REPORT_ITEMS + 2)
        ],
        "changed_files": [
            {
                "path": f"changed_{index}.py",
                "summary": "Changed file.",
            }
            for index in range(MAX_REPAIR_REPORT_ITEMS + 2)
        ],
        "evidence": [],
    }

    payload = _repair_report_payload(report)

    assert len(payload["files_considered"]) == MAX_REPAIR_REPORT_ITEMS
    assert len(payload["facts"]) == MAX_REPAIR_REPORT_FACTS
    assert len(payload["patch_operations"]) == MAX_REPAIR_REPORT_OPERATIONS
    assert len(payload["open_questions"]) == MAX_REPAIR_REPORT_ITEMS
    assert len(payload["patch_operations"][0]["content"]) <= (
        MAX_REPAIR_EXCERPT_CHARS
    )


def test_patch_operations_reject_unbalanced_markdown_fences(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_operations import (
        compile_patch_operations,
    )

    repo_root = tmp_path / "write_repo"
    repo_root.mkdir()
    (repo_root / "docs.md").write_text(
        "```env\n"
        "EXISTING=value\n"
        "```\n",
        encoding="utf-8",
    )

    artifacts, created_files, changed_files, errors = compile_patch_operations(
        repo_root=repo_root,
        patch_operations=[
            {
                "operation_id": "bad-fence",
                "kind": "insert_after",
                "path": "docs.md",
                "anchor": "EXISTING=value\n",
                "content": "NEW=value\n```\n",
                "summary": "Adds a setting.",
            }
        ],
        max_files=10,
        max_diff_chars=4000,
    )

    assert artifacts == []
    assert created_files == []
    assert changed_files == [
        {
            "path": "docs.md",
            "change_type": "modify",
            "summary": "Adds a setting.",
        }
    ]
    assert errors == [
        "Markdown code fences are unbalanced after patch operations.",
    ]


def test_writing_workspace_persists_and_invalidates_by_base_identity(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.workspace import (
        prepare_writing_workspace,
    )

    workspace_root = tmp_path / "workspace"

    first = prepare_writing_workspace(
        workspace_root=workspace_root,
        session_id="session-one",
        base_identity="base-a",
        mode="create_new_project",
    )
    second = prepare_writing_workspace(
        workspace_root=workspace_root,
        session_id="session-one",
        base_identity="base-a",
        mode="create_new_project",
    )
    third = prepare_writing_workspace(
        workspace_root=workspace_root,
        session_id="session-one",
        base_identity="base-b",
        mode="create_new_project",
    )

    assert first["session_id"] == "session-one"
    assert second["session_id"] == "session-one"
    assert first["public_handle"] == second["public_handle"]
    assert third["invalidated_previous"] is True
    assert "workspace" not in repr(first)
    assert str(workspace_root) not in repr(first)


def test_code_writing_llm_routes_reuse_pm_for_synthesis() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.llm_config import (
        resolve_code_writing_llm_settings,
    )

    settings = resolve_code_writing_llm_settings(
        {
            "CODING_AGENT_PM_LLM_BASE_URL": "http://pm.example/v1",
            "CODING_AGENT_PM_LLM_API_KEY": "pm-key",
            "CODING_AGENT_PM_LLM_MODEL": "pm-model",
            "CODING_AGENT_PROGRAMMER_LLM_BASE_URL": "http://worker.example/v1",
            "CODING_AGENT_PROGRAMMER_LLM_API_KEY": "worker-key",
            "CODING_AGENT_PROGRAMMER_LLM_MODEL": "worker-model",
        }
    )

    assert settings["pm"]["route_name"] == "CODING_AGENT_PM_LLM"
    assert settings["programmer"]["route_name"] == "CODING_AGENT_PROGRAMMER_LLM"
    assert settings["synthesis"] == settings["pm"]


def test_prompt_budget_uses_ceil_char_count_over_four() -> None:
    from kazusa_ai_chatbot.coding_agent.context_budget import (
        estimate_input_tokens,
        prompt_budget_metadata,
    )

    assert estimate_input_tokens(0) == 0
    assert estimate_input_tokens(1) == 1
    assert estimate_input_tokens(4) == 1
    assert estimate_input_tokens(5) == 2
    budget = prompt_budget_metadata(
        system_prompt="x" * 20_000,
        payload_text="y" * 31_000,
        target_input_tokens=34_000,
    )

    assert budget["estimated_input_tokens"] < 42_000
    assert budget["over_hard_token_cap"] is False
    assert budget["over_hard_char_cap"] is True
    assert budget["over_hard_cap"] is True


async def test_writing_pm_blocks_prompt_before_invoke_when_over_budget(
    monkeypatch,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import product_manager

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("PM LLM should not be invoked over budget.")

    monkeypatch.setattr(product_manager._writing_pm_llm, "ainvoke", fail_if_called)

    trace: dict[str, object] = {}
    decision = await product_manager.decide_writing_work(
        {
            "question": "Update one behavior.",
            "mode": "edit_existing_repository",
            "repository_summary": {"repo": "fixture"},
            "reading_reports": [{
                "answer_text": "x" * 180_000,
                "evidence_rows": [],
                "limitations": [],
            }],
            "owner_candidates": [],
            "previous_writing_reports": [],
        },
        trace=trace,
    )

    context_budget = trace["context_budget"]

    assert decision["status"] == "overloaded"
    assert trace["blocked_before_invoke"] is True
    assert context_budget["over_hard_cap"] is True
    assert context_budget["estimated_input_tokens"] > 42_000


async def test_writing_pm_retries_empty_json_response(monkeypatch) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import product_manager

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.content = content

    calls: list[object] = []

    async def fake_ainvoke(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return FakeResponse("")
        return FakeResponse(
            '{"status": "need_programmers", '
            '"mode": "create_new_project", '
            '"intent": "new_project_cli_or_tool", '
            '"file_demands": [{'
            '"demand_id": "main", '
            '"role": "script programmer", '
            '"purpose": "Create the script.", '
            '"file_kind": "new", '
            '"preferred_path": "tool.py", '
            '"questions": ["Create the script."], '
            '"required_slots": ["script artifact"]'
            '}], '
            '"missing_slots": [], '
            '"reading_requests": [], '
            '"external_evidence_requests": []}'
        )

    monkeypatch.setattr(product_manager._writing_pm_llm, "ainvoke", fake_ainvoke)

    trace: dict[str, object] = {}
    decision = await product_manager.decide_writing_work(
        {
            "question": "Create one small script.",
            "mode": "create_new_project",
            "repository_summary": None,
            "reading_reports": [],
            "owner_candidates": [],
            "previous_writing_reports": [],
        },
        trace=trace,
    )

    assert decision["status"] == "need_programmers"
    assert len(calls) == 2
    assert len(trace["attempts"]) == 2
    assert trace["attempts"][0]["parsed_output"] == {}
    assert trace["attempts"][1]["parsed_output"]["status"] == "need_programmers"
    assert (
        calls[1][0].content
        == product_manager.WRITING_PM_COMPACT_RECOVERY_PROMPT
    )
    assert trace["attempts"][1]["prompt_kind"] == "compact_recovery"


async def test_writing_pm_uses_compact_recovery_after_empty_retries(
    monkeypatch,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import product_manager

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.content = content

    calls: list[object] = []

    async def fake_ainvoke(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return FakeResponse("not json")
        if len(calls) == 2:
            return FakeResponse("")
        return FakeResponse(
            '{"status": "need_programmers", '
            '"mode": "edit_existing_repository", '
            '"intent": "bounded existing-source patch", '
            '"file_demands": [{'
            '"demand_id": "runtime", '
            '"role": "runtime programmer", '
            '"purpose": "Update the runtime owner.", '
            '"file_kind": "existing", '
            '"preferred_path": "src/runtime.py", '
            '"questions": ["Update the runtime owner."], '
            '"required_slots": ["runtime behavior"]'
            '}], '
            '"missing_slots": [], '
            '"reading_requests": [], '
            '"external_evidence_requests": []}'
        )

    monkeypatch.setattr(product_manager._writing_pm_llm, "ainvoke", fake_ainvoke)

    trace: dict[str, object] = {}
    decision = await product_manager.decide_writing_work(
        {
            "question": "Update one existing behavior.",
            "mode": "edit_existing_repository",
            "repository_summary": {"repo": "fixture"},
            "reading_reports": [{"status": "sufficient"}],
            "supervisor_evidence_state": {
                "remaining_reading_attempts": 0,
            },
            "owner_candidates": [],
            "previous_writing_reports": [],
        },
        trace=trace,
    )

    assert decision["status"] == "need_programmers"
    assert len(calls) == 3
    assert len(trace["attempts"]) == 3
    assert trace["attempts"][0]["parsed_output"] == {}
    assert trace["attempts"][1]["parsed_output"] == {}
    assert trace["attempts"][2]["parsed_output"]["status"] == "need_programmers"
    assert trace["attempts"][1]["prompt_kind"] == "full_json_retry"
    assert calls[1][0].content == product_manager.WRITING_PM_PROMPT
    assert (
        calls[2][0].content
        == product_manager.WRITING_PM_COMPACT_RECOVERY_PROMPT
    )


async def test_writing_pm_records_timeout_and_recovers_compactly(
    monkeypatch,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import product_manager

    async def fake_ainvoke(messages, **kwargs):
        await asyncio.sleep(1)
        raise AssertionError("wait_for should timeout before this returns")

    monkeypatch.setattr(product_manager._writing_pm_llm, "ainvoke", fake_ainvoke)
    monkeypatch.setattr(product_manager, "PM_LLM_CALL_TIMEOUT_SECONDS", 0.01)

    trace: dict[str, object] = {}
    decision = await product_manager.decide_writing_work(
        {
            "question": "Update one existing behavior.",
            "mode": "edit_existing_repository",
            "repository_summary": {"repo": "fixture"},
            "reading_reports": [{"status": "sufficient"}],
            "supervisor_evidence_state": {
                "remaining_reading_attempts": 0,
            },
            "owner_candidates": [],
            "previous_writing_reports": [],
        },
        trace=trace,
    )

    assert decision["status"] == "needs_user_input"
    assert len(trace["attempts"]) == 2
    assert trace["attempts"][0]["timed_out"] is True
    assert trace["attempts"][0]["prompt_kind"] == "full"
    assert trace["attempts"][1]["timed_out"] is True
    assert trace["attempts"][1]["prompt_kind"] == "compact_recovery"


async def test_programmer_retries_malformed_json_response(monkeypatch) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import programmer

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.content = content

    calls: list[object] = []

    async def fake_ainvoke(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return FakeResponse("not json")
        return FakeResponse(
            '{"status": "succeeded", '
            '"files_considered": ["tool.py"], '
            '"facts": [], '
            '"patch_operations": [{'
            '"operation_id": "create-tool", '
            '"kind": "create_file", '
            '"path": "tool.py", '
            '"content": "print(\\"ok\\")\\n", '
            '"summary": "Create a tool."'
            '}], '
            '"patch_artifacts": [], '
            '"created_files": [], '
            '"changed_files": [], '
            '"open_questions": []}'
        )

    monkeypatch.setattr(programmer._programmer_llm, "ainvoke", fake_ainvoke)

    trace: dict[str, object] = {}
    report = await programmer.run_programmer_assignment(
        question="Create one small script.",
        mode="create_new_project",
        assignment={
            "assignment_id": "main",
            "role": "script programmer",
            "scope": {
                "kind": "file",
                "values": ["tool.py"],
            },
            "questions": ["Create the script."],
            "required_slots": ["script artifact"],
        },
        repository_summary=None,
        reading_evidence=[],
        external_evidence=[],
        trace=trace,
    )

    assert report["status"] == "succeeded"
    assert len(calls) == 2
    assert len(trace["attempts"]) == 2
    assert trace["attempts"][0]["parsed_output"] == {}
    assert trace["attempts"][1]["parsed_output"]["status"] == "succeeded"


async def test_programmer_records_timeout_as_blocked(monkeypatch) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import programmer

    async def fake_ainvoke(messages, **kwargs):
        await asyncio.sleep(1)
        raise AssertionError("wait_for should timeout before this returns")

    monkeypatch.setattr(programmer._programmer_llm, "ainvoke", fake_ainvoke)
    monkeypatch.setattr(
        programmer,
        "PROGRAMMER_LLM_CALL_TIMEOUT_SECONDS",
        0.01,
    )

    trace: dict[str, object] = {}
    report = await programmer.run_programmer_assignment(
        question="Create one small artifact.",
        mode="create_new_project",
        assignment={
            "assignment_id": "main",
            "role": "script programmer",
            "scope": {
                "kind": "file",
                "values": ["tool.py"],
            },
            "questions": ["Create the script."],
            "required_slots": ["script artifact"],
        },
        repository_summary=None,
        reading_evidence=[],
        external_evidence=[],
        trace=trace,
    )

    assert report["status"] == "blocked"
    assert len(trace["attempts"]) == 1
    assert trace["attempts"][0]["timed_out"] is True


async def test_writing_supervisor_repairs_pm_contract_before_programmer_dispatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    first_decision = {
        "status": "need_programmers",
        "mode": "create_new_project",
        "intent": "create small tool",
        "file_demands": [{
            "demand_id": "tool",
            "role": "tool programmer",
            "purpose": "Create one tool module.",
            "file_kind": "new",
            "preferred_path": "../tool.py",
            "preferred_name": "tool.py",
            "placement_hint": "src",
            "related_paths": [],
            "read_only_paths": [],
            "interface_contract": {
                "component": "tool module",
                "inputs": ["function call"],
                "outputs": ["integer result"],
                "invariants": ["module stays importable"],
            },
            "integration_contract": {
                "provides_to_pm": ["tool module content"],
                "consumes_from": [],
            },
            "change_goal": "Create one tool module.",
            "questions": ["Create the tool module."],
            "required_slots": ["tool module artifact"],
            "validation_expectations": ["tool module imports"],
            "forbidden_paths": [],
        }],
        "assignments": [],
        "missing_slots": [],
        "reading_requests": [],
        "external_evidence_requests": [],
    }
    repaired_decision = {
        "status": "need_programmers",
        "mode": "create_new_project",
        "intent": "create small tool",
        "file_demands": [{
            "demand_id": "tool",
            "role": "tool programmer",
            "purpose": "Create one tool module.",
            "file_kind": "new",
            "preferred_path": "src/tool.py",
            "preferred_name": "tool.py",
            "placement_hint": "src",
            "related_paths": [],
            "read_only_paths": [],
            "interface_contract": {
                "component": "tool module",
                "inputs": ["function call"],
                "outputs": ["integer result"],
                "invariants": ["module stays importable"],
            },
            "integration_contract": {
                "provides_to_pm": ["tool module content"],
                "consumes_from": [],
            },
            "change_goal": "Create one tool module.",
            "questions": ["Create the tool module."],
            "required_slots": ["tool module artifact"],
            "validation_expectations": ["tool module imports"],
            "forbidden_paths": [],
        }],
        "assignments": [],
        "missing_slots": [],
        "reading_requests": [],
        "external_evidence_requests": [],
    }
    pm_calls: list[dict[str, object]] = []
    programmer_calls: list[str] = []

    async def fake_decide_writing_work(pm_input, trace=None):
        pm_calls.append(pm_input)
        if len(pm_calls) == 1:
            return first_decision
        return repaired_decision

    async def fake_run_programmer_assignment(**kwargs):
        assignment = kwargs["assignment"]
        programmer_calls.append(assignment["assignment_id"])
        report = {
            "assignment_id": assignment["assignment_id"],
            "status": "succeeded",
            "files_considered": ["src/tool.py"],
            "facts": [],
            "patch_operations": [{
                "operation_id": "create-tool",
                "kind": "create_file",
                "path": "src/tool.py",
                "content": "def value() -> int:\n    return 42\n",
                "summary": "Creates the tool module.",
            }],
            "open_questions": [],
            "created_files": [{
                "path": "src/tool.py",
                "role": "tool module",
            }],
            "changed_files": [],
            "evidence": [],
        }
        return report

    async def fake_synthesize_patch_proposal(**kwargs):
        answer = "Proposed tool module."
        result = answer, kwargs["limitations"]
        return result

    monkeypatch.setattr(supervisor, "decide_writing_work", fake_decide_writing_work)
    monkeypatch.setattr(
        supervisor,
        "run_programmer_assignment",
        fake_run_programmer_assignment,
    )
    monkeypatch.setattr(
        supervisor,
        "synthesize_patch_proposal",
        fake_synthesize_patch_proposal,
    )

    trace: dict[str, object] = {}
    result = await supervisor.run_writing_supervisor(
        {
            "question": "Create one small tool module.",
            "mode_hint": "create_new_project",
            "repository": None,
            "source_scope": None,
            "reading_result": None,
            "workspace_root": str(tmp_path / "workspace"),
            "max_artifact_chars": 8000,
        },
        trace=trace,
    )

    assert result["status"] == "succeeded", result
    assert len(pm_calls) == 2
    assert "file_resolution_feedback" in pm_calls[1]
    feedback = pm_calls[1]["file_resolution_feedback"]
    assert isinstance(feedback, dict)
    assert feedback["status"] == "repair_required"
    assert programmer_calls == ["tool"]
    assert trace["pm_initial_contract_file_resolution_1"]["status"] == (
        "repair_required"
    )
    assert trace["pm_initial_contract_file_resolution_2"]["status"] == "accepted"
    assert trace["pm_initial_contract_after_file_resolution_2"]["status"] == (
        "accepted"
    )


async def test_writing_supervisor_replans_after_validation_exposes_bad_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    repo_root = tmp_path / "edit_repo"
    (repo_root / "src").mkdir(parents=True)
    repo_file = repo_root / "src" / "runtime.py"
    repo_file.write_text(
        "def record(label: str) -> list[str]:\n"
        "    return []\n",
        encoding="utf-8",
    )
    _run_git(["init"], repo_root)
    _run_git(["config", "user.email", "test@example.com"], repo_root)
    _run_git(["config", "user.name", "Test User"], repo_root)
    _run_git(["add", "."], repo_root)
    _run_git(["commit", "-m", "initial"], repo_root)

    repository = {
        "provider": "github",
        "owner": "fixture",
        "repo": "instrumented",
        "source_url": "https://github.com/fixture/instrumented",
        "requested_ref": None,
        "resolved_ref": "main",
        "current_commit": "d" * 40,
        "default_branch": "main",
        "local_root": str(repo_root),
        "storage_kind": "existing_local_checkout",
        "managed_checkout": False,
        "workspace_root": str(tmp_path / "workspace"),
        "cache_key": "fixture-instrumented",
        "dirty_state": "clean",
    }
    source_scope = {
        "kind": "repository",
        "repo_relative_path": None,
        "source_url": "local://fixture/instrumented",
        "requested_ref": None,
        "interpretation": "fixture repository",
    }
    reading_result = {
        "status": "succeeded",
        "answer_text": "Runtime evidence is available.",
        "evidence": [{
            "path": "src/runtime.py",
            "line_start": 1,
            "line_end": 2,
            "symbol_or_topic": "runtime function",
            "excerpt": repo_file.read_text(encoding="utf-8"),
            "reason": "Shows the current runtime interface.",
        }],
        "limitations": [],
        "trace_summary": ["reading:evidence"],
    }
    first_decision = {
        "status": "need_programmers",
        "mode": "edit_existing_repository",
        "intent": "add bounded instrumentation",
        "file_demands": [
            {
                "demand_id": "runtime",
                "role": "runtime programmer",
                "purpose": "Update the runtime record function.",
                "file_kind": "existing",
                "preferred_path": "src/runtime.py",
                "preferred_name": "runtime.py",
                "placement_hint": "src",
                "related_paths": ["src/runtime.py"],
                "read_only_paths": [],
                "interface_contract": {
                    "component": "runtime record function",
                    "inputs": ["label string"],
                    "outputs": ["recorded labels"],
                    "invariants": ["runtime remains importable"],
                },
                "integration_contract": {
                    "provides_to_pm": ["runtime implementation"],
                    "consumes_from": [],
                },
                "change_goal": "Add one runtime observation.",
                "questions": ["Add one runtime observation."],
                "required_slots": ["runtime records only one observed label"],
                "validation_expectations": ["runtime function imports"],
                "forbidden_paths": [],
            },
            {
                "demand_id": "tests",
                "role": "test programmer",
                "purpose": "Add focused runtime tests.",
                "file_kind": "new",
                "preferred_path": "tests/test_runtime.py",
                "preferred_name": "test_runtime.py",
                "placement_hint": "tests",
                "related_paths": ["src/runtime.py"],
                "read_only_paths": ["src/runtime.py"],
                "interface_contract": {
                    "component": "runtime tests",
                    "inputs": ["runtime record function"],
                    "outputs": ["pytest coverage"],
                    "invariants": ["tests use the runtime API"],
                },
                "integration_contract": {
                    "provides_to_pm": ["test coverage"],
                    "consumes_from": ["runtime implementation"],
                },
                "change_goal": "Add coverage for runtime labels.",
                "questions": ["Add coverage for both labels."],
                "required_slots": ["tests assert both observed labels"],
                "validation_expectations": ["pytest covers both labels"],
                "forbidden_paths": [],
            },
        ],
        "assignments": [],
        "missing_slots": [],
        "reading_requests": [],
        "external_evidence_requests": [],
    }
    second_decision = {
        "status": "need_programmers",
        "mode": "edit_existing_repository",
        "intent": "repair shared instrumentation interface",
        "file_demands": [
            {
                "demand_id": "runtime-repair",
                "role": "runtime repair programmer",
                "purpose": "Repair the runtime record function.",
                "file_kind": "existing",
                "preferred_path": "src/runtime.py",
                "preferred_name": "runtime.py",
                "placement_hint": "src",
                "related_paths": ["src/runtime.py"],
                "read_only_paths": [],
                "interface_contract": {
                    "component": "runtime record contract",
                    "inputs": ["label string"],
                    "outputs": ["recorded labels"],
                    "invariants": ["runtime exposes the shared contract"],
                },
                "integration_contract": {
                    "provides_to_pm": ["runtime implementation"],
                    "consumes_from": [],
                },
                "change_goal": "Repair runtime behavior.",
                "questions": ["Make runtime return both observed labels."],
                "required_slots": ["runtime records both labels"],
                "validation_expectations": ["runtime supports the tests"],
                "forbidden_paths": [],
            },
            {
                "demand_id": "tests-repair",
                "role": "test repair programmer",
                "purpose": "Repair focused runtime tests.",
                "file_kind": "new",
                "preferred_path": "tests/test_runtime.py",
                "preferred_name": "test_runtime.py",
                "placement_hint": "tests",
                "related_paths": ["src/runtime.py"],
                "read_only_paths": ["src/runtime.py"],
                "interface_contract": {
                    "component": "runtime tests",
                    "inputs": ["runtime record function"],
                    "outputs": ["pytest coverage"],
                    "invariants": ["tests use the shared runtime contract"],
                },
                "integration_contract": {
                    "provides_to_pm": ["test coverage"],
                    "consumes_from": ["runtime implementation"],
                },
                "change_goal": "Repair runtime tests.",
                "questions": ["Assert the shared runtime contract."],
                "required_slots": ["tests assert both observed labels"],
                "validation_expectations": ["tests match runtime behavior"],
                "forbidden_paths": [],
            },
        ],
        "assignments": [],
        "missing_slots": [],
        "reading_requests": [],
        "external_evidence_requests": [],
    }
    pm_calls: list[dict[str, object]] = []
    programmer_calls: list[tuple[str, bool]] = []

    async def fake_decide_writing_work(pm_input, trace=None):
        pm_calls.append(pm_input)
        if len(pm_calls) == 1:
            return first_decision
        return second_decision

    def runtime_bad_report(assignment):
        return {
            "assignment_id": assignment["assignment_id"],
            "status": "succeeded",
            "files_considered": ["src/runtime.py"],
            "facts": [],
            "patch_operations": [{
                "operation_id": "runtime-one-label",
                "kind": "replace",
                "path": "src/runtime.py",
                "anchor": "def record(label: str) -> list[str]:\n"
                "    return []\n",
                "content": "def record(label: str) -> list[str]:\n"
                "    if label == \"hit\":\n"
                "        return [\"hit\"]\n"
                "    return []\n",
                "summary": "Records one label.",
            }],
            "patch_artifacts": [],
            "open_questions": [],
            "created_files": [],
            "changed_files": [{
                "path": "src/runtime.py",
                "change_type": "modify",
                "summary": "Records one label.",
            }],
            "evidence": reading_result["evidence"],
        }

    def tests_bad_report(assignment):
        return {
            "assignment_id": assignment["assignment_id"],
            "status": "succeeded",
            "files_considered": ["tests/test_runtime.py"],
            "facts": [],
            "patch_operations": [{
                "operation_id": "test-two-labels",
                "kind": "create_file",
                "path": "tests/test_runtime.py",
                "content": "from src.runtime import record\n"
                "\n"
                "\n"
                "def test_record_reports_both_labels() -> None:\n"
                "    assert record(\"hit\") == [\"hit\"]\n"
                "    assert record(\"miss\") == [\"miss\"]\n",
                "summary": "Adds mismatched coverage.",
            }],
            "patch_artifacts": [],
            "open_questions": [],
            "created_files": [{
                "path": "tests/test_runtime.py",
                "role": "test module",
            }],
            "changed_files": [{
                "path": "tests/test_runtime.py",
                "change_type": "add",
                "summary": "Adds mismatched coverage.",
            }],
            "evidence": reading_result["evidence"],
        }

    def runtime_good_report(assignment):
        return {
            "assignment_id": assignment["assignment_id"],
            "status": "succeeded",
            "files_considered": ["src/runtime.py"],
            "facts": [],
            "patch_operations": [{
                "operation_id": "runtime-two-labels",
                "kind": "replace",
                "path": "src/runtime.py",
                "anchor": "def record(label: str) -> list[str]:\n"
                "    return []\n",
                "content": "def record(label: str) -> list[str]:\n"
                "    if label in {\"hit\", \"miss\"}:\n"
                "        return [label]\n"
                "    return []\n",
                "summary": "Records both labels.",
            }],
            "patch_artifacts": [],
            "open_questions": [],
            "created_files": [],
            "changed_files": [{
                "path": "src/runtime.py",
                "change_type": "modify",
                "summary": "Records both labels.",
            }],
            "evidence": reading_result["evidence"],
        }

    def tests_good_report(assignment):
        return {
            "assignment_id": assignment["assignment_id"],
            "status": "succeeded",
            "files_considered": ["tests/test_runtime.py"],
            "facts": [],
            "patch_operations": [{
                "operation_id": "test-two-labels",
                "kind": "create_file",
                "path": "tests/test_runtime.py",
                "content": "from src.runtime import record\n"
                "\n"
                "\n"
                "def test_record_reports_both_labels() -> None:\n"
                "    assert record(\"hit\") == [\"hit\"]\n"
                "    assert record(\"miss\") == [\"miss\"]\n",
                "summary": "Adds matching coverage.",
            }],
            "patch_artifacts": [],
            "open_questions": [],
            "created_files": [{
                "path": "tests/test_runtime.py",
                "role": "test module",
            }],
            "changed_files": [{
                "path": "tests/test_runtime.py",
                "change_type": "add",
                "summary": "Adds matching coverage.",
            }],
            "evidence": reading_result["evidence"],
        }

    async def fake_run_programmer_assignment(**kwargs):
        assignment = kwargs["assignment"]
        programmer_calls.append((
            assignment["assignment_id"],
            kwargs.get("repair_context") is not None,
        ))
        if assignment["assignment_id"] == "runtime":
            return runtime_bad_report(assignment)
        if assignment["assignment_id"] == "tests":
            return tests_bad_report(assignment)
        if assignment["assignment_id"] == "runtime-repair":
            return runtime_good_report(assignment)
        if assignment["assignment_id"] == "tests-repair":
            return tests_good_report(assignment)
        raise AssertionError(
            f"unexpected assignment {assignment['assignment_id']}"
        )

    async def fake_synthesize_patch_proposal(**kwargs):
        return "Validated proposal.", kwargs["limitations"]

    monkeypatch.setattr(supervisor, "decide_writing_work", fake_decide_writing_work)
    monkeypatch.setattr(
        supervisor,
        "run_programmer_assignment",
        fake_run_programmer_assignment,
    )
    monkeypatch.setattr(
        supervisor,
        "synthesize_patch_proposal",
        fake_synthesize_patch_proposal,
    )

    trace: dict[str, object] = {}
    result = await supervisor.run_writing_supervisor(
        {
            "question": "Add bounded instrumentation with matching tests.",
            "mode_hint": "edit_existing_repository",
            "repository": repository,
            "source_scope": source_scope,
            "reading_result": reading_result,
            "workspace_root": str(tmp_path / "workspace"),
            "max_artifact_chars": 8000,
        },
        trace=trace,
    )

    assert result["status"] == "succeeded", result
    assert len(pm_calls) == 2
    assert "validation_feedback" in pm_calls[1]
    assert pm_calls[1]["validation_feedback"]["status"] == "failed"
    repair_calls = [call for call in programmer_calls if call[1]]
    assert repair_calls == [
        ("runtime", True),
        ("tests", True),
        ("runtime-repair", True),
        ("tests-repair", True),
    ]
    assert any(
        item.startswith("writing_pm:validation_replan")
        for item in result["trace_summary"]
    )
    assert result["validation"]["status"] == "succeeded"
    assert {artifact["files"][0] for artifact in result["patch_artifacts"]} == {
        "src/runtime.py",
        "tests/test_runtime.py",
    }


def test_validation_replan_rejects_subset_replacement() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        _validation_with_replan_scope,
    )

    validation = {
        "status": "succeeded",
        "parsed": True,
        "sandbox_applied": True,
        "errors": [],
        "warnings": [],
        "files": ["src/runtime.py"],
    }
    patch_artifacts = [{
        "artifact_id": "compiled-1",
        "base": "repository",
        "diff_text": "diff --git a/src/runtime.py b/src/runtime.py\n",
        "files": ["src/runtime.py"],
        "summary": "Partial replacement.",
    }]

    guarded = _validation_with_replan_scope(
        previous_files={"src/runtime.py", "tests/test_runtime.py"},
        patch_artifacts=patch_artifacts,
        validation=validation,
    )

    assert guarded["status"] == "failed"
    assert guarded["sandbox_applied"] is True
    assert "tests/test_runtime.py" in guarded["errors"][0]


def test_validation_replan_restores_previous_candidate_when_empty() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        _should_restore_previous_candidate,
    )

    previous_artifacts = [{
        "artifact_id": "compiled-1",
        "base": "repository",
        "diff_text": "diff --git a/src/runtime.py b/src/runtime.py\n",
        "files": ["src/runtime.py"],
        "summary": "Previous candidate.",
    }]
    validation = {
        "status": "rejected",
        "parsed": False,
        "sandbox_applied": False,
        "errors": ["No patch artifacts were provided."],
        "warnings": [],
        "files": [],
    }

    should_restore = _should_restore_previous_candidate(
        previous_patch_artifacts=previous_artifacts,
        patch_artifacts=[],
        validation=validation,
    )

    assert should_restore is True


def test_validation_replan_does_not_restore_over_success() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        _should_restore_previous_candidate,
    )

    previous_artifacts = [{
        "artifact_id": "compiled-1",
        "base": "repository",
        "diff_text": "diff --git a/src/runtime.py b/src/runtime.py\n",
        "files": ["src/runtime.py"],
        "summary": "Previous candidate.",
    }]
    validation = {
        "status": "succeeded",
        "parsed": True,
        "sandbox_applied": True,
        "errors": [],
        "warnings": [],
        "files": ["src/runtime.py"],
    }

    should_restore = _should_restore_previous_candidate(
        previous_patch_artifacts=previous_artifacts,
        patch_artifacts=[],
        validation=validation,
    )

    assert should_restore is False


async def test_programmer_trace_records_prompt_budget_and_evidence(
    monkeypatch,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import programmer

    class FakeResponse:
        content = (
            '{"status": "blocked", "files_considered": [], "facts": [], '
            '"patch_operations": [], "patch_artifacts": [], '
            '"created_files": [], "changed_files": [], '
            '"open_questions": ["not enough detail"]}'
        )

    async def fake_ainvoke(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(programmer._programmer_llm, "ainvoke", fake_ainvoke)

    trace: dict[str, object] = {}
    report = await programmer.run_programmer_assignment(
        question="Add a focused behavior change.",
        mode="edit_existing_repository",
        assignment={
            "assignment_id": "runtime",
            "role": "python_developer",
            "scope": {
                "kind": "file",
                "values": ["src/runtime.py"],
            },
            "questions": ["Change the runtime branch."],
            "required_slots": ["runtime behavior"],
        },
        repository_summary=None,
        reading_evidence=[
            {
                "path": "src/runtime.py",
                "line_start": 10,
                "line_end": 20,
                "symbol_or_topic": "runtime branch",
                "excerpt": "raise ValueError('old')",
                "reason": "Runtime owner evidence.",
            }
        ],
        external_evidence=[],
        trace=trace,
    )

    context_budget = trace["context_budget"]

    assert report["status"] == "blocked"
    assert context_budget["over_hard_cap"] is False
    assert context_budget["selected_evidence_refs"] == ["src/runtime.py:10-20"]
    assert context_budget["pruned_evidence_count"] == 0


async def test_synthesis_trace_records_prompt_budget_and_artifacts(
    monkeypatch,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import synthesizer

    class FakeResponse:
        content = '{"answer_text": "Patch proposal only.", "limitations": []}'

    async def fake_ainvoke(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(synthesizer._synthesis_llm, "ainvoke", fake_ainvoke)

    trace: dict[str, object] = {}
    answer, limitations = await synthesizer.synthesize_patch_proposal(
        question="Add a small helper.",
        mode="create_new_project",
        pm_decision={
            "status": "need_programmers",
            "mode": "create_new_project",
            "intent": "new_project_cli_or_tool",
            "assignments": [],
            "missing_slots": [],
            "external_evidence_requests": [],
        },
        programmer_reports=[],
        patch_artifacts=[
            {
                "artifact_id": "main",
                "base": "new_file",
                "diff_text": "diff --git a/tool.py b/tool.py\n",
                "files": ["tool.py"],
                "summary": "Adds one file.",
            }
        ],
        validation={
            "status": "succeeded",
            "parsed": True,
            "sandbox_applied": True,
            "errors": [],
            "warnings": [],
            "files": ["tool.py"],
        },
        external_evidence=[],
        limitations=[],
        repository_summary=None,
        preferred_language=None,
        max_answer_chars=500,
        trace=trace,
    )

    context_budget = trace["context_budget"]

    assert answer == "Patch proposal only."
    assert limitations == []
    assert context_budget["artifact_ids"] == ["main"]
    assert context_budget["over_hard_cap"] is False


def test_synthesis_answer_removes_false_no_limitations_claim() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.synthesizer import (
        _answer_with_required_limitations,
    )

    answer = (
        "The patch proposal validated successfully. "
        "There are no reported limitations or missing information."
    )

    updated_answer = _answer_with_required_limitations(
        answer,
        ["Missing behavior tests were not found."],
        max_answer_chars=500,
    )

    assert "no reported limitations" not in updated_answer.lower()
    assert "Missing behavior tests were not found." in updated_answer
