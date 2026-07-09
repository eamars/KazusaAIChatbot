from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _default_writing_alignment_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep deterministic writing-supervisor tests off the live LLM seam."""

    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    async def fake_alignment(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "confidence": 100,
            "request_satisfied": True,
            "reasons": [
                f"Matched {len(kwargs['acceptance_criteria'])} criteria."
            ],
            "blockers": [],
            "feedback_for_pm": "",
        }

    monkeypatch.setattr(
        supervisor,
        "evaluate_artifact_alignment",
        fake_alignment,
        raising=False,
    )


def test_patch_validation_exposes_only_review_materialization_boundary() -> None:
    from kazusa_ai_chatbot.coding_agent.code_patching import patch_validation

    assert hasattr(patch_validation, "materialize_patch_artifacts_for_review")
    assert not hasattr(patch_validation, "validate_patch_artifacts")
    assert not hasattr(patch_validation, "_python_test_execution_error")
    assert not hasattr(patch_validation, "_sandbox_test_env")


def test_patch_validation_does_not_run_generated_python_tests() -> None:
    from kazusa_ai_chatbot.coding_agent.code_patching import patch_validation

    source_path = Path(patch_validation.__file__)
    source_text = source_path.read_text(encoding="utf-8")

    assert "_python_test_execution_error" not in source_text
    assert "sys.executable" not in source_text
    assert '"pytest"' not in source_text
    assert "Patched Python tests" not in source_text


def test_writing_supervisor_uses_review_materialization_boundary_only() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    source_path = Path(supervisor.__file__)
    source_text = source_path.read_text(encoding="utf-8")

    assert "materialize_patch_artifacts_for_review" in source_text
    assert "validate_patch_artifacts" not in source_text


def test_package_coherence_rejects_missing_generated_module_import() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.package_coherence import (
        evaluate_package_coherence,
    )

    report = evaluate_package_coherence([
        {
            "artifact_id": "runtime",
            "file_label": "runtime",
            "file_kind": "source",
            "content_format": "python",
            "path": "src/runtime.py",
            "content": "def run() -> int:\n    return 1\n",
            "purpose": "Provide runtime logic.",
        },
        {
            "artifact_id": "tests",
            "file_label": "tests",
            "file_kind": "test",
            "content_format": "python",
            "path": "tests/test_runtime.py",
            "content": (
                "from src.cli_wrapper import main\n\n"
                "def test_main() -> None:\n"
                "    assert main() == 0\n"
            ),
            "purpose": "Test runtime CLI.",
        },
    ])

    assert report["status"] == "failed"
    error_text = "\n".join(report["errors"])
    assert "missing generated local module 'src.cli_wrapper'" in error_text
    assert "tests/test_runtime.py" in error_text


def test_package_coherence_rejects_missing_imported_symbol() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.package_coherence import (
        evaluate_package_coherence,
    )

    report = evaluate_package_coherence([
        {
            "artifact_id": "runtime",
            "file_label": "runtime",
            "file_kind": "source",
            "content_format": "python",
            "path": "src/runtime.py",
            "content": "def run() -> int:\n    return 1\n",
            "purpose": "Provide runtime logic.",
        },
        {
            "artifact_id": "tests",
            "file_label": "tests",
            "file_kind": "test",
            "content_format": "python",
            "path": "tests/test_runtime.py",
            "content": "from src.runtime import main\n",
            "purpose": "Test runtime CLI.",
        },
    ])

    assert report["status"] == "failed"
    error_text = "\n".join(report["errors"])
    assert "missing generated symbol 'main' from module 'src.runtime'" in error_text


def test_package_coherence_rejects_direct_call_signature_mismatch() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.package_coherence import (
        evaluate_package_coherence,
    )

    report = evaluate_package_coherence([
        {
            "artifact_id": "cli",
            "file_label": "cli",
            "file_kind": "source",
            "content_format": "python",
            "path": "src/toml_linter_cli.py",
            "content": (
                "def main(argv: list[str]) -> int:\n"
                "    return 0\n"
            ),
            "purpose": "Provide CLI.",
        },
        {
            "artifact_id": "tests",
            "file_label": "tests",
            "file_kind": "test",
            "content_format": "python",
            "path": "tests/test_toml_linter_cli.py",
            "content": (
                "from src.toml_linter_cli import main as cli_main\n\n"
                "def test_cli_main() -> None:\n"
                "    assert cli_main() == 0\n"
            ),
            "purpose": "Test CLI.",
        },
    ])

    assert report["status"] == "failed"
    error_text = "\n".join(report["errors"])
    assert "calls 'src.toml_linter_cli.main' with 0 positional arguments" in error_text
    assert "requires at least 1" in error_text


def test_package_coherence_rejects_duplicate_cli_entrypoints() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.package_coherence import (
        evaluate_package_coherence,
    )

    report = evaluate_package_coherence([
        {
            "artifact_id": "cli",
            "file_label": "cli",
            "file_kind": "source",
            "content_format": "python",
            "path": "src/tool_cli.py",
            "content": "def main() -> int:\n    return 0\n",
            "purpose": "Provide CLI.",
        },
        {
            "artifact_id": "wrapper",
            "file_label": "wrapper",
            "file_kind": "source",
            "content_format": "python",
            "path": "src/tool_wrapper.py",
            "content": "def main() -> int:\n    return 0\n",
            "purpose": "Provide CLI wrapper.",
        },
    ])

    assert report["status"] == "failed"
    error_text = "\n".join(report["errors"])
    assert "multiple generated CLI entrypoint wrappers" in error_text


def test_package_coherence_allows_cli_tests_importing_main() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.package_coherence import (
        evaluate_package_coherence,
    )

    report = evaluate_package_coherence([
        {
            "artifact_id": "cli",
            "file_label": "cli",
            "file_kind": "source",
            "content_format": "python",
            "path": "src/tool_cli.py",
            "content": "def main() -> int:\n    return 0\n",
            "purpose": "Provide CLI.",
        },
        {
            "artifact_id": "cli_tests",
            "file_label": "cli tests",
            "file_kind": "test",
            "content_format": "python",
            "path": "tests/test_tool_cli.py",
            "content": (
                "from src.tool_cli import main\n\n"
                "def test_main() -> None:\n"
                "    assert main() == 0\n"
            ),
            "purpose": "Test CLI.",
        },
    ])

    assert report["status"] == "succeeded"
    assert report["errors"] == []


def test_writing_pm_normalizes_programmer_task_action() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
        normalize_writing_pm_decision,
    )

    decision = normalize_writing_pm_decision({
        "status": "create_programmer_task",
        "reason": "The assigned work is one bounded source artifact.",
        "information_request": None,
        "child_pm_task": None,
        "programmer_task": {
            "task_id": "runtime",
            "artifact_purpose": "Provide the runtime utility.",
            "required_behavior": ["define VALUE"],
            "provided_interfaces": ["VALUE constant"],
            "consumed_interfaces": [],
            "imports": [],
            "output_format": "python source",
        },
        "repair_instruction": None,
        "completion_report": None,
        "blocker": None,
    })

    assert decision["status"] == "create_programmer_task"
    assert decision["programmer_task"] is not None
    assert decision["programmer_task"]["task_id"] == "runtime"
    assert decision["programmer_task"]["consumed_fact_ids"] == []
    assert decision["child_pm_task"] is None


def test_writing_pm_rejects_missing_action_payload() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
        normalize_writing_pm_decision,
    )

    decision = normalize_writing_pm_decision({
        "status": "create_programmer_task",
        "reason": "missing task",
        "information_request": None,
        "child_pm_task": None,
        "programmer_task": None,
        "repair_instruction": None,
        "completion_report": None,
        "blocker": None,
    })

    assert decision["status"] == "blocked"
    assert decision["blocker"] is not None
    assert "programmer_task" in decision["blocker"]["summary"]


def test_writing_pm_rejects_phase2_repair_action() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
        normalize_writing_pm_decision,
    )

    decision = normalize_writing_pm_decision({
        "status": "repair_child",
        "reason": "Repair is not a Phase 2 writing action.",
        "information_request": None,
        "child_pm_task": None,
        "programmer_task": None,
        "repair_instruction": {
            "child_id": "runtime",
            "feedback": "change the generated artifact",
            "expected_correction": "return a corrected artifact",
        },
        "completion_report": None,
        "blocker": None,
    })

    assert decision["status"] == "blocked"
    assert decision["blocker"] is not None
    assert "unsupported status" in decision["blocker"]["summary"]


def test_writing_pm_normalizes_information_request_action() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
        normalize_writing_pm_decision,
    )

    decision = normalize_writing_pm_decision({
        "status": "request_information",
        "reason": "Source behavior is needed before dependent tests.",
        "information_request": {
            "request_id": "read_source",
            "needed_facts": ["function signature", "return shape"],
            "target_artifacts": ["src/runtime.py"],
            "reason_for_next_instruction": "Tests must match source behavior.",
        },
        "child_pm_task": None,
        "programmer_task": None,
        "repair_instruction": None,
        "completion_report": None,
        "blocker": None,
    })

    assert decision["status"] == "request_information"
    assert decision["information_request"] is not None
    assert decision["information_request"]["target_artifacts"] == ["src/runtime.py"]


def test_file_agent_reserves_safe_new_artifact_paths() -> None:
    from kazusa_ai_chatbot.coding_agent.file_agent import (
        reserve_new_artifact_paths,
    )

    reservation = reserve_new_artifact_paths([
        {
            "artifact_id": "runtime",
            "file_label": "runtime utility",
            "file_kind": "source",
            "content_format": "python",
            "purpose": "Provide the runtime utility.",
            "imports": [],
            "provided_interfaces": [],
            "consumed_interfaces": [],
            "required_behavior": ["be importable"],
            "preferred_name": "runtime_tool.py",
        },
        {
            "artifact_id": "usage_docs",
            "file_label": "usage docs",
            "file_kind": "docs",
            "content_format": "markdown",
            "purpose": "Document usage.",
            "imports": [],
            "provided_interfaces": [],
            "consumed_interfaces": [],
            "required_behavior": ["include invocation example"],
        },
    ])

    assert reservation["status"] == "accepted"
    assert reservation["reserved_paths"][0]["path"] == "src/runtime_tool.py"
    assert reservation["reserved_paths"][1]["path"] == "docs/usage_docs.md"


def test_file_agent_rejects_duplicate_reserved_paths() -> None:
    from kazusa_ai_chatbot.coding_agent.file_agent import (
        reserve_new_artifact_paths,
    )

    base_contract: dict[str, Any] = {
        "file_kind": "source",
        "content_format": "python",
        "purpose": "Provide a helper.",
        "imports": [],
        "provided_interfaces": [],
        "consumed_interfaces": [],
        "required_behavior": ["be importable"],
        "preferred_name": "helper.py",
    }
    reservation = reserve_new_artifact_paths([
        {"artifact_id": "first", "file_label": "first helper", **base_contract},
        {"artifact_id": "second", "file_label": "second helper", **base_contract},
    ])

    assert reservation["status"] == "repair_required"
    assert "duplicated" in " ".join(reservation["errors"])


def test_programmer_parser_accepts_one_fenced_artifact() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
        _extract_single_output_block,
    )

    content = _extract_single_output_block(
        "```python\nVALUE = 1\n```",
        content_format="python",
    )

    assert content == "VALUE = 1"


def test_programmer_diagnostics_reject_broad_exception_catches() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
        _artifact_diagnostics,
    )

    diagnostics = _artifact_diagnostics(
        "try:\n    run()\nexcept Exception as exc:\n    print(exc)\n",
        content_format="python",
    )

    assert diagnostics == [
        "Programmer output catches a broad exception type.",
    ]


def test_coding_supervisor_work_ledger_projects_compact_state() -> None:
    from kazusa_ai_chatbot.coding_agent.work_ledger import (
        CodingSupervisorWorkLedger,
    )

    ledger = CodingSupervisorWorkLedger(goal="Create a small utility.")
    ledger.record_writing_attempt(
        attempt_index=1,
        writing_result={
            "status": "need_reading",
            "created_files": [],
            "patch_artifacts": [],
            "reading_requests": [{"request_id": "source_readback"}],
            "external_evidence_requests": [],
            "limitations": [],
        },
    )
    ledger.record_supervisor_fact({
        "request_id": "source_readback",
        "kind": "generated_artifact_readback",
        "task": "Inspect generated source behavior.",
        "resolved": True,
        "result": "Generated source provides parse_rows(path: str).",
    })
    ledger.record_generated_artifacts([
        {
            "artifact_id": "runtime",
            "file_label": "runtime",
            "file_kind": "source",
            "content_format": "python",
            "path": "src/runtime.py",
            "content": "def parse_rows(path: str) -> list[dict[str, str]]:\n"
            "    return []\n",
            "purpose": "Provide runtime parsing.",
        }
    ])

    projection = ledger.projection()

    assert projection["external_evidence_count"] == 0
    assert projection["events"][0]["kind"] == "writing_attempt"
    assert projection["supervisor_facts"][0]["request_id"] == "source_readback"
    assert projection["generated_artifacts"] == [
        {
            "artifact_id": "runtime",
            "path": "src/runtime.py",
            "file_kind": "source",
            "content_format": "python",
            "purpose": "Provide runtime parsing.",
        }
    ]


def test_patcher_materializes_generated_new_files(tmp_path: Path) -> None:
    from kazusa_ai_chatbot.coding_agent.code_patching.patcher import (
        materialize_patch_artifacts,
    )

    report = materialize_patch_artifacts(
        repo_root=None,
        patcher_input={
            "artifact_package_id": "package-one",
            "artifacts": [
                {
                    "artifact_id": "runtime",
                    "file_label": "runtime utility",
                    "file_kind": "source",
                    "content_format": "python",
                    "path": "src/runtime_tool.py",
                    "content": "VALUE = 1\n",
                    "purpose": "Provide the runtime utility.",
                }
            ],
            "reserved_paths": [
                {
                    "artifact_id": "runtime",
                    "file_label": "runtime utility",
                    "path": "src/runtime_tool.py",
                    "file_kind": "source",
                    "content_format": "python",
                    "purpose": "Provide the runtime utility.",
                }
            ],
            "max_artifact_chars": 4000,
        },
        max_files=8,
        max_diff_chars=4000,
        trace={},
    )

    assert report["status"] == "succeeded"
    assert report["created_files"] == [
        {
            "path": "src/runtime_tool.py",
            "role": "Provide the runtime utility.",
        }
    ]
    assert "new file mode" in report["patch_artifacts"][0]["diff_text"]


async def test_code_writing_runs_lifecycle_programmer_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    pm_calls = 0

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        nonlocal pm_calls
        pm_calls += 1
        assert pm_input["work_item"]["goal"]
        if pm_calls == 1:
            return {
                "status": "create_programmer_task",
                "reason": "One source artifact can satisfy this request.",
                "information_request": None,
                "child_pm_task": None,
                "programmer_task": {
                    "task_id": "runtime",
                    "artifact_purpose": "Provide the runtime utility.",
                    "required_behavior": ["define VALUE"],
                    "provided_interfaces": ["VALUE constant"],
                    "consumed_interfaces": [],
                    "imports": [],
                    "output_format": "python source",
                },
                "repair_instruction": None,
                "completion_report": None,
                "blocker": None,
            }
        return {
            "status": "complete",
            "reason": "The artifact was generated.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": {
                "pm_id": "writing_pm_root",
                "status": "complete",
                "provided_facts": ["runtime artifact generated"],
                "created_artifacts": [
                    {
                        "artifact_id": "runtime",
                        "purpose": "Provide the runtime utility.",
                    }
                ],
                "consumed_facts": [],
                "open_risks": [],
                "next_dependency_needs": [],
            },
            "blocker": None,
        }

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        return {
            "artifact_id": artifact_contract["artifact_id"],
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": "VALUE = 1\n",
            "diagnostics": [],
        }

    async def fake_synthesis(**kwargs: Any) -> tuple[str, list[str]]:
        return "Generated the requested artifact package.", []

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "runtime",
                    "requirement": "Create a tiny runtime utility.",
                    "evidence_needed": "Generated source defines the utility.",
                }
            ],
            "limitations": [],
        }

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)
    monkeypatch.setattr(
        supervisor,
        "run_writing_programmer_contract",
        fake_programmer,
    )
    monkeypatch.setattr(supervisor, "synthesize_patch_proposal", fake_synthesis)

    result = await supervisor.run_writing_supervisor({
        "question": "Create a tiny runtime utility.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "succeeded"
    assert result["created_files"][0]["path"] == "src/runtime.py"
    assert result["validation"]["status"] == "succeeded"


async def test_code_writing_runs_alignment_feedback_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    alignment_calls = 0
    feedback_seen = False

    def programmer_decision(task_id: str) -> dict[str, Any]:
        return {
            "status": "create_programmer_task",
            "reason": "Create the runtime artifact.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": {
                "task_id": task_id,
                "artifact_purpose": "Provide the runtime utility.",
                "required_behavior": ["define VALUE and render_header"],
                "provided_interfaces": ["VALUE", "render_header"],
                "consumed_interfaces": [],
                "imports": [],
                "output_format": "python source",
            },
            "repair_instruction": None,
            "completion_report": None,
            "blocker": None,
        }

    def complete_decision() -> dict[str, Any]:
        return {
            "status": "complete",
            "reason": "The artifact was generated.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": {
                "pm_id": "writing_pm_root",
                "status": "complete",
                "provided_facts": ["runtime artifact generated"],
                "created_artifacts": [{
                    "artifact_id": "runtime",
                    "purpose": "Provide the runtime utility.",
                }],
                "consumed_facts": [],
                "open_risks": [],
                "next_dependency_needs": [],
            },
            "blocker": None,
        }

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        nonlocal feedback_seen

        child_feedback = pm_input["child_feedback"]
        child_reports = pm_input["direct_child_reports"]
        if child_feedback:
            feedback_seen = True
            assert "criterion-header" in str(child_feedback[0])
            if not child_reports:
                return programmer_decision("runtime_fixed")
            return complete_decision()
        if not child_reports:
            return programmer_decision("runtime")
        return complete_decision()

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        artifact_id = artifact_contract["artifact_id"]
        code = "VALUE = 1\n"
        if artifact_id == "runtime_fixed":
            code = (
                "VALUE = 1\n\n"
                "def render_header() -> str:\n"
                "    return 'Header'\n"
            )
        return {
            "artifact_id": artifact_id,
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": code,
            "diagnostics": [],
        }

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "criterion-header",
                    "requirement": "Runtime source renders a header.",
                    "evidence_needed": "Generated source defines render_header.",
                }
            ],
            "limitations": [],
        }

    async def fake_alignment(**kwargs: Any) -> dict[str, Any]:
        nonlocal alignment_calls

        alignment_calls += 1
        if alignment_calls == 1:
            return {
                "status": "fail",
                "confidence": 80,
                "request_satisfied": False,
                "reasons": ["criterion-header is not satisfied."],
                "blockers": ["Generated source does not define render_header."],
                "feedback_for_pm": (
                    "criterion-header is missing: add render_header to the "
                    "generated runtime source."
                ),
            }
        return {
            "status": "pass",
            "confidence": 90,
            "request_satisfied": True,
            "reasons": ["criterion-header is satisfied."],
            "blockers": [],
            "feedback_for_pm": "",
        }

    async def fake_synthesis(**kwargs: Any) -> tuple[str, list[str]]:
        return "Generated the aligned artifact package.", kwargs["limitations"]

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "evaluate_artifact_alignment", fake_alignment)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)
    monkeypatch.setattr(
        supervisor,
        "run_writing_programmer_contract",
        fake_programmer,
    )
    monkeypatch.setattr(supervisor, "synthesize_patch_proposal", fake_synthesis)

    result = await supervisor.run_writing_supervisor({
        "question": "Create a tiny runtime utility that renders a header.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "succeeded"
    assert result["alignment"]["status"] == "pass"
    assert alignment_calls == 2
    assert feedback_seen is True
    assert any(
        row == "writing_alignment:status=pass"
        for row in result["trace_summary"]
    )


async def test_code_writing_alignment_feedback_replaces_named_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    alignment_calls = 0
    cli_calls = 0
    repair_seen = False

    def programmer_decision(
        task_id: str,
        purpose: str,
        output_format: str = "python source",
    ) -> dict[str, Any]:
        return {
            "status": "create_programmer_task",
            "reason": purpose,
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": {
                "task_id": task_id,
                "artifact_purpose": purpose,
                "required_behavior": ["provide coherent CLI behavior"],
                "provided_interfaces": ["main()"],
                "consumed_interfaces": [],
                "imports": [],
                "output_format": output_format,
            },
            "repair_instruction": None,
            "completion_report": None,
            "blocker": None,
        }

    def complete_decision() -> dict[str, Any]:
        return {
            "status": "complete",
            "reason": "The package is complete.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": {
                "pm_id": "writing_pm_root",
                "status": "complete",
                "provided_facts": ["CLI package generated."],
                "created_artifacts": [],
                "consumed_facts": [],
                "open_risks": [],
                "next_dependency_needs": [],
            },
            "blocker": None,
        }

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        nonlocal repair_seen

        child_reports = pm_input["direct_child_reports"]
        child_feedback = pm_input["child_feedback"]
        if child_feedback:
            if not repair_seen:
                repair_seen = True
                assert child_feedback[0]["stage"] == "artifact_alignment"
                assert "src/cli_wrapper.py" in str(child_feedback[0])
                return programmer_decision(
                    "cli_wrapper",
                    "Repair src/cli_wrapper.py to match tests.",
                )
            return complete_decision()
        if not child_reports:
            return programmer_decision("cli_wrapper", "Create CLI wrapper.")
        if len(child_reports) == 1:
            return programmer_decision(
                "cli_tests",
                "Create CLI tests.",
                "python test",
            )
        return complete_decision()

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        nonlocal cli_calls

        artifact_id = artifact_contract["artifact_id"]
        if artifact_id == "cli_wrapper":
            cli_calls += 1
            result = "bad" if cli_calls == 1 else "ok"
            code = f"def main() -> str:\n    return '{result}'\n"
        else:
            code = (
                "from src.cli_wrapper import main\n\n"
                "def test_main() -> None:\n"
                "    assert main() == 'ok'\n"
            )
        return {
            "artifact_id": artifact_id,
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": code,
            "diagnostics": [],
        }

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "cli",
                    "requirement": "CLI and tests agree on visible behavior.",
                    "evidence_needed": "Generated CLI and tests match.",
                }
            ],
            "limitations": [],
        }

    async def fake_alignment(**kwargs: Any) -> dict[str, Any]:
        nonlocal alignment_calls

        alignment_calls += 1
        if alignment_calls == 1:
            return {
                "status": "fail",
                "confidence": 100,
                "request_satisfied": False,
                "reasons": ["src/cli_wrapper.py returns the wrong value."],
                "blockers": [
                    "src/cli_wrapper.py must be repaired to satisfy "
                    "tests/test_cli_tests.py."
                ],
                "feedback_for_pm": (
                    "Regenerate src/cli_wrapper.py so tests/test_cli_tests.py "
                    "passes against the same path."
                ),
            }
        return {
            "status": "pass",
            "confidence": 100,
            "request_satisfied": True,
            "reasons": ["CLI and tests now agree."],
            "blockers": [],
            "feedback_for_pm": "",
        }

    async def fake_synthesis(**kwargs: Any) -> tuple[str, list[str]]:
        return "Generated repaired CLI package.", kwargs["limitations"]

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "evaluate_artifact_alignment", fake_alignment)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)
    monkeypatch.setattr(
        supervisor,
        "run_writing_programmer_contract",
        fake_programmer,
    )
    monkeypatch.setattr(supervisor, "synthesize_patch_proposal", fake_synthesis)

    result = await supervisor.run_writing_supervisor({
        "question": "Create CLI and tests.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "succeeded"
    assert result["alignment"]["status"] == "pass"
    assert alignment_calls == 2
    assert cli_calls == 2
    assert {
        row["path"]
        for row in result["created_files"]
    } == {
        "src/cli_wrapper.py",
        "tests/test_cli_tests.py",
    }
    cli_diffs = [
        artifact["diff_text"]
        for artifact in result["patch_artifacts"]
        if artifact["files"] == ["src/cli_wrapper.py"]
    ]
    assert len(cli_diffs) == 1
    assert "return 'ok'" in cli_diffs[0]


async def test_code_writing_repairs_invalid_pm_action_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    pm_calls = 0
    repair_feedback_seen = False

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        nonlocal pm_calls
        nonlocal repair_feedback_seen

        pm_calls += 1
        if pm_calls == 1:
            return {
                "status": "create_child_pm",
                "reason": "Invalid missing child payload.",
                "information_request": None,
                "child_pm_task": None,
                "programmer_task": None,
                "repair_instruction": None,
                "completion_report": None,
                "blocker": None,
            }
        if pm_calls == 2:
            repair_feedback_seen = True
            assert "child_pm_task" in str(pm_input["child_feedback"][0])
            return {
                "status": "create_programmer_task",
                "reason": "Use a programmer task after structural repair.",
                "information_request": None,
                "child_pm_task": None,
                "programmer_task": {
                    "task_id": "runtime",
                    "artifact_purpose": "Provide the runtime utility.",
                    "required_behavior": ["define VALUE"],
                    "provided_interfaces": ["VALUE constant"],
                    "consumed_interfaces": [],
                    "imports": [],
                    "output_format": "python source",
                },
                "repair_instruction": None,
                "completion_report": None,
                "blocker": None,
            }
        return {
            "status": "complete",
            "reason": "The artifact was generated.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": {
                "pm_id": "writing_pm_root",
                "status": "complete",
                "provided_facts": ["runtime artifact generated"],
                "created_artifacts": [{
                    "artifact_id": "runtime",
                    "purpose": "Provide the runtime utility.",
                }],
                "consumed_facts": [],
                "open_risks": [],
                "next_dependency_needs": [],
            },
            "blocker": None,
        }

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        return {
            "artifact_id": artifact_contract["artifact_id"],
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": "VALUE = 1\n",
            "diagnostics": [],
        }

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "runtime",
                    "requirement": "Create a tiny runtime utility.",
                    "evidence_needed": "Generated source defines the utility.",
                }
            ],
            "limitations": [],
        }

    async def fake_synthesis(**kwargs: Any) -> tuple[str, list[str]]:
        return "Generated the requested artifact package.", []

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)
    monkeypatch.setattr(
        supervisor,
        "run_writing_programmer_contract",
        fake_programmer,
    )
    monkeypatch.setattr(supervisor, "synthesize_patch_proposal", fake_synthesis)

    result = await supervisor.run_writing_supervisor({
        "question": "Create a tiny runtime utility.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "succeeded"
    assert repair_feedback_seen is True
    assert any(
        row.startswith("writing_pm:invalid_action_repair")
        for row in result["trace_summary"]
    )


async def test_code_writing_returns_information_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "request_information",
            "reason": "Need source behavior before dependent work.",
            "information_request": {
                "request_id": "read_source",
                "needed_facts": ["function signature", "return shape"],
                "target_artifacts": ["src/runtime.py"],
                "reason_for_next_instruction": "Tests must match source.",
            },
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": None,
            "blocker": None,
        }

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "tests",
                    "requirement": "Create tests matching source behavior.",
                    "evidence_needed": "Tests consume source interface.",
                }
            ],
            "limitations": [],
        }

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)

    result = await supervisor.run_writing_supervisor({
        "question": "Create dependent tests.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "need_external_evidence"
    assert result["external_evidence_requests"][0]["request_id"] == "read_source"


async def test_code_writing_requests_generated_artifact_readback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    pm_calls = 0

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        nonlocal pm_calls
        pm_calls += 1
        if pm_calls == 1:
            return {
                "status": "create_programmer_task",
                "reason": "Create source before dependent tests.",
                "information_request": None,
                "child_pm_task": None,
                "programmer_task": {
                    "task_id": "runtime",
                    "artifact_purpose": "Provide the runtime utility.",
                    "required_behavior": ["define parse_rows"],
                    "provided_interfaces": [
                        "parse_rows(path: str) -> list[dict[str, str]]"
                    ],
                    "consumed_interfaces": [],
                    "imports": [],
                    "output_format": "python source",
                },
                "repair_instruction": None,
                "completion_report": None,
                "blocker": None,
            }
        return {
            "status": "request_information",
            "reason": "Dependent tests need generated source facts.",
            "information_request": {
                "request_id": "runtime_readback",
                "needed_facts": ["parse_rows signature", "returned row shape"],
                "target_artifacts": ["src/runtime.py"],
                "reason_for_next_instruction": "Tests must match source behavior.",
            },
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": None,
            "blocker": None,
        }

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        return {
            "artifact_id": artifact_contract["artifact_id"],
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": (
                "def parse_rows(path: str) -> list[dict[str, str]]:\n"
                "    return []\n"
            ),
            "diagnostics": [],
        }

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "tests",
                    "requirement": "Create source and matching tests.",
                    "evidence_needed": "Tests consume generated source.",
                }
            ],
            "limitations": [],
        }

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)
    monkeypatch.setattr(
        supervisor,
        "run_writing_programmer_contract",
        fake_programmer,
    )

    result = await supervisor.run_writing_supervisor({
        "question": "Create source and matching tests.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "need_reading"
    assert result["reading_requests"][0]["request_id"] == "runtime_readback"
    assert result["pending_artifacts"][0]["artifact_id"] == "runtime"
    readback_source = result["reading_source"]
    repository = readback_source["repository"]
    source_path = Path(repository["local_root"]) / "src" / "runtime.py"
    assert source_path.read_text(encoding="utf-8").startswith("def parse_rows")


async def test_code_writing_resumes_with_prior_artifact_and_readback_fact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    prior_artifact = {
        "artifact_id": "runtime",
        "file_label": "runtime",
        "file_kind": "source",
        "content_format": "python",
        "path": "src/runtime.py",
        "content": (
            "def parse_rows(path: str) -> list[dict[str, str]]:\n"
            "    return []\n"
        ),
        "purpose": "Provide row parsing.",
    }

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        child_reports = pm_input["direct_child_reports"]
        if len(child_reports) == 1:
            assert child_reports[0]["created_artifacts"][0]["artifact_id"] == "runtime"
            return {
                "status": "create_programmer_task",
                "reason": "Readback facts are available for dependent tests.",
                "information_request": None,
                "child_pm_task": None,
                "programmer_task": {
                    "task_id": "runtime_tests",
                    "artifact_purpose": "Test the runtime parser.",
                    "required_behavior": ["assert parse_rows returns rows"],
                    "provided_interfaces": [],
                    "consumed_interfaces": [
                        "parse_rows(path: str) -> list[dict[str, str]]",
                    ],
                    "consumed_fact_ids": ["runtime_readback"],
                    "imports": ["from src.runtime import parse_rows"],
                    "output_format": "python test",
                },
                "repair_instruction": None,
                "completion_report": None,
                "blocker": None,
            }
        return {
            "status": "complete",
            "reason": "Source and tests are ready.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": {
                "pm_id": "writing_pm_root",
                "status": "complete",
                "provided_facts": ["source and tests generated"],
                "created_artifacts": [
                    {
                        "artifact_id": "runtime",
                        "purpose": "Provide row parsing.",
                    },
                    {
                        "artifact_id": "runtime_tests",
                        "purpose": "Test row parsing.",
                    },
                ],
                "consumed_facts": ["runtime_readback"],
                "open_risks": [],
                "next_dependency_needs": [],
            },
            "blocker": None,
        }

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        return {
            "artifact_id": artifact_contract["artifact_id"],
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": (
                "from src.runtime import parse_rows\n\n"
                "def test_parse_rows_returns_rows(tmp_path):\n"
                "    assert parse_rows(str(tmp_path / 'input.txt')) == []\n"
            ),
            "diagnostics": [],
        }

    async def fake_synthesis(**kwargs: Any) -> tuple[str, list[str]]:
        return "Generated the requested artifact package.", []

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "tests",
                    "requirement": "Create tests matching generated source.",
                    "evidence_needed": "Tests consume source readback facts.",
                }
            ],
            "limitations": [],
        }

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)
    monkeypatch.setattr(
        supervisor,
        "run_writing_programmer_contract",
        fake_programmer,
    )
    monkeypatch.setattr(supervisor, "synthesize_patch_proposal", fake_synthesis)

    result = await supervisor.run_writing_supervisor({
        "question": "Create source and matching tests.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
        "prior_generated_artifacts": [prior_artifact],
        "supervisor_facts": [
            {
                "request_id": "runtime_readback",
                "kind": "generated_artifact_readback",
                "task": "Inspect generated runtime source.",
                "resolved": True,
                "result": (
                    "parse_rows(path: str) returns list[dict[str, str]]."
                ),
            }
        ],
    })

    assert result["status"] == "succeeded"
    created_paths = {row["path"] for row in result["created_files"]}
    assert created_paths == {"src/runtime.py", "tests/test_runtime_tests.py"}


async def test_code_writing_rejects_dependent_programmer_without_readback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    pm_calls = 0

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        nonlocal pm_calls
        pm_calls += 1
        if pm_calls == 1:
            return {
                "status": "create_programmer_task",
                "reason": "Create source before tests.",
                "information_request": None,
                "child_pm_task": None,
                "programmer_task": {
                    "task_id": "runtime",
                    "artifact_purpose": "Provide the runtime utility.",
                    "required_behavior": ["define parse_rows"],
                    "provided_interfaces": [
                        "parse_rows(path: str) -> list[dict[str, str]]",
                    ],
                    "consumed_interfaces": [],
                    "imports": [],
                    "output_format": "python source",
                },
                "repair_instruction": None,
                "completion_report": None,
                "blocker": None,
            }
        if pm_calls == 2:
            return {
                "status": "create_programmer_task",
                "reason": "Attempt dependent tests from report memory.",
                "information_request": None,
                "child_pm_task": None,
                "programmer_task": {
                    "task_id": "runtime_tests",
                    "artifact_purpose": "Test the runtime utility.",
                    "required_behavior": ["assert parse_rows behavior"],
                    "provided_interfaces": [],
                    "consumed_interfaces": [
                        "parse_rows(path: str) -> list[dict[str, str]]",
                    ],
                    "imports": ["from src.runtime import parse_rows"],
                    "output_format": "python test",
                },
                "repair_instruction": None,
                "completion_report": None,
                "blocker": None,
            }
        assert pm_input["child_feedback"]
        return {
            "status": "request_information",
            "reason": "Readback is required before dependent tests.",
            "information_request": {
                "request_id": "runtime_readback",
                "needed_facts": ["actual parse_rows interface"],
                "target_artifacts": ["src/runtime.py"],
                "reason_for_next_instruction": "Tests must match source.",
            },
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": None,
            "blocker": None,
        }

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        return {
            "artifact_id": artifact_contract["artifact_id"],
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": (
                "def parse_rows(path: str) -> list[dict[str, str]]:\n"
                "    return []\n"
            ),
            "diagnostics": [],
        }

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "tests",
                    "requirement": "Create source and matching tests.",
                    "evidence_needed": "Tests consume generated source.",
                }
            ],
            "limitations": [],
        }

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)
    monkeypatch.setattr(
        supervisor,
        "run_writing_programmer_contract",
        fake_programmer,
    )

    result = await supervisor.run_writing_supervisor({
        "question": "Create source and matching tests.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "need_reading"
    assert result["reading_requests"][0]["request_id"] == "runtime_readback"
    assert result["pending_artifacts"][0]["artifact_id"] == "runtime"
    assert any(
        row.startswith("writing_pm:programmer_rejected")
        for row in result["trace_summary"]
    )


async def test_code_writing_repairs_source_free_review_import_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    feedback_passes = 0

    def programmer_decision(
        *,
        task_id: str,
        purpose: str,
        behavior: str,
        imports: list[str],
        output_format: str,
    ) -> dict[str, Any]:
        return {
            "status": "create_programmer_task",
            "reason": purpose,
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": {
                "task_id": task_id,
                "artifact_purpose": purpose,
                "required_behavior": [behavior],
                "provided_interfaces": [],
                "consumed_interfaces": [],
                "imports": imports,
                "output_format": output_format,
            },
            "repair_instruction": None,
            "completion_report": None,
            "blocker": None,
        }

    def complete_decision() -> dict[str, Any]:
        return {
            "status": "complete",
            "reason": "The package is complete.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": {
                "pm_id": "writing_pm_root",
                "status": "complete",
                "provided_facts": ["CSV normalizer package generated."],
                "created_artifacts": [],
                "consumed_facts": [],
                "open_risks": [],
                "next_dependency_needs": [],
            },
            "blocker": None,
        }

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        nonlocal feedback_passes

        child_reports = pm_input["direct_child_reports"]
        child_feedback = pm_input["child_feedback"]
        if child_feedback:
            if not child_reports:
                feedback_passes += 1
                feedback_text = str(child_feedback[0])
                assert "unresolved local Python imports" in feedback_text
                assert "src.csv_normalizer_logic" in feedback_text
                return programmer_decision(
                    task_id="csv_normalizer_logic",
                    purpose="Provide the CSV normalization logic.",
                    behavior="define normalize_csv(text: str) -> str",
                    imports=[],
                    output_format="python source",
                )
            if len(child_reports) == 1:
                return programmer_decision(
                    task_id="csv_normalizer_tests",
                    purpose="Test the CSV normalization logic.",
                    behavior="assert normalize_csv trims cells",
                    imports=[
                        "from src.csv_normalizer_logic import normalize_csv",
                    ],
                    output_format="python test",
                )
            return complete_decision()

        if not child_reports:
            return programmer_decision(
                task_id="csv_normalizer_logic",
                purpose="Provide the CSV normalization logic.",
                behavior="define normalize_csv(text: str) -> str",
                imports=[],
                output_format="python source",
            )
        if len(child_reports) == 1:
            return programmer_decision(
                task_id="csv_normalizer_tests",
                purpose="Test the CSV normalization logic.",
                behavior="assert normalize_csv trims cells",
                imports=["from csv_normalizer import normalize_csv"],
                output_format="python test",
            )
        return complete_decision()

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        artifact_id = artifact_contract["artifact_id"]
        if artifact_id == "csv_normalizer_logic":
            code = (
                "def normalize_csv(text: str) -> str:\n"
                "    rows = []\n"
                "    for line in text.splitlines():\n"
                "        cells = [cell.strip() for cell in line.split(',')]\n"
                "        rows.append(','.join(cells))\n"
                "    return '\\n'.join(rows)\n"
            )
        else:
            imports = artifact_contract["imports"]
            code = (
                f"{imports[0]}\n\n"
                "def test_normalize_csv_trims_cells() -> None:\n"
                "    assert normalize_csv(' a , b ') == 'a,b'\n"
            )
        return {
            "artifact_id": artifact_id,
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": code,
            "diagnostics": [],
        }

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "csv",
                    "requirement": "Create CSV normalizer source and tests.",
                    "evidence_needed": "Generated tests import generated source.",
                }
            ],
            "limitations": [],
        }

    async def fake_synthesis(**kwargs: Any) -> tuple[str, list[str]]:
        return "Generated a coherent CSV normalizer package.", kwargs["limitations"]

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)
    monkeypatch.setattr(
        supervisor,
        "run_writing_programmer_contract",
        fake_programmer,
    )
    monkeypatch.setattr(supervisor, "synthesize_patch_proposal", fake_synthesis)

    result = await supervisor.run_writing_supervisor({
        "question": "Create CSV normalizer source and tests.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "succeeded"
    assert result["validation"]["status"] == "succeeded"
    assert feedback_passes == 1
    assert {
        row["path"]
        for row in result["created_files"]
    } == {
        "src/csv_normalizer_logic.py",
        "tests/test_csv_normalizer_tests.py",
    }
    assert not any(
        "unresolved local Python imports" in item
        for item in result["limitations"]
    )
    assert any(
        row.startswith("writing_validation_feedback:attempt=1")
        for row in result["trace_summary"]
    )


async def test_code_writing_repairs_package_coherence_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    feedback_passes = 0

    def programmer_decision(
        *,
        task_id: str,
        purpose: str,
        behavior: str,
        imports: list[str],
        output_format: str,
    ) -> dict[str, Any]:
        return {
            "status": "create_programmer_task",
            "reason": purpose,
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": {
                "task_id": task_id,
                "artifact_purpose": purpose,
                "required_behavior": [behavior],
                "provided_interfaces": [],
                "consumed_interfaces": [],
                "imports": imports,
                "output_format": output_format,
            },
            "repair_instruction": None,
            "completion_report": None,
            "blocker": None,
        }

    def complete_decision() -> dict[str, Any]:
        return {
            "status": "complete",
            "reason": "The package is complete.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": {
                "pm_id": "writing_pm_root",
                "status": "complete",
                "provided_facts": ["Runtime package generated."],
                "created_artifacts": [],
                "consumed_facts": [],
                "open_risks": [],
                "next_dependency_needs": [],
            },
            "blocker": None,
        }

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        nonlocal feedback_passes

        child_reports = pm_input["direct_child_reports"]
        child_feedback = pm_input["child_feedback"]
        if child_feedback:
            if not child_reports:
                feedback_passes += 1
                assert child_feedback[0]["stage"] == "package_coherence"
                feedback_text = str(child_feedback[0])
                assert "missing generated symbol 'main'" in feedback_text
                return programmer_decision(
                    task_id="runtime_fixed",
                    purpose="Provide fixed runtime entrypoint.",
                    behavior="define main() -> int",
                    imports=[],
                    output_format="python source",
                )
            if len(child_reports) == 1:
                return programmer_decision(
                    task_id="runtime_fixed_tests",
                    purpose="Test fixed runtime entrypoint.",
                    behavior="assert main returns zero",
                    imports=["from src.runtime_fixed import main"],
                    output_format="python test",
                )
            return complete_decision()

        if not child_reports:
            return programmer_decision(
                task_id="runtime",
                purpose="Provide runtime logic.",
                behavior="define run() -> int",
                imports=[],
                output_format="python source",
            )
        if len(child_reports) == 1:
            return programmer_decision(
                task_id="runtime_tests",
                purpose="Test runtime entrypoint.",
                behavior="assert main returns zero",
                imports=["from src.runtime import main"],
                output_format="python test",
            )
        return complete_decision()

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        artifact_id = artifact_contract["artifact_id"]
        if artifact_id == "runtime":
            code = "def run() -> int:\n    return 1\n"
        elif artifact_id == "runtime_tests":
            code = (
                "from src.runtime import main\n\n"
                "def test_main() -> None:\n"
                "    assert main() == 0\n"
            )
        elif artifact_id == "runtime_fixed":
            code = "def main() -> int:\n    return 0\n"
        else:
            code = (
                "from src.runtime_fixed import main\n\n"
                "def test_main() -> None:\n"
                "    assert main() == 0\n"
            )
        return {
            "artifact_id": artifact_id,
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": code,
            "diagnostics": [],
        }

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "runtime",
                    "requirement": "Create source and matching entrypoint tests.",
                    "evidence_needed": "Generated tests import generated source.",
                }
            ],
            "limitations": [],
        }

    async def fake_synthesis(**kwargs: Any) -> tuple[str, list[str]]:
        return "Generated a coherent runtime package.", kwargs["limitations"]

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)
    monkeypatch.setattr(
        supervisor,
        "run_writing_programmer_contract",
        fake_programmer,
    )
    monkeypatch.setattr(supervisor, "synthesize_patch_proposal", fake_synthesis)

    result = await supervisor.run_writing_supervisor({
        "question": "Create source and matching entrypoint tests.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "succeeded"
    assert feedback_passes == 1
    assert {
        row["path"]
        for row in result["created_files"]
    } == {
        "src/runtime_fixed.py",
        "tests/test_runtime_fixed_tests.py",
    }
    assert not any(
        "missing generated symbol" in item
        for item in result["limitations"]
    )
    assert any(
        row.startswith("writing_package_coherence:status=failed")
        for row in result["trace_summary"]
    )


async def test_code_writing_rejects_duplicate_progress_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    feedback_seen = False
    programmer_calls = 0

    def programmer_decision(task_id: str) -> dict[str, Any]:
        return {
            "status": "create_programmer_task",
            "reason": f"Create {task_id}.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": {
                "task_id": task_id,
                "artifact_purpose": f"Create {task_id}.",
                "required_behavior": ["define VALUE"],
                "provided_interfaces": ["VALUE"],
                "consumed_interfaces": [],
                "imports": [],
                "output_format": "python source",
            },
            "repair_instruction": None,
            "completion_report": None,
            "blocker": None,
        }

    def complete_decision() -> dict[str, Any]:
        return {
            "status": "complete",
            "reason": "The package is complete.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": {
                "pm_id": "writing_pm_root",
                "status": "complete",
                "provided_facts": ["Runtime artifact generated."],
                "created_artifacts": [],
                "consumed_facts": [],
                "open_risks": [],
                "next_dependency_needs": [],
            },
            "blocker": None,
        }

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        nonlocal feedback_seen

        child_reports = pm_input["direct_child_reports"]
        child_feedback = pm_input["child_feedback"]
        if child_feedback:
            feedback_seen = True
            assert child_feedback[0]["stage"] == "package_progress"
            assert "duplicate generated artifact path" in str(child_feedback[0])
            return complete_decision()
        if not child_reports:
            return programmer_decision("runtime")
        return programmer_decision("runtime")

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        nonlocal programmer_calls

        programmer_calls += 1
        return {
            "artifact_id": artifact_contract["artifact_id"],
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": "VALUE = 1\n",
            "diagnostics": [],
        }

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "runtime",
                    "requirement": "Create a runtime artifact.",
                    "evidence_needed": "Generated source defines VALUE.",
                }
            ],
            "limitations": [],
        }

    async def fake_synthesis(**kwargs: Any) -> tuple[str, list[str]]:
        return "Generated runtime source.", kwargs["limitations"]

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)
    monkeypatch.setattr(
        supervisor,
        "run_writing_programmer_contract",
        fake_programmer,
    )
    monkeypatch.setattr(supervisor, "synthesize_patch_proposal", fake_synthesis)

    result = await supervisor.run_writing_supervisor({
        "question": "Create a runtime artifact.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "succeeded"
    assert feedback_seen is True
    assert programmer_calls == 1
    assert {
        row["path"]
        for row in result["created_files"]
    } == {"src/runtime.py"}
    assert any(
        row.endswith("reason=package_progress")
        for row in result["trace_summary"]
    )


async def test_code_writing_repairs_rejected_source_free_patcher_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    feedback_passes = 0

    def programmer_decision(task_id: str, behavior: str) -> dict[str, Any]:
        return {
            "status": "create_programmer_task",
            "reason": behavior,
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": {
                "task_id": task_id,
                "artifact_purpose": behavior,
                "required_behavior": [behavior],
                "provided_interfaces": [],
                "consumed_interfaces": [],
                "imports": [],
                "output_format": "python source",
            },
            "repair_instruction": None,
            "completion_report": None,
            "blocker": None,
        }

    def complete_decision() -> dict[str, Any]:
        return {
            "status": "complete",
            "reason": "The package is complete.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": {
                "pm_id": "writing_pm_root",
                "status": "complete",
                "provided_facts": ["Compact helper generated."],
                "created_artifacts": [],
                "consumed_facts": [],
                "open_risks": [],
                "next_dependency_needs": [],
            },
            "blocker": None,
        }

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        nonlocal feedback_passes

        child_reports = pm_input["direct_child_reports"]
        child_feedback = pm_input["child_feedback"]
        if child_feedback:
            if not child_reports:
                feedback_passes += 1
                feedback_text = str(child_feedback[0])
                assert "Compiled patch operations exceed the diff limit" in (
                    feedback_text
                )
                return programmer_decision(
                    task_id="compact_helper",
                    behavior="Create a compact helper.",
                )
            return complete_decision()

        if not child_reports:
            return programmer_decision(
                task_id="oversized_helper",
                behavior="Create an oversized helper.",
            )
        return complete_decision()

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        artifact_id = artifact_contract["artifact_id"]
        if artifact_id == "oversized_helper":
            code = (
                "def helper() -> str:\n"
                f"    return {('x' * 2500)!r}\n"
            )
        else:
            code = "def helper() -> str:\n    return 'ok'\n"
        return {
            "artifact_id": artifact_id,
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": code,
            "diagnostics": [],
        }

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "helper",
                    "requirement": "Create a compact helper.",
                    "evidence_needed": "Generated helper source.",
                }
            ],
            "limitations": [],
        }

    async def fake_synthesis(**kwargs: Any) -> tuple[str, list[str]]:
        return "Generated a compact helper.", kwargs["limitations"]

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)
    monkeypatch.setattr(
        supervisor,
        "run_writing_programmer_contract",
        fake_programmer,
    )
    monkeypatch.setattr(supervisor, "synthesize_patch_proposal", fake_synthesis)

    result = await supervisor.run_writing_supervisor({
        "question": "Create a compact helper.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
        "max_artifact_chars": 800,
    })

    assert result["status"] == "succeeded"
    assert feedback_passes == 1
    assert {
        row["path"]
        for row in result["created_files"]
    } == {"src/compact_helper.py"}
    assert not any(
        "Compiled patch operations exceed the diff limit" in item
        for item in result["limitations"]
    )


async def test_code_writing_repairs_empty_pm_completion_with_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    feedback_passes = 0

    def programmer_decision() -> dict[str, Any]:
        return {
            "status": "create_programmer_task",
            "reason": "Create the requested helper source.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": {
                "task_id": "helper",
                "artifact_purpose": "Create the requested helper source.",
                "required_behavior": ["define helper() -> str"],
                "provided_interfaces": [],
                "consumed_interfaces": [],
                "imports": [],
                "output_format": "python source",
            },
            "repair_instruction": None,
            "completion_report": None,
            "blocker": None,
        }

    def complete_decision() -> dict[str, Any]:
        return {
            "status": "complete",
            "reason": "The package is complete.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": {
                "pm_id": "writing_pm_root",
                "status": "complete",
                "provided_facts": ["Helper generated."],
                "created_artifacts": [],
                "consumed_facts": [],
                "open_risks": [],
                "next_dependency_needs": [],
            },
            "blocker": None,
        }

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        nonlocal feedback_passes

        child_reports = pm_input["direct_child_reports"]
        child_feedback = pm_input["child_feedback"]
        if child_feedback:
            if not child_reports:
                feedback_passes += 1
                feedback_text = str(child_feedback[0])
                assert "completed without generated artifacts" in feedback_text
                return programmer_decision()
            return complete_decision()
        return complete_decision()

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        return {
            "artifact_id": artifact_contract["artifact_id"],
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": "def helper() -> str:\n    return 'ok'\n",
            "diagnostics": [],
        }

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "helper",
                    "requirement": "Create the requested helper source.",
                    "evidence_needed": "Generated helper source.",
                }
            ],
            "limitations": [],
        }

    async def fake_synthesis(**kwargs: Any) -> tuple[str, list[str]]:
        return "Generated helper source.", kwargs["limitations"]

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)
    monkeypatch.setattr(
        supervisor,
        "run_writing_programmer_contract",
        fake_programmer,
    )
    monkeypatch.setattr(supervisor, "synthesize_patch_proposal", fake_synthesis)

    result = await supervisor.run_writing_supervisor({
        "question": "Create helper source.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "succeeded"
    assert feedback_passes == 1
    assert {
        row["path"]
        for row in result["created_files"]
    } == {"src/helper.py"}


async def test_code_writing_rejects_nested_child_pm_and_resumes_same_pm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    child_saw_depth_feedback = False

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        nonlocal child_saw_depth_feedback

        pm_id = pm_input["pm_id"]
        child_reports = pm_input["direct_child_reports"]
        child_feedback = pm_input["child_feedback"]
        context_limits = pm_input["context_limits"]
        if pm_id == "writing_pm_root" and not child_reports:
            assert context_limits["remaining_child_pm_depth"] == 1
            return {
                "status": "create_child_pm",
                "reason": "A child PM can coordinate the utility artifact.",
                "information_request": None,
                "child_pm_task": {
                    "child_pm_id": "utility_child_pm",
                    "domain": "writing",
                    "goal": "Create the utility source artifact.",
                    "scope": "utility source only",
                    "constraints": ["new artifact only"],
                    "expected_report": ["source artifact generated"],
                },
                "programmer_task": None,
                "repair_instruction": None,
                "completion_report": None,
                "blocker": None,
            }
        if pm_id == "utility_child_pm" and not child_reports and not child_feedback:
            assert context_limits["remaining_child_pm_depth"] == 0
            return {
                "status": "create_child_pm",
                "reason": "Attempt one more PM layer.",
                "information_request": None,
                "child_pm_task": {
                    "child_pm_id": "too_deep_pm",
                    "domain": "writing",
                    "goal": "Create the utility source artifact.",
                    "scope": "same source artifact",
                    "constraints": ["new artifact only"],
                    "expected_report": ["source artifact generated"],
                },
                "programmer_task": None,
                "repair_instruction": None,
                "completion_report": None,
                "blocker": None,
            }
        if pm_id == "utility_child_pm" and child_reports:
            return {
                "status": "complete",
                "reason": "The child PM work is complete.",
                "information_request": None,
                "child_pm_task": None,
                "programmer_task": None,
                "repair_instruction": None,
                "completion_report": {
                    "pm_id": "utility_child_pm",
                    "status": "complete",
                    "provided_facts": ["utility source generated"],
                    "created_artifacts": [
                        {
                            "artifact_id": "utility",
                            "purpose": "Provide the utility source.",
                        }
                    ],
                    "consumed_facts": [],
                    "open_risks": [],
                    "next_dependency_needs": [],
                },
                "blocker": None,
            }
        if pm_id == "utility_child_pm" and child_feedback:
            child_saw_depth_feedback = True
            assert child_feedback[0]["remaining_child_pm_depth"] == 0
            return {
                "status": "create_programmer_task",
                "reason": "Use a programmer after delegation feedback.",
                "information_request": None,
                "child_pm_task": None,
                "programmer_task": {
                    "task_id": "utility",
                    "artifact_purpose": "Provide the utility source.",
                    "required_behavior": ["define VALUE"],
                    "provided_interfaces": ["VALUE constant"],
                    "consumed_interfaces": [],
                    "imports": [],
                    "output_format": "python source",
                },
                "repair_instruction": None,
                "completion_report": None,
                "blocker": None,
            }
        return {
            "status": "complete",
            "reason": "The root PM work is complete.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": {
                "pm_id": "writing_pm_root",
                "status": "complete",
                "provided_facts": ["utility package generated"],
                "created_artifacts": [
                    {
                        "artifact_id": "utility",
                        "purpose": "Provide the utility source.",
                    }
                ],
                "consumed_facts": [],
                "open_risks": [],
                "next_dependency_needs": [],
            },
            "blocker": None,
        }

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        return {
            "artifact_id": artifact_contract["artifact_id"],
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": "VALUE = 1\n",
            "diagnostics": [],
        }

    async def fake_synthesis(**kwargs: Any) -> tuple[str, list[str]]:
        return "Generated the requested artifact package.", []

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "utility",
                    "requirement": "Create a small utility.",
                    "evidence_needed": "Generated source defines the utility.",
                }
            ],
            "limitations": [],
        }

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)
    monkeypatch.setattr(
        supervisor,
        "run_writing_programmer_contract",
        fake_programmer,
    )
    monkeypatch.setattr(supervisor, "synthesize_patch_proposal", fake_synthesis)

    result = await supervisor.run_writing_supervisor({
        "question": "Create a small utility.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "succeeded"
    assert child_saw_depth_feedback is True
    assert any(
        row.startswith("writing_pm:child_pm_rejected")
        for row in result["trace_summary"]
    )
