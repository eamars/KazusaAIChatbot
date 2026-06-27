from pathlib import Path
from typing import Any

import pytest


def test_writing_pm_normalizes_new_artifact_contracts() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
        normalize_writing_pm_decision,
    )

    decision = normalize_writing_pm_decision({
        "status": "need_programmers",
        "feature_goal": "Create a small utility.",
        "artifact_items": [
            {
                "artifact_id": "utility",
                "file_label": "utility module",
                "file_kind": "source",
                "content_format": "python",
                "purpose": "Count input records.",
                "imports": ["from collections import Counter"],
                "provided_interfaces": [
                    {
                        "name": "count_records",
                        "kind": "function",
                        "contract": "accept rows and return counts",
                    }
                ],
                "consumed_interfaces": [],
                "required_behavior": ["return an empty count for empty input"],
                "preferred_name": "record_counter.py",
            }
        ],
        "selected_artifacts": [],
        "external_evidence_requests": [],
        "limitations": [],
    })

    assert decision["status"] == "need_programmers"
    assert decision["feature_goal"] == "Create a small utility."
    assert decision["artifact_items"][0]["artifact_id"] == "utility"
    assert decision["artifact_items"][0]["preferred_name"] == "record_counter.py"
    assert decision["external_evidence_requests"] == []


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

    base_item: dict[str, Any] = {
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
        {"artifact_id": "first", "file_label": "first helper", **base_item},
        {"artifact_id": "second", "file_label": "second helper", **base_item},
    ])

    assert reservation["status"] == "repair_required"
    assert "duplicated" in " ".join(reservation["errors"])


def test_file_agent_uses_artifact_id_before_label_for_generated_names() -> None:
    from kazusa_ai_chatbot.coding_agent.file_agent import (
        reserve_new_artifact_paths,
    )

    reservation = reserve_new_artifact_paths([
        {
            "artifact_id": "worker",
            "file_label": "runtime helper",
            "file_kind": "source",
            "content_format": "python",
            "purpose": "Provide worker behavior.",
            "imports": [],
            "provided_interfaces": [],
            "consumed_interfaces": [],
            "required_behavior": ["be importable"],
        }
    ])

    assert reservation["status"] == "accepted"
    assert reservation["reserved_paths"][0]["path"] == "src/worker.py"


def test_programmer_parser_accepts_one_fenced_artifact() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
        _extract_single_output_block,
    )

    content = _extract_single_output_block(
        "```python\nVALUE = 1\n```",
        content_format="python",
    )

    assert content == "VALUE = 1"


def test_programmer_parser_accepts_markdown_with_inner_fences() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
        _extract_single_output_block,
    )

    content = _extract_single_output_block(
        "````markdown\n# Usage\n\n```bash\npython tool.py\n```\n````",
        content_format="markdown",
    )

    assert "```bash\npython tool.py\n```" in content


def test_programmer_diagnostics_allow_requested_literal_state_names() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
        _artifact_diagnostics,
    )

    diagnostics = _artifact_diagnostics(
        '[state_names]\nWAITING = "pending"\nFINISHED = "done"\n',
        content_format="text",
    )

    assert diagnostics == []


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


def test_programmer_diagnostics_allow_specific_exception_fallback_pass() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
        _artifact_diagnostics,
    )

    diagnostics = _artifact_diagnostics(
        (
            "import urllib.error\n\n"
            "def fetch() -> dict[str, str]:\n"
            "    result = {'title': '', 'h1': ''}\n"
            "    try:\n"
            "        raise urllib.error.URLError('offline')\n"
            "    except urllib.error.URLError:\n"
            "        pass\n"
            "    return result\n"
        ),
        content_format="python",
    )

    assert diagnostics == []


def test_programmer_diagnostics_reject_pass_only_function_body() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
        _artifact_diagnostics,
    )

    diagnostics = _artifact_diagnostics(
        "def fetch() -> None:\n    pass\n",
        content_format="python",
    )

    assert diagnostics == [
        "Programmer output contains a pass placeholder.",
    ]


def test_programmer_diagnostics_reject_unbalanced_markdown_fences() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
        _artifact_diagnostics,
    )

    diagnostics = _artifact_diagnostics(
        "# Usage\n\n```bash\npython tool.py\n",
        content_format="markdown",
    )

    assert diagnostics == [
        "Programmer output contains unbalanced Markdown code fences.",
    ]


def test_patcher_materializes_generated_new_files(tmp_path: Path) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patcher import (
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


def test_programmer_contract_uses_reserved_local_module_import() -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
        _local_python_modules_by_artifact,
        _programmer_contract,
    )

    source_item: dict[str, Any] = {
        "artifact_id": "worker_core",
        "file_label": "worker source",
        "file_kind": "source",
        "content_format": "python",
        "purpose": "Provide worker behavior.",
        "imports": [],
        "provided_interfaces": [
            {
                "name": "run",
                "kind": "function",
                "contract": "run the worker",
            }
        ],
        "consumed_interfaces": [],
        "required_behavior": ["define run"],
    }
    consumer_item: dict[str, Any] = {
        "artifact_id": "worker_tests",
        "file_label": "worker tests",
        "file_kind": "test",
        "content_format": "python",
        "purpose": "Test worker behavior.",
        "imports": ["from worker import run"],
        "provided_interfaces": [],
        "consumed_interfaces": [
            {
                "name": "run",
                "provider": "worker_core",
                "contract": "run the worker",
            }
        ],
        "required_behavior": ["import run from the worker source"],
    }
    reserved_paths = [
        {
            "artifact_id": "worker_core",
            "file_label": "worker source",
            "path": "src/worker_core.py",
            "file_kind": "source",
            "content_format": "python",
            "purpose": "Provide worker behavior.",
        },
        {
            "artifact_id": "worker_tests",
            "file_label": "worker tests",
            "path": "tests/test_worker.py",
            "file_kind": "test",
            "content_format": "python",
            "purpose": "Test worker behavior.",
        },
    ]
    local_modules = _local_python_modules_by_artifact(
        artifact_items=[source_item, consumer_item],
        reserved_paths=reserved_paths,
    )

    contract = _programmer_contract(
        consumer_item,
        local_modules=local_modules,
    )

    assert contract["imports"] == ["from worker_core import run"]


def test_validation_accepts_src_sibling_imports(tmp_path: Path) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing.patcher import (
        materialize_patch_artifacts,
    )
    from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
        validate_patch_artifacts,
    )

    report = materialize_patch_artifacts(
        repo_root=None,
        patcher_input={
            "artifact_package_id": "package-one",
            "artifacts": [
                {
                    "artifact_id": "worker",
                    "file_label": "worker source",
                    "file_kind": "source",
                    "content_format": "python",
                    "path": "src/worker.py",
                    "content": "def run(value: int) -> int:\n    return value + 1\n",
                    "purpose": "Provide worker behavior.",
                },
                {
                    "artifact_id": "main",
                    "file_label": "main source",
                    "file_kind": "source",
                    "content_format": "python",
                    "path": "src/main.py",
                    "content": (
                        "from worker import run\n\n"
                        "def execute(value: int) -> int:\n"
                        "    return run(value)\n"
                    ),
                    "purpose": "Use worker behavior.",
                },
            ],
            "reserved_paths": [
                {
                    "artifact_id": "worker",
                    "file_label": "worker source",
                    "path": "src/worker.py",
                    "file_kind": "source",
                    "content_format": "python",
                    "purpose": "Provide worker behavior.",
                },
                {
                    "artifact_id": "main",
                    "file_label": "main source",
                    "path": "src/main.py",
                    "file_kind": "source",
                    "content_format": "python",
                    "purpose": "Use worker behavior.",
                },
            ],
            "max_artifact_chars": 4000,
        },
        max_files=8,
        max_diff_chars=8000,
        trace={},
    )

    validation = validate_patch_artifacts(
        repo_root=None,
        workspace_root=tmp_path / "workspace",
        patch_artifacts=report["patch_artifacts"],
        max_files=8,
        max_diff_chars=8000,
    )

    assert validation["status"] == "succeeded"


async def test_code_writing_runs_new_artifact_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        assert pm_input["mode"] == "create_new_project"
        assert pm_input["acceptance_criteria"]
        return {
            "status": "need_programmers",
            "feature_goal": "Create a small utility.",
            "artifact_items": [
                {
                    "artifact_id": "runtime",
                    "file_label": "runtime utility",
                    "file_kind": "source",
                    "content_format": "python",
                    "purpose": "Provide the runtime utility.",
                    "imports": [],
                    "provided_interfaces": [],
                    "consumed_interfaces": [],
                    "required_behavior": ["define VALUE"],
                    "preferred_name": "runtime_tool.py",
                }
            ],
            "selected_artifacts": [],
            "external_evidence_requests": [],
            "limitations": [],
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

    async def fake_alignment(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "confidence": 100,
            "request_satisfied": True,
            "reasons": ["Generated source matches the preserved requirement."],
            "blockers": [],
            "feedback_for_pm": "",
        }

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
        "question": "Create a tiny runtime utility.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "succeeded"
    assert result["created_files"][0]["path"] == "src/runtime_tool.py"
    assert result["changed_files"][0]["change_type"] == "add"
    assert result["validation"]["status"] == "succeeded"


async def test_code_writing_repairs_after_validation_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    pm_inputs: list[dict[str, Any]] = []

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        pm_inputs.append(pm_input)
        assert pm_input["acceptance_criteria"]
        behavior = "define VALUE = 1"
        if "validation_feedback" in pm_input:
            behavior = "define VALUE = 2"
        return {
            "status": "need_programmers",
            "feature_goal": "Create a small utility.",
            "artifact_items": [
                {
                    "artifact_id": "runtime",
                    "file_label": "runtime utility",
                    "file_kind": "source",
                    "content_format": "python",
                    "purpose": "Provide the runtime utility.",
                    "imports": [],
                    "provided_interfaces": [],
                    "consumed_interfaces": [],
                    "required_behavior": [behavior],
                    "preferred_name": "runtime_tool.py",
                }
            ],
            "selected_artifacts": [],
            "external_evidence_requests": [],
            "limitations": [],
        }

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        value = "2" if "VALUE = 2" in artifact_contract["required_behavior"][0] else "1"
        return {
            "artifact_id": artifact_contract["artifact_id"],
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": f"VALUE = {value}\n",
            "diagnostics": [],
        }

    validation_calls = 0

    def fake_validation(**kwargs: Any) -> dict[str, Any]:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            return {
                "status": "failed",
                "parsed": True,
                "sandbox_applied": False,
                "errors": ["Generated value did not match expected behavior."],
                "warnings": [],
                "files": ["src/runtime_tool.py"],
            }
        return {
            "status": "succeeded",
            "parsed": True,
            "sandbox_applied": True,
            "errors": [],
            "warnings": [],
            "files": ["src/runtime_tool.py"],
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
                    "evidence_needed": "Generated source defines VALUE.",
                }
            ],
            "limitations": [],
        }

    async def fake_alignment(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "confidence": 100,
            "request_satisfied": True,
            "reasons": ["Generated source matches the preserved requirement."],
            "blockers": [],
            "feedback_for_pm": "",
        }

    monkeypatch.setattr(supervisor, "derive_acceptance_criteria", fake_acceptance)
    monkeypatch.setattr(supervisor, "evaluate_artifact_alignment", fake_alignment)
    monkeypatch.setattr(supervisor, "decide_writing_work", fake_pm)
    monkeypatch.setattr(
        supervisor,
        "run_writing_programmer_contract",
        fake_programmer,
    )
    monkeypatch.setattr(supervisor, "validate_patch_artifacts", fake_validation)
    monkeypatch.setattr(supervisor, "synthesize_patch_proposal", fake_synthesis)

    result = await supervisor.run_writing_supervisor({
        "question": "Create a tiny runtime utility.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "succeeded"
    assert validation_calls == 2
    assert len(pm_inputs) == 2
    assert "validation_feedback" in pm_inputs[1]
    assert "writing_repair:validation_feedback_to_pm" in result["trace_summary"]


async def test_code_writing_repairs_after_alignment_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_writing import supervisor

    pm_inputs: list[dict[str, Any]] = []

    async def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "acceptance_criteria": [
                {
                    "criterion_id": "runnable",
                    "requirement": "Create a runnable command-line utility.",
                    "evidence_needed": "Generated source exposes main().",
                }
            ],
            "limitations": [],
        }

    async def fake_pm(
        pm_input: dict[str, Any],
        *,
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        pm_inputs.append(pm_input)
        behavior = "define helper only"
        if "alignment_feedback" in pm_input:
            behavior = "define helper and main"
        return {
            "status": "need_programmers",
            "feature_goal": "Create a command-line utility.",
            "artifact_items": [
                {
                    "artifact_id": "runtime",
                    "file_label": "runtime utility",
                    "file_kind": "source",
                    "content_format": "python",
                    "purpose": "Provide the runtime utility.",
                    "imports": [],
                    "provided_interfaces": [],
                    "consumed_interfaces": [],
                    "required_behavior": [behavior],
                    "preferred_name": "runtime_tool.py",
                }
            ],
            "selected_artifacts": [],
            "external_evidence_requests": [],
            "limitations": [],
        }

    async def fake_programmer(
        *,
        artifact_contract: dict[str, Any],
        trace: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        if "main" in artifact_contract["required_behavior"][0]:
            artifact = "def helper() -> int:\n    return 1\n\ndef main() -> None:\n    helper()\n"
        else:
            artifact = "def helper() -> int:\n    return 1\n"
        return {
            "artifact_id": artifact_contract["artifact_id"],
            "status": "succeeded",
            "content_format": artifact_contract["content_format"],
            "code_artifact": artifact,
            "diagnostics": [],
        }

    alignment_calls = 0

    async def fake_alignment(**kwargs: Any) -> dict[str, Any]:
        nonlocal alignment_calls
        alignment_calls += 1
        if alignment_calls == 1:
            return {
                "status": "fail",
                "confidence": 95,
                "request_satisfied": False,
                "reasons": ["The source only exposes a helper."],
                "blockers": ["No runnable command-line entrypoint is present."],
                "feedback_for_pm": "Add a source contract for a runnable entrypoint.",
            }
        return {
            "status": "pass",
            "confidence": 95,
            "request_satisfied": True,
            "reasons": ["The source now exposes main()."],
            "blockers": [],
            "feedback_for_pm": "",
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
    monkeypatch.setattr(supervisor, "evaluate_artifact_alignment", fake_alignment)
    monkeypatch.setattr(supervisor, "synthesize_patch_proposal", fake_synthesis)

    result = await supervisor.run_writing_supervisor({
        "question": "Create a tiny command-line utility.",
        "mode_hint": "create_new_project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert result["status"] == "succeeded"
    assert alignment_calls == 2
    assert len(pm_inputs) == 2
    assert "alignment_feedback" in pm_inputs[1]
    assert "writing_repair:alignment_feedback_to_pm" in result["trace_summary"]


async def test_propose_code_change_rejects_existing_source_without_fetching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import code_fetching
    from kazusa_ai_chatbot.coding_agent import propose_code_change

    async def fail_fetching(request: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("Fetching must not run for Phase 2 source edits.")

    monkeypatch.setattr(code_fetching, "run", fail_fetching)

    response = await propose_code_change({
        "question": "Modify the existing repository behavior.",
        "repo_hint": "example/project",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert response["status"] == "rejected"
    assert response["mode"] == "edit_existing_repository"
    assert response["patch_artifacts"] == []
    assert response["trace_summary"] == ["writing:existing_source_rejected"]
