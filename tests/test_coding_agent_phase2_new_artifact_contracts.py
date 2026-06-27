from pathlib import Path
from typing import Any

import pytest


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
