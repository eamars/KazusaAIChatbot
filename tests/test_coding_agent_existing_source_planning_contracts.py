from __future__ import annotations

from pathlib import Path

import pytest


FIXTURE_ROOT = Path("tests/fixtures/coding_agent_existing_source_gates")

GATE_EXPECTATIONS = [
    (
        "gate_01_log_counter",
        "Add JSON output to the log counter CLI.",
        ["log_counter.py", "tests/test_log_counter.py"],
        ["log_counter.py"],
    ),
    (
        "gate_02_contacts_jsonl_to_csv",
        "Add field ordering and strict malformed JSON handling.",
        [
            "contacts_jsonl_to_csv/converter.py",
            "contacts_jsonl_to_csv/cli.py",
            "tests/test_converter.py",
            "README.md",
        ],
        [
            "contacts_jsonl_to_csv/converter.py",
            "contacts_jsonl_to_csv/cli.py",
        ],
    ),
    (
        "gate_03_markdown_link_checker",
        "Ignore fenced code links and support duplicate heading anchors.",
        [
            "mdlinkcheck/anchors.py",
            "mdlinkcheck/scanner.py",
            "tests/test_anchors.py",
            "tests/test_scanner.py",
        ],
        ["mdlinkcheck/anchors.py", "mdlinkcheck/scanner.py"],
    ),
    (
        "gate_04_issue_tracker_soft_delete",
        "Implement soft delete across model, store, and API behavior.",
        [
            "issue_tracker/models.py",
            "issue_tracker/store.py",
            "issue_tracker/api.py",
            "tests/test_store.py",
            "tests/test_api.py",
            "README.md",
        ],
        [
            "issue_tracker/models.py",
            "issue_tracker/store.py",
            "issue_tracker/api.py",
        ],
    ),
    (
        "gate_05_inventory_sync_fetch_cache",
        "Add fetch timeout, retry, cache, and CLI flags.",
        [
            "inventory_sync/fetch.py",
            "inventory_sync/cli.py",
            "tests/test_fetch.py",
            "tests/test_cli.py",
            "README.md",
        ],
        ["inventory_sync/fetch.py", "inventory_sync/cli.py"],
    ),
]


@pytest.mark.parametrize(
    ("fixture_dir", "question", "evidence_paths", "owner_paths"),
    GATE_EXPECTATIONS,
)
def test_file_agent_ranks_source_owners_for_gate_shapes(
    fixture_dir: str,
    question: str,
    evidence_paths: list[str],
    owner_paths: list[str],
) -> None:
    from kazusa_ai_chatbot.coding_agent.file_agent import (
        plan_existing_source_files,
    )

    repository = {"local_root": str(FIXTURE_ROOT / fixture_dir)}
    reading_result = {
        "answer_text": "Relevant source and tests were found.",
        "evidence": [
            {
                "path": path,
                "summary": f"Evidence for {path}",
                "excerpt": "",
            }
            for path in evidence_paths
        ],
    }

    file_plan = plan_existing_source_files(
        question=question,
        repository=repository,
        source_scope={"kind": "local_path"},
        reading_result=reading_result,
    )

    assert file_plan["status"] == "accepted"
    assert set(owner_paths).issubset(set(file_plan["owned_path_candidates"]))
    assert not set(owner_paths).intersection(
        set(file_plan["test_or_doc_path_candidates"])
    )
    first_ranked = file_plan["ranked_source_owner_candidates"][0]
    assert first_ranked["path"] in owner_paths
    assert file_plan["file_contexts"]


def test_file_agent_rejects_unsafe_or_secret_context_paths(tmp_path: Path) -> None:
    from kazusa_ai_chatbot.coding_agent.file_agent import (
        plan_existing_source_files,
    )

    safe_file = tmp_path / "app.py"
    safe_file.write_text("VALUE = 1\n", encoding="utf-8")
    secret_file = tmp_path / ".env"
    secret_file.write_text("TOKEN=value\n", encoding="utf-8")
    reading_result = {
        "answer_text": "Evidence rows include unsafe paths.",
        "evidence": [
            {"path": "../escape.py", "summary": "escape"},
            {"path": ".env", "summary": "secret"},
            {"path": "app.py", "summary": "source"},
        ],
    }

    file_plan = plan_existing_source_files(
        question="Change VALUE.",
        repository={"local_root": str(tmp_path)},
        source_scope={"kind": "local_path"},
        reading_result=reading_result,
    )

    assert file_plan["status"] == "accepted"
    assert file_plan["owned_path_candidates"] == ["app.py"]
    rejected_paths = {row["path"] for row in file_plan["rejected_paths"]}
    assert "../escape.py" in rejected_paths
    assert ".env" in rejected_paths


def test_modifying_pm_prompt_defaults_to_companion_updates() -> None:
    from kazusa_ai_chatbot.coding_agent.code_modifying.product_manager import (
        MODIFYING_PM_PROMPT,
    )

    prompt = MODIFYING_PM_PROMPT.casefold()

    assert "file_plan.test_or_doc_path_candidates" in prompt
    assert "source-change task should include" in prompt
    assert "do not omit companion tests or docs" in prompt


@pytest.mark.asyncio
async def test_modifying_supervisor_repairs_test_only_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_modifying import supervisor

    source_file = tmp_path / "app.py"
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_app.py"
    test_file.write_text("from app import VALUE\n\nassert VALUE == 1\n", encoding="utf-8")
    decisions: list[dict[str, object]] = [
        {
            "status": "create_programmer_task",
            "reason": "Tests mention the behavior.",
            "owned_paths": ["tests/test_app.py"],
            "read_only_paths": [],
            "required_evidence_ids": ["evidence-2"],
            "programmer_task": {
                "task_id": "task-tests-only",
                "target_paths": ["tests/test_app.py"],
                "change_goal": "Update only the test.",
                "required_behavior": ["Tests assert VALUE 2."],
                "forbidden_changes": [],
                "consumed_interfaces": [],
                "expected_operations": ["replace"],
                "acceptance_checks": ["Tests cover VALUE."],
                "local_risks": [],
            },
        },
        {
            "status": "create_programmer_task",
            "reason": "Runtime owner must change with companion tests.",
            "owned_paths": ["app.py"],
            "read_only_paths": ["tests/test_app.py"],
            "required_evidence_ids": ["evidence-1", "evidence-2"],
            "programmer_task": {
                "task_id": "task-runtime",
                "target_paths": ["app.py", "tests/test_app.py"],
                "change_goal": "Update VALUE and its assertion.",
                "required_behavior": ["VALUE is 2."],
                "forbidden_changes": [],
                "consumed_interfaces": [],
                "expected_operations": ["replace"],
                "acceptance_checks": ["Tests assert VALUE."],
                "local_risks": [],
            },
        },
    ]
    pm_inputs: list[dict[str, object]] = []
    programmer_payloads: list[dict[str, object]] = []

    async def fake_pm(payload: dict[str, object]) -> dict[str, object]:
        pm_inputs.append(payload)
        return decisions.pop(0)

    async def fake_programmer(payload: dict[str, object]) -> dict[str, object]:
        programmer_payloads.append(payload)
        return {
            "artifacts": [
                {
                    "artifact_id": "artifact-app",
                    "status": "succeeded",
                    "task_id": "task-runtime",
                    "target_path": "app.py",
                    "evidence_ids": ["evidence-1"],
                    "operation_kind": "replace",
                    "exact_anchor": "VALUE = 1\n",
                    "replacement_or_insert_content": "VALUE = 2\n",
                    "operation_summary": "Update VALUE.",
                    "risk_notes": [],
                    "tests_or_docs_to_update": ["tests/test_app.py"],
                },
                {
                    "artifact_id": "artifact-test",
                    "status": "succeeded",
                    "task_id": "task-runtime",
                    "target_path": "tests/test_app.py",
                    "evidence_ids": ["evidence-2"],
                    "operation_kind": "replace",
                    "exact_anchor": "assert VALUE == 1\n",
                    "replacement_or_insert_content": "assert VALUE == 2\n",
                    "operation_summary": "Update VALUE test.",
                    "risk_notes": [],
                    "tests_or_docs_to_update": [],
                }
            ],
            "answer_text": "Prepared runtime owner change.",
            "limitations": [],
            "raw_output": "{}",
        }

    monkeypatch.setattr(supervisor, "run_modifying_pm", fake_pm)
    monkeypatch.setattr(supervisor, "run_modifying_programmer", fake_programmer)

    result = await supervisor.run({
        "question": "Change VALUE to 2 and update tests.",
        "reading_result": {
            "answer_text": "app.py and tests/test_app.py are relevant.",
            "evidence": [
                {"path": "app.py", "summary": "Runtime owner."},
                {"path": "tests/test_app.py", "summary": "Focused test."},
            ],
        },
        "repository": {"local_root": str(tmp_path)},
        "source_scope": {"kind": "local_path"},
    })

    assert result["status"] == "succeeded"
    assert [row["path"] for row in result["changed_files"]] == [
        "app.py",
        "tests/test_app.py",
    ]
    assert len(pm_inputs) == 2
    assert pm_inputs[1]["repair_feedback"]["feedback_source"] == "handoff_validation"
    assert programmer_payloads[0]["programmer_task"]["task_id"] == "task-runtime"
    trace_summary = result["trace_summary"]
    assert "modifying:file_plan_ready" in trace_summary
    assert "modifying_pm:decision=create_programmer_task" in trace_summary
    assert "modifying_pm:programmer_task=task-runtime" in trace_summary
    assert "modifying_pm:sufficiency=programmer_artifacts_ready" in trace_summary


@pytest.mark.asyncio
async def test_modifying_supervisor_repairs_blocked_programmer_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_modifying import supervisor

    programmer_results = [
        {
            "artifacts": [
                {
                    "artifact_id": "artifact-test",
                    "status": "blocked",
                    "task_id": "task-runtime",
                    "target_path": "tests/test_app.py",
                    "blocker": "python imports must be top-level",
                }
            ],
            "answer_text": "",
            "limitations": [],
        },
        {
            "artifacts": [
                {
                    "artifact_id": "artifact-test",
                    "status": "succeeded",
                    "task_id": "task-runtime",
                    "target_path": "tests/test_app.py",
                    "evidence_ids": ["evidence-1"],
                    "operation_kind": "replace",
                    "exact_anchor": "import app\n",
                    "replacement_or_insert_content": "import json\n\nimport app\n",
                    "operation_summary": "Move JSON import to top level.",
                    "risk_notes": [],
                    "tests_or_docs_to_update": [],
                }
            ],
            "answer_text": "Prepared repaired test artifact.",
            "limitations": [],
        },
    ]
    payloads: list[dict[str, object]] = []

    async def fake_programmer(payload: dict[str, object]) -> dict[str, object]:
        payloads.append(payload)
        return programmer_results.pop(0)

    monkeypatch.setattr(supervisor, "run_modifying_programmer", fake_programmer)

    result, trace = await supervisor._run_programmer_with_contract_repair({
        "question": "Update test imports.",
        "file_contexts": [{"path": "tests/test_app.py"}],
    }, required_target_paths={"tests/test_app.py"})

    assert trace == ["modifying:programmer_contract_repair"]
    assert result["artifacts"][0]["status"] == "succeeded"
    assert len(payloads) == 2
    repair_feedback = payloads[1]["repair_feedback"]
    assert repair_feedback["feedback_source"] == "contract_validation"
    assert "python imports must be top-level" in (
        repair_feedback["validation"]["errors"][0]
    )


@pytest.mark.asyncio
async def test_modifying_supervisor_repairs_missing_target_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_modifying import supervisor

    programmer_results = [
        {
            "artifacts": [
                {
                    "artifact_id": "artifact-source",
                    "status": "succeeded",
                    "task_id": "task-runtime",
                    "target_path": "app.py",
                    "evidence_ids": ["evidence-1"],
                    "operation_kind": "replace",
                    "exact_anchor": "VALUE = 1\n",
                    "replacement_or_insert_content": "VALUE = 2\n",
                    "operation_summary": "Update source.",
                    "risk_notes": [],
                    "tests_or_docs_to_update": ["tests/test_app.py"],
                }
            ],
            "answer_text": "",
            "limitations": [],
        },
        {
            "artifacts": [
                {
                    "artifact_id": "artifact-source",
                    "status": "succeeded",
                    "task_id": "task-runtime",
                    "target_path": "app.py",
                    "evidence_ids": ["evidence-1"],
                    "operation_kind": "replace",
                    "exact_anchor": "VALUE = 1\n",
                    "replacement_or_insert_content": "VALUE = 2\n",
                    "operation_summary": "Update source.",
                    "risk_notes": [],
                    "tests_or_docs_to_update": [],
                },
                {
                    "artifact_id": "artifact-test",
                    "status": "succeeded",
                    "task_id": "task-runtime",
                    "target_path": "tests/test_app.py",
                    "evidence_ids": ["evidence-2"],
                    "operation_kind": "replace",
                    "exact_anchor": "assert VALUE == 1\n",
                    "replacement_or_insert_content": "assert VALUE == 2\n",
                    "operation_summary": "Update focused test.",
                    "risk_notes": [],
                    "tests_or_docs_to_update": [],
                },
            ],
            "answer_text": "Prepared repaired artifacts.",
            "limitations": [],
        },
    ]
    payloads: list[dict[str, object]] = []

    async def fake_programmer(payload: dict[str, object]) -> dict[str, object]:
        payloads.append(payload)
        return programmer_results.pop(0)

    monkeypatch.setattr(supervisor, "run_modifying_programmer", fake_programmer)

    result, trace = await supervisor._run_programmer_with_contract_repair(
        {
            "question": "Update source and test.",
            "file_contexts": [
                {"path": "app.py"},
                {"path": "tests/test_app.py"},
            ],
        },
        required_target_paths={"app.py", "tests/test_app.py"},
    )

    assert trace == ["modifying:programmer_contract_repair"]
    assert len(result["artifacts"]) == 2
    assert len(payloads) == 2
    repair_feedback = payloads[1]["repair_feedback"]
    assert repair_feedback["feedback_source"] == "contract_validation"
    assert "tests/test_app.py" in repair_feedback["validation"]["errors"][0]


@pytest.mark.asyncio
async def test_modifying_supervisor_repairs_unchanged_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_modifying import supervisor

    programmer_results = [
        {
            "artifacts": [
                {
                    "artifact_id": "artifact-source",
                    "status": "succeeded",
                    "task_id": "task-runtime",
                    "target_path": "app.py",
                    "evidence_ids": ["evidence-1"],
                    "operation_kind": "replace",
                    "exact_anchor": "VALUE = 1\n",
                    "replacement_or_insert_content": "VALUE = 1\n",
                    "operation_summary": "No-op.",
                    "risk_notes": [],
                    "tests_or_docs_to_update": [],
                }
            ],
            "answer_text": "",
            "limitations": [],
        },
        {
            "artifacts": [
                {
                    "artifact_id": "artifact-source",
                    "status": "succeeded",
                    "task_id": "task-runtime",
                    "target_path": "app.py",
                    "evidence_ids": ["evidence-1"],
                    "operation_kind": "replace",
                    "exact_anchor": "VALUE = 1\n",
                    "replacement_or_insert_content": "VALUE = 2\n",
                    "operation_summary": "Update source.",
                    "risk_notes": [],
                    "tests_or_docs_to_update": [],
                }
            ],
            "answer_text": "Prepared repaired artifact.",
            "limitations": [],
        },
    ]
    payloads: list[dict[str, object]] = []

    async def fake_programmer(payload: dict[str, object]) -> dict[str, object]:
        payloads.append(payload)
        return programmer_results.pop(0)

    monkeypatch.setattr(supervisor, "run_modifying_programmer", fake_programmer)

    result, trace = await supervisor._run_programmer_with_contract_repair(
        {
            "question": "Update source.",
            "file_contexts": [
                {"path": "app.py", "content": "VALUE = 1\n"},
            ],
        },
        required_target_paths={"app.py"},
    )

    assert trace == ["modifying:programmer_contract_repair"]
    assert result["artifacts"][0]["replacement_or_insert_content"] == "VALUE = 2\n"
    repair_feedback = payloads[1]["repair_feedback"]
    assert "unchanged" in repair_feedback["validation"]["errors"][0]
