def test_modifying_programmer_artifact_accepts_structured_replace() -> None:
    from kazusa_ai_chatbot.coding_agent.code_modifying.models import (
        normalize_modification_artifact,
    )

    artifact = normalize_modification_artifact({
        "artifact_id": "artifact-1",
        "status": "succeeded",
        "task_id": "task-1",
        "target_path": "app.py",
        "evidence_ids": ["ev-1"],
        "operation_kind": "replace",
        "exact_anchor": "VALUE = 1\n",
        "replacement_or_insert_content": "VALUE = 2\n",
        "operation_summary": "Update value.",
        "risk_notes": ["Low risk."],
        "tests_or_docs_to_update": ["tests/test_app.py"],
    })

    assert artifact["status"] == "succeeded"
    assert artifact["operation_kind"] == "replace"
    assert artifact["target_path"] == "app.py"


def test_modifying_programmer_artifact_accepts_create_file_projection() -> None:
    from kazusa_ai_chatbot.coding_agent.code_modifying.models import (
        artifact_to_patch_operation,
        normalize_modification_artifact,
    )

    artifact = normalize_modification_artifact({
        "artifact_id": "artifact-create-1",
        "status": "succeeded",
        "task_id": "task-1",
        "target_path": "counter_cli/formatters.py",
        "evidence_ids": ["ev-1"],
        "operation_kind": "create_file",
        "exact_anchor": "",
        "replacement_or_insert_content": (
            "def format_total(total: int) -> str:\n"
            "    return f'Total: {total}'\n"
        ),
        "operation_summary": "Create formatter helper.",
        "risk_notes": ["New runtime module must be imported by caller."],
        "tests_or_docs_to_update": ["tests/test_counter_cli.py"],
    })
    operation = artifact_to_patch_operation(artifact)

    assert artifact["status"] == "succeeded"
    assert artifact["operation_kind"] == "create_file"
    assert operation is not None
    assert operation["kind"] == "create_file"
    assert operation["path"] == "counter_cli/formatters.py"
    assert "anchor" not in operation


def test_modifying_programmer_artifact_rejects_raw_diff() -> None:
    from kazusa_ai_chatbot.coding_agent.code_modifying.models import (
        normalize_modification_artifact,
    )

    artifact = normalize_modification_artifact({
        "artifact_id": "artifact-1",
        "status": "succeeded",
        "task_id": "task-1",
        "target_path": "app.py",
        "evidence_ids": ["ev-1"],
        "operation_kind": "replace",
        "exact_anchor": "VALUE = 1\n",
        "replacement_or_insert_content": (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
        ),
        "operation_summary": "Raw diff.",
        "risk_notes": [],
        "tests_or_docs_to_update": [],
    })

    assert artifact["status"] == "blocked"
    assert "raw diff" in artifact["blocker"]


def test_modifying_programmer_artifact_rejects_indented_python_import() -> None:
    from kazusa_ai_chatbot.coding_agent.code_modifying.models import (
        normalize_modification_artifact,
    )

    artifact = normalize_modification_artifact({
        "artifact_id": "artifact-1",
        "status": "succeeded",
        "task_id": "task-1",
        "target_path": "tests/test_app.py",
        "evidence_ids": ["ev-1"],
        "operation_kind": "replace",
        "exact_anchor": "def test_app():\n    assert True\n",
        "replacement_or_insert_content": (
            "def test_app():\n"
            "    import json\n"
            "    assert json.loads('{}') == {}\n"
        ),
        "operation_summary": "Add JSON assertion.",
        "risk_notes": [],
        "tests_or_docs_to_update": [],
    })

    assert artifact["status"] == "blocked"
    assert "top-level" in artifact["blocker"]


def test_modifying_programmer_artifact_rejects_method_without_receiver() -> None:
    from kazusa_ai_chatbot.coding_agent.code_modifying.models import (
        normalize_modification_artifact,
    )

    artifact = normalize_modification_artifact({
        "artifact_id": "artifact-1",
        "status": "succeeded",
        "task_id": "task-1",
        "target_path": "store.py",
        "evidence_ids": ["ev-1"],
        "operation_kind": "replace",
        "exact_anchor": "    def list_items(self) -> list[str]:\n        return []\n",
        "replacement_or_insert_content": (
            "    def list_items(*, include_archived: bool = False) -> list[str]:\n"
            "        return list(self._items)\n"
        ),
        "operation_summary": "Update method.",
        "risk_notes": [],
        "tests_or_docs_to_update": [],
    })

    assert artifact["status"] == "blocked"
    assert "self or cls" in artifact["blocker"]


def test_modifying_pm_repair_rejects_executed_output_source() -> None:
    from kazusa_ai_chatbot.coding_agent.code_modifying.models import (
        normalize_modifying_pm_decision,
    )

    decision = normalize_modifying_pm_decision({
        "status": "repair_child",
        "reason": "The generated tests failed.",
        "owned_paths": ["app.py"],
        "read_only_paths": ["tests/test_app.py"],
        "required_evidence_ids": ["ev-1"],
        "repair_instruction": {
            "child_id": "task-1",
            "feedback_source": "executed_test_output",
            "feedback": "pytest failed.",
            "expected_correction": "Fix the test failure.",
        },
    })

    assert decision["status"] == "blocked"
    assert decision["blocker"] is not None
    assert "structural" in decision["blocker"]["summary"]


def test_modifying_pm_accepts_programmer_task_with_evidence() -> None:
    from kazusa_ai_chatbot.coding_agent.code_modifying.models import (
        normalize_modifying_pm_decision,
    )

    decision = normalize_modifying_pm_decision({
        "status": "create_programmer_task",
        "reason": "The source evidence identifies one target file.",
        "owned_paths": ["app.py"],
        "read_only_paths": ["tests/test_app.py"],
        "required_evidence_ids": ["ev-1"],
        "programmer_task": {
            "task_id": "task-1",
            "target_paths": ["app.py"],
            "change_goal": "Update VALUE.",
            "required_behavior": ["VALUE is 2."],
            "forbidden_changes": ["Do not edit tests in this task."],
            "consumed_interfaces": ["tests expect VALUE."],
            "expected_operations": ["replace"],
            "acceptance_checks": ["Tests should assert VALUE."],
            "local_risks": ["Repeated anchors."],
        },
    })

    assert decision["status"] == "create_programmer_task"
    assert decision["programmer_task"] is not None
    assert decision["programmer_task"]["task_id"] == "task-1"


def test_modifying_pm_accepts_mixed_create_and_edit_programmer_task() -> None:
    from kazusa_ai_chatbot.coding_agent.code_modifying.models import (
        normalize_modifying_pm_decision,
    )

    decision = normalize_modifying_pm_decision({
        "status": "create_programmer_task",
        "reason": "A new helper file and existing caller edit are both needed.",
        "owned_paths": ["counter_cli/main.py", "counter_cli/formatters.py"],
        "read_only_paths": ["tests/test_counter_cli.py"],
        "required_evidence_ids": ["ev-1"],
        "programmer_task": {
            "task_id": "task-1",
            "target_paths": ["counter_cli/main.py", "counter_cli/formatters.py"],
            "change_goal": "Create formatter helper and wire main.",
            "required_behavior": ["Output uses the shared formatter."],
            "forbidden_changes": ["Do not edit protected verification tests."],
            "consumed_interfaces": ["main.py currently prints totals."],
            "expected_operations": ["create_file", "replace"],
            "acceptance_checks": ["Run existing counter CLI tests."],
            "local_risks": ["Import path must match package layout."],
        },
    })

    assert decision["status"] == "create_programmer_task"
    assert decision["programmer_task"] is not None
    assert decision["programmer_task"]["expected_operations"] == [
        "create_file",
        "replace",
    ]
