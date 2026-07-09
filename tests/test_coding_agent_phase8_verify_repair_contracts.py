"""Deterministic contracts for controlled verify-and-repair orchestration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest


pytestmark = pytest.mark.asyncio

SOURCE_IDENTITY = {
    "provider": "github",
    "owner": "fixture",
    "repo": "demo",
    "current_commit": "abc123",
    "dirty_state": "clean",
}
SOURCE_SCOPE = {
    "kind": "repository",
    "repo_relative_path": None,
    "source_url": "local://github/fixture/demo",
    "requested_ref": "main",
    "interpretation": "test checkout",
}
INITIAL_ARTIFACT = {
    "artifact_id": "seed-app",
    "base": "repository",
    "diff_text": "diff --git a/app.py b/app.py\n",
    "files": ["app.py"],
    "summary": "Seed source change.",
}
REPAIRED_ARTIFACT = {
    "artifact_id": "repair-app",
    "base": "repository",
    "diff_text": "diff --git a/app.py b/app.py\n",
    "files": ["app.py"],
    "summary": "Repair source change.",
}
TEST_ARTIFACT = {
    "artifact_id": "repair-test",
    "base": "repository",
    "diff_text": "diff --git a/tests/test_app.py b/tests/test_app.py\n",
    "files": ["tests/test_app.py"],
    "summary": "Unsafe test edit.",
}
CLI_ARTIFACT = {
    "artifact_id": "repair-cli",
    "base": "repository",
    "diff_text": "diff --git a/cli.py b/cli.py\n",
    "files": ["cli.py"],
    "summary": "Repair CLI wiring.",
}


async def test_verify_repair_rejects_missing_approval(tmp_path: Path) -> None:
    """Verification cannot create managed apply storage without approval."""

    from kazusa_ai_chatbot.coding_agent import verify_and_repair_code_change

    response = await verify_and_repair_code_change(_request(tmp_path, approval=None))

    assert response["status"] == "rejected"
    assert response["attempts"] == []
    assert any("approval" in item.casefold() for item in response["limitations"])


async def test_verify_repair_rejects_source_free_request(tmp_path: Path) -> None:
    """Verification targets source-backed repair only."""

    from kazusa_ai_chatbot.coding_agent import verify_and_repair_code_change

    request = _request(tmp_path)
    request.pop("local_root_hint")

    response = await verify_and_repair_code_change(request)

    assert response["status"] == "rejected"
    assert response["attempts"] == []
    assert any("source" in item.casefold() for item in response["limitations"])


async def test_verify_repair_rejects_unsupported_execution_spec(
    tmp_path: Path,
) -> None:
    """Verifier checks execution tools before patch application."""

    from kazusa_ai_chatbot.coding_agent import verify_and_repair_code_change

    request = _request(tmp_path)
    request["execution_specs"] = [{
        "tool": "pip_install",
        "paths": ["app.py"],
        "pytest_selectors": [],
        "timeout_seconds": 10,
    }]

    response = await verify_and_repair_code_change(request)

    assert response["status"] == "rejected"
    assert response["attempts"] == []
    assert any("unsupported" in item.casefold() for item in response["limitations"])


async def test_verify_repair_omits_initial_protected_verification_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Initial proposal test edits are omitted before approved apply."""

    from kazusa_ai_chatbot.coding_agent.code_verifying import supervisor
    from kazusa_ai_chatbot.coding_agent import verify_and_repair_code_change

    source_root = _source_root(tmp_path)
    apply_requests: list[dict[str, object]] = []

    async def fake_fetch(request: dict[str, object]) -> dict[str, object]:
        return _fetching_result(source_root)

    def fake_apply(request: dict[str, object]) -> dict[str, object]:
        apply_requests.append(request)
        changed_paths = [
            path
            for artifact in request["patch_artifacts"]
            for path in artifact["files"]
        ]
        return _apply_response(
            apply_package_id="apply-protected-filter",
            changed_paths=changed_paths,
        )

    def fake_execute(request: dict[str, object]) -> dict[str, object]:
        return _execution_response(status="succeeded")

    monkeypatch.setattr(supervisor.code_fetching, "run", fake_fetch)
    monkeypatch.setattr(supervisor, "apply_approved_patch", fake_apply)
    monkeypatch.setattr(supervisor, "execute_code_check", fake_execute)

    request = _request(tmp_path)
    request["initial_patch_artifacts"] = [INITIAL_ARTIFACT, TEST_ARTIFACT]
    response = await verify_and_repair_code_change(request)

    assert response["status"] == "succeeded"
    assert len(apply_requests) == 1
    assert apply_requests[0]["patch_artifacts"] == [INITIAL_ARTIFACT]
    assert response["final_changed_files"] == [
        {
            "path": "app.py",
            "change_type": "modify",
            "summary": "Applied change.",
        }
    ]
    assert any(
        "Omitted protected verification path from approved apply: "
        "tests/test_app.py" in item
        for item in response["limitations"]
    )
    assert "verify_repair:protected_initial_artifacts_omitted count=1" in (
        response["trace_summary"]
    )


async def test_verify_repair_attempts_fresh_apply_and_preserves_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Failed execution triggers one repair in a fresh managed workspace."""

    from kazusa_ai_chatbot.coding_agent.code_verifying import supervisor
    from kazusa_ai_chatbot.coding_agent import verify_and_repair_code_change

    source_root = _source_root(tmp_path)
    before_text = (source_root / "app.py").read_text(encoding="utf-8")
    apply_package_ids = ["apply-one", "apply-two"]
    apply_requests: list[dict[str, object]] = []
    execution_requests: list[dict[str, object]] = []
    proposal_requests: list[dict[str, object]] = []

    async def fake_fetch(request: dict[str, object]) -> dict[str, object]:
        return _fetching_result(source_root)

    async def fake_propose(request: dict[str, object]) -> dict[str, object]:
        proposal_requests.append(request)
        return _proposal_response(
            patch_artifacts=[REPAIRED_ARTIFACT],
            changed_paths=["app.py"],
        )

    def fake_apply(request: dict[str, object]) -> dict[str, object]:
        apply_requests.append(request)
        apply_package_id = apply_package_ids.pop(0)
        return _apply_response(
            apply_package_id=apply_package_id,
            changed_paths=[
                path
                for artifact in request["patch_artifacts"]
                for path in artifact["files"]
            ],
        )

    def fake_execute(request: dict[str, object]) -> dict[str, object]:
        execution_requests.append(request)
        if len(execution_requests) == 1:
            return _execution_response(
                status="failed",
                stdout="FAILED tests/test_app.py::test_value\n"
                f"{source_root}\\app.py leaked path\n",
                stderr="AssertionError: VALUE should be 2\n",
            )
        return _execution_response(status="succeeded")

    monkeypatch.setattr(supervisor.code_fetching, "run", fake_fetch)
    monkeypatch.setattr(supervisor, "propose_code_change", fake_propose)
    monkeypatch.setattr(supervisor, "apply_approved_patch", fake_apply)
    monkeypatch.setattr(supervisor, "execute_code_check", fake_execute)

    response = await verify_and_repair_code_change(_request(tmp_path))

    assert response["status"] == "succeeded"
    assert len(response["attempts"]) == 2
    assert [row["apply_package_id"] for row in response["attempts"]] == [
        "apply-one",
        "apply-two",
    ]
    assert len(apply_requests) == 2
    assert len(execution_requests) == 2
    assert proposal_requests
    repair_feedback = proposal_requests[0]["repair_feedback"]
    assert repair_feedback["feedback_source"] == "execution_verification"
    assert repair_feedback["attempt_index"] == 1
    assert "tests/test_app.py" in json.dumps(repair_feedback)
    assert str(source_root.resolve()) not in json.dumps(
        repair_feedback,
        ensure_ascii=False,
    )
    assert (source_root / "app.py").read_text(encoding="utf-8") == before_text


async def test_verify_repair_stops_after_attempt_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Repair attempts are capped deterministically."""

    from kazusa_ai_chatbot.coding_agent.code_verifying import supervisor
    from kazusa_ai_chatbot.coding_agent import verify_and_repair_code_change

    source_root = _source_root(tmp_path)
    propose_count = 0

    async def fake_fetch(request: dict[str, object]) -> dict[str, object]:
        return _fetching_result(source_root)

    async def fake_propose(request: dict[str, object]) -> dict[str, object]:
        nonlocal propose_count
        propose_count += 1
        return _proposal_response(
            patch_artifacts=[REPAIRED_ARTIFACT],
            changed_paths=["app.py"],
        )

    def fake_apply(request: dict[str, object]) -> dict[str, object]:
        apply_package_id = f"apply-{len(request['patch_artifacts'])}-{propose_count}"
        return _apply_response(
            apply_package_id=apply_package_id,
            changed_paths=[
                path
                for artifact in request["patch_artifacts"]
                for path in artifact["files"]
            ],
        )

    def fake_execute(request: dict[str, object]) -> dict[str, object]:
        return _execution_response(status="failed")

    monkeypatch.setattr(supervisor.code_fetching, "run", fake_fetch)
    monkeypatch.setattr(supervisor, "propose_code_change", fake_propose)
    monkeypatch.setattr(supervisor, "apply_approved_patch", fake_apply)
    monkeypatch.setattr(supervisor, "execute_code_check", fake_execute)

    request = _request(tmp_path)
    request["repair_attempt_limit"] = 1
    response = await verify_and_repair_code_change(request)

    assert response["status"] == "failed"
    assert len(response["attempts"]) == 2
    assert propose_count == 1
    assert response["final_execution"][0]["status"] == "failed"


async def test_verify_repair_reports_terminal_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Timed-out final verification remains a timed-out terminal status."""

    from kazusa_ai_chatbot.coding_agent.code_verifying import supervisor
    from kazusa_ai_chatbot.coding_agent import verify_and_repair_code_change

    source_root = _source_root(tmp_path)

    async def fake_fetch(request: dict[str, object]) -> dict[str, object]:
        return _fetching_result(source_root)

    def fake_apply(request: dict[str, object]) -> dict[str, object]:
        return _apply_response(
            apply_package_id="apply-timeout",
            changed_paths=["app.py"],
        )

    def fake_execute(request: dict[str, object]) -> dict[str, object]:
        return _execution_response(status="timed_out")

    monkeypatch.setattr(supervisor.code_fetching, "run", fake_fetch)
    monkeypatch.setattr(supervisor, "apply_approved_patch", fake_apply)
    monkeypatch.setattr(supervisor, "execute_code_check", fake_execute)

    request = _request(tmp_path)
    request["repair_attempt_limit"] = 0
    response = await verify_and_repair_code_change(request)

    assert response["status"] == "timed_out"
    assert response["attempts"][0]["execution_statuses"] == ["timed_out"]


async def test_verify_repair_updates_required_paths_after_each_repair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Later repairs must preserve source paths introduced by prior repairs."""

    from kazusa_ai_chatbot.coding_agent.code_verifying import supervisor
    from kazusa_ai_chatbot.coding_agent import verify_and_repair_code_change

    source_root = _source_root(tmp_path)
    propose_count = 0
    apply_requests: list[dict[str, object]] = []

    async def fake_fetch(request: dict[str, object]) -> dict[str, object]:
        return _fetching_result(source_root)

    async def fake_propose(request: dict[str, object]) -> dict[str, object]:
        nonlocal propose_count
        propose_count += 1
        if propose_count == 1:
            return _proposal_response(
                patch_artifacts=[REPAIRED_ARTIFACT, CLI_ARTIFACT],
                changed_paths=["app.py", "cli.py"],
            )
        return _proposal_response(
            patch_artifacts=[REPAIRED_ARTIFACT],
            changed_paths=["app.py"],
        )

    def fake_apply(request: dict[str, object]) -> dict[str, object]:
        apply_requests.append(request)
        return _apply_response(
            apply_package_id=f"apply-{len(apply_requests)}",
            changed_paths=[
                path
                for artifact in request["patch_artifacts"]
                for path in artifact["files"]
            ],
        )

    def fake_execute(request: dict[str, object]) -> dict[str, object]:
        return _execution_response(status="failed")

    monkeypatch.setattr(supervisor.code_fetching, "run", fake_fetch)
    monkeypatch.setattr(supervisor, "propose_code_change", fake_propose)
    monkeypatch.setattr(supervisor, "apply_approved_patch", fake_apply)
    monkeypatch.setattr(supervisor, "execute_code_check", fake_execute)

    request = _request(tmp_path)
    request["repair_attempt_limit"] = 2
    response = await verify_and_repair_code_change(request)

    assert response["status"] == "failed"
    assert len(response["attempts"]) == 2
    assert len(apply_requests) == 2
    assert propose_count == 2
    assert "Repair proposal omitted required source owner path: cli.py" in (
        response["limitations"]
    )


async def test_verify_repair_feedback_preserves_failure_summary_for_repair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Repair feedback keeps failing tests and assertion summaries."""

    feedback = _feedback_with_failure(tmp_path)

    assert feedback["feedback_source"] == "execution_verification"
    assert "tests/test_cli.py::test_cli_flag" in feedback["failed_paths"]
    assert any("AssertionError" in item for item in feedback["failure_summaries"])


async def test_verify_repair_feedback_redacts_absolute_paths_and_raw_output(
    tmp_path: Path,
) -> None:
    """Repair feedback removes local roots and raw command noise."""

    feedback = _feedback_with_failure(tmp_path)
    serialized = json.dumps(feedback, ensure_ascii=False)

    assert str(tmp_path.resolve()) not in serialized
    assert "python -m pytest" not in serialized
    assert len(feedback["stdout_excerpt"]) < 600
    assert len(feedback["stderr_excerpt"]) < 600


async def test_verify_repair_request_includes_required_owner_and_context_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Repair proposal requests carry source-owner and test evidence paths."""

    from kazusa_ai_chatbot.coding_agent.code_verifying import supervisor

    feedback = _feedback_with_failure(tmp_path)
    request = supervisor.build_repair_proposal_request(
        base_request=_request(tmp_path),
        repository=_repository(tmp_path / "source"),
        source_scope=SOURCE_SCOPE,
        repair_feedback=feedback,
        previous_patch_artifacts=[INITIAL_ARTIFACT],
        required_source_owner_paths=["inventory_sync/fetch.py"],
        protected_verification_paths=["tests/test_cli.py"],
    )

    assert request["repair_feedback"]["required_source_owner_paths"] == [
        "inventory_sync/fetch.py",
    ]
    assert request["repair_feedback"]["protected_verification_paths"] == [
        "tests/test_cli.py",
    ]
    assert "tests/test_cli.py" in json.dumps(request["repair_feedback"])


async def test_verify_repair_rejects_repair_omitting_required_owner_paths(
    tmp_path: Path,
) -> None:
    """A repair proposal must continue touching required source owners."""

    from kazusa_ai_chatbot.coding_agent.code_verifying import supervisor

    errors = supervisor.validate_repair_proposal(
        proposal=_proposal_response(
            patch_artifacts=[TEST_ARTIFACT],
            changed_paths=["tests/test_app.py"],
        ),
        required_source_owner_paths=["app.py"],
        protected_verification_paths=[],
    )

    assert errors == ["Repair proposal omitted required source owner path: app.py"]


async def test_verify_repair_rejects_repair_modifying_protected_verification_tests(
    tmp_path: Path,
) -> None:
    """Repairs cannot pass by changing focused verification tests."""

    from kazusa_ai_chatbot.coding_agent.code_verifying import supervisor

    errors = supervisor.validate_repair_proposal(
        proposal=_proposal_response(
            patch_artifacts=[REPAIRED_ARTIFACT, TEST_ARTIFACT],
            changed_paths=["app.py", "tests/test_app.py"],
        ),
        required_source_owner_paths=["app.py"],
        protected_verification_paths=["tests/test_app.py"],
    )

    assert errors == [
        "Repair proposal modified protected verification path: tests/test_app.py",
    ]


async def test_verify_repair_compileall_paths_are_not_protected_targets(
    tmp_path: Path,
) -> None:
    """Compile targets are executable source paths, not read-only tests."""

    from kazusa_ai_chatbot.coding_agent.code_verifying import supervisor

    protected_paths = supervisor._protected_verification_paths([{
        "tool": "python_compileall",
        "paths": ["app.py"],
        "pytest_selectors": [],
        "timeout_seconds": 10,
    }])
    errors = supervisor.validate_repair_proposal(
        proposal=_proposal_response(
            patch_artifacts=[REPAIRED_ARTIFACT],
            changed_paths=["app.py"],
        ),
        required_source_owner_paths=["app.py"],
        protected_verification_paths=protected_paths,
    )

    assert protected_paths == []
    assert errors == []


async def test_execution_repair_handoff_keeps_verification_tests_read_only(
    tmp_path: Path,
) -> None:
    """Repair handoff targets source owners and reads focused tests as evidence."""

    from kazusa_ai_chatbot.coding_agent.code_modifying import supervisor

    repair_feedback = {
        "feedback_source": "execution_verification",
        "failed_paths": ["tests/test_stats_tools.py"],
        "failure_summaries": ["assert 8 == 6"],
        "required_source_owner_paths": ["stats_tools.py"],
        "protected_verification_paths": ["tests/test_stats_tools.py"],
    }
    file_plan = {
        "status": "accepted",
        "source_scope": SOURCE_SCOPE,
        "evidence": [],
        "file_contexts": [
            {
                "path": "stats_tools.py",
                "content": "def median(values):\n    return sorted(values)[len(values)//2]\n",
            },
            {
                "path": "tests/test_stats_tools.py",
                "content": "def test_median_even_length_averages_middle_pair():\n"
                "    assert median([10, 2, 4, 8]) == 6\n",
            },
        ],
        "owned_path_candidates": ["stats_tools.py"],
        "read_only_path_candidates": [],
        "caller_path_candidates": [],
        "test_or_doc_path_candidates": ["tests/test_stats_tools.py"],
        "missing_owner_signals": [],
        "limits": {},
    }
    invalid_task = {
        "task_id": "repair-test-instead-of-source",
        "target_paths": ["tests/test_stats_tools.py"],
        "objective": "Repair the failed median verification.",
        "required_behavior": [],
        "forbidden_changes": [],
    }

    errors = supervisor._handoff_validation_errors(
        task=invalid_task,
        decision={"read_only_paths": []},
        file_plan=file_plan,
        programmer_task_count=0,
        repair_feedback=repair_feedback,
    )
    feedback = supervisor._handoff_repair_feedback(
        task=invalid_task,
        handoff_errors=errors,
        file_plan=file_plan,
        repair_feedback=repair_feedback,
    )

    assert any("protected verification evidence" in item for item in errors)
    assert "Programmer task omitted required source-owner path 'stats_tools.py'." in (
        errors
    )
    assert feedback["required_source_owner_paths"] == ["stats_tools.py"]
    assert feedback["protected_verification_paths"] == ["tests/test_stats_tools.py"]
    assert feedback["allowed_source_target_paths"] == ["stats_tools.py"]

    valid_task = {
        "task_id": "repair-source",
        "target_paths": ["stats_tools.py"],
        "objective": "Repair the failed median verification.",
        "required_behavior": [],
        "forbidden_changes": [],
    }
    valid_errors = supervisor._handoff_validation_errors(
        task=valid_task,
        decision={"read_only_paths": []},
        file_plan=file_plan,
        programmer_task_count=0,
        repair_feedback=repair_feedback,
    )
    payload = supervisor._programmer_payload(
        request={
            "question": "Fix median.",
            "source_scope": SOURCE_SCOPE,
            "repair_feedback": repair_feedback,
        },
        reading_result={"answer_text": "Median even length fails."},
        file_plan=file_plan,
        decision={
            "status": "create_programmer_task",
            "programmer_task": valid_task,
            "read_only_paths": [],
        },
        task=valid_task,
    )
    context_paths = [context["path"] for context in payload["file_contexts"]]

    assert valid_errors == []
    assert context_paths == ["stats_tools.py", "tests/test_stats_tools.py"]


async def test_execution_repair_handoff_allows_caller_source_collaborators(
    tmp_path: Path,
) -> None:
    """Repair handoff can target caller wiring with required source owners."""

    from kazusa_ai_chatbot.coding_agent.code_modifying import supervisor

    repair_feedback = {
        "feedback_source": "execution_verification",
        "failed_paths": ["tests/test_cli.py"],
        "failure_summaries": ["unrecognized arguments: --ignore-case"],
        "required_source_owner_paths": ["wordcount/counter.py"],
        "protected_verification_paths": ["tests/test_counter.py", "tests/test_cli.py"],
    }
    file_plan = {
        "status": "accepted",
        "source_scope": SOURCE_SCOPE,
        "evidence": [],
        "file_contexts": [
            {
                "path": "wordcount/counter.py",
                "content": "def count_words(text, *, ignore_case=False):\n"
                "    return {}\n",
            },
            {
                "path": "wordcount/cli.py",
                "content": "def main(argv=None):\n    count_words(args.text)\n",
            },
            {
                "path": "tests/test_cli.py",
                "content": "def test_cli_ignore_case_flag(capsys):\n"
                "    main(['--ignore-case', 'Cat cat DOG'])\n",
            },
        ],
        "owned_path_candidates": ["wordcount/counter.py"],
        "read_only_path_candidates": [],
        "caller_path_candidates": ["wordcount/cli.py"],
        "test_or_doc_path_candidates": ["tests/test_cli.py"],
        "missing_owner_signals": [],
        "limits": {},
    }
    source_task = {
        "task_id": "repair-cli-wiring",
        "target_paths": ["wordcount/counter.py", "wordcount/cli.py"],
        "objective": "Wire the existing ignore_case behavior through the CLI.",
        "required_behavior": [],
        "forbidden_changes": [],
    }

    errors = supervisor._handoff_validation_errors(
        task=source_task,
        decision={"read_only_paths": []},
        file_plan=file_plan,
        programmer_task_count=0,
        repair_feedback=repair_feedback,
    )
    payload = supervisor._programmer_payload(
        request={
            "question": "Wire --ignore-case.",
            "source_scope": SOURCE_SCOPE,
            "repair_feedback": repair_feedback,
        },
        reading_result={"answer_text": "CLI flag parsing fails."},
        file_plan=file_plan,
        decision={
            "status": "create_programmer_task",
            "programmer_task": source_task,
            "read_only_paths": [],
        },
        task=source_task,
    )
    context_paths = [context["path"] for context in payload["file_contexts"]]

    assert errors == []
    assert context_paths == [
        "wordcount/counter.py",
        "wordcount/cli.py",
        "tests/test_cli.py",
    ]


async def test_execution_repair_feedback_lists_allowed_source_targets(
    tmp_path: Path,
) -> None:
    """Invalid doc targets get repaired toward source owners and callers."""

    from kazusa_ai_chatbot.coding_agent.code_modifying import supervisor

    repair_feedback = {
        "feedback_source": "execution_verification",
        "failed_paths": ["tests/test_fetch.py", "tests/test_cli.py"],
        "failure_summaries": ["fetch_page() got an unexpected keyword argument"],
        "required_source_owner_paths": ["inventory_sync/fetch.py"],
        "protected_verification_paths": ["tests/test_fetch.py", "tests/test_cli.py"],
    }
    file_plan = {
        "status": "accepted",
        "source_scope": SOURCE_SCOPE,
        "evidence": [],
        "file_contexts": [
            {
                "path": "README.md",
                "content": "# Inventory Sync\n",
            },
            {
                "path": "inventory_sync/fetch.py",
                "content": "def fetch_page(url, *, timeout=10):\n    return ''\n",
            },
            {
                "path": "inventory_sync/cli.py",
                "content": "def main(argv=None):\n    print(fetch_page(args.url))\n",
            },
            {
                "path": "tests/test_fetch.py",
                "content": "def test_fetch_uses_cache_without_second_network_call():\n"
                "    pass\n",
            },
        ],
        "owned_path_candidates": ["inventory_sync/fetch.py"],
        "read_only_path_candidates": [],
        "caller_path_candidates": ["inventory_sync/cli.py"],
        "test_or_doc_path_candidates": ["README.md", "tests/test_fetch.py"],
        "missing_owner_signals": [],
        "limits": {},
    }
    invalid_task = {
        "task_id": "repair-doc",
        "target_paths": ["inventory_sync/fetch.py", "README.md"],
        "objective": "Repair fetch cache behavior.",
        "required_behavior": [],
        "forbidden_changes": [],
    }

    errors = supervisor._handoff_validation_errors(
        task=invalid_task,
        decision={"read_only_paths": []},
        file_plan=file_plan,
        programmer_task_count=0,
        repair_feedback=repair_feedback,
    )
    feedback = supervisor._handoff_repair_feedback(
        task=invalid_task,
        handoff_errors=errors,
        file_plan=file_plan,
        repair_feedback=repair_feedback,
    )

    assert "Programmer target path 'README.md' is not handoff-owned." in errors
    assert feedback["allowed_source_target_paths"] == [
        "inventory_sync/cli.py",
        "inventory_sync/fetch.py",
    ]
    assert "README.md" not in feedback["allowed_source_target_paths"]

    valid_task = {
        "task_id": "repair-source",
        "target_paths": ["inventory_sync/fetch.py", "inventory_sync/cli.py"],
        "objective": "Repair fetch cache behavior and CLI wiring.",
        "required_behavior": [],
        "forbidden_changes": [],
    }
    valid_errors = supervisor._handoff_validation_errors(
        task=valid_task,
        decision={"read_only_paths": []},
        file_plan=file_plan,
        programmer_task_count=0,
        repair_feedback=repair_feedback,
    )

    assert valid_errors == []


def _feedback_with_failure(tmp_path: Path) -> dict[str, object]:
    from kazusa_ai_chatbot.coding_agent.code_verifying import supervisor

    execution = _execution_response(
        status="failed",
        stdout=(
            "python -m pytest tests/test_cli.py\n"
            "FAILED tests/test_cli.py::test_cli_flag\n"
            f"{tmp_path.resolve()}\\source\\inventory_sync\\fetch.py\n"
        ),
        stderr="AssertionError: --timeout was not forwarded\n",
    )
    feedback = supervisor.build_execution_repair_feedback(
        attempt_index=1,
        execution_results=[execution],
        workspace_root=tmp_path / "workspace",
        source_root=tmp_path / "source",
        max_chars=1200,
    )
    return feedback


def _request(
    tmp_path: Path,
    *,
    approval: dict[str, object] | None | bool = True,
) -> dict[str, object]:
    request: dict[str, object] = {
        "question": "Repair VALUE and keep tests unchanged.",
        "local_root_hint": str(tmp_path / "source"),
        "source_scope_hint": "repository",
        "workspace_root": str(tmp_path / "workspace"),
        "preferred_language": "English",
        "max_answer_chars": 2000,
        "max_artifact_chars": 8000,
        "execution_specs": [{
            "tool": "pytest",
            "paths": [],
            "pytest_selectors": ["tests/test_app.py"],
            "timeout_seconds": 10,
        }],
        "repair_attempt_limit": 1,
        "max_repair_feedback_chars": 1200,
        "initial_patch_artifacts": [INITIAL_ARTIFACT],
        "expected_source_identity": SOURCE_IDENTITY,
    }
    if approval is True:
        request["approval"] = _approval()
    elif isinstance(approval, dict):
        request["approval"] = approval
    return request


def _approval() -> dict[str, object]:
    approval = {
        "approved": True,
        "approved_by": "verify-repair-contract",
        "approved_at": "2026-07-08T00:00:00Z",
        "approval_reason": "Deterministic verify repair contract.",
    }
    return approval


def _source_root(tmp_path: Path) -> Path:
    source_root = tmp_path / "source"
    source_root.mkdir(exist_ok=True)
    (source_root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    return source_root


def _repository(source_root: Path) -> dict[str, object]:
    repository = {
        **SOURCE_IDENTITY,
        "source_url": "https://github.com/fixture/demo",
        "requested_ref": None,
        "resolved_ref": "main",
        "default_branch": "main",
        "local_root": str(source_root),
        "storage_kind": "existing_local_checkout",
        "managed_checkout": False,
        "workspace_root": None,
        "cache_key": None,
    }
    return repository


def _fetching_result(source_root: Path) -> dict[str, object]:
    result = {
        "status": "succeeded",
        "message": "resolved",
        "repository": _repository(source_root),
        "source_scope": SOURCE_SCOPE,
        "limitations": [],
        "trace_summary": ["fetching:resolved"],
    }
    return result


def _proposal_response(
    *,
    patch_artifacts: list[dict[str, object]],
    changed_paths: list[str],
) -> dict[str, object]:
    response = {
        "status": "succeeded",
        "mode": "edit_existing_repository",
        "answer_text": "Prepared repair proposal.",
        "repository": None,
        "source_scope": SOURCE_SCOPE,
        "evidence": [],
        "patch_artifacts": patch_artifacts,
        "created_files": [],
        "changed_files": [
            {
                "path": path,
                "change_type": "modify",
                "summary": "Changed file.",
            }
            for path in changed_paths
        ],
        "validation": {
            "status": "succeeded",
            "parsed": True,
            "sandbox_applied": True,
            "errors": [],
            "warnings": [],
            "files": changed_paths,
        },
        "external_evidence": [],
        "session": None,
        "limitations": [],
        "trace_summary": ["proposal:succeeded"],
    }
    return response


def _apply_response(
    *,
    apply_package_id: str,
    changed_paths: list[str],
) -> dict[str, object]:
    response = {
        "status": "succeeded",
        "apply_package_id": apply_package_id,
        "source_identity": SOURCE_IDENTITY,
        "apply_workspace_ref": {
            "kind": "managed_apply_workspace",
            "apply_package_id": apply_package_id,
            "source_identity": SOURCE_IDENTITY,
            "applied_files": changed_paths,
        },
        "applied_files": changed_paths,
        "changed_files": [
            {
                "path": path,
                "change_type": "modify",
                "summary": "Applied change.",
            }
            for path in changed_paths
        ],
        "validation": {
            "status": "succeeded",
            "errors": [],
            "warnings": [],
        },
        "limitations": [],
        "trace_summary": ["patch_apply:succeeded"],
    }
    return response


def _execution_response(
    *,
    status: str,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, object]:
    response = {
        "status": status,
        "tool": "pytest",
        "exit_code": 1 if status == "failed" else 0,
        "timed_out": status == "timed_out",
        "duration_ms": 15,
        "stdout_excerpt": stdout,
        "stderr_excerpt": stderr,
        "output_truncated": False,
        "executed_paths": ["tests/test_app.py"],
        "limitations": [],
        "trace_summary": [f"code_execution:status={status}"],
    }
    if status == "timed_out":
        response["exit_code"] = None
    return response
