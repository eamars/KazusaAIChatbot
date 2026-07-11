"""Private evaluation lifecycle contracts for the Phase D action loop."""

import json
import hashlib
import subprocess
from contextlib import asynccontextmanager

import pytest


@pytest.mark.asyncio
async def test_private_evaluation_continuation_never_changes_public_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the engine selector inside the private evaluation boundary."""

    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    captured_request: dict[str, object] = {}

    async def continued(request: dict[str, object]) -> dict[str, object]:
        captured_request.update(request)
        return {"status": "completed", "run_id": "run-one"}

    monkeypatch.setattr(evaluation, "continue_coding_run", continued)
    response = await evaluation.continue_evaluation_coding_run(
        {"workspace_root": "workspace", "run_id": "run-one", "action": "status"},
        engine_id="pipeline_v1",
    )

    assert response["status"] == "completed"
    assert captured_request["run_id"] == "run-one"


@pytest.mark.asyncio
async def test_private_action_loop_start_persists_an_isolated_finish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Exercise private start, controller dispatch, and loop-state projection."""

    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    async def finished_controller(**_kwargs) -> dict[str, object]:
        return {
            "status": "ok",
            "action": {
                "schema_version": "coding_action.v1",
                "action_id": "finish-one",
                "action": "finish",
                "reason": "sufficient evidence",
                "args": {
                    "summary": "Source-grounded answer is complete.",
                    "acceptance_criteria": [],
                    "evidence_refs": [],
                    "known_limitations": [],
                },
            },
        }

    monkeypatch.setattr(evaluation, "invoke_controller", finished_controller)
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/fixture/demo.git"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=source_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Contract Test",
            "-c",
            "user.email=contract@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source_root,
        check=True,
    )
    response = await evaluation.run_evaluation_coding_run(
        {
            "workspace_root": str(tmp_path / "workspace"),
            "question": "Read the source.",
            "objective_type": "read_only",
            "local_root_hint": str(source_root),
            "source_scope_hint": "repository",
        },
        engine_id="action_loop_v1",
    )

    assert response["status"] == "completed"
    assert (
        tmp_path
        / "workspace"
        / "coding_runs"
        / response["run_id"]
        / "action_loop"
        / "state.json"
    ).is_file()


@pytest.mark.asyncio
async def test_private_action_loop_lock_contention_mutates_no_run_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Stop before source, index, candidate, or run-state mutation on contention."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop import supervisor
    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    @asynccontextmanager
    async def unavailable_lock(**_kwargs):
        yield False

    monkeypatch.setattr(supervisor, "acquire_workspace_locks", unavailable_lock)
    workspace_root = tmp_path / "workspace"
    result = await evaluation.run_evaluation_coding_run(
        {
            "workspace_root": str(workspace_root),
            "question": "Read the source.",
            "objective_type": "read_only",
            "local_root_hint": str(tmp_path / "missing-source"),
        },
        engine_id="action_loop_v1",
    )

    assert result["status"] == "blocked"
    assert result["blocker"]["code"] == "coding_run_lock_unavailable"
    assert not (workspace_root / "coding_runs").exists()
    assert not (workspace_root / "repository_indexes").exists()


@pytest.mark.asyncio
async def test_private_action_loop_iterates_over_a_pinned_source_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Resolve, index, search, and finish through repeated controller turns."""

    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "module.py").write_text(
        "DISCOVERABLE_VALUE = 42\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/fixture/demo.git"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=source_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Contract Test",
            "-c",
            "user.email=contract@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source_root,
        check=True,
    )
    controller_contexts: list[dict[str, object]] = []

    async def iterative_controller(**kwargs) -> dict[str, object]:
        context = json.loads(kwargs["context"])
        controller_contexts.append(context)
        if len(controller_contexts) == 1:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "search-one",
                "action": "search",
                "reason": "Find the relevant source.",
                "args": {"mode": "literal", "query": "DISCOVERABLE_VALUE"},
            }
        else:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "finish-one",
                "action": "finish",
                "reason": "The indexed evidence answers the task.",
                "args": {
                    "summary": "The value is defined in module.py.",
                    "acceptance_criteria": [],
                    "evidence_refs": ["module.py:1"],
                    "known_limitations": [],
                },
            }
        return {"status": "ok", "action": action}

    monkeypatch.setattr(evaluation, "invoke_controller", iterative_controller)
    response = await evaluation.run_evaluation_coding_run(
        {
            "workspace_root": str(tmp_path / "workspace"),
            "question": "Where is DISCOVERABLE_VALUE defined?",
            "objective_type": "read_only",
            "local_root_hint": str(source_root),
            "source_scope_hint": "repository",
        },
        engine_id="action_loop_v1",
    )

    assert response["status"] == "completed"
    assert response["answer_text"] == "The value is defined in module.py."
    assert len(controller_contexts) == 2
    assert controller_contexts[1]["observations"][0]["kind"] == "search_result"
    run_root = tmp_path / "workspace" / "coding_runs" / response["run_id"]
    run_state = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    assert run_state["index_snapshot_id"] != "evaluation-snapshot"
    assert run_state["source_request"]["local_root_hint"] == str(source_root)
    pins_path = next(
        (tmp_path / "workspace" / "repository_indexes").rglob("pins.json"),
    )
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    assert pins[run_state["index_snapshot_id"]] == [response["run_id"]]


@pytest.mark.asyncio
async def test_no_progress_budget_blocks_on_third_repeat_and_resets_on_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Bound repeated identical observations without losing durable history."""

    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/fixture/demo.git"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=source_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Contract Test",
            "-c",
            "user.email=contract@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source_root,
        check=True,
    )
    call_count = 0

    async def controller(**_kwargs) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": f"repeat-{call_count}",
                "action": "search",
                "reason": "Locate the requested symbol.",
                "args": {"mode": "literal", "query": "NOT_PRESENT"},
                "working_note": f"attempt {call_count}",
            }
        else:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "finish-after-resume",
                "action": "finish",
                "reason": "The repeated search blocker was resolved by retry.",
                "args": {
                    "summary": "No matching symbol exists in the source.",
                    "acceptance_criteria": [],
                    "evidence_refs": [],
                    "known_limitations": [],
                },
            }
        return {"status": "ok", "action": action}

    monkeypatch.setattr(evaluation, "invoke_controller", controller)
    workspace_root = tmp_path / "workspace"
    started = await evaluation.run_evaluation_coding_run(
        {
            "workspace_root": str(workspace_root),
            "question": "Find NOT_PRESENT.",
            "objective_type": "read_only",
            "local_root_hint": str(source_root),
            "source_scope_hint": "repository",
        },
        engine_id="action_loop_v1",
    )

    assert started["status"] == "blocked"
    assert call_count == 3
    assert started["blocker"] == {
        "blocker_type": "budget",
        "code": "controller_no_progress_budget_exhausted",
        "resume_target": "retry_loop",
        "latest_safe_evidence": {
            "outcome": "ok",
            "kind": "search_result",
            "evidence": [],
        },
    }
    run_root = workspace_root / "coding_runs" / started["run_id"]
    blocked_state = json.loads(
        (run_root / "action_loop" / "state.json").read_text(encoding="utf-8")
    )
    assert blocked_state["consecutive_no_progress_count"] == 3

    resumed = await evaluation.continue_evaluation_coding_run(
        {
            "workspace_root": str(workspace_root),
            "run_id": started["run_id"],
            "action": "respond_to_blocker",
            "revision_instruction": "Continue with the same source evidence.",
        },
        engine_id="action_loop_v1",
    )

    assert resumed["status"] == "completed"
    assert call_count == 4
    resumed_state = json.loads(
        (run_root / "action_loop" / "state.json").read_text(encoding="utf-8")
    )
    assert resumed_state["consecutive_no_progress_count"] == 1


def test_no_progress_signature_changes_with_query_cursor_evidence_or_revision() -> None:
    """Avoid conflating materially different exploration observations."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.supervisor import (
        _record_no_progress_signature,
    )

    loop_state: dict[str, object] = {"consecutive_no_progress_count": 0}
    action = {"action": "search", "args": {"mode": "literal", "query": "one"}}
    observation = {"outcome": "ok", "kind": "search_result", "evidence": []}

    assert not _record_no_progress_signature(
        loop_state=loop_state,
        action=action,
        observation=observation,
        candidate_revision_before=0,
    )
    assert loop_state["consecutive_no_progress_count"] == 1
    assert not _record_no_progress_signature(
        loop_state=loop_state,
        action={"action": "search", "args": {"mode": "literal", "query": "two"}},
        observation=observation,
        candidate_revision_before=0,
    )
    assert loop_state["consecutive_no_progress_count"] == 1
    assert not _record_no_progress_signature(
        loop_state=loop_state,
        action={
            "action": "search",
            "args": {"mode": "literal", "query": "two", "cursor": "next"},
        },
        observation=observation,
        candidate_revision_before=0,
    )
    assert loop_state["consecutive_no_progress_count"] == 1
    assert not _record_no_progress_signature(
        loop_state=loop_state,
        action={"action": "search", "args": {"mode": "literal", "query": "two"}},
        observation={
            "outcome": "ok",
            "kind": "search_result",
            "evidence": [{"repo_path": "module.py", "content_sha256": "a" * 64}],
        },
        candidate_revision_before=0,
    )
    assert loop_state["consecutive_no_progress_count"] == 1
    assert not _record_no_progress_signature(
        loop_state=loop_state,
        action={"action": "search", "args": {"mode": "literal", "query": "two"}},
        observation={
            "outcome": "ok",
            "kind": "search_result",
            "evidence": [{"repo_path": "module.py", "content_sha256": "a" * 64}],
        },
        candidate_revision_before=1,
    )
    assert loop_state["consecutive_no_progress_count"] == 1


@pytest.mark.asyncio
async def test_private_action_loop_edits_candidate_and_materializes_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Bind a candidate edit to observed hash/revision and review artifacts."""

    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    source_root = tmp_path / "source"
    source_root.mkdir()
    original_content = "VALUE = 1\n"
    replacement_content = "VALUE = 2\n"
    (source_root / "module.py").write_text(original_content, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/fixture/edit.git"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=source_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Contract Test",
            "-c",
            "user.email=contract@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source_root,
        check=True,
    )
    action_number = 0

    async def editing_controller(**_kwargs) -> dict[str, object]:
        nonlocal action_number
        action_number += 1
        if action_number == 1:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "search-one",
                "action": "search",
                "reason": "Find the source owner.",
                "args": {"mode": "literal", "query": "VALUE"},
            }
        elif action_number == 2:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "edit-one",
                "action": "edit",
                "reason": "Apply the requested source correction.",
                "args": {
                    "operation": "replace_file_small",
                    "repo_path": "module.py",
                    "expected_sha256": hashlib.sha256(
                        original_content.encode("utf-8")
                    ).hexdigest(),
                    "expected_candidate_revision": 0,
                    "replacement": replacement_content,
                },
            }
        else:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "finish-one",
                "action": "finish",
                "reason": "The candidate satisfies the requested change.",
                "args": {
                    "summary": "Updated module.py.",
                    "acceptance_criteria": ["VALUE is now 2."],
                    "evidence_refs": ["module.py:1"],
                    "known_limitations": [],
                },
            }
        return {"status": "ok", "action": action}

    monkeypatch.setattr(evaluation, "invoke_controller", editing_controller)
    workspace_root = tmp_path / "workspace"
    response = await evaluation.run_evaluation_coding_run(
        {
            "workspace_root": str(workspace_root),
            "question": "Change VALUE to 2.",
            "objective_type": "propose_patch",
            "local_root_hint": str(source_root),
            "source_scope_hint": "repository",
        },
        engine_id="action_loop_v1",
    )

    assert response["status"] == "awaiting_approval"
    assert response["patch_artifacts"]
    assert response["proposal_digest"]
    assert (source_root / "module.py").read_text(encoding="utf-8") == original_content
    run_root = workspace_root / "coding_runs" / response["run_id"]
    assert (run_root / "candidate" / "source" / "module.py").read_text(
        encoding="utf-8"
    ) == replacement_content
    run_state = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    assert run_state["candidate_revision"] == 1
    assert run_state["proposal_digest"] == response["proposal_digest"]


@pytest.mark.asyncio
async def test_private_action_loop_continuation_persists_structured_approval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Apply one reviewed source-free candidate through private continuation."""

    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    action_number = 0

    async def source_free_controller(**_kwargs) -> dict[str, object]:
        nonlocal action_number
        action_number += 1
        if action_number == 1:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "create-one",
                "action": "edit",
                "reason": "Create the requested source file.",
                "args": {
                    "operation": "create_file",
                    "repo_path": "app.py",
                    "expected_candidate_revision": 0,
                    "replacement": "print('ready')\n",
                },
            }
        else:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "finish-one",
                "action": "finish",
                "reason": "The requested artifact is ready for review.",
                "args": {
                    "summary": "Created app.py.",
                    "acceptance_criteria": ["app.py exists."],
                    "evidence_refs": ["app.py:1"],
                    "known_limitations": [],
                },
            }
        return {"status": "ok", "action": action}

    monkeypatch.setattr(evaluation, "invoke_controller", source_free_controller)
    workspace_root = tmp_path / "workspace"
    started = await evaluation.run_evaluation_coding_run(
        {
            "workspace_root": str(workspace_root),
            "question": "Create a minimal app.py.",
            "objective_type": "propose_patch",
        },
        engine_id="action_loop_v1",
    )
    assert started["status"] == "awaiting_approval"
    response = await evaluation.continue_evaluation_coding_run(
        {
            "workspace_root": str(workspace_root),
            "run_id": started["run_id"],
            "action": "approve_and_verify",
            "approval": {
                "approved": True,
                "approved_by": "benchmark-user",
                "approved_at": "2026-07-11T00:00:00Z",
                "approval_reason": "Benchmark approval.",
                "approval_evidence": {
                    "source_message_id": "approval-one",
                },
            },
        },
        engine_id="action_loop_v1",
    )

    assert response["status"] == "completed"
    loop_root = workspace_root / "coding_runs" / started["run_id"] / "action_loop"
    state = json.loads((loop_root / "state.json").read_text(encoding="utf-8"))
    assert state["approvals"][0]["approved_by"] == "benchmark-user"
    assert state["apply_attempts"][0]["status"] == "succeeded"


@pytest.mark.asyncio
async def test_private_blocker_resume_preserves_verbatim_user_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Resume the same pinned loop while leaving answer semantics to the model."""

    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/fixture/block.git"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=source_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Contract Test",
            "-c",
            "user.email=contract@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source_root,
        check=True,
    )
    contexts: list[dict[str, object]] = []

    async def blocking_controller(**kwargs) -> dict[str, object]:
        contexts.append(json.loads(kwargs["context"]))
        if len(contexts) == 1:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "block-one",
                "action": "block",
                "reason": "A user choice is required.",
                "args": {
                    "blocker_type": "needs_user_input",
                    "question": "Which behavior should be documented?",
                    "options": ["current", "new"],
                    "blocking_evidence_refs": [],
                },
            }
        else:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "finish-one",
                "action": "finish",
                "reason": "The user supplied the missing semantic choice.",
                "args": {
                    "summary": "The selected behavior is documented.",
                    "acceptance_criteria": [],
                    "evidence_refs": ["module.py:1"],
                    "known_limitations": [],
                },
            }
        return {"status": "ok", "action": action}

    monkeypatch.setattr(evaluation, "invoke_controller", blocking_controller)
    workspace_root = tmp_path / "workspace"
    started = await evaluation.run_evaluation_coding_run(
        {
            "workspace_root": str(workspace_root),
            "question": "Document the selected behavior.",
            "objective_type": "read_only",
            "local_root_hint": str(source_root),
        },
        engine_id="action_loop_v1",
    )
    assert started["status"] == "blocked"
    assert started["blocker"]["resume_target"] == "retry_loop"
    answer = 'Use 新 behavior exactly; keep `VALUE` unchanged.'
    resumed = await evaluation.continue_evaluation_coding_run(
        {
            "workspace_root": str(workspace_root),
            "run_id": started["run_id"],
            "action": "respond_to_blocker",
            "revision_instruction": answer,
        },
        engine_id="action_loop_v1",
    )

    assert resumed["status"] == "completed"
    assert contexts[1]["observations"][-1]["evidence"][0]["user_answer"] == answer
    state = json.loads(
        (
            workspace_root
            / "coding_runs"
            / started["run_id"]
            / "action_loop"
            / "state.json"
        ).read_text(encoding="utf-8")
    )
    assert state["candidate_revision"] == 0


@pytest.mark.asyncio
async def test_blocker_continuation_resets_segment_counters_and_preserves_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Keep prior evidence while starting fresh per-segment budget counters."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import (
        CandidateState,
    )
    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    run_root = tmp_path / "workspace" / "coding_runs" / "run-one"
    loop_root = run_root / "action_loop"
    loop_root.mkdir(parents=True)
    CandidateState.create(run_root / "candidate")
    state = {
        "run_id": "run-one",
        "objective_type": "read_only",
        "status": "blocked",
        "goal": "Read.",
        "acceptance_criteria": [],
        "source_identity_digest": "x",
        "index_snapshot_id": "missing",
        "candidate_revision": 0,
        "changed_paths": [],
        "current_failure": None,
        "working_note": "",
        "observations": [{"sequence": 1, "kind": "prior"}],
        "observation_count": 1,
        "action_count": 1,
        "invalid_output_count": 2,
        "consecutive_no_progress_count": 2,
        "run_action_count": 5,
        "segment_started_at_epoch_seconds": 1,
        "source_request": {},
        "repository": None,
        "source_scope": None,
        "blocker": {
            "resume_target": "retry_loop",
            "evidence": "prior",
        },
    }
    (loop_root / "state.json").write_text(json.dumps(state), encoding="utf-8")

    async def finish(**_kwargs: object) -> dict[str, object]:
        return {
            "status": "ok",
            "action": {
                "schema_version": "coding_action.v1",
                "action_id": "f",
                "action": "finish",
                "reason": "done",
                "args": {
                    "summary": "done",
                    "acceptance_criteria": [],
                    "evidence_refs": [],
                    "known_limitations": [],
                },
            },
        }

    monkeypatch.setattr(evaluation, "invoke_controller", finish)
    result = await evaluation.continue_evaluation_coding_run(
        {
            "workspace_root": str(tmp_path / "workspace"),
            "run_id": "run-one",
            "action": "respond_to_blocker",
            "revision_instruction": "answer",
        },
        engine_id="action_loop_v1",
    )
    saved = json.loads((loop_root / "state.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert saved["observations"][0]["kind"] == "prior"
    assert any(
        row.get("kind") == "user_blocker_response"
        for row in saved["observations"]
    )
    assert saved["invalid_output_count"] == 0
    assert saved["run_action_count"] == 0


@pytest.mark.asyncio
async def test_failed_verification_runs_current_repair_before_new_approval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Run repaired candidate evidence without reusing its prior apply approval."""

    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    source_root = tmp_path / "source"
    source_root.mkdir()
    original_content = "VALUE = 1\n"
    invalid_content = "VALUE = 2\n"
    repaired_content = "VALUE = 3\n"
    (source_root / "module.py").write_text(original_content, encoding="utf-8")
    tests_root = source_root / "tests"
    tests_root.mkdir()
    (tests_root / "test_module.py").write_text(
        "from module import VALUE\n\n\n"
        "def test_value() -> None:\n"
        "    assert VALUE == 3\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/fixture/fail.git"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=source_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Contract Test",
            "-c",
            "user.email=contract@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source_root,
        check=True,
    )
    action_number = 0

    async def repair_controller(**_kwargs) -> dict[str, object]:
        nonlocal action_number
        action_number += 1
        if action_number == 1:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "edit-one",
                "action": "edit",
                "reason": "Apply the candidate edit.",
                "args": {
                    "operation": "replace_file_small",
                    "repo_path": "module.py",
                    "expected_sha256": hashlib.sha256(
                        original_content.encode("utf-8")
                    ).hexdigest(),
                    "expected_candidate_revision": 0,
                    "replacement": invalid_content,
                },
            }
        elif action_number == 2:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "finish-one",
                "action": "finish",
                "reason": "Submit the candidate for verification.",
                "args": {
                    "summary": "Updated module.py.",
                    "acceptance_criteria": [],
                    "evidence_refs": ["module.py:1"],
                    "known_limitations": [],
                },
            }
        elif action_number == 3:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "edit-repair",
                "action": "edit",
                "reason": "Repair the candidate using the failed verification.",
                "args": {
                    "operation": "replace_file_small",
                    "repo_path": "module.py",
                    "expected_sha256": hashlib.sha256(
                        invalid_content.encode("utf-8")
                    ).hexdigest(),
                    "expected_candidate_revision": 1,
                    "replacement": repaired_content,
                },
            }
        elif action_number == 4:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "run-after-failure",
                "action": "run",
                "reason": "Run the approved focused check against candidate v2.",
                "args": {
                    "profile": "focused",
                    "targets": ["tests/test_module.py"],
                    "intent": "Confirm the repaired candidate behavior.",
                },
            }
        else:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "finish-repaired-candidate",
                "action": "finish",
                "reason": "Submit the repaired candidate for a new approval.",
                "args": {
                    "summary": "Repaired module.py and verified candidate v2.",
                    "acceptance_criteria": [],
                    "evidence_refs": ["module.py:1"],
                    "known_limitations": [],
                },
            }
        return {"status": "ok", "action": action}

    monkeypatch.setattr(evaluation, "invoke_controller", repair_controller)
    workspace_root = tmp_path / "workspace"
    started = await evaluation.run_evaluation_coding_run(
        {
            "workspace_root": str(workspace_root),
            "question": "Update module.py.",
            "objective_type": "propose_patch",
            "local_root_hint": str(source_root),
        },
        engine_id="action_loop_v1",
    )
    approval = {
        "approved": True,
        "approved_by": "benchmark-user",
        "approved_at": "2026-07-11T00:00:00Z",
        "approval_reason": "Verify the exact reviewed candidate.",
        "approval_evidence": {
            "source_message_id": "approval-repair",
        },
    }
    result = await evaluation.continue_evaluation_coding_run(
        {
            "workspace_root": str(workspace_root),
            "run_id": started["run_id"],
            "action": "approve_and_verify",
            "approval": approval,
            "execution_specs": [
                {"tool": "pytest", "pytest_selectors": ["tests/test_module.py"]}
            ],
        },
        engine_id="action_loop_v1",
    )

    assert result["status"] == "awaiting_approval"
    assert (source_root / "module.py").read_text(encoding="utf-8") == original_content
    state = json.loads(
        (
            workspace_root
            / "coding_runs"
            / started["run_id"]
            / "action_loop"
            / "state.json"
        ).read_text(encoding="utf-8")
    )
    prior_apply = state["apply_attempts"][0]
    assert prior_apply["proposal_digest"] != state["proposal_digest"]
    assert state["execution_attempts"][0]["status"] == "failed"
    assert state["current_failure"] is None
    observations = state["observations"]
    run_observation = next(
        row for row in observations if row["kind"] == "run_result"
    )
    assert run_observation["outcome"] == "ok"
    assert "candidate_execution_identity" not in run_observation["evidence"][0]
    prior_apply_source = (
        workspace_root
        / "patch_apply"
        / prior_apply["apply_package_id"]
        / "source"
        / "module.py"
    )
    assert prior_apply_source.read_text(encoding="utf-8") == invalid_content
    assert len(state["approvals"]) == 1
