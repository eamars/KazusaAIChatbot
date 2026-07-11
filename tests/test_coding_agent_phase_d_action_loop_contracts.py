"""Closed action-loop parser contracts."""

import json

import pytest


def _candidate_execution_base(revision: int = 0) -> dict[str, object]:
    """Build one deterministic current-candidate execution identity."""

    return {
        "run_id": "a" * 32,
        "candidate_id": "b" * 64,
        "candidate_revision": revision,
        "candidate_manifest_digest": "c" * 64,
        "base_snapshot_id": "d" * 64,
        "execution_policy_digest": "e" * 64,
    }


def test_parser_rejects_unknown_keys_and_invalid_capabilities() -> None:
    """Reject model actions outside the objective capability set."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.parser import (
        parse_action,
    )

    rejected = parse_action(
        {
            "schema_version": "coding_action.v1",
            "action_id": "one",
            "action": "edit",
            "reason": "change it",
            "args": {},
            "unexpected": True,
        },
        allowed_actions={"read", "finish"},
    )

    assert rejected["status"] == "invalid_action"


def test_parser_rejects_operation_extras_and_boolean_revision() -> None:
    """Keep each edit operation exact and reject JSON booleans as integers."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.parser import (
        parse_action,
    )

    base_action = {
        "schema_version": "coding_action.v1",
        "action_id": "edit-one",
        "action": "edit",
        "reason": "Apply the observed change.",
        "args": {
            "operation": "delete_file",
            "repo_path": "module.py",
            "expected_sha256": "a" * 64,
            "expected_candidate_revision": 0,
        },
    }
    extra_field = json.loads(json.dumps(base_action))
    extra_field["args"]["replacement"] = "ignored"
    boolean_revision = json.loads(json.dumps(base_action))
    boolean_revision["args"]["expected_candidate_revision"] = True

    assert parse_action(
        extra_field,
        allowed_actions={"edit"},
    )["status"] == "invalid_action"
    assert parse_action(
        boolean_revision,
        allowed_actions={"edit"},
    )["status"] == "invalid_action"


def test_invalid_output_three_strike_blocks() -> None:
    """Turn repeated invalid controller output into a typed blocker."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.parser import (
        invalid_output_blocker,
    )

    blocker = invalid_output_blocker(3)

    assert blocker == {
        "blocker_type": "controller_contract_failure",
        "resume_target": "retry_loop",
    }


def test_controller_prompt_keeps_block_routing_deterministic() -> None:
    """Keep resume routing outside the model-owned blocker contract."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.prompts import (
        CONTROLLER_PROMPT,
    )

    assert "Deterministic code assigns the resume target" in CONTROLLER_PROMPT
    assert "`resume_target`" not in CONTROLLER_PROMPT


def test_environment_blocker_resumes_at_verification() -> None:
    """Keep missing external dependencies bound to the native verifier."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.supervisor import (
        _deterministic_blocker,
    )

    blocker = _deterministic_blocker(
        {
            "blocker_type": "environment",
            "question": "Install the missing dependency.",
            "options": ["Install and retry."],
            "blocking_evidence_refs": ["execution_verification"],
        },
        loop_state={
            "current_failure": {
                "kind": "execution_verification",
            },
        },
    )

    assert blocker["resume_target"] == "retry_verification"


def test_context_exposes_semantic_capabilities_without_host_details() -> None:
    """Keep controller context bounded to semantic action availability."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.context import (
        render_controller_context,
    )

    context = render_controller_context(
        goal="Read the source.",
        capabilities=["read", "search", "finish"],
        working_notes="Start from the relevant module.",
        observations=[{"summary": "one safe read result"}],
    )

    assert '"capabilities": ["read", "search", "finish"]' in context
    assert "C:\\" not in context


def test_context_reducer_preserves_required_fields_and_valid_json() -> None:
    """Evict old evidence structurally without slicing the JSON document."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.context import (
        render_controller_context,
    )

    observations = [
        {"sequence": number, "summary": f"old-{number}-" + "x" * 4000}
        for number in range(20)
    ]
    context = render_controller_context(
        goal="Keep this goal.",
        acceptance_criteria=["Keep this criterion."],
        capabilities=["read", "search", "finish"],
        source_identity_digest="source-digest",
        candidate_revision=7,
        changed_paths=["module.py"],
        current_failure={
            "kind": "execution_verification",
            "summary": "latest failure",
            "candidate_revision": 7,
        },
        working_notes="current note",
        observations=observations,
    )
    payload = json.loads(context)

    assert payload["goal"] == "Keep this goal."
    assert payload["acceptance_criteria"] == ["Keep this criterion."]
    assert payload["capabilities"] == ["read", "search", "finish"]
    assert payload["candidate_revision"] == 7
    assert payload["current_failure"] == {
        "kind": "execution_verification",
        "summary": "latest failure",
        "candidate_revision": 7,
    }
    assert payload["observations"][-1]["sequence"] == 19
    assert len(context) <= 50_000


def test_context_redacts_host_paths_from_failure_summaries() -> None:
    """Project operational failure paths out of the model-facing context."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.context import (
        render_controller_context,
    )

    context = render_controller_context(
        goal="Repair the candidate.",
        capabilities=["edit", "finish"],
        current_failure={
            "kind": "edit_rejected",
            "summary": r"failed at C:\workspace\private\candidate.py",
            "candidate_revision": 1,
        },
        working_notes=r"Inspect C:\workspace\private\notes.txt next.",
        observations=[{
            "outcome": "failed",
            "kind": "run_result",
            "evidence": [{
                "stderr_excerpt": r"C:\workspace\private\candidate.py failed",
                "limitations": [r"See C:\workspace\private\trace.json"],
            }],
        }],
    )

    assert "C:\\workspace" not in context
    assert "<managed_path>" in context


def test_reconciliation_ignores_an_older_observed_finish(tmp_path) -> None:
    """Recover only the latest observed action's terminal state."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop import supervisor

    loop_root = tmp_path / "run" / "action_loop"
    loop_root.mkdir(parents=True)
    finish = {
        "action": "finish",
        "args": {"summary": "Older answer."},
    }
    search = {
        "action": "search",
        "args": {"mode": "literal", "query": "new evidence"},
    }
    (loop_root / "actions.jsonl").write_text(
        "\n".join((
            json.dumps({"sequence": 1, "parsed_action": finish}),
            json.dumps({"sequence": 2, "parsed_action": search}),
        )) + "\n",
        encoding="utf-8",
    )
    (loop_root / "observations.jsonl").write_text(
        "\n".join((
            json.dumps({
                "sequence": 1,
                "action_sequence": 1,
                "kind": "finish_result",
            }),
            json.dumps({
                "sequence": 2,
                "action_sequence": 2,
                "kind": "search_result",
            }),
        )) + "\n",
        encoding="utf-8",
    )
    state = {
        "status": "active",
        "action_count": 0,
        "observation_count": 0,
        "working_note": "",
        "run_action_count": 0,
    }
    observations: list[object] = []

    result = supervisor._reconcile_orphan_action(
        loop_root,
        state,
        observations,
    )

    assert result is None
    assert state["status"] == "active"
    assert state["action_count"] == 2


def test_read_only_loop_cannot_dispatch_edit_or_run(tmp_path) -> None:
    """Report unsupported effects when a read-only capability set is enforced."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.actions import (
        execute_action,
    )

    observation = execute_action(
        action={"action": "edit", "args": {}},
        workspace_root=tmp_path,
        snapshot_id="missing",
    )

    assert observation == {"outcome": "unavailable", "kind": "action_unavailable"}


@pytest.mark.asyncio
async def test_controller_reports_missing_route_as_typed_blocker(monkeypatch) -> None:
    """Keep unavailable route configuration out of service-import failures."""

    from kazusa_ai_chatbot import config
    from kazusa_ai_chatbot.coding_agent.code_action_loop.supervisor import (
        invoke_controller,
    )

    monkeypatch.setattr(config, "CODING_AGENT_ACTION_LOOP_LLM_BASE_URL", "")
    monkeypatch.setattr(config, "CODING_AGENT_ACTION_LOOP_LLM_API_KEY", "")
    monkeypatch.setattr(config, "CODING_AGENT_ACTION_LOOP_LLM_MODEL", "")
    result = await invoke_controller(context="{}", allowed_actions={"finish"})

    assert result["blocker_type"] == "controller_configuration_missing"


def test_parser_rejects_unbounded_or_invalid_action_specific_arguments() -> None:
    """Keep semantic actions closed before any dispatcher observes their args."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.parser import parse_action

    common = {
        "schema_version": "coding_action.v1",
        "action_id": "action-1",
        "reason": "Find the relevant source.",
    }
    command_smuggling = parse_action(
        {
            **common,
            "action": "run",
            "args": {
                "profile": "focused",
                "intent": "Run focused validation.",
                "command": "pytest -q",
            },
        },
        allowed_actions={"run"},
    )
    incomplete_search = parse_action(
        {
            **common,
            "action": "search",
            "args": {"mode": "literal"},
        },
        allowed_actions={"search"},
    )
    oversized_reason = parse_action(
        {
            **common,
            "reason": "x" * 601,
            "action": "finish",
            "args": {
                "summary": "Done.",
                "acceptance_criteria": [],
                "evidence_refs": [],
                "known_limitations": [],
            },
        },
        allowed_actions={"finish"},
    )

    assert command_smuggling["status"] == "invalid_action"
    assert incomplete_search["status"] == "invalid_action"
    assert oversized_reason["status"] == "invalid_action"


def test_parser_enforces_edit_preconditions_by_operation() -> None:
    """Require hashes and operation-specific fields before candidate mutation."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.parser import parse_action

    common = {
        "schema_version": "coding_action.v1",
        "action_id": "edit-one",
        "action": "edit",
        "reason": "Apply one bounded edit.",
    }
    missing_hash = parse_action(
        {
            **common,
            "args": {
                "operation": "replace_file_small",
                "repo_path": "module.py",
                "expected_candidate_revision": 0,
                "replacement": "VALUE = 2\n",
            },
        },
        allowed_actions={"edit"},
    )
    rename_without_target = parse_action(
        {
            **common,
            "args": {
                "operation": "rename_file",
                "repo_path": "module.py",
                "expected_candidate_revision": 0,
                "expected_sha256": "a" * 64,
            },
        },
        allowed_actions={"edit"},
    )
    create_with_source_hash = parse_action(
        {
            **common,
            "args": {
                "operation": "create_file",
                "repo_path": "new.py",
                "expected_candidate_revision": 0,
                "expected_sha256": "a" * 64,
                "replacement": "VALUE = 1\n",
            },
        },
        allowed_actions={"edit"},
    )

    assert missing_hash["status"] == "invalid_action"
    assert rename_without_target["status"] == "invalid_action"
    assert create_with_source_hash["status"] == "invalid_action"


def test_read_action_returns_a_bounded_snapshot_span(tmp_path) -> None:
    """Resolve read evidence through the pinned snapshot without host paths."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.actions import (
        execute_action,
    )
    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "module.py").write_text(
        "line one\nline two\nline three\n",
        encoding="utf-8",
    )
    snapshot = build_repository_snapshot(
        workspace_root=tmp_path / "workspace",
        source_root=source_root,
        source_identity={"current_commit": "fixture"},
    )

    observation = execute_action(
        action={
            "action": "read",
            "args": {"repo_path": "module.py", "start_line": 2, "end_line": 3},
        },
        workspace_root=tmp_path / "workspace",
        snapshot_id=snapshot["snapshot_id"],
    )

    assert observation["outcome"] == "ok"
    assert observation["kind"] == "read_result"
    assert observation["evidence"] == [{
        "repo_path": "module.py",
        "start_line": 2,
        "end_line": 3,
        "content": "line two\nline three\n",
    }]


def test_search_cursor_is_bound_to_current_candidate_overlay_revision(
    tmp_path,
) -> None:
    """Expose paging while rejecting a cursor after a candidate mutation."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.actions import (
        execute_action,
    )
    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import CandidateState
    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    for index in range(25):
        (source_root / f"module_{index:02d}.py").write_text(
            "NEEDLE = 1\n",
            encoding="utf-8",
        )
    workspace_root = tmp_path / "workspace"
    snapshot = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity={"current_commit": "fixture"},
    )
    run_root = workspace_root / "coding_runs" / "run"
    candidate = CandidateState.create(
        run_root / "candidate",
        source_root=source_root,
    )
    first = execute_action(
        action={
            "action": "search",
            "args": {"mode": "literal", "query": "NEEDLE"},
        },
        workspace_root=workspace_root,
        snapshot_id=snapshot["snapshot_id"],
        run_root=run_root,
        objective_type="propose_patch",
    )
    candidate.apply_journaled_mutation(
        operation_id="create-one",
        kind="create_file",
        repo_path="new.py",
        replacement="OTHER = 2\n",
        expected_revision=0,
        expected_source_sha256=None,
    )
    second = execute_action(
        action={
            "action": "search",
            "args": {
                "mode": "literal",
                "query": "NEEDLE",
                "cursor": first["cursor"],
            },
        },
        workspace_root=workspace_root,
        snapshot_id=snapshot["snapshot_id"],
        run_root=run_root,
        objective_type="propose_patch",
    )

    assert isinstance(first["cursor"], str)
    assert second["outcome"] == "stale_cursor"


def test_read_action_resolves_a_created_candidate_symbol(tmp_path) -> None:
    """Read a symbol from the live overlay instead of a stale base snapshot."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.actions import (
        execute_action,
    )
    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import CandidateState
    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    workspace_root = tmp_path / "workspace"
    snapshot = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity={"current_commit": "fixture"},
    )
    run_root = workspace_root / "coding_runs" / "run"
    candidate = CandidateState.create(run_root / "candidate")
    candidate.apply_journaled_mutation(
        operation_id="create-one",
        kind="create_file",
        repo_path="widget.py",
        replacement="def build_widget() -> str:\n    return 'ready'\n",
        expected_revision=0,
        expected_source_sha256=None,
    )

    observation = execute_action(
        action={
            "action": "read",
            "args": {"repo_path": "widget.py", "symbol": "build_widget"},
        },
        workspace_root=workspace_root,
        snapshot_id=snapshot["snapshot_id"],
        run_root=run_root,
        objective_type="propose_patch",
    )

    assert observation["outcome"] == "ok"
    assert observation["evidence"][0]["content"].startswith(
        "def build_widget()",
    )


def test_run_action_uses_only_trusted_structured_execution_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Map a semantic run profile to stored approved checks without commands."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop import actions

    captured_request: dict[str, object] = {}

    def execute_check(request: dict[str, object]) -> dict[str, object]:
        captured_request.update(request)
        return {
            "status": "succeeded",
            "tool": "pytest",
            "exit_code": 0,
            "timed_out": False,
            "duration_ms": 1,
            "stdout_excerpt": "1 passed",
            "stderr_excerpt": "",
            "output_truncated": False,
            "executed_paths": ["tests/test_module.py"],
            "limitations": [],
            "trace_summary": ["fixture"],
        }

    monkeypatch.setattr(actions, "execute_code_check", execute_check)
    observation = actions.execute_action(
        action={
            "action": "run",
            "args": {
                "profile": "focused",
                "targets": ["tests/test_module.py"],
                "intent": "Verify the candidate change.",
            },
        },
        workspace_root=tmp_path,
        snapshot_id="snapshot",
        run_context={
            "workspace_root": str(tmp_path),
            "candidate_execution_base": _candidate_execution_base(),
            "execution_specs": [
                {"tool": "pytest", "pytest_selectors": ["tests/test_module.py"]}
            ],
        },
    )

    assert observation["outcome"] == "ok"
    assert observation["kind"] == "run_result"
    assert captured_request["execution"] == {
        "tool": "pytest",
        "pytest_selectors": ["tests/test_module.py"],
    }
    assert "candidate_execution_identity" in captured_request
    assert "apply_workspace_ref" not in captured_request
    assert "command" not in captured_request


def test_loop_budget_blocker_carries_retry_and_safe_evidence() -> None:
    """Require wall/run budget exhaustion to preserve a retryable safe state."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.supervisor import (
        _budget_blocker,
    )

    assert _budget_blocker("wall_time", {"kind": "search_result"}) == {
        "blocker_type": "budget",
        "code": "controller_wall_time_budget_exhausted",
        "resume_target": "retry_loop",
        "latest_safe_evidence": {"kind": "search_result"},
    }


def test_approval_binding_rejects_replayed_source_message(tmp_path) -> None:
    """Consume one approval identity for only its exact reviewed candidate."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop import supervisor

    run_root = tmp_path / "coding_runs" / "run-one"
    candidate_root = run_root / "candidate" / "source"
    candidate_root.mkdir(parents=True)
    (candidate_root / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    loop_state = {
        "proposal_digest": "a" * 64,
        "candidate_revision": 1,
        "candidate_tree_digest": supervisor._candidate_tree_digest(
            candidate_root
        ),
        "approvals": [],
    }
    approval = {
        "approved": True,
        "approved_by": "user",
        "approved_at": "2026-07-12T00:00:00Z",
        "approval_reason": "Approve this candidate.",
        "approval_evidence": {"source_message_id": "message-one"},
    }

    binding, error = supervisor._bind_approval(
        run_root=run_root,
        loop_state=loop_state,
        approval=approval,
    )
    assert error == ""
    loop_state["approvals"].append({"approval_binding": binding})
    replayed, replay_error = supervisor._bind_approval(
        run_root=run_root,
        loop_state=loop_state,
        approval=approval,
    )

    assert replayed is None
    assert "already been consumed" in replay_error


def test_budget_blocker_names_run_action_limit() -> None:
    """Keep the run-action budget independently typed and retryable."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.supervisor import (
        _budget_blocker,
    )

    blocker = _budget_blocker("run_action", {})

    assert blocker["code"] == "controller_run_action_budget_exhausted"


def test_committed_orphan_edit_reconstructs_once_without_replay(tmp_path) -> None:
    """Rebuild one observation from committed state without rerunning mutation."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import (
        CandidateState,
    )
    from kazusa_ai_chatbot.coding_agent.code_action_loop.supervisor import (
        _reconcile_orphan_action,
    )

    run_root = tmp_path / "run"
    loop_root = run_root / "action_loop"
    loop_root.mkdir(parents=True)
    candidate = CandidateState.create(run_root / "candidate")
    candidate.apply_journaled_mutation(
        operation_id="one",
        kind="create_file",
        repo_path="a.py",
        replacement="VALUE = 1\n",
        expected_revision=0,
        expected_source_sha256=None,
    )
    action = {
        "action": "edit",
        "args": {
            "operation": "create_file",
            "repo_path": "a.py",
            "expected_candidate_revision": 0,
            "replacement": "VALUE = 1\n",
        },
    }
    action_record = {
        "sequence": 1,
        "parsed_action": action,
        "operation_id": "one",
    }
    (loop_root / "actions.jsonl").write_text(
        f"{json.dumps(action_record)}\n",
        encoding="utf-8",
    )
    state = {
        "candidate_revision": 0,
        "overlay_revision": 0,
        "patch_operations": [],
        "changed_paths": [],
        "status": "active",
        "run_id": "run",
        "objective_type": "propose_patch",
        "action_count": 0,
    }
    observations: list[object] = []
    assert _reconcile_orphan_action(loop_root, state, observations) is None
    assert len(observations) == 1
    assert state["candidate_revision"] == 1
    assert state["patch_operations"] == [{
        "kind": "create_file",
        "path": "a.py",
        "content": "VALUE = 1\n",
        "operation_id": "one",
        "expected_candidate_revision": 0,
    }]


def test_candidate_written_orphan_recovers_once_without_replay(
    tmp_path,
    monkeypatch,
) -> None:
    """Recover the canonical journal phase before reconstructing its observation."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import (
        CandidateState,
    )
    from kazusa_ai_chatbot.coding_agent.code_action_loop.supervisor import (
        _reconcile_orphan_action,
    )
    from kazusa_ai_chatbot.coding_agent.repository_index.overlay import (
        CandidateOverlay,
    )

    run_root = tmp_path / "run"
    loop_root = run_root / "action_loop"
    loop_root.mkdir(parents=True)
    candidate = CandidateState.create(run_root / "candidate")

    def fail_overlay_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated overlay interruption")

    monkeypatch.setattr(CandidateOverlay, "upsert", fail_overlay_write)
    with pytest.raises(RuntimeError, match="overlay interruption"):
        candidate.apply_journaled_mutation(
            operation_id="one",
            kind="create_file",
            repo_path="a.py",
            replacement="VALUE = 1\n",
            expected_revision=0,
            expected_source_sha256=None,
        )
    monkeypatch.undo()
    action = {
        "action": "edit",
        "args": {
            "operation": "create_file",
            "repo_path": "a.py",
            "expected_candidate_revision": 0,
            "replacement": "VALUE = 1\n",
        },
    }
    action_record = {
        "sequence": 1,
        "parsed_action": action,
        "operation_id": "one",
    }
    (loop_root / "actions.jsonl").write_text(
        f"{json.dumps(action_record)}\n",
        encoding="utf-8",
    )
    state = {
        "candidate_revision": 0,
        "overlay_revision": 0,
        "patch_operations": [],
        "changed_paths": [],
        "status": "active",
        "run_id": "run",
        "objective_type": "propose_patch",
        "action_count": 0,
    }
    observations: list[object] = []
    assert _reconcile_orphan_action(loop_root, state, observations) is None
    recovered_candidate = CandidateState.load(run_root / "candidate")
    assert recovered_candidate.journal[0]["state"] == "committed"
    assert len(observations) == 1


def test_irreconcilable_orphan_blocks_without_success_observation(tmp_path) -> None:
    """Refuse a mutation replay when no matching journal identity exists."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import (
        CandidateState,
    )
    from kazusa_ai_chatbot.coding_agent.code_action_loop.supervisor import (
        _reconcile_orphan_action,
    )

    run_root = tmp_path / "run"
    loop_root = run_root / "action_loop"
    loop_root.mkdir(parents=True)
    CandidateState.create(run_root / "candidate")
    action_record = {
        "sequence": 1,
        "parsed_action": {"action": "edit"},
    }
    (loop_root / "actions.jsonl").write_text(
        f"{json.dumps(action_record)}\n",
        encoding="utf-8",
    )
    state = {
        "candidate_revision": 0,
        "status": "active",
        "run_id": "run",
        "objective_type": "propose_patch",
    }
    observations: list[object] = []
    result = _reconcile_orphan_action(loop_root, state, observations)
    assert result["blocker"]["blocker_type"] == "candidate_recovery_failed"
    assert result["blocker"]["resume_target"] == "retry_loop"
    assert observations == []


def test_observed_finish_recovers_terminal_state_without_replay(tmp_path) -> None:
    """Complete a finish transition whose observation outran state persistence."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop import supervisor

    run_root = tmp_path / "run"
    loop_root = run_root / "action_loop"
    loop_root.mkdir(parents=True)
    action = {
        "action": "finish",
        "args": {
            "summary": "Grounded answer.",
            "acceptance_criteria": [],
            "evidence_refs": ["module.py:1"],
            "known_limitations": [],
        },
    }
    (loop_root / "actions.jsonl").write_text(
        json.dumps({"sequence": 1, "parsed_action": action}) + "\n",
        encoding="utf-8",
    )
    (loop_root / "observations.jsonl").write_text(
        json.dumps({
            "sequence": 1,
            "action_sequence": 1,
            "outcome": "ok",
            "kind": "finish_result",
        }) + "\n",
        encoding="utf-8",
    )
    state = {
        "run_id": "run",
        "objective_type": "read_only",
        "status": "active",
        "action_count": 0,
        "observation_count": 0,
    }
    observations: list[object] = []

    result = supervisor._reconcile_orphan_action(
        loop_root,
        state,
        observations,
    )

    assert result["status"] == "completed"
    assert result["answer_text"] == "Grounded answer."
    assert state["action_count"] == 1


def test_run_orphan_reconstructs_persisted_terminal_result_without_execution(
    tmp_path,
    monkeypatch,
) -> None:
    """Recover one bound run observation without repeating verification."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop import supervisor

    run_root = tmp_path / "run"
    loop_root = run_root / "action_loop"
    loop_root.mkdir(parents=True)
    action = {
        "schema_version": "coding_action.v1",
        "action_id": "run-one",
        "action": "run",
        "reason": "Run the approved focused check.",
        "args": {
            "profile": "focused",
            "targets": ["tests/test_module.py"],
            "intent": "Verify the candidate behavior.",
        },
    }
    trusted_context = {
        "workspace_root": str(tmp_path / "workspace"),
        "candidate_execution_base": _candidate_execution_base(),
        "execution_specs": [{
            "tool": "pytest",
            "pytest_selectors": ["tests/test_module.py"],
        }],
    }
    supervisor._append_action_record(
        loop_root,
        1,
        action,
        operation_id="run-operation",
    )
    spec_digest = supervisor._persist_run_execution_spec(
        loop_root=loop_root,
        sequence=1,
        action=action,
        run_context=trusted_context,
    )
    supervisor._persist_run_execution_result(
        loop_root=loop_root,
        sequence=1,
        spec_digest=spec_digest,
        observation={
            "outcome": "ok",
            "kind": "run_result",
            "evidence": [{"status": "succeeded", "tool": "pytest"}],
        },
    )

    def fail_if_executed(*args: object, **kwargs: object) -> None:
        raise AssertionError("orphan recovery repeated execution")

    monkeypatch.setattr(supervisor, "execute_action", fail_if_executed)
    state = {
        "candidate_revision": 0,
        "index_snapshot_id": "snapshot",
        "status": "active",
        "run_id": "run",
        "objective_type": "propose_patch",
        "action_count": 0,
        "run_action_count": 0,
        "trusted_execution_context": trusted_context,
    }
    observations: list[object] = []
    first_result = supervisor._reconcile_orphan_action(
        loop_root,
        state,
        observations,
    )
    second_result = supervisor._reconcile_orphan_action(
        loop_root,
        state,
        observations,
    )

    assert first_result is None
    assert second_result is None
    assert len(observations) == 1
    assert observations[0]["kind"] == "run_result"
    assert state["action_count"] == 1
    assert state["run_action_count"] == 1


def test_run_orphan_blocks_when_terminal_evidence_identity_mismatches(
    tmp_path,
) -> None:
    """Block an orphan run whose terminal result is bound to another spec."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop import supervisor

    loop_root = tmp_path / "run" / "action_loop"
    loop_root.mkdir(parents=True)
    action = {
        "action": "run",
        "args": {
            "profile": "derived_base",
            "intent": "Verify the candidate.",
        },
    }
    trusted_context = {
        "candidate_execution_base": _candidate_execution_base(),
        "execution_specs": [{"tool": "python_compileall", "paths": ["."]}],
    }
    supervisor._append_action_record(
        loop_root,
        1,
        action,
        operation_id="run-operation",
    )
    spec_digest = supervisor._persist_run_execution_spec(
        loop_root=loop_root,
        sequence=1,
        action=action,
        run_context=trusted_context,
    )
    supervisor._persist_run_execution_result(
        loop_root=loop_root,
        sequence=1,
        spec_digest=f"wrong-{spec_digest}",
        observation={
            "outcome": "ok",
            "kind": "run_result",
            "evidence": [{"status": "succeeded"}],
        },
    )
    state = {
        "candidate_revision": 0,
        "status": "active",
        "run_id": "run",
        "objective_type": "propose_patch",
        "action_count": 0,
        "run_action_count": 0,
        "trusted_execution_context": trusted_context,
    }
    observations: list[object] = []

    result = supervisor._reconcile_orphan_action(loop_root, state, observations)

    assert result["blocker"] == {
        "blocker_type": "candidate_recovery_failed",
        "code": "orphan_run_evidence_mismatch",
        "resume_target": "retry_loop",
    }
    assert observations == []


@pytest.mark.asyncio
async def test_run_action_limit_blocks_before_a_ninth_controller_or_executor_call(
    tmp_path,
    monkeypatch,
) -> None:
    """Stop a spent run-action segment before another semantic turn starts."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop import supervisor

    calls = {"controller": 0, "executor": 0}

    async def controller(**kwargs: object) -> dict[str, object]:
        calls["controller"] += 1
        return {"status": "blocked"}

    def executor(**kwargs: object) -> dict[str, object]:
        calls["executor"] += 1
        return {"outcome": "ok", "kind": "run_result", "evidence": []}

    monkeypatch.setattr(supervisor, "execute_action", executor)
    run_root = tmp_path / "coding_runs" / "run"
    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import (
        CandidateState,
    )

    CandidateState.create(run_root / "candidate")
    state = _loop_state_for_budget_test(
        run_action_count=supervisor.MAX_RUN_ACTIONS,
    )
    result = await supervisor._run_controller_loop(
        run_root=run_root,
        loop_state=state,
        allowed_actions={"run", "finish"},
        controller=controller,
    )

    assert result["blocker"]["code"] == "controller_run_action_budget_exhausted"
    assert calls == {"controller": 0, "executor": 0}


@pytest.mark.asyncio
async def test_wall_time_limit_blocks_before_the_next_controller_call(
    tmp_path,
    monkeypatch,
) -> None:
    """Enforce wall time at the real loop boundary before another turn."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop import supervisor

    calls = {"controller": 0}

    async def controller(**kwargs: object) -> dict[str, object]:
        calls["controller"] += 1
        return {"status": "blocked"}

    times = iter((100.0, 101.0))
    monkeypatch.setattr(supervisor, "MAX_SEGMENT_WALL_SECONDS", 1)
    monkeypatch.setattr(supervisor.time, "time", lambda: next(times))
    run_root = tmp_path / "coding_runs" / "run"
    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import (
        CandidateState,
    )

    CandidateState.create(run_root / "candidate")
    state = _loop_state_for_budget_test(run_action_count=0)
    result = await supervisor._run_controller_loop(
        run_root=run_root,
        loop_state=state,
        allowed_actions={"finish"},
        controller=controller,
    )

    assert result["blocker"]["code"] == "controller_wall_time_budget_exhausted"
    assert calls["controller"] == 0


@pytest.mark.asyncio
async def test_completed_turn_is_durable_before_the_next_controller_call(
    tmp_path,
) -> None:
    """Persist active counters and notes before requesting another action."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop import supervisor

    call_count = 0

    async def controller(**kwargs: object) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "status": "ok",
                "action": {
                    "action": "note",
                    "args": {
                        "completed": ["inspected source"],
                        "remaining": ["finish answer"],
                        "assumptions": [],
                    },
                    "working_note": "Source inspection is complete.",
                },
            }
        raise RuntimeError("simulated process interruption")

    run_root = tmp_path / "coding_runs" / "run"
    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import (
        CandidateState,
    )

    CandidateState.create(run_root / "candidate")
    state = _loop_state_for_budget_test(run_action_count=0)
    with pytest.raises(RuntimeError, match="process interruption"):
        await supervisor._run_controller_loop(
            run_root=run_root,
            loop_state=state,
            allowed_actions={"note", "finish"},
            controller=controller,
        )

    saved = json.loads(
        (run_root / "action_loop" / "state.json").read_text(encoding="utf-8"),
    )
    assert saved["action_count"] == 1
    assert saved["working_note"] == "Source inspection is complete."
    assert saved["observations"][0]["kind"] == "note_result"


def _loop_state_for_budget_test(*, run_action_count: int) -> dict[str, object]:
    """Build the complete durable state required by real-loop budget tests."""

    return {
        "run_id": "run",
        "objective_type": "read_only",
        "status": "active",
        "goal": "Inspect the source.",
        "acceptance_criteria": [],
        "source_identity_digest": "source",
        "index_snapshot_id": "snapshot",
        "candidate_revision": 0,
        "changed_paths": [],
        "current_failure": None,
        "observation_count": 0,
        "working_note": "",
        "observations": [],
        "action_count": 0,
        "invalid_output_count": 0,
        "consecutive_no_progress_count": 0,
        "run_action_count": run_action_count,
        "segment_started_at_epoch_seconds": 0,
        "source_request": {},
        "repository": None,
        "source_scope": None,
    }
