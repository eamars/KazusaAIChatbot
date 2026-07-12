"""Private benchmark-only coding-run engine selection."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from kazusa_ai_chatbot.coding_agent.code_action_loop import (
    supervisor as action_loop_supervisor,
)
from kazusa_ai_chatbot.coding_agent.code_action_loop.actions import execute_action
from kazusa_ai_chatbot.coding_agent.code_action_loop.state import CandidateState
from kazusa_ai_chatbot.coding_agent.code_action_loop.supervisor import (
    continue_action_loop,
    invoke_controller,
    start_action_loop,
)
from kazusa_ai_chatbot.coding_agent.code_fetching import run as fetch_source
from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
    build_canonical_operation_records,
    canonical_proposal_digest,
    compile_patch_operations,
)
from kazusa_ai_chatbot.coding_agent.code_verifying.execution_planning import (
    derive_base_execution_plan,
    patch_artifact_digest,
)
from kazusa_ai_chatbot.coding_agent.coding_run import (
    supervisor as pipeline_supervisor,
)
from kazusa_ai_chatbot.coding_agent.coding_run.ledger import (
    build_run_paths,
    load_events,
    new_run_id,
    public_response,
    write_ledger,
)
from kazusa_ai_chatbot.coding_agent.coding_run.locking import (
    LOCK_TIMEOUT_SECONDS,
    acquire_workspace_locks,
    build_lock_keys,
)
from kazusa_ai_chatbot.coding_agent.coding_run.supervisor import (
    continue_coding_run,
    start_coding_run,
)


ENVIRONMENT_SCENARIO_SCHEMA = "coding_agent_environment_precondition.v1"
ENVIRONMENT_REVIEW_SCHEMA = "coding_agent_environment_review.v1"
ENVIRONMENT_OPERATION_ID = "dependency-loader-mapping"
ENVIRONMENT_REPO_PATH = "dep_tool/loader.py"
ENVIRONMENT_REPLACEMENT = (
    '"""Config loading helpers with an intentionally incomplete '
    'implementation."""\n\n'
    "from __future__ import annotations\n\n\n"
    "def load_config(text: str) -> dict[str, str]:\n"
    '    """Load configuration text into a mapping."""\n\n'
    '    key, value = text.split(":", 1)\n'
    "    return {key.strip(): value.strip()}\n"
)

STAGE5A_SCENARIO_SCHEMA = "coding_agent_stage5a_scenario.v1"
STAGE5A_SCENARIO_DRIVERS = frozenset((
    "source_backed_bug_fix",
    "source_free_creation",
    "small_feature",
    "revision",
    "preflight",
    "verification_repair",
    "blocker_response",
    "same_source_concurrency",
    "mixed_create_edit",
    "repository_scale",
    "stale_index_cursor",
    "delete",
    "rename",
))


def build_environment_scenario_precondition(
    request: Mapping[str, object],
    *,
    fixture_manifest_sha256: str,
    approval: Mapping[str, object],
    execution_specs: list[dict[str, object]],
) -> dict[str, object]:
    """Build the engine-neutral proposal-at-approval scenario identity."""

    source_root = _request_source_root(request)
    operation = {
        "kind": "replace_file_small",
        "path": ENVIRONMENT_REPO_PATH,
        "content": ENVIRONMENT_REPLACEMENT,
        "operation_id": ENVIRONMENT_OPERATION_ID,
        "expected_candidate_revision": 0,
    }
    records = build_canonical_operation_records(
        repo_root=source_root,
        patch_operations=[operation],
        candidate_revision=1,
    )
    operation["expected_source_sha256"] = records[0][
        "expected_source_sha256"
    ]
    proposal_digest = canonical_proposal_digest(records)
    artifacts, created_files, changed_files, errors = compile_patch_operations(
        repo_root=source_root,
        patch_operations=[operation],
        max_files=1,
        max_diff_chars=20_000,
    )
    if errors or not artifacts:
        raise ValueError("environment scenario proposal could not be compiled")
    review = {
        "schema_version": ENVIRONMENT_REVIEW_SCHEMA,
        "canonical_operation_records": records,
        "proposal_digest": proposal_digest,
        "patch_artifacts": artifacts,
        "created_files": created_files,
        "changed_files": changed_files,
        "summary": "Parse one configuration mapping before verification.",
    }
    review_sha256 = hashlib.sha256(_review_bytes(review)).hexdigest()
    identity = {
        "provider": "github",
        "owner": "fixture",
        "repo": "gate_09_missing_dependency",
        "resolved_ref": "master",
        "fixture_manifest_sha256": fixture_manifest_sha256,
    }
    digest_payload = {
        "schema_version": ENVIRONMENT_SCENARIO_SCHEMA,
        "source_identity": identity,
        "canonical_operation_records": records,
        "proposal_digest": proposal_digest,
        "review_sha256": review_sha256,
        "approval": dict(approval),
        "execution_specs": execution_specs,
    }
    precondition = {
        **digest_payload,
        "scenario_precondition_digest": _canonical_digest(digest_payload),
        "review": review,
        "patch_operation": operation,
    }
    validate_environment_scenario_precondition(precondition)
    return precondition


def validate_environment_scenario_precondition(
    precondition: Mapping[str, object],
) -> None:
    """Reject drift in the private engine-neutral scenario contract."""

    if precondition.get("schema_version") != ENVIRONMENT_SCENARIO_SCHEMA:
        raise ValueError("environment scenario schema is invalid")
    review = precondition.get("review")
    if not isinstance(review, Mapping):
        raise ValueError("environment scenario review is missing")
    if hashlib.sha256(_review_bytes(review)).hexdigest() != precondition.get(
        "review_sha256"
    ):
        raise ValueError("environment scenario review identity mismatch")
    records = precondition.get("canonical_operation_records")
    if not isinstance(records, list) or not records:
        raise ValueError("environment scenario operations are missing")
    if canonical_proposal_digest(records) != precondition.get("proposal_digest"):
        raise ValueError("environment scenario proposal identity mismatch")
    digest_payload = {
        key: precondition[key]
        for key in (
            "schema_version",
            "source_identity",
            "canonical_operation_records",
            "proposal_digest",
            "review_sha256",
            "approval",
            "execution_specs",
        )
    }
    if _canonical_digest(digest_payload) != precondition.get(
        "scenario_precondition_digest"
    ):
        raise ValueError("environment scenario precondition identity mismatch")


async def materialize_evaluation_scenario(
    request: Mapping[str, object],
    *,
    engine_id: str,
    precondition: Mapping[str, object],
) -> dict[str, object]:
    """Materialize one neutral proposal into an engine-native private ledger."""

    validate_environment_scenario_precondition(precondition)
    if engine_id == "pipeline_v1":
        result = await _materialize_pipeline_scenario(request, precondition)
        return result
    if engine_id == "action_loop_v1":
        result = await _materialize_action_loop_scenario(request, precondition)
        return result
    return {
        "status": "rejected",
        "limitations": ["Benchmark engine is unsupported."],
    }


async def run_evaluation_coding_run(
    request: Mapping[str, object],
    *,
    engine_id: str,
) -> dict[str, object]:
    """Run one benchmark request through the selected isolated engine.

    Args:
        request: Public coding-run request copied into a benchmark workspace.
        engine_id: The locked engine selected by the benchmark case runner.

    Returns:
        A public-safe benchmark response without registering a runtime route.
    """

    if engine_id == "pipeline_v1":
        response = await start_coding_run(dict(request))
        return response
    if engine_id != "action_loop_v1":
        return {
            "status": "rejected",
            "limitations": ["Benchmark engine is unsupported."],
        }
    result = await _run_action_loop_evaluation(dict(request))
    return result


async def _run_action_loop_evaluation(
    request: dict[str, object],
) -> dict[str, object]:
    """Run one isolated durable controller loop through its internal engine."""

    workspace_root_text = request.get("workspace_root")
    if not isinstance(workspace_root_text, str) or not workspace_root_text:
        return {
            "status": "rejected",
            "limitations": ["Evaluation workspace root is required."],
        }
    run_id = uuid4().hex
    run_root = Path(workspace_root_text) / "coding_runs" / run_id
    result = await start_action_loop(
        request,
        run_root=run_root,
        controller=invoke_controller,
    )
    return result


async def run_stage5a_scenario_driver(
    request: Mapping[str, object],
    *,
    scenario_driver: str,
) -> dict[str, object]:
    """Run one deterministic Stage 5A lifecycle through the private loop.

    Args:
        request: Isolated benchmark request with a workspace and optional source.
        scenario_driver: Closed lifecycle contract selected by the v3 manifest.

    Returns:
        Durable scenario evidence and the private run projections required by
        Stage 5A category acceptance checks.

    Raises:
        ValueError: If the driver or required workspace contract is invalid.
    """

    if scenario_driver not in STAGE5A_SCENARIO_DRIVERS:
        raise ValueError("Stage 5A scenario driver is unsupported")
    workspace_root = request.get("workspace_root")
    if not isinstance(workspace_root, str) or not workspace_root:
        raise ValueError("Stage 5A scenario workspace is required")
    if scenario_driver == "same_source_concurrency":
        result = await _run_concurrent_stage5a_scenarios(
            request=dict(request),
        )
        return result
    result = await _run_one_stage5a_scenario(
        request=dict(request),
        scenario_driver=scenario_driver,
    )
    return result


async def _run_concurrent_stage5a_scenarios(
    *,
    request: dict[str, object],
) -> dict[str, object]:
    """Run two same-source scenarios concurrently through the source lock."""

    started_count = 0
    all_started = asyncio.Event()

    async def run_one(label: str) -> dict[str, object]:
        nonlocal started_count
        started_count += 1
        if started_count == 2:
            all_started.set()
        await all_started.wait()
        worker_request = dict(request)
        worker_request["question"] = (
            f"Stage 5A concurrent worker {label} prepares one source edit."
        )
        result = await _run_one_stage5a_scenario(
            request=worker_request,
            scenario_driver="source_backed_bug_fix",
        )
        result["worker"] = label
        return result

    results = await asyncio.gather(run_one("one"), run_one("two"))
    ledgers = [result["run_ledger_path"] for result in results]
    lock_evidence = [result["lock_evidence"] for result in results]
    return {
        "status": "completed",
        "scenario_driver": "same_source_concurrency",
        "scenario_schema": STAGE5A_SCENARIO_SCHEMA,
        "scenario_evidence": {
            "overlap_observed": True,
            "isolated_ledgers": len(set(ledgers)) == 2,
            "lock_evidence": lock_evidence,
            "worker_ledgers": ledgers,
        },
        "workers": results,
    }


async def _run_one_stage5a_scenario(
    *,
    request: dict[str, object],
    scenario_driver: str,
) -> dict[str, object]:
    """Materialize one category-specific scenario under canonical locks."""

    workspace_root = Path(str(request["workspace_root"]))
    run_root = workspace_root / "coding_runs" / uuid4().hex
    request["objective_type"] = "propose_patch"
    lock_keys = build_lock_keys(
        run_id=run_root.name,
        source_identity=action_loop_supervisor._lock_source_identity(request),
    )
    lock_wait_started = time.monotonic()
    async with acquire_workspace_locks(
        workspace_root=workspace_root,
        keys=lock_keys,
        timeout_seconds=LOCK_TIMEOUT_SECONDS,
    ) as acquired:
        lock_acquired = time.monotonic()
        if not acquired:
            raise ValueError("Stage 5A scenario could not acquire its locks")
        prepared = await action_loop_supervisor._prepare_run(
            request=request,
            run_root=run_root,
        )
        if prepared.get("status") != "ready":
            raise ValueError("Stage 5A scenario preparation failed")
        loop_state = prepared.get("loop_state")
        if not isinstance(loop_state, dict):
            raise ValueError("Stage 5A scenario loop state is missing")
        await asyncio.sleep(0.02)
        scenario_evidence = await _materialize_stage5a_driver(
            request=request,
            run_root=run_root,
            loop_state=loop_state,
            scenario_driver=scenario_driver,
        )
        lock_released = time.monotonic()
    scenario_path = run_root / "stage5a_scenario.json"
    scenario_record = {
        "schema_version": STAGE5A_SCENARIO_SCHEMA,
        "scenario_driver": scenario_driver,
        "run_id": run_root.name,
        "scenario_evidence": scenario_evidence,
        "lock_evidence": {
            "wait_started": lock_wait_started,
            "acquired": lock_acquired,
            "released": lock_released,
            "wait_seconds": max(0.0, lock_acquired - lock_wait_started),
        },
    }
    scenario_path.write_text(
        json.dumps(scenario_record, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    action_loop_supervisor._persist_loop_state(run_root, loop_state)
    return {
        "status": "completed",
        "scenario_driver": scenario_driver,
        "scenario_path": str(scenario_path),
        "run_ledger_path": str(run_root / "run.json"),
        "scenario_evidence": scenario_evidence,
        "lock_evidence": scenario_record["lock_evidence"],
    }


async def _materialize_stage5a_driver(
    *,
    request: Mapping[str, object],
    run_root: Path,
    loop_state: dict[str, object],
    scenario_driver: str,
) -> dict[str, object]:
    """Apply the deterministic lifecycle operations for one Stage 5A driver."""

    operations = _stage5a_operations(
        request=request,
        run_root=run_root,
        scenario_driver=scenario_driver,
    )
    operation_records: list[dict[str, object]] = []
    proposal_revisions: list[str] = []
    search_evidence: dict[str, object] = {}
    if scenario_driver in {"repository_scale", "stale_index_cursor"}:
        search_query = (
            "target_module"
            if scenario_driver == "repository_scale"
            else "module"
        )
        search_action = {
            "schema_version": "coding_action.v1",
            "action_id": f"{scenario_driver}-search",
            "action": "search",
            "reason": "Collect the scenario discovery evidence.",
            "args": {"mode": "path", "query": search_query},
        }
        search_evidence = execute_action(
            action=search_action,
            workspace_root=run_root.parent.parent,
            snapshot_id=str(loop_state["index_snapshot_id"]),
            run_root=run_root,
            objective_type="propose_patch",
        )
        if scenario_driver == "repository_scale":
            search_evidence["target_path_supplied"] = False
            search_evidence["discovered_file_count"] = _candidate_file_count(
                run_root / "candidate" / "source",
            )
        else:
            search_evidence["old_cursor"] = search_evidence.get("cursor")
    for index, operation in enumerate(operations, start=1):
        action = _stage5a_edit_action(
            operation=operation,
            run_root=run_root,
            loop_state=loop_state,
            index=index,
        )
        operation_id = f"stage5a-{run_root.name}-{index}"
        action_loop_supervisor._append_action_record(
            run_root / "action_loop",
            index,
            action,
            operation_id=operation_id,
        )
        observation = execute_action(
            action=action,
            workspace_root=run_root.parent.parent,
            snapshot_id=str(loop_state["index_snapshot_id"]),
            run_root=run_root,
            objective_type="propose_patch",
            operation_id=operation_id,
        )
        if observation.get("outcome") != "ok":
            raise ValueError(
                f"Stage 5A scenario operation failed: {observation.get('kind')}"
            )
        action_loop_supervisor._merge_edit_outcome(
            run_root=run_root,
            loop_state=loop_state,
            observation=observation,
        )
        observation["sequence"] = index
        observation["action_sequence"] = index
        observation["candidate_revision"] = loop_state["candidate_revision"]
        action_loop_supervisor._append_jsonl(
            run_root / "action_loop" / "observations.jsonl",
            observation,
        )
        loop_state["observations"].append(observation)
        loop_state["action_count"] = index
        loop_state["observation_count"] = len(loop_state["observations"])
        action_loop_supervisor._persist_active_loop_state(run_root, loop_state)
        if scenario_driver == "revision" and index == 1:
            first_error = action_loop_supervisor._finalize_candidate(
                run_root,
                loop_state,
            )
            if first_error:
                raise ValueError(first_error)
            proposal_revisions.append(str(loop_state["proposal_digest"])
            )
            loop_state["scenario_revision_request"] = (
                "Revise the proposal with the requested continuation."
            )
    if scenario_driver == "verification_repair":
        first_error = action_loop_supervisor._finalize_candidate(
            run_root,
            loop_state,
        )
        if first_error:
            raise ValueError(first_error)
        repair_result = await _execute_stage5a_verification_repair(
            run_root=run_root,
            loop_state=loop_state,
        )
        return {
            "operations": loop_state["patch_operations"],
            "operation_count": len(loop_state["patch_operations"]),
            "candidate_revision": loop_state["candidate_revision"],
            "proposal_revisions": [str(loop_state["proposal_digest"])],
            "search": search_evidence,
            "preflight_plan": None,
            "execution_attempts": repair_result,
            "approval_required": True,
            "execution_before_approval": False,
        }
    finalization_error = action_loop_supervisor._finalize_candidate(
        run_root,
        loop_state,
    )
    if finalization_error:
        raise ValueError(finalization_error)
    proposal_revisions.append(str(loop_state["proposal_digest"]))
    if scenario_driver == "stale_index_cursor":
        cursor = search_evidence.get("cursor")
        if isinstance(cursor, str):
            stale_action = {
                "schema_version": "coding_action.v1",
                "action_id": "stale-cursor-reuse",
                "action": "search",
                "reason": "Re-use evidence from before the candidate revision.",
                "args": {
                    "mode": "path",
                    "query": "module",
                    "cursor": cursor,
                },
            }
            stale_result = execute_action(
                action=stale_action,
                workspace_root=run_root.parent.parent,
                snapshot_id=str(loop_state["index_snapshot_id"]),
                run_root=run_root,
                objective_type="propose_patch",
            )
            search_evidence["stale_rejected"] = stale_result.get(
                "outcome"
            ) in {"stale", "stale_cursor"}
    scenario_execution_attempts: list[dict[str, object]] = []
    if scenario_driver == "preflight":
        plan = derive_base_execution_plan(
            candidate_root=run_root / "candidate" / "source",
            patch_artifacts=loop_state["patch_artifacts"],
            run_id=run_root.name,
            source_identity=loop_state["index_source_identity"],
            proposal_revision=1,
        )
    else:
        plan = None
    result = {
        "operations": loop_state["patch_operations"],
        "operation_count": len(loop_state["patch_operations"]),
        "candidate_revision": loop_state["candidate_revision"],
        "proposal_revisions": proposal_revisions,
        "search": search_evidence,
        "preflight_plan": plan,
        "execution_attempts": scenario_execution_attempts,
        "approval_required": True,
        "execution_before_approval": False,
    }
    if scenario_driver == "blocker_response":
        result["blocker"] = {
            "blocker_type": "environment",
            "status": "open",
            "unresolved": True,
        }
        result["verbatim_response"] = (
            "The external dependency remains unavailable; preserve the blocker."
        )
    return result


def _stage5a_operations(
    *,
    request: Mapping[str, object],
    run_root: Path,
    scenario_driver: str,
) -> list[dict[str, object]]:
    """Return one closed operation sequence for a Stage 5A scenario."""

    source_root = run_root / "candidate" / "source"
    runtime_paths = sorted(
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*.py")
        if "tests" not in path.parts
    )
    runtime_path = runtime_paths[0] if runtime_paths else "module.py"
    if scenario_driver == "source_free_creation":
        return [
            {
                "operation": "create_file",
                "repo_path": "app.py",
                "replacement": "VALUE = 1\n",
            },
            {
                "operation": "create_file",
                "repo_path": "tests/test_app.py",
                "replacement": (
                    "from app import VALUE\n\n"
                    "def test_value():\n"
                    "    assert VALUE == 1\n"
                ),
            },
        ]
    if scenario_driver == "mixed_create_edit":
        return [
            {
                "operation": "create_file",
                "repo_path": "generated.py",
                "replacement": "VALUE = 1\n",
            },
            {
                "operation": "replace_file_small",
                "repo_path": "generated.py",
                "replacement": "VALUE = 2\n",
            },
        ]
    if scenario_driver == "delete":
        delete_paths = [
            path for path in runtime_paths if Path(path).name == "obsolete.py"
        ]
        if not delete_paths:
            raise ValueError("delete scenario fixture omitted obsolete.py")
        return [{"operation": "delete_file", "repo_path": delete_paths[0]}]
    if scenario_driver == "rename":
        rename_paths = [
            path
            for path in runtime_paths
            if Path(path).name in {"old.py", "old_name.py"}
        ]
        if not rename_paths:
            raise ValueError("rename scenario fixture omitted its source path")
        source_path = rename_paths[0]
        target_path = str(Path(source_path).with_name("new.py")).replace(
            "\\",
            "/",
        )
        return [{
            "operation": "rename_file",
            "repo_path": source_path,
            "target_path": target_path,
        }]
    if scenario_driver == "verification_repair":
        return [{
            "operation": "replace_file_small",
            "repo_path": runtime_path,
            "replacement": "VALUE = 0\n",
        }]
    if scenario_driver == "revision":
        return [
            {
                "operation": "replace_file_small",
                "repo_path": runtime_path,
                "replacement": "VALUE = 2\n",
            },
            {
                "operation": "replace_file_small",
                "repo_path": runtime_path,
                "replacement": "VALUE = 3\n",
            },
        ]
    return [{
        "operation": "replace_file_small",
        "repo_path": runtime_path,
        "replacement": "VALUE = 2\n",
    }]


def _stage5a_edit_action(
    *,
    operation: Mapping[str, object],
    run_root: Path,
    loop_state: Mapping[str, object],
    index: int,
) -> dict[str, object]:
    """Build one deterministic edit action with current candidate evidence."""

    repo_path = str(operation["repo_path"])
    candidate = CandidateState.load(run_root / "candidate")
    current_content = candidate.read_safe_text(repo_path)
    args = {
        "operation": operation["operation"],
        "repo_path": repo_path,
        "expected_candidate_revision": candidate.revision,
    }
    if current_content is not None:
        args["expected_sha256"] = hashlib.sha256(
            current_content.encode("utf-8"),
        ).hexdigest()
    replacement = operation.get("replacement")
    if isinstance(replacement, str):
        args["replacement"] = replacement
    target_path = operation.get("target_path")
    if isinstance(target_path, str):
        args["target_path"] = target_path
    return {
        "schema_version": "coding_action.v1",
        "action_id": f"stage5a-edit-{index}",
        "action": "edit",
        "reason": "Materialize the locked Stage 5A scenario operation.",
        "args": args,
    }


async def _execute_stage5a_verification_repair(
    *,
    run_root: Path,
    loop_state: dict[str, object],
) -> list[dict[str, object]]:
    """Run a failed proposal, canonical repair, and current-effect verification."""

    execution_specs = [{
        "tool": "pytest",
        "pytest_selectors": ["tests/test_module.py"],
        "timeout_seconds": 60,
    }]

    async def repair_controller(**_kwargs: object) -> dict[str, object]:
        candidate = CandidateState.load(run_root / "candidate")
        content = candidate.read_safe_text("module.py")
        if content == "VALUE = 0\n":
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "stage5a-repair-edit",
                "action": "edit",
                "reason": "Repair the failed current candidate.",
                "args": {
                    "operation": "replace_file_small",
                    "repo_path": "module.py",
                    "expected_sha256": hashlib.sha256(
                        content.encode("utf-8"),
                    ).hexdigest(),
                    "expected_candidate_revision": candidate.revision,
                    "replacement": "VALUE = 2\n",
                },
            }
        else:
            action = {
                "schema_version": "coding_action.v1",
                "action_id": "stage5a-repair-finish",
                "action": "finish",
                "reason": "Submit the repaired current candidate.",
                "args": {
                    "summary": "The repaired candidate is ready for approval.",
                    "acceptance_criteria": [],
                    "evidence_refs": ["module.py:1"],
                    "known_limitations": [],
                },
            }
        return {"status": "ok", "action": action}

    first_approval = _stage5a_approval("stage5a-approval-first")
    binding, binding_error = action_loop_supervisor._bind_approval(
        run_root=run_root,
        loop_state=loop_state,
        approval=first_approval,
    )
    if binding_error or binding is None:
        raise ValueError(binding_error or "first Stage 5A approval binding failed")
    effect_error = action_loop_supervisor._initialize_pending_effect(
        run_root=run_root,
        loop_state=loop_state,
        approval=first_approval,
        approval_binding=binding,
        execution_specs=execution_specs,
    )
    if effect_error:
        raise ValueError(effect_error)
    loop_state["approvals"].append({
        **first_approval,
        "approval_binding": binding,
    })
    loop_state["current_approval_binding"] = binding
    action_loop_supervisor._persist_loop_state(run_root, loop_state)
    await action_loop_supervisor._apply_and_verify(
        request={
            "approval": first_approval,
            "execution_specs": execution_specs,
        },
        run_root=run_root,
        loop_state=loop_state,
        controller=repair_controller,
    )
    refreshed_state = json.loads(
        (run_root / "action_loop" / "state.json").read_text(encoding="utf-8")
    )
    if not isinstance(refreshed_state, dict):
        raise ValueError("Stage 5A repaired state is invalid")
    second_approval = _stage5a_approval("stage5a-approval-repair")
    await action_loop_supervisor._continue_locked_action_loop(
        request={
            "action": "approve_and_verify",
            "approval": second_approval,
            "execution_specs": execution_specs,
        },
        run_root=run_root,
        loop_state=refreshed_state,
        controller=repair_controller,
    )
    final_state = json.loads(
        (run_root / "action_loop" / "state.json").read_text(encoding="utf-8")
    )
    attempts = final_state.get("execution_attempts")
    if not isinstance(attempts, list):
        raise ValueError("Stage 5A execution journal is missing")
    return [
        dict(attempt)
        for attempt in attempts
        if isinstance(attempt, Mapping)
    ]


def _stage5a_approval(source_message_id: str) -> dict[str, object]:
    """Build one deterministic approval identity for a synthetic repair run."""

    return {
        "approved": True,
        "approved_by": "stage5a-reviewer",
        "approved_at": "2026-07-12T00:00:00Z",
        "approval_reason": "Approve the exact Stage 5A candidate.",
        "approval_evidence": {
            "source_message_id": source_message_id,
        },
    }


def _candidate_file_count(candidate_root: Path) -> int:
    """Count safe candidate files for the repository-scale gate."""

    return sum(1 for path in candidate_root.rglob("*") if path.is_file())


async def continue_evaluation_coding_run(
    request: Mapping[str, object],
    *,
    engine_id: str,
) -> dict[str, object]:
    """Continue one private benchmark lifecycle through its locked engine."""

    scenario_error = _scenario_continuation_error(request, engine_id=engine_id)
    if scenario_error:
        return {
            "status": "rejected",
            "run_id": str(request.get("run_id", "")),
            "limitations": [scenario_error],
        }
    if engine_id == "pipeline_v1":
        result = await continue_coding_run(dict(request))
        return result
    if engine_id != "action_loop_v1":
        return {
            "status": "rejected",
            "limitations": ["Benchmark engine is unsupported."],
        }
    workspace_root = request.get("workspace_root")
    run_id = request.get("run_id")
    if not isinstance(workspace_root, str) or not isinstance(run_id, str):
        return {
            "status": "rejected",
            "limitations": ["Evaluation continuation requires workspace and run id."],
        }
    state_path = (
        Path(workspace_root)
        / "coding_runs"
        / run_id
        / "action_loop"
        / "state.json"
    )
    if not state_path.is_file():
        return {
            "status": "rejected",
            "run_id": run_id,
            "limitations": ["Evaluation action-loop state is missing."],
        }
    if request.get("action") == "approve_and_verify":
        approval = request.get("approval")
        if not _structured_approval_is_valid(approval):
            return {
                "status": "rejected",
                "run_id": run_id,
                "limitations": ["Evaluation approval is incomplete."],
            }
    result = await continue_action_loop(
        request,
        run_root=state_path.parent.parent,
        controller=invoke_controller,
    )
    return result


def _structured_approval_is_valid(approval: object) -> bool:
    """Validate structural approval evidence without interpreting its meaning."""

    if not isinstance(approval, Mapping) or approval.get("approved") is not True:
        return False
    valid = all(
        isinstance(approval.get(key), str) and bool(approval[key].strip())
        for key in ("approved_by", "approved_at", "approval_reason")
    )
    return valid


async def _materialize_pipeline_scenario(
    request: Mapping[str, object],
    precondition: Mapping[str, object],
) -> dict[str, object]:
    """Acquire canonical locks before writing a legacy private scenario."""

    workspace_root = request.get("workspace_root")
    if not isinstance(workspace_root, str) or not workspace_root:
        raise ValueError("evaluation scenario workspace is required")
    run_id = new_run_id()
    lock_keys = build_lock_keys(
        run_id=run_id,
        source_identity=action_loop_supervisor._lock_source_identity(request),
    )
    async with acquire_workspace_locks(
        workspace_root=Path(workspace_root),
        keys=lock_keys,
        timeout_seconds=LOCK_TIMEOUT_SECONDS,
    ) as acquired:
        if not acquired:
            run_root = Path(workspace_root) / "coding_runs" / run_id
            busy_response = action_loop_supervisor._lock_busy_response(run_root)
            return busy_response
        result = await _materialize_locked_pipeline_scenario(
            request=request,
            precondition=precondition,
            workspace_root=workspace_root,
            run_id=run_id,
        )
        return result


async def _materialize_locked_pipeline_scenario(
    *,
    request: Mapping[str, object],
    precondition: Mapping[str, object],
    workspace_root: str,
    run_id: str,
) -> dict[str, object]:
    """Write the neutral proposal while canonical run/source locks are held."""

    paths = build_run_paths(
        workspace_root_text=workspace_root,
        run_id=run_id,
        create=True,
    )
    if isinstance(paths, str):
        raise ValueError(paths)
    source_result = await fetch_source(dict(request))
    if source_result["status"] != "succeeded":
        raise ValueError("evaluation scenario source could not be resolved")
    repository = source_result["repository"]
    source_scope = source_result["source_scope"]
    if not isinstance(repository, dict) or not isinstance(source_scope, dict):
        raise ValueError("evaluation scenario source identity is incomplete")
    source_root = _request_source_root(request)
    review = _scenario_mapping(precondition, "review")
    patch_artifacts = _scenario_list(review, "patch_artifacts")
    ledger = pipeline_supervisor._initial_ledger(
        run_id=run_id,
        request=dict(request),
        objective_type=str(request["objective_type"]),
    )
    ledger["status"] = "awaiting_approval"
    ledger["answer_text"] = str(review["summary"])
    ledger["repository"] = repository
    ledger["source_scope"] = source_scope
    ledger["patch_artifacts"] = patch_artifacts
    ledger["created_files"] = _scenario_list(review, "created_files")
    ledger["changed_files"] = _scenario_list(review, "changed_files")
    ledger["proposal_revision"] = 1
    ledger["patch_artifact_digest"] = patch_artifact_digest(patch_artifacts)
    ledger["execution_plan"] = derive_base_execution_plan(
        candidate_root=source_root,
        patch_artifacts=patch_artifacts,
        run_id=run_id,
        source_identity=pipeline_supervisor._proposal_source_identity(ledger),
        proposal_revision=1,
    )
    _bind_scenario_fields(ledger, precondition)
    _write_scenario_review(paths.run_dir, review, precondition)
    write_ledger(paths, ledger)
    pipeline_supervisor._record_event(
        paths=paths,
        ledger=ledger,
        event_type="proposal_ready",
        summary="Neutral evaluation proposal is ready for approval.",
        public_payload={
            "scenario_precondition_digest": precondition[
                "scenario_precondition_digest"
            ],
        },
    )
    write_ledger(paths, ledger)
    response = public_response(ledger=ledger, events=load_events(paths))
    return response


async def _materialize_action_loop_scenario(
    request: Mapping[str, object],
    precondition: Mapping[str, object],
) -> dict[str, object]:
    """Write the neutral proposal into the action loop's private state."""

    workspace_root = request.get("workspace_root")
    if not isinstance(workspace_root, str) or not workspace_root:
        raise ValueError("evaluation scenario workspace is required")
    run_root = Path(workspace_root) / "coding_runs" / uuid4().hex
    lock_keys = build_lock_keys(
        run_id=run_root.name,
        source_identity=action_loop_supervisor._lock_source_identity(request),
    )
    async with acquire_workspace_locks(
        workspace_root=Path(workspace_root),
        keys=lock_keys,
        timeout_seconds=LOCK_TIMEOUT_SECONDS,
    ) as acquired:
        if not acquired:
            busy_response = action_loop_supervisor._lock_busy_response(run_root)
            return busy_response
        result = await _materialize_locked_action_loop_scenario(
            request=request,
            precondition=precondition,
            run_root=run_root,
        )
        return result


async def _materialize_locked_action_loop_scenario(
    *,
    request: Mapping[str, object],
    precondition: Mapping[str, object],
    run_root: Path,
) -> dict[str, object]:
    """Materialize one engine-neutral scenario under canonical locks."""

    prepared = await action_loop_supervisor._prepare_run(
        request=request,
        run_root=run_root,
    )
    if prepared.get("status") != "ready":
        prepared_response = dict(prepared)
        return prepared_response
    loop_state = prepared.get("loop_state")
    if not isinstance(loop_state, dict):
        raise ValueError("evaluation action-loop state is missing")
    operation = _scenario_mapping(precondition, "patch_operation")
    records = _scenario_list(precondition, "canonical_operation_records")
    action = {
        "schema_version": "coding_action.v1",
        "action_id": "scenario-precondition-edit",
        "action": "edit",
        "reason": "Materialize the locked evaluation proposal.",
        "args": {
            "operation": operation["kind"],
            "repo_path": operation["path"],
            "expected_candidate_revision": operation[
                "expected_candidate_revision"
            ],
            "expected_sha256": operation["expected_source_sha256"],
            "replacement": operation["content"],
        },
    }
    action_loop_supervisor._append_action_record(
        run_root / "action_loop",
        1,
        action,
        operation_id=ENVIRONMENT_OPERATION_ID,
    )
    observation = execute_action(
        action=action,
        workspace_root=run_root.parent.parent,
        snapshot_id=str(loop_state["index_snapshot_id"]),
        run_root=run_root,
        objective_type=str(loop_state["objective_type"]),
        operation_id=ENVIRONMENT_OPERATION_ID,
    )
    if observation.get("outcome") != "ok":
        raise ValueError("evaluation action-loop proposal edit failed")
    evidence = observation.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evaluation action-loop edit evidence is missing")
    patch_operation = evidence[0].get("patch_operation")
    if not isinstance(patch_operation, dict):
        raise ValueError("evaluation action-loop operation is missing")
    if patch_operation.get("operation_id") != ENVIRONMENT_OPERATION_ID:
        raise ValueError("evaluation action-loop operation identity drifted")
    action_loop_supervisor._merge_edit_outcome(
        run_root=run_root,
        loop_state=loop_state,
        observation=observation,
    )
    observation["sequence"] = 1
    observation["action_sequence"] = 1
    observation["candidate_revision"] = loop_state["candidate_revision"]
    observation["index_snapshot_id"] = loop_state["index_snapshot_id"]
    action_loop_supervisor._append_jsonl(
        run_root / "action_loop" / "observations.jsonl",
        observation,
    )
    state_observations = loop_state["observations"]
    if not isinstance(state_observations, list):
        raise ValueError("evaluation action-loop observations are invalid")
    state_observations.append(observation)
    loop_state["action_count"] = 1
    loop_state["observation_count"] = 1
    finalization_error = action_loop_supervisor._finalize_candidate(
        run_root,
        loop_state,
    )
    if finalization_error:
        raise ValueError(finalization_error)
    if loop_state.get("canonical_operation_records") != records:
        raise ValueError("action-loop scenario operation identity mismatch")
    if loop_state.get("proposal_digest") != precondition.get("proposal_digest"):
        raise ValueError("action-loop scenario proposal identity mismatch")
    review = _scenario_mapping(precondition, "review")
    loop_state["status"] = "awaiting_approval"
    loop_state["answer_text"] = str(review["summary"])
    _bind_scenario_fields(loop_state, precondition)
    _write_scenario_review(run_root, review, precondition)
    action_loop_supervisor._persist_loop_state(run_root, loop_state)
    projection = action_loop_supervisor._public_projection(loop_state)
    return projection


def _bind_scenario_fields(
    state: dict[str, object],
    precondition: Mapping[str, object],
) -> None:
    """Bind neutral scenario identities into one engine-native state."""

    state["scenario_precondition_digest"] = precondition[
        "scenario_precondition_digest"
    ]
    state["scenario_review_sha256"] = precondition["review_sha256"]
    state["scenario_canonical_operation_records"] = precondition[
        "canonical_operation_records"
    ]
    state["scenario_proposal_digest"] = precondition["proposal_digest"]
    state["scenario_approval"] = precondition["approval"]
    state["scenario_execution_specs"] = precondition["execution_specs"]


def _write_scenario_review(
    run_root: Path,
    review: Mapping[str, object],
    precondition: Mapping[str, object],
) -> None:
    """Persist the identical neutral review artifact beside native state."""

    scenario_root = run_root / "evaluation_precondition"
    scenario_root.mkdir(parents=True, exist_ok=True)
    review_path = scenario_root / "review.json"
    content = _review_bytes(review)
    temporary_path = scenario_root / "review.json.tmp"
    with temporary_path.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(review_path)
    if hashlib.sha256(content).hexdigest() != precondition["review_sha256"]:
        raise ValueError("persisted scenario review identity mismatch")


def _request_source_root(request: Mapping[str, object]) -> Path:
    """Resolve the explicit local source root used by the private scenario."""

    root = request.get("local_root_hint")
    if not isinstance(root, str) or not root:
        raise ValueError("environment scenario requires a local source root")
    path = Path(root).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError("environment scenario source root is not a directory")
    return path


def _scenario_mapping(
    value: Mapping[str, object],
    key: str,
) -> dict[str, object]:
    candidate = value.get(key)
    if not isinstance(candidate, Mapping):
        raise ValueError(f"environment scenario {key} is invalid")
    mapped_candidate = dict(candidate)
    return mapped_candidate


def _scenario_list(
    value: Mapping[str, object],
    key: str,
) -> list[dict[str, object]]:
    candidate = value.get(key)
    if not isinstance(candidate, list) or not all(
        isinstance(item, Mapping) for item in candidate
    ):
        raise ValueError(f"environment scenario {key} is invalid")
    return [dict(item) for item in candidate]


def _canonical_digest(value: Mapping[str, object]) -> str:
    """Hash one semantic JSON object without presentation whitespace."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest


def _review_bytes(value: Mapping[str, object]) -> bytes:
    """Serialize the identical cross-engine review artifact."""

    content = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"{content}\n".encode("utf-8")


def _scenario_continuation_error(
    request: Mapping[str, object],
    *,
    engine_id: str,
) -> str:
    """Require a scenario continuation to replay its exact approval contract."""

    if request.get("action") != "approve_and_verify":
        return ""
    workspace_root = request.get("workspace_root")
    run_id = request.get("run_id")
    if not isinstance(workspace_root, str) or not isinstance(run_id, str):
        return ""
    run_root = Path(workspace_root) / "coding_runs" / run_id
    state_path = run_root / "run.json"
    if engine_id == "action_loop_v1":
        state_path = run_root / "action_loop" / "state.json"
    if not state_path.is_file():
        return ""
    loaded = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not loaded.get(
        "scenario_precondition_digest"
    ):
        return ""
    if request.get("approval") != loaded.get("scenario_approval"):
        return "Evaluation scenario approval identity mismatch."
    if request.get("execution_specs") != loaded.get(
        "scenario_execution_specs"
    ):
        return "Evaluation scenario execution identity mismatch."
    return ""
