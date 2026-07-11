"""Private benchmark-only coding-run engine selection."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from kazusa_ai_chatbot.coding_agent.code_action_loop import (
    supervisor as action_loop_supervisor,
)
from kazusa_ai_chatbot.coding_agent.code_action_loop.actions import execute_action
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
