"""Durable orchestration for one bounded coding action loop."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from itertools import count
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot import config as cfg
from kazusa_ai_chatbot.coding_agent.code_action_loop.actions import execute_action
from kazusa_ai_chatbot.coding_agent.code_action_loop.context import (
    render_controller_context,
)
from kazusa_ai_chatbot.coding_agent.code_action_loop.parser import parse_action
from kazusa_ai_chatbot.coding_agent.code_action_loop.prompts import CONTROLLER_PROMPT
from kazusa_ai_chatbot.coding_agent.code_action_loop.state import CandidateState
from kazusa_ai_chatbot.coding_agent.code_fetching import run as fetch_source
from kazusa_ai_chatbot.coding_agent.code_executing import execute_code_check
from kazusa_ai_chatbot.coding_agent.code_patching.apply import apply_approved_patch
from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
    build_canonical_operation_records,
    canonical_proposal_digest,
    compile_patch_operations,
)
from kazusa_ai_chatbot.coding_agent.coding_run.locking import (
    LOCK_TIMEOUT_SECONDS,
    acquire_workspace_locks,
    build_lock_keys,
)
from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
    build_repository_snapshot,
)
from kazusa_ai_chatbot.coding_agent.repository_index.identity import (
    source_identity_hash,
)
from kazusa_ai_chatbot.coding_agent.repository_index.storage import pin_snapshot
from kazusa_ai_chatbot.coding_agent.safety import managed_source_tree_digest
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output


MAX_RUN_ACTIONS = 8
MAX_SEGMENT_WALL_SECONDS = 1800
MAX_CONSECUTIVE_INVALID_OUTPUTS = 3

ControllerInvoker = Callable[..., Awaitable[dict[str, object]]]


async def start_action_loop(
    request: Mapping[str, object],
    *,
    run_root: Path,
    controller: ControllerInvoker,
) -> dict[str, object]:
    """Prepare and execute one private or public action-loop run.

    Args:
        request: Durable coding request after its objective was selected.
        run_root: Isolated run directory beneath the caller's workspace.
        controller: Single model-call boundary used once per semantic turn.

    Returns:
        Public-safe run projection and durable run identifier.
    """

    objective_type = request.get("objective_type", "read_only")
    allowed_actions = _objective_capabilities(objective_type)
    if allowed_actions is None:
        return {
            "status": "rejected",
            "limitations": ["Evaluation objective is unsupported."],
        }
    lock_keys = build_lock_keys(
        run_id=run_root.name,
        source_identity=_lock_source_identity(request),
    )
    async with acquire_workspace_locks(
        workspace_root=run_root.parent.parent,
        keys=lock_keys,
        timeout_seconds=LOCK_TIMEOUT_SECONDS,
    ) as acquired:
        if not acquired:
            busy_response = _lock_busy_response(run_root)
            return busy_response
        prepared = await _prepare_run(request=request, run_root=run_root)
        if prepared["status"] != "ready":
            return prepared
        loop_state = prepared["loop_state"]
        if not isinstance(loop_state, dict):
            raise ValueError("prepared action-loop state is invalid")
        result = await _run_controller_loop(
            run_root=run_root,
            loop_state=loop_state,
            allowed_actions=allowed_actions,
            controller=controller,
        )
        return result


async def continue_action_loop(
    request: Mapping[str, object],
    *,
    run_root: Path,
    controller: ControllerInvoker,
) -> dict[str, object]:
    """Continue one durable action-loop run through its existing identity."""

    state_path = run_root / "action_loop" / "state.json"
    if not state_path.is_file():
        return {
            "status": "rejected",
            "run_id": run_root.name,
            "limitations": ["Evaluation action-loop state is missing."],
        }
    state_value = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state_value, dict):
        raise ValueError("persisted action-loop state is invalid")
    lock_keys = build_lock_keys(
        run_id=run_root.name,
        source_identity=_lock_source_identity(state_value),
    )
    async with acquire_workspace_locks(
        workspace_root=run_root.parent.parent,
        keys=lock_keys,
        timeout_seconds=LOCK_TIMEOUT_SECONDS,
    ) as acquired:
        if not acquired:
            busy_response = _lock_busy_response(run_root)
            return busy_response
        refreshed_state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(refreshed_state, dict):
            raise ValueError("persisted action-loop state is invalid")
        result = await _continue_locked_action_loop(
            request=request,
            run_root=run_root,
            loop_state=refreshed_state,
            controller=controller,
        )
        return result


async def _continue_locked_action_loop(
    *,
    request: Mapping[str, object],
    run_root: Path,
    loop_state: dict[str, object],
    controller: ControllerInvoker,
) -> dict[str, object]:
    """Continue after the caller has acquired the canonical run/source locks."""

    action = request.get("action")
    if action == "status":
        projection = _public_projection(loop_state)
        return projection
    if action == "cancel":
        loop_state["status"] = "cancelled"
        _persist_loop_state(run_root, loop_state)
        projection = _public_projection(loop_state)
        return projection
    if loop_state.get("status") in {"applying", "verifying"}:
        pending_effect = loop_state.get("pending_effect")
        if not isinstance(pending_effect, Mapping):
            raise ValueError("pending continuation effect is missing")
        saved_approval = pending_effect.get("approval")
        saved_execution_specs = pending_effect.get("execution_specs")
        if not isinstance(saved_approval, Mapping) or not isinstance(
            saved_execution_specs,
            list,
        ):
            raise ValueError("pending continuation effect is incomplete")
        result = await _apply_and_verify(
            request={
                "approval": dict(saved_approval),
                "execution_specs": saved_execution_specs,
            },
            run_root=run_root,
            loop_state=loop_state,
            controller=controller,
        )
        return result
    if action == "respond_to_blocker":
        answer = request.get("revision_instruction")
        blocker = loop_state.get("blocker")
        if (
            loop_state["status"] != "blocked"
            or not isinstance(answer, str)
            or not answer.strip()
            or not isinstance(blocker, Mapping)
            or blocker.get("resume_target") == "none"
        ):
            return {
                "status": "rejected",
                "run_id": run_root.name,
                "limitations": ["Evaluation blocker response is invalid."],
            }
        observations = loop_state["observations"]
        if not isinstance(observations, list):
            raise ValueError("persisted action-loop observations are invalid")
        observation = {
            "sequence": _next_observation_sequence(loop_state, observations),
            "outcome": "ok",
            "kind": "user_blocker_response",
            "summary": "The user supplied a response to the active blocker.",
            "evidence": [{"user_answer": answer}],
        }
        observations.append(observation)
        _append_jsonl(run_root / "action_loop" / "observations.jsonl", observation)
        loop_state["status"] = "active"
        loop_state.pop("blocker", None)
        loop_state["consecutive_no_progress_count"] = 0
        loop_state["invalid_output_count"] = 0
        loop_state["run_action_count"] = 0
        loop_state["segment_started_at_epoch_seconds"] = int(time.time())
        loop_state.pop("last_no_progress_signature", None)
        loop_state.pop("last_no_progress_evidence", None)
        _persist_active_loop_state(run_root, loop_state)
        allowed_actions = _objective_capabilities(loop_state["objective_type"])
        if allowed_actions is None:
            raise ValueError("persisted action-loop objective is unsupported")
        result = await _run_controller_loop(
            run_root=run_root,
            loop_state=loop_state,
            allowed_actions=allowed_actions,
            controller=controller,
        )
        return result
    if action != "approve_and_verify" or loop_state["status"] != "awaiting_approval":
        return {
            "status": "rejected",
            "run_id": run_root.name,
            "limitations": ["Evaluation continuation is unavailable."],
        }
    approval = request.get("approval")
    if not isinstance(approval, Mapping):
        return {
            "status": "rejected",
            "run_id": run_root.name,
            "limitations": ["Evaluation approval is incomplete."],
        }
    binding, binding_error = _bind_approval(
        run_root=run_root,
        loop_state=loop_state,
        approval=approval,
    )
    if binding_error:
        return {
            "status": "rejected",
            "run_id": run_root.name,
            "limitations": [binding_error],
        }
    if binding is None:
        raise ValueError("approval binding was not materialized")
    effect_error = _initialize_pending_effect(
        run_root=run_root,
        loop_state=loop_state,
        approval=approval,
        approval_binding=binding,
        execution_specs=request.get("execution_specs", []),
    )
    if effect_error:
        return {
            "status": "rejected",
            "run_id": run_root.name,
            "limitations": [effect_error],
        }
    approvals = loop_state.get("approvals")
    if not isinstance(approvals, list):
        approvals = []
        loop_state["approvals"] = approvals
    approval_record = dict(approval)
    approval_record["approval_binding"] = binding
    approvals.append(approval_record)
    loop_state["current_approval_binding"] = binding
    _persist_loop_state(run_root, loop_state)
    verification_result = await _apply_and_verify(
        request=request,
        run_root=run_root,
        loop_state=loop_state,
        controller=controller,
    )
    return verification_result


async def _prepare_run(
    *,
    request: Mapping[str, object],
    run_root: Path,
) -> dict[str, object]:
    """Resolve source, create a candidate, and pin its immutable base index."""

    run_root.mkdir(parents=True, exist_ok=False)
    workspace_root = run_root.parent.parent
    objective_type = request.get("objective_type", "read_only")
    source_request = dict(request)
    repository: dict[str, object] | None = None
    source_scope: dict[str, object] | None = None
    source_root: Path | None = None
    if any(
        request.get(key)
        for key in (
            "source_url",
            "repo_url",
            "repo_hint",
            "local_root_hint",
            "local_path_hint",
            "inline_sources",
        )
    ):
        source_result = await fetch_source(source_request)
        if source_result["status"] != "succeeded":
            response = {
                "status": source_result["status"],
                "run_id": run_root.name,
                "limitations": source_result["limitations"],
                "trace_summary": source_result["trace_summary"],
            }
            _write_json(run_root / "run.json", response)
            return response
        repository = dict(source_result["repository"] or {})
        source_scope = dict(source_result["source_scope"] or {})
        local_root = repository.get("local_root")
        if not isinstance(local_root, str) or not local_root:
            raise ValueError("resolved repository omitted its local root")
        source_root = Path(local_root)
        source_identity = {
            key: repository[key]
            for key in (
                "provider",
                "owner",
                "repo",
                "requested_ref",
                "resolved_ref",
                "current_commit",
                "dirty_state",
            )
            if key in repository
        }
    elif objective_type == "read_only":
        response = {
            "status": "needs_user_input",
            "run_id": run_root.name,
            "limitations": ["A source is required for read-only evaluation."],
        }
        _write_json(run_root / "run.json", response)
        return response
    else:
        source_identity = {
            "provider": "source_free",
            "run_id": run_root.name,
        }

    candidate = CandidateState.create(
        run_root / "candidate",
        source_root=source_root,
    )
    index_source_root = source_root or candidate.root / "source"
    snapshot = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=index_source_root,
        source_identity=source_identity,
    )
    if snapshot.get("status") != "complete":
        response = {
            "status": "blocked",
            "run_id": run_root.name,
            "blocker": snapshot,
            "limitations": ["Repository indexing could not complete."],
        }
        _write_json(run_root / "run.json", response)
        return response
    pin_snapshot(
        workspace_root=workspace_root,
        source_identity=source_identity,
        snapshot_id=str(snapshot["snapshot_id"]),
        owner_id=run_root.name,
    )
    goal = request.get("question")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("action-loop goal is required")
    acceptance_value = request.get("acceptance_criteria")
    if isinstance(acceptance_value, list) and all(
        isinstance(item, str) and item.strip()
        for item in acceptance_value
    ):
        acceptance_criteria = list(acceptance_value)
    else:
        acceptance_criteria = [goal]
    loop_state: dict[str, object] = {
        "schema_version": "coding_action_loop_state.v1",
        "engine_id": "action_loop_v1",
        "run_id": run_root.name,
        "objective_type": objective_type,
        "goal": goal,
        "acceptance_criteria": acceptance_criteria,
        "status": "active",
        "index_snapshot_id": snapshot["snapshot_id"],
        "index_source_identity": source_identity,
        "source_identity_digest": source_identity_hash(source_identity),
        "lock_source_identity": _lock_source_identity(request),
        "candidate_revision": candidate.revision,
        "overlay_revision": candidate.revision,
        "action_count": 0,
        "invalid_output_count": 0,
        "consecutive_no_progress_count": 0,
        "working_note": "",
        "changed_paths": [],
        "patch_operations": [],
        "approvals": [],
        "apply_attempts": [],
        "execution_attempts": [],
        "effect_history": [],
        "current_failure": None,
        "observations": [],
        "observation_count": 0,
        "started_at_epoch_seconds": int(time.time()),
        "segment_started_at_epoch_seconds": int(time.time()),
        "run_action_count": 0,
        "source_request": {
            key: request[key]
            for key in (
                "local_root_hint",
                "local_path_hint",
                "source_url",
                "repo_url",
                "repo_hint",
                "source_scope_hint",
            )
            if key in request
        },
        "repository": repository,
        "source_scope": source_scope,
    }
    _write_json(run_root / "action_loop" / "state.json", loop_state)
    _write_run_ledger(run_root, loop_state)
    prepared = {"status": "ready", "loop_state": loop_state}
    return prepared


async def _run_controller_loop(
    *,
    run_root: Path,
    loop_state: dict[str, object],
    allowed_actions: set[str],
    controller: ControllerInvoker,
) -> dict[str, object]:
    """Run validated controller turns until finish, block, or a hard budget."""

    loop_root = run_root / "action_loop"
    loop_root.mkdir(parents=True, exist_ok=True)
    observations = loop_state["observations"]
    if not isinstance(observations, list):
        raise ValueError("action-loop observations are invalid")
    reconciliation = _reconcile_orphan_action(loop_root, loop_state, observations)
    if reconciliation is not None:
        _persist_loop_state(run_root, loop_state)
        return reconciliation
    try:
        candidate_state = CandidateState.load(run_root / "candidate")
        candidate_state.require_recovered_before_next_action()
        if candidate_state.revision != loop_state["candidate_revision"]:
            raise ValueError("candidate revision does not match loop state")
    except (OSError, ValueError) as exc:
        safe_error = str(exc).replace(str(run_root), "<managed_run>")
        loop_state["status"] = "blocked"
        loop_state["blocker"] = {
            "blocker_type": "candidate_recovery_failed",
            "code": "candidate_state_validation_failed",
            "detail": safe_error[:1000],
            "resume_target": "retry_loop",
        }
        _persist_loop_state(run_root, loop_state)
        projection = _public_projection(loop_state)
        return projection
    _persist_active_loop_state(run_root, loop_state)
    starting_sequence = int(loop_state["action_count"])
    loop_state["segment_started_at_epoch_seconds"] = int(time.time())
    for sequence in count(starting_sequence + 1):
        effective_actions = set(allowed_actions)
        trusted_run_context = loop_state.get("trusted_execution_context")
        if isinstance(trusted_run_context, dict):
            effective_actions.add("run")
        elapsed_seconds = time.time() - int(
            loop_state["segment_started_at_epoch_seconds"],
        )
        if elapsed_seconds >= MAX_SEGMENT_WALL_SECONDS:
            loop_state["status"] = "blocked"
            loop_state["blocker"] = _budget_blocker(
                "wall_time",
                _latest_safe_evidence(observations),
            )
            break
        if int(loop_state.get("run_action_count", 0)) >= MAX_RUN_ACTIONS:
            loop_state["status"] = "blocked"
            loop_state["blocker"] = _budget_blocker(
                "run_action",
                _latest_safe_evidence(observations),
            )
            break
        context = render_controller_context(
            goal=str(loop_state["goal"]),
            acceptance_criteria=list(loop_state["acceptance_criteria"]),
            capabilities=sorted(effective_actions),
            source_identity_digest=str(loop_state["source_identity_digest"]),
            candidate_revision=int(loop_state["candidate_revision"]),
            changed_paths=list(loop_state["changed_paths"]),
            current_failure=(
                loop_state.get("current_failure")
                if isinstance(loop_state.get("current_failure"), Mapping)
                else None
            ),
            working_notes=str(loop_state["working_note"]),
            observations=observations,
        )
        _write_context_manifest(loop_root, sequence, context, observations)
        controller_result = await controller(
            context=context,
            allowed_actions=effective_actions,
        )
        if controller_result.get("status") == "blocked":
            loop_state["status"] = "blocked"
            loop_state["blocker"] = {
                "blocker_type": "environment",
                "code": "controller_configuration_missing",
                "resume_target": "retry_loop",
            }
            break
        action = controller_result.get("action")
        if controller_result.get("status") != "ok" or not isinstance(action, dict):
            invalid_count = int(loop_state["invalid_output_count"]) + 1
            loop_state["invalid_output_count"] = invalid_count
            raw_output = controller_result.get("raw_output")
            _append_invalid_controller_record(
                loop_root=loop_root,
                sequence=sequence,
                controller_result=controller_result,
                raw_output=raw_output if isinstance(raw_output, str) else None,
            )
            validation_message = controller_result.get("message")
            if not isinstance(validation_message, str):
                validation_message = str(
                    controller_result.get("status", "invalid_action")
                )
            observation = {
                "sequence": _next_observation_sequence(
                    loop_state,
                    observations,
                ),
                "action_sequence": sequence,
                "outcome": "rejected",
                "kind": str(controller_result.get("status", "invalid_action")),
                "summary": (
                    f"coding_action.v1 rejected: {validation_message}. "
                    f"Allowed actions: {', '.join(sorted(effective_actions))}."
                ),
            }
            _append_jsonl(loop_root / "observations.jsonl", observation)
            observations.append(observation)
            loop_state["action_count"] = sequence
            _persist_active_loop_state(run_root, loop_state)
            if invalid_count >= MAX_CONSECUTIVE_INVALID_OUTPUTS:
                loop_state["status"] = "blocked"
                loop_state["blocker"] = {
                    "blocker_type": "controller_contract_failure",
                    "resume_target": "retry_loop",
                }
                break
            continue
        loop_state["invalid_output_count"] = 0
        raw_output = controller_result.get("raw_output")
        operation_id = _dispatch_operation_id(
            run_id=str(loop_state["run_id"]),
            sequence=sequence,
            action=action,
        )
        _append_action_record(
            loop_root,
            sequence,
            action,
            operation_id=operation_id,
            raw_output=raw_output if isinstance(raw_output, str) else None,
        )
        action_name = action.get("action")
        run_spec_digest: str | None = None
        if action_name == "run":
            if not isinstance(trusted_run_context, dict):
                raise ValueError("run action omitted trusted execution context")
            run_spec_digest = _persist_run_execution_spec(
                loop_root=loop_root,
                sequence=sequence,
                action=action,
                run_context=trusted_run_context,
            )
        candidate_revision_before = int(loop_state["candidate_revision"])
        observation = execute_action(
            action=action,
            workspace_root=run_root.parent.parent,
            snapshot_id=str(loop_state["index_snapshot_id"]),
            run_root=run_root,
            objective_type=str(loop_state["objective_type"]),
            run_context=(
                trusted_run_context
                if isinstance(trusted_run_context, dict)
                else None
            ),
            operation_id=operation_id,
        )
        if action_name == "edit" and observation.get("outcome") == "ok":
            _merge_edit_outcome(
                run_root=run_root,
                loop_state=loop_state,
                observation=observation,
            )
        observation["sequence"] = _next_observation_sequence(
            loop_state,
            observations,
        )
        observation["action_sequence"] = sequence
        observation["candidate_revision"] = loop_state["candidate_revision"]
        observation["index_snapshot_id"] = loop_state["index_snapshot_id"]
        observation["request"] = {
            "action": action["action"],
            "args": action["args"],
        }
        if run_spec_digest is not None:
            _persist_run_execution_result(
                loop_root=loop_root,
                sequence=sequence,
                spec_digest=run_spec_digest,
                observation=observation,
            )
        _append_jsonl(loop_root / "observations.jsonl", observation)
        observations.append(observation)
        loop_state["action_count"] = sequence
        working_note = action.get("working_note")
        if isinstance(working_note, str):
            loop_state["working_note"] = working_note
        if action_name == "run":
            run_count = int(loop_state.get("run_action_count", 0))
            loop_state["run_action_count"] = run_count + 1
            if observation.get("outcome") == "ok":
                loop_state["current_failure"] = None
        no_progress_exhausted = _record_no_progress_signature(
            loop_state=loop_state,
            action=action,
            observation=observation,
            candidate_revision_before=candidate_revision_before,
        )
        _persist_active_loop_state(run_root, loop_state)
        if no_progress_exhausted:
            loop_state["status"] = "blocked"
            loop_state["blocker"] = _no_progress_blocker(loop_state)
            break
        if action_name == "finish":
            finish_args = action["args"]
            if not isinstance(finish_args, dict):
                raise ValueError("finish action arguments are invalid")
            if loop_state["objective_type"] == "read_only":
                loop_state["answer_text"] = finish_args["summary"]
                loop_state["status"] = "completed"
                break
            finalization_error = _finalize_candidate(run_root, loop_state)
            if finalization_error:
                loop_state["current_failure"] = {
                    "kind": "finalization_failed",
                    "summary": finalization_error,
                    "candidate_revision": int(loop_state["candidate_revision"]),
                    "effect_id": "",
                    "execution_index": None,
                }
                finalization_observation = {
                    "sequence": _next_observation_sequence(
                        loop_state,
                        observations,
                    ),
                    "action_sequence": sequence,
                    "outcome": "failed",
                    "kind": "finalization_failed",
                    "summary": finalization_error,
                }
                _append_jsonl(
                    loop_root / "observations.jsonl",
                    finalization_observation,
                )
                observations.append(finalization_observation)
                _persist_active_loop_state(run_root, loop_state)
                continue
            loop_state["answer_text"] = finish_args["summary"]
            loop_state["status"] = "awaiting_approval"
            break
        if action_name == "block":
            block_args = action["args"]
            if not isinstance(block_args, dict):
                raise ValueError("block action arguments are invalid")
            loop_state["status"] = "blocked"
            loop_state["blocker"] = _deterministic_blocker(
                block_args,
                loop_state=loop_state,
            )
            break
    _persist_loop_state(run_root, loop_state)
    result = _public_projection(loop_state)
    return result


def _objective_capabilities(objective_type: object) -> set[str] | None:
    """Return the one protocol's semantic action set for an objective."""

    if objective_type == "read_only":
        return {"read", "search", "note", "finish", "block"}
    if objective_type in {"propose_patch", "verify_repair"}:
        return {"read", "search", "edit", "note", "finish", "block"}
    return None


def _lock_source_identity(value: Mapping[str, object]) -> dict[str, object] | None:
    """Build the same stable source identity used by the Phase C lock owner."""

    stored_identity = value.get("lock_source_identity")
    if isinstance(stored_identity, Mapping):
        source_identity = dict(stored_identity)
        return source_identity
    repository = value.get("repository")
    if isinstance(repository, Mapping):
        source_identity = {
            key: repository[key]
            for key in (
                "provider",
                "owner",
                "repo",
                "requested_ref",
                "resolved_ref",
                "current_commit",
            )
            if key in repository
        }
        return source_identity or None
    source_identity = {
        key: value[key]
        for key in (
            "source_url",
            "repo_url",
            "repo_hint",
            "local_root_hint",
            "local_path_hint",
            "requested_ref",
            "source_scope_hint",
        )
        if key in value and isinstance(value[key], str)
    }
    return source_identity or None


def _lock_busy_response(run_root: Path) -> dict[str, object]:
    """Return a typed retry-loop blocker without mutating durable state."""

    return {
        "status": "blocked",
        "run_id": run_root.name,
        "blocker": {
            "blocker_type": "budget",
            "code": "coding_run_lock_unavailable",
            "resume_target": "retry_loop",
        },
        "limitations": ["Coding action-loop locks are currently unavailable."],
    }


def _record_no_progress_signature(
    *,
    loop_state: dict[str, object],
    action: Mapping[str, object],
    observation: Mapping[str, object],
    candidate_revision_before: int,
) -> bool:
    """Persist consecutive identical no-progress evidence for one loop turn.

    The controller's identifiers and narrative are deliberately absent from the
    signature. Only the validated semantic request, the candidate revision
    before execution, and stable safe observation identity participate.
    """

    action_name = action.get("action")
    action_args = action.get("args")
    if not isinstance(action_name, str) or not isinstance(action_args, Mapping):
        raise ValueError("validated action omitted its canonical signature data")
    safe_observation = _stable_observation_identity(observation)
    signature_data = {
        "action": action_name,
        "args": _stable_value(action_args),
        "candidate_revision_before": candidate_revision_before,
        "observation": safe_observation,
    }
    signature = hashlib.sha256(
        json.dumps(
            signature_data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    previous_signature = loop_state.get("last_no_progress_signature")
    if previous_signature == signature:
        count = int(loop_state.get("consecutive_no_progress_count", 0)) + 1
    else:
        count = 1
    loop_state["last_no_progress_signature"] = signature
    loop_state["last_no_progress_evidence"] = safe_observation
    loop_state["consecutive_no_progress_count"] = count
    return count >= 3


def _stable_observation_identity(
    observation: Mapping[str, object],
) -> dict[str, object]:
    """Keep only stable prompt-safe outcome and evidence identity fields."""

    stable: dict[str, object] = {}
    for key in ("outcome", "kind", "code", "status", "evidence"):
        if key in observation:
            stable[key] = _stable_value(observation[key])
    return stable


def _stable_value(value: object) -> object:
    """Recursively remove volatile fields before durable signature hashing."""

    volatile_keys = {
        "action_id",
        "reason",
        "working_note",
        "sequence",
        "timestamp",
        "artifact_path",
        "artifact_paths",
        "run_id",
        "workspace_root",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
            if str(key) not in volatile_keys
        }
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_stable_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError("action-loop signature value is not JSON-compatible")


def _no_progress_blocker(loop_state: Mapping[str, object]) -> dict[str, object]:
    """Bind repeated safe evidence to the plan-required loop budget blocker."""

    latest_evidence = loop_state.get("last_no_progress_evidence")
    if not isinstance(latest_evidence, Mapping):
        raise ValueError("no-progress blocker omitted its latest safe evidence")
    return {
        "blocker_type": "budget",
        "code": "controller_no_progress_budget_exhausted",
        "resume_target": "retry_loop",
        "latest_safe_evidence": dict(latest_evidence),
    }


def _budget_blocker(
    budget_name: str,
    latest_safe_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Render one durable retry-loop blocker for a deterministic budget."""

    return {
        "blocker_type": "budget",
        "code": f"controller_{budget_name}_budget_exhausted",
        "resume_target": "retry_loop",
        "latest_safe_evidence": dict(latest_safe_evidence),
    }


def _latest_safe_evidence(observations: list[object]) -> dict[str, object]:
    """Return the newest persisted prompt-safe evidence for a budget blocker."""

    for observation in reversed(observations):
        if isinstance(observation, Mapping):
            evidence = _stable_observation_identity(observation)
            return evidence
    return {}


def _next_observation_sequence(
    loop_state: dict[str, object],
    observations: list[object],
) -> int:
    """Advance the observation ledger independently from action dispatch."""

    persisted_count = loop_state.get("observation_count", 0)
    if not isinstance(persisted_count, int) or isinstance(persisted_count, bool):
        raise ValueError("action-loop observation count is invalid")
    existing_sequences = [
        observation["sequence"]
        for observation in observations
        if isinstance(observation, Mapping)
        and isinstance(observation.get("sequence"), int)
        and not isinstance(observation.get("sequence"), bool)
    ]
    next_sequence = max([persisted_count, *existing_sequences], default=0) + 1
    loop_state["observation_count"] = next_sequence
    return next_sequence


def _reconcile_orphan_action(
    loop_root: Path,
    loop_state: dict[str, object],
    observations: list[object],
) -> dict[str, object] | None:
    """Reconcile an action record without an observation before controller use."""

    action_path = loop_root / "actions.jsonl"
    observation_path = loop_root / "observations.jsonl"
    actions = _read_jsonl(action_path) if action_path.is_file() else []
    action_by_sequence = {
        row["sequence"]: row
        for row in actions
        if isinstance(row.get("sequence"), int)
    }
    lagged_terminal_action: Mapping[str, object] | None = None
    for persisted_observation in _read_jsonl(observation_path):
        observation_was_persisted_in_state = persisted_observation in observations
        if not observation_was_persisted_in_state:
            observations.append(persisted_observation)
        sequence = persisted_observation.get("action_sequence")
        if (
            isinstance(sequence, int)
        ):
            loop_state["action_count"] = max(
                int(loop_state.get("action_count", 0)),
                sequence,
            )
            action_record = action_by_sequence.get(sequence)
            parsed_action = (
                action_record.get("parsed_action")
                if isinstance(action_record, Mapping)
                else None
            )
            if isinstance(parsed_action, Mapping):
                working_note = parsed_action.get("working_note")
                if isinstance(working_note, str):
                    loop_state["working_note"] = working_note
                if (
                    not observation_was_persisted_in_state
                    and parsed_action.get("action") == "edit"
                    and persisted_observation.get("outcome") == "ok"
                ):
                    _merge_edit_outcome(
                        run_root=loop_root.parent,
                        loop_state=loop_state,
                        observation=persisted_observation,
                    )
        if (
            not observation_was_persisted_in_state
            and persisted_observation.get("kind") == "run_result"
        ):
            loop_state["run_action_count"] = (
                int(loop_state.get("run_action_count", 0)) + 1
            )
    if lagged_terminal_action is None and loop_state.get("status") == "active":
        last_action_record = action_by_sequence.get(
            int(loop_state.get("action_count", 0))
        )
        last_action = (
            last_action_record.get("parsed_action")
            if isinstance(last_action_record, Mapping)
            else None
        )
        if (
            isinstance(last_action, Mapping)
            and last_action.get("action") in {"finish", "block"}
            and any(
                isinstance(row, Mapping)
                and row.get("action_sequence")
                == int(loop_state.get("action_count", 0))
                for row in observations
            )
        ):
            lagged_terminal_action = last_action
    if lagged_terminal_action is not None:
        terminal_action_name = lagged_terminal_action.get("action")
        terminal_sequence = int(loop_state.get("action_count", 0))
        finalization_failure = next(
            (
                row
                for row in observations
                if isinstance(row, Mapping)
                and row.get("action_sequence") == terminal_sequence
                and row.get("kind") == "finalization_failed"
            ),
            None,
        )
        if terminal_action_name == "finish" and isinstance(
            finalization_failure,
            Mapping,
        ):
            loop_state["status"] = "active"
            loop_state["current_failure"] = {
                "kind": "finalization_failed",
                "summary": str(finalization_failure.get("summary", "")),
                "candidate_revision": int(loop_state["candidate_revision"]),
                "effect_id": "",
                "execution_index": None,
            }
        else:
            terminal_result = _recover_observed_terminal_action(
                run_root=loop_root.parent,
                loop_root=loop_root,
                loop_state=loop_state,
                action=lagged_terminal_action,
                observations=observations,
            )
            if terminal_result is not None:
                return terminal_result
    if not action_path.is_file():
        return None
    observed = {
        row.get("action_sequence")
        for row in observations
        if isinstance(row, Mapping)
        and isinstance(row.get("action_sequence"), int)
    }
    orphans = [row for row in actions if row.get("sequence") not in observed]
    if not orphans:
        return None
    orphan = orphans[0]
    action = orphan.get("parsed_action")
    if isinstance(action, Mapping) and action.get("action") == "run":
        recovered_observation, failure_code = _recover_run_execution_result(
            loop_root=loop_root,
            sequence=orphan.get("sequence"),
            action=action,
            loop_state=loop_state,
        )
        if recovered_observation is not None:
            if not isinstance(recovered_observation.get("sequence"), int):
                recovered_observation["sequence"] = _next_observation_sequence(
                    loop_state,
                    observations,
                )
            observations.append(recovered_observation)
            _append_jsonl(observation_path, recovered_observation)
            loop_state["action_count"] = recovered_observation[
                "action_sequence"
            ]
            loop_state["run_action_count"] = (
                int(loop_state.get("run_action_count", 0)) + 1
            )
            return None
        loop_state["status"] = "blocked"
        loop_state["blocker"] = {
            "blocker_type": "candidate_recovery_failed",
            "code": failure_code,
            "resume_target": "retry_loop",
        }
        projection = _public_projection(loop_state)
        return projection
    if isinstance(action, Mapping) and action.get("action") == "edit":
        try:
            candidate = CandidateState.load(loop_root.parent / "candidate")
            candidate.recover()
            candidate = CandidateState.load(loop_root.parent / "candidate")
        except (OSError, ValueError) as exc:
            safe_error = str(exc).replace(
                str(loop_root.parent),
                "<managed_run>",
            )
            loop_state["status"] = "blocked"
            loop_state["blocker"] = {
                "blocker_type": "candidate_recovery_failed",
                "code": "candidate_journal_recovery_failed",
                "detail": safe_error[:1000],
                "resume_target": "retry_loop",
            }
            projection = _public_projection(loop_state)
            return projection
        journal = candidate.journal
        orphan_operation_id = orphan.get("operation_id")
        journal_operation = next(
            (
                row
                for row in journal
                if row.get("operation_id") == orphan_operation_id
            ),
            None,
        )
        expected_revisions = {
            loop_state.get("candidate_revision"),
            int(loop_state.get("candidate_revision", 0)) + 1,
        }
        if (
            isinstance(journal_operation, dict)
            and journal_operation.get("state") == "committed"
            and candidate.revision in expected_revisions
        ):
            observation = _reconstruct_edit_observation(
                sequence=orphan.get("sequence"),
                action=action,
                journal_operation=journal_operation,
                candidate_revision=candidate.revision,
            )
            observation["sequence"] = _next_observation_sequence(
                loop_state,
                observations,
            )
            observations.append(observation)
            _append_jsonl(observation_path, observation)
            loop_state["action_count"] = orphan.get("sequence")
            _merge_edit_outcome(
                run_root=loop_root.parent,
                loop_state=loop_state,
                observation=observation,
            )
            return None
        if (
            isinstance(journal_operation, dict)
            and journal_operation.get("state") == "rolled_back"
        ):
            observation = {
                "sequence": _next_observation_sequence(
                    loop_state,
                    observations,
                ),
                "action_sequence": orphan.get("sequence"),
                "outcome": "rejected",
                "kind": "edit_recovered_rolled_back",
                "summary": "Interrupted candidate edit was rolled back safely.",
                "evidence": [],
            }
            observations.append(observation)
            _append_jsonl(observation_path, observation)
            loop_state["action_count"] = orphan.get("sequence")
            return None
    if not isinstance(action, Mapping) or action.get("action") in {"edit", "run"}:
        loop_state["status"] = "blocked"
        loop_state["blocker"] = {
            "blocker_type": "candidate_recovery_failed",
            "code": "orphan_action_irreconcilable",
            "resume_target": "retry_loop",
        }
        projection = _public_projection(loop_state)
        return projection
    action_dict = dict(action)
    observation = execute_action(
        action=action_dict,
        workspace_root=loop_root.parent.parent.parent,
        snapshot_id=str(loop_state["index_snapshot_id"]),
        run_root=loop_root.parent,
        objective_type=str(loop_state["objective_type"]),
        run_context=(
            loop_state.get("trusted_execution_context")
            if isinstance(loop_state.get("trusted_execution_context"), dict)
            else None
        ),
        operation_id=(
            str(orphan["operation_id"])
            if isinstance(orphan.get("operation_id"), str)
            else None
        ),
    )
    observation["sequence"] = _next_observation_sequence(
        loop_state,
        observations,
    )
    observation["action_sequence"] = orphan.get("sequence")
    observations.append(observation)
    _append_jsonl(observation_path, observation)
    loop_state["action_count"] = orphan.get("sequence")
    working_note = action.get("working_note")
    if isinstance(working_note, str):
        loop_state["working_note"] = working_note
    if action.get("action") == "finish":
        finish_args = action.get("args")
        if not isinstance(finish_args, Mapping):
            raise ValueError("orphan finish arguments are invalid")
        if loop_state["objective_type"] == "read_only":
            loop_state["answer_text"] = finish_args["summary"]
            loop_state["status"] = "completed"
        else:
            error = _finalize_candidate(loop_root.parent, loop_state)
            if error:
                loop_state["status"] = "active"
                loop_state["current_failure"] = {
                    "kind": "finalization_failed",
                    "summary": error,
                    "candidate_revision": int(
                        loop_state["candidate_revision"],
                    ),
                    "effect_id": "",
                    "execution_index": None,
                }
                finalization_observation = {
                    "sequence": _next_observation_sequence(
                        loop_state,
                        observations,
                    ),
                    "action_sequence": orphan.get("sequence"),
                    "outcome": "failed",
                    "kind": "finalization_failed",
                    "summary": error,
                }
                observations.append(finalization_observation)
                _append_jsonl(observation_path, finalization_observation)
                return None
            else:
                loop_state["answer_text"] = finish_args["summary"]
                loop_state["status"] = "awaiting_approval"
        projection = _public_projection(loop_state)
        return projection
    if action.get("action") == "block":
        block_args = action.get("args")
        if not isinstance(block_args, Mapping):
            raise ValueError("orphan block arguments are invalid")
        loop_state["status"] = "blocked"
        loop_state["blocker"] = _deterministic_blocker(
            block_args,
            loop_state=loop_state,
        )
        projection = _public_projection(loop_state)
        return projection
    return None


def _persist_run_execution_spec(
    *,
    loop_root: Path,
    sequence: int,
    action: Mapping[str, object],
    run_context: Mapping[str, object],
) -> str:
    """Persist the deterministic identity of one approved run request."""

    spec_digest = _run_execution_spec_digest(action, run_context)
    artifact = {
        "schema_version": "coding_run_execution.v1",
        "sequence": sequence,
        "status": "prepared",
        "spec_digest": spec_digest,
    }
    _write_json(
        loop_root / "run_executions" / str(sequence) / "spec.json",
        artifact,
    )
    return spec_digest


def _recover_observed_terminal_action(
    *,
    run_root: Path,
    loop_root: Path,
    loop_state: dict[str, object],
    action: Mapping[str, object],
    observations: list[object],
) -> dict[str, object] | None:
    """Complete state lag after a terminal observation became durable."""

    action_name = action.get("action")
    arguments = action.get("args")
    if action_name not in {"finish", "block"}:
        return None
    if not isinstance(arguments, Mapping):
        raise ValueError("terminal action arguments are invalid")
    if action_name == "block":
        loop_state["status"] = "blocked"
        loop_state["blocker"] = _deterministic_blocker(
            arguments,
            loop_state=loop_state,
        )
        projection = _public_projection(loop_state)
        return projection
    if loop_state["objective_type"] == "read_only":
        loop_state["answer_text"] = arguments["summary"]
        loop_state["status"] = "completed"
        projection = _public_projection(loop_state)
        return projection
    finalization_error = _finalize_candidate(run_root, loop_state)
    if finalization_error:
        loop_state["status"] = "active"
        loop_state["current_failure"] = {
            "kind": "finalization_failed",
            "summary": finalization_error,
            "candidate_revision": int(loop_state["candidate_revision"]),
            "effect_id": "",
            "execution_index": None,
        }
        finalization_observation = {
            "sequence": _next_observation_sequence(
                loop_state,
                observations,
            ),
            "action_sequence": int(loop_state["action_count"]),
            "outcome": "failed",
            "kind": "finalization_failed",
            "summary": finalization_error,
        }
        _append_jsonl(
            loop_root / "observations.jsonl",
            finalization_observation,
        )
        observations.append(finalization_observation)
        return None
    loop_state["answer_text"] = arguments["summary"]
    loop_state["status"] = "awaiting_approval"
    projection = _public_projection(loop_state)
    return projection


def _reconstruct_edit_observation(
    *,
    sequence: object,
    action: Mapping[str, object],
    journal_operation: Mapping[str, object],
    candidate_revision: int,
) -> dict[str, object]:
    """Rebuild review evidence for one already-committed candidate mutation."""

    args = action.get("args")
    if not isinstance(sequence, int) or not isinstance(args, Mapping):
        raise ValueError("orphan edit action identity is invalid")
    operation_kind = args.get("operation")
    repo_path = args.get("repo_path")
    expected_revision = args.get("expected_candidate_revision")
    if (
        not isinstance(operation_kind, str)
        or not isinstance(repo_path, str)
        or not isinstance(expected_revision, int)
        or journal_operation.get("kind") != operation_kind
        or journal_operation.get("repo_path") != repo_path
        or journal_operation.get("expected_candidate_revision")
        != expected_revision
        or journal_operation.get("resulting_candidate_revision")
        != candidate_revision
    ):
        raise ValueError("orphan edit journal identity mismatch")
    patch_operation: dict[str, object] = {
        "kind": operation_kind,
        "path": repo_path,
    }
    if operation_kind == "replace_anchor":
        patch_operation["kind"] = "replace"
    if operation_kind in {
        "create_file",
        "replace_file_small",
        "replace_anchor",
        "insert_before",
        "insert_after",
    }:
        replacement = args.get("replacement")
        if not isinstance(replacement, str):
            raise ValueError("orphan edit replacement identity is invalid")
        patch_operation["content"] = replacement
    if operation_kind in {"replace_anchor", "insert_before", "insert_after"}:
        anchor = args.get("anchor")
        if not isinstance(anchor, str):
            raise ValueError("orphan edit anchor identity is invalid")
        patch_operation["anchor"] = anchor
    if operation_kind == "rename_file":
        target_path = args.get("target_path")
        if (
            not isinstance(target_path, str)
            or journal_operation.get("target_path") != target_path
        ):
            raise ValueError("orphan edit rename identity mismatch")
        patch_operation["target_path"] = target_path
    operation_id = journal_operation.get("operation_id")
    if not isinstance(operation_id, str):
        raise ValueError("orphan edit operation identity is invalid")
    patch_operation["operation_id"] = operation_id
    patch_operation["expected_candidate_revision"] = expected_revision
    expected_sha256 = args.get("expected_sha256")
    if isinstance(expected_sha256, str):
        if journal_operation.get("expected_source_sha256") != expected_sha256:
            raise ValueError("orphan edit source identity mismatch")
        patch_operation["expected_source_sha256"] = expected_sha256
    observation = {
        "sequence": 0,
        "action_sequence": sequence,
        "outcome": "ok",
        "kind": "edit_result",
        "summary": (
            "A committed candidate mutation was reconstructed without replay."
        ),
        "evidence": [{
            "operation_id": operation_id,
            "patch_operation": patch_operation,
            "candidate_revision": candidate_revision,
        }],
    }
    return observation


def _persist_run_execution_result(
    *,
    loop_root: Path,
    sequence: int,
    spec_digest: str,
    observation: Mapping[str, object],
) -> None:
    """Persist one terminal run result before its loop observation."""

    persisted_observation = dict(observation)
    result_digest = _canonical_json_digest(persisted_observation)
    artifact = {
        "schema_version": "coding_run_execution.v1",
        "sequence": sequence,
        "status": "terminal",
        "spec_digest": spec_digest,
        "result_digest": result_digest,
        "observation": persisted_observation,
    }
    _write_json(
        loop_root / "run_executions" / str(sequence) / "result.json",
        artifact,
    )


def _recover_run_execution_result(
    *,
    loop_root: Path,
    sequence: object,
    action: Mapping[str, object],
    loop_state: Mapping[str, object],
) -> tuple[dict[str, object] | None, str]:
    """Validate and recover one terminal run result without re-execution."""

    if not isinstance(sequence, int):
        return None, "orphan_run_evidence_mismatch"
    run_context = loop_state.get("trusted_execution_context")
    if not isinstance(run_context, Mapping):
        return None, "orphan_run_evidence_missing"
    expected_spec_digest = _run_execution_spec_digest(action, run_context)
    artifact_root = loop_root / "run_executions" / str(sequence)
    spec_path = artifact_root / "spec.json"
    result_path = artifact_root / "result.json"
    if not spec_path.is_file() or not result_path.is_file():
        return None, "orphan_run_evidence_missing"
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "orphan_run_evidence_mismatch"
    if not isinstance(spec, dict) or not isinstance(result, dict):
        return None, "orphan_run_evidence_mismatch"
    if spec != {
        "schema_version": "coding_run_execution.v1",
        "sequence": sequence,
        "status": "prepared",
        "spec_digest": expected_spec_digest,
    }:
        return None, "orphan_run_evidence_mismatch"
    if (
        result.get("schema_version") != "coding_run_execution.v1"
        or result.get("sequence") != sequence
        or result.get("status") != "terminal"
        or result.get("spec_digest") != expected_spec_digest
    ):
        return None, "orphan_run_evidence_mismatch"
    observation = result.get("observation")
    if not isinstance(observation, dict):
        return None, "orphan_run_evidence_mismatch"
    if result.get("result_digest") != _canonical_json_digest(observation):
        return None, "orphan_run_evidence_mismatch"
    if observation.get("kind") != "run_result" or observation.get(
        "outcome",
    ) not in {"ok", "failed", "rejected", "unavailable"}:
        return None, "orphan_run_evidence_mismatch"
    recovered = dict(observation)
    recovered["action_sequence"] = sequence
    recovered["candidate_revision"] = loop_state.get("candidate_revision")
    recovered["index_snapshot_id"] = loop_state.get("index_snapshot_id")
    recovered["request"] = {
        "action": action["action"],
        "args": action["args"],
    }
    return recovered, ""


def _run_execution_spec_digest(
    action: Mapping[str, object],
    run_context: Mapping[str, object],
) -> str:
    """Bind one semantic run action to its approved structured execution set."""

    candidate_execution_base = run_context.get("candidate_execution_base")
    execution_specs = run_context.get("execution_specs")
    if (
        not isinstance(candidate_execution_base, Mapping)
        or not isinstance(execution_specs, list)
    ):
        raise ValueError("trusted execution context identity is invalid")
    identity = {
        "action": {
            "action": action["action"],
            "args": _stable_value(action["args"]),
        },
        "candidate_execution_base": _stable_value(candidate_execution_base),
        "execution_specs": _stable_value(execution_specs),
    }
    digest = _canonical_json_digest(identity)
    return digest


def _canonical_json_digest(value: object) -> str:
    """Return a stable SHA-256 identity for JSON-compatible durable evidence."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest


def _refresh_candidate_execution_context(
    *,
    run_root: Path,
    loop_state: dict[str, object],
) -> None:
    """Bind execution authorization to the exact current candidate revision."""

    run_context = loop_state.get("trusted_execution_context")
    if not isinstance(run_context, dict):
        raise ValueError("trusted execution context is missing")
    execution_specs = run_context.get("execution_specs")
    policy_digest = run_context.get("execution_policy_digest")
    candidate_revision = loop_state.get("candidate_revision")
    base_snapshot_id = loop_state.get("index_snapshot_id")
    candidate_source = run_root / "candidate" / "source"
    if (
        not isinstance(execution_specs, list)
        or policy_digest != _canonical_json_digest(execution_specs)
        or not isinstance(candidate_revision, int)
        or not isinstance(base_snapshot_id, str)
        or not candidate_source.is_dir()
    ):
        raise ValueError("candidate execution authorization is invalid")
    candidate_id = hashlib.sha256(
        f"{run_root.name}:candidate".encode("utf-8"),
    ).hexdigest()
    run_context["candidate_execution_base"] = {
        "run_id": run_root.name,
        "candidate_id": candidate_id,
        "candidate_revision": candidate_revision,
        "candidate_manifest_digest": _candidate_tree_digest(candidate_source),
        "base_snapshot_id": base_snapshot_id,
        "execution_policy_digest": policy_digest,
    }


def _candidate_tree_digest(root: Path) -> str:
    """Hash one candidate source tree for isolated execution binding."""

    tree_digest = managed_source_tree_digest(root)
    return tree_digest


def _append_action_record(
    loop_root: Path,
    sequence: int,
    action: dict[str, object],
    *,
    operation_id: str,
    raw_output: str | None = None,
) -> None:
    """Persist one validated action before deterministic dispatch."""

    serialized = raw_output or json.dumps(action, ensure_ascii=False, sort_keys=True)
    record = {
        "sequence": sequence,
        "raw_output_hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "parsed_action": action,
        "operation_id": operation_id,
        "validation": "ok",
    }
    _append_jsonl(loop_root / "actions.jsonl", record)
    if raw_output is not None:
        _append_jsonl(
            loop_root / "raw_outputs.jsonl",
            {"sequence": sequence, "raw_output": raw_output},
        )


def _dispatch_operation_id(
    *,
    run_id: str,
    sequence: int,
    action: Mapping[str, object],
) -> str:
    """Derive one stable effect identity before action dispatch."""

    action_id = action.get("action_id")
    if not isinstance(action_id, str):
        raise ValueError("validated action omitted its action id")
    identity = f"{run_id}\0{sequence}\0{action_id}"
    operation_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return operation_id


def _bind_approval(
    *,
    run_root: Path,
    loop_state: Mapping[str, object],
    approval: Mapping[str, object],
) -> tuple[dict[str, object] | None, str]:
    """Bind user approval evidence to the exact durable review artifact."""

    if approval.get("approved") is not True:
        return None, "Approval requires approved=True."
    for field_name in ("approved_by", "approved_at", "approval_reason"):
        value = approval.get(field_name)
        if not isinstance(value, str) or not value.strip():
            return None, "Approval is incomplete."
    proposal_digest = loop_state.get("proposal_digest")
    candidate_revision = loop_state.get("candidate_revision")
    candidate_tree_digest = loop_state.get("candidate_tree_digest")
    if (
        not isinstance(proposal_digest, str)
        or not isinstance(candidate_revision, int)
        or not isinstance(candidate_tree_digest, str)
    ):
        return None, "The reviewed proposal identity is incomplete."
    actual_tree_digest = _candidate_tree_digest(
        run_root / "candidate" / "source",
    )
    if actual_tree_digest != candidate_tree_digest:
        return None, "The candidate changed after proposal finalization."
    evidence = approval.get("approval_evidence")
    if not isinstance(evidence, Mapping):
        return None, "Approval evidence is required."
    source_message_id = evidence.get("source_message_id")
    if not isinstance(source_message_id, str) or not source_message_id.strip():
        return None, "Approval evidence source identity is required."
    approvals = loop_state.get("approvals", [])
    if isinstance(approvals, list):
        for prior_approval in approvals:
            if not isinstance(prior_approval, Mapping):
                continue
            prior_binding = prior_approval.get("approval_binding")
            if (
                isinstance(prior_binding, Mapping)
                and prior_binding.get("source_message_id") == source_message_id
            ):
                return None, "Approval evidence has already been consumed."
    binding = {
        "schema_version": "coding_action_loop_approval_binding.v1",
        "proposal_digest": proposal_digest,
        "candidate_revision": candidate_revision,
        "candidate_tree_digest": candidate_tree_digest,
        "approval_evidence_digest": _canonical_json_digest(dict(evidence)),
        "source_message_id": source_message_id,
    }
    return binding, ""


def _initialize_pending_effect(
    *,
    run_root: Path,
    loop_state: dict[str, object],
    approval: Mapping[str, object],
    approval_binding: dict[str, object],
    execution_specs: object,
) -> str:
    """Persist-ready apply/verification intent before any external effect."""

    if not isinstance(execution_specs, list) or not all(
        isinstance(specification, dict)
        for specification in execution_specs
    ):
        return "Evaluation execution specs are invalid."
    proposal_digest = loop_state.get("proposal_digest")
    candidate_revision = loop_state.get("candidate_revision")
    candidate_tree_digest = loop_state.get("candidate_tree_digest")
    if (
        not isinstance(proposal_digest, str)
        or not isinstance(candidate_revision, int)
        or not isinstance(candidate_tree_digest, str)
    ):
        return "The reviewed proposal identity is incomplete."
    structured_specs = [dict(specification) for specification in execution_specs]
    effect_id = hashlib.sha256(
        (
            f"{run_root.name}\0{proposal_digest}\0{candidate_revision}\0"
            f"{candidate_tree_digest}\0"
            f"{_canonical_json_digest(approval_binding)}\0"
            f"{_canonical_json_digest(structured_specs)}"
        ).encode("utf-8"),
    ).hexdigest()
    loop_state["pending_effect"] = {
        "schema_version": "coding_action_loop_effect.v1",
        "effect_id": effect_id,
        "state": "applying",
        "proposal_digest": proposal_digest,
        "candidate_revision": candidate_revision,
        "candidate_tree_digest": candidate_tree_digest,
        "approval": dict(approval),
        "approval_binding": approval_binding,
        "execution_specs": structured_specs,
        "execution_spec_digest": _canonical_json_digest(structured_specs),
        "apply_package_id": effect_id[:32],
        "next_execution_index": 0,
        "execution_results": [],
    }
    loop_state["status"] = "applying"
    return ""


def _append_invalid_controller_record(
    *,
    loop_root: Path,
    sequence: int,
    controller_result: Mapping[str, object],
    raw_output: str | None,
) -> None:
    """Persist rejected controller output before returning its schema error."""

    serialized = raw_output or json.dumps(
        dict(controller_result),
        ensure_ascii=False,
        sort_keys=True,
    )
    record = {
        "sequence": sequence,
        "raw_output_hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "validation": controller_result.get("status", "invalid_action"),
        "validation_message": controller_result.get("message", ""),
    }
    _append_jsonl(loop_root / "actions.jsonl", record)
    if raw_output is not None:
        _append_jsonl(
            loop_root / "raw_outputs.jsonl",
            {"sequence": sequence, "raw_output": raw_output},
        )


def _merge_edit_outcome(
    *,
    run_root: Path,
    loop_state: dict[str, object],
    observation: Mapping[str, object],
) -> None:
    """Advance durable candidate state from one successful edit observation."""

    evidence = observation.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        raise ValueError("successful edit observation omitted its operation")
    operation_evidence = evidence[0]
    if not isinstance(operation_evidence, dict):
        raise ValueError("edit operation evidence is invalid")
    patch_operation = operation_evidence.get("patch_operation")
    candidate_revision = operation_evidence.get("candidate_revision")
    if not isinstance(patch_operation, dict) or not isinstance(
        candidate_revision,
        int,
    ):
        raise ValueError("edit operation revision evidence is invalid")
    patch_operations = loop_state["patch_operations"]
    changed_paths = loop_state["changed_paths"]
    if not isinstance(patch_operations, list) or not isinstance(changed_paths, list):
        raise ValueError("action-loop edit state is invalid")
    patch_operations.append(patch_operation)
    source_path = patch_operation.get("path")
    target_path = patch_operation.get("target_path")
    for path in (source_path, target_path):
        if isinstance(path, str) and path not in changed_paths:
            changed_paths.append(path)
    loop_state["candidate_revision"] = candidate_revision
    loop_state["overlay_revision"] = candidate_revision
    for field_name in (
        "patch_artifacts",
        "created_files",
        "changed_files",
        "canonical_operation_records",
        "proposal_digest",
        "candidate_tree_digest",
        "current_approval_binding",
        "pending_effect",
    ):
        loop_state.pop(field_name, None)
    if isinstance(loop_state.get("trusted_execution_context"), dict):
        _refresh_candidate_execution_context(
            run_root=run_root,
            loop_state=loop_state,
        )


def _deterministic_blocker(
    block_args: Mapping[str, object],
    *,
    loop_state: Mapping[str, object],
) -> dict[str, object]:
    """Bind the model's semantic blocker to a deterministic resume target."""

    blocker_type = block_args["blocker_type"]
    if blocker_type == "safety":
        resume_target = "none"
    elif blocker_type == "environment" and (
        isinstance(loop_state.get("current_failure"), Mapping)
        or isinstance(loop_state.get("trusted_execution_context"), Mapping)
    ):
        resume_target = "retry_verification"
    else:
        resume_target = "retry_loop"
    blocker = {
        "blocker_type": blocker_type,
        "question": block_args["question"],
        "options": block_args["options"],
        "resume_target": resume_target,
        "blocking_evidence_refs": block_args["blocking_evidence_refs"],
    }
    return blocker


async def _apply_and_verify(
    *,
    request: Mapping[str, object],
    run_root: Path,
    loop_state: dict[str, object],
    controller: ControllerInvoker,
) -> dict[str, object]:
    """Apply the exact reviewed operations and run trusted structured checks."""

    repository = loop_state["repository"]
    candidate_baseline = "empty_source_free"
    source_root = ""
    if isinstance(repository, dict):
        local_root = repository.get("local_root")
        if isinstance(local_root, str):
            source_root = local_root
            candidate_baseline = "resolved_source"
        source_identity = dict(repository)
    else:
        source_identity = {
            "provider": "source_free",
            "owner": run_root.name,
            "repo": "generated",
            "current_commit": f"source-free:{run_root.name}",
            "dirty_state": "clean",
        }
    patch_artifacts = loop_state.get("patch_artifacts")
    records = loop_state.get("canonical_operation_records")
    proposal_digest = loop_state.get("proposal_digest")
    candidate_revision = loop_state.get("candidate_revision")
    if (
        not isinstance(patch_artifacts, list)
        or not isinstance(records, list)
        or not isinstance(proposal_digest, str)
        or not isinstance(candidate_revision, int)
    ):
        raise ValueError("approved action-loop proposal state is invalid")
    candidate_tree_digest = loop_state.get("candidate_tree_digest")
    approval_binding = loop_state.get("current_approval_binding")
    if not isinstance(candidate_tree_digest, str) or not isinstance(
        approval_binding,
        dict,
    ):
        raise ValueError("approved action-loop binding is invalid")
    execution_specs_value = request.get("execution_specs", [])
    if not isinstance(execution_specs_value, list) or not all(
        isinstance(specification, dict)
        for specification in execution_specs_value
    ):
        return {
            "status": "rejected",
            "run_id": run_root.name,
            "limitations": ["Evaluation execution specs are invalid."],
        }
    execution_specs = [dict(specification) for specification in execution_specs_value]
    effect_id = hashlib.sha256(
        (
            f"{run_root.name}\0{proposal_digest}\0{candidate_revision}\0"
            f"{candidate_tree_digest}\0"
            f"{_canonical_json_digest(approval_binding)}\0"
            f"{_canonical_json_digest(execution_specs)}"
        ).encode("utf-8"),
    ).hexdigest()
    pending_effect = loop_state.get("pending_effect")
    if not isinstance(pending_effect, dict):
        approval_value = request.get("approval")
        if not isinstance(approval_value, Mapping):
            raise ValueError("approved action-loop request omitted approval")
        pending_effect = {
            "schema_version": "coding_action_loop_effect.v1",
            "effect_id": effect_id,
            "state": "applying",
            "proposal_digest": proposal_digest,
            "candidate_revision": candidate_revision,
            "candidate_tree_digest": candidate_tree_digest,
            "approval": dict(approval_value),
            "approval_binding": approval_binding,
            "execution_specs": execution_specs,
            "execution_spec_digest": _canonical_json_digest(execution_specs),
            "apply_package_id": effect_id[:32],
            "next_execution_index": 0,
            "execution_results": [],
        }
        loop_state["pending_effect"] = pending_effect
        loop_state["status"] = "applying"
        _persist_loop_state(run_root, loop_state)
    elif (
        pending_effect.get("schema_version")
        != "coding_action_loop_effect.v1"
        or pending_effect.get("state") not in {"applying", "verifying"}
    ):
        raise ValueError("pending continuation effect state is invalid")
    elif (
        pending_effect.get("effect_id") != effect_id
        or pending_effect.get("proposal_digest") != proposal_digest
        or pending_effect.get("candidate_revision") != candidate_revision
        or pending_effect.get("candidate_tree_digest") != candidate_tree_digest
    ):
        raise ValueError("pending continuation effect identity is stale")
    else:
        saved_specs = pending_effect.get("execution_specs")
        if (
            not isinstance(saved_specs, list)
            or not all(isinstance(specification, dict) for specification in saved_specs)
            or pending_effect.get("execution_spec_digest")
            != _canonical_json_digest(saved_specs)
            or pending_effect.get("approval_binding") != approval_binding
            or pending_effect.get("apply_package_id") != effect_id[:32]
            or not isinstance(pending_effect.get("approval"), Mapping)
        ):
            raise ValueError("pending continuation execution specs are invalid")
        execution_specs = [dict(specification) for specification in saved_specs]
        active_execution_index = pending_effect.get("active_execution_index")
        if isinstance(active_execution_index, int) and not isinstance(
            active_execution_index,
            bool,
        ):
            prior_results = pending_effect.get("execution_results")
            if (
                not isinstance(prior_results, list)
                or len(prior_results) != active_execution_index
            ):
                raise ValueError("pending execution recovery evidence is invalid")
            pending_effect.pop("apply_result", None)
            pending_effect.pop("active_execution_index", None)
            pending_effect["next_execution_index"] = active_execution_index
            pending_effect["state"] = "applying"
            loop_state["status"] = "applying"
            _persist_loop_state(run_root, loop_state)
    apply_request = {
        "workspace_root": str(run_root.parent.parent),
        "source_root": source_root,
        "candidate_baseline": candidate_baseline,
        "source_identity": source_identity,
        "expected_source_identity": source_identity,
        "patch_artifacts": patch_artifacts,
        "approval": pending_effect["approval"],
        "approval_binding": approval_binding,
        "canonical_operation_records": records,
        "proposal_digest": proposal_digest,
        "candidate_revision": candidate_revision,
        "candidate_tree_digest": candidate_tree_digest,
        "apply_package_id": pending_effect["apply_package_id"],
        "max_files": max(1, len(loop_state["changed_paths"])),
        "max_diff_chars": 200_000,
    }
    apply_result_value = pending_effect.get("apply_result")
    if isinstance(apply_result_value, dict) and apply_result_value.get(
        "status"
    ) == "succeeded":
        apply_result = apply_result_value
    else:
        apply_result = apply_approved_patch(apply_request)
        pending_effect["apply_result"] = apply_result
        pending_effect["state"] = (
            "applied" if apply_result["status"] == "succeeded" else "apply_failed"
        )
    apply_attempts = loop_state.get("apply_attempts")
    if not isinstance(apply_attempts, list):
        apply_attempts = []
        loop_state["apply_attempts"] = apply_attempts
    if not apply_attempts or apply_attempts[-1] != apply_result:
        apply_attempts.append(apply_result)
    _persist_loop_state(run_root, loop_state)
    if apply_result["status"] != "succeeded":
        if apply_result["status"] == "rejected":
            loop_state["status"] = "awaiting_approval"
            pending_effect["state"] = "rejected"
            effect_history = loop_state.get("effect_history")
            if not isinstance(effect_history, list):
                effect_history = []
                loop_state["effect_history"] = effect_history
            effect_history.append(pending_effect)
            loop_state.pop("pending_effect", None)
            loop_state.pop("current_approval_binding", None)
            _persist_loop_state(run_root, loop_state)
            return {
                "status": "rejected",
                "run_id": run_root.name,
                "limitations": list(apply_result["limitations"]),
            }
        result = await _resume_after_verification_failure(
            run_root=run_root,
            loop_state=loop_state,
            failure_kind="apply_failed",
            failure_summary="; ".join(apply_result["limitations"]),
            controller=controller,
        )
        return result
    execution_attempts = loop_state.get("execution_attempts")
    if not isinstance(execution_attempts, list):
        execution_attempts = []
        loop_state["execution_attempts"] = execution_attempts
    next_execution_index = pending_effect.get("next_execution_index", 0)
    if not isinstance(next_execution_index, int) or isinstance(
        next_execution_index,
        bool,
    ):
        raise ValueError("pending execution index is invalid")
    loop_state["status"] = "verifying"
    pending_effect["state"] = "verifying"
    _persist_loop_state(run_root, loop_state)
    for execution_index in range(next_execution_index, len(execution_specs)):
        execution_spec = execution_specs[execution_index]
        pending_effect["active_execution_index"] = execution_index
        _persist_active_loop_state(run_root, loop_state)
        execution_result = execute_code_check({
            "workspace_root": str(run_root.parent.parent),
            "apply_package_id": apply_result["apply_package_id"],
            "apply_workspace_ref": apply_result["apply_workspace_ref"],
            "execution": execution_spec,
        })
        execution_attempts.append(execution_result)
        effect_results = pending_effect.get("execution_results")
        if not isinstance(effect_results, list):
            raise ValueError("pending execution results are invalid")
        effect_results.append(execution_result)
        pending_effect["next_execution_index"] = execution_index + 1
        pending_effect.pop("active_execution_index", None)
        _persist_active_loop_state(run_root, loop_state)
        if execution_result["status"] != "succeeded":
            execution_output = "\n".join(
                part
                for part in (
                    execution_result["stdout_excerpt"],
                    execution_result["stderr_excerpt"],
                )
                if part
            )
            failure_summary = (
                f"{execution_result['tool']} verification returned "
                f"{execution_result['status']}: "
                f"{execution_output}"
            )
            result = await _resume_after_verification_failure(
                run_root=run_root,
                loop_state=loop_state,
                failure_kind="execution_verification",
                failure_summary=failure_summary,
                controller=controller,
                trusted_execution_context={
                    "workspace_root": str(run_root.parent.parent),
                    "prior_apply_package_id": apply_result["apply_package_id"],
                    "prior_apply_workspace_ref": apply_result[
                        "apply_workspace_ref"
                    ],
                    "execution_specs": execution_specs,
                    "execution_policy_digest": _canonical_json_digest(
                        execution_specs,
                    ),
                },
            )
            return result
    loop_state["status"] = "completed"
    pending_effect["state"] = "completed"
    loop_state["current_failure"] = None
    _persist_loop_state(run_root, loop_state)
    projection = _public_projection(loop_state)
    return projection


async def _resume_after_verification_failure(
    *,
    run_root: Path,
    loop_state: dict[str, object],
    failure_kind: str,
    failure_summary: str,
    controller: ControllerInvoker,
    trusted_execution_context: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return bounded apply or execution evidence to the same controller loop."""

    observations = loop_state["observations"]
    if not isinstance(observations, list):
        raise ValueError("action-loop failure state is invalid")
    summary = failure_summary[:8000]
    observation = {
        "sequence": _next_observation_sequence(loop_state, observations),
        "outcome": "failed",
        "kind": failure_kind,
        "summary": summary,
    }
    observations.append(observation)
    pending_effect_value = loop_state.get("pending_effect")
    effect_id = ""
    execution_index: int | None = None
    if isinstance(pending_effect_value, Mapping):
        effect_value = pending_effect_value.get("effect_id")
        if isinstance(effect_value, str):
            effect_id = effect_value
        active_index = pending_effect_value.get("active_execution_index")
        if isinstance(active_index, int) and not isinstance(active_index, bool):
            execution_index = active_index
        else:
            next_index = pending_effect_value.get("next_execution_index")
            if (
                isinstance(next_index, int)
                and not isinstance(next_index, bool)
                and next_index > 0
            ):
                execution_index = next_index - 1
    loop_state["current_failure"] = {
        "kind": failure_kind,
        "summary": summary,
        "candidate_revision": int(loop_state["candidate_revision"]),
        "effect_id": effect_id,
        "execution_index": execution_index,
    }
    _append_jsonl(run_root / "action_loop" / "observations.jsonl", observation)
    loop_state["status"] = "active"
    pending_effect = loop_state.pop("pending_effect", None)
    if isinstance(pending_effect, dict):
        pending_effect["state"] = "failed"
        effect_history = loop_state.get("effect_history")
        if not isinstance(effect_history, list):
            effect_history = []
            loop_state["effect_history"] = effect_history
        effect_history.append(pending_effect)
    loop_state.pop("current_approval_binding", None)
    if trusted_execution_context is not None:
        loop_state["trusted_execution_context"] = trusted_execution_context
        _refresh_candidate_execution_context(
            run_root=run_root,
            loop_state=loop_state,
        )
    _persist_active_loop_state(run_root, loop_state)
    allowed_actions = _objective_capabilities(loop_state["objective_type"])
    if allowed_actions is None:
        raise ValueError("persisted action-loop objective is unsupported")
    result = await _run_controller_loop(
        run_root=run_root,
        loop_state=loop_state,
        allowed_actions=allowed_actions,
        controller=controller,
    )
    return result


def _finalize_candidate(
    run_root: Path,
    loop_state: dict[str, object],
) -> str:
    """Compile reviewed artifacts and bind their ordered operation digest."""

    patch_operations = loop_state["patch_operations"]
    if not isinstance(patch_operations, list) or not patch_operations:
        return "Proposal finish requires at least one successful edit."
    repository = loop_state["repository"]
    repo_root: Path | None = None
    if isinstance(repository, dict):
        local_root = repository.get("local_root")
        if isinstance(local_root, str) and local_root:
            repo_root = Path(local_root)
    artifacts, created_files, changed_files, errors = compile_patch_operations(
        repo_root=repo_root,
        patch_operations=patch_operations,
        max_files=max(1, len(loop_state["changed_paths"])),
        max_diff_chars=200_000,
    )
    if errors:
        return "; ".join(errors)
    if not artifacts:
        return "Proposal finalization produced no review artifacts."
    candidate_revision = loop_state["candidate_revision"]
    if not isinstance(candidate_revision, int):
        raise ValueError("candidate revision is invalid during finalization")
    try:
        records = build_canonical_operation_records(
            repo_root=repo_root,
            patch_operations=patch_operations,
            candidate_revision=candidate_revision,
        )
    except (OSError, ValueError) as exc:
        safe_error = str(exc).replace(str(run_root), "<managed_run>")
        return f"Candidate provenance could not be finalized: {safe_error}"
    loop_state["patch_artifacts"] = artifacts
    loop_state["created_files"] = created_files
    loop_state["changed_files"] = changed_files
    loop_state["canonical_operation_records"] = records
    loop_state["proposal_digest"] = canonical_proposal_digest(records)
    candidate_tree_digest = _candidate_tree_digest(
        run_root / "candidate" / "source",
    )
    loop_state["candidate_tree_digest"] = candidate_tree_digest
    review_root = run_root / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        review_root / "proposal.json",
        {
            "schema_version": "coding_action_loop_proposal.v1",
            "candidate_revision": candidate_revision,
            "proposal_digest": loop_state["proposal_digest"],
            "candidate_tree_digest": candidate_tree_digest,
            "canonical_operation_records": records,
            "patch_artifacts": artifacts,
            "changed_files": changed_files,
        },
    )
    loop_state["current_failure"] = None
    return ""


def _write_context_manifest(
    loop_root: Path,
    sequence: int,
    context: str,
    observations: list[object],
) -> None:
    """Record the exact context identity and included observation sequences."""

    observation_sequences = [
        observation["sequence"]
        for observation in observations
        if isinstance(observation, dict)
        and isinstance(observation.get("sequence"), int)
    ]
    manifest = {
        "schema_version": "coding_action_loop_context.v1",
        "sequence": sequence,
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "context_chars": len(context),
        "observation_sequences": observation_sequences,
    }
    _append_jsonl(loop_root / "context_manifest.jsonl", manifest)
    _write_json(loop_root / "context_manifest.json", manifest)


def _persist_loop_state(
    run_root: Path,
    loop_state: dict[str, object],
) -> None:
    """Persist loop and public run state after one terminal segment."""

    _write_json(run_root / "action_loop" / "state.json", loop_state)
    _write_json(
        run_root / "action_loop" / "working_notes.json",
        {"working_note": loop_state["working_note"]},
    )
    _write_run_ledger(run_root, loop_state)


def _persist_active_loop_state(
    run_root: Path,
    loop_state: dict[str, object],
) -> None:
    """Persist one completed turn before requesting another controller action."""

    _write_json(run_root / "action_loop" / "state.json", loop_state)
    _write_json(
        run_root / "action_loop" / "working_notes.json",
        {"working_note": loop_state["working_note"]},
    )


def _write_run_ledger(
    run_root: Path,
    loop_state: Mapping[str, object],
) -> None:
    """Write the private engine's benchmark-compatible durable ledger."""

    ledger = {
        "ledger_schema_version": "coding_run.v4",
        "engine_id": "action_loop_v1",
        "run_id": run_root.name,
        "status": loop_state["status"],
        "objective_type": loop_state["objective_type"],
        "goal": loop_state["goal"],
        "answer_text": loop_state.get("answer_text", ""),
        "source_request": loop_state["source_request"],
        "repository": loop_state["repository"],
        "source_scope": loop_state["source_scope"],
        "index_snapshot_id": loop_state["index_snapshot_id"],
        "candidate_revision": loop_state["candidate_revision"],
        "blocker": loop_state.get("blocker"),
        "patch_artifacts": loop_state.get("patch_artifacts", []),
        "changed_files": loop_state.get("changed_files", []),
        "canonical_operation_records": loop_state.get(
            "canonical_operation_records",
            [],
        ),
        "proposal_digest": loop_state.get("proposal_digest", ""),
        "scenario_precondition_digest": loop_state.get(
            "scenario_precondition_digest"
        ),
        "scenario_review_sha256": loop_state.get("scenario_review_sha256"),
        "scenario_canonical_operation_records": loop_state.get(
            "scenario_canonical_operation_records",
            [],
        ),
        "scenario_proposal_digest": loop_state.get(
            "scenario_proposal_digest"
        ),
        "approvals": loop_state.get("approvals", []),
        "apply_attempts": loop_state.get("apply_attempts", []),
        "execution_attempts": loop_state.get("execution_attempts", []),
        "effect_history": loop_state.get("effect_history", []),
        "current_failure": loop_state.get("current_failure"),
    }
    _write_json(run_root / "run.json", ledger)


def _public_projection(loop_state: Mapping[str, object]) -> dict[str, object]:
    """Project prompt-safe terminal state without local workspace details."""

    projection = {
        "status": loop_state["status"],
        "run_id": loop_state["run_id"],
        "objective_type": loop_state["objective_type"],
        "answer_text": loop_state.get("answer_text", ""),
        "patch_artifacts": loop_state.get("patch_artifacts", []),
        "changed_files": loop_state.get("changed_files", []),
        "canonical_operation_records": loop_state.get(
            "canonical_operation_records",
            [],
        ),
        "proposal_digest": loop_state.get("proposal_digest", ""),
        "limitations": [],
        "trace_summary": [f"action_loop:{loop_state['status']}"],
    }
    blocker = loop_state.get("blocker")
    if isinstance(blocker, dict):
        projection["blocker"] = blocker
    return projection


async def invoke_controller(
    *,
    context: str,
    allowed_actions: set[str],
) -> dict[str, object]:
    """Invoke the one configured controller and structurally validate its action.

    Args:
        context: Bounded dynamic semantic task and observation payload.
        allowed_actions: Capabilities exposed for the current durable loop state.

    Returns:
        Parsed validated action or a typed configuration/contract observation.
    """

    route_values = _controller_route_values()
    if route_values is None:
        return {
            "status": "blocked",
            "blocker_type": "controller_configuration_missing",
            "resume_target": "retry_loop",
        }
    base_url, api_key, model = route_values
    controller = LLInterface()
    config = LLMCallConfig(
        stage_name=__name__,
        route_name="CODING_AGENT_ACTION_LOOP_LLM",
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.1,
        top_p=0.7,
        top_k=None,
        max_completion_tokens=cfg.CODING_AGENT_ACTION_LOOP_LLM_MAX_COMPLETION_TOKENS,
        presence_penalty=None,
        timeout_seconds=120,
        thinking=LLMThinkingConfig(
            enabled=cfg.CODING_AGENT_ACTION_LOOP_LLM_THINKING_ENABLED,
        ),
    )
    response = await controller.ainvoke([
        SystemMessage(content=CONTROLLER_PROMPT),
        HumanMessage(content=context),
    ], config=config)
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, dict):
        return {"status": "invalid_json", "raw_output": response.content}
    parsed_action = parse_action(parsed, allowed_actions=allowed_actions)
    controller_result = dict(parsed_action)
    controller_result["raw_output"] = response.content
    return controller_result


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    """Append one ordered durable loop artifact row."""

    serialized = json.dumps(row, ensure_ascii=False, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(f"{serialized}\n")
        file_handle.flush()
        os.fsync(file_handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read ordered object rows from one durable loop artifact."""

    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError("durable loop JSONL row is invalid")
        rows.append(parsed)
    return rows


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one bounded loop lifecycle artifact deterministically."""

    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode(
        "utf-8",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("wb") as file_handle:
        file_handle.write(serialized)
        file_handle.write(b"\n")
        file_handle.flush()
        os.fsync(file_handle.fileno())
    os.replace(temporary_path, path)


def _controller_route_values() -> tuple[str, str, str] | None:
    """Read the action-loop route only when a controller call is requested."""

    base_url = cfg.CODING_AGENT_ACTION_LOOP_LLM_BASE_URL
    api_key = cfg.CODING_AGENT_ACTION_LOOP_LLM_API_KEY
    model = cfg.CODING_AGENT_ACTION_LOOP_LLM_MODEL
    if not all((base_url, api_key, model)):
        return None
    route_values = (base_url, api_key, model)
    return route_values
