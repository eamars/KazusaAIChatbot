"""Public-boundary replay helpers for residual Cognition V2 Appraisal cases."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter, time_ns
from types import SimpleNamespace
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2 import facade
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    reset_validation_capture,
    validation_capture_snapshot,
    write_diagnostic_artifact,
)
from kazusa_ai_chatbot.cognition_core_v2.parallel_executor import (
    ParallelExecutionResult,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_node
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
    commit_cognition_output,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
    build_text_surface_input_from_global_state,
)
from tests.cognition_core_v2_test_helpers import canonical_character_identity

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT_ROOT = (
    _ROOT / "test_artifacts" / "cognition_core_v2_appraisal_boundary"
)


def _clone(value: object) -> object:
    """Return a JSON-safe deep copy for durable comparison artifacts."""

    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )


def _message_question_id(messages: Sequence[object]) -> str | None:
    """Read the target question id from one model payload when present."""

    for message in messages:
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, Mapping):
            continue
        question = parsed.get("question")
        if isinstance(question, Mapping):
            question_id = question.get("question_id")
            if isinstance(question_id, str):
                return question_id
    return None


class _TargetCandidateLLM:
    """Inject one preserved candidate and delegate every repair to the model."""

    def __init__(
        self,
        delegate: Any,
        *,
        question_id: str,
        first_response_text: str,
    ) -> None:
        self.delegate = delegate
        self.question_id = question_id
        self.first_response_text = first_response_text
        self.injected = False
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *args: object,
        config: object,
        **kwargs: object,
    ) -> Any:
        """Inject only the target's initial response, then use the real LLM."""

        started_at = perf_counter()
        target_question_id = _message_question_id(messages)
        response_source = "live_model"
        if target_question_id == self.question_id and not self.injected:
            response = SimpleNamespace(content=self.first_response_text)
            self.injected = True
            response_source = "preserved_or_controlled_candidate"
        else:
            response = await self.delegate.ainvoke(
                messages,
                *args,
                config=config,
                **kwargs,
            )
        self.calls.append({
            "question_id": target_question_id,
            "response_source": response_source,
            "raw_output": str(getattr(response, "content", "")),
            "messages": [
                {
                    "type": type(message).__name__,
                    "content": str(getattr(message, "content", "")),
                }
                for message in messages
            ],
            "duration_ms": round((perf_counter() - started_at) * 1000, 3),
        })
        return response


def _branch_definitions(graph: object) -> list[dict[str, object]]:
    """Project branch definitions without retaining executable objects."""

    definitions = getattr(graph, "definitions", {})
    if not isinstance(definitions, Mapping):
        return []
    rows: list[dict[str, object]] = []
    for branch_id, definition in definitions.items():
        rows.append({
            "branch_id": str(branch_id),
            "dependencies": list(getattr(definition, "dependencies", ())),
            "action_tendencies": list(
                getattr(definition, "action_tendencies", ())
            ),
            "required": bool(getattr(definition, "required", False)),
            "goal_kind": str(getattr(definition, "goal_kind", "")),
            "dependency_options": [
                list(option)
                for option in getattr(definition, "dependency_options", ())
            ],
            "branch_intent_guidance": str(
                getattr(definition, "branch_intent_guidance", "")
            ),
        })
    return rows


def _surface_state(
    input_payload: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the smallest adapter-neutral state accepted by the L3 builder."""

    return {
        "cognitive_episode": deepcopy(input_payload["episode"]),
        "cognition_core_output": deepcopy(dict(output)),
        "pre_surface_action_results": [],
        "character_profile": canonical_character_identity(
            marker="appraisal-boundary"
        ),
        "user_name": "current user",
        "platform_user_id": "boundary-user",
        "platform_bot_id": "boundary-character",
        "chat_history_recent": [],
        "scene_participant_bindings": [],
    }


async def _run_boundary_once(
    *,
    input_payload: Mapping[str, Any],
    question: Mapping[str, Any],
    planned_questions: Sequence[Mapping[str, Any]] | None = None,
    question_id: str,
    mode: str,
    candidate_result: Mapping[str, Any] | None,
    first_response_text: str | None,
    artifact_id: str,
    monkeypatch: Any,
) -> dict[str, Any]:
    """Run one candidate or paired-control execution through public boundaries."""

    action_calls: list[dict[str, Any]] = []
    graph_calls: list[dict[str, Any]] = []
    commit_calls: list[dict[str, Any]] = []
    persisted_capsules: list[dict[str, Any]] = []
    event_calls: list[dict[str, Any]] = []
    captured_result: dict[str, Any] = {}
    reset_validation_capture(f"appraisal_boundary_{artifact_id}_{mode}")

    async def empty_dependency_graph(
        graph: object,
        *_args: Any,
        completed_external_dependencies: set[str] | None = None,
        **_kwargs: Any,
    ) -> ParallelExecutionResult:
        graph_calls.append({
            "definitions": _branch_definitions(graph),
            "completed_external_dependencies": sorted(
                completed_external_dependencies or set()
            ),
        })
        return ParallelExecutionResult()

    async def silence_action_plan(**kwargs: Any) -> dict[str, Any]:
        action_calls.append({
            "primary_bid": _clone(kwargs.get("primary_bid")),
            "supporting_bids": _clone(kwargs.get("supporting_bids", [])),
            "evidence": _clone(kwargs.get("evidence", [])),
            "resolver_context": _clone(kwargs.get("resolver_context")),
            "current_goal_progress": _clone(
                kwargs.get("current_goal_progress")
            ),
            "required_resolver_evidence_dependency": _clone(
                kwargs.get("required_resolver_evidence_dependency")
            ),
        })
        return {
            "intention": {
                "route": "silence",
                "intention": "remain silent",
                "target_roles": [],
                "reason": "no valid admitted bid",
                "goal_continuation_ref": None,
            },
            "action_requests": [],
            "resolver_requests": [],
            "goal_resolution": "blocked",
            "resolver_pending_resolution": None,
            "resolver_goal_progress": None,
        }

    async def replace_user_state(
        owner_key: str,
        expected_previous_state: Mapping[str, Any],
        replacement: Mapping[str, Any],
    ) -> bool:
        commit_calls.append({
            "state_scope": "user",
            "owner_key": owner_key,
            "expected_previous_state": _clone(expected_previous_state),
            "replacement_state": _clone(replacement),
        })
        return True

    async def replace_character_state(
        *,
        expected_updated_at: str,
        replacement: Mapping[str, Any],
    ) -> bool:
        commit_calls.append({
            "state_scope": "character",
            "expected_updated_at": expected_updated_at,
            "replacement_state": _clone(replacement),
        })
        return True

    async def record_event(**kwargs: Any) -> None:
        event_calls.append(_clone(kwargs))

    monkeypatch.setattr(facade, "execute_dependency_graph", empty_dependency_graph)
    monkeypatch.setattr(facade, "plan_actions", silence_action_plan)
    monkeypatch.setattr(
        facade.failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(
        facade.llm_tracing,
        "current_trace_id",
        lambda: f"appraisal-boundary-{artifact_id}-{mode}",
    )
    monkeypatch.setattr(
        facade.failure_capsule,
        "_schedule_persistence",
        lambda document: persisted_capsules.append(deepcopy(document)),
    )
    monkeypatch.setattr(
        cognition_node,
        "compare_and_replace_user_cognition_state",
        replace_user_state,
    )
    monkeypatch.setattr(
        cognition_node,
        "compare_and_replace_character_cognition_state",
        replace_character_state,
    )
    monkeypatch.setattr(cognition_node, "record_cognition_v2_event", record_event)

    planned_question_rows = (
        [deepcopy(dict(row)) for row in planned_questions]
        if planned_questions is not None
        else [deepcopy(dict(question))]
    )
    if mode == "omit":
        monkeypatch.setattr(
            facade,
            "plan_semantic_questions",
            lambda *_args, **_kwargs: [],
        )
    else:
        monkeypatch.setattr(
            facade,
            "plan_semantic_questions",
            lambda *_args, **_kwargs: deepcopy(planned_question_rows),
        )

    base_services = build_cognition_core_services()
    capturing_llm: _TargetCandidateLLM | None = None
    if mode == "candidate":
        if first_response_text is None:
            raise AssertionError("candidate mode requires a first response")
        capturing_llm = _TargetCandidateLLM(
            base_services.llm,
            question_id=question_id,
            first_response_text=first_response_text,
        )
        services = replace(base_services, llm=capturing_llm)
        original_appraise = facade.appraise_semantic_question

        async def capture_appraisal(*args: Any, **kwargs: Any) -> Any:
            result = await original_appraise(*args, **kwargs)
            captured_result["result"] = deepcopy(result)
            return result

        monkeypatch.setattr(facade, "appraise_semantic_question", capture_appraisal)
    elif mode == "accepted_control":
        if candidate_result is None:
            raise AssertionError("accepted control requires a candidate result")
        services = base_services

        async def replay_accepted_result(*_args: Any, **_kwargs: Any) -> Any:
            return deepcopy(dict(candidate_result))

        monkeypatch.setattr(
            facade,
            "appraise_semantic_question",
            replay_accepted_result,
        )
    elif mode == "omit":
        services = base_services
    else:
        raise AssertionError(f"unsupported Appraisal boundary mode: {mode}")

    run_error: CognitionExecutionError | None = None
    output: Mapping[str, Any] | None = None
    try:
        output = await facade.run_cognition(dict(input_payload), services)
    except CognitionExecutionError as exc:
        run_error = exc
    surface_input: Mapping[str, Any] | None = None
    if output is not None:
        expected_character_updated_at = input_payload["mutable_state"].get(
            "updated_at"
        )
        await commit_cognition_output(
            output,
            expected_character_updated_at=(
                expected_character_updated_at
                if isinstance(expected_character_updated_at, str)
                else None
            ),
        )
        surface_input = build_text_surface_input_from_global_state(
            _surface_state(input_payload, output),
            interaction_style_context="bounded Appraisal replay control",
        )
    capture = validation_capture_snapshot()
    if capture is None:
        raise AssertionError("Appraisal boundary capture is missing")
    if capturing_llm is not None and not capturing_llm.injected:
        raise AssertionError(
            f"target question {question_id} did not receive the preserved candidate"
        )
    action_input = action_calls[-1] if action_calls else {}
    snapshot = {
        "replacement_state": (
            _clone(output["state_update"]["replacement_state"])
            if output is not None
            else None
        ),
        "supporting_bids": (
            _clone(output["supporting_bids"])
            if output is not None
            else []
        ),
        "admitted_bid": (
            _clone(output.get("admitted_bid"))
            if output is not None
            else None
        ),
        "run_error": {
            "type": type(run_error).__name__,
            "error_code": run_error.error_code,
            "stage": run_error.stage,
            "safe_checkpoint": run_error.safe_checkpoint,
            "message": str(run_error),
        } if run_error is not None else None,
        "final_branch_definitions": _clone(graph_calls),
        "workspace_and_goal_context": {
            "generated_branch_calls": _clone(graph_calls),
            "primary_bid": _clone(action_input.get("primary_bid")),
            "supporting_bids": _clone(action_input.get("supporting_bids", [])),
            "current_goal_progress": _clone(
                action_input.get("current_goal_progress")
            ),
        },
        "action_plan_inputs": _clone(action_input),
        "commit_payload": _clone(commit_calls),
        "surface_input": _clone(surface_input),
    }
    return {
        "mode": mode,
        "output": output,
        "run_error": (
            {
                "type": type(run_error).__name__,
                "error_code": run_error.error_code,
                "stage": run_error.stage,
                "safe_checkpoint": run_error.safe_checkpoint,
                "message": str(run_error),
            }
            if run_error is not None
            else None
        ),
        "capture": capture,
        "candidate_calls": capturing_llm.calls if capturing_llm else [],
        "candidate_result": captured_result.get("result"),
        "persisted_capsules": persisted_capsules,
        "commit_calls": commit_calls,
        "event_calls": event_calls,
        "surface_input": surface_input,
        "snapshot": snapshot,
    }


def _search_evidence(
    candidate_run: Mapping[str, Any],
    expected_error_fragments: Sequence[str],
) -> None:
    """Require the named initial failure to remain in durable replay evidence."""

    haystack = json.dumps(
        {
            "capture": candidate_run["capture"],
            "persisted_capsules": candidate_run["persisted_capsules"],
        },
        ensure_ascii=False,
        default=str,
    )
    if not any(
        fragment in haystack for fragment in expected_error_fragments
    ):
        observed_errors = [
            str(stage.get("error") or "")
            for stage in candidate_run["capture"].get("stages", [])
            if isinstance(stage, Mapping) and stage.get("error")
        ]
        raise AssertionError(
            "the preserved candidate did not produce the named failure: "
            + ", ".join(expected_error_fragments)
            + "; observed: "
            + " || ".join(observed_errors)
        )


async def replay_appraisal_through_public_boundary(
    *,
    input_payload: Mapping[str, Any],
    question: Mapping[str, Any],
    first_response_text: str,
    case_id: str,
    expected_error_fragments: Sequence[str],
    source_trace_id: str,
    source_path: Path | None = None,
    source_sha256: str | None = None,
    candidate_classification: str = "preserved_failed_candidate",
    controlled_mutation: Mapping[str, Any] | None = None,
    require_repair_call: bool = True,
    monkeypatch: Any,
) -> dict[str, Any]:
    """Exercise candidate exclusion and a disposition-matched control run.

    The candidate run uses the public facade and a real repair call. The
    control run uses the same public boundary with either the accepted repaired
    result or an omitted question, so downstream candidate-sensitive data is
    compared at the state, persistence, workspace, action, and L3 surfaces.
    """

    question_id = question.get("question_id")
    if not isinstance(question_id, str):
        raise TypeError("replay question id is invalid")
    if source_sha256 is None:
        source_sha256 = hashlib.sha256(first_response_text.encode()).hexdigest()
    candidate_run = await _run_boundary_once(
        input_payload=input_payload,
        question=question,
        question_id=question_id,
        mode="candidate",
        candidate_result=None,
        first_response_text=first_response_text,
        artifact_id=case_id,
        monkeypatch=monkeypatch,
    )
    calls = candidate_run["candidate_calls"]
    if require_repair_call and len(calls) < 2:
        raise AssertionError(
            "the controlled candidate did not reach a real repair call"
        )
    _search_evidence(candidate_run, expected_error_fragments)

    if candidate_run["run_error"] is not None:
        if candidate_run["run_error"]["error_code"] == (
            "cognition_boundary_rejected"
        ):
            if candidate_run["commit_calls"] or candidate_run["event_calls"]:
                raise AssertionError(
                    "terminal boundary replay committed a side effect"
                )
            if candidate_run["snapshot"]["action_plan_inputs"]:
                raise AssertionError(
                    "terminal boundary replay reached action planning"
                )
            if candidate_run["surface_input"] is not None:
                raise AssertionError(
                    "terminal boundary replay reached L3 projection"
                )
            artifact = {
                "schema_version": (
                    "cognition_core_v2_appraisal_public_boundary.v1"
                ),
                "case_id": case_id,
                "source": {
                    "trace_id": source_trace_id,
                    "path": (
                        str(source_path)
                        if source_path is not None
                        else None
                    ),
                    "sha256": source_sha256,
                },
                "candidate": {
                    "classification": candidate_classification,
                    "candidate_sha256": hashlib.sha256(
                        first_response_text.encode()
                    ).hexdigest(),
                    "controlled_mutation": (
                        _clone(controlled_mutation)
                        if controlled_mutation is not None
                        else None
                    ),
                    "first_attempt_performance_evidence": False,
                    "expected_first_failure": list(expected_error_fragments),
                    "real_repair_call": len(calls) >= 2,
                    "calls": _clone(candidate_run["candidate_calls"]),
                },
                "disposition": {
                    "boundary_status": "terminal_rejection",
                    "boundary_error": _clone(candidate_run["run_error"]),
                    "control_mode": "none",
                    "accepted_repaired_result": False,
                },
                "candidate_snapshot": _clone(candidate_run["snapshot"]),
                "capture": _clone(candidate_run["capture"]),
                "failure_capsule": _clone(candidate_run["persisted_capsules"]),
                "commit_events": _clone(candidate_run["event_calls"]),
            }
            artifact_path = write_diagnostic_artifact(
                f"{case_id}_{time_ns()}",
                artifact,
                artifact_root=_ARTIFACT_ROOT,
            )
            return {
                "artifact_path": artifact_path,
                "artifact": artifact,
                "candidate_run": candidate_run,
                "control_run": None,
            }
        if candidate_run["run_error"]["error_code"] != (
            "required_selection_without_admitted_bid"
        ):
            raise AssertionError(
                "public replay failed with an unexpected execution error: "
                f"{candidate_run['run_error']}"
            )
        control_run = await _run_boundary_once(
            input_payload=input_payload,
            question=question,
            question_id=question_id,
            mode="omit",
            candidate_result=None,
            first_response_text=None,
            artifact_id=case_id,
            monkeypatch=monkeypatch,
        )
        if candidate_run["snapshot"] != control_run["snapshot"]:
            raise AssertionError(
                "fail-closed candidate changed downstream data relative to "
                "the omitted-question control"
            )
        for run_name, run in (
            ("candidate", candidate_run),
            ("control", control_run),
        ):
            if run["commit_calls"] or run["event_calls"]:
                raise AssertionError(
                    f"{run_name} fail-closed replay committed a side effect"
                )
            if run["snapshot"]["action_plan_inputs"]:
                raise AssertionError(
                    f"{run_name} fail-closed replay reached action planning"
                )
            if run["surface_input"] is not None:
                raise AssertionError(
                    f"{run_name} fail-closed replay reached L3 projection"
                )
        artifact = {
            "schema_version": "cognition_core_v2_appraisal_public_boundary.v1",
            "case_id": case_id,
            "source": {
                "trace_id": source_trace_id,
                "path": str(source_path) if source_path is not None else None,
                "sha256": source_sha256,
            },
            "candidate": {
                "classification": candidate_classification,
                "candidate_sha256": hashlib.sha256(
                    first_response_text.encode()
                ).hexdigest(),
                "controlled_mutation": (
                    _clone(controlled_mutation)
                    if controlled_mutation is not None
                    else None
                ),
                "first_attempt_performance_evidence": False,
                "expected_first_failure": list(expected_error_fragments),
                "real_repair_call": len(calls) >= 2,
                "calls": _clone(candidate_run["candidate_calls"]),
            },
            "disposition": {
                "boundary_status": "fail_closed",
                "boundary_error": _clone(candidate_run["run_error"]),
                "control_mode": "omit",
                "accepted_repaired_result": False,
            },
            "candidate_snapshot": _clone(candidate_run["snapshot"]),
            "control_snapshot": _clone(control_run["snapshot"]),
            "capture": _clone(candidate_run["capture"]),
            "failure_capsule": _clone(candidate_run["persisted_capsules"]),
            "commit_events": _clone(candidate_run["event_calls"]),
        }
        artifact_path = write_diagnostic_artifact(
            f"{case_id}_{time_ns()}",
            artifact,
            artifact_root=_ARTIFACT_ROOT,
        )
        return {
            "artifact_path": artifact_path,
            "artifact": artifact,
            "candidate_run": candidate_run,
            "control_run": control_run,
        }

    appraisal_rows = candidate_run["output"]["cognition_observability"][
        "appraisals"
    ]
    if not isinstance(appraisal_rows, list) or len(appraisal_rows) != 1:
        raise AssertionError("the public replay did not expose one appraisal row")
    appraisal_row = appraisal_rows[0]
    if not isinstance(appraisal_row, Mapping):
        raise TypeError("the appraisal observability row is invalid")
    accepted = (
        appraisal_row.get("status") == "completed"
        and not appraisal_row.get("failure_code")
    )
    control_mode = "accepted_control" if accepted else "omit"
    control_run = await _run_boundary_once(
        input_payload=input_payload,
        question=question,
        question_id=question_id,
        mode=control_mode,
        candidate_result=(
            candidate_run["candidate_result"] if accepted else None
        ),
        first_response_text=None,
        artifact_id=case_id,
        monkeypatch=monkeypatch,
    )
    if candidate_run["snapshot"] != control_run["snapshot"]:
        raise AssertionError(
            "candidate-sensitive downstream data differs from the matched "
            f"{control_mode} control"
        )
    if not accepted:
        if candidate_run["output"].get("admitted_bid") is not None:
            raise AssertionError("rejected appraisal admitted a bid")
        if candidate_run["output"]["supporting_bids"]:
            raise AssertionError("rejected appraisal retained supporting bids")
        action_inputs = candidate_run["snapshot"]["action_plan_inputs"]
        if action_inputs.get("primary_bid") is not None:
            raise AssertionError("rejected appraisal reached action planning")
        if candidate_run["surface_input"].get("primary_bid") is not None:
            raise AssertionError("rejected appraisal reached a primary surface bid")

    artifact = {
        "schema_version": "cognition_core_v2_appraisal_public_boundary.v1",
        "case_id": case_id,
        "source": {
            "trace_id": source_trace_id,
            "path": str(source_path) if source_path is not None else None,
            "sha256": source_sha256,
        },
        "candidate": {
            "classification": candidate_classification,
            "candidate_sha256": hashlib.sha256(
                first_response_text.encode()
            ).hexdigest(),
            "controlled_mutation": (
                _clone(controlled_mutation)
                if controlled_mutation is not None
                else None
            ),
            "first_attempt_performance_evidence": False,
            "expected_first_failure": list(expected_error_fragments),
            "real_repair_call": require_repair_call,
            "calls": _clone(candidate_run["candidate_calls"]),
        },
        "disposition": {
            "appraisal_status": appraisal_row.get("status"),
            "appraisal_failure_code": appraisal_row.get("failure_code"),
            "control_mode": control_mode,
            "accepted_repaired_result": accepted,
        },
        "candidate_snapshot": _clone(candidate_run["snapshot"]),
        "control_snapshot": _clone(control_run["snapshot"]),
        "capture": _clone(candidate_run["capture"]),
        "failure_capsule": _clone(candidate_run["persisted_capsules"]),
        "commit_events": _clone(candidate_run["event_calls"]),
    }
    artifact_path = write_diagnostic_artifact(
        f"{case_id}_{time_ns()}",
        artifact,
        artifact_root=_ARTIFACT_ROOT,
    )
    return {
        "artifact_path": artifact_path,
        "artifact": artifact,
        "candidate_run": candidate_run,
        "control_run": control_run,
    }


__all__ = ["replay_appraisal_through_public_boundary"]
