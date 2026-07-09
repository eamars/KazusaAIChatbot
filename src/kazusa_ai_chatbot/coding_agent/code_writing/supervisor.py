"""Supervisor for the new-artifact code-writing flow."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

from kazusa_ai_chatbot.coding_agent.code_writing.acceptance import (
    derive_acceptance_criteria,
)
from kazusa_ai_chatbot.coding_agent.code_writing.diagnostic_trace import (
    WritingDiagnosticTracer,
)
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    ArtifactReservationResult,
    CodeWritingRequest,
    CodeWritingResult,
    ExternalEvidenceSummary,
    GeneratedArtifact,
    PatchValidationSummary,
    ReservedArtifactPath,
    WritingAcceptanceCriterion,
    WritingAlignmentResult,
    WritingArtifactContract,
    WritingChildPMTask,
    WritingContentFormat,
    WritingExternalEvidenceRequest,
    WritingFileKind,
    WritingInformationRequest,
    WritingPatcherInput,
    WritingPatcherReport,
    WritingPMDecision,
    WritingPMInput,
    WritingProgrammerContract,
    WritingProgrammerResult,
    WritingProgrammerTask,
)
from kazusa_ai_chatbot.coding_agent.code_patching.patcher import (
    materialize_patch_artifacts,
)
from kazusa_ai_chatbot.coding_agent.code_patching.patch_validation import (
    _safe_repo_relative_path,
    materialize_patch_artifacts_for_review,
)
from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
    decide_writing_work,
)
from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
    run_writing_programmer_contract,
)
from kazusa_ai_chatbot.coding_agent.code_writing.synthesizer import (
    DEFAULT_MAX_ANSWER_CHARS,
    synthesize_patch_proposal,
)
from kazusa_ai_chatbot.coding_agent.code_writing.workspace import (
    prepare_writing_workspace,
)
from kazusa_ai_chatbot.coding_agent.file_agent import (
    reserve_new_artifact_paths,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import ensure_path_inside

DEFAULT_MAX_ARTIFACT_CHARS = 48000
MAX_PATCH_FILES = 32
MAX_PM_LIFECYCLE_STEPS = 8
MAX_PM_DEPTH = 4
MAX_CHILD_PM_DELEGATION_DEPTH = 1
MAX_CHILD_REPORTS = 16
MAX_REVIEW_VALIDATION_FEEDBACK_PASSES = 1
READBACK_ROOT_NAME = "writing_readback"


@dataclass
class _PMNodeResult:
    """Internal result for one PM lifecycle subtree."""

    status: str
    generated_artifacts: list[GeneratedArtifact]
    direct_child_reports: list[dict[str, object]]
    information_requests: list[WritingInformationRequest]
    limitations: list[str]
    final_decision: WritingPMDecision | None


async def run_writing_supervisor(
    request: CodeWritingRequest,
    *,
    trace: dict[str, object] | None = None,
) -> CodeWritingResult:
    """Produce a new-artifact patch proposal through writing roles."""

    question = request.get("question", "").strip()
    mode = request.get("mode_hint", "create_new_project")
    if mode != "create_new_project":
        result = _existing_source_rejected_result(mode=mode)
        return result

    if not question:
        result = _result(
            status="needs_user_input",
            mode="create_new_project",
            answer_text="Please provide the code-writing request.",
            patch_artifacts=[],
            created_files=[],
            changed_files=[],
            external_evidence_requests=[],
            external_evidence=[],
            validation=_empty_validation("failed"),
            session=None,
            limitations=[],
            trace_summary=["writing_pm:missing question"],
        )
        return result

    workspace_root = request.get("workspace_root")
    if not workspace_root:
        result = _result(
            status="failed",
            mode="create_new_project",
            answer_text="Code writing requires configured proposal storage.",
            patch_artifacts=[],
            created_files=[],
            changed_files=[],
            external_evidence_requests=[],
            external_evidence=[],
            validation=_empty_validation("failed"),
            session=None,
            limitations=["Missing proposal storage root."],
            trace_summary=["writing:missing storage"],
        )
        return result

    external_evidence = _external_evidence_from_request(request)
    supervisor_facts = _supervisor_facts_from_request(request)
    supervisor_state = _supervisor_state_from_request(request)
    prior_generated_artifacts = _prior_generated_artifacts_from_request(request)
    session = prepare_writing_workspace(
        workspace_root=workspace_root,
        session_id=request.get("session_id"),
        base_identity=_base_identity(question=question),
        mode="create_new_project",
    )
    diagnostic = WritingDiagnosticTracer(
        workspace_root=workspace_root,
        session_id=session["session_id"],
    )
    diagnostic.event(
        stage="writing_supervisor",
        event="session_prepared",
        session_id=session["session_id"],
        public_handle=session["public_handle"],
        trace_path=diagnostic.path,
        question_chars=len(question),
        external_evidence_count=len(external_evidence),
    )
    trace_summary = [
        f"writing_session:handle={session['public_handle']}",
        "writing_pm:mode=create_new_project",
    ]
    acceptance_trace: dict[str, object] = {}
    acceptance_call = diagnostic.start(
        stage="acceptance_llm",
        question_chars=len(question),
    )
    acceptance = await derive_acceptance_criteria(
        question=question,
        trace=acceptance_trace,
    )
    diagnostic.end(
        stage="acceptance_llm",
        call_id=acceptance_call,
        status=acceptance["status"],
        criteria_count=len(acceptance["acceptance_criteria"]),
        raw_output_chars=_trace_text_chars(acceptance_trace, "raw_output"),
    )
    _record_internal_trace(trace, "acceptance", acceptance_trace)
    trace_summary.append(
        "writing_acceptance:"
        f"status={acceptance['status']} "
        f"criteria={len(acceptance['acceptance_criteria'])}"
    )
    if acceptance["status"] != "pass":
        result = _result(
            status="failed",
            mode="create_new_project",
            answer_text="Code writing could not preserve acceptance criteria.",
            patch_artifacts=[],
            created_files=[],
            changed_files=[],
            external_evidence_requests=[],
            external_evidence=external_evidence,
            validation=_empty_validation("failed"),
            alignment=None,
            session=session,
            limitations=acceptance["limitations"],
            trace_summary=trace_summary,
        )
        return result

    max_artifact_chars = request.get(
        "max_artifact_chars",
        DEFAULT_MAX_ARTIFACT_CHARS,
    )
    root_input = _root_pm_input(
        question=question,
        acceptance_criteria=acceptance["acceptance_criteria"],
        external_evidence=external_evidence,
        supervisor_facts=supervisor_facts,
        supervisor_state=supervisor_state,
        prior_generated_artifacts=prior_generated_artifacts,
        child_feedback=[],
    )
    pm_result = await _run_pm_node(
        pm_input=root_input,
        initial_generated_artifacts=prior_generated_artifacts,
        depth=0,
        diagnostic=diagnostic,
        trace=trace,
        trace_summary=trace_summary,
        trace_label="root",
    )
    if pm_result.status in ("need_external_evidence", "need_reading"):
        result = _information_request_result(
            status=pm_result.status,
            requests=pm_result.information_requests,
            generated_artifacts=pm_result.generated_artifacts,
            session=session,
            workspace_root=Path(workspace_root),
            external_evidence=external_evidence,
            trace_summary=trace_summary,
        )
        return result
    if pm_result.status != "succeeded":
        result = _pm_blocked_result(
            pm_result=pm_result,
            session=session,
            external_evidence=external_evidence,
            trace_summary=trace_summary,
        )
        return result
    if not pm_result.generated_artifacts:
        trace_summary.append("writing_empty_completion_feedback:attempt=1")
        retry_input = _root_pm_input(
            question=question,
            acceptance_criteria=acceptance["acceptance_criteria"],
            external_evidence=external_evidence,
            supervisor_facts=supervisor_facts,
            supervisor_state=supervisor_state,
            prior_generated_artifacts=[],
            child_feedback=[_missing_generated_artifacts_feedback()],
        )
        pm_result = await _run_pm_node(
            pm_input=retry_input,
            initial_generated_artifacts=[],
            depth=0,
            diagnostic=diagnostic,
            trace=trace,
            trace_summary=trace_summary,
            trace_label="empty_completion_feedback_1",
        )
        if pm_result.status in ("need_external_evidence", "need_reading"):
            result = _information_request_result(
                status=pm_result.status,
                requests=pm_result.information_requests,
                generated_artifacts=pm_result.generated_artifacts,
                session=session,
                workspace_root=Path(workspace_root),
                external_evidence=external_evidence,
                trace_summary=trace_summary,
            )
            return result
        if pm_result.status != "succeeded":
            result = _pm_blocked_result(
                pm_result=pm_result,
                session=session,
                external_evidence=external_evidence,
                trace_summary=trace_summary,
            )
            return result
        if not pm_result.generated_artifacts:
            result = _pm_blocked_result(
                pm_result=_PMNodeResult(
                    status="failed",
                    generated_artifacts=[],
                    direct_child_reports=pm_result.direct_child_reports,
                    information_requests=[],
                    limitations=["PM completed without generated artifacts."],
                    final_decision=pm_result.final_decision,
                ),
                session=session,
                external_evidence=external_evidence,
                trace_summary=trace_summary,
            )
            return result

    reserved_paths = _reserved_paths_from_artifacts(
        pm_result.generated_artifacts,
    )
    patcher_report = _materialize_patcher_report(
        generated_artifacts=pm_result.generated_artifacts,
        reserved_paths=reserved_paths,
        max_artifact_chars=max_artifact_chars,
        diagnostic=diagnostic,
        trace=trace,
        trace_summary=trace_summary,
    )
    validation = _materialize_review_validation(
        workspace_root=Path(workspace_root),
        patcher_report=patcher_report,
        max_artifact_chars=max_artifact_chars,
        diagnostic=diagnostic,
        trace_summary=trace_summary,
    )
    for feedback_attempt in range(MAX_REVIEW_VALIDATION_FEEDBACK_PASSES):
        if not _should_run_review_validation_feedback(validation):
            break

        feedback = _review_validation_feedback(
            validation=validation,
            patcher_report=patcher_report,
        )
        attempt_number = feedback_attempt + 1
        trace_summary.append(
            "writing_validation_feedback:"
            f"attempt={attempt_number} "
            f"errors={len(validation['errors'])}"
        )
        diagnostic.event(
            stage="review_materialization_feedback",
            event="retry_started",
            attempt=attempt_number,
            error_count=len(validation["errors"]),
        )
        retry_input = _root_pm_input(
            question=question,
            acceptance_criteria=acceptance["acceptance_criteria"],
            external_evidence=external_evidence,
            supervisor_facts=supervisor_facts,
            supervisor_state=supervisor_state,
            prior_generated_artifacts=[],
            child_feedback=[feedback],
        )
        retry_result = await _run_pm_node(
            pm_input=retry_input,
            initial_generated_artifacts=[],
            depth=0,
            diagnostic=diagnostic,
            trace=trace,
            trace_summary=trace_summary,
            trace_label=f"validation_feedback_{attempt_number}",
        )
        if retry_result.status in ("need_external_evidence", "need_reading"):
            result = _information_request_result(
                status=retry_result.status,
                requests=retry_result.information_requests,
                generated_artifacts=retry_result.generated_artifacts,
                session=session,
                workspace_root=Path(workspace_root),
                external_evidence=external_evidence,
                trace_summary=trace_summary,
            )
            return result
        if retry_result.status != "succeeded":
            trace_summary.append(
                "writing_validation_feedback:"
                f"attempt={attempt_number} "
                f"status={retry_result.status}"
            )
            break
        if not retry_result.generated_artifacts:
            trace_summary.append(
                "writing_validation_feedback:"
                f"attempt={attempt_number} "
                "status=failed reason=no_generated_artifacts"
            )
            break

        pm_result = retry_result
        reserved_paths = _reserved_paths_from_artifacts(
            pm_result.generated_artifacts,
        )
        patcher_report = _materialize_patcher_report(
            generated_artifacts=pm_result.generated_artifacts,
            reserved_paths=reserved_paths,
            max_artifact_chars=max_artifact_chars,
            diagnostic=diagnostic,
            trace=trace,
            trace_summary=trace_summary,
            trace_label=f"_validation_feedback_{attempt_number}",
        )
        validation = _materialize_review_validation(
            workspace_root=Path(workspace_root),
            patcher_report=patcher_report,
            max_artifact_chars=max_artifact_chars,
            diagnostic=diagnostic,
            trace_summary=trace_summary,
            trace_label=f"validation_feedback_{attempt_number}",
        )

    synthesis_trace: dict[str, object] = {}
    limitations = _dedupe_strings([
        *pm_result.limitations,
        *patcher_report["diagnostics"],
        *validation["errors"],
    ])
    final_decision = pm_result.final_decision or _fallback_pm_decision()
    synthesis_call = diagnostic.start(
        stage="synthesis_llm",
        generated_artifact_count=len(pm_result.generated_artifacts),
        patch_artifact_count=len(patcher_report["patch_artifacts"]),
        limitation_count=len(limitations),
    )
    answer_text, limitations = await synthesize_patch_proposal(
        question=question,
        pm_decision=final_decision,
        generated_artifacts=pm_result.generated_artifacts,
        patch_artifacts=patcher_report["patch_artifacts"],
        validation=validation,
        external_evidence=external_evidence,
        limitations=limitations,
        preferred_language=request.get("preferred_language"),
        max_answer_chars=request.get("max_answer_chars", DEFAULT_MAX_ANSWER_CHARS),
        trace=synthesis_trace,
    )
    diagnostic.end(
        stage="synthesis_llm",
        call_id=synthesis_call,
        answer_chars=len(answer_text),
        limitation_count=len(limitations),
        raw_output_chars=_trace_text_chars(synthesis_trace, "raw_output"),
    )
    _record_internal_trace(trace, "synthesis", synthesis_trace)

    result = _result(
        status=_status_from_validation(validation),
        mode="create_new_project",
        answer_text=answer_text,
        patch_artifacts=patcher_report["patch_artifacts"],
        created_files=patcher_report["created_files"],
        changed_files=patcher_report["changed_files"],
        external_evidence_requests=[],
        external_evidence=external_evidence,
        validation=validation,
        alignment=None,
        session=session,
        limitations=limitations,
        trace_summary=trace_summary,
    )
    return result


async def _run_pm_node(
    *,
    pm_input: WritingPMInput,
    initial_generated_artifacts: list[GeneratedArtifact] | None = None,
    depth: int,
    diagnostic: WritingDiagnosticTracer,
    trace: dict[str, object] | None,
    trace_summary: list[str],
    trace_label: str,
) -> _PMNodeResult:
    if depth > MAX_PM_DEPTH:
        result = _PMNodeResult(
            status="failed",
            generated_artifacts=[],
            direct_child_reports=[],
            information_requests=[],
            limitations=["PM hierarchy exceeded the supported depth."],
            final_decision=None,
        )
        return result

    generated_artifacts = list(initial_generated_artifacts or [])
    direct_child_reports = list(pm_input["direct_child_reports"])
    child_feedback = list(pm_input["child_feedback"])
    final_decision: WritingPMDecision | None = None

    for step_index in range(MAX_PM_LIFECYCLE_STEPS):
        step_input: WritingPMInput = {
            **pm_input,
            "direct_child_reports": direct_child_reports[-MAX_CHILD_REPORTS:],
            "child_feedback": child_feedback,
        }
        pm_trace: dict[str, object] = {}
        pm_call = diagnostic.start(
            stage="pm_decision_llm",
            pm_id=pm_input["pm_id"],
            depth=depth,
            step=step_index + 1,
            trace_label=trace_label,
            input_chars=_json_char_count(step_input),
            direct_child_report_count=len(step_input["direct_child_reports"]),
            child_feedback_count=len(step_input["child_feedback"]),
        )
        decision = await decide_writing_work(step_input, trace=pm_trace)
        diagnostic.end(
            stage="pm_decision_llm",
            call_id=pm_call,
            pm_id=pm_input["pm_id"],
            depth=depth,
            step=step_index + 1,
            status=decision["status"],
            output_chars=_json_char_count(decision),
            raw_output_chars=_trace_text_chars(pm_trace, "raw_output"),
            attempt_count=_trace_list_count(pm_trace, "attempts"),
        )
        final_decision = decision
        trace_key = f"pm_{trace_label}_{step_index + 1}"
        _record_internal_trace(trace, trace_key, pm_trace)
        trace_summary.append(
            "writing_pm:decision "
            f"pm={pm_input['pm_id']} "
            f"status={decision['status']}"
        )

        if decision["status"] == "request_information":
            request = decision["information_request"]
            if request is None:
                return _invalid_pm_result(
                    "PM information request was missing.",
                    decision,
                )
            status = _information_status_from_request(
                request=request,
                generated_artifacts=generated_artifacts,
            )
            result = _PMNodeResult(
                status=status,
                generated_artifacts=generated_artifacts,
                direct_child_reports=direct_child_reports,
                information_requests=[request],
                limitations=[],
                final_decision=decision,
            )
            return result

        if decision["status"] == "create_child_pm":
            child_task = decision["child_pm_task"]
            if child_task is None:
                return _invalid_pm_result("PM child task was missing.", decision)
            if depth >= MAX_CHILD_PM_DELEGATION_DEPTH:
                child_feedback.append(
                    _child_pm_depth_feedback(
                        child_task=child_task,
                        depth=depth,
                    )
                )
                diagnostic.event(
                    stage="pm_delegation_guard",
                    event="child_pm_rejected",
                    pm_id=pm_input["pm_id"],
                    child_pm_id=child_task["child_pm_id"],
                    depth=depth,
                    max_child_pm_depth=MAX_CHILD_PM_DELEGATION_DEPTH,
                )
                trace_summary.append(
                    "writing_pm:child_pm_rejected "
                    f"pm={pm_input['pm_id']} "
                    f"depth={depth}"
                )
                continue
            child_input = _child_pm_input(
                parent_input=pm_input,
                child_task=child_task,
                depth=depth + 1,
            )
            child_call = diagnostic.start(
                stage="child_pm_lifecycle",
                parent_pm_id=pm_input["pm_id"],
                child_pm_id=child_task["child_pm_id"],
                depth=depth + 1,
                trace_label=f"{trace_label}_{child_task['child_pm_id']}",
            )
            child_result = await _run_pm_node(
                pm_input=child_input,
                depth=depth + 1,
                diagnostic=diagnostic,
                trace=trace,
                trace_summary=trace_summary,
                trace_label=f"{trace_label}_{child_task['child_pm_id']}",
            )
            diagnostic.end(
                stage="child_pm_lifecycle",
                call_id=child_call,
                parent_pm_id=pm_input["pm_id"],
                child_pm_id=child_task["child_pm_id"],
                status=child_result.status,
                generated_artifact_count=len(child_result.generated_artifacts),
                limitation_count=len(child_result.limitations),
            )
            if child_result.status in ("need_external_evidence", "need_reading"):
                return child_result
            if child_result.status != "succeeded":
                child_feedback.append({
                    "stage": "child_pm",
                    "child_id": child_task["child_pm_id"],
                    "summary": "; ".join(child_result.limitations),
                })
                continue
            generated_artifacts.extend(child_result.generated_artifacts)
            report = _child_report_from_result(
                child_id=child_task["child_pm_id"],
                child_result=child_result,
            )
            direct_child_reports.append(report)
            continue

        if decision["status"] == "create_programmer_task":
            programmer_task = decision["programmer_task"]
            if programmer_task is None:
                return _invalid_pm_result("PM programmer task was missing.", decision)
            dependency_feedback = _programmer_dependency_feedback(
                programmer_task=programmer_task,
                available_facts=step_input["available_facts"],
                generated_artifacts=generated_artifacts,
            )
            if dependency_feedback is not None:
                child_feedback.append(dependency_feedback)
                diagnostic.event(
                    stage="programmer_contract_guard",
                    event="readback_fact_required",
                    pm_id=pm_input["pm_id"],
                    task_id=programmer_task["task_id"],
                    generated_artifact_count=len(generated_artifacts),
                )
                trace_summary.append(
                    "writing_pm:programmer_rejected "
                    f"pm={pm_input['pm_id']} "
                    f"task={programmer_task['task_id']} "
                    "reason=missing_readback_fact"
                )
                continue
            outcome = await _run_programmer_task(
                programmer_task=programmer_task,
                diagnostic=diagnostic,
                trace=trace,
                trace_summary=trace_summary,
                trace_label=f"{trace_label}_{programmer_task['task_id']}",
            )
            direct_child_reports.append(outcome["report"])
            if outcome["diagnostics"]:
                child_feedback.append({
                    "stage": "programmer",
                    "child_id": programmer_task["task_id"],
                    "summary": "; ".join(outcome["diagnostics"]),
                })
                continue
            generated_artifacts.extend(outcome["generated_artifacts"])
            continue

        if decision["status"] == "repair_child":
            return _invalid_pm_result(
                "PM repair action is outside the current writing scope.",
                decision,
            )

        if decision["status"] == "complete":
            result = _PMNodeResult(
                status="succeeded",
                generated_artifacts=generated_artifacts,
                direct_child_reports=direct_child_reports,
                information_requests=[],
                limitations=[],
                final_decision=decision,
            )
            return result

        blocker = decision["blocker"]
        limitations = [decision["reason"]]
        if blocker is not None:
            limitations = _dedupe_strings([
                blocker["summary"],
                *blocker["missing_facts"],
                blocker["why_information_request_is_not_enough"],
            ])
        result = _PMNodeResult(
            status="blocked",
            generated_artifacts=generated_artifacts,
            direct_child_reports=direct_child_reports,
            information_requests=[],
            limitations=limitations,
            final_decision=decision,
        )
        return result

    result = _PMNodeResult(
        status="failed",
        generated_artifacts=generated_artifacts,
        direct_child_reports=direct_child_reports,
        information_requests=[],
        limitations=["PM lifecycle step limit was reached."],
        final_decision=final_decision,
    )
    return result


async def _run_programmer_task(
    *,
    programmer_task: WritingProgrammerTask,
    diagnostic: WritingDiagnosticTracer,
    trace: dict[str, object] | None,
    trace_summary: list[str],
    trace_label: str,
) -> dict[str, object]:
    contract = _artifact_contract_from_programmer_task(programmer_task)
    reservation_call = diagnostic.start(
        stage="file_agent_reservation",
        task_id=programmer_task["task_id"],
        trace_label=trace_label,
        contract_chars=_json_char_count(contract),
    )
    reservation = reserve_new_artifact_paths([contract])
    diagnostic.end(
        stage="file_agent_reservation",
        call_id=reservation_call,
        task_id=programmer_task["task_id"],
        status=reservation["status"],
        reserved_path_count=len(reservation["reserved_paths"]),
        error_count=len(reservation["errors"]),
    )
    _record_internal_trace(trace, f"file_agent_{trace_label}", reservation)
    trace_summary.append(
        "file_agent:reservation "
        f"status={reservation['status']} "
        f"paths={len(reservation['reserved_paths'])}"
    )
    if reservation["status"] != "accepted":
        report = _programmer_report(
            task=programmer_task,
            generated_artifact=None,
            diagnostics=reservation["errors"],
        )
        return {
            "generated_artifacts": [],
            "diagnostics": reservation["errors"],
            "report": report,
        }

    programmer_contract = _programmer_contract(
        contract,
        reserved_path=reservation["reserved_paths"][0],
    )
    programmer_trace: dict[str, object] = {}
    programmer_call = diagnostic.start(
        stage="programmer_llm",
        task_id=programmer_task["task_id"],
        artifact_id=programmer_contract["artifact_id"],
        file_kind=programmer_contract["file_kind"],
        content_format=programmer_contract["content_format"],
        contract_chars=_json_char_count(programmer_contract),
    )
    programmer_result = await run_writing_programmer_contract(
        artifact_contract=programmer_contract,
        trace=programmer_trace,
    )
    diagnostic.end(
        stage="programmer_llm",
        call_id=programmer_call,
        task_id=programmer_task["task_id"],
        artifact_id=programmer_contract["artifact_id"],
        status=programmer_result["status"],
        artifact_chars=len(programmer_result["code_artifact"]),
        diagnostic_count=len(programmer_result["diagnostics"]),
        raw_output_chars=_trace_text_chars(programmer_trace, "raw_output"),
        attempt_count=_trace_list_count(programmer_trace, "attempts"),
        timed_out_attempt_count=_timed_out_attempt_count(programmer_trace),
    )
    _record_internal_trace(
        trace,
        f"programmer_{programmer_contract['artifact_id']}",
        programmer_trace,
    )
    trace_summary.append(
        "programmer_report "
        f"{programmer_contract['artifact_id']} "
        f"status={programmer_result['status']} "
        f"chars={len(programmer_result['code_artifact'])}"
    )
    diagnostics = _programmer_diagnostics([programmer_result])
    generated_artifact: GeneratedArtifact | None = None
    if not diagnostics:
        generated_artifact = _generated_artifact(
            artifact_contract=contract,
            reserved_path=reservation["reserved_paths"][0],
            programmer_result=programmer_result,
        )
    report = _programmer_report(
        task=programmer_task,
        generated_artifact=generated_artifact,
        diagnostics=diagnostics,
    )
    generated_artifacts = []
    if generated_artifact is not None:
        generated_artifacts.append(generated_artifact)
    return {
        "generated_artifacts": generated_artifacts,
        "diagnostics": diagnostics,
        "report": report,
    }


def _child_pm_depth_feedback(
    *,
    child_task: WritingChildPMTask,
    depth: int,
) -> dict[str, object]:
    feedback = {
        "stage": "child_pm",
        "child_id": child_task["child_pm_id"],
        "summary": (
            "The supervisor rejected another child PM because this PM is "
            "already at the supported delegation depth. Issue one programmer "
            "task, request missing information, complete, or block."
        ),
        "pm_depth": depth,
        "remaining_child_pm_depth": 0,
    }
    return feedback


def _programmer_dependency_feedback(
    *,
    programmer_task: WritingProgrammerTask,
    available_facts: list[dict[str, object]],
    generated_artifacts: list[GeneratedArtifact],
) -> dict[str, object] | None:
    """Require readback evidence before generated-artifact interfaces are consumed."""

    if not generated_artifacts:
        return None

    consumed_interfaces = _object_string_list(
        programmer_task.get("consumed_interfaces"),
    )
    if not consumed_interfaces:
        return None

    resolved_readback_ids = _resolved_generated_readback_fact_ids(
        available_facts,
    )
    consumed_fact_ids = _object_string_list(
        programmer_task.get("consumed_fact_ids"),
    )
    for fact_id in consumed_fact_ids:
        if fact_id in resolved_readback_ids:
            return None

    artifact_refs = sorted(_generated_artifact_refs(generated_artifacts))
    if resolved_readback_ids:
        summary = (
            "This programmer task consumes interfaces from generated artifacts "
            "but did not cite a resolved supervisor readback fact. Use one of "
            "the available readback request_id values in consumed_fact_ids."
        )
    else:
        summary = (
            "This programmer task consumes interfaces from generated artifacts "
            "before a resolved supervisor readback fact is available. Request "
            "generated-artifact readback before assigning the dependent task."
        )
    feedback = {
        "stage": "programmer_contract",
        "child_id": programmer_task["task_id"],
        "summary": summary,
        "generated_artifacts": artifact_refs[:8],
        "available_readback_fact_ids": resolved_readback_ids,
        "consumed_fact_ids": consumed_fact_ids,
    }
    return feedback


def _resolved_generated_readback_fact_ids(
    available_facts: list[dict[str, object]],
) -> list[str]:
    request_ids: list[str] = []
    for fact in available_facts:
        kind = fact.get("kind")
        request_id = fact.get("request_id")
        summary = fact.get("summary")
        if kind != "generated_artifact_readback":
            continue
        if fact.get("resolved") is not True:
            continue
        if not isinstance(request_id, str) or not request_id.strip():
            continue
        if not isinstance(summary, str) or not summary.strip():
            continue
        request_ids.append(request_id)
    return request_ids


def _reports_from_prior_artifacts(
    prior_generated_artifacts: list[GeneratedArtifact],
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for artifact in prior_generated_artifacts:
        report = {
            "child_id": f"prior_{artifact['artifact_id']}",
            "status": "complete",
            "provided_facts": [
                "Generated artifact is available from an earlier supervised step.",
            ],
            "created_artifacts": [
                {
                    "artifact_id": artifact["artifact_id"],
                    "path": artifact["path"],
                    "purpose": artifact["purpose"],
                }
            ],
            "open_risks": [],
        }
        reports.append(report)
    return reports


def _root_pm_input(
    *,
    question: str,
    acceptance_criteria: list[WritingAcceptanceCriterion],
    external_evidence: list[ExternalEvidenceSummary],
    supervisor_facts: list[dict[str, object]],
    supervisor_state: dict[str, object],
    prior_generated_artifacts: list[GeneratedArtifact],
    child_feedback: list[dict[str, object]],
) -> WritingPMInput:
    available_facts: list[dict[str, object]] = [
        {
            "kind": "acceptance_criteria",
            "summary": "Preserved user-visible requirements.",
            "criteria": acceptance_criteria,
        }
    ]
    for evidence in external_evidence:
        available_facts.append({
            "kind": "external_evidence",
            "request_id": evidence["request_id"],
            "resolved": evidence["resolved"],
            "summary": evidence["result"],
        })
    for fact in supervisor_facts:
        available_facts.append({
            "kind": fact["kind"],
            "request_id": fact["request_id"],
            "resolved": fact["resolved"],
            "summary": fact["result"],
            "task": fact["task"],
        })
    if supervisor_state:
        available_facts.append({
            "kind": "supervisor_work_ledger",
            "summary": "Compact supervisor state for this coding run.",
            "state": supervisor_state,
        })
    pm_input: WritingPMInput = {
        "pm_id": "writing_pm_root",
        "domain": "writing",
        "work_item": {
            "goal": question,
            "scope": "new-artifact writing request",
            "constraints": [
                "new artifacts only",
                "do not mutate the caller workspace",
                "do not run commands or tests",
            ],
            "expected_result": "patch proposal package for new artifacts",
        },
        "available_facts": available_facts,
        "direct_child_reports": _reports_from_prior_artifacts(
            prior_generated_artifacts,
        ),
        "child_feedback": child_feedback,
        "context_limits": _pm_context_limits(depth=0),
    }
    return pm_input


def _child_pm_input(
    *,
    parent_input: WritingPMInput,
    child_task: WritingChildPMTask,
    depth: int,
) -> WritingPMInput:
    pm_input: WritingPMInput = {
        "pm_id": child_task["child_pm_id"],
        "domain": child_task["domain"],
        "work_item": {
            "goal": child_task["goal"],
            "scope": child_task["scope"],
            "constraints": child_task["constraints"],
            "expected_result": "; ".join(child_task["expected_report"]),
        },
        "available_facts": parent_input["available_facts"],
        "direct_child_reports": [],
        "child_feedback": [],
        "context_limits": _pm_context_limits(depth=depth),
    }
    return pm_input


def _pm_context_limits(*, depth: int) -> dict[str, object]:
    remaining_depth = MAX_CHILD_PM_DELEGATION_DEPTH - depth
    if remaining_depth < 0:
        remaining_depth = 0
    context_limits = {
        "max_prompt_chars": 50000,
        "pm_depth": depth,
        "remaining_child_pm_depth": remaining_depth,
        "child_pm_delegation_depth": MAX_CHILD_PM_DELEGATION_DEPTH,
    }
    return context_limits


def _artifact_contract_from_programmer_task(
    task: WritingProgrammerTask,
) -> WritingArtifactContract:
    file_kind, content_format = _kind_and_format(task["output_format"])
    contract: WritingArtifactContract = {
        "artifact_id": task["task_id"],
        "file_label": task["task_id"],
        "file_kind": file_kind,
        "content_format": content_format,
        "purpose": task["artifact_purpose"],
        "imports": task["imports"],
        "provided_interfaces": task["provided_interfaces"],
        "consumed_interfaces": task["consumed_interfaces"],
        "required_behavior": task["required_behavior"],
    }
    return contract


def _kind_and_format(output_format: str) -> tuple[WritingFileKind, WritingContentFormat]:
    text = output_format.casefold()
    has_python_format = "python" in text or ".py" in text
    if "test" in text and has_python_format:
        return "test", "python"
    if has_python_format:
        return "source", "python"
    if ".md" in text or "markdown" in text or "readme" in text:
        return "docs", "markdown"
    if "json" in text:
        return "config", "json"
    if "csv" in text:
        return "data", "csv"
    if "text" in text or "toml" in text or "yaml" in text:
        return "config", "text"
    return "source", "python"


def _programmer_contract(
    artifact_contract: WritingArtifactContract,
    *,
    reserved_path: ReservedArtifactPath,
) -> WritingProgrammerContract:
    contract: WritingProgrammerContract = {
        "artifact_id": artifact_contract["artifact_id"],
        "file_label": reserved_path["file_label"],
        "file_kind": reserved_path["file_kind"],
        "content_format": reserved_path["content_format"],
        "purpose": artifact_contract["purpose"],
        "imports": artifact_contract["imports"],
        "provided_interfaces": artifact_contract["provided_interfaces"],
        "consumed_interfaces": artifact_contract["consumed_interfaces"],
        "required_behavior": artifact_contract["required_behavior"],
    }
    return contract


def _generated_artifact(
    *,
    artifact_contract: WritingArtifactContract,
    reserved_path: ReservedArtifactPath,
    programmer_result: WritingProgrammerResult,
) -> GeneratedArtifact:
    artifact: GeneratedArtifact = {
        "artifact_id": programmer_result["artifact_id"],
        "file_label": artifact_contract["file_label"],
        "file_kind": reserved_path["file_kind"],
        "content_format": programmer_result["content_format"],
        "path": reserved_path["path"],
        "content": programmer_result["code_artifact"],
        "purpose": artifact_contract["purpose"],
    }
    return artifact


def _programmer_report(
    *,
    task: WritingProgrammerTask,
    generated_artifact: GeneratedArtifact | None,
    diagnostics: list[str],
) -> dict[str, object]:
    created_artifacts: list[dict[str, object]] = []
    if generated_artifact is not None:
        created_artifacts.append({
            "artifact_id": generated_artifact["artifact_id"],
            "path": generated_artifact["path"],
            "purpose": generated_artifact["purpose"],
        })
    report = {
        "child_id": task["task_id"],
        "status": "blocked" if diagnostics else "complete",
        "provided_facts": [
            *task["provided_interfaces"],
            *task["required_behavior"],
        ],
        "created_artifacts": created_artifacts,
        "open_risks": diagnostics,
    }
    return report


def _child_report_from_result(
    *,
    child_id: str,
    child_result: _PMNodeResult,
) -> dict[str, object]:
    report = {
        "child_id": child_id,
        "status": child_result.status,
        "provided_facts": _reports_to_facts(child_result.direct_child_reports),
        "created_artifacts": [
            {
                "artifact_id": artifact["artifact_id"],
                "path": artifact["path"],
                "purpose": artifact["purpose"],
            }
            for artifact in child_result.generated_artifacts
        ],
        "open_risks": child_result.limitations,
    }
    return report


def _reports_to_facts(reports: list[dict[str, object]]) -> list[str]:
    facts: list[str] = []
    for report in reports:
        provided = report.get("provided_facts")
        if not isinstance(provided, list):
            continue
        for value in provided:
            if not isinstance(value, str) or value in facts:
                continue
            facts.append(value)
    return facts


def _external_requests_from_information(
    request: WritingInformationRequest,
) -> list[WritingExternalEvidenceRequest]:
    task_parts = [*request["needed_facts"]]
    if request["target_artifacts"]:
        task_parts.append(
            "Targets: " + ", ".join(request["target_artifacts"]),
        )
    task = "; ".join(task_parts)
    external_request: WritingExternalEvidenceRequest = {
        "request_id": request["request_id"],
        "task": task,
        "reason": request["reason_for_next_instruction"],
    }
    return [external_request]


def _reading_requests_from_information(
    requests: list[WritingInformationRequest],
) -> list[dict[str, object]]:
    reading_requests: list[dict[str, object]] = []
    for request in requests:
        task_parts = [*request["needed_facts"]]
        if request["target_artifacts"]:
            task_parts.append(
                "Targets: " + ", ".join(request["target_artifacts"]),
            )
        reading_requests.append({
            "request_id": request["request_id"],
            "task": "; ".join(task_parts),
            "reason": request["reason_for_next_instruction"],
            "target_artifacts": request["target_artifacts"],
        })
    return reading_requests


def _information_status_from_request(
    *,
    request: WritingInformationRequest,
    generated_artifacts: list[GeneratedArtifact],
) -> str:
    if not request["target_artifacts"] or not generated_artifacts:
        return "need_external_evidence"

    generated_refs = _generated_artifact_refs(generated_artifacts)
    for target in request["target_artifacts"]:
        if target in generated_refs:
            return "need_reading"
    return "need_external_evidence"


def _generated_artifact_refs(
    generated_artifacts: list[GeneratedArtifact],
) -> set[str]:
    refs: set[str] = set()
    for artifact in generated_artifacts:
        refs.add(artifact["artifact_id"])
        refs.add(artifact["path"])
        refs.add(artifact["file_label"])
    return refs


def _reserved_paths_from_artifacts(
    generated_artifacts: list[GeneratedArtifact],
) -> list[ReservedArtifactPath]:
    reserved_paths: list[ReservedArtifactPath] = []
    for artifact in generated_artifacts:
        reserved_paths.append({
            "artifact_id": artifact["artifact_id"],
            "file_label": artifact["file_label"],
            "path": artifact["path"],
            "file_kind": artifact["file_kind"],
            "content_format": artifact["content_format"],
            "purpose": artifact["purpose"],
        })
    return reserved_paths


def _materialize_patcher_report(
    *,
    generated_artifacts: list[GeneratedArtifact],
    reserved_paths: list[ReservedArtifactPath],
    max_artifact_chars: int,
    diagnostic: WritingDiagnosticTracer,
    trace: dict[str, object] | None,
    trace_summary: list[str],
    trace_label: str = "",
) -> WritingPatcherReport:
    patcher_input: WritingPatcherInput = {
        "artifact_package_id": "new-artifact-package",
        "artifacts": generated_artifacts,
        "reserved_paths": reserved_paths,
        "max_artifact_chars": max_artifact_chars,
    }
    patcher_trace: dict[str, object] = {}
    patcher_call = diagnostic.start(
        stage="patcher_materialization",
        generated_artifact_count=len(generated_artifacts),
        reserved_path_count=len(reserved_paths),
        max_artifact_chars=max_artifact_chars,
    )
    report = materialize_patch_artifacts(
        repo_root=None,
        patcher_input=patcher_input,
        max_files=MAX_PATCH_FILES,
        max_diff_chars=max_artifact_chars,
        trace=patcher_trace,
    )
    diagnostic.end(
        stage="patcher_materialization",
        call_id=patcher_call,
        status=report["status"],
        patch_artifact_count=len(report["patch_artifacts"]),
        diagnostic_count=len(report["diagnostics"]),
    )
    _record_internal_trace(trace, f"patcher{trace_label}", patcher_trace)
    trace_summary.append(
        "writing_patcher:"
        f"status={report['status']} "
        f"artifacts={len(report['patch_artifacts'])} "
        f"diagnostics={len(report['diagnostics'])}"
    )
    return report


def _materialize_review_validation(
    *,
    workspace_root: Path,
    patcher_report: WritingPatcherReport,
    max_artifact_chars: int,
    diagnostic: WritingDiagnosticTracer,
    trace_summary: list[str],
    trace_label: str = "",
) -> PatchValidationSummary:
    review_call = diagnostic.start(
        stage="review_materialization",
        trace_label=trace_label or "initial",
        patch_artifact_count=len(patcher_report["patch_artifacts"]),
        max_artifact_chars=max_artifact_chars,
    )
    validation = materialize_patch_artifacts_for_review(
        repo_root=None,
        workspace_root=workspace_root,
        patch_artifacts=patcher_report["patch_artifacts"],
        max_files=MAX_PATCH_FILES,
        max_diff_chars=max_artifact_chars,
    )
    diagnostic.end(
        stage="review_materialization",
        call_id=review_call,
        trace_label=trace_label or "initial",
        status=validation["status"],
        file_count=len(validation["files"]),
        error_count=len(validation["errors"]),
    )
    validation = _validation_with_patcher_diagnostics(
        validation,
        patcher_report["diagnostics"],
    )
    if trace_label:
        trace_summary.append(
            "writing_review_package:"
            f"{trace_label}:status={validation['status']}"
        )
    else:
        trace_summary.append(
            f"writing_review_package:status={validation['status']}"
        )
    return validation


def _should_run_review_validation_feedback(
    validation: PatchValidationSummary,
) -> bool:
    should_retry = validation["status"] in ("failed", "rejected") and bool(
        validation["errors"]
    )
    return should_retry


def _missing_generated_artifacts_feedback() -> dict[str, object]:
    feedback = {
        "stage": "pm_completion",
        "child_id": "writing_pm_root",
        "summary": (
            "The PM completed the package without generated artifacts. Return "
            "a complete replacement package for the same user goal. Create the "
            "programmer tasks needed for every required source, test, docs, "
            "config, and data artifact, then complete only after child reports "
            "exist."
        ),
        "errors": ["PM completed without generated artifacts."],
        "materialized_files": [],
        "proposed_files": [],
    }
    return feedback


def _review_validation_feedback(
    *,
    validation: PatchValidationSummary,
    patcher_report: WritingPatcherReport,
) -> dict[str, object]:
    proposed_paths = _changed_paths_from_patcher_report(patcher_report)
    feedback = {
        "stage": "review_materialization",
        "child_id": "artifact_package",
        "summary": (
            "Review materialization failed for the generated package. Return a "
            "complete replacement package for the same user goal. Include "
            "every source, test, docs, config, and data artifact still needed; "
            "the failed package will not be carried forward. Correct the "
            "reported path, import, syntax, or materialization issues without "
            "running commands or claiming tests passed."
        ),
        "errors": validation["errors"][:8],
        "materialized_files": validation["files"][:16],
        "proposed_files": proposed_paths[:16],
    }
    return feedback


def _changed_paths_from_patcher_report(
    patcher_report: WritingPatcherReport,
) -> list[str]:
    paths: list[str] = []
    for row in patcher_report["changed_files"]:
        path = row.get("path")
        if not isinstance(path, str) or not path.strip():
            continue
        if path in paths:
            continue
        paths.append(path)
    return paths


def _prepare_readback_source(
    *,
    generated_artifacts: list[GeneratedArtifact],
    session_id: str,
    workspace_root: Path,
) -> dict[str, object]:
    """Write generated artifacts into a bounded read-only source workspace."""

    root = workspace_root.expanduser().resolve(strict=False)
    readback_root = ensure_path_inside(
        root / READBACK_ROOT_NAME / session_id,
        root,
    )
    if readback_root.exists():
        shutil.rmtree(readback_root)
    readback_root.mkdir(parents=True, exist_ok=True)
    for artifact in generated_artifacts:
        safe_path = _safe_repo_relative_path(artifact["path"])
        if safe_path is None:
            continue
        target_path = ensure_path_inside(readback_root / safe_path, readback_root)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(artifact["content"], encoding="utf-8")

    source_identity = _generated_artifact_identity(generated_artifacts)
    repository = {
        "provider": "github",
        "owner": "managed",
        "repo": "generated-artifacts",
        "source_url": f"local://coding-agent-readback/{session_id}",
        "requested_ref": None,
        "resolved_ref": "generated",
        "current_commit": source_identity,
        "default_branch": "generated",
        "local_root": str(readback_root),
        "storage_kind": "managed_download",
        "managed_checkout": True,
        "workspace_root": str(root),
        "cache_key": None,
        "dirty_state": "clean",
    }
    source_scope = {
        "kind": "repository",
        "repo_relative_path": None,
        "source_url": f"local://coding-agent-readback/{session_id}",
        "requested_ref": None,
        "interpretation": "Generated artifact readback workspace.",
    }
    readback_source = {
        "repository": repository,
        "source_scope": source_scope,
    }
    return readback_source


def _generated_artifact_identity(
    generated_artifacts: list[GeneratedArtifact],
) -> str:
    hasher = hashlib.sha256()
    for artifact in generated_artifacts:
        hasher.update(artifact["path"].encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(artifact["content"].encode("utf-8"))
        hasher.update(b"\0")
    identity = "generated-sha256:" + hasher.hexdigest()
    return identity


def _information_request_result(
    *,
    status: str,
    requests: list[WritingInformationRequest],
    generated_artifacts: list[GeneratedArtifact],
    session: dict[str, object],
    workspace_root: Path,
    external_evidence: list[ExternalEvidenceSummary],
    trace_summary: list[str],
) -> CodeWritingResult:
    external_requests: list[WritingExternalEvidenceRequest] = []
    reading_requests: list[dict[str, object]] = []
    reading_source: dict[str, object] | None = None
    if status == "need_reading":
        reading_requests = _reading_requests_from_information(requests)
        reading_source = _prepare_readback_source(
            generated_artifacts=generated_artifacts,
            session_id=session["session_id"],
            workspace_root=workspace_root,
        )
    else:
        for request in requests:
            external_requests.extend(_external_requests_from_information(request))
    result = _result(
        status=status,
        mode="create_new_project",
        answer_text="Supervisor-mediated information is required before writing.",
        patch_artifacts=[],
        created_files=[],
        changed_files=[],
        external_evidence_requests=external_requests,
        external_evidence=external_evidence,
        validation=_empty_validation("failed"),
        session=session,
        limitations=[],
        trace_summary=trace_summary,
        reading_requests=reading_requests,
        reading_source=reading_source,
        pending_artifacts=generated_artifacts,
    )
    return result


def _pm_blocked_result(
    *,
    pm_result: _PMNodeResult,
    session: dict[str, object],
    external_evidence: list[ExternalEvidenceSummary],
    trace_summary: list[str],
) -> CodeWritingResult:
    status = "needs_user_input" if pm_result.status == "blocked" else "failed"
    result = _result(
        status=status,
        mode="create_new_project",
        answer_text="Code writing PM could not complete the requested work.",
        patch_artifacts=[],
        created_files=[],
        changed_files=[],
        external_evidence_requests=[],
        external_evidence=external_evidence,
        validation=_empty_validation("failed"),
        session=session,
        limitations=pm_result.limitations,
        trace_summary=trace_summary,
    )
    return result


def _invalid_pm_result(
    reason: str,
    decision: WritingPMDecision,
) -> _PMNodeResult:
    result = _PMNodeResult(
        status="failed",
        generated_artifacts=[],
        direct_child_reports=[],
        information_requests=[],
        limitations=[reason],
        final_decision=decision,
    )
    return result


def _existing_source_rejected_result(*, mode: str) -> CodeWritingResult:
    result = _result(
        status="rejected",
        mode=mode,
        answer_text=(
            "This writing stage creates new artifacts only. Existing-source "
            "semantic edits require the code modifying capability."
        ),
        patch_artifacts=[],
        created_files=[],
        changed_files=[],
        external_evidence_requests=[],
        external_evidence=[],
        validation=_empty_validation("rejected"),
        session=None,
        limitations=[
            "Existing-source semantic edits are outside the current writing scope.",
        ],
        trace_summary=["writing:existing_source_rejected"],
    )
    return result


def _programmer_diagnostics(
    programmer_results: list[WritingProgrammerResult],
) -> list[str]:
    diagnostics: list[str] = []
    for result in programmer_results:
        if result["status"] == "succeeded":
            continue
        diagnostics.extend(result["diagnostics"])
    deduped_diagnostics = _dedupe_strings(diagnostics)
    return deduped_diagnostics


def _external_evidence_from_request(
    request: CodeWritingRequest,
) -> list[ExternalEvidenceSummary]:
    external_evidence = request.get("external_evidence", [])
    if not isinstance(external_evidence, list):
        return []
    return external_evidence


def _supervisor_facts_from_request(
    request: CodeWritingRequest,
) -> list[dict[str, object]]:
    supervisor_facts = request.get("supervisor_facts", [])
    if not isinstance(supervisor_facts, list):
        return []
    facts: list[dict[str, object]] = []
    for fact in supervisor_facts:
        if not isinstance(fact, dict):
            continue
        request_id = fact.get("request_id")
        task = fact.get("task")
        result = fact.get("result")
        if not isinstance(request_id, str):
            continue
        if not isinstance(task, str):
            continue
        if not isinstance(result, str):
            continue
        facts.append({
            "request_id": request_id,
            "kind": str(fact.get("kind") or "supervisor_fact"),
            "task": task,
            "resolved": fact.get("resolved") is True,
            "result": result,
        })
    return facts


def _supervisor_state_from_request(
    request: CodeWritingRequest,
) -> dict[str, object]:
    supervisor_state = request.get("supervisor_evidence_state", {})
    if not isinstance(supervisor_state, dict):
        return {}
    return supervisor_state


def _prior_generated_artifacts_from_request(
    request: CodeWritingRequest,
) -> list[GeneratedArtifact]:
    prior_artifacts = request.get("prior_generated_artifacts", [])
    if not isinstance(prior_artifacts, list):
        return []

    artifacts: list[GeneratedArtifact] = []
    for value in prior_artifacts:
        artifact = _valid_prior_generated_artifact(value)
        if artifact is None:
            continue
        artifacts.append(artifact)
    return artifacts


def _valid_prior_generated_artifact(
    value: object,
) -> GeneratedArtifact | None:
    if not isinstance(value, dict):
        return None

    artifact_id = value.get("artifact_id")
    file_label = value.get("file_label")
    file_kind = value.get("file_kind")
    content_format = value.get("content_format")
    path = value.get("path")
    content = value.get("content")
    purpose = value.get("purpose")
    required_values = [
        artifact_id,
        file_label,
        file_kind,
        content_format,
        path,
        content,
        purpose,
    ]
    if not all(isinstance(item, str) and item.strip() for item in required_values):
        return None

    artifact: GeneratedArtifact = {
        "artifact_id": artifact_id,
        "file_label": file_label,
        "file_kind": file_kind,
        "content_format": content_format,
        "path": path,
        "content": content,
        "purpose": purpose,
    }
    return artifact


def _base_identity(*, question: str) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    return f"new-artifact:{digest}"


def _status_from_validation(validation: PatchValidationSummary) -> str:
    if validation["status"] == "succeeded":
        return "succeeded"
    if validation["status"] == "rejected":
        return "rejected"
    return "failed"


def _status_from_validation_and_alignment(
    *,
    validation: PatchValidationSummary,
    alignment: WritingAlignmentResult | None,
) -> str:
    validation_status = _status_from_validation(validation)
    if validation_status != "succeeded":
        return validation_status
    if alignment is not None and alignment["status"] != "pass":
        return "failed"
    return "succeeded"


def _validation_with_patcher_diagnostics(
    validation: PatchValidationSummary,
    diagnostics: list[str],
) -> PatchValidationSummary:
    if not diagnostics:
        return validation
    status = validation["status"]
    if status == "succeeded":
        status = "failed"
    errors = _dedupe_strings([*validation["errors"], *diagnostics])
    updated_validation: PatchValidationSummary = {
        **validation,
        "status": status,
        "errors": errors,
    }
    return updated_validation


def _empty_validation(status: str) -> PatchValidationSummary:
    validation_status = "failed"
    if status == "rejected":
        validation_status = "rejected"
    validation: PatchValidationSummary = {
        "status": validation_status,
        "parsed": False,
        "sandbox_applied": False,
        "errors": [],
        "warnings": [],
        "files": [],
    }
    return validation


def _fallback_pm_decision() -> WritingPMDecision:
    decision: WritingPMDecision = {
        "status": "complete",
        "reason": "Generated artifacts are ready for synthesis.",
        "information_request": None,
        "child_pm_task": None,
        "programmer_task": None,
        "repair_instruction": None,
        "completion_report": {
            "pm_id": "writing_pm_root",
            "status": "complete",
            "provided_facts": [],
            "created_artifacts": [],
            "consumed_facts": [],
            "open_risks": [],
            "next_dependency_needs": [],
        },
        "blocker": None,
    }
    return decision


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value in deduped:
            continue
        deduped.append(value)
    return deduped


def _object_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    strings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        strings.append(text)
    return strings


def _json_char_count(value: object) -> int:
    """Return the rendered JSON size used for diagnostic volume tracking."""

    rendered = json.dumps(value, ensure_ascii=False, default=str)
    count = len(rendered)
    return count


def _trace_text_chars(trace: dict[str, object], key: str) -> int:
    """Return the length of one text field in an internal trace dictionary."""

    value = trace.get(key)
    if not isinstance(value, str):
        return 0
    count = len(value)
    return count


def _trace_list_count(trace: dict[str, object], key: str) -> int:
    """Return the number of rows in one list field from a trace dictionary."""

    value = trace.get(key)
    if not isinstance(value, list):
        return 0
    count = len(value)
    return count


def _timed_out_attempt_count(trace: dict[str, object]) -> int:
    """Count recorded LLM attempts marked as timed out."""

    attempts = trace.get("attempts")
    if not isinstance(attempts, list):
        return 0

    count = 0
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        if attempt.get("timed_out") is True:
            count += 1
    return count


def _record_internal_trace(
    trace: dict[str, object] | None,
    key: str,
    value: dict[str, object],
) -> None:
    if trace is None:
        return
    trace[key] = value


def _result(
    *,
    status: str,
    mode: str,
    answer_text: str,
    patch_artifacts: list[dict[str, object]],
    created_files: list[dict[str, str]],
    changed_files: list[dict[str, str]],
    external_evidence_requests: list[WritingExternalEvidenceRequest],
    external_evidence: list[ExternalEvidenceSummary],
    validation: PatchValidationSummary,
    session,
    limitations: list[str],
    trace_summary: list[str],
    alignment: WritingAlignmentResult | None = None,
    reading_requests: list[dict[str, object]] | None = None,
    reading_source: dict[str, object] | None = None,
    pending_artifacts: list[GeneratedArtifact] | None = None,
) -> CodeWritingResult:
    result: CodeWritingResult = {
        "status": status,
        "mode": mode,
        "answer_text": answer_text,
        "patch_artifacts": patch_artifacts,
        "created_files": created_files,
        "changed_files": changed_files,
        "external_evidence_requests": external_evidence_requests,
        "external_evidence": external_evidence,
        "validation": validation,
        "session": session,
        "limitations": limitations,
        "trace_summary": trace_summary,
    }
    if alignment is not None:
        result["alignment"] = alignment
    if reading_requests is not None:
        result["reading_requests"] = reading_requests
    if reading_source is not None:
        result["reading_source"] = reading_source
    if pending_artifacts is not None:
        result["pending_artifacts"] = pending_artifacts
    return result
