"""Supervisor for the module-level PM/programmer code-writing flow."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from kazusa_ai_chatbot.coding_agent.code_fetching.models import (
    CodeRepositoryRef,
    CodeSourceScope,
)
from kazusa_ai_chatbot.coding_agent.code_reading.models import (
    CodeEvidenceRow,
    CodeReadingResult,
)
from kazusa_ai_chatbot.coding_agent.code_writing.file_plan_evaluator import (
    evaluate_file_plan,
)
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    ChangedFileSummary,
    CodeWritingRequest,
    CodeWritingResult,
    CreatedFileSummary,
    ExternalEvidenceSummary,
    ModuleProgrammerContract,
    ModuleProgrammerResult,
    PatchArtifact,
    PatchValidationSummary,
    SourceOwnerCandidate,
    SourceOwnershipResolution,
    WritingFileModuleContract,
    WritingMode,
    WritingPatcherInput,
    WritingPatcherReport,
    WritingPMDecision,
    WritingPMInput,
    WritingProgrammerFact,
    WritingProgrammerReport,
    WritingReadingEvidenceRequest,
)
from kazusa_ai_chatbot.coding_agent.code_writing.module_contract_evaluator import (
    evaluate_module_contract,
)
from kazusa_ai_chatbot.coding_agent.code_writing.module_product_manager import (
    decide_module_programmer_contract,
)
from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
    _safe_repo_relative_path,
    validate_patch_artifacts,
)
from kazusa_ai_chatbot.coding_agent.code_writing.patcher import (
    materialize_patch_artifacts,
)
from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
    decide_writing_work,
)
from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
    run_module_programmer_contract,
)
from kazusa_ai_chatbot.coding_agent.code_writing.source_owners import (
    collect_source_owner_candidates,
)
from kazusa_ai_chatbot.coding_agent.code_writing.source_ownership import (
    decide_source_ownership,
)
from kazusa_ai_chatbot.coding_agent.code_writing.synthesizer import (
    DEFAULT_MAX_ANSWER_CHARS,
    synthesize_patch_proposal,
)
from kazusa_ai_chatbot.coding_agent.code_writing.workspace import (
    prepare_writing_workspace,
)
from kazusa_ai_chatbot.coding_agent.file_agent import (
    resolve_writing_file_demands,
)

DEFAULT_MAX_ARTIFACT_CHARS = 48000
MAX_PATCH_FILES = 32
MAX_PM_ASSIGNMENT_REPAIR_ATTEMPTS = 1
MAX_MODULE_CONTRACT_REPAIR_ATTEMPTS = 1
MAX_VALIDATION_REPAIR_ATTEMPTS = 1
MAX_READING_REPORT_ANSWER_CHARS = 2200
MAX_READING_REPORT_EVIDENCE_REFS = 24
MAX_READING_REPORT_EVIDENCE_ROWS = 12
MAX_READING_REPORT_EVIDENCE_ROWS_PER_PATH = 2
MAX_READING_REPORT_EXCERPT_CHARS = 240
MAX_READING_REPORT_LIMITATIONS = 4
MAX_READING_REPORT_LIMITATION_CHARS = 220
MAX_FILE_CONTEXT_CHARS = 12000
MAX_SELECTED_EVIDENCE_ROWS = 12
MAX_SELECTED_EVIDENCE_EXCERPT_CHARS = 600
MAX_LIST_ITEMS = 12


async def run_writing_supervisor(
    request: CodeWritingRequest,
    *,
    trace: dict[str, object] | None = None,
) -> CodeWritingResult:
    """Produce a patch proposal through the current role sequence."""

    question = request.get("question", "").strip()
    mode = _mode(request)
    if not question:
        return _result(
            status="needs_user_input",
            mode=mode,
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

    workspace_root = request.get("workspace_root")
    if not workspace_root:
        return _result(
            status="failed",
            mode=mode,
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

    repository = request.get("repository")
    source_scope = request.get("source_scope")
    reading_result = request.get("reading_result")
    supervisor_evidence_state = request.get("supervisor_evidence_state", {})
    repository_summary = _repository_summary(repository)
    reading_evidence = _reading_evidence(reading_result)
    external_evidence = _external_evidence_from_request(request)
    owner_candidates = collect_source_owner_candidates(
        repo_root=_validation_repo_root(repository),
        reading_evidence=reading_evidence,
    )
    session = prepare_writing_workspace(
        workspace_root=workspace_root,
        session_id=request.get("session_id"),
        base_identity=_base_identity(
            mode=mode,
            question=question,
            repository=repository,
            source_scope=source_scope,
        ),
        mode=mode,
    )
    trace_summary = [
        f"writing_session:handle={session['public_handle']}",
        f"writing_pm:mode={mode}",
    ]

    programmer_reports: list[WritingProgrammerReport] = []
    pm_decision = await _decide_pm(
        question=question,
        mode=mode,
        repository_summary=repository_summary,
        reading_result=reading_result,
        supervisor_evidence_state=supervisor_evidence_state,
        programmer_reports=programmer_reports,
        external_evidence=external_evidence,
        trace=trace,
        trace_key="pm_initial",
    )
    trace_summary.append(
        "writing_pm:work_plan "
        f"status={pm_decision['status']} "
        f"file_demands={len(pm_decision['file_demands'])}"
    )

    if mode == "edit_existing_repository" and reading_result is None:
        if pm_decision["status"] != "need_reading":
            pm_decision = _source_evidence_needed_decision(pm_decision)
        return _pm_terminal_result(
            mode=mode,
            pm_decision=pm_decision,
            session=session,
            external_evidence=external_evidence,
            trace_summary=trace_summary,
        )

    external_result = _external_evidence_result_if_needed(
        mode=mode,
        pm_decision=pm_decision,
        session=session,
        external_evidence=external_evidence,
        trace_summary=trace_summary,
    )
    if external_result is not None:
        return external_result

    if pm_decision["status"] != "need_module_pms":
        return _pm_terminal_result(
            mode=mode,
            pm_decision=pm_decision,
            session=session,
            external_evidence=external_evidence,
            trace_summary=trace_summary,
        )

    pm_decision, file_plan_evaluation = await _accept_or_repair_file_plan(
        question=question,
        mode=mode,
        repository_summary=repository_summary,
        reading_result=reading_result,
        supervisor_evidence_state=supervisor_evidence_state,
        programmer_reports=programmer_reports,
        external_evidence=external_evidence,
        repository=repository,
        source_scope=source_scope,
        owner_candidates=owner_candidates,
        pm_decision=pm_decision,
        trace=trace,
        trace_summary=trace_summary,
        trace_prefix="pm_initial_file_plan",
    )
    if file_plan_evaluation["status"] != "accepted":
        return _file_plan_rejected_result(
            mode=mode,
            evaluation=file_plan_evaluation,
            session=session,
            external_evidence=external_evidence,
            trace_summary=trace_summary,
        )

    external_result = _external_evidence_result_if_needed(
        mode=mode,
        pm_decision=pm_decision,
        session=session,
        external_evidence=external_evidence,
        trace_summary=trace_summary,
    )
    if external_result is not None:
        return external_result
    if pm_decision["status"] != "need_module_pms":
        return _pm_terminal_result(
            mode=mode,
            pm_decision=pm_decision,
            session=session,
            external_evidence=external_evidence,
            trace_summary=trace_summary,
        )

    max_artifact_chars = request.get(
        "max_artifact_chars",
        DEFAULT_MAX_ARTIFACT_CHARS,
    )
    patcher_report: WritingPatcherReport | None = None
    validation = _empty_validation("failed")
    for validation_attempt in range(MAX_VALIDATION_REPAIR_ATTEMPTS + 1):
        try:
            programmer_reports = await _run_file_module_wave(
                mode=mode,
                pm_decision=pm_decision,
                repository=repository,
                reading_evidence=reading_evidence,
                trace=trace,
                trace_summary=trace_summary,
            )
        except ValueError as exc:
            return _result(
                status="rejected",
                mode=mode,
                answer_text=str(exc),
                patch_artifacts=[],
                created_files=[],
                changed_files=[],
                external_evidence_requests=[],
                external_evidence=external_evidence,
                validation=_empty_validation("rejected"),
                session=session,
                limitations=[str(exc)],
                trace_summary=[
                    *trace_summary,
                    "writing:rejected module contract",
                ],
            )

        trace_key_suffix = (
            "initial" if validation_attempt == 0
            else f"validation_repair_{validation_attempt}"
        )
        patcher_report = _materialize_patcher_report(
            question=question,
            mode=mode,
            repository=repository,
            source_scope=source_scope,
            file_contracts=pm_decision["file_contracts"],
            programmer_reports=programmer_reports,
            integration_notes=_pm_integration_notes(pm_decision),
            max_artifact_chars=max_artifact_chars,
            trace=trace,
            trace_key=f"patcher_{trace_key_suffix}",
            trace_summary=trace_summary,
        )
        validation = validate_patch_artifacts(
            repo_root=_validation_repo_root(repository),
            workspace_root=Path(workspace_root),
            patch_artifacts=patcher_report["patch_artifacts"],
            max_files=MAX_PATCH_FILES,
            max_diff_chars=max_artifact_chars,
        )
        validation = _validation_with_operation_errors(
            validation,
            patcher_report["edit_diagnostics"],
        )
        trace_summary.append(
            "writing_validation:"
            f"{trace_key_suffix} status={validation['status']}"
        )
        if validation["status"] == "succeeded":
            break
        if validation_attempt >= MAX_VALIDATION_REPAIR_ATTEMPTS:
            trace_summary.append(
                "writing_validation:repair_exhausted "
                f"status={validation['status']}"
            )
            break

        pm_decision = await _decide_pm(
            question=question,
            mode=mode,
            repository_summary=repository_summary,
            reading_result=reading_result,
            supervisor_evidence_state=supervisor_evidence_state,
            programmer_reports=programmer_reports,
            external_evidence=external_evidence,
            validation_feedback=validation,
            trace=trace,
            trace_key=f"pm_validation_repair_{validation_attempt + 1}",
        )
        trace_summary.append(
            "writing_pm:validation_repair "
            f"status={pm_decision['status']} "
            f"file_demands={len(pm_decision['file_demands'])}"
        )

        external_result = _external_evidence_result_if_needed(
            mode=mode,
            pm_decision=pm_decision,
            session=session,
            external_evidence=external_evidence,
            trace_summary=trace_summary,
        )
        if external_result is not None:
            return external_result
        if pm_decision["status"] != "need_module_pms":
            return _pm_terminal_result(
                mode=mode,
                pm_decision=pm_decision,
                session=session,
                external_evidence=external_evidence,
                trace_summary=trace_summary,
            )

        pm_decision, file_plan_evaluation = await _accept_or_repair_file_plan(
            question=question,
            mode=mode,
            repository_summary=repository_summary,
            reading_result=reading_result,
            supervisor_evidence_state=supervisor_evidence_state,
            programmer_reports=programmer_reports,
            external_evidence=external_evidence,
            repository=repository,
            source_scope=source_scope,
            owner_candidates=owner_candidates,
            pm_decision=pm_decision,
            trace=trace,
            trace_summary=trace_summary,
            trace_prefix=(
                f"pm_validation_repair_{validation_attempt + 1}_file_plan"
            ),
        )
        if file_plan_evaluation["status"] != "accepted":
            return _file_plan_rejected_result(
                mode=mode,
                evaluation=file_plan_evaluation,
                session=session,
                external_evidence=external_evidence,
                trace_summary=trace_summary,
            )

        external_result = _external_evidence_result_if_needed(
            mode=mode,
            pm_decision=pm_decision,
            session=session,
            external_evidence=external_evidence,
            trace_summary=trace_summary,
        )
        if external_result is not None:
            return external_result
        if pm_decision["status"] != "need_module_pms":
            return _pm_terminal_result(
                mode=mode,
                pm_decision=pm_decision,
                session=session,
                external_evidence=external_evidence,
                trace_summary=trace_summary,
            )

    if patcher_report is None:
        return _result(
            status="failed",
            mode=mode,
            answer_text="The writing patcher did not produce a patch report.",
            patch_artifacts=[],
            created_files=[],
            changed_files=[],
            external_evidence_requests=[],
            external_evidence=external_evidence,
            validation=validation,
            session=session,
            limitations=["The writing patcher did not produce a patch report."],
            trace_summary=[*trace_summary, "writing:missing patcher report"],
        )

    synthesis_trace: dict[str, object] = {}
    limitations = _report_limitations(programmer_reports)
    limitations.extend(patcher_report["edit_diagnostics"])
    limitations.extend(validation["errors"])
    answer_text, limitations = await synthesize_patch_proposal(
        question=question,
        mode=mode,
        pm_decision=pm_decision,
        programmer_reports=programmer_reports,
        patch_artifacts=patcher_report["patch_artifacts"],
        validation=validation,
        external_evidence=external_evidence,
        limitations=limitations,
        repository_summary=repository_summary,
        preferred_language=request.get("preferred_language"),
        max_answer_chars=request.get("max_answer_chars", DEFAULT_MAX_ANSWER_CHARS),
        trace=synthesis_trace,
    )
    _record_internal_trace(trace, "synthesis", synthesis_trace)

    return _result(
        status=_status_from_validation(validation),
        mode=mode,
        answer_text=answer_text,
        patch_artifacts=patcher_report["patch_artifacts"],
        created_files=patcher_report["created_files"],
        changed_files=patcher_report["changed_files"],
        external_evidence_requests=[],
        external_evidence=external_evidence,
        validation=validation,
        session=session,
        limitations=limitations,
        trace_summary=[
            *trace_summary,
            f"writing_validation:status={validation['status']}",
        ],
    )


async def _run_file_module_wave(
    *,
    mode: WritingMode,
    pm_decision: WritingPMDecision,
    repository: CodeRepositoryRef | None,
    reading_evidence: list[CodeEvidenceRow],
    trace: dict[str, object] | None,
    trace_summary: list[str],
) -> list[WritingProgrammerReport]:
    """Run one Module PM and one module programmer per file contract."""

    programmer_reports: list[WritingProgrammerReport] = []
    for file_contract in pm_decision["file_contracts"]:
        file_contract_id = file_contract["file_contract_id"]
        selected_evidence = _selected_evidence_for_contract(
            reading_evidence=reading_evidence,
            file_contract=file_contract,
        )
        module_pm_input = _module_pm_input_for_contract(
            mode=mode,
            pm_decision=pm_decision,
            file_contract=file_contract,
            repository=repository,
            selected_evidence=selected_evidence,
        )
        module_contract = await _accepted_module_contract(
            file_contract=file_contract,
            module_pm_input=module_pm_input,
            trace=trace,
            trace_key=f"module_pm_{file_contract_id}",
            trace_summary=trace_summary,
        )
        programmer_trace: dict[str, object] = {}
        programmer_result = await run_module_programmer_contract(
            module_contract=module_contract,
            trace=programmer_trace,
        )
        _record_internal_trace(
            trace,
            f"programmer_{file_contract_id}",
            programmer_trace,
        )
        report = _programmer_report_from_module_result(
            file_contract=file_contract,
            module_contract=module_contract,
            programmer_result=programmer_result,
            selected_evidence=selected_evidence,
        )
        programmer_reports.append(report)
        trace_summary.append(
            "programmer_report "
            f"{file_contract_id} status={report['status']} "
            f"code_chars={len(report['code_artifact'])}"
        )
    return programmer_reports


async def _accepted_module_contract(
    *,
    file_contract: WritingFileModuleContract,
    module_pm_input: dict[str, object],
    trace: dict[str, object] | None,
    trace_key: str,
    trace_summary: list[str],
) -> ModuleProgrammerContract:
    """Run Module PM and one structural repair loop before programmer dispatch."""

    current_input = dict(module_pm_input)
    for attempt_index in range(MAX_MODULE_CONTRACT_REPAIR_ATTEMPTS + 1):
        module_pm_trace: dict[str, object] = {}
        module_contract = await decide_module_programmer_contract(
            current_input,
            trace=module_pm_trace,
        )
        evaluation = evaluate_module_contract(
            file_contract=file_contract,
            module_contract=module_contract,
            module_pm_input=current_input,
        )
        _record_internal_trace(
            trace,
            f"{trace_key}_{attempt_index + 1}",
            {
                "module_pm_trace": module_pm_trace,
                "evaluation": evaluation,
            },
        )
        trace_summary.append(
            "module_pm:"
            f"{file_contract['file_contract_id']} "
            f"evaluation={evaluation['status']}"
        )
        if evaluation["status"] == "accepted":
            return module_contract
        if attempt_index >= MAX_MODULE_CONTRACT_REPAIR_ATTEMPTS:
            raise ValueError(_module_contract_error_text(evaluation))
        current_input["module_contract_feedback"] = evaluation

    raise ValueError("Module contract evaluation did not complete.")


def _module_pm_input_for_contract(
    *,
    mode: WritingMode,
    pm_decision: WritingPMDecision,
    file_contract: WritingFileModuleContract,
    repository: CodeRepositoryRef | None,
    selected_evidence: list[CodeEvidenceRow],
) -> dict[str, object]:
    file_label = _file_label(file_contract)
    context_rows = _file_context_for_contract(
        repository=repository,
        selected_evidence=selected_evidence,
        file_contract=file_contract,
    )
    interface_contract = file_contract.get("interface_contract", {})
    integration_contract = file_contract.get("integration_contract", {})
    source_file_chars = _total_source_file_chars(context_rows)
    input_payload = {
        "file_label": file_label,
        "edit_mode": _edit_mode_for_contract(mode, file_contract),
        "content_format": _content_format_for_contract(file_contract),
        "module_purpose": file_contract.get("purpose", ""),
        "lifecycle_owner": _lifecycle_owner(interface_contract),
        "provided_interfaces": _provided_interfaces(
            interface_contract,
            integration_contract,
            file_contract=file_contract,
        ),
        "consumed_interfaces": _consumed_interfaces(integration_contract),
        "existing_source_anchors": _existing_source_anchors(
            interface_contract,
            file_contract=file_contract,
            selected_evidence=selected_evidence,
        ),
        "integration_behaviors": _integration_behaviors(
            integration_contract,
            file_contract=file_contract,
        ),
        "imports": _imports_for_contract(
            pm_decision=pm_decision,
            file_contract=file_contract,
            file_label=file_label,
        ),
        "current_file_context": _file_context_summary(context_rows),
        "source_file_chars": source_file_chars,
        "selected_evidence": _compact_selected_evidence(selected_evidence),
        "required_behavior": _required_behavior(file_contract),
        "cross_slice_interfaces": _cross_slice_interfaces(
            pm_decision=pm_decision,
            current_file_contract_id=file_contract.get("file_contract_id", ""),
        ),
    }
    return input_payload


def _programmer_report_from_module_result(
    *,
    file_contract: WritingFileModuleContract,
    module_contract: ModuleProgrammerContract,
    programmer_result: ModuleProgrammerResult,
    selected_evidence: list[CodeEvidenceRow],
) -> WritingProgrammerReport:
    code_artifact = programmer_result.get("code_artifact", "")
    status = "succeeded" if code_artifact.strip() else "no_patch"
    open_questions: list[str] = []
    if status != "succeeded":
        open_questions.append("Programmer returned no code artifact.")
    evidence_refs = [_evidence_ref(row) for row in selected_evidence[:MAX_LIST_ITEMS]]
    fact: WritingProgrammerFact = {
        "kind": "module_code_artifact",
        "summary": (
            f"Programmer returned {len(code_artifact)} characters for "
            f"{module_contract['file_label']}."
        ),
        "evidence_refs": evidence_refs,
    }
    report: WritingProgrammerReport = {
        "assignment_id": file_contract["file_contract_id"],
        "file_contract_id": file_contract["file_contract_id"],
        "file_label": module_contract["file_label"],
        "edit_mode": module_contract["edit_mode"],
        "status": status,
        "files_considered": file_contract.get("owned_paths", []),
        "facts": [fact],
        "code_artifact": code_artifact,
        "open_questions": open_questions,
        "created_files": [],
        "changed_files": [],
        "evidence": selected_evidence,
    }
    return report


async def _decide_pm(
    *,
    question: str,
    mode: WritingMode,
    repository_summary: dict[str, object] | None,
    reading_result: CodeReadingResult | None,
    supervisor_evidence_state: dict[str, object],
    programmer_reports: list[WritingProgrammerReport],
    external_evidence: list[ExternalEvidenceSummary],
    validation_feedback: PatchValidationSummary | None = None,
    file_resolution_feedback: dict[str, object] | None = None,
    file_plan_feedback: dict[str, object] | None = None,
    trace: dict[str, object] | None = None,
    trace_key: str = "pm",
) -> WritingPMDecision:
    pm_trace: dict[str, object] = {}
    pm_input: WritingPMInput = {
        "question": question,
        "mode": mode,
        "repository_summary": repository_summary,
        "reading_reports": _reading_reports(reading_result),
        "supervisor_evidence_state": supervisor_evidence_state,
        "previous_writing_reports": programmer_reports,
    }
    if validation_feedback is not None:
        pm_input["validation_feedback"] = validation_feedback
    if file_resolution_feedback is not None:
        pm_input["file_resolution_feedback"] = file_resolution_feedback
    if file_plan_feedback is not None:
        pm_input["file_plan_feedback"] = file_plan_feedback
    if external_evidence:
        pm_input["external_evidence"] = external_evidence
    pm_decision = await decide_writing_work(pm_input, trace=pm_trace)
    _record_internal_trace(trace, trace_key, pm_trace)
    return pm_decision


async def _accept_or_repair_file_plan(
    *,
    question: str,
    mode: WritingMode,
    repository_summary: dict[str, object] | None,
    reading_result: CodeReadingResult | None,
    supervisor_evidence_state: dict[str, object],
    programmer_reports: list[WritingProgrammerReport],
    external_evidence: list[ExternalEvidenceSummary],
    repository: CodeRepositoryRef | None,
    source_scope: CodeSourceScope | None,
    owner_candidates: list[SourceOwnerCandidate],
    pm_decision: WritingPMDecision,
    trace: dict[str, object] | None,
    trace_summary: list[str],
    trace_prefix: str,
) -> tuple[WritingPMDecision, dict[str, object]]:
    """Resolve PM file demands and repair the handoff before Module PM dispatch."""

    current_decision = pm_decision
    evaluation: dict[str, object] = _accepted_file_plan_evaluation()
    for attempt_index in range(MAX_PM_ASSIGNMENT_REPAIR_ATTEMPTS + 1):
        ownership_trace: dict[str, object] = {}
        ownership_resolution = await decide_source_ownership(
            question=question,
            mode=mode,
            file_demands=current_decision["file_demands"],
            owner_candidates=owner_candidates,
            reading_evidence=_reading_evidence(reading_result),
            trace=ownership_trace,
        )
        _record_internal_trace(
            trace,
            f"{trace_prefix}_source_ownership_{attempt_index + 1}",
            ownership_trace,
        )
        trace_summary.append(
            "writing_source_ownership:"
            f"status={ownership_resolution['status']}"
        )
        if ownership_resolution["status"] == "need_reading":
            source_needed = _source_ownership_needed_decision(
                pm_decision=current_decision,
                ownership_resolution=ownership_resolution,
            )
            return source_needed, _accepted_file_plan_evaluation()

        if ownership_resolution["status"] == "repair_required":
            evaluation = _evaluation_from_source_ownership(
                ownership_resolution,
            )
            repair_feedback = ownership_resolution
            if attempt_index >= MAX_PM_ASSIGNMENT_REPAIR_ATTEMPTS:
                return current_decision, evaluation

            current_decision = await _decide_pm(
                question=question,
                mode=mode,
                repository_summary=repository_summary,
                reading_result=reading_result,
                supervisor_evidence_state=supervisor_evidence_state,
                programmer_reports=programmer_reports,
                external_evidence=external_evidence,
                file_resolution_feedback=repair_feedback,
                trace=trace,
                trace_key=f"{trace_prefix}_ownership_repair_{attempt_index + 1}",
            )
            trace_summary.append(
                "writing_source_ownership:repair "
                f"status={current_decision['status']} "
                f"file_demands={len(current_decision['file_demands'])}"
            )
            if current_decision["status"] != "need_module_pms":
                return current_decision, _accepted_file_plan_evaluation()
            continue

        current_decision = _pm_decision_with_source_ownership(
            current_decision,
            ownership_resolution,
        )
        resolution = resolve_writing_file_demands(
            mode=mode,
            repository=repository,
            source_scope=source_scope,
            owner_candidates=owner_candidates,
            file_demands=current_decision["file_demands"],
        )
        _record_internal_trace(
            trace,
            f"{trace_prefix}_file_resolution_{attempt_index + 1}",
            resolution,
        )
        trace_summary.append(
            f"writing_file_agent:resolution status={resolution['status']}"
        )

        if resolution["status"] == "accepted":
            resolved_decision = _pm_decision_with_file_contracts(
                current_decision,
                resolution["file_contracts"],
            )
            evaluation = evaluate_file_plan(
                file_contracts=resolved_decision["file_contracts"],
                source_scope=source_scope,
                mode=mode,
            )
            _record_internal_trace(
                trace,
                f"{trace_prefix}_after_file_resolution_{attempt_index + 1}",
                evaluation,
            )
            trace_summary.append(
                f"writing_file_plan:check status={evaluation['status']}"
            )
            if evaluation["status"] == "accepted":
                return resolved_decision, evaluation
            repair_feedback_key = "file_plan_feedback"
            repair_feedback = evaluation
        else:
            evaluation = _evaluation_from_file_resolution(resolution)
            repair_feedback_key = "file_resolution_feedback"
            repair_feedback = resolution

        if attempt_index >= MAX_PM_ASSIGNMENT_REPAIR_ATTEMPTS:
            return current_decision, evaluation

        current_decision = await _decide_pm(
            question=question,
            mode=mode,
            repository_summary=repository_summary,
            reading_result=reading_result,
            supervisor_evidence_state=supervisor_evidence_state,
            programmer_reports=programmer_reports,
            external_evidence=external_evidence,
            file_resolution_feedback=(
                repair_feedback if repair_feedback_key == "file_resolution_feedback"
                else None
            ),
            file_plan_feedback=(
                repair_feedback if repair_feedback_key == "file_plan_feedback"
                else None
            ),
            trace=trace,
            trace_key=f"{trace_prefix}_repair_{attempt_index + 1}",
        )
        trace_summary.append(
            "writing_file_plan:repair "
            f"status={current_decision['status']} "
            f"file_demands={len(current_decision['file_demands'])}"
        )
        if current_decision["status"] != "need_module_pms":
            return current_decision, _accepted_file_plan_evaluation()

    return current_decision, evaluation


def _accepted_file_plan_evaluation() -> dict[str, object]:
    return {
        "status": "accepted",
        "errors": [],
        "repair_feedback": [],
    }


def _evaluation_from_file_resolution(
    resolution: dict[str, object],
) -> dict[str, object]:
    return {
        "status": "repair_required",
        "errors": list(resolution["errors"]),
        "repair_feedback": list(resolution["repair_feedback"]),
    }


def _evaluation_from_source_ownership(
    resolution: SourceOwnershipResolution,
) -> dict[str, object]:
    evaluation = {
        "status": "repair_required",
        "errors": list(resolution["errors"]),
        "repair_feedback": list(resolution["repair_feedback"]),
    }
    return evaluation


def _source_ownership_needed_decision(
    *,
    pm_decision: WritingPMDecision,
    ownership_resolution: SourceOwnershipResolution,
) -> WritingPMDecision:
    missing_slots: list[str] = []
    for request in ownership_resolution["reading_requests"]:
        for slot in request["required_slots"]:
            if slot not in missing_slots:
                missing_slots.append(slot)
    if not missing_slots:
        missing_slots = ["Current source ownership evidence is missing."]

    decision: WritingPMDecision = {
        "status": "need_reading",
        "mode": pm_decision["mode"],
        "intent": pm_decision["intent"],
        "file_demands": [],
        "file_contracts": [],
        "cross_module_imports": {},
        "missing_slots": missing_slots,
        "reading_requests": ownership_resolution["reading_requests"],
        "external_evidence_requests": [],
    }
    return decision


def _pm_decision_with_source_ownership(
    pm_decision: WritingPMDecision,
    ownership_resolution: SourceOwnershipResolution,
) -> WritingPMDecision:
    decisions_by_id = {
        decision["demand_id"]: decision
        for decision in ownership_resolution["decisions"]
        if decision["status"] == "accepted"
    }
    file_demands: list[dict[str, object]] = []
    for demand in pm_decision["file_demands"]:
        copied_demand = dict(demand)
        demand_id = copied_demand.get("demand_id")
        if isinstance(demand_id, str) and demand_id in decisions_by_id:
            decision = decisions_by_id[demand_id]
            copied_demand["preferred_path"] = decision["owned_path"]
            copied_demand["read_only_paths"] = decision["read_only_paths"]
        file_demands.append(copied_demand)

    resolved_decision: WritingPMDecision = {
        "status": pm_decision["status"],
        "mode": pm_decision["mode"],
        "intent": pm_decision["intent"],
        "file_demands": file_demands,
        "file_contracts": pm_decision["file_contracts"],
        "cross_module_imports": pm_decision["cross_module_imports"],
        "missing_slots": pm_decision["missing_slots"],
        "reading_requests": pm_decision["reading_requests"],
        "external_evidence_requests": pm_decision["external_evidence_requests"],
    }
    return resolved_decision


def _pm_decision_with_file_contracts(
    pm_decision: WritingPMDecision,
    file_contracts: list[WritingFileModuleContract],
) -> WritingPMDecision:
    resolved_decision: WritingPMDecision = {
        "status": pm_decision["status"],
        "mode": pm_decision["mode"],
        "intent": pm_decision["intent"],
        "file_demands": pm_decision["file_demands"],
        "file_contracts": file_contracts,
        "cross_module_imports": pm_decision["cross_module_imports"],
        "missing_slots": pm_decision["missing_slots"],
        "reading_requests": pm_decision["reading_requests"],
        "external_evidence_requests": pm_decision["external_evidence_requests"],
    }
    return resolved_decision


def _file_plan_rejected_result(
    *,
    mode: WritingMode,
    evaluation: dict[str, object],
    session,
    external_evidence: list[ExternalEvidenceSummary],
    trace_summary: list[str],
) -> CodeWritingResult:
    errors = _file_plan_errors(evaluation)
    validation = _empty_validation("rejected")
    validation["errors"] = errors
    return _result(
        status="rejected",
        mode=mode,
        answer_text="The writing file plan needs repair: " + "; ".join(errors),
        patch_artifacts=[],
        created_files=[],
        changed_files=[],
        external_evidence_requests=[],
        external_evidence=external_evidence,
        validation=validation,
        session=session,
        limitations=errors,
        trace_summary=[*trace_summary, "writing:rejected file plan"],
    )


def _external_evidence_result_if_needed(
    *,
    mode: WritingMode,
    pm_decision: WritingPMDecision,
    session,
    external_evidence: list[ExternalEvidenceSummary],
    trace_summary: list[str],
) -> CodeWritingResult | None:
    if not pm_decision["external_evidence_requests"]:
        return None
    return _result(
        status="need_external_evidence",
        mode=mode,
        answer_text="External evidence is required before proposing a patch.",
        patch_artifacts=[],
        created_files=[],
        changed_files=[],
        external_evidence_requests=pm_decision["external_evidence_requests"],
        external_evidence=external_evidence,
        validation=_empty_validation("failed"),
        session=session,
        limitations=[
            request["reason"]
            for request in pm_decision["external_evidence_requests"]
        ],
        trace_summary=[*trace_summary, "writing:need_external_evidence"],
    )


def _pm_terminal_result(
    *,
    mode: WritingMode,
    pm_decision: WritingPMDecision,
    session,
    external_evidence: list[ExternalEvidenceSummary],
    trace_summary: list[str],
) -> CodeWritingResult:
    status = "needs_user_input"
    reading_requests: list[WritingReadingEvidenceRequest] | None = None
    if pm_decision["status"] == "need_reading":
        status = "need_reading"
        reading_requests = pm_decision["reading_requests"]
    if pm_decision["status"] == "rejected":
        status = "rejected"
    answer_text = _missing_slot_text(pm_decision)
    return _result(
        status=status,
        mode=mode,
        answer_text=answer_text,
        patch_artifacts=[],
        created_files=[],
        changed_files=[],
        external_evidence_requests=[],
        external_evidence=external_evidence,
        validation=_empty_validation(status),
        session=session,
        limitations=_missing_slot_limitations(pm_decision),
        trace_summary=[
            *trace_summary,
            f"writing_pm:sufficiency={pm_decision['status']}",
        ],
        reading_requests=reading_requests,
    )


def _source_evidence_needed_decision(
    pm_decision: WritingPMDecision,
) -> WritingPMDecision:
    required_slots = list(pm_decision["missing_slots"])
    for demand in pm_decision["file_demands"]:
        for slot in demand.get("required_slots", []):
            if slot not in required_slots:
                required_slots.append(slot)
    if not required_slots:
        required_slots = [
            "Current source owners for the requested behavior.",
            "Tests or validation paths that cover the current behavior.",
        ]
    reading_request: WritingReadingEvidenceRequest = {
        "request_id": "source-evidence",
        "task": (
            "Collect current repository evidence needed before bounded "
            "implementation planning."
        ),
        "reason": "Source evidence is required before file planning.",
        "required_slots": required_slots[:8],
    }
    return {
        "status": "need_reading",
        "mode": pm_decision["mode"],
        "intent": pm_decision["intent"],
        "file_demands": [],
        "file_contracts": [],
        "cross_module_imports": {},
        "missing_slots": required_slots[:8],
        "reading_requests": [reading_request],
        "external_evidence_requests": [],
    }


def _mode(request: CodeWritingRequest) -> WritingMode:
    mode_hint = request.get("mode_hint")
    if mode_hint in ("edit_existing_repository", "create_new_project"):
        return mode_hint
    if request.get("repository") is None:
        return "create_new_project"
    return "edit_existing_repository"


def _repository_summary(
    repository: CodeRepositoryRef | None,
) -> dict[str, object] | None:
    if repository is None:
        return None
    return {
        "provider": repository["provider"],
        "owner": repository["owner"],
        "repo": repository["repo"],
        "source_url": repository["source_url"],
        "requested_ref": repository["requested_ref"],
        "resolved_ref": repository["resolved_ref"],
        "current_commit": repository["current_commit"],
        "default_branch": repository["default_branch"],
        "storage_kind": repository["storage_kind"],
        "managed_checkout": repository["managed_checkout"],
        "dirty_state": repository["dirty_state"],
    }


def _base_identity(
    *,
    mode: WritingMode,
    question: str,
    repository: CodeRepositoryRef | None,
    source_scope: CodeSourceScope | None,
) -> str:
    if repository is not None:
        parts = [
            mode,
            repository["provider"],
            repository["owner"],
            repository["repo"],
            repository["current_commit"],
            str(source_scope),
        ]
    else:
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
        parts = [mode, digest]
    return "|".join(parts)


def _reading_reports(
    reading_result: CodeReadingResult | None,
) -> list[dict[str, object]]:
    if reading_result is None:
        return []
    report = {
        "status": reading_result["status"],
        "answer_text": _truncate_text(
            reading_result["answer_text"],
            MAX_READING_REPORT_ANSWER_CHARS,
        ),
        "evidence_refs": [
            _evidence_ref(row)
            for row in reading_result["evidence"][:MAX_READING_REPORT_EVIDENCE_REFS]
        ],
        "evidence": _compact_reading_report_evidence(reading_result["evidence"]),
        "limitations": _bounded_text_list(
            reading_result["limitations"],
            max_items=MAX_READING_REPORT_LIMITATIONS,
            max_chars=MAX_READING_REPORT_LIMITATION_CHARS,
        ),
    }
    return [report]


def _compact_reading_report_evidence(
    evidence: list[CodeEvidenceRow],
) -> list[dict[str, object]]:
    compact_rows: list[dict[str, object]] = []
    rows_by_path: dict[str, int] = {}
    for row in evidence:
        safe_path = _safe_repo_relative_path(row["path"])
        if safe_path is None:
            continue
        if rows_by_path.get(safe_path, 0) >= MAX_READING_REPORT_EVIDENCE_ROWS_PER_PATH:
            continue
        compact_rows.append({
            "path": safe_path,
            "line_start": row["line_start"],
            "line_end": row["line_end"],
            "symbol_or_topic": row["symbol_or_topic"],
            "excerpt": _truncate_text(
                row["excerpt"],
                MAX_READING_REPORT_EXCERPT_CHARS,
            ),
            "reason": _truncate_text(
                row["reason"],
                MAX_READING_REPORT_LIMITATION_CHARS,
            ),
        })
        rows_by_path[safe_path] = rows_by_path.get(safe_path, 0) + 1
        if len(compact_rows) >= MAX_READING_REPORT_EVIDENCE_ROWS:
            break
    return compact_rows


def _reading_evidence(
    reading_result: CodeReadingResult | None,
) -> list[CodeEvidenceRow]:
    if reading_result is None:
        return []
    return reading_result["evidence"]


def _external_evidence_from_request(
    request: CodeWritingRequest,
) -> list[ExternalEvidenceSummary]:
    evidence = request.get("external_evidence", [])
    if not isinstance(evidence, list):
        return []
    return evidence


def _selected_evidence_for_contract(
    *,
    reading_evidence: list[CodeEvidenceRow],
    file_contract: WritingFileModuleContract,
) -> list[CodeEvidenceRow]:
    target_paths = _safe_paths([
        *file_contract.get("owned_paths", []),
        *file_contract.get("read_only_paths", []),
    ])
    selected_rows = [
        row
        for row in reading_evidence
        if _row_matches_any_path(row, target_paths)
    ]
    if not selected_rows:
        selected_rows = list(reading_evidence)
    return selected_rows[:MAX_SELECTED_EVIDENCE_ROWS]


def _row_matches_any_path(row: CodeEvidenceRow, target_paths: list[str]) -> bool:
    safe_path = _safe_repo_relative_path(row["path"])
    if safe_path is None:
        return False
    row_path = PurePosixPath(safe_path)
    for target_path in target_paths:
        target = PurePosixPath(target_path)
        if row_path == target or target in row_path.parents:
            return True
    return False


def _file_context_for_contract(
    *,
    repository: CodeRepositoryRef | None,
    selected_evidence: list[CodeEvidenceRow],
    file_contract: WritingFileModuleContract,
) -> list[dict[str, object]]:
    repo_root = _validation_repo_root(repository)
    context_rows: list[dict[str, object]] = []
    if repo_root is not None:
        for safe_path in _safe_paths([
            *file_contract.get("owned_paths", []),
            *file_contract.get("read_only_paths", []),
        ]):
            context = _read_file_context(repo_root=repo_root, safe_path=safe_path)
            if context is not None:
                context_rows.append(context)
    for row in selected_evidence:
        safe_path = _safe_repo_relative_path(row["path"])
        if safe_path is None:
            continue
        context_rows.append({
            "path": safe_path,
            "line_start": row["line_start"],
            "line_end": row["line_end"],
            "text": _truncate_text(
                row["excerpt"],
                MAX_SELECTED_EVIDENCE_EXCERPT_CHARS,
            ),
        })
    return context_rows[:MAX_LIST_ITEMS]


def _read_file_context(
    *,
    repo_root: Path,
    safe_path: str,
) -> dict[str, object] | None:
    file_path = repo_root / safe_path
    try:
        resolved_root = repo_root.expanduser().resolve(strict=True)
        resolved_file = file_path.expanduser().resolve(strict=True)
    except OSError:
        return None
    if resolved_file == resolved_root or resolved_root not in resolved_file.parents:
        return None
    if not resolved_file.is_file():
        return None
    try:
        text = resolved_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    full_chars = len(text)
    if full_chars > MAX_FILE_CONTEXT_CHARS:
        text = text[:MAX_FILE_CONTEXT_CHARS].rstrip()
    return {
        "path": safe_path,
        "line_start": 1,
        "line_end": len(text.splitlines()),
        "text": text,
        "full_chars": full_chars,
    }


def _file_context_summary(context_rows: list[dict[str, object]]) -> str:
    parts: list[str] = []
    for row in context_rows:
        path = row.get("path", "current file")
        text = row.get("text", "")
        if isinstance(text, str) and text.strip():
            parts.append(f"{path}: {text}")
    return "\n\n".join(parts)[:MAX_FILE_CONTEXT_CHARS]


def _compact_selected_evidence(
    selected_evidence: list[CodeEvidenceRow],
) -> list[dict[str, object]]:
    compact_rows: list[dict[str, object]] = []
    for row in selected_evidence[:MAX_SELECTED_EVIDENCE_ROWS]:
        compact_rows.append({
            "text": _truncate_text(
                row["excerpt"],
                MAX_SELECTED_EVIDENCE_EXCERPT_CHARS,
            ),
            "evidence_refs": [_evidence_ref(row)],
        })
    return compact_rows


def _file_label(file_contract: WritingFileModuleContract) -> str:
    return (
        file_contract.get("role")
        or file_contract.get("file_contract_id")
        or "file_module"
    )


def _edit_mode_for_contract(
    mode: WritingMode,
    file_contract: WritingFileModuleContract,
) -> str:
    file_kind = file_contract.get("file_kind")
    if mode == "create_new_project" or file_kind in {"new", "test", "docs", "config"}:
        return "complete_file"
    return "symbol_bundle"


def _content_format_for_contract(
    file_contract: WritingFileModuleContract,
) -> str:
    owned_path = file_contract.get("owned_path", "")
    suffix = PurePosixPath(owned_path).suffix.casefold()
    if suffix == ".py":
        return "python"
    file_kind = file_contract.get("file_kind")
    if file_kind in {"docs", "config"}:
        return "text"
    if suffix in {".md", ".markdown", ".txt", ".toml", ".yaml", ".yml", ".json"}:
        return "text"
    return "python"


def _lifecycle_owner(interface_contract: dict[str, object]) -> str:
    owner = interface_contract.get("lifecycle_owner")
    if isinstance(owner, str) and owner.strip():
        return owner.strip()
    component = interface_contract.get("component")
    if isinstance(component, str) and component.strip():
        return component.strip()
    return ""


def _provided_interfaces(
    interface_contract: dict[str, object],
    integration_contract: dict[str, object],
    *,
    file_contract: WritingFileModuleContract,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for name in _string_values(interface_contract.get("outputs")):
        items.append({"name": name, "kind": "function", "contract": ""})
    for name in _string_values(interface_contract.get("exports")):
        if not any(i["name"] == name for i in items):
            items.append({"name": name, "kind": "function", "contract": ""})
    for name in _string_values(integration_contract.get("provides_to_pm")):
        if not any(i["name"] == name for i in items):
            items.append({"name": name, "kind": "function", "contract": ""})
    if not items:
        for name in _string_values(file_contract.get("required_slots")):
            items.append({"name": name, "kind": "function", "contract": ""})
    return items[:MAX_LIST_ITEMS]


def _consumed_interfaces(
    integration_contract: dict[str, object],
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for name in _string_values(integration_contract.get("consumes_from")):
        items.append({"name": name, "provider_slice_id": "", "contract": ""})
    for name in _string_values(integration_contract.get("called_by")):
        if not any(i["name"] == name for i in items):
            items.append({"name": name, "provider_slice_id": "", "contract": ""})
    for name in _string_values(integration_contract.get("calls")):
        if not any(i["name"] == name for i in items):
            items.append({"name": name, "provider_slice_id": "", "contract": ""})
    return items[:MAX_LIST_ITEMS]


def _existing_source_anchors(
    interface_contract: dict[str, object],
    *,
    file_contract: WritingFileModuleContract,
    selected_evidence: list[CodeEvidenceRow] | None = None,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for inv in _string_values(interface_contract.get("invariants")):
        if inv not in seen_names:
            items.append({
                "name": inv,
                "kind": "function",
                "required_action": "preserve",
            })
            seen_names.add(inv)
    if selected_evidence:
        for row in selected_evidence:
            for name in _symbols_from_excerpt(row.get("excerpt", "")):
                if name not in seen_names:
                    items.append({
                        "name": name,
                        "kind": "function",
                        "required_action": "preserve",
                    })
                    seen_names.add(name)
    return items[:MAX_LIST_ITEMS]


def _symbols_from_excerpt(excerpt: str) -> list[str]:
    if not isinstance(excerpt, str):
        return []
    symbols: list[str] = []
    for line in excerpt.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("class "):
            name = stripped[6:].split("(")[0].split(":")[0].strip()
            if name:
                symbols.append(name)
        elif stripped.startswith("def "):
            name = stripped[4:].split("(")[0].strip()
            if name:
                symbols.append(name)
        elif stripped.startswith("async def "):
            name = stripped[10:].split("(")[0].strip()
            if name:
                symbols.append(name)
    return symbols


def _total_source_file_chars(context_rows: list[dict[str, object]]) -> int:
    total = 0
    for row in context_rows:
        full_chars = row.get("full_chars")
        if isinstance(full_chars, int):
            total += full_chars
        else:
            text = row.get("text", "")
            if isinstance(text, str):
                total += len(text)
    return total


def _integration_behaviors(
    integration_contract: dict[str, object],
    *,
    file_contract: WritingFileModuleContract,
) -> list[str]:
    values = _string_values(file_contract.get("validation_expectations"))
    if not values:
        change_goal = file_contract.get("change_goal")
        if isinstance(change_goal, str) and change_goal.strip():
            values.append(change_goal.strip())
    return _dedupe_strings(values)[:MAX_LIST_ITEMS]


def _cross_slice_interfaces(
    *,
    pm_decision: WritingPMDecision,
    current_file_contract_id: str,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for fc in pm_decision.get("file_contracts", []):
        fc_id = fc.get("file_contract_id", "")
        if fc_id == current_file_contract_id:
            continue
        ifc = fc.get("interface_contract", {})
        for name in _string_values(ifc.get("outputs")):
            items.append({
                "provider_slice_id": fc_id,
                "name": name,
                "contract": "",
            })
        for name in _string_values(ifc.get("exports")):
            if not any(
                i["provider_slice_id"] == fc_id and i["name"] == name
                for i in items
            ):
                items.append({
                    "provider_slice_id": fc_id,
                    "name": name,
                    "contract": "",
                })
    return items[:MAX_LIST_ITEMS]


def _imports_for_contract(
    *,
    pm_decision: WritingPMDecision,
    file_contract: WritingFileModuleContract,
    file_label: str,
) -> list[str]:
    import_lines: list[str] = []
    imports_by_key = pm_decision.get("cross_module_imports", {})
    for key in (file_contract["file_contract_id"], file_label):
        imports = imports_by_key.get(key)
        if isinstance(imports, list):
            import_lines.extend(_string_values(imports))
    import_lines.extend(_string_values(file_contract.get("cross_file_imports")))
    return _dedupe_strings(import_lines)[:MAX_LIST_ITEMS]


def _required_behavior(file_contract: WritingFileModuleContract) -> list[str]:
    values = _string_values(file_contract.get("validation_expectations"))
    values.extend(_string_values(file_contract.get("work_instructions")))
    values.extend(_string_values(file_contract.get("required_slots")))
    if not values and file_contract.get("purpose"):
        values.append(file_contract["purpose"])
    return _dedupe_strings(values)[:MAX_LIST_ITEMS]


def _materialize_patcher_report(
    *,
    question: str,
    mode: WritingMode,
    repository: CodeRepositoryRef | None,
    source_scope: CodeSourceScope | None,
    file_contracts: list[WritingFileModuleContract],
    programmer_reports: list[WritingProgrammerReport],
    integration_notes: list[str],
    max_artifact_chars: int,
    trace: dict[str, object] | None,
    trace_key: str,
    trace_summary: list[str],
) -> WritingPatcherReport:
    owned_path_map = _owned_path_map(file_contracts)
    patcher_input: WritingPatcherInput = {
        "question": question,
        "mode": mode,
        "base_identity": _base_identity(
            mode=mode,
            question=question,
            repository=repository,
            source_scope=source_scope,
        ),
        "owned_path_map": owned_path_map,
        "base_file_summaries": _base_file_summaries(
            repository=repository,
            owned_path_map=owned_path_map,
        ),
        "selected_programmer_reports": programmer_reports,
        "pm_integration_notes": integration_notes,
        "artifact_limits": {
            "max_files": MAX_PATCH_FILES,
            "max_diff_chars": max_artifact_chars,
        },
    }
    patcher_trace: dict[str, object] = {}
    report = materialize_patch_artifacts(
        repo_root=_validation_repo_root(repository),
        patcher_input=patcher_input,
        max_files=MAX_PATCH_FILES,
        max_diff_chars=max_artifact_chars,
        trace=patcher_trace,
    )
    _record_internal_trace(trace, trace_key, patcher_trace)
    trace_summary.append(
        "writing_patcher:"
        f"status={report['status']} "
        f"artifacts={len(report['patch_artifacts'])} "
        f"code_artifacts={_code_artifact_count(programmer_reports)} "
        f"diagnostics={len(report['edit_diagnostics'])}"
    )
    return report


def _owned_path_map(
    file_contracts: list[WritingFileModuleContract],
) -> dict[str, str]:
    owned_path_map: dict[str, str] = {}
    for file_contract in file_contracts:
        owner = file_contract["file_contract_id"]
        for safe_path in _safe_paths(file_contract.get("owned_paths", [])):
            owned_path_map.setdefault(safe_path, owner)
    return owned_path_map


def _base_file_summaries(
    *,
    repository: CodeRepositoryRef | None,
    owned_path_map: dict[str, str],
) -> list[dict[str, object]]:
    repo_root = _validation_repo_root(repository)
    summaries: list[dict[str, object]] = []
    for safe_path in sorted(owned_path_map):
        summary: dict[str, object] = {
            "path": safe_path,
            "owner": owned_path_map[safe_path],
            "exists": False,
        }
        if repo_root is not None:
            file_path = repo_root / safe_path
            try:
                resolved_root = repo_root.expanduser().resolve(strict=True)
                resolved_file = file_path.expanduser().resolve(strict=True)
            except OSError:
                summaries.append(summary)
                continue
            if (
                resolved_file == resolved_root
                or resolved_root not in resolved_file.parents
                or not resolved_file.is_file()
            ):
                summaries.append(summary)
                continue
            try:
                text = resolved_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                summaries.append(summary)
                continue
            summary["exists"] = True
            summary["line_count"] = len(text.splitlines())
            summary["char_count"] = len(text)
        summaries.append(summary)
    return summaries


def _pm_integration_notes(pm_decision: WritingPMDecision) -> list[str]:
    notes = [pm_decision["intent"]]
    notes.extend(pm_decision["missing_slots"])
    return _dedupe_strings([note for note in notes if note])


def _code_artifact_count(reports: list[WritingProgrammerReport]) -> int:
    return sum(1 for report in reports if report.get("code_artifact"))


def _file_plan_errors(evaluation: dict[str, object]) -> list[str]:
    raw_errors = evaluation.get("errors", [])
    if not isinstance(raw_errors, list):
        return ["File-plan evaluation did not return errors."]
    errors = [
        error
        for error in raw_errors
        if isinstance(error, str) and error.strip()
    ]
    if not errors:
        errors = ["File plan was not accepted."]
    return errors


def _module_contract_error_text(evaluation: dict[str, object]) -> str:
    errors = _file_plan_errors(evaluation)
    return (
        "Module programmer contract needs repair before programmer dispatch: "
        + "; ".join(errors)
    )


def _report_limitations(reports: list[WritingProgrammerReport]) -> list[str]:
    limitations: list[str] = []
    for report in reports:
        if report["status"] == "succeeded":
            continue
        limitations.extend(report["open_questions"])
    return limitations


def _validation_repo_root(repository: CodeRepositoryRef | None) -> Path | None:
    if repository is None:
        return None
    return Path(repository["local_root"])


def _status_from_validation(validation: PatchValidationSummary) -> str:
    if validation["status"] == "succeeded":
        return "succeeded"
    if validation["status"] == "rejected":
        return "rejected"
    return "failed"


def _validation_with_operation_errors(
    validation: PatchValidationSummary,
    operation_errors: list[str],
) -> PatchValidationSummary:
    if not operation_errors:
        return validation
    status = validation["status"]
    if status == "succeeded":
        status = "failed"
    errors = list(validation["errors"])
    for error in operation_errors:
        if error in errors:
            continue
        errors.append(error)
    return {
        **validation,
        "status": status,
        "errors": errors,
    }


def _missing_slot_text(pm_decision: WritingPMDecision) -> str:
    if pm_decision["status"] == "need_reading":
        reading_requests = pm_decision.get("reading_requests", [])
        if reading_requests:
            slots = reading_requests[0].get("required_slots", [])
            if slots:
                return "Source evidence is required: " + "; ".join(slots)
        return "Source evidence is required before proposing a patch."
    missing_slots = pm_decision["missing_slots"]
    if not missing_slots:
        return "Please narrow or clarify the code-writing request."
    return "Please narrow or clarify: " + "; ".join(missing_slots)


def _missing_slot_limitations(pm_decision: WritingPMDecision) -> list[str]:
    if pm_decision["status"] == "need_reading":
        limitations = list(pm_decision["missing_slots"])
        if not limitations:
            limitations = [
                "Source evidence is required before file planning.",
            ]
        return limitations
    limitations = list(pm_decision["missing_slots"])
    if not limitations:
        limitations = ["The writing PM could not identify a bounded patch scope."]
    return limitations


def _empty_validation(status: str) -> PatchValidationSummary:
    validation_status = "failed"
    if status == "rejected":
        validation_status = "rejected"
    return {
        "status": validation_status,
        "parsed": False,
        "sandbox_applied": False,
        "errors": [],
        "warnings": [],
        "files": [],
    }


def _safe_paths(values: list[str]) -> list[str]:
    paths: list[str] = []
    for value in values:
        safe_path = _safe_repo_relative_path(value)
        if safe_path is None or safe_path in paths:
            continue
        paths.append(safe_path)
    return paths


def _string_values(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _joined_text(values: object, *, fallback: str = "") -> str:
    strings = _string_values(values)
    if strings:
        return "; ".join(strings)
    return fallback


def _bounded_text_list(
    values: list[str],
    *,
    max_items: int,
    max_chars: int,
) -> list[str]:
    bounded_values: list[str] = []
    for value in values:
        bounded_values.append(_truncate_text(value, max_chars))
        if len(bounded_values) >= max_items:
            break
    return bounded_values


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    suffix = " ... [truncated]"
    if max_chars <= len(suffix):
        return value[:max_chars].rstrip()
    return value[:max_chars - len(suffix)].rstrip() + suffix


def _evidence_ref(row: CodeEvidenceRow) -> str:
    return f"{row['path']}:{row['line_start']}-{row['line_end']}"


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value in deduped:
            continue
        deduped.append(value)
    return deduped


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
    mode: WritingMode,
    answer_text: str,
    patch_artifacts: list[PatchArtifact],
    created_files: list[CreatedFileSummary],
    changed_files: list[ChangedFileSummary],
    external_evidence_requests,
    external_evidence: list[ExternalEvidenceSummary],
    validation: PatchValidationSummary,
    session,
    limitations: list[str],
    trace_summary: list[str],
    reading_requests: list[WritingReadingEvidenceRequest] | None = None,
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
    if reading_requests is not None:
        result["reading_requests"] = reading_requests
    return result
