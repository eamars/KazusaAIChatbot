"""Supervisor for the new-artifact code-writing flow."""

from __future__ import annotations

import hashlib
from pathlib import Path

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
    WritingArtifactItem,
    WritingPatcherInput,
    WritingPatcherReport,
    WritingPMDecision,
    WritingPMInput,
    WritingProgrammerContract,
    WritingProgrammerResult,
)
from kazusa_ai_chatbot.coding_agent.code_writing.acceptance import (
    derive_acceptance_criteria,
    evaluate_artifact_alignment,
)
from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
    validate_patch_artifacts,
)
from kazusa_ai_chatbot.coding_agent.code_writing.patcher import (
    materialize_patch_artifacts,
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

DEFAULT_MAX_ARTIFACT_CHARS = 48000
MAX_PATCH_FILES = 32
MAX_VALIDATION_REPAIR_ATTEMPTS = 1
MAX_PREVIOUS_ARTIFACT_EXCERPT_CHARS = 1600


async def run_writing_supervisor(
    request: CodeWritingRequest,
    *,
    trace: dict[str, object] | None = None,
) -> CodeWritingResult:
    """Produce a new-artifact patch proposal through the writing roles."""

    question = request.get("question", "").strip()
    mode = request.get("mode_hint", "create_new_project")
    if mode != "create_new_project":
        return _existing_source_rejected_result(mode=mode)

    if not question:
        return _result(
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

    workspace_root = request.get("workspace_root")
    if not workspace_root:
        return _result(
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

    external_evidence = _external_evidence_from_request(request)
    session = prepare_writing_workspace(
        workspace_root=workspace_root,
        session_id=request.get("session_id"),
        base_identity=_base_identity(question=question),
        mode="create_new_project",
    )
    trace_summary = [
        f"writing_session:handle={session['public_handle']}",
        "writing_pm:mode=create_new_project",
    ]
    acceptance_trace: dict[str, object] = {}
    acceptance = await derive_acceptance_criteria(
        question=question,
        trace=acceptance_trace,
    )
    _record_internal_trace(trace, "acceptance", acceptance_trace)
    trace_summary.append(
        "writing_acceptance:"
        f"status={acceptance['status']} "
        f"criteria={len(acceptance['acceptance_criteria'])}"
    )
    if acceptance["status"] != "pass":
        return _result(
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

    max_artifact_chars = request.get(
        "max_artifact_chars",
        DEFAULT_MAX_ARTIFACT_CHARS,
    )
    acceptance_criteria = acceptance["acceptance_criteria"]
    previous_artifacts: list[dict[str, object]] = []
    validation_feedback: PatchValidationSummary | None = None
    alignment_feedback: WritingAlignmentResult | None = None
    reservation_feedback: ArtifactReservationResult | None = None
    pm_decision: WritingPMDecision | None = None
    generated_artifacts: list[GeneratedArtifact] = []
    patcher_report: WritingPatcherReport | None = None
    validation: PatchValidationSummary | None = None
    alignment: WritingAlignmentResult | None = None

    for attempt_index in range(MAX_VALIDATION_REPAIR_ATTEMPTS + 1):
        repair_label = "" if attempt_index == 0 else f"_repair_{attempt_index}"
        if attempt_index > 0:
            trace_summary.append(f"writing_repair:attempt={attempt_index}")

        pm_decision = await _decide_pm(
            question=question,
            acceptance_criteria=acceptance_criteria,
            external_evidence=external_evidence,
            previous_artifacts=previous_artifacts,
            validation_feedback=validation_feedback,
            alignment_feedback=alignment_feedback,
            reservation_feedback=reservation_feedback,
            trace=trace,
            trace_key=f"pm_initial{repair_label}",
        )
        trace_summary.append(
            "writing_pm:decision "
            f"status={pm_decision['status']} "
            f"artifacts={len(pm_decision['artifact_items'])}"
        )

        if pm_decision["status"] == "need_external_evidence":
            return _pm_terminal_result(
                pm_decision=pm_decision,
                session=session,
                external_evidence=external_evidence,
                trace_summary=trace_summary,
            )
        if pm_decision["status"] != "need_programmers":
            return _pm_terminal_result(
                pm_decision=pm_decision,
                session=session,
                external_evidence=external_evidence,
                trace_summary=trace_summary,
            )

        reservation = reserve_new_artifact_paths(pm_decision["artifact_items"])
        _record_internal_trace(trace, f"file_agent{repair_label}", reservation)
        trace_summary.append(
            "file_agent:reservation "
            f"status={reservation['status']} "
            f"paths={len(reservation['reserved_paths'])}"
        )
        if reservation["status"] != "accepted":
            if attempt_index < MAX_VALIDATION_REPAIR_ATTEMPTS:
                reservation_feedback = reservation
                validation_feedback = None
                previous_artifacts = []
                continue
            return _reservation_rejected_result(
                reservation=reservation,
                session=session,
                external_evidence=external_evidence,
                trace_summary=trace_summary,
            )

        programmer_results = await _run_programmers(
            artifact_items=pm_decision["artifact_items"],
            reserved_paths=reservation["reserved_paths"],
            trace=trace,
            trace_summary=trace_summary,
            trace_label=repair_label,
        )
        generated_artifacts = _generated_artifacts(
            artifact_items=pm_decision["artifact_items"],
            reserved_paths=reservation["reserved_paths"],
            programmer_results=programmer_results,
        )
        diagnostics = _programmer_diagnostics(programmer_results)
        if diagnostics:
            return _programmer_failed_result(
                diagnostics=diagnostics,
                session=session,
                external_evidence=external_evidence,
                trace_summary=trace_summary,
            )

        patcher_report = _materialize_patcher_report(
            generated_artifacts=generated_artifacts,
            reserved_paths=reservation["reserved_paths"],
            max_artifact_chars=max_artifact_chars,
            trace=trace,
            trace_summary=trace_summary,
            trace_label=repair_label,
        )
        validation = validate_patch_artifacts(
            repo_root=None,
            workspace_root=Path(workspace_root),
            patch_artifacts=patcher_report["patch_artifacts"],
            max_files=MAX_PATCH_FILES,
            max_diff_chars=max_artifact_chars,
        )
        validation = _validation_with_patcher_diagnostics(
            validation,
            patcher_report["diagnostics"],
        )
        trace_summary.append(f"writing_validation:status={validation['status']}")
        if validation["status"] == "succeeded":
            alignment_trace: dict[str, object] = {}
            alignment = await evaluate_artifact_alignment(
                question=question,
                acceptance_criteria=acceptance_criteria,
                pm_decision=pm_decision,
                generated_artifacts=generated_artifacts,
                validation=validation,
                trace=alignment_trace,
            )
            _record_internal_trace(
                trace,
                f"alignment{repair_label}",
                alignment_trace,
            )
            trace_summary.append(
                "writing_alignment:"
                f"status={alignment['status']} "
                f"confidence={alignment['confidence']}"
            )
            if alignment["status"] == "pass":
                break
            if attempt_index < MAX_VALIDATION_REPAIR_ATTEMPTS:
                alignment_feedback = alignment
                validation_feedback = None
                reservation_feedback = None
                previous_artifacts = _previous_artifact_summaries(
                    generated_artifacts,
                )
                trace_summary.append("writing_repair:alignment_feedback_to_pm")
                continue
            break
        if attempt_index >= MAX_VALIDATION_REPAIR_ATTEMPTS:
            break

        validation_feedback = validation
        alignment_feedback = None
        reservation_feedback = None
        previous_artifacts = _previous_artifact_summaries(generated_artifacts)
        trace_summary.append("writing_repair:validation_feedback_to_pm")

    if pm_decision is None or patcher_report is None or validation is None:
        return _result(
            status="failed",
            mode="create_new_project",
            answer_text="Code writing did not produce a patch package.",
            patch_artifacts=[],
            created_files=[],
            changed_files=[],
            external_evidence_requests=[],
            external_evidence=external_evidence,
            validation=_empty_validation("failed"),
            session=session,
            limitations=["No patch package was produced."],
            trace_summary=trace_summary,
        )

    synthesis_trace: dict[str, object] = {}
    alignment_blockers: list[str] = []
    if alignment is not None and alignment["status"] != "pass":
        alignment_blockers = alignment["blockers"]
    limitations = _dedupe_strings([
        *pm_decision["limitations"],
        *patcher_report["diagnostics"],
        *validation["errors"],
        *alignment_blockers,
    ])
    answer_text, limitations = await synthesize_patch_proposal(
        question=question,
        pm_decision=pm_decision,
        generated_artifacts=generated_artifacts,
        patch_artifacts=patcher_report["patch_artifacts"],
        validation=validation,
        external_evidence=external_evidence,
        limitations=limitations,
        preferred_language=request.get("preferred_language"),
        max_answer_chars=request.get("max_answer_chars", DEFAULT_MAX_ANSWER_CHARS),
        trace=synthesis_trace,
    )
    _record_internal_trace(trace, "synthesis", synthesis_trace)

    return _result(
        status=_status_from_validation_and_alignment(
            validation=validation,
            alignment=alignment,
        ),
        mode="create_new_project",
        answer_text=answer_text,
        patch_artifacts=patcher_report["patch_artifacts"],
        created_files=patcher_report["created_files"],
        changed_files=patcher_report["changed_files"],
        external_evidence_requests=[],
        external_evidence=external_evidence,
        validation=validation,
        alignment=alignment,
        session=session,
        limitations=limitations,
        trace_summary=trace_summary,
    )


async def _decide_pm(
    *,
    question: str,
    acceptance_criteria: list[WritingAcceptanceCriterion],
    external_evidence: list[ExternalEvidenceSummary],
    previous_artifacts: list[dict[str, object]],
    validation_feedback: PatchValidationSummary | None = None,
    alignment_feedback: WritingAlignmentResult | None = None,
    reservation_feedback: ArtifactReservationResult | None = None,
    trace: dict[str, object] | None,
    trace_key: str,
) -> WritingPMDecision:
    pm_trace: dict[str, object] = {}
    pm_input: WritingPMInput = {
        "question": question,
        "mode": "create_new_project",
        "external_evidence": external_evidence,
        "previous_artifacts": previous_artifacts,
        "acceptance_criteria": acceptance_criteria,
    }
    if validation_feedback is not None:
        pm_input["validation_feedback"] = validation_feedback
    if alignment_feedback is not None:
        pm_input["alignment_feedback"] = alignment_feedback
    if reservation_feedback is not None:
        pm_input["reservation_feedback"] = reservation_feedback
    decision = await decide_writing_work(pm_input, trace=pm_trace)
    _record_internal_trace(trace, trace_key, pm_trace)
    return decision


async def _run_programmers(
    *,
    artifact_items: list[WritingArtifactItem],
    reserved_paths: list[ReservedArtifactPath],
    trace: dict[str, object] | None,
    trace_summary: list[str],
    trace_label: str = "",
) -> list[WritingProgrammerResult]:
    results: list[WritingProgrammerResult] = []
    local_modules = _local_python_modules_by_artifact(
        artifact_items=artifact_items,
        reserved_paths=reserved_paths,
    )
    for artifact_item in artifact_items:
        contract = _programmer_contract(
            artifact_item,
            local_modules=local_modules,
        )
        programmer_trace: dict[str, object] = {}
        result = await run_writing_programmer_contract(
            artifact_contract=contract,
            trace=programmer_trace,
        )
        results.append(result)
        _record_internal_trace(
            trace,
            f"programmer_{contract['artifact_id']}{trace_label}",
            programmer_trace,
        )
        trace_summary.append(
            "programmer_report "
            f"{contract['artifact_id']} status={result['status']} "
            f"chars={len(result['code_artifact'])}"
        )
    return results


def _programmer_contract(
    artifact_item: WritingArtifactItem,
    *,
    local_modules: dict[str, str] | None = None,
) -> WritingProgrammerContract:
    contract: WritingProgrammerContract = {
        "artifact_id": artifact_item["artifact_id"],
        "file_label": artifact_item["file_label"],
        "file_kind": artifact_item["file_kind"],
        "content_format": artifact_item["content_format"],
        "purpose": artifact_item["purpose"],
        "imports": _contract_imports(
            artifact_item,
            local_modules=local_modules or {},
        ),
        "provided_interfaces": artifact_item["provided_interfaces"],
        "consumed_interfaces": artifact_item["consumed_interfaces"],
        "required_behavior": artifact_item["required_behavior"],
    }
    return contract


def _local_python_modules_by_artifact(
    *,
    artifact_items: list[WritingArtifactItem],
    reserved_paths: list[ReservedArtifactPath],
) -> dict[str, str]:
    modules: dict[str, str] = {}
    items_by_id = {item["artifact_id"]: item for item in artifact_items}
    for reserved_path in reserved_paths:
        item = items_by_id.get(reserved_path["artifact_id"])
        if item is None:
            continue
        if item["file_kind"] != "source" or item["content_format"] != "python":
            continue
        path = Path(reserved_path["path"])
        if path.suffix.casefold() != ".py":
            continue
        module_name = path.stem
        modules[item["artifact_id"]] = module_name
        modules[item["file_label"].casefold()] = module_name
    return modules


def _contract_imports(
    artifact_item: WritingArtifactItem,
    *,
    local_modules: dict[str, str],
) -> list[str]:
    imports = list(artifact_item["imports"])
    for consumed_interface in artifact_item["consumed_interfaces"]:
        name = consumed_interface.get("name")
        provider = consumed_interface.get("provider")
        if not isinstance(name, str) or not isinstance(provider, str):
            continue
        module_name = local_modules.get(provider)
        if module_name is None:
            module_name = local_modules.get(provider.casefold())
        if module_name is None:
            continue
        imports = _without_local_interface_import(
            imports,
            interface_name=name,
        )
        imports.append(f"from {module_name} import {name}")
    return _dedupe_strings(imports)


def _without_local_interface_import(
    imports: list[str],
    *,
    interface_name: str,
) -> list[str]:
    kept: list[str] = []
    for import_statement in imports:
        if _imports_interface_name(import_statement, interface_name):
            continue
        kept.append(import_statement)
    return kept


def _imports_interface_name(import_statement: str, interface_name: str) -> bool:
    if not import_statement.startswith("from "):
        return False
    marker = " import "
    if marker not in import_statement:
        return False
    imported = import_statement.split(marker, 1)[1]
    names = [part.strip().split(" as ", 1)[0] for part in imported.split(",")]
    return interface_name in names


def _generated_artifacts(
    *,
    artifact_items: list[WritingArtifactItem],
    reserved_paths: list[ReservedArtifactPath],
    programmer_results: list[WritingProgrammerResult],
) -> list[GeneratedArtifact]:
    items_by_id = {item["artifact_id"]: item for item in artifact_items}
    paths_by_id = {path["artifact_id"]: path for path in reserved_paths}
    artifacts: list[GeneratedArtifact] = []
    for result in programmer_results:
        if result["status"] != "succeeded":
            continue
        artifact_id = result["artifact_id"]
        artifact_item = items_by_id[artifact_id]
        reserved_path = paths_by_id[artifact_id]
        artifact: GeneratedArtifact = {
            "artifact_id": artifact_id,
            "file_label": artifact_item["file_label"],
            "file_kind": artifact_item["file_kind"],
            "content_format": artifact_item["content_format"],
            "path": reserved_path["path"],
            "content": result["code_artifact"],
            "purpose": artifact_item["purpose"],
        }
        artifacts.append(artifact)
    return artifacts


def _materialize_patcher_report(
    *,
    generated_artifacts: list[GeneratedArtifact],
    reserved_paths: list[ReservedArtifactPath],
    max_artifact_chars: int,
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
    report = materialize_patch_artifacts(
        repo_root=None,
        patcher_input=patcher_input,
        max_files=MAX_PATCH_FILES,
        max_diff_chars=max_artifact_chars,
        trace=patcher_trace,
    )
    _record_internal_trace(trace, f"patcher{trace_label}", patcher_trace)
    trace_summary.append(
        "writing_patcher:"
        f"status={report['status']} "
        f"artifacts={len(report['patch_artifacts'])} "
        f"diagnostics={len(report['diagnostics'])}"
    )
    return report


def _previous_artifact_summaries(
    generated_artifacts: list[GeneratedArtifact],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for artifact in generated_artifacts:
        summaries.append({
            "artifact_id": artifact["artifact_id"],
            "file_kind": artifact["file_kind"],
            "content_format": artifact["content_format"],
            "purpose": artifact["purpose"],
            "content_excerpt": _artifact_excerpt(artifact["content"]),
        })
    return summaries


def _artifact_excerpt(content: str) -> str:
    if len(content) <= MAX_PREVIOUS_ARTIFACT_EXCERPT_CHARS:
        return content
    return content[:MAX_PREVIOUS_ARTIFACT_EXCERPT_CHARS].rstrip()


def _pm_terminal_result(
    *,
    pm_decision: WritingPMDecision,
    session: dict[str, object],
    external_evidence: list[ExternalEvidenceSummary],
    trace_summary: list[str],
) -> CodeWritingResult:
    status = "failed"
    validation_status = "failed"
    if pm_decision["status"] == "need_external_evidence":
        status = "need_external_evidence"
    elif pm_decision["status"] == "rejected":
        status = "rejected"
        validation_status = "rejected"
    elif pm_decision["status"] == "sufficient":
        status = "failed"

    answer_text = _terminal_answer(pm_decision)
    return _result(
        status=status,
        mode="create_new_project",
        answer_text=answer_text,
        patch_artifacts=[],
        created_files=[],
        changed_files=[],
        external_evidence_requests=pm_decision["external_evidence_requests"],
        external_evidence=external_evidence,
        validation=_empty_validation(validation_status),
        session=session,
        limitations=pm_decision["limitations"],
        trace_summary=trace_summary,
    )


def _terminal_answer(pm_decision: WritingPMDecision) -> str:
    if pm_decision["status"] == "need_external_evidence":
        return "External evidence is required before writing artifacts."
    if pm_decision["limitations"]:
        return "; ".join(pm_decision["limitations"])
    return "The writing PM did not produce artifact work."


def _reservation_rejected_result(
    *,
    reservation: ArtifactReservationResult,
    session: dict[str, object],
    external_evidence: list[ExternalEvidenceSummary],
    trace_summary: list[str],
) -> CodeWritingResult:
    return _result(
        status="failed",
        mode="create_new_project",
        answer_text="File reservation failed before programmer dispatch.",
        patch_artifacts=[],
        created_files=[],
        changed_files=[],
        external_evidence_requests=[],
        external_evidence=external_evidence,
        validation=_empty_validation("failed"),
        session=session,
        limitations=reservation["errors"],
        trace_summary=trace_summary,
    )


def _programmer_failed_result(
    *,
    diagnostics: list[str],
    session: dict[str, object],
    external_evidence: list[ExternalEvidenceSummary],
    trace_summary: list[str],
) -> CodeWritingResult:
    return _result(
        status="failed",
        mode="create_new_project",
        answer_text="One or more programmers did not return usable artifacts.",
        patch_artifacts=[],
        created_files=[],
        changed_files=[],
        external_evidence_requests=[],
        external_evidence=external_evidence,
        validation=_empty_validation("failed"),
        session=session,
        limitations=diagnostics,
        trace_summary=trace_summary,
    )


def _existing_source_rejected_result(*, mode: str) -> CodeWritingResult:
    return _result(
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


def _programmer_diagnostics(
    programmer_results: list[WritingProgrammerResult],
) -> list[str]:
    diagnostics: list[str] = []
    for result in programmer_results:
        if result["status"] == "succeeded":
            continue
        diagnostics.extend(result["diagnostics"])
    return _dedupe_strings(diagnostics)


def _external_evidence_from_request(
    request: CodeWritingRequest,
) -> list[ExternalEvidenceSummary]:
    external_evidence = request.get("external_evidence", [])
    if not isinstance(external_evidence, list):
        return []
    return external_evidence


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
    return {
        "status": validation_status,
        "parsed": False,
        "sandbox_applied": False,
        "errors": [],
        "warnings": [],
        "files": [],
    }


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
    mode: str,
    answer_text: str,
    patch_artifacts: list[dict[str, object]],
    created_files: list[dict[str, str]],
    changed_files: list[dict[str, str]],
    external_evidence_requests,
    external_evidence: list[ExternalEvidenceSummary],
    validation: PatchValidationSummary,
    session,
    limitations: list[str],
    trace_summary: list[str],
    alignment: WritingAlignmentResult | None = None,
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
    return result
