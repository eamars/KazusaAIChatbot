"""Supervisor for the PM/programmer code-reading flow."""

from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
    EvidenceCollectionError,
)
from kazusa_ai_chatbot.coding_agent.code_reading.models import (
    CodeEvidenceRow,
    CodeReadingRequest,
    CodeReadingResult,
    PMDecision,
    PMInput,
    ProgrammerAssignment,
    ProgrammerReport,
    ReadingManagerState,
)
from kazusa_ai_chatbot.coding_agent.code_reading.planner import (
    rejection_reason,
    source_scope_rejection_reason,
)
from kazusa_ai_chatbot.coding_agent.code_reading.product_manager import (
    decide_reading_work,
    selected_evidence_from_reports,
    validate_programmer_assignment,
)
from kazusa_ai_chatbot.coding_agent.code_reading.programmer import (
    run_programmer_assignment,
)
from kazusa_ai_chatbot.coding_agent.code_reading.repository_map import (
    build_repository_map_summary,
)
from kazusa_ai_chatbot.coding_agent.code_reading.synthesizer import (
    synthesize_from_programmer_reports,
)

DEFAULT_MAX_ANSWER_CHARS = 4000
MAX_PROGRAMMERS_PER_WAVE = 3
MAX_PROGRAMMER_WAVES = 3
MAX_PROGRAMMER_REPORTS_PER_PM = 6
MAX_FILES_PER_PROGRAMMER = 6
MAX_EXCERPT_CHARS_PER_PROGRAMMER = 12000


def run_reading_supervisor(
    request: CodeReadingRequest,
    *,
    trace: dict[str, object] | None = None,
) -> CodeReadingResult:
    """Answer a source-code question through PM/programmer decomposition."""

    question = request.get("question", "").strip()
    if not question:
        result = _result(
            status="needs_user_input",
            answer_text="Please provide the code-reading question.",
            evidence=[],
            limitations=[],
            trace_summary=["reading_pm:missing question"],
        )
        return result

    rejected_reason = rejection_reason(question)
    if rejected_reason is not None:
        result = _result(
            status="rejected",
            answer_text=rejected_reason,
            evidence=[],
            limitations=[rejected_reason],
            trace_summary=["reading_pm:rejected unsupported request"],
        )
        return result

    repository = request.get("repository")
    source_scope = request.get("source_scope")
    if repository is None or source_scope is None:
        result = _result(
            status="failed",
            answer_text="Reading requires a successful Phase 0 repository source.",
            evidence=[],
            limitations=["Missing repository or source scope."],
            trace_summary=["reading_pm:missing source contract"],
        )
        return result

    scope_rejection = source_scope_rejection_reason(source_scope)
    if scope_rejection is not None:
        result = _result(
            status="rejected",
            answer_text=scope_rejection,
            evidence=[],
            limitations=[scope_rejection],
            trace_summary=["reading_pm:rejected unsafe source scope"],
        )
        return result

    repo_root = Path(repository["local_root"])
    if not repo_root.exists() or not repo_root.is_dir():
        result = _result(
            status="failed",
            answer_text="Repository checkout is unavailable for reading.",
            evidence=[],
            limitations=["Repository local checkout is unavailable."],
            trace_summary=["reading_pm:missing local checkout"],
        )
        return result

    repository_summary = _repository_summary(repository)
    repo_map_summary = build_repository_map_summary(repository, source_scope)
    trace_summary = [
        f"reading_pm:repository_map files={repo_map_summary['total_safe_files']}",
    ]
    state: ReadingManagerState = {
        "request": request,
        "repository_summary": repository_summary,
        "source_scope": source_scope,
        "repo_map_summary": repo_map_summary,
        "pm_decisions": [],
        "programmer_reports": [],
        "selected_evidence": [],
        "limitations": [],
        "trace_summary": trace_summary,
        "wave_count": 0,
    }

    for wave_index in range(MAX_PROGRAMMER_WAVES):
        state["wave_count"] = wave_index + 1
        pm_trace: dict[str, object] = {}
        pm_input = _pm_input(
            question=question,
            repository_summary=repository_summary,
            source_scope=source_scope,
            repo_map_summary=repo_map_summary,
            previous_reports=state["programmer_reports"],
        )
        pm_decision = decide_reading_work(pm_input, trace=pm_trace)
        pm_decision = _trim_assignments_to_report_budget(
            pm_decision=pm_decision,
            existing_reports=state["programmer_reports"],
            trace_summary=trace_summary,
        )
        state["pm_decisions"].append(pm_decision)
        _record_internal_trace(trace, f"pm_wave_{wave_index + 1}", pm_trace)
        trace_summary.append(
            "reading_pm:work_plan "
            f"intent={pm_decision['intent']} "
            f"status={pm_decision['status']} "
            f"assignments={len(pm_decision['assignments'])}"
        )

        if pm_decision["status"] == "need_programmers":
            overload_result = _overload_result_if_any(
                pm_decision,
                state["programmer_reports"],
                trace_summary,
            )
            if overload_result is not None:
                return overload_result
            try:
                _run_programmer_wave(
                    repository=repository,
                    source_scope=source_scope,
                    pm_decision=pm_decision,
                    state=state,
                    trace=trace,
                )
            except (EvidenceCollectionError, ValueError) as exc:
                result = _result(
                    status="rejected",
                    answer_text=str(exc),
                    evidence=[],
                    limitations=[str(exc)],
                    trace_summary=[
                        *trace_summary,
                        "reading_pm:rejected unsafe evidence collection",
                    ],
                )
                return result
            continue

        selected_evidence = selected_evidence_from_reports(
            state["programmer_reports"]
        )
        state["selected_evidence"] = selected_evidence
        if pm_decision["status"] == "sufficient":
            result = _synthesize_result(
                request=request,
                question=question,
                repository_summary=repository_summary,
                pm_decision=pm_decision,
                programmer_reports=state["programmer_reports"],
                selected_evidence=selected_evidence,
                trace_summary=trace_summary,
                trace=trace,
            )
            return result

        if pm_decision["status"] == "overloaded":
            result = _result(
                status="needs_user_input",
                answer_text=_missing_slot_text(pm_decision),
                evidence=selected_evidence,
                limitations=_missing_slot_limitations(pm_decision),
                trace_summary=[*trace_summary, "reading_pm:sufficiency=overloaded"],
            )
            return result

        result = _result(
            status="needs_user_input",
            answer_text=_missing_slot_text(pm_decision),
            evidence=selected_evidence,
            limitations=_missing_slot_limitations(pm_decision),
            trace_summary=[
                *trace_summary,
                "reading_pm:sufficiency=needs_user_input",
            ],
        )
        return result

    final_review = _final_pm_review(
        question=question,
        repository_summary=repository_summary,
        source_scope=source_scope,
        repo_map_summary=repo_map_summary,
        state=state,
        trace_summary=trace_summary,
        trace=trace,
    )
    if final_review is not None:
        return final_review

    selected_evidence = selected_evidence_from_reports(state["programmer_reports"])
    result = _result(
        status="needs_user_input",
        answer_text=(
            "The question needs a narrower scope before Phase 1 can answer "
            "from bounded programmer reports."
        ),
        evidence=selected_evidence,
        limitations=[
            "Programmer wave limit reached before PM declared the evidence sufficient.",
        ],
        trace_summary=[*trace_summary, "reading_pm:sufficiency=wave_limit"],
    )
    return result


def _final_pm_review(
    *,
    question: str,
    repository_summary: dict[str, object],
    source_scope: dict[str, object],
    repo_map_summary: dict[str, object],
    state: ReadingManagerState,
    trace_summary: list[str],
    trace: dict[str, object] | None,
) -> CodeReadingResult | None:
    if not state["programmer_reports"]:
        return None

    pm_trace: dict[str, object] = {}
    pm_input = _pm_input(
        question=question,
        repository_summary=repository_summary,
        source_scope=source_scope,
        repo_map_summary=repo_map_summary,
        previous_reports=state["programmer_reports"],
    )
    pm_input["review_mode"] = "final_no_more_programmers"
    pm_decision = decide_reading_work(pm_input, trace=pm_trace)
    state["pm_decisions"].append(pm_decision)
    _record_internal_trace(trace, "pm_final_review", pm_trace)
    trace_summary.append(
        "reading_pm:final_review "
        f"intent={pm_decision['intent']} "
        f"status={pm_decision['status']} "
        f"assignments={len(pm_decision['assignments'])}"
    )

    selected_evidence = selected_evidence_from_reports(
        state["programmer_reports"]
    )
    state["selected_evidence"] = selected_evidence
    if pm_decision["status"] == "sufficient":
        result = _synthesize_result(
            request=state["request"],
            question=question,
            repository_summary=repository_summary,
            pm_decision=pm_decision,
            programmer_reports=state["programmer_reports"],
            selected_evidence=selected_evidence,
            trace_summary=trace_summary,
            trace=trace,
        )
        return result

    if pm_decision["status"] == "overloaded":
        result = _result(
            status="needs_user_input",
            answer_text=_missing_slot_text(pm_decision),
            evidence=selected_evidence,
            limitations=_missing_slot_limitations(pm_decision),
            trace_summary=[*trace_summary, "reading_pm:sufficiency=overloaded"],
        )
        return result

    if pm_decision["status"] == "needs_user_input":
        result = _result(
            status="needs_user_input",
            answer_text=_missing_slot_text(pm_decision),
            evidence=selected_evidence,
            limitations=_missing_slot_limitations(pm_decision),
            trace_summary=[
                *trace_summary,
                "reading_pm:sufficiency=needs_user_input",
            ],
        )
        return result

    return None


def _run_programmer_wave(
    *,
    repository: dict[str, object],
    source_scope: dict[str, object],
    pm_decision: PMDecision,
    state: ReadingManagerState,
    trace: dict[str, object] | None,
) -> None:
    assignments = pm_decision["assignments"]
    for assignment in assignments:
        validate_programmer_assignment(assignment, source_scope)
        state["trace_summary"].append(f"programmer:{assignment['role']}")
        programmer_trace: dict[str, object] = {}
        report = run_programmer_assignment(
            repository,
            assignment,
            source_scope,
            max_files=MAX_FILES_PER_PROGRAMMER,
            max_excerpt_chars=MAX_EXCERPT_CHARS_PER_PROGRAMMER,
            trace=programmer_trace,
        )
        state["programmer_reports"].append(report)
        _record_internal_trace(
            trace,
            f"programmer_{assignment['assignment_id']}",
            programmer_trace,
        )
        state["trace_summary"].append(
            "programmer_report "
            f"{assignment['role']} status={report['status']} "
            f"evidence={len(report['evidence'])}"
        )


def _synthesize_result(
    *,
    request: CodeReadingRequest,
    question: str,
    repository_summary: dict[str, object],
    pm_decision: PMDecision,
    programmer_reports: list[ProgrammerReport],
    selected_evidence: list[CodeEvidenceRow],
    trace_summary: list[str],
    trace: dict[str, object] | None,
) -> CodeReadingResult:
    synthesis_trace: dict[str, object] = {}
    max_answer_chars = request.get("max_answer_chars", DEFAULT_MAX_ANSWER_CHARS)
    answer_text, limitations = synthesize_from_programmer_reports(
        question=question,
        pm_decision=pm_decision,
        programmer_reports=programmer_reports,
        evidence=selected_evidence,
        limitations=_report_limitations(programmer_reports),
        repository_summary=repository_summary,
        preferred_language=request.get("preferred_language"),
        max_answer_chars=max_answer_chars,
        trace=synthesis_trace,
    )
    _record_internal_trace(trace, "synthesis", synthesis_trace)
    result = _result(
        status="succeeded",
        answer_text=answer_text,
        evidence=selected_evidence,
        limitations=limitations,
        trace_summary=[*trace_summary, "reading_pm:sufficiency=sufficient"],
    )
    return result


def _pm_input(
    *,
    question: str,
    repository_summary: dict[str, object],
    source_scope: dict[str, object],
    repo_map_summary: dict[str, object],
    previous_reports: list[ProgrammerReport],
) -> PMInput:
    pm_input: PMInput = {
        "question": question,
        "repository_summary": repository_summary,
        "source_scope": dict(source_scope),
        "repo_map_summary": repo_map_summary,
        "previous_reports": previous_reports,
    }
    return pm_input


def _repository_summary(repository: dict[str, object]) -> dict[str, object]:
    summary = {
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
    return summary


def _overload_result_if_any(
    pm_decision: PMDecision,
    existing_reports: list[ProgrammerReport],
    trace_summary: list[str],
) -> CodeReadingResult | None:
    assignments = pm_decision["assignments"]
    if len(assignments) > MAX_PROGRAMMERS_PER_WAVE:
        result = _result(
            status="needs_user_input",
            answer_text="Please narrow the question to at most three source scopes.",
            evidence=[],
            limitations=["PM requested too many programmer assignments."],
            trace_summary=[*trace_summary, "reading_pm:sufficiency=overloaded"],
        )
        return result

    total_reports = len(existing_reports) + len(assignments)
    if total_reports > MAX_PROGRAMMER_REPORTS_PER_PM:
        result = _result(
            status="needs_user_input",
            answer_text="Please narrow the question before more source reading.",
            evidence=[],
            limitations=["Programmer report limit would be exceeded."],
            trace_summary=[*trace_summary, "reading_pm:sufficiency=overloaded"],
        )
        return result

    return None


def _trim_assignments_to_report_budget(
    *,
    pm_decision: PMDecision,
    existing_reports: list[ProgrammerReport],
    trace_summary: list[str],
) -> PMDecision:
    """Apply the supervisor-owned total report cap to PM assignments."""

    if pm_decision["status"] != "need_programmers":
        return pm_decision

    remaining_reports = MAX_PROGRAMMER_REPORTS_PER_PM - len(existing_reports)
    if remaining_reports <= 0:
        return pm_decision

    assignments = pm_decision["assignments"]
    if len(assignments) <= remaining_reports:
        return pm_decision

    trimmed_decision: PMDecision = {
        "status": pm_decision["status"],
        "intent": pm_decision["intent"],
        "required_slots": pm_decision["required_slots"],
        "assignments": assignments[:remaining_reports],
        "missing_slots": pm_decision["missing_slots"],
    }
    trace_summary.append(
        "reading_pm:assignment_budget_trimmed "
        f"remaining_reports={remaining_reports}"
    )
    return trimmed_decision


def _missing_slot_text(pm_decision: PMDecision) -> str:
    missing_slots = pm_decision["missing_slots"]
    if not missing_slots:
        return "Please narrow or clarify the code-reading question."
    text = "Please narrow or clarify: " + "; ".join(missing_slots)
    return text


def _missing_slot_limitations(pm_decision: PMDecision) -> list[str]:
    limitations = list(pm_decision["missing_slots"])
    if not limitations:
        limitations = ["The PM could not identify enough bounded evidence slots."]
    return limitations


def _report_limitations(reports: list[ProgrammerReport]) -> list[str]:
    limitations: list[str] = []
    for report in reports:
        if report["status"] == "succeeded":
            continue
        limitations.extend(report["open_questions"])
    return limitations


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
    answer_text: str,
    evidence: list[CodeEvidenceRow],
    limitations: list[str],
    trace_summary: list[str],
) -> CodeReadingResult:
    result: CodeReadingResult = {
        "status": status,
        "answer_text": answer_text,
        "evidence": evidence,
        "limitations": limitations,
        "trace_summary": trace_summary,
    }
    return result
