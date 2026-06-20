"""Orchestration for the code-reading subagent."""

from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
    EvidenceCollectionError,
    collect_evidence,
    find_definition_paths,
)
from kazusa_ai_chatbot.coding_agent.code_reading.models import (
    CodeReadingRequest,
    CodeReadingResult,
)
from kazusa_ai_chatbot.coding_agent.code_reading.planner import (
    build_plan,
    rejection_reason,
    source_scope_rejection_reason,
)
from kazusa_ai_chatbot.coding_agent.code_reading.synthesizer import (
    synthesize_answer,
)

DEFAULT_MAX_ANSWER_CHARS = 4000


def run(request: CodeReadingRequest) -> CodeReadingResult:
    """Answer a source-code question from a successful Phase 0 source contract."""

    question = request.get("question", "").strip()
    if not question:
        return _result(
            status="needs_user_input",
            answer_text="Please provide the code-reading question.",
            evidence=[],
            limitations=[],
            trace_summary=["reading:missing question"],
        )

    rejected_reason = rejection_reason(question)
    if rejected_reason is not None:
        return _result(
            status="rejected",
            answer_text=rejected_reason,
            evidence=[],
            limitations=[rejected_reason],
            trace_summary=["reading:rejected unsupported request"],
        )

    repository = request.get("repository")
    source_scope = request.get("source_scope")
    if repository is None or source_scope is None:
        return _result(
            status="failed",
            answer_text="Reading requires a successful Phase 0 repository source.",
            evidence=[],
            limitations=["Missing repository or source scope."],
            trace_summary=["reading:missing source contract"],
        )

    scope_rejection = source_scope_rejection_reason(source_scope)
    if scope_rejection is not None:
        return _result(
            status="rejected",
            answer_text=scope_rejection,
            evidence=[],
            limitations=[scope_rejection],
            trace_summary=["reading:rejected unsafe source scope"],
        )

    repo_root = Path(repository["local_root"])
    if not repo_root.exists() or not repo_root.is_dir():
        return _result(
            status="failed",
            answer_text="Repository checkout is unavailable for reading.",
            evidence=[],
            limitations=["Repository local checkout is unavailable."],
            trace_summary=["reading:missing local checkout"],
        )

    plan = build_plan(question, source_scope)
    if plan.broad:
        answer_text = synthesize_answer(
            question=question,
            plan=plan,
            evidence=[],
            preferred_language=request.get("preferred_language"),
            max_answer_chars=request.get(
                "max_answer_chars",
                DEFAULT_MAX_ANSWER_CHARS,
            ),
        )
        return _result(
            status="needs_user_input",
            answer_text=answer_text,
            evidence=[],
            limitations=[],
            trace_summary=["reading:broad question needs scope"],
        )

    if plan.symbol is not None:
        definition_paths = find_definition_paths(
            repo_root=repo_root,
            source_scope=source_scope,
            symbol=plan.symbol,
        )
        if len(definition_paths) > 1:
            return _result(
                status="needs_user_input",
                answer_text=(
                    f"Symbol `{plan.symbol}` is ambiguous across "
                    f"{', '.join(definition_paths[:5])}; please specify one."
                ),
                evidence=[],
                limitations=[],
                trace_summary=["reading:ambiguous symbol"],
            )

    try:
        bundle = collect_evidence(
            repo_root=repo_root,
            source_scope=source_scope,
            plan=plan,
        )
    except EvidenceCollectionError as exc:
        return _result(
            status="rejected",
            answer_text=str(exc),
            evidence=[],
            limitations=[str(exc)],
            trace_summary=["reading:rejected unsafe evidence collection"],
        )

    max_answer_chars = request.get("max_answer_chars", DEFAULT_MAX_ANSWER_CHARS)
    answer_text = synthesize_answer(
        question=question,
        plan=plan,
        evidence=bundle.rows,
        preferred_language=request.get("preferred_language"),
        max_answer_chars=max_answer_chars,
    )
    return _result(
        status="succeeded",
        answer_text=answer_text,
        evidence=bundle.rows,
        limitations=bundle.limitations,
        trace_summary=bundle.trace_summary,
    )


def _result(
    *,
    status: str,
    answer_text: str,
    evidence: list,
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
