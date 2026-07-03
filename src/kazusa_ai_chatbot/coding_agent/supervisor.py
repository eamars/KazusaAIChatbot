"""Top-level standalone coding-agent supervisor."""

import inspect
import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

import kazusa_ai_chatbot.coding_agent.code_fetching as code_fetching
import kazusa_ai_chatbot.coding_agent.code_reading as code_reading
import kazusa_ai_chatbot.coding_agent.code_writing as code_writing
from kazusa_ai_chatbot.coding_agent.external_evidence import (
    collect_external_evidence,
)
from kazusa_ai_chatbot.coding_agent.code_fetching.models import (
    CodeRepositoryRef,
)
from kazusa_ai_chatbot.coding_agent.code_writing.synthesizer import (
    DEFAULT_MAX_ANSWER_CHARS,
    _answer_with_required_limitations,
)
from kazusa_ai_chatbot.coding_agent.models import (
    CodingAgentBackgroundOperation,
    CodingAgentBackgroundRequest,
    CodingAgentBackgroundResponse,
    CodingAgentWriteRequest,
    CodingPatchProposalResponse,
    CodingAgentRepositorySummary,
    CodingAgentRequest,
    CodingAgentResponse,
)
from kazusa_ai_chatbot.coding_agent.work_ledger import (
    CodingSupervisorWorkLedger,
)
from kazusa_ai_chatbot.config import (
    CODING_AGENT_PM_LLM_API_KEY,
    CODING_AGENT_PM_LLM_BASE_URL,
    CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS,
    CODING_AGENT_PM_LLM_MODEL,
    CODING_AGENT_PM_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

DIRTY_CHECKOUT_LIMITATION = (
    "Existing local checkout is dirty; evidence may include "
    "uncommitted local changes."
)
GIT_INTERNAL_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])\.git(?![A-Za-z0-9_])")
ENV_FILE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])\.env(?![A-Za-z0-9_])")
CACHE_KEY_TOKEN_RE = re.compile(r"cache_key", re.IGNORECASE)
WRITE_EVIDENCE_EXCERPT_OMITTED = (
    "[source excerpt omitted from patch proposal response]"
)
BACKGROUND_CODING_ROUTER_PROMPT = '''\
You are the top-level supervisor inside a coding agent.

Choose the single supported operation that should handle the accepted coding
task.

Operations:
- code_reading: answer a question about a codebase, project design, source
  behavior, architecture, dependency use, tests, or repository structure.
- code_writing: propose new-file code artifacts for source-free coding tasks.
  The coding agent may return proposal artifacts, but it does not edit existing
  source, apply patches, run commands, install packages, deploy, or validate by
  executing the target project.
- unsupported: the task is not a coding task, or it requires live execution,
  existing-source edits, deployment, credential access, package installation,
  adapter delivery, or real-world mutation as the primary result.

Return strict JSON:
{
  "operation": "code_reading | code_writing | unsupported",
  "reason": "short reason"
}
'''
WRITE_LOOP_LIMIT = 6
MAX_EXISTING_REPO_FOLLOWUP_READING_ATTEMPTS = 1
BACKGROUND_CODING_ROUTER_TIMEOUT_SECONDS = 300
BACKGROUND_CODING_ROUTER_INVALID_REASON = (
    "Coding-agent background supervisor returned an invalid operation."
)


_background_coding_router_llm = LLInterface()
_background_coding_router_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CODING_AGENT_PM_LLM",
    base_url=CODING_AGENT_PM_LLM_BASE_URL,
    api_key=CODING_AGENT_PM_LLM_API_KEY,
    model=CODING_AGENT_PM_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    timeout_seconds=BACKGROUND_CODING_ROUTER_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_PM_LLM_THINKING_ENABLED,
    ),
)


async def handle_background_coding_task(
    request: CodingAgentBackgroundRequest,
) -> CodingAgentBackgroundResponse:
    """Route one accepted background coding task inside the coding agent."""

    question = _bounded_request_body(request.get("question"))
    if not question:
        response = _background_failure_response(
            status="failed",
            operation="unsupported",
            limitation="Coding agent did not receive a task question.",
            trace_summary=["background_coding:missing_question"],
        )
        return response

    operation, reason = await _decide_background_coding_operation(request)
    if operation == "code_reading":
        reading_response = await answer_code_question(
            _background_reading_request(request),
        )
        response = _background_response_from_reading(
            reading_response,
            route_reason=reason,
        )
        return response
    if operation == "code_writing":
        writing_response = await propose_code_change(
            _background_writing_request(request),
        )
        response = _background_response_from_writing(
            writing_response,
            route_reason=reason,
        )
        return response

    failure_status = "rejected"
    if reason == BACKGROUND_CODING_ROUTER_INVALID_REASON:
        failure_status = "failed"
    response = _background_failure_response(
        status=failure_status,
        operation="unsupported",
        limitation=_unsupported_background_limitation(reason),
        trace_summary=[
            f"background_coding:unsupported:{_safe_request_text(reason)}",
        ],
    )
    return response


async def _decide_background_coding_operation(
    request: CodingAgentBackgroundRequest,
) -> tuple[CodingAgentBackgroundOperation, str]:
    """Ask the coding-agent supervisor route to choose the supported operation."""

    payload = {
        "task": _bounded_request_body(request.get("question")),
        "source_summary": _bounded_request_body(request.get("source_summary")),
        "available_operations": [
            "code_reading",
            "code_writing",
            "unsupported",
        ],
        "operation_limits": [
            "No patch application.",
            "No shell command execution.",
            "No package installation.",
            "No adapter delivery.",
        ],
    }
    payload_text = json.dumps(payload, ensure_ascii=False)
    messages = [
        SystemMessage(content=BACKGROUND_CODING_ROUTER_PROMPT),
        HumanMessage(content=payload_text),
    ]
    response = await _background_coding_router_llm.ainvoke(
        messages,
        config=_background_coding_router_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    operation = _normalize_background_coding_operation(parsed)
    if operation == "unsupported" and not _parsed_operation_is_supported(parsed):
        return operation, BACKGROUND_CODING_ROUTER_INVALID_REASON
    reason = ""
    if isinstance(parsed, dict):
        reason = _safe_request_text(parsed.get("reason"))
    return operation, reason


def _parsed_operation_is_supported(parsed: object) -> bool:
    """Return whether the LLM emitted a recognized operation field."""

    if not isinstance(parsed, dict):
        return False
    operation = parsed.get("operation")
    return operation in ("code_reading", "code_writing", "unsupported")


def _normalize_background_coding_operation(
    parsed: object,
) -> CodingAgentBackgroundOperation:
    """Validate the supervisor route decision without semantic fallback."""

    if not isinstance(parsed, dict):
        return "unsupported"
    operation = parsed.get("operation")
    if operation in ("code_reading", "code_writing", "unsupported"):
        return operation
    return "unsupported"


def _unsupported_background_limitation(reason: str) -> str:
    """Return the public limitation for an unsupported background route."""

    if reason == BACKGROUND_CODING_ROUTER_INVALID_REASON:
        return BACKGROUND_CODING_ROUTER_INVALID_REASON
    return "Coding agent supervisor rejected the task as unsupported."


def _background_reading_request(
    request: CodingAgentBackgroundRequest,
) -> CodingAgentRequest:
    """Project a background task into the public code-reading contract."""

    reading_request: CodingAgentRequest = {
        "question": _bounded_request_body(request.get("question")),
    }
    _copy_coding_source_fields(request, reading_request)
    max_answer_chars = request.get("max_answer_chars")
    if max_answer_chars is not None:
        reading_request["max_answer_chars"] = max_answer_chars
    preferred_language = request.get("preferred_language")
    if preferred_language:
        reading_request["preferred_language"] = preferred_language
    workspace_root = request.get("workspace_root")
    if workspace_root:
        reading_request["workspace_root"] = workspace_root
    return reading_request


def _background_writing_request(
    request: CodingAgentBackgroundRequest,
) -> CodingAgentWriteRequest:
    """Project a background task into the public code-writing contract."""

    writing_request: CodingAgentWriteRequest = {
        "question": _bounded_request_body(request.get("question")),
    }
    _copy_coding_source_fields(request, writing_request)
    optional_fields = (
        "workspace_root",
        "preferred_language",
        "max_answer_chars",
        "max_artifact_chars",
        "session_id",
    )
    for field_name in optional_fields:
        value = request.get(field_name)
        if value is not None:
            writing_request[field_name] = value
    return writing_request


def _copy_coding_source_fields(
    source: CodingAgentBackgroundRequest,
    target: CodingAgentRequest | CodingAgentWriteRequest,
) -> None:
    """Copy public source fields without interpreting user text."""

    source_fields = (
        "source_url",
        "repo_url",
        "repo_hint",
        "local_root_hint",
        "local_path_hint",
        "requested_ref",
        "source_scope_hint",
    )
    for field_name in source_fields:
        value = source.get(field_name)
        if value is not None:
            target[field_name] = value


def _background_response_from_reading(
    response: CodingAgentResponse,
    *,
    route_reason: str,
) -> CodingAgentBackgroundResponse:
    """Normalize a code-reading answer for background-worker consumption."""

    trace_summary = [
        f"background_coding:code_reading:{_safe_request_text(route_reason)}",
        *response["trace_summary"],
    ]
    result: CodingAgentBackgroundResponse = {
        "status": response["status"],
        "operation": "code_reading",
        "answer_text": response["answer_text"],
        "repository": response["repository"],
        "source_scope": response["source_scope"],
        "evidence": response["evidence"],
        "patch_artifacts": [],
        "created_files": [],
        "changed_files": [],
        "validation": None,
        "limitations": response["limitations"],
        "trace_summary": trace_summary,
    }
    return result


def _background_response_from_writing(
    response: CodingPatchProposalResponse,
    *,
    route_reason: str,
) -> CodingAgentBackgroundResponse:
    """Normalize a code-writing proposal for background-worker consumption."""

    trace_summary = [
        f"background_coding:code_writing:{_safe_request_text(route_reason)}",
        *response["trace_summary"],
    ]
    result: CodingAgentBackgroundResponse = {
        "status": response["status"],
        "operation": "code_writing",
        "answer_text": response["answer_text"],
        "repository": response["repository"],
        "source_scope": response["source_scope"],
        "evidence": response["evidence"],
        "patch_artifacts": response["patch_artifacts"],
        "created_files": response["created_files"],
        "changed_files": response["changed_files"],
        "validation": response["validation"],
        "limitations": response["limitations"],
        "trace_summary": trace_summary,
    }
    return result


def _background_failure_response(
    *,
    status: str,
    operation: CodingAgentBackgroundOperation,
    limitation: str,
    trace_summary: list[str],
) -> CodingAgentBackgroundResponse:
    """Build one normalized background coding failure."""

    if status not in ("failed", "needs_user_input", "rejected"):
        status = "failed"
    result: CodingAgentBackgroundResponse = {
        "status": status,
        "operation": operation,
        "answer_text": "",
        "repository": None,
        "source_scope": None,
        "evidence": [],
        "patch_artifacts": [],
        "created_files": [],
        "changed_files": [],
        "validation": None,
        "limitations": [limitation],
        "trace_summary": trace_summary,
    }
    return result


async def answer_code_question(
    request: CodingAgentRequest,
) -> CodingAgentResponse:
    """Answer a source-code question through fetching and code reading."""

    fetching_result = await code_fetching.run(request)
    if fetching_result["status"] != "succeeded":
        response: CodingAgentResponse = {
            "status": fetching_result["status"],
            "answer_text": "",
            "repository": None,
            "source_scope": None,
            "evidence": [],
            "limitations": fetching_result["limitations"],
            "trace_summary": fetching_result["trace_summary"],
        }
        return response

    repository = fetching_result["repository"]
    source_scope = fetching_result["source_scope"]
    if repository is None or source_scope is None:
        response = {
            "status": "failed",
            "answer_text": "",
            "repository": None,
            "source_scope": None,
            "evidence": [],
            "limitations": [
                *fetching_result["limitations"],
                "Fetching succeeded without repository or source scope.",
            ],
            "trace_summary": fetching_result["trace_summary"],
        }
        return response

    reading_request = {
        "question": request.get("question", ""),
        "repository": repository,
        "source_scope": source_scope,
    }
    preferred_language = request.get("preferred_language")
    if preferred_language is not None:
        reading_request["preferred_language"] = preferred_language
    max_answer_chars = request.get("max_answer_chars")
    if max_answer_chars is not None:
        reading_request["max_answer_chars"] = max_answer_chars

    reading_result = code_reading.run(reading_request)

    limitations = [*fetching_result["limitations"]]
    if repository["dirty_state"] == "dirty":
        limitations.append(DIRTY_CHECKOUT_LIMITATION)
    limitations.extend(reading_result["limitations"])

    response = {
        "status": reading_result["status"],
        "answer_text": reading_result["answer_text"],
        "repository": _repository_summary(repository),
        "source_scope": source_scope,
        "evidence": reading_result["evidence"],
        "limitations": limitations,
        "trace_summary": [
            *fetching_result["trace_summary"],
            *reading_result["trace_summary"],
        ],
    }
    return response


async def propose_code_change(
    request: CodingAgentWriteRequest,
) -> CodingPatchProposalResponse:
    """Produce a public-safe patch proposal without mutating the target repo."""

    workspace_root = request.get("workspace_root")
    if not workspace_root or not _workspace_root_is_valid(workspace_root):
        response = _write_response(
            status="failed",
            mode=_write_mode_from_request(request),
            answer_text="Code writing requires configured proposal storage.",
            repository=None,
            source_scope=None,
            evidence=[],
            patch_artifacts=[],
            created_files=[],
            changed_files=[],
            validation={
                "status": "failed",
                "parsed": False,
                "sandbox_applied": False,
                "errors": ["Missing or invalid proposal storage root."],
                "warnings": [],
                "files": [],
            },
            external_evidence=[],
            session=None,
            limitations=["Missing or invalid proposal storage root."],
            trace_summary=["writing:missing storage"],
        )
        return response

    if not _has_explicit_source(request):
        response = await _propose_new_project_change(request)
        return response

    response = _write_response(
        status="rejected",
        mode="edit_existing_repository",
        answer_text=(
            "This writing stage creates new artifacts only. Existing-source "
            "semantic edits require the code modifying capability."
        ),
        repository=None,
        source_scope=None,
        evidence=[],
        patch_artifacts=[],
        created_files=[],
        changed_files=[],
        validation={
            "status": "rejected",
            "parsed": False,
            "sandbox_applied": False,
            "errors": [
                "Existing-source semantic edits are outside the current "
                "writing scope."
            ],
            "warnings": [],
            "files": [],
        },
        external_evidence=[],
        session=None,
        limitations=[
            "Existing-source semantic edits are outside the current writing scope.",
        ],
        trace_summary=["writing:existing_source_rejected"],
    )
    return response

    fetching_result = await code_fetching.run(request)
    if fetching_result["status"] != "succeeded":
        response = _write_response(
            status=fetching_result["status"],
            mode="edit_existing_repository",
            answer_text="",
            repository=None,
            source_scope=None,
            evidence=[],
            patch_artifacts=[],
            created_files=[],
            changed_files=[],
            validation={
                "status": "failed",
                "parsed": False,
                "sandbox_applied": False,
                "errors": [],
                "warnings": [],
                "files": [],
            },
            external_evidence=[],
            session=None,
            limitations=fetching_result["limitations"],
            trace_summary=fetching_result["trace_summary"],
        )
        return response

    repository = fetching_result["repository"]
    source_scope = fetching_result["source_scope"]
    if repository is None or source_scope is None:
        response = _write_response(
            status="failed",
            mode="edit_existing_repository",
            answer_text="",
            repository=None,
            source_scope=None,
            evidence=[],
            patch_artifacts=[],
            created_files=[],
            changed_files=[],
            validation={
                "status": "failed",
                "parsed": False,
                "sandbox_applied": False,
                "errors": ["Fetching succeeded without source contract."],
                "warnings": [],
                "files": [],
            },
            external_evidence=[],
            session=None,
            limitations=[
                *fetching_result["limitations"],
                "Fetching succeeded without repository or source scope.",
            ],
            trace_summary=fetching_result["trace_summary"],
        )
        return response

    response = await _propose_existing_repo_change(
        request=request,
        repository=repository,
        source_scope=source_scope,
        fetching_limitations=fetching_result["limitations"],
        fetching_trace_summary=fetching_result["trace_summary"],
    )
    return response


async def _propose_new_project_change(
    request: CodingAgentWriteRequest,
) -> CodingPatchProposalResponse:
    ledger = CodingSupervisorWorkLedger(
        goal=str(request.get("question") or ""),
    )
    trace_summary: list[str] = []
    last_writing_result: dict[str, object] | None = None
    for attempt_index in range(WRITE_LOOP_LIMIT):
        writing_request = _writing_request(
            request=request,
            mode="create_new_project",
            repository=None,
            source_scope=None,
            reading_result=None,
            supervisor_evidence_state=ledger.projection(),
            prior_generated_artifacts=ledger.generated_artifacts,
        )
        if ledger.external_evidence:
            writing_request["external_evidence"] = ledger.external_evidence
        if ledger.supervisor_facts:
            writing_request["supervisor_facts"] = ledger.supervisor_facts
        writing_result = await _maybe_await(code_writing.run(writing_request))
        last_writing_result = writing_result
        ledger.record_writing_attempt(
            attempt_index=attempt_index + 1,
            writing_result=writing_result,
        )
        ledger.record_generated_artifacts(
            writing_result.get("pending_artifacts"),
        )
        trace_summary.extend(writing_result["trace_summary"])

        if writing_result["status"] == "need_external_evidence":
            requests = writing_result["external_evidence_requests"]
            if not requests:
                response = _write_loop_failure(
                    mode="create_new_project",
                    repository=None,
                    source_scope=None,
                    evidence=[],
                    external_evidence=ledger.external_evidence,
                    trace_summary=trace_summary,
                    reason="Writing requested external evidence without tasks.",
                    session=writing_result.get("session"),
                )
                return response
            collected_evidence = await collect_external_evidence(
                requests,
                trace_summary=trace_summary,
            )
            ledger.record_external_evidence(collected_evidence)
            continue

        if writing_result["status"] == "need_reading":
            reading_result = _run_generated_readback_for_write(
                request=request,
                writing_result=writing_result,
            )
            trace_summary.extend(reading_result["trace_summary"])
            if (
                reading_result["status"] != "succeeded"
                and not _reading_has_usable_evidence(reading_result)
            ):
                response = _write_loop_failure(
                    mode="create_new_project",
                    repository=None,
                    source_scope=None,
                    evidence=reading_result["evidence"],
                    external_evidence=ledger.external_evidence,
                    trace_summary=trace_summary,
                    reason="Generated artifact readback did not produce usable evidence.",
                    session=writing_result.get("session"),
                )
                return response
            ledger.record_supervisor_fact(
                _supervisor_fact_from_readback(
                    writing_result=writing_result,
                    reading_result=reading_result,
                ),
            )
            trace_summary.append(
                f"generated_readback:evidence={len(reading_result['evidence'])}"
            )
            continue

        response = _write_response_from_result(
            writing_result=writing_result,
            repository=None,
            source_scope=None,
            evidence=[],
            limitations=writing_result["limitations"],
            trace_summary=trace_summary,
        )
        return response

    response = _write_loop_limit_response(
        mode="create_new_project",
        repository=None,
        source_scope=None,
        evidence=[],
        external_evidence=ledger.external_evidence,
        trace_summary=trace_summary,
        last_writing_result=last_writing_result,
    )
    return response


def _run_generated_readback_for_write(
    *,
    request: CodingAgentWriteRequest,
    writing_result: dict[str, object],
) -> dict[str, object]:
    reading_source = writing_result.get("reading_source")
    if not isinstance(reading_source, dict):
        result = _missing_generated_readback_result()
        return result
    repository = reading_source.get("repository")
    source_scope = reading_source.get("source_scope")
    if not isinstance(repository, dict) or not isinstance(source_scope, dict):
        result = _missing_generated_readback_result()
        return result

    reading_request = {
        "question": _generated_readback_question_for_writing_result(
            writing_result,
        ),
        "repository": repository,
        "source_scope": source_scope,
        "read_only_context_handoff": True,
    }
    preferred_language = request.get("preferred_language")
    if preferred_language:
        reading_request["preferred_language"] = preferred_language
    max_answer_chars = request.get("max_answer_chars")
    if max_answer_chars is not None:
        reading_request["max_answer_chars"] = max_answer_chars
    reading_result = code_reading.run(reading_request)
    return reading_result


def _missing_generated_readback_result() -> dict[str, object]:
    result = {
        "status": "failed",
        "answer_text": "",
        "evidence": [],
        "limitations": ["Generated artifact readback source was missing."],
        "trace_summary": ["generated_readback:missing_source"],
    }
    return result


def _supervisor_fact_from_readback(
    *,
    writing_result: dict[str, object],
    reading_result: dict[str, object],
) -> dict[str, object]:
    requests = _safe_reading_request_summaries(
        writing_result.get("reading_requests")
    )
    request_id = "generated_readback"
    task = "Generated artifact readback."
    if requests:
        first_request = requests[0]
        request_id_value = first_request.get("request_id")
        task_value = first_request.get("task")
        if isinstance(request_id_value, str) and request_id_value.strip():
            request_id = request_id_value
        if isinstance(task_value, str) and task_value.strip():
            task = task_value
    limitations = _safe_request_list(reading_result.get("limitations"))
    limitation_text = "; ".join(limitations)
    fact = {
        "request_id": request_id,
        "kind": "generated_artifact_readback",
        "task": task,
        "resolved": reading_result["status"] == "succeeded",
        "result": _readback_fact_result(reading_result),
        "limitation": limitation_text,
    }
    return fact


def _readback_fact_result(reading_result: dict[str, object]) -> str:
    answer_text = str(reading_result.get("answer_text") or "").strip()
    if answer_text:
        return answer_text

    evidence = reading_result.get("evidence")
    if not isinstance(evidence, list):
        return ""
    evidence_paths: list[str] = []
    for row in evidence:
        if not isinstance(row, dict):
            continue
        path = row.get("path")
        if not isinstance(path, str) or path in evidence_paths:
            continue
        evidence_paths.append(path)
    if not evidence_paths:
        return ""
    result = "Evidence was found in: " + ", ".join(evidence_paths[:6])
    return result


async def _propose_existing_repo_change(
    *,
    request: CodingAgentWriteRequest,
    repository: CodeRepositoryRef,
    source_scope: dict[str, object],
    fetching_limitations: list[str],
    fetching_trace_summary: list[str],
) -> CodingPatchProposalResponse:
    reading_result: dict[str, object] | None = None
    external_evidence: list[dict[str, object]] = []
    trace_summary = [*fetching_trace_summary]
    last_writing_result: dict[str, object] | None = None
    reading_attempts: list[dict[str, object]] = []
    completed_reading_requests: list[dict[str, object]] = []
    remaining_reading_attempts = MAX_EXISTING_REPO_FOLLOWUP_READING_ATTEMPTS

    reading_result = _run_initial_reading_for_write(
        request=request,
        repository=repository,
        source_scope=source_scope,
    )
    reading_attempts.append(
        _reading_attempt_summary(
            attempt_kind="initial",
            reading_result=reading_result,
            reading_requests=[],
        )
    )
    trace_summary.extend(reading_result["trace_summary"])
    if (
        reading_result["status"] != "succeeded"
        and not _reading_has_usable_evidence(reading_result)
    ):
        limitations = _existing_repo_limitations(
            fetching_limitations=fetching_limitations,
            repository=repository,
            reading_result=reading_result,
            writing_result=_empty_writing_result_for_reading_failure(),
        )
        response = _write_response(
            status=reading_result["status"],
            mode="edit_existing_repository",
            answer_text=reading_result["answer_text"],
            repository=_repository_summary(repository),
            source_scope=source_scope,
            evidence=reading_result["evidence"],
            patch_artifacts=[],
            created_files=[],
            changed_files=[],
            validation={
                "status": "failed",
                "parsed": False,
                "sandbox_applied": False,
                "errors": ["Initial source reading did not produce usable evidence."],
                "warnings": [],
                "files": [],
            },
            external_evidence=external_evidence,
            session=None,
            limitations=limitations,
            trace_summary=trace_summary,
        )
        return response
    if reading_result["status"] != "succeeded":
        trace_summary.append(
            "reading_partial:"
            f"status={reading_result['status']} "
            f"evidence={len(reading_result['evidence'])}"
        )
    trace_summary.append(f"reading_merge:evidence={len(reading_result['evidence'])}")

    for _ in range(WRITE_LOOP_LIMIT):
        writing_request = _writing_request(
            request=request,
            mode="edit_existing_repository",
            repository=repository,
            source_scope=source_scope,
            reading_result=reading_result,
            supervisor_evidence_state=_supervisor_evidence_state(
                reading_attempts=reading_attempts,
                completed_reading_requests=completed_reading_requests,
                remaining_reading_attempts=remaining_reading_attempts,
                reading_result=reading_result,
            ),
        )
        if external_evidence:
            writing_request["external_evidence"] = external_evidence
        writing_result = await _maybe_await(code_writing.run(writing_request))
        last_writing_result = writing_result
        trace_summary.extend(writing_result["trace_summary"])

        if writing_result["status"] == "need_reading":
            requested_reading = _safe_reading_request_summaries(
                writing_result.get("reading_requests")
            )
            if remaining_reading_attempts <= 0:
                reason = (
                    "Writing PM requested source reading after supervisor "
                    "reading budget was exhausted."
                )
                response = _write_loop_failure(
                    mode="edit_existing_repository",
                    repository=_repository_summary(repository),
                    source_scope=source_scope,
                    evidence=_evidence_from_reading_result(reading_result),
                    external_evidence=external_evidence,
                    trace_summary=[
                        *trace_summary,
                        "reading_budget:exhausted "
                        f"evidence={len(_evidence_from_reading_result(reading_result))}",
                    ],
                    reason=reason,
                    session=writing_result.get("session"),
                    trace=writing_result.get("trace"),
                )
                return response
            next_reading_result = _run_reading_for_write(
                request=request,
                repository=repository,
                source_scope=source_scope,
                writing_result=writing_result,
            )
            remaining_reading_attempts -= 1
            completed_reading_requests.extend(requested_reading)
            reading_attempts.append(
                _reading_attempt_summary(
                    attempt_kind="followup",
                    reading_result=next_reading_result,
                    reading_requests=requested_reading,
                )
            )
            trace_summary.extend(next_reading_result["trace_summary"])
            if (
                next_reading_result["status"] != "succeeded"
                and not _reading_has_usable_evidence(next_reading_result)
            ):
                limitations = _existing_repo_limitations(
                    fetching_limitations=fetching_limitations,
                    repository=repository,
                    reading_result=next_reading_result,
                    writing_result=writing_result,
                )
                response = _write_response(
                    status=next_reading_result["status"],
                    mode="edit_existing_repository",
                    answer_text=next_reading_result["answer_text"],
                    repository=_repository_summary(repository),
                    source_scope=source_scope,
                    evidence=next_reading_result["evidence"],
                    patch_artifacts=[],
                    created_files=[],
                    changed_files=[],
                    validation={
                        "status": "failed",
                        "parsed": False,
                        "sandbox_applied": False,
                        "errors": ["Source reading did not produce usable evidence."],
                        "warnings": [],
                        "files": [],
                    },
                    external_evidence=external_evidence,
                    session=writing_result.get("session"),
                    limitations=limitations,
                    trace_summary=trace_summary,
                )
                return response
            if next_reading_result["status"] != "succeeded":
                trace_summary.append(
                    "reading_partial:"
                    f"status={next_reading_result['status']} "
                    f"evidence={len(next_reading_result['evidence'])}"
                )
            reading_result = _merge_reading_results(
                current_result=reading_result,
                next_result=next_reading_result,
            )
            trace_summary.append(
                f"reading_merge:evidence={len(reading_result['evidence'])}"
            )
            continue

        if writing_result["status"] == "need_external_evidence":
            requests = writing_result["external_evidence_requests"]
            if not requests:
                response = _write_loop_failure(
                    mode="edit_existing_repository",
                    repository=_repository_summary(repository),
                    source_scope=source_scope,
                    evidence=_evidence_from_reading_result(reading_result),
                    external_evidence=external_evidence,
                    trace_summary=trace_summary,
                    reason="Writing requested external evidence without tasks.",
                    session=writing_result.get("session"),
                )
                return response
            collected_evidence = await collect_external_evidence(
                requests,
                trace_summary=trace_summary,
            )
            external_evidence.extend(collected_evidence)
            continue

        limitations = _existing_repo_limitations(
            fetching_limitations=fetching_limitations,
            repository=repository,
            reading_result=reading_result,
            writing_result=writing_result,
        )
        response = _write_response_from_result(
            writing_result=writing_result,
            repository=_repository_summary(repository),
            source_scope=source_scope,
            evidence=_evidence_from_reading_result(reading_result),
            limitations=limitations,
            trace_summary=trace_summary,
        )
        return response

    response = _write_loop_limit_response(
        mode="edit_existing_repository",
        repository=_repository_summary(repository),
        source_scope=source_scope,
        evidence=_evidence_from_reading_result(reading_result),
        external_evidence=external_evidence,
        trace_summary=trace_summary,
        last_writing_result=last_writing_result,
    )
    return response


def _run_initial_reading_for_write(
    *,
    request: CodingAgentWriteRequest,
    repository: CodeRepositoryRef,
    source_scope: dict[str, object],
) -> dict[str, object]:
    reading_request = {
        "question": _initial_reading_question_for_write_request(request),
        "repository": repository,
        "source_scope": source_scope,
        "read_only_context_handoff": True,
    }
    preferred_language = request.get("preferred_language")
    if preferred_language is not None:
        reading_request["preferred_language"] = preferred_language
    max_answer_chars = request.get("max_answer_chars")
    if max_answer_chars is not None:
        reading_request["max_answer_chars"] = max_answer_chars

    reading_result = code_reading.run(reading_request)
    return reading_result


def _run_reading_for_write(
    *,
    request: CodingAgentWriteRequest,
    repository: CodeRepositoryRef,
    source_scope: dict[str, object],
    writing_result: dict[str, object],
) -> dict[str, object]:
    reading_request = {
        "question": _reading_question_for_writing_result(writing_result),
        "repository": repository,
        "source_scope": source_scope,
        "read_only_context_handoff": True,
    }
    preferred_language = request.get("preferred_language")
    if preferred_language is not None:
        reading_request["preferred_language"] = preferred_language
    max_answer_chars = request.get("max_answer_chars")
    if max_answer_chars is not None:
        reading_request["max_answer_chars"] = max_answer_chars

    reading_result = code_reading.run(reading_request)
    return reading_result


def _empty_writing_result_for_reading_failure() -> dict[str, object]:
    writing_result: dict[str, object] = {
        "limitations": ["Source evidence is required before patch proposal."],
    }
    return writing_result


def _existing_repo_limitations(
    *,
    fetching_limitations: list[str],
    repository: CodeRepositoryRef,
    reading_result: dict[str, object] | None,
    writing_result: dict[str, object],
) -> list[str]:
    limitations = [*fetching_limitations]
    if repository["dirty_state"] == "dirty":
        limitations.append(DIRTY_CHECKOUT_LIMITATION)
    if reading_result is not None:
        limitations.extend(reading_result["limitations"])
    limitations.extend(writing_result["limitations"])
    return limitations


def _evidence_from_reading_result(
    reading_result: dict[str, object] | None,
) -> list[dict[str, object]]:
    if reading_result is None:
        return []
    evidence = reading_result["evidence"]
    return evidence


def _reading_has_usable_evidence(reading_result: dict[str, object]) -> bool:
    evidence = reading_result.get("evidence")
    return isinstance(evidence, list) and bool(evidence)


def _merge_reading_results(
    *,
    current_result: dict[str, object] | None,
    next_result: dict[str, object],
) -> dict[str, object]:
    if current_result is None:
        return next_result

    evidence_rows: list[dict[str, object]] = []
    seen_evidence: set[tuple[object, object, object, object]] = set()
    for result in (current_result, next_result):
        for row in result["evidence"]:
            key = (
                row.get("path"),
                row.get("line_start"),
                row.get("line_end"),
                row.get("excerpt"),
            )
            if key in seen_evidence:
                continue
            seen_evidence.add(key)
            evidence_rows.append(row)

    limitations: list[str] = []
    for result in (current_result, next_result):
        for limitation in result["limitations"]:
            if limitation not in limitations:
                limitations.append(limitation)

    answer_texts = [
        str(current_result["answer_text"]),
        str(next_result["answer_text"]),
    ]
    if answer_texts[0] == answer_texts[1]:
        answer_text = answer_texts[0]
    else:
        answer_text = "\n\nFollow-up reading:\n".join(answer_texts)

    merged_result = {
        "status": next_result["status"],
        "answer_text": answer_text,
        "evidence": evidence_rows,
        "limitations": limitations,
        "trace_summary": [
            *current_result["trace_summary"],
            *next_result["trace_summary"],
        ],
    }
    return merged_result


def _write_loop_failure(
    *,
    mode: str,
    repository: CodingAgentRepositorySummary | None,
    source_scope: dict[str, object] | None,
    evidence: list[dict[str, object]],
    external_evidence: list[dict[str, object]],
    trace_summary: list[str],
    reason: str,
    session: object,
    trace: object | None = None,
) -> CodingPatchProposalResponse:
    response = _write_response(
        status="failed",
        mode=mode,
        answer_text=reason,
        repository=repository,
        source_scope=source_scope,
        evidence=evidence,
        patch_artifacts=[],
        created_files=[],
        changed_files=[],
        validation={
            "status": "failed",
            "parsed": False,
            "sandbox_applied": False,
            "errors": [reason],
            "warnings": [],
            "files": [],
        },
        external_evidence=external_evidence,
        session=session,
        limitations=[reason],
        trace_summary=[*trace_summary, "writing_loop:failed"],
        trace=trace,
    )
    return response


def _write_loop_limit_response(
    *,
    mode: str,
    repository: CodingAgentRepositorySummary | None,
    source_scope: dict[str, object] | None,
    evidence: list[dict[str, object]],
    external_evidence: list[dict[str, object]],
    trace_summary: list[str],
    last_writing_result: dict[str, object] | None,
) -> CodingPatchProposalResponse:
    reason = "Code writing loop reached its limit before a final patch proposal."
    session = None
    if last_writing_result is not None:
        session = last_writing_result["session"]
    response = _write_loop_failure(
        mode=mode,
        repository=repository,
        source_scope=source_scope,
        evidence=evidence,
        external_evidence=external_evidence,
        trace_summary=trace_summary,
        reason=reason,
        session=session,
    )
    return response


def _repository_summary(
    repository: CodeRepositoryRef,
) -> CodingAgentRepositorySummary:
    summary: CodingAgentRepositorySummary = {
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


def _workspace_root_is_valid(workspace_root: str) -> bool:
    root = Path(workspace_root).expanduser().resolve(strict=False)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    if not root.is_dir():
        return False
    return True


def _has_explicit_source(request: CodingAgentWriteRequest) -> bool:
    source_fields = (
        "source_url",
        "repo_url",
        "repo_hint",
        "local_root_hint",
        "local_path_hint",
    )
    has_source = any(bool(request.get(field)) for field in source_fields)
    return has_source


def _write_mode_from_request(request: CodingAgentWriteRequest) -> str:
    if _has_explicit_source(request):
        return "edit_existing_repository"
    return "create_new_project"


def _writing_request(
    *,
    request: CodingAgentWriteRequest,
    mode: str,
    repository: dict[str, object] | None,
    source_scope: dict[str, object] | None,
    reading_result: dict[str, object] | None,
    supervisor_evidence_state: dict[str, object] | None = None,
    prior_generated_artifacts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    writing_request: dict[str, object] = {
        "question": request.get("question", ""),
        "mode_hint": mode,
        "repository": repository,
        "source_scope": source_scope,
        "reading_result": reading_result,
        "workspace_root": request["workspace_root"],
    }
    if supervisor_evidence_state is not None:
        writing_request["supervisor_evidence_state"] = supervisor_evidence_state
    if prior_generated_artifacts:
        writing_request["prior_generated_artifacts"] = prior_generated_artifacts
    optional_fields = (
        "preferred_language",
        "session_id",
        "max_answer_chars",
        "max_artifact_chars",
    )
    for field in optional_fields:
        value = request.get(field)
        if value is not None:
            writing_request[field] = value
    return writing_request


def _reading_question_for_writing_result(
    writing_result: dict[str, object],
) -> str:
    pm_evidence_request = _pm_evidence_request_text(writing_result)
    reading_question = (
        "Read-only repository evidence survey for future implementation "
        "workflow. Identify existing files, symbols, contracts, validation "
        "paths, documentation entry points, tests, and behavior boundaries "
        "needed by the writing PM. For each requested runtime behavior, find "
        "the code owner that currently enforces similar behavior or report "
        "that no owner was found. Include tests or test patterns that verify "
        "behavior when they are visible. Documentation entry points are "
        "supporting evidence, not proof of runtime behavior. For "
        "error-reporting requests, identify existing raise sites, error "
        "message branches, exception handlers, and import locations. When "
        "the request adds reporting, counters, summaries, routing, labels, "
        "or other stateful dimensions, identify whether the current runtime "
        "method receives that dimension and include caller/import sites "
        "needed for a limited interface update. Do not create or describe "
        "implementation changes; only report current evidence.\n\n"
        f"Writing PM evidence request:\n{pm_evidence_request}"
    )
    return reading_question


def _generated_readback_question_for_writing_result(
    writing_result: dict[str, object],
) -> str:
    pm_evidence_request = _pm_evidence_request_text(writing_result)
    reading_question = (
        "Read-only generated-artifact readback for a code-writing workflow. "
        "Inspect only the provided generated artifacts. Report the observable "
        "interfaces, file formats, output shapes, literals, imports, error "
        "surfaces, and behavior details needed before later generated "
        "artifacts such as tests, documentation, or command wrappers consume "
        "this work. Do not propose implementation changes, do not run code, "
        "and do not describe command results. If a requested fact is absent "
        "from the generated artifacts, state that it is absent.\n\n"
        f"Writing PM readback request:\n{pm_evidence_request}"
    )
    return reading_question


def _initial_reading_question_for_write_request(
    request: CodingAgentWriteRequest,
) -> str:
    user_request = _bounded_request_body(request.get("question"))
    reading_question = (
        "Read-only repository evidence survey for a limited patch proposal. "
        "Use the current user request as the requirements source. Identify "
        "existing files, symbols, contracts, validation paths, documentation "
        "entry points, tests, and behavior boundaries needed before a writing "
        "PM assigns implementation work. For each requested runtime behavior, "
        "find the code owner that currently enforces similar behavior or "
        "report that no owner was found. Include tests or test patterns that "
        "verify behavior when they are visible. Documentation entry points are "
        "supporting evidence, not proof of runtime behavior. When the request "
        "adds reporting, counters, summaries, routing, labels, or other "
        "stateful dimensions, identify whether the current runtime method "
        "receives that dimension and include caller/import sites needed for a "
        "limited interface update. Do not create or describe implementation "
        "changes; only report current evidence.\n\n"
        f"User request:\n{user_request}"
    )
    return reading_question


def _pm_evidence_request_text(writing_result: dict[str, object]) -> str:
    reading_requests = writing_result.get("reading_requests")
    if isinstance(reading_requests, list) and reading_requests:
        blocks = []
        for index, request in enumerate(reading_requests, start=1):
            if not isinstance(request, dict):
                continue
            task = _safe_request_text(request.get("task"))
            reason = _safe_request_text(request.get("reason"))
            slots = _safe_request_list(request.get("required_slots"))
            targets = _safe_request_list(request.get("target_artifacts"))
            block_lines = [f"Request {index}:"]
            if task:
                block_lines.append(f"Task: {task}")
            if reason:
                block_lines.append(f"Reason: {reason}")
            if slots:
                block_lines.append("Required evidence slots:")
                block_lines.extend(f"- {slot}" for slot in slots)
            if targets:
                block_lines.append("Target artifacts:")
                block_lines.extend(f"- {target}" for target in targets)
            blocks.append("\n".join(block_lines))
        if blocks:
            return "\n\n".join(blocks)

    limitations = _safe_request_list(writing_result.get("limitations"))
    if limitations:
        return (
            "Collect current repository evidence for these PM-declared "
            "missing source facts:\n"
        + "\n".join(f"- {slot}" for slot in limitations)
    )
    return (
        "Collect current repository evidence needed before limited "
        "implementation planning."
    )


def _supervisor_evidence_state(
    *,
    reading_attempts: list[dict[str, object]],
    completed_reading_requests: list[dict[str, object]],
    remaining_reading_attempts: int,
    reading_result: dict[str, object] | None,
) -> dict[str, object]:
    evidence_count = 0
    last_status = "not_started"
    last_limitations: list[str] = []
    if reading_result is not None:
        evidence_count = len(_evidence_from_reading_result(reading_result))
        status = reading_result.get("status")
        if isinstance(status, str):
            last_status = status
        last_limitations = _safe_request_list(reading_result.get("limitations"))

    evidence_state = {
        "reading_attempts": reading_attempts,
        "completed_reading_requests": completed_reading_requests[-6:],
        "remaining_reading_attempts": max(remaining_reading_attempts, 0),
        "merged_reading_evidence_count": evidence_count,
        "last_reading_status": last_status,
        "last_reading_limitations": last_limitations[:6],
    }
    return evidence_state


def _reading_attempt_summary(
    *,
    attempt_kind: str,
    reading_result: dict[str, object],
    reading_requests: list[dict[str, object]],
) -> dict[str, object]:
    summary = {
        "kind": attempt_kind,
        "status": _safe_request_text(reading_result.get("status")),
        "evidence_count": len(_evidence_from_reading_result(reading_result)),
        "requests": reading_requests[:3],
        "limitations": _safe_request_list(reading_result.get("limitations"))[:4],
    }
    return summary


def _safe_reading_request_summaries(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    summaries: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        summary = {
            "request_id": _safe_request_text(item.get("request_id")),
            "task": _safe_request_text(item.get("task")),
            "reason": _safe_request_text(item.get("reason")),
            "required_slots": _safe_request_list(item.get("required_slots"))[:8],
        }
        summaries.append(summary)
        if len(summaries) >= 6:
            break
    return summaries


def _bounded_request_body(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return text[:20000].rstrip()


def _safe_request_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    texts: list[str] = []
    for item in value:
        text = _safe_request_text(item)
        if text:
            texts.append(text)
    return texts


def _safe_request_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.strip().split())
    return text[:600].rstrip()


async def _maybe_await(value):
    if inspect.isawaitable(value):
        awaited_value = await value
        return awaited_value
    return value


def _write_response_from_result(
    *,
    writing_result: dict[str, object],
    repository: CodingAgentRepositorySummary | None,
    source_scope: dict[str, object] | None,
    evidence: list[dict[str, object]],
    limitations: list[str],
    trace_summary: list[str],
) -> CodingPatchProposalResponse:
    response = _write_response(
        status=writing_result["status"],
        mode=writing_result["mode"],
        answer_text=writing_result["answer_text"],
        repository=repository,
        source_scope=source_scope,
        evidence=evidence,
        patch_artifacts=writing_result["patch_artifacts"],
        created_files=writing_result["created_files"],
        changed_files=writing_result.get("changed_files", []),
        validation=writing_result["validation"],
        alignment=writing_result.get("alignment"),
        external_evidence=writing_result.get("external_evidence", []),
        session=writing_result.get("session"),
        limitations=limitations,
        trace_summary=trace_summary,
        trace=writing_result.get("trace"),
    )
    return response


def _write_response(
    *,
    status: str,
    mode: str,
    answer_text: str,
    repository: CodingAgentRepositorySummary | None,
    source_scope: dict[str, object] | None,
    evidence: list[dict[str, object]],
    patch_artifacts: list[dict[str, object]],
    created_files: list[dict[str, str]],
    changed_files: list[dict[str, str]],
    validation: dict[str, object],
    external_evidence: list[dict[str, object]],
    session: dict[str, object] | None,
    limitations: list[str],
    trace_summary: list[str],
    trace: object | None = None,
    alignment: dict[str, object] | None = None,
) -> CodingPatchProposalResponse:
    answer_text = _answer_with_required_limitations(
        answer_text,
        limitations,
        max_answer_chars=DEFAULT_MAX_ANSWER_CHARS,
    )
    response: CodingPatchProposalResponse = {
        "status": status,
        "mode": mode,
        "answer_text": answer_text,
        "repository": repository,
        "source_scope": source_scope,
        "evidence": _public_write_evidence(evidence),
        "patch_artifacts": patch_artifacts,
        "created_files": created_files,
        "changed_files": changed_files,
        "validation": validation,
        "external_evidence": external_evidence,
        "session": session,
        "limitations": limitations,
        "trace_summary": trace_summary,
    }
    if alignment is not None:
        response["alignment"] = alignment
    if isinstance(trace, dict):
        response["trace"] = trace
    response = _sanitize_write_response(response)
    return response


def _public_write_evidence(
    evidence: list[dict[str, object]],
) -> list[dict[str, object]]:
    public_rows: list[dict[str, object]] = []
    for row in evidence:
        public_rows.append({
            "path": str(row.get("path", "")),
            "line_start": _public_int(row.get("line_start")),
            "line_end": _public_int(row.get("line_end")),
            "symbol_or_topic": "source evidence",
            "excerpt": WRITE_EVIDENCE_EXCERPT_OMITTED,
            "reason": "Selected by the read-only evidence survey.",
        })
    return public_rows


def _public_int(value: object) -> int:
    if isinstance(value, int):
        return value
    return 0


def _sanitize_write_response(
    response: CodingPatchProposalResponse,
) -> CodingPatchProposalResponse:
    sanitized = _sanitize_public_value(response)
    return sanitized


def _sanitize_public_value(value):
    if isinstance(value, str):
        text = GIT_INTERNAL_TOKEN_RE.sub("[git-internal]", value)
        text = ENV_FILE_TOKEN_RE.sub("[environment-file]", text)
        text = CACHE_KEY_TOKEN_RE.sub("[cache-key]", text)
        return text
    if isinstance(value, list):
        return [_sanitize_public_value(item) for item in value]
    if isinstance(value, dict):
        return {
            _sanitize_public_key(key): _sanitize_public_value(item)
            for key, item in value.items()
        }
    return value


def _sanitize_public_key(key):
    if not isinstance(key, str):
        return key
    return CACHE_KEY_TOKEN_RE.sub("[cache-key]", key)
