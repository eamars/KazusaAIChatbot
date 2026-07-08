"""Top-level standalone coding-agent supervisor."""

import inspect
import json
import re
from pathlib import Path
from pathlib import PurePosixPath

from langchain_core.messages import HumanMessage, SystemMessage

import kazusa_ai_chatbot.coding_agent.code_fetching as code_fetching
import kazusa_ai_chatbot.coding_agent.code_modifying as code_modifying
import kazusa_ai_chatbot.coding_agent.code_reading as code_reading
import kazusa_ai_chatbot.coding_agent.code_writing as code_writing
from kazusa_ai_chatbot.coding_agent.code_modifying.models import (
    artifact_to_patch_operation,
)
from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
    compile_patch_operations,
)
from kazusa_ai_chatbot.coding_agent.code_patching.patch_validation import (
    materialize_patch_artifacts_for_review,
)
from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
    list_scoped_safe_files,
)
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
MAX_WRITE_FALLBACK_EVIDENCE_FILES = 10
MAX_WRITE_FALLBACK_EXCERPT_CHARS = 1200
REPAIR_CONTEXT_CALLER_STEMS = {"api", "app", "cli", "main", "routes", "views"}
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
- code_modifying: propose reviewable existing-source patch artifacts when the
  task includes an explicit source structure.
- unsupported: the task is not a coding task, or it requires live execution,
  deployment, credential access, package installation, adapter delivery, or
  real-world mutation as the primary result.

Return strict JSON:
{
  "operation": "code_reading | code_writing | code_modifying | unsupported",
  "reason": "short reason"
}
'''
WRITE_LOOP_LIMIT = 6
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
    if operation in ("code_writing", "code_modifying"):
        writing_response = await propose_code_change(
            _background_writing_request(request),
        )
        response = _background_response_from_writing(
            writing_response,
            operation=operation,
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


async def decide_background_coding_operation(
    request: CodingAgentBackgroundRequest,
) -> tuple[CodingAgentBackgroundOperation, str]:
    """Choose the durable coding-run objective for accepted coding work."""

    operation, reason = await _decide_background_coding_operation(request)
    return operation, reason


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
            "code_modifying",
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
    return operation in (
        "code_reading",
        "code_writing",
        "code_modifying",
        "unsupported",
    )


def _normalize_background_coding_operation(
    parsed: object,
) -> CodingAgentBackgroundOperation:
    """Validate the supervisor route decision without semantic fallback."""

    if not isinstance(parsed, dict):
        return "unsupported"
    operation = parsed.get("operation")
    if operation in (
        "code_reading",
        "code_writing",
        "code_modifying",
        "unsupported",
    ):
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
    operation: CodingAgentBackgroundOperation,
    route_reason: str,
) -> CodingAgentBackgroundResponse:
    """Normalize a code-writing proposal for background-worker consumption."""

    trace_summary = [
        f"background_coding:{operation}:{_safe_request_text(route_reason)}",
        *response["trace_summary"],
    ]
    result: CodingAgentBackgroundResponse = {
        "status": response["status"],
        "operation": operation,
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
    trace_summary = [*fetching_trace_summary]

    reading_result = _run_initial_reading_for_write(
        request=request,
        repository=repository,
        source_scope=source_scope,
    )
    trace_summary.extend(reading_result["trace_summary"])
    if not _reading_has_usable_evidence(reading_result):
        fallback_reading_result = _fallback_reading_result_for_write(
            request=request,
            repository=repository,
            source_scope=source_scope,
            prior_reading_result=reading_result,
        )
        if _reading_has_usable_evidence(fallback_reading_result):
            reading_result = fallback_reading_result
            trace_summary.extend(fallback_reading_result["trace_summary"])
    else:
        reading_result, repair_trace = _repair_supplemented_reading_result(
            request=request,
            repository=repository,
            source_scope=source_scope,
            reading_result=reading_result,
        )
        trace_summary.extend(repair_trace)
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
            external_evidence=[],
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

    modifying_request = _modifying_request(
        request=request,
        repository=repository,
        source_scope=source_scope,
        reading_result=reading_result,
    )
    modifying_result = await _maybe_await(code_modifying.run(modifying_request))
    trace_summary.extend(modifying_result["trace_summary"])
    limitations = _existing_repo_limitations(
        fetching_limitations=fetching_limitations,
        repository=repository,
        reading_result=reading_result,
        writing_result=modifying_result,
    )
    response = _write_response_from_modifying_result(
        request=request,
        modifying_result=modifying_result,
        repository=repository,
        source_scope=source_scope,
        evidence=_evidence_from_reading_result(reading_result),
        limitations=limitations,
        trace_summary=trace_summary,
    )
    if _write_response_needs_modifying_repair(response):
        repair_request = _modifying_request(
            request=request,
            repository=repository,
            source_scope=source_scope,
            reading_result=reading_result,
        )
        repair_request["repair_feedback"] = {
            "validation": response["validation"],
            "previous_modification_artifacts": modifying_result.get(
                "modification_artifacts",
                [],
            ),
            "instruction": (
                "Return a corrected complete artifact list that fixes the "
                "validation errors. Do not repeat invalid syntax, missing "
                "imports, unsafe paths, or no-op artifacts."
            ),
        }
        trace_summary.append("modifying:repair_retry")
        modifying_result = await _maybe_await(code_modifying.run(repair_request))
        trace_summary.extend(modifying_result["trace_summary"])
        limitations = _existing_repo_limitations(
            fetching_limitations=fetching_limitations,
            repository=repository,
            reading_result=reading_result,
            writing_result=modifying_result,
        )
        response = _write_response_from_modifying_result(
            request=request,
            modifying_result=modifying_result,
            repository=repository,
            source_scope=source_scope,
            evidence=_evidence_from_reading_result(reading_result),
            limitations=limitations,
            trace_summary=trace_summary,
        )
    return response


def _write_response_needs_modifying_repair(
    response: CodingPatchProposalResponse,
) -> bool:
    if response["status"] != "failed":
        return False
    validation = response.get("validation")
    if not isinstance(validation, dict):
        return False
    errors = validation.get("errors")
    return isinstance(errors, list) and bool(errors)


def _modifying_request(
    *,
    request: CodingAgentWriteRequest,
    repository: CodeRepositoryRef,
    source_scope: dict[str, object],
    reading_result: dict[str, object],
) -> dict[str, object]:
    modifying_request: dict[str, object] = {
        "question": request.get("question", ""),
        "reading_result": reading_result,
        "repository": repository,
        "source_scope": source_scope,
        "workspace_root": request["workspace_root"],
        "supervisor_facts": [],
    }
    optional_fields = (
        "preferred_language",
        "max_answer_chars",
        "max_artifact_chars",
        "repair_feedback",
    )
    for field in optional_fields:
        value = request.get(field)
        if value is not None:
            modifying_request[field] = value
    return modifying_request


def _write_response_from_modifying_result(
    *,
    request: CodingAgentWriteRequest,
    modifying_result: dict[str, object],
    repository: CodeRepositoryRef,
    source_scope: dict[str, object],
    evidence: list[dict[str, object]],
    limitations: list[str],
    trace_summary: list[str],
) -> CodingPatchProposalResponse:
    max_artifact_chars = _max_artifact_chars_from_request(request)
    repo_root = Path(repository["local_root"])
    workspace_root = Path(str(request["workspace_root"]))
    validation = {
        "status": "failed",
        "parsed": False,
        "sandbox_applied": False,
        "errors": [],
        "warnings": [],
        "files": [],
    }
    patch_artifacts = []
    created_files = []
    changed_files = []

    if modifying_result["status"] == "succeeded":
        patch_operations = _patch_operations_from_modifying_result(
            modifying_result,
        )
        patch_artifacts, created_files, changed_files, operation_errors = (
            compile_patch_operations(
                repo_root=repo_root,
                patch_operations=patch_operations,
                max_files=32,
                max_diff_chars=max_artifact_chars,
            )
        )
        if operation_errors:
            validation["errors"] = operation_errors
            trace_summary.append(
                f"patch_operations:failed errors={len(operation_errors)}"
            )
        else:
            validation = materialize_patch_artifacts_for_review(
                repo_root=repo_root,
                workspace_root=workspace_root,
                patch_artifacts=patch_artifacts,
                max_files=32,
                max_diff_chars=max_artifact_chars,
            )
            trace_summary.append(f"patch_validation:{validation['status']}")

    response_status = str(modifying_result["status"])
    if modifying_result["status"] == "succeeded":
        response_status = validation["status"]

    response = _write_response(
        status=response_status,
        mode="edit_existing_repository",
        answer_text=str(modifying_result.get("answer_text") or ""),
        repository=_repository_summary(repository),
        source_scope=source_scope,
        evidence=evidence,
        patch_artifacts=patch_artifacts,
        created_files=_result_file_summaries(
            modifying_result.get("created_files"),
            created_files,
        ),
        changed_files=_result_file_summaries(
            modifying_result.get("changed_files"),
            changed_files,
        ),
        validation=validation,
        external_evidence=[],
        session=None,
        limitations=limitations,
        trace_summary=trace_summary,
        trace=modifying_result.get("trace"),
    )
    return response


def _patch_operations_from_modifying_result(
    modifying_result: dict[str, object],
) -> list[dict[str, object]]:
    operations: list[dict[str, object]] = []
    artifacts = modifying_result.get("modification_artifacts")
    if not isinstance(artifacts, list):
        return operations
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        operation = artifact_to_patch_operation(artifact)
        if operation is None:
            continue
        operations.append(operation)
    return operations


def _result_file_summaries(
    value: object,
    fallback: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        return fallback
    summaries: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        summary: dict[str, str] = {}
        for key, item_value in item.items():
            if not isinstance(key, str) or not isinstance(item_value, str):
                continue
            summary[key] = item_value
        if summary:
            summaries.append(summary)
    if not summaries:
        return fallback
    return summaries


def _max_artifact_chars_from_request(request: CodingAgentWriteRequest) -> int:
    value = request.get("max_artifact_chars")
    if isinstance(value, int) and value > 0:
        return value
    return 64000


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


def _fallback_reading_result_for_write(
    *,
    request: CodingAgentWriteRequest,
    repository: CodeRepositoryRef,
    source_scope: dict[str, object],
    prior_reading_result: dict[str, object],
) -> dict[str, object]:
    """Build bounded source evidence for source-backed patch proposals."""

    repo_root = Path(repository["local_root"]).expanduser().resolve(strict=True)
    safe_files = list_scoped_safe_files(
        repo_root=repo_root,
        source_scope=source_scope,  # type: ignore[arg-type]
    )
    ranked_paths = _rank_fallback_write_paths(
        safe_files=safe_files,
        question=_fallback_search_text_for_write_request(request),
        priority_paths=_repair_context_priority_paths(
            request=request,
            safe_files=safe_files,
        ),
    )
    evidence: list[dict[str, object]] = []
    repair_feedback = _structured_repair_feedback(request)
    for safe_path in ranked_paths[:MAX_WRITE_FALLBACK_EVIDENCE_FILES]:
        file_path = repo_root / safe_path
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        excerpt = _bounded_fallback_excerpt(content)
        if not excerpt:
            continue
        evidence.append({
            "path": safe_path,
            "line_start": 1,
            "line_end": _fallback_line_count(excerpt),
            "symbol_or_topic": "write evidence fallback",
            "excerpt": excerpt,
            "reason": _fallback_evidence_reason(repair_feedback),
        })
    if not evidence:
        result = dict(prior_reading_result)
        result["trace_summary"] = ["reading_fallback:evidence=0"]
        return result
    limitations = list(prior_reading_result.get("limitations") or [])
    limitations.append(_fallback_limitation(repair_feedback))
    result = {
        "status": "succeeded",
        "answer_text": (
            "Deterministic bounded source evidence fallback selected safe "
            "source, test, and documentation files for the patch proposal."
        ),
        "evidence": evidence,
        "limitations": limitations,
        "trace_summary": [f"reading_fallback:evidence={len(evidence)}"],
    }
    return result


def _repair_supplemented_reading_result(
    *,
    request: CodingAgentWriteRequest,
    repository: CodeRepositoryRef,
    source_scope: dict[str, object],
    reading_result: dict[str, object],
) -> tuple[dict[str, object], list[str]]:
    """Add bounded repair context when structured verification feedback exists."""

    if _structured_repair_feedback(request) is None:
        return reading_result, []

    fallback_result = _fallback_reading_result_for_write(
        request=request,
        repository=repository,
        source_scope=source_scope,
        prior_reading_result=reading_result,
    )
    fallback_evidence = fallback_result.get("evidence")
    if not isinstance(fallback_evidence, list) or not fallback_evidence:
        return reading_result, ["reading_repair_supplement:evidence=0"]

    original_evidence = reading_result.get("evidence")
    if not isinstance(original_evidence, list):
        original_evidence = []
    merged_evidence = _merged_evidence_rows(
        primary=fallback_evidence,
        secondary=original_evidence,
    )
    supplemented_result = dict(reading_result)
    supplemented_result["evidence"] = merged_evidence
    supplemented_result["limitations"] = _merged_texts(
        reading_result.get("limitations"),
        fallback_result.get("limitations"),
    )
    trace = [
        *list(fallback_result.get("trace_summary") or []),
        f"reading_repair_supplement:evidence={len(merged_evidence)}",
    ]
    return supplemented_result, trace


def _merged_evidence_rows(
    *,
    primary: list[object],
    secondary: list[object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for row in [*primary, *secondary]:
        if not isinstance(row, dict):
            continue
        path = row.get("path")
        if not isinstance(path, str):
            continue
        if path in seen_paths:
            continue
        rows.append(dict(row))
        seen_paths.add(path)
    return rows


def _merged_texts(*values: object) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, list):
            continue
        for item in value:
            text = _safe_request_text(item)
            if not text or text in seen:
                continue
            texts.append(text)
            seen.add(text)
    return texts


def _fallback_evidence_reason(
    repair_feedback: dict[str, object] | None,
) -> str:
    if repair_feedback is not None:
        return (
            "Deterministic fallback selected this safe source file from "
            "structured repair feedback for a repair patch proposal."
        )
    return (
        "Deterministic fallback selected this safe source file for "
        "a source-backed patch proposal."
    )


def _fallback_limitation(repair_feedback: dict[str, object] | None) -> str:
    if repair_feedback is not None:
        return (
            "Structured repair feedback triggered deterministic bounded "
            "source evidence fallback for the repair proposal."
        )
    return (
        "Initial reading PM returned no evidence; deterministic bounded "
        "source evidence fallback was used for the patch proposal."
    )


def _rank_fallback_write_paths(
    *,
    safe_files: list[str],
    question: str,
    priority_paths: list[str] | None = None,
) -> list[str]:
    terms = _fallback_terms(question)
    priority_order = {
        path: index
        for index, path in enumerate(priority_paths or [])
    }
    ranked = sorted(
        safe_files,
        key=lambda path: (
            0 if path in priority_order else 1,
            priority_order.get(path, len(priority_order)),
            -_fallback_path_score(path, terms),
            _fallback_path_rank(path),
            path.casefold(),
        ),
    )
    return ranked


def _fallback_search_text_for_write_request(
    request: CodingAgentWriteRequest,
) -> str:
    parts = [str(request.get("question") or "")]
    repair_feedback = _structured_repair_feedback(request)
    if repair_feedback is not None:
        parts.append(_repair_feedback_reading_block(repair_feedback))
    return "\n".join(part for part in parts if part)


def _repair_context_priority_paths(
    *,
    request: CodingAgentWriteRequest,
    safe_files: list[str],
) -> list[str]:
    repair_feedback = _structured_repair_feedback(request)
    if repair_feedback is None:
        return []

    safe_file_set = set(safe_files)
    priority_paths: list[str] = []

    def add_path(path: str) -> None:
        if path not in safe_file_set or path in priority_paths:
            return
        priority_paths.append(path)

    required_paths = _safe_repair_feedback_paths(
        repair_feedback,
        "required_source_owner_paths",
    )
    previous_paths = _repair_feedback_patch_artifact_paths(
        repair_feedback.get("previous_patch_artifacts"),
    )
    protected_paths = _safe_repair_feedback_paths(
        repair_feedback,
        "protected_verification_paths",
    )
    failed_paths = _safe_repair_feedback_paths(repair_feedback, "failed_paths")

    for path in required_paths:
        add_path(path)
    for path in previous_paths:
        add_path(path)
    for path in _caller_collaborator_paths(
        seed_paths=[*required_paths, *previous_paths],
        safe_files=safe_files,
    ):
        add_path(path)
    for path in _tested_source_paths(
        test_paths=[*protected_paths, *failed_paths],
        safe_files=safe_files,
    ):
        add_path(path)
    for path in protected_paths:
        add_path(path)
    for path in failed_paths:
        add_path(path)
    return priority_paths


def _caller_collaborator_paths(
    *,
    seed_paths: list[str],
    safe_files: list[str],
) -> list[str]:
    seed_dirs = {
        PurePosixPath(path).parent.as_posix()
        for path in seed_paths
        if path
    }
    collaborators: list[str] = []
    for safe_path in safe_files:
        path = PurePosixPath(safe_path)
        if path.parent.as_posix() not in seed_dirs:
            continue
        if path.stem not in REPAIR_CONTEXT_CALLER_STEMS:
            continue
        if _fallback_path_rank(safe_path) != 0:
            continue
        collaborators.append(safe_path)
    return collaborators


def _tested_source_paths(
    *,
    test_paths: list[str],
    safe_files: list[str],
) -> list[str]:
    tested_stems = {
        _tested_source_stem(path)
        for path in test_paths
    }
    tested_stems.discard("")
    if not tested_stems:
        return []
    paths: list[str] = []
    for safe_path in safe_files:
        if _fallback_path_rank(safe_path) != 0:
            continue
        if PurePosixPath(safe_path).stem not in tested_stems:
            continue
        paths.append(safe_path)
    return paths


def _tested_source_stem(path: str) -> str:
    stem = PurePosixPath(path).stem
    if stem.startswith("test_"):
        return stem[5:]
    if stem.endswith("_test"):
        return stem[:-5]
    return stem


def _safe_repair_feedback_paths(
    repair_feedback: dict[str, object],
    key: str,
) -> list[str]:
    return _safe_request_list(repair_feedback.get(key))


def _repair_feedback_patch_artifact_paths(value: object) -> list[str]:
    paths: list[str] = []
    if not isinstance(value, list):
        return paths
    for artifact in value:
        if not isinstance(artifact, dict):
            continue
        files = artifact.get("files")
        for path in _safe_request_list(files):
            if path in paths:
                continue
            paths.append(path)
    return paths


def _structured_repair_feedback(
    request: CodingAgentWriteRequest,
) -> dict[str, object] | None:
    repair_feedback = request.get("repair_feedback")
    if not isinstance(repair_feedback, dict):
        return None
    return repair_feedback


def _fallback_terms(question: str) -> set[str]:
    terms: set[str] = set()
    for raw_term in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", question):
        terms.add(raw_term.casefold())
    return terms


def _fallback_path_score(path: str, terms: set[str]) -> int:
    lowered_path = path.casefold()
    score = 0
    for term in terms:
        if term in lowered_path:
            score += 3
    if lowered_path.startswith("tests/") or "/tests/" in lowered_path:
        score += 2
    if lowered_path.endswith((".md", ".rst")):
        score += 1
    return score


def _fallback_path_rank(path: str) -> int:
    lowered_path = path.casefold()
    if lowered_path.endswith(".py") and not lowered_path.startswith("tests/"):
        return 0
    if lowered_path.startswith("tests/") or "/tests/" in lowered_path:
        return 1
    if lowered_path.endswith((".md", ".rst")):
        return 2
    return 3


def _bounded_fallback_excerpt(content: str) -> str:
    excerpt = content[:MAX_WRITE_FALLBACK_EXCERPT_CHARS].strip()
    return excerpt


def _fallback_line_count(excerpt: str) -> int:
    if not excerpt:
        return 1
    return excerpt.count("\n") + 1


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
    repair_feedback = _structured_repair_feedback(request)
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
    if repair_feedback is not None:
        reading_question = (
            f"{reading_question}\n\n"
            f"{_repair_feedback_reading_block(repair_feedback)}"
        )
    return reading_question


def _repair_feedback_reading_block(repair_feedback: dict[str, object]) -> str:
    lines = [
        "Repair feedback from prior verification:",
        (
            "Inspect the required source-owner paths, caller/import sites, "
            "wrappers, and protected verification paths as read-only evidence."
        ),
        (
            "The protected verification paths are read-only; report what they "
            "verify and keep edits scoped to runtime source owners or callers."
        ),
    ]
    feedback_source = _safe_request_text(repair_feedback.get("feedback_source"))
    if feedback_source:
        lines.append(f"Feedback source: {feedback_source}")
    attempt_index = repair_feedback.get("attempt_index")
    if isinstance(attempt_index, int):
        lines.append(f"Attempt index: {attempt_index}")
    _append_repair_feedback_list(
        lines=lines,
        label="Failed paths",
        items=_safe_repair_feedback_paths(repair_feedback, "failed_paths"),
    )
    _append_repair_feedback_list(
        lines=lines,
        label="Failure summaries",
        items=_safe_request_list(repair_feedback.get("failure_summaries")),
    )
    _append_repair_feedback_list(
        lines=lines,
        label="Required source-owner paths",
        items=_safe_repair_feedback_paths(
            repair_feedback,
            "required_source_owner_paths",
        ),
    )
    _append_repair_feedback_list(
        lines=lines,
        label="Protected verification paths",
        items=_safe_repair_feedback_paths(
            repair_feedback,
            "protected_verification_paths",
        ),
    )
    _append_repair_feedback_list(
        lines=lines,
        label="Previous patch artifact files",
        items=_repair_feedback_patch_artifact_paths(
            repair_feedback.get("previous_patch_artifacts"),
        ),
    )
    return "\n".join(lines)


def _append_repair_feedback_list(
    *,
    lines: list[str],
    label: str,
    items: list[str],
) -> None:
    if not items:
        return
    lines.append(f"{label}:")
    for item in items[:8]:
        lines.append(f"- {item}")


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
