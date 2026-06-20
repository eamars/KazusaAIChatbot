"""Top-level standalone coding-agent supervisor."""

import kazusa_ai_chatbot.coding_agent.code_fetching as code_fetching
import kazusa_ai_chatbot.coding_agent.code_reading as code_reading
from kazusa_ai_chatbot.coding_agent.code_fetching.models import (
    CodeRepositoryRef,
)
from kazusa_ai_chatbot.coding_agent.models import (
    CodingAgentRepositorySummary,
    CodingAgentRequest,
    CodingAgentResponse,
)

DIRTY_CHECKOUT_LIMITATION = (
    "Existing local checkout is dirty; evidence may include "
    "uncommitted local changes."
)


async def answer_code_question(
    request: CodingAgentRequest,
) -> CodingAgentResponse:
    """Answer a source-code question through Phase 0 fetching and Phase 1 reading."""

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
