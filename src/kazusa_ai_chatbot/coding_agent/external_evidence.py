"""Supervisor-owned external evidence adapter for coding-agent workflows."""

from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    ExternalEvidenceSummary,
    WritingExternalEvidenceRequest,
)
from kazusa_ai_chatbot.rag.web_agent3 import WebAgent3

WEB_EVIDENCE_CONTEXT = {
    "coding_agent_task": "patch_proposal_external_evidence",
}


async def collect_external_evidence(
    requests: list[WritingExternalEvidenceRequest],
    *,
    trace_summary: list[str],
) -> list[ExternalEvidenceSummary]:
    """Resolve PM-requested public evidence through the web helper contract."""

    evidence: list[ExternalEvidenceSummary] = []
    web_agent = WebAgent3()
    for request in requests:
        result = await web_agent.run(
            task=request["task"],
            context=WEB_EVIDENCE_CONTEXT,
        )
        summary: ExternalEvidenceSummary = {
            "request_id": request["request_id"],
            "task": request["task"],
            "resolved": bool(result.get("resolved", False)),
            "result": str(result.get("result", "")),
        }
        if not summary["resolved"]:
            summary["limitation"] = "External evidence was unavailable."
        evidence.append(summary)
        trace_summary.append(
            "external_evidence "
            f"request={request['request_id']} resolved={summary['resolved']}"
        )
    return evidence
