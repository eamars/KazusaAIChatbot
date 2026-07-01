"""Evidence subagent contract tests for the complex-task resolver."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.complex_task_resolver import (
    COMPLEX_TASK_SUBAGENT_REQUEST_VERSION,
    COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
    validate_complex_task_subagent_request,
    validate_complex_task_subagent_result,
)
from kazusa_ai_chatbot.complex_task_resolver.subagents import (
    ComplexTaskEvidenceSubagent,
    UnavailableEvidenceSubagent,
)


class _FakeWebAgent:
    """Capture evidence requests while returning a helper-agent envelope."""

    def __init__(self, result: dict[str, object] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._result = result or {
            "resolved": True,
            "status": "success",
            "reason": "found enough source evidence",
            "result": "Source-backed product facts found.",
            "attempts": 1,
            "knowledge_metadata": {},
            "cache": {
                "enabled": False,
                "hit": False,
                "cache_name": "web_agent3",
                "reason": "agent_not_cacheable",
            },
        }

    async def run(
        self,
        task: str,
        context: dict[str, object],
        max_attempts: int = 3,
    ) -> dict[str, object]:
        self.calls.append({
            "task": task,
            "context": context,
            "max_attempts": max_attempts,
        })
        result = dict(self._result)
        return result


@pytest.mark.asyncio
async def test_evidence_subagent_calls_web_agent3_declared_io() -> None:
    """Collect evidence through the production WebAgent3 helper contract."""

    request = validate_complex_task_subagent_request({
        "schema_version": COMPLEX_TASK_SUBAGENT_REQUEST_VERSION,
        "node_id": "evidence_1",
        "subagent": "evidence",
        "action": "retrieve_source_facts",
        "objective": "Fetch current public benchmark data.",
        "payload": {"query": "public benchmark data"},
        "constraints": {"max_results": 3},
    })
    context = {
        "root_question": "Compare two GPUs.",
        "parent_chain_summary": "Need current evidence.",
        "sibling_summaries": [],
        "available_evidence": [],
        "time_context": {"current_date": "2026-06-30"},
    }
    web_agent = _FakeWebAgent()
    subagent = ComplexTaskEvidenceSubagent(web_agent=web_agent)

    result = await subagent.run(request, context, max_attempts=2)
    validated = validate_complex_task_subagent_result(result)

    assert validated["schema_version"] == COMPLEX_TASK_SUBAGENT_RESULT_VERSION
    assert validated["resolved"] is True
    assert validated["status"] == "resolved"
    assert validated["attempts"] == 1
    assert validated["result"]["summary"] == "Source-backed product facts found."
    assert validated["trace"]["web_agent3"]["max_attempts"] == 2
    assert validated["trace"]["web_agent3"]["local_time_context"] == {
        "current_local_datetime": "2026-06-30",
    }
    assert validated["unresolved_items"] == []
    assert web_agent.calls == [{
        "task": (
            "Objective: Fetch current public benchmark data.\n"
            "Search query: public benchmark data"
        ),
        "context": {
            "root_question": "Compare two GPUs.",
            "parent_chain_summary": "Need current evidence.",
            "sibling_summaries": [],
            "available_evidence": [],
            "time_context": {"current_date": "2026-06-30"},
            "local_time_context": {
                "current_local_datetime": "2026-06-30",
            },
            "original_query": (
                "Objective: Fetch current public benchmark data.\n"
                "Search query: public benchmark data"
            ),
        },
        "max_attempts": 2,
    }]

    compatibility = UnavailableEvidenceSubagent(
        reason="web_search unavailable in this review environment",
    )
    compatibility_result = await compatibility.run(request, context, max_attempts=1)
    assert compatibility_result["status"] == "unavailable"


@pytest.mark.asyncio
async def test_evidence_subagent_preserves_partial_web_agent_status() -> None:
    """Weak or incomplete web evidence should remain partial downstream."""

    request = validate_complex_task_subagent_request({
        "schema_version": COMPLEX_TASK_SUBAGENT_REQUEST_VERSION,
        "node_id": "evidence_1",
        "subagent": "evidence",
        "action": "retrieve_source_facts",
        "objective": "Fetch current product pricing.",
        "payload": {"query": "current product pricing"},
        "constraints": {},
    })
    context = {
        "root_question": "Compare product prices.",
        "parent_chain_summary": "",
        "sibling_summaries": [],
        "available_evidence": [],
        "time_context": {"current_date": "2026-06-30"},
    }
    web_agent = _FakeWebAgent(result={
        "resolved": True,
        "status": "partial",
        "reason": "source is relevant but lacks current pricing",
        "result": "Found an old product page but no current price.",
        "attempts": 1,
        "knowledge_metadata": {"source_fit": "partial"},
        "cache": {
            "enabled": False,
            "hit": False,
            "cache_name": "web_agent3",
            "reason": "agent_not_cacheable",
        },
    })
    subagent = ComplexTaskEvidenceSubagent(web_agent=web_agent)

    result = await subagent.run(request, context, max_attempts=1)
    validated = validate_complex_task_subagent_result(result)

    assert validated["resolved"] is False
    assert validated["status"] == "partial"
    assert validated["result"]["summary"] == (
        "Found an old product page but no current price."
    )
    assert validated["result"]["source_quality"] == "partial"
    assert validated["result"]["source_reason"] == (
        "source is relevant but lacks current pricing"
    )
    assert validated["trace"]["web_agent3"]["status"] == "partial"
    assert validated["trace"]["web_agent3"]["reason"] == (
        "source is relevant but lacks current pricing"
    )
    assert validated["trace"]["web_agent3"]["knowledge_metadata"] == {
        "source_fit": "partial",
    }
    assert validated["unresolved_items"] == [
        "Found an old product page but no current price.",
    ]


@pytest.mark.asyncio
async def test_evidence_subagent_preserves_declared_source_hints() -> None:
    """Pass source hints to WebAgent3 instead of reducing them to keywords."""

    request = validate_complex_task_subagent_request({
        "schema_version": COMPLEX_TASK_SUBAGENT_REQUEST_VERSION,
        "node_id": "evidence_1",
        "subagent": "evidence",
        "action": "retrieve_source_facts",
        "objective": "Fetch the current public version.",
        "payload": {
            "query": "latest stable release",
            "target_date": "2026-06-29",
            "source_hint": "example.com/download",
            "source_url": "example.net/current",
        },
        "constraints": {
            "source_preference": [
                "https://example.org/releases",
                "example.com/download",
            ],
            "official_url": "https://example.edu/stable",
        },
    })
    context = {
        "root_question": "Find current version.",
        "parent_chain_summary": "",
        "sibling_summaries": [],
        "available_evidence": [],
        "time_context": {"current_date": "2026-06-30"},
    }
    web_agent = _FakeWebAgent()
    subagent = ComplexTaskEvidenceSubagent(web_agent=web_agent)

    await subagent.run(request, context, max_attempts=1)

    assert web_agent.calls[0]["task"] == (
        "Objective: Fetch the current public version.\n"
        "Preferred source: https://example.com/download\n"
        "Preferred source: https://example.net/current\n"
        "Preferred source: https://example.org/releases\n"
        "Preferred source: https://example.edu/stable\n"
        "Search query: latest stable release\n"
        "As-of date: 2026-06-29"
    )


def test_subagent_result_rejects_hidden_fixture_hints() -> None:
    """Keep review fixture expected answers out of subagent result payloads."""

    with pytest.raises(ValueError):
        validate_complex_task_subagent_result({
            "schema_version": COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
            "resolved": True,
            "status": "resolved",
            "result": {
                "expected_final_answer": "hidden reference answer",
            },
            "attempts": 1,
            "cache": {"enabled": False},
            "trace": {"case_id": "ctr_001_agent_harness_comparison"},
            "unresolved_items": [],
        })

    with pytest.raises(ValueError):
        validate_complex_task_subagent_result({
            "schema_version": COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
            "resolved": True,
            "status": "resolved",
            "result": {
                "summary": "ctr_027_semantic_collapse_negative",
            },
            "attempts": 1,
            "cache": {"enabled": False},
            "trace": {},
            "unresolved_items": [],
        })


def test_subagent_result_allows_use_cases_and_ide_text() -> None:
    """Do not confuse ordinary semantic evidence with fixture metadata."""

    validated = validate_complex_task_subagent_result({
        "schema_version": COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
        "resolved": True,
        "status": "resolved",
        "result": {
            "summary": (
                "GitHub Copilot has IDE assistant use cases, while Codex CLI "
                "and OpenAI API have different operating surfaces."
            ),
        },
        "attempts": 1,
        "cache": {"enabled": False},
        "trace": {},
        "unresolved_items": [],
    })

    assert validated["resolved"] is True
