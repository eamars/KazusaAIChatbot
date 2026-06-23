"""Focused real-LLM test for existing-source ownership decisions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


_TEST_NAME = "coding_agent_writing_source_ownership_live_llm"
_GATE_01_REPO_ROOT = (
    Path("test_artifacts")
    / "coding_agent_live_workspaces"
    / "phase2_hard_gates"
    / "gate_01"
    / "r572b3de25416"
)


async def test_gate_01_source_ownership_failure_mode() -> None:
    """Source Ownership PM should choose owners, not related consumers."""

    from kazusa_ai_chatbot.coding_agent.code_writing.source_owners import (
        collect_source_owner_candidates,
    )
    from kazusa_ai_chatbot.coding_agent.code_writing.source_ownership import (
        decide_source_ownership,
    )

    if not _GATE_01_REPO_ROOT.is_dir():
        pytest.fail(f"Gate 01 checkout is missing: {_GATE_01_REPO_ROOT}")

    file_demands = _gate_01_file_demands()
    reading_evidence = _gate_01_reading_evidence(_GATE_01_REPO_ROOT)
    owner_candidates = collect_source_owner_candidates(
        repo_root=_GATE_01_REPO_ROOT,
        reading_evidence=reading_evidence,
        max_candidates=12,
    )
    trace: dict[str, object] = {}

    resolution = await decide_source_ownership(
        question=(
            "Recreate the supplied archived development plan as a proposed "
            "patch artifact for this repository state."
        ),
        mode="edit_existing_repository",
        file_demands=file_demands,
        owner_candidates=owner_candidates,
        reading_evidence=reading_evidence,
        trace=trace,
    )
    decisions = {
        decision["demand_id"]: decision
        for decision in resolution["decisions"]
    }
    trace_path = write_llm_trace(
        _TEST_NAME,
        "gate_01_source_ownership",
        {
            "file_demands": file_demands,
            "owner_candidates": owner_candidates,
            "resolution": resolution,
            "trace": trace,
        },
    )

    print(f"trace_path={trace_path}")
    print(json.dumps(resolution, ensure_ascii=False, indent=2))

    assert resolution["status"] == "accepted"
    assert (
        decisions["cache2_stats_tracking"]["owned_path"]
        == "src/kazusa_ai_chatbot/rag/cache2_runtime.py"
    )
    assert (
        decisions["health_endpoint_update"]["owned_path"]
        == "src/kazusa_ai_chatbot/service.py"
    )
    assert decisions["howto_docs_update"]["owned_path"] == "docs/HOWTO.md"


def _gate_01_file_demands() -> list[dict[str, Any]]:
    demands = [
        {
            "demand_id": "cache2_stats_tracking",
            "role": "RAG Runtime Developer",
            "purpose": (
                "Implement internal tracking of cache hits and misses grouped "
                "by retrieval agent within the Cache2 runtime."
            ),
            "file_kind": "existing",
            "interface_contract": {
                "component": "RAGCache2Runtime",
                "inputs": ["agent identifier during lookup operations"],
                "outputs": [
                    "sanitized agent_name, hit_count, miss_count, hit_rate rows"
                ],
                "invariants": [
                    "must not expose raw cache keys, queries, or user IDs",
                    "counters increment on every lookup attempt",
                ],
            },
            "integration_contract": {
                "provides_to_pm": ["sanitized Cache2 per-agent stats"],
                "consumes_from": [],
            },
            "change_goal": "Add per-agent hit/miss counters and a stats method.",
            "work_instructions": [
                "Modify the cache lookup path to increment hit and miss counters.",
                "Return sanitized per-agent stats.",
            ],
            "required_slots": [
                "per-agent hit counter",
                "per-agent miss counter",
                "sanitized stats method",
            ],
            "validation_expectations": [
                "Counters increment correctly for hits and misses."
            ],
        },
        {
            "demand_id": "health_endpoint_update",
            "role": "Service API Developer",
            "purpose": (
                "Extend the existing /health response to include sanitized "
                "Cache2 agent statistics."
            ),
            "file_kind": "existing",
            "interface_contract": {
                "component": "HealthResponse",
                "inputs": ["sanitized stats from RAGCache2Runtime"],
                "outputs": ["existing health fields plus cache2.agents"],
                "invariants": ["status, db, and scheduler remain present"],
            },
            "integration_contract": {
                "provides_to_pm": ["updated /health response"],
                "consumes_from": ["cache2_stats_tracking"],
            },
            "change_goal": "Integrate Cache2 stats into the health endpoint.",
            "work_instructions": [
                "Update the health response model.",
                "Update the /health handler to include Cache2 stats.",
            ],
            "required_slots": [
                "health response model",
                "health endpoint handler",
            ],
            "validation_expectations": [
                "GET /health returns existing fields and cache2.agents."
            ],
        },
        {
            "demand_id": "howto_docs_update",
            "role": "Technical Writer",
            "purpose": "Document the enriched /health response for operators.",
            "file_kind": "docs",
            "interface_contract": {
                "component": "HOWTO Documentation",
                "inputs": ["final /health response shape"],
                "outputs": ["updated health endpoint documentation"],
                "invariants": ["existing documentation structure remains intact"],
            },
            "integration_contract": {
                "provides_to_pm": ["operator-facing documentation"],
                "consumes_from": ["health_endpoint_update"],
            },
            "change_goal": "Update HOWTO health endpoint documentation.",
            "work_instructions": [
                "Document cache2.agents fields under the health endpoint section."
            ],
            "required_slots": ["health endpoint documentation"],
            "validation_expectations": [
                "Documentation describes cache2.agents fields."
            ],
        },
    ]
    return demands


def _gate_01_reading_evidence(repo_root: Path) -> list[dict[str, Any]]:
    rows = [
        _evidence_row(
            repo_root,
            path="src/kazusa_ai_chatbot/rag/cache2_runtime.py",
            line_start=135,
            line_end=170,
            topic="Cache2 runtime class and get path",
            reason="Shows the runtime owns cache lookups and counters.",
        ),
        _evidence_row(
            repo_root,
            path="src/kazusa_ai_chatbot/service.py",
            line_start=124,
            line_end=130,
            topic="Health response model",
            reason="Shows the health response model owner.",
        ),
        _evidence_row(
            repo_root,
            path="src/kazusa_ai_chatbot/service.py",
            line_start=455,
            line_end=467,
            topic="Health endpoint handler",
            reason="Shows the service owns GET /health.",
        ),
        _evidence_row(
            repo_root,
            path="src/kazusa_ai_chatbot/rag/user_lookup_agent.py",
            line_start=19,
            line_end=65,
            topic="Cache runtime consumer",
            reason="Shows a consumer of RAGCache2Runtime, not the runtime owner.",
        ),
        _evidence_row(
            repo_root,
            path="docs/HOWTO.md",
            line_start=194,
            line_end=200,
            topic="Health endpoint documentation",
            reason="Shows the operator documentation section for GET /health.",
        ),
        _evidence_row(
            repo_root,
            path="tests/test_rag_initializer_cache2.py",
            line_start=83,
            line_end=107,
            topic="Existing Cache2 test pattern",
            reason="Shows existing tests around Cache2 runtime behavior.",
        ),
    ]
    return rows


def _evidence_row(
    repo_root: Path,
    *,
    path: str,
    line_start: int,
    line_end: int,
    topic: str,
    reason: str,
) -> dict[str, Any]:
    file_path = repo_root / path
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    excerpt = "\n".join(lines[line_start - 1:line_end])
    row = {
        "path": path,
        "line_start": line_start,
        "line_end": line_end,
        "symbol_or_topic": topic,
        "excerpt": excerpt,
        "reason": reason,
    }
    return row
