"""Run one production RAG2 recall-quality case and write raw JSON evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.llm_route_report import LLM_ROUTE_CONFIGS
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_projection import (
    project_known_facts,
)
from kazusa_ai_chatbot.rag.quote_aware_sequence import (
    call_quote_aware_rag_supervisor,
)
from kazusa_ai_chatbot.time_boundary import (
    local_time_context_from_storage_utc,
    storage_utc_now_iso,
)

DEFAULT_CASES_PATH = Path(
    "test_artifacts/rag2_recall_quality/inputs/"
    "contextual_case_matrix_alt10.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "test_artifacts/rag2_recall_quality/"
    "production_rag2_e2e_20260524"
)
RAG_ROUTE_NAMES = frozenset({
    "RAG_PLANNER_LLM",
    "RAG_SUBAGENT_LLM",
    "EMBEDDING",
})


def main() -> int:
    """Run one selected case from the recall-quality matrix."""

    _configure_utf8_stdio()
    args = _parse_args()
    cases = _load_cases(args.cases)
    case = _select_case(cases, args.case_id)
    artifact = asyncio.run(_run_case(case))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{case['case_id']}.json"
    _write_json(output_path, artifact)
    print(
        f"wrote {output_path} "
        f"loops={artifact['observed_load']['loop_count']} "
        f"dispatches={len(artifact['dispatches'])}"
    )
    return_value = 0
    return return_value


def _configure_utf8_stdio() -> None:
    """Keep Windows console output usable for loaded CJK case data."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a single-case RAG2 run."""

    parser = argparse.ArgumentParser(
        description="Run one production RAG2 recall-quality case.",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    return args


def _load_cases(path: Path) -> list[dict[str, Any]]:
    """Load JSONL recall-quality cases from disk."""

    case_lines = path.read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in case_lines if line.strip()]
    return cases


def _select_case(
    cases: list[dict[str, Any]],
    case_id: str,
) -> dict[str, Any]:
    """Return the requested case row."""

    for case in cases:
        if case["case_id"] == case_id:
            return case
    raise ValueError(f"case_id not found: {case_id}")


async def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    """Execute the production RAG2 path for a single case."""

    started_at = time.perf_counter()
    run_started_at_utc = _utc_now_second_iso()
    storage_timestamp_utc = storage_utc_now_iso()
    context = _case_context(case, storage_timestamp_utc)
    character_name = _character_name_from_case(case)

    rag_supervisor_result = await call_quote_aware_rag_supervisor(
        fresh_query=case["decontextualized_input"],
        reply_context={},
        character_name=character_name,
        context=context,
    )
    projected_rag_result = project_known_facts(
        rag_supervisor_result["known_facts"],
        current_user_id=case["global_user_id"],
        character_user_id=CHARACTER_GLOBAL_USER_ID,
        answer=str(rag_supervisor_result["answer"]),
        unknown_slots=rag_supervisor_result["unknown_slots"],
        loop_count=int(rag_supervisor_result["loop_count"] or 0),
    )

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    artifact: dict[str, Any] = {
        "artifact_version": "production_rag2_e2e.v1",
        "run_started_at_utc": run_started_at_utc,
        "run_finished_at_utc": _utc_now_second_iso(),
        "duration_ms": duration_ms,
        "case_id": case["case_id"],
        "case_type": case["case_type"],
        "target_lane": case["target_lane"],
        "input_text": case["input_text"],
        "decontextualized_input": case["decontextualized_input"],
        "quality_question": case["quality_question"],
        "search_anchors": list(case.get("search_anchors", [])),
        "platform": case["platform"],
        "platform_channel_id": case["platform_channel_id"],
        "channel_type": case["channel_type"],
        "display_name": case["display_name"],
        "context_shape": _context_shape(case),
        "run_context": {
            "storage_timestamp_utc": storage_timestamp_utc,
            "local_time_context": context["local_time_context"],
            "character_name": character_name,
            "rag_routes": _rag_routes(),
            "data_source": "real MongoDB via production RAG2 helpers",
            "input_source": str(DEFAULT_CASES_PATH),
        },
        "expected_sources": case.get("expected_sources", []),
        "distractor_sources": case.get("distractor_sources", []),
        "dispatches": _dispatches_from_projected(projected_rag_result),
        "rag_supervisor_result": rag_supervisor_result,
        "projected_rag_result": projected_rag_result,
        "observed_load": {
            "loop_count": rag_supervisor_result["loop_count"],
            "known_fact_count": len(rag_supervisor_result["known_facts"]),
            "unknown_slot_count": len(rag_supervisor_result["unknown_slots"]),
            "llm_calls": "not_exposed",
            "embedding_calls": "not_exposed",
            "db_or_search_calls": "not_exposed",
            "evidence_char_count": _evidence_char_count(projected_rag_result),
        },
    }
    return artifact


def _case_context(
    case: dict[str, Any],
    storage_timestamp_utc: str,
) -> dict[str, Any]:
    """Build the runtime RAG context from one experiment input case."""

    character_name = _character_name_from_case(case)
    local_time_context = local_time_context_from_storage_utc(
        storage_timestamp_utc
    )
    context = {
        "platform": case["platform"],
        "platform_channel_id": case["platform_channel_id"],
        "channel_type": case["channel_type"],
        "character_profile": {
            "global_user_id": CHARACTER_GLOBAL_USER_ID,
            "name": character_name,
        },
        "active_turn_platform_message_ids": [],
        "active_turn_conversation_row_ids": [],
        "global_user_id": case["global_user_id"],
        "user_name": case["display_name"],
        "user_profile": {
            "global_user_id": case["global_user_id"],
            "display_name": case["display_name"],
        },
        "current_timestamp_utc": storage_timestamp_utc,
        "local_time_context": local_time_context,
        "prompt_message_context": {
            "body_text": case["input_text"],
            "addressed_to_global_user_ids": [CHARACTER_GLOBAL_USER_ID],
            "broadcast": False,
            "mentions": [],
            "attachments": [],
            "reply": None,
        },
        "channel_topic": "",
        "chat_history_recent": case["chat_history_recent"],
        "chat_history_wide": case["chat_history_wide"],
        "reply_context": {},
        "indirect_speech_context": "",
        "conversation_progress": None,
        "conversation_episode_state": None,
        "promoted_reflection_context": None,
    }
    return context


def _character_name_from_case(case: dict[str, Any]) -> str:
    """Infer the active character display name from assistant history rows."""

    histories = (
        list(case.get("chat_history_recent", []))
        + list(case.get("chat_history_wide", []))
    )
    for row in histories:
        if row.get("role") != "assistant":
            continue
        display_name = str(row.get("display_name", "")).strip()
        if display_name:
            return display_name
    return_value = "active character"
    return return_value


def _context_shape(case: dict[str, Any]) -> dict[str, Any]:
    """Describe the case context included in the RAG call."""

    context_shape = {
        "chat_history_recent_count": len(case["chat_history_recent"]),
        "chat_history_wide_count": len(case["chat_history_wide"]),
        "has_reply_context": False,
        "has_conversation_progress": False,
        "has_conversation_episode_state": False,
        "has_promoted_reflection_context": False,
    }
    return context_shape


def _dispatches_from_projected(projected_rag_result: dict[str, Any]) -> list[dict]:
    """Return public supervisor dispatch trace from projected RAG output."""

    supervisor_trace = projected_rag_result["supervisor_trace"]
    dispatches = list(supervisor_trace["dispatched"])
    return dispatches


def _rag_routes() -> list[dict[str, str]]:
    """Return non-secret RAG model route metadata."""

    routes = [
        {
            "route": row["route"],
            "model": row["model"],
            "source_url": row["source_url"],
        }
        for row in LLM_ROUTE_CONFIGS
        if row["route"] in RAG_ROUTE_NAMES
    ]
    return routes


def _evidence_char_count(projected_rag_result: dict[str, Any]) -> int:
    """Count prompt-facing RAG evidence characters for load inspection."""

    public_keys = (
        "answer",
        "third_party_profiles",
        "memory_evidence",
        "recall_evidence",
        "conversation_evidence",
        "external_evidence",
    )
    total = 0
    for key in public_keys:
        total += len(json.dumps(projected_rag_result[key], ensure_ascii=False))
    return total


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one raw evidence artifact as UTF-8 JSON."""

    content = json.dumps(value, ensure_ascii=False, indent=2)
    path.write_text(f"{content}\n", encoding="utf-8")


def _utc_now_second_iso() -> str:
    """Return current UTC time at second precision."""

    now = datetime.now(tz=UTC).replace(microsecond=0)
    return_value = now.isoformat()
    return return_value


if __name__ == "__main__":
    raise SystemExit(main())
