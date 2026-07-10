"""Live LLM inspection tests for the standalone complex-task resolver."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx
import pytest
from kazusa_ai_chatbot.complex_task_resolver import (
    COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
    COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
    COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
    validate_complex_task_resolution_packet,
    validate_complex_task_resolver_context,
    validate_complex_task_resolver_request,
)
from kazusa_ai_chatbot.complex_task_resolver.service import resolve_complex_task
from kazusa_ai_chatbot.complex_task_resolver.subagent.media import (
    MediaSubagent as ExternalMediaSubagent,
)
from kazusa_ai_chatbot.config import (
    JSON_REPAIR_LLM_BASE_URL,
    JSON_REPAIR_LLM_MODEL,
    SEARXNG_URL,
    WEB_SEARCH_LLM_BASE_URL,
    WEB_SEARCH_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.web_agent3 import direct_searxng, url_reader
from kazusa_ai_chatbot.media_inspection import service as media_service
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "complex_task_resolver_review_cases.json"
)
_ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "test_artifacts"
    / "complex_task_resolver"
)
_FORBIDDEN_REVIEW_KEYS = (
    "expected_review_outcome",
    "required_stages",
    "expected_graph_trace",
    "expected_final_answer",
    "performance_reference_summary",
    "expected_subagent_calls",
    "minimum_viable_answer",
    "forbidden_failure_modes",
)


async def test_live_review_dependency_preflight() -> None:
    """Record dependency availability before live resolver inspection."""

    preflight = await _run_preflight()
    trace_path = write_llm_trace(
        "complex_task_resolver_live_llm",
        "dependency_preflight",
        preflight,
    )
    print(json.dumps(preflight, ensure_ascii=True, sort_keys=True))
    print(f"COMPLEX_TASK_RESOLVER_PREFLIGHT_TRACE={trace_path}")

    if preflight["resolver_llm"]["status"] != "available":
        pytest.skip(preflight["resolver_llm"]["reason"])
    if preflight["json_repair_llm"]["status"] != "available":
        pytest.skip(preflight["json_repair_llm"]["reason"])


async def test_live_followup_tasks_are_emitted_and_executed() -> None:
    """Verify live resolver stages can emit structured executable follow-ups."""

    preflight = await _run_preflight()
    if preflight["resolver_llm"]["status"] != "available":
        pytest.skip(preflight["resolver_llm"]["reason"])
    if preflight["json_repair_llm"]["status"] != "available":
        pytest.skip(preflight["json_repair_llm"]["reason"])

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": (
            "Investigate a deliberately incomplete public answer research "
            "task: compare whether a 120 W router for 8 hours and a 60 W "
            "laptop for 3 hours can fit within a 1500 Wh portable power "
            "station budget. First decompose the known demand facts. If any "
            "required calculation or evidence step is still missing after an "
            "intermediate pass, emit structured followup_tasks so the "
            "resolver can execute the missing step before final synthesis. "
            "Return semantic knowledge only, not final dialog."
        ),
        "reason": "Live Stage 5A follow-up task contract check.",
        "source": "live_llm_review",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": (
            "This review case is designed to check structured resolver "
            "follow-up task output through public module IO."
        ),
        "persona_context_summary": (
            "Return factual structure for review, not character dialog."
        ),
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    options = {
        "schema_version": COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
        "limits": {
            "max_iterations": 4,
            "max_nodes": 8,
            "max_depth": 3,
            "max_subagent_attempts": 1,
        },
    }

    packet = validate_complex_task_resolution_packet(
        await resolve_complex_task(request, context, options)
    )
    trace_summary = packet["trace_summary"]
    followup_events = trace_summary.get("followup_event_log", [])
    if not isinstance(followup_events, list):
        followup_events = []
    stage_io_log = _packet_stage_io_log(packet)
    trace_payload = {
        "preflight": preflight,
        "packet": packet,
        "followup_events": followup_events,
        "stage_io_log": stage_io_log,
        "evidence_result": _packet_evidence_result(packet),
    }
    trace_path = write_llm_trace(
        "complex_task_resolver_live_llm",
        "stage_5a_followup_tasks",
        trace_payload,
    )

    print(f"COMPLEX_TASK_RESOLVER_FOLLOWUP_TRACE={trace_path}")
    print("FOLLOWUP_EVENTS_START")
    print(json.dumps(followup_events, ensure_ascii=False, indent=2))
    print("FOLLOWUP_EVENTS_END")
    print("FOLLOWUP_RESOLVER_OUTPUT_START")
    print(_format_resolver_output(packet))
    print("FOLLOWUP_RESOLVER_OUTPUT_END")

    assert _stage_log_contains_followup_tasks(stage_io_log)
    assert any(
        isinstance(event, dict) and event.get("event") == "created"
        for event in followup_events
    )
    assert _has_semantic_output(packet)


_CASE_RUN_OVERRIDES = {
    "ctr_032_emergency_power_subagent_recursion": {
        "required_subagents": ("algorithmic", "evidence"),
        "required_resolved_subagents": ("algorithmic",),
        "required_min_depth": 2,
        "max_nodes": 8,
        "max_depth": 3,
        "max_iterations": 4,
    },
}


async def _run_live_review_case(
    *,
    case_id: str,
    case_number: str,
    artifact_name: str,
    trace_case_name: str,
    required_subagents: tuple[str, ...] = (),
    required_resolved_subagents: tuple[str, ...] = (),
    required_min_depth: int = 1,
    max_nodes: int = 8,
    max_depth: int = 3,
    max_iterations: int = 8,
) -> None:
    """Run one review fixture through the standalone live resolver harness."""

    preflight = await _run_preflight()
    if preflight["resolver_llm"]["status"] != "available":
        pytest.skip(preflight["resolver_llm"]["reason"])
    if preflight["json_repair_llm"]["status"] != "available":
        pytest.skip(preflight["json_repair_llm"]["reason"])

    case = _load_case(case_id)
    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": case["user_question"],
        "reason": f"Standalone live LLM review case {case_number}.",
        "source": "live_llm_review",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "No caller-collected evidence is supplied; resolver subagents own retrieval.",
        "persona_context_summary": "Return factual structure for review, not final character dialog.",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    options = {
        "schema_version": COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
        "limits": {
            "max_iterations": max_iterations,
            "max_nodes": max_nodes,
            "max_depth": max_depth,
            "max_subagent_attempts": 1,
        },
    }

    packet = validate_complex_task_resolution_packet(
        await resolve_complex_task(request, context, options)
    )
    subagent_call_log = _packet_subagent_call_log(packet)
    evidence_result = _packet_evidence_result(packet)
    stage_io_log = _packet_stage_io_log(packet)
    prompt_payload_errors = _packet_forbidden_metadata_errors(packet)
    artifact_path = _write_case_artifact(
        case=case,
        case_number=case_number,
        artifact_name=artifact_name,
        preflight=preflight,
        evidence_result=evidence_result,
        packet=packet,
        prompt_payload_errors=prompt_payload_errors,
        subagent_call_log=subagent_call_log,
        stage_io_log=stage_io_log,
    )
    trace_path = write_llm_trace(
        "complex_task_resolver_live_llm",
        trace_case_name,
        {
            "case_id": case["case_id"],
            "preflight": preflight,
            "evidence_result": evidence_result,
            "packet": packet,
            "artifact_path": str(artifact_path),
            "prompt_payload_errors": prompt_payload_errors,
            "subagent_call_log": subagent_call_log,
            "stage_io_log": stage_io_log,
        },
    )
    mermaid = _packet_mermaid(packet)

    print(f"COMPLEX_TASK_RESOLVER_CASE_{case_number}_ARTIFACT={artifact_path}")
    print(f"COMPLEX_TASK_RESOLVER_CASE_{case_number}_TRACE={trace_path}")
    print(f"CASE_{case_number}_MERMAID_START")
    print(mermaid)
    print(f"CASE_{case_number}_MERMAID_END")
    print(f"CASE_{case_number}_EVIDENCE_RESULT_START")
    print(_format_evidence_result(evidence_result))
    print(f"CASE_{case_number}_EVIDENCE_RESULT_END")
    print(f"CASE_{case_number}_RESOLVER_OUTPUT_START")
    print(_format_resolver_output(packet))
    print(f"CASE_{case_number}_RESOLVER_OUTPUT_END")

    assert prompt_payload_errors == []
    assert str(packet["investigation_summary"]).strip()
    assert _has_semantic_output(packet)
    if required_subagents:
        observed_subagents = {
            str(call["subagent"])
            for call in subagent_call_log
        }
        missing_subagents = sorted(
            set(required_subagents) - observed_subagents
        )
        assert missing_subagents == []
    if required_resolved_subagents:
        resolved_subagents = {
            str(call["subagent"])
            for call in subagent_call_log
            if call["resolved"] is True
        }
        missing_resolved_subagents = sorted(
            set(required_resolved_subagents) - resolved_subagents
        )
        assert missing_resolved_subagents == []
    observed_depth = max(
        int(node["depth"])
        for node in packet["graph"]["nodes"].values()
    )
    assert observed_depth >= required_min_depth
    assert _synthesis_dependency_violations(packet) == []



async def _run_preflight() -> dict[str, object]:
    """Check live dependencies used by the standalone review harness."""

    resolver_llm = await _endpoint_status(WEB_SEARCH_LLM_BASE_URL)
    json_repair_llm = await _endpoint_status(JSON_REPAIR_LLM_BASE_URL)
    network = await _public_network_status()
    web_read = await _web_read_status()
    web_search = await _web_search_status()
    preflight = {
        "resolver_llm": {
            "route": "WEB_SEARCH_LLM",
            "base_url": WEB_SEARCH_LLM_BASE_URL,
            "model": WEB_SEARCH_LLM_MODEL,
            **resolver_llm,
        },
        "json_repair_llm": {
            "route": "JSON_REPAIR_LLM",
            "base_url": JSON_REPAIR_LLM_BASE_URL,
            "model": JSON_REPAIR_LLM_MODEL,
            **json_repair_llm,
        },
        "web_read": web_read,
        "web_search": web_search,
        "network": network,
        "rag_mongo_embedding": {
            "status": "not_selected",
            "reason": "live review uses WebAgent3 evidence only",
        },
        "mcp": {
            "status": "unused",
            "reason": "MCP is outside the Phase 1 review dependency set",
        },
    }
    return preflight


def _packet_subagent_call_log(
    packet: dict[str, object],
) -> list[dict[str, object]]:
    """Read resolver-owned subagent call trace from the returned packet."""

    trace_summary = packet["trace_summary"]
    call_log = trace_summary.get("subagent_call_log", [])
    if not isinstance(call_log, list):
        raise AssertionError("packet trace subagent_call_log must be a list")
    for call in call_log:
        if not isinstance(call, dict):
            raise AssertionError("packet trace subagent calls must be objects")
    return call_log


def _packet_stage_io_log(packet: dict[str, object]) -> list[dict[str, object]]:
    """Read bounded LLM stage inputs and parsed outputs from packet trace."""

    trace_summary = packet["trace_summary"]
    stage_io_log = trace_summary.get("stage_io_log", [])
    if not isinstance(stage_io_log, list):
        raise AssertionError("packet trace stage_io_log must be a list")
    for stage_io in stage_io_log:
        if not isinstance(stage_io, dict):
            raise AssertionError("packet trace stage IO entries must be objects")
    return stage_io_log


def _packet_evidence_result(packet: dict[str, object]) -> dict[str, object]:
    """Project resolver-owned subagent evidence for human review."""

    subagent_results: list[dict[str, object]] = []
    for call in _packet_subagent_call_log(packet):
        result = call.get("result", {})
        if not isinstance(result, dict):
            result = {}
        trace = call.get("trace", {})
        if not isinstance(trace, dict):
            trace = {}
        subagent_results.append({
            "node_id": call.get("node_id", ""),
            "subagent": call.get("subagent", ""),
            "action": call.get("action", ""),
            "objective": call.get("objective", ""),
            "resolved": call.get("resolved", False),
            "status": call.get("status", ""),
            "result": result,
            "attempts": call.get("attempts", 0),
            "trace": trace,
            "unresolved_items": call.get("unresolved_items", []),
        })
    return {
        "source": "resolver_trace_summary.subagent_call_log",
        "subagent_result_count": len(subagent_results),
        "subagent_results": subagent_results,
    }


def _format_evidence_result(evidence_result: dict[str, object]) -> str:
    """Render the resolver-owned evidence projection without raw JSON noise."""

    results = evidence_result.get("subagent_results", [])
    if not isinstance(results, list) or not results:
        return "No resolver-owned subagent evidence was captured."
    lines: list[str] = []
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        lines.append(
            f"{index}. {item.get('subagent', '')} "
            f"{item.get('status', '')} for {item.get('node_id', '')}"
        )
        objective = str(item.get("objective", "")).strip()
        if objective:
            lines.append(f"   Objective: {objective}")
        result = item.get("result", {})
        if isinstance(result, dict):
            display_text = _first_present_string(
                result,
                ("display", "summary", "result_str", "formula"),
            )
            if display_text:
                lines.append(f"   Result: {display_text}")
        unresolved = item.get("unresolved_items", [])
        if isinstance(unresolved, list) and unresolved:
            lines.append(f"   Blockers: {'; '.join(str(x) for x in unresolved)}")
    return "\n".join(lines) if lines else "No formatted evidence rows."


def _format_resolver_output(packet: dict[str, object]) -> str:
    """Render the semantic packet sections consumed by the next LLM."""

    lines = [
        "Investigation summary:",
        str(packet["investigation_summary"]).strip(),
        "",
        "Knowledge we know so far:",
    ]
    lines.extend(_format_packet_items(packet["knowledge_we_know_so_far"]))
    lines.extend(["", "Knowledge still lacking:"])
    lines.extend(_format_packet_items(packet["knowledge_still_lacking"]))
    lines.extend(["", "Recommended next iteration:"])
    lines.extend(_format_packet_items(packet["recommended_next_iteration"]))
    lines.extend(["", "Evidence boundary notes:"])
    lines.extend(_format_packet_items(packet["evidence_boundary_notes"]))
    rendered_output = "\n".join(lines)
    return rendered_output


def _format_packet_items(value: object) -> list[str]:
    """Render one semantic packet list for human review output."""

    if not isinstance(value, list) or not value:
        lines = ["- none"]
        return lines
    lines = [
        f"- {str(item).strip()}"
        for item in value
        if str(item).strip()
    ]
    if not lines:
        lines = ["- none"]
    return lines


def _has_semantic_output(packet: dict[str, object]) -> bool:
    """Return whether the packet contains at least one useful semantic row."""

    for field_name in (
        "knowledge_we_know_so_far",
        "knowledge_still_lacking",
        "recommended_next_iteration",
        "evidence_boundary_notes",
    ):
        items = packet[field_name]
        if isinstance(items, list) and items:
            return_value = True
            return return_value
    return_value = False
    return return_value


def _first_present_string(
    data: dict[str, object],
    keys: tuple[str, ...],
) -> str:
    """Return the first non-empty string from a dict."""

    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _stage_log_contains_followup_tasks(
    stage_io_log: list[dict[str, object]],
) -> bool:
    """Return whether any live LLM stage emitted structured follow-up tasks."""

    for stage_row in stage_io_log:
        parsed_output = stage_row.get("parsed_output")
        if not isinstance(parsed_output, dict):
            continue
        followup_tasks = parsed_output.get("followup_tasks")
        if isinstance(followup_tasks, list) and followup_tasks:
            return_value = True
            return return_value
    return_value = False
    return return_value


def _synthesis_dependency_violations(
    packet: dict[str, object],
) -> list[str]:
    """Return synthesis nodes resolved before earlier prerequisites."""

    graph = packet["graph"]
    nodes = graph["nodes"]
    violations: list[str] = []
    for node_id, node in nodes.items():
        if node["node_kind"] != "synthesis":
            continue
        if node["status"] != "resolved":
            continue
        parent_id = node["parent_id"]
        if parent_id is None:
            continue
        parent = nodes[parent_id]
        for sibling_id in parent["children"]:
            if sibling_id == node_id:
                break
            blockers = _subtree_unresolved_nodes(nodes, sibling_id)
            if blockers:
                violations.append(node_id)
                break
    return violations


def _subtree_unresolved_nodes(
    nodes: dict[str, object],
    node_id: str,
) -> list[str]:
    """Collect unresolved node ids in one packet graph subtree."""

    node = nodes[node_id]
    blockers: list[str] = []
    if node["status"] in ("pending", "blocked", "cannot_answer"):
        blockers.append(node_id)
    for child_id in node["children"]:
        child_blockers = _subtree_unresolved_nodes(nodes, child_id)
        blockers.extend(child_blockers)
    return blockers


async def _endpoint_status(base_url: str) -> dict[str, str]:
    """Return availability for an OpenAI-compatible model endpoint."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/models")
    except httpx.HTTPError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    if response.status_code >= 500:
        return {
            "status": "unavailable",
            "reason": f"server error {response.status_code}",
        }
    return {"status": "available", "reason": f"http {response.status_code}"}


async def _public_network_status() -> dict[str, str]:
    """Check public outbound HTTP availability."""

    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            response = await client.get("https://example.com")
    except httpx.HTTPError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    return {"status": "available", "reason": f"http {response.status_code}"}


async def _web_read_status() -> dict[str, str]:
    """Check process-local URL read capability."""

    result = await url_reader.web_url_read(
        url="https://example.com",
        startChar=0,
        maxLength=200,
    )
    if result.startswith("Error:"):
        return {"status": "unavailable", "reason": result[:200]}
    return {"status": "available", "reason": "read example.com"}


async def _web_search_status() -> dict[str, str]:
    """Check SearXNG-backed web search capability."""

    if not SEARXNG_URL:
        return {
            "status": "unavailable",
            "reason": "SEARXNG_URL is not configured",
        }
    result = await direct_searxng.web_search(query="OpenAI Codex CLI", pageno=1)
    if result.startswith("Error:"):
        return {"status": "unavailable", "reason": result[:200]}
    return {"status": "available", "reason": "SearXNG returned results"}




def _load_case(case_id: str) -> dict[str, object]:
    """Load one review fixture case by id."""

    fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    for case in fixture["cases"]:
        if case["case_id"] == case_id:
            return case
    raise AssertionError(f"review case not found: {case_id}")


def _load_review_cases() -> list[dict[str, object]]:
    """Load all live review fixture cases in file order."""

    fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    cases = fixture["cases"]
    if not isinstance(cases, list):
        raise AssertionError("review fixture cases must be a list")
    return cases



def _packet_forbidden_metadata_errors(packet: dict[str, object]) -> list[str]:
    """Ensure hidden fixture metadata did not enter resolver output."""

    errors: list[str] = []
    serialized = json.dumps(packet, ensure_ascii=False)
    for forbidden_key in _FORBIDDEN_REVIEW_KEYS:
        if forbidden_key in serialized:
            errors.append(f"resolver packet leaked {forbidden_key}")
    return errors


def _write_case_artifact(
    *,
    case: dict[str, object],
    case_number: str,
    artifact_name: str,
    preflight: dict[str, object],
    evidence_result: dict[str, object],
    packet: dict[str, object],
    prompt_payload_errors: list[str],
    subagent_call_log: list[dict[str, object]] | None = None,
    stage_io_log: list[dict[str, object]] | None = None,
) -> Path:
    """Write one raw structured case evidence artifact."""

    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACT_DIR / artifact_name
    document = {
        "case_number": case_number,
        "case_id": case["case_id"],
        "category": case["category"],
        "fixture_review_outcome": case.get("expected_review_outcome", ""),
        "hidden_expected_answers_in_prompt": False,
        "prompt_payload_anti_cheat_errors": prompt_payload_errors,
        "user_question": case["user_question"],
        "dependency_preflight": preflight,
        "evidence_result": evidence_result,
        "subagent_calls": subagent_call_log or [],
        "llm_stage_io": stage_io_log or [],
        "resolver_graph_mermaid": _packet_mermaid(packet),
        "final_packet": packet,
    }
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _packet_mermaid(packet: dict[str, object]) -> str:
    """Render the validated packet graph as a Mermaid flowchart."""

    graph = packet["graph"]
    nodes = graph["nodes"]
    lines = ["graph TD"]
    for node_id, node in nodes.items():
        label = _mermaid_label(
            f"{node_id}\\n{node['node_kind']}\\n{node['status']}"
        )
        lines.append(f"  {node_id}[\"{label}\"]")
    for node_id, node in nodes.items():
        for child_id in node["children"]:
            lines.append(f"  {node_id} --> {child_id}")
        if node["collapsed_into"]:
            lines.append(f"  {node_id} -. collapsed .-> {node['collapsed_into']}")
    mermaid = "\n".join(lines)
    return mermaid


def _mermaid_label(value: str) -> str:
    """Escape a compact Mermaid node label."""

    label = value.replace("\"", "'")
    return label


def _case_number_from_id(case_id: str) -> str:
    """Return the two-digit fixture number from a canonical case id."""

    match = re.match(r"^ctr_(\d{3})_", case_id)
    if match is None:
        raise AssertionError(f"invalid review case id: {case_id}")
    return match.group(1)[-2:]


def _case_artifact_stem(case: dict[str, object]) -> str:
    """Return a deterministic artifact stem for one fixture case."""

    case_id = str(case["case_id"])
    case_number = _case_number_from_id(case_id)
    slug = re.sub(r"^ctr_\d{3}_", "", case_id)
    slug = re.sub(r"[^a-z0-9_]+", "_", slug)
    return f"case_{case_number}_{slug}"


def _make_live_review_case(case: dict[str, object]):
    """Create one named live LLM test for a fixture case."""

    case_id = str(case["case_id"])
    case_number = _case_number_from_id(case_id)
    artifact_stem = _case_artifact_stem(case)
    overrides = _CASE_RUN_OVERRIDES.get(case_id, {})

    async def _test() -> None:
        await _run_live_review_case(
            case_id=case_id,
            case_number=case_number,
            artifact_name=f"{artifact_stem}.json",
            trace_case_name=artifact_stem,
            **overrides,
        )

    _test.__name__ = f"test_live_review_{artifact_stem}"
    _test.__doc__ = (
        "Run fixture case "
        f"{case_number} through the standalone complex-task resolver."
    )
    return _test


for _case in _load_review_cases():
    _generated_test = _make_live_review_case(_case)
    globals()[_generated_test.__name__] = _generated_test


_PUBLIC_PYTHON_LOGO_URL = (
    "https://raw.githubusercontent.com/github/explore/main/topics/python/"
    "python.png"
)


@pytest.mark.live_internet
async def test_live_media_inspection_external_image_exact_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetch a real public image and review its visual evidence result."""

    capturing_llm = _MediaCapturingLLM(media_service._media_inspection_llm)
    monkeypatch.setattr(media_service, "_media_inspection_llm", capturing_llm)
    task = _external_media_task(
        _PUBLIC_PYTHON_LOGO_URL,
        "What colors are most visible in this Python logo image?",
    )
    result = await ExternalMediaSubagent().run(task, {})
    trace_path = write_llm_trace(
        "complex_task_resolver_live_llm",
        "external_image_exact_question",
        {
            "task": task,
            "result": result,
            "raw_model_output": capturing_llm.raw_output,
            "judgment": "manual_review_required_for_external_visual_grounding",
        },
    )

    assert result["status"] in {"resolved", "partial"}
    assert result["trace"]["media_inspection_called"] is True
    assert capturing_llm.raw_output.strip()
    assert trace_path.exists()


@pytest.mark.live_internet
async def test_live_media_inspection_external_image_fetch_refusal() -> None:
    """Review the private-network refusal before any external media call."""

    task = _external_media_task(
        "http://127.0.0.1/private.png",
        "What colors are visible in this image?",
    )
    result = await ExternalMediaSubagent().run(task, {})
    trace_path = write_llm_trace(
        "complex_task_resolver_live_llm",
        "external_image_fetch_refusal",
        {
            "task": task,
            "result": result,
            "judgment": "private-network fetch must fail before visual inspection",
        },
    )

    assert result["status"] == "failed"
    assert result["trace"]["media_inspection_called"] is False
    assert "private" in result["result"]["evidence_boundary_notes"][0]
    assert trace_path.exists()


def _external_media_task(url: str, question: str) -> dict[str, object]:
    """Build one bounded complex-resolver external-image request."""

    result = {
        "schema_version": "complex_task_subagent_request.v1",
        "node_id": "external_media_review",
        "subagent": "media",
        "action": "inspect_media",
        "objective": question,
        "payload": {"url": url, "question": question},
        "constraints": {},
    }
    return result


class _MediaCapturingLLM:
    """Retain raw production media-inspection model output for review."""

    def __init__(self, delegate: object) -> None:
        """Store the configured production LLM delegate."""

        self._delegate = delegate
        self.raw_output = ""

    async def ainvoke(self, messages, config):
        """Forward one invocation and retain the model response body."""

        response = await self._delegate.ainvoke(messages, config=config)
        self.raw_output = str(response.content)
        return response
