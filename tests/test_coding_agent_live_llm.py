"""Primary real-LLM gates for Phase 1 code reading."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.config import BACKGROUND_WORK_LLM_BASE_URL
from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


_TEST_NAME = "coding_agent_phase1_live_llm"


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _repository(tmp_path: Path, name: str, files: dict[str, list[str]]) -> dict[str, Any]:
    repo_root = tmp_path / name
    for relative_path, lines in files.items():
        _write_text(repo_root / relative_path, lines)

    repository = {
        "provider": "github",
        "owner": "fixture",
        "repo": name,
        "source_url": f"https://github.com/fixture/{name}",
        "requested_ref": None,
        "resolved_ref": "main",
        "current_commit": "f" * 40,
        "default_branch": "main",
        "local_root": str(repo_root),
        "storage_kind": "existing_local_checkout",
        "managed_checkout": False,
        "workspace_root": str(tmp_path / "workspace"),
        "cache_key": f"github-fixture-{name}-main",
        "dirty_state": "clean",
    }
    return repository


def _repository_summary(repository: dict[str, Any]) -> dict[str, Any]:
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


def _scope(
    kind: str = "repository",
    repo_relative_path: str | None = None,
) -> dict[str, Any]:
    if repo_relative_path is None:
        source_url = "local://fixture/repository"
        interpretation = "entire repository"
    else:
        source_url = f"local://fixture/repository/{repo_relative_path}"
        interpretation = repo_relative_path

    scope = {
        "kind": kind,
        "repo_relative_path": repo_relative_path,
        "source_url": source_url,
        "requested_ref": None,
        "interpretation": interpretation,
    }
    return scope


def _architecture_files() -> dict[str, list[str]]:
    files = {
        "README.md": [
            "# Order Gateway",
            "The service accepts order commands and records payment outcomes.",
        ],
        "src/order_gateway/api.py": [
            "from order_gateway.service import OrderService",
            "",
            "def create_order_endpoint(payload: dict) -> dict:",
            "    service = OrderService()",
            "    return service.create_order(payload)",
        ],
        "src/order_gateway/service.py": [
            "from order_gateway.payments import PaymentGateway",
            "from order_gateway.ledger import LedgerWriter",
            "",
            "class OrderService:",
            "    def create_order(self, payload: dict) -> dict:",
            "        payment = PaymentGateway().charge_order(payload)",
            "        LedgerWriter().record_payment(payment)",
            "        return {'status': payment['status']}",
        ],
        "src/order_gateway/payments.py": [
            "class PaymentGateway:",
            "    def charge_order(self, payload: dict) -> dict:",
            "        return {'status': 'charged', 'order_id': payload['id']}",
        ],
        "src/order_gateway/ledger.py": [
            "class LedgerWriter:",
            "    def record_payment(self, payment: dict) -> None:",
            "        assert payment['status'] == 'charged'",
        ],
    }
    return files


def _pipeline_files() -> dict[str, list[str]]:
    files = {
        "src/telemetry/adapters/http_events.py": [
            "def decode_event(request_body: dict) -> dict:",
            "    return {'device_id': request_body['device'], 'value': request_body['value']}",
        ],
        "src/telemetry/pipeline/normalizer.py": [
            "def normalize_event(event: dict) -> dict:",
            "    event['normalized_value'] = float(event['value'])",
            "    return event",
        ],
        "src/telemetry/pipeline/enricher.py": [
            "def attach_site_context(event: dict, site_lookup: dict) -> dict:",
            "    event['site_id'] = site_lookup[event['device_id']]",
            "    return event",
        ],
        "src/telemetry/storage/writer.py": [
            "def write_measurement(event: dict, database: object) -> None:",
            "    database.measurements.insert_one(event)",
        ],
        "src/telemetry/service.py": [
            "from telemetry.adapters.http_events import decode_event",
            "from telemetry.pipeline.normalizer import normalize_event",
            "from telemetry.pipeline.enricher import attach_site_context",
            "from telemetry.storage.writer import write_measurement",
            "",
            "def ingest_measurement(request_body: dict, site_lookup: dict, database: object) -> None:",
            "    event = decode_event(request_body)",
            "    event = normalize_event(event)",
            "    event = attach_site_context(event, site_lookup)",
            "    write_measurement(event, database)",
        ],
    }
    return files


def _control_files() -> dict[str, list[str]]:
    files = {
        "src/fan_control/controller.py": [
            "class FanController:",
            "    def __init__(self) -> None:",
            "        self.integral_error = 0.0",
            "        self.previous_error = 0.0",
            "",
            "    def update_speed(self, target_temp: float, measured_temp: float) -> int:",
            "        temperature_error = measured_temp - target_temp",
            "        self.integral_error += temperature_error",
            "        derivative_error = temperature_error - self.previous_error",
            "        self.previous_error = temperature_error",
            "        speed = 18 * temperature_error + 2 * self.integral_error + derivative_error",
            "        return max(0, min(100, int(speed)))",
        ],
        "src/fan_control/service.py": [
            "from fan_control.controller import FanController",
            "",
            "def regulate_cooling(sensor: object, motor: object) -> None:",
            "    controller = FanController()",
            "    measured_temp = sensor.read_temperature()",
            "    fan_speed = controller.update_speed(24.0, measured_temp)",
            "    motor.set_speed_percent(fan_speed)",
        ],
    }
    return files


def _symbol_files() -> dict[str, list[str]]:
    files = {
        "src/retry/budget.py": [
            "class RetryBudget:",
            "    def __init__(self, max_attempts: int) -> None:",
            "        self.max_attempts = max_attempts",
            "",
            "    def allow(self, attempt_count: int) -> bool:",
            "        return attempt_count < self.max_attempts",
        ],
        "src/retry/worker.py": [
            "from retry.budget import RetryBudget",
            "",
            "def should_retry(attempt_count: int) -> bool:",
            "    budget = RetryBudget(max_attempts=3)",
            "    return budget.allow(attempt_count)",
        ],
    }
    return files


def _directory_files() -> dict[str, list[str]]:
    files = {
        "src/workers/queue.py": [
            "class WorkQueue:",
            "    def claim_next(self) -> dict | None:",
            "        return {'job_id': 'job-1', 'status': 'claimed'}",
        ],
        "src/workers/scheduler.py": [
            "from workers.queue import WorkQueue",
            "",
            "def run_scheduler_tick() -> str:",
            "    job = WorkQueue().claim_next()",
            "    if job is None:",
            "        return 'idle'",
            "    return job['status']",
        ],
    }
    return files


async def _skip_if_llm_unavailable() -> None:
    base_url = _effective_route_base_url()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/models")
    except httpx.HTTPError as exc:
        pytest.skip(f"Coding-agent LLM endpoint is unavailable: {base_url}; {exc}")

    if response.status_code >= 500:
        pytest.skip(
            "Coding-agent LLM endpoint returned server error "
            f"{response.status_code}: {base_url}"
        )


def _effective_route_base_url() -> str:
    from os import getenv

    base_url = getenv("CODING_AGENT_LLM_BASE_URL") or BACKGROUND_WORK_LLM_BASE_URL
    return base_url


def _trace_payload(
    *,
    case_id: str,
    command_subject: str,
    repository: dict[str, Any] | None,
    raw_trace: dict[str, object],
    result: object,
    validation: dict[str, object],
) -> dict[str, object]:
    payload = {
        "case_id": case_id,
        "command_subject": command_subject,
        "effective_route_base_url": _effective_route_base_url(),
        "repository": _safe_repository(repository),
        "raw_stage_trace": raw_trace,
        "result": result,
        "validation": validation,
        "judgment": "manual_review_required_for_phase1_code_reading_quality",
    }
    return payload


def _safe_repository(
    repository: dict[str, Any] | None,
) -> dict[str, object] | None:
    if repository is None:
        return None

    safe = dict(_repository_summary(repository))
    safe["local_root"] = "<redacted>"
    safe["workspace_root"] = "<redacted>"
    safe["cache_key"] = "<redacted>"
    return safe


def _write_case_trace(
    *,
    case_id: str,
    command_subject: str,
    repository: dict[str, Any] | None,
    raw_trace: dict[str, object],
    result: object,
    validation: dict[str, object],
) -> Path:
    trace_path = write_llm_trace(
        _TEST_NAME,
        case_id,
        _trace_payload(
            case_id=case_id,
            command_subject=command_subject,
            repository=repository,
            raw_trace=raw_trace,
            result=result,
            validation=validation,
        ),
    )
    print(
        "coding_agent_live_trace="
        f"{trace_path} case_id={case_id} validation={validation}"
    )
    return trace_path


def _assert_stage_trace(raw_trace: dict[str, object]) -> None:
    assert raw_trace.get("effective_route")
    assert raw_trace.get("raw_output")
    assert raw_trace.get("parsed_output") is not None


def _assert_public_safe(value: object) -> None:
    serialized = repr(value)
    assert "local_root" not in serialized
    assert "workspace_root" not in serialized
    assert "cache_key" not in serialized


def _assert_assignment_limits(decision: dict[str, Any]) -> None:
    assignments = decision.get("assignments", [])
    assert isinstance(assignments, list)
    assert 0 < len(assignments) <= 3
    for assignment in assignments:
        assert set(assignment) == {
            "assignment_id",
            "role",
            "scope",
            "questions",
            "required_slots",
        }
        assert assignment["questions"]
        assert assignment["required_slots"]
        scope = assignment["scope"]
        assert scope["kind"] in {"file", "directory", "symbol", "search"}
        assert scope["values"]


def _assignment(
    *,
    assignment_id: str,
    role: str,
    kind: str,
    values: list[str],
    questions: list[str],
    required_slots: list[str],
) -> dict[str, Any]:
    return {
        "assignment_id": assignment_id,
        "role": role,
        "scope": {
            "kind": kind,
            "values": values,
        },
        "questions": questions,
        "required_slots": required_slots,
    }


def _evidence_text(rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in rows:
        parts.append(
            " ".join(
                [
                    str(row["path"]),
                    str(row["symbol_or_topic"]),
                    str(row["excerpt"]),
                    str(row["reason"]),
                ]
            )
        )
    return "\n".join(parts)


def _selected_evidence() -> list[dict[str, Any]]:
    evidence = [
        {
            "path": "src/order_gateway/service.py",
            "line_start": 5,
            "line_end": 8,
            "symbol_or_topic": "OrderService.create_order",
            "excerpt": (
                "payment = PaymentGateway().charge_order(payload)\n"
                "LedgerWriter().record_payment(payment)"
            ),
            "reason": "Matched service orchestration evidence.",
        },
        {
            "path": "src/order_gateway/payments.py",
            "line_start": 1,
            "line_end": 3,
            "symbol_or_topic": "PaymentGateway.charge_order",
            "excerpt": (
                "class PaymentGateway:\n"
                "    def charge_order(self, payload: dict) -> dict:"
            ),
            "reason": "Matched payment gateway definition evidence.",
        },
    ]
    return evidence


def _programmer_report_from_evidence(
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "assignment_id": "payment-flow-reader",
        "status": "succeeded",
        "files_read": [
            "src/order_gateway/service.py",
            "src/order_gateway/payments.py",
        ],
        "facts": [
            {
                "kind": "call_flow",
                "summary": (
                    "OrderService.create_order calls "
                    "PaymentGateway.charge_order and then LedgerWriter."
                ),
                "evidence_refs": [
                    "src/order_gateway/service.py:5",
                    "src/order_gateway/payments.py:1",
                ],
            }
        ],
        "evidence": evidence,
        "open_questions": [],
    }
    return report


def _pm_decision(intent: str) -> dict[str, Any]:
    decision = {
        "status": "sufficient",
        "intent": intent,
        "required_slots": ["entry point", "downstream call"],
        "assignments": [],
        "missing_slots": [],
    }
    return decision


def _run_pm_decision(
    *,
    question: str,
    repository: dict[str, Any],
    source_scope: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, object]]:
    from kazusa_ai_chatbot.coding_agent.code_reading.product_manager import (
        decide_reading_work,
    )
    from kazusa_ai_chatbot.coding_agent.code_reading.repository_map import (
        build_repository_map_summary,
    )

    raw_trace: dict[str, object] = {}
    repo_map_summary = build_repository_map_summary(repository, source_scope)
    pm_input = {
        "question": question,
        "repository_summary": _repository_summary(repository),
        "source_scope": source_scope,
        "repo_map_summary": repo_map_summary,
        "previous_reports": [],
    }
    decision = decide_reading_work(
        pm_input,
        trace=raw_trace,
    )
    return decision, raw_trace


def _run_programmer(
    *,
    repository: dict[str, Any],
    source_scope: dict[str, Any],
    assignment: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, object]]:
    from kazusa_ai_chatbot.coding_agent.code_reading.programmer import (
        run_programmer_assignment,
    )

    raw_trace: dict[str, object] = {}
    report = run_programmer_assignment(
        repository,
        assignment,
        source_scope,
        max_files=6,
        max_excerpt_chars=12000,
        trace=raw_trace,
    )
    return report, raw_trace


def _run_synthesizer(
    *,
    question: str,
    pm_decision: dict[str, Any],
    programmer_reports: list[dict[str, Any]],
    selected_evidence: list[dict[str, Any]],
    limitations: list[str],
) -> tuple[str, dict[str, object]]:
    from kazusa_ai_chatbot.coding_agent.code_reading.synthesizer import (
        synthesize_from_programmer_reports,
    )

    raw_trace: dict[str, object] = {}
    answer = synthesize_from_programmer_reports(
        question=question,
        pm_decision=pm_decision,
        programmer_reports=programmer_reports,
        evidence=selected_evidence,
        limitations=limitations,
        repository_summary={
            "provider": "fixture",
            "owner": "fixture",
            "repo": "synthesis",
        },
        preferred_language="English",
        max_answer_chars=1800,
        trace=raw_trace,
    )
    answer_text = answer[0] if isinstance(answer, tuple) else answer
    return answer_text, raw_trace


async def test_live_pm_decides_architecture_overview(tmp_path: Path) -> None:
    await _skip_if_llm_unavailable()
    repository = _repository(tmp_path, "architecture_repo", _architecture_files())
    decision, raw_trace = _run_pm_decision(
        question=(
            "Explain the architecture of this order gateway and identify the "
            "main responsibility boundaries."
        ),
        repository=repository,
        source_scope=_scope(),
    )
    validation = {
        "status": decision["status"],
        "intent": decision["intent"],
        "assignment_count": len(decision.get("assignments", [])),
    }
    _write_case_trace(
        case_id="test_live_pm_decides_architecture_overview",
        command_subject="PM architecture overview decision",
        repository=repository,
        raw_trace=raw_trace,
        result=decision,
        validation=validation,
    )

    _assert_stage_trace(raw_trace)
    assert decision["status"] == "need_programmers"
    assert decision["intent"] == "architecture_overview"
    _assert_assignment_limits(decision)


async def test_live_pm_decides_pipeline_or_data_flow(tmp_path: Path) -> None:
    await _skip_if_llm_unavailable()
    repository = _repository(tmp_path, "pipeline_repo", _pipeline_files())
    decision, raw_trace = _run_pm_decision(
        question=(
            "How does a telemetry measurement move from HTTP input to storage?"
        ),
        repository=repository,
        source_scope=_scope(),
    )
    validation = {
        "status": decision["status"],
        "intent": decision["intent"],
        "assignment_count": len(decision.get("assignments", [])),
    }
    _write_case_trace(
        case_id="test_live_pm_decides_pipeline_or_data_flow",
        command_subject="PM pipeline or data-flow decision",
        repository=repository,
        raw_trace=raw_trace,
        result=decision,
        validation=validation,
    )

    _assert_stage_trace(raw_trace)
    assert decision["status"] == "need_programmers"
    assert decision["intent"] == "pipeline_or_data_flow"
    _assert_assignment_limits(decision)


async def test_live_pm_decides_symbol_behavior(tmp_path: Path) -> None:
    await _skip_if_llm_unavailable()
    repository = _repository(tmp_path, "symbol_repo", _symbol_files())
    decision, raw_trace = _run_pm_decision(
        question="What does RetryBudget.allow do?",
        repository=repository,
        source_scope=_scope(),
    )
    validation = {
        "status": decision["status"],
        "intent": decision["intent"],
        "assignment_count": len(decision.get("assignments", [])),
    }
    _write_case_trace(
        case_id="test_live_pm_decides_symbol_behavior",
        command_subject="PM symbol behavior decision",
        repository=repository,
        raw_trace=raw_trace,
        result=decision,
        validation=validation,
    )

    _assert_stage_trace(raw_trace)
    assert decision["status"] == "need_programmers"
    assert decision["intent"] == "symbol_behavior"
    _assert_assignment_limits(decision)


async def test_live_pm_handles_ambiguous_or_too_broad_request(
    tmp_path: Path,
) -> None:
    await _skip_if_llm_unavailable()
    repository = _repository(tmp_path, "architecture_repo", _architecture_files())
    decision, raw_trace = _run_pm_decision(
        question="Explain everything in this repository.",
        repository=repository,
        source_scope=_scope(),
    )
    validation = {
        "status": decision["status"],
        "intent": decision["intent"],
        "assignment_count": len(decision.get("assignments", [])),
    }
    _write_case_trace(
        case_id="test_live_pm_handles_ambiguous_or_too_broad_request",
        command_subject="PM too-broad request decision",
        repository=repository,
        raw_trace=raw_trace,
        result=decision,
        validation=validation,
    )

    _assert_stage_trace(raw_trace)
    assert decision["status"] in {"needs_user_input", "overloaded"}
    assert not decision.get("assignments")


async def test_live_pm_rejects_unsupported_write_request(tmp_path: Path) -> None:
    await _skip_if_llm_unavailable()
    repository = _repository(tmp_path, "architecture_repo", _architecture_files())
    decision, raw_trace = _run_pm_decision(
        question="Rewrite the payment gateway to use a new API.",
        repository=repository,
        source_scope=_scope(),
    )
    validation = {
        "status": decision["status"],
        "intent": decision["intent"],
        "assignment_count": len(decision.get("assignments", [])),
    }
    _write_case_trace(
        case_id="test_live_pm_rejects_unsupported_write_request",
        command_subject="PM unsupported write decision",
        repository=repository,
        raw_trace=raw_trace,
        result=decision,
        validation=validation,
    )

    _assert_stage_trace(raw_trace)
    assert decision["status"] == "needs_user_input"
    assert decision["intent"] == "unsupported_request"
    assert not decision.get("assignments")


async def test_live_programmer_reads_file_scope(tmp_path: Path) -> None:
    await _skip_if_llm_unavailable()
    repository = _repository(tmp_path, "architecture_repo", _architecture_files())
    assignment = _assignment(
        assignment_id="order-service-file-reader",
        role="interface reader",
        kind="file",
        values=["src/order_gateway/service.py"],
        questions=["What does OrderService.create_order call?"],
        required_slots=["entry point", "downstream call"],
    )
    report, raw_trace = _run_programmer(
        repository=repository,
        source_scope=_scope("file", "src/order_gateway/service.py"),
        assignment=assignment,
    )
    validation = {
        "status": report["status"],
        "files_read": report["files_read"],
        "evidence_count": len(report["evidence"]),
    }
    _write_case_trace(
        case_id="test_live_programmer_reads_file_scope",
        command_subject="Programmer file-scope report",
        repository=repository,
        raw_trace=raw_trace,
        result=report,
        validation=validation,
    )

    _assert_stage_trace(raw_trace)
    _assert_public_safe(report)
    assert report["status"] == "succeeded"
    assert report["files_read"] == ["src/order_gateway/service.py"]
    assert "charge_order" in _evidence_text(report["evidence"])


async def test_live_programmer_reads_directory_scope(tmp_path: Path) -> None:
    await _skip_if_llm_unavailable()
    repository = _repository(tmp_path, "pipeline_repo", _pipeline_files())
    assignment = _assignment(
        assignment_id="pipeline-directory-reader",
        role="data-flow reader",
        kind="directory",
        values=["src/telemetry/pipeline"],
        questions=["How does the pipeline transform telemetry events?"],
        required_slots=["normalization", "enrichment"],
    )
    report, raw_trace = _run_programmer(
        repository=repository,
        source_scope=_scope("directory", "src/telemetry/pipeline"),
        assignment=assignment,
    )
    validation = {
        "status": report["status"],
        "files_read": report["files_read"],
        "evidence_count": len(report["evidence"]),
    }
    _write_case_trace(
        case_id="test_live_programmer_reads_directory_scope",
        command_subject="Programmer directory-scope report",
        repository=repository,
        raw_trace=raw_trace,
        result=report,
        validation=validation,
    )

    _assert_stage_trace(raw_trace)
    _assert_public_safe(report)
    assert report["status"] == "succeeded"
    assert set(report["files_read"]) <= {
        "src/telemetry/pipeline/normalizer.py",
        "src/telemetry/pipeline/enricher.py",
    }
    evidence_text = _evidence_text(report["evidence"])
    assert "normalize_event" in evidence_text
    assert "attach_site_context" in evidence_text


async def test_live_programmer_reads_symbol_or_search_scope(
    tmp_path: Path,
) -> None:
    await _skip_if_llm_unavailable()
    repository = _repository(tmp_path, "symbol_repo", _symbol_files())
    assignment = _assignment(
        assignment_id="retry-budget-symbol-reader",
        role="symbol reader",
        kind="symbol",
        values=["RetryBudget.allow"],
        questions=["What behavior does RetryBudget.allow implement?"],
        required_slots=["symbol definition", "representative use"],
    )
    report, raw_trace = _run_programmer(
        repository=repository,
        source_scope=_scope(),
        assignment=assignment,
    )
    validation = {
        "status": report["status"],
        "files_read": report["files_read"],
        "evidence_count": len(report["evidence"]),
    }
    _write_case_trace(
        case_id="test_live_programmer_reads_symbol_or_search_scope",
        command_subject="Programmer symbol-scope report",
        repository=repository,
        raw_trace=raw_trace,
        result=report,
        validation=validation,
    )

    _assert_stage_trace(raw_trace)
    _assert_public_safe(report)
    assert report["status"] == "succeeded"
    assert "src/retry/budget.py" in report["files_read"]
    assert "RetryBudget" in _evidence_text(report["evidence"])
    assert "allow" in _evidence_text(report["evidence"])


async def test_live_programmer_reports_no_evidence(tmp_path: Path) -> None:
    await _skip_if_llm_unavailable()
    repository = _repository(tmp_path, "symbol_repo", _symbol_files())
    assignment = _assignment(
        assignment_id="missing-symbol-reader",
        role="symbol reader",
        kind="search",
        values=["NonexistentCircuitBreaker"],
        questions=["Where is NonexistentCircuitBreaker implemented?"],
        required_slots=["symbol definition"],
    )
    report, raw_trace = _run_programmer(
        repository=repository,
        source_scope=_scope(),
        assignment=assignment,
    )
    validation = {
        "status": report["status"],
        "files_read": report["files_read"],
        "evidence_count": len(report["evidence"]),
    }
    _write_case_trace(
        case_id="test_live_programmer_reports_no_evidence",
        command_subject="Programmer no-evidence report",
        repository=repository,
        raw_trace=raw_trace,
        result=report,
        validation=validation,
    )

    _assert_stage_trace(raw_trace)
    _assert_public_safe(report)
    assert report["status"] == "no_evidence"
    assert report["evidence"] == []
    assert report["facts"] == []
    assert report["open_questions"]


async def test_live_synthesizer_produces_grounded_answer() -> None:
    await _skip_if_llm_unavailable()
    evidence = _selected_evidence()
    report = _programmer_report_from_evidence(evidence)
    answer, raw_trace = _run_synthesizer(
        question="How does order creation call payment and ledger code?",
        pm_decision=_pm_decision("pipeline_or_data_flow"),
        programmer_reports=[report],
        selected_evidence=evidence,
        limitations=[],
    )
    validation = {
        "answer_chars": len(answer),
        "mentions_payment_gateway": "PaymentGateway" in answer,
        "mentions_ledger_writer": "LedgerWriter" in answer,
    }
    _write_case_trace(
        case_id="test_live_synthesizer_produces_grounded_answer",
        command_subject="Synthesizer grounded answer",
        repository=None,
        raw_trace=raw_trace,
        result={"answer_text": answer},
        validation=validation,
    )

    _assert_stage_trace(raw_trace)
    assert "PaymentGateway" in answer
    assert "charge_order" in answer
    assert "LedgerWriter" in answer
    assert "ShadowLedger" not in answer


async def test_live_synthesizer_preserves_limitation() -> None:
    await _skip_if_llm_unavailable()
    evidence = _selected_evidence()
    report = _programmer_report_from_evidence(evidence)
    limitation = "No selected evidence showed retry backoff configuration."
    answer, raw_trace = _run_synthesizer(
        question="How does order creation work, including retry backoff?",
        pm_decision=_pm_decision("pipeline_or_data_flow"),
        programmer_reports=[report],
        selected_evidence=evidence,
        limitations=[limitation],
    )
    validation = {
        "answer_chars": len(answer),
        "preserved_limitation": "retry backoff" in answer.casefold(),
    }
    _write_case_trace(
        case_id="test_live_synthesizer_preserves_limitation",
        command_subject="Synthesizer limitation preservation",
        repository=None,
        raw_trace=raw_trace,
        result={"answer_text": answer, "limitations": [limitation]},
        validation=validation,
    )

    _assert_stage_trace(raw_trace)
    assert "retry backoff" in answer.casefold()
    assert "exponential" not in answer.casefold()


async def test_live_synthesizer_blocks_ungrounded_identifier() -> None:
    await _skip_if_llm_unavailable()
    evidence = _selected_evidence()
    report = _programmer_report_from_evidence(evidence)
    answer, raw_trace = _run_synthesizer(
        question="Does ShadowLedger persist order payment details?",
        pm_decision=_pm_decision("insufficient_evidence"),
        programmer_reports=[report],
        selected_evidence=evidence,
        limitations=["No selected evidence matched the requested identifier."],
    )
    validation = {
        "answer_chars": len(answer),
        "blocked_shadow_ledger": "ShadowLedger" not in answer,
    }
    _write_case_trace(
        case_id="test_live_synthesizer_blocks_ungrounded_identifier",
        command_subject="Synthesizer grounding block",
        repository=None,
        raw_trace=raw_trace,
        result={"answer_text": answer},
        validation=validation,
    )

    _assert_stage_trace(raw_trace)
    assert "ShadowLedger" not in answer
    assert "PaymentGateway" in answer or "selected evidence" in answer


async def test_live_answer_code_question_pipeline_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _skip_if_llm_unavailable()
    repository = _repository(tmp_path, "pipeline_repo", _pipeline_files())
    response = await _answer_with_fetching_fixture(
        monkeypatch=monkeypatch,
        repository=repository,
        source_scope=_scope(),
        question="How does a telemetry measurement flow from input to storage?",
    )
    validation = {
        "status": response["status"],
        "evidence_count": len(response["evidence"]),
        "answer_chars": len(response["answer_text"]),
    }
    _write_case_trace(
        case_id="test_live_answer_code_question_pipeline_flow",
        command_subject="Direct answer pipeline flow",
        repository=repository,
        raw_trace={"response_trace_summary": response["trace_summary"]},
        result=response,
        validation=validation,
    )

    _assert_public_safe(response)
    assert response["status"] == "succeeded"
    assert response["evidence"]
    answer = response["answer_text"]
    assert "decode_event" in answer
    assert "normalize_event" in answer
    assert "write_measurement" in answer


async def test_live_answer_code_question_control_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _skip_if_llm_unavailable()
    repository = _repository(tmp_path, "control_repo", _control_files())
    response = await _answer_with_fetching_fixture(
        monkeypatch=monkeypatch,
        repository=repository,
        source_scope=_scope(),
        question=(
            "Explain the feedback control flow that converts measured "
            "temperature into fan motor speed."
        ),
    )
    validation = {
        "status": response["status"],
        "evidence_count": len(response["evidence"]),
        "answer_chars": len(response["answer_text"]),
    }
    _write_case_trace(
        case_id="test_live_answer_code_question_control_flow",
        command_subject="Direct answer control flow",
        repository=repository,
        raw_trace={"response_trace_summary": response["trace_summary"]},
        result=response,
        validation=validation,
    )

    _assert_public_safe(response)
    assert response["status"] == "succeeded"
    assert response["evidence"]
    answer = response["answer_text"]
    assert "temperature_error" in answer
    assert "integral_error" in answer
    assert "set_speed_percent" in answer


async def _answer_with_fetching_fixture(
    *,
    monkeypatch: pytest.MonkeyPatch,
    repository: dict[str, Any],
    source_scope: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    from kazusa_ai_chatbot.coding_agent import answer_code_question
    from kazusa_ai_chatbot.coding_agent import code_fetching

    async def fake_fetching_run(_request: dict[str, Any]) -> dict[str, Any]:
        result = {
            "status": "succeeded",
            "message": "fixture resolved",
            "repository": repository,
            "source_scope": source_scope,
            "limitations": [],
            "trace_summary": ["fetch:fixture resolved"],
        }
        return result

    monkeypatch.setattr(code_fetching, "run", fake_fetching_run)
    response = await answer_code_question(
        {
            "question": question,
            "source_url": repository["source_url"],
            "preferred_language": "English",
            "max_answer_chars": 2200,
            "workspace_root": str(Path(repository["workspace_root"])),
        }
    )
    return response
