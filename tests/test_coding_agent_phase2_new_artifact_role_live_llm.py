"""Live LLM role diagnostics for new-artifact code writing."""

from __future__ import annotations

from copy import deepcopy
import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
import pytest

from kazusa_ai_chatbot.coding_agent.code_writing.patcher import (
    materialize_patch_artifacts,
)
from kazusa_ai_chatbot.coding_agent.code_writing.acceptance import (
    derive_acceptance_criteria,
    evaluate_artifact_alignment,
)
from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
    decide_writing_work,
)
from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
    run_writing_programmer_contract,
)
from kazusa_ai_chatbot.coding_agent.code_writing.synthesizer import (
    synthesize_patch_proposal,
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

pytestmark = pytest.mark.live_llm

GATE_FIXTURE_PATH = Path(
    "test_artifacts/live_gate/coding_agent_phase2_new_artifact_gates.json"
)
TRACE_DIR = Path("test_artifacts/llm_traces/coding_agent_phase2_roles")
MAX_ARTIFACT_CHARS = 12000
ROLE_REVIEW_TIMEOUT_SECONDS = 300

GATE_05_FULL_REQUEST = '''\
Create a small Python project that reads a CSV inventory of pages, fetches each
listed URL, extracts the HTML title and first h1 heading, merges those values
with the inventory rows, and writes a consolidated CSV report. It should include
a CLI, source modules, mocked HTTP tests, and a README that explains the input
CSV columns and command workflow. The project may use only the Python standard
library.
'''


PM_ROLE_REVIEW_PROMPT = '''\
You review one product-manager output for new-file code writing.

The product manager does not write code. Its job is to create artifact
contracts that independent programmers can implement without seeing each
other's work.

# Review Rules
Pass only when the output is ready for independent programmers:
1. The requested artifact set is represented well enough for the user request.
2. Source artifacts define clear public interfaces when another artifact must
   consume them.
3. Test artifacts consume source interfaces using the same behavior and output
   shape described by the source artifact.
4. Interface contracts describe arguments, return shape or side effects, and
   enough behavior to prevent incompatible source and test implementations.
5. A consumed interface must not shorten the provider contract so much that
   the return shape, side effect, or error behavior is lost.
6. When source and tests both depend on file content, data rows, config
   objects, or command-line arguments, the shared input shape must appear in
   the source and test contracts, not only in README or docs.
7. Documentation and config artifacts must match the same user-visible tool.

Fail when the PM output is structurally valid but too vague, when source and
test contracts can reasonably lead to different data shapes, when a needed
artifact is missing, or when the output asks a programmer to infer hidden peer
behavior.

# Output Format
Return strict JSON:
{
  "status": "pass | fail",
  "confidence": 0,
  "reasons": ["short reason"],
  "blockers": ["blocking weakness, or empty list"]
}
'''

PROGRAMMER_ROLE_REVIEW_PROMPT = '''\
You review one programmer output for new-file code writing.

The programmer receives exactly one artifact contract and must return exactly
one complete artifact. Judge the artifact against the contract, not against a
different implementation you might prefer.

# Review Rules
Pass only when the artifact satisfies the given contract:
1. It implements or documents the requested artifact kind and content format.
2. Required imports are satisfied, and extra standard library imports are fine
   when they support the same contract.
3. Provided interfaces use the names, arguments, return shapes, side effects,
   and important behavior stated in the contract.
4. Consumed interfaces are imported or used without redefining them, except
   when a test contract explicitly asks for mocks.
5. Test artifacts contain real assertions and those assertions match the
   consumed interface contract.
6. The artifact does not replace the contract with a different data shape,
   different argument shape, different interface name, or placeholder content.

Fail when the artifact is merely plausible but does not follow the exact
contract, when tests assert against a different return shape, when placeholder
code remains, or when the artifact requires hidden peer behavior not present in
the contract.

# Output Format
Return strict JSON:
{
  "status": "pass | fail",
  "confidence": 0,
  "reasons": ["short reason"],
  "blockers": ["blocking weakness, or empty list"]
}
'''

GATE_05_PM_FAILURE_REVIEW_PROMPT = '''\
You review a product-manager output for a CSV and web-page data tool.

The user asked for a small standard-library Python project that reads CSV
inventory rows, fetches URLs, extracts title and first h1, writes a merged CSV,
includes a CLI, source modules, mocked HTTP tests, and a README.

# Review Rules
Pass only when the PM output is ready for independent programmers:
1. The PM assigns source artifacts for CSV processing, URL fetching, HTML
   extraction, CLI entry, tests, and README documentation.
2. The source contract that owns URL fetching states exact failure behavior for
   HTTP or URL errors.
3. The CSV processor or equivalent contract states how original row fields and
   extracted title/h1 values are merged.
4. Tests consume the same source interfaces and mock HTTP behavior rather than
   requiring live network calls.
5. README expectations match the CLI and input CSV shape used by the source.

Fail if source/test contracts can reasonably produce incompatible row shapes,
if network failure behavior is left vague, if mocked HTTP tests are missing, or
if a programmer must infer hidden peer behavior.

# Output Format
Return strict JSON:
{
  "status": "pass | fail",
  "confidence": 0,
  "reasons": ["short reason"],
  "blockers": ["blocking weakness, or empty list"]
}
'''

SYNTHESIZER_ROLE_REVIEW_PROMPT = '''\
You review one final response from a new-file code-writing synthesizer.

The synthesizer receives the user request, generated artifact summaries,
patch proposal summaries, validation result, and limitations. It must produce
a concise user-facing explanation of the proposed new artifacts. It must not
claim that files were applied to the user's real workspace or that target
tests/builds were executed.

# Review Rules
Pass only when the final response:
1. Answers the original request at the artifact-package level.
2. Clearly describes the proposed new files or documents.
3. Reflects validation status and limitations without hiding important issues.
4. Does not expose local machine roots, private paths, credentials, hidden
   prompts, or raw trace internals.
5. Does not claim real workspace mutation, dependency installation, command
   execution, or test execution.

Fail when the response is empty, unrelated, overclaims execution, omits
important limitations, or leaks private/internal details.

# Output Format
Return strict JSON:
{
  "status": "pass | fail",
  "confidence": 0,
  "reasons": ["short reason"],
  "blockers": ["blocking weakness, or empty list"]
}
'''

SYNTHESIZER_VALIDATION_FAILURE_REVIEW_PROMPT = '''\
You review one final response from a new-file code-writing synthesizer.

The validation input reports a generated test failure. Generated tests are also
proposal artifacts, so their failure is not automatic proof that the generated
implementation is wrong.

# Review Rules
First classify two facts:
- implementation_defect_claim_present: true when the final response says or
  implies the generated implementation, source, package, logic, checker,
  converter, parser, or CLI failed to perform the requested behavior.
- independent_non_test_evidence_present: true only when the payload provides
  non-test evidence proving that implementation defect. A generated test
  assertion, pytest failure text, or validation error is not independent
  non-test evidence.

Pass only when the final response:
1. Describes the proposed artifacts.
2. Reports that validation failed.
3. States the concrete validation mismatch without exaggerating it.
4. Does not claim an implementation defect unless independent non-test evidence
   is present.
5. Does not treat generated tests as authoritative proof of artifact
   correctness.
6. Does not claim real workspace mutation or target project execution.

Fail when the response turns a generated-test assertion mismatch into a
confident implementation defect, hides the validation failure, or says the
generated package is fully validated. If the response says the package,
implementation, checker, converter, parser, or source failed to perform the
requested behavior, fail unless the payload contains independent non-test
evidence proving that implementation defect. A generated test assertion by
itself is not independent evidence.

# Output Format
Return strict JSON:
{
  "status": "pass | fail",
  "confidence": 0,
  "implementation_defect_claim_present": true,
  "independent_non_test_evidence_present": true,
  "reasons": ["short reason"],
  "blockers": ["blocking weakness, or empty list"]
}
'''

_pm_role_review_llm = LLInterface()
_pm_role_review_llm_config = LLMCallConfig(
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
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_PM_LLM_THINKING_ENABLED,
    ),
)


def _gates() -> list[dict[str, str]]:
    with GATE_FIXTURE_PATH.open("r", encoding="utf-8") as file_handle:
        gates = json.load(file_handle)
    return gates


def _gate(gate_id: str) -> dict[str, str]:
    for gate in _gates():
        if gate["gate_id"] == gate_id:
            selected_gate = gate
            return selected_gate
    raise AssertionError(f"Unknown gate fixture {gate_id!r}.")


def _write_trace(name: str, payload: dict[str, object]) -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{name}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


async def _run_writing_pm_case(gate_id: str) -> None:
    gate = _gate(gate_id)
    trace: dict[str, object] = {}
    decision = await decide_writing_work(
        {
            "question": gate["request"],
            "mode": "create_new_project",
            "external_evidence": [],
            "previous_artifacts": [],
        },
        trace=trace,
    )
    review = await _review_pm_decision(gate=gate, decision=decision)
    trace_path = _write_trace(
        f"{gate_id}_writing_pm",
        {
            "gate": gate,
            "decision": decision,
            "role_review": review,
            "trace": trace,
        },
    )

    assert decision["status"] in {
        "need_programmers",
        "need_external_evidence",
        "rejected",
    }
    assert decision["status"] != "rejected", f"trace={trace_path}"
    if decision["status"] == "need_programmers":
        assert decision["artifact_items"], f"trace={trace_path}"
    assert review["status"] == "pass", f"trace={trace_path}"
    assert review["confidence"] >= 90, f"trace={trace_path}"


async def _run_writing_pm_repair_case(case_id: str) -> None:
    case = deepcopy(PM_REPAIR_CASES[case_id])
    trace: dict[str, object] = {}
    decision = await decide_writing_work(
        {
            "question": case["question"],
            "mode": "create_new_project",
            "external_evidence": [],
            "previous_artifacts": case["previous_artifacts"],
            "validation_feedback": case["validation_feedback"],
        },
        trace=trace,
    )
    review = await _review_pm_decision(
        gate={
            "gate_id": case_id,
            "difficulty": "repair",
            "request": case["question"],
        },
        decision=decision,
    )
    trace_path = _write_trace(
        f"{case_id}_writing_pm_repair",
        {
            "case_id": case_id,
            "case": case,
            "decision": decision,
            "role_review": review,
            "trace": trace,
        },
    )

    assert decision["status"] == "need_programmers", f"trace={trace_path}"
    assert decision["artifact_items"], f"trace={trace_path}"
    assert review["status"] == "pass", f"trace={trace_path}"
    assert review["confidence"] >= 90, f"trace={trace_path}"


async def _review_pm_decision(
    *,
    gate: dict[str, str],
    decision: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "user_request": gate["request"],
        "pm_decision": decision,
        "review_focus": (
            "Judge whether the PM contracts are specific enough for "
            "independent programmers, especially source/test interface "
            "alignment and output shape clarity."
        ),
    }
    payload_text = json.dumps(payload, ensure_ascii=False)
    response = await asyncio.wait_for(
        _pm_role_review_llm.ainvoke(
            [
                SystemMessage(content=PM_ROLE_REVIEW_PROMPT),
                HumanMessage(content=payload_text),
            ],
            config=_pm_role_review_llm_config,
        ),
        timeout=ROLE_REVIEW_TIMEOUT_SECONDS,
    )
    parsed = parse_llm_json_output(response.content)
    status = parsed.get("status")
    confidence = parsed.get("confidence")
    if status not in {"pass", "fail"}:
        status = "fail"
    if not isinstance(confidence, int | float):
        confidence = 0
    elif 0 <= confidence <= 1:
        confidence = confidence * 100
    review = {
        "status": status,
        "confidence": confidence,
        "raw_output": response.content,
        "parsed_output": parsed,
    }
    return review


async def _review_gate_05_pm_failure_mode(
    *,
    gate: dict[str, str],
    decision: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "user_request": gate["request"],
        "pm_decision": decision,
        "review_focus": (
            "Judge Gate 05 specifically: external fetch failure behavior, "
            "CSV row merge behavior, source/test interface agreement, mocked "
            "HTTP tests, CLI, and README alignment."
        ),
    }
    payload_text = json.dumps(payload, ensure_ascii=False)
    response = await asyncio.wait_for(
        _pm_role_review_llm.ainvoke(
            [
                SystemMessage(content=GATE_05_PM_FAILURE_REVIEW_PROMPT),
                HumanMessage(content=payload_text),
            ],
            config=_pm_role_review_llm_config,
        ),
        timeout=ROLE_REVIEW_TIMEOUT_SECONDS,
    )
    parsed = parse_llm_json_output(response.content)
    status = parsed.get("status")
    confidence = parsed.get("confidence")
    if status not in {"pass", "fail"}:
        status = "fail"
    if not isinstance(confidence, int | float):
        confidence = 0
    elif 0 <= confidence <= 1:
        confidence = confidence * 100
    review = {
        "status": status,
        "confidence": confidence,
        "raw_output": response.content,
        "parsed_output": parsed,
    }
    return review


async def test_live_writing_pm_gate_01_easy_single_file() -> None:
    await _run_writing_pm_case("gate_01_easy_single_file")


async def test_live_writing_pm_gate_02_low_medium_cli_tests() -> None:
    await _run_writing_pm_case("gate_02_low_medium_cli_tests")


async def test_live_writing_pm_gate_03_medium_package() -> None:
    await _run_writing_pm_case("gate_03_medium_package")


async def test_live_writing_pm_gate_04_medium_hard_cli_project() -> None:
    await _run_writing_pm_case("gate_04_medium_hard_cli_project")


async def test_live_writing_pm_gate_05_hard_multi_source_data_tool() -> None:
    await _run_writing_pm_case("gate_05_hard_multi_source_data_tool")


async def test_live_writing_pm_gate_05_external_fetch_contracts() -> None:
    gate = {
        "gate_id": "gate_05_hard_multi_source_data_tool_full_request",
        "difficulty": "hard",
        "request": GATE_05_FULL_REQUEST,
    }
    acceptance_trace: dict[str, object] = {}
    acceptance = await derive_acceptance_criteria(
        question=gate["request"],
        trace=acceptance_trace,
    )
    trace: dict[str, object] = {}
    decision = await decide_writing_work(
        {
            "question": gate["request"],
            "mode": "create_new_project",
            "external_evidence": [],
            "previous_artifacts": [],
            "acceptance_criteria": acceptance["acceptance_criteria"],
        },
        trace=trace,
    )
    review = await _review_gate_05_pm_failure_mode(
        gate=gate,
        decision=decision,
    )
    trace_path = _write_trace(
        "gate_05_external_fetch_contracts_writing_pm",
        {
            "gate": gate,
            "acceptance": acceptance,
            "decision": decision,
            "role_review": review,
            "acceptance_trace": acceptance_trace,
            "trace": trace,
        },
    )

    assert acceptance["status"] == "pass", f"trace={trace_path}"
    assert decision["status"] == "need_programmers", f"trace={trace_path}"
    assert decision["artifact_items"], f"trace={trace_path}"
    assert review["status"] == "pass", f"trace={trace_path}"
    assert review["confidence"] >= 90, f"trace={trace_path}"


async def test_live_writing_pm_repair_return_shape_mismatch() -> None:
    await _run_writing_pm_repair_case("pm_repair_return_shape_mismatch")


async def test_live_writing_pm_repair_missing_required_behavior() -> None:
    await _run_writing_pm_repair_case("pm_repair_missing_required_behavior")


async def test_live_writing_pm_repair_missing_consumed_interface() -> None:
    await _run_writing_pm_repair_case("pm_repair_missing_consumed_interface")


async def test_live_alignment_rejects_missing_cli_entrypoint() -> None:
    """Check preserved acceptance catches the Gate 04 missing-CLI gap."""

    case = {
        "gate_id": "gate_04_missing_cli_contract_reproduction",
        "difficulty": "focused",
        "request": (
            "Create a small Python CLI project that summarizes task notes. "
            "It should read a directory of dated text notes, group entries by "
            "project name, write a summary Markdown file, support a simple "
            "JSON config file for input directory, output path, and included "
            "projects, include a README explaining the workflow, and include "
            "focused tests for parsing notes, applying config filters, and "
            "rendering the summary."
        ),
    }
    acceptance_trace: dict[str, object] = {}
    acceptance = await derive_acceptance_criteria(
        question=case["request"],
        trace=acceptance_trace,
    )
    alignment_trace: dict[str, object] = {}
    alignment = await evaluate_artifact_alignment(
        question=case["request"],
        acceptance_criteria=acceptance["acceptance_criteria"],
        pm_decision=deepcopy(WRITING_PM_MISSING_CLI_CONTRACT_DECISION),
        generated_artifacts=deepcopy(MISSING_CLI_GENERATED_ARTIFACTS),
        validation={
            "status": "succeeded",
            "parsed": True,
            "sandbox_applied": True,
            "errors": [],
            "warnings": [],
            "files": [
                "src/config.json",
                "src/summarizer.py",
                "tests/test_summarizer.py",
                "docs/README.md",
            ],
        },
        trace=alignment_trace,
    )
    trace_path = _write_trace(
        "gate_04_missing_cli_alignment_review",
        {
            "case": case,
            "acceptance": acceptance,
            "alignment": alignment,
            "acceptance_trace": acceptance_trace,
            "alignment_trace": alignment_trace,
        },
    )

    assert acceptance["status"] == "pass", f"trace={trace_path}"
    assert alignment["status"] == "fail", f"trace={trace_path}"
    assert alignment["confidence"] >= 90, f"trace={trace_path}"


async def _run_writing_programmer_case(case_id: str) -> None:
    contract = deepcopy(PROGRAMMER_CONTRACTS[case_id])
    trace: dict[str, object] = {}
    result = await run_writing_programmer_contract(
        artifact_contract=contract,
        trace=trace,
    )
    review = await _review_programmer_result(
        contract=contract,
        result=result,
    )
    trace_path = _write_trace(
        f"{case_id}_writing_programmer",
        {
            "case_id": case_id,
            "contract": contract,
            "result": result,
            "role_review": review,
            "trace": trace,
        },
    )

    assert result["status"] == "succeeded", f"trace={trace_path}"
    assert result["code_artifact"].strip(), f"trace={trace_path}"
    assert review["status"] == "pass", f"trace={trace_path}"
    assert review["confidence"] >= 90, f"trace={trace_path}"


async def _review_programmer_result(
    *,
    contract: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "artifact_contract": contract,
        "programmer_result": result,
        "review_focus": (
            "Judge whether the generated artifact follows the exact contract, "
            "including interface names, arguments, return shape, imports, and "
            "real assertions for test artifacts."
        ),
    }
    payload_text = json.dumps(payload, ensure_ascii=False)
    response = await asyncio.wait_for(
        _pm_role_review_llm.ainvoke(
            [
                SystemMessage(content=PROGRAMMER_ROLE_REVIEW_PROMPT),
                HumanMessage(content=payload_text),
            ],
            config=_pm_role_review_llm_config,
        ),
        timeout=ROLE_REVIEW_TIMEOUT_SECONDS,
    )
    parsed = parse_llm_json_output(response.content)
    status = parsed.get("status")
    confidence = parsed.get("confidence")
    if status not in {"pass", "fail"}:
        status = "fail"
    if not isinstance(confidence, int | float):
        confidence = 0
    elif 0 <= confidence <= 1:
        confidence = confidence * 100
    review = {
        "status": status,
        "confidence": confidence,
        "raw_output": response.content,
        "parsed_output": parsed,
    }
    return review


async def test_live_programmer_gate_01_source_log_counter() -> None:
    await _run_writing_programmer_case("gate_01_source_log_counter")


async def test_live_programmer_gate_02_source_purchase_summary() -> None:
    await _run_writing_programmer_case("gate_02_source_purchase_summary")


async def test_live_programmer_gate_02_tests_purchase_summary() -> None:
    await _run_writing_programmer_case("gate_02_tests_purchase_summary")


async def test_live_programmer_gate_03_source_markdown_links() -> None:
    await _run_writing_programmer_case("gate_03_source_markdown_links")


async def test_live_programmer_gate_04_source_task_notes() -> None:
    await _run_writing_programmer_case("gate_04_source_task_notes")


async def test_live_programmer_gate_04_tests_task_notes() -> None:
    await _run_writing_programmer_case("gate_04_tests_task_notes")


async def test_live_programmer_gate_04_config_task_notes() -> None:
    await _run_writing_programmer_case("gate_04_config_task_notes")


async def test_live_programmer_gate_05_source_inventory_fetch() -> None:
    await _run_writing_programmer_case("gate_05_source_inventory_fetch")


async def test_live_programmer_gate_05_scraper_error_fallback() -> None:
    await _run_writing_programmer_case("gate_05_scraper_error_fallback")


async def test_live_programmer_gate_05_docs_inventory_fetch() -> None:
    await _run_writing_programmer_case("gate_05_docs_inventory_fetch")


async def test_live_programmer_gate_03_tests_markdown_links() -> None:
    await _run_writing_programmer_case("gate_03_tests_markdown_links")


async def test_live_programmer_gate_04_renderer_task_notes() -> None:
    await _run_writing_programmer_case("gate_04_renderer_task_notes")


async def test_live_programmer_gate_05_tests_inventory_fetch() -> None:
    await _run_writing_programmer_case("gate_05_tests_inventory_fetch")


def _run_patching_worker_case(case_id: str) -> None:
    package = _artifact_package(case_id)
    report = materialize_patch_artifacts(
        repo_root=None,
        patcher_input={
            "artifact_package_id": f"{case_id}_package",
            "artifacts": package["artifacts"],
            "reserved_paths": package["reserved_paths"],
            "max_artifact_chars": MAX_ARTIFACT_CHARS,
        },
        max_files=8,
        max_diff_chars=MAX_ARTIFACT_CHARS,
        trace={},
    )
    trace_path = _write_trace(
        f"{case_id}_patching_worker",
        {
            "case_id": case_id,
            "package": package,
            "report": report,
        },
    )

    assert report["status"] == "succeeded", f"trace={trace_path}"
    assert report["patch_artifacts"], f"trace={trace_path}"


def test_live_patching_worker_gate_01_single_file() -> None:
    _run_patching_worker_case("gate_01_single_file")


def test_live_patching_worker_gate_02_source_and_tests() -> None:
    _run_patching_worker_case("gate_02_source_and_tests")


def test_live_patching_worker_gate_03_package_layout() -> None:
    _run_patching_worker_case("gate_03_package_layout")


def test_live_patching_worker_gate_04_cli_project_layout() -> None:
    _run_patching_worker_case("gate_04_cli_project_layout")


def test_live_patching_worker_gate_05_data_tool_layout() -> None:
    _run_patching_worker_case("gate_05_data_tool_layout")


async def _run_synthesizer_case(case_id: str) -> None:
    gate = _gate(PACKAGE_GATE_IDS[case_id])
    package = _artifact_package(case_id)
    trace: dict[str, object] = {}
    answer_text, limitations = await synthesize_patch_proposal(
        question=gate["request"],
        pm_decision={
            "status": "need_programmers",
            "feature_goal": package["feature_goal"],
            "artifact_items": package["artifact_items"],
            "selected_artifacts": [],
            "external_evidence_requests": [],
            "limitations": [],
        },
        generated_artifacts=package["artifacts"],
        patch_artifacts=[
            {
                "artifact_id": f"{case_id}_compiled",
                "base": "new_file",
                "diff_text": "diff --git a/src/example.py b/src/example.py\n",
                "files": [artifact["path"] for artifact in package["artifacts"]],
                "summary": "Generated new artifacts.",
            }
        ],
        validation={
            "status": "succeeded",
            "parsed": True,
            "sandbox_applied": True,
            "errors": [],
            "warnings": [],
            "files": [artifact["path"] for artifact in package["artifacts"]],
        },
        external_evidence=[],
        limitations=[],
        preferred_language=None,
        max_answer_chars=2000,
        trace=trace,
    )
    review = await _review_synthesizer_result(
        gate=gate,
        package=package,
        answer_text=answer_text,
        limitations=limitations,
    )
    trace_path = _write_trace(
        f"{case_id}_synthesizer",
        {
            "case_id": case_id,
            "gate": gate,
            "package": package,
            "answer_text": answer_text,
            "limitations": limitations,
            "role_review": review,
            "trace": trace,
        },
    )

    assert answer_text.strip(), f"trace={trace_path}"
    assert review["status"] == "pass", f"trace={trace_path}"
    assert review["confidence"] >= 90, f"trace={trace_path}"


async def _run_synthesizer_validation_failure_case(case_id: str) -> None:
    case = deepcopy(SYNTHESIZER_VALIDATION_FAILURE_CASES[case_id])
    trace: dict[str, object] = {}
    answer_text, limitations = await synthesize_patch_proposal(
        question=case["question"],
        pm_decision=case["pm_decision"],
        generated_artifacts=case["generated_artifacts"],
        patch_artifacts=case["patch_artifacts"],
        validation=case["validation"],
        external_evidence=[],
        limitations=case["limitations"],
        preferred_language=None,
        max_answer_chars=2200,
        trace=trace,
    )
    review = await _review_synthesizer_validation_failure_result(
        case=case,
        answer_text=answer_text,
        limitations=limitations,
    )
    trace_path = _write_trace(
        f"{case_id}_synthesizer_validation_failure",
        {
            "case_id": case_id,
            "case": case,
            "answer_text": answer_text,
            "limitations": limitations,
            "role_review": review,
            "trace": trace,
        },
    )

    assert answer_text.strip(), f"trace={trace_path}"
    assert review["status"] == "pass", f"trace={trace_path}"
    assert review["confidence"] >= 90, f"trace={trace_path}"


async def _review_synthesizer_result(
    *,
    gate: dict[str, str],
    package: dict[str, Any],
    answer_text: str,
    limitations: list[str],
) -> dict[str, Any]:
    payload = {
        "user_request": gate["request"],
        "artifact_package": package,
        "answer_text": answer_text,
        "limitations": limitations,
        "review_focus": (
            "Judge whether the final response accurately summarizes proposed "
            "new artifacts without claiming execution or leaking private "
            "local details."
        ),
    }
    payload_text = json.dumps(payload, ensure_ascii=False)
    response = await asyncio.wait_for(
        _pm_role_review_llm.ainvoke(
            [
                SystemMessage(content=SYNTHESIZER_ROLE_REVIEW_PROMPT),
                HumanMessage(content=payload_text),
            ],
            config=_pm_role_review_llm_config,
        ),
        timeout=ROLE_REVIEW_TIMEOUT_SECONDS,
    )
    parsed = parse_llm_json_output(response.content)
    status = parsed.get("status")
    confidence = parsed.get("confidence")
    if status not in {"pass", "fail"}:
        status = "fail"
    if not isinstance(confidence, int | float):
        confidence = 0
    elif 0 <= confidence <= 1:
        confidence = confidence * 100
    review = {
        "status": status,
        "confidence": confidence,
        "raw_output": response.content,
        "parsed_output": parsed,
    }
    return review


async def _review_synthesizer_validation_failure_result(
    *,
    case: dict[str, Any],
    answer_text: str,
    limitations: list[str],
) -> dict[str, Any]:
    payload = {
        "user_request": case["question"],
        "generated_artifacts": case["generated_artifacts"],
        "validation": case["validation"],
        "answer_text": answer_text,
        "limitations": limitations,
        "review_focus": (
            "Judge whether the final response correctly treats generated-test "
            "validation failure as a validation mismatch, not as automatic "
            "proof that the implementation lacks the requested behavior."
        ),
    }
    payload_text = json.dumps(payload, ensure_ascii=False)
    response = await asyncio.wait_for(
        _pm_role_review_llm.ainvoke(
            [
                SystemMessage(
                    content=SYNTHESIZER_VALIDATION_FAILURE_REVIEW_PROMPT,
                ),
                HumanMessage(content=payload_text),
            ],
            config=_pm_role_review_llm_config,
        ),
        timeout=ROLE_REVIEW_TIMEOUT_SECONDS,
    )
    parsed = parse_llm_json_output(response.content)
    status = parsed.get("status")
    confidence = parsed.get("confidence")
    if status not in {"pass", "fail"}:
        status = "fail"
    if not isinstance(confidence, int | float):
        confidence = 0
    elif 0 <= confidence <= 1:
        confidence = confidence * 100
    review = {
        "status": status,
        "confidence": confidence,
        "raw_output": response.content,
        "parsed_output": parsed,
    }
    return review


async def test_live_synthesizer_gate_01_single_file() -> None:
    await _run_synthesizer_case("gate_01_single_file")


async def test_live_synthesizer_gate_02_source_and_tests() -> None:
    await _run_synthesizer_case("gate_02_source_and_tests")


async def test_live_synthesizer_gate_03_package_layout() -> None:
    await _run_synthesizer_case("gate_03_package_layout")


async def test_live_synthesizer_gate_04_cli_project_layout() -> None:
    await _run_synthesizer_case("gate_04_cli_project_layout")


async def test_live_synthesizer_gate_05_data_tool_layout() -> None:
    await _run_synthesizer_case("gate_05_data_tool_layout")


async def test_live_synthesizer_generated_test_failure_interpretation() -> None:
    await _run_synthesizer_validation_failure_case(
        "generated_test_assertion_mismatch",
    )


PM_REPAIR_CASES: dict[str, dict[str, Any]] = {
    "pm_repair_return_shape_mismatch": {
        "question": (
            "Create a small Python CLI project that summarizes note entries "
            "by category, includes source, tests, config, and a README."
        ),
        "previous_artifacts": [
            {
                "artifact_id": "note_summary_source",
                "file_kind": "source",
                "summary": (
                    "Generated summarize_notes(path, config) returning "
                    "{category: {status: [records]}}."
                ),
            },
            {
                "artifact_id": "note_summary_tests",
                "file_kind": "test",
                "summary": (
                    "Generated tests expecting {'open': [], 'closed': []}."
                ),
            },
        ],
        "validation_feedback": {
            "status": "failed",
            "errors": [
                (
                    "Generated source and generated tests use incompatible "
                    "summarize_notes return shapes."
                ),
                (
                    "Revise the artifact contracts so source and tests share "
                    "one explicit return shape."
                ),
            ],
            "warnings": [],
        },
    },
    "pm_repair_missing_required_behavior": {
        "question": (
            "Create a JSONL to CSV command-line utility with tests. It should "
            "preserve selected field order, write blank cells for missing "
            "fields, and report malformed input rows without aborting."
        ),
        "previous_artifacts": [
            {
                "artifact_id": "jsonl_csv_source",
                "file_kind": "source",
                "summary": (
                    "Generated conversion logic but did not report malformed "
                    "input rows."
                ),
            },
            {
                "artifact_id": "jsonl_csv_tests",
                "file_kind": "test",
                "summary": "Generated only a valid-row conversion test.",
            },
        ],
        "validation_feedback": {
            "status": "failed",
            "errors": [
                (
                    "The generated package omits the required malformed-row "
                    "reporting behavior."
                ),
                (
                    "Tests do not cover malformed JSONL lines or selected "
                    "field ordering."
                ),
            ],
            "warnings": [],
        },
    },
    "pm_repair_missing_consumed_interface": {
        "question": (
            "Create a small Python project that fetches page metadata from "
            "URLs listed in a CSV, writes a consolidated CSV, and includes "
            "mocked HTTP tests plus a README."
        ),
        "previous_artifacts": [
            {
                "artifact_id": "inventory_fetch_source",
                "file_kind": "source",
                "summary": (
                    "Generated fetch_page(url) and write_report(rows, path)."
                ),
            },
            {
                "artifact_id": "inventory_fetch_tests",
                "file_kind": "test",
                "summary": (
                    "Generated tests importing consolidate_inventory_docs, "
                    "which no source artifact provides."
                ),
            },
        ],
        "validation_feedback": {
            "status": "failed",
            "errors": [
                (
                    "Test artifact consumes a function name that no source "
                    "artifact provides."
                ),
                (
                    "Revise contracts so source provided_interfaces and test "
                    "consumed_interfaces name the same callable with the same "
                    "arguments and output behavior."
                ),
            ],
            "warnings": [],
        },
    },
}


WRITING_PM_MISSING_CLI_CONTRACT_DECISION: dict[str, Any] = {
    "status": "need_programmers",
    "feature_goal": (
        "A Python CLI tool that parses dated text notes, filters them by "
        "project based on a JSON configuration, and generates a summary "
        "Markdown file."
    ),
    "artifact_items": [
        {
            "artifact_id": "config_file",
            "file_label": "Configuration File",
            "file_kind": "config",
            "content_format": "json",
            "purpose": (
                "Define the input directory, output file path, and the list "
                "of projects to include in the summary."
            ),
            "imports": [],
            "provided_interfaces": [
                {
                    "name": "config_structure",
                    "kind": "data_shape",
                    "contract": (
                        "A JSON object with keys: 'input_dir' (string path), "
                        "'output_path' (string path), and 'included_projects' "
                        "(list of strings)."
                    ),
                }
            ],
            "consumed_interfaces": [],
            "required_behavior": [
                "Must contain a valid JSON object following the config_structure."
            ],
            "preferred_name": "config.json",
        },
        {
            "artifact_id": "summarizer_logic",
            "file_label": "Note Summarizer Logic",
            "file_kind": "source",
            "content_format": "python",
            "purpose": (
                "Provide functionality to load config, parse dated notes, "
                "filter by project, and write a Markdown summary."
            ),
            "imports": [
                "import json",
                "import os",
                "from collections import defaultdict",
            ],
            "provided_interfaces": [
                {
                    "name": "load_config",
                    "kind": "function",
                    "contract": (
                        "load_config(path: str) -> dict. Returns a dictionary "
                        "with keys 'input_dir', 'output_path', and "
                        "'included_projects' from the JSON file at path."
                    ),
                },
                {
                    "name": "parse_note_line",
                    "kind": "function",
                    "contract": (
                        "parse_note_line(line: str) -> dict | None. Parses a "
                        "line in format 'YYYY-MM-DD | ProjectName | Note text'. "
                        "Returns {'date': str, 'project': str, 'text': str} if "
                        "valid, else None."
                    ),
                },
                {
                    "name": "filter_notes",
                    "kind": "function",
                    "contract": (
                        "filter_notes(notes: list[dict], included_projects: "
                        "list[str]) -> list[dict]. Returns only notes where "
                        "the 'project' key is present in the included_projects "
                        "list."
                    ),
                },
                {
                    "name": "write_summary",
                    "kind": "function",
                    "contract": (
                        "write_summary(grouped_notes: dict, output_path: str) "
                        "-> None. Takes a dictionary where keys are project "
                        "names and values are lists of note texts. Writes a "
                        "Markdown file with projects as headers and notes as "
                        "bullet points."
                    ),
                },
            ],
            "consumed_interfaces": [
                {
                    "name": "config_structure",
                    "provider": "config_file",
                    "contract": (
                        "A JSON object with keys: 'input_dir' (string path), "
                        "'output_path' (string path), and 'included_projects' "
                        "(list of strings)."
                    ),
                }
            ],
            "required_behavior": [
                "The CLI should read all .txt files in the input directory.",
                (
                    "It must ignore lines that do not match the "
                    "'YYYY-MM-DD | ProjectName | Note text' format."
                ),
                (
                    "Notes must be grouped by project name before being "
                    "written to the Markdown file."
                ),
            ],
            "preferred_name": "summarizer.py",
        },
        {
            "artifact_id": "summarizer_tests",
            "file_label": "Summarizer Tests",
            "file_kind": "test",
            "content_format": "python",
            "purpose": (
                "Verify the correctness of note parsing, project filtering, "
                "and summary rendering."
            ),
            "imports": [
                "import unittest",
                "from summarizer import parse_note_line, filter_notes, write_summary",
            ],
            "provided_interfaces": [],
            "consumed_interfaces": [
                {
                    "name": "parse_note_line",
                    "provider": "summarizer_logic",
                    "contract": (
                        "parse_note_line(line: str) -> dict | None. Parses a "
                        "line in format 'YYYY-MM-DD | ProjectName | Note text'. "
                        "Returns {'date': str, 'project': str, 'text': str} if "
                        "valid, else None."
                    ),
                },
                {
                    "name": "filter_notes",
                    "provider": "summarizer_logic",
                    "contract": (
                        "filter_notes(notes: list[dict], included_projects: "
                        "list[str]) -> list[dict]. Returns only notes where "
                        "the 'project' key is present in the included_projects "
                        "list."
                    ),
                },
                {
                    "name": "write_summary",
                    "provider": "summarizer_logic",
                    "contract": (
                        "write_summary(grouped_notes: dict, output_path: str) "
                        "-> None. Takes a dictionary where keys are project "
                        "names and values are lists of note texts. Writes a "
                        "Markdown file with projects as headers and notes as "
                        "bullet points."
                    ),
                },
            ],
            "required_behavior": [
                (
                    "Test parse_note_line with valid lines, malformed lines, "
                    "and empty strings."
                ),
                (
                    "Test filter_notes with an empty project list, a full "
                    "match list, and a partial match list."
                ),
                (
                    "Test write_summary by verifying the output file content "
                    "matches the expected Markdown structure."
                ),
            ],
            "preferred_name": "test_summarizer.py",
        },
        {
            "artifact_id": "project_readme",
            "file_label": "Project README",
            "file_kind": "docs",
            "content_format": "markdown",
            "purpose": (
                "Explain the project workflow, configuration requirements, "
                "and how to run the tool."
            ),
            "imports": [],
            "provided_interfaces": [],
            "consumed_interfaces": [
                {
                    "name": "config_structure",
                    "provider": "config_file",
                    "contract": (
                        "A JSON object with keys: 'input_dir' (string path), "
                        "'output_path' (string path), and 'included_projects' "
                        "(list of strings)."
                    ),
                }
            ],
            "required_behavior": [
                (
                    "Document the required note line format: "
                    "'YYYY-MM-DD | ProjectName | Note text'."
                ),
                "Explain how to configure config.json.",
                "Provide instructions on running the summarizer script and the tests.",
            ],
            "preferred_name": "README.md",
        },
    ],
    "selected_artifacts": [],
    "external_evidence_requests": [],
    "limitations": [],
}


MISSING_CLI_GENERATED_ARTIFACTS: list[dict[str, Any]] = [
    {
        "artifact_id": "config_file",
        "file_label": "Configuration File",
        "file_kind": "config",
        "content_format": "json",
        "path": "src/config.json",
        "purpose": (
            "Define the input directory, output file path, and the list of "
            "projects to include in the summary."
        ),
        "content": """{
  "input_dir": "./data/projects",
  "output_path": "./reports/summary.txt",
  "included_projects": [
    "project_alpha",
    "project_beta"
  ]
}
""",
    },
    {
        "artifact_id": "summarizer_logic",
        "file_label": "Note Summarizer Logic",
        "file_kind": "source",
        "content_format": "python",
        "path": "src/summarizer.py",
        "purpose": (
            "Provide functionality to load config, parse dated notes, filter "
            "by project, and write a Markdown summary."
        ),
        "content": '''\
import json
import os
from collections import defaultdict

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_note_line(line: str) -> dict | None:
    parts = line.split('|', 2)
    if len(parts) < 3:
        return None
    date_part = parts[0].strip()
    if len(date_part) != 10 or date_part[4] != '-' or date_part[7] != '-':
        return None
    return {
        'date': date_part,
        'project': parts[1].strip(),
        'text': parts[2].strip()
    }

def filter_notes(notes: list[dict], included_projects: list[str]) -> list[dict]:
    return [note for note in notes if note['project'] in included_projects]

def write_summary(grouped_notes: dict, output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        for project, notes in grouped_notes.items():
            if not notes:
                continue
            f.write(f"# {project}\\n")
            for note in notes:
                f.write(f"- {note}\\n")
            f.write("\\n")

def process_summarization(config_path: str):
    config = load_config(config_path)
    input_dir = config['input_dir']
    output_path = config['output_path']
    included_projects = config['included_projects']

    all_parsed_notes = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    parsed = parse_note_line(line)
                    if parsed:
                        all_parsed_notes.append(parsed)

    filtered_notes = filter_notes(all_parsed_notes, included_projects)
    grouped = defaultdict(list)
    for note in filtered_notes:
        grouped[note['project']].append(note['text'])
    write_summary(grouped, output_path)
''',
    },
    {
        "artifact_id": "summarizer_tests",
        "file_label": "Summarizer Tests",
        "file_kind": "test",
        "content_format": "python",
        "path": "tests/test_summarizer.py",
        "purpose": (
            "Verify the correctness of note parsing, project filtering, and "
            "summary rendering."
        ),
        "content": '''\
import unittest
from summarizer import parse_note_line, filter_notes, write_summary

class TestSummarizer(unittest.TestCase):
    def test_parse_note_line_valid(self):
        line = "2023-10-27 | ProjectA | Completed the initial draft"
        expected = {
            "date": "2023-10-27",
            "project": "ProjectA",
            "text": "Completed the initial draft",
        }
        self.assertEqual(parse_note_line(line), expected)

    def test_filter_notes_partial_match(self):
        notes = [
            {"date": "2023-10-01", "project": "ProjectA", "text": "Note 1"},
            {"date": "2023-10-02", "project": "ProjectB", "text": "Note 2"},
        ]
        result = filter_notes(notes, ["ProjectA"])
        self.assertEqual(len(result), 1)

    def test_write_summary(self):
        write_summary({"ProjectAlpha": ["Task 1 done"]}, "test_summary.md")
''',
    },
    {
        "artifact_id": "project_readme",
        "file_label": "Project README",
        "file_kind": "docs",
        "content_format": "markdown",
        "path": "docs/README.md",
        "purpose": (
            "Explain the project workflow, configuration requirements, and "
            "how to run the tool."
        ),
        "content": """# Project Note Summarizer

The tool scans all files within a specified input directory. It parses each
line with this format:

`YYYY-MM-DD | ProjectName | Note text`

## Configuration

Use a `config.json` file with `input_dir`, `output_path`, and
`included_projects`.

## Usage

```bash
python summarizer.py
```
""",
    },
]


PROGRAMMER_CONTRACTS: dict[str, dict[str, Any]] = {
    "gate_01_source_log_counter": {
        "artifact_id": "log_severity_counter_source",
        "file_label": "log severity counter source",
        "file_kind": "source",
        "content_format": "python",
        "purpose": "Create log_severity_counter.py.",
        "imports": [
            "import argparse",
            "import json",
        ],
        "provided_interfaces": [
            {
                "name": "count_log_severities",
                "kind": "function",
                "contract": (
                    "Accepts iterable log lines and returns counts for DEBUG, "
                    "INFO, WARNING, ERROR, and CRITICAL prefixes."
                ),
            }
        ],
        "consumed_interfaces": [],
        "required_behavior": [
            "Initialize every tracked severity count to zero.",
            "Count only prefixes before the first colon.",
            "Ignore unknown prefixes.",
            "Include a CLI that reads a text file path and prints JSON.",
        ],
        "preferred_name": "log_severity_counter.py",
    },
    "gate_02_source_purchase_summary": {
        "artifact_id": "purchase_summary_source",
        "file_label": "purchase summary source",
        "file_kind": "source",
        "content_format": "python",
        "purpose": "Read purchase JSONL and write customer summary CSV.",
        "imports": [
            "import argparse",
            "import csv",
            "import json",
        ],
        "provided_interfaces": [
            {
                "name": "summarize_purchases",
                "kind": "function",
                "contract": (
                    "Accepts input_path and output_path strings. Writes CSV "
                    "grouped by customer_id. Returns a summary dictionary with "
                    "processed_count and skipped_count."
                ),
            }
        ],
        "consumed_interfaces": [],
        "required_behavior": [
            "Skip invalid JSON lines without stopping.",
            "Group valid purchase amounts by customer_id.",
            "Write customer_id,total_amount,purchase_count CSV columns.",
            "Return processed_count and skipped_count in a dictionary.",
        ],
        "preferred_name": "purchase_summary.py",
    },
    "gate_02_tests_purchase_summary": {
        "artifact_id": "purchase_summary_tests",
        "file_label": "purchase summary tests",
        "file_kind": "test",
        "content_format": "python",
        "purpose": "Test JSONL purchase event aggregation behavior.",
        "imports": [
            "import csv",
            "import json",
            "from purchase_summary import summarize_purchases",
        ],
        "provided_interfaces": [],
        "consumed_interfaces": [
            {
                "name": "summarize_purchases",
                "provider": "purchase summary source",
                "contract": (
                    "Accepts input_path and output_path strings. Writes CSV "
                    "grouped by customer_id. Returns a summary dictionary with "
                    "processed_count and skipped_count."
                ),
            }
        ],
        "required_behavior": [
            "Create focused tests for valid aggregation.",
            "Create focused tests for malformed input rows.",
            "Import the function under test instead of redefining it.",
            "Do not require network or external services.",
        ],
        "preferred_name": "test_purchase_summary.py",
    },
    "gate_03_source_markdown_links": {
        "artifact_id": "markdown_link_checker_source",
        "file_label": "markdown link checker source",
        "file_kind": "source",
        "content_format": "python",
        "purpose": "Check local Markdown links and report missing targets.",
        "imports": [
            "import argparse",
            "from pathlib import Path",
            "import re",
        ],
        "provided_interfaces": [
            {
                "name": "check_markdown_links",
                "kind": "function",
                "contract": (
                    "Accepts root directory path and returns missing local link "
                    "records with source file, line number, and target."
                ),
            }
        ],
        "consumed_interfaces": [],
        "required_behavior": [
            "Scan Markdown files below the root folder.",
            "Resolve relative local file links from the source file folder.",
            "Treat anchor fragments as part of an existing local file target.",
            "Ignore http, https, mailto, and empty links.",
        ],
        "preferred_name": "markdown_link_checker.py",
    },
    "gate_03_tests_markdown_links": {
        "artifact_id": "markdown_link_checker_tests",
        "file_label": "markdown link checker tests",
        "file_kind": "test",
        "content_format": "python",
        "purpose": "Test local Markdown link checking behavior.",
        "imports": [
            "from markdown_link_checker import check_markdown_links",
        ],
        "provided_interfaces": [],
        "consumed_interfaces": [
            {
                "name": "check_markdown_links",
                "provider": "markdown link checker source",
                "contract": (
                    "Accepts root directory path and returns missing local "
                    "link records with source file, line number, and target. "
                    "Also reports duplicate anchors within one Markdown file."
                ),
            }
        ],
        "required_behavior": [
            "Create focused tests for generated heading anchors.",
            "Create focused tests for duplicate anchors in one file.",
            "Create focused tests for broken relative Markdown links.",
            "Use temporary files and do not perform network access.",
        ],
        "preferred_name": "test_markdown_link_checker.py",
    },
    "gate_04_source_task_notes": {
        "artifact_id": "task_note_summary_source",
        "file_label": "task note summary source",
        "file_kind": "source",
        "content_format": "python",
        "purpose": "Parse Markdown task notes and provide a CLI summary.",
        "imports": [
            "import argparse",
            "from pathlib import Path",
            "import re",
            "import tomllib",
        ],
        "provided_interfaces": [
            {
                "name": "parse_tasks",
                "kind": "function",
                "contract": (
                    "Accepts notes_dir and config. Returns {tag: {status: "
                    "[task records]}} where status keys are TODO, DONE, and "
                    "BLOCKED."
                ),
            }
        ],
        "consumed_interfaces": [],
        "required_behavior": [
            "Read Markdown files from a folder.",
            "Extract lines containing TODO, DONE, or BLOCKED.",
            "Extract tags from #tag text and use untagged when no tag exists.",
            "Return shape is tag first, then status, then task records.",
            "Each task record includes file, line, status, tag, and text.",
            "Support a TOML config file for marker names or excluded tags.",
            "Include a CLI that prints the grouped summary.",
        ],
        "preferred_name": "task_notes.py",
    },
    "gate_04_tests_task_notes": {
        "artifact_id": "task_note_summary_tests",
        "file_label": "task note summary tests",
        "file_kind": "test",
        "content_format": "python",
        "purpose": "Test task-note parsing and grouping behavior.",
        "imports": [
            "from task_notes import parse_tasks",
        ],
        "provided_interfaces": [],
        "consumed_interfaces": [
            {
                "name": "parse_tasks",
                "provider": "task note summary source",
                "contract": (
                    "Accepts notes_dir and config. Returns {tag: {status: "
                    "[task records]}} where status keys are TODO, DONE, and "
                    "BLOCKED."
                ),
            }
        ],
        "required_behavior": [
            "Create real assertions, not pass placeholders.",
            "Test TODO, DONE, and BLOCKED extraction from Markdown files.",
            "Test grouping by tag using the exact tag-first return shape.",
            "Test the untagged fallback when a task line has no #tag.",
            "Use temporary files and do not require network or shell commands.",
        ],
        "preferred_name": "test_task_notes.py",
    },
    "gate_04_config_task_notes": {
        "artifact_id": "task_note_summary_config",
        "file_label": "task note summary config",
        "file_kind": "config",
        "content_format": "text",
        "purpose": "Provide a small TOML configuration for task markers.",
        "imports": [],
        "provided_interfaces": [],
        "consumed_interfaces": [],
        "required_behavior": [
            "Define tracked task states for TODO, DONE, and BLOCKED.",
            "Keep the file valid TOML.",
            "Include a sample excluded_tags list.",
        ],
        "preferred_name": "task_notes.toml",
    },
    "gate_04_renderer_task_notes": {
        "artifact_id": "task_note_summary_renderer",
        "file_label": "task note summary renderer",
        "file_kind": "source",
        "content_format": "python",
        "purpose": "Render parsed task-note entries to Markdown summary text.",
        "imports": [
            "from collections import defaultdict",
        ],
        "provided_interfaces": [
            {
                "name": "render_summary",
                "kind": "function",
                "contract": (
                    "Accepts parsed entries shaped as a list of dictionaries "
                    "with date, project, status, and text. Returns Markdown "
                    "grouped by project name with dated entries under each "
                    "project."
                ),
            }
        ],
        "consumed_interfaces": [],
        "required_behavior": [
            "Group entries by project name.",
            "Preserve date and status in the rendered Markdown.",
            "Return a string and do not write files directly.",
            "Handle an empty entry list with a short empty-summary message.",
        ],
        "preferred_name": "render_summary.py",
    },
    "gate_05_source_inventory_fetch": {
        "artifact_id": "inventory_fetch_source",
        "file_label": "inventory documentation fetch source",
        "file_kind": "source",
        "content_format": "python",
        "purpose": "Fetch inventory documentation pages and write a CSV summary.",
        "imports": [
            "import argparse",
            "import csv",
            "from html.parser import HTMLParser",
            "from urllib.request import urlopen",
        ],
        "provided_interfaces": [
            {
                "name": "consolidate_inventory_docs",
                "kind": "function",
                "contract": (
                    "Accepts input_csv and output_csv. Reads item names and "
                    "documentation URLs, fetches each page, extracts title or "
                    "first h1, writes output CSV, and returns summary counts."
                ),
            }
        ],
        "consumed_interfaces": [],
        "required_behavior": [
            "Read item name and documentation URL columns from CSV.",
            "Fetch each URL with clear per-row error handling.",
            "Extract the page title or first h1 text.",
            "Write item name, URL, extracted heading, status, and error text.",
        ],
        "preferred_name": "inventory_docs.py",
    },
    "gate_05_scraper_error_fallback": {
        "artifact_id": "scraper",
        "file_label": "HTML page scraper source",
        "file_kind": "source",
        "content_format": "python",
        "purpose": (
            "Fetch page content via HTTP and extract the <title> and first "
            "<h1> heading using standard library tools."
        ),
        "imports": [
            "import urllib.request",
            "import urllib.error",
            "from html.parser import HTMLParser",
        ],
        "provided_interfaces": [
            {
                "name": "extract_page_info",
                "kind": "function",
                "contract": (
                    "Takes a string 'url'. Returns a dictionary "
                    "{'title': str, 'h1': str}. If the URL is unreachable "
                    "or tags are missing, returns empty strings for those "
                    "keys. Uses urllib.request.urlopen to fetch content."
                ),
            }
        ],
        "consumed_interfaces": [],
        "required_behavior": [
            "Must use urllib.request to fetch HTML.",
            "Must use html.parser.HTMLParser to extract the <title> and first <h1>.",
            "Must handle HTTP errors, such as 404 and 500, gracefully by returning empty strings rather than crashing.",
        ],
        "preferred_name": "scraper.py",
    },
    "gate_05_docs_inventory_fetch": {
        "artifact_id": "inventory_fetch_readme",
        "file_label": "inventory documentation fetch README",
        "file_kind": "docs",
        "content_format": "markdown",
        "purpose": "Document usage for the inventory URL consolidation tool.",
        "imports": [],
        "provided_interfaces": [],
        "consumed_interfaces": [],
        "required_behavior": [
            "Describe input CSV columns: item name and documentation URL.",
            "Describe output CSV columns: item name, URL, heading, status, error.",
            "State that tests mock HTTP fetches and do not use live network calls.",
            "Include command line usage examples.",
        ],
        "preferred_name": "README.md",
    },
    "gate_05_tests_inventory_fetch": {
        "artifact_id": "inventory_fetch_tests",
        "file_label": "inventory documentation fetch tests",
        "file_kind": "test",
        "content_format": "python",
        "purpose": "Test inventory documentation consolidation behavior.",
        "imports": [
            "from inventory_docs import consolidate_inventory_docs",
        ],
        "provided_interfaces": [],
        "consumed_interfaces": [
            {
                "name": "consolidate_inventory_docs",
                "provider": "inventory documentation fetch source",
                "contract": (
                    "Accepts input_csv and output_csv. Reads item names and "
                    "documentation URLs, fetches each page through a mockable "
                    "fetch boundary, extracts title or first h1, writes output "
                    "CSV, and returns summary counts."
                ),
            }
        ],
        "required_behavior": [
            "Create tests using mocked HTTP behavior, not live network calls.",
            "Test title and first h1 extraction.",
            "Test fetch failure records an error row without aborting.",
            "Test consolidated CSV keeps original inventory columns plus fetched metadata.",
        ],
        "preferred_name": "test_inventory_docs.py",
    },
}

PACKAGE_GATE_IDS = {
    "gate_01_single_file": "gate_01_easy_single_file",
    "gate_02_source_and_tests": "gate_02_low_medium_cli_tests",
    "gate_03_package_layout": "gate_03_medium_package",
    "gate_04_cli_project_layout": "gate_04_medium_hard_cli_project",
    "gate_05_data_tool_layout": "gate_05_hard_multi_source_data_tool",
}

SYNTHESIZER_VALIDATION_FAILURE_CASES: dict[str, dict[str, Any]] = {
    "generated_test_assertion_mismatch": {
        "question": (
            "Create a small Python package that checks local document links. "
            "It should expose reusable checking logic, a CLI, and focused "
            "tests for generated anchors, duplicate anchors, and broken "
            "relative links."
        ),
        "pm_decision": {
            "status": "need_programmers",
            "feature_goal": "Create local document link checking artifacts.",
            "artifact_items": [],
            "selected_artifacts": [],
            "external_evidence_requests": [],
            "limitations": [],
        },
        "generated_artifacts": [
            {
                "artifact_id": "document_checker_source",
                "file_label": "document checker source",
                "file_kind": "source",
                "content_format": "python",
                "path": "src/document_checker.py",
                "content": (
                    "Implements check_document_links(root_dir). Missing "
                    "anchor reports use message text without a leading #."
                ),
                "purpose": (
                    "Scan local documents, collect anchors, report duplicate "
                    "anchors, and report broken relative links."
                ),
            },
            {
                "artifact_id": "document_checker_tests",
                "file_label": "document checker tests",
                "file_kind": "test",
                "content_format": "python",
                "path": "tests/test_document_checker.py",
                "content": (
                    "Tests expect a missing-anchor error message to include "
                    "the literal '#missing-anchor' substring."
                ),
                "purpose": (
                    "Test anchor generation, duplicate anchor detection, and "
                    "broken relative link detection."
                ),
            },
        ],
        "patch_artifacts": [
            {
                "artifact_id": "document_checker_compiled",
                "base": "new_file",
                "diff_text": (
                    "diff --git a/src/document_checker.py "
                    "b/src/document_checker.py\n"
                ),
                "files": [
                    "src/document_checker.py",
                    "tests/test_document_checker.py",
                ],
                "summary": "Generated source and test artifacts.",
            }
        ],
        "validation": {
            "status": "failed",
            "parsed": True,
            "sandbox_applied": False,
            "errors": [
                (
                    "Generated Python tests fail in isolated validation: "
                    "F.. [100%] test_broken_relative_links: "
                    "AssertionError: False is not true: Should detect link "
                    "to non-existent anchor in existing file."
                )
            ],
            "warnings": [],
            "files": [
                "src/document_checker.py",
                "tests/test_document_checker.py",
            ],
        },
        "limitations": [
            (
                "Generated Python tests fail in isolated validation: "
                "F.. [100%] test_broken_relative_links: AssertionError: "
                "False is not true: Should detect link to non-existent "
                "anchor in existing file."
            )
        ],
    },
}


def _artifact_package(case_id: str) -> dict[str, Any]:
    if case_id == "gate_01_single_file":
        contract_ids = ["gate_01_source_log_counter"]
        feature_goal = "Create one Python file for log severity counting."
    elif case_id == "gate_02_source_and_tests":
        contract_ids = [
            "gate_02_source_purchase_summary",
            "gate_02_tests_purchase_summary",
        ]
        feature_goal = "Create purchase summary source and tests."
    elif case_id == "gate_03_package_layout":
        contract_ids = ["gate_03_source_markdown_links"]
        feature_goal = "Create local Markdown link checker package content."
    elif case_id == "gate_04_cli_project_layout":
        contract_ids = [
            "gate_04_source_task_notes",
            "gate_04_tests_task_notes",
            "gate_04_config_task_notes",
        ]
        feature_goal = "Create task-note summary CLI project content."
    elif case_id == "gate_05_data_tool_layout":
        contract_ids = [
            "gate_05_source_inventory_fetch",
            "gate_05_docs_inventory_fetch",
        ]
        feature_goal = "Create inventory documentation consolidation tool."
    else:
        raise AssertionError(f"Unknown package fixture {case_id!r}.")

    artifact_items = [
        deepcopy(PROGRAMMER_CONTRACTS[contract_id])
        for contract_id in contract_ids
    ]
    artifacts = [
        _artifact_from_contract(contract)
        for contract in artifact_items
    ]
    reserved_paths = [
        {
            "artifact_id": artifact["artifact_id"],
            "file_label": artifact["file_label"],
            "path": artifact["path"],
            "file_kind": artifact["file_kind"],
            "content_format": artifact["content_format"],
            "purpose": artifact["purpose"],
        }
        for artifact in artifacts
    ]
    package = {
        "feature_goal": feature_goal,
        "artifact_items": artifact_items,
        "artifacts": artifacts,
        "reserved_paths": reserved_paths,
    }
    return package


def _artifact_from_contract(contract: dict[str, Any]) -> dict[str, Any]:
    path = _path_for_contract(contract)
    content = _placeholder_content(contract)
    artifact = {
        "artifact_id": contract["artifact_id"],
        "file_label": contract["file_label"],
        "file_kind": contract["file_kind"],
        "content_format": contract["content_format"],
        "path": path,
        "content": content,
        "purpose": contract["purpose"],
    }
    return artifact


def _path_for_contract(contract: dict[str, Any]) -> str:
    suffixes = {
        "python": ".py",
        "markdown": ".md",
        "text": ".toml",
        "json": ".json",
        "csv": ".csv",
    }
    directory = {
        "source": "src",
        "test": "tests",
        "docs": "docs",
        "config": "config",
        "data": "data",
    }[contract["file_kind"]]
    suffix = suffixes[contract["content_format"]]
    path = f"{directory}/{contract['artifact_id']}{suffix}"
    return path


def _placeholder_content(contract: dict[str, Any]) -> str:
    content_format = contract["content_format"]
    if content_format == "python":
        content = "def placeholder() -> int:\n    return 1\n"
    elif content_format == "markdown":
        content = "# Usage\n\nDescribe the generated tool.\n"
    elif content_format == "json":
        content = "{}\n"
    elif content_format == "csv":
        content = "name,value\n"
    else:
        content = 'markers = ["TODO", "DONE", "BLOCKED"]\n'
    return content
