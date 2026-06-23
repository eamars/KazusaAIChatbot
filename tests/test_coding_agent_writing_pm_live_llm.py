"""Real-LLM role test for the top-level writing PM ideal input."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
    WRITING_PM_PROMPT,
    _pm_payload,
    decide_writing_work,
)
from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


_TEST_NAME = "coding_agent_writing_pm_ideal_input_live_llm"
_CASES_PATH = (
    Path("test_artifacts")
    / "live_gate"
    / "coding_agent_pm_ideal_inputs.json"
)
_ALLOWED_PM_STATUSES = {
    "need_reading",
    "need_module_pms",
    "ready_to_write",
    "needs_user_input",
    "overloaded",
    "rejected",
}
_FORBIDDEN_TOP_PM_KEYS = {
    "base_revision",
    "code_artifact",
    "diff",
    "file_content",
    "mutex_id",
    "owned_path",
    "owned_paths",
    "patch",
    "patch_hunks",
    "repo_relative_path",
    "symbols_to_define",
    "unified_diff",
}
_FORBIDDEN_TOP_PM_INPUT_KEYS = {
    "current_file_context",
    "evidence",
    "evidence_refs",
    "excerpt",
    "imports",
    "line_end",
    "line_start",
    "owner_candidates",
    "path",
    "source_file_hints",
}
_FORBIDDEN_TOP_PM_OUTPUT_KEYS = {
    *_FORBIDDEN_TOP_PM_KEYS,
    "current_file_context",
    "imports",
    "placement_hint",
    "preferred_name",
    "preferred_path",
    "questions",
    "read_only_paths",
    "related_paths",
}
_SCOPE_STRING_FIELDS = (
    "preferred_path",
    "preferred_name",
    "placement_hint",
)
_SCOPE_LIST_FIELDS = (
    "forbidden_paths",
    "read_only_paths",
    "related_paths",
)


def _load_cases() -> dict[str, Any]:
    """Load the PM ideal-input fixture."""

    text = _CASES_PATH.read_text(encoding="utf-8")
    loaded = json.loads(text)
    return loaded


async def test_phase2_role_03() -> None:
    """Run the top-level writing PM on one ideal role input."""

    cases = _load_cases()
    case = cases["top_level"]["case_01"]
    pm_input = case["pm_input"]
    model_payload = _pm_payload(pm_input)
    trace: dict[str, object] = {}

    decision = await decide_writing_work(pm_input, trace=trace)
    evaluation = _evaluate_top_level_pm(
        model_payload=model_payload,
        decision=decision,
        trace=trace,
    )
    trace_path = write_llm_trace(
        _TEST_NAME,
        case["case_id"],
        {
            "category": case["category"],
            "system_prompt": WRITING_PM_PROMPT,
            "pm_input": pm_input,
            "model_payload": model_payload,
            "pm_trace": trace,
            "normalized_decision": decision,
            "evaluation": evaluation,
        },
    )

    print(f"TRACE_PATH={trace_path.as_posix()}")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=str))

    assert evaluation["status"] == "passed", "\n".join(evaluation["errors"])


def _evaluate_top_level_pm(
    *,
    model_payload: dict[str, Any],
    decision: dict[str, Any],
    trace: dict[str, object],
) -> dict[str, object]:
    """Evaluate top-level PM output against role-boundary hard gates."""

    errors: list[str] = []
    warnings: list[str] = []
    raw_output = trace.get("raw_output")
    parsed_output = trace.get("parsed_output")

    if not isinstance(raw_output, str) or not raw_output.strip():
        errors.append("Top-level PM returned empty raw output.")
    if not isinstance(parsed_output, dict):
        errors.append("Top-level PM raw output did not parse as a JSON object.")

    leaked_input_keys = sorted(
        _forbidden_keys(model_payload, _FORBIDDEN_TOP_PM_INPUT_KEYS)
    )
    if leaked_input_keys:
        errors.append(
            "Top-level PM model payload included source-scope fields: "
            + ", ".join(leaked_input_keys)
        )

    status = decision.get("status")
    if status not in _ALLOWED_PM_STATUSES:
        errors.append(f"Top-level PM returned unsupported status {status!r}.")
    if status != "need_module_pms":
        errors.append(
            "Top-level PM did not return module/file work for the ideal input; "
            f"status was {status!r}."
        )

    file_demands = decision.get("file_demands")
    if not isinstance(file_demands, list) or not file_demands:
        errors.append("Top-level PM returned no semantic file demands.")
    else:
        for index, demand in enumerate(file_demands, start=1):
            errors.extend(_file_demand_errors(demand, index=index))

    cross_module_imports = decision.get("cross_module_imports")
    if not isinstance(cross_module_imports, dict):
        errors.append("Top-level PM did not expose cross_module_imports.")
    elif cross_module_imports:
        errors.append("Top-level PM emitted import lines owned by Module PM.")

    leaked_keys = sorted(
        _forbidden_keys(parsed_output, _FORBIDDEN_TOP_PM_OUTPUT_KEYS)
    )
    if leaked_keys:
        errors.append(
            "Top-level PM emitted fields owned by lower roles: "
            + ", ".join(leaked_keys)
        )

    status_text = "failed" if errors else "passed"
    evaluation = {
        "status": status_text,
        "errors": errors,
        "warnings": warnings,
        "file_demand_count": len(file_demands) if isinstance(file_demands, list) else 0,
    }
    return evaluation


def _file_demand_errors(value: object, *, index: int) -> list[str]:
    """Return structural errors for one PM-owned file demand."""

    if not isinstance(value, dict):
        return [f"File demand {index} is not a JSON object."]

    errors: list[str] = []
    if not _clean_string(value.get("purpose")):
        errors.append(f"File demand {index} has no purpose.")
    if not isinstance(value.get("interface_contract"), dict):
        errors.append(f"File demand {index} has no interface_contract object.")
    if not isinstance(value.get("integration_contract"), dict):
        errors.append(f"File demand {index} has no integration_contract object.")
    if not isinstance(value.get("validation_expectations"), list):
        errors.append(f"File demand {index} has no validation_expectations list.")
    if not isinstance(value.get("work_instructions"), list):
        errors.append(f"File demand {index} has no work_instructions list.")
    for field in _SCOPE_STRING_FIELDS:
        if _clean_string(value.get(field)):
            errors.append(
                f"File demand {index} set {field}, which belongs to File Agent."
            )
    for field in _SCOPE_LIST_FIELDS:
        field_value = value.get(field)
        if isinstance(field_value, list) and field_value:
            errors.append(
                f"File demand {index} set {field}, which belongs to File Agent."
            )
    return errors


def _forbidden_keys(value: object, forbidden_keys: set[str]) -> set[str]:
    """Return forbidden keys found anywhere inside a role output."""

    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden_keys:
                found.add(key)
            found.update(_forbidden_keys(child, forbidden_keys))
    elif isinstance(value, list):
        for item in value:
            found.update(_forbidden_keys(item, forbidden_keys))
    return found


def _clean_string(value: object) -> str:
    """Return a stripped string or an empty value."""

    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    return cleaned
