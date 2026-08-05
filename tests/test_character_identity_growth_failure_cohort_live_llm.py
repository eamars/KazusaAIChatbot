"""One-case-at-a-time replay gate for the Asuna identity-growth cohort."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth import llm
from kazusa_ai_chatbot.character_identity_growth.projection import (
    build_identity_review_input,
)
from kazusa_ai_chatbot.llm_interface import LLInterface


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ROOT = Path(__file__).resolve().parents[1]
_DIAGNOSTIC_DIRECTORY = _ROOT / "test_artifacts" / "diagnostics"
_REPLAY_MANIFEST_PATH = (
    _DIAGNOSTIC_DIRECTORY / "asuna_identity_growth_replay_v1.json"
)
_V2_RESULT_PATH = (
    _DIAGNOSTIC_DIRECTORY / "asuna_identity_growth_replay_v2_result.json"
)
_EXPECTED_REPLAY_CASE_COUNT = 185
_EXPECTED_HISTORICAL_FAILURE_COUNT = 42
_EXPECTED_END_TO_END_VALID_COUNT = 176
_EXPECTED_HISTORICAL_VALID_COUNT = 40


if not _REPLAY_MANIFEST_PATH.exists():
    pytest.skip(
        'identity-growth replay manifest is unavailable; run the recovery '
        'script with its source exports before enabling this live cohort',
        allow_module_level=True,
    )


def _load_replay_cases() -> list[dict[str, object]]:
    """Load the frozen case manifest without contacting a live service."""

    manifest = json.loads(_REPLAY_MANIFEST_PATH.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 185:
        raise AssertionError("identity replay manifest must contain 185 cases")
    return [case for case in cases if isinstance(case, dict)]


_REPLAY_CASES = _load_replay_cases()
_REPLAY_CASE_IDS = [str(case["case_id"]) for case in _REPLAY_CASES]


class _CaptureInvoker:
    """Invoke the configured consolidation model and retain raw evidence."""

    def __init__(self) -> None:
        self._inner = LLInterface()
        self.calls: list[dict[str, object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: Any,
    ) -> object:
        """Run one live call and retain its prompt and normalized response."""

        response = await self._inner.ainvoke(messages, config=config)
        self.calls.append({
            "stage_name": config.stage_name,
            "route_name": config.route_name,
            "model": config.model,
            "messages": [
                {
                    "role": getattr(message, "type", ""),
                    "content": str(getattr(message, "content", "")),
                }
                for message in messages
            ],
            "raw_output": str(response.content),
            "usage": dict(response.usage),
        })
        return response


def _case_by_id(case_id: str) -> dict[str, object]:
    """Return one frozen replay case by its stable local test id."""

    for case in _REPLAY_CASES:
        if case["case_id"] == case_id:
            return case
    raise AssertionError(f"unknown identity replay case: {case_id}")


def _load_result_document() -> dict[str, object]:
    """Load the structured V2 result document or create its base shape."""

    if _V2_RESULT_PATH.exists():
        document = json.loads(_V2_RESULT_PATH.read_text(encoding="utf-8"))
    else:
        document = {
            "schema_version": "asuna_identity_growth_replay_v2_result.v1",
            "manifest_path": str(
                _REPLAY_MANIFEST_PATH.relative_to(_ROOT)
            ),
            "results": [],
        }
    if not isinstance(document, dict):
        raise AssertionError("identity replay result document must be an object")
    return document


def _cohort_counts(results: list[dict[str, object]]) -> dict[str, int]:
    """Calculate the reliability denominator and semantic disposition counts."""

    proposal_valid_rows = [
        row
        for row in results
        if isinstance(row.get("proposal"), dict)
        and isinstance(row["proposal"].get("attempt_count"), int)
        and row["proposal"]["attempt_count"] <= llm.IDENTITY_STAGE_ATTEMPT_LIMIT
    ]
    review_valid_rows = [
        row
        for row in proposal_valid_rows
        if isinstance(row.get("review"), dict)
        and isinstance(row["review"].get("attempt_count"), int)
        and row["review"]["attempt_count"] <= llm.IDENTITY_STAGE_ATTEMPT_LIMIT
    ]
    valid_rows = [
        row
        for row in review_valid_rows
        if row.get("status") == "valid_semantic_disposition"
    ]
    historical_rows = [
        row for row in results if row.get("historical_failure") is True
    ]
    historical_valid_rows = [
        row for row in valid_rows if row.get("historical_failure") is True
    ]
    return {
        "manifest_case_count": _EXPECTED_REPLAY_CASE_COUNT,
        "result_case_count": len(results),
        "prompt_safe_replay_input_count": sum(
            isinstance(case.get("replay_input"), dict)
            for case in _REPLAY_CASES
        ),
        "metadata_only_case_count": sum(
            not isinstance(case.get("replay_input"), dict)
            for case in _REPLAY_CASES
        ),
        "historical_failure_denominator": len(historical_rows),
        "historical_failure_expected": _EXPECTED_HISTORICAL_FAILURE_COUNT,
        "proposal_valid_within_attempt_cap": len(proposal_valid_rows),
        "review_valid_after_proposal": len(review_valid_rows),
        "end_to_end_valid": len(valid_rows),
        "historical_failure_valid": len(historical_valid_rows),
        "terminal_contract_or_provider_failures": sum(
            row.get("status") != "valid_semantic_disposition"
            for row in results
        ),
    }


def _assert_complete_cohort_thresholds(
    document: dict[str, object],
) -> None:
    """Require complete prompt-safe replay coverage and the release gates."""

    results = document.get("results")
    if not isinstance(results, list) or not all(
        isinstance(row, dict) for row in results
    ):
        raise AssertionError("identity replay results must be object rows")
    typed_results = [row for row in results if isinstance(row, dict)]
    result_ids = [str(row.get("case_id")) for row in typed_results]
    if len(result_ids) != len(set(result_ids)):
        raise AssertionError("identity replay results contain duplicate case ids")
    if set(result_ids) != set(_REPLAY_CASE_IDS):
        raise AssertionError(
            "identity replay results do not cover the frozen manifest"
        )
    manifest_by_id = {
        str(case["case_id"]): case for case in _REPLAY_CASES
    }
    for row in typed_results:
        case = manifest_by_id[str(row["case_id"])]
        if row.get("historical_failure") is not case.get(
            "historical_failure"
        ):
            raise AssertionError(
                "result historical-failure membership differs from the manifest"
            )
        if row.get("source_fidelity") != case.get("source_fidelity"):
            raise AssertionError(
                "result source fidelity differs from the manifest"
            )
        calls = row.get("calls")
        if not isinstance(calls, list) or len(calls) < 2:
            raise AssertionError(
                "each completed replay case must retain proposal and review calls"
            )
        for call in calls:
            if not isinstance(call, dict):
                raise AssertionError("replay call evidence must be an object")
            if (
                call.get("route_name") != llm._identity_llm_config.route_name
                or call.get("model") != llm._identity_llm_config.model
            ):
                raise AssertionError(
                    "replay call configuration differs from the identity route"
                )
    manifest_failure_count = sum(
        case.get("historical_failure") is True for case in _REPLAY_CASES
    )
    if manifest_failure_count != _EXPECTED_HISTORICAL_FAILURE_COUNT:
        raise AssertionError(
            "historical failure denominator changed without plan approval"
        )
    counts = _cohort_counts(typed_results)
    document["cohort_counts"] = counts
    if counts["prompt_safe_replay_input_count"] != _EXPECTED_REPLAY_CASE_COUNT:
        raise AssertionError(
            "the sign-off cohort requires prompt-safe replay input for every case"
        )
    if counts["historical_failure_denominator"] != (
        _EXPECTED_HISTORICAL_FAILURE_COUNT
    ):
        raise AssertionError(
            "result historical-failure denominator does not match the manifest"
        )
    if counts["review_valid_after_proposal"] * 100 < (
        counts["proposal_valid_within_attempt_cap"] * 95
    ):
        raise AssertionError(
            "identity growth replay is below the 95% review-after-proposal gate"
        )
    if counts["end_to_end_valid"] < _EXPECTED_END_TO_END_VALID_COUNT:
        raise AssertionError(
            "identity growth replay is below the 176/185 end-to-end gate"
        )
    if counts["historical_failure_valid"] < _EXPECTED_HISTORICAL_VALID_COUNT:
        raise AssertionError(
            "identity growth replay is below the 40/42 historical-failure gate"
        )


def _append_result(result: dict[str, object]) -> Path:
    """Append one inspected replay result to the structured evidence file."""

    document = _load_result_document()
    results = document.get("results")
    if not isinstance(results, list):
        raise AssertionError("identity replay result file has invalid results")
    results[:] = [
        row for row in results
        if isinstance(row, dict) and row.get("case_id") != result["case_id"]
    ]
    results.append(result)
    cohort_error: AssertionError | None = None
    if len(results) == _EXPECTED_REPLAY_CASE_COUNT:
        try:
            _assert_complete_cohort_thresholds(document)
        except AssertionError as exc:
            cohort_error = exc
    _V2_RESULT_PATH.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if cohort_error is not None:
        raise cohort_error
    return _V2_RESULT_PATH


@pytest.mark.parametrize("case_id", _REPLAY_CASE_IDS)
async def test_live_replay_case(case_id: str) -> None:
    """Replay one frozen case and require valid proposal/review dispositions."""

    case = _case_by_id(case_id)
    replay_input = case.get("replay_input")
    if not isinstance(replay_input, dict):
        blocked_result = {
            "case_id": case_id,
            "status": "blocked_missing_replay_input",
            "historical_failure": case["historical_failure"],
            "source_fidelity": case["source_fidelity"],
        }
        _append_result(blocked_result)
        pytest.fail(
            f"{case_id} has no prompt-safe replay input; metadata-only "
            "cases remain outside the reliability numerator"
        )

    replay_input.setdefault(
        "allowed_paths",
        sorted(models.ALLOWED_IDENTITY_PATHS),
    )
    capture = _CaptureInvoker()
    result: dict[str, object] = {
        "case_id": case_id,
        "historical_failure": case["historical_failure"],
        "historical_failure_stage": case["historical_failure_stage"],
        "source_fidelity": case["source_fidelity"],
        "calls": capture.calls,
    }
    try:
        proposal_result = await llm.propose_identity_growth(
            replay_input,
            invoker=capture,
        )
        result["proposal"] = {
            "decision": proposal_result.decision,
            "attempt_count": proposal_result.attempt_count,
            "prompt_chars": proposal_result.prompt_chars,
            "output_chars": proposal_result.output_chars,
            "validation_error_codes": list(
                proposal_result.validation_error_codes
            ),
        }
        review_input = build_identity_review_input(
            proposal_input=replay_input,
            proposal=proposal_result.decision,
        )
        review_result = await llm.review_identity_growth(
            review_input,
            invoker=capture,
        )
        result["review"] = {
            "decision": review_result.decision,
            "attempt_count": review_result.attempt_count,
            "prompt_chars": review_result.prompt_chars,
            "output_chars": review_result.output_chars,
            "validation_error_codes": list(
                review_result.validation_error_codes
            ),
        }
        result["status"] = "valid_semantic_disposition"
    except Exception as exc:
        result["status"] = "contract_or_provider_failure"
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _append_result(result)
        raise

    artifact_path = _append_result(result)
    print(f"IDENTITY_GROWTH_REPLAY_ARTIFACT={artifact_path}")
    assert result["status"] == "valid_semantic_disposition"
