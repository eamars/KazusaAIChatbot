"""Deterministic algorithmic subagent tests for complex-task resolver."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.complex_task_resolver import (
    COMPLEX_TASK_SUBAGENT_REQUEST_VERSION,
    validate_complex_task_subagent_request,
)
from kazusa_ai_chatbot.complex_task_resolver.algorithmic import AlgorithmicSubagent


@pytest.mark.asyncio
async def test_algorithmic_subagent_evaluates_normalized_expression() -> None:
    """Calculate a caller-prepared expression without semantic conversion."""

    request = validate_complex_task_subagent_request({
        "schema_version": COMPLEX_TASK_SUBAGENT_REQUEST_VERSION,
        "node_id": "math_1",
        "subagent": "algorithmic",
        "action": "evaluate_expression",
        "objective": "Evaluate prepared watt-hour arithmetic.",
        "payload": {
            "expression": "(45 * 6) + 12 + (12 * 6) + 60",
            "label": "total_watt_hours",
        },
        "constraints": {},
    })
    subagent = AlgorithmicSubagent()

    result = await subagent.run(request, {}, max_attempts=1)

    assert result["resolved"] is True
    assert result["status"] == "resolved"
    assert result["result"]["result_str"] == "414"
    assert result["result"]["result_type"] == "int"
    assert result["result"]["display"] == (
        "total_watt_hours: (45 * 6) + 12 + (12 * 6) + 60 = 414"
    )


@pytest.mark.asyncio
async def test_algorithmic_subagent_allows_safe_numeric_helpers() -> None:
    """Support a small safe math surface for prepared expressions."""

    request = validate_complex_task_subagent_request({
        "schema_version": COMPLEX_TASK_SUBAGENT_REQUEST_VERSION,
        "node_id": "math_2",
        "subagent": "algorithmic",
        "action": "evaluate_expression",
        "objective": "Evaluate prepared numeric helper expression.",
        "payload": {
            "expression": "round(sqrt(81) + statistics.mean([3, 6, 9]), 2)",
            "label": "safe_helper_result",
        },
        "constraints": {},
    })
    subagent = AlgorithmicSubagent()

    result = await subagent.run(request, {}, max_attempts=1)

    assert result["resolved"] is True
    assert result["result"]["result_str"] == "15.0"
    assert result["result"]["result_type"] == "float"


@pytest.mark.asyncio
async def test_algorithmic_subagent_rejects_unknown_operations() -> None:
    """Fail closed when a request is outside the expression contract."""

    request = validate_complex_task_subagent_request({
        "schema_version": COMPLEX_TASK_SUBAGENT_REQUEST_VERSION,
        "node_id": "math_3",
        "subagent": "algorithmic",
        "action": "weighted_score",
        "objective": "Use the retired operation contract.",
        "payload": {"expression": "2 + 2"},
        "constraints": {},
    })
    subagent = AlgorithmicSubagent()

    result = await subagent.run(request, {}, max_attempts=1)

    assert result["resolved"] is False
    assert result["status"] == "invalid"
    assert result["unresolved_items"] == [
        "unsupported algorithmic action: weighted_score",
    ]


@pytest.mark.asyncio
async def test_algorithmic_subagent_rejects_unsafe_expression() -> None:
    """Reject imports, unknown names, and private attribute access."""

    request = validate_complex_task_subagent_request({
        "schema_version": COMPLEX_TASK_SUBAGENT_REQUEST_VERSION,
        "node_id": "math_4",
        "subagent": "algorithmic",
        "action": "evaluate_expression",
        "objective": "Reject unsafe syntax.",
        "payload": {
            "expression": "__import__('os').system('echo unsafe')",
            "label": "unsafe",
        },
        "constraints": {},
    })
    subagent = AlgorithmicSubagent()

    result = await subagent.run(request, {}, max_attempts=1)

    assert result["resolved"] is False
    assert result["status"] == "invalid"
    assert result["unresolved_items"] == [
        "Unknown name: __import__",
    ]


@pytest.mark.asyncio
async def test_algorithmic_subagent_rejects_non_numeric_result() -> None:
    """Keep the calculation subagent from returning arbitrary objects."""

    request = validate_complex_task_subagent_request({
        "schema_version": COMPLEX_TASK_SUBAGENT_REQUEST_VERSION,
        "node_id": "math_5",
        "subagent": "algorithmic",
        "action": "evaluate_expression",
        "objective": "Reject non-numeric output.",
        "payload": {
            "expression": "['not', 'a', 'number']",
            "label": "bad_result",
        },
        "constraints": {},
    })
    subagent = AlgorithmicSubagent()

    result = await subagent.run(request, {}, max_attempts=1)

    assert result["resolved"] is False
    assert result["status"] == "invalid"
    assert result["unresolved_items"] == [
        "expression result: expected numeric or boolean value",
    ]
