"""Interface contracts for the durable coding-run API."""

from __future__ import annotations


def test_coding_run_api_is_exported() -> None:
    """The durable run entrypoints are available at the package boundary."""

    from kazusa_ai_chatbot.coding_agent import continue_coding_run
    from kazusa_ai_chatbot.coding_agent import get_coding_run
    from kazusa_ai_chatbot.coding_agent import start_coding_run
    from kazusa_ai_chatbot.coding_agent.coding_run import (
        continue_coding_run as submodule_continue,
    )
    from kazusa_ai_chatbot.coding_agent.coding_run import (
        get_coding_run as submodule_get,
    )
    from kazusa_ai_chatbot.coding_agent.coding_run import (
        start_coding_run as submodule_start,
    )

    assert start_coding_run is submodule_start
    assert continue_coding_run is submodule_continue
    assert get_coding_run is submodule_get


def test_coding_run_models_are_public() -> None:
    """The run supervisor publishes stable request and response contracts."""

    from kazusa_ai_chatbot.coding_agent.coding_run.models import (
        CodingRunAttempt,
        CodingRunBlocker,
        CodingRunContinueRequest,
        CodingRunEvent,
        CodingRunGetRequest,
        CodingRunLedger,
        CodingRunResponse,
        CodingRunStartRequest,
    )

    assert CodingRunAttempt
    assert CodingRunBlocker
    assert CodingRunContinueRequest
    assert CodingRunEvent
    assert CodingRunGetRequest
    assert CodingRunLedger
    assert CodingRunResponse
    assert CodingRunStartRequest
