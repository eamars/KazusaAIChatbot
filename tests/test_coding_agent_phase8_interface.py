"""Interface contracts for the coding-agent verify-and-repair API."""

from __future__ import annotations


def test_verify_repair_api_is_exported() -> None:
    """The direct verifier is available from the package boundary."""

    from kazusa_ai_chatbot.coding_agent import verify_and_repair_code_change
    from kazusa_ai_chatbot.coding_agent.code_verifying import (
        verify_and_repair_code_change as submodule_entrypoint,
    )

    assert verify_and_repair_code_change is submodule_entrypoint


def test_verify_repair_models_are_public() -> None:
    """The verifier publishes request and response data contracts."""

    from kazusa_ai_chatbot.coding_agent.code_verifying.models import (
        CodingVerifyRepairRequest,
        CodingVerifyRepairResponse,
        ExecutionRepairFeedback,
        VerifyRepairAttempt,
    )

    assert CodingVerifyRepairRequest
    assert CodingVerifyRepairResponse
    assert ExecutionRepairFeedback
    assert VerifyRepairAttempt
