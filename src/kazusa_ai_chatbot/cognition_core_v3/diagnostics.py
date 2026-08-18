"""V3 public diagnostics projection and protected chain trace metadata.

The V3 public stage-trace record carries exactly the V2 validation-capture
stage field set so existing observability surfaces keep their contract, plus
two protected chain-scope fields: the registered chain name and the attempt
number within that stage's owner cap. Configuration identity is projected the
same way as in V2: route and generation settings only, never credentials.
Protected failure metadata crosses the boundary as closed typed values alone;
raw candidate text, validator prose, provider exception messages, and provider
metadata stay inside the harness-owned capture and never appear in protected
chain records.
"""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    StageFailure,
    StageResult,
)

# Exact V2 public stage-trace field set preserved by the V3 record projection.
STAGE_TRACE_PUBLIC_FIELDS = (
    "stage_id",
    "branch_id",
    "config",
    "system_prompt",
    "human_payload",
    "raw_output",
    "parsed_output",
    "parse_status",
    "started_at_monotonic",
    "ended_at_monotonic",
    "duration_ms",
    "error",
)

# Protected chain-scope fields added to the V2 public field set.
PROTECTED_CHAIN_FIELDS = (
    "chain_name",
    "attempt_number",
)

CONFIG_IDENTITY_FIELDS = (
    "stage_name",
    "route_name",
    "base_url",
    "model",
    "temperature",
    "top_p",
    "top_k",
    "max_completion_tokens",
    "presence_penalty",
    "timeout_seconds",
    "thinking_enabled",
)

PROTECTED_FAILURE_FIELDS = (
    "chain_name",
    "stage_name",
    "failure_class",
    "error_code",
    "repair_attempted",
)


def project_config_identity(config: object) -> dict[str, object]:
    """Keep route identity and generation settings without exposing API keys.

    Args:
        config: An LLM call configuration owned by the stage boundary.

    Returns:
        The exact V2 config-identity projection; credential attributes are
        never read or named in the result.
    """
    thinking = getattr(config, "thinking", None)
    projected = {
        "stage_name": getattr(config, "stage_name", None),
        "route_name": getattr(config, "route_name", None),
        "base_url": getattr(config, "base_url", None),
        "model": getattr(config, "model", None),
        "temperature": getattr(config, "temperature", None),
        "top_p": getattr(config, "top_p", None),
        "top_k": getattr(config, "top_k", None),
        "max_completion_tokens": getattr(config, "max_completion_tokens", None),
        "presence_penalty": getattr(config, "presence_penalty", None),
        "timeout_seconds": getattr(config, "timeout_seconds", None),
        "thinking_enabled": getattr(thinking, "enabled", None),
    }
    return projected


def build_chain_trace_record(
    *,
    chain_name: str,
    stage_id: str,
    config: object,
    system_prompt: str,
    human_payload: str,
    raw_output: str | None,
    parsed_output: object | None,
    parse_status: str,
    started_at: float,
    ended_at: float,
    branch_id: str | None = None,
    attempt_number: int = 1,
    error: str | None = None,
) -> dict[str, object]:
    """Build one protected chain trace record for a stage attempt.

    The public field set matches the V2 validation-capture stage record
    exactly; ``chain_name`` and ``attempt_number`` are the only additions.
    Configuration is projected through :func:`project_config_identity`, so no
    credential attribute ever enters the record.

    Args:
        chain_name: Registered chain owning this stage attempt.
        stage_id: Stable registered stage identity for the attempt.
        config: Stage-bound LLM configuration, projected without credentials.
        system_prompt: Static prompt supplied to the model.
        human_payload: Current-run dynamic prompt payload.
        raw_output: Normalized raw model output when invocation succeeded.
        parsed_output: Parser result before structural validation, if any.
        parse_status: Stage parse or validation status for evidence review.
        started_at: Monotonic stage start time.
        ended_at: Monotonic stage end time.
        branch_id: Optional activated goal branch identity.
        attempt_number: 1-based attempt position within the owner cap.
        error: Concrete failure text when the stage failed.

    Returns:
        The protected chain trace record with the exact public field set.
    """
    return {
        "chain_name": chain_name,
        "stage_id": stage_id,
        "branch_id": branch_id,
        "config": project_config_identity(config),
        "system_prompt": system_prompt,
        "human_payload": human_payload,
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "parse_status": parse_status,
        "started_at_monotonic": started_at,
        "ended_at_monotonic": ended_at,
        "duration_ms": max(0, int((ended_at - started_at) * 1000)),
        "attempt_number": attempt_number,
        "error": error,
    }


def project_protected_chain_failure(failure: StageFailure) -> dict[str, object]:
    """Project one typed stage failure into protected chain metadata.

    Only closed typed fields cross the protection boundary: chain and stage
    identity, the bounded failure class, the exact error code, and the repair
    disposition. Raw candidate text, validator prose, provider exception
    messages, and provider metadata never appear in this projection; raw
    evidence stays inside the harness-owned capture.

    Args:
        failure: The typed stage failure record from a bounded stage attempt.

    Returns:
        The protected failure metadata with exactly its closed field set.
    """
    return {
        "chain_name": failure.chain_name,
        "stage_name": failure.stage_name,
        "failure_class": failure.failure_class,
        "error_code": failure.error_code,
        "repair_attempted": failure.repair_attempted,
    }


def project_protected_chain_result(result: StageResult) -> dict[str, object]:
    """Project one stage result into protected chain metadata.

    The projection carries the acceptance outcome for every stage and adds the
    typed failure fields only when a bounded attempt exhausted or terminated;
    it never carries local-state payloads, semantic summaries, raw output, or
    provider metadata.

    Args:
        result: The bounded stage execution result from the chain executor.

    Returns:
        The protected result metadata for observability surfaces.
    """
    record: dict[str, object] = {
        "chain_name": result.chain_name,
        "stage_name": result.stage_name,
        "accepted": result.accepted,
    }
    if result.failure is not None:
        record["failure"] = project_protected_chain_failure(result.failure)
    return record
