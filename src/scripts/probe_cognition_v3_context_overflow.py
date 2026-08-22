"""Dry-run and explicitly opted-in live V3 serving-overflow probe."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import string
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Literal

import httpx
from langchain_core.messages import HumanMessage
from openai import BadRequestError, OpenAIError

from kazusa_ai_chatbot.cognition_core_v3.budget import (
    MINIMUM_SERVING_WINDOW_TOKENS,
    estimate_message_tokens,
)
from kazusa_ai_chatbot.config import (
    COGNITION_CORE_ENGINE,
    COGNITION_STAGE_TIMEOUT_SECONDS,
    CognitionRouteSettingV1,
    CognitionV3RouteSettingsV1,
    get_selected_cognition_route_settings,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMInvoker,
    LLMThinkingConfig,
)

OVERFLOW_PROBE_SCHEMA_VERSION = "cognition_v3_context_overflow_probe.v1"
V3_CHAIN_ROUTE_NAME = "COGNITION_V3_CHAIN_LLM"
ERROR_MESSAGE_MAX_CHARS = 2_000
_OVERFLOW_PAYLOAD_DOMAIN = b"kazusa:cognition-v3:context-overflow:v1:"
_OVERFLOW_PAYLOAD_CHUNK_SIZE = 4_096
_OVERFLOW_PAYLOAD_COUNTER_BYTES = 8
_OVERFLOW_PAYLOAD_ALPHABET = (
    string.ascii_letters + string.digits + string.punctuation
)
CONTEXT_OVERFLOW_REJECTION_MARKERS = (
    "context length",
    "context_length",
    "context window",
    "maximum context",
    "prompt is too long",
    "too many tokens",
    "exceeds the available context size",
    "exceed_context_size_error",
)

OverflowProbeDisposition = Literal[
    "expected_rejection",
    "success",
    "transport_failure",
]


def build_overflow_probe_payload(char_count: int) -> str:
    """Build exact-length deterministic high-entropy printable ASCII input."""

    if char_count < 0:
        raise ValueError("overflow probe payload length cannot be negative")
    if char_count == 0:
        return ""

    chunks: list[str] = []
    remaining = char_count
    chunk_index = 0
    alphabet_length = len(_OVERFLOW_PAYLOAD_ALPHABET)
    while remaining > 0:
        chunk_length = min(remaining, _OVERFLOW_PAYLOAD_CHUNK_SIZE)
        counter_bytes = chunk_index.to_bytes(
            _OVERFLOW_PAYLOAD_COUNTER_BYTES,
            "big",
        )
        digest = hashlib.shake_256(
            _OVERFLOW_PAYLOAD_DOMAIN + counter_bytes,
        ).digest(chunk_length)
        chunk = "".join(
            _OVERFLOW_PAYLOAD_ALPHABET[value % alphabet_length]
            for value in digest
        )
        chunks.append(chunk)
        remaining -= chunk_length
        chunk_index += 1

    payload = "".join(chunks)
    return payload


@dataclass(frozen=True)
class OverflowProbeDryRun:
    """Dry-run validation facts without any live provider call."""

    route_name: str
    declared_context_window_tokens: int
    payload_estimate_tokens: int
    payload_exceeds_declared_window: bool
    dry_run: bool = True


@dataclass(frozen=True)
class OverflowProbeLiveEvidence:
    """Sanitized raw evidence from one live oversized provider request."""

    schema_version: str
    recorded_at_utc: str
    route_name: str
    endpoint_sha256: str
    model_name: str
    declared_context_window_tokens: int
    configured_max_completion_tokens: int
    payload_message_count: int
    payload_character_count: int
    payload_sha256: str
    payload_estimate_tokens: int
    payload_exceeds_declared_window: bool
    disposition: OverflowProbeDisposition
    wall_time_ms: int
    provider_status_code: int | None
    usage_reported: bool
    usage: dict[str, object]
    response_content_characters: int
    response_content_sha256: str
    error_type: str
    error_message: str
    dry_run: bool = False


def run_overflow_probe_dry_run(
    *,
    route_name: str,
    declared_context_window_tokens: int,
    payload_messages: tuple[str, ...],
) -> OverflowProbeDryRun:
    """Validate a synthetic overflow payload without issuing a request."""

    if not route_name.strip():
        raise ValueError("overflow probe route_name must be non-empty")
    if declared_context_window_tokens < MINIMUM_SERVING_WINDOW_TOKENS:
        raise ValueError(
            "overflow probe requires a declared window of at least "
            f"{MINIMUM_SERVING_WINDOW_TOKENS} tokens"
        )
    if not payload_messages:
        raise ValueError("overflow probe payload cannot be empty")

    payload_estimate = estimate_message_tokens(list(payload_messages))
    report = OverflowProbeDryRun(
        route_name=route_name,
        declared_context_window_tokens=declared_context_window_tokens,
        payload_estimate_tokens=payload_estimate,
        payload_exceeds_declared_window=(
            payload_estimate > declared_context_window_tokens
        ),
    )
    return report


def _sanitize_external_error(
    error: BaseException,
    *,
    endpoint: str,
    credential: str,
) -> str:
    """Bound an external failure message after removing route secrets."""

    sanitized = str(error)
    for secret_value in (endpoint, credential):
        if secret_value:
            sanitized = sanitized.replace(secret_value, "[redacted]")
    bounded_message = sanitized[:ERROR_MESSAGE_MAX_CHARS]
    return bounded_message


def _is_context_overflow_rejection(error: BadRequestError) -> bool:
    """Return whether a provider bad request names context overflow."""

    rejection_text = f"{error} {error.body}".lower()
    matches_overflow = any(
        marker in rejection_text
        for marker in CONTEXT_OVERFLOW_REJECTION_MARKERS
    )
    return matches_overflow


async def run_overflow_probe_live(
    *,
    llm: LLMInvoker,
    route_setting: CognitionRouteSettingV1,
    route_name: str,
    declared_context_window_tokens: int,
    payload_messages: tuple[str, ...],
    timeout_seconds: float,
) -> OverflowProbeLiveEvidence:
    """Issue one non-streaming oversized call to the exact V3 chain route."""

    if route_name != V3_CHAIN_ROUTE_NAME:
        raise ValueError(
            f"live overflow probe route must be {V3_CHAIN_ROUTE_NAME}",
        )
    context_window_tokens = route_setting.context_window_tokens
    if context_window_tokens != declared_context_window_tokens:
        raise ValueError(
            "live overflow probe declared window must match loaded route config",
        )
    if route_setting.thinking_enabled:
        raise ValueError("live overflow probe requires thinking to be disabled")
    dry_run = run_overflow_probe_dry_run(
        route_name=route_name,
        declared_context_window_tokens=declared_context_window_tokens,
        payload_messages=payload_messages,
    )
    if not dry_run.payload_exceeds_declared_window:
        raise ValueError(
            "live overflow probe payload must exceed the declared window",
        )

    call_config = LLMCallConfig(
        stage_name="scripts.probe_cognition_v3_context_overflow",
        route_name=V3_CHAIN_ROUTE_NAME,
        base_url=route_setting.base_url,
        api_key=route_setting.api_key,
        model=route_setting.model,
        temperature=0.1,
        top_p=0.7,
        top_k=None,
        max_completion_tokens=route_setting.max_completion_tokens,
        presence_penalty=None,
        timeout_seconds=timeout_seconds,
        thinking=LLMThinkingConfig(enabled=route_setting.thinking_enabled),
        context_window_tokens=context_window_tokens,
    )
    provider_messages = [
        HumanMessage(content=message)
        for message in payload_messages
    ]
    started_at = perf_counter()
    provider_status_code: int | None = None
    usage: dict[str, object] = {}
    response_content = ""
    error_type = ""
    error_message = ""
    try:
        response = await llm.ainvoke(
            provider_messages,
            config=call_config,
        )
    except BadRequestError as exc:
        if _is_context_overflow_rejection(exc):
            disposition: OverflowProbeDisposition = "expected_rejection"
        else:
            disposition = "transport_failure"
        provider_status_code = exc.status_code
        error_type = type(exc).__name__
        error_message = _sanitize_external_error(
            exc,
            endpoint=route_setting.base_url,
            credential=route_setting.api_key,
        )
    except (httpx.TransportError, OpenAIError, OSError) as exc:
        disposition = "transport_failure"
        provider_status_code = getattr(exc, "status_code", None)
        error_type = type(exc).__name__
        error_message = _sanitize_external_error(
            exc,
            endpoint=route_setting.base_url,
            credential=route_setting.api_key,
        )
    else:
        disposition = "success"
        usage = dict(response.usage)
        response_content = response.content

    wall_time_ms = max(0, round((perf_counter() - started_at) * 1_000))
    payload_json = json.dumps(
        payload_messages,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload_digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    endpoint_digest = hashlib.sha256(
        route_setting.base_url.encode("utf-8"),
    ).hexdigest()
    response_digest = ""
    if response_content:
        response_digest = hashlib.sha256(
            response_content.encode("utf-8"),
        ).hexdigest()
    recorded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    evidence = OverflowProbeLiveEvidence(
        schema_version=OVERFLOW_PROBE_SCHEMA_VERSION,
        recorded_at_utc=recorded_at,
        route_name=V3_CHAIN_ROUTE_NAME,
        endpoint_sha256=endpoint_digest,
        model_name=route_setting.model,
        declared_context_window_tokens=context_window_tokens,
        configured_max_completion_tokens=(
            route_setting.max_completion_tokens
        ),
        payload_message_count=len(payload_messages),
        payload_character_count=sum(len(message) for message in payload_messages),
        payload_sha256=payload_digest,
        payload_estimate_tokens=dry_run.payload_estimate_tokens,
        payload_exceeds_declared_window=True,
        disposition=disposition,
        wall_time_ms=wall_time_ms,
        provider_status_code=provider_status_code,
        usage_reported=bool(usage),
        usage=usage,
        response_content_characters=len(response_content),
        response_content_sha256=response_digest,
        error_type=error_type,
        error_message=error_message,
    )
    return evidence


async def _run_loaded_live_probe(
    *,
    route_setting: CognitionRouteSettingV1,
    route_name: str,
    declared_context_window_tokens: int,
    payload_messages: tuple[str, ...],
) -> OverflowProbeLiveEvidence:
    """Own one live interface session and close it after the probe call."""

    llm = LLInterface()
    try:
        evidence = await run_overflow_probe_live(
            llm=llm,
            route_setting=route_setting,
            route_name=route_name,
            declared_context_window_tokens=declared_context_window_tokens,
            payload_messages=payload_messages,
            timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
        )
    finally:
        await llm.aclose()
    return evidence


def _build_parser() -> argparse.ArgumentParser:
    """Build the overflow-probe CLI parser."""

    parser = argparse.ArgumentParser(
        description="Dry-run or explicitly invoke the V3 serving-overflow probe.",
    )
    parser.add_argument("--route-name", required=True)
    parser.add_argument("--context-window-tokens", type=int, required=True)
    parser.add_argument("--payload-char-count", type=int, required=True)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Issue one oversized probe invocation after validating V3 config.",
    )
    return parser


def main() -> int:
    """Emit raw JSON evidence for a dry run or explicitly opted-in live call."""

    parser = _build_parser()
    args = parser.parse_args()
    if args.payload_char_count <= 0:
        parser.error("--payload-char-count must be positive")
    payload = (build_overflow_probe_payload(args.payload_char_count),)
    dry_run = run_overflow_probe_dry_run(
        route_name=args.route_name,
        declared_context_window_tokens=args.context_window_tokens,
        payload_messages=payload,
    )
    if not args.live:
        print(json.dumps(asdict(dry_run), sort_keys=True))
        return 0

    if COGNITION_CORE_ENGINE != "v3":
        raise ValueError(
            "live overflow probe requires COGNITION_CORE_ENGINE=v3",
        )
    selected_settings = get_selected_cognition_route_settings()
    if not isinstance(selected_settings, CognitionV3RouteSettingsV1):
        raise TypeError("selected cognition route settings are not V3")
    evidence = asyncio.run(
        _run_loaded_live_probe(
            route_setting=selected_settings.chain,
            route_name=args.route_name,
            declared_context_window_tokens=args.context_window_tokens,
            payload_messages=payload,
        )
    )
    print(json.dumps(asdict(evidence), ensure_ascii=False, sort_keys=True))
    exit_code = 0 if evidence.disposition == "expected_rejection" else 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
