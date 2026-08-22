"""Deterministic tests for V3 estimator calibration and overflow probing."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from dataclasses import asdict

import httpx
import pytest
from openai import BadRequestError

from kazusa_ai_chatbot.cognition_core_v3.budget import estimate_message_tokens
from kazusa_ai_chatbot.config import CognitionRouteSettingV1
from kazusa_ai_chatbot.llm_interface import (
    BackendDescriptor,
    LLMCallConfig,
    LLMResponse,
)
from scripts import probe_cognition_v3_context_overflow as overflow_probe_module
from scripts.calibrate_cognition_v3_token_estimator import (
    compute_calibration_report,
)
from scripts.probe_cognition_v3_context_overflow import (
    V3_CHAIN_ROUTE_NAME,
    build_overflow_probe_payload,
    run_overflow_probe_dry_run,
    run_overflow_probe_live,
)
from scripts.probe_cognition_v3_context_overflow import (
    main as overflow_probe_main,
)


class _FakeLLM:
    """Capture the probe call and return or raise one configured outcome."""

    def __init__(
        self,
        *,
        response: LLMResponse | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[object, LLMCallConfig]] = []

    async def ainvoke(
        self,
        messages: object,
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        """Record one interface invocation before returning its result."""

        self.calls.append((messages, config))
        if self.error is not None:
            raise self.error
        if self.response is None:
            raise AssertionError("fake LLM requires a response or error")
        return self.response


def _route_setting() -> CognitionRouteSettingV1:
    """Build one private test route with the exact V3 chain constraints."""

    setting = CognitionRouteSettingV1(
        base_url="https://private.example/v1",
        api_key="private-credential",
        model="chain-model",
        max_completion_tokens=8_192,
        thinking_enabled=False,
        context_window_tokens=50_000,
    )
    return setting


def _llm_response(*, usage: dict[str, object]) -> LLMResponse:
    """Build one normalized success response for the fake interface."""

    backend = BackendDescriptor(
        route_name=V3_CHAIN_ROUTE_NAME,
        backend_kind="openai_compatible",
        model_family="unknown",
        model="chain-model",
        normalized_base_url="https://private.example/v1",
        thinking_strategy="disabled",
        confidence="unknown",
        generation=0,
    )
    response = LLMResponse(
        content="OK",
        backend=backend,
        raw_response=None,
        usage=usage,
    )
    return response


def _payload(content: str) -> dict[str, object]:
    return {
        "payload_id": "p",
        "category": "anchor_only",
        "messages": [{"role": "system", "content": content}],
    }


def test_token_calibration_is_deterministic_and_meets_holdout_contract() -> None:
    """Calibration rounds safely and the holdout accepts zero underestimates."""

    calibration_payloads = [
        _payload("short calibration"),
        _payload("longer calibration payload"),
    ]
    holdout_payloads = [
        _payload("short holdout"),
        _payload("longer holdout payload"),
    ]
    calibration_tokens = [80, 100]
    holdout_tokens = [80, 100]

    first = compute_calibration_report(
        calibration_payloads,
        holdout_payloads,
        calibration_tokens,
        holdout_tokens,
    )
    second = compute_calibration_report(
        calibration_payloads,
        holdout_payloads,
        calibration_tokens,
        holdout_tokens,
    )

    assert first == second
    assert first.calibration_multiplier >= 1.00
    assert first.calibration_underestimates == 0
    assert first.holdout_underestimates == 0
    assert first.holdout_median_overestimate <= 0.35
    assert first.accepted is True


def test_overflow_probe_payload_is_exact_deterministic_and_high_density() -> None:
    """The synthetic probe input is exact, reproducible, and tokenizer-hostile."""

    for char_count in (0, 1, 31, 4_096, 4_097):
        payload = build_overflow_probe_payload(char_count)
        assert len(payload) == char_count
        assert payload == build_overflow_probe_payload(char_count)

    large_payload = build_overflow_probe_payload(260_000)
    assert len(large_payload) == 260_000
    assert len(set(large_payload)) >= 80
    longest_run = max(
        len(tuple(run))
        for _, run in itertools.groupby(large_payload)
    )
    assert longest_run < 12
    assert estimate_message_tokens([large_payload]) > 50_176

    with pytest.raises(ValueError, match="cannot be negative"):
        build_overflow_probe_payload(-1)


def test_overflow_probe_cli_uses_the_payload_generator(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI obtains its payload from the deterministic generator."""

    requested_length = 32
    calls: list[int] = []

    def recording_generator(char_count: int) -> str:
        calls.append(char_count)
        return "Ab3!"

    monkeypatch.setattr(
        overflow_probe_module,
        "build_overflow_probe_payload",
        recording_generator,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_cognition_v3_context_overflow",
            "--route-name",
            "CHAIN",
            "--context-window-tokens",
            "50000",
            "--payload-char-count",
            str(requested_length),
        ],
    )

    assert overflow_probe_module.main() == 0
    cli_evidence = json.loads(capsys.readouterr().out)
    assert calls == [requested_length]
    assert cli_evidence["payload_estimate_tokens"] == estimate_message_tokens(
        ["Ab3!"]
    )


def test_overflow_probe_dry_run_is_effect_free_and_validates_route_contract(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dry run estimates a synthetic payload and never invokes a provider."""

    generated_payload = build_overflow_probe_payload(250_000)
    oversized = run_overflow_probe_dry_run(
        route_name="CHAIN",
        declared_context_window_tokens=50_000,
        payload_messages=(generated_payload,),
    )
    assert oversized.dry_run is True
    assert oversized.payload_exceeds_declared_window is True
    assert oversized.payload_estimate_tokens > 50_000

    with pytest.raises(ValueError, match="declared window"):
        run_overflow_probe_dry_run(
            route_name="CHAIN",
            declared_context_window_tokens=10_000,
            payload_messages=("x" * 1_000,),
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_cognition_v3_context_overflow",
            "--route-name",
            "CHAIN",
            "--context-window-tokens",
            "50000",
            "--payload-char-count",
            "250000",
        ],
    )
    assert overflow_probe_main() == 0
    cli_evidence = json.loads(capsys.readouterr().out)
    assert cli_evidence == {
        "declared_context_window_tokens": 50_000,
        "dry_run": True,
        "payload_estimate_tokens": oversized.payload_estimate_tokens,
        "payload_exceeds_declared_window": True,
        "route_name": "CHAIN",
    }


@pytest.mark.asyncio
async def test_overflow_probe_live_success_is_cutover_blocking_evidence() -> None:
    """A provider success remains visible after one interface invocation."""

    route_setting = _route_setting()
    llm = _FakeLLM(
        response=_llm_response(
            usage={"input_tokens": 65_100, "output_tokens": 1},
        ),
    )
    evidence = await run_overflow_probe_live(
        llm=llm,
        route_setting=route_setting,
        route_name=V3_CHAIN_ROUTE_NAME,
        declared_context_window_tokens=50_000,
        payload_messages=(build_overflow_probe_payload(250_000),),
        timeout_seconds=45.0,
    )

    assert evidence.disposition == "success"
    assert evidence.schema_version == "cognition_v3_context_overflow_probe.v1"
    assert evidence.dry_run is False
    assert evidence.wall_time_ms >= 0
    assert evidence.payload_exceeds_declared_window is True
    assert evidence.payload_character_count == 250_000
    assert evidence.usage_reported is True
    assert evidence.usage == {"input_tokens": 65_100, "output_tokens": 1}
    assert evidence.response_content_characters == 2
    assert evidence.response_content_sha256
    assert evidence.endpoint_sha256 == hashlib.sha256(
        route_setting.base_url.encode("utf-8"),
    ).hexdigest()
    assert len(llm.calls) == 1
    _, call_config = llm.calls[0]
    assert call_config.route_name == V3_CHAIN_ROUTE_NAME
    assert call_config.base_url == route_setting.base_url
    assert call_config.api_key == route_setting.api_key
    assert call_config.model == route_setting.model
    assert call_config.max_completion_tokens == 8_192
    assert call_config.context_window_tokens == 50_000
    assert call_config.timeout_seconds == 45.0
    serialized_evidence = json.dumps(asdict(evidence), sort_keys=True)
    assert route_setting.base_url not in serialized_evidence
    assert route_setting.api_key not in serialized_evidence
    assert "ttft" not in serialized_evidence.lower()


@pytest.mark.asyncio
async def test_overflow_probe_live_success_preserves_missing_usage() -> None:
    """A success without provider usage remains explicitly inconclusive."""

    route_setting = _route_setting()
    llm = _FakeLLM(response=_llm_response(usage={}))
    evidence = await run_overflow_probe_live(
        llm=llm,
        route_setting=route_setting,
        route_name=V3_CHAIN_ROUTE_NAME,
        declared_context_window_tokens=50_000,
        payload_messages=("x" * 250_000,),
        timeout_seconds=45.0,
    )

    assert evidence.disposition == "success"
    assert evidence.usage_reported is False
    assert evidence.usage == {}
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_overflow_probe_live_records_expected_provider_rejection() -> None:
    """One provider bad request records a sanitized expected rejection."""

    route_setting = _route_setting()
    request = httpx.Request(
        "POST",
        f"{route_setting.base_url}/chat/completions",
    )
    response = httpx.Response(400, request=request)
    error = BadRequestError(
        (
            "request (200861 tokens) exceeds the available context size "
            "(50176 tokens), try increasing it at "
            f"{route_setting.base_url} using {route_setting.api_key}"
        ),
        response=response,
        body={
            "error": {
                "message": (
                    "request (200861 tokens) exceeds the available context "
                    "size (50176 tokens), try increasing it"
                ),
                "type": "exceed_context_size_error",
                "code": "exceed_context_size_error",
            }
        },
    )
    llm = _FakeLLM(error=error)
    evidence = await run_overflow_probe_live(
        llm=llm,
        route_setting=route_setting,
        route_name=V3_CHAIN_ROUTE_NAME,
        declared_context_window_tokens=50_000,
        payload_messages=("x" * 250_000,),
        timeout_seconds=45.0,
    )

    assert evidence.disposition == "expected_rejection"
    assert evidence.provider_status_code == 400
    assert evidence.usage_reported is False
    assert evidence.usage == {}
    assert evidence.error_type == "BadRequestError"
    assert "exceeds the available context size" in evidence.error_message
    assert len(llm.calls) == 1
    serialized_evidence = json.dumps(asdict(evidence), sort_keys=True)
    assert route_setting.base_url not in serialized_evidence
    assert route_setting.api_key not in serialized_evidence


@pytest.mark.asyncio
async def test_overflow_probe_live_keeps_unrelated_bad_request_inconclusive() -> None:
    """An unrelated provider rejection cannot prove context enforcement."""

    route_setting = _route_setting()
    request = httpx.Request(
        "POST",
        f"{route_setting.base_url}/chat/completions",
    )
    response = httpx.Response(400, request=request)
    error = BadRequestError(
        "unsupported request field",
        response=response,
        body={"error": "unsupported request field"},
    )
    llm = _FakeLLM(error=error)
    evidence = await run_overflow_probe_live(
        llm=llm,
        route_setting=route_setting,
        route_name=V3_CHAIN_ROUTE_NAME,
        declared_context_window_tokens=50_000,
        payload_messages=("x" * 250_000,),
        timeout_seconds=45.0,
    )

    assert evidence.disposition == "transport_failure"
    assert evidence.provider_status_code == 400
    assert evidence.error_type == "BadRequestError"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_overflow_probe_live_records_transport_failure_once() -> None:
    """One transport failure stays distinct from provider rejection."""

    route_setting = _route_setting()
    request = httpx.Request(
        "POST",
        f"{route_setting.base_url}/chat/completions",
    )
    error = httpx.ConnectError(
        (
            f"cannot reach {route_setting.base_url} with "
            f"{route_setting.api_key}"
        ),
        request=request,
    )
    llm = _FakeLLM(error=error)
    evidence = await run_overflow_probe_live(
        llm=llm,
        route_setting=route_setting,
        route_name=V3_CHAIN_ROUTE_NAME,
        declared_context_window_tokens=50_000,
        payload_messages=("x" * 250_000,),
        timeout_seconds=45.0,
    )

    assert evidence.disposition == "transport_failure"
    assert evidence.provider_status_code is None
    assert evidence.usage_reported is False
    assert evidence.error_type == "ConnectError"
    assert len(llm.calls) == 1
    serialized_evidence = json.dumps(asdict(evidence), sort_keys=True)
    assert route_setting.base_url not in serialized_evidence
    assert route_setting.api_key not in serialized_evidence


@pytest.mark.asyncio
async def test_overflow_probe_live_rejects_wrong_route_or_fitting_payload() -> None:
    """Live mode validates route identity and overflow before model work."""

    route_setting = _route_setting()
    llm = _FakeLLM(response=_llm_response(usage={"input_tokens": 1}))
    with pytest.raises(ValueError, match=V3_CHAIN_ROUTE_NAME):
        await run_overflow_probe_live(
            llm=llm,
            route_setting=route_setting,
            route_name="COGNITION_V3_SIDECAR_LLM",
            declared_context_window_tokens=50_000,
            payload_messages=("x" * 250_000,),
            timeout_seconds=45.0,
        )
    with pytest.raises(ValueError, match="must exceed"):
        await run_overflow_probe_live(
            llm=llm,
            route_setting=route_setting,
            route_name=V3_CHAIN_ROUTE_NAME,
            declared_context_window_tokens=50_000,
            payload_messages=("x" * 1_000,),
            timeout_seconds=45.0,
        )

    assert llm.calls == []
