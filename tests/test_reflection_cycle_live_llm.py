"""Live LLM checks for read-only reflection-cycle prompts."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.reflection_cycle.models import (
    DAILY_REQUIRED_FIELDS,
    HOURLY_REQUIRED_FIELDS,
    ReflectionInputSet,
    ReflectionLLMResult,
    ReflectionScopeInput,
)
from kazusa_ai_chatbot.reflection_cycle.prompts import (
    build_daily_synthesis_prompt,
    build_hourly_reflection_prompt,
    run_daily_synthesis_llm,
    run_hourly_reflection_llm,
)
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


async def _skip_if_endpoint_unavailable(base_url: str) -> None:
    """Skip live tests when the configured LLM endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f'{base_url.rstrip("/")}/models')
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {base_url}: {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{base_url}"
        )
    model_payload = response.json()
    models = model_payload.get("data", [])
    if not models:
        pytest.skip(f"LLM endpoint has no loaded models: {base_url}")
    ping_payload = {
        "model": CONSOLIDATION_LLM_MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {CONSOLIDATION_LLM_API_KEY}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        ping_response = await client.post(
            f'{base_url.rstrip("/")}/chat/completions',
            json=ping_payload,
            headers=headers,
        )
    if ping_response.status_code >= 400:
        pytest.skip(
            "LLM endpoint is reachable but chat completion is unavailable: "
            f"{ping_response.status_code} {ping_response.text}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the consolidation LLM route is reachable."""

    await _skip_if_endpoint_unavailable(CONSOLIDATION_LLM_BASE_URL)


async def test_live_readonly_private_scope_reflection(ensure_live_llm) -> None:
    """Private-scope hourly prompt should produce the required schema."""

    del ensure_live_llm
    result = await _run_hourly_case(_scope("private"))

    _assert_required_fields(result.parsed_output, HOURLY_REQUIRED_FIELDS)
    _assert_no_validation_warnings(result.validation_warnings)


async def test_live_readonly_group_scope_reflection(ensure_live_llm) -> None:
    """Group-scope hourly prompt should produce the required schema."""

    del ensure_live_llm
    result = await _run_hourly_case(_scope("group"))

    _assert_required_fields(result.parsed_output, HOURLY_REQUIRED_FIELDS)
    _assert_no_validation_warnings(result.validation_warnings)


async def test_live_readonly_daily_synthesis(ensure_live_llm) -> None:
    """Daily synthesis should consume hourly outputs and return its own schema."""

    del ensure_live_llm
    input_set = _input_set()
    hourly_results = [
        _parsed_hourly_result(input_set.selected_scopes[0]),
        _parsed_hourly_result(input_set.selected_scopes[1]),
    ]
    prompt = build_daily_synthesis_prompt(
        input_set=input_set,
        channel_scope=input_set.selected_scopes[0],
        hourly_results=hourly_results,
    )
    result = await run_daily_synthesis_llm(prompt=prompt)
    trace_path = write_llm_trace(
        "reflection_cycle_readonly_live_llm",
        "daily_synthesis",
        {
            "prompt_preview": prompt.prompt_preview,
            "raw_output": result.raw_output,
            "parsed_output": result.parsed_output,
            "validation_warnings": result.validation_warnings,
        },
    )

    _assert_required_fields(result.parsed_output, DAILY_REQUIRED_FIELDS)
    _assert_no_validation_warnings(result.validation_warnings)
    assert trace_path.exists()


async def _run_hourly_case(scope: ReflectionScopeInput) -> ReflectionLLMResult:
    """Run one hourly live LLM case and write an inspectable trace."""

    prompt = build_hourly_reflection_prompt(scope)
    result = await run_hourly_reflection_llm(
        scope_ref=scope.scope_ref,
        prompt=prompt,
    )
    trace_path = write_llm_trace(
        "reflection_cycle_readonly_live_llm",
        scope.scope_ref,
        {
            "channel_type": scope.channel_type,
            "prompt_preview": prompt.prompt_preview,
            "raw_output": result.raw_output,
            "parsed_output": result.parsed_output,
            "validation_warnings": result.validation_warnings,
        },
    )

    assert trace_path.exists()
    return result


def _assert_required_fields(
    parsed_output: dict[str, Any],
    required_fields: tuple[str, ...],
) -> None:
    """Assert all required schema fields are present."""

    for field_name in required_fields:
        assert field_name in parsed_output


def _assert_no_validation_warnings(warnings: list[str]) -> None:
    """Assert schema validation produced no warnings."""

    assert warnings == []


def _input_set() -> ReflectionInputSet:
    """Build the daily synthesis input fixture."""

    private_scope = _scope("private")
    later_private_scope = _scope("private")
    later_private_scope.scope_ref = "scope_live_private_later"
    later_private_scope.first_timestamp = "2026-05-03T22:00:00+00:00"
    later_private_scope.last_timestamp = "2026-05-03T22:05:00+00:00"
    for message in later_private_scope.messages:
        message["timestamp"] = "2026-05-03T22:00:00+00:00"
    input_set = ReflectionInputSet(
        lookback_hours=24,
        requested_start="2026-05-03T00:00:00+00:00",
        requested_end="2026-05-04T00:00:00+00:00",
        effective_start="2026-05-03T00:00:00+00:00",
        effective_end="2026-05-04T00:00:00+00:00",
        fallback_used=False,
        fallback_reason="",
        selected_scopes=[private_scope, later_private_scope],
        query_diagnostics={},
    )
    return input_set


def _parsed_hourly_result(scope: ReflectionScopeInput) -> ReflectionLLMResult:
    """Build a parsed hourly result for daily live synthesis."""

    prompt = build_hourly_reflection_prompt(scope)
    parsed_output = {
        "topic_summary": (
            f"{scope.channel_type} scope covered project planning and tone "
            "calibration."
        ),
        "participant_observations": [
            {
                "participant_ref": "participant_1",
                "observation": "Asked for structured help and reacted to direct guidance.",
                "evidence_strength": "medium",
            }
        ],
        "conversation_quality_feedback": [
            (
                "The character should stay concrete and avoid turning workflow "
                "discussion into personal lore."
            )
        ],
        "privacy_notes": [
            "No user-specific details are safe for persistence from this fixture."
        ],
        "confidence": "medium",
    }
    result = ReflectionLLMResult(
        scope_ref=scope.scope_ref,
        prompt=prompt,
        raw_output="{}",
        parsed_output=parsed_output,
        validation_warnings=[],
    )
    return result


def _scope(channel_type: str) -> ReflectionScopeInput:
    """Build a private or group scope fixture for live LLM tests."""

    if channel_type == "group":
        messages = [
            _user_message(
                "participant-a",
                "Can the character keep track of team goals without storing personal details?",
            ),
            _assistant_message(
                "She can summarize the team topic, but persistence needs stricter privacy checks."
            ),
            _user_message(
                "participant-b",
                "I mostly care that she notices when the group moves from design to testing.",
            ),
            _assistant_message(
                "Then the reflection should record topic movement, not individual identity."
            ),
        ]
    else:
        messages = [
            _user_message(
                "participant-a",
                "I want the character to reflect on today's planning conversation.",
            ),
            _assistant_message(
                "She can evaluate topic progress and response quality without writing memory yet."
            ),
            _user_message(
                "participant-a",
                "Good. Keep the approval based on useful output, not just schema validity.",
            ),
            _assistant_message(
                "Then the artifact should show raw output, warnings, and a manual review note."
            ),
        ]
    scope = ReflectionScopeInput(
        scope_ref=f"scope_live_{channel_type}",
        platform="qq",
        platform_channel_id=f"live-{channel_type}",
        channel_type=channel_type,
        assistant_message_count=2,
        user_message_count=2,
        total_message_count=4,
        first_timestamp="2026-05-03T23:00:00+00:00",
        last_timestamp="2026-05-03T23:05:00+00:00",
        messages=messages,
    )
    return scope


def _user_message(platform_user_id: str, body_text: str) -> dict[str, Any]:
    """Build one user fixture message."""

    message = {
        "role": "user",
        "platform_user_id": platform_user_id,
        "global_user_id": f"global-{platform_user_id}",
        "display_name": platform_user_id,
        "body_text": body_text,
        "attachments": [],
        "timestamp": "2026-05-03T23:00:00+00:00",
    }
    return message


def _assistant_message(body_text: str) -> dict[str, Any]:
    """Build one assistant fixture message."""

    message = {
        "role": "assistant",
        "platform_user_id": "platform-bot",
        "global_user_id": "character-global",
        "display_name": "Character",
        "body_text": body_text,
        "attachments": [],
        "timestamp": "2026-05-03T23:01:00+00:00",
    }
    return message
