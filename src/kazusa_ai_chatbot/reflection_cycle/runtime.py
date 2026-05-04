"""Read-only runtime for reflection-cycle evaluation."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.reflection_cycle.models import (
    ChannelReflectionResult,
    DailySynthesisResult,
    READONLY_REFLECTION_PROMPT_VERSION,
    ReflectionEvaluationResult,
    ReflectionInputSet,
    ReflectionLLMResult,
    ReflectionScopeInput,
)
from kazusa_ai_chatbot.reflection_cycle.prompts import (
    build_daily_synthesis_prompt,
    build_hourly_reflection_prompt,
    build_skipped_daily_result,
    build_skipped_hourly_result,
    run_daily_synthesis_llm,
    run_hourly_reflection_llm,
)
from kazusa_ai_chatbot.reflection_cycle.selector import (
    collect_reflection_inputs,
    normalize_utc_datetime,
)


async def run_readonly_reflection_evaluation(
    *,
    lookback_hours: int = 24,
    now: datetime | None = None,
    output_dir: str,
    use_real_llm: bool,
) -> ReflectionEvaluationResult:
    """Collect inputs, optionally run LLM prompts, and write a local artifact.

    Args:
        lookback_hours: Requested message evaluation window.
        now: Optional deterministic clock value.
        output_dir: Local artifact directory.
        use_real_llm: Whether to call the configured consolidation LLM.

    Returns:
        Evaluation result with prompt diagnostics and the artifact path.
    """

    input_set = await collect_reflection_inputs(
        lookback_hours=lookback_hours,
        now=now,
    )
    channel_results = await _run_channel_reflections(
        input_set=input_set,
        use_real_llm=use_real_llm,
    )
    hourly_results = [
        hourly_result
        for channel_result in channel_results
        for hourly_result in channel_result.hourly_results
    ]
    daily_results = [
        channel_result.daily_result
        for channel_result in channel_results
    ]
    artifact_path = _write_artifact(
        output_dir=Path(output_dir),
        input_set=input_set,
        channel_results=channel_results,
        use_real_llm=use_real_llm,
    )
    result = ReflectionEvaluationResult(
        input_set=input_set,
        channel_results=channel_results,
        hourly_results=hourly_results,
        daily_results=daily_results,
        artifact_path=artifact_path,
    )
    return result


async def _run_channel_reflections(
    *,
    input_set: ReflectionInputSet,
    use_real_llm: bool,
) -> list[ChannelReflectionResult]:
    """Run hourly buckets and daily synthesis for each selected channel."""

    channel_results: list[ChannelReflectionResult] = []
    for channel_scope in input_set.selected_scopes:
        hourly_scopes = _split_scope_into_hourly_scopes(channel_scope)
        hourly_results = await _run_hourly_reflections(
            hourly_scopes=hourly_scopes,
            use_real_llm=use_real_llm,
        )
        channel_input_set = _channel_input_set(
            input_set=input_set,
            hourly_scopes=hourly_scopes,
            channel_scope=channel_scope,
        )
        daily_result = await _run_daily_synthesis(
            input_set=channel_input_set,
            channel_scope=channel_scope,
            hourly_results=hourly_results,
            use_real_llm=use_real_llm,
        )
        channel_result = ChannelReflectionResult(
            channel_scope=channel_scope,
            hourly_scopes=hourly_scopes,
            hourly_results=hourly_results,
            daily_result=daily_result,
        )
        channel_results.append(channel_result)
    return_value = channel_results
    return return_value


async def _run_hourly_reflections(
    *,
    hourly_scopes: list[ReflectionScopeInput],
    use_real_llm: bool,
) -> list[ReflectionLLMResult]:
    """Run or build prompt-only reflection results for message-bearing hours."""

    hourly_results: list[ReflectionLLMResult] = []
    for hourly_scope in hourly_scopes:
        if not use_real_llm:
            hourly_results.append(build_skipped_hourly_result(hourly_scope))
            continue
        prompt = build_hourly_reflection_prompt(hourly_scope)
        hourly_result = await run_hourly_reflection_llm(
            scope_ref=hourly_scope.scope_ref,
            prompt=prompt,
        )
        hourly_results.append(hourly_result)
    return_value = hourly_results
    return return_value


async def _run_daily_synthesis(
    *,
    input_set: ReflectionInputSet,
    channel_scope: ReflectionScopeInput,
    hourly_results: list[ReflectionLLMResult],
    use_real_llm: bool,
) -> DailySynthesisResult:
    """Run or build prompt-only daily synthesis for one channel."""

    if not use_real_llm:
        return_value = build_skipped_daily_result(
            input_set=input_set,
            channel_scope=channel_scope,
            hourly_results=hourly_results,
        )
        return return_value
    prompt = build_daily_synthesis_prompt(
        input_set=input_set,
        channel_scope=channel_scope,
        hourly_results=hourly_results,
    )
    result = await run_daily_synthesis_llm(prompt=prompt)
    return result


def _split_scope_into_hourly_scopes(
    channel_scope: ReflectionScopeInput,
) -> list[ReflectionScopeInput]:
    """Split one selected channel into message-bearing UTC hour buckets."""

    buckets: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for message in channel_scope.messages:
        timestamp = _parse_message_timestamp(str(message["timestamp"]))
        bucket_start = _hour_start(timestamp)
        buckets[bucket_start].append(message)

    hourly_scopes: list[ReflectionScopeInput] = []
    for bucket_start in sorted(buckets):
        messages = sorted(
            buckets[bucket_start],
            key=lambda item: str(item["timestamp"]),
        )
        assistant_count = sum(
            1
            for message in messages
            if message.get("role") == "assistant"
        )
        user_count = sum(
            1
            for message in messages
            if message.get("role") == "user"
        )
        hourly_scope = ReflectionScopeInput(
            scope_ref=_hourly_scope_ref(channel_scope.scope_ref, bucket_start),
            platform=channel_scope.platform,
            platform_channel_id=channel_scope.platform_channel_id,
            channel_type=channel_scope.channel_type,
            assistant_message_count=assistant_count,
            user_message_count=user_count,
            total_message_count=len(messages),
            first_timestamp=str(messages[0]["timestamp"]),
            last_timestamp=str(messages[-1]["timestamp"]),
            messages=messages,
        )
        hourly_scopes.append(hourly_scope)
    return_value = hourly_scopes
    return return_value


def _channel_input_set(
    *,
    input_set: ReflectionInputSet,
    hourly_scopes: list[ReflectionScopeInput],
    channel_scope: ReflectionScopeInput,
) -> ReflectionInputSet:
    """Build a per-channel input set for daily synthesis."""

    channel_input_set = ReflectionInputSet(
        lookback_hours=input_set.lookback_hours,
        requested_start=input_set.requested_start,
        requested_end=input_set.requested_end,
        effective_start=input_set.effective_start,
        effective_end=input_set.effective_end,
        fallback_used=input_set.fallback_used,
        fallback_reason=input_set.fallback_reason,
        selected_scopes=hourly_scopes,
        query_diagnostics={
            "channel_scope_ref": channel_scope.scope_ref,
            "active_hour_count": len(hourly_scopes),
        },
    )
    return channel_input_set


def _parse_message_timestamp(value: str) -> datetime:
    """Parse one conversation timestamp and normalize it to UTC."""

    parsed_timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    normalized_timestamp = parsed_timestamp.astimezone(timezone.utc)
    return normalized_timestamp


def _hour_start(value: datetime) -> datetime:
    """Return the UTC hour start for one timestamp."""

    hour_start = value.astimezone(timezone.utc).replace(
        minute=0,
        second=0,
        microsecond=0,
    )
    return hour_start


def _hourly_scope_ref(channel_scope_ref: str, hour_start: datetime) -> str:
    """Build a stable hourly reference under the selected channel scope."""

    suffix = hour_start.strftime("%Y%m%dT%HZ")
    return_value = f"{channel_scope_ref}_{suffix}"
    return return_value


def _write_artifact(
    *,
    output_dir: Path,
    input_set: ReflectionInputSet,
    channel_results: list[ChannelReflectionResult],
    use_real_llm: bool,
) -> Path:
    """Write the read-only evaluation artifact under the requested directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = _artifact_path(output_dir)
    git_sha = _git_sha()
    payload = {
        "artifact_type": "reflection_cycle_readonly_evaluation",
        "artifact_version": 1,
        "prompt_version": READONLY_REFLECTION_PROMPT_VERSION,
        "git_sha": git_sha,
        "created_at": _utc_now_iso(),
        "readonly": True,
        "use_real_llm": use_real_llm,
        "lookback_hours": input_set.lookback_hours,
        "window": {
            "requested_start": input_set.requested_start,
            "requested_end": input_set.requested_end,
            "effective_start": input_set.effective_start,
            "effective_end": input_set.effective_end,
        },
        "fallback_used": input_set.fallback_used,
        "fallback_reason": input_set.fallback_reason,
        "selected_channels": [
            _scope_artifact_summary(scope)
            for scope in input_set.selected_scopes
        ],
        "query_diagnostics": input_set.query_diagnostics,
        "channel_summaries": [
            _channel_result_summary(result)
            for result in channel_results
        ],
        "hourly_reflections": [
            _hourly_result_artifact(
                channel_result=channel_result,
                hourly_scope=hourly_scope,
                hourly_result=hourly_result,
            )
            for channel_result in channel_results
            for hourly_scope, hourly_result in zip(
                channel_result.hourly_scopes,
                channel_result.hourly_results,
            )
        ],
        "daily_syntheses": [
            _daily_result_artifact(result)
            for result in channel_results
        ],
        "manual_review_notes": _manual_review_notes(
            input_set=input_set,
            use_real_llm=use_real_llm,
        ),
    }
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return artifact_path


def _scope_artifact_summary(scope: ReflectionScopeInput) -> dict[str, Any]:
    """Return selected-scope metadata without transcript rows."""

    summary = {
        "scope_ref": scope.scope_ref,
        "platform": scope.platform,
        "platform_channel_id": scope.platform_channel_id,
        "channel_type": scope.channel_type,
        "assistant_message_count": scope.assistant_message_count,
        "user_message_count": scope.user_message_count,
        "total_message_count": scope.total_message_count,
        "first_timestamp": scope.first_timestamp,
        "last_timestamp": scope.last_timestamp,
        "prompt_message_rows": len(scope.messages),
    }
    return summary


def _channel_result_summary(result: ChannelReflectionResult) -> dict[str, Any]:
    """Return channel-level reflection counts for the artifact."""

    summary = {
        "channel_scope_ref": result.channel_scope.scope_ref,
        "platform": result.channel_scope.platform,
        "platform_channel_id": result.channel_scope.platform_channel_id,
        "channel_type": result.channel_scope.channel_type,
        "source_message_rows": len(result.channel_scope.messages),
        "active_hour_count": len(result.hourly_scopes),
    }
    return summary


def _hourly_result_artifact(
    *,
    channel_result: ChannelReflectionResult,
    hourly_scope: ReflectionScopeInput,
    hourly_result: ReflectionLLMResult,
) -> dict[str, Any]:
    """Return artifact payload for one hourly reflection result."""

    hour_start = _hour_start(_parse_message_timestamp(hourly_scope.first_timestamp))
    hour_end = hour_start + timedelta(hours=1)
    artifact = {
        "channel_scope_ref": channel_result.channel_scope.scope_ref,
        "hourly_scope_ref": hourly_scope.scope_ref,
        "prompt_version": READONLY_REFLECTION_PROMPT_VERSION,
        "platform": hourly_scope.platform,
        "platform_channel_id": hourly_scope.platform_channel_id,
        "channel_type": hourly_scope.channel_type,
        "hour_start": hour_start.isoformat(),
        "hour_end": hour_end.isoformat(),
        "assistant_message_count": hourly_scope.assistant_message_count,
        "user_message_count": hourly_scope.user_message_count,
        "total_message_count": hourly_scope.total_message_count,
        "first_timestamp": hourly_scope.first_timestamp,
        "last_timestamp": hourly_scope.last_timestamp,
        "llm_skipped": hourly_result.llm_skipped,
        "prompt_chars": hourly_result.prompt.prompt_chars,
        "prompt_preview": hourly_result.prompt.prompt_preview,
        "prompt_validation_warnings": hourly_result.prompt.validation_warnings,
        "raw_output": hourly_result.raw_output,
        "parsed_output": hourly_result.parsed_output,
        "validation_warnings": hourly_result.validation_warnings,
    }
    return artifact


def _daily_result_artifact(result: ChannelReflectionResult) -> dict[str, Any]:
    """Return artifact payload for the daily synthesis result."""

    daily_result = result.daily_result
    artifact = {
        "channel_scope_ref": result.channel_scope.scope_ref,
        "prompt_version": READONLY_REFLECTION_PROMPT_VERSION,
        "platform": result.channel_scope.platform,
        "platform_channel_id": result.channel_scope.platform_channel_id,
        "channel_type": result.channel_scope.channel_type,
        "active_hour_count": len(result.hourly_scopes),
        "llm_skipped": daily_result.llm_skipped,
        "prompt_chars": daily_result.prompt.prompt_chars,
        "prompt_preview": daily_result.prompt.prompt_preview,
        "prompt_validation_warnings": daily_result.prompt.validation_warnings,
        "raw_output": daily_result.raw_output,
        "parsed_output": daily_result.parsed_output,
        "validation_warnings": daily_result.validation_warnings,
    }
    return artifact


def _manual_review_notes(
    *,
    input_set: ReflectionInputSet,
    use_real_llm: bool,
) -> list[str]:
    """Build manual review notes for the local artifact."""

    notes = [
        (
            "The runtime performs conversation_history reads only and writes "
            "this local artifact only."
        ),
        "No memory, lore, scheduler, dispatcher, cognition, or chat path is invoked.",
        (
            "Daily synthesis prompt is built from compact active-hour slots "
            "for each channel, not raw transcripts or full hourly objects."
        ),
    ]
    if not use_real_llm:
        notes.append("LLM execution was disabled; review prompt previews only.")
    if input_set.fallback_used:
        notes.append(input_set.fallback_reason)
    if not input_set.selected_scopes:
        notes.append(
            "No monitored channel was available in requested or fallback windows."
        )
    return notes


def _artifact_path(output_dir: Path) -> Path:
    """Return a unique artifact path for this local evaluation run."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return_value = output_dir / f"readonly_reflection_evaluation_{timestamp}.json"
    return return_value


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO string."""

    return_value = normalize_utc_datetime(None).isoformat()
    return return_value


def _git_sha() -> str:
    """Return the current repository commit hash for local artifacts."""

    repo_root = Path(__file__).resolve().parents[3]
    try:
        completed_process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        git_sha = f"unavailable: {exc}"
        return git_sha

    if completed_process.returncode != 0:
        error_text = completed_process.stderr.strip()
        if not error_text:
            error_text = f"git exited with {completed_process.returncode}"
        git_sha = f"unavailable: {error_text}"
        return git_sha

    git_sha = completed_process.stdout.strip()
    return git_sha
