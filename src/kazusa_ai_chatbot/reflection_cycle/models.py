"""Data contracts for read-only reflection-cycle evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


READONLY_REFLECTION_MAX_SCOPES = 25
READONLY_REFLECTION_MAX_MESSAGES_PER_SCOPE = 120
READONLY_REFLECTION_MAX_MESSAGE_CHARS = 280
READONLY_REFLECTION_HOURLY_PROMPT_MAX_CHARS = 8000
READONLY_REFLECTION_DAILY_PROMPT_MAX_CHARS = 25000
READONLY_REFLECTION_ARTIFACT_PROMPT_PREVIEW_CHARS = 2000
READONLY_REFLECTION_MONITOR_ELIGIBILITY_HOURS = 24
READONLY_REFLECTION_FALLBACK_LOOKBACK_HOURS = 168
READONLY_REFLECTION_DAILY_SLOT_TEXT_CHARS = 180


HOURLY_REQUIRED_FIELDS = (
    "topic_summary",
    "participant_observations",
    "conversation_quality_feedback",
    "privacy_notes",
    "confidence",
)
DAILY_REQUIRED_FIELDS = (
    "day_summary",
    "active_hour_summaries",
    "cross_hour_topics",
    "conversation_quality_patterns",
    "privacy_risks",
    "synthesis_limitations",
    "confidence",
)


@dataclass
class ReflectionScopeInput:
    """Messages and counters for one monitored conversation scope."""

    scope_ref: str
    platform: str
    platform_channel_id: str
    channel_type: str
    assistant_message_count: int
    user_message_count: int
    total_message_count: int
    first_timestamp: str
    last_timestamp: str
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReflectionInputSet:
    """Complete read-only input collection for one reflection evaluation."""

    lookback_hours: int
    requested_start: str
    requested_end: str
    effective_start: str
    effective_end: str
    fallback_used: bool
    fallback_reason: str
    selected_scopes: list[ReflectionScopeInput]
    query_diagnostics: dict[str, Any]


@dataclass
class PromptBuildResult:
    """Prompt text plus diagnostics for one reflection LLM call."""

    system_prompt: str
    human_payload: dict[str, Any]
    human_prompt: str
    prompt_chars: int
    prompt_preview: str
    validation_warnings: list[str]


@dataclass
class ReflectionLLMResult:
    """Raw and parsed output for one reflection prompt."""

    scope_ref: str
    prompt: PromptBuildResult
    raw_output: str
    parsed_output: dict[str, Any]
    validation_warnings: list[str]
    llm_skipped: bool = False


@dataclass
class DailySynthesisResult:
    """Raw and parsed output for the daily synthesis prompt."""

    prompt: PromptBuildResult
    raw_output: str
    parsed_output: dict[str, Any]
    validation_warnings: list[str]
    llm_skipped: bool = False


@dataclass
class ChannelReflectionResult:
    """Hourly and daily reflection results for one selected channel."""

    channel_scope: ReflectionScopeInput
    hourly_scopes: list[ReflectionScopeInput]
    hourly_results: list[ReflectionLLMResult]
    daily_result: DailySynthesisResult


@dataclass
class ReflectionEvaluationResult:
    """Result returned by the read-only reflection evaluation runtime."""

    input_set: ReflectionInputSet
    channel_results: list[ChannelReflectionResult]
    hourly_results: list[ReflectionLLMResult]
    daily_results: list[DailySynthesisResult]
    artifact_path: Path
