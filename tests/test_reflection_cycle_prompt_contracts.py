"""Prompt contract tests for read-only reflection-cycle evaluation."""

from __future__ import annotations

from kazusa_ai_chatbot.reflection_cycle.models import (
    READONLY_REFLECTION_DAILY_PROMPT_MAX_CHARS,
    READONLY_REFLECTION_HOURLY_PROMPT_MAX_CHARS,
    ReflectionInputSet,
    ReflectionLLMResult,
    ReflectionScopeInput,
)
from kazusa_ai_chatbot.reflection_cycle.projection import (
    build_hourly_reflection_payload,
    build_prompt_result,
    validate_daily_synthesis_output,
    validate_hourly_reflection_output,
)
from kazusa_ai_chatbot.reflection_cycle.prompts import (
    build_daily_synthesis_prompt,
    build_hourly_reflection_prompt,
    build_skipped_daily_result,
    build_skipped_hourly_result,
)


def test_hourly_prompt_is_bounded_and_deidentified() -> None:
    """Hourly prompt should fit the fixed budget and hide participant ids."""

    scope = _scope_with_long_messages("private")

    prompt = build_hourly_reflection_prompt(scope)

    assert prompt.prompt_chars <= READONLY_REFLECTION_HOURLY_PROMPT_MAX_CHARS
    assert "global-user-1" not in prompt.human_prompt
    assert "platform-user-1" not in prompt.human_prompt
    assert "Alice Example" not in prompt.human_prompt
    assert "platform_channel_id" not in prompt.human_prompt
    assert "participant_1" in prompt.human_prompt
    assert "evaluation_mode" in prompt.human_payload
    assert "scope_metadata" in prompt.human_payload
    assert "conversation" in prompt.human_payload
    assert "review_questions" in prompt.human_payload
    assert "评估模式" not in prompt.human_payload
    assert "评审问题" not in prompt.human_payload
    conversation = prompt.human_payload["conversation"]
    messages = conversation["messages"]
    assert "text" in messages[0]
    assert "lore_candidates" not in prompt.system_prompt
    assert "progress_projection" not in prompt.system_prompt
    assert "open_loops" not in prompt.system_prompt


def test_reflection_prompts_require_chinese_free_text() -> None:
    """Hourly and daily prompts should force generated free text into Chinese."""

    input_set = _input_set()
    hourly_result = _hourly_result(input_set.selected_scopes[0])
    hourly_prompt = build_hourly_reflection_prompt(input_set.selected_scopes[0])
    daily_prompt = build_daily_synthesis_prompt(
        input_set=input_set,
        channel_scope=input_set.selected_scopes[0],
        hourly_results=[hourly_result],
    )

    assert "# 语言政策" in hourly_prompt.system_prompt
    assert "# 语言政策" in daily_prompt.system_prompt
    assert "所有由你新生成的内部自由文本字段都必须使用简体中文" in hourly_prompt.system_prompt
    assert "所有由你新生成的内部自由文本字段都必须使用简体中文" in daily_prompt.system_prompt
    assert "# 输入格式" in hourly_prompt.system_prompt
    assert "# 输出格式" in daily_prompt.system_prompt
    assert "# Generation Procedure" not in hourly_prompt.system_prompt
    assert "# Input Format" not in daily_prompt.system_prompt
    assert '"evaluation_mode"' in hourly_prompt.system_prompt
    assert '"active_hour_slots"' in daily_prompt.system_prompt
    assert '"channel"' in daily_prompt.system_prompt
    assert '"评估模式"' not in hourly_prompt.system_prompt
    assert '"范围元数据"' not in daily_prompt.system_prompt
    assert "用简体中文写出" not in hourly_prompt.system_prompt
    assert "用简体中文写出" not in daily_prompt.system_prompt
    assert "confidence" in hourly_prompt.system_prompt
    assert "low|medium|high" in daily_prompt.system_prompt
    assert (
        "exact hour value copied from active_hour_slots"
        in daily_prompt.system_prompt
    )


def test_daily_prompt_consumes_hourly_outputs_not_raw_transcripts() -> None:
    """Daily synthesis must not receive raw transcript rows."""

    input_set = _input_set()
    hourly_result = _hourly_result(input_set.selected_scopes[0])

    prompt = build_daily_synthesis_prompt(
        input_set=input_set,
        channel_scope=input_set.selected_scopes[0],
        hourly_results=[hourly_result],
    )

    assert prompt.prompt_chars <= READONLY_REFLECTION_DAILY_PROMPT_MAX_CHARS
    assert "Can you help me with the budget migration" not in prompt.human_prompt
    assert "topic_summary" in prompt.human_prompt
    assert "active_hour_slots" in prompt.human_payload
    assert "hourly_reflections" not in prompt.human_payload
    assert "scope_ref" not in prompt.human_prompt
    assert "小时反思" not in prompt.human_payload
    hourly_projection = prompt.human_payload["active_hour_slots"][0]
    assert "hour" in hourly_projection
    assert "conversation_quality_feedback" in hourly_projection
    assert "participant_observations" not in hourly_projection
    assert "active_hour_summaries" in prompt.system_prompt


def test_daily_prompt_marks_omitted_compact_items() -> None:
    """Daily prompt should expose when compact hourly lists were shortened."""

    input_set = _input_set()
    hourly_result = _hourly_result(input_set.selected_scopes[0])
    hourly_result.parsed_output["conversation_quality_feedback"] = [
        "The character stayed concrete.",
        "The character asked useful follow-up questions.",
        "The character avoided broad speculation.",
    ]
    hourly_result.parsed_output["privacy_notes"] = [
        "No obvious privacy risk.",
        "Avoid storing names if later details appear.",
    ]

    prompt = build_daily_synthesis_prompt(
        input_set=input_set,
        channel_scope=input_set.selected_scopes[0],
        hourly_results=[hourly_result],
    )

    hourly_projection = prompt.human_payload["active_hour_slots"][0]
    assert hourly_projection["conversation_quality_feedback"] == [
        "The character stayed concrete."
    ]
    assert hourly_projection["conversation_quality_feedback_omitted_count"] == 2
    assert hourly_projection["privacy_notes"] == ["No obvious privacy risk."]
    assert hourly_projection["privacy_notes_omitted_count"] == 1


def test_prompt_builder_drops_old_messages_when_over_budget() -> None:
    """Prompt budget enforcement should remove oldest rows before invocation."""

    scope = _scope_with_many_long_messages()
    payload = build_hourly_reflection_payload(scope)

    prompt = build_prompt_result(
        system_prompt="instruction",
        human_payload=payload,
        max_prompt_chars=4500,
    )

    messages = prompt.human_payload["conversation"]["messages"]
    assert prompt.prompt_chars <= 4500
    assert len(messages) < len(payload["conversation"]["messages"])
    assert "message-0" not in prompt.human_prompt
    assert "message-19" in prompt.human_prompt
    assert prompt.validation_warnings
    assert prompt.validation_warnings[0].startswith("Prompt 超出预算")


def test_hourly_output_validation_warns_on_forward_fields() -> None:
    """Hourly schema should stay narrow for read-only approval."""

    warnings = validate_hourly_reflection_output({
        "topic_summary": "Project planning.",
        "participant_observations": [],
        "conversation_quality_feedback": [],
        "privacy_notes": [],
        "confidence": "medium",
        "lore_candidates": [],
    })

    assert "出现未请求的前瞻字段: lore_candidates" in warnings


def test_daily_output_validation_requires_daily_fields() -> None:
    """Daily schema validation should require daily synthesis fields."""

    warnings = validate_daily_synthesis_output({
        "day_summary": "Daily summary.",
        "active_hour_summaries": [],
        "cross_hour_topics": [],
        "conversation_quality_patterns": [],
        "privacy_risks": [],
        "synthesis_limitations": [],
        "confidence": "high",
    })

    assert warnings == []


def test_daily_output_validation_rejects_rewritten_hours() -> None:
    """Daily summaries should preserve exact hour labels from input slots."""

    warnings = validate_daily_synthesis_output(
        {
            "day_summary": "Daily summary.",
            "active_hour_summaries": [
                {
                    "hour": "2026-05-04T10:00:00+12:00",
                    "summary": "Converted timezone.",
                }
            ],
            "cross_hour_topics": [],
            "conversation_quality_patterns": [],
            "privacy_risks": [],
            "synthesis_limitations": [],
            "confidence": "high",
        },
        allowed_hours={"2026-05-03T22:00:00+00:00"},
    )

    assert warnings == [(
        "active_hour_summaries.hour 未逐字复制输入 hour: "
        "2026-05-04T10:00:00+12:00"
    )]


def test_skipped_results_preserve_prompt_contracts() -> None:
    """Prompt-only evaluation should still expose inspectable prompt diagnostics."""

    input_set = _input_set()
    hourly_result = build_skipped_hourly_result(input_set.selected_scopes[0])
    daily_result = build_skipped_daily_result(
        input_set=input_set,
        channel_scope=input_set.selected_scopes[0],
        hourly_results=[hourly_result],
    )

    assert hourly_result.llm_skipped is True
    assert daily_result.llm_skipped is True
    assert hourly_result.validation_warnings == ["已跳过 LLM 执行"]
    assert daily_result.validation_warnings == ["已跳过 LLM 执行"]
    assert hourly_result.prompt.prompt_preview
    assert daily_result.prompt.prompt_preview


def _input_set() -> ReflectionInputSet:
    """Build a deterministic input set for prompt contract tests."""

    scope = _scope_with_long_messages("private")
    input_set = ReflectionInputSet(
        lookback_hours=24,
        requested_start="2026-05-03T00:00:00+00:00",
        requested_end="2026-05-04T00:00:00+00:00",
        effective_start="2026-05-03T00:00:00+00:00",
        effective_end="2026-05-04T00:00:00+00:00",
        fallback_used=False,
        fallback_reason="",
        selected_scopes=[scope],
        query_diagnostics={},
    )
    return input_set


def _hourly_result(scope: ReflectionScopeInput) -> ReflectionLLMResult:
    """Build one parsed hourly result for daily prompt tests."""

    prompt = build_hourly_reflection_prompt(scope)
    result = ReflectionLLMResult(
        scope_ref=scope.scope_ref,
        prompt=prompt,
        raw_output="{}",
        parsed_output={
            "topic_summary": "A project planning conversation.",
            "participant_observations": [
                {
                    "participant_ref": "participant_1",
                    "observation": "Asked for structured help.",
                    "evidence_strength": "high",
                }
            ],
            "conversation_quality_feedback": [
                "The character was useful when it stayed concrete."
            ],
            "privacy_notes": ["No persistence-safe details yet."],
            "confidence": "medium",
        },
        validation_warnings=[],
    )
    return result


def _scope_with_long_messages(channel_type: str) -> ReflectionScopeInput:
    """Build a scope with enough text to exercise prompt trimming."""

    long_text = " ".join(["detail"] * 400)
    messages = [
        {
            "role": "user",
            "platform_user_id": "platform-user-1",
            "global_user_id": "global-user-1",
            "display_name": "Alice Example",
            "body_text": f"Can you help me with the budget migration? {long_text}",
            "attachments": [],
            "timestamp": "2026-05-03T22:00:00+00:00",
        },
        {
            "role": "assistant",
            "platform_user_id": "platform-bot",
            "global_user_id": "character-global",
            "display_name": "Character",
            "body_text": f"I can split the work into safe review steps. {long_text}",
            "attachments": [],
            "timestamp": "2026-05-03T22:01:00+00:00",
        },
    ]
    scope = ReflectionScopeInput(
        scope_ref=f"scope_{channel_type}",
        platform="qq",
        platform_channel_id="private-channel-id",
        channel_type=channel_type,
        assistant_message_count=1,
        user_message_count=1,
        total_message_count=2,
        first_timestamp="2026-05-03T22:00:00+00:00",
        last_timestamp="2026-05-03T22:01:00+00:00",
        messages=messages,
    )
    return scope


def _scope_with_many_long_messages() -> ReflectionScopeInput:
    """Build a scope with enough rows to exercise prompt row dropping."""

    long_text = " ".join(["detail"] * 400)
    messages = []
    for index in range(20):
        role = "assistant" if index % 2 else "user"
        message = {
            "role": role,
            "platform_user_id": f"platform-{role}-{index}",
            "global_user_id": f"global-{role}-{index}",
            "display_name": f"Speaker {index}",
            "body_text": f"message-{index} {long_text}",
            "attachments": [],
            "timestamp": f"2026-05-03T22:{index:02d}:00+00:00",
        }
        messages.append(message)
    scope = ReflectionScopeInput(
        scope_ref="scope_many_messages",
        platform="qq",
        platform_channel_id="private-channel-id",
        channel_type="private",
        assistant_message_count=10,
        user_message_count=10,
        total_message_count=20,
        first_timestamp="2026-05-03T22:00:00+00:00",
        last_timestamp="2026-05-03T22:19:00+00:00",
        messages=messages,
    )
    return scope
