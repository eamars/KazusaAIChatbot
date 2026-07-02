"""Daily affect settling for persistent character mood and global vibe."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot import db
from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.config import (
    AFFECT_SETTLING_WAKE_PREP_MINUTES,
    CHARACTER_SLEEP_LOCAL_PERIOD,
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    CONSOLIDATION_LLM_MODEL,
    CONSOLIDATION_LLM_THINKING_ENABLED,
    REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME,
)
from kazusa_ai_chatbot.db.schemas import CharacterReflectionRunDoc
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.reflection_cycle.models import (
    AFFECT_SETTLING_PROMPT_VERSION,
    PromptBuildResult,
    REFLECTION_RUN_KIND_DAILY_AFFECT_SETTLING,
    REFLECTION_RUN_KIND_DAILY_CHANNEL,
    REFLECTION_RUN_KIND_HOURLY,
    REFLECTION_STATUS_DRY_RUN,
    REFLECTION_STATUS_SKIPPED,
    REFLECTION_STATUS_SUCCEEDED,
    ReflectionWorkerResult,
)
from kazusa_ai_chatbot.reflection_cycle.projection import build_prompt_result
from kazusa_ai_chatbot.reflection_cycle import repository
from kazusa_ai_chatbot.time_boundary import (
    local_time_context_from_storage_utc,
    normalize_storage_utc_iso,
    parse_storage_utc_datetime,
    storage_utc_now_iso,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output


LOCAL_CLOCK_TEXT_LENGTH = 5
LOCAL_CLOCK_SEPARATOR_INDEX = 2
LOCAL_CLOCK_HOUR_END_INDEX = 2
LOCAL_CLOCK_MINUTE_START_INDEX = 3
MAX_LOCAL_HOUR = 23
MAX_LOCAL_MINUTE = 59
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
MINUTES_PER_DAY = HOURS_PER_DAY * MINUTES_PER_HOUR
LOCAL_DATE_END_INDEX = 10
LOCAL_TIME_START_INDEX = 11
LOCAL_TIME_END_INDEX = 16
PREVIOUS_DAY_OFFSET_DAYS = 1

AFFECT_SETTLING_PROMPT_MAX_CHARS = 12000
AFFECT_SETTLING_REVIEW_PROMPT_MAX_CHARS = 8000
AFFECT_SETTLING_AFTER_PROMOTION_GRACE_MINUTES = 15
AFFECT_SETTLING_WAKE_DEFER_GRACE_MINUTES = 15
AFFECT_SETTLING_LLM_TEMPERATURE = 0.2
AFFECT_SETTLING_LLM_TOP_P = 0.8
AFFECT_SETTLING_REVIEW_LLM_TEMPERATURE = 0.0
AFFECT_SETTLING_REVIEW_LLM_TOP_P = 0.8
_AFFECT_REQUIRED_FIELDS = ("mood", "global_vibe", "reflection_summary")
_REVIEW_REQUIRED_FIELDS = ("write_decision", "review_reason")
logger = logging.getLogger(__name__)

AFFECT_SETTLING_SYSTEM_PROMPT = '''\
# 任务
你负责在角色睡眠/休息结束前，审阅当前 mood, global_vibe, reflection_summary，并输出一个更接近人类隔夜沉淀后的版本。

# 约束
- 只输出有效 JSON。
- JSON key 必须保持英文。
- mood, global_vibe, reflection_summary 是自由文本，不要把它们改成固定枚举。
- 你的目标是渐进衰减，不是强行开心、失忆、道歉或压制情绪。
- 如果昨天确实发生了严重冲突，新的状态仍可以生气、戒备或受伤，只是应体现睡眠后的距离感。
- 不要依据任何操作性元数据判断情绪。
- 只能根据 current_affect, daily_reflection_cards, sleep_window_reflection_cards 和 review_questions 输出。

# 输出格式
{
  "mood": "自由文本，角色当前即时情绪",
  "global_vibe": "自由文本，角色整体氛围",
  "reflection_summary": "自由文本，解释这次隔夜沉淀后的持续心理状态"
}
'''

AFFECT_SETTLING_REVIEW_SYSTEM_PROMPT = '''\
# 任务
你是 daily affect settling 的结构与角色可信度审阅器。

# 判断标准
- 只判断 proposed_affect 是否是对 current_affect 的渐进沉淀。
- 不要使用固定情绪词表或枚举。
- 不要替作者改写 mood/global_vibe/reflection_summary。
- 如果输出过度清空情绪、突然转好、忽略明确冲突、或结构缺字段，reject。
- 如果输出保留因果连续性、变化幅度可信、且字段为非空自由文本，accept。

# 输出格式
{
  "write_decision": "accept 或 reject",
  "review_reason": "简短说明"
}
'''

_affect_settling_llm = LLInterface()
_affect_settling_review_llm = LLInterface()
_affect_settling_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CONSOLIDATION_LLM",
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=AFFECT_SETTLING_LLM_TEMPERATURE,
    top_p=AFFECT_SETTLING_LLM_TOP_P,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)
_affect_settling_review_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CONSOLIDATION_LLM",
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=AFFECT_SETTLING_REVIEW_LLM_TEMPERATURE,
    top_p=AFFECT_SETTLING_REVIEW_LLM_TOP_P,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)


def compute_affect_settling_due_local_time(
    *,
    sleep_local_period: str,
    promotion_run_after_local_time: str,
    after_promotion_grace_minutes: int,
    wake_prep_minutes: int,
) -> str:
    """Return the local clock time when daily affect settling becomes due."""

    _sleep_start_minutes, sleep_end_minutes = _sleep_period_bounds(
        sleep_local_period,
    )
    promotion_minutes = _local_clock_minutes(promotion_run_after_local_time)
    due_minutes = max(
        promotion_minutes + after_promotion_grace_minutes,
        sleep_end_minutes - wake_prep_minutes,
    )
    due_time = _minutes_to_clock(due_minutes)
    return due_time


def validate_affect_settling_timing(
    *,
    sleep_local_period: str = CHARACTER_SLEEP_LOCAL_PERIOD,
    promotion_run_after_local_time: str = REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME,
    after_promotion_grace_minutes: int = (
        AFFECT_SETTLING_AFTER_PROMOTION_GRACE_MINUTES
    ),
    wake_prep_minutes: int = AFFECT_SETTLING_WAKE_PREP_MINUTES,
    wake_defer_grace_minutes: int = AFFECT_SETTLING_WAKE_DEFER_GRACE_MINUTES,
) -> None:
    """Fail fast when affect settling cannot run inside the wake window.

    Empty sleep period disables the affect-settling schedule through the
    shared sleep-period contract.
    """

    if not sleep_local_period:
        return

    _sleep_start_minutes, sleep_end_minutes = _sleep_period_bounds(
        sleep_local_period,
    )
    promotion_minutes = _local_clock_minutes(promotion_run_after_local_time)
    due_minutes = max(
        promotion_minutes + after_promotion_grace_minutes,
        sleep_end_minutes - wake_prep_minutes,
    )
    latest_minutes = sleep_end_minutes + wake_defer_grace_minutes
    if due_minutes > latest_minutes:
        raise ValueError(
            "AFFECT_SETTLING due time cannot be later than sleep end plus "
            "wake defer grace"
        )


def local_datetime_is_in_affect_settling_window(
    local_datetime: str,
    *,
    sleep_local_period: str = CHARACTER_SLEEP_LOCAL_PERIOD,
    promotion_run_after_local_time: str = REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME,
    after_promotion_grace_minutes: int = (
        AFFECT_SETTLING_AFTER_PROMOTION_GRACE_MINUTES
    ),
    wake_prep_minutes: int = AFFECT_SETTLING_WAKE_PREP_MINUTES,
    wake_defer_grace_minutes: int = AFFECT_SETTLING_WAKE_DEFER_GRACE_MINUTES,
) -> bool:
    """Return whether the local timestamp is inside the affect-settling window."""

    if not sleep_local_period:
        return False

    _sleep_start_minutes, sleep_end_minutes = _sleep_period_bounds(
        sleep_local_period,
    )
    promotion_minutes = _local_clock_minutes(promotion_run_after_local_time)
    due_minutes = max(
        promotion_minutes + after_promotion_grace_minutes,
        sleep_end_minutes - wake_prep_minutes,
    )
    latest_minutes = sleep_end_minutes + wake_defer_grace_minutes
    current_clock = local_datetime[LOCAL_TIME_START_INDEX:LOCAL_TIME_END_INDEX]
    current_minutes = _local_clock_minutes(current_clock)
    if latest_minutes >= MINUTES_PER_DAY and current_minutes < due_minutes:
        current_minutes += MINUTES_PER_DAY
    return_value = due_minutes <= current_minutes <= latest_minutes
    return return_value


def settling_local_date_for_due_affect_settling(
    local_datetime: str,
    *,
    sleep_local_period: str = CHARACTER_SLEEP_LOCAL_PERIOD,
    promotion_run_after_local_time: str = REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME,
    after_promotion_grace_minutes: int = (
        AFFECT_SETTLING_AFTER_PROMOTION_GRACE_MINUTES
    ),
    wake_prep_minutes: int = AFFECT_SETTLING_WAKE_PREP_MINUTES,
) -> str:
    """Return the sleep-ending date due for worker execution, if any."""

    if not sleep_local_period:
        return_value = ""
        return return_value

    _sleep_start_minutes, sleep_end_minutes = _sleep_period_bounds(
        sleep_local_period,
    )
    promotion_minutes = _local_clock_minutes(promotion_run_after_local_time)
    due_minutes = max(
        promotion_minutes + after_promotion_grace_minutes,
        sleep_end_minutes - wake_prep_minutes,
    )
    current_clock = local_datetime[LOCAL_TIME_START_INDEX:LOCAL_TIME_END_INDEX]
    current_minutes = _local_clock_minutes(current_clock)
    due_clock_minutes = due_minutes % MINUTES_PER_DAY
    if due_minutes >= MINUTES_PER_DAY and current_minutes < due_clock_minutes:
        current_minutes += MINUTES_PER_DAY
    if current_minutes < due_minutes:
        return_value = ""
        return return_value
    return_value = local_datetime[:LOCAL_DATE_END_INDEX]
    return return_value


async def should_pause_self_cognition_for_affect_settling(
    *,
    now: datetime,
) -> bool:
    """Return whether self-cognition should pause for pending affect settling."""

    now_utc = normalize_storage_utc_iso(now.isoformat())
    local_time_context = local_time_context_from_storage_utc(now_utc)
    local_datetime = local_time_context["current_local_datetime"]
    if not local_datetime_is_in_affect_settling_window(local_datetime):
        return_value = False
        return return_value

    settling_local_date = local_datetime[:LOCAL_DATE_END_INDEX]
    run_id = repository.daily_affect_settling_run_id(
        settling_local_date=settling_local_date,
    )
    existing = await repository.reflection_run_by_id(run_id)
    return_value = not _affect_settling_doc_blocks_retry(existing)
    return return_value


def build_affect_settling_payload(
    *,
    settling_local_date: str,
    character_state: dict[str, Any],
    daily_docs: list[CharacterReflectionRunDoc],
    sleep_window_docs: list[CharacterReflectionRunDoc],
) -> dict[str, Any]:
    """Build the prompt payload without operational ids or freshness tokens."""

    payload = {
        "evaluation_mode": "daily_affect_settling",
        "settling_local_date": settling_local_date,
        "current_affect": {
            "mood": str(character_state.get("mood", "") or ""),
            "global_vibe": str(character_state.get("global_vibe", "") or ""),
            "reflection_summary": str(
                character_state.get("reflection_summary", "") or ""
            ),
        },
        "source_counts": {
            "daily_reflection_cards": len(daily_docs),
            "sleep_window_reflection_cards": len(sleep_window_docs),
        },
        "daily_reflection_cards": [
            _daily_doc_card(document)
            for document in daily_docs
        ],
        "sleep_window_reflection_cards": [
            _hourly_doc_card(document)
            for document in sleep_window_docs
        ],
        "review_questions": [
            "What emotional residue should plausibly survive sleep?",
            "What sharpness should decay because it is no longer immediate?",
            "What should remain unchanged because evidence still supports it?",
            "Does the new affect read like gradual rest, not a sudden reset?",
        ],
    }
    return payload


def build_affect_settling_prompt(
    payload: dict[str, Any],
) -> PromptBuildResult:
    """Serialize the affect-settling proposal prompt."""

    prompt = build_prompt_result(
        system_prompt=AFFECT_SETTLING_SYSTEM_PROMPT,
        human_payload=payload,
        max_prompt_chars=AFFECT_SETTLING_PROMPT_MAX_CHARS,
    )
    return prompt


async def run_affect_settling_proposal_llm(
    *,
    prompt: PromptBuildResult,
) -> dict[str, Any]:
    """Run the consolidation LLM for one affect-settling proposal."""

    response = await _affect_settling_llm.ainvoke([
        SystemMessage(content=prompt.system_prompt),
        HumanMessage(content=prompt.human_prompt),
    ], config=_affect_settling_llm_config)
    raw_output = str(response.content)
    parsed_output = parse_llm_json_output(raw_output)
    return parsed_output


async def run_affect_settling_review_llm(
    *,
    prompt: PromptBuildResult,
) -> dict[str, Any]:
    """Run the consolidation LLM for affect-settling structural review."""

    response = await _affect_settling_review_llm.ainvoke([
        SystemMessage(content=prompt.system_prompt),
        HumanMessage(content=prompt.human_prompt),
    ], config=_affect_settling_review_llm_config)
    raw_output = str(response.content)
    parsed_output = parse_llm_json_output(raw_output)
    return parsed_output


async def load_affect_settling_source_documents(
    *,
    settling_local_date: str,
) -> tuple[list[CharacterReflectionRunDoc], list[CharacterReflectionRunDoc]]:
    """Load previous-day daily docs and current sleep-window hourly docs."""

    local_date = date.fromisoformat(settling_local_date)
    previous_local_date = (
        local_date - timedelta(days=PREVIOUS_DAY_OFFSET_DAYS)
    ).isoformat()
    daily_docs = await repository.reflection_runs_for_kind_date(
        run_kind=REFLECTION_RUN_KIND_DAILY_CHANNEL,
        character_local_date=previous_local_date,
    )
    hourly_docs: list[CharacterReflectionRunDoc] = []
    hourly_docs.extend(
        await repository.reflection_runs_for_kind_date(
            run_kind=REFLECTION_RUN_KIND_HOURLY,
            character_local_date=settling_local_date,
        )
    )
    if _sleep_period_crosses_midnight(CHARACTER_SLEEP_LOCAL_PERIOD):
        hourly_docs.extend(
            await repository.reflection_runs_for_kind_date(
                run_kind=REFLECTION_RUN_KIND_HOURLY,
                character_local_date=previous_local_date,
            )
        )

    usable_daily_docs = [
        document
        for document in daily_docs
        if _doc_has_succeeded_output(document)
    ]
    sleep_window_docs = [
        document
        for document in hourly_docs
        if _doc_has_succeeded_output(document)
        and _hourly_doc_in_sleep_window(
            document,
            settling_local_date=settling_local_date,
        )
    ]
    return_value = (usable_daily_docs, sleep_window_docs)
    return return_value


async def run_daily_affect_settling(
    *,
    settling_local_date: str,
    dry_run: bool,
    enable_character_state_write: bool,
    character_state_refresh_callback: Callable[[], Any] | None = None,
) -> ReflectionWorkerResult:
    """Run one persistent daily affect-settling pass."""

    result = ReflectionWorkerResult(
        run_kind=REFLECTION_RUN_KIND_DAILY_AFFECT_SETTLING,
        dry_run=dry_run,
    )
    run_id = repository.daily_affect_settling_run_id(
        settling_local_date=settling_local_date,
    )
    result.run_ids.append(run_id)

    existing = await repository.reflection_run_by_id(run_id)
    if _affect_settling_doc_blocks_retry(existing):
        result.skipped_count = 1
        result.defer_reason = "daily affect settling already terminal"
        return result

    result.processed_count = 1
    character_state = await db.get_character_runtime_state()
    state_warning = _character_state_warning(character_state)
    if state_warning:
        await _persist_affect_settling_run(
            settling_local_date=settling_local_date,
            status=REFLECTION_STATUS_SKIPPED,
            source_run_ids=[],
            output={
                "skip_reason": state_warning,
                "retryable": False,
            },
            validation_warnings=[state_warning],
        )
        result.skipped_count = 1
        result.validation_warnings.append(state_warning)
        return result

    daily_docs, sleep_window_docs = await load_affect_settling_source_documents(
        settling_local_date=settling_local_date,
    )
    payload = build_affect_settling_payload(
        settling_local_date=settling_local_date,
        character_state=character_state,
        daily_docs=daily_docs,
        sleep_window_docs=sleep_window_docs,
    )
    prompt = build_affect_settling_prompt(payload)
    proposal_raw = await run_affect_settling_proposal_llm(prompt=prompt)
    proposal, proposal_warnings = _validate_affect_proposal(proposal_raw)
    validation_warnings = list(prompt.validation_warnings)
    validation_warnings.extend(proposal_warnings)
    source_run_ids = _source_run_ids(daily_docs, sleep_window_docs)
    if proposal_warnings:
        await _persist_affect_settling_run(
            settling_local_date=settling_local_date,
            status=REFLECTION_STATUS_SKIPPED,
            source_run_ids=source_run_ids,
            output={
                "skip_reason": "proposal_structurally_invalid",
                "retryable": False,
                "proposal": proposal_raw,
            },
            validation_warnings=validation_warnings,
        )
        result.skipped_count = 1
        result.validation_warnings.extend(validation_warnings)
        return result

    review_prompt = _build_affect_settling_review_prompt(
        payload=payload,
        proposal=proposal,
    )
    review_raw = await run_affect_settling_review_llm(prompt=review_prompt)
    review, review_warnings = _validate_review(review_raw)
    validation_warnings.extend(review_prompt.validation_warnings)
    validation_warnings.extend(review_warnings)
    if review_warnings:
        await _persist_affect_settling_run(
            settling_local_date=settling_local_date,
            status=REFLECTION_STATUS_SKIPPED,
            source_run_ids=source_run_ids,
            output={
                "skip_reason": "review_structurally_invalid",
                "retryable": False,
                "proposal": proposal,
                "review": review_raw,
            },
            validation_warnings=validation_warnings,
        )
        result.skipped_count = 1
        result.validation_warnings.extend(validation_warnings)
        return result

    if review["write_decision"] != "accept":
        await _persist_affect_settling_run(
            settling_local_date=settling_local_date,
            status=REFLECTION_STATUS_SKIPPED,
            source_run_ids=source_run_ids,
            output={
                "skip_reason": "reviewer_rejected",
                "retryable": False,
                "proposal": proposal,
                "review": review,
            },
            validation_warnings=validation_warnings,
        )
        result.skipped_count = 1
        result.validation_warnings.extend(validation_warnings)
        return result

    if dry_run:
        await _persist_affect_settling_run(
            settling_local_date=settling_local_date,
            status=REFLECTION_STATUS_DRY_RUN,
            source_run_ids=source_run_ids,
            output={
                "skip_reason": "dry_run",
                "retryable": True,
                "proposal": proposal,
                "review": review,
            },
            validation_warnings=validation_warnings,
        )
        result.skipped_count = 1
        result.validation_warnings.extend(validation_warnings)
        return result

    if not enable_character_state_write:
        await _persist_affect_settling_run(
            settling_local_date=settling_local_date,
            status=REFLECTION_STATUS_SKIPPED,
            source_run_ids=source_run_ids,
            output={
                "skip_reason": "character_state_write_disabled",
                "retryable": False,
                "proposal": proposal,
                "review": review,
            },
            validation_warnings=validation_warnings,
        )
        result.skipped_count = 1
        result.validation_warnings.extend(validation_warnings)
        return result

    expected_updated_at = str(character_state.get("updated_at") or "")
    write_succeeded = await db.compare_and_upsert_character_state(
        expected_updated_at=expected_updated_at,
        mood=proposal["mood"],
        global_vibe=proposal["global_vibe"],
        reflection_summary=proposal["reflection_summary"],
        updated_at_utc=storage_utc_now_iso(),
    )
    if not write_succeeded:
        await _persist_affect_settling_run(
            settling_local_date=settling_local_date,
            status=REFLECTION_STATUS_SKIPPED,
            source_run_ids=source_run_ids,
            output={
                "skip_reason": "stale_character_state",
                "retryable": False,
                "proposal": proposal,
                "review": review,
            },
            validation_warnings=validation_warnings,
        )
        result.skipped_count = 1
        result.validation_warnings.extend(validation_warnings)
        return result

    await _persist_affect_settling_run(
        settling_local_date=settling_local_date,
        status=REFLECTION_STATUS_SUCCEEDED,
        source_run_ids=source_run_ids,
        output={
            "mood": proposal["mood"],
            "global_vibe": proposal["global_vibe"],
            "reflection_summary": proposal["reflection_summary"],
            "proposal": proposal,
            "review": review,
        },
        validation_warnings=validation_warnings,
    )
    try:
        await _call_refresh_callback(character_state_refresh_callback)
    except Exception as exc:
        logger.exception(
            f"Affect-settling runtime state refresh failed: {exc}"
        )
        await event_logging.record_runtime_error_event(
            component="reflection_cycle.affect_settling",
            error_class=type(exc).__name__,
            error_preview=str(exc),
            stack_fingerprint="affect_settling_runtime_state_refresh",
            top_frame_module=__name__,
            recovered=True,
            run_id=run_id,
        )
    result.succeeded_count = 1
    result.validation_warnings.extend(validation_warnings)
    return result


def _build_affect_settling_review_prompt(
    *,
    payload: dict[str, Any],
    proposal: dict[str, str],
) -> PromptBuildResult:
    """Build the LLM reviewer prompt."""

    review_payload = {
        "evaluation_mode": "daily_affect_settling_review",
        "current_affect": payload["current_affect"],
        "settling_local_date": payload["settling_local_date"],
        "source_counts": payload["source_counts"],
        "daily_reflection_cards": payload["daily_reflection_cards"],
        "sleep_window_reflection_cards": payload["sleep_window_reflection_cards"],
        "proposed_affect": proposal,
        "review_questions": [
            "Is the proposed affect a gradual change from current_affect?",
            "Does it preserve justified anger or hurt when evidence supports it?",
            "Does it avoid fixed mood labels and deterministic vocabulary?",
            "Should this exact proposal be written?",
        ],
    }
    prompt = build_prompt_result(
        system_prompt=AFFECT_SETTLING_REVIEW_SYSTEM_PROMPT,
        human_payload=review_payload,
        max_prompt_chars=AFFECT_SETTLING_REVIEW_PROMPT_MAX_CHARS,
    )
    return prompt


async def _persist_affect_settling_run(
    *,
    settling_local_date: str,
    status: str,
    source_run_ids: list[str],
    output: dict[str, Any],
    validation_warnings: list[str],
    error: str = "",
) -> None:
    """Persist one affect-settling audit row."""

    document = repository.build_daily_affect_settling_run_document(
        settling_local_date=settling_local_date,
        prompt_version=AFFECT_SETTLING_PROMPT_VERSION,
        source_run_ids=source_run_ids,
        output=output,
        status=status,
        attempt_count=1,
        validation_warnings=validation_warnings,
        error=error,
    )
    await repository.upsert_run(document)


async def _call_refresh_callback(
    callback: Callable[[], Any] | None,
) -> None:
    """Call an optional sync or async runtime refresh callback."""

    if callback is None:
        return
    value = callback()
    if inspect.isawaitable(value):
        await value


def _validate_affect_proposal(
    proposal: Any,
) -> tuple[dict[str, str], list[str]]:
    """Validate required proposal fields without interpreting their content."""

    if not isinstance(proposal, dict):
        return ({}, ["proposal must be a JSON object"])
    output: dict[str, str] = {}
    warnings: list[str] = []
    for field_name in _AFFECT_REQUIRED_FIELDS:
        value = proposal.get(field_name)
        if not isinstance(value, str) or not value.strip():
            warnings.append(f"proposal.{field_name} must be a non-empty string")
            continue
        output[field_name] = value.strip()
    return_value = (output, warnings)
    return return_value


def _validate_review(review: Any) -> tuple[dict[str, str], list[str]]:
    """Validate the reviewer decision envelope."""

    if not isinstance(review, dict):
        return ({}, ["review must be a JSON object"])
    warnings: list[str] = []
    output: dict[str, str] = {}
    for field_name in _REVIEW_REQUIRED_FIELDS:
        value = review.get(field_name)
        if not isinstance(value, str) or not value.strip():
            warnings.append(f"review.{field_name} must be a non-empty string")
            continue
        output[field_name] = value.strip()
    if output.get("write_decision") not in {"accept", "reject"}:
        warnings.append("review.write_decision must be accept or reject")
    return_value = (output, warnings)
    return return_value


def _character_state_warning(character_state: dict[str, Any]) -> str:
    """Return a structural missing-state warning or an empty string."""

    if not character_state:
        return "missing_character_state"
    if not str(character_state.get("updated_at") or "").strip():
        return "missing_character_state_updated_at"
    for field_name in _AFFECT_REQUIRED_FIELDS:
        value = character_state.get(field_name)
        if not isinstance(value, str) or not value.strip():
            return f"missing_character_state_{field_name}"
    return_value = ""
    return return_value


def _affect_settling_doc_blocks_retry(
    document: CharacterReflectionRunDoc | None,
) -> bool:
    """Return whether an existing affect run should block another attempt."""

    if document is None:
        return False
    status = str(document.get("status", "") or "")
    if status == REFLECTION_STATUS_SUCCEEDED:
        return True
    if status != REFLECTION_STATUS_SKIPPED:
        return False
    output = document.get("output")
    if not isinstance(output, dict):
        return False
    return_value = output.get("retryable") is False
    return return_value


def _source_run_ids(
    daily_docs: list[CharacterReflectionRunDoc],
    sleep_window_docs: list[CharacterReflectionRunDoc],
) -> list[str]:
    """Return unique source run ids in prompt evidence order."""

    seen: set[str] = set()
    source_run_ids: list[str] = []
    for document in [*daily_docs, *sleep_window_docs]:
        run_id = str(document.get("run_id", "") or "").strip()
        if not run_id or run_id in seen:
            continue
        source_run_ids.append(run_id)
        seen.add(run_id)
    return source_run_ids


def _daily_doc_card(document: CharacterReflectionRunDoc) -> dict[str, Any]:
    """Project a daily reflection row into a prompt-facing card."""

    output = document.get("output")
    if not isinstance(output, dict):
        output = {}
    scope = document.get("scope")
    if not isinstance(scope, dict):
        scope = {}
    card = {
        "channel_type": str(scope.get("channel_type", "") or ""),
        "day_summary": str(output.get("day_summary", "") or ""),
        "cross_hour_topics": _string_list(output.get("cross_hour_topics")),
        "conversation_quality_patterns": _string_list(
            output.get("conversation_quality_patterns"),
        ),
        "privacy_risks": _string_list(output.get("privacy_risks")),
        "synthesis_limitations": _string_list(
            output.get("synthesis_limitations"),
        ),
        "confidence": str(output.get("confidence", "") or ""),
    }
    return card


def _hourly_doc_card(document: CharacterReflectionRunDoc) -> dict[str, Any]:
    """Project an hourly reflection row into a prompt-facing card."""

    output = document.get("output")
    if not isinstance(output, dict):
        output = {}
    scope = document.get("scope")
    if not isinstance(scope, dict):
        scope = {}
    card = {
        "channel_type": str(scope.get("channel_type", "") or ""),
        "topic_summary": str(output.get("topic_summary", "") or ""),
        "conversation_quality_feedback": _string_list(
            output.get("conversation_quality_feedback"),
        ),
        "privacy_notes": _string_list(output.get("privacy_notes")),
        "active_character_utterances": _string_list(
            output.get("active_character_utterances"),
        ),
        "confidence": str(output.get("confidence", "") or ""),
    }
    return card


def _string_list(value: Any) -> list[str]:
    """Return string items from a prompt-facing list value."""

    if not isinstance(value, list):
        return []
    return_value = [
        str(item)
        for item in value
        if str(item).strip()
    ]
    return return_value


def _doc_has_succeeded_output(document: CharacterReflectionRunDoc) -> bool:
    """Return whether a reflection run has succeeded prompt evidence."""

    output = document.get("output")
    return_value = (
        document.get("status") == REFLECTION_STATUS_SUCCEEDED
        and isinstance(output, dict)
        and bool(output)
    )
    return return_value


def _hourly_doc_in_sleep_window(
    document: CharacterReflectionRunDoc,
    *,
    settling_local_date: str,
) -> bool:
    """Return whether one hourly doc falls inside the configured sleep window."""

    if not CHARACTER_SLEEP_LOCAL_PERIOD:
        return False
    hour_start = str(document.get("hour_start", "") or "")
    if not hour_start:
        return False
    timestamp = parse_storage_utc_datetime(hour_start)
    local_time_context = local_time_context_from_storage_utc(
        normalize_storage_utc_iso(timestamp.isoformat()),
    )
    local_datetime = local_time_context["current_local_datetime"]
    local_date = local_datetime[:LOCAL_DATE_END_INDEX]
    local_clock = local_datetime[LOCAL_TIME_START_INDEX:LOCAL_TIME_END_INDEX]
    local_minutes = _local_clock_minutes(local_clock)
    sleep_start_minutes, sleep_end_minutes = _sleep_period_bounds(
        CHARACTER_SLEEP_LOCAL_PERIOD,
    )
    if sleep_start_minutes < sleep_end_minutes:
        return_value = (
            local_date == settling_local_date
            and sleep_start_minutes <= local_minutes < sleep_end_minutes
        )
        return return_value

    settling_date = date.fromisoformat(settling_local_date)
    previous_date = (
        settling_date - timedelta(days=PREVIOUS_DAY_OFFSET_DAYS)
    ).isoformat()
    return_value = (
        local_date == previous_date
        and local_minutes >= sleep_start_minutes
    ) or (
        local_date == settling_local_date
        and local_minutes < sleep_end_minutes
    )
    return return_value


def _sleep_period_crosses_midnight(sleep_local_period: str) -> bool:
    """Return whether the configured sleep period spans local midnight."""

    if not sleep_local_period:
        return False
    start_minutes, end_minutes = _sleep_period_bounds(sleep_local_period)
    return_value = start_minutes > end_minutes
    return return_value


def _sleep_period_bounds(sleep_local_period: str) -> tuple[int, int]:
    """Parse exact sleep period bounds into local minutes."""

    parts = sleep_local_period.split("-", maxsplit=1)
    if len(parts) != 2:
        raise ValueError("sleep_local_period must use HH:MM-HH:MM")
    start_minutes = _local_clock_minutes(parts[0])
    end_minutes = _local_clock_minutes(parts[1])
    return_value = (start_minutes, end_minutes)
    return return_value


def _local_clock_minutes(value: str) -> int:
    """Parse exact ``HH:MM`` text into minutes after local midnight."""

    if (
        len(value) != LOCAL_CLOCK_TEXT_LENGTH
        or value[LOCAL_CLOCK_SEPARATOR_INDEX] != ":"
    ):
        raise ValueError("local clock value must use HH:MM")
    hour_text = value[:LOCAL_CLOCK_HOUR_END_INDEX]
    minute_text = value[LOCAL_CLOCK_MINUTE_START_INDEX:]
    if not hour_text.isdecimal() or not minute_text.isdecimal():
        raise ValueError("local clock value must use HH:MM")
    hour = int(hour_text)
    minute = int(minute_text)
    if hour > MAX_LOCAL_HOUR or minute > MAX_LOCAL_MINUTE:
        raise ValueError("local clock value must use HH:MM")
    return_value = (hour * MINUTES_PER_HOUR) + minute
    return return_value


def _minutes_to_clock(minutes: int) -> str:
    """Return ``HH:MM`` text for local minutes, preserving local-day clock."""

    clock_minutes = minutes % MINUTES_PER_DAY
    hour = clock_minutes // MINUTES_PER_HOUR
    minute = clock_minutes % MINUTES_PER_HOUR
    return_value = f"{hour:02d}:{minute:02d}"
    return return_value


validate_affect_settling_timing()
