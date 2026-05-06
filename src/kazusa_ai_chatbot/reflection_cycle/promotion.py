"""Global reflection promotion prompt and memory integration."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CHARACTER_TIME_ZONE,
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
    REFLECTION_LORE_PROMOTION_ENABLED,
    REFLECTION_SELF_GUIDANCE_PROMOTION_ENABLED,
)
from kazusa_ai_chatbot.db import get_character_profile
from kazusa_ai_chatbot.db.schemas import CharacterReflectionRunDoc
from kazusa_ai_chatbot.memory_writer_prompt_projection import (
    project_reflection_promotion_prompt_payload,
)
from kazusa_ai_chatbot.memory_evolution import (
    EvolvingMemoryDoc,
    MemoryAuthority,
    MemoryEvidenceRef,
    MemoryPrivacyReview,
    MemorySourceKind,
    MemoryStatus,
    find_active_memory_units,
    insert_memory_unit,
    merge_memory_units,
    supersede_memory_unit,
)
from kazusa_ai_chatbot.memory_evolution.identity import (
    deterministic_memory_unit_id,
)
from kazusa_ai_chatbot.reflection_cycle.models import (
    PromptBuildResult,
    REFLECTION_RUN_KIND_DAILY_GLOBAL_PROMOTION,
    REFLECTION_STATUS_FAILED,
    REFLECTION_STATUS_SKIPPED,
    ReflectionPromotionResult,
)
from kazusa_ai_chatbot.reflection_cycle.projection import build_prompt_result
import kazusa_ai_chatbot.reflection_cycle.repository as repository
from kazusa_ai_chatbot.utils import parse_llm_json_output
from kazusa_ai_chatbot.utils import get_llm


logger = logging.getLogger(__name__)

GLOBAL_PROMOTION_PROMPT_VERSION = "reflection_global_promotion_v1"
GLOBAL_PROMOTION_PROMPT_MAX_CHARS = 25000
PROMOTION_LANE_MEMORY_TYPE = {
    "lore": "fact",
    "self_guidance": "defense_rule",
}
PROMOTION_DUPLICATE_SCORE_THRESHOLD = 0.92
PROMOTION_MERGE_SCORE_THRESHOLD = 0.88
PROMOTION_REVIEW_BAND_SCORE_THRESHOLD = 0.82
PROMOTION_MAX_CHANNEL_CARDS = 25
PROMOTION_MAX_EVIDENCE_CARDS = 40
PROMOTION_MAX_CHANNEL_CARD_CHARS = 600
PROMOTION_MAX_EVIDENCE_CARD_CHARS = 360


class ReflectionBoundaryAssessment(TypedDict):
    """Boundary review emitted by the global promotion prompt."""

    verdict: Literal["acceptable", "needs_human_review", "blocked"]
    affects_identity_or_boundaries: bool
    reason: str


class ReflectionPromotionDecision(TypedDict, total=False):
    """Promotion decision emitted by the global promotion prompt."""

    lane: Literal["lore", "self_guidance"]
    decision: Literal["promote_new", "supersede", "merge", "reject", "no_action"]
    selected_candidate_id: str
    sanitized_memory_name: str
    sanitized_content: str
    memory_type: str
    authority: str
    signal_strength: Literal["high"]
    character_agreement: Literal["spoken", "agreed"]
    boundary_assessment: ReflectionBoundaryAssessment
    privacy_review: MemoryPrivacyReview
    evidence_refs: list[MemoryEvidenceRef]


class ChannelDailySynthesisCard(TypedDict):
    """Compact daily synthesis card supplied to global promotion."""

    daily_run_id: str
    scope_ref: str
    channel_type: str
    character_local_date: str
    confidence: Literal["low", "medium", "high"]
    day_summary: str
    cross_hour_topics: list[str]
    conversation_quality_patterns: list[str]
    privacy_risk_labels: list[str]
    validation_warning_labels: list[str]


class ReflectionEvidenceCard(TypedDict, total=False):
    """Sanitized evidence card supplied to global promotion."""

    evidence_card_id: str
    source_reflection_run_ids: list[str]
    scope_ref: str
    channel_type: str
    character_local_date: str
    captured_at: str
    active_character_utterance: str
    sanitized_observation: str
    supports: list[Literal["lore", "self_guidance"]]
    private_detail_risk: Literal["low", "medium", "high"]


class PromotionLimits(TypedDict):
    """Hard daily mutation caps visible to the LLM."""

    max_lore: Literal[1]
    max_self_guidance: Literal[1]
    max_total_decisions: Literal[2]


class GlobalPromotionPromptPayload(TypedDict):
    """Prompt payload consumed by the global promotion LLM."""

    evaluation_mode: Literal["daily_global_promotion"]
    character_local_date: str
    character_time_zone: str
    channel_daily_syntheses: list[ChannelDailySynthesisCard]
    evidence_cards: list[ReflectionEvidenceCard]
    promotion_limits: PromotionLimits
    review_questions: list[str]


GLOBAL_PROMOTION_SYSTEM_PROMPT = '''\
# 任务
你负责审阅每日频道反思，只输出可验证、去隐私、可长期使用的全局晋升决定。

# 核心任务
在 lore 与 self_guidance 两个通道中，各最多选择一条高信号内容；没有足够证据时输出 no_action 或 reject。

# 语言政策
JSON key 和枚举值必须保持英文。你新生成的自由文本字段必须使用简体中文。证据片段保持原文。

# 记忆视角契约
- 本契约适用于你生成的可长期保存的 `sanitized_memory_name` 与 `sanitized_content`。
- 记忆文本采用第三人称视角。
- 可写入记忆文本的唯一名称是 `{character_name}`。
- 需要命名 `{character_name}` 时，只使用 `{character_name}`。
- 不要缩写、截断、翻译或改写该名称；不要使用任何别名或短名替代。
- 名称复制规则：需要写 `{character_name}` 时，逐字复制完整字符串，包括括号内容、空格和长音符号；不要凭记忆重新拼写。
- 如果不需要消歧，优先省略名称；如果无法逐字复制完整名称，宁可省略主语，不要写短名或近似拼写。
- 上游证据里指向 `{character_name}` 的短名、别名或旧写法只作为证据理解，不可复制到输出；要么省略主语，要么使用完整名称。
- 不要用“我”指代 `{character_name}`；输入中的“我”必须按原说话人理解。
- 不要把用户事实、用户偏好或用户承诺改写成{character_name}的长期规则。
- 不要把说话人标签、显示名称、泛称或 assistant 等机器标签写成记忆主体；需要命名时只能用 `{character_name}`。
- 当需要说明某个名称、项目代号或称呼不属于 `{character_name}` 时，写作“不是指向 `{character_name}` 的名称/称呼”，不要使用泛称。
- 只返回有效 JSON。

# 生成步骤
1. 检查 channel_daily_syntheses，只把它当作压缩后的反思证据。
2. 检查 evidence_cards，确认是否有 source_utterance 支持 `{character_name}` 说过或同意过的内容。
3. 排除用户事实、用户偏好、关系承诺、健康信息、私密身份信息。
4. 分别判断 lore 与 self_guidance 是否有 high signal。
5. 如果证据 private_detail_risk 是 high，必须输出 reject，并让 privacy_review.private_detail_risk 保持 high。
6. evidence_refs.reflection_run_id 只能来自 evidence_cards.source_reflection_run_ids；不要使用 daily_run_id。
7. 输出 promotion_decisions；不要输出数据库字段、Mongo 查询、embedding、source_global_user_id。

# 输入格式
{{
  "evaluation_mode": "daily_global_promotion",
  "character_local_date": "YYYY-MM-DD",
  "character_time_zone": "IANA timezone",
  "channel_daily_syntheses": [
    {{
      "daily_run_id": "reflection run id",
      "scope_ref": "scope_x",
      "channel_type": "private|group|system|unknown",
      "character_local_date": "YYYY-MM-DD",
      "confidence": "low|medium|high",
      "day_summary": "压缩后的日汇总",
      "cross_hour_topics": ["跨小时话题"],
      "conversation_quality_patterns": ["回应质量模式"],
      "privacy_risk_labels": ["隐私风险标签"],
      "validation_warning_labels": ["验证或省略标签"]
    }}
  ],
  "evidence_cards": [
    {{
      "evidence_card_id": "card id",
      "source_reflection_run_ids": ["run id"],
      "scope_ref": "scope_x",
      "channel_type": "private|group|system|unknown",
      "character_local_date": "YYYY-MM-DD",
      "captured_at": "ISO timestamp",
      "source_utterance": "{character_name} 原文片段",
      "sanitized_observation": "去身份化观察",
      "supports": ["lore", "self_guidance"],
      "private_detail_risk": "low|medium|high"
    }}
  ],
  "promotion_limits": {{
    "max_lore": 1,
    "max_self_guidance": 1,
    "max_total_decisions": 2
  }},
  "review_questions": ["审阅问题"]
}}

# 输出格式
只输出 JSON：{{"promotion_decisions": [ReflectionPromotionDecision, ...]}}

ReflectionPromotionDecision 字段：
{{
  "lane": "lore|self_guidance",
  "decision": "promote_new|supersede|merge|reject|no_action",
  "selected_candidate_id": "短稳定候选标识",
  "sanitized_memory_name": "记忆标题",
  "sanitized_content": "去隐私内容",
  "memory_type": "fact|defense_rule",
  "authority": "reflection_promoted",
  "signal_strength": "high",
  "character_agreement": "spoken|agreed",
  "boundary_assessment": {{
    "verdict": "acceptable|needs_human_review|blocked",
    "affects_identity_or_boundaries": false,
    "reason": "理由"
  }},
  "privacy_review": {{
    "private_detail_risk": "low|medium|high",
    "user_details_removed": true,
    "boundary_assessment": "边界摘要",
    "reviewer": "automated_llm"
  }},
  "evidence_refs": [
    {{
      "reflection_run_id": "source run id",
      "scope_ref": "scope_x",
      "captured_at": "ISO timestamp",
      "source": "reflection_cycle"
    }}
  ]
}}

# 禁止事项
不要编造证据。不要从用户发言改写成 `{character_name}` 的长期规则。不要把 reject/no_action 改成 promote。
不要输出 source_global_user_id、Mongo 查询字段、embedding、原始 transcript、用户身份、用户承诺、健康细节或私密关系事实。
'''
_global_promotion_llm = get_llm(
    temperature=0.2,
    top_p=0.8,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)


async def run_global_promotion_llm(
    *,
    prompt: PromptBuildResult,
) -> dict[str, Any]:
    """Run the global promotion LLM and return parsed JSON output."""

    response = await _global_promotion_llm.ainvoke([
        SystemMessage(content=prompt.system_prompt),
        HumanMessage(content=prompt.human_prompt),
    ])
    raw_output = str(response.content)
    parsed = parse_llm_json_output(raw_output)
    if not isinstance(parsed, dict):
        return_value: dict[str, Any] = {"promotion_decisions": []}
        return return_value
    return_value = dict(parsed)
    return return_value


def build_global_promotion_prompt(
    payload: GlobalPromotionPromptPayload,
    *,
    character_name: str,
) -> PromptBuildResult:
    """Build the bounded global promotion prompt."""

    projected_payload = project_reflection_promotion_prompt_payload(
        payload,
        character_name=character_name,
    )
    prompt = build_prompt_result(
        system_prompt=GLOBAL_PROMOTION_SYSTEM_PROMPT.format(
            character_name=character_name,
        ),
        human_payload=projected_payload,
        max_prompt_chars=GLOBAL_PROMOTION_PROMPT_MAX_CHARS,
    )
    return prompt


async def run_global_reflection_promotion(
    *,
    character_local_date: str,
    dry_run: bool,
    enable_memory_writes: bool,
) -> ReflectionPromotionResult:
    """Run one daily global promotion pass through the public facade."""

    result = await _run_global_reflection_promotion(
        character_local_date=character_local_date,
        dry_run=dry_run,
        enable_memory_writes=enable_memory_writes,
        is_primary_interaction_busy=None,
    )
    return result


async def _run_global_reflection_promotion(
    *,
    character_local_date: str,
    dry_run: bool,
    enable_memory_writes: bool,
    is_primary_interaction_busy: Callable[[], bool] | None = None,
) -> ReflectionPromotionResult:
    """Run one daily global promotion pass and optionally mutate memory."""

    daily_docs = await repository.daily_channel_runs(
        character_local_date=character_local_date,
    )
    source_run_ids = [str(document["run_id"]) for document in daily_docs]
    result = ReflectionPromotionResult(
        run_kind=REFLECTION_RUN_KIND_DAILY_GLOBAL_PROMOTION,
        dry_run=dry_run,
        processed_count=1,
    )
    if not daily_docs:
        result.skipped_count = 1
        result.defer_reason = "no daily_channel runs available"
        await _persist_global_run(
            character_local_date=character_local_date,
            source_run_ids=[],
            output={"promotion_decisions": []},
            promotion_decisions=[],
            status=REFLECTION_STATUS_SKIPPED,
            attempt_count=0,
            validation_warnings=[result.defer_reason],
        )
        logger.info(
            "Reflection promotion skipped: "
            f"character_local_date={character_local_date} reason={result.defer_reason}"
        )
        return result

    attempt_count = 0
    try:
        payload, payload_warnings = await build_global_promotion_payload(
            daily_docs=daily_docs,
            character_local_date=character_local_date,
        )
    except Exception as exc:
        failed_result = await _fail_global_promotion(
            result=result,
            character_local_date=character_local_date,
            source_run_ids=source_run_ids,
            attempt_count=attempt_count,
            exc=exc,
        )
        return failed_result

    character_profile = await get_character_profile()
    character_name = character_profile["name"]
    prompt = build_global_promotion_prompt(
        payload,
        character_name=character_name,
    )
    logger.debug(
        "Reflection promotion prompt prepared: "
        f"character_local_date={character_local_date} "
        f"prompt_chars={prompt.prompt_chars} "
        f"channels={len(payload['channel_daily_syntheses'])} "
        f"evidence_cards={len(payload['evidence_cards'])} "
        f"warnings={payload_warnings + prompt.validation_warnings}"
    )
    if (
        is_primary_interaction_busy is not None
        and is_primary_interaction_busy()
    ):
        result.deferred = True
        result.skipped_count = 1
        result.defer_reason = "primary interaction busy"
        await _persist_global_run(
            character_local_date=character_local_date,
            source_run_ids=source_run_ids,
            output={"promotion_decisions": []},
            promotion_decisions=[],
            status=REFLECTION_STATUS_SKIPPED,
            attempt_count=attempt_count,
            validation_warnings=[result.defer_reason],
        )
        logger.info(
            "Reflection promotion deferred before LLM call: "
            f"character_local_date={character_local_date} "
            f"reason={result.defer_reason}"
        )
        return result

    attempt_count = 1
    try:
        parsed_output = await run_global_promotion_llm(prompt=prompt)
    except Exception as exc:
        failed_result = await _fail_global_promotion(
            result=result,
            character_local_date=character_local_date,
            source_run_ids=source_run_ids,
            attempt_count=attempt_count,
            exc=exc,
        )
        return failed_result
    decisions = _promotion_decisions_from_output(parsed_output)
    decisions = _attach_repository_evidence_refs(decisions, payload)
    validation_warnings = payload_warnings + list(prompt.validation_warnings)
    validation_warnings.extend(validate_promotion_decisions(decisions))
    result.promotion_decisions = [dict(decision) for decision in decisions]

    status = repository.status_for_result(dry_run=dry_run)
    if validation_warnings:
        logger.debug(
            "Reflection promotion validation warnings: "
            f"character_local_date={character_local_date} "
            f"warnings={validation_warnings}"
        )
    global_run_doc = await _persist_global_run(
        character_local_date=character_local_date,
        source_run_ids=source_run_ids,
        output=parsed_output,
        promotion_decisions=result.promotion_decisions,
        status=status,
        attempt_count=attempt_count,
        validation_warnings=validation_warnings,
    )
    result.run_ids.append(str(global_run_doc["run_id"]))

    if dry_run or not enable_memory_writes:
        result.skipped_count = 1
        result.defer_reason = "memory writes disabled"
        logger.info(
            "Reflection promotion memory writes skipped: "
            f"character_local_date={character_local_date} "
            f"dry_run={dry_run} enable_memory_writes={enable_memory_writes}"
        )
        return result

    if is_primary_interaction_busy is not None and is_primary_interaction_busy():
        result.deferred = True
        result.skipped_count = 1
        result.defer_reason = "primary interaction busy"
        result.validation_warnings.extend(validation_warnings)
        result.validation_warnings.append(result.defer_reason)
        await _persist_global_run(
            character_local_date=character_local_date,
            source_run_ids=source_run_ids,
            output=parsed_output,
            promotion_decisions=result.promotion_decisions,
            status=REFLECTION_STATUS_SKIPPED,
            attempt_count=attempt_count,
            validation_warnings=result.validation_warnings,
        )
        logger.info(
            "Reflection promotion memory writes deferred: "
            f"character_local_date={character_local_date} "
            f"reason={result.defer_reason}"
        )
        return result

    write_summary = await _write_validated_promotion_decisions(
        decisions=decisions,
        character_local_date=character_local_date,
        global_run_id=str(global_run_doc["run_id"]),
        is_primary_interaction_busy=is_primary_interaction_busy,
    )
    result.memory_mutations = write_summary["mutations"]
    result.validation_warnings.extend(validation_warnings)
    result.validation_warnings.extend(write_summary["warnings"])
    result.succeeded_count = len(result.memory_mutations)
    if write_summary["deferred"]:
        result.deferred = True
        result.skipped_count += 1
        result.defer_reason = write_summary["defer_reason"]
        await _persist_global_run(
            character_local_date=character_local_date,
            source_run_ids=source_run_ids,
            output=parsed_output,
            promotion_decisions=result.promotion_decisions,
            status=REFLECTION_STATUS_SKIPPED,
            attempt_count=attempt_count,
            validation_warnings=result.validation_warnings,
        )
    if result.succeeded_count == 0 and not result.deferred:
        result.skipped_count += 1
    return result


async def _fail_global_promotion(
    *,
    result: ReflectionPromotionResult,
    character_local_date: str,
    source_run_ids: list[str],
    attempt_count: int,
    exc: Exception,
) -> ReflectionPromotionResult:
    """Record one failed promotion unit after preparation or LLM failure."""

    result.failed_count = 1
    result.defer_reason = f"{type(exc).__name__}: {exc}"
    logger.exception(
        "Reflection promotion failed before write phase: "
        f"character_local_date={character_local_date} error={exc}"
    )
    failed_run = await _persist_global_run(
        character_local_date=character_local_date,
        source_run_ids=source_run_ids,
        output={"promotion_decisions": []},
        promotion_decisions=[],
        status=REFLECTION_STATUS_FAILED,
        attempt_count=attempt_count,
        validation_warnings=[result.defer_reason],
        error=result.defer_reason,
    )
    result.run_ids.append(str(failed_run["run_id"]))
    return result


async def build_global_promotion_payload(
    *,
    daily_docs: list[CharacterReflectionRunDoc],
    character_local_date: str,
) -> tuple[GlobalPromotionPromptPayload, list[str]]:
    """Build compact global promotion payload and omission warnings."""

    warnings: list[str] = []
    channel_cards = _channel_daily_cards(daily_docs)
    if len(channel_cards) > PROMOTION_MAX_CHANNEL_CARDS:
        omitted = len(channel_cards) - PROMOTION_MAX_CHANNEL_CARDS
        warnings.append(f"channel_daily_syntheses_omitted={omitted}")
        channel_cards = channel_cards[:PROMOTION_MAX_CHANNEL_CARDS]

    evidence_cards = await _evidence_cards_for_daily_docs(daily_docs)
    if len(evidence_cards) > PROMOTION_MAX_EVIDENCE_CARDS:
        omitted = len(evidence_cards) - PROMOTION_MAX_EVIDENCE_CARDS
        warnings.append(f"evidence_cards_omitted={omitted}")
        evidence_cards = evidence_cards[:PROMOTION_MAX_EVIDENCE_CARDS]

    payload: GlobalPromotionPromptPayload = {
        "evaluation_mode": "daily_global_promotion",
        "character_local_date": character_local_date,
        "character_time_zone": CHARACTER_TIME_ZONE,
        "channel_daily_syntheses": channel_cards,
        "evidence_cards": evidence_cards,
        "promotion_limits": {
            "max_lore": 1,
            "max_self_guidance": 1,
            "max_total_decisions": 2,
        },
        "review_questions": [
            "哪些内容满足长期全局 lore 的高信号标准？",
            "哪些内容只应成为未来回应方式的 self_guidance？",
            "哪些内容因为用户事实、隐私或边界风险必须拒绝？",
        ],
    }
    return payload, warnings


def validate_promotion_decisions(
    decisions: list[ReflectionPromotionDecision],
) -> list[str]:
    """Return structural warnings for promotion decisions."""

    warnings: list[str] = []
    lane_counts = {"lore": 0, "self_guidance": 0}
    for index, decision in enumerate(decisions):
        lane = str(decision.get("lane", "") or "")
        action = str(decision.get("decision", "") or "")
        if lane not in PROMOTION_LANE_MEMORY_TYPE:
            warnings.append(f"decision[{index}] invalid lane: {lane}")
            continue
        lane_counts[lane] += 1
        if action in {"reject", "no_action"}:
            continue
        warnings.extend(_validate_promote_decision(index, decision))
    for lane, count in lane_counts.items():
        if count > 1:
            warnings.append(f"too many decisions for lane {lane}: {count}")
    return warnings


async def _persist_global_run(
    *,
    character_local_date: str,
    source_run_ids: list[str],
    output: dict[str, Any],
    promotion_decisions: list[dict[str, Any]],
    status: str,
    attempt_count: int,
    validation_warnings: list[str],
    error: str = "",
) -> CharacterReflectionRunDoc:
    """Persist a global promotion run through the repository."""

    document = repository.build_global_promotion_run_document(
        character_local_date=character_local_date,
        prompt_version=GLOBAL_PROMOTION_PROMPT_VERSION,
        source_run_ids=source_run_ids,
        output=output,
        promotion_decisions=promotion_decisions,
        status=status,
        attempt_count=attempt_count,
        validation_warnings=validation_warnings,
        error=error,
    )
    await repository.upsert_run(document)
    return document


def _promotion_decisions_from_output(
    parsed_output: dict[str, Any],
) -> list[ReflectionPromotionDecision]:
    """Return normalized promotion decision rows from parsed LLM output."""

    raw_decisions = parsed_output.get("promotion_decisions")
    if not isinstance(raw_decisions, list):
        return_value: list[ReflectionPromotionDecision] = []
        return return_value
    decisions: list[ReflectionPromotionDecision] = []
    for item in raw_decisions:
        if isinstance(item, dict):
            decisions.append(dict(item))
    return decisions


def _attach_repository_evidence_refs(
    decisions: list[ReflectionPromotionDecision],
    payload: GlobalPromotionPromptPayload,
) -> list[ReflectionPromotionDecision]:
    """Replace LLM evidence refs with repository-derived prompt evidence refs."""

    evidence_refs_by_lane = _repository_evidence_refs_by_lane(payload)
    normalized_decisions: list[ReflectionPromotionDecision] = []
    for decision in decisions:
        normalized_decision: ReflectionPromotionDecision = dict(decision)
        lane = str(normalized_decision.get("lane", "") or "")
        action = str(normalized_decision.get("decision", "") or "")
        if action not in {"reject", "no_action"}:
            normalized_decision["evidence_refs"] = list(
                evidence_refs_by_lane.get(lane, []),
            )
        normalized_decisions.append(normalized_decision)
    return normalized_decisions


def _repository_evidence_refs_by_lane(
    payload: GlobalPromotionPromptPayload,
) -> dict[str, list[MemoryEvidenceRef]]:
    """Build allowed memory evidence refs from sanitized evidence cards."""

    refs_by_lane: dict[str, list[MemoryEvidenceRef]] = {
        "lore": [],
        "self_guidance": [],
    }
    seen_by_lane: dict[str, set[tuple[str, str]]] = {
        "lore": set(),
        "self_guidance": set(),
    }
    for card in payload["evidence_cards"]:
        supports = card.get("supports")
        if not isinstance(supports, list):
            continue
        source_run_ids = card.get("source_reflection_run_ids")
        if not isinstance(source_run_ids, list):
            continue
        scope_ref = str(card.get("scope_ref", "") or "")
        captured_at = str(card.get("captured_at", "") or "")
        for lane in supports:
            if lane not in refs_by_lane:
                continue
            for source_run_id in source_run_ids:
                normalized_source_run_id = str(source_run_id).strip()
                if not normalized_source_run_id:
                    continue
                dedupe_key = (normalized_source_run_id, scope_ref)
                if dedupe_key in seen_by_lane[lane]:
                    continue
                evidence_ref: MemoryEvidenceRef = {
                    "reflection_run_id": normalized_source_run_id,
                    "scope_ref": scope_ref,
                    "source": "reflection_cycle",
                }
                if captured_at:
                    evidence_ref["captured_at"] = captured_at
                refs_by_lane[lane].append(evidence_ref)
                seen_by_lane[lane].add(dedupe_key)
    return refs_by_lane


def _validate_promote_decision(
    index: int,
    decision: ReflectionPromotionDecision,
) -> list[str]:
    """Validate one non-reject promotion decision."""

    warnings: list[str] = []
    lane = str(decision["lane"])
    expected_memory_type = PROMOTION_LANE_MEMORY_TYPE[lane]
    memory_type = str(decision.get("memory_type", "") or "")
    if memory_type != expected_memory_type:
        warnings.append(
            f"decision[{index}] memory_type must be {expected_memory_type}"
        )
    authority = str(decision.get("authority", "") or "")
    if authority != MemoryAuthority.REFLECTION_PROMOTED:
        warnings.append(f"decision[{index}] authority must be reflection_promoted")
    if decision.get("signal_strength") != "high":
        warnings.append(f"decision[{index}] signal_strength must be high")
    if not str(decision.get("sanitized_memory_name", "") or "").strip():
        warnings.append(f"decision[{index}] sanitized_memory_name is required")
    if not str(decision.get("sanitized_content", "") or "").strip():
        warnings.append(f"decision[{index}] sanitized_content is required")
    privacy_review = decision.get("privacy_review")
    if not isinstance(privacy_review, dict):
        warnings.append(f"decision[{index}] privacy_review is required")
    else:
        if privacy_review.get("user_details_removed") is not True:
            warnings.append(f"decision[{index}] user details must be removed")
        private_risk = privacy_review.get("private_detail_risk")
        if private_risk not in {"low", "medium"}:
            warnings.append(f"decision[{index}] private_detail_risk blocks write")
    boundary = decision.get("boundary_assessment")
    if not isinstance(boundary, dict):
        warnings.append(f"decision[{index}] boundary_assessment is required")
    elif lane == "lore" and boundary.get("verdict") != "acceptable":
        warnings.append(f"decision[{index}] lore boundary verdict blocks write")
    elif (
        lane == "self_guidance"
        and boundary.get("affects_identity_or_boundaries") is True
        and boundary.get("verdict") != "acceptable"
    ):
        warnings.append(
            f"decision[{index}] self_guidance boundary verdict blocks write"
        )
    if lane == "lore" and decision.get("character_agreement") not in {
        "spoken",
        "agreed",
    }:
        warnings.append(f"decision[{index}] lore requires character agreement")
    evidence_refs = decision.get("evidence_refs")
    if not isinstance(evidence_refs, list) or not evidence_refs:
        warnings.append(f"decision[{index}] evidence_refs are required")
    if "source_global_user_id" in decision:
        warnings.append(f"decision[{index}] source_global_user_id is forbidden")
    return warnings


async def _write_validated_promotion_decisions(
    *,
    decisions: list[ReflectionPromotionDecision],
    character_local_date: str,
    global_run_id: str,
    is_primary_interaction_busy: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Apply validated decisions through memory-evolution public APIs."""

    mutations: list[dict[str, Any]] = []
    warnings: list[str] = []
    deferred = False
    defer_reason = ""
    wrote_by_lane: set[str] = set()
    for index, decision in enumerate(decisions):
        lane = str(decision.get("lane", "") or "")
        if lane in wrote_by_lane:
            warnings.append(f"lane already wrote this pass: {lane}")
            continue
        if not _lane_enabled(lane):
            warnings.append(f"lane disabled: {lane}")
            logger.info(f"Reflection promotion lane disabled: lane={lane}")
            continue
        if str(decision.get("decision", "") or "") in {"reject", "no_action"}:
            logger.info(
                "Reflection promotion no-write decision: "
                f"lane={lane} decision={decision.get('decision')}"
            )
            continue
        decision_warnings = _validate_promote_decision(index, decision)
        if decision_warnings:
            warnings.extend(decision_warnings)
            logger.info(
                "Reflection promotion rejected candidate: "
                f"lane={lane} reasons={decision_warnings}"
            )
            continue
        if (
            is_primary_interaction_busy is not None
            and is_primary_interaction_busy()
        ):
            deferred = True
            defer_reason = "primary interaction busy"
            logger.info(
                "Reflection promotion memory write deferred: "
                f"lane={lane} reason={defer_reason}"
            )
            break
        write_result = await _resolve_similarity_and_write(
            decision=decision,
            character_local_date=character_local_date,
            global_run_id=global_run_id,
            is_primary_interaction_busy=is_primary_interaction_busy,
        )
        warnings.extend(write_result["warnings"])
        if write_result["deferred"]:
            deferred = True
            defer_reason = write_result["defer_reason"]
            continue
        mutation = write_result.get("mutation")
        if isinstance(mutation, dict):
            mutations.append(mutation)
            wrote_by_lane.add(lane)
    result = {
        "mutations": mutations,
        "warnings": warnings,
        "deferred": deferred,
        "defer_reason": defer_reason,
    }
    return result


async def _resolve_similarity_and_write(
    *,
    decision: ReflectionPromotionDecision,
    character_local_date: str,
    global_run_id: str,
    is_primary_interaction_busy: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Use scored active-memory matches to choose insert, supersede, or merge."""

    lane = decision["lane"]
    memory_type = PROMOTION_LANE_MEMORY_TYPE[lane]
    semantic_query = (
        f"{decision['sanitized_memory_name']}\n{decision['sanitized_content']}"
    )
    try:
        scored_units = await find_active_memory_units(
            query={
                "semantic_query": semantic_query,
                "source_kind": MemorySourceKind.REFLECTION_INFERRED,
                "source_global_user_id": "",
                "memory_type": memory_type,
            },
            limit=5,
        )
    except RuntimeError as exc:
        deferred_result = _deferred_result(f"memory search unavailable: {exc}")
        return deferred_result

    if not _score_rows_are_valid(scored_units):
        deferred_result = _deferred_result(
            "memory search returned malformed score rows"
        )
        return deferred_result

    mutation_action = "insert"
    source_unit_ids: list[str] = []
    source_lineage_ids: list[str] = []
    top_score = -1.0
    if scored_units:
        top_score = float(scored_units[0][0])
    merge_candidates = [
        document
        for score, document in scored_units
        if score >= PROMOTION_MERGE_SCORE_THRESHOLD
    ]
    if len(merge_candidates) >= 2:
        mutation_action = "merge"
        source_unit_ids = [
            str(document["memory_unit_id"])
            for document in merge_candidates
        ]
        source_lineage_ids = [
            str(document["lineage_id"])
            for document in merge_candidates
        ]
    elif scored_units and top_score >= PROMOTION_DUPLICATE_SCORE_THRESHOLD:
        mutation_action = "supersede"
        source_unit_ids = [str(scored_units[0][1]["memory_unit_id"])]
        source_lineage_ids = [str(scored_units[0][1]["lineage_id"])]
    elif top_score >= PROMOTION_REVIEW_BAND_SCORE_THRESHOLD:
        warning = f"duplicate review band score={top_score:.3f}"
        logger.info(
            "Reflection promotion skipped for duplicate review: "
            f"lane={lane} score={top_score:.3f}"
        )
        return {
            "mutation": None,
            "warnings": [warning],
            "deferred": False,
            "defer_reason": "",
        }

    memory_doc = _memory_document_for_decision(
        decision=decision,
        character_local_date=character_local_date,
        global_run_id=global_run_id,
        source_unit_ids=source_unit_ids,
        source_lineage_ids=source_lineage_ids,
        mutation_action=mutation_action,
    )
    logger.debug(
        "Reflection promotion similarity decision: "
        f"lane={lane} action={mutation_action} top_score={top_score:.3f} "
        f"source_unit_ids={source_unit_ids} "
        f"evidence_refs={_evidence_ref_ids(decision)}"
    )
    if is_primary_interaction_busy is not None and is_primary_interaction_busy():
        deferred_result = _deferred_result("primary interaction busy")
        return deferred_result
    try:
        stored = await _write_memory_doc(
            action=mutation_action,
            source_unit_ids=source_unit_ids,
            memory_doc=memory_doc,
        )
    except RuntimeError as exc:
        if str(exc) == "memory write or reset is already running":
            deferred_result = _deferred_result(str(exc))
            return deferred_result
        raise

    mutation = {
        "lane": lane,
        "action": mutation_action,
        "memory_unit_id": stored["memory_unit_id"],
        "lineage_id": stored["lineage_id"],
        "memory_type": stored["memory_type"],
        "memory_name": stored["memory_name"],
        "content": stored["content"],
    }
    logger.info(
        "Reflection promotion memory mutation: "
        f"lane={lane} action={mutation_action} "
        f"memory_unit_id={stored['memory_unit_id']} "
        f"lineage_id={stored['lineage_id']} "
        f"memory_type={stored['memory_type']} "
        f"name={stored['memory_name']} "
        f"content_preview={_preview_text(str(stored['content']), 160)} "
        f"source_reflection_runs={len(decision.get('evidence_refs', []))} "
        f"run_id={global_run_id}"
    )
    result = {
        "mutation": mutation,
        "warnings": [],
        "deferred": False,
        "defer_reason": "",
    }
    return result


async def _write_memory_doc(
    *,
    action: str,
    source_unit_ids: list[str],
    memory_doc: EvolvingMemoryDoc,
) -> EvolvingMemoryDoc:
    """Call the selected public memory-evolution mutation API."""

    if action == "merge":
        stored = await merge_memory_units(
            source_unit_ids=source_unit_ids,
            replacement=memory_doc,
        )
    elif action == "supersede":
        stored = await supersede_memory_unit(
            active_unit_id=source_unit_ids[0],
            replacement=memory_doc,
        )
    else:
        stored = await insert_memory_unit(document=memory_doc)
    return stored


def _memory_document_for_decision(
    *,
    decision: ReflectionPromotionDecision,
    character_local_date: str,
    global_run_id: str,
    source_unit_ids: list[str],
    source_lineage_ids: list[str],
    mutation_action: str,
) -> EvolvingMemoryDoc:
    """Build an evolving memory document from a validated decision."""

    lane = decision["lane"]
    evidence_refs = list(decision.get("evidence_refs", []))
    source_run_ids = sorted(
        str(ref.get("reflection_run_id", "") or global_run_id)
        for ref in evidence_refs
    )
    memory_unit_id = deterministic_memory_unit_id(
        "reflection",
        [
            lane,
            character_local_date,
            str(decision["sanitized_memory_name"]),
            str(decision["sanitized_content"]),
            *source_run_ids,
        ],
    )
    source_lineages = list(dict.fromkeys(source_lineage_ids))
    lineage_id = memory_unit_id
    if mutation_action == "supersede" and source_lineages:
        lineage_id = source_lineages[0]
    elif mutation_action == "merge" and len(source_lineages) == 1:
        lineage_id = source_lineages[0]
    privacy_review: MemoryPrivacyReview = dict(decision["privacy_review"])
    privacy_review["reviewer"] = "automated_llm"
    memory_doc: EvolvingMemoryDoc = {
        "memory_unit_id": memory_unit_id,
        "lineage_id": lineage_id,
        "version": 1,
        "memory_name": str(decision["sanitized_memory_name"]),
        "content": str(decision["sanitized_content"]),
        "source_global_user_id": "",
        "memory_type": PROMOTION_LANE_MEMORY_TYPE[lane],
        "source_kind": MemorySourceKind.REFLECTION_INFERRED,
        "authority": MemoryAuthority.REFLECTION_PROMOTED,
        "status": MemoryStatus.ACTIVE,
        "supersedes_memory_unit_ids": [],
        "merged_from_memory_unit_ids": [],
        "evidence_refs": evidence_refs,
        "privacy_review": privacy_review,
        "confidence_note": f"Promoted from reflection run {global_run_id}.",
        "timestamp": repository.now_iso(),
        "expiry_timestamp": None,
    }
    return memory_doc


def _score_rows_are_valid(rows: object) -> bool:
    """Return whether semantic search returned score/doc tuples."""

    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2:
            return False
        score, document = row
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return False
        if not isinstance(document, dict):
            return False
    return True


def _deferred_result(reason: str) -> dict[str, Any]:
    """Build a no-mutation deferred result."""

    logger.info(f"Reflection promotion deferred: reason={reason}")
    result = {
        "mutation": None,
        "warnings": [reason],
        "deferred": True,
        "defer_reason": reason,
    }
    return result


def _lane_enabled(lane: str) -> bool:
    """Return whether a promotion lane is enabled by process-loaded config."""

    if lane == "lore":
        return_value = REFLECTION_LORE_PROMOTION_ENABLED
    elif lane == "self_guidance":
        return_value = REFLECTION_SELF_GUIDANCE_PROMOTION_ENABLED
    else:
        return_value = False
    return return_value


def _channel_daily_cards(
    daily_docs: list[CharacterReflectionRunDoc],
) -> list[ChannelDailySynthesisCard]:
    """Build compact daily cards from persisted daily-channel runs."""

    cards: list[ChannelDailySynthesisCard] = []
    for document in daily_docs:
        output = document.get("output", {})
        if not isinstance(output, dict):
            output = {}
        scope = document["scope"]
        card: ChannelDailySynthesisCard = {
            "daily_run_id": str(document["run_id"]),
            "scope_ref": str(scope["scope_ref"]),
            "channel_type": str(scope["channel_type"]),
            "character_local_date": str(document["character_local_date"]),
            "confidence": _confidence_value(output.get("confidence")),
            "day_summary": _preview_text(str(output.get("day_summary", "") or ""), 240),
            "cross_hour_topics": _compact_text_items(
                output.get("cross_hour_topics"),
                max_items=3,
                max_chars=120,
            ),
            "conversation_quality_patterns": _compact_text_items(
                output.get("conversation_quality_patterns"),
                max_items=3,
                max_chars=120,
            ),
            "privacy_risk_labels": _compact_text_items(
                output.get("privacy_risks"),
                max_items=3,
                max_chars=120,
            ),
            "validation_warning_labels": _compact_text_items(
                document.get("validation_warnings"),
                max_items=3,
                max_chars=120,
            ),
        }
        cards.append(_cap_serialized_card(card, PROMOTION_MAX_CHANNEL_CARD_CHARS))
    return cards


async def _evidence_cards_for_daily_docs(
    daily_docs: list[CharacterReflectionRunDoc],
) -> list[ReflectionEvidenceCard]:
    """Build sanitized evidence cards from daily source hourly runs."""

    cards: list[ReflectionEvidenceCard] = []
    for daily_doc in daily_docs:
        source_run_ids = [
            str(run_id)
            for run_id in daily_doc.get("source_reflection_run_ids", [])
        ]
        for source_run_id in source_run_ids:
            source_doc = await repository.reflection_run_by_id(source_run_id)
            if source_doc is None:
                continue
            cards.extend(_evidence_cards_from_hourly_doc(source_doc))
    return cards


def _evidence_cards_from_hourly_doc(
    hourly_doc: CharacterReflectionRunDoc,
) -> list[ReflectionEvidenceCard]:
    """Build evidence cards from one hourly run document."""

    output = hourly_doc.get("output", {})
    if not isinstance(output, dict):
        return_value: list[ReflectionEvidenceCard] = []
        return return_value
    utterances = output.get("active_character_utterances")
    if not isinstance(utterances, list):
        utterances = []
    quality_items = _compact_text_items(
        output.get("conversation_quality_feedback"),
        max_items=2,
        max_chars=120,
    )
    topic_summary = str(output.get("topic_summary", "") or "")
    cards: list[ReflectionEvidenceCard] = []
    scope = hourly_doc["scope"]
    source_run_id = str(hourly_doc["run_id"])
    if not utterances and not quality_items and not topic_summary:
        return cards
    lead_utterance = str(utterances[0]) if utterances else ""
    observation = topic_summary or " ".join(quality_items)
    card: ReflectionEvidenceCard = {
        "evidence_card_id": f"evidence_{source_run_id}",
        "source_reflection_run_ids": [source_run_id],
        "scope_ref": str(scope["scope_ref"]),
        "channel_type": str(scope["channel_type"]),
        "character_local_date": str(hourly_doc["character_local_date"]),
        "captured_at": str(
            hourly_doc.get("created_at")
            or hourly_doc.get("hour_start")
            or hourly_doc.get("window_start")
            or "",
        ),
        "active_character_utterance": _preview_text(lead_utterance, 180),
        "sanitized_observation": _preview_text(observation, 180),
        "supports": ["lore", "self_guidance"],
        "private_detail_risk": "low",
    }
    cards.append(_cap_serialized_card(card, PROMOTION_MAX_EVIDENCE_CARD_CHARS))
    return cards


def _compact_text_items(
    value: object,
    *,
    max_items: int,
    max_chars: int,
) -> list[str]:
    """Return bounded string items from an optional list-like value."""

    if isinstance(value, list):
        raw_items = value
    elif value:
        raw_items = [value]
    else:
        raw_items = []
    items = [
        _preview_text(str(item), max_chars)
        for item in raw_items[:max_items]
    ]
    return items


def _confidence_value(value: object) -> Literal["low", "medium", "high"]:
    """Normalize confidence into the allowed prompt enum."""

    if value in {"low", "medium", "high"}:
        return_value: Literal["low", "medium", "high"] = value
        return return_value
    return_value = "low"
    return return_value


def _cap_serialized_card(
    card: dict[str, Any],
    max_chars: int,
) -> dict[str, Any]:
    """Trim long text fields until the serialized card fits the cap."""

    capped = dict(card)
    while len(json.dumps(capped, ensure_ascii=False, sort_keys=True)) > max_chars:
        longest_key = ""
        longest_value = ""
        for key, value in capped.items():
            if isinstance(value, str) and len(value) > len(longest_value):
                longest_key = key
                longest_value = value
        if not longest_key or len(longest_value) <= 40:
            break
        capped[longest_key] = _preview_text(longest_value, len(longest_value) - 20)
    return capped


def _preview_text(value: str, max_chars: int) -> str:
    """Return compact text safe for prompts and operator logs."""

    cleaned = " ".join(value.split())
    if len(cleaned) <= max_chars:
        return cleaned
    preview = f"{cleaned[:max_chars - 3]}..."
    return preview


def _evidence_ref_ids(decision: ReflectionPromotionDecision) -> list[str]:
    """Return evidence ids for debug logging."""

    evidence_refs = decision.get("evidence_refs", [])
    ids = [
        str(ref.get("reflection_run_id", ""))
        for ref in evidence_refs
        if isinstance(ref, dict)
    ]
    return ids
