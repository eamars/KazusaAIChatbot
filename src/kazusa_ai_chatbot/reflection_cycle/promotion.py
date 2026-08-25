"""Global reflection promotion prompt and memory integration."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    CONSOLIDATION_LLM_MODEL,
    CONSOLIDATION_LLM_THINKING_ENABLED,
    REFLECTION_LORE_PROMOTION_ENABLED,
    REFLECTION_SELF_GUIDANCE_PROMOTION_ENABLED,
)
from kazusa_ai_chatbot.db import get_character_profile
from kazusa_ai_chatbot.db.schemas import (
    CharacterReflectionRunDoc,
    ReflectionEpisodeRefDoc,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
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
from kazusa_ai_chatbot.memory_writer_prompt_projection import (
    project_reflection_promotion_prompt_payload,
)
from kazusa_ai_chatbot.reflection_cycle import repository
from kazusa_ai_chatbot.reflection_cycle.models import (
    REFLECTION_RUN_KIND_DAILY_GLOBAL_PROMOTION,
    REFLECTION_STATUS_FAILED,
    REFLECTION_STATUS_SKIPPED,
    REFLECTION_STATUS_SUCCEEDED,
    PromptBuildResult,
    ReflectionPromotionResult,
)
from kazusa_ai_chatbot.reflection_cycle.projection import build_prompt_result
from kazusa_ai_chatbot.time_boundary import format_storage_utc_for_llm
from kazusa_ai_chatbot.utils import parse_llm_json_output

logger = logging.getLogger(__name__)

GLOBAL_PROMOTION_PROMPT_VERSION = "reflection_global_promotion_v2"
GLOBAL_PROMOTION_REVIEW_PROMPT_VERSION = "reflection_global_promotion_review_v1"
GLOBAL_PROMOTION_PROMPT_MAX_CHARS = 25000
GLOBAL_PROMOTION_REVIEW_PROMPT_MAX_CHARS = 25000
PROMOTION_LANE_MEMORY_TYPE = {
    "lore": "fact",
    "self_guidance": "defense_rule",
}
PROMOTION_DUPLICATE_SCORE_THRESHOLD = 0.92
PROMOTION_MERGE_SCORE_THRESHOLD = 0.88
PROMOTION_REVIEW_BAND_SCORE_THRESHOLD = 0.82
PROMOTION_DUPLICATE_REPLACEMENT_ERROR = "replacement memory_unit_id already exists"
PROMOTION_MAX_CHANNEL_CARDS = 25
PROMOTION_MAX_EVIDENCE_CARDS = 40
PROMOTION_MAX_CHANNEL_CARD_CHARS = 600
# The canonical repository envelope is 462 characters with readable fields
# empty, leaving 178 characters for bounded readable evidence.
PROMOTION_MAX_EVIDENCE_CARD_CHARS = 640
# The independent reviewer receives only a bounded, lane-matched evidence set.
PROMOTION_MAX_REVIEW_EVIDENCE_CARDS = 8
GLOBAL_PROMOTION_ATTEMPT_LIMIT = 3
PROMOTION_MUTATING_ACTIONS = frozenset({
    "promote_new",
    "supersede",
    "merge",
})
_PROMOTION_DECISION_KEYS = frozenset({
    "lane",
    "decision",
    "selected_candidate_id",
    "sanitized_memory_name",
    "sanitized_content",
    "memory_type",
    "authority",
    "signal_strength",
    "character_agreement",
    "boundary_assessment",
    "privacy_review",
    "evidence_refs",
})
_PROMOTER_PRIVACY_REVIEW_KEYS = frozenset({
    "global_applicability",
    "target_specific_meaning_removed",
    "affects_identity_or_boundaries",
    "private_detail_risk",
    "user_details_removed",
    "boundary_assessment",
    "reviewer",
})
_PROMOTION_REVIEW_KEYS = frozenset({
    "selected_candidate_id",
    "decision",
    "global_applicability",
    "target_specific_meaning_removed",
    "affects_identity_or_boundaries",
    "private_detail_risk",
    "user_details_removed",
    "reason",
})
_PROMOTION_REVIEW_DECISIONS = frozenset({"accept", "reject"})
_PROMOTION_REVIEW_REASON_MAX_CHARS = 500
_CARD_READABLE_TEXT_KEYS = frozenset({
    "active_character_utterance",
    "day_summary",
    "sanitized_observation",
})
_CARD_READABLE_LIST_KEYS = frozenset({
    "conversation_quality_patterns",
    "cross_hour_topics",
    "privacy_risk_labels",
    "source_privacy_notes",
    "validation_warning_labels",
})


class GlobalPromotionContractError(RuntimeError):
    """Raised after bounded full replacements remain contract-invalid."""

    def __init__(
        self,
        *,
        attempt_count: int,
        validation_errors: list[str],
    ) -> None:
        super().__init__(
            "global promotion contract failed after "
            f"{attempt_count} complete replacements: "
            + "; ".join(validation_errors)
        )
        self.attempt_count = attempt_count
        self.validation_errors = list(validation_errors)


class GlobalPromotionReviewContractError(RuntimeError):
    """Raised when the one-shot independent review is structurally invalid."""

    def __init__(self, validation_errors: list[str]) -> None:
        super().__init__(
            "global promotion review contract failed: "
            + "; ".join(validation_errors)
        )
        self.validation_errors = list(validation_errors)


class ReflectionBoundaryAssessment(TypedDict):
    """Boundary review emitted by the global promotion prompt."""

    verdict: Literal["acceptable", "needs_human_review", "blocked"]
    affects_identity_or_boundaries: bool
    reason: str


class ReflectionScopeCertificate(TypedDict):
    """Six-field independent scope and privacy certificate."""

    global_applicability: Literal["global", "scoped", "absent"]
    target_specific_meaning_removed: bool
    affects_identity_or_boundaries: bool
    private_detail_risk: Literal["low", "medium", "high"]
    user_details_removed: bool
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
    promoter_privacy_review: MemoryPrivacyReview
    reviewer_privacy_review: ReflectionScopeCertificate
    review_decision: Literal["accept", "reject", "not_requested"]
    review_admitted: bool


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
    source_privacy_notes: list[str]
    private_detail_risk: Literal["low", "medium", "high", "unreviewed"]


class PromotionLimits(TypedDict):
    """Hard daily mutation caps visible to the LLM."""

    max_lore: Literal[1]
    max_self_guidance: Literal[1]
    max_total_decisions: Literal[2]


class GlobalPromotionPromptPayload(TypedDict):
    """Prompt payload consumed by the global promotion LLM."""

    evaluation_mode: Literal["daily_global_promotion"]
    character_local_date: str
    channel_daily_syntheses: list[ChannelDailySynthesisCard]
    evidence_cards: list[ReflectionEvidenceCard]
    promotion_limits: PromotionLimits
    review_questions: list[str]


class PromotionReviewPromptPayload(TypedDict):
    """Prompt payload for the independent candidate review stage."""

    evaluation_mode: Literal["daily_global_promotion_review"]
    character_local_date: str
    candidates: list[dict[str, str]]
    evidence_cards: list[ReflectionEvidenceCard]


GLOBAL_PROMOTION_SYSTEM_PROMPT = '''\
# 任务
你负责审阅每日频道反思，只输出可验证、去隐私、可长期使用的全局晋升决定。

# 核心任务
在 lore 与 self_guidance 两个通道中，各最多选择一条高信号内容；没有足够证据时输出 no_action 或 reject。

# 语言政策
JSON 字段名和枚举值必须保持英文。你新生成的自由文本字段必须使用简体中文。证据片段保持原文。

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

# 生成步骤
1. 检查 channel_daily_syntheses，只把它当作压缩后的反思证据。
2. 检查 evidence_cards，确认是否有 source_utterance 支持 `{character_name}` 说过或同意过的内容。
3. 排除用户事实、用户偏好、关系承诺、健康信息、私密身份信息。
   先查看 source_privacy_notes；private_detail_risk 为 unreviewed 表示来源评估缺失，
   不得把它当作 low。只有经过当前审阅确认后，才可给出写入结论。
4. 分别判断 lore 与 self_guidance 是否有高信号。
5. 如果证据 private_detail_risk 是 high，必须输出 reject，并让 privacy_review.private_detail_risk 保持 high。
6. evidence_refs.reflection_run_id 只能来自 evidence_cards.source_reflection_run_ids；不要使用 daily_run_id。
7. 输出 promotion_decisions；不要输出数据库字段、数据库查询、向量、source_global_user_id。

# 输入格式
{{
  "evaluation_mode": "daily_global_promotion",
  "character_local_date": "YYYY-MM-DD",
  "channel_daily_syntheses": [
    {{
      "daily_run_id": "反思运行标识",
      "scope_ref": "范围标识",
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
      "evidence_card_id": "证据卡标识",
      "source_reflection_run_ids": ["反思运行标识"],
      "scope_ref": "范围标识",
      "channel_type": "private|group|system|unknown",
      "character_local_date": "YYYY-MM-DD",
      "captured_at": "YYYY-MM-DD HH:MM 证据标签",
      "source_utterance": "{character_name} 原文片段",
      "sanitized_observation": "去身份化观察",
      "supports": ["lore", "self_guidance"],
      "source_privacy_notes": ["来源隐私说明"],
      "private_detail_risk": "low|medium|high|unreviewed"
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
promotion_decisions 字段包含以下决策对象：

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
    "global_applicability": "global|scoped|absent",
    "target_specific_meaning_removed": true,
    "affects_identity_or_boundaries": false,
    "private_detail_risk": "low|medium|high",
    "user_details_removed": true,
    "boundary_assessment": "边界摘要",
    "reviewer": "automated_llm"
  }},
  "evidence_refs": [
    {{
      "reflection_run_id": "来源运行标识",
      "scope_ref": "范围标识",
      "captured_at": "YYYY-MM-DD HH:MM 证据标签",
      "source": "reflection_cycle"
    }}
  ]
}}

# 禁止事项
不要编造证据。不要从用户发言改写成 `{character_name}` 的长期规则。不要把 reject/no_action 改成 promote_new。
不要输出 source_global_user_id、数据库查询字段、向量、原始记录、用户身份、用户承诺、健康细节或私密关系事实。
privacy_review 必须完整返回 global_applicability、target_specific_meaning_removed、
affects_identity_or_boundaries、private_detail_risk、user_details_removed、
boundary_assessment 和 reviewer；这些字段必须与 boundary_assessment 保持一致。
'''

GLOBAL_PROMOTION_REVIEW_SYSTEM_PROMPT = '''\
# 任务
你负责独立审阅已经提出的全局晋升候选项。你只能判断每个候选项是否可接受及其范围和隐私证书，
不得改写候选项的名称、内容、通道、类型、权威或证据。

# 独立性
候选项只包含待审阅的精确含义和稳定标识。不要寻找或推测提出阶段的判断；只依据候选项、来源证据
和 source_privacy_notes 独立判断。source_privacy_notes 为空或 private_detail_risk 为 unreviewed
时，表示来源评估未完成，不等于 low。

# 审阅步骤
1. 对每个 selected_candidate_id 恰好返回一条审阅结果。
2. 移除来源用户、被称呼对象、关系对象和私密场景后，判断候选含义是否仍对角色与一般他人准确且适合。
3. 只有含义在移除特定对象后仍然成立时，才使用 global 并将 target_specific_meaning_removed 设为 true。
4. 任何身份、权限、同意或边界影响都要求 affects_identity_or_boundaries 为 true，并拒绝该候选项。
5. 只有不必要的用户细节已移除且当前审阅风险为 low 时，才可 accept；否则使用 reject。

# 输出格式
只返回合法 JSON，顶层只能有 reviews：
{{
  "reviews": [
    {{
      "selected_candidate_id": "候选项中的稳定标识",
      "decision": "accept|reject",
      "global_applicability": "global|scoped|absent",
      "target_specific_meaning_removed": true,
      "affects_identity_or_boundaries": false,
      "private_detail_risk": "low|medium|high",
      "user_details_removed": true,
      "reason": "有边界的独立审阅理由"
    }}
  ]
}}

# 禁止事项
不要输出候选项未提供的名称、内容、通道、类型、权威、协议或证据。不要输出提出阶段的
privacy_review、boundary_assessment、global_applicability 或其他范围判断。不要输出数据库字段、
数据库查询、向量、source_global_user_id、原始记录、用户身份、用户承诺或敏感隐私信息。
'''
_llm_interface = LLInterface()
_global_promotion_llm = LLInterface()
_global_promotion_review_llm = LLInterface()
_global_promotion_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CONSOLIDATION_LLM",
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=0.2,
    top_p=0.8,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)
_global_promotion_review_llm_config = LLMCallConfig(
    stage_name=f"{__name__}.review",
    route_name="CONSOLIDATION_LLM",
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=0.2,
    top_p=0.8,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)


async def run_global_promotion_llm(
    *,
    prompt: PromptBuildResult,
) -> dict[str, Any]:
    """Run the global promotion LLM and return parsed JSON output."""

    response = await _global_promotion_llm.ainvoke([
        SystemMessage(content=prompt.system_prompt),
        HumanMessage(content=prompt.human_prompt),
    ], config=_global_promotion_llm_config)
    raw_output = str(response.content)
    parsed = parse_llm_json_output(raw_output)
    if not isinstance(parsed, dict):
        raise TypeError("global promotion output must be a JSON object")
    return_value = dict(parsed)
    return return_value


async def run_global_promotion_review_llm(
    *,
    prompt: PromptBuildResult,
) -> dict[str, Any]:
    """Run the one-shot independent scope/privacy review LLM."""

    response = await _global_promotion_review_llm.ainvoke([
        SystemMessage(content=prompt.system_prompt),
        HumanMessage(content=prompt.human_prompt),
    ], config=_global_promotion_review_llm_config)
    raw_output = str(response.content)
    parsed = parse_llm_json_output(raw_output)
    if not isinstance(parsed, dict):
        raise TypeError("global promotion review output must be a JSON object")
    return_value = dict(parsed)
    return return_value


async def _run_validated_global_promotion_llm(
    *,
    prompt: PromptBuildResult,
    payload: GlobalPromotionPromptPayload,
) -> tuple[
    dict[str, Any],
    list[ReflectionPromotionDecision],
    int,
    list[str],
]:
    """Request complete replacements until the closed contract validates."""

    prior_errors: list[str] = []
    latest_errors: list[str] = []
    for attempt_count in range(1, GLOBAL_PROMOTION_ATTEMPT_LIMIT + 1):
        parsed_output = await run_global_promotion_llm(prompt=prompt)
        latest_errors = _global_promotion_contract_errors(parsed_output)
        if not latest_errors:
            decisions = _promotion_decisions_from_output(parsed_output)
            decisions = _attach_repository_evidence_refs(
                decisions,
                payload,
            )
            latest_errors = validate_promotion_decisions(decisions)
        if not latest_errors:
            return (
                parsed_output,
                decisions,
                attempt_count,
                prior_errors,
            )
        prior_errors.extend(
            f"attempt[{attempt_count}] {error}"
            for error in latest_errors
        )
    raise GlobalPromotionContractError(
        attempt_count=GLOBAL_PROMOTION_ATTEMPT_LIMIT,
        validation_errors=latest_errors,
    )


async def _review_promotion_candidates(
    *,
    decisions: list[ReflectionPromotionDecision],
    payload: GlobalPromotionPromptPayload,
) -> tuple[list[ReflectionPromotionDecision], list[str]]:
    """Review mutating promoter candidates once before similarity or writes."""

    normalized_decisions = [dict(decision) for decision in decisions]
    candidates = [
        decision
        for decision in normalized_decisions
        if decision.get("decision") in PROMOTION_MUTATING_ACTIONS
    ]
    if not candidates:
        return_value = (normalized_decisions, [])
        return return_value

    review_payload = _promotion_review_payload(candidates, payload)
    review_prompt = build_global_promotion_review_prompt(review_payload)
    parsed_review = await run_global_promotion_review_llm(prompt=review_prompt)
    review_errors = _promotion_review_contract_errors(
        parsed_review,
        candidates,
    )
    if review_errors:
        raise GlobalPromotionReviewContractError(review_errors)

    reviews = parsed_review["reviews"]
    reviews_by_candidate_id = {
        str(review["selected_candidate_id"]): review
        for review in reviews
    }
    reviewed_decisions: list[ReflectionPromotionDecision] = []
    for decision in normalized_decisions:
        if decision.get("decision") not in PROMOTION_MUTATING_ACTIONS:
            reviewed_decisions.append(decision)
            continue
        candidate_id = str(decision["selected_candidate_id"])
        review = reviews_by_candidate_id[candidate_id]
        reviewer_certificate = _reviewer_certificate_from_review(review)
        promoter_certificate = dict(decision["privacy_review"])
        decision["promoter_privacy_review"] = promoter_certificate
        decision["reviewer_privacy_review"] = reviewer_certificate
        decision["review_decision"] = review["decision"]
        decision["review_admitted"] = _review_admits_write(
            promoter_certificate=promoter_certificate,
            reviewer_certificate=reviewer_certificate,
            review_decision=str(review["decision"]),
        )
        decision["privacy_review"] = _privacy_review_from_certificate(
            reviewer_certificate,
        )
        reviewed_decisions.append(decision)
    result = (reviewed_decisions, list(review_prompt.validation_warnings))
    return result


def _promotion_review_payload(
    candidates: list[ReflectionPromotionDecision],
    payload: GlobalPromotionPromptPayload,
) -> PromotionReviewPromptPayload:
    """Project candidate meaning and source evidence without promoter judgments."""

    candidate_payload = [
        {
            "selected_candidate_id": str(decision["selected_candidate_id"]),
            "lane": str(decision["lane"]),
            "memory_type": str(decision["memory_type"]),
            "sanitized_memory_name": str(decision["sanitized_memory_name"]),
            "sanitized_content": str(decision["sanitized_content"]),
        }
        for decision in candidates
    ]
    candidate_lanes = {
        str(candidate.get("lane", ""))
        for candidate in candidates
    }
    evidence_cards: list[ReflectionEvidenceCard] = []
    seen_card_ids: set[str] = set()
    source_cards = payload["evidence_cards"]
    ordered_lanes = list(dict.fromkeys(
        str(candidate.get("lane", ""))
        for candidate in candidates
    ))
    selected_source_indexes: set[int] = set()
    for lane in ordered_lanes:
        for source_index, card in enumerate(source_cards):
            supports = card.get("supports", [])
            card_id = str(card.get("evidence_card_id", "") or "")
            if (
                not isinstance(supports, list)
                or lane not in {str(value) for value in supports}
                or (card_id and card_id in seen_card_ids)
            ):
                continue
            evidence_cards.append(
                _cap_serialized_card(
                    dict(card),
                    PROMOTION_MAX_EVIDENCE_CARD_CHARS,
                ),
            )
            selected_source_indexes.add(source_index)
            if card_id:
                seen_card_ids.add(card_id)
            break
    for source_index, card in enumerate(source_cards):
        if source_index in selected_source_indexes:
            continue
        supports = card.get("supports", [])
        if not isinstance(supports, list):
            continue
        if not candidate_lanes.intersection(str(lane) for lane in supports):
            continue
        card_id = str(card.get("evidence_card_id", "") or "")
        if card_id and card_id in seen_card_ids:
            continue
        bounded_card = _cap_serialized_card(
            dict(card),
            PROMOTION_MAX_EVIDENCE_CARD_CHARS,
        )
        evidence_cards.append(bounded_card)
        if card_id:
            seen_card_ids.add(card_id)
        if len(evidence_cards) >= PROMOTION_MAX_REVIEW_EVIDENCE_CARDS:
            break
    review_payload: PromotionReviewPromptPayload = {
        "evaluation_mode": "daily_global_promotion_review",
        "character_local_date": payload["character_local_date"],
        "candidates": candidate_payload,
        "evidence_cards": evidence_cards,
    }
    return review_payload


def _promotion_review_contract_errors(
    parsed_review: Mapping[str, Any],
    candidates: list[ReflectionPromotionDecision],
) -> list[str]:
    """Validate exact one-to-one reviewer coverage and certificate shape."""

    errors: list[str] = []
    if set(parsed_review) != {"reviews"}:
        return ["review output must contain exactly reviews"]
    raw_reviews = parsed_review.get("reviews")
    if not isinstance(raw_reviews, list):
        return ["reviews must be a list"]
    expected_ids = {
        str(candidate["selected_candidate_id"])
        for candidate in candidates
    }
    if len(raw_reviews) != len(expected_ids):
        errors.append("review coverage must match every mutating candidate")
    observed_ids: list[str] = []
    for index, raw_review in enumerate(raw_reviews):
        if not isinstance(raw_review, Mapping):
            errors.append(f"review[{index}] must be an object")
            continue
        if set(raw_review) != _PROMOTION_REVIEW_KEYS:
            errors.append(f"review[{index}] has an invalid key set")
            continue
        candidate_id = raw_review.get("selected_candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            errors.append(f"review[{index}] selected_candidate_id is required")
            continue
        observed_ids.append(candidate_id)
        if candidate_id not in expected_ids:
            errors.append(f"review[{index}] references an unknown candidate")
        if raw_review.get("decision") not in _PROMOTION_REVIEW_DECISIONS:
            errors.append(f"review[{index}] has an invalid decision")
        global_applicability = raw_review.get("global_applicability")
        if global_applicability not in {"global", "scoped", "absent"}:
            errors.append(f"review[{index}] has an invalid applicability")
        for field_name in (
            "target_specific_meaning_removed",
            "affects_identity_or_boundaries",
            "user_details_removed",
        ):
            if not isinstance(raw_review.get(field_name), bool):
                errors.append(f"review[{index}] {field_name} must be boolean")
        if raw_review.get("private_detail_risk") not in {
            "low",
            "medium",
            "high",
        }:
            errors.append(f"review[{index}] has an invalid privacy risk")
        reason = raw_review.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > _PROMOTION_REVIEW_REASON_MAX_CHARS
        ):
            errors.append(f"review[{index}] reason is invalid")
    if len(observed_ids) != len(set(observed_ids)):
        errors.append("review candidate ids must be unique")
    if set(observed_ids) != expected_ids:
        errors.append("review candidate ids must cover exactly the candidates")
    return errors


def _reviewer_certificate_from_review(
    review: Mapping[str, Any],
) -> ReflectionScopeCertificate:
    """Extract the six validated reviewer certificate fields."""

    certificate: ReflectionScopeCertificate = {
        "global_applicability": review["global_applicability"],
        "target_specific_meaning_removed": review[
            "target_specific_meaning_removed"
        ],
        "affects_identity_or_boundaries": review[
            "affects_identity_or_boundaries"
        ],
        "private_detail_risk": review["private_detail_risk"],
        "user_details_removed": review["user_details_removed"],
        "reason": review["reason"],
    }
    return certificate


def _review_admits_write(
    *,
    promoter_certificate: Mapping[str, Any],
    reviewer_certificate: Mapping[str, Any],
    review_decision: str,
) -> bool:
    """Return whether both certificates independently admit one write."""

    promoter_scope_ok = (
        promoter_certificate.get("global_applicability") == "global"
        and promoter_certificate.get("target_specific_meaning_removed") is True
        and promoter_certificate.get("affects_identity_or_boundaries") is False
    )
    reviewer_scope_ok = (
        reviewer_certificate.get("global_applicability") == "global"
        and reviewer_certificate.get("target_specific_meaning_removed") is True
        and reviewer_certificate.get("affects_identity_or_boundaries") is False
        and reviewer_certificate.get("private_detail_risk") == "low"
        and reviewer_certificate.get("user_details_removed") is True
    )
    return_value = (
        review_decision == "accept"
        and promoter_scope_ok
        and reviewer_scope_ok
    )
    return return_value


def _privacy_review_from_certificate(
    certificate: ReflectionScopeCertificate,
) -> MemoryPrivacyReview:
    """Map the actual final reviewer certificate to the memory contract."""

    privacy_review: MemoryPrivacyReview = {
        "global_applicability": certificate["global_applicability"],
        "target_specific_meaning_removed": certificate[
            "target_specific_meaning_removed"
        ],
        "affects_identity_or_boundaries": certificate[
            "affects_identity_or_boundaries"
        ],
        "private_detail_risk": certificate["private_detail_risk"],
        "user_details_removed": certificate["user_details_removed"],
        "boundary_assessment": certificate["reason"],
        "reviewer": "automated_llm",
    }
    return privacy_review


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


def build_global_promotion_review_prompt(
    payload: PromotionReviewPromptPayload,
) -> PromptBuildResult:
    """Build the independent candidate scope/privacy review prompt."""

    prompt = build_prompt_result(
        system_prompt=GLOBAL_PROMOTION_REVIEW_SYSTEM_PROMPT,
        human_payload=payload,
        max_prompt_chars=GLOBAL_PROMOTION_REVIEW_PROMPT_MAX_CHARS,
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

    global_run_id = repository.daily_global_promotion_run_id(
        character_local_date=character_local_date,
        prompt_version=GLOBAL_PROMOTION_PROMPT_VERSION,
    )
    existing_run = await repository.reflection_run_by_id(global_run_id)
    result = ReflectionPromotionResult(
        run_kind=REFLECTION_RUN_KIND_DAILY_GLOBAL_PROMOTION,
        dry_run=dry_run,
        processed_count=1,
    )
    if (
        existing_run is not None
        and str(existing_run.get("run_kind", ""))
        == REFLECTION_RUN_KIND_DAILY_GLOBAL_PROMOTION
        and str(existing_run.get("status", "")) == REFLECTION_STATUS_SUCCEEDED
    ):
        result.skipped_count = 1
        result.defer_reason = "daily global promotion already succeeded"
        result.run_ids.append(global_run_id)
        logger.info(
            "Reflection promotion skipped: "
            f"character_local_date={character_local_date} "
            f"run_id={global_run_id} reason={result.defer_reason}"
        )
        return result

    daily_docs = await repository.daily_channel_runs(
        character_local_date=character_local_date,
    )
    source_run_ids = [str(document["run_id"]) for document in daily_docs]
    source_episode_refs = repository.union_source_episode_refs(daily_docs)
    if not daily_docs:
        result.skipped_count = 1
        result.defer_reason = "no daily_channel runs available"
        await _persist_global_run(
            character_local_date=character_local_date,
            source_run_ids=[],
            source_episode_refs=[],
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
    except Exception as exc:  # noqa: BLE001 - payload construction fails closed
        failed_result = await _fail_global_promotion(
            result=result,
            character_local_date=character_local_date,
            source_run_ids=source_run_ids,
            source_episode_refs=source_episode_refs,
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
            source_episode_refs=source_episode_refs,
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

    try:
        (
            parsed_output,
            decisions,
            attempt_count,
            contract_warnings,
        ) = await _run_validated_global_promotion_llm(
            prompt=prompt,
            payload=payload,
        )
    except Exception as exc:  # noqa: BLE001 - promoter failure fails closed
        attempt_count = int(getattr(exc, "attempt_count", 1))
        failed_result = await _fail_global_promotion(
            result=result,
            character_local_date=character_local_date,
            source_run_ids=source_run_ids,
            source_episode_refs=source_episode_refs,
            attempt_count=attempt_count,
            exc=exc,
        )
        return failed_result
    try:
        decisions, review_warnings = await _review_promotion_candidates(
            decisions=decisions,
            payload=payload,
        )
    except Exception as exc:  # noqa: BLE001 - reviewer failure fails closed
        failed_result = await _fail_global_promotion(
            result=result,
            character_local_date=character_local_date,
            source_run_ids=source_run_ids,
            source_episode_refs=source_episode_refs,
            attempt_count=attempt_count,
            exc=exc,
            output=parsed_output,
            promotion_decisions=[dict(decision) for decision in decisions],
            validation_warnings=(
                payload_warnings
                + list(prompt.validation_warnings)
                + contract_warnings
            ),
        )
        return failed_result
    validation_warnings = (
        payload_warnings
        + list(prompt.validation_warnings)
        + contract_warnings
        + review_warnings
    )
    result.promotion_decisions = [dict(decision) for decision in decisions]

    status = repository.status_for_result(dry_run=dry_run)
    if validation_warnings:
        logger.debug(
            "Reflection promotion validation warnings: "
            f"character_local_date={character_local_date} "
            f"warnings={validation_warnings}"
        )

    if dry_run or not enable_memory_writes:
        result.skipped_count = 1
        result.defer_reason = "memory writes disabled"
        persist_status = status
        if not dry_run:
            persist_status = REFLECTION_STATUS_SKIPPED
        global_run_doc = await _persist_global_run(
            character_local_date=character_local_date,
            source_run_ids=source_run_ids,
            source_episode_refs=source_episode_refs,
            output=parsed_output,
            promotion_decisions=result.promotion_decisions,
            status=persist_status,
            attempt_count=attempt_count,
            validation_warnings=validation_warnings,
        )
        result.run_ids.append(str(global_run_doc["run_id"]))
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
        global_run_doc = await _persist_global_run(
            character_local_date=character_local_date,
            source_run_ids=source_run_ids,
            source_episode_refs=source_episode_refs,
            output=parsed_output,
            promotion_decisions=result.promotion_decisions,
            status=REFLECTION_STATUS_SKIPPED,
            attempt_count=attempt_count,
            validation_warnings=result.validation_warnings,
        )
        result.run_ids.append(str(global_run_doc["run_id"]))
        logger.info(
            "Reflection promotion memory writes deferred: "
            f"character_local_date={character_local_date} "
            f"reason={result.defer_reason}"
        )
        return result

    try:
        write_summary = await _write_validated_promotion_decisions(
            decisions=decisions,
            character_local_date=character_local_date,
            global_run_id=global_run_id,
            is_primary_interaction_busy=is_primary_interaction_busy,
        )
    except Exception as exc:
        result.failed_count = 1
        result.defer_reason = f"{type(exc).__name__}: {exc}"
        result.validation_warnings.extend(validation_warnings)
        result.validation_warnings.append(result.defer_reason)
        logger.exception(
            "Reflection promotion failed during write phase: "
            f"character_local_date={character_local_date}, error={exc}"
        )
        failed_run = await _persist_global_run(
            character_local_date=character_local_date,
            source_run_ids=source_run_ids,
            source_episode_refs=source_episode_refs,
            output=parsed_output,
            promotion_decisions=result.promotion_decisions,
            status=REFLECTION_STATUS_FAILED,
            attempt_count=attempt_count,
            validation_warnings=result.validation_warnings,
            error=result.defer_reason,
        )
        result.run_ids.append(str(failed_run["run_id"]))
        return result

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
            source_episode_refs=source_episode_refs,
            output=parsed_output,
            promotion_decisions=result.promotion_decisions,
            status=REFLECTION_STATUS_SKIPPED,
            attempt_count=attempt_count,
            validation_warnings=result.validation_warnings,
        )
        result.run_ids.append(str(global_run_id))
    if result.succeeded_count == 0 and not result.deferred:
        result.skipped_count += 1
    if not result.deferred:
        replay_skip = any(
            warning == PROMOTION_DUPLICATE_REPLACEMENT_ERROR
            or warning.startswith("replacement already active:")
            for warning in write_summary["warnings"]
        )
        persist_status = REFLECTION_STATUS_SUCCEEDED
        if replay_skip:
            persist_status = REFLECTION_STATUS_SKIPPED
        global_run_doc = await _persist_global_run(
            character_local_date=character_local_date,
            source_run_ids=source_run_ids,
            source_episode_refs=source_episode_refs,
            output=parsed_output,
            promotion_decisions=result.promotion_decisions,
            status=persist_status,
            attempt_count=attempt_count,
            validation_warnings=result.validation_warnings,
        )
        result.run_ids.append(str(global_run_doc["run_id"]))
    return result


async def _fail_global_promotion(
    *,
    result: ReflectionPromotionResult,
    character_local_date: str,
    source_run_ids: list[str],
    source_episode_refs: list[ReflectionEpisodeRefDoc],
    attempt_count: int,
    exc: Exception,
    output: dict[str, Any] | None = None,
    promotion_decisions: list[dict[str, Any]] | None = None,
    validation_warnings: list[str] | None = None,
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
        source_episode_refs=source_episode_refs,
        output=output or {"promotion_decisions": []},
        promotion_decisions=promotion_decisions or [],
        status=REFLECTION_STATUS_FAILED,
        attempt_count=attempt_count,
        validation_warnings=(validation_warnings or []) + [result.defer_reason],
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
    source_episode_refs: list[ReflectionEpisodeRefDoc],
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
        source_episode_refs=source_episode_refs,
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


def _boundary_assessment_contract_errors(
    index: int,
    boundary_assessment: Mapping[str, Any],
) -> list[str]:
    """Validate the promoter boundary assessment without semantic repair."""

    errors: list[str] = []
    expected_keys = {
        "verdict",
        "affects_identity_or_boundaries",
        "reason",
    }
    if set(boundary_assessment) != expected_keys:
        errors.append(
            f"decision[{index}] boundary_assessment has an invalid key set"
        )
        return errors
    if boundary_assessment.get("verdict") not in {
        "acceptable",
        "needs_human_review",
        "blocked",
    }:
        errors.append(f"decision[{index}] boundary verdict is invalid")
    if not isinstance(
        boundary_assessment.get("affects_identity_or_boundaries"),
        bool,
    ):
        errors.append(
            f"decision[{index}] boundary effect must be boolean"
        )
    reason = boundary_assessment.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append(f"decision[{index}] boundary reason is required")
    return errors


def _promoter_privacy_review_contract_errors(
    index: int,
    privacy_review: Mapping[str, Any],
    boundary_assessment: Mapping[str, Any] | None,
) -> list[str]:
    """Validate the promoter's exact learned-memory privacy certificate."""

    errors: list[str] = []
    if set(privacy_review) != _PROMOTER_PRIVACY_REVIEW_KEYS:
        errors.append(
            f"decision[{index}] privacy_review has an invalid key set"
        )
        return errors
    if privacy_review.get("global_applicability") not in {
        "global",
        "scoped",
        "absent",
    }:
        errors.append(f"decision[{index}] privacy applicability is invalid")
    for field_name in (
        "target_specific_meaning_removed",
        "affects_identity_or_boundaries",
        "user_details_removed",
    ):
        if not isinstance(privacy_review.get(field_name), bool):
            errors.append(f"decision[{index}] privacy {field_name} must be boolean")
    if privacy_review.get("private_detail_risk") not in {
        "low",
        "medium",
        "high",
    }:
        errors.append(f"decision[{index}] privacy risk is invalid")
    boundary_summary = privacy_review.get("boundary_assessment")
    if not isinstance(boundary_summary, str) or not boundary_summary.strip():
        errors.append(f"decision[{index}] privacy boundary assessment is required")
    if privacy_review.get("reviewer") != "automated_llm":
        errors.append(f"decision[{index}] privacy reviewer must be automated_llm")
    if (
        isinstance(boundary_assessment, Mapping)
        and isinstance(
            boundary_assessment.get("affects_identity_or_boundaries"),
            bool,
        )
        and privacy_review.get("affects_identity_or_boundaries")
        != boundary_assessment.get("affects_identity_or_boundaries")
    ):
        errors.append(
            f"decision[{index}] privacy and boundary effects disagree"
        )
    return errors


def _global_promotion_contract_errors(
    parsed_output: Mapping[str, Any],
) -> list[str]:
    """Return closed root and decision-shape errors for one replacement."""

    errors: list[str] = []
    if set(parsed_output) != {"promotion_decisions"}:
        errors.append(
            "output must contain exactly promotion_decisions"
        )
        return errors
    raw_decisions = parsed_output.get("promotion_decisions")
    if not isinstance(raw_decisions, list):
        return ["promotion_decisions must be a list"]
    if len(raw_decisions) > 2:
        errors.append("promotion_decisions exceeds the daily limit")
    for index, raw_decision in enumerate(raw_decisions):
        if not isinstance(raw_decision, Mapping):
            errors.append(f"decision[{index}] must be an object")
            continue
        if set(raw_decision) != _PROMOTION_DECISION_KEYS:
            errors.append(f"decision[{index}] has an invalid key set")
            continue
        lane = raw_decision.get("lane")
        if lane not in PROMOTION_LANE_MEMORY_TYPE:
            errors.append(f"decision[{index}] has an invalid lane")
            continue
        action = raw_decision.get("decision")
        if action not in {
            "promote_new",
            "supersede",
            "merge",
            "reject",
            "no_action",
        }:
            errors.append(f"decision[{index}] has an invalid action")
        expected_memory_type = PROMOTION_LANE_MEMORY_TYPE[str(lane)]
        if raw_decision.get("memory_type") != expected_memory_type:
            errors.append(
                f"decision[{index}] memory_type must be "
                f"{expected_memory_type}"
            )
        for field_name in (
            "selected_candidate_id",
            "sanitized_memory_name",
            "sanitized_content",
            "authority",
            "signal_strength",
            "character_agreement",
        ):
            if not isinstance(raw_decision.get(field_name), str):
                errors.append(
                    f"decision[{index}] {field_name} must be text"
                )
        boundary_assessment = raw_decision.get("boundary_assessment")
        if not isinstance(boundary_assessment, Mapping):
            errors.append(
                f"decision[{index}] boundary_assessment must be an object"
            )
        else:
            errors.extend(
                _boundary_assessment_contract_errors(
                    index,
                    boundary_assessment,
                )
            )
        privacy_review = raw_decision.get("privacy_review")
        if not isinstance(privacy_review, Mapping):
            errors.append(
                f"decision[{index}] privacy_review must be an object"
            )
        else:
            errors.extend(
                _promoter_privacy_review_contract_errors(
                    index,
                    privacy_review,
                    boundary_assessment,
                )
            )
        if not isinstance(raw_decision.get("evidence_refs"), list):
            errors.append(
                f"decision[{index}] evidence_refs must be a list"
            )
    return errors


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
        promoter_privacy_errors = _promoter_privacy_review_contract_errors(
            index,
            privacy_review,
            decision.get("boundary_assessment"),
        )
        warnings.extend(promoter_privacy_errors)
        if promoter_privacy_errors:
            return_value = warnings
            return return_value
        if privacy_review.get("global_applicability") != "global":
            warnings.append(
                f"decision[{index}] global applicability blocks write"
            )
        if privacy_review.get("target_specific_meaning_removed") is not True:
            warnings.append(
                f"decision[{index}] target-specific meaning remains"
            )
        if privacy_review.get("affects_identity_or_boundaries") is not False:
            warnings.append(
                f"decision[{index}] identity or boundary effect blocks write"
            )
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
        if decision.get("review_admitted") is not True:
            warnings.append(
                f"independent review did not admit candidate: lane={lane}"
            )
            logger.info(
                "Reflection promotion reviewer blocked candidate: "
                f"lane={lane} review_decision={decision.get('review_decision')}"
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
    memory_unit_id = str(memory_doc["memory_unit_id"])
    if mutation_action == "supersede" and source_unit_ids == [memory_unit_id]:
        warning = f"replacement already active: memory_unit_id={memory_unit_id}"
        logger.info(
            "Reflection promotion skipped for active replacement replay: "
            f"lane={lane} memory_unit_id={memory_unit_id} "
            f"run_id={global_run_id}"
        )
        result = {
            "mutation": None,
            "warnings": [warning],
            "deferred": False,
            "defer_reason": "",
        }
        return result
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
    except ValueError as exc:
        if str(exc) == PROMOTION_DUPLICATE_REPLACEMENT_ERROR:
            warning = str(exc)
            logger.info(
                "Reflection promotion skipped for duplicate replacement id: "
                f"lane={lane} memory_unit_id={memory_unit_id} "
                f"run_id={global_run_id}"
            )
            result = {
                "mutation": None,
                "warnings": [warning],
                "deferred": False,
                "defer_reason": "",
            }
            return result
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
    if (
        (mutation_action == "supersede" and source_lineages)
        or (mutation_action == "merge" and len(source_lineages) == 1)
    ):
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
    source_privacy_notes = _compact_text_items(
        output.get("privacy_notes"),
        max_items=2,
        max_chars=48,
    )
    topic_summary = str(output.get("topic_summary", "") or "")
    cards: list[ReflectionEvidenceCard] = []
    scope = hourly_doc["scope"]
    source_run_id = str(hourly_doc["run_id"])
    if not utterances and not quality_items and not topic_summary:
        return cards
    lead_utterance = str(utterances[0]) if utterances else ""
    observation = topic_summary or " ".join(quality_items)
    captured_at_source = str(
        hourly_doc.get("created_at")
        or hourly_doc.get("hour_start")
        or hourly_doc.get("window_start")
        or "",
    )
    captured_at = format_storage_utc_for_llm(captured_at_source)
    card: ReflectionEvidenceCard = {
        "evidence_card_id": f"evidence_{source_run_id}",
        "source_reflection_run_ids": [source_run_id],
        "scope_ref": str(scope["scope_ref"]),
        "channel_type": str(scope["channel_type"]),
        "character_local_date": str(hourly_doc["character_local_date"]),
        "captured_at": captured_at,
        "active_character_utterance": _preview_text(lead_utterance, 180),
        "sanitized_observation": _preview_text(observation, 180),
        "supports": ["lore", "self_guidance"],
        "source_privacy_notes": source_privacy_notes,
        "private_detail_risk": _source_privacy_risk(output),
    }
    cards.append(_cap_serialized_card(card, PROMOTION_MAX_EVIDENCE_CARD_CHARS))
    return cards


def _source_privacy_risk(output: Mapping[str, Any]) -> str:
    """Return an explicit source risk or preserve an unresolved assessment."""

    risk = output.get("private_detail_risk")
    if risk in {"low", "medium", "high"}:
        return_value = str(risk)
        return return_value
    return_value = "unreviewed"
    return return_value


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
    """Bound declared readable fields while preserving envelope semantics.

    Identity, provenance, enum, and permission fields are never shortened. A
    caller-owned envelope that cannot fit through its readable fields fails
    closed instead of returning a card over the declared serialized cap.
    """

    if not isinstance(card, dict):
        raise TypeError("card must be a dictionary")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    capped = dict(card)
    while True:
        serialized_length = len(
            json.dumps(capped, ensure_ascii=False, sort_keys=True),
        )
        if serialized_length <= max_chars:
            return capped

        longest_key = ""
        longest_index = -1
        longest_value = ""
        for key, value in capped.items():
            if (
                isinstance(value, str)
                and key in _CARD_READABLE_TEXT_KEYS
                and len(value) > len(longest_value)
            ):
                longest_key = key
                longest_index = -1
                longest_value = value
            elif isinstance(value, list) and key in _CARD_READABLE_LIST_KEYS:
                for index, item in enumerate(value):
                    if isinstance(item, str) and len(item) > len(longest_value):
                        longest_key = key
                        longest_index = index
                        longest_value = item

        if not longest_key:
            raise ValueError(
                "card cannot fit within serialized cap without changing "
                "identity/provenance/enums",
            )

        target_chars = max(0, len(longest_value) - 20)
        if target_chars <= 3:
            bounded_value = ""
        else:
            bounded_value = _preview_text(longest_value, target_chars)
        if bounded_value == longest_value:
            bounded_value = ""
        if longest_index >= 0:
            bounded_items = list(capped[longest_key])
            bounded_items[longest_index] = bounded_value
            capped[longest_key] = bounded_items
        else:
            capped[longest_key] = bounded_value


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
