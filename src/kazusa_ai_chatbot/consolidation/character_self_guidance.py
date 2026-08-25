"""Accepted conversation-derived character self-guidance lane."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    CONSOLIDATION_LLM_MODEL,
    CONSOLIDATION_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.memory_evolution.identity import (
    deterministic_memory_unit_id,
)
from kazusa_ai_chatbot.memory_evolution.models import (
    MemoryAuthority,
    MemorySourceKind,
    MemoryStatus,
)
from kazusa_ai_chatbot.memory_evolution.repository import insert_memory_unit
from kazusa_ai_chatbot.rag.prompt_projection import project_tool_result_for_llm
from kazusa_ai_chatbot.utils import parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

SELF_GUIDANCE_MEMORY_TYPE = "defense_rule"
_MAX_MEMORY_NAME_CHARS = 80
_MAX_CONTENT_CHARS = 500
_SCOPE_CERTIFICATE_KEYS = frozenset({
    "global_applicability",
    "target_specific_meaning_removed",
    "affects_identity_or_boundaries",
    "private_detail_risk",
    "user_details_removed",
    "reason",
})
_SCOPE_APPLICABILITY_VALUES = frozenset({"global", "scoped", "absent"})
_SCOPE_PRIVATE_DETAIL_RISK_VALUES = frozenset({"low", "medium", "high"})
_SCOPE_REASON_MAX_CHARS = 500


class SelfGuidanceContractError(ValueError):
    """Represent malformed self-guidance model output at its owning stage."""


_SPECIALIST_PROMPT = '''\
你是负责持久化整理的角色自我指导专项处理器。

判断给定来源是否包含由当前角色负责、可长期使用的未来行为指导。该指导必须同时有
用户请求或提议作为依据，并有最终对话中的角色接受作为依据。

# 判断步骤
1. 阅读用户请求或提议、最终对话、近期聊天上下文和 source_refs。
2. 当已接受的行为普遍属于角色，而不是属于某个当前用户的义务时，选择 write。
3. 当缺少接受、行为只是暂时性的，或持久归属属于其他记忆通道时，选择 no_action。
4. 写出简洁、通用的指导。在保留来源依据的含义时，省略对未来行为没有必要的用户私密细节。

# 适用范围证书
移除来源用户、被称呼对象、关系对象和私密场景后，再评估是否写入。只有在已接受的行为
对角色与一般他人都仍然准确且合适时，才使用 `global`；含义依赖被移除的对象或上下文时，
使用 `scoped`；不存在可长期保存的已接受行为时，使用 `absent`。只有在复核后已接受的行为
不再依赖特定对象时，才将 `target_specific_meaning_removed` 设为 true；即使原本不存在
特定对象含义，也设为 true；只要被移除的对象仍然承载含义，就设为 false。候选改变角色
身份、权限、同意或边界时，将 `affects_identity_or_boundaries` 设为 true。根据持久化抽象
评估私密细节风险；只有不必要的用户细节已经去除时，才将 `user_details_removed` 设为 true。

# 输出格式
只返回合法 JSON：
{
  "action": "write | no_action",
  "memory_name": "已接受的角色行为指导的简短标题",
  "content": "由角色承担的通用未来行为指导",
  "global_applicability": "global | scoped | absent",
  "target_specific_meaning_removed": true,
  "affects_identity_or_boundaries": false,
  "private_detail_risk": "low | medium | high",
  "user_details_removed": true,
  "reason": "有边界的适用范围与隐私理由"
}
对于 `no_action`，`memory_name` 和 `content` 使用空字符串；每次响应都返回完整的适用范围证书。
'''

_REVIEW_PROMPT = '''\
你是一个角色自我指导候选项的审阅器。

审阅候选项是否是由当前角色负责、可长期使用的未来行为指导，并且是否有给定请求和最终
对话接受作为依据。

# 审阅步骤
1. 候选已有充分依据、具有普遍性且属于角色时，选择 accept。
2. 仅在保留已接受行为的前提下，调整措辞、标题或隐私敏感细节时，选择 revise。
3. 缺少接受依据、属于其他记忆通道，或加入给定来源不支持的行为时，选择 reject。
4. 移除来源用户、被称呼对象、关系对象和私密场景后，独立评估候选项。只有在已接受的行为
   对角色与一般他人都仍然准确且合适时，才保留 `global`。复核后已接受的行为不再依赖特定
   对象时，将 `target_specific_meaning_removed` 设为 true；即使原本不存在该含义，也设为
   true。结合持久化抽象评估身份、权限、同意和边界影响以及私密细节风险；根据此次独立审阅
   报告 `affects_identity_or_boundaries`、`private_detail_risk` 和 `user_details_removed`。

# 输出格式
只返回合法 JSON：
{
  "decision": "accept | revise | reject",
  "memory_name": "接受或修订后要持久化的标题",
  "content": "接受或修订后要持久化的内容",
  "global_applicability": "global | scoped | absent",
  "target_specific_meaning_removed": true,
  "affects_identity_or_boundaries": false,
  "private_detail_risk": "low | medium | high",
  "user_details_removed": true,
  "reason": "有边界的适用范围与隐私理由"
}
对于 `reject`，`memory_name` 和 `content` 使用空字符串；每次响应都返回完整的适用范围证书。
'''

_self_guidance_specialist_llm = LLInterface()
_self_guidance_reviewer_llm = LLInterface()
_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CONSOLIDATION_LLM",
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=0.1,
    top_p=0.9,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)


async def character_self_guidance_specialist(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract and review accepted character-owned self-guidance.

    Args:
        state: Consolidator state after lane source-policy acceptance.

    Returns:
        A state patch containing ``character_self_guidance`` when accepted.
    """

    source_refs = _source_refs_from_state(state)
    if not source_refs:
        return_value = {"character_self_guidance": {}}
        return return_value

    try:
        candidate = await _extract_self_guidance_candidate(state, source_refs)
    except SelfGuidanceContractError as exc:
        logger.warning(f"Self-guidance specialist contract rejected: {exc}")
        return_value = {"character_self_guidance": {}}
        return return_value
    if not candidate:
        return_value = {"character_self_guidance": {}}
        return return_value

    try:
        reviewed_candidate = await _review_self_guidance_candidate(
            state,
            source_refs,
            candidate,
        )
    except SelfGuidanceContractError as exc:
        logger.warning(f"Self-guidance reviewer contract rejected: {exc}")
        return_value = {"character_self_guidance": {}}
        return return_value
    if not _scope_certificate_admits_write(reviewed_candidate):
        return_value = {"character_self_guidance": {}}
        return return_value
    return_value = {"character_self_guidance": reviewed_candidate}
    return return_value


async def persist_character_self_guidance_from_state(
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Persist a reviewed character self-guidance candidate.

    Args:
        state: Consolidator state carrying a reviewed candidate and source refs.

    Returns:
        Stored memory mutation metadata, or ``None`` when no candidate exists.
    """

    candidate = state.get("character_self_guidance")
    if not isinstance(candidate, Mapping):
        return_value = None
        return return_value

    memory_name = text_or_empty(candidate.get("memory_name"))
    content = text_or_empty(candidate.get("content"))
    source_refs = _source_refs_from_state(state)
    reviewer_certificate = candidate.get("reviewer_scope_certificate")
    if (
        not memory_name
        or not content
        or not source_refs
        or not _scope_certificate_admits_write(candidate)
        or not isinstance(reviewer_certificate, Mapping)
    ):
        return_value = None
        return return_value

    storage_timestamp_utc = text_or_empty(state.get("storage_timestamp_utc"))
    memory_doc = _memory_document(
        memory_name=memory_name,
        content=content,
        source_refs=source_refs,
        storage_timestamp_utc=storage_timestamp_utc,
        reviewer_certificate=reviewer_certificate,
    )
    stored = await insert_memory_unit(document=memory_doc)
    result = {
        "memory_unit_id": stored["memory_unit_id"],
        "lineage_id": stored["lineage_id"],
        "memory_type": stored["memory_type"],
        "memory_name": stored["memory_name"],
        "content": stored["content"],
    }
    return result


async def _extract_self_guidance_candidate(
    state: Mapping[str, Any],
    source_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run the specialist LLM and validate its candidate shape."""

    payload = _prompt_payload(state, source_refs)
    response = await _self_guidance_specialist_llm.ainvoke(
        [
            SystemMessage(content=_SPECIALIST_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=_llm_config,
    )
    try:
        result = parse_llm_json_output(response.content)
        if not isinstance(result, Mapping):
            raise TypeError(
                "self-guidance specialist parsed output must be an object"
            )
        candidate = _normalize_specialist_result(result)
    except (TypeError, ValueError) as exc:
        raise SelfGuidanceContractError(
            f"self-guidance specialist output is structurally invalid: {exc}"
        ) from exc
    return candidate


async def _review_self_guidance_candidate(
    state: Mapping[str, Any],
    source_refs: list[dict[str, Any]],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the reviewer LLM and validate the accepted candidate shape."""

    payload = _prompt_payload(state, source_refs)
    payload["candidate"] = project_tool_result_for_llm({
        "memory_name": candidate["memory_name"],
        "content": candidate["content"],
        "memory_type": candidate["memory_type"],
    })
    response = await _self_guidance_reviewer_llm.ainvoke(
        [
            SystemMessage(content=_REVIEW_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=_llm_config,
    )
    try:
        result = parse_llm_json_output(response.content)
        if not isinstance(result, Mapping):
            raise TypeError(
                "self-guidance reviewer parsed output must be an object"
            )
        reviewed_candidate = _normalize_reviewer_result(result, candidate)
    except (TypeError, ValueError) as exc:
        raise SelfGuidanceContractError(
            f"self-guidance reviewer output is structurally invalid: {exc}"
        ) from exc
    return reviewed_candidate


def _prompt_payload(
    state: Mapping[str, Any],
    source_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the prompt-safe payload for specialist and reviewer LLMs."""

    payload = {
        "timestamp": state.get("local_time_context", {}),
        "character_name": _character_name(state),
        "decontextualized_input": text_or_empty(
            state.get("decontextualized_input")
        ),
        "final_dialog": project_tool_result_for_llm(
            state.get("final_dialog", [])
        ),
        "chat_history_recent": project_tool_result_for_llm(
            state.get("chat_history_recent", [])
        ),
        "source_refs": project_tool_result_for_llm(source_refs),
    }
    return payload


def _normalize_specialist_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate specialist output without semantic post-classification."""

    action = text_or_empty(result.get("action"))
    if action == "no_action":
        _require_exact_keys(
            result,
            {"action", "memory_name", "content"} | _SCOPE_CERTIFICATE_KEYS,
            stage="self-guidance specialist",
        )
        _normalize_scope_certificate(result)
        return_value: dict[str, Any] = {}
        return return_value
    if action != "write":
        raise ValueError(f"invalid self-guidance action: {action!r}")

    _require_exact_keys(
        result,
        {"action", "memory_name", "content"} | _SCOPE_CERTIFICATE_KEYS,
        stage="self-guidance specialist",
    )
    scope_certificate = _normalize_scope_certificate(result)
    candidate = _candidate_fields(result)
    candidate["specialist_scope_certificate"] = scope_certificate
    return candidate


def _normalize_reviewer_result(
    result: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate reviewer output without semantic post-classification."""

    decision = text_or_empty(result.get("decision"))
    if decision == "reject":
        _require_exact_keys(
            result,
            {"decision", "memory_name", "content"} | _SCOPE_CERTIFICATE_KEYS,
            stage="self-guidance reviewer",
        )
        _normalize_scope_certificate(result)
        return_value: dict[str, Any] = {}
        return return_value
    if decision not in {"accept", "revise"}:
        raise ValueError(f"invalid self-guidance review decision: {decision!r}")

    _require_exact_keys(
        result,
        {"decision", "memory_name", "content"} | _SCOPE_CERTIFICATE_KEYS,
        stage="self-guidance reviewer",
    )
    reviewer_certificate = _normalize_scope_certificate(result)
    if decision == "accept":
        memory_name = text_or_empty(result.get("memory_name")) or text_or_empty(
            candidate.get("memory_name")
        )
        content = text_or_empty(result.get("content")) or text_or_empty(
            candidate.get("content")
        )
        result = {
            **dict(result),
            "memory_name": memory_name,
            "content": content,
        }

    reviewed_candidate = _candidate_fields(result)
    specialist_certificate = candidate.get("specialist_scope_certificate")
    if not isinstance(specialist_certificate, Mapping):
        raise TypeError("self-guidance specialist scope certificate is required")
    reviewed_candidate["specialist_scope_certificate"] = dict(
        specialist_certificate
    )
    reviewed_candidate["reviewer_scope_certificate"] = reviewer_certificate
    return reviewed_candidate


def _candidate_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    """Read and structurally validate candidate text fields."""

    memory_name = text_or_empty(result.get("memory_name"))
    content = text_or_empty(result.get("content"))
    if not memory_name:
        raise ValueError("self-guidance memory_name is required")
    if not content:
        raise ValueError("self-guidance content is required")

    candidate = {
        "memory_name": memory_name[:_MAX_MEMORY_NAME_CHARS].strip(),
        "content": content[:_MAX_CONTENT_CHARS].strip(),
        "memory_type": SELF_GUIDANCE_MEMORY_TYPE,
    }
    return candidate


def _require_exact_keys(
    result: Mapping[str, Any],
    expected_keys: set[str],
    *,
    stage: str,
) -> None:
    """Require one self-guidance model response to use its closed shape."""

    if set(result) != expected_keys:
        raise ValueError(f"{stage} output keys are invalid")


def _normalize_scope_certificate(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and project one model-authored scope/privacy certificate."""

    global_applicability = text_or_empty(
        result.get("global_applicability")
    )
    if global_applicability not in _SCOPE_APPLICABILITY_VALUES:
        raise ValueError("invalid self-guidance global_applicability")

    target_specific_meaning_removed = result.get(
        "target_specific_meaning_removed"
    )
    if not isinstance(target_specific_meaning_removed, bool):
        raise TypeError("self-guidance target removal flag must be boolean")

    affects_identity_or_boundaries = result.get(
        "affects_identity_or_boundaries"
    )
    if not isinstance(affects_identity_or_boundaries, bool):
        raise TypeError("self-guidance identity boundary flag must be boolean")

    private_detail_risk = text_or_empty(result.get("private_detail_risk"))
    if private_detail_risk not in _SCOPE_PRIVATE_DETAIL_RISK_VALUES:
        raise ValueError("invalid self-guidance private_detail_risk")

    user_details_removed = result.get("user_details_removed")
    if not isinstance(user_details_removed, bool):
        raise TypeError("self-guidance user detail flag must be boolean")

    reason = text_or_empty(result.get("reason"))
    if not reason or len(reason) > _SCOPE_REASON_MAX_CHARS:
        raise ValueError("self-guidance scope reason is invalid")

    certificate = {
        "global_applicability": global_applicability,
        "target_specific_meaning_removed": target_specific_meaning_removed,
        "affects_identity_or_boundaries": affects_identity_or_boundaries,
        "private_detail_risk": private_detail_risk,
        "user_details_removed": user_details_removed,
        "reason": reason,
    }
    return certificate


def _scope_certificate_is_valid(value: object) -> bool:
    """Return whether a stored candidate contains one exact certificate."""

    if not isinstance(value, Mapping):
        return False
    if set(value) != _SCOPE_CERTIFICATE_KEYS:
        return False
    try:
        _normalize_scope_certificate(value)
    except (TypeError, ValueError):
        return False
    return True


def _scope_certificate_admits_write(
    candidate: Mapping[str, Any],
) -> bool:
    """Require independent global and privacy-safe certificates for writing."""

    specialist_certificate = candidate.get("specialist_scope_certificate")
    reviewer_certificate = candidate.get("reviewer_scope_certificate")
    if not _scope_certificate_is_valid(specialist_certificate):
        return False
    if not _scope_certificate_is_valid(reviewer_certificate):
        return False

    specialist = specialist_certificate
    reviewer = reviewer_certificate
    return_value = (
        specialist["global_applicability"] == "global"
        and reviewer["global_applicability"] == "global"
        and specialist["target_specific_meaning_removed"] is True
        and reviewer["target_specific_meaning_removed"] is True
        and specialist["affects_identity_or_boundaries"] is False
        and reviewer["affects_identity_or_boundaries"] is False
        and reviewer["private_detail_risk"] == "low"
        and reviewer["user_details_removed"] is True
    )
    return return_value


def _source_refs_from_state(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return structurally valid self-guidance source refs from state."""

    raw_refs = state.get("character_self_guidance_source_refs")
    if not isinstance(raw_refs, list):
        return_value: list[dict[str, Any]] = []
        return return_value
    source_refs = [dict(ref) for ref in raw_refs if isinstance(ref, Mapping)]
    return source_refs


def _character_name(state: Mapping[str, Any]) -> str:
    """Return the active character display name for prompt context."""

    character_profile = state.get("character_profile")
    if isinstance(character_profile, Mapping):
        character_name = text_or_empty(character_profile.get("name"))
        if character_name:
            return character_name
    return_value = "the active character"
    return return_value


def _memory_document(
    *,
    memory_name: str,
    content: str,
    source_refs: list[dict[str, Any]],
    storage_timestamp_utc: str,
    reviewer_certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one evolving memory document for accepted self-guidance."""

    source_key = json.dumps(
        source_refs,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    memory_unit_id = deterministic_memory_unit_id(
        "conversation",
        [SELF_GUIDANCE_MEMORY_TYPE, memory_name, content, source_key],
    )
    memory_doc = {
        "memory_unit_id": memory_unit_id,
        "lineage_id": memory_unit_id,
        "version": 1,
        "memory_name": memory_name,
        "content": content,
        "source_global_user_id": "",
        "memory_type": SELF_GUIDANCE_MEMORY_TYPE,
        "source_kind": MemorySourceKind.CONVERSATION_EXTRACTED,
        "authority": MemoryAuthority.CONVERSATION_ACCEPTED,
        "status": MemoryStatus.ACTIVE,
        "supersedes_memory_unit_ids": [],
        "merged_from_memory_unit_ids": [],
        "evidence_refs": source_refs,
        "privacy_review": {
            "global_applicability": reviewer_certificate[
                "global_applicability"
            ],
            "target_specific_meaning_removed": reviewer_certificate[
                "target_specific_meaning_removed"
            ],
            "affects_identity_or_boundaries": reviewer_certificate[
                "affects_identity_or_boundaries"
            ],
            "private_detail_risk": reviewer_certificate["private_detail_risk"],
            "user_details_removed": reviewer_certificate["user_details_removed"],
            "boundary_assessment": reviewer_certificate["reason"],
            "reviewer": "automated_llm",
        },
        "confidence_note": (
            "Accepted by the character in final dialog and reviewed for "
            "character-owned self-guidance."
        ),
        "timestamp": storage_timestamp_utc,
    }
    return memory_doc
