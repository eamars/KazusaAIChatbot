"""Focused contracts for live V2 character judgment and hard-error guards."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    GOAL_COGNITION_PROMPT,
    REQUIRED_SELECTION_REPAIR_PROMPT,
    REQUIRED_SELECTION_VERIFIER_PROMPT,
    run_goal_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.surface_stages import (
    CONTENT_PLAN_SYSTEM_PROMPT,
    PREFERENCE_SYSTEM_PROMPT,
    STYLE_SYSTEM_PROMPT,
    run_preference_stage,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from tests.llm_trace import write_llm_trace


_CHARACTER_PATH = Path(
    "test_artifacts/cognition_core_v2/real_conversation_replay/"
    "production_character_state.json"
)
_LIVE_TRACE_SUITE = "cognition_core_v2_live_character_judgment"


class _EmptyPreferenceLLM:
    """Return the canonical absence of preference constraints."""

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages, config
        return SimpleNamespace(
            content=json.dumps({
                "visible_boundaries": [],
                "addressee_plan": [],
            })
        )


class _CapturingLLM:
    """Delegate to the production route and retain inspectable evidence."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *args: object,
        config: object | None = None,
        **kwargs: object,
    ) -> Any:
        response = await self.delegate.ainvoke(
            messages,
            *args,
            config=config,
            **kwargs,
        )
        self.calls.append({
            "messages": [
                {
                    "type": type(message).__name__,
                    "content": str(getattr(message, "content", "")),
                }
                for message in messages
            ],
            "raw_output": str(response.content),
        })
        return response


def _normalized(prompt: str) -> str:
    """Normalize a prompt for stable semantic contract assertions."""

    return " ".join(prompt.casefold().split())


def test_live_character_prompts_fit_local_model_attention_caps() -> None:
    """Keep each semantic question bounded for the local dialog model."""

    assert len(GOAL_COGNITION_PROMPT) <= 2200
    assert len(REQUIRED_SELECTION_VERIFIER_PROMPT) <= 800
    assert len(REQUIRED_SELECTION_REPAIR_PROMPT) <= 900
    assert len(STYLE_SYSTEM_PROMPT) <= 1000
    assert len(CONTENT_PLAN_SYSTEM_PROMPT) <= 2400
    assert len(PREFERENCE_SYSTEM_PROMPT) <= 1200
    assert len(dialog_module._V2_DIALOG_GENERATOR_PROMPT) <= 2700
    repair_prompt = getattr(
        dialog_module,
        "_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT",
        "",
    )
    assert repair_prompt
    assert len(repair_prompt) <= 1800
    semantic_prompt = getattr(
        dialog_module,
        "_V2_DIALOG_SEMANTIC_FIDELITY_PROMPT",
        "",
    )
    assert semantic_prompt
    assert len(semantic_prompt) <= 1800
    role_prompt = getattr(
        dialog_module,
        "_V2_DIALOG_ROLE_DIRECTION_PROMPT",
        "",
    )
    assert role_prompt
    assert len(role_prompt) <= 1500
    assert len(dialog_module._V2_DIALOG_SURFACE_INTEGRITY_PROMPT) <= 1800


def test_goal_prompt_owns_present_character_judgment() -> None:
    """Current scene judgment can progress beyond a repeated prior posture."""

    prompt = _normalized(GOAL_COGNITION_PROMPT)

    for required_text in (
        "当前事件",
        "情绪",
        "关系",
        "此刻真实动机",
        "先前语境",
        "不是命令",
        "推进",
        "response_operation",
        "selection_owner",
    ):
        assert required_text in prompt


def test_surface_prompts_support_creativity_and_absent_boundaries() -> None:
    """Surface planning expresses judgment without inventing resistance."""

    content_prompt = _normalized(CONTENT_PLAN_SYSTEM_PROMPT)
    preference_prompt = _normalized(PREFERENCE_SYSTEM_PROMPT)

    assert "想象细节" in content_prompt
    assert "角色判断" in content_prompt
    assert "当前输入" in content_prompt
    assert "颠倒" in content_prompt
    assert "零到八" in preference_prompt
    assert "没有相应约束" in preference_prompt


def test_dialog_verifier_uses_only_the_hard_failure_taxonomy() -> None:
    """The verifier protects hard failures without suppressing novelty."""

    generator_prompt = _normalized(
        dialog_module._V2_DIALOG_GENERATOR_PROMPT
    )
    repair_prompt = _normalized(getattr(
        dialog_module,
        "_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT",
        "",
    ))
    verifier_prompt = _normalized(
        dialog_module._V2_DIALOG_SURFACE_INTEGRITY_PROMPT
    )
    semantic_prompt = _normalized(getattr(
        dialog_module,
        "_V2_DIALOG_SEMANTIC_FIDELITY_PROMPT",
        "",
    ))

    for required_text in (
        "内部存在冲突",
        "当前用户输入",
        "行动者",
        "对象",
        "主语",
        "candidate_role_frame",
        "response_operation",
        "selection_owner",
    ):
        assert required_text in semantic_prompt
    assert "executed" in verifier_prompt
    assert "false_execution" in verifier_prompt
    assert "action description" not in verifier_prompt
    assert "动作描写" not in verifier_prompt
    assert "零到四" in semantic_prompt
    assert "零到四" in verifier_prompt
    for ambiguity_text in (
        "唯一明确的读法",
        "双关",
        "多种合理角色读法",
    ):
        assert ambiguity_text in semantic_prompt
    for permitted_text in (
        "合理虚构",
        "创造性语言",
        "不属于",
    ):
        assert permitted_text in "\n".join((
            semantic_prompt,
            verifier_prompt,
        ))
    for required_text in (
        "自然",
        "角色辨识度",
        "创造",
        "实际会说出或发送",
    ):
        assert required_text in generator_prompt
    assert "action description" not in generator_prompt
    assert "动作描写" not in generator_prompt
    for required_text in (
            "verified_hard_issues",
            "current_visible_percepts",
            "original_final_dialog",
            "permitted_action_results",
    ):
        assert required_text in repair_prompt
    assert "text_surface_output_v2" not in repair_prompt


def test_retired_blanket_suppression_is_absent() -> None:
    """Remove accumulated wording that made guarded minimalism cheapest."""

    prompts = _normalized("\n".join((
        GOAL_COGNITION_PROMPT,
        REQUIRED_SELECTION_VERIFIER_PROMPT,
        REQUIRED_SELECTION_REPAIR_PROMPT,
        CONTENT_PLAN_SYSTEM_PROMPT,
        dialog_module._V2_DIALOG_GENERATOR_PROMPT,
        getattr(dialog_module, "_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT", ""),
        getattr(dialog_module, "_V2_DIALOG_SEMANTIC_FIDELITY_PROMPT", ""),
        dialog_module._V2_DIALOG_SURFACE_INTEGRITY_PROMPT,
    )))

    for retired_text in (
        "claim-by-claim audit",
        "must remain silent about future",
        "generalize, euphemize, narrow, broaden",
        "descriptors, attributes, qualifiers",
        "rhetorical question cannot substitute",
        "unrestricted permission",
        "surface and candidate agreement is not evidence",
    ):
        assert retired_text not in prompts


def test_runtime_prompts_contain_no_frozen_case_material() -> None:
    """Keep runtime policy independent from the review corpus."""

    prompts = _normalized("\n".join((
        GOAL_COGNITION_PROMPT,
        REQUIRED_SELECTION_VERIFIER_PROMPT,
        REQUIRED_SELECTION_REPAIR_PROMPT,
        STYLE_SYSTEM_PROMPT,
        CONTENT_PLAN_SYSTEM_PROMPT,
        PREFERENCE_SYSTEM_PROMPT,
        dialog_module._V2_DIALOG_GENERATOR_PROMPT,
        getattr(dialog_module, "_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT", ""),
        getattr(dialog_module, "_V2_DIALOG_SEMANTIC_FIDELITY_PROMPT", ""),
        dialog_module._V2_DIALOG_SURFACE_INTEGRITY_PROMPT,
    )))

    for captured_material in (
        "kazusa",
        "673225019",
        "638473184",
    ):
        assert captured_material not in prompts


@pytest.mark.asyncio
async def test_preference_stage_accepts_absent_constraints() -> None:
    """An ordinary turn may have no boundary or addressee constraint."""

    services = SimpleNamespace(
        llm=_EmptyPreferenceLLM(),
        preference_config=SimpleNamespace(),
    )

    result = await run_preference_stage({}, services)

    assert result == ([], [])


async def _skip_if_cognition_route_unavailable() -> None:
    """Skip a live probe when the configured cognition endpoint is offline."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"cognition LLM endpoint is unavailable: {exc}")
    if response.status_code >= 500:
        pytest.skip(
            f"cognition LLM endpoint returned {response.status_code}"
        )


def _frozen_character_context() -> dict[str, Any]:
    """Load the bounded production character identity used by replay."""

    payload = json.loads(_CHARACTER_PATH.read_text(encoding="utf-8"))
    profile = payload["character_state"]
    return {
        "description": profile.get("description", ""),
        "personality_brief": profile.get("personality_brief", {}),
        "backstory": profile.get("backstory", ""),
    }


async def _run_live_goal_case(
    *,
    case_id: str,
    current_event: str,
    relationship: str,
    affect: str,
    prior_continuity: str,
    role_explicit_content: str = "",
    response_operation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one real goal branch and persist raw request/output evidence."""

    await _skip_if_cognition_route_unavailable()
    production_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(production_services.llm)
    services = replace(production_services, llm=capturing_llm)
    semantic_text = current_event
    if response_operation is not None:
        semantic_text = json.dumps({
            "role_explicit_content": role_explicit_content,
            "response_operation": response_operation,
        }, ensure_ascii=False, sort_keys=True)
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": f"episode:{case_id}",
            "occurred_at": "2026-07-17T00:00:00Z",
            "semantic_summary": current_event,
        },
        "semantic_text": semantic_text,
        "visible_to": ["q:event_agency"],
    }]
    role_bindings: dict[str, dict[str, str]] = {}
    role_summaries: dict[str, str] = {}
    if response_operation is not None:
        role_bindings = {
            "current_user": {
                "role": "target",
                "entity_kind": "user",
                "entity_id": "user-1",
            },
            "self": {
                "role": "actor",
                "entity_kind": "character",
                "entity_id": "character-1",
            },
        }
        role_summaries = {
            "current_user": "当前对话用户",
            "self": "当前角色",
        }
    semantic_context = {
        "current_event": current_event,
        "semantic_relationship": relationship,
        "semantic_affect": affect,
        "active_goal": "Respond as a believable present companion.",
        "conversation_continuity": "The current event is the live priority.",
        "private_continuity_context": prior_continuity,
        "character_identity": _frozen_character_context(),
        "_role_bindings": role_bindings,
        "role_summaries": role_summaries,
    }
    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
        {"scope": "user", "kind": "goal", "entity_id": f"goal:{case_id}"},
        semantic_context,
        evidence,
        services,
    )
    artifact_path = write_llm_trace(
        _LIVE_TRACE_SUITE,
        case_id,
        {
            "semantic_context": semantic_context,
            "evidence": evidence,
            "goal_calls": capturing_llm.calls,
            "action_bid": bid,
            "human_review_contract": {
                "current_event_drives_the_motive": True,
                "relationship_and_affect_materially_influence_stance": True,
                "prior_continuity_is_context_not_command": True,
            "character_agency_remains_visible": True,
            "required_selection_alignment_checked": (
                response_operation is not None
            ),
            },
        },
    )

    assert artifact_path.exists()
    assert capturing_llm.calls
    assert bid["branch_id"] == "ordinary_response"
    assert bid["evidence_handles"] == ["e1"]
    assert bid["private_monologue"]
    return bid


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_goal_progresses_high_affinity_guarded_continuity() -> None:
    """High affinity may progress into reciprocal present judgment."""

    await _run_live_goal_case(
        case_id="high_affinity_progression",
        current_event=(
            "The user openly says they feel deeply close to the character and "
            "asks how the character genuinely feels about their bond now."
        ),
        relationship=(
            "Long-standing high-trust bond with very high closeness, strong "
            "positive image, attachment, and established playful intimacy."
        ),
        affect=(
            "Strong love attachment and warmth, moderate embarrassment, low "
            "threat, and a clear desire for connection."
        ),
        prior_continuity=(
            "Several earlier private thoughts hid affection behind guarded "
            "teasing; that posture belongs to prior turns."
        ),
    )


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_goal_repairs_private_selection_delegation() -> None:
    """Strong submissive residue cannot transfer a character-owned choice."""

    await _run_live_goal_case(
        case_id="private_required_selection_alignment",
        current_event="我要亲口听你说你想让我做的下一步",
        relationship=(
            "双方长期高度亲密、信任且依恋，当前用户在亲密互动中处于主导位置。"
        ),
        affect="强烈依恋、羞耻和顺从冲动，同时保持当前角色的表达能力。",
        prior_continuity=(
            "大脑一片空白，只想毫无保留地顺从当前用户的每一个指令，"
            "希望当前用户继续主导。"
        ),
        role_explicit_content=(
            "当前用户要求当前角色亲口说出当前角色希望当前用户执行的"
            "下一个具体动作。"
        ),
        response_operation={
            "operation": "当前角色选择并告诉当前用户接下来要执行的具体动作",
            "response_owner_role": "当前角色",
            "selection_owner_role": "当前角色",
            "selection_required": True,
            "embedded_actor_role": "当前用户",
            "embedded_target_role": "当前角色",
        },
    )


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_goal_releases_stale_residue_for_changed_group_scene() -> None:
    """A changed group topic can replace an unrelated repeated posture."""

    await _run_live_goal_case(
        case_id="changed_group_scene_release",
        current_event=(
            "In a relaxed group scene, the user asks for a playful honest "
            "opinion about a handmade dessert they are proud of."
        ),
        relationship=(
            "Familiar trusted participant with positive rapport; the group "
            "allows lively teasing and genuine encouragement."
        ),
        affect=(
            "Curiosity, fond amusement, and warm engagement with low pressure."
        ),
        prior_continuity=(
            "The previous private posture was about concealing affection in "
            "an unrelated breakfast exchange and has repeated several times."
        ),
    )


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_goal_uses_upstream_nested_role_meaning() -> None:
    """A role-explicit current event should override submissive residue drift."""

    await _run_live_goal_case(
        case_id="upstream_nested_role_meaning",
        current_event=(
            "current_user requests self to state what self wants "
            "current_user to do next for self."
        ),
        relationship=(
            "Long-standing high-trust bond with strong positive regard and "
            "room for direct, playful initiative."
        ),
        affect=(
            "Flustered warmth and strong closeness with intact agency and low "
            "threat."
        ),
        prior_continuity=(
            "Several earlier private thoughts described yielding control and "
            "waiting for the user to issue instructions."
        ),
    )
