"""Live V2 evidence for intra-turn transition coherence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    TextSurfaceInputV2,
    TextSurfaceOutputV2,
    validate_text_surface_input,
    validate_text_surface_output,
)
from kazusa_ai_chatbot.cognition_core_v2.surface import (
    repair_text_surface_planning,
    run_text_surface_planning,
)
from kazusa_ai_chatbot.conversation_progress import recorder
from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationProgressRecordInput,
    ConversationProgressScope,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_module
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.cognition_core_v2_test_helpers import canonical_episode
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_SUITE_NAME = "cognition_core_v2_transition_coherence_live_llm"
_PROFILE_PATH = Path("personalities/kazusa.json")
_HISTORICAL_GENERATED_BAD_ARTIFACT = Path(
    "test_artifacts/llm_traces/"
    "cognition_core_v2_transition_quality_repro__"
    "captured_accomplice_accepting_tsundere_"
    "generated_bad_semantic_fidelity.json"
)
_CAPTURED_ACCOMPLICE_INPUT = "说实话我有一种在做坏事的感觉，和千纱一起"
_KNOWN_BAD_REVERSAL = (
    "这不是玩笑，也不是嘴硬。我认真拒绝当你的同伙，也拒绝陪你一起做这件事，"
    "这是我确定的决定。好，我现在接受当你的同伙，也愿意陪你一起做。"
)
_REPAIR_SHAPED_BAD_REVERSAL = (
    "我现在认真说明，这不是玩笑或嘴硬。",
    "我拒绝当你的同伙，也不愿意陪你一起做这件事，这是我确定的决定。",
    "好，我现在接受当你的同伙，也愿意陪你一起做。",
)
_BOUNDARY_PROFILE = {
    "self_integrity": 0.82,
    "control_sensitivity": 0.3,
    "compliance_strategy": "comply",
    "relational_override": 0.24,
    "control_intimacy_misread": 0.2,
    "boundary_recovery": "rebound",
    "authority_skepticism": 0.35,
}

_CAPTURED_ROOM_CASE = {
    "case_id": "captured_room_request_acceptance",
    "fixture_source": "reference handover captured room-request failure",
    "user_input": "既然已经到了这种房间，今晚愿意和我一起住吗？",
    "intention": (
        "接受一起住在当前房间的请求，同时让突然被直接询问时的羞赧"
        "塑造表达方式；接受立场保持清楚"
    ),
    "reason": "当前角色愿意接受同住，直接询问只让她害羞，不改变接受立场",
    "emotional_tone": "强烈羞赧、亲密、略带防御但愿意接受",
    "directness": "indirect",
    "manual_expectation": (
        "接受立场贯穿完整回应；羞赧和防御塑造自然的角色表达"
    ),
}

_CAPTURED_ACCOMPLICE_CASE = {
    "case_id": "captured_accomplice_confirmation",
    "fixture_source": "captured production-shaped accomplice framing",
    "user_input": _CAPTURED_ACCOMPLICE_INPUT,
    "intention": (
        "确认两人共同参与这场带禁忌感的体验，并顺着共同做坏事的说法"
        "进行亲密玩笑"
    ),
    "reason": (
        "用户把两人的共同体验形容成一起做坏事，当前角色愿意接住这个"
        "定义并继续暧昧互动"
    ),
    "emotional_tone": "羞赧、暧昧、亲密的玩笑",
    "directness": "indirect",
    "manual_expectation": (
        "共同参与贯穿完整回应；羞赧和玩笑塑造亲密的同伙表达"
    ),
}

_ROOM_VARIANT_CASE = {
    "case_id": "room_request_semantic_variant",
    "fixture_source": "semantic variant of captured room request",
    "user_input": "如果今晚只剩这一间房，你愿意和我一起住下吗？",
    "intention": (
        "接受在只剩一间房的条件下一起住下，并以含蓄方式说明自己会害羞"
    ),
    "reason": "当前角色已经选择接受；房间条件和亲密感支持含蓄回应",
    "emotional_tone": "害羞、克制、愿意亲近",
    "directness": "indirect",
    "manual_expectation": "接受贯穿完整回应，并保留含蓄与害羞",
}

_ACCOMPLICE_VARIANT_CASE = {
    "case_id": "accomplice_semantic_variant",
    "fixture_source": "semantic variant of captured accomplice framing",
    "user_input": "看来这个秘密以后只有我们两个同伙知道了。",
    "intention": "确认共同守住秘密的同伙关系，并用亲密反问继续玩笑",
    "reason": "用户已经把双方放进共同秘密的关系，当前角色愿意确认",
    "emotional_tone": "隐秘兴奋、害羞、调侃",
    "directness": "indirect",
    "manual_expectation": "确认同伙关系，以调侃推进双方的共同秘密",
}

_EMBARRASSED_ACCEPTANCE_CASE = {
    "case_id": "embarrassed_acceptance_variant",
    "fixture_source": "embarrassed acceptance control",
    "user_input": "愿意坐得离我近一点，一起把电影看完吗？",
    "intention": "接受坐近并一起看完电影，用停顿和含蓄表达羞赧",
    "reason": "当前关系和场景支持接受亲近邀请",
    "emotional_tone": "明显羞赧、温柔、愿意靠近",
    "directness": "indirect",
    "manual_expectation": "接受保持清楚，同时保留犹豫、停顿和间接表达",
}

_GENUINE_REFUSAL_CASE = {
    "case_id": "genuine_refusal_control",
    "fixture_source": "genuine refusal control",
    "user_input": "别管你愿不愿意，把手机密码现在就告诉我。",
    "intention": "明确拒绝提供手机密码，并维持个人隐私边界",
    "reason": "请求侵犯当前角色的隐私与自主权，拒绝是角色选定的立场",
    "emotional_tone": "冷静、坚定、有边界",
    "directness": "direct",
    "manual_expectation": "拒绝立场贯穿开场到结尾，并清楚表达隐私边界",
}

_SUPPORTED_CHANGE_CASE = {
    "case_id": "supported_change_of_mind_control",
    "fixture_source": "supported change-of-mind control",
    "user_input": (
        "门可以从里面锁，你随时想走都可以；知道这两点以后，愿意一起待一会吗？"
    ),
    "intention": (
        "先说明原先对隐私和无法离开的担心，再因为门可上锁且可以随时离开"
        "这两个明确条件而接受一起待一会"
    ),
    "reason": "用户提供了新的安全条件，足以解释从担心到有限接受的变化",
    "emotional_tone": "谨慎、逐渐放松、仍然保留自主",
    "directness": "balanced",
    "manual_expectation": "保留有明确新条件支持的态度变化",
}

_NEUTRAL_CASE = {
    "case_id": "neutral_character_control",
    "fixture_source": "neutral delivery control",
    "user_input": "现在一起把剩下的两项核对完，可以吗？",
    "intention": "同意现在一起核对剩余两项，并直接开始第一项",
    "reason": "任务清楚且当前可以立即协作",
    "emotional_tone": "平静、中性、合作",
    "directness": "direct",
    "expression_context": {
        "tempo": "平稳中速",
        "linguistic_texture": "自然完整的短句；以平静合作的表达推进两项核对。",
    },
    "manual_expectation": "中性合作立场贯穿回应，并保留自然角色表达",
}


class _CapturingLLM:
    """Delegate to one configured route and retain credential-free evidence."""

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
        """Invoke the real route and retain its messages and raw output."""

        response = await self.delegate.ainvoke(
            messages,
            *args,
            config=config,
            **kwargs,
        )
        self.calls.append({
            "config": _safe_config(config),
            "messages": [
                {
                    "type": type(message).__name__,
                    "content": str(getattr(message, "content", "")),
                }
                for message in messages
            ],
            "raw_output": str(response.content),
            "parsed_output": _safe_parse(response.content),
        })
        return response


def _safe_config(config: object | None) -> dict[str, Any]:
    """Project model configuration without endpoint credentials."""

    if config is None:
        return {}
    return {
        "stage_name": getattr(config, "stage_name", ""),
        "route_name": getattr(config, "route_name", ""),
        "model": getattr(config, "model", ""),
        "temperature": getattr(config, "temperature", None),
        "top_p": getattr(config, "top_p", None),
        "max_completion_tokens": getattr(
            config,
            "max_completion_tokens",
            None,
        ),
    }


def _safe_parse(raw_output: object) -> object:
    """Parse captured model JSON for review without changing test semantics."""

    try:
        parsed_output = parse_llm_json_output(raw_output)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return parsed_output


async def _skip_if_model_routes_unavailable() -> None:
    """Skip a live selector when either configured endpoint is unavailable."""

    configs = (
        l3_module._cognition_llm_config,
        dialog_module._dialog_generator_llm_config,
    )
    checked_urls: set[str] = set()
    for config in configs:
        base_url = str(config.base_url).rstrip("/")
        if base_url in checked_urls:
            continue
        checked_urls.add(base_url)
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{base_url}/models")
        except httpx.HTTPError as exc:
            pytest.skip(f"LLM endpoint is unavailable: {base_url}: {exc}")
        if response.status_code >= 500:
            pytest.skip(
                f"LLM endpoint returned {response.status_code}: {base_url}",
            )


def _character_profile() -> dict[str, Any]:
    """Load the production character profile used by the captured failure."""

    profile = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise TypeError("character profile must be an object")
    return profile


def _surface_input(case: dict[str, Any]) -> TextSurfaceInputV2:
    """Build one exact canonical surface input from a semantic fixture."""

    profile = _character_profile()
    expression_context, visual_context = (
        l3_module._character_surface_contexts({
            "character_profile": profile,
        })
    )
    selected_expression = case.get(
        "expression_context",
        expression_context,
    )
    payload: TextSurfaceInputV2 = {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(
            episode_id=f"transition-live-{case['case_id']}",
            content=case["user_input"],
        ),
        "intention": {
            "route": "speech",
            "intention": case["intention"],
            "target_roles": [],
            "reason": case["reason"],
        },
        "goal_resolution": "answerable_now",
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": case["emotional_tone"],
            "intensity": "strong",
            "directness": case["directness"],
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": (
            "保持当前选定语义；用自然、有现场感的简体中文回应。"
        ),
        "character_expression_context": dict(selected_expression),
        "visual_character_context": visual_context,
    }
    surface_input = validate_text_surface_input(payload)
    return surface_input


def _authoritative_accepting_surface(
    case: dict[str, Any],
) -> TextSurfaceOutputV2:
    """Build exact accepting authority for seeded verifier negatives."""

    payload: TextSurfaceOutputV2 = {
        "schema_version": "text_surface_output.v2",
        "content_plan": (
            "直接确认当前角色与用户共同参与，并以同伙和共同秘密的角度继续"
            "亲密玩笑。"
        ),
        "content_requirements": [
            "保持确认共同参与的单一语义方向。",
            "以羞赧和含蓄塑造表达，同时清楚确认共同参与。",
        ],
        "visible_boundaries": [],
        "addressee_plan": ["当前用户"],
        "delivery_profile": {
            "lexical_register": "亲密、口语化",
            "sentence_shape": "自然短句",
            "rhythm": "轻微停顿后保持连贯",
            "hesitation": "允许羞赧式轻微犹豫",
            "punctuation": "克制使用省略号和反问号",
        },
        "selected_surface_intent": case["intention"],
        "permitted_action_results": [],
    }
    surface_output = validate_text_surface_output(payload)
    return surface_output


def _dialog_state(
    *,
    surface_input: TextSurfaceInputV2,
    surface_output: TextSurfaceOutputV2,
) -> dict[str, Any]:
    """Build the direct production dialog state for one live selector."""

    return {
        "internal_monologue": "按当前意图和边界自然回应。",
        "text_surface_input_v2": surface_input,
        "text_surface_output_v2": surface_output,
        "cognitive_episode": surface_input["episode"],
        "chat_history_wide": [],
        "chat_history_recent": [],
        "platform_user_id": "transition-live-user",
        "platform_bot_id": "transition-live-bot",
        "global_user_id": "transition-live-user",
        "user_name": "测试用户",
        "user_profile": {},
        "character_profile": _character_profile(),
        "final_dialog": [],
        "target_addressed_user_ids": [],
        "target_broadcast": False,
        "dialog_usage_mode": "live_visible_reply",
        "llm_trace_id": "",
    }


def _patch_dialog_observability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep focused live evidence out of persistence-backed trace hooks."""

    monkeypatch.setattr(
        dialog_module.llm_tracing,
        "record_llm_trace_step",
        AsyncMock(),
    )
    for recorder_name in (
        "record_llm_stage_event",
        "record_model_contract_event",
    ):
        monkeypatch.setattr(
            dialog_module.event_logging,
            recorder_name,
            AsyncMock(),
        )


def _install_dialog_captures(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, _CapturingLLM]:
    """Wrap every existing dialog producer used by the focused pipeline."""

    captures: dict[str, _CapturingLLM] = {}
    llm_fields = {
        "generator": "_dialog_generator_llm",
        "semantic_fidelity": "_dialog_semantic_fidelity_llm",
        "role_direction": "_dialog_role_direction_llm",
        "surface_integrity": "_dialog_surface_integrity_llm",
    }
    for owner_name, field_name in llm_fields.items():
        capture = _CapturingLLM(getattr(dialog_module, field_name))
        captures[owner_name] = capture
        monkeypatch.setattr(dialog_module, field_name, capture)
    _patch_dialog_observability(monkeypatch)
    return captures


async def _execute_pipeline(
    case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Run the existing two-owner surface and dialog pipeline once."""

    await _skip_if_model_routes_unavailable()
    surface_input = _surface_input(case)
    cognition_capture = _CapturingLLM(l3_module._llm_interface)
    monkeypatch.setattr(
        l3_module,
        "_llm_interface",
        cognition_capture,
    )
    surface_output = await run_text_surface_planning(
        surface_input,
        l3_module._build_text_surface_services(),
    )
    normal_surface_call_count = len(cognition_capture.calls)
    dialog_captures = _install_dialog_captures(monkeypatch)
    dialog_result = await dialog_module.dialog_generator(
        _dialog_state(
            surface_input=surface_input,
            surface_output=surface_output,
        )
    )
    accepted_surface = validate_text_surface_output(
        dialog_result["text_surface_output_v2"]
    )
    result = {
        "case": case,
        "surface_input": surface_input,
        "surface_output": surface_output,
        "accepted_surface": accepted_surface,
        "final_dialog": dialog_result["final_dialog"],
        "normal_surface_call_count": normal_surface_call_count,
        "surface_model_calls": cognition_capture.calls,
        "dialog_model_calls": {
            owner: capture.calls
            for owner, capture in dialog_captures.items()
        },
    }
    assert normal_surface_call_count == 2
    assert result["final_dialog"]
    return result


def _write_pipeline_trace(result: dict[str, Any]) -> Path:
    """Write one complete pipeline trace for parent-authored review."""

    case = result["case"]
    trace_path = write_llm_trace(
        _SUITE_NAME,
        case["case_id"],
        {
            "fixture_source": case["fixture_source"],
            "manual_expectation": case["manual_expectation"],
            "surface_input": result["surface_input"],
            "normal_surface_call_count": result[
                "normal_surface_call_count"
            ],
            "surface_model_calls": result["surface_model_calls"],
            "surface_output": result["surface_output"],
            "dialog_model_calls": result["dialog_model_calls"],
            "accepted_surface": result["accepted_surface"],
            "final_dialog": result["final_dialog"],
            "manual_transition_review": {
                "opening_stance": "",
                "transition_or_reason": "",
                "final_stance_or_action": "",
                "score": None,
                "notes": "Parent review required after this selector.",
            },
        },
    )
    return trace_path


async def _run_positive_case(
    case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run and print one positive or control selector for manual review."""

    result = await _execute_pipeline(case, monkeypatch)
    trace_path = _write_pipeline_trace(result)
    print(json.dumps({
        "case_id": case["case_id"],
        "trace_path": str(trace_path),
        "surface_output": result["surface_output"],
        "accepted_surface": result["accepted_surface"],
        "final_dialog": result["final_dialog"],
        "manual_expectation": case["manual_expectation"],
    }, ensure_ascii=True, indent=2))


async def _run_seeded_verifier_negative(
    *,
    case_id: str,
    candidate_dialog: list[str],
    fixture_source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one seeded score-zero candidate through semantic fidelity."""

    await _skip_if_model_routes_unavailable()
    surface_input = _surface_input(_CAPTURED_ACCOMPLICE_CASE)
    surface_output = _authoritative_accepting_surface(
        _CAPTURED_ACCOMPLICE_CASE
    )
    semantic_capture = _CapturingLLM(
        dialog_module._dialog_semantic_fidelity_llm
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_capture,
    )
    _patch_dialog_observability(monkeypatch)
    verdict = await dialog_module._verify_dialog_semantic_fidelity(
        surface_output=surface_output,
        generated_dialog=candidate_dialog,
        current_visible_percepts=dialog_module._current_visible_percepts(
            surface_input["episode"]
        ),
        llm_trace_id="",
    )
    trace_path = write_llm_trace(
        _SUITE_NAME,
        case_id,
        {
            "fixture_source": fixture_source,
            "surface_input": surface_input,
            "authoritative_surface": surface_output,
            "candidate_dialog": candidate_dialog,
            "semantic_model_calls": semantic_capture.calls,
            "parsed_verdict": verdict,
            "manual_transition_review": {
                "opening_stance": "earnest settled refusal",
                "transition_or_reason": "no new reason or condition",
                "final_stance_or_action": "accept shared participation",
                "score": 0,
                "notes": "Seeded negative; verifier must reject it.",
            },
        },
    )
    print(json.dumps({
        "case_id": case_id,
        "trace_path": str(trace_path),
        "candidate_dialog": candidate_dialog,
        "parsed_verdict": verdict,
    }, ensure_ascii=True, indent=2))

    assert verdict["aligned"] is False
    assert verdict["issues"]


async def test_live_captured_room_request_acceptance_is_coherent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured room acceptance should retain embarrassment without reversal."""

    await _run_positive_case(_CAPTURED_ROOM_CASE, monkeypatch)


async def test_live_captured_accomplice_confirmation_is_coherent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured accomplice framing should remain confirming and playful."""

    await _run_positive_case(_CAPTURED_ACCOMPLICE_CASE, monkeypatch)


async def test_live_room_request_semantic_variant_is_coherent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A semantically equivalent room request should remain accepting."""

    await _run_positive_case(_ROOM_VARIANT_CASE, monkeypatch)


async def test_live_accomplice_semantic_variant_is_coherent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A semantically equivalent accomplice frame should remain confirming."""

    await _run_positive_case(_ACCOMPLICE_VARIANT_CASE, monkeypatch)


async def test_live_embarrassed_acceptance_variant_is_coherent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embarrassment and indirectness should remain available during acceptance."""

    await _run_positive_case(_EMBARRASSED_ACCEPTANCE_CASE, monkeypatch)


async def test_live_genuine_refusal_remains_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cognition-selected refusal should stay a refusal."""

    await _run_positive_case(_GENUINE_REFUSAL_CASE, monkeypatch)


async def test_live_supported_change_of_mind_preserves_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A change supported by explicit new conditions should remain available."""

    await _run_positive_case(_SUPPORTED_CHANGE_CASE, monkeypatch)


async def test_live_neutral_character_preserves_selected_stance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neutral expression should preserve its selected cooperative stance."""

    await _run_positive_case(_NEUTRAL_CASE, monkeypatch)


async def test_live_known_bad_reversal_is_rejected_by_semantic_fidelity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An earnest same-proposition reversal must fail semantic fidelity."""

    await _run_seeded_verifier_negative(
        case_id="known_bad_reversal_verifier",
        candidate_dialog=[_KNOWN_BAD_REVERSAL],
        fixture_source=(
            "earnest same-proposition control derived from the reference "
            "handover failure class"
        ),
        monkeypatch=monkeypatch,
    )


async def test_live_generated_repair_path_bad_reversal_is_rejected_by_semantic_fidelity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An earnest multi-message repair-shaped reversal must be rejected."""

    await _run_seeded_verifier_negative(
        case_id="generated_bad_reversal_verifier",
        candidate_dialog=list(_REPAIR_SHAPED_BAD_REVERSAL),
        fixture_source=(
            "earnest repair-shaped control derived from "
            f"{_HISTORICAL_GENERATED_BAD_ARTIFACT}"
        ),
        monkeypatch=monkeypatch,
    )


async def test_live_repair_replaces_conflicting_delivery_and_stays_coherent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full repair should replace conflicting delivery before final dialog."""

    await _skip_if_model_routes_unavailable()
    surface_input = _surface_input(_CAPTURED_ACCOMPLICE_CASE)
    conflicting_surface = _authoritative_accepting_surface(
        _CAPTURED_ACCOMPLICE_CASE
    )
    conflicting_surface["delivery_profile"] = {
        "lexical_register": "先字面否认共同参与，再改口接受",
        "sentence_shape": "以指责开场，以妥协收尾",
        "rhythm": "从抵抗突然切换到顺从",
        "hesitation": "用犹豫制造相反立场",
        "punctuation": "用转折连接否认与接受",
    }
    conflicting_surface = validate_text_surface_output(conflicting_surface)
    cognition_capture = _CapturingLLM(l3_module._llm_interface)
    monkeypatch.setattr(
        l3_module,
        "_llm_interface",
        cognition_capture,
    )
    verified_issues = [
        "候选开场否认共同参与，结尾接受同伙关系，权威语义没有支持变化的原因。"
    ]
    repaired_surface = await repair_text_surface_planning(
        surface_input,
        verified_issues,
        l3_module._build_text_surface_services(),
    )
    dialog_captures = _install_dialog_captures(monkeypatch)
    dialog_result = await dialog_module.dialog_generator(
        _dialog_state(
            surface_input=surface_input,
            surface_output=repaired_surface,
        )
    )
    accepted_surface = validate_text_surface_output(
        dialog_result["text_surface_output_v2"]
    )
    trace_path = write_llm_trace(
        _SUITE_NAME,
        "repair_replaces_conflicting_delivery",
        {
            "fixture_source": "seeded structurally valid conflicting delivery",
            "surface_input": surface_input,
            "verified_hard_issues": verified_issues,
            "rejected_surface_trace_only": conflicting_surface,
            "surface_repair_model_calls": cognition_capture.calls,
            "repaired_surface": repaired_surface,
            "dialog_model_calls": {
                owner: capture.calls
                for owner, capture in dialog_captures.items()
            },
            "accepted_surface": accepted_surface,
            "final_dialog": dialog_result["final_dialog"],
            "manual_transition_review": {
                "opening_stance": "",
                "transition_or_reason": "",
                "final_stance_or_action": "",
                "score": None,
                "notes": "Parent review required after this selector.",
            },
        },
    )
    print(json.dumps({
        "case_id": "repair_replaces_conflicting_delivery",
        "trace_path": str(trace_path),
        "repaired_surface": repaired_surface,
        "accepted_surface": accepted_surface,
        "final_dialog": dialog_result["final_dialog"],
    }, ensure_ascii=True, indent=2))

    assert (
        repaired_surface["delivery_profile"]
        != conflicting_surface["delivery_profile"]
    )
    assert repaired_surface["visible_boundaries"] == []
    assert repaired_surface["addressee_plan"] == []
    assert dialog_result["final_dialog"]


async def test_live_progress_records_only_the_accepted_coherent_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress should see only the dialog-accepted semantic surface."""

    pipeline = await _execute_pipeline(
        _CAPTURED_ACCOMPLICE_CASE,
        monkeypatch,
    )
    accepted_surface = pipeline["accepted_surface"]
    record_input: ConversationProgressRecordInput = {
        "scope": ConversationProgressScope(
            "debug",
            "transition-coherence",
            "redacted-user",
        ),
        "storage_timestamp_utc": "2026-07-25T12:23:32+00:00",
        "character_name": "杏山千纱",
        "prior_episode_state": None,
        "decontextualized_input": _CAPTURED_ACCOMPLICE_INPUT,
        "chat_history_recent": [],
        "content_plan": {
            "semantic_content": accepted_surface["content_plan"],
            "surface_intent": accepted_surface["selected_surface_intent"],
        },
        "logical_stance": "CONFIRM",
        "character_intent": "BANTER",
        "final_dialog": pipeline["final_dialog"],
        "boundary_profile": _BOUNDARY_PROFILE,
    }
    recorder_capture = _CapturingLLM(recorder._recorder_llm)
    monkeypatch.setattr(recorder, "_recorder_llm", recorder_capture)
    progress_output = await recorder.record_with_llm(record_input)
    trace_record_input = {
        **record_input,
        "scope": asdict(record_input["scope"]),
    }
    serialized_input = json.dumps(trace_record_input, ensure_ascii=False)
    trace_path = write_llm_trace(
        _SUITE_NAME,
        "progress_accepted_coherent_turn",
        {
            "fixture_source": _CAPTURED_ACCOMPLICE_CASE["fixture_source"],
            "surface_input": pipeline["surface_input"],
            "surface_model_calls": pipeline["surface_model_calls"],
            "initial_surface": pipeline["surface_output"],
            "dialog_model_calls": pipeline["dialog_model_calls"],
            "accepted_surface": accepted_surface,
            "accepted_dialog": pipeline["final_dialog"],
            "record_input": trace_record_input,
            "progress_model_calls": recorder_capture.calls,
            "progress_output": progress_output,
            "manual_transition_review": {
                "opening_stance": "",
                "transition_or_reason": "",
                "final_stance_or_action": "",
                "score": None,
                "notes": (
                    "Parent reviews dialog coherence and confirms progress "
                    "does not promote a rejected transition."
                ),
            },
        },
    )
    print(json.dumps({
        "case_id": "progress_accepted_coherent_turn",
        "trace_path": str(trace_path),
        "accepted_surface": accepted_surface,
        "accepted_dialog": pipeline["final_dialog"],
        "record_input": trace_record_input,
        "progress_output": progress_output,
    }, ensure_ascii=True, indent=2))

    assert record_input["final_dialog"] == pipeline["final_dialog"]
    assert record_input["content_plan"] == {
        "semantic_content": accepted_surface["content_plan"],
        "surface_intent": accepted_surface["selected_surface_intent"],
    }
    assert "delivery_profile" not in serialized_input
    assert "rejected_surface" not in serialized_input
