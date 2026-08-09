"""Contract regressions for visible speech and current-turn fidelity."""

from __future__ import annotations

import json
import typing
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import kazusa_ai_chatbot.cognition_core_v2 as cognition_core_v2
from kazusa_ai_chatbot.cognition_core_v2 import contracts as surface_contracts
from kazusa_ai_chatbot.cognition_core_v2 import surface as surface_module
from kazusa_ai_chatbot.cognition_core_v2 import surface_stages
from kazusa_ai_chatbot.action_spec import results as action_results
from kazusa_ai_chatbot.brain_service.post_turn import settle_episode_trace
from kazusa_ai_chatbot.consolidation.source_policy import (
    build_consolidation_source_views,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.nodes.dialog_agent import (
    DialogAgentState,
    dialog_generator,
)
from tests.cognition_core_v2_test_helpers import canonical_episode


class _SurfaceLLM:
    """Capture stage-local projections while returning exact stage shapes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del config
        system = str(getattr(messages[0], "content", ""))
        human = str(getattr(messages[1], "content", ""))
        payload = json.loads(human)["surface"]
        self.calls.append((system, payload))
        if "visible_boundaries" in system and "addressee_plan" in system:
            result = {
                "visible_boundaries": ["Use visible speech only."],
                "addressee_plan": list(payload.get("addressee_plan", [])),
            }
        elif "content_plan" in system and "content_requirements" in system:
            result = {
                "content_plan": "Perform the requested response operation.",
                "content_requirements": ["Preserve current-turn meaning."],
                "delivery_profile": {
                    "lexical_register": "plain",
                    "sentence_shape": "concise",
                    "rhythm": "steady",
                    "hesitation": "light",
                    "punctuation": "restrained",
                },
            }
        elif "visual_directives" in system:
            result = {
                "visual_directives": "A still-frame emotional composition.",
            }
        else:
            raise AssertionError("unexpected surface stage")
        return SimpleNamespace(content=json.dumps(result))


def _surface_input() -> dict[str, object]:
    """Build one canonical surface input containing a raw physical quirk."""

    return {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(content="Infer an answer from this turn."),
        "intention": {
            "route": "speech",
            "intention": "answer by inference",
            "target_roles": [],
            "reason": "the current request asks for an inference",
        },
        "goal_resolution": "answerable_now",
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "warm",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": "brief conversational speech",
        "character_expression_context": {
            "tempo": "steady",
            "linguistic_texture": "Light hesitation in concise spoken clauses.",
        },
        "visual_character_context": (
            "A physical mannerism accompanies emotion."
        ),
    }


def _surface_services(llm: _SurfaceLLM) -> SimpleNamespace:
    """Bind one capturing model to the text-surface stages."""

    config = SimpleNamespace()
    return SimpleNamespace(
        llm=llm,
        content_plan_config=config,
        preference_config=config,
        visual_config=config,
    )


def _surface_output() -> dict[str, object]:
    """Build the target speech-safe surface output contract."""

    return {
        "schema_version": "text_surface_output.v2",
        "content_plan": "Answer the current request by inference.",
        "content_requirements": [
            "Preserve the requested response operation and current time scope.",
        ],
        "visible_boundaries": ["Return only literal visible speech."],
        "addressee_plan": [{
            "handle": "current_user",
            "display_name": "Current User",
            "semantic_role": "direct_recipient",
            "wording_policy": "second_person_allowed",
        }],
        "delivery_profile": {
            "lexical_register": "warm",
            "sentence_shape": "concise",
            "rhythm": "steady",
            "hesitation": "light",
            "punctuation": "restrained",
        },
        "selected_surface_intent": "answer by inference",
        "permitted_action_results": [],
    }


def _dialog_state() -> dict[str, object]:
    """Build the minimal direct-renderer state with canonical grounding."""

    episode = canonical_episode(
        content="Infer which option fits my stated preference.",
    )
    surface_input = _surface_input()
    surface_input["episode"] = episode
    return {
        "internal_monologue": "I can answer directly.",
        "text_surface_input_v2": surface_input,
        "text_surface_output_v2": _surface_output(),
        "chat_history_wide": [],
        "chat_history_recent": [],
        "platform_user_id": "platform-user",
        "platform_bot_id": "platform-bot",
        "global_user_id": "global-user",
        "user_name": "Current User",
        "user_profile": {},
        "character_profile": {},
        "cognitive_episode": episode,
        "final_dialog": [],
        "target_addressed_user_ids": [],
        "target_broadcast": False,
        "dialog_usage_mode": "unit_test",
        "llm_trace_id": "visible-speech-test",
    }


@pytest.fixture(autouse=True)
def _stub_recorders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep focused renderer tests away from persistent event sinks."""

    monkeypatch.setattr(
        dialog_module,
        "repair_text_surface_for_dialog",
        AsyncMock(return_value=_surface_output()),
        raising=False,
    )
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


@pytest.mark.asyncio
async def test_text_and_visual_planners_are_terminal_siblings() -> None:
    """Physical voice traits never cross from visual into the text surface."""

    llm = _SurfaceLLM()

    output = await surface_module.run_text_surface_planning(
        _surface_input(),
        _surface_services(llm),
    )

    assert set(output) == {
        "schema_version",
        "content_plan",
        "content_requirements",
        "visible_boundaries",
        "addressee_plan",
        "delivery_profile",
        "selected_surface_intent",
        "permitted_action_results",
    }
    assert len(llm.calls) == 2
    for system, payload in llm.calls:
        if "delivery_profile 必须恰好包含" in system:
            assert payload["character_expression_context"] == {
                "tempo": "steady",
                "linguistic_texture": (
                    "Light hesitation in concise spoken clauses."
                ),
            }
        else:
            assert "character_expression_context" not in payload
        assert "visual_character_context" not in payload

    visual_services_type = getattr(
        surface_contracts,
        "VisualSurfaceServicesV2",
    )
    visual_services = visual_services_type(
        llm=llm,
        visual_config=SimpleNamespace(),
    )
    visual_output = await surface_module.run_visual_surface_planning(
        _surface_input(),
        visual_services,
    )

    assert visual_output == {
        "schema_version": "visual_surface_output.v2",
        "visual_directives": "A still-frame emotional composition.",
        "selected_surface_intent": "answer by inference",
    }
    visual_system, visual_payload = llm.calls[-1]
    assert "visual_directives" in visual_system
    assert visual_payload["visual_character_context"] == (
        "A physical mannerism accompanies emotion."
    )


def test_runtime_prompts_define_live_speech_and_hard_error_contracts() -> None:
    """Reusable prompts support vivid speech while guarding hard failures."""

    content_prompt = surface_stages.CONTENT_PLAN_SYSTEM_PROMPT.lower()
    visual_prompt = surface_stages.VISUAL_SYSTEM_PROMPT.lower()
    dialog_prompt = dialog_module._V2_DIALOG_GENERATOR_PROMPT.lower()
    repair_prompt = getattr(
        dialog_module,
        "_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT",
        "",
    ).lower()
    semantic_prompt = (
        dialog_module._V2_DIALOG_SEMANTIC_FIDELITY_PROMPT.lower()
    )
    role_prompt = (
        dialog_module._V2_DIALOG_ROLE_DIRECTION_PROMPT.lower()
    )
    surface_prompt = (
        dialog_module._V2_DIALOG_SURFACE_INTEGRITY_PROMPT.lower()
    )
    surface_repair_prompt = (
        surface_stages.DIALOG_COMPLIANCE_REPAIR_SYSTEM_PROMPT.lower()
    )
    verifier_prompt = f"{semantic_prompt}\n{surface_prompt}"

    assert "想象细节" in content_prompt
    assert "角色判断" in content_prompt
    assert "当前输入" in content_prompt
    assert "行动者" in content_prompt
    assert "对象" in content_prompt
    assert "executed" in content_prompt
    assert "已记录" in content_prompt
    assert "待执行" in content_prompt
    assert "delivery_profile" in content_prompt
    assert "lexical_register" in content_prompt
    assert "action description" not in content_prompt
    assert "动作描写" not in content_prompt
    assert "终端图像" in visual_prompt
    assert "visual_directives" in visual_prompt
    assert "message pacing" not in visual_prompt
    assert "自然" in dialog_prompt
    assert "拒绝" in semantic_prompt
    assert "协商" in semantic_prompt
    assert "附加条件" in semantic_prompt
    assert "角色辨识度" in dialog_prompt
    assert "创造" in dialog_prompt
    assert "实际会说出或发送" in dialog_prompt
    assert "action description" not in dialog_prompt
    assert "动作描写" not in dialog_prompt
    assert "行动者" in dialog_prompt
    assert "对象" in dialog_prompt
    assert "executed" in dialog_prompt
    assert "pacing_guidance" not in dialog_prompt
    assert "visual_directives" not in dialog_prompt
    assert "verified_hard_issues" in repair_prompt
    assert "current_visible_percepts" not in repair_prompt
    assert "text_surface_output_v2" in repair_prompt
    assert "repair_context" in repair_prompt
    assert "visible_boundaries 的具体来源类型" in surface_repair_prompt
    assert "隐私" in surface_repair_prompt
    assert "安全" in surface_repair_prompt
    assert "内容审查" in surface_repair_prompt
    assert "权威语境" in surface_repair_prompt
    assert "l3" not in repair_prompt
    assert "surface owner" not in repair_prompt
    assert "l3" not in dialog_prompt
    assert "surface owner" not in dialog_prompt
    assert "current_visible_percepts" in verifier_prompt
    assert "role_explicit_content" in semantic_prompt
    assert "response_operation" in semantic_prompt
    assert "selection_owner" in semantic_prompt
    assert "已经从本阶段输入中移除" in semantic_prompt
    assert "角色方向检查独占" in semantic_prompt
    assert "本阶段核对候选的内部语义连贯" in semantic_prompt
    assert "保留在输入中的非选择 response_operation" in semantic_prompt
    assert "负责核对行动者、对象" in semantic_prompt
    assert "并列动作覆盖度" in semantic_prompt
    assert "属于内容完整性检查" in semantic_prompt
    assert "hard_errors 必须引用候选原文" in semantic_prompt
    assert "ascii token hard_errors" in semantic_prompt
    assert "愿望、请求或祈使句" in role_prompt
    assert "说出、回答、选择或发送" in role_prompt
    assert "选择哪项动作" in role_prompt
    assert "不得以不够具体" in role_prompt
    assert "内部存在冲突" in verifier_prompt
    assert "当前用户输入" in verifier_prompt
    assert "行动者" in verifier_prompt
    assert "对象" in verifier_prompt
    assert "主语" in verifier_prompt
    assert "action description" not in verifier_prompt
    assert "动作描写" not in verifier_prompt
    assert "executed" in verifier_prompt
    assert "合理虚构" in verifier_prompt
    assert "不属于" in verifier_prompt
    assert "false_execution" in surface_prompt
    for retired_text in (
        "claim-by-claim audit",
        "must remain silent about future",
        "generalize, euphemize, narrow, broaden",
        "descriptors, attributes, qualifiers",
        "rhetorical question cannot substitute",
        "unrestricted permission",
    ):
        assert retired_text not in "\n".join((
            content_prompt,
            dialog_prompt,
            verifier_prompt,
        ))


def test_public_api_exports_sibling_text_and_visual_surfaces() -> None:
    """The public V2 package exposes both independent surface entrypoints."""

    assert cognition_core_v2.VisualSurfaceOutputV2 is not None
    assert cognition_core_v2.VisualSurfaceServicesV2 is not None
    assert callable(cognition_core_v2.run_visual_surface_planning)


def test_dialog_state_requires_current_cognitive_episode() -> None:
    """The renderer state exposes the canonical current-turn grounding."""

    hints = typing.get_type_hints(DialogAgentState)

    assert "cognitive_episode" in hints
    global_hints = typing.get_type_hints(GlobalPersonaState)
    assert "visual_surface_output_v2" in global_hints


def test_visual_directives_convert_only_to_terminal_trace_evidence() -> None:
    """Visual directives become a private image artifact without delivery."""

    output = action_results.build_visual_surface_output(
        fragments=["A still-frame emotional composition."],
        created_at="2026-07-17T00:00:00Z",
    )

    assert output == {
        "schema_version": "surface_output.v1",
        "surface_kind": "image",
        "visibility": "private",
        "action_attempt_id": None,
        "fragments": ["A still-frame emotional composition."],
        "artifact_refs": [],
        "delivery_intent": "do_not_deliver",
        "created_at": "2026-07-17T00:00:00Z",
    }


def test_terminal_visual_trace_has_no_consolidation_consumer() -> None:
    """Private image directives remain absent from LLM-facing projections."""

    created_at = "2026-07-17T00:00:00Z"
    text_output = action_results.build_text_surface_output(
        fragments=["Visible literal speech."],
        created_at=created_at,
    )
    visual_output = action_results.build_visual_surface_output(
        fragments=["A still-frame emotional composition."],
        created_at=created_at,
    )
    trace = settle_episode_trace(
        episode=canonical_episode(
            episode_id="episode-terminal-visual",
            content="Visible literal speech.",
        ),
        cognition_output=None,
        action_specs=[],
        action_results=[],
        surface_outputs=[text_output, visual_output],
        terminal_status="completed_visible",
        attempt_diagnostics=[],
        delivery_correlation={
            "schema_version": "delivery_correlation.v1",
            "delivery_intent": "deliver_now",
            "tracking_id": "delivery-terminal-visual",
            "receipt_status": "delivered",
            "receipt_ref": "receipt-terminal-visual",
        },
        settled_at=created_at,
    )

    projection = action_results.project_episode_trace_for_consolidation(trace)

    assert projection["surface_outputs"] == [{
        "surface_kind": "text",
        "visibility": "user_visible",
        "delivery_intent": "deliver_now",
        "fragments": ["Visible literal speech."],
    }]
    assert "still-frame" not in json.dumps(projection)
    source_views = build_consolidation_source_views({
        "consolidation_origin": {"trigger_source": "user_message"},
        "episode_trace_projection": projection,
    })
    assert "still-frame" not in json.dumps(source_views)


def test_dialog_projection_reuses_shared_episode_size_bound() -> None:
    """A valid compact episode does not gain a dialog-only percept-count cap."""

    episode = canonical_episode(content="Visible percept 0.")
    episode["percepts"] = [
        {
            "schema_version": "percept.v1",
            "percept_kind": "dialog",
            "source_kind": "dialog",
            "source_id": f"percept:visible:{index}",
            "content": {
                "semantic_text": f"Visible percept {index}.",
                "text": f"Visible percept {index}.",
            },
            "observed_at": episode["created_at"],
        }
        for index in range(17)
    ]

    percepts = dialog_module._current_visible_percepts(episode)

    assert len(percepts) == 17


@pytest.mark.asyncio
async def test_verifier_receives_bounded_visible_percepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Focused checks receive only their authoritative current-turn fields."""

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(
        return_value=SimpleNamespace(
            content='{"final_dialog": ["This option fits your preference."]}',
        ),
    )
    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(
        return_value=SimpleNamespace(
            content='{"aligned": true, "hard_errors": []}',
        ),
    )
    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(
        return_value=SimpleNamespace(
            content='{"aligned": true, "issues": []}',
        ),
    )
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_llm,
    )

    result = await dialog_generator(_dialog_state())

    assert result == {
        "final_dialog": ["This option fits your preference."],
        "text_surface_output_v2": _surface_output(),
    }
    generator_payload = json.loads(
        generator_llm.ainvoke.await_args.args[0][1].content,
    )
    assert "character_expression_context" not in json.dumps(
        generator_payload
    )
    assert "visual_character_context" not in json.dumps(generator_payload)
    assert "visual_directives" not in json.dumps(generator_payload)
    compliance_payload = json.loads(
        semantic_llm.ainvoke.await_args.args[0][1].content,
    )
    assert set(compliance_payload) == {
        "candidate_final_dialog",
        "candidate_role_frame",
        "current_visible_percepts",
        "authoritative_surface_semantics",
    }
    assert compliance_payload["candidate_role_frame"] == {
        "speaker_role": "当前角色",
        "first_person_role": "当前角色",
        "second_person_role": "当前用户",
    }
    assert compliance_payload["current_visible_percepts"] == [{
        "input_source": "dialog",
        "content": {
            "semantic_text": "Infer which option fits my stated preference.",
            "text": "Infer which option fits my stated preference.",
        },
        "speaker_role": "当前用户",
        "addressee_role": "当前角色",
        "first_person_role": "当前用户",
        "implicit_imperative_subject_role": "当前角色",
    }, {
        "input_source": "local_time_context",
        "content": {
            "local_time_context": {
                "current_local_datetime": "2026-07-14 12:00",
                "current_local_weekday": "Tuesday",
            },
        },
    }]
    assert compliance_payload["authoritative_surface_semantics"] == {
        "selected_surface_intent": "answer by inference",
        "content_plan": "Answer the current request by inference.",
        "content_requirements": [
            "Preserve the requested response operation and current time scope.",
        ],
        "visible_boundaries": ["Return only literal visible speech."],
        "addressee_plan": [{
            "handle": "current_user",
            "display_name": "Current User",
            "semantic_role": "direct_recipient",
            "wording_policy": "second_person_allowed",
        }],
    }
    surface_payload = json.loads(
        surface_llm.ainvoke.await_args.args[0][1].content,
    )
    assert set(surface_payload) == {
        "candidate_final_dialog",
        "completed_source_evidence",
        "permitted_action_results",
    }
    semantic_llm.ainvoke.assert_awaited_once()
    surface_llm.ainvoke.assert_awaited_once()
    rendered = json.dumps(compliance_payload)
    for forbidden_field in (
        "delivery_profile",
        "permitted_action_results",
        "metadata",
        "target_scope",
        "origin_metadata",
        "storage_timestamp_utc",
    ):
        assert forbidden_field not in rendered


@pytest.mark.asyncio
async def test_dialog_preserves_explicit_high_risk_language_when_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dialog harness does not keyword-filter aligned visible content."""

    candidate = "我现在真的想死了。"
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps(
            {"final_dialog": [candidate]},
            ensure_ascii=False,
        ),
    ))
    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content='{"aligned": true, "hard_errors": []}',
    ))
    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content='{"aligned": true, "issues": []}',
    ))
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_llm,
    )

    result = await dialog_generator(_dialog_state())

    assert result == {
        "final_dialog": [candidate],
        "text_surface_output_v2": _surface_output(),
    }
    semantic_payload = json.loads(
        semantic_llm.ainvoke.await_args.args[0][1].content,
    )
    surface_payload = json.loads(
        surface_llm.ainvoke.await_args.args[0][1].content,
    )
    assert semantic_payload["candidate_final_dialog"] == [candidate]
    assert surface_payload["candidate_final_dialog"] == [candidate]
    semantic_llm.ainvoke.assert_awaited_once()
    surface_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_focused_verifiers_merge_four_issues_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both focused owners contribute within the merged eight-issue bound."""

    semantic_issues = [f"semantic issue {index}" for index in range(4)]
    surface_issue_rows = [
        {
            "kind": "false_execution",
            "evidence": evidence,
            "explanation": f"surface issue {index}",
        }
        for index, evidence in enumerate((
            "This",
            "option",
            "fits",
            "preference",
        ))
    ]
    surface_issues = [
        (
            f"{row['kind']}: {row['evidence']!r} - "
            f"{row['explanation']}"
        )
        for row in surface_issue_rows
    ]
    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({
            "aligned": False,
            "hard_errors": semantic_issues,
        }),
    ))
    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({
            "aligned": False,
            "issues": surface_issue_rows,
        }),
    ))
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_llm,
    )

    state = _dialog_state()
    verdict = await dialog_module._verify_dialog_compliance(
        surface_output=state["text_surface_output_v2"],
        generated_dialog=["This option fits your preference."],
        current_visible_percepts=dialog_module._current_visible_percepts(
            state["cognitive_episode"],
        ),
        llm_trace_id="bounded-merge",
    )

    assert verdict == {
        "semantic_fidelity": {
            "status": "misaligned",
            "issues": semantic_issues,
        },
        "role_direction": {
            "status": "aligned",
            "violations": [],
        },
        "surface_integrity": {
            "status": "misaligned",
            "issues": surface_issue_rows,
        },
    }
    assert dialog_module._dialog_verifier_aggregate_repair_issues(
        verdict
    ) == semantic_issues + surface_issues
    semantic_llm.ainvoke.assert_awaited_once()
    surface_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_focused_verifier_exhausts_on_a_fifth_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized focused verdict becomes unavailable after three tries."""

    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({
            "aligned": False,
            "hard_errors": [
                f"semantic issue {index}"
                for index in range(5)
            ],
        }),
    ))
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )

    verdict = await dialog_module._verify_dialog_semantic_fidelity(
        surface_output=_surface_output(),
        generated_dialog=["This option fits your preference."],
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "Choose one option.",
        }],
        llm_trace_id="focused-overflow",
    )

    assert verdict == {"status": "unavailable", "issues": []}
    assert semantic_llm.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_semantic_verifier_regenerates_invalid_structure_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Semantic verification repairs shape without changing its source packet."""

    invalid_response = json.dumps({
        "aligned": True,
        "Issues": ["x" * 9000],
    })
    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(side_effect=[
        SimpleNamespace(content=invalid_response),
        SimpleNamespace(content='{"aligned": true, "hard_errors": []}'),
    ])
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    trace_recorder = dialog_module.llm_tracing.record_llm_trace_step

    verdict = await dialog_module._verify_dialog_semantic_fidelity(
        surface_output=_surface_output(),
        generated_dialog=["I choose the next action."],
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "Choose what happens next.",
        }],
        llm_trace_id="semantic-structure-repair",
    )

    assert verdict == {"aligned": True, "issues": []}
    assert semantic_llm.ainvoke.await_count == 2
    first_messages = semantic_llm.ainvoke.await_args_list[0].args[0]
    repair_messages = semantic_llm.ainvoke.await_args_list[1].args[0]
    assert [message.type for message in repair_messages] == [
        "system",
        "human",
        "ai",
        "human",
    ]
    assert repair_messages[0].content == first_messages[0].content
    assert repair_messages[1].content == first_messages[1].content
    assert len(repair_messages[2].content) <= (
        dialog_module.DIALOG_VERIFIER_REJECTED_OUTPUT_MAX_CHARS
    )
    assert "dialog semantic fidelity fields are not exact" in (
        repair_messages[3].content
    )
    assert "missing=['hard_errors']" in repair_messages[3].content
    assert "unexpected=['Issues']" in repair_messages[3].content
    assert '{"aligned": true, "hard_errors": []}' in (
        repair_messages[3].content
    )
    assert dialog_module.DIALOG_SEMANTIC_VERDICT_FALSE_EXAMPLE in (
        repair_messages[3].content
    )
    assert "unexpected 字段" in (
        repair_messages[3].content
    )
    assert "不能出现在替代对象里" in (
        repair_messages[3].content
    )
    assert "structure" in repair_messages[3].content.lower()
    assert trace_recorder.await_count == 2
    rejected_trace = trace_recorder.await_args_list[0].kwargs
    accepted_trace = trace_recorder.await_args_list[1].kwargs
    assert rejected_trace["status"] == "failed"
    assert rejected_trace["parse_status"] == "contract_error"
    assert rejected_trace["sequence"] == 0
    assert accepted_trace["status"] == "succeeded"
    assert accepted_trace["parse_status"] == "succeeded"
    assert accepted_trace["sequence"] == 1


@pytest.mark.asyncio
async def test_role_verifier_regenerates_invalid_structure_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Role verification preserves the authoritative five-field role tuple."""

    invalid_response = json.dumps({
        "aligned": True,
        "Issues": ["x" * 9000],
    })
    role_llm = MagicMock()
    role_llm.ainvoke = AsyncMock(side_effect=[
        SimpleNamespace(content=invalid_response),
        SimpleNamespace(content='{"aligned": true, "violations": []}'),
    ])
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        role_llm,
    )
    trace_recorder = dialog_module.llm_tracing.record_llm_trace_step
    percepts = [{
        "input_source": "dialog_text",
        "content": {
            "semantic_text": "Choose the next action.",
            "response_operation": {
                "operation": "the character chooses the next action",
                "response_owner_role": "current_character",
                "selection_owner_role": "current_character",
                "selection_required": True,
                "embedded_actor_role": "current_character",
                "embedded_target_role": "current_user",
            },
        },
    }]

    verdict = await dialog_module._verify_dialog_role_direction(
        generated_dialog=["I choose the next action."],
        current_visible_percepts=percepts,
        llm_trace_id="role-structure-repair",
    )

    assert verdict == {"aligned": True, "violations": []}
    assert role_llm.ainvoke.await_count == 2
    first_messages = role_llm.ainvoke.await_args_list[0].args[0]
    repair_messages = role_llm.ainvoke.await_args_list[1].args[0]
    assert [message.type for message in repair_messages] == [
        "system",
        "human",
        "ai",
        "human",
    ]
    assert repair_messages[0].content == first_messages[0].content
    assert repair_messages[1].content == first_messages[1].content
    assert len(repair_messages[2].content) <= (
        dialog_module.DIALOG_VERIFIER_REJECTED_OUTPUT_MAX_CHARS
    )
    assert "dialog compliance fields are not exact" in (
        repair_messages[3].content
    )
    payload = json.loads(first_messages[1].content)
    assert payload["required_role_operations"] == [{
        "response_owner_role": "current_character",
        "selection_owner_role": "current_character",
        "selection_required": True,
        "embedded_actor_role": "current_character",
        "embedded_target_role": "current_user",
    }]
    assert trace_recorder.await_count == 2
    rejected_trace = trace_recorder.await_args_list[0].kwargs
    accepted_trace = trace_recorder.await_args_list[1].kwargs
    assert rejected_trace["status"] == "failed"
    assert rejected_trace["parse_status"] == "contract_error"
    assert rejected_trace["sequence"] == 0
    assert accepted_trace["status"] == "succeeded"
    assert accepted_trace["parse_status"] == "succeeded"
    assert accepted_trace["sequence"] == 1


@pytest.mark.asyncio
async def test_surface_verifier_regenerates_invalid_structure_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface verification repairs structure with the same evidence packet."""

    invalid_response = json.dumps({
        "aligned": True,
        "issues": "x" * 9000,
    })
    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(side_effect=[
        SimpleNamespace(content=invalid_response),
        SimpleNamespace(content='{"aligned": true, "issues": []}'),
    ])
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_llm,
    )
    trace_recorder = dialog_module.llm_tracing.record_llm_trace_step

    verdict = await dialog_module._verify_dialog_surface_integrity(
        surface_output=_dialog_state()["text_surface_output_v2"],
        generated_dialog=["I can answer that directly."],
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "Answer directly.",
        }],
        llm_trace_id="surface-structure-repair",
    )

    assert verdict == {"aligned": True, "issues": []}
    assert surface_llm.ainvoke.await_count == 2
    first_messages = surface_llm.ainvoke.await_args_list[0].args[0]
    repair_messages = surface_llm.ainvoke.await_args_list[1].args[0]
    assert [message.type for message in repair_messages] == [
        "system",
        "human",
        "ai",
        "human",
    ]
    assert repair_messages[0].content == first_messages[0].content
    assert repair_messages[1].content == first_messages[1].content
    assert len(repair_messages[2].content) <= (
        dialog_module.DIALOG_VERIFIER_REJECTED_OUTPUT_MAX_CHARS
    )
    assert "surface compliance issues are invalid" in (
        repair_messages[3].content
    )
    assert trace_recorder.await_count == 2
    rejected_trace = trace_recorder.await_args_list[0].kwargs
    accepted_trace = trace_recorder.await_args_list[1].kwargs
    assert rejected_trace["status"] == "failed"
    assert rejected_trace["parse_status"] == "contract_error"
    assert rejected_trace["sequence"] == 0
    assert accepted_trace["status"] == "succeeded"
    assert accepted_trace["parse_status"] == "succeeded"
    assert accepted_trace["sequence"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verifier_name", "empty_field"),
    [
        ("semantic", "issues"),
        ("role", "violations"),
        ("surface", "issues"),
    ],
)
async def test_focused_verifier_exhaustion_returns_unavailable(
    verifier_name: str,
    empty_field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three structural failures mark only the focused verifier unavailable."""

    invalid_response = SimpleNamespace(
        content='{"aligned": true, "Issues": []}',
    )
    verifier_llm = MagicMock()
    verifier_llm.ainvoke = AsyncMock(return_value=invalid_response)
    if verifier_name == "semantic":
        monkeypatch.setattr(
            dialog_module,
            "_dialog_semantic_fidelity_llm",
            verifier_llm,
        )
        verifier_call = dialog_module._verify_dialog_semantic_fidelity(
            surface_output=_surface_output(),
            generated_dialog=["I choose the next action."],
            current_visible_percepts=[{
                "input_source": "dialog_text",
                "content": "Choose the next action.",
            }],
            llm_trace_id="semantic-structure-exhaustion",
        )
    elif verifier_name == "role":
        monkeypatch.setattr(
            dialog_module,
            "_dialog_role_direction_llm",
            verifier_llm,
        )
        verifier_call = dialog_module._verify_dialog_role_direction(
            generated_dialog=["I choose the next action."],
            current_visible_percepts=[{
                "input_source": "dialog_text",
                "content": {
                    "response_operation": {
                        "response_owner_role": "current_character",
                        "selection_owner_role": "current_character",
                        "selection_required": True,
                        "embedded_actor_role": "current_character",
                        "embedded_target_role": "current_user",
                    },
                },
            }],
            llm_trace_id="role-structure-exhaustion",
        )
    else:
        monkeypatch.setattr(
            dialog_module,
            "_dialog_surface_integrity_llm",
            verifier_llm,
        )
        verifier_call = dialog_module._verify_dialog_surface_integrity(
            surface_output=_dialog_state()["text_surface_output_v2"],
            generated_dialog=["I choose the next action."],
            current_visible_percepts=[{
                "input_source": "dialog_text",
                "content": "Choose the next action.",
            }],
            llm_trace_id="surface-structure-exhaustion",
        )

    verdict = await verifier_call

    assert verdict == {
        "status": "unavailable",
        empty_field: [],
    }
    assert verifier_llm.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_role_direction_verifier_skips_without_required_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary turns add no focused selection-owner model call."""

    role_llm = MagicMock()
    role_llm.ainvoke = AsyncMock(side_effect=AssertionError(
        "role verifier must not run without required selection",
    ))
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        role_llm,
    )

    verdict = await dialog_module._verify_dialog_role_direction(
        generated_dialog=["I can answer that directly."],
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "Tell me whether you agree.",
        }],
        llm_trace_id="role-direction-skip",
    )

    assert verdict == {"aligned": True, "violations": []}
    role_llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_selection_role_reversal_remains_semantic_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-selection actor reversal is retained by semantic fidelity."""

    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({
            "aligned": False,
            "hard_errors": [
                "候选明确颠倒了当前角色与当前用户的行动方向。"
            ],
        }, ensure_ascii=False),
    ))
    role_llm = MagicMock()
    role_llm.ainvoke = AsyncMock(side_effect=AssertionError(
        "selection-only verifier must skip non-selection operations",
    ))
    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content='{"aligned": true, "issues": []}',
    ))
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        role_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_llm,
    )
    state = _dialog_state()
    percepts = [{
        "input_source": "dialog",
        "content": {
            "semantic_text": "当前用户要求当前角色执行直接动作。",
            "role_explicit_content": (
                "当前角色是行动者，当前用户是动作对象。"
            ),
            "response_operation": {
                "operation": "当前角色对当前用户执行动作",
                "response_owner_role": "当前角色",
                "selection_owner_role": "当前角色",
                "selection_required": False,
                "embedded_actor_role": "当前角色",
                "embedded_target_role": "当前用户",
            },
        },
    }]

    verdict = await dialog_module._verify_dialog_compliance(
        surface_output=state["text_surface_output_v2"],
        generated_dialog=["换你对我执行这个动作。"],
        current_visible_percepts=percepts,
        llm_trace_id="non-selection-role-reversal",
    )

    assert verdict == {
        "semantic_fidelity": {
            "status": "misaligned",
            "issues": [
                "候选明确颠倒了当前角色与当前用户的行动方向。"
            ],
        },
        "role_direction": {
            "status": "aligned",
            "violations": [],
        },
        "surface_integrity": {
            "status": "aligned",
            "issues": [],
        },
    }
    semantic_llm.ainvoke.assert_awaited_once()
    role_llm.ainvoke.assert_not_awaited()
    surface_llm.ainvoke.assert_awaited_once()
    semantic_payload = json.loads(
        semantic_llm.ainvoke.await_args.args[0][1].content
    )
    semantic_content = (
        semantic_payload["current_visible_percepts"][0]["content"]
    )
    assert "role_explicit_content" in semantic_content
    assert "response_operation" in semantic_content


@pytest.mark.asyncio
async def test_selection_role_fields_stay_out_of_semantic_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection ownership reaches only the dedicated role verifier."""

    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content='{"aligned": true, "hard_errors": []}',
    ))
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    percept = {
        "input_source": "dialog_text",
        "content": {
            "semantic_text": "Tell me the next action you want me to take.",
            "text": "Tell me the next action you want me to take.",
            "role_explicit_content": (
                "The current character selects an action for the current user."
            ),
            "response_operation": {
                "operation": "the character selects the next action",
                "response_owner_role": "current_character",
                "selection_owner_role": "current_character",
                "selection_required": True,
                "embedded_actor_role": "current_user",
                "embedded_target_role": "current_character",
            },
        },
    }

    verdict = await dialog_module._verify_dialog_semantic_fidelity(
        surface_output=_surface_output(),
        generated_dialog=["Come sit beside me."],
        current_visible_percepts=[percept],
        llm_trace_id="selection-fields-excluded-from-semantic",
    )

    assert verdict == {"aligned": True, "issues": []}
    payload = json.loads(
        semantic_llm.ainvoke.await_args.args[0][1].content
    )
    semantic_content = (
        payload["current_visible_percepts"][0]["content"]
    )
    assert semantic_content == {
        "semantic_text": "Tell me the next action you want me to take.",
        "text": "Tell me the next action you want me to take.",
    }


def test_hard_verifier_and_repair_exclude_drifted_l3_prose() -> None:
    """Hard gates retain typed facts while repair returns to the L3 owner."""

    surface_prompt = dialog_module._V2_DIALOG_SURFACE_INTEGRITY_PROMPT
    repair_prompt = dialog_module._V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT

    assert "active_visible_boundaries" not in surface_prompt
    assert "delivery_profile" not in surface_prompt
    assert "text_surface_output_v2" in repair_prompt
    assert "current_visible_percepts" not in repair_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate,aligned",
    [
        ("Tell me what to do next; I will follow your choice.", False),
        ("Next, hold my hand and stay close to me.", True),
    ],
)
async def test_role_direction_verifier_owns_required_selection(
    candidate: str,
    aligned: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The focused owner rejects delegation and preserves correct selection."""

    role_llm = MagicMock()
    role_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({
            "aligned": aligned,
            "violations": [] if aligned else [{
                "kind": "selection_owner_transfer",
                "evidence": "I will follow your choice.",
                "explanation": (
                    "选择所有者从当前角色错误地变为当前用户。"
                ),
            }],
        }),
    ))
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        role_llm,
    )
    percept = {
        "input_source": "dialog_text",
        "content": {
            "semantic_text": "Tell me what you want me to do next.",
            "role_explicit_content": (
                "当前用户要求当前角色直接告诉当前用户当前角色下一步要做什么"
            ),
            "response_operation": {
                "operation": "当前角色选择并告诉当前用户下一步动作",
                "response_owner_role": "当前角色",
                "selection_owner_role": "当前角色",
                "selection_required": True,
                "embedded_actor_role": "当前用户",
                "embedded_target_role": "当前角色",
            },
        },
    }

    verdict = await dialog_module._verify_dialog_role_direction(
        generated_dialog=[candidate],
        current_visible_percepts=[percept],
        llm_trace_id="role-direction-required-selection",
    )

    assert verdict["aligned"] is aligned
    role_llm.ainvoke.assert_awaited_once()
    payload = json.loads(role_llm.ainvoke.await_args.args[0][1].content)
    assert set(payload) == {
        "candidate_final_dialog",
        "candidate_role_frame",
        "required_role_operations",
    }
    assert payload["required_role_operations"] == [{
        "response_owner_role": "当前角色",
        "selection_owner_role": "当前角色",
        "selection_required": True,
        "embedded_actor_role": "当前用户",
        "embedded_target_role": "当前角色",
    }]


@pytest.mark.asyncio
async def test_role_direction_verifier_requires_exact_candidate_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hallucinated role evidence cannot reject a bounded dialog candidate."""

    role_llm = MagicMock()
    role_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({
            "aligned": False,
            "violations": [{
                "kind": "selection_owner_transfer",
                "evidence": "I will follow your choice.",
                "explanation": "The user is said to own the selection.",
            }],
        }),
    ))
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        role_llm,
    )
    percept = {
        "input_source": "dialog_text",
        "content": {
            "semantic_text": "Tell me what you want me to do next.",
            "role_explicit_content": (
                "当前用户要求当前角色直接告诉当前用户当前角色下一步要做什么"
            ),
            "response_operation": {
                "operation": "当前角色选择并告诉当前用户下一步动作",
                "response_owner_role": "当前角色",
                "selection_owner_role": "当前角色",
                "selection_required": True,
                "embedded_actor_role": "当前用户",
                "embedded_target_role": "当前角色",
            },
        },
    }

    verdict = await dialog_module._verify_dialog_role_direction(
        generated_dialog=["Next, hold my hand and stay close to me."],
        current_visible_percepts=[percept],
        llm_trace_id="role-direction-candidate-evidence",
    )

    assert verdict == {"status": "unavailable", "violations": []}
    assert role_llm.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_surface_verifier_requires_exact_candidate_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vague taxonomy restatement cannot block visible dialog."""

    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({
            "aligned": False,
            "issues": ["Action or stage narration."],
        }),
    ))
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_llm,
    )

    verdict = await dialog_module._verify_dialog_surface_integrity(
        surface_output=_dialog_state()["text_surface_output_v2"],
        generated_dialog=["Um... I agree."],
        current_visible_percepts=[{
            "input_source": "dialog_text",
            "content": "Do you agree?",
        }],
        llm_trace_id="surface-evidence",
    )

    assert verdict == {"status": "unavailable", "issues": []}
    assert surface_llm.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_false_execution_verdict_uses_one_grounded_llm_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported capability execution invokes one grounded repair."""

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(side_effect=[
        SimpleNamespace(
            content='{"final_dialog": ["I changed the platform alarm."]}',
        ),
        SimpleNamespace(
            content='{"final_dialog": ["This option fits your preference."]}',
        ),
    ])
    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(side_effect=[
        SimpleNamespace(content='{"aligned": true, "hard_errors": []}'),
        SimpleNamespace(content='{"aligned": true, "hard_errors": []}'),
    ])
    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(side_effect=[
        SimpleNamespace(content=json.dumps({
            "aligned": False,
            "issues": [{
                "kind": "false_execution",
                "evidence": "changed the platform alarm",
                "explanation": "No executed result supports this claim.",
            }],
        })),
        SimpleNamespace(content='{"aligned": true, "issues": []}'),
    ])
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_llm,
    )

    result = await dialog_generator(_dialog_state())

    assert result == {
        "final_dialog": ["This option fits your preference."],
        "text_surface_output_v2": _surface_output(),
    }
    assert generator_llm.ainvoke.await_count == 2
    assert semantic_llm.ainvoke.await_count == 2
    assert surface_llm.ainvoke.await_count == 2
    repair_payload = json.loads(
        generator_llm.ainvoke.await_args_list[1].args[0][1].content,
    )
    assert set(repair_payload) == {
        "candidate_role_frame",
        "repair_context",
        "text_surface_output_v2",
        "user_name",
    }
    assert repair_payload["candidate_role_frame"] == {
        "speaker_role": "当前角色",
        "first_person_role": "当前角色",
        "second_person_role": "当前用户",
    }
    assert repair_payload["text_surface_output_v2"] == _surface_output()
    assert repair_payload["repair_context"] == {
        "verified_hard_issues": [
            "false_execution: 'changed the platform alarm' - "
            "No executed result supports this claim.",
        ],
    }
    assert "current_visible_percepts" not in repair_payload


@pytest.mark.asyncio
async def test_second_rejection_uses_terminal_unverified_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded third candidate remains deliverable after two rejections."""

    invalid_dialog = "Ask me what to do next; I will follow your choice."
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({"final_dialog": [invalid_dialog]}),
    ))
    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({
            "aligned": False,
            "hard_errors": ["Subject reversal remains."],
        }),
    ))
    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content='{"aligned": true, "issues": []}',
    ))
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_llm,
    )

    result = await dialog_generator(_dialog_state())

    assert result == {
        "final_dialog": [invalid_dialog],
        "text_surface_output_v2": _surface_output(),
    }
    assert generator_llm.ainvoke.await_count == 3
    assert semantic_llm.ainvoke.await_count == 2
    assert surface_llm.ainvoke.await_count == 2
    repair_payload = json.loads(
        generator_llm.ainvoke.await_args_list[1].args[0][1].content,
    )
    assert "text_surface_output_v2" in repair_payload
    assert "repair_context" in repair_payload


@pytest.mark.asyncio
async def test_surface_owner_repair_replaces_all_owned_fields() -> None:
    """The L3 owner replaces its fields while preserving canonical truth."""

    class _RepairLLM:
        """Return one complete surface-owned semantic replacement."""

        def __init__(self) -> None:
            self.messages: list[object] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            self.messages = list(messages)
            response = SimpleNamespace(content=json.dumps({
                "content_plan": "角色明确说出自己希望用户执行的下一步。",
                "content_requirements": [
                    "由当前角色作出选择并告诉当前用户。",
                ],
                "visible_boundaries": [
                    "角色可以拒绝、协商或附加条件。",
                ],
                "addressee_plan": [{
                    "handle": "current_user",
                    "display_name": "当前用户",
                    "semantic_role": "direct_recipient",
                    "wording_policy": "second_person_allowed",
                }],
                "delivery_profile": {
                    "lexical_register": "直接口语",
                    "sentence_shape": "紧凑短句",
                    "rhythm": "平稳",
                    "hesitation": "无额外犹豫",
                    "punctuation": "克制",
                },
            }, ensure_ascii=False))
            return response

    llm = _RepairLLM()
    services = _surface_services(llm)
    input_payload = _surface_input()
    input_payload["permitted_action_results"] = [{
        "action_kind": "future_speak",
        "status": "pending",
        "semantic_result": "等待后台执行。",
        "target_roles": [],
    }]

    repaired = await surface_module.repair_text_surface_planning(
        input_payload,
        ["选择所有者从当前角色错误地变为当前用户。"],
        services,
    )

    assert repaired["content_plan"] == (
        "角色明确说出自己希望用户执行的下一步。"
    )
    assert repaired["content_requirements"] == [
        "由当前角色作出选择并告诉当前用户。",
    ]
    assert repaired["visible_boundaries"] == [
        "角色可以拒绝、协商或附加条件。",
    ]
    assert repaired["addressee_plan"] == [{
        "handle": "current_user",
        "display_name": "当前用户",
        "semantic_role": "direct_recipient",
        "wording_policy": "second_person_allowed",
    }]
    assert repaired["delivery_profile"] == {
        "lexical_register": "直接口语",
        "sentence_shape": "紧凑短句",
        "rhythm": "平稳",
        "hesitation": "无额外犹豫",
        "punctuation": "克制",
    }
    assert repaired["selected_surface_intent"] == (
        input_payload["intention"]["intention"]
    )
    assert repaired["permitted_action_results"] == (
        input_payload["permitted_action_results"]
    )
    payload = json.loads(getattr(llm.messages[1], "content"))
    repair_context = payload["surface"]["dialog_compliance_repair"]
    assert repair_context == {
        "verified_hard_issues": [
            "选择所有者从当前角色错误地变为当前用户。",
        ],
    }


@pytest.mark.asyncio
async def test_surface_owner_repair_regenerates_invalid_contract_once() -> None:
    """The semantic producer repairs structure within its two-attempt cap."""

    class _RepairingLLM:
        """Return one invalid candidate followed by a complete replacement."""

        def __init__(self) -> None:
            self.messages: list[list[object]] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            self.messages.append(list(messages))
            if len(self.messages) == 1:
                response = SimpleNamespace(content='{"content_plan": 3}')
            else:
                response = SimpleNamespace(content=json.dumps({
                    "content_plan": "当前角色明确选择下一步并告诉当前用户。",
                    "content_requirements": ["保持选择所有者为当前角色。"],
                    "visible_boundaries": [],
                    "addressee_plan": [{
                        "handle": "current_user",
                        "display_name": "当前用户",
                        "semantic_role": "direct_recipient",
                        "wording_policy": "second_person_allowed",
                    }],
                    "delivery_profile": {
                        "lexical_register": "直接口语",
                        "sentence_shape": "紧凑短句",
                        "rhythm": "平稳",
                        "hesitation": "无额外犹豫",
                        "punctuation": "克制",
                    },
                }, ensure_ascii=False))
            return response

    llm = _RepairingLLM()
    repaired = await surface_module.repair_text_surface_planning(
        _surface_input(),
        ["选择所有者从当前角色错误地变为当前用户。"],
        _surface_services(llm),
    )

    assert repaired["content_plan"] == (
        "当前角色明确选择下一步并告诉当前用户。"
    )
    assert len(llm.messages) == 2
    repair_system = str(getattr(llm.messages[1][0], "content", ""))
    repair_payload = json.loads(
        str(getattr(llm.messages[1][1], "content", "{}"))
    )
    assert repair_system == str(
        getattr(llm.messages[0][0], "content", "")
    )
    assert (
        repair_payload["contract_repair"]["repair_instruction"]
        == surface_stages.SURFACE_REPAIR_INSTRUCTION
    )
    assert repair_payload["contract_repair"]["invalid_candidate"] == (
        '{"content_plan": 3}'
    )
    assert (
        repair_payload["surface"]["dialog_compliance_repair"][
            "verified_hard_issues"
        ]
        == ["选择所有者从当前角色错误地变为当前用户。"]
    )


@pytest.mark.asyncio
async def test_surface_owner_repair_exhaustion_has_post_commit_metadata() -> None:
    """Three invalid replacements expose typed post-commit metadata."""

    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content='{"content_plan": 3}',
    ))

    with pytest.raises(
        surface_contracts.CognitionExecutionError,
    ) as error_info:
        await surface_module.repair_text_surface_planning(
            _surface_input(),
            ["选择所有者从当前角色错误地变为当前用户。"],
            _surface_services(llm),
        )

    error = error_info.value
    assert error.error_code == (
        "surface_dialog_compliance_repair_contract_exhausted"
    )
    assert error.stage == "surface.dialog_compliance_repair"
    assert error.attempt_count == 3
    assert error.safe_checkpoint == "post_cognition_commit"
    assert error.retryable is False
    assert llm.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_dialog_repair_uses_l3_replacement_as_rendering_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wording owner renders one L3 replacement instead of raw percepts."""

    replacement = _surface_output()
    replacement["content_plan"] = (
        "当前角色明确告诉当前用户下一步该执行的动作。"
    )
    surface_repair = AsyncMock(return_value=replacement)
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps(
            {"final_dialog": ["下一步，握住我的手，别移开视线。"]},
            ensure_ascii=False,
        ),
    ))
    monkeypatch.setattr(
        dialog_module,
        "repair_text_surface_for_dialog",
        surface_repair,
    )
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)

    repaired_dialog, repaired_surface = (
        await dialog_module._repair_dialog_hard_failure(
            repair_issues=["选择所有者被错误地交给当前用户。"],
            surface_input=_surface_input(),
            user_name="Current User",
            llm_trace_id="surface-owner-repair",
        )
    )

    assert repaired_dialog == ["下一步，握住我的手，别移开视线。"]
    assert repaired_surface == replacement
    surface_repair.assert_awaited_once_with(
        surface_input=_surface_input(),
        verified_hard_issues=["选择所有者被错误地交给当前用户。"],
    )
    repair_payload = json.loads(
        generator_llm.ainvoke.await_args.args[0][1].content,
    )
    assert repair_payload["text_surface_output_v2"] == replacement
    assert repair_payload["repair_context"] == {
        "verified_hard_issues": ["选择所有者被错误地交给当前用户。"],
    }
    assert "current_visible_percepts" not in repair_payload


@pytest.mark.asyncio
async def test_dialog_semantic_exhaustion_delivers_bounded_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two rejected candidates finish with normal terminal dialog delivery."""

    invalid_dialog = "你来替我决定我想让你做什么。"
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps(
            {"final_dialog": [invalid_dialog]},
            ensure_ascii=False,
        ),
    ))
    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({
            "aligned": False,
            "hard_errors": ["当前角色仍把自己的选择交给当前用户。"],
        }, ensure_ascii=False),
    ))
    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content='{"aligned": true, "issues": []}',
    ))
    surface_repair = AsyncMock(return_value=_surface_output())
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "repair_text_surface_for_dialog",
        surface_repair,
    )
    state = _dialog_state()
    state["text_surface_input_v2"] = _surface_input()

    result = await dialog_generator(state)

    assert result == {
        "final_dialog": [invalid_dialog],
        "text_surface_output_v2": _surface_output(),
    }
    assert generator_llm.ainvoke.await_count == 3
    assert semantic_llm.ainvoke.await_count == 2
    assert surface_llm.ainvoke.await_count == 2
