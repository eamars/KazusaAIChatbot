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
        if "style_guidance" in system and "content_plan" not in system:
            result = {"style_guidance": "speech-safe cadence"}
        elif "content_plan" in system and "content_requirements" in system:
            result = {
                "content_plan": "Perform the requested response operation.",
                "content_requirements": ["Preserve current-turn meaning."],
            }
        elif "visible_boundaries" in system and "addressee_plan" in system:
            result = {
                "visible_boundaries": ["Use visible speech only."],
                "addressee_plan": ["Address the current user."],
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
        "character_voice_context": "A physical mannerism accompanies emotion.",
    }


def _surface_services(llm: _SurfaceLLM) -> SimpleNamespace:
    """Bind one capturing model to the text-surface stages."""

    config = SimpleNamespace()
    return SimpleNamespace(
        llm=llm,
        style_config=config,
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
        "addressee_plan": ["Address the current user."],
        "style_guidance": "Warm, concise spoken wording.",
        "selected_surface_intent": "answer by inference",
        "permitted_action_results": [],
    }


def _dialog_state() -> dict[str, object]:
    """Build the minimal direct-renderer state with canonical grounding."""

    return {
        "internal_monologue": "I can answer directly.",
        "text_surface_output_v2": _surface_output(),
        "chat_history_wide": [],
        "chat_history_recent": [],
        "platform_user_id": "platform-user",
        "platform_bot_id": "platform-bot",
        "global_user_id": "global-user",
        "user_name": "Current User",
        "user_profile": {},
        "character_profile": {},
        "cognitive_episode": canonical_episode(
            content="Infer which option fits my stated preference.",
        ),
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
        "style_guidance",
        "selected_surface_intent",
        "permitted_action_results",
    }
    assert len(llm.calls) == 3
    for system, payload in llm.calls:
        if "style_guidance" in system and "content_plan" not in system:
            assert payload["character_voice_context"] == (
                "A physical mannerism accompanies emotion."
            )
        else:
            assert "character_voice_context" not in payload

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
    assert visual_payload["character_voice_context"] == (
        "A physical mannerism accompanies emotion."
    )


def test_runtime_prompts_define_live_speech_and_hard_error_contracts() -> None:
    """Reusable prompts support vivid speech while guarding hard failures."""

    style_prompt = surface_stages.STYLE_SYSTEM_PROMPT.lower()
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
    surface_prompt = (
        dialog_module._V2_DIALOG_SURFACE_INTEGRITY_PROMPT.lower()
    )
    verifier_prompt = f"{semantic_prompt}\n{surface_prompt}"

    assert "表达指导" in style_prompt
    assert "用词" in style_prompt
    assert "想象细节" in content_prompt
    assert "角色判断" in content_prompt
    assert "当前输入" in content_prompt
    assert "行动者" in content_prompt
    assert "对象" in content_prompt
    assert "executed" in content_prompt
    assert "已记录" in content_prompt
    assert "待执行" in content_prompt
    assert "action description" not in content_prompt
    assert "动作描写" not in content_prompt
    assert "终端图像" in visual_prompt
    assert "visual_directives" in visual_prompt
    assert "message pacing" not in visual_prompt
    assert "自然" in dialog_prompt
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
    assert "current_visible_percepts" in repair_prompt
    assert "text_surface_output_v2" not in repair_prompt
    assert "current_visible_percepts" in verifier_prompt
    assert "role_explicit_content" in semantic_prompt
    assert "response_operation" in semantic_prompt
    assert "selection_owner" in semantic_prompt
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
            content='{"aligned": true, "issues": []}',
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

    assert result == {"final_dialog": ["This option fits your preference."]}
    generator_payload = json.loads(
        generator_llm.ainvoke.await_args.args[0][1].content,
    )
    assert "character_voice_context" not in json.dumps(generator_payload)
    assert "visual_directives" not in json.dumps(generator_payload)
    compliance_payload = json.loads(
        semantic_llm.ainvoke.await_args.args[0][1].content,
    )
    assert set(compliance_payload) == {
        "candidate_final_dialog",
        "candidate_role_frame",
        "current_visible_percepts",
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
    rendered = json.dumps({
        "semantic": compliance_payload,
        "surface": surface_payload,
    })
    for forbidden_field in (
        "content_plan",
        "content_requirements",
        "addressee_plan",
        "style_guidance",
        "selected_surface_intent",
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
        content='{"aligned": true, "issues": []}',
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

    assert result == {"final_dialog": [candidate]}
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
            "issues": semantic_issues,
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
        "aligned": False,
        "issues": semantic_issues + surface_issues,
    }
    semantic_llm.ainvoke.assert_awaited_once()
    surface_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_focused_verifier_rejects_a_fifth_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A focused owner cannot consume the merged verdict's issue budget."""

    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({
            "aligned": False,
            "issues": [f"semantic issue {index}" for index in range(5)],
        }),
    ))
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )

    with pytest.raises(
        dialog_module.StateContractError,
        match="issues are invalid",
    ):
        await dialog_module._verify_dialog_semantic_fidelity(
            generated_dialog=["This option fits your preference."],
            current_visible_percepts=[{
                "input_source": "dialog_text",
                "content": "Choose one option.",
            }],
            llm_trace_id="focused-overflow",
        )


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

    assert verdict == {"aligned": True, "issues": []}
    role_llm.ainvoke.assert_not_awaited()


def test_hard_verifier_and_repair_exclude_drifted_l3_prose() -> None:
    """Hard gates use typed facts and execution truth, not L3 prose."""

    surface_prompt = dialog_module._V2_DIALOG_SURFACE_INTEGRITY_PROMPT
    repair_prompt = dialog_module._V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT

    assert "active_visible_boundaries" not in surface_prompt
    assert "style_guidance" not in surface_prompt
    assert "不提供自由" in repair_prompt
    assert "content plan、boundary 或 style guidance" in repair_prompt


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
            "issues": [] if aligned else [
                "选择所有者从当前角色错误地变为当前用户。",
            ],
        }),
    ))
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        role_llm,
    )
    percept = {
        "input_source": "dialog_text",
        "content": "Tell me what you want me to do next.",
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
    assert payload["required_role_operations"] == [percept]


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

    with pytest.raises(
        dialog_module.StateContractError,
        match="surface issue fields are not exact",
    ):
        await dialog_module._verify_dialog_surface_integrity(
            surface_output=_dialog_state()["text_surface_output_v2"],
            generated_dialog=["Um... I agree."],
            current_visible_percepts=[{
                "input_source": "dialog_text",
                "content": "Do you agree?",
            }],
            llm_trace_id="surface-evidence",
        )


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
        SimpleNamespace(content='{"aligned": true, "issues": []}'),
        SimpleNamespace(content='{"aligned": true, "issues": []}'),
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

    assert result == {"final_dialog": ["This option fits your preference."]}
    assert generator_llm.ainvoke.await_count == 2
    assert semantic_llm.ainvoke.await_count == 2
    assert surface_llm.ainvoke.await_count == 2
    repair_payload = json.loads(
        generator_llm.ainvoke.await_args_list[1].args[0][1].content,
    )
    assert set(repair_payload) == {
        "candidate_role_frame",
        "current_visible_percepts",
        "original_final_dialog",
        "permitted_action_results",
        "user_name",
        "verified_hard_issues",
    }
    assert repair_payload["current_visible_percepts"] == [{
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
    assert repair_payload["verified_hard_issues"] == [
        "false_execution: 'changed the platform alarm' - "
        "No executed result supports this claim.",
    ]
    assert "text_surface_output_v2" not in repair_payload


@pytest.mark.asyncio
async def test_repaired_dialog_must_pass_the_same_hard_error_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repair cannot deliver the same verified role reversal."""

    invalid_dialog = "Ask me what to do next; I will follow your choice."
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({"final_dialog": [invalid_dialog]}),
    ))
    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({
            "aligned": False,
            "issues": ["Subject reversal remains."],
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

    with pytest.raises(
        dialog_module.StateContractError,
        match="remains hard-invalid after one repair",
    ):
        await dialog_generator(_dialog_state())

    assert generator_llm.ainvoke.await_count == 2
    assert semantic_llm.ainvoke.await_count == 2
    assert surface_llm.ainvoke.await_count == 2
    repair_payload = json.loads(
        generator_llm.ainvoke.await_args_list[1].args[0][1].content,
    )
    assert "surface_repair_context" not in repair_payload
