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
        if "exactly style_guidance" in system:
            result = {"style_guidance": "speech-safe cadence"}
        elif "exactly content_plan" in system:
            result = {
                "content_plan": "Perform the requested response operation.",
                "content_requirements": ["Preserve current-turn meaning."],
            }
        elif "exactly visible_boundaries" in system:
            result = {
                "visible_boundaries": ["Use visible speech only."],
                "addressee_plan": ["Address the current user."],
            }
        elif "exactly visual_directives" in system:
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
    }
    assert len(llm.calls) == 3
    for system, payload in llm.calls:
        if "exactly style_guidance" in system:
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
    assert "exactly visual_directives" in visual_system
    assert visual_payload["character_voice_context"] == (
        "A physical mannerism accompanies emotion."
    )


def test_runtime_prompts_define_general_speech_and_fidelity_contracts() -> None:
    """Reusable prompts state the product rule without captured examples."""

    style_prompt = surface_stages.STYLE_SYSTEM_PROMPT.lower()
    content_prompt = surface_stages.CONTENT_PLAN_SYSTEM_PROMPT.lower()
    visual_prompt = surface_stages.VISUAL_SYSTEM_PROMPT.lower()
    dialog_prompt = dialog_module._V2_DIALOG_GENERATOR_PROMPT.lower()
    verifier_prompt = dialog_module._V2_DIALOG_COMPLIANCE_PROMPT.lower()

    assert "speech-safe" in style_prompt
    assert "must never suggest any detail" in style_prompt
    assert "requested response operation" in content_prompt
    assert "descriptors, attributes, qualifiers" in content_prompt
    assert "generalize, euphemize, narrow, broaden" in content_prompt
    assert "unrestricted permission" in content_prompt
    assert "must remain silent about future" in content_prompt
    assert "rhetorical question cannot substitute" in content_prompt
    assert "no physical actuator" in content_prompt
    assert "first-person execution" in content_prompt
    assert "style alone cannot authorize" in content_prompt
    assert "literal future rule" in content_prompt
    assert "exclusivity condition" in content_prompt
    assert "time scope" in content_prompt
    assert "image-generation" in visual_prompt
    assert "visual_directives" in visual_prompt
    assert "message pacing" not in visual_prompt
    assert "only words the character could literally type or say" in dialog_prompt
    assert "response operation" in dialog_prompt
    assert "descriptors, attributes, qualifiers" in dialog_prompt
    assert "generalize, euphemize, narrow, broaden" in dialog_prompt
    assert "unrestricted permission" in dialog_prompt
    assert "must remain silent about future" in dialog_prompt
    assert "rhetorical question cannot substitute" in dialog_prompt
    assert "no physical actuator" in dialog_prompt
    assert "first-person execution" in dialog_prompt
    assert "performed, completed, delivered, or received" in dialog_prompt
    assert "style alone cannot authorize" in dialog_prompt
    assert "literal future rule" in dialog_prompt
    assert "exclusivity condition" in dialog_prompt
    assert "unmatched enclosing punctuation" in dialog_prompt
    assert "pacing_guidance" not in dialog_prompt
    assert "visual_directives" not in dialog_prompt
    assert "current_visible_percepts" in verifier_prompt
    assert "current_visible_percepts are the semantic authority" in verifier_prompt
    assert "action or stage narration" in verifier_prompt
    assert "descriptors, attributes, qualifiers" in verifier_prompt
    assert "generalize, euphemize, narrow, broaden" in verifier_prompt
    assert "unrestricted permission" in verifier_prompt
    assert "must remain silent about future" in verifier_prompt
    assert "claim-by-claim audit" in verifier_prompt
    assert "surface and candidate agreement is not evidence" in verifier_prompt
    assert "merely restates" in verifier_prompt
    assert "redirects, or asks back" in verifier_prompt
    assert "first-person execution" in verifier_prompt
    assert "requested physical movement" in verifier_prompt
    assert "performed, completed, delivered, or received" in verifier_prompt
    assert "unmatched enclosing punctuation" in verifier_prompt
    assert "time scope" in verifier_prompt


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
    trace = action_results.build_episode_trace(
        episode_id="episode-terminal-visual",
        trigger_source="user_message",
        created_at=created_at,
        action_specs=[],
        action_results=[],
        surface_outputs=[text_output, visual_output],
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
            "percept_id": f"percept:visible:{index}",
            "input_source": "dialog_text",
            "content": f"Visible percept {index}.",
            "visibility": "model_visible",
            "metadata": {},
        }
        for index in range(17)
    ]

    percepts = dialog_module._current_visible_percepts(episode)

    assert len(percepts) == 17


@pytest.mark.asyncio
async def test_verifier_receives_bounded_visible_percepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verifier compares both plan and candidate to the current turn."""

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(
        return_value=SimpleNamespace(
            content='{"final_dialog": ["This option fits your preference."]}',
        ),
    )
    compliance_llm = MagicMock()
    compliance_llm.ainvoke = AsyncMock(
        return_value=SimpleNamespace(
            content='{"aligned": true, "issues": []}',
        ),
    )
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(dialog_module, "_dialog_compliance_llm", compliance_llm)

    result = await dialog_generator(_dialog_state())

    assert result == {"final_dialog": ["This option fits your preference."]}
    generator_payload = json.loads(
        generator_llm.ainvoke.await_args.args[0][1].content,
    )
    assert "character_voice_context" not in json.dumps(generator_payload)
    assert "visual_directives" not in json.dumps(generator_payload)
    compliance_payload = json.loads(
        compliance_llm.ainvoke.await_args.args[0][1].content,
    )
    assert compliance_payload["current_visible_percepts"] == [{
        "input_source": "dialog_text",
        "content": "Infer which option fits my stated preference.",
        "speaker_role": "current_user",
        "addressee_role": "self",
        "first_person_role": "current_user",
        "implicit_imperative_subject_role": "self",
    }]
    rendered = json.dumps(compliance_payload)
    for forbidden_field in (
        "metadata",
        "target_scope",
        "origin_metadata",
        "storage_timestamp_utc",
        "local_time_context",
    ):
        assert forbidden_field not in rendered


@pytest.mark.asyncio
async def test_negative_verdict_uses_one_grounded_llm_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Narration or meaning drift invokes only the existing grounded repair."""

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(side_effect=[
        SimpleNamespace(
            content='{"final_dialog": ["*moves closer* Ask me instead."]}',
        ),
        SimpleNamespace(
            content='{"final_dialog": ["This option fits your preference."]}',
        ),
    ])
    compliance_llm = MagicMock()
    compliance_llm.ainvoke = AsyncMock(
        return_value=SimpleNamespace(content=json.dumps({
            "aligned": False,
            "issues": [
                "Remove action narration and perform the requested inference.",
            ],
        })),
    )
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(dialog_module, "_dialog_compliance_llm", compliance_llm)

    result = await dialog_generator(_dialog_state())

    assert result == {"final_dialog": ["This option fits your preference."]}
    assert generator_llm.ainvoke.await_count == 2
    assert compliance_llm.ainvoke.await_count == 1
    repair_payload = json.loads(
        generator_llm.ainvoke.await_args_list[1].args[0][1].content,
    )
    assert repair_payload["repair_context"]["current_visible_percepts"] == [{
        "input_source": "dialog_text",
        "content": "Infer which option fits my stated preference.",
        "speaker_role": "current_user",
        "addressee_role": "self",
        "first_person_role": "current_user",
        "implicit_imperative_subject_role": "self",
    }]
    assert repair_payload["repair_context"]["issues"] == [
        "Remove action narration and perform the requested inference.",
    ]
