"""Stage 07 reflection-triggered cognition dry-run tests."""

from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_episode import validate_cognitive_episode
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l1 as l1_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2 as l2_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2d as l2d_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as l3_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2c2 as l2c2_module
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_prompt_selection import (
    CognitionPromptSelectionError,
    CognitionPromptStage,
    build_cognition_prompt_source_payload,
    select_cognition_prompt_variant,
)
from kazusa_ai_chatbot.reflection_cycle.cognition_dry_run import (
    ReflectionCognitionDryRunError,
    build_reflection_signal_cognitive_episode,
    run_reflection_cognition_dry_run,
)
from kazusa_ai_chatbot.reflection_cycle import cognition_dry_run as dry_run_module


_APPROVED_STAGES: tuple[CognitionPromptStage, ...] = (
    "l1_subconscious",
    "l2a_conscious_framing",
    "l2b_boundary_appraisal",
    "l2c1_judgment_synthesis",
    "l2d_action_selection",
    "l2c2_social_context_appraisal",
    "l3_style_agent",
    "l3_content_anchor_agent",
    "l3_preference_adapter",
    "l3_visual_agent",
)
_EXPECTED_PROMPT_KEYS = [
    "l1_subconscious.reflection_signal_reflection_artifact",
    "l2a_conscious_framing.reflection_signal_reflection_artifact",
    "l2b_boundary_appraisal.reflection_signal_reflection_artifact",
    "l2c1_judgment_synthesis.reflection_signal_reflection_artifact",
    "l2c2_social_context_appraisal.reflection_signal_reflection_artifact",
    "l2d_action_selection.reflection_signal_reflection_artifact",
]
_FORBIDDEN_DRY_RUN_TOKENS = (
    "call_consolidation_subgraph",
    "save_conversation",
    "save_assistant_message",
    "dispatcher.dispatch",
    "runtime.invalidate",
    "get_rag_cache2_runtime",
)
_PROMPT_FINGERPRINTS = (
    (
        "_COGNITION_SUBCONSCIOUS_PROMPT",
        l1_module._COGNITION_SUBCONSCIOUS_PROMPT,
        3069,
        "ded0a7507ac36786961beeaf5e6771e960c4c42816288911182ea09529965c81",
    ),
    (
        "_COGNITION_CONSCIOUSNESS_PROMPT",
        l2_module._COGNITION_CONSCIOUSNESS_PROMPT,
        5701,
        "911012f371c386fd89c1e09fd170938235be89de9ef0a36d1510ef3d64b236e4",
    ),
    (
        "_BOUNDARY_CORE_PROMPT",
        l2_module._BOUNDARY_CORE_PROMPT,
        5144,
        "55ee910a4f3a2c2201a9425e86990ac39fba1588cf9ae48c9e6d849727995022",
    ),
    (
        "_JUDGEMENT_CORE_PROMPT",
        l2_module._JUDGEMENT_CORE_PROMPT,
        3797,
        "32d57cf70d58bd87332c29f97e16c93a6907cc9a0661a5b30d5067d73876bd45",
    ),
    (
        "_CONTEXTUAL_AGENT_PROMPT",
        l2c2_module._CONTEXTUAL_AGENT_PROMPT,
        4202,
        "03609614a7996464bcf77fc775d423d8e62e2cfe6d7e08bf89dc471eb79ceff6",
    ),
    (
        "_STYLE_AGENT_PROMPT",
        l3_module._STYLE_AGENT_PROMPT,
        7033,
        "2b06474ba46bca3a348fd8d926fdd93e7aa49225030f1d4f58d7737174bf8c71",
    ),
    (
        "_CONTENT_ANCHOR_AGENT_PROMPT",
        l3_module._CONTENT_ANCHOR_AGENT_PROMPT,
        21115,
        "9162c058e5fa9295a17c079afade56ead3e6a2edcc307174f2e20da004442447",
    ),
    (
        "_PREFERENCE_ADAPTER_PROMPT",
        l3_module._PREFERENCE_ADAPTER_PROMPT,
        7660,
        "75ac96a8aeec479cb963662ffa7f86346a15a0c8816e248eb415d0b4195d07c6",
    ),
    (
        "_VISUAL_AGENT_PROMPT",
        l3_module._VISUAL_AGENT_PROMPT,
        7965,
        "371a2ae8b10a28460b677fed37552c6a9e8358274712e03ef07351a77c289b53",
    ),
)


class _DummyResponse:
    """Small LangChain-like response wrapper for dry-run prompt tests."""

    def __init__(self, content: str) -> None:
        """Create a dummy LLM response.

        Args:
            content: Response text returned from the fake LLM.
        """
        self.content = content


class _CapturingAsyncLLM:
    """Async fake LLM that records messages and returns fixed JSON."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Create the capturing fake.

        Args:
            payload: JSON-serializable payload returned by `ainvoke`.
        """
        self.payload = payload
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> _DummyResponse:
        """Record prompt messages and return the configured payload.

        Args:
            messages: Prompt messages passed by the cognition handler.

        Returns:
            Dummy response containing serialized JSON.
        """
        self.messages = messages
        content = json.dumps(self.payload, ensure_ascii=False)
        response = _DummyResponse(content)
        return response


class _BusyProbe:
    """Callable busy probe with call-count inspection."""

    def __init__(self, busy: bool) -> None:
        """Create a deterministic busy probe.

        Args:
            busy: Value returned whenever the probe is called.
        """
        self.busy = busy
        self.call_count = 0

    def __call__(self) -> bool:
        """Return the configured busy value.

        Returns:
            Whether the primary interaction path is busy.
        """
        self.call_count += 1
        return_value = self.busy
        return return_value


class _CapturingCognitionCallable:
    """Injected cognition callable that records dry-run states."""

    def __init__(self, output: dict[str, Any]) -> None:
        """Create the injected cognition callable.

        Args:
            output: Dict returned from each invocation.
        """
        self.output = output
        self.calls = 0
        self.states: list[dict[str, Any]] = []

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Record the dry-run state and return configured output.

        Args:
            state: Global persona state built by the dry-run runner.

        Returns:
            Configured cognition result copy.
        """
        self.calls += 1
        self.states.append(deepcopy(state))
        return_value = deepcopy(self.output)
        return return_value


def _time_context() -> dict[str, str]:
    """Build a fixed character-local time context for dry-run tests.

    Returns:
        Minimal local time context mapping.
    """
    time_context = {
        "current_local_datetime": "2026-05-10 09:30",
        "current_local_weekday": "Sunday",
    }
    return time_context


def _promoted_reflection_context() -> dict[str, list[dict[str, str]] | list[str]]:
    """Build promoted reflection context with one model-visible artifact.

    Returns:
        `PromotedReflectionContext`-compatible mapping containing promoted lore
        and self-guidance lanes.
    """
    context = {
        "promoted_lore": [
            {
                "memory_name": "quiet_library_preference",
                "content": "The active character protects quiet study spaces.",
                "memory_type": "lore",
                "updated_at": "2026-05-09T22:15:00+00:00",
                "confidence_note": "Repeated in promoted reflection.",
            },
        ],
        "promoted_self_guidance": [
            {
                "memory_name": "avoid_overcommitting",
                "content": "Keep reflective conclusions tentative in speech.",
                "memory_type": "self_guidance",
                "updated_at": "2026-05-09T22:20:00+00:00",
                "confidence_note": "Promoted as internal guidance.",
            },
        ],
        "source_dates": ["2026-05-09"],
        "retrieval_notes": [
            "Only active reflection-promoted memory rows are included.",
        ],
    }
    return context


def _canonical_context(context: dict[str, Any]) -> str:
    """Render promoted reflection context with the stable payload shape.

    Args:
        context: Promoted reflection context to render.

    Returns:
        Canonical JSON string used as the reflection percept content.
    """
    rendered = json.dumps(context, sort_keys=True, ensure_ascii=False)
    return rendered


def _expected_episode_id(context: dict[str, Any]) -> str:
    """Return the deterministic reflection dry-run episode id.

    Args:
        context: Promoted reflection context used to derive the id.

    Returns:
        Expected `episode_id` for the reflection dry-run episode.
    """
    digest = hashlib.sha256(
        _canonical_context(context).encode("utf-8"),
    ).hexdigest()[:16]
    episode_id = f"reflection:dry_run:{digest}"
    return episode_id


def _character_profile() -> dict[str, Any]:
    """Build a cognition-compatible character profile fixture.

    Returns:
        Character profile with all fields consumed by L1/L2/L3 prompt render.
    """
    profile = {
        "name": "Kazusa",
        "personality_brief": {
            "mbti": "INTJ",
            "logic": "precise and guarded",
            "tempo": "measured",
            "defense": "uses dry deflection",
            "quirks": "notices quiet rooms",
            "taboos": "do not overpromise",
        },
        "mood": "calm",
        "global_vibe": "quiet",
        "reflection_summary": "",
        "boundary_profile": {
            "self_integrity": 0.7,
            "control_sensitivity": 0.4,
            "relational_override": 0.3,
            "control_intimacy_misread": 0.2,
            "compliance_strategy": "negotiate",
            "boundary_recovery": "recenter",
            "authority_skepticism": 0.5,
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.2,
            "hesitation_density": 0.3,
            "counter_questioning": 0.2,
            "softener_density": 0.3,
            "formalism_avoidance": 0.6,
            "abstraction_reframing": 0.4,
            "direct_assertion": 0.5,
            "emotional_leakage": 0.2,
            "rhythmic_bounce": 0.3,
            "self_deprecation": 0.1,
        },
    }
    return profile


def _user_profile() -> dict[str, Any]:
    """Build a minimal user profile fixture.

    Returns:
        User profile fields consumed by cognition prompt render.
    """
    profile = {
        "global_user_id": "reflection_cycle",
        "affinity": 500,
        "last_relationship_insight": "",
    }
    return profile


def _expected_empty_rag_result() -> dict[str, Any]:
    """Build the exact empty RAG shape required for dry-run cognition.

    Returns:
        Empty projected RAG payload for dry-run cognition state.
    """
    rag_result = {
        "answer": "",
        "user_image": {
            "user_memory_context": {
                "stable_patterns": [],
                "recent_shifts": [],
                "objective_facts": [],
                "milestones": [],
                "active_commitments": [],
            },
        },
        "user_memory_unit_candidates": [],
        "character_image": {},
        "third_party_profiles": [],
        "memory_evidence": [],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {
            "loop_count": 0,
            "unknown_slots": [],
            "dispatched": [],
        },
    }
    return rag_result


def _llm_output_payloads() -> dict[str, dict[str, Any]]:
    """Build fixed JSON outputs for every cognition LLM stage.

    Returns:
        Mapping of LLM attribute names to stage-compatible payloads.
    """
    payloads = {
        "_subconscious_llm": {
            "emotional_appraisal": "steady",
            "interaction_subtext": "reflection",
        },
        "_conscious_llm": {
            "internal_monologue": "Keep the reflection tentative.",
            "character_intent": "PROVIDE",
            "logical_stance": "CONFIRM",
        },
        "_boundary_core_llm": {
            "boundary_issue": "none",
            "boundary_summary": "no boundary issue",
            "behavior_primary": "answer",
            "behavior_secondary": "none",
            "acceptance": "allow",
            "stance_bias": "confirm",
            "identity_policy": "accept",
            "pressure_policy": "absorb",
            "trajectory": "stable",
        },
        "_judgement_core_llm": {
            "logical_stance": "CONFIRM",
            "character_intent": "PROVIDE",
            "judgment_note": "reflection dry run remains internal",
        },
        "_action_initializer_llm": {
            "action_requests": [],
        },
        "_contextual_agent_llm": {
            "social_distance": "neutral",
            "emotional_intensity": "low",
            "vibe_check": "quiet",
            "relational_dynamic": "stable",
        },
        "_style_agent_llm": {
            "rhetorical_strategy": "brief",
            "linguistic_style": "plain",
            "forbidden_phrases": [],
        },
        "_content_anchor_agent_llm": {
            "content_anchors": ["[DECISION] keep internal", "[SCOPE] dry run"],
        },
        "_preference_adapter_llm": {
            "accepted_user_preferences": [],
        },
        "_visual_agent_llm": {
            "facial_expression": ["neutral"],
            "body_language": ["still"],
            "gaze_direction": ["down"],
            "visual_vibe": ["quiet"],
        },
    }
    return payloads


def _empty_interaction_style_context() -> dict[str, Any]:
    """Build an empty interaction-style context without database access.

    Returns:
        Prompt-facing interaction-style overlay fixture.
    """
    overlay = {
        "speech_guidelines": [],
        "social_guidelines": [],
        "pacing_guidelines": [],
        "engagement_guidelines": [],
        "confidence": "empty",
    }
    context = {
        "user_style": overlay,
        "application_order": ["user_style"],
    }
    return context


async def _fake_build_interaction_style_context(**_kwargs: Any) -> dict[str, Any]:
    """Return empty interaction-style context without database access.

    Args:
        **_kwargs: Ignored context lookup parameters.

    Returns:
        Empty prompt-facing interaction-style context.
    """
    context = _empty_interaction_style_context()
    return context


def _patch_cognition_llms(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, _CapturingAsyncLLM]:
    """Patch every L1/L2d/L3 cognition LLM with a capturing fake.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Mapping of patched LLM attribute names to fake LLMs.
    """
    payloads = _llm_output_payloads()
    llms = {
        llm_name: _CapturingAsyncLLM(payload)
        for llm_name, payload in payloads.items()
    }
    monkeypatch.setattr(l1_module, "_subconscious_llm", llms["_subconscious_llm"])
    monkeypatch.setattr(l2_module, "_conscious_llm", llms["_conscious_llm"])
    monkeypatch.setattr(l2_module, "_boundary_core_llm", llms["_boundary_core_llm"])
    monkeypatch.setattr(l2_module, "_judgement_core_llm", llms["_judgement_core_llm"])
    monkeypatch.setattr(
        l2d_module,
        "_action_initializer_llm",
        llms["_action_initializer_llm"],
    )
    monkeypatch.setattr(
        l2c2_module,
        "_contextual_agent_llm",
        llms["_contextual_agent_llm"],
    )
    monkeypatch.setattr(l3_module, "_style_agent_llm", llms["_style_agent_llm"])
    monkeypatch.setattr(
        l3_module,
        "_content_anchor_agent_llm",
        llms["_content_anchor_agent_llm"],
    )
    monkeypatch.setattr(
        l3_module,
        "_preference_adapter_llm",
        llms["_preference_adapter_llm"],
    )
    monkeypatch.setattr(l3_module, "_visual_agent_llm", llms["_visual_agent_llm"])
    return llms


def test_reflection_episode_builder_creates_valid_episode() -> None:
    """Builder should create the exact reflection_signal episode contract."""
    context = _promoted_reflection_context()
    episode = build_reflection_signal_cognitive_episode(
        promoted_reflection_context=context,
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        output_mode="think_only",
    )

    validate_cognitive_episode(episode)

    assert episode["episode_id"] == _expected_episode_id(context)
    assert episode["trigger_source"] == "reflection_signal"
    assert episode["input_sources"] == ["reflection_artifact"]
    assert episode["output_mode"] == "think_only"
    assert episode["storage_timestamp_utc"] == "2026-05-09T21:30:00+00:00"
    assert episode["local_time_context"] == _time_context()
    assert episode["target_scope"] == {
        "platform": "reflection_cycle",
        "platform_channel_id": "reflection_dry_run",
        "channel_type": "reflection_dry_run",
        "current_platform_user_id": "reflection_cycle",
        "current_global_user_id": "reflection_cycle",
        "current_display_name": "reflection_cycle",
        "target_addressed_user_ids": [],
        "target_broadcast": False,
    }
    assert episode["origin_metadata"] == {
        "platform": "reflection_cycle",
        "platform_message_id": "reflection:dry_run",
        "active_turn_platform_message_ids": [],
        "active_turn_conversation_row_ids": [],
        "debug_modes": {"think_only": True, "no_remember": True},
    }
    assert episode["percepts"] == [
        {
            "percept_id": "reflection:artifact:promoted_context",
            "input_source": "reflection_artifact",
            "content": _canonical_context(context),
            "visibility": "model_visible",
            "metadata": {"source": "promoted_reflection_context"},
        },
    ]


def test_reflection_episode_builder_rejects_empty_promoted_context() -> None:
    """Builder should reject context with no promoted reflection lanes."""
    with pytest.raises(
        ReflectionCognitionDryRunError,
        match="promoted reflection context is empty",
    ):
        build_reflection_signal_cognitive_episode(
            promoted_reflection_context={"retrieval_notes": ["nothing active"]},
            storage_timestamp_utc="2026-05-09T21:30:00+00:00",
            local_time_context=_time_context(),
        )


def test_reflection_episode_builder_rejects_unsupported_output_mode() -> None:
    """Builder should fail closed for output modes outside the dry-run set."""
    with pytest.raises(
        ReflectionCognitionDryRunError,
        match="reflection output_mode is not supported",
    ):
        build_reflection_signal_cognitive_episode(
            promoted_reflection_context=_promoted_reflection_context(),
            storage_timestamp_utc="2026-05-09T21:30:00+00:00",
            local_time_context=_time_context(),
            output_mode="visible_reply",
        )


def test_selector_returns_reflection_variant_for_every_stage() -> None:
    """Reflection episodes should select the exact dry-run prompt variant."""
    episode = build_reflection_signal_cognitive_episode(
        promoted_reflection_context=_promoted_reflection_context(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        output_mode="preview",
    )

    for stage in _APPROVED_STAGES:
        selection = select_cognition_prompt_variant(
            episode=episode,
            stage=stage,
        )

        assert selection == {
            "stage": stage,
            "variant": "reflection_signal_reflection_artifact",
            "prompt_key": f"{stage}.reflection_signal_reflection_artifact",
            "trigger_source": "reflection_signal",
            "input_sources": ["reflection_artifact"],
            "output_mode": "preview",
        }


def test_source_payload_projects_only_reflection_artifact() -> None:
    """Reflection payload projection should expose only artifact content."""
    context = _promoted_reflection_context()
    episode = build_reflection_signal_cognitive_episode(
        promoted_reflection_context=context,
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        output_mode="think_only",
    )
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l1_subconscious",
    )

    payload = build_cognition_prompt_source_payload(
        episode=episode,
        selection=selection,
    )

    assert payload == {"reflection_artifact": _canonical_context(context)}
    assert "cognitive_episode" not in payload
    assert "trigger_source" not in payload
    assert "input_sources" not in payload


def test_source_payload_rejects_duplicate_reflection_artifacts() -> None:
    """Reflection payload projection should require one artifact percept."""
    episode = build_reflection_signal_cognitive_episode(
        promoted_reflection_context=_promoted_reflection_context(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        output_mode="think_only",
    )
    second_percept = deepcopy(episode["percepts"][0])
    second_percept["percept_id"] = "reflection:artifact:extra_context"
    episode["percepts"].append(second_percept)
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l1_subconscious",
    )

    with pytest.raises(CognitionPromptSelectionError, match="reflection_artifact"):
        build_cognition_prompt_source_payload(
            episode=episode,
            selection=selection,
        )


def test_selector_rejects_reflection_visible_reply_output_mode() -> None:
    """Reflection prompt selection should fail closed for visible replies."""
    episode = build_reflection_signal_cognitive_episode(
        promoted_reflection_context=_promoted_reflection_context(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        output_mode="think_only",
    )
    episode["output_mode"] = "visible_reply"

    with pytest.raises(CognitionPromptSelectionError, match="output_mode"):
        select_cognition_prompt_variant(
            episode=episode,
            stage="l1_subconscious",
        )


@pytest.mark.asyncio
async def test_dry_run_returns_busy_skip_without_cognition_call() -> None:
    """Busy dry run should return the exact skipped audit without cognition."""
    busy_probe = _BusyProbe(True)
    cognition = _CapturingCognitionCallable({"unused": True})

    audit = await run_reflection_cognition_dry_run(
        promoted_reflection_context=_promoted_reflection_context(),
        character_profile=_character_profile(),
        user_profile=_user_profile(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        is_primary_interaction_busy=busy_probe,
        call_cognition_subgraph_func=cognition,
        output_mode="preview",
    )

    assert busy_probe.call_count == 1
    assert cognition.calls == 0
    assert audit == {
        "status": "skipped_busy",
        "skip_reason": "primary_interaction_busy",
        "cognition_called": False,
        "episode_id": "",
        "trigger_source": "reflection_signal",
        "input_sources": ["reflection_artifact"],
        "output_mode": "preview",
        "prompt_variant": "reflection_signal_reflection_artifact",
        "prompt_keys": [],
        "cognition_output_keys": [],
    }


@pytest.mark.asyncio
async def test_dry_run_rejects_output_mode_before_busy_probe() -> None:
    """Runner should validate output mode before checking primary load."""
    cognition = _CapturingCognitionCallable({"unused": True})

    def _failing_busy_probe() -> bool:
        """Fail if output-mode validation did not happen first.

        Returns:
            This function never returns in a passing test.
        """
        raise AssertionError("busy probe should not be called")

    with pytest.raises(
        ReflectionCognitionDryRunError,
        match="reflection output_mode is not supported",
    ):
        await run_reflection_cognition_dry_run(
            promoted_reflection_context=_promoted_reflection_context(),
            character_profile=_character_profile(),
            user_profile=_user_profile(),
            storage_timestamp_utc="2026-05-09T21:30:00+00:00",
            local_time_context=_time_context(),
            is_primary_interaction_busy=_failing_busy_probe,
            call_cognition_subgraph_func=cognition,
            output_mode="visible_reply",
        )

    assert cognition.calls == 0


@pytest.mark.asyncio
async def test_dry_run_returns_empty_context_skip_without_cognition_call() -> None:
    """Empty promoted reflection context should skip before episode build."""
    busy_probe = _BusyProbe(False)
    cognition = _CapturingCognitionCallable({"unused": True})

    audit = await run_reflection_cognition_dry_run(
        promoted_reflection_context={"retrieval_notes": ["nothing active"]},
        character_profile=_character_profile(),
        user_profile=_user_profile(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        is_primary_interaction_busy=busy_probe,
        call_cognition_subgraph_func=cognition,
        output_mode="silent",
    )

    assert busy_probe.call_count == 1
    assert cognition.calls == 0
    assert audit == {
        "status": "skipped_empty_context",
        "skip_reason": "promoted_reflection_context_empty",
        "cognition_called": False,
        "episode_id": "",
        "trigger_source": "reflection_signal",
        "input_sources": ["reflection_artifact"],
        "output_mode": "silent",
        "prompt_variant": "reflection_signal_reflection_artifact",
        "prompt_keys": [],
        "cognition_output_keys": [],
    }


@pytest.mark.asyncio
async def test_dry_run_calls_injected_cognition_once_and_returns_audit() -> None:
    """Non-busy dry run should build exact state and completed audit."""
    context = _promoted_reflection_context()
    busy_probe = _BusyProbe(False)
    cognition = _CapturingCognitionCallable({
        "zeta": "last",
        "alpha": "first",
    })

    audit = await run_reflection_cognition_dry_run(
        promoted_reflection_context=context,
        character_profile=_character_profile(),
        user_profile=_user_profile(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        is_primary_interaction_busy=busy_probe,
        call_cognition_subgraph_func=cognition,
        output_mode="think_only",
    )

    assert busy_probe.call_count == 1
    assert cognition.calls == 1
    assert audit == {
        "status": "completed",
        "skip_reason": "",
        "cognition_called": True,
        "episode_id": _expected_episode_id(context),
        "trigger_source": "reflection_signal",
        "input_sources": ["reflection_artifact"],
        "output_mode": "think_only",
        "prompt_variant": "reflection_signal_reflection_artifact",
        "prompt_keys": _EXPECTED_PROMPT_KEYS,
        "cognition_output_keys": ["alpha", "zeta"],
    }

    dry_run_state = cognition.states[0]
    assert dry_run_state["user_input"] == (
        "Reflection dry run over promoted reflection artifact."
    )
    assert dry_run_state["prompt_message_context"] == {
        "body_text": "",
        "addressed_to_global_user_ids": [],
        "broadcast": False,
        "mentions": [],
        "attachments": [],
    }
    assert dry_run_state["user_multimedia_input"] == []
    assert dry_run_state["platform"] == "reflection_cycle"
    assert dry_run_state["platform_channel_id"] == "reflection_dry_run"
    assert dry_run_state["channel_type"] == "reflection_dry_run"
    assert dry_run_state["platform_message_id"] == "reflection:dry_run"
    assert dry_run_state["platform_user_id"] == "reflection_cycle"
    assert dry_run_state["global_user_id"] == "reflection_cycle"
    assert dry_run_state["user_name"] == "reflection_cycle"
    assert dry_run_state["platform_bot_id"] == "reflection_cycle"
    assert dry_run_state["chat_history_wide"] == []
    assert dry_run_state["chat_history_recent"] == []
    assert dry_run_state["reply_context"] == {}
    assert dry_run_state["indirect_speech_context"] == ""
    assert dry_run_state["channel_topic"] == ""
    assert dry_run_state["debug_modes"] == {
        "think_only": True,
        "no_remember": True,
    }
    assert dry_run_state["should_respond"] is False
    assert dry_run_state["decontexualized_input"] == (
        "Reflection dry run over promoted reflection artifact."
    )
    assert dry_run_state["referents"] == []
    assert dry_run_state["promoted_reflection_context"] == context
    assert dry_run_state["rag_result"] == _expected_empty_rag_result()
    assert dry_run_state["internal_monologue"] == ""
    assert dry_run_state["action_directives"] == {}
    assert dry_run_state["interaction_subtext"] == ""
    assert dry_run_state["emotional_appraisal"] == ""
    assert dry_run_state["character_intent"] == ""
    assert dry_run_state["logical_stance"] == ""
    assert dry_run_state["final_dialog"] == []
    assert dry_run_state["mood"] == ""
    assert dry_run_state["global_vibe"] == ""
    assert dry_run_state["reflection_summary"] == ""
    assert dry_run_state["subjective_appraisals"] == []
    assert dry_run_state["affinity_delta"] == 0
    assert dry_run_state["last_relationship_insight"] == ""
    assert dry_run_state["new_facts"] == []
    assert dry_run_state["future_promises"] == []


def test_dry_run_module_contains_no_write_delivery_or_scheduler_calls() -> None:
    """Dry-run module should not reference write or delivery entrypoints."""
    module_source = inspect.getsource(dry_run_module)

    for token in _FORBIDDEN_DRY_RUN_TOKENS:
        assert token not in module_source


@pytest.mark.asyncio
async def test_reflection_prompt_rendering_uses_only_artifact_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mocked cognition prompts should receive only reflection_artifact content."""
    context = _promoted_reflection_context()
    llms = _patch_cognition_llms(monkeypatch)
    monkeypatch.setattr(
        l3_module,
        "build_interaction_style_context",
        _fake_build_interaction_style_context,
    )

    audit = await run_reflection_cognition_dry_run(
        promoted_reflection_context=context,
        character_profile=_character_profile(),
        user_profile=_user_profile(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        is_primary_interaction_busy=_BusyProbe(False),
        call_cognition_subgraph_func=cognition_module.call_cognition_subgraph,
        output_mode="think_only",
    )

    assert audit["status"] == "completed"
    l1_l2_llm_names = (
        "_subconscious_llm",
        "_conscious_llm",
        "_boundary_core_llm",
        "_judgement_core_llm",
    )
    for llm_name in l1_l2_llm_names:
        fake_llm = llms[llm_name]
        assert fake_llm.messages
        prompt_payload = json.loads(fake_llm.messages[1].content)
        assert prompt_payload["reflection_artifact"] == _canonical_context(context)
        assert "cognitive_episode" not in prompt_payload
        assert "trigger_source" not in prompt_payload
        assert "input_sources" not in prompt_payload
        assert "hourly_reflections" not in prompt_payload
        assert "daily_syntheses" not in prompt_payload
        assert "source_message_refs" not in prompt_payload
        assert "raw_reflection_run" not in prompt_payload
        if "promoted_reflection_context" in prompt_payload:
            assert prompt_payload["promoted_reflection_context"] == {}

    l2d_llm = llms["_action_initializer_llm"]
    assert l2d_llm.messages
    l2d_context = l2d_llm.messages[1].content
    assert l2d_context.startswith("当前行动上下文：")
    assert "触发来源：reflection_signal" in l2d_context
    assert "输入来源：reflection_artifact" in l2d_context
    assert "距离=neutral" in l2d_context
    assert "强度=low" in l2d_context
    assert "氛围=quiet" in l2d_context
    assert "关系=stable" in l2d_context
    assert "cognitive_episode" not in l2d_context
    assert "raw_reflection_run" not in l2d_context
    assert "available_capabilities" not in l2d_context

    invoked_llm_names = (
        *l1_l2_llm_names,
        "_contextual_agent_llm",
        "_action_initializer_llm",
    )
    for llm_name, fake_llm in llms.items():
        if llm_name not in invoked_llm_names:
            assert fake_llm.messages == []


def test_text_chat_prompt_fingerprints_remain_stable() -> None:
    """Existing text-chat prompt constants should remain byte-for-byte stable."""
    for prompt_name, prompt_text, expected_bytes, expected_digest in (
        _PROMPT_FINGERPRINTS
    ):
        encoded_prompt = prompt_text.encode("utf-8")
        digest = hashlib.sha256(encoded_prompt).hexdigest()

        assert len(encoded_prompt) == expected_bytes, prompt_name
        assert digest == expected_digest, prompt_name
