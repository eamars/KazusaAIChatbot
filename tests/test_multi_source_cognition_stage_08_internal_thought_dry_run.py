"""Stage 08 internal-thought cognition dry-run tests."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from copy import deepcopy
from typing import Any, get_args

import pytest

from kazusa_ai_chatbot import internal_thought_cognition as dry_run_module
from kazusa_ai_chatbot.cognition_episode import validate_cognitive_episode
from kazusa_ai_chatbot.internal_thought_cognition import (
    InternalActionLatch,
    InternalThoughtCognitionDryRunAudit,
    InternalThoughtCognitionDryRunError,
    InternalThoughtDryRunOutputMode,
    InternalThoughtDryRunStatus,
    InternalThoughtResidue,
    PublicSceneResidue,
    build_internal_thought_cognitive_episode,
    run_internal_thought_cognition_dry_run,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l1 as l1_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2 as l2_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2d as l2d_module
from kazusa_ai_chatbot.cognition_chain_core.action_selection_prompt import (
    ACTION_ROUTER_PROMPT,
)
from kazusa_ai_chatbot.cognition_chain_core.stages import l3 as l3_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2c2 as l2c2_module
from kazusa_ai_chatbot.cognition_chain_core.prompt_selection import (
    CognitionPromptSelectionError,
    CognitionPromptStage,
    build_cognition_prompt_source_payload,
    select_cognition_prompt_variant,
)


_APPROVED_STAGES: tuple[CognitionPromptStage, ...] = (
    "l1_subconscious",
    "l2a_conscious_framing",
    "l2b_boundary_appraisal",
    "l2c1_judgment_synthesis",
    "l2d_action_selection",
    "l2c2_social_context_appraisal",
    "l3_style_agent",
    "l3_content_plan_agent",
    "l3_preference_adapter",
    "l3_visual_agent",
)
_EXPECTED_PROMPT_KEYS = [
    "l1_subconscious.internal_thought_internal_monologue",
    "l2a_conscious_framing.internal_thought_internal_monologue",
    "l2b_boundary_appraisal.internal_thought_internal_monologue",
    "l2c1_judgment_synthesis.internal_thought_internal_monologue",
    "l2c2_social_context_appraisal.internal_thought_internal_monologue",
    "l2d_action_selection.internal_thought_internal_monologue",
]
_EXPECTED_PROMPT_KEYS_WITHOUT_VISUAL = list(_EXPECTED_PROMPT_KEYS)
_FORBIDDEN_DRY_RUN_TOKENS = (
    "call_consolidation_subgraph",
    "save_conversation",
    "save_assistant_message",
    "dispatcher.dispatch",
    "runtime.invalidate",
    "get_rag_cache2_runtime",
    "schedule",
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
        "ACTION_ROUTER_PROMPT",
        ACTION_ROUTER_PROMPT,
        23173,
        "71d43a9066182d82e21806a0923d24b325c31937583519faa970fca930b5ab59",
    ),
    (
        "_STYLE_AGENT_PROMPT",
        l3_module._STYLE_AGENT_PROMPT,
        9438,
        "5f87f9353423815b947cebb707de8dc4623867741b62ad635103692c498a9851",
    ),
    (
        "_CONTENT_PLAN_AGENT_PROMPT",
        l3_module._CONTENT_PLAN_AGENT_PROMPT,
        10425,
        "38b4bbe41532b54c029538f0a63843101184191f413994250d28207a5e1ec19f",
    ),
    (
        "_PREFERENCE_ADAPTER_PROMPT",
        l3_module._PREFERENCE_ADAPTER_PROMPT,
        4865,
        "a2396cb3e0979941375b0599dfdf9331a86e577988f51bc80da6491fd3225bde",
    ),
    (
        "_VISUAL_AGENT_PROMPT",
        l3_module._VISUAL_AGENT_PROMPT,
        7936,
        "28e8ac752d123f76a55b0703a61d292e55aca49cf0ea6d9efd80c0f7f5775b65",
    ),
)


@pytest.fixture(autouse=True)
def _default_visual_directives_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep default dry-run tests independent from local service config."""

    monkeypatch.setattr(
        dry_run_module,
        "COGNITION_VISUAL_DIRECTIVES_ENABLED",
        True,
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


_COGNITION_LLM_DISPATCH_MAP = {
    "的潜意识层": "_subconscious_llm",
    "的意识层": "_conscious_llm",
    "的裁决核心": "_judgement_core_llm",
    "的社交观察脑": "_contextual_agent_llm",
}


class _DispatchingAsyncLLM:
    """Fake LLM that dispatches to per-stage fakes based on system prompt."""

    def __init__(self, stage_llms: dict[str, _CapturingAsyncLLM]) -> None:
        self.stage_llms = stage_llms
        self._model_key = ("fake", "dispatching")

    async def ainvoke(self, messages: list[Any]) -> _DummyResponse:
        system_text = messages[0].content if messages else ""
        for marker, llm_name in _COGNITION_LLM_DISPATCH_MAP.items():
            if marker in system_text:
                return await self.stage_llms[llm_name].ainvoke(messages)
        return await self.stage_llms["_subconscious_llm"].ainvoke(messages)


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


def _residue() -> dict[str, str]:
    """Build a valid internal-thought residue fixture.

    Returns:
        Internal-thought residue with model-visible private monologue content.
    """
    residue = {
        "residue_id": "internal:residue:quiet-room",
        "internal_monologue": "Notice the quiet room before deciding anything.",
        "source": "runtime_internal_thought",
    }
    return residue


def _action_latch() -> dict[str, str]:
    """Build an audit-only action-latch fixture.

    Returns:
        Action latch that must remain non-executable in Stage 08.
    """
    latch = {
        "latch_id": "internal:latch:pause",
        "action_text": "Pause before speaking.",
        "latch_reason": "Avoid turning a private thought into public action.",
        "status": "audit_only",
    }
    return latch


def _canonical_episode_content(
    *,
    residue: dict[str, str],
    action_latch: dict[str, str] | None = None,
) -> str:
    """Render the expected internal-thought percept content.

    Args:
        residue: Internal-thought residue included in the percept.
        action_latch: Optional audit-only action latch included in the percept.

    Returns:
        Canonical JSON string expected in the episode percept content.
    """
    content_payload = {
        "residue": residue,
        "action_latch": action_latch or {},
    }
    rendered = json.dumps(
        content_payload,
        sort_keys=True,
        ensure_ascii=False,
    )
    return rendered


def _expected_episode_id(content: str) -> str:
    """Return the deterministic internal-thought dry-run episode id.

    Args:
        content: Canonical internal-thought percept content.

    Returns:
        Expected episode id derived from the content digest.
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    episode_id = f"internal_thought:dry_run:{digest}"
    return episode_id


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
        "global_user_id": "internal_thought",
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
            "interaction_subtext": "internal thought",
        },
        "_conscious_llm": {
            "internal_monologue": "Keep the private thought private.",
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
            "judgment_note": "internal thought dry run remains private",
        },
        "_action_selection_llm": {
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
        "_content_plan_agent_llm": {
            "content_plan": {
                "semantic_content": "Keep the internal-thought dry run private.",
                "rendering": "No visible reply is emitted.",
            },
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
        "_action_selection_llm",
        llms["_action_selection_llm"],
    )
    monkeypatch.setattr(
        l2c2_module,
        "_contextual_agent_llm",
        llms["_contextual_agent_llm"],
    )
    monkeypatch.setattr(l3_module, "_style_agent_llm", llms["_style_agent_llm"])
    monkeypatch.setattr(
        l3_module,
        "_content_plan_agent_llm",
        llms["_content_plan_agent_llm"],
    )
    monkeypatch.setattr(
        l3_module,
        "_preference_adapter_llm",
        llms["_preference_adapter_llm"],
    )
    monkeypatch.setattr(l3_module, "_visual_agent_llm", llms["_visual_agent_llm"])
    from kazusa_ai_chatbot.cognition_chain_core.contracts import (
        CognitionChainServices,
    )
    from kazusa_ai_chatbot.utils import parse_llm_json_output

    cognition_dispatch = _DispatchingAsyncLLM({
        "_subconscious_llm": llms["_subconscious_llm"],
        "_conscious_llm": llms["_conscious_llm"],
        "_judgement_core_llm": llms["_judgement_core_llm"],
        "_contextual_agent_llm": llms["_contextual_agent_llm"],
    })
    fake_services = CognitionChainServices(
        cognition_llm=cognition_dispatch,
        boundary_core_llm=llms["_boundary_core_llm"],
        action_selection_llm=llms["_action_selection_llm"],
        style_llm=llms["_style_agent_llm"],
        content_plan_llm=llms["_content_plan_agent_llm"],
        preference_llm=llms["_preference_adapter_llm"],
        visual_llm=llms["_visual_agent_llm"],
        parse_json=parse_llm_json_output,
        logger=logging.getLogger(__name__),
    )
    monkeypatch.setattr(
        cognition_module,
        "build_cognition_chain_services",
        lambda: fake_services,
    )
    return llms


def _internal_thought_dry_run_state() -> dict[str, Any]:
    """Build a dry-run state for prompt rendering through the cognition graph.

    Returns:
        Global persona state fixture for the internal-thought dry run.
    """
    episode = build_internal_thought_cognitive_episode(
        residue=_residue(),
        action_latch=_action_latch(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        output_mode="think_only",
    )
    state = {
        "character_profile": _character_profile(),
        "storage_timestamp_utc": "2026-05-09T21:30:00+00:00",
        "local_time_context": _time_context(),
        "user_input": "Internal thought dry run over private cognition residue.",
        "prompt_message_context": {
            "body_text": "",
            "addressed_to_global_user_ids": [],
            "broadcast": False,
            "mentions": [],
            "attachments": [],
        },
        "cognitive_episode": episode,
        "user_multimedia_input": [],
        "platform": "internal_thought",
        "platform_channel_id": "internal_thought_dry_run",
        "channel_type": "internal_thought_dry_run",
        "platform_message_id": "internal_thought:dry_run",
        "platform_user_id": "internal_thought",
        "global_user_id": "internal_thought",
        "user_name": "internal_thought",
        "user_profile": _user_profile(),
        "platform_bot_id": "internal_thought",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "",
        "promoted_reflection_context": {},
        "debug_modes": {"think_only": True, "no_remember": True},
        "should_respond": False,
        "decontexualized_input": (
            "Internal thought dry run over private cognition residue."
        ),
        "referents": [],
        "rag_result": _expected_empty_rag_result(),
        "internal_monologue": "",
        "action_directives": {},
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "character_intent": "",
        "logical_stance": "",
        "final_dialog": [],
        "mood": "",
        "global_vibe": "",
        "reflection_summary": "",
        "subjective_appraisals": [],
        "affinity_delta": 0,
        "last_relationship_insight": "",
        "new_facts": [],
        "future_promises": [],
    }
    return state


def test_public_contract_names_are_available() -> None:
    """Module should expose the exact public Stage 08 dry-run API."""
    assert set(get_args(InternalThoughtDryRunOutputMode)) == {
        "think_only",
        "preview",
        "silent",
    }
    assert set(get_args(InternalThoughtDryRunStatus)) == {
        "completed",
        "skipped_busy",
        "skipped_empty_residue",
    }
    assert issubclass(InternalThoughtCognitionDryRunError, ValueError)
    assert callable(build_internal_thought_cognitive_episode)
    assert callable(run_internal_thought_cognition_dry_run)
    assert InternalThoughtResidue.__required_keys__ == {
        "residue_id",
        "internal_monologue",
        "source",
    }
    assert InternalActionLatch.__required_keys__ == {
        "latch_id",
        "action_text",
        "latch_reason",
        "status",
    }
    assert PublicSceneResidue.__required_keys__ == {
        "scene_residue_id",
        "source_episode_id",
        "summary",
        "visibility",
        "merge_status",
    }
    assert InternalThoughtCognitionDryRunAudit.__required_keys__ == {
        "status",
        "skip_reason",
        "cognition_called",
        "episode_id",
        "residue_id",
        "action_latch_id",
        "trigger_source",
        "input_sources",
        "output_mode",
        "prompt_variant",
        "prompt_keys",
        "cognition_output_keys",
    }


def test_internal_thought_builder_creates_valid_episode() -> None:
    """Builder should create the exact internal_thought episode contract."""
    residue = _residue()
    latch = _action_latch()
    content = _canonical_episode_content(
        residue=residue,
        action_latch=latch,
    )

    episode = build_internal_thought_cognitive_episode(
        residue=residue,
        action_latch=latch,
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        output_mode="preview",
    )

    validate_cognitive_episode(episode)

    assert episode["episode_id"] == _expected_episode_id(content)
    assert episode["trigger_source"] == "internal_thought"
    assert episode["input_sources"] == ["internal_monologue"]
    assert episode["output_mode"] == "preview"
    assert episode["storage_timestamp_utc"] == "2026-05-09T21:30:00+00:00"
    assert episode["local_time_context"] == _time_context()
    assert episode["target_scope"] == {
        "platform": "internal_thought",
        "platform_channel_id": "internal_thought_dry_run",
        "channel_type": "internal_thought_dry_run",
        "current_platform_user_id": "internal_thought",
        "current_global_user_id": "internal_thought",
        "current_display_name": "internal_thought",
        "target_addressed_user_ids": [],
        "target_broadcast": False,
    }
    assert episode["origin_metadata"] == {
        "platform": "internal_thought",
        "platform_message_id": "internal_thought:dry_run",
        "active_turn_platform_message_ids": [],
        "active_turn_conversation_row_ids": [],
        "debug_modes": {"think_only": True, "no_remember": True},
    }
    assert episode["percepts"] == [
        {
            "percept_id": "internal_thought:percept:internal_monologue",
            "input_source": "internal_monologue",
            "content": content,
            "visibility": "model_visible",
            "metadata": {"source": "runtime_internal_thought"},
        },
    ]


def test_internal_thought_builder_can_disable_visual_directives() -> None:
    """Builder parameter should seed the internal visual skip flag."""
    episode = build_internal_thought_cognitive_episode(
        residue=_residue(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        visual_directives_enabled=False,
    )

    assert episode["origin_metadata"]["debug_modes"] == {
        "think_only": True,
        "no_remember": True,
        "no_visual_directives": True,
    }


def test_internal_thought_builder_obeys_global_visual_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global config false should disable visual directives for dry-run episodes."""
    monkeypatch.setattr(
        dry_run_module,
        "COGNITION_VISUAL_DIRECTIVES_ENABLED",
        False,
    )

    episode = dry_run_module.build_internal_thought_cognitive_episode(
        residue=_residue(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
    )

    assert episode["origin_metadata"]["debug_modes"] == {
        "think_only": True,
        "no_remember": True,
        "no_visual_directives": True,
    }


def test_internal_thought_builder_uses_empty_latch_payload_when_absent() -> None:
    """Builder should encode absent action latch as an empty object."""
    residue = _residue()
    content = _canonical_episode_content(residue=residue)

    episode = build_internal_thought_cognitive_episode(
        residue=residue,
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
    )

    assert episode["episode_id"] == _expected_episode_id(content)
    assert episode["percepts"][0]["content"] == content


def test_internal_thought_builder_rejects_unsupported_output_mode() -> None:
    """Builder should fail closed for output modes outside the dry-run set."""
    with pytest.raises(
        InternalThoughtCognitionDryRunError,
        match="internal thought output_mode is not supported",
    ):
        build_internal_thought_cognitive_episode(
            residue=_residue(),
            storage_timestamp_utc="2026-05-09T21:30:00+00:00",
            local_time_context=_time_context(),
            output_mode="visible_reply",
        )


def test_internal_thought_builder_rejects_empty_residue_text() -> None:
    """Builder should reject private residue with no monologue text."""
    residue = dict(_residue())
    residue["internal_monologue"] = ""

    with pytest.raises(
        InternalThoughtCognitionDryRunError,
        match="internal_monologue is empty",
    ):
        build_internal_thought_cognitive_episode(
            residue=residue,
            storage_timestamp_utc="2026-05-09T21:30:00+00:00",
            local_time_context=_time_context(),
        )


def test_internal_thought_builder_rejects_wrong_residue_source() -> None:
    """Builder should reject residue not produced by runtime internal thought."""
    residue = dict(_residue())
    residue["source"] = "reflection_signal"

    with pytest.raises(
        InternalThoughtCognitionDryRunError,
        match="residue.source is not supported",
    ):
        build_internal_thought_cognitive_episode(
            residue=residue,
            storage_timestamp_utc="2026-05-09T21:30:00+00:00",
            local_time_context=_time_context(),
        )


def test_internal_thought_builder_rejects_over_cap_monologue() -> None:
    """Builder should reject private monologue before prompt construction."""
    residue = dict(_residue())
    residue["internal_monologue"] = "x" * 4001

    with pytest.raises(
        InternalThoughtCognitionDryRunError,
        match="internal_monologue exceeds 4000 characters",
    ):
        build_internal_thought_cognitive_episode(
            residue=residue,
            storage_timestamp_utc="2026-05-09T21:30:00+00:00",
            local_time_context=_time_context(),
        )


def test_internal_thought_builder_rejects_non_audit_latch() -> None:
    """Builder should reject action latch statuses that imply execution."""
    latch = dict(_action_latch())
    latch["status"] = "ready"

    with pytest.raises(
        InternalThoughtCognitionDryRunError,
        match="action_latch.status is not supported",
    ):
        build_internal_thought_cognitive_episode(
            residue=_residue(),
            action_latch=latch,
            storage_timestamp_utc="2026-05-09T21:30:00+00:00",
            local_time_context=_time_context(),
        )


def test_internal_thought_builder_rejects_over_cap_latch_text() -> None:
    """Builder should reject oversized action text before cognition."""
    latch = dict(_action_latch())
    latch["action_text"] = "x" * 1001

    with pytest.raises(
        InternalThoughtCognitionDryRunError,
        match="action_latch.action_text exceeds 1000 characters",
    ):
        build_internal_thought_cognitive_episode(
            residue=_residue(),
            action_latch=latch,
            storage_timestamp_utc="2026-05-09T21:30:00+00:00",
            local_time_context=_time_context(),
        )


def test_selector_returns_internal_thought_variant_for_every_stage() -> None:
    """Internal-thought episodes should select the exact dry-run variant."""
    episode = build_internal_thought_cognitive_episode(
        residue=_residue(),
        action_latch=_action_latch(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        output_mode="silent",
    )

    for stage in _APPROVED_STAGES:
        selection = select_cognition_prompt_variant(
            episode=episode,
            stage=stage,
        )

        assert selection == {
            "stage": stage,
            "variant": "internal_thought_internal_monologue",
            "prompt_key": f"{stage}.internal_thought_internal_monologue",
            "trigger_source": "internal_thought",
            "input_sources": ["internal_monologue"],
            "output_mode": "silent",
        }


def test_source_payload_projects_internal_thought_residue() -> None:
    """Internal-thought payload should expose only residue source content."""
    residue = _residue()
    latch = _action_latch()
    episode = build_internal_thought_cognitive_episode(
        residue=residue,
        action_latch=latch,
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
    )
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l1_subconscious",
    )

    payload = build_cognition_prompt_source_payload(
        episode=episode,
        selection=selection,
    )

    assert payload == {
        "internal_thought_residue": {
            "residue_id": residue["residue_id"],
            "internal_monologue": residue["internal_monologue"],
            "action_latch": latch,
        },
    }
    assert "cognitive_episode" not in payload
    assert "trigger_source" not in payload
    assert "input_sources" not in payload


def test_source_payload_projects_empty_action_latch_when_absent() -> None:
    """Internal-thought payload should preserve absent latch as empty mapping."""
    residue = _residue()
    episode = build_internal_thought_cognitive_episode(
        residue=residue,
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
    )
    selection = select_cognition_prompt_variant(
        episode=episode,
        stage="l1_subconscious",
    )

    payload = build_cognition_prompt_source_payload(
        episode=episode,
        selection=selection,
    )

    assert payload == {
        "internal_thought_residue": {
            "residue_id": residue["residue_id"],
            "internal_monologue": residue["internal_monologue"],
            "action_latch": {},
        },
    }


def test_source_payload_rejects_duplicate_internal_monologue_percepts() -> None:
    """Internal-thought payload projection should require one percept."""
    episode = build_internal_thought_cognitive_episode(
        residue=_residue(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
    )
    second_percept = deepcopy(episode["percepts"][0])
    second_percept["percept_id"] = "internal_thought:percept:duplicate"
    episode["percepts"].append(second_percept)
    selection = {
        "stage": "l1_subconscious",
        "variant": "internal_thought_internal_monologue",
        "prompt_key": "l1_subconscious.internal_thought_internal_monologue",
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": "think_only",
    }

    with pytest.raises(CognitionPromptSelectionError, match="internal_monologue"):
        build_cognition_prompt_source_payload(
            episode=episode,
            selection=selection,
        )


def test_source_payload_rejects_malformed_internal_thought_json() -> None:
    """Internal-thought payload projection should require canonical JSON."""
    episode = build_internal_thought_cognitive_episode(
        residue=_residue(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
    )
    episode["percepts"][0]["content"] = "{not-json"
    selection = {
        "stage": "l1_subconscious",
        "variant": "internal_thought_internal_monologue",
        "prompt_key": "l1_subconscious.internal_thought_internal_monologue",
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": "think_only",
    }

    with pytest.raises(
        CognitionPromptSelectionError,
        match="internal_monologue percept content is malformed",
    ):
        build_cognition_prompt_source_payload(
            episode=episode,
            selection=selection,
        )


def test_source_payload_rejects_non_string_action_latch_values() -> None:
    """Internal-thought payload projection should require string latch values."""
    episode = build_internal_thought_cognitive_episode(
        residue=_residue(),
        action_latch=_action_latch(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
    )
    content_payload = json.loads(episode["percepts"][0]["content"])
    content_payload["action_latch"]["action_text"] = ["not", "a", "string"]
    episode["percepts"][0]["content"] = json.dumps(
        content_payload,
        sort_keys=True,
        ensure_ascii=False,
    )
    selection = {
        "stage": "l1_subconscious",
        "variant": "internal_thought_internal_monologue",
        "prompt_key": "l1_subconscious.internal_thought_internal_monologue",
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": "think_only",
    }

    with pytest.raises(
        CognitionPromptSelectionError,
        match="action_latch values must be strings",
    ):
        build_cognition_prompt_source_payload(
            episode=episode,
            selection=selection,
        )


def test_selector_rejects_internal_thought_visible_reply_output_mode() -> None:
    """Internal-thought prompt selection should fail closed for visible reply."""
    episode = build_internal_thought_cognitive_episode(
        residue=_residue(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
    )
    episode["output_mode"] = "visible_reply"

    with pytest.raises(CognitionPromptSelectionError, match="output_mode"):
        select_cognition_prompt_variant(
            episode=episode,
            stage="l1_subconscious",
        )


def test_selector_rejects_internal_thought_dialog_text_source() -> None:
    """Internal-thought prompt selection should fail closed for dialog text."""
    episode = build_internal_thought_cognitive_episode(
        residue=_residue(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
    )
    episode["input_sources"] = ["dialog_text"]
    episode["percepts"][0]["input_source"] = "dialog_text"

    with pytest.raises(CognitionPromptSelectionError, match="input_sources"):
        select_cognition_prompt_variant(
            episode=episode,
            stage="l1_subconscious",
        )


def test_selector_rejects_non_internal_trigger_source() -> None:
    """Internal-thought source should not be selected for other triggers."""
    episode = build_internal_thought_cognitive_episode(
        residue=_residue(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
    )
    episode["trigger_source"] = "scheduled_recall"

    with pytest.raises(CognitionPromptSelectionError, match="trigger_source"):
        select_cognition_prompt_variant(
            episode=episode,
            stage="l1_subconscious",
        )


@pytest.mark.asyncio
async def test_dry_run_returns_busy_skip_without_cognition_call() -> None:
    """Busy primary interaction should skip without building cognition state."""
    residue = _residue()
    latch = _action_latch()
    busy_probe = _BusyProbe(True)
    cognition = _CapturingCognitionCallable({"unused": "value"})

    audit = await run_internal_thought_cognition_dry_run(
        residue=residue,
        action_latch=latch,
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
        "status": "skipped_busy",
        "skip_reason": "primary_interaction_busy",
        "cognition_called": False,
        "episode_id": "",
        "residue_id": residue["residue_id"],
        "action_latch_id": latch["latch_id"],
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": "silent",
        "prompt_variant": "internal_thought_internal_monologue",
        "prompt_keys": [],
        "cognition_output_keys": [],
    }


@pytest.mark.asyncio
async def test_dry_run_rejects_output_mode_before_busy_probe() -> None:
    """Runner should validate output mode before probing primary load."""
    busy_probe = _BusyProbe(True)
    cognition = _CapturingCognitionCallable({"unused": "value"})

    with pytest.raises(
        InternalThoughtCognitionDryRunError,
        match="internal thought output_mode is not supported",
    ):
        await run_internal_thought_cognition_dry_run(
            residue=_residue(),
            character_profile=_character_profile(),
            user_profile=_user_profile(),
            storage_timestamp_utc="2026-05-09T21:30:00+00:00",
            local_time_context=_time_context(),
            is_primary_interaction_busy=busy_probe,
            call_cognition_subgraph_func=cognition,
            output_mode="visible_reply",
        )

    assert busy_probe.call_count == 0
    assert cognition.calls == 0


@pytest.mark.asyncio
async def test_dry_run_returns_empty_residue_skip_without_cognition_call() -> None:
    """Empty private monologue should skip without raising or calling cognition."""
    residue = dict(_residue())
    residue["internal_monologue"] = ""
    busy_probe = _BusyProbe(False)
    cognition = _CapturingCognitionCallable({"unused": "value"})

    audit = await run_internal_thought_cognition_dry_run(
        residue=residue,
        character_profile=_character_profile(),
        user_profile=_user_profile(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        is_primary_interaction_busy=busy_probe,
        call_cognition_subgraph_func=cognition,
        output_mode="think_only",
    )

    assert busy_probe.call_count == 1
    assert cognition.calls == 0
    assert audit == {
        "status": "skipped_empty_residue",
        "skip_reason": "internal_thought_residue_empty",
        "cognition_called": False,
        "episode_id": "",
        "residue_id": residue["residue_id"],
        "action_latch_id": "",
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": "think_only",
        "prompt_variant": "internal_thought_internal_monologue",
        "prompt_keys": [],
        "cognition_output_keys": [],
    }


@pytest.mark.asyncio
async def test_dry_run_rejects_over_cap_residue_before_cognition_call() -> None:
    """Runner should reject over-cap private monologue before cognition."""
    residue = dict(_residue())
    residue["internal_monologue"] = "x" * 4001
    busy_probe = _BusyProbe(False)
    cognition = _CapturingCognitionCallable({"unused": "value"})

    with pytest.raises(
        InternalThoughtCognitionDryRunError,
        match="internal_monologue exceeds 4000 characters",
    ):
        await run_internal_thought_cognition_dry_run(
            residue=residue,
            character_profile=_character_profile(),
            user_profile=_user_profile(),
            storage_timestamp_utc="2026-05-09T21:30:00+00:00",
            local_time_context=_time_context(),
            is_primary_interaction_busy=busy_probe,
            call_cognition_subgraph_func=cognition,
            output_mode="think_only",
        )

    assert busy_probe.call_count == 1
    assert cognition.calls == 0


@pytest.mark.asyncio
async def test_dry_run_calls_injected_cognition_once_and_returns_audit() -> None:
    """Non-busy dry run should build exact state and completed audit."""
    residue = _residue()
    latch = _action_latch()
    content = _canonical_episode_content(
        residue=residue,
        action_latch=latch,
    )
    busy_probe = _BusyProbe(False)
    cognition = _CapturingCognitionCallable({
        "zeta": "last",
        "alpha": "first",
    })

    audit = await run_internal_thought_cognition_dry_run(
        residue=residue,
        action_latch=latch,
        character_profile=_character_profile(),
        user_profile=_user_profile(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        is_primary_interaction_busy=busy_probe,
        call_cognition_subgraph_func=cognition,
        output_mode="preview",
    )

    assert busy_probe.call_count == 1
    assert cognition.calls == 1
    assert audit == {
        "status": "completed",
        "skip_reason": "",
        "cognition_called": True,
        "episode_id": _expected_episode_id(content),
        "residue_id": residue["residue_id"],
        "action_latch_id": latch["latch_id"],
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": "preview",
        "prompt_variant": "internal_thought_internal_monologue",
        "prompt_keys": _EXPECTED_PROMPT_KEYS,
        "cognition_output_keys": ["alpha", "zeta"],
    }
    assert set(audit) == InternalThoughtCognitionDryRunAudit.__required_keys__
    assert "public_scene_residue" not in audit

    dry_run_state = cognition.states[0]
    assert dry_run_state["user_input"] == (
        "Internal thought dry run over private cognition residue."
    )
    assert dry_run_state["prompt_message_context"] == {
        "body_text": "",
        "addressed_to_global_user_ids": [],
        "broadcast": False,
        "mentions": [],
        "attachments": [],
    }
    assert dry_run_state["user_multimedia_input"] == []
    assert dry_run_state["platform"] == "internal_thought"
    assert dry_run_state["platform_channel_id"] == "internal_thought_dry_run"
    assert dry_run_state["channel_type"] == "internal_thought_dry_run"
    assert dry_run_state["platform_message_id"] == "internal_thought:dry_run"
    assert dry_run_state["platform_user_id"] == "internal_thought"
    assert dry_run_state["global_user_id"] == "internal_thought"
    assert dry_run_state["user_name"] == "internal_thought"
    assert dry_run_state["platform_bot_id"] == "internal_thought"
    assert dry_run_state["chat_history_wide"] == []
    assert dry_run_state["chat_history_recent"] == []
    assert dry_run_state["reply_context"] == {}
    assert dry_run_state["indirect_speech_context"] == ""
    assert dry_run_state["channel_topic"] == ""
    assert dry_run_state["promoted_reflection_context"] == {}
    assert dry_run_state["debug_modes"] == {
        "think_only": True,
        "no_remember": True,
    }
    assert dry_run_state["should_respond"] is False
    assert dry_run_state["decontexualized_input"] == (
        "Internal thought dry run over private cognition residue."
    )
    assert dry_run_state["referents"] == []
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
    assert "public_scene_residue" not in dry_run_state
    assert "consolidation_state" not in dry_run_state

    public_surfaces = {
        "prompt_message_context": dry_run_state["prompt_message_context"],
        "chat_history_wide": dry_run_state["chat_history_wide"],
        "chat_history_recent": dry_run_state["chat_history_recent"],
        "reply_context": dry_run_state["reply_context"],
        "rag_result": dry_run_state["rag_result"],
        "final_dialog": dry_run_state["final_dialog"],
        "new_facts": dry_run_state["new_facts"],
        "future_promises": dry_run_state["future_promises"],
        "action_directives": dry_run_state["action_directives"],
    }
    public_surface_text = json.dumps(public_surfaces, ensure_ascii=False)
    assert residue["internal_monologue"] not in public_surface_text
    assert latch["action_text"] not in public_surface_text


@pytest.mark.asyncio
async def test_dry_run_visual_disable_omits_prompt_key_and_sets_state_flag() -> None:
    """Per-run visual disable should update audit keys and cognition state."""
    residue = _residue()
    busy_probe = _BusyProbe(False)
    cognition = _CapturingCognitionCallable({
        "alpha": "first",
    })

    audit = await run_internal_thought_cognition_dry_run(
        residue=residue,
        character_profile=_character_profile(),
        user_profile=_user_profile(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        is_primary_interaction_busy=busy_probe,
        call_cognition_subgraph_func=cognition,
        visual_directives_enabled=False,
    )

    assert cognition.calls == 1
    assert audit["prompt_keys"] == _EXPECTED_PROMPT_KEYS_WITHOUT_VISUAL
    assert "l3_visual_agent.internal_thought_internal_monologue" not in (
        audit["prompt_keys"]
    )
    dry_run_state = cognition.states[0]
    assert dry_run_state["debug_modes"] == {
        "think_only": True,
        "no_remember": True,
        "no_visual_directives": True,
    }
    assert dry_run_state["cognitive_episode"]["origin_metadata"][
        "debug_modes"
    ] == dry_run_state["debug_modes"]


@pytest.mark.asyncio
async def test_dry_run_global_visual_config_disables_prompt_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global config false should disable visual prompt audit by default."""
    monkeypatch.setattr(
        dry_run_module,
        "COGNITION_VISUAL_DIRECTIVES_ENABLED",
        False,
    )
    cognition = _CapturingCognitionCallable({
        "alpha": "first",
    })

    audit = await dry_run_module.run_internal_thought_cognition_dry_run(
        residue=_residue(),
        character_profile=_character_profile(),
        user_profile=_user_profile(),
        storage_timestamp_utc="2026-05-09T21:30:00+00:00",
        local_time_context=_time_context(),
        is_primary_interaction_busy=_BusyProbe(False),
        call_cognition_subgraph_func=cognition,
    )

    assert audit["prompt_keys"] == _EXPECTED_PROMPT_KEYS_WITHOUT_VISUAL
    assert cognition.states[0]["debug_modes"] == {
        "think_only": True,
        "no_remember": True,
        "no_visual_directives": True,
    }


def test_dry_run_module_contains_no_write_delivery_or_scheduler_calls() -> None:
    """Dry-run module should not reference write or delivery entrypoints."""
    module_source = inspect.getsource(dry_run_module)

    for token in _FORBIDDEN_DRY_RUN_TOKENS:
        assert token not in module_source


@pytest.mark.asyncio
async def test_internal_thought_prompt_rendering_uses_only_residue_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mocked cognition prompts should receive only internal residue content."""
    residue = _residue()
    latch = _action_latch()
    llms = _patch_cognition_llms(monkeypatch)
    monkeypatch.setattr(
        l3_module,
        "call_interaction_style_context_loader",
        _fake_build_interaction_style_context,
    )

    result = await cognition_module.call_cognition_subgraph(
        _internal_thought_dry_run_state(),
    )

    assert sorted(result) == [
        "action_specs",
        "character_intent",
        "emotional_appraisal",
        "emotional_intensity",
        "interaction_subtext",
        "internal_monologue",
        "judgment_note",
        "logical_stance",
        "rag_result",
        "relational_dynamic",
        "resolver_capability_requests",
        "social_distance",
        "vibe_check",
    ]
    assert result["rag_result"] == _expected_empty_rag_result()
    assert result["resolver_capability_requests"] == []
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
        assert prompt_payload["internal_thought_residue"] == {
            "residue_id": residue["residue_id"],
            "internal_monologue": residue["internal_monologue"],
            "action_latch": latch,
        }
        assert "cognitive_episode" not in prompt_payload
        assert "trigger_source" not in prompt_payload
        assert "input_sources" not in prompt_payload
        assert "reflection_artifact" not in prompt_payload
        assert "hourly_reflections" not in prompt_payload
        assert "daily_syntheses" not in prompt_payload
        assert "source_message_refs" not in prompt_payload
        assert "raw_reflection_run" not in prompt_payload
        if "promoted_reflection_context" in prompt_payload:
            assert prompt_payload["promoted_reflection_context"] == {}

    l2d_llm = llms["_action_selection_llm"]
    assert l2d_llm.messages
    l2d_context = l2d_llm.messages[1].content
    l2d_payload = json.loads(l2d_context)
    assert l2d_payload["source"]["trigger_source"] == "internal_thought"
    assert "internal_monologue" in l2d_payload["source"]["input_sources"]
    assert l2d_payload["cognition"]["social_distance"] == "neutral"
    assert l2d_payload["cognition"]["emotional_intensity"] == "low"
    assert l2d_payload["cognition"]["vibe_check"] == "quiet"
    assert l2d_payload["cognition"]["relational_dynamic"] == "stable"
    assert "cognitive_episode" not in l2d_context
    assert "raw_reflection_run" not in l2d_context
    assert "available_capabilities" not in l2d_context

    invoked_llm_names = (
        *l1_l2_llm_names,
        "_contextual_agent_llm",
        "_action_selection_llm",
    )
    for llm_name, fake_llm in llms.items():
        if llm_name not in invoked_llm_names:
            assert fake_llm.messages == []


def test_text_chat_and_reflection_prompt_fingerprints_remain_stable() -> None:
    """Existing prompt constants should remain byte-for-byte stable."""
    for prompt_name, prompt_text, expected_bytes, expected_digest in (
        _PROMPT_FINGERPRINTS
    ):
        encoded_prompt = prompt_text.encode("utf-8")
        digest = hashlib.sha256(encoded_prompt).hexdigest()

        assert len(encoded_prompt) == expected_bytes, prompt_name
        assert digest == expected_digest, prompt_name
