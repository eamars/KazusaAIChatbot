"""Scripted-LLM fixtures for V3 integration tests.

Every fixture answers model calls from deterministic per-stage content, so
these tests exercise engine behavior without a live model or network access.
The invoker records each call in order with its routed stage name; a test may
override any stage's content through a mapping or a callable.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CognitionChainServicesV3,
)
from kazusa_ai_chatbot.llm_interface.contracts import (
    BackendDescriptor,
    LLMCallConfig,
    LLMResponse,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
)
from tests.test_cognition_chain_connector_mapping import _global_state

# Semantic owners routed through the six appraisal question families.
APPRAISAL_STAGE_NAMES = (
    "event_agency",
    "relationship_social",
    "moral_identity",
    "goal_threat_outcome",
    "epistemic_comparison_memory",
    "existential_drive",
)

# One dummy route config per semantic owner bound by CognitionChainServicesV3.
SERVICE_STAGE_NAMES = (
    *APPRAISAL_STAGE_NAMES,
    "goal_ordinary_response",
    "goal_active_branch",
    "workspace_collapse",
    "action_planning",
    "action_authorization",
    "resolver_authorization",
)

DEFAULT_APPRAISAL_CONTENT = json.dumps(
    {
        "event_agency": {"propositions": [], "deltas": []},
    }
)

# Dummy route limit; every scripted response is far shorter than this value.
DUMMY_MAX_COMPLETION_TOKENS = 8192

DEFAULT_ACTION_PLAN_CONTENT = json.dumps({
    "action_requests": [],
    "resolver_requests": [],
    "goal_resolution": "blocked",
    "resolver_pending_resolution": None,
    "resolver_goal_progress": None,
})


def episode_evidence_handle(payload: Mapping[str, Any]) -> str:
    """Return the handle of the current-episode evidence row.

    Raises KeyError when the payload carries no episode-sourced row, which is
    a programming error for the canonical fixture these tests run against.
    """
    for row in payload["evidence"]:
        if row["evidence_ref"]["source_kind"] == "episode":
            return row["evidence_handle"]
    raise KeyError("canonical fixture evidence has no episode-sourced row")


def ordinary_goal_draft(episode_handle: str) -> str:
    """Build the scripted ordinary-response goal draft as JSON text.

    The draft carries no target roles so its materialized bid keeps an empty
    role list, cites only the episode evidence handle, and reports a stable
    non-sensitive relational decision for the fixture's relationship state.
    """
    draft = {
        "intention": "Reply to the user's greeting in character.",
        "desired_outcome": "The user receives an in-character reply.",
        "concrete_detail": "Greet the user and open a topic of interest.",
        "reason": "The user opened the conversation with a simple greeting.",
        "private_monologue": "A quiet hello is an invitation, not a demand.",
        "target_role_handles": [],
        "evidence_handles": [episode_handle],
        "expected_consequences": ["The conversation continues from the greeting."],
        "confidence": "medium",
        "relational_willingness": {
            "applicability": "not_relationship_sensitive",
            "stance": "not_applicable",
            "current_user_relationship_state": "not_applicable",
            "reason": '关系状态稳定',
            "evidence_handles": [episode_handle],
        },
    }
    return json.dumps(draft)


def relationship_social_care_candidate(episode_handle: str) -> str:
    """Build the scripted relationship candidate carrying one care delta.

    The candidate selects the episode evidence row and reports a single
    integer care delta, so canonical runs exercise the exactly-once
    relationship carrier application through a unique receipt instead of an
    empty result.
    """
    candidate = {
        "explanation": "The greeting shows continued interest from the user.",
        "deltas": [
            {
                "path": "care",
                "value": 2,
                "reason": "The greeting keeps the relationship attentive.",
            }
        ],
        "propositions": [],
        "selected_evidence_handles": [episode_handle],
    }
    return json.dumps(candidate)


def default_scripted_responses(episode_handle: str) -> dict[str, str]:
    """Canonical scripted content for every engine stage over the fixture.

    Appraisal and terminal stages answer with bounded contentless results;
    the relationship family additionally carries one care delta so carrier
    assertions have substance; the ordinary goal stage answers with a stable
    accepted draft; action planning answers with an empty deterministic
    decision. The remaining owners are unreachable in the canonical run and
    default to empty text.
    """
    def empty_group(families):
        return {
            family: {"propositions": [], "deltas": []}
            for family in families
        }

    responses = {
        "A1": json.dumps(
            empty_group(
                ("event_agency", "goal_threat_outcome", "epistemic_comparison_memory")
            ),
            ensure_ascii=False,
        ),
        "A2": json.dumps(
            {
                "relationship_social": {
                    "propositions": [],
                    "deltas": [{
                        "target_path": "relationship.r1.care",
                        "delta": 2,
                        "evidence_handles": [episode_handle],
                        "reason": "The greeting keeps the relationship attentive.",
                    }],
                },
                "moral_identity": {"propositions": [], "deltas": []},
                "existential_drive": {"propositions": [], "deltas": []},
            },
            ensure_ascii=False,
        ),
        "G1a": ordinary_goal_draft(episode_handle),
        "P1": DEFAULT_ACTION_PLAN_CONTENT,
    }
    return responses


class ScriptedLLMInvoker:
    """Answer every engine model call from scripted per-stage content.

    Each ainvoke is recorded in order with the routed stage name and its
    zero-based attempt index within that stage; sync invoke raises because
    the engine only uses the async binding. Per-stage content resolution
    prefers an explicit mapping entry, then a callable answering for the
    stage, then the canonical default supplied at construction time.
    """

    def __init__(
        self,
        responses: Mapping[str, str] | Callable[[str, int], str] | None = None,
        defaults: Mapping[str, str] | None = None,
    ) -> None:
        self.calls: list[str] = []
        self.stage_attempts: dict[str, int] = {}
        self._responses: (
            Mapping[str, str] | Callable[[str, int], str] | None
        ) = responses or {}
        self._defaults: dict[str, str] = dict(defaults) if defaults else {}

    def _content_for(self, stage_name: str, attempt_index: int) -> str:
        """Resolve one stage's scripted content for the given attempt."""
        if callable(self._responses):
            return self._responses(stage_name, attempt_index)
        content = self._responses.get(stage_name)
        if content is not None:
            return content
        try:
            return self._defaults[stage_name]
        except KeyError as exc:
            raise KeyError(
                f"no scripted content for stage {stage_name!r}"
            ) from exc

    async def ainvoke(self, messages, *, config) -> LLMResponse:
        """Record the routed stage and answer with its scripted content."""
        del messages  # The prompt payload is not inspected by these tests.
        stage_name = config.stage_name.split(".")[0]
        attempt_index = self.stage_attempts.get(stage_name, 0)
        self.stage_attempts[stage_name] = attempt_index + 1
        self.calls.append(stage_name)
        content = self._content_for(stage_name, attempt_index)
        backend = BackendDescriptor(
            route_name=config.route_name,
            backend_kind="openai",
            model_family="test",
            model=config.model,
            normalized_base_url=config.base_url,
            thinking_strategy="none",
            confidence=1.0,
            generation=0,
        )
        response = LLMResponse(
            content=content,
            backend=backend,
            raw_response=None,
            usage={},
        )
        return response

    def invoke(self, messages, *, config):
        """Reject sync invocations; the engine only uses ainvoke."""
        raise AssertionError("the engine only uses the async binding")


def make_stage_config(stage_name: str) -> LLMCallConfig:
    """Build one dummy route config for a semantic owner stage."""
    stage_config = LLMCallConfig(
        stage_name=stage_name,
        route_name=f"route-{stage_name}",
        base_url="http://127.0.0.1:9",
        api_key=f"key-{stage_name}",
        model=f"model-{stage_name}",
        temperature=None,
        top_p=None,
        top_k=None,
        max_completion_tokens=DUMMY_MAX_COMPLETION_TOKENS,
        presence_penalty=None,
        context_window_tokens=50000,
    )
    return stage_config


def make_v3_services(
    invoker: ScriptedLLMInvoker,
    *,
    include_sidecar: bool = False,
    subconscious_enabled: bool = False,
) -> CognitionChainServicesV3:
    """Bind the scripted invoker to the requested V3 primary/sidecar lanes."""

    if subconscious_enabled and not include_sidecar:
        raise ValueError("subconscious tests require an injected sidecar lane")
    chain_lane = make_stage_config("cognition_core_v3.chain")
    services = CognitionChainServicesV3(
        llm=invoker,
        chain_lane=chain_lane,
        sidecar_lane=(
            make_stage_config("cognition_core_v3.sidecar")
            if include_sidecar
            else None
        ),
        subconscious_enabled=subconscious_enabled,
    )
    return services


@pytest.fixture()
def cognition_payload():
    """Canonical V2-shaped input built from the connector-mapping state."""
    payload = build_cognition_input_from_global_state(_global_state())
    return payload


@pytest.fixture()
def scripted_invoker(cognition_payload):
    """Scripted invoker answering every stage with canonical content."""
    handle = episode_evidence_handle(cognition_payload)
    invoker = ScriptedLLMInvoker(defaults=default_scripted_responses(handle))
    return invoker


@pytest.fixture()
def v3_services(scripted_invoker):
    """Engine services bound to the scripted invoker and dummy routes."""
    services = make_v3_services(scripted_invoker)
    return services
