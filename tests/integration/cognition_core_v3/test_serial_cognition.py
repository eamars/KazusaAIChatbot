"""Deterministic coverage for the serialized primary cognition path."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.cognition_shared.contracts import (
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    bind_v2_attempt_ledger,
    create_v2_attempt_ledger,
    reset_v2_attempt_ledger,
)
from kazusa_ai_chatbot.cognition_core_v3 import facade as v3_facade
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


def _empty_family_rows(families):
    return {
        family: {"propositions": [], "deltas": []}
        for family in families
    }


class SerialScriptedLLM:
    """Answer by serial step identity instead of parallel stage identity."""

    def __init__(self, episode_handle: str) -> None:
        self.episode_handle = episode_handle
        self.calls: list[str] = []

    async def ainvoke(self, messages, *, config):
        del messages
        step = config.stage_name.split(".")[0]
        self.calls.append(step)
        if step == "A1":
            content = json.dumps(
                _empty_family_rows(
                    (
                        "event_agency",
                        "goal_threat_outcome",
                        "epistemic_comparison_memory",
                    )
                ),
                ensure_ascii=False,
            )
        elif step == "A2":
            content = json.dumps(
                _empty_family_rows(
                    (
                        "relationship_social",
                        "moral_identity",
                        "existential_drive",
                    )
                ),
                ensure_ascii=False,
            )
        elif step == "G1a":
            content = json.dumps(
                {
                    "intention": "Reply to the user's greeting in character.",
                    "desired_outcome": "The user receives an in-character reply.",
                    "concrete_detail": "Greet the user and open a topic.",
                    "reason": "The user opened with a simple greeting.",
                    "private_monologue": "A quiet hello is an invitation.",
                    "target_role_handles": [],
                    "evidence_handles": [self.episode_handle],
                    "expected_consequences": [
                        "The conversation continues from the greeting."
                    ],
                    "confidence": "medium",
                    "relational_willingness": {
                        "applicability": "not_relationship_sensitive",
                        "stance": "not_applicable",
                        "current_user_relationship_state": "not_applicable",
                        "reason": "关系状态稳定",
                        "evidence_handles": [self.episode_handle],
                    },
                },
                ensure_ascii=False,
            )
        elif step == "P1":
            content = json.dumps(
                {
                    "action_requests": [],
                    "resolver_requests": [],
                    "goal_resolution": "blocked",
                    "resolver_pending_resolution": None,
                    "resolver_goal_progress": None,
                }
            )
        else:
            raise AssertionError(f"unexpected serial step {step!r}")
        return LLMResponse(
            content=content,
            backend=BackendDescriptor(
                route_name=config.route_name,
                backend_kind="openai",
                model_family="test",
                model=config.model,
                normalized_base_url=config.base_url,
                thinking_strategy="none",
                confidence=1.0,
                generation=0,
            ),
            raw_response=None,
            usage={},
        )


def _serial_services(llm: SerialScriptedLLM) -> CognitionChainServicesV3:
    config = LLMCallConfig(
        stage_name="cognition_core_v3.chain",
        route_name="COGNITION_V3_CHAIN_LLM",
        base_url="http://127.0.0.1:9",
        api_key="test-key",
        model="test-model",
        temperature=0.1,
        top_p=0.7,
        top_k=None,
        max_completion_tokens=8192,
        presence_penalty=None,
        context_window_tokens=50000,
    )
    return CognitionChainServicesV3(
        llm=llm,
        chain_lane=config,
        sidecar_lane=None,
        subconscious_enabled=False,
    )


@pytest.mark.asyncio
async def test_serial_cognition_path_produces_valid_v2_output() -> None:
    payload = build_cognition_input_from_global_state(_global_state())
    episode_handle = next(
        row["evidence_handle"]
        for row in payload["evidence"]
        if row["evidence_ref"]["source_kind"] == "episode"
    )
    llm = SerialScriptedLLM(episode_handle)
    ledger_token = bind_v2_attempt_ledger(
        create_v2_attempt_ledger("serial-integration"),
        graph_attempt=1,
    )
    try:
        output = await v3_facade._run_serial_cognition(
            payload,
            _serial_services(llm),
        )
    finally:
        reset_v2_attempt_ledger(ledger_token)

    validated = validate_cognition_core_output(output)
    assert validated["schema_version"] == "cognition_core_output.v2"
    assert validated["diagnostics"]["stage_status"] == {
        "input_validation": "completed",
        "deterministic_preliminary": "completed",
        "semantic_appraisal": "completed",
        "final_reduction": "completed",
        "branch_cognition": "completed",
        "workspace_collapse": "completed",
        "action_planning": "completed",
    }
    assert llm.calls == ["A1", "A2", "G1a", "P1"]
