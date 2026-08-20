"""End-to-end bounded-sidecar coverage for the Cognition V3 facade."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreInputV2,
    validate_cognition_core_input,
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.cognition_core_v3 import facade as v3_facade
from kazusa_ai_chatbot.cognition_core_v3.session import ChainSessionRegistry
from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    project_resolver_observation_for_cognition,
)
from kazusa_ai_chatbot.llm_interface.contracts import (
    BackendDescriptor,
    LLMResponse,
)
from tests.integration.cognition_core_v3.conftest import (
    make_v3_services,
    ordinary_goal_draft,
)


def _question_payload(messages: Sequence[object]) -> dict[str, object]:
    """Read the typed V3 question payload from the latest human message."""

    packet = json.loads(str(getattr(messages[-1], "content", "")))
    if not isinstance(packet, list):
        raise TypeError("V3 question packet must be a list")
    for section in packet:
        if not isinstance(section, Mapping):
            continue
        question = section.get("question")
        if not isinstance(question, Mapping):
            continue
        payload = question.get("payload")
        if isinstance(payload, Mapping):
            return dict(payload)
    raise AssertionError("V3 question packet has no typed payload")


def _empty_appraisal_group(messages: Sequence[object]) -> str:
    """Build one exact empty appraisal response for the supplied families."""

    payload = _question_payload(messages)
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise TypeError("appraisal packet has no question list")
    return json.dumps(
        {
            row["family"]: [
                {
                    "question_id": row["family"],
                    "proposition": None,
                    "delta": None,
                }
            ]
            for row in questions
            if isinstance(row, Mapping) and isinstance(row.get("family"), str)
        },
        ensure_ascii=False,
    )


def _repair_a1_group() -> str:
    """Return the fixed cold A1 appraisal grouping for injected repair."""

    families = (
        "event_agency",
        "goal_threat_outcome",
        "epistemic_comparison_memory",
    )
    return json.dumps(
        {
            family: [
                {
                    "question_id": family,
                    "proposition": None,
                    "delta": None,
                }
            ]
            for family in families
        },
        ensure_ascii=False,
    )


def _tail_ordinary_draft(messages: Sequence[object]) -> str:
    """Build a recurrence ordinary bid without regenerating willingness."""

    payload = _question_payload(messages)
    evidence_handles = payload.get("evidence_handles")
    if not isinstance(evidence_handles, list) or not evidence_handles:
        raise AssertionError("recurrence ordinary packet has no evidence")
    draft = json.loads(ordinary_goal_draft(str(evidence_handles[-1])))
    draft.pop("relational_willingness")
    return json.dumps(draft, ensure_ascii=False)


class _SidecarFacadeLLM:
    """Script primary and sidecar calls while preserving each routed stage."""

    def __init__(
        self,
        *,
        malformed_authorization: bool = False,
        provider_failure: bool = False,
    ) -> None:
        self.calls: list[str] = []
        self.l1_calls = 0
        self.l1_packets: list[dict[str, object]] = []
        self.l1_cancelled = False
        self.repair_calls = 0
        self.repair_inputs: list[str] = []
        self.malformed_authorization = malformed_authorization
        self.provider_failure = provider_failure

    async def ainvoke(self, messages, *, config) -> LLMResponse:
        """Return typed primary, L1, X1, and X2 fixture content."""

        stage_name = config.stage_name.rsplit(".repair", 1)[0]
        self.calls.append(stage_name)
        if stage_name == "L1":
            self.l1_calls += 1
            packet = json.loads(str(getattr(messages[-1], "content", "")))
            if not isinstance(packet, Mapping):
                raise TypeError("L1 packet must be a JSON object")
            self.l1_packets.append(dict(packet))
            if self.l1_calls == 1:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.l1_cancelled = True
                    raise
            content = json.dumps(
                {
                    "schema_version": "l1_residue.v1",
                    "emotional_appraisal": "当前请求带有明确压力。",
                    "interaction_subtext": "用户正在等待回应。",
                    "salience_hints": ["e1"],
                    "risk_flags": ["boundary_pressure"],
                },
                ensure_ascii=False,
            )
            return self._response(content, config)
        if stage_name == "A1":
            await asyncio.sleep(0)
            return self._response("{malformed", config)
        if stage_name == "R.A1":
            await asyncio.sleep(0)
        if stage_name in {"A2", "R.A1", "R.A2"}:
            return self._response(_empty_appraisal_group(messages), config)
        if stage_name == "G1a":
            payload = _question_payload(messages)
            handles = payload.get("evidence_handles")
            if not isinstance(handles, list) or not handles:
                raise AssertionError("cold ordinary packet has no evidence")
            return self._response(ordinary_goal_draft(str(handles[0])), config)
        if stage_name == "R.G1a":
            return self._response(_tail_ordinary_draft(messages), config)
        if stage_name == "R.G1b":
            payload = _question_payload(messages)
            roster = payload.get("branch_roster")
            if not isinstance(roster, list):
                raise AssertionError("active recurrence packet has no roster")
            bids = []
            for row in roster:
                if not isinstance(row, Mapping):
                    raise TypeError("active recurrence roster row is invalid")
                draft = json.loads(_tail_ordinary_draft(messages))
                draft["branch_id"] = row["branch_id"]
                bids.append(draft)
            return self._response(json.dumps({"bids": bids}), config)
        if stage_name == "R.W1":
            return self._response(
                json.dumps(
                    {
                        "primary_bid_handle": "b1",
                        "supporting_bid_handles": [],
                        "suppressed_bid_handles": [],
                    }
                ),
                config,
            )
        if stage_name == "P1":
            return self._response(
                json.dumps(
                    {
                        "action_requests": [
                            {
                                "bid_handle": "b1",
                                "action_handle": "a1",
                                "decision": "check",
                                "semantic_goal": "核对既有任务状态。",
                                "reason": "当前目标需要确认已有状态。",
                            },
                            {
                                "bid_handle": "b1",
                                "action_handle": "a1",
                                "decision": "check",
                                "semantic_goal": "核对既有任务状态。",
                                "reason": "重复候选不能越过授权。",
                            },
                        ],
                        "resolver_requests": [],
                        "goal_resolution": "blocked",
                        "resolver_pending_resolution": None,
                        "resolver_goal_progress": None,
                    },
                    ensure_ascii=False,
                ),
                config,
            )
        if stage_name == "R.P1":
            return self._response(
                json.dumps(
                    {
                        "action_requests": [],
                        "resolver_requests": [
                            {
                                "bid_handle": "b1",
                                "resolver_handle": "r1",
                                "semantic_goal": "准备一个最小批准问题。",
                                "reason": "当前目标仍缺少用户控制的确认。",
                            }
                        ],
                        "goal_resolution": "requires_required_evidence",
                        "resolver_pending_resolution": None,
                        "resolver_goal_progress": None,
                    },
                    ensure_ascii=False,
                ),
                config,
            )
        if stage_name in {"action_authorization", "resolver_authorization"}:
            if self.provider_failure:
                raise RuntimeError("sidecar provider is unavailable")
            if self.malformed_authorization:
                return self._response(
                    json.dumps({"decisions": {"c1": "invalid"}}),
                    config,
                )
            payload = json.loads(str(getattr(messages[-1], "content", "")))
            candidates = payload.get("candidates", {})
            if not isinstance(candidates, Mapping):
                candidates = {}
            decisions = {
                handle: index == 0
                for index, handle in enumerate(candidates)
            }
            return self._response(json.dumps({"decisions": decisions}), config)
        raise AssertionError(f"unexpected V3 stage {stage_name!r}")

    def invoke(self, messages, *, config) -> LLMResponse:
        """Provide the injected synchronous JSON-repair lane only."""

        if config.stage_name != "json_repair":
            raise AssertionError("V3 JSON repair must stay on the sidecar lane")
        self.calls.append("json_repair")
        self.repair_calls += 1
        self.repair_inputs.append(str(getattr(messages[-1], "content", "")))
        return self._response(_repair_a1_group(), config)

    @staticmethod
    def _response(content: str, config) -> LLMResponse:
        """Construct one routed fake response without external state."""

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


def _continuation_input(
    payload: Mapping[str, Any],
    prior_output: Mapping[str, Any],
) -> CognitionCoreInputV2:
    """Append one canonical resolver observation with the V2 handle shape."""

    continuation = deepcopy(dict(payload))
    continuation["mutable_state"] = deepcopy(
        prior_output["state_update"]["replacement_state"]
    )
    observation, _facts = project_resolver_observation_for_cognition(
        {
            "observation_id": "sidecar-resolver-observation",
            "semantic_summary": "Resolver returned a bounded observation.",
        },
        occurred_at="2026-08-20T00:00:00Z",
    )
    observation["evidence_handle"] = f"e{len(continuation['evidence']) + 1}"
    continuation["evidence"].append(observation)
    continuation["resolver_cycle_index"] = 1
    relational = prior_output.get("relational_willingness")
    if not isinstance(relational, Mapping):
        raise TypeError("cold output must carry relational willingness")
    continuation["current_turn_relational_willingness"] = {
        "schema_version": "current_turn_relational_willingness.v2",
        "episode_id": continuation["episode"]["episode_id"],
        "branch_id": "ordinary_response",
        "decision": deepcopy(dict(relational)),
    }
    return validate_cognition_core_input(continuation)


@pytest.mark.asyncio
async def test_l1_repair_x1_x2_preemption_order_cancellation_and_failure_are_bounded(
    cognition_payload: CognitionCoreInputV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Facade sidecars stay advisory, serialized, and deny closed on failure."""

    monkeypatch.setattr(
        v3_facade,
        "_CHAIN_SESSION_REGISTRY",
        ChainSessionRegistry(),
    )
    invoker = _SidecarFacadeLLM()
    services = make_v3_services(
        invoker,
        include_sidecar=True,
        subconscious_enabled=True,
    )
    cold_output = await v3_facade.run_cognition(cognition_payload, services)
    continuation = _continuation_input(cognition_payload, cold_output)
    tail_output = await v3_facade.run_cognition(continuation, services)

    validate_cognition_core_output(cold_output)
    validate_cognition_core_output(tail_output)
    assert invoker.l1_calls == 2
    assert set(invoker.l1_packets[0]) == {
        "current_percept_text",
        "qualitative_affect_bands",
        "boundary_summary",
        "supplied_evidence_handles",
    }
    assert "evidence" not in invoker.l1_packets[0]
    assert invoker.l1_cancelled is True
    assert invoker.repair_calls == 1
    assert "sidecar_l1_preempted_by_repair" in cold_output["diagnostics"][
        "warnings"
    ]
    assert cold_output["action_requests"] == [
        {
            "action_kind": "accepted_task_status_check",
            "decision": "check",
            "context_ref": "",
            "semantic_goal": "核对既有任务状态。",
            "reason": "当前目标需要确认已有状态。",
            "target_roles": [],
            "evidence_handles": ["e1"],
        }
    ]
    assert [row["capability"] for row in tail_output["resolver_requests"]] == [
        "approval_preparation"
    ]
    repair_index = invoker.calls.index("json_repair")
    assert invoker.calls.index("A1") < repair_index
    action_index = invoker.calls.index("action_authorization")
    resolver_index = invoker.calls.index("resolver_authorization")
    assert repair_index < action_index < resolver_index
    assert invoker.calls.count("action_authorization") == 1
    assert invoker.calls.count("resolver_authorization") == 1

    malformed_invoker = _SidecarFacadeLLM(malformed_authorization=True)
    malformed_services = make_v3_services(
        malformed_invoker,
        include_sidecar=True,
    )
    malformed_output = await v3_facade.run_cognition(
        cognition_payload,
        malformed_services,
    )
    assert malformed_output["action_requests"] == []
    assert malformed_invoker.calls.count("action_authorization") == 3
    assert "sidecar_action_authorization_malformed" in malformed_output[
        "diagnostics"
    ]["warnings"]

    failing_invoker = _SidecarFacadeLLM(provider_failure=True)
    failing_output = await v3_facade.run_cognition(
        cognition_payload,
        make_v3_services(failing_invoker, include_sidecar=True),
    )
    assert failing_output["intention"]["route"] == "speech"
    assert failing_output["action_requests"] == []
    assert failing_invoker.calls.count("action_authorization") == 3
    assert "sidecar_action_authorization_unavailable" in failing_output[
        "diagnostics"
    ]["warnings"]

    absent_invoker = _SidecarFacadeLLM()
    absent_output = await v3_facade.run_cognition(
        cognition_payload,
        make_v3_services(absent_invoker),
    )
    assert absent_output["intention"]["route"] == "speech"
    assert absent_output["action_requests"] == []
    assert "action_authorization" not in absent_invoker.calls
    assert absent_invoker.repair_calls == 0
    assert "sidecar_authorization_denied" in absent_output["diagnostics"][
        "warnings"
    ]
