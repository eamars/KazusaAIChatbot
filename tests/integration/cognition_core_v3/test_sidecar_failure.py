"""End-to-end bounded-sidecar coverage for the Cognition V3 facade."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import pairwise
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v3 import facade as v3_facade
from kazusa_ai_chatbot.cognition_core_v3.registry import (
    APPRAISAL_STAGE_FAMILIES,
)
from kazusa_ai_chatbot.cognition_core_v3.session import ChainSessionRegistry
from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    project_resolver_observation_for_cognition,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionCoreInputV2,
    validate_cognition_core_input,
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.llm_interface.contracts import (
    BackendDescriptor,
    LLMResponse,
)
from tests.integration.cognition_core_v3.conftest import (
    make_v3_services,
    ordinary_goal_draft,
)


def _is_explicit_reanchor_packet(content: str) -> bool:
    """Recognize the exact re-anchor carrier emitted by the executor."""

    try:
        packet = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(packet, Mapping) or set(packet) != {"reanchor"}:
        return False
    anchor = packet["reanchor"]
    if not isinstance(anchor, Mapping):
        return False
    if set(anchor) != {"accepted_products", "current_question"}:
        return False
    accepted_products = anchor["accepted_products"]
    current_question = anchor["current_question"]
    if not isinstance(accepted_products, list):
        return False
    if not all(isinstance(row, Mapping) for row in accepted_products):
        return False
    if not isinstance(current_question, Mapping):
        return False
    if set(current_question) != {"contract_name", "facts", "interludes"}:
        return False
    if not isinstance(current_question["contract_name"], str):
        return False
    if not current_question["contract_name"]:
        return False
    if not isinstance(current_question["facts"], Mapping):
        return False
    interludes = current_question["interludes"]
    return isinstance(interludes, list) and all(
        isinstance(row, Mapping) for row in interludes
    )


def _message_snapshot(message: object) -> dict[str, Any]:
    """Project one request message into exact structural evidence."""

    content = str(getattr(message, "content", ""))
    return {
        "message_type": type(message).__name__,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content_chars": len(content),
        "explicit_reanchor": _is_explicit_reanchor_packet(content),
    }


def _has_nonempty_l1_residue(messages: Sequence[object]) -> bool:
    """Detect the sealed question payload's nested L1 residue."""

    if not messages:
        return False
    try:
        packet = json.loads(str(getattr(messages[-1], "content", "")))
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(packet, list):
        return False
    question_rows = [
        section["question"]
        for section in packet
        if (
            isinstance(section, Mapping)
            and set(section) == {"question"}
            and isinstance(section["question"], Mapping)
        )
    ]
    if len(question_rows) != 1:
        return False
    payload = question_rows[0].get("payload")
    residue = payload.get("l1_residue") if isinstance(payload, Mapping) else None
    return isinstance(residue, Mapping) and bool(residue)


_REPAIR_STAGE_PATTERN = re.compile(
    r"^(?P<base>.+)\.repair(?P<attempt>[1-9][0-9]*)$"
)


def _call_stage_name(call: Mapping[str, Any]) -> str | None:
    config = call.get("config")
    stage_name = config.get("stage_name") if isinstance(config, Mapping) else None
    return stage_name if isinstance(stage_name, str) else None


def _is_sequential_repair_transition(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    """Require explicit same-owner, successive repair-stage telemetry."""

    previous_stage = _call_stage_name(previous)
    current_stage = _call_stage_name(current)
    if previous_stage is None or current_stage is None:
        return False
    current_match = _REPAIR_STAGE_PATTERN.fullmatch(current_stage)
    if current_match is None:
        return False
    current_base = current_match.group("base")
    current_attempt = int(current_match.group("attempt"))
    previous_match = _REPAIR_STAGE_PATTERN.fullmatch(previous_stage)
    if previous_match is None:
        return current_base == previous_stage and current_attempt == 1
    return (
        current_base == previous_match.group("base")
        and current_attempt == int(previous_match.group("attempt")) + 1
    )


def _is_appraisal_recovery_stage_transition(
    previous_stage: str | None,
    current_stage: str | None,
) -> bool:
    """Recognize only registry-ordered grouped-appraisal recovery stages."""

    if previous_stage is None or current_stage is None:
        return False
    for prefix in ("", "R."):
        for stage_index, (stage_id, families) in enumerate(
            APPRAISAL_STAGE_FAMILIES
        ):
            stage_name = f"{prefix}{stage_id}"
            registered_stages = [
                stage_name,
                *(f"{stage_name}.{family}" for family in families),
            ]
            if stage_index + 1 < len(APPRAISAL_STAGE_FAMILIES):
                next_stage_id = APPRAISAL_STAGE_FAMILIES[stage_index + 1][0]
                registered_stages.append(f"{prefix}{next_stage_id}")
            else:
                registered_stages.append(f"{prefix}G1a")
            if (previous_stage, current_stage) in pairwise(registered_stages):
                return True
    return False


def _prefix_evidence(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove exact prefixes or the two sealed continuation transitions."""

    primary_calls = [call for call in calls if call["lane"] == "primary"]
    continuations: list[dict[str, Any]] = []
    for previous, current in pairwise(primary_calls):
        previous_messages = previous["messages"]
        current_messages = current["messages"]
        exact_prefix = current_messages[: len(previous_messages)] == previous_messages
        common_prefix_count = 0
        for previous_row, current_row in zip(previous_messages, current_messages):
            if previous_row != current_row:
                break
            common_prefix_count += 1
        repair_stage_telemetry = _is_sequential_repair_transition(previous, current)
        human_tail_replacement = (
            len(previous_messages) == len(current_messages)
            and len(previous_messages) >= 2
            and previous_messages[:-1] == current_messages[:-1]
            and previous_messages[-1].get("message_type") == "HumanMessage"
            and current_messages[-1].get("message_type") == "HumanMessage"
            and previous_messages[-1] != current_messages[-1]
        )
        appraisal_recovery_stage_telemetry = (
            human_tail_replacement
            and _is_appraisal_recovery_stage_transition(
                _call_stage_name(previous), _call_stage_name(current)
            )
        )
        explicit_reanchor = (
            len(current_messages) == 2
            and len(previous_messages) >= 1
            and current_messages[0] == previous_messages[0]
            and current_messages[-1].get("message_type") == "HumanMessage"
            and current_messages[-1].get("explicit_reanchor") is True
        )
        if exact_prefix:
            classification, transition = "exact", None
            prefix_message_count = len(previous_messages)
        elif explicit_reanchor:
            classification, transition = "permitted_transition", "explicit_reanchor"
            prefix_message_count = 1
        elif appraisal_recovery_stage_telemetry:
            classification, transition = (
                "permitted_transition", "appraisal_recovery_tail_replacement"
            )
            prefix_message_count = len(previous_messages) - 1
        elif human_tail_replacement and repair_stage_telemetry:
            classification, transition = "permitted_transition", "repair_tail_replacement"
            prefix_message_count = len(previous_messages) - 1
        else:
            classification, transition = "invalid", None
            prefix_message_count = common_prefix_count
        prefix_chars = sum(
            row["content_chars"] for row in current_messages[:prefix_message_count]
        )
        continuations.append({
            "previous_sequence": previous["sequence"],
            "current_sequence": current["sequence"],
            "classification": classification,
            "transition": transition,
            "valid": classification != "invalid",
            "exact_prefix": exact_prefix,
            "permitted_transition": classification == "permitted_transition",
            "invalid": classification == "invalid",
            "previous_stage_name": _call_stage_name(previous),
            "current_stage_name": _call_stage_name(current),
            "repair_stage_telemetry": repair_stage_telemetry,
            "appraisal_recovery_stage_telemetry": appraisal_recovery_stage_telemetry,
            "prefix_message_count": prefix_message_count,
            "prefix_chars": prefix_chars,
            "suffix_chars": current["serialized_request_chars"] - prefix_chars,
        })
    return {
        "continuation_count": len(continuations),
        "all_exact": bool(continuations) and all(row["exact_prefix"] for row in continuations),
        "all_continuations_valid": bool(continuations) and all(row["valid"] for row in continuations),
        "invalid_count": sum(row["invalid"] for row in continuations),
        "continuations": continuations,
    }


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
    families = payload.get("families")
    if not isinstance(families, list):
        raise TypeError("appraisal packet has no family list")
    return json.dumps(
        {
            row["family"]: {"propositions": [], "deltas": []}
            for row in families
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
            family: {"propositions": [], "deltas": []}
            for family in families
        },
        ensure_ascii=False,
    )


def _tail_ordinary_draft(messages: Sequence[object]) -> str:
    """Build a recurrence ordinary bid without regenerating willingness."""

    payload = _question_payload(messages)
    contract = payload.get("goal_output_contract")
    evidence_handles = (
        contract.get("allowed_evidence_handles")
        if isinstance(contract, Mapping)
        else None
    )
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
        self.l1_active = False
        self.primary_started_while_l1_active = False
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
                self.l1_active = True
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.l1_active = False
                    self.l1_cancelled = True
                    raise
            self.l1_active = False
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
            self.primary_started_while_l1_active = (
                self.primary_started_while_l1_active or self.l1_active
            )
            await asyncio.sleep(0)
            return self._response("{malformed", config)
        if stage_name.startswith(("A1.", "A2.", "R.A1.", "R.A2.")):
            return self._response(_empty_appraisal_group(messages), config)
        if stage_name == "R.A1":
            await asyncio.sleep(0)
        if stage_name in {"A2", "R.A1", "R.A2"}:
            return self._response(_empty_appraisal_group(messages), config)
        if stage_name == "G1a":
            payload = _question_payload(messages)
            contract = payload.get("goal_output_contract")
            handles = (
                contract.get("allowed_evidence_handles")
                if isinstance(contract, Mapping)
                else None
            )
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


def _captured_primary_call(
    sequence: int,
    messages: Sequence[object],
    *,
    stage_name: str,
) -> dict[str, object]:
    """Build one privacy-safe primary call record for verifier tests."""

    snapshots = [_message_snapshot(message) for message in messages]
    return {
        "lane": "primary",
        "sequence": sequence,
        "config": {"stage_name": stage_name},
        "messages": snapshots,
        "serialized_request_chars": sum(
            row["content_chars"] for row in snapshots
        ),
    }


def test_l1_residue_detection_requires_the_sealed_question_payload() -> None:
    """L1 evidence comes from the question payload section only."""

    packet = json.dumps(
        [
            {"constraints_and_operational_state": {}},
            {
                "question": {
                    "contract_name": "semantic_appraisal_group.v1",
                    "instruction": "sealed",
                    "payload": {"l1_residue": {"risk_flags": ["pressure"]}},
                }
            },
        ],
        ensure_ascii=False,
    )
    assert _has_nonempty_l1_residue([HumanMessage(content=packet)]) is True
    assert _has_nonempty_l1_residue(
        [HumanMessage(content=json.dumps({"l1_residue": {"risk": True}}))]
    ) is False
    assert _has_nonempty_l1_residue(
        [
            HumanMessage(
                content=json.dumps(
                    {"arbitrary": {"payload": {"l1_residue": {"risk": True}}}}
                )
            )
        ]
    ) is False
    assert _has_nonempty_l1_residue(
        [
            HumanMessage(
                content=json.dumps(
                    [
                        {
                            "question": {
                                "payload": {"l1_residue": {}}
                            }
                        }
                    ]
                )
            )
        ]
    ) is False


def test_primary_prefix_evidence_classifies_only_sealed_transitions() -> None:
    """Prefix evidence requires stage telemetry for repair transitions."""

    system = SystemMessage(content="system")
    accepted_human = HumanMessage(content="accepted question")
    accepted_answer = AIMessage(content="accepted answer")
    exact = _prefix_evidence(
        [
            _captured_primary_call(
                1,
                [system, accepted_human, accepted_answer],
                stage_name="A1",
            ),
            _captured_primary_call(
                2,
                [
                    system,
                    accepted_human,
                    accepted_answer,
                    HumanMessage(content="next"),
                ],
                stage_name="G1a",
            ),
        ]
    )
    assert exact["all_exact"] is True
    assert exact["all_continuations_valid"] is True
    assert exact["continuations"][0]["classification"] == "exact"

    repair = _prefix_evidence(
        [
            _captured_primary_call(
                1,
                [
                    system,
                    accepted_human,
                    accepted_answer,
                    HumanMessage(content="original repair tail"),
                ],
                stage_name="A1",
            ),
            _captured_primary_call(
                2,
                [
                    system,
                    accepted_human,
                    accepted_answer,
                    HumanMessage(content="replacement repair tail"),
                ],
                stage_name="A1.repair1",
            ),
            _captured_primary_call(
                3,
                [
                    system,
                    accepted_human,
                    accepted_answer,
                    HumanMessage(content="second replacement repair tail"),
                ],
                stage_name="A1.repair2",
            ),
        ]
    )
    repair_rows = repair["continuations"]
    assert repair["all_exact"] is False
    assert repair["all_continuations_valid"] is True
    assert [row["classification"] for row in repair_rows] == [
        "permitted_transition",
        "permitted_transition",
    ]
    assert [row["transition"] for row in repair_rows] == [
        "repair_tail_replacement",
        "repair_tail_replacement",
    ]
    assert [row["repair_stage_telemetry"] for row in repair_rows] == [
        True,
        True,
    ]

    reanchor = _prefix_evidence(
        [
            _captured_primary_call(
                1,
                [system, accepted_human, accepted_answer],
                stage_name="A1",
            ),
            _captured_primary_call(
                2,
                [
                    system,
                    HumanMessage(
                        content=json.dumps(
                            {
                                "reanchor": {
                                    "accepted_products": [],
                                    "current_question": {
                                        "contract_name": "A1.v1",
                                        "facts": {},
                                        "interludes": [],
                                    },
                                }
                            },
                            separators=(",", ":"),
                        )
                    ),
                ],
                stage_name="A1",
            ),
        ]
    )
    reanchor_row = reanchor["continuations"][0]
    assert reanchor["all_exact"] is False
    assert reanchor["all_continuations_valid"] is True
    assert reanchor_row["classification"] == "permitted_transition"
    assert reanchor_row["transition"] == "explicit_reanchor"

    normal_replacement = _prefix_evidence(
        [
            _captured_primary_call(
                1,
                [
                    system,
                    accepted_human,
                    accepted_answer,
                    HumanMessage(content="original repair tail"),
                ],
                stage_name="A1",
            ),
            _captured_primary_call(
                2,
                [
                    system,
                    accepted_human,
                    accepted_answer,
                    HumanMessage(content="normal-stage replacement"),
                ],
                stage_name="A1",
            ),
        ]
    )
    normal_row = normal_replacement["continuations"][0]
    assert normal_replacement["all_continuations_valid"] is False
    assert normal_row["classification"] == "invalid"
    assert normal_row["repair_stage_telemetry"] is False

    different_stage = _prefix_evidence(
        [
            _captured_primary_call(
                1,
                [
                    system,
                    accepted_human,
                    accepted_answer,
                    HumanMessage(content="original repair tail"),
                ],
                stage_name="A1",
            ),
            _captured_primary_call(
                2,
                [
                    system,
                    accepted_human,
                    accepted_answer,
                    HumanMessage(content="different-stage replacement"),
                ],
                stage_name="G1a.repair1",
            ),
        ]
    )
    different_row = different_stage["continuations"][0]
    assert different_stage["all_continuations_valid"] is False
    assert different_row["classification"] == "invalid"
    assert different_row["repair_stage_telemetry"] is False

    skipped_repair = _prefix_evidence(
        [
            _captured_primary_call(
                1,
                [
                    system,
                    accepted_human,
                    accepted_answer,
                    HumanMessage(content="first repair tail"),
                ],
                stage_name="A1.repair1",
            ),
            _captured_primary_call(
                2,
                [
                    system,
                    accepted_human,
                    accepted_answer,
                    HumanMessage(content="skipped repair tail"),
                ],
                stage_name="A1.repair3",
            ),
        ]
    )
    skipped_row = skipped_repair["continuations"][0]
    assert skipped_repair["all_continuations_valid"] is False
    assert skipped_row["classification"] == "invalid"
    assert skipped_row["repair_stage_telemetry"] is False

    invalid_reanchor = _prefix_evidence(
        [
            _captured_primary_call(
                1,
                [system, accepted_human, accepted_answer],
                stage_name="A1",
            ),
            _captured_primary_call(
                2,
                [
                    system,
                    HumanMessage(content=json.dumps({"reanchor": {}})),
                ],
                stage_name="A1",
            ),
        ]
    )
    invalid_reanchor_row = invalid_reanchor["continuations"][0]
    assert invalid_reanchor["all_continuations_valid"] is False
    assert invalid_reanchor_row["classification"] == "invalid"

    invalid_reanchor_fields = _prefix_evidence(
        [
            _captured_primary_call(
                1,
                [system, accepted_human, accepted_answer],
                stage_name="A1",
            ),
            _captured_primary_call(
                2,
                [
                    system,
                    HumanMessage(
                        content=json.dumps(
                            {
                                "reanchor": {
                                    "accepted_products": [],
                                    "current_question": {
                                        "contract_name": "A1.v1",
                                        "facts": [],
                                        "interludes": [],
                                    },
                                }
                            }
                        )
                    ),
                ],
                stage_name="A1",
            ),
        ]
    )
    invalid_fields_row = invalid_reanchor_fields["continuations"][0]
    assert invalid_reanchor_fields["all_continuations_valid"] is False
    assert invalid_fields_row["classification"] == "invalid"


def test_primary_prefix_evidence_accepts_registry_ordered_appraisal_recovery() -> None:
    """Appraisal rollback accepts only the canonical family progression."""

    system = SystemMessage(content="system")
    accepted_human = HumanMessage(content="accepted question")
    accepted_answer = AIMessage(content="accepted answer")

    for prefix in ("", "R."):
        stage_names: list[str] = []
        for stage_index, (stage_id, families) in enumerate(
            APPRAISAL_STAGE_FAMILIES
        ):
            stage_name = f"{prefix}{stage_id}"
            stage_names.extend([
                stage_name,
                *(f"{stage_name}.{family}" for family in families),
            ])
            if stage_index + 1 == len(APPRAISAL_STAGE_FAMILIES):
                stage_names.append(f"{prefix}G1a")

        evidence = _prefix_evidence([
            _captured_primary_call(
                sequence,
                [
                    system,
                    accepted_human,
                    accepted_answer,
                    HumanMessage(content=f"recovery-{sequence}"),
                ],
                stage_name=stage_name,
            )
            for sequence, stage_name in enumerate(stage_names, start=1)
        ])
        rows = evidence["continuations"]
        assert evidence["all_exact"] is False
        assert evidence["all_continuations_valid"] is True
        assert [row["classification"] for row in rows] == [
            "permitted_transition"
        ] * (len(stage_names) - 1)
        assert [row["transition"] for row in rows] == [
            "appraisal_recovery_tail_replacement"
        ] * (len(stage_names) - 1)
        assert all(
            row["appraisal_recovery_stage_telemetry"] is True
            for row in rows
        )

    invalid_transitions = (
        ("A1.event_agency", "A1.event_agency"),
        ("A1.goal_threat_outcome", "A1.event_agency"),
        ("A1.unknown", "A1.unknown_next"),
        ("A1", "R.A1.event_agency"),
        ("R.A1", "A1.event_agency"),
        ("G1a", "G1a.recovery"),
    )
    for previous_stage, current_stage in invalid_transitions:
        evidence = _prefix_evidence([
            _captured_primary_call(
                1,
                [
                    system,
                    accepted_human,
                    accepted_answer,
                    HumanMessage(content="old tail"),
                ],
                stage_name=previous_stage,
            ),
            _captured_primary_call(
                2,
                [
                    system,
                    accepted_human,
                    accepted_answer,
                    HumanMessage(content="new tail"),
                ],
                stage_name=current_stage,
            ),
        ])
        row = evidence["continuations"][0]
        assert evidence["all_continuations_valid"] is False
        assert row["classification"] == "invalid"
        assert row["appraisal_recovery_stage_telemetry"] is False

    deeper_prefix_mutation = _prefix_evidence([
        _captured_primary_call(
            1,
            [
                system,
                accepted_human,
                accepted_answer,
                HumanMessage(content="old tail"),
            ],
            stage_name="A1",
        ),
        _captured_primary_call(
            2,
            [
                system,
                HumanMessage(content="mutated accepted prefix"),
                accepted_answer,
                HumanMessage(content="new tail"),
            ],
            stage_name="A1.event_agency",
        ),
    ])
    mutated_row = deeper_prefix_mutation["continuations"][0]
    assert deeper_prefix_mutation["all_continuations_valid"] is False
    assert mutated_row["classification"] == "invalid"
    assert mutated_row["appraisal_recovery_stage_telemetry"] is False


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
    assert invoker.primary_started_while_l1_active is True
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
