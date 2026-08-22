"""Resolver-recurrence coverage for the Cognition V3 serial chain."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionCoreInputV2,
    CognitionCoreOutputV2,
    CognitionExecutionError,
    validate_cognition_core_input,
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    bind_v2_attempt_ledger,
    create_v2_attempt_ledger,
    reset_v2_attempt_ledger,
    snapshot_v2_guarded_attempt_ledger,
)
from kazusa_ai_chatbot.cognition_core_v3 import facade as v3_facade
from kazusa_ai_chatbot.cognition_core_v3.session import (
    ChainSessionRegistry,
    build_session_key,
    create_cold_session,
)
from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref
from kazusa_ai_chatbot.cognition_resolver import guardrail
from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    project_resolver_observation_for_cognition,
)
from kazusa_ai_chatbot.config import (
    COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS,
    COGNITION_RESOLVER_MAX_CYCLES,
)
from kazusa_ai_chatbot.llm_interface.contracts import (
    BackendDescriptor,
    LLMResponse,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2 as persona_module
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_scene_context_from_global_state,
)
from tests.integration.cognition_core_v3.conftest import (
    make_v3_services,
    ordinary_goal_draft,
)
from tests.test_cognition_chain_connector_mapping import _global_state


def _question_payload(messages: Sequence[object]) -> dict[str, object]:
    """Read the one typed V3 question payload from recorded messages."""

    content = getattr(messages[-1], "content", "")
    packet = json.loads(content)
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
    """Return an exact empty result for the dynamically supplied families."""

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


def _tail_ordinary_draft(messages: Sequence[object]) -> str:
    """Build a recurrence ordinary draft without a regenerated stance."""

    payload = _question_payload(messages)
    contract = payload.get("goal_output_contract")
    evidence_handles = (
        contract.get("allowed_evidence_handles")
        if isinstance(contract, Mapping)
        else None
    )
    if not isinstance(evidence_handles, list) or not evidence_handles:
        raise AssertionError("tail ordinary question has no evidence handles")
    draft = json.loads(ordinary_goal_draft(str(evidence_handles[-1])))
    draft.pop("relational_willingness")
    return json.dumps(draft, ensure_ascii=False)


class _RecurrenceLLM:
    """Record cold and recurrence packets while returning typed fixtures."""

    def __init__(
        self,
        *,
        block_first_tail: bool = False,
        authorize_effects: bool = False,
    ) -> None:
        self.calls: list[str] = []
        self.messages_by_stage: dict[str, list[tuple[str, ...]]] = {}
        self.primary_in_flight = 0
        self.maximum_primary_in_flight = 0
        self.tail_started = asyncio.Event()
        self.release_tail = asyncio.Event()
        if not block_first_tail:
            self.release_tail.set()
        self._block_first_tail = block_first_tail
        self._blocked = False
        self._authorize_effects = authorize_effects

    async def ainvoke(self, messages, *, config) -> LLMResponse:
        stage_name = config.stage_name.rsplit(".repair", 1)[0]
        self.calls.append(stage_name)
        self.messages_by_stage.setdefault(stage_name, []).append(
            tuple(str(getattr(message, "content", "")) for message in messages)
        )
        self.primary_in_flight += 1
        self.maximum_primary_in_flight = max(
            self.maximum_primary_in_flight,
            self.primary_in_flight,
        )
        try:
            if (
                stage_name == "R.A1"
                and self._block_first_tail
                and not self._blocked
            ):
                self._blocked = True
                self.tail_started.set()
                await self.release_tail.wait()
            content = self._content_for(stage_name, messages)
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
        finally:
            self.primary_in_flight -= 1

    def _content_for(self, stage_name: str, messages: Sequence[object]) -> str:
        if stage_name in {"A1", "A2", "A3", "R.A1", "R.A2", "R.A3"}:
            return _empty_appraisal_group(messages)
        if stage_name == "G1a":
            payload = _question_payload(messages)
            contract = payload.get("goal_output_contract")
            handles = (
                contract.get("allowed_evidence_handles")
                if isinstance(contract, Mapping)
                else None
            )
            if not isinstance(handles, list) or not handles:
                raise AssertionError("cold ordinary question has no evidence")
            return ordinary_goal_draft(str(handles[0]))
        if stage_name == "R.G1a":
            return _tail_ordinary_draft(messages)
        if stage_name == "P1" and self._authorize_effects:
            return json.dumps(
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
                            "reason": "重复候选需要独立授权。",
                        },
                    ],
                    "resolver_requests": [],
                    "goal_resolution": "blocked",
                    "resolver_pending_resolution": None,
                    "resolver_goal_progress": None,
                },
                ensure_ascii=False,
            )
        if stage_name == "R.P1" and self._authorize_effects:
            return json.dumps(
                {
                    "action_requests": [],
                    "resolver_requests": [
                        {
                            "bid_handle": "b1",
                            "resolver_handle": "r1",
                            "semantic_goal": "准备一个最小批准问题。",
                            "reason": "当前目标仍缺少用户控制的确认。",
                        },
                        {
                            "bid_handle": "b1",
                            "resolver_handle": "r1",
                            "semantic_goal": "准备一个最小批准问题。",
                            "reason": "重复候选需要独立授权。",
                        },
                    ],
                    "goal_resolution": "requires_required_evidence",
                    "resolver_pending_resolution": None,
                    "resolver_goal_progress": None,
                },
                ensure_ascii=False,
            )
        if stage_name in {"P1", "R.P1"}:
            return json.dumps({
                "action_requests": [],
                "resolver_requests": [],
                "goal_resolution": "blocked",
                "resolver_pending_resolution": None,
                "resolver_goal_progress": None,
            })
        if stage_name in {
            "action_authorization",
            "resolver_authorization",
        }:
            packet = json.loads(str(getattr(messages[-1], "content", "")))
            candidates = packet.get("candidates", {})
            if not isinstance(candidates, Mapping):
                raise TypeError("authorization packet has no candidates")
            return json.dumps(
                {
                    "decisions": {
                        handle: index == 0
                        for index, handle in enumerate(candidates)
                    }
                }
            )
        if stage_name in {"G1b", "R.G1b"}:
            payload = _question_payload(messages)
            roster = payload.get("branch_roster")
            if not isinstance(roster, list):
                raise AssertionError("active recurrence question has no roster")
            bids = []
            for row in roster:
                if not isinstance(row, Mapping):
                    raise TypeError("active recurrence roster row is invalid")
                if stage_name == "G1b":
                    evidence_handles = payload.get(
                        "allowed_evidence_handles"
                    )
                    if not isinstance(evidence_handles, list):
                        raise AssertionError(
                            "active cold question has no evidence handles"
                        )
                    draft = json.loads(
                        ordinary_goal_draft(str(evidence_handles[0]))
                    )
                    draft.pop("relational_willingness")
                else:
                    evidence_handles = payload.get(
                        "allowed_evidence_handles"
                    )
                    if not isinstance(evidence_handles, list):
                        raise AssertionError(
                            "active recurrence question has no evidence handles"
                        )
                    draft = json.loads(
                        ordinary_goal_draft(str(evidence_handles[0]))
                    )
                    draft.pop("relational_willingness")
                draft["branch_id"] = row["branch_id"]
                bids.append(draft)
            return json.dumps({"bids": bids}, ensure_ascii=False)
        if stage_name in {"W1", "R.W1"}:
            return json.dumps(
                {
                    "primary_bid_handle": "b1",
                    "supporting_bid_handles": [],
                    "suppressed_bid_handles": [],
                }
            )
        raise AssertionError(f"unexpected V3 stage {stage_name!r}")

    def invoke(self, messages, *, config):
        """Keep the fixture on the engine's asynchronous surface."""

        del messages, config
        raise AssertionError("V3 recurrence uses asynchronous LLM invocation")


def _cycle_input(
    initial_payload: Mapping[str, Any],
    prior_output: Mapping[str, Any],
    *,
    cycle_index: int,
) -> CognitionCoreInputV2:
    """Build one canonical resolver continuation from the prior output."""

    payload = deepcopy(dict(initial_payload))
    payload["mutable_state"] = deepcopy(
        prior_output["state_update"]["replacement_state"]
    )
    observation, _facts = project_resolver_observation_for_cognition(
        {
            "observation_id": f"resolver-observation-{cycle_index}",
            "semantic_summary": f"Resolver observation {cycle_index}.",
        },
        occurred_at="2026-08-20T00:00:00Z",
    )
    observation["evidence_handle"] = f"e{len(payload['evidence']) + 1}"
    payload["evidence"].append(observation)
    payload["resolver_cycle_index"] = cycle_index
    relational_willingness = prior_output.get("relational_willingness")
    if not isinstance(relational_willingness, Mapping):
        raise TypeError("cold output must carry relational willingness")
    payload["current_turn_relational_willingness"] = {
        "schema_version": "current_turn_relational_willingness.v2",
        "episode_id": payload["episode"]["episode_id"],
        "branch_id": "ordinary_response",
        "decision": deepcopy(dict(relational_willingness)),
    }
    return validate_cognition_core_input(payload)


def _fresh_session_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate process-local continuation state for one integration test."""

    monkeypatch.setattr(
        v3_facade,
        "_CHAIN_SESSION_REGISTRY",
        ChainSessionRegistry(),
    )


def _payload_with_active_recurrence_goal(
    payload: CognitionCoreInputV2,
) -> CognitionCoreInputV2:
    """Add one active sibling so cold and recurrence both exercise G1b."""

    variant = deepcopy(payload)
    evidence_ref = variant["evidence"][0]["evidence_ref"]
    variant["mutable_state"]["goals"] = [{
        "entity_id": "goal:bond-protection",
        "description": "Protect the current boundary.",
        "salience": 70,
        "role_refs": [],
        "evidence_refs": [dict(evidence_ref)],
        "created_at": "2026-07-14T00:00:00Z",
        "updated_at": "2026-07-14T00:00:00Z",
        "status": "pursuing",
        "goal_kind": "bond_protection",
        "importance": 80,
        "progress": 10,
        "obstruction": 0,
        "urgency": 60,
        "expected_success": 50,
        "controllability": 50,
        "recoverability": 50,
    }]
    return validate_cognition_core_input(variant)


@pytest.mark.asyncio
async def test_resolver_observation_reattaches_short_tail_and_commits_once(
    cognition_payload: CognitionCoreInputV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching resolver row uses the tail and advances one V3 session."""

    _fresh_session_registry(monkeypatch)
    llm = _RecurrenceLLM()
    family_batches: list[tuple[str, tuple[str, ...]]] = []
    original_stage_builder = v3_facade.v3_prompt.build_appraisal_stage_question

    def capture_stage(
        *,
        planned_questions,
        stage_name,
        l1_residue=None,
        relation_context=None,
    ):
        family_batches.append(
            (
                stage_name,
                tuple(
                    str(question["question_kind"])
                    for question in planned_questions
                ),
            )
        )
        return original_stage_builder(
            planned_questions=planned_questions,
            stage_name=stage_name,
            l1_residue=l1_residue,
            relation_context=relation_context,
        )

    monkeypatch.setattr(
        v3_facade.v3_prompt,
        "build_appraisal_stage_question",
        capture_stage,
    )
    services = replace(
        make_v3_services(llm),
        turn_deadline_seconds=417,
    )

    cold_output = await v3_facade.run_cognition(cognition_payload, services)
    session_key = build_session_key(
        episode_id=cognition_payload["episode"]["episode_id"],
        state_scope=cognition_payload["state_scope"],
        owner_identity=(
            f"{services.chain_lane.base_url}|{services.chain_lane.model}"
        ),
    )
    session = v3_facade._CHAIN_SESSION_REGISTRY.get(session_key)
    assert session is not None
    cold_messages = session.accepted_messages
    cold_products = session.accepted_products
    cold_attempt_ledger = dict(session.attempt_ledger)
    assert all(
        "typed_product" in product
        for product in cold_products
    )
    assert session.current_roster == ("ordinary_response",)
    session_advancements = []
    session_timestamps: list[tuple[float, float]] = []
    original_put = v3_facade._CHAIN_SESSION_REGISTRY.put

    def record_session_advancement(candidate) -> None:
        session_advancements.append(candidate)
        session_timestamps.append(
            (candidate.last_used_monotonic, candidate.expires_monotonic)
        )
        original_put(candidate)

    monkeypatch.setattr(
        v3_facade._CHAIN_SESSION_REGISTRY,
        "put",
        record_session_advancement,
    )
    continuation = _cycle_input(cognition_payload, cold_output, cycle_index=1)
    output = await v3_facade.run_cognition(continuation, services)

    validate_cognition_core_output(output)
    assert llm.calls[:5] == ["A1", "A2", "G1a", "P1", "R.A1"]
    assert llm.calls[5:] == ["R.A2", "R.G1a", "R.P1"]
    assert output["state_update"]["expected_previous_state"] == (
        continuation["mutable_state"]
    )
    assert output["relational_willingness"] == cold_output["relational_willingness"]
    assert "session_reattached" in output["diagnostics"]["warnings"]
    assert llm.messages_by_stage["R.A1"][0][1:-1] == tuple(
        content for _, content in cold_messages
    )
    advanced_session = v3_facade._CHAIN_SESSION_REGISTRY.get(session_key)
    assert advanced_session is not None
    assert session_advancements == [advanced_session]
    assert advanced_session.last_output == output
    assert advanced_session.accepted_products[: len(cold_products)] == (
        cold_products
    )
    assert advanced_session.attempt_ledger["serial_appraisal"] > (
        cold_attempt_ledger["serial_appraisal"]
    )
    expected_ttl = (
        COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS
        * COGNITION_RESOLVER_MAX_CYCLES
        + services.turn_deadline_seconds
        + 30
    )
    assert len(session_timestamps) == 1
    advanced_at, expires_at = session_timestamps[0]
    assert expires_at == advanced_at + expected_ttl

    tail_packets = [
        json.loads(packet[-1])
        for stage_name in ("R.A1", "R.A2")
        for packet in llm.messages_by_stage[stage_name]
    ]
    appended_handle = continuation["evidence"][-1]["evidence_handle"]
    for packet in tail_packets:
        question = next(section["question"] for section in packet if "question" in section)
        for row in question["payload"]["families"]:
            assert row["evidence_handles"] == [appended_handle]

    assert "R.G1b" not in llm.messages_by_stage
    ordinary_packet = json.loads(llm.messages_by_stage["R.G1a"][0][-1])
    assert not any(
        interlude.get("notice_kind") == "I2"
        for section in ordinary_packet
        for interlude in section.get("interludes", [])
    )
    planning_packet = json.loads(llm.messages_by_stage["R.P1"][0][-1])
    assert any(
        interlude.get("notice_kind") == "I2"
        for section in planning_packet
        for interlude in section.get("interludes", [])
    )
    assert family_batches == [
        (
            "A1",
            ("event_agency", "goal_threat_outcome", "epistemic_comparison_memory"),
        ),
        (
            "A2",
            ("relationship_social", "moral_identity", "existential_drive"),
        ),
        (
            "A1",
            ("event_agency", "goal_threat_outcome", "epistemic_comparison_memory"),
        ),
        (
            "A2",
            ("relationship_social", "moral_identity"),
        ),
    ]


@pytest.mark.asyncio
async def test_cold_and_recurrence_goal_questions_share_dialogue_bindings(
    cognition_payload: CognitionCoreInputV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold and R-tail G1 questions carry one exact dialogue binding set."""

    _fresh_session_registry(monkeypatch)
    payload = _payload_with_active_recurrence_goal(cognition_payload)
    llm = _RecurrenceLLM()
    services = make_v3_services(llm)

    cold_output = await v3_facade.run_cognition(payload, services)
    continuation = _cycle_input(payload, cold_output, cycle_index=1)
    await v3_facade.run_cognition(continuation, services)

    expected_bindings = [{
        "speaker_handle": "current_user",
        "addressee_handle": "self",
        "first_person_handle": "current_user",
        "implicit_imperative_subject_handle": "self",
        "second_person_handle": "self",
    }]
    payloads = {}
    for stage_name in ("G1a", "G1b", "R.G1a", "R.G1b"):
        packet = json.loads(llm.messages_by_stage[stage_name][0][-1])
        payloads[stage_name] = next(
            section["question"]["payload"]
            for section in packet
            if isinstance(section, Mapping) and "question" in section
        )
        assert payloads[stage_name]["dialogue_role_bindings"] == (
            expected_bindings
        )

    assert payloads["G1a"]["dialogue_role_bindings"] == (
        payloads["G1b"]["dialogue_role_bindings"]
    )
    assert payloads["R.G1a"]["dialogue_role_bindings"] == (
        payloads["R.G1b"]["dialogue_role_bindings"]
    )
    assert "hello" not in json.dumps(payloads, ensure_ascii=False)


@pytest.mark.asyncio
async def test_recurrence_goal_projection_uses_post_reduction_matter_state(
    cognition_payload: CognitionCoreInputV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolver-tail G1a receives lifecycle facts from the reduced state."""

    _fresh_session_registry(monkeypatch)
    llm = _RecurrenceLLM()
    services = make_v3_services(llm)
    original_reduce = v3_facade._reduce_serial_appraisals
    reduction_count = 0

    def inject_resolved_threat(*args, **kwargs):
        nonlocal reduction_count
        reduction_count += 1
        result = original_reduce(*args, **kwargs)
        if reduction_count != 2:
            return result
        final_state = deepcopy(result[0])
        final_state["threats"] = [{
            "entity_id": "threat:resolver-tail",
            "description": "The resolver observation closed this threat.",
            "salience": 60,
            "role_refs": [],
            "evidence_refs": [],
            "created_at": cognition_payload["episode"]["created_at"],
            "updated_at": cognition_payload["episode"]["created_at"],
            "status": "resolved",
            "likelihood": 0,
            "expected_harm": 0,
            "uncertainty": 0,
            "controllability": 50,
            "coping_potential": 50,
            "residual_pressure": 0,
        }]
        return (final_state, *result[1:])

    monkeypatch.setattr(
        v3_facade,
        "_reduce_serial_appraisals",
        inject_resolved_threat,
    )
    cold_output = await v3_facade.run_cognition(cognition_payload, services)
    continuation = _cycle_input(cognition_payload, cold_output, cycle_index=1)

    await v3_facade.run_cognition(continuation, services)

    recurrence_packet = json.loads(llm.messages_by_stage["R.G1a"][0][-1])
    recurrence_payload = next(
        section["question"]["payload"]
        for section in recurrence_packet
        if isinstance(section, Mapping) and "question" in section
    )
    authoritative_state = recurrence_payload["authoritative_state"]
    threat_rows = authoritative_state["matter_projections"]["threats"]
    assert len(threat_rows) == 1
    threat_row = threat_rows[0]
    assert set(threat_row) == {
        "handle",
        "description",
        "lifecycle",
        "salience",
        "duration",
        "causal_roles",
        "uncertainty",
        "residual_pressure",
    }
    assert threat_row["handle"] == "t1"
    assert threat_row["description"] == (
        "The resolver observation closed this threat."
    )
    assert threat_row["lifecycle"] == "已解决"
    assert "entity_id" not in json.dumps(authoritative_state)
    assert recurrence_packet[-1]["question"]["payload"][
        "authoritative_state"
    ] == authoritative_state


@pytest.mark.asyncio
async def test_live_resolver_loop_commits_one_terminal_replacement_after_recurrence(
    cognition_payload: CognitionCoreInputV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One parent replay still yields one capability and terminal commit."""

    state = _global_state()
    continuation_ref = build_goal_continuation_ref(
        source_episode_id="episode-1",
        source_message_id="message-1",
        branch_id="ordinary_response",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "resolver-goal-1",
        },
    )
    resolver_request = {
        "schema_version": "resolver_capability_request.v1",
        "capability_kind": "task_resolution_request",
        "objective": "Retrieve one bounded resolver fact.",
        "reason": "The terminal response needs one grounded fact.",
        "priority": "now",
        "goal_continuation_ref": continuation_ref,
    }
    observation = {
        "schema_version": "resolver_observation.v1",
        "observation_id": "resolver-observation-live-1",
        "capability_kind": "task_resolution_request",
        "request_objective": resolver_request["objective"],
        "request_reason": resolver_request["reason"],
        "status": "succeeded",
        "prompt_safe_summary": "One bounded fact is available.",
        "evidence_refs": [{
            "schema_version": "evidence_ref.v1",
            "evidence_kind": "system_event",
            "evidence_id": "resolver-evidence-live-1",
            "owner": "cognition_resolver",
            "excerpt": "One bounded source-backed fact.",
            "observed_at": "2026-08-20T00:00:00Z",
        }],
        "task_resolution_evidence_state": {
            "schema_version": "resolver_evidence_state.v1",
            "state": "complete",
            "remaining_needs": [],
        },
        "goal_continuation_ref": continuation_ref,
        "created_at_utc": "2026-08-20T00:00:00Z",
    }
    terminal_output = {
        "state_update": {
            "state_scope": "user",
            "replacement_state": {"terminal": "replacement"},
        },
    }
    subgraph_calls: list[tuple[int, int, bool]] = []
    parent_child_epochs: list[int] = []
    successful_child_epochs: list[int] = []
    coordinator = guardrail.create_cognition_retry_coordinator(
        "live-resolver-parent-retry",
    )

    async def load_action_context(current_state: dict) -> dict:
        return {
            **current_state,
            "action_selection_context": {"coding_runs": []},
        }

    async def load_pending(current_state: dict) -> dict:
        return current_state

    async def run_cognition_subgraph(
        current_state: dict,
        *,
        commit: bool,
        retry_coordinator: object | None = None,
        **_kwargs: object,
    ) -> dict:
        assert retry_coordinator is coordinator

        async def run_child(
            _payload: CognitionCoreInputV2,
            _services: object,
        ) -> CognitionCoreOutputV2:
            parent_child_epochs.append(coordinator.epoch)
            if len(parent_child_epochs) == 1:
                raise CognitionExecutionError(
                    "one parent checkpoint replay is required",
                    error_code="goal_bid_structure_exhausted",
                    branch_id="ordinary_response",
                    stage="goal_cognition",
                    safe_checkpoint="pre_state_commit",
                    retryable=False,
                )
            successful_child_epochs.append(coordinator.epoch)
            return {"state_update": {"state_scope": "user"}}

        await guardrail.run_guarded_cognition(
            cognition_payload,
            object(),
            run_child=run_child,
        )
        resolver_state = current_state["resolver_state"]
        observations = resolver_state["observations"]
        subgraph_calls.append((
            resolver_state["cycle_index"],
            len(observations),
            commit,
        ))
        if observations:
            return {
                "cognition_core_output": terminal_output,
                "resolver_capability_requests": [],
                "goal_resolution": "answerable_now",
                "action_specs": [],
            }
        return {
            "cognition_core_output": {
                "state_update": {"state_scope": "user"},
            },
            "resolver_capability_requests": [resolver_request],
            "goal_resolution": "requires_required_evidence",
            "action_specs": [],
            "cognition_scene_context": build_scene_context_from_global_state(
                current_state
            ),
        }

    execute_capability = AsyncMock(return_value=observation)
    commit_output = AsyncMock()
    monkeypatch.setattr(
        persona_module,
        "_load_live_action_selection_context",
        load_action_context,
    )
    monkeypatch.setattr(
        persona_module,
        "load_matching_pending_resume_into_state",
        load_pending,
    )
    monkeypatch.setattr(
        persona_module,
        "call_cognition_subgraph",
        run_cognition_subgraph,
    )
    monkeypatch.setattr(
        persona_module,
        "execute_resolver_capability_request",
        execute_capability,
    )
    monkeypatch.setattr(
        persona_module,
        "commit_cognition_output",
        commit_output,
    )

    coordinator_token = guardrail.bind_cognition_retry_coordinator(coordinator)
    try:
        resolved = await persona_module.stage_1_goal_resolver(state)
    finally:
        guardrail.reset_cognition_retry_coordinator(coordinator_token)

    assert subgraph_calls == [(0, 0, False), (1, 1, False)]
    assert parent_child_epochs == [0, 1, 1]
    assert successful_child_epochs == [1, 1]
    execute_capability.assert_awaited_once()
    commit_output.assert_awaited_once_with(terminal_output)
    assert resolved["cognition_core_output"] is terminal_output
    assert resolved["cognition_state_committed"] is True


def test_post_i1_roster_uses_only_current_active_goal_statuses(
    cognition_payload: CognitionCoreInputV2,
) -> None:
    """Terminal or absent prior goals cannot authorize the next G1 roster."""

    session = create_cold_session(
        payload=cognition_payload,
        episode_id=cognition_payload["episode"]["episode_id"],
        owner_identity="recurrence-roster-test",
        ttl_seconds=60,
        current_roster=("social_care", "ordinary_response"),
    )
    assert session.current_roster == ("social_care", "ordinary_response")
    roster = v3_facade._post_i1_goal_roster({
        "goals": [
            {
                "goal_kind": "relationship_connection",
                "status": "pursuing",
            },
            {
                "goal_kind": "safety",
                "status": "satisfied",
            },
        ]
    })

    assert [definition.branch_id for definition in roster] == [
        "ordinary_response",
        "relationship_connection",
    ]
    active_safety_roster = v3_facade._post_i1_goal_roster({
        "goals": [{
            "goal_kind": "safety",
            "status": "pursuing",
        }]
    })
    assert [definition.branch_id for definition in active_safety_roster] == [
        "ordinary_response",
        "safety_coping",
    ]


@pytest.mark.asyncio
async def test_recurrence_zero_one_two_consumes_each_prior_replacement_state(
    cognition_payload: CognitionCoreInputV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cycle indexes advance only from each validated prior replacement."""

    _fresh_session_registry(monkeypatch)
    llm = _RecurrenceLLM()
    services = make_v3_services(llm)

    output_zero = await v3_facade.run_cognition(cognition_payload, services)
    input_one = _cycle_input(cognition_payload, output_zero, cycle_index=1)
    output_one = await v3_facade.run_cognition(input_one, services)
    input_two = _cycle_input(input_one, output_one, cycle_index=2)
    output_two = await v3_facade.run_cognition(input_two, services)

    validate_cognition_core_output(output_two)
    assert output_one["state_update"]["expected_previous_state"] == input_one[
        "mutable_state"
    ]
    assert output_two["state_update"]["expected_previous_state"] == input_two[
        "mutable_state"
    ]
    assert llm.calls.count("A1") == 1
    assert llm.calls.count("A2") == 1
    assert llm.calls.count("R.A1") == 2
    assert llm.calls.count("R.A2") == 2


@pytest.mark.asyncio
async def test_divergent_or_concurrently_claimed_session_cold_rebuilds_without_mixing_transcript(
    cognition_payload: CognitionCoreInputV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Divergence and a second owner take explicit independent cold paths."""

    _fresh_session_registry(monkeypatch)
    llm = _RecurrenceLLM()
    services = make_v3_services(llm)
    cold_output = await v3_facade.run_cognition(cognition_payload, services)
    divergent = _cycle_input(cognition_payload, cold_output, cycle_index=1)
    divergent["private_continuity_context"] = "different continuation"

    divergent_output = await v3_facade.run_cognition(divergent, services)
    assert any(
        warning.startswith("session_rebuilt_input_divergence:")
        for warning in divergent_output["diagnostics"]["warnings"]
    )
    assert llm.calls[-4:] == ["A1", "A2", "G1a", "P1"]

    _fresh_session_registry(monkeypatch)
    slow_llm = _RecurrenceLLM(block_first_tail=True)
    slow_services = make_v3_services(slow_llm)
    cold_output = await v3_facade.run_cognition(cognition_payload, slow_services)
    continuation = _cycle_input(cognition_payload, cold_output, cycle_index=1)
    first = asyncio.create_task(
        v3_facade.run_cognition(continuation, slow_services)
    )
    await slow_llm.tail_started.wait()
    second = asyncio.create_task(
        v3_facade.run_cognition(continuation, slow_services)
    )
    await asyncio.sleep(0)
    assert not second.done()
    slow_llm.release_tail.set()
    first_output, second_output = await asyncio.gather(first, second)

    validate_cognition_core_output(first_output)
    validate_cognition_core_output(second_output)
    assert "session_rebuilt_concurrent_owner" in second_output["diagnostics"][
        "warnings"
    ]
    assert slow_llm.calls.count("R.A1") == 1
    assert slow_llm.calls.count("A1") == 2
    assert slow_llm.maximum_primary_in_flight == 1


@pytest.mark.asyncio
async def test_parent_checkpoint_retry_preserves_branch_attempt_arithmetic_and_effect_idempotency(
    cognition_payload: CognitionCoreInputV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parent replay preserves attempts and finalizes one output per cycle.

    External connector-state commits remain outside this facade-level contract.
    """

    _fresh_session_registry(monkeypatch)
    llm = _RecurrenceLLM(authorize_effects=True)
    services = make_v3_services(llm, include_sidecar=True)
    ledger = create_v2_attempt_ledger("v3-recurrence-parent-retry")
    ledger_token = bind_v2_attempt_ledger(ledger, graph_attempt=1)
    coordinator = guardrail.create_cognition_retry_coordinator(
        "v3-recurrence-parent-retry",
    )
    coordinator_token = guardrail.bind_cognition_retry_coordinator(coordinator)
    child_epochs: list[int] = []

    async def run_child(
        payload: CognitionCoreInputV2,
        _services: object,
    ) -> CognitionCoreOutputV2:
        child_epochs.append(coordinator.epoch)
        output = await v3_facade.run_cognition(payload, services)
        if len(child_epochs) == 1:
            raise CognitionExecutionError(
                "ordinary bid needs the parent checkpoint replay",
                error_code="goal_bid_structure_exhausted",
                branch_id="ordinary_response",
                stage="goal_cognition",
                safe_checkpoint="pre_state_commit",
                retryable=False,
            )
        return output

    try:
        output_zero = await guardrail.run_guarded_cognition(
            cognition_payload,
            services,
            run_child=run_child,
        )
        input_one = _cycle_input(cognition_payload, output_zero, cycle_index=1)
        output_one = await guardrail.run_guarded_cognition(
            input_one,
            services,
            run_child=run_child,
        )
        guarded_snapshot = snapshot_v2_guarded_attempt_ledger()
    finally:
        guardrail.reset_cognition_retry_coordinator(coordinator_token)
        reset_v2_attempt_ledger(ledger_token)

    validate_cognition_core_output(output_one)
    assert child_epochs == [0, 1, 1]
    assert coordinator.epoch == 1
    assert [row["action_kind"] for row in output_zero["action_requests"]] == [
        "accepted_task_status_check"
    ]
    assert output_zero["resolver_requests"] == []
    assert output_one["action_requests"] == []
    assert [row["capability"] for row in output_one["resolver_requests"]] == [
        "approval_preparation"
    ]
    assert llm.calls.count("action_authorization") == 2
    assert llm.calls.count("resolver_authorization") == 1
    assert guarded_snapshot is not None
    assert [epoch["epoch"] for epoch in guarded_snapshot["epochs"]] == [0, 1]
    assert guarded_snapshot["epochs"][0]["attempts"]
    assert guarded_snapshot["epochs"][1]["attempts"]
    assert guarded_snapshot["parent_recovery"]["disposition"] == "recovered"
