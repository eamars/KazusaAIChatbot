"""Real-LLM regressions for relationship-context size recovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
import sys
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import (
    build_acquaintance_user_state,
    build_character_production_state,
    run_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    EVENT_FIELDS,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_relationship_context,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.cognition_core_v2_test_helpers import (
    canonical_episode,
    canonical_identity_context,
)
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


_NOW = "2026-08-06T00:00:00Z"
_RELATIONSHIP_CONTEXT_LIMIT = 900
_SOURCE_CORRELATION_ID = "chat:qq:ch_1f677493d7a52025:438485259"
_SOURCE_TRACE_ID = "llmtrace_175868f4ff924dfd8832229a600eea9f"
_CURRENT_EVENT = "@一之濑明日奈 ？到底是谁在叫的最大声呢！"


class _CapturingLLM:
    """Delegate to configured routes and retain raw and parsed outputs."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *args: object,
        config: object | None = None,
        **kwargs: object,
    ) -> Any:
        """Invoke one real model call and capture its inspectable boundary."""

        response = await self.delegate.ainvoke(
            messages,
            *args,
            config=config,
            **kwargs,
        )
        raw_output = str(getattr(response, "content", ""))
        parsed_output: object = {}
        parse_error = ""
        try:
            parsed_output = parse_llm_json_output(
                raw_output,
                deterministic_only=True,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
        self.calls.append({
            "route": {
                "stage_name": str(getattr(config, "stage_name", "")),
                "route_name": str(getattr(config, "route_name", "")),
                "model": str(getattr(config, "model", "")),
            },
            "messages": [
                str(getattr(message, "content", ""))
                for message in messages
            ],
            "raw_model_output": raw_output,
            "parsed_output": parsed_output,
            "parse_error": parse_error,
        })
        return response


def _serialized_size(value: Mapping[str, Any]) -> int:
    """Measure a context with the same encoding used by the strict guard."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return len(serialized)


def _summary(prefix: str, length: int) -> str:
    """Build bounded fixture text with a deterministic character count."""

    if length <= len(prefix):
        return prefix[:length]
    return prefix + "x" * (length - len(prefix))


def _relationship_event(
    *,
    entity_id: str,
    relationship_id: str,
    description: str,
    salience: int,
) -> dict[str, Any]:
    """Build one valid relationship-scoped event for the incident shape."""

    event = {
        "entity_id": entity_id,
        "description": description,
        "salience": salience,
        "role_refs": [{
            "role": "affected_relationship",
            "entity_kind": "relationship",
            "entity_id": relationship_id,
        }],
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": entity_id,
            "occurred_at": _NOW,
            "semantic_summary": description,
        }],
        "created_at": _NOW,
        "updated_at": _NOW,
        "status": "active",
        "outcome_impact": 0,
    }
    event.update({
        field_name: 0
        for field_name in EVENT_FIELDS - {"status", "outcome_impact"}
    })
    return event


def _incident_user_state() -> dict[str, Any]:
    """Build a synthetic state matching the observed 914-char case shape."""

    state = dict(build_acquaintance_user_state(
        global_user_id="context-limit-user",
        updated_at=_NOW,
    ))
    relationship_id = state["relationship"]["relationship_id"]
    state["active_events"] = [
        _relationship_event(
            entity_id="event-1",
            relationship_id=relationship_id,
            description=_summary(
                "当前关系事件仍然需要角色继续观察：",
                160,
            ),
            salience=90,
        ),
        _relationship_event(
            entity_id="event-2",
            relationship_id=relationship_id,
            description=_summary(
                "当前关系事件的后续影响尚未完全确定：",
                155,
            ),
            salience=80,
        ),
    ]
    validated_state = validate_cognition_state(state)
    return validated_state


def _raw_oversized_relationship_context() -> dict[str, Any]:
    """Build a synthetic legacy packet at the exact 914-char size."""

    state = build_acquaintance_user_state(
        global_user_id="context-limit-user",
        updated_at=_NOW,
    )
    relationship_id = state["relationship"]["relationship_id"]
    context: dict[str, Any] = {
        "schema_version": "relationship_operational_context.v1",
        "relationship_id": relationship_id,
        "axes": {
            field_name: state["relationship"][field_name]
            for field_name in (
                "familiarity",
                "positive_regard",
                "trust",
                "attachment",
                "desired_closeness",
                "perceived_closeness",
                "care",
                "boundary_safety",
                "exclusivity",
                "unresolved_injury",
                "salience",
            )
        },
        "causal_context": [
            {
                "entity_kind": "event",
                "semantic_summary": _summary("当前事件摘要A：", 160),
                "salience": "极高",
                "lifecycle": "active",
                "freshness": "即时",
            },
            {
                "entity_kind": "event",
                "semantic_summary": _summary("当前事件摘要B：", 155),
                "salience": "高",
                "lifecycle": "active",
                "freshness": "即时",
            },
        ],
        "affect": [],
        "relationship_freshness": "即时",
        "evidence_freshness": "无证据",
    }
    serialized_size = _serialized_size(context)
    if serialized_size != 914:
        raise AssertionError(
            "incident fixture must remain 914 characters; "
            f"observed {serialized_size}"
        )
    return context


def _input_payload(
    *,
    state: dict[str, Any],
    relationship_context: dict[str, Any],
) -> dict[str, Any]:
    """Build one minimal group-chat V2 input for the live bidding path."""

    character = build_character_production_state(updated_at=_NOW)
    episode_id = "context-size-live-episode"
    return {
        "schema_version": "cognition_core_input.v2",
        "episode": canonical_episode(
            episode_id=episode_id,
            content=_CURRENT_EVENT,
            current_global_user_id=state["owner_user_id"],
        ),
        "state_scope": "user",
        "mutable_state": state,
        "character_constraints": {
            "drives": character["drives"],
            "standards": character["standards"],
            "meaning_state": character["meaning_state"],
            "personality_judgment": {
                "logic": "evidence-led",
                "defense": "reserved under pressure",
                "quirks": "brief hesitation",
                "taboos": "preserve character agency",
            },
        },
        "character_identity_context": canonical_identity_context(),
        "evidence": [{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": f"episode:{episode_id}",
                "occurred_at": _NOW,
                "semantic_summary": _CURRENT_EVENT,
            },
            "semantic_text": _CURRENT_EVENT,
            "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS["episode"]),
        }],
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "resolver_context": "resolver_status=idle",
        "scene_context": {
            "channel_scope": "group",
            "character_role": "the addressed character",
            "current_user_role": "the current speaker",
            "semantic_scene": _CURRENT_EVENT,
            "public_group_scene": "A group conversation is active.",
            "conversation_continuity": "Continue only this current event.",
            "semantic_temporal_context": "immediate",
        },
        "private_continuity_context": (
            "Use the current event and relationship context only."
        ),
        "relationship_context": relationship_context,
    }


async def _run_case(
    *,
    case_id: str,
    input_payload: dict[str, Any],
    packet_size_before_consume: int,
) -> tuple[
    dict[str, Any] | None,
    dict[str, str] | None,
    list[dict[str, Any]],
    object,
]:
    """Run one live case and persist raw evidence before contract asserts."""

    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    output: dict[str, Any] | None = None
    failure: dict[str, str] | None = None
    try:
        result = await run_cognition(input_payload, services)
        output = dict(result)
    except Exception as exc:
        failure = {
            "error_class": type(exc).__name__,
            "message": str(exc),
        }
    trace_path = write_llm_trace(
        "cognition_core_v2_context_size_live_llm",
        case_id,
        {
            "input_kind": "synthetic_incident_shape_reconstruction",
            "source_correlation_id": _SOURCE_CORRELATION_ID,
            "source_trace_id": _SOURCE_TRACE_ID,
            "source_evidence_note": (
                "The protected source trace records the contract failure and "
                "metadata, but not the full relationship packet. The 914 "
                "character packet is reconstructed from the RCA shape."
            ),
            "failure_mode": "relationship operational context is oversized",
            "packet_size_before_consume": packet_size_before_consume,
            "input_payload": input_payload,
            "model_calls": capturing_llm.calls,
            "output": output,
            "failure": failure,
            "contract_expectations": {
                "producer_size_max": _RELATIONSHIP_CONTEXT_LIMIT,
                "consumer_recovers_size_only_overflow": True,
                "required_model_path": "goal bidding",
                "required_evidence_handle": "e1",
            },
        },
    )
    return output, failure, capturing_llm.calls, trace_path


@pytest.mark.live_llm
async def test_live_relationship_producer_fits_before_goal_bidding() -> None:
    """The producer must fit the incident shape before the live bid path."""

    state = _incident_user_state()
    relationship_context = project_relationship_context(
        state,
        effective_at=_NOW,
    )
    producer_size = _serialized_size(relationship_context)
    payload = _input_payload(
        state=state,
        relationship_context=relationship_context,
    )
    output, failure, calls, trace_path = await _run_case(
        case_id="producer_fits_914_char_incident_before_bidding",
        input_payload=payload,
        packet_size_before_consume=producer_size,
    )

    assert trace_path.exists()
    assert producer_size <= _RELATIONSHIP_CONTEXT_LIMIT, (
        f"producer emitted {producer_size} chars; trace={trace_path}"
    )
    assert failure is None, f"live cognition failed; trace={trace_path}"
    assert output is not None
    assert any(
        "goal" in call["route"]["stage_name"]
        for call in calls
    )


@pytest.mark.live_llm
async def test_live_relationship_consumer_recovers_oversized_packet() -> None:
    """The consumer must fit a legacy 914-char packet before validation."""

    state = dict(build_acquaintance_user_state(
        global_user_id="context-limit-user",
        updated_at=_NOW,
    ))
    relationship_context = _raw_oversized_relationship_context()
    consumer_input_size = _serialized_size(relationship_context)
    payload = _input_payload(
        state=state,
        relationship_context=relationship_context,
    )
    output, failure, calls, trace_path = await _run_case(
        case_id="consumer_recovers_914_char_legacy_packet_before_bidding",
        input_payload=payload,
        packet_size_before_consume=consumer_input_size,
    )

    assert trace_path.exists()
    assert consumer_input_size == 914
    assert consumer_input_size > _RELATIONSHIP_CONTEXT_LIMIT
    assert failure is None, f"consumer rejected overflow; trace={trace_path}"
    assert output is not None
    assert any(
        "goal" in call["route"]["stage_name"]
        for call in calls
    )
