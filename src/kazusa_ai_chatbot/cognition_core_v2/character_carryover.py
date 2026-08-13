"""Bounded source-free character operational carry-over cognition."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timezone
from typing import Any, Literal

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot.cognition_core_v2.output_projection import (
    build_state_update,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_semantic_appraisals,
    apply_state_update,
)
from kazusa_ai_chatbot.config import (
    COGNITION_LLM_CHARACTER_CARRYOVER_API_KEY,
    COGNITION_LLM_CHARACTER_CARRYOVER_BASE_URL,
    COGNITION_LLM_CHARACTER_CARRYOVER_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_CHARACTER_CARRYOVER_MODEL,
    COGNITION_LLM_CHARACTER_CARRYOVER_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMInvoker,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime
from kazusa_ai_chatbot.utils import parse_llm_json_output


CHARACTER_CARRYOVER_SCHEMA = "character_carryover_decision.v1"
CHARACTER_CARRYOVER_RESULT_SCHEMA = "character_carryover_result.v2"
CHARACTER_CARRYOVER_ATTEMPT_LIMIT = 3
CHARACTER_CARRYOVER_OUTPUT_CHAR_LIMIT = 8000
CHARACTER_CARRYOVER_DYNAMIC_CHAR_LIMIT = 8000
CHARACTER_CARRYOVER_INPUT_CHAR_LIMIT = 16000
CHARACTER_CARRYOVER_PROPOSITION_LIMIT = 4
CHARACTER_CARRYOVER_DELTA_LIMIT = 4
CHARACTER_CARRYOVER_EVIDENCE_LIMIT = 4
CHARACTER_CARRYOVER_REASON_LIMIT = 160
CHARACTER_CARRYOVER_DELTA_LIMIT_ABSOLUTE = 40
CHARACTER_CARRYOVER_SEMANTIC_VALUE_LIMIT = 500
CHARACTER_CARRYOVER_DEADLINE_SECONDS = 45.0

_CARRYOVER_REASON_CODES = frozenset({
    "no_lingering_effect",
    "already_represented",
    "transient_scene_only",
    "unsupported",
    "lingering_character_effect",
})
_CARRYOVER_CANDIDATE_KINDS = frozenset({
    "event",
    "threat",
    "knowledge_gap",
})
_ALLOWED_AXES = {
    "event": frozenset({
        "outcome_impact",
        "responsibility",
        "intentionality",
        "harm",
        "unfairness",
        "exposure",
        "repair_need",
        "reparability",
        "expectation_mismatch",
        "norm_violation",
        "contamination_risk",
        "identity_threat",
        "comparison_gap",
        "vastness",
        "memory_warmth",
        "temporal_loss",
    }),
    "threat": frozenset({
        "likelihood",
        "expected_harm",
        "uncertainty",
        "controllability",
        "coping_potential",
        "residual_pressure",
    }),
    "knowledge_gap": frozenset({
        "relevance",
        "uncertainty",
        "learnability",
        "novelty",
        "model_accommodation",
    }),
}


CHARACTER_CARRYOVER_PROMPT = '''You decide whether a settled episode leaves a
source-free operational effect on the active character. Use only the provided
opaque evidence handles and closed candidate kinds. Never output an emotion,
cause class, relationship change, identity change, source text, user identity,
quote, promise, or style instruction. Native deterministic reduction derives
all emotions from accepted causal state.

# Generation Procedure
First decide whether an enduring character-level effect exists. If none exists,
return no_change. If one exists, choose one candidate kind and up to four
numeric axis deltas that describe only a source-free operational consequence.
Every chosen proposition must cite retained evidence handles. Do not invent
handles or fields.

# Axis Selection
For an event candidate, select only the allowed axes that directly describe
the observable consequence in the supplied evidence:
- deliberate obstruction, imposed harm, or a violated boundary: use harm and,
  when supported, unfairness and/or intentionality;
- an enduring irreversible loss: use a negative outcome_impact;
- contamination or rejection of a basic norm: use contamination_risk and/or
  norm_violation.
Do not substitute a neutral or unrelated axis when the evidence directly
supports one of these observations. Each selected axis remains an
evidence-backed observation, not an emotion, cause class, inferred actor, or
added fact. Omit an axis when the evidence does not support it.
For every proposition, assign one to three semantic roles using only the
provided handles: actor, experiencer, target, or object paired with self,
unspecified_other, or group_context. These roles describe the causal structure
of the proposition; they do not identify a person or create a relationship.
If the input includes replacement_error_code, replace the complete prior
proposal while preserving the same evidence-grounded semantic context.

# Output Format
Return exactly one JSON object.
- action: "no_change" or "apply".
- reason_code: one of no_lingering_effect, already_represented,
  transient_scene_only, unsupported, lingering_character_effect.
- question_id: "character_carryover".
- propositions: [] for no_change, or one to four objects for apply.
Each proposition object has exactly kind, semantic_value, evidence_handles,
role_assignments, and deltas. kind is event, threat, or knowledge_gap.
semantic_value is a short source-free label. evidence_handles contains one or
more supplied handles. role_assignments contains one to three unique objects
with exactly role and entity_handle keys. role is actor, experiencer, target,
or object; entity_handle is self, unspecified_other, or group_context.
deltas is an object whose keys are allowed axes for that kind and whose values
are integers from -40 through 40. At least one delta must be non-zero after
bounded normalization.
'''


@dataclass(frozen=True)
class CharacterCarryoverServicesV1:
    """Own the explicit route and invoker for one carry-over decision."""

    llm: LLMInvoker
    config: LLMCallConfig


@dataclass(frozen=True)
class CharacterCarryoverDecisionV1:
    """Record the accepted semantic disposition before native reduction."""

    schema_version: str
    action: Literal["no_change", "apply"]
    reason_code: str
    privacy_disposition: Literal["source_free", "unsafe"]
    semantic_appraisal: dict[str, Any] | None


@dataclass(frozen=True)
class CharacterCarryoverResultV2:
    """Return zero or one character-state update with a bounded disposition."""

    schema_version: str
    decision: CharacterCarryoverDecisionV1
    state_update: dict[str, Any] | None
    disposition: Literal["no_change", "apply", "degraded"]
    error_code: str | None
    attempts: int


def build_character_carryover_services() -> CharacterCarryoverServicesV1:
    """Build the dedicated bounded route for character carry-over."""

    if COGNITION_LLM_CHARACTER_CARRYOVER_MAX_COMPLETION_TOKENS > 8192:
        raise ValueError("character carry-over completion cap exceeds 8192")
    config = LLMCallConfig(
        stage_name="cognition_core_v2.character_carryover",
        route_name="COGNITION_LLM_CHARACTER_CARRYOVER",
        base_url=COGNITION_LLM_CHARACTER_CARRYOVER_BASE_URL,
        api_key=COGNITION_LLM_CHARACTER_CARRYOVER_API_KEY,
        model=COGNITION_LLM_CHARACTER_CARRYOVER_MODEL,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        max_completion_tokens=(
            COGNITION_LLM_CHARACTER_CARRYOVER_MAX_COMPLETION_TOKENS
        ),
        presence_penalty=None,
        thinking=LLMThinkingConfig(
            enabled=COGNITION_LLM_CHARACTER_CARRYOVER_THINKING_ENABLED,
        ),
    )
    return CharacterCarryoverServicesV1(llm=LLInterface(), config=config)


async def run_character_carryover_cognition(
    *,
    source_episode_id: str,
    evidence: Sequence[Mapping[str, Any]],
    base_state: Mapping[str, Any],
    effective_at: str,
    services: CharacterCarryoverServicesV1,
) -> CharacterCarryoverResultV2:
    """Evaluate one settled episode and return a native character update.

    Args:
        source_episode_id: Durable idempotency root for the settled episode.
        evidence: Ref-complete current-episode rows selected by the router.
        base_state: Latest character cognition state after predecessor waiting.
        effective_at: UTC timestamp used for the resulting replacement state.
        services: Explicit model invoker and route configuration.

    Returns:
        A no-change, applied, or degraded result with no partial state update.
    """

    normalized_evidence = _normalize_evidence(
        source_episode_id=source_episode_id,
        evidence=evidence,
    )
    if not normalized_evidence:
        result = _no_change_result(
            reason_code="no_lingering_effect",
        )
        return result
    payload = _build_dynamic_payload(
        evidence=normalized_evidence,
        base_state=base_state,
    )
    payload_text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(payload_text) > CHARACTER_CARRYOVER_DYNAMIC_CHAR_LIMIT:
        result = _degraded_result(
            reason_code="unsupported",
            error_code="input_limit",
            attempts=0,
        )
        return result
    if (
        len(CHARACTER_CARRYOVER_PROMPT) + len(payload_text)
        > CHARACTER_CARRYOVER_INPUT_CHAR_LIMIT
    ):
        result = _degraded_result(
            reason_code="unsupported",
            error_code="input_limit",
            attempts=0,
        )
        return result
    system_message = SystemMessage(content=CHARACTER_CARRYOVER_PROMPT)
    deadline_at = time.monotonic() + CHARACTER_CARRYOVER_DEADLINE_SECONDS
    replacement_error_code: str | None = None
    for attempt in range(1, CHARACTER_CARRYOVER_ATTEMPT_LIMIT + 1):
        attempt_payload = payload
        if replacement_error_code is not None:
            attempt_payload = dict(payload)
            attempt_payload["replacement_error_code"] = replacement_error_code
        attempt_payload_text = json.dumps(
            attempt_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(attempt_payload_text) > CHARACTER_CARRYOVER_DYNAMIC_CHAR_LIMIT:
            return _degraded_result(
                reason_code="unsupported",
                error_code="input_limit",
                attempts=attempt - 1,
            )
        if (
            len(CHARACTER_CARRYOVER_PROMPT) + len(attempt_payload_text)
            > CHARACTER_CARRYOVER_INPUT_CHAR_LIMIT
        ):
            return _degraded_result(
                reason_code="unsupported",
                error_code="input_limit",
                attempts=attempt - 1,
            )
        human_message = HumanMessage(content=attempt_payload_text)
        remaining_seconds = deadline_at - time.monotonic()
        if remaining_seconds <= 0:
            return _degraded_result(
                reason_code="unsupported",
                error_code="deadline_exceeded",
                attempts=attempt - 1,
            )
        try:
            response = await asyncio.wait_for(
                services.llm.ainvoke(
                    [system_message, human_message],
                    config=services.config,
                ),
                timeout=remaining_seconds,
            )
        except asyncio.TimeoutError:
            if attempt == CHARACTER_CARRYOVER_ATTEMPT_LIMIT:
                result = _degraded_result(
                    reason_code="unsupported",
                    error_code="deadline_exceeded",
                    attempts=attempt,
                )
                return result
            continue
        except (httpx.HTTPError, OpenAIError):
            if attempt == CHARACTER_CARRYOVER_ATTEMPT_LIMIT:
                result = _degraded_result(
                    reason_code="unsupported",
                    error_code="provider_exhausted",
                    attempts=attempt,
                )
                return result
            continue
        raw_output = response.content
        if not isinstance(raw_output, str):
            raw_output = str(raw_output)
        if len(raw_output) > CHARACTER_CARRYOVER_OUTPUT_CHAR_LIMIT:
            if attempt == CHARACTER_CARRYOVER_ATTEMPT_LIMIT:
                result = _degraded_result(
                    reason_code="unsupported",
                    error_code="output_limit",
                    attempts=attempt,
                )
                return result
            continue
        try:
            parsed_output = parse_llm_json_output(
                raw_output,
                deterministic_only=True,
            )
        except (TypeError, ValueError):
            if attempt == CHARACTER_CARRYOVER_ATTEMPT_LIMIT:
                return _degraded_result(
                    reason_code="unsupported",
                    error_code="contract_exhausted",
                    attempts=attempt,
                )
            continue
        if _contains_forbidden_model_authority(parsed_output):
            result = _unsafe_result(attempts=attempt)
            return result
        decision_payload = _validate_decision_payload(
            parsed_output,
            evidence=normalized_evidence,
        )
        if decision_payload is None:
            if attempt == CHARACTER_CARRYOVER_ATTEMPT_LIMIT:
                result = _degraded_result(
                    reason_code="unsupported",
                    error_code="contract_exhausted",
                    attempts=attempt,
                )
                return result
            continue
        if decision_payload["action"] == "no_change":
            result = _no_change_result(
                reason_code=decision_payload["reason_code"],
                attempts=attempt,
            )
            return result
        state_update = _reduce_apply_decision(
            base_state=base_state,
            effective_at=effective_at,
            decision_payload=decision_payload,
            evidence=normalized_evidence,
        )
        if state_update is None:
            if attempt == CHARACTER_CARRYOVER_ATTEMPT_LIMIT:
                result = _degraded_result(
                    reason_code="unsupported",
                    error_code="state_rejected",
                    attempts=attempt,
                )
                return result
            replacement_error_code = "state_rejected"
            continue
        decision = CharacterCarryoverDecisionV1(
            schema_version=CHARACTER_CARRYOVER_SCHEMA,
            action="apply",
            reason_code=decision_payload["reason_code"],
            privacy_disposition="source_free",
            semantic_appraisal=decision_payload["semantic_appraisal"],
        )
        result = CharacterCarryoverResultV2(
            schema_version=CHARACTER_CARRYOVER_RESULT_SCHEMA,
            decision=decision,
            state_update=state_update,
            disposition="apply",
            error_code=None,
            attempts=attempt,
        )
        return result
    raise AssertionError("character carry-over attempt loop did not terminate")


def _normalize_evidence(
    *,
    source_episode_id: str,
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep up to four complete current-episode evidence rows in source order."""

    normalized_rows: list[dict[str, Any]] = []
    seen_handles: set[str] = set()
    for index, row in enumerate(evidence, start=1):
        evidence_ref = row.get("evidence_ref")
        if isinstance(evidence_ref, Mapping):
            source_kind = evidence_ref.get("source_kind")
            source_id = evidence_ref.get("source_id")
            occurred_at = evidence_ref.get("occurred_at")
            semantic_summary = evidence_ref.get("semantic_summary")
        else:
            source_kind = row.get("source_kind")
            source_id = row.get("source_id")
            occurred_at = row.get("occurred_at")
            semantic_summary = row.get("semantic_summary")
        handle = row.get("evidence_handle")
        if not isinstance(handle, str) or not handle.strip():
            handle = f"evidence:{index}"
        if (
            handle in seen_handles
            or not isinstance(source_kind, str)
            or not isinstance(source_id, str)
            or not isinstance(occurred_at, str)
            or not isinstance(semantic_summary, str)
            or not source_kind.strip()
            or not source_id.strip()
            or not occurred_at.strip()
            or not semantic_summary.strip()
        ):
            continue
        try:
            native_occurred_at = _native_utc_z(occurred_at)
        except (TypeError, ValueError):
            continue
        seen_handles.add(handle)
        text = row.get("semantic_text", semantic_summary)
        if not isinstance(text, str) or not text.strip():
            text = semantic_summary
        normalized_rows.append({
            "evidence_handle": handle,
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": source_episode_id,
                "occurred_at": native_occurred_at,
                "semantic_summary": "character operational event",
            },
            "semantic_text": text.strip(),
            "visible_to": ["character_carryover"],
        })
        if len(normalized_rows) == CHARACTER_CARRYOVER_EVIDENCE_LIMIT:
            break
    return normalized_rows


def _build_dynamic_payload(
    *,
    evidence: Sequence[Mapping[str, Any]],
    base_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the fixed source-free model payload for one operational decision."""

    payload = {
        "question": {
            "question_id": "character_carryover",
            "question_kind": "character_carryover",
        },
        "evidence": [
            {
                "evidence_handle": row["evidence_handle"],
                "semantic_text": row["semantic_text"],
            }
            for row in evidence
        ],
        "candidate_kinds": sorted(_CARRYOVER_CANDIDATE_KINDS),
        "allowed_axes": {
            kind: sorted(axes)
            for kind, axes in _ALLOWED_AXES.items()
        },
        "state_summary": {
            "active_event_count": len(base_state["active_events"]),
            "active_threat_count": len(base_state["threats"]),
            "open_gap_count": len(base_state["knowledge_gaps"]),
        },
    }
    return payload


def _contains_forbidden_model_authority(value: Any) -> bool:
    """Reject model output that claims native emotion or cause authority."""

    if isinstance(value, Mapping):
        if {"emotion_id", "cause_class"}.intersection(value):
            return True
        return any(
            _contains_forbidden_model_authority(nested)
            for nested in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_model_authority(item) for item in value)
    return False


def _validate_decision_payload(
    parsed: Mapping[str, Any],
    *,
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Validate one bounded model decision without semantic reinterpretation."""

    if not isinstance(parsed, Mapping):
        return None
    if set(parsed) != {
        "action",
        "reason_code",
        "question_id",
        "propositions",
    }:
        return None
    action = parsed["action"]
    reason_code = parsed["reason_code"]
    question_id = parsed["question_id"]
    propositions = parsed["propositions"]
    if (
        not isinstance(action, str)
        or not isinstance(reason_code, str)
        or action not in {"no_change", "apply"}
        or reason_code not in _CARRYOVER_REASON_CODES
        or question_id != "character_carryover"
        or not isinstance(propositions, list)
    ):
        return None
    if action == "no_change":
        if propositions:
            return None
        return {
            "action": "no_change",
            "reason_code": reason_code,
            "semantic_appraisal": None,
        }
    if not 1 <= len(propositions) <= CHARACTER_CARRYOVER_PROPOSITION_LIMIT:
        return None
    allowed_handles = {
        row["evidence_handle"]
        for row in evidence
    }
    normalized_propositions: list[dict[str, Any]] = []
    total_delta_count = 0
    for proposition in propositions:
        if (
            not isinstance(proposition, Mapping)
            or set(proposition) != {
                "kind",
                "semantic_value",
                "evidence_handles",
                "role_assignments",
                "deltas",
            }
        ):
            return None
        kind = proposition["kind"]
        semantic_value = proposition["semantic_value"]
        evidence_handles = proposition["evidence_handles"]
        role_assignments = proposition["role_assignments"]
        deltas = proposition["deltas"]
        stripped_semantic_value = (
            semantic_value.strip()
            if isinstance(semantic_value, str)
            else ""
        )
        if (
            not isinstance(kind, str)
            or kind not in _CARRYOVER_CANDIDATE_KINDS
            or not isinstance(semantic_value, str)
            or not stripped_semantic_value
            or (
                len(stripped_semantic_value)
                > CHARACTER_CARRYOVER_SEMANTIC_VALUE_LIMIT
            )
            or not isinstance(evidence_handles, list)
            or not isinstance(deltas, Mapping)
            or not evidence_handles
            or any(
                not isinstance(handle, str) or not handle.strip()
                for handle in evidence_handles
            )
            or not set(evidence_handles).issubset(allowed_handles)
            or len(set(evidence_handles)) != len(evidence_handles)
        ):
            return None
        if not isinstance(role_assignments, list) or not 1 <= len(
            role_assignments
        ) <= 3:
            return None
        normalized_roles: list[dict[str, str]] = []
        seen_roles: set[tuple[str, str]] = set()
        seen_role_values: set[str] = set()
        for assignment in role_assignments:
            role = (
                assignment.get("role")
                if isinstance(assignment, Mapping)
                else None
            )
            entity_handle = (
                assignment.get("entity_handle")
                if isinstance(assignment, Mapping)
                else None
            )
            if (
                not isinstance(assignment, Mapping)
                or set(assignment) != {"role", "entity_handle"}
                or not isinstance(role, str)
                or role not in {
                    "actor",
                    "experiencer",
                    "target",
                    "object",
                }
                or not isinstance(entity_handle, str)
                or entity_handle not in {
                    "self",
                    "unspecified_other",
                    "group_context",
                }
            ):
                return None
            role_key = (
                role,
                entity_handle,
            )
            if role_key in seen_roles or role in seen_role_values:
                return None
            seen_roles.add(role_key)
            seen_role_values.add(role)
            normalized_roles.append({
                "role": role,
                "entity_handle": entity_handle,
            })
        normalized_deltas: dict[str, int] = {}
        for axis, raw_delta in deltas.items():
            if (
                axis not in _ALLOWED_AXES[kind]
                or isinstance(raw_delta, bool)
                or not isinstance(raw_delta, int)
            ):
                return None
            normalized_deltas[axis] = max(
                -CHARACTER_CARRYOVER_DELTA_LIMIT_ABSOLUTE,
                min(CHARACTER_CARRYOVER_DELTA_LIMIT_ABSOLUTE, raw_delta),
            )
        if not normalized_deltas or not any(normalized_deltas.values()):
            return None
        total_delta_count += len(normalized_deltas)
        if total_delta_count > CHARACTER_CARRYOVER_DELTA_LIMIT:
            return None
        normalized_propositions.append({
            "kind": kind,
            "semantic_value": stripped_semantic_value,
            "evidence_handles": list(evidence_handles),
            "role_assignments": normalized_roles,
            "deltas": normalized_deltas,
        })
    semantic_appraisal = {
        "question_id": "character_carryover",
        "selected_evidence_handles": sorted({
            handle
            for proposition in normalized_propositions
            for handle in proposition["evidence_handles"]
        }),
        "selected_role_handles": sorted({
            assignment["entity_handle"]
            for proposition in normalized_propositions
            for assignment in proposition["role_assignments"]
        }),
        "propositions": normalized_propositions,
        "deltas": [
            {
                "kind": proposition["kind"],
                "axis": axis,
                "delta": delta,
                "evidence_handles": proposition["evidence_handles"],
            }
            for proposition in normalized_propositions
            for axis, delta in proposition["deltas"].items()
        ],
    }
    return {
        "action": "apply",
        "reason_code": reason_code,
        "semantic_appraisal": semantic_appraisal,
    }


def _reduce_apply_decision(
    *,
    base_state: Mapping[str, Any],
    effective_at: str,
    decision_payload: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Trial-reduce a valid source-free proposal through the native reducer."""

    appraisal = decision_payload["semantic_appraisal"]
    if not isinstance(appraisal, Mapping):
        return None
    semantic_result, handle_to_ref = _build_native_appraisal(
        appraisal,
        evidence=evidence,
    )
    try:
        native_effective_at = _native_utc_z(effective_at)
        preliminary_state = apply_semantic_appraisals(
            base_state,
            [semantic_result],
            evidence,
            handle_to_ref,
        )
        replacement_state = apply_state_update(
            preliminary_state,
            elapsed_seconds=0,
            updated_at=native_effective_at,
        )
    except ValueError:
        return None
    state_update = build_state_update(base_state, replacement_state)
    if not state_update["changed_paths"]:
        return None
    return state_update


def _native_utc_z(value: str) -> str:
    """Normalize storage UTC text to the native UTC-Z state format."""

    parsed = parse_storage_utc_datetime(value)
    native_timestamp = parsed.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )
    return native_timestamp


def _build_native_appraisal(
    appraisal: Mapping[str, Any],
    *,
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Translate the carry-over decision into existing appraisal reducer input."""

    propositions = appraisal["propositions"]
    native_propositions: list[dict[str, Any]] = []
    native_deltas: list[dict[str, Any]] = []
    handle_to_ref: dict[str, dict[str, str]] = {
        "self": {
            "scope": "character",
            "kind": "character",
            "entity_id": "character:global",
        },
        "unspecified_other": {
            "scope": "character",
            "kind": "third_party",
            "entity_id": "operational:unspecified_other",
        },
        "group_context": {
            "scope": "character",
            "kind": "group",
            "entity_id": "operational:group_context",
        },
    }
    for index, proposition in enumerate(propositions, start=1):
        kind = proposition["kind"]
        evidence_handles = proposition["evidence_handles"]
        handle = f"candidate:{kind}:{evidence_handles[0]}"
        prompt_handle = f"cc{index}"
        handle_to_ref[prompt_handle] = {
            "scope": "character",
            "kind": kind,
            "entity_id": handle,
        }
        native_propositions.append({
            "proposition_kind": "operational_observation",
            "subject_handle": prompt_handle,
            "evidence_handles": evidence_handles,
            "role_assignments": [
                {
                    "role": assignment["role"],
                    "entity_handle": assignment["entity_handle"],
                }
                for assignment in proposition["role_assignments"]
            ],
            "semantic_value": proposition["semantic_value"],
        })
        for axis, delta in proposition["deltas"].items():
            native_deltas.append({
                "target_path": f"{_field_for_kind(kind)}.{prompt_handle}.{axis}",
                "delta": delta,
                "evidence_handles": evidence_handles,
                "reason": "character operational carry-over",
            })
    semantic_result = {
        "question_id": "character_carryover",
        "selected_evidence_handles": [
            row["evidence_handle"]
            for row in evidence
            if row["evidence_handle"] in appraisal["selected_evidence_handles"]
        ],
        "selected_role_handles": appraisal["selected_role_handles"],
        "propositions": native_propositions,
        "deltas": native_deltas,
        "explanation": "source-free operational carry-over",
    }
    return semantic_result, handle_to_ref


def _field_for_kind(kind: str) -> str:
    """Return the state-list field owned by one carry-over candidate kind."""

    field_name = {
        "event": "active_events",
        "threat": "threats",
        "knowledge_gap": "knowledge_gaps",
    }[kind]
    return field_name


def _no_change_result(
    *,
    reason_code: str,
    attempts: int = 0,
) -> CharacterCarryoverResultV2:
    """Build the typed terminal result for a valid no-change decision."""

    decision = CharacterCarryoverDecisionV1(
        schema_version=CHARACTER_CARRYOVER_SCHEMA,
        action="no_change",
        reason_code=reason_code,
        privacy_disposition="source_free",
        semantic_appraisal=None,
    )
    result = CharacterCarryoverResultV2(
        schema_version=CHARACTER_CARRYOVER_RESULT_SCHEMA,
        decision=decision,
        state_update=None,
        disposition="no_change",
        error_code=None,
        attempts=attempts,
    )
    return result


def _degraded_result(
    *,
    reason_code: str,
    error_code: str,
    attempts: int,
) -> CharacterCarryoverResultV2:
    """Build a fail-closed result that carries no state update."""

    decision = CharacterCarryoverDecisionV1(
        schema_version=CHARACTER_CARRYOVER_SCHEMA,
        action="no_change",
        reason_code=reason_code,
        privacy_disposition="source_free",
        semantic_appraisal=None,
    )
    result = CharacterCarryoverResultV2(
        schema_version=CHARACTER_CARRYOVER_RESULT_SCHEMA,
        decision=decision,
        state_update=None,
        disposition="degraded",
        error_code=error_code,
        attempts=attempts,
    )
    return result


def _unsafe_result(*, attempts: int) -> CharacterCarryoverResultV2:
    """Build a privacy-rejected result for forbidden model authority claims."""

    decision = CharacterCarryoverDecisionV1(
        schema_version=CHARACTER_CARRYOVER_SCHEMA,
        action="no_change",
        reason_code="unsupported",
        privacy_disposition="unsafe",
        semantic_appraisal=None,
    )
    result = CharacterCarryoverResultV2(
        schema_version=CHARACTER_CARRYOVER_RESULT_SCHEMA,
        decision=decision,
        state_update=None,
        disposition="degraded",
        error_code="privacy_rejected",
        attempts=attempts,
    )
    return result


__all__ = [
    "CharacterCarryoverDecisionV1",
    "CharacterCarryoverResultV2",
    "CharacterCarryoverServicesV1",
    "build_character_carryover_services",
    "run_character_carryover_cognition",
]
