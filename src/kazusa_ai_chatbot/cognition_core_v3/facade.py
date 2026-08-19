"""V3 public cognition engine over the unchanged V2 substrate.

The engine runs one parallel first wave of registry-ordered appraisal chains
plus isolated goal chains for every preliminary branch kind, reduces the
accepted prefix into a provisional state, runs the fresh canonical terminal
outcome stage on that reduction, applies final reduction and relationship
maintenance once, then reactivates dependency-ready branch kinds before the
complete-bid join. Every wave shares one invocation-wide attempt ledger whose
per-stage caps mirror the V2 model attempt policy, so global arithmetic stays
exact while chain failures stay isolated.

The deterministic head (elapsed update, preliminary goals, prompt projection,
question planning), final reduction, relationship maintenance, workspace
collapse, action planning, and output assembly ride on the unchanged V2
substrate helpers, so the public contract remains exactly
``CognitionCoreInputV2``/``CognitionCoreOutputV2``. Goal chains fail closed per
kind with a typed error code; required-branch escalation, partial failure
surfacing, and protected replay capture reuse the V2 behavior verbatim.

Accepted appraisal content bridges into native state through code-owned rows:
propositions attach to their source evidence row's candidate event root so the
native materializer resolves them through causal-event provenance, except
terminal outcome propositions whose subject binds to the unique lifecycle-
eligible entity of the asserted kind. Axis deltas translate into exact native
increment rows only when their axis matches exactly one permitted concrete
path for the stage's authorized evidence domain; every unbound or ambiguous
delta is dropped with a deterministic warning instead of reaching the native
reducer. Role assignments stay empty because the producer contract carries no
role data, recorded as a documented parity gap in the package README.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import replace
from typing import Any

import httpx
from openai import OpenAIError

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    select_final_branches,
    select_preliminary_branches,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    ActionBidV2,
    BranchDefinition,
    CognitionCoreInputV2,
    CognitionCoreOutputV2,
    CognitionCoreServicesV2,
    CognitionExecutionError,
    CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS,
    is_targetless_group_self_cognition_episode,
    validate_cognition_core_input,
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    capture_validation_event,
)
from kazusa_ai_chatbot.cognition_core_v2.facade import (
    _apply_final_relationship_maintenance,
    _bids_with_live_goals,
    _bind_pending_resolution,
    _branch_context,
    _build_cognition_observability,
    _cognition_elapsed_seconds,
    _deduplicate_diagnostics_warnings,
    _empty_collapse,
    _elapsed_ms,
    _episode_updated_at,
    _fact_without_producer,
    _goal_for_branch,
    _goal_projection,
    _mark_cognition_partial_failures,
    _native_relationship_context,
    _ordinary_relational_decision,
    _raise_for_unrecoverable_required_branch_failures,
    _reduce_appraisals_with_isolation,
    _resolver_progress,
    _selected_bid,
    _workspace_current_event,
    _workspace_goal_contexts,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    _materialize_recurrence_relational_willingness,
)
from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
    V2_APPRAISAL_TOTAL_ATTEMPTS,
    V2_MODEL_TOTAL_ATTEMPTS,
    bind_v2_attempt_ledger,
    create_v2_attempt_ledger,
    current_v2_attempt_ledger,
    reset_v2_attempt_ledger,
    snapshot_v2_attempt_ledger,
)
from kazusa_ai_chatbot.cognition_core_v2.output_projection import (
    build_state_update,
    default_expression_policy,
    project_affect,
    project_relationship,
)
from kazusa_ai_chatbot.cognition_core_v2.parallel_executor import (
    BranchFailure,
    ParallelExecutionResult,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    _goal_outcome_eligible_entities,
    _index_projected_handles,
    _permitted_delta_paths,
    plan_semantic_questions,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    ENTITY_LIST_FIELDS,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_state_update,
    create_deterministic_goals,
)
from kazusa_ai_chatbot.cognition_episode import (
    project_dialog_response_operation,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ResolverValidationError,
    validate_current_turn_relational_willingness,
)
from kazusa_ai_chatbot.llm_interface.contracts import LLMCallConfig
from kazusa_ai_chatbot.llm_interface.detection import detect_backend_descriptor
from kazusa_ai_chatbot.llm_tracing import failure_capsule
from kazusa_ai_chatbot.utils import parse_llm_json_output

from kazusa_ai_chatbot.cognition_core_v3.action_selection import plan_actions
from kazusa_ai_chatbot.cognition_core_v3.appraisal import (
    FAMILY_DELTA_AXES,
    FAMILY_IDENTITY_CATEGORY_SETS,
    FAMILY_IDENTITY_OPTIONAL_CATEGORY_SETS,
    STATIC_APPRAISAL_SYSTEM_PROMPT,
    build_family_question_tail,
    build_repair_instruction,
    build_terminal_outcome_request,
    classify_appaisal_candidate,
    reduce_appraisal_results,
    render_accepted_context,
)
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
    CacheDomainIdentity,
    PROVIDER_FAILURE_CLASS,
    StageResult,
    STRUCTURAL_FAILURE_CLASS,
    hash_credential_identity,
    hash_static_prompt,
)
from kazusa_ai_chatbot.cognition_core_v3.diagnostics import (
    build_chain_trace_record,
)
from kazusa_ai_chatbot.cognition_core_v3.execution import (
    AttemptLedger,
    ChainOutcome,
    ChainTaskSpec,
    StageAttemptOutcome,
    start_wave,
)
from kazusa_ai_chatbot.cognition_core_v3.goal_cognition import (
    ORDINARY_GOAL_KIND,
    STATIC_GOAL_SYSTEM_PROMPT,
    bind_selected_response_operation,
    build_goal_question_tail,
    build_goal_repair_instruction,
    project_conversation_progress_evidence,
    project_goal_evidence_row,
    project_required_selection_operations,
    resolve_goal_disposition,
    validate_goal_bid_draft,
    validate_selection_goal_draft,
)
from kazusa_ai_chatbot.cognition_core_v3.registry import (
    APPRAISAL_FIRST_WAVE_CHAINS,
    ChainSpec,
    GOAL_CHAINS,
    TERMINAL_OUTCOME_CHAIN,
)
from kazusa_ai_chatbot.cognition_core_v3.transcript import (
    TranscriptState,
    build_repair_request,
    domain_matches,
    extend_accepted,
    start_chain,
    start_fresh_from_checkpoint,
    to_invoker_messages,
)
from kazusa_ai_chatbot.cognition_core_v3.workspace import (
    collapse_authoritative_relational_bid,
    collapse_bids_via_partition,
)

_PROVIDER_EXCEPTIONS = (
    OpenAIError,
    httpx.HTTPError,
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
)
_CANONICALIZE_EXCEPTIONS = (AttributeError, KeyError, TypeError, ValueError)

_STAGE_CONFIG_FIELDS: dict[str, str] = {
    "event_agency": "appraisal_event_agency_config",
    "moral_identity": "appraisal_moral_identity_config",
    "relationship_social": "appraisal_relationship_social_config",
    "epistemic_comparison_memory": (
        "appraisal_epistemic_comparison_memory_config"
    ),
    "existential_drive": "appraisal_existential_drive_config",
    "goal_threat_outcome": "appraisal_goal_threat_outcome_config",
}

_CURRENT_PROTECTED_CHAIN_SCOPE: ContextVar[
    list[dict[str, Any]] | None
] = ContextVar("cognition_v3_protected_chain_records", default=None)


def bind_protected_chain_records() -> Token:
    """Bind a fresh per-trace protected chain record scope.

    The bound scope owns every trace record appended while it stays active in
    the current execution context; ``run_cognition`` binds one when none is
    already active so concurrent traces keep isolated record sets. Returns the
    reset token for ``reset_protected_chain_records``.
    """

    return _CURRENT_PROTECTED_CHAIN_SCOPE.set([])


def snapshot_protected_chain_records() -> tuple[dict[str, Any], ...]:
    """Return the active scope's protected chain trace records in order.

    Records carry full internal content (raw model output and parsed
    candidates); callers project one record through
    ``project_protected_chain_failure`` or ``project_protected_chain_result``
    before exposing it publicly. Returns an empty tuple when no scope is bound.
    """

    records = _CURRENT_PROTECTED_CHAIN_SCOPE.get()
    return tuple(records) if records is not None else ()


def reset_protected_chain_records(token: Token) -> None:
    """Drop the protected chain record scope bound by ``token``."""

    _CURRENT_PROTECTED_CHAIN_SCOPE.reset(token)


def _stage_config(stage_name: str, services: CognitionCoreServicesV2) -> LLMCallConfig:
    """Resolve one registered stage's injected configuration by name."""

    return getattr(services, _STAGE_CONFIG_FIELDS[stage_name])


def _goal_stage_config(
    goal_kind: str, services: CognitionCoreServicesV2
) -> LLMCallConfig:
    """Route a goal chain to the V2 ordinary or active-branch configuration.

    Required-selection episodes route through the ordinary configuration just
    like the ordinary branch itself, matching the V2 routing rule.
    """

    if goal_kind == ORDINARY_GOAL_KIND:
        return services.goal_ordinary_response_config
    return services.goal_active_branch_config


def _cache_domain_identity(
    config: LLMCallConfig, static_system_prompt: str
) -> CacheDomainIdentity:
    """Derive the route cache-domain identity for one stage's model call.

    Every component resolves deterministically from the injected route
    configuration and the family-owned byte-identical static prompt, so two
    attempts of one stage agree on their domain while a routed change to URL,
    credential, backend kind, model, or template strategy changes it.
    """

    descriptor = detect_backend_descriptor(config=config, generation=1)
    return CacheDomainIdentity(
        normalized_backend_url=descriptor.normalized_base_url,
        credential_identity_hash=hash_credential_identity(config.api_key),
        backend_kind=descriptor.backend_kind,
        model=descriptor.model,
        template_strategy=descriptor.thinking_strategy,
        static_system_prompt_hash=hash_static_prompt(static_system_prompt),
    )


def _family_state_projection(
    question_kind: str,
    identity_by_question: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Project one family's prompt-safe identity categories in sorted order.

    Optional-category families fall back through their registered variant sets
    toward the smallest available projection; required-category families fail
    fast when a category is missing from the projected context.
    """

    if question_kind not in identity_by_question:
        raise ValueError(f"family {question_kind} has no projected identity context")
    context = identity_by_question[question_kind]
    if question_kind in FAMILY_IDENTITY_OPTIONAL_CATEGORY_SETS:
        available_categories = set(context)
        chosen: frozenset[str] = frozenset()
        for variant in reversed(
            list(FAMILY_IDENTITY_OPTIONAL_CATEGORY_SETS[question_kind])
        ):
            if variant <= available_categories:
                chosen = variant
                break
    else:
        required = FAMILY_IDENTITY_CATEGORY_SETS[question_kind]
        missing = required - set(context)
        if missing:
            raise ValueError(
                f"family {question_kind} identity categories are incomplete: "
                f"{sorted(missing)}"
            )
        chosen = required
    return {category: context[category] for category in sorted(chosen)}


def _record_chain_trace(
    *,
    chain_name: str,
    stage_id: str,
    config: LLMCallConfig | None,
    system_prompt: str,
    human_payload: str,
    raw_output: str | None,
    parsed_output: object,
    parse_status: str,
    started_at: float,
    ended_at: float,
    branch_id: str | None = None,
    attempt_number: int = 1,
    error: str | None = None,
) -> None:
    """Append one protected chain trace record to the active per-trace scope.

    Producers called outside a bound scope (standalone unit tests) append
    nothing, so every recorded trace belongs to exactly one invocation; bind
    ``bind_protected_chain_records`` first when capture is required.
    """

    records = _CURRENT_PROTECTED_CHAIN_SCOPE.get()
    if records is None:
        return
    records.append(
        build_chain_trace_record(
            chain_name=chain_name,
            stage_id=stage_id,
            config=config,
            system_prompt=system_prompt,
            human_payload=human_payload,
            raw_output=raw_output,
            parsed_output=parsed_output,
            parse_status=parse_status,
            started_at=started_at,
            ended_at=ended_at,
            branch_id=branch_id,
            attempt_number=attempt_number,
            error=error,
        )
    )


def _classify_stage_candidate(
    question_kind: str,
    candidate: object,
    evidence_handles: Sequence[str],
    accepted_local_state: Mapping[str, Any] | None,
) -> StageAttemptOutcome:
    """Classify one parsed stage candidate against the family admission contract.

    The permitted delta axes come from the registered family so a producer can
    never widen its own write surface beyond what the chain owns.
    """

    return classify_appaisal_candidate(
        question_kind,
        candidate,
        evidence_handles,
        permitted_delta_paths=frozenset(FAMILY_DELTA_AXES[question_kind]),
        accepted_local_state=accepted_local_state,
    )


def _make_appraisal_stage_producers(
    services: CognitionCoreServicesV2,
    projection: Any,
    question_by_kind: Mapping[str, Mapping[str, Any]],
    chain: ChainSpec,
) -> dict[str, StageProducer]:
    """Bind transcript-isolated stage producers for one appraisal chain.

    Every producer in the returned mapping shares one chain-local transcript
    state: accepted continuations preserve their prefix byte-for-byte across
    this chain's stages only, a structural rejection answers with a bounded
    local repair request carrying the exact contract error and closed allowed
    values, provider failures retry the stage's base request on the next
    attempt, and a route cache-domain mismatch restarts from the canonical
    checkpoint tail. Stages without an authorized question keep the
    deterministic empty accepted state without any model call; their projection
    travels through later tails' accepted-prefix summaries instead of assistant
    text.
    """

    stages_with_questions = [
        stage for stage in chain.stages if stage in question_by_kind
    ]
    next_question_stage: dict[str, str | None] = {}
    position = 0
    while position < len(stages_with_questions):
        current_stage = stages_with_questions[position]
        following_stage = (
            stages_with_questions[position + 1]
            if position + 1 < len(stages_with_questions)
            else None
        )
        next_question_stage[current_stage] = following_stage
        position += 1

    transcript: dict[str, TranscriptState | None] = {"current": None}
    stage_base: dict[str, TranscriptState] = {}
    pending_repair: dict[str, tuple[str, str | None]] = {}

    def _question_tail(stage_name: str, accepted_prefix) -> str:
        question = question_by_kind[stage_name]
        return build_family_question_tail(
            stage_name,
            _family_state_projection(
                stage_name,
                projection.identity_by_question,
            ),
            question["evidence_handles"],
            render_accepted_context(accepted_prefix),
        )

    async def produce(context) -> StageAttemptOutcome:
        stage_name = context.stage_name
        question = question_by_kind.get(stage_name)
        if question is None:
            return StageAttemptOutcome(
                accepted=True,
                local_state={
                    "selected_evidence_handles": [],
                    "propositions": [],
                    "deltas": [],
                },
                semantic_summary=None,
            )
        config = _stage_config(stage_name, services)
        identity = _cache_domain_identity(
            config, STATIC_APPRAISAL_SYSTEM_PROMPT
        )
        tail = _question_tail(stage_name, context.accepted_prefix)
        sent_payload = tail
        current = transcript["current"]
        if current is None:
            current = start_chain(
                STATIC_APPRAISAL_SYSTEM_PROMPT, tail, identity
            )
            transcript["current"] = current
            stage_base[stage_name] = current
            messages = to_invoker_messages(
                current.messages,
                static_system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
            )
        elif not domain_matches(current, identity):
            current = start_fresh_from_checkpoint(current, tail, identity)
            transcript["current"] = current
            stage_base[stage_name] = current
            pending_repair.pop(stage_name, None)
            messages = to_invoker_messages(
                current.messages,
                static_system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
            )
        else:
            base_state = stage_base.get(stage_name)
            if base_state is None:
                stage_base[stage_name] = current
                base_state = current
                messages = to_invoker_messages(
                    base_state.messages,
                    static_system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
                )
            else:
                repair_entry = pending_repair.pop(stage_name, None)
                if repair_entry is not None and repair_entry[0]:
                    invalid_raw, failure_detail = repair_entry
                    instruction = build_repair_instruction(
                        stage_name, failure_detail
                    )
                    sent_payload = instruction
                    messages = to_invoker_messages(
                        build_repair_request(base_state, invalid_raw, instruction),
                        static_system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
                    )
                else:
                    messages = to_invoker_messages(
                        base_state.messages,
                        static_system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
                    )
        started_at = time.monotonic()
        try:
            response = await services.llm.ainvoke(messages, config=config)
            raw_output = getattr(response, "content", "")
        except _PROVIDER_EXCEPTIONS as exc:
            _record_chain_trace(
                chain_name=context.chain_name,
                stage_id=f"{context.chain_name}:{stage_name}",
                config=config,
                system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
                human_payload=sent_payload,
                raw_output=None,
                parsed_output=None,
                parse_status="provider_error",
                started_at=started_at,
                ended_at=time.monotonic(),
                attempt_number=context.attempt_number,
                error=str(exc),
            )
            return StageAttemptOutcome(
                accepted=False,
                local_state=None,
                semantic_summary=None,
                failure_class=PROVIDER_FAILURE_CLASS,
            )
        ended_at = time.monotonic()
        try:
            candidate = parse_llm_json_output(raw_output, deterministic_only=True)
        except _CANONICALIZE_EXCEPTIONS as exc:
            parse_detail = f"原始输出无法解析为 JSON 对象：{exc}"
            pending_repair[stage_name] = (raw_output, parse_detail)
            _record_chain_trace(
                chain_name=context.chain_name,
                stage_id=f"{context.chain_name}:{stage_name}",
                config=config,
                system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
                human_payload=sent_payload,
                raw_output=raw_output,
                parsed_output=None,
                parse_status="parse_failed",
                started_at=started_at,
                ended_at=ended_at,
                attempt_number=context.attempt_number,
                error=str(exc),
            )
            return StageAttemptOutcome(
                accepted=False,
                local_state=None,
                semantic_summary=None,
                failure_class=STRUCTURAL_FAILURE_CLASS,
                detail=parse_detail,
            )
        accepted_states = [
            result.local_state
            for result in context.accepted_prefix
            if result.accepted and isinstance(result.local_state, Mapping)
        ]
        outcome = _classify_stage_candidate(
            stage_name,
            candidate,
            question["evidence_handles"],
            accepted_states[-1] if accepted_states else None,
        )
        _record_chain_trace(
            chain_name=context.chain_name,
            stage_id=f"{context.chain_name}:{stage_name}",
            config=config,
            system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
            human_payload=sent_payload,
            raw_output=raw_output,
            parsed_output=candidate,
            parse_status="accepted" if outcome.accepted else "rejected",
            started_at=started_at,
            ended_at=time.monotonic(),
            attempt_number=context.attempt_number,
            error=None if outcome.accepted else outcome.detail,
        )
        if not outcome.accepted:
            if outcome.failure_class == STRUCTURAL_FAILURE_CLASS:
                pending_repair[stage_name] = (raw_output, outcome.detail)
            return outcome
        base_state = stage_base[stage_name]
        following_stage = next_question_stage.get(stage_name)
        if following_stage is None:
            extension_tail: str | None = None
        else:
            successor_result = StageResult(
                chain_name=chain.name,
                stage_name=stage_name,
                accepted=True,
                local_state=outcome.local_state,
                semantic_summary=outcome.semantic_summary,
            )
            successor_prefix = context.accepted_prefix + (successor_result,)
            extension_tail = _question_tail(following_stage, successor_prefix)
        transcript["current"] = extend_accepted(
            base_state, raw_output, extension_tail
        )
        return outcome

    return {stage_name: produce for stage_name in chain.stages}


def _make_terminal_stage_producer(
    services: CognitionCoreServicesV2,
    provisional_state: Any,
    evidence_handles: Sequence[str],
    question_by_kind: Mapping[str, Mapping[str, Any]],
) -> dict[str, StageProducer]:
    """Bind the fresh canonical terminal-outcome stage producer.

    The terminal chain runs on a clean transcript: no prior wave history enters
    its tail beyond the accepted-prefix reduction and typed omissions recorded
    by the provisional state it is built from. Structural rejections answer with
    a bounded local repair request; provider failures retry the base request.
    When the planner planned no ``goal_threat_outcome`` question for this
    episode, the stage keeps the deterministic contentless accepted state
    without any model call, mirroring wave-A stages without an authorized
    question.
    """

    outcome_question = question_by_kind.get("goal_threat_outcome")
    transcript: dict[str, TranscriptState | None] = {"current": None}
    stage_base: dict[str, TranscriptState] = {}
    pending_repair: dict[str, tuple[str, str | None]] = {}

    async def produce(context) -> StageAttemptOutcome:
        if outcome_question is None:
            return StageAttemptOutcome(
                accepted=True,
                local_state={
                    "selected_evidence_handles": [],
                    "propositions": [],
                    "deltas": [],
                },
                semantic_summary=None,
            )
        stage_name = context.stage_name
        config = _stage_config(stage_name, services)
        identity = _cache_domain_identity(
            config, STATIC_APPRAISAL_SYSTEM_PROMPT
        )
        tail = build_terminal_outcome_request(provisional_state, evidence_handles)
        sent_payload = tail
        current = transcript["current"]
        if current is None:
            current = start_chain(
                STATIC_APPRAISAL_SYSTEM_PROMPT, tail, identity
            )
            transcript["current"] = current
            stage_base[stage_name] = current
            messages = to_invoker_messages(
                current.messages,
                static_system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
            )
        elif not domain_matches(current, identity):
            current = start_fresh_from_checkpoint(current, tail, identity)
            transcript["current"] = current
            stage_base[stage_name] = current
            pending_repair.pop(stage_name, None)
            messages = to_invoker_messages(
                current.messages,
                static_system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
            )
        else:
            base_state = stage_base.get(stage_name)
            if base_state is None:
                stage_base[stage_name] = current
                base_state = current
                messages = to_invoker_messages(
                    base_state.messages,
                    static_system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
                )
            else:
                repair_entry = pending_repair.pop(stage_name, None)
                if repair_entry is not None and repair_entry[0]:
                    invalid_raw, failure_detail = repair_entry
                    instruction = build_repair_instruction(
                        stage_name, failure_detail
                    )
                    sent_payload = instruction
                    messages = to_invoker_messages(
                        build_repair_request(base_state, invalid_raw, instruction),
                        static_system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
                    )
                else:
                    messages = to_invoker_messages(
                        base_state.messages,
                        static_system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
                    )
        started_at = time.monotonic()
        try:
            response = await services.llm.ainvoke(messages, config=config)
            raw_output = getattr(response, "content", "")
        except _PROVIDER_EXCEPTIONS as exc:
            _record_chain_trace(
                chain_name=context.chain_name,
                stage_id=f"{context.chain_name}:{stage_name}",
                config=config,
                system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
                human_payload=sent_payload,
                raw_output=None,
                parsed_output=None,
                parse_status="provider_error",
                started_at=started_at,
                ended_at=time.monotonic(),
                attempt_number=context.attempt_number,
                error=str(exc),
            )
            return StageAttemptOutcome(
                accepted=False,
                local_state=None,
                semantic_summary=None,
                failure_class=PROVIDER_FAILURE_CLASS,
            )
        ended_at = time.monotonic()
        try:
            candidate = parse_llm_json_output(raw_output, deterministic_only=True)
        except _CANONICALIZE_EXCEPTIONS as exc:
            parse_detail = f"原始输出无法解析为 JSON 对象：{exc}"
            pending_repair[stage_name] = (raw_output, parse_detail)
            _record_chain_trace(
                chain_name=context.chain_name,
                stage_id=f"{context.chain_name}:{stage_name}",
                config=config,
                system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
                human_payload=sent_payload,
                raw_output=raw_output,
                parsed_output=None,
                parse_status="parse_failed",
                started_at=started_at,
                ended_at=ended_at,
                attempt_number=context.attempt_number,
                error=str(exc),
            )
            return StageAttemptOutcome(
                accepted=False,
                local_state=None,
                semantic_summary=None,
                failure_class=STRUCTURAL_FAILURE_CLASS,
                detail=parse_detail,
            )
        outcome = _classify_stage_candidate(
            stage_name, candidate, evidence_handles, None
        )
        _record_chain_trace(
            chain_name=context.chain_name,
            stage_id=f"{context.chain_name}:{stage_name}",
            config=config,
            system_prompt=STATIC_APPRAISAL_SYSTEM_PROMPT,
            human_payload=sent_payload,
            raw_output=raw_output,
            parsed_output=candidate,
            parse_status="accepted" if outcome.accepted else "rejected",
            started_at=started_at,
            ended_at=time.monotonic(),
            attempt_number=context.attempt_number,
            error=None if outcome.accepted else outcome.detail,
        )
        if not outcome.accepted:
            if outcome.failure_class == STRUCTURAL_FAILURE_CLASS:
                pending_repair[stage_name] = (raw_output, outcome.detail)
        return outcome

    return {stage_name: produce for stage_name in TERMINAL_OUTCOME_CHAIN.stages}


def _validate_relational_carrier(
    episode_id: str,
    carrier: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Revalidate the current-turn relational carrier before code-carry.

    Recurrence turns carry an authoritative stance instead of asking the model
    to re-decide it; a carrier that cannot be validated against the episode is
    a typed pre-state-commit failure rather than a semantic guess.
    """

    if not isinstance(episode_id, str) or not episode_id.strip():
        raise CognitionExecutionError(
            "current-turn relational carrier requires episode identity",
            error_code="current_turn_relational_carrier_invalid",
            branch_id="ordinary_response",
            stage="goal_cognition",
            attempt_count=0,
            safe_checkpoint="pre_state_commit",
            retryable=False,
        )
    try:
        return validate_current_turn_relational_willingness(
            carrier,
            episode_id=episode_id,
        )
    except (ResolverValidationError, KeyError, TypeError) as exc:
        raise CognitionExecutionError(
            f"current-turn relational carrier is invalid: {exc}",
            error_code="current_turn_relational_carrier_invalid",
            branch_id="ordinary_response",
            stage="goal_cognition",
            attempt_count=0,
            safe_checkpoint="pre_state_commit",
            retryable=False,
        ) from exc


def _goal_semantic_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Filter the branch context into prompt-safe goal-tail content.

    Mirrors the V2 semantic-context contract: private underscore keys and the
    separately rendered fields (evidence, projection, role summaries, appraisal
    summaries) are excluded; character constraints drop their judgment field
    and scene context drops its raw role labels so role facts stay handle-only.
    """
    filtered = {
        key: value
        for key, value in context.items()
        if not str(key).startswith("_")
        and key
        not in {
            "evidence",
            "goal_projection",
            "role_summaries",
            "appraisal_summaries",
        }
    }
    constraints = filtered.get("character_constraints")
    if isinstance(constraints, Mapping):
        filtered["character_constraints"] = {
            key: value
            for key, value in constraints.items()
            if key != "personality_judgment"
        }
    scene_context = filtered.get("scene_context")
    if isinstance(scene_context, Mapping):
        filtered["scene_context"] = {
            key: value
            for key, value in scene_context.items()
            if key not in {"character_role", "current_user_role"}
        }
    return filtered


def _make_goal_stage_producer(
    services: CognitionCoreServicesV2,
    definition: BranchDefinition,
    state: Mapping[str, Any],
    context: Mapping[str, Any],
    payload: Mapping[str, Any],
):
    """Bind one isolated goal chain producer over a wave's semantic inputs.

    The tail carries this kind's goal projection, the full authorized evidence
    domain with projected content, action tendencies and branch intent, role
    handles with summaries, the filtered semantic context (identity,
    constraints, affect, relationship state, continuity fields), accepted
    appraisal summaries and required-selection facts in selection mode;
    required-selection mode is an episode-driven input fact carried by code
    through its bounded draft contract. Recurrence turns with an authoritative
    current-turn carrier revalidate it before any model call and materialize
    that stance onto the accepted draft instead of accepting a fresh
    model-decided one.
    """

    goal_kind = definition.goal_kind
    config = _goal_stage_config(goal_kind, services)
    evidence_handles = [row["evidence_handle"] for row in payload["evidence"]]
    episode_evidence_handles = {
        row["evidence_handle"]
        for row in payload["evidence"]
        if row["evidence_ref"]["source_kind"]
        in CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS
    }
    selection_operations = project_required_selection_operations(
        payload["evidence"],
    )
    if len(selection_operations) > 1:
        raise CognitionExecutionError(
            "episode carries multiple required selection operations"
        )
    selection_mode = (
        goal_kind == ORDINARY_GOAL_KIND and bool(selection_operations)
    )
    role_bindings: Mapping[str, Mapping[str, str]] = context["_role_bindings"]
    goal = _goal_for_branch(state, goal_kind)
    progress_evidence = (
        project_conversation_progress_evidence(payload["evidence"])
        if selection_mode else []
    )
    partitioned_handles = {
        operation["evidence_handle"] for operation in selection_operations
    } | {row["evidence_handle"] for row in progress_evidence}
    tail_rows = (
        [
            row for row in payload["evidence"]
            if row["evidence_handle"] not in partitioned_handles
        ]
        if selection_mode else list(payload["evidence"])
    )
    tail = build_goal_question_tail(
        goal_kind,
        _goal_projection(goal, goal_kind),
        evidence_handles,
        selection_mode=selection_mode,
        action_tendencies=list(definition.action_tendencies),
        branch_intent_guidance=(
            definition.branch_intent_guidance
            if not selection_mode and goal_kind != ORDINARY_GOAL_KIND else ""
        ),
        role_bindings=role_bindings,
        role_summaries=context["role_summaries"],
        semantic_context=_goal_semantic_context(context),
        appraisal_summaries=list(context["appraisal_summaries"]),
        evidence_rows=[project_goal_evidence_row(row) for row in tail_rows],
        selection_operations=(
            list(selection_operations) if selection_mode else []
        ),
        progress_evidence=progress_evidence,
    )
    transcript: dict[str, TranscriptState | None] = {"current": None}
    stage_base: dict[str, TranscriptState] = {}
    pending_repair: dict[str, tuple[str, str | None]] = {}

    async def produce(context_) -> StageAttemptOutcome:
        if goal_kind == ORDINARY_GOAL_KIND:
            relational_carrier = payload.get(
                "current_turn_relational_willingness"
            )
            cycle_index = payload.get("resolver_cycle_index", 0)
            if (
                cycle_index > 0
                and not isinstance(relational_carrier, Mapping)
            ):
                raise CognitionExecutionError(
                    "current-turn relational carrier is missing on recurrence"
                )
        identity = _cache_domain_identity(config, STATIC_GOAL_SYSTEM_PROMPT)
        sent_payload = tail
        current = transcript["current"]
        if current is None:
            current = start_chain(
                STATIC_GOAL_SYSTEM_PROMPT, tail, identity
            )
            transcript["current"] = current
            stage_base[goal_kind] = current
            messages = to_invoker_messages(
                current.messages, static_system_prompt=STATIC_GOAL_SYSTEM_PROMPT
            )
        elif not domain_matches(current, identity):
            current = start_fresh_from_checkpoint(current, tail, identity)
            transcript["current"] = current
            stage_base[goal_kind] = current
            pending_repair.pop(goal_kind, None)
            messages = to_invoker_messages(
                current.messages, static_system_prompt=STATIC_GOAL_SYSTEM_PROMPT
            )
        else:
            base_state = stage_base.get(goal_kind)
            if base_state is None:
                stage_base[goal_kind] = current
                base_state = current
                messages = to_invoker_messages(
                    base_state.messages,
                    static_system_prompt=STATIC_GOAL_SYSTEM_PROMPT,
                )
            else:
                repair_entry = pending_repair.pop(goal_kind, None)
                if repair_entry is not None and repair_entry[0]:
                    invalid_raw, failure_detail = repair_entry
                    instruction = build_goal_repair_instruction(
                        goal_kind,
                        selection_mode,
                        failure_detail,
                        frozenset(role_bindings),
                    )
                    sent_payload = instruction
                    messages = to_invoker_messages(
                        build_repair_request(base_state, invalid_raw, instruction),
                        static_system_prompt=STATIC_GOAL_SYSTEM_PROMPT,
                    )
                else:
                    messages = to_invoker_messages(
                        base_state.messages,
                        static_system_prompt=STATIC_GOAL_SYSTEM_PROMPT,
                    )
        started_at = time.monotonic()
        try:
            response = await services.llm.ainvoke(messages, config=config)
            raw_output = getattr(response, "content", "")
        except _PROVIDER_EXCEPTIONS as exc:
            _record_chain_trace(
                chain_name=goal_kind,
                stage_id=f"{goal_kind}:{goal_kind}",
                config=config,
                system_prompt=STATIC_GOAL_SYSTEM_PROMPT,
                human_payload=sent_payload,
                raw_output=None,
                parsed_output=None,
                parse_status="provider_error",
                started_at=started_at,
                ended_at=time.monotonic(),
                branch_id=definition.branch_id,
                attempt_number=context_.attempt_number,
                error=str(exc),
            )
            return StageAttemptOutcome(
                accepted=False,
                local_state=None,
                semantic_summary=None,
                failure_class=PROVIDER_FAILURE_CLASS,
            )
        ended_at = time.monotonic()
        try:
            candidate = parse_llm_json_output(raw_output, deterministic_only=True)
        except _CANONICALIZE_EXCEPTIONS as exc:
            parse_detail = f"原始输出无法解析为 JSON 对象：{exc}"
            pending_repair[goal_kind] = (raw_output, parse_detail)
            _record_chain_trace(
                chain_name=goal_kind,
                stage_id=f"{goal_kind}:{goal_kind}",
                config=config,
                system_prompt=STATIC_GOAL_SYSTEM_PROMPT,
                human_payload=sent_payload,
                raw_output=raw_output,
                parsed_output=None,
                parse_status="parse_failed",
                started_at=started_at,
                ended_at=ended_at,
                branch_id=definition.branch_id,
                attempt_number=context_.attempt_number,
                error=str(exc),
            )
            return StageAttemptOutcome(
                accepted=False,
                local_state=None,
                semantic_summary=None,
                failure_class=STRUCTURAL_FAILURE_CLASS,
                detail=parse_detail,
            )
        evidence_domain = set(evidence_handles)
        role_domain = set(role_bindings)
        try:
            if selection_mode:
                draft = validate_selection_goal_draft(
                    candidate,
                    evidence_handles=evidence_domain,
                    role_handles=role_domain,
                )
                bound_operation = bind_selected_response_operation(
                    draft["selected_response_operation"],
                    selection_operations[0]["response_operation"],
                )
                local_state: dict[str, Any] = dict(draft)
                local_state["selected_response_operation"] = bound_operation
            else:
                local_state = validate_goal_bid_draft(
                    candidate,
                    goal_kind=goal_kind,
                    evidence_handles=evidence_domain,
                    role_handles=role_domain,
                )
        except ValueError as exc:
            pending_repair[goal_kind] = (raw_output, str(exc))
            _record_chain_trace(
                chain_name=goal_kind,
                stage_id=f"{goal_kind}:{goal_kind}",
                config=config,
                system_prompt=STATIC_GOAL_SYSTEM_PROMPT,
                human_payload=sent_payload,
                raw_output=raw_output,
                parsed_output=candidate,
                parse_status="rejected",
                started_at=started_at,
                ended_at=time.monotonic(),
                branch_id=definition.branch_id,
                attempt_number=context_.attempt_number,
                error=str(exc),
            )
            return StageAttemptOutcome(
                accepted=False,
                local_state=None,
                semantic_summary=None,
                failure_class=STRUCTURAL_FAILURE_CLASS,
                detail=str(exc),
            )
        if goal_kind == ORDINARY_GOAL_KIND:
            relational_carrier = payload.get(
                "current_turn_relational_willingness"
            )
            if isinstance(relational_carrier, Mapping):
                validated_carrier = _validate_relational_carrier(
                    payload["episode"]["episode_id"],
                    relational_carrier,
                )
                local_state["relational_willingness"] = (
                    _materialize_recurrence_relational_willingness(
                        validated_carrier,
                        episode_evidence_handles,
                    )
                )
        _record_chain_trace(
            chain_name=goal_kind,
            stage_id=f"{goal_kind}:{goal_kind}",
            config=config,
            system_prompt=STATIC_GOAL_SYSTEM_PROMPT,
            human_payload=sent_payload,
            raw_output=raw_output,
            parsed_output=candidate,
            parse_status="accepted",
            started_at=started_at,
            ended_at=time.monotonic(),
            branch_id=definition.branch_id,
            attempt_number=context_.attempt_number,
        )
        return StageAttemptOutcome(
            accepted=True,
            local_state=local_state,
            semantic_summary=None,
        )

    return produce


def _materialize_goal_bid(
    definition: BranchDefinition,
    state: Mapping[str, Any],
    role_bindings: Mapping[str, Mapping[str, str]],
    local_state: Mapping[str, Any],
) -> ActionBidV2:
    """Map one accepted goal draft into the exact V2 action bid shape.

    The goal reference derivation and target-role materialization mirror the
    V2 branch handler; selection mode maps its bounded fields onto the V2
    intention triple while carrying the code-bound operation, and ordinary
    drafts carry their validated or carrier-materialized relational stance.
    """

    goal_kind = definition.goal_kind
    goal = _goal_for_branch(state, goal_kind)
    if goal is None:
        goal_ref: dict[str, str] = {
            "scope": state["state_scope"],
            "kind": "goal",
            "entity_id": f"goal:{goal_kind}:episode",
        }
    else:
        goal_ref = {
            "scope": state["state_scope"],
            "kind": "goal",
            "entity_id": goal["entity_id"],
        }
    target_roles = [
        dict(role_bindings[handle])
        for handle in local_state["target_role_handles"]
    ]
    if "selection" in local_state:
        selection_text = local_state["selection"]
        bid: ActionBidV2 = {
            "branch_id": definition.branch_id,
            "goal_ref": goal_ref,
            "intention": selection_text,
            "desired_outcome": selection_text,
            "concrete_detail": selection_text,
            "reason": local_state["reason"],
            "private_monologue": local_state["private_monologue"],
            "target_roles": target_roles,
            "evidence_handles": list(local_state["evidence_handles"]),
            "expected_consequences": list(
                local_state["expected_consequences"]
            ),
            "confidence": local_state["confidence"],
        }
        bid["selected_response_operation"] = dict(
            local_state["selected_response_operation"]
        )
    else:
        bid = {
            "branch_id": definition.branch_id,
            "goal_ref": goal_ref,
            "intention": local_state["intention"],
            "desired_outcome": local_state["desired_outcome"],
            "concrete_detail": local_state["concrete_detail"],
            "reason": local_state["reason"],
            "private_monologue": local_state["private_monologue"],
            "target_roles": target_roles,
            "evidence_handles": list(local_state["evidence_handles"]),
            "expected_consequences": list(
                local_state["expected_consequences"]
            ),
            "confidence": local_state["confidence"],
        }
    if (
        goal_kind == ORDINARY_GOAL_KIND
        and "relational_willingness" in local_state
    ):
        bid["relational_willingness"] = dict(
            local_state["relational_willingness"]
        )
    return bid


def _join_goal_wave(
    definitions: Sequence[BranchDefinition],
    outcomes_by_kind: Mapping[str, ChainOutcome | None],
    state: Mapping[str, Any],
    role_bindings: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, ActionBidV2], dict[str, str]]:
    """Materialize one wave's goal chain outcomes into V2 action bids.

    Available kinds produce materialized bids keyed by goal kind; unavailable
    kinds fail closed with the exact typed error code of their last recorded
    failure and no cross-kind substitution, so a required-selection exhaustion
    stays closed without falling back to any other branch. A kind whose chain
    crashed or was cancelled carries the generic contract-exhaustion code in
    its synthesized failure record; the exception class name travels with the
    deterministic v3_chain_unavailable warning instead.
    """

    bids_by_kind: dict[str, ActionBidV2] = {}
    unavailable_kinds: dict[str, str] = {}
    for definition in definitions:
        goal_kind = definition.goal_kind
        outcome = outcomes_by_kind.get(goal_kind)
        if (
            isinstance(outcome, ChainOutcome)
        ):
            disposition = resolve_goal_disposition(goal_kind, outcome)
            if (
                disposition.available
                and isinstance(disposition.bid, Mapping)
            ):
                bids_by_kind[goal_kind] = _materialize_goal_bid(
                    definition,
                    state,
                    role_bindings,
                    disposition.bid,
                )
            else:
                unavailable_kinds[goal_kind] = (
                    disposition.error_code
                    or APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE
                )
        else:
            unavailable_kinds[goal_kind] = APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE
    return bids_by_kind, unavailable_kinds


def _synthesize_branch_execution(
    definitions: Sequence[BranchDefinition],
    bids_by_kind: Mapping[str, ActionBidV2],
    unavailable_kinds: Mapping[str, str],
    ledger: AttemptLedger,
) -> ParallelExecutionResult:
    """Synthesize one wave's branch execution from complete-bid join results.

    Available definitions contribute materialized bids keyed by V2 branch id and
    unavailable ones contribute typed failure records, so the protected
    required-branch escalation and observability helpers run verbatim over the
    synthesized result. Every started definition keeps a started_at entry even
    when its chain failed; synthesized waves record no per-branch timing, so
    overlap and dependency wait stay at zero rather than inventing measurements.
    Synthesized failures carry no exception identity: the engine's protected
    consumers read only the typed error code from these records.
    """

    branch_id_by_kind = {
        definition.goal_kind: definition.branch_id for definition in definitions
    }
    results: dict[str, Any] = {}
    started_at: dict[str, float] = {
        definition.branch_id: 0.0 for definition in definitions
    }
    failed_ids: set[str] = set()
    failure_records: dict[str, BranchFailure] = {}
    for goal_kind, bid in bids_by_kind.items():
        results[branch_id_by_kind[goal_kind]] = bid
    for goal_kind, error_code in unavailable_kinds.items():
        branch_id = branch_id_by_kind[goal_kind]
        failed_ids.add(branch_id)
        failure_records[branch_id] = BranchFailure(
            branch_id=branch_id,
            error_code=error_code,
            stage="goal_cognition",
            attempt_count=ledger.attempts_used(goal_kind),
            safe_checkpoint="final_reduction",
            retryable=False,
            exception_class=None,
        )
    return ParallelExecutionResult(
        results=results,
        warnings=[],
        started_at=started_at,
        ended_at={},
        maximum_concurrency=len(definitions),
        failed_branch_ids=failed_ids,
        failure_records=failure_records,
    )


_TERMINAL_PROPOSITION_KINDS = {
    "goal_completed": "goal",
    "event_completed": "event",
    "event_repaired": "event",
    "threat_resolved": "threat",
    "knowledge_answered": "knowledge_gap",
}


def _bridge_appraisal_rows(
    chains: Sequence[ChainSpec],
    outcomes_by_name: Mapping[str, ChainOutcome],
    question_by_kind: Mapping[str, Mapping[str, Any]],
    evidence_positions: Mapping[str, int],
    handle_to_ref: Mapping[str, Mapping[str, str]],
    preliminary_state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str], list[str]]:
    """Bridge accepted v3 stage content into native V2 appraisal rows.

    Every accepted terminal-outcome stage with bounded content contributes one
    row for its deterministic ``q:<stage>`` identity so fresh canonical
    terminal assertions reach the final reduction; a terminal stage skipped
    because no outcome question was planned stays contentless and contributes
    none, mirroring wave-A stages without a planned question.
    Propositions attach to their source evidence row's candidate event root
    with code-owned provenance so the native materializer resolves them through
    causal-event checks, except terminal outcome propositions whose subject
    binds to the unique lifecycle-eligible entity of the asserted kind; a
    terminal assertion without exactly one eligible entity is dropped as an
    unbound warning instead of guessing its target. Axis deltas translate into
    exact native increment rows only when their axis matches exactly one
    permitted concrete path for the stage's authorized evidence domain, their
    value is an integer, and selected evidence survives; every other delta is
    dropped with a deterministic per-item warning so nothing unresolvable
    reaches the native reducer. Role assignments stay empty: the producer
    contract carries no role data, recorded as a documented parity gap in the
    package README.
    """

    handle_by_entity = {
        ref["entity_id"]: handle for handle, ref in handle_to_ref.items()
    }
    terminal_handles_by_kind = {}
    for entity_kind in ENTITY_LIST_FIELDS:
        eligible = _goal_outcome_eligible_entities(
            preliminary_state, entity_kind
        )
        terminal_handles_by_kind[entity_kind] = [
            handle_by_entity[entity["entity_id"]]
            for entity in eligible
            if entity["entity_id"] in handle_by_entity
        ]
    handle_map = _index_projected_handles(handle_to_ref)

    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    warnings: list[str] = []
    for chain in chains:
        outcome = outcomes_by_name.get(chain.name)
        last_results: dict[str, StageResult] = {}
        if isinstance(outcome, ChainOutcome):
            for result in reversed(outcome.results):
                last_results.setdefault(result.stage_name, result)
        for stage_name in chain.stages:
            question = question_by_kind.get(stage_name)
            if question is None:
                if chain.name != TERMINAL_OUTCOME_CHAIN.name:
                    continue
                question = {
                    "question_id": f"q:{stage_name}",
                    "permitted_delta_paths": _permitted_delta_paths(
                        stage_name,
                        handle_map,
                        preliminary_state,
                        list(evidence_positions),
                    ),
                }
            result = last_results.get(stage_name)
            if (
                isinstance(result, StageResult)
                and result.accepted
                and result.semantic_summary is not None
                and isinstance(result.local_state, Mapping)
            ):
                local_state = result.local_state
                propositions: list[dict[str, Any]] = []
                for item in local_state["propositions"]:
                    origin = item["origin_evidence_handle"]
                    entity_kind = _TERMINAL_PROPOSITION_KINDS.get(item["kind"])
                    if entity_kind is None:
                        position = evidence_positions.get(origin)
                        if position is None:
                            raise ValueError(
                                "appraisal proposition origin has no authorized"
                                f" evidence row: {origin!r}"
                            )
                        subject_handle = f"ce{position}"
                    else:
                        candidates = terminal_handles_by_kind[entity_kind]
                        if len(candidates) != 1:
                            warnings.append(
                                f"v3_terminal_unbound:{stage_name}:"
                                f"{item['kind']}"
                            )
                            continue
                        subject_handle = candidates[0]
                    propositions.append({
                        "proposition_kind": item["kind"],
                        "subject_handle": subject_handle,
                        "evidence_handles": [origin],
                        "role_assignments": [],
                        "semantic_value": item["statement"],
                    })
                explanation = result.semantic_summary
                if not isinstance(explanation, str) or not explanation:
                    raise ValueError(
                        f"accepted appraisal stage has no bounded summary:"
                        f" {stage_name}"
                    )
                selected = list(local_state["selected_evidence_handles"])
                native_deltas: list[dict[str, Any]] = []
                for item in local_state["deltas"]:
                    axis = item["path"]
                    matches = [
                        path
                        for path in question["permitted_delta_paths"]
                        if path.rsplit(".", 1)[-1] == axis
                    ]
                    value = item["value"]
                    if isinstance(value, bool) or not isinstance(value, int):
                        warnings.append(
                            f"v3_delta_non_integer:{stage_name}:{axis}"
                        )
                        continue
                    if not selected:
                        warnings.append(
                            f"v3_delta_no_evidence:{stage_name}:{axis}"
                        )
                        continue
                    if len(matches) != 1:
                        warning_code = (
                            "v3_delta_ambiguous" if matches else "v3_delta_unbound"
                        )
                        warnings.append(
                            f"{warning_code}:{stage_name}:{axis}"
                        )
                        continue
                    native_deltas.append({
                        "target_path": matches[0],
                        "delta": value,
                        "evidence_handles": selected,
                        "reason": item["reason"],
                    })
                rows.append({
                    "question_id": question["question_id"],
                    "selected_evidence_handles": selected,
                    "selected_role_handles": [],
                    "propositions": propositions,
                    "deltas": native_deltas,
                    "explanation": explanation,
                })
            elif not (isinstance(result, StageResult) and result.accepted):
                failures[question["question_id"]] = (
                    result.failure.error_code
                    if isinstance(result, StageResult)
                    and not result.accepted
                    and result.failure is not None
                    else APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE
                )
    return rows, failures, warnings


async def run_cognition(
    input_payload: CognitionCoreInputV2,
    services: CognitionCoreServicesV2,
) -> CognitionCoreOutputV2:
    """Run the V3 cognition engine with failure-only protected replay capture.

    The shell mirrors the V2 facade: one shared attempt ledger bound for the
    whole run, a failure capsule that records only terminal or partial-failure
    evidence, and deterministic success-path capture owned by the native
    validation event pipeline. Protected chain trace records are per-trace
    scopes: each call binds its own scope when none is active so concurrent
    calls keep isolated record sets.
    """

    ledger_token = None
    if current_v2_attempt_ledger() is None:
        ledger_token = bind_v2_attempt_ledger(
            create_v2_attempt_ledger(),
            graph_attempt=1,
        )
    records_token = None
    if _CURRENT_PROTECTED_CHAIN_SCOPE.get() is None:
        records_token = bind_protected_chain_records()
    try:
        session = failure_capsule.begin_failure_capsule(
            trace_id=llm_tracing.current_trace_id(),
            entrypoint="cognition_core_v3.run_cognition",
            input_payload=input_payload,
        )
        try:
            output = await _run_cognition(input_payload, services)
        except Exception as exc:
            failure_capsule.mark_failure(
                session,
                failure_kind="terminal_failure",
                stage_name="cognition_core_v3.run_cognition",
                details={},
                exception=exc,
            )
            failure_capsule.finish_failure_capsule(
                session,
                outcome="terminal_failure",
                exception=exc,
                attempt_ledger=snapshot_v2_attempt_ledger(),
            )
            raise

        _mark_cognition_partial_failures(session, output)
        failure_capsule.finish_failure_capsule(
            session,
            outcome=None,
            attempt_ledger=snapshot_v2_attempt_ledger(),
        )
        return output
    finally:
        if ledger_token is not None:
            reset_v2_attempt_ledger(ledger_token)
        if records_token is not None:
            reset_protected_chain_records(records_token)


async def _run_cognition(
    input_payload: CognitionCoreInputV2,
    services: CognitionCoreServicesV2,
) -> CognitionCoreOutputV2:
    """Run the bounded V3 cognition pipeline over the unchanged V2 substrate.

    The deterministic head runs verbatim on the preliminary state; wave A
    carries the five registry appraisal chains and every dependency-ready goal
    chain as isolated fresh-boundary chains under one global attempt ledger;
    accepted stage content bridges into native appraisal rows that reduce over
    the PRELIMINARY state before relationship maintenance, wave B runs newly
    ready goal chains, and collapse, action planning, and output assembly run
    verbatim with protected replay capture.
    """

    started_at = time.perf_counter()
    payload = validate_cognition_core_input(input_payload)
    previous_state = validate_cognition_state(payload["mutable_state"])
    updated_at = _episode_updated_at(payload["episode"])
    elapsed_seconds = _cognition_elapsed_seconds(previous_state, updated_at)
    warnings: list[str] = []
    stage_status: dict[str, str] = {
        "input_validation": "completed",
        "deterministic_preliminary": "skipped",
        "semantic_appraisal": "skipped",
        "final_reduction": "skipped",
        "branch_cognition": "skipped",
        "workspace_collapse": "skipped",
        "action_planning": "skipped",
    }

    fact_pairs = [
        (fact["producer"], _fact_without_producer(fact))
        for fact in payload["direct_facts"]
    ]
    reducer_relationship_context = _native_relationship_context(
        payload.get("relationship_context"),
    )
    preliminary_state = apply_state_update(
        previous_state,
        direct_facts=fact_pairs,
        elapsed_seconds=elapsed_seconds,
        updated_at=updated_at,
        character_constraints=payload["character_constraints"],
        relationship_context=reducer_relationship_context,
    )
    preliminary_state = create_deterministic_goals(
        preliminary_state,
        character_constraints=payload["character_constraints"],
        relationship_context=reducer_relationship_context,
        evidence=payload["evidence"],
        updated_at=updated_at,
    )
    preliminary_state = validate_cognition_state(preliminary_state)
    projection = project_state_for_prompt(
        preliminary_state,
        character_constraints=payload["character_constraints"],
        character_identity_context=payload["character_identity_context"],
        relationship_context=payload.get("relationship_context"),
        character_operational_context=payload.get(
            "character_operational_context",
        ),
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
    )
    questions = plan_semantic_questions(
        payload["evidence"],
        preliminary_state,
        projection.handle_to_ref,
    )
    question_by_kind = {question["question_kind"]: question for question in questions}
    stage_status["deterministic_preliminary"] = "completed"

    ledger_limits: dict[str, int] = {}
    for chain in APPRAISAL_FIRST_WAVE_CHAINS:
        for stage_name in chain.stages:
            ledger_limits[stage_name] = V2_APPRAISAL_TOTAL_ATTEMPTS
    for stage_name in TERMINAL_OUTCOME_CHAIN.stages:
        ledger_limits[stage_name] = V2_APPRAISAL_TOTAL_ATTEMPTS
    for chain in GOAL_CHAINS:
        for goal_kind in chain.stages:
            ledger_limits[goal_kind] = V2_MODEL_TOTAL_ATTEMPTS
    ledger = AttemptLedger(limits=ledger_limits)

    preliminary_branches = [
        replace(definition, dependencies=(), dependency_options=())
        for definition in select_preliminary_branches(preliminary_state["goals"])
    ]
    branch_context = _branch_context(
        projection,
        preliminary_state,
        payload["evidence"],
        scene_context=payload["scene_context"],
        private_continuity_context=payload["private_continuity_context"],
        past_dialog_cognition_context=payload[
            "past_dialog_cognition_context"
        ],
        group_engagement_action_context=payload[
            "group_engagement_action_context"
        ],
    )

    # The ordinary branch is skipped on evidence-free episodes, mirroring the
    # V2 handler; it keeps a started marker but produces no chain or bid.
    wave_a_goal_definitions = [
        definition
        for definition in preliminary_branches
        if not (
            definition.goal_kind == ORDINARY_GOAL_KIND
            and not payload["evidence"]
        )
    ]
    wave_a_specs: list[ChainTaskSpec] = [
        ChainTaskSpec(
            chain.name,
            chain.stages,
            _make_appraisal_stage_producers(
                services,
                projection,
                question_by_kind,
                chain,
            ),
        )
        for chain in APPRAISAL_FIRST_WAVE_CHAINS
    ]
    wave_a_specs.extend(
        ChainTaskSpec(
            definition.goal_kind,
            (definition.goal_kind,),
            {
                definition.goal_kind: _make_goal_stage_producer(
                    services,
                    definition,
                    preliminary_state,
                    branch_context,
                    payload,
                )
            },
        )
        for definition in wave_a_goal_definitions
    )

    wave_a = await start_wave(wave_a_specs, ledger=ledger).complete()

    (
        preliminary_bids,
        unavailable_kinds_a,
    ) = _join_goal_wave(
        wave_a_goal_definitions,
        {
            definition.goal_kind: wave_a.outcomes.get(definition.goal_kind)
            for definition in wave_a_goal_definitions
        },
        preliminary_state,
        branch_context["_role_bindings"],
    )
    for goal_kind in unavailable_kinds_a:
        if goal_kind not in wave_a.outcomes:
            warnings.append(
                f"v3_chain_unavailable:{goal_kind}:"
                f"{wave_a.failed_chains.get(goal_kind, 'cancelled')}"
            )

    preliminary_execution = _synthesize_branch_execution(
        preliminary_branches,
        preliminary_bids,
        unavailable_kinds_a,
        ledger,
    )
    warnings.extend(preliminary_execution.warnings)
    _raise_for_unrecoverable_required_branch_failures(
        preliminary_execution,
        preliminary_branches,
    )

    appraisal_chain_outcomes = [
        wave_a.outcomes[chain.name]
        for chain in APPRAISAL_FIRST_WAVE_CHAINS
        if chain.name in wave_a.outcomes
    ]
    provisional_state = reduce_appraisal_results(appraisal_chain_outcomes)
    evidence_handles = [row["evidence_handle"] for row in payload["evidence"]]
    terminal_producers = _make_terminal_stage_producer(
        services,
        provisional_state,
        evidence_handles,
        question_by_kind,
    )
    terminal_wave = await start_wave(
        [
            ChainTaskSpec(
                TERMINAL_OUTCOME_CHAIN.name,
                TERMINAL_OUTCOME_CHAIN.stages,
                terminal_producers,
            )
        ],
        ledger=ledger,
    ).complete()

    outcomes_by_name: dict[str, ChainOutcome] = {
        **wave_a.outcomes,
        **terminal_wave.outcomes,
    }
    bridged_chains: list[ChainSpec] = [
        *APPRAISAL_FIRST_WAVE_CHAINS,
        TERMINAL_OUTCOME_CHAIN,
    ]
    for chain in bridged_chains:
        if chain.name not in outcomes_by_name:
            exception_class = (
                wave_a.failed_chains.get(chain.name)
                or terminal_wave.failed_chains.get(chain.name)
                or "cancelled"
            )
            warnings.append(
                f"v3_chain_unavailable:{chain.name}:{exception_class}"
            )
    evidence_positions: dict[str, int] = {
        row["evidence_handle"]: index
        for index, row in enumerate(payload["evidence"], start=1)
    }
    (
        appraisal_results,
        appraisal_failures,
        bridge_warnings,
    ) = _bridge_appraisal_rows(
        bridged_chains,
        outcomes_by_name,
        question_by_kind,
        evidence_positions,
        projection.handle_to_ref,
        preliminary_state,
    )
    warnings.extend(bridge_warnings)
    stage_status["semantic_appraisal"] = "completed"

    (
        final_state,
        appraisal_results,
        reduction_failures,
        comparison_results,
        accepted_relationship_deltas,
    ) = _reduce_appraisals_with_isolation(
        preliminary_state,
        appraisal_results,
        payload["evidence"],
        projection.handle_to_ref,
        updated_at=updated_at,
        character_constraints=payload["character_constraints"],
        relationship_context=reducer_relationship_context,
    )
    appraisal_failures.update(reduction_failures)
    warnings.extend(
        f"semantic_appraisal_failed:{error_code}"
        for error_code in reduction_failures.values()
    )
    final_state = _apply_final_relationship_maintenance(
        final_state,
        episode=payload["episode"],
        elapsed_seconds=elapsed_seconds,
        accepted_relationship_deltas=accepted_relationship_deltas,
        direct_facts=payload["direct_facts"],
    )
    if final_state["state_scope"] == "user":
        final_state = apply_state_update(
            final_state,
            elapsed_seconds=0,
            updated_at=updated_at,
            character_constraints=payload["character_constraints"],
            relationship_context=reducer_relationship_context,
        )
        final_state = create_deterministic_goals(
            final_state,
            character_constraints=payload["character_constraints"],
            relationship_context=reducer_relationship_context,
            evidence=payload["evidence"],
            updated_at=updated_at,
            reconcile_salience_gated_goals=True,
        )
        final_state = validate_cognition_state(final_state)
    stage_status["final_reduction"] = "completed"
    final_projection = project_state_for_prompt(
        final_state,
        character_constraints=payload["character_constraints"],
        character_identity_context=payload["character_identity_context"],
        relationship_context=payload.get("relationship_context"),
        character_operational_context=payload.get(
            "character_operational_context",
        ),
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
    )

    successful_questions = {result["question_id"] for result in appraisal_results}
    final_branches = select_final_branches(
        preliminary_branches,
        final_state["goals"],
        successful_questions,
    )
    started_preliminary_ids = {
        definition.branch_id for definition in wave_a_goal_definitions
    }
    new_branch_definitions = [
        definition
        for definition in final_branches
        if definition.branch_id not in started_preliminary_ids
    ]

    final_execution: ParallelExecutionResult | None = None
    if new_branch_definitions:
        wave_b_context = _branch_context(
            final_projection,
            final_state,
            payload["evidence"],
            appraisal_results,
            scene_context=payload["scene_context"],
            private_continuity_context=payload[
                "private_continuity_context"
            ],
            past_dialog_cognition_context=payload[
                "past_dialog_cognition_context"
            ],
            group_engagement_action_context=payload[
                "group_engagement_action_context"
            ],
        )
        wave_b_specs = [
            ChainTaskSpec(
                definition.goal_kind,
                (definition.goal_kind,),
                {
                    definition.goal_kind: _make_goal_stage_producer(
                        services,
                        definition,
                        final_state,
                        wave_b_context,
                        payload,
                    )
                },
            )
            for definition in new_branch_definitions
        ]
        wave_b = await start_wave(wave_b_specs, ledger=ledger).complete()
        (
            final_bids,
            unavailable_kinds_b,
        ) = _join_goal_wave(
            new_branch_definitions,
            {
                definition.goal_kind: wave_b.outcomes.get(definition.goal_kind)
                for definition in new_branch_definitions
            },
            final_state,
            wave_b_context["_role_bindings"],
        )
        for goal_kind in unavailable_kinds_b:
            if goal_kind not in wave_b.outcomes:
                warnings.append(
                    f"v3_chain_unavailable:{goal_kind}:"
                    f"{wave_b.failed_chains.get(goal_kind, 'cancelled')}"
                )
        final_execution = _synthesize_branch_execution(
            new_branch_definitions,
            final_bids,
            unavailable_kinds_b,
            ledger,
        )
        warnings.extend(final_execution.warnings)
    stage_status["branch_cognition"] = "completed"
    if final_execution is not None:
        _raise_for_unrecoverable_required_branch_failures(
            final_execution,
            new_branch_definitions,
        )

    bids: list[ActionBidV2] = list(preliminary_execution.results.values())
    if final_execution is not None:
        bids.extend(final_execution.results.values())
    bids = [bid for bid in bids if isinstance(bid, Mapping)]
    generated_bids = list(bids)
    eligible_bids, stale_branch_ids = _bids_with_live_goals(
        bids,
        final_state,
    )
    warnings.extend(
        f"stale_goal_bid_dropped:{branch_id}"
        for branch_id in stale_branch_ids
    )
    relational_decision = _ordinary_relational_decision(eligible_bids)
    if (
        relational_decision is not None
        and relational_decision["applicability"] == "relationship_sensitive"
    ):
        collapse = collapse_authoritative_relational_bid(
            eligible_bids,
            relational_decision,
        )
        warnings.append("authoritative_relational_willingness")
    else:
        try:
            collapse = await collapse_bids_via_partition(
                eligible_bids,
                services,
                current_event=_workspace_current_event(payload["evidence"]),
                goal_context_by_ref=_workspace_goal_contexts(
                    eligible_bids,
                    final_state,
                ),
            ) if eligible_bids else _empty_collapse()
        except Exception as exc:
            raise CognitionExecutionError(
                f"workspace collapse failed: {exc}"
            ) from exc
    primary_bid = collapse.get("primary_bid")
    supporting_bids = collapse.get("supporting_bids", [])
    episode_operation = project_dialog_response_operation(payload["episode"])
    if (
        primary_bid is None
        and episode_operation is not None
        and episode_operation["selection_required"]
    ):
        raise CognitionExecutionError(
            "required selection has no admitted bid",
            error_code="required_selection_without_admitted_bid",
            stage="workspace_collapse",
            safe_checkpoint="final_reduction",
        )
    try:
        action_plan = await plan_actions(
            primary_bid=primary_bid,
            supporting_bids=supporting_bids,
            episode=payload["episode"],
            evidence=payload["evidence"],
            available_actions=payload["available_actions"],
            available_resolvers=payload["available_resolver_capabilities"],
            resolver_context=payload["resolver_context"],
            group_engagement_action_context=payload[
                "group_engagement_action_context"
            ],
            scene_context=payload["scene_context"],
            runtime_capability_limits=payload.get(
                "runtime_capability_limits",
                [],
            ),
            services=services,
            current_goal_progress=payload.get("resolver_goal_progress"),
            required_resolver_evidence_dependency=payload.get(
                "required_resolver_evidence_dependency"
            ),
        )
    except CognitionExecutionError:
        raise
    except Exception as exc:
        raise CognitionExecutionError(f"action planning failed: {exc}") from exc
    intention = action_plan["intention"]
    action_requests = action_plan["action_requests"]
    resolver_requests = action_plan["resolver_requests"]
    pending_resolution = _bind_pending_resolution(
        action_plan["resolver_pending_resolution"],
        payload.get("pending_resolver_resume"),
    )
    stage_status["workspace_collapse"] = "completed"
    stage_status["action_planning"] = "completed"

    admitted_bid = _selected_bid(intention, primary_bid, supporting_bids)
    affect = project_affect(final_state["affect_activations"], final_state)
    relationship = project_relationship(final_state.get("relationship"))
    expression_policy = default_expression_policy(
        intention["route"],
        affect,
        selected_branch_id=intention.get("selected_branch_id"),
        activations=final_state["affect_activations"],
    )
    selected_bid_reason = (
        admitted_bid["reason"]
        if admitted_bid is not None
        else "没有有依据的目标候选"
    )
    diagnostics = {
        "run_id": str(payload["episode"].get("episode_id", "episode")),
        "stage_status": stage_status,
        "selected_question_count": len(questions),
        "dispatched_question_count": len(questions),
        "selected_branch_count": (
            len(preliminary_branches) + len(new_branch_definitions)
        ),
        "dispatched_branch_count": (
            len(preliminary_execution.started_at)
            + (len(final_execution.started_at) if final_execution else 0)
        ),
        "completed_branch_count": (
            len(preliminary_execution.results)
            + (len(final_execution.results) if final_execution else 0)
        ),
        "failed_branch_count": (
            len(preliminary_execution.failed_branch_ids)
            + (len(final_execution.failed_branch_ids) if final_execution else 0)
        ),
        "overlap_ms": max(
            preliminary_execution.overlap_ms,
            final_execution.overlap_ms if final_execution else 0,
        ),
        "dependency_wait_ms": max(
            preliminary_execution.dependency_wait_ms,
            final_execution.dependency_wait_ms if final_execution else 0,
        ),
        "total_ms": _elapsed_ms(started_at),
        "warnings": _deduplicate_diagnostics_warnings(warnings),
    }
    cognition_observability = _build_cognition_observability(
        questions=questions,
        appraisal_results=appraisal_results,
        appraisal_failures=appraisal_failures,
        preliminary_branches=preliminary_branches,
        preliminary_execution=preliminary_execution,
        final_branches=new_branch_definitions,
        final_execution=final_execution,
        collapse=collapse,
        selected_bid_reason=selected_bid_reason,
        diagnostics=diagnostics,
        relational_willingness=relational_decision,
    )
    output: dict[str, Any] = {
        "schema_version": "cognition_core_output.v2",
        "intention": intention,
        "goal_continuation_ref": intention["goal_continuation_ref"],
        "supporting_bids": [
            bid for bid in supporting_bids
            if admitted_bid is None or bid["branch_id"] != admitted_bid["branch_id"]
        ],
        "state_update": build_state_update(
            previous_state,
            final_state,
            comparison_results,
        ),
        "affect_projection": affect,
        "action_requests": action_requests,
        "resolver_requests": resolver_requests,
        "goal_resolution": action_plan["goal_resolution"],
        "resolver_pending_resolution": pending_resolution,
        "resolver_goal_progress": action_plan["resolver_goal_progress"],
        "resolver_progress": _resolver_progress(resolver_requests),
        "selected_bid_reason": selected_bid_reason,
        "private_monologue": (
            admitted_bid["private_monologue"]
            if admitted_bid is not None
            else "当前角色没有有依据的行动理由。"
        ),
        "expression_policy": expression_policy,
        "diagnostics": diagnostics,
        "cognition_observability": cognition_observability,
    }
    if admitted_bid is not None:
        output["admitted_bid"] = admitted_bid
    if relationship is not None:
        output["relationship_projection"] = relationship
    if relational_decision is not None:
        output["relational_willingness"] = dict(relational_decision)
    if is_targetless_group_self_cognition_episode(payload["episode"]):
        response_status = action_plan.get(
            "self_cognition_response_contract_status"
        )
        if isinstance(response_status, str):
            output["self_cognition_response_contract_status"] = (
                response_status
            )
        response = action_plan.get("self_cognition_response")
        if isinstance(response, Mapping):
            output["self_cognition_response"] = dict(response)
    all_branch_definitions = [
        *preliminary_branches,
        *new_branch_definitions,
    ]
    capture_validation_event(
        "dependency_graph",
        {
            "branch_definitions": [
                {
                    "branch_id": definition.branch_id,
                    "dependencies": list(definition.dependencies),
                    "action_tendencies": list(definition.action_tendencies),
                    "required": definition.required,
                    "goal_kind": definition.goal_kind,
                }
                for definition in all_branch_definitions
            ],
        },
    )
    capture_validation_event(
        "branch_execution",
        {
            "maximum_concurrency": max(
                preliminary_execution.maximum_concurrency,
                final_execution.maximum_concurrency if final_execution else 0,
            ),
            "generated_bids": generated_bids,
            "eligible_bids": eligible_bids,
            "failed_branch_ids": sorted({
                *preliminary_execution.failed_branch_ids,
                *(
                    final_execution.failed_branch_ids
                    if final_execution is not None
                    else set()
                ),
            }),
        },
    )
    capture_validation_event(
        "emotion_derivation",
        {
            "state_before_derivation": preliminary_state,
            "state_after_derivation": final_state,
            "affect_projection": affect,
        },
    )
    capture_validation_event(
        "workspace_selection",
        {
            "appraisal_results": appraisal_results,
            "comparison_results": comparison_results,
            "final_intention": intention,
        },
    )
    return validate_cognition_core_output(output)
