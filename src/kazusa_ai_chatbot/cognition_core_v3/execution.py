"""Bounded serial primary-chain execution helpers for Cognition V3."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import partial

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
    V2AttemptBudgetExhausted,
    current_v2_attempt_ledger,
    record_v2_attempt_disposition,
    reserve_v2_model_attempt,
)
from kazusa_ai_chatbot.cognition_core_v3 import prompt as v3_prompt
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
    EXHAUSTION_FAILURE_CLASS,
    STRUCTURAL_FAILURE_CLASS,
    StageFailure,
    StageResult,
)
from kazusa_ai_chatbot.cognition_core_v3.lane import (
    LaneDeadlineError,
    SidecarAdmissionError,
    SidecarAdmissionLedger,
    SidecarCoordinator,
    SidecarInvocationState,
)
from kazusa_ai_chatbot.cognition_core_v3.registry import SERIAL_CHAIN_STEPS
from kazusa_ai_chatbot.cognition_core_v3.transcript import ChainTranscriptV1
from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMInvoker
from kazusa_ai_chatbot.utils import parse_llm_json_output


class ExecutorContractError(RuntimeError):
    """Fail-closed executor topology or attempt-cap violation."""


class TurnDeadlineExceeded(TimeoutError):
    """The current invocation cannot start another model request."""


def check_turn_deadline(deadline_monotonic: float | None) -> float | None:
    """Return remaining seconds, or raise before a new model request."""

    if deadline_monotonic is None:
        return None
    remaining_seconds = deadline_monotonic - time.monotonic()
    if remaining_seconds <= 0:
        raise TurnDeadlineExceeded("cognition turn deadline expired")
    return remaining_seconds


def config_for_turn_deadline(
    config: LLMCallConfig,
    deadline_monotonic: float | None,
) -> LLMCallConfig:
    """Bound one model configuration to the invocation's remaining time."""

    remaining_seconds = check_turn_deadline(deadline_monotonic)
    if remaining_seconds is None:
        return config
    configured_timeout = config.timeout_seconds
    timeout_seconds = (
        remaining_seconds
        if configured_timeout is None
        else min(float(configured_timeout), remaining_seconds)
    )
    if timeout_seconds <= 0:
        raise TurnDeadlineExceeded("cognition turn deadline expired")
    return replace(config, timeout_seconds=timeout_seconds)


class _DeadlineBoundSyncInvoker:
    """Guard a synchronous repair worker at its provider boundary."""

    def __init__(
        self,
        llm: LLMInvoker,
        deadline_monotonic: float | None,
    ) -> None:
        self._llm = llm
        self._deadline_monotonic = deadline_monotonic

    def invoke(self, messages, *, config):
        check_turn_deadline(self._deadline_monotonic)
        return self._llm.invoke(messages, config=config)


JsonRepairCallback = Callable[
    [str, str, Mapping[str, object]],
    Awaitable[dict[str, object] | None],
]


def _accepted_product(
    question: v3_prompt.ChainQuestion,
    product: object,
) -> dict[str, object]:
    """Project one exact validated product beside its accepted answer."""

    return {
        "question": question.contract_name,
        "typed_product": deepcopy(product),
    }


@dataclass(frozen=True)
class SerialChainStep:
    """One immutable serial chain step and its bounded producer."""

    step_id: str
    stage_kind: str
    producer: StageProducer


@dataclass(frozen=True)
class SerialChainResult:
    """Materialized result for one cleanly completed serial chain."""

    step_results: tuple[StageResult, ...]

    @property
    def accepted_steps(self) -> tuple[StageResult, ...]:
        """Return only accepted stage results in serial order."""

        return tuple(result for result in self.step_results if result.accepted)


@dataclass
class SerialChainHarness:
    """Owned append-only transcript, attempt ledger, and context budget."""

    transcript: ChainTranscriptV1
    ledger: AttemptLedger
    budget: object

    def append_question(self, content: str) -> None:
        """Append the next primary question to the owned transcript."""

        self.transcript = self.transcript.append_question(content)

    def accept_answer(
        self,
        content: str,
        product: Mapping[str, object] | None = None,
    ) -> None:
        """Append one accepted assistant answer to the owned transcript."""

        self.transcript = self.transcript.accept_answer(content, product)

    def rollback_tail(self) -> str:
        """Roll back the current assistant tail and return its question."""

        self.transcript, question = self.transcript.rollback_tail_answer()
        return question

    def queue_interlude(self, interlude: Mapping[str, object]) -> None:
        """Queue one deterministic notice for the next question."""

        self.transcript = self.transcript.append_interlude_to_next_question(
            interlude
        )


@dataclass(frozen=True)
class StageAttemptOutcome:
    """Raw producer outcome for one bounded attempt before disposition."""

    accepted: bool
    local_state: dict[str, object] | None
    semantic_summary: str | None
    failure_class: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ExecutorContext:
    """Deterministic context handed to one stage producer attempt."""

    chain_name: str
    stage_name: str
    attempt_number: int
    accepted_prefix: tuple[StageResult, ...]


StageProducer = Callable[[ExecutorContext], Awaitable[StageAttemptOutcome]]


@dataclass
class AttemptLedger:
    """Global bounded attempt arithmetic for one cognition invocation."""

    limits: Mapping[str, int]
    _counts: dict[str, int] = field(default_factory=dict)

    def attempts_used(self, owner_name: str) -> int:
        """Return completed attempt reservations for one owner so far."""

        return self._counts.get(owner_name, 0)

    def can_reserve(self, owner_name: str) -> bool:
        """Check whether the next reservation stays inside the owner's cap."""

        if owner_name not in self.limits:
            raise ExecutorContractError(f"Unknown attempt-cap owner {owner_name!r}")
        limit = self.limits[owner_name]
        if limit <= 0:
            raise ExecutorContractError(
                f"Owner {owner_name!r} has a non-positive attempt limit"
            )
        return self.attempts_used(owner_name) < limit

    def reserve(self, owner_name: str) -> int:
        """Reserve the next attempt number for one owner."""

        if not self.can_reserve(owner_name):
            raise ExecutorContractError(
                f"Owner {owner_name!r} attempt cap already exhausted"
            )
        next_number = self.attempts_used(owner_name) + 1
        self._counts[owner_name] = next_number
        return next_number

    def record_attempt(self, owner_name: str) -> int:
        """Project one already-authorized model attempt into chain metadata."""

        if not isinstance(owner_name, str) or not owner_name.strip():
            raise ExecutorContractError("attempt projection owner is required")
        next_number = self.attempts_used(owner_name) + 1
        self._counts[owner_name] = next_number
        return next_number


async def run_serial_harness_step(
    *,
    harness: SerialChainHarness,
    step_id: str,
    stage_kind: str,
    producer: StageProducer,
) -> StageResult:
    """Run one harness-owned serial step with shared attempt arithmetic."""

    if step_id not in SERIAL_CHAIN_STEPS:
        raise ExecutorContractError(f"Unknown serial chain step {step_id!r}")
    attempt_number = harness.ledger.reserve(stage_kind)
    context = ExecutorContext(
        chain_name="serial",
        stage_name=stage_kind,
        attempt_number=attempt_number,
        accepted_prefix=(),
    )
    raw = await producer(context)
    if raw.accepted:
        if raw.local_state is None or raw.semantic_summary is None:
            raise ExecutorContractError(
                "Accepted serial steps require typed local state and summary"
            )
        result = StageResult(
            chain_name="serial",
            stage_name=stage_kind,
            accepted=True,
            local_state=raw.local_state,
            semantic_summary=raw.semantic_summary,
        )
        harness.accept_answer(
            raw.semantic_summary,
            {"step_id": step_id, **raw.local_state},
        )
        return result
    failure_class = raw.failure_class or STRUCTURAL_FAILURE_CLASS
    return StageResult(
        chain_name="serial",
        stage_name=stage_kind,
        accepted=False,
        local_state=None,
        semantic_summary=None,
        failure=StageFailure(
            chain_name="serial",
            stage_name=stage_kind,
            failure_class=failure_class,
            error_code=raw.detail or APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
            repair_attempted=attempt_number > 1,
        ),
    )


async def invoke_serial_model_step(
    *,
    harness: SerialChainHarness,
    system_content: str,
    llm: LLMInvoker,
    config: LLMCallConfig,
    deterministic_only: bool = False,
    deadline_monotonic: float | None = None,
) -> dict[str, object]:
    """Invoke one serial model step from the append-only transcript."""

    if not system_content:
        raise ExecutorContractError(
            "Serial model steps require a non-empty system head"
        )
    messages: list[BaseMessage] = [SystemMessage(content=system_content)]
    for role, content in harness.transcript.to_messages():
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            raise ExecutorContractError(f"Unknown transcript role {role!r}")
    effective_deadline = (
        deadline_monotonic
        if deadline_monotonic is not None
        else harness.transcript.deadline_monotonic
    )
    effective_config = config_for_turn_deadline(config, effective_deadline)
    response = await llm.ainvoke(messages, config=effective_config)
    raw_output = getattr(response, "content", "")
    return parse_llm_json_output(
        raw_output,
        deterministic_only=deterministic_only,
    )


async def invoke_lane_scoped_json_repair(
    *,
    raw_output: str,
    candidate_id: str,
    attempt_coordinates: Mapping[str, object],
    llm: LLMInvoker,
    config: LLMCallConfig,
    coordinator: SidecarCoordinator,
    admissions: SidecarAdmissionLedger,
    invocation_state: SidecarInvocationState,
    deadline_monotonic: float | None = None,
) -> dict[str, object] | None:
    """Repair one malformed primary candidate through the owned sidecar lane.

    The canonical parser still performs its deterministic pass before this
    helper is called.  This helper admits one injected repair model call while
    the producing V2 attempt remains live, preempts unfinished L1 work, and
    drains a synchronous repair worker before releasing the sidecar claim on
    cancellation.

    Args:
        raw_output: The malformed assistant candidate retained by the caller.
        candidate_id: Stable non-content identity for this producer attempt.
        attempt_coordinates: The producing V2 model-attempt reservation.
        llm: The injected V3 model invoker.
        config: The V3 sidecar route configuration for this repair call.
        coordinator: The process-local serialized sidecar owner.
        admissions: The invocation-local sidecar admission authority.
        invocation_state: The invocation-local task and diagnostics owner.

    Returns:
        The canonically repaired object, or ``None`` when repair admission is
        unavailable or the injected parser cannot recover an object.
    """

    try:
        check_turn_deadline(deadline_monotonic)
    except TurnDeadlineExceeded:
        return None

    try:
        admissions.reserve_json_repair(
            candidate_id=candidate_id,
            attempt_coordinates=attempt_coordinates,
        )
    except SidecarAdmissionError:
        return None

    try:
        await invocation_state.preempt_l1_for_repair()
        async with coordinator.claim(
            stream_kind="json_repair",
            invocation_state=invocation_state,
            deadline_monotonic=deadline_monotonic,
        ):
            effective_config = config_for_turn_deadline(
                config,
                deadline_monotonic,
            )
            repair_task = asyncio.create_task(
                asyncio.to_thread(
                    partial(
                        parse_llm_json_output,
                        raw_output,
                        deterministic_only=False,
                        repair_llm=_DeadlineBoundSyncInvoker(
                            llm,
                            deadline_monotonic,
                        ),
                        repair_config=effective_config,
                    )
                )
            )
            try:
                repaired_output = await asyncio.shield(repair_task)
            except asyncio.CancelledError:
                await asyncio.shield(repair_task)
                raise
    except (LaneDeadlineError, TurnDeadlineExceeded):
        return None
    if not isinstance(repaired_output, dict):
        return None
    repaired_object = dict(repaired_output)
    return repaired_object


async def invoke_serial_question_sequence(
    *,
    harness: SerialChainHarness,
    system_content: str,
    llm: LLMInvoker,
    config: LLMCallConfig,
    questions: Sequence[v3_prompt.ChainQuestion],
    deadline_monotonic: float | None = None,
) -> list[dict[str, object]]:
    """Invoke registered serial questions and return parsed raw outputs."""

    parsed_results: list[dict[str, object]] = []
    stage_kinds = {
        "semantic_appraisal_group.v1": "appraisal",
        "ordinary_goal_bid.v1": "goal_ordinary",
        "active_goal_bid_group.v1": "goal_active",
        "workspace_partition.v1": "workspace",
        "action_plan.v1": "action_planning",
    }
    for question in questions:
        stage_kind = stage_kinds.get(question.contract_name)
        if stage_kind is None:
            raise ExecutorContractError(
                f"Unknown serial question contract {question.contract_name!r}"
            )
        harness.ledger.reserve(stage_kind)
        question_text = v3_prompt.build_question_message(question)
        harness.append_question(question_text)
        parsed = await invoke_serial_model_step(
            harness=harness,
            system_content=system_content,
            llm=llm,
            config=config,
            deadline_monotonic=deadline_monotonic,
        )
        harness.accept_answer(
            json.dumps(parsed, ensure_ascii=False),
            _accepted_product(question, parsed),
        )
        parsed_results.append(parsed)
    return parsed_results


async def invoke_serial_question_with_repair(
    *,
    harness: SerialChainHarness,
    system_content: str,
    llm: LLMInvoker,
    config: LLMCallConfig,
    question: v3_prompt.ChainQuestion,
    validator: Callable[[dict[str, object]], object],
    attempt_limit: int,
    first_packet_sections: Sequence[Mapping[str, object]] | None = None,
    interludes: Sequence[Mapping[str, object]] = (),
    attempt_owner: str | None = None,
    v2_stage: str | None = None,
    v2_branch_id: str | None = None,
    deterministic_only: bool = False,
    json_repair_callback: JsonRepairCallback | None = None,
    deadline_monotonic: float | None = None,
) -> tuple[object | None, str | None]:
    """Invoke one serial question with bounded tail-safe repair attempts.

    ``first_packet_sections`` supplies the cold-turn carriers. They are
    rendered only while the accepted transcript is empty, so each rejected
    attempt retains the complete packet and every accepted later row keeps the
    compact question format.
    """

    if attempt_limit <= 0:
        raise ExecutorContractError("Serial repair attempt_limit must be positive")
    if not isinstance(question, v3_prompt.ChainQuestion):
        raise ExecutorContractError("Serial repair requires a registered question")
    if (v2_stage is None) != (v2_branch_id is None):
        raise ExecutorContractError(
            "V2 attempt stage and branch must be supplied together"
        )

    transcript_messages = harness.transcript.to_messages()
    base_messages: list[BaseMessage] = [SystemMessage(content=system_content)]
    for role, content in transcript_messages:
        if role == "human":
            base_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            base_messages.append(AIMessage(content=content))
        else:
            raise ExecutorContractError(f"Unknown transcript role {role!r}")

    if first_packet_sections is not None and not transcript_messages:
        (
            constraints_and_operational_state,
            relationship_and_mutable_state,
            episode_and_scene,
            evidence_and_affordances,
        ) = first_packet_sections
        question_text = v3_prompt.build_first_user_message(
            constraints_and_operational_state=(
                constraints_and_operational_state
            ),
            relationship_and_mutable_state=(
                relationship_and_mutable_state
            ),
            episode_and_scene=episode_and_scene,
            evidence_and_affordances=evidence_and_affordances,
            question=question,
        )
    else:
        question_text = v3_prompt.build_question_message(
            question,
            interludes=interludes,
        )
    effective_deadline = (
        deadline_monotonic
        if deadline_monotonic is not None
        else harness.transcript.deadline_monotonic
    )
    last_error: str | None = None
    raw_output: str | None = None

    for attempt_number in range(1, attempt_limit + 1):
        try:
            check_turn_deadline(effective_deadline)
        except TurnDeadlineExceeded:
            return None, raw_output
        coordinates: Mapping[str, object] | None = None
        if v2_stage is not None and current_v2_attempt_ledger() is not None:
            try:
                coordinates = reserve_v2_model_attempt(
                    stage=v2_stage,
                    branch_id=v2_branch_id,
                    local_attempt=attempt_number,
                )
            except V2AttemptBudgetExhausted:
                return None, raw_output
        if attempt_owner is not None:
            harness.ledger.record_attempt(attempt_owner)
        payload_text = question_text
        if attempt_number > 1 and last_error is not None:
            payload_text = f"{question_text}\n[contract_repair]\n{last_error}"
        attempt_config = replace(
            config,
            stage_name=f"{config.stage_name}.repair{attempt_number}",
        )
        messages = [*base_messages, HumanMessage(content=payload_text)]
        repaired_output = False
        try:
            attempt_config = config_for_turn_deadline(
                attempt_config,
                effective_deadline,
            )
            response = await llm.ainvoke(messages, config=attempt_config)
            raw_output = getattr(response, "content", "")
            parsed = parse_llm_json_output(
                raw_output,
                deterministic_only=(
                    deterministic_only
                    or json_repair_callback is not None
                ),
            )
            if (
                not parsed
                and json_repair_callback is not None
                and coordinates is not None
            ):
                check_turn_deadline(effective_deadline)
                repaired = await json_repair_callback(
                    str(raw_output),
                    f"{config.stage_name}:attempt:{attempt_number}",
                    coordinates,
                )
                if repaired is not None:
                    parsed = repaired
                    repaired_output = True
            if not isinstance(parsed, Mapping):
                raise TypeError("serial answer must be a JSON object")
            validated = validator(dict(parsed))
        except TurnDeadlineExceeded:
            if coordinates is not None:
                record_v2_attempt_disposition(
                    coordinates,
                    disposition="exhausted",
                )
            return None, raw_output
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            last_error = str(exc)
            if coordinates is not None:
                record_v2_attempt_disposition(
                    coordinates,
                    disposition=(
                        "exhausted"
                        if attempt_number == attempt_limit
                        else "regenerate"
                    ),
                )
            continue
        except (OpenAIError, httpx.HTTPError, ConnectionError, OSError) as exc:
            last_error = str(exc)
            if coordinates is not None:
                record_v2_attempt_disposition(
                    coordinates,
                    disposition=(
                        "exhausted"
                        if attempt_number == attempt_limit
                        else "regenerate"
                    ),
                )
            continue

        harness.append_question(payload_text)
        accepted_answer = (
            json.dumps(parsed, ensure_ascii=False, sort_keys=True)
            if repaired_output
            else raw_output or ""
        )
        harness.accept_answer(
            accepted_answer,
            _accepted_product(question, validated),
        )
        if coordinates is not None:
            record_v2_attempt_disposition(
                coordinates,
                disposition=(
                    "accepted"
                    if attempt_number == 1
                    else "recovered"
                ),
            )
        return validated, raw_output

    return None, raw_output


async def run_serial_chain(
    steps: Sequence[SerialChainStep],
    *,
    ledger: AttemptLedger,
) -> SerialChainResult:
    """Run one serial chain step-by-step under shared attempt arithmetic."""

    if not steps:
        raise ExecutorContractError("A serial chain requires at least one step")
    if len({step.step_id for step in steps}) != len(steps):
        raise ExecutorContractError("One serial chain cannot run the same step twice")
    registered_steps = set(SERIAL_CHAIN_STEPS)
    for step in steps:
        if step.step_id not in registered_steps:
            raise ExecutorContractError(f"Unknown serial chain step {step.step_id!r}")

    results: list[StageResult] = []
    for step in steps:
        attempts = ledger.attempts_used(step.stage_kind)
        try:
            attempt_number = ledger.reserve(step.stage_kind)
        except ExecutorContractError:
            failure = StageFailure(
                chain_name="serial",
                stage_name=step.stage_kind,
                failure_class=EXHAUSTION_FAILURE_CLASS,
                error_code=APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
                repair_attempted=attempts > 0,
            )
            results.append(
                StageResult(
                    chain_name="serial",
                    stage_name=step.stage_kind,
                    accepted=False,
                    local_state=None,
                    semantic_summary=None,
                    failure=failure,
                )
            )
            continue
        context = ExecutorContext(
            chain_name="serial",
            stage_name=step.stage_kind,
            attempt_number=attempt_number,
            accepted_prefix=tuple(result for result in results if result.accepted),
        )
        raw = await step.producer(context)
        if raw.accepted:
            if raw.local_state is None or raw.semantic_summary is None:
                raise ExecutorContractError(
                    "Accepted serial steps require typed local state and summary"
                )
            result = StageResult(
                chain_name="serial",
                stage_name=step.stage_kind,
                accepted=True,
                local_state=raw.local_state,
                semantic_summary=raw.semantic_summary,
            )
        else:
            failure_class = raw.failure_class or STRUCTURAL_FAILURE_CLASS
            result = StageResult(
                chain_name="serial",
                stage_name=step.stage_kind,
                accepted=False,
                local_state=None,
                semantic_summary=None,
                failure=StageFailure(
                    chain_name="serial",
                    stage_name=step.stage_kind,
                    failure_class=failure_class,
                    error_code=raw.detail or APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
                    repair_attempted=attempt_number > 1,
                ),
            )
        results.append(result)

    return SerialChainResult(tuple(results))
