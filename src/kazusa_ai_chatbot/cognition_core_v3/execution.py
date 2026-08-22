"""Bounded serial primary-chain execution helpers for Cognition V3."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
    V2AttemptBudgetExhausted,
    current_v2_attempt_ledger,
    record_v2_attempt_disposition,
    reserve_v2_model_attempt_batch,
)
from kazusa_ai_chatbot.cognition_core_v3 import prompt as v3_prompt
from kazusa_ai_chatbot.cognition_core_v3.budget import (
    BudgetAdmission,
    CognitionContextLimitError,
    ContextBudgetLedger,
    estimate_message_tokens,
)
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
    EXHAUSTION_FAILURE_CLASS,
    FAILURE_CLASSES,
    PROVIDER_FAILURE_CLASS,
    STRUCTURAL_FAILURE_CLASS,
    StageFailure,
    StageResult,
)
from kazusa_ai_chatbot.cognition_core_v3.diagnostics import (
    record_accepted_transcript,
    record_chain_step,
    record_chain_system_head,
    record_token_ledger,
)
from kazusa_ai_chatbot.cognition_core_v3.lane import (
    LaneDeadlineError,
    SidecarAdmissionError,
    SidecarAdmissionLedger,
    SidecarCoordinator,
    SidecarInvocationState,
)
from kazusa_ai_chatbot.cognition_core_v3.registry import (
    APPRAISAL_FAMILY_ORDER,
    SERIAL_CHAIN_STEPS,
)
from kazusa_ai_chatbot.cognition_core_v3.transcript import ChainTranscriptV1
from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMInvoker
from kazusa_ai_chatbot.utils import parse_llm_json_output


class ExecutorContractError(RuntimeError):
    """Fail-closed executor topology or attempt-cap violation."""


class TurnDeadlineExceeded(TimeoutError):
    """The current invocation cannot start another model request."""


QuestionDispositionKind = Literal[
    "accepted",
    "structural_exhausted",
    "provider_exhausted",
    "deadline_exhausted",
    "budget_exhausted",
]


@dataclass(frozen=True)
class SerialQuestionDisposition:
    """Typed terminal disposition for one repaired serial question."""

    kind: QuestionDispositionKind


@dataclass(frozen=True)
class SerialQuestionResult:
    """Immutable value returned after one bounded serial question."""

    validated: object | None
    raw_output: str | None
    disposition: SerialQuestionDisposition


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

_REPAIR_ERROR_CODES = frozenset({
    "contract_error",
    "empty_output",
    "provider_error",
})
_REPAIR_PERMITTED_HANDLE_CAP = 64
_REANCHOR_SEQUENCE_CAP = 32
_REANCHOR_MAPPING_CAP = 64
_REANCHOR_STRING_CAP = 256
_REANCHOR_SAFE_STRING_KEYS = frozenset({
    "action",
    "action_kind",
    "action_handle",
    "applicability",
    "assumptions_or_inferences",
    "attempted_paths",
    "authority",
    "authorized",
    "availability",
    "branch_id",
    "blockers",
    "capability",
    "confidence",
    "concrete_detail",
    "contract_name",
    "current_user_relationship_state",
    "current_focus",
    "decision",
    "decision_mode",
    "deliverables",
    "description",
    "desired_outcome",
    "direction",
    "entity_kind",
    "expected_consequences",
    "evidence_dependencies",
    "family",
    "final_response_requirements",
    "goal_kind",
    "goal_resolution",
    "handle",
    "intention",
    "kind",
    "missing_user_inputs",
    "note",
    "notice",
    "notice_kind",
    "operator",
    "question",
    "question_id",
    "question_kind",
    "resolver",
    "resolver_handle",
    "role",
    "selection",
    "selected_action",
    "selected_goal",
    "selected_resolver",
    "selected_response_operation",
    "schema_version",
    "semantic_goal",
    "source_kind",
    "source_backed_facts",
    "stale_branch_ids",
    "statement",
    "stance",
    "stage",
    "status",
    "temporal_alignment",
    "permission",
    "provenance_role",
})
_REANCHOR_DROP_KEYS = frozenset({
    "candidate",
    "content",
    "explanation",
    "message",
    "private_monologue",
    "prose",
    "reason",
    "rationale",
    "raw_output",
    "response_text",
    "semantic_summary",
    "semantic_text",
    "summary",
    "text",
})
_REANCHOR_MAPPING_HANDLE_KEYS = frozenset({
    "action_handles",
    "bid_handles",
    "resolver_handles",
    "role_bindings",
})
_REANCHOR_DROP = object()


def _question_permitted_handles(
    value: object,
    *,
    field_name: str = "",
) -> list[str]:
    """Collect bounded handle-domain values from one registered payload."""

    handles: list[str] = []
    if isinstance(value, Mapping):
        for nested_name, nested_value in value.items():
            if not isinstance(nested_name, str):
                continue
            if field_name == "role_bindings" or field_name.endswith(
                "_handles"
            ):
                handles.append(nested_name)
            handles.extend(
                _question_permitted_handles(
                    nested_value,
                    field_name=nested_name,
                )
            )
        return handles
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for nested_value in value:
            handles.extend(
                _question_permitted_handles(
                    nested_value,
                    field_name=field_name,
                )
            )
        return handles
    if (
        isinstance(value, str)
        and value
        and (
            field_name == "handle"
            or field_name.endswith(("_handle", "_handles"))
        )
    ):
        handles.append(value)
    return handles


def _permitted_question_handles(
    question: v3_prompt.ChainQuestion,
) -> list[str]:
    """Return one sorted, duplicate-free handle domain for repair feedback."""

    collected_handles = _question_permitted_handles(question.payload)
    permitted_handles = sorted(set(collected_handles))
    return permitted_handles[:_REPAIR_PERMITTED_HANDLE_CAP]


def _repair_failure_fact(
    exc: BaseException,
    *,
    attempt_index: int,
    raw_output: str | None,
) -> dict[str, object]:
    """Project one failure into a closed, prompt-safe repair fact."""

    failure_kind = getattr(exc, "failure_kind", None)
    empty_raw_output = raw_output is None or not raw_output.strip()
    if isinstance(failure_kind, str) and failure_kind in FAILURE_CLASSES:
        error_code = failure_kind
        error_class = failure_kind
    elif isinstance(exc, (OpenAIError, httpx.HTTPError, ConnectionError, OSError)):
        error_code = "provider_error"
        error_class = PROVIDER_FAILURE_CLASS
    elif empty_raw_output:
        error_code = "empty_output"
        error_class = STRUCTURAL_FAILURE_CLASS
    else:
        error_code = "contract_error"
        error_class = STRUCTURAL_FAILURE_CLASS

    if error_code not in _REPAIR_ERROR_CODES and error_code not in FAILURE_CLASSES:
        error_code = "contract_error"
        error_class = STRUCTURAL_FAILURE_CLASS

    field_path = getattr(exc, "field_path", None)
    if not isinstance(field_path, str) or not field_path:
        field_path = getattr(exc, "path", None)
    if not isinstance(field_path, str) or not field_path:
        field_path = "$"

    fact = {
        "attempt_index": attempt_index,
        "error_class": error_class,
        "error_code": error_code,
        "field_path": field_path,
    }
    return fact


def _repair_payload_text(
    question_text: str,
    *,
    question: v3_prompt.ChainQuestion,
    attempt_index: int,
    failure_facts: Sequence[Mapping[str, object]],
) -> str:
    """Append compact monotonic typed repair facts to one stage question."""

    appendix = {
        "attempt_index": attempt_index,
        "error_facts": [dict(fact) for fact in failure_facts],
        "expected_contract": question.contract_name,
        "permitted_handles": _permitted_question_handles(question),
    }
    appendix_text = json.dumps(
        appendix,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload_text = f"{question_text}\n[contract_repair]\n{appendix_text}"
    return payload_text


def _consume_remaining_repair_attempts(
    *,
    harness: SerialChainHarness,
    completed_attempt: int,
    attempt_limit: int,
    attempt_owner: str | None,
    v2_stage: str | None,
    v2_branch_ids: tuple[str, ...] | None,
    v2_local_attempt_start: int,
) -> None:
    """Record local exhaustion and any still-admissible V2 reservations."""

    v2_budget_exhausted = False
    for skipped_attempt in range(completed_attempt + 1, attempt_limit + 1):
        if attempt_owner is not None:
            harness.ledger.record_attempt(attempt_owner)
        if v2_stage is None or v2_budget_exhausted:
            continue
        try:
            coordinates = reserve_v2_model_attempt_batch(
                stage=v2_stage,
                branch_ids=v2_branch_ids,
                local_attempt=(
                    v2_local_attempt_start + skipped_attempt - 1
                ),
            )
        except V2AttemptBudgetExhausted:
            v2_budget_exhausted = True
            continue
        for coordinate in coordinates:
            record_v2_attempt_disposition(
                coordinate,
                disposition="exhausted",
            )


def _message_contents(messages: Sequence[BaseMessage]) -> list[str]:
    """Project the exact outgoing message contents for token estimation."""

    contents: list[str] = []
    for message in messages:
        content = message.content
        if not isinstance(content, str):
            content = str(content)
        contents.append(content)
    return contents


def _record_budget_admission(
    harness: SerialChainHarness,
    admission: BudgetAdmission,
) -> None:
    """Retain bounded admission facts in the invocation transcript carrier."""

    token_ledger = dict(harness.transcript.token_ledger or {})
    token_ledger.update({
        "declared_context_window_tokens": (
            harness.budget.plan.serving_window_tokens
        ),
        "normal_total_ceiling_tokens": (
            harness.budget.plan.normal_total_ceiling_tokens
        ),
        "extended_total_ceiling_tokens": (
            harness.budget.plan.extended_total_ceiling_tokens
        ),
        "max_estimated_prompt_tokens": max(
            int(token_ledger.get("max_estimated_prompt_tokens", 0)),
            admission.estimated_prompt_tokens,
        ),
        "max_reserved_completion_tokens": max(
            int(token_ledger.get("max_reserved_completion_tokens", 0)),
            admission.reserved_completion_tokens,
        ),
        "max_estimated_total_context_tokens": max(
            int(token_ledger.get("max_estimated_total_context_tokens", 0)),
            admission.estimated_total_context_tokens,
        ),
        "active_total_ceiling_tokens": admission.active_total_ceiling_tokens,
        "extension_available": int(admission.extension_available),
        "extension_used": int(admission.extension_used),
    })
    harness.transcript = replace(
        harness.transcript,
        token_ledger=token_ledger,
    )
    record_token_ledger(token_ledger)


def _admit_primary_request(
    *,
    harness: SerialChainHarness,
    messages: Sequence[BaseMessage],
    config: LLMCallConfig,
) -> BudgetAdmission:
    """Admit one exact primary request before it reaches the provider."""

    completion_tokens = config.max_completion_tokens
    if not isinstance(completion_tokens, int) or isinstance(
        completion_tokens,
        bool,
    ) or completion_tokens <= 0:
        raise ExecutorContractError(
            "primary request requires a positive completion-token cap"
        )
    if not isinstance(harness.budget, ContextBudgetLedger):
        raise ExecutorContractError(
            "serial harness requires a ContextBudgetLedger"
        )
    estimated_prompt_tokens = estimate_message_tokens(
        _message_contents(messages),
    )
    admission = harness.budget.admit(
        estimated_prompt_tokens=estimated_prompt_tokens,
        reserved_completion_tokens=completion_tokens,
    )
    _record_budget_admission(harness, admission)
    return admission


def _registered_step_id(stage_name: str) -> str:
    """Project a configured stage name onto its registered chain step."""

    base_name = stage_name
    repair_prefix, repair_marker, repair_number = stage_name.rpartition(
        ".repair",
    )
    if repair_marker and repair_number.isdigit() and repair_number:
        base_name = repair_prefix
    parts = base_name.split(".")
    if (
        len(parts) >= 2
        and parts[-1] in APPRAISAL_FAMILY_ORDER
        and parts[-2] in {"A1", "A2"}
    ):
        return parts[-2]
    return parts[-1]


def _record_primary_attempt_step(
    *,
    harness: SerialChainHarness,
    config: LLMCallConfig,
    messages: Sequence[BaseMessage],
    payload_text: str,
    attempt_number: int,
    started_at: float,
    admission: BudgetAdmission | None,
    status: str,
    parse_status: str,
    disposition: str,
    reanchored: bool,
    warning_codes: Sequence[str] = (),
) -> None:
    """Record exact bounded facts for one primary request or reservation."""

    prompt_contents = _message_contents(messages)
    prompt_chars = sum(len(content) for content in prompt_contents)
    shared_prefix_chars = sum(len(content) for content in prompt_contents[:-1])
    try:
        estimated_prompt_tokens = estimate_message_tokens(prompt_contents)
    except (TypeError, ValueError):
        estimated_prompt_tokens = 0
    try:
        estimated_new_suffix_tokens = estimate_message_tokens([payload_text])
    except (TypeError, ValueError):
        estimated_new_suffix_tokens = 0
    completion_tokens = config.max_completion_tokens
    reserved_completion_tokens = (
        completion_tokens
        if isinstance(completion_tokens, int)
        and not isinstance(completion_tokens, bool)
        else 0
    )
    if admission is not None:
        estimated_prompt_tokens = admission.estimated_prompt_tokens
        reserved_completion_tokens = admission.reserved_completion_tokens
        estimated_total_context_tokens = (
            admission.estimated_total_context_tokens
        )
        active_total_ceiling_tokens = admission.active_total_ceiling_tokens
        extension_available = admission.extension_available
        extension_used = admission.extension_used
    else:
        estimated_total_context_tokens = (
            estimated_prompt_tokens + reserved_completion_tokens
        )
        active_total_ceiling_tokens = harness.budget.active_total_ceiling_tokens
        extension_available = harness.budget.extension_available
        extension_used = harness.budget.extension_used
    cache_class = (
        "reanchor"
        if reanchored
        else "warm"
        if len(prompt_contents) > 2
        else "cold"
    )
    record_chain_step({
        "step_id": _registered_step_id(config.stage_name),
        "stage_kind": _registered_step_id(config.stage_name),
        "lane_kind": "primary",
        "sidecar_stream_kind": "",
        "status": status,
        "attempt_count": attempt_number,
        "duration_ms": max(0, int((time.perf_counter() - started_at) * 1000)),
        "queue_wait_ms": harness.primary_queue_wait_ms,
        "in_flight_at_start": harness.primary_in_flight_at_start,
        "prompt_chars": prompt_chars,
        "new_suffix_chars": len(payload_text),
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "reserved_completion_tokens": reserved_completion_tokens,
        "estimated_total_context_tokens": estimated_total_context_tokens,
        "active_total_ceiling_tokens": active_total_ceiling_tokens,
        "extension_available": extension_available,
        "extension_used": extension_used,
        "estimated_new_suffix_tokens": estimated_new_suffix_tokens,
        "declared_shared_prefix_chars": shared_prefix_chars,
        "cache_class": cache_class,
        "parse_status": parse_status,
        "repair_count": max(0, attempt_number - 1),
        "disposition": disposition,
        "warning_codes": list(warning_codes),
    })


def _reanchor_projection(
    value: object,
    *,
    field_name: str = "",
    depth: int = 0,
) -> object:
    """Keep only bounded deterministic typed facts for a re-anchor."""

    if depth > 6:
        return _REANCHOR_DROP
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _REANCHOR_DROP
    if isinstance(value, str):
        if (
            field_name in _REANCHOR_DROP_KEYS
            or field_name not in _REANCHOR_SAFE_STRING_KEYS
            and not field_name.endswith(("_handle", "_handles"))
        ):
            return _REANCHOR_DROP
        return value[:_REANCHOR_STRING_CAP]
    if isinstance(value, Mapping):
        projected: dict[str, object] = {}
        keys = sorted(value, key=lambda item: str(item))
        for key in keys[:_REANCHOR_MAPPING_CAP]:
            if not isinstance(key, str) or key in _REANCHOR_DROP_KEYS:
                continue
            nested = _reanchor_projection(
                value[key],
                field_name=key,
                depth=depth + 1,
            )
            if (
                nested is _REANCHOR_DROP
                and field_name in _REANCHOR_MAPPING_HANDLE_KEYS
            ):
                nested = _reanchor_projection(
                    value[key],
                    field_name="handle",
                    depth=depth + 1,
                )
            if nested is not _REANCHOR_DROP:
                projected[key] = nested
        return projected
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        projected_items: list[object] = []
        for item in value[:_REANCHOR_SEQUENCE_CAP]:
            nested = _reanchor_projection(
                item,
                field_name=field_name,
                depth=depth + 1,
            )
            if nested is not _REANCHOR_DROP:
                projected_items.append(nested)
        return projected_items
    return _REANCHOR_DROP


def _build_reanchor_question_text(
    *,
    harness: SerialChainHarness,
    question: v3_prompt.ChainQuestion,
    interludes: Sequence[Mapping[str, object]],
) -> str:
    """Build a compact re-anchor from accepted products and current facts."""

    accepted_products = [
        projected
        for product in harness.transcript.accepted_products
        if (
            projected := _reanchor_projection(product)
        ) is not _REANCHOR_DROP
    ]
    facts = _reanchor_projection(question.payload)
    interlude_facts = _reanchor_projection(list(interludes))
    anchor = {
        "accepted_products": accepted_products,
        "current_question": {
            "contract_name": question.contract_name,
            "facts": facts if facts is not _REANCHOR_DROP else {},
            "interludes": (
                interlude_facts
                if interlude_facts is not _REANCHOR_DROP
                else []
            ),
        },
    }
    anchor_text = json.dumps(
        {"reanchor": anchor},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return anchor_text


def _messages_from_transcript(
    transcript: ChainTranscriptV1,
    system_content: str,
) -> list[BaseMessage]:
    """Build provider messages from the current immutable transcript."""

    messages: list[BaseMessage] = [SystemMessage(content=system_content)]
    for role, content in transcript.to_messages():
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            raise ExecutorContractError(f"Unknown transcript role {role!r}")
    return messages


def _consume_reanchor(
    *,
    harness: SerialChainHarness,
    question: v3_prompt.ChainQuestion,
    interludes: Sequence[Mapping[str, object]],
) -> str:
    """Consume the shared token and replace the transcript with its anchor."""

    if not isinstance(harness.budget, ContextBudgetLedger):
        raise ExecutorContractError(
            "serial harness requires a ContextBudgetLedger"
        )
    harness.budget.consume_reanchor()
    anchor_text = _build_reanchor_question_text(
        harness=harness,
        question=question,
        interludes=interludes,
    )
    harness.transcript = harness.transcript.reanchor(anchor_text)
    token_ledger = dict(harness.transcript.token_ledger or {})
    token_ledger["reanchor_used"] = 1
    harness.transcript = replace(
        harness.transcript,
        token_ledger=token_ledger,
    )
    record_token_ledger(token_ledger)
    return anchor_text


def _drop_unaccepted_reanchor_question(
    harness: SerialChainHarness,
    *,
    reanchored_current_question: bool,
) -> None:
    """Remove an unaccepted re-anchor question before the next stage."""

    if not reanchored_current_question:
        return
    if not harness.transcript.messages:
        return
    if harness.transcript.messages[-1].role != "human":
        return
    harness.transcript = replace(
        harness.transcript,
        messages=(),
        pending_interludes=(),
    )


def _replace_unaccepted_reanchor_question(
    harness: SerialChainHarness,
    content: str,
) -> None:
    """Replace only the current re-anchor tail before accepting its answer."""

    if not isinstance(content, str) or not content.strip():
        raise ExecutorContractError(
            "Re-anchored questions must be non-empty strings"
        )
    messages = harness.transcript.messages
    if not messages or messages[-1].role != "human":
        raise ExecutorContractError(
            "A re-anchored answer requires an unaccepted human tail"
        )
    current_question = messages[-1]
    replacement = replace(current_question, content=content)
    harness.transcript = replace(
        harness.transcript,
        messages=messages[:-1] + (replacement,),
        pending_interludes=(),
    )


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
    budget: ContextBudgetLedger
    system_content: str = ""
    primary_queue_wait_ms: int = 0
    primary_in_flight_at_start: int = 0

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
        record_accepted_transcript(
            self.transcript.to_messages(),
            system_content=self.system_content,
        )

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
    harness.system_content = system_content
    record_chain_system_head(system_content)
    messages = _messages_from_transcript(
        harness.transcript,
        system_content,
    )
    effective_deadline = (
        deadline_monotonic
        if deadline_monotonic is not None
        else harness.transcript.deadline_monotonic
    )
    effective_config = config_for_turn_deadline(config, effective_deadline)
    reanchored = False
    try:
        _admit_primary_request(
            harness=harness,
            messages=messages,
            config=effective_config,
        )
    except CognitionContextLimitError:
        if harness.transcript.reanchor_used:
            raise
        check_turn_deadline(effective_deadline)
        current_tail = ""
        if harness.transcript.messages:
            current_message = harness.transcript.messages[-1]
            if current_message.role == "human":
                current_tail = current_message.content
        try:
            registered_facts = json.loads(current_tail)
        except (TypeError, ValueError):
            registered_facts = {}
        if not isinstance(registered_facts, (Mapping, list)):
            registered_facts = {}
        reanchor_question = v3_prompt.ChainQuestion(
            contract_name="serial_model_step.v1",
            payload={"registered_facts": registered_facts},
        )
        _consume_reanchor(
            harness=harness,
            question=reanchor_question,
            interludes=(),
        )
        reanchored = True
        messages = _messages_from_transcript(
            harness.transcript,
            system_content,
        )
        try:
            _admit_primary_request(
                harness=harness,
                messages=messages,
                config=effective_config,
            )
        except CognitionContextLimitError:
            _drop_unaccepted_reanchor_question(
                harness,
                reanchored_current_question=reanchored,
            )
            raise
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
    observation_context: Mapping[str, object] | None = None,
    interludes: Sequence[Mapping[str, object]] = (),
    attempt_owner: str | None = None,
    v2_stage: str | None = None,
    v2_branch_ids: tuple[str, ...] | None = None,
    v2_local_attempt_start: int = 1,
    deterministic_only: bool = False,
    json_repair_callback: JsonRepairCallback | None = None,
    deadline_monotonic: float | None = None,
) -> SerialQuestionResult:
    """Invoke one serial question with bounded tail-safe repair attempts.

    ``observation_context`` supplies the first-consumer carriers. It is
    rendered only while the accepted transcript is empty, so each rejected
    attempt retains the complete packet and every accepted later row keeps the
    compact question format.
    """

    if attempt_limit <= 0:
        raise ExecutorContractError("Serial repair attempt_limit must be positive")
    if not isinstance(question, v3_prompt.ChainQuestion):
        raise ExecutorContractError("Serial repair requires a registered question")
    if (v2_stage is None) != (v2_branch_ids is None):
        raise ExecutorContractError(
            "V2 attempt stage and branch roster must be supplied together"
        )
    if v2_branch_ids is not None and not v2_branch_ids:
        raise ExecutorContractError("V2 attempt branch roster is required")
    if v2_local_attempt_start <= 0:
        raise ExecutorContractError(
            "V2 local attempt start must be positive"
        )
    if v2_stage is not None and current_v2_attempt_ledger() is None:
        raise ExecutorContractError(
            "V2 attempt coordinates require an ambient invocation ledger"
        )
    harness.system_content = system_content
    record_chain_system_head(system_content)
    transcript_messages = harness.transcript.to_messages()
    base_messages = _messages_from_transcript(
        harness.transcript,
        system_content,
    )

    if observation_context is not None and not transcript_messages:
        question_text = v3_prompt.build_first_user_message(
            observation_context=observation_context,
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
    raw_output: str | None = None
    failure_facts: list[Mapping[str, object]] = []
    failed_raw_outputs: set[str] = set()
    consecutive_empty_outputs = 0
    reanchored_current_question = False
    last_failure_kind: Literal["structural", "provider"] | None = None
    provider_failure_seen = False

    for attempt_number in range(1, attempt_limit + 1):
        attempt_started_at = time.perf_counter()
        try:
            check_turn_deadline(effective_deadline)
        except TurnDeadlineExceeded:
            deadline_messages = [
                *base_messages,
                HumanMessage(content=question_text),
            ]
            _record_primary_attempt_step(
                harness=harness,
                config=config,
                messages=deadline_messages,
                payload_text=question_text,
                attempt_number=attempt_number,
                started_at=attempt_started_at,
                admission=None,
                status="deadline",
                parse_status="deadline_exhausted",
                disposition="deadline_exhausted",
                reanchored=reanchored_current_question,
            )
            _drop_unaccepted_reanchor_question(
                harness,
                reanchored_current_question=reanchored_current_question,
            )
            return SerialQuestionResult(
                validated=None,
                raw_output=raw_output,
                disposition=SerialQuestionDisposition(
                    kind="deadline_exhausted",
                ),
            )
        payload_text = question_text
        if attempt_number > 1:
            payload_text = _repair_payload_text(
                question_text,
                question=question,
                attempt_index=attempt_number,
                failure_facts=failure_facts,
            )
        attempt_config = config
        if attempt_number > 1:
            attempt_config = replace(
                config,
                stage_name=f"{config.stage_name}.repair{attempt_number - 1}",
            )
        if reanchored_current_question:
            messages = [
                *base_messages[:-1],
                HumanMessage(content=payload_text),
            ]
        else:
            messages = [*base_messages, HumanMessage(content=payload_text)]
        repaired_output = False
        attempt_raw_output: str | None = None
        coordinates: tuple[Mapping[str, object], ...] | None = None
        admission: BudgetAdmission | None = None
        try:
            attempt_config = config_for_turn_deadline(
                attempt_config,
                effective_deadline,
            )
            try:
                admission = _admit_primary_request(
                    harness=harness,
                    messages=messages,
                    config=attempt_config,
                )
            except CognitionContextLimitError:
                if harness.transcript.reanchor_used:
                    _record_primary_attempt_step(
                        harness=harness,
                        config=attempt_config,
                        messages=messages,
                        payload_text=payload_text,
                        attempt_number=attempt_number,
                        started_at=attempt_started_at,
                        admission=admission,
                        status="budget",
                        parse_status="budget_exhausted",
                        disposition="budget_exhausted",
                        reanchored=reanchored_current_question,
                    )
                    raise
                _record_primary_attempt_step(
                    harness=harness,
                    config=attempt_config,
                    messages=messages,
                    payload_text=payload_text,
                    attempt_number=attempt_number,
                    started_at=attempt_started_at,
                    admission=admission,
                    status="budget",
                    parse_status="budget_exhausted",
                    disposition="reanchor",
                    reanchored=reanchored_current_question,
                )
                try:
                    check_turn_deadline(effective_deadline)
                except TurnDeadlineExceeded:
                    _record_primary_attempt_step(
                        harness=harness,
                        config=attempt_config,
                        messages=messages,
                        payload_text=payload_text,
                        attempt_number=attempt_number,
                        started_at=attempt_started_at,
                        admission=admission,
                        status="deadline",
                        parse_status="deadline_exhausted",
                        disposition="deadline_exhausted",
                        reanchored=reanchored_current_question,
                    )
                    _drop_unaccepted_reanchor_question(
                        harness,
                        reanchored_current_question=reanchored_current_question,
                    )
                    return SerialQuestionResult(
                        validated=None,
                        raw_output=raw_output,
                        disposition=SerialQuestionDisposition(
                            kind="deadline_exhausted",
                        ),
                    )
                question_text = _consume_reanchor(
                    harness=harness,
                    question=question,
                    interludes=interludes,
                )
                base_messages = _messages_from_transcript(
                    harness.transcript,
                    system_content,
                )
                reanchored_current_question = True
                payload_text = question_text
                if attempt_number > 1:
                    payload_text = _repair_payload_text(
                        question_text,
                        question=question,
                        attempt_index=attempt_number,
                        failure_facts=failure_facts,
                    )
                messages = [
                    *base_messages[:-1],
                    HumanMessage(content=payload_text),
                ]
                try:
                    admission = _admit_primary_request(
                        harness=harness,
                        messages=messages,
                        config=attempt_config,
                    )
                except CognitionContextLimitError:
                    _record_primary_attempt_step(
                        harness=harness,
                        config=attempt_config,
                        messages=messages,
                        payload_text=payload_text,
                        attempt_number=attempt_number,
                        started_at=attempt_started_at,
                        admission=admission,
                        status="budget",
                        parse_status="budget_exhausted",
                        disposition="budget_exhausted",
                        reanchored=reanchored_current_question,
                    )
                    _drop_unaccepted_reanchor_question(
                        harness,
                        reanchored_current_question=reanchored_current_question,
                    )
                    raise
            if v2_stage is not None:
                try:
                    coordinates = reserve_v2_model_attempt_batch(
                        stage=v2_stage,
                        branch_ids=v2_branch_ids,
                        local_attempt=(
                            v2_local_attempt_start + attempt_number - 1
                        ),
                    )
                except V2AttemptBudgetExhausted:
                    _record_primary_attempt_step(
                        harness=harness,
                        config=attempt_config,
                        messages=messages,
                        payload_text=payload_text,
                        attempt_number=attempt_number,
                        started_at=attempt_started_at,
                        admission=admission,
                        status="budget",
                        parse_status="budget_exhausted",
                        disposition="budget_exhausted",
                        reanchored=reanchored_current_question,
                    )
                    _drop_unaccepted_reanchor_question(
                        harness,
                        reanchored_current_question=reanchored_current_question,
                    )
                    return SerialQuestionResult(
                        validated=None,
                        raw_output=raw_output,
                        disposition=SerialQuestionDisposition(
                            kind="budget_exhausted",
                        ),
                    )
            if attempt_owner is not None:
                harness.ledger.record_attempt(attempt_owner)
            response = await llm.ainvoke(messages, config=attempt_config)
            response_content = getattr(response, "content", "")
            attempt_raw_output = (
                response_content
                if isinstance(response_content, str)
                else str(response_content or "")
            )
            raw_output = attempt_raw_output
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
                    coordinates[0],
                )
                if repaired is not None:
                    parsed = repaired
                    repaired_output = True
            if not isinstance(parsed, Mapping):
                raise TypeError("serial answer must be a JSON object")
            validated = validator(dict(parsed))
        except TurnDeadlineExceeded:
            if coordinates is not None:
                for coordinate in coordinates:
                    record_v2_attempt_disposition(
                        coordinate,
                        disposition="exhausted",
                    )
            _record_primary_attempt_step(
                harness=harness,
                config=attempt_config,
                messages=messages,
                payload_text=payload_text,
                attempt_number=attempt_number,
                started_at=attempt_started_at,
                admission=admission,
                status="deadline",
                parse_status="deadline_exhausted",
                disposition="deadline_exhausted",
                reanchored=reanchored_current_question,
            )
            _drop_unaccepted_reanchor_question(
                harness,
                reanchored_current_question=reanchored_current_question,
            )
            return SerialQuestionResult(
                validated=None,
                raw_output=raw_output,
                disposition=SerialQuestionDisposition(
                    kind="deadline_exhausted",
                ),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            last_failure_kind = "structural"
            failure_fact = _repair_failure_fact(
                exc,
                attempt_index=attempt_number,
                raw_output=attempt_raw_output,
            )
            failure_facts.append(failure_fact)
            repeated_raw_output = (
                attempt_raw_output is not None
                and attempt_raw_output in failed_raw_outputs
            )
            if attempt_raw_output is None:
                consecutive_empty_outputs = 0
            elif not attempt_raw_output.strip():
                consecutive_empty_outputs += 1
            else:
                consecutive_empty_outputs = 0
                failed_raw_outputs.add(attempt_raw_output)
            short_circuit = repeated_raw_output or consecutive_empty_outputs >= 2
            if coordinates is not None:
                for coordinate in coordinates:
                    record_v2_attempt_disposition(
                        coordinate,
                        disposition=(
                            "exhausted"
                            if attempt_number == attempt_limit or short_circuit
                            else "regenerate"
                        ),
                    )
            structural_disposition = (
                "exhausted" if attempt_number == attempt_limit or short_circuit
                else "regenerate"
            )
            _record_primary_attempt_step(
                harness=harness,
                config=attempt_config,
                messages=messages,
                payload_text=payload_text,
                attempt_number=attempt_number,
                started_at=attempt_started_at,
                admission=admission,
                status="structural",
                parse_status="contract_error",
                disposition=structural_disposition,
                reanchored=reanchored_current_question,
            )
            if short_circuit:
                _consume_remaining_repair_attempts(
                    harness=harness,
                    completed_attempt=attempt_number,
                    attempt_limit=attempt_limit,
                    attempt_owner=attempt_owner,
                    v2_stage=v2_stage,
                    v2_branch_ids=v2_branch_ids,
                    v2_local_attempt_start=v2_local_attempt_start,
                )
                _drop_unaccepted_reanchor_question(
                    harness,
                    reanchored_current_question=reanchored_current_question,
                )
                return SerialQuestionResult(
                    validated=None,
                    raw_output=raw_output,
                    disposition=SerialQuestionDisposition(
                        kind=(
                            "provider_exhausted"
                            if provider_failure_seen
                            else "structural_exhausted"
                        ),
                    ),
                )
            continue
        except (OpenAIError, httpx.HTTPError, ConnectionError, OSError) as exc:
            last_failure_kind = "provider"
            provider_failure_seen = True
            failure_fact = _repair_failure_fact(
                exc,
                attempt_index=attempt_number,
                raw_output=attempt_raw_output,
            )
            failure_facts.append(failure_fact)
            consecutive_empty_outputs = 0
            if coordinates is not None:
                for coordinate in coordinates:
                    record_v2_attempt_disposition(
                        coordinate,
                        disposition=(
                            "exhausted"
                            if attempt_number == attempt_limit
                            else "regenerate"
                        ),
                    )
            provider_disposition = (
                "exhausted" if attempt_number == attempt_limit else "regenerate"
            )
            _record_primary_attempt_step(
                harness=harness,
                config=attempt_config,
                messages=messages,
                payload_text=payload_text,
                attempt_number=attempt_number,
                started_at=attempt_started_at,
                admission=admission,
                status="provider",
                parse_status="provider_error",
                disposition=provider_disposition,
                reanchored=reanchored_current_question,
            )
            continue

        if reanchored_current_question:
            _replace_unaccepted_reanchor_question(harness, payload_text)
        else:
            harness.append_question(payload_text)
        accepted_answer = (
            json.dumps(
                parsed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if repaired_output
            else raw_output or ""
        )
        harness.accept_answer(
            accepted_answer,
            _accepted_product(question, validated),
        )
        if coordinates is not None:
            for coordinate in coordinates:
                record_v2_attempt_disposition(
                    coordinate,
                    disposition=(
                        "accepted"
                        if attempt_number == 1
                        else "recovered"
                    ),
                )
        _record_primary_attempt_step(
            harness=harness,
            config=attempt_config,
            messages=messages,
            payload_text=payload_text,
            attempt_number=attempt_number,
            started_at=attempt_started_at,
            admission=admission,
            status="accepted",
            parse_status="accepted",
            disposition=(
                "accepted" if attempt_number == 1 else "recovered"
            ),
            reanchored=reanchored_current_question,
        )
        return SerialQuestionResult(
            validated=validated,
            raw_output=raw_output,
            disposition=SerialQuestionDisposition(kind="accepted"),
        )

    _drop_unaccepted_reanchor_question(
        harness,
        reanchored_current_question=reanchored_current_question,
    )
    terminal_kind: QuestionDispositionKind
    if provider_failure_seen or last_failure_kind == "provider":
        terminal_kind = "provider_exhausted"
    elif last_failure_kind == "structural":
        terminal_kind = "structural_exhausted"
    else:
        terminal_kind = "budget_exhausted"
    return SerialQuestionResult(
        validated=None,
        raw_output=raw_output,
        disposition=SerialQuestionDisposition(kind=terminal_kind),
    )


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
