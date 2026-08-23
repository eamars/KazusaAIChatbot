"""Strict public and internal contracts for the agentic resolver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

REQUEST_SCHEMA_VERSION = "agentic_resolver_request.v1"
RESULT_SCHEMA_VERSION = "agentic_resolver_result.v1"
SUBAGENT_RESULT_SCHEMA_VERSION = "agentic_resolver_subagent_result.v1"

PROJECT_CONTEXT_TOKEN_CAP = 50_000
COMPLETION_RESERVE_TOKEN_CAP = 8_000
MODEL_STEP_HARD_CAP = 16
TOOL_CALL_HARD_CAP = 12
STRUCTURAL_REPLACEMENT_CAP = 2
ROOT_SUBAGENT_RUN_CAP = 3
MODEL_VISIBLE_RESULT_CHAR_CAP = 8_000
SKILL_COUNT_CAP = 64
SKILL_DESCRIPTION_CHAR_CAP = 500
SKILL_BODY_CHAR_CAP = 16_000
SESSION_TIMEOUT_HARD_CAP_SECONDS = 600.0
TOOL_TIMEOUT_HARD_CAP_SECONDS = 180.0

MAX_CONTEXT_ITEMS = 32
MAX_CONTEXT_TEXT_CHARS = 2_000
MAX_SUMMARY_CHARS = 4_000
MAX_RESULT_ITEMS = 16
MAX_SUBAGENT_DESCRIPTION_CHARS = 200
MAX_SUBAGENT_OBJECTIVE_CHARS = 4_000

ResolverStatus = Literal[
    "resolved",
    "partial",
    "needs_user_input",
    "approval_required",
    "unavailable",
    "budget_exhausted",
    "failed",
]
RESOLVER_STATUSES = frozenset({
    "resolved",
    "partial",
    "needs_user_input",
    "approval_required",
    "unavailable",
    "budget_exhausted",
    "failed",
})


class AgenticResolverContractError(ValueError):
    """Identify a strict resolver contract violation with a stable code."""

    def __init__(self, message: str, *, code: str = "invalid_contract") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AgenticResolverContextV1:
    """Prompt-safe context supplied with one semantic objective."""

    facts: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    desired_output: str = ""

    @classmethod
    def from_mapping(cls, value: object) -> AgenticResolverContextV1:
        """Validate an exact JSON context object.

        Args:
            value: Candidate caller-authored prompt-safe context.

        Returns:
            An immutable context with bounded strings and arrays.
        """

        data = _exact_mapping(
            value,
            {"facts", "constraints", "desired_output"},
            "context",
        )
        context = cls(
            facts=_text_tuple(
                data["facts"],
                "context.facts",
                maximum_items=MAX_CONTEXT_ITEMS,
                maximum_chars=MAX_CONTEXT_TEXT_CHARS,
            ),
            constraints=_text_tuple(
                data["constraints"],
                "context.constraints",
                maximum_items=MAX_CONTEXT_ITEMS,
                maximum_chars=MAX_CONTEXT_TEXT_CHARS,
            ),
            desired_output=_text(
                data["desired_output"],
                "context.desired_output",
                maximum=MAX_CONTEXT_TEXT_CHARS,
                allow_empty=True,
            ),
        )
        return context

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object used by the task protocol."""

        value = {
            "facts": list(self.facts),
            "constraints": list(self.constraints),
            "desired_output": self.desired_output,
        }
        return value


@dataclass(frozen=True)
class AgenticResolverRequestV1:
    """One bounded semantic request accepted by the public runtime."""

    objective: str
    context: AgenticResolverContextV1 = field(
        default_factory=AgenticResolverContextV1
    )
    schema_version: str = field(default=REQUEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        bounded_objective = _text(
            self.objective,
            "objective",
            maximum=MAX_SUBAGENT_OBJECTIVE_CHARS,
        )
        object.__setattr__(self, "objective", bounded_objective)
        if not isinstance(self.context, AgenticResolverContextV1):
            raise AgenticResolverContractError(
                "context: expected AgenticResolverContextV1"
            )

    @classmethod
    def from_mapping(cls, value: object) -> AgenticResolverRequestV1:
        """Validate one exact public request JSON object."""

        data = _exact_mapping(
            value,
            {"schema_version", "objective", "context"},
            "agentic_resolver_request",
        )
        if data["schema_version"] != REQUEST_SCHEMA_VERSION:
            raise AgenticResolverContractError(
                f"schema_version: expected {REQUEST_SCHEMA_VERSION}"
            )
        context = AgenticResolverContextV1.from_mapping(data["context"])
        request = cls(
            objective=_text(
                data["objective"],
                "objective",
                maximum=MAX_SUBAGENT_OBJECTIVE_CHARS,
            ),
            context=context,
        )
        return request

    def to_dict(self) -> dict[str, object]:
        """Return the exact public request JSON object."""

        value = {
            "schema_version": self.schema_version,
            "objective": self.objective,
            "context": self.context.to_dict(),
        }
        return value


@dataclass(frozen=True)
class AgenticResolverEvidenceV1:
    """One terminal evidence claim bound to an accepted observation."""

    observation_id: str
    summary: str
    provenance_refs: tuple[str, ...]
    limitations: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> AgenticResolverEvidenceV1:
        """Validate one exact terminal evidence JSON row."""

        data = _exact_mapping(
            value,
            {
                "observation_id",
                "summary",
                "provenance_refs",
                "limitations",
            },
            "submit_result.evidence",
        )
        evidence = cls(
            observation_id=_text(
                data["observation_id"],
                "evidence.observation_id",
                maximum=200,
            ),
            summary=_text(
                data["summary"],
                "evidence.summary",
                maximum=MAX_CONTEXT_TEXT_CHARS,
            ),
            provenance_refs=_text_tuple(
                data["provenance_refs"],
                "evidence.provenance_refs",
                maximum_items=MAX_RESULT_ITEMS,
                maximum_chars=MAX_CONTEXT_TEXT_CHARS,
            ),
            limitations=_text_tuple(
                data["limitations"],
                "evidence.limitations",
                maximum_items=MAX_RESULT_ITEMS,
                maximum_chars=MAX_CONTEXT_TEXT_CHARS,
            ),
        )
        return evidence

    def to_dict(self) -> dict[str, object]:
        """Return the public evidence JSON projection."""

        value = {
            "observation_id": self.observation_id,
            "summary": self.summary,
            "provenance_refs": list(self.provenance_refs),
            "limitations": list(self.limitations),
        }
        return value


@dataclass(frozen=True)
class SubmitResultV1:
    """Model-authored semantic terminal fields for submit_result."""

    status: ResolverStatus
    summary: str
    evidence: tuple[AgenticResolverEvidenceV1, ...]
    completed_tasks: tuple[str, ...]
    remaining_needs: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> SubmitResultV1:
        """Validate terminal arguments without changing semantic channels."""

        data = _exact_mapping(
            value,
            {
                "status",
                "summary",
                "evidence",
                "completed_tasks",
                "remaining_needs",
            },
            "submit_result",
        )
        status = _text(data["status"], "submit_result.status", maximum=40)
        if status not in RESOLVER_STATUSES:
            raise AgenticResolverContractError(
                "submit_result.status: unsupported value"
            )
        evidence_rows = _mapping_sequence(
            data["evidence"],
            "submit_result.evidence",
            maximum=MAX_RESULT_ITEMS,
        )
        evidence = tuple(
            AgenticResolverEvidenceV1.from_mapping(row)
            for row in evidence_rows
        )
        completed_tasks = _text_tuple(
            data["completed_tasks"],
            "submit_result.completed_tasks",
            maximum_items=MAX_RESULT_ITEMS,
            maximum_chars=MAX_CONTEXT_TEXT_CHARS,
        )
        remaining_needs = _text_tuple(
            data["remaining_needs"],
            "submit_result.remaining_needs",
            maximum_items=MAX_RESULT_ITEMS,
            maximum_chars=MAX_CONTEXT_TEXT_CHARS,
        )
        if status == "resolved" and remaining_needs:
            raise AgenticResolverContractError(
                "submit_result: resolved cannot retain remaining_needs"
            )
        if status == "partial" and (not evidence or not remaining_needs):
            raise AgenticResolverContractError(
                "submit_result: partial requires evidence and remaining_needs"
            )
        result = cls(
            status=status,
            summary=_text(
                data["summary"],
                "submit_result.summary",
                maximum=MAX_SUMMARY_CHARS,
            ),
            evidence=evidence,
            completed_tasks=completed_tasks,
            remaining_needs=remaining_needs,
        )
        return result


@dataclass
class AgenticResolverUsageV1:
    """Code-owned counters and provider usage for one resolver session."""

    model_steps: int = 0
    tool_calls: int = 0
    subagent_runs: int = 0
    contract_errors: int = 0
    compactions: int = 0
    estimated_context_tokens_peak: int = 0
    provider_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return the bounded public usage JSON object."""

        value = {
            "model_steps": self.model_steps,
            "tool_calls": self.tool_calls,
            "subagent_runs": self.subagent_runs,
            "contract_errors": self.contract_errors,
            "compactions": self.compactions,
            "estimated_context_tokens_peak": (
                self.estimated_context_tokens_peak
            ),
            "provider_usage": dict(self.provider_usage),
        }
        return value


@dataclass(frozen=True)
class AgenticResolverResultV1:
    """Validated public terminal result returned by resolve."""

    session_id: str
    status: ResolverStatus
    summary: str
    evidence: tuple[AgenticResolverEvidenceV1, ...]
    completed_tasks: tuple[str, ...]
    remaining_needs: tuple[str, ...]
    usage: AgenticResolverUsageV1
    schema_version: str = field(default=RESULT_SCHEMA_VERSION, init=False)

    def to_dict(self) -> dict[str, object]:
        """Return the exact public terminal JSON projection."""

        value = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "status": self.status,
            "summary": self.summary,
            "evidence": [row.to_dict() for row in self.evidence],
            "completed_tasks": list(self.completed_tasks),
            "remaining_needs": list(self.remaining_needs),
            "usage": self.usage.to_dict(),
        }
        return value


@dataclass(frozen=True)
class AgenticResolverSubagentTaskV1:
    """Model-authored self-contained child task accepted by run_subagent."""

    description: str
    objective: str
    context: AgenticResolverContextV1

    @classmethod
    def from_mapping(cls, value: object) -> AgenticResolverSubagentTaskV1:
        """Validate one exact child task without inheriting parent history."""

        data = _exact_mapping(
            value,
            {"description", "objective", "context"},
            "run_subagent",
        )
        task = cls(
            description=_text(
                data["description"],
                "run_subagent.description",
                maximum=MAX_SUBAGENT_DESCRIPTION_CHARS,
            ),
            objective=_text(
                data["objective"],
                "run_subagent.objective",
                maximum=MAX_SUBAGENT_OBJECTIVE_CHARS,
            ),
            context=AgenticResolverContextV1.from_mapping(data["context"]),
        )
        return task


@dataclass(frozen=True)
class AgenticResolverSubagentEvidenceV1:
    """Child evidence details projected without a parent observation handle."""

    summary: str
    provenance_refs: tuple[str, ...]
    limitations: tuple[str, ...]

    @classmethod
    def from_terminal_evidence(
        cls,
        evidence: AgenticResolverEvidenceV1,
    ) -> AgenticResolverSubagentEvidenceV1:
        """Project child terminal evidence while dropping its private ID."""

        projected = cls(
            summary=evidence.summary,
            provenance_refs=evidence.provenance_refs,
            limitations=evidence.limitations,
        )
        return projected

    def to_dict(self) -> dict[str, object]:
        """Return the parent-visible child evidence detail object."""

        value = {
            "summary": self.summary,
            "provenance_refs": list(self.provenance_refs),
            "limitations": list(self.limitations),
        }
        return value


@dataclass(frozen=True)
class AgenticResolverSubagentResultV1:
    """Bounded typed child projection returned to the parent session."""

    subagent_id: str
    observation_id: str
    description: str
    status: ResolverStatus
    summary: str
    evidence: tuple[AgenticResolverSubagentEvidenceV1, ...]
    remaining_needs: tuple[str, ...]
    schema_version: str = field(
        default=SUBAGENT_RESULT_SCHEMA_VERSION,
        init=False,
    )
    message_type: str = field(default="subagent_result", init=False)

    def to_dict(self) -> dict[str, object]:
        """Return the parent-visible child result JSON object."""

        value = {
            "schema_version": self.schema_version,
            "message_type": self.message_type,
            "subagent_id": self.subagent_id,
            "observation_id": self.observation_id,
            "description": self.description,
            "status": self.status,
            "summary": self.summary,
            "evidence": [row.to_dict() for row in self.evidence],
            "remaining_needs": list(self.remaining_needs),
        }
        return value


@dataclass(frozen=True)
class AgenticResolverLimitsV1:
    """Caller-lowerable resolver limits bounded by fixed project maxima."""

    context_window_tokens: int = PROJECT_CONTEXT_TOKEN_CAP
    completion_reserve_tokens: int = COMPLETION_RESERVE_TOKEN_CAP
    max_model_steps: int = 8
    max_tool_calls: int = 6
    max_contract_replacements: int = STRUCTURAL_REPLACEMENT_CAP
    max_subagent_runs: int = ROOT_SUBAGENT_RUN_CAP
    max_tool_result_characters: int = MODEL_VISIBLE_RESULT_CHAR_CAP
    max_subagent_result_characters: int = MODEL_VISIBLE_RESULT_CHAR_CAP
    max_skills: int = SKILL_COUNT_CAP
    max_skill_description_characters: int = SKILL_DESCRIPTION_CHAR_CAP
    max_skill_body_characters: int = SKILL_BODY_CHAR_CAP
    session_timeout_seconds: float = 300.0
    tool_timeout_seconds: float = TOOL_TIMEOUT_HARD_CAP_SECONDS

    def __post_init__(self) -> None:
        _bounded_positive_int(
            self.context_window_tokens,
            "context_window_tokens",
            PROJECT_CONTEXT_TOKEN_CAP,
        )
        _bounded_positive_int(
            self.completion_reserve_tokens,
            "completion_reserve_tokens",
            COMPLETION_RESERVE_TOKEN_CAP,
        )
        if self.completion_reserve_tokens >= self.context_window_tokens:
            raise AgenticResolverContractError(
                "completion_reserve_tokens must be below context_window_tokens"
            )
        _bounded_positive_int(
            self.max_model_steps,
            "max_model_steps",
            MODEL_STEP_HARD_CAP,
        )
        _bounded_positive_int(
            self.max_tool_calls,
            "max_tool_calls",
            TOOL_CALL_HARD_CAP,
        )
        _bounded_positive_int(
            self.max_contract_replacements,
            "max_contract_replacements",
            STRUCTURAL_REPLACEMENT_CAP,
        )
        _bounded_positive_int(
            self.max_subagent_runs,
            "max_subagent_runs",
            ROOT_SUBAGENT_RUN_CAP,
        )
        _bounded_positive_int(
            self.max_tool_result_characters,
            "max_tool_result_characters",
            MODEL_VISIBLE_RESULT_CHAR_CAP,
        )
        _bounded_positive_int(
            self.max_subagent_result_characters,
            "max_subagent_result_characters",
            MODEL_VISIBLE_RESULT_CHAR_CAP,
        )
        _bounded_positive_int(self.max_skills, "max_skills", SKILL_COUNT_CAP)
        _bounded_positive_int(
            self.max_skill_description_characters,
            "max_skill_description_characters",
            SKILL_DESCRIPTION_CHAR_CAP,
        )
        _bounded_positive_int(
            self.max_skill_body_characters,
            "max_skill_body_characters",
            SKILL_BODY_CHAR_CAP,
        )
        _bounded_positive_number(
            self.session_timeout_seconds,
            "session_timeout_seconds",
            SESSION_TIMEOUT_HARD_CAP_SECONDS,
        )
        _bounded_positive_number(
            self.tool_timeout_seconds,
            "tool_timeout_seconds",
            TOOL_TIMEOUT_HARD_CAP_SECONDS,
        )

    @property
    def input_ceiling_tokens(self) -> int:
        """Return the context capacity remaining after completion reserve."""

        ceiling = self.context_window_tokens - self.completion_reserve_tokens
        return ceiling


def validated_request(value: object) -> AgenticResolverRequestV1:
    """Return a validated public request from a typed or JSON value."""

    if isinstance(value, AgenticResolverRequestV1):
        return value
    request = AgenticResolverRequestV1.from_mapping(value)
    return request


def _exact_mapping(
    value: object,
    expected_keys: set[str],
    label: str,
) -> Mapping[str, object]:
    """Require one mapping with exactly the declared contract keys."""

    if not isinstance(value, Mapping):
        raise AgenticResolverContractError(f"{label}: expected object")
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unknown = sorted(actual_keys - expected_keys)
        raise AgenticResolverContractError(
            f"{label}: missing={missing} unknown={unknown}"
        )
    return value


def _text(
    value: object,
    label: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    """Require one bounded string without silently truncating it."""

    if not isinstance(value, str):
        raise AgenticResolverContractError(f"{label}: expected string")
    normalized = value.strip()
    if not normalized and not allow_empty:
        raise AgenticResolverContractError(f"{label}: expected non-empty string")
    if len(normalized) > maximum:
        raise AgenticResolverContractError(
            f"{label}: exceeds {maximum} characters"
        )
    return normalized


def _text_tuple(
    value: object,
    label: str,
    *,
    maximum_items: int,
    maximum_chars: int,
) -> tuple[str, ...]:
    """Require one bounded sequence of non-empty strings."""

    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise AgenticResolverContractError(f"{label}: expected string list")
    if len(value) > maximum_items:
        raise AgenticResolverContractError(
            f"{label}: exceeds {maximum_items} items"
        )
    items = tuple(
        _text(item, f"{label}[{index}]", maximum=maximum_chars)
        for index, item in enumerate(value)
    )
    return items


def _mapping_sequence(
    value: object,
    label: str,
    *,
    maximum: int,
) -> list[Mapping[str, object]]:
    """Require one bounded list of mapping rows."""

    if not isinstance(value, list):
        raise AgenticResolverContractError(f"{label}: expected list")
    if len(value) > maximum:
        raise AgenticResolverContractError(
            f"{label}: exceeds {maximum} items"
        )
    rows: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise AgenticResolverContractError(
                f"{label}[{index}]: expected object"
            )
        rows.append(item)
    return rows


def _bounded_positive_int(value: int, label: str, maximum: int) -> None:
    """Require a positive non-boolean integer within a hard maximum."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AgenticResolverContractError(f"{label}: expected positive integer")
    if value > maximum:
        raise AgenticResolverContractError(f"{label}: exceeds hard maximum")


def _bounded_positive_number(
    value: float,
    label: str,
    maximum: float,
) -> None:
    """Require a positive timeout within its hard maximum."""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise AgenticResolverContractError(f"{label}: expected positive number")
    if value > maximum:
        raise AgenticResolverContractError(f"{label}: exceeds hard maximum")
