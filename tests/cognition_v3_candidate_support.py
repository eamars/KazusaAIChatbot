"""V3-only candidate trial support for governed live tests."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Awaitable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from langchain_core.messages import BaseMessage

from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CognitionChainServicesV3,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    validate_cognition_core_input,
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.llm_interface.contracts import (
    LLMCallConfig,
    LLMResponse,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LIVE_CASE_MANIFEST_PATH = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "cognition_core_v3_live_case_manifest.json"
)
DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "test_artifacts" / "cognition_core_v3"
TRIAL_ARTIFACT_SCHEMA = "cognition_v3_candidate_trial.v2"
PAIR_INVALIDATION_SCHEMA = "cognition_v3_matched_pair_invalidation.v1"
TRIAL_INDICES = frozenset({1, 2, 3})
ELIGIBLE_RESULT = "eligible_semantic_result"
HARD_BOUNDARY_FAILURE = "hard_boundary_failure"
INVALID_NO_RESULT = "provider_or_harness_invalid_no_result"


class TrialAlreadySealedError(RuntimeError):
    """Raised when a caller attempts to replace a completed trial artifact."""


class TrialExecutionError(RuntimeError):
    """Raised after a failed trial has been durably captured."""

    def __init__(self, message: str, *, artifact_path: Path) -> None:
        super().__init__(message)
        self.artifact_path = artifact_path


@dataclass(frozen=True)
class TrialIdentity:
    """Stable identity for one V3 candidate sample."""

    baseline_id: str
    case_id: str
    engine: str
    trial_index: int
    attempt_index: int = 1

    def validate(self) -> "TrialIdentity":
        if not self.baseline_id or not _safe_component(self.baseline_id):
            raise ValueError("baseline_id must be filesystem-safe")
        if not self.case_id or not _safe_component(self.case_id):
            raise ValueError("case_id must be filesystem-safe")
        if self.engine != "v3":
            raise ValueError("candidate engine must be v3")
        if self.trial_index not in TRIAL_INDICES:
            raise ValueError("trial_index must be 1, 2, or 3")
        if self.attempt_index not in {1, 2, 3}:
            raise ValueError("attempt_index must be 1, 2, or 3")
        return self

    @property
    def trial_id(self) -> str:
        return f"{self.baseline_id}:{self.case_id}:v3:trial-{self.trial_index}"


class EngineRunner(Protocol):
    def __call__(
        self,
        input_payload: Mapping[str, Any],
        services: CognitionChainServicesV3,
    ) -> Awaitable[Mapping[str, Any]]:
        """Run the effect-free V3 facade."""


class CapturingLLMInvoker:
    """Capture protected provider requests and responses without credentials."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.calls: list[dict[str, Any]] = []
        self._next_sequence = 1

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        record = self._start_record(messages, config=config, invocation="async")
        started_at = time.perf_counter()
        try:
            response = await self._delegate.ainvoke(messages, config=config)
        except Exception as exc:
            record["wall_time_ms"] = _elapsed_ms(started_at)
            record["exception"] = _exception_record(exc)
            raise
        record["wall_time_ms"] = _elapsed_ms(started_at)
        record["response"] = _response_record(response)
        return response

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        record = self._start_record(messages, config=config, invocation="sync")
        started_at = time.perf_counter()
        try:
            response = self._delegate.invoke(messages, config=config)
        except Exception as exc:
            record["wall_time_ms"] = _elapsed_ms(started_at)
            record["exception"] = _exception_record(exc)
            raise
        record["wall_time_ms"] = _elapsed_ms(started_at)
        record["response"] = _response_record(response)
        return response

    def _start_record(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
        invocation: str,
    ) -> dict[str, Any]:
        record = {
            "sequence": self._next_sequence,
            "invocation": invocation,
            "config": _config_record(config),
            "messages": [_message_record(message) for message in messages],
        }
        self._next_sequence += 1
        self.calls.append(record)
        return record


def load_live_case_manifest() -> dict[str, Any]:
    value = json.loads(LIVE_CASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("live-case manifest root must be an object")
    return value


def find_case_row(case_id: str) -> dict[str, Any]:
    rows = load_live_case_manifest()["cases"]
    matches = [row for row in rows if row.get("case_id") == case_id]
    if len(matches) != 1:
        raise KeyError(f"expected one manifest row for {case_id!r}")
    return deepcopy(matches[0])


def render_case_input(manifest_row: Mapping[str, Any]) -> dict[str, Any]:
    canonical_input = deepcopy(manifest_row["canonical_input"])
    return deepcopy(validate_cognition_core_input(canonical_input))


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def trial_artifact_path(root: Path, identity: TrialIdentity) -> Path:
    identity.validate()
    filename = (
        f"{identity.case_id}__{identity.engine}__trial-{identity.trial_index}"
        f"__attempt-{identity.attempt_index}.json"
    )
    return root / identity.baseline_id / "raw_trials" / filename


def matched_pair_invalidation_path(root: Path, identity: TrialIdentity) -> Path:
    identity.validate()
    if identity.attempt_index == 1:
        raise ValueError("attempt 1 has no replacement invalidation")
    suffix = "" if identity.attempt_index == 2 else "__replacement-attempt-3"
    return root / identity.baseline_id / "invalidations" / (
        f"{identity.case_id}__trial-{identity.trial_index}{suffix}.json"
    )


async def run_effect_free_trial(
    manifest_row: Mapping[str, Any],
    *,
    identity: TrialIdentity,
    services: CognitionChainServicesV3,
    runner: EngineRunner,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    environment_fingerprint: Mapping[str, Any] | None = None,
    rerun_invalidation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run and seal one direct V3 facade trial without external effects."""

    identity.validate()
    if manifest_row.get("case_id") != identity.case_id:
        raise ValueError("trial identity and manifest case differ")
    artifact_path = trial_artifact_path(artifact_root, identity)
    if artifact_path.exists():
        raise TrialAlreadySealedError(f"trial artifact is already sealed: {artifact_path}")
    if identity.attempt_index != 1 and rerun_invalidation is None:
        raise ValueError("replacement attempt requires a retained invalidation")

    input_payload = render_case_input(manifest_row)
    input_snapshot = deepcopy(input_payload)
    input_sha256 = canonical_sha256(input_snapshot)
    capturing_llm = CapturingLLMInvoker(services.llm)
    captured_services = replace(services, llm=capturing_llm)
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started_at = time.perf_counter()
    output: dict[str, Any] | None = None
    exception: Exception | None = None
    validator_result: dict[str, Any] = {
        "input": "passed",
        "output": "not_run",
        "input_unchanged": False,
    }
    try:
        candidate = await runner(input_payload, captured_services)
        output = dict(validate_cognition_core_output(candidate))
        validator_result["output"] = "passed"
    except Exception as exc:  # noqa: BLE001
        exception = exc
        validator_result["output"] = "failed"
    validator_result["input_unchanged"] = canonical_sha256(input_payload) == input_sha256
    if not validator_result["input_unchanged"] and exception is None:
        exception = RuntimeError("engine mutated the canonical input")
    artifact = {
        "schema_version": TRIAL_ARTIFACT_SCHEMA,
        "baseline_id": identity.baseline_id,
        "trial_id": identity.trial_id,
        "case_id": identity.case_id,
        "engine": identity.engine,
        "trial_index": identity.trial_index,
        "attempt_index": identity.attempt_index,
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_ms": _elapsed_ms(started_at),
        "disposition": _trial_disposition(exception),
        "semantic_result_available": output is not None,
        "input_sha256": input_sha256,
        "canonical_input": input_snapshot,
        "output_sha256": canonical_sha256(output) if output is not None else None,
        "typed_output": output,
        "validator_result": validator_result,
        "model_calls": capturing_llm.calls,
        "environment_fingerprint": dict(environment_fingerprint or {}),
        "effect_free_contract": {
            "direct_engine_facade_only": True,
            "state_commit": False,
            "action_execution": False,
            "resolver_execution": False,
            "surface_delivery": False,
            "database_semantic_write": False,
            "adapter_delivery": False,
            "scheduler_effect": False,
        },
        "exception": _exception_record(exception) if exception else None,
    }
    sealed_path = seal_trial_artifact(artifact_path, artifact)
    artifact["artifact_path"] = sealed_path.as_posix()
    artifact["artifact_sha256"] = hashlib.sha256(sealed_path.read_bytes()).hexdigest()
    if exception is not None:
        raise TrialExecutionError(
            f"trial {identity.trial_id} failed with {artifact['disposition']}",
            artifact_path=sealed_path,
        ) from exception
    return artifact


def seal_trial_artifact(path: Path, artifact: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
    except FileExistsError as exc:
        raise TrialAlreadySealedError(f"trial artifact is already sealed: {path}") from exc
    return path


def baseline_id_from_environment() -> str:
    value = os.environ.get("COGNITION_V3_BASELINE_ID", "").strip()
    if not value or not _safe_component(value):
        raise RuntimeError("COGNITION_V3_BASELINE_ID is required and filesystem-safe")
    return value


def trial_index_from_environment() -> int:
    try:
        value = int(os.environ.get("COGNITION_V3_TRIAL_INDEX", ""))
    except ValueError as exc:
        raise RuntimeError("COGNITION_V3_TRIAL_INDEX must be 1, 2, or 3") from exc
    if value not in TRIAL_INDICES:
        raise RuntimeError("COGNITION_V3_TRIAL_INDEX must be 1, 2, or 3")
    return value


def attempt_index_from_environment() -> int:
    try:
        value = int(os.environ.get("COGNITION_V3_ATTEMPT_INDEX", "1"))
    except ValueError as exc:
        raise RuntimeError("COGNITION_V3_ATTEMPT_INDEX must be 1, 2, or 3") from exc
    if value not in {1, 2, 3}:
        raise RuntimeError("COGNITION_V3_ATTEMPT_INDEX must be 1, 2, or 3")
    return value


def _trial_disposition(exception: Exception | None) -> str:
    if exception is None:
        return ELIGIBLE_RESULT
    if isinstance(exception, (ConnectionError, OSError, TimeoutError)):
        return INVALID_NO_RESULT
    return HARD_BOUNDARY_FAILURE


def _safe_component(value: str) -> bool:
    return all(character.isalnum() or character in "-_." for character in value)


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))


def _message_record(message: BaseMessage) -> dict[str, Any]:
    return {
        "message_type": type(message).__name__,
        "content": _json_safe(getattr(message, "content", "")),
    }


def _config_record(config: LLMCallConfig) -> dict[str, Any]:
    return {
        "stage_name": config.stage_name,
        "route_name": config.route_name,
        "base_url_sha256": hashlib.sha256(config.base_url.encode("utf-8")).hexdigest(),
        "model": config.model,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_completion_tokens": config.max_completion_tokens,
        "presence_penalty": config.presence_penalty,
        "timeout_seconds": config.timeout_seconds,
        "thinking_enabled": config.thinking.enabled,
    }


def _response_record(response: LLMResponse) -> dict[str, Any]:
    backend = response.backend
    return {
        "content": response.content,
        "backend": {
            "route_name": backend.route_name,
            "backend_kind": backend.backend_kind,
            "model_family": backend.model_family,
            "model": backend.model,
            "normalized_base_url_sha256": hashlib.sha256(
                backend.normalized_base_url.encode("utf-8")
            ).hexdigest(),
            "thinking_strategy": backend.thinking_strategy,
            "confidence": backend.confidence,
            "generation": backend.generation,
        },
        "usage": _json_safe(response.usage),
        "provider_raw_content": _json_safe(getattr(response.raw_response, "content", None)),
        "provider_response_metadata": _json_safe(
            getattr(response.raw_response, "response_metadata", None)
        ),
    }


def _exception_record(exception: Exception) -> dict[str, str]:
    return {
        "type": type(exception).__name__,
        "module": type(exception).__module__,
        "message": str(exception)[:2000],
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in list(value.items())[:100]}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in list(value)[:100]]
    return repr(value)[:4000]
