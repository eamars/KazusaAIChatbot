"""Live matched-pair performance evidence for Cognition V2 and V3."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import threading
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import pytest
from langchain_core.messages import BaseMessage

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreInputV2,
    CognitionCoreServicesV2,
    validate_cognition_core_input,
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.cognition_core_v2.facade import (
    run_cognition as run_cognition_v2,
)
from kazusa_ai_chatbot.cognition_core_v3 import facade as v3_facade
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CognitionChainServicesV3,
)
from kazusa_ai_chatbot.cognition_core_v3.registry import (
    APPRAISAL_STAGE_FAMILIES,
)
from kazusa_ai_chatbot.cognition_core_v3.session import ChainSessionRegistry
from kazusa_ai_chatbot.config import COGNITION_CORE_ENGINE
from kazusa_ai_chatbot.llm_interface.contracts import (
    LLMCallConfig,
    LLMResponse,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from tests.cognition_core_v3_comparison_harness import (
    DEFAULT_ARTIFACT_ROOT,
    _config_record,
    baseline_id_from_environment,
    canonical_sha256,
    find_case_row,
    load_live_case_manifest,
    render_case_input,
)
from tests.test_cognition_core_v3_candidate_live_llm import (
    sanitized_v3_environment_fingerprint,
)

pytestmark = pytest.mark.live_llm

PERFORMANCE_SCHEMA = "cognition_v3_performance.v1"
PERFORMANCE_RUN_ID_ENV = "COGNITION_V3_PERFORMANCE_RUN_ID"
COLD_TRIAL_COUNT = 5
WARM_TRIAL_COUNT = 20
RESOLVER_TWO_CYCLE_PAIR_COUNT = 10
SIDECAR_TRIAL_COUNT = 20
PRIMARY_REDUCTION_MINIMUM = 0.25
COLD_FIRST_PRIMARY_RATIO_MAXIMUM = 1.20
FULL_RUN_MEDIAN_RATIO_MAXIMUM = 1.10
FULL_RUN_P95_RATIO_MAXIMUM = 1.15
RESOLVER_V2_RATIO_MAXIMUM = 0.75
RESOLVER_COLD_REBUILD_RATIO_MAXIMUM = 0.60

EngineName = Literal["v2", "v3"]

_V2_CONFIG_FIELDS = (
    "appraisal_event_agency_config",
    "appraisal_relationship_social_config",
    "appraisal_moral_identity_config",
    "appraisal_goal_threat_outcome_config",
    "appraisal_epistemic_comparison_memory_config",
    "appraisal_existential_drive_config",
    "goal_ordinary_response_config",
    "goal_active_branch_config",
    "workspace_collapse_config",
    "action_planning_config",
    "action_authorization_config",
    "resolver_authorization_config",
)


def _safe_component(value: str) -> bool:
    """Return whether one artifact path component is bounded and safe."""

    return bool(value) and all(
        character.isalnum() or character in {"-", "_", "."}
        for character in value
    )


def _performance_run_id() -> str:
    """Return the explicit run identity used for exclusive artifacts."""

    value = os.environ.get(PERFORMANCE_RUN_ID_ENV, "").strip()
    if not _safe_component(value):
        raise RuntimeError(
            f"{PERFORMANCE_RUN_ID_ENV} must be a filesystem-safe value"
        )
    return value


def _performance_artifact_path(node_name: str) -> Path:
    """Return the exclusive raw-artifact path for one performance node."""

    path = (
        DEFAULT_ARTIFACT_ROOT
        / baseline_id_from_environment()
        / "performance"
        / _performance_run_id()
        / f"{node_name}.json"
    )
    return path


def _seal_performance_artifact(
    node_name: str,
    artifact: Mapping[str, Any],
) -> Path:
    """Write one immutable performance artifact without replacing evidence."""

    path = _performance_artifact_path(node_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        artifact,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
    except FileExistsError as exc:
        raise RuntimeError(
            f"performance artifact is already sealed: {path}"
        ) from exc
    return path


def _route_identity(config: LLMCallConfig | None) -> tuple[str, str] | None:
    """Return the endpoint/model identity used only for lane classification."""

    if config is None:
        return None
    return config.base_url.rstrip("/").lower(), config.model


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
    if (
        not isinstance(current_question["contract_name"], str)
        or not current_question["contract_name"]
    ):
        return False
    if not isinstance(current_question["facts"], Mapping):
        return False
    interludes = current_question["interludes"]
    return isinstance(interludes, list) and all(
        isinstance(row, Mapping) for row in interludes
    )


_REPAIR_STAGE_PATTERN = re.compile(
    r"^(?P<base>.+)\.repair(?P<attempt>[1-9][0-9]*)$"
)


def _call_stage_name(call: Mapping[str, Any]) -> str | None:
    """Return one captured call's stage telemetry when it is well-shaped."""

    config = call.get("config")
    if not isinstance(config, Mapping):
        return None
    stage_name = config.get("stage_name")
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
    previous_base = previous_match.group("base")
    previous_attempt = int(previous_match.group("attempt"))
    return (
        current_base == previous_base
        and current_attempt == previous_attempt + 1
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
            if (previous_stage, current_stage) in pairwise(
                registered_stages
            ):
                return True
    return False


def _message_snapshot(message: BaseMessage) -> dict[str, Any]:
    """Project one request message into exact structural performance evidence."""

    content = str(getattr(message, "content", ""))
    snapshot = {
        "message_type": type(message).__name__,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content_chars": len(content),
        "explicit_reanchor": _is_explicit_reanchor_packet(content),
    }
    return snapshot


def _has_nonempty_l1_residue(messages: Sequence[BaseMessage]) -> bool:
    """Return whether the last request packet carries an admitted L1 residue."""

    if not messages:
        return False
    content = str(getattr(messages[-1], "content", ""))
    try:
        packet = json.loads(content)
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
    if not isinstance(payload, Mapping):
        return False
    residue = payload.get("l1_residue")
    return isinstance(residue, Mapping) and bool(residue)


class PerformanceCapturingInvoker:
    """Measure real provider calls while preserving lane and prefix evidence."""

    def __init__(
        self,
        delegate: Any,
        *,
        primary_config: LLMCallConfig,
        sidecar_config: LLMCallConfig | None,
    ) -> None:
        self._delegate = delegate
        self._primary_identity = _route_identity(primary_config)
        self._sidecar_identity = _route_identity(sidecar_config)
        self._lock = threading.Lock()
        self._origin_ns = time.perf_counter_ns()
        self._next_sequence = 1
        self._active = {"primary": 0, "sidecar": 0, "unknown": 0}
        self._maximum = {"primary": 0, "sidecar": 0, "unknown": 0}
        self.calls: list[dict[str, Any]] = []

    def reset_capture(self) -> None:
        """Clear completed cold-call metrics while retaining invoker identity."""

        with self._lock:
            if any(self._active.values()):
                raise RuntimeError("cannot reset performance capture in flight")
            self._origin_ns = time.perf_counter_ns()
            self._next_sequence = 1
            self._maximum = {"primary": 0, "sidecar": 0, "unknown": 0}
            self.calls = []

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        """Measure one asynchronous provider request."""

        record = self._begin_call(messages, config=config, invocation="async")
        try:
            response = await self._delegate.ainvoke(messages, config=config)
        except Exception as exc:
            self._finish_call(record, exception=exc)
            raise
        self._finish_call(record, response=response)
        return response

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        """Measure one synchronous provider request."""

        record = self._begin_call(messages, config=config, invocation="sync")
        try:
            response = self._delegate.invoke(messages, config=config)
        except Exception as exc:
            self._finish_call(record, exception=exc)
            raise
        self._finish_call(record, response=response)
        return response

    def _lane_for_config(self, config: LLMCallConfig) -> str:
        """Classify one request by its configured endpoint/model identity."""

        identity = _route_identity(config)
        if identity == self._primary_identity:
            return "primary"
        if self._sidecar_identity is not None and identity == self._sidecar_identity:
            return "sidecar"
        return "unknown"

    def _begin_call(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
        invocation: str,
    ) -> dict[str, Any]:
        """Append one request record and increment its live-lane counter."""

        started_ns = time.perf_counter_ns()
        lane = self._lane_for_config(config)
        snapshots = [_message_snapshot(message) for message in messages]
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            self._active[lane] += 1
            self._maximum[lane] = max(
                self._maximum[lane],
                self._active[lane],
            )
            record: dict[str, Any] = {
                "sequence": sequence,
                "invocation": invocation,
                "lane": lane,
                "config": _config_record(config),
                "started_ms": (started_ns - self._origin_ns) / 1_000_000,
                "messages": snapshots,
                "serialized_request_chars": sum(
                    row["content_chars"] for row in snapshots
                ),
                "l1_residue_attached": _has_nonempty_l1_residue(messages),
                "_started_ns": started_ns,
            }
            self.calls.append(record)
        return record

    def _finish_call(
        self,
        record: dict[str, Any],
        *,
        response: LLMResponse | None = None,
        exception: Exception | None = None,
    ) -> None:
        """Finish one request record and decrement its live-lane counter."""

        ended_ns = time.perf_counter_ns()
        lane = record["lane"]
        with self._lock:
            record["ended_ms"] = (ended_ns - self._origin_ns) / 1_000_000
            record["wall_time_ms"] = (
                ended_ns - record.pop("_started_ns")
            ) / 1_000_000
            if response is not None:
                record["usage"] = dict(response.usage)
                record["response_chars"] = len(response.content)
                record["response_sha256"] = hashlib.sha256(
                    response.content.encode("utf-8")
                ).hexdigest()
            if exception is not None:
                record["exception"] = {
                    "type": type(exception).__name__,
                    "message": str(exception),
                }
            self._active[lane] -= 1

    @property
    def maximum_concurrency(self) -> dict[str, int]:
        """Return the maximum observed concurrency by lane."""

        with self._lock:
            maximum = dict(self._maximum)
        return maximum


def _build_v2_services(
    base_services: CognitionChainServicesV3,
    llm: PerformanceCapturingInvoker,
) -> CognitionCoreServicesV2:
    """Bind all V2 semantic owners to the accepted V3 primary route."""

    configs: dict[str, LLMCallConfig] = {}
    for field_name in _V2_CONFIG_FIELDS:
        owner_name = field_name.removesuffix("_config")
        configs[field_name] = replace(
            base_services.chain_lane,
            stage_name=f"cognition_core_v2.{owner_name}",
            route_name=f"PERF_V2_{owner_name.upper()}",
            thinking=LLMThinkingConfig(enabled=False),
        )
    services = CognitionCoreServicesV2(llm=llm, **configs)
    return services


def _base_v3_services() -> CognitionChainServicesV3:
    """Build and validate the selected V3 service family for live evidence."""

    if COGNITION_CORE_ENGINE != "v3":
        raise RuntimeError("Gate 7 performance capture requires the V3 engine")
    services = build_cognition_core_services()
    if not isinstance(services, CognitionChainServicesV3):
        raise TypeError("selected cognition services are not V3 services")
    return services


def _performance_services(
    base_services: CognitionChainServicesV3,
    engine: EngineName,
    *,
    subconscious_enabled: bool = False,
) -> tuple[
    CognitionCoreServicesV2 | CognitionChainServicesV3,
    PerformanceCapturingInvoker,
]:
    """Create one engine service bundle with an isolated timing capture."""

    capture = PerformanceCapturingInvoker(
        base_services.llm,
        primary_config=base_services.chain_lane,
        sidecar_config=base_services.sidecar_lane,
    )
    if engine == "v2":
        services = _build_v2_services(base_services, capture)
        return services, capture
    services = replace(
        base_services,
        llm=capture,
        subconscious_enabled=subconscious_enabled,
    )
    return services, capture


def _prefix_evidence(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove exact prefixes or the two sealed continuation transitions."""

    primary_calls = [call for call in calls if call["lane"] == "primary"]
    continuations: list[dict[str, Any]] = []
    for previous, current in pairwise(primary_calls):
        previous_messages = previous["messages"]
        current_messages = current["messages"]
        exact_prefix = current_messages[: len(previous_messages)] == (
            previous_messages
        )
        common_prefix_count = 0
        for previous_row, current_row in zip(
            previous_messages,
            current_messages,
        ):
            if previous_row != current_row:
                break
            common_prefix_count += 1
        repair_stage_telemetry = _is_sequential_repair_transition(
            previous,
            current,
        )
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
                _call_stage_name(previous),
                _call_stage_name(current),
            )
        )
        appraisal_recovery_tail_replacement = (
            human_tail_replacement
            and appraisal_recovery_stage_telemetry
        )
        repair_tail_replacement = (
            human_tail_replacement
            and repair_stage_telemetry
        )
        explicit_reanchor = (
            len(current_messages) == 2
            and len(previous_messages) >= 1
            and current_messages[0] == previous_messages[0]
            and current_messages[-1].get("message_type") == "HumanMessage"
            and current_messages[-1].get("explicit_reanchor") is True
        )
        if exact_prefix:
            classification = "exact"
            transition = None
            prefix_message_count = len(previous_messages)
        elif explicit_reanchor:
            classification = "permitted_transition"
            transition = "explicit_reanchor"
            prefix_message_count = 1
        elif appraisal_recovery_tail_replacement:
            classification = "permitted_transition"
            transition = "appraisal_recovery_tail_replacement"
            prefix_message_count = len(previous_messages) - 1
        elif repair_tail_replacement:
            classification = "permitted_transition"
            transition = "repair_tail_replacement"
            prefix_message_count = len(previous_messages) - 1
        else:
            classification = "invalid"
            transition = None
            prefix_message_count = common_prefix_count
        prefix_chars = sum(
            row["content_chars"]
            for row in current_messages[:prefix_message_count]
        )
        suffix_chars = current["serialized_request_chars"] - prefix_chars
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
            "appraisal_recovery_stage_telemetry": (
                appraisal_recovery_stage_telemetry
            ),
            "prefix_message_count": prefix_message_count,
            "prefix_chars": prefix_chars,
            "suffix_chars": suffix_chars,
        })
    all_valid = bool(continuations) and all(
        row["valid"] for row in continuations
    )
    proof = {
        "continuation_count": len(continuations),
        "all_exact": bool(continuations) and all(
            row["exact_prefix"] for row in continuations
        ),
        "all_continuations_valid": all_valid,
        "invalid_count": sum(row["invalid"] for row in continuations),
        "continuations": continuations,
    }
    return proof


def _occupied_wall_ms(calls: Sequence[Mapping[str, Any]]) -> float:
    """Return the union of provider-call intervals for interlude accounting."""

    intervals = sorted(
        (float(call["started_ms"]), float(call["ended_ms"]))
        for call in calls
    )
    if not intervals:
        return 0.0
    occupied = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        occupied += current_end - current_start
        current_start, current_end = start, end
    occupied += current_end - current_start
    return occupied


def _overlap_ms(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> float:
    """Return the intersection duration of two measured request intervals."""

    started = max(float(left["started_ms"]), float(right["started_ms"]))
    ended = min(float(left["ended_ms"]), float(right["ended_ms"]))
    return max(0.0, ended - started)


def _capture_summary(
    capture: PerformanceCapturingInvoker,
    *,
    full_run_wall_ms: float,
) -> dict[str, Any]:
    """Project one completed call capture into fixed performance metrics."""

    calls = sorted(capture.calls, key=lambda row: row["sequence"])
    primary_calls = [call for call in calls if call["lane"] == "primary"]
    sidecar_calls = [call for call in calls if call["lane"] == "sidecar"]
    unknown_calls = [call for call in calls if call["lane"] == "unknown"]
    overlap_total = sum(
        _overlap_ms(primary, sidecar)
        for primary in primary_calls
        for sidecar in sidecar_calls
    )
    l1_join = {
        "A1": sum(
            call["l1_residue_attached"]
            for call in primary_calls
            if call["config"]["stage_name"] in {"A1", "R.A1"}
        ),
        "G1a": sum(
            call["l1_residue_attached"]
            for call in primary_calls
            if call["config"]["stage_name"] in {"G1a", "R.G1a"}
        ),
    }
    first_primary_started_ms = (
        float(primary_calls[0]["started_ms"]) if primary_calls else None
    )
    first_sidecar = sidecar_calls[0] if sidecar_calls else None
    summary = {
        "full_run_wall_ms": full_run_wall_ms,
        "summed_primary_wall_ms": sum(
            float(call["wall_time_ms"]) for call in primary_calls
        ),
        "first_primary_wall_ms": (
            float(primary_calls[0]["wall_time_ms"])
            if primary_calls
            else None
        ),
        "first_primary_started_ms": first_primary_started_ms,
        "deterministic_interlude_ms": max(
            0.0,
            full_run_wall_ms - _occupied_wall_ms(calls),
        ),
        "summed_sidecar_wall_ms": sum(
            float(call["wall_time_ms"]) for call in sidecar_calls
        ),
        "primary_request_count": len(primary_calls),
        "sidecar_request_count": len(sidecar_calls),
        "unknown_request_count": len(unknown_calls),
        "maximum_concurrency": capture.maximum_concurrency,
        "foreign_primary_interleaves": len(unknown_calls),
        "prefix_evidence": _prefix_evidence(calls),
        "primary_sidecar_overlap_ms": overlap_total,
        "primary_started_while_sidecar_active": bool(
            first_sidecar is not None
            and first_primary_started_ms is not None
            and float(first_sidecar["started_ms"]) <= first_primary_started_ms
            < float(first_sidecar["ended_ms"])
        ),
        "l1_join": l1_join,
        "l1_dropped": bool(sidecar_calls) and sum(l1_join.values()) == 0,
        "calls": calls,
    }
    return summary


async def _execute_engine(
    engine: EngineName,
    payload: CognitionCoreInputV2,
    base_services: CognitionChainServicesV3,
    *,
    subconscious_enabled: bool = False,
) -> dict[str, Any]:
    """Run one real engine invocation and retain its complete timing record."""

    services, capture = _performance_services(
        base_services,
        engine,
        subconscious_enabled=subconscious_enabled,
    )
    started_at = time.perf_counter()
    output: Mapping[str, Any] | None = None
    error: dict[str, str] | None = None
    try:
        if engine == "v2":
            candidate = await run_cognition_v2(payload, services)
        else:
            candidate = await v3_facade.run_cognition(payload, services)
        output = validate_cognition_core_output(candidate)
    except Exception as exc:  # noqa: BLE001 - evidence records every failure
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    full_run_wall_ms = (time.perf_counter() - started_at) * 1000
    record = {
        "engine": engine,
        "semantic_result_available": output is not None,
        "contract_disposition": "passed" if output is not None else "failed",
        "output_sha256": canonical_sha256(output) if output is not None else None,
        "diagnostic_warnings": (
            list(output["diagnostics"]["warnings"])
            if output is not None
            else []
        ),
        "error": error,
        "metrics": _capture_summary(
            capture,
            full_run_wall_ms=full_run_wall_ms,
        ),
    }
    return record


async def _matched_pair(
    payload: CognitionCoreInputV2,
    base_services: CognitionChainServicesV3,
    *,
    pair_index: int,
) -> dict[str, Any]:
    """Run one alternating pair and retry only a no-result invalid pair once."""

    order: tuple[EngineName, EngineName] = (
        ("v2", "v3") if pair_index % 2 else ("v3", "v2")
    )
    attempts: list[dict[str, Any]] = []
    for attempt_index in (1, 2):
        results: dict[str, Any] = {}
        for engine in order:
            results[engine] = await _execute_engine(
                engine,
                deepcopy(payload),
                base_services,
            )
        attempt = {
            "attempt_index": attempt_index,
            "order": list(order),
            "results": results,
        }
        attempts.append(attempt)
        if all(
            result["contract_disposition"] == "passed"
            for result in results.values()
        ):
            return {
                "pair_index": pair_index,
                "eligible": True,
                "attempts": attempts,
                "accepted_attempt_index": attempt_index,
            }
        if any(
            result["semantic_result_available"]
            for result in results.values()
        ):
            break
    return {
        "pair_index": pair_index,
        "eligible": False,
        "attempts": attempts,
        "accepted_attempt_index": None,
    }


def _accepted_results(
    pairs: Sequence[Mapping[str, Any]],
    engine: EngineName,
) -> list[Mapping[str, Any]]:
    """Return the accepted engine result from every eligible pair."""

    results: list[Mapping[str, Any]] = []
    for pair in pairs:
        accepted_attempt_index = pair["accepted_attempt_index"]
        if not isinstance(accepted_attempt_index, int):
            continue
        attempt = pair["attempts"][accepted_attempt_index - 1]
        results.append(attempt["results"][engine])
    return results


def _median(values: Sequence[float]) -> float:
    """Return the exact ordinary median, including even central means."""

    if not values:
        raise ValueError("median requires at least one observation")
    return float(statistics.median(values))


def _p95(values: Sequence[float]) -> float:
    """Return the nearest-rank p95 required by the fixed protocol."""

    if not values:
        raise ValueError("p95 requires at least one observation")
    ordered = sorted(values)
    rank = math.ceil(0.95 * len(ordered))
    return float(ordered[rank - 1])


def _metric_values(
    results: Sequence[Mapping[str, Any]],
    metric_name: str,
) -> list[float]:
    """Return one required numeric metric from each accepted result."""

    values: list[float] = []
    for result in results:
        value = result["metrics"][metric_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"missing numeric performance metric {metric_name}")
        values.append(float(value))
    return values


def _ratio(numerator: float, denominator: float) -> float:
    """Return one performance ratio with an explicit positive denominator."""

    if denominator <= 0:
        raise ValueError("performance ratio denominator must be positive")
    return numerator / denominator


def _pair_aggregate(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate the shared V2/V3 matched performance aggregates."""

    v2_results = _accepted_results(pairs, "v2")
    v3_results = _accepted_results(pairs, "v3")
    if len(v2_results) != len(pairs) or len(v3_results) != len(pairs):
        return {
            "eligible_pair_count": len(v2_results),
            "required_pair_count": len(pairs),
            "all_pairs_eligible": False,
        }
    v2_primary = _metric_values(v2_results, "summed_primary_wall_ms")
    v3_primary = _metric_values(v3_results, "summed_primary_wall_ms")
    v2_full = _metric_values(v2_results, "full_run_wall_ms")
    v3_full = _metric_values(v3_results, "full_run_wall_ms")
    v2_primary_median = _median(v2_primary)
    v3_primary_median = _median(v3_primary)
    aggregate = {
        "eligible_pair_count": len(v2_results),
        "required_pair_count": len(pairs),
        "all_pairs_eligible": True,
        "v2_primary_median_ms": v2_primary_median,
        "v3_primary_median_ms": v3_primary_median,
        "primary_median_reduction": (
            v2_primary_median - v3_primary_median
        ) / v2_primary_median,
        "v2_full_median_ms": _median(v2_full),
        "v3_full_median_ms": _median(v3_full),
        "full_median_v2_ratio": _ratio(_median(v3_full), _median(v2_full)),
        "v2_full_p95_ms": _p95(v2_full),
        "v3_full_p95_ms": _p95(v3_full),
        "full_p95_v2_ratio": _ratio(_p95(v3_full), _p95(v2_full)),
        "v3_prefix_all_exact": all(
            result["metrics"]["prefix_evidence"]["all_continuations_valid"]
            for result in v3_results
        ),
        "v3_primary_max_in_flight": max(
            result["metrics"]["maximum_concurrency"]["primary"]
            for result in v3_results
        ),
        "v3_foreign_primary_interleaves": sum(
            result["metrics"]["foreign_primary_interleaves"]
            for result in v3_results
        ),
    }
    return aggregate


def _base_artifact(
    node_name: str,
    base_services: CognitionChainServicesV3,
) -> dict[str, Any]:
    """Build the common non-secret evidence header for one live node."""

    artifact = {
        "schema_version": PERFORMANCE_SCHEMA,
        "baseline_id": baseline_id_from_environment(),
        "performance_run_id": _performance_run_id(),
        "node_name": node_name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_kind": "sealed synthetic fixture",
        "environment_fingerprint": sanitized_v3_environment_fingerprint(
            base_services
        ),
        "protocol": {
            "non_streaming": True,
            "ttft_claimed": False,
            "outliers_removed": False,
            "median_rule": "mean_of_two_central_values_when_even",
            "p95_rule": "nearest_rank_ceil_0.95_n",
            "v2_bound_to_v3_primary": True,
            "v2_thinking_enabled": False,
        },
    }
    return artifact


def _assert_common_pair_thresholds(aggregate: Mapping[str, Any]) -> None:
    """Assert the fixed full-run, lane, and prefix performance gates."""

    assert aggregate["all_pairs_eligible"] is True
    assert aggregate["v3_prefix_all_exact"] is True
    assert aggregate["v3_primary_max_in_flight"] == 1
    assert aggregate["v3_foreign_primary_interleaves"] == 0
    assert aggregate["full_median_v2_ratio"] <= FULL_RUN_MEDIAN_RATIO_MAXIMUM
    assert aggregate["full_p95_v2_ratio"] <= FULL_RUN_P95_RATIO_MAXIMUM


async def test_live_performance_cold_full_turn() -> None:
    """Measure five isolated cold matched V2/V3 full turns."""

    node_name = "test_live_performance_cold_full_turn"
    base_services = _base_v3_services()
    case_ids = [
        "ordinary_neutral_response",
        "event_agency_and_moral_chain",
        "relationship_reciprocity",
        "goal_completion_terminalization",
        "epistemic_comparison",
    ]
    pairs = []
    for pair_index, case_id in enumerate(case_ids, start=1):
        pair = await _matched_pair(
            render_case_input(find_case_row(case_id)),
            base_services,
            pair_index=pair_index,
        )
        pair["case_id"] = case_id
        pairs.append(pair)
    aggregate = _pair_aggregate(pairs)
    if aggregate["all_pairs_eligible"]:
        v2_first = _metric_values(
            _accepted_results(pairs, "v2"),
            "first_primary_wall_ms",
        )
        v3_first = _metric_values(
            _accepted_results(pairs, "v3"),
            "first_primary_wall_ms",
        )
        aggregate["cold_first_primary_v2_ratio"] = _ratio(
            _median(v3_first),
            _median(v2_first),
        )
    artifact = _base_artifact(node_name, base_services)
    artifact.update({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "trial_count": COLD_TRIAL_COUNT,
        "pairs": pairs,
        "aggregate": aggregate,
    })
    _seal_performance_artifact(node_name, artifact)
    _assert_common_pair_thresholds(aggregate)
    assert (
        aggregate["cold_first_primary_v2_ratio"]
        <= COLD_FIRST_PRIMARY_RATIO_MAXIMUM
    )


async def _run_warm_pair_block(
    *,
    node_name: str,
    payloads: Sequence[CognitionCoreInputV2],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run excluded engine warm-ups followed by twenty matched live pairs."""

    if len(payloads) != WARM_TRIAL_COUNT:
        raise ValueError("warm performance blocks require twenty inputs")
    base_services = _base_v3_services()
    warmups = {
        "v2": await _execute_engine("v2", deepcopy(payloads[0]), base_services),
        "v3": await _execute_engine("v3", deepcopy(payloads[0]), base_services),
    }
    warmups_passed = all(
        result["contract_disposition"] == "passed"
        for result in warmups.values()
    )
    pairs = []
    for pair_index, payload in enumerate(payloads, start=1):
        pairs.append(
            await _matched_pair(
                deepcopy(payload),
                base_services,
                pair_index=pair_index,
            )
        )
    aggregate = _pair_aggregate(pairs)
    aggregate["excluded_warmups_passed"] = warmups_passed
    artifact = _base_artifact(node_name, base_services)
    artifact.update({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "excluded_warmups": warmups,
        "trial_count": WARM_TRIAL_COUNT,
        "pairs": pairs,
        "aggregate": aggregate,
    })
    return artifact, aggregate


async def test_live_performance_warm_exact_repeat() -> None:
    """Measure twenty exact repeats after one excluded warm-up per engine."""

    node_name = "test_live_performance_warm_exact_repeat"
    payload = render_case_input(find_case_row("ordinary_neutral_response"))
    artifact, aggregate = await _run_warm_pair_block(
        node_name=node_name,
        payloads=[deepcopy(payload) for _ in range(WARM_TRIAL_COUNT)],
    )
    _seal_performance_artifact(node_name, artifact)
    _assert_common_pair_thresholds(aggregate)
    assert aggregate["excluded_warmups_passed"] is True
    assert aggregate["primary_median_reduction"] >= PRIMARY_REDUCTION_MINIMUM


async def test_live_performance_warm_changed_tail() -> None:
    """Measure twenty frozen case-tail changes after excluded warm-ups."""

    node_name = "test_live_performance_warm_changed_tail"
    manifest = load_live_case_manifest()
    case_rows = manifest["cases"][:WARM_TRIAL_COUNT]
    payloads = [render_case_input(row) for row in case_rows]
    artifact, aggregate = await _run_warm_pair_block(
        node_name=node_name,
        payloads=payloads,
    )
    artifact["case_ids"] = [row["case_id"] for row in case_rows]
    _seal_performance_artifact(node_name, artifact)
    _assert_common_pair_thresholds(aggregate)
    assert aggregate["excluded_warmups_passed"] is True
    assert aggregate["primary_median_reduction"] >= PRIMARY_REDUCTION_MINIMUM


def _resolver_cold_input() -> CognitionCoreInputV2:
    """Derive the exact cycle-zero prefix of the sealed resolver case."""

    payload = render_case_input(
        find_case_row("resolver_observation_continuation")
    )
    evidence = payload["evidence"]
    if len(evidence) < 2:
        raise ValueError("resolver performance case has no appended observation")
    appended = evidence[-1]
    if appended["evidence_ref"]["source_kind"] != "resolver_observation":
        raise ValueError("resolver performance tail is not an observation")
    payload["evidence"] = evidence[:-1]
    payload.pop("resolver_cycle_index", None)
    payload.pop("current_turn_relational_willingness", None)
    validated = validate_cognition_core_input(payload)
    return validated


def _resolver_continuation_input(
    cold_payload: CognitionCoreInputV2,
    cold_output: Mapping[str, Any],
) -> CognitionCoreInputV2:
    """Build one canonical cycle-one input from the sealed observation tail."""

    payload = deepcopy(cold_payload)
    payload["mutable_state"] = deepcopy(
        cold_output["state_update"]["replacement_state"]
    )
    sealed = render_case_input(
        find_case_row("resolver_observation_continuation")
    )
    payload["evidence"].append(deepcopy(sealed["evidence"][-1]))
    payload["resolver_cycle_index"] = 1
    relational_willingness = cold_output.get("relational_willingness")
    if not isinstance(relational_willingness, Mapping):
        raise TypeError("cold output must carry relational willingness")
    payload["current_turn_relational_willingness"] = {
        "schema_version": "current_turn_relational_willingness.v2",
        "episode_id": payload["episode"]["episode_id"],
        "branch_id": "ordinary_response",
        "decision": deepcopy(dict(relational_willingness)),
    }
    validated = validate_cognition_core_input(payload)
    return validated


async def _resolver_engine_trial(
    engine: EngineName,
    base_services: CognitionChainServicesV3,
) -> dict[str, Any]:
    """Measure one two-cycle engine trial and isolate continuation timing."""

    if engine == "v3":
        v3_facade._CHAIN_SESSION_REGISTRY = ChainSessionRegistry()
    services, capture = _performance_services(base_services, engine)
    cold_payload = _resolver_cold_input()
    if engine == "v2":
        cold_candidate = await run_cognition_v2(cold_payload, services)
    else:
        cold_candidate = await v3_facade.run_cognition(cold_payload, services)
    cold_output = validate_cognition_core_output(cold_candidate)
    continuation = _resolver_continuation_input(cold_payload, cold_output)
    capture.reset_capture()
    started_at = time.perf_counter()
    if engine == "v2":
        tail_candidate = await run_cognition_v2(continuation, services)
    else:
        tail_candidate = await v3_facade.run_cognition(continuation, services)
    tail_output = validate_cognition_core_output(tail_candidate)
    full_run_wall_ms = (time.perf_counter() - started_at) * 1000
    record = {
        "engine": engine,
        "semantic_result_available": True,
        "contract_disposition": "passed",
        "cold_output_sha256": canonical_sha256(cold_output),
        "output_sha256": canonical_sha256(tail_output),
        "diagnostic_warnings": list(tail_output["diagnostics"]["warnings"]),
        "metrics": _capture_summary(
            capture,
            full_run_wall_ms=full_run_wall_ms,
        ),
    }
    return record


async def _v3_recurrence_cold_rebuild(
    base_services: CognitionChainServicesV3,
) -> dict[str, Any]:
    """Measure the same cycle-one payload after forcing a V3 session miss."""

    v3_facade._CHAIN_SESSION_REGISTRY = ChainSessionRegistry()
    cold_services, _ = _performance_services(base_services, "v3")
    cold_payload = _resolver_cold_input()
    cold_candidate = await v3_facade.run_cognition(cold_payload, cold_services)
    cold_output = validate_cognition_core_output(cold_candidate)
    continuation = _resolver_continuation_input(cold_payload, cold_output)
    v3_facade._CHAIN_SESSION_REGISTRY = ChainSessionRegistry()
    return await _execute_engine("v3", continuation, base_services)


async def test_live_performance_resolver_continuation() -> None:
    """Measure ten matched two-cycle resolver trials and V3 cold rebuilds."""

    node_name = "test_live_performance_resolver_continuation"
    base_services = _base_v3_services()
    trials = []
    for pair_index in range(1, RESOLVER_TWO_CYCLE_PAIR_COUNT + 1):
        order: tuple[EngineName, EngineName] = (
            ("v2", "v3") if pair_index % 2 else ("v3", "v2")
        )
        results: dict[str, Any] = {}
        for engine in order:
            results[engine] = await _resolver_engine_trial(engine, base_services)
        cold_rebuild = await _v3_recurrence_cold_rebuild(base_services)
        trials.append({
            "pair_index": pair_index,
            "order": list(order),
            "results": results,
            "v3_cold_rebuild": cold_rebuild,
        })
    v2_tail = [
        trial["results"]["v2"]["metrics"]["full_run_wall_ms"]
        for trial in trials
    ]
    v3_tail = [
        trial["results"]["v3"]["metrics"]["full_run_wall_ms"]
        for trial in trials
    ]
    v3_cold = [
        trial["v3_cold_rebuild"]["metrics"]["full_run_wall_ms"]
        for trial in trials
    ]
    aggregate = {
        "two_cycle_pair_count": len(trials),
        "continuation_invocation_count": len(trials) * 2,
        "v2_tail_median_ms": _median(v2_tail),
        "v3_tail_median_ms": _median(v3_tail),
        "v3_cold_rebuild_median_ms": _median(v3_cold),
        "resolver_tail_v2_ratio": _ratio(
            _median(v3_tail),
            _median(v2_tail),
        ),
        "resolver_tail_v3_cold_rebuild_ratio": _ratio(
            _median(v3_tail),
            _median(v3_cold),
        ),
        "all_v3_sessions_reattached": all(
            "session_reattached" in trial["results"]["v3"][
                "diagnostic_warnings"
            ]
            for trial in trials
        ),
        "v3_prefix_all_exact": all(
            trial["results"]["v3"]["metrics"]["prefix_evidence"][
                "all_continuations_valid"
            ]
            for trial in trials
        ),
        "v3_primary_max_in_flight": max(
            trial["results"]["v3"]["metrics"]["maximum_concurrency"][
                "primary"
            ]
            for trial in trials
        ),
    }
    artifact = _base_artifact(node_name, base_services)
    artifact.update({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "two_cycle_pair_count": RESOLVER_TWO_CYCLE_PAIR_COUNT,
        "continuation_invocation_count": (
            RESOLVER_TWO_CYCLE_PAIR_COUNT * 2
        ),
        "trials": trials,
        "aggregate": aggregate,
    })
    _seal_performance_artifact(node_name, artifact)
    assert aggregate["all_v3_sessions_reattached"] is True
    assert aggregate["v3_prefix_all_exact"] is True
    assert aggregate["v3_primary_max_in_flight"] == 1
    assert aggregate["resolver_tail_v2_ratio"] <= RESOLVER_V2_RATIO_MAXIMUM
    assert (
        aggregate["resolver_tail_v3_cold_rebuild_ratio"]
        <= RESOLVER_COLD_REBUILD_RATIO_MAXIMUM
    )


async def test_live_performance_sidecar_overlap() -> None:
    """Measure twenty L1-enabled turns on the one serialized sidecar lane."""

    node_name = "test_live_performance_sidecar_overlap"
    base_services = _base_v3_services()
    if base_services.sidecar_lane is None:
        raise RuntimeError("sidecar overlap evidence requires a sidecar route")
    payload = render_case_input(find_case_row("ordinary_neutral_response"))
    trials = []
    for trial_index in range(1, SIDECAR_TRIAL_COUNT + 1):
        result = await _execute_engine(
            "v3",
            deepcopy(payload),
            base_services,
            subconscious_enabled=True,
        )
        result["trial_index"] = trial_index
        trials.append(result)
    aggregate = {
        "trial_count": len(trials),
        "all_contracts_passed": all(
            trial["contract_disposition"] == "passed" for trial in trials
        ),
        "primary_max_in_flight": max(
            trial["metrics"]["maximum_concurrency"]["primary"]
            for trial in trials
        ),
        "sidecar_max_in_flight": max(
            trial["metrics"]["maximum_concurrency"]["sidecar"]
            for trial in trials
        ),
        "foreign_primary_interleaves": sum(
            trial["metrics"]["foreign_primary_interleaves"]
            for trial in trials
        ),
        "overlap_trial_count": sum(
            trial["metrics"]["primary_sidecar_overlap_ms"] > 0
            for trial in trials
        ),
        "primary_started_while_sidecar_active_count": sum(
            trial["metrics"]["primary_started_while_sidecar_active"]
            for trial in trials
        ),
        "l1_join_A1_count": sum(
            trial["metrics"]["l1_join"]["A1"] for trial in trials
        ),
        "l1_join_G1a_count": sum(
            trial["metrics"]["l1_join"]["G1a"] for trial in trials
        ),
        "l1_dropped_count": sum(
            trial["metrics"]["l1_dropped"] for trial in trials
        ),
    }
    artifact = _base_artifact(node_name, base_services)
    artifact.update({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "trial_count": SIDECAR_TRIAL_COUNT,
        "trials": trials,
        "aggregate": aggregate,
    })
    _seal_performance_artifact(node_name, artifact)
    assert aggregate["all_contracts_passed"] is True
    assert aggregate["primary_max_in_flight"] == 1
    assert aggregate["sidecar_max_in_flight"] == 1
    assert aggregate["foreign_primary_interleaves"] == 0
    assert aggregate["overlap_trial_count"] == SIDECAR_TRIAL_COUNT
    assert (
        aggregate["primary_started_while_sidecar_active_count"]
        == SIDECAR_TRIAL_COUNT
    )
