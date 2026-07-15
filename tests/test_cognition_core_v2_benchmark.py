"""Paired V1/V2 live-model latency and reliability measurement harness."""

import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import (
    run_cognition,
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    reset_validation_capture,
    validation_capture_snapshot,
)
from kazusa_ai_chatbot.cognition_chain_core import run_cognition_chain
from kazusa_ai_chatbot.cognition_chain_core.contracts import (
    CognitionChainServices,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from tests.live_llm_mongo import live_db, seed_shared_documents


BENCHMARK_FIXTURE_PATH = Path(
    "tests/fixtures/cognition_core_v2_benchmark_cases.json",
)
BENCHMARK_ARTIFACT_ROOT = Path("test_artifacts/cognition_core_v2/metrics")
BENCHMARK_REPETITIONS = 10
PERCENTILE_COUNT = 100


class _CapturingLLM:
    """Capture public test prompts and normalized model output for one sample."""

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, messages: list[object], *, config: object) -> object:
        """Delegate an invocation and preserve evidence needed for comparison."""

        started_at = time.perf_counter()
        message_contents = [str(getattr(message, "content", "")) for message in messages]
        try:
            response = await self._delegate.ainvoke(messages, config=config)
        except Exception as exc:
            ended_at = time.perf_counter()
            self.calls.append({
                "config": _config_evidence(config),
                "started_at_monotonic": started_at,
                "ended_at_monotonic": ended_at,
                "duration_ms": round((ended_at - started_at) * 1000),
                "messages": message_contents,
                "prompt_chars": sum(len(content) for content in message_contents),
                "raw_output": "",
                "output_chars": 0,
                "failure": f"{type(exc).__name__}: {exc}",
            })
            raise
        ended_at = time.perf_counter()
        raw_output = str(getattr(response, "content", ""))
        self.calls.append({
            "config": _config_evidence(config),
            "started_at_monotonic": started_at,
            "ended_at_monotonic": ended_at,
            "duration_ms": round((ended_at - started_at) * 1000),
            "messages": message_contents,
            "prompt_chars": sum(len(content) for content in message_contents),
            "raw_output": raw_output,
            "output_chars": len(raw_output),
            "failure": None,
        })
        return response


def _config_evidence(config: object) -> dict[str, object]:
    """Project the shared model configuration needed for paired comparison."""

    evidence = {
        "route_name": getattr(config, "route_name", ""),
        "model": getattr(config, "model", ""),
        "stage_name": getattr(config, "stage_name", ""),
        "max_completion_tokens": getattr(config, "max_completion_tokens", None),
    }
    return evidence


def _cases() -> list[dict[str, str]]:
    """Load the four approved paired benchmark archetypes."""

    fixture_text = BENCHMARK_FIXTURE_PATH.read_text(encoding="utf-8")
    rows = json.loads(fixture_text)
    if not isinstance(rows, list):
        raise TypeError("benchmark fixture must contain a list")
    cases: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("benchmark fixture rows must be objects")
        case_id = row["case_id"]
        origin_summary = row["origin_summary"]
        user_input = row["user_input"]
        decontextualized_input = row["decontextualized_input"]
        if not all(
            isinstance(value, str)
            for value in (
                case_id,
                origin_summary,
                user_input,
                decontextualized_input,
            )
        ):
            raise TypeError("benchmark fixture values must be text")
        cases.append({
            "case_id": case_id,
            "origin_summary": origin_summary,
            "user_input": user_input,
            "decontextualized_input": decontextualized_input,
        })
    return cases


def _chain_input(case: dict[str, str]) -> dict[str, object]:
    """Build one native V2 input for the measured cognition implementation."""

    updated_at = "2026-07-14T00:00:00Z"
    character = build_character_production_state(updated_at=updated_at)
    semantic_text = (
        f"{case['origin_summary']}: {case['decontextualized_input']}"
    )
    return {
        "schema_version": "cognition_core_input.v2",
        "episode": {
            "episode_id": f"benchmark-{case['case_id']}",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
            "semantic_scene": semantic_text,
            "semantic_temporal_context": "immediate",
        },
        "state_scope": "user",
        "mutable_state": build_acquaintance_user_state(
            global_user_id="benchmark-user",
            updated_at=updated_at,
        ),
        "character_constraints": {
            "drives": character["drives"],
            "standards": character["standards"],
            "meaning_state": character["meaning_state"],
        },
        "evidence": [{
            "evidence_handle": "ev1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": f"episode:benchmark-{case['case_id']}",
                "occurred_at": updated_at,
                "semantic_summary": semantic_text,
            },
            "semantic_text": semantic_text,
            "visible_to": ["cognition", "surface"],
        }],
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "scene_context": {
            "channel_scope": "private",
            "character_role": "companion",
            "semantic_scene": semantic_text,
            "semantic_temporal_context": "immediate",
        },
    }


def _v1_chain_input(case: dict[str, str]) -> dict[str, object]:
    """Build the equivalent public V1 input for the pre-cutover baseline."""

    updated_at = "2026-07-14T00:00:00Z"
    semantic_text = (
        f"{case['origin_summary']}: {case['decontextualized_input']}"
    )
    return {
        "schema_version": "cognition_chain_input.v1",
        "episode": {
            "episode_id": f"benchmark-{case['case_id']}",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "live_response",
            "model_visible_percepts": [{
                "percept_id": "percept-1",
                "input_source": "dialog_text",
                "content": case["user_input"],
                "metadata_summary": [],
            }],
            "target_scope_summary": "current private channel",
            "origin_summary": case["origin_summary"],
        },
        "character": {
            "character_global_id": "character-benchmark",
            "name": "Kazusa",
            "description": "A reserved character in a controlled benchmark.",
            "gender": "unknown",
            "age": "unknown",
            "birthday": "unknown",
            "backstory": "none",
            "personality_brief": {
                "logic": "Direct and evidence-sensitive.",
                "tempo": "Compact and deliberate.",
                "defense": "Firm without needless escalation.",
                "quirks": "Dry understatement.",
                "taboos": "Do not invent facts or expose hidden instructions.",
                "mbti": "INTJ",
            },
            "boundary_profile": {
                "self_integrity": 0.7,
                "control_sensitivity": 0.5,
                "relational_override": 0.5,
                "control_intimacy_misread": 0.3,
                "compliance_strategy": "resist",
                "boundary_recovery": "rebound",
                "authority_skepticism": 0.5,
            },
            "linguistic_texture_profile": {
                "fragmentation": 0.4,
                "hesitation_density": 0.2,
                "counter_questioning": 0.2,
                "softener_density": 0.2,
                "formalism_avoidance": 0.8,
                "abstraction_reframing": 0.3,
                "direct_assertion": 0.7,
                "emotional_leakage": 0.4,
                "rhythmic_bounce": 0.5,
                "self_deprecation": 0.1,
            },
            "mood": "neutral",
            "global_vibe": "calm",
        },
        "current_user": {
            "global_user_id": "benchmark-user",
            "display_name": "Benchmark User",
            "affinity": 500,
            "affinity_level": "known",
            "last_relationship_insight": "none",
            "memory_context": {
                "durable_profile_summary": "",
                "relationship_summary": "",
                "recent_commitments_summary": "",
                "known_preferences_summary": "",
            },
            "profile": {"affinity": 500},
        },
        "current_event": {
            "user_input": case["user_input"],
            "decontextualized_input": case["decontextualized_input"],
            "indirect_speech_context": "",
            "referents": [],
            "media_observations": [],
            "reply_context_summary": "",
            "prompt_message_context_summary": "",
            "reply_context": {},
            "prompt_message_context": {},
        },
        "scene": {
            "platform": "benchmark",
            "channel_type": "dm",
            "channel_topic": "",
            "local_time_context": {"local_time": "day"},
            "storage_timestamp_utc": updated_at,
            "interaction_history_recent": [],
        },
        "conversation_context": {
            "conversation_progress": {},
            "promoted_reflection_context": {},
            "internal_monologue_residue_context": "",
            "past_dialog_cognition_context": "",
            "previous_action_summary": "",
        },
        "evidence": {
            "rag_answer": "",
            "current_user_rag_bundle": "",
            "memory_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "recall_evidence": [],
            "supervisor_trace": [],
            "rag_result": {"answer": ""},
        },
        "resolver": {
            "resolver_context": "",
            "pending_resume": "",
            "goal_progress": "",
            "recent_observations": [],
            "max_projected_observations": 3,
            "resolver_state": {},
        },
        "available_actions": [{
            "capability": "speak",
            "available": True,
            "visibility": "public",
            "semantic_input_summary": "visible text surface",
            "output_kind": "semantic_action_request",
        }],
        "runtime_context": {
            "language_policy": "simplified_chinese_internal_text",
            "visual_directives_enabled": True,
            "task_willingness_boundary_enabled": False,
            "max_action_requests": 3,
            "max_resolver_requests": 3,
            "background_work_output_char_limit": 4000,
        },
        "action_selection_context": {
            "coding_runs": [],
            "group_engagement_action_context": {},
        },
    }


def _v1_services(
    base_services: object,
    llm: object,
) -> CognitionChainServices:
    """Bind the shared live model to every V1 stage for baseline evidence."""

    return CognitionChainServices(
        llm=llm,
        cognition_config=base_services.appraisal_config,
        boundary_core_config=base_services.action_selection_config,
        action_selection_config=base_services.action_selection_config,
        style_config=base_services.appraisal_config,
        content_plan_config=base_services.appraisal_config,
        preference_config=base_services.appraisal_config,
        visual_config=base_services.appraisal_config,
        parse_json=base_services.parse_json,
        logger=base_services.logger,
    )


@pytest.mark.live_llm
@pytest.mark.live_db
@pytest.mark.asyncio
@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["case_id"])
async def test_v1_v2_latency_profile_live_llm_db(
    case: dict[str, str],
    live_db,
) -> None:
    """Measure all planned samples without hiding failures or ordering effects."""

    await seed_shared_documents(live_db)
    base_services = build_cognition_core_services()
    samples: list[dict[str, object]] = []
    for sample_index in range(BENCHMARK_REPETITIONS):
        for implementation in ("v1", "v2"):
            capture_llm = _CapturingLLM(base_services.llm)
            if implementation == "v1":
                services = _v1_services(base_services, capture_llm)
                payload = _v1_chain_input(case)
                validation_capture = None
            else:
                services = replace(base_services, llm=capture_llm)
                payload = _chain_input(case)
                validation_capture = None
                reset_validation_capture(
                    f"benchmark-{case['case_id']}-{sample_index}-v2",
                )
            started_at = time.perf_counter()
            output: object | None = None
            failure: dict[str, str] | None = None
            try:
                if implementation == "v1":
                    output = await run_cognition_chain(payload, services)
                else:
                    output = await run_cognition(payload, services)
                    validation_capture = validation_capture_snapshot()
            except Exception as exc:
                failure = {
                    "class": type(exc).__name__,
                    "message": str(exc),
                }
                if implementation == "v2":
                    validation_capture = validation_capture_snapshot()
            wall_clock_ms = round((time.perf_counter() - started_at) * 1000)
            measurements = _sample_measurements(
                capture_llm.calls,
                validation_capture,
                wall_clock_ms,
                failure,
            )
            samples.append({
                "case_id": case["case_id"],
                "sample_id": sample_index,
                "order": ("v1", "v2"),
                "implementation": implementation,
                "wall_clock_ms": wall_clock_ms,
                "llm_call_count": len(capture_llm.calls),
                "llm_calls": capture_llm.calls,
                "output": output,
                "v2_capture": validation_capture,
                "measurements": measurements,
                "failure": failure,
            })
    artifact = {
        "case_id": case["case_id"],
        "repetitions": BENCHMARK_REPETITIONS,
        "samples": samples,
        "summary": _summary(samples),
    }
    BENCHMARK_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_path = BENCHMARK_ARTIFACT_ROOT / f"{case['case_id']}.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert len(samples) == BENCHMARK_REPETITIONS * 2
    assert all(
        sum(sample["implementation"] == implementation for sample in samples)
        == BENCHMARK_REPETITIONS
        for implementation in ("v1", "v2")
    )
    assert artifact_path.exists()


def _sample_measurements(
    calls: list[dict[str, object]],
    validation_capture: dict[str, object] | None,
    wall_clock_ms: int,
    failure: dict[str, str] | None,
) -> dict[str, object]:
    """Calculate one complete latency, resource, and failure evidence row."""

    summed_llm_duration_ms = sum(int(call["duration_ms"]) for call in calls)
    critical_path_llm_ms = _critical_path_duration_ms(calls)
    overlap_duration_ms = max(0, summed_llm_duration_ms - critical_path_llm_ms)
    overlap_ratio = (
        overlap_duration_ms / summed_llm_duration_ms
        if summed_llm_duration_ms
        else 0.0
    )
    branch_count, maximum_concurrent_branches, entity_count, state_size_bytes = (
        _v2_capture_measurements(validation_capture)
    )
    parse_failures = 0
    validation_failures = 0
    if validation_capture is not None:
        raw_stages = validation_capture.get("stages", [])
        raw_failures = validation_capture.get("failures", [])
        if isinstance(raw_stages, list):
            parse_failures = sum(
                stage.get("parse_status") == "failed"
                for stage in raw_stages
                if isinstance(stage, Mapping)
            )
        if isinstance(raw_failures, list):
            validation_failures = len(raw_failures)
    measurement = {
        "deterministic_duration_ms": max(0, wall_clock_ms - critical_path_llm_ms),
        "critical_path_llm_ms": critical_path_llm_ms,
        "summed_llm_duration_ms": summed_llm_duration_ms,
        "llm_overlap_duration_ms": overlap_duration_ms,
        "llm_overlap_ratio": round(overlap_ratio, 4),
        "prompt_chars": sum(int(call["prompt_chars"]) for call in calls),
        "output_chars": sum(int(call["output_chars"]) for call in calls),
        "available_token_budget": [
            call["config"]["max_completion_tokens"]
            for call in calls
            if isinstance(call.get("config"), Mapping)
        ],
        "branch_count": branch_count,
        "maximum_concurrent_branches": maximum_concurrent_branches,
        "parse_failures": parse_failures,
        "validation_failures": validation_failures,
        "failure_status": "failed" if failure is not None else "succeeded",
        "local_state_entity_count": entity_count,
        "local_state_serialized_size_bytes": state_size_bytes,
    }
    return measurement


def _critical_path_duration_ms(calls: list[dict[str, object]]) -> int:
    """Measure the union of recorded LLM intervals rather than summing overlap."""

    intervals = sorted(
        (
            float(call["started_at_monotonic"]),
            float(call["ended_at_monotonic"]),
        )
        for call in calls
    )
    if not intervals:
        return 0
    interval_start, interval_end = intervals[0]
    covered_duration_seconds = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start <= interval_end:
            interval_end = max(interval_end, next_end)
            continue
        covered_duration_seconds += interval_end - interval_start
        interval_start, interval_end = next_start, next_end
    covered_duration_seconds += interval_end - interval_start
    duration_ms = round(covered_duration_seconds * 1000)
    return duration_ms


def _v2_capture_measurements(
    validation_capture: dict[str, object] | None,
) -> tuple[int, int, int, int]:
    """Extract branch and local-state sizes from V2's diagnostic sidecar."""

    if validation_capture is None:
        return 0, 0, 0, 0
    events = validation_capture.get("events", [])
    if not isinstance(events, list):
        return 0, 0, 0, 0
    branch_count = 0
    maximum_concurrent_branches = 0
    state: object | None = None
    for event in events:
        if not isinstance(event, Mapping):
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if event.get("event_id") == "dependency_graph":
            definitions = payload.get("branch_definitions", [])
            if isinstance(definitions, list):
                branch_count = len(definitions)
        if event.get("event_id") == "branch_execution":
            concurrency = payload.get("maximum_concurrency", 0)
            if isinstance(concurrency, int):
                maximum_concurrent_branches = concurrency
        if event.get("event_id") == "emotion_derivation":
            state = payload.get("state_after_derivation")
    if not isinstance(state, Mapping):
        return branch_count, maximum_concurrent_branches, 0, 0
    entity_count = sum(
        len(value)
        for value in state.values()
        if isinstance(value, Mapping)
    )
    serialized_state = json.dumps(state, ensure_ascii=False, sort_keys=True)
    return (
        branch_count,
        maximum_concurrent_branches,
        entity_count,
        len(serialized_state.encode("utf-8")),
    )


def _summary(samples: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Calculate wall-clock statistics while retaining completed and failed counts."""

    summary: dict[str, dict[str, float]] = {}
    for implementation in ("v1", "v2"):
        durations = [
            float(sample["wall_clock_ms"])
            for sample in samples
            if sample["implementation"] == implementation
            and sample["failure"] is None
        ]
        failure_count = sum(
            sample["implementation"] == implementation
            and sample["failure"] is not None
            for sample in samples
        )
        if not durations:
            summary[implementation] = {
                "count": 0.0,
                "failures": float(failure_count),
            }
            continue
        summary[implementation] = {
            "count": float(len(durations)),
            "failures": float(failure_count),
            "minimum_ms": min(durations),
            "mean_ms": statistics.mean(durations),
            "median_ms": statistics.median(durations),
            "p90_ms": _percentile(durations, 90),
            "p95_ms": _percentile(durations, 95),
            "maximum_ms": max(durations),
            "standard_deviation_ms": statistics.pstdev(durations),
        }
    return summary


def _percentile(values: list[float], percentile: int) -> float:
    """Calculate an inclusive fixed-sample percentile for the raw durations."""

    quantiles = statistics.quantiles(
        values,
        n=PERCENTILE_COUNT,
        method="inclusive",
    )
    percentile_value = quantiles[percentile - 1]
    return percentile_value


def test_summary_includes_tail_percentiles_and_failure_count() -> None:
    """Keep the fixed-sample summary transparent about tails and failures."""

    samples = [
        {"implementation": "v1", "wall_clock_ms": 10, "failure": None},
        {"implementation": "v1", "wall_clock_ms": 20, "failure": None},
        {"implementation": "v1", "wall_clock_ms": 30, "failure": None},
        {"implementation": "v1", "wall_clock_ms": 40, "failure": None},
        {"implementation": "v1", "wall_clock_ms": 50, "failure": None},
        {"implementation": "v1", "wall_clock_ms": 60, "failure": None},
        {"implementation": "v1", "wall_clock_ms": 70, "failure": None},
        {"implementation": "v1", "wall_clock_ms": 80, "failure": None},
        {"implementation": "v1", "wall_clock_ms": 90, "failure": None},
        {"implementation": "v1", "wall_clock_ms": 100, "failure": None},
        {"implementation": "v2", "wall_clock_ms": 10, "failure": None},
        {"implementation": "v2", "wall_clock_ms": 20, "failure": None},
        {"implementation": "v2", "wall_clock_ms": 30, "failure": None},
        {"implementation": "v2", "wall_clock_ms": 40, "failure": None},
        {"implementation": "v2", "wall_clock_ms": 50, "failure": None},
        {"implementation": "v2", "wall_clock_ms": 60, "failure": None},
        {"implementation": "v2", "wall_clock_ms": 70, "failure": None},
        {"implementation": "v2", "wall_clock_ms": 80, "failure": None},
        {"implementation": "v2", "wall_clock_ms": 90, "failure": None},
        {"implementation": "v2", "wall_clock_ms": 100, "failure": {"class": "TimeoutError"}},
    ]

    summary = _summary(samples)

    assert summary["v1"]["p90_ms"] == 91.0
    assert summary["v1"]["p95_ms"] == 95.5
    assert summary["v2"]["failures"] == 1.0
