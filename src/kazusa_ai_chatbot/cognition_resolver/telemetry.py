"""Sanitized telemetry helpers for cognition resolver traces."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.cognition_resolver.contracts import (
    MAX_RESOLVER_TRACE_CHARS,
    ResolverCycleStateV1,
    ResolverCycleTraceV1,
    validate_resolver_cycle_trace,
)
from kazusa_ai_chatbot.cognition_resolver.state import validate_resolver_state
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState

RESOLVER_TELEMETRY_COMPONENT = "nodes.cognition_resolver"
RESOLVER_TELEMETRY_SCHEMA_VERSION = "resolver_telemetry.v1"
DEFAULT_RESOLVER_TRACE_DIR = Path("test_artifacts") / "cognition_resolver"
FAST_DURATION_MAX_MS = 1000
NORMAL_DURATION_MAX_MS = 5000
MAX_TELEMETRY_LIST_ITEMS = 8

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]+")
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def build_resolver_cycle_event(
    state: GlobalPersonaState,
    trace: ResolverCycleTraceV1,
    *,
    duration_ms: int | float | None = None,
) -> dict[str, Any]:
    """Build one sanitized event-shaped record for a resolver cycle.

    Args:
        state: Persona state containing validated resolver state.
        trace: One prompt-safe resolver cycle trace row.
        duration_ms: Optional elapsed time for the cycle.

    Returns:
        A sanitized dictionary suitable for a future dedicated event recorder.
    """

    resolver_state = validate_resolver_state(state["resolver_state"])
    validated_trace = validate_resolver_cycle_trace(trace)
    observation_statuses = _observation_statuses_for_trace(
        resolver_state,
        validated_trace,
    )
    metrics = _cycle_metrics(validated_trace, resolver_state, duration_ms)
    event = {
        "schema_version": RESOLVER_TELEMETRY_SCHEMA_VERSION,
        "component": RESOLVER_TELEMETRY_COMPONENT,
        "event_kind": "resolver_cycle",
        "status": _safe_text(validated_trace["terminal_reason"]),
        "metrics": metrics,
        "labels": {
            "selected_capability_kind": _safe_text(
                validated_trace["selected_capability_kind"],
            ),
            "observation_status": _joined_label(observation_statuses),
            "pending_resume_status": _pending_resume_status(resolver_state),
            "final_surface_decision": _safe_text(
                validated_trace["final_surface_decision"],
            ),
            "duration_label": _duration_label(duration_ms),
        },
        "payload": _cycle_payload(validated_trace),
        "refs": [],
        "warning_codes": [],
    }
    return_value = event
    return return_value


def build_resolver_terminal_event(
    state: GlobalPersonaState,
    *,
    duration_ms: int | float | None = None,
) -> dict[str, Any]:
    """Build a sanitized terminal event-shaped record for one resolver run.

    Args:
        state: Persona state containing validated resolver state.
        duration_ms: Optional elapsed time for the complete resolver run.

    Returns:
        A sanitized dictionary containing bounded per-cycle summaries.
    """

    resolver_state = validate_resolver_state(state["resolver_state"])
    observations = resolver_state["observations"]
    traces = resolver_state["cycle_traces"]
    metrics = _terminal_metrics(resolver_state, duration_ms)
    event = {
        "schema_version": RESOLVER_TELEMETRY_SCHEMA_VERSION,
        "component": RESOLVER_TELEMETRY_COMPONENT,
        "event_kind": "resolver_terminal",
        "status": resolver_state["status"],
        "metrics": metrics,
        "labels": {
            "terminal_reason": _safe_text(resolver_state["terminal_reason"]),
            "capability_kinds": _joined_label(_capability_kinds(observations)),
            "observation_status": _joined_label(_observation_statuses(
                observations,
            )),
            "pending_resume_status": _pending_resume_status(resolver_state),
            "duration_label": _duration_label(duration_ms),
        },
        "payload": {
            "terminal_reason": _safe_text(resolver_state["terminal_reason"]),
            "observations": _observation_payloads(observations),
            "cycles": [_cycle_payload(trace) for trace in traces],
        },
        "refs": [],
        "warning_codes": [],
    }
    return_value = event
    return return_value


def write_human_readable_resolver_trace(
    state: GlobalPersonaState,
    output_dir: str | Path = DEFAULT_RESOLVER_TRACE_DIR,
    *,
    filename_stem: str = "resolver_trace",
) -> Path:
    """Write a bounded local Markdown trace for manual resolver inspection.

    Args:
        state: Persona state containing validated resolver state.
        output_dir: Directory where the trace artifact should be written.
        filename_stem: Human-readable filename stem, sanitized before use.

    Returns:
        The written trace path.
    """

    resolver_state = validate_resolver_state(state["resolver_state"])
    trace_dir = Path(output_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = _safe_filename_stem(filename_stem)
    trace_path = trace_dir / f"{safe_stem}.md"
    trace_text = "\n".join(_trace_markdown_lines(resolver_state))
    trace_path.write_text(trace_text, encoding="utf-8")
    return_value = trace_path
    return return_value


def _cycle_metrics(
    trace: ResolverCycleTraceV1,
    resolver_state: ResolverCycleStateV1,
    duration_ms: int | float | None,
) -> dict[str, int]:
    """Build numeric cycle metrics without raw refs."""

    metrics = {
        "cycle_index": trace["cycle_index"],
        "cycle_count": len(resolver_state["cycle_traces"]),
        "resolver_request_count": len(
            trace["l2d_resolver_capability_requests"],
        ),
        "action_spec_count": len(trace["l2d_action_specs_summary"]),
        "observation_count": len(trace["observation_ids"]),
    }
    normalized_duration = _duration_ms(duration_ms)
    if normalized_duration is not None:
        metrics["duration_ms"] = normalized_duration
    return_value = metrics
    return return_value


def _terminal_metrics(
    resolver_state: ResolverCycleStateV1,
    duration_ms: int | float | None,
) -> dict[str, int]:
    """Build numeric terminal metrics without raw refs."""

    metrics = {
        "cycle_count": len(resolver_state["cycle_traces"]),
        "observation_count": len(resolver_state["observations"]),
        "held_action_spec_count": len(resolver_state["held_action_specs"]),
    }
    normalized_duration = _duration_ms(duration_ms)
    if normalized_duration is not None:
        metrics["duration_ms"] = normalized_duration
    return_value = metrics
    return return_value


def _cycle_payload(trace: ResolverCycleTraceV1) -> dict[str, Any]:
    """Build one bounded cycle payload for logs and artifacts."""

    payload = {
        "cycle_index": trace["cycle_index"],
        "l1": {
            "emotional_appraisal": _safe_text(
                trace["l1_emotional_appraisal"],
            ),
            "interaction_subtext": _safe_text(
                trace["l1_interaction_subtext"],
            ),
        },
        "l2": {
            "internal_monologue_summary": _safe_text(
                trace["l2_internal_monologue_summary"],
            ),
            "logical_stance": _safe_text(trace["l2_logical_stance"]),
            "character_intent": _safe_text(trace["l2_character_intent"]),
            "judgment_note": _safe_text(trace["l2_judgment_note"]),
        },
        "l2d": {
            "resolver_capabilities": _capability_request_summaries(
                trace["l2d_resolver_capability_requests"],
            ),
            "action_specs": _safe_text_list(trace["l2d_action_specs_summary"]),
        },
        "selected_capability_kind": _safe_text(
            trace["selected_capability_kind"],
        ),
        "final_surface_decision": _safe_text(trace["final_surface_decision"]),
        "terminal_reason": _safe_text(trace["terminal_reason"]),
    }
    return_value = payload
    return return_value


def _capability_request_summaries(requests: list[dict[str, Any]]) -> list[str]:
    """Summarize resolver requests without exposing prompt bodies or ids."""

    summaries: list[str] = []
    for request in requests[:MAX_TELEMETRY_LIST_ITEMS]:
        capability_kind = _safe_text(request["capability_kind"])
        priority = _safe_text(request["priority"])
        objective = _safe_text(request["objective"])
        summary = (
            f"capability={capability_kind}; priority={priority}; "
            f"objective={objective}"
        )
        summaries.append(summary)
    return_value = summaries
    return return_value


def _observation_statuses_for_trace(
    resolver_state: ResolverCycleStateV1,
    trace: ResolverCycleTraceV1,
) -> list[str]:
    """Return statuses for observations referenced by a trace."""

    status_by_id = {
        observation["observation_id"]: observation["status"]
        for observation in resolver_state["observations"]
    }
    statuses = []
    for observation_id in trace["observation_ids"]:
        status = status_by_id.get(observation_id)
        if status is not None:
            statuses.append(status)
    return_value = statuses
    return return_value


def _observation_statuses(observations: list[dict[str, Any]]) -> list[str]:
    """Return all observation statuses in run order."""

    statuses = [
        _safe_text(observation["status"])
        for observation in observations[:MAX_TELEMETRY_LIST_ITEMS]
    ]
    return_value = statuses
    return return_value


def _observation_payloads(
    observations: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build bounded observation payloads without observation ids."""

    payloads = []
    for observation in observations[:MAX_TELEMETRY_LIST_ITEMS]:
        payload = {
            "capability_kind": _safe_text(observation["capability_kind"]),
            "status": _safe_text(observation["status"]),
            "summary": _safe_text(observation["prompt_safe_summary"]),
        }
        payloads.append(payload)
    return_value = payloads
    return return_value


def _capability_kinds(observations: list[dict[str, Any]]) -> list[str]:
    """Return unique observed capability kinds in run order."""

    seen: set[str] = set()
    capability_kinds: list[str] = []
    for observation in observations:
        capability_kind = _safe_text(observation["capability_kind"])
        if capability_kind in seen:
            continue
        seen.add(capability_kind)
        capability_kinds.append(capability_kind)
    return_value = capability_kinds[:MAX_TELEMETRY_LIST_ITEMS]
    return return_value


def _pending_resume_status(resolver_state: ResolverCycleStateV1) -> str:
    """Return pending-resume status when present."""

    pending_resume = resolver_state.get("pending_resume")
    if pending_resume is None:
        return_value = "none"
        return return_value
    return_value = _safe_text(pending_resume["status"])
    return return_value


def _duration_ms(value: int | float | None) -> int | None:
    """Normalize an optional duration into non-negative milliseconds."""

    if value is None:
        return_value = None
        return return_value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return_value = None
        return return_value
    rounded_value = int(max(0, round(value)))
    return_value = rounded_value
    return return_value


def _duration_label(value: int | float | None) -> str:
    """Classify an optional duration for operational scanning."""

    normalized_duration = _duration_ms(value)
    if normalized_duration is None:
        return_value = "not_recorded"
        return return_value
    if normalized_duration <= FAST_DURATION_MAX_MS:
        return_value = "fast"
        return return_value
    if normalized_duration <= NORMAL_DURATION_MAX_MS:
        return_value = "normal"
        return return_value
    return_value = "slow"
    return return_value


def _safe_text_list(values: list[str]) -> list[str]:
    """Sanitize and bound a list of short trace strings."""

    safe_values = [
        _safe_text(value)
        for value in values[:MAX_TELEMETRY_LIST_ITEMS]
    ]
    return_value = safe_values
    return return_value


def _joined_label(values: list[str]) -> str:
    """Join short label values while preserving an explicit empty marker."""

    safe_values = [_safe_text(value, limit=80) for value in values if value]
    if not safe_values:
        return_value = "none"
        return return_value
    return_value = "|".join(safe_values[:MAX_TELEMETRY_LIST_ITEMS])
    return return_value


def _safe_text(
    value: object,
    *,
    limit: int = MAX_RESOLVER_TRACE_CHARS,
) -> str:
    """Return a bounded single-line text value."""

    if not isinstance(value, str):
        return_value = ""
        return return_value
    cleaned = _CONTROL_CHARS_RE.sub(" ", value).strip()
    if len(cleaned) <= limit:
        return_value = cleaned
        return return_value
    body_limit = max(0, limit - len("..."))
    clipped = cleaned[:body_limit].rstrip()
    return_value = f"{clipped}..."
    return return_value


def _safe_filename_stem(value: str) -> str:
    """Return a stable filesystem-safe trace filename stem."""

    stripped = value.strip()
    if not stripped:
        stripped = "resolver_trace"
    cleaned = _SAFE_FILENAME_RE.sub("_", stripped)
    safe_stem = cleaned.strip("._")[:120]
    if not safe_stem:
        safe_stem = "resolver_trace"
    return_value = safe_stem
    return return_value


def _trace_markdown_lines(
    resolver_state: ResolverCycleStateV1,
) -> list[str]:
    """Render a bounded human-readable resolver trace."""

    lines = [
        "# Cognition Resolver Trace",
        "",
        f"- status: {_safe_text(resolver_state['status'])}",
        f"- terminal_reason: {_safe_text(resolver_state['terminal_reason'])}",
        f"- cycle_count: {len(resolver_state['cycle_traces'])}",
        f"- observation_count: {len(resolver_state['observations'])}",
        f"- pending_resume_status: {_pending_resume_status(resolver_state)}",
        "",
    ]
    for trace in resolver_state["cycle_traces"]:
        _extend_trace_cycle_lines(lines, trace)
    if resolver_state["observations"]:
        lines.extend(["## Observations", ""])
        for observation in resolver_state["observations"]:
            lines.append(
                "- "
                f"capability={_safe_text(observation['capability_kind'])}; "
                f"status={_safe_text(observation['status'])}; "
                f"summary={_safe_text(observation['prompt_safe_summary'])}"
            )
        lines.append("")
    return_value = lines
    return return_value


def _extend_trace_cycle_lines(
    lines: list[str],
    trace: ResolverCycleTraceV1,
) -> None:
    """Append one cycle section to a Markdown trace."""

    payload = _cycle_payload(trace)
    lines.extend([
        f"## Cycle {payload['cycle_index']}",
        "",
        f"- selected_capability: {payload['selected_capability_kind']}",
        f"- final_surface_decision: {payload['final_surface_decision']}",
        f"- terminal_reason: {payload['terminal_reason']}",
        f"- L1 emotional_appraisal: {payload['l1']['emotional_appraisal']}",
        f"- L1 interaction_subtext: {payload['l1']['interaction_subtext']}",
        f"- L2 internal_monologue: {payload['l2']['internal_monologue_summary']}",
        f"- L2 logical_stance: {payload['l2']['logical_stance']}",
        f"- L2 character_intent: {payload['l2']['character_intent']}",
        f"- L2 judgment_note: {payload['l2']['judgment_note']}",
    ])
    resolver_capabilities = payload["l2d"]["resolver_capabilities"]
    if resolver_capabilities:
        lines.append("- L2d resolver_capabilities:")
        for summary in resolver_capabilities:
            lines.append(f"  - {summary}")
    action_specs = payload["l2d"]["action_specs"]
    if action_specs:
        lines.append("- L2d action_specs:")
        for summary in action_specs:
            lines.append(f"  - {summary}")
    lines.append("")
