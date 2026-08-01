"""Public-evidence specialist over the complex-task resolver public IO."""

from __future__ import annotations

from collections.abc import Mapping

from kazusa_ai_chatbot.action_spec.models import EVIDENCE_REF_VERSION
from kazusa_ai_chatbot.complex_task_resolver import (
    COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
    COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
    COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
    ComplexTaskValidationError,
    project_complex_task_packet,
    resolve_complex_task,
)
from kazusa_ai_chatbot.task_resolution.contracts import (
    TaskResolutionEvidenceV1,
    TaskResolutionExecutionContextV1,
    TaskSpecialistRequestV1,
    TaskSpecialistResultV1,
)
from kazusa_ai_chatbot.task_resolution.specialists import (
    _bounded_text,
    _require_handler_coding_objective_mode,
    _specialist_evidence,
    _specialist_result,
    _validated_handler_inputs,
)


SPECIALIST = "public_research"


async def resolve_with_public_research(
    request: dict[str, object],
    execution_context: TaskResolutionExecutionContextV1,
) -> TaskSpecialistResultV1:
    """Resolve one selected subgoal through the public complex-resolver IO."""

    task_request, context = _validated_handler_inputs(request, execution_context)
    _require_handler_coding_objective_mode(
        task_request,
        specialist=SPECIALIST,
    )
    resolver_request = {
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": task_request["objective"],
        "reason": "Task resolution requested public source evidence.",
        "source": "l2d",
        "priority": "normal",
    }
    resolver_context = {
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": context["conversation_summary"],
        "persona_context_summary": context["persona_summary"],
        "time_context": dict(context["local_time_context"]),
        "available_evidence": _available_evidence(task_request),
    }
    resolver_options = {
        "schema_version": COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
        "limits": {},
    }
    try:
        packet = await resolve_complex_task(
            resolver_request,
            resolver_context,
            resolver_options,
        )
        projection = project_complex_task_packet(packet)
    except ComplexTaskValidationError:
        return _specialist_result(
            specialist=SPECIALIST,
            status="failed",
            remaining_needs=[task_request["objective"]],
            reason="Public research returned invalid public resolver output.",
        )

    provenance_refs = _public_provenance_refs(packet)
    summary = _research_summary(projection)
    limitations = _projection_text_items(projection, "knowledge_still_lacking")
    if not provenance_refs or not summary:
        return _specialist_result(
            specialist=SPECIALIST,
            status="incompatible",
            remaining_needs=_remaining_needs(task_request),
            reason="Public research did not return evidence for this subgoal.",
        )
    evidence = _specialist_evidence(
        request=task_request,
        specialist=SPECIALIST,
        summary=summary,
        provenance_refs=provenance_refs,
        limitations=limitations,
    )
    status = "partial" if limitations else "resolved"
    return _specialist_result(
        specialist=SPECIALIST,
        status=status,
        evidence=[evidence],
        completed_subgoals=[task_request["objective"]],
        remaining_needs=limitations,
        reason="Public research returned provenance-bearing evidence.",
    )


def _available_evidence(request: TaskSpecialistRequestV1) -> list[dict[str, object]]:
    """Map validated task evidence into the complex resolver's public ref IO."""

    refs: list[dict[str, object]] = []
    for evidence in request["available_evidence"]:
        refs.append({
            "schema_version": EVIDENCE_REF_VERSION,
            "evidence_kind": "tool_result",
            "evidence_id": evidence["evidence_id"],
            "owner": evidence["specialist"],
            "excerpt": evidence["summary"],
            "observed_at": None,
        })
    return refs


def _public_provenance_refs(packet: object) -> list[str]:
    """Project public evidence IDs without passing raw provider payloads on."""

    if not isinstance(packet, Mapping):
        return []
    graph = packet.get("graph")
    if not isinstance(graph, Mapping):
        return []
    nodes = graph.get("nodes")
    if not isinstance(nodes, Mapping):
        return []
    provenance_refs: list[str] = []
    for raw_node in nodes.values():
        if not isinstance(raw_node, Mapping):
            continue
        raw_refs = raw_node.get("evidence_refs")
        if not isinstance(raw_refs, list):
            continue
        for raw_ref in raw_refs:
            if not isinstance(raw_ref, Mapping):
                continue
            evidence_id = raw_ref.get("evidence_id")
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                continue
            normalized_id = evidence_id.strip()
            if normalized_id.startswith(("http://", "https://")):
                provenance_ref = normalized_id
            else:
                provenance_ref = f"public_research:{normalized_id}"
            if provenance_ref not in provenance_refs:
                provenance_refs.append(provenance_ref)
            if len(provenance_refs) >= 8:
                return provenance_refs
    return provenance_refs


def _research_summary(projection: Mapping[str, object]) -> str:
    """Use the resolver's compact semantic summary as specialist evidence."""

    summary_rows: list[str] = []
    investigation_summary = projection.get("investigation_summary")
    if isinstance(investigation_summary, str) and investigation_summary.strip():
        summary_rows.append(investigation_summary.strip())
    known_items = _projection_text_items(projection, "knowledge_we_know_so_far")
    for known_item in known_items:
        if known_item not in summary_rows:
            summary_rows.append(known_item)
    if not summary_rows:
        return ""
    return _bounded_text("\n".join(summary_rows))


def _projection_text_items(
    projection: Mapping[str, object],
    field_name: str,
) -> list[str]:
    """Return a bounded text-list projection from public resolver output."""

    raw_items = projection.get(field_name)
    if not isinstance(raw_items, list):
        return []
    items: list[str] = []
    for raw_item in raw_items[:8]:
        if isinstance(raw_item, str) and raw_item.strip():
            items.append(_bounded_text(raw_item))
    return items


def _remaining_needs(request: TaskSpecialistRequestV1) -> list[str]:
    """Retain canonical unresolved needs after a public failure."""

    if request["remaining_needs"]:
        return list(request["remaining_needs"])
    return [request["objective"]]
