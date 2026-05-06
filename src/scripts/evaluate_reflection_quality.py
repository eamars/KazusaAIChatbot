"""Evaluate exported reflection-cycle data for diagnostic review."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bson import json_util

from scripts._db_export import configure_stdout


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for reflection quality analysis.

    Returns:
        Configured argument parser for exported reflection artifacts.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate exported reflection-run and promoted-memory JSON.",
    )
    parser.add_argument(
        "--reflection-runs",
        type=Path,
        required=True,
        help="Exported character_reflection_runs JSON path.",
    )
    parser.add_argument(
        "--promoted-memory",
        type=Path,
        help="Exported reflection_inferred memory JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination report JSON path.",
    )
    return parser


def _load_documents(path: Path) -> list[dict[str, Any]]:
    """Load a read-only export payload and return its document list.

    Args:
        path: JSON export created by the project export scripts.

    Returns:
        Exported document rows from the payload.
    """

    raw_text = path.read_text(encoding="utf-8")
    payload = json_util.loads(raw_text)
    documents = payload["documents"]
    return_value = [
        dict(document)
        for document in documents
        if isinstance(document, dict)
    ]
    return return_value


def _counter_dict(counter: Counter) -> dict[str, int]:
    """Convert a Counter to a JSON-stable string-keyed dictionary."""

    return_value = {
        str(key): int(value)
        for key, value in counter.items()
    }
    return return_value


def _text_present(value: object) -> bool:
    """Return whether a generated text field contains non-empty text."""

    return_value = isinstance(value, str) and bool(value.strip())
    return return_value


def _snippet(value: object, *, limit: int = 160) -> str:
    """Return a compact string preview for diagnostics.

    Args:
        value: Raw value to preview.
        limit: Maximum returned character count.

    Returns:
        Trimmed text preview.
    """

    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return_value = f"{text[: limit - 1].rstrip()}…"
    return return_value


def _required_output_fields(run_kind: str) -> tuple[str, ...]:
    """Return required output fields for a reflection run kind."""

    if run_kind == "hourly_slot":
        return_value = (
            "topic_summary",
            "participant_observations",
            "conversation_quality_feedback",
            "privacy_notes",
            "confidence",
            "hourly_scope_ref",
            "active_character_utterances",
        )
        return return_value
    if run_kind == "daily_channel":
        return_value = (
            "day_summary",
            "active_hour_summaries",
            "cross_hour_topics",
            "conversation_quality_patterns",
            "privacy_risks",
            "synthesis_limitations",
            "confidence",
        )
        return return_value
    if run_kind == "daily_global_promotion":
        return_value = ("promotion_decisions",)
        return return_value
    return_value = ()
    return return_value


def _scope_channel_id(document: dict[str, Any]) -> str:
    """Return the platform channel id from a reflection-run scope."""

    scope = document.get("scope")
    if not isinstance(scope, dict):
        return_value = ""
        return return_value
    return_value = str(scope.get("platform_channel_id", ""))
    return return_value


def _run_shape_issues(document: dict[str, Any]) -> list[str]:
    """Return structural issues found in one reflection-run document."""

    issues: list[str] = []
    run_kind = str(document.get("run_kind", ""))
    output = document.get("output")
    if not isinstance(output, dict):
        issues.append("missing_output_object")
        return issues

    for field in _required_output_fields(run_kind):
        if field not in output:
            issues.append(f"missing_output_field:{field}")

    if run_kind == "hourly_slot":
        if not _text_present(output.get("topic_summary")):
            issues.append("empty_topic_summary")
        for list_field in (
            "participant_observations",
            "conversation_quality_feedback",
            "privacy_notes",
        ):
            value = output.get(list_field)
            if not isinstance(value, list) or not value:
                issues.append(f"empty_or_invalid_list:{list_field}")

    if run_kind == "daily_channel":
        if not _text_present(output.get("day_summary")):
            issues.append("empty_day_summary")
        active_hours = output.get("active_hour_summaries")
        if not isinstance(active_hours, list) or not active_hours:
            issues.append("empty_active_hour_summaries")

    if run_kind == "daily_global_promotion":
        decisions = output.get("promotion_decisions")
        if not isinstance(decisions, list):
            issues.append("invalid_promotion_decisions")

    return issues


def _confidence_stats(documents: list[dict[str, Any]]) -> dict[str, int]:
    """Count reflection confidence labels across run outputs."""

    confidence = Counter()
    for document in documents:
        output = document.get("output")
        if not isinstance(output, dict):
            continue
        label = str(output.get("confidence", "")).strip() or "missing"
        confidence[label] += 1
    return_value = _counter_dict(confidence)
    return return_value


def _hourly_stats(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Build quality indicators for hourly reflection rows."""

    hourly_rows = [
        document
        for document in documents
        if document.get("run_kind") == "hourly_slot"
    ]
    active_character_utterance_count = 0
    feedback_without_character = 0
    privacy_note_count = 0
    observation_count = 0
    for document in hourly_rows:
        output = document.get("output")
        if not isinstance(output, dict):
            continue
        utterances = output.get("active_character_utterances")
        utterance_len = len(utterances) if isinstance(utterances, list) else 0
        active_character_utterance_count += utterance_len
        feedback = output.get("conversation_quality_feedback")
        feedback_len = len(feedback) if isinstance(feedback, list) else 0
        if utterance_len == 0 and feedback_len > 0:
            feedback_without_character += 1
        notes = output.get("privacy_notes")
        privacy_note_count += len(notes) if isinstance(notes, list) else 0
        observations = output.get("participant_observations")
        observation_count += len(observations) if isinstance(observations, list) else 0

    stats = {
        "hourly_count": len(hourly_rows),
        "total_participant_observations": observation_count,
        "total_privacy_notes": privacy_note_count,
        "total_active_character_utterances": active_character_utterance_count,
        "hourly_rows_with_feedback_but_no_character_utterance": feedback_without_character,
    }
    return stats


def _promotion_stats(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize global promotion decisions stored in reflection runs."""

    promotion_rows = [
        document
        for document in documents
        if document.get("run_kind") == "daily_global_promotion"
    ]
    lane_counts = Counter()
    decision_counts = Counter()
    memory_type_counts = Counter()
    decision_previews: list[dict[str, str]] = []
    inconsistent_no_action_decisions: list[dict[str, str]] = []
    for document in promotion_rows:
        decisions = document.get("promotion_decisions")
        if not isinstance(decisions, list):
            output = document.get("output")
            if not isinstance(output, dict):
                continue
            decisions = output.get("promotion_decisions")
        if not isinstance(decisions, list):
            continue
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            decision_label = str(decision.get("decision", "missing"))
            signal_strength = str(decision.get("signal_strength", ""))
            character_agreement = str(decision.get("character_agreement", ""))
            lane_counts[str(decision.get("lane", "missing"))] += 1
            decision_counts[decision_label] += 1
            memory_type_counts[str(decision.get("memory_type", "missing"))] += 1
            decision_previews.append({
                "lane": str(decision.get("lane", "")),
                "decision": decision_label,
                "memory_name": _snippet(
                    decision.get("sanitized_memory_name", ""),
                ),
                "content": _snippet(
                    decision.get("sanitized_content", ""),
                ),
            })
            if (
                decision_label == "no_action"
                and (
                    signal_strength == "high"
                    or character_agreement == "spoken"
                )
            ):
                boundary_assessment = decision.get("boundary_assessment")
                if not isinstance(boundary_assessment, dict):
                    boundary_assessment = {}
                inconsistent_no_action_decisions.append({
                    "run_id": str(document.get("run_id", "")),
                    "lane": str(decision.get("lane", "")),
                    "signal_strength": signal_strength,
                    "character_agreement": character_agreement,
                    "reason": _snippet(boundary_assessment.get("reason", "")),
                })

    stats = {
        "promotion_run_count": len(promotion_rows),
        "lane_counts": _counter_dict(lane_counts),
        "decision_counts": _counter_dict(decision_counts),
        "memory_type_counts": _counter_dict(memory_type_counts),
        "decision_previews": decision_previews,
        "inconsistent_no_action_decisions": inconsistent_no_action_decisions,
    }
    return stats


def _bigrams(text: str) -> set[str]:
    """Return character bigrams for rough duplicate detection."""

    compact = "".join(str(text or "").split())
    if len(compact) < 2:
        return_value: set[str] = set()
        return return_value
    return_value = {
        compact[index:index + 2]
        for index in range(len(compact) - 1)
    }
    return return_value


def _similarity(left: str, right: str) -> float:
    """Return a rough character-bigram Jaccard similarity score."""

    left_terms = _bigrams(left)
    right_terms = _bigrams(right)
    if not left_terms or not right_terms:
        return 0.0
    overlap = left_terms & right_terms
    combined = left_terms | right_terms
    score = len(overlap) / len(combined)
    return score


def _promoted_memory_stats(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize reflection-promoted memory rows and duplicate risk."""

    type_counts = Counter(document.get("memory_type") for document in documents)
    status_counts = Counter(document.get("status") for document in documents)
    duplicate_candidates: list[dict[str, Any]] = []
    for left_index, left_doc in enumerate(documents):
        for right_doc in documents[left_index + 1:]:
            left_text = f"{left_doc.get('memory_name', '')} {left_doc.get('content', '')}"
            right_text = f"{right_doc.get('memory_name', '')} {right_doc.get('content', '')}"
            score = _similarity(left_text, right_text)
            if score < 0.18:
                continue
            duplicate_candidates.append({
                "score": round(score, 3),
                "left_memory_type": str(left_doc.get("memory_type", "")),
                "left_memory_name": _snippet(left_doc.get("memory_name", "")),
                "right_memory_type": str(right_doc.get("memory_type", "")),
                "right_memory_name": _snippet(right_doc.get("memory_name", "")),
            })

    stats = {
        "promoted_memory_count": len(documents),
        "memory_type_counts": _counter_dict(type_counts),
        "status_counts": _counter_dict(status_counts),
        "duplicate_risk_pairs": duplicate_candidates,
        "rows": [
            {
                "memory_type": str(document.get("memory_type", "")),
                "status": str(document.get("status", "")),
                "memory_name": _snippet(document.get("memory_name", "")),
                "content": _snippet(document.get("content", "")),
                "privacy_review": document.get("privacy_review", {}),
                "evidence_ref_count": len(document.get("evidence_refs") or []),
            }
            for document in documents
        ],
    }
    return stats


def _quality_findings(
    *,
    run_documents: list[dict[str, Any]],
    promoted_memory_documents: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build high-signal quality findings from deterministic indicators."""

    findings: list[dict[str, str]] = []
    shape_issue_count = sum(
        len(_run_shape_issues(document))
        for document in run_documents
    )
    failed_count = sum(
        1
        for document in run_documents
        if document.get("status") != "succeeded"
    )
    validation_warning_count = sum(
        len(document.get("validation_warnings") or [])
        for document in run_documents
    )

    if failed_count == 0 and shape_issue_count == 0:
        findings.append({
            "severity": "positive",
            "topic": "structural reliability",
            "finding": "All exported reflection runs succeeded and matched their expected output shape.",
        })
    else:
        findings.append({
            "severity": "high",
            "topic": "structural reliability",
            "finding": (
                f"Detected {failed_count} non-succeeded run(s) and "
                f"{shape_issue_count} output-shape issue(s)."
            ),
        })

    if validation_warning_count:
        findings.append({
            "severity": "medium",
            "topic": "validation warnings",
            "finding": f"Found {validation_warning_count} stored validation warning(s).",
        })

    hourly = _hourly_stats(run_documents)
    no_character_feedback = hourly[
        "hourly_rows_with_feedback_but_no_character_utterance"
    ]
    if no_character_feedback:
        findings.append({
            "severity": "medium",
            "topic": "actionability",
            "finding": (
                f"{no_character_feedback} hourly row(s) gave conversation-quality "
                "advice even though the active character had no utterances in that hour."
            ),
        })

    promoted_stats = _promoted_memory_stats(promoted_memory_documents)
    duplicate_pairs = promoted_stats["duplicate_risk_pairs"]
    if duplicate_pairs:
        findings.append({
            "severity": "medium",
            "topic": "promotion deduplication",
            "finding": (
                f"Detected {len(duplicate_pairs)} promoted-memory pair(s) with "
                "rough duplicate risk."
            ),
        })

    promotion_stats = _promotion_stats(run_documents)
    inconsistent_no_action = promotion_stats[
        "inconsistent_no_action_decisions"
    ]
    if inconsistent_no_action:
        findings.append({
            "severity": "medium",
            "topic": "promotion validation",
            "finding": (
                f"Detected {len(inconsistent_no_action)} no_action promotion "
                "decision(s) with high-signal or spoken-agreement metadata."
            ),
        })

    defense_rule_count = promoted_stats["memory_type_counts"].get("defense_rule", 0)
    if defense_rule_count:
        findings.append({
            "severity": "medium",
            "topic": "retrieval authority",
            "finding": (
                f"{defense_rule_count} reflection-promoted self-guidance row(s) "
                "exist in shared memory and need to stay out of ordinary factual RAG evidence."
            ),
        })

    return findings


def _run_quality_report(
    *,
    run_documents: list[dict[str, Any]],
    promoted_memory_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the complete quality report payload."""

    by_scope_date: dict[str, dict[str, int]] = defaultdict(dict)
    for document in run_documents:
        scope = document.get("scope")
        scope_ref = ""
        if isinstance(scope, dict):
            scope_ref = str(scope.get("platform_channel_id", ""))
        character_local_date = str(document.get("character_local_date", ""))
        key = f"{scope_ref}:{character_local_date}"
        run_kind = str(document.get("run_kind", ""))
        by_scope_date[key][run_kind] = by_scope_date[key].get(run_kind, 0) + 1

    shape_issues = [
        {
            "run_id": str(document.get("run_id", "")),
            "run_kind": str(document.get("run_kind", "")),
            "issues": _run_shape_issues(document),
        }
        for document in run_documents
        if _run_shape_issues(document)
    ]
    validation_warnings = [
        {
            "run_id": str(document.get("run_id", "")),
            "run_kind": str(document.get("run_kind", "")),
            "warnings": list(document.get("validation_warnings") or []),
        }
        for document in run_documents
        if document.get("validation_warnings")
    ]

    report = {
        "summary": {
            "reflection_run_count": len(run_documents),
            "promoted_memory_count": len(promoted_memory_documents),
            "run_kind_counts": _counter_dict(
                Counter(document.get("run_kind") for document in run_documents)
            ),
            "status_counts": _counter_dict(
                Counter(document.get("status") for document in run_documents)
            ),
            "character_local_date_counts": _counter_dict(
                Counter(
                    document.get("character_local_date")
                    for document in run_documents
                )
            ),
            "scope_counts": _counter_dict(
                Counter(
                    _scope_channel_id(document)
                    for document in run_documents
                )
            ),
            "confidence_counts": _confidence_stats(run_documents),
        },
        "coverage_by_scope_date": dict(by_scope_date),
        "hourly_quality": _hourly_stats(run_documents),
        "promotion_quality": _promotion_stats(run_documents),
        "promoted_memory_quality": _promoted_memory_stats(
            promoted_memory_documents,
        ),
        "validation_warnings": validation_warnings,
        "shape_issues": shape_issues,
        "quality_findings": _quality_findings(
            run_documents=run_documents,
            promoted_memory_documents=promoted_memory_documents,
        ),
    }
    return report


def main() -> None:
    """Run the reflection quality analyzer from exported JSON files."""

    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()

    run_documents = _load_documents(args.reflection_runs)
    promoted_memory_documents: list[dict[str, Any]] = []
    if args.promoted_memory:
        promoted_memory_documents = _load_documents(args.promoted_memory)

    report = _run_quality_report(
        run_documents=run_documents,
        promoted_memory_documents=promoted_memory_documents,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json_util.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"wrote reflection quality report to {args.output} "
        f"runs={len(run_documents)} promoted_memory={len(promoted_memory_documents)}"
    )


if __name__ == "__main__":
    main()
