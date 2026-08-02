"""Recover prompt-safe replay inputs for the Asuna growth cohort."""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.character_identity_growth import llm, models
from kazusa_ai_chatbot.utils import parse_llm_json_output


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_DIRECTORY = ROOT / "test_artifacts" / "diagnostics"
MANIFEST_PATH = DIAGNOSTIC_DIRECTORY / (
    "asuna_identity_growth_replay_v1.json"
)
RUNS_PATH = DIAGNOSTIC_DIRECTORY / (
    "asuna_growth_review_20260802_runs.json"
)
HISTORY_PATH = DIAGNOSTIC_DIRECTORY / (
    "asuna_growth_review_20260802_identity_history_current.json"
)
TRACE_STEPS_PATH = DIAGNOSTIC_DIRECTORY / (
    "asuna_growth_review_20260802_identity_trace_steps_current.json"
)
REVISIONS_PATH = DIAGNOSTIC_DIRECTORY / (
    "asuna_growth_review_20260802_revisions.json"
)
AUDIT_PATH = DIAGNOSTIC_DIRECTORY / (
    "asuna_identity_growth_replay_recovery_audit.json"
)

EXPECTED_CASE_COUNT = 185
SUMMARY_TEXT_LIMIT = models.IDENTITY_EVIDENCE_CARD_TEXT_LIMIT
FORBIDDEN_SUMMARY_PATTERN = re.compile(
    r"(?i)(?:user_message:|identity-(?:evidence|candidate)|"
    r"llmtrace_|(?:chat|platform|channel)[_-]|\b(?:qq|debug)\b)"
)
SUMMARY_SYSTEM_PROMPT = """\
Summarize one stored conversation episode into a prompt-safe character-identity
evidence card. Return exactly one JSON object with these keys:
decontextualized_event, character_cognition_summary,
visible_self_expression_summary.

Use abstract, detail-free language. Preserve only durable character-owned
meaning when it is clearly present. Never include names, participant identity,
quotes, intimate or private facts, channel or platform details, identifiers,
or instructions. If the episode does not establish durable character-owned
identity meaning, say so explicitly in the summaries. Each value must be at
most 400 characters.
"""


def _parser() -> argparse.ArgumentParser:
    """Build the recovery command-line parser."""

    parser = argparse.ArgumentParser(
        description="Recover Asuna identity-growth replay inputs.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=EXPECTED_CASE_COUNT)
    parser.add_argument("--force", action="store_true")
    return parser


def _load_json(path: Path) -> Any:
    """Load one local diagnostic artifact."""

    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    """Write one generated diagnostic artifact."""

    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _json_objects(text: str) -> list[dict[str, object]]:
    """Extract complete JSON objects embedded in one protected prompt."""

    decoder = json.JSONDecoder()
    objects: list[dict[str, object]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def _proposal_inputs_by_trace(
    trace_steps: list[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Extract exact proposal envelopes from retained V2 trace prompts."""

    result: dict[str, list[dict[str, object]]] = defaultdict(list)
    for step in trace_steps:
        trace_id = str(step.get("trace_id") or "").strip()
        if not trace_id:
            continue
        for message in step.get("raw_messages", []):
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            for candidate in _json_objects(content):
                if candidate.get("schema_version") != (
                    models.IDENTITY_PROPOSAL_INPUT_SCHEMA_VERSION
                ):
                    continue
                required_keys = {
                    "schema_version",
                    "current_identity",
                    "evidence_cards",
                    "current_candidates",
                    "allowed_paths",
                }
                if set(candidate) != required_keys:
                    continue
                encoded = json.dumps(
                    candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if encoded not in {
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for item in result[trace_id]
                }:
                    result[trace_id].append(candidate)
    return result


def _episode_runs(runs_document: Mapping[str, object]) -> list[dict[str, object]]:
    """Return the 185 episode runs in manifest order."""

    raw_runs = runs_document.get("documents")
    if not isinstance(raw_runs, list):
        raise ValueError("growth run export documents must be a list")
    runs = [
        dict(row)
        for row in raw_runs
        if isinstance(row, Mapping) and row.get("run_kind") == "episode"
    ]
    runs.sort(key=lambda row: str(row.get("started_at") or ""))
    if len(runs) != EXPECTED_CASE_COUNT:
        raise ValueError("growth run export must contain 185 episodes")
    return runs


def _timestamp_text(value: object) -> str:
    """Normalize one exported timestamp into ISO-8601 text."""

    if isinstance(value, Mapping) and "$date" in value:
        value = value["$date"]
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def _root_metadata(
    root_id: str,
    history_by_root: Mapping[str, list[Mapping[str, object]]],
    *,
    fallback_timestamp: object,
) -> dict[str, str]:
    """Derive non-sensitive provenance metadata for one root."""

    rows = history_by_root.get(root_id, [])
    timestamps = [
        _timestamp_text(row.get("timestamp"))
        for row in rows
        if row.get("timestamp")
    ]
    captured_at = min(timestamps) if timestamps else _timestamp_text(
        fallback_timestamp
    )
    channel_types = {
        str(row.get("channel_type") or "").strip()
        for row in rows
    }
    scope_kind = "group" if "group" in channel_types else "private"
    return {
        "captured_at": captured_at,
        "character_local_date": captured_at[:10],
        "scope_kind": scope_kind,
    }


def _generic_card() -> dict[str, str]:
    """Return an honest fallback when semantic reconstruction is unavailable."""

    return {
        "decontextualized_event": (
            "The retained episode does not establish a durable character-owned "
            "identity change."
        ),
        "character_cognition_summary": (
            "No durable self-concept conclusion is available from the retained "
            "evidence."
        ),
        "visible_self_expression_summary": (
            "No direct durable self-definition is available from the retained "
            "evidence."
        ),
    }


def _safe_summary(parsed: object) -> dict[str, str] | None:
    """Accept only bounded, detail-free summarizer output."""

    if not isinstance(parsed, Mapping):
        return None
    expected = {
        "decontextualized_event",
        "character_cognition_summary",
        "visible_self_expression_summary",
    }
    if set(parsed) != expected:
        return None
    result: dict[str, str] = {}
    for key in sorted(expected):
        value = parsed.get(key)
        if not isinstance(value, str):
            return None
        clean = " ".join(value.split())
        if len(clean) > SUMMARY_TEXT_LIMIT:
            return None
        if FORBIDDEN_SUMMARY_PATTERN.search(clean):
            return None
        result[key] = clean
    if not result["decontextualized_event"]:
        return None
    return result


def _transcript_for_root(
    root_id: str,
    history_by_root: Mapping[str, list[Mapping[str, object]]],
) -> str:
    """Build a bounded, identifier-free summarizer input."""

    rows = history_by_root.get(root_id, [])
    parts: list[str] = []
    for row in rows:
        role = "character" if row.get("role") == "assistant" else "participant"
        body = str(row.get("body_text") or "").strip()
        if body:
            parts.append(f"{role}: {body[:1200]}")
    transcript = "\n".join(parts)
    return transcript[:12000]


async def _summarize_root(
    root_id: str,
    history_by_root: Mapping[str, list[Mapping[str, object]]],
) -> tuple[dict[str, str], str]:
    """Summarize one root and return the card plus disposition."""

    transcript = _transcript_for_root(root_id, history_by_root)
    if not transcript:
        return _generic_card(), "fallback_no_history"
    response = await llm._identity_llm.ainvoke(
        [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content="EPISODE TRANSCRIPT:\n" + transcript),
        ],
        config=llm._identity_llm_config,
    )
    parsed = parse_llm_json_output(
        response.content,
        deterministic_only=True,
    )
    safe_summary = _safe_summary(parsed)
    if safe_summary is None:
        return _generic_card(), "fallback_invalid_summary"
    return safe_summary, "llm_summary"


def _revision_identities(
    revisions_document: Mapping[str, object],
) -> dict[int, dict[str, object]]:
    """Index exported effective identities by revision number."""

    raw_revisions = revisions_document.get("documents")
    if not isinstance(raw_revisions, list):
        raise ValueError("identity revision export documents must be a list")
    result: dict[int, dict[str, object]] = {}
    for row in raw_revisions:
        if not isinstance(row, Mapping):
            continue
        revision_number = row.get("revision_number")
        identity = row.get("effective_identity")
        if (
            isinstance(revision_number, int)
            and not isinstance(revision_number, bool)
            and isinstance(identity, Mapping)
        ):
            result[revision_number] = deepcopy(dict(identity))
    return result


def _replace_handles(value: object, replacements: Mapping[str, str]) -> object:
    """Replace protected handles in an exact envelope with local aliases."""

    if isinstance(value, str):
        result = value
        for source, target in replacements.items():
            result = result.replace(source, target)
        return result
    if isinstance(value, list):
        return [_replace_handles(item, replacements) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _replace_handles(item, replacements)
            for key, item in value.items()
        }
    return value


def _localize_input(
    source_input: Mapping[str, object],
    *,
    case_index: int,
) -> dict[str, object]:
    """Localize repository handles and preserve the exact semantic envelope."""

    localized = deepcopy(dict(source_input))
    cards = localized.get("evidence_cards")
    candidates = localized.get("current_candidates")
    if not isinstance(cards, list) or not isinstance(candidates, list):
        raise ValueError("proposal input cards and candidates must be lists")
    replacements: dict[str, str] = {}
    for index, card in enumerate(cards, start=1):
        if not isinstance(card, Mapping):
            raise ValueError("proposal evidence card must be an object")
        source_id = str(card.get("evidence_ref_id") or "").strip()
        if not source_id:
            raise ValueError("proposal evidence card needs an evidence id")
        replacements[source_id] = f"evidence-{case_index:03d}-{index:02d}"
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, Mapping):
            raise ValueError("proposal candidate must be an object")
        source_id = str(candidate.get("candidate_id") or "").strip()
        if not source_id:
            raise ValueError("proposal candidate needs a candidate id")
        replacements[source_id] = f"candidate-{case_index:03d}-{index:02d}"
    localized = _replace_handles(localized, replacements)
    localized["allowed_paths"] = sorted(models.ALLOWED_IDENTITY_PATHS)
    return localized


def _generated_input(
    *,
    case_index: int,
    case: Mapping[str, object],
    run: Mapping[str, object],
    identity: Mapping[str, object],
    cards: list[dict[str, object]],
) -> dict[str, object]:
    """Build one V2 input from a current identity and safe cards."""

    return {
        "schema_version": models.IDENTITY_PROPOSAL_INPUT_SCHEMA_VERSION,
        "current_identity": deepcopy(dict(identity)),
        "evidence_cards": cards,
        "current_candidates": [],
        "allowed_paths": sorted(models.ALLOWED_IDENTITY_PATHS),
    }


def _card_from_summary(
    *,
    case_index: int,
    root_index: int,
    metadata: Mapping[str, str],
    summary: Mapping[str, str],
) -> dict[str, object]:
    """Build one local prompt-safe evidence card."""

    evidence_ref_id = f"evidence-{case_index:03d}-{root_index:02d}"
    return {
        "schema_version": models.IDENTITY_EVIDENCE_CARD_SCHEMA_VERSION,
        "evidence_ref_id": evidence_ref_id,
        "source_kind": "settled_episode",
        "character_local_date": metadata["character_local_date"],
        "scope_kind": metadata["scope_kind"],
        "decontextualized_event": summary["decontextualized_event"],
        "character_cognition_summary": summary[
            "character_cognition_summary"
        ],
        "visible_self_expression_summary": summary[
            "visible_self_expression_summary"
        ],
    }


def _audit_default() -> dict[str, object]:
    """Return the recovery audit base document."""

    return {
        "schema_version": "asuna_identity_growth_replay_recovery_audit.v1",
        "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "model_route": llm._identity_llm_config.route_name,
        "model": llm._identity_llm_config.model,
        "processed_cases": [],
        "summary_counts": {
            "exact_trace_envelope": 0,
            "llm_summary": 0,
            "fallback_invalid_summary": 0,
            "fallback_no_history": 0,
            "no_evidence_sentinel": 0,
        },
    }


async def _recover(args: argparse.Namespace) -> None:
    """Recover the selected manifest slice."""

    manifest = _load_json(MANIFEST_PATH)
    runs = _episode_runs(_load_json(RUNS_PATH))
    history_document = _load_json(HISTORY_PATH)
    trace_document = _load_json(TRACE_STEPS_PATH)
    revisions = _revision_identities(_load_json(REVISIONS_PATH))
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError("replay manifest must contain 185 cases")
    history_rows = history_document.get("documents")
    if not isinstance(history_rows, list):
        raise ValueError("history export documents must be a list")
    history_by_root: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in history_rows:
        if isinstance(row, Mapping) and row.get("source_episode_id"):
            history_by_root[str(row["source_episode_id"])].append(row)
    trace_rows = trace_document.get("documents")
    if not isinstance(trace_rows, list):
        raise ValueError("trace export documents must be a list")
    exact_by_trace = _proposal_inputs_by_trace(trace_rows)

    audit = _load_json(AUDIT_PATH) if AUDIT_PATH.exists() else _audit_default()
    if not isinstance(audit, dict):
        raise ValueError("recovery audit must be an object")
    processed_cases = audit.setdefault("processed_cases", [])
    summary_counts = audit.setdefault("summary_counts", {})
    if not isinstance(processed_cases, list) or not isinstance(
        summary_counts,
        dict,
    ):
        raise ValueError("recovery audit shape is invalid")

    selected_end = min(args.offset + args.limit, EXPECTED_CASE_COUNT)
    if args.offset < 0 or args.limit < 0:
        raise ValueError("offset and limit must be non-negative")
    for case_position in range(args.offset, selected_end):
        raw_case = cases[case_position]
        if not isinstance(raw_case, dict):
            raise ValueError("replay case must be an object")
        case = raw_case
        if (
            not args.force
            and isinstance(case.get("replay_input"), dict)
        ):
            continue
        run = runs[case_position]
        base_revision = case.get("base_revision_number")
        if not isinstance(base_revision, int) or isinstance(base_revision, bool):
            raise ValueError("case base revision must be an integer")
        identity = revisions.get(base_revision)
        if identity is None:
            raise ValueError("case base revision has no exported identity")

        root_ids = [
            str(root_id)
            for root_id in run.get("root_episode_ids", [])
            if str(root_id).strip()
        ]
        exact_objects: list[dict[str, object]] = []
        for root_id in root_ids:
            exact_objects.extend(exact_by_trace.get(root_id, []))
        unique_objects = {
            json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ): item
            for item in exact_objects
            if item.get("schema_version") == (
                models.IDENTITY_PROPOSAL_INPUT_SCHEMA_VERSION
            )
        }
        source_by_root: list[dict[str, object] | None] = []
        for root_id in root_ids:
            root_objects = exact_by_trace.get(root_id, [])
            source_by_root.append(root_objects[0] if root_objects else None)

        source_cards: list[dict[str, object]] = []
        source_identity: Mapping[str, object] = identity
        source_candidates: list[Mapping[str, object]] = []
        recovery_kinds: list[str] = []
        if unique_objects:
            first_object = next(iter(unique_objects.values()))
            raw_identity = first_object.get("current_identity")
            if isinstance(raw_identity, Mapping):
                source_identity = raw_identity
            raw_candidates = first_object.get("current_candidates")
            if isinstance(raw_candidates, list):
                source_candidates = [
                    item for item in raw_candidates
                    if isinstance(item, Mapping)
                ]

        for root_index, root_id in enumerate(root_ids, start=1):
            source_object = source_by_root[root_index - 1]
            raw_cards = (
                source_object.get("evidence_cards", [])
                if source_object is not None
                else []
            )
            if isinstance(raw_cards, list) and raw_cards:
                for raw_card in raw_cards:
                    if isinstance(raw_card, Mapping):
                        source_cards.append(dict(raw_card))
                recovery_kinds.append("exact_trace_envelope")
                continue
            metadata = _root_metadata(
                root_id,
                history_by_root,
                fallback_timestamp=run.get("started_at"),
            )
            if not root_id:
                summary = _generic_card()
                kind = "no_evidence_sentinel"
            else:
                summary, kind = await _summarize_root(
                    root_id,
                    history_by_root,
                )
            source_cards.append(
                _card_from_summary(
                    case_index=case_position + 1,
                    root_index=root_index,
                    metadata=metadata,
                    summary=summary,
                )
            )
            recovery_kinds.append(kind)

        if not source_cards:
            metadata = _root_metadata(
                "",
                history_by_root,
                fallback_timestamp=run.get("started_at"),
            )
            source_cards.append(
                _card_from_summary(
                    case_index=case_position + 1,
                    root_index=1,
                    metadata=metadata,
                    summary=_generic_card(),
                )
            )
            recovery_kinds.append("no_evidence_sentinel")

        source_input = {
            "schema_version": models.IDENTITY_PROPOSAL_INPUT_SCHEMA_VERSION,
            "current_identity": deepcopy(dict(source_identity)),
            "evidence_cards": source_cards,
            "current_candidates": [deepcopy(dict(item)) for item in source_candidates],
            "allowed_paths": sorted(models.ALLOWED_IDENTITY_PATHS),
        }
        localized = _localize_input(
            source_input,
            case_index=case_position + 1,
        )
        case["replay_input"] = localized
        if any(kind == "llm_summary" for kind in recovery_kinds):
            case["reconstruction_note"] = (
                "Prompt-safe evidence recovered from protected conversation "
                "history through bounded detail-free semantic summarization; "
                "raw transcript content is excluded."
            )
        elif any(kind == "exact_trace_envelope" for kind in recovery_kinds):
            case["reconstruction_note"] = (
                "Prompt-safe V2 proposal envelope recovered from protected "
                "trace evidence; repository handles were replaced with local "
                "aliases."
            )
        else:
            case["reconstruction_note"] = (
                "No eligible root evidence was retained; a deterministic "
                "no-evidence sentinel is used only to exercise the bounded "
                "contract gate."
            )

        case_id = str(case.get("case_id") or f"case-{case_position + 1:03d}")
        processed_cases[:] = [
            item
            for item in processed_cases
            if isinstance(item, Mapping) and item.get("case_id") != case_id
        ]
        processed_cases.append({
            "case_id": case_id,
            "source_fidelity": case.get("source_fidelity"),
            "root_count": len(root_ids),
            "recovery_kinds": sorted(recovery_kinds),
            "card_count": len(source_cards),
            "input_sha256": hashlib.sha256(
                json.dumps(
                    localized,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        })
        for kind in recovery_kinds:
            summary_counts[kind] = int(summary_counts.get(kind, 0)) + 1
        _write_json(MANIFEST_PATH, manifest)
        _write_json(AUDIT_PATH, audit)
        print(
            f"recovered {case_id} cards={len(source_cards)} "
            f"kinds={','.join(sorted(set(recovery_kinds)))}",
            flush=True,
        )
    await llm._identity_llm.aclose()


def main() -> None:
    """Run the recovery command."""

    args = _parser().parse_args()
    asyncio.run(_recover(args))


if __name__ == "__main__":
    main()
