"""Compact supervisor-owned memory for one coding-agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MAX_LEDGER_EVENTS = 16
MAX_LEDGER_FACTS = 12
MAX_LEDGER_ARTIFACTS = 24
MAX_LEDGER_TEXT_CHARS = 800


@dataclass
class CodingSupervisorWorkLedger:
    """Track compact cross-workflow facts without growing role prompts."""

    goal: str
    events: list[dict[str, object]] = field(default_factory=list)
    supervisor_facts: list[dict[str, object]] = field(default_factory=list)
    external_evidence: list[dict[str, object]] = field(default_factory=list)
    generated_artifacts: list[dict[str, object]] = field(default_factory=list)

    def record_writing_attempt(
        self,
        *,
        attempt_index: int,
        writing_result: dict[str, object],
    ) -> None:
        """Record one writing workflow outcome for later supervisor projection."""

        event = {
            "kind": "writing_attempt",
            "attempt_index": attempt_index,
            "status": _bounded_text(writing_result.get("status")),
            "created_file_count": _list_count(writing_result.get("created_files")),
            "patch_artifact_count": _list_count(
                writing_result.get("patch_artifacts")
            ),
            "reading_request_count": _list_count(
                writing_result.get("reading_requests")
            ),
            "external_request_count": _list_count(
                writing_result.get("external_evidence_requests")
            ),
            "limitations": _bounded_text_list(
                writing_result.get("limitations"),
                limit=4,
            ),
        }
        self._append_event(event)

    def record_external_evidence(
        self,
        evidence_rows: list[dict[str, object]],
    ) -> None:
        """Store compact public evidence returned by supervisor mediation."""

        for row in evidence_rows:
            self.external_evidence.append(_compact_external_evidence(row))
        self.external_evidence = self.external_evidence[-MAX_LEDGER_FACTS:]
        event = {
            "kind": "external_evidence",
            "resolved_count": sum(
                1 for row in evidence_rows if bool(row.get("resolved"))
            ),
            "total_count": len(evidence_rows),
        }
        self._append_event(event)

    def record_supervisor_fact(self, fact: dict[str, object]) -> None:
        """Store one compact fact resolved by the top-level supervisor."""

        self.supervisor_facts.append(_compact_supervisor_fact(fact))
        self.supervisor_facts = self.supervisor_facts[-MAX_LEDGER_FACTS:]
        event = {
            "kind": "supervisor_fact",
            "request_id": _bounded_text(fact.get("request_id")),
            "fact_kind": _bounded_text(fact.get("kind")),
            "resolved": bool(fact.get("resolved")),
        }
        self._append_event(event)

    def record_generated_artifacts(
        self,
        artifacts: object,
    ) -> None:
        """Preserve generated artifacts across supervisor-mediated handoffs."""

        if not isinstance(artifacts, list):
            return
        for artifact in artifacts:
            compact_artifact = _generated_artifact(artifact)
            if compact_artifact is None:
                continue
            self._upsert_generated_artifact(compact_artifact)
        self.generated_artifacts = self.generated_artifacts[-MAX_LEDGER_ARTIFACTS:]
        event = {
            "kind": "generated_artifacts",
            "artifact_count": len(self.generated_artifacts),
        }
        self._append_event(event)

    def projection(self) -> dict[str, object]:
        """Return bounded state suitable for a downstream role prompt."""

        projection = {
            "goal": _bounded_text(self.goal),
            "events": self.events[-MAX_LEDGER_EVENTS:],
            "supervisor_facts": self.supervisor_facts[-MAX_LEDGER_FACTS:],
            "generated_artifacts": [
                _generated_artifact_summary(artifact)
                for artifact in self.generated_artifacts[-MAX_LEDGER_ARTIFACTS:]
            ],
            "external_evidence_count": len(self.external_evidence),
        }
        return projection

    def _append_event(self, event: dict[str, object]) -> None:
        self.events.append(event)
        self.events = self.events[-MAX_LEDGER_EVENTS:]

    def _upsert_generated_artifact(self, artifact: dict[str, object]) -> None:
        next_artifacts: list[dict[str, object]] = []
        for existing in self.generated_artifacts:
            if (
                existing["artifact_id"] == artifact["artifact_id"]
                or existing["path"] == artifact["path"]
            ):
                continue
            next_artifacts.append(existing)
        next_artifacts.append(artifact)
        self.generated_artifacts = next_artifacts


def _compact_external_evidence(row: dict[str, object]) -> dict[str, object]:
    compact_row = {
        "request_id": _bounded_text(row.get("request_id")),
        "task": _bounded_text(row.get("task")),
        "resolved": bool(row.get("resolved")),
        "result": _bounded_text(row.get("result")),
        "limitation": _bounded_text(row.get("limitation")),
    }
    return compact_row


def _compact_supervisor_fact(fact: dict[str, object]) -> dict[str, object]:
    compact_fact = {
        "request_id": _bounded_text(fact.get("request_id")),
        "kind": _bounded_text(fact.get("kind")),
        "task": _bounded_text(fact.get("task")),
        "resolved": bool(fact.get("resolved")),
        "result": _bounded_text(fact.get("result")),
        "limitation": _bounded_text(fact.get("limitation")),
    }
    return compact_fact


def _generated_artifact(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    artifact_id = _bounded_text(value.get("artifact_id"))
    path = _bounded_text(value.get("path"))
    content = value.get("content")
    if not artifact_id or not path or not isinstance(content, str):
        return None
    artifact = {
        "artifact_id": artifact_id,
        "file_label": _bounded_text(value.get("file_label")) or artifact_id,
        "file_kind": _bounded_text(value.get("file_kind")),
        "content_format": _bounded_text(value.get("content_format")),
        "path": path,
        "content": content,
        "purpose": _bounded_text(value.get("purpose")),
    }
    return artifact


def _generated_artifact_summary(
    artifact: dict[str, object],
) -> dict[str, object]:
    summary = {
        "artifact_id": _bounded_text(artifact.get("artifact_id")),
        "path": _bounded_text(artifact.get("path")),
        "file_kind": _bounded_text(artifact.get("file_kind")),
        "content_format": _bounded_text(artifact.get("content_format")),
        "purpose": _bounded_text(artifact.get("purpose")),
    }
    return summary


def _bounded_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.strip().split())
    bounded_text = text[:MAX_LEDGER_TEXT_CHARS].rstrip()
    return bounded_text


def _bounded_text_list(value: object, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []

    texts: list[str] = []
    for item in value:
        text = _bounded_text(item)
        if text:
            texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def _list_count(value: object) -> int:
    if not isinstance(value, list):
        return 0
    return len(value)
