"""Strict Brain-owned wire contracts for cognition observations."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

COGNITION_OBSERVATION_SCHEMA_VERSION = "cognition_run_observation.v1"
COGNITION_OBSERVATION_DISCLOSURE_POLICY = (
    "approved_cognition_observation.v1"
)
COGNITION_OBSERVATION_MAX_NODES = 64
COGNITION_OBSERVATION_MAX_EDGES = 96
COGNITION_OBSERVATION_MAX_SECTIONS = 96
COGNITION_OBSERVATION_MAX_SECTION_REFS = 12
COGNITION_OBSERVATION_MAX_FIELDS = 24
COGNITION_OBSERVATION_MAX_RECORDS = 24
COGNITION_OBSERVATION_MAX_RECORD_FIELDS = 16
COGNITION_OBSERVATION_MAX_LABEL_CHARS = 80
COGNITION_OBSERVATION_MAX_NODE_SUMMARY_CHARS = 180
COGNITION_OBSERVATION_MAX_SECTION_SUMMARY_CHARS = 600
COGNITION_OBSERVATION_MAX_SCALAR_CHARS = 4000
COGNITION_OBSERVATION_MAX_LIST_ITEMS = 24
COGNITION_OBSERVATION_MAX_LIST_ITEM_CHARS = 2000
COGNITION_OBSERVATION_MAX_PAYLOAD_CHARS = 131072
COGNITION_OBSERVATION_MAX_CORRELATION_CHARS = 120

COGNITION_OBSERVATION_EXCLUSIONS = [
    "prompt",
    "raw_model_output",
    "embedding",
    "raw_message",
    "message_envelope",
    "database_identifier",
    "adapter_identifier",
    "action_parameter",
    "handler_metadata",
    "worker_error_text",
]

_IDENTIFIER_PATTERN = r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$"
_KEY_PATTERN = r"^[a-z][a-z0-9_]*$"
_OBSERVATION_STATUS_VALUES = {
    "completed",
    "empty",
    "skipped",
    "failed",
    "partial",
    "not_reported",
}
_STATUS_PRIORITY = (
    "failed",
    "partial",
    "completed",
    "empty",
    "skipped",
    "not_reported",
)

ObservationRunKind = Literal["live_turn", "self_cognition"]
ObservationStatus = Literal[
    "completed",
    "failed",
    "partial",
]
ObservationSectionStatus = Literal[
    "completed",
    "empty",
    "skipped",
    "failed",
    "partial",
    "not_reported",
]
ObservationPresentation = Literal["fields", "records"]
ObservationEdgeKind = Literal["sequence", "reference"]

ObservationIdentifier: TypeAlias = Annotated[
    StrictStr,
    StringConstraints(
        min_length=1,
        max_length=COGNITION_OBSERVATION_MAX_LABEL_CHARS,
        pattern=_IDENTIFIER_PATTERN,
    ),
]
ObservationKey: TypeAlias = Annotated[
    StrictStr,
    StringConstraints(
        min_length=1,
        max_length=COGNITION_OBSERVATION_MAX_LABEL_CHARS,
        pattern=_KEY_PATTERN,
    ),
]
ObservationLabel: TypeAlias = Annotated[
    StrictStr,
    StringConstraints(
        min_length=1,
        max_length=COGNITION_OBSERVATION_MAX_LABEL_CHARS,
    ),
]
ObservationSummary: TypeAlias = Annotated[
    StrictStr,
    StringConstraints(
        max_length=COGNITION_OBSERVATION_MAX_SECTION_SUMMARY_CHARS,
    ),
]
ObservationNodeSummary: TypeAlias = Annotated[
    StrictStr,
    StringConstraints(max_length=COGNITION_OBSERVATION_MAX_NODE_SUMMARY_CHARS),
]
ObservationCorrelationValue: TypeAlias = Annotated[
    StrictStr,
    StringConstraints(
        min_length=1,
        max_length=COGNITION_OBSERVATION_MAX_CORRELATION_CHARS,
    ),
]

ObservationScalar: TypeAlias = (
    StrictStr | StrictInt | StrictFloat | StrictBool | None
)
ObservationValue: TypeAlias = ObservationScalar | list[ObservationScalar]


class CognitionObservationCorrelationV1(BaseModel):
    """Optional protected correlation references for one observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: ObservationCorrelationValue | None = None
    llm_trace_id: ObservationCorrelationValue | None = None
    cognition_invocation_id: ObservationCorrelationValue | None = None
    source_calendar_run_id: ObservationCorrelationValue | None = None


class CognitionObservationDisclosureV1(BaseModel):
    """Stable disclosure policy attached to every published observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: Literal["approved_cognition_observation.v1"]
    excluded: list[ObservationKey] = Field(
        min_length=len(COGNITION_OBSERVATION_EXCLUSIONS),
        max_length=len(COGNITION_OBSERVATION_EXCLUSIONS),
    )

    @model_validator(mode="after")
    def validate_exclusions(self) -> CognitionObservationDisclosureV1:
        """Require the exact ordered exclusion vocabulary."""

        if self.excluded != COGNITION_OBSERVATION_EXCLUSIONS:
            raise ValueError("observation disclosure exclusions are not exact")
        return self


class CognitionObservationFieldV1(BaseModel):
    """One ordered scalar or flat-scalar-list detail field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: ObservationKey
    label: ObservationLabel
    value: ObservationValue

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, value: object) -> object:
        """Reject mappings, nested lists, non-finite numbers, and coercion."""

        _validate_observation_value(value)
        return value


class CognitionObservationRecordV1(BaseModel):
    """One ordered semantic record in a records section."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: ObservationKey
    label: ObservationLabel
    summary: ObservationSummary
    fields: list[CognitionObservationFieldV1] = Field(
        max_length=COGNITION_OBSERVATION_MAX_RECORD_FIELDS,
    )

    @model_validator(mode="after")
    def validate_field_keys(self) -> CognitionObservationRecordV1:
        """Keep record field keys unique and producer ordered."""

        keys = [field.key for field in self.fields]
        if len(keys) != len(set(keys)):
            raise ValueError("record field keys must be unique")
        return self


class CognitionObservationSectionV1(BaseModel):
    """One catalog section containing ordered fields and/or records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: ObservationIdentifier
    label: ObservationLabel
    category: ObservationKey
    presentation: ObservationPresentation
    status: ObservationSectionStatus
    summary: ObservationSummary
    fields: list[CognitionObservationFieldV1] = Field(
        max_length=COGNITION_OBSERVATION_MAX_FIELDS,
    )
    records: list[CognitionObservationRecordV1] = Field(
        max_length=COGNITION_OBSERVATION_MAX_RECORDS,
    )
    reported_record_count: Annotated[StrictInt, Field(ge=0)]
    displayed_record_count: Annotated[StrictInt, Field(ge=0)]
    truncated: StrictBool

    @model_validator(mode="after")
    def validate_shape_and_counts(self) -> CognitionObservationSectionV1:
        """Enforce presentation, uniqueness, and truthful count invariants."""

        field_keys = [field.key for field in self.fields]
        if len(field_keys) != len(set(field_keys)):
            raise ValueError("section field keys must be unique")
        record_keys = [record.key for record in self.records]
        if len(record_keys) != len(set(record_keys)):
            raise ValueError("section record keys must be unique")
        expected_record_keys = [
            f"item_{index:02d}" for index in range(1, len(self.records) + 1)
        ]
        if record_keys != expected_record_keys:
            raise ValueError(
                "section record keys must be item_XX in source order"
            )
        if self.displayed_record_count != len(self.records):
            raise ValueError("displayed record count is not truthful")
        if self.reported_record_count < self.displayed_record_count:
            raise ValueError("reported record count is below displayed count")
        if self.truncated != (
            self.reported_record_count > self.displayed_record_count
        ):
            raise ValueError("section truncation flag is not truthful")
        if self.presentation == "fields" and (
            self.records
            or self.reported_record_count
            or self.displayed_record_count
            or self.truncated
        ):
            raise ValueError("fields section cannot contain records")
        return self


class CognitionObservationNodeV1(BaseModel):
    """One producer-owned graph node referencing catalog sections."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: ObservationIdentifier
    label: ObservationLabel
    stage: ObservationLabel
    lane: ObservationLabel
    column: Annotated[StrictInt, Field(ge=1, le=64)]
    category: ObservationKey
    status: ObservationSectionStatus
    summary: ObservationNodeSummary
    section_refs: list[ObservationIdentifier] = Field(
        min_length=1,
        max_length=COGNITION_OBSERVATION_MAX_SECTION_REFS,
    )

    @model_validator(mode="after")
    def validate_refs(self) -> CognitionObservationNodeV1:
        """Require unique references while preserving producer order."""

        if len(self.section_refs) != len(set(self.section_refs)):
            raise ValueError("node section references must be unique")
        return self


class CognitionObservationEdgeV1(BaseModel):
    """One sequence or reference edge between producer-owned nodes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: ObservationIdentifier
    target: ObservationIdentifier
    kind: ObservationEdgeKind
    label: Annotated[
        StrictStr,
        StringConstraints(max_length=COGNITION_OBSERVATION_MAX_LABEL_CHARS),
    ]


class CognitionRunObservationV1(BaseModel):
    """Validated terminal cognition observation published by Brain service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["cognition_run_observation.v1"]
    run_kind: ObservationRunKind
    status: ObservationStatus
    generated_at: datetime
    correlation: CognitionObservationCorrelationV1
    sections: list[CognitionObservationSectionV1] = Field(
        min_length=1,
        max_length=COGNITION_OBSERVATION_MAX_SECTIONS,
    )
    nodes: list[CognitionObservationNodeV1] = Field(
        min_length=1,
        max_length=COGNITION_OBSERVATION_MAX_NODES,
    )
    edges: list[CognitionObservationEdgeV1] = Field(
        max_length=COGNITION_OBSERVATION_MAX_EDGES,
    )
    disclosure: CognitionObservationDisclosureV1

    @field_validator("generated_at")
    @classmethod
    def normalize_generated_at(cls, value: datetime) -> datetime:
        """Require an aware instant and normalize it to UTC."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_serializer("generated_at")
    def serialize_generated_at(self, value: datetime) -> str:
        """Serialize UTC timestamps with the canonical terminal ``Z``."""

        return value.astimezone(timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        )

    @model_validator(mode="after")
    def validate_graph_and_catalog(self) -> CognitionRunObservationV1:
        """Enforce references, catalog presence, node summaries, and status."""

        section_by_id = {section.section_id: section for section in self.sections}
        if len(section_by_id) != len(self.sections):
            raise ValueError("section identifiers must be unique")
        node_by_id = {node.node_id: node for node in self.nodes}
        if len(node_by_id) != len(self.nodes):
            raise ValueError("node identifiers must be unique")
        for node in self.nodes:
            if any(ref not in section_by_id for ref in node.section_refs):
                raise ValueError("node references an unknown section")
            expected_status = _aggregate_status(
                section_by_id[ref].status for ref in node.section_refs
            )
            if node.status != expected_status:
                raise ValueError("node status does not aggregate its sections")
            expected_summary = next(
                (
                    section_by_id[ref].summary
                    for ref in node.section_refs
                    if section_by_id[ref].summary
                ),
                node.status,
            )
            if node.summary != expected_summary[:
                COGNITION_OBSERVATION_MAX_NODE_SUMMARY_CHARS
            ]:
                raise ValueError("node summary does not follow section order")
        for edge in self.edges:
            if edge.source not in node_by_id or edge.target not in node_by_id:
                raise ValueError("edge endpoint is unknown")
        _validate_required_sections(self.run_kind, section_by_id)
        has_partial_component = any(
            status in {"failed", "partial"}
            for status in (
                *(section.status for section in self.sections),
                *(node.status for node in self.nodes),
            )
        )
        if self.status == "completed" and has_partial_component:
            raise ValueError("completed observation contains partial component")
        if self.status == "partial" and not has_partial_component:
            raise ValueError("partial observation has no partial component")
        serialized = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(serialized) > COGNITION_OBSERVATION_MAX_PAYLOAD_CHARS:
            raise ValueError("observation payload is over budget")
        return self


def _validate_observation_value(value: object) -> None:
    """Validate a wire scalar or one-level scalar list without coercion."""

    if isinstance(value, list):
        if len(value) > COGNITION_OBSERVATION_MAX_LIST_ITEMS:
            raise ValueError("observation scalar list is over budget")
        for item in value:
            _validate_observation_scalar(item, list_item=True)
        return
    _validate_observation_scalar(value, list_item=False)


def _validate_observation_scalar(value: object, *, list_item: bool) -> None:
    """Validate one strict JSON scalar."""

    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("observation numbers must be finite")
        return
    if isinstance(value, str):
        maximum = (
            COGNITION_OBSERVATION_MAX_LIST_ITEM_CHARS
            if list_item
            else COGNITION_OBSERVATION_MAX_SCALAR_CHARS
        )
        if len(value) > maximum:
            raise ValueError("observation text is over budget")
        return
    raise ValueError("observation value must be a strict scalar or flat list")


def _aggregate_status(statuses: object) -> ObservationSectionStatus:
    """Aggregate section dispositions in the fixed priority order."""

    values = set(statuses)
    for status in _STATUS_PRIORITY:
        if status in values:
            return status  # type: ignore[return-value]
    return "not_reported"


def _validate_required_sections(
    run_kind: ObservationRunKind,
    sections: dict[str, CognitionObservationSectionV1],
) -> None:
    """Require exactly the run-kind base catalog without cross-kind sections."""

    live_sections = {
        "input.turn",
        "decision.response",
        "cognition.appraisals",
        "cognition.goal",
        "cognition.response_plan",
        "cognition.affect",
        "reasoning.subjective",
        "reasoning.context_consumption",
        "evidence.memory",
        "evidence.shared_memory_prewarm",
        "context.conversation_progress",
        "context.public_group_scene",
        "action.requests",
        "action.results",
        "action.continuation",
        "surface.visual_directives",
        "surface.visible_messages",
    }
    self_sections = {
        "cognition.appraisals",
        "cognition.goal",
        "cognition.response_plan",
        "cognition.affect",
        "reasoning.subjective",
        "reasoning.context_consumption",
        "evidence.memory",
        "evidence.shared_memory_prewarm",
        "context.conversation_progress",
        "context.public_group_scene",
        "action.requests",
        "action.results",
        "action.continuation",
        "surface.visual_directives",
        "surface.visible_messages",
        "self.source",
        "self.route",
        "self.consolidation",
    }
    required = live_sections if run_kind == "live_turn" else self_sections
    forbidden = self_sections - live_sections if run_kind == "live_turn" else (
        live_sections - self_sections
    )
    section_ids = set(sections)
    if not required <= section_ids:
        missing = sorted(required - section_ids)
        raise ValueError(f"observation is missing required sections: {missing}")
    if forbidden & section_ids:
        unexpected = sorted(forbidden & section_ids)
        raise ValueError(f"observation contains wrong-run sections: {unexpected}")
