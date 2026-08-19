"""Effect-free capture utilities for governed cognition V2/V3 comparisons.

The harness calls a cognition facade directly with injected services. It owns
only protected test-artifact writes; connector commits, action and resolver
execution, text or visual surfaces, adapters, delivery, and schedulers are
outside this module's callable boundary.
"""

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
from typing import Any, Protocol, TypeVar

from langchain_core.messages import BaseMessage

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreInputV2,
    CognitionCoreOutputV2,
    CognitionCoreServicesV2,
    validate_cognition_core_input,
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.llm_interface.contracts import (
    LLMCallConfig,
    LLMResponse,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LIVE_CASE_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "cognition_core_v3_live_case_manifest.json"
)
DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "test_artifacts" / "cognition_core_v3"
TRIAL_ARTIFACT_SCHEMA = "cognition_v3_comparison_trial.v1"
PAIR_INVALIDATION_SCHEMA = "cognition_v3_matched_pair_invalidation.v1"
BASELINE_INDEX_SCHEMA = "cognition_v3_baseline_index.v1"
BASELINE_INDEX_PHASE = "sealed_gate_1"
ENGINE_NAMES = frozenset({"v2", "v3"})
TRIAL_INDICES = frozenset({1, 2, 3})
ELIGIBLE_RESULT = "eligible_semantic_result"
HARD_BOUNDARY_FAILURE = "hard_boundary_failure"
INVALID_NO_RESULT = "provider_or_harness_invalid_no_result"
RERUN_REASONS = frozenset({
    "provider_transport_no_result",
    "harness_invalid_no_result",
})
BASELINE_ARTIFACT_KINDS = frozenset({
    "baseline_governance",
    "defect_registry",
    "deterministic_test_report",
    "eligible_raw_trial",
    "invalidated_raw_attempt",
    "local_semantic_reset",
    "pair_invalidation",
    "production_data_extract",
    "semantic_reset_evidence",
    "v2_review",
})
BASELINE_INDEX_SUMMARY = {
    "case_count": 24,
    "eligible_v2_trial_count": 72,
    "invalidated_v2_attempt_count": 2,
    "v2_review_count": 72,
    "pair_invalidation_count": 2,
    "local_semantic_reset_count": 6,
    "production_data_extract_count": 0,
    "inherited_defect_count": 7,
    "eligible_hard_boundary_failure_count": 0,
}

OutputT = TypeVar("OutputT")


class EngineRunner(Protocol):
    """One async cognition engine facade accepted by the harness."""

    def __call__(
        self,
        input_payload: CognitionCoreInputV2,
        services: CognitionCoreServicesV2,
    ) -> Awaitable[CognitionCoreOutputV2]:
        """Return one complete cognition result without connector effects."""


class TrialAlreadySealedError(RuntimeError):
    """Raised when a caller attempts to replace a completed trial artifact."""


class TrialExecutionError(RuntimeError):
    """Raised after a failed engine call has been durably captured."""

    def __init__(self, message: str, *, artifact_path: Path) -> None:
        super().__init__(message)
        self.artifact_path = artifact_path


@dataclass(frozen=True)
class TrialIdentity:
    """Stable identity for one governed engine sample."""

    baseline_id: str
    case_id: str
    engine: str
    trial_index: int
    attempt_index: int = 1

    def validate(self) -> TrialIdentity:
        """Reject identities outside the fixed comparison protocol."""

        if not self.baseline_id or not _safe_component(self.baseline_id):
            raise ValueError("baseline_id must be a filesystem-safe value")
        if not self.case_id or not _safe_component(self.case_id):
            raise ValueError("case_id must be a filesystem-safe value")
        if self.engine not in ENGINE_NAMES:
            raise ValueError("engine must be v2 or v3")
        if self.trial_index not in TRIAL_INDICES:
            raise ValueError("trial_index must be 1, 2, or 3")
        if self.attempt_index not in {1, 2, 3}:
            raise ValueError("attempt_index must be 1, 2, or 3")
        return self

    @property
    def trial_id(self) -> str:
        """Return the immutable logical-trial identifier."""

        return (
            f"{self.baseline_id}:{self.case_id}:{self.engine}:"
            f"trial-{self.trial_index}"
        )


class CapturingLLMInvoker:
    """Delegate LLM calls while preserving protected raw comparison evidence."""

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
        """Capture one asynchronous request, response, and wall time."""

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
        """Capture one synchronous request, response, and wall time."""

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
        """Append one request record before the provider call starts."""

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
    """Load the frozen live-case manifest as one JSON object."""

    value = json.loads(LIVE_CASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("live-case manifest root must be an object")
    return value


def find_case_row(case_id: str) -> dict[str, Any]:
    """Return the single frozen manifest row for ``case_id``."""

    rows = load_live_case_manifest()["cases"]
    matches = [row for row in rows if row.get("case_id") == case_id]
    if len(matches) != 1:
        raise KeyError(f"expected one manifest row for {case_id!r}")
    return deepcopy(matches[0])


def render_case_input(manifest_row: Mapping[str, Any]) -> CognitionCoreInputV2:
    """Render an isolated input using only the frozen manifest row.

    This function reads no database, clock, random source, network service, or
    environment value. Validation and a deep copy are the only operations.
    """

    if "canonical_input" not in manifest_row:
        raise KeyError("manifest row is missing canonical_input")
    canonical_input = deepcopy(manifest_row["canonical_input"])
    validated = validate_cognition_core_input(canonical_input)
    return deepcopy(validated)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value with stable byte ordering."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 digest of a canonical JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    """Return whether ``value`` is one lowercase SHA-256 digest."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_index_file(root: Path, relative_path: object) -> Path:
    """Resolve one POSIX artifact path below its declared root."""

    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\\" in relative_path
    ):
        raise ValueError("baseline index path must be a non-empty POSIX path")
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("baseline index path escapes its declared root")
    resolved_root = root.resolve()
    resolved_path = (resolved_root / path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            "baseline index path escapes its declared root"
        ) from exc
    return resolved_path


def validate_baseline_index(
    index: Mapping[str, Any],
    *,
    baseline_root: Path,
    repository_root: Path,
) -> None:
    """Validate the complete Gate 1 path and byte-hash closure.

    Args:
        index: Parsed ``cognition_v3_baseline_index.v1`` candidate.
        baseline_root: Directory containing protected baseline artifacts.
        repository_root: Repository root containing governed files.

    Raises:
        ValueError: A schema, fixed count, path, or byte hash differs from the
            sealed Gate 1 contract.
    """

    expected_root_fields = {
        "schema_version",
        "baseline_id",
        "phase",
        "repository",
        "governed_files",
        "canonical_input_sha256",
        "architecture_path_closure",
        "artifact_records",
        "summary",
    }
    if set(index) != expected_root_fields:
        raise ValueError("baseline index root fields differ")
    if index["schema_version"] != BASELINE_INDEX_SCHEMA:
        raise ValueError("baseline index schema version differs")
    if index["phase"] != BASELINE_INDEX_PHASE:
        raise ValueError("baseline index phase differs")
    baseline_id = index["baseline_id"]
    if not isinstance(baseline_id, str) or baseline_root.name != baseline_id:
        raise ValueError("baseline index identity differs from its directory")

    repository = index["repository"]
    if not isinstance(repository, Mapping) or set(repository) != {
        "branch",
        "head",
    }:
        raise ValueError("baseline repository identity is invalid")
    if not isinstance(repository["branch"], str) or not repository["branch"]:
        raise ValueError("baseline repository branch is invalid")
    head = repository["head"]
    if (
        not isinstance(head, str)
        or len(head) != 40
        or any(character not in "0123456789abcdef" for character in head)
    ):
        raise ValueError("baseline repository HEAD is invalid")

    governed_files = index["governed_files"]
    if not isinstance(governed_files, list) or not governed_files:
        raise ValueError("baseline governed files are missing")
    governed_by_role: dict[str, Mapping[str, Any]] = {}
    governed_paths: set[str] = set()
    for record in governed_files:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "role",
            "verify_current",
        }:
            raise ValueError("baseline governed-file record is invalid")
        path = record["path"]
        file_path = _resolve_index_file(repository_root, path)
        if path in governed_paths:
            raise ValueError("baseline governed-file path is duplicated")
        governed_paths.add(path)
        role = record["role"]
        if not isinstance(role, str) or not role:
            raise ValueError("baseline governed-file role is invalid")
        if role in governed_by_role:
            raise ValueError("baseline governed-file role is duplicated")
        governed_by_role[role] = record
        if not _is_sha256(record["sha256"]):
            raise ValueError("baseline governed-file hash is invalid")
        if not isinstance(record["verify_current"], bool):
            raise TypeError("baseline governed-file verification flag differs")
        if record["verify_current"]:
            if not file_path.is_file():
                raise ValueError("baseline governed file is missing")
            actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if actual_hash != record["sha256"]:
                raise ValueError("baseline governed-file hash differs")
        elif role != "approved_execution_contract_at_gate_1_entry":
            raise ValueError("only the Gate 1 entry plan may use a frozen hash")

    required_governed_roles = {
        "approved_execution_contract_at_gate_1_entry",
        "governing_architecture",
        "architecture_manifest",
        "live_case_manifest",
        "token_calibration_corpus",
        "comparison_harness",
        "comparison_contract_test",
        "live_case_test",
        "manifest_contract_test",
    }
    if set(governed_by_role) != required_governed_roles:
        raise ValueError("baseline governed-file roles differ")

    artifact_records = index["artifact_records"]
    if not isinstance(artifact_records, list) or not artifact_records:
        raise ValueError("baseline artifact records are missing")
    indexed_paths: list[str] = []
    kind_counts = {kind: 0 for kind in BASELINE_ARTIFACT_KINDS}
    eligible_trial_ids: set[str] = set()
    defect_registry: Mapping[str, Any] | None = None
    for record in artifact_records:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "kind",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("baseline artifact record is invalid")
        path = record["path"]
        if path == "baseline_index.json":
            raise ValueError("baseline index cannot hash itself")
        artifact_path = _resolve_index_file(baseline_root, path)
        indexed_paths.append(path)
        kind = record["kind"]
        if kind not in BASELINE_ARTIFACT_KINDS:
            raise ValueError("baseline artifact kind is invalid")
        kind_counts[kind] += 1
        if not _is_sha256(record["sha256"]):
            raise ValueError("baseline artifact hash is invalid")
        if (
            not isinstance(record["size_bytes"], int)
            or isinstance(record["size_bytes"], bool)
            or record["size_bytes"] < 0
        ):
            raise ValueError("baseline artifact size is invalid")
        if not artifact_path.is_file():
            raise ValueError("baseline artifact is missing")
        artifact_bytes = artifact_path.read_bytes()
        if len(artifact_bytes) != record["size_bytes"]:
            raise ValueError("baseline artifact size differs")
        if hashlib.sha256(artifact_bytes).hexdigest() != record["sha256"]:
            raise ValueError("baseline artifact hash differs")

        if kind in {"eligible_raw_trial", "invalidated_raw_attempt"}:
            raw_artifact = json.loads(artifact_bytes.decode("utf-8"))
            if kind == "eligible_raw_trial":
                if (
                    raw_artifact.get("disposition") != ELIGIBLE_RESULT
                    or raw_artifact.get("semantic_result_available") is not True
                ):
                    raise ValueError("eligible raw trial classification differs")
                trial_id = raw_artifact.get("trial_id")
                if not isinstance(trial_id, str) or trial_id in eligible_trial_ids:
                    raise ValueError("eligible raw trial identity is invalid")
                eligible_trial_ids.add(trial_id)
            elif raw_artifact.get("semantic_result_available") is not False:
                raise ValueError("invalidated attempt retained a semantic result")
        elif kind == "pair_invalidation":
            invalidation = json.loads(artifact_bytes.decode("utf-8"))
            if (
                invalidation.get("schema_version") != PAIR_INVALIDATION_SCHEMA
                or invalidation.get("reason") != "harness_invalid_no_result"
            ):
                raise ValueError("pair invalidation classification differs")
        elif kind == "local_semantic_reset":
            reset_record = json.loads(artifact_bytes.decode("utf-8"))
            if reset_record.get("schema_version") != "local_semantic_reset.v1":
                raise ValueError("local semantic-reset schema differs")
        elif kind == "defect_registry":
            if defect_registry is not None:
                raise ValueError("baseline defect registry is duplicated")
            defect_registry = json.loads(artifact_bytes.decode("utf-8"))

    if indexed_paths != sorted(indexed_paths):
        raise ValueError("baseline artifact records are not path-sorted")
    if len(indexed_paths) != len(set(indexed_paths)):
        raise ValueError("baseline artifact path is duplicated")
    actual_paths = sorted(
        path.relative_to(baseline_root).as_posix()
        for path in baseline_root.rglob("*")
        if path.is_file() and path.name != "baseline_index.json"
    )
    if indexed_paths != actual_paths:
        raise ValueError("baseline artifact path closure differs")

    if index["summary"] != BASELINE_INDEX_SUMMARY:
        raise ValueError("baseline fixed summary differs")
    if kind_counts["eligible_raw_trial"] != 72:
        raise ValueError("baseline eligible V2 trial count differs")
    if kind_counts["invalidated_raw_attempt"] != 2:
        raise ValueError("baseline invalidated-attempt count differs")
    if kind_counts["v2_review"] != 72:
        raise ValueError("baseline V2 review count differs")
    if kind_counts["pair_invalidation"] != 2:
        raise ValueError("baseline pair-invalidation count differs")
    if kind_counts["local_semantic_reset"] != 6:
        raise ValueError("baseline semantic-reset count differs")
    if kind_counts["production_data_extract"] != 0:
        raise ValueError("baseline production-data extract count differs")
    if len(eligible_trial_ids) != 72:
        raise ValueError("baseline eligible trial identities differ")

    canonical_input_sha256 = index["canonical_input_sha256"]
    live_manifest_record = governed_by_role["live_case_manifest"]
    live_manifest_path = _resolve_index_file(
        repository_root,
        live_manifest_record["path"],
    )
    live_manifest = json.loads(live_manifest_path.read_text(encoding="utf-8"))
    expected_input_hashes = {
        row["case_id"]: canonical_sha256(row["canonical_input"])
        for row in live_manifest["cases"]
    }
    if canonical_input_sha256 != expected_input_hashes:
        raise ValueError("baseline canonical input hashes differ")
    if len(expected_input_hashes) != 24:
        raise ValueError("baseline canonical input count differs")

    closure = index["architecture_path_closure"]
    if not isinstance(closure, Mapping) or set(closure) != {
        "architecture_manifest_path",
        "architecture_manifest_sha256",
        "owned_paths_sha256",
        "path_count",
        "create_count",
        "modify_count",
        "fingerprint_artifact_path",
        "fingerprint_artifact_sha256",
    }:
        raise ValueError("baseline architecture path closure is invalid")
    architecture_record = governed_by_role["architecture_manifest"]
    if (
        closure["architecture_manifest_path"] != architecture_record["path"]
        or closure["architecture_manifest_sha256"]
        != architecture_record["sha256"]
    ):
        raise ValueError("baseline architecture manifest closure differs")
    architecture_path = _resolve_index_file(
        repository_root,
        architecture_record["path"],
    )
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    owned_paths = architecture["owned_paths"]
    if closure["owned_paths_sha256"] != canonical_sha256(owned_paths):
        raise ValueError("baseline owned-path hash differs")
    if (
        closure["create_count"] != len(owned_paths["create"])
        or closure["modify_count"] != len(owned_paths["modify"])
        or closure["path_count"]
        != len(owned_paths["create"]) + len(owned_paths["modify"])
        or closure["create_count"] != 32
        or closure["modify_count"] != 84
        or closure["path_count"] != 116
    ):
        raise ValueError("baseline architecture path counts differ")
    fingerprint_path = _resolve_index_file(
        baseline_root,
        closure["fingerprint_artifact_path"],
    )
    fingerprint_hash = hashlib.sha256(fingerprint_path.read_bytes()).hexdigest()
    if fingerprint_hash != closure["fingerprint_artifact_sha256"]:
        raise ValueError("baseline path-fingerprint hash differs")
    fingerprints = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    expected_path_dispositions = {
        path: "create" for path in owned_paths["create"]
    }
    expected_path_dispositions.update({
        path: "modify" for path in owned_paths["modify"]
    })
    actual_path_dispositions = {
        record["path"]: record["disposition"]
        for record in fingerprints["records"]
    }
    if (
        fingerprints.get("path_count") != 116
        or actual_path_dispositions != expected_path_dispositions
    ):
        raise ValueError("baseline architecture fingerprints differ")

    if defect_registry is None:
        raise ValueError("baseline defect registry is missing")
    if (
        defect_registry.get("schema_version")
        != "v2_semantic_baseline_defects.v1"
        or defect_registry.get("baseline_id") != baseline_id
        or len(defect_registry.get("defects", [])) != 7
        or defect_registry.get("hard_boundary_failures") != []
    ):
        raise ValueError("baseline defect registry closure differs")


def sanitized_environment_fingerprint(
    services: CognitionCoreServicesV2,
) -> dict[str, Any]:
    """Project route configuration without credentials or raw endpoints."""

    route_rows: list[dict[str, str]] = []
    for field_name in services.__dataclass_fields__:
        if field_name == "llm":
            continue
        config = getattr(services, field_name)
        route_rows.append({
            "service_field": field_name,
            "stage_name": config.stage_name,
            "route_name": config.route_name,
            "model": config.model,
            "base_url_sha256": hashlib.sha256(
                config.base_url.encode("utf-8")
            ).hexdigest(),
        })
    return {
        "schema_version": "cognition_v3_environment_fingerprint.v1",
        "route_count": len(route_rows),
        "routes": route_rows,
    }


def trial_artifact_path(root: Path, identity: TrialIdentity) -> Path:
    """Return the exclusive raw-artifact path for one attempt."""

    identity.validate()
    filename = (
        f"{identity.case_id}__{identity.engine}__trial-{identity.trial_index}"
        f"__attempt-{identity.attempt_index}.json"
    )
    return root / identity.baseline_id / "raw_trials" / filename


def matched_pair_invalidation_path(
    root: Path,
    identity: TrialIdentity,
) -> Path:
    """Return the governed invalidation record for one logical trial pair."""

    identity.validate()
    if identity.attempt_index == 1:
        raise ValueError("attempt 1 has no replacement invalidation")
    suffix = "" if identity.attempt_index == 2 else "__replacement-attempt-3"
    filename = (
        f"{identity.case_id}__trial-{identity.trial_index}{suffix}.json"
    )
    return root / identity.baseline_id / "invalidations" / filename


def assert_pair_rerun_allowed(
    invalidation: Mapping[str, Any],
    *,
    identity: TrialIdentity,
    artifact_root: Path,
    corrected_input_sha256: str,
) -> None:
    """Validate one retained matched-pair invalidation before attempt two."""

    if invalidation.get("schema_version") != PAIR_INVALIDATION_SCHEMA:
        raise ValueError("matched-pair invalidation schema is invalid")
    if invalidation.get("reason") != "harness_invalid_no_result":
        raise ValueError("matched-pair invalidation reason is invalid")
    expected_identity = {
        "baseline_id": identity.baseline_id,
        "case_id": identity.case_id,
        "trial_index": identity.trial_index,
    }
    if any(
        invalidation.get(key) != value
        for key, value in expected_identity.items()
    ):
        raise ValueError("matched-pair invalidation identity differs")
    if invalidation.get("replacement_attempt_index") != identity.attempt_index:
        raise ValueError("matched-pair invalidation replacement is invalid")
    if invalidation.get("corrected_input_sha256") != corrected_input_sha256:
        raise ValueError("corrected comparison input differs from invalidation")

    members = invalidation.get("pair_members")
    if not isinstance(members, list) or len(members) != len(ENGINE_NAMES):
        raise ValueError("matched-pair invalidation members are incomplete")
    members_by_engine = {
        member.get("engine"): member
        for member in members
        if isinstance(member, Mapping)
    }
    if set(members_by_engine) != ENGINE_NAMES:
        raise ValueError("matched-pair invalidation engines are incomplete")

    retained_count = 0
    baseline_root = artifact_root / identity.baseline_id
    for engine in sorted(ENGINE_NAMES):
        member = members_by_engine[engine]
        prior_attempt_index = identity.attempt_index - 1
        if member.get("attempt_index") != prior_attempt_index:
            raise ValueError("invalidated pair member attempt is invalid")
        prior_identity = TrialIdentity(
            baseline_id=identity.baseline_id,
            case_id=identity.case_id,
            engine=engine,
            trial_index=identity.trial_index,
            attempt_index=prior_attempt_index,
        )
        prior_path = trial_artifact_path(artifact_root, prior_identity)
        status = member.get("status")
        if status == "retained_invalid_artifact":
            retained_count += 1
            if not prior_path.is_file():
                raise ValueError("retained invalid artifact is missing")
            expected_relative_path = prior_path.relative_to(
                baseline_root
            ).as_posix()
            if member.get("artifact_path") != expected_relative_path:
                raise ValueError("retained invalid artifact path differs")
            actual_sha256 = hashlib.sha256(prior_path.read_bytes()).hexdigest()
            if member.get("artifact_sha256") != actual_sha256:
                raise ValueError("retained invalid artifact hash differs")
            prior_artifact = json.loads(prior_path.read_text(encoding="utf-8"))
            if prior_artifact.get("semantic_result_available") is not False:
                raise TrialAlreadySealedError(
                    "a retained semantic result can never be pair-invalidated"
                )
            if prior_artifact.get("disposition") != member.get(
                "original_disposition"
            ):
                raise ValueError("retained invalid disposition differs")
            if prior_artifact.get("input_sha256") != invalidation.get(
                "invalidated_input_sha256"
            ):
                raise ValueError("invalidated input hash differs")
        elif status == "invalidated_before_execution":
            if member.get("artifact_path") is not None:
                raise ValueError("unexecuted pair member declares an artifact")
            if member.get("artifact_sha256") is not None:
                raise ValueError("unexecuted pair member declares a hash")
            if prior_path.exists():
                raise TrialAlreadySealedError(
                    "pair member marked unexecuted already has an artifact"
                )
        else:
            raise ValueError("matched-pair invalidation status is invalid")
    if retained_count != 1:
        raise ValueError("exactly one retained invalid artifact is required")


def assert_rerun_allowed(
    prior_artifacts: Sequence[Mapping[str, Any]],
    *,
    reason: str,
) -> None:
    """Allow one rerun only when every prior candidate has no semantic result."""

    if reason not in RERUN_REASONS:
        raise ValueError("rerun reason is outside the closed invalidity set")
    if not prior_artifacts:
        raise ValueError("a rerun requires retained prior artifacts")
    for artifact in prior_artifacts:
        if artifact.get("disposition") != INVALID_NO_RESULT:
            raise TrialAlreadySealedError(
                "an eligible or hard-boundary result can never be rerun"
            )
        if artifact.get("semantic_result_available") is not False:
            raise TrialAlreadySealedError(
                "a retained semantic result can never be rerun"
            )


async def run_effect_free_trial(
    manifest_row: Mapping[str, Any],
    *,
    identity: TrialIdentity,
    services: CognitionCoreServicesV2,
    runner: EngineRunner,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    environment_fingerprint: Mapping[str, Any] | None = None,
    rerun_invalidation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run and seal one direct-facade trial without external effects."""

    identity.validate()
    if manifest_row.get("case_id") != identity.case_id:
        raise ValueError("trial identity and manifest case differ")
    artifact_path = trial_artifact_path(artifact_root, identity)
    if artifact_path.exists():
        raise TrialAlreadySealedError(
            f"trial artifact is already sealed: {artifact_path}"
        )

    input_payload = render_case_input(manifest_row)
    input_snapshot = deepcopy(input_payload)
    input_sha256 = canonical_sha256(input_snapshot)
    if identity.attempt_index > 1:
        if rerun_invalidation is None:
            raise ValueError(
                "attempt 2 requires an explicit retained-pair invalidation"
            )
        assert_pair_rerun_allowed(
            rerun_invalidation,
            identity=identity,
            artifact_root=artifact_root,
            corrected_input_sha256=input_sha256,
        )
    elif rerun_invalidation is not None:
        raise ValueError("attempt 1 cannot consume a rerun invalidation")
    capturing_llm = CapturingLLMInvoker(services.llm)
    captured_services = replace(services, llm=capturing_llm)
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started_at = time.perf_counter()
    output: CognitionCoreOutputV2 | None = None
    exception: Exception | None = None
    validator_result: dict[str, Any] = {
        "input": "passed",
        "output": "not_run",
        "input_unchanged": False,
    }
    try:
        candidate = await runner(input_payload, captured_services)
        output = validate_cognition_core_output(candidate)
        validator_result["output"] = "passed"
    # The artifact boundary records every engine/provider failure, then raises
    # TrialExecutionError after sealing rather than converting it to success.
    except Exception as exc:  # noqa: BLE001
        exception = exc
        validator_result["output"] = "failed"
    validator_result["input_unchanged"] = (
        canonical_sha256(input_payload) == input_sha256
    )
    if not validator_result["input_unchanged"] and exception is None:
        exception = RuntimeError("engine mutated the canonical comparison input")

    completed_at_utc = datetime.now(timezone.utc).isoformat()
    disposition = _trial_disposition(exception)
    semantic_result_available = output is not None
    artifact: dict[str, Any] = {
        "schema_version": TRIAL_ARTIFACT_SCHEMA,
        "baseline_id": identity.baseline_id,
        "trial_id": identity.trial_id,
        "case_id": identity.case_id,
        "engine": identity.engine,
        "trial_index": identity.trial_index,
        "attempt_index": identity.attempt_index,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "wall_time_ms": _elapsed_ms(started_at),
        "disposition": disposition,
        "semantic_result_available": semantic_result_available,
        "input_sha256": input_sha256,
        "canonical_input": input_snapshot,
        "output_sha256": canonical_sha256(output) if output is not None else None,
        "typed_output": output,
        "validator_result": validator_result,
        "model_calls": capturing_llm.calls,
        "environment_fingerprint": dict(
            environment_fingerprint
            if environment_fingerprint is not None
            else sanitized_environment_fingerprint(services)
        ),
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
    artifact["artifact_sha256"] = hashlib.sha256(
        sealed_path.read_bytes()
    ).hexdigest()
    if exception is not None:
        raise TrialExecutionError(
            f"trial {identity.trial_id} failed with {disposition}",
            artifact_path=sealed_path,
        ) from exception
    return artifact


def seal_trial_artifact(path: Path, artifact: Mapping[str, Any]) -> Path:
    """Write one artifact exclusively so no completed sample is replaced."""

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
        raise TrialAlreadySealedError(
            f"trial artifact is already sealed: {path}"
        ) from exc
    return path


def baseline_id_from_environment() -> str:
    """Return the explicitly sealed baseline identifier for a live run."""

    value = os.environ.get("COGNITION_V3_BASELINE_ID", "").strip()
    if not value:
        raise RuntimeError("COGNITION_V3_BASELINE_ID is required")
    if not _safe_component(value):
        raise RuntimeError("COGNITION_V3_BASELINE_ID is not filesystem-safe")
    return value


def trial_index_from_environment() -> int:
    """Return the explicit one-of-three live-trial index."""

    raw_value = os.environ.get("COGNITION_V3_TRIAL_INDEX", "").strip()
    try:
        trial_index = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("COGNITION_V3_TRIAL_INDEX must be 1, 2, or 3") from exc
    if trial_index not in TRIAL_INDICES:
        raise RuntimeError("COGNITION_V3_TRIAL_INDEX must be 1, 2, or 3")
    return trial_index


def attempt_index_from_environment() -> int:
    """Return the explicit live-attempt index, defaulting to the first."""

    raw_value = os.environ.get("COGNITION_V3_ATTEMPT_INDEX", "1").strip()
    try:
        attempt_index = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            "COGNITION_V3_ATTEMPT_INDEX must be 1, 2, or 3"
        ) from exc
    if attempt_index not in {1, 2, 3}:
        raise RuntimeError("COGNITION_V3_ATTEMPT_INDEX must be 1, 2, or 3")
    return attempt_index


def _trial_disposition(exception: Exception | None) -> str:
    """Classify whether one trial produced an eligible semantic result."""

    if exception is None:
        return ELIGIBLE_RESULT
    if isinstance(exception, (ConnectionError, OSError, TimeoutError)):
        return INVALID_NO_RESULT
    return HARD_BOUNDARY_FAILURE


def _safe_component(value: str) -> bool:
    """Return whether a value is safe as one artifact path component."""

    return all(character.isalnum() or character in "-_." for character in value)


def _elapsed_ms(started_at: float) -> int:
    """Return non-negative integer wall time."""

    return max(0, int((time.perf_counter() - started_at) * 1000))


def _message_record(message: BaseMessage) -> dict[str, Any]:
    """Project one LangChain message into JSON-compatible evidence."""

    return {
        "message_type": type(message).__name__,
        "content": _json_safe(getattr(message, "content", "")),
    }


def _config_record(config: LLMCallConfig) -> dict[str, Any]:
    """Project one stage configuration while excluding its credential."""

    return {
        "stage_name": config.stage_name,
        "route_name": config.route_name,
        "base_url_sha256": hashlib.sha256(
            config.base_url.encode("utf-8")
        ).hexdigest(),
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
    """Project one normalized and provider-native response for review."""

    raw_response = response.raw_response
    return {
        "content": response.content,
        "backend": {
            "route_name": response.backend.route_name,
            "backend_kind": response.backend.backend_kind,
            "model_family": response.backend.model_family,
            "model": response.backend.model,
            "normalized_base_url_sha256": hashlib.sha256(
                response.backend.normalized_base_url.encode("utf-8")
            ).hexdigest(),
            "thinking_strategy": response.backend.thinking_strategy,
            "confidence": response.backend.confidence,
            "generation": response.backend.generation,
        },
        "usage": _json_safe(response.usage),
        "provider_raw_content": _json_safe(
            getattr(raw_response, "content", None)
        ),
        "provider_response_metadata": _json_safe(
            getattr(raw_response, "response_metadata", None)
        ),
    }


def _exception_record(exception: Exception) -> dict[str, str]:
    """Return bounded exception evidence without a traceback."""

    return {
        "type": type(exception).__name__,
        "module": type(exception).__module__,
        "message": str(exception)[:2000],
    }


def _json_safe(value: Any) -> Any:
    """Convert provider metadata to a bounded JSON-compatible structure."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in list(value.items())[:100]
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_json_safe(item) for item in list(value)[:100]]
    return repr(value)[:4000]
