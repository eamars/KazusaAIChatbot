"""Resumable, blinded quality matrix for Cognition Core V2 model bindings."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import secrets
import statistics
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import BaseMessage

from control_console.brain_model_routes import fetch_available_models
from kazusa_ai_chatbot import utils as llm_utils
from kazusa_ai_chatbot.character_profile import (
    validate_character_profile_seed,
)
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    EvidenceRefV1,
    PerceptV1,
    TargetScopeV1,
    build_text_chat_media_description_rows,
    build_user_message_episode,
)
from kazusa_ai_chatbot.cognition_core_v2 import run_cognition
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    CognitionCoreInputV2,
    CognitionCoreOutputV2,
    CognitionCoreServicesV2,
    CognitionExecutionError,
    validate_cognition_core_input,
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    reset_validation_capture,
    validation_capture_snapshot,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    validate_cognition_state,
)
from kazusa_ai_chatbot.conversation_history_prompt_projection import (
    project_conversation_history_for_llm,
)
from kazusa_ai_chatbot.db import (
    close_db,
    get_character_profile,
    get_conversation_by_platform_message_id,
    get_conversation_history,
    get_user_profile,
    split_character_profile_runtime_state,
)
from kazusa_ai_chatbot.llm_interface import (
    LLMCallConfig,
    LLMResponse,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
    build_cognition_input_from_global_state,
)
from kazusa_ai_chatbot.time_boundary import (
    local_time_context_from_storage_utc,
    parse_storage_utc_datetime,
)


ARTIFACT_ROOT = Path("test_artifacts/cognition_model_assignment")
SOURCE_MANIFEST_SCHEMA = "cognition_model_assignment_source_cases.v1"
SNAPSHOT_SCHEMA = "cognition_model_assignment_case.v1"
SNAPSHOT_INDEX_SCHEMA = "cognition_model_assignment_snapshot_index.v1"
LEDGER_SCHEMA = "cognition_model_assignment_ledger.v1"
UNBLINDING_SCHEMA = "cognition_model_assignment_unblinding.v1"
SAMPLE_ARTIFACT_SCHEMA = "cognition_model_assignment_sample.v1"
INSPECTION_SCHEMA = "cognition_model_assignment_inspection.v1"
REVIEW_QUEUE_SCHEMA = "cognition_model_assignment_review_queue.v1"
REVIEW_SCHEMA = "cognition_model_assignment_review.v1"
AGGREGATE_SCHEMA = "cognition_model_assignment_aggregate.v1"

FACTOR_FIELDS = (
    "appraisal_event_agency_config",
    "goal_ordinary_response_config",
    "workspace_collapse_config",
    "action_planning_config",
)
FACTOR_SERVICE_FIELDS = {
    "appraisal_event_agency_config": (
        "appraisal_event_agency_config",
        "appraisal_relationship_social_config",
        "appraisal_moral_identity_config",
        "appraisal_goal_threat_outcome_config",
        "appraisal_epistemic_comparison_memory_config",
        "appraisal_existential_drive_config",
    ),
    "goal_ordinary_response_config": (
        "goal_ordinary_response_config",
        "goal_active_branch_config",
    ),
    "workspace_collapse_config": (
        "workspace_collapse_config",
    ),
    "action_planning_config": (
        "action_planning_config",
        "action_authorization_config",
        "resolver_authorization_config",
    ),
}
PROFILE_REPRESENTATIVE_FIELDS = {
    "D": "appraisal_event_agency_config",
    "M": "appraisal_relationship_social_config",
}
CONFIGURED_PROFILE_SERVICE_FIELDS = {
    "D": (
        "appraisal_event_agency_config",
        "goal_ordinary_response_config",
        "workspace_collapse_config",
        "action_planning_config",
        "action_authorization_config",
        "resolver_authorization_config",
    ),
    "M": (
        "appraisal_relationship_social_config",
        "appraisal_moral_identity_config",
        "appraisal_goal_threat_outcome_config",
        "appraisal_epistemic_comparison_memory_config",
        "appraisal_existential_drive_config",
        "goal_active_branch_config",
    ),
}
MODEL_LEVELS = ("D", "M")
CASE_COUNT = 8
MATRIX_CELL_COUNT = 16
REPETITION_COUNT = 3
SAMPLE_COUNT = CASE_COUNT * MATRIX_CELL_COUNT * REPETITION_COUNT
CONTEXT_ROW_LIMIT = 8
CONTEXT_QUERY_LIMIT = 64
ROTATION_STEP = 5
BLIND_LABEL_HEX_CHARS = 16
SOURCE_EXCERPT_MAX_CHARS = 800

TERMINAL_STATUSES = frozenset({
    "completed",
    "transport_failed",
    "contract_failed",
})
LEDGER_STATUSES = frozenset({
    "pending",
    "running",
    *TERMINAL_STATUSES,
})
INSPECTION_STATUSES = frozenset({
    "pending",
    "accepted",
    "finding_recorded",
})
SCORE_FIELDS = (
    "input_responsiveness",
    "character_state_consistency",
    "situational_suitability",
    "role_evidence_grounding",
    "cross_stage_coherence",
)
VERDICT_VALUES = (
    "better",
    "equivalent",
    "minor_loss",
    "material_loss",
    "critical_loss",
)
VERDICT_RANKS = {
    "critical_loss": 0,
    "material_loss": 1,
    "minor_loss": 2,
    "equivalent": 3,
    "better": 4,
}

CharacterReader = Callable[[], Awaitable[Mapping[str, Any]]]
UserReader = Callable[[str], Awaitable[Mapping[str, Any]]]
SourceMessageReader = Callable[..., Awaitable[Mapping[str, Any] | None]]
HistoryReader = Callable[..., Awaitable[list[Mapping[str, Any]]]]
CloseDatabase = Callable[[], Awaitable[None]]
ModelFetcher = Callable[..., Awaitable[dict[str, Any]]]
RunCognition = Callable[
    [CognitionCoreInputV2, CognitionCoreServicesV2],
    Awaitable[CognitionCoreOutputV2],
]


def enumerate_assignment_matrix() -> list[dict[str, Any]]:
    """Return all two-level assignments in fixed binary factor order."""

    matrix: list[dict[str, Any]] = []
    bit_values = (8, 4, 2, 1)
    for cell_index in range(MATRIX_CELL_COUNT):
        assignment = {
            factor_field: (
                "M"
                if cell_index & bit_value
                else "D"
            )
            for factor_field, bit_value in zip(
                FACTOR_FIELDS,
                bit_values,
                strict=True,
            )
        }
        matrix.append({
            "cell_id": f"Q{cell_index:02d}",
            "assignment": assignment,
        })
    return matrix


def build_source_manifest(
    manifest_path: Path,
    cases: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Write one exact local source manifest for snapshot preparation."""

    manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "case_count": len(cases),
        "cases": [dict(case) for case in cases],
    }
    _validate_source_manifest(manifest)
    _write_json(manifest_path, manifest)
    return manifest


def load_source_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and validate the ignored source-case manifest."""

    manifest = _load_json(manifest_path)
    _validate_source_manifest(manifest)
    return manifest


def _validate_source_manifest(manifest: Mapping[str, Any]) -> None:
    """Enforce the fixed source-case schema without reading message content."""

    required_root_fields = {"schema_version", "case_count", "cases"}
    optional_root_fields = {"selection_date", "source"}
    manifest_fields = set(manifest)
    if (
        not required_root_fields.issubset(manifest_fields)
        or not manifest_fields.issubset(
            required_root_fields | optional_root_fields
        )
    ):
        raise ValueError("source manifest fields are not exact")
    if manifest["schema_version"] != SOURCE_MANIFEST_SCHEMA:
        raise ValueError("source manifest schema version is unsupported")
    if manifest["case_count"] != CASE_COUNT:
        raise ValueError(f"source manifest must declare {CASE_COUNT} cases")
    cases = manifest["cases"]
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        raise ValueError(f"source manifest must contain {CASE_COUNT} cases")
    for field_name in optional_root_fields:
        if field_name not in manifest:
            continue
        field_value = manifest[field_name]
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(
                f"source manifest {field_name} must be non-empty text"
            )

    expected_case_fields = {
        "case_id",
        "platform",
        "platform_channel_id",
        "platform_message_id",
        "scenario_dimension",
    }
    case_ids: set[str] = set()
    source_identities: set[tuple[str, str, str]] = set()
    for case in cases:
        if not isinstance(case, Mapping) or set(case) != expected_case_fields:
            raise ValueError("source manifest case fields are not exact")
        for field_name in expected_case_fields:
            field_value = case[field_name]
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(
                    f"source manifest {field_name} must be non-empty text"
                )
        case_id = str(case["case_id"])
        if not _is_safe_artifact_component(case_id):
            raise ValueError("source manifest case id is not path-safe")
        source_identity = (
            str(case["platform"]),
            str(case["platform_channel_id"]),
            str(case["platform_message_id"]),
        )
        if case_id in case_ids:
            raise ValueError("source manifest case ids must be unique")
        if source_identity in source_identities:
            raise ValueError("source manifest message identities must be unique")
        case_ids.add(case_id)
        source_identities.add(source_identity)


def _is_safe_artifact_component(value: str) -> bool:
    """Return whether text is safe as one local artifact filename component."""

    return bool(value) and all(
        character.isalnum() or character in {"-", "_"}
        for character in value
    )


def build_services_for_cell(
    base_services: CognitionCoreServicesV2,
    cell: Mapping[str, Any],
    *,
    llm: object | None = None,
) -> CognitionCoreServicesV2:
    """Apply one assignment while retaining every factor-local setting."""

    assignment = _validated_assignment(cell)
    profile_by_level = {
        level: getattr(base_services, service_field)
        for level, service_field in PROFILE_REPRESENTATIVE_FIELDS.items()
    }
    replacement_configs: dict[str, LLMCallConfig] = {}
    for factor_field in FACTOR_FIELDS:
        selected_profile = profile_by_level[assignment[factor_field]]
        for service_field in FACTOR_SERVICE_FIELDS[factor_field]:
            original_config = getattr(base_services, service_field)
            replacement_configs[service_field] = replace(
                original_config,
                base_url=selected_profile.base_url,
                api_key=selected_profile.api_key,
                model=selected_profile.model,
            )

    selected_llm = base_services.llm if llm is None else llm
    services = replace(
        base_services,
        llm=selected_llm,  # type: ignore[arg-type]
        **replacement_configs,
    )
    return services


def _validated_assignment(cell: Mapping[str, Any]) -> dict[str, str]:
    """Return one exact four-factor assignment."""

    if set(cell) != {"cell_id", "assignment"}:
        raise ValueError("matrix cell fields are not exact")
    cell_id = cell["cell_id"]
    if not isinstance(cell_id, str) or not cell_id.startswith("Q"):
        raise ValueError("matrix cell id is invalid")
    assignment = cell["assignment"]
    if not isinstance(assignment, Mapping):
        raise ValueError("matrix cell assignment must be an object")
    if set(assignment) != set(FACTOR_FIELDS):
        raise ValueError("matrix factor fields are invalid")
    normalized_assignment: dict[str, str] = {}
    for factor_field in FACTOR_FIELDS:
        level = assignment[factor_field]
        if level not in MODEL_LEVELS:
            raise ValueError("matrix assignment level is invalid")
        normalized_assignment[factor_field] = str(level)
    return normalized_assignment


def route_profile_digest(services: CognitionCoreServicesV2) -> str:
    """Hash all factor settings and both model profiles."""

    profiles = {
        "factor_configs": {
            factor_field: {
                service_field: _complete_config_projection(
                    getattr(services, service_field),
                )
                for service_field in FACTOR_SERVICE_FIELDS[factor_field]
            }
            for factor_field in FACTOR_FIELDS
        },
        "model_profiles": {
            level: _endpoint_identity_projection(
                getattr(services, service_field),
            )
            for level, service_field in PROFILE_REPRESENTATIVE_FIELDS.items()
        },
    }
    profile_digest = _stable_digest(profiles)
    return profile_digest


def _endpoint_identity_projection(
    config: LLMCallConfig,
) -> dict[str, str]:
    """Project the atomic endpoint identity selected by one factor."""

    projection = {
        "base_url": config.base_url,
        "api_key": config.api_key,
        "model": config.model,
    }
    return projection


def _validate_configured_profile_ownership(
    services: CognitionCoreServicesV2,
) -> None:
    """Require the current stage assignment across both historical profiles."""

    identities: dict[str, dict[str, str]] = {}
    for level, service_fields in CONFIGURED_PROFILE_SERVICE_FIELDS.items():
        representative = getattr(
            services,
            PROFILE_REPRESENTATIVE_FIELDS[level],
        )
        expected_identity = _endpoint_identity_projection(representative)
        identities[level] = expected_identity
        for service_field in service_fields:
            stage_identity = _endpoint_identity_projection(
                getattr(services, service_field),
            )
            if stage_identity != expected_identity:
                raise ValueError(
                    f"configured {level} stage bindings do not share one "
                    "endpoint identity"
                )
    if identities["D"] == identities["M"]:
        raise ValueError("D and M profiles must identify distinct models")


async def verify_configured_model_profiles(
    services: CognitionCoreServicesV2,
    *,
    model_fetcher: ModelFetcher = fetch_available_models,
) -> dict[str, Any]:
    """Verify that both configured model IDs are listed by their providers."""

    _validate_configured_profile_ownership(services)
    profiles = {
        level: getattr(services, service_field)
        for level, service_field in PROFILE_REPRESENTATIVE_FIELDS.items()
    }

    provider_results: dict[tuple[str, str], dict[str, Any]] = {}
    profile_results: dict[str, dict[str, str]] = {}
    for level, config in profiles.items():
        provider_key = (config.base_url.rstrip("/"), config.api_key)
        if provider_key not in provider_results:
            provider_results[provider_key] = await model_fetcher(
                config.base_url,
                config.api_key,
            )
        result = provider_results[provider_key]
        if result.get("status") != "available":
            raise RuntimeError(
                f"configured {level} model provider is unavailable"
            )
        raw_models = result.get("models")
        if not isinstance(raw_models, list):
            raise RuntimeError("provider model list is malformed")
        model_ids = {
            str(row.get("id", "")).strip()
            for row in raw_models
            if isinstance(row, Mapping)
        }
        if config.model not in model_ids:
            raise RuntimeError(
                f"configured {level} model is absent from its provider list"
            )
        profile_results[level] = {
            "model_digest": _stable_digest(config.model),
            "provider_digest": _stable_digest(config.base_url.rstrip("/")),
        }

    summary = {
        "status": "ready",
        "profile_count": len(profile_results),
        "provider_count": len(provider_results),
        "profiles": profile_results,
        "route_profile_digest": route_profile_digest(services),
    }
    return summary


async def preflight_environment(
    manifest_path: Path,
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    get_character_profile_func: CharacterReader = get_character_profile,
    get_user_profile_func: UserReader = get_user_profile,
    get_source_message_func: SourceMessageReader = (
        get_conversation_by_platform_message_id
    ),
    get_conversation_history_func: HistoryReader = get_conversation_history,
    close_db_func: CloseDatabase = close_db,
    model_fetcher: ModelFetcher = fetch_available_models,
) -> dict[str, Any]:
    """Validate model and read-only data readiness without writing snapshots."""

    manifest = load_source_manifest(manifest_path)
    services = build_cognition_core_services()
    model_summary = await verify_configured_model_profiles(
        services,
        model_fetcher=model_fetcher,
    )
    try:
        material = await _collect_case_material(
            manifest,
            get_character_profile_func=get_character_profile_func,
            get_user_profile_func=get_user_profile_func,
            get_source_message_func=get_source_message_func,
            get_conversation_history_func=get_conversation_history_func,
        )
    finally:
        await close_db_func()

    summary = {
        "schema_version": "cognition_model_assignment_preflight.v1",
        "status": "ready",
        "case_count": len(material["cases"]),
        "profile_digest": material["profile_digest"],
        "character_state_digest": material["character_state_digest"],
        "route_profile_digest": model_summary["route_profile_digest"],
        "model_profile_count": model_summary["profile_count"],
        "model_provider_count": model_summary["provider_count"],
        "database_closed": True,
        "checked_at": _utc_now_iso(),
    }
    _write_json(artifact_root / "preflight_summary.json", summary)
    return summary


async def build_case_snapshots(
    manifest_path: Path,
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    get_character_profile_func: CharacterReader = get_character_profile,
    get_user_profile_func: UserReader = get_user_profile,
    get_source_message_func: SourceMessageReader = (
        get_conversation_by_platform_message_id
    ),
    get_conversation_history_func: HistoryReader = get_conversation_history,
    close_db_func: CloseDatabase = close_db,
) -> dict[str, Any]:
    """Freeze eight validated inputs through read-only database helpers."""

    manifest = load_source_manifest(manifest_path)
    try:
        material = await _collect_case_material(
            manifest,
            get_character_profile_func=get_character_profile_func,
            get_user_profile_func=get_user_profile_func,
            get_source_message_func=get_source_message_func,
            get_conversation_history_func=get_conversation_history_func,
        )
    finally:
        await close_db_func()

    snapshots_root = artifact_root / "snapshots"
    profile_path = snapshots_root / "active_profile.json"
    _write_frozen_json(profile_path, material["character_profile"])
    privacy_tokens = _character_privacy_tokens(
        material["character_profile"],
    )
    privacy_token_path = artifact_root / "tracked_forbidden_tokens.txt"
    _write_frozen_text(
        privacy_token_path,
        "".join(f"{token}\n" for token in privacy_tokens),
    )

    case_index_rows: list[dict[str, Any]] = []
    snapshot_digests: list[str] = []
    for case in material["cases"]:
        case_id = case["case_id"]
        snapshot_path = snapshots_root / f"{case_id}.json"
        snapshot = {
            "schema_version": SNAPSHOT_SCHEMA,
            "case_id": case_id,
            "scenario_dimension": case["scenario_dimension"],
            "captured_at": material["captured_at"],
            "source_identity": case["source_identity"],
            "source_digest": case["source_digest"],
            "profile_digest": material["profile_digest"],
            "character_state_digest": material[
                "character_state_digest"
            ],
            "user_state_digest": case["user_state_digest"],
            "input_digest": case["input_digest"],
            "profile_snapshot_path": "snapshots/active_profile.json",
            "source_snapshot": case["source_snapshot"],
            "input": case["input"],
            "review_projection": case["review_projection"],
        }
        snapshot_digest = _snapshot_payload_digest(snapshot)
        snapshot["snapshot_digest"] = snapshot_digest
        snapshot_digests.append(snapshot_digest)
        _write_frozen_json(
            snapshot_path,
            snapshot,
            ignored_fields={"captured_at"},
        )
        case_index_rows.append({
            "case_id": case_id,
            "scenario_dimension": case["scenario_dimension"],
            "snapshot_path": f"snapshots/{case_id}.json",
            "snapshot_digest": snapshot_digest,
            "input_digest": case["input_digest"],
        })

    snapshot_set_digest = _stable_digest(snapshot_digests)
    snapshot_index = {
        "schema_version": SNAPSHOT_INDEX_SCHEMA,
        "case_count": len(case_index_rows),
        "snapshot_set_digest": snapshot_set_digest,
        "profile_digest": material["profile_digest"],
        "character_state_digest": material["character_state_digest"],
        "profile_snapshot_path": "snapshots/active_profile.json",
        "privacy_token_path": "tracked_forbidden_tokens.txt",
        "database_closed": True,
        "cases": case_index_rows,
    }
    _write_frozen_json(snapshots_root / "index.json", snapshot_index)
    return snapshot_index


async def _collect_case_material(
    manifest: Mapping[str, Any],
    *,
    get_character_profile_func: CharacterReader,
    get_user_profile_func: UserReader,
    get_source_message_func: SourceMessageReader,
    get_conversation_history_func: HistoryReader,
) -> dict[str, Any]:
    """Read and validate the exact material needed for frozen inputs."""

    character_profile_value = await get_character_profile_func()
    if not isinstance(character_profile_value, Mapping):
        raise TypeError("active character profile must be a mapping")
    character_profile = _json_projection(character_profile_value)
    if not isinstance(character_profile, dict) or not character_profile:
        raise ValueError("active character profile is missing")
    static_profile, _ = split_character_profile_runtime_state(
        character_profile,
    )
    validate_character_profile_seed(static_profile)
    character_state_value = character_profile.get("cognition_state")
    if not isinstance(character_state_value, Mapping):
        raise ValueError("active character cognition state is missing")
    character_state = validate_cognition_state(character_state_value)
    if character_state["state_scope"] != "character":
        raise ValueError("active character cognition state scope is invalid")

    captured_at = _utc_now_iso()
    profile_digest = _stable_digest(character_profile)
    character_state_digest = _stable_digest(character_state)
    cases: list[dict[str, Any]] = []
    raw_cases = manifest["cases"]
    for case_value in raw_cases:
        case = dict(case_value)
        source_row_value = await get_source_message_func(
            platform=case["platform"],
            platform_channel_id=case["platform_channel_id"],
            platform_message_id=case["platform_message_id"],
        )
        if not isinstance(source_row_value, Mapping):
            raise ValueError(
                f"source row is missing for case {case['case_id']}"
            )
        source_row = dict(source_row_value)
        _validate_source_row(case, source_row)
        global_user_id = str(source_row["global_user_id"])
        user_profile_value = await get_user_profile_func(global_user_id)
        if not isinstance(user_profile_value, Mapping):
            raise TypeError("source user profile must be a mapping")
        user_profile = dict(user_profile_value)
        user_state_value = user_profile.get("cognition_state")
        if not isinstance(user_state_value, Mapping):
            raise ValueError(
                f"source user cognition state is missing for "
                f"{case['case_id']}"
            )
        user_state = validate_cognition_state(user_state_value)
        if (
            user_state["state_scope"] != "user"
            or user_state["owner_user_id"] != global_user_id
        ):
            raise ValueError("source user cognition state owner is invalid")

        source_timestamp = str(source_row["timestamp"])
        history_rows = await get_conversation_history_func(
            platform=case["platform"],
            platform_channel_id=case["platform_channel_id"],
            limit=CONTEXT_QUERY_LIMIT,
            to_timestamp=source_timestamp,
            sort_direction=-1,
        )
        prior_rows = _strictly_prior_rows(
            history_rows,
            source_timestamp=source_timestamp,
        )
        source_snapshot = _source_snapshot(source_row)
        cognition_input = _build_frozen_cognition_input(
            source_row,
            character_profile=character_profile,
            character_state=character_state,
            user_state=user_state,
        )
        input_digest = _stable_digest(cognition_input)
        review_projection = _build_review_projection(
            source_snapshot,
            prior_rows=prior_rows,
            character_profile=character_profile,
            cognition_input=cognition_input,
            user_state=user_state,
            captured_at=captured_at,
        )
        cases.append({
            "case_id": case["case_id"],
            "scenario_dimension": case["scenario_dimension"],
            "source_identity": {
                "platform": case["platform"],
                "platform_channel_id": case["platform_channel_id"],
                "platform_message_id": case["platform_message_id"],
            },
            "source_snapshot": source_snapshot,
            "source_digest": _stable_digest(source_snapshot),
            "user_state_digest": _stable_digest(user_state),
            "input": cognition_input,
            "input_digest": input_digest,
            "review_projection": review_projection,
        })

    material = {
        "captured_at": captured_at,
        "character_profile": character_profile,
        "profile_digest": profile_digest,
        "character_state_digest": character_state_digest,
        "cases": cases,
    }
    return material


def _validate_source_row(
    case: Mapping[str, Any],
    source_row: Mapping[str, Any],
) -> None:
    """Validate one persisted user message against its manifest identity."""

    identity_fields = (
        "platform",
        "platform_channel_id",
        "platform_message_id",
    )
    for field_name in identity_fields:
        if source_row.get(field_name) != case[field_name]:
            raise ValueError("source row identity does not match manifest")
    if source_row.get("role") != "user":
        raise ValueError("source row must be a persisted user message")
    required_text_fields = (
        "body_text",
        "timestamp",
        "channel_type",
        "platform_user_id",
        "global_user_id",
        "display_name",
    )
    for field_name in required_text_fields:
        field_value = source_row.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"source row {field_name} must be non-empty")
    parse_storage_utc_datetime(str(source_row["timestamp"]))


def _strictly_prior_rows(
    history_rows: Sequence[Mapping[str, Any]],
    *,
    source_timestamp: str,
) -> list[dict[str, Any]]:
    """Keep at most eight rows with timestamps strictly before the source."""

    source_time = parse_storage_utc_datetime(source_timestamp)
    prior_rows: list[dict[str, Any]] = []
    for row in history_rows:
        row_timestamp = row.get("timestamp")
        if not isinstance(row_timestamp, str) or not row_timestamp.strip():
            continue
        try:
            row_time = parse_storage_utc_datetime(row_timestamp)
        except ValueError:
            continue
        if row_time < source_time:
            prior_rows.append(_source_snapshot(row))
    bounded_rows = prior_rows[-CONTEXT_ROW_LIMIT:]
    return bounded_rows


def _source_snapshot(source_row: Mapping[str, Any]) -> dict[str, Any]:
    """Project one message without embeddings, raw wire text, or media bytes."""

    projected_fields = (
        "platform",
        "platform_channel_id",
        "channel_type",
        "role",
        "platform_message_id",
        "platform_user_id",
        "global_user_id",
        "display_name",
        "body_text",
        "content_type",
        "addressed_to_global_user_ids",
        "mentions",
        "broadcast",
        "reply_context",
        "timestamp",
    )
    snapshot = {
        field_name: _json_projection(source_row[field_name])
        for field_name in projected_fields
        if field_name in source_row
    }
    attachments = source_row.get("attachments")
    snapshot["attachments"] = _project_attachment_descriptions(attachments)
    row_id = source_row.get("conversation_row_id")
    if row_id is None:
        row_id = source_row.get("_id")
    if row_id is not None:
        snapshot["conversation_row_id"] = str(row_id)
    return snapshot


def _project_attachment_descriptions(value: object) -> list[dict[str, str]]:
    """Retain only typed attachment kind, content type, and description."""

    if not isinstance(value, list):
        return []
    attachments: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        projected: dict[str, str] = {}
        for field_name in ("media_kind", "content_type", "description"):
            field_value = item.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                projected[field_name] = field_value.strip()
        if projected:
            attachments.append(projected)
    return attachments


def _build_frozen_cognition_input(
    source_row: Mapping[str, Any],
    *,
    character_profile: Mapping[str, Any],
    character_state: Mapping[str, Any],
    user_state: Mapping[str, Any],
) -> CognitionCoreInputV2:
    """Build one current-state counterfactual through production contracts."""

    platform = str(source_row["platform"])
    platform_channel_id = str(source_row["platform_channel_id"])
    platform_message_id = str(source_row["platform_message_id"])
    body_text = str(source_row["body_text"])
    source_timestamp = str(source_row["timestamp"])
    episode_id = (
        f"user_message:{platform}:{platform_channel_id}:"
        f"{platform_message_id}"
    )
    dialog_percept: PerceptV1 = {
        "schema_version": "percept.v1",
        "percept_kind": "dialog",
        "source_kind": "dialog",
        "source_id": platform_message_id,
        "content": {
            "semantic_text": body_text,
            "speaker_role": CURRENT_USER_ROLE,
            "addressee_role": CURRENT_CHARACTER_ROLE,
        },
        "observed_at": source_timestamp,
    }
    attachment_rows = _project_attachment_descriptions(
        source_row.get("attachments"),
    )
    media_rows = build_text_chat_media_description_rows(attachment_rows)
    media_percepts: list[PerceptV1] = []
    for media_index, media_row in enumerate(media_rows, start=1):
        content_type = media_row["content_type"]
        source_kind = (
            "image_observation"
            if content_type.startswith("image/")
            else "audio_observation"
        )
        observation = media_row.get("image_observation")
        media_percepts.append({
            "schema_version": "percept.v1",
            "percept_kind": source_kind,
            "source_kind": source_kind,
            "source_id": f"{episode_id}:media:{media_index}",
            "content": {
                "content_type": content_type,
                "description": media_row["description"],
                "observation": (
                    dict(observation)
                    if isinstance(observation, Mapping)
                    else {}
                ),
            },
            "observed_at": source_timestamp,
        })

    addressed_values = source_row.get("addressed_to_global_user_ids")
    addressed_user_ids = (
        [
            str(value)
            for value in addressed_values
            if isinstance(value, str) and value.strip()
        ]
        if isinstance(addressed_values, list)
        else []
    )
    target_scope: TargetScopeV1 = {
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "channel_type": str(source_row["channel_type"]),
        "current_platform_user_id": str(source_row["platform_user_id"]),
        "current_global_user_id": str(source_row["global_user_id"]),
        "current_display_name": str(source_row["display_name"]),
        "target_addressed_user_ids": addressed_user_ids,
        "target_broadcast": (
            source_row.get("broadcast")
            if isinstance(source_row.get("broadcast"), bool)
            else False
        ),
        "permission_ref": f"{platform}:{platform_channel_id}",
    }
    row_id = source_row.get("conversation_row_id")
    if row_id is None:
        row_id = source_row.get("_id")
    row_ids = [str(row_id)] if row_id is not None else []
    origin = {
        "schema_version": "user_message_origin.v1",
        "owner": "brain_service.intake",
        "platform": platform,
        "platform_message_id": platform_message_id,
        "active_turn_platform_message_ids": [platform_message_id],
        "active_turn_conversation_row_ids": row_ids,
        "debug_modes": {},
        "correlation_id": episode_id,
        "privacy_scope": "conversation",
        "delivery_permission_ref": f"{platform}:{platform_channel_id}",
        "created_at": source_timestamp,
    }
    evidence_ref: EvidenceRefV1 = {
        "schema_version": "evidence_ref.v1",
        "evidence_kind": "conversation_message",
        "evidence_id": platform_message_id,
        "owner": "brain_service.intake",
        "excerpt": body_text[:SOURCE_EXCERPT_MAX_CHARS],
        "observed_at": source_timestamp,
    }
    episode = build_user_message_episode(
        episode_id=episode_id,
        origin=origin,
        target_scope=target_scope,
        dialog_percept=dialog_percept,
        media_percepts=media_percepts,
        evidence_refs=[evidence_ref],
        local_time_context=local_time_context_from_storage_utc(
            source_timestamp,
        ),
        created_at=source_timestamp,
        debug_controls={},
    )
    global_state = {
        "cognitive_episode": episode,
        "global_user_id": str(source_row["global_user_id"]),
        "user_name": str(source_row["display_name"]),
        "user_input": body_text,
        "user_multimedia_input": media_rows,
        "rag_result": {},
        "character_profile": dict(character_profile),
        "channel_name": str(source_row.get("channel_name", "")),
    }
    cognition_input = build_cognition_input_from_global_state(
        global_state,  # type: ignore[arg-type]
        mutable_state=user_state,
        character_state=character_state,
    )
    validated_input = validate_cognition_core_input(cognition_input)
    return validated_input


def _build_review_projection(
    source_snapshot: Mapping[str, Any],
    *,
    prior_rows: Sequence[Mapping[str, Any]],
    character_profile: Mapping[str, Any],
    cognition_input: Mapping[str, Any],
    user_state: Mapping[str, Any],
    captured_at: str,
) -> dict[str, Any]:
    """Build reviewer evidence without inventing upstream semantic artifacts."""

    character_name_value = character_profile.get("name")
    character_name = (
        character_name_value.strip()
        if isinstance(character_name_value, str)
        else ""
    )
    prior_context = project_conversation_history_for_llm(
        prior_rows,
        character_name=character_name,
        max_rows=CONTEXT_ROW_LIMIT,
    )
    review_projection = {
        "current_input": source_snapshot["body_text"],
        "typed_target_data": {
            "channel_type": source_snapshot["channel_type"],
            "addressed_to_global_user_ids": source_snapshot.get(
                "addressed_to_global_user_ids",
                [],
            ),
            "mentions": source_snapshot.get("mentions", []),
            "broadcast": source_snapshot.get("broadcast", False),
            "reply_context": source_snapshot.get("reply_context", {}),
        },
        "source_timestamp": source_snapshot["timestamp"],
        "snapshot_timestamp": captured_at,
        "state_updated_at": user_state["updated_at"],
        "model_visible_character_constraints": cognition_input[
            "character_constraints"
        ],
        "initial_affect_activations": user_state["affect_activations"],
        "initial_relationship": user_state["relationship"],
        "bounded_prior_context": prior_context,
        "model_visible_prior_context": False,
        "temporal_interpretation": "current_state_counterfactual",
    }
    return review_projection


def _character_privacy_tokens(
    character_profile: Mapping[str, Any],
) -> list[str]:
    """Collect configured identity tokens for tracked-file privacy checks."""

    tokens: set[str] = set()
    name_value = character_profile.get("name")
    if isinstance(name_value, str) and name_value.strip():
        tokens.add(name_value.strip())
    for field_name in ("aliases", "alternate_names"):
        raw_values = character_profile.get(field_name)
        if not isinstance(raw_values, list):
            continue
        tokens.update(
            value.strip()
            for value in raw_values
            if isinstance(value, str) and value.strip()
        )
    if not tokens:
        raise ValueError("active character profile has no privacy token")
    ordered_tokens = sorted(tokens, key=lambda value: (-len(value), value))
    return ordered_tokens


def load_snapshot_index(
    *,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Load and validate the immutable snapshot index."""

    index_path = artifact_root / "snapshots" / "index.json"
    snapshot_index = _load_json(index_path)
    expected_fields = {
        "schema_version",
        "case_count",
        "snapshot_set_digest",
        "profile_digest",
        "character_state_digest",
        "profile_snapshot_path",
        "privacy_token_path",
        "database_closed",
        "cases",
    }
    if set(snapshot_index) != expected_fields:
        raise ValueError("snapshot index fields are not exact")
    if snapshot_index["schema_version"] != SNAPSHOT_INDEX_SCHEMA:
        raise ValueError("snapshot index schema is unsupported")
    cases = snapshot_index["cases"]
    if (
        snapshot_index["case_count"] != CASE_COUNT
        or not isinstance(cases, list)
        or len(cases) != CASE_COUNT
    ):
        raise ValueError(f"snapshot index must contain {CASE_COUNT} cases")
    if snapshot_index["database_closed"] is not True:
        raise ValueError("snapshot index lacks database-close evidence")
    if snapshot_index["profile_snapshot_path"] != (
        "snapshots/active_profile.json"
    ):
        raise ValueError("snapshot profile path is invalid")
    if snapshot_index["privacy_token_path"] != (
        "tracked_forbidden_tokens.txt"
    ):
        raise ValueError("snapshot privacy-token path is invalid")

    profile_path = artifact_root / snapshot_index["profile_snapshot_path"]
    profile = _load_json(profile_path)
    if _stable_digest(profile) != snapshot_index["profile_digest"]:
        raise ValueError("snapshot profile digest changed")
    character_state = profile.get("cognition_state")
    if not isinstance(character_state, Mapping):
        raise ValueError("snapshot character state is missing")
    if (
        _stable_digest(character_state)
        != snapshot_index["character_state_digest"]
    ):
        raise ValueError("snapshot character-state digest changed")

    privacy_token_path = (
        artifact_root / snapshot_index["privacy_token_path"]
    )
    if (
        not privacy_token_path.is_file()
        or not privacy_token_path.read_text(encoding="utf-8").strip()
    ):
        raise ValueError("snapshot privacy-token evidence is missing")

    seen_case_ids: set[str] = set()
    snapshot_digests: list[str] = []
    for case_row_value in cases:
        if not isinstance(case_row_value, Mapping):
            raise ValueError("snapshot index case row must be an object")
        case_row = case_row_value
        expected_case_fields = {
            "case_id",
            "scenario_dimension",
            "snapshot_path",
            "snapshot_digest",
            "input_digest",
        }
        if set(case_row) != expected_case_fields:
            raise ValueError("snapshot index case fields are not exact")
        case_id = case_row["case_id"]
        if (
            not isinstance(case_id, str)
            or not _is_safe_artifact_component(case_id)
            or case_id in seen_case_ids
        ):
            raise ValueError("snapshot index case identity is invalid")
        seen_case_ids.add(case_id)
        expected_snapshot_path = f"snapshots/{case_id}.json"
        if case_row["snapshot_path"] != expected_snapshot_path:
            raise ValueError("snapshot case path is invalid")
        snapshot = _load_json(artifact_root / expected_snapshot_path)
        _validate_frozen_snapshot(
            snapshot,
            case_row=case_row,
            snapshot_index=snapshot_index,
        )
        snapshot_digests.append(str(case_row["snapshot_digest"]))
    if _stable_digest(snapshot_digests) != (
        snapshot_index["snapshot_set_digest"]
    ):
        raise ValueError("snapshot-set digest changed")
    return snapshot_index


def _validate_frozen_snapshot(
    snapshot: Mapping[str, Any],
    *,
    case_row: Mapping[str, Any],
    snapshot_index: Mapping[str, Any],
) -> None:
    """Verify one frozen case and every digest used by execution."""

    expected_fields = {
        "schema_version",
        "case_id",
        "scenario_dimension",
        "captured_at",
        "source_identity",
        "source_digest",
        "profile_digest",
        "character_state_digest",
        "user_state_digest",
        "input_digest",
        "profile_snapshot_path",
        "source_snapshot",
        "input",
        "review_projection",
        "snapshot_digest",
    }
    if set(snapshot) != expected_fields:
        raise ValueError("frozen snapshot fields are not exact")
    if snapshot["schema_version"] != SNAPSHOT_SCHEMA:
        raise ValueError("frozen snapshot schema is unsupported")
    if (
        snapshot["case_id"] != case_row["case_id"]
        or snapshot["scenario_dimension"] != case_row["scenario_dimension"]
    ):
        raise ValueError("frozen snapshot case identity changed")
    if snapshot["profile_snapshot_path"] != (
        snapshot_index["profile_snapshot_path"]
    ):
        raise ValueError("frozen snapshot profile path changed")
    digest_pairs = (
        ("profile_digest", snapshot_index["profile_digest"]),
        (
            "character_state_digest",
            snapshot_index["character_state_digest"],
        ),
        ("input_digest", case_row["input_digest"]),
        ("snapshot_digest", case_row["snapshot_digest"]),
    )
    for field_name, expected_digest in digest_pairs:
        if snapshot[field_name] != expected_digest:
            raise ValueError(f"frozen snapshot {field_name} changed")
    source_snapshot = snapshot["source_snapshot"]
    if _stable_digest(source_snapshot) != snapshot["source_digest"]:
        raise ValueError("frozen snapshot source digest changed")
    input_payload = validate_cognition_core_input(snapshot["input"])
    if _stable_digest(input_payload) != snapshot["input_digest"]:
        raise ValueError("frozen snapshot input digest changed")
    if (
        _stable_digest(input_payload["mutable_state"])
        != snapshot["user_state_digest"]
    ):
        raise ValueError("frozen snapshot user-state digest changed")
    calculated_snapshot_digest = _snapshot_payload_digest(snapshot)
    if calculated_snapshot_digest != snapshot["snapshot_digest"]:
        raise ValueError("frozen snapshot digest changed")


def _snapshot_payload_digest(snapshot: Mapping[str, Any]) -> str:
    """Digest one case while excluding capture time and its own digest."""

    digest_payload = {
        key: value
        for key, value in snapshot.items()
        if key not in {"captured_at", "snapshot_digest"}
    }
    digest = _stable_digest(digest_payload)
    return digest


def build_ledger_contract(
    snapshot_rows: Sequence[Mapping[str, Any]],
    *,
    snapshot_set_digest: str,
    route_profile_digest: str,
    blind_seed: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the balanced 384-row ledger and separate unblinding key."""

    if len(snapshot_rows) != CASE_COUNT:
        raise ValueError(f"ledger requires {CASE_COUNT} snapshot rows")
    if not blind_seed:
        raise ValueError("ledger blind seed must be non-empty")
    matrix = enumerate_assignment_matrix()
    rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    block_index = 0
    sample_index = 0
    for snapshot_row in snapshot_rows:
        case_id = snapshot_row["case_id"]
        input_digest = snapshot_row["input_digest"]
        for repetition in range(1, REPETITION_COUNT + 1):
            rotation = (ROTATION_STEP * block_index) % MATRIX_CELL_COUNT
            ordered_cells = [
                *matrix[rotation:],
                *matrix[:rotation],
            ]
            if block_index % 2 == 1:
                ordered_cells.reverse()
            for cell in ordered_cells:
                sample_index += 1
                sample_id = f"sample_{sample_index:04d}"
                blind_label = _blind_label(
                    blind_seed,
                    sample_id=sample_id,
                    case_id=str(case_id),
                    repetition=repetition,
                    cell_id=str(cell["cell_id"]),
                )
                assignment_digest = _assignment_digest(
                    blind_seed,
                    cell,
                )
                artifact_path = f"runs/{blind_label}.json"
                inspection_path = f"inspections/{blind_label}.json"
                rows.append({
                    "sequence": sample_index,
                    "sample_id": sample_id,
                    "blind_label": blind_label,
                    "case_id": case_id,
                    "repetition": repetition,
                    "cell_id": cell["cell_id"],
                    "input_digest": input_digest,
                    "status": "pending",
                    "artifact_path": artifact_path,
                    "inspection_path": inspection_path,
                    "inspection_status": "pending",
                    "technical_retry_count": 0,
                })
                assignments.append({
                    "sample_id": sample_id,
                    "blind_label": blind_label,
                    "cell_id": cell["cell_id"],
                    "assignment": cell["assignment"],
                    "assignment_digest": assignment_digest,
                })
            block_index += 1

    if sample_index != SAMPLE_COUNT:
        raise AssertionError("ledger construction produced an invalid count")
    ledger = {
        "schema_version": LEDGER_SCHEMA,
        "created_at": _utc_now_iso(),
        "case_count": CASE_COUNT,
        "matrix_cell_count": MATRIX_CELL_COUNT,
        "repetition_count": REPETITION_COUNT,
        "sample_count": SAMPLE_COUNT,
        "snapshot_set_digest": snapshot_set_digest,
        "route_profile_digest": route_profile_digest,
        "rows": rows,
    }
    unblinding_key = {
        "schema_version": UNBLINDING_SCHEMA,
        "created_at": ledger["created_at"],
        "blind_seed_digest": _stable_digest(blind_seed),
        "route_profile_digest": route_profile_digest,
        "assignments": assignments,
    }
    return ledger, unblinding_key


def verify_unblinding_contract(
    unblinding_key: Mapping[str, Any],
    *,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the separate key maps every ledger row to its canonical cell."""

    expected_fields = {
        "schema_version",
        "created_at",
        "blind_seed_digest",
        "route_profile_digest",
        "assignments",
    }
    if set(unblinding_key) != expected_fields:
        raise ValueError("unblinding key fields are not exact")
    if unblinding_key["schema_version"] != UNBLINDING_SCHEMA:
        raise ValueError("unblinding key schema is unsupported")
    if unblinding_key["created_at"] != ledger["created_at"]:
        raise ValueError("unblinding key creation time changed")
    if (
        unblinding_key["route_profile_digest"]
        != ledger["route_profile_digest"]
    ):
        raise ValueError("unblinding key route digest changed")
    if not _is_sha256_digest(unblinding_key["blind_seed_digest"]):
        raise ValueError("unblinding key seed digest is invalid")

    ledger_rows = ledger.get("rows")
    assignments = unblinding_key["assignments"]
    if (
        not isinstance(ledger_rows, list)
        or not isinstance(assignments, list)
        or len(assignments) != len(ledger_rows)
        or len(assignments) != SAMPLE_COUNT
    ):
        raise ValueError("unblinding assignment count is invalid")
    matrix_by_cell = {
        cell["cell_id"]: cell["assignment"]
        for cell in enumerate_assignment_matrix()
    }
    assignment_digests_by_cell: dict[str, str] = {}
    seen_sample_ids: set[str] = set()
    for ledger_row, assignment_value in zip(
        ledger_rows,
        assignments,
        strict=True,
    ):
        if not isinstance(ledger_row, Mapping):
            raise ValueError("ledger row must be an object")
        if not isinstance(assignment_value, Mapping):
            raise ValueError("unblinding assignment must be an object")
        assignment_row = assignment_value
        expected_assignment_fields = {
            "sample_id",
            "blind_label",
            "cell_id",
            "assignment",
            "assignment_digest",
        }
        if set(assignment_row) != expected_assignment_fields:
            raise ValueError("unblinding assignment fields are not exact")
        identity_fields = ("sample_id", "blind_label", "cell_id")
        if any(
            assignment_row[field_name] != ledger_row[field_name]
            for field_name in identity_fields
        ):
            raise ValueError("unblinding assignment identity changed")
        sample_id = str(assignment_row["sample_id"])
        if sample_id in seen_sample_ids:
            raise ValueError("unblinding sample assignments are duplicated")
        seen_sample_ids.add(sample_id)
        cell_id = str(assignment_row["cell_id"])
        if cell_id not in matrix_by_cell:
            raise ValueError("unblinding assignment cell is invalid")
        raw_assignment = assignment_row["assignment"]
        if (
            not isinstance(raw_assignment, Mapping)
            or dict(raw_assignment) != matrix_by_cell[cell_id]
        ):
            raise ValueError("unblinding assignment mapping changed")
        assignment_digest = assignment_row["assignment_digest"]
        if not _is_sha256_digest(assignment_digest):
            raise ValueError("unblinding assignment digest is invalid")
        prior_digest = assignment_digests_by_cell.setdefault(
            cell_id,
            str(assignment_digest),
        )
        if prior_digest != assignment_digest:
            raise ValueError("unblinding assignment digest changed by sample")
    if len(set(assignment_digests_by_cell.values())) != MATRIX_CELL_COUNT:
        raise ValueError("unblinding cell digests are not distinct")

    verification = {
        "status": "valid",
        "assignment_count": len(assignments),
        "cell_count": len(assignment_digests_by_cell),
    }
    return verification


def initialize_ledger(
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    blind_seed: str | None = None,
) -> dict[str, Any]:
    """Create or verify the execution ledger without making model calls."""

    snapshot_index = load_snapshot_index(artifact_root=artifact_root)
    services = build_cognition_core_services()
    current_route_digest = route_profile_digest(services)
    ledger_path = artifact_root / "ledger.json"
    unblinding_path = artifact_root / "unblinding_key.json"
    if ledger_path.exists() or unblinding_path.exists():
        if not ledger_path.exists() or not unblinding_path.exists():
            raise RuntimeError("ledger initialization is partially present")
        existing_ledger = _load_json(ledger_path)
        existing_unblinding_key = _load_json(unblinding_path)
        verification = verify_ledger_contract(
            existing_ledger,
            snapshot_index=snapshot_index,
            route_profile_digest=current_route_digest,
            require_artifacts=False,
            artifact_root=artifact_root,
        )
        unblinding_verification = verify_unblinding_contract(
            existing_unblinding_key,
            ledger=existing_ledger,
        )
        verification["unblinding_assignment_count"] = (
            unblinding_verification["assignment_count"]
        )
        return verification

    selected_seed = blind_seed or secrets.token_hex(32)
    ledger, unblinding_key = build_ledger_contract(
        snapshot_index["cases"],
        snapshot_set_digest=snapshot_index["snapshot_set_digest"],
        route_profile_digest=current_route_digest,
        blind_seed=selected_seed,
    )
    verify_unblinding_contract(unblinding_key, ledger=ledger)
    _write_json(ledger_path, ledger)
    _write_json(unblinding_path, unblinding_key)
    verification = verify_ledger_contract(
        ledger,
        snapshot_index=snapshot_index,
        route_profile_digest=current_route_digest,
        require_artifacts=False,
        artifact_root=artifact_root,
    )
    verification["unblinding_assignment_count"] = SAMPLE_COUNT
    return verification


def verify_ledger_contract(
    ledger: Mapping[str, Any],
    *,
    snapshot_index: Mapping[str, Any],
    route_profile_digest: str,
    require_artifacts: bool,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Verify sample order, coverage, digests, statuses, and artifact refs."""

    expected_fields = {
        "schema_version",
        "created_at",
        "case_count",
        "matrix_cell_count",
        "repetition_count",
        "sample_count",
        "snapshot_set_digest",
        "route_profile_digest",
        "rows",
    }
    if set(ledger) != expected_fields:
        raise ValueError("ledger fields are not exact")
    if ledger["schema_version"] != LEDGER_SCHEMA:
        raise ValueError("ledger schema is unsupported")
    if (
        ledger["case_count"] != CASE_COUNT
        or ledger["matrix_cell_count"] != MATRIX_CELL_COUNT
        or ledger["repetition_count"] != REPETITION_COUNT
        or ledger["sample_count"] != SAMPLE_COUNT
    ):
        raise ValueError("ledger declared counts are invalid")
    if (
        ledger["snapshot_set_digest"]
        != snapshot_index["snapshot_set_digest"]
    ):
        raise ValueError("ledger snapshot digest changed")
    if ledger["route_profile_digest"] != route_profile_digest:
        raise ValueError("ledger route profile digest changed")

    rows = ledger["rows"]
    if not isinstance(rows, list) or len(rows) != SAMPLE_COUNT:
        raise ValueError(f"ledger must contain {SAMPLE_COUNT} rows")
    snapshot_by_case = {
        row["case_id"]: row
        for row in snapshot_index["cases"]
    }
    case_ids = list(snapshot_by_case)
    expected_order = _expected_case_repetition_cell_order(case_ids)
    actual_order: list[tuple[str, int, str]] = []
    sample_ids: set[str] = set()
    blind_labels: set[str] = set()
    coverage: Counter[tuple[str, int, str]] = Counter()
    status_counts: Counter[str] = Counter()
    inspection_counts: Counter[str] = Counter()
    for sequence, row_value in enumerate(rows, start=1):
        if not isinstance(row_value, Mapping):
            raise ValueError("ledger row must be an object")
        row = row_value
        expected_row_fields = {
            "sequence",
            "sample_id",
            "blind_label",
            "case_id",
            "repetition",
            "cell_id",
            "input_digest",
            "status",
            "artifact_path",
            "inspection_path",
            "inspection_status",
            "technical_retry_count",
        }
        if set(row) != expected_row_fields:
            raise ValueError("ledger row fields are not exact")
        if row.get("sequence") != sequence:
            raise ValueError("ledger row sequence is invalid")
        sample_id = row.get("sample_id")
        blind_label = row.get("blind_label")
        case_id = row.get("case_id")
        repetition = row.get("repetition")
        cell_id = row.get("cell_id")
        if not all(
            isinstance(value, str) and value
            for value in (sample_id, blind_label, case_id, cell_id)
        ):
            raise ValueError("ledger row identity is invalid")
        if not isinstance(repetition, int):
            raise ValueError("ledger repetition must be an integer")
        if sample_id != f"sample_{sequence:04d}":
            raise ValueError("ledger sample id is invalid")
        if not _is_blind_label(str(blind_label)):
            raise ValueError("ledger blind label is invalid")
        if sample_id in sample_ids or blind_label in blind_labels:
            raise ValueError("ledger sample and blind identities must be unique")
        sample_ids.add(sample_id)
        blind_labels.add(blind_label)
        if case_id not in snapshot_by_case:
            raise ValueError("ledger references an unknown case")
        expected_input_digest = snapshot_by_case[case_id]["input_digest"]
        if row.get("input_digest") != expected_input_digest:
            raise ValueError("ledger input digest changed")
        if row["artifact_path"] != f"runs/{blind_label}.json":
            raise ValueError("ledger artifact path is invalid")
        if row["inspection_path"] != f"inspections/{blind_label}.json":
            raise ValueError("ledger inspection path is invalid")
        status = row.get("status")
        inspection_status = row.get("inspection_status")
        if status not in LEDGER_STATUSES:
            raise ValueError("ledger sample status is invalid")
        if inspection_status not in INSPECTION_STATUSES:
            raise ValueError("ledger inspection status is invalid")
        if status not in TERMINAL_STATUSES and inspection_status != "pending":
            raise ValueError("non-terminal ledger row cannot be inspected")
        technical_retry_count = row["technical_retry_count"]
        if (
            isinstance(technical_retry_count, bool)
            or not isinstance(technical_retry_count, int)
            or technical_retry_count not in {0, 1}
        ):
            raise ValueError("ledger technical retry count is invalid")
        if status in TERMINAL_STATUSES and require_artifacts:
            artifact_path = artifact_root / str(row["artifact_path"])
            if not artifact_path.is_file():
                raise ValueError("terminal ledger row is missing its artifact")
            artifact = _load_json(artifact_path)
            _validate_terminal_artifact_reference(artifact, row=row)
        if inspection_status != "pending" and require_artifacts:
            inspection_path = artifact_root / str(row["inspection_path"])
            if not inspection_path.is_file():
                raise ValueError(
                    "inspected ledger row is missing its sidecar"
                )
        status_counts[str(status)] += 1
        inspection_counts[str(inspection_status)] += 1
        order_entry = (str(case_id), repetition, str(cell_id))
        actual_order.append(order_entry)
        coverage[order_entry] += 1
    if actual_order != expected_order:
        raise ValueError("ledger balanced order changed")
    if any(count != 1 for count in coverage.values()):
        raise ValueError("ledger sample coverage is not exact")

    terminal_count = sum(
        status_counts[status]
        for status in TERMINAL_STATUSES
    )
    inspected_count = (
        inspection_counts["accepted"]
        + inspection_counts["finding_recorded"]
    )
    verification = {
        "status": "valid",
        "sample_count": len(rows),
        "pending_count": status_counts["pending"],
        "running_count": status_counts["running"],
        "completed_count": status_counts["completed"],
        "transport_failed_count": status_counts["transport_failed"],
        "contract_failed_count": status_counts["contract_failed"],
        "terminal_count": terminal_count,
        "inspected_count": inspected_count,
        "snapshot_set_digest": ledger["snapshot_set_digest"],
        "route_profile_digest": ledger["route_profile_digest"],
    }
    return verification


def _validate_terminal_artifact_reference(
    artifact: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
) -> None:
    """Reconcile terminal artifact identity with its owning ledger row."""

    required_fields = {
        "schema_version",
        "sample_id",
        "blind_label",
        "case_id",
        "repetition",
        "input_digest",
        "terminal_status",
        "technical_retry_count",
    }
    if not required_fields.issubset(artifact):
        raise ValueError("terminal artifact fields are incomplete")
    if artifact["schema_version"] != SAMPLE_ARTIFACT_SCHEMA:
        raise ValueError("terminal artifact schema is unsupported")
    identity_fields = (
        "sample_id",
        "blind_label",
        "case_id",
        "repetition",
        "input_digest",
    )
    if any(
        artifact[field_name] != row[field_name]
        for field_name in identity_fields
    ):
        raise ValueError("terminal artifact identity changed")
    if artifact["terminal_status"] != row["status"]:
        raise ValueError("terminal artifact status changed")
    if (
        artifact["technical_retry_count"]
        != row["technical_retry_count"]
    ):
        raise ValueError("terminal artifact retry count changed")


def _expected_case_repetition_cell_order(
    case_ids: Sequence[str],
) -> list[tuple[str, int, str]]:
    """Return the fixed rotated/reversed cell sequence for every block."""

    matrix = enumerate_assignment_matrix()
    expected: list[tuple[str, int, str]] = []
    block_index = 0
    for case_id in case_ids:
        for repetition in range(1, REPETITION_COUNT + 1):
            rotation = (ROTATION_STEP * block_index) % MATRIX_CELL_COUNT
            ordered_cells = [
                *matrix[rotation:],
                *matrix[:rotation],
            ]
            if block_index % 2 == 1:
                ordered_cells.reverse()
            expected.extend(
                (
                    case_id,
                    repetition,
                    str(cell["cell_id"]),
                )
                for cell in ordered_cells
            )
            block_index += 1
    return expected


def _blind_label(
    blind_seed: str,
    *,
    sample_id: str,
    case_id: str,
    repetition: int,
    cell_id: str,
) -> str:
    """Build an opaque reviewer label from a secret ledger seed."""

    message = f"{sample_id}|{case_id}|{repetition}|{cell_id}"
    digest = hmac.new(
        blind_seed.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    blind_label = f"B{digest[:BLIND_LABEL_HEX_CHARS]}"
    return blind_label


def _assignment_digest(
    blind_seed: str,
    cell: Mapping[str, Any],
) -> str:
    """Build an opaque assignment fingerprint for a blinded artifact."""

    payload = _canonical_json(cell)
    digest = hmac.new(
        blind_seed.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest


class _CapturingLLM:
    """Capture factor calls while delegating to the configured interface."""

    def __init__(
        self,
        delegate: object,
        *,
        factor_by_config_id: Mapping[int, str],
        redaction_values: Sequence[str],
    ) -> None:
        self._delegate = delegate
        self._factor_by_config_id = dict(factor_by_config_id)
        self._redaction_values = tuple(redaction_values)
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        """Capture one ordered async factor call and preserve its failure."""

        call_record: dict[str, Any] = {
            "call_index": len(self.calls) + 1,
            "factor_binding": self._factor_by_config_id[id(config)],
            "config": _generation_config_projection(config),
            "messages": [
                str(message.content)
                for message in messages
            ],
            "started_at_monotonic": time.perf_counter(),
            "raw_output": None,
            "deterministic_parsed_output": None,
            "deterministic_parse_failure": None,
            "usage": {},
            "failure": None,
        }
        self.calls.append(call_record)
        try:
            response = await self._delegate.ainvoke(
                messages,
                config=config,
            )
        except Exception as exc:
            call_record["ended_at_monotonic"] = time.perf_counter()
            call_record["duration_ms"] = _duration_ms(call_record)
            call_record["failure"] = _redact_text(
                f"{type(exc).__name__}: {exc}",
                self._redaction_values,
            )
            raise

        raw_output = str(response.content)
        call_record["ended_at_monotonic"] = time.perf_counter()
        call_record["duration_ms"] = _duration_ms(call_record)
        call_record["raw_output"] = raw_output
        call_record["raw_output_sha256"] = _stable_digest(raw_output)
        parsed_output, parse_failure = _capture_deterministic_parse(
            raw_output,
            redaction_values=self._redaction_values,
        )
        call_record["deterministic_parsed_output"] = parsed_output
        call_record["deterministic_parse_failure"] = parse_failure
        call_record["usage"] = _json_projection(response.usage)
        return response

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        """Reject sync factor calls because Core V2 uses the async boundary."""

        del messages, config
        raise RuntimeError("Core V2 factor call unexpectedly used sync invoke")


class _RepairCapturingInvoker:
    """Capture actual JSON-repair calls without changing parser ownership."""

    def __init__(
        self,
        delegate: object,
        *,
        redaction_values: Sequence[str],
    ) -> None:
        self._delegate = delegate
        self._redaction_values = tuple(redaction_values)
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        """Capture one synchronous repair request and normalized response."""

        call_record: dict[str, Any] = {
            "call_index": len(self.calls) + 1,
            "config": _generation_config_projection(config),
            "messages": [
                str(message.content)
                for message in messages
            ],
            "started_at_monotonic": time.perf_counter(),
            "raw_output": None,
            "deterministic_parsed_output": None,
            "deterministic_parse_failure": None,
            "failure": None,
        }
        self.calls.append(call_record)
        try:
            response = self._delegate.invoke(messages, config=config)
        except Exception as exc:
            call_record["ended_at_monotonic"] = time.perf_counter()
            call_record["duration_ms"] = _duration_ms(call_record)
            call_record["failure"] = _redact_text(
                f"{type(exc).__name__}: {exc}",
                self._redaction_values,
            )
            raise

        raw_output = str(response.content)
        call_record["ended_at_monotonic"] = time.perf_counter()
        call_record["duration_ms"] = _duration_ms(call_record)
        call_record["raw_output"] = raw_output
        call_record["raw_output_sha256"] = _stable_digest(raw_output)
        parsed_output, parse_failure = _capture_deterministic_parse(
            raw_output,
            redaction_values=self._redaction_values,
        )
        call_record["deterministic_parsed_output"] = parsed_output
        call_record["deterministic_parse_failure"] = parse_failure
        return response

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        """Capture an asynchronous repair request if the parser uses one."""

        call_record: dict[str, Any] = {
            "call_index": len(self.calls) + 1,
            "config": _generation_config_projection(config),
            "messages": [
                str(message.content)
                for message in messages
            ],
            "started_at_monotonic": time.perf_counter(),
            "raw_output": None,
            "deterministic_parsed_output": None,
            "deterministic_parse_failure": None,
            "failure": None,
        }
        self.calls.append(call_record)
        try:
            response = await self._delegate.ainvoke(
                messages,
                config=config,
            )
        except Exception as exc:
            call_record["ended_at_monotonic"] = time.perf_counter()
            call_record["duration_ms"] = _duration_ms(call_record)
            call_record["failure"] = _redact_text(
                f"{type(exc).__name__}: {exc}",
                self._redaction_values,
            )
            raise
        raw_output = str(response.content)
        call_record["ended_at_monotonic"] = time.perf_counter()
        call_record["duration_ms"] = _duration_ms(call_record)
        call_record["raw_output"] = raw_output
        call_record["raw_output_sha256"] = _stable_digest(raw_output)
        parsed_output, parse_failure = _capture_deterministic_parse(
            raw_output,
            redaction_values=self._redaction_values,
        )
        call_record["deterministic_parsed_output"] = parsed_output
        call_record["deterministic_parse_failure"] = parse_failure
        return response


def _capture_deterministic_parse(
    raw_output: str,
    *,
    redaction_values: Sequence[str],
) -> tuple[dict[str, Any], str | None]:
    """Capture deterministic parse evidence without affecting stage behavior."""

    try:
        parsed_output = llm_utils.parse_llm_json_output(
            raw_output,
            deterministic_only=True,
        )
    except Exception as exc:
        failure = _redact_text(
            f"{type(exc).__name__}: {exc}",
            redaction_values,
        )
        return {}, failure
    return parsed_output, None


@contextmanager
def _capture_json_repairs(
    *,
    redaction_values: Sequence[str],
) -> Any:
    """Temporarily capture the canonical parser's real repair invocations."""

    original_invoker = llm_utils._parse_json_with_llm
    capture_invoker = _RepairCapturingInvoker(
        original_invoker,
        redaction_values=redaction_values,
    )
    llm_utils._parse_json_with_llm = capture_invoker
    try:
        yield capture_invoker
    finally:
        llm_utils._parse_json_with_llm = original_invoker


async def execute_sample(
    sample_row: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    base_services: CognitionCoreServicesV2,
    *,
    assignment: Mapping[str, Any],
    assignment_digest: str,
    artifact_root: Path = ARTIFACT_ROOT,
    run_cognition_func: RunCognition = run_cognition,
    model_fetcher: ModelFetcher = fetch_available_models,
) -> dict[str, Any]:
    """Execute one blinded sample from a deep-copied local input snapshot."""

    input_payload_value = snapshot["input"]
    if not isinstance(input_payload_value, Mapping):
        raise ValueError("snapshot cognition input must be an object")
    input_payload = validate_cognition_core_input(input_payload_value)
    actual_input_digest = _stable_digest(input_payload)
    expected_input_digest = sample_row.get("input_digest")
    if expected_input_digest and expected_input_digest != actual_input_digest:
        raise ValueError("sample input digest changed before execution")

    configured_values = _route_identity_values(base_services)
    attempt_artifacts: list[dict[str, Any]] = []
    terminal_status = "contract_failed"
    validated_output: CognitionCoreOutputV2 | None = None
    technical_retry_count = 0
    availability_recheck: dict[str, Any] | None = None
    for attempt_index in range(2):
        working_input = deepcopy(input_payload)
        services_without_capture = build_services_for_cell(
            base_services,
            assignment,
        )
        factor_by_config_id = {
            id(getattr(services_without_capture, service_field)): factor_field
            for factor_field in FACTOR_FIELDS
            for service_field in FACTOR_SERVICE_FIELDS[factor_field]
        }
        capture_llm = _CapturingLLM(
            base_services.llm,
            factor_by_config_id=factor_by_config_id,
            redaction_values=configured_values,
        )
        services = replace(
            services_without_capture,
            llm=capture_llm,
        )
        attempt_case_id = (
            f"{sample_row['blind_label']}:attempt:{attempt_index + 1}"
        )
        reset_validation_capture(attempt_case_id)
        output: CognitionCoreOutputV2 | None = None
        failure: dict[str, Any] | None = None
        started_at = time.perf_counter()
        with _capture_json_repairs(
            redaction_values=configured_values,
        ) as repair_capture:
            try:
                output = await run_cognition_func(
                    working_input,
                    services,
                )
                output = validate_cognition_core_output(output)
            except Exception as exc:
                failure = _exception_evidence(
                    exc,
                    redaction_values=configured_values,
                )
                caught_exception = exc
            else:
                caught_exception = None
        capture = validation_capture_snapshot()
        attempt_artifact = {
            "attempt": attempt_index + 1,
            "duration_ms": round(
                (time.perf_counter() - started_at) * 1000
            ),
            "factor_calls": capture_llm.calls,
            "json_repair_calls": repair_capture.calls,
            "validation_capture": _sanitized_validation_capture(
                capture,
                redaction_values=configured_values,
            ),
            "output": output,
            "failure": failure,
        }
        attempt_artifacts.append(attempt_artifact)
        if caught_exception is None:
            terminal_status = "completed"
            validated_output = output
            break
        if _is_transport_failure(caught_exception):
            terminal_status = "transport_failed"
            if attempt_index == 0:
                try:
                    availability_recheck = (
                        await verify_configured_model_profiles(
                            base_services,
                            model_fetcher=model_fetcher,
                        )
                    )
                except (RuntimeError, ValueError) as exc:
                    availability_recheck = {
                        "status": "failed",
                        "error": _redact_text(
                            f"{type(exc).__name__}: {exc}",
                            configured_values,
                        ),
                    }
                    break
                technical_retry_count = 1
                continue
            break
        if isinstance(
            caught_exception,
            (CognitionExecutionError, CognitionContractError),
        ):
            terminal_status = "contract_failed"
            break
        raise caught_exception

    factor_call_counts = {
        factor_field: sum(
            call["factor_binding"] == factor_field
            for attempt in attempt_artifacts
            for call in attempt["factor_calls"]
        )
        for factor_field in FACTOR_FIELDS
    }
    artifact = {
        "schema_version": SAMPLE_ARTIFACT_SCHEMA,
        "sample_id": sample_row["sample_id"],
        "blind_label": sample_row["blind_label"],
        "case_id": sample_row["case_id"],
        "repetition": sample_row["repetition"],
        "input_digest": actual_input_digest,
        "assignment_digest": assignment_digest,
        "terminal_status": terminal_status,
        "technical_retry_count": technical_retry_count,
        "source_input": input_payload,
        "review_projection": snapshot["review_projection"],
        "attempts": attempt_artifacts,
        "validated_output": validated_output,
        "reaction_projection": _reaction_projection(validated_output),
        "factor_call_counts": factor_call_counts,
        "availability_recheck": availability_recheck,
        "completed_at": _utc_now_iso(),
    }
    redacted_artifact = _redact_artifact(
        artifact,
        redaction_values=configured_values,
    )
    artifact_path = artifact_root / str(sample_row["artifact_path"])
    _write_json(artifact_path, redacted_artifact)
    return redacted_artifact


def _reaction_projection(
    output: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Project reaction-bearing output fields for human quality review."""

    if output is None:
        return None
    fields_to_keep = (
        "intention",
        "admitted_bid",
        "supporting_bids",
        "affect_projection",
        "relationship_projection",
        "action_requests",
        "resolver_requests",
        "goal_resolution",
        "resolver_pending_resolution",
        "resolver_goal_progress",
        "resolver_progress",
        "selected_bid_reason",
        "private_monologue",
        "expression_policy",
        "diagnostics",
        "cognition_observability",
    )
    projection = {
        field_name: _json_projection(output[field_name])
        for field_name in fields_to_keep
        if field_name in output
    }
    return projection


def _sanitized_validation_capture(
    capture: Mapping[str, Any] | None,
    *,
    redaction_values: Sequence[str],
) -> dict[str, Any] | None:
    """Strip route identity while retaining prompts, parsed results, and errors."""

    if capture is None:
        return None
    sanitized = _json_projection(capture)
    if not isinstance(sanitized, dict):
        raise TypeError("validation capture projection must be an object")
    stages = sanitized.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            config = stage.get("config")
            if isinstance(config, dict):
                for field_name in ("route_name", "base_url", "model"):
                    config.pop(field_name, None)
    redacted = _redact_artifact(
        sanitized,
        redaction_values=redaction_values,
    )
    if not isinstance(redacted, dict):
        raise TypeError("redacted validation capture must be an object")
    return redacted


def _exception_evidence(
    exc: BaseException,
    *,
    redaction_values: Sequence[str],
) -> dict[str, Any]:
    """Project one typed failure without provider or credential identity."""

    failure = {
        "class": type(exc).__name__,
        "message": _redact_text(str(exc), redaction_values),
        "error_code": getattr(exc, "error_code", None),
        "stage": getattr(exc, "stage", None),
        "branch_id": getattr(exc, "branch_id", None),
        "attempt_count": getattr(exc, "attempt_count", None),
        "retryable": getattr(exc, "retryable", None),
        "transport_failure": _is_transport_failure(exc),
    }
    return failure


def _is_transport_failure(exc: BaseException) -> bool:
    """Recognize external transport failures through wrapped exception chains."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (
                httpx.TransportError,
                TimeoutError,
                ConnectionError,
                OSError,
            ),
        ):
            return True
        next_exception = current.__cause__ or current.__context__
        current = next_exception
    return False


def _duration_ms(call_record: Mapping[str, Any]) -> int:
    """Calculate one capture duration from monotonic boundaries."""

    started_at = float(call_record["started_at_monotonic"])
    ended_at = float(call_record["ended_at_monotonic"])
    duration_ms = max(0, round((ended_at - started_at) * 1000))
    return duration_ms


def _generation_config_projection(
    config: LLMCallConfig,
) -> dict[str, Any]:
    """Project generation settings while withholding endpoint identity."""

    projection = {
        "stage_name": config.stage_name,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_completion_tokens": config.max_completion_tokens,
        "presence_penalty": config.presence_penalty,
        "timeout_seconds": config.timeout_seconds,
        "thinking_enabled": config.thinking.enabled,
    }
    return projection


def _complete_config_projection(
    config: LLMCallConfig,
) -> dict[str, Any]:
    """Project every config field for an in-memory route identity digest."""

    projection = {
        field.name: _json_projection(getattr(config, field.name))
        for field in fields(LLMCallConfig)
    }
    return projection


def _route_identity_values(
    services: CognitionCoreServicesV2,
) -> tuple[str, ...]:
    """Collect configured values that must be redacted from blinded artifacts."""

    values: set[str] = set()
    configs = [
        getattr(services, service_field)
        for factor_field in FACTOR_FIELDS
        for service_field in FACTOR_SERVICE_FIELDS[factor_field]
    ]
    repair_config = getattr(
        llm_utils,
        "_parse_json_with_llm_config",
        None,
    )
    if isinstance(repair_config, LLMCallConfig):
        configs.append(repair_config)
    for config in configs:
        values.update({
            config.base_url,
            config.base_url.rstrip("/"),
            config.api_key,
            config.model,
        })
    clean_values = tuple(
        sorted(
            (
                value
                for value in values
                if isinstance(value, str) and value
            ),
            key=len,
            reverse=True,
        )
    )
    return clean_values


def _redact_text(text: str, values: Sequence[str]) -> str:
    """Replace configured route identities in one local evidence string."""

    redacted = text
    for value in values:
        redacted = redacted.replace(value, "[redacted-route-identity]")
    return redacted


def _redact_artifact(
    value: object,
    *,
    redaction_values: Sequence[str],
) -> object:
    """Recursively remove configured route identities from blinded evidence."""

    if isinstance(value, str):
        redacted = _redact_text(value, redaction_values)
        return redacted
    if isinstance(value, Mapping):
        redacted_mapping = {
            str(key): _redact_artifact(
                item,
                redaction_values=redaction_values,
            )
            for key, item in value.items()
        }
        return redacted_mapping
    if isinstance(value, list):
        redacted_list = [
            _redact_artifact(
                item,
                redaction_values=redaction_values,
            )
            for item in value
        ]
        return redacted_list
    return value


async def run_next_sample(
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    run_cognition_func: RunCognition = run_cognition,
    model_fetcher: ModelFetcher = fetch_available_models,
) -> dict[str, Any]:
    """Execute exactly one pending row after reconciling prior inspection."""

    with _exclusive_run_lock(artifact_root):
        result = await _run_next_sample_locked(
            artifact_root=artifact_root,
            run_cognition_func=run_cognition_func,
            model_fetcher=model_fetcher,
        )
    return result


@contextmanager
def _exclusive_run_lock(artifact_root: Path) -> Any:
    """Hold one process-exclusive lock for a complete run-next command."""

    artifact_root.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_root / "run-next.lock"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        file_descriptor = os.open(lock_path, flags)
    except FileExistsError as exc:
        raise RuntimeError(
            "another run-next command holds the execution lock"
        ) from exc
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as lock_file:
            lock_payload = {
                "process_id": os.getpid(),
                "acquired_at": _utc_now_iso(),
            }
            lock_file.write(_canonical_json(lock_payload))
        yield
    finally:
        lock_path.unlink(missing_ok=True)


async def _run_next_sample_locked(
    *,
    artifact_root: Path,
    run_cognition_func: RunCognition,
    model_fetcher: ModelFetcher,
) -> dict[str, Any]:
    """Execute one sample while the process-exclusive lock is held."""

    ledger_path = artifact_root / "ledger.json"
    snapshot_index = load_snapshot_index(artifact_root=artifact_root)
    ledger = _load_json(ledger_path)
    ledger_changed = _reconcile_inspection_sidecars(
        ledger,
        artifact_root=artifact_root,
    )
    if ledger_changed:
        _write_json_atomic(ledger_path, ledger)

    base_services = build_cognition_core_services()
    current_route_digest = route_profile_digest(base_services)
    verify_ledger_contract(
        ledger,
        snapshot_index=snapshot_index,
        route_profile_digest=current_route_digest,
        require_artifacts=True,
        artifact_root=artifact_root,
    )
    if any(
        row["status"] == "running"
        for row in ledger["rows"]
    ):
        raise RuntimeError("ledger contains an unresolved running sample")
    uninspected_terminal = [
        row
        for row in ledger["rows"]
        if (
            row["status"] in TERMINAL_STATUSES
            and row["inspection_status"] == "pending"
        )
    ]
    if uninspected_terminal:
        raise RuntimeError(
            "previous terminal sample requires an inspection sidecar"
        )
    pending_rows = [
        row
        for row in ledger["rows"]
        if row["status"] == "pending"
    ]
    if not pending_rows:
        raise RuntimeError("ledger has no pending sample")
    sample_row = pending_rows[0]

    unblinding_key = _load_json(artifact_root / "unblinding_key.json")
    verify_unblinding_contract(unblinding_key, ledger=ledger)
    assignment_row = _assignment_for_sample(
        unblinding_key,
        sample_id=sample_row["sample_id"],
    )
    matrix_by_cell = {
        cell["cell_id"]: cell
        for cell in enumerate_assignment_matrix()
    }
    assignment = matrix_by_cell[assignment_row["cell_id"]]
    snapshot_row = next(
        row
        for row in snapshot_index["cases"]
        if row["case_id"] == sample_row["case_id"]
    )
    snapshot_path = artifact_root / str(snapshot_row["snapshot_path"])
    snapshot = _load_json(snapshot_path)
    if snapshot["snapshot_digest"] != snapshot_row["snapshot_digest"]:
        raise RuntimeError("snapshot digest changed before model execution")

    sample_row["status"] = "running"
    _write_json_atomic(ledger_path, ledger)
    artifact = await execute_sample(
        sample_row,
        snapshot,
        base_services,
        assignment=assignment,
        assignment_digest=assignment_row["assignment_digest"],
        artifact_root=artifact_root,
        run_cognition_func=run_cognition_func,
        model_fetcher=model_fetcher,
    )
    sample_row["status"] = artifact["terminal_status"]
    sample_row["technical_retry_count"] = artifact[
        "technical_retry_count"
    ]
    _write_json_atomic(ledger_path, ledger)

    summary = {
        "sample_id": sample_row["sample_id"],
        "blind_label": sample_row["blind_label"],
        "case_id": sample_row["case_id"],
        "repetition": sample_row["repetition"],
        "terminal_status": sample_row["status"],
        "artifact_path": sample_row["artifact_path"],
        "inspection_path": sample_row["inspection_path"],
        "technical_retry_count": sample_row["technical_retry_count"],
    }
    if sample_row["status"] == "transport_failed":
        raise RuntimeError(
            f"sample {sample_row['blind_label']} exhausted transport retry"
        )
    return summary


def _assignment_for_sample(
    unblinding_key: Mapping[str, Any],
    *,
    sample_id: str,
) -> dict[str, Any]:
    """Load one exact assignment from the separate unblinding key."""

    if unblinding_key.get("schema_version") != UNBLINDING_SCHEMA:
        raise ValueError("unblinding key schema is unsupported")
    assignments = unblinding_key.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("unblinding assignments must be a list")
    matches = [
        dict(row)
        for row in assignments
        if isinstance(row, Mapping) and row.get("sample_id") == sample_id
    ]
    if len(matches) != 1:
        raise ValueError("sample assignment is missing or duplicated")
    return matches[0]


def _reconcile_inspection_sidecars(
    ledger: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> bool:
    """Apply parent-authored inspection dispositions to terminal rows."""

    changed = False
    rows = ledger.get("rows")
    if not isinstance(rows, list):
        raise ValueError("ledger rows must be a list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("ledger row must be mutable")
        if (
            row["status"] not in TERMINAL_STATUSES
            or row["inspection_status"] != "pending"
        ):
            continue
        inspection_path = artifact_root / str(row["inspection_path"])
        if not inspection_path.is_file():
            continue
        inspection = _load_json(inspection_path)
        expected_fields = {
            "schema_version",
            "blind_label",
            "inspection_status",
            "technical_notes",
            "inspected_at",
        }
        if set(inspection) != expected_fields:
            raise ValueError("inspection sidecar fields are not exact")
        if inspection["schema_version"] != INSPECTION_SCHEMA:
            raise ValueError("inspection sidecar schema is unsupported")
        if inspection["blind_label"] != row["blind_label"]:
            raise ValueError("inspection sidecar identity is invalid")
        if inspection["inspection_status"] not in {
            "accepted",
            "finding_recorded",
        }:
            raise ValueError("inspection disposition is invalid")
        if not isinstance(inspection["technical_notes"], str):
            raise ValueError("inspection technical notes must be text")
        row["inspection_status"] = inspection["inspection_status"]
        changed = True
    return changed


def ledger_status(
    *,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Return current ledger counts after applying inspection sidecars."""

    ledger_path = artifact_root / "ledger.json"
    ledger = _load_json(ledger_path)
    changed = _reconcile_inspection_sidecars(
        ledger,
        artifact_root=artifact_root,
    )
    if changed:
        _write_json_atomic(ledger_path, ledger)
    snapshot_index = load_snapshot_index(artifact_root=artifact_root)
    services = build_cognition_core_services()
    verification = verify_ledger_contract(
        ledger,
        snapshot_index=snapshot_index,
        route_profile_digest=route_profile_digest(services),
        require_artifacts=True,
        artifact_root=artifact_root,
    )
    unblinding_key = _load_json(artifact_root / "unblinding_key.json")
    unblinding_verification = verify_unblinding_contract(
        unblinding_key,
        ledger=ledger,
    )
    verification["unblinding_assignment_count"] = (
        unblinding_verification["assignment_count"]
    )
    return verification


def build_review_queue(
    *,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Emit raw blinded review references after every sample is inspected."""

    status = ledger_status(artifact_root=artifact_root)
    if (
        status["terminal_count"] != SAMPLE_COUNT
        or status["inspected_count"] != SAMPLE_COUNT
    ):
        raise RuntimeError(
            "review queue requires all terminal samples to be inspected"
        )
    ledger = _load_json(artifact_root / "ledger.json")
    snapshot_index = load_snapshot_index(artifact_root=artifact_root)
    snapshot_by_case = {
        row["case_id"]: row
        for row in snapshot_index["cases"]
    }
    review_rows = []
    for row in ledger["rows"]:
        snapshot_row = snapshot_by_case[row["case_id"]]
        review_rows.append({
            "blind_label": row["blind_label"],
            "case_id": row["case_id"],
            "scenario_dimension": snapshot_row["scenario_dimension"],
            "repetition": row["repetition"],
            "artifact_path": row["artifact_path"],
            "review_path": f"reviews/{row['blind_label']}.json",
        })
    queue = {
        "schema_version": REVIEW_QUEUE_SCHEMA,
        "created_at": _utc_now_iso(),
        "review_count": len(review_rows),
        "score_fields": list(SCORE_FIELDS),
        "verdict_values": list(VERDICT_VALUES),
        "rows": review_rows,
    }
    _write_json(artifact_root / "review_queue.json", queue)
    (artifact_root / "reviews").mkdir(parents=True, exist_ok=True)
    return queue


def aggregate_from_artifacts(
    *,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Load completed raw evidence and parent-authored reviews for arithmetic."""

    ledger = _load_json(artifact_root / "ledger.json")
    artifacts: dict[str, dict[str, Any]] = {}
    reviews: dict[str, dict[str, Any]] = {}
    for row in ledger["rows"]:
        blind_label = row["blind_label"]
        artifact_value = _load_json(
            artifact_root / str(row["artifact_path"]),
        )
        review_path = artifact_root / "reviews" / f"{blind_label}.json"
        review_value = _load_json(review_path)
        _validate_review(review_value, blind_label=blind_label)
        artifacts[blind_label] = artifact_value
        reviews[blind_label] = review_value
    aggregate = aggregate_matrix_evidence(
        ledger,
        artifacts=artifacts,
        reviews=reviews,
    )
    _write_json(artifact_root / "matrix_aggregate.json", aggregate)
    return aggregate


def _validate_review(
    review: Mapping[str, Any],
    *,
    blind_label: str,
) -> None:
    """Validate one parent-authored semantic review without judging it."""

    expected_fields = {
        "schema_version",
        "blind_label",
        "scores",
        "baseline_relative_verdict",
        "rationale",
        "reviewed_at",
    }
    if set(review) != expected_fields:
        raise ValueError("review fields are not exact")
    if review["schema_version"] != REVIEW_SCHEMA:
        raise ValueError("review schema is unsupported")
    if review["blind_label"] != blind_label:
        raise ValueError("review blind label is invalid")
    scores = review["scores"]
    if (
        not isinstance(scores, Mapping)
        or set(scores) != set(SCORE_FIELDS)
    ):
        raise ValueError("review score fields are invalid")
    for score in scores.values():
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 0 <= score <= 4
        ):
            raise ValueError("review score must be an integer from zero to four")
    if review["baseline_relative_verdict"] not in VERDICT_VALUES:
        raise ValueError("review verdict is invalid")
    rationale = review["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("review rationale must be non-empty")


def aggregate_matrix_evidence(
    ledger: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    reviews: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply fixed quality gates and assignment selection arithmetic."""

    rows = ledger["rows"]
    if not isinstance(rows, list) or len(rows) != SAMPLE_COUNT:
        raise ValueError(f"aggregate requires {SAMPLE_COUNT} ledger rows")
    matrix = enumerate_assignment_matrix()
    cell_by_id = {
        cell["cell_id"]: cell
        for cell in matrix
    }
    scores_by_cell: dict[str, list[float]] = {
        cell_id: []
        for cell_id in cell_by_id
    }
    dimensions_by_cell: dict[str, dict[str, list[int]]] = {
        cell_id: {
            score_field: []
            for score_field in SCORE_FIELDS
        }
        for cell_id in cell_by_id
    }
    verdicts_by_cell: dict[str, list[str]] = {
        cell_id: []
        for cell_id in cell_by_id
    }
    verdicts_by_cell_case: dict[str, dict[str, list[str]]] = {
        cell_id: {}
        for cell_id in cell_by_id
    }
    statuses_by_cell: dict[str, Counter[str]] = {
        cell_id: Counter()
        for cell_id in cell_by_id
    }
    for row in rows:
        blind_label = row["blind_label"]
        cell_id = row["cell_id"]
        case_id = row["case_id"]
        if blind_label not in artifacts or blind_label not in reviews:
            raise ValueError("aggregate evidence is incomplete")
        artifact = artifacts[blind_label]
        review = reviews[blind_label]
        scores = review["scores"]
        if not isinstance(scores, Mapping):
            raise ValueError("aggregate review scores must be an object")
        dimension_values = [
            int(scores[score_field])
            for score_field in SCORE_FIELDS
        ]
        scores_by_cell[cell_id].append(
            statistics.mean(dimension_values)
        )
        for score_field in SCORE_FIELDS:
            dimensions_by_cell[cell_id][score_field].append(
                int(scores[score_field])
            )
        verdict = str(review["baseline_relative_verdict"])
        if verdict not in VERDICT_VALUES:
            raise ValueError("aggregate review verdict is invalid")
        verdicts_by_cell[cell_id].append(verdict)
        verdicts_by_cell_case[cell_id].setdefault(case_id, []).append(
            verdict
        )
        terminal_status = str(artifact["terminal_status"])
        if terminal_status not in TERMINAL_STATUSES:
            raise ValueError("aggregate artifact is not terminal")
        statuses_by_cell[cell_id][terminal_status] += 1

    baseline_mean = statistics.mean(scores_by_cell["Q00"])
    baseline_input_mean = statistics.mean(
        dimensions_by_cell["Q00"]["input_responsiveness"]
    )
    baseline_character_mean = statistics.mean(
        dimensions_by_cell["Q00"]["character_state_consistency"]
    )
    baseline_call_totals = _baseline_factor_call_totals(
        rows,
        artifacts=artifacts,
    )
    total_baseline_calls = sum(baseline_call_totals.values())
    if total_baseline_calls <= 0:
        raise ValueError("baseline artifacts contain no factor calls")

    cell_results: list[dict[str, Any]] = []
    for cell_id, cell in cell_by_id.items():
        cell_scores = scores_by_cell[cell_id]
        aggregate_mean = statistics.mean(cell_scores)
        input_mean = statistics.mean(
            dimensions_by_cell[cell_id]["input_responsiveness"]
        )
        character_mean = statistics.mean(
            dimensions_by_cell[cell_id][
                "character_state_consistency"
            ]
        )
        verdicts = verdicts_by_cell[cell_id]
        verdict_counts = Counter(verdicts)
        equivalent_or_better_ratio = (
            verdict_counts["better"] + verdict_counts["equivalent"]
        ) / len(verdicts)
        repeated_material_case = any(
            sum(verdict == "material_loss" for verdict in case_verdicts) > 1
            for case_verdicts in verdicts_by_cell_case[cell_id].values()
        )
        materially_weak_case = any(
            statistics.median(
                VERDICT_RANKS[verdict]
                for verdict in case_verdicts
            ) <= VERDICT_RANKS["material_loss"]
            for case_verdicts in verdicts_by_cell_case[cell_id].values()
        )
        statuses = statuses_by_cell[cell_id]
        assignment = cell["assignment"]
        moe_call_count = sum(
            baseline_call_totals[factor_field]
            for factor_field in FACTOR_FIELDS
            if assignment[factor_field] == "M"
        )
        moe_call_share = moe_call_count / total_baseline_calls
        quality_qualified = all((
            sum(statuses.values()) == CASE_COUNT * REPETITION_COUNT,
            statuses["contract_failed"] == 0,
            verdict_counts["critical_loss"] == 0,
            not materially_weak_case,
            aggregate_mean >= baseline_mean * 0.9,
            input_mean >= baseline_input_mean - 0.25,
            character_mean >= baseline_character_mean - 0.25,
            equivalent_or_better_ratio >= 0.8,
            verdict_counts["material_loss"] <= 1,
            not repeated_material_case,
        ))
        route_group_count = len(set(assignment.values()))
        cell_results.append({
            "cell_id": cell_id,
            "assignment": dict(assignment),
            "sample_count": len(cell_scores),
            "aggregate_mean": round(aggregate_mean, 4),
            "dimension_means": {
                score_field: round(
                    statistics.mean(
                        dimensions_by_cell[cell_id][score_field]
                    ),
                    4,
                )
                for score_field in SCORE_FIELDS
            },
            "verdict_counts": dict(verdict_counts),
            "status_counts": dict(statuses),
            "equivalent_or_better_ratio": round(
                equivalent_or_better_ratio,
                4,
            ),
            "moe_call_share": round(moe_call_share, 6),
            "route_group_count": route_group_count,
            "quality_qualified": quality_qualified,
        })

    qualified_cells = [
        cell
        for cell in cell_results
        if cell["quality_qualified"]
    ]
    qualified_cells.sort(
        key=lambda cell: (
            -float(cell["moe_call_share"]),
            -float(cell["aggregate_mean"]),
            int(cell["route_group_count"]),
            str(cell["cell_id"]),
        )
    )
    selected_cell = qualified_cells[0] if qualified_cells else None
    main_effects = _factor_main_effects(rows, reviews=reviews)
    pairwise_effects = _pairwise_interaction_effects(
        rows,
        reviews=reviews,
        cell_by_id=cell_by_id,
    )
    aggregate = {
        "schema_version": AGGREGATE_SCHEMA,
        "sample_count": len(rows),
        "baseline_cell_id": "Q00",
        "baseline_aggregate_mean": round(baseline_mean, 4),
        "baseline_factor_call_totals": baseline_call_totals,
        "cells": sorted(
            cell_results,
            key=lambda cell: cell["cell_id"],
        ),
        "main_effects": main_effects,
        "pairwise_interaction_effects": pairwise_effects,
        "qualified_cell_ids": [
            cell["cell_id"]
            for cell in qualified_cells
        ],
        "selected_cell_id": (
            selected_cell["cell_id"]
            if selected_cell is not None
            else None
        ),
        "selected_assignment": (
            selected_cell["assignment"]
            if selected_cell is not None
            else None
        ),
    }
    return aggregate


def _baseline_factor_call_totals(
    rows: Sequence[Mapping[str, Any]],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Sum baseline-observed calls for each factor binding."""

    totals = {
        factor_field: 0
        for factor_field in FACTOR_FIELDS
    }
    for row in rows:
        if row["cell_id"] != "Q00":
            continue
        artifact = artifacts[row["blind_label"]]
        raw_counts = artifact["factor_call_counts"]
        if not isinstance(raw_counts, Mapping):
            raise ValueError("artifact factor call counts must be an object")
        for factor_field in FACTOR_FIELDS:
            count = raw_counts[factor_field]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("artifact factor call count is invalid")
            totals[factor_field] += count
    return totals


def _sample_mean(review: Mapping[str, Any]) -> float:
    """Calculate one five-dimension sample mean."""

    scores = review["scores"]
    if not isinstance(scores, Mapping):
        raise ValueError("review scores must be an object")
    sample_mean = statistics.mean(
        int(scores[score_field])
        for score_field in SCORE_FIELDS
    )
    return sample_mean


def _factor_main_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    reviews: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """Calculate M-minus-D quality effects for each factor."""

    matrix_by_cell = {
        cell["cell_id"]: cell["assignment"]
        for cell in enumerate_assignment_matrix()
    }
    effects: dict[str, float] = {}
    for factor_field in FACTOR_FIELDS:
        level_d_scores: list[float] = []
        level_m_scores: list[float] = []
        for row in rows:
            assignment = matrix_by_cell[row["cell_id"]]
            sample_score = _sample_mean(reviews[row["blind_label"]])
            if assignment[factor_field] == "D":
                level_d_scores.append(sample_score)
            else:
                level_m_scores.append(sample_score)
        effects[factor_field] = round(
            statistics.mean(level_m_scores)
            - statistics.mean(level_d_scores),
            4,
        )
    return effects


def _pairwise_interaction_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    reviews: Mapping[str, Mapping[str, Any]],
    cell_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """Calculate factorial pair interactions averaged over other factors."""

    interactions: dict[str, float] = {}
    for left_index, left_factor in enumerate(FACTOR_FIELDS):
        for right_factor in FACTOR_FIELDS[left_index + 1:]:
            grouped_scores: dict[tuple[str, str], list[float]] = {
                (left_level, right_level): []
                for left_level in MODEL_LEVELS
                for right_level in MODEL_LEVELS
            }
            for row in rows:
                cell = cell_by_id[row["cell_id"]]
                assignment = cell["assignment"]
                level_pair = (
                    assignment[left_factor],
                    assignment[right_factor],
                )
                grouped_scores[level_pair].append(
                    _sample_mean(reviews[row["blind_label"]])
                )
            means = {
                levels: statistics.mean(scores)
                for levels, scores in grouped_scores.items()
            }
            interaction = (
                means[("M", "M")]
                - means[("M", "D")]
                - means[("D", "M")]
                + means[("D", "D")]
            )
            key = f"{left_factor}__{right_factor}"
            interactions[key] = round(interaction, 4)
    return interactions


def _load_json(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object from a local artifact path."""

    raw_text = path.read_text(encoding="utf-8")
    decoded = json.loads(raw_text)
    if not isinstance(decoded, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return decoded


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one UTF-8 structured evidence object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    projected = _json_projection(payload)
    artifact_text = json.dumps(
        projected,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(f"{artifact_text}\n", encoding="utf-8")


def _write_frozen_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    ignored_fields: set[str] | None = None,
) -> None:
    """Write immutable snapshot evidence or verify an identical existing file."""

    if path.exists():
        existing = _load_json(path)
        projected_payload = _json_projection(payload)
        if not isinstance(projected_payload, dict):
            raise TypeError("frozen JSON payload must be an object")
        excluded_fields = ignored_fields or set()
        existing_comparable = {
            key: value
            for key, value in existing.items()
            if key not in excluded_fields
        }
        payload_comparable = {
            key: value
            for key, value in projected_payload.items()
            if key not in excluded_fields
        }
        if existing_comparable != payload_comparable:
            raise RuntimeError(f"frozen artifact changed: {path}")
        return
    _write_json(path, payload)


def _write_frozen_text(path: Path, text: str) -> None:
    """Write immutable UTF-8 text or verify an identical existing file."""

    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise RuntimeError(f"frozen artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace one local ledger file from a fully written sibling file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    _write_json(temporary_path, payload)
    temporary_path.replace(path)


def _json_projection(value: object) -> object:
    """Convert local mappings and runtime objects to JSON-safe values."""

    if is_dataclass(value) and not isinstance(value, type):
        projection = _json_projection(asdict(value))
        return projection
    if isinstance(value, Mapping):
        projection = {
            str(key): _json_projection(item)
            for key, item in value.items()
        }
        return projection
    if isinstance(value, (list, tuple)):
        projection = [
            _json_projection(item)
            for item in value
        ]
        return projection
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_json(value: object) -> str:
    """Serialize one value deterministically for a digest."""

    projected = _json_projection(value)
    canonical = json.dumps(
        projected,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return canonical


def _stable_digest(value: object) -> str:
    """Return a SHA-256 digest of one canonical JSON projection."""

    canonical = _canonical_json(value)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest


def _is_sha256_digest(value: object) -> bool:
    """Return whether a value is one lowercase hexadecimal SHA-256 digest."""

    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _is_blind_label(value: str) -> bool:
    """Return whether a label matches the fixed opaque artifact form."""

    if (
        len(value) != BLIND_LABEL_HEX_CHARS + 1
        or not value.startswith("B")
    ):
        return False
    return all(
        character in "0123456789abcdef"
        for character in value[1:]
    )


def _utc_now_iso() -> str:
    """Return a second-precision UTC timestamp for local evidence metadata."""

    now = datetime.now(timezone.utc)
    timestamp = now.isoformat(timespec="seconds")
    return timestamp


def _build_parser() -> argparse.ArgumentParser:
    """Build the character-neutral matrix command-line contract."""

    parser = argparse.ArgumentParser(
        description="Cognition Core V2 model-assignment quality matrix",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument(
        "--case-manifest",
        type=Path,
        required=True,
    )
    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument(
        "--case-manifest",
        type=Path,
        required=True,
    )
    subparsers.add_parser("initialize-ledger")
    subparsers.add_parser("run-next")
    subparsers.add_parser("status")
    subparsers.add_parser("verify-ledger")
    subparsers.add_parser("build-review-queue")
    subparsers.add_parser("aggregate")
    return parser


async def _dispatch_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run one exact CLI command and return structured status evidence."""

    if args.command == "preflight":
        result = await preflight_environment(args.case_manifest)
    elif args.command == "snapshot":
        result = await build_case_snapshots(args.case_manifest)
    elif args.command == "initialize-ledger":
        result = initialize_ledger()
    elif args.command == "run-next":
        result = await run_next_sample()
    elif args.command == "status":
        result = ledger_status()
    elif args.command == "verify-ledger":
        result = ledger_status()
    elif args.command == "build-review-queue":
        result = build_review_queue()
    elif args.command == "aggregate":
        result = aggregate_from_artifacts()
    else:
        raise ValueError(f"unsupported command: {args.command}")
    return result


def main() -> int:
    """Run one process-bounded matrix command with structured failure output."""

    parser = _build_parser()
    args = parser.parse_args()
    try:
        result = asyncio.run(_dispatch_command(args))
    except Exception as exc:
        failure = {
            "status": "failed",
            "error_class": type(exc).__name__,
            "error": (
                "command failed; exception details withheld by the "
                "credential-safe CLI boundary"
            ),
        }
        print(
            json.dumps(failure, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
