"""Focused tests for the revision-zero character identity seed boundary."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_character_production_state,
)
from kazusa_ai_chatbot.db import character as character_module


FORBIDDEN_PROFILE_FIELDS = (
    "_id",
    "global_user_id",
    "cognition_state",
    "updated_at",
    "mood",
    "global_vibe",
    "reflection_summary",
    "tone",
    "speech_patterns",
)


def _valid_seed_payload(name: str = "Test Character") -> dict[str, object]:
    """Build one complete generic effective-identity seed."""

    return {
        "name": name,
        "description": "A deliberate observer learning through experience.",
        "gender": "unspecified",
        "age": 24,
        "birthday": "January 1",
        "backstory": "They value evidence, agency, and durable self-knowledge.",
        "personality_brief": {
            "mbti": "INTJ",
            "logic": "Tests assumptions against observed outcomes.",
            "tempo": "Measured and concise.",
            "defense": "Creates distance before reconsidering.",
            "quirks": "Pauses to compare present choices with earlier ones.",
            "taboos": "Rejects coerced identity claims.",
        },
        "boundary_profile": {
            "self_integrity": 0.8,
            "control_sensitivity": 0.6,
            "relational_override": 0.4,
            "control_intimacy_misread": 0.3,
            "authority_skepticism": 0.7,
            "compliance_strategy": "resist",
            "boundary_recovery": "rebound",
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.2,
            "hesitation_density": 0.2,
            "counter_questioning": 0.4,
            "softener_density": 0.3,
            "formalism_avoidance": 0.8,
            "abstraction_reframing": 0.5,
            "direct_assertion": 0.7,
            "emotional_leakage": 0.4,
            "rhythmic_bounce": 0.5,
            "self_deprecation": 0.1,
        },
        "self_image": {
            "self_concept": (
                "I am a deliberate observer whose choices can revise "
                "earlier assumptions."
            ),
            "current_growth_edges": [
                "Distinguish useful caution from avoidant distance.",
            ],
        },
        "visual_characterization": (
            "A composed adult with attentive posture and practical clothing."
        ),
    }


def _write_seed(path: Path, payload: dict[str, object]) -> Path:
    """Write one UTF-8 profile fixture and return its path."""

    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_profile_module_imports_before_database_package() -> None:
    """Profile loading must not depend on database-package import order."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from kazusa_ai_chatbot.character_profile import "
                "load_character_profile_seed"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stderr


def test_application_does_not_carry_packaged_profile() -> None:
    """The application must not bundle a packaged character profile."""

    repository_root = Path(__file__).resolve().parents[1]
    package_profile_directory = (
        repository_root
        / "src"
        / "kazusa_ai_chatbot"
        / "character_profiles"
    )
    assert not package_profile_directory.exists()


def test_every_repository_profile_is_a_complete_canonical_identity() -> None:
    """Each selectable profile must validate as one closed revision-zero body."""

    from kazusa_ai_chatbot.character_profile import load_character_profile_seed

    repository_root = Path(__file__).resolve().parents[1]
    profile_paths = sorted(
        (repository_root / "personalities").glob("*.json")
    )

    assert profile_paths
    for profile_path in profile_paths:
        seed = load_character_profile_seed(profile_path)
        assert frozenset(seed) == models.TOP_LEVEL_IDENTITY_KEYS
        assert seed["self_image"]["self_concept"].strip()
        assert len(seed["self_image"]["current_growth_edges"]) <= 5
        assert seed["visual_characterization"].strip()


def test_profile_loader_returns_validated_full_identity(tmp_path: Path) -> None:
    """The maintenance loader should return one complete identity snapshot."""

    from kazusa_ai_chatbot.character_profile import load_character_profile_seed

    profile_path = _write_seed(
        tmp_path / "profile.json",
        _valid_seed_payload(),
    )

    seed = load_character_profile_seed(profile_path)

    assert frozenset(seed) == models.TOP_LEVEL_IDENTITY_KEYS
    assert seed["name"] == "Test Character"
    assert seed["self_image"]["self_concept"]
    assert seed["visual_characterization"]


@pytest.mark.parametrize("field_name", FORBIDDEN_PROFILE_FIELDS)
def test_profile_loader_rejects_nonidentity_fields(
    tmp_path: Path,
    field_name: str,
) -> None:
    """Operational, retired, and duplicate fields stay outside identity."""

    from kazusa_ai_chatbot.character_profile import load_character_profile_seed

    payload = _valid_seed_payload()
    payload[field_name] = "forbidden"
    profile_path = _write_seed(tmp_path / "profile.json", payload)

    with pytest.raises(ValueError, match="unknown keys"):
        load_character_profile_seed(profile_path)


@pytest.mark.parametrize(
    ("profile_name", "field_name"),
    [
        ("personality_brief", "logic"),
        ("boundary_profile", "self_integrity"),
        ("linguistic_texture_profile", "fragmentation"),
        ("self_image", "self_concept"),
    ],
)
def test_profile_loader_rejects_missing_required_identity_fields(
    tmp_path: Path,
    profile_name: str,
    field_name: str,
) -> None:
    """Revision zero must contain every declared nested identity field."""

    from kazusa_ai_chatbot.character_profile import load_character_profile_seed

    payload = _valid_seed_payload()
    del payload[profile_name][field_name]
    profile_path = _write_seed(tmp_path / "profile.json", payload)

    with pytest.raises(ValueError, match=field_name):
        load_character_profile_seed(profile_path)


def test_profile_loader_accepts_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Maintenance callers may provide a path relative to their directory."""

    from kazusa_ai_chatbot.character_profile import load_character_profile_seed

    _write_seed(tmp_path / "profile.json", _valid_seed_payload())
    monkeypatch.chdir(tmp_path)

    seed = load_character_profile_seed(Path("profile.json"))

    assert seed["name"] == "Test Character"


@pytest.mark.asyncio
async def test_operational_character_state_insert_then_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean startup creates only cognition state and its timestamp."""

    class _Collection:
        def __init__(self) -> None:
            self.document: dict[str, object] | None = None
            self.insert_one = AsyncMock(side_effect=self._insert)

        async def find_one(self, *_args, **_kwargs):
            return self.document

        async def _insert(self, document: dict[str, object]):
            self.document = dict(document)

    class _Database:
        def __init__(self) -> None:
            self.character_state = _Collection()

    database = _Database()
    monkeypatch.setattr(
        character_module,
        "get_db",
        AsyncMock(return_value=database),
    )

    inserted = await character_module.ensure_operational_character_state()
    verified = await character_module.ensure_operational_character_state()

    assert inserted == "inserted"
    assert verified == "verified"
    assert frozenset(database.character_state.document) == {
        "_id",
        "cognition_state",
        "updated_at",
    }
    assert (
        database.character_state.document["cognition_state"]["state_scope"]
        == "character"
    )


@pytest.mark.asyncio
async def test_legacy_semantic_character_state_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Semantic singleton state must stop clean-cutover startup."""

    document = {
        "_id": "global",
        "name": "Legacy Character",
        "cognition_state": build_character_production_state(
            updated_at="2026-07-19T00:00:00Z",
        ),
        "updated_at": "2026-07-19T00:00:00Z",
    }

    class _Collection:
        async def find_one(self, *_args, **_kwargs):
            return document

    class _Database:
        character_state = _Collection()

    monkeypatch.setattr(
        character_module,
        "get_db",
        AsyncMock(return_value=_Database()),
    )

    with pytest.raises(
        character_module.LegacyCharacterStateError,
        match="clean target",
    ):
        await character_module.ensure_operational_character_state()


@pytest.mark.asyncio
async def test_profile_facade_composes_latest_identity_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The graph facade must compose max identity with operational state."""

    identity = _valid_seed_payload()
    cognition_state = build_character_production_state(
        updated_at="2026-07-19T00:00:00Z",
    )
    monkeypatch.setattr(
        character_module,
        "get_current_identity",
        AsyncMock(return_value={
            "revision_number": 4,
            "effective_identity": identity,
        }),
    )
    monkeypatch.setattr(
        character_module,
        "get_character_runtime_state",
        AsyncMock(return_value={
            "cognition_state": cognition_state,
            "updated_at": "2026-07-19T00:00:00Z",
        }),
    )

    profile = await character_module.get_character_profile(
        character_id="character-test",
    )

    assert profile["name"] == identity["name"]
    assert profile["self_image"] == identity["self_image"]
    assert profile["cognition_state"] == cognition_state
    assert profile["global_user_id"] == "character-test"
    assert "revision_number" not in profile


@pytest.mark.asyncio
async def test_clean_startup_crashes_without_pre_seeded_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean startup must crash if no identity revision is pre-seeded."""

    from kazusa_ai_chatbot import service as service_module
    from kazusa_ai_chatbot.db.character_identity_growth import (
        IdentityLedgerNotFoundError,
    )

    ensure_operational = AsyncMock(return_value="inserted")
    get_current = AsyncMock(
        side_effect=IdentityLedgerNotFoundError("missing"),
    )
    monkeypatch.setattr(
        service_module,
        "ensure_operational_character_state",
        ensure_operational,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "get_current_identity",
        get_current,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="No character identity revision"):
        await service_module._load_startup_character_profile()

    get_current.assert_awaited_once_with(
        character_id=service_module.CHARACTER_GLOBAL_USER_ID,
    )
    ensure_operational.assert_not_awaited()


@pytest.mark.asyncio
async def test_restart_uses_existing_latest_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restart must load max revision without reapplying a seed."""

    from kazusa_ai_chatbot import service as service_module

    identity = _valid_seed_payload(name="Grown Character")
    revision = {
        "revision_number": 7,
        "effective_identity": identity,
    }
    cognition_state = build_character_production_state(
        updated_at="2026-07-26T00:00:00Z",
    )
    ensure_operational = AsyncMock(return_value="verified")
    monkeypatch.setattr(
        service_module,
        "ensure_operational_character_state",
        ensure_operational,
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "get_current_identity",
        AsyncMock(return_value=revision),
        raising=False,
    )
    monkeypatch.setattr(
        service_module,
        "get_character_runtime_state",
        AsyncMock(return_value={
            "cognition_state": cognition_state,
            "updated_at": "2026-07-26T00:00:00Z",
        }),
    )

    static_profile, runtime_state = (
        await service_module._load_startup_character_profile()
    )

    ensure_operational.assert_awaited_once_with()
    assert static_profile == identity
    assert runtime_state["cognition_state"] == cognition_state


@pytest.mark.asyncio
async def test_runtime_profile_snapshot_refreshes_latest_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime profile reads should replace the prior revision completely."""

    from kazusa_ai_chatbot import service as service_module

    first_profile = {
        **_valid_seed_payload(name="revision-old"),
        "global_user_id": service_module.CHARACTER_GLOBAL_USER_ID,
    }
    latest_profile = {
        **_valid_seed_payload(name="revision-new"),
        "global_user_id": service_module.CHARACTER_GLOBAL_USER_ID,
    }
    get_profile = AsyncMock(side_effect=[first_profile, latest_profile])
    monkeypatch.setattr(
        service_module,
        "get_character_profile",
        get_profile,
    )

    first = await service_module._load_latest_character_profile_snapshot()
    latest = await service_module._load_latest_character_profile_snapshot()

    assert first["name"] == "revision-old"
    assert latest["name"] == "revision-new"
    assert service_module._active_character_name() == "revision-new"
    assert get_profile.await_count == 2
