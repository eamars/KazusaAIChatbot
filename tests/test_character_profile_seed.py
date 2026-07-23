"""Focused tests for the native character-profile seed boundary."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_character_production_state,
)
from kazusa_ai_chatbot.db import character as character_module


RUNTIME_OWNED_PROFILE_FIELDS = (
    "_id",
    "global_user_id",
    "self_image",
    "cognition_state",
    "updated_at",
    "mood",
    "global_vibe",
    "reflection_summary",
)


def _valid_seed_payload(name: str = "Test Character") -> dict[str, object]:
    """Build the smallest valid static profile used by seed tests."""

    boundary_profile = {
        "self_integrity": 0.8,
        "control_sensitivity": 0.6,
        "relational_override": 0.4,
        "control_intimacy_misread": 0.3,
        "authority_skepticism": 0.7,
        "compliance_strategy": "resist",
        "boundary_recovery": "rebound",
    }
    linguistic_texture_profile = {
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
    }
    return {
        "name": name,
        "personality_brief": {"core": "observant and direct"},
        "boundary_profile": boundary_profile,
        "linguistic_texture_profile": linguistic_texture_profile,
    }


def _write_seed(path: Path, payload: dict[str, object]) -> Path:
    """Write one UTF-8 profile fixture and return its absolute path."""

    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_profile_module_imports_before_database_package() -> None:
    """Profile loading must not depend on database-package import order."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from kazusa_ai_chatbot.character_profile import "
                "load_character_profile_seed; "
                "seed = load_character_profile_seed(Path("
                "'personalities/asuna.json').resolve()); "
                "assert seed['name'].endswith('(Ichinose Asuna)')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stderr


def test_checked_in_asuna_profile_is_valid_and_not_kazusa() -> None:
    """The active checked-in profile must be the Asuna personality seed."""

    from kazusa_ai_chatbot.character_profile import load_character_profile_seed

    profile_path = (
        Path(__file__).resolve().parents[1]
        / "personalities"
        / "asuna.json"
    )
    seed = load_character_profile_seed(profile_path)
    serialized_seed = json.dumps(seed, ensure_ascii=True)

    assert seed["name"].endswith("(Ichinose Asuna)")
    assert "Kazusa" not in serialized_seed


def test_active_profile_bindings_use_asuna_seed() -> None:
    """Runtime defaults must point at the active Asuna personality seed."""

    repository_root = Path(__file__).resolve().parents[1]
    binding_files = (
        repository_root / "Dockerfile",
        repository_root / "tests" / "conftest.py",
    )

    for binding_file in binding_files:
        binding_text = binding_file.read_text(encoding="utf-8")
        assert "asuna.json" in binding_text
        assert "kazusa.json" not in binding_text.casefold()


def test_profile_loader_returns_validated_static_seed(tmp_path: Path) -> None:
    """The loader should return static seed data from an absolute path."""

    from kazusa_ai_chatbot.character_profile import load_character_profile_seed

    profile_path = _write_seed(
        tmp_path / "profile.json",
        _valid_seed_payload(),
    )

    seed = load_character_profile_seed(profile_path)

    assert seed["name"] == "Test Character"
    assert "cognition_state" not in seed
    assert "updated_at" not in seed


def test_profile_loader_rejects_runtime_owned_fields(tmp_path: Path) -> None:
    """Static profile files must not contain runtime-owned singleton fields."""

    from kazusa_ai_chatbot.character_profile import load_character_profile_seed

    payload = _valid_seed_payload()
    payload["cognition_state"] = {"state_scope": "character"}
    profile_path = _write_seed(tmp_path / "profile.json", payload)

    with pytest.raises(ValueError, match="cognition_state"):
        load_character_profile_seed(profile_path)


@pytest.mark.parametrize("field_name", RUNTIME_OWNED_PROFILE_FIELDS)
def test_profile_loader_rejects_every_runtime_owned_field(
    tmp_path: Path,
    field_name: str,
) -> None:
    """Every mutable singleton field must stay outside static seed files."""

    from kazusa_ai_chatbot.character_profile import load_character_profile_seed

    payload = _valid_seed_payload()
    payload[field_name] = "runtime-owned"
    profile_path = _write_seed(tmp_path / "profile.json", payload)

    with pytest.raises(ValueError, match=field_name):
        load_character_profile_seed(profile_path)


@pytest.mark.parametrize(
    ("profile_name", "field_name"),
    [
        ("boundary_profile", "self_integrity"),
        ("linguistic_texture_profile", "fragmentation"),
    ],
)
def test_profile_loader_rejects_missing_required_profile_fields(
    tmp_path: Path,
    profile_name: str,
    field_name: str,
) -> None:
    """Static seeds must include the complete boundary and language vectors."""

    from kazusa_ai_chatbot.character_profile import load_character_profile_seed

    payload = _valid_seed_payload()
    del payload[profile_name][field_name]
    profile_path = _write_seed(tmp_path / "profile.json", payload)

    with pytest.raises(ValueError, match=field_name):
        load_character_profile_seed(profile_path)


def test_profile_loader_rejects_relative_paths() -> None:
    """Normal startup must receive an existing absolute profile path."""

    from kazusa_ai_chatbot.character_profile import load_character_profile_seed

    with pytest.raises(ValueError):
        load_character_profile_seed(Path("profile.json"))


@pytest.mark.asyncio
async def test_profile_seed_insert_then_verify_preserves_runtime_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Insert and repeat verification must preserve mutable state."""

    from kazusa_ai_chatbot.character_profile import load_character_profile_seed
    from kazusa_ai_chatbot.db.character import ensure_character_profile_seed

    seed = _valid_seed_payload()
    existing_runtime_state = build_character_production_state(
        updated_at="2026-07-19T00:00:00Z",
    )

    class _Collection:
        def __init__(self, existing: dict[str, object] | None) -> None:
            self.find_one = AsyncMock(return_value=existing)
            self.insert_one = AsyncMock()
            self.update_one = AsyncMock()

    class _Database:
        def __init__(self, existing: dict[str, object] | None) -> None:
            self.character_state = _Collection(existing)

    insert_db = _Database(None)
    monkeypatch.setattr(
        character_module,
        "get_db",
        AsyncMock(return_value=insert_db),
    )
    seed_path = _write_seed(tmp_path / "profile.json", seed)
    loaded_seed = load_character_profile_seed(seed_path)
    insert_result = await ensure_character_profile_seed(loaded_seed)

    assert insert_result == "inserted"
    inserted_document = insert_db.character_state.insert_one.await_args.args[0]
    assert inserted_document["_id"] == "global"
    assert inserted_document["name"] == "Test Character"
    assert inserted_document["cognition_state"]["state_scope"] == "character"

    verified_document = {
        **seed,
        "_id": "global",
        "cognition_state": existing_runtime_state,
        "updated_at": "2026-07-19T00:00:00+00:00",
        "mood": {"label": "steady"},
    }
    verify_db = _Database(verified_document)
    monkeypatch.setattr(
        character_module,
        "get_db",
        AsyncMock(return_value=verify_db),
    )

    verify_result = await ensure_character_profile_seed(seed)

    assert verify_result == "verified"
    verify_db.character_state.insert_one.assert_not_awaited()
    verify_db.character_state.update_one.assert_not_awaited()
    assert verified_document["mood"] == {"label": "steady"}


@pytest.mark.asyncio
async def test_profile_seed_rejects_conflicting_existing_identity(monkeypatch) -> None:
    """A different existing character identity must stop startup."""

    from kazusa_ai_chatbot.db.character import ensure_character_profile_seed

    class _Collection:
        find_one = AsyncMock(return_value={
            "_id": "global",
            "name": "Another Character",
            "cognition_state": build_character_production_state(
                updated_at="2026-07-19T00:00:00Z",
            ),
        })

    class _Database:
        character_state = _Collection()

    monkeypatch.setattr(
        character_module,
        "get_db",
        AsyncMock(return_value=_Database()),
    )

    with pytest.raises(ValueError, match="name"):
        await ensure_character_profile_seed(_valid_seed_payload())
