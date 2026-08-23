"""Deterministic startup skill discovery and lazy-loading tests."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pytest
import yaml

from agentic_resolver.contracts import (
    AgenticResolverContractError,
    AgenticResolverLimitsV1,
)
from agentic_resolver.json_protocol import skill_catalog_message
from agentic_resolver.skills import discover_skills

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "Instructions for a bounded sample task.",
    body: str = "# Instructions\n\nUse the supplied facts.",
) -> Path:
    """Create one exact one-level temporary skill bundle."""

    bundle = root / name
    bundle.mkdir(parents=True)
    skill_path = bundle / "SKILL.md"
    skill_path.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return skill_path


def test_startup_scan_discovers_one_level_skill_bundles(tmp_path: Path) -> None:
    """Only direct bundle directories with SKILL.md enter the fixed catalog."""

    _write_skill(tmp_path, "beta-skill")
    _write_skill(tmp_path, "alpha-skill")
    (tmp_path / "flat.md").write_text("ignored", encoding="utf-8")

    catalog = discover_skills([tmp_path])

    assert [definition.name for definition in catalog.definitions] == [
        "alpha-skill",
        "beta-skill",
    ]


def test_catalog_injection_is_json_name_description_only(tmp_path: Path) -> None:
    """Startup model context excludes paths, bodies, and executable metadata."""

    _write_skill(tmp_path, "sample-skill", body="SECRET BODY")
    catalog = discover_skills([tmp_path])
    message = skill_catalog_message(
        catalog_digest=catalog.catalog_digest,
        skills=catalog.summaries(),
    )
    parsed = json.loads(message)

    assert set(parsed["skills"][0]) == {"name", "description"}
    assert "SECRET BODY" not in message
    assert "SKILL.md" not in message


def test_skill_tool_loads_full_body_lazily(tmp_path: Path) -> None:
    """The complete Markdown body is available only after explicit selection."""

    _write_skill(tmp_path, "sample-skill", body="# Full Body\nDo the task.")
    catalog = discover_skills([tmp_path])

    definition = catalog.load("sample-skill")

    assert definition.body == "# Full Body\nDo the task."
    assert definition.source_path.name == "SKILL.md"


def test_malformed_duplicate_or_oversized_skills_fail_startup(
    tmp_path: Path,
) -> None:
    """Invalid catalogs fail before any model session can start."""

    malformed_root = tmp_path / "malformed"
    malformed_bundle = malformed_root / "bad-skill"
    malformed_bundle.mkdir(parents=True)
    (malformed_bundle / "SKILL.md").write_text(
        "name: bad-skill",
        encoding="utf-8",
    )
    with pytest.raises(AgenticResolverContractError, match="frontmatter"):
        discover_skills([malformed_root])

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_skill(first_root, "duplicate-skill")
    _write_skill(second_root, "duplicate-skill")
    with pytest.raises(AgenticResolverContractError, match="duplicate"):
        discover_skills([first_root, second_root])

    oversized_root = tmp_path / "oversized"
    _write_skill(oversized_root, "large-skill", body="x" * 20)
    with pytest.raises(AgenticResolverContractError, match="body"):
        discover_skills(
            [oversized_root],
            limits=AgenticResolverLimitsV1(
                max_skill_body_characters=10,
            ),
        )


def test_skill_discovery_rejects_symlink_escape(tmp_path: Path) -> None:
    """A resolved SKILL.md outside its explicit root fails containment."""

    root = tmp_path / "root"
    bundle = root / "escaped-skill"
    bundle.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text(
        "---\nname: escaped-skill\n"
        "description: Escaped instructions.\n---\nBody",
        encoding="utf-8",
    )
    try:
        (bundle / "SKILL.md").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"filesystem cannot create test symlink: {exc}")

    with pytest.raises(AgenticResolverContractError, match="escapes"):
        discover_skills([root])


def test_skill_frontmatter_uses_safe_yaml_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery invokes yaml.safe_load and no custom constructor surface."""

    _write_skill(tmp_path, "sample-skill")
    original_safe_load = yaml.safe_load
    calls: list[str] = []

    def _recording_safe_load(value: str):
        calls.append(value)
        parsed = original_safe_load(value)
        return parsed

    monkeypatch.setattr(yaml, "safe_load", _recording_safe_load)

    catalog = discover_skills([tmp_path])

    assert catalog.definitions[0].name == "sample-skill"
    assert len(calls) == 1


def test_yaml_frontmatter_dependency_is_declared() -> None:
    """Packaging explicitly declares the safe YAML parser dependency."""

    pyproject_text = (REPOSITORY_ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert '"PyYAML>=6.0"' in pyproject_text
    assert importlib.metadata.version("PyYAML")
