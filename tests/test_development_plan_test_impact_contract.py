"""Enforce the executable-plan test-impact contract."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    REPOSITORY_ROOT
    / "development_plans"
    / "archive"
    / "completed"
    / "short_term"
    / "development_plan_test_impact_traceability_and_cognition_unit_structure_bigbang_plan.md"
)
SKILL_PATH = REPOSITORY_ROOT / ".agents" / "skills" / "development-plan" / "SKILL.md"
PLAN_CONTRACT_PATH = (
    REPOSITORY_ROOT
    / ".agents"
    / "skills"
    / "development-plan"
    / "references"
    / "plan_contract.md"
)
EXECUTION_GATES_PATH = (
    REPOSITORY_ROOT
    / ".agents"
    / "skills"
    / "development-plan"
    / "references"
    / "execution_gates.md"
)


def _read(path: Path) -> str:
    """Read one repository contract as UTF-8 text."""

    text = path.read_text(encoding="utf-8")
    return text


def test_skill_requires_exact_test_impact_matrix() -> None:
    """The planning skill requires exact source-to-test ownership rows."""

    skill_text = _read(SKILL_PATH)

    assert "Test Impact And Traceability" in skill_text
    assert "exact deterministic pytest node IDs" in skill_text
    assert "passing broader suite" in skill_text


def test_plan_contract_requires_traceability_fields() -> None:
    """The plan reference defines every required impact-matrix field."""

    contract_text = _read(PLAN_CONTRACT_PATH)

    assert "## Test Impact And Traceability" in contract_text
    for field_name in (
        "repository-relative path",
        "changed symbol, field, interface, or contract",
        "semantic owner",
        "exact deterministic pytest node IDs",
        "observable regression prevented",
    ):
        assert field_name in contract_text


def test_execution_gates_require_changed_source_collection_check() -> None:
    """Execution gates require exact node collection before acceptance."""

    gates_text = _read(EXECUTION_GATES_PATH)

    assert "changed production source path" in gates_text
    assert "exact mapped node IDs" in gates_text
    assert "uncollected mapped node" in gates_text


def test_plan_contains_its_own_exact_impact_matrix() -> None:
    """The self-referential plan names the two missed regression nodes."""

    plan_text = _read(PLAN_PATH)

    assert "## Test Impact And Traceability" in plan_text
    assert (
        "tests/unit/cognition_core_v2/test_semantic_source_planner.py::"
        "test_moral_identity_questions_exclude_standard_handles"
    ) in plan_text
    assert (
        "tests/unit/cognition_resolver/test_contracts.py::"
        "test_current_turn_carrier_rejects_v1_or_incomplete_decision"
    ) in plan_text
