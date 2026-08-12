"""Documentation contracts for the parent-checkpoint guardrail boundary."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    """Read one governed README as UTF-8 text."""

    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def test_resolver_readme_documents_canonical_checkpoint_owner() -> None:
    """Resolver docs identify the connector and bounded two-epoch owner."""

    content = _read("src/kazusa_ai_chatbot/cognition_resolver/README.md")

    assert "CognitionRetryCoordinator" in content
    assert "canonical\nconnector owns the checkpoint" in content
    assert "exactly two epochs" in content
    assert "idle `self_cognition` calls do not bind" in content


def test_core_readme_documents_two_epoch_owner_budget() -> None:
    """Core docs preserve three calls per owner in each guarded epoch."""

    content = _read("src/kazusa_ai_chatbot/cognition_core_v2/README.md")

    assert "three-call owner cap independently in epoch" in content
    assert "Epoch one remains active" in content
    assert "cognition_attempt_ledger.v2" in content


def test_brain_service_readme_documents_shared_replay_token() -> None:
    """Service docs identify shared retry arbitration and exact eligibility."""

    content = _read("src/kazusa_ai_chatbot/brain_service/README.md")

    assert "one context-local replay coordinator" in content
    assert "goal_bid_structure_exhausted" in content
    assert "goal_bid_provider_exhausted" in content
    assert "non-committing" in content


def test_nodes_readme_documents_connector_guard_owner() -> None:
    """Node docs keep preparation and commit outside parent replay."""

    content = _read("src/kazusa_ai_chatbot/nodes/README.md")

    assert "Parent-checkpoint guardrail boundary" in content
    assert "canonical `CognitionCoreInputV2`" in content
    assert "never enters the guardrail" in content
    assert "final validated output" in content


def test_llm_tracing_readme_documents_outer_capsule_lineage() -> None:
    """Tracing docs separate bounded outer lineage from inner attempts."""

    content = _read("src/kazusa_ai_chatbot/llm_tracing/README.md")

    assert "cognition_parent_guardrail_capsule.v1" in content
    assert "cognition_attempt_ledger.v2" in content
    assert "stores no checkpoint" in content
    assert "inner model attempts" in content
