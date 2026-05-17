"""Documentation contract tests for self-cognition delivery architecture."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL_README = _ROOT / "src/kazusa_ai_chatbot/self_cognition/README.md"
_COGNITION_CONTRACTS_DOC = (
    _ROOT / "development_plans/reference/designs/cognition_contracts_design.md"
)
_PROACTIVE_OUTPUT_README = (
    _ROOT / "src/kazusa_ai_chatbot/proactive_output/README.md"
)
_LEGACY_DOCS = [
    _ROOT / "development_plans/reference/designs/self_cognition_tracking_icd.md",
    _ROOT / "development_plans/reference/designs/self_cognition_reasoning_basis.md",
    _ROOT / "development_plans/reference/designs/self_cognition_loop_architecture.md",
]


def _read_text(path: Path) -> str:
    """Read one repository text file as UTF-8."""

    content = path.read_text(encoding="utf-8")
    return content


def test_legacy_private_candidate_docs_have_superseded_banner() -> None:
    """Legacy private-candidate docs should state supersession up front."""

    for path in _LEGACY_DOCS:
        first_lines = "\n".join(_read_text(path).splitlines()[:20])
        assert "Superseded Architecture Document" in first_lines
        assert "Status: superseded" in first_lines
        assert (
            "Superseded by plan: "
            "development_plans/active/bugfix/"
            "self_cognition_speak_delivery_bugfix_plan.md"
        ) in first_lines
        assert (
            "Canonical current doc: "
            "src/kazusa_ai_chatbot/self_cognition/README.md"
        ) in first_lines


def test_canonical_self_cognition_readme_defines_delivery_target_before_cognition(
) -> None:
    """Canonical self-cognition docs should state pre-cognition target binding."""

    content = _read_text(_CANONICAL_README)

    assert "SelfCognitionDeliveryTarget" in content
    assert "before cognition" in content
    assert "known private channel" in content
    assert "self-cognition source channel" in content


def test_canonical_docs_do_not_authorize_production_not_requested_for_speak(
) -> None:
    """Current docs must not describe selected production speak as no-dispatch."""

    content = _read_text(_CANONICAL_README)
    lower_content = content.lower()

    assert "selected `speak`" in content
    assert "must attempt delivery" in lower_content
    assert "dispatch_status=not_requested" not in content
    assert "production_handoff: false" not in content
    assert "private tracking artifacts" not in lower_content


def test_cognition_contracts_doc_names_selected_self_cognition_speak_delivery(
) -> None:
    """Shared cognition contracts should include self-cognition speak delivery."""

    content = _read_text(_COGNITION_CONTRACTS_DOC)

    assert "selected self-cognition `speak`" in content
    assert "same shared cognition/dialog/persistence path" in content
    assert "runtime adapter bridge" in content


def test_proactive_output_doc_does_not_govern_self_cognition_speak() -> None:
    """Proactive-output docs should not shadow self-cognition delivery rules."""

    content = _read_text(_PROACTIVE_OUTPUT_README)

    assert "does not govern selected self-cognition speech" in content
    assert "selected self-cognition `speak`" in content
    assert "runtime adapter bridge" in content
