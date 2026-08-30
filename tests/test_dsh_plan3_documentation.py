"""Tests-first gates for the Plan 3 documentation cutover."""

from __future__ import annotations

from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _docs() -> str:
    """Return the canonical operator documentation bundle."""

    paths = (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "README_CN.md",
        REPOSITORY_ROOT / "docs" / "HOWTO.md",
        REPOSITORY_ROOT / "docs" / "SUBAGENT_INTERFACES.md",
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.fail(f"planned documentation paths are unavailable: {missing}")
    return "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)


def test_plan3_docs_describe_dsh_only_task_execution_and_retained_rag3() -> None:
    """Docs must describe DSH task ownership and the retained RAG3 path."""

    docs = _docs()

    assert "dsh" in docs
    assert "task" in docs
    assert "rag3" in docs or "local_context" in docs
    assert "complex_task_resolver" not in docs


def test_plan3_docs_name_exact_fourteen_tool_catalog_and_public_media_boundary() -> None:
    """Docs must pin the fourteen-tool and public-media boundaries."""

    docs = _docs()

    assert "fourteen" in docs or "14" in docs
    assert "kazusa_inspect_public_media" in docs
    assert "http(s)" in docs or "public media" in docs


def test_plan3_docs_contain_no_legacy_executor_interfaces() -> None:
    """No authoritative documentation may advertise retired executors."""

    docs = _docs()

    forbidden = (
        "continue_bound_coding_run",
        "background_work_llm_",
        "complex_task_resolver",
    )
    present = [term for term in forbidden if term in docs]

    assert not present, f"legacy documentation interfaces remain: {present}"


def test_plan3_architecture_names_exact_v2_epochs_and_binding_flow() -> None:
    """The architecture docs must name the versioned binding flow."""

    docs = _docs()

    for term in ("v2", "epoch", "binding", "generation", "checkpoint"):
        assert term in docs


def _dsh_icd_sections() -> tuple[str, ...]:
    """Return the three bounded DSH ICD sections under this amendment."""

    section_specs = (
        (
            REPOSITORY_ROOT / "src" / "kazusa_ai_chatbot" / "dsh_interaction"
            / "README.md",
            "# brain–dsh interaction boundary",
            None,
        ),
        (
            REPOSITORY_ROOT / "src" / "kazusa_ai_chatbot" / "brain_service"
            / "README.md",
            "## dsh interaction, task edge, and runtime readiness",
            "### live chat intake and settlement",
        ),
        (
            REPOSITORY_ROOT / "src" / "kazusa_ai_chatbot" / "cognition_core_v3"
            / "README.md",
            "## dsh interaction judgment",
            "## state and affect ownership",
        ),
    )
    sections: list[str] = []
    for path, start, end in section_specs:
        if not path.exists():
            pytest.fail(f"DSH ICD is unavailable: {path}")
        document = path.read_text(encoding="utf-8").lower()
        if start not in document:
            pytest.fail(f"DSH ICD section is unavailable: {path}")
        section = document.split(start, 1)[1]
        if end is not None:
            if end not in section:
                pytest.fail(f"DSH ICD section boundary is unavailable: {path}")
            section = section.split(end, 1)[0]
        sections.append(section)
    return tuple(sections)


def test_character_owned_dsh_icds_exclude_user_relay_contract() -> None:
    """The three DSH ICDs describe only the character-owned V2 boundary."""

    dsh_icds = _dsh_icd_sections()
    docs = "\n".join(dsh_icds)

    required = (
        "dsh_brain_interaction.v2",
        "character-owned",
        "full reusable cognition loop",
        "complete bounded semantic context",
        "self_goal_resolution",
        "waiting dsh hook",
        "audit",
        "replay",
        "nonce",
        "one-shot grant",
        "no dialog",
        "l3",
        "adapter",
        "no user prompt/reply",
        "waiting-state lifecycle",
    )
    for term in required:
        assert term in docs, f"DSH ICDs omit required contract term: {term}"
    assert "no dialog, l3, or adapter" in docs

    assert "post /runtime/dsh/interactions`" in dsh_icds[0]
    assert "post /runtime/dsh/interactions`" in dsh_icds[1]
    assert "sole authenticated" in dsh_icds[1]
    assert "`question`: `answer` or `reject`" in docs
    assert "`approval`: `allow_once` or `reject`" in docs
    assert "`plan_review`: `answer`, `allow_once`, or `reject`" in docs

    retired_vocabulary = (
        "`relay_to_user`",
        "`continue_waiting`",
        "/runtime/dsh/interactions/checkpoint",
        "pending interaction",
        "exact user reply",
        "visible question",
    )
    present = [term for term in retired_vocabulary if term in docs]
    assert not present, f"retired DSH user-relay vocabulary remains: {present}"

    assert "task_resolution_request" not in docs
    assert "human_clarification" not in docs
    assert "approval_preparation" not in docs
