"""Static ownership checks for character identity growth."""

from __future__ import annotations

from pathlib import Path
import re


_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = _ROOT / "src" / "kazusa_ai_chatbot" / "character_identity_growth"
_DB_OWNER = (
    _ROOT
    / "src"
    / "kazusa_ai_chatbot"
    / "db"
    / "character_identity_growth.py"
)
_BOOTSTRAP = (
    _ROOT
    / "src"
    / "kazusa_ai_chatbot"
    / "db"
    / "bootstrap.py"
)
_OPERATOR_LOADER = (
    _ROOT
    / "src"
    / "scripts"
    / "load_character_profile.py"
)


def test_domain_package_contains_no_raw_database_operations() -> None:
    """Only the dedicated DB module may access MongoDB."""

    forbidden = (
        "get_db(",
        "pymongo",
        "motor.",
        ".insert_one(",
        ".update_one(",
        ".replace_one(",
        ".delete_one(",
        ".aggregate(",
    )

    assert _find_tokens(forbidden) == []


def test_domain_package_contains_no_legacy_identity_authority() -> None:
    """The big-bang owner must not call retired growth or mutable self-image."""

    forbidden = (
        "global_" + "character_" + "growth",
        "upsert_" + "character_" + "self_image",
        "character_state.self_image",
        "promoted_global_growth",
    )

    assert _find_tokens(forbidden) == []


def test_domain_package_contains_no_named_character_literals() -> None:
    """Production growth behavior must remain character-generic."""

    named_character = re.compile(
        r"\b(?:asuna|kazusa|qingche)\b",
        flags=re.IGNORECASE,
    )
    offenders: list[str] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        if named_character.search(source):
            offenders.append(str(path.relative_to(_ROOT)))

    assert offenders == []


def test_persistence_owner_uses_exactly_three_growth_collections() -> None:
    """Persistence must retain the plan's three-collection boundary."""

    source = _DB_OWNER.read_text(encoding="utf-8")

    assert (
        source.count('"character_identity_revisions"')
        == 1
    )
    assert (
        source.count('"character_identity_growth_candidates"')
        == 1
    )
    assert (
        source.count('"character_identity_growth_runs"')
        == 1
    )
    assert "character_identity_consumption" not in source
    assert "character_identity_health" not in source


def test_bootstrap_creates_identity_collections_without_legacy_growth() -> None:
    """Clean bootstrap must not create the retired soft-growth ledger."""

    source = _BOOTSTRAP.read_text(encoding="utf-8")
    legacy_fragments = (
        "GLOBAL_" + "CHARACTER_" + "GROWTH",
        "ensure_global_" + "character_" + "growth_indexes",
    )

    assert "GROWTH_COLLECTION_NAMES" in source
    assert "ensure_character_identity_growth_indexes" in source
    assert all(fragment not in source for fragment in legacy_fragments)


def test_operator_reset_boundary_is_revisioned_and_llm_free() -> None:
    """The force loader must require audit identity and call the ledger."""

    loader_source = _OPERATOR_LOADER.read_text(encoding="utf-8")
    owner_source = _DB_OWNER.read_text(encoding="utf-8")

    assert "--operator-action-id" in loader_source
    assert "create_operator_reset_revision" in loader_source
    assert "save_character_profile" not in loader_source
    assert "kazusa_ai_chatbot.llm" not in owner_source
    assert ".ainvoke(" not in owner_source


def test_database_and_operator_growth_code_is_character_generic() -> None:
    """Persistence and operator mechanics may contain no named character."""

    named_character = re.compile(
        r"\b(?:asuna|kazusa|qingche)\b",
        flags=re.IGNORECASE,
    )
    offenders = [
        str(path.relative_to(_ROOT))
        for path in (_DB_OWNER, _OPERATOR_LOADER)
        if named_character.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def _python_files() -> list[Path]:
    """Return Python files beneath the required domain package."""

    assert _PACKAGE.is_dir()
    return sorted(_PACKAGE.glob("*.py"))


def _find_tokens(tokens: tuple[str, ...]) -> list[str]:
    """Return forbidden-token matches from the domain package."""

    offenders: list[str] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        for token in tokens:
            if token in source:
                offenders.append(f"{path.relative_to(_ROOT)}:{token}")
    return offenders
