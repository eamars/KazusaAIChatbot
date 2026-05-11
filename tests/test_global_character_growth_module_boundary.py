"""Static boundary checks for global character growth."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = _ROOT / "src" / "kazusa_ai_chatbot" / "global_character_growth"


def test_global_growth_package_does_not_touch_character_state_writers() -> None:
    """Growth traits must not write legacy character-state surfaces."""

    forbidden = (
        "upsert_" + "character_" + "self_image",
        "upsert_" + "character_" + "state",
        "save_" + "character_" + "profile",
        "character_state",
    )

    offenders = _find_forbidden_tokens(_PACKAGE, forbidden)

    assert offenders == []


def test_global_growth_package_does_not_read_style_surfaces() -> None:
    """User and group style images are separate scoped adaptation surfaces."""

    forbidden = (
        "user_style_image",
        "group_channel_style",
        "interaction_style_images",
        "user_memory_context",
    )

    offenders = _find_forbidden_tokens(_PACKAGE, forbidden)

    assert offenders == []


def test_raw_mongodb_calls_stay_out_of_growth_package() -> None:
    """Only the named DB interface may own raw Mongo operations."""

    forbidden = (
        "get_db(",
        ".insert_one(",
        ".update_one(",
        ".update_many(",
        ".delete_one(",
        ".delete_many(",
        ".replace_one(",
    )

    offenders = _find_forbidden_tokens(_PACKAGE, forbidden)

    assert offenders == []


def test_no_live_path_modules_import_global_character_growth() -> None:
    """Service, RAG, L3, dialog, and adapters should remain untouched."""

    checked_paths = [
        _ROOT / "src" / "kazusa_ai_chatbot" / "service.py",
        _ROOT / "src" / "kazusa_ai_chatbot" / "brain_service",
        _ROOT / "src" / "kazusa_ai_chatbot" / "rag",
        _ROOT / "src" / "kazusa_ai_chatbot" / "nodes" / "dialog_agent.py",
        _ROOT / "src" / "kazusa_ai_chatbot" / "nodes" / "persona_supervisor2_cognition_l3.py",
    ]

    offenders: list[str] = []
    for path in checked_paths:
        if path.is_file():
            candidates = [path]
        else:
            candidates = list(path.rglob("*.py"))
        for candidate in candidates:
            source = candidate.read_text(encoding="utf-8")
            if "global_character_growth" in source:
                offenders.append(str(candidate.relative_to(_ROOT)))

    assert offenders == []


def _find_forbidden_tokens(root: Path, forbidden: tuple[str, ...]) -> list[str]:
    """Return forbidden-token matches under an existing package root."""

    offenders: list[str] = []
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                offenders.append(f"{path.relative_to(_ROOT)}:{token}")
    return offenders
