from kazusa_ai_chatbot.brain_service.delivery_mentions import (
    build_inline_delivery_mentions,
)


def test_build_inline_delivery_mentions_matches_multiple_users() -> None:
    """Build one render candidate for each exact authored tag."""

    candidates = build_inline_delivery_mentions(
        text="@Alex @Moca check this",
        users=[
            {
                "display_name": "Alex",
                "platform_user_id": "1001",
                "global_user_id": "global-alex",
            },
            {
                "display_name": "Moca",
                "platform_user_id": "1002",
                "global_user_id": "global-moca",
            },
        ],
        character_global_user_id="character",
    )

    assert candidates == [
        {
            "entity_kind": "user",
            "display_name": "Alex",
            "platform_user_id": "1001",
        },
        {
            "entity_kind": "user",
            "display_name": "Moca",
            "platform_user_id": "1002",
        },
    ]


def test_build_inline_delivery_mentions_requires_authored_token() -> None:
    """Known users are not disclosed when the outbound text lacks their tag."""

    candidates = build_inline_delivery_mentions(
        text="Alex should check this",
        users=[
            {
                "display_name": "Alex",
                "platform_user_id": "1001",
                "global_user_id": "global-alex",
            },
        ],
        character_global_user_id="character",
    )

    assert candidates == []


def test_build_inline_delivery_mentions_omits_ambiguous_display_names() -> None:
    """Duplicate visible names fail closed instead of choosing an id."""

    candidates = build_inline_delivery_mentions(
        text="@Alex check this",
        users=[
            {
                "display_name": "Alex",
                "platform_user_id": "1001",
                "global_user_id": "global-alex-1",
            },
            {
                "display_name": "Alex",
                "platform_user_id": "1002",
                "global_user_id": "global-alex-2",
            },
        ],
        character_global_user_id="character",
    )

    assert candidates == []


def test_build_inline_delivery_mentions_omits_unrenderable_and_character() -> None:
    """Candidates need platform ids and must not target the active character."""

    candidates = build_inline_delivery_mentions(
        text="@Alex @Kazusa check this",
        users=[
            {
                "display_name": "Alex",
                "platform_user_id": "",
                "global_user_id": "global-alex",
            },
            {
                "display_name": "Kazusa",
                "platform_user_id": "bot-id",
                "global_user_id": "character",
            },
        ],
        character_global_user_id="character",
    )

    assert candidates == []


def test_build_inline_delivery_mentions_ignores_fenced_code_only_tags() -> None:
    """Tags inside fenced code blocks are not render candidates."""

    candidates = build_inline_delivery_mentions(
        text="```text\n@Alex\n```\nplain text",
        users=[
            {
                "display_name": "Alex",
                "platform_user_id": "1001",
                "global_user_id": "global-alex",
            },
        ],
        character_global_user_id="character",
    )

    assert candidates == []


def test_build_inline_delivery_mentions_rejects_prefix_name_match() -> None:
    """A shorter name must not match inside a longer mention token."""

    candidates = build_inline_delivery_mentions(
        text="@Alexandra check this",
        users=[
            {
                "display_name": "Alex",
                "platform_user_id": "1001",
                "global_user_id": "global-alex",
            },
        ],
        character_global_user_id="character",
    )

    assert candidates == []


def test_build_inline_delivery_mentions_rejects_embedded_token() -> None:
    """A tag must not match when embedded in a larger word-like token."""

    candidates = build_inline_delivery_mentions(
        text="email@Alex check this",
        users=[
            {
                "display_name": "Alex",
                "platform_user_id": "1001",
                "global_user_id": "global-alex",
            },
        ],
        character_global_user_id="character",
    )

    assert candidates == []
