from slug_tools.slug import slugify


def test_slugify_collapses_punctuation_and_repeated_separators() -> None:
    assert slugify(" Hello,   World!! ") == "hello-world"


def test_slugify_keeps_numbers() -> None:
    assert slugify("Release 2026.07") == "release-202607"
