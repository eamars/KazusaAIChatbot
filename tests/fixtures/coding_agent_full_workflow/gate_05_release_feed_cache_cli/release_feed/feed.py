"""Release feed rendering."""

from __future__ import annotations


def visible_releases(
    releases: list[dict[str, object]],
    *,
    include_drafts: bool = False,
) -> list[dict[str, object]]:
    """Return releases that should be visible in CLI output."""

    if include_drafts:
        return list(releases)
    visible = [
        release
        for release in releases
        if not release.get("draft")
    ]
    return visible


def render_titles(
    releases: list[dict[str, object]],
    *,
    include_drafts: bool = False,
) -> list[str]:
    """Render visible release titles."""

    visible = visible_releases(releases, include_drafts=include_drafts)
    titles = [str(release["title"]) for release in visible]
    return titles
