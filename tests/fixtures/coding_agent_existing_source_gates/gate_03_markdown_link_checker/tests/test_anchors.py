from mdlinkcheck.anchors import collect_anchors, slugify_heading


def test_slugify_heading() -> None:
    slug = slugify_heading("Install Guide!")

    assert slug == "install-guide"


def test_collect_anchors_counts_duplicates() -> None:
    anchors = collect_anchors("# Install\n## Install\n")

    assert anchors["install"] == 2
