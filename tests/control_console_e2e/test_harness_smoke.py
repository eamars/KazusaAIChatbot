from __future__ import annotations


def test_e2e_harness_starts_console_and_writes_summary(
    e2e_console,
    e2e_artifact_dir,
    e2e_summary_writer,
) -> None:
    """Harness smoke test for isolated console startup and artifact summaries."""

    with e2e_console() as console:
        session = console.request_json("GET", "/api/auth/session")

    summary = e2e_summary_writer(
        name="harness_smoke",
        conclusion="pass",
        details={
            "console_url": console.base_url,
            "session": session,
        },
    )

    assert session == {"authenticated": False}
    assert summary.exists()
    assert summary.parent == e2e_artifact_dir
