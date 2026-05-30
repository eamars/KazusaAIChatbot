"""Validation command for the code repair fixture."""

from __future__ import annotations

import json

from message_window import collapse_followups


FOLLOWUP_WINDOW_SECONDS = 90


def main() -> None:
    """Validate that follow-up collapsing respects author and time boundaries."""

    messages = [
        {
            "author_id": "user-a",
            "body": "first report",
            "created_at": "2026-05-29T08:00:00Z",
        },
        {
            "author_id": "user-a",
            "body": "extra detail",
            "created_at": "2026-05-29T08:00:45Z",
        },
        {
            "author_id": "user-a",
            "body": "new topic after the cutoff",
            "created_at": "2026-05-29T08:04:10Z",
        },
        {
            "author_id": "user-b",
            "body": "separate author reply",
            "created_at": "2026-05-29T08:04:20Z",
        },
        {
            "author_id": "user-a",
            "body": "return after another speaker",
            "created_at": "2026-05-29T08:04:40Z",
        },
    ]
    expected = [
        {
            "author_id": "user-a",
            "messages": ["first report", "extra detail"],
        },
        {
            "author_id": "user-a",
            "messages": ["new topic after the cutoff"],
        },
        {
            "author_id": "user-b",
            "messages": ["separate author reply"],
        },
        {
            "author_id": "user-a",
            "messages": ["return after another speaker"],
        },
    ]

    actual = collapse_followups(
        messages,
        window_seconds=FOLLOWUP_WINDOW_SECONDS,
    )
    if actual != expected:
        payload = {
            "expected": expected,
            "actual": actual,
        }
        rendered_payload = json.dumps(payload, indent=2, sort_keys=True)
        raise AssertionError(
            "collapse_followups merged messages across a required boundary:\n"
            f"{rendered_payload}"
        )

    print("code_repair fixture check passed")


if __name__ == "__main__":
    main()
