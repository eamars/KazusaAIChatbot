"""Prompt and payload helpers for text-only artifact generation."""

from __future__ import annotations

from typing import Any

BACKGROUND_ARTIFACT_WORKER_PROMPT = '''\
You are a bounded text artifact worker for a character brain service.

Create only the requested text artifact. Do not claim to execute code, inspect
files, mutate repositories, install packages, browse the web, download assets,
create images, handle attachments, or send the result to the user.

Allowed work kinds:
- coding_snippet: provide a small self-contained code snippet or explanation.
- text_rewrite: rewrite or polish the supplied text summary.
- summary: summarize the supplied text summary.

Return only legal JSON:
{
  "status": "succeeded | failed",
  "artifact_text": "the finished text artifact, capped by max_output_chars",
  "failure_summary": "short reason when failed, otherwise empty"
}
'''


def build_background_artifact_worker_payload(
    *,
    work_kind: str,
    objective: str,
    input_summary: str,
    max_output_chars: int,
) -> dict[str, Any]:
    """Build worker-visible semantic input without queue or delivery internals."""

    payload: dict[str, Any] = {
        "work_kind": work_kind,
        "objective": objective.strip(),
        "input_summary": input_summary.strip(),
        "max_output_chars": int(max_output_chars),
    }
    return payload
