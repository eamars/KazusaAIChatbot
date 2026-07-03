"""Data shapes for trace-backed past-dialog cognition residual."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict


PastDialogCognitionSource = Literal["reply_context", "conversation_evidence"]


@dataclass(frozen=True)
class PastDialogCognitionCandidate:
    """One already-attached Kazusa-authored dialog eligible for residual lookup.

    Args:
        visible_text: Prompt-visible text from the attached past dialog.
        llm_trace_id: Trace run id stored on the assistant conversation row.
        created_at: Conversation row creation timestamp or equivalent ordering
            value.
        source: Structural attachment path that supplied this row.
        role: Conversation author role from storage.
        global_user_id: Internal author id for the stored row.
        conversation_row_id: Conversation-history row id or stable row ref.
        platform_message_id: Platform-local message id from the stored row.
        platform: Platform key for diagnostics and source validation.
        platform_channel_id: Channel/group/private scope of the stored row.
    """

    visible_text: str
    llm_trace_id: str
    created_at: object
    source: PastDialogCognitionSource
    role: str
    global_user_id: str
    conversation_row_id: str
    platform_message_id: str
    platform: str
    platform_channel_id: str


class PastDialogCognitionLookupResult(TypedDict):
    """Result returned by the residual context builder."""

    past_dialog_cognition_context: str
    candidate_count: int
    selected_count: int
    status: str
    diagnostics: list[dict[str, str]]
