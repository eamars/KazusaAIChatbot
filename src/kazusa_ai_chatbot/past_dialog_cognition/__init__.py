"""Past-dialog cognition residual projection for L2a-only context."""

from kazusa_ai_chatbot.past_dialog_cognition.models import (
    PastDialogCognitionCandidate,
    PastDialogCognitionLookupResult,
)
from kazusa_ai_chatbot.past_dialog_cognition.runtime import (
    build_past_dialog_cognition_context,
    build_past_dialog_cognition_context_from_rag_result,
    candidate_from_conversation_row,
    candidates_from_conversation_rows,
    conversation_row_ids_from_rag_result,
)

__all__ = [
    "PastDialogCognitionCandidate",
    "PastDialogCognitionLookupResult",
    "build_past_dialog_cognition_context",
    "build_past_dialog_cognition_context_from_rag_result",
    "candidate_from_conversation_row",
    "candidates_from_conversation_rows",
    "conversation_row_ids_from_rag_result",
]
