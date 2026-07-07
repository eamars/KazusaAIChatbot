"""Standalone coding-agent package."""

from kazusa_ai_chatbot.coding_agent.models import (
    CodeEvidenceReference,
    CodingAgentBackgroundRequest,
    CodingAgentBackgroundResponse,
    CodingAgentWriteRequest,
    CodingPatchProposalResponse,
    CodingAgentRepositorySummary,
    CodingAgentRequest,
    CodingAgentResponse,
    InlineSourceInput,
)
from kazusa_ai_chatbot.coding_agent.code_patching.apply import (
    apply_approved_patch,
)
from kazusa_ai_chatbot.coding_agent.code_patching.models import (
    ApplyWorkspaceRef,
    CodingPatchApplyRequest,
    CodingPatchApplyResponse,
    PatchApplyApproval,
    PatchApplyValidation,
    PatchSourceIdentity,
)
from kazusa_ai_chatbot.coding_agent.supervisor import (
    answer_code_question,
    handle_background_coding_task,
    propose_code_change,
)

__all__ = [
    "CodeEvidenceReference",
    "CodingAgentBackgroundRequest",
    "CodingAgentBackgroundResponse",
    "CodingAgentWriteRequest",
    "CodingPatchProposalResponse",
    "CodingAgentRepositorySummary",
    "CodingAgentRequest",
    "CodingAgentResponse",
    "ApplyWorkspaceRef",
    "CodingPatchApplyRequest",
    "CodingPatchApplyResponse",
    "PatchApplyApproval",
    "PatchApplyValidation",
    "PatchSourceIdentity",
    "InlineSourceInput",
    "answer_code_question",
    "apply_approved_patch",
    "handle_background_coding_task",
    "propose_code_change",
]
