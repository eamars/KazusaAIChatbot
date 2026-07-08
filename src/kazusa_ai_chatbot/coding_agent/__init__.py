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
from kazusa_ai_chatbot.coding_agent.code_executing import (
    CodeExecutionRequest,
    CodeExecutionResponse,
    CodeExecutionSpec,
    execute_code_check,
)
from kazusa_ai_chatbot.coding_agent.code_verifying import (
    CodingVerifyRepairRequest,
    CodingVerifyRepairResponse,
    ExecutionRepairFeedback,
    VerifyRepairAttempt,
    verify_and_repair_code_change,
)
from kazusa_ai_chatbot.coding_agent.coding_run import (
    CodingRunAttempt,
    CodingRunBlocker,
    CodingRunContinueRequest,
    CodingRunEvent,
    CodingRunGetRequest,
    CodingRunLedger,
    CodingRunResponse,
    CodingRunStartRequest,
    continue_coding_run,
    get_coding_run,
    start_coding_run,
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
    "CodeExecutionRequest",
    "CodeExecutionResponse",
    "CodeExecutionSpec",
    "CodingVerifyRepairRequest",
    "CodingVerifyRepairResponse",
    "ExecutionRepairFeedback",
    "VerifyRepairAttempt",
    "ApplyWorkspaceRef",
    "CodingPatchApplyRequest",
    "CodingPatchApplyResponse",
    "PatchApplyApproval",
    "PatchApplyValidation",
    "PatchSourceIdentity",
    "CodingRunAttempt",
    "CodingRunBlocker",
    "CodingRunContinueRequest",
    "CodingRunEvent",
    "CodingRunGetRequest",
    "CodingRunLedger",
    "CodingRunResponse",
    "CodingRunStartRequest",
    "InlineSourceInput",
    "answer_code_question",
    "apply_approved_patch",
    "continue_coding_run",
    "execute_code_check",
    "get_coding_run",
    "handle_background_coding_task",
    "propose_code_change",
    "start_coding_run",
    "verify_and_repair_code_change",
]
