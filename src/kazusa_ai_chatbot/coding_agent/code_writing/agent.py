"""Public orchestration for the code-writing subagent."""

from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    CodeWritingRequest,
    CodeWritingResult,
)
from kazusa_ai_chatbot.coding_agent.code_writing.supervisor import (
    run_writing_supervisor,
)


async def run(request: CodeWritingRequest) -> CodeWritingResult:
    """Produce a patch proposal through the writing supervisor."""

    trace: dict[str, object] = {}
    result = await run_writing_supervisor(request, trace=trace)
    if trace:
        result["trace"] = trace
    return result
