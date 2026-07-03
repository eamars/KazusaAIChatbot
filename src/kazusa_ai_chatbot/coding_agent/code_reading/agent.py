"""Public orchestration for the code-reading subagent."""

from kazusa_ai_chatbot.coding_agent.code_reading.models import (
    CodeReadingRequest,
    CodeReadingResult,
)
from kazusa_ai_chatbot.coding_agent.code_reading.supervisor import (
    run_reading_supervisor,
)


def run(request: CodeReadingRequest) -> CodeReadingResult:
    """Answer a source-code question through the reading supervisor."""

    result = run_reading_supervisor(request)
    return result
