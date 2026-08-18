"""Cache-affine semantic-chain cognition engine over the V2-shaped substrate.

The package exposes one public entrypoint, ``run_cognition``, with the exact
``CognitionCoreInputV2``/``CognitionCoreOutputV2`` contract of the selected
engine family. The deterministic orchestrator owns chain selection, stage
order, visibility, checkpoints, attempt caps, validation, and failure
disposition; each semantic owner runs a bounded cache-affine transcript under
its own static system prompt.

The protected replay capture API (failure-only) is exposed alongside it so
replay harnesses can read the exact stage attempts recorded during one run.
"""

from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
    BOUNDARY_REJECTED_ERROR_CODE,
    EXHAUSTION_FAILURE_CLASS,
    PROVIDER_FAILURE_CLASS,
    StageFailure,
    StageResult,
    STRUCTURAL_FAILURE_CLASS,
)
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    clear_protected_chain_records,
    protected_chain_records,
    run_cognition,
)

__all__ = [
    "APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE",
    "BOUNDARY_REJECTED_ERROR_CODE",
    "EXHAUSTION_FAILURE_CLASS",
    "PROVIDER_FAILURE_CLASS",
    "StageFailure",
    "StageResult",
    "STRUCTURAL_FAILURE_CLASS",
    "clear_protected_chain_records",
    "protected_chain_records",
    "run_cognition",
]
