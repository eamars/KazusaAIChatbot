"""Public event logging interface for runtime callers."""

from __future__ import annotations

from kazusa_ai_chatbot.event_logging.models import (
    EventLogWriteResult,
    EventRefRecord,
    EventScopeInput,
    EventSeverity,
    SelfCognitionBudget,
)
from kazusa_ai_chatbot.event_logging.recording import (
    EVENT_LOG_WRITE_TIMEOUT_SECONDS,
    record_database_operation_event,
    record_cognition_v2_event,
    record_dialog_quality_event,
    record_dispatcher_event,
    record_llm_stage_event,
    record_model_contract_event,
    record_pipeline_turn_event,
    record_process_event,
    record_queue_intake_event,
    record_rag_stage_event,
    record_resource_health_event,
    record_runtime_error_event,
    record_self_cognition_event,
    record_worker_event,
)
from kazusa_ai_chatbot.event_logging.snapshots import write_analysis_snapshot
from kazusa_ai_chatbot.event_logging.status import (
    build_reflection_stats,
    build_runtime_status,
    build_self_cognition_stats,
)

__all__ = [
    "EVENT_LOG_WRITE_TIMEOUT_SECONDS",
    "EventLogWriteResult",
    "EventRefRecord",
    "EventScopeInput",
    "EventSeverity",
    "SelfCognitionBudget",
    "build_reflection_stats",
    "build_runtime_status",
    "build_self_cognition_stats",
    "record_database_operation_event",
    "record_cognition_v2_event",
    "record_dialog_quality_event",
    "record_dispatcher_event",
    "record_llm_stage_event",
    "record_model_contract_event",
    "record_pipeline_turn_event",
    "record_process_event",
    "record_queue_intake_event",
    "record_rag_stage_event",
    "record_resource_health_event",
    "record_runtime_error_event",
    "record_self_cognition_event",
    "record_worker_event",
    "write_analysis_snapshot",
]
