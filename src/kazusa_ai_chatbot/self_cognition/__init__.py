"""Self-cognition contracts, artifact builders, and worker entrypoints."""

from kazusa_ai_chatbot.self_cognition.artifacts import write_tracking_artifacts
from kazusa_ai_chatbot.self_cognition.runner import (
    build_self_cognition_case_artifacts,
    build_self_cognition_case_artifacts_async,
    run_self_cognition_case,
)
from kazusa_ai_chatbot.self_cognition.tracking import (
    build_action_attempt,
    build_action_candidate,
    build_idempotency_key,
    build_route_effect,
    build_run_record,
    build_trigger_record,
    classify_route,
)
from kazusa_ai_chatbot.self_cognition.worker import (
    SelfCognitionWorkerHandle,
    SelfCognitionWorkerResult,
    run_self_cognition_worker_tick,
    start_self_cognition_worker,
    stop_self_cognition_worker,
)

__all__ = [
    "SelfCognitionWorkerHandle",
    "SelfCognitionWorkerResult",
    "build_self_cognition_case_artifacts",
    "build_self_cognition_case_artifacts_async",
    "build_action_attempt",
    "build_action_candidate",
    "build_idempotency_key",
    "build_route_effect",
    "build_run_record",
    "build_trigger_record",
    "classify_route",
    "run_self_cognition_case",
    "run_self_cognition_worker_tick",
    "start_self_cognition_worker",
    "stop_self_cognition_worker",
    "write_tracking_artifacts",
]
