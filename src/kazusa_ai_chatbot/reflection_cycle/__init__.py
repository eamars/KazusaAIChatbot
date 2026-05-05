"""Character reflection-cycle public interfaces."""

from __future__ import annotations

from kazusa_ai_chatbot.reflection_cycle.models import (
    ChannelReflectionResult,
    ReflectionEvaluationResult,
    ReflectionInputSet,
    ReflectionPromotionResult,
    ReflectionScopeInput,
    ReflectionWorkerHandle,
    ReflectionWorkerResult,
)
from kazusa_ai_chatbot.reflection_cycle.runtime import (
    collect_reflection_inputs,
    run_readonly_reflection_evaluation,
)
from kazusa_ai_chatbot.reflection_cycle.context import (
    build_promoted_reflection_context,
)
from kazusa_ai_chatbot.reflection_cycle.promotion import (
    run_global_reflection_promotion,
)
from kazusa_ai_chatbot.reflection_cycle.worker import (
    run_daily_channel_reflection_cycle,
    run_hourly_reflection_cycle,
    start_reflection_cycle_worker,
    stop_reflection_cycle_worker,
)

__all__ = [
    "ChannelReflectionResult",
    "ReflectionEvaluationResult",
    "ReflectionInputSet",
    "ReflectionPromotionResult",
    "ReflectionScopeInput",
    "ReflectionWorkerHandle",
    "ReflectionWorkerResult",
    "build_promoted_reflection_context",
    "collect_reflection_inputs",
    "run_daily_channel_reflection_cycle",
    "run_global_reflection_promotion",
    "run_hourly_reflection_cycle",
    "run_readonly_reflection_evaluation",
    "start_reflection_cycle_worker",
    "stop_reflection_cycle_worker",
]
