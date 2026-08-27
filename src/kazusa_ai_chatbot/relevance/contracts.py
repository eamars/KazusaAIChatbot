"""Canonical contracts shared by the relevance producers and coordinator."""

from __future__ import annotations

from typing import Literal, TypedDict

from kazusa_ai_chatbot.action_spec.results import EpisodeAttemptDiagnosticV1


class FrontlineDecision(TypedDict):
    """Validated semantic action returned by the frontline route."""

    intake_action: Literal["discard", "start", "append"]
    append_target: Literal["none", "open_1", "open_2", "open_3"]
    prelude_targets: list[Literal["prelude_1", "prelude_2"]]
    reason: str


class SettledRelevanceDecision(TypedDict):
    """Validated persona-aware response action."""

    response_action: Literal["ignore", "proceed", "wait"]
    reason_to_respond: str
    use_reply_feature: bool
    channel_topic: str
    indirect_speech_context: str


class RelevanceEvaluationEnvelope(TypedDict):
    """Validated relevance decision plus bounded attempt metadata."""

    decision: FrontlineDecision | SettledRelevanceDecision
    attempt_diagnostics: list[EpisodeAttemptDiagnosticV1]
