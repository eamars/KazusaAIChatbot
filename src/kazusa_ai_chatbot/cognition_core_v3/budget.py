"""Deterministic context estimation and total request-window ledger for V3."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

NORMAL_TOTAL_CEILING_TOKENS = 50_000
EXTENDED_TOTAL_CEILING_TOKENS = 65_000
CALIBRATION_MULTIPLIER = 1.00
MINIMUM_SERVING_WINDOW_TOKENS = 50_000


class CognitionContextLimitError(RuntimeError):
    """The next request cannot fit the active serving/ceiling contract."""


def _in_cjk_range(codepoint: int) -> bool:
    """Return whether one Unicode codepoint is Han or CJK punctuation."""

    return (
        0x2E80 <= codepoint <= 0x2EFF
        or 0x2F00 <= codepoint <= 0x2FDF
        or 0x3000 <= codepoint <= 0x303F
        or 0x3040 <= codepoint <= 0x30FF
        or 0x31F0 <= codepoint <= 0x31FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xAC00 <= codepoint <= 0xD7AF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0xFF00 <= codepoint <= 0xFFEF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def cjk_codepoint_count(text: str) -> int:
    """Count Han, Hiragana, Katakana, Hangul, and CJK punctuation points."""

    return sum(1 for character in text if _in_cjk_range(ord(character)))


def estimate_message_tokens(
    messages: Sequence[str],
    *,
    calibration_multiplier: float = CALIBRATION_MULTIPLIER,
) -> int:
    """Estimate prompt tokens using the fixed CJK-aware deterministic formula."""

    if not messages:
        raise ValueError("token estimation requires at least one message")
    if calibration_multiplier < 1.0:
        raise ValueError("calibration multiplier must be at least 1.0")

    cjk_count = sum(cjk_codepoint_count(message) for message in messages)
    utf8_bytes = sum(len(message.encode("utf-8")) for message in messages)
    cjk_utf8_bytes = sum(
        len(character.encode("utf-8"))
        for message in messages
        for character in message
        if _in_cjk_range(ord(character))
    )
    non_cjk_bytes = max(0, utf8_bytes - cjk_utf8_bytes)
    base_units = (
        cjk_count
        + ceil(non_cjk_bytes / 4)
        + 16 * len(messages)
        + 32
    )
    estimate = ceil(base_units * calibration_multiplier)
    return estimate


@dataclass(frozen=True)
class ContextBudgetPlan:
    """Frozen serving-window and total-ceiling declarations."""

    serving_window_tokens: int
    normal_total_ceiling_tokens: int = NORMAL_TOTAL_CEILING_TOKENS
    extended_total_ceiling_tokens: int = EXTENDED_TOTAL_CEILING_TOKENS

    def __post_init__(self) -> None:
        """Fail construction on an unusable serving window or ceiling."""

        if self.serving_window_tokens < MINIMUM_SERVING_WINDOW_TOKENS:
            raise ValueError(
                "serving window must be at least "
                f"{MINIMUM_SERVING_WINDOW_TOKENS} tokens"
            )
        if self.normal_total_ceiling_tokens != NORMAL_TOTAL_CEILING_TOKENS:
            raise ValueError("normal total ceiling is fixed at 50000 tokens")
        if self.extended_total_ceiling_tokens != EXTENDED_TOTAL_CEILING_TOKENS:
            raise ValueError("extended total ceiling is fixed at 65000 tokens")


@dataclass(frozen=True)
class BudgetAdmission:
    """Accepted per-step reservation facts."""

    estimated_prompt_tokens: int
    reserved_completion_tokens: int
    estimated_total_context_tokens: int
    active_total_ceiling_tokens: int
    extension_available: bool
    extension_used: bool


@dataclass
class ContextBudgetLedger:
    """Invocation-wide total request-window ceiling and re-anchor token."""

    plan: ContextBudgetPlan
    active_total_ceiling_tokens: int = NORMAL_TOTAL_CEILING_TOKENS
    extension_used: bool = False
    reanchor_used: bool = False

    @property
    def extension_available(self) -> bool:
        """Return whether the serving window permits the extended ceiling."""

        return (
            self.plan.serving_window_tokens
            >= self.plan.extended_total_ceiling_tokens
        )

    def admit(
        self,
        *,
        estimated_prompt_tokens: int,
        reserved_completion_tokens: int,
    ) -> BudgetAdmission:
        """Reserve one step and choose the normal or one-time extended tier."""

        if not isinstance(estimated_prompt_tokens, int) or isinstance(
            estimated_prompt_tokens,
            bool,
        ):
            raise TypeError("estimated_prompt_tokens must be an integer")
        if not isinstance(reserved_completion_tokens, int) or isinstance(
            reserved_completion_tokens,
            bool,
        ):
            raise TypeError("reserved_completion_tokens must be an integer")
        if estimated_prompt_tokens <= 0 or reserved_completion_tokens <= 0:
            raise ValueError("budget reservation values must be positive")

        estimated_total = estimated_prompt_tokens + reserved_completion_tokens
        if estimated_total > self.plan.serving_window_tokens:
            raise CognitionContextLimitError(
                "request exceeds the declared serving window"
            )

        if estimated_total <= self.active_total_ceiling_tokens:
            admission = BudgetAdmission(
                estimated_prompt_tokens=estimated_prompt_tokens,
                reserved_completion_tokens=reserved_completion_tokens,
                estimated_total_context_tokens=estimated_total,
                active_total_ceiling_tokens=self.active_total_ceiling_tokens,
                extension_available=self.extension_available,
                extension_used=self.extension_used,
            )
            return admission

        if (
            self.extension_available
            and not self.extension_used
            and estimated_total
            <= self.plan.extended_total_ceiling_tokens
        ):
            self.extension_used = True
            self.active_total_ceiling_tokens = (
                self.plan.extended_total_ceiling_tokens
            )
            return BudgetAdmission(
                estimated_prompt_tokens=estimated_prompt_tokens,
                reserved_completion_tokens=reserved_completion_tokens,
                estimated_total_context_tokens=estimated_total,
                active_total_ceiling_tokens=self.active_total_ceiling_tokens,
                extension_available=True,
                extension_used=True,
            )

        raise CognitionContextLimitError(
            "request exceeds the active total context ceiling"
        )

    def consume_reanchor(self) -> None:
        """Consume the one shared re-anchor token."""

        if self.reanchor_used:
            raise CognitionContextLimitError("re-anchor token already consumed")
        self.reanchor_used = True


__all__ = [
    "CALIBRATION_MULTIPLIER",
    "EXTENDED_TOTAL_CEILING_TOKENS",
    "MINIMUM_SERVING_WINDOW_TOKENS",
    "NORMAL_TOTAL_CEILING_TOKENS",
    "BudgetAdmission",
    "CognitionContextLimitError",
    "ContextBudgetLedger",
    "ContextBudgetPlan",
    "cjk_codepoint_count",
    "estimate_message_tokens",
]
