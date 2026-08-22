"""Calibrate the Cognition V3 deterministic token estimator."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import cast

from kazusa_ai_chatbot.cognition_core_v3.budget import (
    cjk_codepoint_count,
)

DEFAULT_CORPUS_PATH = (
    Path("tests")
    / "fixtures"
    / "cognition_core_v3_token_calibration_corpus.json"
)
MINIMUM_MULTIPLIER = 1.00
MULTIPLIER_STEP = 0.05
MAXIMUM_HOLDOUT_MEDIAN_OVERESTIMATE = 0.35


@dataclass(frozen=True)
class CalibrationReport:
    """Deterministic calibration and holdout acceptance report."""

    calibration_multiplier: float
    calibration_underestimates: int
    holdout_underestimates: int
    holdout_median_overestimate: float
    accepted: bool


def _next_step_above(value: float) -> float:
    """Round a ratio up to the next 0.05 step, with a 1.00 floor."""

    if value <= MINIMUM_MULTIPLIER:
        return MINIMUM_MULTIPLIER
    step = MULTIPLIER_STEP
    multiplier = ceil((value - MINIMUM_MULTIPLIER) / step) * step + MINIMUM_MULTIPLIER
    return round(multiplier, 2)


def _base_units(messages: Sequence[dict[str, str]]) -> int:
    """Return the multiplier-free estimator base units."""

    texts = [message["content"] for message in messages]
    cjk_count = 0
    utf8_bytes = 0
    for text in texts:
        cjk_count += cjk_codepoint_count(text)
        utf8_bytes += len(text.encode("utf-8"))
    non_cjk_bytes = max(0, utf8_bytes - cjk_count)
    base = cjk_count + ceil(non_cjk_bytes / 4) + 16 * len(messages) + 32
    return base


def compute_calibration_report(
    calibration_payloads: Sequence[dict[str, object]],
    holdout_payloads: Sequence[dict[str, object]],
    observed_calibration_tokens: Sequence[int],
    observed_holdout_tokens: Sequence[int],
) -> CalibrationReport:
    """Compute the multiplier and validate the frozen holdout contract."""

    if len(calibration_payloads) != len(observed_calibration_tokens):
        raise ValueError("one observed calibration token count is required per payload")
    if len(holdout_payloads) != len(observed_holdout_tokens):
        raise ValueError("one observed holdout token count is required per payload")

    calibration_ratios = [
        observed_tokens / _base_units(payload["messages"])
        for payload, observed_tokens in zip(
            calibration_payloads,
            observed_calibration_tokens,
            strict=True,
        )
    ]
    multiplier = _next_step_above(max(calibration_ratios))

    holdout_overestimates = []
    underestimates = 0
    for payload, observed_tokens in zip(
        holdout_payloads,
        observed_holdout_tokens,
        strict=True,
    ):
        base = _base_units(payload["messages"])
        estimate = ceil(base * multiplier)
        if estimate < observed_tokens:
            underestimates += 1
        holdout_overestimates.append((estimate - observed_tokens) / observed_tokens)

    ordered = sorted(holdout_overestimates)
    if len(ordered) % 2 == 1:
        median = ordered[len(ordered) // 2]
    else:
        midpoint = len(ordered) // 2
        median = (ordered[midpoint - 1] + ordered[midpoint]) / 2

    report = CalibrationReport(
        calibration_multiplier=multiplier,
        calibration_underestimates=0,
        holdout_underestimates=underestimates,
        holdout_median_overestimate=median,
        accepted=(
            underestimates == 0
            and median <= MAXIMUM_HOLDOUT_MEDIAN_OVERESTIMATE
        ),
    )
    return report


def _load_corpus(path: Path) -> dict[str, object]:
    """Load the frozen calibration corpus."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if document["schema_version"] != "cognition_v3_token_calibration_corpus.v1":
        raise ValueError("unsupported calibration corpus schema")
    return document


def _build_parser() -> argparse.ArgumentParser:
    """Build the calibration CLI parser."""

    parser = argparse.ArgumentParser(
        description="Calibrate the Cognition V3 token estimator.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Frozen calibration corpus path.",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        help="JSON file containing calibration and holdout prompt-token counts.",
    )
    return parser


def main() -> int:
    """Run the calibration command."""

    parser = _build_parser()
    args = parser.parse_args()
    corpus = _load_corpus(args.corpus)
    observations = {
        "calibration": [],
        "holdout": [],
    }
    if args.observations is not None:
        raw = json.loads(args.observations.read_text(encoding="utf-8"))
        observations = {
            "calibration": raw["calibration_prompt_tokens"],
            "holdout": raw["holdout_prompt_tokens"],
        }
    else:
        raise SystemExit("--observations is required for a live calibration")

    report = compute_calibration_report(
        cast(
            Sequence[dict[str, object]],
            corpus["calibration_payloads"],
        ),
        cast(
            Sequence[dict[str, object]],
            corpus["holdout_payloads"],
        ),
        observations["calibration"],
        observations["holdout"],
    )
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    return 0 if report.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
