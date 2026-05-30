"""Run goal resolver POC cases and write raw evaluation artifacts."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.goal_resolver_poc.artifacts import (
    write_casebook,
    write_json,
    write_run_artifact,
    write_summary,
)
from kazusa_ai_chatbot.goal_resolver_poc.casebook import GOAL_RESOLVER_CASES
from kazusa_ai_chatbot.goal_resolver_poc.models import (
    CASE_EVALUATION_ACCEPTED_STATUSES,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_REPAIR_PASSES,
)
from kazusa_ai_chatbot.goal_resolver_poc.runner import (
    run_goal_resolver_case_async,
    select_cases,
)
from kazusa_ai_chatbot.mcp_client import mcp_manager


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "test_artifacts" / "goal_resolver_poc"


def _configure_console() -> None:
    """Use UTF-8 console output for CJK model traces."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run goal resolver POC cases.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all casebook cases.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run one case id. Repeat for multiple cases.",
    )
    parser.add_argument(
        "--output-dir",
        default="test_artifacts/goal_resolver_poc",
        help="Directory for raw JSON artifacts.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Maximum resolver loop iterations per pass.",
    )
    parser.add_argument(
        "--repair-passes",
        type=int,
        default=DEFAULT_REPAIR_PASSES,
        help="Evaluator-driven repair passes after an initial failure.",
    )
    return parser


def _resolve_output_dir(path_text: str) -> Path:
    """Resolve an artifact path inside the goal resolver POC output root."""

    requested_path = Path(path_text)
    if not requested_path.is_absolute():
        requested_path = REPO_ROOT / requested_path

    resolved_path = requested_path.resolve()
    allowed_root = DEFAULT_OUTPUT_DIR.resolve()
    try:
        resolved_path.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            f"--output-dir must stay within {allowed_root}; got {resolved_path}"
        ) from exc

    return resolved_path


def _summary_payload(
    runs: list[dict[str, Any]],
    output_dir: Path,
    mcp_status: str,
) -> dict[str, Any]:
    """Build a compact raw summary for machine inspection."""

    rows = []
    accepted_count = 0
    for run in runs:
        evaluation = run["case_evaluation"]
        terminal_contract_valid = bool(run["terminal_contract_valid"])
        accepted = (
            evaluation["status"] in CASE_EVALUATION_ACCEPTED_STATUSES
            and terminal_contract_valid
        )
        if accepted:
            accepted_count += 1
        row = {
            "case_id": run["case_id"],
            "terminal_mode": run["terminal_mode"],
            "evaluation_status": evaluation["status"],
            "score": evaluation["score"],
            "iterations": run["iterations"],
            "accepted": accepted,
            "terminal_contract_valid": terminal_contract_valid,
            "run_artifact": str(
                output_dir / f"goal_resolver_run_{run['case_id']}.json"
            ),
        }
        rows.append(row)

    summary = {
        "accepted_count": accepted_count,
        "total_count": len(runs),
        "all_accepted": accepted_count == len(runs),
        "mcp_status": mcp_status,
        "runs": rows,
    }
    return summary


async def _start_mcp_best_effort() -> str:
    """Start MCP tools when configured, without blocking non-web cases."""

    try:
        await mcp_manager.start()
    except Exception as exc:
        status = f"unavailable: {exc}"
    else:
        status = "started"
    return status


async def _stop_mcp_best_effort() -> None:
    """Stop MCP tools at the process boundary."""

    try:
        await mcp_manager.stop()
    except Exception as exc:
        print(f"MCP stop failed: {exc}", file=sys.stderr)


async def main_async(argv: list[str] | None = None) -> int:
    """Run selected POC cases and return a process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.all and not args.case_id:
        parser.error("pass --all or at least one --case-id")

    try:
        output_dir = _resolve_output_dir(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.all:
        cases = [dict(case) for case in GOAL_RESOLVER_CASES]
    else:
        cases = select_cases(args.case_id)

    write_casebook(output_dir, cases)
    mcp_status = await _start_mcp_best_effort()
    runs: list[dict[str, Any]] = []
    try:
        for case in cases:
            run = await run_goal_resolver_case_async(
                case,
                output_dir,
                max_iterations=args.max_iterations,
                repair_passes=args.repair_passes,
            )
            runs.append(run)
            write_run_artifact(output_dir, run)
            status = run["case_evaluation"]["status"]
            print(
                f"{run['case_id']}: {run['terminal_mode']} / "
                f"{status} / {run['iterations']} iterations"
            )
    finally:
        await _stop_mcp_best_effort()

    summary = _summary_payload(runs, output_dir, mcp_status)
    write_summary(output_dir, summary)
    write_json(output_dir / "goal_resolver_raw_runs.json", runs)
    if summary["all_accepted"]:
        exit_code = 0
    else:
        exit_code = 1
    return exit_code


def main() -> None:
    """Synchronous console entrypoint."""

    _configure_console()
    exit_code = asyncio.run(main_async())
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
