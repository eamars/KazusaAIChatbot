"""Live LLM checks for the recursive coding PM role."""

from __future__ import annotations

import json
from typing import Any

import pytest

from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
    ACTION_PAYLOAD_KEYS,
    PM_STATUSES,
    decide_writing_work,
)
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_SUITE_NAME = "coding_agent_pm_lifecycle_role_live_llm"


async def _run_pm_lifecycle_case(case: dict[str, Any]) -> None:
    """Run one production PM decision case and write an inspection trace."""

    trace: dict[str, object] = {}
    decision = await decide_writing_work(case["pm_input"], trace=trace)
    trace_path = write_llm_trace(
        _SUITE_NAME,
        case["case_id"],
        {
            "case_id": case["case_id"],
            "behavior_contract": case["behavior_contract"],
            "input_kind": case["input_kind"],
            "expected_status_family": case["expected_status_family"],
            "pm_input": case["pm_input"],
            "decision": decision,
            "production_trace": trace,
            "prompt_chars": _prompt_chars(trace),
            "model": {
                "route": trace.get("effective_route"),
                "model": trace.get("model"),
                "thinking_enabled": trace.get("thinking_enabled"),
            },
        },
    )
    print(f"PM lifecycle live trace={trace_path}")

    status = decision["status"]
    assert status in PM_STATUSES, f"trace={trace_path}"
    expected_statuses = case.get(
        "acceptable_status_families",
        [case["expected_status_family"]],
    )
    assert status in expected_statuses, f"trace={trace_path}"
    assert decision["reason"].strip(), f"trace={trace_path}"

    populated_payloads = [
        key
        for key in ACTION_PAYLOAD_KEYS.values()
        if decision.get(key) is not None
    ]
    assert len(populated_payloads) == 1, f"trace={trace_path}"
    expected_payload_key = ACTION_PAYLOAD_KEYS[status]
    assert populated_payloads[0] == expected_payload_key, f"trace={trace_path}"
    assert isinstance(decision[expected_payload_key], dict), f"trace={trace_path}"
    _assert_case_specific_contract(case, decision, trace_path)


def _prompt_chars(trace: dict[str, object]) -> int:
    system_prompt = trace.get("system_prompt")
    human_payload_text = trace.get("human_payload_text")
    if not isinstance(system_prompt, str) or not isinstance(human_payload_text, str):
        return 0
    return len(system_prompt) + len(human_payload_text)


def _assert_case_specific_contract(
    case: dict[str, Any],
    decision: dict[str, Any],
    trace_path: str,
) -> None:
    """Check role-specific contract risks that the generic shape cannot see."""

    if case["case_id"] != "gate_02_source_contract_after_processing":
        if case["case_id"] == "gate_04_tests_repeat_source_literals":
            _assert_gate_04_test_contract(decision, trace_path)
        return

    programmer_task = decision["programmer_task"]
    interfaces = "\n".join(programmer_task["provided_interfaces"])
    assert "Callable" not in interfaces, f"trace={trace_path}"
    assert "Generator" not in interfaces, f"trace={trace_path}"
    assert "Tuple" not in interfaces, f"trace={trace_path}"
    assert "Result" in interfaces, f"trace={trace_path}"
    has_named_record = (
        "TypedDict" in interfaces
        or "dataclass" in interfaces
        or "NamedTuple" in interfaces
    )
    assert has_named_record, f"trace={trace_path}"
    assert "rows" in interfaces, f"trace={trace_path}"
    has_column_names = (
        "field_names" in interfaces
        or "header" in interfaces
        or "headers" in interfaces
        or "columns" in interfaces
    )
    assert has_column_names, f"trace={trace_path}"
    has_malformed_report = (
        "malformed_line_numbers" in interfaces
        or "malformed_lines" in interfaces
        or "errors" in interfaces
        or "warnings" in interfaces
    )
    assert has_malformed_report, f"trace={trace_path}"


def _assert_gate_04_test_contract(
    decision: dict[str, Any],
    trace_path: str,
) -> None:
    """Check that test tasks preserve source-owned format literals."""

    programmer_task = decision["programmer_task"]
    task_text = "\n".join(
        [
            *programmer_task["required_behavior"],
            *programmer_task["consumed_interfaces"],
        ]
    )
    assert ".txt" in task_text, f"trace={trace_path}"
    assert "Project:" in task_text, f"trace={trace_path}"


PM_LIFECYCLE_CASES: dict[str, dict[str, Any]] = {
    "gate_01_single_file_programmer_task": {
        "case_id": "gate_01_single_file_programmer_task",
        "behavior_contract": (
            "For a single independent script, the PM should be able to create "
            "one bounded programmer task directly."
        ),
        "input_kind": "hard_gate_derived",
        "expected_status_family": "create_programmer_task",
        "pm_input": {
            "pm_id": "writing_pm_gate_01",
            "domain": "writing",
            "work_item": {
                "goal": (
                    "Create a single Python command-line script that reads a "
                    "plain text application log file and counts entries by "
                    "severity. Valid lines start with DEBUG, INFO, WARNING, "
                    "ERROR, or CRITICAL followed by a space and the message. "
                    "The script prints one count per severity, reports "
                    "malformed lines skipped, handles a missing input file "
                    "clearly, and uses only the Python standard library."
                ),
                "scope": "one standalone new Python script",
                "constraints": [
                    "new artifact only",
                    "standard library only",
                    "no tests required for this work item",
                ],
                "expected_result": (
                    "one complete programmer task for one command-line script"
                ),
            },
            "available_facts": [],
            "direct_child_reports": [],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "gate_02_source_and_tests_first_programmer_task": {
        "case_id": "gate_02_source_and_tests_first_programmer_task",
        "behavior_contract": (
            "For a small utility with source and tests, the PM should start "
            "with one direct programmer task for the source artifact when the "
            "source contract is clear."
        ),
        "input_kind": "hard_gate_derived",
        "expected_status_family": "create_programmer_task",
        "pm_input": {
            "pm_id": "writing_pm_gate_02",
            "domain": "writing",
            "work_item": {
                "goal": (
                    "Create a small Python utility that converts JSONL records "
                    "into CSV. It should include a command-line script and "
                    "focused tests. The CLI accepts input and output paths, an "
                    "optional list of fields, preserves stable column order, "
                    "writes blank cells for missing fields, reports malformed "
                    "JSON lines without aborting, and uses only the Python "
                    "standard library."
                ),
                "scope": "new utility source plus focused tests",
                "constraints": [
                    "new artifacts only",
                    "standard library only",
                    "source and tests must agree on interface behavior",
                ],
                "expected_result": (
                    "one direct programmer task for the conversion source or CLI"
                ),
            },
            "available_facts": [],
            "direct_child_reports": [],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "gate_02_source_report_tests_programmer_task": {
        "case_id": "gate_02_source_report_tests_programmer_task",
        "behavior_contract": (
            "When the source child report gives the actual callable interface "
            "and behavior facts, the PM should create a direct programmer task "
            "for focused tests instead of adding another PM layer."
        ),
        "input_kind": "captured_trace_followup",
        "expected_status_family": "create_programmer_task",
        "pm_input": {
            "pm_id": "writing_pm_gate_02",
            "domain": "writing",
            "work_item": {
                "goal": (
                    "Create a small Python utility that converts JSONL records "
                    "into CSV. It should include a command-line script and "
                    "focused tests. The CLI accepts input and output paths, an "
                    "optional list of fields, preserves stable column order, "
                    "writes blank cells for missing fields, reports malformed "
                    "JSON lines without aborting, and uses only the Python "
                    "standard library."
                ),
                "scope": "new utility source plus focused tests",
                "constraints": [
                    "new artifacts only",
                    "standard library only",
                    "tests must match the reported source interface",
                ],
                "expected_result": (
                    "one direct programmer task for focused test coverage"
                ),
            },
            "available_facts": [],
            "direct_child_reports": [
                {
                    "child_id": "converter_source_programmer",
                    "status": "complete",
                    "provided_facts": [
                        "function convert_jsonl_to_csv(input_path, output_path, fields=None)",
                        "CLI accepts input path, output path, and optional comma-separated fields",
                        "malformed JSON lines are reported to stderr and skipped",
                        "missing fields are written as blank CSV cells",
                    ],
                    "created_artifacts": [
                        {
                            "artifact_id": "converter_source",
                            "path": "src/converter.py",
                            "purpose": "JSONL to CSV conversion source and CLI",
                        }
                    ],
                    "open_risks": [],
                }
            ],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "gate_02_source_contract_after_processing": {
        "case_id": "gate_02_source_contract_after_processing",
        "behavior_contract": (
            "When output names depend on reading JSONL records, the PM should "
            "give the programmer a direct Python function contract that returns "
            "a named result after processing. It should not split a lazy row "
            "generator from a header value that is unknown before iteration."
        ),
        "input_kind": "captured_failure_pattern",
        "expected_status_family": "create_programmer_task",
        "pm_input": {
            "pm_id": "writing_pm_gate_02_contract",
            "domain": "writing",
            "work_item": {
                "goal": (
                    "Create the source artifact for a Python utility that "
                    "converts JSONL records into CSV. The source is consumed by "
                    "a later CLI and focused tests. The conversion accepts "
                    "JSONL lines and an optional list of fields. If fields are "
                    "not supplied, the first valid JSON object decides the "
                    "stable column order. Missing field values become blank "
                    "cells. Malformed JSON lines are reported without stopping "
                    "conversion."
                ),
                "scope": "source artifact consumed by later CLI and tests",
                "constraints": [
                    "new artifact only",
                    "standard library only",
                    (
                        "the PM must name the result fields that downstream "
                        "CLI and tests will consume"
                    ),
                    (
                        "the header field names are known only after the source "
                        "has read valid input rows"
                    ),
                ],
                "expected_result": (
                    "one direct programmer task with a Python-native result "
                    "contract for completed conversion output"
                ),
            },
            "available_facts": [
                {
                    "kind": "captured_failure",
                    "summary": (
                        "A previous PM contract returned a row generator and a "
                        "header tuple. The CLI wrote the returned header before "
                        "the generator processed the first valid row, so the "
                        "header was empty when fields were not supplied."
                    ),
                }
            ],
            "direct_child_reports": [],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "gate_02_tests_need_csv_shape_readback": {
        "case_id": "gate_02_tests_need_csv_shape_readback",
        "behavior_contract": (
            "Before assigning tests for a generated CSV writer, the PM should "
            "request actual source behavior when the child report does not say "
            "whether the output includes a header row or only data rows."
        ),
        "input_kind": "captured_failure_pattern",
        "expected_status_family": "request_information",
        "pm_input": {
            "pm_id": "writing_pm_gate_02_tests",
            "domain": "writing",
            "work_item": {
                "goal": (
                    "Create focused tests for a Python JSONL-to-CSV utility. "
                    "The tests must verify normal conversion, stable field "
                    "order, blank cells for missing fields, and malformed JSON "
                    "line handling."
                ),
                "scope": "test artifact that depends on generated source",
                "constraints": [
                    "tests must match the actual generated source behavior",
                    (
                        "CSV row shape must be known before tests assert exact "
                        "rows"
                    ),
                ],
                "expected_result": (
                    "workspace-fact request for the generated source behavior"
                ),
            },
            "available_facts": [
                {
                    "kind": "child_report",
                    "summary": (
                        "The source artifact exists and exposes a conversion "
                        "function, but the report does not say whether the CSV "
                        "writer writes a header row before data rows."
                    ),
                }
            ],
            "direct_child_reports": [
                {
                    "child_id": "jsonl_to_csv_engine",
                    "status": "complete",
                    "provided_facts": [
                        (
                            "convert_jsonl_to_csv(input_stream, output_writer, "
                            "fields=None) returns processed_count and "
                            "error_count"
                        ),
                        (
                            "field order comes from provided fields or the "
                            "first valid JSON object"
                        ),
                        (
                            "malformed JSON lines are counted without stopping "
                            "conversion"
                        ),
                    ],
                    "created_artifacts": [
                        {
                            "artifact_id": "jsonl_to_csv_engine",
                            "path": "src/jsonl_csv_engine.py",
                            "purpose": "JSONL to CSV conversion source",
                        }
                    ],
                    "open_risks": [],
                }
            ],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "gate_02_actual_tests_need_csv_shape_readback": {
        "case_id": "gate_02_actual_tests_need_csv_shape_readback",
        "behavior_contract": (
            "When a PM is about to assign tests that assert exact CSV rows, "
            "and the completed source and CLI reports do not state whether the "
            "header is written as an output row, the PM should request source "
            "behavior readback before assigning the tests."
        ),
        "input_kind": "captured_production_trace",
        "expected_status_family": "request_information",
        "pm_input": {
            "pm_id": "writing_pm_root",
            "domain": "writing",
            "work_item": {
                "goal": (
                    "Create a small Python utility that converts JSONL records "
                    "into CSV. It should include a command-line script and "
                    "focused tests. The CLI must accept input and output paths, "
                    "an optional list of fields, preserve stable column order, "
                    "write blank cells for missing fields, report malformed "
                    "JSON lines without aborting the whole conversion, and use "
                    "only the Python standard library."
                ),
                "scope": "new-artifact writing request",
                "constraints": [
                    "new artifacts only",
                    "do not mutate the caller workspace",
                    "do not run commands or tests",
                ],
                "expected_result": "patch proposal package for new artifacts",
            },
            "available_facts": [
                {
                    "kind": "acceptance_criteria",
                    "summary": "Preserved user-visible requirements.",
                    "criteria": [
                        {
                            "criterion_id": "cli_interface",
                            "requirement": (
                                "The utility must be a command-line script "
                                "that accepts an input file path, an output "
                                "file path, and an optional list of fields."
                            ),
                            "evidence_needed": (
                                "A Python script with a main entry point using "
                                "argparse or sys.argv to handle these specific "
                                "inputs."
                            ),
                        },
                        {
                            "criterion_id": "conversion_logic",
                            "requirement": (
                                "Convert JSONL records into CSV format, "
                                "ensuring stable column order and writing "
                                "blank cells for missing fields."
                            ),
                            "evidence_needed": (
                                "CSV output where columns are consistent "
                                "across rows and missing keys in JSON objects "
                                "result in empty comma-separated values."
                            ),
                        },
                        {
                            "criterion_id": "error_handling",
                            "requirement": (
                                "Report malformed JSON lines to stderr or "
                                "stdout without aborting the conversion of "
                                "subsequent valid lines."
                            ),
                            "evidence_needed": (
                                "Code containing a try-except block around the "
                                "JSON parsing logic that logs an error and "
                                "continues the loop."
                            ),
                        },
                        {
                            "criterion_id": "dependency_constraint",
                            "requirement": (
                                "Use only the Python standard library for all "
                                "functionality."
                            ),
                            "evidence_needed": (
                                "Import statements limited to built-in modules "
                                "(e.g., json, csv, argparse, sys)."
                            ),
                        },
                        {
                            "criterion_id": "testing",
                            "requirement": (
                                "Include focused tests verifying the "
                                "conversion logic and error handling."
                            ),
                            "evidence_needed": (
                                "A test file or block using unittest or "
                                "pytest-style assertions covering valid JSONL, "
                                "missing fields, and malformed lines."
                            ),
                        },
                    ],
                }
            ],
            "direct_child_reports": [
                {
                    "child_id": "jsonl_to_csv_engine",
                    "status": "complete",
                    "provided_facts": [
                        (
                            "convert_jsonl_to_csv(input_stream: Iterable[str], "
                            "output_writer: csv.Writer, "
                            "fields: Optional[List[str]] = None) -> "
                            "ConversionResult"
                        ),
                        (
                            "ConversionResult: TypedDict { 'processed_count': "
                            "int, 'error_count': int }"
                        ),
                        "Read lines from an input iterable of strings.",
                        (
                            "Parse each line as JSON. If parsing fails, record "
                            "the error and continue to the next line."
                        ),
                        (
                            "If a list of fields is provided, use only those "
                            "fields in that specific order for CSV columns."
                        ),
                        (
                            "If no fields are provided, determine the column "
                            "headers from the keys of the first successfully "
                            "parsed JSON object."
                        ),
                        (
                            "Write records to a CSV writer. Missing keys in "
                            "subsequent objects must result in blank cells."
                        ),
                        "Ensure stable column ordering throughout the output file.",
                        (
                            "Return a summary record containing the number of "
                            "successfully processed lines and the number of "
                            "malformed lines encountered."
                        ),
                    ],
                    "created_artifacts": [
                        {
                            "artifact_id": "jsonl_to_csv_engine",
                            "path": "src/jsonl_csv_engine.py",
                            "purpose": (
                                "Provide a core function to convert JSONL "
                                "records to CSV format with field filtering "
                                "and error resilience."
                            ),
                        }
                    ],
                    "open_risks": [],
                },
                {
                    "child_id": "jsonl_to_csv_cli",
                    "status": "complete",
                    "provided_facts": [
                        (
                            "Main entry point: python cli.py <input_path> "
                            "<output_path> [--fields field1,field2,...]"
                        ),
                        (
                            "Use argparse to accept three arguments: a "
                            "required input file path, a required output file "
                            "path, and an optional comma-separated list of "
                            "fields."
                        ),
                        (
                            "Open the input file for reading and the output "
                            "file for writing in text mode."
                        ),
                        (
                            "Initialize a csv.writer object using the opened "
                            "output file."
                        ),
                        (
                            "Call convert_jsonl_to_csv from "
                            "src/jsonl_csv_engine.py, passing the input file "
                            "handle as the input_stream and the csv.writer "
                            "instance as the output_writer."
                        ),
                        (
                            "If fields are provided via the CLI, split the "
                            "comma-separated string into a list of strings and "
                            "pass it to the engine."
                        ),
                        (
                            "Print the resulting ConversionResult "
                            "(processed_count and error_count) to stdout upon "
                            "completion."
                        ),
                        (
                            "Ensure all file handles are properly closed using "
                            "context managers."
                        ),
                        "Use only Python standard library modules.",
                    ],
                    "created_artifacts": [
                        {
                            "artifact_id": "jsonl_to_csv_cli",
                            "path": "src/jsonl_csv_cli.py",
                            "purpose": (
                                "A command-line script to convert JSONL files "
                                "to CSV using the jsonl_to_csv_engine."
                            ),
                        }
                    ],
                    "open_risks": [],
                },
            ],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "gate_03_package_information_request": {
        "case_id": "gate_03_package_information_request",
        "behavior_contract": (
            "When tests depend on generated source that already exists, the PM "
            "should request workspace facts before writing the dependent task."
        ),
        "input_kind": "captured_failure_pattern",
        "expected_status_family": "request_information",
        "pm_input": {
            "pm_id": "writing_pm_gate_03_tests",
            "domain": "writing",
            "work_item": {
                "goal": (
                    "Create focused tests for a Markdown link checker package. "
                    "The tests must match the reusable checker behavior and "
                    "cover anchor generation, duplicate anchors, and broken "
                    "relative links."
                ),
                "scope": "test artifact that depends on generated source",
                "constraints": [
                    "do not infer source return shape from memory",
                    "tests must consume the actual generated checker interface",
                ],
                "expected_result": (
                    "workspace-fact request for generated source behavior"
                ),
            },
            "available_facts": [
                {
                    "kind": "child_report",
                    "summary": (
                        "A source artifact was generated for Markdown link "
                        "checking, but no evidence-backed function signature "
                        "or return shape has been read back yet."
                    ),
                }
            ],
            "direct_child_reports": [
                {
                    "child_id": "source_file_pm",
                    "status": "complete",
                    "created_artifacts": [
                        {
                            "artifact_id": "markdown_checker_source",
                            "path": "src/checker.py",
                        }
                    ],
                    "provided_facts": [
                        "source exists",
                        "source is intended to expose reusable checking logic",
                    ],
                }
            ],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "gate_03_core_interface_schema_reproduction": {
        "case_id": "gate_03_core_interface_schema_reproduction",
        "behavior_contract": (
            "For a reusable Python source artifact that returns structured "
            "data consumed by later CLI and test artifacts, the PM should use "
            "Python-native type hints with named input and output items, named "
            "list element types, and named dict or record fields."
        ),
        "input_kind": "captured_failure_pattern",
        "expected_status_family": "create_programmer_task",
        "pm_input": {
            "pm_id": "writing_pm_gate_03_root",
            "domain": "writing",
            "work_item": {
                "goal": (
                    "Create a small Python package for checking local Markdown "
                    "links. It should provide a reusable function and a CLI. "
                    "The checker must scan markdown files under a directory, "
                    "collect headings as anchors, report duplicate anchors "
                    "inside one file, report broken relative links to local "
                    "markdown files or anchors, and include focused tests for "
                    "anchor generation, duplicate anchors, and broken relative "
                    "links."
                ),
                "scope": "new package source, CLI, and focused tests",
                "constraints": [
                    "new artifacts only",
                    "standard library only",
                    "later CLI and tests must consume the source interface",
                    (
                        "Python interfaces should use named type hints for "
                        "structured input and output data"
                    ),
                ],
                "expected_result": (
                    "first direct programmer task for the reusable checker "
                    "source artifact"
                ),
            },
            "available_facts": [
                {
                    "kind": "acceptance_criteria",
                    "summary": "Preserved user-visible requirements.",
                    "criteria": [
                        {
                            "criterion_id": "reusable_function",
                            "requirement": (
                                "The package must provide a reusable Python "
                                "function that performs the markdown link check."
                            ),
                            "evidence_needed": (
                                "A public function in the source code that "
                                "accepts a directory path and returns results "
                                "of the link validation."
                            ),
                        },
                        {
                            "criterion_id": "cli_interface",
                            "requirement": (
                                "The package must include a command-line "
                                "interface to run the checker."
                            ),
                            "evidence_needed": (
                                "A CLI entry point that allows users to specify "
                                "a directory for scanning."
                            ),
                        },
                        {
                            "criterion_id": "duplicate_anchor_reporting",
                            "requirement": (
                                "The tool must report duplicate anchors found "
                                "within a single file."
                            ),
                            "evidence_needed": (
                                "Output identifying files with non-unique "
                                "heading anchors."
                            ),
                        },
                        {
                            "criterion_id": "broken_link_reporting",
                            "requirement": (
                                "The tool must report broken relative links to "
                                "local markdown files and broken anchor links."
                            ),
                            "evidence_needed": (
                                "Reports identifying links to missing files or "
                                "anchors that do not exist in the target file."
                            ),
                        },
                        {
                            "criterion_id": "test_suite",
                            "requirement": (
                                "Focused tests must cover anchor generation, "
                                "duplicate anchor detection, and broken "
                                "relative link identification."
                            ),
                            "evidence_needed": (
                                "A test suite with cases covering those "
                                "behaviors."
                            ),
                        },
                    ],
                }
            ],
            "direct_child_reports": [],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "gate_04_child_report_followup": {
        "case_id": "gate_04_child_report_followup",
        "behavior_contract": (
            "When direct child reporting shows the assigned work is incomplete "
            "but enough interface facts are present, the PM should create a "
            "direct programmer task for the missing artifact."
        ),
        "input_kind": "captured_failure_pattern",
        "expected_status_family": "create_programmer_task",
        "pm_input": {
            "pm_id": "writing_pm_gate_04",
            "domain": "writing",
            "work_item": {
                "goal": (
                    "Create a small Python CLI project that summarizes task "
                    "notes from dated text files, groups entries by project, "
                    "writes Markdown, supports JSON config for input, output, "
                    "and included projects, includes README, and includes "
                    "focused tests."
                ),
                "scope": "new multi-file task-note summary project",
                "constraints": [
                    "CLI access is required",
                    "JSON config support is required",
                    "README must match actual workflow",
                ],
                "expected_result": "complete coherent artifact set",
            },
            "available_facts": [],
            "direct_child_reports": [
                {
                    "child_id": "task_note_project_pm",
                    "status": "complete",
                    "assigned_goal": (
                        "Own the complete task-note CLI project, including "
                        "parser, renderer, CLI entrypoint, JSON config "
                        "handling, README, and focused tests."
                    ),
                    "assigned_scope": (
                        "complete multi-file project, not a source-only slice"
                    ),
                    "provided_facts": [
                        "parser and renderer artifacts were created",
                        "function parse_notes(input_dir) returns entries with project, date, and text fields",
                        "function render_summary(entries, included_projects=None) returns Markdown text",
                        "no command-line entrypoint was reported",
                    ],
                    "created_artifacts": [
                        {
                            "artifact_id": "parser",
                            "purpose": "parse dated task notes",
                        },
                        {
                            "artifact_id": "renderer",
                            "purpose": "render Markdown summary",
                        },
                    ],
                    "open_risks": ["missing CLI entrypoint"],
                }
            ],
            "child_feedback": [
                {
                    "stage": "child_report",
                    "child_id": "task_note_project_pm",
                    "summary": (
                        "The direct child report does not identify a CLI "
                        "entrypoint or explain whether the CLI exists."
                    ),
                }
            ],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "gate_04_unspecified_input_format_define_locally": {
        "case_id": "gate_04_unspecified_input_format_define_locally",
        "behavior_contract": (
            "For a new Python CLI project with an unspecified text input "
            "format, the PM should define a simple local format through the "
            "generated artifacts instead of repeatedly requesting external "
            "evidence for a public standard."
        ),
        "input_kind": "captured_failure_pattern",
        "expected_status_family": "create_child_pm",
        "acceptable_status_families": [
            "create_child_pm",
            "create_programmer_task",
        ],
        "pm_input": {
            "pm_id": "writing_pm_gate_04",
            "domain": "writing",
            "work_item": {
                "goal": (
                    "Create a small Python CLI project that summarizes task "
                    "notes. It should read a directory of dated text notes, "
                    "group entries by project name, write a summary Markdown "
                    "file, support a simple JSON config file for input "
                    "directory, output path, and included projects, include a "
                    "README explaining the workflow, and include focused tests "
                    "for parsing notes, applying config filters, and rendering "
                    "the summary."
                ),
                "scope": "new-artifact writing request",
                "constraints": [
                    "new artifacts only",
                    "do not mutate the caller workspace",
                    "do not run commands or tests",
                ],
                "expected_result": "patch proposal package for new artifacts",
            },
            "available_facts": [
                {
                    "kind": "external_evidence",
                    "summary": (
                        "No public standard input format was found for the "
                        "requested dated task notes."
                    ),
                    "limitation": "External evidence was unavailable.",
                }
            ],
            "direct_child_reports": [],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "gate_04_tests_repeat_source_literals": {
        "case_id": "gate_04_tests_repeat_source_literals",
        "behavior_contract": (
            "When assigning tests for generated parser and renderer code, the "
            "PM should repeat exact source-owned literals such as file "
            "extension and note markers so the test programmer does not invent "
            "a different input format."
        ),
        "input_kind": "captured_failure_pattern",
        "expected_status_family": "create_programmer_task",
        "pm_input": {
            "pm_id": "writing_pm_gate_04_tests",
            "domain": "writing",
            "work_item": {
                "goal": (
                    "Create focused tests for a Python task-note summary "
                    "project. The tests must verify parsing notes, applying "
                    "project filters, and rendering Markdown output."
                ),
                "scope": "test artifact that depends on generated source",
                "constraints": [
                    "tests must match the generated parser input format",
                    "tests must match the generated renderer output format",
                ],
                "expected_result": (
                    "one direct programmer task for focused tests"
                ),
            },
            "available_facts": [],
            "direct_child_reports": [
                {
                    "child_id": "note_processor",
                    "status": "complete",
                    "provided_facts": [
                        (
                            "process_notes(input_dir: str, "
                            "included_projects: List[str]) -> "
                            "Dict[str, List[NoteEntry]]"
                        ),
                        "Iterate through all .txt files in the input directory.",
                        "Extract the date from YYYY-MM-DD.txt filenames.",
                        (
                            "Parse entries that begin with the exact marker "
                            "'Project: <project_name>'."
                        ),
                    ],
                    "created_artifacts": [
                        {
                            "artifact_id": "note_processor",
                            "path": "src/note_processor.py",
                        }
                    ],
                    "open_risks": [],
                },
                {
                    "child_id": "markdown_renderer",
                    "status": "complete",
                    "provided_facts": [
                        (
                            "render_summary(grouped_notes: Dict[str, "
                            "List[NoteEntry]], output_path: str) -> None"
                        ),
                        (
                            "For each project, write a Markdown second-level "
                            "header using '## <project>'."
                        ),
                        (
                            "For each note, write a bullet prefixed with the "
                            "note date."
                        ),
                    ],
                    "created_artifacts": [
                        {
                            "artifact_id": "markdown_renderer",
                            "path": "src/markdown_renderer.py",
                        }
                    ],
                    "open_risks": [],
                },
            ],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "broad_project_child_pm": {
        "case_id": "broad_project_child_pm",
        "behavior_contract": (
            "When the assigned work is a broad project slice with several "
            "dependent artifacts and no first artifact contract is yet safe, "
            "the PM should create a child PM for that cohesive sub-area."
        ),
        "input_kind": "synthetic_role_boundary",
        "expected_status_family": "create_child_pm",
        "pm_input": {
            "pm_id": "writing_pm_project_root",
            "domain": "writing",
            "work_item": {
                "goal": (
                    "Create the data-processing portion of a new command-line "
                    "project, including input parsing, domain model, report "
                    "rendering, focused tests, and a short README section."
                ),
                "scope": (
                    "cohesive data-processing sub-area with several dependent "
                    "new artifacts"
                ),
                "constraints": [
                    "new artifacts only",
                    "standard library only",
                    "source, tests, and docs must agree on interfaces",
                ],
                "expected_result": (
                    "child PM task that owns local artifact ordering and reports "
                    "the selected interfaces upward"
                ),
            },
            "available_facts": [],
            "direct_child_reports": [],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "gate_05_dependent_tests_information_request": {
        "case_id": "gate_05_dependent_tests_information_request",
        "behavior_contract": (
            "Before creating mocked HTTP tests for generated fetch code, the "
            "PM should request readback of actual source behavior."
        ),
        "input_kind": "captured_failure_pattern",
        "expected_status_family": "request_information",
        "pm_input": {
            "pm_id": "writing_pm_gate_05_tests",
            "domain": "writing",
            "work_item": {
                "goal": (
                    "Create mocked HTTP tests for a small standard-library "
                    "project that reads a CSV inventory of pages, fetches each "
                    "URL, extracts HTML title and first h1, merges those "
                    "values with inventory rows, and writes a consolidated CSV."
                ),
                "scope": "test artifact that depends on generated source",
                "constraints": [
                    "tests must not perform real network access",
                    "tests must match actual fetch error behavior",
                    "tests must match actual CSV merge behavior",
                ],
                "expected_result": (
                    "workspace-fact request for generated source behavior"
                ),
            },
            "available_facts": [
                {
                    "kind": "artifact_manifest",
                    "created_files": [
                        "src/fetcher.py",
                        "src/processor.py",
                        "src/main.py",
                    ],
                    "summary": (
                        "Source files were generated, but actual function "
                        "signatures and error behavior have not been read."
                    ),
                }
            ],
            "direct_child_reports": [],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "gate_05_child_pm_complete_project_report": {
        "case_id": "gate_05_child_pm_complete_project_report",
        "behavior_contract": (
            "When a child PM reports a complete multi-artifact project that "
            "covers the parent PM's assigned scope, the parent PM should "
            "complete with an upward report instead of assigning duplicate "
            "programmer tasks for the same artifacts."
        ),
        "input_kind": "captured_failure_pattern",
        "expected_status_family": "complete",
        "pm_input": {
            "pm_id": "writing_pm_root",
            "domain": "writing",
            "work_item": {
                "goal": (
                    "Create a small Python project that reads a CSV inventory "
                    "of pages, fetches each listed URL, extracts the HTML title "
                    "and first h1 heading, merges those values with the "
                    "inventory rows, and writes a consolidated CSV report. It "
                    "should include a CLI, source modules, mocked HTTP tests, "
                    "and a README that explains the input CSV columns and "
                    "command workflow. The project may use only the Python "
                    "standard library."
                ),
                "scope": "new-artifact writing request",
                "constraints": [
                    "new artifacts only",
                    "do not mutate the caller workspace",
                    "do not run commands or tests",
                ],
                "expected_result": "patch proposal package for new artifacts",
            },
            "available_facts": [],
            "direct_child_reports": [
                {
                    "child_id": "page_extractor_project_pm",
                    "status": "succeeded",
                    "provided_facts": [
                        (
                            "Shared data record PageMetadata = "
                            "TypedDict('PageMetadata', {'title': str, "
                            "'h1': str}) is defined and used across modules."
                        ),
                        (
                            "The system integrates the CLI entry point, CSV "
                            "processor, and HTML extractor pipeline."
                        ),
                        (
                            "HTTP requests are mocked in tests to ensure no "
                            "real network calls occur during verification."
                        ),
                        (
                            "README documents the required url column and "
                            "command workflow."
                        ),
                    ],
                    "created_artifacts": [
                        {
                            "artifact_id": "html_extractor",
                            "purpose": (
                                "Fetches HTML and extracts title and first h1."
                            ),
                        },
                        {
                            "artifact_id": "csv_processor",
                            "purpose": (
                                "Reads URLs from CSV, invokes the extractor, "
                                "and writes merged results to a new CSV."
                            ),
                        },
                        {
                            "artifact_id": "cli_entry_point",
                            "purpose": (
                                "Provides a command-line interface for input "
                                "and output paths."
                            ),
                        },
                        {
                            "artifact_id": "test_html_extractor",
                            "purpose": (
                                "Verifies extraction logic using mocked HTTP."
                            ),
                        },
                        {
                            "artifact_id": "write_readme",
                            "purpose": (
                                "Documents input CSV format and CLI usage."
                            ),
                        },
                    ],
                    "open_risks": [
                        (
                            "Regex-based HTML parsing is fragile for complex "
                            "HTML."
                        )
                    ],
                }
            ],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "child_pm_report_complete": {
        "case_id": "child_pm_report_complete",
        "behavior_contract": (
            "When direct child reports and validation show the assigned work "
            "is done, the PM should complete with a compact report upward."
        ),
        "input_kind": "synthetic_role_boundary",
        "expected_status_family": "complete",
        "pm_input": {
            "pm_id": "writing_pm_source_slice",
            "domain": "writing",
            "work_item": {
                "goal": "Own the source-file slice for a JSONL to CSV utility.",
                "scope": "source artifact only",
                "constraints": [
                    "new artifact only",
                    "standard library only",
                    "report facts needed by downstream tests",
                ],
                "expected_result": "PM report upward",
            },
            "available_facts": [
                {
                    "kind": "review_package",
                    "summary": (
                        "Generated source artifact was materialized for review "
                        "and has no path safety errors."
                    ),
                }
            ],
            "direct_child_reports": [
                {
                    "child_id": "jsonl_csv_programmer",
                    "status": "complete",
                    "provided_facts": [
                        "function convert_jsonl_to_csv(input_path, output_path, fields=None)",
                        "malformed JSON lines are skipped and counted",
                    ],
                    "created_artifacts": [
                        {
                            "artifact_id": "jsonl_csv_source",
                            "purpose": "conversion source and CLI",
                        }
                    ],
                    "open_risks": [],
                }
            ],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
    "blocked_insufficient_goal": {
        "case_id": "blocked_insufficient_goal",
        "behavior_contract": (
            "When the assigned work item is too ambiguous and no narrow "
            "workspace read can answer it, the PM should block specifically."
        ),
        "input_kind": "synthetic_boundary",
        "expected_status_family": "blocked",
        "pm_input": {
            "pm_id": "writing_pm_ambiguous",
            "domain": "writing",
            "work_item": {
                "goal": "Create the tool exactly as needed for my workflow.",
                "scope": "unknown new artifacts",
                "constraints": [
                    "do not guess hidden workflow requirements",
                    "no existing workspace artifact is available to inspect",
                ],
                "expected_result": "specific blocker",
            },
            "available_facts": [],
            "direct_child_reports": [],
            "child_feedback": [],
            "context_limits": {"max_prompt_chars": 50000},
        },
    },
}


async def test_live_pm_gate_01_single_file_programmer_task() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["gate_01_single_file_programmer_task"],
    )


async def test_live_pm_gate_02_source_and_tests_first_programmer_task() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["gate_02_source_and_tests_first_programmer_task"],
    )


async def test_live_pm_gate_02_source_report_tests_programmer_task() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["gate_02_source_report_tests_programmer_task"],
    )


async def test_live_pm_gate_02_source_contract_after_processing() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["gate_02_source_contract_after_processing"],
    )


async def test_live_pm_gate_02_tests_need_csv_shape_readback() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["gate_02_tests_need_csv_shape_readback"],
    )


async def test_live_pm_gate_02_actual_tests_need_csv_shape_readback() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["gate_02_actual_tests_need_csv_shape_readback"],
    )


async def test_live_pm_gate_03_package_information_request() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["gate_03_package_information_request"],
    )


async def test_live_pm_gate_03_core_interface_schema_reproduction() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["gate_03_core_interface_schema_reproduction"],
    )


async def test_live_pm_gate_04_child_report_followup() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["gate_04_child_report_followup"],
    )


async def test_live_pm_gate_04_unspecified_input_format_define_locally() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["gate_04_unspecified_input_format_define_locally"],
    )


async def test_live_pm_gate_04_tests_repeat_source_literals() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["gate_04_tests_repeat_source_literals"],
    )


async def test_live_pm_gate_05_dependent_tests_information_request() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["gate_05_dependent_tests_information_request"],
    )


async def test_live_pm_gate_05_child_pm_complete_project_report() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["gate_05_child_pm_complete_project_report"],
    )


async def test_live_pm_broad_project_child_pm() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["broad_project_child_pm"],
    )


async def test_live_pm_child_pm_report_complete() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["child_pm_report_complete"],
    )


async def test_live_pm_blocked_insufficient_goal() -> None:
    await _run_pm_lifecycle_case(
        PM_LIFECYCLE_CASES["blocked_insufficient_goal"],
    )
