# Native V2 Failure-Mode Closure — 2026-07-23

## Result

The native Cognition Core V2 observability path now distinguishes successful,
partial, failed, missing, and malformed cognition telemetry. Invalid telemetry
does not enter the operator graph as a successful result.

This closes the failure-coverage implementation slice of Stage 3. Stage 3
remains `in_progress` because the independent in-app Browser acceptance and
the previously recorded broader-regression reconciliation are still open.

## Failure-mode map

| Failure mode | Existing rule or code path | Previous gap | Closure evidence |
| --- | --- | --- | --- |
| Malformed JSON, unknown keys, wrong types, missing fields, conflicting fields | Canonical JSON parsing plus bounded producing-stage repair/regeneration | Existing policy was not reflected in native graph state | Existing bounded-error overlay; native contract tests remain green |
| Recoverable numeric bound violation | Deterministic normalization and revalidation | No native observability invariant check | Existing bounded-error overlay; metric contract tests |
| Provider transient or context-limit failure | Typed failure classification and fail-closed service response | Hard V2 failure had no native failure node | `test_response_graph_projects_terminal_native_failure_metadata` |
| Appraisal failure | Independent appraisal slot isolation | Failure was emitted as `not_reported` with no cause | Failed appraisal rows now carry `failure_code`; contract and graph tests |
| Mixed branch completion/failure | Parallel executor preserves failed slots | Mixed execution was labeled `completed` | `test_response_graph_marks_mixed_native_branch_failure_as_partial`; browser E2E |
| Required branch failure or dependency skip | Required failures fail closed before collapse | Operator graph did not preserve the typed terminal state | Existing dependency failure tests plus terminal failure graph projection |
| Collapse or action-planning failure | `CognitionExecutionError` at the owning stage | Failure response exposed only generic operational state | Terminal `v2.failure` node carries bounded code, stage, checkpoint, retryability |
| Missing native observability | V2 output envelope | Native V2 output could appear without telemetry | Explicit `native_observability_missing` failure node |
| Malformed native observability | Native observability contract | Invalid rows/metrics were silently dropped by graph projection | Explicit `native_observability_invalid` failure node |
| Inconsistent counts/timing/partition | Native observability contract | Counts, timings, and branch selection could contradict each other | Six metric cases plus selection mismatch rejection tests |
| Orphan console dependency edge | Console graph projection | Invalid endpoints were retained after node filtering | Orphan edge is dropped and covered by console contract test |
| Browser rendering of failure state | Shared graph node/inspector renderer | Failure code was projected but below the visible inspector area | Failure code moved to the top; system-Chrome E2E screenshots verify it |
| Structurally valid but semantically wrong model output | LLM owns semantic judgment | Deterministic rewriting would hide model-quality evidence | Preserved as quality evidence; no semantic rewrite added |

## Verification

- Focused cognition, dependency, service-graph, and console suites: **55 passed**.
- Non-DB cognition integration suite: **5 passed, 4 deselected**.
- Control-console graph browser suite: **4 passed** using the system-Chrome
  Playwright harness; browser console diagnostics were empty.
- Python `py_compile`: passed for all modified Python modules and tests.
- `git diff --check`: passed; only expected Windows line-ending warnings were
  emitted.
- In-app Browser: unavailable; `agent.browsers.list()` returned no sessions.

Browser screenshots from the final E2E run:

- `C:\Users\rba90\AppData\Local\Temp\pytest-of-rba90\pytest-301\test_native_v2_failure_states_0\artifacts\native_v2_failure_branch.png`
- `C:\Users\rba90\AppData\Local\Temp\pytest-of-rba90\pytest-301\test_native_v2_failure_states_0\artifacts\native_v2_terminal_failure.png`
- `C:\Users\rba90\AppData\Local\Temp\pytest-of-rba90\pytest-301\test_native_v2_failure_states_0\artifacts\native_v2_failure_parallel.png`

The retained real-LLM emotion and boundary cases remain the Stage 3 quality
overlay; this closure adds deterministic failure plumbing and browser guards,
not censorship or semantic emotion rewriting.
