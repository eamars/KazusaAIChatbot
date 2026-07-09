# coding agent final integration Gemma4 baseline results

## Summary

- Baseline model family: Gemma4 route configuration used for the final
  integration run before the Ornith comparison.
- Hardening commit under test:
  `568fa71 Complete coding agent full workflow hardening`.
- Baseline review commit:
  `690f973 Record coding agent final integration gate review`.
- Review artifact:
  `development_plans/active/short_term/coding_agent_final_integration_gate_reviews.md`.
- Test file:
  `tests/test_coding_agent_full_workflow_integration_live_llm.py`.
- Entry boundary: real L2d action selection, action-spec execution,
  accepted-task/background-work state, background worker tick, and coding
  worker.

## Gate Results

| Gate | Result | Runtime | Main observation |
|---|---:|---:|---|
| 01 read-only source question | pass | 55.92s | Read-only codebase answer was source-grounded and side-effect free. |
| 02 source-free proposal revision | fail | 996.03s | L2d and run binding were correct, but the coding LLM generated duplicate artifact paths during revised source-free proposal generation. |
| 03 existing-source runtime-only revision | pass | 358.06s | Same-run revision and summary preserved runtime-only changed files. |
| 04 approval and focused verification | pass | 164.01s | Protected test artifact was omitted before managed apply; focused pytest passed. |
| 05 hard multi-file workflow | pass | 337.38s | Durable run identity, approval, summary, terminal cancel rejection, and final status worked; README scope drift noted. |

## Baseline Judgment

The Gemma4 run passed 4 of 5 real LLM gates. The failed gate did not expose a
deterministic code defect; deterministic validation rejected a malformed
source-free revised artifact package. This baseline is the comparison point
for the Ornith/Qwen-compatible model run.

## Raw Trace References

The raw trace files are local test artifacts under
`test_artifacts/llm_traces/` and may be regenerated. The committed review
artifact above is the stable baseline evidence.

