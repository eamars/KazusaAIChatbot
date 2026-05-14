# RAG2.1 Initializer Subagent Contract Failure Record

## Summary

- Status: terminated_failed
- Closed date: 2026-05-14
- Original goal: prove whether a RAG2.1 initializer/subagent contract could
  replace RAG2 by simplifying initializer output and moving retrieval parameter
  generation into capability agents.
- Final decision: do not migrate to RAG2.1. Keep production RAG2 and harden
  concrete RAG2 failures instead.
- This document is a historical failure record only. Do not execute it as an
  implementation plan.

## Decision

RAG2.1 is terminated as a design direction for this phase.

The attempted design did not show enough reliability or necessity to justify a
production migration. The useful ideas from the attempt are narrower and should
be treated as future RAG2 hardening opportunities, not as a replacement system:

- capability agents should own low-level parameter generation;
- evidence can benefit from explicit source-authority envelopes;
- subagents may eventually return typed follow-up needs;
- real-data and real-LLM comparisons must be used before migration decisions.

## Why It Failed

The experiment simplified the initializer in the wrong place. It reduced the
model-facing route contract while preserving hard semantic responsibilities
somewhere else in the system.

Observed failure modes:

- The experimental initializer lost critical speaker scope such as
  `speaker=active_character` and `speaker=current_user`.
- It misrouted named-person impression questions to memory/conversation instead
  of `Person-context`.
- It produced schema-invalid dependency output in multiple live cases.
- It reproduced RAG2's amusement-park overreach failure by adding weather to an
  opening-status query where weather was explicitly out of scope.
- It did not implement RAG2's full production surface: continuation/refiner
  behavior, Cache2 re-entry, finalizer behavior, and the full live-DB supervisor
  contract were not replacement-complete.
- The later prefix-only idea was judged insufficient because cross-capability
  chains still require decomposition, follow-up planning, or an agentic
  retrieval loop. Moving the ambiguity from initializer to subagent does not
  remove the workload.

## Evidence

### Existing RAG2 Real-LLM Baseline

Executed 68 existing RAG2 live tests one pytest node at a time.

- Route/real-conversation/recall/continuation group:
  - Artifact:
    `test_artifacts/rag2_1_existing_rag2_live_baseline/route_recall_continuation/summary.json`
  - Result: 36 total, 33 passed, 3 failed.
- Full supervisor live-DB/finalizer group:
  - Artifact:
    `test_artifacts/rag2_1_existing_rag2_live_baseline/supervisor_live_db/summary.json`
  - Result: 32 total, 31 passed, 1 failed.
- Rerun of the four RAG2 failures:
  - Artifact:
    `test_artifacts/rag2_1_existing_rag2_live_baseline/rerun_failures/summary.json`
  - Result: 4 total, 0 passed, 4 failed again.

RAG2 had stable issues, but it remained much closer to production readiness
than the RAG2.1 experiment.

### RAG2.1 Initializer Comparison

Executed the RAG2.1 experimental initializer against existing RAG2 initializer
and route cases that mapped to the experiment surface.

- Artifact:
  `test_artifacts/rag2_1_existing_initializer_comparison.json`
- Result: 30 total, 14 passed, 12 failed, 4 errors.

The failures were concentrated around route semantics that RAG2 had accumulated
through boundary-specific prompt rules and tests.

### Real-Data Side-By-Side Probe

Executed bounded real-data comparisons against recent conversation history.

- Sample artifact:
  `test_artifacts/rag2_1_real_data_conversation_sample.json`
- Comparison artifacts:
  - `test_artifacts/rag2_1_comparison_case_001.json`
  - `test_artifacts/rag2_1_comparison_case_001_retry.json`
  - `test_artifacts/rag2_1_comparison_case_001_retry2.json`

One successful retry showed both RAG2 and RAG2.1 could find the expected source
for a single exact-phrase case, and RAG2.1 added an authority tag. Earlier runs
were affected by model or embedding endpoint availability. This was not enough
evidence to justify migration.

### Research Conclusion

Industry RAG systems do not generally solve workload complexity by making the
top-level initializer prefix-only. The common pattern is:

```text
query planner/router
-> specialized retriever or knowledge source
-> query rewriting/decomposition/filter generation
-> hybrid retrieval
-> rerank/compress
-> grounded synthesis with citations
```

Relevant sources reviewed:

- Azure AI Search agentic retrieval:
  `https://learn.microsoft.com/et-ee/azure/search/agentic-retrieval-overview`
- OpenAI Retrieval:
  `https://developers.openai.com/api/docs/guides/retrieval`
- OpenAI File Search:
  `https://developers.openai.com/api/docs/guides/tools-file-search`
- LlamaIndex RouterRetriever:
  `https://developers.llamaindex.ai/python/framework-api-reference/retrievers/router/`
- LangChain SelfQueryRetriever:
  `https://reference.langchain.com/python/langchain-classic/retrievers/self_query/base/SelfQueryRetriever`
- GraphRAG:
  `https://arxiv.org/abs/2404.16130`
- RAPTOR:
  `https://arxiv.org/abs/2401.18059`
- Self-RAG:
  `https://arxiv.org/abs/2310.11511`

The research supports specialized retrieval agents and query decomposition, but
not a production migration to the attempted RAG2.1 design.

## Cleanup Decision

Temporary RAG2.1 experiment source and tests must be removed:

- `experiments/rag2_1/`
- `tests/test_experiments_rag2_1_core.py`
- `tests/test_experiments_rag2_1_real_data_compare.py`

The ignored `test_artifacts/` evidence files are intentionally retained so the
failure decision remains auditable.

## Future Direction

Do not revive RAG2.1 as a replacement without a new approved plan and a clearer
research-backed contract.

Preferred next work:

- Fix the four stable RAG2 live-test failures.
- Add authority-envelope ideas directly to RAG2 only if they can be introduced
  without changing the initializer contract.
- Evaluate any future agentic retrieval changes with real-LLM tests and real
  data before production migration.
