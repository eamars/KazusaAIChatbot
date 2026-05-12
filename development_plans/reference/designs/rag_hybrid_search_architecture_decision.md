# RAG Hybrid Search Architecture Decision

## Status

Decision-only reference. Do not execute this document as a development plan.

## Date

2026-05-12

## Context

The RAG failure investigation for QQ group `905393941` evaluated whether the
current conversation hybrid search should be replaced with an indexed lexical
search design using Atlas Search BM25, reciprocal-rank fusion, strict lexical
gating, and evidence-role gating.

The initializer is out of scope for this decision. Future retrieval changes
must stay inside database search agents unless a separate approved plan expands
that boundary.

## Evaluation Summary

The fair comparison used copied production `conversation_history` rows, not
synthetic-only rows. The copied Atlas Search experiment populated isolated
collection `rag_hybrid_bm25_experiment_real_copy` with 479 matching production
rows and built Atlas Search index `rag_hybrid_bm25_real_copy_text_index`.

Broad 8-case real fixture:

| Method | Result | FP/FN | Avg expected rank | Avg latency | Avg judge prompt chars |
|---|---:|---:|---:|---:|---:|
| Current `hybrid` | 8/8 | 0/0 | 1.00 | 89.6 ms | 3982 |
| Raw Atlas BM25 real-copy | 5/8 | 3/0 | 5.40 | 19.0 ms | 7737 |
| Atlas BM25 strict gate | 8/8 | 0/0 | 1.80 | 19.0 ms | 973 |
| RRF Atlas strict plus role gate | 8/8 | 0/0 | 1.40 | 126.4 ms | 2639 |

Failure-mode 8-case real fixture:

| Method | Result | FP/FN | Avg expected rank | Avg latency | Avg judge prompt chars |
|---|---:|---:|---:|---:|---:|
| Current `hybrid` | 7/8 | 0/1 | 5.00 | 54.5 ms | 3318 |
| Raw Atlas BM25 real-copy | 4/8 | 2/2 | 3.00 | 21.5 ms | 7982 |
| Atlas BM25 strict gate | 5/8 | 0/3 | 1.00 | 21.5 ms | 744 |
| RRF Atlas strict plus role gate | 7/8 | 0/1 | 2.40 | 122.2 ms | 1797 |

The earlier isolated synthetic Atlas result used only 4 synthetic cases. Treat
that result as a mechanism check only; it is not a valid comparison against the
8-case real QQ fixtures.

## Decision

Keep the current production hybrid search as the default. Do not replace it
with raw BM25, RRF alone, local lexical scanning, or Atlas BM25 plus gating
based on the current evidence.

The current hybrid remains the better overall production choice because it
matches or beats the candidate designs on pass/fail accuracy while keeping
lower measured latency and lower implementation complexity. The Atlas/RRF
prototype improves prompt pressure and expected rank on the harder fixture, but
the measured sequential latency is worse and the improvement is not sufficient
to justify production replacement.

## Recommended Future Action

No production action is required now.

If this area is revisited, create a new active plan with these constraints:

- Keep initializer behavior and slot contracts unchanged.
- Limit changes to database search agents and shared retrieval helpers.
- Use real production-row copies or production shadow mode, not synthetic-only
  fixtures, for acceptance metrics.
- Run vector and Atlas BM25 branches in parallel before judging latency.
- Compare against current `hybrid` on the same broad and failure-mode real
  fixtures.
- Require equal or better pass/fail accuracy, no false-positive regression,
  lower or acceptable p95 latency, and lower LLM prompt pressure before any
  rollout.
- Keep local BM25 scanning as an experiment-only diagnostic; do not put it on
  the live path.
- Use evidence-role gating only after retrieval has strict lexical or semantic
  support; role-only gating is not sufficient for absent-topic negatives.

## Non-Goals

- No initializer redesign.
- No production Atlas Search index rollout.
- No database migration.
- No replacement of the current production hybrid search.
