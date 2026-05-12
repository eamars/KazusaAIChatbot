# RAG hybrid search experiment

This folder contains a read-only experiment for comparing conversation
retrieval methods against real MongoDB conversation history.

It does not change production RAG behavior. The reusable logic lives in
`hybrid_search.py`; `run.py` is only the CLI wrapper.

## Purpose

This experiment supports the bugfix plan at
`development_plans/active/bugfix/rag_hybrid_search_time_config_plan.md`.
It exists to separate embedding quality from retrieval orchestration,
projection, top-k, threshold, and time-window failures.

The current conclusion from the QQ `905393941` investigation is:

- semantic retrieval can find the GPU answer row;
- keyword-only retrieval can resolve against the later denial/meta cluster;
- top-k must be paired with selected evidence limits;
- attachment descriptions must be projected, not only embedded;
- a semantic-only floor below `0.70` admits false positives in this data;
- production and the experiment now default to the shared configured floor,
  currently `RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR=0.72`;
- link-style literal-trap negatives should use URL/domain anchors such as
  `xhslink`, not broad site-name anchors such as `小红书`.

Run from the repository root:

```powershell
venv\Scripts\python.exe -m experiments.rag_hybrid_search.run `
  --output test_artifacts/rag_hybrid_search_experiment.json `
  --report test_artifacts/rag_hybrid_search_experiment.md
```

Run the failure-mode fixture:

```powershell
venv\Scripts\python.exe -m experiments.rag_hybrid_search.run `
  --cases experiments/rag_hybrid_search/failure_mode_cases.json `
  --output test_artifacts/rag_hybrid_search_failure_modes.json `
  --report test_artifacts/rag_hybrid_search_failure_modes.md
```

Run threshold diagnostics:

```powershell
venv\Scripts\python.exe -m experiments.rag_hybrid_search.run `
  --cases experiments/rag_hybrid_search/failure_mode_cases.json `
  --semantic-only-floor 0.65 `
  --output test_artifacts/rag_hybrid_search_failure_modes_floor065.json `
  --report test_artifacts/rag_hybrid_search_failure_modes_floor065.md

venv\Scripts\python.exe -m experiments.rag_hybrid_search.run `
  --cases experiments/rag_hybrid_search/failure_mode_cases.json `
  --semantic-only-floor 0.72 `
  --output test_artifacts/rag_hybrid_search_failure_modes_floor072.json `
  --report test_artifacts/rag_hybrid_search_failure_modes_floor072.md
```

The experiment evaluates:

- semantic vector retrieval;
- keyword retrieval over explicit anchors;
- hybrid retrieval, which merges semantic and keyword candidates and expands
  bounded neighboring message context around high-priority hits.

Hybrid ranking prefers rows supported by both semantic and keyword retrieval,
then keyword-only rows, then local neighbor context, then semantic-only rows.
When no keyword anchor returns anything, semantic-only rows must clear the
configurable `--semantic-only-floor` before they can become hybrid evidence.

The default `cases.json` fixture is a broad signal profile. The
`failure_mode_cases.json` fixture intentionally stresses query-visible
anchors, wrong time windows, selected-limit loss, attachment-only rows, and
literal-trap negatives.

Post-implementation reference artifacts:

- `test_artifacts/rag_hybrid_search_experiment_post_impl_v2.json`: hybrid
  `8/8`, false positives `0`, false negatives `0`.
- `test_artifacts/rag_hybrid_search_failure_modes_post_impl_v2.json`: hybrid
  `7/8`, false positives `0`, false negatives `1`; the remaining miss is the
  intentionally wrong UTC-day window diagnostic.

## Fixture Expectations

`cases.json` is a broad signal benchmark and contains some hand-picked anchors.
It should not be treated as full production proof because production still
depends on an LLM-generated query and anchor contract.

`failure_mode_cases.json` is the stricter RCA fixture. It is expected to catch:

- broad anchor misses, such as `显卡` at low top-k;
- synonym mismatch, such as `市场占有率` versus `市场份额`;
- wrong UTC-day windows for local `昨天`;
- selected-limit projection loss;
- attachment-only evidence loss;
- literal-trap false positives such as a 小红书-like screenshot for a
  小红书-link query.

Artifacts are written under `test_artifacts/` and are not production state.
