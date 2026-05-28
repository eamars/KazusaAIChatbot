# Memory Evidence ICD

`kazusa_ai_chatbot.rag.memory_evidence` owns durable memory evidence retrieval.

## Public Contract

```python
from kazusa_ai_chatbot.rag.memory_evidence import MemoryEvidenceAgent

await MemoryEvidenceAgent().run(
    task: str,
    context: dict,
    max_attempts: int = 1,
)
```

The returned shape remains the standard capability result:

```python
{
    "resolved": bool,
    "result": {
        "selected_summary": str,
        "capability": "memory_evidence",
        "primary_worker": str,
        "supporting_workers": list[str],
        "source_policy": str,
        "resolved_refs": list[dict],
        "projection_payload": dict,
        "worker_payloads": dict,
        "evidence": list[str],
        "missing_context": list[str],
        "conflicts": list[str],
        "observation_candidates": list[dict],
        "source_hints": list[dict],
    },
    "attempts": int,
    "cache": dict,
}
```

## Semantic Ownership

Memory evidence retrieves shared durable memory and scoped current-user
continuity. It owns hybrid persistent-memory search, exact durable-memory
anchors, and `user_memory_units` evidence for current-user recognition,
preferences, user-specific lore, and prior shared interactions.

## Non-Ownership

It does not search conversation rows, resolve user identities, rank relationship
state, read live external facts, or decide persona stance. It keeps scoped
current-user continuity distinct from shared/global memory and reports missing
context when scoped retrieval lacks the required current user.

## Internal Flow

```text
selector -> selected memory worker -> evidence projection -> capability result
```

Deterministic routing handles clear scoped-memory and persistent-memory tasks.
The selector LLM chooses between memory workers only when deterministic routing
does not resolve the worker.

## Module Ownership

| Module | Ownership |
|---|---|
| `agent.py` | Capability wrapper and run orchestration. |
| `selector.py` | Deterministic and LLM worker selection for persistent and scoped memory evidence. |
| `contracts.py` | Memory evidence result helpers, coverage buckets, and uncached capability contract. |
| `projection.py` | Worker result to memory rows, summaries, refs, observation candidates, and source hints. |
| `workers/persistent_search.py` | Hybrid durable-memory search. |
| `workers/persistent_keyword.py` | Exact durable-memory keyword retrieval. |
| `workers/user_memory.py` | Scoped current-user continuity retrieval. |

## Worker Roster

- `persistent_memory_search_agent`
- `persistent_memory_keyword_agent`
- `user_memory_evidence_agent`

Worker names, cache names, result payloads, and fact-source metadata remain
unchanged.

## Cache Policy

The top-level capability is not cached and reports
`capability_orchestrator_uncached`. Worker cache policy remains worker-local and
uses Cache 2 dependency invalidation. Scoped user-memory retrieval keeps its
existing source and authority metadata.

## LLM And Prompt Policy

Prompt constants, LLM instances, message construction, JSON parsing, retry
behavior, and call ordering are preserved from the flat modules. Stable
instructions stay in `SystemMessage`; runtime task/context stays in
`HumanMessage`.

## Verification

- `py_compile` every changed Python file in this package.
- Static grep for deleted flat memory module import paths returns no
  production-source matches.
- Prompt stability audit confirms selector and worker prompt payloads are
  unchanged.
- Memory-focused deterministic tests remain the behavioral authority.
