# Conversation Evidence ICD

`kazusa_ai_chatbot.rag.conversation_evidence` owns retrieval of factual
evidence from conversation history.

## Public Contract

```python
from kazusa_ai_chatbot.rag.conversation_evidence import ConversationEvidenceAgent

await ConversationEvidenceAgent().run(
    task: str,
    context: dict,
    max_attempts: int = 1,
)
```

The returned shape remains:

```python
{
    "resolved": bool,
    "result": {
        "selected_summary": str,
        "capability": "conversation_evidence",
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

Conversation evidence answers what was said, by whom, when, and in what nearby
message relation. It owns semantic conversation search, exact phrase retrieval,
structured message filters, aggregate message counts/rankings, URL/message
provenance, and active-turn exclusion.

## Non-Ownership

It does not resolve profile state, durable memory, live public facts, recall
state, or persona stance. User metadata predicates belong to person context.
External web facts belong to web evidence or live context. It refuses or reports
missing context when a concrete person reference is required but unavailable.

## Internal Flow

```text
selector -> selected worker -> worker result projection -> capability result
```

Deterministic prefix and coverage checks may choose a worker without an LLM.
Otherwise the selector LLM chooses one conversation worker. Worker payloads stay
trace material; prompt-facing evidence is projected and sanitized before
cognition.

## Module Ownership

| Module | Ownership |
|---|---|
| `agent.py` | Capability wrapper and run orchestration. |
| `selector.py` | Deterministic and LLM worker selection, speaker-scope handoff, and selector context shaping. |
| `contracts.py` | Typed projection contracts and standard capability result helpers. |
| `projection.py` | Worker result to evidence projection, relation packets, coverage buckets, refs, and URL refs. |
| `active_turn_filter.py` | Active-turn conversation-row and platform-message exclusion helpers. |
| `workers/search.py` | Hybrid semantic and literal conversation retrieval. |
| `workers/filter.py` | Structured conversation row filtering. |
| `workers/aggregate.py` | Message count and ranking aggregates. |
| `workers/keyword.py` | Exact literal conversation keyword retrieval. |

## Worker Roster

- `conversation_search_agent`
- `conversation_filter_agent`
- `conversation_aggregate_agent`
- `conversation_keyword_agent`

Worker names, cache names, result payloads, and fact-source metadata remain
unchanged.

## Cache Policy

The top-level capability is not cached and reports
`capability_orchestrator_uncached`. Worker cache policy remains worker-local and
uses Cache 2 dependency invalidation. Open or recent conversation windows remain
uncached when worker policy requires fresh reads.

## LLM And Prompt Policy

Prompt constants, LLM instances, message construction, JSON parsing, retry
behavior, and call ordering are preserved from the flat modules. Stable
instructions stay in `SystemMessage`; runtime task/context stays in
`HumanMessage`.

## Verification

- `py_compile` every changed Python file in this package.
- Static grep for deleted flat conversation module import paths returns no
  production-source matches.
- Prompt stability audit confirms selector and worker prompt payloads are
  unchanged.
- Conversation-focused deterministic tests remain the behavioral authority.
