# Live Context ICD

`kazusa_ai_chatbot.rag.live_context` owns current-time and live external fact
resolution.

## Public Contract

```python
from kazusa_ai_chatbot.rag.live_context import LiveContextAgent

await LiveContextAgent().run(
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
        "capability": "live_context",
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

Live context answers supported runtime local date/time facts directly and
resolves target/scope for changing external facts such as weather, temperature,
opening status, schedules, prices, exchange rates, and current public state.
External live facts delegate to `web_agent3` after target resolution.

## Non-Ownership

It does not treat memory as the source of live values. Memory and conversation
workers may only help resolve stable target or scope. If target, location, or
scope is missing, the capability reports missing context instead of guessing.

## Internal Flow

```text
fact-type detection -> runtime answer or target resolution -> web delegation
```

Deterministic runtime facts avoid an LLM when the answer is already present in
runtime context. External live facts use the selector and existing workers to
resolve target/scope before web evidence retrieval.

## Module Ownership

| Module | Ownership |
|---|---|
| `agent.py` | Capability wrapper, result shaping, and run orchestration. |
| `selector.py` | Deterministic and LLM live-plan selection for runtime and external live facts. |
| `runtime_facts.py` | Runtime date, time, weekday, and selected-summary helpers. |
| `target_resolution.py` | Target cleaning, target marker extraction, worker text extraction, URL refs, location refs, and web task shaping. |

## Worker Roster

- `runtime_context_provider`
- `conversation_search_agent`
- `persistent_memory_search_agent`
- `web_agent3`

Worker names, cache names, result payloads, and fact-source metadata remain
unchanged.

## Cache Policy

The top-level capability is not cached and reports
`capability_orchestrator_uncached`. Runtime facts are computed from current
context. Delegated workers keep their existing cache policies.

## LLM And Prompt Policy

Prompt constants, LLM instances, message construction, JSON parsing, retry
behavior, and call ordering are preserved from the flat module. Stable
instructions stay in `SystemMessage`; runtime task/context stays in
`HumanMessage`.

## Verification

- `py_compile` every changed Python file in this package.
- Static grep for the deleted flat live-context module import path returns no
  production-source matches.
- Prompt stability audit confirms selector prompt payloads are unchanged.
- Live-context focused deterministic tests remain the behavioral authority.
