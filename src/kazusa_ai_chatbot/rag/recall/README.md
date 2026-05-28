# Recall ICD

`kazusa_ai_chatbot.rag.recall` owns scoped retrieval of active agreements,
commitments, plans, open loops, and current-episode state.

## Public Contract

```python
from kazusa_ai_chatbot.rag.recall import RecallAgent

await RecallAgent().run(
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
        "capability": "recall",
        "primary_worker": "recall_agent",
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

Recall reconciles volatile current-episode progress, active commitments,
pending scheduled events, and gated history proof for questions about active
agreements, ongoing promises, current plans, open loops, and where the current
episode stands.

## Non-Ownership

Recall does not create durable memory, search broad conversation history for
ordinary message evidence, resolve profiles, or decide persona stance. Durable
knowledge writes remain consolidation work after the live response path.

## Internal Flow

```text
mode selection -> collectors -> candidate ranking -> optional review -> result
```

Deterministic mode selection and candidate ranking handle the common path. The
review LLM is used only for slots that need candidate review under the existing
contract.

## Module Ownership

| Module | Ownership |
|---|---|
| `agent.py` | Recall wrapper and run orchestration. |
| `contracts.py` | Recall context requirements, candidate/result helpers, volatile cache status, source ordering, and mode selection. |
| `review.py` | Candidate ranking, conflict notes, recall type/freshness projection, observation projection, and LLM candidate review. |
| `collectors/progress.py` | Current-episode progress collector and progress-entry extraction. |
| `collectors/commitments.py` | Active-commitment collector. |
| `collectors/scheduled_events.py` | Scheduled-event collector. |
| `collectors/history.py` | History-proof collector. |

## Collector Roster

- `ProgressCollector`
- `ActiveCommitmentCollector`
- `ScheduledEventCollector`
- `HistoryEvidenceCollector`

Collector names, result payloads, and fact-source metadata remain unchanged.

## Cache Policy

Recall is volatile and not cached. It reports the existing volatile cache status
and reads only scoped runtime/progress/scheduler/history material.

## LLM And Prompt Policy

Prompt constants, LLM instances, message construction, JSON parsing, retry
behavior, and call ordering are preserved from the flat module. Stable
instructions stay in `SystemMessage`; runtime candidates and task context stay
in `HumanMessage`.

## Verification

- `py_compile` every changed Python file in this package.
- Static grep for the deleted flat recall module import path returns no
  production-source matches.
- Prompt stability audit confirms review prompt payloads are unchanged.
- Recall-focused deterministic tests remain the behavioral authority.
