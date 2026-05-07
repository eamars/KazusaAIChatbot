# rag2 phase4 continuation plan

## Summary

- Goal: Add bounded Phase 4 continuation to RAG2 through a binary refiner that can produce a self-contained refined query for the existing initializer.
- Plan class: large
- Status: completed
- Mandatory skills: `local-llm-architecture`, `py-style`, `test-style-and-execution`, `cjk-safety`
- Overall cutover strategy: keep the public RAG2 contract unchanged while adding a trace-only continuation layer above the existing initializer, dispatcher, specialist agents, and Cache2.
- Highest-risk areas: bypassing Cache2, moving slot ownership out of the initializer, leaking rejected intermediate material as evidence, or creating an unbounded live-response loop.
- Acceptance criteria: continuation is binary; `reason` is diagnostic only; `refined_query` is self-contained; Cache2 remains active on refined-query re-entry; irrelevant first-pass material or missing user constraints stop; real LLM and real DB tests pass.

## Approved Architecture

RAG2 continuation is a thin layer over existing infrastructure:

```text
rag_initializer(original_query)        # Cache2 active
  -> dispatcher/executor
  -> evaluator sees first-pass unresolved result material
  -> continuation_refiner:
       should_continue=false -> finalize
       should_continue=true, refined_query="self-contained improved input"
  -> rag_initializer(refined_query)    # Cache2 active, different query key
  -> dispatcher/executor
  -> evaluator/finalizer
```

The continuation refiner owns only this output:

```python
{
    "should_continue": bool,
    "refined_query": str,
    "reason": str,
}
```

Ownership boundaries:

- The continuation refiner decides only whether the first-pass result justifies another bounded pass.
- The refined query must read like a natural-language user query, not a slot or worker route.
- The existing initializer owns all slot generation for the second pass.
- The existing dispatcher owns all routing.
- Cache2 remains active because the second initializer call uses the refined query as normal initializer input.
- `reason` is trace/log metadata only and is never used for control flow or fed to the initializer.
- First-pass intermediate material and rejected rows stay out of public evidence and may appear only in supervisor trace.

## Public Contract

`call_rag_supervisor(...)` remains unchanged:

```python
{
    "answer": str,
    "known_facts": list[dict],
    "unknown_slots": list[str],
    "loop_count": int,
}
```

`project_known_facts(...)` keeps the same public evidence keys. Continuation metadata, when present, is projected only under:

```text
rag_result.supervisor_trace.dispatched[*].continuation
```

The trace shape is:

```python
{
    "should_continue": bool,
    "refined_query": str,
    "reason": str,
}
```

## Scope

Modified areas:

- `src/kazusa_ai_chatbot/rag/continuation.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
- `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/README.md`
- focused deterministic and live tests

Unchanged areas:

- Cache2 policy/runtime/persistence files
- cognition, dialog, consolidation, scheduler, adapters, and database schemas

## Verification Evidence

Syntax check passed:

```powershell
venv\Scripts\python.exe -m py_compile `
  src\kazusa_ai_chatbot\rag\continuation.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_supervisor2.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_projection.py `
  src\kazusa_ai_chatbot\rag\memory_evidence_agent.py `
  tests\test_rag_continuation.py `
  tests\test_rag_phase3_supervisor_integration.py `
  tests\test_persona_supervisor2_rag2_integration.py `
  tests\test_rag_projection.py `
  tests\test_rag_phase4_continuation_live_llm.py
```

Deterministic focused regression passed:

```powershell
venv\Scripts\python.exe -m pytest `
  tests\test_rag_continuation.py `
  tests\test_rag_phase3_supervisor_integration.py `
  tests\test_persona_supervisor2_rag2_integration.py `
  tests\test_rag_projection.py `
  tests\test_rag_phase3_capability_agents.py -q
```

Result: `77 passed`.

Real LLM tests were run one at a time and inspected:

- `test_live_refiner_continues_from_stale_memory_direction`
- `test_live_refiner_stops_on_missing_user_constraints`
- live refiner stop case for irrelevant first-pass result material

Real LLM + real DB tests were run one at a time and inspected:

- `test_live_db_memory_byproduct_produces_refined_query`
- `test_live_db_supervisor_reenters_initializer_with_cache2_active`

Temporary live DB memory cleanup was verified:

```text
temporary_rag2_phase4_memory_rows=0
```
