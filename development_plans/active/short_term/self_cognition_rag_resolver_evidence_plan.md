# Self-Cognition Resolver RAG Evidence Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` or `superpowers:subagent-driven-development`
> when available. In this session, execution is parent-led because the current
> user explicitly requested immediate execution and the current tool contract
> does not expose sub-agent delegation as a safe default.

**Goal:** Let resolver-enabled self-cognition/internal-thought turns use the
existing RAG2 evidence path when, and only when, the cognition core asks for
`rag_evidence`. This closes the current integration gap where L2d can select
evidence lookup but the RAG cognitive episode adapter rejects non-user-message
episodes.

**Architecture:** Preserve the three-layer cognition core. L1, L2, and L2d own
semantic judgment and action direction. Deterministic Python only validates a
selected resolver capability request, projects the already-built cognitive
episode into the existing RAG2 request boundary, executes the existing RAG2
supervisor, and feeds the observation back into the resolver loop.

**Tech Stack:** Python, Kazusa cognitive episode contract, existing RAG2
supervisor, cognition resolver capability contract, L2d action initializer
prompt, pytest, real LLM diagnostic artifacts.

---

## Plan Metadata

- Plan class: medium production integration
- Status: in_progress
- Source branch: `resolver-goal-poc`
- Working branch: `resolver-self-cognition-rag`
- Owner: Codex
- Created: 2026-06-01
- Related plan:
  - `development_plans/active/short_term/cognition_preserving_goal_resolver_production_plan.md`
- Related source:
  - `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`
  - `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
  - `src/kazusa_ai_chatbot/cognition_resolver/loop.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`
  - `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - `src/kazusa_ai_chatbot/self_cognition/projection.py`
  - `src/kazusa_ai_chatbot/rag/README.md`

## Non-Negotiable Rules

- Kazusa remains a cognition core system, not a chatbot shell, coding harness,
  or generic tool runner.
- All semantic thinking and action direction must continue through L1 -> L2 ->
  L2d.
- No deterministic rule may say "if context is insufficient, lookup evidence".
  Cognition may request `rag_evidence`; code may only execute that request.
- Do not create a new RAG. Wire self-cognition evidence lookup into the existing
  RAG2 path.
- Do not tune prompts with lookup-table examples that are shaped to pass tests.
  Prompt edits must be generic and organically linked to the existing capability
  contract.
- Do not expose raw durable ids to an LLM unless the stage needs them. If an LLM
  ever has to choose from multiple internal items, use small ordinal aliases and
  map back deterministically. This plan should not require such selection.
- Do not force visible speech after self-cognition. If L2d produces a coherent
  no-speak decision, preserve it.

## Current Gap

The resolver can already execute `rag_evidence` and feed `rag_result` into the
next cognition cycle:

```text
L2d resolver_capability_requests
  -> execute_resolver_capability_request(...)
  -> run_rag_evidence_for_persona_state(...)
  -> build_text_chat_rag_request(...)
  -> call_rag_supervisor(...)
  -> observation.rag_result
  -> next cognition cycle
```

However, `build_text_chat_rag_request(...)` currently rejects every
`trigger_source` except `user_message`. A self-cognition/internal-thought turn
can therefore select `rag_evidence` in principle, but the capability will fail
before RAG2 sees the request.

The L2d prompt also carries a softer risk: its current internal-thought
guidance makes `self_goal_resolution` visually prominent. It does not form a
hard lookup table, but it should be reorganized so `rag_evidence`,
`self_goal_resolution`, ordinary action selection, future cognition, and empty
action are peer choices based on the actual cognitive gap.

## Target Behavior

For a resolver-enabled `internal_thought` + `internal_monologue` episode:

```text
cognition cycle
  -> L2d decides whether evidence is needed
  -> if L2d selects rag_evidence:
       existing RAG2 receives a normal request with self-cognition context
       resolver records the RAG observation
       next cognition cycle reasons with that evidence
  -> if L2d selects self_goal_resolution, speak, future cognition, or no action:
       resolver preserves that decision when the trace gives a coherent reason
```

Positive real LLM validation means either:

- L2d calls `rag_evidence` for a self-cognition case that reasonably needs
  conversation, memory, relationship, or public evidence; or
- L2d does not call RAG, but the trace gives a reasonable character/cognition
  reason for resolving, deferring, speaking, or staying silent without lookup.

## In Scope

- Extend the existing RAG cognitive episode adapter to support
  `internal_thought` episodes whose `input_sources` contain
  `internal_monologue`.
- Preserve existing user-message RAG request shape and tests.
- Keep the RAG request boundary least-exposure and prompt-safe.
- Reorganize `_ACTION_INITIALIZER_PROMPT` so the internal-thought path is a
  balanced capability choice rather than a self-resolution default.
- Add deterministic tests that prove:
  - user-message behavior remains unchanged;
  - unsupported non-user trigger sources remain rejected;
  - internal-thought RAG requests can be built;
  - the resolver capability path can execute RAG for an internal-thought state
    when the capability request is already selected by cognition.
- Run real LLM diagnostics and write human-readable evidence under
  `test_artifacts/cognition_resolver/`.

## Out of Scope

- New RAG workers, new memory retrieval architecture, or direct DB lookup tools.
- Deterministic context-insufficiency classifiers.
- Broad scheduler, adapter delivery, consolidation, or memory lifecycle changes.
- Prompt examples that mention the validation cases or encode a case-specific
  answer path.
- Mandatory progressive dialog or parallel speak/tool output. That future
  design remains parked as an optional combo-output concept.

## Design Decisions

1. Reuse `rag_evidence`.

   Self-cognition does not get a new `conversation_context_lookup` or
   `memory_lookup` capability. The existing RAG2 initializer and dispatcher own
   the decision to use conversation evidence, memory evidence, person context,
   web evidence, recall, or finalization.

2. Extend the adapter, not the resolver loop.

   The resolver loop already projects successful RAG observations into the next
   cognition cycle. The missing production boundary is the RAG adapter's episode
   support.

3. Keep source-specific projection explicit.

   The adapter should branch on `episode["trigger_source"]`. User-message
   projection continues to use `project_text_chat_compatibility_fields(...)`.
   Internal-thought projection should read `target_scope`, `origin_metadata`,
   turn clock fields, and existing prompt-safe context arguments without
   exposing the full cognitive episode.

4. Keep LLM decisions simple.

   L2d sees capability names and natural-language context, not internal ids or
   handler details. It only decides whether a natural-language evidence
   objective is needed. Python executes the selected request.

5. Evaluate with behavior, not tool worship.

   A RAG call is useful evidence of capability coverage, but not a mandatory
   pass condition. A no-RAG trace can pass if it shows the cognition core made a
   coherent judgment from available context.

## Implementation Plan

- [x] Add deterministic adapter coverage for internal-thought RAG request
      projection.
- [x] Implement internal-thought support in
      `rag/cognitive_episode_adapter.py`.
- [x] Add deterministic resolver capability coverage proving an
      internal-thought state can execute selected `rag_evidence` through the
      existing RAG path.
- [x] Reorganize `_ACTION_INITIALIZER_PROMPT` around peer choices for
      internal-thought episodes without adding case-specific examples.
- [x] Run prompt contract, adapter, and resolver deterministic tests.
- [x] Run real LLM diagnostic cases and inspect every LLM-relevant output for
      whether RAG was selected or reasonably not selected.
- [x] Write human-readable real LLM review artifact.
- [x] If the real LLM result is positive, commit this branch and merge it into
      `resolver-goal-poc`.

## Verification Commands

```powershell
venv\Scripts\python -m py_compile `
  src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py `
  tests\test_rag_cognitive_episode_adapter.py `
  tests\test_cognition_resolver_loop.py

venv\Scripts\python -m pytest `
  tests\test_rag_cognitive_episode_adapter.py `
  tests\test_cognition_resolver_loop.py `
  tests\test_cognition_prompt_contract_text.py `
  -q
```

Real LLM diagnostics must be run one case at a time when possible, with output
inspected before claiming pass.

## Independent Review Checklist

- [ ] No deterministic semantic lookup rule was introduced.
- [ ] No new RAG or DB search path was introduced.
- [ ] Prompt changes remain generic and do not mention validation cases.
- [ ] User-message RAG request shape is unchanged.
- [ ] Internal-thought RAG request exposes only fields RAG2 needs.
- [ ] No LLM stage is asked to copy UUIDs or durable raw ids.
- [ ] Real LLM evidence is human-readable and includes the selected resolver
      path, key L2d output, and final judgment.

## Execution Evidence

Plan review completed before implementation:

- no deterministic semantic lookup rule in the plan;
- no new RAG or direct DB search path in the plan;
- no prompt lookup table or validation-case mapping in the plan;
- real LLM acceptance checks behavior, not mandatory tool usage.

Deterministic verification:

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py tests\test_rag_cognitive_episode_adapter.py tests\test_cognition_resolver_loop.py tests\test_cognition_prompt_contract_text.py`
  passed.
- `venv\Scripts\python -m pytest tests\test_rag_cognitive_episode_adapter.py tests\test_cognition_resolver_loop.py tests\test_cognition_prompt_contract_text.py tests\test_persona_supervisor2_action_initializer.py -q`
  passed with 74 tests.

Real LLM diagnostic:

- Command:
  `venv\Scripts\python test_artifacts\cognition_resolver\real_db_comparison_20260601\diagnose_l2d_self_cognition.py`
- Raw output:
  `test_artifacts/cognition_resolver/real_db_comparison_20260601/self_cognition_l2d_diagnostics.json`
- Human-readable review:
  `development_plans/reference/evidence/self_cognition_rag_resolver_evidence_review_20260601.md`

Results:

- `R04_self_cognition_price_topic` selected `rag_evidence` from
  `internal_thought`; RAG2 dispatched conversation and person evidence workers.
  The local model later attempted an equivalent repeated lookup after
  inconclusive evidence; the resolver blocked that duplicate privately for the
  non-user source.
- `R05_self_cognition_photo_topic` did not call RAG, but selected
  `self_goal_resolution` and then private `trigger_future_cognition` with the
  coherent reason that no photo evidence had appeared and direct speech would
  be premature.

The real LLM result is positive under the agreed criterion: at least one
self-cognition case called RAG for evidence, and the no-RAG case opted out with
a reasonable cognition-owned explanation.

Merge evidence:

- Feature branch commit: `ee9da4f Wire self cognition resolver RAG evidence`
- Merge commit on `resolver-goal-poc`:
  `43c97c1 Merge self cognition resolver RAG evidence`
