# rag2 cognition identity evidence content bugfix plan

## Summary

- Goal: stop cognition-facing RAG evidence content from using raw user,
  message, memory, or source IDs as evidence targets; replace them with
  display names, semantic roles, or omission while preserving the existing
  returned `rag_result` shape.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: bigbang for cognition-facing evidence wording;
  compatible for `rag_result` top-level keys and internal diagnostic metadata.
- Highest-risk areas: stripping legitimate user-quoted identifiers,
  accidentally changing consolidation metadata, hiding useful negative
  evidence, and letting continuation promotion reintroduce source IDs.
- Acceptance criteria: no raw ID appears as a speaker, owner, source, or target
  inside cognition-facing RAG evidence text; cognition still receives the same
  public `rag_result` fields; trace and consolidation metadata keep IDs where
  the machine needs them.

## Context

The user-reported defect is content-level, not shape-level:

```text
Conversation evidence and memory evidence return IDs instead of user names.
This can confuse cognition regarding its target.
```

The direct answer established during RCA is:

```text
Returning an ID to cognition is not useful. It has value only for internal
traceability, dedupe, database lookup, or consolidation. Cognition needs the
semantic subject: current user, active character, display name, or third party.
```

Evidence from the current repository:

- `src/kazusa_ai_chatbot/rag/README.md` states that prompt-facing evidence
  must not expose storage IDs, source rows, raw refs, adapter syntax, or raw
  UTC timestamps. Source IDs belong in `supervisor_trace` or helper payloads,
  not primary evidence consumed by cognition.
- `src/kazusa_ai_chatbot/nodes/README.md` states the boundary: RAG returns
  evidence; cognition owns interpretation. The same README says prompt inputs
  should be semantic descriptors instead of raw database internals.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` treats
  `rag_result.answer` as the highest-priority direct retrieval conclusion when
  it answers the current input. Therefore raw IDs in `answer` or evidence text
  become high-authority facts, not harmless metadata.
- `tests/test_rag_projection.py` already expects formatted conversation
  evidence to use display names and local time while stripping row/message IDs.
  That test proves the intended contract is human-readable evidence, not raw
  database identity.
- The reported trace showed promoted RAG evidence containing strings like
  `ID: 1445207392`, followed by `rag_result.answer` repeating those IDs into
  cognition. That is the exact failure mode this plan fixes.

The active broader draft
`development_plans/active/bugfix/rag2_public_output_contract_leak_bugfix_plan.md`
targets public RAG process wording in general. This plan is narrower: it fixes
raw identity/source ID content in cognition-facing evidence without changing
the returned data structure.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing,
  archiving, or signing off this plan.
- `local-llm-architecture`: load before changing RAG prompts, evidence
  projection, cognition prompt projection, or LLM-to-LLM evidence wording.
- `debug-llm`: load before running live/readable RAG or cognition evidence
  checks, comparing LLM outputs, or writing review artifacts.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files containing CJK prompt or test
  strings.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute production changes while this plan status is `draft`.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, and plan edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Preserve the existing `rag_result` top-level shape and evidence collection
  types.
- Do not add a new RAG result schema, a split cognition payload, a feature
  flag, compatibility shim, alternate RAG path, retry loop, or additional LLM
  call.
- Do not change retrieval ranking, vector/BM25 search, Cache2, database schema,
  adapters, dialog rendering, scheduler, or persistence.
- Do not make cognition responsible for ignoring bad RAG evidence. RAG and the
  cognition prompt projection boundary must provide prompt-safe semantic
  content.
- Prompt-facing generated evidence must be short enough to fit as a plain
  string in LLM-to-LLM communication.
- Raw IDs may remain in trace, raw helper payloads, source refs, and
  consolidation candidates when those fields are machine-owned.
- Raw IDs must not appear as evidence subjects, speakers, owners, source
  labels, or answer targets in prompt-facing evidence text.
- If an identifier is the actual user-asked fact, preserve it only as quoted or
  explicitly described source content, not as a provenance label.
- If a display name is unavailable, use a semantic role such as `current user`,
  `active character`, `unknown speaker`, or omit the owner phrase. Do not fall
  back to a raw ID in cognition-facing wording.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Add focused tests proving raw IDs are not useful cognition evidence and must
  not appear in prompt-facing RAG content.
- Cover the originally reported path: continuation-promoted conversation
  evidence whose candidate/source material contains bare `ID: <number>`.
- Cover memory evidence where scoped current-user memory metadata contains
  `scope_global_user_id`; cognition-facing summary/content must name the
  current user or display name instead of treating the ID as the target.
- Preserve existing public `rag_result` keys:
  `answer`, `user_image`, `user_memory_unit_candidates`, `character_image`,
  `third_party_profiles`, `memory_evidence`, `recall_evidence`,
  `conversation_evidence`, `external_evidence`, and `supervisor_trace`.
- Keep internal IDs in `supervisor_trace`, raw helper payloads, and
  consolidation candidates where they are needed for machine ownership.
- Tighten generated RAG content so `promotion_summary`, finalizer answer,
  conversation evidence conclusions, and memory evidence summaries use
  semantic target labels instead of source IDs.
- Ensure the existing sanitizer catches provenance-style bare ID labels such as
  `ID: 1445207392` and `ID：1445207392` when they occur in generated public
  evidence text.
- Ensure sanitizer behavior is line/local enough that it does not destroy
  quoted source text where a user intentionally discussed an identifier.
- Update RAG documentation to state the content rule explicitly:
  IDs are never evidence targets for cognition.
- Produce one readable debug-LLM or saved-trace validation artifact showing the
  reported failure class before final sign-off.

## Deferred

- Do not redesign `rag_result`.
- Do not split `rag_result` into separate cognition/debug/consolidation
  payloads in this plan.
- Do not remove `supervisor_trace`.
- Do not remove `user_memory_unit_candidates`.
- Do not change memory consolidation ownership metadata.
- Do not tune retrieval quality, search recall, ranking, Cache2, model routing,
  or slot selection.
- Do not broaden this into a full cleanup of all RAG process wording except
  where process/source wording causes raw ID leakage.
- Do not add broad keyword filtering over user input.

## Cutover Policy

Overall strategy: bigbang for cognition-facing ID wording; compatible for
returned data shape and machine-owned metadata.

| Area | Policy | Instruction |
|---|---|---|
| `rag_result` top-level shape | compatible | Preserve existing keys and collection types. |
| Generated `rag_result.answer` content | bigbang | Replace raw source/user/message IDs with display names, semantic roles, or omission. |
| `conversation_evidence` text | bigbang | Speaker/source wording must use display name, semantic role, and local time when available; no raw source ID labels. |
| `memory_evidence` summary/content text | bigbang | Owner/subject wording must use current user, display name, active character, or third-party name; no scoped ID as target text. |
| `supervisor_trace` and raw helper payloads | compatible | Preserve IDs and source refs for diagnostics, but keep them out of cognition-facing evidence content. |
| Consolidation candidates | compatible | Preserve IDs needed for merge/evolve ownership. |
| Tests | bigbang | Add or update focused tests so old ID-leaking content fails. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- For bigbang areas, rewrite old public wording directly instead of preserving
  legacy ID text.
- For compatible areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

The same `rag_result` shape remains valid.

Bad cognition-facing content:

```text
Conversation evidence: ID: 1445207392 said ...
Memory evidence: scope_global_user_id=user-1 ...
```

Good cognition-facing content:

```text
Conversation evidence: Current user Sublime previously asked about 杏山千纱.
杏山千纱 replied in a guarded tone. This is supporting behavioral evidence,
not direct evidence of her current state.
```

Good memory evidence content:

```text
Memory evidence: This memory is about the current user Sublime.
```

If the source identity cannot be resolved:

```text
Conversation evidence: An unresolved speaker made a nearby comment, but the
speaker identity is not confirmed. Do not treat this as direct evidence about
the current user.
```

Raw IDs may still appear only in machine-owned trace/candidate fields that are
not used as prompt-facing evidence text.

## Design Decisions

- The fix belongs at RAG evidence content production and prompt-facing
  projection boundaries, not in cognition reasoning rules.
- The existing shape remains because the user explicitly rejected a data
  structure redesign.
- ID values are not facts for cognition. They are operational references.
- Display names are preferred when available. Stable semantic roles are the
  fallback because they preserve target meaning even when a display name is
  unavailable.
- Negative evidence remains useful. If RAG cannot confirm a target because the
  only matches are wrong-source or unresolved-ID evidence, say that directly
  without printing the ID.
- Sanitization must target generated provenance/source labels, not raw quoted
  source content globally.

## Contracts And Data Shapes

The public `rag_result` shape remains:

```python
{
    "answer": str,
    "user_image": dict,
    "user_memory_unit_candidates": list[dict],
    "character_image": dict,
    "third_party_profiles": list[str],
    "memory_evidence": list[dict],
    "recall_evidence": list[dict],
    "conversation_evidence": list[str],
    "external_evidence": list[dict],
    "supervisor_trace": dict,
}
```

Prompt-facing content contract:

```text
Allowed:
- display names;
- current user / active character / third-party semantic roles;
- local readable time;
- short source quotes;
- concise uncertainty and source-bound caveats.

Forbidden:
- raw user IDs as speaker or owner labels;
- raw message IDs as source labels;
- raw memory IDs as evidence targets;
- raw source IDs inside generated summaries;
- provenance labels such as "ID: 1445207392" or "ID：1445207392";
- backend fields presented as evidence text.
```

Machine-owned metadata contract:

```text
Allowed outside cognition-facing evidence text:
- source refs;
- platform_message_id;
- scope_global_user_id;
- memory unit ids;
- raw helper payload ids;
- trace dispatch ids.
```

## LLM Call And Context Budget

- Do not add LLM calls.
- Do not make prompts longer with broad schema explanations.
- Finalizer and continuation prompts may be shortened or clarified to require
  semantic identity labels.
- Generated evidence summaries must remain compact enough to be read as plain
  strings by a local/weaker LLM.

Smallest current contract:

```text
Semantic question:
Identify the evidence target cognition should reason about.

Inputs required:
Existing retrieved rows, display_name/current_user/active_character context,
existing fact summaries, and existing trace metadata.

Output fields required downstream:
The same `rag_result` fields with prompt-facing content strings that name
semantic targets instead of IDs.

Deterministic owners:
ID-to-display-name/role mapping, sanitizer recovery, trace preservation,
metadata preservation, and test enforcement.

Rejected complexity:
New payload schema, new agent, new LLM call, cognition ignore-rules, broad
source quote scrubbing, or retrieval redesign.
```

## Change Surface

Expected production files:

- `src/kazusa_ai_chatbot/rag/evidence_formatting.py`
  - Extend prompt-facing sanitizer/recovery to catch provenance-style bare ID
    labels in generated public evidence text.
  - Keep URL/source-quote behavior narrow and recoverable.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
  - Tighten continuation assessor and finalizer wording so
    `promotion_summary` and `final_answer` do not use source IDs as evidence.
  - Ensure promoted summaries become direct semantic fact sentences.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
  - Ensure conversation and memory evidence text uses display name or semantic
    role when available.
  - Preserve trace/source refs and consolidation candidates.
- `src/kazusa_ai_chatbot/rag/README.md`
  - Document the identity content rule.

Expected test files:

- `tests/test_rag_projection.py`
  - Add projection tests for conversation and memory evidence ID-to-name/role
    content.
  - Keep the existing public-key-shape test passing.
- `tests/test_rag_finalizer_time_context.py`
  - Add continuation/finalizer tests proving promoted summaries and final
    answers do not leak `ID: <number>` as a source label.
- `tests/test_cognition_live_llm_prompt_contracts.py`
  - Update or add prompt-contract coverage proving cognition-facing RAG
    evidence contains semantic identity text, not raw IDs.

Allowed plan files:

- `development_plans/active/bugfix/rag2_cognition_identity_evidence_content_bugfix_plan.md`
- `development_plans/README.md`

Allowed readable validation artifacts:

- One report under an existing `experiments/.../reports/` directory, created
  only during approved execution.

## Overdesign Guardrail

- Do not solve excessive RAG volume broadly in this plan.
- Do not introduce a new data structure.
- Do not split RAG into separate cognition/debug/consolidation outputs.
- Do not add a generalized semantic evidence abstraction.
- Do not make downstream cognition compensate for raw IDs.
- Do not remove machine-owned IDs where non-LLM systems need them.
- Do not sanitize every numeric token. Only source/provenance ID labels and
  raw ID metadata rendered as evidence text are in scope.

## Agent Autonomy Boundaries

- The parent agent owns plan approval state, focused test contract,
  verification, readable validation, independent review, and final sign-off.
- The production-code subagent owns changes only inside the listed production
  files.
- The review subagent owns independent code review only and must not implement
  fixes.
- The execution agent may add small local helper functions only inside the
  approved files when they reduce duplication and preserve this plan's
  contract.
- The execution agent must stop for user approval before changing any
  unlisted production file, cognition prompt behavior beyond prompt-facing RAG
  projection, data schema, database ownership, adapters, or persistence.

## Implementation Order

1. Establish deterministic failing tests for the reported ID leak.
   - Add a `tests/test_rag_projection.py` case where conversation evidence
     source material contains `ID: 1445207392`; expected public evidence
     contains the display name or semantic role and not the ID.
   - Add a memory evidence case where scoped current-user memory has
     `scope_global_user_id`; expected summary/content says current user or
     display name and does not present the ID as evidence content.
2. Establish finalizer/continuation failing tests.
   - Add a `tests/test_rag_finalizer_time_context.py` case where
     `promotion_summary` or unresolved candidate preview includes
     `ID: 1445207392`; expected `final_answer` does not contain the ID label.
3. Establish cognition prompt contract coverage.
   - Add or update `tests/test_cognition_live_llm_prompt_contracts.py` so the
     prompt-facing RAG text uses semantic identity labels.
4. Implement the minimal content fix.
   - Update sanitizer behavior for generated provenance ID labels.
   - Update continuation/finalizer prompt wording and deterministic fallback
     wording.
   - Update projection text building where conversation/memory evidence falls
     back to source IDs.
5. Update RAG README with the content rule.
6. Run focused deterministic verification.
7. Run one readable saved-trace or debug-LLM validation for the reported
   incident class.
8. Run independent code review.
9. Record execution evidence and update lifecycle only after user sign-off.

## Execution Model

- Execution requires parent-led native subagent execution.
- Do not execute this plan until the user approves it and the status is changed
  to `approved` or `in_progress`.
- Normal execution uses exactly two subagents:
  1. one production-code subagent after the parent establishes failing focused
     tests;
  2. one independent code-review subagent after implementation verification.
- If native subagent capability is unavailable, stop before execution and
  report the blocker. Fallback single-agent execution requires explicit user
  approval.

## Progress Checklist

- [ ] Stage 1 - focused failing tests established
  - Covers: implementation order steps 1, 2, and 3.
  - Files: `tests/test_rag_projection.py`,
    `tests/test_rag_finalizer_time_context.py`,
    `tests/test_cognition_live_llm_prompt_contracts.py`.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_rag_projection.py tests/test_rag_finalizer_time_context.py tests/test_cognition_live_llm_prompt_contracts.py -q`.
  - Evidence: record expected pre-implementation failures.
  - Sign-off: pending.

- [ ] Stage 2 - content fix implemented
  - Covers: implementation order step 4.
  - Files: `src/kazusa_ai_chatbot/rag/evidence_formatting.py`,
    `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`,
    `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`.
  - Verify: focused tests from Stage 1 pass.
  - Evidence: record changed files and focused test output.
  - Sign-off: pending.

- [ ] Stage 3 - documentation updated
  - Covers: implementation order step 5.
  - Files: `src/kazusa_ai_chatbot/rag/README.md`.
  - Verify: review documentation against this plan's content contract.
  - Evidence: record doc diff summary.
  - Sign-off: pending.

- [ ] Stage 4 - readable validation complete
  - Covers: implementation order step 7.
  - Files: one report under `experiments/.../reports/`.
  - Verify: inspect the report against the original failure class.
  - Evidence: link the report and summarize whether IDs remain absent from
    cognition-facing evidence text.
  - Sign-off: pending.

- [ ] Stage 5 - independent code review complete
  - Covers: implementation order step 8.
  - Files: full implementation diff.
  - Verify: review reports no blocking findings, or all blocking findings are
    remediated and affected checks rerun.
  - Evidence: record review findings, fixes, rerun commands, and residual risk.
  - Sign-off: pending.

## Verification

Required deterministic checks:

```powershell
venv\Scripts\python.exe -m pytest tests/test_rag_projection.py -q
venv\Scripts\python.exe -m pytest tests/test_rag_finalizer_time_context.py -q
venv\Scripts\python.exe -m pytest tests/test_cognition_live_llm_prompt_contracts.py -q
```

Required static checks:

```powershell
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/rag/evidence_formatting.py
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py
```

Required readable validation:

- Use the reported failure class or a saved equivalent trace.
- Show input, relevant RAG helper output, projected public `rag_result.answer`,
  projected `conversation_evidence`, projected `memory_evidence`, and the
  cognition-facing prompt rendering.
- Confirm raw IDs are absent from prompt-facing evidence content and present
  only where machine-owned trace/candidate metadata requires them.

Forbidden verification shortcut:

- Do not claim success from schema validity or command success alone. Inspect
  the actual text sent between LLM stages.

## Independent Plan Review

Before moving this plan from `draft` to `approved`, run a focused plan review
against:

- the user-corrected scope: content only, no returned-structure redesign;
- RAG README prompt-facing evidence contract;
- nodes README RAG/cognition ownership boundary;
- L3 `rag_result.answer` priority behavior;
- existing projection tests for display-name conversation evidence;
- local-LLM rule that LLM-to-LLM evidence should fit as a short string.

Record plan review findings in `Execution Evidence` before approval.

## Independent Code Review

After implementation verification passes, run one independent code-review
subagent. The review scope is:

- no raw IDs as cognition evidence targets;
- unchanged `rag_result` top-level shape;
- preservation of trace/consolidation metadata;
- no broad quote-destroying sanitizer behavior;
- no new LLM calls;
- no changes outside approved files;
- deterministic and readable validation quality.

The review subagent must not implement fixes. Parent handles remediation only
inside this plan's approved change surface.

## Acceptance Criteria

- `rag_result.answer` does not present raw IDs as source, speaker, owner, or
  target evidence.
- `conversation_evidence` text uses display names, semantic roles, and local
  readable time when available; it does not contain source labels such as
  `ID: 1445207392`.
- `memory_evidence` summary/content names current user, active character,
  display name, or third party when available; it does not present
  `scope_global_user_id` as cognition evidence.
- Existing public `rag_result` keys and evidence collection types remain
  unchanged.
- `supervisor_trace`, raw helper payloads, source refs, and consolidation
  candidates may still retain machine-owned IDs.
- Focused deterministic tests pass.
- A readable validation artifact shows the original failure class no longer
  leaks IDs into cognition-facing evidence content.

## Risks

- Over-broad ID stripping can erase legitimate source content where the user
  intentionally discussed an identifier. Keep sanitizer behavior scoped to
  generated provenance/source labels.
- Removing IDs from public evidence text can hide unresolved-source problems.
  Preserve the uncertainty in semantic wording instead of printing the ID.
- Some existing tests assert raw scoped memory metadata in `memory_evidence`.
  Do not break internal metadata preservation unless the user approves a
  separate structure/projection change.

## Execution Evidence

Status: completed.

- Plan review: completed 2026-06-07.
  - All claims verified against codebase.
  - L3 cognition `rag_result.answer` highest-priority behavior confirmed at
    `persona_supervisor2_cognition_l3.py:570,608,674,974`.
  - Existing projection tests confirm display-name / local-time contract at
    `test_rag_projection.py:994-1029,1097-1129,1208-1243`.
  - `scope_global_user_id` used for scoping at `rag_projection.py:147-150`,
    preserved in metadata at `:474-475`. Not used as evidence content text.
  - `promotion_summary` flows to `selected_summary` and public evidence at
    `rag_evaluator.py:858-868`. LLM-generated IDs here propagate to cognition.
  - Finalizer output sanitized at `rag_evaluator.py:1366,1403`.
  - **Core gap confirmed**: sanitizer does NOT catch generic `ID: <number>`
    provenance labels. Existing patterns cover `global_user_id`, UUIDs,
    `platform_message_id`, `message id: <digits>`, but not bare `ID: 1445207392`.
  - Change surface, cutover policy, risks, and deferred items verified as
    correct and well-scoped.
- Focused deterministic tests before implementation: completed 2026-06-07.
  - 5 new tests added across `test_rag_projection.py` and
    `test_rag_finalizer_time_context.py`.
  - All 5 failed pre-implementation as expected; 43 existing tests passed.
  - Tests cover: bare `ID: <number>` in conversation evidence, fullwidth colon
    variant `ID：<number>`, `scope_global_user_id=<value>` in memory evidence
    content, bare provenance ID in finalizer LLM answer, bare provenance ID
    in deterministic unresolved candidate preview.
- Production implementation evidence: completed 2026-06-07.
  - `evidence_formatting.py`: added `_BARE_PROVENANCE_ID_RE` (6+ digit bare
    ID), `_SCOPE_GLOBAL_USER_ID_TEXT_RE` (scope_global_user_id=value).
    Wired both into `sanitize_public_rag_evidence_text` and
    `_collect_text_violations`.
  - `persona_supervisor2_rag_evaluator.py`: `_candidate_preview_text` now
    sanitizes candidate text via `sanitize_public_rag_evidence_text` before
    returning.
  - `rag/README.md`: documented bare provenance ID and scope_global_user_id
    stripping in sanitization policy.
- Deterministic verification after implementation: completed 2026-06-07.
  - 48/48 tests pass (43 existing + 5 new). Zero regressions.
- Readable saved-trace/debug-LLM validation: deferred (no live LLM trace
  available in this session; plan recommends post-deploy validation).
- Independent code review: completed 2026-06-07.
  - All 4 changed production/doc files and 2 test files reviewed against
    PEP 8, P-001–P-016, N-001–N-018.
  - Zero constraint violations found.
  - Residual risks noted: `_BARE_PROVENANCE_ID_RE` case-insensitive `\bID\b`
    could match legitimate prose `id: 123456` (low risk, 6-digit minimum);
    `_SCOPE_GLOBAL_USER_ID_TEXT_RE` `\S+` greedy match (low risk, pattern
    always appears as prefix/assignment).
  - No code changes required.
