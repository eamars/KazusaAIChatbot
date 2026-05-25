# prompt prefix and input format optimization plan

## Summary

- Goal: identify and safely optimize prompt surfaces that hurt local LM Studio
  first-token latency, with `# 输入格式` / `# Input Format` removal handled as a
  measured exploratory gate before production prompt changes.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `py-style`, `cjk-safety`, `test-style-and-execution`,
  `python-venv`
- Overall cutover strategy: compatible experiment first; later surgical
  bigbang prompt rewrites only for prompts that pass the experiment and are
  approved for execution.
- Highest-risk areas: changing character judgment, weakening JSON output
  reliability, removing input-format instructions that local models still need,
  optimizing low-impact prompts while ignoring hot live prefixes, and expanding
  the plan into dialog streaming or backend-specific session management.
- Acceptance criteria: the experiment produces debug-LLM review artifacts for
  baseline versus input-format-omitted prompts; final optimization touches only
  approved candidates in the explicit `Change Surface`; changed prompts are
  rewritten into coherent prompt contracts after section removal; focused
  prompt contract tests and selected live LLM comparisons show no behavior
  regression; first-token-sensitive prompt prefixes become more stable.

## Context

The local LM Studio KV/prefix-cache experiment is recorded at:

```text
test_artifacts/experiments/lmstudio_kv_cache_probe/20260525T091841/
```

The observed result was:

| Condition | Prompt processing | LM Studio TTFT |
|---|---:|---:|
| Cold long prompt | `2.080s` | `3.420s` |
| Exact repeat warm average | `0.000594s` | `0.0585s` |
| Same prefix, changed tail | `0.292s` | `0.361s` |
| Early system-prefix miss average | `2.113s` | `2.680s` |

The useful local deployment metric is first-token time, not cache-hit counters.
The experiment shows that keeping long prompt prefixes stable matters, while
changing only tail input remains comparatively cheap.

The current production LLM call shape is generally good: most calls send a
`SystemMessage` first and a dynamic `HumanMessage` after it. The main risk is
that several hot prompts format changing runtime fields into the system prompt
itself. That creates early-prefix churn for LM Studio.

The user explicitly deprioritized dialog generation because it is already fast.
Dialog prompt changes are listed for inventory completeness only and are not a
primary target in this plan.

Initial L3b Content Anchor ablation evidence is recorded at:

```text
test_artifacts/experiments/l3_content_anchor_input_format_ablation/20260525T094905574052/
```

Conclusion from that experiment:

- Removing only the `_CONTENT_ANCHOR_AGENT_PROMPT` `# 输入格式` section reduced
  the rendered prompt from `11110` chars to `6389` chars and reduced prompt
  tokens by about `1599` tokens in both tested cases.
- Baseline and omitted-input-format variants both returned parseable,
  contract-valid JSON with the same anchor label sequences.
- The fact-answer case preserved the same decision, fact, answer, social,
  avoid-repeat, progression, and scope behavior.
- The unresolved-referent case preserved clarification behavior and did not
  guess from old context.
- The omitted-input-format variant was more faithful to the actual payload in
  the unresolved-referent probe: it used `these`, while the baseline copied
  the example phrase `这些` from the input-format block.
- This is positive evidence for L3b only. It is not approval to remove
  input-format sections globally.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing,
  archiving, or signing off this plan.
- `local-llm-architecture`: load before changing prompt boundaries, live-path
  LLM call shape, RAG/cognition/dialog ownership, or context placement.
- `debug-llm`: load before running live LLM comparisons and before writing
  human-readable quality review artifacts.
- `py-style`: load before editing Python source or experiment harnesses.
- `cjk-safety`: load before editing Python files that contain CJK prompt text.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `python-venv`: use `venv\Scripts\python` for Python commands.

## Mandatory Rules

- Do not execute production prompt changes while this plan status is `draft`.
- Do not add persistent cache storage, prompt-cache databases, backend session
  reuse, or LM Studio-native conversation state to production.
- Optimize for user-visible first-token time and stage duration, not provider
  cache-hit counters.
- Keep stable role, policy, and output contract text in the system prompt.
  Move volatile runtime facts to the human JSON payload or a late prompt tail.
- Do not move RAG evidence into persona or final wording ownership. RAG returns
  evidence; cognition decides stance; dialog renders visible text.
- Do not remove `# 输出格式` / `# Output Format` sections in this plan.
  The exploratory test targets only input-format sections unless a later plan
  explicitly expands scope.
- Do not remove `# 输入格式` / `# Input Format` from production prompts until a
  live LLM comparison shows parser validity and behavior equivalence for that
  exact prompt family.
- Removing an input-format section must not leave unexplained prompt inputs.
  For every approved prompt family, the remaining prompt body must explain each
  top-level human-payload field and every decision-critical nested field at the
  point where the field is used. If a field cannot be explained without
  recreating a schema dump, keep the input-format section for that prompt.
- Any major LLM prompt change in this plan includes removing an input-format
  section or moving volatile runtime fields out of a system prompt. For every
  such prompt, rewrite the affected prompt as a coherent prompt contract instead
  of mechanically deleting or moving lines. The rewrite must preserve the
  stage role, ownership boundary, generation procedure, output-format contract,
  enum semantics, and payload-field meaning.
- After every production prompt rewrite, run a prompt-flow review. The review
  must confirm the headings read in a usable order, no instruction points to a
  removed input-format section or deleted example, no visible field is left for
  the LLM to guess from the JSON key alone, and examples are not copied as
  facts.
- Do not run live LLM tests in large blind batches. Run one prompt family at a
  time, inspect output, and produce a debug-LLM review.
- Do not optimize dialog generator or dialog evaluator prompts in the first
  production pass. They are fast and explicitly deprioritized by the user.
- Do not change model routing, graph topology, database schemas, Cache2 policy,
  adapter delivery, scheduler behavior, or persistence behavior.
- Event logging must not store raw prompts, raw user messages, raw model
  outputs, credentials, or cache keys. Experiment artifacts may store synthetic
  prompts under `test_artifacts/experiments/`.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation, verification,
  handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` uses parent-led native subagent execution unless
  the user explicitly approves fallback execution.

## Must Do

- Preserve current production behavior until the exploratory input-format
  ablation is complete and reviewed.
- Create or extend an experiment harness under `experiments/` that can compare
  baseline prompts against input-format-omitted variants using the local LM
  Studio route.
- Write raw evidence under `test_artifacts/experiments/` and a human-authored
  debug-LLM review for each comparison set.
- Evaluate input-format omission before production optimization. Do not start
  final prompt rewrites until the exploratory results are reviewed.
- For every approved production prompt change, rewrite the prompt at the
  section level so the remaining instructions still flow naturally after the
  removed input-format section or relocated runtime fields.
- For every approved input-format removal, produce a field-coverage note that
  maps each removed input field to its remaining semantic explanation in the
  prompt body. Do not leave bare field names that force the LLM to infer their
  meaning from the JSON payload alone.
- List the production-scope candidate prompts in this plan and classify
  deferred prompt groups separately. Broad prompt inventory is discovery
  evidence, not final execution scope.
- In the final optimization pass, move volatile fields out of early system
  prompts for approved hot-path candidates.
- Update prompt-contract tests that intentionally assert prompt headings only
  after the matching prompt family passes the ablation gate.
- Keep dialog generation out of the first optimization implementation pass.

## Deferred

- Do not implement response streaming in this plan.
- Do not change dialog generation or dialog evaluator prompts in the first
  production pass.
- Do not optimize every small RAG helper prompt unless the exploratory review
  shows large benefit and no behavior risk.
- Do not remove output-format sections.
- Do not change JSON parser behavior, repair prompts, schemas, or validation
  policy unless a focused failure requires a separate plan.
- Do not add backend-specific LM Studio session reuse or OpenAI/DeepSeek cache
  key configuration.
- Do not change prompt language policy, character profile schema, memory
  semantics, or route selection.

## Cutover Policy

Overall strategy: compatible experiment first, then surgical bigbang prompt
rewrites for approved candidates.

| Area | Policy | Instruction |
|---|---|---|
| Exploratory harness | compatible | Add experiment code and artifacts without changing production behavior. |
| Input-format ablation | compatible | Compare baseline and omitted-input-format variants outside production. |
| Production input-format removal | bigbang per prompt | For each approved prompt, remove the input-format section directly; do not keep dual prompt branches. |
| Volatile field relocation | bigbang per prompt | Move runtime-varying fields to the human payload for the approved prompt; do not add feature flags or fallback prompt paths. |
| Dialog prompts | no change | Leave dialog generator and evaluator prompts unchanged in the first pass. |
| Tests | bigbang per changed prompt | Rewrite tests that assert old prompt text only for prompts changed by this plan. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- For bigbang areas, rewrite the old prompt directly instead of preserving old
  and new variants.
- For compatible areas, preserve only the experiment surfaces explicitly
  listed in this plan.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

The target state has two completed layers:

1. Exploratory evidence exists showing which prompt families can omit
   `# 输入格式` / `# Input Format` without changing parse validity or semantic
   decisions.
2. Approved hot-path prompts have stable early system prefixes. Runtime facts
   such as mood, global vibe, user name, affinity guidance, relationship
   insight, timestamp, and similar per-call values are moved to the human JSON
   payload or a late prompt tail.
3. Approved prompts with removed input-format sections still read as complete
   prompt contracts: the generation procedure, field meanings, and output
   contract remain explicit without requiring schema inference from JSON keys.

No production state, database, graph topology, or adapter contract changes.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Optimization metric | Use LM Studio TTFT and prompt-processing duration | Local deployment cares about first visible token, not provider cache counters. |
| Experiment first | Run input-format omission outside production | Local LLMs may rely on explicit input descriptions; removal must be evidence-based. |
| Output format | Keep output-format sections | JSON shape reliability is higher value than a small prompt-size reduction. |
| Dialog | Defer dialog prompt optimization | User reported dialog generation is fast and not worth first-pass risk. |
| Backend-specific cache APIs | Do not add | LM Studio native behavior is not a portable OpenAI-compatible contract. |
| Final prompt changes | Bigbang per approved prompt | Dual prompt branches add contract complexity without runtime value. |
| Execution scope | Limit first production pass to explicit `Change Surface` files | The broad prompt inventory is useful evidence but should not authorize unrelated prompt edits. |
| Major prompt edits | Rewrite affected prompts as coherent contracts | Local models should not be asked to infer missing flow after section deletion. |

## Contracts And Data Shapes

### Experiment Evidence

The exploratory harness must write one raw JSON document per run:

```python
{
    "schema": "prompt_input_format_ablation.v1",
    "created_utc": str,
    "route": {
        "route_prefix": str,
        "base_url": str,
        "model": str,
    },
    "prompt_family": str,
    "cases": [
        {
            "case_id": str,
            "variant": "baseline | omit_input_format",
            "system_prompt_chars": int,
            "human_prompt_chars": int,
            "prompt_sha256_16": str,
            "duration_seconds": float,
            "parsed_status": str,
            "raw_output": str,
            "parsed_output": dict,
        }
    ],
    "comparison": {
        "parser_equivalent": bool,
        "key_fields_equivalent": bool,
        "agent_notes": str,
    },
}
```

The script must not produce the human-readable quality review. The parent
agent writes the debug-LLM review after inspecting raw evidence.

### Input-Format Omission

Input-format omission means removing the section headed by exactly one of:

```text
# 输入格式
## 输入格式
# Input Format
## Input Format
```

The human JSON payload shape remains unchanged. The output-format section
remains in the system prompt.

## LLM Call And Context Budget

- No new production LLM calls are added.
- The exploratory harness uses local LM Studio only and runs outside the live
  response path.
- Live LLM experiment cases must run one prompt family at a time and produce
  debug-LLM review artifacts.
- Default context cap for planning is `50k tokens`; candidate prompts in this
  plan are below that cap based on character-count inspection.
- Production prompt changes should reduce or preserve prompt size. Any prompt
  size increase requires user approval.

## Candidate Prompt Inventory

This section lists prompts that may be changed by this plan. The broader AST
inventory of every prompt with an input-format heading is discovery evidence
only; it must be regenerated in Stage 1 and recorded in `Execution Evidence`,
not copied into the execution contract as authorized change scope.

### Production Change Candidates

These are the only production prompt families that this plan may change after
the experiment and user approval gates pass.

| Priority | Prompt | Current issue | Planned action |
|---|---|---|---|
| P0 | `src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py::_RELEVANCE_SYSTEM_PROMPT` | Formats character name, mood, global vibe, user name, affinity guidance, relationship insight, and platform bot id into the system prompt. | Move volatile per-user/per-turn values to human JSON. Remove input-format only if the relevance-family ablation passes. |
| P0 | `src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py::_RELEVANCE_SYSTEM_NOISY_PROMPT` | Same as above; longer and hot in noisy group contexts. | Move volatile values to human JSON. Remove input-format only if the noisy relevance-family ablation passes. |
| P0 | `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py::_COGNITION_SUBCONSCIOUS_PROMPT` | Formats mood, global vibe, and relationship insight into the system prompt. | Move volatile state to human JSON. Remove input-format only if the L1 ablation passes. |
| P0 | `src/kazusa_ai_chatbot/rag/web_search_agent.py::_WEB_SEARCH_GENERATOR_PROMPT` | Formats timestamp into the system prompt. | Move timestamp to human JSON. Remove input-format only if the web generator ablation passes. |
| P0 | `src/kazusa_ai_chatbot/rag/web_search_agent.py::_WEB_SEARCH_EVALUATOR_PROMPT` | Formats timestamp into the system prompt. | Move timestamp to human JSON. Remove input-format only if the web evaluator ablation passes. |
| P1 | `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py::_CONTENT_ANCHOR_AGENT_PROMPT` | Very long prompt with positive L3b ablation evidence already recorded. | Do not move character identity in this plan. Remove input-format only if the user approves the L3b evidence and field-coverage review. |
| P1 | `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py::_MSG_DECONTEXUALIZER_PROMPT` | Uses prompt template substitution via `.replace(...)` and has a live-path input-format section that may be latency-relevant. | Convert template substitution to `.format(...)`. Evaluate input-format omission with matching ablation evidence; remove input-format only if decontextualizer behavior and referent output remain acceptable. |

### Experiment-Only Or Deferred Prompt Groups

These prompt groups may be measured by the exploratory harness, but this plan
does not authorize production edits to them. Production changes to these groups
require a later plan update or a separate approved plan.

| Group | Boundary |
|---|---|
| RAG initializer and dispatcher | Measure only; do not change production prompts in this plan. |
| L2/L2c2 cognition prompts | Measure only; volatile state is already mostly in human payload. |
| Other L3 prompts: style, preference adapter, visual | Measure only; leave unchanged unless a later approved plan targets them. |
| Vision descriptor | Measure only; not part of first production pass. |
| RAG evaluator and helper agents | Measure only; avoid spending first-pass implementation time on low-impact helper prompts. |
| Background consolidation, reflection, growth, and recorder prompts | Measure only; outside live first-token path for normal chat. |
| Utility JSON repair prompt | Leave unchanged; parser/repair policy is outside this plan. |
| Dialog generator and evaluator | Leave unchanged; dialog is fast and explicitly deprioritized. |

## Change Surface

### Create

- `experiments/prompt_input_format_ablation.py`: outside-production harness
  for baseline versus input-format-omitted prompt comparison.
- `test_artifacts/experiments/prompt_input_format_ablation/<run>/run.json`:
  raw evidence generated by the harness.
- `test_artifacts/experiments/prompt_input_format_ablation/<run>/debug_review.md`:
  human-authored debug-LLM review written after inspecting raw evidence.

### Modify

- `development_plans/README.md`: register this draft active short-term plan.
- `development_plans/active/short_term/prompt_prefix_and_input_format_optimization_plan.md`:
  maintain this work contract, review findings, and execution evidence.
- `src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py`: after approval,
  rewrite `_RELEVANCE_SYSTEM_PROMPT` and `_RELEVANCE_SYSTEM_NOISY_PROMPT` as
  coherent contracts; move volatile fields from system prompt to human payload;
  remove input-format sections only for variants with passing relevance-family
  ablation evidence and field-coverage review.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`: after
  approval, rewrite `_COGNITION_SUBCONSCIOUS_PROMPT` as a coherent contract;
  move volatile mood/global-vibe/relationship state from system prompt to human
  payload; remove the input-format section only if L1 ablation and
  field-coverage review pass.
- `src/kazusa_ai_chatbot/rag/web_search_agent.py`: after approval, rewrite
  `_WEB_SEARCH_GENERATOR_PROMPT` and `_WEB_SEARCH_EVALUATOR_PROMPT` as coherent
  contracts; move timestamp from system prompt to human payload; remove
  input-format sections only if matching web ablation and field-coverage review
  pass. Do not change `_WEB_SEARCH_FINALIZER_PROMPT` in this plan.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`: after
  explicit user approval of L3b evidence, rewrite only
  `_CONTENT_ANCHOR_AGENT_PROMPT` as a coherent contract and remove its
  input-format section only if field-coverage review passes. Do not change
  `_STYLE_AGENT_PROMPT`, `_PREFERENCE_ADAPTER_PROMPT`, or
  `_VISUAL_AGENT_PROMPT` in this plan.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`:
  user-approved scope expansion after execution started; convert
  `_MSG_DECONTEXUALIZER_PROMPT` placeholder substitution from `.replace(...)`
  to `.format(...)`, escape literal JSON braces as needed, then evaluate its
  input-format section with a matching ablation run. Remove the input-format
  section only if decontextualizer ablation evidence, debug-LLM review, and
  field-coverage review pass.
- Focused tests that assert prompt headings or payload shape for prompts
  changed by this plan.

### Keep

- All production prompt files not listed in `Modify`: unchanged.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: keep unchanged in the first
  optimization pass.
- Production graph topology, model route config, Cache2, database schema,
  adapters, scheduler, and event-log storage behavior.

## Overdesign Guardrail

- Actual problem: local first-token latency suffers when hot long prompt
  prefixes change near the beginning, and many prompt input-format sections may
  be redundant but unproven.
- Minimal change: run a live local ablation experiment first, then make
  surgical prompt changes only for approved hot-path candidates.
- Ownership boundaries: LLM prompts own semantic instructions and output
  contracts; deterministic code owns validation, parsing, telemetry safety,
  persistence, cache policy, and adapter delivery.
- Rejected complexity: persistent prompt cache, session reuse, provider-specific
  cache keys, dual prompt branches, feature flags, output-format removal,
  dialog streaming, graph rewiring, and prompt rewrites outside the explicit
  change surface.
- Evidence threshold: add any rejected complexity only after live LM Studio
  evidence shows the smaller prompt-stability changes do not improve TTFT or
  fail to preserve behavior.

## Agent Autonomy Boundaries

- The responsible agent may choose local experiment implementation mechanics
  only when they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate cache
  strategies, compatibility layers, fallback paths, or extra features.
- The responsible agent must treat changes outside the listed change surface as
  blocked until the plan is updated and approved.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, unapproved prompt rewrites, or route changes.
- The responsible agent must keep raw LLM evidence out of event logging and
  inside local experiment artifacts.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Establish prompt inventory and baseline evidence.
   - Run the existing prompt inventory greps and AST scan used to draft this
     plan.
   - Record which prompts contain input-format sections and which prompts have
     volatile system-prefix fields.
2. Build the exploratory ablation harness.
   - Create `experiments/prompt_input_format_ablation.py`.
   - The harness must read prompt constants by AST or direct safe import only
     when import has no production side effects.
   - The harness must remove only the input-format section for the ablation
     variant.
3. Run exploratory live LLM comparisons one prompt family at a time.
   - Required production-candidate set: relevance non-noisy, relevance noisy,
     L1 subconscious, web search generator, and web search evaluator.
   - User-approved scope clarification adds message decontextualizer after
     execution started; run its ablation before any input-format removal.
   - L3b content anchor already has pre-approval evidence. Rerun it only if the
     route, payload construction, or user review asks for a fresh comparison.
   - Experiment-only/deferred groups may be measured for discovery, but those
     results do not authorize production edits in this plan.
   - Dialog prompts remain excluded from first-pass optimization.
4. Write debug-LLM review artifacts.
   - Each review must include run context, real input, raw output, parsed
     output, comparison summary, quality notes, validation, and raw evidence
     paths.
5. Stop for user review.
   - Do not begin production prompt rewrites until the user approves the
     reviewed prompt families.
6. Implement approved final prompt optimizations.
   - Move volatile runtime fields from hot system prompts to human payloads.
   - Remove input-format sections only for prompt families approved by the
     exploratory review.
   - Rewrite each changed prompt at the section level so role, procedure,
     payload-field meanings, and output contract remain readable without the
     removed section.
7. Audit remaining prompt-body field explanations for every approved removal.
   - For each prompt family with `# 输入格式` / `# Input Format` removed, create
     a short field-coverage note in the debug-LLM review or execution evidence.
   - The note must list each top-level human payload field and each
     decision-critical nested field removed from the input-format section.
   - Each listed field must point to a remaining prompt rule that explains its
     meaning or to a minimal inline explanation added during the rewrite.
   - If any field has no remaining explanation, restore the input-format
     section for that prompt or add a concise explanation outside a schema-dump
     format before proceeding.
8. Perform prompt-flow and functionality review for every changed prompt.
   - Render at least one real system-plus-human prompt fixture per changed
     prompt family.
   - Confirm there are no dangling references to removed sections, examples,
     or obsolete field names.
   - Confirm the remaining prompt gives the LLM enough semantic explanation to
     use every current-run input without guessing from JSON names alone.
9. Update focused tests.
   - Rewrite only tests tied to changed prompts.
   - Preserve output-format and parser-contract assertions.
10. Verify and run independent code review.

## Execution Model

- Parent agent owns orchestration, experiment evidence, debug-LLM reviews,
  focused tests, verification, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test/experiment contract before
  production implementation starts.
- Production-code subagent: exactly one native subagent, started only after
  the user approves exploratory results and the plan status becomes
  `approved` or `in_progress`; owns production prompt changes only.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews plan alignment, prompt safety, tests,
  and evidence; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - prompt inventory refreshed
  - Covers: implementation order step 1.
  - Verify: record AST prompt inventory and `rg` input-format scan output in
    `Execution Evidence`.
  - Sign-off: after evidence is recorded.
- [x] Stage 2 - exploratory ablation harness created
  - Covers: implementation order step 2.
  - Verify: `venv\Scripts\python -m py_compile experiments/prompt_input_format_ablation.py`.
  - Sign-off: after the harness writes raw evidence for one smoke prompt.
- [x] Stage 3 - first live LLM comparison set completed
  - Covers: implementation order steps 3 and 4 for the required
    production-candidate set.
  - Verify: every prompt family has raw JSON evidence and a debug-LLM review.
  - Sign-off: after reviews are inspected and recorded.
- [x] Stage 4 - user review gate completed
  - Covers: implementation order step 5.
  - Verify: user approval identifies exact prompt families allowed for final
    optimization.
  - Sign-off: after plan status and execution boundary are updated.
- [x] Stage 5 - approved production prompt optimizations implemented
  - Covers: implementation order steps 6, 7, 8, and 9.
  - Verify: prompt-flow review passes, focused tests pass, and no unapproved
    prompt files changed.
  - Sign-off: after changed files, prompt-flow evidence, and test evidence are
    recorded.
- [x] Stage 6 - final verification completed
  - Covers: verification section.
  - Verify: static greps, focused tests, and approved live LLM checks pass.
  - Sign-off: after evidence is recorded.
- [x] Stage 7 - independent code review completed
  - Covers: independent code review gate.
  - Verify: review findings, fixes, rerun commands, and approval status are
    recorded in `Execution Evidence`.
  - Sign-off: before marking the plan completed.

## Verification

### Static Greps

- `rg -n "# 输入格式|## 输入格式|# Input Format|## Input Format" src/kazusa_ai_chatbot`
  - Expected before final optimization: matches remain for all prompts.
  - Expected after final optimization: changed prompt families no longer have
    input-format headings; unchanged/deferred prompts may still match.
- `rg -n "timestamp=.*format|mood=.*format|global_vibe=.*format|user_name=.*format|last_relationship_insight=.*format" src/kazusa_ai_chatbot/nodes src/kazusa_ai_chatbot/rag`
  - Expected after final optimization: no matches for approved hot prompts.
    Matches in deferred or unrelated prompts must be listed as allowed.
- `rg -n "_DIALOG_GENERATOR_PROMPT|_DIALOG_EVALUATOR_PROMPT" src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Expected after first pass: dialog prompt constants still exist and are not
    included in the production optimization diff.

### Syntax

- `venv\Scripts\python -m py_compile experiments/prompt_input_format_ablation.py`
- `venv\Scripts\python -m py_compile <each changed production Python file>`

### Prompt Flow And Functionality Review

For every production prompt changed by this plan:

- render at least one representative system-plus-human prompt from the real
  handler or from a fixture that matches the handler payload exactly;
- confirm the prompt has a coherent role, generation procedure, field
  explanation, and output-format sequence after input-format removal;
- confirm there are no dangling references to `# 输入格式`, `# Input Format`,
  removed examples, obsolete payload names, or "above/below format" wording
  that no longer exists;
- confirm every top-level human payload field and decision-critical nested
  field is explained where the prompt uses it;
- run the matching live LLM comparison or approved post-change spot check and
  confirm parse validity plus semantic behavior remain acceptable.

### Focused Tests

Run focused tests for every changed prompt family. Current known prompt-heading
tests that may need updates only if their prompts change:

- `venv\Scripts\python -m pytest tests/test_msg_decontexualizer.py -q`
- `venv\Scripts\python -m pytest tests/test_persona_supervisor2_action_initializer.py -q`
- `venv\Scripts\python -m pytest tests/test_rag_phase3_capability_agents.py -q`
- `venv\Scripts\python -m pytest tests/test_conversation_progress_recorder.py -q`
- `venv\Scripts\python -m pytest tests/test_internal_monologue_residue_prompt_boundaries.py -q`
- `venv\Scripts\python -m pytest tests/test_global_character_growth_prompt_contracts.py -q`
- `venv\Scripts\python -m pytest tests/test_reflection_cycle_prompt_contracts.py -q`
- `venv\Scripts\python -m pytest tests/test_reflection_cycle_stage1c_promotion.py -q`

### Live LLM Debug Checks

- Run the ablation harness one prompt family at a time against local LM Studio.
- Produce one debug-LLM review per comparison set.
- Required pass criteria for each prompt family before production removal:
  - baseline and omitted-input-format variants both return parseable output;
  - required output keys and enum values remain valid;
  - decision-critical fields are equivalent or better by human review;
  - no new generic-assistant behavior, persona drift, or RAG/cognition
    boundary confusion appears.

### Field Explanation Coverage

For every production prompt that removes an input-format section, record a
field-coverage table in `Execution Evidence` or the matching debug-LLM review:

| Field | Remaining explanation location | Status |
|---|---|---|
| `example_top_level_field` | prompt section or rule name | explained |

Required result:

- every top-level human payload field is listed;
- every decision-critical nested field is listed;
- every listed field has status `explained`;
- no field relies only on the JSON key name for semantic meaning;
- no replacement section recreates the removed input-format schema dump.

## Independent Plan Review

Review date: 2026-05-25.

Review focus:

- Remove execution-irrelevant information from the plan body.
- Make production change surface explicit.
- Require coherent prompt rewrites for major LLM prompt changes.
- Ensure prompts remain functional after input-format sections are removed.

Findings and remediation:

| Finding | Severity | Remediation |
|---|---|---|
| The broad input-format inventory made the plan read like it authorized many prompt edits beyond the final execution target. | High | Replaced the broad prompt table with production change candidates and experiment-only/deferred groups. Stage 1 still regenerates broad inventory as evidence. |
| The original change surface did not explicitly list every production prompt file that could be changed, especially L3b content anchor. | High | `Change Surface` now names every production file and prompt constant that may change, and marks all other prompt files unchanged. |
| The plan allowed section removal without explicitly requiring the execution agent to rewrite the affected prompt flow. | High | Added mandatory prompt-rewrite rules, implementation steps, and verification gates for section-level coherent rewrites. |
| Field explanation coverage existed but did not fully prove the new prompt remained functional after the removed section. | Medium | Added prompt-flow and functionality review requiring rendered prompt inspection, dangling-reference checks, field explanation checks, and live/post-change behavior checks. |

Review result: no remaining plan-review blockers for approved execution.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and experiment artifact.
- Prompt ownership and local-LLM architecture alignment.
- Whether any input-format removal is backed by debug-LLM evidence.
- Whether volatile fields were moved to human payload without changing output
  contracts or downstream consumers.
- Whether dialog prompts, route config, graph topology, persistence, adapters,
  and event logging were left untouched as required.
- Whether verification and evidence match the actual diff.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- The exploratory input-format ablation has raw evidence and debug-LLM reviews.
- The user has approved exact prompt families for final optimization.
- Only approved prompt families are changed.
- No production prompt file outside the explicit `Change Surface` `Modify`
  list is changed.
- Hot approved prompts no longer put volatile runtime values at the beginning
  of the system prompt.
- Approved input-format removals preserve parse validity and semantic behavior
  under live LM Studio checks.
- Approved input-format removals include field-coverage evidence proving the
  remaining prompt explains all required input semantics.
- Every changed prompt has prompt-flow review evidence showing the prompt still
  functions as a coherent contract after section removal or field relocation.
- Dialog generation and evaluator prompts remain unchanged in the first pass.
- Focused deterministic tests and approved live LLM reviews pass.
- Independent code review is complete and recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Removing input-format sections weakens local model JSON reliability | Keep output-format sections and require live ablation before removal | Parse status and debug-LLM review per prompt family |
| Removing input-format sections leaves unexplained payload fields | Require field-coverage notes and restore the section if any field lacks a remaining explanation | Field explanation coverage table |
| Mechanical section deletion leaves an incoherent prompt | Require section-level prompt rewrite and prompt-flow review for every changed prompt | Prompt flow and functionality review |
| Prefix optimization changes character behavior | Move only runtime facts, not semantic rules; compare baseline/after outputs | Focused live LLM comparisons |
| Low-impact prompts consume implementation time | Prioritize P0 hot prompts and defer dialog/background work | Change-surface review and git diff |
| Raw prompts leak into persistent telemetry | Keep raw evidence only in local `test_artifacts` | Event logging grep/review |
| Backend behavior differs after leaving LM Studio | Do not add backend-specific session/cache contracts | Code review and config diff |

## Execution Evidence

- Pre-approval supporting evidence:
  - L3b `_CONTENT_ANCHOR_AGENT_PROMPT` ablation run:
    `test_artifacts/experiments/l3_content_anchor_input_format_ablation/20260525T094905574052/run.json`
  - L3b debug-LLM review:
    `test_artifacts/experiments/l3_content_anchor_input_format_ablation/20260525T094905574052/debug_review.md`
  - Result: no drastic behavior change in the two tested L3b cases; both
    variants produced valid `content_anchors` with matching label sequences;
    omitted-input-format output was more faithful to the actual unresolved
    referent phrase in the synthetic payload.
- Execution started after user approval on 2026-05-25.
- Pre-execution plan commit: `880c7f8 Add prompt optimization development plan`.
- Stage 1 evidence:
  - Prompt input-format inventory refreshed with:
    `rg -n "# 输入格式|## 输入格式|# Input Format|## Input Format" src/kazusa_ai_chatbot`
  - Volatile system-prefix scan refreshed with:
    `rg -n "timestamp=.*format|mood=.*format|global_vibe=.*format|user_name=.*format|last_relationship_insight=.*format|format\(" src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py src/kazusa_ai_chatbot/rag/web_search_agent.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- Stage 2 evidence:
  - Created `experiments/prompt_input_format_ablation.py`.
  - Verified with:
    `venv\Scripts\python -m py_compile experiments/prompt_input_format_ablation.py`
  - Verified CLI shape with:
    `venv\Scripts\python experiments/prompt_input_format_ablation.py --help`
- Stage 3 evidence:
  - Private relevance run:
    `test_artifacts/experiments/prompt_input_format_ablation/relevance_private/20260525T101625848776/run.json`
  - Private relevance review:
    `test_artifacts/experiments/prompt_input_format_ablation/relevance_private/20260525T101625848776/debug_review.md`
  - Noisy relevance run:
    `test_artifacts/experiments/prompt_input_format_ablation/relevance_noisy/20260525T101635011241/run.json`
  - Noisy relevance review:
    `test_artifacts/experiments/prompt_input_format_ablation/relevance_noisy/20260525T101635011241/debug_review.md`
  - L1 subconscious run:
    `test_artifacts/experiments/prompt_input_format_ablation/l1_subconscious/20260525T101644892204/run.json`
  - L1 subconscious review:
    `test_artifacts/experiments/prompt_input_format_ablation/l1_subconscious/20260525T101644892204/debug_review.md`
  - Web generator run:
    `test_artifacts/experiments/prompt_input_format_ablation/web_generator/20260525T101716257111/run.json`
  - Web generator review:
    `test_artifacts/experiments/prompt_input_format_ablation/web_generator/20260525T101716257111/debug_review.md`
  - Web evaluator run:
    `test_artifacts/experiments/prompt_input_format_ablation/web_evaluator/20260525T101724243292/run.json`
  - Web evaluator review:
    `test_artifacts/experiments/prompt_input_format_ablation/web_evaluator/20260525T101724243292/debug_review.md`
  - Result: all five required production-candidate families preserved parser
    or tool-call validity, required keys, and decision-critical behavior.
    L1 became slightly sharper in tone but stayed within first-reaction and
    subtext ownership.
- Stage 4 evidence:
  - User approved execution on 2026-05-25 and instructed execution with
    subagents.
  - Exact production rewrite scope is limited to:
    `persona_relevance_agent::_RELEVANCE_SYSTEM_PROMPT`,
    `persona_relevance_agent::_RELEVANCE_SYSTEM_NOISY_PROMPT`,
    `persona_supervisor2_cognition_l1::_COGNITION_SUBCONSCIOUS_PROMPT`,
    `web_search_agent::_WEB_SEARCH_GENERATOR_PROMPT`,
    `web_search_agent::_WEB_SEARCH_EVALUATOR_PROMPT`, and
    `persona_supervisor2_cognition_l3::_CONTENT_ANCHOR_AGENT_PROMPT`.
- Scope clarification after execution started:
  - User identified `_MSG_DECONTEXUALIZER_PROMPT` as also having targeted
    input-format guidance and `.replace(...)` prompt substitution.
  - The plan scope now includes
    `persona_supervisor2_msg_decontexualizer::_MSG_DECONTEXUALIZER_PROMPT`
    for `.replace(...)` to `.format(...)` conversion and input-format
    ablation. Its input-format section must not be removed until the matching
    ablation, debug-LLM review, and field-coverage review pass.
- Stage 3 scope-expansion evidence:
  - Message decontextualizer run:
    `test_artifacts/experiments/prompt_input_format_ablation/msg_decontextualizer/20260525T103855744619/run.json`
  - Message decontextualizer review:
    `test_artifacts/experiments/prompt_input_format_ablation/msg_decontextualizer/20260525T103855744619/debug_review.md`
  - Result: baseline and omitted-input-format variants both parsed, preserved
    reply confirmation expansion, and preserved unresolved referent signaling.
    The omitted variant reduced rendered system prompt size from `4241` to
    `3339` chars and prompt tokens by `283` in both tested cases.
- Stage 5 implementation evidence:
  - Production files changed:
    `src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py`,
    `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`,
    `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`,
    `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`,
    and `src/kazusa_ai_chatbot/rag/web_search_agent.py`.
  - Test files changed:
    `tests/test_persona_relevance_agent.py` and
    `tests/test_msg_decontexualizer.py`.
  - Relevance prompts now keep stable role, decision, and output contract
    text in the system prompt. Per-run `character_name`, bot id, mood,
    global vibe, affinity guidance, relationship insight, user identity,
    reply context, direct-address metadata, group attention, and engagement
    context are explained in the prompt body and passed in Human JSON.
  - L1 subconscious prompt now keeps source recognition, first-reaction
    procedure, MBTI guidance, and output contract in the system prompt.
    Per-run mood, global vibe, relationship insight, user input, indirect
    speech context, media observations, reflection artifact, and internal
    thought residue are explained in prompt rules and passed in Human JSON.
  - Web generator and evaluator prompts now read `reference_time` from Human
    JSON. Task, context, expected response, search/read history, call history,
    tool-result distinction, and evaluator feedback remain explained in the
    prompt procedure; output/tool contracts are unchanged.
  - L3 content anchor replaced the removed schema dump with field guidance for
    `user_input`, `dialog_act_intent`, cognitive stage summaries, retrieved
    memory/context evidence, conversation progress, interaction style,
    referents, media observations, and candidate grounding.
  - Message decontextualizer converted placeholder rendering from
    `.replace(...)` to `.format(...)`, escaped literal JSON braces, and
    replaced the removed schema dump with field guidance for `user_input`,
    speaker/bot identity, typed message context, mentions, attachments,
    history, reply context, channel topic, and indirect speech context.
  - Prompt-flow render check passed for:
    `_RELEVANCE_SYSTEM_PROMPT: 3019`,
    `_RELEVANCE_SYSTEM_NOISY_PROMPT: 5234`,
    `_COGNITION_SUBCONSCIOUS_PROMPT: 1464`,
    `_WEB_SEARCH_GENERATOR_PROMPT: 872`,
    `_WEB_SEARCH_EVALUATOR_PROMPT: 2428`,
    `_CONTENT_ANCHOR_AGENT_PROMPT: 7821`,
    `_MSG_DECONTEXUALIZER_PROMPT: 4055`.
  - The render check also confirmed no changed prompt still has `# 输入格式`
    or `# Input Format`.
  - Field-coverage table for removed input-format sections:

| Prompt family | Field or field group | Remaining explanation location | Status |
|---|---|---|---|
| Relevance private/noisy | `current_run_context.character_name`, `platform_bot_id` | `# 本轮语境` identity rule | explained |
| Relevance private/noisy | `current_run_context.mood`, `global_vibe` | `# 本轮语境` and decision steps say these only adjust qualified messages | explained |
| Relevance private/noisy | `current_run_context.affinity_level`, `affinity_instruction`, `last_relationship_insight` | `# 本轮语境`, `# 响应决策逻辑`, and noisy evidence tier 5 | explained |
| Relevance private/noisy | `user_message.user_name`, `platform_user_id`, `content`, `channel_name` | `# 本轮语境`, `# 思考路径`, and direct/third-party decision rules | explained |
| Relevance private/noisy | `user_message.directly_addressed`, `reply_context.reply_to_platform_user_id`, `reply_to_display_name`, `reply_excerpt` | `# 本轮语境`, `# 证据层级`, and `use_reply_feature` rules | explained |
| Relevance private/noisy | `conversation_history` | `# 本轮语境`, conversation-continuity rules, and noisy evidence tier 3 | explained |
| Relevance noisy | `group_attention` | `# 本轮语境` and noisy evidence tier 2 | explained |
| Relevance noisy | `user_engagement_context.engagement_guidelines`, `confidence` | `# 本轮语境` and noisy evidence tier 4 | explained |
| L1 subconscious | `character_state.mood`, `global_vibe`, `last_relationship_insight` | `# 本轮语境` emotion-filter rules | explained |
| L1 subconscious | `user_input`, `indirect_speech_context` | `# 来源识别`, `# 本轮语境`, and generation flow | explained |
| L1 subconscious | `media_observations.image_observations`, `audio_observations` | `# 本轮语境` and generation flow media rules | explained |
| L1 subconscious | `reflection_artifact`, `internal_thought_residue.internal_monologue`, `action_latch` | `# 来源识别` and generation flow source rules | explained |
| Web generator | `task`, `context`, `reference_time`, appended `messages` | `# 本轮输入`, `# 任务流程`, and `# 语言与时间` | explained |
| Web evaluator | `task`, `expected_response`, `call_history`, `retry`, `reference_time` | `# 本轮输入`, `# 审计步骤`, and `# 消息时效与计算` | explained |
| L3 content anchor | `decontexualized_input`, `referents`, `media_observations` | `# 本轮输入字段说明` and Clarification override/media rules | explained |
| L3 content anchor | `rag_result.answer`, evidence arrays, `user_image.user_memory_context`, `character_image`, `third_party_profiles`, `supervisor_trace` | `# 本轮输入字段说明`, `[FACT]`, `[ANSWER]`, and dependency tree rules | explained |
| L3 content anchor | `internal_monologue`, `logical_stance`, `character_intent`, `selected_text_surface_intent` | `# 本轮输入字段说明`, dependency tree, and `[DECISION]`/`[ANSWER]` rules | explained |
| L3 content anchor | `memory_lifecycle_context`, `interaction_style_context`, `conversation_progress` | `# 本轮输入字段说明`, `[SOCIAL]`, `[PROGRESSION]`, and open-loop rules | explained |
| L3 content anchor | `reflection_artifact`, `internal_thought_residue` | `# 本轮输入字段说明` source-ownership rules | explained |
| Message decontextualizer | `user_input`, `platform_user_id`, `user_name`, `platform_bot_id` | `# 本轮输入字段说明` identity and rewrite-scope rules | explained |
| Message decontextualizer | `prompt_message_context.body_text`, `addressed_to_global_user_ids`, `broadcast`, `mentions` | `# 本轮输入字段说明` typed-envelope evidence rule | explained |
| Message decontextualizer | `prompt_message_context.attachments.description`, `summary_status` | `# 本轮输入字段说明` attachment referent rule | explained |
| Message decontextualizer | `chat_history.display_name`, `body_text` | `# 本轮输入字段说明` history referent and reported-speech rule | explained |
| Message decontextualizer | `reply_context.reply_to_display_name`, `reply_excerpt` | `# 本轮输入字段说明` strong reply-evidence rule | explained |
| Message decontextualizer | `channel_topic`, `indirect_speech_context` | `# 本轮输入字段说明` weak-hint rule | explained |

- Stage 6 verification evidence:
  - `rg -n "# 输入格式|## 输入格式|# Input Format|## Input Format" ...`
    over the changed prompt files returned only deferred unchanged prompt
    headings: vision descriptor, L3 style, L3 preference, L3 visual, and web
    finalizer.
  - `rg -n 'timestamp=.*format|mood=.*format|global_vibe=.*format|user_name=.*format|last_relationship_insight=.*format|\.replace\(\s*"\{[^\"]+\}"|\.replace\(\s*''\{[^'']+\}''' src/kazusa_ai_chatbot/nodes src/kazusa_ai_chatbot/rag`
    returned no matches; exit code `1` is the expected no-match result.
  - `rg -n "_DIALOG_GENERATOR_PROMPT|_DIALOG_EVALUATOR_PROMPT" src/kazusa_ai_chatbot/nodes/dialog_agent.py`
    confirmed dialog prompt constants remain present and outside this diff.
  - `venv\Scripts\python -m py_compile experiments/prompt_input_format_ablation.py <changed production files>`
    passed.
  - `git diff --check` passed, with only existing LF-to-CRLF warnings from
    Git.
  - Focused deterministic tests passed:
    `66 passed, 10 deselected`.
  - Post-review targeted payload tests passed:
    `tests/test_web_search_agent.py` plus
    `tests/test_cognition_prompt_contract_text.py::test_l1_subconscious_payload_passes_character_state_in_human_json`
    returned `7 passed`.
  - Selected multi-source regression slice passed:
    `56 passed, 4 deselected`.
  - Live LLM checks were run one at a time and inspected:
    decontextualizer unresolved referent passed; noisy relevance preserved
    `should_respond=True` and `use_reply_feature=True`; content anchor
    birthday fact case returned a grounded `[FACT]` anchor; full cognition
    weather-English case passed through L1-L4 and preserved English reply
    preference.
  - Known unrelated/stale failures remain outside this plan: prompt fingerprint
    goldens for broad cognition prompt bytes, one L2 boundary-profile wording
    assertion, and older live/full-stack fixtures missing current
    `cognitive_episode` or `local_time_context` fields.
- Independent review remediation before re-check:
  - Removed the duplicated web generator `# 可用工具` block because tool schemas
    already come from `_generator_llm.bind_tools(_ALL_TOOLS)`.
  - Tightened `_WEB_SEARCH_GENERATOR_PROMPT` from the reviewed `1994` chars to
    `872` chars, below the review baseline of `892` chars.
  - Updated `experiments/prompt_input_format_ablation.py` fixtures so
    relevance cases include `current_run_context`, L1 includes
    `character_state`, and web generator/evaluator include `reference_time`.
  - Added deterministic payload assertions for web generator/evaluator
    `reference_time` and L1 `character_state`.
  - Replaced prose-only field coverage with the table above.
- Stage 7 independent code review evidence:
  - Review subagent: `019e5ec4-a7ce-7bc3-ab30-03e859263f8c`.
  - Initial findings:
    - High: `_WEB_SEARCH_GENERATOR_PROMPT` duplicated tool descriptions and
      grew beyond the original prompt-size baseline.
    - Medium: web/L1 experiment fixtures did not yet match final Human JSON
      payloads.
    - Low: field coverage was prose instead of the required table.
  - Fixes made:
    - Removed the generator tool-description block and left tool schemas in
      `_generator_llm.bind_tools(_ALL_TOOLS)`.
    - Tightened generator wording to `872` chars versus review baseline
      `892`.
    - Updated the experiment harness fixtures for final payload fields.
    - Added deterministic payload tests for web `reference_time` and L1
      `character_state`.
    - Added the field-coverage table in Stage 5 evidence.
  - Rerun commands:
    `venv\Scripts\python -m py_compile experiments/prompt_input_format_ablation.py src/kazusa_ai_chatbot/rag/web_search_agent.py tests/test_web_search_agent.py tests/test_cognition_prompt_contract_text.py`;
    `venv\Scripts\python -m pytest tests/test_web_search_agent.py tests/test_cognition_prompt_contract_text.py::test_l1_subconscious_payload_passes_character_state_in_human_json -q`;
    focused deterministic suite listed above;
    static greps listed above;
    `git diff --check`.
  - Re-review result: no remaining blocking findings. All three findings were
    resolved. Stage 7 approved.
  - Residual risk: re-review did not rerun live LLM checks; it accepted the
    existing live evidence plus deterministic remediation checks because the
    fixes tightened prompt size and corrected payload placement without adding
    behavior.
- Post-completion cleanup and review correction:
  - User requested cleanup of the generated experiment harness under
    `experiments/` after completion.
  - Removed local ignored file `experiments/prompt_input_format_ablation.py`
    and generated bytecode
    `experiments/__pycache__/prompt_input_format_ablation.cpython-313.pyc`.
  - Independent cleanup review found no high/blocking findings. It requested
    clearer private relevance prompt text for `user_message.user_name`,
    `platform_user_id`, `content`, and `channel_name`; the prompt and focused
    test were updated accordingly.
