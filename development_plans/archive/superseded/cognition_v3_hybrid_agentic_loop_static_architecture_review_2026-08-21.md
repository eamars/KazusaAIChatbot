# Cognition V3 Static Architectural Review — 2026-08-21

## Document control

- **Status:** Superseded historical review. Its findings informed the
  completed handleless cognition cutover; the reviewed hybrid-chain design is
  no longer the production cognition architecture.
- **Document type:** Architectural alignment review — input to the Gate 7
  consolidated failure-mode register required by the active plan and the
  2026-08-21 handover.
- **Reviewed implementation:** branch `feature/cognition_core_v3_cache_affine`,
  worktree state at review time (last commit `186d954e`, dirty worktree
  preserved).
- **Reviewed against:**
  - `docs/architecture/cognition_v3_hybrid_agentic_loop_architecture.md`
    (the independent target architecture);
  - `development_plans/active/short_term/cognition_v3_hybrid_agentic_loop_reconciliation_plan.md`
    (Confirmed Decisions 1–49, Target State, Contracts And Data Shapes,
    Runtime Or Resource Constraints, Cutover Policy, governing amendments);
  - `development_plans/active/short_term/cognition_v3_hybrid_agentic_loop_reconciliation_plan_handover_2026-08-21.md`.
- **Method:** direct read of `anchor.py`, `registry.py`, `contracts.py`,
  `__init__.py`, `prompt.py`, `execution.py`, `transcript.py`, `budget.py`,
  plus four parallel reviewer passes over `facade.py`, the semantic-owner
  modules (`appraisal.py`, `goal_cognition.py`, `workspace.py`,
  `action_selection.py`), the lane/session layer (`lane.py`, `session.py`,
  `subconscious.py`, `diagnostics.py`), and the integration surface outside
  the package. Load-bearing claims (dead call sites, I1 omission, budget
  wiring, observability producers) were independently re-verified by
  repository-wide grep. Line numbers refer to the worktree state above.
- **Execution authority:** none. Remediation stays governed by the active
  plan's owner-group process; this document records findings only.

## Overall verdict

The reconciled single-chain **shell is real and largely faithful**: one
byte-stable system anchor, volatility-ordered first packet, append-only
alternating transcript on a FIFO-serialized primary lane, tail-rollback
repair that never persists rejected drafts, sidecar isolation for
L1/repair/X1/X2 with correct admission arithmetic, and a session registry
with mostly correct reattachment gating. The integration surface (config
loaders, selector, connector injection, guardrail genericization,
JSON-repair injection, DB/event/console contracts) is substantively
plan-conformant.

The implementation diverges from the architecture in four load-bearing ways:

1. **The cold path is not the canonical chain** — no I1/I2 interludes; goal,
   workspace, and planning stages run on pre-appraisal state; the P1 question
   is built before any bid exists. The recurrence path implements the correct
   ordering, so cold and recurrence are two semantically divergent
   implementations of the same contract.
2. **The context-budget / serving-window / re-anchor subsystem is dead code**
   — implemented as a library with zero runtime call sites.
3. **Engine-side observability is unwired** — no `cognition_chain_run.v1`
   document, `cognition_chain` event, or protected chain transcript is ever
   produced by a live invocation; only the consumers exist.
4. **The semantic-owner modules fork the canonical V2 validators** the plan
   explicitly required to be imported, with real behavioral drift.

Both Gate 7 hard-failure clusters have deterministic static root causes
(section 1).

---

## 1. Statically confirmed root causes of the Gate 7 hard failures

### 1a. `PromptContractError: private metadata field 'entity_id'`

Case `goal_completion_terminalization`, trial 1, failure at the 5th model
call (= W1). Three-way contradiction, fully deterministic:

- `workspace.py:122,185` — `validate_complete_bids` **requires**
  `goal_ref.entity_id` and `target_roles[].entity_id` on every admitted bid;
  `facade.py:327–337` (`_materialize_goal_bid`) writes
  `goal_ref = {scope, kind, entity_id}` into every bid.
- `prompt.py:810–815` — `build_workspace_question_payload` copies bids
  **verbatim** into the W1 question payload and is the only question builder
  that neither sanitizes (`_sanitize_prompt_value`, used by the P1 builder at
  `prompt.py:831`) nor projects. `goal_contexts` travels unsanitized through
  the same payload.
- `prompt.py:86,199–202` — `ChainQuestion.__post_init__` rejects `entity_id`
  with exactly the observed error text.

**Every turn that reaches W1 with ≥2 non-sensitive complete bids must
hard-fail.** A safe projection already exists and is bypassed:
`workspace.py:288–304` (`prepare_partition.prompt_payload`) projects only
branch_id/intention/desired_outcome/reason, but the facade consumes only its
`handles` and feeds raw bids into the question instead.

**Fix direction:** not sanitization. Architecture §5.1 and the plan's Target
State step registry define W1's input as "complete bids **already present**
[in the transcript]; no re-serialization" — the W1 question should be
handle-only. Do not weaken the private-field check.

### 1b. `SessionContractError: output expected_previous_state does not match cold mutable_state`

Case `multi_goal_competition`, all three trials, after the full 9-call chain
at session-store time:

- `session.py:297–305` (`create_cold_session`) compares
  `output.state_update["expected_previous_state"] != payload["mutable_state"]`
  with **naive Python `!=`**.
- `expected_previous_state` is produced by V2 `build_state_update` →
  `_canonicalize_replacement_state`
  (`cognition_core_v2/output_projection.py:180–210`), which **sorts**
  goals/threats/events/knowledge_gaps by `(created_at, entity_id)` and
  reorders affect activations. The incoming `mutable_state` is raw and
  unsorted.
- Any multi-entity state not already in canonical sort order fails
  deterministically — exactly the multi-goal fixture; single/zero-entity
  states sort to identity and pass. The same naive compare sits latent at
  `session.py:478–483` (`advance_session_after_output`), currently masked
  because reattached inputs echo the already-sorted replacement state.

**Fix direction:** the plan's recurrence table (~line 1102) demands
*canonical value* equality — normalize both sides (apply
`_canonicalize_replacement_state` or compare canonical digests of normalized
values). Do not weaken the check.

### 1c. Subjective-identity cluster (handover `self`/`我`/`current_user` audit)

- `prompt.py:607` — `build_first_packet_sections` hardcodes
  `scene_section["participant_bindings"] = []`, even for multi-participant
  scenes. The engine manual (`anchor.py:20`) tells the model that `pN`
  handles and the exact allowed sets come from the payload — which never
  supplies them.
- `prompt.py:711–724` — the grouped appraisal question carries only
  family / evidence handles / permitted delta paths / question text: **no
  permitted subject/object/role-handle domains, no role bindings**. Role
  domains are enforced only post-hoc (`appraisal.py:825`), so the model
  guesses subjects blind → repair churn and the observed
  third-party-typed-as-`self` inversion (retained local reset).
- Goal questions *do* carry `role_bindings` + role summaries
  (`prompt.py:765–766`) — the A-stage/G-stage information discontinuity
  hypothesized by the handover is real.
- Cold G1b question is `{"branch_roster": ...}` only (`prompt.py:843` via
  `facade.py:2842–2861`) — no evidence handles or semantic context, unlike
  the recurrence G1b (`facade.py:2155–2159`), while the validator rejects
  out-of-domain handles.

### 1d. Sidecar-overlap node (`l1_dropped_count=20`, overlap 0/20)

Consistent with the sibling payload gaps above, but the concurrency cause was
not conclusively established statically: `subconscious.py` is contract-clean
and lane admission arithmetic is correct. This node needs the artifact
correlation the handover prescribes before an owner is assigned.

---

## 2. High-severity architecture mismatches

### 2a. The cold path is not the plan's chain (Decision 6; architecture §5.1/§5.3)

- **No I1 interlude anywhere.** `_build_i1_notice` (`facade.py:487`) has
  **zero call sites** (grep-verified) — the ≤600-char state-transition notice
  never reaches the model on *any* path, cold or recurrence.
- **Cold deterministic reduction runs after P1**, not between A2 and G1a:
  `_reduce_appraisals_with_isolation` is called at `facade.py:3112` after all
  model steps; goal/workspace payloads are built from `preliminary_state`
  (`facade.py:1169, 2909, 2988`). This falsifies the architecture's §5.3
  claim that "emotion state is derived deterministically at I1, before any
  bid exists" — cold bids are conditioned on pre-appraisal state.
- **No I2 in the cold path**: the live-goal filter runs after P1
  (`facade.py:3153`); W1 eligibility gates on raw bid count
  (`facade.py:2980`), not admitted candidates. A stale-goal bid can trigger a
  W1 call that I2 would have suppressed.
- **Cold P1 is prebuilt before any bid exists** (`facade.py:1272–1286`:
  `primary_bid=None, supporting_bids=[]`, sent verbatim at 3093); collapse
  happens *after* P1 (3169); the P1 validator's handle domain contains only
  the ordinary bid (`facade.py:3038–3040`) — branch bids can never legally
  carry action requests, and authorization (3205) can pair a W1-selected
  primary with a `bid_handles` map that does not contain it.
- **The recurrence path implements the correct ordering** (reduction at
  1971–1992 before bid revision; `i2_interlude` at 2214–2348; collapse before
  P1 at 2269; full P1 payload with all eligible bid handles at 2276–2303).
- The cold output tail duplicates `_build_serial_output` (~250 lines,
  `facade.py:1378–1636` vs `3105–3445`) with drift already visible
  (stage-status timing, synthesized-execution bid membership, independent
  collapse recomputation).

### 2b. Budget / serving-window / re-anchor subsystem is dead code (Decisions 42–43; architecture §9.5, §11; goal G7)

Grep-verified package-wide: `ContextBudgetLedger.admit()`,
`consume_reanchor()`, `ChainTranscriptV1.reanchor()`, and
`estimate_message_tokens()` have **zero call sites** in the engine (the
estimator is used only by `src/scripts/probe_cognition_v3_context_overflow.py`).
The facade constructs the ledger (`facade.py:1720, 2741`) and reads only
`extension_used` for diagnostics (2394, 3417). Consequences:

- No 50,000-token admission check before any primary request
  (`execution.py:570` invokes with no estimate).
- The 65,000 extension can never activate — Gate 7's "no fixed candidate
  required activating the 65,000 tier" is vacuously true.
- `extension_used`/`reanchor_used` diagnostics can never be true; the
  re-anchor degradation ladder (§11.2) and degeneration recovery (§9.8) are
  unreachable.
- The §9.5 client-side refusal — the defense against what the architecture
  calls "the most dangerous failure in this design" — does not exist at
  runtime, and the Gate 7 overflow probe simultaneously showed the serving
  layer *accepting* an over-window request. Both layers of the
  silent-truncation defense are currently absent/unproven.
- `budget.py:57` deviates from the plan's fixed estimator formula:
  `non_cjk_bytes = utf8_bytes − cjk_codepoint_count` subtracts the codepoint
  *count* instead of CJK *bytes* (~3 per codepoint), inflating pure-CJK
  estimates ~1.5× (conservative direction, but it will burn the 50k budget
  faster once admission is wired). `CALIBRATION_MULTIPLIER = 1.00` sits at
  the plan's floor with no runtime consumer.

### 2c. Engine-side observability producer is unwired (plan Contracts; architecture §14; Decision 6 `status=skipped`)

Grep-verified: `save_cognition_chain_run`, `record_cognition_chain_event`,
and `record_cognition_chain_transcript` have **zero production call sites**
(definitions, re-exports, READMEs, and console consumers only). No
`cognition_chain_run.v1` document, no `cognition_chain` event, and no
protected chain-transcript row is ever produced by a live V3 invocation;
per-step records (cache_class, prompt_chars, new_suffix_chars, skipped
statuses) have no carrier; `diagnostics.py` is not imported by the facade.
The Gate 5 consumers (db helpers, brain-service fields, console panel — each
individually plan-conformant) were accepted against an engine that never
feeds them. **These are the metrics goal G2 (cache affinity) is judged by** —
the performance gate currently has no engine-side prefix-share evidence
channel.

### 2d. Grouped-appraisal failure ladder missing (Decision 15; architecture §9.3)

On a grouped step returning `None`, the facade marks all planned families
exhausted (`facade.py:2834–2839`). The retry-once-then-split-to-finer-
partition (2→3→6) logic exists nowhere in `facade.py` or `execution.py`.
One structural A1 failure silently loses three families — the widened blast
radius §9.3 was written to prevent.

---

## 3. Plan-decision violations (medium severity)

### 3a. Canonical shared-helper rule violated five times (plan lines 390–411)

The plan requires importing renamed canonical V2 owners with "no copied
validator … no second vocabulary". The V2 side of the extraction is clean
(all seven canonical names public, call sites renamed, no underscore twins;
the appraisal path imports canonical helpers correctly). The V3 side forks:

| V3 fork | Canonical V2 owner (exists, verified) | Drift |
|---|---|---|
| `workspace.py:308` `validate_partition` | `v2.workspace.validate_workspace_partition:434` | line-for-line copy |
| `workspace.py:421` `collapse_authoritative_relational_bid` | `v2.workspace:53` | brittle willingness equality (`schema_version` presence asymmetry, line 462) |
| `goal_cognition.py:252` `validate_goal_bid_draft` | `v2.goal_cognition:1666` | skips `_normalize_nonowning_goal_fields`; **never passes `episode_handles`** — the manual's own "at least one evidence handle from the current episode" willingness rule (`anchor.py:62`) is unenforced on the non-selection G1a path |
| `action_selection.py:886` `validate_action_plan_decision` | `v2.action_selection:951` | no exact top-level field check (unknown extras pass silently, contradicting the manual's "恰好是" rule); invalid request rows **silently dropped/truncated to 3** instead of entering tail-rollback repair — a second disposition vocabulary vs Decision 39 |
| `action_selection.py:2347` `_validate_authorization_decisions` + prompts at 2439/2609 | `v2.action_authorization.validate_authorization_decisions:443` + V2 prompt constants | prompts are byte-identical **copies** today (verified); silent drift now possible; plan named these for reuse explicitly |

### 3b. Repair policy (Decision 39; architecture §9.1–9.2)

- `execution.py:557–558` — appendix is last-error-only and untyped:
  `f"{question_text}\n[contract_repair]\n{last_error}"` replaces rather than
  accumulates (not monotonic), carries no attempt index, typed error code,
  exact field path, or permitted handles — only `str(exc)`.
- No identical-retry or two-consecutive-empty short circuit anywhere.
- `execution.py:559–562` — first attempt mislabeled `.repair1` in
  `stage_name` telemetry.
- `execution.py:630` — repaired accepted rows serialized without compact
  separators (non-canonical encoding for an accepted transcript row).

### 3c. Session reattachment gaps (Decisions 30–36; recurrence table)

- Three cycle-carrier checks from the sealed table are absent in
  `session.py`: `resolver_goal_progress` equality with the prior output
  (including absence), `required_resolver_evidence_dependency` binding, and
  `pending_resolver_resume` binding to the newly appended evidence row
  (`session.py:344–436`).
- `session.py:408–424` — new evidence row admission checks neither
  `source_id` novelty/non-emptiness nor the next-canonical-`e<N>` handle rule
  (any novel string passes).
- `session.py:170–179` — digest primitive omits `allow_nan=False` and
  key-type rejection (plan: floats/non-string keys/non-finite fail before
  hashing).
- `session.py:351–357` — divergence records coarse
  `divergent_field="immutable_projection"`; plan requires the exact field
  name.
- `session.py:384` — evidence prefix compared with Python `==`, not
  row-for-row byte equality under canonical encoding.
- `session.py:55–80` — `ChainSessionV1` deviates from the sealed field list
  (no `schema_version` field; extra `expected_mutable_state`,
  `expected_relational_willingness`); `session.py:274` — magic
  `ttl_seconds=3600.0` fallback the plan does not sanction (callers compute
  the TTL formula; the fallback should fail closed).

### 3d. Decision 38 (attempt policy)

Coordinates use new vocabulary (`cold_appraisal_A1`,
`resolver_delta_appraisal_1`, `branch_id="cycle:N"`; `facade.py:886–890,
2829, 1958`) rather than stable V2 owner/branch coordinates, and grouped
steps reserve one attempt per group, not per family — ledger arithmetic no
longer matches the V2 schema. Reservation silently skips when no ambient
ledger context exists (`execution.py:545`), making "every model attempt
reserved" context-dependent.

### 3e. Decision 20 (per-branch exhaustion)

G1b exhaustion drops **all** branch bids as a group
(`facade.py:2954–2956`, `v3_chain_unavailable:active_goal_group`); the
per-branch omission with a canonical assistant projection of independently
validated sibling bids is not implemented, and no per-branch failures reach
observability (`facade.py:3323–3328` synthesizes with
`unavailable_kinds={}`).

### 3f. Decisions 1–2 (package public surface)

`cognition_core_v3/__init__.py` exports internal diagnostic types
(`StageFailure`, `StageResult`, failure-class constants) that Decision 2
says cease to be public, and omits `CognitionChainServicesV3` and all five
V2 re-exports. Wiring is unaffected — consumers import V2 directly
(`nodes/persona_supervisor2_l3_surface.py:41–42`,
`reflection_cycle/affect_settling.py:13`). **Plan defect:** two of the five
Decision-1 names (`validate_cognition_input`,
`validate_cognition_core_output`) never existed as V2 exports as written
(code has `validate_cognition_core_input`; the V2 `__init__` exports neither
validator) — realizing the decision requires a plan amendment or new V2
exports.

### 3g. Completion caps (Target State table)

At least the recurrence W1 call (`facade.py:2247`,
`replace(services.chain_lane, stage_name="R.W1")`) applies no
`max_completion_tokens=2048` override — the lane's 8192 rides through. A
facade-wide audit of the cap table (appraisal 4096, W1 2048, L1/X 1024) is
warranted.

### 3h. Minor

- `db/cognition_chain_runs.py:110–136` — validate→find→insert, not the
  specified atomic idempotent upsert (concurrent duplicate → unique-index
  failure → `False`); existing rows are not refreshed; validation failure
  returns `False` without the plan's sanitized local warning; array caps
  (steps ≤96, session_events ≤16, markers/warnings ≤32) unenforced.
- `lane.py:181–185` — release re-acquires the condition inside `finally`
  unshielded; a second cancellation during that acquire leaves `_owner` set
  permanently (lane deadlock). `lane.py:750–755` — L1 drain swallows only
  `CancelledError`; other L1 exceptions propagate into the repair admission
  path (contradicts Decision 28's "L1 never changes deterministic control
  flow" for this edge).
- `registry.py:43–47` — with `appraisal_group_count=3`, two steps share step
  id `"A1"` (ambiguous diagnostics; duplicate L1-join handling,
  `facade.py:2787–2798`). `prompt.py:834–840` —
  `build_serial_question_sequence`'s `zip` mislabels step ids when a group
  has no planned families (`build_grouped_appraisal_questions` skips empty
  groups while the id list does not).
- `prompt.py:224–238` — `_sanitize_prompt_value` re-declares the forbidden
  field set inline instead of reusing `_PRIVATE_RUNTIME_FIELDS` (equal today;
  drift hazard). Two sanitization dispositions coexist for the same fields:
  silent strip (first packet, P1) vs fail-closed rejection (`ChainQuestion`).
- `prompt.py:552–565` — `__all__` declared mid-file; nine public functions
  defined after it are unexported. `facade.py:123–135` imports five private
  underscore names from `action_selection` (misfactored live/dead split).
- Normalized first-packet evidence rows keep only
  handle/source_kind/semantic_summary (`prompt.py:609–617`) — no
  `provenance_role`, which the P1 `future_speak` contract
  (`anchor.py:81`) requires the model to match; potential
  contract-information gap to correlate against artifacts.

---

## 4. Stale pre-reconciliation layer (hygiene; active confusion risk)

The plan's Change Surface says "no production file is deleted", but a large
superseded design remains in place, exported, and in places armed:

- **Dead but armed:** `action_selection.py:2211–2345` retains the V2
  service-bound `invoke_semantic_authorizer` loop the plan explicitly forbids
  V3 from reusing — it reads `services.action_authorization_config`, a field
  that does not exist on `CognitionChainServicesV3` (AttributeError if hit),
  and is the *default* path of `authorize_action_requests` /
  `authorize_resolver_requests` when no executor is injected (the facade
  always injects one today). The dead fresh-boundary W1/P1 paths
  (`workspace.py:479–752`, `action_selection.py:1258–1989`) call
  `parse_llm_json_output` without `deterministic_only` or the injected
  sidecar pair — if ever invoked they would route through the global repair
  LLM the plan forbids for V3.
- **Dead with conflicting contracts:** ~70% of `appraisal.py` (185–726)
  implements the abandoned per-owner-chain design with an *incompatible*
  appraisal output contract and its own static system prompt;
  `goal_cognition.py:107–120, 655–926` (static goal prompt,
  question/repair/disposition builders); `transcript.py:209–426`
  (`TranscriptState` + `start_chain`/`extend_accepted`/`build_repair_request`/
  `start_fresh_from_checkpoint`/`domain_matches`/`fits_prompt_budget`);
  `contracts.py` `ChainCheckpoint`/`CacheDomainIdentity`; `execution.py`
  `run_serial_chain`/`run_serial_harness_step`/`SerialChainStep` and the
  validation-free `invoke_serial_question_sequence` (auto-accepts unvalidated
  output, re-serialized rather than raw bytes); all of `diagnostics.py`
  (three builders, no production callers). None has a production call site
  (grep-verified).
- **Latent contract trap:** dead `transcript.build_repair_request`
  (290–324) sends the rejected raw draft back to the provider — exactly what
  Decision 39 and V2 convention forbid; its survival invites reuse.
- **Stale documentation:** module docstrings of `__init__.py`, `facade.py`,
  `transcript.py`, and `diagnostics.py` still describe the abandoned
  "per-owner static system prompt" topology, contradicting both the code
  beneath them and the reconciled architecture. Duplicate constants and a
  dead sibling projection function remain in `goal_cognition.py`
  (93–95 vs 583–586; 498–530 vs 589–625). Typos persist in exported names
  (`APPRASAL_*`, `classify_appaisal_candidate`).

---

## 5. What conforms (verified, abbreviated)

Anchor byte-stability and dynamic-field rejection (`anchor.py`; no env reads,
no run ids, manual-then-identity ordering preserved under `sort_keys` via
single-key section lists); first-packet volatility ordering with full-packet
retention across cold repair attempts (`execution.py:508–530`); alternating
roles, no tool role, no assistant prefill; interlude notices prefixed to the
next user question; G1b runs on relationship-sensitive turns with the
deterministic ordinary-primary collapse and downstream effect denial
(`facade.py:2924, 3169–3186, 1045`); `answerable_now` resolver suppression
(1098–1105); no-sidecar dispositions (L1 skip, deterministic-only parse,
deny-all: 691–697, 2830, 1076–1078); L1 `task.done()`-only join at A1 then
G1a with cancel-on-miss (2788, 2844–2853); X1 before X2, both after P1 and
after L1 drain (1080/1110, 3200–3218); tail-rollback semantics (failed
answers never enter the transcript); lane registries, FIFO fairness,
admission limits, repair-preempts-L1, recursive-claim typed errors,
deadline-before-admission (`lane.py`); session keying/TTL-formula/LRU/
single-owner/concurrent-claim semantics (`session.py:95–167, 309–376`;
`facade.py:1639–1647, 2550–2614`); `subconscious.py` fully conformant to
`L1ResidueV1`; turn deadline checked before every chain/sidecar/L1/repair
request with owner-disposition degradation; output validated via
`validate_cognition_core_output` (`facade.py:3413`).

Integration surface: config loaders exact (selector closed {v2,v3} default
v2, parsed first; V3 bundle validation incl. window ≥50000, thinking
rejected, sidecar all-or-none, group count closed, deadline 30..600;
`api_key` repr=False); connector constructs configs inside the selected
branch and injects all three runtime values (2026-08-20 runtime-policy
injection amendment honored); no `os.getenv` anywhere in the V3 package;
`LLMCallConfig.context_window_tokens` appended as final defaulted field and
not transported to providers; route report labels `COGNITION_LLM`
`shared_non_core`; JSON-repair injection signatures exact with
both-or-neither pair rule and fail-closed `deterministic_only`; guardrail
`ServicesT` pass-through imports neither services dataclass; db read helper
exact-intersection with no global-latest fallback and all five indexes;
`record_cognition_chain_event` keyword-only with the exact allowed argument
list; brain-service graph fields resolved independently per graph;
`cognition_engine_descriptor.v1` exact with mechanical extended-tier
derivation; protected transcript full/metadata/off modes honored in
`llm_tracing`.

---

## 6. Recommended owner grouping for remediation

Ordered inputs to the plan's consolidated failure-mode register; each group
maps to the smallest production owner per the handover's process. This
document assigns no execution.

1. **W1 packet contract** (§1a): handle-only W1 question per architecture
   §5.1; keep the private-field rejection intact. Owner: W1 payload seam
   (`prompt.py` builder + facade W1 call sites).
2. **Session state equality** (§1b): canonical normalization on both sides at
   `session.py:297` and `:478`; close the §3c reattachment-table gaps in the
   same owner pass. Owner: `session.py`.
3. **Cold-path chain order** (§2a): wire the I1 notice + pre-goal reduction +
   I2 into the cold path; unify cold/recurrence through the shared tail. This
   is also the change that makes architecture §5.3's order-proof claim true.
   Owner: `facade.py` cold sequence.
4. **Identity information carriers** (§1c): real `participant_bindings`,
   permitted subject/object/role domains in the appraisal payload,
   evidence/context in cold G1b — one owner, since all three are "the manual
   points at a payload set the payload never carries." Correlate with the
   handover's listed raw candidate transcripts before finalizing the payload
   shape (per the handover's instruction not to infer the final fix from code
   observations alone).
5. **Budget and observability wiring** (§2b, §2c): both are "library exists,
   engine never calls it" — mechanical wiring plus the estimator-formula
   correction. Blocking for Gate 7's serving/overflow, cache/prefix, and
   performance thresholds regardless of semantic fixes.
6. **Grouped-appraisal fallback ladder** (§2d): Decision 15's 2→3→6 split
   retry. Owner: facade/execution appraisal step handling.
7. **Canonical-helper re-pointing and dead-layer excision** (§3a, §4): lower
   urgency for Gate 7 eligibility, but the `validate_goal_bid_draft`
   episode-evidence drift and the P1 silent row-dropping are behavioral, not
   cosmetic, and should ride with their semantic owners. Dead-layer removal
   requires a plan-surface decision (Change Surface currently says no
   production file is deleted).
8. **Repair-policy completeness** (§3b): monotonic typed appendices,
   identical/empty short-circuit, cap-table audit (§3g).

## 7. Process notes for the plan record

- **Plan defect:** Decision 1 names two V2 validator exports that never
  existed as written (§3f); realizing it needs a plan amendment or new V2
  exports.
- **Gate-evidence gap:** Gate 5 acceptance validated observability
  *consumers* without an engine *producer* (§2c); Gate 7's cache/prefix
  thresholds depend on that producer existing. The Gate 7 "65k tier never
  activated" observation is vacuous while `admit()` is unwired (§2b).
- **Confidence:** all zero-call-site claims (I1 notice, budget admission,
  re-anchor, observability producers, dead legacy layer) and both Gate 7
  root-cause seams were grep-verified by the root reviewer against the
  worktree. Line numbers are anchored to the reviewed worktree state and will
  shift with remediation edits. Two items are flagged PLAUSIBLE rather than
  confirmed: Decision 14 termination-semantics enforcement location
  (may live in V2 merge validators) and the sidecar-overlap concurrency cause
  (§1d).

No production or test source was edited while producing this review. This is
a documentation-only review record, and no documentation unit test was
created.
