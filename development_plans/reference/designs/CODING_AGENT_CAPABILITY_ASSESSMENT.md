# Kazusa Coding Agent — Generic-Capability Assessment

## Document Control

- Owning area: coding-agent architecture evaluation
- Applies to: `src/kazusa_ai_chatbot/coding_agent/**` and its upstream call
  chain (`cognition_chain_core` L2d → `action_spec` → `accepted_task` →
  `background_work` → coding-agent worker adapter)
- Source evidence: full code read of the coding-agent subpackages
  (~25,400 lines) plus the upstream cognition/action/background-work path
- Design context supplied by the owner: (a) the coding agent is a subagent
  under the greater cognition loop, not a user-facing tool; (b) the
  PM/programmer/synthesizer split exists to minimize dependency on LLM
  capability — models are generic (not coding-tuned) and swappable, and LLM
  performance must stay decoupled from the deterministic workflow;
  (c) continuous user↔agent IO is not a goal today but should be reservable
- Status: revision 2 (2026-07-09) — supersedes revision 1's conclusions
  where they conflict

## Purpose

Answer four questions about the coding agent:

1. Can it handle generic coding requests like a standard coding agent
   (setting aside the Python-only language focus)?
2. What is the gap versus generic agents such as Codex CLI, OpenCode,
   Claude Code, Aider, and OpenHands?
3. Does the "distributed roles, trade smartness for time" justification hold
   for the constraint (local generic LLM under 35B, under 50k context, must
   survive model swaps)? What would a from-scratch design look like?
4. How should the feedback loop be designed, who owns each loop, and how
   should the coding agent fit the greater agentic framework?

---

## 0. System context — where the coding agent actually sits

This assessment initially compared the coding agent to interactive CLI
agents. That is the wrong frame. The implemented system is:

```
user message
  → cognition subgraph  L1 (subconscious) → L2a/L2b (framing/boundary)
                        → L2c1/L2c2 (judgment/social) → L2d (action selection)
  → L2d emits semantic_action_requests   (capability: accepted_coding_task_request,
                                          decision ∈ start | status | revise_proposal |
                                          summarize | approve_and_verify | cancel)
  → persona supervisor materializes action_spec.v1 (deterministic; task_brief
    is the user's verbatim input, not model prose)
  → accepted_task row + background_work job (coding_agent_worker_payload.v1)
  → [same turn] character speaks an acknowledgement; job waits
  → background worker tick (skipped while primary interaction is busy)
      → adapter maps operation → start_coding_run / continue_coding_run /
        get_coding_run; CODING_AGENT_WORKSPACE_ROOT injected from config
  → result written back to the accepted-task row
  → delivery tick runs a FRESH cognition turn with artifact_text as input;
    the persona narrates the result to the user's channel
```

Load-bearing properties of this shape:

- **Everything is asynchronous and turn-based.** A coding operation runs
  atomically inside one worker tick; each user follow-up (status, revise,
  approve, cancel) is a new accepted task referencing the same
  `coding_run_ref`. There is no live session, by design.
- **Human approval is semantic, not mechanical.** There is no approval UI.
  When the user consents in chat, L2d chooses `decision=approve_and_verify`
  and the adapter *fabricates* the structured approval object
  (`approved=True`, `approved_by` from source scope or `"current_user"` —
  `background_work/subagent/coding_agent.py:518-543`). The coding-run
  supervisor then enforces that only a run in `awaiting_approval` advances.
  The integrity of "a human approved this" therefore rests entirely on L2d's
  interpretation of the user's message. (See §5 for a recommendation.)
- **Execution specs for approved verification can be LLM-planned from the
  user's prose** (`EXECUTION_SPEC_PLANNER_PROMPT` in the adapter), defaulting
  to `python_compileall .`. Verification scope is thus decided far from the
  proposal that is being verified. (See §4 for a recommendation.)
- **The cognition layer never parses coding results for control flow.** It
  stores `artifact_text` and re-narrates it. The structured
  `worker_metadata.v2` (evidence refs, patch summaries, attempts,
  `allowed_next_actions`) is available but today mainly informational.

---

## 1. Is it a generic coding agent today?

**No — and given §0, the right question is narrower: is it a generic
*coding capability* for the cognition loop?** Even by that measure the band
is narrow. The language focus (Python) is the smaller half of the gap; the
structural limits dominate.

### What it can do (implemented and verified in code)

| Capability | Where | Notes |
|---|---|---|
| Answer questions about a codebase | `code_reading` | PM → programmer-wave → synthesizer; ripgrep-backed evidence; citation-enforced answers |
| Propose a new multi-file project | `code_writing` | Full-file generation via contract-isolated programmer calls; up to 32 new files |
| Propose bounded edits to existing files | `code_modifying` + `code_patching` | Exact-anchor structured operations compiled into unified diffs |
| Statically validate proposals | `code_patching/patch_validation.py` | Sandbox `git apply` + `ast.parse` + symbol/import resolution + test-quality heuristics |
| Apply an approved patch to a managed copy | `code_patching/apply.py` | Never mutates the live tree; requires structured approval + source-identity match |
| Run verification | `code_executing` | Exactly two tools: `python_compileall`, focused `pytest`; argv-only, ≤60s, scrubbed env |
| Verify-and-repair | `code_verifying` | Max 2 repair attempts on ≤4k chars of redacted failure summary; pytest selector paths protected from edits |
| Durable, resumable runs | `coding_run` | JSON ledger + JSONL events; `awaiting_approval` gate between propose and verify |
| Fetch sources | `code_fetching` | Public GitHub, raw files, local checkouts (public origin), inline pasted code |
| Web evidence | `external_evidence.py` | PM-requested lookups via `WebAgent3` |

### What it structurally cannot do (independent of language)

1. **No mid-task observation.** No LLM role can request an action and see
   its result within the same workflow phase; retrieval and execution are
   deterministic code sequenced around the LLM calls. The one exception is
   the reading PM, which sees prior programmer reports between waves.
2. **No arbitrary command execution** — closed two-tool enum, argv-only.
3. **No dependency installation** — missing packages are terminal failures.
4. **No file delete or rename**; `code_modifying` cannot create files and
   `code_writing` cannot edit them, so "add a module and wire it into
   existing callers" is not expressible as one proposal.
5. **Bounded exploration that refuses rather than scales** — 3 reading
   waves, 6 programmer reports, ≤10 modification context files; broad
   requests return "narrow the scope."
6. **No git workflow output** — diffs as artifacts, no branch/commit/PR.
7. **Public GitHub only** — no auth, SSH, private hosts.
8. **No cross-run memory or repo index** — the repo map is regex-rebuilt
   per run.
9. **Weak semantic self-check** — the artifact-alignment judge
   (`code_writing/acceptance.py:evaluate_artifact_alignment`) is implemented
   but never called; shipped validation is purely structural.

### Verdict

Within its band — codebase Q&A, small greenfield Python artifacts, small
anchored edits, approval-gated focused verification — it works and is
unusually safe and auditable. Outside that band (multi-step tasks that need
iteration against real execution output, mixed create+edit changes,
refactors, tooling/dependency work), it cannot complete the task, and its
caps make it refuse rather than degrade.

---

## 2. Gap versus generic coding agents

Generic agents (Codex CLI, OpenCode, Claude Code, Aider, OpenHands) share
one architecture: a model in a loop that calls tools, observes real
results, and decides the next action until done. Kazusa inverts this:
deterministic code decides every action; the LLM fills structured blanks.

| Dimension | Generic agents | Kazusa coding agent |
|---|---|---|
| Control flow | Model-driven loop; plan emerges from observations | Fixed feed-forward pipeline; the plan *is* the pipeline |
| Feedback during generation | Runs code/tests mid-task, iterates on real output | Static checks only; execution happens once, post-approval, ≤2 repairs on redacted summaries |
| Exploration | Unlimited on-demand grep/read | Pre-collected evidence bundles under fixed caps |
| Edit expressiveness | Create/edit/delete/rename, whole-tree refactors | Anchored edits to ≤8 pre-selected paths OR new files, never both |
| Shell / dependencies | Arbitrary (sandboxed), installs deps | Two allowlisted tools, no installs |
| Iteration budget | Until done (token-budgeted) | Hard stage caps regardless of task state |
| Model requirements | Tool-calling-tuned models, usually frontier | **Any generic instruct model** — strict-JSON single calls, retry-once, graceful degradation |
| Deployment shape | Interactive session with a human operator | Async subagent under a cognition loop, persona-narrated results |
| Safety/audit | Conventions + sandboxes | **Stronger**: path containment, redaction everywhere, approval state machine, protected test paths, event-sourced ledgers |

Two corrections to the naive comparison, given the design context:

- **The model-portability row is a real differentiator, not a deficiency.**
  Every generic agent above implicitly assumes a model trained for agentic
  tool use. Kazusa's contract — one bounded JSON decision per call,
  normalize, retry once, degrade — is close to the weakest capability
  assumption an orchestrator can make. That is a legitimate requirement the
  generic agents simply do not have.
- **The deployment-shape row is a choice, not a gap.** The async
  accepted-task lifecycle is the correct shape for a persona chatbot that
  must keep talking while work happens. The genuine gaps are the first
  five rows — above all **feedback during generation**, which is where the
  fix rate of every generic agent comes from and which does not depend on
  deployment shape at all.

---

## 3. Does the justification hold? What would a from-scratch design be?

The stated justification, now with the owner's clarification: local generic
(non-coding-tuned) model under 35B and under 50k context; the role split
minimizes dependency on any one model's capability and decouples LLM
performance from the deterministic workflow, so a future model swap does not
severely impact the system; the accepted cost is time.

### What the justification gets right — more than revision 1 credited

- **The per-call contract is the right model-agnostic interface.** One
  system prompt + one bounded JSON payload + strict parsing + one retry +
  graceful degradation works on any instruct-following model. Native
  function-calling, long agentic traces, and self-managed context — what
  generic agents demand — vary wildly across generic local models. If
  model-swap resilience is a hard requirement, *the shape of each LLM call*
  in this codebase is correct.
- **Deterministic compensation is the true decoupling layer.** Anchor
  exactness, AST symbol/import validation, ungrounded-term scrubbing,
  sandbox `git apply` — these convert "model quality" problems into
  "bounded retry" problems, and they transfer unchanged across models.
- **Small prompts per call** (~42k-token hard cap, per-field truncation)
  correctly serve the 50k window.
- **Determinism, auditability, containment** — differentiated assets.

### Where the justification still breaks down

The critical observation: **model-portability and pipeline-vs-loop are
orthogonal.** The design conflates "each LLM call must be simple and
structured" (correct, forced by the model constraint) with "the *sequence*
of calls must be fixed in advance" (not forced by anything). A loop can be
built from exactly the same model-agnostic primitive: the model returns one
strict-JSON *action* per call, deterministic code executes it, and the next
call includes the observation. That is a ReAct loop without native tool
calling — same weakest-assumption contract, but the sequence is now driven
by observations instead of by the pipeline.

Concretely, what the fixed sequence costs:

1. **The feedback loop is amputated, and small generic models need it
   most.** A generic 30B model's zero-shot patch correctness is low; its
   ability to fix code *given the actual failure output* is much higher.
   Today the generation path never sees a real traceback; repair sees ≤4k
   redacted chars, at most twice, only after human approval.
2. **Information dies at every role boundary.** Reports are compacted to
   bounded facts (800-char ledger texts, 450-char excerpts). Each hop is a
   lossy summary of a lossy summary; the PM reasons through a keyhole.
   Role decomposition *multiplies* the small-context penalty on tasks that
   need cross-step state.
3. **The time trade never pays off, because caps stop the clock.** Worst
   case ~11 LLM calls for reading, ~9+ for writing, up to 300s each —
   then the pipeline stops at its stage caps whether or not the task is
   done. Time is spent; convergence is not bought.
4. **Decoupling is only partially achieved anyway.** The role prompts embed
   heavy behavioral assumptions (JSON escaping rules, Python idioms,
   "double-escape `\n` inside JSON" coaching) that were evidently tuned
   against specific model failure modes. A swap still changes behavior; the
   validators, not the role split, are what actually absorb it.
5. **Observable drift confirms the architecture is expensive to maintain:**
   the never-called alignment judge, `create_child_pm` declared but
   unhandled in `code_modifying`, dead helpers in its supervisor that would
   `NameError` if invoked.

### From-scratch design under the same constraints

Keep the model-agnostic call contract; free the sequence. Under the
constraint "generic model, <35B, <50k, swappable":

1. **Keep unchanged:** `code_fetching`, the managed apply workspace,
   `coding_run` ledgers and approval state machine, path containment,
   redaction, and all deterministic validators — repurposed as guards on
   loop actions rather than pipeline stages.
2. **Replace the inner PM/programmer pipelines with a JSON-action loop:**
   each turn, the model receives (task, notes file, last-N bounded
   observations) and returns one strict-JSON action from a closed enum —
   `read {path, range}`, `search {pattern, glob}`, `edit {path, anchor,
   replacement}` (plus `create`/`delete` kinds), `run_check {spec}`,
   `note {text}`, `done {summary}`. Deterministic code validates and
   executes; observations are truncated tool-side to fixed budgets before
   entering the next prompt. No native tool-calling required; parsing and
   retry-once machinery already exists in this codebase.
3. **Harness-owned context:** system prompt + repo map + model-maintained
   notes file + rolling window of bounded observations, evicted
   oldest-first. The notes file is the cross-step memory (replacing lossy
   role-to-role compaction with model-authored state).
4. **Budget by tokens/attempts, not stages,** so effort scales with task
   difficulty and stops on spend, not on an arbitrary wave count.
5. **Keep role decomposition for exactly one thing: read-only exploration
   fan-out** (`code_reading`'s shape is right for that — disposable
   contexts returning compact reports). Generation and repair belong in one
   continuous context.
6. **Model-swap resilience is preserved** because the per-call contract is
   unchanged; what a swap now affects is *loop efficiency* (more or fewer
   turns to converge), which degrades gracefully, instead of *stage
   quality*, which fails brittly.

---

## 4. Feedback-loop design — the detailed recommendation

Agreed premise: the missing feedback loop is the highest-leverage problem.
The design below is expressed as three nested loops with explicit owners,
plus a routing table saying which failure signal goes to which owner. It is
implementable inside the current pipeline architecture (it does not require
the §3 loop rewrite, though it becomes simpler with it).

### Principle: approval should gate *side effects toward the user's target*,
### not *the agent's ability to learn*

Today the sequence is: propose → validate statically → **human approval** →
apply to managed copy → execute → (≤2 repairs). But the managed apply
workspace and the scrubbed two-tool executor already make apply+execute
side-effect-free with respect to the real checkout — that containment was
built and is the system's best asset. The single most valuable change is to
move apply+execute **before** proposal delivery, so every proposal the
cognition layer receives has already been compiled and focus-tested, and
repair has already consumed real failure output. Approval then gates what it
should: delivering/applying the change to the user's actual target and any
higher-privilege verification.

Trust note, stated honestly: pre-approval execution means running
LLM-generated code without a human in the loop. The existing mitigations
(managed copy outside the source root, scrubbed env with no inherited PATH,
no network step, argv-only closed tool set, 60s timeouts, protected
verification paths) are exactly the right ones and carry over unchanged.
Make pre-approval execution a deployment config flag
(`CODING_AGENT_PREFLIGHT_EXECUTION`) so a stricter deployment can keep
today's ordering.

### The three loops and their owners

**Loop A — mechanical repair (seconds; owner: the programmer role's caller,
i.e. `code_modifying`/`code_writing` supervisors).**
Signal: anchor-not-found / anchor-ambiguous, JSON contract violations,
`ast.parse` failures, placeholder-stub diagnostics.
This loop exists (`MAX_PROGRAMMER_CONTRACT_REPAIRS = 2`) and is correctly
owned. Improvements:
- Feed the *exact* validator message plus the ±20 lines around the intended
  anchor back into the reprompt (today the programmer gets error text but
  not refreshed file context around the failure point).
- Do not escalate Loop A failures to the PM; they are mechanical. Only
  after exhausting local retries should the supervisor surface a blocker.

**Loop B — execution repair (minutes; owner: a promoted `code_verifying`,
renamed conceptually to the "verification supervisor").**
Signal: `python_compileall` failures, failing pytest selectors, import
errors at runtime, timeouts.
This is the loop to build out. Design:

1. *When it runs:* immediately after static validation of any
   `propose_patch` objective (pre-approval, per the flag above), and again
   after approval exactly as today.
2. *What executes:* specs derived **deterministically from the proposal**,
   not from user prose — `python_compileall` over changed files' packages,
   plus pytest selectors chosen by mapping changed source paths to their
   test companions (the `_tested_source_paths` stem-mapping logic in
   `coding_agent/supervisor.py` already does this mapping for evidence
   ranking; reuse it for spec planning). The adapter's LLM spec planner
   (`EXECUTION_SPEC_PLANNER_PROMPT`) should become a fallback for
   user-requested extra verification only — spec planning belongs to the
   owner of the proposal, inside the coding agent, not in the background
   worker adapter.
3. *What the repair prompt sees — the "failure evidence bundle":* replace
   the current ≤4k-char redacted prose with a structured object, same shape
   discipline as reading evidence rows so it flows through existing PM/
   programmer contracts:
   ```json
   {
     "failure_id": "exec-1",
     "tool": "pytest",
     "selector": "tests/test_converter.py::test_scan_roundtrip",
     "failure_kind": "assertion | exception | import_error | timeout | compile_error",
     "exception_type": "KeyError",
     "exception_message": "'record_id'",
     "assertion_diff": "- expected: 3\n+ actual: 2",
     "trace_frames": [
       {"path": "src/converter.py", "line": 141, "function": "scan",
        "code_line": "row = index[record_id]"}
     ],
     "related_evidence_ids": ["evidence-4"]
   }
   ```
   Redaction still applies (strip roots, env values), but *structure* is
   preserved instead of flattened to prose. Trace frames give the modifying
   programmer the exact file:line to re-read — feed those files' bounded
   context into the repair task automatically (the repair-context priority
   ranking in `coding_agent/supervisor.py` already half-implements this).
4. *Loop budget:* replace the fixed `MAX_REPAIR_ATTEMPTS = 2` with a
   configurable attempt/token budget per run (default higher, e.g. 4–6
   attempts pre-approval where iteration is cheap and safe), with two
   deterministic stop rules: stop on repeated identical failure signature
   (no progress), and stop on regression (previously-passing spec now
   failing).
5. *Repair scope:* route repair to `code_modifying` with the failure bundle
   as `repair_feedback` (the plumbing exists — `repair_feedback` with
   `feedback_source: execution_verification` is already a first-class
   input); do **not** restart the whole PM tree per attempt. Protected
   verification paths remain read-only to repairs, exactly as today.
6. *Environment failures are not repairs:* a missing third-party dependency
   or absent interpreter is a **blocker**, not a repair signal (the agent
   cannot install anything). Classify `ModuleNotFoundError` for
   non-local modules as `failure_kind: environment` and route it to Loop C
   instead of burning repair attempts on it.

**Loop C — semantic/task-level repair (turns; owner: split between the
coding-run supervisor and the cognition layer).**
Signal: everything execution cannot catch — built the wrong thing, missed a
requirement, needs information only the user has, environment blockers.
Design:

1. *Wire in the alignment judge.* `evaluate_artifact_alignment` exists and
   is dead. Call it after Loop B succeeds: input = acceptance criteria +
   artifact manifest + validation/execution summary; output =
   `aligned | misaligned {missing_criteria}`. On `misaligned`, one bounded
   re-plan pass through the PM with the missing criteria as feedback — this
   is the semantic analogue of the existing validation-feedback pass.
   Owner: the writing/modifying supervisors.
2. *Structured blockers upward.* Extend `CodingRunResponse` blockers into a
   typed channel: `{"blocker_kind": "needs_user_input | environment |
   scope", "question": "...", "options": [...]}`. The worker adapter maps
   these into `worker_metadata`, and the delivery turn's persona can then
   *ask the user the actual question* instead of narrating a generic
   failure. The user's answer arrives as a new accepted task
   (`revise_proposal` with the answer in `task_brief`) against the same
   `coding_run_ref`. This turns the existing async lifecycle into the
   outer feedback loop — no continuous IO needed, but it reserves the slot
   for it (see §5).
3. *Owner boundary rule:* the coding agent owns loops whose signals are
   machine-checkable (A and B, plus alignment); the cognition layer owns
   loops whose resolution requires the user (questions, approvals, scope
   changes). The routing table below is the contract.

### Failure-signal routing table (the ownership contract)

| Signal | Detected by | Routed to | Loop |
|---|---|---|---|
| Anchor not found / ambiguous | `patch_operations` | Programmer reprompt with refreshed anchor context | A |
| JSON/contract violation, stub content | programmer diagnostics | Programmer reprompt | A |
| Syntax error, unresolved symbol/import (static) | `patch_validation` | Programmer contract repair | A |
| Compile failure, failing focused test, runtime exception | `code_executing` | `code_modifying` repair with failure evidence bundle | B |
| Repeated identical failure signature | verification supervisor | Stop; escalate as blocker with the bundle attached | B → C |
| Missing third-party dependency, missing interpreter | `code_executing` output classifier | Blocker (`environment`) to cognition; user decides | C |
| Artifacts don't satisfy acceptance criteria | alignment judge | One PM re-plan pass; then blocker if still misaligned | C |
| Ambiguous requirement, missing fact only user has | PM `request_information` | Typed `needs_user_input` blocker → persona asks user | C |
| User feedback on delivered proposal | L2d (`revise_proposal`) | New proposal attempt on same run | C |

---

## 5. Fit within the greater agentic framework

The accepted-task shape (§0) is right for this system and should be kept.
The recommendations below sharpen the seams rather than reshape them.

1. **Keep the coding agent a durable specialist service with a semantic
   API.** The upward contract — six operations, `coding_run_ref`, redacted
   public projections, `allowed_next_actions` — is a good subagent
   interface. Generic-coding capability should grow *behind* this contract
   (per §4), not by exposing the cognition layer to coding internals.

2. **Make `allowed_next_actions` and typed blockers the affordance
   channel.** Today the cognition layer treats coding results as narration
   input. The metadata already carries `allowed_next_actions`; L2d's action
   router should receive them as the authoritative affordance list for the
   next turn (so the persona offers "approve / revise / cancel" only when
   the run is actually `awaiting_approval`), and typed blockers (§4 Loop C)
   should become questions the persona relays verbatim. This closes the
   outer loop without any new interaction machinery.

3. **Strengthen the approval evidence chain.** The adapter currently
   fabricates `approved=True` from an L2d decision; the human-consent link
   is a model inference. Two cheap hardenings: (a) carry the triggering
   user message id and a bounded verbatim quote into the approval object
   (`approval_evidence: {message_id, quote}`) so the run ledger records
   *what the user actually said*; (b) have the action-spec validator
   require that an `approve_and_verify` spec originates from a
   `user_message` trigger (already enforced) **and** that the quote is
   non-empty. This keeps approval semantic (per the design intent) while
   making it auditable after the fact.

4. **Reserve — don't build — continuous IO.** The run ledger + events file
   is already a session substrate: any future interactive surface (a CLI, a
   web view, a richer chat mode) can attach to the same `coding_run_ref`,
   read `events.jsonl`, and issue the same six operations. To reserve the
   capability, do two things now: (a) keep every new feature (failure
   bundles, blockers, alignment verdicts) as *events on the ledger* rather
   than transient response fields; (b) add one run action `respond_to_blocker`
   (symmetric to `revise_proposal`) so a future surface — or today's
   persona turn — can answer a typed question mid-lifecycle. Nothing else
   is needed until a live surface exists.

5. **One writer per checkout.** When Loop B moves execution earlier and
   budgets rise, background ticks may run longer. The runtime already skips
   ticks while the primary interaction is busy; add per-`coding_run` and
   per-source-identity locking (the ledger is the natural lock owner) so
   concurrent accepted tasks against the same repo serialize instead of
   racing managed copies.

6. **Benchmark at the seam.** Build a fixed internal benchmark (~30–50
   tasks: bug fixes, small features, mixed create+edit changes on pinned
   repos) driven through `start_coding_run`/`continue_coding_run` — the
   real seam — scoring end-state test pass rate, wall time, and LLM-call
   count. This makes "trades smartness for time" and every §4 change
   falsifiable, and it directly measures model-swap resilience: rerun the
   suite per candidate model.

---

## 6. Revised roadmap

Ordered by leverage; each phase is independently shippable and none weakens
the safety shell.

**Phase 0 — hygiene (days).** Wire in or delete the alignment judge; handle
or remove `create_child_pm` in `code_modifying`; remove the dead
supervisor helpers; allow `create_file` (and delete/rename) operations in
`code_modifying` so mixed changes are one proposal
(`compile_patch_operations` already supports `create_file`).

**Phase 1 — Loop B (weeks).** Pre-approval apply+execute behind a config
flag; deterministic spec derivation from changed files (move spec planning
out of the worker adapter); structured failure evidence bundles; budgeted
repair with no-progress/regression stop rules; environment-failure
classification. This is the single highest-leverage change and touches no
trust boundary that containment doesn't already cover.

**Phase 2 — Loop C and framework seams (weeks).** Typed blockers +
`respond_to_blocker`; `allowed_next_actions` into L2d affordances; approval
evidence chain; per-run/source locking; the benchmark harness.

**Phase 3 — inner-loop evolution (1–2 months, benchmark-gated).** Implement
the §3 JSON-action loop as an alternative engine behind the same
`coding_run` contract; route `verify_repair` objectives to it first, then
`propose_patch` once it beats the pipeline on the benchmark. Keep the
pipeline as the fallback engine — the model-agnostic per-call contract is
identical, so both engines share validators, executor, ledger, and models.

**Phase 4 — breadth (as needed).** Tree-sitter repo map and symbol
extraction; per-language executor plugins (still argv-only allowlist:
`npm test`, `go test`, `cargo test`); dependency installation as an
approval-gated structured step; authenticated/private sources; optional
branch/commit output in the managed copy.

### What not to change

Path containment, output redaction, the approval state machine, protected
verification paths, the event-sourced ledger, the strict-JSON
single-decision-per-call model contract, and the async accepted-task
lifecycle. These are the system's differentiated assets; every
recommendation above is designed to fit inside them.
