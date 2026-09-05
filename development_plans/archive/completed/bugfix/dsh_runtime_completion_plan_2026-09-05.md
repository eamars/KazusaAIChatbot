# DSH integration purpose

- **Status:** completed.
- **Completed:** 2026-09-05; fresh real-model behavior and runtime recovery
  evidence reviewed by the implementing agent.
- **Scope:** the entire existing and new DSH-related codebase, regardless of
  authorship, commit history, or the current working-tree diff.

## What DSH is supposed to do

DSH gives the character brain the ability to carry out substantive tasks that
require reasoning, evidence gathering, and tool use beyond an ordinary chat
response. Its integration should make those abilities usable through natural
conversation while preserving the character's judgment and continuity.

- Understand an admitted task and use the actual model and available tools to
  carry it out. This includes investigation, information retrieval, workspace
  operations, and coding within the task's permissions.
- Use relevant local, conversational, remembered, media, and external evidence.
  Results must distinguish observed facts, uncertainty, missing information,
  partial progress, and completed work.
- Obtain character judgment when a task needs clarification or permission.
  Task execution must respect that judgment and the user's actual authority.
- Support work that finishes during the current conversation and work that
  continues in the background. Continued work must retain its purpose,
  context, evidence, and relationship to the originating conversation.
- Return useful results through the character's normal reasoning and visible
  response. Internal task machinery must not replace character expression.
- Keep ongoing work inspectable, controllable, and recoverable. Cancellation,
  interruption, restart, and communication failure must leave truthful state
  and preserve work that has legitimately completed.
- Deliver results to the correct audience without duplicate completion
  messages, invented success, secret exposure, or cross-conversation leakage.
- Operate as part of a platform-independent character brain, with clear
  responsibility for task execution, character judgment, and delivery.

## Owner requirements

**Probe-first engineering is the highest engineering priority.** The next
agent must apply
[probe-first-engineering](../../../../.agents/skills/probe-first-engineering/SKILL.md).
No unit tests may be written before the codebase is proven to work through
actual execution with real LLMs. Unit-test counts, mocks, static checks, and
historical green results cannot establish that proof.

The next agent must use real LLM tests to rediscover the purpose and intention
of the DSH integration and its architecture. Existing implementation choices
and test expectations are hypotheses to examine, rather than proof of the
intended design. All real LLM tests must be preserved.

High code standards remain required: clear responsibilities, understandable
contracts, truthful failures, controlled resource lifetimes, and correct
permission, persistence, and delivery behavior.

## Fresh investigation boundary

The next agent's understanding must come from the owner's current requirements,
the current codebase, and newly observed real-model behavior.

Past development plans, implementation narratives, review conclusions, test
matrices, failed-attempt prescriptions, and historical sign-off claims are
excluded from the task's decision context. They must not supply architecture,
acceptance assumptions, or instructions for the next agent. This plan carries
no obligations forward from those attempts.

## Intended outcome

The complete DSH integration demonstrably performs the purposes above with
real LLMs. Its architecture and actual behavior are understood from fresh
execution evidence, and its visible results are useful, grounded, and coherent
with the character.

## Fresh execution record — 2026-09-05

Implementation and review used the current code and newly captured runs. The
first real foreground run exposed a shutdown failure; real coding and research
then exposed task retry failures. Repairs followed those observations. The
three existing live behavior cases remain, with coding and public-source
research added. Focused deterministic regressions followed real-model success.

### Architecture established through execution

The debug adapter enters the normal Brain intake, RAG and cognition path.
Cognition requests substantive work through the task-resolution service. That
service owns the inline budget and durable task binding; the resolver owns
thread identity, leases and authenticated sidecar RPC. Installed DSH Standard
owns native execution and SQLite session continuation. Its signed semantic
gateway supplies scoped Brain evidence, and its interaction bridge obtains
actual cognition decisions for questions and permission requests.

An inline result returns to cognition and dialog. A checkpoint is promoted to
one accepted task and one background job; completion re-enters cognition and
normal adapter delivery. An internal DSH approval can advance cognition while
the originating turn is still running, so that turn's optimistic commit may
retry. Its already admitted task must survive that retry, including when the
new model output describes the same goal differently.

### Observed failures and repairs

| Observed failure | Repair and resulting boundary |
|---|---|
| Post-turn shutdown raised `NameError: _source_turn_ref`, then swallowed cancellation and exceeded the 45-second shutdown bound. | Reuse the existing source-turn mapper, reconcile/audit the interrupted write, and propagate cancellation so the worker exits. |
| A coding task successfully created and ran its program, then the parent cognition retry tried to insert its binding again. | Reuse the validated completed result for the same trusted source and goal identity. Preserve the original executed objective. |
| A replayed result was rejected because regenerated objective prose differed. | Bind the capability result by its trusted goal continuation and project the original executed objective with its evidence. A different goal remains invalid. |
| Research checkpointed and acquired a job; a cognition retry attempted new admission/promotion and faulted the existing work. | Replay the existing checkpoint and retain its accepted task and background job. Source/goal checks precede reuse. |
| Fresh signed activations intermittently failed as expired: Python issuance was 1 ms ahead of Node's Windows wall clock. | Allow at most 50 ms of future issuance at the sidecar receiver. Expiration, MAC, identity, scope and fencing checks remain exact. |
| Package-manager build failed on literal approval placeholders in `pnpm-workspace.yaml`. | Declare the required native install hooks and explicit decisions for packages without necessary builds. The pinned build succeeds. |
| Dialog removed the citation in a valid inline DSH result. | Extract bounded source URLs from the validated inline resolver result as well as delayed tool-result percepts. Existing URL normalization retains only task evidence sources. |
| DSH over-interpreted tense as review frequency, and later claimed source verification from search links/prior knowledge after HTTPS failed. | Require observed support in summary/findings, label inferences, and return partial for unverified source requirements. Explain the observed Windows TLS failure through the existing exact-command approval contract. |
| The live harness sampled a delivered callback before its durable receipt and concealed forced shutdown. | Wait for durable delivery, include deferred research/coding results, and fail an unclean Brain shutdown. |

### Fresh real-model evidence

Artifacts live under `test_artifacts/dsh_behavior_e2e/`. Each case retains exact
inputs, raw model/tool calls, result evidence, durable records, visible output,
timing and cleanup. These are real Brain, DSH, model-gateway, MongoDB and debug
callback runs with isolated synthetic identities/workspaces. They are not
scripted-model proofs. The configured routes used Gemma 4 for cognition/dialog
and Qwen 27B for DSH. Existing configured character/database startup was checked
separately from the isolated conversation fixtures.

| Case artifact | Reviewed outcome |
|---|---|
| `foreground_20260905T040439Z_187100d1` | Both release notes were actually read; the clarification returned Mira and the checksum prerequisite. Persisted conversation continuity was loaded on turn two. No operational failure or identifier leakage; clean shutdown. Elapsed 156.09 seconds. |
| `internal_20260905T032416Z_795dd37d` | Real cognition answered Mira and denied an unsupported rollout-success claim; signed durable decisions matched, with no visible internal message. |
| `deferred_20260905T032619Z_5af3f939` | Elevated to stable, evening to morning, Rowan retained, threshold explicitly unknown. One callback to the source channel, one delivery attempt, durable delivered state and clean shutdown. Any cadence reading in the final wording is explicitly an inference. |
| `workspace_20260905T040819Z_a522620e` | Actual `summarize.py` creation/execution, inspected `report.json` with numeric `row_count=2` and `total=27.5`, useful visible result, successful cognition retry and clean shutdown. Final-build repetition passed in 219.22 seconds. The initial successful coding proof was `workspace_20260905T031958Z_653219e8`. |
| `research_20260905T035421Z_93f9b221` | Actual official-page text contains the DictReader definition and comma delimiter default. Scoped history searches were empty. Internal approvals, inline checkpoint, promotion, parent cognition retry and background continuation completed; the correct answer and exact URL were delivered once, with clean shutdown. Elapsed 587.66 seconds. |

The research run `research_20260905T034942Z_faec3f18` passed technical checks but
failed this agent's evidence review: DSH's warning admitted that the page had
not been retrieved, while its summary claimed verification. It is retained as
failure evidence and excluded from successful source-verification proof. The
later run above contains the actual retrieved passages and closes that finding.

### Recovery, boundary and regression evidence

Supplemental artifacts are under
`test_artifacts/dsh_runtime_completion_20260905/`. The process probes use real
DSH, RPC, SQLite and guarded MongoDB with a deterministic model endpoint; they
establish lifecycle behavior, separately from the real-model cases above.

- `sidecar-clock-fixed/result.json`: authenticated operation, invalid authority
  rejection, separate sessions, unavailable semantic worker, SQLite checkpoint
  restart, and exact replay after a lost terminal response all passed.
- `final-brain-task-replay/result.json`: reworded completed-task retry preserves
  the exact result, binding and thread, without executing another operation.
- `promotion-replay/result.json`: real inline checkpoint and durable promotion;
  reworded replay preserves the result, binding, thread, accepted task and job.
- `final-transport-loss/result.json`: killing the sidecar produces a typed
  failed/blocked capability result and truthful faulted binding.
- `authority-clock-checks.json`: a Python-signed token passes at 1 and 50 ms
  future issuance; 51 ms future issuance and expired authority are rejected.
- `configured-readiness.json` and configured startup logs: normal Brain health
  reported an available database and sidecar readiness matched the configured
  route, catalogs, profile, release, policy and repository workspace.
- Focused deterministic checks: 19 conversation-progress/capability tests and
  9 dialog tests passed. The sidecar's pinned `pnpm` build passed strict `tsc`.

### Review limits and environment observations

This is the implementing agent's review of actual behavior and artifacts;
there is no claim of independent-agent review. The owner requested operational
proof ahead of process, so retired matrices and prior sign-off narratives were
not used as acceptance requirements.

Several fresh test-database bootstraps encountered MongoDB `AutoReconnect`
before the Brain started; subsequent runs succeeded. One protected trace write
timed out in the successful research run, while the harness retained raw model
and tool calls. The research case took almost ten minutes because repeated
Windows HTTPS commands required fresh cognition approvals. These observations
are retained as environmental/performance limits, rather than hidden by the
successful functional results.

Live delivery proof uses the debug adapter. The configured startup smoke kept
scheduled workers disabled to avoid unrelated scheduled work; isolated deferred
cases exercised the actual background worker and delivery path. Owned probe
processes and uniquely guarded test databases were cleaned up after each run.

All five behavior cases above passed technical checks and this agent's review
of the actual tool evidence, decisions, visible outputs and durable outcomes.
The reproduced DSH crash paths are repaired, the required package build works,
and the observed completion/recovery paths support functional sign-off. The
MongoDB and latency observations above remain explicit limits of this record.
