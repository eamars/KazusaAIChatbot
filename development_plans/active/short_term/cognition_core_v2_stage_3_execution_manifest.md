# Cognition Core V2 Stage 3 Execution Manifest

## Summary

- Record role: mandatory contract, budget, command, and checkpoint companion to
  [cognition_core_v2_stage_3_system_adoption_plan.md](cognition_core_v2_stage_3_system_adoption_plan.md).
- Exact inventory companion:
  [cognition_core_v2_stage_3_change_radius.md](cognition_core_v2_stage_3_change_radius.md).
- Plan class: high_risk_migration.
- Status: in_progress.
- Authority: the parent plan governs; user-authorized implementation is active;
  this file grants no production-data authority.
- Reread rule: reread the parent and this complete manifest after compaction
  and after each signed checkpoint.

## User Quality Sign-off Record — 2026-07-22

The user approved the Phase 3 artifact set, including the consolidated raw
Chinese real-LLM dialog/monologue report and the emotion, abuse-boundary, role,
mechanical-path, and bounded-error evidence overlay. This closes artifact
quality review for the retained Phase 3 evidence. The manifest remains
`in_progress` while the separately tracked external Browser acceptance and
remaining lifecycle completion checks are pending.

## Approved Remediation Decision — 2026-07-19

The user approved the Stage 3 answerability-separation remediation. The
implementation uses the existing Cognition Core, action-selection, resolver,
supervisor, and resolver-loop change surface recorded in the change-radius
companion.

- `goal_resolution` is a Cognition-Core-owned, validated semantic decision for
  the accepted user goal. Its allowed values are `answerable_now`,
  `requires_required_evidence`, `requires_user_input`, and `blocked`.
- RAG source fields such as `conversation_evidence.resolved` remain
  source-coverage results. Missing optional or unrelated evidence does not
  force the overall goal to remain unresolved.
- `answerable_now` terminates optional retrieval and allows the current episode
  to settle. Required-evidence and user-input decisions preserve their
  resolver/clarification paths, while technical failures remain typed and
  bounded.
- LLM output owns the semantic decision. Deterministic code validates it,
  enforces capability/loop/no-progress limits, and records trace settlement.
  The remediation adds no LLM stage, raises no cap, uses no keyword classifier,
  and adds no compatibility vocabulary.
- The verification contract is focused failing-then-passing tests, affected
  deterministic suites, one-at-a-time real-LLM private-18 and required-evidence
  cases, and a refreshed sequential 40-case call/latency/quality/trace ledger.
  The existing Stage 3 p95 gate remains binding.

## Follow-up Decision Record — 2026-07-19

The first post-remediation live checks establish the next bounded prompt
refinement:

- `private_18` reached a complete direct answer with eight LLM calls and no
  resolver stage.
- `private_08` preserved the required-evidence/user-clarification behavior but
  repeated the failed resolver request before producing its clarification.
- When resolver context reports a missing required referent or user-provided
  detail, the action-planning prompt directs the LLM-owned decision to
  `requires_user_input` and omits another resolver request.
- The refinement remains inside the existing prompt/contract boundary. No
  keyword classifier, deterministic semantic rewrite, new LLM stage, cap
  increase, or compatibility vocabulary is introduced.
- The two live cases are rerun one at a time before the sequential latency,
  call-count, quality, and trace ledger. The existing p95 gate remains binding.

### Rejected-tail remediation decision — 2026-07-20

The user-approved rejected-case work addressed the two tails whose latency was
not justified by the resulting cognition:

- `group_16` inherited the prior `group_09` human-clarification pending row
  because the loader matched only platform/channel/user scope. Pending HIL and
  approval resumes now require the typed current-turn
  `reply_context.reply_to_message_id` to equal the pending source message id;
  unrelated same-scope messages remain independent turns. The existing scope,
  expiry, future-row, and same-source checks remain unchanged.
- `private_03` treated a direct question about the character's own current
  experience as a possible local-context retrieval goal. The action-planning
  prompt now states that persona/private-monologue-supported self-reports are
  `answerable_now`, and that optional empty or failed local context cannot prove
  the character's private state. The decision remains LLM-owned; no keyword
  classifier, deterministic semantic rewrite, new stage, cap increase, or
  compatibility vocabulary was added.

Focused verification recorded 45 passed tests, including the new unrelated
same-scope pending regression and self-report prompt contract. The adjacent
action/persona suite recorded 63 passed and three stale route-expectation
failures outside this remediation: two persona action-route expectations and
the previously known three-private-action speech expectation.

Clean one-at-a-time guarded live evidence is recorded under
`test_artifacts/cognition_core_v2/stage_3/answerability_remediation_rejected_clean/`:

- `group_16`: cold start, 7 LLM calls, 72.4 seconds, no resolver stage, one
  settled trace, one lifecycle record, and visible raw-language dialog.
- `private_03`: restart, 8 LLM calls, 70.8 seconds, no local-context resolver
  stage, one settled trace, one lifecycle record, and visible raw-language
  dialog.

Independent review by `Hilbert` found the two production changes structurally
sound and compliant with LLM ownership and CJK safety. Stage 3 remains open
because broader non-live regression failures, Browser acceptance, final user
signoff, and a separate existing user-input final-cognition contract gap in
`cognition_resolver/loop.py` remain unresolved. The loop gap is outside these
two rejected tails; the revised p95 gate is addressed by the user decision
below.

### User Decision Record — latency ceiling — 2026-07-20

The user approved a 120,000 ms ceiling for the fixed-sequence ordinary
foreground p95. The refreshed sequence p95 is 103,807 ms, so the revised p95
gate passes. The observed 125,418 ms maximum remains separately reported as a
tail metric and is not used as the p95 acceptance criterion. The existing
120,000 ms limit for an individual blocking LLM call remains binding, and this
decision raises no LLM, prompt, output, repair, retry, or resolver-cycle cap.
All other technical, regression, Browser, quality, and user-signoff gates
remain binding.

### User Decision Record — emotion and bounded-error sign-off overlay — 2026-07-21

The newly created Chinese emotion and abuse probes are retained as a mandatory
Stage 3 user-quality sign-off overlay. They supplement the frozen 40-case
source sequence and do not replace it.

- Sadness-to-crying, four secondary-emotion, high-attachment-abuse,
  verbal-abuse-boundary, mechanical abuse-to-sadness, and full dialog rendering
  tests remain in the exact change-radius inventory.
- Real-LLM selectors run one case at a time and are inspected from their raw
  input, prompt-stage output, parsed output, state projection, visible dialog,
  and settled trace. A green pytest result alone is insufficient.
- The full `/chat` abuse-to-sadness rendering case is the positive mechanical
  proof: a pre-established negative event outcome projected very-high sadness
  and reached visible Chinese dialog without adding a `crying` emotion id.
- The outcome-neutral natural abuse probe remains a required diagnostic case.
  Its observed `+5` outcome delta is a semantic model/prompt quality failure;
  deterministic code must not rewrite it into a negative outcome.
- Contract-invalid candidates and failed repair attempts are excluded from
  affect evidence. The producing stage owns bounded regeneration or complete
  replacement; after its explicit cap the path fails closed with a typed
  terminal result and settled trace.
- A recoverable bound violation may use only a contract-defined deterministic
  normalization followed by revalidation. An unsafe or still-invalid value
  follows the same bounded regeneration/fail-closed path.
- The first high-positive-regard attempt, the seeded negative-outcome
  `perceived_closeness` attempt, and the fear action-planning repair rows remain
  recorded as bounded-error evidence rather than being counted as emotion
  outcomes.

The agent-authored consolidation is
`test_artifacts/cognition_core_v2/stage_3/emotion_boundary_bounded_error_signoff_review.md`.
It records the exact selectors, observed Chinese outputs, raw evidence paths,
and remaining residual risks.

### Capability observation refinement — 2026-07-19

The required-evidence trace showed that local context recall logs an unresolved
referent and skips retrieval, while its typed observation is projected as an
empty successful recall. The capability will emit a bounded `blocked`
observation with a prompt-safe user-input-required summary when its structured
referent precondition is missing. This supplies clearer semantic context to
Cognition Core without assigning or rewriting `goal_resolution`; the LLM
remains the semantic owner.

### Direct-answer triage wording — 2026-07-19

The live probes preserved safe answers but still over-selected resolver work:
the general relationship question used one resolver pass in a 13-call run,
and the unresolved-referent clarification used 19 calls. The action-planning
prompt will state that a general question, opinion, analysis, or advice request
answerable from the accepted bid, current input, monologue, and available
context defaults to `answerable_now`. Resolver availability, empty/failed
optional evidence, or unrelated missing evidence is insufficient reason for
retrieval. Only explicitly necessary missing facts or user-controlled missing
information select the two resolver paths. The choice remains LLM-owned.

### Independent review remediation — 2026-07-19

The independent implementation review identified four remaining remediation
items and one accepted configuration disposition:

- Local-context recall with a structured unresolved referent will expose a
  typed user-input blocker observation. The resolver gets one final Cognition
  Core pass for an LLM-owned clarification. A repeated blocked request, or a
  final pass with neither a clarification action nor another resolver request,
  is deterministically cleared and settled with a prompt-safe clarification
  surface for user-message episodes; non-user sources remain private.
- A contract test will cover `conversation_evidence.resolved=false` beside an
  answerable goal and prove that optional retrieval is not requested.
- MongoDB connection diagnostics will use a sanitized endpoint description
  rather than logging the raw URI, credentials, or query options.
- The change-radius companion will list the direct remediation contract
  fixtures.
- The harness's `.env` loading is an intentional user-authorized input path
  when no explicit mapping is supplied. Exact reserved database-name and
  URI-path guards, child-process injection, and endpoint fingerprinting remain
  the isolation controls; `.env` contents are not inspected or copied by the
  assistant.

The 40-case p95 gate, broader non-live regression failures, Browser session
acceptance, and final user sign-off remain open. This record does not close
Stage 3.

### Live trace finalization polling — 2026-07-19

The sequential live refresh showed that `group_17` and `group_18` reached
visible dialog and a skipped lifecycle while the trace row remained `running`
at the harness's fixed 15-second poll boundary. The single-case live harness
will use a 60-second terminal-trace poll window so post-turn finalization is
observed without turning a successful case into a harness failure. This is a
test-observation adjustment only; it does not add runtime work or relax the
Stage 3 p95/signoff gates.

## Frozen Contracts

### Character Profile Seed

Public module:

```python
# src/kazusa_ai_chatbot/character_profile.py
def load_character_profile_seed(path: Path) -> CharacterProfileSeedV1: ...

# src/kazusa_ai_chatbot/db/character.py
async def ensure_character_profile_seed(
    seed: CharacterProfileSeedV1,
) -> Literal["inserted", "verified"]: ...
```

`character_profile.py` owns file loading and validation only. DB persistence
stays in `db.character`. `CharacterProfileSeedV1` requires:

- a non-empty string `name`;
- a non-empty mapping `personality_brief`;
- a `boundary_profile` containing 0.0-1.0 numeric values for
  `self_integrity`, `control_sensitivity`, `relational_override`,
  `control_intimacy_misread`, and `authority_skepticism`;
- `boundary_profile.compliance_strategy` equal to `resist`, `evade`, or
  `comply`, and `boundary_profile.boundary_recovery` equal to `rebound`,
  `delayed_rebound`, `decay`, or `detach`;
- a `linguistic_texture_profile` containing numeric values for
  `fragmentation`, `hesitation_density`, `counter_questioning`,
  `softener_density`, `formalism_avoidance`, `abstraction_reframing`,
  `direct_assertion`, `emotional_leakage`, `rhythmic_bounce`, and
  `self_deprecation`.

Numeric profile values must satisfy the ranges already defined by the current
character schema. Optional static character fields are preserved. The loader
rejects invalid JSON, a non-object root, non-absolute/missing paths, and every
runtime-owned top-level field: `_id`, `global_user_id`, `self_image`,
`cognition_state`, `updated_at`, `mood`, `global_vibe`, and
`reflection_summary`.

`ensure_character_profile_seed` has two results:

- `inserted`: atomically create `_id: "global"` with all static seed fields,
  `cognition_state=build_character_production_state(now)`, and
  `updated_at=now` in one document.
- `verified`: the existing row has the same non-empty character `name` and a
  valid native `cognition_state`; leave every field untouched.

A different/missing existing `name` or missing/invalid native
`cognition_state` fails startup for Stage 4 migration. `db_bootstrap()` creates
collections/indexes only: it removes its current character singleton insert
and cognition-state backfill, so one owner creates the seed. There is no
overwrite, merge, `--force`, fallback profile, or relative-path search in
normal startup.
Fresh bootstrap also removes its current `drop_collection` calls for
`rag_cache_index`, `rag_metadata_index`, and `background_artifact_jobs`.
Legacy discovery/disposition belongs exclusively to the Stage 4 allowlist;
normal service startup never destroys an unexpected collection.
`src/scripts/load_character_profile.py` remains a maintenance command but uses
the same loader/validator and explicit overwrite policy; normal startup never
depends on it.

Fresh-user initialization retains `db.users.resolve_global_user_id` and
`create_user_profile` as its single owner. The first platform identity creates
one user profile with `build_acquaintance_user_state(updated_at=now)`; repeated
identity resolution returns the same profile and preserves its cognition state.
The fresh-database gate proves this for both group and private target scopes.

Startup order is exact:

```text
parse required configuration
-> connect and bootstrap native collections/indexes
-> load and validate absolute profile path
-> atomically insert or verify character singleton
-> validate native character cognition default can be built
-> start intake, scheduler, reflection, self-cognition, task, and background workers
-> report ready
```

`Dockerfile` sets `CHARACTER_PROFILE_PATH=/app/personalities/kazusa.json` after
copying `personalities/`. `config.py` exposes the stripped raw value without
failing at import; `require_character_profile_path()` validates it at service
startup. Local and test service processes supply an absolute path, while
scripts that merely import configuration remain usable.

### Fresh Database Isolation

The Stage 3 harness accepts the ordinary MongoDB configuration with one exact
reserved database name:

```text
MONGODB_URI=<configured MongoDB URI>
MONGODB_DB_NAME=_test_kazusa_core_v2
CHARACTER_PROFILE_PATH=<absolute path>
```

`Stage3MongoEndpointIdentityV1` remains an informational fingerprint in
evidence. It is canonical JSON with `scheme`, `hosts`,
`tls`, `replica_set`, and `direct_connection`. `scheme` is `mongodb` or
`mongodb+srv`; hosts are sorted and lower-cased, ordinary Mongo hosts use port
27017 when omitted, and SRV uses the canonical hostname without DNS
resolution. `tls` honors `tls`/`ssl`, defaulting true for SRV and false for
ordinary Mongo. `replica_set` is normalized or empty and
`direct_connection` is a Boolean. Credentials, database name, and query order
are excluded. The endpoint fingerprint is SHA-256 over UTF-8 canonical JSON.

Before importing service configuration or starting a subprocess, the harness:

1. loads the ordinary MongoDB URI/database settings without importing service
   configuration;
2. requires the exact reserved database name and rejects a URI database path
   that disagrees;
3. in `cold_start` mode inspects only the reserved database and requires no
   persistent collections, then writes a local run-session record containing
   the informational URI fingerprint/name;
4. injects the same URI and exact name into the service subprocess, together
   with an internal database guard marker;
5. records the informational digest, never credentials, in evidence;
6. in `restart` mode requires the run-session database name/fingerprint to
   match and requires the same reserved database to contain collections;
7. permits cleanup only after repeating the session/name checks and drops only
   the exact reserved database.

Missing inputs or a mismatched URI path stop the test before service import.
The ordinary URI is intentional; the database name is the primary isolation
guard for this environment.

Negative tests prove that a different database name and an embedded URI path
are rejected, while the same endpoint with the exact reserved database name is
accepted. Endpoint identity normalization remains covered as informational
evidence behavior.

### Canonical Cognitive Episode

Public module: `src/kazusa_ai_chatbot/cognition_episode.py`.

```python
TriggerSource = Literal[
    "user_message",
    "internal_thought",
    "self_cognition",
    "scheduled_tick",
    "tool_result",
]

def build_trigger_source_registry() -> dict[str, TriggerSourceSpecV1]: ...
def build_user_message_episode(
    *,
    episode_id: str,
    origin: UserMessageOriginV1,
    target_scope: TargetScopeV1,
    dialog_percept: PerceptV1,
    media_percepts: Sequence[PerceptV1],
    evidence_refs: Sequence[EvidenceRefV1],
    local_time_context: LocalTimeContextDoc,
    created_at: str,
    debug_controls: DebugControlsV1,
) -> CognitiveEpisodeV1: ...
def build_internal_thought_episode(
    *,
    latch: InternalActionLatchV1,
    evidence_refs: Sequence[EvidenceRefV1],
    local_time_context: LocalTimeContextDoc,
    created_at: str,
    claim_token: str,
) -> CognitiveEpisodeV1: ...
def build_self_cognition_episode(
    *,
    case: SelfCognitionCase,
    percepts: Sequence[PerceptV1],
    evidence_refs: Sequence[EvidenceRefV1],
    local_time_context: LocalTimeContextDoc,
    created_at: str,
) -> CognitiveEpisodeV1: ...
def build_scheduled_tick_episode(
    *,
    case: SelfCognitionCase,
    calendar_run: CalendarRunDoc,
    percepts: Sequence[PerceptV1],
    evidence_refs: Sequence[EvidenceRefV1],
    local_time_context: LocalTimeContextDoc,
    created_at: str,
) -> CognitiveEpisodeV1: ...
def build_tool_result_episode(
    *,
    result: ToolResultReadyV1,
    evidence_refs: Sequence[EvidenceRefV1],
    local_time_context: LocalTimeContextDoc,
    created_at: str,
) -> CognitiveEpisodeV1: ...
```

All five builders return the authoritative episode envelope with:

```python
{
    "schema_version": "cognitive_episode.v1",
    "episode_id": str,
    "trigger_source": TriggerSource,
    "origin_metadata": OriginMetadataV1,
    "target_scope": dict[str, object],
    "percepts": list[PerceptV1],
    "evidence_refs": list[EvidenceRefV1],
    "created_at": str,
    "privacy_scope": str,
    "continuation_depth": int,
}
```

`TriggerSourceSpecV1` is the exact registry shape from the authoritative
cognition contract design: schema version, source kind, owner, entrypoint, LLM
visibility, evidence policy, persistence policy, rate-limit policy,
privacy policy, and allowed continuation depth. The five entries are complete
at startup; source owners may not create an unregistered episode.

Source origin records are mechanically typed and prompt-hidden except for a
bounded semantic source summary:

| Type | Required source fields |
|---|---|
| `UserMessageOriginV1` | `platform`, current/active message ids, conversation row ids, debug modes, correlation id |
| `InternalThoughtOriginV1` | prior episode id, action-latch ref, continuation depth, observed-context refs, correlation id |
| `SelfCognitionOriginV1` | source-case kind/ref, observed-context refs, target scope ref, correlation id |
| `ScheduledTickOriginV1` | calendar event id, claim id, scheduled-for UTC, continuation objective ref, target scope ref |
| `ToolResultOriginV1` | task id, task kind, result ref, completion UTC, target scope ref, correlation id |

Every origin record also contains `schema_version`, `owner`, `privacy_scope`,
`delivery_permission_ref`, and `created_at`. Raw ids never enter prompts.

Exact runtime ownership separates event/claim production from the one episode
production callsite:

| Source | Event/claim owner | Sole episode-production owner/callsite |
|---|---|---|
| `user_message` | brain intake queue | `service._process_queued_chat_item` |
| `internal_thought` | `db.internal_action_latches` issue/claim lifecycle | `self_cognition.runner.build_self_cognition_case_artifacts_async`, invoked by `self_cognition.worker.run_self_cognition_worker_tick` only with a claimed latch |
| `self_cognition` | `self_cognition.sources` selects one grounded ordinary case | `self_cognition.runner.build_self_cognition_case_artifacts_async` |
| `scheduled_tick` | `calendar_scheduler.worker.run_calendar_worker_tick` claims the run and `calendar_scheduler.handlers.handle_commitment_due_cognition_run` emits the due case | `self_cognition.runner.build_self_cognition_case_artifacts_async` |
| `tool_result` | background worker and `accepted_task.lifecycle` make a typed result ready | `background_work.delivery.run_background_work_delivery_tick`, using projection from `background_work.result_source` |

Event/claim owners never build episodes. Each row's sole episode-production
callsite calls the matching public builder in `cognition_episode.py`.

`ToolResultReadyV1` carries `task_id`, `task_kind`, bounded
`semantic_summary`, bounded artifact or failure text, `completed_at`, requester
`target_scope`, original message/evidence refs, and prompt-safe
`coding_run_context` when applicable.

The old-to-new mapping is fixed:

- `storage_timestamp_utc` becomes `created_at`;
- `local_time_context` becomes one prompt-safe time percept plus mechanical
  origin metadata where required;
- `input_sources` is derived from `percepts[*].source_kind` and is not stored;
- `output_mode` is removed. Normal chat permits visible reply, `silent` and
  `listen_only` are private-only, `preview` is preview-only, and scheduled
  delivery follows the registered source policy;
- `DebugControlsV1` contains `think_only`, `no_remember`, and
  `no_visual_directives`. `think_only` disables visible delivery,
  `no_remember` deterministically prohibits consolidation writes, and
  `no_visual_directives` removes the visual affordance;
- `TargetScopeV1` carries platform, channel, user, addressee/broadcast, and
  permission ref. Prompt projection substitutes configured role names and
  hides ids;
- the dialog percept has role-explicit semantic text plus structured reply and
  mention subobjects. Each accepted image/media observation becomes a separate
  percept, preserving the current maximum of four media items;
- every LLM-to-LLM handoff contains one bounded prompt-safe semantic string;
  typed ids and control metadata remain prompt-hidden.

Source policy:

| Source | Model-visible evidence | Permitted result | Continuation |
|---|---|---|---|
| `user_message` | accepted turn, retrieved context, bounded continuity | visible reply, private silence, actions | action contract, max depth 1 |
| `internal_thought` | active prior action latch plus bounded residue/observed context | private result; visible output only through selected `speak` and existing permission | action contract, max depth 1 |
| `self_cognition` | observed eligible source case plus bounded continuity | private result; visible output only with existing proactive permission and adapter availability | action contract, max depth 1 |
| `scheduled_tick` | claimed commitment/objective and current evidence | action, private result, or permitted delivery | action contract, max depth 1 |
| `tool_result` | prompt-safe task result and original objective context | permitted delivery, follow-up action, or private result | action contract, max depth 1 |

Internal monologue residue and promoted reflection remain valid evidence. They
cannot initiate an episode. The internal-thought builder rejects a missing,
expired, already-consumed, or depth-exhausted action latch. `system_probe`
remains a diagnostic operation and must not enter normal cognition as a
production episode.

### Durable Internal Action Latch

`src/kazusa_ai_chatbot/db/internal_action_latches.py` owns collection
`internal_action_latches`; `db.schemas` owns `InternalActionLatchV1`:

```python
class InternalActionLatchV1(TypedDict):
    schema_version: Literal["internal_action_latch.v1"]
    latch_id: str
    idempotency_key: str
    source_episode_id: str
    source_action_attempt_id: str
    continuation_objective: str
    evidence_refs: list[EvidenceRefV1]
    target_scope: TargetScopeV1
    privacy_scope: str
    continuation_depth: int
    status: Literal["pending", "claimed", "consumed", "expired", "failed"]
    not_before: str
    expires_at: str
    claimed_by: str
    claim_token: str
    claim_expires_at: str
    attempt_count: int
    max_attempts: Literal[3]
    last_error_code: str
    consumed_episode_id: str
    created_at: str
    updated_at: str
    purge_after: str

class InternalActionLatchClaimV1(TypedDict):
    latch: InternalActionLatchV1
    claim_token: str

async def issue_internal_action_latch(
    *,
    source_episode_id: str,
    source_action_attempt_id: str,
    continuation_objective: str,
    evidence_refs: Sequence[EvidenceRefV1],
    target_scope: TargetScopeV1,
    privacy_scope: str,
    continuation_depth: int,
    now: str,
) -> InternalActionLatchV1: ...
async def claim_due_internal_action_latch(
    *, worker_id: str, now: str
) -> InternalActionLatchClaimV1 | None: ...
async def release_internal_action_latch(
    *, latch_id: str, claim_token: str, retry_at: str, error_code: str, now: str
) -> InternalActionLatchV1: ...
async def consume_internal_action_latch(
    *, latch_id: str, claim_token: str, consumed_episode_id: str, now: str
) -> InternalActionLatchV1: ...
async def fail_internal_action_latch(
    *, latch_id: str, claim_token: str, error_code: str, now: str
) -> InternalActionLatchV1: ...
async def expire_due_internal_action_latches(*, now: str) -> int: ...
```

Deterministic orchestration emits one latch after a settled action result asks
for `continuation_mode="immediate_followup"` and
`episode_type="internal_thought"`. The idempotency key is derived from the
source action-attempt id. The emitter sets parent depth plus one, rejects depth
above one, sets `expires_at` to 15 minutes, a claim lease to 300 seconds, three
technical attempts, and
`purge_after=expiry_from_storage_iso(created_at,
ttl_days=AUDIT_LOG_TTL_DAYS)`; the current default is 90 days but the expression,
not a duplicated literal, is authoritative.

The self-cognition worker atomically claims a due pending latch or reclaims an
expired lease, increments `attempt_count`, and supplies the matching
`claim_token` to the builder. Residue alone cannot create this source. The
builder exposes only the prompt-safe continuation objective and admitted
evidence. A successful or typed non-retryable settled trace consumes the
latch; retryable technical failure releases it while attempts remain; exhausted
attempts mark it failed; expiry marks it expired. Unique `idempotency_key`
prevents duplicate episodes/actions across retries and restarts.

`issue_internal_action_latch` uses atomic `$setOnInsert` keyed by the unique
idempotency key and returns the existing identical latch on replay; mismatched
semantic material for the same key fails. Claim is one atomic
pending-or-lease-expired transition. Release/consume/fail require the current
claim token and fail on stale ownership. Expiry updates only unclaimed expired
rows. All state transitions validate the prior status and write `updated_at`.

Bootstrap creates a unique idempotency index, a
`(status, not_before, expires_at)` claim index, a `claim_expires_at` lease
index, and a TTL index on `purge_after`. The current dry-run latch and the
runner's fabricated `local_tracking` latch are retired in the same checkpoint.

Forbidden trigger literals after cutover, outside historical archives and the
three Stage-4-only files, are:

```text
reflection_signal
scheduled_recall
system_probe
accepted_task_result_ready
```

### Capability Availability And Action Authority

`action_spec.registry` remains the declarative `CapabilitySpecV1` authority.
Runtime code projects the authoritative design's `AffordanceSpecV1` from a
side-effect-free availability snapshot:

```python
class AvailabilityProbeResultV1(TypedDict):
    status: Literal["available", "degraded", "unavailable"]
    reason_code: str
    checked_at: str
    expires_at: str

class RuntimeCapabilitySnapshotV1(TypedDict):
    checked_at: str
    expires_at: str
    route_health: Mapping[str, str]
    repository_access: Mapping[str, Literal["read_write", "read_only", "down"]]
    worker_status: Mapping[str, str]
    scheduler_status: str
    adapter_target_status: Mapping[str, str]
    coding_workspace_status: str
    permissions: Mapping[str, bool]

def build_episode_affordances(
    capabilities: Mapping[str, CapabilitySpecV1],
    context: ActionAvailabilityContextV1,
    snapshot: RuntimeCapabilitySnapshotV1,
) -> list[AffordanceSpecV1]: ...

async def recheck_action_affordance(
    capability_kind: str,
    context: ActionAvailabilityContextV1,
    fresh_snapshot: RuntimeCapabilitySnapshotV1,
) -> AvailabilityProbeResultV1: ...
```

`AffordanceSpecV1` retains the exact authoritative fields: schema version,
capability kind, owner, surface, availability, visibility, latency/cost/risk
tiers, allowed cognition/continuation modes, permission policy, parameter
summary, and prompt affordance. Snapshot collection reads health, repository,
worker, scheduler, adapter-target, workspace, and permission state without
enqueueing or executing anything. Snapshots expire after five seconds; the
evaluator obtains a fresh snapshot immediately before effects.

Allowed reason codes are `ready`, `queue_only`, `permission_denied`,
`target_unavailable`, `repository_unavailable`, `worker_unavailable`,
`route_unavailable`, `workspace_unavailable`, and `unsupported_work_kind`.
`available` is projected. `degraded` is projected only when the durable owner
can accept the request and return typed `pending`; `unavailable` is omitted.
Prompts never interpret raw health strings.

The complete registry matrix is:

| Capability | Runtime owner/probe | Availability requirement | Prompt visibility |
|---|---|---|---|
| `memory_lifecycle_update` | memory lifecycle specialist | specialist registered and repository writable; no degraded route | available |
| `apply_memory_lifecycle_update` | memory lifecycle executor | repository writable; no degraded route | internal only |
| `speak` | L3 text + adapter registry | healthy text route, delivery permission, and background adapter target when applicable; no degraded route | available |
| `trigger_future_cognition` | calendar/orchestrator | repository+scheduler; `queue_only` when repository is writable but worker is unavailable | available/degraded |
| `future_speak` | background work/calendar | background+scheduler repositories and worker; `queue_only` when repositories are writable but worker is unavailable | available/degraded |
| `accepted_task_request` | accepted-task lifecycle | supported worker+repository; `queue_only` when repository is writable but worker is unavailable | available/degraded |
| `accepted_coding_task_request` | accepted-task coding owner | supported kind, repository, valid workspace, worker; `queue_only` only with writable queue+valid workspace+unavailable worker | available/degraded |
| `accepted_task_status_check` | accepted-task repository | repository readable | available |
| `background_work_request` | background-work router | supported kind+queue; `queue_only` when queue is writable but worker is unavailable | available/degraded |

`accepted_task_request` becomes an explicit registry entry; any separate
allowlist loses authority. The registry owns declarative schema, permission,
rate, audit, prompt, and handler identity. The evaluator owns authorization and
fresh availability checks. Handler modules own effects. Action results and the
single settlement owner record outcomes/trace identity. No current capability
is removed.

### Safe Technical Failure Settlement

The existing typed safe-checkpoint policy applies to every source:

- retry only a classified transient failure before non-idempotent effects;
- retain every existing retry/cycle cap; the new latch alone has its fixed
  three-attempt contract stated above;
- resume from the latest validated checkpoint rather than rerun completed
  actions;
- if recovery exhausts, emit a typed failed action/stage result and settle the
  episode trace;
- keep chat intake responsive while background episodes reach terminal state;
- record `error_code`, stage, attempt count, safe checkpoint, retryability, and
  final route without exposing backend text to the user.

A raw busy/backend exception may not escape as the visible string
`KazusaLiveBot is busy right now, please try again later.` and may not prevent
trace settlement. Content drift continues through the planned run.

### Single Episode Trace Settlement

Pure schemas/builders remain in `action_spec.results`. Runtime settlement moves
to one public owner:

```python
# src/kazusa_ai_chatbot/brain_service/post_turn.py
def settle_episode_trace(
    *,
    episode: CognitiveEpisodeV1,
    cognition_output: CognitionCoreOutputV2 | None,
    action_specs: Sequence[ActionSpecV1],
    action_results: Sequence[ActionResultV1],
    surface_outputs: Sequence[SurfaceOutputV1],
    terminal_status: EpisodeTerminalStatusV1,
    attempt_diagnostics: Sequence[EpisodeAttemptDiagnosticV1],
    delivery_correlation: DeliveryCorrelationV1,
    settled_at: str,
) -> EpisodeTraceV2: ...
```

`EpisodeTraceV2` updates the authoritative reference design and has exact
fields: `schema_version="episode_trace.v2"`, `episode_id`, `trigger_source`,
`terminal_status`, `cognition_refs`, `action_specs`, `action_results`,
`surface_outputs`, `attempt_diagnostics`, `delivery_correlation`, `created_at`,
and `settled_at`. Terminal status is one of `completed_visible`,
`completed_private`, `completed_action`, `scheduled`, `failed`, or `cancelled`.
Each `EpisodeAttemptDiagnosticV1` records stage, error code, attempt count, safe
checkpoint, retryability, and final status. `DeliveryCorrelationV1` records
delivery intent, tracking id, receipt status (`not_applicable`, `pending`,
`delivered`, `failed`, or `unknown`), and receipt ref. A strict validator rejects
unknown versions, missing terminal fields, duplicate action-attempt ids, and
delivery/status contradictions.

Persona and self-cognition nodes only accumulate components. Runtime calls
`brain_service.post_turn.settle_episode_trace` once after primary
cognition/actions/L3 and after a delivery tracking id is assigned, before
`ChatResponse` completion and consolidation. A visible intent normally settles
with receipt status `pending`; adapter delivery persists the later actual
receipt separately and never mutates the immutable trace.

The current post-response memory-lifecycle task remains one background call but
stops rebuilding the episode trace. Its exact audit contract is:

```python
# src/kazusa_ai_chatbot/brain_service/contracts.py
class PostTurnLifecycleRecordV1(TypedDict):
    schema_version: Literal["post_turn_lifecycle_record.v1"]
    lifecycle_record_id: str
    source_episode_id: str
    delivery_tracking_id: str
    action_projections: list[ConsolidationActionProjectionV1]
    status: Literal["skipped", "completed", "partial", "failed"]
    error_codes: list[str]
    created_at: str
    purge_after: str

# src/kazusa_ai_chatbot/brain_service/post_turn.py
def build_post_turn_lifecycle_record(
    *,
    source_episode_id: str,
    delivery_tracking_id: str,
    action_specs: Sequence[ActionSpecV1],
    action_results: Sequence[ActionResultV1],
    error_codes: Sequence[str],
    created_at: str,
) -> PostTurnLifecycleRecordV1: ...

# src/kazusa_ai_chatbot/db/post_turn_lifecycle.py
async def upsert_post_turn_lifecycle_record(
    record: PostTurnLifecycleRecordV1,
) -> Literal["inserted", "verified"]: ...
```

`lifecycle_record_id` is deterministic from `source_episode_id`; one primary
episode has one record. Action effects remain in their existing state/memory
owners; the record stores prompt-safe consolidation projections only. Empty
work is `skipped`; all succeeded is `completed`; mixed terminal results are
`partial`; no successful result after an attempted lifecycle action is
`failed`. Error codes are typed/sanitized and contain no backend text.

### LLM Contract Error Settlement

The `AGENTS.md` LLM error-state rule is binding for every V2 semantic, action,
surface, and verifier stage:

- Parse every raw response through the canonical
  `kazusa_ai_chatbot.utils.parse_llm_json_output(...)` entry point before
  semantic evaluation. Stage-local parsers and semantic repairers are not
  permitted.
- Classify malformed structure, unknown keys, wrong types, unsupported enum or
  handle values, missing fields, and conflicting fields as non-recoverable
  contract errors. The producing stage owns one bounded regeneration or
  complete replacement according to its existing cap; after the cap, fail
  closed with a typed result and settled trace.
- Classify a numeric bound violation as recoverable only when the owning
  contract defines a safe deterministic normalization. Normalize, revalidate,
  and record the original value and disposition. If normalization is unsafe or
  the value remains invalid, use the same bounded regeneration/fail-closed path.
- Keep every invalid candidate out of affect derivation, action authorization,
  persistence, scheduling, dialog, and adapter delivery. A structurally valid
  but semantically wrong value remains model-quality evidence and is not
  deterministically rewritten.

The retained emotion sign-off evidence includes the natural abuse signed-delta
failure, the high-attachment relationship-bound failure, the seeded
`perceived_closeness` bound failure, and the fear action-planning repair rows.
Those artifacts are technical evidence only and are excluded from affect
conclusions. The consolidated review is
`test_artifacts/cognition_core_v2/stage_3/emotion_boundary_bounded_error_signoff_review.md`.

Collection `post_turn_lifecycle_records` has unique indexes on
`lifecycle_record_id` and `source_episode_id`, an index on
`(delivery_tracking_id, created_at)`, and a TTL index on `purge_after`.
`purge_after=expiry_from_storage_iso(created_at,
ttl_days=AUDIT_LOG_TTL_DAYS)`. Atomic `$setOnInsert` returns `verified` only
for an identical existing record and fails on mismatch. This is
persistence-specialist audit work, not a new cognition source. Primary
consolidation consumes `EpisodeTraceV2`; the post-turn task persists this
record independently.

Required outcome rows:

| Outcome | Trace requirements | Consolidation |
|---|---|---|
| visible speech | cognition ref, speak action/result, user-visible text surface | yes, source policy applies |
| character silence | cognition ref, private surface/reason, no visible fragment | yes, private policy applies |
| action only | action spec/result, private/tool surface as applicable | yes |
| scheduled/pending | durable result ref and continuation | yes |
| rejected/unavailable | attempted spec and typed result | diagnostics; writes only when source policy permits |
| recovered retry | one final trace plus attempt diagnostics | yes, no duplicate action |
| exhausted/failed | typed terminal failure and completed attempt evidence | diagnostics; bounded source policy |
| non-delivered visible intent | surface intent plus delivery correlation | no fabricated delivered dialog |

`nodes/persona_supervisor2.py` and `self_cognition/runner.py` return components;
the appropriate service/worker orchestration boundary calls the one settlement
owner. Consolidation's `has_consolidatable_output` accepts the settled trace
rather than graph state.

### Stage 3 Evidence

`Stage3FreshDatabaseEvidenceV1` contains:

```python
{
    "schema_version": "stage3_fresh_database_evidence.v1",
    "database_endpoint_fingerprint": str,
    "database_name": "_test_kazusa_core_v2",
    "database_absent_before_start": True,
    "profile_bootstrap_result": "inserted",
    "restart_profile_result": "verified",
    "source_cases": list[dict],
    "action_cases": list[dict],
    "internal_latch_cases": list[dict],
    "trace_cardinality": dict[str, int],
    "post_turn_lifecycle_cardinality": dict[str, int],
    "collections_and_indexes": list[dict],
    "llm_call_ledger": list[dict],
    "latency_summary_ms": dict[str, int],
    "technical_failures": list[dict],
    "quality_review_path": str,
}
```

`Stage3NativeSchemaManifestV1` contains collection/index names, owning module,
native schema/version, creation condition, retention/TTL, and sanitized
representative shape. It contains no production observations.

## LLM Call And Context Budget

Counting method: count every `LLInterface.ainvoke` as one call; repairs count
separately. For CJK-heavy prompts, estimate two characters per token, so the
50,000-character projection ceiling is conservatively below the 50k-token
context cap. The parent records actual prompt characters, completion tokens
when exposed, duration, blocking/background classification, and retry count.

All values below are maxima, not targets. Named model means the existing config
route; Stage 3 cannot substitute a larger route.

| Path/stage | Calls per unit after cutover | Model/output cap | Input cap | Latency class |
|---|---:|---|---:|---|
| frontline relevance | 1 | `RELEVANCE_AGENT_LLM_MODEL`, 512 tokens | 8,000 chars | blocking |
| settled relevance | 1 decision + 1 authoritative repair | `RELEVANCE_AGENT_LLM_MODEL`, 512 tokens | 16,000 chars | blocking |
| message decontextualizer | 1 | `MSG_DECONTEXTUALIZER_LLM_MODEL`, 8,192 tokens | 50,000 chars | blocking |
| vision description | 1 per admitted media item, at most 4 | `VISION_DESCRIPTOR_LLM_MODEL`, 8,192 tokens | four current media payloads | blocking |
| V2 semantic appraisal | 6 per cycle | `COGNITION_LLM_MODEL`, 8,192 tokens | 50,000 chars/call | blocking in foreground; worker in background source |
| V2 branch cognition | 14 per cycle | `COGNITION_LLM_MODEL`, 8,192 tokens | 50,000 chars/call | same |
| workspace collapse | 1 per nonempty cycle | `COGNITION_LLM_MODEL`, 8,192 tokens | 50,000 chars | same |
| action planning | 1 per cycle | `COGNITION_LLM_MODEL`, 8,192 tokens | 50,000 chars | same |
| action authorization + repair | 2 per cycle | `COGNITION_LLM_MODEL`, 8,192 tokens | 50,000 chars/call | same |
| complete V2 resolver | 24 per cycle, 72 per three-cycle episode | `COGNITION_LLM_MODEL`, 8,192 tokens | as above | same |
| L3 style/content/preference | 3; visual adds 1 only when explicitly enabled | `COGNITION_LLM_MODEL`, 8,192 tokens | 50,000 chars/call | blocking for selected surface |
| dialog/fidelity route | 5: generator, one repair, semantic fidelity, role direction, surface integrity | `DIALOG_GENERATOR_LLM_MODEL`, 8,192 tokens | 50,000 chars/call | blocking |
| consolidation lane router | 1 per episode | `CONSOLIDATION_LLM_MODEL`, 8,192 tokens | 50,000 chars | background |
| memory-unit consolidation | 1 extractor + at most 3 candidates × (1 merge + 1 rewrite + 1 stability) = 10 | `CONSOLIDATION_LLM_MODEL`, 8,192 tokens | 50,000 chars/call | background |
| self-guidance consolidation | 2 | `CONSOLIDATION_LLM_MODEL`, 8,192 tokens | 50,000 chars/call | background |
| image consolidation | 1 summary + at most 1 compression = 2 | `CONSOLIDATION_LLM_MODEL`, 8,192 tokens | 50,000 chars/call | background |
| complete consolidation episode | at most 15 across the preceding lane-selected calls | `CONSOLIDATION_LLM_MODEL`, 8,192 tokens | as above | background |
| hourly reflection | 2 attempts/slot; 3 slots/tick = 6 | `CONSOLIDATION_LLM_MODEL`, 8,192 tokens | 50,000 chars/call | scheduled background |
| daily reflection | 2 attempts/channel; at most 25 scopes = 50 | `CONSOLIDATION_LLM_MODEL`, 8,192 tokens | 50,000 chars/call | scheduled background |
| promotion/style/group digest | 1 per eligible artifact/scope; at most 25 scopes for scope-bound passes | `CONSOLIDATION_LLM_MODEL`, 8,192 tokens | 50,000 chars/call | scheduled background |
| ordinary background job | 1 router + text-artifact router/generator when selected = 3/job; 2 claims/tick = 6 | `BACKGROUND_WORK_LLM_MODEL`, 8,192 tokens and 3,000 output chars | 8,000 chars | background |

The coding-agent worker retains its current independent PM/programmer/action
loop budgets and `CODING_AGENT_REPAIR_MAX_CALLS=4`; Stage 3 changes only its
typed result re-entry and proves its call ledger is byte-for-byte unchanged.

Hard rules:

- Stage 3 changes routing/context projection only; `LLMCallConfig` output caps
  and model routes remain unchanged.
- The V2 resolver remains capped at three cycles. Per cycle it permits at most
  six appraisals, fourteen branch calls, one collapse, one action-planning
  call, one authorization call, and one authorization repair. Stage 3 may
  reduce observed calls through truthful source/capability projection but may
  not raise any component cap.
- Source metadata ids, raw traces, registry handler ids, unavailable capability
  detail, and unpromoted reflection are dropped before prompts.
- Projection drop order is exact: raw trace/debug metadata, unavailable
  capability diagnostics, duplicate evidence, oldest low-ranked retrieved
  evidence, oldest continuity rows, then optional descriptive profile detail.
  Current user text, required role/source semantics, selected action schema,
  permissions, and required contract fields are never dropped.
- Any prompt over 50,000 characters fails the focused gate and must be reduced
  at its owning projection, not by raising the cap.
- A single blocking LLM call exceeding 120 seconds or fixed-sequence ordinary
  foreground p95 exceeding 120 seconds blocks the gate. The maximum foreground
  case is reported separately and does not redefine the p95 criterion.
  Consolidation, reflection, self-cognition, and background jobs remain
  non-blocking for chat intake; a queue wait never extends the foreground
  response deadline.
- Final evidence compares before/after call formulas and measured counts for
  each of the five sources. Any increase blocks completion and requires user
  approval.
- Anti-cheat gate: production prompts/code cannot contain Stage 3 case ids,
  fixture dialog strings, expected monologue/dialog fragments, or per-case
  branches. The fixture contains inputs and technical invariants only; human
  quality judgment is written after generation and never fed back into the
  same run.

## Exact Change Surface

The mandatory exact file inventory and boundary rules are in
[cognition_core_v2_stage_3_change_radius.md](cognition_core_v2_stage_3_change_radius.md).
That companion is part of this plan and must be amended before an unlisted
production or test path changes.


## Granular Checkpoint Steps

### A — Baseline And Test Contract

- A1: record `git status --short` and `git diff --name-only`.
- A2: run the forbidden-source/action/trace occurrence commands below and save
  exact file lists in the Stage 3 artifact directory.
- A3: validate the dedicated URI/name/fingerprints without importing service.
- A4: add focused failing profile-seed/bootstrap tests.
- A5: add focused failing five-source/availability/trace-cardinality tests.
- A6: record current V2 call formulas, fixed-sequence latency baseline, and
  expected failure output.

### B — Profile And Bootstrap

- B1: implement the public profile loader and rejection tests.
- B2: implement atomic insert/verify DB operation and race test.
- B3: wire required absolute config and Docker path.
- B4: order service startup before workers/intake.
- B5: keep maintenance loader on the shared validator.
- B6: prove fresh insert, repeat verify, conflicting-name failure, and runtime
  state preservation, including rejected missing boundary/linguistic fields and
  rejected legacy runtime fields.
- B7: record native bootstrap collections/indexes.
- B8: prove first-contact user creation and idempotent group/private identity
  resolution with native user cognition state.

### C — Canonical Sources

- C1: change trigger literal and origin unions.
- C2: implement the source registry, five public builders, and source/output
  invariants.
- C3: migrate user-message intake.
- C4: add durable latch schema/repository/indexes, issue/claim/lease/retry/
  consume/restart tests, then migrate action-latch-driven internal-thought and
  non-scheduled self-cognition cases.
- C5: migrate scheduled calendar cases.
- C6: migrate accepted/background result re-entry.
- C7: update RAG/consolidation/progress projections.
- C8: remove normal-runtime forbidden labels and record zero-reference evidence.

### D — Capability Availability

- D1: register `accepted_task_request` and prove complete capability roster.
- D2: add availability records/context to existing registry contracts.
- D3: bind deterministic probes to existing owners at service assembly.
- D4: project available/degraded affordances only.
- D5: recheck selected capability in evaluator before effects.
- D6: return typed unavailable/pending results without semantic rewrites.
- D7: delete separate action-authority allowlists.
- D8: prove baseline selection capacity and available/degraded/unavailable
  routing across all nine capabilities and every reason code.

### E — Trace Settlement

- E1: add focused outcome-matrix tests.
- E2: move runtime settlement to `brain_service.post_turn`.
- E3: remove persona direct trace build.
- E4: remove self-cognition direct trace build.
- E5: remove service competing trace build.
- E6: make consolidation/progress consume settled trace.
- E7: preserve delivery receipt ownership/correlation.
- E8: prove exactly one trace for every outcome and retry.
- E9: persist exactly one idempotent post-turn lifecycle record per source
  episode with native indexes, TTL, status, and mismatch tests.

### F — Runtime Adoption And Scaffold Retirement

- F1: wire internal-thought and self-cognition source cases through V2.
- F2: wire calendar due cases through V2.
- F3: wire accepted/background results through V2.
- F4: keep reflection offline and project promoted evidence only.
- F4a: prove a group-review case flows through `reflection_cycle.worker`, is
  promoted by policy, and becomes grounded evidence in a later
  `self_cognition` episode without raw reflection injection.
- F5: exercise future cognition/future speak/task status/background work.
- F6: prove worker failures do not block chat intake.
- F7: replace scaffold tests with runtime-owner tests.
- F8: delete scaffold modules.
- F9: run import/reference/delete gates.

### G — Operability

- G1: expose canonical source/terminal/call/latency status in health/ops.
- G2: update protected trace projection/export and redaction tests.
- G3: update console contracts/repository/routes.
- G4: update source/action/trace graph and review surfaces.
- G5: run API and browser console tests; record screenshots.
- G6: update operator scripts on DB/public-service boundaries.
- G7: update subsystem/root/HOWTO docs and reference design.
- G8: run link, terminology, and stale-manual-bootstrap scans.

### H — Fresh Database And Real LLM

- H1: assert DB absent and start service with seed result `inserted`.
- H2: run one `user_message` source case.
- H3: run one action-latch-grounded `internal_thought` source case.
- H4: run one grounded `self_cognition` source case.
- H5: run one claimed `scheduled_tick` source case.
- H6: run one completed `tool_result` source case.
- H6a: prove reply target, mentions, and all admitted media observations survive
  intake-to-episode projection with configured role names and no raw ids.
- H7: exercise every available action and unavailable/degraded path.
- H8: stop/restart service; require seed result `verified` and continuity.
- H8a: stop/restart with a claimed latch, require safe lease recovery, one
  consumed episode, and no duplicate action.
- H9: run focused real-LLM technical cases one at a time.
- H10: run the final 20 group + 20 private sequence one case at a time without
  content evaluation during execution.
- H11: consolidate monologue/dialog/action/trace/latency evidence and native
  schema manifest for user review.

### I — Regression And Handoff

- I1: run affected deterministic suites.
- I2: run static greps, import smoke, diff check, and doc links.
- I3: validate Stage 3 evidence and schema artifacts.
- I4: reconcile Stage 4 target/allowlist links.
- I5: update plan checklist/evidence and registry without completing Stage 3.
- I6: hand full evidence to independent code review.

### J — Independent Review And Completion

- J1: run the parent plan's independent code-review gate with the complete
  plan/companions, diff, Stage 3 evidence, native schema manifest, and Stage 4
  placeholder.
- J2: record every finding and route in-scope production fixes to the original
  implementation owner.
- J3: rerun every deterministic, DB, live-LLM, console, static, budget, or link
  gate affected by remediation.
- J4: present the complete 40-case monologue/dialog report and residual risks
  for user quality review.
- J5: require explicit user Stage 3 completion sign-off.
- J6: set all three Stage 3 documents to `completed`, archive them together,
  update the registry and Stage 4 links, and preserve the evidence paths.

## Exact Verification Commands

Run from repository root with `venv\Scripts\python`.

### Static Inventory

```powershell
rg -n "reflection_signal|scheduled_recall|system_probe|accepted_task_result_ready" src tests
rg -n "internal_thought" src/kazusa_ai_chatbot tests
rg -n "build_episode_trace|has_consolidatable_output" src/kazusa_ai_chatbot tests
rg -n "accepted_task_request|accepted_coding_task_request|accepted_task_status_check|background_work_request|trigger_future_cognition|future_speak" src/kazusa_ai_chatbot tests
rg -n "load_character_profile|CHARACTER_PROFILE_PATH" README.md docs src tests Dockerfile
rg -n "group_0[1-9]|group_[12][0-9]|private_0[1-9]|private_[12][0-9]" src/kazusa_ai_chatbot personalities
```

Checkpoint A records all matches. Final expected result for the first command:
zero matches in normal runtime/tests; historical archive text and the three
Stage-4-only files are allowed only when the exact line describes migration.
The internal-thought command may match only the trigger registry/builder,
self-cognition runtime owner, evidence/consolidation source policies, docs, and
their exact tests; dry-run/scaffold imports are forbidden.
Final trace command may match `action_spec.results` pure helpers and
`brain_service.post_turn.settle_episode_trace`; runtime trace construction in
persona/self-cognition/service is forbidden. The case-id command must return
zero matches; `rg` exit code 1 is expected.

### Focused Deterministic Tests

```powershell
venv\Scripts\python -m pytest tests/test_character_profile_seed.py -q
venv\Scripts\python -m pytest tests/test_internal_action_latches.py -q
venv\Scripts\python -m pytest tests/test_post_turn_lifecycle_record.py -q
venv\Scripts\python -m pytest tests/test_stage3_trigger_source_cutover.py -q
venv\Scripts\python -m pytest tests/test_cognitive_episode_contract.py -q
venv\Scripts\python -m pytest tests/test_action_spec_models.py tests/test_action_spec_results.py tests/test_action_spec_future_cognition.py -q
venv\Scripts\python -m pytest tests/test_stage3_auxiliary_v2_contract.py -q
venv\Scripts\python -m pytest tests/test_consolidation_origin_metadata.py tests/test_consolidation_origin_policy.py tests/test_consolidation_target_routing.py -q
```

Expected: all pass after implementation. Before implementation, new tests fail
only for named missing contracts/old source literals.

### Affected Runtime Regression

```powershell
venv\Scripts\python -m pytest tests/test_self_cognition_integration.py tests/test_self_cognition_tracking.py tests/test_self_cognition_event_logging.py -q
venv\Scripts\python -m pytest tests/test_accepted_task_lifecycle.py tests/test_background_work_delivery.py tests/test_background_work_future_speak.py -q
venv\Scripts\python -m pytest tests/test_service_background_consolidation.py tests/test_service_health.py tests/test_service_ops_status.py tests/test_service_event_logging.py -q
venv\Scripts\python -m pytest tests/test_db.py tests/test_config.py tests/test_script_db_boundary.py -q
venv\Scripts\python -m pytest tests/test_cognition_core_v2_state.py tests/test_cognition_core_v2_integration.py tests/test_cognition_resolver_loop.py -q
```

Expected: all pass. No skip caused by a Stage 3 code failure is accepted.

### Isolated Fresh Database

```powershell
venv\Scripts\python -m pytest tests/test_stage3_fresh_database_bootstrap.py -q
venv\Scripts\python -m pytest tests/test_stage3_background_reasoning_live_db.py -q
```

Expected: each command runs only after guard validation; cold bootstrap,
five-source persistence, trace cardinality, and restart assertions pass.

### Control Console

```powershell
venv\Scripts\python -m pytest tests/test_control_console_contracts.py tests/test_control_console_repository.py tests/test_control_console_cognition_debug_visibility.py tests/test_control_console_redaction.py -q
venv\Scripts\python -m pytest tests/control_console_e2e/test_stage3_fresh_database_e2e.py -q
```

Expected: API and browser tests pass; protected fields remain absent. Browser
evidence follows the control-console skill's cache/stale-Chrome rules.

### Real LLM

Each source/edge case is invoked in its own command:

```powershell
venv\Scripts\python -m pytest tests/test_stage3_fresh_database_e2e_live_llm.py::test_live_user_message_source -q -s
venv\Scripts\python -m pytest tests/test_stage3_fresh_database_e2e_live_llm.py::test_live_internal_thought_source -q -s
venv\Scripts\python -m pytest tests/test_stage3_fresh_database_e2e_live_llm.py::test_live_self_cognition_source -q -s
venv\Scripts\python -m pytest tests/test_stage3_fresh_database_e2e_live_llm.py::test_live_scheduled_tick_source -q -s
venv\Scripts\python -m pytest tests/test_stage3_fresh_database_e2e_live_llm.py::test_live_tool_result_source -q -s
venv\Scripts\python -m pytest tests/test_stage3_fresh_database_e2e_live_llm.py::test_live_group_review_promoted_reflection -q -s
venv\Scripts\python -m pytest tests/test_stage3_fresh_database_e2e_live_llm.py::test_live_media_reply_mentions_preserved -q -s
venv\Scripts\python -m pytest tests/test_stage3_fresh_database_e2e_live_llm.py::test_live_action_affordance_routes -q -s
```

#### Emotion, abuse-boundary, and bounded-error sign-off overlay

These selectors are retained in addition to the frozen source sequence. Run
each selector separately, inspect its raw artifact, and record its disposition
before starting the next selector:

```powershell
venv\Scripts\python -m pytest tests/test_cognition_core_v2_crying_sadness_e2e_live_llm.py::test_live_crying_on_sadness_natural_sequence -q -s -o addopts=
venv\Scripts\python -m pytest tests/test_cognition_core_v2_crying_sadness_e2e_live_llm.py::test_live_crying_on_sadness_explicit_permission -q -s -o addopts=
venv\Scripts\python -m pytest tests/test_cognition_core_v2_crying_sadness_e2e_live_llm.py::test_live_crying_sadness_anger_control -q -s -o addopts=
venv\Scripts\python -m pytest tests/test_cognition_core_v2_secondary_crying_e2e_live_llm.py::test_live_secondary_crying_from_fear -q -s -o addopts=
venv\Scripts\python -m pytest tests/test_cognition_core_v2_secondary_crying_e2e_live_llm.py::test_live_secondary_crying_from_shame -q -s -o addopts=
venv\Scripts\python -m pytest tests/test_cognition_core_v2_secondary_crying_e2e_live_llm.py::test_live_secondary_crying_from_loneliness -q -s -o addopts=
venv\Scripts\python -m pytest tests/test_cognition_core_v2_secondary_crying_e2e_live_llm.py::test_live_secondary_crying_from_anger -q -s -o addopts=
venv\Scripts\python -m pytest tests/test_cognition_core_v2_high_attachment_abuse_e2e_live_llm.py::test_live_high_attachment_sustained_abuse_natural_only -q -s -o addopts=
venv\Scripts\python -m pytest tests/test_cognition_core_v2_high_attachment_abuse_e2e_live_llm.py::test_live_high_attachment_abuse_then_cutoff_natural_only -q -s -o addopts=
venv\Scripts\python -m pytest tests/test_cognition_core_v2_verbal_abuse_boundary_e2e_live_llm.py::test_live_sustained_verbal_abuse_natural_only -q -s -o addopts=
venv\Scripts\python -m pytest tests/test_cognition_core_v2_verbal_abuse_boundary_e2e_live_llm.py::test_live_verbal_abuse_then_rejection_natural_only -q -s -o addopts=
venv\Scripts\python -m pytest tests/test_cognition_core_v2_abuse_to_sadness_e2e_live_llm.py::test_live_abuse_to_sadness_through_valued_loss -q -s -o addopts=
venv\Scripts\python -m pytest tests/test_cognition_core_v2_abuse_to_sadness_e2e_live_llm.py::test_live_abuse_to_sadness_downstream_from_negative_outcome -q -s -o addopts=
venv\Scripts\python -m pytest tests/test_cognition_core_v2_abuse_to_sadness_dialog_e2e_live_llm.py::test_live_abuse_to_sadness_renders_visible_dialog -q -s -o addopts=
venv\Scripts\python -m pytest tests/test_cognition_core_v2_abuse_to_sadness_mechanical.py -q -o addopts=
```

The overlay's readable input/output, monologue, affect, trace, and failure
disposition is recorded in
`test_artifacts/cognition_core_v2/stage_3/emotion_boundary_bounded_error_signoff_review.md`.
The natural abuse and bounded-error failures remain diagnostic sign-off
evidence; invalid candidates never count as affect evidence.

The frozen 40-case fixture copies the sanitized inputs, order, and target-scope
facts from the Stage 2 group/private proof artifacts; it does not copy Stage 2
model outputs or database state. The harness creates the fresh native state and
runs one process per case in sequence. `run-case` uses `cold_start` when the
guarded run-session record is absent and the database is absent; every later
case uses `restart` validation against that record/database. It flushes one
trace/evidence row before the next input. These commands are exact:

```powershell
1..20 | ForEach-Object { $id = 'group_{0:d2}' -f $_; venv\Scripts\python tests/stage3_fresh_database.py run-case --fixture tests/fixtures/stage3_fresh_database_cases.json --case-id $id --output-dir test_artifacts/cognition_core_v2/stage_3/fresh_40_turn_signoff; if ($LASTEXITCODE -ne 0) { throw "fatal Stage 3 case failure: $id" } }
1..20 | ForEach-Object { $id = 'private_{0:d2}' -f $_; venv\Scripts\python tests/stage3_fresh_database.py run-case --fixture tests/fixtures/stage3_fresh_database_cases.json --case-id $id --output-dir test_artifacts/cognition_core_v2/stage_3/fresh_40_turn_signoff; if ($LASTEXITCODE -ne 0) { throw "fatal Stage 3 case failure: $id" } }
venv\Scripts\python tests/stage3_fresh_database.py build-report --fixture tests/fixtures/stage3_fresh_database_cases.json --input-dir test_artifacts/cognition_core_v2/stage_3/fresh_40_turn_signoff --output test_artifacts/cognition_core_v2/stage_3/fresh_40_turn_signoff/cognition_v2_stage3_fresh_40_turn_monologue_dialog_review.md
```

Expected technical result: 40 completed terminal traces, monologue/dialog/
actions/latency for every case, no fatal pipeline failure, and budget thresholds
met. Content judgment occurs only after the complete report is built.

### Final Static/Document Gates

```powershell
venv\Scripts\python -m compileall -q src/kazusa_ai_chatbot src/control_console src/scripts
venv\Scripts\python -m pytest tests/test_stage3_documentation_links.py tests/test_documentation_harmonization.py -q
git diff --check
$placeholderPattern = ('T' + 'BD|T' + 'ODO|similar' + ' to|handle' + ' edge cases|add' + ' tests')
rg -n $placeholderPattern development_plans/active/short_term/cognition_core_v2_stage_3_system_adoption_plan.md development_plans/active/short_term/cognition_core_v2_stage_3_execution_manifest.md development_plans/active/short_term/cognition_core_v2_stage_3_change_radius.md
```

Expected: compile, link, harmonization, and diff checks pass. Placeholder scan
returns zero matches; `rg` exit code 1 is the expected zero-match result.

## Stage 4 Handoff Inventory

Stage 3 hands Stage 4:

- approved/completed parent plan, this execution manifest, and the mandatory
  change-radius companion;
- `Stage3NativeSchemaManifestV1`;
- `Stage3FreshDatabaseEvidenceV1`;
- preliminary source-code legacy collection/field/trigger inventory;
- the exact three-file Stage-4-only allowlist;
- fresh-database indexes/TTL/ownership evidence;
- unresolved production-data facts explicitly left for Stage 4 discovery;
- final independent review and user quality sign-off.

Stage 3 does not hand off production credentials, production samples, assumed
row counts, or an executable transform derived without production discovery.

## Execution Evidence

- Independent plan review: 2026-07-18 review and re-review findings incorporated;
  further agent review stopped by user instruction; user authorized execution
  on 2026-07-19.
- Checkpoint A baseline: complete on 2026-07-19. Evidence includes the exact
  occurrence inventories, pre-service environment guard results, focused
  expected-failure output, and call/latency baseline under
  `test_artifacts/cognition_core_v2/stage_3/`.
- Database-isolation amendment: user-authorized on 2026-07-19 because a
  separate MongoDB endpoint is unavailable; the exact reserved database name
  is now the primary guard on the configured URI.
- Deterministic implementation verification through the current cutover is
  recorded in
  `test_artifacts/cognition_core_v2/stage_3/checkpoint_i_verification_summary.md`.
  The current record includes 100 focused contract passes, 69
  availability/profile/latch/lifecycle passes, 79 canonical-source passes,
  157 affected-runtime passes, 121 DB/config/script passes with 7 live cases
  deselected, 56 cognition-loop passes with 4 intentional deselections, 5
  health/ops passes, 27 console API passes, 8 documentation passes, 5
  architecture-doc passes, and 9 fresh-harness passes. The database guard
  contract suite adds 4 passes. Console external E2E,
  collection, compile, diff, and static scans are green.
- Answerability remediation verification: focused contract/capability coverage
  now includes typed blocker, one-final-cognition convergence, source
  separation, and URI-redaction tests. The clean follow-up sequence under
  `test_artifacts/cognition_core_v2/stage_3/answerability_remediation_followup2/`
  recorded `private_18` at 8 calls and 65.7 seconds with no resolver stage and
  a complete direct answer; `private_08` at 11 calls and 91.6 seconds with one
  blocked local-context attempt and a semantically correct raw-language
  clarification. The first post-blocker retry exposed a silent final pass and
  is preserved in the prior follow-up directory; the typed fallback corrected
  it. The direct-answer and required-user-input convergence improvements are
  demonstrated; the full sequential p95 gate remained open under the then-current
  baseline-relative threshold. The deterministic
  source-separation regression supplies a `conversation_evidence.resolved=false`
  row, preserves an LLM-owned `answerable_now` decision, and passes with one
  cognition call and zero capability executions. The latest affected
  deterministic rerun recorded 135 passes, one preexisting route-expectation
  failure, and seven deselected tests.
- Checkpoint H technical execution is complete: the configured URI/name
  preflight passed with the exact `_test_kazusa_core_v2` database guard. The guarded
  output directory completed cold-start bootstrap and all 40 frozen
  group/private restart cases; each clean result produced one terminal visible
  trace and one lifecycle record with a succeeded persisted trace. The
  refreshed sequence captured 313 LLM calls, a 75.6-second average foreground
  duration, a 103.8-second nearest-rank p95, and a 125.4-second maximum.
  `group_17` and `group_18` required exact failed-row cleanup after the
  harness's 15-second trace poll produced false negatives; both passed after
  the poll window was widened to 60 seconds. The eight focused source/edge
  real-LLM commands
  completed individually; their source comparison and latency disposition are
  recorded in
  `test_artifacts/cognition_core_v2/stage_3/focused_source_comparison.md`.
  Final residual-risk review and user quality signoff remain pending.
- The revised latency gate passes: the refreshed Stage 3 nearest-rank p95 is
  103,807 ms, within the user-approved 120,000 ms ceiling. The 125,418 ms
  maximum remains a separately reported tail metric. No LLM budget or output
  cap was raised.
- The final broader non-live collection recorded 3,229 passed, 2 skipped, 21
  failures, and 744 deselected after `tests/conftest.py` supplied the
  canonical profile path by default. The 21 failures are outside the
  manifest's targeted affected-regression command set but within the broader
  change-radius inventory; the exact list is recorded in
  `test_artifacts/cognition_core_v2/stage_3/broad_non_live_regression_summary.md`.
  Checkpoint I remains open for that reconciliation.
- In-app Browser acceptance is externally pending because no Browser session
  is available; the external console E2E passes.
- Independent review completed under reviewer identity `Hilbert`. The review
  findings were remediated or dispositioned, and the affected deterministic
  gates were rerun. The user approved the Phase 3 artifact set on 2026-07-22;
  external Browser acceptance and the remaining lifecycle completion checks
  remain pending.
- Checkpoint J artifact-quality sign-off: complete on 2026-07-22.
- Checkpoint J full Stage 3 completion: pending external Browser acceptance and
  the remaining lifecycle completion checks.

## Latest Verification Addendum — 2026-07-21 Chinese-Only Semantic Contract

- The model-facing semantic contract is now Chinese-only. The temporary
  dual-language projection branch was removed from interaction-style,
  conversation-progress, cognition, goal, action, and surface prompts.
  Chinese context therefore uses Chinese role labels and Chinese explanatory
  text consistently.
- Exact machine-facing handles and fixed enum/schema tokens remain only where
  typed validation or persistence requires them (for example `self`,
  `current_user`, `actor`, `target`, `speech`, and `action`). These are
  structural contract values, not natural-language labels emitted in the
  Chinese semantic text.
- The final representative real-LLM dialog run passed:
  `test_artifacts/cognition_core_v2/crying_sadness_e2e/abuse_to_sadness_visible_dialog_82f0a133f1/`.
  It produced visible Chinese dialog with a final sadness projection at
  `极高` intensity and no forbidden English role terms in the inspected
  cognition, semantic, surface, or response-message fields.
- The retained emotion/boundary sign-off selectors were executed one at a
  time and inspected: sadness-to-crying (3), secondary crying probes for fear,
  shame, loneliness, and anger (4), high-attachment abuse (2), verbal-abuse
  boundary (2), natural and seeded abuse-to-sadness (2), visible dialog (1),
  and deterministic mechanical reachability (1): 15 selectors passed.
- The latest non-live repository collection recorded `3,264 passed`, `2
  skipped`, and `758 deselected`. Control-console gates recorded `25 passed`,
  and the fresh-database console E2E recorded `1 passed`. Compile, diff, and
  targeted Chinese-only contract scans passed.
- The shame probe now clears isolated affect activations before derivation;
  legal `completed_action` terminals with accepted action requests are
  recognized by the abuse-boundary technical assertion. These are test and
  evidence-contract corrections and do not add a crying tag or a verbal
  permission path.
- The user approved the Phase 3 artifact set on 2026-07-22. Checkpoint J
  artifact-quality sign-off is complete; external Browser acceptance and the
  remaining lifecycle completion checks are still pending.
