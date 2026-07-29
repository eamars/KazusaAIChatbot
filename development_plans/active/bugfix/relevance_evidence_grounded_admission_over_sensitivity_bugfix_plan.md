# Relevance evidence-grounded admission over-sensitivity bugfix plan

## Summary

- Goal: prevent unaddressed group messages from entering cognition through
  invented recipient grounding while preserving character speech when either
  the interaction is genuinely relevant or the message materially intersects
  the character's active state.
- Plan class: `large`.
- Status: `draft`.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `llm-trace-debug`, `debug-llm`, `py-style`,
  `cjk-safety`, and `test-style-and-execution`.
- Overall cutover strategy: big-bang replacement of the model-facing
  participation-evidence contract, with compatible public relevance actions,
  model route, call count, coordinator boundary, and input/completion caps.
- Highest-risk areas: turning evidence validation into a deterministic text
  classifier, allowing generic character traits to admit ambient group noise,
  losing valid canonical-name or continuity cases, dropping cited evidence
  during prompt fitting, and colliding with parallel edits to `service.py`.
- Acceptance criteria: the captured
  `a72b0182de174ec0b0ff533891c2e294` case is discarded at frontline and ignored
  if forced through settled relevance; `"一"` and `"你"` cannot become character
  address evidence by themselves; canonical full-name, typed target/reply,
  private-scope, open-turn, and bot-continuity positives remain admitted;
  concrete active-state salience can admit a message addressed to someone else
  without rewriting its recipient; public relevance actions and LLM budgets
  remain unchanged; and the implementation changes none of the protected
  parallel-development paths.

## Context

The relevance package is the owner of cognition admission:

```text
typed message
  -> frontline relevance: discard | start | append
  -> settled relevance: ignore | proceed | wait
  -> atomic claim
  -> cognition
```

`kazusa_ai_chatbot.relevance` therefore owns the semantic decision that must
stop an irrelevant message before cognition. The settlement coordinator owns
ordering, deadlines, slots, and the atomic claim; it does not reinterpret a
valid relevance decision. Cognition is downstream and cannot repair an
incorrect admission without already paying the cost and accepting polluted
context.

The captured failure evidence is:

- delivery/conversation correlation id:
  `a72b0182de174ec0b0ff533891c2e294`;
- active character: `一之濑明日奈`;
- current text: `直接找你一换一是吧`;
- typed character target, reply target, broadcast, open turn, and bot
  continuity: absent;
- frontline prompt SHA-256:
  `58349a1122ee473bf76c5230e9faeac65018a79872051db80f095fddf10b2c82`;
- frontline output SHA-256:
  `40d769a4ea0fddce3cd41034d2e38adda9a76187e8f76e7d93adc6c2f7d654b9`;
- frontline output:
  `{"intake_action":"start","append_target":"none","prelude_targets":[],"reason":"明确使用角色规范名称称呼"}`;
- settled-input SHA-256:
  `b00bf9441f1fa125369493f8c1dd7c6beb8773f92545ecadb1fca45b79e056da`.

Exact replay showed that the local relevance model treated the numeral `"一"`
inside `"一换一"` as a nickname or prefix for `一之濑明日奈`, then bound the
deictic pronoun `"你"` to the character. Counterfactual replay stopped the
admission when the character name changed, `"一"` changed or disappeared, or
`"你"` disappeared.

The root cause is an ownership-contract defect, not the visible bad action:

1. Both relevance prompts permit the model to invent natural-name address
   evidence from unrestricted message text.
2. Neither model response has to cite a supplied recipient or participation
   evidence item.
3. Deterministic validation checks JSON shape and slot existence but cannot
   reject a semantically unsupported name claim.
4. Settled history collapses every non-current human to
   `other_participant`, removing stable identity needed to ground who `"你"`
   refers to.
5. The same weak model applies substantially the same unrestricted heuristic
   at both relevance gates, so the first false grounding is repeated instead
   of corrected.
6. The full decontextualizer runs only after the settled `proceed` claim, so it
   cannot protect cognition admission.

There is also a policy defect. The desired rule is:

```text
interaction relevance
OR
concrete character-state salience
```

The current settled prompt instead says that interesting content cannot
establish relevance. At the same time, `service.py` passes
`_runtime_character_state.get("mood", "")`, while the runtime snapshot stores
the native state under `cognition_state`. The relevance model therefore sees
neither a valid state-salience lane nor a valid way to cite one.

The current worktree contains an in-progress conversation-progress migration
that already changes `service.py`, cognition, node, database, RAG, state, and
utility files. This plan captures the relevance bug independently. It does not
consume, revise, copy, clean up, or test-drive those parallel changes.

## Mandatory Skills

- `development-plan`: load before approving, executing, reviewing, updating,
  signing off, or archiving this plan.
- `local-llm-architecture`: load before changing relevance prompts, evidence
  projections, response contracts, context fitting, or model-call behavior.
- `no-prepost-user-input`: load before designing or reviewing deterministic
  evidence extraction and validation. Deterministic code may expose and
  validate provenance; the LLM remains the semantic judge.
- `llm-trace-debug`: load before retrieving or correlating protected incident
  or regression traces.
- `debug-llm`: load before running real relevance calls or producing the
  required human-readable LLM evaluation artifacts.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python prompts or tests containing CJK
  text.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute production changes while this plan status is `draft`.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, documentation, and plan edits.
- Check `git status --short` before every implementation stage and before final
  sign-off.
- Do not read `.env`.
- Treat the current parallel-development diff as protected user work.
- Do not modify, stage, format, revert, move, or include in this plan's commit
  any existing parallel change under:
  - `src/kazusa_ai_chatbot/conversation_progress/**`;
  - `src/kazusa_ai_chatbot/db/**`;
  - `src/kazusa_ai_chatbot/rag/**`;
  - `src/kazusa_ai_chatbot/nodes/**`;
  - `src/kazusa_ai_chatbot/cognition_core_v2/**`;
  - `src/kazusa_ai_chatbot/brain_service/post_turn.py`;
  - `src/kazusa_ai_chatbot/state.py`;
  - `src/kazusa_ai_chatbot/utils.py`;
  - `src/control_console/**`;
  - `src/scripts/migrate_conversation_progress_v2.py`;
  - conversation-progress tests and fixtures; or
  - the in-progress conversation-progress plan.
- Start implementation from an isolated branch or worktree based on the
  committed post-parallel baseline. Do not implement against the current dirty
  `service.py`.
- `service.py` integration is allowed only after its parallel owner has
  completed and released the file. The implementation agent must record the
  baseline commit and confirm that only `_build_frontline_state(...)` and
  `_settled_state_from_lease(...)` need relevance-specific edits.
- If either service symbol or either relevance-agent contract has materially
  changed at that baseline, stop and request plan reconciliation. Do not absorb
  the parallel design into this bugfix.
- The draft
  `cognition_core_v2_short_horizon_global_state_composition_bigbang_plan.md`
  also names the relevance agents and `service.py`. Its relevance slice and
  this plan are mutually exclusive execution scopes. Before either plan is
  approved for those files, lifecycle ownership must identify one plan as the
  sole owner; this plan does not edit the other plan.
- Preserve the public `FrontlineDecision` fields and actions:
  `discard|start|append`.
- Preserve the public `SettledRelevanceDecision` fields and actions:
  `ignore|proceed|wait`.
- Preserve the existing settlement coordinator, atomic cognition claim,
  deadlines, quiet window, hard deadline, prelude rules, open-turn slots,
  native-reply feasibility boundary, and failure lifecycle.
- Preserve the existing `RELEVANCE_AGENT_LLM` route, FIFO one-in-flight
  executor, zero-thinking configuration, and model-call count.
- Keep the frontline rendered-input cap at 8,000 characters and completion cap
  at 256 tokens.
- Keep the settled rendered-input cap at 16,000 characters and completion cap
  at 512 tokens.
- Do not add a model, retry loop, JSON-repair call, embedding call, retrieval
  call, database read, cache layer, feature flag, compatibility module, or
  alternate relevance path.
- Continue parsing every model response through
  `parse_llm_json_output(..., deterministic_only=True)`.
- The LLM owns recipient interpretation, contextual reference, explicit
  whole-group invitation, semantic continuity, and whether a message
  materially intersects active character state.
- Deterministic code owns typed protocol facts, bounded projection, stable
  opaque handles, literal full-name candidate spans, evidence-reference
  existence, cross-field consistency, caps, and fail-closed disposition.
- Literal name matching is provenance generation only. It must never directly
  choose `start`, `append`, `proceed`, or `ignore`.
- `"你"`, `"你们"`, another pronoun, one character from the canonical name, a
  numeral, or a name prefix is never standalone character-recipient evidence.
- A canonical-name evidence item exists only when the complete current
  `active_character_name` occurs contiguously in the visible message.
- Do not invent a character alias list. Approved aliases remain deferred until
  a canonical character-profile alias contract exists.
- Generic answerability, helpfulness, sentiment, topic popularity, and stable
  personality traits do not establish character-state salience.
- Character-state admission requires a concrete semantic intersection with at
  least one supplied active-state evidence reference.
- Preserve the model's actual recipient independently from its reason to
  admit. State-salience admission may retain `participant_n` or `unknown` as
  recipient and must not rewrite that recipient to `character`.
- `use_reply_feature` must remain `false` for state-salience admission when the
  recipient is not the character.
- Evidence references absent from the final cap-fitted payload are invalid.
- Unsupported non-authoritative admission fails closed:
  `discard` at frontline and `ignore` at settled relevance.
- Preserve the existing authoritative typed-target behavior and its bounded
  settled repair/failure policy.
- After automatic context compaction, reread this entire plan before
  implementation, verification, review, lifecycle changes, or final
  reporting.
- Before final completion, run the `Independent Code Review` gate and record
  its result under `Execution Evidence`.

## Must Do

- Add one shared relevance-local participation-evidence module.
- Generate bounded interaction evidence from private scope, typed
  target/reply/broadcast fields, the complete canonical-name span, open-turn
  slots, bot continuity, and settled history.
- Add a current-message evidence item that the LLM may use only for an
  explicitly whole-group invitation, never as character-name proof.
- Replace settled `other_participant` identity collapse with stable
  `participant_1` through `participant_8` handles across speaker, target, and
  reply projections.
- Project a bounded set of active character-state candidates from the native
  `cognition_state.v2` snapshot without exposing raw ids or numeric telemetry.
- Add `recipient_relation`, `admission_basis`,
  `interaction_evidence_refs`, and `character_state_refs` to each
  model-facing relevance output contract.
- Validate every returned reference against the exact final payload and enforce
  action/basis/recipient consistency before returning the existing public
  decision.
- Record the validated participation assessment beside the existing public
  decision in protected LLM trace diagnostics.
- Update frontline policy so a group message may start or append only through
  interaction relevance or concrete character-state salience.
- Update settled policy so a group turn may proceed through either of those
  bases while retaining the actual recipient.
- Pass the current character cognition-state snapshot into both relevance
  state builders after the exclusive `service.py` integration gate.
- Freeze the redacted incident and counterfactuals as a source-faithful test
  fixture.
- Add deterministic contract, projection, cap-fitting, participant-handle,
  fail-closed, and service-wiring tests.
- Add one-at-a-time real-LLM tests for the incident, canonical-name positive,
  state-salience positive, and unmatched-state negative at both relevance
  stages.
- Update relevance, brain-service, root architecture, and operational
  documentation to state the exact two-basis cognition-admission rule.

## Deferred

- Character nickname, short-name, transliteration, honorific, fuzzy-name, and
  learned-alias support.
- Adapter mention parsing, reply hydration, message-envelope normalization, or
  platform-specific syntax changes.
- Any database schema, migration, query, collection, index, or persistence
  change.
- Any change to conversation progress, RAG, cognition core, persona nodes,
  decontextualization, dialog, memory, consolidation, reflection, scheduler,
  control console, or adapters.
- Moving the full message decontextualizer before relevance.
- Adding a third relevance gate, verifier model, ensemble, embedding
  similarity, or keyword classifier.
- Static hobby or personality-interest admission. This plan uses only current
  native character state with active pressure or salience.
- Relationship-state redesign, operational-state composition, carry-over
  cognition, state decay, or UI state visualization.
- Tuning response ratio, random participation, channel-noise thresholds, or
  generic engagement propensity.
- Changing native-reply rendering or quote-target selection.
- Cleaning up unrelated legacy fields in `service.py`, including the obsolete
  mood projection, unless a separately approved owner removes them.

## Cutover Policy

Overall strategy: big-bang for the model-facing evidence contract; compatible
for public runtime decisions and operational behavior.

| Area | Policy | Instruction |
|---|---|---|
| Model-facing participation output | big-bang | Replace free-form unsupported participation reasoning with the exact evidence-referenced assessment in both relevance agents. |
| Canonical-name evidence | big-bang | Accept only a contiguous complete `active_character_name` candidate; remove prompt permission to infer short names or prefixes. |
| Settled participant relations | big-bang | Replace undifferentiated other humans with stable bounded `participant_n` handles in one contract update. |
| Character-state relevance | big-bang | Introduce the active-state evidence lane at both gates; remove the blanket policy that interesting content can never establish relevance. |
| Public relevance decisions | compatible | Preserve action names, public fields, coordinator consumers, graph routing, and `should_respond` mapping. |
| Service integration | additive | Add the same native cognition-state snapshot to both relevance state builders after exclusive ownership is available. |
| LLM routing and budgets | compatible | Preserve route, call count, caps, temperature, thinking, FIFO, and existing authoritative repair behavior. |
| Tests and documentation | big-bang | Replace old prompt assertions and fixtures with the exact two-basis rule and evidence contract. |

## Cutover Policy Enforcement

- Rewrite the two model-facing contracts in place; do not add legacy and new
  prompt branches.
- Keep compatibility only for the public surfaces explicitly listed above.
- Do not add field aliases or fallback mappers for the old prompt output.
- Update each producing prompt, parser, trace projection, fixture, and test in
  the same implementation scope.
- Any change to this cutover table requires user approval before
  implementation.

## Target State

The group-chat cognition-admission rule is:

```text
interaction_relevant =
    typed character target/reply
    OR typed/explicit whole-group invitation
    OR complete canonical-name address
    OR exact open-turn/bot/history continuity grounded by supplied evidence

character_state_salient =
    current message materially intersects at least one supplied active-state ref
    AND speaking could advance, protect, resolve, or investigate that state

frontline admission =
    interaction_relevant OR character_state_salient

cognition admission =
    settled proceed
    AND (interaction_relevant OR character_state_salient)
```

Private scope remains interaction-relevant by protocol.

For the captured incident with no matching active state:

```json
{
  "intake_action": "discard",
  "append_target": "none",
  "prelude_targets": [],
  "recipient_relation": "unknown",
  "admission_basis": "none",
  "interaction_evidence_refs": [],
  "character_state_refs": [],
  "reason": "代词没有角色接收者依据；也没有角色状态交集证据"
}
```

Settled history may resolve the recipient to `participant_1`; typed target or
reply evidence may resolve it only as `other_participant`. If the bounded
evidence does not establish who `"你"` means, the correct `recipient_relation`
is `unknown`; it is never changed to `character` merely because the current
text contains `"一"` or `"你"`.

For a message directed to another participant that concretely intersects an
active character knowledge gap:

```json
{
  "response_action": "proceed",
  "recipient_relation": "participant_1",
  "admission_basis": "character_state_salience",
  "interaction_evidence_refs": ["history_2"],
  "character_state_refs": ["state_1"],
  "reason_to_respond": "消息直接补充了角色正在追查的未解信息",
  "use_reply_feature": false,
  "channel_topic": "相关未解信息",
  "indirect_speech_context": "当前话语原本指向 participant_1"
}
```

The model-facing assessment is validated and then stripped before returning
the existing public decision shape. The protected trace retains it for RCA:

```python
{
    **public_decision,
    "participation_assessment": {
        "recipient_relation": "participant_1",
        "admission_basis": "character_state_salience",
        "interaction_evidence_refs": ["history_2"],
        "character_state_refs": ["state_1"],
    },
}
```

## Design Decisions

- Relevance remains the sole live admission owner. Cognition and dialog do not
  receive a compensating suppression rule.
- The fix constrains evidence rather than suppressing pronouns or numerals.
  `"你"` remains interpretable when supplied interaction history grounds it.
- Deterministic literal matching creates only a complete canonical-name
  candidate. The LLM still decides whether that occurrence is an address,
  quotation, correction, or unrelated mention.
- The active character name remains visible as identity context, but it cannot
  support a character-recipient claim unless a corresponding evidence ref is
  present.
- Current-message text remains available as semantic evidence. Its generic
  ref can support an explicit whole-group invitation but cannot support a
  canonical-name or direct-character claim.
- Stable participant handles preserve discourse identity without exposing
  platform or global user ids.
- Recipient identity and reason to speak are separate judgments. This permits
  believable interjection when character state makes an otherwise indirect
  message salient.
- Character-state candidates are transient and bounded. Stable standards and
  default low-pressure drives are excluded because they would turn broad
  character traits into ambient-message admission.
- State candidate selection uses deterministic lifecycle and pressure
  thresholds only to bound context. The LLM decides semantic intersection;
  code does not compare message keywords with state descriptions.
- Both relevance gates consume the same evidence vocabulary so settled
  relevance can correct, rather than repeat, a claim unsupported by its final
  payload.
- The existing full decontextualizer remains after admission. Reordering it
  would expand latency, context, and ownership scope.
- Public action shapes remain stable because the coordinator needs only the
  validated action. Evidence assessment is an internal semantic/trace
  contract.
- The service reuses the already loaded process-local character snapshot.
  Relevance performs no database read.

## Contracts And Data Shapes

### Internal participation assessment

Both model-facing outputs include these exact internal fields:

```python
class ParticipationAssessment(TypedDict):
    recipient_relation: Literal[
        "character",
        "group",
        "current_author",
        "other_participant",
        "participant_1",
        "participant_2",
        "participant_3",
        "participant_4",
        "participant_5",
        "participant_6",
        "participant_7",
        "participant_8",
        "unknown",
    ]
    admission_basis: Literal[
        "interaction_relevance",
        "character_state_salience",
        "none",
    ]
    interaction_evidence_refs: list[str]
    character_state_refs: list[str]
```

Each evidence list contains no duplicates and at most three refs. Public
`FrontlineDecision` and `SettledRelevanceDecision` remain unchanged after the
internal assessment is validated.

### Interaction evidence catalog

The final cap-fitted payload contains:

```python
class InteractionEvidenceItem(TypedDict):
    ref: str
    kind: Literal[
        "private_scope",
        "typed_character_target",
        "typed_character_reply",
        "typed_broadcast",
        "typed_other_target",
        "typed_other_reply",
        "typed_unknown_reply",
        "canonical_name_span",
        "current_message",
        "open_turn",
        "bot_continuity",
        "history_context",
    ]
    summary: str
```

Deterministic ref names are:

- `scope_private`;
- `target_character`;
- `reply_character`;
- `target_broadcast`;
- `target_other`;
- `reply_other`;
- `reply_unknown`;
- `name_1`;
- `message_1`;
- existing `open_1` through `open_3`;
- `continuity_1`; and
- `history_1` through `history_10`.

`name_1` is present only when the complete canonical name is a contiguous
substring of the final visible current-message body. `message_1` is always
present but can support only an explicit group invitation, recipient
correction, or non-character/unknown recipient judgment; it cannot validate
direct character address. Negative typed recipient refs may explain
`discard|ignore`, but they cannot satisfy `interaction_relevance`.

### Stable participant handles

For the final capped settled history:

1. Reserve `character` and `current_author`.
2. Scan rows oldest to newest.
3. Assign the first distinct remaining resolved human identity to
   `participant_1`, then continue through `participant_8`.
4. Reuse the same handle in `speaker_relation`, target summaries, reply
   summaries, and history evidence.
5. Map unresolved identity to `unknown_participant`.
6. If more than eight other identities remain, map the overflow to
   `other_participants` and do not expose it as a selectable recipient.
7. Keep all raw global/platform ids outside the model payload and trace
   assessment.

### Character-state evidence catalog

`service.py` supplies the process-local native character cognition state.
`participation_evidence.py` validates and projects it into:

```python
class CharacterStateEvidenceItem(TypedDict):
    ref: str
    kind: Literal[
        "goal",
        "threat",
        "event",
        "knowledge_gap",
        "affect",
        "drive",
        "meaning",
    ]
    summary: str
    attention: Literal["active", "pressured"]
```

Candidate eligibility is exact:

- goal: status `pursuing|blocked` and salience at least 25;
- threat: status `active` and salience or residual pressure at least 25;
- event: status `active` and salience at least 25;
- knowledge gap: status `open|reduced` and salience at least 25;
- affect: phase `active`, cause status `active`, score at least 40, and a
  resolvable active root description;
- drive: pressure at least 61; and
- meaning: meaning-state salience at least 61.

Standards are excluded. State items are sorted by descending source strength,
then fixed kind order
`threat, goal, knowledge_gap, event, affect, drive, meaning`, then source
order. The model receives at most six items, each summary is at most 160
characters, and the rendered state catalog is at most 1,400 characters. Raw
ids, timestamps, evidence refs, scalar values, and owner ids remain private.
Source strength means goal/event/knowledge-gap salience, the greater of threat
salience and residual pressure, affect score, drive pressure, or meaning-state
salience, as applicable.

The semantic admission test is still performed by the LLM. A supplied state
item is a candidate, not proof that the current message intersects it.

### Model-only frontline output

```json
{
  "intake_action": "discard|start|append",
  "append_target": "none|open_1|open_2|open_3",
  "prelude_targets": [],
  "recipient_relation": "character|group|current_author|other_participant|participant_1..participant_8|unknown",
  "admission_basis": "interaction_relevance|character_state_salience|none",
  "interaction_evidence_refs": [],
  "character_state_refs": [],
  "reason": "最多 80 字符"
}
```

### Model-only settled output

```json
{
  "response_action": "ignore|proceed|wait",
  "recipient_relation": "character|group|current_author|other_participant|participant_1..participant_8|unknown",
  "admission_basis": "interaction_relevance|character_state_salience|none",
  "interaction_evidence_refs": [],
  "character_state_refs": [],
  "reason_to_respond": "最多 180 字符",
  "use_reply_feature": false,
  "channel_topic": "最多 60 字符",
  "indirect_speech_context": "最多 100 字符"
}
```

The authoritative settled semantic-disposition contract retains its current
action vocabulary and bounded retry policy. Its deterministic mapping creates
the corresponding internal assessment from typed refs before returning the
existing public decision.

### Validation matrix

| Condition | Required result |
|---|---|
| Frontline `start|append` | `admission_basis` is not `none`. |
| Frontline `discard` | `admission_basis == none`; no character-state refs. |
| Frontline `append` | selected append slot is visible and cited as interaction evidence. |
| Settled `proceed|wait` | `admission_basis` is not `none`. |
| Settled `ignore` | May retain a valid basis for `already_resolved`, recipient withdrawal, or unavailable media; otherwise uses `none`. |
| `interaction_relevance` | At least one visible interaction ref supports the claimed recipient/continuity class. |
| Group recipient `character` | Requires typed character target/reply, complete-name, open-turn, bot-continuity, or character-grounded history ref; `message_1` alone is invalid. |
| Group recipient `group` | Requires typed broadcast or `message_1` judged as an explicit whole-group invitation. |
| Recipient `participant_n` | Requires a cited visible history row carrying the same stable handle. |
| Recipient `other_participant` | Requires typed other-target/reply evidence or `message_1` as a semantic non-character-recipient judgment; it cannot establish admission. |
| `character_state_salience` | At least one visible state ref; recipient may remain another participant or unknown. |
| State salience with non-character recipient | `use_reply_feature == false`. |
| Non-`proceed` action | `use_reply_feature == false`. |
| Missing, duplicate, invented, dropped, or wrong-kind ref | Contract-invalid and follows the existing fail-closed/authoritative failure path. |

## LLM Call And Context Budget

### Before

| Stage | Calls | Input cap | Completion cap | Thinking |
|---|---:|---:|---:|---|
| Frontline relevance | At most one per active message, with existing deterministic authoritative shortcut | 8,000 characters | 256 tokens | disabled |
| Settled relevance | At most one per assessment, plus the existing bounded authoritative hard-deadline repair when applicable | 16,000 characters | 512 tokens | disabled |

### After

The table remains identical. This plan adds zero normal-path LLM calls, zero
model routes, zero retries, and zero database/retrieval calls.

Prompt fitting rules are:

1. Preserve system policy, exact output contract, current message, typed
   target/reply evidence, and complete-name candidate.
2. Preserve only action/slot refs visible in the final payload.
3. Remove lowest-strength character-state candidates first.
4. Remove oldest settled history evidence next while retaining the effective
   latest fragment.
5. Apply the existing prelude, continuity, history, fragment, and open-turn
   reductions.
6. Rebuild the evidence catalog and action contract after every reduction.
7. Reject the projection if it still exceeds the existing hard cap.

Worst-case tests must prove:

- frontline rendered system plus human content is at most 8,000 characters;
- settled rendered system plus human content is at most 16,000 characters;
- every returned/citable ref exists in the exact serialized human payload;
- no raw identity or native character-state telemetry appears; and
- the action space is regenerated after slots or evidence are dropped.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/relevance/participation_evidence.py`
  - Shared bounded interaction/state evidence projection, stable participant
    handles, internal assessment validation, and trace-safe projection.
- `tests/fixtures/relevance/a72b0182de174ec0b0ff533891c2e294.json`
  - Redacted source-faithful incident, relevant bounded history, typed envelope
    facts, old hashes, and counterfactual cases.
- `tests/test_relevance_participation_evidence.py`
  - Deterministic evidence, state selection, handle, consistency, and
    cap-fitting tests.
- `tests/test_relevance_evidence_grounding_live_llm.py`
  - One-case-at-a-time real-LLM frontline and settled regressions.

### Modify

- `src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py`
  - Evidence catalog, two-basis prompt, model-only output, exact validation,
    fail-closed mapping, and trace assessment.
- `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py`
  - Stable history handles, evidence/state projection, two-basis prompt,
    exact validation, authoritative assessment mapping, recipient-preserving
    reply rule, and trace assessment.
- `src/kazusa_ai_chatbot/service.py`
  - After exclusive integration is available, pass a copied native cognition
    state only from `_build_frontline_state(...)` and
    `_settled_state_from_lease(...)`.
- `src/kazusa_ai_chatbot/relevance/README.md`
  - Normative two-basis admission, evidence, participant, state, and public
    compatibility contract.
- `src/kazusa_ai_chatbot/brain_service/README.md`
  - Service-owned state-snapshot handoff and unchanged claim/lifecycle
    boundary.
- `README.md`
  - Architecture-level statement of evidence-grounded interaction or
    character-state admission.
- `docs/HOWTO.md`
  - Operational payload, caps, route, and real-LLM verification commands.
- `tests/test_frontline_relevance_agent.py`
  - Prompt, parser, evidence-ref, state, cap, shortcut, and fail-closed tests.
- `tests/test_persona_relevance_agent.py`
  - Prompt, parser, stable handles, state, cap, authoritative, and reply-anchor
    tests.
- `tests/test_service_input_queue.py`
  - Exact process-local state handoff at both service projection symbols.
- `development_plans/README.md`
  - Lifecycle registration and later status/archive transitions only.

### Verify Without Modifying

- `src/kazusa_ai_chatbot/brain_service/turn_settlement.py`
- `src/kazusa_ai_chatbot/brain_service/graph.py`
- `src/kazusa_ai_chatbot/relevance/__init__.py`
- `src/kazusa_ai_chatbot/state.py`
- `tests/test_relevance_turn_settlement.py`
- `tests/test_relevance_turn_settlement_graph.py`
- existing relevance live-LLM suites

### Delete

- None.

## Overdesign Guardrail

- Add exactly one production module.
- Keep exactly two semantic LLM gates and their current public actions.
- Keep evidence refs ephemeral and trace-local; do not persist them.
- Use existing native cognition state; do not add an operational-state model.
- Use literal complete-name provenance and stable opaque handles; do not build
  alias resolution, NER, embeddings, regex classification, or a parser.
- Keep state projection to six candidates and 1,400 characters.
- Keep the full decontextualizer in its current location.
- Keep service integration to two named symbols after the exclusive gate.
- If implementation requires a new database read, model call, public state
  field, coordinator action, graph node, or compatibility layer, stop and
  request a plan revision.

## Agent Autonomy Boundaries

The implementation agent may:

- make the exact create/modify changes listed under `Change Surface`;
- update adjacent relevance tests when required by the new exact internal
  contract;
- tighten local caps within the stated 8,000/16,000 outer limits;
- improve prompt wording while preserving the exact rule and output shape;
- generate test/debug artifacts under
  `test_artifacts/relevance_evidence_grounding/`; and
- fix defects found by the focused relevance tests inside the listed files.

The implementation agent must stop and request direction if:

- any target file has uncommitted changes owned by parallel work;
- the post-parallel baseline changes the native state shape or either named
  service projection symbol;
- the fix would require touching a protected path;
- a valid direct-address case requires fuzzy or alias matching;
- the 256-token frontline completion cap cannot hold the exact output;
- the 8,000/16,000-character caps cannot hold the required evidence;
- a new LLM call, model route, retry, DB read, cache, graph node, public action,
  or persisted field appears necessary;
- deterministic code would need to decide semantic relevance from message
  words; or
- real-LLM evidence shows the two-basis rule cannot be made reliable within
  this contract.

The implementation agent may not:

- reformat or clean unrelated files;
- stage the whole worktree;
- alter the parallel plan or its evidence;
- reinterpret existing parallel diffs as part of this bugfix; or
- mark the plan complete without the required real-LLM artifacts and
  independent code review.

## Implementation Order

1. Obtain explicit implementation authorization and change this plan to
   `approved` or `in_progress`.
2. Wait for the current `service.py` parallel owner to finish and provide a
   committed baseline.
3. Create an isolated branch/worktree from that baseline; record commit,
   `git status --short`, and target-file hashes.
4. Confirm the change-surface allowlist and mutual exclusion with the
   global-state draft's relevance slice.
5. Freeze the redacted incident and counterfactual fixture without querying or
   changing production data.
6. Add deterministic evidence-contract tests.
7. Add `participation_evidence.py` and make those tests pass.
8. Change frontline projection, prompt, parser, trace, and tests.
9. Change settled projection, stable handles, prompt, parser, authoritative
   mapping, trace, and tests.
10. Integrate the copied native character cognition state at the two named
    service symbols and add focused service-wiring tests.
11. Update the relevance ICD, brain-service ICD, root README, and HOWTO.
12. Run focused deterministic tests.
13. Run each new real-LLM case separately and inspect its artifact before
    starting the next.
14. Run existing positive/negative relevance live cases separately.
15. Run the full non-live regression suite.
16. Run independent code review, remediate findings, rerun affected tests, and
    record evidence.
17. Recheck the diff allowlist and protected parallel paths.
18. Update lifecycle status and archive only after every acceptance criterion
    passes.

## Execution Model

- One implementation agent owns the allowlisted files sequentially.
- No concurrent implementation worker may edit this repository during the
  `service.py` integration stage.
- A read-only independent reviewer may inspect the final diff after
  implementation.
- Real-LLM tests run one test case at a time with output inspection between
  cases.
- The parent agent owns baseline recording, prompt-contract decisions, service
  integration, final verification, lifecycle status, and user reporting.
- The worktree remains isolated from the parallel-development branch until the
  bugfix is complete and reviewed.

## Progress Checklist

### Stage 0: Authorization And Isolation

- [ ] User explicitly authorizes implementation.
- [ ] Plan status is `approved` or `in_progress`.
- [ ] Parallel `service.py` owner has completed and released the file.
- [ ] Sole lifecycle ownership for relevance/service files is recorded.
- [ ] Isolated baseline commit, clean status, and target hashes are recorded.

### Stage 1: Evidence Contract

- [ ] Incident fixture is frozen and redacted.
- [ ] Shared evidence module exists.
- [ ] Complete-name provenance excludes `"一"` from `"一换一"`.
- [ ] Stable participant handles preserve speaker/target/reply identity.
- [ ] Active-state projection satisfies exact thresholds and caps.
- [ ] Internal assessment consistency and fail-closed tests pass.

### Stage 2: Relevance Agents

- [ ] Frontline uses the exact two-basis rule.
- [ ] Settled relevance uses the exact two-basis rule.
- [ ] Recipient is preserved independently from admission basis.
- [ ] Dropped or invented refs cannot produce admission.
- [ ] Authoritative typed participation behavior remains valid.
- [ ] Public decisions remain shape-compatible.

### Stage 3: Service And Documentation

- [ ] Frontline receives the process-local native character state.
- [ ] Settled relevance receives the same snapshot.
- [ ] Service performs no relevance-specific DB read.
- [ ] Relevance ICD is updated.
- [ ] Brain-service ICD is updated.
- [ ] Root README and HOWTO are updated.

### Stage 4: Verification

- [ ] Focused deterministic suite passes.
- [ ] Incident frontline real-LLM case passes.
- [ ] Incident settled real-LLM case passes.
- [ ] Canonical-name positive cases pass.
- [ ] State-salience positive cases pass with recipient preserved.
- [ ] Unmatched-state negative cases pass.
- [ ] Existing relevance positives and negatives pass.
- [ ] Full non-live regression passes.
- [ ] Call count, route, completion caps, and input caps are unchanged.
- [ ] Protected parallel paths are absent from this plan's diff.

### Stage 5: Review And Closure

- [ ] Independent code review is recorded.
- [ ] All findings are resolved or explicitly accepted by the user.
- [ ] Affected tests are rerun after remediation.
- [ ] Final diff allowlist is clean.
- [ ] Execution evidence is complete.
- [ ] User approves completion and lifecycle archive.

## Verification

### Static Contract Checks

```powershell
git status --short
git diff --name-only <recorded-baseline-commit> -- src tests README.md docs/HOWTO.md
rg -n "一般兴趣.*不能|有趣.*不能自动|明确使用角色规范名称称呼" src/kazusa_ai_chatbot/relevance
rg -n "recipient_relation|admission_basis|interaction_evidence_refs|character_state_refs" src/kazusa_ai_chatbot/relevance tests
rg -n "participant_[1-8]|unknown_participant|other_participant" src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py tests/test_persona_relevance_agent.py
rg -n "FRONTLINE_RELEVANCE_MAX_INPUT_CHARS|SETTLED_RELEVANCE_MAX_INPUT_CHARS|FRONTLINE_RELEVANCE_COMPLETION_TOKEN_CAP" src/kazusa_ai_chatbot/relevance src/kazusa_ai_chatbot/config.py
```

The first prompt search must find no active rule that blanket-rejects concrete
character-state salience and no active permission to infer canonical-name
address without an evidence ref.

### Focused Deterministic Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_relevance_participation_evidence.py tests/test_frontline_relevance_agent.py tests/test_persona_relevance_agent.py tests/test_relevance_turn_settlement.py tests/test_relevance_turn_settlement_graph.py tests/test_service_input_queue.py -q
```

Required deterministic cases include:

- exact incident message with no character/state evidence;
- `"一"` removed, replaced, and repeated;
- `"你"` removed and replaced;
- full canonical-name contiguous occurrence;
- canonical name quoted rather than addressed;
- typed character target, character reply, and broadcast;
- private scope;
- one valid open-turn continuation and one ambiguous multi-slot case;
- recent and stale bot continuity;
- stable participant handles across speaker/target/reply rows;
- another-recipient message with matching active-state evidence;
- same message with no matching state;
- generic helpful/interesting content with no state intersection;
- invented, duplicated, wrong-kind, and cap-dropped refs;
- state-salience proceed with `use_reply_feature=false`;
- malformed model JSON and malformed internal assessment;
- worst-case 8,000/16,000-character fitting; and
- unchanged public decision keys.

### New Real-LLM Tests

Run each command separately and inspect the readable artifact before the next:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests/test_relevance_evidence_grounding_live_llm.py::test_live_incident_frontline_discards -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_relevance_evidence_grounding_live_llm.py::test_live_incident_settled_ignores -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_relevance_evidence_grounding_live_llm.py::test_live_canonical_name_frontline_starts -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_relevance_evidence_grounding_live_llm.py::test_live_canonical_name_settled_proceeds -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_relevance_evidence_grounding_live_llm.py::test_live_state_salience_frontline_starts_for_other_recipient -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_relevance_evidence_grounding_live_llm.py::test_live_state_salience_settled_proceeds_without_reply_anchor -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_relevance_evidence_grounding_live_llm.py::test_live_unmatched_state_frontline_discards -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_relevance_evidence_grounding_live_llm.py::test_live_unmatched_state_settled_ignores -q -s
```

Each artifact under `test_artifacts/relevance_evidence_grounding/` records:

- case id and stage;
- redacted semantic input;
- rendered input character count and SHA-256;
- model route and configured limits;
- raw output;
- parsed public decision;
- validated participation assessment;
- available and cited refs;
- expected outcome and pass/fail judgment; and
- reviewer notes about recipient and reason-to-speak quality.

A patched or mocked model response cannot satisfy this gate.

### Existing Real-LLM Regression Cases

Run these existing cases separately:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests/test_relevance_cross_channel_failure_live_llm.py::test_live_natural_name_address_starts_frontline -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_relevance_cross_channel_failure_live_llm.py::test_live_natural_name_address_proceeds_settled -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_relevance_cross_channel_failure_live_llm.py::test_live_recent_bot_continuity_starts_frontline -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_relevance_cross_channel_failure_live_llm.py::test_live_recent_bot_continuity_proceeds_settled -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_relevance_cross_channel_failure_live_llm.py::test_live_unaddressed_prompt_injection_discards_frontline -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_relevance_cross_channel_failure_live_llm.py::test_live_latest_recipient_switch_ignores_settled_turn -q -s
```

### Full Non-Live Regression

```powershell
venv\Scripts\python.exe -m pytest -q
```

### Final Diff Isolation

The final diff may contain only the files listed under `Create` and `Modify`.
The parent agent must compare it with the recorded baseline and separately
confirm that no protected parallel path was staged or changed.

## Independent Code Review

After implementation and verification, a read-only reviewer must inspect:

- whether relevance remains the sole admission owner;
- whether any deterministic code semantically classifies user text;
- whether `"一"` or `"你"` can support character address without a valid ref;
- whether complete canonical-name and continuity positives remain possible;
- whether state evidence represents active state rather than generic traits;
- whether another recipient is preserved under state-salience admission;
- whether cap fitting invalidates removed refs;
- whether authoritative and malformed-output paths remain bounded;
- whether public decision shapes, call counts, routes, and caps remain stable;
- whether trace diagnostics expose raw participant or state ids; and
- whether the diff touches any protected parallel-development path.

The reviewer records findings by severity, file, and line. The parent agent
resolves every blocking/high finding, reruns affected tests, and records the
review result under `Execution Evidence`.

## Acceptance Criteria

1. The exact `直接找你一换一是吧` incident with
   `active_character_name=一之濑明日奈`, no typed participation evidence, no
   continuity, and no matching active state returns frontline `discard`.
2. The same incident, when directly exercised at settled relevance, returns
   `ignore`.
3. `"一"`, `"一换一"`, and `"你"` never create `canonical_name_span` and cannot
   validate recipient `character` by themselves.
4. The complete canonical name can still support natural direct address after
   the LLM cites `name_1`.
5. Typed character target/reply, private scope, explicit group invitation,
   open-turn continuation, and recent bot continuity remain valid interaction
   relevance paths.
6. A concrete message/state intersection can produce `start` and `proceed`
   even when the actual recipient is `participant_n` or `unknown`.
7. State-salience admission preserves that recipient and keeps
   `use_reply_feature=false` unless independent direct-character evidence
   exists.
8. Generic answerability, helpfulness, emotionality, or interest without a
   supplied active-state intersection returns `discard`/`ignore`.
9. Settled history uses stable participant handles consistently and exposes no
   raw ids.
10. Invented, wrong-kind, duplicate, and cap-dropped evidence refs cannot
    produce non-authoritative admission.
11. Existing authoritative typed-participation, wait, already-resolved,
    recipient-withdrawal, media, and operational-failure behavior passes.
12. Public relevance decisions and downstream coordinator/state actions retain
    their existing fields and values.
13. Frontline remains at one call, 8,000 input characters, 256 completion
    tokens, and thinking disabled.
14. Settled relevance remains at one call per assessment, 16,000 input
    characters, 512 completion tokens, thinking disabled, with only its
    existing bounded authoritative repair.
15. Relevance performs zero new database, retrieval, embedding, or cache calls.
16. Focused deterministic, one-at-a-time real-LLM, existing relevance
    regression, and full non-live tests pass.
17. Readable LLM artifacts show the model's recipient, basis, and cited refs
    for every new live case.
18. Documentation states the same interaction-relevance OR
    character-state-salience rule.
19. Independent code review has no unresolved blocking/high finding.
20. The final diff contains no protected parallel-development file or hunk
    outside the exact allowlist.

## Risks

| Risk | Mitigation |
|---|---|
| Weak model cites a generic message ref as character address | Ref-kind validation forbids `message_1` from validating direct character address. |
| Literal matching becomes a hidden deterministic classifier | It creates only a candidate; the LLM decides address semantics. |
| Full-name quotations are mistaken for address | Live/deterministic quoted-name cases and LLM recipient judgment remain required. |
| Character state admits ambient noise | Only active/pressured bounded items are supplied; semantic intersection and a cited state ref are mandatory. |
| Stable traits make every topic relevant | Standards and default low-pressure drives are excluded. |
| Actual other recipient is erased | Recipient and admission basis are separate fields; state admission preserves recipient and disables reply anchoring. |
| Prompt growth harms local-model reliability | Same caps/calls, six state items, exact drop order, worst-case projection tests. |
| Evidence is cited after cap removal | Catalog and action contract are rebuilt from the final serialized payload. |
| Both gates repeat the same hallucination | Both require refs available at their own final payload; unsupported claims fail closed independently. |
| Service integration overwrites parallel work | Exclusive post-parallel baseline, isolated worktree, two-symbol allowlist, stop-on-drift gate. |
| Global-state draft later rewrites relevance | Mutually exclusive lifecycle ownership is required before either overlapping scope executes. |
| Public consumers break on new model fields | Internal assessment is stripped before returning the unchanged public decision. |

## Execution Evidence

Planning evidence recorded on 2026-07-30:

- relevance ICD and brain-service live-intake contract inspected;
- frontline and settled prompts, projections, parsers, caps, and trace paths
  inspected;
- cognition-claim routing inspected;
- native character cognition-state shape and runtime snapshot ownership
  inspected;
- focused relevance tests and live regression inventory inspected;
- current parallel worktree paths recorded and treated as protected; and
- no production or test file changed while producing this draft.

Execution has not started because the plan status is `draft`. When authorized,
append:

- isolated baseline commit and target-file hashes;
- implementation commits and diff allowlist;
- focused deterministic results;
- one artifact link and judgment per real-LLM case;
- full non-live result;
- independent review findings and remediation;
- final protected-path audit; and
- user sign-off and archive commit.
