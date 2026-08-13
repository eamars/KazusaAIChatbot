# Cognition Core V2

`cognition_core_v2` owns the validated persistent cognition state used by
Stage 2. User state and the singleton character state are separate mutable
scopes. The exact state contract is enforced by `state_models.py`; structured
role references, complete evidence records, canonical singular entity kinds,
bounded axes, root ownership, and activation identity are validated before
state crosses the database boundary.

`transition_guards.py` accepts only trusted direct facts and bounded semantic
deltas. `state_reducers.py` performs elapsed evolution, cause-first event
comparison, guarded goal creation and lifecycle transitions, deterministic
event identity, retention, and activation-cache recomputation. Emotion rows are
derived projections: every activation retains typed roots, phase, trend, score,
cause status, and timestamps.

Evidence retention is deterministic and bounded. During one accepted appraisal
batch, each target retains every source cited by that batch before historical
rows fill the remaining capacity. Outside that batch, relationship state keeps
the newest eight unique rows, while causal entities keep their first/root row
and newest seven unique rows. A terminal meaning may repeat idempotently only in
the appraisal batch that produced that terminal transition; later batches
still observe strict terminal immutability. `relationship_connection` is owned
by the typed desired-versus-perceived closeness gap and is satisfied when that
gap closes.

The twenty-one emotion formulas are exercised from typed natural causes in
`tests/test_cognition_core_v2_emotion_lifecycle.py`. Cross-scope character
constraints and optional relationship context are passed as dedicated
projections; they are not merged into mutable state. Character sleep recovery
is deterministic and separate from user elapsed decay.

Character identity is resolved from the latest immutable revision once per
episode. Appraisal owners receive only their bounded identity partitions.
Goal cognition receives `core`, `personality`, `boundaries`, and `self_image`,
so reviewed changes to backstory, character judgment, boundaries, or
self-concept can alter later goals without exposing revision history or old
values. Text and visual surfaces receive separate expression-only projections.

Database-backed callers use `db.users` for user-owned state and `db.character`
for the character singleton. The test database harness requires the exact
`_test_kazusa_live_llm` name, validates seeded V2 state, and gives every
mutable test row a unique owner.

The public Stage 2 surface consists of `run_cognition(...)`,
`run_text_surface_planning(...)`, `repair_text_surface_planning(...)`, and
`run_visual_surface_planning(...)`.
Cognition runs deterministic preparation, scoped semantic appraisal,
dependency-ready goal branches, complete-bid collapse, route validation, and
one replacement-state update. The caller commits that update before action,
surface, resolver, or dialog work.

## Required-Selection Role Operation

The episode-level `response_operation` is input provenance. It records who
responds, who owns an unspecified choice, and any embedded actor/target roles
fixed by the current input. Required-selection goal cognition emits a separate
`selected_response_operation` after the character chooses the concrete action.
Both use the canonical `DialogResponseOperation` shape.

`response_owner_role` and `selection_owner_role` own the agreement, selection,
telling, request, and confirmation wrappers; they do not by themselves fix the
actor or target of the nested action. The `selected_response_operation.operation`
text and its embedded endpoints type one concrete selected nested action after
those wrappers are removed. For a character-owned request such as "I want/ask
you to do X to me", the character remains the response and selection owner while
the current user is the actor of X and the character is the target of X. When
the input contains multiple clauses, goal cognition types the decisive selected
embedded action that candidate wording must preserve; compatible wrapper,
condition, or secondary clauses do not replace its endpoints.

The selected operation is carried through the admitted bid, selected
intention, and `TextSurfaceInputV2`. Deterministic validators preserve every
known non-`无` response-owner, selection-owner, actor, and target role. The
surface prompt projection cannot rewrite this control carrier; dialog role
verification consumes it as the authority for required-selection turns.
Missing or conflicting selected operations fail at the cognition contract
boundary within the existing bounded regeneration policy. Numeric dialog
scores cannot override a typed role-direction hard gate.

## Stage Model Routing

Core V2 receives one independent `LLMCallConfig` for each existing semantic
model owner:

| Semantic owner | Service field | Environment route |
|---|---|---|
| Event and agency appraisal | `appraisal_event_agency_config` | `COGNITION_LLM_APPRAISAL_EVENT_AGENCY` |
| Relationship and social appraisal | `appraisal_relationship_social_config` | `COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL` |
| Moral and identity appraisal | `appraisal_moral_identity_config` | `COGNITION_LLM_APPRAISAL_MORAL_IDENTITY` |
| Goal, threat, and outcome appraisal | `appraisal_goal_threat_outcome_config` | `COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME` |
| Epistemic, comparison, and memory appraisal | `appraisal_epistemic_comparison_memory_config` | `COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY` |
| Existential and drive appraisal | `appraisal_existential_drive_config` | `COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE` |
| Ordinary-response and required-selection goal | `goal_ordinary_response_config` | `COGNITION_LLM_GOAL_ORDINARY_RESPONSE` |
| Active persistent-goal branches without typed required selection | `goal_active_branch_config` | `COGNITION_LLM_GOAL_ACTIVE_BRANCH` |
| Workspace collapse | `workspace_collapse_config` | `COGNITION_LLM_WORKSPACE_COLLAPSE` |
| Action planning and goal resolution | `action_planning_config` | `COGNITION_LLM_ACTION_PLANNING` |
| Action authorization | `action_authorization_config` | `COGNITION_LLM_ACTION_AUTHORIZATION` |
| Resolver authorization | `resolver_authorization_config` | `COGNITION_LLM_RESOLVER_AUTHORIZATION` |

Every initial call, provider retry, structural replacement, and trace row uses
the config selected by that semantic owner. A typed required-selection turn
uses a specialized producer on the dense ordinary-goal route regardless of its
branch; it replaces that branch's generic goal call and adds no evaluator
route. Active branches without a typed required selection retain the
active-goal route. Stage routes are complete required environment bundles and
have no route inheritance or fallback. The generic `COGNITION_LLM` route
continues to serve cognition callers outside this Core V2 boundary.

The first wave contains six appraisal families, the
ordinary-response goal, and dependency-ready active-goal branches can submit
up to twenty model tasks concurrently. Final dependency-ready goal work,
workspace collapse, action planning, and the applicable authorization stage
remain ordered after that wave. The six appraisal routes use a code-owned
2,048-token completion default. Goal and action-planning routes use the
8,192-token semantic default; workspace and authorization routes use the
1,024-token structured default. Surface content, preference, and visual
routes use 8,192, 4,096, and 2,048 tokens respectively. Every cognition-owned
call has the bounded `COGNITION_STAGE_TIMEOUT_SECONDS` timeout, 120 seconds
by default and configurable only within 10 to 600 seconds. Each appraisal
family runs at most eight serial micro-appraisal items on its existing route.
Each item keeps one normal call and one bounded replacement attempt.

Goal, threat, and outcome appraisal uses affirmative entity-specific terminal
assertions: `goal_completed`, `event_completed`, `threat_resolved`,
`event_repaired`, and `knowledge_answered`, alongside goal release and
supersession. `outcome_pending` records an explicit nonterminal observation
without state mutation or candidate materialization. The payload maps subject,
object, role-assignment entity, and evidence fields to their exact handle
domains. Persistent events use `ev1..evN`, evidence uses `e1..eN`, and current
candidate events, threats, and knowledge gaps use `ceN`, `ctN`, and `ckN`.
The retained question maps every permitted candidate to its exact origin
evidence; any structured use of that candidate must cite that evidence. The
model-facing item uses singular nullable `proposition` and `delta` fields.
Deterministic code accumulates at most eight accepted items, derives selected
handles and explanation metadata from actual structured content, and treats an
empty or exact-repeat item as bounded termination. It also maps the structured
`self` and `current_user` handles to their
Chinese semantic-text references so role fields and prose use the same actor
roots without sharing representations. Structural validation binds each
terminal kind to its exact entity kind. A valid assertion atomically
establishes the terminal axis before the unchanged transition guard runs,
including when the assertion first materializes a causal candidate. Terminal
candidates bypass the nonterminal salience-pruning path. After all same-batch
numeric observations are reduced, the accepted terminal assertions reassert
their canonical axes without repeating or weakening the guarded transition.
Role-signature event matching reuses only active events; exact canonical event
IDs remain authoritative across statuses. Every candidate is trial-reduced
inside its micro item's attempt cap.
Final reduction validates each added appraisal by replaying the bounded
accepted prefix from the original state, so handle composition is preserved
and one residual rejection is omitted without discarding other appraisal
results or the preliminary character response.

Current-event scene text, the bounded public group scene, participant
conversation continuity, and private residue continuity are separate inputs.
In group chat, public scene order and visible participants are the authority
for public facts; participant continuity remains scoped to the current user.
Private continuity reaches goal-cognition
branches only and remains non-binding prior context: each branch decides
whether the current event, affect, relationship, and active goal call for
progressing, revising, or leaving that posture. Branch bids distinguish
analytic `reason` from first-person `private_monologue`; public output exposes
that distinction as `selected_bid_reason` and `private_monologue`.

Trace-backed past-dialog continuity uses its own optional
`past_dialog_cognition_context` carrier, capped at 1,800 characters. It reaches
goal cognition only, separately from the 1,000-character internal-monologue
`private_continuity_context`. It is weak private context for understanding an
already-linked prior character dialog, not evidence, a command, selected
stance, action-planning input, surface content, or dialog wording.

Targetless group self-cognition may receive one exact
`group_engagement_action_context` containing bounded
`engagement_guidelines` and a semantic `confidence` descriptor. The same
immutable advisory projection reaches goal cognition and action planning only.
It helps judge participation in the currently observed group scene; it cannot
create a topic, fact, relationship belief, permission, route, or unsupported
reason to speak. Appraisal and workspace collapse receive no copy.

Targetless group self-cognition has one additional model-facing contract:
self_cognition_response. Cognition chooses exactly stay_silent or
propose_visible_reply, cites up to four supplied evidence handles, and may
select only self, current_group_scene, or a supplied pN participant handle. A
proposal also supplies a closed participation basis, a bounded semantic
response goal, and a bounded reason. The object contains no route, permission,
platform identity, adapter instruction, dispatch instruction, or final
wording. Deterministic code validates this contract and derives the route; the
canonical speak capability remains excluded from the generic action-planning
roster and is materialized only through the existing L3 surface path.

For user dialog, the canonical percept may carry bounded
`role_explicit_content` and structured `response_operation` values authored by
the existing upstream decontextualizer LLM. The operation identifies the
response owner, any required selection owner, and embedded actor and target.
The raw sentence and deterministic speaker/addressee frame remain intact. V2
consumes this semantic projection unchanged as current episode meaning. Goal
cognition owns the concrete required-selection choice and emits
`selected_response_operation`; downstream stages carry that selected authority
instead of independently reinterpreting nested roles or reusing the input
operation for the embedded action.

In a group episode, visible third-party participants use an episode-local
typed binding such as `p1`, paired with its display name and the
`third_party` entity kind. The binding is transient: it is available to
decontextualization, cognition goal selection, L3 surface planning, and dialog
verification, but it is not a user identifier, a persistent memory handle, or
a delivery recipient. A typed `pN` target is rendered by name or an explicit
third-person reference; only a `current_user` addressee row with
`second_person_allowed` authorizes `你`. The structured addressee plan is
propagated as surface authority and validated again at the dialog boundary.
The dialog path therefore preserves target semantics without post-generation
name substitution or deterministic text rewriting.

Goal-bid output uses an exact route-to-capability-field matrix. A malformed bid
receives up to three total LLM attempts while deterministic validation remains
strict. The three-call limit belongs to each goal-producing stage and branch
for the whole cognition invocation. The service's one clean graph retry may
consume only unused calls from the same ledger and cannot reset it. Exhausted
goal branches are non-retryable. A failed required branch may continue only
when its phase already contains a complete validated sibling bid; the failure
and `required_branch_recovered_by_valid_bid:<branch_id>` warning remain visible,
and only complete bids reach workspace collapse. With no complete sibling,
cognition raises before collapse. Every goal attempt is eligible for the
protected turn trace.

When upstream episode evidence carries a typed required selection, deterministic
routing selects one specialized goal producer instead of the generic goal
producer. Deterministic code partitions its input into authoritative required
operations, complete model-visible conversation-progress evidence, and
optional supporting evidence. The producing call emits one authoritative
`selection`, reason, role/evidence handles, consequences, and confidence. It
must cite every required operation. The goal LLM cites only
progress rows that materially constrain the current choice and leaves
unrelated history uncited. Completed, rejected, and superseded progress remains
model-visible and may be reopened only when the current input explicitly
requests it. RAG conversation-history rows remain optionally citeable
supporting evidence.
Deterministic validation owns exact fields, bounds, provenance partitioning,
required-operation handle coverage, and mapping the one selection into the
existing complete bid. The goal LLM owns progress relevance and the actual
choice, with no discarded relation matrix. Structural failures retry the same
producing prompt under the existing bounded goal contract, with the validation
error plus exact required and allowed evidence handle sets attached to each
complete regeneration. Required-selection parsing
uses deterministic cleanup only, so malformed output cannot invoke the shared
JSON-repair model. There is no semantic verifier, negative verdict,
evaluator-authored replacement, or recheck. Ordinary turns keep the ordinary
goal producer.

Goal cognition owns the character's semantic objective, including the complete
character-owned relationship stance and grounded acceptance, refusal,
negotiation, or conditional participation. It is capability-neutral: it does
not receive runtime capability limits, infer tool availability from character
identity, or promise unverified future effects. Missing current facts remain a
goal to answer after obtaining evidence. Workspace collapse receives the typed
current episode plus bounded persistent-goal provenance for each nonordinary
bid. It admits a persistent branch only when the current event concerns the
same concrete matter; unrelated active goals are suppressed while the ordinary
response remains the baseline. Runtime feasibility and resolver selection stay
owned by action planning.

## Current-Turn Relational Willingness

The ordinary-response goal owner produces one exact transient
`relational_willingness.v2` decision per turn. The decision carries a
relationship-sensitive applicability with a native
`current_user_relationship_state` (`unestablished`,
`developing_or_uncertain`, or `established`) and one ordered stance (`reject`,
`deflect`, `negotiate`, `conditional_accept`, or `accept`), or marks a request
that is not relationship-sensitive with
`not_relationship_sensitive/not_applicable/not_applicable`. The ordinary
response prompt, including its typed required-selection form, requires the
field; typed selection on an active branch retains its existing output
contract and does not re-decide relational willingness. Relationship state is
descriptive context rather than a permission matrix: every sensitive stance is
valid for each real relationship state. Deterministic validation enforces the
structural non-sensitive pairing and complete sensitive fields:

| Applicability | Current-user relationship state | Allowed stance |
| --- | --- | --- |
| `not_relationship_sensitive` | `not_applicable` | `not_applicable` |
| `relationship_sensitive` | any real relationship state | `reject`, `deflect`, `negotiate`, `conditional_accept`, or `accept` |

The sensitive decision cannot use `not_applicable` for either stance or
relationship state. A missing, unknown-enum, or internally
inconsistent ordinary decision is a structural contract error that regenerates
through the same goal owner and, after bounded attempts, fails closed before
state commit. Deterministic code never derives, upgrades, or rewrites the
stance or relationship state from prose, relationship numbers, or memory text.

The decision must cite at least one current-episode evidence handle.
Each evidence row receives one transient `provenance_role` derived only from
trusted source-kind and memory-scope metadata. `current_episode` rows are the
current request and scene; `current_user_history_only` rows explain current-user
history; `character_or_world_context_only` rows (shared character/world memory
and promoted reflection) inform character compatibility and knowledge. The
character weighs these descriptive sources with the current episode and other
evidence; `contextual_fact_only` rows are general context.
Unknown provenance fails closed at the deterministic boundary. No raw user id
or relationship id reaches the model.

Relationship axes and boundary profiles reach the model as domain-specific
semantic descriptions. Zero trust and zero boundary-safety retain their own
axis meanings, and the compliance strategy is projected only as a bounded
pressure-response style. Persisted standards remain in raw state while the
live model-facing standards projection is empty; no standard handles are
emitted.
Relationship appraisal receives one canonical `relationship` payload with the
same axis semantics; no duplicate relationship alias is emitted. The goal LLM
classifies `current_user_relationship_state` from that qualitative projection;
deterministic code validates only the declared object's internal consistency,
handle bounds, and current-episode coverage.

Generic goal branches use branch-specific model contracts. `ordinary_response`
retains the nine generic fields plus transient `relational_willingness`, while
active generic branches such as `self_improvement` receive only the nine
generic fields. Typed required-selection branches keep their existing
ordinary or active selection contract. A generic goal repair reuses the same
static branch contract as its initial call. Exact-field failures report the
observed, missing, and unexpected top-level key sets; they retain the full
candidate in protected diagnostics but do not echo it or the `invalid_draft`
token in model-facing exact-field feedback.

### Branch-owned generic intent guidance

Each default branch definition owns one bounded
`branch_intent_guidance` string. The value is a semantic attention focus, not a
positive or negative motive label. The current event, identity, role direction,
boundaries, and supplied evidence remain authoritative. The default values are
limited to 240 characters; a custom definition may retain the empty neutral
default and omit the descriptor from its generic prompt.

All fourteen registry rows have a fixed value. Exactly thirteen nonordinary
generic initial and repair payloads project the value under the
`branch_intent_guidance` key. `ordinary_response` keeps its existing ordinary
prompt and uses its row only as registry/documentation context. Typed
required-selection paths also omit the value. Branch identity is always the
registry `branch_id`; local workspace `bN` handles are not branch identities.

| Branch | Semantic responsibility | Literal runtime guidance |
|---|---|---|
| `ordinary_response` | Neutral current-event baseline with existing relational-willingness ownership. | 为当前事件提供中性的上下文基线；在适用时保留现有 relational_willingness 的归属，不引入其他分支的专门焦点。 |
| `relationship_connection` | Voluntary, context-appropriate reciprocal interpersonal connection. | 评估是否以及如何通过自愿且符合当前情境的互惠参与来建立、维持、调整或修复人际连接。 |
| `bond_protection` | Protect an important bond from evidenced threat or damage. | 评估当前事件是否对重要关系纽带造成有证据支持的威胁或损害，并考虑相称的保护或修复。 |
| `trust_verification` | Check whether trust is warranted under current uncertainty. | 评估当前证据是否支持信任、保留信任或需要核实；不把不确定性直接解释为背叛。 |
| `autonomy_boundary` | Protect current-character-owned autonomy and boundaries when grounded. | 评估当前事件是否对角色自身的自主权、意愿或明确边界造成有证据支持的压力或代价；在有依据时保护边界，不假定恶意。 |
| `safety_coping` | Manage evidenced threat or strain proportionately. | 评估当前事件是否存在有证据支持的威胁或压力，并考虑相称的保护或应对；不凭空升级恐惧。 |
| `obstruction_strategy` | Resolve an obstacle blocking current-goal progress. | 评估当前事件是否阻碍当前目标的进展，并考虑相称的解决、对抗或修复。 |
| `loss_recovery` | Process evidenced loss through recovery, adaptation, or grief. | 评估当前事件是否构成有依据的损失，并考虑恢复、适应或适当的哀悼；不强迫悲伤。 |
| `moral_repair` | Assess evidence-supported responsibility for harm and pursue proportionate repair. | 评估当前角色是否对伤害负有有证据支持的责任；如有，考虑相称的修复或道歉。 |
| `social_care` | Respond to grounded needs of affected people through care or support. | 评估受当前事件影响的人是否有有依据的需要，并考虑相称的支持或照护；不强迫温柔。 |
| `reciprocal_response` | Determine a proportionate response to another actor; reciprocity is not compliance or matched valence. | 确定当前角色对另一方行为的有证据支持且相称的回应；互惠不等于服从，也不要求匹配情绪价性。 |
| `epistemic_exploration` | Reduce uncertainty through exploration, questions, or comparison. | 通过探索、提问或比较，减少当前有依据的不确定性并增进理解；区分求知与无依据的断言。 |
| `meaning_reconstruction` | Rebuild coherent meaning after an evidenced narrative or existential disruption. | 在当前事件造成有依据的叙事或存在性中断后，评估如何重建连贯意义；不强迫乐观。 |
| `self_improvement` | Find an evidence-grounded opportunity to learn, correct, or develop capability without presuming deficiency or success. | 评估当前角色是否有有证据支持的学习、纠错或能力发展机会；不预设缺陷、乐观或成功。 |

When evidence does not support a nonordinary branch's specialized focus, its
goal producer still emits the existing complete bid contract with no supported
basis for specialized progress. It cites only relevant supplied evidence and
does not borrow the ordinary motive. Existing workspace collapse owns whether
that bid is suppressed; deterministic code does not rewrite semantic bid
fields. Active-goal descriptions may still contain evidence-grounded valence as
contextual state; that is distinct from static branch polarity.

When the ordinary owner declares a turn `relationship_sensitive`, the workspace
stage uses the deterministic authoritative collapse: the ordinary bid becomes
primary, no supporting bid is exposed, every other bid is recorded as
competing, and the preservation reason is recorded in diagnostics. No workspace
model call runs on that path. Non-sensitive turns keep the existing
model-authored collapse. Action planning receives the exact decision and
deterministically denies action and resolver effects for `reject`, `deflect`,
`negotiate`, and `conditional_accept`; only `accept` (and non-sensitive turns)
enters the effect-authorization path. The same decision is copied into
`TextSurfaceInputV2` so content, preference, and repair stages preserve the
stance and relationship state without re-deciding them. Workspace, action
planning, surface planning, and dialog never reinterpret the decision.

Action planning treats local-model output as a bounded proposal rather than an
execution precondition. It canonicalizes the known envelope, keeps usable
rows, drops invalid rows individually, ignores unknown fields, and caps each
request list at three. Mutually exclusive action and resolver requests remain
a semantic contract error. If three total planning attempts remain unusable,
the turn continues with an empty action plan; if three total authorization
attempts remain unusable, every candidate is denied. Neither containment path
authorizes work, changes the visible speech route, or reduces the
registry-driven three-request capacity.

The action-planning envelope also carries the required Cognition-Core-owned
`goal_resolution`: `answerable_now`, `requires_required_evidence`,
`requires_user_input`, or `blocked`. This is the semantic judgment of whether
the accepted user goal can be answered now; it is distinct from any
source-specific RAG `resolved` field. `answerable_now` suppresses optional
resolver requests before resolver authorization and recurrence. Required
evidence, user-input, and technical-blocked decisions retain their existing
typed paths. Deterministic code validates and enforces the decision without
reclassifying it from keywords or adding another LLM stage.

For generic evidence work, the planner has one resolver-facing semantic choice:
`task_resolution_request`. The planner decides only whether current evidence is
sufficient or that this capability is needed. It does not choose a specialist,
an execution horizon, a queue worker, a timeout, a checkpoint, or tool
parameters. The resolver owns inline execution and deterministic promotion;
task resolution owns next-specialist selection inside its fixed session limits.
The task-resolution row also requires the model-owned JSON boolean
`start_in_background`. `true` enters durable checkpoint creation and accepted
task promotion directly; `false` preserves the bounded inline-first path and
uses the same durable continuation only when that inline run defers. The
boolean is preserved through authorization and recurrence as a route decision;
deterministic code owns the queue, worker, checkpoint, and idempotency details.

Action planning preserves the current user's requested effect in every resolver
`semantic_goal`: the target, scope, and explicit time constraints survive
intact, and missing evidence may be stated only as a dependency of that effect.
Capability, permission, feasibility, and API support are runtime constraints
owned by deterministic runtime stages; they become semantic audit objectives
only when the current user explicitly asks whether the operation is possible or
authorized. Evidence rows carry a transient `provenance_role` projected by the
deterministic provenance helper; `current_episode` rows are the authoritative
current request and scene, while history, character/world, and contextual rows
remain supporting context. No raw source id or storage metadata reaches the
planner. Downstream task resolution consumes the emitted objective unchanged
and does not repair action-planning semantic substitution. An empty resolver
progress shell remains `null`; an invented checklist is a structural contract
error handled by the existing bounded same-stage regeneration path.

The shared surface input receives semantic intention, bounded affect and
    relationship projections, complete-bid projections, permitted action results,
    interaction style, an exact tempo/linguistic-texture expression context, a
    bounded recent-character-dialog projection, and a separate bounded
    visual-character context. Normal text planning makes exactly two parallel
    calls. Unified content planning atomically returns the content plan,
    requirements, a five-field delivery profile, and optional
    `lexical_avoidances`; preference planning
returns only real visible boundaries and addressee constraints. Neither call
receives the visual-character context, and preference receives no character
expression context. The delivery profile is limited to lexical register,
sentence shape, rhythm, hesitation, and punctuation and cannot authorize a
semantic stance. The exact text output contains no raw character profile or
visual directives. The independent one-call visual planner may observe the
isolated visual-character context and produces image-generation directives as
terminal private evidence; it has no downstream image or dialog model. Raw
episode traces retain those directives for audit, while every model-facing
consolidation projection excludes their fragments.

### Surface score and confidence boundary

Only `surface_content_plan` and
`surface_dialog_compliance_repair` are research candidates for an owner-local
score-selection contract. The current execution has no accepted calibration
corpus or thresholds, so production keeps the existing first-valid and
degraded-surface behavior; the evaluator and selection path is not active.
Future activation requires the plan's independent held-out evidence, finite
`score` contract, bounded blocking issues, three-call producer cap, evaluator
cap, threshold return, highest-score exhaustion, and deterministic tie order.

The V2 `confidence` field remains a bounded semantic descriptor and advisory
context. It is not a score, ranking input, threshold input, authorization
signal, or delivery gate. Goal, action, group-engagement, and branch
observation contracts reject numeric or boolean confidence values. Workspace
candidate-quality comparison does not receive confidence. Preference, visual,
goal, action, authorization, persistence, queue, delivery, and stateful retry
owners remain outside this score contract, and the dialog renderer keeps its
existing numeric evaluator behavior.

    `lexical_avoidances` contains only concrete current-turn expression fragments
    such as a repeated recent opening, stale filler, stale address, or wording
    that obscures the selected intent. It is a surface-owned continuity hint,
    never a topic, moral, or refusal policy. The dialog renderer preserves the
    selected semantics while avoiding those literal fragments, and a deterministic
    literal check routes a hit through the existing bounded repair path.

    Content planning expresses the selected character judgment using the current
scene, affect, relationship, and interaction style. Coherent imaginative
detail is allowed when it remains compatible with current input, active
constraints, and actor/target/subject roles. Preference planning emits only
real visible boundaries and addressee constraints, so both lists may be empty.
Dialog owns natural character-specific chat-ready wording. Three focused
hard-error checks run in parallel on the existing dialog-model route within
the bounded verifier path, followed by the deterministic expression-continuity
check. Current visible percepts retain their shared
32,000-character cap. Semantic fidelity separately caps authoritative surface
semantics at 11,000 characters, candidate dialog at 12,000 characters, and its
complete serialized payload at 50,000 characters. Semantic fidelity receives
current model-visible percept rows, the candidate role frame, candidate
dialog, and the authoritative selected surface intent, content plan,
requirements, visible boundaries, and lexical avoidances. It rejects internal contradiction,
direct current-input conflict, non-selection role reversal, and unsupported
within-turn opposite-stance transitions. Delivery profile and action-result
fields are excluded from semantic authority. Role direction receives only typed
selection-required role tuples and rejects selection-owner transfer or
actor/target reversal. Selection-required role fields are excluded from the
semantic-fidelity projection, which retains the raw current-input meaning and
cannot rejudge role-owned operation completion. Surface integrity receives
permitted action results and candidate dialog; it rejects only false system,
tool, platform, or other character-brain execution claims.
Text planning owns expressed meaning and interaction progress without
supplying staging forms. Dialog expresses emotion, character, and interaction
posture through sendable wording and cadence. Action narration is outside the
fatal taxonomy and remains unchanged when the model produces it; the runtime
prompts neither request it nor create a rejection or repair rule for it.
Source percepts and generated character speech carry separate typed pronoun
frames before role direction is compared. Novelty, coherent drift,
character-owned refusal, negotiation, and supported changes of mind are not
failures by themselves. Deterministic code validates the exact numeric score
shapes and merges available scores with an equal-weight geometric mean. The
dialog threshold is currently `0.50`; a below-threshold candidate without hard
issues remains eligible for degraded ranking after bounded exhaustion. Explicit
semantic hard errors, role violations, false-execution issues, lexical
violations, empty candidates, and state or delivery failures remain fail-closed.
Each owner is bounded to four issues and the merged result to eight. A negative
result returns canonical surface input plus bounded verified issues to the
text-surface owner for one complete replacement of `content_plan`,
`content_requirements`, `delivery_profile`, `lexical_avoidances`,
`visible_boundaries`, and `addressee_plan`. Rejected surface fields and rejected dialog are trace-only
and are absent from both repair-model payloads. Selected intent, action truth,
the exact relational stance, and runtime capability limits are reconstructed
from canonical input before each dialog retry. Every candidate, including the
terminal candidate, passes the semantic, role, surface, and expression-continuity
checks. After bounded exhaustion the highest-scoring eligible candidate is
delivered as degraded output; when no eligible candidate remains the delivery
boundary raises a typed failure. No hard-invalid or unverified dialog reaches
post-turn consumers. The protected turn trace
records rejected checks, surfaces, and dialog candidates as diagnostic
evidence.

Each focused verifier validates its own exact JSON verdict. A structurally
invalid parsed verdict receives up to two complete replacements using the
unchanged system and semantic payload plus the latest bounded rejected
assistant candidate and exact contract error. The replacement remains inside
that verifier and does not create another dialog candidate. All attempts are
recorded in the protected trace. Semantic fidelity uses the
collision-resistant producer field `hard_errors`; deterministic validation
rejects boolean aliases, non-finite values, and out-of-range scores before
bounded regeneration. Role direction uses typed `violations` limited to
`selection_owner_transfer` and `typed_operation_role_reversal`; surface
integrity retains evidence-bearing `issues`. Exhaustion marks only that
verifier `unavailable`; available numeric dimensions continue to rank
structurally valid candidates, and an all-focused outage receives aggregate
score `0.0` for deterministic degraded tie-breaking.

## Document Control

Stage 2 native cognition contract. Source of truth: the V2 contracts,
state models, reducers, and focused test suites in this package.

## Purpose

Provide one bounded cognition implementation for persistent user and
character state, semantic appraisal, goal evolution, emotion lifecycle
derivation, and surface planning.

## Boundary

Callers provide typed episode evidence and validated state. This package owns
semantic cognition and replacement-state production; persistence, action
execution, dialog wording, and adapter delivery remain downstream owners.

## Public Entrypoints

- `run_cognition(...)`
- `run_text_surface_planning(...)`
- `repair_text_surface_planning(...)`
- `run_visual_surface_planning(...)`
- `run_character_morning_refresh(...)`
- `validate_cognition_input(...)`
- `validate_cognition_core_output(...)`

## Runtime Flow

Input validation, bounded semantic appraisal, dependency-ready goal branches,
complete-bid collapse, route validation, replacement-state reduction, and
typed output validation run in one inspectable call.

## Short-Horizon Operational Context

The character-scope `CharacterCognitionStateV2` remains the sole persisted
short-horizon global authority. Callers derive an elapsed-effective full view,
then pass only a bounded `character_operational_context` to the approved
consumer branch. Current-user relationship context remains user scoped and is
projected separately; its durable relationship id never crosses the public
operational boundary.

The service reuses one immutable interaction-style snapshot for settled
relevance, V2 cognition, and L3 surface. The V2 input/output and L3 input are
the source of the graph-owned `cognition_context_consumption.v1` record. That
record is observability only: it uses bounded public selections, digests, and
typed health; it does not alter cognition semantics or introduce a second
state authority.

Operational context packets use compact canonical JSON accounting
(`ensure_ascii=False`, sorted keys, and compact separators). Relationship
operational context is capped at 900 decoded characters and character
operational context at 1,200 decoded characters, including its final
`context_digest`. Producers fit packets before publication; the Cognition V2
input validator applies the same owner-specific fit to a copied packet before
strict validation and returns the fitted packet to downstream stages. The fit
may middle-truncate bounded summaries or remove optional rows, but preserves
identity, axes, handles, timestamps, provenance, and required current-turn
facts. Malformed structure remains a contract error; only irreducible required
overflow reaches the typed context-limit invariant.

Relationship fitting first retains the longest possible causal summaries with
an 80-character floor, then drops the lowest-priority causal rows from the
end of their stable salience/recency order, and then drops affect rows.
Character fitting drops pressures from the end of their stable order before
affect rows. The character digest is recomputed when fitting changes the body
and remains stable for a valid no-op consumer fit.

## Context Fade And Sleep Phase

Aged conversational context is discarded deterministically before projection;
it is never presented to a model together with an instruction to discount it.
Group-scene ambient turns older than `GROUP_SCENE_MAX_TURN_AGE_MINUTES`
(default 120 minutes) relative to the trigger are dropped inside
`filter_group_scene_ambient_turns`, which is shared by
`build_group_scene_context` and the persona Stage 0 decontextualizer; the
filtered sequence also supplies group scope participants. The trigger is never
filtered and `omitted_turn_count` counts only count-based truncation.
Conversation-progress events older than their retention-tier threshold are
dropped on the read path
by `conversation_progress.policy.prune_aged_progress_packet` immediately after
packet selection, using `CONVERSATION_PROGRESS_BACKGROUND_MAX_AGE_MINUTES`
(120), `CONVERSATION_PROGRESS_ACTIVE_SCENE_MAX_AGE_MINUTES` (360), and
`CONVERSATION_PROGRESS_DECISION_CRITICAL_MAX_AGE_MINUTES` (2880). When no
event survives, or the newest surviving event is older than
`CONVERSATION_PROGRESS_NARRATIVE_MAX_AGE_MINUTES` (360), the complete
narrative field set is cleared to its canonical empty shape. Pruning issues no
database write; the next recorded turn persists the pruned form.

Progress evidence rows carry each originating event's own `updated_at` as
`evidence_ref.occurred_at`, normalized to the V2 UTC-Z second-truncated
format, and `scene_context.semantic_temporal_context` is derived from the
newest surviving event age using the `project_duration` vocabulary rather than
a hardcoded literal.

`scene_context.character_sleep_phase` is an optional validated field produced
by `project_character_sleep_phase(now, *, sleep_local_period,
character_time_zone, wake_prep_minutes)`. The vocabulary is frozen and
deterministic: `清醒时段` outside the window, `睡眠中` inside the window, and
`即将醒来` inside the final `wake_prep_minutes` before the exclusive window
end. The two in-window labels cover exactly the half-open local window that
`is_self_cognition_sleep_period` reports, including midnight wrap; an empty
period is outside. The field reaches goal cognition only and never appraisal,
surface, `CharacterOperationalContextV1`, or `TextSurfaceInputV2`.

## Morning Refresh

`run_character_morning_refresh(state, *, elapsed_sleep_seconds, updated_at)`
is the public deterministic character morning-refresh transition. It owns the
character-scope guard, the `apply_sleep_recovery` reducer call, and
`validate_cognition_state` on its output, and returns
`CharacterMorningRefreshResultV2` with the recovered state, the applied
elapsed seconds, and bounded deterministic transition counts. It knows nothing
about local dates, run identifiers, or persistence.
`reflection_cycle.affect_settling` calls this entrypoint and keeps scheduling,
idempotency, the guarded write, the refresh callback, and the audit row.

## Failure Behavior

Every recoverable V2 producer and verifier has at least three total local
attempts except semantic appraisal. Each appraisal micro item has one initial
call plus at most one complete-replacement attempt, and each family has at most
eight items. Existing longer semantic ledgers keep their cap. The outcome
ladder is `accepted`, `recovered`, `accepted_degraded`, then `unrecoverable`.
Appraisal and optional visual exhaustion are omitted; decontextualization keeps
the normalized original input; workspace keeps the highest-priority complete
bid; action planning returns no work; authorization denies; and text-surface
exhaustion projects a validated neutral surface from canonical V2 truth.

Malformed canonical input, invalid persistent state, unsupported routes,
unresolved required dependencies after the bounded branch policy, failed
commit or post-commit invariants, and total model unavailability with no owned
fallback remain execution errors. Unknown fields, invalid semantic values,
unsupported or duplicate role/evidence handles, missing required citations,
invalid consequences, and invalid relational pairings stay producer-owned
contract errors. Goal cognition regenerates a complete candidate within its
cumulative budget and never deletes model-authored handles or values into
acceptance. Recoverable and degraded outcomes owned by other stages follow the
normal persistence and delivery path. Callers commit only validated replacement
state.

All four public entrypoints capture their raw arguments before validation in a
ContextVar-isolated protected failure buffer. `repair_text_surface_planning`
captures both the raw surface input and `verified_hard_issues`; the other
entrypoints capture their raw input payload. Clean runs discard the buffer.
Terminal exceptions, failed appraisals or branches, recovered model attempts,
and degraded surfaces schedule one failure capsule without delaying or
changing the returned output or raised exception.

The direct V2 model owners—semantic appraisal, goal cognition, workspace
collapse, action planning, semantic authorization, and generic surface
stages—record every provider and contract attempt with its one-based attempt
index, non-secret call configuration, exact messages and response, parsed
output, and concrete error. The canonical JSON-repair fallback records its own
model call into the active capsule without changing parser or retry behavior.
Goal attempts additionally record the cognition invocation, graph attempt,
branch, producing stage, local attempt, cumulative producer attempt, configured
limit, attempt disposition, and final branch disposition. The invocation-wide
ledger is protected diagnostic data and does not enter event logs, public
responses, or operational status.

When the live persona connector is bound to the parent-checkpoint guardrail,
the model ledger keeps the existing three-call owner cap independently in epoch
zero and the one permitted parent-recovery epoch. Epoch one remains active for
later resolver cycles and cannot create another epoch. The unguarded
`cognition_attempt_ledger.v1` snapshot remains unchanged and is retained by
each inner `run_cognition` capsule. The outer guardrail writer may additionally
store bounded `cognition_attempt_ledger.v2` epoch metadata; it contains only
owner coordinates, dispositions, and the parent-recovery summary.

## Aggregate Prompt Budgets

Each V2 model owner budgets its complete deterministic serialization, including
its static system prompt, rather than relying on independent producer-field
limits. Semantic appraisal owns a 20,000-character aggregate packet containing
its question contract, one top-level semantic-evidence registry, and its
authorized state projection. Exact
`permitted_delta_paths` remain private validator authority; the model receives
grouped `state_field`, `handles`, and `axes` domains and returns the same
canonical `state_field.handle.axis` target path.

Appraisal reduces identity, constraints, then state rows before evidence text;
goal cognition reduces supplemental context, scene, constraints, identity, and
then evidence. Identity and scene reductions use fixed semantic floors and
middle truncation while preserving core identity, boundaries, evidence rows,
handles, source kind, and source order. Both owners preserve at least 96
characters per reduced evidence text, or the complete original when it is
shorter. Goal cognition owns the equivalent single-registry packet under its
fixed 36,000-character aggregate cap, with the current episode retaining
highest source priority. Its goal projection and canonical role summaries are
each serialized once; duplicate evidence, goal projection, role summaries,
and scene role labels are absent from supplemental semantic context.
Past-dialog and group-engagement contexts are supplemental and are removed in
a stable order before required evidence text is reduced. Action planning uses
a 32,000-character aggregate cap and replaces its optional group-engagement
block with the exact empty shape before applying the existing over-cap
disposition.

Every bounded repair or replacement attempt measures its owner-defined dynamic
content before invoking its model. The appraisal initial and repair ceilings
are 20,000 and 24,000 characters, action and resolver authorization use
20,000 and 24,000 characters, and each surface stage uses 32,000 characters.
Semantic appraisal repair feedback keeps the failed rule and exact offending
path while omitting only the validator-owned permitted-path suffix when the
same path domains are already present in `allowed_values`; the protected trace
retains the complete original validation error.
Generic and required-selection goal cognition share the 36,000-character
aggregate cap. Required-selection regeneration reuses the initial static
system prompt; its dynamic `repair_feedback` carries the validation error,
field contract, permitted handles, and a non-empty producer instruction tuple.
A
required-selection regeneration that would cross the cap consumes no
additional model call and fails at the existing pre-state-commit boundary.

Pre-invocation cap exhaustion follows the outcome owned by each stage:

- an irreducible appraisal family is omitted with typed diagnostics;
- an irreducible required-selection producer fails at the pre-state-commit
  boundary with no model call;
- workspace collapse selects its stable first complete bid;
- action planning returns a blocked empty proposal;
- action and resolver authorization deny every candidate;
- text-surface planning returns its validated degraded surface;
- visual-surface planning raises its typed optional-stage failure.

These dispositions consume zero model calls at the over-cap boundary and
authorize no action or resolver side effect. Canonical input validation,
persistent-state validation, reducer and commit invariants, and required owner
failures continue through their typed unrecoverable paths.

## Testing Contract

Deterministic owner tests mirror source modules under
`tests/unit/cognition_core_v2/`, `tests/unit/cognition_resolver/`, and
`tests/unit/nodes/`. The authoritative source-to-test mapping is
`tests/ownership/source_test_impact_manifest.json`; the verifier requires one
exact deterministic node for every mapped semantic owner.

When a mapped production source changes, run the impact command from the
captured baseline:

```powershell
venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run
```

The command validates exact node collection before running the impacted tests.
Cross-boundary propagation tests live under `tests/integration/`, and live LLM
cases run one case at a time with their trace artifact inspected.

## Forbidden Paths

This package does not access adapters, raw database clients, final dialog
wording, platform wire syntax, or untyped relationship scalars.

## Evidence Authority

Every `CognitionEvidenceV2` row carries one closed `authority` value:
`current_event`, `public_scene`, `participant_continuity`,
`private_motive_only`, `character_world_context`,
`conditional_character_guidance`, or `contextual_fact_only`. The validator
rejects unknown values and scopes promoted self-guidance to
`conditional_character_guidance`; it cannot become a current fact or a goal
without current-event support. Conversation-progress evidence preserves the
event's own source timestamp and carries a bounded `temporal_provenance` age
descriptor. Promoted reflection rows preserve valid source timestamps and
invalid timestamp rows fail closed at the projection boundary.

Goal cognition receives the authority label as metadata and keeps the existing
goal output schema, route selection, call count, retry ledger, and 36,000
character aggregate cap. Authority labels guide source weighting; they are not
automatic stance or response decisions.

## One-Objective Goal Arbitration

Goal cognition establishes one primary objective from the current episode,
typed response operation, and observable public scene. A progress item may
continue only when it concerns the same concrete matter; otherwise it remains
supplemental context. Private residue supplies motive, tone, or hesitation and
cannot establish an external fact. Conditional self-guidance supplies tactics
only after the objective is fixed. `selection`, `intention`, `reason`,
`desired_outcome`, `concrete_detail`, and `expected_consequences` stay causally
attached to that objective, and ordered sub-actions are allowed only when they
serve it.
