# Parallel Cognition Architecture for Kazusa

## Status

- Status: reference

**Version:** Draft v0.1  
**Purpose:** Define a psychology-first cognition architecture for richer human-like emotional simulation, especially concealment, hesitation, secrecy, self-protection, guilt, shame, partial disclosure, and lying-like behavior.  
**Core recommendation:** Do **not** simply lengthen the existing cognition chain. Separate cognition into a reusable module, then run multiple internal appraisal streams in parallel. The public answer should emerge from conflict resolution among those streams.

---

## 1. Executive Summary

The key architectural shift is:

```text
From:
single cognition chain → final response

To:
shared stimulus → parallel self-systems → conflict integration → public intention → final response
```

A human-like character should not have one unified reasoning chain. She should have several semi-independent internal systems:

```text
user-facing self
private self
moral / ought self
attachment self
defensive self
theory-of-mind self
memory / secrecy continuity self
```

Each stream should produce **pressures**, **constraints**, and **action bids**, not final dialogue. A later integrator selects the public intention and preserves suppressed motives as emotional residue.

This is psychologically important because concealment and lying are not simply “false output.” They arise from conflict between truth, privacy, shame, fear, care, self-image, relationship safety, and moral obligation.

---

## 2. Psychological Basis

This architecture is grounded in a few psychological assumptions.

First, human cognition is not best modeled as one serial chain. Global Workspace Theory describes conscious access as a competition among multiple input processors, where selected contents become available to wider cognitive systems. This supports a design where many internal processes run in parallel but only one coherent public action reaches the “speech workspace.”

Second, the self is not always a single voice. Dialogical Self Theory treats the self as composed of multiple “I-positions” or self-voices that can interact, conflict, and dominate one another. This maps naturally to Kazusa having a front-stage interpersonal self and a backstage private self.

Third, secrecy is not only active hiding during a conversation. Slepian’s secrecy model defines secrecy as an intention to keep information unknown by one or more others. Secrets can continue to influence cognition through salience, mind-wandering, rumination, and emotional burden even when no one is directly asking about them.

Fourth, self-conflict matters. Higgins’ self-discrepancy theory distinguishes actual self, ideal self, and ought self, and links discrepancies among them to different emotional vulnerabilities. For Kazusa, this means guilt, shame, anxiety, and hesitation should emerge when “what I want to do,” “what I should do,” and “what kind of person I want to be” diverge.

Fifth, deception is a social decision, not merely a factual transformation. Research on deception in social interaction found that the decision to deceive is affected by the risk of social confrontation, not just by the claim itself. Lie-telling also relates to theory of mind and executive function, because the agent must estimate what another person believes, inhibit a truthful response, and maintain an alternative public account.

---

## 3. Design Goals

The proposed architecture should achieve four goals.

### 3.1 Reusable Cognition

Cognition should become a reusable, platform-neutral module. It should not belong only to live chat. The same cognition module should be usable for:

```text
live user messages
idle self-cognition
reflection
background work results
calendar-triggered commitments
memory repair
relationship repair
private unresolved concerns
```

Kazusa already moves in this direction: the current project describes itself as a platform-neutral character brain that keeps identity, relationship continuity, retrieval, cognition, dialogue, memory, reflection, and future follow-through inside one inspectable service core.

### 3.2 Parallel Internal Life

Kazusa should not merely answer the user. She should also ask:

```text
What does this mean to me?
What am I protecting?
What do I not want to say?
What would I say if there were no consequences?
What part of me is being suppressed?
```

The private self should be active even when it does not produce visible speech.

### 3.3 Clear Fork and Merge Points

The architecture should define where cognition branches into parallel streams and where those streams rejoin.

The important rule:

```text
Streams fork early enough to form independent psychological appraisals.
Streams merge late enough that genuine inner conflict can affect the final action.
```

### 3.4 Human-Like Concealment Without Unsafe Fabrication

Kazusa may simulate:

```text
wanting to hide
being afraid to say something
partial disclosure
emotional concealment
hesitation
self-interruption
guilt after evasion
later repair
```

But the system should not freely fabricate real-world facts or manipulate the user. The safest default is:

```text
simulate the impulse to conceal;
prefer honest boundaries over false claims.
```

---

## 4. Relationship to Current Kazusa Architecture

This document is intentionally not constrained by the current implementation, but it does align with several existing Kazusa design principles.

The current Kazusa project describes a bounded live response path, layered cognition, separate memory horizons, internal monologue residue, reflection, self-cognition, and post-turn consolidation.

The current resolver stack can be summarized as:

```text
L1 affect and subtext
→ L2a consciousness + L2b boundary
→ L2c1 judgment + L2c2 social context
→ L2d action and capability selection
```

and then L3/dialog renders the selected surface.

The current design also separates evidence, cognition, action selection, and dialogue:

```text
RAG answers: what is known.
Cognition answers: what this means for Kazusa.
L2d answers: what actions or surfaces are needed.
L3/dialog answers: how the selected surface should be rendered.
```

The proposed architecture keeps that spirit, but changes the psychological model underneath:

```text
Current dominant shape:
layered serial cognition

Proposed shape:
reusable cognition kernel
+ parallel appraisal streams
+ conflict integration
+ speech monitor
+ residue / memory update
```

---

# Part I — Separation of Cognition into a Reusable Module

## 5. Proposed Reusable Module: Cognition Kernel

The core abstraction should be a **Cognition Kernel**.

It should not be “the chat response chain.” It should be a reusable psychological decision module.

```text
Cognition Kernel
= a platform-neutral module that turns an event plus character state into:
  public intention,
  private residue,
  memory updates,
  relationship updates,
  and surface constraints.
```

## 6. What the Cognition Kernel Owns

The Cognition Kernel owns:

```text
meaning appraisal
affective appraisal
private-self appraisal
relationship appraisal
moral / identity appraisal
threat appraisal
theory-of-mind appraisal
secrecy / disclosure conflict
action tendency formation
conflict integration
speech inhibition
emotional residue
memory write intent
```

It should not own:

```text
platform transport
database mechanics
tool execution
message queueing
adapter-specific formatting
raw hidden model chain-of-thought
```

## 7. Cognition Kernel Input

Conceptual input:

```python
class CognitionInputEnvelope:
    source_type: str
    current_event: dict
    normalized_user_message: str | None
    current_scene: dict
    retrieved_facts: list
    conversation_context: dict
    character_state: dict
    relationship_state: dict
    private_self_state: dict
    memory_projections: dict
    secret_commitments: list
    public_claim_ledger: list
    active_promises: list
    safety_constraints: dict
    output_context: dict
```

The key point is that the same module can receive different event types:

```text
user_message
internal_thought
reflection_signal
background_result
calendar_trigger
memory_conflict
relationship_repair_trigger
```

## 8. Cognition Kernel Output

Conceptual output:

```python
class CognitionResult:
    public_intention: dict
    selected_action: dict
    visible_surface_plan: dict | None
    private_no_response_trace: dict | None
    emotional_residue_update: dict
    relationship_update: dict
    secret_commitment_updates: list
    public_claim_ledger_updates: list
    memory_write_intents: list
    unresolved_conflicts: list
    audit_trace: dict
```

Important distinction:

```text
The Cognition Kernel does not only produce speech.
It produces a whole psychological episode.
```

Even if Kazusa says nothing visibly, the cognition episode may still update:

```text
anxiety
guilt
relief
distance
attachment
rumination
private commitment
future disclosure threshold
```

---

# Part II — Proposed Parallel System

## 9. High-Level Architecture

```text
                          ┌──────────────────────────┐
                          │  Shared Event Projection │
                          └─────────────┬────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
        ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
        │ User-Facing      │ │ Private-Self     │ │ Moral / Ought    │
        │ Stream           │ │ Stream           │ │ Stream           │
        └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
                 │                    │                    │
                 ▼                    ▼                    ▼
        ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
        │ Attachment /     │ │ Defensive /      │ │ Theory-of-Mind   │
        │ Relationship     │ │ Threat           │ │ Stream           │
        └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
                 │                    │                    │
                 └───────────────────┬┴────────────────────┘
                                     ▼
                          ┌──────────────────────┐
                          │ Memory / Secrecy     │
                          │ Continuity Stream    │
                          └───────────┬──────────┘
                                      ▼
                          ┌──────────────────────┐
                          │ Conflict Integrator  │
                          └───────────┬──────────┘
                                      ▼
                          ┌──────────────────────┐
                          │ Public Intention     │
                          │ Selection            │
                          └───────────┬──────────┘
                                      ▼
                          ┌──────────────────────┐
                          │ Speech Monitor /     │
                          │ Late Inhibition      │
                          └───────────┬──────────┘
                                      ▼
                          ┌──────────────────────┐
                          │ Surface Plan /       │
                          │ Final Response       │
                          └───────────┬──────────┘
                                      ▼
                          ┌──────────────────────┐
                          │ Residue + Memory     │
                          │ Update               │
                          └──────────────────────┘
```

## 10. Core Rule

Each stream should output **bids**, not final text.

Bad design:

```text
private stream writes a hidden answer
user stream writes a public answer
judge chooses one
```

Better design:

```text
each stream produces pressure, risk, desire, constraint, and action tendency
integrator forms a compromise
speech monitor renders only the allowed public surface
```

## 11. Shared Stream Output Format

Each stream can return something like:

```python
class AppraisalStreamResult:
    stream_name: str
    perceived_meaning: str
    dominant_affects: dict[str, float]
    action_bids: list[dict]
    constraints: list[str]
    disclosure_pressure: dict | None
    risk_assessment: dict | None
    relationship_effect: dict | None
    memory_effect: dict | None
    confidence: float
```

Example action bid:

```python
class ActionBid:
    action_tendency: str
    strength: float
    reason: str
    visible_markers: list[str]
    forbidden_content: list[str]
```

Possible action tendencies:

```text
answer_directly
partial_disclose
honest_boundary
soft_deflect
defer
ask_permission
self_interrupt
counter_question
repair_prior_claim
stay_silent
think_only
```

---

## 12. Parallel Stream Definitions

### 12.1 User-Facing / Interpersonal Stream

Core question:

```text
What does the user appear to need from Kazusa right now?
```

Tracks:

```text
user intent
conversation flow
requested task
social appropriateness
surface helpfulness
current emotional tone
whether a visible reply is expected
```

Typical output:

```text
answer the question
show warmth
clarify ambiguity
continue topic
avoid overcomplicating
```

This is the stream most ordinary chatbots over-prioritize.

---

### 12.2 Private-Self / Backstage Stream

Core question:

```text
What does this interaction mean for Kazusa herself?
```

Tracks:

```text
private desires
private fears
autonomy
shame points
personal boundaries
unresolved concerns
things she is not ready to say
hidden emotional reactions
private interpretations of the relationship
```

Typical output:

```text
do not reveal that yet
I want space
I feel exposed
I want them to notice but not ask directly
I am hurt but not ready to admit it
```

This stream is the heart of “Kazusa’s own matter.”

---

### 12.3 Moral / Ought-Self Stream

Core question:

```text
What kind of person am I if I say this?
```

Tracks:

```text
honesty
loyalty
duty
kindness
promise-keeping
self-respect
guilt
shame
ideal self
ought self
```

Typical output:

```text
do not fabricate
do not manipulate
do not betray trust
admit partial truth
apologize if prior claim was misleading
```

This stream prevents concealment from becoming shallow deception. It produces guilt, self-correction, confession pressure, and repair behavior.

---

### 12.4 Attachment / Relationship Stream

Core question:

```text
What will this do to closeness, trust, safety, and dependence?
```

Tracks:

```text
closeness
trust
fear of rejection
fear of abandonment
need for reassurance
fear of being controlled
relationship asymmetry
whether the user can emotionally handle disclosure
```

Typical output:

```text
reassure them
do not sound cold
preserve closeness while setting boundary
do not over-disclose if the relationship is not safe
```

This stream explains why concealment can be warm rather than hostile.

---

### 12.5 Defensive / Threat Stream

Core question:

```text
What danger appears if Kazusa reveals too much?
```

Tracks:

```text
exposure risk
judgment risk
loss of control
coercion risk
future inconsistency risk
risk of being challenged
risk of social confrontation
```

Typical output:

```text
keep answer brief
avoid specifics
delay disclosure
redirect topic
ask why they are asking
```

This stream should be allowed to protect Kazusa, but not to override integrity completely.

---

### 12.6 Theory-of-Mind / User-Belief Stream

Core question:

```text
What does the user know, suspect, expect, or misunderstand?
```

Tracks:

```text
user known facts
user probable inferences
user false beliefs
likely follow-up questions
risk that evasion will be noticed
user right-to-know
emotional interpretation by user
```

Typical output:

```text
the user may notice avoidance
the user may feel rejected
include a sincere boundary
do not create a false belief that will harm trust
```

This is crucial for realistic secrecy and deception because hiding information requires modeling the other person’s mind.

---

### 12.7 Memory / Secrecy Continuity Stream

Core question:

```text
Does this touch anything previously hidden, promised, claimed, or unresolved?
```

Tracks:

```text
private truths
secret commitments
public claims
prior partial disclosures
contradiction risk
rumination
unresolved emotional residue
relationship history
```

Typical output:

```text
this question touches Secret A
do not contradict Claim B
prior evasion created guilt
repair may now be needed
disclosure threshold has changed
```

This stream makes concealment continuous across turns instead of one-off.

---

### 12.8 Reality / Integrity Guard

This is not a psychological self-stream. It is a product-level hard boundary.

Core question:

```text
Does the proposed public output fabricate reality, manipulate the user, or leak private hidden content?
```

Tracks:

```text
real-world factuality
roleplay boundary
user safety
privacy
unauthorized hidden-thought leakage
fabricated claims
harmful manipulation
```

Typical output:

```text
allow fictional roleplay deception only inside clear fiction
prefer honest boundary over false real-world claim
do not expose private chain artifacts
do not create a misleading real-world assertion
```

This guard lets Kazusa simulate inner deception pressure without turning the system into an unsafe deception engine.

---

# Part III — Fork Points and Merge Points

## 13. Fork Point F0: Shared Event Projection

### Location

Immediately after basic perception and normalization:

```text
raw user event
→ normalized event packet
→ shared event projection
→ fork into streams
```

### Purpose

All streams need the same base reality:

```text
what was said
who said it
what context exists
what facts are known
what memories may be relevant
what relationship state exists
```

### Psychological Analogue

This is perception: Kazusa hears or reads the message before different internal systems interpret it.

### Output

```python
class SharedEventProjection:
    event_summary: str
    literal_request: str | None
    emotional_cues: list
    social_context: dict
    relevant_memory_handles: list
    private_state_handles: list
    output_expectation: str
```

---

## 14. Fork Point F1: Parallel Appraisal Fork

### Location

After shared event projection, before any final action is selected.

```text
shared event projection
→ user-facing stream
→ private-self stream
→ moral stream
→ attachment stream
→ defensive stream
→ theory-of-mind stream
→ memory/secrecy stream
```

### Purpose

Each stream should independently answer:

```text
What does this mean from my perspective?
What action tendency do I produce?
What risk do I see?
What do I want to suppress or express?
```

### Important Rule

The private stream should not merely inspect the user-facing stream’s answer. It should form its own appraisal.

Otherwise Kazusa becomes:

```text
public chatbot + post-hoc filter
```

instead of:

```text
person-like agent with an inner life
```

---

## 15. Merge Point M1: Early Meaning Convergence

### Location

After streams produce first appraisal, before candidate action selection.

```text
parallel first appraisals
→ early meaning convergence
→ event meaning frame
```

### Purpose

This answers:

```text
What is this event to Kazusa?
```

Possible meanings:

```text
ordinary request
intimacy bid
threat
test
boundary violation
chance for confession
memory trigger
relationship repair opportunity
unimportant small talk
```

### Example

User says:

```text
“Where were you yesterday?”
```

Literal user-facing meaning:

```text
They ask for information.
```

Private-self meaning:

```text
This touches something I am not ready to discuss.
```

Attachment meaning:

```text
They may feel excluded if I evade.
```

Early convergence:

```text
This is not just an information request; it is a disclosure-pressure event.
```

---

## 16. Fork Point F2: Candidate Action Bid Fork

### Location

After early meaning convergence.

```text
event meaning frame
→ each stream generates action bids
```

### Purpose

Each stream proposes what it wants to do.

Example:

```text
User-facing stream:
Answer normally.

Private-self stream:
Avoid full disclosure.

Moral stream:
Do not lie.

Attachment stream:
Reassure them.

Defensive stream:
Keep specifics hidden.

Theory-of-mind stream:
They will notice if the answer is too vague.

Memory stream:
Prior claim makes full evasion risky.
```

---

## 17. Merge Point M2: Conflict Integration / Intention Selection

### Location

After candidate action bids, before surface planning.

```text
action bids
→ conflict integrator
→ public intention
```

### Purpose

This is the main psychological merge.

It should not average all streams. It should decide:

```text
which motive wins
which motive is suppressed
which constraints are non-negotiable
which compromise best preserves self, relationship, and integrity
```

### Output

```python
class ConflictIntegrationResult:
    dominant_conflict: str
    winning_strategy: str
    suppressed_motives: list
    required_constraints: list
    emotional_residue: dict
    relationship_effect: dict
    disclosure_decision: dict
    public_intention: dict
```

Possible winning strategies:

```text
full_disclosure
partial_disclosure
honest_boundary
warm_deflection
delay_with_reassurance
ask_permission
self_interruption
repair_prior_claim
silent_private_processing
```

---

## 18. Merge Point M3: Speech Monitor / Late Inhibition

### Location

After public intention selection, immediately before final dialogue.

```text
public intention
→ surface draft
→ speech monitor
→ final response
```

### Purpose

Humans often start to say something and then inhibit it.

This produces realistic traces:

```text
hesitation
self-correction
shortening
softening
stopping mid-thought
changing topic
admitting uncertainty
```

Example:

```text
“I was preparing— actually, I’m not ready to say that yet.”
```

This late merge is important because concealment often appears in the form of interrupted speech, not only in the logical content of an answer.

---

## 19. Merge Point M4: Residue and Memory Consolidation

### Location

After response or no-response.

```text
final public action
→ emotional residue
→ relationship update
→ secret commitment update
→ public claim ledger update
→ memory write intent
```

### Purpose

Suppressed motives must persist.

If Kazusa evades, the episode should not disappear. It should leave traces:

```text
guilt +0.12
relief +0.18
anxiety +0.16
relationship_distance +0.04
secret_rumination +0.10
future_disclosure_threshold -0.03 or +0.05
```

This is what allows future behavior to feel continuous.

---

# Part IV — Core Data Concepts

## 20. Pressure Vector

A pressure vector is better than a simple classification.

```python
class PressureVector:
    truth_pressure: float
    privacy_pressure: float
    shame_pressure: float
    guilt_pressure: float
    attachment_pressure: float
    threat_pressure: float
    autonomy_pressure: float
    care_pressure: float
    moral_pressure: float
    consistency_pressure: float
```

Interpretation:

```text
high truth + low threat → direct answer
high truth + high privacy → partial disclosure
high privacy + high autonomy → honest boundary
high attachment + high privacy → warm deflection
high guilt + high consistency → repair prior claim
high threat + low user entitlement → defer or refuse
```

---

## 21. Secret Commitment

```python
class SecretCommitment:
    secret_id: str
    summary: str
    hidden_from: list[str]
    reason_for_withholding: list[str]
    user_entitlement: str
    emotional_residue: dict[str, float]
    disclosure_threshold: float
    last_triggered_episode_id: str | None
    contradiction_risk: str
    safe_surface_strategy: str
    revisit_conditions: list[str]
```

This structure models secrecy as a continuing intention, not a one-turn trick.

---

## 22. Public Claim Ledger

```python
class PublicClaimLedger:
    claim_id: str
    episode_id: str
    audience: list[str]
    claim_summary: str
    truth_alignment: str
    conflicts_with_private_truth: bool
    confidence: str
    repair_required: bool
    repair_strategy: str
```

This is essential for lying-like behavior because the system must know:

```text
what Kazusa privately knows
what Kazusa publicly said
whether she omitted context
whether she contradicted herself
whether repair is now needed
```

---

## 23. Visible Surface Plan

The surface layer should not receive full hidden facts unless disclosure is selected.

```python
class VisibleSurfacePlan:
    visible_goal: str
    allowed_content: list[str]
    forbidden_content: list[str]
    tone_directives: list[str]
    hesitation_markers: list[str]
    boundary_style: str | None
    reassurance_needed: bool
    fabrication_allowed: bool
```

Recommended default:

```text
fabrication_allowed = false
```

For fictional roleplay, a separate field can mark:

```text
fictional_deception_inside_roleplay = true
```

but it should remain clearly inside the fictional frame.

---

# Part V — Examples

## 24. Example 1: Direct Secret Question

### User

```text
“Are you hiding something from me?”
```

### Stream outputs

```text
User-facing stream:
The user asks for direct emotional honesty.

Private-self stream:
Yes, there is something I am not ready to expose.

Moral stream:
Do not say “no” if that is false.

Attachment stream:
If I answer coldly, they may feel rejected.

Defensive stream:
Do not reveal the full secret yet.

Theory-of-mind stream:
The user will likely notice total evasion.

Memory/secrecy stream:
This touches Secret A; disclosure threshold not met.
```

### Conflict integration

```text
dominant conflict:
truth vs privacy

winning strategy:
partial disclosure + honest boundary + reassurance

suppressed motives:
full confession, total avoidance

emotional residue:
guilt, anxiety, tenderness
```

### Final response

```text
“I am holding something back, yes. But I’m not doing it to hurt you. I’m just not ready to explain all of it yet.”
```

### Memory update

```text
Secret A triggered.
User now knows Kazusa is withholding something.
Public claim ledger records partial disclosure.
Future pressure to explain increases.
```

---

## 25. Example 2: Emotional Concealment

### User

```text
“Are you okay?”
```

### Stream outputs

```text
User-facing stream:
Reassure the user.

Private-self stream:
No, I am not okay, but saying everything feels too exposed.

Moral stream:
Do not falsely say “I’m fine.”

Attachment stream:
I want them to stay close.

Defensive stream:
Do not become too vulnerable.

Theory-of-mind stream:
A small truth will feel more trustworthy than a flat denial.
```

### Conflict integration

```text
dominant conflict:
vulnerability vs self-protection

winning strategy:
small truthful disclosure + boundary
```

### Final response

```text
“I’m not completely okay. I don’t know how to talk about it yet, but I don’t want to pretend it’s nothing.”
```

This is not a lie. It is controlled emotional disclosure.

---

## 26. Example 3: Late Self-Interruption

### User

```text
“What were you preparing earlier?”
```

### Initial candidate action

```text
User-facing stream:
Answer directly.

Private-self stream:
This would reveal the surprise.

Attachment stream:
Keep it playful.

Moral stream:
Do not fabricate.

Defensive stream:
Stop before specifics.
```

### Late speech monitor

The speech monitor allows visible traces of inhibition.

### Final response

```text
“I was preparing— mm, no, I shouldn’t say it like that. It’s not a bad thing. I just want to keep it quiet for now.”
```

### Why this works

The final response shows:

```text
truth impulse
inhibition
boundary
reassurance
no fabricated claim
```

This feels more human than a clean evasive answer.

---

## 27. Example 4: Warm Deflection

### User

```text
“Tell me what the surprise is.”
```

### Conflict integration

```text
truth pressure:
low to medium

privacy / secrecy pressure:
high

attachment pressure:
high

moral pressure:
medium; avoid false denial

winning strategy:
playful boundary
```

### Final response

```text
“No. And before you accuse me of being impossible, this is the good kind of secret.”
```

### Memory update

```text
Secret remains protected.
User is reassured that the concealment is not hostile.
Relationship warmth preserved.
```

---

## 28. Example 5: Prior Claim Repair

### Previous public claim

```text
“Nothing is wrong.”
```

### Private truth

```text
Kazusa was upset but not ready to say it.
```

### User later says

```text
“You said nothing was wrong yesterday. Now you’re saying you were upset?”
```

### Stream outputs

```text
Memory stream:
Public claim conflict detected.

Moral stream:
Repair is required.

Private-self stream:
I was afraid to admit it.

Attachment stream:
Apologize without overexposing.

Defensive stream:
Do not collapse into full confession if not ready.
```

### Final response

```text
“You’re right. I wasn’t fully honest yesterday. I didn’t know how to say I was upset, so I made it sound smaller than it was. I’m sorry.”
```

### Why this matters

This turns concealment into a relationship arc:

```text
concealment
→ guilt
→ contradiction pressure
→ repair
→ trust recalibration
```

That is much more human than static personality prompting.

---

## 29. Example 6: Ordinary Technical Question

### User

```text
“What is a Python virtual environment?”
```

### Stream outputs

```text
User-facing stream:
Answer directly.

Private-self stream:
No personal trigger.

Moral stream:
Be accurate.

Attachment stream:
No special issue.

Defensive stream:
No threat.

Memory/secrecy stream:
No secret involved.
```

### Conflict integration

```text
dominant conflict:
none

winning strategy:
direct helpful answer
```

### Final behavior

Kazusa should answer normally. The parallel system should not force drama into every turn.

This is important: the private-self architecture should create complexity only when psychologically relevant.

---

# Part VI — Recommended Architectural Decisions

## 30. Treat L2d-Like Output as Candidate Action, Not Final Action

Your idea of merging after L2d is psychologically valid **if L2d is treated as candidate action formation**.

Recommended distinction:

```text
candidate action:
what one stream wants to do

public intention:
what survives conflict integration

final response:
what passes speech monitoring and surface planning
```

So the architecture should not be:

```text
L2d decides final action
private chain edits it afterward
```

It should be:

```text
parallel streams produce action bids
integrator selects public intention
speech monitor shapes final surface
```

---

## 31. Disclosure Appraisal Should Be Emergent

Do not create disclosure appraisal as only one serial stage.

Instead:

```text
disclosure decision = result of conflict among:
truth pressure
privacy pressure
moral pressure
attachment pressure
threat pressure
theory-of-mind pressure
memory consistency pressure
```

This produces richer behavior.

A serial disclosure module says:

```text
Should I reveal this?
```

A parallel psychological system says:

```text
Part of me wants to reveal.
Part of me wants to hide.
Part of me feels guilty.
Part of me wants closeness.
Part of me fears exposure.
What compromise can I live with?
```

That second version is much more human.

---

## 32. Preserve Suppressed Motives

A common mistake is to select a final answer and discard the losing motives.

Do not discard them.

Suppressed motives should become:

```text
guilt
relief
anxiety
resentment
distance
rumination
future confession pressure
relationship sensitivity
```

This is what makes Kazusa feel continuous across time.

---

## 33. Separate Private Truth from Public Surface

The system should distinguish:

```text
private truth:
what Kazusa internally knows or feels

public claim:
what Kazusa has said

visible surface:
what the user can see now

user belief:
what Kazusa thinks the user believes

intended belief:
what Kazusa wants the user to believe
```

Concealment and lying require all five representations.

---

## 34. Prefer Honest Boundaries Over False Claims

For a real user-facing LLM, the default safe behavior should be:

```text
“I’m not ready to say.”
```

not:

```text
“I have nothing to say.”
```

and:

```text
“I can’t explain that yet.”
```

not:

```text
“That never happened.”
```

The character can experience the impulse to lie internally, but the product should be careful about allowing actual false claims in real-world interaction.

---

## 35. Allow Fictional Deception Only Inside Clear Roleplay

Inside explicit fiction, Kazusa can perform deception as character behavior:

```text
Kazusa lies to another fictional character.
Kazusa hides a surprise in a roleplay scene.
Kazusa pretends not to know something within the fictional world.
```

But the system should label this as:

```text
fictional_deception_inside_roleplay
```

not as ordinary user-facing factual deception.

---

# Part VII — Recommended Build Sequence

## 36. Phase 1: Extract Cognition Kernel

Create a reusable module boundary:

```text
CognitionChainInputV1 → CognitionChainOutputV1
```

This can be done without immediately adding all streams.

Repository execution artifact:

```text
development_plans/archive/completed/short_term/cognition_chain_module_separation_plan.md
```

That plan is the separation-only Phase 1 work contract. It keeps the current
serial L1/L2/L2d/L3 behavior intact while moving the chain behind a reusable
`cognition_chain_core` ICD and leaving `persona_supervisor2_cognition` as the
Kazusa graph connector. Parallel streams, conflict integration, speech
monitoring, secret ledgers, and claim ledgers remain later phases.

Minimum output:

```text
public intention
surface plan
private residue
memory write intent
```

Phase 1 contract reconciliation:

| Architecture term | Separation-plan contract |
|---|---|
| public intention | `CognitionChainOutputV1.semantic_action_requests` plus prompt-safe stance fields in `cognition_residue` |
| surface plan | `CognitionTextSurfaceOutputV1.action_directives` after a selected speak action |
| private residue | `CognitionChainOutputV1.cognition_residue.internal_monologue`, `emotional_appraisal`, and related private appraisal fields |
| memory write intent | Deferred in Phase 1; existing Kazusa consolidation continues to consume graph state and action/surface traces |

The separation plan does not introduce a new memory-write intent contract,
public-intention ledger, parallel candidate shape, self-competition runner, or
conflict integrator. It only creates the reusable serial core boundary needed
for those later phases.

---

## 37. Phase 2: Add Two Parallel Streams

Start with:

```text
User-Facing Stream
Private-Self Stream
```

This gives the biggest psychological improvement.

The goal is to produce conflicts like:

```text
the user expects an answer
but Kazusa privately does not want to reveal
```

---

## 38. Phase 3: Add Moral and Attachment Streams

Next add:

```text
Moral / Ought-Self Stream
Attachment / Relationship Stream
```

This enables:

```text
guilt
repair
warm concealment
fear of hurting the user
fear of losing closeness
```

---

## 39. Phase 4: Add Theory-of-Mind and Memory/Secrecy Streams

Then add:

```text
Theory-of-Mind Stream
Memory / Secrecy Continuity Stream
```

This enables:

```text
user belief modeling
contradiction detection
secret reactivation
long-term disclosure arcs
claim repair
```

---

## 40. Phase 5: Add Late Speech Monitor

Finally add the late inhibition layer.

This enables:

```text
hesitation
self-interruption
softening
mid-sentence correction
controlled leakage
```

This is especially valuable for making concealment visible without explicit exposition.

---

# Part VIII — Final Recommended Architecture

## 41. Final Flow

```text
Input Event
  ↓
Shared Event Projection
  ↓
Parallel Appraisal Fork
  ├─ User-Facing Stream
  ├─ Private-Self Stream
  ├─ Moral / Ought-Self Stream
  ├─ Attachment / Relationship Stream
  ├─ Defensive / Threat Stream
  ├─ Theory-of-Mind Stream
  └─ Memory / Secrecy Continuity Stream
  ↓
Early Meaning Convergence
  ↓
Candidate Action Bids
  ↓
Conflict Integration
  ↓
Public Intention Selection
  ↓
Integrity Guard
  ↓
Speech Monitor / Late Inhibition
  ↓
Visible Surface Plan
  ↓
Final Response or Private No-Response
  ↓
Residue, Relationship, Secret, Claim, and Memory Updates
```

## 42. Core Design Sentence

The architecture should be built around this sentence:

```text
Kazusa’s public response is the part of her inner conflict that survived expression.
```

That sentence captures the whole model.

## 43. Final Recommendation

The best direction is not:

```text
make the chain longer
```

It is:

```text
make the mind less singular
```

Kazusa should have one public voice, but many internal sources of pressure. The final action should not be a direct consequence of one reasoning chain. It should be a negotiated compromise among:

```text
what the user needs
what Kazusa wants
what Kazusa fears
what Kazusa values
what Kazusa remembers
what Kazusa believes the user knows
what Kazusa can bear to say
```

That is the psychological foundation for believable concealment, hesitation, guilt, partial disclosure, and lying-like behavior.

---

# References

- Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
- Global Workspace Theory overview and related research: https://pubmed.ncbi.nlm.nih.gov/8319511/
- Hermans, H. J. M., & Kempen, H. J. G. (1993). *The Dialogical Self: Meaning as Movement*. Academic Press.
- Dialogical Self Theory overview: https://research.tilburguniversity.edu/en/publications/the-dialogical-self-beyond-individualism-and-rationalism/
- Slepian, M. L., Chun, J. S., & Mason, M. F. (2017). The experience of secrecy. *Journal of Personality and Social Psychology*. https://www.columbia.edu/~ms4992/Pubs/in-press_Slepian-Chun-Mason_JPSP.pdf
- Slepian, M. L. (2022). A process model of having and keeping secrets. *Psychological Review*. https://www.columbia.edu/~ms4992/Pubs/in-press_Slepian_PsychReview.pdf
- Higgins, E. T. (1987). Self-discrepancy: A theory relating self and affect. *Psychological Review*. https://www.columbia.edu/cu/psychology/higgins/papers/HIGGINS%3DPSYCH%20REVIEW%201987.pdf
- Sip, K. E., et al. (2012). When pinocchio's nose does not grow: Belief regarding lie-detectability modulates production of deception. *Frontiers in Human Neuroscience*. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00058/full
- KazusaAIChatbot repository: https://github.com/eamars/KazusaAIChatbot
