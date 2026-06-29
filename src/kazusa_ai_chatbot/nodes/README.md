# Cognition Nodes

`kazusa_ai_chatbot.nodes` owns the persona-facing runtime nodes that turn a
surviving trigger into internal cognition, selected action/surface work, and
post-turn consolidation input.

This package is not a generic assistant chain. It is the character-brain layer
inside the larger service boundary:

```text
adapter/debug client
  -> brain service queue/intake
  -> relevance and perception
  -> persona nodes
       decontextualization -> cognition resolver -> selected surfaces
  -> persistence/consolidation
  -> scheduler/reflection outside live chat
```

The critical design decision is that these nodes separate evidence, thought,
decision, action selection, expression, and persistence. RAG retrieves what is
known. Cognition decides what the evidence means for the active character. L2d
selects zero or more semantic actions. Selected L3 surface handlers and dialog
render output when a surface is needed. Consolidation later decides what
durable state should change from prompt-safe episode evidence.

## Module Boundary

The package currently contains five major node groups:

| Area | Main files | Owns |
| --- | --- | --- |
| Relevance and perception | `persona_relevance_agent.py`, `persona_supervisor2_msg_decontexualizer.py` | Whether to answer, current media observation, current-message rewrite, referent status. |
| Persona orchestration | `persona_supervisor2.py` | Live turn graph: decontextualization, cognition resolver, memory lifecycle, selected action/surface or no-response routing. |
| Cognition and action initialization | `persona_supervisor2_cognition*.py`, `boundary_profile.py`, `linguistic_texture.py` | Layered internal appraisal, stance, boundary judgment, L2d action initialization, and selected L3 surface directives. |
| Dialog | `dialog_agent.py` | Final text rendering from selected L3 surface directives. |
| Consolidation handoff | `persona_supervisor2.py` | Completed persona state is handed to `kazusa_ai_chatbot.consolidation`, which owns extraction helpers, origin projection, target validation, and durable write routing. |

The nodes consume platform-neutral state. Platform wire syntax must already be
normalized by adapters and the brain service into `message_envelope`,
`prompt_message_context`, `reply_context`, `CognitiveEpisode`, global user ids,
and bounded history fields.

## Live Persona Turn

The top-level service graph routes into `persona_supervisor2` only after the
queue, media descriptor, relevance gate, and conversation-progress loader have
done their work.

Inside `persona_supervisor2`, the live persona graph is:

```text
stage_0_msg_decontexualizer
  -> stage_1_goal_resolver
       bounded resolver loop
       first cycle may prewarm shared `memory` evidence before L2a
       each cycle runs L1 subconscious
       each cycle runs L2 consciousness + boundary + judgment + social context
       each cycle runs L2d semantic action selection
       optional cognition-selected capabilities:
         RAG 2 evidence, web/current evidence, HIL blockers,
         approval blockers, private self-resolution
       state["internal_monologue"]
       state["action_specs"]
  -> stage_2_memory_lifecycle
       consumes `memory_lifecycle_update` route intents
       aliases active commitments for specialist review
       materializes private `apply_memory_lifecycle_update` actions
       writes prompt-safe `memory_lifecycle_context`
  -> stage_2a_background_work_enqueue
       persists selected `background_work_request` jobs
       writes prompt-safe pending results into `pre_surface_action_results`
  -> route
       selected speak action -> stage_3_action
           selected L3 text directives
           dialog_agent
           text surface output + action results
       no visible surface    -> stage_3_no_response
           private surface output + private action results
  -> final_dialog + episode_trace + consolidation_state
```

The resolver is a recurrence controller, not a separate assistant harness. RAG,
web/current evidence, HIL blockers, approval blockers, and private
self-resolution are capability observations that feed another complete
L1/L2/L2d cognition pass before final L3/dialog rendering. Simple turns can
still behave like a one-cycle resolver: cognition selects `speak` or silence
without requesting extra evidence.

The first resolver cycle may also run a limited shared-memory prewarm in
parallel with early cognition work. That prewarm uses the existing RAG intake
and shared persistent-memory worker, merges only safe shared
`memory_evidence` before L2a, and is not recorded as a resolver capability
observation. Scoped `user_memory_units` retrieval remains available only
through cognition-selected RAG 2 evidence.

The returned `consolidation_state` is the completed persona snapshot. The
service uses it after visible surface handling to record conversation progress
and run consolidation. That timing is intentional: the user should not wait for
memory writes, and durable writes should not happen before the selected
surfaces and action results are known.

## Action Spec And Surface Routing

L2d is the only cognition node that may initialize action work. It runs after
L2c1 judgment and L2c2 social-context appraisal, when the final stance, intent,
and social temperature already exist, and before selected L3 surface handlers
run. L2d emits semantic `action_requests`; deterministic code materializes
valid requests into graph-visible `ActionSpecV1` rows.

L2a consciousness, L2b boundary appraisal, L2c1 judgment, and L2c2 social
context do not emit action specs. L2a interprets the stimulus, L2b constrains
the safe envelope, L2c1 adjudicates stance/intent, and L2c2 gives L2d the
same social-distance evidence L3 used to own. Action selection is a later,
separate concern so weaker local models are not asked to interpret, judge,
plan, and write executable envelopes in one step.

Text response is now a selected surface action. When L2d selects `speak`,
the text surface path runs L3 directives plus `dialog_agent` and records the
final text as `SurfaceOutputV1(surface_kind="text")`. When no visible surface
is selected, the graph may still produce private finalization or action results
for consolidation without fabricating user-visible dialog.

Memory lifecycle is also split by ownership. L2d may select
`memory_lifecycle_update` when the current turn might affect an active
commitment, but it does not choose a commitment, alias, database id, or lifecycle
status. The memory lifecycle specialist runs before selected L3 text, chooses
prompt-safe aliases, and deterministic code resolves those aliases into
`apply_memory_lifecycle_update` actions. L3 receives only
`memory_lifecycle_context` role anchors, so content can avoid reopening a
fulfilled promise without seeing persistence ids.

Background work handoff follows the same semantic/action separation. L2d may
select `background_work_request` with only a semantic task brief, reason, and
bounded output size. The persona graph persists those jobs in
`stage_2a_background_work_enqueue` before any selected L3 text surface runs,
then places the pending queue result in `pre_surface_action_results`. A
background-work router later chooses the worker and task; L2d does not choose
worker-local parameters. L3 may acknowledge only the semantic pending or failed
state; it must not see raw job ids, target ids, adapter ids, leases, retry
state, filesystem paths, worker names, or worker internals.

After a visible text response is sent, the service may run a post-surface
lifecycle review for active commitments. That review is fed only by direct
current-user active-commitment rows plus the completed `final_dialog` and
user-visible text-surface fragments. It does not use RAG reachability or
state-projected active commitments as its row source. The prompt receives only
prompt-safe aliases, and deterministic code materializes only
`apply_memory_lifecycle_update` action specs through the existing lifecycle
action path.

L3 text and L3 image are treated as registered surface handlers. The current
runtime implements text-surface routing and keeps visual directives/promptable
image guidance available for future image surfaces; it does not call an
external image generation service in this implementation slice.

Action and surface outcomes are collected into `EpisodeTraceV1`:

- `action_specs`: validated action residues selected for the episode.
- `action_results`: validation, private-action, scheduling, or rejection
  outcomes.
- `surface_outputs`: text, private, image, or future surface artifacts.

The consolidator consumes a prompt-safe projection of the episode trace. It
does not select actions, execute actions, call the dispatcher, call the
scheduler, or trigger cognition.

Durable consolidation routing lives in `kazusa_ai_chatbot.consolidation`.
Nodes produce prompt-safe state and helper-node outputs; the consolidation
package builds the deterministic target plan, validates write lanes, and keeps
user, group-channel, character, and internal targets separate before DB helper
calls.

## Cognitive Episode Contract

`CognitiveEpisode` is the source-neutral wrapper around the current stimulus.
It lets the same cognition graph run over several source shapes without asking
every prompt to rediscover transport details.

Current supported cognition prompt variants are selected by
`cognition_chain_core/prompt_selection.py`:

| Trigger source | Input source | Output modes | Normal use |
| --- | --- | --- | --- |
| `user_message` | `dialog_text` plus optional `image_observation` and `audio_observation` | `visible_reply`, `think_only`, `silent` | Normal live chat. |
| `background_work_result_ready` | `background_work_result` | `visible_reply`, `think_only`, `silent` | Completed or failed background-work result returning through cognition and dialog. |
| `reflection_signal` | `reflection_artifact` | `think_only`, `preview`, `silent` | Reflection dry-run cognition. |
| `internal_thought` | `internal_monologue` | `think_only`, `preview`, `silent` | Legacy prompt-variant label used by current self-cognition worker paths. Architecturally this is a self-cognition trigger, not a downstream action consumer. |

The selector validates the episode and exposes only prompt-safe fields:

- current-turn media observations are bounded and structured,
- background-work result sources expose only task/result/failure summaries and
  semantic request context; legacy artifact result sources keep their old
  artifact/failure summary projection,
- reflection artifacts are passed as one visible artifact,
- internal-thought dry runs pass a residue id, an internal monologue, and an
  audit-only action latch,
- raw platform syntax and storage internals stay outside cognition prompts.

## Psychological Model

The layer names borrow psychological language as engineering metaphors. They
are not clinical claims. The goal is to make a character response inspectable
and reconstructable:

```text
stimulus
  -> perceptual cleanup
  -> retrieved evidence
  -> instinctive affect
  -> conscious interpretation
  -> boundary judgment
  -> social/directive shaping
  -> dialog wording
  -> durable memory decision
```

The implementation treats "subconscious", "consciousness", and "dialog" as
separate contracts so that a weaker local model is not asked to solve every
problem in one prompt. Each stage has a small output shape, and downstream
stages receive the prior stage's result as evidence rather than hidden state.

## How Consciousness Is Generated

Consciousness in this system is not a free-form stream of raw model reasoning.
It is an explicit, schema-shaped artifact produced by the L2 cognition stack.
The public "thought chain" is therefore:

```text
L1 emotional_appraisal + interaction_subtext
  -> L2a internal_monologue + logical_stance candidate + character_intent candidate
  -> L2b boundary_core_assessment
  -> L2c1 final logical_stance + character_intent + judgment_note
  -> L2c2 social_distance + emotional_intensity + vibe_check + relational_dynamic
  -> L2d action_requests/action_specs
  -> selected L3 surface directives
  -> dialog final_dialog when a text surface is selected
  -> episode_trace
```

`internal_monologue` is the explicit character-facing thought artifact. It is
not the model provider's hidden chain of thought. It is generated text that
summarizes how the character interprets the current stimulus, retrieved
evidence, memory, relationship state, and instinctive affect.

The system keeps this distinction because the runtime needs inspectability
without depending on hidden model reasoning. If a turn needs to be audited, the
stable artifacts are `emotional_appraisal`, `interaction_subtext`,
`internal_monologue`, `logical_stance`, `character_intent`,
`boundary_core_assessment`, `judgment_note`, `action_specs`,
`action_directives`, `surface_outputs`, `action_results`, `episode_trace`, and
`final_dialog` when a text surface exists.

## Running Example

The examples below use one non-private, public-knowledge style turn. The user
asks for a short explanation of a Python virtual environment:

```json
{
  "user_input": "Kazusa, can you explain what a Python virtual environment is in one short paragraph?",
  "decontexualized_input": "The user asks the active character to explain what a Python virtual environment is in one short paragraph.",
  "rag_result": {
    "answer": "A Python virtual environment is an isolated project environment for Python packages and interpreter settings.",
    "memory_evidence": [],
    "conversation_evidence": [],
    "external_evidence": [
      {
        "summary": "Python virtual environments isolate project dependencies so different projects can use different package versions."
      }
    ],
    "recall_evidence": []
  },
  "conversation_progress": {
    "current_thread": "short technical explanation",
    "next_affordances": ["answer directly", "keep the explanation concise"]
  }
}
```

The values are illustrative. They are not copied from a private conversation,
and they do not imply a required wording for production outputs.

When RAG returns memory, recall, or conversation evidence, those fields are
already cognition-ready. They use conclusion-first evidence blocks with
configured local timestamps and explicit uncertainty. Raw message ids, raw
storage ids, CQ wire text, attachment URLs, embeddings, and source rows remain
outside the public evidence fields and may appear only in trace/debug surfaces.

## L1: Subconscious

File: `cognition_chain_core/stages/l1.py`

L1 is the fast affective layer. It receives the current stimulus plus the
character's immediate mood, global vibe, relationship insight for the user,
and MBTI-derived instinct hint. L1 no longer receives `reflection_summary` as
live carry-over; internal monologue residue is projected only into L2a.

It outputs:

```python
{
    "emotional_appraisal": str,
    "interaction_subtext": str,
}
```

Design role:

- capture the first emotional impulse before logic or social etiquette,
- identify the social smell of the message, such as pressure, teasing,
  attention seeking, command tone, or neutral routine contact,
- avoid deciding what to do,
- avoid inventing threat or intimacy when the message is ordinary,
- treat media observations as visible facts, not as proof of user intent.

This is the "deep subconscious" entry point in the live response chain. It
does not reconstruct the whole answer. It gives later layers the affective
starting condition: what the stimulus felt like before the character decided
what was reasonable.

Example input:

```json
{
  "user_input": "Kazusa, can you explain what a Python virtual environment is in one short paragraph?",
  "indirect_speech_context": "",
  "media_observations": {
    "image_observations": [],
    "audio_observations": []
  }
}
```

Example output:

```json
{
  "emotional_appraisal": "Calm; this is a normal request.",
  "interaction_subtext": "routine information request"
}
```

Interpretation: L1 detects no pressure, flirtation, threat, or command
conflict. That neutral affect becomes a low-friction starting condition for
L2.

## L2a: Consciousness

File: `cognition_chain_core/stages/l2.py`

L2a is the rational interpretation layer. It reads L1, decontextualized input,
RAG evidence, user memory context, current commitments, promoted reflection
context, conversation progress, projected internal monologue residue, affinity,
mood, and global vibe.

When the current turn is structurally anchored to a prior Kazusa-authored
dialog by reply/quote context or conversation-evidence source refs, L2a may
also receive `past_dialog_cognition_context`. That context is private
trace-backed cognition residual for the anchored past dialog. It is weaker
than current visible input and public evidence, omitted when unavailable, and
must not be consumed by L1, L2b, L2c1, L2c2, L2d, L3, dialog, consolidation,
scheduler, reflection, adapters, or public RAG projection.

On the first resolver cycle, L2a is also the join point for the bounded
shared-memory prewarm result. If that prewarm finds confirmed shared
`memory` rows, L2a sees them as normal `rag_result.memory_evidence`; if it
fails or returns no usable evidence, L2a receives the same base `rag_result`
shape as before. L2b continues independently of this join.

It outputs a candidate:

```python
{
    "internal_monologue": str,
    "character_intent": str,
    "logical_stance": str,
}
```

`logical_stance` is selected from the response-decision vocabulary:

```text
CONFIRM | REFUSE | TENTATIVE | DIVERGE | CHALLENGE
```

`character_intent` is selected from the action-intent vocabulary:

```text
PROVIDE | BANTAR | REJECT | EVADE | CONFRONT | DISMISS | CLARIFY
```

Design role:

- turn affect plus evidence into a coherent first-person interpretation,
- prioritize evidence that is directly relevant to the current input,
- treat unrelated memories as background rather than decision proof,
- decide the broad stance and intended response action,
- use current media observations only as current-turn evidence,
- treat active commitments as current reality when the turn is about them,
- use promoted reflection context only as soft global background.

This is where consciousness is first generated as an inspectable artifact.
Emotion affects the thought here by biasing the internal monologue and stance,
but it is not allowed to override factual evidence or make ordinary input
hostile by itself.

Example input:

```json
{
  "emotional_appraisal": "Calm; this is a normal request.",
  "interaction_subtext": "routine information request",
  "decontextualized_input": "The user asks the active character to explain what a Python virtual environment is in one short paragraph.",
  "rag_result": {
    "answer": "A Python virtual environment is an isolated project environment for Python packages and interpreter settings.",
    "external_evidence": [
      {
        "summary": "Virtual environments isolate project dependencies so projects can use different package versions."
      }
    ]
  },
  "user_memory_context": {
    "active_commitments": [],
    "objective_facts": []
  },
  "conversation_progress": {
    "next_affordances": ["answer directly", "keep the explanation concise"]
  }
}
```

Example output:

```json
{
  "internal_monologue": "This is a straightforward technical question. I should answer directly, keep it compact, and not add unrelated setup steps.",
  "logical_stance": "CONFIRM",
  "character_intent": "PROVIDE"
}
```

Interpretation: the thought artifact shows the conscious decision: answer the
question, use the retrieved fact, and respect the requested length.

## L2b: Boundary Core

File: `cognition_chain_core/stages/l2.py`

L2b runs in parallel after L1 and before final judgment. It reads the
decontextualized input, reason to respond, channel topic, indirect speech
context, L1 affect, affinity, and the character's `boundary_profile`.

It outputs:

```python
{
    "boundary_core_assessment": {
        "boundary_issue": str,
        "boundary_summary": str,
        "behavior_primary": str,
        "behavior_secondary": str,
        "acceptance": str,
        "stance_bias": str,
        "identity_policy": str,
        "pressure_policy": str,
        "trajectory": str,
    }
}
```

Design role:

- decide whether the input touches autonomy, identity, control, authority, or
  relationship distortion,
- translate personality-level boundary parameters into response constraints,
- account for affinity without letting intimacy erase identity boundaries,
- produce a maximum safe response envelope for the judgment core.

Boundary Core is deliberately separate from Consciousness. L2a may be moved by
relationship, mood, or evidence; L2b asks what the character can accept without
losing self-integrity.

Example input:

```json
{
  "decontextualized_input": "The user asks the active character to explain what a Python virtual environment is in one short paragraph.",
  "reason_to_respond": "directly addressed technical question",
  "channel_topic": "Python tooling",
  "indirect_speech_context": "",
  "interaction_subtext": "routine information request",
  "emotional_appraisal": "Calm; this is a normal request.",
  "affinity_context": {
    "level": "friendly neutral",
    "instruction": "respond normally while keeping boundaries intact"
  }
}
```

Example output:

```json
{
  "boundary_core_assessment": {
    "boundary_issue": "none",
    "boundary_summary": "The request asks for a brief explanation and does not pressure identity, autonomy, or authority.",
    "behavior_primary": "comply",
    "behavior_secondary": "none",
    "acceptance": "allow",
    "stance_bias": "confirm",
    "identity_policy": "accept",
    "pressure_policy": "absorb",
    "trajectory": "Safe routine explanation; no boundary repair needed."
  }
}
```

Interpretation: Boundary Core permits the L2a candidate. It does not create a
new answer; it defines the safe envelope for the answer.

## L2c: Judgment Core

File: `cognition_chain_core/stages/l2.py`

L2c merges the Consciousness candidate and Boundary Core assessment. It also
reads referent-resolution status.

It outputs:

```python
{
    "logical_stance": str,
    "character_intent": str,
    "judgment_note": str,
}
```

Design role:

- produce the final L2 stance and intent,
- make Boundary Core the hard constraint,
- preserve L2a where safe,
- force `CLARIFY` when required referents are unresolved,
- return the result to a socially plausible human response rather than a raw
  reflex.

This is the point where the system can say consciousness has settled into a
decision. L1 affect remains visible through `emotional_appraisal`; L2a thought
remains visible through `internal_monologue`; L2c1 is the explicit final
decision.

Example input:

```json
{
  "referents": [],
  "internal_monologue_candidate": "This is a straightforward technical question. I should answer directly, keep it compact, and not add unrelated setup steps.",
  "logical_stance_candidate": "CONFIRM",
  "character_intent_candidate": "PROVIDE",
  "boundary_core_assessment": {
    "boundary_issue": "none",
    "acceptance": "allow",
    "stance_bias": "confirm",
    "identity_policy": "accept",
    "pressure_policy": "absorb"
  }
}
```

Example output:

```json
{
  "logical_stance": "CONFIRM",
  "character_intent": "PROVIDE",
  "judgment_note": "Boundary Core allows the routine explanation, so the final decision is to answer directly and concisely."
}
```

Interpretation: final stance and intent are now settled. Later layers may
shape wording, but they must not convert this into refusal, evasion, or a new
topic.

## L2c2: Social Context Appraisal

File: `cognition_chain_core/stages/l2c2.py`

L2c2 is the social temperature layer. It keeps the existing contextual-agent
responsibility and output fields, but it now runs before L2d so action
selection can see social distance and relationship framing before choosing
zero, one, or many actions.

It outputs:

```python
{
    "social_distance": str,
    "emotional_intensity": str,
    "vibe_check": str,
    "relational_dynamic": str,
}
```

L2c2 does not decide whether to answer, select an action, or generate visible
wording. L2d consumes these fields for action selection; selected L3 surface
handlers consume the same fields later for expression packaging.

## L3: Expression Directives

File: `cognition_chain_core/stages/l3.py`

L3 does not change the L2 decision or choose actions. It turns selected
surface actions into presentation constraints for dialog or future surface
handlers.

The L3 branches are:

| Node | Output | Purpose |
| --- | --- | --- |
| `l3_interaction_style_context_loader` | `interaction_style_context` | Sanitized user/channel interaction-style overlays from storage. |
| `l3_style_agent` | `rhetorical_strategy`, `linguistic_style`, `forbidden_phrases` | How to package the already-decided stance in character voice. |
| `l3_content_plan_agent` | `content_plan` | One resolved semantic plan for what dialog should render. |
| `l3_preference_adapter` | `accepted_user_preferences` | User expression preferences that the character has accepted and can execute. |
| `l3_visual_agent` | `facial_expression`, `body_language`, `gaze_direction`, `visual_vibe` | Optional still-frame visual directives and promptable guidance for image generation surfaces. |

L3 is where a selected surface action becomes an executable communication
plan. The most important text artifact is `content_plan`. Dialog renders this
single resolved plan, so L3 must include the visible facts, conclusions, code,
and scope before dialog runs.

L3 does not decide whether to create a visible text surface. That decision
belongs to the selected `speak` action from L2d. If no visible surface is selected,
the persona graph does not run L3 text or dialog.

Example input:

```json
{
  "internal_monologue": "This is a straightforward technical question. I should answer directly, keep it compact, and not add unrelated setup steps.",
  "logical_stance": "CONFIRM",
  "character_intent": "PROVIDE",
  "judgment_note": "Boundary Core allows the routine explanation, so the final decision is to answer directly and concisely.",
  "decontextualized_input": "The user asks the active character to explain what a Python virtual environment is in one short paragraph.",
  "rag_result": {
    "answer": "A Python virtual environment is an isolated project environment for Python packages and interpreter settings."
  },
  "conversation_progress": {
    "next_affordances": ["answer directly", "keep the explanation concise"]
  }
}
```

Example output:

Shown as the combined L3 surface before L4 collection.

```json
{
  "social_distance": "Friendly and task-focused.",
  "emotional_intensity": "Low and steady.",
  "vibe_check": "Practical technical explanation.",
  "relational_dynamic": "The user asks for a compact explanation; the character can answer without extra probing.",
  "rhetorical_strategy": "Give a direct definition, then one concrete reason it matters.",
  "linguistic_style": "Concise, plain, lightly conversational.",
  "forbidden_phrases": [],
  "content_plan": {
    "visible_goal": "Answer the user's question directly and concisely.",
    "semantic_content": "A Python virtual environment is an isolated workspace for a project, so its packages and settings do not interfere with other projects.",
    "voice": "Concise, plain, lightly conversational.",
    "rendering": "One visible chat bubble; one short paragraph; no setup tutorial."
  },
  "accepted_user_preferences": [
    "Keep the reply to one short paragraph."
  ],
  "facial_expression": [
    "Neutral attentive expression with relaxed brows."
  ],
  "body_language": [
    "Still, composed posture facing the conversation."
  ],
  "gaze_direction": [
    "Looking toward the conversation partner."
  ],
  "visual_vibe": [
    "Simple chat-space framing with calm, practical focus."
  ]
}
```

Interpretation: L3 has turned the decision into a communication plan. The
dialog generator can now write the answer without re-deciding whether to
answer.

## L4: Collector

File: `cognition_chain_core/stages/l3.py`

L4 collects L3 outputs into `action_directives`:

```python
{
    "action_directives": {
        "contextual_directives": {
            "social_distance": str,
            "emotional_intensity": str,
            "vibe_check": str,
            "relational_dynamic": str,
        },
        "linguistic_directives": {
            "rhetorical_strategy": str,
            "linguistic_style": str,
            "accepted_user_preferences": list[str],
            "content_plan": dict[str, str],
            "forbidden_phrases": list[str],
        },
        "visual_directives": {
            "facial_expression": list[str],
            "body_language": list[str],
            "gaze_direction": list[str],
            "visual_vibe": list[str],
        },
    }
}
```

`action_directives` is the contract between cognition and dialog. Dialog should
not reinterpret the user request, choose a different stance, or invent new
facts. It should render these directives.

Example input:

```json
{
  "social_distance": "Friendly and task-focused.",
  "emotional_intensity": "Low and steady.",
  "vibe_check": "Practical technical explanation.",
  "relational_dynamic": "The user asks for a compact explanation; the character can answer without extra probing.",
  "rhetorical_strategy": "Give a direct definition, then one concrete reason it matters.",
  "linguistic_style": "Concise, plain, lightly conversational.",
  "accepted_user_preferences": ["Keep the reply to one short paragraph."],
  "content_plan": {
    "visible_goal": "Answer the user's question directly and concisely.",
    "semantic_content": "A Python virtual environment is an isolated workspace for a project, so its packages and settings do not interfere with other projects.",
    "voice": "Concise, plain, lightly conversational.",
    "rendering": "One visible chat bubble; one short paragraph; no setup tutorial."
  },
  "forbidden_phrases": [],
  "facial_expression": ["Neutral attentive expression with relaxed brows."],
  "body_language": ["Still, composed posture facing the conversation."],
  "gaze_direction": ["Looking toward the conversation partner."],
  "visual_vibe": ["Simple chat-space framing with calm, practical focus."]
}
```

Example output:

```json
{
  "action_directives": {
    "contextual_directives": {
      "social_distance": "Friendly and task-focused.",
      "emotional_intensity": "Low and steady.",
      "vibe_check": "Practical technical explanation.",
      "relational_dynamic": "The user asks for a compact explanation; the character can answer without extra probing."
    },
    "linguistic_directives": {
      "rhetorical_strategy": "Give a direct definition, then one concrete reason it matters.",
      "linguistic_style": "Concise, plain, lightly conversational.",
      "accepted_user_preferences": ["Keep the reply to one short paragraph."],
      "content_plan": {
        "visible_goal": "Answer the user's question directly and concisely.",
        "semantic_content": "A Python virtual environment is an isolated workspace for a project, so its packages and settings do not interfere with other projects.",
        "voice": "Concise, plain, lightly conversational.",
        "rendering": "One visible chat bubble; one short paragraph; no setup tutorial."
      },
      "forbidden_phrases": []
    },
    "visual_directives": {
      "facial_expression": ["Neutral attentive expression with relaxed brows."],
      "body_language": ["Still, composed posture facing the conversation."],
      "gaze_direction": ["Looking toward the conversation partner."],
      "visual_vibe": ["Simple chat-space framing with calm, practical focus."]
    }
  }
}
```

Interpretation: L4 is a collection boundary. It performs no new semantic
judgment; it packages L3 outputs into the exact shape dialog consumes.

## Emotion-To-Thought-To-Dialog Flow

The implemented flow is:

```text
1. User stimulus
   Message body, reply metadata, mentions, attachments, channel context,
   current user profile, character profile, time context, and recent history.

2. Perception cleanup
   Media descriptor turns images/audio into bounded observations.
   Relevance decides whether a reply should happen.
   Decontextualizer rewrites only context-dependent references.

3. Cognition resolver
   Each cycle runs L1 -> L2 -> L2d. When L2d selects evidence, the resolver
   calls RAG 2 or a web/current evidence capability and projects the
   observation into the next cognition cycle. Evidence stays evidence; it does
   not speak as the character.

4. L1 affect
   `emotional_appraisal` and `interaction_subtext` capture immediate
   emotional pressure, attraction, irritation, neutrality, or uncertainty.

5. L2 consciousness
   `internal_monologue` interprets affect plus evidence.
   `logical_stance` and `character_intent` represent the candidate decision.

6. L2 boundary judgment
   `boundary_core_assessment` constrains what the character can accept.
   Judgment Core emits final `logical_stance`, `character_intent`,
   and `judgment_note`.

7. L2c2 social context, L2d action selection, and L3 expression plan
   L2c2 produces social temperature and relationship framing before action
   selection.
   L2d selects zero or more semantic actions. When a text surface is selected,
   style, content-plan, preference, and visual agents build
   `action_directives` using the L2c2 social fields.

8. Dialog or selected surface generation
   Dialog generator writes `final_dialog` from `internal_monologue` and
   `action_directives`.

9. Episode trace and post-turn consolidation
   Background nodes use the completed episode trace to update durable memory,
   relationship state, character state, images, and scheduled promises.
```

Emotion affects thought at three points:

- L1 creates the first affective appraisal.
- L2a reads that appraisal while forming the internal monologue and stance.
- L2c2 uses the settled emotional temperature to shape social distance and
  relationship framing before action selection.
- L3 style and content agents use the settled decision plus L2c2 social fields
  to package a selected surface.

Emotion does not directly write dialog. It must pass through the L2 decision
and L3 directive contracts before visible text is generated.

## Dialog Contract

File: `dialog_agent.py`

Dialog has one node:

```text
generator -> final_dialog
```

The generator receives:

- `internal_monologue`,
- `action_directives.linguistic_directives`,
- `action_directives.contextual_directives`,
- user display name,
- immutable character voice constraints from `linguistic_texture_profile`.

The dialog layer must not:

- change `logical_stance`,
- accept or reject a user request on its own,
- invent facts not authorized by `content_plan`,
- expose physical actions, facial expressions, or hidden internal monologue in
  a text chat response,
- turn visual directives into chat text.

Example input:

```json
{
  "internal_monologue": "This is a straightforward technical question. I should answer directly, keep it compact, and not add unrelated setup steps.",
  "action_directives": {
    "contextual_directives": {
      "social_distance": "Friendly and task-focused.",
      "emotional_intensity": "Low and steady.",
      "vibe_check": "Practical technical explanation.",
      "relational_dynamic": "The user asks for a compact explanation; the character can answer without extra probing."
    },
    "linguistic_directives": {
      "rhetorical_strategy": "Give a direct definition, then one concrete reason it matters.",
      "linguistic_style": "Concise, plain, lightly conversational.",
      "accepted_user_preferences": ["Keep the reply to one short paragraph."],
      "content_plan": {
        "visible_goal": "Answer the user's question directly and concisely.",
        "semantic_content": "A Python virtual environment is an isolated workspace for a project, so its packages and settings do not interfere with other projects.",
        "voice": "Concise, plain, lightly conversational.",
        "rendering": "One visible chat bubble; one short paragraph; no setup tutorial."
      },
      "forbidden_phrases": []
    }
  },
  "tone_history": [],
  "user_name": "Alex"
}
```

Example output:

```json
{
  "final_dialog": [
    "A Python virtual environment is a small isolated workspace for one project, so the packages you install there do not interfere with other Python projects on the same machine."
  ]
}
```

Interpretation: dialog has produced visible wording from the content plan. It did
not mention facial expression, expose the internal monologue, add setup steps,
or change the decision.

## Consolidation Boundary

Consolidation runs after selected visible surfaces and private action results
are available. It consumes the completed persona state, including:

- `decontexualized_input`,
- `rag_result`,
- `internal_monologue`,
- `emotional_appraisal`,
- `interaction_subtext`,
- `logical_stance`,
- `character_intent`,
- `action_directives`,
- `final_dialog`,
- `action_specs`,
- `action_results`,
- `surface_outputs`,
- prompt-safe `episode_trace_projection`.

Consolidation treats these fields with different evidence strength. The final
dialog, selected action results, and explicit user facts are stronger evidence
than raw affect. Internal monologue and emotional appraisal can explain
subjective reaction, but they must not become durable user facts by themselves.

That rule prevents self-generated dialog, momentary affect, or private action
metadata from polluting long-term memory.

## Design Invariants

Keep these invariants when changing the node package:

- Relevance decides whether to enter the persona turn; L2d may still select no
  visible surface. L3 and dialog must not carry a second response gate.
- Decontextualization rewrites references; it must not answer the user or infer
  deep motive.
- RAG returns evidence; cognition owns interpretation.
- L1 may produce affect, but it must not decide acceptance or refusal.
- L2a may propose stance, but Boundary Core constrains the safe envelope.
- Judgment Core is the final owner of `logical_stance` and
  `character_intent`.
- L2c2 may shape social-context evidence, but it must not select actions or
  create visible output.
- L3 may shape expression, but it must not change the L2 decision.
- L2d is the only semantic action selection; L2a, L2b, L2c1, and L2c2 must not emit
  action specs.
- L3 text/image handlers run only for selected surface actions.
- Dialog renders directives; it must not make policy, memory, or permission
  decisions.
- Consolidation writes durable state only after selected surfaces, action
  results, or private finalization make the episode consolidatable.
- Consolidation consumes prompt-safe episode-trace evidence; it must not
  execute actions, dispatch, schedule, or trigger cognition.
- Reflection and self-cognition worker runs reuse the shared cognition graph, but
  raw reflection output and private thought residue do not automatically enter
  normal chat.

## Local-LLM Rationale

The cognition stack is intentionally decomposed for local or weaker models.
One prompt that retrieves facts, interprets emotion, enforces boundaries,
chooses a stance, writes style constraints, and generates final text is hard to
audit and brittle under latency pressure.

This package instead uses small stage contracts:

- each LLM call owns one semantic job,
- deterministic code validates required fields,
- graph edges make dependency order explicit,
- prompt inputs are shaped as semantic descriptors instead of raw database
  internals,
- the normal path stays bounded and inspectable.

The result is not a hidden, mystical mind. It is a structured character
cognition pipeline whose intermediate artifacts can be inspected, tested, and
used by background memory systems without collapsing every concern into the
final reply.
