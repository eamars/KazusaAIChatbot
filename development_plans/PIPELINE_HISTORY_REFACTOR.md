# Pipeline Conversation History & Stage Decoupling Refactor

## Problem Statement

The pipeline uses a single fixed `chat_history` depth fed uniformly to every stage. This exhibits two compounding issues:

1. **Relevance accuracy degrades** with insufficient history — cannot detect conversation continuation, generates vague `channel_topic`/`user_topic`
2. **Downstream pollution** — raw `user_input` and full `chat_history` bleed through every stage, causing each LLM to re-interpret content that prior stages already abstracted

Additionally, `user_topic` conflates two distinct purposes: a general paraphrase of `user_input` (redundant) and a semantic flag for indirect speech (essential). The linguistic agent conflates style decisions with content generation, causing local LLMs to underspecify `content_anchors` — particularly missing actual answers to user questions.

---

## Design Principles

- Each stage consumes the **outputs** of the prior stage, not the original raw inputs
- `user_input` and `chat_history` are progressively abstracted; downstream stages receive distillations, not raw data
- Every stage has a single, well-defined input contract with no incidental data

---

## Changes

### 1. Tiered `chat_history` Depth

Replace the single fixed history limit with two slices loaded before the pipeline starts:

| Slice                 | Depth        | Used By                                   |
| --------------------- | ------------ | ----------------------------------------- |
| `chat_history_wide`   | 10 messages  | Relevance Agent only                      |
| `chat_history_recent` | 3–5 messages | Decontextualizer, L3 agents, Dialog Agent |

The `GlobalPersonaState` carries both. Stages explicitly declare which slice they consume. 

chat_history_recent can be generated based on chat_history_wide without needing to query the database. chat_history_wide should be read from the config, while chat_history_recent shoudl be hard coded based on LLM performance. 

---

### 2. Split `user_topic` into Targeted Fields

Remove the generic `user_topic` paraphrase. Replace with:

- **`indirect_speech_context`** — only populated for Situation B (user is speaking *to others* about the character, not addressing the character directly). Null/empty for all direct-address cases.
- **`channel_topic`** — unchanged; represents the broader channel discourse topic.

**Relevance agent prompt change:** remove the generic Situation A `user_topic` generation rule. Only populate `indirect_speech_context` when the pronoun/addressee analysis confirms Situation B.

**Downstream consumption:**

- `indirect_speech_context` is passed to L1 Subconscious and L2 Consciousness only when non-null, providing the framing "this message is about the character, not to them"
- `channel_topic` continues to flow to decontextualizer and L3 Contextual Agent as supplementary background

---

### 3. L1 Subconscious: Keep Raw `user_input`

L1 receives **raw `user_input`** by design. The subconscious reacts to surface form before cognitive processing — this correctly mimics the limbic system's pre-rational stimulus response. `indirect_speech_context` is added when non-null.

**L1 input contract:**

```
user_input                  (raw — intentional)
indirect_speech_context     (Situation B framing, nullable)
character_profile           (mood, global_vibe, MBTI, last_relationship_insight)
```

---

### 4. L2 Consciousness: Consume `decontextualized_input`

L2 and all subsequent cognition stages receive `decontextualized_input` (the pronoun-resolved, context-anchored canonical form), not `user_input`.

**Remove from L2 input:** `user_input`

---

### 5. Split L3 Linguistic Agent into Style Agent + Content Anchor Agent

**Root cause:** the linguistic agent conflates planning (identifying what anchors are needed) with generation (producing the actual answer/content for each anchor). Local LLMs fail at the combined task, producing directive-only anchors like `[ANSWER] 回答用户的问题` without the answer itself.

#### 5a. Style Agent (replaces current linguistic agent's style concerns)

**Responsibility:** HOW the character speaks

**Inputs:**

```
social_distance             (from L3 Contextual)
emotional_intensity         (from L3 Contextual)
expression_willingness      (from L3 Contextual)
relational_dynamic          (from L3 Contextual)
affinity_block              (relationship level + instruction)
character linguistic texture (hesitation, fragmentation, etc.)
chat_history_recent         (speech pattern detection)
channel_topic               (register calibration)
```

**Outputs:** `rhetorical_strategy`, `linguistic_style`, `forbidden_phrases`, `accepted_user_preferences`

#### 5b. Content Anchor Agent (new)

**Responsibility:** WHAT the character says — fully resolved content

**Inputs:**

```
decontextualized_input      (what was asked/said)
research_facts              (from RAG stage)
logical_stance              (from L2 judgment — hard constraint)
character_intent            (from L2 consciousness)
internal_monologue          (from L2 — emotional framing)
```

**Outputs:** `content_anchors` with fully resolved content per anchor type:

- `[ANSWER]` — actual answer to the user's question, not a directive to answer it
- `[FACT]` — specific fact to reference
- `[DECISION]` — character's concrete decision, derived from `logical_stance`
- `[SOCIAL]` — specific social gesture or reference
- `[EMOTION]` — concrete emotional expression point
- `[SCOPE]` — word count range + anchors that must be covered

**Constraint:** `logical_stance` from L2 is treated as a hard constraint. The content anchor agent generates *within* the stance, not independently.

**Parallelism:** Style Agent and Content Anchor Agent run in parallel (both depend on L3 Contextual outputs and L2 outputs respectively, not on each other).

---

### 6. Dialog Agent: Remove `user_input` and `decontextualized_input`

With fully resolved `content_anchors`, the dialog agent is a **pure renderer**. It has no need to re-read what the user said.

**Dialog Agent input contract:**

```
action_directives           (assembled by L4 Collector from Style + Content + Contextual outputs)
character_profile           (immutable voice constraints: linguistic texture, taboos, tempo, defense)
chat_history_recent         (3–5 turns: prevents repetition, maintains conversational rhythm)
```

**Remove:** `user_input`, `decontextualized_input`, `user_topic`, `channel_topic`

---

## Final Pipeline Data Flow

```
user_input + chat_history_wide
    ↓
[Relevance Agent]
    produces: should_respond, channel_topic, indirect_speech_context,
              use_reply_feature, reason_to_respond
    ↓
[Msg Decontextualizer]
    consumes: user_input, chat_history_recent, channel_topic, indirect_speech_context
    produces: decontextualized_input
    ↓
    ┌─────────────────────────────────────────────┐
    │ [L1 Subconscious]                           │
    │   consumes: user_input (raw), indirect_speech_context
    │   produces: emotional_appraisal, interaction_subtext
    └─────────────────────────────────────────────┘
    ↓
    ┌─────────────────────────────────────────────────────────┐
    │ [L2a Consciousness] ──────────────────────────────────  │
    │   consumes: decontextualized_input, L1 outputs          │
    │   produces: internal_monologue, character_intent        │
    │                                                         │
    │ [L2b Boundary Core] ──────────────────────────────────  │
    │   consumes: decontextualized_input, character_profile   │
    │   produces: boundary_core_assessment                    │
    └─────────────────────────────────────────────────────────┘
    ↓
    [L2c Judgment Core]
        consumes: L2a + L2b outputs
        produces: logical_stance, judgment_note
    ↓
    ┌──────────────────────────────────────────────────────────────────┐
    │ [L3 Contextual Agent]     [L3 Style Agent]  [L3 Content Anchor] │
    │   consumes:                consumes:          consumes:          │
    │     channel_topic,           L3 Contextual,    decontextualized, │
    │     affinity,                affinity,          research_facts,  │
    │     chat_history_recent      chat_history_recent  logical_stance │
    │   produces:                produces:          produces:          │
    │     social_distance,         rhetorical_strategy, content_anchors│
    │     emotional_intensity,     linguistic_style,    (fully resolved)│
    │     vibe_check,              forbidden_phrases,                  │
    │     relational_dynamic,      accepted_user_prefs                 │
    │     expression_willingness                                       │
    └──────────────────────────────────────────────────────────────────┘
    ↓
    [L4 Collector]
        consumes: all L3 outputs
        produces: action_directives
    ↓
    [Dialog Agent]
        consumes: action_directives, character_profile, chat_history_recent
        produces: final_dialog
```

---

## Implementation Order

1. **Tiered history slicing** — load `chat_history_wide` and `chat_history_recent` at pipeline entry; update `GlobalPersonaState` schema; wire each stage to correct slice. Low risk, immediate quality improvement for relevance.

2. **Split `user_topic` → `indirect_speech_context`** — update relevance agent prompt and output schema; update all downstream consumers to use new field name; remove redundant Situation A paraphrase generation.

3. **Remove `user_input` from L2+ cognition stages** — swap `user_input` for `decontextualized_input` in L2a, L2b, L2c input structs. Verify L1 still receives raw `user_input`.

4. **Create Content Anchor Agent** — new agent consuming `decontextualized_input` + `research_facts` + L2 outputs; runs in parallel with Style Agent in L3 layer; L4 Collector assembles both outputs.

5. **Refactor Style Agent** — strip content anchor responsibility from current linguistic agent; rename and trim its input contract.

6. **Remove `user_input`/`decontextualized_input` from Dialog Agent** — verify `action_directives` quality is sufficient before cutting; run parallel A/B if uncertain.

7. Use real LLM tests to verify the workflow. Recommended to remove all unit tests that doesn't invoke LLM, and use LLM tests as default unit test to verify the most important factor: the prompt quality and effect. 

---

## Risks

- **Content Anchor Agent quality gate**: if `content_anchors` are underspecified, dialog agent produces hollow responses. Validate anchor completeness before removing `decontextualized_input` from dialog (step 6 depends on step 4 being stable).
- **`indirect_speech_context` false negatives**: if relevance fails to detect Situation B, L1 will appraise the message as direct address and the character may respond as if addressed. Monitor relevance agent Situation B accuracy after prompt change.
- **L3 parallelism**: Style Agent depends on L3 Contextual outputs (`social_distance`, `expression_willingness`). Content Anchor Agent depends on L2 outputs. Both can run in parallel with each other but Style Agent must wait for L3 Contextual. Verify graph wiring in `call_cognition_subgraph`.
