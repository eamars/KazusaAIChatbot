> Superseded Architecture Document
>
> Status: superseded
> Superseded by plan: development_plans/active/bugfix/self_cognition_speak_delivery_bugfix_plan.md
> Canonical current doc: src/kazusa_ai_chatbot/self_cognition/README.md
> Supersession rule: private-candidate-only and no-production-delivery claims
> in this document are no longer architecture authority. Current production
> self-cognition selected `speak` must resolve a target before cognition and
> attempt delivery after dialog rendering.

# self cognition loop architecture

## Summary

- Goal: Define self-cognition as an idle-only entry path into the existing
  source-aware cognition core, so the character's current L1/L2/L3 cognition
  layers perform the private dry run instead of a separate prompt pretending to
  be the character.
- Plan class: architectural reference
- Status: reference
- Execution rule: this document is not an implementation contract. Promote a
  bounded slice into `development_plans/active/short_term/` before any
  production work.
- Mandatory skills for future executable plans: `development-plan-writing`,
  `local-llm-architecture`, `no-prepost-user-input`, `py-style`,
  `test-style-and-execution`, `database-data-pull` when using real data, and
  `cjk-safety` when editing Python files that contain CJK prompt text.
- Highest-risk areas: bypassing the production cognition core, treating
  private thought as a fake user message, generated thought becoming durable
  evidence, private leakage, unbounded background loops, permissionless
  proactive behavior, and live response-path latency.

## Foundation

This reference builds on the completed multi-source cognition architecture
record:

`development_plans/archive/completed/short_term/multi_source_cognition_architecture_plan.md`

That plan's core direction is the controlling constraint:

```text
trigger source
-> source-specific episode builder
-> shared input-source/percept normalization
-> shared context stack planner
-> shared RAG supervisor
-> shared L1/L2/L3 cognition nodes
-> shared action/output router
-> origin-aware persistence, audit, delivery, or no-op
```

For self-cognition, the important lesson is not merely that another background
LLM can summarize conversation. The lesson is that a new source should wake the
same cognitive process. A valid self-cognition design therefore enters through
`CognitiveEpisode` and `GlobalPersonaState`, then calls the production
`call_cognition_subgraph` path.

The current production source closest to self-cognition is
`trigger_source="internal_thought"` with
`input_sources=["internal_monologue"]` and `output_mode="think_only"`. A future
production plan may add a dedicated `self_cognition` trigger or prompt variant,
but the first proof of concept should reuse the existing internal-thought
variant so it exercises the real shared cognition layers.

POC interpretation rule: if the existing internal-thought variant frames the
source packet as something the character is reading or being shown, that is
evidence for a future source-specific prompt variant. It is not evidence to
return to a standalone self-cognition prompt.

## Explicit Goal

The self-cognition loop should answer this question:

> If the character privately reprocesses a completed interaction while idle,
> how do the existing cognition layers reinterpret the recent evidence, current
> relationship context, and private state, and what should that teach future
> growth or continuity systems?

The goal is not to generate final dialog, mutate personality, write user
memory, or produce a separate reflection summary. The immediate goal is to
obtain the real cognition-core output:

- `emotional_appraisal`
- `interaction_subtext`
- `internal_monologue`
- `logical_stance`
- `character_intent`
- `action_directives`

Those fields are the proof that the character's current cognition layers, not
a standalone self-reflection prompt, processed the self-cognition input.

## Architectural Position

Self-cognition is a background cognition source.

```text
completed interaction evidence
  + current character profile/runtime state
  + current user profile and memory evidence
  + conversation progress
  + optional private residue/action latch
-> self-cognition source packet
-> CognitiveEpisode(trigger_source=internal_thought,
                   input_sources=[internal_monologue],
                   output_mode=think_only)
-> GlobalPersonaState dry-run input
-> production call_cognition_subgraph
-> actual L1/L2/L3 cognition output
-> audit-only artifact
-> later promotion or growth systems
```

The source packet is input evidence, not a replacement prompt. It may describe
what the dry run is inspecting, but it must not define a new output schema for
the character. The production cognition nodes own the character's semantic
judgment and output shape.

## Input Contract

The dry-run input has two layers.

First, the self-cognition source packet is a compact, prompt-safe evidence
object embedded as private internal monologue residue:

```python
{
    "run_kind": "post_turn_micro_self_cognition",
    "goal": "reprocess_recent_interaction_through_shared_cognition",
    "source_window": {
        "platform": str,
        "platform_channel_id": str,
        "channel_type": str,
        "global_user_id": str,
        "display_name": str,
        "conversation_row_ids": list[str],
        "last_timestamp": str,
    },
    "recent_messages": [
        {
            "role": "user|assistant",
            "speaker": str,
            "timestamp": str,
            "body_text": str,
        }
    ],
    "conversation_progress": dict,
    "current_inner_state": {
        "mood": str,
        "global_vibe": str,
        "reflection_summary": str,
        "self_image_snapshot": dict,
    },
    "memory_evidence": {
        "stable_patterns": list[dict],
        "recent_shifts": list[dict],
        "objective_facts": list[dict],
        "milestones": list[dict],
        "active_commitments": list[dict],
    },
    "dry_run_focus": [
        "what unresolved private residue should remain salient?",
        "what stance should future cognition preserve?",
        "what should not be reopened unless the user brings it up?"
    ],
}
```

Second, the cognition entry input is a normal `GlobalPersonaState` dry-run
state:

```python
{
    "character_profile": CharacterProfileDoc,
    "timestamp": str,
    "time_context": TimeContextDoc,
    "user_input": "Self-cognition dry run over private interaction residue.",
    "prompt_message_context": empty PromptMessageContext,
    "cognitive_episode": CognitiveEpisode(...),
    "platform": "qq",
    "platform_channel_id": str,
    "channel_type": str,
    "platform_user_id": str,
    "global_user_id": str,
    "user_name": str,
    "user_profile": UserProfileDoc,
    "platform_bot_id": str,
    "chat_history_wide": bounded recent typed history,
    "conversation_progress": prompt-facing episode state,
    "debug_modes": {"think_only": True, "no_remember": True},
    "should_respond": False,
    "decontexualized_input": "Self-cognition dry run ...",
    "rag_result": prompt-facing user memory context and empty retrieval lanes,
}
```

This is not a fake `/chat` call. The current user text is not inserted as
`user_input`; it is evidence inside the private source packet and recent
history.

## Output Contract

The required dry-run output is the production cognition output:

```python
{
    "internal_monologue": str,
    "action_directives": dict,
    "interaction_subtext": str,
    "emotional_appraisal": str,
    "character_intent": str,
    "logical_stance": str,
}
```

The audit artifact should also record:

- the source packet;
- the `CognitiveEpisode`;
- the prompt variant keys selected by the internal-thought source;
- the sanitized `GlobalPersonaState` input used for the dry run;
- the cognition output fields above.

Any later extraction of private residue, procedural learning candidates, or
future attention hints must be downstream of this cognition output. Those
candidate structures are not the self-cognition proof and must not replace the
shared cognition call.

## Relationship To Reflection And Character Growth

Reflection and self-cognition have different jobs:

- Reflection reviews completed interaction windows from a slow evidence-review
  posture and may promote bounded memory.
- Self-cognition re-enters the character's current cognition stack with private
  recent evidence and asks what the character's own stance should carry
  forward.
- Character-state evolution may later read accepted reflection or
  self-cognition artifacts, but it must remain a separate promotion path.

This keeps growth inspectable:

```text
self-cognition dry run
-> real cognition output
-> audit
-> optional growth/procedure candidate extraction
-> validation/review
-> future prompt-facing guidance or state evolution plan
```

## Deterministic Gates

Deterministic code must own:

- idle/busy gating;
- run frequency and loop caps;
- prompt input caps;
- `CognitiveEpisode` validation;
- source row ids and lineage;
- privacy and visibility labels;
- no-write guarantees;
- no-send guarantees;
- audit artifact writing.

The LLM cognition layers own semantic appraisal, stance, intent, and action
directive generation. Deterministic code must not keyword-classify user
requests into facts, preferences, permissions, commitments, or proactive
authorizations.

## Non-Goals

This architecture does not authorize:

- a standalone self-cognition character prompt;
- direct character-state writes;
- direct `memory` or `user_memory_units` writes;
- autonomous sends;
- scheduler or dispatcher integration;
- direct personality-profile mutation;
- recursive inner loops;
- live `/chat` LLM call increases;
- raw reflection output in normal cognition;
- treating generated thought as user evidence;
- treating recent user messages as fake current `/chat` input.

## Acceptance Criteria For This Reference

This reference is useful when it:

- explicitly states that self-cognition must reuse current L1/L2/L3 cognition;
- defines self-cognition as a `CognitiveEpisode`-driven background source;
- makes actual cognition output the required proof artifact;
- separates input evidence from model output schema;
- preserves live response-path latency;
- keeps downstream growth and autonomy candidates outside the first proof;
- keeps autonomous contact behind explicit future permission policy.
