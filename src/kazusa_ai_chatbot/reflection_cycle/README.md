# Reflection Cycle Interface Control Document

## Document Control

- ICD id: `RC-ICD-001`
- Owning package: `kazusa_ai_chatbot.reflection_cycle`
- Interface boundary: conversation-history reads -> read-only reflection
  artifact
- Runtime consumers: local evaluation CLI, future scheduler integration,
  future memory-evolution integration
- Upstream data owner: `kazusa_ai_chatbot.db.conversation_reflection`
- Downstream owners: human reviewers now; memory-evolution pipeline in a
  separate integration stage

This document defines the contract for the read-only reflection cycle. It is
the source of truth for what data the cycle may read, what it may send to the
reflection LLM, what it may write, and how other modules may call it.

## Purpose

The reflection cycle evaluates monitored conversations outside
the active user-response path. It is meant to answer:

- what topics the active character participated in,
- what participant behavior can be observed without exposing identity,
- how response quality can improve,
- what privacy risks would block future persistence,
- whether hourly outputs can be combined into a daily synthesis.

The current implementation is a read-only evaluation path. It does not update
memory, lore, character state, conversation progress, scheduled events, or
chat output.

## Scope

This ICD covers:

- Monitored channel selection.
- Bounded transcript retrieval through the reflection DB interface.
- Prompt-facing projection and prompt budgets.
- Hourly reflection and daily synthesis LLM contracts.
- Local artifact writing for manual review.
- Public module entry points.
- Dependency and import rules.

This ICD does not cover:

- Memory evolution or lore promotion.
- Autonomous message dispatch.
- Scheduler wiring.
- `/chat` request handling.
- Active cognition, dialog generation, or consolidation writes.
- Platform adapter parsing.

Those features must integrate through explicit future interfaces rather than
by importing internal reflection helpers directly.

## Boundary Summary

```text
conversation_history
  -> db.conversation_reflection read-only interfaces
  -> reflection_cycle.selector
       deterministic monitored-channel selection
  -> reflection_cycle.projection
       prompt-safe payload construction and budget trimming
  -> reflection_cycle.prompts
       active-hour reflection LLM and per-channel daily synthesis LLM
  -> reflection_cycle.runtime
       local JSON artifact only
```

The reflection cycle must stay off the active conversation path. The current
user-facing cognition loop takes priority over background reflection work.

## Public Entry Points

Consumers should use only the package exports:

```python
from kazusa_ai_chatbot.reflection_cycle import (
    collect_reflection_inputs,
    run_readonly_reflection_evaluation,
)
```

### `collect_reflection_inputs`

```python
async def collect_reflection_inputs(
    *,
    lookback_hours: int = 24,
    now: datetime | None = None,
) -> ReflectionInputSet
```

This function selects monitored channels and returns bounded message
inputs. It is read-only and must communicate with MongoDB only through
`kazusa_ai_chatbot.db.conversation_reflection`.

Monitored means:

```text
the latest assistant message in the channel is inside the 24-hour monitor window
```

This rule is evaluated by querying recent character messages, not by a counter,
state collection, or background tracking variable. If the requested monitor
window contains no matching channel, the selector may use the
bounded recent fallback window. The returned `ReflectionInputSet` must record:

- requested window,
- effective window,
- `fallback_used`,
- `fallback_reason`,
- selected scopes,
- query diagnostics.

### `run_readonly_reflection_evaluation`

```python
async def run_readonly_reflection_evaluation(
    *,
    lookback_hours: int = 24,
    now: datetime | None = None,
    output_dir: str,
    use_real_llm: bool,
) -> ReflectionEvaluationResult
```

This function collects inputs, splits selected channels into active UTC hour
buckets, builds or runs hourly reflections, builds or runs per-channel daily
synthesis, and writes one local artifact under `output_dir`.

The artifact is for inspection. It is not a database write contract.

## CLI Interface

The local evaluation CLI is:

```powershell
python -m scripts.run_reflection_cycle_readonly --lookback-hours 24 --output-dir test_artifacts\reflection_cycle_readonly
```

Use `--real-llm` only for manual approval runs. Real LLM outputs must be
inspected qualitatively; schema validity alone is not approval.

## Data Source Interface

The reflection cycle depends on this DB module:

```python
from kazusa_ai_chatbot.db.conversation_reflection import (
    explain_monitored_channel_query,
    list_recent_character_message_channels,
    list_reflection_scope_messages,
)
```

The selector must not call `get_db`, `.find(...)`, `.aggregate(...)`, or
`.command(...)` directly. Database commands belong in
`db.conversation_reflection`.

### Monitored Channel Rows

`list_recent_character_message_channels(...)` returns aggregate-like rows:

```python
{
    "_id": {
        "platform": str,
        "platform_channel_id": str,
        "channel_type": str,
    },
    "character_message_count": int,
    "first_character_message_timestamp": str,
    "last_character_message_timestamp": str,
}
```

The selector fetches the actual evaluation-window messages for each monitored
channel and derives assistant, user, total, first, and last counters from the
fetched message rows.

### Message Rows

`list_reflection_scope_messages(...)` must use an allowlist projection. The
approved fields are:

```python
{
    "_id": 0,
    "platform": 1,
    "platform_channel_id": 1,
    "channel_type": 1,
    "role": 1,
    "platform_user_id": 1,
    "global_user_id": 1,
    "display_name": 1,
    "body_text": 1,
    "timestamp": 1,
    "attachments.description": 1,
}
```

The DB interface must not load embeddings, raw wire text, attachment binary
payloads, or arbitrary attachment metadata for reflection. If a future
reflection prompt needs another stored field, this ICD and the allowlist test
must be updated in the same change.

## Prompt-Facing Projection

`reflection_cycle.projection` owns the LLM-facing payload. Local LLMs receive
semantic labels and bounded text, not raw database telemetry.

The prompt payload keeps machine-facing JSON keys in English, matching the
existing cognition and consolidator style. Chinese belongs in instructions,
review questions, descriptive label values, and generated free-text outputs.

Hourly payload shape:

```python
{
    "evaluation_mode": "readonly_hourly_reflection",
    "scope_metadata": {
        "scope_ref": "scope_x",
        "platform": "qq",
        "channel_type": "private|group|system|unknown",
        "activity_labels": {
            "message_volume": "短对话",
            "assistant_presence": "角色参与度均衡",
            "participant_diversity": "单用户互动",
            "window_span": "多轮时间窗口",
        },
    },
    "conversation": {
        "message_order": "chronological",
        "messages": [
            {
                "role": "user|assistant",
                "speaker_ref": "participant_1|active_character",
                "time_position": "开场|前段|中段|后段|收尾|单条",
                "text": "bounded body text",
                "attachment_context": ["bounded attachment description"],
            }
        ],
    },
    "review_questions": ["中文评估问题"],
}
```

Daily payload shape:

```python
{
    "evaluation_mode": "readonly_daily_synthesis",
    "window": {
        "requested_start": "ISO timestamp",
        "requested_end": "ISO timestamp",
        "fallback_used": bool,
        "fallback_reason": str,
    },
    "channel": {
        "channel_type": "private|group|system|unknown",
    },
    "active_hour_slots": [
        {
            "hour": "UTC hour-start ISO timestamp",
            "topic_summary": "bounded hourly topic summary",
            "conversation_quality_feedback": ["bounded feedback"],
            "conversation_quality_feedback_omitted_count": 0,
            "privacy_notes": ["bounded privacy notes"],
            "privacy_notes_omitted_count": 0,
            "validation_warnings": ["bounded validation warning"],
            "validation_warnings_omitted_count": 0,
            "confidence": "low|medium|high",
        }
    ],
    "review_questions": ["中文评估问题"],
}
```

`active_hour_slots` contains only message-bearing hours that produced hourly
reflection data. Once a channel is monitored, every hour in the evaluation
window is eligible; the runtime skips an hour only when no user or assistant
message exists in that hour. User-only and assistant-only hours are still
included. Missing hours mean no message-bearing hourly reflection data is
available for that hour. The daily LLM must not infer content from missing
hours.

Daily slots keep only the lead compact item for list-like hourly fields. When
more same-category items exist, the corresponding `*_omitted_count` field makes
that budget loss visible to reviewers and to the daily synthesis prompt.

## Attachment Policy

Reflection may use attachment descriptions only.

The DB read allowlist includes `attachments.description`. The prompt projection
then applies additional bounds:

- at most three attachment descriptions per message,
- at most 160 characters per attachment description,
- no `base64_data`,
- no URLs unless they are already part of the description text,
- no attachment object is emitted when no description exists.

Binary attachment payloads and raw wire data are outside the reflection cycle
contract.

## LLM Contracts

The prompt contracts live in `prompts.py`. Each LLM-backed stage must keep:

```text
prompt constant
LLM instance
handler function
```

This mirrors the existing cognition and consolidator layout.

### Hourly Reflection Output

The hourly LLM must return only:

```python
{
    "topic_summary": str,
    "participant_observations": [
        {
            "participant_ref": "participant_1",
            "observation": str,
            "evidence_strength": "low|medium|high",
        }
    ],
    "conversation_quality_feedback": list[str],
    "privacy_notes": list[str],
    "confidence": "low|medium|high",
}
```

Forward-looking fields such as `lore_candidates`, `progress_projection`, and
`open_loops` are intentionally rejected in the read-only evaluation schema.

### Daily Synthesis Output

The daily LLM must return only:

```python
{
    "day_summary": str,
    "active_hour_summaries": [
        {
            "hour": "UTC hour-start ISO timestamp",
            "summary": str,
        }
    ],
    "cross_hour_topics": list[str],
    "conversation_quality_patterns": list[str],
    "privacy_risks": list[str],
    "synthesis_limitations": list[str],
    "confidence": "low|medium|high",
}
```

Daily synthesis receives compact active-hour slots and channel metadata. It
must not receive raw transcripts or full hourly reflection objects.
`active_hour_summaries.hour` must copy an input `active_hour_slots.hour` value
exactly; the model must not convert time zones or rewrite the timestamp shape.

## Language Policy

Reflection prompts follow the existing cognition and consolidator convention:

- JSON schema keys remain stable machine-facing keys.
- Required enum values remain stable values.
- `participant_ref`, `scope_ref`, IDs, URLs, code, commands, and model labels
  remain unchanged.
- Newly generated free-text fields must be Simplified Chinese.
- Source text, quotations, names, titles, aliases, and external evidence remain
  in the original language when precision matters.
- Do not duplicate content bilingually unless the source already does.

Do not repeat "write in Simplified Chinese" inside every schema sample value.
The language policy block owns that instruction.

## Artifact Contract

`runtime.py` writes one local JSON artifact for review. The artifact may include:

- prompt version and current git SHA,
- selection windows,
- fallback status,
- selected-channel summaries,
- query diagnostics,
- active-hour summaries,
- prompt previews,
- prompt validation warnings,
- raw LLM outputs when `use_real_llm=True`,
- parsed LLM outputs,
- manual review notes.

The artifact must not be treated as persistent memory, lore, scheduler state,
or conversation state.

## Dependency Rules

Allowed imports from `reflection_cycle`:

- dataclasses from `reflection_cycle.models`,
- public entry points from `reflection_cycle.__init__`,
- internal modules within `reflection_cycle`,
- read-only functions from `db.conversation_reflection`,
- shared config and LLM utility helpers.

Forbidden imports and calls:

- `reflection_cycle` importing `service`, `/chat` handlers, adapters, cognition
  nodes, dialog nodes, consolidator write paths, scheduler dispatch, or memory
  persistence modules.
- `selector.py` importing `get_db` or executing MongoDB commands directly.
- DB modules importing `reflection_cycle`.
- prompt projection importing DB clients.
- runtime writing MongoDB collections.
- reflection writing `user_image`, `user_memory_units`, user profiles,
  consolidator outputs, character memory, lore, scheduler state, or dialog
  state.
- hourly or daily reflection dispatching user-visible messages.

If a future integration needs memory writes or autonomous messages, it must add
a separate interface layer. It must not turn the read-only runtime into a
mixed read/write orchestration module.

## Integration Points

### Current Local Evaluation

The only current side effect is a local artifact file. This is the approved
path for evaluating prompt quality and performance on recent data.

### Future Scheduler Integration

A scheduler may call `run_readonly_reflection_evaluation(...)` or a future
write-capable facade. Scheduler code must not call internal selector,
projection, or prompt helpers directly.

The read-only selector's bounded fallback is for local evaluation only. A
future write-capable production worker must idle when no channel has a latest
assistant message inside the monitor window.

### Future Memory Evolution

Memory evolution may consume approved reflection outputs after human or
automated quality gates. It must define its own persistence contract, including
privacy stripping, provenance, merge history, and lore conflict behavior.

The read-only reflection output is evidence for review, not authorization to
write memory.

User image and user memory remain consolidator-owned. Reflection may emit
participant observations for character-training evaluation only; those
observations are not a contract to create, update, or override user image.

Source evidence references for future memory evolution must be threaded from
input/repository metadata beside the prompt projection. They must not be
reconstructed from LLM output because prompt-facing projection intentionally
removes or abstracts identifying fields. If a future write-capable path needs
conversation-history join keys such as message `_id`, the field must be added
to a persistence-only allowlist and kept out of the LLM prompt payload.

### Future Autonomous Message Injection

Autonomous messages must integrate behind the cognition boundary through a
dedicated queue or signal interface. Reflection must not call `/chat` directly
and must not invoke dialog generation as a side effect of hourly or daily
evaluation.

## Verification Checklist

Before merging reflection-cycle changes, verify:

- Selector tests prove MongoDB access goes through `db.conversation_reflection`.
- DB tests prove message reads use the allowlist projection.
- Prompt contract tests prove prompt examples match emitted payload keys.
- Prompt contract tests prove free-text language policy is centralized.
- Daily prompt tests prove raw transcripts are not included.
- Runtime tests prove prompt-only execution writes only local artifacts.
- Live LLM tests, when run, are executed one at a time and inspected
  qualitatively.

## Change Control

Changes to this ICD are required when:

- a new field enters the reflection DB allowlist,
- prompt payload keys change,
- output schemas change,
- a new module consumes reflection outputs,
- reflection gains any write capability,
- scheduler or autonomous-message integration is added.

Changing the read-only boundary is a breaking architectural decision. It must
be reviewed as an integration change, not a prompt tweak.
