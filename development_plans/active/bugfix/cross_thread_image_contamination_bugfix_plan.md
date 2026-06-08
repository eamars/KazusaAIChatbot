# cross thread image contamination bugfix plan

## Summary

- Goal: document the observed cross-thread image contamination where Kazusa
  answered Jigsaw's correction as if another user's adjacent image belonged to
  the Jigsaw thread.
- Plan class: small
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: not selected; this draft is observation-only and
  does not authorize production-code, prompt, schema, or test changes.
- Highest-risk areas: blaming queue dropping alone when the visible reply is a
  semantic thread-binding failure, over-filtering group ambient context that
  Kazusa legitimately needs, and treating image context as globally current
  instead of source-bound to the speaker/thread that supplied it.
- Acceptance criteria: this draft is complete when the failure mode,
  production evidence, observed boundaries, and excluded implementation scope
  are recorded for later investigation.

## Context

This draft records the adjacent failure after Jigsaw corrected Kazusa's
self/other inversion.

At `2026-06-07T14:00:14.337025+00:00`, Jigsaw replied to Kazusa's delivered
message and corrected the target:

```text
@杏山千纱 什么你你我我的，看看我是啥样的
```

At `2026-06-07T14:00:41.832424+00:00`, another user, `Nyan`, sent an
unaddressed image-only message. The stored row has an empty `body_text`, an
image attachment, and no addressee.

At `2026-06-07T14:00:51.551788+00:00`, Kazusa produced a visible reply
addressed to Jigsaw:

```text
诶？
突然发这种照片干嘛啊……
什么啊，明明是想看我的反应吧？
感觉某人好像在故意试探什么的……
唔，不知道算不算是在调情呢
Jigsaw 到底是什么意思呀~
既然都主动发过来了，那我也随便看看啦
不过这种突然的关注真的会让人有点不知所措诶
所以说……对方到底是认真的还是在开玩笑？
```

The response mixes Jigsaw's correction thread with a different user's adjacent
image event. The message is visibly addressed to Jigsaw but speaks as though
Jigsaw sent or caused the image.

## Evidence

Primary evidence artifacts:

- `test_artifacts/diagnostics/kazusa_jigsaw_pm5_window.json`
- `test_artifacts/diagnostics/kazusa_jigsaw_event_log_pm5.json`
- `test_artifacts/diagnostics/jigsaw_conversation_progress_state.json`
- `test_artifacts/diagnostics/incident_internal_monologue_residue_by_created_at.json`
- `test_artifacts/diagnostics/kazusa_jigsaw_incident_review.md`

Observed stored conversation sequence:

| UTC time | Speaker | Message id | Stored summary |
|---|---|---|---|
| `14:00:14.337025` | Jigsaw `411706805` | `1153013760` | Bot-addressed correction asking Kazusa to describe Jigsaw. |
| `14:00:17.086100` | `•᷄ࡇ•᷅` `1611902585` | `1033213276` | Bot-addressed reply to the same Kazusa message: `阿巴阿巴`. |
| `14:00:22.809007` | Ab `2485343605` | `1148120616` | Unaddressed text: `何意味`. |
| `14:00:41.832424` | Nyan `736608397` | `649753219` | Unaddressed image-only row. |
| `14:00:51.551788` | Kazusa | `973230530` | Visible reply addressed to Jigsaw that discusses a sudden photo and flirtation/testing. |

Observed event-log state:

- The event-log window contains 116 `queue_intake` warning rows with
  `status="dropped"`.
- Nyan's image message `649753219` has a matching queue event:
  `queue_intake:dropped`, `protected_by_reply=False`, `listen_only=False`.
- The visible Kazusa reply at `14:00:51` has `dialog_quality` status passed and
  no failure codes.
- The event-log family summary has no `rag_stage` event for this failure
  sequence.

Observed conversation-progress state:

- Jigsaw's progress state after the correction records
  `conversation_mode="带有防御性的暧昧调侃"`.
- It records the current thread as
  `通过性格类型话题进行心理博弈与暧昧拉扯`.
- It records an open loop waiting for whether the user is flirting or joking.

Observed residue state:

- Residue row `701381ac4a7945ee9d666fb8f6d4f243`, sourced from
  `user_message:qq:905393941:1153013760`, records:

```text
他刚才的调侃让我心里那点悸动和窃喜还没散呢，我得小心应对，不能让他觉得我太容易被撩拨了，但也不能显得太冷淡。
```

The correction was therefore persisted as teasing/flirtation residue rather
than repair of a misunderstood target.

## Observed Failure Mode

The system produced a visible response whose target user and evidence source
do not match:

```text
Addressed target: Jigsaw.
Response content source: adjacent unaddressed image from Nyan plus inferred
flirtation/testing context.
```

The failure is not merely that Kazusa noticed the group image. A group-chat
character may notice ambient events. The failure is that she treated that
ambient image as part of the active Jigsaw reply thread and addressed the
result to Jigsaw.

## Current Boundary Read

The current evidence points away from these ownership boundaries:

- Not primarily QQ adapter envelope construction: the Nyan image row is
  stored separately, unaddressed, and with no Jigsaw identity.
- Not primarily reply hydration: Jigsaw's correction correctly replies to
  Kazusa's delivered row, and the adjacent image has no reply context.
- Not primarily delivery receipts: the referenced Kazusa row had delivered
  platform metadata.

The current evidence points toward these ownership boundaries:

- Queue/intake and response-survivor selection under noisy group-chat
  conditions, because unaddressed image rows were dropped but still appear
  temporally adjacent to the surviving persona work.
- Prompt-facing recent-history or media-observation projection, if dropped or
  adjacent image context can still enter the active turn without source/thread
  boundaries.
- Conversation-progress and residue interpretation, because Jigsaw's repair
  was reframed as flirtation before the image-contaminated visible response.
- Dialog/content-anchor validation, because the final visible reply passed
  despite source-target mismatch.

## Observation-Only Scope

This draft only records the failure mode and observations.

It does not authorize:

- production-code edits;
- prompt edits;
- test edits;
- queue policy changes;
- relevance changes;
- media descriptor changes;
- RAG routing changes;
- recent-history projection changes;
- conversation-progress changes;
- residue-recorder changes;
- dialog evaluator changes;
- adapter changes;
- database migration or data repair.

## Required Next Discovery Before Executable Plan

The next stage must inspect the exact historical prompt-facing state or
reproduce a controlled equivalent before proposing a fix:

- whether the Nyan image description entered Jigsaw's active prompt-facing
  history, media observations, L1 input, L2 input, or L3 content anchors;
- whether the queue survivor at `14:00:51` was the Jigsaw correction, another
  protected bot-addressed row, or a coalesced/collapsed survivor;
- whether dropped queue rows remain available in prompt-facing recent history
  for the current surviving turn;
- whether source-bound media observations preserve the speaker and addressee
  identity when projected to cognition/dialog;
- whether conversation-progress or residue caused the correction to be treated
  as a flirtation frame before the image entered the answer;
- whether the dialog evaluator can observe enough source-target structure to
  reject a reply that attributes another user's image to the addressed user.

The executable plan must be written only after this discovery can identify the
smallest owning stage and the smallest safe change surface.

