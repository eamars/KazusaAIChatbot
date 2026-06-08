# self other inversion personality question bugfix plan

## Summary

- Goal: document the observed self/other inversion when Jigsaw asked Kazusa to
  describe him and identify the current evidence needed before an executable
  bugfix plan is written.
- Plan class: small
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: not selected; this draft is observation-only and
  does not authorize production-code, prompt, schema, or test changes.
- Highest-risk areas: treating an adapter/identity problem as the root cause
  when stored envelope data is correct, overfitting a prompt rule to one
  Chinese sentence, and weakening character judgment by suppressing natural
  reciprocal questions instead of preserving referent ownership.
- Acceptance criteria: this draft is complete when the failure mode,
  production evidence, observed boundaries, and excluded implementation scope
  are recorded for later investigation.

## Context

This draft records one confirmed failure from the QQ group conversation around
the assistant message:

```text
唔……这个啊
大概是那种……会想很多、稍微有点难懂的类型吧？
诶，怎么突然问这个呀
Jigsaw 呢？你是什么样的人嘛
```

The anchor row is in `conversation_history` for QQ group `905393941` at
`2026-06-07T13:58:12.556184+00:00`. The visible user message immediately
before it was from QQ user `411706805`, display name `Jigsaw`:

```text
@杏山千纱 你觉得我是啥样的人，mtbi接近哪个
```

The user's question asks Kazusa to describe the user and estimate a nearby
MBTI type. Kazusa's answer starts with vague traits, then asks `Jigsaw 呢？你是
什么样的人嘛`, which flips the target back to the user as if the requested
description had not been supplied. Another group participant immediately
called out the problem:

```text
妈耶，又没分清你我
```

This is a character-brain quality failure. The defect is not that Kazusa spoke;
Jigsaw directly addressed her, and she had a grounded reason to answer. The
defect is that the answer did not preserve the user/character referent relation
for a direct self-description request.

## Evidence

Primary evidence artifacts:

- `test_artifacts/diagnostics/kazusa_jigsaw_pm5_window.json`
- `test_artifacts/diagnostics/kazusa_jigsaw_event_log_pm5.json`
- `test_artifacts/diagnostics/kazusa_jigsaw_anchor_matches.json`
- `test_artifacts/diagnostics/jigsaw_user_profile.json`
- `test_artifacts/diagnostics/jigsaw_user_memory_context.json`
- `test_artifacts/diagnostics/jigsaw_user_memories_raw.json`
- `test_artifacts/diagnostics/incident_internal_monologue_residue_by_created_at.json`
- `test_artifacts/diagnostics/kazusa_jigsaw_incident_review.md`

Observed stored envelope state:

- The inbound Jigsaw row has `platform="qq"`, `platform_channel_id="905393941"`,
  `role="user"`, `platform_user_id="411706805"`, and
  `global_user_id="fa874545-02e6-4127-a24e-30819f941d83"`.
- The inbound `body_text` is clean semantic text:
  `@杏山千纱 你觉得我是啥样的人，mtbi接近哪个`.
- The inbound mention resolves the bot user `3768713357` to the active
  character global id `00000000-0000-4000-8000-000000000001`.
- The inbound `addressed_to_global_user_ids` contains only the active character
  global id.
- The assistant row is addressed back to Jigsaw's global user id.
- Delivery receipt metadata exists for the assistant row:
  `delivery_status="delivered"`, `delivery_adapter="napcat"`,
  platform message id `238569441`.

Observed memory/profile state:

- Jigsaw has a user profile row with QQ account `411706805`, affinity `494`,
  and relationship insight `处于一种带有防御性的窃喜中`.
- Jigsaw has six active user-memory-unit rows. They contain relationship and
  behavior continuity, but no literal MBTI label.
- The event-log window contains no `rag_stage` events for the incident window,
  even though RAG stage telemetry exists in the codebase.

Observed residue state:

- Residue row `d0eb691cec114c97b6f32b2984dd3fcf`, sourced from
  `user_message:qq:905393941:1605224422`, records:

```text
他居然在认真思考我的类型？这种被重视的感觉太让人害羞了，心里那股悸动和窃喜还没散呢。
```

This residue text interprets Jigsaw's request as if Jigsaw were thinking about
Kazusa's type, even though the visible user input asks Kazusa to describe
Jigsaw.

## Observed Failure Mode

The system preserved platform identity and addressing correctly, but the
semantic interpretation path inverted the described subject:

```text
User asks: What kind of person am I?
Kazusa behaves as if: You are thinking about what kind of person I am.
```

The visible answer also fails to satisfy the requested MBTI-adjacent answer. It
gives a vague impression and then asks the user to self-describe, which is not
an adequate answer to the direct prompt.

## Current Boundary Read

The current evidence points away from these ownership boundaries:

- Not primarily NapCat normalization: `body_text`, mention metadata, and
  addressee fields are correct.
- Not primarily delivery receipt or reply hydration: the anchor assistant row
  has delivered platform metadata.
- Not primarily MongoDB identity resolution: Jigsaw's QQ id maps to a stable
  global user id and the assistant row is addressed back to that id.

The current evidence points toward these ownership boundaries:

- Current-turn semantic interpretation in cognition and residue recording.
- Evidence selection for user self-description questions; no logged RAG stage
  appears despite available user profile and memory context.
- Dialog/content-anchor acceptance, because the final text did not preserve the
  original answer target.

## Observation-Only Scope

This draft only records the failure mode and observations.

It does not authorize:

- production-code edits;
- prompt edits;
- test edits;
- model route changes;
- RAG routing changes;
- conversation-progress changes;
- residue-recorder changes;
- dialog evaluator changes;
- database migration or data repair.

## Required Next Discovery Before Executable Plan

The next stage must inspect the exact historical prompt-facing state or
reproduce a controlled equivalent before proposing a fix:

- decontextualizer output for message `1605224422`;
- cognition L1/L2/L2d artifacts if available from traces or reproducible run;
- L3 content anchors and dialog evaluator outputs for the anchor reply;
- whether the resolver selected or skipped `rag_evidence`;
- whether `internal_monologue_residue_context` from earlier turns biased the
  L2a interpretation;
- whether current `user_memory_context` is normally present in L2a without a
  RAG call for this class of user-profile question.

The executable plan must be written only after this discovery can identify the
smallest owning stage and the smallest safe change surface.

