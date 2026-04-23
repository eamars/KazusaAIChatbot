---
name: no-prepost-user-input
description: Preserve an LLM-first architecture for user-input interpretation. Use this skill whenever you are designing, reviewing, or modifying code that decides whether user instructions, preferences, permissions, accepted commands, or commitments should be persisted or acted on. Trigger especially when you see deterministic pre-processing or post-processing over user input, acceptance gating added after an LLM call, local keyword classification of user-directed rules, or code that rewrites LLM decisions into a different channel. Prefer prompting, schema design, and clearer LLM output contracts over hardcoded filters.
---

# No Pre/Post Processing User Input

This skill protects a specific architectural rule in this workspace:

- User-input interpretation should be decided by the LLM.
- Do not add deterministic pre-processing or post-processing logic that overrides, filters, reclassifies, or "corrects" the model's decision about accepted user commands, preferences, permissions, or commitments.

## When to use this skill

Use this skill when working on any code path that:

- reads `decontexualized_input`
- transforms LLM outputs like `accepted_user_preferences`, `new_facts`, or `future_promises`
- decides whether a user request was accepted
- converts accepted commands into commitments or operational state
- introduces keyword-based routing for user-imposed speaking rules, address rules, or formatting rules

This skill is especially relevant in:

- cognition preference handling
- consolidator facts/promise harvesting
- commitment persistence
- evaluator prompts that decide which channel a result belongs to

## Core rule

Do not do any of the following to user-input interpretation after the LLM has spoken:

- drop an LLM-emitted fact or promise because a local allowlist/disallowlist says so
- infer acceptance from `logical_stance`/`character_intent` with a deterministic gate
- infer `commitment_type` from string matching on `action`
- keyword-match a user request and force it into `new_facts` or out of `future_promises`
- rewrite an accepted command into an implicit personality/preference fact by local code

If the LLM output is wrong, fix the prompt, schema, or examples rather than adding local corrective logic.

## Preferred design pattern

When a user command might become durable state:

1. Make the LLM output contract explicit.
2. Ask the LLM to choose the correct channel.
3. Preserve that output faithfully in application code.
4. Only do structural sanitation that is unrelated to semantics.

Good examples of allowed structural sanitation:

- checking that `new_facts` is a list before iterating
- checking that each promise row is a dict before storing it
- generating UUIDs or timestamps for persistence
- preserving an LLM-supplied enum/string field as-is

Bad examples:

- `if not allow_acceptance: result["future_promises"] = []`
- `if "主人" in action: commitment_type = "address_preference"`
- `if fact looks like style rule: drop it`

## Channel selection guidance

When the user gives an instruction like reply style, address style, suffixes, or language constraints:

- If the model decides it was accepted as an ongoing operational rule, it should usually be emitted as a commitment-like structure such as `future_promises` / active commitments.
- It should not be rewritten by local code into an implicit stable fact about what the character "likes" or "is like".
- If the model decides it was not truly accepted, keep it out of durable state by improving the prompt or evaluator, not by adding deterministic gates.

## Review checklist

When reviewing a change, scan for:

- helper names like `_filter_*`, `_infer_*`, `_normalize_*`, `_classify_*`
- keyword matching over user-facing strings
- deterministic gates using stance/intent to override LLM outputs
- code that changes persistence channels after the LLM response

If found:

- remove the deterministic semantic logic
- move the decision into the prompt / JSON schema / evaluator instructions
- add tests that assert faithful passthrough of the LLM's chosen channel

## Expected output when applying this skill

When you fix a violation, aim for this shape:

- prompts explain the distinction between facts vs commitments vs accepted rules
- schemas include the fields the LLM needs to emit directly
- post-LLM code persists the chosen structure without semantic reinterpretation
- tests verify passthrough, not deterministic override
