---
name: local-llm-architecture
description: Use this skill whenever the user asks architectural questions about LLM systems, especially prompt design, agent/subagent responsibility boundaries, routing, retrieval planning, tool/capability design, reliability, latency, or how to divide work between an initializer/planner and specialized agents. This skill is particularly important for this repo because the target runtime uses a local/weaker LLM with finite context and chatbot latency constraints; invoke it even when the user does not explicitly mention local LLMs.
---

# Local LLM Architecture

Use this skill when reasoning about system-level design for this chatbot or any LLM pipeline where the deployed model is local, weaker, latency-constrained, or operating with limited context.

The goal is to avoid designs that look elegant with a frontier model but become brittle, slow, or expensive with the actual runtime model.

## Core Assumptions

- The production model may be substantially weaker than the coding assistant.
- Context may be large but not effectively usable for schema discovery, multi-hop reasoning, or long instruction following.
- Chatbot responses need bounded latency. Multi-minute agent loops are unacceptable for normal user messages.
- The system should degrade gracefully when a query falls outside supported capabilities.
- The LLM should reason over semantic concepts, not raw database structure or implementation-specific schemas.

## Design Posture

Prefer bounded, inspectable architecture:

- Small set of explicit capabilities over many ad hoc tools.
- Typed or semi-typed intermediate representations over free-form prose when routing matters.
- Specialized agents own low-level parameter generation for their domain.
- Deterministic code owns validation, permissions, limits, schema mapping, and execution.
- The planner/initializer should decide what information is needed, not how to query every storage backend.

Avoid designs where:

- A single initializer must understand every database shape and generate all low-level parameters.
- A weak local LLM is expected to infer schema or tool semantics from long prompts.
- A retrieval agent can silently accept tasks outside its domain and produce plausible but wrong results.
- Retry loops are the default path for ordinary queries.
- More tools are added without clarifying semantic ownership.

## Responsibility Boundaries

Use this default separation:

```text
User query
  -> lightweight planner / initializer: identify required facts and dependencies
  -> router: select the semantic capability or specialist
  -> specialist agent: generate low-level query parameters for one domain
  -> deterministic validator/executor: enforce schema, cost, scope, and safety
  -> answer synthesizer: explain results and uncertainty
```

The planner should produce a logical target such as:

```json
{
  "intent_type": "user_list",
  "needs": "users whose display names end with 子",
  "constraints": [
    {"field": "display_name", "op": "ends_with", "value": "子"}
  ]
}
```

It should not need to know the physical storage layout, such as MongoDB field paths, aggregation syntax, index names, or dedup rules.

## Capability Design

When proposing tools or agents, define:

- **Semantic ownership:** what class of question this capability owns.
- **Input contract:** what structured arguments it accepts.
- **Output contract:** exact cardinality and shape, such as one user vs many users.
- **Refusal conditions:** tasks it should reject instead of attempting.
- **Latency budget:** normal number of LLM calls and database calls.
- **Fallback behavior:** whether to ask clarification, return partial results, or escalate to another capability.

Prefer capabilities like:

```text
user_lookup        -> resolve one named user
user_list          -> enumerate users by metadata/predicate
message_search     -> find messages by keyword or semantic query
message_list       -> list messages by structured filters
memory_search      -> retrieve profile/persistent memories
profile_read       -> read one resolved user's profile
web_search/fetch   -> external information only
```

Do not overload one capability with incompatible cardinality unless the contract explicitly includes a mode such as `lookup_one` vs `list`.

## Prompt Design Guidance

Prompts for local LLMs should be short, explicit, and contract-oriented.

Good prompts:

- Give a small roster of allowed capabilities.
- Include refusal rules for out-of-domain requests.
- Provide 3-6 high-value examples, not a huge gallery.
- Ask for strict JSON only when downstream code validates it.
- Keep schema names semantic and stable.

Risky prompts:

- Long prose policies that require remembering exceptions far from the output format.
- Mixed responsibilities, such as planning, routing, argument generation, and judging in one call.
- Free-form slot labels where a prefix can contradict the content.
- Instructions that require the model to infer database schema from examples.

## Latency Strategy

Design for the common path:

```text
1 LLM call to plan/route
0-N deterministic tool calls
optional 1 LLM call to summarize
```

Use repair loops only when:

- The query is high-confidence answerable.
- The failure is localized and cheap to repair.
- There is a hard iteration cap.

Prefer one-shot deterministic helpers for structured tasks such as user enumeration, exact filters, and profile reads.

## Failure Handling

Make unsupported or ambiguous cases explicit.

Examples:

- If "user" could mean profiled users or observed chat participants, either choose the repo's default source or ask a clarification.
- If a message-search agent receives a user-metadata predicate, it should return an incompatible-intent error rather than searching message content.
- If a capability returns multiple candidates where a downstream step needs exactly one, stop and ask for disambiguation or use a documented tie-breaker.

## Review Checklist

Before recommending an architecture or prompt change, check:

- Would this still work with a weaker local LLM?
- Is the LLM being asked to infer hidden database structure?
- Is the responsibility boundary between planner, router, specialist, and executor clear?
- Can each agent reject out-of-domain work?
- Are cardinality expectations explicit?
- Is the normal latency path acceptable for a chatbot?
- Is there a deterministic validation layer before execution?
- Does the design improve one semantic capability rather than adding accidental complexity?
