# RAG Supervisor 2 — Inner Loop Agents Development Plan

## Context

`persona_supervisor2_rag_supervisor2.py` is an experimental progressive RAG supervisor that resolves
complex multi-hop queries (e.g. "他上次说的那个链接里有什么信息么") through an
Initializer → Dispatcher → Executor → Evaluator loop driven by `unknown_slots`.

The current dispatcher calls raw tools directly in one shot. For a local LLM this produces weak
arguments. The fix is to replace each fuzzy tool call with a dedicated inner-loop agent that owns
argument quality for exactly one tool.

**Separation of concerns:**

- Outer loop (supervisor2): slot routing + cross-tool fallback decisions
- Inner loops (these agents): argument refinement for one specific tool

---

## Agents to Build

| File                                        | Wraps                              |
| ------------------------------------------- | ---------------------------------- |
| `rag/conversation_search_agent.py`       | `search_conversation`              |
| `rag/conversation_keyword_agent.py`      | `search_conversation_keyword`      |
| `rag/conversation_filter_agent.py`       | `get_conversation`                 |
| `rag/persistent_memory_search_agent.py`  | `search_persistent_memory`         |
| `rag/persistent_memory_keyword_agent.py` | `search_persistent_memory_keyword` |

`web_search_agent2.py` already exists and is not duplicated.

---

## Shared Structure

Every agent is a plain async function (no LangGraph graph):

```python
async def <agent_name>(
    task: str,         # slot description from unknown_slots
    context: dict,     # known_facts from outer loop for parameter hints
    max_attempts: int = 3,
) -> dict:             # {"resolved": bool, "result": str, "attempts": int}

    feedback = ""
    for attempt in range(max_attempts):
        args     = await _generator(task, context, feedback)
        result   = await _tool(args)
        resolved, feedback = await _judge(task, result)
        if resolved:
            break

    return {"resolved": resolved, "result": result, "attempts": attempt + 1}
```

Three internal async functions per agent:

- `_generator(task, context, feedback) -> dict` — LLM bound to one tool, produces args
- `_tool(args)` — direct tool call, no logic
- `_judge(task, result) -> tuple[bool, str]` — plain LLM, returns (resolved, feedback)

The feedback string is the only state passed between iterations.
It must be specific and actionable (e.g. "keyword too long, use only the core noun").

---

## Agent-Specific Notes

### 1. `conversation_search_agent.py`

- Tool: `search_conversation` (semantic vector search)
- Generator prompt: guide toward natural-language semantic queries, not keyword lists
- Judge feedback signals: query too vague / wrong topic angle / returned irrelevant messages

### 2. `conversation_keyword_agent.py`

- Tool: `search_conversation_keyword` (regex/keyword match)
- Generator prompt: extract the shortest unambiguous core noun or phrase
- Judge feedback signals: keyword too long / no match / try a synonym or shorter form

### 3. `conversation_filter_agent.py`

- Tool: `get_conversation` (structured filter — platform, channel, user, time range, limit)
- Generator prompt: derive filter values from `known_facts` (e.g. resolved user name, timestamp from prior slot); widen time range or increase limit if results are insufficient
- Judge feedback signals: too few results / time range too narrow / wrong user filter / increase limit

### 4. `persistent_memory_search_agent.py`

- Tool: `search_persistent_memory` (semantic vector search over stored memories)
- Generator prompt: guide toward impression/fact framing ("千纱对X的看法" not just "X")
- Judge feedback signals: query too abstract / wrong memory_type filter / no relevant memories

### 4. `persistent_memory_keyword_agent.py`

- Tool: `search_persistent_memory_keyword` (regex/keyword match over stored memories)
- Generator prompt: same as conversation keyword — shortest unambiguous term
- Judge feedback signals: keyword too specific / try a shorter or more general term

---

## Build Order

Build and test one agent before starting the next.

1. `conversation_search_agent` — baseline; establishes the pattern
2. `conversation_keyword_agent` — same shape, different prompt
3. `conversation_filter_agent` — structured filters, different generator/judge logic
4. `persistent_memory_search_agent` — copy of 1, different tool and prompt
5. `persistent_memory_keyword_agent` — copy of 2, different tool and prompt

Each agent must have a `test_main()` that can be run standalone before moving on.

---

## Integration into supervisor2

After all 4 agents are built, update `persona_supervisor2_rag_supervisor2.py`:

1. Replace `_RAG_SUPERVISOR_TOOLS` (raw tool list) with the 5 agents + `web_search_agent2`
2. Update `_DISPATCHER_PROMPT` — dispatcher now picks which agent/tool to delegate to, not which tool args to generate
3. Update `rag_executor` — call the selected agent function instead of raw tool invocation
4. `rag_evaluator` — reads `resolved` and `result` from agent return dict into `known_facts`

---

## Scope

These agents are only wired into `persona_supervisor2_rag_supervisor2.py` at this stage.
No connection to the main graph or existing RAG pipeline.
